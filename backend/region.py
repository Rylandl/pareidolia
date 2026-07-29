from __future__ import annotations

import hashlib
import json
import math
import os
import threading
import time
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .acus import (
    _orientation_profile,
    _plane_basis,
    _refine_needle,
    _robust_common_normal,
)
from .acus_compute import hessian_line_fields
from .rectify import VolumeData


_REGION_LOCK = threading.Lock()
_MEMORY_CACHE: dict[str, dict[str, Any]] = {}


def _axis_tiles(size: int, halo: int, target_core: int) -> list[tuple[int, int]]:
    start, stop = halo, size - halo
    if stop <= start:
        raise ValueError("the cuboid is too small for the requested real-data halo")
    count = max(1, int(math.ceil((stop - start) / target_core)))
    edges = np.linspace(start, stop, count + 1).round().astype(np.int32)
    return [(int(edges[index]), int(edges[index + 1])) for index in range(count)]


def _grid_axis(size: int, half_context: int, stride: int) -> list[int]:
    low, high = half_context, size - half_context
    if high < low:
        return []
    values = list(range(low, high + 1, stride))
    if values and values[-1] != high:
        values.append(high)
    return values


def _deduplicate_needles(
    needles: list[dict[str, Any]], minimum_distance: float
) -> list[dict[str, Any]]:
    if not needles:
        return []
    cell_size = max(1.0, minimum_distance)
    minimum_distance2 = minimum_distance * minimum_distance
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted: list[dict[str, Any]] = []
    for needle in sorted(needles, key=lambda item: float(item["score"]), reverse=True):
        center = np.asarray(needle["center"], dtype=np.float32)
        bucket = tuple(int(math.floor(float(value) / cell_size)) for value in center)
        too_close = False
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = tuple(bucket[axis] + offset[axis] for axis in range(3))
            if any(
                float(np.sum((center - existing) ** 2)) < minimum_distance2
                for existing in buckets.get(neighbor, [])
            ):
                too_close = True
                break
        if too_close:
            continue
        accepted.append(needle)
        buckets.setdefault(bucket, []).append(center)
    return accepted


def _pattern_signature(density: np.ndarray) -> np.ndarray:
    if density.size == 0:
        return np.zeros((0, 4), dtype=np.float32)
    row_sum = np.sum(density, axis=1, keepdims=True)
    normalized = np.divide(
        density,
        np.maximum(row_sum, 1.0e-6),
        out=np.zeros_like(density),
        where=row_sum > 1.0e-6,
    )
    signature = np.abs(np.fft.rfft(normalized, axis=1))[:, 1:5].astype(np.float32)
    scale = float(np.max(signature)) if signature.size else 0.0
    return signature / max(scale, 1.0e-6)


def _signature_alignment(
    first: np.ndarray | None, second: np.ndarray | None
) -> float | None:
    if first is None or second is None or not first.size or not second.size:
        return None
    best = -1.0
    for lag in range(-3, 4):
        if lag < 0:
            left, right = first[-lag:], second[: len(second) + lag]
        elif lag > 0:
            left, right = first[: len(first) - lag], second[lag:]
        else:
            left, right = first, second
        if not left.size or not right.size:
            continue
        a, b = left.reshape(-1), right.reshape(-1)
        denominator = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denominator <= 1.0e-7:
            continue
        best = max(best, float(np.dot(a, b) / denominator))
    return float(np.clip(best, 0.0, 1.0)) if best >= 0.0 else None


def _cache_key(volume: VolumeData, settings: dict[str, Any]) -> str:
    identity = {
        "name": volume.name,
        "shape": volume.shape_xyz,
        "origin": volume.origin_xyz,
        "low": round(volume.low, 4),
        "high": round(volume.high, 4),
        "settings": settings,
        "version": 4,
    }
    encoded = json.dumps(identity, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()[:20]


def _cached_result(cache_key: str) -> dict[str, Any] | None:
    cached = _MEMORY_CACHE.get(cache_key)
    if cached is None:
        cache_path = Path("work/region-cache") / f"{cache_key}.json"
        if cache_path.is_file():
            try:
                cached = json.loads(cache_path.read_text())
                _MEMORY_CACHE[cache_key] = cached
            except (OSError, json.JSONDecodeError):
                cached = None
    if cached is None:
        return None
    return {
        **cached,
        "stats": {**cached["stats"], "cacheHit": True, "elapsedMs": 0.0},
    }


def _store_cache(cache_key: str, result: dict[str, Any]) -> None:
    _MEMORY_CACHE[cache_key] = result
    cache_path = Path("work/region-cache") / f"{cache_key}.json"
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(result, separators=(",", ":")))
    except OSError:
        pass


def fit_acus_region(volume: VolumeData, request: dict[str, Any]) -> dict[str, Any]:
    """Bake a cuboid-wide finite-needle catalog and multi-cell Acus summary field."""
    started = time.perf_counter()
    cube_size = int(np.clip(request.get("cubeSize", 64), 24, 128))
    scale = float(np.clip(request.get("scale", 1.25), 0.7, 3.0))
    candidate_spacing = int(np.clip(request.get("spacing", 4), 3, 10))
    max_needles = int(np.clip(request.get("maxNeedles", 160), 24, 320))
    needle_length = float(np.clip(request.get("needleLength", 16.0), 6.0, 48.0))
    requested_padding = int(
        np.clip(request.get("contextPadding", math.ceil(needle_length)), 0, 48)
    )
    gaussian_radius = max(1, int(math.ceil(2.5 * scale)))
    cross_section_radius = max(2.0, float(math.ceil(scale * 1.5)))
    minimum_padding = int(
        math.ceil(gaussian_radius + needle_length * 0.5 + cross_section_radius)
    )
    minimum_padding = int(math.ceil(minimum_padding / 4.0) * 4)
    halo = max(requested_padding, minimum_padding)
    grid_stride = int(np.clip(request.get("gridStride", 16), 8, 64))
    tile_core = int(np.clip(request.get("tileCore", 56), 32, 96))
    catalog_bin_size = int(np.clip(request.get("catalogBinSize", 32), 16, 64))
    max_needles_per_bin = int(
        np.clip(request.get("maxNeedlesPerBin", 32), 8, 96)
    )
    settings = {
        "cubeSize": cube_size,
        "scale": scale,
        "candidateSpacing": candidate_spacing,
        "maxNeedles": max_needles,
        "needleLength": needle_length,
        "requestedPadding": requested_padding,
        "effectivePadding": halo,
        "minimumPadding": minimum_padding,
        "gridStride": grid_stride,
        "tileCore": tile_core,
        "catalogBinSize": catalog_bin_size,
        "maxNeedlesPerBin": max_needles_per_bin,
    }
    cache_key = _cache_key(volume, settings)
    with _REGION_LOCK:
        if not bool(request.get("force", False)):
            cached = _cached_result(cache_key)
            if cached is not None:
                return cached

        x_size, y_size, z_size = volume.shape_xyz
        x_tiles = _axis_tiles(x_size, halo, tile_core)
        y_tiles = _axis_tiles(y_size, halo, tile_core)
        z_tiles = _axis_tiles(z_size, halo, tile_core)
        tile_specs = []
        for tile_id, ((z0, z1), (y0, y1), (x0, x1)) in enumerate(
            product(z_tiles, y_tiles, x_tiles)
        ):
            tile_specs.append(
                {
                    "id": tile_id,
                    "core": (x0, x1, y0, y1, z0, z1),
                    "padded": (
                        x0 - halo,
                        x1 + halo,
                        y0 - halo,
                        y1 + halo,
                        z0 - halo,
                        z1 + halo,
                    ),
                }
            )
        grouped: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for spec in tile_specs:
            px0, px1, py0, py1, pz0, pz1 = spec["padded"]
            grouped.setdefault((pz1 - pz0, py1 - py0, px1 - px0), []).append(spec)

        radius = max(3, int(math.ceil(scale * 2.5)))
        line_field_ms = 0.0
        batch_launches = 0
        compute_backend = "cpu"
        compute_device: str | None = None
        maximum_batch_voxels = int(os.environ.get("ACUS_GPU_BATCH_VOXELS", "8000000"))

        def load_tile_data(specs: list[dict[str, Any]]) -> list[np.ndarray]:
            cubes: list[np.ndarray] = []
            for spec in specs:
                px0, px1, py0, py1, pz0, pz1 = spec["padded"]
                raw = np.asarray(volume.array[pz0:pz1, py0:py1, px0:px1])
                cubes.append(volume.normalize(raw).astype(np.float32) / 255.0)
            return cubes

        calibration_scales: list[float] = []
        for shape, specs in grouped.items():
            voxels_per_tile = int(np.prod(shape))
            items_per_batch = max(1, maximum_batch_voxels // voxels_per_tile)
            for batch_start in range(0, len(specs), items_per_batch):
                batch_specs = specs[batch_start : batch_start + items_per_batch]
                _, compute_meta = hessian_line_fields(load_tile_data(batch_specs), scale)
                line_field_ms += float(compute_meta["elapsedMs"])
                batch_launches += int(compute_meta["batchLaunches"])
                compute_backend = str(compute_meta["backend"])
                compute_device = compute_meta.get("device")
                calibration_scales.extend(
                    float(value) for value in compute_meta["strengthScales"]
                )
        region_strength_scale = float(np.median(calibration_scales))

        candidate_records: list[dict[str, Any]] = []
        for shape, specs in grouped.items():
            voxels_per_tile = int(np.prod(shape))
            items_per_batch = max(1, maximum_batch_voxels // voxels_per_tile)
            for batch_start in range(0, len(specs), items_per_batch):
                batch_specs = specs[batch_start : batch_start + items_per_batch]
                fields, compute_meta = hessian_line_fields(
                    load_tile_data(batch_specs), scale, region_strength_scale
                )
                line_field_ms += float(compute_meta["elapsedMs"])
                batch_launches += int(compute_meta["batchLaunches"])
                compute_backend = str(compute_meta["backend"])
                compute_device = compute_meta.get("device")
                for spec, (score, _) in zip(batch_specs, fields):
                    x0, x1, y0, y1, z0, z1 = spec["core"]
                    px0, _, py0, _, pz0, _ = spec["padded"]
                    first_x = halo + int(math.ceil((x0 - halo) / candidate_spacing)) * candidate_spacing
                    first_y = halo + int(math.ceil((y0 - halo) / candidate_spacing)) * candidate_spacing
                    first_z = halo + int(math.ceil((z0 - halo) / candidate_spacing)) * candidate_spacing
                    for block_z in range(first_z, z1, candidate_spacing):
                        for block_y in range(first_y, y1, candidate_spacing):
                            for block_x in range(first_x, x1, candidate_spacing):
                                gx1 = min(block_x + candidate_spacing, x_size - halo)
                                gy1 = min(block_y + candidate_spacing, y_size - halo)
                                gz1 = min(block_z + candidate_spacing, z_size - halo)
                                block = score[
                                    block_z - pz0 : gz1 - pz0,
                                    block_y - py0 : gy1 - py0,
                                    block_x - px0 : gx1 - px0,
                                ]
                                if not block.size:
                                    continue
                                flat = int(np.argmax(block))
                                value = float(block.flat[flat])
                                if value < 0.015:
                                    continue
                                dz, dy, dx = np.unravel_index(flat, block.shape)
                                candidate_records.append(
                                    {
                                        "score": value,
                                        "tileId": spec["id"],
                                        "point": (
                                            block_x + int(dx),
                                            block_y + int(dy),
                                            block_z + int(dz),
                                        ),
                                        "bin": (
                                            (block_x - halo) // catalog_bin_size,
                                            (block_y - halo) // catalog_bin_size,
                                            (block_z - halo) // catalog_bin_size,
                                        ),
                                    }
                                )

        candidate_bins: dict[tuple[int, int, int], list[dict[str, Any]]] = {}
        for candidate in candidate_records:
            candidate_bins.setdefault(candidate["bin"], []).append(candidate)
        selected_candidates = [
            candidate
            for values in candidate_bins.values()
            for candidate in sorted(
                values, key=lambda item: float(item["score"]), reverse=True
            )[:max_needles_per_bin]
        ]
        selected_by_tile: dict[int, list[dict[str, Any]]] = {}
        for candidate in selected_candidates:
            selected_by_tile.setdefault(int(candidate["tileId"]), []).append(candidate)

        raw_needles: list[dict[str, Any]] = []
        for shape, specs in grouped.items():
            active_specs = [spec for spec in specs if spec["id"] in selected_by_tile]
            voxels_per_tile = int(np.prod(shape))
            items_per_batch = max(1, maximum_batch_voxels // voxels_per_tile)
            for batch_start in range(0, len(active_specs), items_per_batch):
                batch_specs = active_specs[batch_start : batch_start + items_per_batch]
                fields, compute_meta = hessian_line_fields(
                    load_tile_data(batch_specs), scale, region_strength_scale
                )
                line_field_ms += float(compute_meta["elapsedMs"])
                batch_launches += int(compute_meta["batchLaunches"])
                compute_backend = str(compute_meta["backend"])
                compute_device = compute_meta.get("device")
                for spec, (score, direction_field) in zip(batch_specs, fields):
                    px0, _, py0, _, pz0, _ = spec["padded"]
                    for selected in selected_by_tile[spec["id"]]:
                        gx, gy, gz = selected["point"]
                        candidate = (
                            float(score[gz - pz0, gy - py0, gx - px0]),
                            gz - pz0,
                            gy - py0,
                            gx - px0,
                        )
                        needle = _refine_needle(
                            score,
                            direction_field,
                            candidate,
                            radius,
                            needle_length,
                            cross_section_radius,
                        )
                        if needle is None:
                            continue
                        center = np.asarray(needle["center"], dtype=np.float32)
                        global_center = center + np.asarray(
                            [px0, py0, pz0], dtype=np.float32
                        )
                        if not (
                            halo <= global_center[0] < x_size - halo
                            and halo <= global_center[1] < y_size - halo
                            and halo <= global_center[2] < z_size - halo
                        ):
                            continue
                        raw_needles.append(
                            {
                                "center": global_center,
                                "direction": np.asarray(
                                    needle["direction"], dtype=np.float32
                                ),
                                "score": float(needle["score"]),
                                "axialCoverage": float(needle["axialCoverage"]),
                                "supportScore": float(needle["supportScore"]),
                            }
                        )

        catalog = _deduplicate_needles(
            raw_needles, float(max(2, candidate_spacing - 1))
        )
        if len(catalog) < 6:
            raise ValueError("not enough supported ridge needles were found in the cuboid")
        centers = np.stack([item["center"] for item in catalog]).astype(np.float32)
        directions = np.stack([item["direction"] for item in catalog]).astype(np.float32)
        scores = np.asarray([item["score"] for item in catalog], dtype=np.float32)
        axial_coverage = np.asarray(
            [item["axialCoverage"] for item in catalog], dtype=np.float32
        )

        half_context = cube_size // 2 + halo
        grid_x = _grid_axis(x_size, half_context, grid_stride)
        grid_y = _grid_axis(y_size, half_context, grid_stride)
        grid_z = _grid_axis(z_size, half_context, grid_stride)
        if not grid_x or not grid_y or not grid_z:
            raise ValueError("the cuboid is too small for the requested N and halo")

        cells: list[dict[str, Any]] = []
        signatures: list[np.ndarray | None] = []
        cell_lookup: dict[tuple[int, int, int], int] = {}
        half_cube = cube_size * 0.5
        for iz, center_z in enumerate(grid_z):
            for iy, center_y in enumerate(grid_y):
                for ix, center_x in enumerate(grid_x):
                    center = np.asarray([center_x, center_y, center_z], dtype=np.float32)
                    inside = np.all(np.abs(centers - center) <= half_cube, axis=1)
                    indices = np.flatnonzero(inside)
                    if len(indices) > max_needles:
                        selected = np.argpartition(scores[indices], -max_needles)[-max_needles:]
                        indices = indices[selected]
                    cell_lookup[(ix, iy, iz)] = len(cells)
                    cell_base = {
                        "index": [ix, iy, iz],
                        "center": [center_x, center_y, center_z],
                    }
                    if len(indices) < 6:
                        cells.append({**cell_base, "valid": False, "needleCount": int(len(indices))})
                        signatures.append(None)
                        continue

                    local_directions = directions[indices]
                    local_weights = scores[indices]
                    normal, eigenvalues, robust_weights = _robust_common_normal(
                        local_directions, local_weights
                    )
                    dominant_axis = int(np.argmax(np.abs(normal)))
                    if normal[dominant_axis] < 0.0:
                        normal = -normal
                    residual_degrees = np.degrees(
                        np.arcsin(
                            np.clip(np.abs(local_directions @ normal), 0.0, 1.0)
                        )
                    )
                    median_residual = float(np.median(residual_degrees))
                    inlier_limit = max(8.0, min(22.0, median_residual * 2.8))
                    inliers = residual_degrees <= inlier_limit
                    total_eigenvalue = max(float(eigenvalues.sum()), 1.0e-7)
                    normal_confidence = float(
                        np.clip(
                            (
                                (eigenvalues[1] - eigenvalues[0])
                                / max(eigenvalues[2], 1.0e-7)
                            )
                            * float(np.mean(inliers)),
                            0.0,
                            1.0,
                        )
                    )
                    coplanarity = float(
                        np.clip(1.0 - eigenvalues[0] / total_eigenvalue, 0.0, 1.0)
                    )
                    u_axis, v_axis = _plane_basis(normal)
                    local_centers = centers[indices]
                    normal_coordinates = (local_centers - center) @ normal
                    family_angles = np.degrees(
                        np.arctan2(local_directions @ v_axis, local_directions @ u_axis)
                    ) % 180.0
                    profile = _orientation_profile(
                        normal_coordinates,
                        family_angles,
                        local_weights * robust_weights,
                        inliers,
                        cube_size,
                    )
                    signature = _pattern_signature(
                        np.asarray(profile["density"], dtype=np.float32)
                    )
                    signatures.append(signature)
                    cells.append(
                        {
                            **cell_base,
                            "valid": True,
                            "normal": np.round(normal, 6).tolist(),
                            "needleCount": int(len(indices)),
                            "inlierFraction": round(float(np.mean(inliers)), 4),
                            "normalConfidence": round(normal_confidence, 4),
                            "coplanarity": round(coplanarity, 4),
                            "medianPlaneResidualDeg": round(median_residual, 3),
                            "medianAxialCoverage": round(
                                float(np.median(axial_coverage[indices])), 4
                            ),
                            "twoModeCoverage": profile["stats"]["meanTwoModeCoverage"],
                            "coveredDepthFraction": profile["stats"]["coveredDepthFraction"],
                        }
                    )

        normal_edges: list[float] = []
        pattern_edges: list[float] = []
        for index, cell in enumerate(cells):
            if not cell["valid"]:
                continue
            ix, iy, iz = cell["index"]
            local_normal_edges: list[float] = []
            local_pattern_edges: list[float] = []
            for axis in range(3):
                for sign in (-1, 1):
                    neighbor_index = [ix, iy, iz]
                    neighbor_index[axis] += sign
                    lookup = cell_lookup.get(tuple(neighbor_index))
                    if lookup is None or not cells[lookup]["valid"]:
                        continue
                    normal = np.asarray(cell["normal"], dtype=np.float32)
                    neighbor_normal = np.asarray(cells[lookup]["normal"], dtype=np.float32)
                    angle = math.degrees(
                        math.acos(
                            float(
                                np.clip(abs(np.dot(normal, neighbor_normal)), 0.0, 1.0)
                            )
                        )
                    )
                    local_normal_edges.append(angle)
                    if sign > 0:
                        normal_edges.append(angle)
                    pattern = _signature_alignment(signatures[index], signatures[lookup])
                    if pattern is not None:
                        local_pattern_edges.append(pattern)
                        if sign > 0:
                            pattern_edges.append(pattern)
            cell["neighborCount"] = len(local_normal_edges)
            cell["neighborNormalMedianDeg"] = (
                round(float(np.median(local_normal_edges)), 3)
                if local_normal_edges
                else None
            )
            cell["neighborPatternMedian"] = (
                round(float(np.median(local_pattern_edges)), 4)
                if local_pattern_edges
                else None
            )

        valid_cells = [cell for cell in cells if cell["valid"]]
        result = {
            "shape": {"x": x_size, "y": y_size, "z": z_size},
            "globalOrigin": {
                "x": volume.origin_xyz[0],
                "y": volume.origin_xyz[1],
                "z": volume.origin_xyz[2],
            },
            "settings": settings,
            "grid": {
                "x": grid_x,
                "y": grid_y,
                "z": grid_z,
                "layout": "cells cover only seeds whose N plus real-data halo fits inside the cuboid",
            },
            "cells": cells,
            "stats": {
                "elapsedMs": round((time.perf_counter() - started) * 1000.0, 1),
                "cacheHit": False,
                "cacheKey": cache_key,
                "computeBackend": compute_backend,
                "computeDevice": compute_device,
                "lineFieldMs": round(line_field_ms, 1),
                "lineFieldBatchLaunches": batch_launches,
                "strengthScale": round(region_strength_scale, 8),
                "tileCount": len(tile_specs),
                "candidateBlockCount": len(candidate_records),
                "selectedCandidateCount": len(selected_candidates),
                "rawNeedleCount": len(raw_needles),
                "needleCount": len(catalog),
                "cellCount": len(cells),
                "validCellCount": len(valid_cells),
                "medianNormalConfidence": round(
                    float(
                        np.median(
                            [cell["normalConfidence"] for cell in valid_cells]
                        )
                    ),
                    4,
                )
                if valid_cells
                else None,
                "medianNeighborNormalDeg": round(float(np.median(normal_edges)), 3)
                if normal_edges
                else None,
                "medianNeighborPattern": round(float(np.median(pattern_edges)), 4)
                if pattern_edges
                else None,
                "constraint": "region evidence field; no sheet identity or surface connectivity assigned",
            },
        }
        _store_cache(cache_key, result)
        return result
