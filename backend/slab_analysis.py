from __future__ import annotations

import json
import math
import os
import time
import warnings
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np

from .acus import _plane_basis, _refine_needles_batch, _robust_common_normal
from .acus_compute import hessian_line_fields


NEEDLE_DTYPE = np.dtype(
    [
        ("center", "<f4", (3,)),
        ("direction", "<f4", (3,)),
        ("score", "<f4"),
        ("axialCoverage", "<f4"),
        ("supportScore", "<f4"),
    ]
)
CELL_DTYPE = np.dtype(
    [
        ("valid", "u1"),
        ("normal", "<f4", (3,)),
        ("needleCount", "<u2"),
        ("inlierFraction", "<f4"),
        ("normalConfidence", "<f4"),
        ("coplanarity", "<f4"),
        ("medianPlaneResidualDeg", "<f4"),
        ("medianAxialCoverage", "<f4"),
        ("twoModeCoverage", "<f4"),
        ("coveredDepthFraction", "<f4"),
        ("neighborNormalMedianDeg", "<f4"),
        ("neighborPatternMedian", "<f4"),
    ]
)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _fixed_tiles(size: int, halo: int, core: int) -> list[tuple[int, int]]:
    if size <= 2 * halo:
        raise ValueError("volume axis is too small for the requested halo")
    return [(start, min(start + core, size - halo)) for start in range(halo, size - halo, core)]


def _grid_axis(size: int, half_context: int, stride: int) -> list[int]:
    low, high = half_context, size - half_context
    values = list(range(low, high + 1, stride)) if high >= low else []
    if values and values[-1] != high:
        values.append(high)
    return values


def _tile_specs(shape_zyx: tuple[int, int, int], halo: int, core: int) -> list[dict[str, Any]]:
    z_tiles, y_tiles, x_tiles = (
        _fixed_tiles(size, halo, core) for size in shape_zyx
    )
    specs = []
    for tile_id, ((z0, z1), (y0, y1), (x0, x1)) in enumerate(
        product(z_tiles, y_tiles, x_tiles)
    ):
        specs.append(
            {
                "id": tile_id,
                "coreZYX": (z0, z1, y0, y1, x0, x1),
                "paddedZYX": (
                    z0 - halo,
                    z1 + halo,
                    y0 - halo,
                    y1 + halo,
                    x0 - halo,
                    x1 + halo,
                ),
            }
        )
    return specs


def _normalization(array: np.ndarray) -> tuple[float, float, float]:
    sample = np.asarray(array[::8, ::8, ::8], dtype=np.float32)
    positive = sample[sample > 0]
    if not len(positive):
        raise ValueError("the slab does not contain any nonzero scan data")
    low = 0.0
    high = float(np.percentile(positive, 99.5))
    if high <= 0:
        raise ValueError("the slab intensity range is degenerate")
    return low, high, max(4.0, high * 0.08)


def _normalize(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    return np.clip((np.asarray(raw, dtype=np.float32) - low) / (high - low), 0.0, 1.0)


def _vector_block_candidates(
    score: np.ndarray,
    spec: dict[str, Any],
    spacing: int,
    halo: int,
    bin_size: int,
    bin_shape_zyx: tuple[int, int, int],
    threshold: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    z0, z1, y0, y1, x0, x1 = spec["coreZYX"]
    pz0, _, py0, _, px0, _ = spec["paddedZYX"]
    core_score = score[z0 - pz0 : z1 - pz0, y0 - py0 : y1 - py0, x0 - px0 : x1 - px0]
    original_shape = core_score.shape
    block_shape = tuple(int(math.ceil(size / spacing)) for size in original_shape)
    padded_shape = tuple(count * spacing for count in block_shape)
    if padded_shape != original_shape:
        padded = np.full(padded_shape, -np.inf, dtype=np.float32)
        padded[: original_shape[0], : original_shape[1], : original_shape[2]] = core_score
    else:
        padded = core_score
    blocks = (
        padded.reshape(
            block_shape[0], spacing,
            block_shape[1], spacing,
            block_shape[2], spacing,
        )
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(*block_shape, spacing**3)
    )
    flat = np.argmax(blocks, axis=-1)
    values = np.take_along_axis(blocks, flat[..., None], axis=-1)[..., 0]
    dz, dy, dx = np.unravel_index(flat, (spacing, spacing, spacing))
    bz, by, bx = np.indices(block_shape, dtype=np.int32)
    global_z = z0 + bz * spacing + dz
    global_y = y0 + by * spacing + dy
    global_x = x0 + bx * spacing + dx
    valid = (
        (values >= threshold)
        & (global_z < z1)
        & (global_y < y1)
        & (global_x < x1)
    )
    values = values[valid].astype(np.float32)
    points_zyx = np.stack(
        [global_z[valid], global_y[valid], global_x[valid]], axis=1
    ).astype(np.int32)
    block_starts = np.stack(
        [z0 + bz[valid] * spacing, y0 + by[valid] * spacing, x0 + bx[valid] * spacing],
        axis=1,
    )
    bin_zyx = (block_starts - halo) // bin_size
    bin_ids = (
        (bin_zyx[:, 0] * bin_shape_zyx[1] + bin_zyx[:, 1])
        * bin_shape_zyx[2]
        + bin_zyx[:, 2]
    ).astype(np.int64)
    return values, points_zyx, bin_ids


def _select_candidates(
    values: np.ndarray,
    points_zyx: np.ndarray,
    bin_ids: np.ndarray,
    maximum_per_bin: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if not len(values):
        return values, points_zyx, bin_ids
    order = np.lexsort((-values, bin_ids))
    ordered_bins = bin_ids[order]
    starts = np.flatnonzero(np.r_[True, ordered_bins[1:] != ordered_bins[:-1]])
    selected_parts = []
    for group_index, start in enumerate(starts):
        stop = starts[group_index + 1] if group_index + 1 < len(starts) else len(order)
        selected_parts.append(order[start : min(stop, start + maximum_per_bin)])
    selected = np.concatenate(selected_parts) if selected_parts else np.empty(0, dtype=np.int64)
    return values[selected], points_zyx[selected], bin_ids[selected]


def _deduplicate_tile_needles(
    needles: list[dict[str, Any]], minimum_distance: float
) -> list[dict[str, Any]]:
    buckets: dict[tuple[int, int, int], list[np.ndarray]] = {}
    accepted: list[dict[str, Any]] = []
    distance2 = minimum_distance * minimum_distance
    for needle in sorted(needles, key=lambda item: float(item["score"]), reverse=True):
        center = np.asarray(needle["center"], dtype=np.float32)
        bucket = tuple(int(math.floor(float(value) / minimum_distance)) for value in center)
        duplicate = False
        for offset in product((-1, 0, 1), repeat=3):
            neighbor = tuple(bucket[axis] + offset[axis] for axis in range(3))
            if any(float(np.sum((center - other) ** 2)) < distance2 for other in buckets.get(neighbor, [])):
                duplicate = True
                break
        if duplicate:
            continue
        accepted.append(needle)
        buckets.setdefault(bucket, []).append(center)
    return accepted


def _compact_profile(
    normal_coordinates: np.ndarray,
    family_angles: np.ndarray,
    weights: np.ndarray,
    inliers: np.ndarray,
    cube_size: int,
) -> tuple[float, float, np.ndarray]:
    depth_sigma = max(2.0, cube_size / 24.0)
    depth_count = int(np.clip(round(cube_size / 2.0), 16, 48))
    depth_centers = np.linspace(
        -cube_size * 0.5 + depth_sigma,
        cube_size * 0.5 - depth_sigma,
        depth_count,
        dtype=np.float32,
    )
    orientation_centers = np.linspace(2.5, 177.5, 36, dtype=np.float32)
    base_weights = weights * inliers.astype(np.float32)
    depth_kernel = np.exp(
        -0.5 * ((depth_centers[:, None] - normal_coordinates[None, :]) / depth_sigma) ** 2
    )
    angle_delta = np.abs(orientation_centers[None, :] - family_angles[:, None])
    angle_delta = np.minimum(angle_delta, 180.0 - angle_delta)
    angle_kernel = np.exp(-0.5 * (angle_delta / 9.0) ** 2)
    weighted_depth = depth_kernel * base_weights[None, :]
    raw_density = weighted_depth @ angle_kernel
    support = weighted_depth.sum(axis=1)
    support_limit = max(float(support.max()) * 0.12, 1.0e-6)
    covered = support >= support_limit
    normalized = np.divide(
        raw_density,
        np.maximum(raw_density.sum(axis=1, keepdims=True), 1.0e-7),
        out=np.zeros_like(raw_density),
        where=raw_density.sum(axis=1, keepdims=True) > 1.0e-7,
    )
    two_mode = []
    for row in normalized[covered]:
        first = int(np.argmax(row))
        excluded = np.zeros(36, dtype=bool)
        excluded[(np.arange(first - 4, first + 5) % 36)] = True
        second_row = np.where(excluded, -1.0, row)
        second = int(np.argmax(second_row))
        selected = np.zeros(36, dtype=bool)
        selected[(np.arange(first - 3, first + 4) % 36)] = True
        selected[(np.arange(second - 3, second + 4) % 36)] = True
        two_mode.append(float(row[selected].sum()))
    signature = np.abs(np.fft.rfft(normalized, axis=1))[:, 1:5].astype(np.float32)
    signature /= max(float(signature.max()), 1.0e-7)
    return (
        float(np.mean(two_mode)) if two_mode else 0.0,
        float(np.mean(covered)) if len(covered) else 0.0,
        signature,
    )


def _analysis_identity(source: Path, settings: dict[str, Any]) -> dict[str, Any]:
    stat = source.stat()
    return {
        "source": str(source.resolve()),
        "sourceBytes": stat.st_size,
        "sourceMtimeNs": stat.st_mtime_ns,
        "settings": settings,
        "version": 1,
    }


def run_slab_analysis(
    source: str | Path,
    output_root: str | Path,
    request: dict[str, Any] | None = None,
) -> dict[str, Any]:
    request = request or {}
    source_path = Path(source)
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    source_array = np.load(source_path, mmap_mode="r")
    if source_array.ndim != 3 or source_array.dtype != np.uint8:
        raise ValueError("slab analysis expects a uint8 ZYX .npy volume")
    shape_zyx = tuple(int(value) for value in source_array.shape)
    cube_size = int(np.clip(request.get("cubeSize", 64), 24, 128))
    scale = float(np.clip(request.get("scale", 1.25), 0.7, 3.0))
    spacing = int(np.clip(request.get("spacing", 4), 3, 8))
    needle_length = float(np.clip(request.get("needleLength", 16.0), 6.0, 48.0))
    halo = int(max(request.get("halo", math.ceil(needle_length)), math.ceil(needle_length)))
    grid_stride = int(np.clip(request.get("gridStride", 32), 8, 128))
    tile_core = int(np.clip(request.get("tileCore", 128), 64, 256))
    bin_size = int(np.clip(request.get("binSize", 32), 16, 64))
    max_per_bin = int(np.clip(request.get("maxNeedlesPerBin", 32), 8, 64))
    max_needles = int(np.clip(request.get("maxNeedles", 160), 24, 320))
    requested_strength_scale = request.get("strengthScale")
    if requested_strength_scale is not None:
        requested_strength_scale = float(requested_strength_scale)
        if not np.isfinite(requested_strength_scale) or requested_strength_scale <= 0:
            raise ValueError("strengthScale must be a finite positive number")
    settings = {
        "cubeSize": cube_size,
        "scale": scale,
        "candidateSpacing": spacing,
        "needleLength": needle_length,
        "halo": halo,
        "gridStride": grid_stride,
        "tileCore": tile_core,
        "binSize": bin_size,
        "maxNeedlesPerBin": max_per_bin,
        "maxNeedles": max_needles,
    }
    if requested_strength_scale is not None:
        settings["fixedStrengthScale"] = requested_strength_scale
    identity = _analysis_identity(source_path, settings)
    manifest_path = output / "analysis.json"
    specs = _tile_specs(shape_zyx, halo, tile_core)
    bin_shape = tuple(int(math.ceil((size - 2 * halo) / bin_size)) for size in shape_zyx)
    total_bins = int(np.prod(bin_shape))
    catalog_path = output / "needles.npy"
    counts_path = output / "needle-counts.npy"
    tile_complete_path = output / "tile-complete.npy"

    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise ValueError("existing slab analysis does not match this source and settings")
        low = float(manifest["normalization"]["low"])
        high = float(manifest["normalization"]["high"])
        air_threshold = float(manifest["normalization"]["airThreshold"])
        strength_scale = manifest.get("strengthScale")
        catalog = np.load(catalog_path, mmap_mode="r+")
        counts = np.load(counts_path, mmap_mode="r+")
        tile_complete = np.load(tile_complete_path, mmap_mode="r+")
    else:
        low, high, air_threshold = _normalization(source_array)
        strength_scale = requested_strength_scale
        catalog = np.lib.format.open_memmap(
            catalog_path, mode="w+", dtype=NEEDLE_DTYPE, shape=(total_bins, max_per_bin)
        )
        counts = np.lib.format.open_memmap(
            counts_path, mode="w+", dtype=np.uint8, shape=(total_bins,)
        )
        tile_complete = np.lib.format.open_memmap(
            tile_complete_path, mode="w+", dtype=np.uint8, shape=(len(specs),)
        )
        manifest = {
            "identity": identity,
            "state": "created",
            "shapeZYX": list(shape_zyx),
            "binShapeZYX": list(bin_shape),
            "tileCount": len(specs),
            "normalization": {"low": low, "high": high, "airThreshold": air_threshold},
            "strengthScale": requested_strength_scale,
            "createdAt": datetime.now(UTC).isoformat(),
        }
        _atomic_json(manifest_path, manifest)

    if strength_scale is None:
        manifest["state"] = "calibrating"
        _atomic_json(manifest_path, manifest)
        sample_limit = int(np.clip(request.get("calibrationTiles", 96), 16, 256))
        calibration_specs = []
        for spec in specs:
            pz0, pz1, py0, py1, px0, px1 = spec["paddedZYX"]
            probe = source_array[pz0:pz1:8, py0:py1:8, px0:px1:8]
            if int(np.max(probe, initial=0)) > air_threshold:
                calibration_specs.append(spec)
        if not calibration_specs:
            raise ValueError("no material-bearing tiles were found in the slab")
        sample_indices = np.linspace(
            0, len(calibration_specs) - 1, min(sample_limit, len(calibration_specs))
        ).round().astype(int)
        scales = []
        for calibration_count, sample_index in enumerate(sample_indices, start=1):
            spec = calibration_specs[int(sample_index)]
            pz0, pz1, py0, py1, px0, px1 = spec["paddedZYX"]
            data = _normalize(source_array[pz0:pz1, py0:py1, px0:px1], low, high)
            _, metadata = hessian_line_fields([data], scale)
            scales.extend(float(value) for value in metadata["strengthScales"])
            if calibration_count % 8 == 0 or calibration_count == len(sample_indices):
                manifest.update(
                    {
                        "calibrationCompletedCount": calibration_count,
                        "calibrationTileCount": len(sample_indices),
                        "updatedAt": datetime.now(UTC).isoformat(),
                    }
                )
                _atomic_json(manifest_path, manifest)
                print(
                    f"calibration {calibration_count}/{len(sample_indices)} tiles",
                    flush=True,
                )
        strength_scale = float(np.median(scales))
        manifest["strengthScale"] = strength_scale
        manifest["calibrationTileCount"] = len(sample_indices)
        _atomic_json(manifest_path, manifest)

    pending = [spec for spec in specs if not tile_complete[spec["id"]]]
    limit_tiles = request.get("limitTiles")
    if limit_tiles is not None:
        pending = pending[: max(0, int(limit_tiles))]
    manifest["state"] = "extracting"
    _atomic_json(manifest_path, manifest)
    started = time.monotonic()
    completed_this_run = 0
    accepted_this_run = 0
    radius = max(3, int(math.ceil(scale * 2.5)))
    cross_section_radius = max(2.0, float(math.ceil(scale * 1.5)))

    next_progress_count = 8

    def checkpoint(force: bool = False) -> None:
        nonlocal next_progress_count
        if not force and completed_this_run < next_progress_count:
            return
        while next_progress_count <= completed_this_run:
            next_progress_count += 8
        catalog.flush()
        counts.flush()
        tile_complete.flush()
        elapsed = max(time.monotonic() - started, 1.0e-6)
        manifest.update(
            {
                "completedTileCount": int(np.count_nonzero(tile_complete)),
                "needleCount": int(np.sum(counts, dtype=np.int64)),
                "tilesPerSecondThisRun": round(completed_this_run / elapsed, 3),
                "acceptedNeedlesThisRun": accepted_this_run,
                "updatedAt": datetime.now(UTC).isoformat(),
            }
        )
        _atomic_json(manifest_path, manifest)
        print(
            f"tiles {manifest['completedTileCount']}/{len(specs)} · "
            f"needles {manifest['needleCount']} · {manifest['tilesPerSecondThisRun']:.2f} tile/s",
            flush=True,
        )

    def process_material_batch(
        batch: list[tuple[dict[str, Any], np.ndarray]],
    ) -> None:
        nonlocal completed_this_run, accepted_this_run
        if not batch:
            return
        fields, compute_metadata = hessian_line_fields(
            [data for _, data in batch], scale, float(strength_scale)
        )
        manifest["computeBackend"] = compute_metadata["backend"]
        manifest["computeDevice"] = compute_metadata["device"]
        manifest["computeFallbackReason"] = compute_metadata["fallbackReason"]
        for (spec, _), (score, direction_field) in zip(batch, fields):
            pz0, _, py0, _, px0, _ = spec["paddedZYX"]
            values, points, bin_ids = _vector_block_candidates(
                score, spec, spacing, halo, bin_size, bin_shape, 0.015
            )
            values, points, bin_ids = _select_candidates(
                values, points, bin_ids, max_per_bin
            )
            local_points = points - np.asarray([pz0, py0, px0], dtype=np.int32)
            candidates = np.column_stack([values, local_points]).astype(
                np.float32, copy=False
            )
            refined = _refine_needles_batch(
                score,
                direction_field,
                candidates,
                radius,
                needle_length,
                cross_section_radius,
            )
            resolved = []
            for needle in refined:
                candidate_index = int(needle["candidateIndex"])
                bin_id = int(bin_ids[candidate_index])
                center = np.asarray(needle["center"], dtype=np.float32) + np.asarray(
                    [px0, py0, pz0], dtype=np.float32
                )
                if not (
                    halo <= center[0] < shape_zyx[2] - halo
                    and halo <= center[1] < shape_zyx[1] - halo
                    and halo <= center[2] < shape_zyx[0] - halo
                ):
                    continue
                resolved.append(
                    {
                        "center": center,
                        "direction": np.asarray(needle["direction"], dtype=np.float32),
                        "score": float(needle["score"]),
                        "axialCoverage": float(needle["axialCoverage"]),
                        "supportScore": float(needle["supportScore"]),
                        "binId": int(bin_id),
                    }
                )
            accepted = _deduplicate_tile_needles(
                resolved, float(max(2, spacing - 1))
            )
            by_bin: dict[int, list[dict[str, Any]]] = {}
            for needle in accepted:
                by_bin.setdefault(int(needle["binId"]), []).append(needle)
            for bin_id, needles in by_bin.items():
                needles.sort(key=lambda item: float(item["score"]), reverse=True)
                needles = needles[:max_per_bin]
                counts[bin_id] = len(needles)
                for slot, needle in enumerate(needles):
                    catalog[bin_id, slot] = (
                        needle["center"],
                        needle["direction"],
                        needle["score"],
                        needle["axialCoverage"],
                        needle["supportScore"],
                    )
                accepted_this_run += len(needles)
            tile_complete[spec["id"]] = 1
            completed_this_run += 1
        checkpoint()

    maximum_batch_voxels = max(
        int(os.environ.get("ACUS_GPU_BATCH_VOXELS", "8000000")), 1
    )
    material_batches: dict[tuple[int, int, int], list[tuple[dict[str, Any], np.ndarray]]] = {}
    for spec in pending:
        pz0, pz1, py0, py1, px0, px1 = spec["paddedZYX"]
        raw = source_array[pz0:pz1, py0:py1, px0:px1]
        if int(np.max(raw, initial=0)) <= air_threshold:
            tile_complete[spec["id"]] = 1
            completed_this_run += 1
            checkpoint()
            continue
        data = _normalize(raw, low, high)
        shape = tuple(int(value) for value in data.shape)
        capacity = max(1, maximum_batch_voxels // int(np.prod(shape)))
        batch = material_batches.setdefault(shape, [])
        batch.append((spec, data))
        if len(batch) >= capacity:
            process_material_batch(batch)
            batch.clear()

    for batch in material_batches.values():
        process_material_batch(batch)
    checkpoint(force=True)

    extraction_complete = bool(np.all(tile_complete))
    if not extraction_complete:
        manifest["state"] = "extraction-partial"
        _atomic_json(manifest_path, manifest)
        return manifest

    grid_x = _grid_axis(shape_zyx[2], cube_size // 2 + halo, grid_stride)
    grid_y = _grid_axis(shape_zyx[1], cube_size // 2 + halo, grid_stride)
    grid_z = _grid_axis(shape_zyx[0], cube_size // 2 + halo, grid_stride)
    grid_shape = (len(grid_z), len(grid_y), len(grid_x))
    cells_path = output / "cells.npy"
    signatures_path = output / "signatures.npy"
    cell_complete_path = output / "cell-complete.npy"
    if cells_path.is_file():
        cells = np.load(cells_path, mmap_mode="r+")
        signatures = np.load(signatures_path, mmap_mode="r+")
        cell_complete = np.load(cell_complete_path, mmap_mode="r+")
    else:
        profile_depth_count = int(np.clip(round(cube_size / 2.0), 16, 48))
        cells = np.lib.format.open_memmap(cells_path, mode="w+", dtype=CELL_DTYPE, shape=grid_shape)
        cells["neighborNormalMedianDeg"] = np.nan
        cells["neighborPatternMedian"] = np.nan
        signatures = np.lib.format.open_memmap(
            signatures_path,
            mode="w+",
            dtype=np.float16,
            shape=(*grid_shape, profile_depth_count, 4),
        )
        cell_complete = np.lib.format.open_memmap(
            cell_complete_path, mode="w+", dtype=np.uint8, shape=grid_shape
        )
        _atomic_json(
            output / "grid.json",
            {"x": grid_x, "y": grid_y, "z": grid_z, "shapeZYX": list(grid_shape)},
        )

    manifest["state"] = "summarizing"
    manifest["gridShapeZYX"] = list(grid_shape)
    manifest["cellCount"] = int(np.prod(grid_shape))
    _atomic_json(manifest_path, manifest)
    half_cube = cube_size * 0.5
    summary_started = time.monotonic()
    completed_cells_this_run = 0
    limit_cells = request.get("limitCells")

    for iz, center_z in enumerate(grid_z):
        for iy, center_y in enumerate(grid_y):
            for ix, center_x in enumerate(grid_x):
                if cell_complete[iz, iy, ix]:
                    continue
                if limit_cells is not None and completed_cells_this_run >= int(limit_cells):
                    manifest["state"] = "summary-partial"
                    _atomic_json(manifest_path, manifest)
                    return manifest
                cell_center = np.asarray([center_x, center_y, center_z], dtype=np.float32)
                low_xyz = cell_center - half_cube - radius
                high_xyz = cell_center + half_cube + radius
                bin_low_xyz = np.floor((low_xyz - halo) / bin_size).astype(int)
                bin_high_xyz = np.floor((high_xyz - halo) / bin_size).astype(int)
                bin_low_xyz = np.maximum(bin_low_xyz, 0)
                bin_high_xyz = np.minimum(
                    bin_high_xyz,
                    np.asarray([bin_shape[2] - 1, bin_shape[1] - 1, bin_shape[0] - 1]),
                )
                ids = []
                for bz in range(bin_low_xyz[2], bin_high_xyz[2] + 1):
                    for by in range(bin_low_xyz[1], bin_high_xyz[1] + 1):
                        base = (bz * bin_shape[1] + by) * bin_shape[2]
                        ids.extend(base + bx for bx in range(bin_low_xyz[0], bin_high_xyz[0] + 1))
                bin_ids = np.asarray(ids, dtype=np.int64)
                bin_records = catalog[bin_ids]
                slot_mask = np.arange(max_per_bin)[None, :] < counts[bin_ids, None]
                records = bin_records[slot_mask]
                if len(records):
                    inside = np.all(np.abs(records["center"] - cell_center) <= half_cube, axis=1)
                    records = records[inside]
                if len(records) > max_needles:
                    chosen = np.argpartition(records["score"], -max_needles)[-max_needles:]
                    records = records[chosen]
                cell = cells[iz, iy, ix]
                cell["needleCount"] = len(records)
                if len(records) >= 6:
                    directions = np.asarray(records["direction"], dtype=np.float32)
                    weights = np.asarray(records["score"], dtype=np.float32)
                    normal, eigenvalues, robust_weights = _robust_common_normal(directions, weights)
                    dominant = int(np.argmax(np.abs(normal)))
                    if normal[dominant] < 0:
                        normal = -normal
                    residual = np.degrees(
                        np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0))
                    )
                    median_residual = float(np.median(residual))
                    inlier_limit = max(8.0, min(22.0, median_residual * 2.8))
                    inliers = residual <= inlier_limit
                    total_eigenvalue = max(float(eigenvalues.sum()), 1.0e-7)
                    confidence = float(
                        np.clip(
                            ((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1.0e-7))
                            * float(np.mean(inliers)),
                            0.0,
                            1.0,
                        )
                    )
                    u_axis, v_axis = _plane_basis(normal)
                    normal_coordinates = (records["center"] - cell_center) @ normal
                    family_angles = np.degrees(
                        np.arctan2(directions @ v_axis, directions @ u_axis)
                    ) % 180.0
                    two_mode, depth_coverage, signature = _compact_profile(
                        normal_coordinates,
                        family_angles,
                        weights * robust_weights,
                        inliers,
                        cube_size,
                    )
                    cell["valid"] = 1
                    cell["normal"] = normal
                    cell["inlierFraction"] = float(np.mean(inliers))
                    cell["normalConfidence"] = confidence
                    cell["coplanarity"] = float(
                        np.clip(1.0 - eigenvalues[0] / total_eigenvalue, 0.0, 1.0)
                    )
                    cell["medianPlaneResidualDeg"] = median_residual
                    cell["medianAxialCoverage"] = float(np.median(records["axialCoverage"]))
                    cell["twoModeCoverage"] = two_mode
                    cell["coveredDepthFraction"] = depth_coverage
                    signatures[iz, iy, ix] = signature.astype(np.float16)
                cell_complete[iz, iy, ix] = 1
                completed_cells_this_run += 1
            if completed_cells_this_run and completed_cells_this_run % max(1, len(grid_x) * 4) == 0:
                cells.flush()
                signatures.flush()
                cell_complete.flush()
                elapsed = max(time.monotonic() - summary_started, 1.0e-6)
                manifest.update(
                    {
                        "completedCellCount": int(np.count_nonzero(cell_complete)),
                        "validCellCount": int(np.count_nonzero(cells["valid"])),
                        "cellsPerSecondThisRun": round(completed_cells_this_run / elapsed, 2),
                        "updatedAt": datetime.now(UTC).isoformat(),
                    }
                )
                _atomic_json(manifest_path, manifest)
                print(
                    f"cells {manifest['completedCellCount']}/{manifest['cellCount']} · "
                    f"valid {manifest['validCellCount']} · {manifest['cellsPerSecondThisRun']:.1f} cell/s",
                    flush=True,
                )

    normal_neighbors = np.full((6, *grid_shape), np.nan, dtype=np.float32)
    pattern_neighbors = np.full((6, *grid_shape), np.nan, dtype=np.float32)
    normals = np.asarray(cells["normal"], dtype=np.float32)
    valid = np.asarray(cells["valid"], dtype=bool)
    signatures32 = np.asarray(signatures, dtype=np.float32)
    slot = 0
    for axis in range(3):
        left_slice = [slice(None)] * 3
        right_slice = [slice(None)] * 3
        left_slice[axis] = slice(0, -1)
        right_slice[axis] = slice(1, None)
        left_key, right_key = tuple(left_slice), tuple(right_slice)
        pair_valid = valid[left_key] & valid[right_key]
        dot = np.abs(np.sum(normals[left_key] * normals[right_key], axis=-1))
        angles = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
        angles = np.where(pair_valid, angles, np.nan)
        normal_neighbors[slot][left_key] = angles
        normal_neighbors[slot + 1][right_key] = angles
        first = signatures32[left_key].reshape(*angles.shape, -1)
        second = signatures32[right_key].reshape(*angles.shape, -1)
        denominator = np.linalg.norm(first, axis=-1) * np.linalg.norm(second, axis=-1)
        agreement = np.divide(
            np.sum(first * second, axis=-1),
            np.maximum(denominator, 1.0e-7),
        )
        agreement = np.where(pair_valid & (denominator > 1.0e-7), agreement, np.nan)
        pattern_neighbors[slot][left_key] = agreement
        pattern_neighbors[slot + 1][right_key] = agreement
        slot += 2
    with warnings.catch_warnings(), np.errstate(all="ignore"):
        warnings.simplefilter("ignore", RuntimeWarning)
        cells["neighborNormalMedianDeg"] = np.nanmedian(normal_neighbors, axis=0)
        cells["neighborPatternMedian"] = np.nanmedian(pattern_neighbors, axis=0)
    cells.flush()
    manifest.update(
        {
            "state": "complete",
            "completedTileCount": len(specs),
            "completedCellCount": int(np.prod(grid_shape)),
            "validCellCount": int(np.count_nonzero(cells["valid"])),
            "needleCount": int(np.sum(counts, dtype=np.int64)),
            "completedAt": datetime.now(UTC).isoformat(),
        }
    )
    _atomic_json(manifest_path, manifest)
    return manifest


def slab_status(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    fetch_path = Path(
        os.environ.get(
            "ACUS_SLAB_FETCH_MANIFEST",
            "/mnt/t5/acus-cross-scroll/pherc0358-z7168-d256-yfull-xfull.fetch.json",
        )
    )
    analysis_path = root / "analysis.json"
    status: dict[str, Any] = {"configured": True, "root": str(root)}
    if fetch_path.is_file():
        fetch = json.loads(fetch_path.read_text())
        status["fetch"] = {
            key: fetch.get(key)
            for key in (
                "state", "completedCount", "totalCount", "remainingCount",
                "downloadedBytesThisRun", "downloadMiBPerSecond", "updatedAt",
            )
        }
        status["source"] = fetch.get("identity")
    if analysis_path.is_file():
        analysis = json.loads(analysis_path.read_text())
        status["analysis"] = {
            key: analysis.get(key)
            for key in (
                "state", "completedTileCount", "tileCount", "needleCount",
                "completedCellCount", "cellCount", "validCellCount",
                "calibrationCompletedCount", "calibrationTileCount",
                "tilesPerSecondThisRun", "cellsPerSecondThisRun", "updatedAt", "completedAt",
            )
        }
        status["settings"] = analysis.get("identity", {}).get("settings")
        status["shapeZYX"] = analysis.get("shapeZYX")
    status["state"] = (
        status.get("analysis", {}).get("state")
        or status.get("fetch", {}).get("state")
        or "not-started"
    )
    return status


def _macro_radial_fit(
    cells: np.ndarray,
    grid_x: list[int],
    grid_y: list[int],
) -> dict[str, Any]:
    """Fit a common XY center to unsigned normal lines in the transverse plane."""
    valid = np.asarray(cells["valid"], dtype=bool)
    normals = np.asarray(cells["normal"], dtype=np.float64)
    confidence = np.asarray(cells["normalConfidence"], dtype=np.float64)
    coplanarity = np.asarray(cells["coplanarity"], dtype=np.float64)
    needle_count = np.asarray(cells["needleCount"], dtype=np.float64)
    xy_norm = np.linalg.norm(normals[..., :2], axis=-1)
    usable = valid & (confidence >= 0.08) & (xy_norm >= 0.25)
    if int(np.count_nonzero(usable)) < 64:
        return {
            "macroRadialFitCellCount": int(np.count_nonzero(usable)),
            "macroRadialCenterXY": None,
            "medianMacroRadialResidualDeg": None,
            "p90MacroRadialResidualDeg": None,
            "macroRadialNullMedianDeg": None,
            "macroRadialExcessDeg": None,
            "medianAbsoluteNormalZ": None,
        }

    _, y_coordinates, x_coordinates = np.meshgrid(
        np.arange(cells.shape[0]), grid_y, grid_x, indexing="ij"
    )
    points = np.stack([x_coordinates[usable], y_coordinates[usable]], axis=1)
    unit_xy = normals[..., :2][usable] / xy_norm[usable, None]
    tangents = np.stack([-unit_xy[:, 1], unit_xy[:, 0]], axis=1)
    weights = (
        np.clip(confidence[usable], 0.05, 1.0)
        * np.clip(coplanarity[usable], 0.1, 1.0)
        * np.sqrt(np.maximum(needle_count[usable], 1.0))
    )
    weighted_tangents = tangents * np.sqrt(weights[:, None])
    weighted_offsets = np.sum(tangents * points, axis=1) * np.sqrt(weights)
    center, _, rank, singular_values = np.linalg.lstsq(
        weighted_tangents, weighted_offsets, rcond=None
    )
    if rank < 2 or singular_values[-1] <= singular_values[0] * 1.0e-3:
        return {
            "macroRadialFitCellCount": int(len(points)),
            "macroRadialCenterXY": None,
            "medianMacroRadialResidualDeg": None,
            "p90MacroRadialResidualDeg": None,
            "macroRadialNullMedianDeg": None,
            "macroRadialExcessDeg": None,
            "medianAbsoluteNormalZ": round(float(np.median(np.abs(normals[..., 2][usable]))), 4),
        }
    radial = points - center[None, :]
    radial_norm = np.linalg.norm(radial, axis=1)
    away_from_center = radial_norm >= 0.5 * min(
        float(np.median(np.diff(grid_x))) if len(grid_x) > 1 else 1.0,
        float(np.median(np.diff(grid_y))) if len(grid_y) > 1 else 1.0,
    )
    unit_radial = radial[away_from_center] / np.maximum(
        radial_norm[away_from_center, None], 1.0e-7
    )
    usable_unit_xy = unit_xy[away_from_center]
    alignment = np.abs(np.sum(usable_unit_xy * unit_radial, axis=1))
    residual = np.degrees(np.arccos(np.clip(alignment, 0.0, 1.0)))
    rng = np.random.default_rng(358)
    null_medians = []
    for _ in range(16):
        shuffled = usable_unit_xy[rng.permutation(len(usable_unit_xy))]
        shuffled_alignment = np.abs(np.sum(shuffled * unit_radial, axis=1))
        shuffled_residual = np.degrees(
            np.arccos(np.clip(shuffled_alignment, 0.0, 1.0))
        )
        null_medians.append(float(np.median(shuffled_residual)))
    observed_median = float(np.median(residual))
    null_median = float(np.median(null_medians))
    return {
        "macroRadialFitCellCount": int(np.count_nonzero(away_from_center)),
        "macroRadialCenterXY": [round(float(center[0]), 2), round(float(center[1]), 2)],
        "medianMacroRadialResidualDeg": round(observed_median, 3),
        "p90MacroRadialResidualDeg": round(float(np.percentile(residual, 90.0)), 3),
        "macroRadialNullMedianDeg": round(null_median, 3),
        "macroRadialExcessDeg": round(null_median - observed_median, 3),
        "medianAbsoluteNormalZ": round(float(np.median(np.abs(normals[..., 2][usable]))), 4),
    }


def slab_overview(
    output_root: str | Path,
    maximum_cells: int = 60000,
    z_index: int | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    analysis = json.loads((root / "analysis.json").read_text())
    if analysis.get("state") != "complete":
        raise ValueError("the cross-scroll slab analysis is not complete")
    grid = json.loads((root / "grid.json").read_text())
    cells = np.load(root / "cells.npy", mmap_mode="r")
    grid_z, grid_y, grid_x = cells.shape
    if z_index is not None and not 0 <= z_index < grid_z:
        raise ValueError(f"zIndex must be between 0 and {grid_z - 1}")
    source_z_indices = list(range(grid_z)) if z_index is None else [z_index]
    view_cells = cells if z_index is None else cells[z_index : z_index + 1]
    sample = max(
        1,
        int(math.ceil(math.sqrt(view_cells.size / max(1, maximum_cells)))),
    )
    y_step = sample
    x_step = sample
    sampled = view_cells[:, ::y_step, ::x_step]
    output_cells = []
    x_values = grid["x"][::x_step]
    y_values = grid["y"][::y_step]
    z_values = [grid["z"][index] for index in source_z_indices]
    source_x_indices = list(range(0, grid_x, x_step))
    source_y_indices = list(range(0, grid_y, y_step))
    for iz, (source_iz, center_z) in enumerate(zip(source_z_indices, z_values)):
        for iy, (source_iy, center_y) in enumerate(zip(source_y_indices, y_values)):
            for ix, (source_ix, center_x) in enumerate(zip(source_x_indices, x_values)):
                cell = sampled[iz, iy, ix]
                valid = bool(cell["valid"])
                output_cells.append(
                    {
                        "index": [source_ix, source_iy, source_iz],
                        "center": [center_x, center_y, center_z],
                        "valid": valid,
                        "normal": np.round(cell["normal"], 6).tolist() if valid else None,
                        "needleCount": int(cell["needleCount"]),
                        "inlierFraction": round(float(cell["inlierFraction"]), 4) if valid else None,
                        "normalConfidence": round(float(cell["normalConfidence"]), 4) if valid else None,
                        "coplanarity": round(float(cell["coplanarity"]), 4) if valid else None,
                        "medianPlaneResidualDeg": round(float(cell["medianPlaneResidualDeg"]), 3) if valid else None,
                        "medianAxialCoverage": round(float(cell["medianAxialCoverage"]), 4) if valid else None,
                        "twoModeCoverage": round(float(cell["twoModeCoverage"]), 4) if valid else None,
                        "coveredDepthFraction": round(float(cell["coveredDepthFraction"]), 4) if valid else None,
                        "neighborNormalMedianDeg": round(float(cell["neighborNormalMedianDeg"]), 3) if valid else None,
                        "neighborPatternMedian": round(float(cell["neighborPatternMedian"]), 4) if valid else None,
                    }
                )
    valid_cells = view_cells["valid"].astype(bool)
    settings = analysis["identity"]["settings"]
    source_sidecar = json.loads(Path(analysis["identity"]["source"]).with_suffix(".json").read_text())
    macro_fit = _macro_radial_fit(cells, grid["x"], grid["y"])
    depth_fits = []
    for iz, center_z in enumerate(grid["z"]):
        depth_fit = _macro_radial_fit(cells[iz : iz + 1], grid["x"], grid["y"])
        depth_fits.append(
            {
                "z": center_z,
                "centerXY": depth_fit["macroRadialCenterXY"],
                "medianResidualDeg": depth_fit["medianMacroRadialResidualDeg"],
                "cellCount": depth_fit["macroRadialFitCellCount"],
            }
        )
    resolved_depth_fits = [item for item in depth_fits if item["centerXY"] is not None]
    if len(resolved_depth_fits) >= 2:
        depth_z = np.asarray([item["z"] for item in resolved_depth_fits], dtype=np.float64)
        depth_centers = np.asarray(
            [item["centerXY"] for item in resolved_depth_fits], dtype=np.float64
        )
        slopes = np.asarray(
            [np.polyfit(depth_z, depth_centers[:, axis], 1)[0] for axis in range(2)]
        )
        center_drift = slopes * float(depth_z[-1] - depth_z[0])
        macro_fit["macroCenterDriftXY"] = [
            round(float(center_drift[0]), 2),
            round(float(center_drift[1]), 2),
        ]
        macro_fit["macroCenterDriftVoxels"] = round(
            float(np.linalg.norm(center_drift)), 2
        )
    else:
        macro_fit["macroCenterDriftXY"] = None
        macro_fit["macroCenterDriftVoxels"] = None
    macro_fit["macroDepthFits"] = depth_fits
    return {
        "shape": {"x": int(analysis["shapeZYX"][2]), "y": int(analysis["shapeZYX"][1]), "z": int(analysis["shapeZYX"][0])},
        "globalOrigin": dict(zip(("x", "y", "z"), source_sidecar.get("originXYZ", [0, 0, 0]))),
        "settings": {
            "cubeSize": settings["cubeSize"],
            "scale": settings["scale"],
            "candidateSpacing": settings["candidateSpacing"],
            "maxNeedles": settings["maxNeedles"],
            "needleLength": settings["needleLength"],
            "requestedPadding": settings["halo"],
            "effectivePadding": settings["halo"],
            "minimumPadding": settings["halo"],
            "gridStride": settings["gridStride"],
            "tileCore": settings["tileCore"],
            "catalogBinSize": settings["binSize"],
            "maxNeedlesPerBin": settings["maxNeedlesPerBin"],
        },
        "grid": {
            "x": x_values,
            "y": y_values,
            "z": z_values,
            "availableZ": grid["z"],
            "layout": (
                f"LOD {sample} of axial field plane {z_index + 1}/{grid_z}"
                if z_index is not None
                else f"LOD {sample} of the complete evidence grid"
            ),
        },
        "view": {
            "mode": "slice" if z_index is not None else "volume",
            "zIndex": z_index,
            "z": grid["z"][z_index] if z_index is not None else None,
            "sourceGridShapeZYX": [grid_z, grid_y, grid_x],
            "displayGridShapeZYX": list(sampled.shape),
        },
        "cells": output_cells,
        "stats": {
            "elapsedMs": 0.0,
            "cacheHit": True,
            "cacheKey": "cross-scroll-slab",
            "computeBackend": "gpu",
            "computeDevice": "NVIDIA GeForce GTX 1080",
            "lineFieldMs": 0.0,
            "lineFieldBatchLaunches": analysis["tileCount"],
            "strengthScale": analysis["strengthScale"],
            "tileCount": analysis["tileCount"],
            "candidateBlockCount": 0,
            "selectedCandidateCount": int(analysis["needleCount"]),
            "rawNeedleCount": int(analysis["needleCount"]),
            "needleCount": int(analysis["needleCount"]),
            "cellCount": int(view_cells.size),
            "validCellCount": int(np.count_nonzero(valid_cells)),
            "medianNormalConfidence": round(float(np.median(view_cells["normalConfidence"][valid_cells])), 4),
            "medianNeighborNormalDeg": round(float(np.nanmedian(view_cells["neighborNormalMedianDeg"][valid_cells])), 3),
            "medianNeighborPattern": round(float(np.nanmedian(view_cells["neighborPatternMedian"][valid_cells])), 4),
            **macro_fit,
            "constraint": "cross-scroll evidence field; no sheet identity or surface connectivity assigned",
        },
    }
