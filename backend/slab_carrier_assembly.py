from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rectify import grayscale_png
from .slab_sheetlet_carriers import (
    CARRIER_SCREEN_VERSION,
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _sample_stack,
    _texture_profile,
)


CARRIER_BOUNDARY_VERSION = 2
CARRIER_ASSEMBLY_VERSION = 2
ASSEMBLY_PREVIEW_VERSION = 2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _carrier_boundary(
    carrier: dict[str, Any], spacing: float = 12.0, maximum_points: int = 128
) -> dict[str, np.ndarray]:
    mask = np.asarray(carrier["supportMask"], dtype=bool)
    interior = np.zeros_like(mask)
    interior[1:-1, 1:-1] = (
        mask[1:-1, 1:-1]
        & mask[1:-1, :-2]
        & mask[1:-1, 2:]
        & mask[:-2, 1:-1]
        & mask[2:, 1:-1]
    )
    boundary = mask & ~interior
    y, x = np.nonzero(boundary)
    if not len(x):
        return {
            "point": np.empty((0, 3), dtype=np.float32),
            "normal": np.empty((0, 3), dtype=np.float32),
            "fiber": np.empty((0, 3), dtype=np.float32),
            "outward": np.empty((0, 3), dtype=np.float32),
        }

    u = np.asarray(carrier["uValues"])[x]
    v = np.asarray(carrier["vValues"])[y]
    bin_u = np.floor((u - float(np.min(u))) / spacing).astype(np.int32)
    bin_v = np.floor((v - float(np.min(v))) / spacing).astype(np.int32)
    bin_width = int(np.max(bin_u)) + 1
    bin_code = bin_u.astype(np.int64) + bin_width * bin_v.astype(np.int64)
    _, selected = np.unique(bin_code, return_index=True)
    if len(selected) > maximum_points:
        center_u, center_v = float(np.mean(u[selected])), float(np.mean(v[selected]))
        angle = np.arctan2(v[selected] - center_v, u[selected] - center_u)
        ordered = selected[np.argsort(angle)]
        selected = ordered[
            np.linspace(0, len(ordered) - 1, maximum_points, dtype=np.int32)
        ]
    x, y = x[selected], y[selected]

    left = np.zeros_like(mask)
    right = np.zeros_like(mask)
    up = np.zeros_like(mask)
    down = np.zeros_like(mask)
    left[:, 1:] = mask[:, :-1]
    right[:, :-1] = mask[:, 1:]
    up[1:, :] = mask[:-1, :]
    down[:-1, :] = mask[1:, :]
    outward_u = left[y, x].astype(np.float32) - right[y, x].astype(np.float32)
    outward_v = up[y, x].astype(np.float32) - down[y, x].astype(np.float32)

    normals = np.asarray(carrier["normalXYZ"])[y, x]
    fibers = np.asarray(carrier["fiberXYZ"])[y, x]
    u_axis = np.asarray(carrier["frame"]["uAxis"], dtype=np.float32)
    v_axis = np.asarray(carrier["frame"]["vAxis"], dtype=np.float32)
    tangent_u = u_axis[None, :] - np.sum(
        u_axis[None, :] * normals, axis=1, keepdims=True
    ) * normals
    tangent_v = v_axis[None, :] - np.sum(
        v_axis[None, :] * normals, axis=1, keepdims=True
    ) * normals
    outward = outward_u[:, None] * tangent_u + outward_v[:, None] * tangent_v
    outward /= np.maximum(np.linalg.norm(outward, axis=1, keepdims=True), 1.0e-8)
    return {
        "point": np.asarray(carrier["surfaceXYZ"])[y, x].astype(np.float32),
        "normal": normals.astype(np.float32),
        "fiber": fibers.astype(np.float32),
        "outward": outward.astype(np.float32),
    }


def build_carrier_boundaries(
    output_root: str | Path,
    force: bool = False,
    minimum_fit_factor: float = 0.7,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / f"sheetlet-carrier-boundaries-v{CARRIER_BOUNDARY_VERSION}.json"
    artifact_path = root / f"sheetlet-carrier-boundaries-v{CARRIER_BOUNDARY_VERSION}.npz"
    screen = json.loads(
        (
            root / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json"
        ).read_text()
    )
    source_path, candidate_payload, component, flakes = _load_carrier_catalog(root)
    legacy_scale_components = {
        int(value["componentId"])
        for value in candidate_payload["candidates"]
        if value.get("candidateClass", "legacy-scale") == "legacy-scale"
    }
    eligible = [
        value
        for value in screen["candidates"]
        if float(value["yield"]["fitFactor"]) >= minimum_fit_factor
        and int(value["componentId"]) in legacy_scale_components
    ]
    settings = {
        "minimumFitFactor": minimum_fit_factor,
        "boundarySpacingVoxels": 12.0,
        "maximumBoundaryPointsPerCarrier": 128,
        "carrierPixelStepVoxels": 4.0,
        "carrierMaximumPixelsPerAxis": 192,
        "includedCandidateClasses": ["legacy-scale"],
        "secondarySeedsDeferredToDedicatedGrowth": True,
    }
    identity = {
        "version": CARRIER_BOUNDARY_VERSION,
        "screenIdentity": screen["identity"],
        "source": str(source_path),
        "sourceMtimeNs": source_path.stat().st_mtime_ns,
        "screenVersion": screen["identity"]["version"],
    }
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity and cached.get("settings") == settings:
            return cached

    cell_indices = np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32)
    cell_shape = np.max(cell_indices, axis=0) + 1
    cell_code = (
        cell_indices[:, 0].astype(np.int64)
        + int(cell_shape[0])
        * (
            cell_indices[:, 1].astype(np.int64)
            + int(cell_shape[1]) * cell_indices[:, 2].astype(np.int64)
        )
    )
    points: list[np.ndarray] = []
    normals: list[np.ndarray] = []
    fibers: list[np.ndarray] = []
    outwards: list[np.ndarray] = []
    point_carrier: list[np.ndarray] = []
    point_offsets = [0]
    cells: list[np.ndarray] = []
    cell_offsets = [0]
    source_ranks = []
    component_ids = []
    member_counts = []
    fit_factors = []
    bounds_low = []
    bounds_high = []
    started = time.monotonic()
    for carrier_index, value in enumerate(eligible):
        component_id = int(value["componentId"])
        member_indices = np.flatnonzero(component == component_id)
        member_flakes = [flakes[int(index)] for index in member_indices]
        carrier = _mls_carrier(
            member_flakes,
            pixel_step=4.0,
            bandwidth=48.0,
            support_radius=48.0,
            maximum_pixels=192,
        )
        boundary = _carrier_boundary(carrier)
        points.append(boundary["point"])
        normals.append(boundary["normal"])
        fibers.append(boundary["fiber"])
        outwards.append(boundary["outward"])
        point_carrier.append(
            np.full(len(boundary["point"]), carrier_index, dtype=np.uint32)
        )
        point_offsets.append(point_offsets[-1] + len(boundary["point"]))
        unique_cells = np.unique(cell_code[member_indices])
        cells.append(unique_cells)
        cell_offsets.append(cell_offsets[-1] + len(unique_cells))
        source_ranks.append(int(value["sourceRank"]))
        component_ids.append(component_id)
        member_counts.append(len(member_indices))
        fit_factors.append(float(value["yield"]["fitFactor"]))
        if len(boundary["point"]):
            bounds_low.append(np.min(boundary["point"], axis=0))
            bounds_high.append(np.max(boundary["point"], axis=0))
        else:
            bounds_low.append(np.full(3, np.nan, dtype=np.float32))
            bounds_high.append(np.full(3, np.nan, dtype=np.float32))
        completed = carrier_index + 1
        if progress is not None and (completed % 100 == 0 or completed == len(eligible)):
            progress(completed, len(eligible), (time.monotonic() - started) * 1000.0)

    arrays = {
        "point": np.concatenate(points).astype(np.float32),
        "normal": np.concatenate(normals).astype(np.float32),
        "fiber": np.concatenate(fibers).astype(np.float32),
        "outward": np.concatenate(outwards).astype(np.float32),
        "pointCarrier": np.concatenate(point_carrier).astype(np.uint32),
        "pointOffset": np.asarray(point_offsets, dtype=np.uint32),
        "cellCode": np.concatenate(cells).astype(np.int64),
        "cellOffset": np.asarray(cell_offsets, dtype=np.uint32),
        "sourceRank": np.asarray(source_ranks, dtype=np.uint32),
        "componentId": np.asarray(component_ids, dtype=np.uint32),
        "memberCount": np.asarray(member_counts, dtype=np.uint32),
        "fitFactor": np.asarray(fit_factors, dtype=np.float32),
        "boundsLow": np.asarray(bounds_low, dtype=np.float32),
        "boundsHigh": np.asarray(bounds_high, dtype=np.float32),
    }
    _atomic_npz(artifact_path, **arrays)
    result = {
        "identity": identity,
        "settings": settings,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "carrierCount": len(eligible),
            "boundaryPointCount": int(len(arrays["point"])),
            "medianBoundaryPointsPerCarrier": round(
                float(np.median(np.diff(arrays["pointOffset"]))), 2
            ),
        },
        "artifact": str(artifact_path.relative_to(root)),
    }
    _atomic_json(summary_path, result)
    return result


def _transported_fiber_angle(
    first_normal: np.ndarray,
    second_normal: np.ndarray,
    first_fiber: np.ndarray,
    second_fiber: np.ndarray,
) -> np.ndarray:
    second = second_normal.copy()
    second[np.sum(first_normal * second, axis=1) < 0.0] *= -1.0
    axis = np.cross(first_normal, second)
    sine2 = np.sum(axis**2, axis=1)
    cosine = np.clip(np.sum(first_normal * second, axis=1), -1.0, 1.0)
    once = np.cross(axis, first_fiber)
    twice = np.cross(axis, once)
    factor = np.divide(
        1.0 - cosine,
        np.maximum(sine2, 1.0e-12),
        out=np.zeros_like(cosine),
        where=sine2 >= 1.0e-12,
    )
    transported = first_fiber + once + twice * factor[:, None]
    transported /= np.maximum(np.linalg.norm(transported, axis=1, keepdims=True), 1.0e-8)
    dot = np.clip(np.abs(np.sum(transported * second_fiber, axis=1)), 0.0, 1.0)
    return np.degrees(np.arccos(dot))


def _score_point_pairs(
    first_index: np.ndarray,
    second_index: np.ndarray,
    arrays: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    point = arrays["point"]
    normal = arrays["normal"]
    fiber = arrays["fiber"]
    outward = arrays["outward"]
    delta = point[second_index] - point[first_index]
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], 1.0e-8)
    first_normal = normal[first_index]
    second_normal = normal[second_index]
    signed_second = second_normal.copy()
    signed_second[np.sum(first_normal * signed_second, axis=1) < 0.0] *= -1.0
    normal_dot = np.clip(np.sum(first_normal * signed_second, axis=1), 0.0, 1.0)
    normal_bend = np.degrees(np.arccos(normal_dot))
    plane_residual = 0.5 * (
        np.abs(np.sum(delta * first_normal, axis=1))
        + np.abs(np.sum(delta * signed_second, axis=1))
    )
    fiber_angle = _transported_fiber_angle(
        first_normal, signed_second, fiber[first_index], fiber[second_index]
    )
    first_facing = np.sum(outward[first_index] * unit, axis=1)
    second_facing = np.sum(outward[second_index] * -unit, axis=1)
    facing = np.minimum(first_facing, second_facing)
    score = np.exp(
        -0.5
        * (
            (distance / 24.0) ** 2
            + (plane_residual / 3.0) ** 2
            + (fiber_angle / 10.0) ** 2
            + (normal_bend / 18.0) ** 2
        )
    ) * np.sqrt(np.clip(first_facing, 0.0, 1.0) * np.clip(second_facing, 0.0, 1.0))
    valid = (
        (distance > 0.25)
        & (distance <= 40.0)
        & (plane_residual <= 7.0)
        & (fiber_angle <= 30.0)
        & (normal_bend <= 40.0)
        & (facing >= 0.1)
        & (score >= 0.08)
    )
    return {
        "valid": valid,
        "score": score,
        "distance": distance,
        "planeResidual": plane_residual,
        "fiberAngle": fiber_angle,
        "normalBend": normal_bend,
        "facing": facing,
    }


def _cell_sets(arrays: dict[str, np.ndarray]) -> list[set[int]]:
    offsets = arrays["cellOffset"]
    codes = arrays["cellCode"]
    return [
        set(int(value) for value in codes[int(offsets[i]) : int(offsets[i + 1])])
        for i in range(len(offsets) - 1)
    ]


def match_carrier_boundaries(
    output_root: str | Path,
    force: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    boundary_summary = build_carrier_boundaries(root)
    boundary_path = root / boundary_summary["artifact"]
    summary_path = root / f"sheetlet-carrier-assembly-v{CARRIER_ASSEMBLY_VERSION}.json"
    edge_path = root / f"sheetlet-carrier-assembly-v{CARRIER_ASSEMBLY_VERSION}-edges.npz"
    identity = {
        "version": CARRIER_ASSEMBLY_VERSION,
        "boundaryIdentity": boundary_summary["identity"],
    }
    if summary_path.is_file() and edge_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            return cached
    with np.load(boundary_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}

    point = arrays["point"]
    carrier = arrays["pointCarrier"]
    bucket_width = 24.0
    maximum_distance = 40.0
    bucket_key = np.floor(point / bucket_width).astype(np.int32)
    unique_key, inverse = np.unique(bucket_key, axis=0, return_inverse=True)
    order = np.argsort(inverse, kind="stable")
    counts = np.bincount(inverse, minlength=len(unique_key))
    starts = np.r_[0, np.cumsum(counts)]
    key_lookup = {tuple(int(v) for v in key): i for i, key in enumerate(unique_key)}
    cell_sets = _cell_sets(arrays)
    pair_hits: dict[tuple[int, int], dict[str, Any]] = {}
    bucket_radius = int(math.ceil(maximum_distance / bucket_width))
    offsets = [
        (dx, dy, dz)
        for dx in range(-bucket_radius, bucket_radius + 1)
        for dy in range(-bucket_radius, bucket_radius + 1)
        for dz in range(-bucket_radius, bucket_radius + 1)
        if (dx, dy, dz) >= (0, 0, 0)
    ]
    started = time.monotonic()
    total_buckets = len(unique_key)
    for bucket_index, key in enumerate(unique_key):
        first_points = order[starts[bucket_index] : starts[bucket_index + 1]]
        key_tuple = tuple(int(value) for value in key)
        for offset in offsets:
            neighbor_key = tuple(key_tuple[i] + offset[i] for i in range(3))
            neighbor_index = key_lookup.get(neighbor_key)
            if neighbor_index is None:
                continue
            second_points = order[starts[neighbor_index] : starts[neighbor_index + 1]]
            product = len(first_points) * len(second_points)
            if not product:
                continue
            for chunk_start in range(0, product, 200_000):
                chunk_stop = min(chunk_start + 200_000, product)
                flat = np.arange(chunk_start, chunk_stop, dtype=np.int64)
                first_index = first_points[flat // len(second_points)]
                second_index = second_points[flat % len(second_points)]
                if neighbor_index == bucket_index:
                    keep = first_index < second_index
                    first_index, second_index = first_index[keep], second_index[keep]
                distinct = carrier[first_index] != carrier[second_index]
                first_index, second_index = first_index[distinct], second_index[distinct]
                if not len(first_index):
                    continue
                scored = _score_point_pairs(first_index, second_index, arrays)
                valid_positions = np.flatnonzero(scored["valid"])
                for position in valid_positions:
                    point_a, point_b = int(first_index[position]), int(second_index[position])
                    carrier_a, carrier_b = int(carrier[point_a]), int(carrier[point_b])
                    if carrier_a > carrier_b:
                        carrier_a, carrier_b = carrier_b, carrier_a
                        point_a, point_b = point_b, point_a
                    pair = (carrier_a, carrier_b)
                    entry = pair_hits.get(pair)
                    score = float(scored["score"][position])
                    if entry is None:
                        entry = {
                            "sourcePoints": set(),
                            "targetPoints": set(),
                            "top": [],
                            "best": None,
                        }
                        pair_hits[pair] = entry
                    entry["sourcePoints"].add(point_a)
                    entry["targetPoints"].add(point_b)
                    entry["top"].append(score)
                    if len(entry["top"]) > 12:
                        entry["top"] = sorted(entry["top"], reverse=True)[:12]
                    if entry["best"] is None or score > entry["best"]["score"]:
                        entry["best"] = {
                            "score": score,
                            "distance": float(scored["distance"][position]),
                            "planeResidual": float(scored["planeResidual"][position]),
                            "fiberAngle": float(scored["fiberAngle"][position]),
                            "normalBend": float(scored["normalBend"][position]),
                            "facing": float(scored["facing"][position]),
                        }
        completed = bucket_index + 1
        if progress is not None and (completed % 500 == 0 or completed == total_buckets):
            progress(completed, total_buckets, (time.monotonic() - started) * 1000.0)

    edges = []
    overlap_rejected = 0
    for (source, target), entry in pair_hits.items():
        support = min(len(entry["sourcePoints"]), len(entry["targetPoints"]))
        if support < 2:
            continue
        if not cell_sets[source].isdisjoint(cell_sets[target]):
            overlap_rejected += 1
            continue
        top_scores = np.asarray(entry["top"], dtype=np.float32)
        aggregate = float(np.median(top_scores[: min(5, len(top_scores))]))
        aggregate *= 0.85 + 0.15 * min(support / 6.0, 1.0)
        best = entry["best"]
        edges.append(
            (
                source,
                target,
                aggregate,
                support,
                best["distance"],
                best["planeResidual"],
                best["fiberAngle"],
                best["normalBend"],
                best["facing"],
            )
        )
    edges.sort(key=lambda value: value[2], reverse=True)
    edge_values = np.asarray(edges, dtype=np.float64) if edges else np.empty((0, 9))
    edge_arrays = {
        "source": edge_values[:, 0].astype(np.uint32),
        "target": edge_values[:, 1].astype(np.uint32),
        "score": edge_values[:, 2].astype(np.float32),
        "support": edge_values[:, 3].astype(np.uint16),
        "distance": edge_values[:, 4].astype(np.float32),
        "planeResidual": edge_values[:, 5].astype(np.float32),
        "fiberAngle": edge_values[:, 6].astype(np.float32),
        "normalBend": edge_values[:, 7].astype(np.float32),
        "facing": edge_values[:, 8].astype(np.float32),
    }
    _atomic_npz(edge_path, **edge_arrays)
    result = {
        "identity": identity,
        "settings": {
            "bucketWidthVoxels": bucket_width,
            "maximumBoundaryDistanceVoxels": maximum_distance,
            "spatialBucketRadius": bucket_radius,
            "maximumPlaneResidualVoxels": 7.0,
            "maximumFiberAngleDeg": 30.0,
            "maximumNormalBendDeg": 40.0,
            "minimumFacingCosine": 0.1,
            "minimumBoundarySupport": 2,
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "carrierCount": int(len(arrays["sourceRank"])),
            "boundaryPointCount": int(len(point)),
            "spatialBucketCount": total_buckets,
            "candidateCarrierPairCount": len(pair_hits),
            "overlappingCellPairRejectedCount": overlap_rejected,
            "edgeCount": len(edges),
        },
        "artifacts": {
            "boundaries": str(boundary_path.relative_to(root)),
            "edges": str(edge_path.relative_to(root)),
        },
    }
    _atomic_json(summary_path, result)
    return result


def _assemble_at_threshold(
    arrays: dict[str, np.ndarray],
    edges: dict[str, np.ndarray],
    threshold: float,
) -> tuple[dict[str, Any], np.ndarray, np.ndarray]:
    node_count = len(arrays["sourceRank"])
    parent = np.arange(node_count, dtype=np.int32)
    tree_size = np.ones(node_count, dtype=np.int32)
    component_cells = _cell_sets(arrays)
    degree = np.zeros(node_count, dtype=np.uint16)
    retained = np.zeros(len(edges["score"]), dtype=bool)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    accepted_indices = np.flatnonzero(edges["score"] >= threshold)
    conflict_count = 0
    for edge_index in accepted_indices[np.argsort(edges["score"][accepted_indices])[::-1]]:
        source = int(edges["source"][edge_index])
        target = int(edges["target"][edge_index])
        first, second = find(source), find(target)
        if first == second:
            retained[edge_index] = True
        else:
            if not component_cells[first].isdisjoint(component_cells[second]):
                conflict_count += 1
                continue
            if int(tree_size[first]) < int(tree_size[second]):
                first, second = second, first
            parent[second] = first
            tree_size[first] += tree_size[second]
            component_cells[first].update(component_cells[second])
            component_cells[second] = set()
            retained[edge_index] = True
        degree[source] += 1
        degree[target] += 1

    roots = np.fromiter((find(index) for index in range(node_count)), dtype=np.int32)
    _, component, carrier_counts = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    member_count = arrays["memberCount"].astype(np.int64)
    flake_counts = np.bincount(
        component, weights=member_count, minlength=len(carrier_counts)
    ).astype(np.int64)
    merged = carrier_counts >= 2
    selected_edges = edges["score"][retained]
    top_ids = np.lexsort((flake_counts, carrier_counts))[::-1][:20]
    top_components = []
    for component_id in top_ids:
        carrier_count = int(carrier_counts[component_id])
        if carrier_count < 2:
            continue
        member = component == component_id
        node_indices = np.flatnonzero(member)
        component_edge = retained & member[edges["source"]] & member[edges["target"]]
        top_components.append(
            {
                "componentId": int(component_id),
                "carrierCount": carrier_count,
                "flakeCount": int(flake_counts[component_id]),
                "uniqueCellCount": int(
                    sum(len(component_cells[find(int(index))]) for index in node_indices[:1])
                ),
                "sourceRanks": arrays["sourceRank"][member].astype(int).tolist(),
                "sheetletComponentIds": arrays["componentId"][member]
                .astype(int)
                .tolist(),
                "extentXYZ": np.round(
                    np.max(arrays["boundsHigh"][member], axis=0)
                    - np.min(arrays["boundsLow"][member], axis=0),
                    2,
                ).tolist(),
                "medianBoundaryScore": round(
                    float(np.median(edges["score"][component_edge])), 4
                ),
                "medianBoundaryDistanceVoxels": round(
                    float(np.median(edges["distance"][component_edge])), 3
                ),
                "medianPlaneResidualVoxels": round(
                    float(np.median(edges["planeResidual"][component_edge])), 3
                ),
                "medianTransportedFiberAngleDeg": round(
                    float(np.median(edges["fiberAngle"][component_edge])), 3
                ),
                "medianLocalNormalBendDeg": round(
                    float(np.median(edges["normalBend"][component_edge])), 3
                ),
            }
        )
    return (
        {
            "threshold": threshold,
            "acceptedEdgeCount": int(len(accepted_indices)),
            "retainedEdgeCount": int(np.count_nonzero(retained)),
            "cellConflictRejectedEdgeCount": conflict_count,
            "linkedCarrierCount": int(np.count_nonzero(carrier_counts[component] >= 2)),
            "mergedComponentCount": int(np.count_nonzero(merged)),
            "largestCarrierCount": int(np.max(carrier_counts[merged])) if np.any(merged) else 1,
            "medianMergedCarrierCount": round(
                float(np.median(carrier_counts[merged])), 2
            )
            if np.any(merged)
            else None,
            "medianRetainedScore": round(float(np.median(selected_edges)), 4)
            if len(selected_edges)
            else None,
            "topComponents": top_components,
        },
        component.astype(np.int32),
        retained,
    )


def assemble_carriers(
    output_root: str | Path,
    thresholds: tuple[float, ...] = (0.25, 0.35, 0.45, 0.55, 0.65),
    selected_threshold: float = 0.45,
    force: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    match_summary = match_carrier_boundaries(root, force=force, progress=progress)
    boundary_path = root / match_summary["artifacts"]["boundaries"]
    edge_path = root / match_summary["artifacts"]["edges"]
    with np.load(boundary_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(edge_path) as payload:
        edges = {key: np.asarray(payload[key]) for key in payload.files}
    sweeps = []
    selected_component = None
    selected_retained = None
    for threshold in thresholds:
        summary, component, retained = _assemble_at_threshold(arrays, edges, threshold)
        sweeps.append(summary)
        if abs(threshold - selected_threshold) < 1.0e-8:
            selected_component = component
            selected_retained = retained
    if selected_component is None or selected_retained is None:
        raise ValueError("selected_threshold must be included in thresholds")
    component_path = root / (
        f"sheetlet-carrier-assembly-v{CARRIER_ASSEMBLY_VERSION}-components.npz"
    )
    _atomic_npz(
        component_path,
        component=selected_component,
        retainedEdge=selected_retained,
    )
    selected = next(
        value for value in sweeps if value["threshold"] == selected_threshold
    )
    result = {
        **match_summary,
        "settings": {
            **match_summary["settings"],
            "thresholdSweep": list(thresholds),
            "selectedThreshold": selected_threshold,
            "assemblyConstraint": "strongest-first union with no repeated Acus cell",
        },
        "sweep": sweeps,
        "selected": selected,
        "artifacts": {
            **match_summary["artifacts"],
            "components": str(component_path.relative_to(root)),
        },
    }
    summary_path = root / f"sheetlet-carrier-assembly-v{CARRIER_ASSEMBLY_VERSION}.json"
    _atomic_json(summary_path, result)
    return result


def build_assembly_previews(
    output_root: str | Path,
    top_count: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    assembly = assemble_carriers(root)
    selected_candidates = assembly["selected"]["topComponents"][:top_count]
    source_path, _, flake_component, flakes = _load_carrier_catalog(root)
    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    artifact_root = root / f"sheetlet-assemblies-v{ASSEMBLY_PREVIEW_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"summary-top{len(selected_candidates)}.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    screen = json.loads(
        (
            root / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json"
        ).read_text()
    )
    screen_by_component = {
        int(value["componentId"]): value for value in screen["candidates"]
    }
    outputs = []
    started = time.monotonic()
    for rank, candidate in enumerate(selected_candidates, start=1):
        sheetlet_ids = np.asarray(candidate["sheetletComponentIds"], dtype=np.uint32)
        member_indices = np.flatnonzero(np.isin(flake_component, sheetlet_ids))
        member_flakes = [flakes[int(index)] for index in member_indices]
        carrier = _mls_carrier(member_flakes)
        stack, sampling_stats = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        yield_stats = _carrier_yield(carrier["stats"], texture)
        candidate_root = artifact_root / (
            f"rank-{rank:02d}-assembly-{int(candidate['componentId'])}"
        )
        candidate_root.mkdir(parents=True, exist_ok=True)
        geometry_path = candidate_root / "carrier.npz"
        _atomic_npz(
            geometry_path,
            uValues=carrier["uValues"],
            vValues=carrier["vValues"],
            surfaceXYZ=carrier["surfaceXYZ"],
            normalXYZ=carrier["normalXYZ"],
            fiberXYZ=carrier["fiberXYZ"],
            supportMask=carrier["supportMask"],
            memberIndex=member_indices.astype(np.uint32),
        )
        stack_path = candidate_root / "depth-stack.npz"
        _atomic_npz(stack_path, depthOffsets=depth_offsets, intensity=stack)
        mask = carrier["supportMask"]
        center_index = int(np.argmin(np.abs(depth_offsets)))
        best_index = int(
            np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"]))
        )
        selected_depths = [4, 8, 12, 16, 20]
        images = {
            "center.png": _contrast(stack[center_index], mask),
            "best-texture.png": _contrast(stack[best_index], mask),
            "depth-montage.png": np.concatenate(
                [_contrast(stack[index], mask) for index in selected_depths], axis=1
            ),
        }
        for filename, image in images.items():
            (candidate_root / filename).write_bytes(grayscale_png(image))
        input_height = np.asarray(
            [
                screen_by_component[int(component_id)]["carrier"][
                    "medianNodeHeightResidualVoxels"
                ]
                for component_id in sheetlet_ids
            ],
            dtype=np.float32,
        )
        input_normal = np.asarray(
            [
                screen_by_component[int(component_id)]["carrier"][
                    "medianNodeNormalResidualDeg"
                ]
                for component_id in sheetlet_ids
            ],
            dtype=np.float32,
        )
        output = {
            "rank": rank,
            "assembly": candidate,
            "memberCount": len(member_flakes),
            "carrier": carrier["stats"],
            "texture": {
                key: texture[key]
                for key in (
                    "bestDepthOffsetVoxels",
                    "bestTextureScore",
                    "centerTextureScore",
                    "medianTextureScoreAcrossDepth",
                    "depthPeakSharpness",
                    "bestPlane",
                    "centerPlane",
                )
            },
            "yield": yield_stats,
            "joinCost": {
                "inputMedianHeightResidualVoxels": round(
                    float(np.median(input_height)), 3
                ),
                "mergedMedianHeightResidualVoxels": carrier["stats"][
                    "medianNodeHeightResidualVoxels"
                ],
                "inputMedianNormalResidualDeg": round(
                    float(np.median(input_normal)), 3
                ),
                "mergedMedianNormalResidualDeg": carrier["stats"][
                    "medianNodeNormalResidualDeg"
                ],
            },
            "sampling": sampling_stats,
            "artifacts": {
                "geometry": str(geometry_path.relative_to(root)),
                "depthStack": str(stack_path.relative_to(root)),
                "centerImage": str((candidate_root / "center.png").relative_to(root)),
                "bestTextureImage": str(
                    (candidate_root / "best-texture.png").relative_to(root)
                ),
                "depthMontage": str(
                    (candidate_root / "depth-montage.png").relative_to(root)
                ),
            },
        }
        _atomic_json(candidate_root / "summary.json", output)
        outputs.append(output)
    result = {
        "identity": {
            "version": ASSEMBLY_PREVIEW_VERSION,
            "assemblyVersion": assembly["identity"]["version"],
            "source": str(source_path),
        },
        "settings": {
            "topCount": len(selected_candidates),
            "pixelStepVoxels": 2.0,
            "normalDepthRangeVoxels": [-12.0, 12.0],
            "normalDepthStepVoxels": 1.0,
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "candidateCount": len(outputs),
        },
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
