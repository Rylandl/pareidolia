from __future__ import annotations

import hashlib
import itertools
import json
import math
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .slab_branch_association import BRANCH_ASSOCIATION_VERSION
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_sheetlet_carriers import build_mls_carrier


ASSOCIATION_INTEGRITY_VERSION = 1

DEFAULT_SETTINGS: dict[str, Any] = {
    "maximumEvidenceCoreDistanceVoxels": 24.0,
    "maximumSampledClearanceVoxels": 12.0,
    "clearanceSweepVoxels": [2.0, 4.0, 6.0, 8.0, 12.0],
    "triangleBucketVoxels": 8.0,
    "intersectionToleranceVoxels": 0.05,
    "maximumStoredIntersectionPoints": 512,
    "orderZeroToleranceVoxels": 0.5,
    "maximumParallelNormalAngleDeg": 12.0,
    "maximumParallelFiberAngleDeg": 18.0,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _load_flakes(root: Path, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    source_z = np.asarray(arrays["sourceZIndex"], dtype=np.int32)
    source_id = np.asarray(arrays["sourceFlakeId"], dtype=np.int32)
    by_plane = {
        int(z_index): json.loads(
            (
                root
                / f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json"
            ).read_text()
        )["flakes"]
        for z_index in np.unique(source_z)
    }
    output = []
    for z_index, flake_id in zip(source_z, source_id):
        flake = by_plane[int(z_index)][int(flake_id)]
        if int(flake["id"]) != int(flake_id):
            raise ValueError("flake cache IDs are not dense and index aligned")
        output.append(flake)
    return output


def _minimum_point_distance(
    points: np.ndarray, evidence: np.ndarray, batch_size: int = 4096
) -> np.ndarray:
    output = np.full(len(points), np.inf, dtype=np.float32)
    if not len(points) or not len(evidence):
        return output
    for start in range(0, len(points), batch_size):
        stop = min(start + batch_size, len(points))
        delta = points[start:stop, None, :] - evidence[None, :, :]
        output[start:stop] = np.sqrt(
            np.min(np.sum(delta * delta, axis=2), axis=1)
        )
    return output


def _keys_for_bounds(
    low: np.ndarray, high: np.ndarray, bucket_size: float
) -> list[tuple[int, int, int]]:
    first = np.floor(low / bucket_size).astype(np.int32)
    last = np.floor(high / bucket_size).astype(np.int32)
    return [
        (x_index, y_index, z_index)
        for x_index in range(int(first[0]), int(last[0]) + 1)
        for y_index in range(int(first[1]), int(last[1]) + 1)
        for z_index in range(int(first[2]), int(last[2]) + 1)
    ]


def _point_buckets(
    points: np.ndarray, bucket_size: float
) -> dict[tuple[int, int, int], np.ndarray]:
    pending: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for point_index, coordinate in enumerate(
        np.floor(points / bucket_size).astype(np.int32)
    ):
        pending[tuple(int(value) for value in coordinate)].append(point_index)
    return {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in pending.items()
    }


def _triangle_buckets(
    triangle_low: np.ndarray,
    triangle_high: np.ndarray,
    bucket_size: float,
) -> dict[tuple[int, int, int], np.ndarray]:
    pending: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for triangle_index, (low, high) in enumerate(zip(triangle_low, triangle_high)):
        for key in _keys_for_bounds(low, high, bucket_size):
            pending[key].append(triangle_index)
    return {
        key: np.asarray(indices, dtype=np.int32)
        for key, indices in pending.items()
    }


def _surface_geometry(
    association_id: int,
    member_indices: np.ndarray,
    flakes: list[dict[str, Any]],
    cell: np.ndarray,
    oriented_depth: np.ndarray,
    carrier_settings: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    member_flakes = [flakes[int(index)] for index in member_indices]
    evidence = np.asarray(
        [flake["center"] for flake in member_flakes], dtype=np.float32
    )
    carrier = build_mls_carrier(
        member_flakes,
        pixel_step=float(carrier_settings["exactCarrierPixelStepVoxels"]),
        maximum_pixels=int(carrier_settings["exactCarrierMaximumPixelsPerAxis"]),
    )
    surface = np.asarray(carrier["surfaceXYZ"], dtype=np.float32)
    normal = np.asarray(carrier["normalXYZ"], dtype=np.float32)
    fiber = np.asarray(carrier["fiberXYZ"], dtype=np.float32)
    mask = np.asarray(carrier["supportMask"], dtype=bool)
    mask &= (
        np.all(np.isfinite(surface), axis=2)
        & np.all(np.isfinite(normal), axis=2)
        & np.all(np.isfinite(fiber), axis=2)
    )
    points = surface[mask]
    evidence_distance = _minimum_point_distance(points, evidence)
    core_mask = (
        evidence_distance
        <= float(settings["maximumEvidenceCoreDistanceVoxels"])
    )
    core_points = points[core_mask]
    core_normals = normal[mask][core_mask]
    core_fibers = fiber[mask][core_mask]

    valid_quad = (
        mask[:-1, :-1]
        & mask[1:, :-1]
        & mask[1:, 1:]
        & mask[:-1, 1:]
    )
    point00 = surface[:-1, :-1][valid_quad]
    point10 = surface[1:, :-1][valid_quad]
    point11 = surface[1:, 1:][valid_quad]
    point01 = surface[:-1, 1:][valid_quad]
    triangles = np.concatenate(
        (
            np.stack((point00, point10, point11), axis=1),
            np.stack((point00, point11, point01), axis=1),
        ),
        axis=0,
    ).astype(np.float32)
    if len(triangles):
        double_area = np.linalg.norm(
            np.cross(
                triangles[:, 1] - triangles[:, 0],
                triangles[:, 2] - triangles[:, 0],
            ),
            axis=1,
        )
        triangles = triangles[double_area > 1.0e-4]
    triangle_low = np.min(triangles, axis=1) if len(triangles) else np.empty((0, 3))
    triangle_high = np.max(triangles, axis=1) if len(triangles) else np.empty((0, 3))
    maximum_clearance = float(settings["maximumSampledClearanceVoxels"])
    triangle_bucket_size = float(settings["triangleBucketVoxels"])
    cell_depth: dict[tuple[int, int, int], list[float]] = defaultdict(list)
    for member_index in member_indices:
        key = tuple(int(value) for value in cell[int(member_index)])
        cell_depth[key].append(float(oriented_depth[int(member_index)]))
    within_cell_collision_count = sum(
        max(len(values) - 1, 0) for values in cell_depth.values()
    )
    return {
        "associationId": association_id,
        "memberIndices": member_indices,
        "evidence": evidence,
        "points": points,
        "corePoints": core_points,
        "coreNormals": core_normals,
        "coreFibers": core_fibers,
        "pointBuckets": _point_buckets(core_points, maximum_clearance),
        "triangles": triangles,
        "triangleLow": triangle_low.astype(np.float32),
        "triangleHigh": triangle_high.astype(np.float32),
        "triangleBuckets": _triangle_buckets(
            triangle_low, triangle_high, triangle_bucket_size
        ),
        "cellDepth": cell_depth,
        "low": np.min(points, axis=0),
        "high": np.max(points, axis=0),
        "stats": {
            "associationId": association_id,
            "flakeCount": len(member_indices),
            "surfacePointCount": len(points),
            "evidenceCorePointCount": len(core_points),
            "triangleCount": len(triangles),
            "withinAssociationCellCollisionCount": within_cell_collision_count,
            "carrier": carrier["stats"],
        },
    }


def _bbox_distance(first: dict[str, Any], second: dict[str, Any]) -> float:
    gap = np.maximum(
        np.maximum(first["low"] - second["high"], second["low"] - first["high"]),
        0.0,
    )
    return float(np.linalg.norm(gap))


def _nearest_sample_distances(
    source: dict[str, Any],
    target: dict[str, Any],
    maximum_distance: float,
) -> tuple[np.ndarray, np.ndarray]:
    source_points = source["corePoints"]
    target_points = target["corePoints"]
    output = np.full(len(source_points), np.inf, dtype=np.float32)
    nearest = np.full(len(source_points), -1, dtype=np.int32)
    if not len(source_points) or not len(target_points):
        return output, nearest
    source_buckets = source["pointBuckets"]
    target_buckets = target["pointBuckets"]
    for key, source_indices in source_buckets.items():
        neighboring = [
            target_buckets.get(
                (key[0] + dx, key[1] + dy, key[2] + dz)
            )
            for dx, dy, dz in itertools.product((-1, 0, 1), repeat=3)
        ]
        neighboring = [values for values in neighboring if values is not None]
        if not neighboring:
            continue
        target_indices = np.concatenate(neighboring)
        delta = (
            source_points[source_indices, None, :]
            - target_points[target_indices][None, :, :]
        )
        distance2 = np.sum(delta * delta, axis=2)
        local_nearest = np.argmin(distance2, axis=1)
        distance = np.sqrt(distance2[np.arange(len(source_indices)), local_nearest])
        valid = distance <= maximum_distance
        output[source_indices[valid]] = distance[valid]
        nearest[source_indices[valid]] = target_indices[local_nearest[valid]]
    return output, nearest


def _axial_angle_deg(first: np.ndarray, second: np.ndarray) -> float:
    return float(
        np.degrees(
            np.arccos(
                np.clip(abs(float(np.dot(first, second))), 0.0, 1.0)
            )
        )
    )


def _closest_sample_geometry(
    first: dict[str, Any],
    second: dict[str, Any],
    first_distance: np.ndarray,
    first_nearest: np.ndarray,
    second_distance: np.ndarray,
    second_nearest: np.ndarray,
) -> dict[str, Any]:
    first_minimum = float(np.min(first_distance, initial=np.inf))
    second_minimum = float(np.min(second_distance, initial=np.inf))
    if first_minimum <= second_minimum:
        first_index = int(np.argmin(first_distance))
        second_index = int(first_nearest[first_index])
    else:
        second_index = int(np.argmin(second_distance))
        first_index = int(second_nearest[second_index])
    first_point = first["corePoints"][first_index]
    second_point = second["corePoints"][second_index]
    first_normal = first["coreNormals"][first_index]
    second_normal = second["coreNormals"][second_index].copy()
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal *= -1.0
    mean_normal = first_normal + second_normal
    mean_normal /= max(float(np.linalg.norm(mean_normal)), 1.0e-8)
    delta = second_point - first_point
    normal_separation = abs(float(np.dot(delta, mean_normal)))
    tangent_separation = math.sqrt(
        max(float(np.dot(delta, delta)) - normal_separation**2, 0.0)
    )
    return {
        "closestSourcePointXYZ": first_point.astype(float).tolist(),
        "closestTargetPointXYZ": second_point.astype(float).tolist(),
        "closestNormalAngleDeg": _axial_angle_deg(first_normal, second_normal),
        "closestFiberAngleDeg": _axial_angle_deg(
            first["coreFibers"][first_index], second["coreFibers"][second_index]
        ),
        "closestNormalSeparationVoxels": normal_separation,
        "closestTangentSeparationVoxels": tangent_separation,
    }


def _shared_cell_order(
    first: dict[str, Any],
    second: dict[str, Any],
    zero_tolerance: float,
) -> dict[str, Any]:
    shared = sorted(set(first["cellDepth"]) & set(second["cellDepth"]))
    gaps = np.asarray(
        [
            float(np.median(second["cellDepth"][key]))
            - float(np.median(first["cellDepth"][key]))
            for key in shared
        ],
        dtype=np.float32,
    )
    negative = int(np.count_nonzero(gaps < -zero_tolerance))
    positive = int(np.count_nonzero(gaps > zero_tolerance))
    tied = len(gaps) - negative - positive
    return {
        "sharedCellCount": len(shared),
        "negativeOrderCount": negative,
        "positiveOrderCount": positive,
        "nearTieOrderCount": tied,
        "orderInversion": bool(negative and positive),
        "medianAbsoluteSharedCellDepthGapVoxels": (
            float(np.median(np.abs(gaps))) if len(gaps) else math.nan
        ),
        "minimumAbsoluteSharedCellDepthGapVoxels": (
            float(np.min(np.abs(gaps))) if len(gaps) else math.nan
        ),
    }


def _pair_classification(
    pair: dict[str, Any], settings: dict[str, Any]
) -> str:
    if int(pair["evidenceCoreIntersectingTrianglePairCount"]):
        return "evidence-core-intersection"
    if int(pair["intersectingTrianglePairCount"]):
        return "support-intersection"
    if bool(pair["orderInversion"]):
        return "shared-cell-order-inversion"
    if int(pair["sharedCellCount"]):
        return "ordered-near-contact"
    if (
        float(pair["closestNormalAngleDeg"])
        <= float(settings["maximumParallelNormalAngleDeg"])
        and float(pair["closestFiberAngleDeg"])
        <= float(settings["maximumParallelFiberAngleDeg"])
    ):
        return "parallel-boundary-approach"
    return "spatial-approach"


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _point_in_triangle_2d(
    point: np.ndarray, triangle: np.ndarray, tolerance: float
) -> bool:
    signs = [
        _cross_2d(
            triangle[(index + 1) % 3] - triangle[index],
            point - triangle[index],
        )
        for index in range(3)
    ]
    return min(signs) >= -tolerance or max(signs) <= tolerance


def _segments_intersect_2d(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
    tolerance: float,
) -> bool:
    if np.any(
        np.maximum(
            np.minimum(first_start, first_stop),
            np.minimum(second_start, second_stop),
        )
        > np.minimum(
            np.maximum(first_start, first_stop),
            np.maximum(second_start, second_stop),
        )
        + tolerance
    ):
        return False
    first_direction = first_stop - first_start
    second_direction = second_stop - second_start
    first_side = _cross_2d(first_direction, second_start - first_start)
    second_side = _cross_2d(first_direction, second_stop - first_start)
    third_side = _cross_2d(second_direction, first_start - second_start)
    fourth_side = _cross_2d(second_direction, first_stop - second_start)
    return (
        min(first_side, second_side) <= tolerance
        and max(first_side, second_side) >= -tolerance
        and min(third_side, fourth_side) <= tolerance
        and max(third_side, fourth_side) >= -tolerance
    )


def _coplanar_triangles_overlap(
    first: np.ndarray,
    second: np.ndarray,
    normal: np.ndarray,
    tolerance: float,
) -> bool:
    drop_axis = int(np.argmax(np.abs(normal)))
    first_2d = np.delete(first, drop_axis, axis=1)
    second_2d = np.delete(second, drop_axis, axis=1)
    if any(
        _point_in_triangle_2d(point, second_2d, tolerance)
        for point in first_2d
    ) or any(
        _point_in_triangle_2d(point, first_2d, tolerance)
        for point in second_2d
    ):
        return True
    return any(
        _segments_intersect_2d(
            first_2d[first_index],
            first_2d[(first_index + 1) % 3],
            second_2d[second_index],
            second_2d[(second_index + 1) % 3],
            tolerance,
        )
        for first_index in range(3)
        for second_index in range(3)
    )


def _segment_triangle_intersection(
    start: np.ndarray,
    stop: np.ndarray,
    triangle: np.ndarray,
) -> np.ndarray | None:
    coordinate_tolerance = 1.0e-7
    direction = stop - start
    first_edge = triangle[1] - triangle[0]
    second_edge = triangle[2] - triangle[0]
    cross = np.cross(direction, second_edge)
    determinant = float(np.dot(first_edge, cross))
    scale = max(
        float(np.linalg.norm(direction))
        * float(np.linalg.norm(first_edge))
        * float(np.linalg.norm(second_edge)),
        1.0,
    )
    if abs(determinant) <= 1.0e-10 * scale:
        return None
    inverse = 1.0 / determinant
    relative = start - triangle[0]
    first_coordinate = float(np.dot(relative, cross)) * inverse
    if (
        first_coordinate < -coordinate_tolerance
        or first_coordinate > 1.0 + coordinate_tolerance
    ):
        return None
    cross_relative = np.cross(relative, first_edge)
    second_coordinate = float(np.dot(direction, cross_relative)) * inverse
    if (
        second_coordinate < -coordinate_tolerance
        or first_coordinate + second_coordinate > 1.0 + coordinate_tolerance
    ):
        return None
    segment_coordinate = float(np.dot(second_edge, cross_relative)) * inverse
    if (
        segment_coordinate < -coordinate_tolerance
        or segment_coordinate > 1.0 + coordinate_tolerance
    ):
        return None
    return start + np.clip(segment_coordinate, 0.0, 1.0) * direction


def _triangle_intersection(
    first: np.ndarray,
    second: np.ndarray,
    tolerance: float,
) -> tuple[np.ndarray | None, bool]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    first_normal = np.cross(first[1] - first[0], first[2] - first[0])
    second_normal = np.cross(second[1] - second[0], second[2] - second[0])
    first_length = float(np.linalg.norm(first_normal))
    second_length = float(np.linalg.norm(second_normal))
    if first_length <= 1.0e-10 or second_length <= 1.0e-10:
        return None, False
    normalized_first = first_normal / first_length
    normalized_second = second_normal / second_length
    parallel = float(np.linalg.norm(np.cross(normalized_first, normalized_second)))
    if parallel <= 1.0e-6:
        plane_distance = np.abs((second - first[0]) @ normalized_first)
        if float(np.max(plane_distance)) > tolerance:
            return None, False
        if _coplanar_triangles_overlap(
            first, second, normalized_first, tolerance
        ):
            return (np.mean(first, axis=0) + np.mean(second, axis=0)) * 0.5, True
        return None, False

    for triangle, target in ((first, second), (second, first)):
        for edge_index in range(3):
            point = _segment_triangle_intersection(
                triangle[edge_index],
                triangle[(edge_index + 1) % 3],
                target,
            )
            if point is not None:
                return point, False
    return None, False


def _mesh_intersections(
    first: dict[str, Any],
    second: dict[str, Any],
    settings: dict[str, Any],
) -> dict[str, Any]:
    if len(first["triangles"]) > len(second["triangles"]):
        source, target = second, first
    else:
        source, target = first, second
    tolerance = float(settings["intersectionToleranceVoxels"])
    bucket_size = float(settings["triangleBucketVoxels"])
    maximum_stored = int(settings["maximumStoredIntersectionPoints"])
    broad_count = 0
    narrow_count = 0
    intersection_count = 0
    core_intersection_count = 0
    coplanar_count = 0
    stored = []
    for source_index, source_triangle in enumerate(source["triangles"]):
        keys = _keys_for_bounds(
            source["triangleLow"][source_index] - tolerance,
            source["triangleHigh"][source_index] + tolerance,
            bucket_size,
        )
        candidates = {
            int(target_index)
            for key in keys
            for target_index in target["triangleBuckets"].get(key, ())
        }
        broad_count += len(candidates)
        for target_index in candidates:
            if np.any(
                source["triangleHigh"][source_index] + tolerance
                < target["triangleLow"][target_index]
            ) or np.any(
                target["triangleHigh"][target_index] + tolerance
                < source["triangleLow"][source_index]
            ):
                continue
            narrow_count += 1
            point, coplanar = _triangle_intersection(
                source_triangle,
                target["triangles"][target_index],
                tolerance,
            )
            if point is None:
                continue
            intersection_count += 1
            coplanar_count += int(coplanar)
            source_distance = float(
                np.min(np.linalg.norm(source["evidence"] - point, axis=1))
            )
            target_distance = float(
                np.min(np.linalg.norm(target["evidence"] - point, axis=1))
            )
            core = (
                source_distance
                <= float(settings["maximumEvidenceCoreDistanceVoxels"])
                and target_distance
                <= float(settings["maximumEvidenceCoreDistanceVoxels"])
            )
            core_intersection_count += int(core)
            if len(stored) < maximum_stored:
                stored.append(
                    {
                        "point": np.asarray(point, dtype=np.float32),
                        "evidenceCore": core,
                        "coplanar": coplanar,
                    }
                )
    return {
        "broadPhaseTrianglePairCount": broad_count,
        "narrowPhaseTrianglePairCount": narrow_count,
        "intersectingTrianglePairCount": intersection_count,
        "evidenceCoreIntersectingTrianglePairCount": core_intersection_count,
        "coplanarIntersectingTrianglePairCount": coplanar_count,
        "stored": stored,
    }


def association_integrity_audit(
    output_root: str | Path,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    window_origin_cell_xyz: tuple[int, int, int] | list[int] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    suffix = window_artifact_suffix(window_origin_cell_xyz)
    branch_stem = f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}"
    monotone_stem = f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}"
    branch_summary_path = root / f"{branch_stem}.json"
    branch_artifact_path = root / f"{branch_stem}.npz"
    monotone_path = root / f"{monotone_stem}.npz"
    branch_summary = json.loads(branch_summary_path.read_text())
    with np.load(monotone_path) as payload:
        monotone = {key: np.asarray(payload[key]) for key in payload.files}
    z_indices = np.unique(monotone["sourceZIndex"].astype(np.int32))
    input_paths = [branch_summary_path, branch_artifact_path, monotone_path]
    input_paths.extend(
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json"
        for z_index in z_indices
    )
    identity: dict[str, Any] = {
        "version": ASSOCIATION_INTEGRITY_VERSION,
        "branchAssociationIdentity": branch_summary["identity"],
        "settings": resolved,
        "inputArtifacts": [_content_identity(path) for path in input_paths],
    }
    if window_origin_cell_xyz is not None:
        identity["windowOriginCellXYZ"] = [
            int(value) for value in window_origin_cell_xyz
        ]
    stem = f"branch-association-integrity-v{ASSOCIATION_INTEGRITY_VERSION}{suffix}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    flakes = _load_flakes(root, monotone)
    branch = monotone["component"].astype(np.int32)
    cell = monotone["cellIndex"].astype(np.int32)
    oriented_depth = monotone["orientedDepth"].astype(np.float32)
    with np.load(branch_artifact_path) as payload:
        branch_association = np.asarray(payload["branchAssociation"], dtype=np.int32)
    association_branch_count = np.bincount(branch_association)
    association_ids = np.flatnonzero(association_branch_count >= 2)
    flake_association = branch_association[branch]
    geometries = []
    for association_id in association_ids:
        member_indices = np.flatnonzero(flake_association == association_id)
        geometries.append(
            _surface_geometry(
                int(association_id),
                member_indices,
                flakes,
                cell,
                oriented_depth,
                branch_summary["settings"],
                resolved,
            )
        )

    maximum_clearance = float(resolved["maximumSampledClearanceVoxels"])
    clearance_sweep = np.asarray(resolved["clearanceSweepVoxels"], dtype=np.float32)
    if len(clearance_sweep) and float(np.max(clearance_sweep)) > maximum_clearance:
        raise ValueError("clearanceSweepVoxels exceeds maximumSampledClearanceVoxels")
    pairs = []
    stored_intersections = []
    total_broad = 0
    total_narrow = 0
    total_intersections = 0
    total_core_intersections = 0
    total_coplanar = 0
    for first_index, first in enumerate(geometries):
        for second in geometries[first_index + 1 :]:
            bbox_distance = _bbox_distance(first, second)
            if bbox_distance > maximum_clearance:
                continue
            first_distance, first_nearest = _nearest_sample_distances(
                first, second, maximum_clearance
            )
            second_distance, second_nearest = _nearest_sample_distances(
                second, first, maximum_clearance
            )
            finite = np.r_[
                first_distance[np.isfinite(first_distance)],
                second_distance[np.isfinite(second_distance)],
            ]
            minimum_distance = float(np.min(finite)) if len(finite) else math.inf
            intersections = _mesh_intersections(first, second, resolved)
            total_broad += intersections["broadPhaseTrianglePairCount"]
            total_narrow += intersections["narrowPhaseTrianglePairCount"]
            total_intersections += intersections["intersectingTrianglePairCount"]
            total_core_intersections += intersections[
                "evidenceCoreIntersectingTrianglePairCount"
            ]
            total_coplanar += intersections[
                "coplanarIntersectingTrianglePairCount"
            ]
            if not len(finite) and not intersections["intersectingTrianglePairCount"]:
                continue
            first_counts = np.asarray(
                [np.count_nonzero(first_distance <= value) for value in clearance_sweep],
                dtype=np.uint32,
            )
            second_counts = np.asarray(
                [np.count_nonzero(second_distance <= value) for value in clearance_sweep],
                dtype=np.uint32,
            )
            pair_index = len(pairs)
            closest = _closest_sample_geometry(
                first,
                second,
                first_distance,
                first_nearest,
                second_distance,
                second_nearest,
            )
            shared_order = _shared_cell_order(
                first,
                second,
                float(resolved["orderZeroToleranceVoxels"]),
            )
            pair = {
                "associationSource": int(first["associationId"]),
                "associationTarget": int(second["associationId"]),
                "bboxDistanceVoxels": bbox_distance,
                "minimumSampledCoreDistanceVoxels": minimum_distance,
                "sourceCorePointCount": len(first["corePoints"]),
                "targetCorePointCount": len(second["corePoints"]),
                "sourceCounts": first_counts,
                "targetCounts": second_counts,
                **closest,
                **shared_order,
                **{
                    key: intersections[key]
                    for key in (
                        "broadPhaseTrianglePairCount",
                        "narrowPhaseTrianglePairCount",
                        "intersectingTrianglePairCount",
                        "evidenceCoreIntersectingTrianglePairCount",
                        "coplanarIntersectingTrianglePairCount",
                    )
                },
            }
            pair["classification"] = _pair_classification(pair, resolved)
            pairs.append(pair)
            for value in intersections["stored"]:
                stored_intersections.append(
                    {
                        "pairIndex": pair_index,
                        **value,
                    }
                )

    source_association = np.asarray(
        [value["associationSource"] for value in pairs], dtype=np.uint32
    )
    target_association = np.asarray(
        [value["associationTarget"] for value in pairs], dtype=np.uint32
    )
    minimum_distance = np.asarray(
        [value["minimumSampledCoreDistanceVoxels"] for value in pairs],
        dtype=np.float32,
    )
    source_counts = (
        np.stack([value["sourceCounts"] for value in pairs])
        if pairs
        else np.empty((0, len(clearance_sweep)), dtype=np.uint32)
    )
    target_counts = (
        np.stack([value["targetCounts"] for value in pairs])
        if pairs
        else np.empty((0, len(clearance_sweep)), dtype=np.uint32)
    )
    intersection_count = np.asarray(
        [value["intersectingTrianglePairCount"] for value in pairs],
        dtype=np.uint32,
    )
    core_intersection_count = np.asarray(
        [value["evidenceCoreIntersectingTrianglePairCount"] for value in pairs],
        dtype=np.uint32,
    )
    closest_source_point = np.asarray(
        [value["closestSourcePointXYZ"] for value in pairs], dtype=np.float32
    ).reshape(-1, 3)
    closest_target_point = np.asarray(
        [value["closestTargetPointXYZ"] for value in pairs], dtype=np.float32
    ).reshape(-1, 3)
    _atomic_npz(
        artifact_path,
        associationSource=source_association,
        associationTarget=target_association,
        minimumSampledCoreDistanceVoxels=minimum_distance,
        clearanceSweepVoxels=clearance_sweep,
        sourceCorePointCountWithinClearance=source_counts,
        targetCorePointCountWithinClearance=target_counts,
        intersectingTrianglePairCount=intersection_count,
        evidenceCoreIntersectingTrianglePairCount=core_intersection_count,
        closestSourcePointXYZ=closest_source_point,
        closestTargetPointXYZ=closest_target_point,
        closestNormalAngleDeg=np.asarray(
            [value["closestNormalAngleDeg"] for value in pairs], dtype=np.float32
        ),
        closestFiberAngleDeg=np.asarray(
            [value["closestFiberAngleDeg"] for value in pairs], dtype=np.float32
        ),
        closestNormalSeparationVoxels=np.asarray(
            [value["closestNormalSeparationVoxels"] for value in pairs],
            dtype=np.float32,
        ),
        closestTangentSeparationVoxels=np.asarray(
            [value["closestTangentSeparationVoxels"] for value in pairs],
            dtype=np.float32,
        ),
        sharedCellCount=np.asarray(
            [value["sharedCellCount"] for value in pairs], dtype=np.uint32
        ),
        sharedCellOrderInversion=np.asarray(
            [value["orderInversion"] for value in pairs], dtype=bool
        ),
        medianAbsoluteSharedCellDepthGapVoxels=np.asarray(
            [
                value["medianAbsoluteSharedCellDepthGapVoxels"]
                for value in pairs
            ],
            dtype=np.float32,
        ),
        intersectionPairIndex=np.asarray(
            [value["pairIndex"] for value in stored_intersections], dtype=np.uint32
        ),
        intersectionPointXYZ=np.asarray(
            [value["point"] for value in stored_intersections], dtype=np.float32
        ).reshape(-1, 3),
        intersectionEvidenceCore=np.asarray(
            [value["evidenceCore"] for value in stored_intersections], dtype=bool
        ),
        intersectionCoplanar=np.asarray(
            [value["coplanar"] for value in stored_intersections], dtype=bool
        ),
    )
    ranked_pairs = sorted(
        pairs,
        key=lambda value: (
            -int(value["evidenceCoreIntersectingTrianglePairCount"]),
            -int(value["intersectingTrianglePairCount"]),
            float(value["minimumSampledCoreDistanceVoxels"]),
        ),
    )
    top_pairs = []
    for value in ranked_pairs[:24]:
        top_pairs.append(
            {
                key: value[key]
                for key in (
                    "associationSource",
                    "associationTarget",
                    "bboxDistanceVoxels",
                    "minimumSampledCoreDistanceVoxels",
                    "sourceCorePointCount",
                    "targetCorePointCount",
                    "intersectingTrianglePairCount",
                    "evidenceCoreIntersectingTrianglePairCount",
                    "coplanarIntersectingTrianglePairCount",
                    "closestSourcePointXYZ",
                    "closestTargetPointXYZ",
                    "closestNormalAngleDeg",
                    "closestFiberAngleDeg",
                    "closestNormalSeparationVoxels",
                    "closestTangentSeparationVoxels",
                    "sharedCellCount",
                    "negativeOrderCount",
                    "positiveOrderCount",
                    "nearTieOrderCount",
                    "orderInversion",
                    "medianAbsoluteSharedCellDepthGapVoxels",
                    "minimumAbsoluteSharedCellDepthGapVoxels",
                    "classification",
                )
            }
        )
    result = {
        "identity": identity,
        "contract": {
            "scope": (
                "all exact-coherent merged associations in one branch-association "
                "window; unassociated branches are outside this audit"
            ),
            "intersectionMeaning": (
                "triangle intersections are violations in the piecewise-linear MLS "
                "surfaces; evidence-core intersections lie within the declared "
                "distance of supporting flakes on both surfaces"
            ),
            "clearanceMeaning": (
                "bidirectional nearest sampled evidence-core points at the carrier "
                "grid spacing; the clearance sweep is descriptive and is not called "
                "physical papyrus thickness"
            ),
            "response": (
                "violations queue the involved local hypotheses for inspection or "
                "re-analysis; they do not silently merge, relabel, or delete evidence"
            ),
        },
        "window": branch_summary["window"],
        "settings": resolved,
        "carriers": [value["stats"] for value in geometries],
        "topSpatialPairs": top_pairs,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "associationCarrierCount": len(geometries),
            "surfacePointCount": sum(len(value["points"]) for value in geometries),
            "evidenceCorePointCount": sum(
                len(value["corePoints"]) for value in geometries
            ),
            "triangleCount": sum(len(value["triangles"]) for value in geometries),
            "spatialAssociationPairCount": len(pairs),
            "associationPairCountWithinClearance": {
                str(float(clearance)): int(
                    np.count_nonzero(minimum_distance <= clearance)
                )
                for clearance in clearance_sweep
            },
            "associationPairWithMeshIntersectionCount": int(
                np.count_nonzero(intersection_count)
            ),
            "associationPairWithEvidenceCoreIntersectionCount": int(
                np.count_nonzero(core_intersection_count)
            ),
            "spatialPairClassificationCounts": {
                classification: sum(
                    value["classification"] == classification for value in pairs
                )
                for classification in sorted(
                    {value["classification"] for value in pairs}
                )
            },
            "withinAssociationCellCollisionCount": sum(
                value["stats"]["withinAssociationCellCollisionCount"]
                for value in geometries
            ),
            "broadPhaseTrianglePairCount": total_broad,
            "narrowPhaseTrianglePairCount": total_narrow,
            "intersectingTrianglePairCount": total_intersections,
            "evidenceCoreIntersectingTrianglePairCount": total_core_intersections,
            "coplanarIntersectingTrianglePairCount": total_coplanar,
            "storedIntersectionPointCount": len(stored_intersections),
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
