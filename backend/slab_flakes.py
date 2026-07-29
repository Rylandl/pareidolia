from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .acus import _minimal_rotation, _plane_basis
from .slab_normal_families import (
    NORMAL_FAMILY_VERSION,
    _catalog_records_for_cell,
    _normal_family_partitions,
    load_normal_families,
)


FLAKE_CACHE_VERSION = 5


def _atomic_compact_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values):
        return 0.0
    order = np.argsort(values)
    ordered_values = np.asarray(values[order], dtype=np.float64)
    ordered_weights = np.maximum(np.asarray(weights[order], dtype=np.float64), 0.0)
    cumulative = np.cumsum(ordered_weights)
    if cumulative[-1] <= 1.0e-10:
        return float(np.median(ordered_values))
    target = float(np.clip(quantile, 0.0, 1.0)) * cumulative[-1]
    return float(ordered_values[min(int(np.searchsorted(cumulative, target)), len(values) - 1)])


def _axial_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
    )


def _angular_distance_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    distance = np.abs(first - second)
    return np.minimum(distance, 180.0 - distance)


def _fit_cell_flakes(
    records: np.ndarray,
    record_ids: np.ndarray,
    cell_center: np.ndarray,
    normal: np.ndarray,
    normal_confidence: float,
    cell_index: tuple[int, int, int],
    cube_size: int,
    maximum_flakes: int,
    depth_bandwidth: float,
    angle_bandwidth: float,
    minimum_needles: int,
    normal_family: int = 0,
    family_coverage: float = 1.0,
    family_ambiguous_fraction: float = 0.0,
    family_component_id: int | None = None,
    family_component_size: int = 0,
) -> list[dict[str, Any]]:
    if len(records) < minimum_needles:
        return []
    normal = np.array(normal, dtype=np.float32, copy=True)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    directions = np.asarray(records["direction"], dtype=np.float32)
    plane_component = directions @ normal
    residual_degrees = np.degrees(
        np.arcsin(np.clip(np.abs(plane_component), 0.0, 1.0))
    )
    median_residual = float(np.median(residual_degrees))
    inlier_limit = max(8.0, min(22.0, median_residual * 2.8))
    projected = directions - plane_component[:, None] * normal[None, :]
    projected_lengths = np.linalg.norm(projected, axis=1)
    usable = (residual_degrees <= inlier_limit) & (projected_lengths >= 0.35)
    if int(np.count_nonzero(usable)) < minimum_needles:
        return []
    projected = projected[usable] / projected_lengths[usable, None]
    usable_records = records[usable]
    usable_ids = record_ids[usable]
    residual_degrees = residual_degrees[usable]
    offsets = np.asarray(usable_records["center"], dtype=np.float32) - cell_center
    depths = offsets @ normal
    u_axis, v_axis = _plane_basis(normal)
    angles = np.degrees(
        np.arctan2(projected @ v_axis, projected @ u_axis)
    ) % 180.0
    weights = (
        np.asarray(usable_records["score"], dtype=np.float32)
        * np.sqrt(
            np.maximum(np.asarray(usable_records["axialCoverage"], dtype=np.float32), 0.05)
            * np.maximum(np.asarray(usable_records["supportScore"], dtype=np.float32), 0.05)
        )
        * np.exp(-0.5 * (residual_degrees / 14.0) ** 2)
    )
    if float(np.sum(weights)) <= 1.0e-8:
        return []

    depth_delta = (depths[:, None] - depths[None, :]) / depth_bandwidth
    angle_delta = _angular_distance_degrees(angles[:, None], angles[None, :])
    angle_delta /= angle_bandwidth
    pair_distance2 = depth_delta**2 + angle_delta**2
    density = np.exp(-0.5 * pair_distance2) @ weights
    order = np.argsort(density)[::-1]
    selected: list[int] = []
    peak_limit = float(density[order[0]]) * 0.16
    for candidate in order:
        if len(selected) >= maximum_flakes or float(density[candidate]) < peak_limit:
            break
        separated = True
        for previous in selected:
            depth_separation = abs(float(depths[candidate] - depths[previous])) / (
                2.0 * depth_bandwidth
            )
            angle_separation = float(
                _angular_distance_degrees(
                    np.asarray(angles[candidate]), np.asarray(angles[previous])
                )
            ) / (2.0 * angle_bandwidth)
            if depth_separation**2 + angle_separation**2 < 1.0:
                separated = False
                break
        if separated:
            selected.append(int(candidate))
    if not selected:
        return []

    selected_depths = depths[selected]
    selected_angles = angles[selected]
    assignment_distance2 = (
        (depths[:, None] - selected_depths[None, :]) / depth_bandwidth
    ) ** 2
    assignment_distance2 += (
        _angular_distance_degrees(angles[:, None], selected_angles[None, :])
        / angle_bandwidth
    ) ** 2
    assignment = np.argmin(assignment_distance2, axis=1)
    minimum_distance2 = np.min(assignment_distance2, axis=1)
    total_weight = max(float(np.sum(weights)), 1.0e-8)
    flakes: list[dict[str, Any]] = []
    for mode_index in range(len(selected)):
        member_mask = (assignment == mode_index) & (minimum_distance2 <= 2.5**2)
        member_count = int(np.count_nonzero(member_mask))
        if member_count < minimum_needles:
            continue
        member_distance2 = assignment_distance2[member_mask, mode_index]
        member_weights = weights[member_mask] * np.exp(-0.5 * member_distance2)
        weight_sum = float(np.sum(member_weights))
        effective_support = weight_sum**2 / max(
            float(np.sum(member_weights**2)), 1.0e-8
        )
        if effective_support < max(3.0, minimum_needles * 0.65):
            continue
        member_depths = depths[member_mask]
        depth = _weighted_quantile(member_depths, member_weights, 0.5)
        member_projected = projected[member_mask]
        axial_matrix = np.einsum(
            "n,ni,nj->ij", member_weights, member_projected, member_projected
        )
        eigenvalues, eigenvectors = np.linalg.eigh(axial_matrix)
        fiber = np.asarray(eigenvectors[:, -1], dtype=np.float32)
        fiber -= normal * float(np.dot(fiber, normal))
        fiber /= max(float(np.linalg.norm(fiber)), 1.0e-8)
        dominant = int(np.argmax(np.abs(fiber)))
        if fiber[dominant] < 0.0:
            fiber = -fiber
        cross_fiber = np.cross(normal, fiber)
        cross_fiber /= max(float(np.linalg.norm(cross_fiber)), 1.0e-8)
        member_centers = np.asarray(usable_records["center"][member_mask], dtype=np.float32)
        weighted_center = np.average(member_centers, axis=0, weights=member_weights)
        center_depth = float(np.dot(weighted_center - cell_center, normal))
        center = weighted_center + normal * (depth - center_depth)
        centered = member_centers - center
        radius_fiber = _weighted_quantile(
            np.abs(centered @ fiber), member_weights, 0.82
        )
        radius_cross = _weighted_quantile(
            np.abs(centered @ cross_fiber), member_weights, 0.82
        )
        radius_limit = cube_size * 0.5
        radius_fiber = float(np.clip(radius_fiber, 3.0, radius_limit))
        radius_cross = float(np.clip(radius_cross, 3.0, radius_limit))
        thickness = 1.4826 * _weighted_quantile(
            np.abs(member_depths - depth), member_weights, 0.5
        )
        member_angles = angles[member_mask]
        circular = np.sum(
            member_weights * np.exp(2.0j * np.radians(member_angles))
        )
        concentration = float(np.clip(abs(circular) / max(weight_sum, 1.0e-8), 0.0, 1.0))
        angular_residual = np.degrees(
            np.arccos(
                np.clip(np.abs(member_projected @ fiber), 0.0, 1.0)
            )
        )
        median_fiber_residual = _weighted_quantile(
            angular_residual, member_weights, 0.5
        )
        support_fraction = float(np.clip(weight_sum / total_weight, 0.0, 1.0))
        thickness_score = 1.0 / (1.0 + (thickness / depth_bandwidth) ** 2)
        quality = float(
            np.clip(
                math.sqrt(max(concentration, 0.0) * max(normal_confidence, 0.0))
                * min(1.0, effective_support / 10.0)
                * (0.55 + 0.45 * support_fraction)
                * math.sqrt(thickness_score),
                0.0,
                1.0,
            )
        )
        if concentration < 0.22 or quality < 0.035:
            continue
        fiber_xy = math.hypot(float(fiber[0]), float(fiber[1]))
        fiber_angle_xy = (
            math.degrees(math.atan2(float(fiber[1]), float(fiber[0]))) % 180.0
            if fiber_xy >= 0.05
            else None
        )
        flakes.append(
            {
                "cellIndex": list(cell_index),
                "cellCenter": np.round(cell_center, 3).tolist(),
                "normalFamily": int(normal_family),
                "familyCoverage": round(float(family_coverage), 4),
                "familyAmbiguousFraction": round(
                    float(family_ambiguous_fraction), 4
                ),
                "familyComponentId": family_component_id,
                "familyComponentSize": int(family_component_size),
                "center": np.round(center, 3).tolist(),
                "normal": np.round(normal, 6).tolist(),
                "fiber": np.round(fiber, 6).tolist(),
                "crossFiber": np.round(cross_fiber, 6).tolist(),
                "depthOffset": round(float(depth), 3),
                "planeOffset": round(float(np.dot(normal, center)), 3),
                "radiusFiber": round(radius_fiber, 3),
                "radiusCrossFiber": round(radius_cross, 3),
                "thickness": round(float(thickness), 3),
                "needleCount": member_count,
                "effectiveSupport": round(float(effective_support), 3),
                "supportFraction": round(support_fraction, 4),
                "fiberConcentration": round(concentration, 4),
                "medianFiberResidualDeg": round(float(median_fiber_residual), 3),
                "quality": round(quality, 4),
                "fiberAngleXYDeg": round(float(fiber_angle_xy), 3)
                if fiber_angle_xy is not None
                else None,
                "_needleIds": set(int(value) for value in usable_ids[member_mask]),
            }
        )
    flakes.sort(key=lambda item: (-float(item["quality"]), float(item["depthOffset"])))
    return flakes


def _flake_pair_metrics(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    first_normal = np.asarray(first["normal"], dtype=np.float32)
    second_normal = np.asarray(second["normal"], dtype=np.float32)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
    normal_angle = _axial_angle_degrees(first_normal, second_normal)
    rotation = _minimal_rotation(first_normal, second_normal)
    transported_fiber = rotation @ np.asarray(first["fiber"], dtype=np.float32)
    fiber_angle = _axial_angle_degrees(
        transported_fiber, np.asarray(second["fiber"], dtype=np.float32)
    )
    delta = np.asarray(second["center"], dtype=np.float32) - np.asarray(
        first["center"], dtype=np.float32
    )
    position_residual = 0.5 * (
        abs(float(np.dot(first_normal, delta)))
        + abs(float(np.dot(second_normal, delta)))
    )
    first_ids = first.get("_needleIds", set())
    second_ids = second.get("_needleIds", set())
    shared_count = len(first_ids.intersection(second_ids))
    shared_fraction = shared_count / max(1, min(len(first_ids), len(second_ids)))
    quality = math.sqrt(float(first["quality"]) * float(second["quality"]))
    compatibility = quality * math.exp(
        -0.5
        * (
            (position_residual / 7.0) ** 2
            + (normal_angle / 12.0) ** 2
            + (fiber_angle / 18.0) ** 2
        )
    )
    independent_score = compatibility * (1.0 - 0.65 * shared_fraction)
    return {
        "positionResidual": position_residual,
        "normalAngle": normal_angle,
        "fiberAngle": fiber_angle,
        "sharedFraction": shared_fraction,
        "compatibility": compatibility,
        "independentScore": independent_score,
    }


def _match_flakes_at_step(
    flakes: list[dict[str, Any]],
    cell_step: int = 1,
) -> list[dict[str, Any]]:
    cell_step = max(1, int(cell_step))
    by_cell: dict[tuple[int, int], list[int]] = {}
    for index, flake in enumerate(flakes):
        cell_x, cell_y, _ = flake["cellIndex"]
        by_cell.setdefault((int(cell_x), int(cell_y)), []).append(index)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    group_index = 0
    for cell_key, first_indices in by_cell.items():
        for offset in ((cell_step, 0), (0, cell_step)):
            neighbor_key = (cell_key[0] + offset[0], cell_key[1] + offset[1])
            second_indices = by_cell.get(neighbor_key)
            if not second_indices:
                continue
            first_array = np.asarray(first_indices, dtype=np.int32)
            second_array = np.asarray(second_indices, dtype=np.int32)
            source_parts.append(np.repeat(first_array, len(second_array)))
            target_parts.append(np.tile(second_array, len(first_array)))
            group_parts.append(
                np.full(len(first_array) * len(second_array), group_index, dtype=np.int32)
            )
            group_index += 1
    if not source_parts:
        return []
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    groups = np.concatenate(group_parts)
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    qualities = np.asarray([flake["quality"] for flake in flakes], dtype=np.float32)
    first_normals = normals[sources]
    second_normals = normals[targets].copy()
    signed_normal_dot = np.sum(first_normals * second_normals, axis=1)
    second_normals[signed_normal_dot < 0.0] *= -1.0
    normal_dot = np.clip(
        np.abs(np.sum(first_normals * second_normals, axis=1)), 0.0, 1.0
    )
    normal_angles = np.degrees(np.arccos(normal_dot))
    normal_cross = np.cross(first_normals, second_normals)
    sine2 = np.sum(normal_cross**2, axis=1)
    cosine = np.clip(np.sum(first_normals * second_normals, axis=1), -1.0, 1.0)
    first_fibers = fibers[sources]
    cross_fiber = np.cross(normal_cross, first_fibers)
    cross_cross_fiber = np.cross(normal_cross, cross_fiber)
    factors = np.divide(
        1.0 - cosine,
        np.maximum(sine2, 1.0e-12),
        out=np.zeros_like(cosine),
        where=sine2 >= 1.0e-12,
    )
    transported_fibers = (
        first_fibers
        + cross_fiber
        + cross_cross_fiber * factors[:, None]
    )
    transported_fibers /= np.maximum(
        np.linalg.norm(transported_fibers, axis=1, keepdims=True), 1.0e-8
    )
    fiber_dot = np.clip(
        np.abs(np.sum(transported_fibers * fibers[targets], axis=1)), 0.0, 1.0
    )
    fiber_angles = np.degrees(np.arccos(fiber_dot))
    delta = centers[targets] - centers[sources]
    position_residuals = 0.5 * (
        np.abs(np.sum(first_normals * delta, axis=1))
        + np.abs(np.sum(second_normals * delta, axis=1))
    )
    shared_fractions = np.zeros(len(sources), dtype=np.float32)
    if cell_step == 1:
        for pair_index, (source, target) in enumerate(zip(sources, targets)):
            first_ids = flakes[int(source)].get("_needleIds", set())
            second_ids = flakes[int(target)].get("_needleIds", set())
            if first_ids and second_ids:
                shared_fractions[pair_index] = len(first_ids.intersection(second_ids)) / max(
                    1, min(len(first_ids), len(second_ids))
                )
    pair_quality = np.sqrt(qualities[sources] * qualities[targets])
    compatibility = pair_quality * np.exp(
        -0.5
        * (
            (position_residuals / 7.0) ** 2
            + (normal_angles / 12.0) ** 2
            + (fiber_angles / 18.0) ** 2
        )
    )
    independent_scores = compatibility * (1.0 - 0.65 * shared_fractions)
    valid = (
        (position_residuals <= 16.0)
        & (normal_angles <= 24.0)
        & (fiber_angles <= 36.0)
        & (compatibility >= 0.055)
    )
    valid_positions = np.flatnonzero(valid)
    if not len(valid_positions):
        return []
    valid_groups = groups[valid_positions]
    group_order = np.argsort(valid_groups, kind="stable")
    ordered_positions = valid_positions[group_order]
    ordered_groups = valid_groups[group_order]
    starts = np.flatnonzero(
        np.r_[True, ordered_groups[1:] != ordered_groups[:-1]]
    )
    matched_positions: list[int] = []
    for group_number, start in enumerate(starts):
        stop = starts[group_number + 1] if group_number + 1 < len(starts) else len(ordered_positions)
        positions = ordered_positions[start:stop]
        positions = positions[np.argsort(independent_scores[positions])[::-1]]
        best_for_source: dict[int, int] = {}
        best_for_target: dict[int, int] = {}
        for position in positions:
            source = int(sources[position])
            target = int(targets[position])
            best_for_source.setdefault(source, int(position))
            best_for_target.setdefault(target, int(position))
        for source, position in best_for_source.items():
            target = int(targets[position])
            reverse_position = best_for_target.get(target)
            if reverse_position is not None and int(sources[reverse_position]) == source:
                matched_positions.append(position)
    links = []
    for position in matched_positions:
        links.append(
            {
                "source": int(sources[position]),
                "target": int(targets[position]),
                "score": round(float(independent_scores[position]), 4),
                "rawCompatibility": round(float(compatibility[position]), 4),
                "positionResidualVoxels": round(float(position_residuals[position]), 3),
                "normalAngleDeg": round(float(normal_angles[position]), 3),
                "fiberAngleDeg": round(float(fiber_angles[position]), 3),
                "sharedNeedleFraction": round(float(shared_fractions[position]), 4),
            }
        )
    return links


def _track_sizes(
    flake_count: int, links: list[dict[str, Any]], minimum_score: float
) -> list[int]:
    parent = list(range(flake_count))

    def find(index: int) -> int:
        while parent[index] != index:
            parent[index] = parent[parent[index]]
            index = parent[index]
        return index

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root != second_root:
            parent[second_root] = first_root

    for link in links:
        if float(link["score"]) >= minimum_score:
            union(int(link["source"]), int(link["target"]))
    counts: dict[int, int] = {}
    for index in range(flake_count):
        root = find(index)
        counts[root] = counts.get(root, 0) + 1
    return list(counts.values())


def _median_or_none(values: list[float], digits: int = 3) -> float | None:
    if not values:
        return None
    return round(float(np.median(np.asarray(values, dtype=np.float64))), digits)


def slab_flake_plane(
    output_root: str | Path,
    z_index: int,
    maximum_flakes: int = 3,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    analysis_path = root / "analysis.json"
    analysis = json.loads(analysis_path.read_text())
    if analysis.get("state") != "complete":
        raise ValueError("the cross-scroll slab analysis is not complete")
    grid = json.loads((root / "grid.json").read_text())
    grid_z = len(grid["z"])
    if not 0 <= z_index < grid_z:
        raise ValueError(f"zIndex must be between 0 and {grid_z - 1}")
    maximum_flakes = int(np.clip(maximum_flakes, 1, 5))
    depth_bandwidth = 4.0
    angle_bandwidth = 12.0
    minimum_needles = 5
    settings = analysis["identity"]["settings"]
    family_result, normal_families = load_normal_families(root)
    identity = {
        "version": FLAKE_CACHE_VERSION,
        "analysisIdentity": analysis["identity"],
        "normalFamilyVersion": NORMAL_FAMILY_VERSION,
        "normalFamilyIdentity": family_result["identity"],
        "zIndex": z_index,
        "maximumFlakes": maximum_flakes,
        "depthBandwidthVoxels": depth_bandwidth,
        "angleBandwidthDeg": angle_bandwidth,
        "minimumNeedles": minimum_needles,
    }
    cache_path = root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k{maximum_flakes}.json"
    membership_path = root / (
        f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k{maximum_flakes}-members.npz"
    )
    if cache_path.is_file() and membership_path.is_file() and not force:
        cached = json.loads(cache_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    cells = np.load(root / "cells.npy", mmap_mode="r")
    catalog = np.load(root / "needles.npy", mmap_mode="r")
    counts = np.load(root / "needle-counts.npy", mmap_mode="r")
    bin_shape_zyx = tuple(int(value) for value in analysis["binShapeZYX"])
    flakes: list[dict[str, Any]] = []
    valid_cell_count = 0
    fitted_cell_count = 0
    primary_fitted_cell_count = 0
    secondary_fitted_cell_count = 0
    cross_family_shared_needle_count = 0
    for cell_y, center_y in enumerate(grid["y"]):
        for cell_x, center_x in enumerate(grid["x"]):
            cell = cells[z_index, cell_y, cell_x]
            if not bool(cell["valid"]):
                continue
            valid_cell_count += 1
            cell_center = np.asarray(
                [center_x, center_y, grid["z"][z_index]], dtype=np.float32
            )
            records, record_ids = _catalog_records_for_cell(
                catalog, counts, cell_center, settings, bin_shape_zyx
            )
            partitions = _normal_family_partitions(
                records,
                record_ids,
                cell,
                normal_families[z_index, cell_y, cell_x],
            )
            fitted_in_cell = False
            fitted_ids_by_family = {0: set(), 1: set()}
            for partition in partitions:
                fitted = _fit_cell_flakes(
                    partition["records"],
                    partition["recordIds"],
                    cell_center,
                    partition["normal"],
                    partition["normalConfidence"],
                    (cell_x, cell_y, z_index),
                    int(settings["cubeSize"]),
                    maximum_flakes,
                    depth_bandwidth,
                    angle_bandwidth,
                    minimum_needles,
                    normal_family=int(partition["normalFamily"]),
                    family_coverage=float(partition["familyCoverage"]),
                    family_ambiguous_fraction=float(
                        partition["ambiguousFraction"]
                    ),
                    family_component_id=partition["familyComponentId"],
                    family_component_size=int(partition["familyComponentSize"]),
                )
                if fitted:
                    fitted_in_cell = True
                    family_index = int(partition["normalFamily"])
                    for flake in fitted:
                        fitted_ids_by_family[family_index].update(
                            flake["_needleIds"]
                        )
                    if family_index == 0:
                        primary_fitted_cell_count += 1
                    else:
                        secondary_fitted_cell_count += 1
                    flakes.extend(fitted)
            cross_family_shared_needle_count += len(
                fitted_ids_by_family[0].intersection(fitted_ids_by_family[1])
            )
            if fitted_in_cell:
                fitted_cell_count += 1
    for index, flake in enumerate(flakes):
        flake["id"] = index
    membership_offsets = [0]
    membership_parts: list[np.ndarray] = []
    for flake in flakes:
        values = np.asarray(sorted(flake["_needleIds"]), dtype=np.uint32)
        membership_parts.append(values)
        membership_offsets.append(membership_offsets[-1] + len(values))
    membership_ids = (
        np.concatenate(membership_parts)
        if membership_parts
        else np.empty(0, dtype=np.uint32)
    )
    links = _match_flakes_at_step(flakes)
    default_track_score = 0.12
    track_sizes = _track_sizes(len(flakes), links, default_track_score)
    linked_track_sizes = [size for size in track_sizes if size >= 2]
    rng = np.random.default_rng(358)
    real_fiber_angles = [float(link["fiberAngleDeg"]) for link in links]
    shuffled_fiber_angles: list[float] = []
    if links and len(flakes) > 1:
        shuffled = rng.permutation(len(flakes))
        for link in links:
            first = flakes[int(link["source"])]
            second = flakes[int(shuffled[int(link["target"])])]
            first_normal = np.asarray(first["normal"], dtype=np.float32)
            second_normal = np.asarray(second["normal"], dtype=np.float32)
            if float(np.dot(first_normal, second_normal)) < 0.0:
                second_normal = -second_normal
            transported = _minimal_rotation(first_normal, second_normal) @ np.asarray(
                first["fiber"], dtype=np.float32
            )
            shuffled_fiber_angles.append(
                _axial_angle_degrees(transported, np.asarray(second["fiber"], dtype=np.float32))
            )
    for flake in flakes:
        flake.pop("_needleIds", None)
    elapsed_ms = (time.monotonic() - started) * 1000.0
    accepted_links = [link for link in links if float(link["score"]) >= default_track_score]
    result = {
        "identity": identity,
        "view": {
            "mode": "slice",
            "zIndex": z_index,
            "z": grid["z"][z_index],
        },
        "settings": {
            "maximumFlakesPerFamily": maximum_flakes,
            "maximumNormalFamiliesPerCell": 2,
            "depthBandwidthVoxels": depth_bandwidth,
            "angleBandwidthDeg": angle_bandwidth,
            "minimumNeedles": minimum_needles,
            "defaultTrackScore": default_track_score,
            "gridStride": settings["gridStride"],
            "cubeSize": settings["cubeSize"],
        },
        "flakes": flakes,
        "links": links,
        "stats": {
            "elapsedMs": round(elapsed_ms, 2),
            "cacheHit": False,
            "validCellCount": valid_cell_count,
            "fittedCellCount": fitted_cell_count,
            "primaryFittedCellCount": primary_fitted_cell_count,
            "secondaryFittedCellCount": secondary_fitted_cell_count,
            "primaryFlakeCount": sum(
                int(flake["normalFamily"]) == 0 for flake in flakes
            ),
            "secondaryFlakeCount": sum(
                int(flake["normalFamily"]) == 1 for flake in flakes
            ),
            "crossFamilySharedNeedleCount": cross_family_shared_needle_count,
            "flakeCount": len(flakes),
            "candidateLinkCount": len(links),
            "acceptedLinkCount": len(accepted_links),
            "linkedTrackCount": len(linked_track_sizes),
            "largestTrackSize": max(linked_track_sizes, default=0),
            "medianTrackSize": _median_or_none([float(value) for value in linked_track_sizes], 2),
            "medianFlakesPerFittedCell": round(
                len(flakes) / max(fitted_cell_count, 1), 3
            ),
            "medianQuality": _median_or_none(
                [float(flake["quality"]) for flake in flakes], 4
            ),
            "medianPositionResidualVoxels": _median_or_none(
                [float(link["positionResidualVoxels"]) for link in accepted_links]
            ),
            "medianNormalAngleDeg": _median_or_none(
                [float(link["normalAngleDeg"]) for link in accepted_links]
            ),
            "medianFiberAngleDeg": _median_or_none(
                [float(link["fiberAngleDeg"]) for link in accepted_links]
            ),
            "fiberShuffledMedianDeg": _median_or_none(shuffled_fiber_angles),
            "medianSharedNeedleFraction": _median_or_none(
                [float(link["sharedNeedleFraction"]) for link in accepted_links], 4
            ),
            "constraint": (
                "local depth-orientation flake hypotheses and mutual adjacent-cell matches; "
                "no physical sheet identity assigned"
            ),
        },
    }
    _atomic_compact_json(cache_path, result)
    membership_temporary = membership_path.with_suffix(".npz.tmp")
    with membership_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            offsets=np.asarray(membership_offsets, dtype=np.uint32),
            ids=membership_ids,
        )
    membership_temporary.replace(membership_path)
    return result
