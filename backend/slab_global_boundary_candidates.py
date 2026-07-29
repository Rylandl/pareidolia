from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .slab_branch_association import (
    BRANCH_ASSOCIATION_VERSION,
    DECISION_REDUNDANT,
    DECISION_RETAINED,
    DEFAULT_SETTINGS as BRANCH_SETTINGS,
    HIT_DTYPE,
    _aggregate_candidates,
    _branch_cell_sets,
    _node_material_state,
    _order_condensation,
    _solve_candidates,
)
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_material_intervals import MATERIAL_INTERVAL_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


GLOBAL_BOUNDARY_CANDIDATE_VERSION = 3

DEFAULT_SETTINGS: dict[str, Any] = {
    "minimumBranchSize": BRANCH_SETTINGS["minimumBranchSize"],
    "maximumBoundaryNodeDegree": 6,
    "minimumBoundaryDirectionConcentration": 0.25,
    "maximumBoundaryForwardNeighborCosine": 0.50,
    "maximumGapCells": BRANCH_SETTINGS["maximumGapCells"],
    "maximumEndpointDistanceVoxels": BRANCH_SETTINGS[
        "maximumEndpointDistanceVoxels"
    ],
    "minimumFacingCosine": BRANCH_SETTINGS["minimumFacingCosine"],
    "edgePaddingVoxels": BRANCH_SETTINGS["edgePaddingVoxels"],
    "minimumHitScore": BRANCH_SETTINGS["minimumHitScore"],
    "selectedThreshold": BRANCH_SETTINGS["selectedThreshold"],
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


def _quantiles(values: np.ndarray, digits: int = 4) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    if not len(values):
        return {name: None for name in names}
    return {
        name: round(float(value), digits)
        for name, value in zip(
            names, np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
        )
    }


def _load_global_node_geometry(
    root: Path,
    node_identity: np.ndarray,
    selected_node: np.ndarray,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    source_z = (node_identity >> np.uint64(32)).astype(np.int32)
    source_id = (node_identity & np.uint64(0xFFFFFFFF)).astype(np.int64)
    selected_position = np.full(len(node_identity), -1, dtype=np.int32)
    selected_position[selected_node] = np.arange(len(selected_node), dtype=np.int32)
    centers = np.empty((len(node_identity), 3), dtype=np.float32)
    raw_depth = np.empty(len(node_identity), dtype=np.float32)
    selected_flakes: list[dict[str, Any] | None] = [None] * len(selected_node)
    planes = np.unique(source_z)
    for plane_offset, z_index in enumerate(planes):
        payload = json.loads(
            (
                root
                / f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json"
            ).read_text()
        )
        records = payload["flakes"]
        selected = np.flatnonzero(source_z == z_index)
        ids = source_id[selected]
        if np.any(ids < 0) or np.any(ids >= len(records)):
            raise ValueError("global node references a missing source flake")
        centers[selected] = np.asarray(
            [records[int(flake_id)]["center"] for flake_id in ids],
            dtype=np.float32,
        )
        raw_depth[selected] = np.asarray(
            [records[int(flake_id)]["depthOffset"] for flake_id in ids],
            dtype=np.float32,
        )
        local_selected = selected[selected_position[selected] >= 0]
        for node_index in local_selected:
            position = int(selected_position[node_index])
            record = records[int(source_id[node_index])]
            if int(record["id"]) != int(source_id[node_index]):
                raise ValueError("flake cache IDs are not dense and index aligned")
            selected_flakes[position] = record
        if progress is not None:
            progress(
                "geometry",
                plane_offset + 1,
                len(planes),
                {"zIndex": int(z_index), "nodeCount": len(selected)},
            )
    if any(value is None for value in selected_flakes):
        raise ValueError("one selected global node has no source flake geometry")
    return centers, raw_depth, [value for value in selected_flakes if value is not None]


def _boundary_outward(
    boundary_flakes: list[dict[str, Any]],
    boundary_node: np.ndarray,
    centers: np.ndarray,
    edge_source: np.ndarray,
    edge_target: np.ndarray,
    retained: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normals = np.asarray(
        [flake["normal"] for flake in boundary_flakes], dtype=np.float32
    )
    position = np.full(len(centers), -1, dtype=np.int32)
    position[boundary_node] = np.arange(len(boundary_node), dtype=np.int32)
    direction_sum = np.zeros((len(boundary_node), 3), dtype=np.float32)
    neighbor_count = np.zeros(len(boundary_node), dtype=np.uint8)
    retained_source = edge_source[retained].astype(np.int64)
    retained_target = edge_target[retained].astype(np.int64)
    for source, target in (
        (retained_source, retained_target),
        (retained_target, retained_source),
    ):
        boundary_position = position[source]
        selected = boundary_position >= 0
        selected_position = boundary_position[selected]
        selected_source = source[selected]
        selected_target = target[selected]
        delta = centers[selected_target] - centers[selected_source]
        selected_normal = normals[selected_position]
        delta -= (
            np.sum(delta * selected_normal, axis=1, keepdims=True)
            * selected_normal
        )
        length = np.linalg.norm(delta, axis=1)
        usable = length >= 1.0
        selected_position = selected_position[usable]
        unit = delta[usable] / length[usable, None]
        np.add.at(direction_sum, selected_position, unit)
        np.add.at(neighbor_count, selected_position, np.uint8(1))
    resultant = np.linalg.norm(direction_sum, axis=1)
    concentration = resultant / np.maximum(neighbor_count.astype(np.float32), 1.0)
    usable = (neighbor_count >= 1) & (resultant >= 1.0e-6)
    outward = np.zeros_like(direction_sum)
    outward[usable] = -direction_sum[usable] / resultant[usable, None]
    maximum_forward_cosine = np.full(len(boundary_node), -1.0, dtype=np.float32)
    for source, target in (
        (retained_source, retained_target),
        (retained_target, retained_source),
    ):
        boundary_position = position[source]
        selected = boundary_position >= 0
        selected_position = boundary_position[selected]
        selected_source = source[selected]
        selected_target = target[selected]
        delta = centers[selected_target] - centers[selected_source]
        selected_normal = normals[selected_position]
        delta -= (
            np.sum(delta * selected_normal, axis=1, keepdims=True)
            * selected_normal
        )
        length = np.linalg.norm(delta, axis=1)
        direction_usable = length >= 1.0
        selected_position = selected_position[direction_usable]
        unit = delta[direction_usable] / length[direction_usable, None]
        forward = np.sum(unit * outward[selected_position], axis=1)
        np.maximum.at(maximum_forward_cosine, selected_position, forward)
    return (
        usable,
        outward,
        concentration,
        neighbor_count,
        maximum_forward_cosine,
    )


def _boundary_geometry_arrays(
    flakes: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    return {
        "normal": np.asarray([flake["normal"] for flake in flakes], dtype=np.float32),
        "fiber": np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32),
        "crossFiber": np.asarray(
            [flake["crossFiber"] for flake in flakes], dtype=np.float32
        ),
        "quality": np.asarray(
            [flake["quality"] for flake in flakes], dtype=np.float32
        ),
        "normalFamily": np.asarray(
            [int(flake.get("normalFamily", 0)) for flake in flakes],
            dtype=np.uint8,
        ),
        "radiusFiber": np.asarray(
            [flake["radiusFiber"] for flake in flakes], dtype=np.float32
        ),
        "radiusCrossFiber": np.asarray(
            [flake["radiusCrossFiber"] for flake in flakes], dtype=np.float32
        ),
    }


def _score_boundary_pair_arrays(
    source: np.ndarray,
    target: np.ndarray,
    axis: np.ndarray,
    branch: np.ndarray,
    centers: np.ndarray,
    outward: np.ndarray,
    geometry: dict[str, np.ndarray],
    material_supported: np.ndarray,
    contested: np.ndarray,
    edge_padding: float,
    minimum_hit_score: float,
) -> np.ndarray:
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    axis = np.asarray(axis, dtype=np.uint8)
    if not (len(source) == len(target) == len(axis)):
        raise ValueError("boundary pair arrays must have equal length")
    if not len(source):
        return np.empty(0, dtype=HIT_DTYPE)

    normals = geometry["normal"]
    fibers = geometry["fiber"]
    first_normal = normals[source]
    second_normal = normals[target].copy()
    signed_dot = np.sum(first_normal * second_normal, axis=1)
    second_normal[signed_dot < 0.0] *= -1.0
    normal_dot = np.clip(np.sum(first_normal * second_normal, axis=1), 0.0, 1.0)
    normal_bend = np.degrees(np.arccos(normal_dot))

    normal_cross = np.cross(first_normal, second_normal)
    sine2 = np.sum(normal_cross**2, axis=1)
    cosine = np.clip(np.sum(first_normal * second_normal, axis=1), -1.0, 1.0)
    first_fiber = fibers[source]
    cross_once = np.cross(normal_cross, first_fiber)
    cross_twice = np.cross(normal_cross, cross_once)
    factors = np.divide(
        1.0 - cosine,
        np.maximum(sine2, 1.0e-12),
        out=np.zeros_like(cosine),
        where=sine2 >= 1.0e-12,
    )
    transported_fiber = first_fiber + cross_once + cross_twice * factors[:, None]
    transported_fiber /= np.maximum(
        np.linalg.norm(transported_fiber, axis=1, keepdims=True), 1.0e-8
    )
    fiber_dot = np.clip(
        np.abs(np.sum(transported_fiber * fibers[target], axis=1)), 0.0, 1.0
    )
    fiber_angle = np.degrees(np.arccos(fiber_dot))

    delta = centers[target] - centers[source]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-6)
    first_slope = np.sum(first_normal * delta, axis=1)
    second_slope = np.sum(second_normal * delta, axis=1)
    edge_residual = 0.5 * np.abs(first_slope + second_slope)
    first_tangent = delta - first_slope[:, None] * first_normal
    second_tangent = delta - second_slope[:, None] * second_normal
    first_tangent /= np.maximum(
        np.linalg.norm(first_tangent, axis=1, keepdims=True), 1.0e-8
    )
    second_tangent /= np.maximum(
        np.linalg.norm(second_tangent, axis=1, keepdims=True), 1.0e-8
    )
    radius_fiber = geometry["radiusFiber"]
    radius_cross = geometry["radiusCrossFiber"]
    cross_fiber = geometry["crossFiber"]
    first_extent = 1.0 / np.maximum(
        np.sqrt(
            (
                np.sum(first_tangent * fibers[source], axis=1)
                / (radius_fiber[source] + edge_padding)
            )
            ** 2
            + (
                np.sum(first_tangent * cross_fiber[source], axis=1)
                / (radius_cross[source] + edge_padding)
            )
            ** 2
        ),
        1.0e-6,
    )
    second_extent = 1.0 / np.maximum(
        np.sqrt(
            (
                np.sum(second_tangent * fibers[target], axis=1)
                / (radius_fiber[target] + edge_padding)
            )
            ** 2
            + (
                np.sum(second_tangent * cross_fiber[target], axis=1)
                / (radius_cross[target] + edge_padding)
            )
            ** 2
        ),
        1.0e-6,
    )
    reach_ratio = np.maximum(
        0.5 * distance / np.maximum(first_extent, 1.0e-6),
        0.5 * distance / np.maximum(second_extent, 1.0e-6),
    )
    reach_excess = np.maximum(reach_ratio - 1.0, 0.0)
    quality = geometry["quality"]
    quality_support = np.sqrt(
        np.clip(quality[source] / 0.35, 0.0, 1.0)
        * np.clip(quality[target] / 0.35, 0.0, 1.0)
    )
    agreement = np.exp(
        -0.5
        * (
            (edge_residual / 4.0) ** 2
            + (fiber_angle / 12.0) ** 2
            + (reach_excess / 0.45) ** 2
        )
    )
    geometry_score = agreement * (0.7 + 0.3 * quality_support)
    unit = delta / distance[:, None]
    first_facing = np.maximum(np.sum(outward[source] * unit, axis=1), 0.0)
    second_facing = np.maximum(np.sum(outward[target] * -unit, axis=1), 0.0)
    facing = np.sqrt(first_facing * second_facing)
    score = geometry_score * facing
    valid = (
        (edge_residual <= 12.0)
        & (fiber_angle <= 40.0)
        & (normal_bend <= 75.0)
        & (geometry["normalFamily"][source] == geometry["normalFamily"][target])
        & (reach_ratio <= 2.0)
        & (geometry_score >= 0.12)
        & (score >= minimum_hit_score)
    )
    selected = np.flatnonzero(valid)
    output = np.empty(len(selected), dtype=HIT_DTYPE)
    if not len(selected):
        return output
    selected_source = source[selected].copy()
    selected_target = target[selected].copy()
    source_branch = branch[selected_source].astype(np.uint32)
    target_branch = branch[selected_target].astype(np.uint32)
    reverse = source_branch > target_branch
    source_branch[reverse], target_branch[reverse] = (
        target_branch[reverse].copy(),
        source_branch[reverse].copy(),
    )
    selected_source[reverse], selected_target[reverse] = (
        selected_target[reverse].copy(),
        selected_source[reverse].copy(),
    )
    output["branchSource"] = source_branch
    output["branchTarget"] = target_branch
    output["nodeSource"] = selected_source.astype(np.uint32)
    output["nodeTarget"] = selected_target.astype(np.uint32)
    output["score"] = score[selected]
    output["geometryScore"] = geometry_score[selected]
    output["facing"] = facing[selected]
    output["distanceVoxels"] = distance[selected]
    output["edgeResidualVoxels"] = edge_residual[selected]
    output["fiberAngleDeg"] = fiber_angle[selected]
    output["normalBendDeg"] = normal_bend[selected]
    output["reachRatio"] = reach_ratio[selected]
    output["endpointSupportedCount"] = (
        material_supported[source[selected]].astype(np.uint8)
        + material_supported[target[selected]].astype(np.uint8)
    )
    output["contestedEndpointCount"] = (
        contested[source[selected]].astype(np.uint8)
        + contested[target[selected]].astype(np.uint8)
    )
    return output


def _expanded_cell_pairs(
    ordered_node: np.ndarray,
    starts: np.ndarray,
    counts: np.ndarray,
    source_cell: np.ndarray,
    target_cell: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    products = counts[source_cell].astype(np.int64) * counts[target_cell].astype(
        np.int64
    )
    group = np.repeat(np.arange(len(products), dtype=np.int64), products)
    prefix = np.cumsum(products)
    within = np.arange(int(prefix[-1]), dtype=np.int64) - np.repeat(
        prefix - products, products
    )
    target_count = counts[target_cell[group]].astype(np.int64)
    source = ordered_node[
        starts[source_cell[group]] + within // target_count
    ]
    target = ordered_node[
        starts[target_cell[group]] + within % target_count
    ]
    return source.astype(np.uint32), target.astype(np.uint32)


def _stream_boundary_hits(
    cell: np.ndarray,
    branch: np.ndarray,
    centers: np.ndarray,
    outward: np.ndarray,
    geometry: dict[str, np.ndarray],
    material_supported: np.ndarray,
    contested: np.ndarray,
    grid_shape_xyz: np.ndarray,
    maximum_gap_cells: int,
    maximum_distance: float,
    minimum_facing: float,
    edge_padding: float,
    minimum_hit_score: float,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None,
    maximum_expanded_pairs: int = 2_000_000,
) -> tuple[np.ndarray, dict[str, int]]:
    shape = np.asarray(grid_shape_xyz, dtype=np.int64)
    code = (
        cell[:, 0].astype(np.int64)
        + shape[0]
        * (cell[:, 1].astype(np.int64) + shape[1] * cell[:, 2].astype(np.int64))
    )
    ordered_node = np.argsort(code, kind="stable").astype(np.uint32)
    ordered_code = code[ordered_node]
    occupied_code, starts, counts = np.unique(
        ordered_code, return_index=True, return_counts=True
    )
    occupied_cell = np.stack(
        (
            occupied_code % shape[0],
            (occupied_code // shape[0]) % shape[1],
            occupied_code // (shape[0] * shape[1]),
        ),
        axis=1,
    ).astype(np.int32)
    dense = np.full(int(np.prod(shape)), -1, dtype=np.int32)
    dense[occupied_code] = np.arange(len(occupied_code), dtype=np.int32)
    offsets = []
    for dz in range(-maximum_gap_cells, maximum_gap_cells + 1):
        for dy in range(-maximum_gap_cells, maximum_gap_cells + 1):
            for dx in range(-maximum_gap_cells, maximum_gap_cells + 1):
                if dz > 0 or (dz == 0 and dy > 0) or (
                    dz == 0 and dy == 0 and dx > 0
                ):
                    offsets.append((dx, dy, dz))

    hit_parts = []
    raw_pair_count = 0
    distinct_branch_pair_count = 0
    spatial_facing_pair_count = 0
    for offset_index, offset in enumerate(offsets):
        delta_cell = np.asarray(offset, dtype=np.int32)
        target_xyz = occupied_cell + delta_cell
        inside = np.all((target_xyz >= 0) & (target_xyz < shape), axis=1)
        source_cell = np.flatnonzero(inside)
        target_code = (
            target_xyz[source_cell, 0].astype(np.int64)
            + shape[0]
            * (
                target_xyz[source_cell, 1].astype(np.int64)
                + shape[1] * target_xyz[source_cell, 2].astype(np.int64)
            )
        )
        target_cell = dense[target_code]
        matched = target_cell >= 0
        source_cell = source_cell[matched]
        target_cell = target_cell[matched]
        products = counts[source_cell].astype(np.int64) * counts[
            target_cell
        ].astype(np.int64)
        cumulative = np.cumsum(products)
        cursor = 0
        while cursor < len(source_cell):
            base = int(cumulative[cursor - 1]) if cursor else 0
            stop = int(
                np.searchsorted(
                    cumulative, base + maximum_expanded_pairs, side="right"
                )
            )
            stop = max(stop, cursor + 1)
            source, target = _expanded_cell_pairs(
                ordered_node,
                starts,
                counts,
                source_cell[cursor:stop],
                target_cell[cursor:stop],
            )
            raw_pair_count += len(source)
            distinct = branch[source] != branch[target]
            source = source[distinct]
            target = target[distinct]
            distinct_branch_pair_count += len(source)
            if len(source):
                delta = centers[target] - centers[source]
                distance = np.linalg.norm(delta, axis=1)
                usable = (distance >= 1.0) & (distance <= maximum_distance)
                unit = np.zeros_like(delta)
                unit[usable] = delta[usable] / distance[usable, None]
                facing = np.minimum(
                    np.sum(outward[source] * unit, axis=1),
                    np.sum(outward[target] * -unit, axis=1),
                )
                usable &= facing >= minimum_facing
                source = source[usable]
                target = target[usable]
                spatial_facing_pair_count += len(source)
                if len(source):
                    axis = np.full(
                        len(source),
                        int(np.argmax(np.abs(delta_cell))),
                        dtype=np.uint8,
                    )
                    scored = _score_boundary_pair_arrays(
                        source,
                        target,
                        axis,
                        branch,
                        centers,
                        outward,
                        geometry,
                        material_supported,
                        contested,
                        edge_padding,
                        minimum_hit_score,
                    )
                    if len(scored):
                        hit_parts.append(scored)
            cursor = stop
        if progress is not None and (
            (offset_index + 1) % 16 == 0 or offset_index + 1 == len(offsets)
        ):
            progress(
                "pairing",
                offset_index + 1,
                len(offsets),
                {
                    "rawPairCount": raw_pair_count,
                    "spatialFacingPairCount": spatial_facing_pair_count,
                    "scoredHitCount": sum(len(value) for value in hit_parts),
                },
            )
    hits = np.concatenate(hit_parts) if hit_parts else np.empty(0, dtype=HIT_DTYPE)
    return hits, {
        "rawSpatialPairCount": raw_pair_count,
        "distinctBranchSpatialPairCount": distinct_branch_pair_count,
        "spatialFacingBoundaryPairCount": spatial_facing_pair_count,
    }


def _local_candidate_branch_pairs(
    root: Path,
    windows: list[dict[str, Any]],
    node_identity: np.ndarray,
    component: np.ndarray,
) -> tuple[np.ndarray, int]:
    branch_pairs = []
    occurrence_count = 0
    for window in windows:
        suffix = window_artifact_suffix(window["originCellXYZ"])
        monotone_path = root / (
            f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz"
        )
        association_path = root / (
            f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}.npz"
        )
        with np.load(monotone_path) as payload:
            local_identity = (
                payload["sourceZIndex"].astype(np.uint64) << np.uint64(32)
            ) | payload["sourceFlakeId"].astype(np.uint64)
        with np.load(association_path) as payload:
            endpoint_identity = np.stack(
                (
                    local_identity[payload["candidateNodeSource"].astype(np.int64)],
                    local_identity[payload["candidateNodeTarget"].astype(np.int64)],
                ),
                axis=1,
            )
        occurrence_count += len(endpoint_identity)
        endpoint_node = np.searchsorted(node_identity, endpoint_identity)
        inside = endpoint_node < len(node_identity)
        matches = np.zeros(endpoint_identity.shape, dtype=bool)
        matches[inside] = node_identity[endpoint_node[inside]] == endpoint_identity[inside]
        if not np.all(matches):
            raise ValueError("a local candidate endpoint is absent from the global graph")
        pairs = np.sort(component[endpoint_node], axis=1)
        branch_pairs.append(pairs[pairs[:, 0] != pairs[:, 1]])
    if not branch_pairs:
        return np.empty((0, 2), dtype=np.uint32), occurrence_count
    return np.unique(np.concatenate(branch_pairs), axis=0), occurrence_count


def _pair_membership(values: np.ndarray, catalog: np.ndarray) -> np.ndarray:
    values = np.ascontiguousarray(values, dtype=np.uint32)
    catalog = np.ascontiguousarray(catalog, dtype=np.uint32)
    if not len(values) or not len(catalog):
        return np.zeros(len(values), dtype=bool)
    dtype = np.dtype([("source", "<u4"), ("target", "<u4")])
    value_key = values.view(dtype).reshape(-1)
    catalog_key = catalog.view(dtype).reshape(-1)
    position = np.searchsorted(catalog_key, value_key)
    inside = position < len(catalog_key)
    matched = np.zeros(len(values), dtype=bool)
    matched[inside] = catalog_key[position[inside]] == value_key[inside]
    return matched


def build_global_boundary_candidates(
    output_root: str | Path,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    schedule_stem = f"tiled-window-schedule-v{WINDOW_SCHEDULER_VERSION}"
    graph_stem = f"global-monotone-graph-v{GLOBAL_MONOTONE_GRAPH_VERSION}"
    schedule_summary_path = root / f"{schedule_stem}.json"
    graph_summary_path = root / f"{graph_stem}.json"
    graph_artifact_path = root / f"{graph_stem}.npz"
    schedule = json.loads(schedule_summary_path.read_text())
    graph_summary = json.loads(graph_summary_path.read_text())
    grid = json.loads((root / "grid.json").read_text())
    local_paths = []
    for window in schedule["windows"]:
        suffix = window_artifact_suffix(window["originCellXYZ"])
        local_paths.extend(
            (
                root
                / f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz",
                root
                / (
                    f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}"
                    f"{suffix}.npz"
                ),
            )
        )
    flake_paths = [
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(len(grid["z"]))
    ]
    input_paths = [
        graph_artifact_path,
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-claims.npy",
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-intervals.npy",
        *local_paths,
        *flake_paths,
    ]
    identity = {
        "version": GLOBAL_BOUNDARY_CANDIDATE_VERSION,
        "settings": resolved,
        "scheduleIdentity": schedule["identity"],
        "globalGraphIdentity": graph_summary["identity"],
        "inputArtifacts": [_content_identity(path) for path in input_paths],
    }
    stem = f"global-boundary-candidates-v{GLOBAL_BOUNDARY_CANDIDATE_VERSION}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(graph_artifact_path) as payload:
        graph = {key: np.asarray(payload[key]) for key in payload.files}
    node_identity = graph["nodeIdentity"].astype(np.uint64)
    cell = graph["nodeCellIndex"].astype(np.int32)
    component = graph["component"].astype(np.int32)
    degree = graph["degree"].astype(np.uint8)
    boundary_degree_limit = int(resolved["maximumBoundaryNodeDegree"])
    boundary_node = np.flatnonzero(
        (degree >= 1)
        & (degree <= boundary_degree_limit)
        & (graph["componentSize"] >= int(resolved["minimumBranchSize"]))
    ).astype(np.uint32)
    centers, raw_depth, boundary_flakes = _load_global_node_geometry(
        root, node_identity, boundary_node, progress
    )
    (
        usable,
        outward,
        direction_concentration,
        usable_neighbor_count,
        maximum_forward_neighbor_cosine,
    ) = _boundary_outward(
        boundary_flakes,
        boundary_node,
        centers,
        graph["edgeSourceNodeIndex"].astype(np.uint32),
        graph["edgeTargetNodeIndex"].astype(np.uint32),
        graph["retained"].astype(bool),
    )
    exposed = usable & (
        direction_concentration
        >= float(resolved["minimumBoundaryDirectionConcentration"])
    ) & (
        maximum_forward_neighbor_cosine
        <= float(resolved["maximumBoundaryForwardNeighborCosine"])
    )
    unexposed_boundary_count = int(np.count_nonzero(~exposed))
    boundary_degree = degree[boundary_node][exposed]
    boundary_node = boundary_node[exposed]
    boundary_flakes = [
        boundary_flakes[index] for index in np.flatnonzero(exposed)
    ]
    outward = outward[exposed]
    direction_concentration = direction_concentration[exposed]
    usable_neighbor_count = usable_neighbor_count[exposed]
    maximum_forward_neighbor_cosine = maximum_forward_neighbor_cosine[exposed]
    boundary_cell = cell[boundary_node]
    boundary_branch = component[boundary_node]
    boundary_centers = centers[boundary_node]
    boundary_source_z = (
        node_identity[boundary_node] >> np.uint64(32)
    ).astype(np.int32)
    boundary_source_id = (
        node_identity[boundary_node] & np.uint64(0xFFFFFFFF)
    ).astype(np.int64)
    material_supported, contested = _node_material_state(
        root, boundary_source_z, boundary_source_id
    )
    if progress is not None:
        progress(
            "exposure",
            len(boundary_node),
            len(exposed),
            {
                "degreeCounts": {
                    str(value): int(np.count_nonzero(boundary_degree == value))
                    for value in np.unique(boundary_degree)
                }
            },
        )

    boundary_geometry = _boundary_geometry_arrays(boundary_flakes)
    hits, pairing_stats = _stream_boundary_hits(
        boundary_cell,
        boundary_branch,
        boundary_centers,
        outward,
        boundary_geometry,
        material_supported,
        contested,
        np.asarray(
            [len(grid["x"]), len(grid["y"]), len(grid["z"])],
            dtype=np.int64,
        ),
        int(resolved["maximumGapCells"]),
        float(resolved["maximumEndpointDistanceVoxels"]),
        float(resolved["minimumFacingCosine"]),
        float(resolved["edgePaddingVoxels"]),
        float(resolved["minimumHitScore"]),
        progress,
    )
    candidates = _aggregate_candidates(hits)
    candidate_branch_pair = np.stack(
        (candidates["branchSource"], candidates["branchTarget"]), axis=1
    ).astype(np.uint32)
    known_branch_pair, local_occurrence_count = _local_candidate_branch_pairs(
        root, schedule["windows"], node_identity, component
    )
    previously_proposed = _pair_membership(
        candidate_branch_pair, known_branch_pair
    )

    branch_count = int(np.max(component, initial=-1)) + 1
    order = _order_condensation(
        cell,
        raw_depth,
        component,
        graph["branchParity"].astype(np.int8),
    )
    branch_cells = _branch_cell_sets(cell, component, branch_count)
    novel_candidates = candidates[~previously_proposed]
    novel_original_index = np.flatnonzero(~previously_proposed)
    solve = _solve_candidates(
        novel_candidates,
        float(resolved["selectedThreshold"]),
        order,
        branch_cells,
    )
    decision = np.zeros(len(candidates), dtype=np.uint8)
    decision[novel_original_index] = solve["decisions"]
    selected = (~previously_proposed) & (
        (decision == DECISION_RETAINED) | (decision == DECISION_REDUNDANT)
    )
    selected_index = np.flatnonzero(selected)
    candidate_endpoint_node = np.stack(
        (
            boundary_node[candidates["nodeSource"].astype(np.int64)],
            boundary_node[candidates["nodeTarget"].astype(np.int64)],
        ),
        axis=1,
    ).astype(np.uint32)
    candidate_endpoint_identity = node_identity[candidate_endpoint_node]
    if np.any(component[candidate_endpoint_node] != candidate_branch_pair):
        raise ValueError("a scored endpoint no longer matches its global branch")

    artifact_arrays = {
        "candidateEndpointIdentity": candidate_endpoint_identity,
        "candidateEndpointNodeIndex": candidate_endpoint_node,
        "candidateBranchSource": candidates["branchSource"].astype(np.uint32),
        "candidateBranchTarget": candidates["branchTarget"].astype(np.uint32),
        "candidateScore": candidates["score"].astype(np.float32),
        "candidateHitCount": candidates["hitCount"].astype(np.uint16),
        "candidateSupport": candidates["support"].astype(np.uint16),
        "candidateGeometryScore": candidates["geometryScore"].astype(np.float32),
        "candidateFacing": candidates["facing"].astype(np.float32),
        "candidateDistanceVoxels": candidates["distanceVoxels"].astype(np.float32),
        "candidateEdgeResidualVoxels": candidates["edgeResidualVoxels"].astype(
            np.float32
        ),
        "candidateFiberAngleDeg": candidates["fiberAngleDeg"].astype(np.float32),
        "candidateNormalBendDeg": candidates["normalBendDeg"].astype(np.float32),
        "candidateReachRatio": candidates["reachRatio"].astype(np.float32),
        "candidateEndpointSupportedCount": candidates[
            "endpointSupportedCount"
        ].astype(np.uint8),
        "candidateContestedEndpointCount": candidates[
            "contestedEndpointCount"
        ].astype(np.uint8),
        "candidateBoundaryDirectionConcentration": np.stack(
            (
                direction_concentration[
                    candidates["nodeSource"].astype(np.int64)
                ],
                direction_concentration[
                    candidates["nodeTarget"].astype(np.int64)
                ],
            ),
            axis=1,
        ).astype(np.float32),
        "candidateBoundaryNodeDegree": np.stack(
            (
                boundary_degree[candidates["nodeSource"].astype(np.int64)],
                boundary_degree[candidates["nodeTarget"].astype(np.int64)],
            ),
            axis=1,
        ).astype(np.uint8),
        "candidateUsableNeighborCount": np.stack(
            (
                usable_neighbor_count[
                    candidates["nodeSource"].astype(np.int64)
                ],
                usable_neighbor_count[
                    candidates["nodeTarget"].astype(np.int64)
                ],
            ),
            axis=1,
        ).astype(np.uint8),
        "candidateMaximumForwardNeighborCosine": np.stack(
            (
                maximum_forward_neighbor_cosine[
                    candidates["nodeSource"].astype(np.int64)
                ],
                maximum_forward_neighbor_cosine[
                    candidates["nodeTarget"].astype(np.int64)
                ],
            ),
            axis=1,
        ).astype(np.float32),
        "candidatePreviouslyProposedLocally": previously_proposed,
        "candidateOrderDecision": decision,
        "candidateSelected": selected,
        "selectedCandidateIndex": selected_index.astype(np.uint32),
    }
    _atomic_npz(artifact_path, **artifact_arrays)
    artifact_identity = _content_identity(artifact_path)
    selected_values = candidates[selected]
    top = []
    for index in selected_index[:20]:
        value = candidates[int(index)]
        top.append(
            {
                "candidateIndex": int(index),
                "branchSource": int(value["branchSource"]),
                "branchTarget": int(value["branchTarget"]),
                "endpointIdentity": candidate_endpoint_identity[int(index)]
                .astype(int)
                .tolist(),
                "score": round(float(value["score"]), 6),
                "support": int(value["support"]),
                "distanceVoxels": round(float(value["distanceVoxels"]), 4),
                "edgeResidualVoxels": round(
                    float(value["edgeResidualVoxels"]), 4
                ),
                "fiberAngleDeg": round(float(value["fiberAngleDeg"]), 4),
                "normalBendDeg": round(float(value["normalBendDeg"]), 4),
            }
        )
    summary = {
        "identity": identity,
        "artifact": artifact_identity,
        "settings": resolved,
        "rules": {
            "boundary": (
                "nodes of degree one through maximumBoundaryNodeDegree on global "
                "primary-family branches with at least minimumBranchSize flakes; "
                "their outward direction is opposite the resultant of retained "
                "tangent-neighbor directions; it must meet the concentration gate, "
                "and no retained neighbor may already occupy the open cone"
            ),
            "candidate": (
                "spatially close mutually facing boundary nodes scored by the existing "
                "edge, fiber, bend, reach, material, order, and cell-collision rules"
            ),
            "novelty": (
                "a global branch pair is excluded if any local window proposed any "
                "endpoint pair that maps to those branches, regardless of its local "
                "decision"
            ),
            "semantics": (
                "selected entries are low-priority candidate joins only; complete-"
                "branch MLS reconstruction and cross-carrier integrity remain required"
            ),
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "globalNodeCount": len(node_identity),
            "globalBranchCount": branch_count,
            "degreeOneEndpointCount": int(
                np.count_nonzero(
                    (degree == 1)
                    & (
                        graph["componentSize"]
                        >= int(resolved["minimumBranchSize"])
                    )
                )
            ),
            "candidateBoundaryNodeCount": len(exposed),
            "exposedBoundaryNodeCount": len(boundary_node),
            "unexposedBoundaryNodeCount": unexposed_boundary_count,
            "exposedBoundaryNodeCountsByDegree": {
                str(value): int(np.count_nonzero(boundary_degree == value))
                for value in np.unique(boundary_degree)
            },
            "materialSupportedBoundaryNodeCount": int(
                np.count_nonzero(material_supported)
            ),
            "contestedBoundaryNodeCount": int(np.count_nonzero(contested)),
            **pairing_stats,
            "scoredEndpointHitCount": len(hits),
            "scoredBranchPairCount": len(candidates),
            "localCandidateOccurrenceCount": local_occurrence_count,
            "locallyProposedGlobalBranchPairCount": len(known_branch_pair),
            "rediscoveredLocalBranchPairCount": int(
                np.count_nonzero(previously_proposed)
            ),
            "novelScoredBranchPairCount": int(
                np.count_nonzero(~previously_proposed)
            ),
            "selectedNovelBranchPairCount": len(selected_index),
            "selectedScore": _quantiles(selected_values["score"]),
            "selectedSupport": _quantiles(selected_values["support"]),
            "selectedDistanceVoxels": _quantiles(
                selected_values["distanceVoxels"]
            ),
            "selectedEdgeResidualVoxels": _quantiles(
                selected_values["edgeResidualVoxels"]
            ),
            "selectedFiberAngleDeg": _quantiles(
                selected_values["fiberAngleDeg"]
            ),
            "selectedNormalBendDeg": _quantiles(
                selected_values["normalBendDeg"]
            ),
            "order": order["stats"],
            "novelSolve": solve["stats"],
        },
        "topSelectedCandidates": top,
    }
    _atomic_json(summary_path, summary)
    return summary
