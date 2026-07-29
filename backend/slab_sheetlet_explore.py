from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION, slab_flake_plane


SHEETLET_EXPLORE_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _median(values: np.ndarray, digits: int = 4) -> float | None:
    if not len(values):
        return None
    return round(float(np.median(values)), digits)


def _score_batch(
    flakes: list[dict[str, Any]],
    groups: list[tuple[list[int], list[int], int]],
    edge_padding: float,
) -> list[tuple[int, int, int, float, float, float, float, float]]:
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []
    for group_index, (first, second, axis) in enumerate(groups):
        first_array = np.asarray(first, dtype=np.int32)
        second_array = np.asarray(second, dtype=np.int32)
        pair_count = len(first_array) * len(second_array)
        source_parts.append(np.repeat(first_array, len(second_array)))
        target_parts.append(np.tile(second_array, len(first_array)))
        group_parts.append(np.full(pair_count, group_index, dtype=np.int32))
        axis_parts.append(np.full(pair_count, axis, dtype=np.uint8))
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    group_ids = np.concatenate(group_parts)
    axes = np.concatenate(axis_parts)
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    cross_fibers = np.asarray(
        [flake["crossFiber"] for flake in flakes], dtype=np.float32
    )
    qualities = np.asarray([flake["quality"] for flake in flakes], dtype=np.float32)
    radius_fiber = np.asarray(
        [flake["radiusFiber"] for flake in flakes], dtype=np.float32
    )
    radius_cross = np.asarray(
        [flake["radiusCrossFiber"] for flake in flakes], dtype=np.float32
    )

    first_normals = normals[sources]
    second_normals = normals[targets].copy()
    signed_dot = np.sum(first_normals * second_normals, axis=1)
    second_normals[signed_dot < 0.0] *= -1.0
    normal_dot = np.clip(
        np.sum(first_normals * second_normals, axis=1), 0.0, 1.0
    )
    normal_bend = np.degrees(np.arccos(normal_dot))

    normal_cross = np.cross(first_normals, second_normals)
    sine2 = np.sum(normal_cross**2, axis=1)
    cosine = np.clip(np.sum(first_normals * second_normals, axis=1), -1.0, 1.0)
    first_fibers = fibers[sources]
    cross_once = np.cross(normal_cross, first_fibers)
    cross_twice = np.cross(normal_cross, cross_once)
    factors = np.divide(
        1.0 - cosine,
        np.maximum(sine2, 1.0e-12),
        out=np.zeros_like(cosine),
        where=sine2 >= 1.0e-12,
    )
    transported_fibers = first_fibers + cross_once + cross_twice * factors[:, None]
    transported_fibers /= np.maximum(
        np.linalg.norm(transported_fibers, axis=1, keepdims=True), 1.0e-8
    )
    fiber_dot = np.clip(
        np.abs(np.sum(transported_fibers * fibers[targets], axis=1)), 0.0, 1.0
    )
    fiber_angle = np.degrees(np.arccos(fiber_dot))

    delta = centers[targets] - centers[sources]
    distance = np.maximum(np.linalg.norm(delta, axis=1), 1.0e-6)
    first_slope = np.sum(first_normals * delta, axis=1)
    second_slope = np.sum(second_normals * delta, axis=1)
    # A smooth curved carrier produces opposite endpoint slopes. Their sum is
    # the extrapolated disagreement where the two finite patches meet.
    edge_residual = 0.5 * np.abs(first_slope + second_slope)

    first_tangent = delta - first_slope[:, None] * first_normals
    second_tangent = delta - second_slope[:, None] * second_normals
    first_tangent /= np.maximum(
        np.linalg.norm(first_tangent, axis=1, keepdims=True), 1.0e-8
    )
    second_tangent /= np.maximum(
        np.linalg.norm(second_tangent, axis=1, keepdims=True), 1.0e-8
    )
    first_extent = 1.0 / np.maximum(
        np.sqrt(
            (
                np.sum(first_tangent * fibers[sources], axis=1)
                / (radius_fiber[sources] + edge_padding)
            )
            ** 2
            + (
                np.sum(first_tangent * cross_fibers[sources], axis=1)
                / (radius_cross[sources] + edge_padding)
            )
            ** 2
        ),
        1.0e-6,
    )
    second_extent = 1.0 / np.maximum(
        np.sqrt(
            (
                np.sum(second_tangent * fibers[targets], axis=1)
                / (radius_fiber[targets] + edge_padding)
            )
            ** 2
            + (
                np.sum(second_tangent * cross_fibers[targets], axis=1)
                / (radius_cross[targets] + edge_padding)
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
    quality_support = np.sqrt(
        np.clip(qualities[sources] / 0.35, 0.0, 1.0)
        * np.clip(qualities[targets] / 0.35, 0.0, 1.0)
    )
    agreement = np.exp(
        -0.5
        * (
            (edge_residual / 4.0) ** 2
            + (fiber_angle / 12.0) ** 2
            + (reach_excess / 0.45) ** 2
        )
    )
    score = agreement * (0.7 + 0.3 * quality_support)
    valid = (
        (edge_residual <= 12.0)
        & (fiber_angle <= 40.0)
        & (normal_bend <= 75.0)
        & (reach_ratio <= 2.0)
        & (score >= 0.12)
    )
    valid_positions = np.flatnonzero(valid)
    if not len(valid_positions):
        return []
    order = np.argsort(group_ids[valid_positions], kind="stable")
    ordered = valid_positions[order]
    ordered_groups = group_ids[ordered]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    matched: list[int] = []
    for group_number, start in enumerate(starts):
        stop = starts[group_number + 1] if group_number + 1 < len(starts) else len(ordered)
        positions = ordered[start:stop]
        positions = positions[np.argsort(score[positions])[::-1]]
        best_source: dict[int, int] = {}
        best_target: dict[int, int] = {}
        for position in positions:
            best_source.setdefault(int(sources[position]), int(position))
            best_target.setdefault(int(targets[position]), int(position))
        for source, position in best_source.items():
            target = int(targets[position])
            reverse = best_target.get(target)
            if reverse is not None and int(sources[reverse]) == source:
                matched.append(position)
    return [
        (
            int(sources[position]),
            int(targets[position]),
            int(axes[position]),
            float(score[position]),
            float(edge_residual[position]),
            float(fiber_angle[position]),
            float(normal_bend[position]),
            float(reach_ratio[position]),
        )
        for position in matched
    ]


def _direction_edge_links(
    flakes: list[dict[str, Any]], cell_step: int, edge_padding: float
) -> dict[str, np.ndarray]:
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for index, flake in enumerate(flakes):
        cell = tuple(int(value) for value in flake["cellIndex"])
        by_cell.setdefault(cell, []).append(index)
    offsets = ((cell_step, 0, 0), (0, cell_step, 0), (0, 0, cell_step))
    pending: list[tuple[list[int], list[int], int]] = []
    pending_pairs = 0
    links: list[tuple[int, int, int, float, float, float, float, float]] = []
    for cell, first in by_cell.items():
        for axis, offset in enumerate(offsets):
            neighbor = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            second = by_cell.get(neighbor)
            if not second:
                continue
            pending.append((first, second, axis))
            pending_pairs += len(first) * len(second)
            if pending_pairs >= 250_000:
                links.extend(_score_batch(flakes, pending, edge_padding))
                pending = []
                pending_pairs = 0
    if pending:
        links.extend(_score_batch(flakes, pending, edge_padding))
    if not links:
        return {
            "source": np.empty(0, dtype=np.uint32),
            "target": np.empty(0, dtype=np.uint32),
            "axis": np.empty(0, dtype=np.uint8),
            "score": np.empty(0, dtype=np.float32),
            "edgeResidual": np.empty(0, dtype=np.float32),
            "fiberAngle": np.empty(0, dtype=np.float32),
            "normalBend": np.empty(0, dtype=np.float32),
            "reachRatio": np.empty(0, dtype=np.float32),
        }
    values = np.asarray(links, dtype=np.float64)
    return {
        "source": values[:, 0].astype(np.uint32),
        "target": values[:, 1].astype(np.uint32),
        "axis": values[:, 2].astype(np.uint8),
        "score": values[:, 3].astype(np.float32),
        "edgeResidual": values[:, 4].astype(np.float32),
        "fiberAngle": values[:, 5].astype(np.float32),
        "normalBend": values[:, 6].astype(np.float32),
        "reachRatio": values[:, 7].astype(np.float32),
    }


def _components_without_cell_collisions(
    node_count: int,
    cell_code: np.ndarray,
    sources: np.ndarray,
    targets: np.ndarray,
    scores: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    parent = np.arange(node_count, dtype=np.int32)
    tree_size = np.ones(node_count, dtype=np.int32)
    degree = np.zeros(node_count, dtype=np.uint8)
    component_cells: dict[int, set[int]] = {}
    retained = np.zeros(len(sources), dtype=bool)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    for edge_index in np.argsort(scores)[::-1]:
        source_value, target_value = sources[edge_index], targets[edge_index]
        source, target = int(source_value), int(target_value)
        first, second = find(source), find(target)
        if first == second:
            retained[edge_index] = True
        else:
            first_cells = component_cells.get(first)
            if first_cells is None:
                first_cells = {int(cell_code[first])}
            second_cells = component_cells.get(second)
            if second_cells is None:
                second_cells = {int(cell_code[second])}
            if not first_cells.isdisjoint(second_cells):
                continue
            if int(tree_size[first]) < int(tree_size[second]):
                first, second = second, first
                first_cells, second_cells = second_cells, first_cells
            parent[second] = first
            tree_size[first] += tree_size[second]
            first_cells.update(second_cells)
            component_cells[first] = first_cells
            component_cells.pop(second, None)
            retained[edge_index] = True
        if not retained[edge_index]:
            continue
        degree[source] = min(255, int(degree[source]) + 1)
        degree[target] = min(255, int(degree[target]) + 1)
    roots = np.fromiter((find(index) for index in range(node_count)), dtype=np.int32)
    unique, inverse, counts = np.unique(roots, return_inverse=True, return_counts=True)
    del unique
    return inverse.astype(np.int32), counts.astype(np.int32), degree, retained


def _sweep_summary(
    flakes: list[dict[str, Any]], links: dict[str, np.ndarray], threshold: float
) -> tuple[dict[str, Any], np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    accepted = links["score"] >= threshold
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
    accepted_indices = np.flatnonzero(accepted)
    sources = links["source"][accepted]
    targets = links["target"][accepted]
    component, component_sizes, degree, retained_local = _components_without_cell_collisions(
        len(flakes), cell_code, sources, targets, links["score"][accepted]
    )
    retained = np.zeros(len(links["score"]), dtype=bool)
    retained[accepted_indices[retained_local]] = True
    node_sizes = component_sizes[component]
    linked = node_sizes >= 2
    linked_component_sizes = component_sizes[component_sizes >= 2]
    pair_code = component.astype(np.int64) * (int(np.max(cell_code)) + 1) + cell_code
    unique_component_cells = len(np.unique(pair_code[linked])) if np.any(linked) else 0
    collision_count = int(np.count_nonzero(linked)) - unique_component_cells
    multi_plane = np.zeros(len(component_sizes), dtype=np.uint8)
    axial_plane_count = int(cell_shape[2])
    for z_index in range(axial_plane_count):
        present = np.unique(component[cell_indices[:, 2] == z_index])
        multi_plane[present] += 1
    long_span_plane_count = max(2, int(math.ceil(axial_plane_count * 0.75)))
    axes = links["axis"][retained]
    top_ids = np.argsort(component_sizes)[-12:][::-1]
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    top_components = []
    for component_id in top_ids:
        size = int(component_sizes[component_id])
        if size < 2:
            continue
        member = component == component_id
        member_centers = centers[member]
        member_cells = cell_indices[member]
        top_components.append(
            {
                "componentId": int(component_id),
                "size": size,
                "uniqueCells": int(len(np.unique(cell_code[member]))),
                "planeCount": int(len(np.unique(member_cells[:, 2]))),
                "extentXYZ": np.round(np.ptp(member_centers, axis=0), 2).tolist(),
                "minimumXYZ": np.round(np.min(member_centers, axis=0), 2).tolist(),
                "maximumXYZ": np.round(np.max(member_centers, axis=0), 2).tolist(),
                "medianDegree": round(float(np.median(degree[member])), 2),
                "maximumDegree": int(np.max(degree[member])),
            }
        )
    return (
        {
            "threshold": threshold,
            "acceptedLinkCount": int(np.count_nonzero(accepted)),
            "retainedLinkCount": int(np.count_nonzero(retained)),
            "cellConflictRejectedLinkCount": int(np.count_nonzero(accepted) - np.count_nonzero(retained)),
            "retainedXLinkCount": int(np.count_nonzero(axes == 0)),
            "retainedYLinkCount": int(np.count_nonzero(axes == 1)),
            "retainedZLinkCount": int(np.count_nonzero(axes == 2)),
            "linkedNodeCount": int(np.count_nonzero(linked)),
            "linkedNodeFraction": round(float(np.mean(linked)), 4),
            "componentCount": int(len(linked_component_sizes)),
            "medianComponentSize": _median(linked_component_sizes.astype(np.float32), 2),
            "p90ComponentSize": round(float(np.percentile(linked_component_sizes, 90)), 2)
            if len(linked_component_sizes)
            else None,
            "largestComponentSize": int(np.max(linked_component_sizes, initial=0)),
            "multiPlaneComponentCount": int(np.count_nonzero(multi_plane >= 2)),
            "axialPlaneCount": axial_plane_count,
            "longSpanPlaneCount": long_span_plane_count,
            "longSpanComponentCount": int(
                np.count_nonzero(multi_plane >= long_span_plane_count)
            ),
            "allAxialPlaneComponentCount": int(
                np.count_nonzero(multi_plane == axial_plane_count)
            ),
            "allSixPlaneComponentCount": int(
                np.count_nonzero(multi_plane >= min(6, axial_plane_count))
            ),
            "cellCollisionCount": collision_count,
            "cellCollisionFraction": round(
                collision_count / max(int(np.count_nonzero(linked)), 1), 5
            ),
            "medianEdgeResidualVoxels": _median(links["edgeResidual"][retained], 3),
            "medianFiberAngleDeg": _median(links["fiberAngle"][retained], 3),
            "medianNormalBendDeg": _median(links["normalBend"][retained], 3),
            "medianReachRatio": _median(links["reachRatio"][retained], 3),
            "topComponents": top_components,
        },
        component,
        component_sizes,
        degree,
        retained,
    )


def _candidate_catalog(
    flakes: list[dict[str, Any]],
    links: dict[str, np.ndarray],
    retained: np.ndarray,
    component: np.ndarray,
    component_sizes: np.ndarray,
    degree: np.ndarray,
    minimum_size: int = 20,
) -> list[dict[str, Any]]:
    candidate_ids = set(
        int(value) for value in np.flatnonzero(component_sizes >= minimum_size)
    )
    members: dict[int, list[int]] = {value: [] for value in candidate_ids}
    for node_index, component_id in enumerate(component):
        key = int(component_id)
        if key in members:
            members[key].append(node_index)
    edge_members: dict[int, list[int]] = {value: [] for value in candidate_ids}
    for edge_index in np.flatnonzero(retained):
        key = int(component[int(links["source"][edge_index])])
        if key in edge_members:
            edge_members[key].append(int(edge_index))
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    cells = np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32)
    quality = np.asarray([flake["quality"] for flake in flakes], dtype=np.float32)
    catalog = []
    for component_id, member_values in members.items():
        member = np.asarray(member_values, dtype=np.int32)
        edge = np.asarray(edge_members[component_id], dtype=np.int32)
        member_centers = centers[member]
        member_normals = normals[member]
        member_fibers = fibers[member]
        normal_matrix = np.einsum("ni,nj->ij", member_normals, member_normals)
        _, normal_vectors = np.linalg.eigh(normal_matrix)
        reference_normal = normal_vectors[:, -1]
        normal_deviation = np.degrees(
            np.arccos(
                np.clip(np.abs(member_normals @ reference_normal), 0.0, 1.0)
            )
        )
        fiber_matrix = np.einsum("ni,nj->ij", member_fibers, member_fibers)
        _, fiber_vectors = np.linalg.eigh(fiber_matrix)
        reference_fiber = fiber_vectors[:, -1]
        fiber_deviation = np.degrees(
            np.arccos(
                np.clip(np.abs(member_fibers @ reference_fiber), 0.0, 1.0)
            )
        )
        extent = np.ptp(member_centers, axis=0)
        edge_residual_p90 = (
            float(np.percentile(links["edgeResidual"][edge], 90)) if len(edge) else 0.0
        )
        edge_fiber_p90 = (
            float(np.percentile(links["fiberAngle"][edge], 90)) if len(edge) else 0.0
        )
        rank = (
            len(member)
            * (1.0 + 0.2 * (len(np.unique(cells[member, 2])) - 1))
            * math.exp(-0.5 * (edge_residual_p90 / 4.0) ** 2)
            * math.exp(-0.5 * (edge_fiber_p90 / 12.0) ** 2)
        )
        axes = links["axis"][edge] if len(edge) else np.empty(0, dtype=np.uint8)
        catalog.append(
            {
                "componentId": component_id,
                "size": len(member),
                "planeCount": int(len(np.unique(cells[member, 2]))),
                "extentXYZ": np.round(extent, 2).tolist(),
                "minimumXYZ": np.round(np.min(member_centers, axis=0), 2).tolist(),
                "maximumXYZ": np.round(np.max(member_centers, axis=0), 2).tolist(),
                "retainedEdgeCount": len(edge),
                "edgeCountXYZ": [int(np.count_nonzero(axes == axis)) for axis in range(3)],
                "medianEdgeResidualVoxels": _median(links["edgeResidual"][edge], 3),
                "p90EdgeResidualVoxels": round(edge_residual_p90, 3),
                "medianEdgeFiberAngleDeg": _median(links["fiberAngle"][edge], 3),
                "p90EdgeFiberAngleDeg": round(edge_fiber_p90, 3),
                "medianEdgeNormalBendDeg": _median(links["normalBend"][edge], 3),
                "p90EdgeNormalBendDeg": round(
                    float(np.percentile(links["normalBend"][edge], 90)), 3
                )
                if len(edge)
                else None,
                "medianNormalDeviationDeg": _median(normal_deviation, 3),
                "p90NormalDeviationDeg": round(float(np.percentile(normal_deviation, 90)), 3),
                "medianGlobalFiberDeviationDeg": _median(fiber_deviation, 3),
                "p90GlobalFiberDeviationDeg": round(
                    float(np.percentile(fiber_deviation, 90)), 3
                ),
                "medianQuality": _median(quality[member], 4),
                "medianDegree": _median(degree[member].astype(np.float32), 2),
                "boundaryNodeFraction": round(float(np.mean(degree[member] <= 1)), 4),
                "rank": round(float(rank), 3),
            }
        )
    catalog.sort(key=lambda value: float(value["rank"]), reverse=True)
    return catalog


def analyze_sheetlets_exploratory(
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json"
    edge_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-edges.npz"
    component_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz"
    candidate_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-candidates.json"
    if (
        summary_path.is_file()
        and edge_path.is_file()
        and component_path.is_file()
        and candidate_path.is_file()
        and not force
    ):
        return json.loads(summary_path.read_text())

    started = time.monotonic()
    grid = json.loads((root / "grid.json").read_text())
    planes = [slab_flake_plane(root, z_index, 3) for z_index in range(len(grid["z"]))]
    flakes: list[dict[str, Any]] = []
    for z_index, plane in enumerate(planes):
        for flake in plane["flakes"]:
            if float(flake["quality"]) < 0.08:
                continue
            copied = dict(flake)
            copied["sourceZIndex"] = z_index
            copied["sourceFlakeId"] = int(flake["id"])
            flakes.append(copied)
    links = _direction_edge_links(flakes, cell_step=1, edge_padding=8.0)
    thresholds = [0.6, 0.7, 0.8, 0.85, 0.9, 0.93]
    sweeps = []
    component_results: dict[
        float, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
    ] = {}
    for threshold in thresholds:
        summary, component, sizes, degree, retained = _sweep_summary(flakes, links, threshold)
        sweeps.append(summary)
        component_results[threshold] = (component, sizes, degree, retained)

    # Prefer a graph that grows useful spans without obvious layer collisions or
    # a single percolating component. This is a construction setting, not a
    # statistical confidence threshold.
    eligible = [
        sweep
        for sweep in sweeps
        if int(sweep["largestComponentSize"]) <= max(2500, int(len(flakes) * 0.01))
    ]
    selected = max(
        eligible or sweeps,
        key=lambda sweep: (
            int(sweep["allAxialPlaneComponentCount"]),
            int(sweep["longSpanComponentCount"]),
            int(sweep["multiPlaneComponentCount"]),
            int(sweep["linkedNodeCount"]),
        ),
    )
    selected_threshold = float(selected["threshold"])
    component, component_sizes, degree, retained = component_results[selected_threshold]
    candidates = _candidate_catalog(
        flakes, links, retained, component, component_sizes, degree, minimum_size=20
    )
    source_z = np.asarray([flake["sourceZIndex"] for flake in flakes], dtype=np.uint8)
    source_flake_id = np.asarray(
        [flake["sourceFlakeId"] for flake in flakes], dtype=np.uint32
    )
    edge_temporary = edge_path.with_suffix(".npz.tmp")
    with edge_temporary.open("wb") as handle:
        np.savez_compressed(handle, **links, selectedLink=retained)
    edge_temporary.replace(edge_path)
    component_temporary = component_path.with_suffix(".npz.tmp")
    with component_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sourceZIndex=source_z,
            sourceFlakeId=source_flake_id,
            component=component.astype(np.uint32),
            componentSize=component_sizes[component].astype(np.uint32),
            degree=degree,
        )
    component_temporary.replace(component_path)
    _atomic_json(
        candidate_path,
        {
            "settings": {
                "selectedThreshold": selected_threshold,
                "minimumComponentSize": 20,
                "normalDeviationIsDescriptiveNotPenalized": True,
            },
            "stats": {
                "candidateCount": len(candidates),
                "medianCandidateSize": _median(
                    np.asarray([value["size"] for value in candidates], dtype=np.float32), 2
                ),
                "medianCandidateP90NormalDeviationDeg": _median(
                    np.asarray(
                        [value["p90NormalDeviationDeg"] for value in candidates],
                        dtype=np.float32,
                    ),
                    3,
                ),
            },
            "candidates": candidates,
        },
    )
    result = {
        "identity": {
            "version": SHEETLET_EXPLORE_VERSION,
            "flakeCacheVersion": FLAKE_CACHE_VERSION,
            "flakeIdentities": [plane["identity"] for plane in planes],
        },
        "settings": {
            "gridSpacingVoxels": int(planes[0]["settings"]["gridStride"]),
            "cellStep": 1,
            "minimumFlakeQuality": 0.08,
            "edgePaddingVoxels": 8.0,
            "edgeResidualScaleVoxels": 4.0,
            "fiberAngleScaleDeg": 12.0,
            "maximumNormalBendDeg": 75.0,
            "normalBendPenalized": False,
            "componentConstraint": "at most one flake per Acus cell",
            "selectedThreshold": selected_threshold,
            "construction": (
                "mutual neighbor matches by transported fiber direction and finite-patch "
                "edge extrapolation; normal change is retained as curvature"
            ),
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "nodeCount": len(flakes),
            "candidateMutualLinkCount": int(len(links["score"])),
            "selected": selected,
            "substantialCandidateCount": len(candidates),
            "topRankedCandidates": candidates[:12],
            "sweeps": sweeps,
            "constraint": (
                "exploratory sheetlet construction; no held-out gating and no page identity claim"
            ),
        },
        "artifacts": {
            "edges": edge_path.name,
            "components": component_path.name,
            "candidates": candidate_path.name,
        },
    }
    _atomic_json(summary_path, result)
    return result
