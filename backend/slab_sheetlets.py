from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flake_holdout import FLAKE_HOLDOUT_VERSION, slab_flake_holdout
from .slab_flakes import FLAKE_CACHE_VERSION, slab_flake_plane


SHEETLET_VERSION = 1
_MEMORY_CACHE: dict[tuple[str, int, int], dict[str, Any]] = {}


def _atomic_compact_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _median(values: list[float], digits: int = 4) -> float | None:
    if not values:
        return None
    return round(float(np.median(np.asarray(values, dtype=np.float64))), digits)


def _match_3d(
    flakes: list[dict[str, Any]], cell_step: int
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for index, flake in enumerate(flakes):
        cell = tuple(int(value) for value in flake["cellIndex"])
        by_cell.setdefault(cell, []).append(index)
    source_parts: list[np.ndarray] = []
    target_parts: list[np.ndarray] = []
    group_parts: list[np.ndarray] = []
    axis_parts: list[np.ndarray] = []
    group_index = 0
    offsets = ((cell_step, 0, 0), (0, cell_step, 0), (0, 0, cell_step))
    for cell, first_indices in by_cell.items():
        for axis, offset in enumerate(offsets):
            neighbor = (
                cell[0] + offset[0],
                cell[1] + offset[1],
                cell[2] + offset[2],
            )
            second_indices = by_cell.get(neighbor)
            if not second_indices:
                continue
            first_array = np.asarray(first_indices, dtype=np.int32)
            second_array = np.asarray(second_indices, dtype=np.int32)
            pair_count = len(first_array) * len(second_array)
            source_parts.append(np.repeat(first_array, len(second_array)))
            target_parts.append(np.tile(second_array, len(first_array)))
            group_parts.append(np.full(pair_count, group_index, dtype=np.int32))
            axis_parts.append(np.full(pair_count, axis, dtype=np.uint8))
            group_index += 1
    if not source_parts:
        return []
    sources = np.concatenate(source_parts)
    targets = np.concatenate(target_parts)
    groups = np.concatenate(group_parts)
    axes = np.concatenate(axis_parts)
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    qualities = np.asarray([flake["quality"] for flake in flakes], dtype=np.float32)
    validations = np.asarray(
        [flake["validationScore"] for flake in flakes], dtype=np.float32
    )
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
    transported = first_fibers + cross_fiber + cross_cross_fiber * factors[:, None]
    transported /= np.maximum(
        np.linalg.norm(transported, axis=1, keepdims=True), 1.0e-8
    )
    fiber_dot = np.clip(
        np.abs(np.sum(transported * fibers[targets], axis=1)), 0.0, 1.0
    )
    fiber_angles = np.degrees(np.arccos(fiber_dot))
    delta = centers[targets] - centers[sources]
    position_residuals = 0.5 * (
        np.abs(np.sum(first_normals * delta, axis=1))
        + np.abs(np.sum(second_normals * delta, axis=1))
    )
    compatibility = np.sqrt(qualities[sources] * qualities[targets]) * np.exp(
        -0.5
        * (
            (position_residuals / 7.0) ** 2
            + (normal_angles / 12.0) ** 2
            + (fiber_angles / 18.0) ** 2
        )
    )
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
    order = np.argsort(valid_groups, kind="stable")
    ordered_positions = valid_positions[order]
    ordered_groups = valid_groups[order]
    starts = np.flatnonzero(np.r_[True, ordered_groups[1:] != ordered_groups[:-1]])
    matches: list[int] = []
    for group_number, start in enumerate(starts):
        stop = starts[group_number + 1] if group_number + 1 < len(starts) else len(ordered_positions)
        positions = ordered_positions[start:stop]
        positions = positions[np.argsort(compatibility[positions])[::-1]]
        best_source: dict[int, int] = {}
        best_target: dict[int, int] = {}
        for position in positions:
            best_source.setdefault(int(sources[position]), int(position))
            best_target.setdefault(int(targets[position]), int(position))
        for source, position in best_source.items():
            target = int(targets[position])
            reverse = best_target.get(target)
            if reverse is not None and int(sources[reverse]) == source:
                matches.append(position)
    axis_names = ("x", "y", "z")
    return [
        {
            "source": int(sources[position]),
            "target": int(targets[position]),
            "axis": axis_names[int(axes[position])],
            "score": round(float(compatibility[position]), 4),
            "positionResidualVoxels": round(float(position_residuals[position]), 3),
            "normalAngleDeg": round(float(normal_angles[position]), 3),
            "fiberAngleDeg": round(float(fiber_angles[position]), 3),
            "endpointValidation": round(
                float(np.sqrt(validations[sources[position]] * validations[targets[position]])),
                4,
            ),
        }
        for position in matches
    ]


def _components(
    node_count: int, links: list[dict[str, Any]], minimum_score: float
) -> tuple[list[int], list[int], list[int]]:
    parent = list(range(node_count))
    degree = [0] * node_count

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
        if float(link["score"]) < minimum_score:
            continue
        source, target = int(link["source"]), int(link["target"])
        union(source, target)
        degree[source] += 1
        degree[target] += 1
    roots = [find(index) for index in range(node_count)]
    counts: dict[int, int] = {}
    for root in roots:
        counts[root] = counts.get(root, 0) + 1
    ranked = sorted(counts, key=lambda root: (-counts[root], root))
    identifiers = {root: index for index, root in enumerate(ranked)}
    return [identifiers[root] for root in roots], [counts[root] for root in roots], degree


def _spatial_shuffle(
    flakes: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    by_plane: dict[int, dict[tuple[int, int], list[int]]] = {}
    for index, flake in enumerate(flakes):
        cell_x, cell_y, cell_z = (int(value) for value in flake["cellIndex"])
        by_plane.setdefault(cell_z, {}).setdefault((cell_x, cell_y), []).append(index)
    output = [dict(flake) for flake in flakes]
    for cells in by_plane.values():
        source_cells = sorted(cells)
        target_cells = [source_cells[index] for index in rng.permutation(len(source_cells))]
        for source_cell, target_cell in zip(source_cells, target_cells):
            source_center = np.asarray(
                flakes[cells[source_cell][0]]["cellCenter"], dtype=np.float32
            )
            target_center = np.asarray(
                flakes[cells[target_cell][0]]["cellCenter"], dtype=np.float32
            )
            delta = target_center - source_center
            target_z = int(flakes[cells[target_cell][0]]["cellIndex"][2])
            for node_index in cells[source_cell]:
                copied = output[node_index]
                copied["cellIndex"] = [target_cell[0], target_cell[1], target_z]
                copied["cellCenter"] = target_center.tolist()
                copied["center"] = (
                    np.asarray(flakes[node_index]["center"], dtype=np.float32) + delta
                ).tolist()
    return output


def _graph_summary(
    node_count: int, links: list[dict[str, Any]], minimum_score: float
) -> dict[str, Any]:
    accepted = [link for link in links if float(link["score"]) >= minimum_score]
    _, sizes, _ = _components(node_count, accepted, minimum_score)
    component_sizes: dict[int, int] = {}
    identifiers, _, _ = _components(node_count, accepted, minimum_score)
    for component_id, size in zip(identifiers, sizes):
        component_sizes[component_id] = size
    linked_sizes = [size for size in component_sizes.values() if size >= 2]
    by_axis = {
        axis: [link for link in accepted if link["axis"] == axis]
        for axis in ("x", "y", "z")
    }
    return {
        "acceptedLinkCount": len(accepted),
        "acceptedXLinkCount": len(by_axis["x"]),
        "acceptedYLinkCount": len(by_axis["y"]),
        "acceptedZLinkCount": len(by_axis["z"]),
        "sheetletCount": len(linked_sizes),
        "linkedNodeCount": sum(linked_sizes),
        "largestSheetletSize": max(linked_sizes, default=0),
        "medianSheetletSize": _median([float(value) for value in linked_sizes], 2),
        "medianPositionResidualVoxels": _median(
            [float(link["positionResidualVoxels"]) for link in accepted], 3
        ),
        "medianNormalAngleDeg": _median(
            [float(link["normalAngleDeg"]) for link in accepted], 3
        ),
        "medianFiberAngleDeg": _median(
            [float(link["fiberAngleDeg"]) for link in accepted], 3
        ),
        "medianZFiberAngleDeg": _median(
            [float(link["fiberAngleDeg"]) for link in by_axis["z"]], 3
        ),
    }


def slab_sheetlet_graph(
    output_root: str | Path,
    cell_step: int = 2,
    repetitions: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    cell_step = int(np.clip(cell_step, 2, 4))
    repetitions = int(np.clip(repetitions, 2, 6))
    grid = json.loads((root / "grid.json").read_text())
    cache_key = (str(root.resolve()), cell_step, repetitions)
    cache_path = root / f"sheetlets-v{SHEETLET_VERSION}-s{cell_step}-r{repetitions}.json"
    if not force and cache_key in _MEMORY_CACHE:
        cached = _MEMORY_CACHE[cache_key]
        cached["stats"]["cacheHit"] = True
        cached["stats"]["elapsedMs"] = 0.0
        return cached
    dependency_paths = [
        path
        for z_index in range(len(grid["z"]))
        for path in (
            root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json",
            root / f"flake-holdout-v{FLAKE_HOLDOUT_VERSION}-z{z_index}-r4.json",
        )
    ]
    if (
        cache_path.is_file()
        and not force
        and all(path.is_file() for path in dependency_paths)
        and max(path.stat().st_mtime_ns for path in dependency_paths)
        <= cache_path.stat().st_mtime_ns
    ):
        cached = json.loads(cache_path.read_text())
        cached["stats"]["cacheHit"] = True
        cached["stats"]["elapsedMs"] = 0.0
        _MEMORY_CACHE[cache_key] = cached
        return cached
    plane_results = [slab_flake_plane(root, z_index, 3) for z_index in range(len(grid["z"]))]
    holdouts = [slab_flake_holdout(root, z_index, 4) for z_index in range(len(grid["z"]))]
    identity = {
        "version": SHEETLET_VERSION,
        "flakeCacheVersion": FLAKE_CACHE_VERSION,
        "holdoutVersion": FLAKE_HOLDOUT_VERSION,
        "flakeIdentities": [result["identity"] for result in plane_results],
        "holdoutIdentities": [result["identity"] for result in holdouts],
        "cellStep": cell_step,
        "repetitions": repetitions,
    }
    started = time.monotonic()
    flakes: list[dict[str, Any]] = []
    for z_index, (plane, holdout) in enumerate(zip(plane_results, holdouts)):
        validation = {
            int(value["flakeId"]): value for value in holdout["validationByFlake"]
        }
        for source in plane["flakes"]:
            support = validation.get(int(source["id"]))
            if not support or not bool(support["validated"]):
                continue
            flake = dict(source)
            flake["zIndex"] = z_index
            flake["flakeId"] = int(source["id"])
            flake["validationScore"] = float(support["validationScore"])
            flakes.append(flake)
    links = _match_3d(flakes, cell_step)
    minimum_score = 0.12
    accepted = [link for link in links if float(link["score"]) >= minimum_score]
    component_ids, component_sizes, degrees = _components(
        len(flakes), accepted, minimum_score
    )
    z_values_by_component: dict[int, list[float]] = {}
    for flake, component_id in zip(flakes, component_ids):
        z_values_by_component.setdefault(component_id, []).append(float(flake["center"][2]))
    nodes = []
    for index, flake in enumerate(flakes):
        component_id = component_ids[index]
        z_values = z_values_by_component[component_id]
        nodes.append(
            {
                "id": index,
                "zIndex": int(flake["zIndex"]),
                "flakeId": int(flake["flakeId"]),
                "sheetletId": component_id,
                "sheetletSize": component_sizes[index],
                "sheetletZSpanVoxels": round(max(z_values) - min(z_values), 3),
                "degree": degrees[index],
                "validationScore": round(float(flake["validationScore"]), 4),
            }
        )
    serialized_links = []
    for link in accepted:
        source, target = int(link["source"]), int(link["target"])
        serialized_links.append(
            {
                **link,
                "sourceZIndex": int(flakes[source]["zIndex"]),
                "targetZIndex": int(flakes[target]["zIndex"]),
                "sourceFlakeId": int(flakes[source]["flakeId"]),
                "targetFlakeId": int(flakes[target]["flakeId"]),
            }
        )
    observed_summary = _graph_summary(len(flakes), accepted, minimum_score)
    null_summaries = []
    rng = np.random.default_rng(0xAC0517)
    for _ in range(repetitions):
        shuffled = _spatial_shuffle(flakes, rng)
        null_links = _match_3d(shuffled, cell_step)
        null_summaries.append(_graph_summary(len(flakes), null_links, minimum_score))
    numeric_keys = [
        key for key, value in observed_summary.items() if isinstance(value, (int, float)) and value is not None
    ]
    null_summary = {
        key: _median(
            [float(summary[key]) for summary in null_summaries if summary.get(key) is not None]
        )
        for key in numeric_keys
    }
    null_summary["repetitions"] = repetitions
    elapsed_ms = (time.monotonic() - started) * 1000.0
    result = {
        "identity": identity,
        "settings": {
            "spacingVoxels": cell_step * int(plane_results[0]["settings"]["gridStride"]),
            "cellStep": cell_step,
            "minimumLinkScore": minimum_score,
            "heldoutValidatedNodesOnly": True,
            "null": "whole validated cell-pattern permutation within each axial plane before full 3D rematching",
            "repetitions": repetitions,
        },
        "nodes": nodes,
        "links": serialized_links,
        "stats": {
            "elapsedMs": round(elapsed_ms, 2),
            "cacheHit": False,
            "nodeCount": len(flakes),
            **observed_summary,
            "null": null_summary,
            "linkNullRatio": round(
                float(observed_summary["acceptedLinkCount"])
                / max(float(null_summary.get("acceptedLinkCount") or 0.0), 1.0e-8),
                4,
            ),
            "zLinkNullRatio": round(
                float(observed_summary["acceptedZLinkCount"])
                / max(float(null_summary.get("acceptedZLinkCount") or 0.0), 1.0e-8),
                4,
            ),
            "constraint": (
                "connected components of mutually matched, held-out-replicated flakes at "
                "non-overlapping 64-voxel X/Y/Z spacing; still hypotheses, not page identities"
            ),
        },
    }
    _atomic_compact_json(cache_path, result)
    _MEMORY_CACHE[cache_key] = result
    return result


def slab_sheetlet_slice(
    output_root: str | Path,
    z_index: int,
    cell_step: int = 2,
    repetitions: int = 4,
) -> dict[str, Any]:
    graph = slab_sheetlet_graph(output_root, cell_step, repetitions)
    grid = json.loads((Path(output_root) / "grid.json").read_text())
    if not 0 <= z_index < len(grid["z"]):
        raise ValueError(f"zIndex must be between 0 and {len(grid['z']) - 1}")
    nodes = [node for node in graph["nodes"] if int(node["zIndex"]) == z_index]
    links = [
        {
            "source": int(link["sourceFlakeId"]),
            "target": int(link["targetFlakeId"]),
            "score": link["score"],
            "axis": link["axis"],
            "positionResidualVoxels": link["positionResidualVoxels"],
            "normalAngleDeg": link["normalAngleDeg"],
            "fiberAngleDeg": link["fiberAngleDeg"],
            "endpointValidation": link["endpointValidation"],
        }
        for link in graph["links"]
        if int(link["sourceZIndex"]) == z_index and int(link["targetZIndex"]) == z_index
    ]
    return {
        "identity": graph["identity"],
        "view": {"mode": "slice", "zIndex": z_index, "z": grid["z"][z_index]},
        "settings": graph["settings"],
        "nodes": nodes,
        "links": links,
        "stats": graph["stats"],
    }
