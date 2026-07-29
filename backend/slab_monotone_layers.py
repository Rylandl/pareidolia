from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_material_intervals import MATERIAL_INTERVAL_VERSION
from .slab_sheetlet_explore import (
    SHEETLET_EXPLORE_VERSION,
    _components_without_cell_collisions,
    _score_batch,
)


MONOTONE_LAYER_VERSION = 1

DEFAULT_SETTINGS: dict[str, Any] = {
    "windowCellsXYZ": [32, 32, 14],
    "minimumFlakeQuality": 0.08,
    "minimumLinkScore": 0.60,
    "gapPenalty": 0.02,
    "edgePaddingVoxels": 8.0,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
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


def _box_sums(array: np.ndarray, shape_zyx: tuple[int, int, int]) -> np.ndarray:
    values = np.asarray(array, dtype=np.int64)
    wz, wy, wx = shape_zyx
    if wz > values.shape[0] or wy > values.shape[1] or wx > values.shape[2]:
        raise ValueError("window shape exceeds the flake-count grid")
    integral = np.pad(values, ((1, 0), (1, 0), (1, 0))).cumsum(0).cumsum(1).cumsum(2)
    return (
        integral[wz:, wy:, wx:]
        - integral[:-wz, wy:, wx:]
        - integral[wz:, :-wy, wx:]
        - integral[wz:, wy:, :-wx]
        + integral[:-wz, :-wy, wx:]
        + integral[:-wz, wy:, :-wx]
        + integral[wz:, :-wy, :-wx]
        - integral[:-wz, :-wy, :-wx]
    )


def _densest_window(
    profiles: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    window_shape_xyz: tuple[int, int, int],
) -> dict[str, Any]:
    counts = np.zeros(grid_shape_zyx, dtype=np.uint8)
    primary = profiles["normalFamily"] == 0
    cell = np.asarray(profiles["cellIndex"][primary], dtype=np.int32)
    counts[cell[:, 2], cell[:, 1], cell[:, 0]] = profiles["claimCount"][primary]
    wx, wy, wz = (
        min(int(value), grid_shape_zyx[2 - axis])
        for axis, value in enumerate(window_shape_xyz)
    )
    sums = _box_sums(counts, (wz, wy, wx))
    origin_zyx = np.unravel_index(int(np.argmax(sums)), sums.shape)
    z0, y0, x0 = (int(value) for value in origin_zyx)
    return {
        "originCellXYZ": [x0, y0, z0],
        "stopCellXYZExclusive": [x0 + wx, y0 + wy, z0 + wz],
        "shapeCellsXYZ": [wx, wy, wz],
        "primaryFlakeClaimCount": int(sums[origin_zyx]),
        "selection": "maximum primary-family flake count in an axis-aligned cell window",
    }


def window_artifact_suffix(
    origin_cell_xyz: tuple[int, int, int] | list[int] | None,
) -> str:
    if origin_cell_xyz is None:
        return ""
    x_index, y_index, z_index = (int(value) for value in origin_cell_xyz)
    return f"-x{x_index}-y{y_index}-z{z_index}"


def _explicit_window(
    profiles: np.ndarray,
    grid_shape_zyx: tuple[int, int, int],
    window_shape_xyz: tuple[int, int, int],
    origin_cell_xyz: tuple[int, int, int],
) -> dict[str, Any]:
    origin = np.asarray(origin_cell_xyz, dtype=np.int32)
    shape = np.asarray(window_shape_xyz, dtype=np.int32)
    grid_shape_xyz = np.asarray(grid_shape_zyx[::-1], dtype=np.int32)
    if np.any(shape <= 0) or np.any(origin < 0) or np.any(origin + shape > grid_shape_xyz):
        raise ValueError("explicit window lies outside the flake-count grid")
    stop = origin + shape
    primary = profiles["normalFamily"] == 0
    cell = np.asarray(profiles["cellIndex"], dtype=np.int32)
    inside = primary & np.all((cell >= origin) & (cell < stop), axis=1)
    return {
        "originCellXYZ": origin.astype(int).tolist(),
        "stopCellXYZExclusive": stop.astype(int).tolist(),
        "shapeCellsXYZ": shape.astype(int).tolist(),
        "primaryFlakeClaimCount": int(np.sum(profiles["claimCount"][inside])),
        "selection": "explicit axis-aligned cell window",
    }


def _load_window_flakes(
    root: Path,
    window: dict[str, Any],
    minimum_quality: float,
) -> list[dict[str, Any]]:
    x0, y0, z0 = window["originCellXYZ"]
    x1, y1, z1 = window["stopCellXYZExclusive"]
    flakes: list[dict[str, Any]] = []
    for z_index in range(z0, z1):
        payload = json.loads(
            (root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json").read_text()
        )
        for source in payload["flakes"]:
            cell_x, cell_y, cell_z = (int(value) for value in source["cellIndex"])
            if (
                int(source.get("normalFamily", 0)) != 0
                or float(source["quality"]) < minimum_quality
                or not (x0 <= cell_x < x1 and y0 <= cell_y < y1 and z0 <= cell_z < z1)
            ):
                continue
            flake = dict(source)
            flake["sourceZIndex"] = z_index
            flake["sourceFlakeId"] = int(source["id"])
            flakes.append(flake)
    return flakes


def _cell_catalog(
    flakes: list[dict[str, Any]],
) -> tuple[dict[tuple[int, int, int], list[int]], list[tuple[tuple[int, int, int], tuple[int, int, int], int]]]:
    by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, flake in enumerate(flakes):
        by_cell[tuple(int(value) for value in flake["cellIndex"])].append(index)
    pairs = []
    offsets = ((1, 0, 0), (0, 1, 0), (0, 0, 1))
    for cell in sorted(by_cell):
        for axis, offset in enumerate(offsets):
            neighbor = tuple(cell[index] + offset[index] for index in range(3))
            if neighbor in by_cell:
                pairs.append((cell, neighbor, axis))
    return dict(by_cell), pairs


def _relative_normal_signs(
    flakes: list[dict[str, Any]],
    by_cell: dict[tuple[int, int, int], list[int]],
    cell_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int], int]],
) -> tuple[dict[tuple[int, int, int], int], dict[str, Any]]:
    cells = sorted(by_cell)
    cell_id = {cell: index for index, cell in enumerate(cells)}
    normals = np.asarray(
        [flakes[by_cell[cell][0]]["normal"] for cell in cells], dtype=np.float32
    )
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), 1.0e-8)
    edges = []
    for first, second, _ in cell_pairs:
        dot = float(np.dot(normals[cell_id[first]], normals[cell_id[second]]))
        edges.append(
            (cell_id[first], cell_id[second], 1 if dot >= 0.0 else -1, abs(dot))
        )

    parent = np.arange(len(cells), dtype=np.int32)
    size = np.ones(len(cells), dtype=np.int32)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    tree: list[list[tuple[int, int]]] = [[] for _ in cells]
    for first, second, relative, _ in sorted(edges, key=lambda value: value[3], reverse=True):
        root_first, root_second = find(first), find(second)
        if root_first == root_second:
            continue
        if int(size[root_first]) < int(size[root_second]):
            root_first, root_second = root_second, root_first
        parent[root_second] = root_first
        size[root_first] += size[root_second]
        tree[first].append((second, relative))
        tree[second].append((first, relative))

    signs = np.zeros(len(cells), dtype=np.int8)
    component_count = 0
    for root in range(len(cells)):
        if signs[root]:
            continue
        component_count += 1
        signs[root] = 1
        queue = deque([root])
        while queue:
            current = queue.popleft()
            for neighbor, relative in tree[current]:
                if signs[neighbor]:
                    continue
                signs[neighbor] = signs[current] * relative
                queue.append(neighbor)
    disagreement = np.asarray(
        [signs[first] * signs[second] != relative for first, second, relative, _ in edges],
        dtype=bool,
    )
    weights = np.asarray([value[3] for value in edges], dtype=np.float32)
    angles = np.degrees(np.arccos(np.clip(weights, 0.0, 1.0)))
    output = {cell: int(signs[index]) for index, cell in enumerate(cells)}
    return output, {
        "cellCount": len(cells),
        "adjacentCellEdgeCount": len(edges),
        "connectedRegionCount": component_count,
        "frustratedEdgeCount": int(np.count_nonzero(disagreement)),
        "frustratedEdgeFraction": round(float(np.mean(disagreement)) if len(edges) else 0.0, 6),
        "frustratedWeightFraction": round(
            float(np.sum(weights[disagreement])) / max(float(np.sum(weights)), 1.0e-8), 6
        ),
        "neighborAxialNormalAngleDeg": _quantiles(angles),
        "signMeaning": (
            "arbitrary relative orientation per connected region; no inward, outward, "
            "recto, verso, or physical side meaning"
        ),
    }


def _all_pair_metrics(
    flakes: list[dict[str, Any]],
    by_cell: dict[tuple[int, int, int], list[int]],
    cell_pairs: list[tuple[tuple[int, int, int], tuple[int, int, int], int]],
    edge_padding: float,
) -> dict[tuple[int, int], tuple[float, float, float, float, float, int]]:
    output: dict[tuple[int, int], tuple[float, float, float, float, float, int]] = {}
    pending: list[tuple[list[int], list[int], int]] = []

    def flush() -> None:
        if not pending:
            return
        for source, target, axis, score, edge, fiber, bend, reach in _score_batch(
            flakes, pending, edge_padding
        ):
            output[(source, target)] = (score, edge, fiber, bend, reach, axis)
        pending.clear()

    for first, second, axis in cell_pairs:
        for source in by_cell[first]:
            for target in by_cell[second]:
                pending.append(([source], [target], axis))
                if len(pending) >= 200_000:
                    flush()
    flush()
    return output


def _monotone_partial_match(
    source_sequence: list[int],
    target_sequence: list[int],
    compatibility: dict[tuple[int, int], tuple[float, ...]],
    minimum_score: float,
    gap_penalty: float,
) -> list[tuple[int, int]]:
    """Maximum-score non-crossing partial match with explicit births/deaths."""
    source_count, target_count = len(source_sequence), len(target_sequence)
    score = np.full((source_count + 1, target_count + 1), -np.inf, dtype=np.float64)
    action = np.zeros((source_count + 1, target_count + 1), dtype=np.uint8)
    score[0, 0] = 0.0
    for source_index in range(1, source_count + 1):
        score[source_index, 0] = score[source_index - 1, 0] - gap_penalty
        action[source_index, 0] = 1
    for target_index in range(1, target_count + 1):
        score[0, target_index] = score[0, target_index - 1] - gap_penalty
        action[0, target_index] = 2
    for source_index in range(1, source_count + 1):
        for target_index in range(1, target_count + 1):
            options = [
                (score[source_index - 1, target_index] - gap_penalty, 1),
                (score[source_index, target_index - 1] - gap_penalty, 2),
            ]
            pair = (
                source_sequence[source_index - 1],
                target_sequence[target_index - 1],
            )
            metrics = compatibility.get(pair)
            if metrics is not None and float(metrics[0]) >= minimum_score:
                options.append(
                    (
                        score[source_index - 1, target_index - 1]
                        + float(metrics[0])
                        - minimum_score,
                        3,
                    )
                )
            best_value, best_action = max(options, key=lambda value: (value[0], value[1]))
            score[source_index, target_index] = best_value
            action[source_index, target_index] = best_action
    matches = []
    source_index, target_index = source_count, target_count
    while source_index or target_index:
        selected = int(action[source_index, target_index])
        if selected == 3:
            matches.append(
                (
                    source_sequence[source_index - 1],
                    target_sequence[target_index - 1],
                )
            )
            source_index -= 1
            target_index -= 1
        elif selected == 1:
            source_index -= 1
        elif selected == 2:
            target_index -= 1
        else:
            raise RuntimeError("partial-match backtracking reached an invalid state")
    matches.reverse()
    return matches


def _ordered_sequences(
    flakes: list[dict[str, Any]],
    by_cell: dict[tuple[int, int, int], list[int]],
    normal_sign: dict[tuple[int, int, int], int],
) -> tuple[dict[tuple[int, int, int], list[int]], np.ndarray, np.ndarray]:
    oriented_depth = np.asarray(
        [
            float(flake["depthOffset"])
            * normal_sign[tuple(int(value) for value in flake["cellIndex"])]
            for flake in flakes
        ],
        dtype=np.float32,
    )
    sequences = {
        cell: sorted(indices, key=lambda index: (float(oriented_depth[index]), index))
        for cell, indices in by_cell.items()
    }
    ordinal = np.zeros(len(flakes), dtype=np.uint8)
    for indices in sequences.values():
        for rank, index in enumerate(indices):
            ordinal[index] = rank
    return sequences, oriented_depth, ordinal


def _current_window_links(
    root: Path,
    flakes: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    component_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz"
    edge_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-edges.npz"
    with np.load(component_path) as payload:
        source_z = np.asarray(payload["sourceZIndex"], dtype=np.int32)
        source_id = np.asarray(payload["sourceFlakeId"], dtype=np.int32)
    local_to_global = np.full(len(flakes), -1, dtype=np.int32)
    for z_index in np.unique(source_z):
        global_indices = np.flatnonzero(source_z == z_index)
        maximum_id = int(np.max(source_id[global_indices], initial=-1))
        lookup = np.full(maximum_id + 1, -1, dtype=np.int32)
        lookup[source_id[global_indices]] = global_indices
        local_indices = [
            index for index, flake in enumerate(flakes) if int(flake["sourceZIndex"]) == z_index
        ]
        if local_indices:
            ids = np.asarray(
                [flakes[index]["sourceFlakeId"] for index in local_indices], dtype=np.int32
            )
            local_to_global[local_indices] = lookup[ids]
    if np.any(local_to_global < 0):
        raise ValueError("window flake is absent from the active sheetlet graph")
    global_to_local = np.full(len(source_z), -1, dtype=np.int32)
    global_to_local[local_to_global] = np.arange(len(flakes), dtype=np.int32)
    with np.load(edge_path) as payload:
        global_source = np.asarray(payload["source"], dtype=np.int32)
        global_target = np.asarray(payload["target"], dtype=np.int32)
        selected = np.asarray(payload["selectedLink"], dtype=bool)
        inside = (
            selected
            & (global_to_local[global_source] >= 0)
            & (global_to_local[global_target] >= 0)
        )
        return {
            "source": global_to_local[global_source[inside]],
            "target": global_to_local[global_target[inside]],
            "axis": np.asarray(payload["axis"], dtype=np.uint8)[inside],
            "score": np.asarray(payload["score"], dtype=np.float32)[inside],
            "edgeResidual": np.asarray(payload["edgeResidual"], dtype=np.float32)[inside],
            "fiberAngle": np.asarray(payload["fiberAngle"], dtype=np.float32)[inside],
            "normalBend": np.asarray(payload["normalBend"], dtype=np.float32)[inside],
        }


def _crossing_count(
    sources: np.ndarray,
    targets: np.ndarray,
    flakes: list[dict[str, Any]],
    ordinal: np.ndarray,
) -> int:
    groups: dict[tuple[tuple[int, ...], tuple[int, ...]], list[tuple[int, int]]] = defaultdict(list)
    for source, target in zip(sources, targets):
        source_cell = tuple(int(value) for value in flakes[int(source)]["cellIndex"])
        target_cell = tuple(int(value) for value in flakes[int(target)]["cellIndex"])
        groups[(source_cell, target_cell)].append(
            (int(ordinal[int(source)]), int(ordinal[int(target)]))
        )
    crossings = 0
    for matches in groups.values():
        for first_index, first in enumerate(matches):
            for second in matches[first_index + 1 :]:
                crossings += int((first[0] - second[0]) * (first[1] - second[1]) < 0)
    return crossings


def _top_branches(
    flakes: list[dict[str, Any]],
    component: np.ndarray,
    component_sizes: np.ndarray,
    limit: int = 12,
) -> list[dict[str, Any]]:
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    cells = np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32)
    output = []
    for component_id in np.argsort(component_sizes)[::-1]:
        size = int(component_sizes[component_id])
        if size < 2 or len(output) >= limit:
            break
        member = component == component_id
        output.append(
            {
                "componentId": int(component_id),
                "flakeCount": size,
                "cellCount": int(len(np.unique(cells[member], axis=0))),
                "axialPlaneCount": int(len(np.unique(cells[member, 2]))),
                "extentXYZ": np.round(np.ptp(centers[member], axis=0), 2).tolist(),
            }
        )
    return output


def prototype_monotone_layers(
    output_root: str | Path,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    window_origin_cell_xyz: tuple[int, int, int] | list[int] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    material_summary_path = root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}.json"
    material_summary = json.loads(material_summary_path.read_text())
    graph_summary_path = root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json"
    graph_summary = json.loads(graph_summary_path.read_text())
    grid = json.loads((root / "grid.json").read_text())
    input_paths = [
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-profiles.npy",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-edges.npz",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz",
    ]
    input_paths.extend(
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(len(grid["z"]))
    )
    input_artifacts = [_content_identity(path) for path in input_paths]
    identity: dict[str, Any] = {
        "version": MONOTONE_LAYER_VERSION,
        "materialIdentity": material_summary["identity"],
        "sheetletGraphIdentity": graph_summary["identity"],
        "settings": resolved,
        "inputArtifacts": input_artifacts,
    }
    if window_origin_cell_xyz is not None:
        identity["windowOriginCellXYZ"] = [
            int(value) for value in window_origin_cell_xyz
        ]
    suffix = window_artifact_suffix(window_origin_cell_xyz)
    stem = f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    profiles = np.load(
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-profiles.npy",
        mmap_mode="r",
    )
    grid_shape_zyx = (len(grid["z"]), len(grid["y"]), len(grid["x"]))
    window_shape = tuple(int(value) for value in resolved["windowCellsXYZ"])
    if window_origin_cell_xyz is None:
        window = _densest_window(profiles, grid_shape_zyx, window_shape)
    else:
        window = _explicit_window(
            profiles,
            grid_shape_zyx,
            window_shape,
            tuple(int(value) for value in window_origin_cell_xyz),
        )
    flakes = _load_window_flakes(
        root, window, float(resolved["minimumFlakeQuality"])
    )
    by_cell, cell_pairs = _cell_catalog(flakes)
    normal_sign, sign_stats = _relative_normal_signs(flakes, by_cell, cell_pairs)
    sequences, oriented_depth, ordinal = _ordered_sequences(
        flakes, by_cell, normal_sign
    )
    compatibility = _all_pair_metrics(
        flakes, by_cell, cell_pairs, float(resolved["edgePaddingVoxels"])
    )
    selected_pairs = []
    unmatched_source_count = 0
    unmatched_target_count = 0
    linked_cell_pair_count = 0
    for first, second, _ in cell_pairs:
        matches = _monotone_partial_match(
            sequences[first],
            sequences[second],
            compatibility,
            float(resolved["minimumLinkScore"]),
            float(resolved["gapPenalty"]),
        )
        selected_pairs.extend(matches)
        linked_cell_pair_count += int(bool(matches))
        unmatched_source_count += len(sequences[first]) - len(matches)
        unmatched_target_count += len(sequences[second]) - len(matches)
    sources = np.asarray([value[0] for value in selected_pairs], dtype=np.uint32)
    targets = np.asarray([value[1] for value in selected_pairs], dtype=np.uint32)
    metrics = np.asarray(
        [compatibility[(int(source), int(target))] for source, target in selected_pairs],
        dtype=np.float32,
    )
    scores = metrics[:, 0] if len(metrics) else np.empty(0, dtype=np.float32)
    cell = np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32)
    cell_code = (
        cell[:, 0].astype(np.int64)
        + grid_shape_zyx[2]
        * (cell[:, 1].astype(np.int64) + grid_shape_zyx[1] * cell[:, 2].astype(np.int64))
    )
    component, component_sizes, degree, retained = _components_without_cell_collisions(
        len(flakes), cell_code, sources, targets, scores
    )
    current = _current_window_links(root, flakes)
    current_pairs = {
        (int(source), int(target))
        for source, target in zip(current["source"], current["target"])
    }
    retained_pairs = {
        (int(sources[index]), int(targets[index])) for index in np.flatnonzero(retained)
    }
    (
        current_component,
        current_component_sizes,
        _,
        current_retained_again,
    ) = _components_without_cell_collisions(
        len(flakes),
        cell_code,
        current["source"].astype(np.uint32),
        current["target"].astype(np.uint32),
        current["score"],
    )
    current_linked_component_sizes = current_component_sizes[
        current_component_sizes >= 2
    ]
    linked = component_sizes[component] >= 2
    linked_component_sizes = component_sizes[component_sizes >= 2]
    shifted = ordinal[sources] != ordinal[targets]
    current_shifted = ordinal[current["source"]] != ordinal[current["target"]]
    current_crossings = _crossing_count(
        current["source"], current["target"], flakes, ordinal
    )
    monotone_crossings = _crossing_count(sources, targets, flakes, ordinal)

    artifact_temporary = artifact_path.with_suffix(".npz.tmp")
    with artifact_temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            sourceZIndex=np.asarray([flake["sourceZIndex"] for flake in flakes], dtype=np.uint8),
            sourceFlakeId=np.asarray([flake["sourceFlakeId"] for flake in flakes], dtype=np.uint32),
            cellIndex=cell.astype(np.uint16),
            normalSign=np.asarray(
                [normal_sign[tuple(int(value) for value in flake["cellIndex"])] for flake in flakes],
                dtype=np.int8,
            ),
            orientedDepth=oriented_depth,
            ordinal=ordinal,
            component=component.astype(np.uint32),
            componentSize=component_sizes[component].astype(np.uint32),
            degree=degree,
            source=sources,
            target=targets,
            axis=metrics[:, 5].astype(np.uint8) if len(metrics) else np.empty(0, dtype=np.uint8),
            score=scores,
            edgeResidual=metrics[:, 1] if len(metrics) else np.empty(0, dtype=np.float32),
            fiberAngle=metrics[:, 2] if len(metrics) else np.empty(0, dtype=np.float32),
            normalBend=metrics[:, 3] if len(metrics) else np.empty(0, dtype=np.float32),
            reachRatio=metrics[:, 4] if len(metrics) else np.empty(0, dtype=np.float32),
            retained=retained,
        )
    artifact_temporary.replace(artifact_path)
    result = {
        "identity": identity,
        "contract": {
            "normalSign": sign_stats["signMeaning"],
            "matching": (
                "order-preserving partial sequence alignment of local primary-family "
                "flake depths; unmatched modes are explicit births/deaths and ordinal "
                "equality is not required"
            ),
            "branchIdentity": (
                "collision-safe local surface branch only; component IDs are not "
                "physical papyrus sheet identities"
            ),
            "geometry": (
                "link evidence remains transported fiber direction, finite-patch edge "
                "agreement, local normal bend, reach, and flake quality"
            ),
        },
        "window": window,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "flakeCount": len(flakes),
            "occupiedCellCount": len(by_cell),
            "cellPairCount": len(cell_pairs),
            "compatiblePairCount": len(compatibility),
            "linkedCellPairCount": linked_cell_pair_count,
            "rawMonotoneLinkCount": len(sources),
            "retainedMonotoneLinkCount": int(np.count_nonzero(retained)),
            "collisionRejectedLinkCount": int(len(retained) - np.count_nonzero(retained)),
            "unmatchedSourceModeCountAcrossPairs": unmatched_source_count,
            "unmatchedTargetModeCountAcrossPairs": unmatched_target_count,
            "ordinalShiftLinkCount": int(np.count_nonzero(shifted & retained)),
            "pairwiseOrderCrossingCount": monotone_crossings,
            "retainedEdgeResidualVoxels": _quantiles(metrics[retained, 1] if len(metrics) else np.empty(0)),
            "retainedFiberAngleDeg": _quantiles(metrics[retained, 2] if len(metrics) else np.empty(0)),
            "retainedNormalBendDeg": _quantiles(metrics[retained, 3] if len(metrics) else np.empty(0)),
            "normalSignSynchronization": sign_stats,
            "branchCount": len(linked_component_sizes),
            "linkedFlakeCount": int(np.count_nonzero(linked)),
            "largestBranchSize": int(np.max(linked_component_sizes, initial=0)),
            "branchSize": _quantiles(linked_component_sizes),
            "topBranches": _top_branches(flakes, component, component_sizes),
            "currentGraphComparison": {
                "currentRetainedLinkCount": len(current_pairs),
                "currentLinksRetainedByWindowCollisionAudit": int(
                    np.count_nonzero(current_retained_again)
                ),
                "currentOrdinalShiftLinkCount": int(np.count_nonzero(current_shifted)),
                "currentPairwiseOrderCrossingCount": current_crossings,
                "currentBranchCount": len(current_linked_component_sizes),
                "currentLargestBranchSize": int(
                    np.max(current_linked_component_sizes, initial=0)
                ),
                "currentLinkedFlakeCount": int(
                    np.count_nonzero(
                        current_component_sizes[current_component] >= 2
                    )
                ),
                "preservedCurrentLinkCount": len(current_pairs & retained_pairs),
                "addedMonotoneLinkCount": len(retained_pairs - current_pairs),
                "removedCurrentLinkCount": len(current_pairs - retained_pairs),
                "currentEdgeResidualVoxels": _quantiles(current["edgeResidual"]),
                "currentFiberAngleDeg": _quantiles(current["fiberAngle"]),
            },
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
