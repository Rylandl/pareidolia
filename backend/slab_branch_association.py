from __future__ import annotations

import hashlib
import json
import math
import time
from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_material_intervals import (
    LABEL_CONTESTED_MATERIAL,
    MATERIAL_INTERVAL_VERSION,
)
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_sheetlet_carriers import build_mls_carrier
from .slab_sheetlet_explore import score_flake_pair_groups


BRANCH_ASSOCIATION_VERSION = 2

DECISION_BELOW_THRESHOLD = 0
DECISION_MATERIAL_DEFERRED = 1
DECISION_ORDER_AMBIGUOUS = 2
DECISION_ORDER_BLOCKED = 3
DECISION_CELL_COLLISION = 4
DECISION_RETAINED = 5
DECISION_REDUNDANT = 6
DECISION_EXACT_PAIR_DEFERRED = 7
DECISION_EXACT_GROUP_PRUNED = 8
DECISION_NAMES = {
    DECISION_BELOW_THRESHOLD: "below-threshold",
    DECISION_MATERIAL_DEFERRED: "material-deferred",
    DECISION_ORDER_AMBIGUOUS: "order-ambiguous",
    DECISION_ORDER_BLOCKED: "order-blocked",
    DECISION_CELL_COLLISION: "cell-collision",
    DECISION_RETAINED: "retained",
    DECISION_REDUNDANT: "redundant-support",
    DECISION_EXACT_PAIR_DEFERRED: "exact-pair-deferred",
    DECISION_EXACT_GROUP_PRUNED: "exact-group-pruned",
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "minimumBranchSize": 2,
    "maximumGapCells": 3,
    "maximumEndpointDistanceVoxels": 128.0,
    "minimumFacingCosine": 0.10,
    "edgePaddingVoxels": 8.0,
    "minimumHitScore": 0.08,
    "thresholdSweep": [0.25, 0.35, 0.45, 0.55],
    "selectedThreshold": 0.45,
    "subwindowCellsXY": [24, 24],
    "minimumOverlapObservations": 2,
    "maximumExactMedianHeightResidualVoxels": 3.0,
    "maximumExactMedianNormalResidualDeg": 6.0,
    "exactCarrierPixelStepVoxels": 4.0,
    "exactCarrierMaximumPixelsPerAxis": 192,
}

HIT_DTYPE = np.dtype(
    [
        ("branchSource", "<u4"),
        ("branchTarget", "<u4"),
        ("nodeSource", "<u4"),
        ("nodeTarget", "<u4"),
        ("score", "<f4"),
        ("geometryScore", "<f4"),
        ("facing", "<f4"),
        ("distanceVoxels", "<f4"),
        ("edgeResidualVoxels", "<f4"),
        ("fiberAngleDeg", "<f4"),
        ("normalBendDeg", "<f4"),
        ("reachRatio", "<f4"),
        ("endpointSupportedCount", "u1"),
        ("contestedEndpointCount", "u1"),
    ]
)

CANDIDATE_DTYPE = np.dtype(
    [
        ("branchSource", "<u4"),
        ("branchTarget", "<u4"),
        ("nodeSource", "<u4"),
        ("nodeTarget", "<u4"),
        ("score", "<f4"),
        ("hitCount", "<u2"),
        ("support", "<u2"),
        ("geometryScore", "<f4"),
        ("facing", "<f4"),
        ("distanceVoxels", "<f4"),
        ("edgeResidualVoxels", "<f4"),
        ("fiberAngleDeg", "<f4"),
        ("normalBendDeg", "<f4"),
        ("reachRatio", "<f4"),
        ("endpointSupportedCount", "u1"),
        ("contestedEndpointCount", "u1"),
    ]
)


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


def _load_artifact_flakes(root: Path, arrays: dict[str, np.ndarray]) -> list[dict[str, Any]]:
    source_z = np.asarray(arrays["sourceZIndex"], dtype=np.int32)
    source_id = np.asarray(arrays["sourceFlakeId"], dtype=np.int32)
    payload_by_plane = {
        int(z_index): json.loads(
            (root / f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json").read_text()
        )["flakes"]
        for z_index in np.unique(source_z)
    }
    flakes = []
    for z_index, flake_id in zip(source_z, source_id):
        flake = payload_by_plane[int(z_index)][int(flake_id)]
        if int(flake["id"]) != int(flake_id):
            raise ValueError("flake cache IDs are not dense and index aligned")
        flakes.append(flake)
    return flakes


def _node_material_state(
    root: Path,
    source_z: np.ndarray,
    source_id: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    claims = np.load(
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-claims.npy",
        mmap_mode="r",
    )
    intervals = np.load(
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-intervals.npy",
        mmap_mode="r",
    )
    supported = np.zeros(len(source_z), dtype=bool)
    contested = np.zeros(len(source_z), dtype=bool)
    for z_index in np.unique(source_z):
        claim_member = (
            (claims["sourceZIndex"] == z_index)
            & (claims["normalFamily"] == 0)
        )
        claim_indices = np.flatnonzero(claim_member)
        if not len(claim_indices):
            continue
        maximum_id = int(np.max(claims["sourceFlakeId"][claim_indices]))
        lookup = np.full(maximum_id + 1, -1, dtype=np.int32)
        lookup[claims["sourceFlakeId"][claim_indices]] = claim_indices
        node_indices = np.flatnonzero(source_z == z_index)
        selected = lookup[source_id[node_indices]]
        if np.any(selected < 0):
            raise ValueError("monotone node is absent from material claim catalog")
        selected_supported = np.asarray(claims["supported"][selected], dtype=bool)
        supported[node_indices] = selected_supported
        interval_index = np.asarray(claims["intervalIndex"][selected], dtype=np.int32)
        valid = selected_supported & (interval_index >= 0)
        if np.any(valid):
            contested[node_indices[valid]] = (
                intervals["state"][interval_index[valid]]
                == LABEL_CONTESTED_MATERIAL
            )
    return supported, contested


def _endpoint_geometry(
    flakes: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    minimum_branch_size: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    source = arrays["source"][arrays["retained"]].astype(np.int32)
    target = arrays["target"][arrays["retained"]].astype(np.int32)
    neighbor = np.full(len(flakes), -1, dtype=np.int32)
    degree = np.zeros(len(flakes), dtype=np.uint8)
    for first, second in zip(source, target):
        degree[first] += 1
        degree[second] += 1
        neighbor[first] = second
        neighbor[second] = first
    endpoint = np.flatnonzero(
        (degree == 1) & (arrays["componentSize"] >= minimum_branch_size)
    )
    outward = np.zeros((len(flakes), 3), dtype=np.float32)
    if len(endpoint):
        delta = centers[endpoint] - centers[neighbor[endpoint]]
        delta -= (
            np.sum(delta * normals[endpoint], axis=1, keepdims=True)
            * normals[endpoint]
        )
        length = np.linalg.norm(delta, axis=1)
        usable = length >= 1.0
        outward[endpoint[usable]] = delta[usable] / length[usable, None]
        endpoint = endpoint[usable]
    return endpoint.astype(np.uint32), outward, centers


def _candidate_endpoint_pairs(
    endpoint: np.ndarray,
    cell: np.ndarray,
    branch: np.ndarray,
    centers: np.ndarray,
    outward: np.ndarray,
    maximum_gap_cells: int,
    maximum_distance: float,
    minimum_facing: float,
) -> list[tuple[int, int, int]]:
    bucket_size = max(1, int(math.ceil(maximum_gap_cells / 2)))
    bucket: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node in endpoint:
        key = tuple((cell[int(node)] // bucket_size).astype(int))
        bucket[key].append(int(node))
    bucket_radius = int(math.ceil(maximum_gap_cells / bucket_size))
    pairs = []
    for source in endpoint:
        source = int(source)
        source_key = tuple((cell[source] // bucket_size).astype(int))
        for dx in range(-bucket_radius, bucket_radius + 1):
            for dy in range(-bucket_radius, bucket_radius + 1):
                for dz in range(-bucket_radius, bucket_radius + 1):
                    key = (
                        source_key[0] + dx,
                        source_key[1] + dy,
                        source_key[2] + dz,
                    )
                    for target in bucket.get(key, ()):
                        if target <= source or int(branch[source]) == int(branch[target]):
                            continue
                        cell_delta = np.abs(cell[target].astype(int) - cell[source].astype(int))
                        if int(np.max(cell_delta)) > maximum_gap_cells or not np.any(cell_delta):
                            continue
                        delta = centers[target] - centers[source]
                        distance = float(np.linalg.norm(delta))
                        if not 1.0 <= distance <= maximum_distance:
                            continue
                        unit = delta / distance
                        facing = min(
                            float(np.dot(outward[source], unit)),
                            float(np.dot(outward[target], -unit)),
                        )
                        if facing < minimum_facing:
                            continue
                        axis = int(np.argmax(cell_delta))
                        pairs.append((source, target, axis))
    return pairs


def _score_endpoint_hits(
    flakes: list[dict[str, Any]],
    pairs: list[tuple[int, int, int]],
    branch: np.ndarray,
    centers: np.ndarray,
    outward: np.ndarray,
    material_supported: np.ndarray,
    contested: np.ndarray,
    edge_padding: float,
    minimum_hit_score: float,
) -> np.ndarray:
    hits = []
    for start in range(0, len(pairs), 200_000):
        batch = pairs[start : start + 200_000]
        groups = [([source], [target], axis) for source, target, axis in batch]
        scored = score_flake_pair_groups(flakes, groups, edge_padding)
        for source, target, _, geometry, edge, fiber, bend, reach in scored:
            delta = centers[target] - centers[source]
            distance = float(np.linalg.norm(delta))
            unit = delta / max(distance, 1.0e-8)
            first_facing = max(float(np.dot(outward[source], unit)), 0.0)
            second_facing = max(float(np.dot(outward[target], -unit)), 0.0)
            facing = math.sqrt(first_facing * second_facing)
            score = float(geometry) * facing
            if score < minimum_hit_score:
                continue
            branch_source, branch_target = int(branch[source]), int(branch[target])
            if branch_source > branch_target:
                branch_source, branch_target = branch_target, branch_source
                source, target = target, source
            hits.append(
                (
                    branch_source,
                    branch_target,
                    source,
                    target,
                    score,
                    geometry,
                    facing,
                    distance,
                    edge,
                    fiber,
                    bend,
                    reach,
                    int(material_supported[source]) + int(material_supported[target]),
                    int(contested[source]) + int(contested[target]),
                )
            )
    if not hits:
        return np.empty(0, dtype=HIT_DTYPE)
    return np.asarray(hits, dtype=HIT_DTYPE)


def _inside_window(cell: np.ndarray, window: dict[str, Any]) -> np.ndarray:
    low = np.asarray(window["originCellXYZ"], dtype=np.int32)
    high = np.asarray(window["stopCellXYZExclusive"], dtype=np.int32)
    return np.all((cell >= low) & (cell < high), axis=1)


def _aggregate_candidates(
    hits: np.ndarray,
    node_inside: np.ndarray | None = None,
) -> np.ndarray:
    if node_inside is not None and len(hits):
        hits = hits[
            node_inside[hits["nodeSource"]]
            & node_inside[hits["nodeTarget"]]
        ]
    if not len(hits):
        return np.empty(0, dtype=CANDIDATE_DTYPE)
    pair_code = (
        hits["branchSource"].astype(np.uint64) << np.uint64(32)
    ) | hits["branchTarget"].astype(np.uint64)
    order = np.argsort(pair_code, kind="stable")
    ordered_code = pair_code[order]
    starts = np.flatnonzero(np.r_[True, ordered_code[1:] != ordered_code[:-1]])
    output = []
    for group_index, start in enumerate(starts):
        stop = starts[group_index + 1] if group_index + 1 < len(starts) else len(order)
        group = hits[order[start:stop]]
        supported = group[group["endpointSupportedCount"] == 2]
        pool = supported if len(supported) else group
        ranked = pool[np.argsort(pool["score"])[::-1]]
        top = ranked[: min(5, len(ranked))]
        support = min(
            len(np.unique(pool["nodeSource"])),
            len(np.unique(pool["nodeTarget"])),
        )
        aggregate = float(np.median(top["score"]))
        aggregate *= 0.90 + 0.10 * min(support / 3.0, 1.0)
        best = ranked[0]
        output.append(
            (
                int(best["branchSource"]),
                int(best["branchTarget"]),
                int(best["nodeSource"]),
                int(best["nodeTarget"]),
                aggregate,
                len(group),
                support,
                float(best["geometryScore"]),
                float(best["facing"]),
                float(best["distanceVoxels"]),
                float(best["edgeResidualVoxels"]),
                float(best["fiberAngleDeg"]),
                float(best["normalBendDeg"]),
                float(best["reachRatio"]),
                int(best["endpointSupportedCount"]),
                int(best["contestedEndpointCount"]),
            )
        )
    candidates = np.asarray(output, dtype=CANDIDATE_DTYPE)
    return candidates[np.argsort(candidates["score"])[::-1]]


def _order_condensation(
    cell: np.ndarray,
    raw_depth: np.ndarray,
    branch: np.ndarray,
    node_parity: np.ndarray | None = None,
) -> dict[str, Any]:
    branch_count = int(np.max(branch, initial=-1)) + 1
    by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for node, value in enumerate(cell):
        by_cell[tuple(int(item) for item in value)].append(node)

    branch_gauge = np.ones(branch_count, dtype=np.int8)
    branch_parity_ambiguous = np.zeros(branch_count, dtype=bool)
    parity_constraint_count = 0
    parity_frustrated_constraint_count = 0
    parity_unresolved_constraint_count = 0
    parity_tied_branch_pair_count = 0
    parity_frustrated_branch_pair_count = 0
    parity_component_count = branch_count
    branch_roots = np.arange(branch_count, dtype=np.int32)
    if node_parity is not None:
        node_parity = np.asarray(node_parity, dtype=np.int8)
        if len(node_parity) != len(branch) or np.any(np.abs(node_parity) != 1):
            raise ValueError("node parity must contain one +/-1 value per node")
        parent = np.arange(branch_count, dtype=np.int32)
        size = np.ones(branch_count, dtype=np.int32)
        parity_to_parent = np.ones(branch_count, dtype=np.int8)

        def find_with_parity(index: int) -> tuple[int, int]:
            trail = []
            cursor = index
            parity = 1
            while int(parent[cursor]) != cursor:
                trail.append(cursor)
                parity *= int(parity_to_parent[cursor])
                cursor = int(parent[cursor])
            root = cursor
            running = parity
            for node in trail:
                edge = int(parity_to_parent[node])
                parent[node] = root
                parity_to_parent[node] = running
                running //= edge
            return root, parity

        def merge(first: int, second: int, relative: int) -> bool:
            first_root, first_parity = find_with_parity(first)
            second_root, second_parity = find_with_parity(second)
            if first_root == second_root:
                return first_parity * second_parity == relative
            if int(size[first_root]) < int(size[second_root]):
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
            parity_to_parent[second_root] = relative * first_parity * second_parity
            size[first_root] += size[second_root]
            return True

        pair_votes: dict[tuple[int, int], list[int]] = defaultdict(
            lambda: [0, 0]
        )
        for nodes in by_cell.values():
            for first_offset, first_node in enumerate(nodes):
                first_branch = int(branch[first_node])
                for second_node in nodes[first_offset + 1 :]:
                    second_branch = int(branch[second_node])
                    if first_branch == second_branch:
                        continue
                    parity_constraint_count += 1
                    relative = int(node_parity[first_node] * node_parity[second_node])
                    key = (
                        min(first_branch, second_branch),
                        max(first_branch, second_branch),
                    )
                    pair_votes[key][int(relative > 0)] += 1
        majority_edges = []
        for (first_branch, second_branch), (negative, positive) in pair_votes.items():
            if positive == negative:
                parity_tied_branch_pair_count += 1
                continue
            relative = 1 if positive > negative else -1
            majority_edges.append(
                (
                    abs(positive - negative),
                    positive + negative,
                    first_branch,
                    second_branch,
                    relative,
                )
            )
        for _, _, first_branch, second_branch, relative in sorted(
            majority_edges, reverse=True
        ):
            first_root, _ = find_with_parity(first_branch)
            second_root, _ = find_with_parity(second_branch)
            if first_root != second_root:
                merge(first_branch, second_branch, relative)

        for branch_index in range(branch_count):
            root, gauge = find_with_parity(branch_index)
            branch_roots[branch_index] = root
            branch_gauge[branch_index] = gauge
        parity_component_count = len(np.unique(branch_roots))
        for _, _, first_branch, second_branch, relative in majority_edges:
            if branch_roots[first_branch] != branch_roots[second_branch]:
                continue
            parity_frustrated_branch_pair_count += int(
                int(branch_gauge[first_branch] * branch_gauge[second_branch])
                != relative
            )

    if node_parity is None:
        aligned_depth = np.asarray(raw_depth, dtype=np.float32)
    else:
        aligned_depth = (
            np.asarray(raw_depth, dtype=np.float32)
            * node_parity
            * branch_gauge[branch.astype(np.int64)]
        )
    adjacency = [set() for _ in range(branch_count)]
    reverse = [set() for _ in range(branch_count)]
    for nodes in by_cell.values():
        nodes.sort(key=lambda node: (float(aligned_depth[node]), node))
        for position, source_node in enumerate(nodes):
            source_branch = int(branch[source_node])
            for target_node in nodes[position + 1 :]:
                target_branch = int(branch[target_node])
                if source_branch == target_branch:
                    continue
                if node_parity is not None:
                    if branch_roots[source_branch] != branch_roots[target_branch]:
                        parity_unresolved_constraint_count += 1
                        continue
                    required = int(
                        node_parity[source_node] * node_parity[target_node]
                    )
                    aligned = int(
                        branch_gauge[source_branch]
                        * branch_gauge[target_branch]
                    )
                    if required != aligned:
                        parity_frustrated_constraint_count += 1
                        continue
                adjacency[source_branch].add(target_branch)
                reverse[target_branch].add(source_branch)

    seen = np.zeros(branch_count, dtype=bool)
    finish_order = []
    adjacency_list = [sorted(values) for values in adjacency]
    reverse_list = [sorted(values) for values in reverse]
    for root in range(branch_count):
        if seen[root]:
            continue
        seen[root] = True
        stack = [(root, 0)]
        while stack:
            node, offset = stack[-1]
            if offset < len(adjacency_list[node]):
                neighbor = adjacency_list[node][offset]
                stack[-1] = (node, offset + 1)
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append((neighbor, 0))
            else:
                finish_order.append(node)
                stack.pop()

    branch_scc = np.full(branch_count, -1, dtype=np.int32)
    scc_sizes = []
    for root in reversed(finish_order):
        if branch_scc[root] >= 0:
            continue
        scc_id = len(scc_sizes)
        branch_scc[root] = scc_id
        stack = [root]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            for neighbor in reverse_list[node]:
                if branch_scc[neighbor] < 0:
                    branch_scc[neighbor] = scc_id
                    stack.append(neighbor)
        scc_sizes.append(size)
    scc_sizes_array = np.asarray(scc_sizes, dtype=np.uint32)
    dag = [set() for _ in scc_sizes]
    for source, targets in enumerate(adjacency):
        source_scc = int(branch_scc[source])
        for target in targets:
            target_scc = int(branch_scc[target])
            if source_scc != target_scc:
                dag[source_scc].add(target_scc)
    dag_list = [sorted(values) for values in dag]
    return {
        "branchScc": branch_scc,
        "sccSizes": scc_sizes_array,
        "dag": dag_list,
        "branchGauge": branch_gauge,
        "branchParityAmbiguous": branch_parity_ambiguous,
        "alignedDepth": aligned_depth,
        "stats": {
            "branchCount": branch_count,
            "orderEdgeCount": sum(len(values) for values in adjacency),
            "sccCount": len(scc_sizes),
            "cyclicSccCount": int(np.count_nonzero(scc_sizes_array > 1)),
            "branchCountInCyclicScc": int(np.sum(scc_sizes_array[scc_sizes_array > 1])),
            "largestCyclicSccSize": int(np.max(scc_sizes_array, initial=0)),
            "parityConstraintCount": parity_constraint_count,
            "parityFrustratedConstraintCount": parity_frustrated_constraint_count,
            "parityUnresolvedConstraintCount": parity_unresolved_constraint_count,
            "parityTiedBranchPairCount": parity_tied_branch_pair_count,
            "parityFrustratedBranchPairCount": (
                parity_frustrated_branch_pair_count
            ),
            "parityInteractionComponentCount": parity_component_count,
            "parityAmbiguousBranchCount": int(
                np.count_nonzero(branch_parity_ambiguous)
            ),
            "constraint": (
                "unsigned branch gauges are aligned only through shared-cell parity; "
                "parity-frustrated and cyclic order regions remain explicit "
                "ambiguities and are deferred rather than collapsed"
            ),
        },
    }


def _branch_cell_sets(cell: np.ndarray, branch: np.ndarray, branch_count: int) -> list[set[int]]:
    shape = np.max(cell, axis=0).astype(np.int64) + 1
    code = (
        cell[:, 0].astype(np.int64)
        + shape[0] * (cell[:, 1].astype(np.int64) + shape[1] * cell[:, 2].astype(np.int64))
    )
    output = [set() for _ in range(branch_count)]
    for branch_index, cell_code in zip(branch, code):
        output[int(branch_index)].add(int(cell_code))
    return output


def _solve_candidates(
    candidates: np.ndarray,
    threshold: float,
    order: dict[str, Any],
    branch_cells: list[set[int]],
) -> dict[str, Any]:
    branch_scc = order["branchScc"]
    scc_sizes = order["sccSizes"]
    dag = order["dag"]
    parity_ambiguous = order.get(
        "branchParityAmbiguous", np.zeros(len(branch_scc), dtype=bool)
    )
    scc_count = len(scc_sizes)
    parent = np.arange(scc_count, dtype=np.int32)
    group_size = np.ones(scc_count, dtype=np.int32)
    members = [{index} for index in range(scc_count)]
    cells = [set() for _ in range(scc_count)]
    for branch_index, values in enumerate(branch_cells):
        cells[int(branch_scc[branch_index])].update(values)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def reachable(source_root: int, target_root: int) -> bool:
        source_root, target_root = find(source_root), find(target_root)
        if source_root == target_root:
            return True
        queue = deque([source_root])
        seen = {source_root}
        while queue:
            current = queue.popleft()
            for scc_id in members[current]:
                for child in dag[scc_id]:
                    child_root = find(child)
                    if child_root == target_root:
                        return True
                    if child_root not in seen:
                        seen.add(child_root)
                        queue.append(child_root)
        return False

    decisions = np.full(len(candidates), DECISION_BELOW_THRESHOLD, dtype=np.uint8)
    for candidate_index in np.argsort(candidates["score"])[::-1]:
        candidate = candidates[candidate_index]
        if float(candidate["score"]) < threshold:
            continue
        if int(candidate["endpointSupportedCount"]) < 2:
            decisions[candidate_index] = DECISION_MATERIAL_DEFERRED
            continue
        source_branch = int(candidate["branchSource"])
        target_branch = int(candidate["branchTarget"])
        if bool(parity_ambiguous[source_branch]) or bool(
            parity_ambiguous[target_branch]
        ):
            decisions[candidate_index] = DECISION_ORDER_AMBIGUOUS
            continue
        source_scc = int(branch_scc[source_branch])
        target_scc = int(branch_scc[target_branch])
        if int(scc_sizes[source_scc]) > 1 or int(scc_sizes[target_scc]) > 1:
            decisions[candidate_index] = DECISION_ORDER_AMBIGUOUS
            continue
        source_root, target_root = find(source_scc), find(target_scc)
        if source_root == target_root:
            decisions[candidate_index] = DECISION_REDUNDANT
            continue
        if reachable(source_root, target_root) or reachable(target_root, source_root):
            decisions[candidate_index] = DECISION_ORDER_BLOCKED
            continue
        if not cells[source_root].isdisjoint(cells[target_root]):
            decisions[candidate_index] = DECISION_CELL_COLLISION
            continue
        if int(group_size[source_root]) < int(group_size[target_root]):
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        group_size[source_root] += group_size[target_root]
        members[source_root].update(members[target_root])
        members[target_root].clear()
        cells[source_root].update(cells[target_root])
        cells[target_root].clear()
        decisions[candidate_index] = DECISION_RETAINED

    scc_root = np.asarray([find(index) for index in range(scc_count)], dtype=np.int32)
    # An SCC is an ambiguity set, not an association. Only singleton SCCs may
    # inherit the association solve's DSU root; every branch in a cyclic SCC
    # keeps its own explicit identity.
    branch_root = np.empty(len(branch_scc), dtype=np.int64)
    for branch_index, scc_id in enumerate(branch_scc):
        if int(scc_sizes[int(scc_id)]) > 1:
            branch_root[branch_index] = scc_count + branch_index
        else:
            branch_root[branch_index] = int(scc_root[int(scc_id)])
    _, branch_group = np.unique(branch_root, return_inverse=True)
    return {
        "decisions": decisions,
        "branchGroup": branch_group.astype(np.uint32),
        "stats": {
            "threshold": threshold,
            "candidateCount": len(candidates),
            "decisionCounts": {
                DECISION_NAMES[value]: int(np.count_nonzero(decisions == value))
                for value in DECISION_NAMES
            },
            "retainedMergeCount": int(np.count_nonzero(decisions == DECISION_RETAINED)),
            "acceptedOrRedundantCount": int(
                np.count_nonzero(
                    (decisions == DECISION_RETAINED)
                    | (decisions == DECISION_REDUNDANT)
                )
            ),
        },
    }


def _subwindows(main_window: dict[str, Any], size_xy: tuple[int, int]) -> list[dict[str, Any]]:
    low = np.asarray(main_window["originCellXYZ"], dtype=np.int32)
    high = np.asarray(main_window["stopCellXYZExclusive"], dtype=np.int32)
    width = high - low
    sub_width = np.asarray(
        [min(size_xy[0], width[0]), min(size_xy[1], width[1]), width[2]],
        dtype=np.int32,
    )
    offsets_x = sorted({0, int(width[0] - sub_width[0])})
    offsets_y = sorted({0, int(width[1] - sub_width[1])})
    output = []
    for y_offset in offsets_y:
        for x_offset in offsets_x:
            origin = low + np.asarray([x_offset, y_offset, 0], dtype=np.int32)
            stop = origin + sub_width
            output.append(
                {
                    "name": f"x{x_offset}-y{y_offset}",
                    "originCellXYZ": origin.tolist(),
                    "stopCellXYZExclusive": stop.tolist(),
                    "shapeCellsXYZ": sub_width.tolist(),
                }
            )
    return output


def _accepted_by_pair(
    candidates: np.ndarray, decisions: np.ndarray
) -> dict[tuple[int, int], bool]:
    return {
        (int(value["branchSource"]), int(value["branchTarget"])): int(decision)
        in (DECISION_RETAINED, DECISION_REDUNDANT)
        for value, decision in zip(candidates, decisions)
    }


def _association_groups(
    branch_group: np.ndarray,
    branch_flake_count: np.ndarray,
    exact_geometry: list[dict[str, Any]] | None = None,
) -> dict[str, Any]:
    linked_branch = branch_flake_count >= 2
    group_count = int(np.max(branch_group)) + 1 if len(branch_group) else 0
    branch_counts = np.bincount(
        branch_group, weights=linked_branch.astype(np.int32), minlength=group_count
    ).astype(np.int32)
    flake_counts = np.bincount(
        branch_group, weights=branch_flake_count, minlength=group_count
    ).astype(np.int32)
    linked_groups = np.unique(branch_group[linked_branch])
    merged_groups = np.flatnonzero(branch_counts >= 2)
    exact_by_group = {
        int(value["associationId"]): value for value in (exact_geometry or [])
    }
    top = []
    for group_index in np.argsort(flake_counts)[::-1]:
        if int(branch_counts[group_index]) < 2 or len(top) >= 12:
            continue
        value = {
            "associationId": int(group_index),
            "branchCount": int(branch_counts[group_index]),
            "flakeCount": int(flake_counts[group_index]),
        }
        if int(group_index) in exact_by_group:
            value["exactGeometry"] = exact_by_group[int(group_index)]
        top.append(value)
    return {
        "initialLinkedBranchCount": int(np.count_nonzero(linked_branch)),
        "remainingLinkedBranchGroupCount": len(linked_groups),
        "branchReductionCount": int(np.count_nonzero(linked_branch)) - len(linked_groups),
        "mergedAssociationCount": len(merged_groups),
        "associatedBranchCount": int(np.sum(branch_counts[merged_groups])),
        "largestAssociationBranchCount": int(np.max(branch_counts, initial=0)),
        "largestAssociationFlakeCount": int(np.max(flake_counts[merged_groups], initial=0)),
        "topAssociations": top,
    }


def _run_overlap_sweep(
    main_candidates: np.ndarray,
    sub_candidates: list[np.ndarray],
    threshold: float,
    order: dict[str, Any],
    branch_cells: list[set[int]],
    minimum_observations: int,
) -> dict[str, Any]:
    main_solve = _solve_candidates(main_candidates, threshold, order, branch_cells)
    sub_solves = [
        _solve_candidates(values, threshold, order, branch_cells)
        for values in sub_candidates
    ]
    sub_maps = [
        _accepted_by_pair(values, solve["decisions"])
        for values, solve in zip(sub_candidates, sub_solves)
    ]
    main_accepted = (
        (main_solve["decisions"] == DECISION_RETAINED)
        | (main_solve["decisions"] == DECISION_REDUNDANT)
    )
    observations = np.zeros(len(main_candidates), dtype=np.uint8)
    accepted_observations = np.zeros(len(main_candidates), dtype=np.uint8)
    stable = np.zeros(len(main_candidates), dtype=bool)
    disagreement = np.zeros(len(main_candidates), dtype=bool)
    for index, candidate in enumerate(main_candidates):
        pair = (int(candidate["branchSource"]), int(candidate["branchTarget"]))
        values = [mapping[pair] for mapping in sub_maps if pair in mapping]
        observations[index] = len(values)
        accepted_observations[index] = sum(values)
        unanimous_accepted = bool(values) and all(values)
        stable[index] = (
            bool(main_accepted[index])
            and len(values) >= minimum_observations
            and unanimous_accepted
        )
        if values:
            disagreement[index] = (
                any(values) != all(values)
                or bool(main_accepted[index]) != unanimous_accepted
            )
    final_candidates = main_candidates[stable]
    final_solve = _solve_candidates(final_candidates, threshold, order, branch_cells)
    final_pair_decision = {
        (int(value["branchSource"]), int(value["branchTarget"])): int(decision)
        for value, decision in zip(final_candidates, final_solve["decisions"])
    }
    final_decisions = np.full(len(main_candidates), DECISION_BELOW_THRESHOLD, dtype=np.uint8)
    for index, candidate in enumerate(main_candidates):
        pair = (int(candidate["branchSource"]), int(candidate["branchTarget"]))
        if pair in final_pair_decision:
            final_decisions[index] = final_pair_decision[pair]
    return {
        "main": main_solve,
        "sub": sub_solves,
        "observations": observations,
        "acceptedObservations": accepted_observations,
        "stable": stable,
        "disagreement": disagreement,
        "finalDecisions": final_decisions,
        "final": final_solve,
        "stats": {
            "threshold": threshold,
            "mainRetainedMergeCount": main_solve["stats"]["retainedMergeCount"],
            "stableCandidateCount": int(np.count_nonzero(stable)),
            "overlapDisagreementCount": int(np.count_nonzero(disagreement)),
            "insufficientOverlapObservationCount": int(
                np.count_nonzero(observations < minimum_observations)
            ),
            "preExactRetainedMergeCount": final_solve["stats"]["retainedMergeCount"],
            "subwindowRetainedMergeCounts": [
                value["stats"]["retainedMergeCount"] for value in sub_solves
            ],
        },
    }


def _exact_carrier_stats(
    member_flakes: list[dict[str, Any]],
    settings: dict[str, Any],
) -> dict[str, float]:
    carrier = build_mls_carrier(
        member_flakes,
        pixel_step=float(settings["exactCarrierPixelStepVoxels"]),
        maximum_pixels=int(settings["exactCarrierMaximumPixelsPerAxis"]),
    )
    stats = carrier["stats"]
    return {
        "medianHeightResidualVoxels": float(
            stats["medianNodeHeightResidualVoxels"]
        ),
        "p90HeightResidualVoxels": float(
            stats["p90NodeHeightResidualVoxels"]
        ),
        "medianNormalResidualDeg": float(stats["medianNodeNormalResidualDeg"]),
        "p90NormalResidualDeg": float(stats["p90NodeNormalResidualDeg"]),
    }


def _audit_exact_candidate_pairs(
    candidates: np.ndarray,
    stable: np.ndarray,
    flakes: list[dict[str, Any]],
    branch: np.ndarray,
    settings: dict[str, Any],
) -> dict[str, np.ndarray]:
    median_height = np.full(len(candidates), np.nan, dtype=np.float32)
    p90_height = np.full(len(candidates), np.nan, dtype=np.float32)
    median_normal = np.full(len(candidates), np.nan, dtype=np.float32)
    p90_normal = np.full(len(candidates), np.nan, dtype=np.float32)
    passed = np.zeros(len(candidates), dtype=bool)
    for candidate_index in np.flatnonzero(stable):
        candidate = candidates[candidate_index]
        member = (branch == int(candidate["branchSource"])) | (
            branch == int(candidate["branchTarget"])
        )
        exact = _exact_carrier_stats(
            [flakes[int(index)] for index in np.flatnonzero(member)], settings
        )
        median_height[candidate_index] = exact["medianHeightResidualVoxels"]
        p90_height[candidate_index] = exact["p90HeightResidualVoxels"]
        median_normal[candidate_index] = exact["medianNormalResidualDeg"]
        p90_normal[candidate_index] = exact["p90NormalResidualDeg"]
        passed[candidate_index] = (
            exact["medianHeightResidualVoxels"]
            <= float(settings["maximumExactMedianHeightResidualVoxels"])
            and exact["medianNormalResidualDeg"]
            <= float(settings["maximumExactMedianNormalResidualDeg"])
        )
    return {
        "medianHeightResidualVoxels": median_height,
        "p90HeightResidualVoxels": p90_height,
        "medianNormalResidualDeg": median_normal,
        "p90NormalResidualDeg": p90_normal,
        "passed": passed,
    }


def _audit_exact_association_groups(
    branch_group: np.ndarray,
    flakes: list[dict[str, Any]],
    branch: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    if not len(branch_group):
        return []
    branch_counts = np.bincount(branch_group)
    merged_groups = np.flatnonzero(branch_counts >= 2)
    flake_group = branch_group[branch]
    output = []
    for group_index in merged_groups:
        member_indices = np.flatnonzero(flake_group == group_index)
        exact = _exact_carrier_stats(
            [flakes[int(index)] for index in member_indices], settings
        )
        output.append(
            {
                "associationId": int(group_index),
                "branchCount": int(branch_counts[group_index]),
                "flakeCount": len(member_indices),
                **exact,
                "gatesPass": bool(
                    exact["medianHeightResidualVoxels"]
                    <= float(settings["maximumExactMedianHeightResidualVoxels"])
                    and exact["medianNormalResidualDeg"]
                    <= float(settings["maximumExactMedianNormalResidualDeg"])
                ),
            }
        )
    return output


def _solve_with_exact_geometry(
    candidates: np.ndarray,
    stable: np.ndarray,
    flakes: list[dict[str, Any]],
    branch: np.ndarray,
    threshold: float,
    order: dict[str, Any],
    branch_cells: list[set[int]],
    settings: dict[str, Any],
) -> dict[str, Any]:
    pair_audit = _audit_exact_candidate_pairs(
        candidates, stable, flakes, branch, settings
    )
    eligible = stable & pair_audit["passed"]
    group_pruned = np.zeros(len(candidates), dtype=bool)
    rounds = []
    final_solve: dict[str, Any] | None = None
    final_indices = np.empty(0, dtype=np.int64)
    final_group_audit: list[dict[str, Any]] = []

    for round_index in range(int(np.count_nonzero(eligible)) + 1):
        indices = np.flatnonzero(eligible)
        solve = _solve_candidates(
            candidates[indices], threshold, order, branch_cells
        )
        group_audit = _audit_exact_association_groups(
            solve["branchGroup"], flakes, branch, settings
        )
        failing_group_ids = {
            int(value["associationId"])
            for value in group_audit
            if not bool(value["gatesPass"])
        }
        removed = []
        for group_index in sorted(failing_group_ids):
            retained = solve["decisions"] == DECISION_RETAINED
            same_group = (
                solve["branchGroup"][candidates[indices]["branchSource"]]
                == group_index
            ) & (
                solve["branchGroup"][candidates[indices]["branchTarget"]]
                == group_index
            )
            local_candidates = np.flatnonzero(retained & same_group)
            if not len(local_candidates):
                continue
            local_index = int(
                local_candidates[
                    np.argmin(candidates[indices[local_candidates]]["score"])
                ]
            )
            candidate_index = int(indices[local_index])
            removed.append(
                {
                    "associationId": group_index,
                    "candidateIndex": candidate_index,
                    "branchSource": int(candidates[candidate_index]["branchSource"]),
                    "branchTarget": int(candidates[candidate_index]["branchTarget"]),
                    "score": round(float(candidates[candidate_index]["score"]), 6),
                }
            )
        rounds.append(
            {
                "round": round_index,
                "eligibleCandidateCount": len(indices),
                "retainedMergeCount": solve["stats"]["retainedMergeCount"],
                "mergedAssociationCount": len(group_audit),
                "failingAssociationCount": len(failing_group_ids),
                "removedWeakestJoins": removed,
            }
        )
        final_solve = solve
        final_indices = indices
        final_group_audit = group_audit
        if not failing_group_ids:
            break
        if not removed:
            raise RuntimeError(
                "an exact-geometry failure has no retained construction edge to prune"
            )
        for value in removed:
            candidate_index = int(value["candidateIndex"])
            eligible[candidate_index] = False
            group_pruned[candidate_index] = True
    else:
        raise RuntimeError("exact-geometry pruning did not converge")

    if final_solve is None:
        raise RuntimeError("exact-geometry solve did not run")
    decisions = np.full(len(candidates), DECISION_BELOW_THRESHOLD, dtype=np.uint8)
    decisions[stable & ~pair_audit["passed"]] = DECISION_EXACT_PAIR_DEFERRED
    decisions[group_pruned] = DECISION_EXACT_GROUP_PRUNED
    decisions[final_indices] = final_solve["decisions"]
    return {
        "pairAudit": pair_audit,
        "groupPruned": group_pruned,
        "rounds": rounds,
        "decisions": decisions,
        "final": final_solve,
        "groupAudit": final_group_audit,
        "stats": {
            "preExactStableCandidateCount": int(np.count_nonzero(stable)),
            "exactPairPassCount": int(np.count_nonzero(stable & pair_audit["passed"])),
            "exactPairDeferredCount": int(
                np.count_nonzero(stable & ~pair_audit["passed"])
            ),
            "exactGroupPrunedCount": int(np.count_nonzero(group_pruned)),
            "exactPruningRoundCount": sum(
                bool(value["removedWeakestJoins"]) for value in rounds
            ),
            "finalRetainedMergeCount": final_solve["stats"]["retainedMergeCount"],
            "finalMergedAssociationCount": len(final_group_audit),
            "finalExactFailureCount": int(
                np.count_nonzero(
                    [not bool(value["gatesPass"]) for value in final_group_audit]
                )
            ),
            "pairMedianHeightResidualVoxels": _quantiles(
                pair_audit["medianHeightResidualVoxels"][stable]
            ),
            "pairMedianNormalResidualDeg": _quantiles(
                pair_audit["medianNormalResidualDeg"][stable]
            ),
        },
    }


def associate_monotone_branches(
    output_root: str | Path,
    force: bool = False,
    settings: dict[str, Any] | None = None,
    window_origin_cell_xyz: tuple[int, int, int] | list[int] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    suffix = window_artifact_suffix(window_origin_cell_xyz)
    monotone_stem = f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}"
    monotone_summary_path = root / f"{monotone_stem}.json"
    monotone_path = root / f"{monotone_stem}.npz"
    monotone_summary = json.loads(monotone_summary_path.read_text())
    input_paths = [
        monotone_path,
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-claims.npy",
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-intervals.npy",
    ]
    input_paths.extend(
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(
            monotone_summary["window"]["originCellXYZ"][2],
            monotone_summary["window"]["stopCellXYZExclusive"][2],
        )
    )
    identity: dict[str, Any] = {
        "version": BRANCH_ASSOCIATION_VERSION,
        "monotoneIdentity": monotone_summary["identity"],
        "settings": resolved,
        "inputArtifacts": [_content_identity(path) for path in input_paths],
    }
    if window_origin_cell_xyz is not None:
        identity["windowOriginCellXYZ"] = [
            int(value) for value in window_origin_cell_xyz
        ]
    stem = f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(monotone_path) as payload:
        arrays = {key: np.asarray(payload[key]) for key in payload.files}
    flakes = _load_artifact_flakes(root, arrays)
    source_z = arrays["sourceZIndex"].astype(np.int32)
    source_id = arrays["sourceFlakeId"].astype(np.int32)
    material_supported, contested = _node_material_state(
        root, source_z, source_id
    )
    endpoint, outward, centers = _endpoint_geometry(
        flakes, arrays, int(resolved["minimumBranchSize"])
    )
    cell = arrays["cellIndex"].astype(np.int32)
    branch = arrays["component"].astype(np.int32)
    pairs = _candidate_endpoint_pairs(
        endpoint,
        cell,
        branch,
        centers,
        outward,
        int(resolved["maximumGapCells"]),
        float(resolved["maximumEndpointDistanceVoxels"]),
        float(resolved["minimumFacingCosine"]),
    )
    hits = _score_endpoint_hits(
        flakes,
        pairs,
        branch,
        centers,
        outward,
        material_supported,
        contested,
        float(resolved["edgePaddingVoxels"]),
        float(resolved["minimumHitScore"]),
    )
    main_candidates = _aggregate_candidates(hits)
    order = _order_condensation(
        cell,
        arrays["rawDepth"],
        branch,
        arrays["branchParity"],
    )
    branch_count = int(np.max(branch, initial=-1)) + 1
    branch_cells = _branch_cell_sets(cell, branch, branch_count)
    subwindows = _subwindows(
        monotone_summary["window"],
        tuple(int(value) for value in resolved["subwindowCellsXY"]),
    )
    sub_candidates = [
        _aggregate_candidates(hits, _inside_window(cell, window))
        for window in subwindows
    ]
    sweeps = []
    selected = None
    for threshold in resolved["thresholdSweep"]:
        sweep = _run_overlap_sweep(
            main_candidates,
            sub_candidates,
            float(threshold),
            order,
            branch_cells,
            int(resolved["minimumOverlapObservations"]),
        )
        sweeps.append(sweep["stats"])
        if abs(float(threshold) - float(resolved["selectedThreshold"])) < 1.0e-8:
            selected = sweep
    if selected is None:
        raise ValueError("selectedThreshold must be included in thresholdSweep")

    exact = _solve_with_exact_geometry(
        main_candidates,
        selected["stable"],
        flakes,
        branch,
        float(resolved["selectedThreshold"]),
        order,
        branch_cells,
        resolved,
    )
    branch_flake_count = np.bincount(branch, minlength=branch_count).astype(np.int32)
    group_stats = _association_groups(
        exact["final"]["branchGroup"],
        branch_flake_count,
        exact["groupAudit"],
    )
    final_accepted = (
        (exact["decisions"] == DECISION_RETAINED)
        | (exact["decisions"] == DECISION_REDUNDANT)
    )
    final_candidates = main_candidates[final_accepted]
    flake_association = exact["final"]["branchGroup"][branch]
    _atomic_npz(
        artifact_path,
        hitBranchSource=hits["branchSource"],
        hitBranchTarget=hits["branchTarget"],
        hitNodeSource=hits["nodeSource"],
        hitNodeTarget=hits["nodeTarget"],
        hitScore=hits["score"],
        hitGeometryScore=hits["geometryScore"],
        hitFacing=hits["facing"],
        hitDistanceVoxels=hits["distanceVoxels"],
        candidateBranchSource=main_candidates["branchSource"],
        candidateBranchTarget=main_candidates["branchTarget"],
        candidateNodeSource=main_candidates["nodeSource"],
        candidateNodeTarget=main_candidates["nodeTarget"],
        candidateScore=main_candidates["score"],
        candidateSupport=main_candidates["support"],
        candidateEndpointSupportedCount=main_candidates["endpointSupportedCount"],
        candidateMainDecision=selected["main"]["decisions"],
        candidateOverlapObservationCount=selected["observations"],
        candidateOverlapAcceptedCount=selected["acceptedObservations"],
        candidateOverlapDisagreement=selected["disagreement"],
        candidateStable=selected["stable"],
        candidateExactPairMedianHeightResidualVoxels=exact["pairAudit"][
            "medianHeightResidualVoxels"
        ],
        candidateExactPairP90HeightResidualVoxels=exact["pairAudit"][
            "p90HeightResidualVoxels"
        ],
        candidateExactPairMedianNormalResidualDeg=exact["pairAudit"][
            "medianNormalResidualDeg"
        ],
        candidateExactPairP90NormalResidualDeg=exact["pairAudit"][
            "p90NormalResidualDeg"
        ],
        candidateExactPairPass=exact["pairAudit"]["passed"],
        candidateExactGroupPruned=exact["groupPruned"],
        candidateFinalDecision=exact["decisions"],
        branchScc=order["branchScc"],
        sccSize=order["sccSizes"],
        branchOrderGauge=order["branchGauge"],
        branchOrderParityAmbiguous=order["branchParityAmbiguous"],
        branchAssociation=exact["final"]["branchGroup"],
        flakeAssociation=flake_association.astype(np.uint32),
    )
    retained_score = final_candidates["score"]
    retained_branch_ids = np.unique(
        np.r_[final_candidates["branchSource"], final_candidates["branchTarget"]]
    ) if len(final_candidates) else np.empty(0, dtype=np.uint32)
    result = {
        "identity": identity,
        "contract": {
            "scope": (
                "chunk-local association of existing primary-family surface branches; "
                "no dense global geometry solve"
            ),
            "joinEvidence": (
                "facing branch endpoints scored by finite-patch edge agreement, "
                "transported fiber direction, normal bend, reach, and flake quality"
            ),
            "materialRole": (
                "endpoint material support is a one-way eligibility gate and never "
                "adds join score; continuity through contested bulk is not join evidence"
            ),
            "orderRole": (
                "branch-relative gauges are aligned only by parity votes in shared "
                "cells; gauge-unresolved, parity-frustrated, cyclic-order, and "
                "overlap-disagreement constraints are explicitly deferred"
            ),
            "exactGeometryRole": (
                "the same-sample MLS carrier fit is a construction gate, not an "
                "independent validation test: incoherent pairs are deferred, then "
                "the weakest retained join is removed from each incoherent transitive "
                "association until every output association passes"
            ),
            "identityMeaning": (
                "association IDs remain local surface hypotheses, not physical sheets"
            ),
        },
        "window": monotone_summary["window"],
        "subwindows": subwindows,
        "settings": resolved,
        "sweeps": sweeps,
        "selected": {
            **selected["stats"],
            **exact["stats"],
            "mainDecisionCounts": selected["main"]["stats"]["decisionCounts"],
            "preExactDecisionCounts": selected["final"]["stats"]["decisionCounts"],
            "finalDecisionCounts": {
                DECISION_NAMES[value]: int(
                    np.count_nonzero(exact["decisions"] == value)
                )
                for value in DECISION_NAMES
            },
            "exactPruningRounds": exact["rounds"],
            "exactAssociationGeometry": exact["groupAudit"],
            "associationGroups": group_stats,
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "nodeCount": len(flakes),
            "endpointCount": len(endpoint),
            "spatialEndpointPairCount": len(pairs),
            "geometryHitCount": len(hits),
            "candidateBranchPairCount": len(main_candidates),
            "candidateWithMaterialSupportCount": int(
                np.count_nonzero(main_candidates["endpointSupportedCount"] == 2)
            ),
            "candidateInContestedBulkCount": int(
                np.count_nonzero(main_candidates["contestedEndpointCount"] == 2)
            ),
            "candidateScore": _quantiles(main_candidates["score"]),
            "order": order["stats"],
            "selectedRetainedScore": _quantiles(retained_score),
            "selectedRetainedBranchCount": len(retained_branch_ids),
            "explicitUnassociatedLinkedBranchCount": (
                group_stats["initialLinkedBranchCount"]
                - group_stats["associatedBranchCount"]
            ),
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
