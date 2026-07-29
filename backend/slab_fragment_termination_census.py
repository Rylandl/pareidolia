from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .slab_branch_association import (
    BRANCH_ASSOCIATION_VERSION,
    DECISION_CELL_COLLISION as LOCAL_DECISION_CELL_COLLISION,
    DECISION_EXACT_GROUP_PRUNED as LOCAL_DECISION_EXACT_GROUP_PRUNED,
    DECISION_EXACT_PAIR_DEFERRED as LOCAL_DECISION_EXACT_PAIR_DEFERRED,
    DECISION_MATERIAL_DEFERRED as LOCAL_DECISION_MATERIAL_DEFERRED,
    DECISION_ORDER_AMBIGUOUS as LOCAL_DECISION_ORDER_AMBIGUOUS,
    DECISION_ORDER_BLOCKED as LOCAL_DECISION_ORDER_BLOCKED,
    DECISION_REDUNDANT as LOCAL_DECISION_REDUNDANT,
    DECISION_RETAINED as LOCAL_DECISION_RETAINED,
    _node_material_state,
)
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_global_boundary_candidates import (
    GLOBAL_BOUNDARY_CANDIDATE_VERSION,
    _boundary_outward,
    _load_global_node_geometry,
)
from .slab_global_branch_association import (
    DECISION_CELL_COLLISION,
    DECISION_EXACT_GROUP_PRUNED,
    DECISION_EXACT_PAIR_DEFERRED,
    DECISION_INPUT_BRANCH_CARRIER_DEFERRED,
    DECISION_INTEGRITY_QUARANTINED,
    DECISION_REDUNDANT,
    DECISION_RETAINED,
    GLOBAL_BRANCH_ASSOCIATION_VERSION,
)
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


FRAGMENT_TERMINATION_CENSUS_VERSION = 1

CATEGORY_NO_COMPATIBLE_CANDIDATE = 0
CATEGORY_WEAK_GEOMETRY = 1
CATEGORY_MATERIAL_DEFERRED = 2
CATEGORY_ORDER_BLOCKED = 3
CATEGORY_OVERLAP_UNRESOLVED = 4
CATEGORY_ORDER_UNRESOLVED = 5
CATEGORY_GEOMETRY_REJECTED = 6
CATEGORY_COLLISION_BLOCKED = 7
CATEGORY_INTEGRITY_REJECTED = 8
CATEGORY_CONTINUED = 9
CATEGORY_NAMES = {
    CATEGORY_NO_COMPATIBLE_CANDIDATE: "no-compatible-candidate",
    CATEGORY_WEAK_GEOMETRY: "weak-geometry",
    CATEGORY_MATERIAL_DEFERRED: "material-deferred",
    CATEGORY_ORDER_BLOCKED: "order-blocked",
    CATEGORY_OVERLAP_UNRESOLVED: "overlap-unresolved",
    CATEGORY_ORDER_UNRESOLVED: "order-unresolved",
    CATEGORY_GEOMETRY_REJECTED: "geometry-rejected",
    CATEGORY_COLLISION_BLOCKED: "collision-blocked",
    CATEGORY_INTEGRITY_REJECTED: "integrity-rejected",
    CATEGORY_CONTINUED: "continued",
}
CATEGORY_PRIORITY = np.asarray([0, 1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=np.int8)

EVIDENCE_NONE = 0
EVIDENCE_LOCAL_WINDOW = 1
EVIDENCE_GLOBAL_BOUNDARY = 2
EVIDENCE_COMPLETE_BRANCH = 3
EVIDENCE_NAMES = {
    EVIDENCE_NONE: "none",
    EVIDENCE_LOCAL_WINDOW: "local-window",
    EVIDENCE_GLOBAL_BOUNDARY: "global-boundary",
    EVIDENCE_COMPLETE_BRANCH: "complete-branch",
}

DENSE_ACUS_CATEGORIES = {
    CATEGORY_NO_COMPATIBLE_CANDIDATE,
    CATEGORY_WEAK_GEOMETRY,
    CATEGORY_OVERLAP_UNRESOLVED,
}
ORDER_REVIEW_CATEGORIES = {
    CATEGORY_ORDER_UNRESOLVED,
    CATEGORY_ORDER_BLOCKED,
}
GEOMETRY_REVIEW_CATEGORIES = {
    CATEGORY_GEOMETRY_REJECTED,
    CATEGORY_COLLISION_BLOCKED,
    CATEGORY_INTEGRITY_REJECTED,
}

DEFAULT_SETTINGS: dict[str, Any] = {
    "minimumAssociationNodeCount": 25,
    "clusterRadiusCells": 1,
    "minimumClusterOutwardCosine": 0.50,
    "targetDistanceVoxels": 32.0,
    "targetSampleHalfWidthVoxels": 16,
    "minimumTargetMaterialFraction": 0.35,
    "maximumCtSampledClusterCount": 512,
    "maximumDenseAcusTargetCount": 128,
    "maximumTargetsPerAssociation": 2,
    "maximumReviewTargetCount": 128,
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


def _identity_node_indexes(
    node_identity: np.ndarray,
    requested_identity: np.ndarray,
    description: str,
) -> np.ndarray:
    requested_identity = np.asarray(requested_identity, dtype=np.uint64)
    position = np.searchsorted(node_identity, requested_identity)
    present = position < len(node_identity)
    if np.any(present):
        present[present] &= (
            node_identity[position[present]] == requested_identity[present]
        )
    if not np.all(present):
        raise ValueError(f"{description} identity is absent globally")
    return position.astype(np.uint32)


def _local_candidate_category(
    main_decision: int,
    final_decision: int,
    stable: bool,
) -> int:
    if final_decision in (
        LOCAL_DECISION_EXACT_PAIR_DEFERRED,
        LOCAL_DECISION_EXACT_GROUP_PRUNED,
    ):
        return CATEGORY_GEOMETRY_REJECTED
    if final_decision in (LOCAL_DECISION_RETAINED, LOCAL_DECISION_REDUNDANT):
        return CATEGORY_CONTINUED
    if main_decision == LOCAL_DECISION_MATERIAL_DEFERRED:
        return CATEGORY_MATERIAL_DEFERRED
    if main_decision == LOCAL_DECISION_ORDER_AMBIGUOUS:
        return CATEGORY_ORDER_UNRESOLVED
    if main_decision == LOCAL_DECISION_ORDER_BLOCKED:
        return CATEGORY_ORDER_BLOCKED
    if main_decision == LOCAL_DECISION_CELL_COLLISION:
        return CATEGORY_COLLISION_BLOCKED
    if main_decision in (LOCAL_DECISION_RETAINED, LOCAL_DECISION_REDUNDANT):
        return (
            CATEGORY_GEOMETRY_REJECTED
            if stable
            else CATEGORY_OVERLAP_UNRESOLVED
        )
    return CATEGORY_WEAK_GEOMETRY


def _global_candidate_category(final_decision: int) -> int:
    if final_decision in (
        DECISION_INPUT_BRANCH_CARRIER_DEFERRED,
        DECISION_EXACT_PAIR_DEFERRED,
        DECISION_EXACT_GROUP_PRUNED,
    ):
        return CATEGORY_GEOMETRY_REJECTED
    if final_decision == DECISION_CELL_COLLISION:
        return CATEGORY_COLLISION_BLOCKED
    if final_decision == DECISION_INTEGRITY_QUARANTINED:
        return CATEGORY_INTEGRITY_REJECTED
    if final_decision in (DECISION_RETAINED, DECISION_REDUNDANT):
        return CATEGORY_CONTINUED
    raise ValueError("unknown complete-branch candidate decision")


def _cluster_termination_regions(
    association: np.ndarray,
    cell: np.ndarray,
    outward: np.ndarray,
    active: np.ndarray,
    radius_cells: int,
    minimum_outward_cosine: float,
) -> np.ndarray:
    association = np.asarray(association, dtype=np.int64)
    cell = np.asarray(cell, dtype=np.int32)
    outward = np.asarray(outward, dtype=np.float32)
    active = np.asarray(active, dtype=bool)
    if not (
        len(association) == len(cell) == len(outward) == len(active)
        and cell.shape == outward.shape
        and cell.shape[1] == 3
    ):
        raise ValueError("termination clustering arrays must be aligned Nx3 data")
    selected = np.flatnonzero(active)
    parent = np.arange(len(selected), dtype=np.int32)
    size = np.ones(len(selected), dtype=np.int32)

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def merge(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if int(size[first_root]) < int(size[second_root]):
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]

    radius = max(0, int(radius_cells))
    offsets = [
        (dx, dy, dz)
        for dz in range(-radius, radius + 1)
        for dy in range(-radius, radius + 1)
        for dx in range(-radius, radius + 1)
    ]
    buckets: dict[tuple[int, int, int, int], int] = {}
    for local_index, endpoint_index in enumerate(selected):
        association_id = int(association[endpoint_index])
        x, y, z = (int(value) for value in cell[endpoint_index])
        for dx, dy, dz in offsets:
            neighbor = buckets.get((association_id, x + dx, y + dy, z + dz))
            if neighbor is None:
                continue
            neighbor_endpoint = int(selected[neighbor])
            if (
                float(np.dot(outward[endpoint_index], outward[neighbor_endpoint]))
                >= minimum_outward_cosine
            ):
                merge(local_index, neighbor)
        key = (association_id, x, y, z)
        if key in buckets:
            raise ValueError("one final association contains a repeated Acus cell")
        buckets[key] = local_index

    roots = np.asarray([find(index) for index in range(len(selected))], dtype=np.int32)
    _, cluster = np.unique(roots, return_inverse=True)
    output = np.full(len(active), -1, dtype=np.int32)
    output[selected] = cluster.astype(np.int32)
    return output


def _target_ct_metrics(
    source: np.ndarray,
    target_xyz: np.ndarray,
    half_width: int,
    air_threshold: float,
    normalization_low: float,
    normalization_high: float,
) -> dict[str, Any]:
    center = np.rint(np.asarray(target_xyz, dtype=np.float64)).astype(np.int64)
    half = max(1, int(half_width))
    low = center - half
    high = center + half
    shape_xyz = np.asarray(source.shape[::-1], dtype=np.int64)
    truncated = bool(np.any(low < 0) or np.any(high > shape_xyz))
    low = np.maximum(low, 0)
    high = np.minimum(high, shape_xyz)
    if np.any(high <= low):
        return {
            "volumeTruncated": True,
            "sampleVoxelCount": 0,
            "materialFraction": 0.0,
            "normalizedIntensityStd": 0.0,
            "normalizedIntensityRangeP90P10": 0.0,
        }
    values = np.asarray(
        source[low[2] : high[2], low[1] : high[1], low[0] : high[0]],
        dtype=np.float32,
    )
    scale = max(float(normalization_high - normalization_low), 1.0e-6)
    normalized = np.clip((values - normalization_low) / scale, 0.0, 1.0)
    p10, p90 = np.percentile(normalized, (10, 90))
    return {
        "volumeTruncated": truncated,
        "sampleVoxelCount": int(values.size),
        "materialFraction": round(float(np.mean(values > air_threshold)), 4),
        "normalizedIntensityStd": round(float(np.std(normalized)), 5),
        "normalizedIntensityRangeP90P10": round(float(p90 - p10), 5),
    }


def census_fragment_terminations(
    output_root: str | Path,
    settings: dict[str, Any] | None = None,
    force: bool = False,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    graph_stem = f"global-monotone-graph-v{GLOBAL_MONOTONE_GRAPH_VERSION}"
    association_stem = (
        f"global-branch-association-v{GLOBAL_BRANCH_ASSOCIATION_VERSION}"
    )
    boundary_stem = (
        f"global-boundary-candidates-v{GLOBAL_BOUNDARY_CANDIDATE_VERSION}"
    )
    schedule_stem = f"tiled-window-schedule-v{WINDOW_SCHEDULER_VERSION}"
    graph_path = root / f"{graph_stem}.npz"
    association_path = root / f"{association_stem}.npz"
    boundary_path = root / f"{boundary_stem}.npz"
    schedule_path = root / f"{schedule_stem}.json"
    analysis_path = root / "analysis.json"
    grid_path = root / "grid.json"
    schedule = json.loads(schedule_path.read_text())
    analysis = json.loads(analysis_path.read_text())
    grid = json.loads(grid_path.read_text())
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
    source_path = Path(analysis["identity"]["source"])
    identity = {
        "version": FRAGMENT_TERMINATION_CENSUS_VERSION,
        "settings": resolved,
        "source": {
            "path": str(source_path),
            "bytes": int(analysis["identity"]["sourceBytes"]),
            "mtimeNs": int(analysis["identity"]["sourceMtimeNs"]),
        },
        "inputArtifacts": [
            _content_identity(path)
            for path in (
                graph_path,
                association_path,
                boundary_path,
                schedule_path,
                analysis_path,
                grid_path,
                *local_paths,
                *flake_paths,
            )
        ],
    }
    stem = f"fragment-termination-census-v{FRAGMENT_TERMINATION_CENSUS_VERSION}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(graph_path) as payload:
        graph = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(association_path) as payload:
        association_artifact = {
            key: np.asarray(payload[key]) for key in payload.files
        }
    with np.load(boundary_path) as payload:
        boundary = {key: np.asarray(payload[key]) for key in payload.files}

    node_identity = graph["nodeIdentity"].astype(np.uint64)
    node_association = association_artifact["nodeAssociation"].astype(np.uint32)
    association_node_count = association_artifact[
        "associationNodeCount"
    ].astype(np.uint32)
    association_branch_count = association_artifact[
        "associationBranchCount"
    ].astype(np.uint32)
    cell = graph["nodeCellIndex"].astype(np.int32)
    association_plane_pair = np.unique(
        np.stack((node_association.astype(np.int64), cell[:, 2]), axis=1),
        axis=0,
    )
    association_plane_count = np.bincount(
        association_plane_pair[:, 0], minlength=len(association_node_count)
    ).astype(np.uint16)
    endpoint_node = np.flatnonzero(
        (graph["degree"] == 1)
        & (
            association_node_count[node_association]
            >= int(resolved["minimumAssociationNodeCount"])
        )
    ).astype(np.uint32)
    endpoint_position = np.full(len(node_identity), -1, dtype=np.int32)
    endpoint_position[endpoint_node] = np.arange(len(endpoint_node), dtype=np.int32)
    endpoint_category = np.full(
        len(endpoint_node), CATEGORY_NO_COMPATIBLE_CANDIDATE, dtype=np.uint8
    )
    evidence_source = np.full(len(endpoint_node), EVIDENCE_NONE, dtype=np.uint8)
    evidence_index = np.full(len(endpoint_node), -1, dtype=np.int32)
    evidence_window = np.full(len(endpoint_node), -1, dtype=np.int16)
    evidence_stage = np.zeros(len(endpoint_node), dtype=np.uint8)
    evidence_score = np.full(len(endpoint_node), np.nan, dtype=np.float32)
    evidence_height = np.full(len(endpoint_node), np.nan, dtype=np.float32)
    evidence_normal = np.full(len(endpoint_node), np.nan, dtype=np.float32)

    def update_endpoint(
        node: int,
        category: int,
        source: int,
        index: int,
        window: int,
        score: float,
        height: float = math.nan,
        normal: float = math.nan,
    ) -> None:
        position = int(endpoint_position[int(node)])
        if position < 0:
            return
        stage = int(source)
        current_priority = int(CATEGORY_PRIORITY[int(endpoint_category[position])])
        priority = int(CATEGORY_PRIORITY[int(category)])
        current_score = float(evidence_score[position])
        replace = (
            stage > int(evidence_stage[position])
            or (
                stage == int(evidence_stage[position])
                and (
                    priority > current_priority
                    or (
                        priority == current_priority
                        and (
                            not np.isfinite(current_score)
                            or (np.isfinite(score) and score > current_score)
                        )
                    )
                )
            )
        )
        if not replace:
            return
        endpoint_category[position] = category
        evidence_source[position] = source
        evidence_index[position] = index
        evidence_window[position] = window
        evidence_stage[position] = stage
        evidence_score[position] = score
        evidence_height[position] = height
        evidence_normal[position] = normal

    local_candidate_count = 0
    for window_index, window in enumerate(schedule["windows"]):
        suffix = window_artifact_suffix(window["originCellXYZ"])
        monotone_path = root / (
            f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz"
        )
        local_association_path = root / (
            f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}.npz"
        )
        with np.load(monotone_path) as payload:
            local_identity = (
                payload["sourceZIndex"].astype(np.uint64) << np.uint64(32)
            ) | payload["sourceFlakeId"].astype(np.uint64)
        with np.load(local_association_path) as payload:
            local = {key: np.asarray(payload[key]) for key in payload.files}
        local_endpoint_identity = np.stack(
            (
                local_identity[local["candidateNodeSource"].astype(np.int64)],
                local_identity[local["candidateNodeTarget"].astype(np.int64)],
            ),
            axis=1,
        )
        local_endpoint_node = _identity_node_indexes(
            node_identity,
            local_endpoint_identity,
            "a local termination candidate endpoint",
        )
        for candidate_index, nodes in enumerate(local_endpoint_node):
            category = _local_candidate_category(
                int(local["candidateMainDecision"][candidate_index]),
                int(local["candidateFinalDecision"][candidate_index]),
                bool(local["candidateStable"][candidate_index]),
            )
            for node in nodes:
                update_endpoint(
                    int(node),
                    category,
                    EVIDENCE_LOCAL_WINDOW,
                    candidate_index,
                    window_index,
                    float(local["candidateScore"][candidate_index]),
                    float(
                        local["candidateExactPairMedianHeightResidualVoxels"]
                        [candidate_index]
                    ),
                    float(
                        local["candidateExactPairMedianNormalResidualDeg"]
                        [candidate_index]
                    ),
                )
        local_candidate_count += len(local_endpoint_node)
        if progress is not None:
            progress(
                "local-evidence",
                window_index + 1,
                len(schedule["windows"]),
                {"candidateCount": len(local_endpoint_node)},
            )

    for candidate_index, nodes in enumerate(boundary["candidateEndpointNodeIndex"]):
        if bool(boundary["candidatePreviouslyProposedLocally"][candidate_index]):
            continue
        decision = int(boundary["candidateOrderDecision"][candidate_index])
        if decision == LOCAL_DECISION_MATERIAL_DEFERRED:
            category = CATEGORY_MATERIAL_DEFERRED
        elif decision == LOCAL_DECISION_ORDER_AMBIGUOUS:
            category = CATEGORY_ORDER_UNRESOLVED
        elif decision == LOCAL_DECISION_ORDER_BLOCKED:
            category = CATEGORY_ORDER_BLOCKED
        elif decision == LOCAL_DECISION_CELL_COLLISION:
            category = CATEGORY_COLLISION_BLOCKED
        elif decision in (LOCAL_DECISION_RETAINED, LOCAL_DECISION_REDUNDANT):
            category = CATEGORY_CONTINUED
        else:
            category = CATEGORY_WEAK_GEOMETRY
        for node in nodes:
            update_endpoint(
                int(node),
                category,
                EVIDENCE_GLOBAL_BOUNDARY,
                candidate_index,
                -1,
                float(boundary["candidateScore"][candidate_index]),
            )

    for candidate_index, nodes in enumerate(
        association_artifact["candidateEndpointNodeIndex"]
    ):
        category = _global_candidate_category(
            int(association_artifact["candidateFinalDecision"][candidate_index])
        )
        for node in nodes:
            update_endpoint(
                int(node),
                category,
                EVIDENCE_COMPLETE_BRANCH,
                candidate_index,
                -1,
                float(association_artifact["candidateMinimumScore"][candidate_index]),
                float(
                    association_artifact[
                        "candidateExactMedianHeightResidualVoxels"
                    ][candidate_index]
                ),
                float(
                    association_artifact["candidateExactMedianNormalResidualDeg"]
                    [candidate_index]
                ),
            )
    for evidence_offset, nodes in enumerate(
        association_artifact["alreadyLinkedEndpointNodeIndex"]
    ):
        for node in nodes:
            update_endpoint(
                int(node),
                CATEGORY_CONTINUED,
                EVIDENCE_COMPLETE_BRANCH,
                evidence_offset,
                -1,
                math.nan,
            )
    excluded_identity = association_artifact[
        "excludedQuarantinedEndpointIdentity"
    ].astype(np.uint64)
    if len(excluded_identity):
        excluded_node = _identity_node_indexes(
            node_identity,
            excluded_identity,
            "a quarantined endpoint",
        )
        for evidence_offset, nodes in enumerate(excluded_node):
            for node in nodes:
                update_endpoint(
                    int(node),
                    CATEGORY_INTEGRITY_REJECTED,
                    EVIDENCE_COMPLETE_BRANCH,
                    evidence_offset,
                    -1,
                    math.nan,
                )

    centers, _, endpoint_flakes = _load_global_node_geometry(
        root, node_identity, endpoint_node, progress
    )
    usable, outward, _, _, _ = _boundary_outward(
        endpoint_flakes,
        endpoint_node,
        centers,
        graph["edgeSourceNodeIndex"].astype(np.uint32),
        graph["edgeTargetNodeIndex"].astype(np.uint32),
        graph["retained"].astype(bool),
    )
    endpoint_source_z = (node_identity[endpoint_node] >> np.uint64(32)).astype(
        np.int32
    )
    endpoint_source_id = (
        node_identity[endpoint_node] & np.uint64(0xFFFFFFFF)
    ).astype(np.int64)
    material_supported, contested = _node_material_state(
        root, endpoint_source_z, endpoint_source_id
    )
    quality = np.asarray(
        [float(flake["quality"]) for flake in endpoint_flakes], dtype=np.float32
    )
    active = usable & (endpoint_category != CATEGORY_CONTINUED)
    cluster_index = _cluster_termination_regions(
        node_association[endpoint_node],
        cell[endpoint_node],
        outward,
        active,
        int(resolved["clusterRadiusCells"]),
        float(resolved["minimumClusterOutwardCosine"]),
    )
    cluster_count = int(np.max(cluster_index, initial=-1)) + 1
    cluster_association = np.empty(cluster_count, dtype=np.uint32)
    cluster_category = np.empty(cluster_count, dtype=np.uint8)
    cluster_endpoint_count = np.empty(cluster_count, dtype=np.uint16)
    cluster_center = np.empty((cluster_count, 3), dtype=np.float32)
    cluster_outward = np.empty((cluster_count, 3), dtype=np.float32)
    cluster_target = np.empty((cluster_count, 3), dtype=np.float32)
    cluster_cell_low = np.empty((cluster_count, 3), dtype=np.uint16)
    cluster_cell_high = np.empty((cluster_count, 3), dtype=np.uint16)
    cluster_material_fraction = np.empty(cluster_count, dtype=np.float32)
    cluster_contested_fraction = np.empty(cluster_count, dtype=np.float32)
    cluster_quality = np.empty(cluster_count, dtype=np.float32)
    cluster_best_endpoint = np.empty(cluster_count, dtype=np.uint32)
    cluster_base_priority = np.empty(cluster_count, dtype=np.float32)
    cluster_category_counts = np.zeros(
        (cluster_count, len(CATEGORY_NAMES)), dtype=np.uint16
    )
    target_distance = float(resolved["targetDistanceVoxels"])
    for cluster_id in range(cluster_count):
        members = np.flatnonzero(cluster_index == cluster_id)
        association_ids = np.unique(node_association[endpoint_node[members]])
        if len(association_ids) != 1:
            raise RuntimeError("one termination cluster spans final associations")
        association_id = int(association_ids[0])
        counts = np.bincount(
            endpoint_category[members], minlength=len(CATEGORY_NAMES)
        ).astype(np.uint16)
        primary = int(
            max(
                np.flatnonzero(counts),
                key=lambda value: int(CATEGORY_PRIORITY[int(value)]),
            )
        )
        member_outward = outward[members]
        mean_outward = np.mean(member_outward, axis=0)
        length = float(np.linalg.norm(mean_outward))
        if length < 1.0e-6:
            mean_outward = member_outward[0]
        else:
            mean_outward /= length
        member_center = centers[endpoint_node[members]]
        center = np.mean(member_center, axis=0)
        priorities = CATEGORY_PRIORITY[endpoint_category[members]].astype(np.int16)
        member_scores = np.nan_to_num(evidence_score[members], nan=-1.0)
        best_local = int(np.lexsort((-member_scores, -priorities))[0])
        best_endpoint = int(members[best_local])
        node_count = int(association_node_count[association_id])
        plane_count = int(association_plane_count[association_id])
        category_bonus = {
            CATEGORY_NO_COMPATIBLE_CANDIDATE: 1.0,
            CATEGORY_WEAK_GEOMETRY: 1.8,
            CATEGORY_OVERLAP_UNRESOLVED: 1.5,
            CATEGORY_ORDER_UNRESOLVED: 1.0,
            CATEGORY_ORDER_BLOCKED: 0.2,
            CATEGORY_GEOMETRY_REJECTED: 1.2,
            CATEGORY_COLLISION_BLOCKED: 0.3,
            CATEGORY_INTEGRITY_REJECTED: 0.1,
            CATEGORY_MATERIAL_DEFERRED: 0.0,
        }.get(primary, 0.0)
        cluster_association[cluster_id] = association_id
        cluster_category[cluster_id] = primary
        cluster_endpoint_count[cluster_id] = len(members)
        cluster_center[cluster_id] = center
        cluster_outward[cluster_id] = mean_outward
        cluster_target[cluster_id] = center + mean_outward * target_distance
        member_cell = cell[endpoint_node[members]]
        cluster_cell_low[cluster_id] = np.min(member_cell, axis=0)
        cluster_cell_high[cluster_id] = np.max(member_cell, axis=0) + 1
        cluster_material_fraction[cluster_id] = float(
            np.mean(material_supported[members])
        )
        cluster_contested_fraction[cluster_id] = float(np.mean(contested[members]))
        cluster_quality[cluster_id] = float(np.median(quality[members]))
        cluster_best_endpoint[cluster_id] = best_endpoint
        cluster_category_counts[cluster_id] = counts
        cluster_base_priority[cluster_id] = float(
            math.log1p(node_count)
            + 0.20 * plane_count
            + 0.15 * math.log1p(len(members))
            + 0.50 * float(cluster_quality[cluster_id])
            + category_bonus
        )

    target_material = np.full(cluster_count, np.nan, dtype=np.float32)
    target_std = np.full(cluster_count, np.nan, dtype=np.float32)
    target_range = np.full(cluster_count, np.nan, dtype=np.float32)
    target_truncated = np.ones(cluster_count, dtype=bool)
    target_sample_count = np.zeros(cluster_count, dtype=np.uint32)
    dense_category = np.isin(cluster_category, list(DENSE_ACUS_CATEGORIES))
    ct_order = np.flatnonzero(dense_category)
    ct_order = ct_order[np.argsort(cluster_base_priority[ct_order])[::-1]]
    ct_order = ct_order[: int(resolved["maximumCtSampledClusterCount"])]
    source = np.load(source_path, mmap_mode="r")
    for offset, cluster_id in enumerate(ct_order):
        metrics = _target_ct_metrics(
            source,
            cluster_target[cluster_id],
            int(resolved["targetSampleHalfWidthVoxels"]),
            float(analysis["normalization"]["airThreshold"]),
            float(analysis["normalization"]["low"]),
            float(analysis["normalization"]["high"]),
        )
        target_material[cluster_id] = metrics["materialFraction"]
        target_std[cluster_id] = metrics["normalizedIntensityStd"]
        target_range[cluster_id] = metrics["normalizedIntensityRangeP90P10"]
        target_truncated[cluster_id] = metrics["volumeTruncated"]
        target_sample_count[cluster_id] = metrics["sampleVoxelCount"]
        if progress is not None and (
            (offset + 1) % 32 == 0 or offset + 1 == len(ct_order)
        ):
            progress(
                "ct-targets",
                offset + 1,
                len(ct_order),
                {"clusterIndex": int(cluster_id)},
            )
    dense_priority = (
        cluster_base_priority
        + 1.5 * np.nan_to_num(target_material, nan=0.0)
        + 0.5 * np.nan_to_num(target_range, nan=0.0)
    ).astype(np.float32)
    dense_eligible = (
        dense_category
        & ~target_truncated
        & np.isfinite(target_material)
        & (
            target_material
            >= float(resolved["minimumTargetMaterialFraction"])
        )
    )
    dense_target = np.zeros(cluster_count, dtype=bool)
    per_association: Counter[int] = Counter()
    for cluster_id in np.flatnonzero(dense_eligible)[
        np.argsort(dense_priority[dense_eligible])[::-1]
    ]:
        association_id = int(cluster_association[cluster_id])
        if per_association[association_id] >= int(
            resolved["maximumTargetsPerAssociation"]
        ):
            continue
        dense_target[cluster_id] = True
        per_association[association_id] += 1
        if int(np.count_nonzero(dense_target)) >= int(
            resolved["maximumDenseAcusTargetCount"]
        ):
            break

    _atomic_npz(
        artifact_path,
        endpointNodeIndex=endpoint_node,
        endpointIdentity=node_identity[endpoint_node],
        endpointAssociation=node_association[endpoint_node],
        endpointBranch=graph["component"][endpoint_node].astype(np.uint32),
        endpointCellIndex=cell[endpoint_node].astype(np.uint16),
        endpointCenterXYZ=centers[endpoint_node].astype(np.float32),
        endpointOutwardXYZ=outward.astype(np.float32),
        endpointOutwardUsable=usable,
        endpointCategory=endpoint_category,
        endpointEvidenceSource=evidence_source,
        endpointEvidenceIndex=evidence_index,
        endpointEvidenceWindowIndex=evidence_window,
        endpointEvidenceScore=evidence_score,
        endpointEvidenceMedianHeightResidualVoxels=evidence_height,
        endpointEvidenceMedianNormalResidualDeg=evidence_normal,
        endpointMaterialSupported=material_supported,
        endpointContestedMaterial=contested,
        endpointQuality=quality,
        endpointClusterIndex=cluster_index,
        clusterAssociation=cluster_association,
        clusterCategory=cluster_category,
        clusterCategoryCounts=cluster_category_counts,
        clusterEndpointCount=cluster_endpoint_count,
        clusterCenterXYZ=cluster_center,
        clusterOutwardXYZ=cluster_outward,
        clusterTargetXYZ=cluster_target,
        clusterCellOriginXYZ=cluster_cell_low,
        clusterCellStopXYZExclusive=cluster_cell_high,
        clusterMaterialSupportedFraction=cluster_material_fraction,
        clusterContestedMaterialFraction=cluster_contested_fraction,
        clusterMedianQuality=cluster_quality,
        clusterBestEndpointIndex=cluster_best_endpoint,
        clusterAssociationNodeCount=association_node_count[cluster_association],
        clusterAssociationBranchCount=association_branch_count[
            cluster_association
        ],
        clusterAssociationPlaneCount=association_plane_count[
            cluster_association
        ],
        clusterBasePriority=cluster_base_priority,
        clusterTargetMaterialFraction=target_material,
        clusterTargetNormalizedIntensityStd=target_std,
        clusterTargetNormalizedIntensityRangeP90P10=target_range,
        clusterTargetVolumeTruncated=target_truncated,
        clusterTargetSampleVoxelCount=target_sample_count,
        clusterDenseAcusPriority=dense_priority,
        clusterDenseAcusTarget=dense_target,
    )

    def cluster_record(cluster_id: int) -> dict[str, Any]:
        best_endpoint = int(cluster_best_endpoint[cluster_id])
        return {
            "clusterIndex": int(cluster_id),
            "category": CATEGORY_NAMES[int(cluster_category[cluster_id])],
            "associationId": int(cluster_association[cluster_id]),
            "associationNodeCount": int(
                association_node_count[int(cluster_association[cluster_id])]
            ),
            "associationBranchCount": int(
                association_branch_count[int(cluster_association[cluster_id])]
            ),
            "associationPlaneCount": int(
                association_plane_count[int(cluster_association[cluster_id])]
            ),
            "endpointCount": int(cluster_endpoint_count[cluster_id]),
            "categoryCounts": {
                name: int(cluster_category_counts[cluster_id, value])
                for value, name in CATEGORY_NAMES.items()
                if int(cluster_category_counts[cluster_id, value])
            },
            "cellOriginXYZ": cluster_cell_low[cluster_id].astype(int).tolist(),
            "cellStopXYZExclusive": cluster_cell_high[cluster_id]
            .astype(int)
            .tolist(),
            "centerXYZ": np.round(cluster_center[cluster_id], 3).tolist(),
            "outwardXYZ": np.round(cluster_outward[cluster_id], 6).tolist(),
            "targetXYZ": np.round(cluster_target[cluster_id], 3).tolist(),
            "materialSupportedFraction": round(
                float(cluster_material_fraction[cluster_id]), 4
            ),
            "contestedMaterialFraction": round(
                float(cluster_contested_fraction[cluster_id]), 4
            ),
            "targetCt": {
                "sampled": bool(np.isfinite(target_material[cluster_id])),
                "volumeTruncated": bool(target_truncated[cluster_id]),
                "materialFraction": (
                    round(float(target_material[cluster_id]), 4)
                    if np.isfinite(target_material[cluster_id])
                    else None
                ),
                "normalizedIntensityStd": (
                    round(float(target_std[cluster_id]), 5)
                    if np.isfinite(target_std[cluster_id])
                    else None
                ),
                "normalizedIntensityRangeP90P10": (
                    round(float(target_range[cluster_id]), 5)
                    if np.isfinite(target_range[cluster_id])
                    else None
                ),
            },
            "bestEvidence": {
                "source": EVIDENCE_NAMES[int(evidence_source[best_endpoint])],
                "index": int(evidence_index[best_endpoint]),
                "windowIndex": int(evidence_window[best_endpoint]),
                "score": (
                    round(float(evidence_score[best_endpoint]), 6)
                    if np.isfinite(evidence_score[best_endpoint])
                    else None
                ),
                "medianHeightResidualVoxels": (
                    round(float(evidence_height[best_endpoint]), 4)
                    if np.isfinite(evidence_height[best_endpoint])
                    else None
                ),
                "medianNormalResidualDeg": (
                    round(float(evidence_normal[best_endpoint]), 4)
                    if np.isfinite(evidence_normal[best_endpoint])
                    else None
                ),
            },
            "basePriority": round(float(cluster_base_priority[cluster_id]), 6),
            "denseAcusPriority": round(float(dense_priority[cluster_id]), 6),
        }

    dense_indices = np.flatnonzero(dense_target)
    dense_indices = dense_indices[np.argsort(dense_priority[dense_indices])[::-1]]
    review_limit = int(resolved["maximumReviewTargetCount"])
    order_indices = np.flatnonzero(
        np.isin(cluster_category, list(ORDER_REVIEW_CATEGORIES))
    )
    order_indices = order_indices[
        np.argsort(cluster_base_priority[order_indices])[::-1]
    ][:review_limit]
    geometry_indices = np.flatnonzero(
        np.isin(cluster_category, list(GEOMETRY_REVIEW_CATEGORIES))
    )
    geometry_indices = geometry_indices[
        np.argsort(cluster_base_priority[geometry_indices])[::-1]
    ][:review_limit]
    endpoint_category_count = {
        name: int(np.count_nonzero(endpoint_category == value))
        for value, name in CATEGORY_NAMES.items()
    }
    cluster_category_count = {
        name: int(np.count_nonzero(cluster_category == value))
        for value, name in CATEGORY_NAMES.items()
        if value != CATEGORY_CONTINUED
    }
    result = {
        "identity": identity,
        "settings": resolved,
        "contract": {
            "scope": (
                "definite degree-one global-graph ends belonging to final v6 "
                "associations with at least minimumAssociationNodeCount flakes"
            ),
            "classification": (
                "local-window, global-boundary, and complete-branch evidence are "
                "reconciled without converting absence into sheet identity; a no-"
                "compatible-candidate end has no current Acus continuation passing "
                "the loose boundary hit gates"
            ),
            "clustering": (
                "nearby ends on one final association are grouped only when their "
                "outward tangent directions agree"
            ),
            "denseAcusQueue": (
                "only no-candidate, weak-geometry, or overlap-unresolved sectors "
                "with nontruncated CT samples and sufficient material are queued; "
                "order and geometry failures remain separate review queues"
            ),
        },
        "artifact": _content_identity(artifact_path),
        "queues": {
            "denseAcus": [cluster_record(int(value)) for value in dense_indices],
            "orderReview": [cluster_record(int(value)) for value in order_indices],
            "geometryReview": [
                cluster_record(int(value)) for value in geometry_indices
            ],
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "localCandidateOccurrenceCount": local_candidate_count,
            "endpointCount": len(endpoint_node),
            "usableOutwardEndpointCount": int(np.count_nonzero(usable)),
            "continuedEndpointCount": endpoint_category_count["continued"],
            "remainingTerminationEndpointCount": int(
                np.count_nonzero(endpoint_category != CATEGORY_CONTINUED)
            ),
            "clusteredTerminationEndpointCount": int(np.count_nonzero(active)),
            "unusableOutwardTerminationEndpointCount": int(
                np.count_nonzero(
                    (endpoint_category != CATEGORY_CONTINUED) & ~usable
                )
            ),
            "endpointCategoryCounts": endpoint_category_count,
            "terminationClusterCount": cluster_count,
            "clusterCategoryCounts": cluster_category_count,
            "clusterEndpointCount": _quantiles(cluster_endpoint_count),
            "clusterAssociationNodeCount": _quantiles(
                association_node_count[cluster_association]
            ),
            "ctSampledClusterCount": len(ct_order),
            "ctSampledTargetMaterialFraction": _quantiles(
                target_material[ct_order]
            ),
            "denseAcusEligibleClusterCount": int(
                np.count_nonzero(dense_eligible)
            ),
            "denseAcusQueuedClusterCount": len(dense_indices),
            "denseAcusQueuedAssociationCount": len(
                np.unique(cluster_association[dense_indices])
            ),
            "denseAcusQueuedCategoryCounts": {
                name: int(
                    np.count_nonzero(cluster_category[dense_indices] == value)
                )
                for value, name in CATEGORY_NAMES.items()
                if int(
                    np.count_nonzero(cluster_category[dense_indices] == value)
                )
            },
            "orderReviewClusterCount": int(
                np.count_nonzero(
                    np.isin(cluster_category, list(ORDER_REVIEW_CATEGORIES))
                )
            ),
            "geometryReviewClusterCount": int(
                np.count_nonzero(
                    np.isin(cluster_category, list(GEOMETRY_REVIEW_CATEGORIES))
                )
            ),
        },
    }
    _atomic_json(summary_path, result)
    return result
