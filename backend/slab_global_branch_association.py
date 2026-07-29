from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .slab_association_integrity import (
    DEFAULT_SETTINGS as INTEGRITY_SETTINGS,
    _bbox_distance,
    _mesh_intersections,
    _surface_geometry,
)
from .slab_branch_association import (
    BRANCH_ASSOCIATION_VERSION,
    DECISION_REDUNDANT as LOCAL_DECISION_REDUNDANT,
    DECISION_RETAINED as LOCAL_DECISION_RETAINED,
    DEFAULT_SETTINGS as BRANCH_SETTINGS,
    _exact_carrier_stats,
)
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_global_branch_candidates import (
    GLOBAL_BRANCH_CANDIDATE_VERSION,
    SOURCE_LOCAL_EXACT_DEFERRED,
    SOURCE_SUBWINDOW_UNRESOLVED,
    build_global_branch_candidates,
)
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


GLOBAL_BRANCH_ASSOCIATION_VERSION = 4

PROVENANCE_LOCAL_EXACT_DEFERRED = SOURCE_LOCAL_EXACT_DEFERRED
PROVENANCE_SUBWINDOW_UNRESOLVED = SOURCE_SUBWINDOW_UNRESOLVED
PROVENANCE_CONTEXT_DISPUTED = 3
PROVENANCE_SINGLE_WINDOW = 4
PROVENANCE_OVERLAP_VALIDATED = 5
PROVENANCE_NAMES = {
    PROVENANCE_LOCAL_EXACT_DEFERRED: "local-exact-deferred",
    PROVENANCE_SUBWINDOW_UNRESOLVED: "subwindow-unresolved",
    PROVENANCE_CONTEXT_DISPUTED: "context-disputed",
    PROVENANCE_SINGLE_WINDOW: "single-window",
    PROVENANCE_OVERLAP_VALIDATED: "overlap-validated",
}

DECISION_INPUT_BRANCH_CARRIER_DEFERRED = 1
DECISION_EXACT_PAIR_DEFERRED = 2
DECISION_CELL_COLLISION = 3
DECISION_EXACT_GROUP_PRUNED = 4
DECISION_INTEGRITY_QUARANTINED = 5
DECISION_RETAINED = 6
DECISION_REDUNDANT = 7

DECISION_NAMES = {
    DECISION_INPUT_BRANCH_CARRIER_DEFERRED: "input-branch-carrier-deferred",
    DECISION_EXACT_PAIR_DEFERRED: "exact-pair-deferred",
    DECISION_CELL_COLLISION: "cell-collision",
    DECISION_EXACT_GROUP_PRUNED: "exact-group-pruned",
    DECISION_INTEGRITY_QUARANTINED: "integrity-quarantined",
    DECISION_RETAINED: "retained",
    DECISION_REDUNDANT: "redundant-support",
}

GRAPH_CELL_COLLISION = 1
GRAPH_RETAINED = 2
GRAPH_REDUNDANT = 3

DEFAULT_SETTINGS: dict[str, Any] = {
    "includeLocalExactDeferredCandidates": True,
    "includeSubwindowUnresolvedCandidates": True,
    "includeContextDisputedCandidates": True,
    "includeSingleWindowCandidates": True,
    "minimumOverlapValidatedObservations": 2,
    "maximumExactMedianHeightResidualVoxels": BRANCH_SETTINGS[
        "maximumExactMedianHeightResidualVoxels"
    ],
    "maximumExactMedianNormalResidualDeg": BRANCH_SETTINGS[
        "maximumExactMedianNormalResidualDeg"
    ],
    "exactCarrierPixelStepVoxels": BRANCH_SETTINGS[
        "exactCarrierPixelStepVoxels"
    ],
    "exactCarrierMaximumPixelsPerAxis": BRANCH_SETTINGS[
        "exactCarrierMaximumPixelsPerAxis"
    ],
    "maximumEvidenceCoreDistanceVoxels": INTEGRITY_SETTINGS[
        "maximumEvidenceCoreDistanceVoxels"
    ],
    "maximumSampledClearanceVoxels": INTEGRITY_SETTINGS[
        "maximumSampledClearanceVoxels"
    ],
    "clearanceSweepVoxels": INTEGRITY_SETTINGS["clearanceSweepVoxels"],
    "triangleBucketVoxels": INTEGRITY_SETTINGS["triangleBucketVoxels"],
    "intersectionToleranceVoxels": INTEGRITY_SETTINGS[
        "intersectionToleranceVoxels"
    ],
    "maximumStoredIntersectionPoints": INTEGRITY_SETTINGS[
        "maximumStoredIntersectionPoints"
    ],
    "orderZeroToleranceVoxels": INTEGRITY_SETTINGS[
        "orderZeroToleranceVoxels"
    ],
    "maximumParallelNormalAngleDeg": INTEGRITY_SETTINGS[
        "maximumParallelNormalAngleDeg"
    ],
    "maximumParallelFiberAngleDeg": INTEGRITY_SETTINGS[
        "maximumParallelFiberAngleDeg"
    ],
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


def _candidate_order(
    score: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    provenance_priority: np.ndarray | None = None,
) -> np.ndarray:
    priority = (
        np.zeros(len(score), dtype=np.int8)
        if provenance_priority is None
        else np.asarray(provenance_priority, dtype=np.int8)
    )
    return np.lexsort(
        (
            np.asarray(target, dtype=np.int64),
            np.asarray(source, dtype=np.int64),
            -priority,
            -np.asarray(score, dtype=np.float64),
        )
    )


def _solve_candidate_graph(
    branch_count: int,
    source: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    eligible: np.ndarray,
    branch_cells: dict[int, set[int]],
    provenance_priority: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    source = np.asarray(source, dtype=np.int64)
    target = np.asarray(target, dtype=np.int64)
    score = np.asarray(score, dtype=np.float64)
    eligible = np.asarray(eligible, dtype=bool)
    if not (len(source) == len(target) == len(score) == len(eligible)):
        raise ValueError("candidate arrays must have equal length")
    touched = sorted(set(source.astype(int)) | set(target.astype(int)))
    if any(value < 0 or value >= branch_count for value in touched):
        raise ValueError("candidate references an out-of-range global branch")
    if any(value not in branch_cells for value in touched):
        raise ValueError("candidate branch is missing its cell catalog")

    parent = {value: value for value in touched}
    size = {value: 1 for value in touched}
    cells = {value: set(branch_cells[value]) for value in touched}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    decisions = np.zeros(len(source), dtype=np.uint8)
    for candidate_index in _candidate_order(
        score, source, target, provenance_priority
    ):
        if not bool(eligible[candidate_index]):
            continue
        source_root = find(int(source[candidate_index]))
        target_root = find(int(target[candidate_index]))
        if source_root == target_root:
            decisions[candidate_index] = GRAPH_REDUNDANT
            continue
        if not cells[source_root].isdisjoint(cells[target_root]):
            decisions[candidate_index] = GRAPH_CELL_COLLISION
            continue
        if size[source_root] < size[target_root]:
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        size[source_root] += size[target_root]
        cells[source_root].update(cells[target_root])
        cells[target_root].clear()
        decisions[candidate_index] = GRAPH_RETAINED

    branch_root = np.arange(branch_count, dtype=np.uint32)
    for branch_id in touched:
        branch_root[branch_id] = find(branch_id)
    _, branch_association = np.unique(branch_root, return_inverse=True)
    return {
        "decisions": decisions,
        "branchAssociation": branch_association.astype(np.uint32),
    }


def _component_members(
    component: np.ndarray, touched: np.ndarray
) -> dict[int, np.ndarray]:
    selected = np.flatnonzero(np.isin(component, touched))
    order = np.argsort(component[selected], kind="stable")
    selected = selected[order]
    selected_component = component[selected]
    starts = np.flatnonzero(
        np.r_[True, selected_component[1:] != selected_component[:-1]]
    )
    output: dict[int, np.ndarray] = {}
    for offset, start in enumerate(starts):
        stop = starts[offset + 1] if offset + 1 < len(starts) else len(selected)
        output[int(selected_component[start])] = selected[start:stop]
    if set(output) != set(np.asarray(touched, dtype=int)):
        raise ValueError("one candidate branch has no global graph members")
    return output


def _load_selected_flakes(
    root: Path, node_identity: np.ndarray, selected_nodes: np.ndarray
) -> dict[int, dict[str, Any]]:
    selected_nodes = np.asarray(selected_nodes, dtype=np.int64)
    identity = node_identity[selected_nodes].astype(np.uint64)
    source_z = (identity >> np.uint64(32)).astype(np.int32)
    source_id = (identity & np.uint64(0xFFFFFFFF)).astype(np.int64)
    output: dict[int, dict[str, Any]] = {}
    for z_index in np.unique(source_z):
        path = root / (
            f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json"
        )
        flakes = json.loads(path.read_text())["flakes"]
        for local_index in np.flatnonzero(source_z == z_index):
            flake_id = int(source_id[local_index])
            flake = flakes[flake_id]
            if int(flake["id"]) != flake_id:
                raise ValueError("flake cache IDs are not dense and index aligned")
            if int(flake.get("normalFamily", 0)) != 0:
                raise ValueError("global association contains a non-primary flake")
            output[int(selected_nodes[local_index])] = flake
    return output


def _candidate_observations(
    root: Path,
    windows: list[dict[str, Any]],
    candidate_pairs: np.ndarray,
) -> dict[str, np.ndarray]:
    lookup = {
        (int(pair[0]), int(pair[1])): index
        for index, pair in enumerate(candidate_pairs)
    }
    candidate_index = []
    window_index = []
    scores = []
    height = []
    normal = []
    for local_window_index, window in enumerate(windows):
        suffix = window_artifact_suffix(window["originCellXYZ"])
        monotone_path = root / (
            f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz"
        )
        association_path = root / (
            f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}.npz"
        )
        with np.load(monotone_path) as payload:
            node_identity = (
                payload["sourceZIndex"].astype(np.uint64) << np.uint64(32)
            ) | payload["sourceFlakeId"].astype(np.uint64)
        with np.load(association_path) as payload:
            accepted = np.isin(
                payload["candidateFinalDecision"],
                [LOCAL_DECISION_RETAINED, LOCAL_DECISION_REDUNDANT],
            )
            for local_candidate_index in np.flatnonzero(accepted):
                first = int(
                    node_identity[
                        int(payload["candidateNodeSource"][local_candidate_index])
                    ]
                )
                second = int(
                    node_identity[
                        int(payload["candidateNodeTarget"][local_candidate_index])
                    ]
                )
                index = lookup.get((min(first, second), max(first, second)))
                if index is None:
                    continue
                candidate_index.append(index)
                window_index.append(local_window_index)
                scores.append(float(payload["candidateScore"][local_candidate_index]))
                height.append(
                    float(
                        payload[
                            "candidateExactPairMedianHeightResidualVoxels"
                        ][local_candidate_index]
                    )
                )
                normal.append(
                    float(
                        payload[
                            "candidateExactPairMedianNormalResidualDeg"
                        ][local_candidate_index]
                    )
                )
    return {
        "candidateIndex": np.asarray(candidate_index, dtype=np.uint32),
        "windowIndex": np.asarray(window_index, dtype=np.uint8),
        "score": np.asarray(scores, dtype=np.float32),
        "medianHeightResidualVoxels": np.asarray(height, dtype=np.float32),
        "medianNormalResidualDeg": np.asarray(normal, dtype=np.float32),
    }


def _aggregate_observations(
    observations: dict[str, np.ndarray], candidate_count: int
) -> dict[str, np.ndarray]:
    index = observations["candidateIndex"].astype(np.int64)
    count = np.bincount(index, minlength=candidate_count).astype(np.uint8)
    minimum = np.full(candidate_count, np.inf, dtype=np.float32)
    maximum = np.full(candidate_count, -np.inf, dtype=np.float32)
    total = np.zeros(candidate_count, dtype=np.float64)
    maximum_height = np.full(candidate_count, -np.inf, dtype=np.float32)
    maximum_normal = np.full(candidate_count, -np.inf, dtype=np.float32)
    np.minimum.at(minimum, index, observations["score"])
    np.maximum.at(maximum, index, observations["score"])
    np.add.at(total, index, observations["score"])
    observed_height = np.asarray(
        observations["medianHeightResidualVoxels"], dtype=np.float32
    )
    finite_height = np.isfinite(observed_height)
    np.maximum.at(
        maximum_height, index[finite_height], observed_height[finite_height]
    )
    observed_normal = np.asarray(
        observations["medianNormalResidualDeg"], dtype=np.float32
    )
    finite_normal = np.isfinite(observed_normal)
    np.maximum.at(
        maximum_normal, index[finite_normal], observed_normal[finite_normal]
    )
    if np.any(count == 0):
        raise ValueError("a selected global join has no accepted local observation")
    return {
        "count": count,
        "minimumScore": minimum,
        "meanScore": (total / count).astype(np.float32),
        "maximumScore": maximum,
        "maximumLocalMedianHeightResidualVoxels": maximum_height,
        "maximumLocalMedianNormalResidualDeg": maximum_normal,
    }


def _candidate_tier_selection(
    unanimous: np.ndarray,
    overlap_validated: np.ndarray,
    observation_count: np.ndarray,
    acceptance_count: np.ndarray,
    include_single_window: bool,
    include_context_disputed: bool,
    minimum_overlap_observations: int,
) -> tuple[np.ndarray, np.ndarray]:
    unanimous = np.asarray(unanimous, dtype=bool)
    overlap_validated = np.asarray(overlap_validated, dtype=bool)
    observation_count = np.asarray(observation_count, dtype=np.int32)
    acceptance_count = np.asarray(acceptance_count, dtype=np.int32)
    if not (
        len(unanimous)
        == len(overlap_validated)
        == len(observation_count)
        == len(acceptance_count)
    ):
        raise ValueError("candidate evidence arrays must have equal length")
    overlap_selected = (
        unanimous
        & overlap_validated
        & (observation_count >= minimum_overlap_observations)
    )
    single_window_selected = (
        unanimous
        & ~overlap_validated
        & (observation_count == 1)
        & include_single_window
    )
    context_disputed_selected = (
        ~unanimous
        & overlap_validated
        & (observation_count >= minimum_overlap_observations)
        & (acceptance_count > 0)
        & (acceptance_count < observation_count)
        & include_context_disputed
    )
    selected = (
        overlap_selected | single_window_selected | context_disputed_selected
    )
    provenance = np.full(
        int(np.count_nonzero(selected)), PROVENANCE_CONTEXT_DISPUTED, dtype=np.uint8
    )
    provenance[overlap_selected[selected]] = PROVENANCE_OVERLAP_VALIDATED
    provenance[single_window_selected[selected]] = PROVENANCE_SINGLE_WINDOW
    return selected, provenance


def _pair_gate_state(
    pair_exact_pass: np.ndarray,
    input_branch_standalone_pass: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    pair_exact_pass = np.asarray(pair_exact_pass, dtype=bool)
    input_branch_standalone_pass = np.asarray(
        input_branch_standalone_pass, dtype=bool
    )
    if len(pair_exact_pass) != len(input_branch_standalone_pass):
        raise ValueError("pair and standalone gate arrays must have equal length")
    decisions = np.zeros(len(pair_exact_pass), dtype=np.uint8)
    decisions[
        ~pair_exact_pass & ~input_branch_standalone_pass
    ] = DECISION_INPUT_BRANCH_CARRIER_DEFERRED
    decisions[
        ~pair_exact_pass & input_branch_standalone_pass
    ] = DECISION_EXACT_PAIR_DEFERRED
    return pair_exact_pass.copy(), decisions


def _pair_exact_audit(
    source: np.ndarray,
    target: np.ndarray,
    members: dict[int, np.ndarray],
    flakes: dict[int, dict[str, Any]],
    settings: dict[str, Any],
    progress: Callable[[str, int, int, dict[str, Any]], None] | None,
) -> dict[str, np.ndarray]:
    height = np.empty(len(source), dtype=np.float32)
    p90_height = np.empty(len(source), dtype=np.float32)
    normal = np.empty(len(source), dtype=np.float32)
    p90_normal = np.empty(len(source), dtype=np.float32)
    for candidate_index, (first, second) in enumerate(zip(source, target)):
        nodes = np.r_[members[int(first)], members[int(second)]]
        exact = _exact_carrier_stats(
            [flakes[int(node)] for node in nodes], settings
        )
        height[candidate_index] = exact["medianHeightResidualVoxels"]
        p90_height[candidate_index] = exact["p90HeightResidualVoxels"]
        normal[candidate_index] = exact["medianNormalResidualDeg"]
        p90_normal[candidate_index] = exact["p90NormalResidualDeg"]
        if progress is not None and (
            (candidate_index + 1) % 25 == 0 or candidate_index + 1 == len(source)
        ):
            progress(
                "pair-exact",
                candidate_index + 1,
                len(source),
                {
                    "passed": int(
                        height[candidate_index]
                        <= float(
                            settings["maximumExactMedianHeightResidualVoxels"]
                        )
                        and normal[candidate_index]
                        <= float(settings["maximumExactMedianNormalResidualDeg"])
                    )
                },
            )
    passed = (
        height <= float(settings["maximumExactMedianHeightResidualVoxels"])
    ) & (normal <= float(settings["maximumExactMedianNormalResidualDeg"]))
    return {
        "medianHeightResidualVoxels": height,
        "p90HeightResidualVoxels": p90_height,
        "medianNormalResidualDeg": normal,
        "p90NormalResidualDeg": p90_normal,
        "passed": passed,
    }


def _branch_exact_audit(
    branch_ids: np.ndarray,
    members: dict[int, np.ndarray],
    flakes: dict[int, dict[str, Any]],
    settings: dict[str, Any],
    progress: Callable[[str, int, int, dict[str, Any]], None] | None,
) -> dict[str, np.ndarray]:
    height = np.empty(len(branch_ids), dtype=np.float32)
    p90_height = np.empty(len(branch_ids), dtype=np.float32)
    normal = np.empty(len(branch_ids), dtype=np.float32)
    p90_normal = np.empty(len(branch_ids), dtype=np.float32)
    for branch_index, branch_id in enumerate(branch_ids):
        exact = _exact_carrier_stats(
            [flakes[int(node)] for node in members[int(branch_id)]], settings
        )
        height[branch_index] = exact["medianHeightResidualVoxels"]
        p90_height[branch_index] = exact["p90HeightResidualVoxels"]
        normal[branch_index] = exact["medianNormalResidualDeg"]
        p90_normal[branch_index] = exact["p90NormalResidualDeg"]
        if progress is not None and (
            (branch_index + 1) % 100 == 0
            or branch_index + 1 == len(branch_ids)
        ):
            progress(
                "branch-exact",
                branch_index + 1,
                len(branch_ids),
                {},
            )
    passed = (
        height <= float(settings["maximumExactMedianHeightResidualVoxels"])
    ) & (normal <= float(settings["maximumExactMedianNormalResidualDeg"]))
    return {
        "medianHeightResidualVoxels": height,
        "p90HeightResidualVoxels": p90_height,
        "medianNormalResidualDeg": normal,
        "p90NormalResidualDeg": p90_normal,
        "passed": passed,
    }


def _association_geometries(
    branch_association: np.ndarray,
    touched: np.ndarray,
    members: dict[int, np.ndarray],
    flakes: dict[int, dict[str, Any]],
    node_cell: np.ndarray,
    node_parity: np.ndarray,
    settings: dict[str, Any],
) -> list[dict[str, Any]]:
    by_association: dict[int, list[int]] = defaultdict(list)
    for branch_id in touched:
        by_association[int(branch_association[int(branch_id)])].append(
            int(branch_id)
        )
    output = []
    for association_id, branch_ids in by_association.items():
        if len(branch_ids) < 2:
            continue
        nodes = np.concatenate([members[value] for value in branch_ids])
        member_flakes = [flakes[int(node)] for node in nodes]
        compact_cell = node_cell[nodes].astype(np.int32)
        input_branch_sizes = [len(members[value]) for value in branch_ids]
        # This is a branch-local gauge coordinate used only to populate the
        # collision/order diagnostic payload. It is not interpreted across
        # independently gauged associations.
        compact_depth = np.asarray(
            [float(value["depthOffset"]) for value in member_flakes],
            dtype=np.float32,
        ) * node_parity[nodes]
        geometry = _surface_geometry(
            association_id,
            np.arange(len(nodes), dtype=np.int64),
            member_flakes,
            compact_cell,
            compact_depth,
            settings,
            settings,
        )
        carrier = geometry["stats"]["carrier"]
        exact = {
            "associationId": association_id,
            "branchIds": branch_ids,
            "branchCount": len(branch_ids),
            "flakeCount": len(nodes),
            "largestInputBranchFlakeCount": max(input_branch_sizes),
            "additionalFlakeCountBeyondLargestInput": (
                len(nodes) - max(input_branch_sizes)
            ),
            "axialPlaneCount": len(np.unique(compact_cell[:, 2])),
            "originCellXYZ": np.min(compact_cell, axis=0).astype(int).tolist(),
            "stopCellXYZExclusive": (
                np.max(compact_cell, axis=0) + 1
            ).astype(int).tolist(),
            "medianHeightResidualVoxels": float(
                carrier["medianNodeHeightResidualVoxels"]
            ),
            "p90HeightResidualVoxels": float(
                carrier["p90NodeHeightResidualVoxels"]
            ),
            "medianNormalResidualDeg": float(
                carrier["medianNodeNormalResidualDeg"]
            ),
            "p90NormalResidualDeg": float(
                carrier["p90NodeNormalResidualDeg"]
            ),
            "surfacePointCount": int(geometry["stats"]["surfacePointCount"]),
            "evidenceCorePointCount": int(
                geometry["stats"]["evidenceCorePointCount"]
            ),
            "triangleCount": int(geometry["stats"]["triangleCount"]),
            "withinAssociationCellCollisionCount": int(
                geometry["stats"]["withinAssociationCellCollisionCount"]
            ),
        }
        exact["gatesPass"] = bool(
            exact["medianHeightResidualVoxels"]
            <= float(settings["maximumExactMedianHeightResidualVoxels"])
            and exact["medianNormalResidualDeg"]
            <= float(settings["maximumExactMedianNormalResidualDeg"])
            and exact["withinAssociationCellCollisionCount"] == 0
        )
        geometry["exact"] = exact
        output.append(geometry)
    return output


def _integrity_audit(
    geometries: list[dict[str, Any]], settings: dict[str, Any]
) -> dict[str, Any]:
    violations = []
    broad_count = 0
    narrow_count = 0
    triangle_intersection_count = 0
    core_intersection_count = 0
    coplanar_count = 0
    bbox_pair_count = 0
    tolerance = float(settings["intersectionToleranceVoxels"])
    for first_index, first in enumerate(geometries):
        for second in geometries[first_index + 1 :]:
            if _bbox_distance(first, second) > tolerance:
                continue
            bbox_pair_count += 1
            audit = _mesh_intersections(first, second, settings)
            broad_count += int(audit["broadPhaseTrianglePairCount"])
            narrow_count += int(audit["narrowPhaseTrianglePairCount"])
            triangle_intersection_count += int(
                audit["intersectingTrianglePairCount"]
            )
            core_intersection_count += int(
                audit["evidenceCoreIntersectingTrianglePairCount"]
            )
            coplanar_count += int(audit["coplanarIntersectingTrianglePairCount"])
            if int(audit["intersectingTrianglePairCount"]):
                violations.append(
                    {
                        "associationSource": int(first["associationId"]),
                        "associationTarget": int(second["associationId"]),
                        **audit,
                    }
                )
    return {
        "violations": violations,
        "stats": {
            "carrierCount": len(geometries),
            "boundingBoxCandidatePairCount": bbox_pair_count,
            "broadPhaseTrianglePairCount": broad_count,
            "narrowPhaseTrianglePairCount": narrow_count,
            "intersectingTrianglePairCount": triangle_intersection_count,
            "evidenceCoreIntersectingTrianglePairCount": core_intersection_count,
            "coplanarIntersectingTrianglePairCount": coplanar_count,
            "violatingCarrierPairCount": len(violations),
        },
    }


def _weakest_retained_candidate(
    candidate_indices: np.ndarray,
    score: np.ndarray,
    endpoint_pair: np.ndarray,
    provenance_priority: np.ndarray,
) -> int:
    return int(
        min(
            candidate_indices,
            key=lambda index: (
                float(score[index]),
                int(provenance_priority[index]),
                int(endpoint_pair[index, 0]),
                int(endpoint_pair[index, 1]),
            ),
        )
    )


def _weakest_integrity_candidates(
    violations: list[dict[str, Any]],
    branch_association: np.ndarray,
    branch_pair: np.ndarray,
    graph_decision: np.ndarray,
    eligible: np.ndarray,
    score: np.ndarray,
    endpoint_pair: np.ndarray,
    provenance_priority: np.ndarray,
) -> np.ndarray:
    branch_association = np.asarray(branch_association, dtype=np.int64)
    branch_pair = np.asarray(branch_pair, dtype=np.int64)
    graph_decision = np.asarray(graph_decision, dtype=np.uint8)
    eligible = np.asarray(eligible, dtype=bool)
    source_association = branch_association[branch_pair[:, 0]]
    target_association = branch_association[branch_pair[:, 1]]
    selected = set()
    for violation in violations:
        association_ids = (
            int(violation["associationSource"]),
            int(violation["associationTarget"]),
        )
        candidates = np.flatnonzero(
            eligible
            & (source_association == target_association)
            & np.isin(source_association, association_ids)
            & (graph_decision == GRAPH_RETAINED)
        )
        if not len(candidates):
            raise RuntimeError(
                "an integrity violation has no retained construction edge to prune"
            )
        selected.add(
            _weakest_retained_candidate(
                candidates,
                score,
                endpoint_pair,
                provenance_priority,
            )
        )
    return np.asarray(sorted(selected), dtype=np.int64)


def associate_global_branches(
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
    schedule_artifact_path = root / f"{schedule_stem}.npz"
    graph_summary_path = root / f"{graph_stem}.json"
    graph_artifact_path = root / f"{graph_stem}.npz"
    schedule = json.loads(schedule_summary_path.read_text())
    graph_summary = json.loads(graph_summary_path.read_text())
    grid = json.loads((root / "grid.json").read_text())
    candidate_summary = build_global_branch_candidates(root, force=force)
    candidate_artifact_path = root / (
        f"global-branch-candidates-v{GLOBAL_BRANCH_CANDIDATE_VERSION}.npz"
    )
    flake_paths = [
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(len(grid["z"]))
    ]
    identity = {
        "version": GLOBAL_BRANCH_ASSOCIATION_VERSION,
        "settings": resolved,
        "scheduleIdentity": schedule["identity"],
        "globalGraphIdentity": graph_summary["identity"],
        "candidateIdentity": candidate_summary["identity"],
        "inputArtifacts": [
            _content_identity(path)
            for path in (
                root / "grid.json",
                candidate_artifact_path,
                *flake_paths,
            )
        ],
    }
    include_local_exact_deferred = bool(
        resolved["includeLocalExactDeferredCandidates"]
    )
    include_subwindow_unresolved = bool(
        resolved["includeSubwindowUnresolvedCandidates"]
    )
    include_context_disputed = bool(
        resolved["includeContextDisputedCandidates"]
    )
    include_single_window = bool(resolved["includeSingleWindowCandidates"])
    include_rescue = include_local_exact_deferred or include_subwindow_unresolved
    if (
        include_local_exact_deferred
        and include_subwindow_unresolved
        and include_context_disputed
        and include_single_window
    ):
        mode_suffix = ""
    elif not include_rescue and include_context_disputed and include_single_window:
        mode_suffix = "-accepted-only"
    elif not include_rescue and not include_context_disputed and include_single_window:
        mode_suffix = "-clean-only"
    elif not include_rescue and not include_context_disputed and not include_single_window:
        mode_suffix = "-overlap-only"
    else:
        tier_flags = {
            "localExact": include_local_exact_deferred,
            "subwindow": include_subwindow_unresolved,
            "context": include_context_disputed,
            "single": include_single_window,
        }
        digest = hashlib.sha256(
            json.dumps(tier_flags, sort_keys=True).encode("utf-8")
        ).hexdigest()[:8]
        mode_suffix = f"-custom-{digest}"
    stem = (
        f"global-branch-association-v{GLOBAL_BRANCH_ASSOCIATION_VERSION}"
        f"{mode_suffix}"
    )
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(schedule_artifact_path) as payload:
        tiled = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(graph_artifact_path) as payload:
        graph = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(candidate_artifact_path) as payload:
        rescue = {key: np.asarray(payload[key]) for key in payload.files}

    selected, evidence_provenance = _candidate_tier_selection(
        tiled["branchJoinUnanimous"],
        tiled["branchJoinOverlapValidated"],
        tiled["branchJoinObservationCount"],
        tiled["branchJoinAcceptanceCount"],
        include_single_window,
        include_context_disputed,
        int(resolved["minimumOverlapValidatedObservations"]),
    )
    evidence_endpoint_pair = tiled["branchJoinEndpointIdentity"][selected].astype(
        np.uint64
    )
    evidence_observation_count = tiled["branchJoinObservationCount"][selected].astype(
        np.uint8
    )
    evidence_acceptance_count = tiled["branchJoinAcceptanceCount"][selected].astype(
        np.uint8
    )
    clean_evidence = evidence_provenance != PROVENANCE_CONTEXT_DISPUTED
    if np.any(
        evidence_observation_count[clean_evidence]
        != evidence_acceptance_count[clean_evidence]
    ):
        raise ValueError("a clean-tier global join is not unanimous")
    if not len(evidence_endpoint_pair):
        raise ValueError("no eligible global branch join evidence is available")
    quarantined = {
        (int(value[0]), int(value[1]))
        for value in tiled["quarantinedBranchJoinEndpointIdentity"]
    }
    locally_quarantined = np.asarray(
        [
            (int(value[0]), int(value[1])) in quarantined
            for value in evidence_endpoint_pair
        ],
        dtype=bool,
    )
    excluded_quarantined_endpoint_pair = evidence_endpoint_pair[
        locally_quarantined
    ]
    excluded_quarantined_provenance = evidence_provenance[locally_quarantined]
    excluded_quarantined_observation_count = evidence_observation_count[
        locally_quarantined
    ]
    excluded_quarantined_acceptance_count = evidence_acceptance_count[
        locally_quarantined
    ]
    integrity_clean = ~locally_quarantined
    evidence_endpoint_pair = evidence_endpoint_pair[integrity_clean]
    evidence_provenance = evidence_provenance[integrity_clean]
    evidence_observation_count = evidence_observation_count[integrity_clean]
    evidence_acceptance_count = evidence_acceptance_count[integrity_clean]

    node_identity = graph["nodeIdentity"].astype(np.uint64)
    evidence_endpoint_node = np.searchsorted(
        node_identity, evidence_endpoint_pair
    ).astype(np.uint32)
    endpoint_inside = evidence_endpoint_node < len(node_identity)
    endpoint_matches = np.zeros(evidence_endpoint_pair.shape, dtype=bool)
    endpoint_matches[endpoint_inside] = (
        node_identity[evidence_endpoint_node[endpoint_inside]]
        == evidence_endpoint_pair[endpoint_inside]
    )
    if not np.all(endpoint_matches):
        raise ValueError("a global join endpoint is absent from the graph")
    component = graph["component"].astype(np.uint32)
    evidence_branch_pair = component[evidence_endpoint_node].astype(np.uint32)
    already_linked = evidence_branch_pair[:, 0] == evidence_branch_pair[:, 1]
    already_linked_endpoint_pair = evidence_endpoint_pair[already_linked]
    already_linked_endpoint_node = evidence_endpoint_node[already_linked]
    already_linked_provenance = evidence_provenance[already_linked]
    already_linked_branch = evidence_branch_pair[already_linked, 0]
    already_linked_observation_count = evidence_observation_count[already_linked]
    already_linked_acceptance_count = evidence_acceptance_count[already_linked]
    novel = ~already_linked
    endpoint_pair = evidence_endpoint_pair[novel]
    endpoint_node = evidence_endpoint_node[novel]
    provenance = evidence_provenance[novel]
    branch_pair = evidence_branch_pair[novel]
    observation_count = evidence_observation_count[novel]
    acceptance_count = evidence_acceptance_count[novel]
    accepted_tier_candidate_count = len(endpoint_pair)
    accepted_observations = _candidate_observations(
        root, schedule["windows"], endpoint_pair
    )
    accepted_aggregate = _aggregate_observations(
        accepted_observations, len(endpoint_pair)
    )
    if np.any(accepted_aggregate["count"] != acceptance_count):
        raise ValueError("local join observations do not reproduce tiled acceptance")

    rescue_selected = (
        (
            (rescue["candidateSource"] == SOURCE_LOCAL_EXACT_DEFERRED)
            & include_local_exact_deferred
        )
        | (
            (rescue["candidateSource"] == SOURCE_SUBWINDOW_UNRESOLVED)
            & include_subwindow_unresolved
        )
    )
    rescue_indices = np.flatnonzero(rescue_selected)
    rescue_endpoint_pair = rescue["candidateEndpointIdentity"][
        rescue_indices
    ].astype(np.uint64)
    rescue_endpoint_node = rescue["candidateEndpointNodeIndex"][
        rescue_indices
    ].astype(np.uint32)
    rescue_branch_pair = np.stack(
        (
            rescue["candidateBranchSource"][rescue_indices],
            rescue["candidateBranchTarget"][rescue_indices],
        ),
        axis=1,
    ).astype(np.uint32)
    if len(rescue_indices):
        if not np.all(
            node_identity[rescue_endpoint_node] == rescue_endpoint_pair
        ):
            raise ValueError("a rescue endpoint identity no longer matches its node")
        if not np.array_equal(
            np.sort(component[rescue_endpoint_node], axis=1), rescue_branch_pair
        ):
            raise ValueError("a rescue endpoint no longer matches its global branch")
        endpoint_pair = np.concatenate([endpoint_pair, rescue_endpoint_pair])
        endpoint_node = np.concatenate([endpoint_node, rescue_endpoint_node])
        provenance = np.concatenate(
            [
                provenance,
                rescue["candidateSource"][rescue_indices].astype(np.uint8),
            ]
        )
        branch_pair = np.concatenate([branch_pair, rescue_branch_pair])
        rescue_evidence_count = rescue["candidateEvidenceCount"][
            rescue_indices
        ].astype(np.uint8)
        observation_count = np.concatenate(
            [observation_count, rescue_evidence_count]
        )
        acceptance_count = np.concatenate(
            [acceptance_count, rescue_evidence_count]
        )

    if not len(endpoint_pair):
        raise ValueError("all selected global join evidence is already linked")
    canonical_branch_pair = np.sort(branch_pair, axis=1)
    if len(np.unique(canonical_branch_pair, axis=0)) != len(branch_pair):
        raise ValueError("multiple endpoint joins address the same global branch pair")

    rescue_candidate_map = np.full(
        len(rescue["candidateSource"]), -1, dtype=np.int64
    )
    rescue_candidate_map[rescue_indices] = (
        accepted_tier_candidate_count + np.arange(len(rescue_indices))
    )
    rescue_evidence_selected = rescue_selected[
        rescue["evidenceCandidateIndex"]
    ]
    observations = {
        "candidateIndex": np.concatenate(
            [
                accepted_observations["candidateIndex"],
                rescue_candidate_map[
                    rescue["evidenceCandidateIndex"][rescue_evidence_selected]
                ].astype(np.uint32),
            ]
        ),
        "windowIndex": np.concatenate(
            [
                accepted_observations["windowIndex"],
                rescue["evidenceWindowIndex"][rescue_evidence_selected].astype(
                    np.uint8
                ),
            ]
        ),
        "score": np.concatenate(
            [
                accepted_observations["score"],
                rescue["evidenceScore"][rescue_evidence_selected].astype(
                    np.float32
                ),
            ]
        ),
        "medianHeightResidualVoxels": np.concatenate(
            [
                accepted_observations["medianHeightResidualVoxels"],
                rescue["evidenceExactMedianHeightResidualVoxels"][
                    rescue_evidence_selected
                ].astype(np.float32),
            ]
        ),
        "medianNormalResidualDeg": np.concatenate(
            [
                accepted_observations["medianNormalResidualDeg"],
                rescue["evidenceExactMedianNormalResidualDeg"][
                    rescue_evidence_selected
                ].astype(np.float32),
            ]
        ),
    }
    aggregate = _aggregate_observations(observations, len(endpoint_pair))
    if np.any(aggregate["count"] != acceptance_count):
        raise ValueError("local evidence observations do not reproduce candidate counts")

    touched = np.unique(branch_pair)
    members = _component_members(component, touched)
    selected_nodes = np.concatenate([members[int(value)] for value in touched])
    flakes = _load_selected_flakes(root, node_identity, selected_nodes)
    node_cell = graph["nodeCellIndex"].astype(np.int32)
    shape = np.asarray(
        [len(grid["x"]), len(grid["y"]), len(grid["z"])], dtype=np.int64
    )
    if np.any(node_cell < 0) or np.any(node_cell >= shape):
        raise ValueError("a global graph node lies outside the declared Acus grid")
    cell_code = (
        node_cell[:, 0].astype(np.int64)
        + shape[0]
        * (
            node_cell[:, 1].astype(np.int64)
            + shape[1] * node_cell[:, 2].astype(np.int64)
        )
    )
    branch_cells = {
        int(branch_id): set(int(value) for value in cell_code[nodes])
        for branch_id, nodes in members.items()
    }
    branch_count = int(np.max(component, initial=0)) + 1
    branch_exact = _branch_exact_audit(
        touched,
        members,
        flakes,
        resolved,
        progress,
    )
    source_branch_audit_index = np.searchsorted(touched, branch_pair[:, 0])
    target_branch_audit_index = np.searchsorted(touched, branch_pair[:, 1])
    input_branch_standalone_pass = (
        branch_exact["passed"][source_branch_audit_index]
        & branch_exact["passed"][target_branch_audit_index]
    )
    pair_exact = _pair_exact_audit(
        branch_pair[:, 0],
        branch_pair[:, 1],
        members,
        flakes,
        resolved,
        progress,
    )
    eligible, pair_gate_decision = _pair_gate_state(
        pair_exact["passed"], input_branch_standalone_pass
    )
    group_pruned = np.zeros(len(endpoint_pair), dtype=bool)
    integrity_quarantined = np.zeros(len(endpoint_pair), dtype=bool)
    rounds = []
    integrity_history = []
    final_solve: dict[str, np.ndarray] | None = None
    final_geometries: list[dict[str, Any]] = []
    final_integrity: dict[str, Any] | None = None

    for round_index in range(len(endpoint_pair) + 1):
        solve = _solve_candidate_graph(
            branch_count,
            branch_pair[:, 0],
            branch_pair[:, 1],
            aggregate["minimumScore"],
            eligible,
            branch_cells,
            provenance,
        )
        geometries = _association_geometries(
            solve["branchAssociation"],
            touched,
            members,
            flakes,
            node_cell,
            graph["branchParity"].astype(np.int8),
            resolved,
        )
        failing = [
            value for value in geometries if not bool(value["exact"]["gatesPass"])
        ]
        removed = []
        for geometry in failing:
            association_id = int(geometry["associationId"])
            same_group = (
                solve["branchAssociation"][branch_pair[:, 0]] == association_id
            ) & (
                solve["branchAssociation"][branch_pair[:, 1]] == association_id
            )
            retained_candidates = np.flatnonzero(
                same_group & (solve["decisions"] == GRAPH_RETAINED)
            )
            if not len(retained_candidates):
                raise RuntimeError(
                    "a failing exact association has no construction edge to prune"
                )
            candidate_index = _weakest_retained_candidate(
                retained_candidates,
                aggregate["minimumScore"],
                endpoint_pair,
                provenance,
            )
            eligible[candidate_index] = False
            group_pruned[candidate_index] = True
            removed.append(candidate_index)
        round_value: dict[str, Any] = {
            "round": round_index,
            "eligibleCandidateCount": int(np.count_nonzero(eligible)) + len(removed),
            "retainedMergeCount": int(
                np.count_nonzero(solve["decisions"] == GRAPH_RETAINED)
            ),
            "redundantJoinCount": int(
                np.count_nonzero(solve["decisions"] == GRAPH_REDUNDANT)
            ),
            "cellCollisionCount": int(
                np.count_nonzero(solve["decisions"] == GRAPH_CELL_COLLISION)
            ),
            "mergedAssociationCount": len(geometries),
            "failingExactAssociationCount": len(failing),
            "removedWeakestCandidateIndices": removed,
        }
        if removed:
            rounds.append(round_value)
            if progress is not None:
                progress("group-round", round_index + 1, 0, round_value)
            continue

        integrity = _integrity_audit(geometries, resolved)
        quarantined_indices = np.empty(0, dtype=np.int64)
        if integrity["violations"]:
            quarantined_indices = _weakest_integrity_candidates(
                integrity["violations"],
                solve["branchAssociation"],
                branch_pair,
                solve["decisions"],
                eligible,
                aggregate["minimumScore"],
                endpoint_pair,
                provenance,
            )
            eligible[quarantined_indices] = False
            integrity_quarantined[quarantined_indices] = True
        round_value["integrity"] = integrity["stats"]
        round_value["integrityQuarantinedCandidateIndices"] = (
            quarantined_indices.astype(int).tolist()
        )
        rounds.append(round_value)
        integrity_history.append(integrity)
        if progress is not None:
            progress("group-round", round_index + 1, 0, round_value)
        if len(quarantined_indices):
            continue
        final_solve = solve
        final_geometries = geometries
        final_integrity = integrity
        break
    else:
        raise RuntimeError("global exact/integrity pruning did not converge")

    if final_solve is None or final_integrity is None:
        raise RuntimeError("global association solve did not produce a final state")
    final_decision = pair_gate_decision.copy()
    final_decision[group_pruned] = DECISION_EXACT_GROUP_PRUNED
    final_decision[integrity_quarantined] = DECISION_INTEGRITY_QUARANTINED
    remaining = eligible
    final_decision[
        remaining & (final_solve["decisions"] == GRAPH_CELL_COLLISION)
    ] = DECISION_CELL_COLLISION
    final_decision[
        remaining & (final_solve["decisions"] == GRAPH_RETAINED)
    ] = DECISION_RETAINED
    final_decision[
        remaining & (final_solve["decisions"] == GRAPH_REDUNDANT)
    ] = DECISION_REDUNDANT
    if np.any(final_decision == 0):
        raise RuntimeError("one global association candidate has no final decision")

    branch_association = final_solve["branchAssociation"].astype(np.uint32)
    node_association = branch_association[component]
    association_branch_count = np.bincount(branch_association).astype(np.uint32)
    association_node_count = np.bincount(
        node_association, minlength=len(association_branch_count)
    ).astype(np.uint32)
    global_branch_node_count = np.bincount(
        component, minlength=branch_count
    ).astype(np.uint32)
    linked_global_branch = global_branch_node_count >= 2
    linked_association_count = len(
        np.unique(branch_association[linked_global_branch])
    )
    construction_join = np.isin(
        final_decision, [DECISION_RETAINED, DECISION_REDUNDANT]
    )
    construction_association = branch_association[
        branch_pair[:, 0][construction_join]
    ]
    association_overlap_join_count = np.bincount(
        construction_association[
            provenance[construction_join] == PROVENANCE_OVERLAP_VALIDATED
        ],
        minlength=len(association_branch_count),
    ).astype(np.uint16)
    association_single_window_join_count = np.bincount(
        construction_association[
            provenance[construction_join] == PROVENANCE_SINGLE_WINDOW
        ],
        minlength=len(association_branch_count),
    ).astype(np.uint16)
    association_context_disputed_join_count = np.bincount(
        construction_association[
            provenance[construction_join] == PROVENANCE_CONTEXT_DISPUTED
        ],
        minlength=len(association_branch_count),
    ).astype(np.uint16)
    association_subwindow_unresolved_join_count = np.bincount(
        construction_association[
            provenance[construction_join] == PROVENANCE_SUBWINDOW_UNRESOLVED
        ],
        minlength=len(association_branch_count),
    ).astype(np.uint16)
    association_local_exact_deferred_join_count = np.bincount(
        construction_association[
            provenance[construction_join] == PROVENANCE_LOCAL_EXACT_DEFERRED
        ],
        minlength=len(association_branch_count),
    ).astype(np.uint16)
    exact_groups = [value["exact"] for value in final_geometries]
    for value in exact_groups:
        association_id = int(value["associationId"])
        overlap_count = int(association_overlap_join_count[association_id])
        single_count = int(
            association_single_window_join_count[association_id]
        )
        context_count = int(
            association_context_disputed_join_count[association_id]
        )
        subwindow_count = int(
            association_subwindow_unresolved_join_count[association_id]
        )
        local_exact_count = int(
            association_local_exact_deferred_join_count[association_id]
        )
        value["overlapValidatedJoinCount"] = overlap_count
        value["singleWindowJoinCount"] = single_count
        value["contextDisputedJoinCount"] = context_count
        value["subwindowUnresolvedJoinCount"] = subwindow_count
        value["localExactDeferredJoinCount"] = local_exact_count
        active_provenance = [
            name
            for name, count in (
                ("overlap-validated", overlap_count),
                ("single-window", single_count),
                ("context-disputed", context_count),
                ("subwindow-unresolved", subwindow_count),
                ("local-exact-deferred", local_exact_count),
            )
            if count
        ]
        value["provenance"] = (
            f"{active_provenance[0]}-only"
            if len(active_provenance) == 1
            else "mixed"
        )
    violation_records = [
        (history_index, value)
        for history_index, history in enumerate(integrity_history)
        for value in history["violations"]
    ]
    intersection_records = [
        (violation_index, point)
        for violation_index, (_, violation) in enumerate(violation_records)
        for point in violation["stored"]
    ]
    _atomic_npz(
        artifact_path,
        candidateEndpointIdentity=endpoint_pair,
        candidateEndpointNodeIndex=endpoint_node,
        candidateProvenance=provenance,
        candidateBranchSource=branch_pair[:, 0],
        candidateBranchTarget=branch_pair[:, 1],
        candidateObservationCount=observation_count,
        candidateAcceptanceCount=acceptance_count,
        candidateMinimumScore=aggregate["minimumScore"],
        candidateMeanScore=aggregate["meanScore"],
        candidateMaximumScore=aggregate["maximumScore"],
        candidateMaximumLocalMedianHeightResidualVoxels=aggregate[
            "maximumLocalMedianHeightResidualVoxels"
        ],
        candidateMaximumLocalMedianNormalResidualDeg=aggregate[
            "maximumLocalMedianNormalResidualDeg"
        ],
        candidateExactMedianHeightResidualVoxels=pair_exact[
            "medianHeightResidualVoxels"
        ],
        candidateExactP90HeightResidualVoxels=pair_exact[
            "p90HeightResidualVoxels"
        ],
        candidateExactMedianNormalResidualDeg=pair_exact[
            "medianNormalResidualDeg"
        ],
        candidateExactP90NormalResidualDeg=pair_exact[
            "p90NormalResidualDeg"
        ],
        candidateExactPass=pair_exact["passed"],
        candidateInputBranchStandalonePass=input_branch_standalone_pass,
        candidateExactGroupPruned=group_pruned,
        candidateIntegrityQuarantined=integrity_quarantined,
        candidateFinalDecision=final_decision,
        alreadyLinkedEndpointIdentity=already_linked_endpoint_pair,
        alreadyLinkedEndpointNodeIndex=already_linked_endpoint_node,
        alreadyLinkedProvenance=already_linked_provenance,
        alreadyLinkedBranch=already_linked_branch,
        alreadyLinkedObservationCount=already_linked_observation_count,
        alreadyLinkedAcceptanceCount=already_linked_acceptance_count,
        excludedQuarantinedEndpointIdentity=excluded_quarantined_endpoint_pair,
        excludedQuarantinedProvenance=excluded_quarantined_provenance,
        excludedQuarantinedObservationCount=(
            excluded_quarantined_observation_count
        ),
        excludedQuarantinedAcceptanceCount=(
            excluded_quarantined_acceptance_count
        ),
        observationCandidateIndex=observations["candidateIndex"],
        observationWindowIndex=observations["windowIndex"],
        observationScore=observations["score"],
        observationMedianHeightResidualVoxels=observations[
            "medianHeightResidualVoxels"
        ],
        observationMedianNormalResidualDeg=observations[
            "medianNormalResidualDeg"
        ],
        auditedBranchId=touched,
        auditedBranchExactMedianHeightResidualVoxels=branch_exact[
            "medianHeightResidualVoxels"
        ],
        auditedBranchExactP90HeightResidualVoxels=branch_exact[
            "p90HeightResidualVoxels"
        ],
        auditedBranchExactMedianNormalResidualDeg=branch_exact[
            "medianNormalResidualDeg"
        ],
        auditedBranchExactP90NormalResidualDeg=branch_exact[
            "p90NormalResidualDeg"
        ],
        auditedBranchExactPass=branch_exact["passed"],
        branchAssociation=branch_association,
        nodeAssociation=node_association,
        associationBranchCount=association_branch_count,
        associationNodeCount=association_node_count,
        associationOverlapValidatedJoinCount=association_overlap_join_count,
        associationSingleWindowJoinCount=association_single_window_join_count,
        associationContextDisputedJoinCount=(
            association_context_disputed_join_count
        ),
        associationSubwindowUnresolvedJoinCount=(
            association_subwindow_unresolved_join_count
        ),
        associationLocalExactDeferredJoinCount=(
            association_local_exact_deferred_join_count
        ),
        integrityRoundIndex=np.asarray(
            [value[0] for value in violation_records], dtype=np.uint16
        ),
        integrityAssociationSource=np.asarray(
            [value[1]["associationSource"] for value in violation_records],
            dtype=np.uint32,
        ),
        integrityAssociationTarget=np.asarray(
            [value[1]["associationTarget"] for value in violation_records],
            dtype=np.uint32,
        ),
        integrityIntersectingTrianglePairCount=np.asarray(
            [
                value[1]["intersectingTrianglePairCount"]
                for value in violation_records
            ],
            dtype=np.uint32,
        ),
        integrityEvidenceCoreIntersectingTrianglePairCount=np.asarray(
            [
                value[1]["evidenceCoreIntersectingTrianglePairCount"]
                for value in violation_records
            ],
            dtype=np.uint32,
        ),
        integrityIntersectionViolationIndex=np.asarray(
            [value[0] for value in intersection_records], dtype=np.uint32
        ),
        integrityIntersectionPointXYZ=np.asarray(
            [value[1]["point"] for value in intersection_records],
            dtype=np.float32,
        ).reshape(-1, 3),
        integrityIntersectionEvidenceCore=np.asarray(
            [value[1]["evidenceCore"] for value in intersection_records],
            dtype=bool,
        ),
        integrityIntersectionCoplanar=np.asarray(
            [value[1]["coplanar"] for value in intersection_records],
            dtype=bool,
        ),
    )

    top_associations = sorted(
        exact_groups,
        key=lambda value: (-int(value["flakeCount"]), int(value["associationId"])),
    )[:20]
    deferred_candidates = []
    for candidate_index in np.flatnonzero(
        ~np.isin(final_decision, [DECISION_RETAINED, DECISION_REDUNDANT])
    ):
        source_audit_index = int(source_branch_audit_index[candidate_index])
        target_audit_index = int(target_branch_audit_index[candidate_index])
        observation_indices = np.flatnonzero(
            observations["candidateIndex"] == candidate_index
        )
        deferred_candidates.append(
            {
                "candidateIndex": int(candidate_index),
                "decision": DECISION_NAMES[int(final_decision[candidate_index])],
                "provenance": PROVENANCE_NAMES[int(provenance[candidate_index])],
                "endpointIdentity": endpoint_pair[candidate_index].astype(int).tolist(),
                "endpointCellXYZ": node_cell[
                    endpoint_node[candidate_index]
                ].astype(int).tolist(),
                "branchSource": int(branch_pair[candidate_index, 0]),
                "branchTarget": int(branch_pair[candidate_index, 1]),
                "sourceBranchFlakeCount": len(
                    members[int(branch_pair[candidate_index, 0])]
                ),
                "targetBranchFlakeCount": len(
                    members[int(branch_pair[candidate_index, 1])]
                ),
                "minimumScore": round(
                    float(aggregate["minimumScore"][candidate_index]), 6
                ),
                "observationCount": int(observation_count[candidate_index]),
                "acceptanceCount": int(acceptance_count[candidate_index]),
                "unacceptedObservationCount": int(
                    observation_count[candidate_index]
                    - acceptance_count[candidate_index]
                ),
                "observedWindowOrigins": [
                    schedule["windows"][int(window_index)]["originCellXYZ"]
                    for window_index in observations["windowIndex"][
                        observation_indices
                    ]
                ],
                "sourceStandaloneCarrier": {
                    "medianHeightResidualVoxels": float(
                        branch_exact["medianHeightResidualVoxels"][
                            source_audit_index
                        ]
                    ),
                    "medianNormalResidualDeg": float(
                        branch_exact["medianNormalResidualDeg"][
                            source_audit_index
                        ]
                    ),
                    "gatesPass": bool(branch_exact["passed"][source_audit_index]),
                },
                "targetStandaloneCarrier": {
                    "medianHeightResidualVoxels": float(
                        branch_exact["medianHeightResidualVoxels"][
                            target_audit_index
                        ]
                    ),
                    "medianNormalResidualDeg": float(
                        branch_exact["medianNormalResidualDeg"][
                            target_audit_index
                        ]
                    ),
                    "gatesPass": bool(branch_exact["passed"][target_audit_index]),
                },
                "joinedCarrier": {
                    "medianHeightResidualVoxels": float(
                        pair_exact["medianHeightResidualVoxels"][candidate_index]
                    ),
                    "medianNormalResidualDeg": float(
                        pair_exact["medianNormalResidualDeg"][candidate_index]
                    ),
                    "gatesPass": bool(pair_exact["passed"][candidate_index]),
                },
            }
        )
    already_linked_evidence = [
        {
            "evidenceIndex": int(index),
            "provenance": PROVENANCE_NAMES[
                int(already_linked_provenance[index])
            ],
            "endpointIdentity": already_linked_endpoint_pair[index]
            .astype(int)
            .tolist(),
            "endpointCellXYZ": node_cell[
                already_linked_endpoint_node[index]
            ]
            .astype(int)
            .tolist(),
            "globalBranch": int(already_linked_branch[index]),
            "branchFlakeCount": int(
                global_branch_node_count[int(already_linked_branch[index])]
            ),
            "observationCount": int(already_linked_observation_count[index]),
            "acceptanceCount": int(already_linked_acceptance_count[index]),
        }
        for index in range(len(already_linked_endpoint_pair))
    ]
    excluded_quarantined_evidence = [
        {
            "evidenceIndex": int(index),
            "provenance": PROVENANCE_NAMES[
                int(excluded_quarantined_provenance[index])
            ],
            "endpointIdentity": excluded_quarantined_endpoint_pair[index]
            .astype(int)
            .tolist(),
            "observationCount": int(
                excluded_quarantined_observation_count[index]
            ),
            "acceptanceCount": int(
                excluded_quarantined_acceptance_count[index]
            ),
            "reason": "a local association containing this join failed the mesh-intersection audit",
        }
        for index in range(len(excluded_quarantined_endpoint_pair))
    ]
    included_tiers = ["overlap-validated"]
    if include_single_window:
        included_tiers.append("single-window")
    if include_context_disputed:
        included_tiers.append("context-disputed")
    if include_subwindow_unresolved:
        included_tiers.append("subwindow-unresolved")
    if include_local_exact_deferred:
        included_tiers.append("local-exact-deferred")
    input_rule = (
        "integrity-clean endpoint evidence enters in explicit provenance tiers: "
        + ", ".join(included_tiers)
    )
    if include_rescue:
        input_rule += (
            "; the rescue tiers passed local score, material, order, and collision "
            "checks but remain candidates until complete-global-branch reconstruction"
        )
    result = {
        "identity": identity,
        "contract": {
            "inputRule": input_rule,
            "scoreRule": (
                "the minimum accepted local score controls deterministic global "
                "construction; overlap-validated, then single-window, then "
                "context-disputed, then subwindow-unresolved, then local-exact-deferred "
                "evidence wins exact score ties, and every supporting local score, "
                "residual, and provenance label is retained"
            ),
            "pairGate": (
                "each join reconstructed from its complete global branches must pass "
                "the 3-voxel/6-degree median MLS gates; standalone branch fits are "
                "diagnostic because sparse fragments may stabilize only after joining"
            ),
            "transitiveGate": (
                "descending-score joins may never create a repeated Acus cell; each "
                "merged carrier is reconstructed and its weakest construction edge "
                "is pruned until exact geometry passes"
            ),
            "integrityGate": (
                "merged association carriers are triangle-intersection audited; all "
                "intersecting carrier pairs lose their weakest retained construction "
                "edge and the solve repeats to a zero-intersection fixed point; "
                "unassociated global branches remain outside this audit"
            ),
            "directionMeaning": (
                "normal and fiber axes remain unsigned; branch parity is only a "
                "local gauge coordinate and no physical side is assigned"
            ),
            "identityMeaning": (
                "output associations are sparse exact-coherent surface hypotheses, "
                "not papyrus sheets, pages, winding order, recto, or verso"
            ),
        },
        "settings": resolved,
        "rounds": rounds,
        "topAssociations": top_associations,
        "deferredCandidates": deferred_candidates,
        "alreadyLinkedEvidence": already_linked_evidence,
        "excludedQuarantinedEvidence": excluded_quarantined_evidence,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "inputTiledJoinCount": len(tiled["branchJoinEndpointIdentity"]),
            "inputQuarantinedJoinCount": len(
                tiled["quarantinedBranchJoinEndpointIdentity"]
            ),
            "inputRescueCandidateCount": int(
                candidate_summary["stats"]["candidateBranchPairCount"]
            ),
            "eligibleTierEvidenceJoinCount": int(np.count_nonzero(selected)),
            "excludedQuarantinedEvidenceCount": len(
                excluded_quarantined_endpoint_pair
            ),
            "selectedEvidenceJoinCount": len(evidence_endpoint_pair),
            "alreadyLinkedEvidenceCount": len(already_linked_endpoint_pair),
            "alreadyLinkedContextDisputedEvidenceCount": int(
                np.count_nonzero(
                    already_linked_provenance
                    == PROVENANCE_CONTEXT_DISPUTED
                )
            ),
            "selectedCandidateJoinCount": len(endpoint_pair),
            "selectedAcceptedTierCandidateJoinCount": (
                accepted_tier_candidate_count
            ),
            "selectedRescueCandidateJoinCount": len(rescue_indices),
            "selectedOverlapValidatedJoinCount": int(
                np.count_nonzero(
                    provenance == PROVENANCE_OVERLAP_VALIDATED
                )
            ),
            "selectedSingleWindowJoinCount": int(
                np.count_nonzero(provenance == PROVENANCE_SINGLE_WINDOW)
            ),
            "selectedContextDisputedJoinCount": int(
                np.count_nonzero(
                    provenance == PROVENANCE_CONTEXT_DISPUTED
                )
            ),
            "selectedSubwindowUnresolvedJoinCount": int(
                np.count_nonzero(
                    provenance == PROVENANCE_SUBWINDOW_UNRESOLVED
                )
            ),
            "selectedLocalExactDeferredJoinCount": int(
                np.count_nonzero(
                    provenance == PROVENANCE_LOCAL_EXACT_DEFERRED
                )
            ),
            "selectedTouchedGlobalBranchCount": len(touched),
            "localObservationCount": len(observations["score"]),
            "minimumCandidateScore": _quantiles(aggregate["minimumScore"]),
            "candidateScoreObservationRange": _quantiles(
                aggregate["maximumScore"] - aggregate["minimumScore"], 7
            ),
            "overlapValidatedCandidateScoreObservationRange": _quantiles(
                (
                    aggregate["maximumScore"] - aggregate["minimumScore"]
                )[provenance == PROVENANCE_OVERLAP_VALIDATED],
                7,
            ),
            "singleWindowCandidateScoreObservationRange": _quantiles(
                (
                    aggregate["maximumScore"] - aggregate["minimumScore"]
                )[provenance == PROVENANCE_SINGLE_WINDOW],
                7,
            ),
            "contextDisputedCandidateScoreObservationRange": _quantiles(
                (
                    aggregate["maximumScore"] - aggregate["minimumScore"]
                )[provenance == PROVENANCE_CONTEXT_DISPUTED],
                7,
            ),
            "subwindowUnresolvedCandidateScoreObservationRange": _quantiles(
                (
                    aggregate["maximumScore"] - aggregate["minimumScore"]
                )[provenance == PROVENANCE_SUBWINDOW_UNRESOLVED],
                7,
            ),
            "localExactDeferredCandidateScoreObservationRange": _quantiles(
                (
                    aggregate["maximumScore"] - aggregate["minimumScore"]
                )[provenance == PROVENANCE_LOCAL_EXACT_DEFERRED],
                7,
            ),
            "pairExactPassCount": int(np.count_nonzero(pair_exact["passed"])),
            "pairExactDeferredCount": int(
                np.count_nonzero(
                    input_branch_standalone_pass & ~pair_exact["passed"]
                )
            ),
            "auditedInputBranchCount": len(touched),
            "standaloneCarrierPassingInputBranchCount": int(
                np.count_nonzero(branch_exact["passed"])
            ),
            "standaloneCarrierFailingInputBranchCount": int(
                np.count_nonzero(~branch_exact["passed"])
            ),
            "candidateWithStandaloneCarrierFailureCount": int(
                np.count_nonzero(~input_branch_standalone_pass)
            ),
            "standaloneFailureCandidatePassingJoinedCarrierCount": int(
                np.count_nonzero(
                    ~input_branch_standalone_pass & pair_exact["passed"]
                )
            ),
            "inputBranchMedianHeightResidualVoxels": _quantiles(
                branch_exact["medianHeightResidualVoxels"]
            ),
            "inputBranchMedianNormalResidualDeg": _quantiles(
                branch_exact["medianNormalResidualDeg"]
            ),
            "pairMedianHeightResidualVoxels": _quantiles(
                pair_exact["medianHeightResidualVoxels"]
            ),
            "pairMedianNormalResidualDeg": _quantiles(
                pair_exact["medianNormalResidualDeg"]
            ),
            "pairMedianHeightChangeFromWorstLocalVoxels": _quantiles(
                pair_exact["medianHeightResidualVoxels"]
                - aggregate["maximumLocalMedianHeightResidualVoxels"]
            ),
            "pairMedianNormalChangeFromWorstLocalDeg": _quantiles(
                pair_exact["medianNormalResidualDeg"]
                - aggregate["maximumLocalMedianNormalResidualDeg"]
            ),
            "finalDecisionCounts": {
                name: int(np.count_nonzero(final_decision == value))
                for value, name in DECISION_NAMES.items()
            },
            "finalDecisionCountsByProvenance": {
                provenance_name: {
                    name: int(
                        np.count_nonzero(
                            (provenance == provenance_value)
                            & (final_decision == decision_value)
                        )
                    )
                    for decision_value, name in DECISION_NAMES.items()
                }
                for provenance_value, provenance_name in PROVENANCE_NAMES.items()
            },
            "exactGroupPrunedCount": int(np.count_nonzero(group_pruned)),
            "integrityQuarantinedCount": int(
                np.count_nonzero(integrity_quarantined)
            ),
            "finalRetainedMergeCount": int(
                np.count_nonzero(final_decision == DECISION_RETAINED)
            ),
            "finalRedundantJoinCount": int(
                np.count_nonzero(final_decision == DECISION_REDUNDANT)
            ),
            "finalMergedAssociationCount": len(exact_groups),
            "finalAssociatedGlobalBranchCount": int(
                np.sum(association_branch_count[association_branch_count >= 2])
            ),
            "initialLinkedGlobalBranchCount": int(
                np.count_nonzero(linked_global_branch)
            ),
            "finalLinkedGlobalBranchGroupCount": linked_association_count,
            "linkedGlobalBranchReductionCount": int(
                np.count_nonzero(linked_global_branch) - linked_association_count
            ),
            "finalAssociatedNodeCount": int(
                np.sum(association_node_count[association_branch_count >= 2])
            ),
            "finalAssociationBranchCountDistribution": {
                str(int(value)): int(
                    np.count_nonzero(association_branch_count == value)
                )
                for value in np.unique(
                    association_branch_count[association_branch_count >= 2]
                )
            },
            "finalAssociationProvenanceCounts": {
                name: sum(value["provenance"] == name for value in exact_groups)
                for name in (
                    "overlap-validated-only",
                    "single-window-only",
                    "context-disputed-only",
                    "subwindow-unresolved-only",
                    "local-exact-deferred-only",
                    "mixed",
                )
            },
            "finalAssociationNodeCount": _quantiles(
                np.asarray([value["flakeCount"] for value in exact_groups])
            ),
            "finalAssociationAxialPlaneCount": _quantiles(
                np.asarray([value["axialPlaneCount"] for value in exact_groups])
            ),
            "finalAllAxialPlaneAssociationCount": int(
                np.count_nonzero(
                    [
                        int(value["axialPlaneCount"]) == int(shape[2])
                        for value in exact_groups
                    ]
                )
            ),
            "finalLongSpanAssociationCount": int(
                np.count_nonzero(
                    [int(value["axialPlaneCount"]) >= 11 for value in exact_groups]
                )
            ),
            "largestAssociationBranchCount": int(
                np.max(association_branch_count, initial=0)
            ),
            "largestAssociationNodeCount": int(
                np.max(
                    association_node_count[association_branch_count >= 2],
                    initial=0,
                )
            ),
            "finalExactFailureCount": int(
                np.count_nonzero(
                    [not bool(value["gatesPass"]) for value in exact_groups]
                )
            ),
            "finalGroupMedianHeightResidualVoxels": _quantiles(
                np.asarray(
                    [value["medianHeightResidualVoxels"] for value in exact_groups]
                )
            ),
            "finalGroupMedianNormalResidualDeg": _quantiles(
                np.asarray(
                    [value["medianNormalResidualDeg"] for value in exact_groups]
                )
            ),
            "finalIntegrity": final_integrity["stats"],
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
