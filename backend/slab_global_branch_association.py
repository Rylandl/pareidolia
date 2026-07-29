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
    DECISION_REDUNDANT,
    DECISION_RETAINED,
    DEFAULT_SETTINGS as BRANCH_SETTINGS,
    _exact_carrier_stats,
)
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


GLOBAL_BRANCH_ASSOCIATION_VERSION = 1

DECISION_EXACT_PAIR_DEFERRED = 1
DECISION_CELL_COLLISION = 2
DECISION_EXACT_GROUP_PRUNED = 3
DECISION_INTEGRITY_QUARANTINED = 4
DECISION_RETAINED = 5
DECISION_REDUNDANT = 6

DECISION_NAMES = {
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
    "minimumOverlapObservations": 2,
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
    score: np.ndarray, source: np.ndarray, target: np.ndarray
) -> np.ndarray:
    return np.lexsort(
        (
            np.asarray(target, dtype=np.int64),
            np.asarray(source, dtype=np.int64),
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
    for candidate_index in _candidate_order(score, source, target):
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
                [DECISION_RETAINED, DECISION_REDUNDANT],
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
    np.maximum.at(
        maximum_height, index, observations["medianHeightResidualVoxels"]
    )
    np.maximum.at(
        maximum_normal, index, observations["medianNormalResidualDeg"]
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
) -> int:
    return int(
        min(
            candidate_indices,
            key=lambda index: (
                float(score[index]),
                int(endpoint_pair[index, 0]),
                int(endpoint_pair[index, 1]),
            ),
        )
    )


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
    local_paths = []
    for window in schedule["windows"]:
        suffix = window_artifact_suffix(window["originCellXYZ"])
        local_paths.extend(
            (
                root
                / f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz",
                root
                / f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}.npz",
            )
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
        "inputArtifacts": [
            _content_identity(path)
            for path in (
                root / "grid.json",
                schedule_artifact_path,
                graph_artifact_path,
                *local_paths,
                *flake_paths,
            )
        ],
    }
    stem = f"global-branch-association-v{GLOBAL_BRANCH_ASSOCIATION_VERSION}"
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

    selected = (
        tiled["branchJoinUnanimous"].astype(bool)
        & tiled["branchJoinOverlapValidated"].astype(bool)
        & (
            tiled["branchJoinObservationCount"]
            >= int(resolved["minimumOverlapObservations"])
        )
    )
    endpoint_pair = tiled["branchJoinEndpointIdentity"][selected].astype(np.uint64)
    observation_count = tiled["branchJoinObservationCount"][selected].astype(
        np.uint8
    )
    acceptance_count = tiled["branchJoinAcceptanceCount"][selected].astype(
        np.uint8
    )
    if np.any(observation_count != acceptance_count):
        raise ValueError("a selected global join is not unanimous")
    if not len(endpoint_pair):
        raise ValueError("no overlap-validated global branch joins are available")
    quarantined = {
        (int(value[0]), int(value[1]))
        for value in tiled["quarantinedBranchJoinEndpointIdentity"]
    }
    if any((int(value[0]), int(value[1])) in quarantined for value in endpoint_pair):
        raise ValueError("an integrity-quarantined local join entered global input")

    node_identity = graph["nodeIdentity"].astype(np.uint64)
    endpoint_node = np.searchsorted(node_identity, endpoint_pair).astype(np.uint32)
    endpoint_inside = endpoint_node < len(node_identity)
    endpoint_matches = np.zeros(endpoint_pair.shape, dtype=bool)
    endpoint_matches[endpoint_inside] = (
        node_identity[endpoint_node[endpoint_inside]]
        == endpoint_pair[endpoint_inside]
    )
    if not np.all(endpoint_matches):
        raise ValueError("a global join endpoint is absent from the graph")
    component = graph["component"].astype(np.uint32)
    branch_pair = component[endpoint_node].astype(np.uint32)
    if np.any(branch_pair[:, 0] == branch_pair[:, 1]):
        raise ValueError("a selected gap join is already inside one global branch")
    canonical_branch_pair = np.sort(branch_pair, axis=1)
    if len(np.unique(canonical_branch_pair, axis=0)) != len(branch_pair):
        raise ValueError("multiple endpoint joins address the same global branch pair")

    observations = _candidate_observations(
        root, schedule["windows"], endpoint_pair
    )
    aggregate = _aggregate_observations(observations, len(endpoint_pair))
    if np.any(aggregate["count"] != acceptance_count):
        raise ValueError("local join observations do not reproduce tiled acceptance")

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
    pair_exact = _pair_exact_audit(
        branch_pair[:, 0],
        branch_pair[:, 1],
        members,
        flakes,
        resolved,
        progress,
    )
    eligible = pair_exact["passed"].copy()
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
        violating_associations = {
            int(value[key])
            for value in integrity["violations"]
            for key in ("associationSource", "associationTarget")
        }
        quarantined_indices = np.empty(0, dtype=np.int64)
        if violating_associations:
            source_association = solve["branchAssociation"][branch_pair[:, 0]]
            target_association = solve["branchAssociation"][branch_pair[:, 1]]
            quarantine = (
                eligible
                & (source_association == target_association)
                & np.isin(source_association, list(violating_associations))
            )
            quarantined_indices = np.flatnonzero(quarantine)
            if not len(quarantined_indices):
                raise RuntimeError(
                    "an integrity violation has no candidate joins to quarantine"
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
    final_decision = np.zeros(len(endpoint_pair), dtype=np.uint8)
    final_decision[~pair_exact["passed"]] = DECISION_EXACT_PAIR_DEFERRED
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
        candidateExactGroupPruned=group_pruned,
        candidateIntegrityQuarantined=integrity_quarantined,
        candidateFinalDecision=final_decision,
        observationCandidateIndex=observations["candidateIndex"],
        observationWindowIndex=observations["windowIndex"],
        observationScore=observations["score"],
        observationMedianHeightResidualVoxels=observations[
            "medianHeightResidualVoxels"
        ],
        observationMedianNormalResidualDeg=observations[
            "medianNormalResidualDeg"
        ],
        branchAssociation=branch_association,
        nodeAssociation=node_association,
        associationBranchCount=association_branch_count,
        associationNodeCount=association_node_count,
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

    exact_groups = [value["exact"] for value in final_geometries]
    top_associations = sorted(
        exact_groups,
        key=lambda value: (-int(value["flakeCount"]), int(value["associationId"])),
    )[:20]
    result = {
        "identity": identity,
        "contract": {
            "inputRule": (
                "only integrity-clean endpoint joins accepted by every observing "
                "window and observed in at least two windows enter the solve"
            ),
            "scoreRule": (
                "the minimum local candidate score controls deterministic global "
                "construction; every local score and exact residual is retained"
            ),
            "pairGate": (
                "each join is reconstructed from both complete global branches and "
                "must pass the 3-voxel/6-degree median MLS carrier gates"
            ),
            "transitiveGate": (
                "descending-score joins may never create a repeated Acus cell; each "
                "merged carrier is reconstructed and its weakest construction edge "
                "is pruned until exact geometry passes"
            ),
            "integrityGate": (
                "merged association carriers are triangle-intersection audited; all "
                "joins in any intersecting association are quarantined and the solve "
                "is repeated; unassociated global branches remain outside this audit"
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
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "inputTiledJoinCount": len(tiled["branchJoinEndpointIdentity"]),
            "inputQuarantinedJoinCount": len(
                tiled["quarantinedBranchJoinEndpointIdentity"]
            ),
            "selectedOverlapValidatedJoinCount": len(endpoint_pair),
            "selectedTouchedGlobalBranchCount": len(touched),
            "localObservationCount": len(observations["score"]),
            "minimumCandidateScore": _quantiles(aggregate["minimumScore"]),
            "candidateScoreObservationRange": _quantiles(
                aggregate["maximumScore"] - aggregate["minimumScore"], 7
            ),
            "pairExactPassCount": int(np.count_nonzero(pair_exact["passed"])),
            "pairExactDeferredCount": int(
                np.count_nonzero(~pair_exact["passed"])
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
