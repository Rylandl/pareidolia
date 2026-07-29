from __future__ import annotations

import hashlib
import json
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .slab_association_integrity import (
    ASSOCIATION_INTEGRITY_VERSION,
    association_integrity_audit,
)
from .slab_branch_association import (
    BRANCH_ASSOCIATION_VERSION,
    DECISION_REDUNDANT,
    DECISION_RETAINED,
    associate_monotone_branches,
)
from .slab_material_intervals import MATERIAL_INTERVAL_VERSION
from .slab_monotone_layers import (
    MONOTONE_LAYER_VERSION,
    prototype_monotone_layers,
    window_artifact_suffix,
)
from .slab_sheetlet_explore import SHEETLET_EXPLORE_VERSION
from .slab_window_reconciliation import (
    WINDOW_RECONCILIATION_VERSION,
    reconcile_overlapping_windows,
)


WINDOW_SCHEDULER_VERSION = 4

DEFAULT_SETTINGS: dict[str, Any] = {
    "windowCellsXYZ": [32, 32, 14],
    "strideCellsXYZ": [24, 24, 14],
    "minimumPrimaryFlakeClaimCount": 1,
    "maximumWorkers": 4,
    "runIntegrityAudit": True,
    "quarantineIntegrityViolations": True,
    "minimumFlakeQuality": 0.08,
    "minimumLinkScore": 0.60,
    "gapPenalty": 0.02,
    "edgePaddingVoxels": 8.0,
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


def _axis_origins(length: int, width: int, stride: int) -> list[int]:
    if width <= 0 or stride <= 0 or width > length:
        raise ValueError("invalid window width or stride")
    output = list(range(0, length - width + 1, stride))
    final = length - width
    if output[-1] != final:
        output.append(final)
    return output


def _window_catalog(
    root: Path, settings: dict[str, Any]
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    grid = json.loads((root / "grid.json").read_text())
    profiles = np.load(
        root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}-profiles.npy",
        mmap_mode="r",
    )
    grid_shape_xyz = np.asarray(
        [len(grid["x"]), len(grid["y"]), len(grid["z"])], dtype=np.int32
    )
    window_shape = np.asarray(settings["windowCellsXYZ"], dtype=np.int32)
    stride = np.asarray(settings["strideCellsXYZ"], dtype=np.int32)
    origins = [
        _axis_origins(
            int(grid_shape_xyz[axis]),
            int(window_shape[axis]),
            int(stride[axis]),
        )
        for axis in range(3)
    ]
    counts = np.zeros(
        tuple(int(value) for value in grid_shape_xyz[::-1]), dtype=np.uint32
    )
    primary = profiles["normalFamily"] == 0
    cell = np.asarray(profiles["cellIndex"][primary], dtype=np.int32)
    counts[cell[:, 2], cell[:, 1], cell[:, 0]] = profiles["claimCount"][primary]
    windows = []
    skipped = []
    minimum_claims = int(settings["minimumPrimaryFlakeClaimCount"])
    for z_index in origins[2]:
        for y_index in origins[1]:
            for x_index in origins[0]:
                origin = np.asarray([x_index, y_index, z_index], dtype=np.int32)
                stop = origin + window_shape
                claim_count = int(
                    np.sum(
                        counts[
                            origin[2] : stop[2],
                            origin[1] : stop[1],
                            origin[0] : stop[0],
                        ]
                    )
                )
                value = {
                    "originCellXYZ": origin.astype(int).tolist(),
                    "stopCellXYZExclusive": stop.astype(int).tolist(),
                    "primaryFlakeClaimCount": claim_count,
                }
                if claim_count >= minimum_claims:
                    windows.append(value)
                else:
                    skipped.append(value)
    for window_index, value in enumerate(windows):
        value["windowIndex"] = window_index
    return windows, {
        "gridShapeCellsXYZ": grid_shape_xyz.astype(int).tolist(),
        "possibleWindowCount": len(windows) + len(skipped),
        "occupiedWindowCount": len(windows),
        "skippedWindowCount": len(skipped),
        "skippedPrimaryFlakeClaimCount": sum(
            value["primaryFlakeClaimCount"] for value in skipped
        ),
        "axisOriginsXYZ": origins,
    }


def _run_window_worker(
    root_value: str,
    origin_value: list[int],
    settings: dict[str, Any],
    force_windows: bool,
) -> dict[str, Any]:
    origin = tuple(int(value) for value in origin_value)
    monotone = prototype_monotone_layers(
        root_value,
        force=force_windows,
        settings={
            "windowCellsXYZ": settings["windowCellsXYZ"],
            "minimumFlakeQuality": settings["minimumFlakeQuality"],
            "minimumLinkScore": settings["minimumLinkScore"],
            "gapPenalty": settings["gapPenalty"],
            "edgePaddingVoxels": settings["edgePaddingVoxels"],
        },
        window_origin_cell_xyz=origin,
    )
    association = associate_monotone_branches(
        root_value,
        force=force_windows,
        settings={"edgePaddingVoxels": settings["edgePaddingVoxels"]},
        window_origin_cell_xyz=origin,
    )
    integrity = None
    if bool(settings["runIntegrityAudit"]):
        integrity = association_integrity_audit(
            root_value,
            force=force_windows,
            window_origin_cell_xyz=origin,
        )
    return {
        "originCellXYZ": list(origin),
        "monotone": {
            "elapsedMs": monotone["stats"]["elapsedMs"],
            "cacheHit": monotone["stats"]["cacheHit"],
            "flakeCount": monotone["stats"]["flakeCount"],
            "rawLinkCount": monotone["stats"]["rawMonotoneLinkCount"],
            "retainedLinkCount": monotone["stats"]["retainedMonotoneLinkCount"],
            "branchCount": monotone["stats"]["branchCount"],
            "parityFrustratedLinkCount": monotone["stats"][
                "parityFrustratedLinkCount"
            ],
            "artifact": monotone["artifact"],
        },
        "association": {
            "elapsedMs": association["stats"]["elapsedMs"],
            "cacheHit": association["stats"]["cacheHit"],
            "candidateCount": association["stats"]["candidateBranchPairCount"],
            "stableCandidateCount": association["selected"][
                "preExactStableCandidateCount"
            ],
            "exactPairPassCount": association["selected"]["exactPairPassCount"],
            "exactPairDeferredCount": association["selected"][
                "exactPairDeferredCount"
            ],
            "exactGroupPrunedCount": association["selected"][
                "exactGroupPrunedCount"
            ],
            "retainedMergeCount": association["selected"][
                "finalRetainedMergeCount"
            ],
            "mergedAssociationCount": association["selected"][
                "finalMergedAssociationCount"
            ],
            "exactFailureCount": association["selected"][
                "finalExactFailureCount"
            ],
            "artifact": association["artifact"],
        },
        "integrity": (
            {
                "elapsedMs": integrity["stats"]["elapsedMs"],
                "cacheHit": integrity["stats"]["cacheHit"],
                "carrierCount": integrity["stats"]["associationCarrierCount"],
                "intersectionPairCount": integrity["stats"][
                    "associationPairWithMeshIntersectionCount"
                ],
                "evidenceCoreIntersectionPairCount": integrity["stats"][
                    "associationPairWithEvidenceCoreIntersectionCount"
                ],
                "cellCollisionCount": integrity["stats"][
                    "withinAssociationCellCollisionCount"
                ],
                "artifact": integrity["artifact"],
            }
            if integrity is not None
            else None
        ),
    }


def _neighbor_pairs(windows: list[dict[str, Any]]) -> list[tuple[list[int], list[int]]]:
    output = []
    for source_index, source in enumerate(windows):
        source_low = np.asarray(source["originCellXYZ"], dtype=np.int32)
        source_high = np.asarray(source["stopCellXYZExclusive"], dtype=np.int32)
        for target in windows[source_index + 1 :]:
            target_low = np.asarray(target["originCellXYZ"], dtype=np.int32)
            target_high = np.asarray(target["stopCellXYZExclusive"], dtype=np.int32)
            changed_axes = np.flatnonzero(source_low != target_low)
            if len(changed_axes) != 1:
                continue
            if np.all(
                np.maximum(source_low, target_low)
                < np.minimum(source_high, target_high)
            ):
                output.append((source["originCellXYZ"], target["originCellXYZ"]))
    return output


def _window_paths(root: Path, origin: list[int]) -> tuple[Path, Path]:
    suffix = window_artifact_suffix(origin)
    return (
        root / f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}.npz",
        root
        / f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}.npz",
    )


def _integrity_path(root: Path, origin: list[int]) -> Path:
    suffix = window_artifact_suffix(origin)
    return root / (
        f"branch-association-integrity-v{ASSOCIATION_INTEGRITY_VERSION}{suffix}.npz"
    )


def _integrity_quarantine_mask(
    root: Path,
    origin: list[int],
    monotone: dict[str, np.ndarray],
    association: dict[str, np.ndarray],
    accepted: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    path = _integrity_path(root, origin)
    if not path.is_file():
        return np.zeros(len(accepted), dtype=bool), {
            "violatingAssociationCount": 0,
            "evidenceCoreViolatingAssociationCount": 0,
            "quarantinedJoinCount": 0,
        }
    with np.load(path) as payload:
        integrity = {key: np.asarray(payload[key]) for key in payload.files}
    violating_pair = integrity["intersectingTrianglePairCount"] > 0
    core_pair = integrity["evidenceCoreIntersectingTrianglePairCount"] > 0
    violating_association = np.unique(
        np.r_[
            integrity["associationSource"][violating_pair],
            integrity["associationTarget"][violating_pair],
        ]
    )
    core_association = np.unique(
        np.r_[
            integrity["associationSource"][core_pair],
            integrity["associationTarget"][core_pair],
        ]
    )
    if not len(violating_association):
        quarantine = np.zeros(len(accepted), dtype=bool)
    else:
        branch = monotone["component"].astype(np.int64)
        source_branch = branch[association["candidateNodeSource"].astype(np.int64)]
        target_branch = branch[association["candidateNodeTarget"].astype(np.int64)]
        branch_association = association["branchAssociation"].astype(np.int64)
        source_group = branch_association[source_branch]
        target_group = branch_association[target_branch]
        quarantine = accepted & (
            np.isin(source_group, violating_association)
            | np.isin(target_group, violating_association)
        )
    return quarantine, {
        "violatingAssociationCount": len(violating_association),
        "evidenceCoreViolatingAssociationCount": len(core_association),
        "quarantinedJoinCount": int(np.count_nonzero(quarantine)),
    }


def _node_identity(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        arrays["sourceZIndex"].astype(np.uint64) << np.uint64(32)
    ) | arrays["sourceFlakeId"].astype(np.uint64)


def _canonical_edges(
    source: np.ndarray,
    target: np.ndarray,
    node_identity: np.ndarray,
    selected: np.ndarray | None = None,
) -> np.ndarray:
    if selected is not None:
        source = source[selected]
        target = target[selected]
    first = node_identity[source.astype(np.int64)]
    second = node_identity[target.astype(np.int64)]
    output = np.stack((np.minimum(first, second), np.maximum(first, second)), axis=1)
    return np.unique(output, axis=0) if len(output) else np.empty((0, 2), dtype=np.uint64)


def _canonical_edges_with_parity(
    source: np.ndarray,
    target: np.ndarray,
    node_identity: np.ndarray,
    relative_parity: np.ndarray,
    selected: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if selected is not None:
        source = source[selected]
        target = target[selected]
        relative_parity = relative_parity[selected]
    if not len(source):
        return np.empty((0, 2), dtype=np.uint64), np.empty(0, dtype=np.int8)
    first = node_identity[source.astype(np.int64)]
    second = node_identity[target.astype(np.int64)]
    values = np.stack((np.minimum(first, second), np.maximum(first, second)), axis=1)
    pair, first_index, inverse = np.unique(
        values, axis=0, return_index=True, return_inverse=True
    )
    parity = np.asarray(relative_parity, dtype=np.int8)
    selected_parity = parity[first_index]
    if np.any(selected_parity[inverse] != parity):
        raise ValueError("one canonical edge has conflicting relative parity")
    return pair, selected_parity


def _bit_count(values: np.ndarray) -> np.ndarray:
    lookup = np.asarray([int(value).bit_count() for value in range(256)], dtype=np.uint8)
    byte_view = np.asarray(values, dtype=np.uint64).view(np.uint8).reshape(-1, 8)
    return np.sum(lookup[byte_view], axis=1, dtype=np.uint16)


def _aggregate_pairs(
    pairs_by_window: list[np.ndarray],
    unique_node_identity: np.ndarray,
    node_window_mask: np.ndarray,
    parity_by_window: list[np.ndarray] | None = None,
) -> dict[str, np.ndarray]:
    pending_pairs = []
    pending_masks = []
    pending_parity = []
    for window_index, pairs in enumerate(pairs_by_window):
        if not len(pairs):
            continue
        pending_pairs.append(pairs)
        pending_masks.append(
            np.full(len(pairs), np.uint64(1) << np.uint64(window_index), dtype=np.uint64)
        )
        if parity_by_window is not None:
            pending_parity.append(parity_by_window[window_index])
    if not pending_pairs:
        empty = {
            "pair": np.empty((0, 2), dtype=np.uint64),
            "observationMask": np.empty(0, dtype=np.uint64),
            "acceptanceMask": np.empty(0, dtype=np.uint64),
            "observationCount": np.empty(0, dtype=np.uint8),
            "acceptanceCount": np.empty(0, dtype=np.uint8),
            "unanimous": np.empty(0, dtype=bool),
            "overlapValidated": np.empty(0, dtype=bool),
        }
        if parity_by_window is not None:
            empty["consensusParity"] = np.empty(0, dtype=np.int8)
            empty["parityUnanimous"] = np.empty(0, dtype=bool)
        return empty
    all_pairs = np.concatenate(pending_pairs)
    all_masks = np.concatenate(pending_masks)
    pair, inverse = np.unique(all_pairs, axis=0, return_inverse=True)
    acceptance_mask = np.zeros(len(pair), dtype=np.uint64)
    np.bitwise_or.at(acceptance_mask, inverse, all_masks)
    source_index = np.searchsorted(unique_node_identity, pair[:, 0])
    target_index = np.searchsorted(unique_node_identity, pair[:, 1])
    observation_mask = node_window_mask[source_index] & node_window_mask[target_index]
    observation_count = _bit_count(observation_mask).astype(np.uint8)
    acceptance_count = _bit_count(acceptance_mask).astype(np.uint8)
    unanimous = acceptance_mask == observation_mask
    result = {
        "pair": pair,
        "observationMask": observation_mask,
        "acceptanceMask": acceptance_mask,
        "observationCount": observation_count,
        "acceptanceCount": acceptance_count,
        "unanimous": unanimous,
        "overlapValidated": observation_count >= 2,
    }
    if parity_by_window is not None:
        all_parity = np.concatenate(pending_parity).astype(np.int16)
        parity_sum = np.zeros(len(pair), dtype=np.int16)
        np.add.at(parity_sum, inverse, all_parity)
        parity_unanimous = np.abs(parity_sum) == acceptance_count
        consensus_parity = np.zeros(len(pair), dtype=np.int8)
        consensus_parity[parity_unanimous] = np.sign(
            parity_sum[parity_unanimous]
        ).astype(np.int8)
        result["consensusParity"] = consensus_parity
        result["parityUnanimous"] = parity_unanimous
    return result


def _window_components(
    windows: list[dict[str, Any]],
    reconciliations: list[dict[str, Any]],
) -> np.ndarray:
    window_lookup = {
        tuple(value["originCellXYZ"]): index for index, value in enumerate(windows)
    }
    parent = np.arange(len(windows), dtype=np.int32)
    size = np.ones(len(windows), dtype=np.int32)

    def find(index: int) -> int:
        while int(parent[index]) != index:
            parent[index] = parent[int(parent[index])]
            index = int(parent[index])
        return index

    for value in reconciliations:
        source = window_lookup[tuple(value["sourceWindow"]["originCellXYZ"])]
        target = window_lookup[tuple(value["targetWindow"]["originCellXYZ"])]
        source_root, target_root = find(source), find(target)
        if source_root == target_root:
            continue
        if int(size[source_root]) < int(size[target_root]):
            source_root, target_root = target_root, source_root
        parent[target_root] = source_root
        size[source_root] += size[target_root]
    roots = np.asarray([find(index) for index in range(len(windows))])
    _, component = np.unique(roots, return_inverse=True)
    return component.astype(np.int16)


def _aggregate_window_outputs(
    root: Path,
    windows: list[dict[str, Any]],
    reconciliations: list[dict[str, Any]],
    expected_primary_node_count: int,
    quarantine_integrity_violations: bool,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    if len(windows) > 64:
        raise ValueError("window scheduler supports at most 64 occupied windows")
    window_component = _window_components(windows, reconciliations)
    node_by_window = []
    raw_edges_by_window = []
    raw_parity_by_window = []
    retained_edges_by_window = []
    retained_parity_by_window = []
    joins_by_window = []
    quarantined_joins_by_window = []
    quarantine_stats = []
    for window_index, window in enumerate(windows):
        monotone_path, association_path = _window_paths(root, window["originCellXYZ"])
        with np.load(monotone_path) as payload:
            monotone = {key: np.asarray(payload[key]) for key in payload.files}
        with np.load(association_path) as payload:
            association = {key: np.asarray(payload[key]) for key in payload.files}
        node_identity = _node_identity(monotone)
        node_by_window.append(node_identity)
        raw_pair, raw_parity = _canonical_edges_with_parity(
            monotone["source"],
            monotone["target"],
            node_identity,
            monotone["relativeParity"],
        )
        raw_edges_by_window.append(raw_pair)
        raw_parity_by_window.append(raw_parity)
        retained_pair, retained_parity = _canonical_edges_with_parity(
            monotone["source"],
            monotone["target"],
            node_identity,
            monotone["relativeParity"],
            monotone["retained"],
        )
        retained_edges_by_window.append(retained_pair)
        retained_parity_by_window.append(retained_parity)
        accepted = (association["candidateFinalDecision"] == DECISION_RETAINED) | (
            association["candidateFinalDecision"] == DECISION_REDUNDANT
        )
        if quarantine_integrity_violations:
            quarantine, local_quarantine_stats = _integrity_quarantine_mask(
                root,
                window["originCellXYZ"],
                monotone,
                association,
                accepted,
            )
        else:
            quarantine = np.zeros(len(accepted), dtype=bool)
            local_quarantine_stats = {
                "violatingAssociationCount": 0,
                "evidenceCoreViolatingAssociationCount": 0,
                "quarantinedJoinCount": 0,
            }
        quarantine_stats.append(local_quarantine_stats)
        joins_by_window.append(
            _canonical_edges(
                association["candidateNodeSource"],
                association["candidateNodeTarget"],
                node_identity,
                accepted & ~quarantine,
            )
        )
        quarantined_joins_by_window.append(
            _canonical_edges(
                association["candidateNodeSource"],
                association["candidateNodeTarget"],
                node_identity,
                quarantine,
            )
        )
    all_node_identity = np.concatenate(node_by_window)
    unique_node_identity, inverse = np.unique(all_node_identity, return_inverse=True)
    node_observation_count = np.bincount(inverse).astype(np.uint8)
    node_window_mask = np.zeros(len(unique_node_identity), dtype=np.uint64)
    offset = 0
    for window_index, node_identity in enumerate(node_by_window):
        count = len(node_identity)
        positions = inverse[offset : offset + count]
        np.bitwise_or.at(
            node_window_mask,
            positions,
            np.full(count, np.uint64(1) << np.uint64(window_index), dtype=np.uint64),
        )
        offset += count
    raw = _aggregate_pairs(
        raw_edges_by_window,
        unique_node_identity,
        node_window_mask,
        raw_parity_by_window,
    )
    retained = _aggregate_pairs(
        retained_edges_by_window,
        unique_node_identity,
        node_window_mask,
        retained_parity_by_window,
    )
    joins = _aggregate_pairs(joins_by_window, unique_node_identity, node_window_mask)
    quarantined_joins = _aggregate_pairs(
        quarantined_joins_by_window, unique_node_identity, node_window_mask
    )

    arrays = {
        "windowOriginCellXYZ": np.asarray(
            [value["originCellXYZ"] for value in windows], dtype=np.uint16
        ),
        "windowOverlapComponent": window_component,
        "nodeIdentity": unique_node_identity,
        "nodeObservationCount": node_observation_count,
        "nodeWindowMask": node_window_mask,
        "rawEdgeIdentity": raw["pair"],
        "rawEdgeObservationCount": raw["observationCount"],
        "rawEdgeAcceptanceCount": raw["acceptanceCount"],
        "rawEdgeUnanimous": raw["unanimous"],
        "rawEdgeOverlapValidated": raw["overlapValidated"],
        "rawEdgeRelativeParity": raw["consensusParity"],
        "rawEdgeParityUnanimous": raw["parityUnanimous"],
        "retainedEdgeIdentity": retained["pair"],
        "retainedEdgeObservationCount": retained["observationCount"],
        "retainedEdgeAcceptanceCount": retained["acceptanceCount"],
        "retainedEdgeUnanimous": retained["unanimous"],
        "retainedEdgeOverlapValidated": retained["overlapValidated"],
        "retainedEdgeRelativeParity": retained["consensusParity"],
        "retainedEdgeParityUnanimous": retained["parityUnanimous"],
        "branchJoinEndpointIdentity": joins["pair"],
        "branchJoinObservationCount": joins["observationCount"],
        "branchJoinAcceptanceCount": joins["acceptanceCount"],
        "branchJoinUnanimous": joins["unanimous"],
        "branchJoinOverlapValidated": joins["overlapValidated"],
        "quarantinedBranchJoinEndpointIdentity": quarantined_joins["pair"],
        "quarantinedBranchJoinObservationCount": quarantined_joins[
            "observationCount"
        ],
        "quarantinedBranchJoinWindowCount": quarantined_joins[
            "acceptanceCount"
        ],
    }

    def pair_stats(values: dict[str, np.ndarray]) -> dict[str, int]:
        multi = values["overlapValidated"]
        output = {
            "uniqueAcceptedPairCount": len(values["pair"]),
            "singleWindowAcceptedPairCount": int(np.count_nonzero(~multi)),
            "multiWindowAcceptedPairCount": int(np.count_nonzero(multi)),
            "unanimousPairCount": int(np.count_nonzero(values["unanimous"])),
            "overlapValidatedUnanimousPairCount": int(
                np.count_nonzero(multi & values["unanimous"])
            ),
            "deferredPairCount": int(np.count_nonzero(~values["unanimous"])),
        }
        if "parityUnanimous" in values:
            output["parityDisagreementPairCount"] = int(
                np.count_nonzero(~values["parityUnanimous"])
            )
        return output

    stats = {
        "expectedPrimaryNodeCount": expected_primary_node_count,
        "coveredPrimaryNodeCount": len(unique_node_identity),
        "primaryNodeCoverageFraction": round(
            len(unique_node_identity) / max(expected_primary_node_count, 1), 6
        ),
        "multiWindowNodeCount": int(
            np.count_nonzero(node_observation_count >= 2)
        ),
        "windowOverlapComponentCount": len(np.unique(window_component)),
        "rawMonotoneEdges": pair_stats(raw),
        "retainedMonotoneEdges": pair_stats(retained),
        "branchJoins": pair_stats(joins),
        "integrityQuarantine": {
            "windowCountWithQuarantine": sum(
                value["quarantinedJoinCount"] > 0 for value in quarantine_stats
            ),
            "localAssociationCount": sum(
                value["violatingAssociationCount"] for value in quarantine_stats
            ),
            "localEvidenceCoreAssociationCount": sum(
                value["evidenceCoreViolatingAssociationCount"]
                for value in quarantine_stats
            ),
            "localJoinOccurrenceCount": sum(
                value["quarantinedJoinCount"] for value in quarantine_stats
            ),
            "uniqueJoinEndpointPairCount": len(quarantined_joins["pair"]),
        },
    }
    return arrays, stats


def run_window_schedule(
    output_root: str | Path,
    force: bool = False,
    force_windows: bool = False,
    settings: dict[str, Any] | None = None,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    material_summary = json.loads(
        (root / f"material-intervals-v{MATERIAL_INTERVAL_VERSION}.json").read_text()
    )
    graph_summary = json.loads(
        (root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json").read_text()
    )
    identity = {
        "version": WINDOW_SCHEDULER_VERSION,
        "materialIdentity": material_summary["identity"],
        "sheetletGraphIdentity": graph_summary["identity"],
        "monotoneVersion": MONOTONE_LAYER_VERSION,
        "branchAssociationVersion": BRANCH_ASSOCIATION_VERSION,
        "associationIntegrityVersion": ASSOCIATION_INTEGRITY_VERSION,
        "windowReconciliationVersion": WINDOW_RECONCILIATION_VERSION,
        "settings": resolved,
    }
    stem = f"tiled-window-schedule-v{WINDOW_SCHEDULER_VERSION}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    windows, coverage = _window_catalog(root, resolved)
    if not windows:
        raise ValueError("window schedule contains no occupied windows")
    if len(windows) > 64:
        raise ValueError(
            "window scheduler supports at most 64 occupied windows; increase the stride"
        )
    maximum_workers = max(1, min(int(resolved["maximumWorkers"]), len(windows)))
    window_outputs = []
    with ProcessPoolExecutor(max_workers=maximum_workers) as executor:
        futures = {
            executor.submit(
                _run_window_worker,
                str(root),
                window["originCellXYZ"],
                resolved,
                force_windows,
            ): window
            for window in windows
        }
        for completed, future in enumerate(as_completed(futures), start=1):
            output = future.result()
            window_outputs.append(output)
            if progress is not None:
                progress("window", completed, len(futures), output)
    output_by_origin = {
        tuple(value["originCellXYZ"]): value for value in window_outputs
    }
    for window in windows:
        window.update(output_by_origin[tuple(window["originCellXYZ"])])

    neighbor_pairs = _neighbor_pairs(windows)
    reconciliations = []
    for completed, (source, target) in enumerate(neighbor_pairs, start=1):
        value = reconcile_overlapping_windows(
            root,
            source_origin_cell_xyz=source,
            target_origin_cell_xyz=target,
            force=force_windows,
        )
        reconciliations.append(value)
        if progress is not None:
            progress("reconcile", completed, len(neighbor_pairs), value)

    expected_primary_nodes = int(graph_summary["stats"]["primaryFamilyNodeCount"])
    arrays, aggregate_stats = _aggregate_window_outputs(
        root,
        windows,
        reconciliations,
        expected_primary_nodes,
        bool(resolved["quarantineIntegrityViolations"])
        and bool(resolved["runIntegrityAudit"]),
    )
    _atomic_npz(artifact_path, **arrays)
    classification_counts = {
        name: sum(value["classification"] == name for value in reconciliations)
        for name in sorted({value["classification"] for value in reconciliations})
    }
    integrity_intersections = sum(
        int(value["integrity"]["intersectionPairCount"])
        for value in window_outputs
        if value["integrity"] is not None
    )
    integrity_core_intersections = sum(
        int(value["integrity"]["evidenceCoreIntersectionPairCount"])
        for value in window_outputs
        if value["integrity"] is not None
    )
    result = {
        "identity": identity,
        "contract": {
            "solveScope": (
                "independent overlapping local windows only; this scheduler does not "
                "perform a whole-volume component or dense geometry solve"
            ),
            "commitRule": (
                "a decision observed by multiple windows is overlap-validated only "
                "when every observing window accepts it; disagreements remain deferred"
            ),
            "directions": (
                "normal and fiber directions remain unsigned; only gauge-covariant "
                "relative edge parity is compared across windows"
            ),
            "identityMeaning": (
                "consensus edges and joins remain sparse local surface evidence, not "
                "global sheet or page identities"
            ),
            "integrityQuarantine": (
                "every accepted join belonging to a local association involved in a "
                "support or evidence-core mesh intersection is excluded from consensus "
                "and retained in a separate quarantine catalog"
            ),
        },
        "coverage": coverage,
        "windows": windows,
        "reconciliations": [
            {
                "sourceWindow": value["sourceWindow"],
                "targetWindow": value["targetWindow"],
                "overlap": value["overlap"],
                "classification": value["classification"],
                "stats": value["stats"],
                "artifact": value["artifact"],
            }
            for value in reconciliations
        ],
        "aggregate": aggregate_stats,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "maximumWorkers": maximum_workers,
            "processedWindowCount": len(windows),
            "neighborReconciliationCount": len(reconciliations),
            "reconciliationClassificationCounts": classification_counts,
            "totalWindowFlakeVisits": sum(
                int(value["monotone"]["flakeCount"]) for value in window_outputs
            ),
            "totalRetainedLocalBranchMergeCount": sum(
                int(value["association"]["retainedMergeCount"])
                for value in window_outputs
            ),
            "totalExactPairDeferredCount": sum(
                int(value["association"]["exactPairDeferredCount"])
                for value in window_outputs
            ),
            "totalExactGroupPrunedCount": sum(
                int(value["association"]["exactGroupPrunedCount"])
                for value in window_outputs
            ),
            "totalFinalExactFailureCount": sum(
                int(value["association"]["exactFailureCount"])
                for value in window_outputs
            ),
            "integrityIntersectionPairCount": integrity_intersections,
            "integrityEvidenceCoreIntersectionPairCount": (
                integrity_core_intersections
            ),
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
