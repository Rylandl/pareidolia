from __future__ import annotations

import hashlib
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

from .slab_branch_association import (
    BRANCH_ASSOCIATION_VERSION,
    DECISION_EXACT_PAIR_DEFERRED,
    DECISION_REDUNDANT,
    DECISION_RETAINED,
)
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


GLOBAL_BRANCH_CANDIDATE_VERSION = 1

SOURCE_LOCAL_EXACT_DEFERRED = 1
SOURCE_SUBWINDOW_UNRESOLVED = 2
SOURCE_NAMES = {
    SOURCE_LOCAL_EXACT_DEFERRED: "local-exact-deferred",
    SOURCE_SUBWINDOW_UNRESOLVED: "subwindow-unresolved",
}

SUBWINDOW_REASON_NOT_APPLICABLE = 0
SUBWINDOW_REASON_INSUFFICIENT_OBSERVATIONS = 1
SUBWINDOW_REASON_OBSERVED_DISAGREEMENT = 2


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


def _candidate_evidence_source(
    main_decision: np.ndarray,
    final_decision: np.ndarray,
) -> tuple[int, np.ndarray]:
    main_decision = np.asarray(main_decision, dtype=np.uint8)
    final_decision = np.asarray(final_decision, dtype=np.uint8)
    if len(main_decision) != len(final_decision):
        raise ValueError("local candidate decision arrays must have equal length")
    final_accepted = np.isin(
        final_decision, [DECISION_RETAINED, DECISION_REDUNDANT]
    )
    if np.any(final_accepted):
        return 0, np.zeros(len(final_decision), dtype=bool)
    exact_deferred = final_decision == DECISION_EXACT_PAIR_DEFERRED
    if np.any(exact_deferred):
        return SOURCE_LOCAL_EXACT_DEFERRED, exact_deferred
    main_accepted = np.isin(
        main_decision, [DECISION_RETAINED, DECISION_REDUNDANT]
    )
    if np.any(main_accepted):
        return SOURCE_SUBWINDOW_UNRESOLVED, main_accepted
    return 0, np.zeros(len(final_decision), dtype=bool)


def build_global_branch_candidates(
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    schedule_stem = f"tiled-window-schedule-v{WINDOW_SCHEDULER_VERSION}"
    graph_stem = f"global-monotone-graph-v{GLOBAL_MONOTONE_GRAPH_VERSION}"
    schedule_summary_path = root / f"{schedule_stem}.json"
    schedule_artifact_path = root / f"{schedule_stem}.npz"
    graph_summary_path = root / f"{graph_stem}.json"
    graph_artifact_path = root / f"{graph_stem}.npz"
    schedule = json.loads(schedule_summary_path.read_text())
    graph_summary = json.loads(graph_summary_path.read_text())
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
    identity = {
        "version": GLOBAL_BRANCH_CANDIDATE_VERSION,
        "scheduleIdentity": schedule["identity"],
        "globalGraphIdentity": graph_summary["identity"],
        "inputArtifacts": [
            _content_identity(path)
            for path in (
                schedule_artifact_path,
                graph_artifact_path,
                *local_paths,
            )
        ],
    }
    stem = f"global-branch-candidates-v{GLOBAL_BRANCH_CANDIDATE_VERSION}"
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
        node_identity = payload["nodeIdentity"].astype(np.uint64)
        component = payload["component"].astype(np.uint32)
    with np.load(schedule_artifact_path) as payload:
        scheduled_endpoint_pair = payload[
            "branchJoinEndpointIdentity"
        ].astype(np.uint64)
        quarantined_endpoint_pair = payload[
            "quarantinedBranchJoinEndpointIdentity"
        ].astype(np.uint64)

    observations: dict[tuple[int, int], list[dict[str, Any]]] = defaultdict(list)
    local_candidate_occurrence_count = 0
    for window_index, window in enumerate(schedule["windows"]):
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
            for candidate_index, (source, target) in enumerate(
                zip(
                    payload["candidateNodeSource"],
                    payload["candidateNodeTarget"],
                )
            ):
                pair = tuple(
                    sorted(
                        (
                            int(local_identity[int(source)]),
                            int(local_identity[int(target)]),
                        )
                    )
                )
                observations[pair].append(
                    {
                        "windowIndex": window_index,
                        "score": float(payload["candidateScore"][candidate_index]),
                        "mainDecision": int(
                            payload["candidateMainDecision"][candidate_index]
                        ),
                        "overlapObservationCount": int(
                            payload["candidateOverlapObservationCount"][
                                candidate_index
                            ]
                        ),
                        "overlapAcceptedCount": int(
                            payload["candidateOverlapAcceptedCount"][
                                candidate_index
                            ]
                        ),
                        "overlapDisagreement": bool(
                            payload["candidateOverlapDisagreement"][
                                candidate_index
                            ]
                        ),
                        "stable": bool(payload["candidateStable"][candidate_index]),
                        "exactMedianHeightResidualVoxels": float(
                            payload[
                                "candidateExactPairMedianHeightResidualVoxels"
                            ][candidate_index]
                        ),
                        "exactMedianNormalResidualDeg": float(
                            payload[
                                "candidateExactPairMedianNormalResidualDeg"
                            ][candidate_index]
                        ),
                        "exactPass": bool(
                            payload["candidateExactPairPass"][candidate_index]
                        ),
                        "finalDecision": int(
                            payload["candidateFinalDecision"][candidate_index]
                        ),
                    }
                )
                local_candidate_occurrence_count += 1

    all_endpoint_pair = np.asarray(list(observations), dtype=np.uint64)
    all_endpoint_node = np.searchsorted(node_identity, all_endpoint_pair)
    endpoint_inside = all_endpoint_node < len(node_identity)
    endpoint_match = np.zeros(all_endpoint_pair.shape, dtype=bool)
    endpoint_match[endpoint_inside] = (
        node_identity[all_endpoint_node[endpoint_inside]]
        == all_endpoint_pair[endpoint_inside]
    )
    if not np.all(endpoint_match):
        raise ValueError("one local candidate endpoint is absent from the global graph")
    all_branch_pair = component[all_endpoint_node]

    reserved_endpoint_pair = np.concatenate(
        [scheduled_endpoint_pair, quarantined_endpoint_pair]
    )
    reserved_node = np.searchsorted(node_identity, reserved_endpoint_pair)
    reserved_inside = reserved_node < len(node_identity)
    reserved_match = np.zeros(reserved_endpoint_pair.shape, dtype=bool)
    reserved_match[reserved_inside] = (
        node_identity[reserved_node[reserved_inside]]
        == reserved_endpoint_pair[reserved_inside]
    )
    if not np.all(reserved_match):
        raise ValueError("one reserved local endpoint is absent from the global graph")
    reserved_branch_pair = component[reserved_node]
    reserved_branch_pairs = {
        tuple(sorted((int(pair[0]), int(pair[1]))))
        for pair in reserved_branch_pair
    }
    quarantine = {
        tuple(int(value) for value in pair) for pair in quarantined_endpoint_pair
    }

    candidates: dict[tuple[int, int], dict[str, Any]] = {}
    source_endpoint_candidate_count = defaultdict(int)
    already_linked_endpoint_candidate_count = 0
    reserved_branch_candidate_count = 0
    quarantined_endpoint_candidate_count = 0
    for endpoint, endpoint_node, branch_pair in zip(
        observations, all_endpoint_node, all_branch_pair
    ):
        values = observations[endpoint]
        main_decision = np.asarray(
            [value["mainDecision"] for value in values], dtype=np.uint8
        )
        final_decision = np.asarray(
            [value["finalDecision"] for value in values], dtype=np.uint8
        )
        source, selected = _candidate_evidence_source(
            main_decision, final_decision
        )
        if not source:
            continue
        source_endpoint_candidate_count[source] += 1
        if int(branch_pair[0]) == int(branch_pair[1]):
            already_linked_endpoint_candidate_count += 1
            continue
        canonical_branch_pair = tuple(
            sorted((int(branch_pair[0]), int(branch_pair[1])))
        )
        if endpoint in quarantine:
            quarantined_endpoint_candidate_count += 1
            continue
        if canonical_branch_pair in reserved_branch_pairs:
            reserved_branch_candidate_count += 1
            continue
        selected_values = [value for value, keep in zip(values, selected) if keep]
        scores = np.asarray(
            [value["score"] for value in selected_values], dtype=np.float32
        )
        value = {
            "branchPair": canonical_branch_pair,
            "endpointPair": endpoint,
            "endpointNode": tuple(int(item) for item in endpoint_node),
            "source": source,
            "minimumScore": float(np.min(scores)),
            "meanScore": float(np.mean(scores)),
            "maximumScore": float(np.max(scores)),
            "evidence": selected_values,
        }
        previous = candidates.get(canonical_branch_pair)
        rank = (
            value["minimumScore"],
            value["source"],
            tuple(-item for item in value["endpointPair"]),
        )
        previous_rank = (
            (
                previous["minimumScore"],
                previous["source"],
                tuple(-item for item in previous["endpointPair"]),
            )
            if previous is not None
            else None
        )
        if previous_rank is None or rank > previous_rank:
            candidates[canonical_branch_pair] = value

    values = sorted(
        candidates.values(),
        key=lambda value: (
            -value["minimumScore"],
            -value["source"],
            value["branchPair"],
        ),
    )
    candidate_endpoint_pair = np.asarray(
        [value["endpointPair"] for value in values], dtype=np.uint64
    ).reshape((-1, 2))
    candidate_endpoint_node = np.asarray(
        [value["endpointNode"] for value in values], dtype=np.uint32
    ).reshape((-1, 2))
    candidate_branch_pair = np.asarray(
        [value["branchPair"] for value in values], dtype=np.uint32
    ).reshape((-1, 2))
    candidate_source = np.asarray(
        [value["source"] for value in values], dtype=np.uint8
    )
    candidate_minimum_score = np.asarray(
        [value["minimumScore"] for value in values], dtype=np.float32
    )
    candidate_mean_score = np.asarray(
        [value["meanScore"] for value in values], dtype=np.float32
    )
    candidate_maximum_score = np.asarray(
        [value["maximumScore"] for value in values], dtype=np.float32
    )
    candidate_evidence_count = np.asarray(
        [len(value["evidence"]) for value in values], dtype=np.uint8
    )
    candidate_subwindow_reason = np.asarray(
        [
            (
                SUBWINDOW_REASON_OBSERVED_DISAGREEMENT
                if any(
                    bool(evidence["overlapDisagreement"])
                    for evidence in value["evidence"]
                )
                else SUBWINDOW_REASON_INSUFFICIENT_OBSERVATIONS
            )
            if int(value["source"]) == SOURCE_SUBWINDOW_UNRESOLVED
            else SUBWINDOW_REASON_NOT_APPLICABLE
            for value in values
        ],
        dtype=np.uint8,
    )

    evidence_candidate_index = []
    evidence_window_index = []
    evidence_score = []
    evidence_overlap_observation_count = []
    evidence_overlap_accepted_count = []
    evidence_overlap_disagreement = []
    evidence_stable = []
    evidence_exact_height = []
    evidence_exact_normal = []
    evidence_exact_pass = []
    for candidate_index, value in enumerate(values):
        for evidence in value["evidence"]:
            evidence_candidate_index.append(candidate_index)
            evidence_window_index.append(evidence["windowIndex"])
            evidence_score.append(evidence["score"])
            evidence_overlap_observation_count.append(
                evidence["overlapObservationCount"]
            )
            evidence_overlap_accepted_count.append(
                evidence["overlapAcceptedCount"]
            )
            evidence_overlap_disagreement.append(
                evidence["overlapDisagreement"]
            )
            evidence_stable.append(evidence["stable"])
            evidence_exact_height.append(
                evidence["exactMedianHeightResidualVoxels"]
            )
            evidence_exact_normal.append(
                evidence["exactMedianNormalResidualDeg"]
            )
            evidence_exact_pass.append(evidence["exactPass"])

    _atomic_npz(
        artifact_path,
        candidateEndpointIdentity=candidate_endpoint_pair,
        candidateEndpointNodeIndex=candidate_endpoint_node,
        candidateBranchSource=candidate_branch_pair[:, 0],
        candidateBranchTarget=candidate_branch_pair[:, 1],
        candidateSource=candidate_source,
        candidateMinimumScore=candidate_minimum_score,
        candidateMeanScore=candidate_mean_score,
        candidateMaximumScore=candidate_maximum_score,
        candidateEvidenceCount=candidate_evidence_count,
        candidateSubwindowReason=candidate_subwindow_reason,
        evidenceCandidateIndex=np.asarray(
            evidence_candidate_index, dtype=np.uint32
        ),
        evidenceWindowIndex=np.asarray(evidence_window_index, dtype=np.uint8),
        evidenceScore=np.asarray(evidence_score, dtype=np.float32),
        evidenceOverlapObservationCount=np.asarray(
            evidence_overlap_observation_count, dtype=np.uint8
        ),
        evidenceOverlapAcceptedCount=np.asarray(
            evidence_overlap_accepted_count, dtype=np.uint8
        ),
        evidenceOverlapDisagreement=np.asarray(
            evidence_overlap_disagreement, dtype=bool
        ),
        evidenceStable=np.asarray(evidence_stable, dtype=bool),
        evidenceExactMedianHeightResidualVoxels=np.asarray(
            evidence_exact_height, dtype=np.float32
        ),
        evidenceExactMedianNormalResidualDeg=np.asarray(
            evidence_exact_normal, dtype=np.float32
        ),
        evidenceExactPass=np.asarray(evidence_exact_pass, dtype=bool),
    )
    artifact = _content_identity(artifact_path)
    result = {
        "identity": identity,
        "contract": {
            "scope": (
                "whole-volume discovery of branch pairs withheld after passing "
                "local score, material, order, and collision checks"
            ),
            "subwindowUnresolvedMeaning": (
                "the main window accepted the branch pair, but overlapping "
                "subwindows did not provide unanimous construction support"
            ),
            "localExactDeferredMeaning": (
                "the pair passed local overlap stability but failed the local MLS "
                "carrier gate; it remains evidence only until complete-global-branch "
                "reconstruction passes"
            ),
            "exclusions": (
                "locally accepted, integrity-quarantined, already-linked, and "
                "already-reserved global branch pairs do not re-enter this catalog"
            ),
            "identityMeaning": (
                "catalog entries are candidate joins, not accepted associations or "
                "physical sheets"
            ),
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "localCandidateOccurrenceCount": local_candidate_occurrence_count,
            "uniqueLocalEndpointPairCount": len(observations),
            "sourceEndpointCandidateCounts": {
                SOURCE_NAMES[source]: int(source_endpoint_candidate_count[source])
                for source in SOURCE_NAMES
            },
            "alreadyLinkedEndpointCandidateCount": (
                already_linked_endpoint_candidate_count
            ),
            "quarantinedEndpointCandidateCount": (
                quarantined_endpoint_candidate_count
            ),
            "reservedBranchCandidateCount": reserved_branch_candidate_count,
            "candidateBranchPairCount": len(values),
            "candidateBranchPairCountsBySource": {
                SOURCE_NAMES[source]: int(np.count_nonzero(candidate_source == source))
                for source in SOURCE_NAMES
            },
            "subwindowUnresolvedReasonCounts": {
                "insufficient-observations": int(
                    np.count_nonzero(
                        candidate_subwindow_reason
                        == SUBWINDOW_REASON_INSUFFICIENT_OBSERVATIONS
                    )
                ),
                "observed-disagreement": int(
                    np.count_nonzero(
                        candidate_subwindow_reason
                        == SUBWINDOW_REASON_OBSERVED_DISAGREEMENT
                    )
                ),
            },
            "candidateEvidenceOccurrenceCount": len(evidence_candidate_index),
            "candidateMinimumScore": _quantiles(candidate_minimum_score),
        },
        "topCandidates": [
            {
                "candidateIndex": index,
                "source": SOURCE_NAMES[int(value["source"])],
                "branchPair": list(value["branchPair"]),
                "endpointIdentity": list(value["endpointPair"]),
                "minimumScore": round(float(value["minimumScore"]), 6),
                "evidenceCount": len(value["evidence"]),
            }
            for index, value in enumerate(values[:20])
        ],
        "artifact": artifact,
    }
    _atomic_json(summary_path, result)
    return result
