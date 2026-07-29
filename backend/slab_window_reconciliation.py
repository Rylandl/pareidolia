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
    DECISION_REDUNDANT,
    DECISION_RETAINED,
)
from .slab_monotone_layers import MONOTONE_LAYER_VERSION, window_artifact_suffix


WINDOW_RECONCILIATION_VERSION = 1


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


def _origin_label(origin: tuple[int, int, int] | list[int] | None) -> str:
    suffix = window_artifact_suffix(origin)
    return suffix[1:] if suffix else "densest"


def _window_paths(
    root: Path, origin: tuple[int, int, int] | list[int] | None
) -> dict[str, Path]:
    suffix = window_artifact_suffix(origin)
    monotone_stem = f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}{suffix}"
    association_stem = (
        f"branch-association-window-v{BRANCH_ASSOCIATION_VERSION}{suffix}"
    )
    return {
        "monotoneSummary": root / f"{monotone_stem}.json",
        "monotoneArtifact": root / f"{monotone_stem}.npz",
        "associationSummary": root / f"{association_stem}.json",
        "associationArtifact": root / f"{association_stem}.npz",
    }


def _node_identity(arrays: dict[str, np.ndarray]) -> np.ndarray:
    return (
        arrays["sourceZIndex"].astype(np.uint64) << np.uint64(32)
    ) | arrays["sourceFlakeId"].astype(np.uint64)


def _load_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as payload:
        return {key: np.asarray(payload[key]) for key in payload.files}


def _inside(cell: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    return np.all((cell >= low) & (cell < high), axis=1)


def _edge_set(
    arrays: dict[str, np.ndarray],
    node_identity: np.ndarray,
    shared_identity: set[int],
    retained_only: bool = True,
) -> set[tuple[int, int]]:
    output = set()
    selected = (
        arrays["retained"]
        if retained_only
        else np.ones(len(arrays["source"]), dtype=bool)
    )
    for source, target in zip(
        arrays["source"][selected],
        arrays["target"][selected],
    ):
        first = int(node_identity[int(source)])
        second = int(node_identity[int(target)])
        if first not in shared_identity or second not in shared_identity:
            continue
        output.add((min(first, second), max(first, second)))
    return output


def _accepted_endpoint_set(
    arrays: dict[str, np.ndarray],
    node_identity: np.ndarray,
    shared_identity: set[int],
) -> set[tuple[int, int]]:
    accepted = (arrays["candidateFinalDecision"] == DECISION_RETAINED) | (
        arrays["candidateFinalDecision"] == DECISION_REDUNDANT
    )
    output = set()
    for source, target in zip(
        arrays["candidateNodeSource"][accepted],
        arrays["candidateNodeTarget"][accepted],
    ):
        first = int(node_identity[int(source)])
        second = int(node_identity[int(target)])
        if first not in shared_identity or second not in shared_identity:
            continue
        output.add((min(first, second), max(first, second)))
    return output


def _choose_two(counts: np.ndarray) -> int:
    values = np.asarray(counts, dtype=np.int64)
    return int(np.sum(values * (values - 1) // 2))


def _partition_overlap_stats(
    source_label: np.ndarray,
    target_label: np.ndarray,
) -> dict[str, Any]:
    if len(source_label) != len(target_label):
        raise ValueError("partition label arrays must be aligned")
    if not len(source_label):
        return {
            "sharedNodeCount": 0,
            "sourceCoassignedPairCount": 0,
            "targetCoassignedPairCount": 0,
            "jointCoassignedPairCount": 0,
            "coassignmentUnionPairCount": 0,
            "coassignmentDisagreementPairCount": 0,
            "coassignmentAgreementFraction": None,
            "sourceSplitGroupCount": 0,
            "targetSplitGroupCount": 0,
            "nodeCountInContextSplit": 0,
        }
    _, source_counts = np.unique(source_label, return_counts=True)
    _, target_counts = np.unique(target_label, return_counts=True)
    pair_label = np.stack((source_label, target_label), axis=1)
    unique_pairs, pair_counts = np.unique(pair_label, axis=0, return_counts=True)
    source_pairs = _choose_two(source_counts)
    target_pairs = _choose_two(target_counts)
    joint_pairs = _choose_two(pair_counts)
    union_pairs = source_pairs + target_pairs - joint_pairs
    disagreement_pairs = source_pairs + target_pairs - 2 * joint_pairs

    targets_by_source: dict[int, set[int]] = defaultdict(set)
    sources_by_target: dict[int, set[int]] = defaultdict(set)
    for source, target in unique_pairs:
        targets_by_source[int(source)].add(int(target))
        sources_by_target[int(target)].add(int(source))
    split_source = {
        source for source, targets in targets_by_source.items() if len(targets) > 1
    }
    split_target = {
        target for target, sources in sources_by_target.items() if len(sources) > 1
    }
    context_split = np.isin(source_label, list(split_source)) | np.isin(
        target_label, list(split_target)
    )
    return {
        "sharedNodeCount": len(source_label),
        "sourceCoassignedPairCount": source_pairs,
        "targetCoassignedPairCount": target_pairs,
        "jointCoassignedPairCount": joint_pairs,
        "coassignmentUnionPairCount": union_pairs,
        "coassignmentDisagreementPairCount": disagreement_pairs,
        "coassignmentAgreementFraction": (
            round(joint_pairs / union_pairs, 6) if union_pairs else 1.0
        ),
        "sourceSplitGroupCount": len(split_source),
        "targetSplitGroupCount": len(split_target),
        "nodeCountInContextSplit": int(np.count_nonzero(context_split)),
    }


def _edge_array(values: set[tuple[int, int]]) -> np.ndarray:
    return np.asarray(sorted(values), dtype=np.uint64).reshape(-1, 2)


def _cell_regions(cells: np.ndarray) -> list[dict[str, Any]]:
    pending = {tuple(int(value) for value in cell) for cell in cells}
    components = []
    offsets = (
        (1, 0, 0),
        (-1, 0, 0),
        (0, 1, 0),
        (0, -1, 0),
        (0, 0, 1),
        (0, 0, -1),
    )
    while pending:
        root = pending.pop()
        component = {root}
        queue = [root]
        while queue:
            current = queue.pop()
            for offset in offsets:
                neighbor = tuple(
                    current[axis] + offset[axis] for axis in range(3)
                )
                if neighbor not in pending:
                    continue
                pending.remove(neighbor)
                component.add(neighbor)
                queue.append(neighbor)
        values = np.asarray(sorted(component), dtype=np.int32)
        components.append(
            {
                "cellCount": len(component),
                "originCellXYZ": np.min(values, axis=0).astype(int).tolist(),
                "stopCellXYZExclusive": (
                    np.max(values, axis=0) + 1
                ).astype(int).tolist(),
            }
        )
    return sorted(components, key=lambda value: value["cellCount"], reverse=True)


def reconcile_overlapping_windows(
    output_root: str | Path,
    target_origin_cell_xyz: tuple[int, int, int] | list[int],
    source_origin_cell_xyz: tuple[int, int, int] | list[int] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    source_paths = _window_paths(root, source_origin_cell_xyz)
    target_paths = _window_paths(root, target_origin_cell_xyz)
    input_paths = [*source_paths.values(), *target_paths.values()]
    source_summary = json.loads(source_paths["monotoneSummary"].read_text())
    target_summary = json.loads(target_paths["monotoneSummary"].read_text())
    source_association_summary = json.loads(
        source_paths["associationSummary"].read_text()
    )
    target_association_summary = json.loads(
        target_paths["associationSummary"].read_text()
    )
    identity = {
        "version": WINDOW_RECONCILIATION_VERSION,
        "sourceMonotoneIdentity": source_summary["identity"],
        "targetMonotoneIdentity": target_summary["identity"],
        "sourceAssociationIdentity": source_association_summary["identity"],
        "targetAssociationIdentity": target_association_summary["identity"],
        "inputArtifacts": [_content_identity(path) for path in input_paths],
    }
    source_label = _origin_label(source_origin_cell_xyz)
    target_label = _origin_label(target_origin_cell_xyz)
    stem = (
        f"window-reconciliation-v{WINDOW_RECONCILIATION_VERSION}"
        f"-{source_label}-to-{target_label}"
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
    source_monotone = _load_arrays(source_paths["monotoneArtifact"])
    target_monotone = _load_arrays(target_paths["monotoneArtifact"])
    source_association = _load_arrays(source_paths["associationArtifact"])
    target_association = _load_arrays(target_paths["associationArtifact"])
    source_identity = _node_identity(source_monotone)
    target_identity = _node_identity(target_monotone)
    shared, source_index, target_index = np.intersect1d(
        source_identity,
        target_identity,
        assume_unique=True,
        return_indices=True,
    )
    source_window = source_summary["window"]
    target_window = target_summary["window"]
    low = np.maximum(
        source_window["originCellXYZ"], target_window["originCellXYZ"]
    ).astype(np.int32)
    high = np.minimum(
        source_window["stopCellXYZExclusive"],
        target_window["stopCellXYZExclusive"],
    ).astype(np.int32)
    if np.any(low >= high):
        raise ValueError("windows do not have a cell overlap")
    source_inside = _inside(source_monotone["cellIndex"], low, high)
    target_inside = _inside(target_monotone["cellIndex"], low, high)
    source_overlap_identity = set(int(value) for value in source_identity[source_inside])
    target_overlap_identity = set(int(value) for value in target_identity[target_inside])
    shared_identity = set(int(value) for value in shared)

    sign_product = (
        source_monotone["normalSign"][source_index].astype(np.int8)
        * target_monotone["normalSign"][target_index].astype(np.int8)
    )
    positive = int(np.count_nonzero(sign_product > 0))
    negative = int(np.count_nonzero(sign_product < 0))
    relative_sign = 1 if positive >= negative else -1
    sign_disagreement = sign_product != relative_sign
    sign_disagreement_cells = source_monotone["cellIndex"][
        source_index[sign_disagreement]
    ]
    sign_regions = _cell_regions(sign_disagreement_cells)
    aligned_depth_difference = np.abs(
        source_monotone["orientedDepth"][source_index]
        - relative_sign * target_monotone["orientedDepth"][target_index]
    )

    source_raw_edges = _edge_set(
        source_monotone, source_identity, shared_identity, retained_only=False
    )
    target_raw_edges = _edge_set(
        target_monotone, target_identity, shared_identity, retained_only=False
    )
    shared_raw_edges = source_raw_edges & target_raw_edges
    source_only_raw_edges = source_raw_edges - target_raw_edges
    target_only_raw_edges = target_raw_edges - source_raw_edges
    source_edges = _edge_set(source_monotone, source_identity, shared_identity)
    target_edges = _edge_set(target_monotone, target_identity, shared_identity)
    shared_edges = source_edges & target_edges
    source_only_edges = source_edges - target_edges
    target_only_edges = target_edges - source_edges

    source_endpoints = _accepted_endpoint_set(
        source_association, source_identity, shared_identity
    )
    target_endpoints = _accepted_endpoint_set(
        target_association, target_identity, shared_identity
    )
    shared_endpoints = source_endpoints & target_endpoints
    source_only_endpoints = source_endpoints - target_endpoints
    target_only_endpoints = target_endpoints - source_endpoints

    monotone_partition = _partition_overlap_stats(
        source_monotone["component"][source_index],
        target_monotone["component"][target_index],
    )
    association_partition = _partition_overlap_stats(
        source_association["flakeAssociation"][source_index],
        target_association["flakeAssociation"][target_index],
    )
    _atomic_npz(
        artifact_path,
        sharedNodeIdentity=shared.astype(np.uint64),
        sourceNodeIndex=source_index.astype(np.uint32),
        targetNodeIndex=target_index.astype(np.uint32),
        signDisagreement=sign_disagreement,
        alignedDepthDifferenceVoxels=aligned_depth_difference.astype(np.float32),
        sourceOnlyRawMonotoneEdgeIdentity=_edge_array(source_only_raw_edges),
        targetOnlyRawMonotoneEdgeIdentity=_edge_array(target_only_raw_edges),
        sourceOnlyMonotoneEdgeIdentity=_edge_array(source_only_edges),
        targetOnlyMonotoneEdgeIdentity=_edge_array(target_only_edges),
        sourceOnlyAcceptedEndpointIdentity=_edge_array(source_only_endpoints),
        targetOnlyAcceptedEndpointIdentity=_edge_array(target_only_endpoints),
    )
    raw_monotone_disagreement = len(source_only_raw_edges) + len(
        target_only_raw_edges
    )
    monotone_disagreement = len(source_only_edges) + len(target_only_edges)
    association_disagreement = len(source_only_endpoints) + len(target_only_endpoints)
    if np.any(sign_disagreement):
        classification = "deferred-normal-sign-overlap"
    elif raw_monotone_disagreement:
        classification = "deferred-monotone-match-overlap"
    elif monotone_disagreement:
        classification = "deferred-collision-context-overlap"
    elif association_disagreement:
        classification = "deferred-association-overlap"
    elif association_partition["coassignmentDisagreementPairCount"]:
        classification = "context-dependent-partition"
    else:
        classification = "committed-overlap-agreement"
    result = {
        "identity": identity,
        "contract": {
            "scope": (
                "stable flake identities and decisions wholly inside the intersection "
                "of two independently solved local windows"
            ),
            "signMeaning": (
                "normal signs are aligned by one relative binary flip for comparison; "
                "the aligned sign still has no physical side meaning"
            ),
            "commitRule": (
                "only overlap decisions present in both windows are agreement evidence; "
                "one-window-only monotone edges or branch joins remain explicitly deferred"
            ),
            "rawVsRetained": (
                "raw order-preserving matches are compared separately from the later "
                "collision-safe graph pruning so context dependence is attributed to "
                "the correct construction stage"
            ),
            "partitionMeaning": (
                "partition comparison is contextual diagnostics because paths outside "
                "the overlap can legitimately change local component labels"
            ),
            "deferredSignMeaning": (
                "cells whose relative sign differs after the best whole-overlap "
                "binary alignment are localized ambiguity regions, not physical "
                "normal-side claims"
            ),
        },
        "sourceWindow": source_window,
        "targetWindow": target_window,
        "overlap": {
            "originCellXYZ": low.astype(int).tolist(),
            "stopCellXYZExclusive": high.astype(int).tolist(),
            "shapeCellsXYZ": (high - low).astype(int).tolist(),
        },
        "classification": classification,
        "deferredNormalSignRegions": sign_regions[:24],
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "sourceOverlapNodeCount": len(source_overlap_identity),
            "targetOverlapNodeCount": len(target_overlap_identity),
            "sharedNodeCount": len(shared),
            "sourceOnlyOverlapNodeCount": len(
                source_overlap_identity - target_overlap_identity
            ),
            "targetOnlyOverlapNodeCount": len(
                target_overlap_identity - source_overlap_identity
            ),
            "relativeNormalSign": relative_sign,
            "normalSignDisagreementCount": int(np.count_nonzero(sign_disagreement)),
            "normalSignDisagreementCellCount": len(
                {tuple(int(item) for item in value) for value in sign_disagreement_cells}
            ),
            "normalSignDisagreementRegionCount": len(sign_regions),
            "largestNormalSignDisagreementRegionCellCount": (
                int(sign_regions[0]["cellCount"]) if sign_regions else 0
            ),
            "normalSignDisagreementFraction": round(
                float(np.mean(sign_disagreement)) if len(shared) else 0.0, 6
            ),
            "maximumAlignedDepthDifferenceVoxels": round(
                float(np.max(aligned_depth_difference, initial=0.0)), 6
            ),
            "sourceRawMonotoneEdgeCount": len(source_raw_edges),
            "targetRawMonotoneEdgeCount": len(target_raw_edges),
            "sharedRawMonotoneEdgeCount": len(shared_raw_edges),
            "sourceOnlyRawMonotoneEdgeCount": len(source_only_raw_edges),
            "targetOnlyRawMonotoneEdgeCount": len(target_only_raw_edges),
            "rawMonotoneEdgeAgreementFraction": round(
                len(shared_raw_edges)
                / max(len(source_raw_edges | target_raw_edges), 1),
                6,
            ),
            "sourceMonotoneEdgeCount": len(source_edges),
            "targetMonotoneEdgeCount": len(target_edges),
            "sharedMonotoneEdgeCount": len(shared_edges),
            "sourceOnlyMonotoneEdgeCount": len(source_only_edges),
            "targetOnlyMonotoneEdgeCount": len(target_only_edges),
            "monotoneEdgeAgreementFraction": round(
                len(shared_edges) / max(len(source_edges | target_edges), 1), 6
            ),
            "sourceAcceptedEndpointCount": len(source_endpoints),
            "targetAcceptedEndpointCount": len(target_endpoints),
            "sharedAcceptedEndpointCount": len(shared_endpoints),
            "sourceOnlyAcceptedEndpointCount": len(source_only_endpoints),
            "targetOnlyAcceptedEndpointCount": len(target_only_endpoints),
            "acceptedEndpointAgreementFraction": round(
                len(shared_endpoints)
                / max(len(source_endpoints | target_endpoints), 1),
                6,
            ),
            "monotonePartition": monotone_partition,
            "associationPartition": association_partition,
            "deferredRawMonotoneEdgeCount": raw_monotone_disagreement,
            "deferredMonotoneEdgeCount": monotone_disagreement,
            "deferredAssociationEndpointCount": association_disagreement,
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
