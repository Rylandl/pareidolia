from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .slab_carrier_growth import _flake_arrays, _score_growth_candidates
from .slab_flakes import FLAKE_CACHE_VERSION, _flake_pair_metrics
from .slab_fragment_termination_census import (
    FRAGMENT_TERMINATION_CENSUS_VERSION,
)
from .slab_gap_census import _content_identity
from .slab_gap_reanalysis import (
    _aligned_target_bounds,
    _extract_target_needles,
    _fit_sample_modes,
)
from .slab_global_branch_association import (
    GLOBAL_BRANCH_ASSOCIATION_VERSION,
    _load_selected_flakes,
)
from .slab_global_monotone_graph import GLOBAL_MONOTONE_GRAPH_VERSION
from .slab_normal_families import _catalog_records_for_cell


TERMINATION_REANALYSIS_VERSION = 1

DEFAULT_SETTINGS: dict[str, Any] = {
    "maximumTargets": 128,
    "targetDistanceVoxels": 32.0,
    "candidateSpacingVoxels": 2,
    "maximumNeedlesPerBin": 256,
    "maximumNeedlesPerCell": 1024,
    "maximumModesPerTarget": 12,
    "maximumCropVoxels": 4_000_000,
    "competitorRadiusCells": 4,
    "minimumCompetitorAssociationNodeCount": 25,
    "scoreThreshold": 0.55,
    "minimumModeMargin": 0.04,
    "minimumOwnershipMargin": 0.04,
    "maximumExistingCellPositionResidualVoxels": 6.0,
    "maximumExistingCellNormalResidualDeg": 12.0,
    "maximumExistingCellFiberResidualDeg": 12.0,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _file_stat_identity(path: Path) -> dict[str, Any]:
    stat = path.stat()
    return {
        "path": path.name,
        "bytes": stat.st_size,
        "mtimeNs": stat.st_mtime_ns,
    }


def _quantiles(values: list[float], digits: int = 4) -> dict[str, float | None]:
    array = np.asarray(values, dtype=np.float64)
    array = array[np.isfinite(array)]
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    if not len(array):
        return {name: None for name in names}
    return {
        name: round(float(value), digits)
        for name, value in zip(
            names, np.percentile(array, (0, 10, 25, 50, 75, 90, 100))
        )
    }


def _nearest_grid_cell(
    point_xyz: np.ndarray,
    grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
) -> tuple[int, int, int]:
    point = np.asarray(point_xyz, dtype=np.float64)
    return tuple(
        int(np.argmin(np.abs(values - point[axis])))
        for axis, values in enumerate(grid_xyz)
    )


def _resolve_open_targets(
    queue: list[dict[str, Any]],
    endpoint_cluster: np.ndarray,
    endpoint_node: np.ndarray,
    endpoint_identity: np.ndarray,
    endpoint_center: np.ndarray,
    endpoint_outward: np.ndarray,
    endpoint_quality: np.ndarray,
    endpoint_evidence_score: np.ndarray,
    node_association: np.ndarray,
    node_cell: np.ndarray,
    grid_xyz: tuple[np.ndarray, np.ndarray, np.ndarray],
    target_distance: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    association_cells = {
        int(association_id): {
            tuple(int(value) for value in cell)
            for cell in node_cell[node_association == int(association_id)]
        }
        for association_id in sorted(
            {int(record["associationId"]) for record in queue}
        )
    }
    resolved: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    seen: set[tuple[int, int, int, int]] = set()
    for queue_rank, record in enumerate(queue, start=1):
        cluster_index = int(record["clusterIndex"])
        association_id = int(record["associationId"])
        members = np.flatnonzero(endpoint_cluster == cluster_index)
        choices = []
        for endpoint_index in members:
            point = (
                endpoint_center[endpoint_index]
                + float(target_distance) * endpoint_outward[endpoint_index]
            )
            cell = _nearest_grid_cell(point, grid_xyz)
            if cell in association_cells[association_id]:
                continue
            evidence_score = float(endpoint_evidence_score[endpoint_index])
            choices.append(
                (
                    float(endpoint_quality[endpoint_index]),
                    evidence_score if np.isfinite(evidence_score) else -1.0,
                    -int(endpoint_index),
                    int(endpoint_index),
                    cell,
                    point,
                )
            )
        if not choices:
            skipped.append(
                {
                    "queueRank": queue_rank,
                    "clusterIndex": cluster_index,
                    "associationId": association_id,
                    "reason": "association-covered-next-cell",
                    "clusterEndpointCount": int(len(members)),
                }
            )
            continue
        selected = max(choices)
        endpoint_index = int(selected[3])
        cell = tuple(int(value) for value in selected[4])
        key = (association_id, *cell)
        if key in seen:
            skipped.append(
                {
                    "queueRank": queue_rank,
                    "clusterIndex": cluster_index,
                    "associationId": association_id,
                    "reason": "duplicate-association-target-cell",
                    "targetCellIndex": list(cell),
                }
            )
            continue
        seen.add(key)
        resolved.append(
            {
                "queueRank": queue_rank,
                "clusterIndex": cluster_index,
                "associationId": association_id,
                "associationNodeCount": int(record["associationNodeCount"]),
                "clusterEndpointCount": int(record["endpointCount"]),
                "openEndpointChoiceCount": len(choices),
                "sourceEndpointIndex": endpoint_index,
                "sourceEndpointNodeIndex": int(endpoint_node[endpoint_index]),
                "sourceEndpointIdentity": int(endpoint_identity[endpoint_index]),
                "sourceEndpointXYZ": np.asarray(
                    endpoint_center[endpoint_index], dtype=np.float32
                ),
                "outwardXYZ": np.asarray(
                    endpoint_outward[endpoint_index], dtype=np.float32
                ),
                "targetXYZ": np.asarray(selected[5], dtype=np.float32),
                "targetCellIndex": cell,
                "denseAcusPriority": float(record["denseAcusPriority"]),
            }
        )
    return resolved, skipped


def _coalesce_target_crops(
    target_xyz: np.ndarray,
    cube_size: int,
    halo: int,
    source_shape_xyz: tuple[int, int, int],
    maximum_crop_voxels: int,
    alignment: int = 1,
) -> list[dict[str, Any]]:
    points = np.asarray(target_xyz, dtype=np.float32)
    if not len(points):
        return []
    source_shape = np.asarray(source_shape_xyz, dtype=np.int32)
    bounds = [
        _aligned_target_bounds(
            point[None, :], cube_size, halo, source_shape, alignment
        )
        for point in points
    ]
    lows = np.stack([value[0] for value in bounds])
    highs = np.stack([value[1] for value in bounds])
    order = np.lexsort((lows[:, 2], lows[:, 1], lows[:, 0]))
    crops: list[dict[str, Any]] = []
    for target_index in order:
        low = lows[target_index]
        high = highs[target_index]
        candidates = []
        for crop_index, crop in enumerate(crops):
            crop_low = crop["lowXYZ"]
            crop_high = crop["highXYZ"]
            overlaps = bool(np.all(low <= crop_high) and np.all(high >= crop_low))
            if not overlaps:
                continue
            union_low = np.minimum(low, crop_low)
            union_high = np.maximum(high, crop_high)
            union_voxels = int(np.prod(union_high - union_low, dtype=np.int64))
            if union_voxels > int(maximum_crop_voxels):
                continue
            current_voxels = int(
                np.prod(crop_high - crop_low, dtype=np.int64)
            )
            candidates.append(
                (union_voxels - current_voxels, crop_index, union_low, union_high)
            )
        if not candidates:
            crops.append(
                {
                    "targetIndices": [int(target_index)],
                    "lowXYZ": low.copy(),
                    "highXYZ": high.copy(),
                }
            )
            continue
        _, crop_index, union_low, union_high = min(
            candidates, key=lambda value: (value[0], value[1])
        )
        crops[crop_index]["targetIndices"].append(int(target_index))
        crops[crop_index]["lowXYZ"] = union_low
        crops[crop_index]["highXYZ"] = union_high
    for crop_index, crop in enumerate(crops):
        crop["cropIndex"] = crop_index
        crop["targetIndices"].sort()
        crop["voxelCount"] = int(
            np.prod(crop["highXYZ"] - crop["lowXYZ"], dtype=np.int64)
        )
    return crops


def _public_mode(mode: dict[str, Any]) -> dict[str, Any]:
    return {
        key: value
        for key, value in mode.items()
        if not key.startswith("_")
    }


def _score_modes(
    modes: list[dict[str, Any]],
    association_id: int,
    association_nodes: dict[int, np.ndarray],
    flakes_by_node: dict[int, dict[str, Any]],
    score_threshold: float,
    minimum_mode_margin: float,
    minimum_ownership_margin: float,
) -> dict[str, Any]:
    if not modes:
        return {
            "modeCount": 0,
            "passed": False,
            "failureReasons": ["no-mode"],
            "bestMode": None,
            "bestAssociationScore": 0.0,
            "modeMargin": 0.0,
            "ownershipMargin": 0.0,
            "bestOwnerAssociationId": None,
            "associationScoreCount": len(association_nodes),
        }
    score_by_association: dict[int, dict[str, np.ndarray]] = {}
    for candidate_association, nodes in association_nodes.items():
        member_flakes = [flakes_by_node[int(node)] for node in nodes]
        combined = member_flakes + modes
        arrays = _flake_arrays(combined)
        score_by_association[candidate_association] = _score_growth_candidates(
            np.arange(len(member_flakes), dtype=np.int64),
            np.arange(len(member_flakes), len(combined), dtype=np.int64),
            arrays,
            combined,
        )
    own = score_by_association[association_id]
    order = np.argsort(own["score"])[::-1]
    best_mode_index = int(order[0])
    best_score = float(own["score"][best_mode_index])
    second_score = float(own["score"][order[1]]) if len(order) > 1 else 0.0
    mode_margin = best_score - second_score
    owner_scores = {
        int(candidate_association): float(values["score"][best_mode_index])
        for candidate_association, values in score_by_association.items()
    }
    owner_order = sorted(owner_scores, key=owner_scores.get, reverse=True)
    best_owner = int(owner_order[0])
    other_score = max(
        (
            score
            for candidate_association, score in owner_scores.items()
            if candidate_association != association_id
        ),
        default=0.0,
    )
    ownership_margin = best_score - other_score
    reasons = []
    if best_score < float(score_threshold):
        reasons.append("score")
    if mode_margin < float(minimum_mode_margin):
        reasons.append("mode-margin")
    if (
        best_owner != association_id
        or ownership_margin < float(minimum_ownership_margin)
    ):
        reasons.append("ownership")
    best_mode = {
        **_public_mode(modes[best_mode_index]),
        "associationScore": round(best_score, 6),
        "heightResidualVoxels": round(
            float(own["heightResidual"][best_mode_index]), 4
        ),
        "normalResidualDeg": round(
            float(own["normalAngle"][best_mode_index]), 4
        ),
        "fiberResidualDeg": round(
            float(own["fiberAngle"][best_mode_index]), 4
        ),
        "nearestPlanarDistanceVoxels": round(
            float(own["nearestPlanarDistance"][best_mode_index]), 4
        ),
    }
    return {
        "modeCount": len(modes),
        "passed": not reasons,
        "failureReasons": reasons,
        "bestMode": best_mode,
        "bestAssociationScore": round(best_score, 6),
        "secondModeAssociationScore": round(second_score, 6),
        "modeMargin": round(mode_margin, 6),
        "ownershipMargin": round(ownership_margin, 6),
        "bestOwnerAssociationId": best_owner,
        "bestCompetingAssociationScore": round(other_score, 6),
        "associationScoreCount": len(association_nodes),
    }


def _existing_cell_mode_match(
    mode: dict[str, Any] | None,
    nodes: np.ndarray,
    flakes_by_node: dict[int, dict[str, Any]],
    node_association: np.ndarray,
    maximum_position_residual: float,
    maximum_normal_residual: float,
    maximum_fiber_residual: float,
) -> dict[str, Any] | None:
    if mode is None or not len(nodes):
        return None
    values = []
    for node in nodes:
        metrics = _flake_pair_metrics(mode, flakes_by_node[int(node)])
        values.append(
            {
                "nodeIndex": int(node),
                "associationId": int(node_association[int(node)]),
                "positionResidualVoxels": round(
                    float(metrics["positionResidual"]), 4
                ),
                "normalResidualDeg": round(float(metrics["normalAngle"]), 4),
                "fiberResidualDeg": round(float(metrics["fiberAngle"]), 4),
                "compatibility": round(float(metrics["compatibility"]), 6),
            }
        )
    best = min(
        values,
        key=lambda value: (
            (value["positionResidualVoxels"] / maximum_position_residual) ** 2
            + (value["normalResidualDeg"] / maximum_normal_residual) ** 2
            + (value["fiberResidualDeg"] / maximum_fiber_residual) ** 2,
            value["nodeIndex"],
        ),
    )
    best["matched"] = bool(
        best["positionResidualVoxels"] <= maximum_position_residual
        and best["normalResidualDeg"] <= maximum_normal_residual
        and best["fiberResidualDeg"] <= maximum_fiber_residual
    )
    return best


def _classify_comparison(
    coarse: dict[str, Any],
    dense: dict[str, Any],
    existing_cell_match: bool = False,
) -> str:
    coarse_pass = bool(coarse["passed"])
    dense_pass = bool(dense["passed"])
    if dense_pass and not coarse_pass:
        return (
            "recovered-stored-cell-mode"
            if existing_cell_match
            else "recovered-new-dense-mode"
        )
    if dense_pass and coarse_pass:
        return "corroborated-coarse-evidence"
    if coarse_pass and not dense_pass:
        return "dense-regression"
    reasons = set(dense["failureReasons"])
    if "no-mode" in reasons:
        return "no-dense-mode"
    if "score" in reasons:
        return "dense-geometry-insufficient"
    if "mode-margin" in reasons:
        return "dense-mode-ambiguous"
    if "ownership" in reasons:
        return "dense-ownership-ambiguous"
    return "dense-unresolved"


def reanalyze_fragment_terminations(
    output_root: str | Path,
    settings: dict[str, Any] | None = None,
    force: bool = False,
    progress: Callable[[str, int, int, dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    census_stem = (
        f"fragment-termination-census-v{FRAGMENT_TERMINATION_CENSUS_VERSION}"
    )
    graph_stem = f"global-monotone-graph-v{GLOBAL_MONOTONE_GRAPH_VERSION}"
    association_stem = (
        f"global-branch-association-v{GLOBAL_BRANCH_ASSOCIATION_VERSION}"
    )
    census_summary_path = root / f"{census_stem}.json"
    census_artifact_path = root / f"{census_stem}.npz"
    graph_path = root / f"{graph_stem}.npz"
    association_path = root / f"{association_stem}.npz"
    analysis_path = root / "analysis.json"
    grid_path = root / "grid.json"
    analysis = json.loads(analysis_path.read_text())
    grid = json.loads(grid_path.read_text())
    census_summary = json.loads(census_summary_path.read_text())
    queue = census_summary["queues"]["denseAcus"][: int(resolved["maximumTargets"])]
    flake_paths = [
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(len(grid["z"]))
    ]
    identity = {
        "version": TERMINATION_REANALYSIS_VERSION,
        "settings": resolved,
        "censusArtifact": _content_identity(census_artifact_path),
        "graphArtifact": _content_identity(graph_path),
        "associationArtifact": _content_identity(association_path),
        "flakeArtifacts": [_content_identity(path) for path in flake_paths],
        "analysisIdentity": analysis["identity"],
        "largeArtifacts": [
            _file_stat_identity(root / "needles.npy"),
            _file_stat_identity(root / "needle-counts.npy"),
            _file_stat_identity(Path(analysis["identity"]["source"])),
        ],
    }
    stem = f"fragment-termination-reanalysis-v{TERMINATION_REANALYSIS_VERSION}"
    output_path = root / f"{stem}.json"
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(census_artifact_path) as payload:
        census = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(graph_path) as payload:
        graph = {key: np.asarray(payload[key]) for key in payload.files}
    with np.load(association_path) as payload:
        association = {key: np.asarray(payload[key]) for key in payload.files}
    grid_xyz = tuple(
        np.asarray(grid[axis], dtype=np.float32) for axis in ("x", "y", "z")
    )
    targets, skipped = _resolve_open_targets(
        queue,
        census["endpointClusterIndex"],
        census["endpointNodeIndex"],
        census["endpointIdentity"],
        census["endpointCenterXYZ"],
        census["endpointOutwardXYZ"],
        census["endpointQuality"],
        census["endpointEvidenceScore"],
        association["nodeAssociation"],
        graph["nodeCellIndex"],
        grid_xyz,
        float(resolved["targetDistanceVoxels"]),
    )
    source_path = Path(analysis["identity"]["source"])
    source = np.load(source_path, mmap_mode="r")
    analysis_settings = analysis["identity"]["settings"]
    crops = _coalesce_target_crops(
        np.asarray([target["targetXYZ"] for target in targets]),
        int(analysis_settings["cubeSize"]),
        int(analysis_settings["halo"]),
        tuple(int(value) for value in source.shape[::-1]),
        int(resolved["maximumCropVoxels"]),
        int(analysis_settings["binSize"]),
    )

    node_cell = graph["nodeCellIndex"].astype(np.int32)
    node_association = association["nodeAssociation"].astype(np.uint32)
    association_node_count = association["associationNodeCount"].astype(np.uint32)
    context_by_target: list[dict[int, np.ndarray]] = []
    exact_cell_associations: list[np.ndarray] = []
    exact_cell_nodes: list[np.ndarray] = []
    selected_nodes: set[int] = set()
    radius = int(resolved["competitorRadiusCells"])
    minimum_nodes = int(resolved["minimumCompetitorAssociationNodeCount"])
    for target in targets:
        target_cell = np.asarray(target["targetCellIndex"], dtype=np.int32)
        near = np.all(np.abs(node_cell - target_cell) <= radius, axis=1)
        near &= association_node_count[node_association] >= minimum_nodes
        near_nodes = np.flatnonzero(near)
        candidate_associations = np.unique(node_association[near_nodes])
        own_association = int(target["associationId"])
        if own_association not in candidate_associations:
            own_nodes = np.flatnonzero(node_association == own_association)
            distance = np.max(np.abs(node_cell[own_nodes] - target_cell), axis=1)
            nearest = own_nodes[np.argsort(distance)[: min(12, len(own_nodes))]]
            near_nodes = np.unique(np.r_[near_nodes, nearest])
            candidate_associations = np.unique(node_association[near_nodes])
        context = {
            int(candidate_association): near_nodes[
                node_association[near_nodes] == candidate_association
            ]
            for candidate_association in candidate_associations
        }
        context_by_target.append(context)
        selected_nodes.update(int(node) for node in near_nodes)
        exact = np.all(node_cell == target_cell, axis=1)
        exact_nodes = np.flatnonzero(exact)
        exact_cell_nodes.append(exact_nodes)
        exact_cell_associations.append(np.unique(node_association[exact_nodes]))
        selected_nodes.update(int(node) for node in exact_nodes)
    selected_node_array = np.asarray(sorted(selected_nodes), dtype=np.int64)
    flakes_by_node = _load_selected_flakes(
        root, graph["nodeIdentity"].astype(np.uint64), selected_node_array
    )

    global_catalog = np.load(root / "needles.npy", mmap_mode="r")
    global_counts = np.load(root / "needle-counts.npy", mmap_mode="r")
    bin_shape = tuple(int(value) for value in analysis["binShapeZYX"])
    target_results: list[dict[str, Any] | None] = [None] * len(targets)
    crop_outputs = []
    for crop_offset, crop in enumerate(crops):
        crop_target_indices = crop["targetIndices"]
        crop_points = np.asarray(
            [targets[index]["targetXYZ"] for index in crop_target_indices],
            dtype=np.float32,
        )
        dense_records, extraction = _extract_target_needles(
            source,
            crop_points,
            analysis,
            int(resolved["candidateSpacingVoxels"]),
            int(resolved["maximumNeedlesPerBin"]),
        )
        crop_outputs.append(
            {
                "cropIndex": crop_offset,
                "targetIndices": crop_target_indices,
                "requestedBoundsLowXYZ": crop["lowXYZ"].astype(int).tolist(),
                "requestedBoundsHighXYZExclusive": crop["highXYZ"]
                .astype(int)
                .tolist(),
                "requestedVoxelCount": int(crop["voxelCount"]),
                "extraction": extraction,
            }
        )
        for target_index in crop_target_indices:
            target = targets[target_index]
            target_xyz = np.asarray(target["targetXYZ"], dtype=np.float32)
            target_cell = tuple(int(value) for value in target["targetCellIndex"])
            dense_modes, dense_diagnostic = _fit_sample_modes(
                dense_records,
                target_xyz,
                target_cell,
                int(analysis_settings["cubeSize"]),
                int(resolved["maximumNeedlesPerCell"]),
                int(resolved["maximumModesPerTarget"]),
            )
            coarse_records, _ = _catalog_records_for_cell(
                global_catalog,
                global_counts,
                target_xyz,
                analysis_settings,
                bin_shape,
            )
            coarse_modes, coarse_diagnostic = _fit_sample_modes(
                coarse_records,
                target_xyz,
                target_cell,
                int(analysis_settings["cubeSize"]),
                int(analysis_settings["maxNeedles"]),
                int(resolved["maximumModesPerTarget"]),
            )
            for mode in dense_modes:
                mode["source"] = "targeted-dense"
            for mode in coarse_modes:
                mode["source"] = "whole-volume-coarse"
            context = context_by_target[target_index]
            coarse_score = _score_modes(
                coarse_modes,
                int(target["associationId"]),
                context,
                flakes_by_node,
                float(resolved["scoreThreshold"]),
                float(resolved["minimumModeMargin"]),
                float(resolved["minimumOwnershipMargin"]),
            )
            dense_score = _score_modes(
                dense_modes,
                int(target["associationId"]),
                context,
                flakes_by_node,
                float(resolved["scoreThreshold"]),
                float(resolved["minimumModeMargin"]),
                float(resolved["minimumOwnershipMargin"]),
            )
            existing_cell_mode = _existing_cell_mode_match(
                dense_score["bestMode"],
                exact_cell_nodes[target_index],
                flakes_by_node,
                node_association,
                float(
                    resolved["maximumExistingCellPositionResidualVoxels"]
                ),
                float(resolved["maximumExistingCellNormalResidualDeg"]),
                float(resolved["maximumExistingCellFiberResidualDeg"]),
            )
            classification = _classify_comparison(
                coarse_score,
                dense_score,
                bool(
                    existing_cell_mode is not None
                    and existing_cell_mode["matched"]
                ),
            )
            target_results[target_index] = {
                "queueRank": int(target["queueRank"]),
                "clusterIndex": int(target["clusterIndex"]),
                "associationId": int(target["associationId"]),
                "associationNodeCount": int(target["associationNodeCount"]),
                "clusterEndpointCount": int(target["clusterEndpointCount"]),
                "openEndpointChoiceCount": int(
                    target["openEndpointChoiceCount"]
                ),
                "sourceEndpointIdentity": int(target["sourceEndpointIdentity"]),
                "sourceEndpointXYZ": np.round(
                    target["sourceEndpointXYZ"], 3
                ).tolist(),
                "outwardXYZ": np.round(target["outwardXYZ"], 6).tolist(),
                "targetXYZ": np.round(target_xyz, 3).tolist(),
                "targetCellIndex": list(target_cell),
                "targetCellAssociationIds": exact_cell_associations[target_index]
                .astype(int)
                .tolist(),
                "localSubstantialAssociationCount": len(context),
                "nearestExistingCellMode": existing_cell_mode,
                "cropIndex": crop_offset,
                "coarse": {"diagnostic": coarse_diagnostic, **coarse_score},
                "dense": {"diagnostic": dense_diagnostic, **dense_score},
                "denseMinusCoarseScore": round(
                    float(dense_score["bestAssociationScore"])
                    - float(coarse_score["bestAssociationScore"]),
                    6,
                ),
                "classification": classification,
            }
            if progress is not None:
                completed = sum(value is not None for value in target_results)
                progress(
                    "targets",
                    completed,
                    len(targets),
                    {"classification": classification},
                )
        if progress is not None:
            progress(
                "crops",
                crop_offset + 1,
                len(crops),
                {
                    "targetCount": len(crop_target_indices),
                    "deduplicatedNeedleCount": extraction[
                        "deduplicatedNeedleCount"
                    ],
                },
            )

    completed_results = [value for value in target_results if value is not None]
    classifications = Counter(
        str(value["classification"]) for value in completed_results
    )
    recovered = [
        value
        for value in completed_results
        if str(value["classification"]).startswith("recovered-")
    ]
    recovered.sort(
        key=lambda value: (
            float(value["dense"]["bestAssociationScore"]),
            float(value["dense"]["ownershipMargin"]),
        ),
        reverse=True,
    )
    score_deltas = [
        float(value["denseMinusCoarseScore"]) for value in completed_results
    ]
    result = {
        "identity": identity,
        "settings": resolved,
        "contract": {
            "scope": (
                "ranked weak-geometry termination regions whose endpoint-resolved "
                "one-cell outward target is not already occupied by the same v6 "
                "association"
            ),
            "comparison": (
                "whole-volume coarse and freshly extracted dense needles are fit "
                "at the identical 64-cube target and scored against identical "
                "local association contexts"
            ),
            "acceptance": (
                "a dense mode must pass association score, within-target mode "
                "margin, and ownership margin among substantial associations "
                "within the declared cell radius"
            ),
            "mutation": (
                "diagnostic only; recovered modes are not inserted into the global "
                "graph or final associations"
            ),
        },
        "skippedTargets": skipped,
        "crops": crop_outputs,
        "targets": completed_results,
        "recoveredTargets": recovered,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "requestedTargetCount": len(queue),
            "openTargetCount": len(targets),
            "skippedTargetCount": len(skipped),
            "cropCount": len(crops),
            "selectedLocalNodeCount": len(selected_node_array),
            "classificationCounts": dict(sorted(classifications.items())),
            "coarsePassCount": sum(
                bool(value["coarse"]["passed"]) for value in completed_results
            ),
            "densePassCount": sum(
                bool(value["dense"]["passed"]) for value in completed_results
            ),
            "incrementalDenseRecoveryCount": len(recovered),
            "newDenseModeRecoveryCount": classifications[
                "recovered-new-dense-mode"
            ],
            "storedCellModeRecoveryCount": classifications[
                "recovered-stored-cell-mode"
            ],
            "coarseNeedlesPerTarget": _quantiles(
                [
                    float(value["coarse"]["diagnostic"]["needleCount"])
                    for value in completed_results
                ]
            ),
            "denseNeedlesPerTarget": _quantiles(
                [
                    float(value["dense"]["diagnostic"]["needleCount"])
                    for value in completed_results
                ]
            ),
            "denseMinusCoarseAssociationScore": _quantiles(score_deltas, 6),
            "gpuCropCount": sum(
                value["extraction"]["computeBackend"] == "gpu"
                for value in crop_outputs
            ),
            "totalDenseDeduplicatedNeedleCountAcrossCrops": sum(
                int(value["extraction"]["deduplicatedNeedleCount"])
                for value in crop_outputs
            ),
        },
    }
    _atomic_json(output_path, result)
    return result
