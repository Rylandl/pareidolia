from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .acus import _refine_needles_batch, _robust_common_normal
from .acus_compute import hessian_line_fields
from .rectify import grayscale_png
from .slab_analysis import (
    NEEDLE_DTYPE,
    _deduplicate_tile_needles,
    _normalize,
    _select_candidates,
    _vector_block_candidates,
)
from .slab_carrier_gaps import (
    CARRIER_GAP_VERSION,
    _internal_gap_components,
    _project_flakes_to_carrier,
)
from .slab_carrier_growth import _flake_arrays, _score_growth_candidates
from .slab_flakes import _catalog_records_for_cell, _fit_cell_flakes
from .slab_gap_census import GAP_CENSUS_VERSION, _content_identity
from .slab_sheetlet_carriers import (
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _sample_stack,
    _texture_profile,
)


GAP_REANALYSIS_VERSION = 4


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _mask_covering_points(
    mask: np.ndarray, spacing_pixels: float, maximum_points: int = 512
) -> np.ndarray:
    """Build a deterministic farthest-point covering of a binary region."""
    points = np.argwhere(np.asarray(mask, dtype=bool)).astype(np.float32)
    if not len(points):
        return np.empty((0, 2), dtype=np.int32)
    spacing = max(float(spacing_pixels), 1.0)
    centroid = np.mean(points, axis=0)
    selected = [int(np.argmin(np.sum((points - centroid) ** 2, axis=1)))]
    minimum_distance2 = np.sum((points - points[selected[0]]) ** 2, axis=1)
    while (
        float(np.max(minimum_distance2)) > spacing**2
        and len(selected) < max(1, int(maximum_points))
    ):
        index = int(np.argmax(minimum_distance2))
        selected.append(index)
        minimum_distance2 = np.minimum(
            minimum_distance2,
            np.sum((points - points[index]) ** 2, axis=1),
        )
    return points[selected].astype(np.int32)


def _cell_normal(
    records: np.ndarray,
) -> tuple[np.ndarray | None, float, dict[str, float]]:
    if len(records) < 6:
        return None, 0.0, {"medianPlaneResidualDeg": 90.0, "inlierFraction": 0.0}
    directions = np.asarray(records["direction"], dtype=np.float32)
    weights = np.asarray(records["score"], dtype=np.float32)
    normal, eigenvalues, _ = _robust_common_normal(directions, weights)
    dominant = int(np.argmax(np.abs(normal)))
    if normal[dominant] < 0.0:
        normal = -normal
    residual = np.degrees(
        np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0))
    )
    median_residual = float(np.median(residual))
    inlier_limit = max(8.0, min(22.0, median_residual * 2.8))
    inlier_fraction = float(np.mean(residual <= inlier_limit))
    confidence = float(
        np.clip(
            (
                (eigenvalues[1] - eigenvalues[0])
                / max(float(eigenvalues[2]), 1.0e-7)
            )
            * inlier_fraction,
            0.0,
            1.0,
        )
    )
    return normal, confidence, {
        "medianPlaneResidualDeg": median_residual,
        "inlierFraction": inlier_fraction,
    }


def _fit_sample_modes(
    records: np.ndarray,
    center: np.ndarray,
    cell_index: tuple[int, int, int],
    cube_size: int,
    maximum_cell_needles: int,
    maximum_modes: int = 5,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    half_cube = cube_size * 0.5
    inside = np.all(np.abs(records["center"] - center) <= half_cube, axis=1)
    local = records[inside]
    local_ids = np.flatnonzero(inside)
    available_needle_count = len(local)
    if len(local) > maximum_cell_needles:
        chosen = np.argpartition(local["score"], -maximum_cell_needles)[
            -maximum_cell_needles:
        ]
        local = local[chosen]
        local_ids = local_ids[chosen]
    normal, confidence, normal_stats = _cell_normal(local)
    if normal is None:
        return [], {
            "availableNeedleCount": int(available_needle_count),
            "needleCount": int(len(local)),
            "normalConfidence": 0.0,
            **normal_stats,
            "modeCount": 0,
        }
    fitted = _fit_cell_flakes(
        local,
        local_ids,
        center,
        normal,
        confidence,
        cell_index,
        cube_size,
        maximum_modes,
        4.0,
        12.0,
        5,
    )
    return fitted, {
        "availableNeedleCount": int(available_needle_count),
        "needleCount": int(len(local)),
        "normalConfidence": round(confidence, 4),
        "medianPlaneResidualDeg": round(normal_stats["medianPlaneResidualDeg"], 3),
        "inlierFraction": round(normal_stats["inlierFraction"], 4),
        "modeCount": len(fitted),
    }


def _aligned_target_bounds(
    sample_points_xyz: np.ndarray,
    cube_size: int,
    halo: int,
    source_shape_xyz: np.ndarray | tuple[int, int, int],
    alignment: int,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(sample_points_xyz, dtype=np.float32)
    source_shape = np.asarray(source_shape_xyz, dtype=np.int32)
    half_cube = float(cube_size) * 0.5
    low = np.floor(np.min(points, axis=0) - half_cube - halo).astype(np.int32)
    high = np.ceil(
        np.max(points, axis=0) + half_cube + halo + 1.0
    ).astype(np.int32)
    low = np.maximum(low, 0)
    high = np.minimum(high, source_shape)
    alignment = max(1, int(alignment))
    for axis in range(3):
        if int(low[axis]) > 0:
            core_low = int(low[axis]) + int(halo)
            core_low = (core_low // alignment) * alignment
            low[axis] = max(0, core_low - int(halo))
        if int(high[axis]) < int(source_shape[axis]):
            core_high = int(high[axis]) - int(halo)
            core_high = (
                (core_high + alignment - 1) // alignment
            ) * alignment
            high[axis] = min(
                int(source_shape[axis]), core_high + int(halo)
            )
    return low, high


def _extract_target_needles(
    source: np.ndarray,
    sample_points_xyz: np.ndarray,
    analysis: dict[str, Any],
    candidate_spacing: int,
    maximum_per_bin: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    settings = analysis["identity"]["settings"]
    cube_size = int(settings["cubeSize"])
    scale = float(settings["scale"])
    needle_length = float(settings["needleLength"])
    halo = int(settings["halo"])
    bin_size = int(settings["binSize"])
    low_xyz, high_xyz = _aligned_target_bounds(
        sample_points_xyz,
        cube_size,
        halo,
        source.shape[::-1],
        bin_size,
    )
    raw = np.asarray(
        source[
            low_xyz[2] : high_xyz[2],
            low_xyz[1] : high_xyz[1],
            low_xyz[0] : high_xyz[0],
        ]
    )
    data = _normalize(
        raw,
        float(analysis["normalization"]["low"]),
        float(analysis["normalization"]["high"]),
    )
    core_shape = tuple(int(value - 2 * halo) for value in data.shape)
    if min(core_shape) <= 0:
        raise ValueError("target gap crop is too small for the Acus halo")
    bin_shape = tuple(
        max(1, int(math.ceil(value / bin_size))) for value in core_shape
    )
    started = time.monotonic()
    fields, compute = hessian_line_fields(
        [data], scale, float(analysis["strengthScale"])
    )
    score, direction_field = fields[0]
    spec = {
        "coreZYX": (
            halo,
            data.shape[0] - halo,
            halo,
            data.shape[1] - halo,
            halo,
            data.shape[2] - halo,
        ),
        "paddedZYX": (0, data.shape[0], 0, data.shape[1], 0, data.shape[2]),
    }
    values, points, bin_ids = _vector_block_candidates(
        score,
        spec,
        candidate_spacing,
        halo,
        bin_size,
        bin_shape,
        0.015,
    )
    values, points, _ = _select_candidates(
        values, points, bin_ids, maximum_per_bin
    )
    candidates = np.column_stack([values, points]).astype(np.float32, copy=False)
    radius = max(3, int(math.ceil(scale * 2.5)))
    cross_section_radius = max(2.0, float(math.ceil(scale * 1.5)))
    refined = _refine_needles_batch(
        score,
        direction_field,
        candidates,
        radius,
        needle_length,
        cross_section_radius,
    )
    resolved = [
        {
            "center": np.asarray(needle["center"], dtype=np.float32) + low_xyz,
            "direction": np.asarray(needle["direction"], dtype=np.float32),
            "score": float(needle["score"]),
            "axialCoverage": float(needle["axialCoverage"]),
            "supportScore": float(needle["supportScore"]),
        }
        for needle in refined
    ]
    accepted = _deduplicate_tile_needles(
        resolved, float(max(2, candidate_spacing - 1))
    )
    records = np.empty(len(accepted), dtype=NEEDLE_DTYPE)
    for index, needle in enumerate(accepted):
        records[index] = (
            needle["center"],
            needle["direction"],
            needle["score"],
            needle["axialCoverage"],
            needle["supportScore"],
        )
    return records, {
        "sourceBoundsXYZ": [
            *low_xyz.astype(int).tolist(),
            *high_xyz.astype(int).tolist(),
        ],
        "cropShapeZYX": list(data.shape),
        "candidateCount": int(len(values)),
        "refinedNeedleCount": int(len(refined)),
        "deduplicatedNeedleCount": int(len(records)),
        "candidateGridAlignmentVoxels": bin_size,
        "computeBackend": compute["backend"],
        "computeDevice": compute["device"],
        "computeFallbackReason": compute["fallbackReason"],
        "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
    }


def _filter_modes_to_gap(
    carrier: dict[str, Any],
    gap_mask: np.ndarray,
    modes: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    if not modes:
        return []
    centers = np.asarray([mode["center"] for mode in modes], dtype=np.float32)
    y, x, inside = _project_flakes_to_carrier(carrier, centers)
    keep = np.zeros(len(modes), dtype=bool)
    valid = np.flatnonzero(inside)
    keep[valid] = gap_mask[y[valid], x[valid]]
    return [mode for mode, accepted in zip(modes, keep) if bool(accepted)]


def _candidate_metrics(
    candidates: list[dict[str, Any]],
    target_rank: np.ndarray,
    state_members: list[np.ndarray],
    flakes: list[dict[str, Any]],
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if not candidates:
        empty = np.empty((len(state_members), 0), dtype=np.float32)
        return empty, {
            "heightResidual": np.empty(0, dtype=np.float32),
            "normalAngle": np.empty(0, dtype=np.float32),
            "fiberAngle": np.empty(0, dtype=np.float32),
            "nearestPlanarDistance": np.empty(0, dtype=np.float32),
        }
    combined = flakes + candidates
    arrays = _flake_arrays(combined)
    indices = np.arange(len(flakes), len(combined), dtype=np.int64)
    scores = np.empty((len(state_members), len(candidates)), dtype=np.float32)
    target_metrics = {
        "heightResidual": np.empty(len(candidates), dtype=np.float32),
        "normalAngle": np.empty(len(candidates), dtype=np.float32),
        "fiberAngle": np.empty(len(candidates), dtype=np.float32),
        "nearestPlanarDistance": np.empty(len(candidates), dtype=np.float32),
    }
    for state_index, members in enumerate(state_members):
        scored = _score_growth_candidates(
            members, indices, arrays, combined
        )
        scores[state_index] = scored["score"]
        selected = target_rank == state_index + 1
        for key in target_metrics:
            target_metrics[key][selected] = scored[key][selected]
    return scores, target_metrics


def _accepted_candidates(
    candidates: list[dict[str, Any]],
    score_by_state: np.ndarray,
    metrics: dict[str, np.ndarray],
    score_threshold: float,
    minimum_mode_margin: float,
    minimum_ownership_margin: float,
    minimum_ct_evidence: float,
    fine_stride: float,
) -> tuple[list[int], list[dict[str, Any]]]:
    groups: dict[tuple[int, int, int], list[int]] = {}
    for index, candidate in enumerate(candidates):
        key = (
            int(candidate["targetRank"]),
            int(candidate["targetGapId"]),
            int(candidate["sampleId"]),
        )
        groups.setdefault(key, []).append(index)
    accepted: list[int] = []
    diagnostics: list[dict[str, Any]] = []
    for key, indices in groups.items():
        target_rank = key[0]
        target_scores = score_by_state[target_rank - 1, indices]
        order = np.argsort(target_scores)[::-1]
        best_index = indices[int(order[0])]
        best_score = float(target_scores[order[0]])
        second_mode_score = (
            float(target_scores[order[1]]) if len(order) > 1 else 0.0
        )
        ownership_order = np.argsort(score_by_state[:, best_index])[::-1]
        best_owner = int(ownership_order[0]) + 1
        best_owner_score = float(score_by_state[ownership_order[0], best_index])
        other_scores = np.delete(score_by_state[:, best_index], target_rank - 1)
        second_owner_score = float(np.max(other_scores, initial=0.0))
        mode_margin = best_score - second_mode_score
        ownership_margin = best_score - second_owner_score
        reasons = []
        if best_score < score_threshold:
            reasons.append("score")
        if mode_margin < minimum_mode_margin:
            reasons.append("mode-margin")
        if best_owner != target_rank or ownership_margin < minimum_ownership_margin:
            reasons.append("ownership")
        ct_evidence = float(candidates[best_index]["depthAlignedTextureScore"])
        if ct_evidence < minimum_ct_evidence:
            reasons.append("ct-evidence")
        threshold_slack = {
            "score": round(best_score - score_threshold, 4),
            "modeMargin": round(mode_margin - minimum_mode_margin, 4),
            "ownershipMargin": round(
                ownership_margin - minimum_ownership_margin, 4
            ),
            "depthAlignedTextureScore": round(
                ct_evidence - minimum_ct_evidence, 4
            ),
        }
        diagnostics.append(
            {
                "targetRank": target_rank,
                "gapId": key[1],
                "sampleId": key[2],
                "candidateIndex": best_index,
                "score": round(best_score, 4),
                "modeMargin": round(mode_margin, 4),
                "bestCarrierRank": best_owner,
                "bestCarrierScore": round(best_owner_score, 4),
                "ownershipMargin": round(ownership_margin, 4),
                "depthAlignedTextureScore": round(ct_evidence, 4),
                "heightResidualVoxels": round(
                    float(metrics["heightResidual"][best_index]), 3
                ),
                "normalResidualDeg": round(
                    float(metrics["normalAngle"][best_index]), 3
                ),
                "fiberResidualDeg": round(
                    float(metrics["fiberAngle"][best_index]), 3
                ),
                "thresholdSlack": threshold_slack,
                "minimumThresholdSlack": round(
                    min(float(value) for value in threshold_slack.values()), 4
                ),
                "accepted": not reasons,
                "rejectionReasons": reasons,
            }
        )
        if not reasons:
            accepted.append(best_index)

    deduplicated: list[int] = []
    for index in sorted(
        accepted,
        key=lambda value: float(
            score_by_state[int(candidates[value]["targetRank"]) - 1, value]
        ),
        reverse=True,
    ):
        center = np.asarray(candidates[index]["center"], dtype=np.float32)
        needle_ids = candidates[index].get("_needleIds", set())
        duplicate = False
        for previous in deduplicated:
            if int(candidates[previous]["targetRank"]) != int(
                candidates[index]["targetRank"]
            ):
                continue
            previous_center = np.asarray(
                candidates[previous]["center"], dtype=np.float32
            )
            previous_ids = candidates[previous].get("_needleIds", set())
            shared = len(needle_ids.intersection(previous_ids)) / max(
                1, min(len(needle_ids), len(previous_ids))
            )
            if (
                float(np.linalg.norm(center - previous_center)) < fine_stride * 0.5
                or shared >= 0.7
            ):
                duplicate = True
                break
        if not duplicate:
            deduplicated.append(index)
    accepted_set = set(deduplicated)
    for diagnostic in diagnostics:
        index = int(diagnostic["candidateIndex"])
        if bool(diagnostic["accepted"]) and index not in accepted_set:
            diagnostic["accepted"] = False
            diagnostic["rejectionReasons"] = ["duplicate"]
    return deduplicated, diagnostics


def _gap_support_on_carrier(
    gap_mask: np.ndarray,
    expanded_initial: dict[str, Any],
    final_carrier: dict[str, Any],
) -> np.ndarray:
    coordinates = np.argwhere(gap_mask)
    points = expanded_initial["surfaceXYZ"][gap_mask]
    valid_points = np.all(np.isfinite(points), axis=1)
    supported = np.zeros(len(points), dtype=bool)
    if np.any(valid_points):
        y, x, inside = _project_flakes_to_carrier(
            final_carrier, points[valid_points]
        )
        valid_indices = np.flatnonzero(valid_points)
        projected = valid_indices[inside]
        supported[projected] = final_carrier["supportMask"][
            y[inside], x[inside]
        ]
    output = np.zeros(gap_mask.shape, dtype=bool)
    output[coordinates[:, 0], coordinates[:, 1]] = supported
    return output


def _post_fit_diagnostics(
    member_flakes: list[dict[str, Any]],
    accepted_indices: list[int],
    candidates: list[dict[str, Any]],
    diagnostics: list[dict[str, Any]],
) -> dict[int, dict[str, Any]]:
    if not accepted_indices:
        return {}
    accepted_modes = [candidates[index] for index in accepted_indices]
    combined = member_flakes + accepted_modes
    arrays = _flake_arrays(combined)
    member_count = len(member_flakes)
    output: dict[int, dict[str, Any]] = {}
    for local_index, candidate_index in enumerate(accepted_indices):
        candidate_position = member_count + local_index
        support_indices = np.asarray(
            [index for index in range(len(combined)) if index != candidate_position],
            dtype=np.int64,
        )
        scored = _score_growth_candidates(
            support_indices,
            np.asarray([candidate_position], dtype=np.int64),
            arrays,
            combined,
        )
        before = next(
            value
            for value in diagnostics
            if int(value["candidateIndex"]) == candidate_index
        )
        score = float(scored["score"][0])
        output[candidate_index] = {
            "score": round(score, 4),
            "scoreDrift": round(score - float(before["score"]), 4),
            "heightResidualVoxels": round(
                float(scored["heightResidual"][0]), 3
            ),
            "normalResidualDeg": round(float(scored["normalAngle"][0]), 3),
            "fiberResidualDeg": round(float(scored["fiberAngle"][0]), 3),
            "nearestPlanarDistanceVoxels": round(
                float(scored["nearestPlanarDistance"][0]), 3
            ),
        }
    return output


def _classify_gap(
    target_rank: int,
    candidate_indices: list[int],
    accepted_indices: set[int],
    candidates: list[dict[str, Any]],
    metrics: dict[str, np.ndarray],
    diagnostics: list[dict[str, Any]],
    score_threshold: float,
) -> dict[str, Any]:
    accepted = [index for index in candidate_indices if index in accepted_indices]
    if accepted:
        return {
            "label": "recovered-missing-evidence",
            "acceptedFineFlakeCount": len(accepted),
        }
    owned_elsewhere = [
        value
        for value in diagnostics
        if int(value["bestCarrierRank"]) != target_rank
        and float(value["bestCarrierScore"]) >= score_threshold
    ]
    if owned_elsewhere:
        best = max(owned_elsewhere, key=lambda value: float(value["bestCarrierScore"]))
        return {
            "label": "owned-by-another-carrier",
            "acceptedFineFlakeCount": 0,
            "bestCarrierRank": int(best["bestCarrierRank"]),
            "bestCarrierScore": float(best["bestCarrierScore"]),
        }
    near_surface = [
        index
        for index in candidate_indices
        if float(metrics["heightResidual"][index]) <= 4.0
    ]
    minimum_near_fiber = min(
        (float(metrics["fiberAngle"][index]) for index in near_surface),
        default=None,
    )
    if minimum_near_fiber is not None and minimum_near_fiber >= 40.0:
        return {
            "label": "orthogonal-near-surface",
            "acceptedFineFlakeCount": 0,
            "minimumNearSurfaceFiberResidualDeg": round(minimum_near_fiber, 3),
        }
    fiber_matched = [
        index
        for index in candidate_indices
        if float(metrics["fiberAngle"][index]) <= 12.0
    ]
    minimum_fiber_height = min(
        (float(metrics["heightResidual"][index]) for index in fiber_matched),
        default=None,
    )
    if minimum_fiber_height is not None and minimum_fiber_height > 6.0:
        return {
            "label": "matching-fiber-at-other-depth",
            "acceptedFineFlakeCount": 0,
            "minimumFiberMatchedHeightResidualVoxels": round(
                minimum_fiber_height, 3
            ),
        }
    if not candidate_indices:
        return {"label": "no-dense-acus-mode", "acceptedFineFlakeCount": 0}
    return {
        "label": "insufficient-joint-agreement",
        "acceptedFineFlakeCount": 0,
    }


def reanalyze_carrier_gaps(
    output_root: str | Path,
    ranks: tuple[int, ...] | None = None,
    fine_stride: float = 8.0,
    candidate_spacing: int = 2,
    maximum_per_bin: int = 256,
    maximum_cell_needles: int = 640,
    score_threshold: float = 0.55,
    minimum_mode_margin: float = 0.04,
    minimum_ownership_margin: float = 0.04,
    minimum_ct_evidence: float = 0.5,
    force: bool = False,
) -> dict[str, Any]:
    """Re-extract Acus needles only inside selected CT-positive carrier gaps."""
    root = Path(output_root)
    output_path = root / f"sheetlet-gap-reanalysis-v{GAP_REANALYSIS_VERSION}.json"
    census_path = root / f"sheetlet-gap-census-v{GAP_CENSUS_VERSION}.json"
    if not census_path.is_file():
        raise ValueError("all-carrier gap census is required before targeted reanalysis")
    census = json.loads(census_path.read_text())
    selected_ranks = tuple(
        sorted(
            set(
                int(value)
                for value in (
                    census["queue"]["ranks"] if ranks is None else ranks
                )
            )
        )
    )
    input_identity = {"census": _content_identity(census_path)}
    settings = {
        "ranks": list(selected_ranks),
        "fineGridStrideVoxels": float(fine_stride),
        "candidateSpacingVoxels": int(candidate_spacing),
        "maximumNeedlesPerBin": int(maximum_per_bin),
        "maximumNeedlesPerCell": int(maximum_cell_needles),
        "scoreThreshold": float(score_threshold),
        "minimumModeMargin": float(minimum_mode_margin),
        "minimumOwnershipMargin": float(minimum_ownership_margin),
        "minimumDepthAlignedTextureScore": float(minimum_ct_evidence),
        "scope": "fully enclosed carrier gaps only; all-carrier ownership required",
    }
    if output_path.is_file() and not force:
        cached = json.loads(output_path.read_text())
        if (
            cached.get("settings") == settings
            and cached.get("identity", {}).get("inputArtifacts") == input_identity
        ):
            return cached

    started = time.monotonic()
    analysis = json.loads((root / "analysis.json").read_text())
    source_path, _, _, flakes = _load_carrier_catalog(root)
    source = np.load(source_path, mmap_mode="r")
    gap_fill = json.loads(
        (root / f"sheetlet-carrier-gaps-v{CARRIER_GAP_VERSION}.json").read_text()
    )
    ct_by_rank = {
        int(state["rank"]): {
            int(value["gapId"]): {
                "ctEvidence": value["ctEvidence"],
                "gate": value["gate"],
            }
            for value in state.get("gaps", [])
        }
        for state in census["states"]
    }
    with np.load(root / gap_fill["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"], dtype=np.uint32)
        member_offset = np.asarray(payload["memberOffset"], dtype=np.uint64)
    state_members = [
        member_index[int(member_offset[index]) : int(member_offset[index + 1])]
        for index in range(len(gap_fill["states"]))
    ]
    if not selected_ranks or min(selected_ranks) < 1 or max(selected_ranks) > len(state_members):
        raise ValueError(f"ranks must fall between 1 and {len(state_members)}")
    missing_ct_ranks = [
        rank
        for rank in selected_ranks
        if rank not in ct_by_rank
        or not any(
            bool(value["gate"]["queuedForDenseAcus"])
            for value in ct_by_rank[rank].values()
        )
    ]
    if missing_ct_ranks:
        raise ValueError(
            "gap census has no CT-positive gaps for ranks "
            + ", ".join(str(value) for value in missing_ct_ranks)
        )

    global_catalog = np.load(root / "needles.npy", mmap_mode="r")
    global_counts = np.load(root / "needle-counts.npy", mmap_mode="r")
    global_settings = analysis["identity"]["settings"]
    bin_shape = tuple(int(value) for value in analysis["binShapeZYX"])
    all_candidates: list[dict[str, Any]] = []
    rank_context: dict[int, dict[str, Any]] = {}

    for rank in selected_ranks:
        members = state_members[rank - 1]
        member_flakes = [flakes[int(index)] for index in members]
        carrier = _mls_carrier(member_flakes)
        expanded = _mls_carrier(member_flakes, support_radius=112.0)
        gaps = _internal_gap_components(
            carrier["supportMask"],
            float(carrier["stats"]["pixelStepVoxels"]),
            float(gap_fill["settings"]["minimumGapAreaSquareVoxels"]),
        )
        gap_outputs = []
        for gap in gaps:
            ct_record = ct_by_rank[rank].get(int(gap["gapId"]), {})
            ct_evidence = ct_record.get("ctEvidence", {})
            ct_gate = ct_record.get(
                "gate",
                {
                    "queuedForDenseAcus": False,
                    "rejectionReasons": ["missing-census-evidence"],
                },
            )
            depth_aligned_texture = float(
                ct_evidence.get("depthAlignedTextureScore", 0.0)
            )
            if not bool(ct_gate["queuedForDenseAcus"]):
                gap_outputs.append(
                    {
                        **{key: value for key, value in gap.items() if key != "mask"},
                        "status": "ct-rejected",
                        "sampleCount": 0,
                        "ctEvidence": ct_evidence,
                        "ctGate": ct_gate,
                        "samplePointsYX": [],
                        "extraction": None,
                        "coarse": {"fittedModeCount": 0, "samples": []},
                        "dense": {
                            "fittedModeCount": 0,
                            "candidateRange": [len(all_candidates), len(all_candidates)],
                            "samples": [],
                        },
                    }
                )
                continue
            samples_yx = _mask_covering_points(
                gap["mask"],
                fine_stride / float(carrier["stats"]["pixelStepVoxels"]),
            )
            sample_points = np.asarray(
                [expanded["surfaceXYZ"][y, x] for y, x in samples_yx],
                dtype=np.float32,
            )
            finite = np.all(np.isfinite(sample_points), axis=1)
            samples_yx = samples_yx[finite]
            sample_points = sample_points[finite]
            if not len(sample_points):
                continue
            dense_records, extraction = _extract_target_needles(
                source,
                sample_points,
                analysis,
                candidate_spacing,
                maximum_per_bin,
            )
            dense_modes: list[dict[str, Any]] = []
            dense_samples = []
            coarse_modes: list[dict[str, Any]] = []
            coarse_samples = []
            for sample_id, center in enumerate(sample_points):
                fitted, diagnostic = _fit_sample_modes(
                    dense_records,
                    center,
                    (sample_id, int(gap["gapId"]) - 1, rank - 1),
                    int(global_settings["cubeSize"]),
                    maximum_cell_needles,
                )
                dense_samples.append({"sampleId": sample_id, **diagnostic})
                for mode in fitted:
                    mode["targetRank"] = rank
                    mode["targetGapId"] = int(gap["gapId"])
                    mode["sampleId"] = sample_id
                    mode["source"] = "targeted-dense"
                    mode["depthAlignedTextureScore"] = depth_aligned_texture
                    dense_modes.append(mode)

                records, record_ids = _catalog_records_for_cell(
                    global_catalog,
                    global_counts,
                    center,
                    global_settings,
                    bin_shape,
                )
                normal, confidence, normal_stats = _cell_normal(records)
                fitted_coarse = (
                    _fit_cell_flakes(
                        records,
                        record_ids,
                        center,
                        normal,
                        confidence,
                        (sample_id, int(gap["gapId"]) - 1, rank - 1),
                        int(global_settings["cubeSize"]),
                        5,
                        4.0,
                        12.0,
                        5,
                    )
                    if normal is not None
                    else []
                )
                coarse_samples.append(
                    {
                        "sampleId": sample_id,
                        "needleCount": int(len(records)),
                        "normalConfidence": round(confidence, 4),
                        "medianPlaneResidualDeg": round(
                            normal_stats["medianPlaneResidualDeg"], 3
                        ),
                        "modeCount": len(fitted_coarse),
                    }
                )
                coarse_modes.extend(fitted_coarse)

            dense_modes = _filter_modes_to_gap(carrier, gap["mask"], dense_modes)
            coarse_modes = _filter_modes_to_gap(carrier, gap["mask"], coarse_modes)
            candidate_start = len(all_candidates)
            all_candidates.extend(dense_modes)
            gap_outputs.append(
                {
                    **{key: value for key, value in gap.items() if key != "mask"},
                    "status": "reanalyzed",
                    "sampleCount": int(len(sample_points)),
                    "ctEvidence": ct_evidence,
                    "ctGate": ct_gate,
                    "samplePointsYX": samples_yx.astype(int).tolist(),
                    "extraction": extraction,
                    "coarse": {
                        "fittedModeCount": len(coarse_modes),
                        "medianNeedlesPerSample": round(
                            float(
                                np.median(
                                    [value["needleCount"] for value in coarse_samples]
                                )
                            ),
                            2,
                        ),
                        "samples": coarse_samples,
                    },
                    "dense": {
                        "fittedModeCount": len(dense_modes),
                        "medianNeedlesPerSample": round(
                            float(
                                np.median(
                                    [value["needleCount"] for value in dense_samples]
                                )
                            ),
                            2,
                        ),
                        "candidateRange": [
                            candidate_start,
                            len(all_candidates),
                        ],
                        "samples": dense_samples,
                    },
                }
            )
        rank_context[rank] = {
            "members": members,
            "memberFlakes": member_flakes,
            "carrier": carrier,
            "expanded": expanded,
            "gaps": gaps,
            "gapOutputs": gap_outputs,
        }

    target_rank = np.asarray(
        [int(candidate["targetRank"]) for candidate in all_candidates],
        dtype=np.int32,
    )
    score_by_state, metrics = _candidate_metrics(
        all_candidates, target_rank, state_members, flakes
    )
    accepted_indices, sample_diagnostics = _accepted_candidates(
        all_candidates,
        score_by_state,
        metrics,
        score_threshold,
        minimum_mode_margin,
        minimum_ownership_margin,
        minimum_ct_evidence,
        fine_stride,
    )
    accepted_set = set(accepted_indices)
    artifact_root = root / f"sheetlet-gap-reanalysis-v{GAP_REANALYSIS_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    rank_outputs = []
    for rank in selected_ranks:
        context = rank_context[rank]
        accepted_for_rank = [
            index
            for index in accepted_indices
            if int(all_candidates[index]["targetRank"]) == rank
        ]
        accepted_modes = [all_candidates[index] for index in accepted_for_rank]
        rank_candidate_indices = [
            index
            for index, candidate in enumerate(all_candidates)
            if int(candidate["targetRank"]) == rank
        ]
        rank_diagnostics = [
            value
            for value in sample_diagnostics
            if int(value["targetRank"]) == rank
        ]
        post_fit = _post_fit_diagnostics(
            context["memberFlakes"],
            accepted_for_rank,
            all_candidates,
            rank_diagnostics,
        )
        for diagnostic in rank_diagnostics:
            candidate_index = int(diagnostic["candidateIndex"])
            if candidate_index in post_fit:
                diagnostic["postFit"] = post_fit[candidate_index]
        final_carrier = _mls_carrier(context["memberFlakes"] + accepted_modes)
        gap_map = np.zeros(context["carrier"]["supportMask"].shape, dtype=np.uint8)
        gap_map[context["carrier"]["supportMask"]] = 48
        initial_gap_pixels = 0
        newly_supported_pixels = 0
        gap_support = []
        for gap in context["gaps"]:
            gap_map[gap["mask"]] = 160
            supported = _gap_support_on_carrier(
                gap["mask"], context["expanded"], final_carrier
            )
            gap_map[supported] = 255
            initial_gap_pixels += int(gap["pixelCount"])
            newly_supported_pixels += int(np.count_nonzero(supported))
            gap_support.append(
                {
                    "gapId": int(gap["gapId"]),
                    "initialPixels": int(gap["pixelCount"]),
                    "newlySupportedPixels": int(np.count_nonzero(supported)),
                    "remainingPixels": int(
                        int(gap["pixelCount"]) - np.count_nonzero(supported)
                    ),
                }
            )
        rank_root = artifact_root / f"rank-{rank:02d}"
        rank_root.mkdir(parents=True, exist_ok=True)
        gap_map_path = rank_root / "gap-map.png"
        gap_map_path.write_bytes(grayscale_png(gap_map))
        depth_offsets = np.arange(-6.0, 6.01, 1.0, dtype=np.float32)
        stack, sampling = _sample_stack(source, final_carrier, depth_offsets)
        texture = _texture_profile(stack, final_carrier["supportMask"], depth_offsets)
        best_index = int(
            np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"]))
        )
        best_path = rank_root / "best-texture.png"
        best_path.write_bytes(
            grayscale_png(_contrast(stack[best_index], final_carrier["supportMask"]))
        )
        near_surface = [
            index
            for index in rank_candidate_indices
            if float(metrics["heightResidual"][index]) <= 4.0
        ]
        fiber_matched = [
            index
            for index in rank_candidate_indices
            if float(metrics["fiberAngle"][index]) <= 12.0
        ]
        minimum_fiber_matched_height = min(
            (float(metrics["heightResidual"][index]) for index in fiber_matched),
            default=None,
        )
        serialized_modes = []
        for index in accepted_for_rank:
            mode = {
                key: value
                for key, value in all_candidates[index].items()
                if key != "_needleIds"
            }
            diagnostic = next(
                value
                for value in rank_diagnostics
                if int(value["candidateIndex"]) == index
            )
            serialized_modes.append({**mode, "acceptance": diagnostic})
        classified_gaps = []
        for gap_output in context["gapOutputs"]:
            gap_id = int(gap_output["gapId"])
            gap_candidates = [
                index
                for index in rank_candidate_indices
                if int(all_candidates[index]["targetGapId"]) == gap_id
            ]
            gap_diagnostics = [
                value
                for value in rank_diagnostics
                if int(value["gapId"]) == gap_id
            ]
            classification = (
                {
                    "label": "ct-negative",
                    "acceptedFineFlakeCount": 0,
                }
                if gap_output["status"] == "ct-rejected"
                else _classify_gap(
                    rank,
                    gap_candidates,
                    accepted_set,
                    all_candidates,
                    metrics,
                    gap_diagnostics,
                    score_threshold,
                )
            )
            classified_gaps.append(
                {**gap_output, "classification": classification}
            )
        rank_outputs.append(
            {
                "rank": rank,
                "initialFlakeCount": len(context["memberFlakes"]),
                "acceptedFineFlakeCount": len(accepted_for_rank),
                "candidateModeCount": len(rank_candidate_indices),
                "initialCarrier": context["carrier"]["stats"],
                "finalCarrier": final_carrier["stats"],
                "gaps": classified_gaps,
                "sampleDiagnostics": rank_diagnostics,
                "failureProfile": {
                    "nearSurfaceCandidateCount": len(near_surface),
                    "minimumNearSurfaceFiberResidualDeg": round(
                        min(
                            (float(metrics["fiberAngle"][index]) for index in near_surface),
                            default=90.0,
                        ),
                        3,
                    ),
                    "fiberMatchedCandidateCount": len(fiber_matched),
                    "minimumFiberMatchedHeightResidualVoxels": (
                        round(minimum_fiber_matched_height, 3)
                        if minimum_fiber_matched_height is not None
                        else None
                    ),
                },
                "acceptedFineFlakes": serialized_modes,
                "gapPixels": {
                    "initial": initial_gap_pixels,
                    "newlySupported": newly_supported_pixels,
                    "remaining": initial_gap_pixels - newly_supported_pixels,
                    "components": gap_support,
                },
                "texture": {
                    key: texture[key] for key in texture if key != "planes"
                },
                "yield": _carrier_yield(final_carrier["stats"], texture),
                "sampling": sampling,
                "artifacts": {
                    "gapMap": str(gap_map_path.relative_to(root)),
                    "bestTextureImage": str(best_path.relative_to(root)),
                },
            }
        )

    result = {
        "identity": {
            "version": GAP_REANALYSIS_VERSION,
            "analysis": analysis["identity"],
            "input": str(census_path),
            "inputArtifacts": input_identity,
        },
        "settings": settings,
        "ranks": rank_outputs,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "auditedRankCount": len(selected_ranks),
            "sampleCount": sum(
                int(gap["sampleCount"])
                for value in rank_outputs
                for gap in value["gaps"]
            ),
            "candidateModeCount": len(all_candidates),
            "acceptedFineFlakeCount": len(accepted_set),
            "allCarrierOwnershipChecks": len(state_members) * len(all_candidates),
            "classificationCounts": {
                label: sum(
                    int(gap["classification"]["label"] == label)
                    for value in rank_outputs
                    for gap in value["gaps"]
                )
                for label in sorted(
                    {
                        gap["classification"]["label"]
                        for value in rank_outputs
                        for gap in value["gaps"]
                    }
                )
            },
        },
    }
    _atomic_json(output_path, result)
    return result
