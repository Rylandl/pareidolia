from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .block import (
    BlockBounds,
    SurfaceBlock,
    augment_surface_block,
    assemble_surface_hierarchy,
    rebuild_surface_block,
)
from .contracts import (
    RawAcusSettings,
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    resolve_pipeline_manifest,
    sha256_file,
)
from .continuity import apply_join_continuity_refinement
from .evidence import CellEvidenceTable, read_evidence_artifact
from .export import write_block_obj, write_block_projection_png
from .geometry import ClippedPatch, DegeneratePlaneIntersection, clip_plane_to_cell
from .matching import TraceMatchSettings, match_face_traces
from .mode_bank import MODE_BANK_SCHEMA, load_mode_bank
from .stratigraphic_continuity import (
    PatchFingerprintTable,
    StratigraphicContinuitySettings,
    _aggregate_fingerprints,
    _mismatch_transform,
    _robust_calibration,
    _score_distributions,
    _side_patch_gauges,
    _support_mask,
    apply_stratigraphic_continuity_refinement,
    build_patch_fingerprints,
    read_patch_fingerprints,
    score_patch_fingerprints,
)
from .stratigraphy import LayerModeTable, evaluate_stratigraphy
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridFace, Int3


CONTEXTUAL_GROWTH_SCHEMA = "pareidolia.cubical-contextual-growth"
CONTEXTUAL_GROWTH_VERSION = 1
JoinKey = tuple[int, int, int, Int3]


@dataclass(frozen=True, slots=True)
class ContextualGrowthSettings:
    """Conservative operational limits for one evidence-backed growth pass."""

    minimum_support_faces: int = 2
    maximum_admission_robust_z: float = 1.0
    maximum_modes_per_trace: int = 0
    maximum_trials: int = 0
    leaf_shape_cells_xyz: Int3 = (4, 4, 3)

    def __post_init__(self) -> None:
        if self.minimum_support_faces < 2:
            raise ValueError("contextual growth requires at least two support faces")
        if not math.isfinite(self.maximum_admission_robust_z):
            raise ValueError("growth admission robust Z must be finite")
        if self.maximum_modes_per_trace < 0 or self.maximum_trials < 0:
            raise ValueError("growth limits must be nonnegative")
        leaf = tuple(int(value) for value in self.leaf_shape_cells_xyz)
        if len(leaf) != 3 or any(value <= 0 for value in leaf):
            raise ValueError("growth leaf shape must be a positive XYZ triple")
        object.__setattr__(self, "leaf_shape_cells_xyz", leaf)

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["leaf_shape_cells_xyz"] = list(self.leaf_shape_cells_xyz)
        return values


@dataclass(frozen=True, slots=True)
class GrowthSupport:
    source_patch_id: int
    source_component_id: int
    face_axis: int
    face_anchor_xyz: Int3
    geometry_score: float
    reduced_chi_square: float
    normal_angle_radians: float
    fiber_angle_radians: float | None
    fiber_quarter_turn: bool

    @property
    def face(self) -> GridFace:
        return GridFace(self.face_axis, self.face_anchor_xyz)

    @property
    def face_key(self) -> tuple[int, Int3]:
        return self.face_axis, self.face_anchor_xyz

    def record(self) -> dict[str, Any]:
        return {
            "sourcePatchId": self.source_patch_id,
            "sourceComponentId": self.source_component_id,
            "face": {
                "axis": self.face_axis,
                "anchorXYZ": list(self.face_anchor_xyz),
            },
            "geometryScore": self.geometry_score,
            "reducedChiSquare": self.reduced_chi_square,
            "normalAngleDegrees": math.degrees(self.normal_angle_radians),
            "fiberAngleDegrees": (
                math.degrees(self.fiber_angle_radians)
                if self.fiber_angle_radians is not None
                else None
            ),
            "fiberQuarterTurn": self.fiber_quarter_turn,
        }


@dataclass(frozen=True, slots=True)
class GrowthCandidate:
    candidate_id: int
    shard_id: str
    target_cell_xyz: Int3
    cell_index: int
    mode_index: int
    normal_hypothesis: int
    evidence_score: float
    material_probability: float
    effective_support: float
    supports: tuple[GrowthSupport, ...]

    def record(self) -> dict[str, Any]:
        return {
            "candidateId": self.candidate_id,
            "shardId": self.shard_id,
            "targetCellXYZ": list(self.target_cell_xyz),
            "cellIndex": self.cell_index,
            "modeIndex": self.mode_index,
            "normalHypothesis": self.normal_hypothesis,
            "evidenceScore": self.evidence_score,
            "materialProbability": self.material_probability,
            "effectiveSupport": self.effective_support,
            "supportCount": len(self.supports),
            "supportFaceCount": len({value.face_key for value in self.supports}),
            "supports": [value.record() for value in self.supports],
        }


def _join_key(match: Any) -> JoinKey:
    return (
        int(match.first_patch_id),
        int(match.second_patch_id),
        int(match.face.axis),
        tuple(int(value) for value in match.face.anchor_xyz),
    )


def _block_statistics(block: SurfaceBlock) -> dict[str, Any]:
    sizes = sorted((len(value.patch_ids) for value in block.components), reverse=True)
    deferred = Counter(value.reason for value in block.deferred_joins)
    return {
        "selectedPatches": len(block.patches),
        "candidateJoins": len(block.candidate_joins),
        "retainedJoins": len(block.joins),
        "deferredJoins": len(block.deferred_joins),
        "deferredByReason": dict(sorted(deferred.items())),
        "components": len(block.components),
        "largestComponentPatchCount": max(sizes, default=0),
        "topComponentPatchCounts": sizes[:20],
        "exteriorTraces": len(block.exterior_traces),
        "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
    }


def _mode_cell_lookup(
    mode_tables: Mapping[str, LayerModeTable],
) -> dict[Int3, tuple[str, LayerModeTable, int]]:
    result: dict[Int3, tuple[str, LayerModeTable, int]] = {}
    for shard_id in sorted(mode_tables):
        table = mode_tables[shard_id]
        table.validate()
        for cell_index, values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in values)
            if cell in result:
                raise ValueError(f"mode-bank cell {cell} is owned by multiple shards")
            result[cell] = shard_id, table, cell_index
    return result


def _other_cell(face: GridFace, source_cell: Int3) -> Int3:
    first, second = face.adjacent_cells()
    if source_cell == first:
        return second
    if source_cell == second:
        return first
    raise ValueError("open trace face is not incident to its source patch cell")


def discover_contextual_growth_candidates(
    block: SurfaceBlock,
    mode_tables: Mapping[str, LayerModeTable],
    settings: ContextualGrowthSettings | None = None,
    *,
    matching_settings: TraceMatchSettings | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[tuple[GrowthCandidate, ...], dict[str, Any]]:
    """Match genuine refined open traces to independently fitted bank modes."""

    resolved = settings or ContextualGrowthSettings()
    matcher = matching_settings or TraceMatchSettings()
    patch_by_id = {value.patch_id: value for value in block.patches}
    component_by_patch = dict(block.component_by_patch)
    selected_by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in block.patches:
        selected_by_cell[patch.cell_xyz].append(patch)
    modes_by_cell = _mode_cell_lookup(mode_tables)
    mode_patch_cache: dict[tuple[str, int], ClippedPatch | None] = {}
    grouped: dict[tuple[str, int], dict[tuple[int, int, Int3], GrowthSupport]] = (
        defaultdict(dict)
    )
    counts = Counter()
    total = len(block.unresolved_interior_traces)
    temporary_patch_id = max(patch_by_id, default=0) + 1

    for trace_index, boundary in enumerate(block.unresolved_interior_traces, start=1):
        counts["unresolvedTraces"] += 1
        source = patch_by_id[boundary.patch_id]
        face = boundary.trace.face
        target_cell = _other_cell(face, source.cell_xyz)
        if not block.bounds.contains_cell(target_cell) or target_cell not in modes_by_cell:
            counts["missingTargetModeCell"] += 1
            if progress is not None:
                progress(trace_index, total)
            continue

        selected_compatible = False
        for target in selected_by_cell.get(target_cell, ()):
            target_trace = target.trace_on(face)
            if target_trace is None:
                continue
            match = match_face_traces(
                boundary.trace,
                source.estimate,
                target_trace,
                target.estimate,
                matcher,
                grid=block.grid,
            )
            if match.accepted:
                selected_compatible = True
                break
        if selected_compatible:
            # This is an ordering/topology decision, not absent local geometry.
            counts["selectedCompatibleTraces"] += 1
            if progress is not None:
                progress(trace_index, total)
            continue

        counts["genuineOpenTraces"] += 1
        shard_id, table, cell_index = modes_by_cell[target_cell]
        accepted: list[tuple[int, Any]] = []
        for mode_index in table.mode_indices_for_cell(cell_index):
            cache_key = shard_id, mode_index
            if cache_key not in mode_patch_cache:
                try:
                    candidate_patch = clip_plane_to_cell(
                        block.grid,
                        target_cell,
                        table.mode(mode_index).estimate,
                        patch_id=temporary_patch_id + len(mode_patch_cache),
                    )
                except DegeneratePlaneIntersection:
                    candidate_patch = None
                mode_patch_cache[cache_key] = candidate_patch
            target = mode_patch_cache[cache_key]
            if target is None:
                continue
            target_trace = target.trace_on(face)
            if target_trace is None:
                continue
            match = match_face_traces(
                boundary.trace,
                source.estimate,
                target_trace,
                target.estimate,
                matcher,
                grid=block.grid,
            )
            if match.accepted:
                accepted.append((mode_index, match))
        accepted.sort(
            key=lambda value: (
                value[1].score,
                float(table.evidence_score[value[0]]),
                float(table.effective_support[value[0]]),
                -value[0],
            ),
            reverse=True,
        )
        if resolved.maximum_modes_per_trace:
            accepted = accepted[: resolved.maximum_modes_per_trace]
        if accepted:
            counts["matchedOpenTraces"] += 1
        else:
            counts["unmatchedOpenTraces"] += 1
        for mode_index, match in accepted:
            support = GrowthSupport(
                source.patch_id,
                component_by_patch[source.patch_id],
                face.axis,
                face.anchor_xyz,
                float(match.score),
                float(match.reduced_chi_square),
                float(match.normal_angle_radians),
                (
                    float(match.fiber_angle_radians)
                    if match.fiber_angle_radians is not None
                    else None
                ),
                bool(match.fiber_quarter_turn),
            )
            support_key = source.patch_id, face.axis, face.anchor_xyz
            prior = grouped[(shard_id, mode_index)].get(support_key)
            if prior is None or support.geometry_score > prior.geometry_score:
                grouped[(shard_id, mode_index)][support_key] = support
        if progress is not None:
            progress(trace_index, total)

    # The mode offset table is monotonic; build a direct mode-to-cell map once.
    owner_by_mode: dict[tuple[str, int], tuple[int, Int3]] = {}
    for shard_id in sorted(mode_tables):
        table = mode_tables[shard_id]
        for cell_index, values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in values)
            for mode_index in table.mode_indices_for_cell(cell_index):
                owner_by_mode[(shard_id, mode_index)] = cell_index, cell
    ordered = sorted(
        grouped.items(),
        key=lambda value: (
            owner_by_mode[value[0]][1][2],
            owner_by_mode[value[0]][1][1],
            owner_by_mode[value[0]][1][0],
            value[0][0],
            value[0][1],
        ),
    )
    candidates: list[GrowthCandidate] = []
    for candidate_id, ((shard_id, mode_index), support_map) in enumerate(ordered):
        table = mode_tables[shard_id]
        cell_index, cell = owner_by_mode[(shard_id, mode_index)]
        mode = table.mode(mode_index)
        supports = tuple(
            sorted(
                support_map.values(),
                key=lambda value: (
                    value.face_axis,
                    value.face_anchor_xyz,
                    value.source_patch_id,
                ),
            )
        )
        candidates.append(
            GrowthCandidate(
                candidate_id,
                shard_id,
                cell,
                cell_index,
                mode_index,
                mode.normal_hypothesis,
                mode.evidence_score,
                mode.material_probability,
                mode.effective_support,
                supports,
            )
        )
    counts["candidateModes"] = len(candidates)
    counts["multiFaceCandidateModes"] = sum(
        len({value.face_key for value in candidate.supports})
        >= resolved.minimum_support_faces
        for candidate in candidates
    )
    return tuple(candidates), dict(sorted(counts.items()))


def _concatenate_fingerprints(
    first: PatchFingerprintTable, second: PatchFingerprintTable
) -> PatchFingerprintTable:
    first.validate()
    second.validate()
    if not np.array_equal(first.depth_offsets_voxels, second.depth_offsets_voxels):
        raise ValueError("fingerprint tables use different depth grids")
    if set(int(value) for value in first.patch_id) & set(
        int(value) for value in second.patch_id
    ):
        raise ValueError("fingerprint tables contain duplicate patch IDs")
    result = PatchFingerprintTable(
        patch_id=np.concatenate((first.patch_id, second.patch_id)),
        anchor_valid=np.concatenate((first.anchor_valid, second.anchor_valid)),
        anchor_shard_index=np.concatenate(
            (first.anchor_shard_index, second.anchor_shard_index)
        ),
        anchor_mode_index=np.concatenate(
            (first.anchor_mode_index, second.anchor_mode_index)
        ),
        anchor_height_residual_voxels=np.concatenate(
            (
                first.anchor_height_residual_voxels,
                second.anchor_height_residual_voxels,
            )
        ),
        anchor_normal_residual_degrees=np.concatenate(
            (
                first.anchor_normal_residual_degrees,
                second.anchor_normal_residual_degrees,
            )
        ),
        anchor_fiber_residual_degrees=np.concatenate(
            (
                first.anchor_fiber_residual_degrees,
                second.anchor_fiber_residual_degrees,
            )
        ),
        context_mode_count=np.concatenate(
            (first.context_mode_count, second.context_mode_count)
        ),
        support_low_voxels=np.concatenate(
            (first.support_low_voxels, second.support_low_voxels)
        ),
        support_high_voxels=np.concatenate(
            (first.support_high_voxels, second.support_high_voxels)
        ),
        normal_xyz=np.concatenate((first.normal_xyz, second.normal_xyz)),
        depth_offsets_voxels=first.depth_offsets_voxels.copy(),
        density=np.concatenate((first.density, second.density)),
        orientation_moment=np.concatenate(
            (first.orientation_moment, second.orientation_moment)
        ),
    )
    result.validate()
    return result


def _graph_context(
    block: SurfaceBlock,
) -> tuple[
    dict[int, list[tuple[int, int, bool]]],
    dict[int, Int3],
]:
    patch_by_id = {value.patch_id: value for value in block.patches}
    adjacency: dict[int, list[tuple[int, int, bool]]] = defaultdict(list)
    for join_index, match in enumerate(block.joins):
        quarter_turn = bool(match.fiber_quarter_turn)
        adjacency[match.first_patch_id].append(
            (match.second_patch_id, join_index, quarter_turn)
        )
        adjacency[match.second_patch_id].append(
            (match.first_patch_id, join_index, quarter_turn)
        )
    return adjacency, {
        patch_id: patch.cell_xyz for patch_id, patch in patch_by_id.items()
    }


def _score_patch_against_side_context(
    candidate_patch_id: int,
    source_patch_id: int,
    face: GridFace,
    excluded_join_index: int,
    fingerprints: PatchFingerprintTable,
    fingerprint_index: Mapping[int, int],
    adjacency: Mapping[int, list[tuple[int, int, bool]]],
    cell_by_patch: Mapping[int, Int3],
    settings: StratigraphicContinuitySettings,
    *,
    fiber_quarter_turn: bool = False,
) -> dict[str, Any]:
    candidate_index = fingerprint_index[candidate_patch_id]
    if not bool(fingerprints.anchor_valid[candidate_index]) or int(
        fingerprints.context_mode_count[candidate_index]
    ) < settings.minimum_context_modes:
        return {"status": "insufficient-candidate-context"}
    source_cell = cell_by_patch[source_patch_id]
    face_coordinate = face.anchor_xyz[face.axis]
    source_lower = source_cell[face.axis] < face_coordinate
    source_gauges = _side_patch_gauges(
        source_patch_id,
        excluded_join_index,
        face.axis,
        face_coordinate,
        source_lower,
        settings.neighborhood_radius_hops,
        adjacency,
        cell_by_patch,
        start_fiber_quarter_turn=fiber_quarter_turn,
    )
    reference_normal = fingerprints.normal_xyz[candidate_index]
    aggregate = _aggregate_fingerprints(
        fingerprints,
        [
            (fingerprint_index[value], source_gauges[value])
            for value in sorted(source_gauges)
        ],
        reference_normal,
        settings,
    )
    if aggregate is None:
        return {"status": "insufficient-side-context"}
    density, moment, coverage, count = aggregate
    if count < settings.minimum_side_patches:
        return {"status": "insufficient-side-patches", "sidePatchCount": count}
    score = _score_distributions(
        fingerprints.depth_offsets_voxels,
        fingerprints.density[candidate_index],
        fingerprints.orientation_moment[candidate_index],
        _support_mask(fingerprints, candidate_index),
        density,
        moment,
        coverage >= settings.minimum_coverage_fraction,
    )
    if score is None:
        return {"status": "insufficient-side-density", "sidePatchCount": count}
    if score["commonDepthSpanVoxels"] < settings.minimum_common_depth_span_voxels:
        return {
            "status": "insufficient-side-depth",
            "sidePatchCount": count,
            **score,
        }
    return {"status": "scored", "sidePatchCount": count, **score}


def calibrate_contextual_growth(
    block: SurfaceBlock,
    fingerprints: PatchFingerprintTable,
    settings: StratigraphicContinuitySettings,
) -> tuple[
    dict[str, Any],
    dict[int, list[tuple[int, int, bool]]],
    dict[int, Int3],
]:
    """Calibrate local-to-local and local-to-half-neighborhood scores on joins."""

    fingerprint_index = {
        int(patch_id): index for index, patch_id in enumerate(fingerprints.patch_id)
    }
    if set(fingerprint_index) != {value.patch_id for value in block.patches}:
        raise ValueError("growth calibration fingerprint and patch sets differ")
    adjacency, cell_by_patch = _graph_context(block)
    local_by_axis: dict[int, list[float]] = defaultdict(list)
    context_by_axis: dict[int, list[float]] = defaultdict(list)
    for join_index, match in enumerate(block.joins):
        local = score_patch_fingerprints(
            fingerprints,
            fingerprint_index[match.first_patch_id],
            fingerprint_index[match.second_patch_id],
            settings,
            fiber_quarter_turn=bool(match.fiber_quarter_turn),
        )
        if local["status"] == "scored":
            local_by_axis[match.face.axis].append(
                _mismatch_transform(local["mismatch"])
            )
        for candidate_id, source_id in (
            (match.first_patch_id, match.second_patch_id),
            (match.second_patch_id, match.first_patch_id),
        ):
            context = _score_patch_against_side_context(
                candidate_id,
                source_id,
                match.face,
                join_index,
                fingerprints,
                fingerprint_index,
                adjacency,
                cell_by_patch,
                settings,
                fiber_quarter_turn=bool(match.fiber_quarter_turn),
            )
            if context["status"] == "scored":
                context_by_axis[match.face.axis].append(
                    _mismatch_transform(context["mismatch"])
                )
    calibration: dict[str, Any] = {}
    for axis in range(3):
        local_values = np.asarray(local_by_axis[axis], dtype=np.float64)
        context_values = np.asarray(context_by_axis[axis], dtype=np.float64)
        if min(len(local_values), len(context_values)) < settings.minimum_calibration_joins:
            calibration[str(axis)] = {
                "state": "insufficient-calibration",
                "localCount": len(local_values),
                "contextCount": len(context_values),
            }
            continue
        local_center, local_scale = _robust_calibration(
            local_values, settings.minimum_log_scale
        )
        context_center, context_scale = _robust_calibration(
            context_values, settings.minimum_log_scale
        )
        calibration[str(axis)] = {
            "state": "calibrated",
            "transform": "negative log similarity",
            "local": {
                "count": len(local_values),
                "median": local_center,
                "effectiveRobustScale": local_scale,
                "threshold": local_center
                + settings.outlier_standard_deviations * local_scale,
            },
            "context": {
                "count": len(context_values),
                "median": context_center,
                "effectiveRobustScale": context_scale,
                "threshold": context_center
                + settings.outlier_standard_deviations * context_scale,
            },
        }
    return calibration, adjacency, cell_by_patch


def _calibrated_score(
    score: dict[str, Any], calibration: Mapping[str, Any]
) -> dict[str, Any]:
    if score.get("status") != "scored":
        return {**score, "calibrated": False, "passes": False}
    transformed = _mismatch_transform(float(score["mismatch"]))
    center = float(calibration["median"])
    scale = float(calibration["effectiveRobustScale"])
    threshold = float(calibration["threshold"])
    return {
        **score,
        "calibrated": True,
        "transformedMismatch": transformed,
        "robustZ": (transformed - center) / scale,
        "threshold": threshold,
        "passes": transformed <= threshold,
    }


def score_contextual_growth_candidates(
    block: SurfaceBlock,
    candidates: tuple[GrowthCandidate, ...],
    selected_fingerprints: PatchFingerprintTable,
    mode_tables: Mapping[str, LayerModeTable],
    raw_settings: RawAcusSettings,
    stratigraphic_settings: StratigraphicContinuitySettings,
    growth_settings: ContextualGrowthSettings,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[list[dict[str, Any]], PatchFingerprintTable, dict[str, Any]]:
    """Require each candidate mode to agree with two calibrated graph contexts."""

    selected_ids = {value.patch_id for value in block.patches}
    if selected_ids != {int(value) for value in selected_fingerprints.patch_id}:
        raise ValueError("selected growth fingerprint and refined block sets differ")
    first_candidate_patch_id = max(selected_ids, default=0) + 1
    candidate_patches: list[ClippedPatch] = []
    candidate_patch_id: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    for candidate in candidates:
        patch_id = first_candidate_patch_id + candidate.candidate_id
        mode = mode_tables[candidate.shard_id].mode(candidate.mode_index)
        try:
            patch = clip_plane_to_cell(
                block.grid,
                candidate.target_cell_xyz,
                mode.estimate,
                patch_id=patch_id,
            )
        except DegeneratePlaneIntersection:
            patch = None
        if patch is None:
            continue
        candidate_patches.append(patch)
        candidate_patch_id[candidate.candidate_id] = patch_id
        normal_family[patch_id] = candidate.normal_hypothesis
    candidate_table = PatchTable.from_patches(
        block.grid, tuple(candidate_patches), normal_family=normal_family
    )
    candidate_fingerprints, fingerprint_statistics = build_patch_fingerprints(
        candidate_table,
        mode_tables,
        raw_settings,
        stratigraphic_settings,
    )
    fingerprints = _concatenate_fingerprints(
        selected_fingerprints, candidate_fingerprints
    )
    calibration, adjacency, cell_by_patch = calibrate_contextual_growth(
        block, selected_fingerprints, stratigraphic_settings
    )
    fingerprint_index = {
        int(patch_id): index for index, patch_id in enumerate(fingerprints.patch_id)
    }
    records: list[dict[str, Any]] = []
    total = len(candidates)
    for index, candidate in enumerate(candidates, start=1):
        patch_id = candidate_patch_id.get(candidate.candidate_id)
        support_records: list[dict[str, Any]] = []
        if patch_id is not None:
            cell_by_patch[patch_id] = candidate.target_cell_xyz
            for support in candidate.supports:
                axis_calibration = calibration[str(support.face_axis)]
                if axis_calibration["state"] != "calibrated":
                    local = {"status": "uncalibrated-face-axis"}
                    context = {"status": "uncalibrated-face-axis"}
                else:
                    local = _calibrated_score(
                        score_patch_fingerprints(
                            fingerprints,
                            fingerprint_index[patch_id],
                            fingerprint_index[support.source_patch_id],
                            stratigraphic_settings,
                            fiber_quarter_turn=support.fiber_quarter_turn,
                        ),
                        axis_calibration["local"],
                    )
                    context = _calibrated_score(
                        _score_patch_against_side_context(
                            patch_id,
                            support.source_patch_id,
                            support.face,
                            -1,
                            fingerprints,
                            fingerprint_index,
                            adjacency,
                            cell_by_patch,
                            stratigraphic_settings,
                            fiber_quarter_turn=support.fiber_quarter_turn,
                        ),
                        axis_calibration["context"],
                    )
                passes = bool(
                    local.get("passes")
                    and context.get("passes")
                    and float(local["robustZ"])
                    <= growth_settings.maximum_admission_robust_z
                    and float(context["robustZ"])
                    <= growth_settings.maximum_admission_robust_z
                )
                support_records.append(
                    {
                        **support.record(),
                        "local": local,
                        "context": context,
                        "passes": passes,
                        "maximumRobustZ": (
                            max(
                                float(local["robustZ"]),
                                float(context["robustZ"]),
                            )
                            if passes
                            else None
                        ),
                    }
                )
        best_by_face: dict[tuple[int, Int3], dict[str, Any]] = {}
        for value in support_records:
            if not value["passes"]:
                continue
            key = (
                int(value["face"]["axis"]),
                tuple(int(part) for part in value["face"]["anchorXYZ"]),
            )
            prior = best_by_face.get(key)
            score_key = (
                float(value["maximumRobustZ"]),
                -float(value["geometryScore"]),
                int(value["sourcePatchId"]),
            )
            if prior is None or score_key < (
                float(prior["maximumRobustZ"]),
                -float(prior["geometryScore"]),
                int(prior["sourcePatchId"]),
            ):
                best_by_face[key] = value
        qualified = len(best_by_face) >= growth_settings.minimum_support_faces
        robust_values = [
            float(value["maximumRobustZ"]) for value in best_by_face.values()
        ]
        records.append(
            {
                **candidate.record(),
                "candidatePatchId": patch_id,
                "fingerprint": {
                    "anchored": (
                        bool(fingerprints.anchor_valid[fingerprint_index[patch_id]])
                        if patch_id is not None
                        else False
                    ),
                    "contextModeCount": (
                        int(
                            fingerprints.context_mode_count[
                                fingerprint_index[patch_id]
                            ]
                        )
                        if patch_id is not None
                        else 0
                    ),
                },
                "supportScores": support_records,
                "qualifiedSupportFaces": [
                    {
                        "axis": key[0],
                        "anchorXYZ": list(key[1]),
                        "sourcePatchId": int(value["sourcePatchId"]),
                        "maximumRobustZ": float(value["maximumRobustZ"]),
                    }
                    for key, value in sorted(best_by_face.items())
                ],
                "qualifiedSupportFaceCount": len(best_by_face),
                "maximumQualifiedRobustZ": max(robust_values, default=None),
                "meanQualifiedRobustZ": (
                    float(np.mean(robust_values)) if robust_values else None
                ),
                "contextQualified": qualified,
            }
        )
        if progress is not None:
            progress(index, total)
    records.sort(
        key=lambda value: (
            not value["contextQualified"],
            -value["qualifiedSupportFaceCount"],
            value["maximumQualifiedRobustZ"]
            if value["maximumQualifiedRobustZ"] is not None
            else math.inf,
            value["candidateId"],
        )
    )
    return records, fingerprints, {
        "candidateFingerprints": fingerprint_statistics,
        "calibration": calibration,
    }


def _candidate_support_face_count(
    block: SurfaceBlock,
    candidate_patch_id: int,
    support_records: list[dict[str, Any]],
) -> tuple[int, list[dict[str, Any]]]:
    joins = {
        (
            frozenset((value.first_patch_id, value.second_patch_id)),
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in block.joins
    }
    closed: list[dict[str, Any]] = []
    faces: set[tuple[int, Int3]] = set()
    for support in support_records:
        if not support["passes"]:
            continue
        axis = int(support["face"]["axis"])
        anchor = tuple(int(value) for value in support["face"]["anchorXYZ"])
        key = (
            frozenset((candidate_patch_id, int(support["sourcePatchId"]))),
            axis,
            anchor,
        )
        if key in joins:
            faces.add((axis, anchor))
            closed.append(
                {
                    "sourcePatchId": int(support["sourcePatchId"]),
                    "face": {"axis": axis, "anchorXYZ": list(anchor)},
                }
            )
    return len(faces), closed


def evaluate_contextual_growth_trials(
    baseline: SurfaceBlock,
    candidate_records: list[dict[str, Any]],
    selected_patches: PatchTable,
    selected_fingerprints: PatchFingerprintTable,
    mode_tables: Mapping[str, LayerModeTable],
    evidence_tables: Mapping[str, CellEvidenceTable],
    source: VolumeSource,
    raw_settings: RawAcusSettings,
    growth_settings: ContextualGrowthSettings,
    *,
    progress: Callable[[int, int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Validate one added bank mode while preserving every refined baseline join."""

    baseline_join_keys = {_join_key(value) for value in baseline.joins}
    baseline_patch_ids = {value.patch_id for value in baseline.patches}
    patch_index = {
        int(patch_id): index for index, patch_id in enumerate(selected_patches.patch_id)
    }
    fingerprint_index = {
        int(patch_id): index
        for index, patch_id in enumerate(selected_fingerprints.patch_id)
    }
    selected_ids_by_cell: dict[Int3, list[int]] = defaultdict(list)
    for patch in baseline.patches:
        selected_ids_by_cell[patch.cell_xyz].append(patch.patch_id)
    qualified = [value for value in candidate_records if value["contextQualified"]]
    trials: list[dict[str, Any]] = []
    total = len(qualified)
    topology_trials = 0
    for trial_index, record in enumerate(qualified, start=1):
        candidate_id = int(record["candidateId"])
        shard_id = str(record["shardId"])
        table = mode_tables[shard_id]
        cell_index = int(record["cellIndex"])
        mode_index = int(record["modeIndex"])
        family = int(record["normalHypothesis"])
        target_cell = tuple(int(value) for value in record["targetCellXYZ"])
        selected_ids = selected_ids_by_cell.get(target_cell, [])
        selected_mode_indices: list[int] = []
        physical_reason: str | None = None
        for patch_id in selected_ids:
            patch_row = patch_index[patch_id]
            fingerprint_row = fingerprint_index[patch_id]
            if int(selected_patches.normal_family[patch_row]) != family:
                physical_reason = "different-selected-normal-family"
                break
            if not bool(selected_fingerprints.anchor_valid[fingerprint_row]):
                physical_reason = "unanchored-selected-layer"
                break
            selected_mode_indices.append(
                int(selected_fingerprints.anchor_mode_index[fingerprint_row])
            )
        if mode_index in selected_mode_indices:
            physical_reason = "candidate-mode-already-selected"
        physical = None
        if physical_reason is None:
            layers = [table.mode(value) for value in sorted(set(selected_mode_indices))]
            layers.append(table.mode(mode_index))
            evidence = evidence_tables[shard_id]
            if tuple(int(value) for value in evidence.cell_xyz[cell_index]) != target_cell:
                raise ValueError("mode/evidence cell ordering disagreement")
            physical = evaluate_stratigraphy(
                layers,
                source,
                raw_settings,
                family,
                float(evidence.normal_confidence[cell_index, family]),
            )
            if physical is None:
                physical_reason = "nonphysical-combined-cell-stratigraphy"

        candidate_patch_id = int(record["candidatePatchId"])
        trial_block: SurfaceBlock | None = None
        candidate_trace_count = 0
        closed_face_count = 0
        closed_supports: list[dict[str, Any]] = []
        baseline_join_loss = len(baseline_join_keys)
        if physical is not None:
            if (
                growth_settings.maximum_trials
                and topology_trials >= growth_settings.maximum_trials
            ):
                trials.append(
                    {
                        "candidateId": candidate_id,
                        "candidatePatchId": int(record["candidatePatchId"]),
                        "targetCellXYZ": list(target_cell),
                        "selectedLayerCount": len(selected_ids),
                        "combinedLayerCount": len(selected_ids) + 1,
                        "physical": True,
                        "physicalFailureReason": "not-evaluated-operational-cap",
                        "physicalScore": physical.score,
                        "closedSupportFaceCount": 0,
                        "closedSupports": [],
                        "baselineJoinLossCount": len(baseline_join_keys),
                        "delta": {
                            key: 0
                            for key in (
                                "selectedPatches",
                                "candidateJoins",
                                "retainedJoins",
                                "deferredJoins",
                                "components",
                                "largestComponentPatchCount",
                                "exteriorTraces",
                                "unresolvedInteriorTraces",
                            )
                        },
                        "recommended": False,
                    }
                )
                continue
            topology_trials += 1
            if progress is not None:
                progress(topology_trials, total, candidate_id)
            candidate_patch = clip_plane_to_cell(
                baseline.grid,
                target_cell,
                table.mode(mode_index).estimate,
                patch_id=candidate_patch_id,
            )
            if candidate_patch is None:
                physical_reason = "degenerate-candidate-polygon"
            else:
                candidate_trace_count = len(candidate_patch.traces)
                allowed_supports = {
                    (
                        candidate_patch_id,
                        int(value["sourcePatchId"]),
                        GridFace(
                            int(value["face"]["axis"]),
                            tuple(
                                int(part)
                                for part in value["face"]["anchorXYZ"]
                            ),
                        ),
                    )
                    for value in record["supportScores"]
                    if value["passes"]
                }
                augmented = augment_surface_block(
                    baseline,
                    (candidate_patch,),
                    allowed_supports=allowed_supports,
                )
                trial_block = rebuild_surface_block(augmented, augmented.joins)
                trial_join_keys = {_join_key(value) for value in trial_block.joins}
                baseline_join_loss = len(baseline_join_keys - trial_join_keys)
                closed_face_count, closed_supports = _candidate_support_face_count(
                    trial_block,
                    candidate_patch_id,
                    record["supportScores"],
                )
        before = _block_statistics(baseline)
        after = _block_statistics(trial_block) if trial_block is not None else before
        delta = {
            key: int(after[key]) - int(before[key])
            for key in (
                "selectedPatches",
                "candidateJoins",
                "retainedJoins",
                "deferredJoins",
                "components",
                "largestComponentPatchCount",
                "exteriorTraces",
                "unresolvedInteriorTraces",
            )
        }
        recommended = bool(
            trial_block is not None
            and baseline_patch_ids.issubset(
                {value.patch_id for value in trial_block.patches}
            )
            and baseline_join_loss == 0
            and closed_face_count >= growth_settings.minimum_support_faces
            and delta["retainedJoins"] >= growth_settings.minimum_support_faces
            and delta["components"] <= 0
            and delta["unresolvedInteriorTraces"]
            <= candidate_trace_count - 2 * closed_face_count
            and not trial_block.deferred_joins
        )
        trials.append(
            {
                "candidateId": candidate_id,
                "candidatePatchId": candidate_patch_id,
                "targetCellXYZ": list(target_cell),
                "selectedLayerCount": len(selected_ids),
                "combinedLayerCount": len(selected_ids) + 1,
                "candidateTraceCount": candidate_trace_count,
                "frontierDeltaBudget": (
                    candidate_trace_count - 2 * closed_face_count
                ),
                "physical": physical is not None,
                "physicalFailureReason": physical_reason,
                "physicalScore": physical.score if physical is not None else None,
                "closedSupportFaceCount": closed_face_count,
                "closedSupports": closed_supports,
                "baselineJoinLossCount": baseline_join_loss,
                "delta": delta,
                "recommended": recommended,
            }
        )
    trials.sort(
        key=lambda value: (
            not value["recommended"],
            -value["closedSupportFaceCount"],
            value["baselineJoinLossCount"],
            value["delta"]["unresolvedInteriorTraces"],
            value["candidateId"],
        )
    )
    return trials


def apply_contextual_growth_trials(
    baseline: SurfaceBlock,
    candidate_records: list[dict[str, Any]],
    trials: list[dict[str, Any]],
    mode_tables: Mapping[str, LayerModeTable],
    settings: ContextualGrowthSettings,
) -> tuple[SurfaceBlock, list[dict[str, Any]]]:
    """Greedily compose independently safe additions under the same global gates."""

    records = {int(value["candidateId"]): value for value in candidate_records}
    baseline_join_keys = {_join_key(value) for value in baseline.joins}
    accepted: list[dict[str, Any]] = []
    added_patches: list[ClippedPatch] = []
    occupied_cells: set[Int3] = set()
    ordered = [value for value in trials if value["recommended"]]
    ordered.sort(
        key=lambda value: (
            -value["closedSupportFaceCount"],
            float(records[value["candidateId"]]["maximumQualifiedRobustZ"]),
            float(records[value["candidateId"]]["meanQualifiedRobustZ"]),
            value["candidateId"],
        )
    )
    current = baseline
    for trial in ordered:
        record = records[int(trial["candidateId"])]
        cell = tuple(int(value) for value in record["targetCellXYZ"])
        if cell in occupied_cells:
            continue
        patch_id = int(trial["candidatePatchId"])
        patch = clip_plane_to_cell(
            baseline.grid,
            cell,
            mode_tables[str(record["shardId"])].mode(int(record["modeIndex"])).estimate,
            patch_id=patch_id,
        )
        if patch is None:
            continue
        allowed_supports = {
            (
                patch_id,
                int(value["sourcePatchId"]),
                GridFace(
                    int(value["face"]["axis"]),
                    tuple(int(part) for part in value["face"]["anchorXYZ"]),
                ),
            )
            for value in record["supportScores"]
            if value["passes"]
        }
        augmented = augment_surface_block(
            current,
            (patch,),
            allowed_supports=allowed_supports,
        )
        tentative = rebuild_surface_block(augmented, augmented.joins)
        tentative_keys = {_join_key(value) for value in tentative.joins}
        if baseline_join_keys - tentative_keys or tentative.deferred_joins:
            continue
        proposed = [*accepted, trial]
        all_supported = True
        for chosen in proposed:
            chosen_record = records[int(chosen["candidateId"])]
            count, _ = _candidate_support_face_count(
                tentative,
                int(chosen["candidatePatchId"]),
                chosen_record["supportScores"],
            )
            if count < settings.minimum_support_faces:
                all_supported = False
                break
        if not all_supported:
            continue
        if len(tentative.components) > len(baseline.components):
            continue
        accepted.append(trial)
        added_patches.append(patch)
        occupied_cells.add(cell)
        current = tentative
    return current, accepted


def _identity(
    input_root: Path,
    mode_bank_root: Path,
    stratigraphic_root: Path,
    settings: ContextualGrowthSettings,
) -> dict[str, Any]:
    implementation_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": CONTEXTUAL_GROWTH_SCHEMA,
        "version": CONTEXTUAL_GROWTH_VERSION,
        "inputRoot": str(input_root),
        "inputPatchManifestSha256": sha256_file(input_root / "selected-patches-v1.json"),
        "inputPatchDataSha256": sha256_file(input_root / "selected-patches-v1.npz"),
        "modeBankRoot": str(mode_bank_root),
        "modeBankManifestSha256": sha256_file(mode_bank_root / "mode-bank.json"),
        "stratigraphicRefinementRoot": str(stratigraphic_root),
        "stratigraphicManifestSha256": sha256_file(
            stratigraphic_root / "stratigraphic-refinement.json"
        ),
        "stratigraphicJoinTableSha256": sha256_file(
            stratigraphic_root / "join-stratigraphic-continuity-v1.npz"
        ),
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(implementation_root / name)
            for name in (
                "contextual_growth.py",
                "stratigraphic_continuity.py",
                "stratigraphy.py",
                "continuity.py",
                "block.py",
                "matching.py",
                "geometry.py",
                "mode_bank.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _write_retained_joins(path: Path, block: SurfaceBlock) -> None:
    keys = [_join_key(value) for value in block.joins]
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray([value[0] for value in keys], dtype=np.uint64),
            secondPatchId=np.asarray([value[1] for value in keys], dtype=np.uint64),
            faceAxis=np.asarray([value[2] for value in keys], dtype=np.int8),
            faceAnchorXYZ=np.asarray([value[3] for value in keys], dtype=np.int32),
        )
    temporary.replace(path)


def run_contextual_growth(
    stratigraphic_refinement_root: str | Path,
    output_root: str | Path,
    *,
    settings: ContextualGrowthSettings | None = None,
    force: bool = False,
    discovery_progress: Callable[[int, int], None] | None = None,
    scoring_progress: Callable[[int, int], None] | None = None,
    trial_progress: Callable[[int, int, int], None] | None = None,
) -> dict[str, Any]:
    """Run one conservative fingerprint-guided full-bank growth pass."""

    started = time.monotonic()
    resolved = settings or ContextualGrowthSettings()
    stratigraphic_root = Path(stratigraphic_refinement_root).resolve()
    output = Path(output_root).resolve()
    refinement_manifest = json.loads(
        (stratigraphic_root / "stratigraphic-refinement.json").read_text()
    )
    if refinement_manifest.get("state") != "complete":
        raise ValueError("contextual growth requires a complete stratigraphic refinement")
    input_root = Path(refinement_manifest["inputRoot"]).resolve()
    mode_bank_root = Path(refinement_manifest["modeBankRoot"]).resolve()
    join_refinement_value = refinement_manifest.get("joinRefinementRoot")
    join_refinement_root = (
        Path(join_refinement_value).resolve() if join_refinement_value else None
    )
    if output in (input_root, mode_bank_root, stratigraphic_root):
        raise ValueError("contextual growth output must differ from every input")
    pipeline_root, pipeline = resolve_pipeline_manifest(input_root)
    pipeline_identity = str(pipeline["identity"]["identitySha256"])
    bank_manifest = json.loads((mode_bank_root / "mode-bank.json").read_text())
    if (
        bank_manifest.get("schema") != MODE_BANK_SCHEMA
        or bank_manifest.get("state") != "complete"
        or bank_manifest["identity"]["inputPipelineIdentitySha256"]
        != pipeline_identity
    ):
        raise ValueError("growth mode bank and selected geometry have different inputs")
    identity = _identity(input_root, mode_bank_root, stratigraphic_root, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("contextual growth output belongs to another identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": CONTEXTUAL_GROWTH_SCHEMA,
        "version": CONTEXTUAL_GROWTH_VERSION,
        "state": "loading",
        "identity": identity,
        "inputRoot": str(input_root),
        "pipelineRoot": str(pipeline_root),
        "modeBankRoot": str(mode_bank_root),
        "stratigraphicRefinementRoot": str(stratigraphic_root),
        "joinRefinementRoot": (
            str(join_refinement_root) if join_refinement_root is not None else None
        ),
    }
    atomic_json(manifest_path, manifest)

    selected_table = read_patch_shard(input_root / "selected-patches-v1", verify=True)
    baseline = assemble_surface_hierarchy(
        selected_table.grid,
        BlockBounds((0, 0, 0), selected_table.grid.shape_cells_xyz),
        selected_table.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    if join_refinement_root is not None:
        baseline = apply_join_continuity_refinement(baseline, join_refinement_root)
    baseline = apply_stratigraphic_continuity_refinement(
        baseline, stratigraphic_root
    )
    selected_fingerprints = read_patch_fingerprints(
        stratigraphic_root / "patch-stratigraphic-fingerprints-v1",
        identity_sha256=str(refinement_manifest["identity"]["identitySha256"]),
        verify=True,
    )
    _, mode_tables = load_mode_bank(mode_bank_root, verify=True)
    raw_settings = RawAcusSettings(**pipeline["identity"]["settings"])
    stratigraphic_settings = StratigraphicContinuitySettings(
        **refinement_manifest["identity"]["settings"]
    )

    manifest["state"] = "discovering"
    atomic_json(manifest_path, manifest)
    candidates, discovery_statistics = discover_contextual_growth_candidates(
        baseline,
        mode_tables,
        resolved,
        progress=discovery_progress,
    )
    manifest["state"] = "scoring"
    atomic_json(manifest_path, manifest)
    candidate_records, _, scoring = score_contextual_growth_candidates(
        baseline,
        candidates,
        selected_fingerprints,
        mode_tables,
        raw_settings,
        stratigraphic_settings,
        resolved,
        progress=scoring_progress,
    )

    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    if source.source_identity["identitySha256"] != source_values["identitySha256"]:
        raise ValueError("native CT source identity changed since reconstruction")
    evidence_tables = {
        shard_id: read_evidence_artifact(
            pipeline_root / "shards" / shard_id / "evidence-v1",
            identity_sha256=pipeline_identity,
            verify=True,
        )
        for shard_id in pipeline["shards"]
    }
    manifest["state"] = "validating"
    atomic_json(manifest_path, manifest)
    trials = evaluate_contextual_growth_trials(
        baseline,
        candidate_records,
        selected_table,
        selected_fingerprints,
        mode_tables,
        evidence_tables,
        source,
        raw_settings,
        resolved,
        progress=trial_progress,
    )
    manifest["state"] = "composing"
    atomic_json(manifest_path, manifest)
    grown, accepted = apply_contextual_growth_trials(
        baseline, candidate_records, trials, mode_tables, resolved
    )

    candidate_by_id = {
        int(value["candidateId"]): value for value in candidate_records
    }
    accepted_by_patch = {
        int(value["candidatePatchId"]): candidate_by_id[int(value["candidateId"])]
        for value in accepted
    }
    selected_index = {
        int(patch_id): index for index, patch_id in enumerate(selected_table.patch_id)
    }
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    trial_by_patch = {int(value["candidatePatchId"]): value for value in accepted}
    for patch in grown.patches:
        if patch.patch_id in selected_index:
            index = selected_index[patch.patch_id]
            configuration_id[patch.patch_id] = int(
                selected_table.configuration_id[index]
            )
            configuration_log_weight[patch.patch_id] = float(
                selected_table.configuration_log_weight[index]
            )
            local_order[patch.patch_id] = int(selected_table.local_order[index])
            normal_family[patch.patch_id] = int(selected_table.normal_family[index])
            continue
        record = accepted_by_patch[patch.patch_id]
        configuration_id[patch.patch_id] = 0xC0000000 + int(record["candidateId"])
        configuration_log_weight[patch.patch_id] = float(
            trial_by_patch[patch.patch_id]["physicalScore"]
        )
        existing_orders = [
            int(selected_table.local_order[selected_index[value.patch_id]])
            for value in baseline.patches
            if value.cell_xyz == patch.cell_xyz
        ]
        local_order[patch.patch_id] = max(existing_orders, default=-1) + 1
        normal_family[patch.patch_id] = int(record["normalHypothesis"])
    grown_table = PatchTable.from_patches(
        grown.grid,
        grown.patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        grown_table,
        settings={
            "source": "calibrated multi-face full-mode contextual growth",
            **resolved.record(),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputRoot": str(input_root),
            "modeBankRoot": str(mode_bank_root),
            "stratigraphicRefinementRoot": str(stratigraphic_root),
            "directions": "axial/unsigned",
        },
        compressed=True,
    )
    growth_payload = {
        "schema": "pareidolia.cubical-contextual-growth-decisions",
        "version": 1,
        "identitySha256": identity_sha256,
        "discoveryStatistics": discovery_statistics,
        "scoring": scoring,
        "candidateCount": len(candidate_records),
        "contextQualifiedCandidateCount": sum(
            bool(value["contextQualified"]) for value in candidate_records
        ),
        "trialCount": len(trials),
        "recommendedTrialCount": sum(bool(value["recommended"]) for value in trials),
        "acceptedCandidateCount": len(accepted),
        "acceptedCandidateIds": [int(value["candidateId"]) for value in accepted],
        "candidates": candidate_records,
        "trials": trials,
    }
    atomic_json(output / "contextual-growth-v1.json", growth_payload)
    _write_retained_joins(output / "retained-joins-v1.npz", grown)
    obj_path = write_block_obj(grown, output / "surface.obj")
    projection_path = write_block_projection_png(
        grown, output / "projections.png", maximum_components=128
    )
    largest_path = write_block_projection_png(
        grown, output / "largest-component.png", maximum_components=1
    )
    top_twelve_path = write_block_projection_png(
        grown, output / "top-12-components.png", maximum_components=12
    )
    before = _block_statistics(baseline)
    after = _block_statistics(grown)
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-contextual-growth-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(input_root),
        "modeBankRoot": str(mode_bank_root),
        "stratigraphicRefinementRoot": str(stratigraphic_root),
        "grid": patch_manifest["grid"],
        "method": {
            "geometry": "independently fitted full-bank mode; no extrapolation",
            "context": (
                "candidate local fingerprint versus source patch and bounded "
                "source-side graph neighborhood on at least two distinct faces"
            ),
            "calibration": (
                "axis-specific retained-join local and local-to-half-neighborhood "
                "negative-log-similarity median/MAD tails"
            ),
            "physicalGate": "selected target-cell bank modes plus one candidate",
            "topologyGate": (
                "complete global reassembly while preserving every refined baseline join"
            ),
        },
        "discovery": discovery_statistics,
        "contextQualifiedCandidateCount": growth_payload[
            "contextQualifiedCandidateCount"
        ],
        "trialCount": len(trials),
        "recommendedTrialCount": growth_payload["recommendedTrialCount"],
        "acceptedCandidateCount": len(accepted),
        "acceptedCandidateIds": growth_payload["acceptedCandidateIds"],
        "baseline": before,
        "grown": after,
        "delta": {
            key: int(after[key]) - int(before[key])
            for key in (
                "selectedPatches",
                "candidateJoins",
                "retainedJoins",
                "deferredJoins",
                "components",
                "largestComponentPatchCount",
                "exteriorTraces",
                "unresolvedInteriorTraces",
            )
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "decisions": "contextual-growth-v1.json",
            "selectedPatches": "selected-patches-v1.npz",
            "retainedJoins": "retained-joins-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponent": largest_path.name,
            "topTwelveComponents": top_twelve_path.name,
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
