from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .block import BlockBounds, SurfaceBlock, assemble_surface_hierarchy
from .contracts import RawAcusSettings, VolumeSource, atomic_json
from .evidence import CellEvidenceTable
from .gaps import GapCensus, GapTraceRecord
from .geometry import DegeneratePlaneIntersection, clip_plane_to_cell
from .matching import TraceMatch, TraceMatchSettings, match_face_traces
from .selection import ConfigurationOption
from .stratigraphy import (
    CellStratigraphy,
    LayerMode,
    LayerModeTable,
    enumerate_stratigraphies,
)
from .topology import GridFace, Int3


CONTINUATION_SEARCH_SCHEMA = "pareidolia.raw-acus-mode-continuation-search"
CONTINUATION_SEARCH_VERSION = 1


@dataclass(frozen=True, slots=True)
class ContinuationSupport:
    source_patch_id: int
    face_axis: int
    face_anchor_xyz: Int3
    match_score: float
    reduced_chi_square: float
    normal_angle_radians: float
    fiber_angle_radians: float | None

    def record(self) -> dict[str, Any]:
        return {
            "sourcePatchId": self.source_patch_id,
            "face": {
                "axis": self.face_axis,
                "anchorXYZ": list(self.face_anchor_xyz),
            },
            "matchScore": self.match_score,
            "reducedChiSquare": self.reduced_chi_square,
            "normalAngleDegrees": math.degrees(self.normal_angle_radians),
            "fiberAngleDegrees": (
                math.degrees(self.fiber_angle_radians)
                if self.fiber_angle_radians is not None
                else None
            ),
        }


@dataclass(frozen=True, slots=True)
class ContinuationCandidate:
    candidate_id: int
    shard_id: str
    target_cell_xyz: Int3
    cell_index: int
    mode_index: int
    normal_hypothesis: int
    evidence_score: float
    material_probability: float
    effective_support: float
    supports: tuple[ContinuationSupport, ...]

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
            "supports": [value.record() for value in self.supports],
        }


@dataclass(frozen=True, slots=True)
class ContinuationDiscovery:
    component_id: int
    mode_gap_count: int
    matched_gap_count: int
    candidates: tuple[ContinuationCandidate, ...]

    def record(self) -> dict[str, Any]:
        return {
            "componentId": self.component_id,
            "modeGapCount": self.mode_gap_count,
            "matchedGapCount": self.matched_gap_count,
            "candidateCount": len(self.candidates),
            "candidates": [value.record() for value in self.candidates],
        }


@dataclass(frozen=True, slots=True)
class ContinuationTrial:
    candidate_id: int
    configuration_rank: int
    target_cell_xyz: Int3
    source_patch_ids: tuple[int, ...]
    source_layer_count: int
    candidate_layer_count: int
    candidate_local_score: float
    closed_gap_count: int
    support_count: int
    minimum_source_component_size_delta: int
    maximum_source_component_size_after: int
    selected_patch_delta: int
    candidate_join_delta: int
    retained_join_delta: int
    deferred_join_delta: int
    component_delta: int
    unresolved_interior_trace_delta: int
    exterior_trace_delta: int
    component_cell_collision_delta: int
    crossing_topology_delta: int
    recommended: bool

    def record(self) -> dict[str, Any]:
        return {
            "candidateId": self.candidate_id,
            "configurationRank": self.configuration_rank,
            "targetCellXYZ": list(self.target_cell_xyz),
            "sourcePatchIds": list(self.source_patch_ids),
            "sourceLayerCount": self.source_layer_count,
            "candidateLayerCount": self.candidate_layer_count,
            "layerCountDelta": self.candidate_layer_count - self.source_layer_count,
            "candidateLocalScore": self.candidate_local_score,
            "closedGapCount": self.closed_gap_count,
            "supportCount": self.support_count,
            "minimumSourceComponentSizeDelta": self.minimum_source_component_size_delta,
            "maximumSourceComponentSizeAfter": self.maximum_source_component_size_after,
            "selectedPatchDelta": self.selected_patch_delta,
            "candidateJoinDelta": self.candidate_join_delta,
            "retainedJoinDelta": self.retained_join_delta,
            "deferredJoinDelta": self.deferred_join_delta,
            "componentDelta": self.component_delta,
            "unresolvedInteriorTraceDelta": self.unresolved_interior_trace_delta,
            "exteriorTraceDelta": self.exterior_trace_delta,
            "componentCellCollisionDelta": self.component_cell_collision_delta,
            "crossingTopologyDelta": self.crossing_topology_delta,
            "recommended": self.recommended,
        }


@dataclass(frozen=True, slots=True)
class ContinuationSearch:
    discovery: ContinuationDiscovery
    trials: tuple[ContinuationTrial, ...]

    def record(self) -> dict[str, Any]:
        return {
            "discovery": self.discovery.record(),
            "statistics": {
                "trialCount": len(self.trials),
                "gapClosingTrialCount": sum(
                    value.closed_gap_count > 0 for value in self.trials
                ),
                "recommendedTrialCount": sum(
                    value.recommended for value in self.trials
                ),
            },
            "trials": [value.record() for value in self.trials],
        }


@dataclass(frozen=True, slots=True)
class ContinuationApplication:
    block: SurfaceBlock
    replacement_patches_by_cell: Mapping[Int3, tuple[Any, ...]]
    applied_trials: tuple[ContinuationTrial, ...]
    required_patch_id_by_candidate: Mapping[int, int]
    verified_closed_gap_count: int


def _cell_mode_lookup(
    mode_tables: Mapping[str, LayerModeTable],
) -> dict[Int3, tuple[str, LayerModeTable, int]]:
    result: dict[Int3, tuple[str, LayerModeTable, int]] = {}
    for shard_id, table in mode_tables.items():
        table.validate()
        for cell_index, values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in values)
            if cell in result:
                raise ValueError(f"mode-bank cell {cell} is owned by multiple shards")
            result[cell] = (shard_id, table, cell_index)
    return result


def discover_mode_continuations(
    block: SurfaceBlock,
    census: GapCensus,
    mode_tables: Mapping[str, LayerModeTable],
    *,
    matching_settings: TraceMatchSettings | None = None,
    maximum_modes_per_gap: int = 3,
) -> ContinuationDiscovery:
    """Find independently fitted target-cell modes that close explicit gaps."""

    if maximum_modes_per_gap <= 0:
        raise ValueError("maximum modes per gap must be positive")
    settings = matching_settings or TraceMatchSettings()
    patch_by_id = {value.patch_id: value for value in block.patches}
    modes_by_cell = _cell_mode_lookup(mode_tables)
    temporary_patch_id = max(patch_by_id, default=0) + 1
    grouped: dict[
        tuple[str, int], list[tuple[GapTraceRecord, TraceMatch]]
    ] = defaultdict(list)
    matched_gap_keys: set[tuple[int, int, Int3]] = set()
    mode_gaps = [
        value for value in census.traces if value.classification == "mode-gap"
    ]
    for gap in mode_gaps:
        source = patch_by_id[gap.patch_id]
        source_trace = source.trace_on(gap.face)
        if source_trace is None:
            raise RuntimeError("gap source patch no longer owns its recorded trace")
        shard_id, table, cell_index = modes_by_cell[gap.target_cell_xyz]
        accepted: list[tuple[int, TraceMatch, LayerMode]] = []
        for mode_index in table.mode_indices_for_cell(cell_index):
            mode = table.mode(mode_index)
            try:
                target = clip_plane_to_cell(
                    block.grid,
                    gap.target_cell_xyz,
                    mode.estimate,
                    patch_id=temporary_patch_id + mode_index,
                )
            except DegeneratePlaneIntersection:
                target = None
            if target is None:
                continue
            target_trace = target.trace_on(gap.face)
            if target_trace is None:
                continue
            match = match_face_traces(
                source_trace,
                source.estimate,
                target_trace,
                target.estimate,
                settings,
                grid=block.grid,
            )
            if match.accepted:
                accepted.append((mode_index, match, mode))
        accepted.sort(
            key=lambda value: (
                value[1].score,
                value[2].evidence_score,
                value[2].effective_support,
                -value[0],
            ),
            reverse=True,
        )
        for mode_index, match, _ in accepted[:maximum_modes_per_gap]:
            grouped[(shard_id, mode_index)].append((gap, match))
        if accepted:
            matched_gap_keys.add((gap.patch_id, gap.face.axis, gap.face.anchor_xyz))

    ordered = sorted(
        grouped.items(),
        key=lambda value: (
            value[1][0][0].target_cell_xyz[2],
            value[1][0][0].target_cell_xyz[1],
            value[1][0][0].target_cell_xyz[0],
            value[0][0],
            value[0][1],
        ),
    )
    candidates: list[ContinuationCandidate] = []
    for candidate_id, ((shard_id, mode_index), values) in enumerate(ordered):
        first_gap = values[0][0]
        _, table, cell_index = modes_by_cell[first_gap.target_cell_xyz]
        mode = table.mode(mode_index)
        supports = tuple(
            ContinuationSupport(
                gap.patch_id,
                gap.face.axis,
                gap.face.anchor_xyz,
                match.score,
                match.reduced_chi_square,
                match.normal_angle_radians,
                match.fiber_angle_radians,
            )
            for gap, match in sorted(
                values,
                key=lambda value: (
                    value[0].face.axis,
                    value[0].face.anchor_xyz,
                    value[0].patch_id,
                ),
            )
        )
        candidates.append(
            ContinuationCandidate(
                candidate_id,
                shard_id,
                first_gap.target_cell_xyz,
                cell_index,
                mode_index,
                mode.normal_hypothesis,
                mode.evidence_score,
                mode.material_probability,
                mode.effective_support,
                supports,
            )
        )
    return ContinuationDiscovery(
        census.component_id,
        len(mode_gaps),
        len(matched_gap_keys),
        tuple(candidates),
    )


def _conditioned_configurations(
    candidate: ContinuationCandidate,
    table: LayerModeTable,
    evidence: CellEvidenceTable,
    source: VolumeSource,
    settings: RawAcusSettings,
    *,
    minimum_layer_count: int,
    maximum_configurations: int,
) -> list[CellStratigraphy]:
    required: LayerMode | None = None
    modes: list[LayerMode] = []
    for mode_index in table.mode_indices_for_cell(candidate.cell_index):
        mode = table.mode(mode_index)
        if mode.normal_hypothesis != candidate.normal_hypothesis:
            continue
        modes.append(mode)
        if mode_index == candidate.mode_index:
            required = mode
    if required is None:
        raise ValueError("continuation mode is absent from its recorded cell")
    values = enumerate_stratigraphies(
        modes,
        source,
        settings,
        candidate.normal_hypothesis,
        float(
            evidence.normal_confidence[
                candidate.cell_index, candidate.normal_hypothesis
            ]
        ),
        required_mode=required,
    )
    retained = [
        value
        for value in values
        if len(value.layers) >= minimum_layer_count
        and any(layer is required for layer in value.layers)
    ]
    return retained[:maximum_configurations]


def _candidate_patches(
    block: SurfaceBlock,
    candidate: ContinuationCandidate,
    configuration: CellStratigraphy,
    *,
    first_patch_id: int,
) -> tuple[tuple[Any, ...], int]:
    patches = []
    required_patch_id = -1
    for layer_index, mode in enumerate(configuration.layers):
        patch_id = first_patch_id + layer_index
        try:
            patch = clip_plane_to_cell(
                block.grid,
                candidate.target_cell_xyz,
                mode.estimate,
                patch_id=patch_id,
            )
        except DegeneratePlaneIntersection:
            return (), -1
        if patch is None:
            return (), -1
        patches.append(patch)
    patch_by_id = {value.patch_id: value for value in block.patches}
    best: tuple[float, int] | None = None
    for patch in patches:
        score = 0.0
        accepted = 0
        for support in candidate.supports:
            source = patch_by_id[support.source_patch_id]
            face = GridFace(support.face_axis, support.face_anchor_xyz)
            source_trace = source.trace_on(face)
            target_trace = patch.trace_on(face)
            if source_trace is None or target_trace is None:
                continue
            match = match_face_traces(
                source_trace,
                source.estimate,
                target_trace,
                patch.estimate,
                grid=block.grid,
            )
            if match.accepted:
                accepted += 1
                score += match.score
        key = (accepted * 1000.0 + score, -patch.patch_id)
        if best is None or key > best:
            best = key
            required_patch_id = patch.patch_id
    if not candidate.supports or best is None or best[0] < 1000.0:
        return (), -1
    return tuple(patches), required_patch_id


def evaluate_mode_continuations(
    baseline: SurfaceBlock,
    discovery: ContinuationDiscovery,
    mode_tables: Mapping[str, LayerModeTable],
    evidence_tables: Mapping[str, CellEvidenceTable],
    source: VolumeSource,
    settings: RawAcusSettings,
    selected_options: Mapping[Int3, ConfigurationOption],
    *,
    reuse: ContinuationSearch | None = None,
    maximum_configurations_per_candidate: int = 3,
    maximum_leaf_shape_cells_xyz: Int3 = (4, 4, 3),
    progress: Callable[[int, int, int, int], None] | None = None,
) -> ContinuationSearch:
    """Validate mode-bank rescues by complete topology-safe reassembly."""

    if maximum_configurations_per_candidate <= 0:
        raise ValueError("maximum candidate configurations must be positive")
    patch_by_id = {value.patch_id: value for value in baseline.patches}
    component_by_patch = dict(baseline.component_by_patch)
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in baseline.components
    }
    baseline_deferred = Counter(value.reason for value in baseline.deferred_joins)
    maximum_patch_id = max(patch_by_id, default=0)
    planned: list[
        tuple[ContinuationCandidate, int, CellStratigraphy]
    ] = []
    for candidate in discovery.candidates:
        table = mode_tables[candidate.shard_id]
        evidence = evidence_tables[candidate.shard_id]
        current_count = len(selected_options[candidate.target_cell_xyz].patches)
        configurations = _conditioned_configurations(
            candidate,
            table,
            evidence,
            source,
            settings,
            minimum_layer_count=current_count,
            maximum_configurations=maximum_configurations_per_candidate,
        )
        planned.extend(
            (candidate, rank, value)
            for rank, value in enumerate(configurations, start=1)
        )

    trials: list[ContinuationTrial] = []
    completed_keys: set[tuple[int, int]] = set()
    if reuse is not None:
        current_candidates = {
            (
                value.candidate_id,
                value.shard_id,
                value.target_cell_xyz,
                value.mode_index,
            )
            for value in discovery.candidates
        }
        reused_candidates = {
            (
                value.candidate_id,
                value.shard_id,
                value.target_cell_xyz,
                value.mode_index,
            )
            for value in reuse.discovery.candidates
        }
        if current_candidates != reused_candidates:
            raise ValueError("reused continuation search has different candidates")
        for trial in reuse.trials:
            if trial.configuration_rank > maximum_configurations_per_candidate:
                continue
            trials.append(trial)
            completed_keys.add((trial.candidate_id, trial.configuration_rank))
    planned = [
        value
        for value in planned
        if (value[0].candidate_id, value[1]) not in completed_keys
    ]
    bounds = BlockBounds((0, 0, 0), baseline.grid.shape_cells_xyz)
    for index, (candidate, rank, configuration) in enumerate(planned, start=1):
        if progress is not None:
            progress(index, len(planned), candidate.candidate_id, rank)
        first_patch_id = maximum_patch_id + 1 + candidate.candidate_id * 100 + rank * 20
        replacement, required_patch_id = _candidate_patches(
            baseline,
            candidate,
            configuration,
            first_patch_id=first_patch_id,
        )
        if not replacement:
            continue
        patches = tuple(
            value
            for value in baseline.patches
            if value.cell_xyz != candidate.target_cell_xyz
        ) + replacement
        block = assemble_surface_hierarchy(
            baseline.grid,
            bounds,
            patches,
            maximum_leaf_shape_cells_xyz=maximum_leaf_shape_cells_xyz,
        )
        join_pairs = {
            frozenset((value.first_patch_id, value.second_patch_id))
            for value in block.joins
        }
        source_ids = tuple(
            sorted({value.source_patch_id for value in candidate.supports})
        )
        closed = sum(
            frozenset((source_id, required_patch_id)) in join_pairs
            for source_id in source_ids
        )
        trial_component_by_patch = dict(block.component_by_patch)
        trial_component_sizes = {
            value.component_id: len(value.patch_ids) for value in block.components
        }
        source_deltas = []
        source_sizes_after = []
        for source_id in source_ids:
            before_size = component_sizes[component_by_patch[source_id]]
            after_size = trial_component_sizes[
                trial_component_by_patch[source_id]
            ]
            source_deltas.append(after_size - before_size)
            source_sizes_after.append(after_size)
        deferred = Counter(value.reason for value in block.deferred_joins)
        retained_delta = len(block.joins) - len(baseline.joins)
        unresolved_delta = len(block.unresolved_interior_traces) - len(
            baseline.unresolved_interior_traces
        )
        collision_delta = deferred["component-cell-collision"] - baseline_deferred[
            "component-cell-collision"
        ]
        topology_delta = deferred["crossing-topology-cycle"] - baseline_deferred[
            "crossing-topology-cycle"
        ]
        current_count = len(selected_options[candidate.target_cell_xyz].patches)
        recommended = (
            closed > 0
            and len(replacement) >= current_count
            and min(source_deltas, default=0) >= 0
            and retained_delta >= 0
            and unresolved_delta <= 0
            and collision_delta <= 0
            and topology_delta <= 0
        )
        trials.append(
            ContinuationTrial(
                candidate.candidate_id,
                rank,
                candidate.target_cell_xyz,
                source_ids,
                current_count,
                len(replacement),
                configuration.score,
                closed,
                len(candidate.supports),
                min(source_deltas, default=0),
                max(source_sizes_after, default=0),
                len(block.patches) - len(baseline.patches),
                len(block.candidate_joins) - len(baseline.candidate_joins),
                retained_delta,
                len(block.deferred_joins) - len(baseline.deferred_joins),
                len(block.components) - len(baseline.components),
                unresolved_delta,
                len(block.exterior_traces) - len(baseline.exterior_traces),
                collision_delta,
                topology_delta,
                recommended,
            )
        )
    trials.sort(
        key=lambda value: (
            not value.recommended,
            -value.closed_gap_count,
            -value.minimum_source_component_size_delta,
            -value.retained_join_delta,
            value.unresolved_interior_trace_delta,
            -value.candidate_local_score,
            value.candidate_id,
            value.configuration_rank,
        )
    )
    return ContinuationSearch(discovery, tuple(trials))


def write_continuation_search(
    path: str | Path,
    search: ContinuationSearch,
    *,
    identity_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": CONTINUATION_SEARCH_SCHEMA,
        "version": CONTINUATION_SEARCH_VERSION,
        "identitySha256": identity_sha256,
        "provenance": dict(provenance or {}),
        **search.record(),
    }
    atomic_json(path, payload)
    return payload


def read_continuation_search(
    path: str | Path,
    *,
    identity_sha256: str | None = None,
) -> ContinuationSearch:
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema") != CONTINUATION_SEARCH_SCHEMA
        or int(payload.get("version", -1)) != CONTINUATION_SEARCH_VERSION
        or (
            identity_sha256 is not None
            and payload.get("identitySha256") != identity_sha256
        )
    ):
        raise ValueError("continuation search identity/schema mismatch")
    discovery_values = payload["discovery"]
    candidates = []
    for value in discovery_values["candidates"]:
        supports = tuple(
            ContinuationSupport(
                int(item["sourcePatchId"]),
                int(item["face"]["axis"]),
                tuple(int(part) for part in item["face"]["anchorXYZ"]),
                float(item["matchScore"]),
                float(item["reducedChiSquare"]),
                math.radians(float(item["normalAngleDegrees"])),
                (
                    math.radians(float(item["fiberAngleDegrees"]))
                    if item.get("fiberAngleDegrees") is not None
                    else None
                ),
            )
            for item in value["supports"]
        )
        candidates.append(
            ContinuationCandidate(
                int(value["candidateId"]),
                str(value["shardId"]),
                tuple(int(part) for part in value["targetCellXYZ"]),
                int(value["cellIndex"]),
                int(value["modeIndex"]),
                int(value["normalHypothesis"]),
                float(value["evidenceScore"]),
                float(value["materialProbability"]),
                float(value["effectiveSupport"]),
                supports,
            )
        )
    discovery = ContinuationDiscovery(
        int(discovery_values["componentId"]),
        int(discovery_values["modeGapCount"]),
        int(discovery_values["matchedGapCount"]),
        tuple(candidates),
    )
    trials = tuple(
        ContinuationTrial(
            int(value["candidateId"]),
            int(value["configurationRank"]),
            tuple(int(part) for part in value["targetCellXYZ"]),
            tuple(int(part) for part in value["sourcePatchIds"]),
            int(value["sourceLayerCount"]),
            int(value["candidateLayerCount"]),
            float(value["candidateLocalScore"]),
            int(value["closedGapCount"]),
            int(value["supportCount"]),
            int(value["minimumSourceComponentSizeDelta"]),
            int(value["maximumSourceComponentSizeAfter"]),
            int(value["selectedPatchDelta"]),
            int(value["candidateJoinDelta"]),
            int(value["retainedJoinDelta"]),
            int(value["deferredJoinDelta"]),
            int(value["componentDelta"]),
            int(value["unresolvedInteriorTraceDelta"]),
            int(value["exteriorTraceDelta"]),
            int(value["componentCellCollisionDelta"]),
            int(value["crossingTopologyDelta"]),
            bool(value["recommended"]),
        )
        for value in payload["trials"]
    )
    return ContinuationSearch(discovery, trials)


def apply_recommended_mode_continuations(
    baseline: SurfaceBlock,
    search: ContinuationSearch,
    mode_tables: Mapping[str, LayerModeTable],
    evidence_tables: Mapping[str, CellEvidenceTable],
    source: VolumeSource,
    settings: RawAcusSettings,
    selected_options: Mapping[Int3, ConfigurationOption],
    *,
    maximum_leaf_shape_cells_xyz: Int3 = (4, 4, 3),
) -> ContinuationApplication:
    """Combine one best independently safe continuation per target cell."""

    candidates = {
        value.candidate_id: value for value in search.discovery.candidates
    }
    chosen: list[ContinuationTrial] = []
    occupied_cells: set[Int3] = set()
    for trial in search.trials:
        if not trial.recommended or trial.target_cell_xyz in occupied_cells:
            continue
        chosen.append(trial)
        occupied_cells.add(trial.target_cell_xyz)
    if not chosen:
        raise ValueError("continuation search contains no recommended trials")
    maximum_patch_id = max(
        (value.patch_id for value in baseline.patches), default=0
    )
    replacements: dict[Int3, tuple[Any, ...]] = {}
    required_patch_ids: dict[int, int] = {}
    for trial in chosen:
        candidate = candidates[trial.candidate_id]
        configurations = _conditioned_configurations(
            candidate,
            mode_tables[candidate.shard_id],
            evidence_tables[candidate.shard_id],
            source,
            settings,
            minimum_layer_count=len(
                selected_options[candidate.target_cell_xyz].patches
            ),
            maximum_configurations=trial.configuration_rank,
        )
        if len(configurations) < trial.configuration_rank:
            raise RuntimeError("recorded continuation configuration is no longer available")
        configuration = configurations[trial.configuration_rank - 1]
        first_patch_id = (
            maximum_patch_id
            + 1
            + candidate.candidate_id * 100
            + trial.configuration_rank * 20
        )
        patches, required_patch_id = _candidate_patches(
            baseline,
            candidate,
            configuration,
            first_patch_id=first_patch_id,
        )
        if not patches:
            raise RuntimeError("recorded continuation no longer produces valid patches")
        replacements[candidate.target_cell_xyz] = patches
        required_patch_ids[candidate.candidate_id] = required_patch_id
    patches = tuple(
        value
        for value in baseline.patches
        if value.cell_xyz not in replacements
    ) + tuple(
        patch
        for cell in sorted(replacements, key=lambda value: (value[2], value[1], value[0]))
        for patch in replacements[cell]
    )
    block = assemble_surface_hierarchy(
        baseline.grid,
        BlockBounds((0, 0, 0), baseline.grid.shape_cells_xyz),
        patches,
        maximum_leaf_shape_cells_xyz=maximum_leaf_shape_cells_xyz,
    )
    join_pairs = {
        frozenset((value.first_patch_id, value.second_patch_id))
        for value in block.joins
    }
    verified = sum(
        frozenset(
            (
                source_id,
                required_patch_ids[trial.candidate_id],
            )
        )
        in join_pairs
        for trial in chosen
        for source_id in trial.source_patch_ids
    )
    expected = sum(value.closed_gap_count for value in chosen)
    if verified < expected:
        raise RuntimeError(
            f"combined continuation solve retained {verified}/{expected} verified joins"
        )
    baseline_component = dict(baseline.component_by_patch)
    baseline_sizes = {
        value.component_id: len(value.patch_ids) for value in baseline.components
    }
    continued_component = dict(block.component_by_patch)
    continued_sizes = {
        value.component_id: len(value.patch_ids) for value in block.components
    }
    for source_id in {
        source_id for trial in chosen for source_id in trial.source_patch_ids
    }:
        before = baseline_sizes[baseline_component[source_id]]
        after = continued_sizes[continued_component[source_id]]
        if after < before:
            raise RuntimeError(
                f"combined continuation shrank source component at patch {source_id}: "
                f"{before} -> {after}"
            )
    return ContinuationApplication(
        block,
        replacements,
        tuple(chosen),
        required_patch_ids,
        verified,
    )
