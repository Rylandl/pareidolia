from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import SurfaceBlock
from .matching import TraceMatch, TraceMatchSettings, match_face_traces
from .selection import ConfigurationOption
from .topology import GridFace, Int3
from .contracts import atomic_json


GAP_CENSUS_SCHEMA = "pareidolia.raw-acus-gap-census"
GAP_CENSUS_VERSION = 1


@dataclass(frozen=True, slots=True)
class GapAlternative:
    option_id: int
    target_patch_id: int
    layer_count: int
    log_weight_penalty: float
    match_score: float
    reduced_chi_square: float
    normal_angle_radians: float
    fiber_angle_radians: float | None

    def record(self) -> dict[str, Any]:
        return {
            "optionId": self.option_id,
            "targetPatchId": self.target_patch_id,
            "layerCount": self.layer_count,
            "logWeightPenalty": self.log_weight_penalty,
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
class GapTraceRecord:
    component_id: int
    patch_id: int
    cell_xyz: Int3
    face: GridFace
    target_cell_xyz: Int3
    classification: str
    selected_option_id: int
    selected_target_layer_count: int
    selected_target_trace_count: int
    compatible_selected_patch_ids: tuple[int, ...]
    compatible_pair_states: tuple[str, ...]
    occupied_compatible_count: int
    alternatives: tuple[GapAlternative, ...]

    @property
    def best_alternative(self) -> GapAlternative | None:
        return self.alternatives[0] if self.alternatives else None

    def record(self) -> dict[str, Any]:
        return {
            "componentId": self.component_id,
            "patchId": self.patch_id,
            "cellXYZ": list(self.cell_xyz),
            "face": {
                "axis": self.face.axis,
                "anchorXYZ": list(self.face.anchor_xyz),
            },
            "targetCellXYZ": list(self.target_cell_xyz),
            "classification": self.classification,
            "selectedOptionId": self.selected_option_id,
            "selectedTargetLayerCount": self.selected_target_layer_count,
            "selectedTargetTraceCount": self.selected_target_trace_count,
            "compatibleSelectedPatchIds": list(
                self.compatible_selected_patch_ids
            ),
            "compatiblePairStates": list(self.compatible_pair_states),
            "occupiedCompatibleCount": self.occupied_compatible_count,
            "alternatives": [value.record() for value in self.alternatives],
        }


@dataclass(frozen=True, slots=True)
class GapCensus:
    component_id: int
    component_patch_count: int
    traces: tuple[GapTraceRecord, ...]

    def record(self) -> dict[str, Any]:
        classifications = Counter(value.classification for value in self.traces)
        by_axis = {
            str(axis): dict(
                sorted(
                    Counter(
                        value.classification
                        for value in self.traces
                        if value.face.axis == axis
                    ).items()
                )
            )
            for axis in range(3)
        }
        recoverable = [
            value.best_alternative.log_weight_penalty
            for value in self.traces
            if value.classification == "recoverable-configuration-gap"
            and value.best_alternative is not None
        ]
        if recoverable:
            quantiles = np.percentile(
                np.asarray(recoverable, dtype=np.float64), (0, 50, 90, 100)
            )
            penalties: dict[str, float | None] = {
                name: round(float(value), 6)
                for name, value in zip(
                    ("minimum", "median", "p90", "maximum"), quantiles
                )
            }
        else:
            penalties = {
                "minimum": None,
                "median": None,
                "p90": None,
                "maximum": None,
            }
        return {
            "component": {
                "id": self.component_id,
                "patchCount": self.component_patch_count,
            },
            "statistics": {
                "unresolvedInteriorTraceCount": len(self.traces),
                "classifications": dict(sorted(classifications.items())),
                "classificationsByFaceAxisXYZ": by_axis,
                "recoverableConfigurationGapPenalty": penalties,
            },
            "traces": [value.record() for value in self.traces],
        }


def _pair_key(first: int, second: int, face: GridFace) -> tuple[int, int, GridFace]:
    return min(first, second), max(first, second), face


def _accepted_match(
    source: Any,
    source_trace: Any,
    target: Any,
    face: GridFace,
    grid: Any,
    settings: TraceMatchSettings,
) -> TraceMatch | None:
    target_trace = target.trace_on(face)
    if target_trace is None:
        return None
    value = match_face_traces(
        source_trace,
        source.estimate,
        target_trace,
        target.estimate,
        settings,
        grid=grid,
    )
    return value if value.accepted else None


def analyze_component_gaps(
    block: SurfaceBlock,
    options_by_cell: Mapping[Int3, tuple[ConfigurationOption, ...]],
    selected_option_ids: Mapping[Int3, int],
    *,
    component_id: int | None = None,
    matching_settings: TraceMatchSettings | None = None,
) -> GapCensus:
    """Classify unresolved traces without weakening global topology vetoes."""

    settings = matching_settings or TraceMatchSettings()
    if not block.components:
        raise ValueError("gap census requires at least one assembled component")
    if component_id is None:
        component = max(
            block.components, key=lambda value: (len(value.patch_ids), -value.component_id)
        )
    else:
        candidates = [
            value for value in block.components if value.component_id == component_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"unknown component id {component_id}")
        component = candidates[0]
    component_ids = set(component.patch_ids)
    patch_by_id = {value.patch_id: value for value in block.patches}
    patches_by_cell: dict[Int3, list[Any]] = defaultdict(list)
    for patch in block.patches:
        patches_by_cell[patch.cell_xyz].append(patch)
    selected_options: dict[Int3, ConfigurationOption] = {}
    for cell, option_id in selected_option_ids.items():
        candidates = [
            value for value in options_by_cell[cell] if value.option_id == option_id
        ]
        if len(candidates) != 1:
            raise ValueError(f"selected option {option_id} is absent from cell {cell}")
        selected_options[cell] = candidates[0]

    joined_keys = {
        (patch_id, value.face)
        for value in block.joins
        for patch_id in (value.first_patch_id, value.second_patch_id)
    }
    candidate_keys = {
        _pair_key(value.first_patch_id, value.second_patch_id, value.face)
        for value in block.candidate_joins
    }
    deferred = {
        _pair_key(
            value.match.first_patch_id,
            value.match.second_patch_id,
            value.match.face,
        ): value.reason
        for value in block.deferred_joins
    }

    records: list[GapTraceRecord] = []
    for boundary in block.unresolved_interior_traces:
        if boundary.patch_id not in component_ids:
            continue
        source = patch_by_id[boundary.patch_id]
        face = boundary.trace.face
        lower, upper = face.adjacent_cells()
        target_cell = upper if source.cell_xyz == lower else lower
        if target_cell not in options_by_cell or target_cell not in selected_options:
            raise ValueError(f"gap target cell {target_cell} has no configuration")
        current = selected_options[target_cell]
        selected_targets = [
            value
            for value in patches_by_cell[target_cell]
            if value.trace_on(face) is not None
        ]
        compatible: list[tuple[Any, TraceMatch, str, bool]] = []
        for target in selected_targets:
            match = _accepted_match(
                source,
                boundary.trace,
                target,
                face,
                block.grid,
                settings,
            )
            if match is None:
                continue
            key = _pair_key(source.patch_id, target.patch_id, face)
            if key in deferred:
                state = deferred[key]
            elif key in candidate_keys:
                state = "candidate-not-retained"
            else:
                state = "ordered-face-assignment"
            compatible.append(
                (target, match, state, (target.patch_id, face) in joined_keys)
            )

        alternatives: list[GapAlternative] = []
        for option in options_by_cell[target_cell]:
            if option.option_id == current.option_id:
                continue
            option_matches: list[tuple[Any, TraceMatch]] = []
            for target in option.patches:
                match = _accepted_match(
                    source,
                    boundary.trace,
                    target,
                    face,
                    block.grid,
                    settings,
                )
                if match is not None:
                    option_matches.append((target, match))
            if not option_matches:
                continue
            target, match = max(
                option_matches,
                key=lambda value: (
                    value[1].score,
                    -value[1].negative_log_likelihood,
                    -value[0].patch_id,
                ),
            )
            alternatives.append(
                GapAlternative(
                    option.option_id,
                    target.patch_id,
                    len(option.patches),
                    current.log_weight - option.log_weight,
                    match.score,
                    match.reduced_chi_square,
                    match.normal_angle_radians,
                    match.fiber_angle_radians,
                )
            )
        alternatives.sort(
            key=lambda value: (
                value.log_weight_penalty,
                -value.match_score,
                value.option_id,
                value.target_patch_id,
            )
        )

        states = {value[2] for value in compatible}
        if "component-cell-collision" in states:
            classification = "component-cell-collision-veto"
        elif "crossing-topology-cycle" in states:
            classification = "crossing-topology-veto"
        elif compatible:
            classification = "ordered-face-assignment"
        elif alternatives:
            classification = "recoverable-configuration-gap"
        else:
            classification = "mode-gap"
        records.append(
            GapTraceRecord(
                component.component_id,
                source.patch_id,
                source.cell_xyz,
                face,
                target_cell,
                classification,
                current.option_id,
                len(current.patches),
                len(selected_targets),
                tuple(value[0].patch_id for value in compatible),
                tuple(value[2] for value in compatible),
                sum(value[3] for value in compatible),
                tuple(alternatives),
            )
        )
    records.sort(
        key=lambda value: (
            value.face.axis,
            value.face.anchor_xyz,
            value.patch_id,
        )
    )
    return GapCensus(component.component_id, len(component.patch_ids), tuple(records))


def write_gap_census(
    path: str | Path,
    census: GapCensus,
    *,
    identity_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": GAP_CENSUS_SCHEMA,
        "version": GAP_CENSUS_VERSION,
        "identitySha256": identity_sha256,
        "provenance": dict(provenance or {}),
        **census.record(),
    }
    atomic_json(path, payload)
    return payload
