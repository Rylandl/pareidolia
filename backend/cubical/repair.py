from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

from .block import BlockBounds, SurfaceBlock, assemble_surface_hierarchy
from .contracts import atomic_json
from .gaps import GapCensus, GapTraceRecord
from .selection import ConfigurationOption
from .topology import Int3


GAP_REPAIR_SEARCH_SCHEMA = "pareidolia.raw-acus-gap-repair-search"
GAP_REPAIR_SEARCH_VERSION = 1


@dataclass(frozen=True, slots=True)
class GapRepairTrial:
    target_cell_xyz: Int3
    current_option_id: int
    candidate_option_id: int
    source_patch_ids: tuple[int, ...]
    target_patch_ids: tuple[int, ...]
    log_weight_penalty: float
    layer_count_delta: int
    closed_gap_count: int
    source_component_size_before: int
    source_component_size_after: int
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
            "targetCellXYZ": list(self.target_cell_xyz),
            "currentOptionId": self.current_option_id,
            "candidateOptionId": self.candidate_option_id,
            "sourcePatchIds": list(self.source_patch_ids),
            "targetPatchIds": list(self.target_patch_ids),
            "logWeightPenalty": self.log_weight_penalty,
            "layerCountDelta": self.layer_count_delta,
            "closedGapCount": self.closed_gap_count,
            "sourceComponentSizeBefore": self.source_component_size_before,
            "sourceComponentSizeAfter": self.source_component_size_after,
            "sourceComponentSizeDelta": (
                self.source_component_size_after
                - self.source_component_size_before
            ),
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
class GapRepairSearch:
    component_id: int
    trials: tuple[GapRepairTrial, ...]

    def record(self) -> dict[str, Any]:
        return {
            "componentId": self.component_id,
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
class GapRepairApplication:
    selected_options: tuple[ConfigurationOption, ...]
    block: SurfaceBlock
    applied_trials: tuple[GapRepairTrial, ...]
    verified_closed_gap_count: int


def _option_lookup(
    options_by_cell: Mapping[Int3, tuple[ConfigurationOption, ...]],
) -> dict[int, ConfigurationOption]:
    result: dict[int, ConfigurationOption] = {}
    for options in options_by_cell.values():
        for option in options:
            if option.option_id in result:
                raise ValueError(f"duplicate configuration option id {option.option_id}")
            result[option.option_id] = option
    return result


def _repair_proposals(
    census: GapCensus,
) -> dict[tuple[Int3, int], list[tuple[GapTraceRecord, int]]]:
    proposals: dict[
        tuple[Int3, int], list[tuple[GapTraceRecord, int]]
    ] = defaultdict(list)
    for trace in census.traces:
        if trace.classification != "recoverable-configuration-gap":
            continue
        for alternative in trace.alternatives:
            proposals[(trace.target_cell_xyz, alternative.option_id)].append(
                (trace, alternative.target_patch_id)
            )
    return proposals


def evaluate_single_cell_gap_repairs(
    baseline: SurfaceBlock,
    options_by_cell: Mapping[Int3, tuple[ConfigurationOption, ...]],
    selected_option_ids: Mapping[Int3, int],
    census: GapCensus,
    *,
    maximum_leaf_shape_cells_xyz: Int3 = (4, 4, 3),
    progress: Callable[[int, int, Int3, int], None] | None = None,
) -> GapRepairSearch:
    """Try each retained target configuration with all other cells frozen."""

    lookup = _option_lookup(options_by_cell)
    selected = {
        cell: lookup[option_id] for cell, option_id in selected_option_ids.items()
    }
    if set(selected) != set(options_by_cell):
        raise ValueError("selected options do not cover the configuration grid")
    proposals = _repair_proposals(census)
    component_by_patch = dict(baseline.component_by_patch)
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in baseline.components
    }
    baseline_deferred = Counter(value.reason for value in baseline.deferred_joins)
    baseline_patch_count = len(baseline.patches)
    bounds = BlockBounds((0, 0, 0), baseline.grid.shape_cells_xyz)
    values: list[GapRepairTrial] = []
    ordered = sorted(
        proposals.items(),
        key=lambda value: (
            value[0][0][2],
            value[0][0][1],
            value[0][0][0],
            value[0][1],
        ),
    )
    for index, ((cell, candidate_id), gaps) in enumerate(ordered, start=1):
        if progress is not None:
            progress(index, len(ordered), cell, candidate_id)
        current = selected[cell]
        candidate = lookup[candidate_id]
        trial_selected = dict(selected)
        trial_selected[cell] = candidate
        patches = tuple(
            patch
            for selected_cell in sorted(
                trial_selected, key=lambda value: (value[2], value[1], value[0])
            )
            for patch in trial_selected[selected_cell].patches
        )
        trial = assemble_surface_hierarchy(
            baseline.grid,
            bounds,
            patches,
            maximum_leaf_shape_cells_xyz=maximum_leaf_shape_cells_xyz,
        )
        trial_component_by_patch = dict(trial.component_by_patch)
        trial_component_sizes = {
            value.component_id: len(value.patch_ids) for value in trial.components
        }
        trial_join_keys = {
            (
                min(value.first_patch_id, value.second_patch_id),
                max(value.first_patch_id, value.second_patch_id),
                value.face,
            )
            for value in trial.joins
        }
        closed = 0
        source_sizes_before: list[int] = []
        source_sizes_after: list[int] = []
        source_patch_ids: list[int] = []
        target_patch_ids: list[int] = []
        for gap, target_patch_id in gaps:
            source_patch_ids.append(gap.patch_id)
            target_patch_ids.append(target_patch_id)
            source_component = component_by_patch[gap.patch_id]
            source_sizes_before.append(component_sizes[source_component])
            if gap.patch_id in trial_component_by_patch:
                resolved_component = trial_component_by_patch[gap.patch_id]
                source_sizes_after.append(trial_component_sizes[resolved_component])
            else:
                source_sizes_after.append(0)
            key = (
                min(gap.patch_id, target_patch_id),
                max(gap.patch_id, target_patch_id),
                gap.face,
            )
            if key in trial_join_keys:
                closed += 1
        deferred = Counter(value.reason for value in trial.deferred_joins)
        retained_delta = len(trial.joins) - len(baseline.joins)
        unresolved_delta = len(trial.unresolved_interior_traces) - len(
            baseline.unresolved_interior_traces
        )
        layer_delta = len(candidate.patches) - len(current.patches)
        size_before = max(source_sizes_before, default=0)
        size_after = max(source_sizes_after, default=0)
        collision_delta = deferred["component-cell-collision"] - baseline_deferred[
            "component-cell-collision"
        ]
        topology_delta = deferred["crossing-topology-cycle"] - baseline_deferred[
            "crossing-topology-cycle"
        ]
        recommended = (
            closed > 0
            and layer_delta >= 0
            and size_after >= size_before
            and retained_delta >= 0
            and unresolved_delta <= 0
            and collision_delta <= 0
            and topology_delta <= 0
        )
        values.append(
            GapRepairTrial(
                cell,
                current.option_id,
                candidate.option_id,
                tuple(sorted(set(source_patch_ids))),
                tuple(sorted(set(target_patch_ids))),
                current.log_weight - candidate.log_weight,
                layer_delta,
                closed,
                size_before,
                size_after,
                len(trial.patches) - baseline_patch_count,
                len(trial.candidate_joins) - len(baseline.candidate_joins),
                retained_delta,
                len(trial.deferred_joins) - len(baseline.deferred_joins),
                len(trial.components) - len(baseline.components),
                unresolved_delta,
                len(trial.exterior_traces) - len(baseline.exterior_traces),
                collision_delta,
                topology_delta,
                recommended,
            )
        )
    values.sort(
        key=lambda value: (
            not value.recommended,
            -value.closed_gap_count,
            -(
                value.source_component_size_after
                - value.source_component_size_before
            ),
            -value.retained_join_delta,
            value.unresolved_interior_trace_delta,
            value.log_weight_penalty,
            value.target_cell_xyz,
            value.candidate_option_id,
        )
    )
    return GapRepairSearch(census.component_id, tuple(values))


def apply_recommended_gap_repairs(
    baseline: SurfaceBlock,
    options_by_cell: Mapping[Int3, tuple[ConfigurationOption, ...]],
    selected_option_ids: Mapping[Int3, int],
    search: GapRepairSearch,
    *,
    maximum_leaf_shape_cells_xyz: Int3 = (4, 4, 3),
) -> GapRepairApplication:
    """Apply the best nonconflicting conservative trial for each target cell."""

    lookup = _option_lookup(options_by_cell)
    selected = {
        cell: lookup[option_id] for cell, option_id in selected_option_ids.items()
    }
    chosen: list[GapRepairTrial] = []
    occupied_cells: set[Int3] = set()
    for trial in search.trials:
        if not trial.recommended or trial.target_cell_xyz in occupied_cells:
            continue
        selected[trial.target_cell_xyz] = lookup[trial.candidate_option_id]
        chosen.append(trial)
        occupied_cells.add(trial.target_cell_xyz)
    if not chosen:
        raise ValueError("gap repair search contains no recommended trial")
    ordered_options = tuple(
        selected[cell]
        for cell in sorted(selected, key=lambda value: (value[2], value[1], value[0]))
    )
    patches = tuple(
        patch for option in ordered_options for patch in option.patches
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
        frozenset((source, target)) in join_pairs
        for trial in chosen
        for source in trial.source_patch_ids
        for target in trial.target_patch_ids
    )
    if verified < sum(value.closed_gap_count for value in chosen):
        raise RuntimeError("combined repair invalidated an individually verified gap")
    return GapRepairApplication(ordered_options, block, tuple(chosen), verified)


def write_gap_repair_search(
    path: str | Path,
    search: GapRepairSearch,
    *,
    identity_sha256: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    payload = {
        "schema": GAP_REPAIR_SEARCH_SCHEMA,
        "version": GAP_REPAIR_SEARCH_VERSION,
        "identitySha256": identity_sha256,
        "provenance": dict(provenance or {}),
        **search.record(),
    }
    atomic_json(path, payload)
    return payload


def read_gap_repair_search(
    path: str | Path, *, identity_sha256: str
) -> GapRepairSearch:
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema") != GAP_REPAIR_SEARCH_SCHEMA
        or int(payload.get("version", -1)) != GAP_REPAIR_SEARCH_VERSION
        or payload.get("identitySha256") != identity_sha256
    ):
        raise ValueError("gap repair search does not match this reconstruction")
    trials = []
    for value in payload["trials"]:
        trials.append(
            GapRepairTrial(
                tuple(value["targetCellXYZ"]),
                int(value["currentOptionId"]),
                int(value["candidateOptionId"]),
                tuple(int(item) for item in value["sourcePatchIds"]),
                tuple(int(item) for item in value["targetPatchIds"]),
                float(value["logWeightPenalty"]),
                int(value["layerCountDelta"]),
                int(value["closedGapCount"]),
                int(value["sourceComponentSizeBefore"]),
                int(value["sourceComponentSizeAfter"]),
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
        )
    return GapRepairSearch(int(payload["componentId"]), tuple(trials))
