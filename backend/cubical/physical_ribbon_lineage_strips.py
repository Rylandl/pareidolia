from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _write_npz
from .physical_ribbon_complete_strip_replay import (
    _load_complete_strip_artifact,
)
from .physical_ribbon_complete_strips import (
    PhysicalRibbonCompleteStripSettings,
    _screen_complete_strip_variants,
    _strict_surface,
)
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_faces import PhysicalRibbonCorridorFaceSettings
from .physical_ribbon_corridor_one_sided import (
    _load_frontier,
    _load_stage_inputs,
)
from .physical_ribbon_corridor_variants import (
    _corridor_boundary_nodes,
    _corridor_settings_from_manifest,
)
from .physical_ribbon_cumulative_replay import (
    cumulative_original_strip_reference,
    load_cumulative_strip_replay_artifact,
)
from .physical_ribbon_patch_corridors import (
    PhysicalRibbonPatchCorridorSettings,
    _beam_set_packing,
    _objective_for_mask,
    _undirected_adjacency,
    solve_patch_corridor_reconfigurations,
)


PHYSICAL_RIBBON_LINEAGE_STRIPS_SCHEMA = (
    "pareidolia.physical-ribbon-lineage-strips"
)
PHYSICAL_RIBBON_LINEAGE_STRIPS_VERSION = 1
PHYSICAL_RIBBON_LINEAGE_STRIPS_STEM = "physical-ribbon-lineage-strips-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonLineageStripSettings:
    """Search settings for complete strips that preserve sheet lineage."""

    maximum_variants_per_corridor: int = 16
    coverage_priority_variant_count: int = 8
    minimum_variant_patch_coverage: float = 0.45
    minimum_anchor_count_per_arc: int = 1
    minimum_strict_surface_area_retention: float = 0.98
    minimum_preclosure_surface_area_retention: float = 0.95

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("lineage-strip variant count must lie in [1, 16]")
        if not 0 <= self.coverage_priority_variant_count <= (
            self.maximum_variants_per_corridor
        ):
            raise ValueError(
                "coverage-priority variant count must fit the retained budget"
            )
        if not 0.0 < self.minimum_variant_patch_coverage <= 1.0:
            raise ValueError("lineage-strip patch coverage must lie in (0, 1]")
        if self.minimum_anchor_count_per_arc < 1:
            raise ValueError("lineage strips require an anchor on both arcs")
        if not 0.0 < self.minimum_strict_surface_area_retention <= 1.0:
            raise ValueError("strict surface-area retention must lie in (0, 1]")
        if not 0.0 < self.minimum_preclosure_surface_area_retention <= (
            self.minimum_strict_surface_area_retention
        ):
            raise ValueError(
                "preclosure area retention must be positive and no larger "
                "than final area retention"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def complete_strip_settings(self) -> PhysicalRibbonCompleteStripSettings:
        return PhysicalRibbonCompleteStripSettings(
            maximum_variants_per_corridor=self.maximum_variants_per_corridor,
            minimum_variant_patch_coverage=self.minimum_variant_patch_coverage,
            minimum_anchor_count_per_arc=self.minimum_anchor_count_per_arc,
            minimum_strict_surface_area_retention=(
                self.minimum_strict_surface_area_retention
            ),
            minimum_preclosure_surface_area_retention=(
                self.minimum_preclosure_surface_area_retention
            ),
        )


def _select_lineage_variant_candidates(
    candidates: Sequence[Mapping[str, Any]],
    *,
    maximum_count: int,
    coverage_priority_count: int,
) -> list[dict[str, Any]]:
    """Retain both factor-optimal and whole-strip-supported assignments.

    The factor beam is ordered by its local unary/pair objective.  That is a
    useful prior, but it can repeatedly retain sparse alternatives that touch
    the two boundary arcs without populating the intervening strip.  Reserve a
    fixed part of the exact-screen budget for complete assignments ranked by
    observed-patch coverage, anchor support, and boundary retention.  The
    subsequent native-CT surface screen remains the acceptance decision.
    """

    if maximum_count < 1:
        raise ValueError("variant selection requires a positive budget")
    if not 0 <= coverage_priority_count <= maximum_count:
        raise ValueError("coverage-priority budget must fit the total budget")
    ordered = [dict(value) for value in candidates]
    factor_count = maximum_count - coverage_priority_count
    selected: list[dict[str, Any]] = []
    selected_beam_ranks: set[int] = set()

    def retain(value: Mapping[str, Any], selection_class: int) -> None:
        beam_rank = int(value["beamRank"])
        if beam_rank in selected_beam_ranks or len(selected) >= maximum_count:
            return
        record = dict(value)
        record["selectionClass"] = selection_class
        selected.append(record)
        selected_beam_ranks.add(beam_rank)

    for value in ordered[:factor_count]:
        retain(value, 0)
    coverage_order = sorted(
        ordered,
        key=lambda value: (
            -float(value["coverage"]),
            -min(int(value["firstAnchorCount"]), int(value["secondAnchorCount"])),
            -float(value["retainedBoundaryFraction"]),
            -float(value["objective"]),
            int(value["beamRank"]),
        ),
    )
    coverage_target = min(maximum_count, factor_count + coverage_priority_count)
    for value in coverage_order:
        if len(selected) >= coverage_target:
            break
        retain(value, 1)
    for value in ordered:
        if len(selected) >= maximum_count:
            break
        retain(value, 0)
    return selected


def _load_complete_strip_replay_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    return load_cumulative_strip_replay_artifact(root)


def _lineage_target_rows(
    strip_manifest: Mapping[str, Any],
    already_replayed_rows: Sequence[int],
) -> np.ndarray:
    """Select unresolved rows whose previous beam included a lineage split."""

    replayed = {int(value) for value in already_replayed_rows}
    rows = []
    for record in strip_manifest["screen"]["rows"]:
        row = int(record["corridorRow"])
        if row in replayed or int(record["eligibleVariantCount"]) > 0:
            continue
        if int(record["componentPreservingVariantCount"]) < int(
            record["variantCount"]
        ):
            rows.append(row)
    return np.asarray(sorted(rows), dtype=np.int32)


def _selected_component_sizes(
    selected_nodes: set[int],
    edge_offset: np.ndarray,
    edge_neighbor: np.ndarray,
) -> tuple[int, tuple[int, ...]]:
    """Return connected-component sizes in an induced selected-node graph."""

    remaining = set(int(value) for value in selected_nodes)
    sizes: list[int] = []
    while remaining:
        seed = remaining.pop()
        stack = [seed]
        size = 0
        while stack:
            node = stack.pop()
            size += 1
            start, stop = int(edge_offset[node]), int(edge_offset[node + 1])
            for neighbor_value in edge_neighbor[start:stop]:
                neighbor = int(neighbor_value)
                if neighbor in remaining:
                    remaining.remove(neighbor)
                    stack.append(neighbor)
        sizes.append(size)
    sizes.sort(reverse=True)
    return len(sizes), tuple(sizes)


def _lineage_preserved(
    baseline_lineage_nodes: set[int],
    added: Sequence[int],
    removed: Sequence[int],
    edge_offset: np.ndarray,
    edge_neighbor: np.ndarray,
) -> tuple[bool, tuple[int, ...]]:
    selected_nodes = set(baseline_lineage_nodes)
    selected_nodes.difference_update(int(value) for value in removed)
    selected_nodes.update(int(value) for value in added)
    component_count, sizes = _selected_component_sizes(
        selected_nodes, edge_offset, edge_neighbor
    )
    return component_count == 1, sizes


def _affected_lineages_preserved(
    lineage_nodes_by_component: Mapping[int, set[int]],
    baseline_component: np.ndarray,
    target_component: int,
    added: Sequence[int],
    removed: Sequence[int],
    edge_offset: np.ndarray,
    edge_neighbor: np.ndarray,
) -> tuple[bool, dict[str, Any]]:
    """Preserve every inherited component touched by a local assignment."""

    affected = {target_component}
    affected.update(
        int(baseline_component[int(value)])
        for value in removed
        if int(baseline_component[int(value)]) >= 0
    )
    removed_set = {int(value) for value in removed}
    deleted: list[int] = []
    split: list[int] = []
    sizes_by_component: dict[int, tuple[int, ...]] = {}
    for component_id in sorted(affected):
        nodes = set(lineage_nodes_by_component[component_id])
        nodes.difference_update(removed_set)
        if component_id == target_component:
            nodes.update(int(value) for value in added)
        count, sizes = _selected_component_sizes(
            nodes, edge_offset, edge_neighbor
        )
        sizes_by_component[component_id] = sizes
        if count == 0:
            deleted.append(component_id)
        elif count > 1:
            split.append(component_id)
    return not deleted and not split, {
        "affectedComponents": tuple(sorted(affected)),
        "deletedComponents": tuple(deleted),
        "splitComponents": tuple(split),
        "componentSizes": sizes_by_component,
    }


def enumerate_lineage_preserving_strip_variants(
    target_rows: Sequence[int],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    reconfiguration: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    corridor_settings: PhysicalRibbonPatchCorridorSettings,
    settings: PhysicalRibbonLineageStripSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Scan complete local factor states under a whole-lineage constraint."""

    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    node_count = len(frontier)
    source_interface = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[
        frontier
    ]
    target_interface = np.asarray(ribbon["targetInterface"], dtype=np.int32)[
        frontier
    ]
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    edge_offset, edge_neighbor, edge_neighbor_score = _undirected_adjacency(
        node_count, edge_first, edge_second, edge_score
    )
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_offset, crossing_neighbor, _ = _undirected_adjacency(
        node_count, crossing_first, crossing_second
    )
    option_offset = np.asarray(
        reconfiguration["corridorOptionOffset"], dtype=np.int64
    )
    option_value = np.asarray(
        reconfiguration["corridorOptionFrontierIndex"], dtype=np.int32
    )
    option_selected = np.asarray(
        reconfiguration["corridorOptionWasSelected"], dtype=np.uint8
    )
    option_candidate = np.asarray(
        reconfiguration["corridorOptionIsPatchCandidate"], dtype=np.uint8
    )
    option_weight = np.asarray(
        reconfiguration["corridorOptionNodeWeight"], dtype=np.float32
    )
    candidate_offset = np.asarray(
        reconfiguration["corridorCandidateOffset"], dtype=np.int64
    )
    candidate_value = np.asarray(
        reconfiguration["corridorCandidateFrontierIndex"], dtype=np.int32
    )
    candidate_nearest = np.asarray(
        reconfiguration["corridorCandidateNearestPatchPixel"], dtype=np.int32
    )
    evidence_eligible = np.asarray(
        reconfiguration["corridorEvidenceEligible"], dtype=np.uint8
    ) > 0
    patch_offset = np.asarray(scored["corridorPatchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(scored["corridorPatchXYZ"], dtype=np.float32)
    boundary_edge_length = np.asarray(
        corridors["boundaryEdgeLengthVoxels"], dtype=np.float32
    )
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    corridor_component = np.asarray(
        corridors["corridorTopologyComponent"], dtype=np.int32
    )
    selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    lineage_nodes_by_component = {
        int(component_id): set(
            int(value)
            for value in np.flatnonzero(
                selected & (component == int(component_id))
            )
        )
        for component_id in np.unique(component[selected])
        if component_id >= 0
    }
    target_set = {int(value) for value in target_rows}
    corridor_count = len(evidence_eligible)

    variant_offset = [0]
    variant_row: list[int] = []
    variant_rank: list[int] = []
    beam_rank_value: list[int] = []
    added_offset = [0]
    added_value: list[int] = []
    removed_offset = [0]
    removed_value: list[int] = []
    objective: list[float] = []
    objective_delta: list[float] = []
    coverage_value: list[float] = []
    first_anchor_value: list[int] = []
    second_anchor_value: list[int] = []
    retained_boundary_value: list[float] = []
    selection_class_value: list[int] = []
    baseline_lineage_count_value: list[int] = []
    retained_lineage_count_value: list[int] = []
    affected_lineage_count_value: list[int] = []
    state_count_value = np.zeros(corridor_count, dtype=np.int32)
    row_stats: list[dict[str, Any]] = []

    for row in range(corridor_count):
        if row not in target_set or not evidence_eligible[row]:
            variant_offset.append(len(variant_row))
            continue
        option_start, option_stop = int(option_offset[row]), int(
            option_offset[row + 1]
        )
        options = option_value[option_start:option_stop]
        weights = option_weight[option_start:option_stop]
        was_selected = option_selected[option_start:option_stop] > 0
        is_candidate = option_candidate[option_start:option_stop] > 0
        local_index = {int(value): index for index, value in enumerate(options)}
        conflict_mask = [0 for _ in options]
        interface_group: dict[int, list[int]] = defaultdict(list)
        for option_index, option_value_i in enumerate(options):
            interface_group[int(source_interface[option_value_i])].append(
                option_index
            )
            interface_group[int(target_interface[option_value_i])].append(
                option_index
            )
        for group in interface_group.values():
            for first_index, left in enumerate(group):
                for right in group[first_index + 1 :]:
                    conflict_mask[left] |= 1 << right
                    conflict_mask[right] |= 1 << left
        for option_index, option_value_i in enumerate(options):
            start, stop = int(crossing_offset[option_value_i]), int(
                crossing_offset[option_value_i + 1]
            )
            for neighbor in crossing_neighbor[start:stop]:
                local_neighbor = local_index.get(int(neighbor))
                if local_neighbor is not None:
                    conflict_mask[option_index] |= 1 << local_neighbor

        pair_support: dict[tuple[int, int], float] = defaultdict(float)
        for option_index, option_value_i in enumerate(options):
            start, stop = int(edge_offset[option_value_i]), int(
                edge_offset[option_value_i + 1]
            )
            for neighbor, score in zip(
                edge_neighbor[start:stop], edge_neighbor_score[start:stop]
            ):
                neighbor_index = local_index.get(int(neighbor))
                if neighbor_index is not None and option_index < neighbor_index:
                    pair_support[(option_index, neighbor_index)] += (
                        continuity_weight * float(score)
                    )
        pair_items = sorted(pair_support.items())
        pair_first = np.asarray(
            [value[0][0] for value in pair_items], dtype=np.int32
        )
        pair_second = np.asarray(
            [value[0][1] for value in pair_items], dtype=np.int32
        )
        pair_weight = np.asarray(
            [value[1] for value in pair_items], dtype=np.float32
        )
        baseline_mask = 0
        for option_index in np.flatnonzero(was_selected):
            baseline_mask |= 1 << int(option_index)
        baseline_objective = _objective_for_mask(
            baseline_mask, weights, pair_first, pair_second, pair_weight
        )
        states, beam_stats = _beam_set_packing(
            weights,
            conflict_mask,
            pair_first,
            pair_second,
            pair_weight,
            baseline_mask,
            beam_width=corridor_settings.configuration_beam_width,
        )
        state_count_value[row] = len(states)

        first_boundary, second_boundary, boundary_edge = _corridor_boundary_nodes(
            row, corridors, scored
        )
        candidate_start, candidate_stop = int(candidate_offset[row]), int(
            candidate_offset[row + 1]
        )
        candidates = candidate_value[candidate_start:candidate_stop]
        nearest = candidate_nearest[candidate_start:candidate_stop]
        candidate_lookup = {
            int(value): index for index, value in enumerate(candidates)
        }
        patch_start, patch_stop = int(patch_offset[row]), int(
            patch_offset[row + 1]
        )
        patch = patch_xyz[patch_start:patch_stop]
        coverage_radius = float(np.median(boundary_edge_length[boundary_edge]))
        component_id = int(corridor_component[int(scored_corridor[row])])
        baseline_lineage_nodes = set(
            int(value)
            for value in np.flatnonzero(selected & (component == component_id))
        )
        if not baseline_lineage_nodes:
            raise ValueError(f"corridor row {row} has no inherited lineage nodes")

        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        rejected_no_addition = 0
        rejected_duplicate = 0
        rejected_coverage = 0
        rejected_anchor = 0
        rejected_lineage = 0
        rejected_deleted_lineage = 0
        rejected_split_lineage = 0
        scanned = 0
        valid_candidates: list[dict[str, Any]] = []
        deepest_beam_rank = -1
        for beam_rank, (state_objective, state_mask) in enumerate(states):
            scanned += 1
            deepest_beam_rank = beam_rank
            proposed = {
                int(options[index])
                for index in range(len(options))
                if state_mask & (1 << index)
            }
            added = tuple(
                sorted(
                    int(options[index])
                    for index in np.flatnonzero(is_candidate)
                    if int(options[index]) in proposed
                )
            )
            if not added:
                rejected_no_addition += 1
                continue
            removed = tuple(
                sorted(
                    int(options[index])
                    for index in np.flatnonzero(was_selected)
                    if int(options[index]) not in proposed
                )
            )
            key = (added, removed)
            if key in seen:
                rejected_duplicate += 1
                continue
            seen.add(key)
            projected = patch[
                np.asarray(
                    [nearest[candidate_lookup[value]] for value in added],
                    dtype=np.int32,
                )
            ]
            coverage = float(
                np.mean(
                    np.any(
                        np.linalg.norm(
                            patch[:, None, :] - projected[None, :, :], axis=2
                        )
                        <= coverage_radius,
                        axis=1,
                    )
                )
            )
            if coverage < settings.minimum_variant_patch_coverage:
                rejected_coverage += 1
                continue
            first_anchor: set[int] = set()
            second_anchor: set[int] = set()
            for value in added:
                start, stop = int(edge_offset[value]), int(edge_offset[value + 1])
                neighbor = set(int(item) for item in edge_neighbor[start:stop])
                first_anchor.update(neighbor & first_boundary)
                second_anchor.update(neighbor & second_boundary)
            if (
                len(first_anchor) < settings.minimum_anchor_count_per_arc
                or len(second_anchor) < settings.minimum_anchor_count_per_arc
            ):
                rejected_anchor += 1
                continue
            preserves_lineage, lineage_audit = _affected_lineages_preserved(
                lineage_nodes_by_component,
                component,
                component_id,
                added,
                removed,
                edge_offset,
                edge_neighbor,
            )
            if not preserves_lineage:
                rejected_lineage += 1
                rejected_deleted_lineage += int(
                    bool(lineage_audit["deletedComponents"])
                )
                rejected_split_lineage += int(
                    bool(lineage_audit["splitComponents"])
                )
                continue

            retained_boundary_fraction = (
                sum(value in proposed for value in first_boundary | second_boundary)
                / max(len(first_boundary | second_boundary), 1)
            )
            target_sizes = lineage_audit["componentSizes"][component_id]
            valid_candidates.append(
                {
                    "beamRank": beam_rank,
                    "added": added,
                    "removed": removed,
                    "objective": float(state_objective),
                    "objectiveDelta": float(
                        state_objective - baseline_objective
                    ),
                    "coverage": coverage,
                    "firstAnchorCount": len(first_anchor),
                    "secondAnchorCount": len(second_anchor),
                    "retainedBoundaryFraction": retained_boundary_fraction,
                    "retainedLineageCount": int(target_sizes[0]),
                    "affectedLineageCount": len(
                        lineage_audit["affectedComponents"]
                    ),
                }
            )

        retained_candidates = _select_lineage_variant_candidates(
            valid_candidates,
            maximum_count=settings.maximum_variants_per_corridor,
            coverage_priority_count=settings.coverage_priority_variant_count,
        )
        for retained_rank, candidate in enumerate(retained_candidates):
            variant_row.append(row)
            variant_rank.append(retained_rank)
            beam_rank_value.append(int(candidate["beamRank"]))
            added = tuple(int(value) for value in candidate["added"])
            removed = tuple(int(value) for value in candidate["removed"])
            added_value.extend(added)
            added_offset.append(len(added_value))
            removed_value.extend(removed)
            removed_offset.append(len(removed_value))
            objective.append(float(candidate["objective"]))
            objective_delta.append(float(candidate["objectiveDelta"]))
            coverage_value.append(float(candidate["coverage"]))
            first_anchor_value.append(int(candidate["firstAnchorCount"]))
            second_anchor_value.append(int(candidate["secondAnchorCount"]))
            retained_boundary_value.append(
                float(candidate["retainedBoundaryFraction"])
            )
            selection_class_value.append(int(candidate["selectionClass"]))
            baseline_lineage_count_value.append(len(baseline_lineage_nodes))
            retained_lineage_count_value.append(
                int(candidate["retainedLineageCount"])
            )
            affected_lineage_count_value.append(
                int(candidate["affectedLineageCount"])
            )

        row_stats.append(
            {
                "corridorRow": row,
                "componentId": component_id,
                "baselineLineageRibbonCount": len(baseline_lineage_nodes),
                "optionCount": len(options),
                "beamStateCount": len(states),
                "scannedStateCount": scanned,
                "deepestScannedBeamRank": deepest_beam_rank,
                "rejectedNoAdditionCount": rejected_no_addition,
                "rejectedDuplicateCount": rejected_duplicate,
                "rejectedCoverageCount": rejected_coverage,
                "rejectedAnchorCount": rejected_anchor,
                "rejectedLineageSplitCount": rejected_lineage,
                "rejectedDeletedLineageCount": rejected_deleted_lineage,
                "rejectedSplitAffectedLineageCount": rejected_split_lineage,
                "validVariantCount": len(valid_candidates),
                "maximumValidPatchCoverage": (
                    round(
                        max(
                            float(value["coverage"])
                            for value in valid_candidates
                        ),
                        6,
                    )
                    if valid_candidates
                    else None
                ),
                "acceptedVariantCount": len(retained_candidates),
                "acceptedBeamRanks": [
                    int(value["beamRank"]) for value in retained_candidates
                ],
                "coveragePriorityVariantCount": sum(
                    int(value["selectionClass"] == 1)
                    for value in retained_candidates
                ),
                "acceptedPatchCoverage": [
                    round(float(value["coverage"]), 6)
                    for value in retained_candidates
                ],
                "acceptedSelectionClass": [
                    "coverage-priority"
                    if int(value["selectionClass"]) == 1
                    else "factor-priority"
                    for value in retained_candidates
                ],
                "beam": beam_stats,
            }
        )
        variant_offset.append(len(variant_row))
        if progress is not None:
            progress(
                f"lineage strips {len(row_stats)}/{len(target_set)} · row {row} · "
                f"retained {len(retained_candidates)}/{len(valid_candidates)} · "
                f"scanned {scanned}/{len(states)}"
            )

    arrays = {
        "corridorVariantOffset": np.asarray(variant_offset, dtype=np.int64),
        "corridorVariantRow": np.asarray(variant_row, dtype=np.int32),
        "corridorVariantRank": np.asarray(variant_rank, dtype=np.int16),
        "corridorVariantBeamRank": np.asarray(beam_rank_value, dtype=np.int32),
        "corridorVariantAddedOffset": np.asarray(added_offset, dtype=np.int64),
        "corridorVariantAddedFrontierIndex": np.asarray(
            added_value, dtype=np.int32
        ),
        "corridorVariantRemovedOffset": np.asarray(removed_offset, dtype=np.int64),
        "corridorVariantRemovedFrontierIndex": np.asarray(
            removed_value, dtype=np.int32
        ),
        "corridorVariantLocalObjective": np.asarray(objective, dtype=np.float32),
        "corridorVariantLocalObjectiveDelta": np.asarray(
            objective_delta, dtype=np.float32
        ),
        "corridorVariantPatchCoverage": np.asarray(
            coverage_value, dtype=np.float32
        ),
        "corridorVariantFirstArcAnchorCount": np.asarray(
            first_anchor_value, dtype=np.int16
        ),
        "corridorVariantSecondArcAnchorCount": np.asarray(
            second_anchor_value, dtype=np.int16
        ),
        "corridorVariantRetainedBoundaryFraction": np.asarray(
            retained_boundary_value, dtype=np.float32
        ),
        "corridorVariantSelectionClass": np.asarray(
            selection_class_value, dtype=np.uint8
        ),
        "corridorVariantBeamStateCount": state_count_value,
        "corridorVariantBaselineLineageRibbonCount": np.asarray(
            baseline_lineage_count_value, dtype=np.int32
        ),
        "corridorVariantRetainedLineageRibbonCount": np.asarray(
            retained_lineage_count_value, dtype=np.int32
        ),
        "corridorVariantAffectedLineageCount": np.asarray(
            affected_lineage_count_value, dtype=np.int16
        ),
    }
    return arrays, {
        "corridorCount": corridor_count,
        "targetCorridorCount": len(target_set),
        "targetCorridorRows": sorted(target_set),
        "enumeratedVariantCount": len(variant_row),
        "corridorsWithVariants": int(np.count_nonzero(np.diff(variant_offset))),
        "beamStateCount": int(np.sum(state_count_value)),
        "rows": row_stats,
        "variantDecision": (
            "complete conflict-free both-arc matching with patch coverage and "
            "one connected induced graph for every inherited sheet lineage "
            "touched by the assignment; the exact-screen budget retains both "
            "factor-priority and whole-strip-coverage-priority states"
        ),
        "singleCellGrowth": False,
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_lineage_strips(
    replay_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonLineageStripSettings | None = None,
    target_rows: Sequence[int] | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonLineageStripSettings()
    replay_path, replay_manifest, replay = _load_complete_strip_replay_artifact(
        replay_root
    )
    strip_reference = cumulative_original_strip_reference(replay_manifest)
    strip_path, strip_manifest, _ = _load_complete_strip_artifact(
        strip_reference["manifestPath"]
    )
    if (
        sha256_file(strip_path) != strip_reference["manifestSha256"]
        or strip_manifest["data"]["sha256"] != strip_reference["dataSha256"]
    ):
        raise ValueError("lineage-strip source audit has changed")
    frontier_path, frontier_manifest, topology = _load_frontier(
        replay_manifest["identity"]["frontier"]["manifestPath"]
    )
    (
        corridor_path,
        corridor_manifest,
        corridor,
        _,
        _,
        _,
        configuration_path,
        configuration_manifest,
        base_configuration,
        ribbon,
    ) = _load_stage_inputs(frontier_manifest)
    face_settings = PhysicalRibbonCorridorFaceSettings(
        **replay_manifest["identity"]["faceSettings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_LINEAGE_STRIPS_SCHEMA,
        "version": PHYSICAL_RIBBON_LINEAGE_STRIPS_VERSION,
        "replay": {
            "manifestPath": str(replay_path),
            "manifestSha256": sha256_file(replay_path),
            "dataSha256": replay_manifest["data"]["sha256"],
        },
        "priorStrips": strip_reference,
        "frontier": {
            "manifestPath": str(frontier_path),
            "manifestSha256": sha256_file(frontier_path),
            "dataSha256": frontier_manifest["data"]["sha256"],
        },
        "corridors": {
            "manifestPath": str(corridor_path),
            "manifestSha256": sha256_file(corridor_path),
            "dataSha256": corridor_manifest["data"]["sha256"],
        },
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "targetRows": (
            [int(value) for value in sorted(set(target_rows))]
            if target_rows is not None
            else None
        ),
        "faceSettings": face_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "screenImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_complete_strips.py")
        ),
        "faceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_faces.py")
        ),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_LINEAGE_STRIPS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_LINEAGE_STRIPS_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    baseline_surface = _strict_surface(replay)
    configuration = dict(topology)
    configuration["selected"] = np.asarray(replay["selected"], dtype=np.uint8)
    configuration["component"] = np.asarray(replay["component"], dtype=np.int32)
    remapped = _remap_corridor_surface(
        corridor,
        baseline_surface,
        base_configuration,
        configuration,
        np.asarray(topology["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    resolved_target_rows = (
        np.asarray(sorted(set(int(value) for value in target_rows)), dtype=np.int32)
        if target_rows is not None
        else _lineage_target_rows(
            strip_manifest,
            replay_manifest["optimization"]["chosenCorridorRows"],
        )
    )
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    continuity_weight = float(
        configuration_manifest["identity"]["settings"]["continuity_weight"]
    )
    if progress is not None:
        progress(
            f"solving lineage factor graphs for rows {resolved_target_rows.tolist()}"
        )
    reconfiguration, reconfiguration_stats = solve_patch_corridor_reconfigurations(
        remapped,
        remapped,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        settings=corridor_settings,
    )
    reconfigured_at = time.monotonic()
    if progress is not None:
        progress("scanning complete states under whole-lineage connectivity")
    variants, enumeration_stats = enumerate_lineage_preserving_strip_variants(
        resolved_target_rows,
        remapped,
        remapped,
        reconfiguration,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        corridor_settings=corridor_settings,
        settings=resolved,
        progress=progress,
    )
    enumerated_at = time.monotonic()
    if progress is not None:
        progress(
            f"screening {len(variants['corridorVariantRow'])} lineage-preserving strips against native CT"
        )
    screen, _, screen_stats = _screen_complete_strip_variants(
        resolved_target_rows,
        variants,
        remapped,
        ribbon,
        topology,
        configuration,
        baseline_surface,
        surface_settings=corridor_settings.surface_settings(),
        face_settings=face_settings,
        settings=resolved.complete_strip_settings(),
        progress=progress,
    )
    screened_at = time.monotonic()
    arrays = {
        **variants,
        **screen,
        "targetCorridorRow": resolved_target_rows,
    }
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_LINEAGE_STRIPS_SCHEMA,
        "version": PHYSICAL_RIBBON_LINEAGE_STRIPS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "target": {
            "lineageCorridorCount": len(resolved_target_rows),
            "lineageCorridorRows": [int(value) for value in resolved_target_rows],
            "lineagePreservingVariantCount": len(
                variants["corridorVariantRow"]
            ),
        },
        "reconfiguration": reconfiguration_stats,
        "enumeration": enumeration_stats,
        "screen": screen_stats,
        "timingSeconds": {
            "factorGraphs": round(reconfigured_at - started, 6),
            "enumeration": round(enumerated_at - reconfigured_at, 6),
            "physicalScreen": round(screened_at - enumerated_at, 6),
            "writing": round(finished - screened_at, 6),
            "total": round(finished - started, 6),
        },
        "method": {
            "decisionUnit": "complete both-arc native-CT strip matching",
            "lineageConstraint": (
                "all retained ribbons and proposed additions for the inherited "
                "strict component must remain one connected topology graph"
            ),
            "surface": (
                "lineage-preserving states still require strict connectivity "
                "or an attached native-CT-gated face path"
            ),
            "selectionMutated": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
