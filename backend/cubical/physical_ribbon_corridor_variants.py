from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_patch_corridors import (
    PHYSICAL_RIBBON_PATCH_CORRIDORS_SCHEMA,
    PHYSICAL_RIBBON_PATCH_CORRIDORS_STEM,
    PhysicalRibbonPatchCorridorSettings,
    _beam_set_packing,
    _evaluate_corridor_connections,
    _objective_for_mask,
    _triangle_region_labels,
    _undirected_adjacency,
    build_physical_ribbon_surface_complex,
    replay_patch_corridor_reconfigurations,
    write_patch_corridor_montage,
    write_replayed_corridor_fragment_montage,
)


PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-variants"
)
PHYSICAL_RIBBON_CORRIDOR_VARIANTS_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM = (
    "physical-ribbon-corridor-variants-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorVariantSettings:
    maximum_variants_per_corridor: int = 4
    minimum_variant_patch_coverage: float = 0.45
    minimum_anchor_count_per_arc: int = 1
    minimum_surface_area_retention: float = 0.98
    maximum_preview_components: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("corridor variant count must lie in [1, 16]")
        if not 0.0 < self.minimum_variant_patch_coverage <= 1.0:
            raise ValueError("variant patch coverage must lie in (0, 1]")
        if self.minimum_anchor_count_per_arc < 1:
            raise ValueError("each variant requires anchors on both arcs")
        if not 0.0 < self.minimum_surface_area_retention <= 1.0:
            raise ValueError("surface-area retention must lie in (0, 1]")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _corridor_boundary_nodes(
    row: int,
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
) -> tuple[set[int], set[int], np.ndarray]:
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    corridor_index = int(scored_corridor[row])
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    start, stop = int(pair_offset[corridor_index]), int(
        pair_offset[corridor_index + 1]
    )
    first_edge = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )[start:stop]
    second_edge = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )[start:stop]
    edge_first = np.asarray(
        corridors["boundaryEdgeFirstFrontierIndex"], dtype=np.int32
    )
    edge_second = np.asarray(
        corridors["boundaryEdgeSecondFrontierIndex"], dtype=np.int32
    )
    first_node = set(
        int(value)
        for value in np.concatenate((edge_first[first_edge], edge_second[first_edge]))
    )
    second_node = set(
        int(value)
        for value in np.concatenate(
            (edge_first[second_edge], edge_second[second_edge])
        )
    )
    return first_node, second_node, np.concatenate((first_edge, second_edge))


def enumerate_corridor_reconfiguration_variants(
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    reconfiguration: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    corridor_settings: PhysicalRibbonPatchCorridorSettings,
    settings: PhysicalRibbonCorridorVariantSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Retain diverse complete factor states instead of one local optimum."""

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

    variant_offset = [0]
    variant_row: list[int] = []
    variant_rank: list[int] = []
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
    state_count_value: list[int] = []
    eligible_state_count = 0
    for row in range(len(evidence_eligible)):
        if not evidence_eligible[row]:
            variant_offset.append(len(variant_row))
            state_count_value.append(0)
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
        for option_index, option in enumerate(options):
            interface_group[int(source_interface[option])].append(option_index)
            interface_group[int(target_interface[option])].append(option_index)
        for group in interface_group.values():
            for first_index, left in enumerate(group):
                for right in group[first_index + 1 :]:
                    conflict_mask[left] |= 1 << right
                    conflict_mask[right] |= 1 << left
        for option_index, option in enumerate(options):
            start, stop = int(crossing_offset[option]), int(
                crossing_offset[option + 1]
            )
            for neighbor in crossing_neighbor[start:stop]:
                local_neighbor = local_index.get(int(neighbor))
                if local_neighbor is not None:
                    conflict_mask[option_index] |= 1 << local_neighbor
        pair_support: dict[tuple[int, int], float] = defaultdict(float)
        for option_index, option in enumerate(options):
            start, stop = int(edge_offset[option]), int(edge_offset[option + 1])
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
        states, _ = _beam_set_packing(
            weights,
            conflict_mask,
            pair_first,
            pair_second,
            pair_weight,
            baseline_mask,
            beam_width=corridor_settings.configuration_beam_width,
        )
        state_count_value.append(len(states))
        first_boundary, second_boundary, boundary_edge = (
            _corridor_boundary_nodes(row, corridors, scored)
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
        seen: set[tuple[tuple[int, ...], tuple[int, ...]]] = set()
        accepted_rank = 0
        for state_objective, state_mask in states:
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
            first_anchor: set[int] = set()
            second_anchor: set[int] = set()
            for value in added:
                start, stop = int(edge_offset[value]), int(edge_offset[value + 1])
                neighbor = set(int(item) for item in edge_neighbor[start:stop])
                first_anchor.update(neighbor & first_boundary)
                second_anchor.update(neighbor & second_boundary)
            if (
                coverage < settings.minimum_variant_patch_coverage
                or len(first_anchor) < settings.minimum_anchor_count_per_arc
                or len(second_anchor) < settings.minimum_anchor_count_per_arc
            ):
                continue
            variant_row.append(row)
            variant_rank.append(accepted_rank)
            added_value.extend(added)
            added_offset.append(len(added_value))
            removed_value.extend(removed)
            removed_offset.append(len(removed_value))
            objective.append(float(state_objective))
            objective_delta.append(float(state_objective - baseline_objective))
            coverage_value.append(coverage)
            first_anchor_value.append(len(first_anchor))
            second_anchor_value.append(len(second_anchor))
            retained_boundary_value.append(
                sum(value in proposed for value in first_boundary | second_boundary)
                / max(len(first_boundary | second_boundary), 1)
            )
            accepted_rank += 1
            eligible_state_count += 1
            if accepted_rank >= settings.maximum_variants_per_corridor:
                break
        variant_offset.append(len(variant_row))
    arrays = {
        "corridorVariantOffset": np.asarray(variant_offset, dtype=np.int64),
        "corridorVariantRow": np.asarray(variant_row, dtype=np.int32),
        "corridorVariantRank": np.asarray(variant_rank, dtype=np.int16),
        "corridorVariantAddedOffset": np.asarray(added_offset, dtype=np.int64),
        "corridorVariantAddedFrontierIndex": np.asarray(
            added_value, dtype=np.int32
        ),
        "corridorVariantRemovedOffset": np.asarray(
            removed_offset, dtype=np.int64
        ),
        "corridorVariantRemovedFrontierIndex": np.asarray(
            removed_value, dtype=np.int32
        ),
        "corridorVariantLocalObjective": np.asarray(
            objective, dtype=np.float32
        ),
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
        "corridorVariantBeamStateCount": np.asarray(
            state_count_value, dtype=np.int32
        ),
    }
    return arrays, {
        "corridorCount": len(evidence_eligible),
        "evidenceEligibleCorridorCount": int(np.count_nonzero(evidence_eligible)),
        "enumeratedVariantCount": len(variant_row),
        "corridorsWithVariants": int(np.count_nonzero(np.diff(variant_offset))),
        "beamStateCount": int(np.sum(state_count_value)),
        "variantDecision": "complete conflict-free joint matching states with both-arc anchors and patch coverage",
        "identityLabelsUsed": False,
    }


def screen_exact_corridor_variants(
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    variants: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    corridor_settings: PhysicalRibbonPatchCorridorSettings,
    settings: PhysicalRibbonCorridorVariantSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconstruct every retained factor state in its complete source sheet."""

    variant_offset = np.asarray(
        variants["corridorVariantOffset"], dtype=np.int64
    )
    variant_count = int(variant_offset[-1])
    added_offset = np.asarray(
        variants["corridorVariantAddedOffset"], dtype=np.int64
    )
    added_value = np.asarray(
        variants["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        variants["corridorVariantRemovedOffset"], dtype=np.int64
    )
    removed_value = np.asarray(
        variants["corridorVariantRemovedFrontierIndex"], dtype=np.int32
    )
    objective_delta = np.asarray(
        variants["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    coverage = np.asarray(
        variants["corridorVariantPatchCoverage"], dtype=np.float32
    )
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    corridor_component = np.asarray(
        corridors["corridorTopologyComponent"], dtype=np.int32
    )
    baseline_selected = np.asarray(configuration["selected"]) > 0
    original_component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source_interface = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[
        frontier
    ]
    target_interface = np.asarray(ribbon["targetInterface"], dtype=np.int32)[
        frontier
    ]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    baseline_triangle = np.asarray(
        surface["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_area = np.asarray(
        surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    baseline_region = _triangle_region_labels(baseline_triangle)

    exact_connected = np.zeros(variant_count, dtype=np.uint8)
    component_split = np.zeros(variant_count, dtype=np.uint8)
    hard_conflict = np.zeros(variant_count, dtype=np.uint8)
    surface_eligible = np.zeros(variant_count, dtype=np.uint8)
    region_before = np.zeros(variant_count, dtype=np.int32)
    region_after = np.zeros(variant_count, dtype=np.int32)
    triangle_before = np.zeros(variant_count, dtype=np.int32)
    triangle_after = np.zeros(variant_count, dtype=np.int32)
    area_before = np.zeros(variant_count, dtype=np.float32)
    area_after = np.zeros(variant_count, dtype=np.float32)
    shared_region_fraction = np.zeros(variant_count, dtype=np.float32)
    chosen_variant = np.full(len(variant_offset) - 1, -1, dtype=np.int32)
    component_records: list[dict[str, Any]] = []

    def has_global_conflict(selected: np.ndarray) -> bool:
        node = np.flatnonzero(selected)
        interface = np.concatenate(
            (source_interface[node], target_interface[node])
        )
        return bool(
            len(interface) != len(np.unique(interface))
            or np.any(selected[crossing_first] & selected[crossing_second])
        )

    rows = np.flatnonzero(np.diff(variant_offset))
    for completed, row_value in enumerate(rows, start=1):
        row = int(row_value)
        component_id = int(corridor_component[int(scored_corridor[row])])
        base_triangle_index = np.flatnonzero(
            np.all(original_component[baseline_triangle] == component_id, axis=1)
        )
        base_regions = len(np.unique(baseline_region[base_triangle_index]))
        base_area = float(np.sum(baseline_area[base_triangle_index]))
        best_key: tuple[float, ...] = (0.0, 0.0, 0.0, 0.0)
        best_variant = -1
        row_records: list[dict[str, Any]] = []
        for variant_index in range(
            int(variant_offset[row]), int(variant_offset[row + 1])
        ):
            added = added_value[
                int(added_offset[variant_index]) : int(
                    added_offset[variant_index + 1]
                )
            ]
            removed = removed_value[
                int(removed_offset[variant_index]) : int(
                    removed_offset[variant_index + 1]
                )
            ]
            selected = baseline_selected.copy()
            selected[removed] = False
            selected[added] = True
            if has_global_conflict(selected):
                hard_conflict[variant_index] = 1
                continue
            local_selected = selected & (original_component == component_id)
            local_selected[added] = selected[added]
            local_component, _ = _component_labels(local_selected, first, second)
            retained_original = local_selected & (
                original_component == component_id
            )
            labels = np.unique(
                local_component[retained_original][
                    local_component[retained_original] >= 0
                ]
            )
            if len(labels) != 1:
                component_split[variant_index] = 1
                continue
            local_configuration = dict(configuration)
            local_configuration["selected"] = local_selected.astype(np.uint8)
            local_configuration["component"] = local_component
            local_surface, _ = build_physical_ribbon_surface_complex(
                ribbon,
                topology,
                local_configuration,
                settings=corridor_settings.surface_settings(),
            )
            local_triangle = np.asarray(
                local_surface["triangleFrontierIndex"], dtype=np.int32
            )
            local_regions = (
                len(np.unique(_triangle_region_labels(local_triangle)))
                if len(local_triangle)
                else 0
            )
            local_area = float(
                np.sum(local_surface["triangleAreaVoxelsSquared"])
            )
            connections = _evaluate_corridor_connections(
                local_surface,
                corridors,
                scored,
                minimum_arc_region_fraction=(
                    corridor_settings.minimum_replay_arc_region_fraction
                ),
                maximum_arc_triangle_distance_edges=(
                    corridor_settings.maximum_replay_arc_triangle_distance_edges
                ),
            )
            connected = bool(connections["boundaryArcsConnected"][row])
            fraction = float(
                connections["boundaryArcSharedRegionFraction"][row]
            )
            area_retention = local_area / max(base_area, 1.0e-6)
            eligible = (
                connected
                and local_regions <= base_regions
                and area_retention >= settings.minimum_surface_area_retention
            )
            exact_connected[variant_index] = int(connected)
            surface_eligible[variant_index] = int(eligible)
            region_before[variant_index] = base_regions
            region_after[variant_index] = local_regions
            triangle_before[variant_index] = len(base_triangle_index)
            triangle_after[variant_index] = len(local_triangle)
            area_before[variant_index] = base_area
            area_after[variant_index] = local_area
            shared_region_fraction[variant_index] = fraction
            key = (
                float(base_regions - local_regions),
                local_area - base_area,
                float(objective_delta[variant_index]),
                float(coverage[variant_index]),
            )
            if eligible and key > best_key:
                best_key = key
                best_variant = variant_index
            row_records.append(
                {
                    "variantIndex": variant_index,
                    "exactConnected": connected,
                    "sharedArcRegionFraction": round(fraction, 4),
                    "triangleRegions": [base_regions, local_regions],
                    "triangleAreaVoxelsSquared": [
                        round(base_area, 4),
                        round(local_area, 4),
                    ],
                    "localObjectiveDelta": round(
                        float(objective_delta[variant_index]), 6
                    ),
                    "eligible": eligible,
                }
            )
        chosen_variant[row] = best_variant
        component_records.append(
            {
                "corridorRow": row,
                "componentId": component_id,
                "chosenVariantIndex": best_variant,
                "selectionKey": [round(value, 6) for value in best_key],
                "variants": row_records,
            }
        )
        if progress is not None and (
            completed == len(rows) or completed % 5 == 0
        ):
            progress(
                f"exact corridor variants {completed}/{len(rows)} · "
                f"accepted {int(np.count_nonzero(chosen_variant >= 0))}"
            )
    arrays = {
        "corridorVariantExactConnected": exact_connected,
        "corridorVariantComponentSplit": component_split,
        "corridorVariantHardConflict": hard_conflict,
        "corridorVariantSurfaceEligible": surface_eligible,
        "corridorVariantTriangleRegionCountBefore": region_before,
        "corridorVariantTriangleRegionCountAfter": region_after,
        "corridorVariantTriangleCountBefore": triangle_before,
        "corridorVariantTriangleCountAfter": triangle_after,
        "corridorVariantTriangleAreaBefore": area_before,
        "corridorVariantTriangleAreaAfter": area_after,
        "corridorVariantSharedArcRegionFraction": shared_region_fraction,
        "corridorChosenExactVariant": chosen_variant,
    }
    return arrays, {
        "variantCount": variant_count,
        "exactConnectedVariantCount": int(np.count_nonzero(exact_connected)),
        "surfaceEligibleVariantCount": int(np.count_nonzero(surface_eligible)),
        "corridorWithExactVariantCount": int(
            np.count_nonzero(chosen_variant >= 0)
        ),
        "componentSplitVariantCount": int(np.count_nonzero(component_split)),
        "hardConflictVariantCount": int(np.count_nonzero(hard_conflict)),
        "components": component_records,
        "selectionPriority": "triangle-region reduction, supported area gain, local factor objective, patch coverage",
        "identityLabelsUsed": False,
    }


def compile_exact_variant_reconfiguration(
    reconfiguration: Mapping[str, np.ndarray],
    variants: Mapping[str, np.ndarray],
    exact: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Present chosen exact variants through the ordinary replay contract."""

    result = {name: np.asarray(value) for name, value in reconfiguration.items()}
    chosen = np.asarray(exact["corridorChosenExactVariant"], dtype=np.int32)
    variant_added_offset = np.asarray(
        variants["corridorVariantAddedOffset"], dtype=np.int64
    )
    variant_added = np.asarray(
        variants["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    variant_removed_offset = np.asarray(
        variants["corridorVariantRemovedOffset"], dtype=np.int64
    )
    variant_removed = np.asarray(
        variants["corridorVariantRemovedFrontierIndex"], dtype=np.int32
    )
    variant_objective = np.asarray(
        variants["corridorVariantLocalObjective"], dtype=np.float32
    )
    variant_delta = np.asarray(
        variants["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    variant_coverage = np.asarray(
        variants["corridorVariantPatchCoverage"], dtype=np.float32
    )
    variant_first_anchor = np.asarray(
        variants["corridorVariantFirstArcAnchorCount"], dtype=np.int16
    )
    variant_second_anchor = np.asarray(
        variants["corridorVariantSecondArcAnchorCount"], dtype=np.int16
    )
    variant_retained = np.asarray(
        variants["corridorVariantRetainedBoundaryFraction"], dtype=np.float32
    )
    region_before = np.asarray(
        exact["corridorVariantTriangleRegionCountBefore"], dtype=np.int32
    )
    region_after = np.asarray(
        exact["corridorVariantTriangleRegionCountAfter"], dtype=np.int32
    )
    area_before = np.asarray(
        exact["corridorVariantTriangleAreaBefore"], dtype=np.float32
    )
    area_after = np.asarray(
        exact["corridorVariantTriangleAreaAfter"], dtype=np.float32
    )
    added_offset = [0]
    added: list[int] = []
    removed_offset = [0]
    removed: list[int] = []
    proposal_objective = np.zeros(len(chosen), dtype=np.float32)
    proposal_delta = np.zeros(len(chosen), dtype=np.float32)
    proposal_coverage = np.zeros(len(chosen), dtype=np.float32)
    retained_boundary = np.ones(len(chosen), dtype=np.float32)
    anchor_count = np.zeros(len(chosen), dtype=np.int32)
    eligible = np.zeros(len(chosen), dtype=np.uint8)
    for row, variant_index_value in enumerate(chosen):
        variant_index = int(variant_index_value)
        if variant_index >= 0:
            current_added = variant_added[
                int(variant_added_offset[variant_index]) : int(
                    variant_added_offset[variant_index + 1]
                )
            ]
            current_removed = variant_removed[
                int(variant_removed_offset[variant_index]) : int(
                    variant_removed_offset[variant_index + 1]
                )
            ]
            added.extend(int(value) for value in current_added)
            removed.extend(int(value) for value in current_removed)
            proposal_objective[row] = variant_objective[variant_index]
            region_gain = int(region_before[variant_index] - region_after[variant_index])
            relative_area_gain = float(
                (area_after[variant_index] - area_before[variant_index])
                / max(float(area_before[variant_index]), 1.0e-6)
            )
            proposal_delta[row] = float(
                100.0 * region_gain
                + max(relative_area_gain, 0.0)
                + max(float(variant_delta[variant_index]), 0.0)
                + 1.0e-3
            )
            proposal_coverage[row] = variant_coverage[variant_index]
            retained_boundary[row] = variant_retained[variant_index]
            anchor_count[row] = int(
                variant_first_anchor[variant_index]
                + variant_second_anchor[variant_index]
            )
            eligible[row] = 1
        added_offset.append(len(added))
        removed_offset.append(len(removed))
    result["corridorEvidenceEligible"] = eligible
    result["corridorProposalAddedOffset"] = np.asarray(
        added_offset, dtype=np.int64
    )
    result["corridorProposalAddedFrontierIndex"] = np.asarray(
        added, dtype=np.int32
    )
    result["corridorProposalRemovedOffset"] = np.asarray(
        removed_offset, dtype=np.int64
    )
    result["corridorProposalRemovedFrontierIndex"] = np.asarray(
        removed, dtype=np.int32
    )
    result["corridorProposalLocalObjective"] = proposal_objective
    result["corridorProposalObjectiveDelta"] = proposal_delta
    result["corridorProposalPatchCoverage"] = proposal_coverage
    result["corridorProposalRetainedBoundaryFraction"] = retained_boundary
    result["corridorProposalBoundaryAnchorCount"] = anchor_count
    return result


def _load_corridor_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_PATCH_CORRIDORS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_PATCH_CORRIDORS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("exact variants require a complete label-free corridor artifact")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _corridor_settings_from_manifest(
    manifest: Mapping[str, Any],
) -> PhysicalRibbonPatchCorridorSettings:
    values = dict(manifest["identity"]["settings"])
    for name in (
        "hermite_tensions",
        "profile_depth_fractions",
        "competing_shift_thicknesses",
    ):
        values[name] = tuple(float(value) for value in values[name])
    return PhysicalRibbonPatchCorridorSettings(**values)


def _checkpoint(
    output: Path,
    stem: str,
    identity: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    statistics: Mapping[str, Any],
) -> tuple[Path, dict[str, Any]]:
    data_path = output / f"{stem}.npz"
    manifest_path = output / f"{stem}.json"
    _write_npz(data_path, arrays)
    payload = {
        "state": "complete",
        "identity": dict(identity),
        "statistics": dict(statistics),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
    }
    atomic_json(manifest_path, payload)
    return manifest_path, payload


def _cached_checkpoint(
    output: Path,
    stem: str,
    identity_sha256: str,
) -> tuple[dict[str, np.ndarray], dict[str, Any]] | None:
    manifest_path = output / f"{stem}.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("state") != "complete"
        or manifest.get("identity", {}).get("identitySha256")
        != identity_sha256
    ):
        return None
    data_path = output / str(manifest["data"]["path"])
    if (
        not data_path.is_file()
        or sha256_file(data_path) != manifest["data"]["sha256"]
    ):
        return None
    return _load_npz(data_path, manifest["data"]["sha256"]), manifest


def run_physical_ribbon_corridor_variants(
    corridor_root: str | Path,
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorVariantSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorVariantSettings()
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_root
    )
    (
        configuration_path,
        configuration_manifest,
        configuration,
        continuity_path,
        continuity_manifest,
        topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_root)
    expected_configuration = corridor_manifest["identity"]["configuration"]
    if (
        expected_configuration["manifestSha256"]
        != sha256_file(configuration_path)
        or expected_configuration["dataSha256"]
        != configuration_manifest["data"]["sha256"]
    ):
        raise ValueError("corridor artifact and configuration do not match")
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_VERSION,
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
        "corridorSettings": corridor_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "corridorImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_corridors.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    final_manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.json"
    final_data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.npz"
    corridor_preview_path = output / "physical-ribbon-corridor-variants.png"
    fragment_preview_path = output / "physical-ribbon-corridor-variant-fragments.png"
    if not force and final_manifest_path.is_file() and final_data_path.is_file():
        cached = json.loads(final_manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256")
            == sha256_file(final_data_path)
        ):
            return cached
    started = time.monotonic()
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    enumeration_identity = {
        **identity,
        "stage": "enumeration",
    }
    enumeration_identity["identitySha256"] = canonical_json_hash(
        enumeration_identity
    )
    cached_enumeration = None if force else _cached_checkpoint(
        output,
        "corridor-variant-enumeration-v1",
        enumeration_identity["identitySha256"],
    )
    if cached_enumeration is None:
        if progress is not None:
            progress("enumerating complete corridor factor states")
        variants, enumeration_stats = enumerate_corridor_reconfiguration_variants(
            corridor,
            corridor,
            corridor,
            ribbon,
            topology,
            configuration,
            continuity_weight=continuity_weight,
            corridor_settings=corridor_settings,
            settings=resolved,
        )
        enumeration_manifest_path, enumeration_manifest = _checkpoint(
            output,
            "corridor-variant-enumeration-v1",
            enumeration_identity,
            variants,
            enumeration_stats,
        )
    else:
        variants, enumeration_manifest = cached_enumeration
        enumeration_stats = enumeration_manifest["statistics"]
        enumeration_manifest_path = output / "corridor-variant-enumeration-v1.json"
    enumerated_at = time.monotonic()
    exact_identity = {
        **identity,
        "stage": "exact-screen",
        "enumerationManifestSha256": sha256_file(enumeration_manifest_path),
        "enumerationDataSha256": enumeration_manifest["data"]["sha256"],
    }
    exact_identity["identitySha256"] = canonical_json_hash(exact_identity)
    cached_exact = None if force else _cached_checkpoint(
        output,
        "corridor-variant-exact-screen-v1",
        exact_identity["identitySha256"],
    )
    if cached_exact is None:
        if progress is not None:
            progress("reconstructing corridor variants in complete source sheets")
        exact, exact_stats = screen_exact_corridor_variants(
            corridor,
            corridor,
            corridor,
            variants,
            ribbon,
            topology,
            configuration,
            corridor_settings=corridor_settings,
            settings=resolved,
            progress=progress,
        )
        exact_manifest_path, exact_manifest = _checkpoint(
            output,
            "corridor-variant-exact-screen-v1",
            exact_identity,
            exact,
            exact_stats,
        )
    else:
        exact, exact_manifest = cached_exact
        exact_stats = exact_manifest["statistics"]
        exact_manifest_path = output / "corridor-variant-exact-screen-v1.json"
    screened_at = time.monotonic()
    compiled = compile_exact_variant_reconfiguration(corridor, variants, exact)
    if progress is not None:
        progress("globally replaying exact corridor variants")
    replay, replay_stats = replay_patch_corridor_reconfigurations(
        corridor,
        corridor,
        corridor,
        compiled,
        ribbon,
        topology,
        configuration,
        settings=corridor_settings,
    )
    replayed_at = time.monotonic()
    arrays = {**variants, **exact, **replay}
    _write_npz(final_data_path, arrays)
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    write_patch_corridor_montage(
        corridor,
        corridor,
        corridor_preview_path,
        maximum_corridors=corridor_settings.maximum_preview_corridors,
        reconfiguration=compiled,
        replay=replay,
    )
    _, fragment_stats = write_replayed_corridor_fragment_montage(
        corridor,
        corridor,
        corridor,
        replay,
        source,
        fragment_preview_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": corridor_manifest.get("geometry", {}),
        "enumeration": enumeration_stats,
        "exactScreen": exact_stats,
        "counterfactualReplay": replay_stats,
        "flattenedReplayFragments": fragment_stats,
        "checkpoints": {
            "enumeration": enumeration_manifest_path.name,
            "exactScreen": exact_manifest_path.name,
        },
        "timingSeconds": {
            "enumeration": round(enumerated_at - started, 6),
            "exactLocalScreen": round(screened_at - enumerated_at, 6),
            "globalReplay": round(replayed_at - screened_at, 6),
            "writingAndPreviews": round(finished - replayed_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": final_data_path.name,
            "bytes": final_data_path.stat().st_size,
            "sha256": sha256_file(final_data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "corridorVariantMontage": corridor_preview_path.name,
            "flattenedReplayFragments": fragment_preview_path.name,
        },
        "method": {
            "decisionUnit": "one complete joint interface matching variant reconstructed inside its full physical-sheet component",
            "selection": "exact edge-connected region reduction and supported area dominate local factor score",
            "mutation": "counterfactual only; source configuration and corridor artifacts remain unchanged",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(final_manifest_path, payload)
    return payload
