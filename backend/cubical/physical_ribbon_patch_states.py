from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_collective import (
    _objective,
    _shared_endpoint_conflicts,
    _triangle_region_counts,
    optimize_collective_patch,
)
from .physical_ribbon_configuration import _adjacency, _component_labels
from .physical_ribbon_continuity import (
    write_continuity_overview,
    write_largest_component_montage,
)
from .physical_ribbon_patch_holes import (
    PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
    PhysicalRibbonPatchHoleSettings,
    build_physical_ribbon_surface_complex,
    extract_surface_boundary_loops,
)


PHYSICAL_RIBBON_PATCH_STATE_SCHEMA = "pareidolia.physical-ribbon-patch-state"
PHYSICAL_RIBBON_PATCH_STATE_VERSION = 1
PHYSICAL_RIBBON_PATCH_STATE_STEM = "physical-ribbon-patch-state-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonPatchStateSettings:
    """General gates for haloed, collective surface reconfiguration.

    A state is not a ribbon or a cell.  It is every CT-supported closed-hole
    frontier belonging to one already reconstructed surface component.  The
    selected surface outside the alternating interface conflicts is an
    immutable halo.  The complete local matching is optimized at once and is
    committed only when exact reconstruction makes that component denser.
    """

    minimum_added_ribbon_count: int = 3
    minimum_objective_gain: float = 0.25
    minimum_context_profile_correlation: float = 0.85
    minimum_competing_layer_margin: float = 0.20
    minimum_patch_coverage: float = 0.25
    minimum_retained_boundary_fraction: float = 0.15
    minimum_boundary_anchor_count: int = 1
    minimum_added_two_core_fraction: float = 0.60
    minimum_added_tangent_ratio: float = 0.02
    minimum_surface_realization_fraction: float = 0.80
    minimum_triangle_area_retention: float = 0.995
    minimum_density_only_retained_boundary_fraction: float = 0.60
    minimum_density_only_boundary_anchor_count: int = 3
    maximum_local_optimization_sweeps: int = 8
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        positive_integer = (
            self.minimum_added_ribbon_count,
            self.minimum_boundary_anchor_count,
            self.minimum_density_only_boundary_anchor_count,
            self.maximum_local_optimization_sweeps,
            self.maximum_preview_components,
        )
        if any(value < 1 for value in positive_integer):
            raise ValueError("patch-state integer settings must be positive")
        if not math.isfinite(self.minimum_objective_gain):
            raise ValueError("patch-state objective gate must be finite")
        for value in (
            self.minimum_context_profile_correlation,
            self.minimum_patch_coverage,
            self.minimum_retained_boundary_fraction,
            self.minimum_added_two_core_fraction,
            self.minimum_added_tangent_ratio,
            self.minimum_surface_realization_fraction,
            self.minimum_triangle_area_retention,
            self.minimum_density_only_retained_boundary_fraction,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("patch-state fractions must lie in [0, 1]")
        if not math.isfinite(self.minimum_competing_layer_margin):
            raise ValueError("competing-layer margin must be finite")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_holes_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("holes root must identify exactly one complete artifact")
    return matches[0]


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def _loop_vertices(holes: Mapping[str, np.ndarray], loop_index: int) -> np.ndarray:
    offset = np.asarray(holes["loopOffset"], dtype=np.int64)
    vertex = np.asarray(holes["loopVertexFrontierIndex"], dtype=np.int32)
    return vertex[int(offset[loop_index]) : int(offset[loop_index + 1])]


def _surface_view(holes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = (
        "selected",
        "component",
        "componentSize",
        "signedNormalXYZ",
        "tangentUxyz",
        "tangentVxyz",
        "chartUV",
        "integrationResidualVoxels",
        "edgeFirstFrontierIndex",
        "edgeSecondFrontierIndex",
        "edgeSelected",
        "triangleFrontierIndex",
        "triangleAreaVoxelsSquared",
        "triangleNormalResidualDegrees",
        "midpointXYZ",
        "thicknessVoxels",
    )
    return {name: np.asarray(holes[name]) for name in names}


def _loops_view(holes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = (
        "loopOffset",
        "loopVertexFrontierIndex",
        "loopTriangleRegion",
        "loopTopologyComponent",
        "loopKind",
        "loopAreaChartVoxelsSquared",
        "loopPerimeterChartVoxels",
        "loopDiameterChartVoxels",
        "loopMedianThicknessVoxels",
        "loopMeanBoundaryEdgeVoxels",
        "loopMacroEligible",
    )
    return {name: np.asarray(holes[name]) for name in names}


def _component_loop_counts(
    loops: Mapping[str, np.ndarray],
) -> dict[int, dict[str, int]]:
    component = np.asarray(loops["loopTopologyComponent"], dtype=np.int32)
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    macro = np.asarray(loops["loopMacroEligible"], dtype=np.uint8) > 0
    result: dict[int, dict[str, int]] = {}
    for component_id in np.unique(component):
        member = component == component_id
        result[int(component_id)] = {
            "interiorHoleCount": int(np.count_nonzero(member & (kind == 1))),
            "macroHoleCount": int(np.count_nonzero(member & macro)),
        }
    return result


def _component_surface_metrics(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
) -> dict[int, dict[str, float | int]]:
    component = np.asarray(surface["component"], dtype=np.int32)
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    area = np.asarray(surface["triangleAreaVoxelsSquared"], dtype=np.float32)
    region_count = _triangle_region_counts(surface)
    loop_count = _component_loop_counts(loops)
    result: dict[int, dict[str, float | int]] = {}
    if len(triangle):
        triangle_component = component[triangle[:, 0]]
        for component_id in np.unique(triangle_component):
            member = triangle_component == component_id
            result[int(component_id)] = {
                "triangleCount": int(np.count_nonzero(member)),
                "triangleAreaVoxelsSquared": float(np.sum(area[member])),
                "triangleRegionCount": int(region_count.get(int(component_id), 0)),
                **loop_count.get(
                    int(component_id),
                    {"interiorHoleCount": 0, "macroHoleCount": 0},
                ),
            }
    for component_id, counts in loop_count.items():
        result.setdefault(
            component_id,
            {
                "triangleCount": 0,
                "triangleAreaVoxelsSquared": 0.0,
                "triangleRegionCount": int(region_count.get(component_id, 0)),
                **counts,
            },
        )
    return result


def _selection_conflicts(
    selected: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    crossing_first: np.ndarray,
    crossing_second: np.ndarray,
) -> tuple[int, int]:
    node = np.flatnonzero(selected)
    endpoint = np.concatenate((source[node], target[node]))
    return (
        int(len(endpoint) - len(np.unique(endpoint))),
        int(np.count_nonzero(selected[crossing_first] & selected[crossing_second])),
    )


def _lineage_audit(
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    final_selected: np.ndarray,
    final_component: np.ndarray,
) -> tuple[dict[int, int], dict[str, int]]:
    mapping: dict[int, int] = {}
    split_count = 0
    deleted_count = 0
    for component_id in np.unique(baseline_component[baseline_selected]):
        inherited = (
            baseline_selected
            & final_selected
            & (baseline_component == component_id)
        )
        values = np.unique(final_component[inherited])
        values = values[values >= 0]
        if not len(values):
            deleted_count += 1
            continue
        if len(values) != 1:
            split_count += 1
            continue
        mapping[int(component_id)] = int(values[0])
    fusion_count = 0
    for component_id in np.unique(final_component[final_selected]):
        member = final_selected & (final_component == component_id)
        inherited = np.unique(
            baseline_component[member & baseline_selected]
        )
        inherited = inherited[inherited >= 0]
        fusion_count += int(len(inherited) > 1)
    return mapping, {
        "deletedPriorComponentCount": int(deleted_count),
        "splitPriorComponentCount": int(split_count),
        "crossPriorComponentFusionCount": int(fusion_count),
    }


def _local_edges(
    options: np.ndarray,
    offset: np.ndarray,
    neighbor: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    local = {int(value): index for index, value in enumerate(options)}
    first_values: list[int] = []
    second_values: list[int] = []
    weights: list[float] = []
    for first_local, first_global in enumerate(options):
        begin, end = int(offset[first_global]), int(offset[first_global + 1])
        for second_global, edge_weight in zip(
            neighbor[begin:end], weight[begin:end]
        ):
            second_local = local.get(int(second_global))
            if second_local is None or first_local >= second_local:
                continue
            first_values.append(first_local)
            second_values.append(second_local)
            weights.append(float(edge_weight))
    return (
        np.asarray(first_values, dtype=np.int32),
        np.asarray(second_values, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
        local,
    )


def _group_candidates(
    hole_rows: np.ndarray,
    holes: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[int, int]]]:
    offset = np.asarray(holes["patchCandidateOffset"], dtype=np.int64)
    candidate = np.asarray(holes["patchCandidateFrontierIndex"], dtype=np.int32)
    alignment = np.asarray(holes["patchCandidateSurfaceAlignment"], dtype=np.float32)
    nearest = np.asarray(holes["patchCandidateNearestPixel"], dtype=np.int32)
    score: dict[int, float] = {}
    row_nearest: dict[int, dict[int, int]] = {}
    for row_value in hole_rows:
        row = int(row_value)
        start, stop = int(offset[row]), int(offset[row + 1])
        mapping: dict[int, int] = {}
        for node, value, pixel in zip(
            candidate[start:stop], alignment[start:stop], nearest[start:stop]
        ):
            node_value = int(node)
            score[node_value] = max(score.get(node_value, 0.0), float(value))
            mapping[node_value] = int(pixel)
        row_nearest[row] = mapping
    nodes = np.asarray(sorted(score), dtype=np.int32)
    values = np.asarray([score[int(node)] for node in nodes], dtype=np.float32)
    return nodes, values, row_nearest


def _proposal_hole_metrics(
    hole_rows: np.ndarray,
    row_nearest: Mapping[int, Mapping[int, int]],
    added: set[int],
    trial_selected: np.ndarray,
    holes: Mapping[str, np.ndarray],
    topology_offset: np.ndarray,
    topology_neighbor: np.ndarray,
) -> list[dict[str, Any]]:
    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    patch_offset = np.asarray(holes["patchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(holes["patchXYZ"], dtype=np.float32)
    mean_edge = np.asarray(holes["loopMeanBoundaryEdgeVoxels"], dtype=np.float32)
    selected_model = np.asarray(holes["selectedModel"], dtype=np.int32)
    correlation = np.asarray(
        holes["zeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    margin = np.asarray(holes["zeroShiftCompetingMargin"], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for row_value in hole_rows:
        row = int(row_value)
        loop = int(scored_loop[row])
        candidate_added = sorted(added.intersection(row_nearest[row]))
        if not candidate_added:
            continue
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        patch = patch_xyz[start:stop]
        projected = patch[
            np.asarray([row_nearest[row][node] for node in candidate_added], dtype=np.int32)
        ]
        coverage_radius = float(mean_edge[loop])
        covered = np.any(
            np.linalg.norm(patch[:, None, :] - projected[None, :, :], axis=2)
            <= coverage_radius,
            axis=1,
        )
        boundary = _loop_vertices(holes, loop)
        boundary_set = set(int(value) for value in boundary)
        anchors: set[int] = set()
        for node in candidate_added:
            begin, end = int(topology_offset[node]), int(topology_offset[node + 1])
            for neighbor in topology_neighbor[begin:end]:
                value = int(neighbor)
                if value in boundary_set and trial_selected[value]:
                    anchors.add(value)
        model = int(selected_model[row])
        records.append(
            {
                "holeRow": row,
                "loopIndex": loop,
                "addedRibbonCount": len(candidate_added),
                "coverage": float(np.mean(covered)) if len(covered) else 0.0,
                "retainedBoundaryFraction": float(np.mean(trial_selected[boundary])),
                "boundaryAnchorCount": len(anchors),
                "contextProfileCorrelation": float(correlation[row, model]),
                "competingLayerMargin": float(margin[row, model]),
            }
        )
    return records


def _added_geometry(
    added: np.ndarray,
    selected: np.ndarray,
    midpoint: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[float, float]:
    if not len(added):
        return 0.0, 0.0
    added_mask = np.zeros(len(selected), dtype=bool)
    added_mask[added] = True
    active = selected[edge_first] & selected[edge_second]
    incident = active & (added_mask[edge_first] | added_mask[edge_second])
    degree = np.zeros(len(selected), dtype=np.int32)
    np.add.at(degree, edge_first[incident], 1)
    np.add.at(degree, edge_second[incident], 1)
    two_core = float(np.mean(degree[added] >= 2))
    if np.count_nonzero(incident) < 2:
        return two_core, 0.0
    direction = midpoint[edge_second[incident]] - midpoint[edge_first[incident]]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-6)
    covariance = np.einsum(
        "i,ij,ik->jk", edge_weight[incident], direction, direction
    )
    eigenvalue = np.linalg.eigvalsh(covariance)
    tangent_ratio = float(eigenvalue[1] / max(float(eigenvalue[2]), 1.0e-6))
    return two_core, tangent_ratio


def solve_component_patch_states(
    holes: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    surface_alignment_weight: float,
    settings: PhysicalRibbonPatchStateSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    unary = np.asarray(configuration["nodeUnaryScore"], dtype=np.float32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    topology_offset, topology_neighbor, topology_weight = _adjacency(
        len(frontier), edge_first, edge_second, edge_score
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        len(frontier), crossing_first, crossing_second
    )
    interface_owner = np.full(
        len(np.asarray(ribbon["interfaceCandidateDegree"])), -1, dtype=np.int32
    )
    selected_node = np.flatnonzero(selected)
    interface_owner[source[selected_node]] = selected_node
    interface_owner[target[selected_node]] = selected_node

    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    loop_component = np.asarray(holes["loopTopologyComponent"], dtype=np.int32)
    rows_by_component: dict[int, list[int]] = defaultdict(list)
    for row, loop in enumerate(scored_loop):
        rows_by_component[int(loop_component[int(loop)])].append(row)

    proposals: list[dict[str, Any]] = []
    rejected_external_candidate_count = 0
    for target_component, row_values in sorted(rows_by_component.items()):
        hole_rows = np.asarray(row_values, dtype=np.int32)
        raw_candidate, raw_alignment, row_nearest = _group_candidates(
            hole_rows, holes
        )
        alignment_by_node = {
            int(node): float(value)
            for node, value in zip(raw_candidate, raw_alignment)
        }
        candidates: list[int] = []
        incumbents: set[int] = set()
        for node_value in raw_candidate:
            node = int(node_value)
            if selected[node]:
                continue
            blockers: set[int] = set()
            for owner in (interface_owner[source[node]], interface_owner[target[node]]):
                if owner >= 0:
                    blockers.add(int(owner))
            begin, end = int(crossing_offset[node]), int(crossing_offset[node + 1])
            blockers.update(
                int(value)
                for value in crossing_neighbor[begin:end]
                if selected[int(value)]
            )
            if any(component[value] != target_component for value in blockers):
                rejected_external_candidate_count += 1
                continue
            candidates.append(node)
            incumbents.update(blockers)
        candidate = np.asarray(sorted(set(candidates)), dtype=np.int32)
        incumbent = np.asarray(sorted(incumbents), dtype=np.int32)
        options = np.concatenate((incumbent, candidate))
        if not len(candidate) or not len(options):
            continue
        local_first, local_second, local_edge_score, local = _local_edges(
            options,
            topology_offset,
            topology_neighbor,
            topology_weight,
        )
        fixed_selected = selected.copy()
        fixed_selected[incumbent] = False
        fixed_support = np.zeros(len(options), dtype=np.float64)
        for option_index, node_value in enumerate(options):
            begin, end = (
                int(topology_offset[node_value]),
                int(topology_offset[node_value + 1]),
            )
            neighbor = topology_neighbor[begin:end]
            weight = topology_weight[begin:end]
            fixed_support[option_index] = float(np.sum(weight[fixed_selected[neighbor]]))
        node_score = unary[options].astype(np.float64)
        node_score += continuity_weight * fixed_support
        for option_index, node_value in enumerate(options):
            node_score[option_index] += surface_alignment_weight * alignment_by_node.get(
                int(node_value), 0.0
            )
        local_edge_weight = continuity_weight * local_edge_score
        shared_first, shared_second = _shared_endpoint_conflicts(
            source[options], target[options]
        )
        cross_local_first: list[int] = []
        cross_local_second: list[int] = []
        for first_local, first_global in enumerate(options):
            begin, end = (
                int(crossing_offset[first_global]),
                int(crossing_offset[first_global + 1]),
            )
            for second_global in crossing_neighbor[begin:end]:
                second_local = local.get(int(second_global))
                if second_local is not None and first_local < second_local:
                    cross_local_first.append(first_local)
                    cross_local_second.append(second_local)
        conflict_first = np.concatenate(
            (shared_first, np.asarray(cross_local_first, dtype=np.int32))
        )
        conflict_second = np.concatenate(
            (shared_second, np.asarray(cross_local_second, dtype=np.int32))
        )
        baseline_local = selected[options]
        baseline_objective = _objective(
            baseline_local,
            node_score,
            local_first,
            local_second,
            local_edge_weight,
        )
        optimized, _ = optimize_collective_patch(
            node_score,
            local_first,
            local_second,
            local_edge_weight,
            conflict_first,
            conflict_second,
            maximum_sweeps=settings.maximum_local_optimization_sweeps,
            initial_selection=baseline_local,
        )

        # The optimizer may discover that an unrelated weak incumbent has a
        # negative marginal.  This operator is an alternating surface repair,
        # not a pruning pass: remove incumbents only when a chosen alternative
        # physically excludes them.
        chosen_added_local = optimized & ~baseline_local
        required_removed = np.zeros(len(options), dtype=bool)
        for first, second in zip(conflict_first, conflict_second):
            if chosen_added_local[first] and baseline_local[second]:
                required_removed[second] = True
            if chosen_added_local[second] and baseline_local[first]:
                required_removed[first] = True
        proposed = baseline_local.copy()
        proposed[required_removed] = False
        proposed[chosen_added_local] = True
        proposal_objective = _objective(
            proposed,
            node_score,
            local_first,
            local_second,
            local_edge_weight,
        )
        added = options[proposed & ~baseline_local]
        removed = options[baseline_local & ~proposed]
        trial_selected = selected.copy()
        trial_selected[removed] = False
        trial_selected[added] = True
        two_core, tangent_ratio = _added_geometry(
            added,
            trial_selected,
            midpoint,
            edge_first,
            edge_second,
            edge_score,
        )
        hole_metrics = _proposal_hole_metrics(
            hole_rows,
            row_nearest,
            set(int(value) for value in added),
            trial_selected,
            holes,
            topology_offset,
            topology_neighbor,
        )
        objective_gain = float(proposal_objective - baseline_objective)
        qualified = (
            len(added) >= settings.minimum_added_ribbon_count
            and objective_gain >= settings.minimum_objective_gain
            and two_core >= settings.minimum_added_two_core_fraction
            and tangent_ratio >= settings.minimum_added_tangent_ratio
            and bool(hole_metrics)
            and all(
                record["contextProfileCorrelation"]
                >= settings.minimum_context_profile_correlation
                and record["competingLayerMargin"]
                >= settings.minimum_competing_layer_margin
                and record["coverage"] >= settings.minimum_patch_coverage
                and record["retainedBoundaryFraction"]
                >= settings.minimum_retained_boundary_fraction
                and record["boundaryAnchorCount"]
                >= settings.minimum_boundary_anchor_count
                for record in hole_metrics
            )
        )
        proposals.append(
            {
                "targetComponent": int(target_component),
                "holeRows": hole_rows,
                "options": options,
                "candidateCount": int(len(candidate)),
                "added": added.astype(np.int32),
                "removed": removed.astype(np.int32),
                "objectiveGain": objective_gain,
                "twoCoreFraction": two_core,
                "tangentRatio": tangent_ratio,
                "holeMetrics": hole_metrics,
                "qualified": bool(qualified),
            }
        )

    proposal_order = sorted(
        range(len(proposals)),
        key=lambda row: (
            float(proposals[row]["objectiveGain"]),
            len(proposals[row]["added"]),
            -int(proposals[row]["targetComponent"]),
        ),
        reverse=True,
    )
    composed = selected.copy()
    applied = np.zeros(len(proposals), dtype=bool)
    for row in proposal_order:
        proposal = proposals[row]
        if not proposal["qualified"]:
            continue
        trial = composed.copy()
        trial[proposal["removed"]] = False
        trial[proposal["added"]] = True
        if _selection_conflicts(
            trial, source, target, crossing_first, crossing_second
        ) != (0, 0):
            continue
        composed = trial
        applied[row] = True
    for row, proposal in enumerate(proposals):
        proposal["composed"] = bool(applied[row])
    final_component, final_size = _component_labels(
        composed, edge_first, edge_second
    )
    arrays = {
        "selected": composed.astype(np.uint8),
        "component": final_component.astype(np.int32),
        "componentSize": final_size.astype(np.int32),
    }
    statistics = {
        "scoredHoleCount": int(len(scored_loop)),
        "surfaceComponentPatchCount": int(len(proposals)),
        "candidateRibbonCount": int(
            sum(int(proposal["candidateCount"]) for proposal in proposals)
        ),
        "externallyBlockedCandidateCount": int(rejected_external_candidate_count),
        "qualifiedPatchCount": int(sum(proposal["qualified"] for proposal in proposals)),
        "composedPatchCount": int(np.count_nonzero(applied)),
        "proposedAddedRibbonCount": int(
            sum(len(proposals[row]["added"]) for row in np.flatnonzero(applied))
        ),
        "proposedRemovedRibbonCount": int(
            sum(len(proposals[row]["removed"]) for row in np.flatnonzero(applied))
        ),
        "singleCellGrowth": False,
        "decisionUnit": "all CT-supported closed-hole frontiers on one surface component",
        "fixedHalo": "every selected ribbon not in an alternating interface or crossing conflict",
    }
    return arrays, proposals, statistics


def _proposal_arrays(
    proposals: list[Mapping[str, Any]], accepted: np.ndarray
) -> dict[str, np.ndarray]:
    hole_offset = [0]
    hole_values: list[int] = []
    option_offset = [0]
    option_values: list[int] = []
    added_offset = [0]
    added_values: list[int] = []
    removed_offset = [0]
    removed_values: list[int] = []
    metric_offset = [0]
    metric_hole_row: list[int] = []
    metric_coverage: list[float] = []
    metric_retained: list[float] = []
    metric_anchor: list[int] = []
    metric_correlation: list[float] = []
    metric_margin: list[float] = []
    for proposal in proposals:
        hole_values.extend(int(value) for value in proposal["holeRows"])
        hole_offset.append(len(hole_values))
        option_values.extend(int(value) for value in proposal["options"])
        option_offset.append(len(option_values))
        added_values.extend(int(value) for value in proposal["added"])
        added_offset.append(len(added_values))
        removed_values.extend(int(value) for value in proposal["removed"])
        removed_offset.append(len(removed_values))
        for record in proposal["holeMetrics"]:
            metric_hole_row.append(int(record["holeRow"]))
            metric_coverage.append(float(record["coverage"]))
            metric_retained.append(float(record["retainedBoundaryFraction"]))
            metric_anchor.append(int(record["boundaryAnchorCount"]))
            metric_correlation.append(float(record["contextProfileCorrelation"]))
            metric_margin.append(float(record["competingLayerMargin"]))
        metric_offset.append(len(metric_hole_row))
    return {
        "patchTargetPriorComponent": np.asarray(
            [proposal["targetComponent"] for proposal in proposals], dtype=np.int32
        ),
        "patchHoleRowOffset": np.asarray(hole_offset, dtype=np.int64),
        "patchHoleRow": np.asarray(hole_values, dtype=np.int32),
        "patchOptionOffset": np.asarray(option_offset, dtype=np.int64),
        "patchOptionFrontierIndex": np.asarray(option_values, dtype=np.int32),
        "patchCandidateCount": np.asarray(
            [proposal["candidateCount"] for proposal in proposals], dtype=np.int32
        ),
        "patchAddedOffset": np.asarray(added_offset, dtype=np.int64),
        "patchAddedFrontierIndex": np.asarray(added_values, dtype=np.int32),
        "patchRemovedOffset": np.asarray(removed_offset, dtype=np.int64),
        "patchRemovedFrontierIndex": np.asarray(removed_values, dtype=np.int32),
        "patchObjectiveGain": np.asarray(
            [proposal["objectiveGain"] for proposal in proposals], dtype=np.float32
        ),
        "patchAddedTwoCoreFraction": np.asarray(
            [proposal["twoCoreFraction"] for proposal in proposals], dtype=np.float32
        ),
        "patchAddedTangentRatio": np.asarray(
            [proposal["tangentRatio"] for proposal in proposals], dtype=np.float32
        ),
        "patchQualified": np.asarray(
            [proposal["qualified"] for proposal in proposals], dtype=np.uint8
        ),
        "patchAccepted": np.asarray(accepted, dtype=np.uint8),
        "patchMetricOffset": np.asarray(metric_offset, dtype=np.int64),
        "patchMetricHoleRow": np.asarray(metric_hole_row, dtype=np.int32),
        "patchMetricCoverage": np.asarray(metric_coverage, dtype=np.float32),
        "patchMetricRetainedBoundaryFraction": np.asarray(
            metric_retained, dtype=np.float32
        ),
        "patchMetricBoundaryAnchorCount": np.asarray(metric_anchor, dtype=np.int32),
        "patchMetricContextProfileCorrelation": np.asarray(
            metric_correlation, dtype=np.float32
        ),
        "patchMetricCompetingLayerMargin": np.asarray(metric_margin, dtype=np.float32),
    }


def run_physical_ribbon_patch_states(
    holes_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonPatchStateSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonPatchStateSettings()
    holes_path, holes_manifest = _resolve_holes_manifest(holes_root)
    holes_data_path = holes_path.parent / str(holes_manifest["data"]["path"])
    holes = _load_npz(holes_data_path, holes_manifest["data"]["sha256"])
    configuration_reference = holes_manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        configuration,
        topology_path,
        topology_manifest,
        topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path) != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("hole analysis configuration provenance changed")
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    surface_alignment_weight = float(
        holes_manifest.get("identity", {})
        .get("settings", {})
        .get("surface_alignment_weight", 0.35)
    )
    identity = {
        "schema": PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_STATE_VERSION,
        "holes": _reference(holes_path, holes_manifest),
        "configuration": _reference(configuration_path, configuration_manifest),
        "topologyContinuity": _reference(topology_path, topology_manifest),
        "ribbonBank": _reference(ribbon_path, ribbon_manifest),
        "continuityWeight": continuity_weight,
        "surfaceAlignmentWeight": surface_alignment_weight,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_PATCH_STATE_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_PATCH_STATE_STEM}.npz"
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
    baseline_surface = _surface_view(holes)
    baseline_loops = _loops_view(holes)
    baseline_metrics = _component_surface_metrics(baseline_surface, baseline_loops)
    arrays, proposals, solve_stats = solve_component_patch_states(
        holes,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        surface_alignment_weight=surface_alignment_weight,
        settings=resolved,
    )
    solved = time.monotonic()
    surface_settings = PhysicalRibbonPatchHoleSettings()
    preliminary_surface, _ = build_physical_ribbon_surface_complex(
        ribbon, topology, arrays, settings=surface_settings
    )
    preliminary_loops, _ = extract_surface_boundary_loops(
        preliminary_surface, settings=surface_settings
    )
    surfaced = time.monotonic()
    baseline_selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    baseline_component = np.asarray(configuration["component"], dtype=np.int32)
    preliminary_selected = np.asarray(arrays["selected"], dtype=np.uint8) > 0
    preliminary_component = np.asarray(arrays["component"], dtype=np.int32)
    mapping, lineage = _lineage_audit(
        baseline_selected,
        baseline_component,
        preliminary_selected,
        preliminary_component,
    )
    preliminary_metrics = _component_surface_metrics(
        preliminary_surface, preliminary_loops
    )
    triangle = np.asarray(
        preliminary_surface["triangleFrontierIndex"], dtype=np.int32
    )
    triangle_node = np.zeros(len(preliminary_selected), dtype=bool)
    if len(triangle):
        triangle_node[np.unique(triangle)] = True
    accepted = np.zeros(len(proposals), dtype=bool)
    exact_records: list[dict[str, Any]] = []
    for row, proposal in enumerate(proposals):
        target_component = int(proposal["targetComponent"])
        if not proposal.get("composed", False):
            continue
        final_component_id = mapping.get(target_component, -1)
        before = baseline_metrics.get(target_component, {})
        after = preliminary_metrics.get(final_component_id, {})
        added = np.asarray(proposal["added"], dtype=np.int32)
        realization = (
            float(np.mean(triangle_node[added])) if len(added) else 0.0
        )
        area_retention = float(after.get("triangleAreaVoxelsSquared", 0.0)) / max(
            float(before.get("triangleAreaVoxelsSquared", 0.0)), 1.0e-6
        )
        lineage_ok = final_component_id >= 0
        if lineage_ok:
            member = preliminary_selected & (
                preliminary_component == final_component_id
            )
            inherited = np.unique(
                baseline_component[member & baseline_selected]
            )
            inherited = inherited[inherited >= 0]
            lineage_ok = len(inherited) == 1 and int(inherited[0]) == target_component
        non_regression = (
            int(after.get("triangleRegionCount", 0))
            <= int(before.get("triangleRegionCount", 0))
            and int(after.get("interiorHoleCount", 0))
            <= int(before.get("interiorHoleCount", 0))
            and int(after.get("macroHoleCount", 0))
            <= int(before.get("macroHoleCount", 0))
            and area_retention >= resolved.minimum_triangle_area_retention
        )
        topology_improved = (
            int(after.get("triangleRegionCount", 0))
            < int(before.get("triangleRegionCount", 0))
            or int(after.get("interiorHoleCount", 0))
            < int(before.get("interiorHoleCount", 0))
            or int(after.get("macroHoleCount", 0))
            < int(before.get("macroHoleCount", 0))
        )
        minimum_retained = min(
            (
                float(record["retainedBoundaryFraction"])
                for record in proposal["holeMetrics"]
            ),
            default=0.0,
        )
        minimum_anchor = min(
            (
                int(record["boundaryAnchorCount"])
                for record in proposal["holeMetrics"]
            ),
            default=0,
        )
        density_improved = (
            int(after.get("triangleCount", 0)) > int(before.get("triangleCount", 0))
            and minimum_retained
            >= resolved.minimum_density_only_retained_boundary_fraction
            and minimum_anchor >= resolved.minimum_density_only_boundary_anchor_count
        )
        improved = topology_improved or density_improved
        accepted[row] = bool(
            lineage_ok
            and non_regression
            and improved
            and realization >= resolved.minimum_surface_realization_fraction
        )
        exact_records.append(
            {
                "patchRow": row,
                "priorComponent": target_component,
                "finalComponent": final_component_id,
                "accepted": bool(accepted[row]),
                "addedSurfaceRealizationFraction": realization,
                "triangleAreaRetention": area_retention,
                "before": before,
                "after": after,
            }
        )

    # Rebuild once after discarding every exact counterexample.  Component
    # patches are disjoint by prior identity, so a failed component can be
    # removed without invalidating the evidence of an accepted component.
    final_selected = baseline_selected.copy()
    for row in np.flatnonzero(accepted):
        final_selected[proposals[row]["removed"]] = False
        final_selected[proposals[row]["added"]] = True
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    final_component, final_size = _component_labels(
        final_selected, edge_first, edge_second
    )
    arrays["selected"] = final_selected.astype(np.uint8)
    arrays["component"] = final_component.astype(np.int32)
    arrays["componentSize"] = final_size.astype(np.int32)
    if np.array_equal(final_selected, preliminary_selected):
        final_surface = preliminary_surface
        final_loops = preliminary_loops
    else:
        final_surface, _ = build_physical_ribbon_surface_complex(
            ribbon, topology, arrays, settings=surface_settings
        )
        final_loops, _ = extract_surface_boundary_loops(
            final_surface, settings=surface_settings
        )
    finished_surface = time.monotonic()

    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    conflict_count = _selection_conflicts(
        final_selected, source, target, crossing_first, crossing_second
    )
    _, final_lineage = _lineage_audit(
        baseline_selected,
        baseline_component,
        final_selected,
        final_component,
    )
    if conflict_count != (0, 0) or any(final_lineage.values()):
        raise RuntimeError("exact patch-state replay violated a hard invariant")
    final_loop_stats = {
        "triangleRegionCount": int(sum(_triangle_region_counts(final_surface).values())),
        "interiorHoleLoopCount": int(
            np.count_nonzero(np.asarray(final_loops["loopKind"]) == 1)
        ),
        "macroEligibleHoleCount": int(
            np.count_nonzero(np.asarray(final_loops["loopMacroEligible"]) > 0)
        ),
    }
    baseline_loop_stats = holes_manifest["loops"]
    output_arrays = {
        **_proposal_arrays(proposals, accepted),
        "selected": arrays["selected"],
        "component": arrays["component"],
        "componentSize": final_surface["componentSize"],
        "signedNormalXYZ": final_surface["signedNormalXYZ"],
        "tangentUxyz": final_surface["tangentUxyz"],
        "tangentVxyz": final_surface["tangentVxyz"],
        "chartUV": final_surface["chartUV"],
        "integrationResidualVoxels": final_surface["integrationResidualVoxels"],
        "triangleFrontierIndex": final_surface["triangleFrontierIndex"],
        "triangleAreaVoxelsSquared": final_surface["triangleAreaVoxelsSquared"],
        "triangleNormalResidualDegrees": final_surface[
            "triangleNormalResidualDegrees"
        ],
        "midpointXYZ": final_surface["midpointXYZ"],
        "thicknessVoxels": final_surface["thicknessVoxels"],
    }
    _write_npz(data_path, output_arrays)
    view = {**topology, **arrays}
    world = configuration_manifest["geometry"]["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        view,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "patch-state-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        view,
        output / "largest-patch-state-ribbon-components.png",
    )
    finished = time.monotonic()
    before_triangle = len(np.asarray(baseline_surface["triangleFrontierIndex"]))
    after_triangle = len(np.asarray(final_surface["triangleFrontierIndex"]))
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_STATE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": configuration_manifest["geometry"],
        "patchStates": {
            **solve_stats,
            "acceptedPatchCount": int(np.count_nonzero(accepted)),
            "acceptedAddedRibbonCount": int(
                sum(len(proposals[row]["added"]) for row in np.flatnonzero(accepted))
            ),
            "acceptedRemovedRibbonCount": int(
                sum(len(proposals[row]["removed"]) for row in np.flatnonzero(accepted))
            ),
            "selectedRibbonCountBefore": int(np.count_nonzero(baseline_selected)),
            "selectedRibbonCountAfter": int(np.count_nonzero(final_selected)),
            "interfaceConflictCount": conflict_count[0],
            "crossingConflictCount": conflict_count[1],
            **final_lineage,
            "exactPatchAudits": exact_records,
        },
        "exactTopology": {
            "strictTriangleCountBefore": int(before_triangle),
            "strictTriangleCountAfter": int(after_triangle),
            "strictTriangleCountDelta": int(after_triangle - before_triangle),
            "triangleRegionCountBefore": int(
                baseline_loop_stats["triangleRegionCount"]
            ),
            "triangleRegionCountAfter": final_loop_stats["triangleRegionCount"],
            "triangleRegionCountDelta": int(
                final_loop_stats["triangleRegionCount"]
                - baseline_loop_stats["triangleRegionCount"]
            ),
            "interiorHoleLoopCountBefore": int(
                baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "interiorHoleLoopCountAfter": final_loop_stats[
                "interiorHoleLoopCount"
            ],
            "interiorHoleLoopCountDelta": int(
                final_loop_stats["interiorHoleLoopCount"]
                - baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "macroEligibleHoleCountBefore": int(
                baseline_loop_stats["macroEligibleHoleCount"]
            ),
            "macroEligibleHoleCountAfter": final_loop_stats[
                "macroEligibleHoleCount"
            ],
            "macroEligibleHoleCountDelta": int(
                final_loop_stats["macroEligibleHoleCount"]
                - baseline_loop_stats["macroEligibleHoleCount"]
            ),
        },
        "timingSeconds": {
            "collectivePatchOptimization": round(solved - started, 6),
            "preliminaryExactSurface": round(surfaced - solved, 6),
            "filteredExactSurface": round(finished_surface - surfaced, 6),
            "writingAndPreviews": round(finished - finished_surface, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(output_arrays),
        },
        "artifacts": {
            "componentOverview": overview.name,
            "largestComponentMontage": montage.name,
        },
        "method": {
            "decisionUnit": (
                "all CT-supported hole frontiers on one reconstructed "
                "surface component"
            ),
            "optimization": (
                "collective alternating interface matching inside a fixed "
                "selected halo"
            ),
            "physicalEvidence": (
                "whole-patch native-CT air-material-air profile and "
                "displaced-layer competition"
            ),
            "exactAcceptance": (
                "surface realization, lineage, triangle density, connected "
                "regions, and closed holes"
            ),
            "singleCellGrowth": False,
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
