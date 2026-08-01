from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_continuity import (
    write_continuity_overview,
    write_largest_component_montage,
)
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    build_physical_ribbon_surface_complex,
    extract_surface_boundary_loops,
)


PHYSICAL_RIBBON_COLLECTIVE_SCHEMA = "pareidolia.physical-ribbon-collective"
PHYSICAL_RIBBON_COLLECTIVE_VERSION = 1
PHYSICAL_RIBBON_COLLECTIVE_STEM = "physical-ribbon-collective-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCollectiveSettings:
    """Dataset-independent gates for collective residual-surface proposals.

    A residual ribbon is never admitted as a one-node move.  Candidate ribbons
    first form a two-dimensional continuation region; a conflict-free subset is
    optimized as one patch and is committed only if the complete patch is
    favorable and geometrically supported.
    """

    minimum_residual_candidate_count: int = 8
    minimum_attached_patch_ribbon_count: int = 8
    minimum_isolated_patch_ribbon_count: int = 24
    minimum_boundary_anchor_count: int = 4
    minimum_patch_objective_gain: float = 0.25
    minimum_patch_tangent_ratio: float = 0.025
    minimum_patch_two_core_fraction: float = 0.60
    minimum_surface_realization_fraction: float = 0.80
    minimum_surface_boundary_edge_count: int = 3
    maximum_local_optimization_sweeps: int = 8
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        positive_integer = (
            self.minimum_residual_candidate_count,
            self.minimum_attached_patch_ribbon_count,
            self.minimum_isolated_patch_ribbon_count,
            self.minimum_boundary_anchor_count,
            self.minimum_surface_boundary_edge_count,
            self.maximum_local_optimization_sweeps,
            self.maximum_preview_components,
        )
        if any(value < 1 for value in positive_integer):
            raise ValueError("collective integer settings must be positive")
        if self.minimum_isolated_patch_ribbon_count < (
            self.minimum_attached_patch_ribbon_count
        ):
            raise ValueError("isolated patches cannot be smaller than attached patches")
        if not math.isfinite(self.minimum_patch_objective_gain):
            raise ValueError("patch objective gate must be finite")
        for value in (
            self.minimum_patch_tangent_ratio,
            self.minimum_patch_two_core_fraction,
            self.minimum_surface_realization_fraction,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("collective geometric fractions must lie in [0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _objective(
    selected: np.ndarray,
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
) -> float:
    return float(
        np.sum(node_score[selected])
        + np.sum(edge_weight[selected[edge_first] & selected[edge_second]])
    )


def _conflict_neighbors(
    node_count: int,
    conflict_first: np.ndarray,
    conflict_second: np.ndarray,
) -> tuple[np.ndarray, ...]:
    values: list[list[int]] = [[] for _ in range(node_count)]
    for first, second in zip(conflict_first, conflict_second):
        first_value, second_value = int(first), int(second)
        if first_value == second_value:
            continue
        values[first_value].append(second_value)
        values[second_value].append(first_value)
    return tuple(
        np.asarray(sorted(set(row)), dtype=np.int32) for row in values
    )


def optimize_collective_patch(
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    conflict_first: np.ndarray,
    conflict_second: np.ndarray,
    *,
    maximum_sweeps: int = 8,
    initial_selection: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Optimize one small residual region while crossing the unary barrier.

    Ordinary coordinate ascent cannot discover a surface whose first ribbon is
    unfavorable but whose jointly selected two-dimensional neighborhood is
    strongly favorable.  The deterministic starts below rank nodes using their
    *potential* internal support, construct a conflict-free patch, and only then
    perform exact add/remove/swap coordinate ascent on the true objective.
    """

    node_score = np.asarray(node_score, dtype=np.float64)
    edge_first = np.asarray(edge_first, dtype=np.int32)
    edge_second = np.asarray(edge_second, dtype=np.int32)
    edge_weight = np.asarray(edge_weight, dtype=np.float64)
    conflict_first = np.asarray(conflict_first, dtype=np.int32)
    conflict_second = np.asarray(conflict_second, dtype=np.int32)
    node_count = len(node_score)
    if node_count == 0:
        return np.empty(0, dtype=bool), 0.0
    incident = np.zeros(node_count, dtype=np.float64)
    np.add.at(incident, edge_first, edge_weight)
    np.add.at(incident, edge_second, edge_weight)
    conflicts = _conflict_neighbors(node_count, conflict_first, conflict_second)
    ranked_starts = (
        node_score + 0.25 * incident,
        node_score + 0.50 * incident,
        node_score + incident,
        incident + 0.05 * node_score,
    )
    seeded_starts: list[np.ndarray] = []
    if initial_selection is not None:
        initial = np.asarray(initial_selection, dtype=bool)
        if initial.shape != (node_count,):
            raise ValueError("initial collective selection has the wrong shape")
        for first, second in zip(conflict_first, conflict_second):
            if initial[int(first)] and initial[int(second)]:
                raise ValueError("initial collective selection violates a conflict")
        seeded_starts.append(initial.copy())
    for start_score in ranked_starts:
        selected = np.zeros(node_count, dtype=bool)
        order = np.lexsort((np.arange(node_count), -start_score))
        for node in order:
            if start_score[node] <= 0.0:
                continue
            if len(conflicts[node]) and np.any(selected[conflicts[node]]):
                continue
            selected[node] = True
        seeded_starts.append(selected)
    best = np.zeros(node_count, dtype=bool)
    best_objective = 0.0
    for selected in seeded_starts:
        current = _objective(
            selected, node_score, edge_first, edge_second, edge_weight
        )
        for _ in range(maximum_sweeps):
            changed = False

            # Exact pruning is deliberately one node at a time: removing all
            # negative marginals simultaneously can destroy pair support twice.
            while np.any(selected):
                active_edge = selected[edge_first] & selected[edge_second]
                support = np.zeros(node_count, dtype=np.float64)
                np.add.at(support, edge_first[active_edge], edge_weight[active_edge])
                np.add.at(support, edge_second[active_edge], edge_weight[active_edge])
                marginal = node_score + support
                active = np.flatnonzero(selected)
                worst = int(active[int(np.argmin(marginal[active]))])
                if marginal[worst] >= -1.0e-9:
                    break
                selected[worst] = False
                current -= float(marginal[worst])
                changed = True

            best_delta = 1.0e-9
            best_trial: np.ndarray | None = None
            for node in np.flatnonzero(~selected):
                trial = selected.copy()
                if len(conflicts[node]):
                    trial[conflicts[node]] = False
                trial[node] = True
                value = _objective(
                    trial, node_score, edge_first, edge_second, edge_weight
                )
                delta = value - current
                if delta > best_delta:
                    best_delta = delta
                    best_trial = trial
            if best_trial is not None:
                selected = best_trial
                current += best_delta
                changed = True
            if not changed:
                break
        current = _objective(
            selected, node_score, edge_first, edge_second, edge_weight
        )
        selection_key = (current, int(np.count_nonzero(selected)))
        best_key = (best_objective, int(np.count_nonzero(best)))
        if selection_key > best_key:
            best = selected
            best_objective = current
    return best, float(best_objective)


def _union_components(
    node_count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for left, right in zip(first, second):
        left_root, right_root = find(int(left)), find(int(right))
        if left_root == right_root:
            continue
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]
    root = np.asarray([find(value) for value in range(node_count)], dtype=np.int32)
    _, inverse, count = np.unique(root, return_inverse=True, return_counts=True)
    return inverse.astype(np.int32), count.astype(np.int32)


def _shared_endpoint_conflicts(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    by_endpoint: dict[int, list[int]] = {}
    for node, (first, second) in enumerate(zip(source, target)):
        by_endpoint.setdefault(int(first), []).append(node)
        by_endpoint.setdefault(int(second), []).append(node)
    conflict: set[tuple[int, int]] = set()
    for members in by_endpoint.values():
        for index, first in enumerate(members):
            for second in members[index + 1 :]:
                conflict.add((min(first, second), max(first, second)))
    if not conflict:
        empty = np.empty(0, dtype=np.int32)
        return empty, empty
    ordered = np.asarray(sorted(conflict), dtype=np.int32)
    return ordered[:, 0], ordered[:, 1]


def _selected_subpatches(
    selected: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
) -> list[np.ndarray]:
    chosen = np.flatnonzero(selected)
    if not len(chosen):
        return []
    dense = np.full(len(selected), -1, dtype=np.int32)
    dense[chosen] = np.arange(len(chosen), dtype=np.int32)
    active = selected[edge_first] & selected[edge_second]
    component, count = _union_components(
        len(chosen), dense[edge_first[active]], dense[edge_second[active]]
    )
    order = np.argsort(-count)
    return [chosen[component == value] for value in order]


def _triangle_region_counts(
    surface: Mapping[str, np.ndarray],
) -> dict[int, int]:
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    if not len(triangle):
        return {}
    parent = np.arange(len(triangle), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    edge_owner: dict[tuple[int, int], int] = {}
    for triangle_index, vertices in enumerate(triangle):
        for index, first in enumerate(vertices):
            second = int(vertices[(index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            previous = edge_owner.get(edge)
            if previous is None:
                edge_owner[edge] = triangle_index
                continue
            first_root, second_root = find(previous), find(triangle_index)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)
    root = np.asarray([find(value) for value in range(len(triangle))], dtype=np.int32)
    component = np.asarray(surface["component"], dtype=np.int32)[triangle[:, 0]]
    result: dict[int, int] = {}
    for value in np.unique(component):
        result[int(value)] = int(len(np.unique(root[component == value])))
    return result


def _patch_geometry(
    nodes: np.ndarray,
    midpoint: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[float, float, int, float]:
    member = np.zeros(len(midpoint), dtype=bool)
    member[nodes] = True
    active = member[edge_first] & member[edge_second]
    first = edge_first[active]
    second = edge_second[active]
    weight = edge_weight[active]
    if not len(first):
        return 0.0, 0.0, 0, 0.0
    direction = midpoint[second] - midpoint[first]
    direction /= np.maximum(
        np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-6
    )
    covariance = np.einsum("i,ij,ik->jk", weight, direction, direction)
    eigenvalue = np.linalg.eigvalsh(covariance)
    tangent_ratio = float(eigenvalue[1] / max(float(eigenvalue[2]), 1.0e-6))
    degree = np.zeros(len(midpoint), dtype=np.int32)
    np.add.at(degree, first, 1)
    np.add.at(degree, second, 1)
    two_core = float(np.mean(degree[nodes] >= 2))
    return tangent_ratio, two_core, int(len(first)), float(np.mean(weight))


def build_collective_residual_assignment(
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    settings: PhysicalRibbonCollectiveSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    prior_component = np.asarray(configuration["component"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    unary = np.asarray(configuration["nodeUnaryScore"], dtype=np.float32)
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )

    interface_count = len(np.asarray(ribbon["interfaceCandidateDegree"]))
    used_interface = np.zeros(interface_count, dtype=bool)
    selected_node = np.flatnonzero(selected)
    used_interface[source[selected_node]] = True
    used_interface[target[selected_node]] = True
    free_endpoint = ~used_interface[source] & ~used_interface[target]
    crossing_blocked = np.zeros(len(frontier), dtype=bool)
    np.logical_or.at(
        crossing_blocked,
        crossing_first[selected[crossing_second]],
        True,
    )
    np.logical_or.at(
        crossing_blocked,
        crossing_second[selected[crossing_first]],
        True,
    )
    available = ~selected & free_endpoint & ~crossing_blocked
    available_node = np.flatnonzero(available)
    dense = np.full(len(frontier), -1, dtype=np.int32)
    dense[available_node] = np.arange(len(available_node), dtype=np.int32)
    residual_edge = available[edge_first] & available[edge_second]
    residual_component, residual_count = _union_components(
        len(available_node),
        dense[edge_first[residual_edge]],
        dense[edge_second[residual_edge]],
    )
    component_of_frontier = np.full(len(frontier), -1, dtype=np.int32)
    component_of_frontier[available_node] = residual_component

    boundary_first = available[edge_first] & selected[edge_second]
    boundary_second = available[edge_second] & selected[edge_first]
    boundary_candidate = np.concatenate(
        (edge_first[boundary_first], edge_second[boundary_second])
    )
    boundary_anchor = np.concatenate(
        (edge_second[boundary_first], edge_first[boundary_second])
    )
    boundary_weight = np.concatenate(
        (edge_score[boundary_first], edge_score[boundary_second])
    )
    boundary_score = np.zeros(len(frontier), dtype=np.float32)
    np.add.at(boundary_score, boundary_candidate, boundary_weight)

    internal_crossing = (
        available[crossing_first]
        & available[crossing_second]
        & (
            component_of_frontier[crossing_first]
            == component_of_frontier[crossing_second]
        )
    )
    crossing_by_component: dict[int, list[tuple[int, int]]] = {}
    for first, second in zip(
        crossing_first[internal_crossing], crossing_second[internal_crossing]
    ):
        component_id = int(component_of_frontier[first])
        crossing_by_component.setdefault(component_id, []).append(
            (int(first), int(second))
        )

    residual_global_first = edge_first[residual_edge]
    residual_global_second = edge_second[residual_edge]
    residual_global_weight = edge_score[residual_edge]
    edges_by_component: dict[int, list[int]] = {}
    for index, first in enumerate(residual_global_first):
        component_id = int(component_of_frontier[first])
        edges_by_component.setdefault(component_id, []).append(index)

    node_order = np.argsort(residual_component, kind="stable")
    node_offsets = np.r_[0, np.cumsum(residual_count)].astype(np.int64)
    residual_patch_offset = [0]
    residual_patch_frontier: list[int] = []
    residual_patch_candidate_count: list[int] = []
    residual_patch_collective_upper: list[float] = []
    proposals: list[dict[str, Any]] = []
    small_region_count = 0
    solved_region_count = 0
    rejected_objective_count = 0
    rejected_geometry_count = 0
    rejected_anchor_count = 0

    for component_id, candidate_count in enumerate(residual_count):
        if candidate_count < settings.minimum_residual_candidate_count:
            small_region_count += 1
            continue
        nodes = available_node[
            node_order[node_offsets[component_id] : node_offsets[component_id + 1]]
        ]
        local = np.full(len(frontier), -1, dtype=np.int32)
        local[nodes] = np.arange(len(nodes), dtype=np.int32)
        edge_index = np.asarray(
            edges_by_component.get(component_id, ()), dtype=np.int32
        )
        local_first = local[residual_global_first[edge_index]]
        local_second = local[residual_global_second[edge_index]]
        local_weight = (
            continuity_weight * residual_global_weight[edge_index]
        ).astype(np.float32)
        node_score = (
            unary[nodes] + continuity_weight * boundary_score[nodes]
        ).astype(np.float32)
        shared_first, shared_second = _shared_endpoint_conflicts(
            source[nodes], target[nodes]
        )
        crossing_records = crossing_by_component.get(component_id, ())
        if crossing_records:
            crossing_array = np.asarray(crossing_records, dtype=np.int32)
            local_cross_first = local[crossing_array[:, 0]]
            local_cross_second = local[crossing_array[:, 1]]
            conflict_first = np.concatenate((shared_first, local_cross_first))
            conflict_second = np.concatenate((shared_second, local_cross_second))
        else:
            conflict_first, conflict_second = shared_first, shared_second
        residual_patch_frontier.extend(int(value) for value in nodes)
        residual_patch_offset.append(len(residual_patch_frontier))
        residual_patch_candidate_count.append(int(len(nodes)))
        residual_patch_collective_upper.append(
            _objective(
                np.ones(len(nodes), dtype=bool),
                node_score,
                local_first,
                local_second,
                local_weight,
            )
        )
        chosen, _ = optimize_collective_patch(
            node_score,
            local_first,
            local_second,
            local_weight,
            conflict_first,
            conflict_second,
            maximum_sweeps=settings.maximum_local_optimization_sweeps,
        )
        solved_region_count += 1
        for subpatch in _selected_subpatches(chosen, local_first, local_second):
            member = np.zeros(len(nodes), dtype=bool)
            member[subpatch] = True
            objective_gain = _objective(
                member, node_score, local_first, local_second, local_weight
            )
            if objective_gain < settings.minimum_patch_objective_gain:
                rejected_objective_count += 1
                continue
            global_nodes = nodes[subpatch]
            anchor_member = np.isin(
                boundary_candidate, global_nodes, assume_unique=False
            )
            anchors = np.unique(boundary_anchor[anchor_member])
            anchor_components = np.unique(prior_component[anchors])
            anchor_components = anchor_components[anchor_components >= 0]
            if len(anchor_components) > 1:
                rejected_anchor_count += 1
                continue
            minimum_count = (
                settings.minimum_attached_patch_ribbon_count
                if len(anchor_components) == 1
                else settings.minimum_isolated_patch_ribbon_count
            )
            if len(global_nodes) < minimum_count or (
                len(anchor_components) == 1
                and len(anchors) < settings.minimum_boundary_anchor_count
            ):
                rejected_anchor_count += 1
                continue
            tangent_ratio, two_core, selected_edge_count, mean_edge_score = (
                _patch_geometry(
                    global_nodes,
                    midpoint,
                    edge_first,
                    edge_second,
                    edge_score,
                )
            )
            if (
                tangent_ratio < settings.minimum_patch_tangent_ratio
                or two_core < settings.minimum_patch_two_core_fraction
            ):
                rejected_geometry_count += 1
                continue
            proposals.append(
                {
                    "residualComponent": component_id,
                    "nodes": global_nodes,
                    "objectiveGain": float(objective_gain),
                    "anchorComponent": (
                        int(anchor_components[0]) if len(anchor_components) else -1
                    ),
                    "anchorCount": int(len(anchors)),
                    "tangentRatio": tangent_ratio,
                    "twoCoreFraction": two_core,
                    "selectedEdgeCount": selected_edge_count,
                    "meanEdgeScore": mean_edge_score,
                }
            )

    proposals.sort(
        key=lambda record: (
            float(record["objectiveGain"]),
            len(record["nodes"]),
        ),
        reverse=True,
    )
    final_selected = selected.copy()
    final_used_interface = used_interface.copy()
    accepted_proposal: list[bool] = []
    rejection_reason: list[int] = []
    proposal_offset = [0]
    proposal_node: list[int] = []
    accepted_count = 0
    accepted_ribbon_count = 0
    for proposal in proposals:
        nodes = np.asarray(proposal["nodes"], dtype=np.int32)
        proposal_node.extend(int(value) for value in nodes)
        proposal_offset.append(len(proposal_node))
        endpoint_conflict = bool(
            np.any(final_used_interface[source[nodes]])
            or np.any(final_used_interface[target[nodes]])
        )
        crossing_conflict = np.zeros(len(frontier), dtype=bool)
        np.logical_or.at(
            crossing_conflict,
            crossing_first[final_selected[crossing_second]],
            True,
        )
        np.logical_or.at(
            crossing_conflict,
            crossing_second[final_selected[crossing_first]],
            True,
        )
        blocked = endpoint_conflict or bool(np.any(crossing_conflict[nodes]))
        if blocked:
            accepted_proposal.append(False)
            rejection_reason.append(1 if endpoint_conflict else 2)
            continue
        final_selected[nodes] = True
        final_used_interface[source[nodes]] = True
        final_used_interface[target[nodes]] = True
        accepted_proposal.append(True)
        rejection_reason.append(0)
        accepted_count += 1
        accepted_ribbon_count += len(nodes)

    final_component, final_component_size = _component_labels(
        final_selected, edge_first, edge_second
    )
    selected_endpoint = np.concatenate(
        (source[final_selected], target[final_selected])
    )
    interface_conflict_count = int(
        len(selected_endpoint) - len(np.unique(selected_endpoint))
    )
    selected_crossing_count = int(
        np.count_nonzero(
            final_selected[crossing_first] & final_selected[crossing_second]
        )
    )
    prior_to_final: dict[int, set[int]] = {}
    for prior_id, final_id in zip(
        prior_component[selected], final_component[selected]
    ):
        prior_to_final.setdefault(int(prior_id), set()).add(int(final_id))
    final_to_prior: dict[int, set[int]] = {}
    for prior_id, final_id in zip(
        prior_component[selected], final_component[selected]
    ):
        final_to_prior.setdefault(int(final_id), set()).add(int(prior_id))
    split_count = sum(len(values) > 1 for values in prior_to_final.values())
    fusion_count = sum(len(values) > 1 for values in final_to_prior.values())
    if interface_conflict_count or selected_crossing_count or split_count or fusion_count:
        raise RuntimeError("collective assignment violated an exact baseline invariant")

    arrays = {
        "residualPatchOffset": np.asarray(residual_patch_offset, dtype=np.int64),
        "residualPatchFrontierIndex": np.asarray(
            residual_patch_frontier, dtype=np.int32
        ),
        "residualPatchCandidateCount": np.asarray(
            residual_patch_candidate_count, dtype=np.int32
        ),
        "residualPatchCollectiveUpperObjective": np.asarray(
            residual_patch_collective_upper, dtype=np.float32
        ),
        "proposalOffset": np.asarray(proposal_offset, dtype=np.int64),
        "proposalFrontierIndex": np.asarray(proposal_node, dtype=np.int32),
        "proposalResidualComponent": np.asarray(
            [record["residualComponent"] for record in proposals], dtype=np.int32
        ),
        "proposalObjectiveGain": np.asarray(
            [record["objectiveGain"] for record in proposals], dtype=np.float32
        ),
        "proposalAnchorComponent": np.asarray(
            [record["anchorComponent"] for record in proposals], dtype=np.int32
        ),
        "proposalAnchorCount": np.asarray(
            [record["anchorCount"] for record in proposals], dtype=np.int32
        ),
        "proposalTangentRatio": np.asarray(
            [record["tangentRatio"] for record in proposals], dtype=np.float32
        ),
        "proposalTwoCoreFraction": np.asarray(
            [record["twoCoreFraction"] for record in proposals], dtype=np.float32
        ),
        "proposalSelectedEdgeCount": np.asarray(
            [record["selectedEdgeCount"] for record in proposals], dtype=np.int32
        ),
        "proposalMeanEdgeScore": np.asarray(
            [record["meanEdgeScore"] for record in proposals], dtype=np.float32
        ),
        "proposalAccepted": np.asarray(accepted_proposal, dtype=np.uint8),
        "proposalRejectionReason": np.asarray(rejection_reason, dtype=np.uint8),
        "selected": final_selected.astype(np.uint8),
        "component": final_component.astype(np.int32),
    }
    stats = {
        "frontierCandidateCount": int(len(frontier)),
        "frontierInterfaceCount": int(len(np.unique(np.concatenate((source, target))))),
        "selectedRibbonCountBefore": int(np.count_nonzero(selected)),
        "selectedInterfaceCountBefore": int(2 * np.count_nonzero(selected)),
        "freeEndpointCandidateCount": int(np.count_nonzero(free_endpoint)),
        "baselineCrossingBlockedCandidateCount": int(
            np.count_nonzero(free_endpoint & crossing_blocked)
        ),
        "availableResidualCandidateCount": int(len(available_node)),
        "residualContinuationEdgeCount": int(np.count_nonzero(residual_edge)),
        "residualRegionCount": int(len(residual_count)),
        "residualRegionWithMinimumCandidatesCount": int(
            np.count_nonzero(
                residual_count >= settings.minimum_residual_candidate_count
            )
        ),
        "largestResidualRegionCandidateCounts": [
            int(value) for value in np.sort(residual_count)[-32:][::-1]
        ],
        "smallResidualRegionCount": int(small_region_count),
        "solvedResidualRegionCount": int(solved_region_count),
        "qualifiedProposalCount": int(len(proposals)),
        "acceptedProposalCount": int(accepted_count),
        "acceptedRibbonCount": int(accepted_ribbon_count),
        "rejectedObjectiveSubpatchCount": int(rejected_objective_count),
        "rejectedGeometrySubpatchCount": int(rejected_geometry_count),
        "rejectedAnchorSubpatchCount": int(rejected_anchor_count),
        "rejectedGlobalConflictProposalCount": int(
            len(proposals) - accepted_count
        ),
        "selectedRibbonCountAfter": int(np.count_nonzero(final_selected)),
        "selectedInterfaceCountAfter": int(len(selected_endpoint)),
        "frontierInterfaceUtilizationBefore": round(
            float(2 * np.count_nonzero(selected))
            / max(float(len(np.unique(np.concatenate((source, target))))), 1.0),
            6,
        ),
        "frontierInterfaceUtilizationAfter": round(
            float(len(selected_endpoint))
            / max(float(len(np.unique(np.concatenate((source, target))))), 1.0),
            6,
        ),
        "componentCountBefore": int(len(np.unique(prior_component[selected]))),
        "componentCountAfter": int(len(final_component_size)),
        "largestComponentRibbonCountsAfter": [
            int(value) for value in final_component_size[:32]
        ],
        "interfaceConflictCount": interface_conflict_count,
        "crossingConflictCount": selected_crossing_count,
        "splitPriorComponentCount": int(split_count),
        "crossPriorComponentFusionCount": int(fusion_count),
        "singleNodeGrowth": False,
        "identityLabelsUsed": False,
    }
    return arrays, stats


def _filter_unrealized_surface_proposals(
    arrays: dict[str, np.ndarray],
    statistics: dict[str, Any],
    surface: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    baseline_configuration: Mapping[str, np.ndarray],
    baseline_surface: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonCollectiveSettings,
) -> bool:
    """Remove graph-positive patches that do not become attached mesh area."""

    proposal_offset = np.asarray(arrays["proposalOffset"], dtype=np.int64)
    proposal_node = np.asarray(arrays["proposalFrontierIndex"], dtype=np.int32)
    accepted = np.asarray(arrays["proposalAccepted"], dtype=np.uint8) > 0
    anchor_component = np.asarray(
        arrays["proposalAnchorComponent"], dtype=np.int32
    )
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    triangle_node = np.zeros(len(arrays["selected"]), dtype=bool)
    if len(triangle):
        triangle_node[np.unique(triangle)] = True
        mesh_edge = np.unique(
            np.sort(
                np.concatenate(
                    (
                        triangle[:, (0, 1)],
                        triangle[:, (1, 2)],
                        triangle[:, (2, 0)],
                    ),
                    axis=0,
                ),
                axis=1,
            ),
            axis=0,
        )
    else:
        mesh_edge = np.empty((0, 2), dtype=np.int32)
    baseline_selected = (
        np.asarray(baseline_configuration["selected"], dtype=np.uint8) > 0
    )
    baseline_surface_node = np.zeros(len(baseline_selected), dtype=bool)
    baseline_triangle = np.asarray(
        baseline_surface["triangleFrontierIndex"], dtype=np.int32
    )
    if len(baseline_triangle):
        baseline_surface_node[np.unique(baseline_triangle)] = True
    realization = np.zeros(len(accepted), dtype=np.float32)
    boundary_count = np.zeros(len(accepted), dtype=np.int32)
    region_delta = np.zeros(len(accepted), dtype=np.int32)
    rejected = np.zeros(len(accepted), dtype=bool)
    for row in np.flatnonzero(accepted):
        nodes = proposal_node[proposal_offset[row] : proposal_offset[row + 1]]
        if not len(nodes):
            rejected[row] = True
            continue
        realization[row] = float(np.mean(triangle_node[nodes]))
        member = np.zeros(len(triangle_node), dtype=bool)
        member[nodes] = True
        if len(mesh_edge):
            boundary = (
                (member[mesh_edge[:, 0]] & baseline_surface_node[mesh_edge[:, 1]])
                | (member[mesh_edge[:, 1]] & baseline_surface_node[mesh_edge[:, 0]])
            )
            boundary_count[row] = int(np.count_nonzero(boundary))
        rejected[row] = (
            realization[row] < settings.minimum_surface_realization_fraction
            or (
                anchor_component[row] >= 0
                and boundary_count[row]
                < settings.minimum_surface_boundary_edge_count
            )
        )
    baseline_region_count = _triangle_region_counts(baseline_surface)
    final_region_count = _triangle_region_counts(surface)
    baseline_component = np.asarray(
        baseline_configuration["component"], dtype=np.int32
    )
    final_component = np.asarray(arrays["component"], dtype=np.int32)
    baseline_selected_node = np.flatnonzero(baseline_selected)
    component_mapping: dict[int, int] = {}
    for component_id in np.unique(baseline_component[baseline_selected]):
        member = baseline_selected_node[
            baseline_component[baseline_selected_node] == component_id
        ]
        value, count = np.unique(final_component[member], return_counts=True)
        component_mapping[int(component_id)] = int(value[int(np.argmax(count))])
    for row in np.flatnonzero(accepted):
        prior_id = int(anchor_component[row])
        if prior_id < 0:
            continue
        final_id = component_mapping[prior_id]
        region_delta[row] = int(
            final_region_count.get(final_id, 0)
            - baseline_region_count.get(prior_id, 0)
        )
        if region_delta[row] > 0:
            rejected[row] = True
    arrays["proposalSurfaceRealizationFraction"] = realization
    arrays["proposalSurfaceBoundaryEdgeCount"] = boundary_count
    arrays["proposalSurfaceRegionCountDelta"] = region_delta
    if not np.any(rejected):
        statistics["surfaceRejectedProposalCount"] = 0
        statistics["surfaceRejectedRibbonCount"] = 0
        return False

    selected = np.asarray(arrays["selected"], dtype=np.uint8) > 0
    rejected_ribbon_count = 0
    for row in np.flatnonzero(rejected):
        nodes = proposal_node[proposal_offset[row] : proposal_offset[row + 1]]
        selected[nodes] = False
        rejected_ribbon_count += len(nodes)
    accepted[rejected] = False
    rejection_reason = np.asarray(
        arrays["proposalRejectionReason"], dtype=np.uint8
    ).copy()
    rejection_reason[rejected] = 3
    edge_first = np.asarray(
        topology["edgeFirstFrontierIndex"], dtype=np.int32
    )
    edge_second = np.asarray(
        topology["edgeSecondFrontierIndex"], dtype=np.int32
    )
    component, component_size = _component_labels(
        selected, edge_first, edge_second
    )
    arrays["proposalAccepted"] = accepted.astype(np.uint8)
    arrays["proposalRejectionReason"] = rejection_reason
    arrays["selected"] = selected.astype(np.uint8)
    arrays["component"] = component.astype(np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    frontier_interface_count = int(
        len(np.unique(np.concatenate((source, target))))
    )
    selected_count = int(np.count_nonzero(selected))
    statistics.update(
        {
            "acceptedProposalCount": int(np.count_nonzero(accepted)),
            "acceptedRibbonCount": int(
                sum(
                    int(proposal_offset[row + 1] - proposal_offset[row])
                    for row in np.flatnonzero(accepted)
                )
            ),
            "surfaceRejectedProposalCount": int(np.count_nonzero(rejected)),
            "surfaceRejectedRibbonCount": int(rejected_ribbon_count),
            "selectedRibbonCountAfter": selected_count,
            "selectedInterfaceCountAfter": 2 * selected_count,
            "frontierInterfaceUtilizationAfter": round(
                2.0 * selected_count / max(float(frontier_interface_count), 1.0),
                6,
            ),
            "componentCountAfter": int(len(component_size)),
            "largestComponentRibbonCountsAfter": [
                int(value) for value in component_size[:32]
            ],
        }
    )
    return True


def run_physical_ribbon_collective(
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCollectiveSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCollectiveSettings()
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
    ) = _load_inputs(configuration_root)
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    identity = {
        "schema": PHYSICAL_RIBBON_COLLECTIVE_SCHEMA,
        "version": PHYSICAL_RIBBON_COLLECTIVE_VERSION,
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "topologyContinuity": {
            "manifestPath": str(topology_path),
            "manifestSha256": sha256_file(topology_path),
            "dataSha256": topology_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "continuityWeight": continuity_weight,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_COLLECTIVE_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_COLLECTIVE_STEM}.npz"
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
    surface_settings = PhysicalRibbonPatchHoleSettings()
    baseline_surface, baseline_surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        topology,
        configuration,
        settings=surface_settings,
    )
    _, baseline_loop_stats = extract_surface_boundary_loops(
        baseline_surface, settings=surface_settings
    )
    baselined = time.monotonic()
    arrays, collective_stats = build_collective_residual_assignment(
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        settings=resolved,
    )
    solved = time.monotonic()
    surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        topology,
        arrays,
        settings=surface_settings,
    )
    preliminary_surface_at = time.monotonic()
    surface_filtered = _filter_unrealized_surface_proposals(
        arrays,
        collective_stats,
        surface,
        ribbon,
        topology,
        configuration,
        baseline_surface,
        settings=resolved,
    )
    if surface_filtered:
        surface, surface_stats = build_physical_ribbon_surface_complex(
            ribbon,
            topology,
            arrays,
            settings=surface_settings,
        )
    _, final_loop_stats = extract_surface_boundary_loops(
        surface, settings=surface_settings
    )
    surfaced = time.monotonic()
    output_arrays = {
        **arrays,
        "componentSize": surface["componentSize"],
        "signedNormalXYZ": surface["signedNormalXYZ"],
        "tangentUxyz": surface["tangentUxyz"],
        "tangentVxyz": surface["tangentVxyz"],
        "chartUV": surface["chartUV"],
        "integrationResidualVoxels": surface["integrationResidualVoxels"],
        "triangleFrontierIndex": surface["triangleFrontierIndex"],
        "triangleAreaVoxelsSquared": surface["triangleAreaVoxelsSquared"],
        "triangleNormalResidualDegrees": surface["triangleNormalResidualDegrees"],
        "midpointXYZ": surface["midpointXYZ"],
        "thicknessVoxels": surface["thicknessVoxels"],
    }
    _write_npz(data_path, output_arrays)
    view = {**topology, **arrays}
    world = configuration_manifest["geometry"]["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        view,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "collective-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        view,
        output / "largest-collective-ribbon-components.png",
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_COLLECTIVE_SCHEMA,
        "version": PHYSICAL_RIBBON_COLLECTIVE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": configuration_manifest["geometry"],
        "collective": collective_stats,
        "surface": surface_stats,
        "exactTopology": {
            "strictTriangleCountBefore": int(
                baseline_surface_stats["triangulation"]["retainedTriangles"]
            ),
            "strictTriangleCountAfter": int(
                surface_stats["triangulation"]["retainedTriangles"]
            ),
            "strictTriangleCountDelta": int(
                surface_stats["triangulation"]["retainedTriangles"]
                - baseline_surface_stats["triangulation"]["retainedTriangles"]
            ),
            "triangleRegionCountBefore": int(
                baseline_loop_stats["triangleRegionCount"]
            ),
            "triangleRegionCountAfter": int(
                final_loop_stats["triangleRegionCount"]
            ),
            "triangleRegionCountDelta": int(
                final_loop_stats["triangleRegionCount"]
                - baseline_loop_stats["triangleRegionCount"]
            ),
            "interiorHoleLoopCountBefore": int(
                baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "interiorHoleLoopCountAfter": int(
                final_loop_stats["interiorHoleLoopCount"]
            ),
            "interiorHoleLoopCountDelta": int(
                final_loop_stats["interiorHoleLoopCount"]
                - baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "macroEligibleHoleCountBefore": int(
                baseline_loop_stats["macroEligibleHoleCount"]
            ),
            "macroEligibleHoleCountAfter": int(
                final_loop_stats["macroEligibleHoleCount"]
            ),
            "macroEligibleHoleCountDelta": int(
                final_loop_stats["macroEligibleHoleCount"]
                - baseline_loop_stats["macroEligibleHoleCount"]
            ),
            "pinchedBoundaryComponentCountBefore": int(
                baseline_loop_stats["pinchedBoundaryComponentCount"]
            ),
            "pinchedBoundaryComponentCountAfter": int(
                final_loop_stats["pinchedBoundaryComponentCount"]
            ),
            "unresolvedBoundaryFanCountBefore": int(
                baseline_loop_stats["unresolvedBoundaryFanCount"]
            ),
            "unresolvedBoundaryFanCountAfter": int(
                final_loop_stats["unresolvedBoundaryFanCount"]
            ),
        },
        "timingSeconds": {
            "baselineExactSurface": round(baselined - started, 6),
            "residualCensusAndOptimization": round(solved - baselined, 6),
            "preliminaryExactSurface": round(preliminary_surface_at - solved, 6),
            "surfaceRealizationFilterAndFinalExactSurface": round(
                surfaced - preliminary_surface_at, 6
            ),
            "writingAndPreviews": round(finished - surfaced, 6),
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
                "a conflict-free multi-ribbon residual continuation patch"
            ),
            "objective": (
                "configuration unary evidence plus selected-boundary and "
                "within-patch two-dimensional continuation support"
            ),
            "hardConstraints": (
                "interface exclusivity, profile non-crossing, one prior sheet "
                "per attached patch, and exact prior-component lineage"
            ),
            "singleNodeGrowth": False,
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
