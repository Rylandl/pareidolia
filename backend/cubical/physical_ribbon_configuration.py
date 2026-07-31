from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bank import PHYSICAL_RIBBON_BANK_SCHEMA
from .physical_ribbon_continuity import (
    PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
    write_continuity_overview,
    write_largest_component_montage,
)


PHYSICAL_RIBBON_CONFIGURATION_SCHEMA = "pareidolia.physical-ribbon-configuration"
PHYSICAL_RIBBON_CONFIGURATION_VERSION = 1
PHYSICAL_RIBBON_CONFIGURATION_STEM = "physical-ribbon-configuration-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonConfigurationSettings:
    profile_crossing_distance_voxels: float = 0.75
    profile_crossing_endpoint_margin_fraction: float = 0.08
    node_selection_cost: float = 0.62
    ray_rank_penalty: float = 0.06
    mutual_first_hit_bonus: float = 0.40
    continuity_weight: float = 0.45
    minimum_swap_gain: float = 0.02
    maximum_optimization_sweeps: int = 4
    minimum_hole_growth_neighbors: int = 3
    minimum_hole_growth_mean_edge_score: float = 0.20
    minimum_hole_growth_tangent_ratio: float = 0.08
    maximum_hole_growth_sweeps: int = 3
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        positive = (
            self.profile_crossing_distance_voxels,
            self.node_selection_cost,
            self.ray_rank_penalty,
            self.mutual_first_hit_bonus,
            self.continuity_weight,
            self.minimum_swap_gain,
            self.minimum_hole_growth_mean_edge_score,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("configuration scales must be finite and positive")
        if not 0.0 < self.profile_crossing_endpoint_margin_fraction < 0.5:
            raise ValueError("crossing endpoint margin must lie in (0, 0.5)")
        if self.maximum_optimization_sweeps < 1:
            raise ValueError("optimization sweep count must be positive")
        if self.minimum_hole_growth_neighbors < 2:
            raise ValueError("hole growth requires at least two neighbors")
        if not 0.0 <= self.minimum_hole_growth_tangent_ratio <= 1.0:
            raise ValueError("hole-growth tangent ratio must lie in [0, 1]")
        if self.maximum_hole_growth_sweeps < 1:
            raise ValueError("hole-growth sweep count must be positive")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": round(float(np.min(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "p99": round(float(np.percentile(finite, 99)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def _adjacency(
    node_count: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.concatenate((first, second)).astype(np.int32)
    neighbor = np.concatenate((second, first)).astype(np.int32)
    if weight is None:
        doubled_weight = np.ones(len(source), dtype=np.float32)
    else:
        doubled_weight = np.concatenate((weight, weight)).astype(np.float32)
    order = np.argsort(source, kind="stable")
    source = source[order]
    neighbor = neighbor[order]
    doubled_weight = doubled_weight[order]
    count = np.bincount(source, minlength=node_count)
    offset = np.r_[0, np.cumsum(count)].astype(np.int64)
    return offset, neighbor, doubled_weight


def _rasterized_profile_pairs(
    first_xyz: np.ndarray,
    second_xyz: np.ndarray,
    *,
    processing_world_start_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    sampling_stride_voxels: int,
) -> tuple[np.ndarray, np.ndarray, int]:
    node_count = len(first_xyz)
    length = np.linalg.norm(second_xyz - first_xyz, axis=1)
    sample_count = max(
        int(math.ceil(float(np.max(length)) / sampling_stride_voxels)) + 1,
        3,
    )
    parameter = np.linspace(0.04, 0.96, sample_count, dtype=np.float32)
    shape = tuple(int(value) for value in processing_shape_sampling_xyz)
    half = 0.5 * (sampling_stride_voxels - 1)
    origin = np.asarray(processing_world_start_xyz, dtype=np.float32) + half
    occupancy: list[np.ndarray] = []
    batch_size = 20_000
    for begin in range(0, node_count, batch_size):
        end = min(begin + batch_size, node_count)
        point = (
            first_xyz[begin:end, None, :] * (1.0 - parameter[None, :, None])
            + second_xyz[begin:end, None, :] * parameter[None, :, None]
        )
        key = np.rint(
            (point - origin[None, None, :]) / sampling_stride_voxels
        ).astype(np.int32)
        inside = np.all(
            (key >= 0) & (key < np.asarray(shape)[None, None, :]), axis=2
        )
        node = np.repeat(
            np.arange(begin, end, dtype=np.int32), sample_count
        )[inside.ravel()]
        cell = np.ravel_multi_index(
            key.reshape(-1, 3)[inside.ravel()].T, shape
        )
        occupancy.append(
            np.unique(cell.astype(np.int64) * node_count + node)
        )
    occupied = np.concatenate(occupancy)
    order = np.argsort(occupied)
    occupied = occupied[order]
    cell = occupied // node_count
    node = (occupied % node_count).astype(np.int32)
    _, count = np.unique(cell, return_counts=True)
    maximum_count = int(np.max(count)) if len(count) else 0
    candidate_first: list[np.ndarray] = []
    candidate_second: list[np.ndarray] = []
    for separation in range(1, maximum_count):
        together = cell[:-separation] == cell[separation:]
        candidate_first.append(node[:-separation][together])
        candidate_second.append(node[separation:][together])
    if not candidate_first:
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            int(len(occupied)),
        )
    first = np.concatenate(candidate_first)
    second = np.concatenate(candidate_second)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    pair_key = np.unique(low.astype(np.int64) * node_count + high)
    return (
        (pair_key // node_count).astype(np.int32),
        (pair_key % node_count).astype(np.int32),
        int(len(occupied)),
    )


def _segment_proximity(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    first_axis = first_stop - first_start
    second_axis = second_stop - second_start
    offset = first_start - second_start
    aa = np.einsum("ij,ij->i", first_axis, first_axis)
    bb = np.einsum("ij,ij->i", first_axis, second_axis)
    cc = np.einsum("ij,ij->i", second_axis, second_axis)
    dd = np.einsum("ij,ij->i", first_axis, offset)
    ee = np.einsum("ij,ij->i", second_axis, offset)
    denominator = aa * cc - bb * bb
    parallel = np.abs(denominator) < 1.0e-8
    safe_denominator = np.where(parallel, 1.0, denominator)
    first_parameter = (bb * ee - cc * dd) / safe_denominator
    second_parameter = (aa * ee - bb * dd) / safe_denominator
    first_parameter[parallel] = 0.5
    second_parameter[parallel] = (
        ee[parallel] + bb[parallel] * first_parameter[parallel]
    ) / np.maximum(cc[parallel], 1.0e-8)
    first_parameter = np.clip(first_parameter, 0.0, 1.0)
    second_parameter = np.clip(second_parameter, 0.0, 1.0)
    for _ in range(2):
        second_parameter = np.clip(
            (ee + bb * first_parameter) / np.maximum(cc, 1.0e-8),
            0.0,
            1.0,
        )
        first_parameter = np.clip(
            (bb * second_parameter - dd) / np.maximum(aa, 1.0e-8),
            0.0,
            1.0,
        )
    separation = (
        offset
        + first_parameter[:, None] * first_axis
        - second_parameter[:, None] * second_axis
    )
    return (
        np.linalg.norm(separation, axis=1),
        first_parameter,
        second_parameter,
    )


def build_profile_crossing_conflicts(
    ribbon: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    *,
    processing_world_start_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    sampling_stride_voxels: int,
    settings: PhysicalRibbonConfigurationSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    position = np.asarray(interfaces["positionXYZ"], dtype=np.float32)
    first_xyz = position[source]
    second_xyz = position[target]
    pair_first, pair_second, occupancy_count = _rasterized_profile_pairs(
        first_xyz,
        second_xyz,
        processing_world_start_xyz=processing_world_start_xyz,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        sampling_stride_voxels=sampling_stride_voxels,
    )
    node_count = len(frontier)
    continuity_first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    continuity_second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    continuity_key = np.unique(
        np.minimum(continuity_first, continuity_second).astype(np.int64)
        * node_count
        + np.maximum(continuity_first, continuity_second)
    )
    pair_key = pair_first.astype(np.int64) * node_count + pair_second
    continuation = np.isin(pair_key, continuity_key, assume_unique=True)
    shared_interface = (
        (source[pair_first] == source[pair_second])
        | (source[pair_first] == target[pair_second])
        | (target[pair_first] == source[pair_second])
        | (target[pair_first] == target[pair_second])
    )
    eligible = ~continuation & ~shared_interface
    pair_first = pair_first[eligible]
    pair_second = pair_second[eligible]
    distance, first_parameter, second_parameter = _segment_proximity(
        first_xyz[pair_first],
        second_xyz[pair_first],
        first_xyz[pair_second],
        second_xyz[pair_second],
    )
    margin = settings.profile_crossing_endpoint_margin_fraction
    crossing = (
        (distance <= settings.profile_crossing_distance_voxels)
        & (first_parameter >= margin)
        & (first_parameter <= 1.0 - margin)
        & (second_parameter >= margin)
        & (second_parameter <= 1.0 - margin)
    )
    arrays = {
        "crossingFirstFrontierIndex": pair_first[crossing],
        "crossingSecondFrontierIndex": pair_second[crossing],
        "crossingDistanceVoxels": distance[crossing].astype(np.float32),
        "crossingFirstParameter": first_parameter[crossing].astype(np.float32),
        "crossingSecondParameter": second_parameter[crossing].astype(np.float32),
    }
    return arrays, {
        "rasterizedProfileOccupancyCount": occupancy_count,
        "cooccupiedProfilePairCount": int(np.count_nonzero(eligible)),
        "exactInteriorCrossingConflictCount": int(np.count_nonzero(crossing)),
        "crossingDistanceVoxels": _percentiles(distance[crossing]),
    }


def _selected_continuity_support(
    selected: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
) -> np.ndarray:
    active = selected[first] & selected[second]
    support = np.zeros(len(selected), dtype=np.float32)
    np.add.at(support, first[active], weight[active])
    np.add.at(support, second[active], weight[active])
    return support


def _configuration_objective(
    selected: np.ndarray,
    unary: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
    continuity_weight: float,
) -> float:
    return float(
        np.sum(unary[selected])
        + continuity_weight
        * np.sum(weight[selected[first] & selected[second]])
    )


def _component_labels(
    selected: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    node_count = len(selected)
    parent = np.arange(node_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    active = selected[first] & selected[second]
    for first_node, second_node in zip(first[active], second[active]):
        first_root = find(int(first_node))
        second_root = find(int(second_node))
        if first_root == second_root:
            continue
        if first_root > second_root:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
    selected_node = np.flatnonzero(selected)
    root = np.asarray([find(int(value)) for value in selected_node])
    _, inverse, size = np.unique(root, return_inverse=True, return_counts=True)
    size_order = np.argsort(-size)
    rank = np.empty(len(size_order), dtype=np.int32)
    rank[size_order] = np.arange(len(size_order))
    component = np.full(node_count, -1, dtype=np.int32)
    component[selected_node] = rank[inverse]
    return component, size[size_order]


def _undirected_edge_key(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    first_u64 = np.asarray(first, dtype=np.uint64)
    second_u64 = np.asarray(second, dtype=np.uint64)
    low = np.minimum(first_u64, second_u64)
    high = np.maximum(first_u64, second_u64)
    return (low << np.uint64(32)) | high


def _support_topology_edge_mask(
    support_first: np.ndarray,
    support_second: np.ndarray,
    topology_first: np.ndarray,
    topology_second: np.ndarray,
) -> np.ndarray:
    support_key = _undirected_edge_key(support_first, support_second)
    topology_key = _undirected_edge_key(topology_first, topology_second)
    ordered_support = np.sort(support_key)
    if len(ordered_support) > 1 and np.any(
        ordered_support[1:] == ordered_support[:-1]
    ):
        raise ValueError("support continuity contains duplicate edges")
    ordered_topology = np.sort(topology_key)
    if len(ordered_topology) > 1 and np.any(
        ordered_topology[1:] == ordered_topology[:-1]
    ):
        raise ValueError("topology continuity contains duplicate edges")
    location = np.searchsorted(ordered_support, topology_key)
    present = location < len(ordered_support)
    present[present] &= (
        ordered_support[location[present]] == topology_key[present]
    )
    if not np.all(present):
        raise ValueError("topology continuity is not a subset of support continuity")
    return np.isin(support_key, topology_key, assume_unique=True)


def optimize_physical_ribbon_configuration(
    ribbon: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    crossings: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonConfigurationSettings,
    topology_continuity: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    node_count = len(frontier)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    physical_score = np.asarray(
        ribbon["physicalEvidenceScore"], dtype=np.float32
    )[frontier]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    source_rank = np.asarray(ribbon["sourceRayRank"], dtype=np.int32)[frontier]
    target_rank = np.asarray(ribbon["targetRayRank"], dtype=np.int32)[frontier]
    mutual = np.asarray(ribbon["mutualFirstHit"])[frontier] > 0
    first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    edge_weight = np.asarray(continuity["edgeScore"], dtype=np.float32)
    topology = topology_continuity or continuity
    if not np.array_equal(
        frontier,
        np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32),
    ):
        raise ValueError("support and topology continuity frontiers differ")
    topology_first = np.asarray(
        topology["edgeFirstFrontierIndex"], dtype=np.int32
    )
    topology_second = np.asarray(
        topology["edgeSecondFrontierIndex"], dtype=np.int32
    )
    topology_weight = np.asarray(topology["edgeScore"], dtype=np.float32)
    topology_edge_mask = _support_topology_edge_mask(
        first,
        second,
        topology_first,
        topology_second,
    )
    crossing_first = np.asarray(
        crossings["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        crossings["crossingSecondFrontierIndex"], dtype=np.int32
    )
    continuity_offset, continuity_neighbor, continuity_neighbor_weight = _adjacency(
        node_count, first, second, edge_weight
    )
    topology_offset, topology_neighbor, topology_neighbor_weight = _adjacency(
        node_count, topology_first, topology_second, topology_weight
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        node_count, crossing_first, crossing_second
    )
    unary = (
        physical_score
        - settings.node_selection_cost
        - settings.ray_rank_penalty * (source_rank + target_rank)
        + settings.mutual_first_hit_bonus * mutual
    ).astype(np.float32)
    initial = np.asarray(continuity["selected"]) > 0
    initial_support = _selected_continuity_support(
        initial, first, second, edge_weight
    )
    retention = unary + settings.continuity_weight * initial_support

    # First produce a physically feasible subset of the incoming configuration.
    # Crossing profiles are alternatives even if both are individually smooth.
    selected = np.zeros(node_count, dtype=bool)
    used_interface = np.zeros(len(interfaces["positionXYZ"]), dtype=bool)
    order = np.flatnonzero(initial)
    order = order[np.argsort(-retention[order])]
    for node in order:
        if used_interface[source[node]] or used_interface[target[node]]:
            continue
        begin, end = crossing_offset[node], crossing_offset[node + 1]
        if np.any(selected[crossing_neighbor[begin:end]]):
            continue
        selected[node] = True
        used_interface[source[node]] = True
        used_interface[target[node]] = True
    crossing_clean_count = int(np.count_nonzero(initial) - np.count_nonzero(selected))

    sweep_records: list[dict[str, Any]] = []
    previous_objective = _configuration_objective(
        selected,
        unary,
        first,
        second,
        edge_weight,
        settings.continuity_weight,
    )
    for sweep in range(settings.maximum_optimization_sweeps):
        interface_owner = np.full(len(interfaces["positionXYZ"]), -1, dtype=np.int32)
        selected_node = np.flatnonzero(selected)
        interface_owner[source[selected_node]] = selected_node
        interface_owner[target[selected_node]] = selected_node
        support = _selected_continuity_support(
            selected, first, second, edge_weight
        )
        estimated_gain = unary + settings.continuity_weight * support
        candidate_order = np.argsort(-estimated_gain)
        accepted = 0
        removed_count = 0
        for node in candidate_order:
            if selected[node]:
                continue
            conflict_values: list[int] = []
            first_owner = int(interface_owner[source[node]])
            second_owner = int(interface_owner[target[node]])
            if first_owner >= 0:
                conflict_values.append(first_owner)
            if second_owner >= 0:
                conflict_values.append(second_owner)
            begin, end = crossing_offset[node], crossing_offset[node + 1]
            crossing_owner = crossing_neighbor[begin:end]
            if len(crossing_owner):
                conflict_values.extend(
                    int(value)
                    for value in crossing_owner[selected[crossing_owner]]
                )
            conflict = np.asarray(
                sorted(set(conflict_values)), dtype=np.int32
            )
            conflict_set = set(int(value) for value in conflict)
            begin, end = continuity_offset[node], continuity_offset[node + 1]
            neighbor = continuity_neighbor[begin:end]
            neighbor_weight = continuity_neighbor_weight[begin:end]
            retained_neighbor = selected[neighbor] & ~np.isin(
                neighbor, conflict, assume_unique=False
            )
            addition = float(
                unary[node]
                + settings.continuity_weight
                * np.sum(neighbor_weight[retained_neighbor])
            )
            removal = float(np.sum(unary[conflict]))
            internal_weight = 0.0
            support_weight = 0.0
            for removed in conflict:
                edge_begin = continuity_offset[removed]
                edge_end = continuity_offset[removed + 1]
                removed_neighbor = continuity_neighbor[edge_begin:edge_end]
                removed_weight = continuity_neighbor_weight[edge_begin:edge_end]
                active_neighbor = selected[removed_neighbor]
                support_weight += float(np.sum(removed_weight[active_neighbor]))
                internal_weight += float(
                    np.sum(
                        removed_weight[
                            np.asarray(
                                [
                                    int(value) in conflict_set
                                    and int(removed) < int(value)
                                    for value in removed_neighbor
                                ],
                                dtype=bool,
                            )
                        ]
                    )
                )
            removal += settings.continuity_weight * (
                support_weight - internal_weight
            )
            gain = addition - removal
            if gain <= settings.minimum_swap_gain:
                continue
            if len(conflict):
                selected[conflict] = False
                interface_owner[source[conflict]] = -1
                interface_owner[target[conflict]] = -1
                removed_count += len(conflict)
            selected[node] = True
            interface_owner[source[node]] = node
            interface_owner[target[node]] = node
            accepted += 1

        # Remove unsupported non-core ribbons whose exact marginal is negative.
        pruned = 0
        while True:
            support = _selected_continuity_support(
                selected, first, second, edge_weight
            )
            marginal = unary + settings.continuity_weight * support
            remove = selected & ~mutual & (marginal < 0.0)
            if not np.any(remove):
                break
            selected[remove] = False
            interface_owner[source[remove]] = -1
            interface_owner[target[remove]] = -1
            pruned += int(np.count_nonzero(remove))
        objective = _configuration_objective(
            selected,
            unary,
            first,
            second,
            edge_weight,
            settings.continuity_weight,
        )
        sweep_records.append(
            {
                "sweep": sweep + 1,
                "acceptedCandidateCount": accepted,
                "removedConflictCount": removed_count,
                "prunedNegativeMarginalCount": pruned,
                "selectedCount": int(np.count_nonzero(selected)),
                "objective": round(objective, 6),
                "objectiveGain": round(objective - previous_objective, 6),
            }
        )
        if accepted == 0 and pruned == 0:
            break
        previous_objective = objective

    # Fill only holes that are already surrounded by one reconstructed sheet.
    # This deliberately overrides a weak unary score, but never interface
    # exclusivity, exact non-intersection, or component ambiguity.
    hole_growth_records: list[dict[str, Any]] = []
    for growth_sweep in range(settings.maximum_hole_growth_sweeps):
        component, _ = _component_labels(
            selected, topology_first, topology_second
        )
        interface_owner = np.full(len(interfaces["positionXYZ"]), -1, dtype=np.int32)
        selected_node = np.flatnonzero(selected)
        interface_owner[source[selected_node]] = selected_node
        interface_owner[target[selected_node]] = selected_node
        edge_from_selected = (
            selected[topology_first] ^ selected[topology_second]
        )
        unselected_end = np.where(
            selected[topology_first[edge_from_selected]],
            topology_second[edge_from_selected],
            topology_first[edge_from_selected],
        )
        cross_weight = topology_weight[edge_from_selected]
        neighbor_count = np.zeros(node_count, dtype=np.int32)
        neighbor_weight = np.zeros(node_count, dtype=np.float32)
        np.add.at(neighbor_count, unselected_end, 1)
        np.add.at(neighbor_weight, unselected_end, cross_weight)
        crossing_blocked = np.zeros(node_count, dtype=bool)
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
        free_endpoint = (
            (interface_owner[source] < 0) & (interface_owner[target] < 0)
        )
        proposed = np.flatnonzero(
            ~selected
            & free_endpoint
            & ~crossing_blocked
            & (neighbor_count >= settings.minimum_hole_growth_neighbors)
            & (
                neighbor_weight / np.maximum(neighbor_count, 1)
                >= settings.minimum_hole_growth_mean_edge_score
            )
        )
        accepted_proposal: list[tuple[float, int, int, float]] = []
        for node in proposed:
            begin, end = topology_offset[node], topology_offset[node + 1]
            neighbor = topology_neighbor[begin:end]
            weight = topology_neighbor_weight[begin:end]
            active = selected[neighbor]
            neighbor = neighbor[active]
            weight = weight[active]
            neighbor_component = np.unique(component[neighbor])
            neighbor_component = neighbor_component[neighbor_component >= 0]
            if len(neighbor_component) != 1:
                continue
            direction = midpoint[neighbor] - midpoint[node]
            direction /= np.maximum(
                np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-6
            )
            covariance = np.einsum(
                "i,ij,ik->jk", weight, direction, direction
            )
            eigenvalue = np.linalg.eigvalsh(covariance)
            tangent_ratio = float(
                eigenvalue[1] / max(float(eigenvalue[2]), 1.0e-6)
            )
            if tangent_ratio < settings.minimum_hole_growth_tangent_ratio:
                continue
            priority = float(np.sum(weight) + 0.25 * physical_score[node])
            accepted_proposal.append(
                (priority, int(node), int(neighbor_component[0]), tangent_ratio)
            )
        accepted_proposal.sort(reverse=True)
        added = 0
        for _, node, component_id, _ in accepted_proposal:
            if selected[node]:
                continue
            if (
                interface_owner[source[node]] >= 0
                or interface_owner[target[node]] >= 0
            ):
                continue
            begin, end = crossing_offset[node], crossing_offset[node + 1]
            if np.any(selected[crossing_neighbor[begin:end]]):
                continue
            edge_begin, edge_end = topology_offset[node], topology_offset[node + 1]
            neighbor = topology_neighbor[edge_begin:edge_end]
            active_neighbor = neighbor[selected[neighbor]]
            active_component = np.unique(component[active_neighbor])
            active_component = active_component[active_component >= 0]
            if len(active_component) != 1 or int(active_component[0]) != component_id:
                continue
            selected[node] = True
            component[node] = component_id
            interface_owner[source[node]] = node
            interface_owner[target[node]] = node
            added += 1
        hole_growth_records.append(
            {
                "sweep": growth_sweep + 1,
                "proposedCount": int(len(proposed)),
                "geometricallySupportedCount": int(len(accepted_proposal)),
                "addedCount": int(added),
                "selectedCount": int(np.count_nonzero(selected)),
            }
        )
        if added == 0:
            break

    support_selected_edge = selected[first] & selected[second]
    selected_edge = support_selected_edge & topology_edge_mask
    selected_crossing = selected[crossing_first] & selected[crossing_second]
    if np.any(selected_crossing):
        raise RuntimeError("configuration optimization retained crossing profiles")
    selected_source = source[selected]
    selected_target = target[selected]
    endpoint = np.concatenate((selected_source, selected_target))
    if len(np.unique(endpoint)) != len(endpoint):
        raise RuntimeError("configuration assigns one interface more than once")
    component, component_size = _component_labels(
        selected, topology_first, topology_second
    )
    arrays = {
        **crossings,
        "nodeUnaryScore": unary,
        "initialSelected": initial.astype(np.uint8),
        "selected": selected.astype(np.uint8),
        "component": component,
        "supportEdgeSelected": support_selected_edge.astype(np.uint8),
        "edgeSelected": selected_edge.astype(np.uint8),
    }
    stats = {
        "frontierCandidateCount": int(node_count),
        "initialSelectedCount": int(np.count_nonzero(initial)),
        "crossingCleanupRemovedCount": crossing_clean_count,
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "selectedMutualFirstHitCount": int(np.count_nonzero(selected & mutual)),
        "selectedInterfaceCount": int(len(endpoint)),
        "selectedSupportEdgeCount": int(
            np.count_nonzero(support_selected_edge)
        ),
        "selectedContinuityEdgeCount": int(np.count_nonzero(selected_edge)),
        "selectedCrossingConflictCount": int(np.count_nonzero(selected_crossing)),
        "configurationObjective": round(
            _configuration_objective(
                selected,
                unary,
                first,
                second,
                edge_weight,
                settings.continuity_weight,
            ),
            6,
        ),
        "optimizationSweeps": sweep_records,
        "holeGrowthSweeps": hole_growth_records,
        "componentCount": int(len(component_size)),
        "componentWithAtLeast8RibbonsCount": int(
            np.count_nonzero(component_size >= 8)
        ),
        "componentWithAtLeast32RibbonsCount": int(
            np.count_nonzero(component_size >= 32)
        ),
        "largestComponentRibbonCounts": [
            int(value) for value in component_size[:32]
        ],
        "identityLabelsUsed": False,
    }
    return arrays, stats


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact data hash differs from manifest: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _load_continuity_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / "physical-ribbon-continuity-v1.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CONTINUITY_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("topology continuity must be complete and label-free")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _load_inputs(
    root: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    value = Path(root).resolve()
    continuity_path = (
        value
        if value.is_file()
        else value / "physical-ribbon-continuity-v1.json"
    )
    continuity_manifest = json.loads(continuity_path.read_text())
    if (
        continuity_manifest.get("schema") != PHYSICAL_RIBBON_CONTINUITY_SCHEMA
        or continuity_manifest.get("state") != "complete"
        or continuity_manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("configuration requires complete label-free continuity")
    continuity_data_path = continuity_path.parent / str(
        continuity_manifest["data"]["path"]
    )
    continuity = _load_npz(
        continuity_data_path, continuity_manifest["data"]["sha256"]
    )
    ribbon_path = Path(
        continuity_manifest["identity"]["ribbonBank"]["manifestPath"]
    )
    if (
        sha256_file(ribbon_path)
        != continuity_manifest["identity"]["ribbonBank"]["manifestSha256"]
    ):
        raise ValueError("ribbon bank changed after continuity solve")
    ribbon_manifest = json.loads(ribbon_path.read_text())
    if ribbon_manifest.get("schema") != PHYSICAL_RIBBON_BANK_SCHEMA:
        raise ValueError("continuity references the wrong ribbon bank")
    ribbon_data_path = ribbon_path.parent / str(ribbon_manifest["data"]["path"])
    ribbon = _load_npz(ribbon_data_path, ribbon_manifest["data"]["sha256"])
    interface_path = Path(
        ribbon_manifest["identity"]["interfaceBank"]["manifestPath"]
    )
    if (
        sha256_file(interface_path)
        != ribbon_manifest["identity"]["interfaceBank"]["manifestSha256"]
    ):
        raise ValueError("interface bank changed after ribbon pairing")
    interface_manifest = json.loads(interface_path.read_text())
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    interfaces = _load_npz(
        interface_data_path, interface_manifest["data"]["sha256"]
    )
    return (
        continuity_path,
        continuity_manifest,
        continuity,
        ribbon_path,
        ribbon_manifest,
        ribbon,
        interface_path,
        interface_manifest,
        interfaces,
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_physical_ribbon_configuration(
    continuity_root: str | Path,
    output_root: str | Path,
    *,
    topology_continuity_root: str | Path | None = None,
    settings: PhysicalRibbonConfigurationSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonConfigurationSettings()
    (
        continuity_path,
        continuity_manifest,
        continuity,
        ribbon_path,
        ribbon_manifest,
        ribbon,
        interface_path,
        interface_manifest,
        interfaces,
    ) = _load_inputs(continuity_root)
    topology_continuity = continuity
    topology_identity: dict[str, Any] | None = None
    if topology_continuity_root is not None:
        (
            topology_path,
            topology_manifest,
            topology_continuity,
        ) = _load_continuity_artifact(topology_continuity_root)
        topology_ribbon = topology_manifest.get("identity", {}).get(
            "ribbonBank", {}
        )
        if topology_ribbon.get("dataSha256") != ribbon_manifest["data"]["sha256"]:
            raise ValueError(
                "topology continuity was not built from the support ribbon bank"
            )
        if topology_manifest.get("geometry") != continuity_manifest.get("geometry"):
            raise ValueError("support and topology continuity geometry differs")
        topology_identity = {
            "manifestPath": str(topology_path),
            "manifestSha256": sha256_file(topology_path),
            "dataSha256": topology_manifest["data"]["sha256"],
        }
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CONFIGURATION_VERSION,
        "continuity": {
            "manifestPath": str(continuity_path),
            "manifestSha256": sha256_file(continuity_path),
            "dataSha256": continuity_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "interfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    if topology_identity is not None:
        identity["topologyContinuity"] = topology_identity
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CONFIGURATION_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CONFIGURATION_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    geometry = continuity_manifest["geometry"]
    source_origin = np.asarray(
        continuity_manifest["source"]["sourceOriginXYZ"], dtype=np.float32
    )
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    processing_stop = np.asarray(
        geometry["processingVoxelBounds"]["stopXYZExclusive"], dtype=np.float32
    )
    processing_shape = np.asarray(
        geometry["processingShapeSamplingXYZ"], dtype=np.int32
    )
    stride_xyz = (processing_stop - processing_start) / processing_shape
    if not np.allclose(stride_xyz, stride_xyz[0]):
        raise ValueError("configuration requires an isotropic sampling stride")
    stride = int(round(float(stride_xyz[0])))
    started = time.monotonic()
    crossings, crossing_stats = build_profile_crossing_conflicts(
        ribbon,
        interfaces,
        continuity,
        processing_world_start_xyz=source_origin + processing_start,
        processing_shape_sampling_xyz=tuple(int(value) for value in processing_shape),
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    crossed = time.monotonic()
    optimized, optimization_stats = optimize_physical_ribbon_configuration(
        ribbon,
        interfaces,
        continuity,
        crossings,
        settings=resolved,
        topology_continuity=topology_continuity,
    )
    solved = time.monotonic()
    _write_npz(data_path, optimized)
    view = {**continuity, **optimized}
    world = geometry["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        view,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "optimized-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        view,
        output / "largest-optimized-ribbon-components.png",
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CONFIGURATION_VERSION,
        "state": "complete",
        "identity": identity,
        "source": continuity_manifest["source"],
        "geometry": geometry,
        "crossings": crossing_stats,
        "configuration": optimization_stats,
        "timingSeconds": {
            "crossingGraph": round(crossed - started, 6),
            "configurationOptimization": round(solved - crossed, 6),
            "writingAndPreviews": round(finished - solved, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(optimized),
        },
        "artifacts": {
            "componentOverview": overview.name,
            "largestComponentMontage": montage.name,
        },
        "method": {
            "objective": (
                "physical pair evidence plus simultaneous two-face continuity, "
                "minus later inward-ray hits and configuration complexity"
            ),
            "hardConstraints": (
                "one ribbon per observed interface and no exact interior "
                "profile crossings"
            ),
            "alternatives": (
                "all rejected candidates remain in the immutable ribbon bank"
            ),
            "topology": (
                "support continuity votes in the configuration objective; a "
                "separate strict continuity graph defines component identity"
                if topology_identity is not None
                else "support and topology use the same continuity graph"
            ),
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
