from __future__ import annotations

import heapq
import itertools
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bank import PHYSICAL_RIBBON_BANK_SCHEMA
from .physical_ribbon_configuration import (
    PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
    _adjacency,
    _component_labels,
)
from .physical_ribbon_continuity import (
    PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
    write_continuity_overview,
    write_largest_component_montage,
)


PHYSICAL_RIBBON_BRIDGING_SCHEMA = "pareidolia.physical-ribbon-bridging"
PHYSICAL_RIBBON_BRIDGING_VERSION = 1
PHYSICAL_RIBBON_BRIDGING_STEM = "physical-ribbon-bridging-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonBridgingSettings:
    minimum_bundle_candidate_count: int = 3
    minimum_anchor_count_per_component: int = 2
    minimum_mean_support_edge_score: float = 0.16
    minimum_bundle_tangent_ratio: float = 0.01
    maximum_bundle_candidate_separation_voxels: float = 8.0
    maximum_bundle_normal_degrees: float = 50.0
    maximum_bundle_thickness_change_voxels: float = 8.0
    maximum_bridge_path_candidate_count: int = 12
    minimum_disjoint_bridge_paths: int = 2
    maximum_shared_apex_candidate_count: int = 1
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        if self.minimum_bundle_candidate_count < 2:
            raise ValueError("bridge bundles require at least two candidates")
        if self.minimum_anchor_count_per_component < 1:
            raise ValueError("bridge bundles require component anchors")
        if not 0.0 < self.minimum_mean_support_edge_score <= 1.0:
            raise ValueError("mean bridge support must lie in (0, 1]")
        if not 0.0 <= self.minimum_bundle_tangent_ratio <= 1.0:
            raise ValueError("bundle tangent ratio must lie in [0, 1]")
        positive = (
            self.maximum_bundle_candidate_separation_voxels,
            self.maximum_bundle_normal_degrees,
            self.maximum_bundle_thickness_change_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("bundle geometry caps must be finite and positive")
        if self.maximum_bundle_normal_degrees >= 90.0:
            raise ValueError("bundle normal cap must be below 90 degrees")
        if self.maximum_bridge_path_candidate_count < 1:
            raise ValueError("maximum bridge path length must be positive")
        if self.minimum_disjoint_bridge_paths < 2:
            raise ValueError("bridge regions require at least two disjoint paths")
        if self.maximum_shared_apex_candidate_count not in (0, 1):
            raise ValueError("shared-apex support is either disabled or one node")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _candidate_clusters(
    candidates: list[int],
    offset: np.ndarray,
    neighbor: np.ndarray,
    midpoint: np.ndarray,
    normal: np.ndarray,
    thickness: np.ndarray,
    settings: PhysicalRibbonBridgingSettings,
) -> list[np.ndarray]:
    candidate_set = set(candidates)
    bridge_neighbor: dict[int, set[int]] = {
        value: set() for value in candidates
    }
    normal_cosine = math.cos(
        math.radians(settings.maximum_bundle_normal_degrees)
    )
    for first, second in itertools.combinations(candidates, 2):
        begin, end = int(offset[first]), int(offset[first + 1])
        ordinary = bool(np.any(neighbor[begin:end] == second))
        delta = float(np.linalg.norm(midpoint[first] - midpoint[second]))
        normal_agreement = abs(float(np.dot(normal[first], normal[second])))
        thickness_change = abs(float(thickness[first] - thickness[second]))
        geometric = (
            delta <= settings.maximum_bundle_candidate_separation_voxels
            and normal_agreement >= normal_cosine
            and thickness_change
            <= settings.maximum_bundle_thickness_change_voxels
        )
        if not ordinary and not geometric:
            continue
        bridge_neighbor[first].add(second)
        bridge_neighbor[second].add(first)
    unseen = set(candidates)
    clusters: list[np.ndarray] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        cluster = [start]
        while stack:
            node = stack.pop()
            for value in bridge_neighbor[node]:
                if value not in candidate_set or value not in unseen:
                    continue
                unseen.remove(value)
                stack.append(value)
                cluster.append(value)
        clusters.append(np.asarray(sorted(cluster), dtype=np.int32))
    return clusters


def build_bridge_bundles(
    ribbon: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonBridgingSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    selected = np.asarray(configuration["selected"]) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    edge_score = np.asarray(continuity["edgeScore"], dtype=np.float32)
    offset, neighbor, neighbor_score = _adjacency(
        len(frontier), first, second, edge_score
    )
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    normal = np.asarray(ribbon["normalXYZ"], dtype=np.float32)[frontier]
    thickness = np.asarray(ribbon["thicknessVoxels"], dtype=np.float32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        len(frontier), crossing_first, crossing_second
    )
    interface_count = len(np.asarray(ribbon["interfaceCandidateDegree"]))
    interface_owner = np.full(interface_count, -1, dtype=np.int32)
    selected_node = np.flatnonzero(selected)
    interface_owner[source[selected_node]] = selected_node
    interface_owner[target[selected_node]] = selected_node
    free_endpoint = (
        (interface_owner[source] < 0) & (interface_owner[target] < 0)
    )
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

    pair_candidates: dict[tuple[int, int], list[int]] = {}
    candidate_support: dict[tuple[int, int], tuple[int, float]] = {}
    multi_component_candidate_count = 0
    for node in np.flatnonzero(~selected & free_endpoint & ~crossing_blocked):
        begin, end = int(offset[node]), int(offset[node + 1])
        adjacent = neighbor[begin:end]
        score = neighbor_score[begin:end]
        active = selected[adjacent]
        adjacent = adjacent[active]
        score = score[active]
        if not len(adjacent):
            continue
        adjacent_component = component[adjacent]
        present = np.unique(adjacent_component[adjacent_component >= 0])
        if len(present) < 2:
            continue
        multi_component_candidate_count += 1
        for value in present:
            member = adjacent_component == value
            candidate_support[(int(node), int(value))] = (
                int(np.count_nonzero(member)),
                float(np.sum(score[member])),
            )
        for first_component, second_component in itertools.combinations(
            present.tolist(), 2
        ):
            pair = (int(first_component), int(second_component))
            pair_candidates.setdefault(pair, []).append(int(node))

    bundle_component_first: list[int] = []
    bundle_component_second: list[int] = []
    bundle_candidate_offset = [0]
    bundle_candidate: list[int] = []
    bundle_score: list[float] = []
    bundle_tangent_ratio: list[float] = []
    bundle_first_anchor_count: list[int] = []
    bundle_second_anchor_count: list[int] = []
    cluster_count = 0
    small_cluster_count = 0
    anchor_rejection_count = 0
    support_rejection_count = 0
    tangent_rejection_count = 0
    for pair, candidates in pair_candidates.items():
        for cluster in _candidate_clusters(
            candidates,
            offset,
            neighbor,
            midpoint,
            normal,
            thickness,
            settings,
        ):
            cluster_count += 1
            if len(cluster) < settings.minimum_bundle_candidate_count:
                small_cluster_count += 1
                continue
            first_anchors: set[int] = set()
            second_anchors: set[int] = set()
            support_sum = 0.0
            support_edge_count = 0
            for node in cluster:
                begin, end = int(offset[node]), int(offset[node + 1])
                adjacent = neighbor[begin:end]
                adjacent_score = neighbor_score[begin:end]
                for component_id, anchors in (
                    (pair[0], first_anchors),
                    (pair[1], second_anchors),
                ):
                    member = selected[adjacent] & (
                        component[adjacent] == component_id
                    )
                    anchors.update(int(value) for value in adjacent[member])
                    support_sum += float(np.sum(adjacent_score[member]))
                    support_edge_count += int(np.count_nonzero(member))
            if (
                len(first_anchors) < settings.minimum_anchor_count_per_component
                or len(second_anchors)
                < settings.minimum_anchor_count_per_component
            ):
                anchor_rejection_count += 1
                continue
            if (
                support_edge_count == 0
                or support_sum / support_edge_count
                < settings.minimum_mean_support_edge_score
            ):
                support_rejection_count += 1
                continue
            points = np.concatenate(
                (
                    midpoint[cluster],
                    midpoint[np.asarray(sorted(first_anchors), dtype=np.int32)],
                    midpoint[np.asarray(sorted(second_anchors), dtype=np.int32)],
                ),
                axis=0,
            )
            centered = points - np.mean(points, axis=0, keepdims=True)
            eigenvalue = np.linalg.eigvalsh(centered.T @ centered)
            tangent_ratio = float(
                eigenvalue[1] / max(float(eigenvalue[2]), 1.0e-6)
            )
            if tangent_ratio < settings.minimum_bundle_tangent_ratio:
                tangent_rejection_count += 1
                continue
            internal_weight = 0.0
            cluster_set = set(int(value) for value in cluster)
            for node in cluster:
                begin, end = int(offset[node]), int(offset[node + 1])
                for adjacent, value in zip(
                    neighbor[begin:end], neighbor_score[begin:end]
                ):
                    if int(node) < int(adjacent) and int(adjacent) in cluster_set:
                        internal_weight += float(value)
            bundle_component_first.append(pair[0])
            bundle_component_second.append(pair[1])
            bundle_candidate.extend(int(value) for value in cluster)
            bundle_candidate_offset.append(len(bundle_candidate))
            bundle_score.append(support_sum + internal_weight)
            bundle_tangent_ratio.append(tangent_ratio)
            bundle_first_anchor_count.append(len(first_anchors))
            bundle_second_anchor_count.append(len(second_anchors))

    arrays = {
        "bundleComponentFirst": np.asarray(
            bundle_component_first, dtype=np.int32
        ),
        "bundleComponentSecond": np.asarray(
            bundle_component_second, dtype=np.int32
        ),
        "bundleCandidateOffset": np.asarray(
            bundle_candidate_offset, dtype=np.int64
        ),
        "bundleCandidateFrontierIndex": np.asarray(
            bundle_candidate, dtype=np.int32
        ),
        "bundleScore": np.asarray(bundle_score, dtype=np.float32),
        "bundleTangentRatio": np.asarray(
            bundle_tangent_ratio, dtype=np.float32
        ),
        "bundleFirstAnchorCount": np.asarray(
            bundle_first_anchor_count, dtype=np.int32
        ),
        "bundleSecondAnchorCount": np.asarray(
            bundle_second_anchor_count, dtype=np.int32
        ),
        "bundleKind": np.zeros(len(bundle_score), dtype=np.uint8),
        "bundlePathCount": np.ones(len(bundle_score), dtype=np.uint8),
        "bundleSharedCandidateCount": np.zeros(
            len(bundle_score), dtype=np.uint8
        ),
    }
    return arrays, {
        "freeCrossingSafeMultiComponentCandidateCount": int(
            multi_component_candidate_count
        ),
        "componentPairWithCandidateCount": int(len(pair_candidates)),
        "spatialCandidateClusterCount": int(cluster_count),
        "smallClusterRejectionCount": int(small_cluster_count),
        "anchorRejectionCount": int(anchor_rejection_count),
        "supportRejectionCount": int(support_rejection_count),
        "tangentRejectionCount": int(tangent_rejection_count),
        "qualifiedConnectedBundleCount": int(len(bundle_score)),
        "bundleCandidateCount": int(len(bundle_candidate)),
    }


def _shortest_region_path(
    region_nodes: set[int],
    boundary_first: list[tuple[int, int, float]],
    boundary_second: list[tuple[int, int, float]],
    offset: np.ndarray,
    neighbor: np.ndarray,
    neighbor_score: np.ndarray,
    physical_score: np.ndarray,
    *,
    banned_nodes: set[int],
    banned_first_anchors: set[int],
    banned_second_anchors: set[int],
) -> tuple[list[int], int, int, float] | None:
    target: dict[int, list[tuple[int, float]]] = {}
    for node, anchor, score in boundary_second:
        if node in banned_nodes or anchor in banned_second_anchors:
            continue
        target.setdefault(node, []).append((anchor, score))
    distance: dict[int, float] = {}
    predecessor: dict[int, int] = {}
    source_anchor: dict[int, int] = {}
    queue: list[tuple[float, int]] = []
    for node, anchor, score in boundary_first:
        if node in banned_nodes or anchor in banned_first_anchors:
            continue
        cost = 1.0 - score
        if cost >= distance.get(node, float("inf")):
            continue
        distance[node] = cost
        predecessor[node] = -1
        source_anchor[node] = anchor
        heapq.heappush(queue, (cost, node))
    best: tuple[float, int, int] | None = None
    while queue:
        cost, node = heapq.heappop(queue)
        if cost != distance.get(node):
            continue
        for target_anchor, boundary_score in target.get(node, ()):
            total = cost + 1.0 - boundary_score
            if best is None or total < best[0]:
                best = (total, node, target_anchor)
        if best is not None and cost > best[0]:
            continue
        begin, end = int(offset[node]), int(offset[node + 1])
        for adjacent, edge_score in zip(
            neighbor[begin:end], neighbor_score[begin:end]
        ):
            value = int(adjacent)
            if value not in region_nodes or value in banned_nodes:
                continue
            step = 0.15 + 1.0 - float(edge_score)
            step += 0.20 * (1.0 - float(physical_score[value]))
            updated = cost + step
            if updated >= distance.get(value, float("inf")):
                continue
            distance[value] = updated
            predecessor[value] = node
            source_anchor[value] = source_anchor[node]
            heapq.heappush(queue, (updated, value))
    if best is None:
        return None
    cost, node, target_anchor = best
    path = [node]
    while predecessor[node] >= 0:
        node = predecessor[node]
        path.append(node)
    path.reverse()
    return path, source_anchor[path[0]], target_anchor, cost


def _extend_path_exclusions(
    region_nodes: set[int],
    path: list[int],
    source: np.ndarray,
    target: np.ndarray,
    crossing_offset: np.ndarray,
    crossing_neighbor: np.ndarray,
    banned_nodes: set[int],
    *,
    preserved_node: int | None = None,
) -> None:
    banned_nodes.update(path)
    used_interface = set(int(source[value]) for value in path)
    used_interface.update(int(target[value]) for value in path)
    for value in region_nodes:
        if value == preserved_node:
            continue
        if (
            int(source[value]) in used_interface
            or int(target[value]) in used_interface
        ):
            banned_nodes.add(value)
    for value in path:
        begin, end = (
            int(crossing_offset[value]),
            int(crossing_offset[value + 1]),
        )
        banned_nodes.update(
            int(adjacent)
            for adjacent in crossing_neighbor[begin:end]
            if int(adjacent) in region_nodes
            and int(adjacent) != preserved_node
        )
    if preserved_node is not None:
        banned_nodes.discard(preserved_node)


def _single_shared_apex_alternative(
    region_nodes: set[int],
    first_path: list[int],
    first_anchor: int,
    second_anchor: int,
    boundary_first: list[tuple[int, int, float]],
    boundary_second: list[tuple[int, int, float]],
    offset: np.ndarray,
    neighbor: np.ndarray,
    neighbor_score: np.ndarray,
    physical_score: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    crossing_offset: np.ndarray,
    crossing_neighbor: np.ndarray,
    *,
    maximum_path_candidate_count: int,
) -> tuple[list[int], int, int, float, int] | None:
    """Find a second lane that shares only one interior fold-nose node."""
    if len(first_path) < 3:
        return None
    first_path_set = set(first_path)
    best: tuple[list[int], int, int, float, int] | None = None
    for apex in first_path[1:-1]:
        banned_nodes: set[int] = set()
        _extend_path_exclusions(
            region_nodes,
            first_path,
            source,
            target,
            crossing_offset,
            crossing_neighbor,
            banned_nodes,
            preserved_node=apex,
        )
        result = _shortest_region_path(
            region_nodes,
            boundary_first,
            boundary_second,
            offset,
            neighbor,
            neighbor_score,
            physical_score,
            banned_nodes=banned_nodes,
            banned_first_anchors={first_anchor},
            banned_second_anchors={second_anchor},
        )
        if result is None:
            continue
        path, alternate_first, alternate_second, cost = result
        if (
            len(path) > maximum_path_candidate_count
            or len(path) < 3
            or apex not in path[1:-1]
            or first_path_set.intersection(path) != {apex}
        ):
            continue
        candidate = (
            path,
            alternate_first,
            alternate_second,
            cost,
            apex,
        )
        if best is None or cost < best[3]:
            best = candidate
    return best


def build_bridge_path_bundles(
    ribbon: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonBridgingSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    node_count = len(frontier)
    selected = np.asarray(configuration["selected"]) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    edge_score = np.asarray(continuity["edgeScore"], dtype=np.float32)
    offset, neighbor, neighbor_score = _adjacency(
        node_count, first, second, edge_score
    )
    bank_index = frontier
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[bank_index]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[bank_index]
    physical_score = np.asarray(
        ribbon["physicalEvidenceScore"], dtype=np.float32
    )[bank_index]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        node_count, crossing_first, crossing_second
    )
    interface_owner = np.full(
        len(np.asarray(ribbon["interfaceCandidateDegree"])),
        -1,
        dtype=np.int32,
    )
    selected_node = np.flatnonzero(selected)
    interface_owner[source[selected_node]] = selected_node
    interface_owner[target[selected_node]] = selected_node
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
    feasible = (
        ~selected
        & (interface_owner[source] < 0)
        & (interface_owner[target] < 0)
        & ~crossing_blocked
    )
    internal = feasible[first] & feasible[second]
    parent = np.arange(node_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first_node, second_node in zip(first[internal], second[internal]):
        first_root = find(int(first_node))
        second_root = find(int(second_node))
        if first_root == second_root:
            continue
        if first_root > second_root:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
    region_nodes: dict[int, list[int]] = {}
    for node in np.flatnonzero(feasible):
        region_nodes.setdefault(find(int(node)), []).append(int(node))
    region_boundary: dict[
        int, dict[int, list[tuple[int, int, float]]]
    ] = {}
    boundary = feasible[first] ^ feasible[second]
    boundary &= selected[first] | selected[second]
    for edge_index in np.flatnonzero(boundary):
        if feasible[first[edge_index]]:
            node = int(first[edge_index])
            anchor = int(second[edge_index])
        else:
            node = int(second[edge_index])
            anchor = int(first[edge_index])
        component_id = int(component[anchor])
        if component_id < 0:
            continue
        root = find(node)
        region_boundary.setdefault(root, {}).setdefault(
            component_id, []
        ).append((node, anchor, float(edge_score[edge_index])))

    bundle_first: list[int] = []
    bundle_second: list[int] = []
    bundle_offset = [0]
    bundle_candidate: list[int] = []
    bundle_score: list[float] = []
    bundle_tangent: list[float] = []
    bundle_first_anchor: list[int] = []
    bundle_second_anchor: list[int] = []
    bundle_path_count: list[int] = []
    bundle_kind: list[int] = []
    bundle_shared_candidate_count: list[int] = []
    multi_component_region_count = 0
    two_component_region_count = 0
    disjoint_path_rejection_count = 0
    overlong_path_rejection_count = 0
    shared_apex_bundle_count = 0
    for root, boundaries in region_boundary.items():
        if len(boundaries) < 2:
            continue
        multi_component_region_count += 1
        if len(boundaries) != 2:
            continue
        two_component_region_count += 1
        component_pair = tuple(sorted(boundaries))
        nodes = set(region_nodes[root])
        paths: list[list[int]] = []
        first_anchors: list[int] = []
        second_anchors: list[int] = []
        costs: list[float] = []
        banned_nodes: set[int] = set()
        for _ in range(settings.minimum_disjoint_bridge_paths):
            result = _shortest_region_path(
                nodes,
                boundaries[component_pair[0]],
                boundaries[component_pair[1]],
                offset,
                neighbor,
                neighbor_score,
                physical_score,
                banned_nodes=banned_nodes,
                banned_first_anchors=set(first_anchors),
                banned_second_anchors=set(second_anchors),
            )
            if result is None:
                break
            path, first_anchor, second_anchor, cost = result
            if len(path) > settings.maximum_bridge_path_candidate_count:
                overlong_path_rejection_count += 1
                break
            paths.append(path)
            first_anchors.append(first_anchor)
            second_anchors.append(second_anchor)
            costs.append(cost)
            _extend_path_exclusions(
                nodes,
                path,
                source,
                target,
                crossing_offset,
                crossing_neighbor,
                banned_nodes,
            )
        shared_candidate_count = 0
        if (
            len(paths) == 1
            and settings.maximum_shared_apex_candidate_count == 1
        ):
            alternate = _single_shared_apex_alternative(
                nodes,
                paths[0],
                first_anchors[0],
                second_anchors[0],
                boundaries[component_pair[0]],
                boundaries[component_pair[1]],
                offset,
                neighbor,
                neighbor_score,
                physical_score,
                source,
                target,
                crossing_offset,
                crossing_neighbor,
                maximum_path_candidate_count=(
                    settings.maximum_bridge_path_candidate_count
                ),
            )
            if alternate is not None:
                path, first_anchor, second_anchor, cost, _ = alternate
                paths.append(path)
                first_anchors.append(first_anchor)
                second_anchors.append(second_anchor)
                costs.append(cost)
                shared_candidate_count = 1
        if len(paths) < settings.minimum_disjoint_bridge_paths:
            disjoint_path_rejection_count += 1
            continue
        candidates = sorted(set(itertools.chain.from_iterable(paths)))
        bundle_first.append(component_pair[0])
        bundle_second.append(component_pair[1])
        bundle_candidate.extend(candidates)
        bundle_offset.append(len(bundle_candidate))
        bundle_score.append(float(10.0 / max(sum(costs), 1.0e-6)))
        bundle_tangent.append(0.0)
        bundle_first_anchor.append(len(set(first_anchors)))
        bundle_second_anchor.append(len(set(second_anchors)))
        bundle_path_count.append(len(paths))
        bundle_kind.append(2 if shared_candidate_count else 1)
        bundle_shared_candidate_count.append(shared_candidate_count)
        shared_apex_bundle_count += int(bool(shared_candidate_count))
    arrays = {
        "bundleComponentFirst": np.asarray(bundle_first, dtype=np.int32),
        "bundleComponentSecond": np.asarray(bundle_second, dtype=np.int32),
        "bundleCandidateOffset": np.asarray(bundle_offset, dtype=np.int64),
        "bundleCandidateFrontierIndex": np.asarray(
            bundle_candidate, dtype=np.int32
        ),
        "bundleScore": np.asarray(bundle_score, dtype=np.float32),
        "bundleTangentRatio": np.asarray(bundle_tangent, dtype=np.float32),
        "bundleFirstAnchorCount": np.asarray(
            bundle_first_anchor, dtype=np.int32
        ),
        "bundleSecondAnchorCount": np.asarray(
            bundle_second_anchor, dtype=np.int32
        ),
        "bundleKind": np.asarray(bundle_kind, dtype=np.uint8),
        "bundlePathCount": np.asarray(bundle_path_count, dtype=np.uint8),
        "bundleSharedCandidateCount": np.asarray(
            bundle_shared_candidate_count, dtype=np.uint8
        ),
    }
    return arrays, {
        "freeCandidateRegionCount": int(len(region_nodes)),
        "multiComponentRegionCount": int(multi_component_region_count),
        "exactlyTwoComponentRegionCount": int(two_component_region_count),
        "disjointPathRejectionCount": int(disjoint_path_rejection_count),
        "overlongPathRejectionCount": int(overlong_path_rejection_count),
        "qualifiedPathBundleCount": int(len(bundle_score)),
        "qualifiedDisjointPathBundleCount": int(
            len(bundle_score) - shared_apex_bundle_count
        ),
        "qualifiedSharedApexBundleCount": int(shared_apex_bundle_count),
        "bundleCandidateCount": int(len(bundle_candidate)),
    }


def _combine_bundles(
    first: Mapping[str, np.ndarray],
    second: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    first_offset = np.asarray(first["bundleCandidateOffset"], dtype=np.int64)
    second_offset = np.asarray(second["bundleCandidateOffset"], dtype=np.int64)
    arrays: dict[str, np.ndarray] = {}
    for name in (
        "bundleComponentFirst",
        "bundleComponentSecond",
        "bundleScore",
        "bundleTangentRatio",
        "bundleFirstAnchorCount",
        "bundleSecondAnchorCount",
        "bundleKind",
        "bundlePathCount",
        "bundleSharedCandidateCount",
    ):
        arrays[name] = np.concatenate((first[name], second[name]))
    arrays["bundleCandidateFrontierIndex"] = np.concatenate(
        (
            first["bundleCandidateFrontierIndex"],
            second["bundleCandidateFrontierIndex"],
        )
    )
    arrays["bundleCandidateOffset"] = np.concatenate(
        (
            first_offset,
            first_offset[-1] + second_offset[1:],
        )
    )
    return arrays


def _bundle_connects_components(
    candidates: list[int],
    initial_selected: np.ndarray,
    component: np.ndarray,
    offset: np.ndarray,
    neighbor: np.ndarray,
    component_pair: tuple[int, int],
    *,
    minimum_anchor_count: int,
) -> tuple[bool, int, int]:
    """Revalidate a tentatively admitted bundle after hard exclusions."""
    local = {node: index for index, node in enumerate(candidates)}
    first_virtual = len(local)
    second_virtual = first_virtual + 1
    parent = np.arange(len(local) + 2, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first_value: int, second_value: int) -> None:
        first_root = find(first_value)
        second_root = find(second_value)
        if first_root == second_root:
            return
        if first_root > second_root:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root

    anchors: tuple[set[int], set[int]] = (set(), set())
    for node, local_index in local.items():
        begin, end = int(offset[node]), int(offset[node + 1])
        for adjacent_value in neighbor[begin:end]:
            adjacent = int(adjacent_value)
            adjacent_local = local.get(adjacent)
            if adjacent_local is not None:
                union(local_index, adjacent_local)
                continue
            if not initial_selected[adjacent]:
                continue
            component_id = int(component[adjacent])
            if component_id == component_pair[0]:
                anchors[0].add(adjacent)
                union(local_index, first_virtual)
            elif component_id == component_pair[1]:
                anchors[1].add(adjacent)
                union(local_index, second_virtual)
    valid = (
        len(anchors[0]) >= minimum_anchor_count
        and len(anchors[1]) >= minimum_anchor_count
        and find(first_virtual) == find(second_virtual)
    )
    return valid, len(anchors[0]), len(anchors[1])


def select_bridge_bundles(
    ribbon: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    bundles: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonBridgingSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    selected = (np.asarray(configuration["selected"]) > 0).copy()
    initial_selected = selected.copy()
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    edge_baseline = np.asarray(
        continuity.get("edgeBaseline", np.ones(len(first))), dtype=bool
    )
    offset, neighbor, _ = _adjacency(
        len(frontier),
        first,
        second,
        np.asarray(continuity["edgeScore"], dtype=np.float32),
    )
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        len(frontier), crossing_first, crossing_second
    )
    interface_owner = np.full(
        len(np.asarray(ribbon["interfaceCandidateDegree"])),
        -1,
        dtype=np.int32,
    )
    selected_node = np.flatnonzero(selected)
    interface_owner[source[selected_node]] = selected_node
    interface_owner[target[selected_node]] = selected_node
    bundle_first = np.asarray(bundles["bundleComponentFirst"], dtype=np.int32)
    bundle_second = np.asarray(bundles["bundleComponentSecond"], dtype=np.int32)
    bundle_offset = np.asarray(bundles["bundleCandidateOffset"], dtype=np.int64)
    bundle_candidate = np.asarray(
        bundles["bundleCandidateFrontierIndex"], dtype=np.int32
    )
    bundle_score = np.asarray(bundles["bundleScore"], dtype=np.float32)
    bundle_kind = np.asarray(bundles["bundleKind"], dtype=np.uint8)
    bundle_selected = np.zeros(len(bundle_first), dtype=np.uint8)
    bundle_selected_first_anchor_count = np.zeros(
        len(bundle_first), dtype=np.int32
    )
    bundle_selected_second_anchor_count = np.zeros(
        len(bundle_first), dtype=np.int32
    )
    node_bridge_bundle = np.full(len(frontier), -1, dtype=np.int32)
    configuration_component = np.asarray(
        configuration["component"], dtype=np.int32
    )
    used_component: set[int] = set()
    for bundle_index in np.argsort(-bundle_score):
        component_pair = (
            int(bundle_first[bundle_index]),
            int(bundle_second[bundle_index]),
        )
        if any(value in used_component for value in component_pair):
            continue
        begin, end = bundle_offset[bundle_index : bundle_index + 2]
        candidates = bundle_candidate[begin:end]
        admitted: list[int] = []
        for node in candidates:
            if (
                interface_owner[source[node]] >= 0
                or interface_owner[target[node]] >= 0
            ):
                continue
            crossing_begin = crossing_offset[node]
            crossing_end = crossing_offset[node + 1]
            if np.any(selected[crossing_neighbor[crossing_begin:crossing_end]]):
                continue
            admitted.append(int(node))
            selected[node] = True
            interface_owner[source[node]] = node
            interface_owner[target[node]] = node
        minimum_admitted = (
            settings.minimum_bundle_candidate_count
            if bundle_kind[bundle_index] == 0
            else settings.minimum_disjoint_bridge_paths
        )
        incomplete_path_bundle = (
            bundle_kind[bundle_index] != 0
            and len(admitted) != len(candidates)
        )
        minimum_anchor_count = settings.minimum_anchor_count_per_component
        if bundle_kind[bundle_index] != 0:
            minimum_anchor_count = max(
                minimum_anchor_count,
                settings.minimum_disjoint_bridge_paths,
            )
        connected, first_anchor_count, second_anchor_count = (
            _bundle_connects_components(
                admitted,
                initial_selected,
                configuration_component,
                offset,
                neighbor,
                component_pair,
                minimum_anchor_count=minimum_anchor_count,
            )
        )
        if (
            len(admitted) < minimum_admitted
            or incomplete_path_bundle
            or not connected
        ):
            selected[admitted] = False
            interface_owner[source[admitted]] = -1
            interface_owner[target[admitted]] = -1
            continue
        bundle_selected[bundle_index] = 1
        bundle_selected_first_anchor_count[bundle_index] = first_anchor_count
        bundle_selected_second_anchor_count[bundle_index] = second_anchor_count
        node_bridge_bundle[admitted] = int(bundle_index)
        used_component.update(component_pair)

    new_selected = selected & ~initial_selected
    selected_edge = (
        initial_selected[first]
        & initial_selected[second]
        & edge_baseline
    )
    both_new_index = np.flatnonzero(new_selected[first] & new_selected[second])
    both_new_bundle = node_bridge_bundle[first[both_new_index]]
    selected_edge[both_new_index] = (
        (both_new_bundle >= 0)
        & (both_new_bundle == node_bridge_bundle[second[both_new_index]])
    )
    for new_is_first in (True, False):
        new_node = first if new_is_first else second
        old_node = second if new_is_first else first
        edge_index = np.flatnonzero(
            new_selected[new_node] & initial_selected[old_node]
        )
        if not len(edge_index):
            continue
        bridge_index = node_bridge_bundle[new_node[edge_index]]
        old_component = configuration_component[old_node[edge_index]]
        allowed = bridge_index >= 0
        allowed &= (
            (old_component == bundle_first[bridge_index])
            | (old_component == bundle_second[bridge_index])
        )
        selected_edge[edge_index] = allowed
    selected_crossing = selected[crossing_first] & selected[crossing_second]
    if np.any(selected_crossing):
        raise RuntimeError("bridge selection introduced a crossing profile")
    endpoint = np.concatenate((source[selected], target[selected]))
    if len(np.unique(endpoint)) != len(endpoint):
        raise RuntimeError("bridge selection reused an observed interface")
    component, component_size = _component_labels(
        selected, first[selected_edge], second[selected_edge]
    )
    adaptive_edge = ~edge_baseline
    selected_endpoint = selected[first] & selected[second]
    arrays = {
        **bundles,
        "bundleSelected": bundle_selected,
        "bundleSelectedFirstAnchorCount": (
            bundle_selected_first_anchor_count
        ),
        "bundleSelectedSecondAnchorCount": (
            bundle_selected_second_anchor_count
        ),
        "nodeBridgeBundle": node_bridge_bundle,
        "initialSelected": initial_selected.astype(np.uint8),
        "selected": selected.astype(np.uint8),
        "component": component,
        "edgeSelected": selected_edge.astype(np.uint8),
    }
    return arrays, {
        "selectedBundleCount": int(np.count_nonzero(bundle_selected)),
        "addedBridgeRibbonCount": int(
            np.count_nonzero(selected & ~initial_selected)
        ),
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "selectedInterfaceCount": int(len(endpoint)),
        "selectedCrossingConflictCount": int(
            np.count_nonzero(selected_crossing)
        ),
        "selectedAdaptiveContinuationEdgeCount": int(
            np.count_nonzero(selected_edge & adaptive_edge)
        ),
        "suppressedAdaptiveSelectedEndpointEdgeCount": int(
            np.count_nonzero(
                selected_endpoint & adaptive_edge & ~selected_edge
            )
        ),
        "componentCount": int(len(component_size)),
        "componentWithAtLeast32RibbonsCount": int(
            np.count_nonzero(component_size >= 32)
        ),
        "largestComponentRibbonCounts": [
            int(value) for value in component_size[:32]
        ],
        "identityLabelsUsed": False,
    }


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact data hash differs from manifest: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


_CONTINUITY_EDGE_FIELDS = (
    "edgeFirstFrontierIndex",
    "edgeSecondFrontierIndex",
    "edgeScore",
    "edgeNormalDegrees",
    "edgeMidpointHeightResidualVoxels",
    "edgeBoundaryHeightResidualVoxels",
    "edgeThicknessChangeVoxels",
    "edgeBoundaryShiftDifferenceVoxels",
)


def _continuity_edge_key(continuity: Mapping[str, np.ndarray]) -> np.ndarray:
    first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.uint64
    )
    second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.uint64
    )
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    return (low << np.uint64(32)) | high


def _merge_continuity_graphs(
    baseline: Mapping[str, np.ndarray],
    proposal: Mapping[str, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Union a broader proposal graph without changing baseline semantics."""
    merged = {name: np.asarray(value) for name, value in baseline.items()}
    baseline_key = _continuity_edge_key(baseline)
    if len(baseline_key) > 1:
        ordered = np.sort(baseline_key)
        if np.any(ordered[1:] == ordered[:-1]):
            raise ValueError("baseline continuity contains duplicate edges")
    if proposal is None:
        merged["edgeBaseline"] = np.ones(len(baseline_key), dtype=np.uint8)
        return merged, {
            "baselineEdgeCount": int(len(baseline_key)),
            "proposalEdgeCount": int(len(baseline_key)),
            "adaptiveEdgeCount": 0,
            "adaptiveNormalDegrees": None,
        }
    if not np.array_equal(
        np.asarray(baseline["frontierRibbonCandidate"]),
        np.asarray(proposal["frontierRibbonCandidate"]),
    ):
        raise ValueError("bridge continuity frontier differs from configuration")
    proposal_key = _continuity_edge_key(proposal)
    if len(proposal_key) > 1:
        ordered_proposal = np.sort(proposal_key)
        if np.any(ordered_proposal[1:] == ordered_proposal[:-1]):
            raise ValueError("bridge continuity contains duplicate edges")
    baseline_order = np.argsort(baseline_key)
    sorted_baseline = baseline_key[baseline_order]
    location = np.searchsorted(sorted_baseline, proposal_key)
    in_baseline = location < len(sorted_baseline)
    in_baseline[in_baseline] &= (
        sorted_baseline[location[in_baseline]] == proposal_key[in_baseline]
    )
    adaptive_index = np.flatnonzero(~in_baseline)
    for name in _CONTINUITY_EDGE_FIELDS:
        merged[name] = np.concatenate(
            (
                np.asarray(baseline[name]),
                np.asarray(proposal[name])[adaptive_index],
            )
        )
    merged["edgeSelected"] = np.concatenate(
        (
            np.asarray(
                baseline.get(
                    "edgeSelected",
                    np.zeros(len(baseline_key), dtype=np.uint8),
                )
            ),
            np.zeros(len(adaptive_index), dtype=np.uint8),
        )
    )
    merged["edgeBaseline"] = np.concatenate(
        (
            np.ones(len(baseline_key), dtype=np.uint8),
            np.zeros(len(adaptive_index), dtype=np.uint8),
        )
    )
    adaptive_normal = np.asarray(
        proposal["edgeNormalDegrees"], dtype=np.float32
    )[adaptive_index]
    normal_stats: dict[str, float] | None = None
    if len(adaptive_normal):
        normal_stats = {
            "minimum": round(float(np.min(adaptive_normal)), 6),
            "median": round(float(np.median(adaptive_normal)), 6),
            "p90": round(float(np.quantile(adaptive_normal, 0.9)), 6),
            "maximum": round(float(np.max(adaptive_normal)), 6),
        }
    return merged, {
        "baselineEdgeCount": int(len(baseline_key)),
        "proposalEdgeCount": int(len(proposal_key)),
        "adaptiveEdgeCount": int(len(adaptive_index)),
        "adaptiveNormalDegrees": normal_stats,
    }


def _load_continuity_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value if value.is_file() else value / "physical-ribbon-continuity-v1.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CONTINUITY_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("bridge continuity must be complete and label-free")
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
    configuration_path = (
        value
        if value.is_file()
        else value / "physical-ribbon-configuration-v1.json"
    )
    configuration_manifest = json.loads(configuration_path.read_text())
    if (
        configuration_manifest.get("schema")
        != PHYSICAL_RIBBON_CONFIGURATION_SCHEMA
        or configuration_manifest.get("state") != "complete"
        or configuration_manifest.get("method", {}).get("identityLabelsUsed")
        is not False
        or configuration_manifest.get("configuration", {}).get(
            "selectedCrossingConflictCount"
        )
        != 0
    ):
        raise ValueError("bridging requires a complete non-crossing configuration")
    configuration_data_path = configuration_path.parent / str(
        configuration_manifest["data"]["path"]
    )
    configuration = _load_npz(
        configuration_data_path, configuration_manifest["data"]["sha256"]
    )
    continuity_identity = configuration_manifest["identity"].get(
        "topologyContinuity",
        configuration_manifest["identity"]["continuity"],
    )
    continuity_path = Path(continuity_identity["manifestPath"])
    if (
        sha256_file(continuity_path)
        != continuity_identity["manifestSha256"]
    ):
        raise ValueError("continuity artifact changed after configuration")
    continuity_manifest = json.loads(continuity_path.read_text())
    if continuity_manifest.get("schema") != PHYSICAL_RIBBON_CONTINUITY_SCHEMA:
        raise ValueError("configuration references the wrong continuity artifact")
    continuity_data_path = continuity_path.parent / str(
        continuity_manifest["data"]["path"]
    )
    continuity = _load_npz(
        continuity_data_path, continuity_manifest["data"]["sha256"]
    )
    ribbon_path = Path(
        configuration_manifest["identity"]["ribbonBank"]["manifestPath"]
    )
    if (
        sha256_file(ribbon_path)
        != configuration_manifest["identity"]["ribbonBank"]["manifestSha256"]
    ):
        raise ValueError("ribbon bank changed after configuration")
    ribbon_manifest = json.loads(ribbon_path.read_text())
    if ribbon_manifest.get("schema") != PHYSICAL_RIBBON_BANK_SCHEMA:
        raise ValueError("configuration references the wrong ribbon bank")
    ribbon_data_path = ribbon_path.parent / str(ribbon_manifest["data"]["path"])
    ribbon = _load_npz(ribbon_data_path, ribbon_manifest["data"]["sha256"])
    return (
        configuration_path,
        configuration_manifest,
        configuration,
        continuity_path,
        continuity_manifest,
        continuity,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_physical_ribbon_bridging(
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    bridge_continuity_root: str | Path | None = None,
    settings: PhysicalRibbonBridgingSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonBridgingSettings()
    (
        configuration_path,
        configuration_manifest,
        configuration,
        continuity_path,
        continuity_manifest,
        continuity,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_root)
    bridge_continuity_identity: dict[str, Any] | None = None
    proposal_continuity: dict[str, np.ndarray] | None = None
    if bridge_continuity_root is not None:
        (
            bridge_continuity_path,
            bridge_continuity_manifest,
            proposal_continuity,
        ) = _load_continuity_artifact(bridge_continuity_root)
        proposal_ribbon = bridge_continuity_manifest.get("identity", {}).get(
            "ribbonBank", {}
        )
        if (
            proposal_ribbon.get("dataSha256")
            != ribbon_manifest["data"]["sha256"]
        ):
            raise ValueError(
                "bridge continuity was not built from the configured ribbon bank"
            )
        bridge_continuity_identity = {
            "manifestPath": str(bridge_continuity_path),
            "manifestSha256": sha256_file(bridge_continuity_path),
            "dataSha256": bridge_continuity_manifest["data"]["sha256"],
        }
    continuity, proposal_graph_stats = _merge_continuity_graphs(
        continuity, proposal_continuity
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_BRIDGING_SCHEMA,
        "version": PHYSICAL_RIBBON_BRIDGING_VERSION,
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
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
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    if bridge_continuity_identity is not None:
        identity["bridgeContinuity"] = bridge_continuity_identity
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_BRIDGING_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_BRIDGING_STEM}.npz"
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
    direct_bundles, direct_bundle_stats = build_bridge_bundles(
        ribbon,
        continuity,
        configuration,
        settings=resolved,
    )
    path_bundles, path_bundle_stats = build_bridge_path_bundles(
        ribbon,
        continuity,
        configuration,
        settings=resolved,
    )
    bundles = _combine_bundles(direct_bundles, path_bundles)
    bundle_stats = {
        "proposalGraph": proposal_graph_stats,
        "direct": direct_bundle_stats,
        "disjointPaths": path_bundle_stats,
        "combinedBundleCount": int(len(bundles["bundleScore"])),
    }
    built = time.monotonic()
    selected, selection_stats = select_bridge_bundles(
        ribbon,
        continuity,
        configuration,
        bundles,
        settings=resolved,
    )
    solved = time.monotonic()
    _write_npz(data_path, selected)
    view = {**continuity, **selected}
    geometry = configuration_manifest["geometry"]
    world = geometry["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        view,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "bridged-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        view,
        output / "largest-bridged-ribbon-components.png",
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_BRIDGING_SCHEMA,
        "version": PHYSICAL_RIBBON_BRIDGING_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": geometry,
        "bundles": bundle_stats,
        "selection": selection_stats,
        "timingSeconds": {
            "bundleConstruction": round(built - started, 6),
            "collisionSafeSelection": round(solved - built, 6),
            "writingAndPreviews": round(finished - solved, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(selected),
        },
        "artifacts": {
            "componentOverview": overview.name,
            "largestComponentMontage": montage.name,
        },
        "method": {
            "bridgeUnit": (
                "a connected bundle of multiple free ribbon candidates with "
                "two-face support from the same pair of component boundaries"
            ),
            "transitivityGuard": (
                "each input component participates in at most one bundle per round"
            ),
            "adaptiveCurvatureGuard": (
                "proposal-only continuation edges activate only through an admitted "
                "bridge bundle; they never directly join two pre-existing selections"
            ),
            "hardConstraints": (
                "all interfaces remain unique and all exact crossing conflicts remain absent"
            ),
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
