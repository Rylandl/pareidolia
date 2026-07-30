from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Hashable, Iterable


Node = Hashable
NodePair = tuple[Node, Node]


def _ordered_pair(first: Node, second: Node) -> NodePair:
    if first == second:
        raise ValueError("a signed graph edge cannot be a self-edge")
    return (
        (first, second)
        if repr(first) <= repr(second)
        else (second, first)
    )


@dataclass(frozen=True, slots=True)
class SignedEdge:
    first: Node
    second: Node
    weight: float

    def __post_init__(self) -> None:
        if self.first == self.second:
            raise ValueError("a signed graph edge cannot be a self-edge")
        if not math.isfinite(self.weight) or self.weight < 0.0:
            raise ValueError("signed graph weights must be finite and nonnegative")


@dataclass(frozen=True, slots=True)
class SignedMerge:
    first_component: int
    second_component: int
    result_component: int
    first_nodes: int
    second_nodes: int
    attractive_weight: float
    repulsive_weight: float

    @property
    def gain(self) -> float:
        return self.attractive_weight - self.repulsive_weight


@dataclass(frozen=True, slots=True)
class SignedPartition:
    component_by_node: dict[Node, int]
    members_by_component: dict[int, tuple[Node, ...]]
    merges: tuple[SignedMerge, ...]
    attractive_weight: float
    repulsive_weight: float
    internal_attractive_weight: float
    internal_repulsive_weight: float
    hard_rejections: int

    @property
    def objective(self) -> float:
        return self.internal_attractive_weight - self.internal_repulsive_weight

    def record(self) -> dict[str, Any]:
        sizes = sorted(
            (len(value) for value in self.members_by_component.values()),
            reverse=True,
        )
        return {
            "method": "greedy agglomerative signed correlation clustering",
            "nodes": len(self.component_by_node),
            "components": len(self.members_by_component),
            "merges": len(self.merges),
            "largestComponentNodes": sizes[0] if sizes else 0,
            "attractiveWeight": round(self.attractive_weight, 6),
            "repulsiveWeight": round(self.repulsive_weight, 6),
            "internalAttractiveWeight": round(
                self.internal_attractive_weight, 6
            ),
            "internalRepulsiveWeight": round(
                self.internal_repulsive_weight, 6
            ),
            "objective": round(self.objective, 6),
            "hardMergeRejections": self.hard_rejections,
        }


def signed_graph_partition(
    nodes: Iterable[Node],
    attractive_edges: Iterable[SignedEdge],
    repulsive_edges: Iterable[SignedEdge],
    *,
    hard_separate_pairs: Iterable[NodePair] = (),
    minimum_merge_gain: float = 0.0,
) -> SignedPartition:
    """Agglomerate a sparse signed graph by exact component-pair gain.

    Merging two current components realizes every attractive and repulsive edge
    crossing that pair.  Their summed signed weight is therefore the exact
    objective gain for that merge.  Recomputing those sums after every union is
    the critical distinction from accepting a locally plausible bridge: one
    shear edge must outweigh all lifted evidence that the two tracks are
    distinct.  The result is a deterministic local optimum, not a claim of an
    exact solution to the NP-hard correlation-clustering objective.
    """

    node_values = tuple(nodes)
    if len(set(node_values)) != len(node_values):
        raise ValueError("signed graph nodes must be unique")
    if not math.isfinite(minimum_merge_gain):
        raise ValueError("minimum signed merge gain must be finite")
    node_set = set(node_values)

    def collect(edges: Iterable[SignedEdge]) -> dict[NodePair, float]:
        result: dict[NodePair, float] = defaultdict(float)
        for value in edges:
            if value.first not in node_set or value.second not in node_set:
                raise ValueError("signed graph edge references an absent node")
            result[_ordered_pair(value.first, value.second)] += value.weight
        return dict(result)

    attractive = collect(attractive_edges)
    repulsive = collect(repulsive_edges)
    hard_pairs = {
        _ordered_pair(first, second) for first, second in hard_separate_pairs
    }
    if any(first not in node_set or second not in node_set for first, second in hard_pairs):
        raise ValueError("hard signed separation references an absent node")
    hard_by_node: dict[Node, set[Node]] = defaultdict(set)
    for first, second in hard_pairs:
        hard_by_node[first].add(second)
        hard_by_node[second].add(first)

    # Cluster IDs are stable creation indices; a merge creates a fresh ID so
    # stale heap records can be discarded without a mutable-version protocol.
    cluster_by_node = {node: index for index, node in enumerate(node_values)}
    members: dict[int, set[Node]] = {
        index: {node} for index, node in enumerate(node_values)
    }
    adjacency: dict[int, dict[int, tuple[float, float]]] = defaultdict(dict)
    for pair in set(attractive) | set(repulsive):
        first, second = pair
        first_cluster = cluster_by_node[first]
        second_cluster = cluster_by_node[second]
        values = (attractive.get(pair, 0.0), repulsive.get(pair, 0.0))
        adjacency[first_cluster][second_cluster] = values
        adjacency[second_cluster][first_cluster] = values

    heap: list[tuple[float, float, int, int]] = []

    def push(first: int, second: int) -> None:
        if first == second or first not in members or second not in members:
            return
        if first > second:
            first, second = second, first
        attractive_weight, repulsive_weight = adjacency[first].get(
            second, (0.0, 0.0)
        )
        gain = attractive_weight - repulsive_weight
        if attractive_weight > 0.0 and gain > minimum_merge_gain:
            heapq.heappush(
                heap,
                (-gain, -attractive_weight, first, second),
            )

    for first in sorted(adjacency):
        for second in sorted(adjacency[first]):
            if first < second:
                push(first, second)

    def hard_incompatible(first: int, second: int) -> bool:
        first_members = members[first]
        second_members = members[second]
        if len(first_members) > len(second_members):
            first_members, second_members = second_members, first_members
        return any(
            hard_by_node.get(node, set()) & second_members
            for node in first_members
        )

    merges: list[SignedMerge] = []
    hard_rejections = 0
    next_cluster = len(node_values)
    while heap:
        negative_gain, negative_attractive, first, second = heapq.heappop(heap)
        if first not in members or second not in members:
            continue
        attractive_weight, repulsive_weight = adjacency[first].get(
            second, (0.0, 0.0)
        )
        gain = attractive_weight - repulsive_weight
        if (
            abs(gain + negative_gain) > 1.0e-10
            or abs(attractive_weight + negative_attractive) > 1.0e-10
        ):
            push(first, second)
            continue
        if gain <= minimum_merge_gain:
            continue
        if hard_incompatible(first, second):
            hard_rejections += 1
            continue
        result = next_cluster
        next_cluster += 1
        first_members = members.pop(first)
        second_members = members.pop(second)
        result_members = first_members | second_members
        members[result] = result_members
        for node in result_members:
            cluster_by_node[node] = result
        neighbors = (set(adjacency[first]) | set(adjacency[second])) - {
            first,
            second,
        }
        adjacency[result] = {}
        for neighbor in neighbors:
            first_values = adjacency[first].get(neighbor, (0.0, 0.0))
            second_values = adjacency[second].get(neighbor, (0.0, 0.0))
            combined = (
                first_values[0] + second_values[0],
                first_values[1] + second_values[1],
            )
            adjacency[result][neighbor] = combined
            adjacency[neighbor].pop(first, None)
            adjacency[neighbor].pop(second, None)
            adjacency[neighbor][result] = combined
        adjacency.pop(first, None)
        adjacency.pop(second, None)
        merges.append(
            SignedMerge(
                first,
                second,
                result,
                len(first_members),
                len(second_members),
                attractive_weight,
                repulsive_weight,
            )
        )
        for neighbor in sorted(neighbors):
            push(result, neighbor)

    canonical_members = {
        min(values, key=repr): tuple(sorted(values, key=repr))
        for values in members.values()
    }
    component_by_node = {
        node: component
        for component, values in canonical_members.items()
        for node in values
    }
    internal_attractive = sum(
        weight
        for (first, second), weight in attractive.items()
        if component_by_node[first] == component_by_node[second]
    )
    internal_repulsive = sum(
        weight
        for (first, second), weight in repulsive.items()
        if component_by_node[first] == component_by_node[second]
    )
    return SignedPartition(
        component_by_node,
        canonical_members,
        tuple(merges),
        sum(attractive.values()),
        sum(repulsive.values()),
        internal_attractive,
        internal_repulsive,
        hard_rejections,
    )
