from __future__ import annotations

import heapq
import math
from collections import defaultdict, deque
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
class SignedSplit:
    source: Node
    target: Node
    input_nodes: int
    first_nodes: int
    second_nodes: int
    severed_attractive_weight: float
    separated_repulsive_weight: float

    @property
    def gain(self) -> float:
        return self.separated_repulsive_weight - self.severed_attractive_weight


@dataclass(frozen=True, slots=True)
class SignedPartition:
    component_by_node: dict[Node, int]
    members_by_component: dict[int, tuple[Node, ...]]
    merges: tuple[SignedMerge, ...]
    splits: tuple[SignedSplit, ...]
    attractive_weight: float
    repulsive_weight: float
    internal_attractive_weight: float
    internal_repulsive_weight: float
    hard_rejections: int
    method: str

    @property
    def objective(self) -> float:
        return self.internal_attractive_weight - self.internal_repulsive_weight

    def record(self) -> dict[str, Any]:
        sizes = sorted(
            (len(value) for value in self.members_by_component.values()),
            reverse=True,
        )
        return {
            "method": self.method,
            "nodes": len(self.component_by_node),
            "components": len(self.members_by_component),
            "merges": len(self.merges),
            "improvingSplits": len(self.splits),
            "splitObjectiveGain": round(
                sum(value.gain for value in self.splits), 6
            ),
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


def _minimum_attractive_cut(
    members: frozenset[Node],
    attractive: dict[NodePair, float],
    source: Node,
    target: Node,
) -> frozenset[Node]:
    """Return the deterministic source side of one undirected minimum cut."""

    ordered = tuple(sorted(members, key=repr))
    index_by_node = {node: index for index, node in enumerate(ordered)}
    source_index = index_by_node[source]
    target_index = index_by_node[target]
    adjacency: list[list[list[float | int]]] = [[] for _ in ordered]

    def add_directed(first: int, second: int, capacity: float) -> None:
        forward: list[float | int] = [second, len(adjacency[second]), capacity]
        reverse: list[float | int] = [first, len(adjacency[first]), 0.0]
        adjacency[first].append(forward)
        adjacency[second].append(reverse)

    for (first_node, second_node), weight in attractive.items():
        if first_node not in members or second_node not in members or weight <= 0.0:
            continue
        first = index_by_node[first_node]
        second = index_by_node[second_node]
        add_directed(first, second, weight)
        add_directed(second, first, weight)

    epsilon = 1.0e-12
    while True:
        level = [-1] * len(ordered)
        level[source_index] = 0
        queue: deque[int] = deque((source_index,))
        while queue:
            first = queue.popleft()
            for edge in adjacency[first]:
                second = int(edge[0])
                if float(edge[2]) > epsilon and level[second] < 0:
                    level[second] = level[first] + 1
                    queue.append(second)
        if level[target_index] < 0:
            break
        cursor = [0] * len(ordered)

        def send(first: int, available: float) -> float:
            if first == target_index:
                return available
            while cursor[first] < len(adjacency[first]):
                edge = adjacency[first][cursor[first]]
                second = int(edge[0])
                capacity = float(edge[2])
                if capacity > epsilon and level[second] == level[first] + 1:
                    pushed = send(second, min(available, capacity))
                    if pushed > epsilon:
                        edge[2] = capacity - pushed
                        reverse = adjacency[second][int(edge[1])]
                        reverse[2] = float(reverse[2]) + pushed
                        return pushed
                cursor[first] += 1
            return 0.0

        while send(source_index, math.inf) > epsilon:
            pass

    reachable = {source_index}
    queue = deque((source_index,))
    while queue:
        first = queue.popleft()
        for edge in adjacency[first]:
            second = int(edge[0])
            if float(edge[2]) > epsilon and second not in reachable:
                reachable.add(second)
                queue.append(second)
    return frozenset(ordered[index] for index in reachable)


def _signed_cut_weights(
    members: frozenset[Node],
    first: frozenset[Node],
    attractive: dict[NodePair, float],
    repulsive: dict[NodePair, float],
) -> tuple[float, float]:
    severed_attractive = sum(
        weight
        for (left, right), weight in attractive.items()
        if left in members
        and right in members
        and ((left in first) != (right in first))
    )
    separated_repulsive = sum(
        weight
        for (left, right), weight in repulsive.items()
        if left in members
        and right in members
        and ((left in first) != (right in first))
    )
    return severed_attractive, separated_repulsive


def _improve_signed_cut(
    members: frozenset[Node],
    initial_first: frozenset[Node],
    attractive: dict[NodePair, float],
    repulsive: dict[NodePair, float],
    source: Node,
    target: Node,
) -> frozenset[Node]:
    """Improve a seeded cut by deterministic positive-gain node flips."""

    first = set(initial_first)
    signed_neighbors: dict[Node, dict[Node, float]] = defaultdict(
        lambda: defaultdict(float)
    )
    for (left, right), weight in attractive.items():
        if left in members and right in members:
            signed_neighbors[left][right] -= weight
            signed_neighbors[right][left] -= weight
    for (left, right), weight in repulsive.items():
        if left in members and right in members:
            signed_neighbors[left][right] += weight
            signed_neighbors[right][left] += weight

    while True:
        best_node: Node | None = None
        best_gain = 1.0e-12
        for node in sorted(members, key=repr):
            if node == source or node == target:
                continue
            node_in_first = node in first
            gain = sum(
                (-weight if ((neighbor in first) != node_in_first) else weight)
                for neighbor, weight in signed_neighbors.get(node, {}).items()
            )
            if gain > best_gain + 1.0e-12 or (
                abs(gain - best_gain) <= 1.0e-12
                and best_node is not None
                and repr(node) < repr(best_node)
            ):
                best_node = node
                best_gain = gain
        if best_node is None:
            break
        if best_node in first:
            first.remove(best_node)
        else:
            first.add(best_node)
    return frozenset(first)


def _refine_signed_splits(
    components: Iterable[set[Node]],
    attractive: dict[NodePair, float],
    repulsive: dict[NodePair, float],
) -> tuple[tuple[frozenset[Node], ...], tuple[SignedSplit, ...]]:
    """Recursively accept exact positive-gain cuts seeded by repulsive pairs."""

    pending = deque(
        frozenset(values)
        for values in sorted(
            components,
            key=lambda values: min((repr(value) for value in values), default=""),
        )
    )
    finished: list[frozenset[Node]] = []
    records: list[SignedSplit] = []
    while pending:
        members = pending.popleft()
        internal_repulsive = tuple(
            (pair, weight)
            for pair, weight in repulsive.items()
            if pair[0] in members and pair[1] in members
        )
        if not internal_repulsive:
            finished.append(members)
            continue
        proposals: dict[frozenset[Node], SignedSplit] = {}
        for (source, target), _weight in sorted(
            internal_repulsive,
            key=lambda value: (-value[1], repr(value[0])),
        ):
            first = _minimum_attractive_cut(
                members,
                attractive,
                source,
                target,
            )
            first = _improve_signed_cut(
                members,
                first,
                attractive,
                repulsive,
                source,
                target,
            )
            second = members - first
            if not first or not second:
                continue
            # Canonicalize complementary cuts so multiple repulsive seeds do
            # not repeatedly score the same bipartition.
            first_key = tuple(sorted((repr(value) for value in first)))
            second_key = tuple(sorted((repr(value) for value in second)))
            canonical_first = first if first_key <= second_key else second
            canonical_second = members - canonical_first
            severed_attractive, separated_repulsive = _signed_cut_weights(
                members,
                canonical_first,
                attractive,
                repulsive,
            )
            record = SignedSplit(
                source,
                target,
                len(members),
                len(canonical_first),
                len(canonical_second),
                severed_attractive,
                separated_repulsive,
            )
            prior = proposals.get(canonical_first)
            if prior is None or record.gain > prior.gain:
                proposals[canonical_first] = record
        improving = tuple(value for value in proposals.values() if value.gain > 1.0e-12)
        if not improving:
            finished.append(members)
            continue
        best = max(
            improving,
            key=lambda value: (
                value.gain,
                value.separated_repulsive_weight,
                -value.severed_attractive_weight,
                repr(value.source),
                repr(value.target),
            ),
        )
        first = _minimum_attractive_cut(
            members,
            attractive,
            best.source,
            best.target,
        )
        first = _improve_signed_cut(
            members,
            first,
            attractive,
            repulsive,
            best.source,
            best.target,
        )
        second = members - first
        records.append(best)
        for values in sorted(
            (first, second),
            key=lambda values: min((repr(value) for value in values), default=""),
        ):
            pending.append(values)
    finished.sort(key=lambda values: min((repr(value) for value in values), default=""))
    return tuple(finished), tuple(records)


def refine_signed_components(
    nodes: Iterable[Node],
    components: Iterable[Iterable[Node]],
    attractive_edges: Iterable[SignedEdge],
    repulsive_edges: Iterable[SignedEdge],
) -> SignedPartition:
    """Split supplied components only when doing so improves signed evidence."""

    node_values = tuple(nodes)
    node_set = set(node_values)
    if len(node_set) != len(node_values):
        raise ValueError("signed graph nodes must be unique")
    supplied = tuple(frozenset(values) for values in components)
    flattened = tuple(value for values in supplied for value in values)
    if len(flattened) != len(set(flattened)) or set(flattened) != node_set:
        raise ValueError("signed components must partition the supplied nodes")

    def collect(edges: Iterable[SignedEdge]) -> dict[NodePair, float]:
        result: dict[NodePair, float] = defaultdict(float)
        for value in edges:
            if value.first not in node_set or value.second not in node_set:
                raise ValueError("signed graph edge references an absent node")
            result[_ordered_pair(value.first, value.second)] += value.weight
        return dict(result)

    attractive = collect(attractive_edges)
    repulsive = collect(repulsive_edges)
    refined, splits = _refine_signed_splits(
        (set(values) for values in supplied),
        attractive,
        repulsive,
    )
    canonical_members = {
        min(values, key=repr): tuple(sorted(values, key=repr))
        for values in refined
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
        tuple(),
        splits,
        sum(attractive.values()),
        sum(repulsive.values()),
        internal_attractive,
        internal_repulsive,
        0,
        "positive-gain signed split refinement of supplied components",
    )


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

    refined_members, splits = _refine_signed_splits(
        members.values(),
        attractive,
        repulsive,
    )
    canonical_members = {
        min(values, key=repr): tuple(sorted(values, key=repr))
        for values in refined_members
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
        splits,
        sum(attractive.values()),
        sum(repulsive.values()),
        internal_attractive,
        internal_repulsive,
        hard_rejections,
        "greedy signed agglomeration with reversible split refinement",
    )
