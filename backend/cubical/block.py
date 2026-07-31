from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Hashable, Iterable, Mapping

import numpy as np

from .geometry import ClippedPatch, FaceTrace
from .matching import (
    TraceMatch,
    TraceMatchSettings,
    align_face_patches,
    face_patch_ranks,
    match_face_traces,
)
from .topology import Float3, GridEdge, GridFace, GridSpec, Int3, cell_face


PatchPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class BlockBounds:
    start_cell_xyz: Int3
    stop_cell_xyz_exclusive: Int3

    def __post_init__(self) -> None:
        start = tuple(int(value) for value in self.start_cell_xyz)
        stop = tuple(int(value) for value in self.stop_cell_xyz_exclusive)
        if len(start) != 3 or len(stop) != 3:
            raise ValueError("block bounds require XYZ triples")
        if any(stop[axis] <= start[axis] for axis in range(3)):
            raise ValueError("block bounds must have positive extent")
        object.__setattr__(self, "start_cell_xyz", start)
        object.__setattr__(self, "stop_cell_xyz_exclusive", stop)

    @property
    def shape_cells_xyz(self) -> Int3:
        return tuple(
            self.stop_cell_xyz_exclusive[axis] - self.start_cell_xyz[axis]
            for axis in range(3)
        )  # type: ignore[return-value]

    def contains_cell(self, cell_xyz: Int3) -> bool:
        return all(
            self.start_cell_xyz[axis]
            <= cell_xyz[axis]
            < self.stop_cell_xyz_exclusive[axis]
            for axis in range(3)
        )

    def contains_face_on_boundary(self, face: GridFace) -> bool:
        coordinate = face.anchor_xyz[face.axis]
        return coordinate in (
            self.start_cell_xyz[face.axis],
            self.stop_cell_xyz_exclusive[face.axis],
        ) and all(
            self.start_cell_xyz[axis]
            <= face.anchor_xyz[axis]
            < self.stop_cell_xyz_exclusive[axis]
            for axis in range(3)
            if axis != face.axis
        )


@dataclass(frozen=True, slots=True)
class WeldedCrossing:
    supporting_edges: tuple[GridEdge, ...]
    edge: GridEdge | None
    grid_vertex_xyz: Int3 | None
    point_xyz: Float3
    t: float | None
    variance: float | None
    observations: tuple[tuple[int, GridEdge], ...]
    maximum_standardized_residual: float


@dataclass(frozen=True, slots=True)
class BoundaryTrace:
    component_id: int
    patch_id: int
    trace: FaceTrace


@dataclass(frozen=True, slots=True)
class ComponentSummary:
    component_id: int
    patch_ids: tuple[int, ...]
    exterior_trace_count: int
    unresolved_interior_trace_count: int


@dataclass(frozen=True, slots=True)
class DeferredJoin:
    match: TraceMatch
    reason: str


@dataclass(frozen=True, slots=True)
class SurfaceJoinSelection:
    """A topology-safe retained edge set before mesh welding/materialization."""

    joins: tuple[TraceMatch, ...]
    deferred_joins: tuple[DeferredJoin, ...]


@dataclass(frozen=True, slots=True)
class SurfaceBlock:
    grid: GridSpec
    bounds: BlockBounds
    patches: tuple[ClippedPatch, ...]
    candidate_joins: tuple[TraceMatch, ...]
    joins: tuple[TraceMatch, ...]
    deferred_joins: tuple[DeferredJoin, ...]
    component_by_patch: tuple[tuple[int, int], ...]
    components: tuple[ComponentSummary, ...]
    welded_crossings: tuple[WeldedCrossing, ...]
    exterior_traces: tuple[BoundaryTrace, ...]
    unresolved_interior_traces: tuple[BoundaryTrace, ...]

    def component_for_patch(self, patch_id: int) -> int:
        values = dict(self.component_by_patch)
        if patch_id not in values:
            raise KeyError(patch_id)
        return values[patch_id]


class _DisjointSet:
    def __init__(self, values: Iterable[object]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: object) -> object:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: object, second: object) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if repr(first_root) <= repr(second_root):
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


class _ParityDisjointSet:
    """Disjoint set carrying one binary orientation relation per member."""

    def __init__(self, values: Iterable[int]) -> None:
        self.parent = {value: value for value in values}
        self.parity = {value: 0 for value in values}
        self.size = {value: 1 for value in values}

    def find(self, value: int) -> tuple[int, int]:
        parent = self.parent[value]
        if parent != value:
            root, relative = self.find(parent)
            self.parity[value] ^= relative
            self.parent[value] = root
        return self.parent[value], self.parity[value]

    def compatible(self, first: int, second: int, required_xor: bool) -> bool:
        first_root, first_parity = self.find(first)
        second_root, second_parity = self.find(second)
        return first_root != second_root or (
            (first_parity ^ second_parity) == int(required_xor)
        )

    def union(self, first: int, second: int, required_xor: bool) -> None:
        first_root, first_parity = self.find(first)
        second_root, second_parity = self.find(second)
        if first_root == second_root:
            if (first_parity ^ second_parity) != int(required_xor):
                raise ValueError("orientation union contradicts retained parity")
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
            first_parity, second_parity = second_parity, first_parity
        self.parent[second_root] = first_root
        self.parity[second_root] = (
            first_parity ^ second_parity ^ int(required_xor)
        )
        self.size[first_root] += self.size.pop(second_root)


class _IntegerPotentialDisjointSet:
    """Disjoint set carrying an integer gauge difference between nodes."""

    def __init__(self, values: Iterable[Hashable]) -> None:
        self.parent = {value: value for value in values}
        # potential[value] = gauge(value) - gauge(parent(value)).
        self.potential = {value: 0 for value in self.parent}
        self.size = {value: 1 for value in self.parent}

    def find(self, value: Hashable) -> tuple[Hashable, int]:
        parent = self.parent[value]
        if parent != value:
            root, relative = self.find(parent)
            self.potential[value] += relative
            self.parent[value] = root
        return self.parent[value], self.potential[value]

    def compatible(
        self, first: Hashable, second: Hashable, required_difference: int
    ) -> bool:
        """Test gauge(second) - gauge(first) == required_difference."""

        first_root, first_potential = self.find(first)
        second_root, second_potential = self.find(second)
        return first_root != second_root or (
            second_potential - first_potential == required_difference
        )

    def union(
        self, first: Hashable, second: Hashable, required_difference: int
    ) -> None:
        first_root, first_potential = self.find(first)
        second_root, second_potential = self.find(second)
        if first_root == second_root:
            if second_potential - first_potential != required_difference:
                raise ValueError("integer potential union contradicts node gauge")
            return
        if self.size[first_root] >= self.size[second_root]:
            self.parent[second_root] = first_root
            self.potential[second_root] = (
                required_difference + first_potential - second_potential
            )
            self.size[first_root] += self.size.pop(second_root)
        else:
            self.parent[first_root] = second_root
            self.potential[first_root] = (
                -required_difference + second_potential - first_potential
            )
            self.size[second_root] += self.size.pop(first_root)


def join_orientation_xor(
    patch_by_id: dict[int, ClippedPatch], match: TraceMatch
) -> bool:
    """Return whether one polygon loop must flip across a matched face trace."""

    first_trace = patch_by_id[match.first_patch_id].trace_on(match.face)
    second_trace = patch_by_id[match.second_patch_id].trace_on(match.face)
    if first_trace is None or second_trace is None:
        raise ValueError("join orientation requires both matched face traces")
    mapping = {
        agreement.first_edge: agreement.second_edge
        for agreement in match.endpoint_agreements
    }
    if set(mapping) != first_trace.endpoint_edges or set(mapping.values()) != (
        second_trace.endpoint_edges
    ):
        raise ValueError("join endpoints do not define a trace orientation map")
    # Shared polygon boundaries must run oppositely in an orientable surface.
    # If their stored trace directions agree, exactly one loop must be flipped.
    return mapping[first_trace.first.edge] == second_trace.first.edge


def _common_crossing_feature(edges: set[GridEdge]) -> object | None:
    if len(edges) == 1:
        return next(iter(edges))
    shared_vertices = set.intersection(
        *(set(edge.endpoint_vertices()) for edge in edges)
    )
    return next(iter(shared_vertices)) if len(shared_vertices) == 1 else None


def _select_consistent_joins(
    patches: tuple[ClippedPatch, ...],
    candidates: tuple[TraceMatch, ...],
    fixed_join_keys: frozenset[tuple[int, int, int, Int3]] = frozenset(),
    candidate_priorities: Mapping[
        tuple[int, int, int, Int3], float
    ] | None = None,
    incompatible_patch_pairs: frozenset[PatchPair] = frozenset(),
    stack_rank_by_patch: Mapping[int, int] | None = None,
) -> tuple[tuple[TraceMatch, ...], tuple[DeferredJoin, ...]]:
    patch_by_id = {value.patch_id: value for value in patches}
    patch_set = _DisjointSet(patch_by_id)
    orientation_set = _ParityDisjointSet(patch_by_id)
    component_cells: dict[object, set[Int3]] = {
        patch_id: {patch.cell_xyz} for patch_id, patch in patch_by_id.items()
    }
    component_members: dict[object, set[int]] = {
        patch_id: {patch_id} for patch_id in patch_by_id
    }
    incompatible_by_patch: dict[int, set[int]] = defaultdict(set)
    for first, second in incompatible_patch_pairs:
        incompatible_by_patch[first].add(second)
        incompatible_by_patch[second].add(first)
    observations = [
        (patch.patch_id, vertex.edge)
        for patch in patches
        for vertex in patch.vertices
    ]
    crossing_set = _DisjointSet(observations)
    crossing_edges: dict[object, set[GridEdge]] = {
        value: {value[1]} for value in observations
    }
    crossing_owners: dict[object, dict[int, GridEdge]] = {
        value: {value[0]: value[1]} for value in observations
    }
    patches_by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in patches:
        patches_by_cell[patch.cell_xyz].append(patch)
    stack_transport = (
        _IntegerPotentialDisjointSet(patch_by_id)
        if stack_rank_by_patch is not None
        else None
    )
    used_traces: set[tuple[int, GridFace]] = set()
    retained_face_ranks: dict[GridFace, list[tuple[int, int]]] = defaultdict(list)
    face_rank_cache: dict[GridFace, tuple[dict[int, int], dict[int, int]]] = {}

    def key(match: TraceMatch) -> tuple[int, int, int, Int3]:
        return (
            match.first_patch_id,
            match.second_patch_id,
            match.face.axis,
            match.face.anchor_xyz,
        )

    def match_ranks(match: TraceMatch) -> tuple[int, int]:
        face = match.face
        if face not in face_rank_cache:
            lower, upper = face.adjacent_cells()
            lower_ranks, upper_ranks, _ = face_patch_ranks(
                patches_by_cell.get(lower, ()),
                patches_by_cell.get(upper, ()),
                face,
            )
            face_rank_cache[face] = lower_ranks, upper_ranks
        lower_ranks, upper_ranks = face_rank_cache[face]
        first_cell = patch_by_id[match.first_patch_id].cell_xyz
        second_cell = patch_by_id[match.second_patch_id].cell_xyz
        lower, upper = face.adjacent_cells()
        if first_cell == lower and second_cell == upper:
            return (
                lower_ranks[match.first_patch_id],
                upper_ranks[match.second_patch_id],
            )
        if second_cell == lower and first_cell == upper:
            return (
                lower_ranks[match.second_patch_id],
                upper_ranks[match.first_patch_id],
            )
        raise ValueError("join patches do not occupy opposite sides of its face")

    def crossing_pairs(match: TraceMatch) -> list[tuple[object, object]]:
        return [
            (
                (match.first_patch_id, value.first_edge),
                (match.second_patch_id, value.second_edge),
            )
            for value in match.endpoint_agreements
        ]

    def stack_transport_relation(
        match: TraceMatch,
    ) -> tuple[int, int, int]:
        if stack_rank_by_patch is None:
            raise RuntimeError("stack transport relation was not configured")
        lower, upper = match.face.adjacent_cells()
        first_cell = patch_by_id[match.first_patch_id].cell_xyz
        second_cell = patch_by_id[match.second_patch_id].cell_xyz
        if first_cell == lower and second_cell == upper:
            lower_patch = match.first_patch_id
            upper_patch = match.second_patch_id
        elif second_cell == lower and first_cell == upper:
            lower_patch = match.second_patch_id
            upper_patch = match.first_patch_id
        else:
            raise ValueError("join does not cross its declared adjacent cells")
        return (
            lower_patch,
            upper_patch,
            int(stack_rank_by_patch[lower_patch])
            - int(stack_rank_by_patch[upper_patch]),
        )

    def crossings_remain_feasible(match: TraceMatch) -> bool:
        pairs = crossing_pairs(match)
        roots = {
            crossing_set.find(value) for pair in pairs for value in pair
        }
        adjacency: dict[object, set[object]] = {root: set() for root in roots}
        for first, second in pairs:
            first_root = crossing_set.find(first)
            second_root = crossing_set.find(second)
            adjacency[first_root].add(second_root)
            adjacency[second_root].add(first_root)
        remaining = set(roots)
        while remaining:
            seed = next(iter(remaining))
            group = {seed}
            frontier = [seed]
            while frontier:
                current = frontier.pop()
                for neighbor in adjacency[current] - group:
                    group.add(neighbor)
                    frontier.append(neighbor)
            remaining -= group
            edges: set[GridEdge] = set()
            owners: dict[int, GridEdge] = {}
            for root in group:
                edges.update(crossing_edges[root])
                for patch_id, edge in crossing_owners[root].items():
                    if patch_id in owners and owners[patch_id] != edge:
                        return False
                    owners[patch_id] = edge
            if _common_crossing_feature(edges) is None:
                return False
        return True

    def union_crossings(match: TraceMatch) -> None:
        for first, second in crossing_pairs(match):
            first_root = crossing_set.find(first)
            second_root = crossing_set.find(second)
            if first_root == second_root:
                continue
            edges = crossing_edges.pop(first_root) | crossing_edges.pop(second_root)
            owners = crossing_owners.pop(first_root)
            owners.update(crossing_owners.pop(second_root))
            crossing_set.union(first_root, second_root)
            root = crossing_set.find(first_root)
            crossing_edges[root] = edges
            crossing_owners[root] = owners

    def components_are_incompatible(first_root: object, second_root: object) -> bool:
        first_members = component_members[first_root]
        second_members = component_members[second_root]
        if len(first_members) > len(second_members):
            first_members, second_members = second_members, first_members
        return any(
            incompatible_by_patch.get(patch_id, set()) & second_members
            for patch_id in first_members
        )

    retained: list[TraceMatch] = []
    deferred: list[DeferredJoin] = []
    ordered = sorted(
        candidates,
        key=lambda value: (
            0
            if key(value) in fixed_join_keys
            else 1,
            -(
                candidate_priorities.get(key(value), value.score)
                if candidate_priorities is not None
                else value.score
            ),
            value.negative_log_likelihood,
            value.face.axis,
            value.face.anchor_xyz,
            value.first_patch_id,
            value.second_patch_id,
        ),
    )
    for match in ordered:
        if not match.accepted:
            deferred.append(DeferredJoin(match, "pair-gate"))
            continue
        trace_keys = (
            (match.first_patch_id, match.face),
            (match.second_patch_id, match.face),
        )
        if any(value in used_traces for value in trace_keys):
            deferred.append(DeferredJoin(match, "trace-occupancy"))
            continue
        first_rank, second_rank = match_ranks(match)
        if any(
            (first_rank - retained_first) * (second_rank - retained_second) < 0
            for retained_first, retained_second in retained_face_ranks[match.face]
        ):
            deferred.append(DeferredJoin(match, "face-order-crossing"))
            continue
        first_root = patch_set.find(match.first_patch_id)
        second_root = patch_set.find(match.second_patch_id)
        if first_root != second_root and (
            component_cells[first_root] & component_cells[second_root]
        ):
            deferred.append(DeferredJoin(match, "component-cell-collision"))
            continue
        if first_root != second_root and components_are_incompatible(
            first_root, second_root
        ):
            deferred.append(DeferredJoin(match, "component-layer-exclusion"))
            continue
        transport_relation: tuple[int, int, int] | None = None
        if stack_transport is not None:
            transport_relation = stack_transport_relation(match)
            if not stack_transport.compatible(*transport_relation):
                deferred.append(DeferredJoin(match, "stack-transport-cycle"))
                continue
        if not crossings_remain_feasible(match):
            deferred.append(DeferredJoin(match, "crossing-topology-cycle"))
            continue
        required_orientation_xor = join_orientation_xor(patch_by_id, match)
        if not orientation_set.compatible(
            match.first_patch_id,
            match.second_patch_id,
            required_orientation_xor,
        ):
            deferred.append(DeferredJoin(match, "orientation-parity-cycle"))
            continue
        union_crossings(match)
        orientation_set.union(
            match.first_patch_id,
            match.second_patch_id,
            required_orientation_xor,
        )
        if stack_transport is not None and transport_relation is not None:
            stack_transport.union(*transport_relation)
        if first_root != second_root:
            cells = component_cells.pop(first_root) | component_cells.pop(second_root)
            members = component_members.pop(first_root) | component_members.pop(
                second_root
            )
            patch_set.union(first_root, second_root)
            merged_root = patch_set.find(first_root)
            component_cells[merged_root] = cells
            component_members[merged_root] = members
        used_traces.update(trace_keys)
        retained_face_ranks[match.face].append((first_rank, second_rank))
        retained.append(match)
    retained.sort(
        key=lambda value: (
            value.face.axis,
            value.face.anchor_xyz,
            value.first_patch_id,
            value.second_patch_id,
        )
    )
    deferred.sort(
        key=lambda value: (
            value.match.face.axis,
            value.match.face.anchor_xyz,
            value.match.first_patch_id,
            value.match.second_patch_id,
        )
    )
    return tuple(retained), tuple(deferred)


def _patch_catalog(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
) -> tuple[tuple[ClippedPatch, ...], dict[int, ClippedPatch]]:
    values = tuple(sorted(patches, key=lambda value: value.patch_id))
    by_id: dict[int, ClippedPatch] = {}
    for patch in values:
        if patch.patch_id in by_id:
            raise ValueError(f"duplicate patch ID {patch.patch_id}")
        if not grid.contains_cell(patch.cell_xyz) or not bounds.contains_cell(
            patch.cell_xyz
        ):
            raise ValueError(f"patch {patch.patch_id} lies outside its block")
        by_id[patch.patch_id] = patch
    return values, by_id


def _trace_endpoint_lookup(trace: FaceTrace) -> dict[GridEdge, object]:
    return {trace.first.edge: trace.first, trace.second.edge: trace.second}


def _summarize_block(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
    candidate_joins: Iterable[TraceMatch],
    *,
    fixed_join_keys: frozenset[tuple[int, int, int, Int3]] = frozenset(),
    candidate_priorities: Mapping[
        tuple[int, int, int, Int3], float
    ] | None = None,
) -> SurfaceBlock:
    patch_values, patch_by_id = _patch_catalog(grid, bounds, patches)
    candidate_values = tuple(
        sorted(
            candidate_joins,
            key=lambda value: (
                value.face.axis,
                value.face.anchor_xyz,
                value.first_patch_id,
                value.second_patch_id,
            ),
        )
    )
    join_values, deferred_values = _select_consistent_joins(
        patch_values,
        candidate_values,
        fixed_join_keys,
        candidate_priorities,
    )
    patch_set = _DisjointSet(patch_by_id)
    crossing_observations = [
        (patch.patch_id, vertex.edge)
        for patch in patch_values
        for vertex in patch.vertices
    ]
    crossing_set = _DisjointSet(crossing_observations)
    joined_trace_keys: set[tuple[int, GridFace]] = set()
    for join in join_values:
        if not join.accepted:
            raise ValueError("surface blocks can contain only accepted joins")
        if join.first_patch_id not in patch_by_id or join.second_patch_id not in patch_by_id:
            raise ValueError("join references a patch outside the block")
        first_patch = patch_by_id[join.first_patch_id]
        second_patch = patch_by_id[join.second_patch_id]
        first_trace = first_patch.trace_on(join.face)
        second_trace = second_patch.trace_on(join.face)
        if first_trace is None or second_trace is None:
            raise ValueError("join references a face not crossed by both patches")
        first_endpoints = _trace_endpoint_lookup(first_trace)
        second_endpoints = _trace_endpoint_lookup(second_trace)
        patch_set.union(join.first_patch_id, join.second_patch_id)
        if len(join.endpoint_agreements) != 2:
            raise ValueError("accepted join does not contain two endpoint agreements")
        for agreement in join.endpoint_agreements:
            if (
                agreement.first_edge not in first_endpoints
                or agreement.second_edge not in second_endpoints
            ):
                raise ValueError("accepted join endpoint is absent from its trace")
            crossing_set.union(
                (join.first_patch_id, agreement.first_edge),
                (join.second_patch_id, agreement.second_edge),
            )
        joined_trace_keys.add((join.first_patch_id, join.face))
        joined_trace_keys.add((join.second_patch_id, join.face))

    root_members: dict[object, list[int]] = defaultdict(list)
    for patch_id in patch_by_id:
        root_members[patch_set.find(patch_id)].append(patch_id)
    component_by_patch: dict[int, int] = {}
    for members in root_members.values():
        component_id = min(members)
        for patch_id in members:
            component_by_patch[patch_id] = component_id

    crossing_values: dict[tuple[int, GridEdge], object] = {}
    for patch in patch_values:
        for vertex in patch.vertices:
            crossing_values[(patch.patch_id, vertex.edge)] = vertex
    crossing_groups: dict[object, list[tuple[int, GridEdge]]] = defaultdict(list)
    for observation in crossing_observations:
        crossing_groups[crossing_set.find(observation)].append(observation)
    welded: list[WeldedCrossing] = []
    for observations in crossing_groups.values():
        edges = {value[1] for value in observations}
        vertices = [crossing_values[value] for value in observations]
        if len(edges) == 1:
            edge = next(iter(edges))
            variances = np.asarray(
                [max(float(value.variance), 1.0e-12) for value in vertices]
            )
            weights = 1.0 / variances
            coordinates = np.asarray([float(value.t) for value in vertices])
            coordinate = float(np.sum(weights * coordinates) / np.sum(weights))
            fused_variance = float(1.0 / np.sum(weights))
            standardized = np.abs(coordinates - coordinate) / np.sqrt(
                variances + fused_variance
            )
            point = edge.point_world(grid, coordinate)
            welded.append(
                WeldedCrossing(
                    supporting_edges=(edge,),
                    edge=edge,
                    grid_vertex_xyz=None,
                    point_xyz=tuple(float(value) for value in point),
                    t=coordinate,
                    variance=fused_variance,
                    observations=tuple(sorted(observations)),
                    maximum_standardized_residual=float(np.max(standardized)),
                )
            )
        else:
            common_vertices = set.intersection(
                *(set(edge.endpoint_vertices()) for edge in edges)
            )
            if len(common_vertices) != 1:
                raise RuntimeError(
                    "a multi-edge welded crossing does not share one grid vertex"
                )
            grid_vertex = next(iter(common_vertices))
            standardized = []
            for vertex in vertices:
                start, stop = vertex.edge.endpoint_vertices()
                coordinate = (
                    float(vertex.t)
                    if grid_vertex == start
                    else 1.0 - float(vertex.t)
                )
                standardized.append(
                    coordinate
                    / math.sqrt(max(float(vertex.variance), 1.0e-12))
                )
            point = grid.vertex_world(grid_vertex)
            welded.append(
                WeldedCrossing(
                    supporting_edges=tuple(sorted(edges)),
                    edge=None,
                    grid_vertex_xyz=grid_vertex,
                    point_xyz=tuple(float(value) for value in point),
                    t=None,
                    variance=None,
                    observations=tuple(sorted(observations)),
                    maximum_standardized_residual=float(np.max(standardized)),
                )
            )
    welded.sort(
        key=lambda value: (
            value.supporting_edges[0].axis,
            value.supporting_edges[0].anchor_xyz,
            value.t if value.t is not None else -1.0,
        )
    )

    exterior: list[BoundaryTrace] = []
    unresolved: list[BoundaryTrace] = []
    for patch in patch_values:
        component_id = component_by_patch[patch.patch_id]
        for trace in patch.traces:
            value = BoundaryTrace(component_id, patch.patch_id, trace)
            if bounds.contains_face_on_boundary(trace.face):
                exterior.append(value)
            elif (patch.patch_id, trace.face) not in joined_trace_keys:
                unresolved.append(value)
    exterior.sort(
        key=lambda value: (
            value.trace.face.axis,
            value.trace.face.anchor_xyz,
            value.component_id,
            value.patch_id,
        )
    )
    unresolved.sort(
        key=lambda value: (
            value.trace.face.axis,
            value.trace.face.anchor_xyz,
            value.component_id,
            value.patch_id,
        )
    )
    exterior_counts = defaultdict(int)
    unresolved_counts = defaultdict(int)
    for value in exterior:
        exterior_counts[value.component_id] += 1
    for value in unresolved:
        unresolved_counts[value.component_id] += 1
    components = tuple(
        ComponentSummary(
            component_id=min(members),
            patch_ids=tuple(sorted(members)),
            exterior_trace_count=exterior_counts[min(members)],
            unresolved_interior_trace_count=unresolved_counts[min(members)],
        )
        for members in sorted(root_members.values(), key=lambda value: min(value))
    )
    return SurfaceBlock(
        grid=grid,
        bounds=bounds,
        patches=patch_values,
        candidate_joins=candidate_values,
        joins=join_values,
        deferred_joins=deferred_values,
        component_by_patch=tuple(sorted(component_by_patch.items())),
        components=components,
        welded_crossings=tuple(welded),
        exterior_traces=tuple(exterior),
        unresolved_interior_traces=tuple(unresolved),
    )


def assemble_surface_block(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
    settings: TraceMatchSettings | None = None,
) -> SurfaceBlock:
    """Assemble one selected patch configuration in a regular cell block."""

    patch_values, _ = _patch_catalog(grid, bounds, patches)
    by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in patch_values:
        by_cell[patch.cell_xyz].append(patch)
    joins: list[TraceMatch] = []
    for cell in sorted(by_cell):
        for axis in range(3):
            neighbor = list(cell)
            neighbor[axis] += 1
            neighbor_tuple = tuple(neighbor)
            if not bounds.contains_cell(neighbor_tuple) or neighbor_tuple not in by_cell:
                continue
            face = cell_face(cell, axis, 1)
            alignment = align_face_patches(
                by_cell[cell],
                by_cell[neighbor_tuple],
                face,
                settings,
                grid=grid,
            )
            joins.extend(alignment.matches)
    return _summarize_block(grid, bounds, patch_values, joins)


def assemble_surface_block_from_candidates(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
    candidates: Iterable[TraceMatch],
    *,
    candidate_priorities: Mapping[
        tuple[int, int, int, Int3], float
    ] | None = None,
) -> SurfaceBlock:
    """Select a complete sheet graph from an explicit join universe.

    Unlike :func:`assemble_surface_block`, callers may supply alternative
    correspondences for the same face trace. Selection enforces trace
    occupancy and order in addition to component-cell uniqueness, crossing
    topology, and orientability. ``candidate_priorities`` changes only the
    deterministic greedy proposal order; all hard constraints remain exact.
    """

    values = tuple(candidates)
    keys = tuple(
        (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in values
    )
    if len(set(keys)) != len(keys):
        raise ValueError("sheet join candidate universe contains duplicates")
    if candidate_priorities is not None:
        unknown = set(candidate_priorities) - set(keys)
        if unknown:
            raise ValueError(
                "sheet join priorities reference absent candidates: "
                f"{sorted(unknown)[:2]}"
            )
        if any(not math.isfinite(float(value)) for value in candidate_priorities.values()):
            raise ValueError("sheet join priorities must be finite")
    return _summarize_block(
        grid,
        bounds,
        patches,
        values,
        candidate_priorities=candidate_priorities,
    )


def select_surface_joins(
    patches: Iterable[ClippedPatch],
    candidates: Iterable[TraceMatch],
    *,
    fixed_join_keys: frozenset[
        tuple[int, int, int, Int3]
    ] = frozenset(),
    candidate_priorities: Mapping[
        tuple[int, int, int, Int3], float
    ] | None = None,
    incompatible_patch_pairs: frozenset[PatchPair] = frozenset(),
    stack_rank_by_patch: Mapping[int, int] | None = None,
) -> SurfaceJoinSelection:
    """Select a complete topology-safe edge set without welding its geometry.

    This is the inexpensive inference boundary used by sheet-level solvers.
    Candidate order may vary between proposals, while trace occupancy, face
    order, component/cell uniqueness, crossing consistency, and orientability
    remain exact. ``fixed_join_keys`` are replayed first and must all survive.
    ``incompatible_patch_pairs`` are lifted graph constraints: no transitive
    component may contain both endpoints of one pair.  ``stack_rank_by_patch``
    activates a component-local integer gauge: every retained continuation
    must transport local layer order consistently around loops within that
    sheet, without coupling unrelated sheets through one global cell gauge.
    """

    patch_values = tuple(patches)
    patch_ids = {value.patch_id for value in patch_values}
    if len(patch_ids) != len(patch_values):
        raise ValueError("sheet join selection requires unique patch IDs")
    candidate_values = tuple(candidates)
    keys = tuple(
        (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in candidate_values
    )
    key_set = set(keys)
    if len(key_set) != len(keys):
        raise ValueError("sheet join candidate universe contains duplicates")
    missing_patches = {
        patch_id
        for value in candidate_values
        for patch_id in (value.first_patch_id, value.second_patch_id)
        if patch_id not in patch_ids
    }
    if missing_patches:
        raise ValueError(
            "sheet joins reference absent patches: "
            f"{sorted(missing_patches)[:2]}"
        )
    canonical_incompatible = frozenset(
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in incompatible_patch_pairs
    )
    if any(first == second for first, second in canonical_incompatible):
        raise ValueError("a sheetlet cannot exclude itself")
    unknown_incompatible = {
        patch_id
        for pair in canonical_incompatible
        for patch_id in pair
        if patch_id not in patch_ids
    }
    if unknown_incompatible:
        raise ValueError(
            "sheet layer exclusions reference absent patches: "
            f"{sorted(unknown_incompatible)[:2]}"
        )
    if stack_rank_by_patch is not None:
        missing_stack_ranks = patch_ids - set(stack_rank_by_patch)
        extra_stack_ranks = set(stack_rank_by_patch) - patch_ids
        if missing_stack_ranks or extra_stack_ranks:
            raise ValueError(
                "sheet stack ranks must cover exactly the patch universe"
            )
        if any(
            not isinstance(value, int) or isinstance(value, bool) or value < 0
            for value in stack_rank_by_patch.values()
        ):
            raise ValueError("sheet stack ranks must be nonnegative integers")
        ranks_by_cell: dict[Int3, set[int]] = defaultdict(set)
        for patch in patch_values:
            rank = int(stack_rank_by_patch[patch.patch_id])
            if rank in ranks_by_cell[patch.cell_xyz]:
                raise ValueError("sheet stack ranks must be unique within a cell")
            ranks_by_cell[patch.cell_xyz].add(rank)
    missing_fixed = set(fixed_join_keys) - key_set
    if missing_fixed:
        raise ValueError(
            "fixed sheet joins are absent from the candidate universe: "
            f"{sorted(missing_fixed)[:2]}"
        )
    if candidate_priorities is not None:
        unknown = set(candidate_priorities) - key_set
        if unknown:
            raise ValueError(
                "sheet join priorities reference absent candidates: "
                f"{sorted(unknown)[:2]}"
            )
        if any(
            not math.isfinite(float(value))
            for value in candidate_priorities.values()
        ):
            raise ValueError("sheet join priorities must be finite")
    retained, deferred = _select_consistent_joins(
        patch_values,
        candidate_values,
        fixed_join_keys,
        candidate_priorities,
        canonical_incompatible,
        stack_rank_by_patch,
    )
    retained_keys = {
        (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in retained
    }
    rejected_fixed = set(fixed_join_keys) - retained_keys
    if rejected_fixed:
        reasons = Counter(
            value.reason
            for value in deferred
            if (
                value.match.first_patch_id,
                value.match.second_patch_id,
                value.match.face.axis,
                value.match.face.anchor_xyz,
            )
            in rejected_fixed
        )
        raise ValueError(
            "fixed sheet joins became infeasible: "
            f"{dict(sorted(reasons.items()))}"
        )
    return SurfaceJoinSelection(retained, deferred)


def rebuild_surface_block(
    block: SurfaceBlock,
    retained_joins: Iterable[TraceMatch],
) -> SurfaceBlock:
    """Rebuild exact welded geometry from a declared subset of accepted joins.

    This is the post-assembly refinement boundary: callers may remove joins
    using independent evidence, but cannot introduce geometry or silently
    reconsider pair-gated alternatives.
    """

    retained = tuple(retained_joins)
    baseline = {
        (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in block.joins
    }
    for value in retained:
        key = (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        if key not in baseline:
            raise ValueError("refinement cannot introduce a non-retained join")
    if len(retained) != len(
        {
            (
                value.first_patch_id,
                value.second_patch_id,
                value.face.axis,
                value.face.anchor_xyz,
            )
            for value in retained
        }
    ):
        raise ValueError("refinement contains duplicate joins")
    return _summarize_block(
        block.grid,
        block.bounds,
        block.patches,
        retained,
    )


def surface_block_from_retained_joins(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
    retained_joins: Iterable[TraceMatch],
) -> SurfaceBlock:
    """Materialize exact welded geometry from a complete retained graph.

    Unlike :func:`rebuild_surface_block`, this boundary is allowed to introduce
    joins because its input is a serialized graph rather than a refinement of
    an already assembled block.  Every declared join is treated as immutable
    and is replayed through the same collision, crossing-topology, and
    orientability selector used during inference.  A rejected declaration is
    therefore a corrupt or incomplete graph artifact, never a silent change in
    connectivity.
    """

    joins = tuple(retained_joins)
    keys = frozenset(
        (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )
        for value in joins
    )
    if len(keys) != len(joins):
        raise ValueError("retained surface graph contains duplicate joins")
    block = _summarize_block(
        grid,
        bounds,
        patches,
        joins,
        fixed_join_keys=keys,
    )
    if len(block.joins) != len(joins):
        reasons = Counter(value.reason for value in block.deferred_joins)
        raise ValueError(
            "retained surface graph violates global topology constraints: "
            f"{dict(sorted(reasons.items()))}"
        )
    return block


def extend_surface_block_joins(
    block: SurfaceBlock,
    additions: Iterable[TraceMatch],
) -> SurfaceBlock:
    """Add candidate joins while preserving every already retained join.

    This is the connectivity analogue of patch augmentation. Existing geometry
    and joins are immutable; new pair-gated candidates still pass the complete
    collision, crossing-topology, and orientability selector.
    """

    def key(value: TraceMatch) -> tuple[int, int, int, Int3]:
        return (
            value.first_patch_id,
            value.second_patch_id,
            value.face.axis,
            value.face.anchor_xyz,
        )

    candidates = {key(value): value for value in block.candidate_joins}
    for value in additions:
        if not value.accepted:
            raise ValueError("join augmentation accepts only pair-gated candidates")
        candidates.setdefault(key(value), value)
    fixed = frozenset(key(value) for value in block.joins)
    return _summarize_block(
        block.grid,
        block.bounds,
        block.patches,
        candidates.values(),
        fixed_join_keys=fixed,
    )


def augment_surface_block(
    block: SurfaceBlock,
    additions: Iterable[ClippedPatch],
    settings: TraceMatchSettings | None = None,
    *,
    allowed_supports: set[tuple[int, int, GridFace]] | None = None,
) -> SurfaceBlock:
    """Add fitted patches by matching only currently unresolved face traces.

    Existing joins are immutable inputs.  Each new polygon can consume at most
    one unresolved neighbor trace on each of its faces; the complete retained
    graph plus those local proposals is then passed through the same global
    collision, crossing, and orientation selector used by full assembly.
    """

    resolved = settings or TraceMatchSettings()
    current = block
    existing_ids = {value.patch_id for value in current.patches}
    for patch in additions:
        if patch.patch_id in existing_ids:
            raise ValueError(f"augmentation duplicates patch ID {patch.patch_id}")
        if not current.bounds.contains_cell(patch.cell_xyz):
            raise ValueError(f"augmentation patch {patch.patch_id} lies outside block")
        patch_by_id = {value.patch_id: value for value in current.patches}
        open_by_face: dict[GridFace, list[BoundaryTrace]] = defaultdict(list)
        for boundary in current.unresolved_interior_traces:
            open_by_face[boundary.trace.face].append(boundary)
        proposed: list[TraceMatch] = []
        for trace in patch.traces:
            accepted: list[TraceMatch] = []
            for boundary in open_by_face.get(trace.face, ()):
                source = patch_by_id[boundary.patch_id]
                if source.cell_xyz == patch.cell_xyz:
                    continue
                if allowed_supports is not None and (
                    patch.patch_id,
                    source.patch_id,
                    trace.face,
                ) not in allowed_supports:
                    continue
                match = match_face_traces(
                    trace,
                    patch.estimate,
                    boundary.trace,
                    source.estimate,
                    resolved,
                    grid=current.grid,
                )
                if match.accepted:
                    accepted.append(match)
            if accepted:
                proposed.append(
                    max(
                        accepted,
                        key=lambda value: (
                            value.score,
                            -value.negative_log_likelihood,
                            -value.second_patch_id,
                        ),
                    )
                )
        current = _summarize_block(
            current.grid,
            current.bounds,
            (*current.patches, patch),
            (*current.joins, *proposed),
            fixed_join_keys=frozenset(
                (
                    value.first_patch_id,
                    value.second_patch_id,
                    value.face.axis,
                    value.face.anchor_xyz,
                )
                for value in current.joins
            ),
        )
        existing_ids.add(patch.patch_id)
    return current


def _shared_block_face(
    first: BlockBounds, second: BlockBounds
) -> tuple[int, int, bool]:
    candidates: list[tuple[int, int, bool]] = []
    for axis in range(3):
        other_match = all(
            first.start_cell_xyz[other] == second.start_cell_xyz[other]
            and first.stop_cell_xyz_exclusive[other]
            == second.stop_cell_xyz_exclusive[other]
            for other in range(3)
            if other != axis
        )
        if not other_match:
            continue
        if first.stop_cell_xyz_exclusive[axis] == second.start_cell_xyz[axis]:
            candidates.append((axis, first.stop_cell_xyz_exclusive[axis], True))
        if second.stop_cell_xyz_exclusive[axis] == first.start_cell_xyz[axis]:
            candidates.append((axis, second.stop_cell_xyz_exclusive[axis], False))
    if len(candidates) != 1:
        raise ValueError("blocks must tile one complete shared rectangular face")
    return candidates[0]


def merge_surface_blocks(
    first: SurfaceBlock,
    second: SurfaceBlock,
    settings: TraceMatchSettings | None = None,
) -> SurfaceBlock:
    """Compose adjacent blocks by matching only their cached exterior traces."""

    if first.grid != second.grid:
        raise ValueError("surface blocks must use the same grid")
    if {value.patch_id for value in first.patches} & {
        value.patch_id for value in second.patches
    }:
        raise ValueError("surface blocks contain overlapping patch IDs")
    axis, coordinate, first_is_lower = _shared_block_face(first.bounds, second.bounds)
    lower = first if first_is_lower else second
    upper = second if first_is_lower else first
    lower_by_face: dict[GridFace, list[ClippedPatch]] = defaultdict(list)
    upper_by_face: dict[GridFace, list[ClippedPatch]] = defaultdict(list)
    patch_by_id = {
        value.patch_id: value for value in (*first.patches, *second.patches)
    }
    for boundary in lower.exterior_traces:
        if (
            boundary.trace.face.axis == axis
            and boundary.trace.face.anchor_xyz[axis] == coordinate
        ):
            lower_by_face[boundary.trace.face].append(patch_by_id[boundary.patch_id])
    for boundary in upper.exterior_traces:
        if (
            boundary.trace.face.axis == axis
            and boundary.trace.face.anchor_xyz[axis] == coordinate
        ):
            upper_by_face[boundary.trace.face].append(patch_by_id[boundary.patch_id])
    seam_joins: list[TraceMatch] = []
    for face in sorted(set(lower_by_face) | set(upper_by_face)):
        alignment = align_face_patches(
            lower_by_face.get(face, ()),
            upper_by_face.get(face, ()),
            face,
            settings,
            grid=first.grid,
        )
        seam_joins.extend(alignment.matches)
    start = tuple(
        min(first.bounds.start_cell_xyz[axis_index], second.bounds.start_cell_xyz[axis_index])
        for axis_index in range(3)
    )
    stop = tuple(
        max(
            first.bounds.stop_cell_xyz_exclusive[axis_index],
            second.bounds.stop_cell_xyz_exclusive[axis_index],
        )
        for axis_index in range(3)
    )
    return _summarize_block(
        first.grid,
        BlockBounds(start, stop),
        (*first.patches, *second.patches),
        (*first.candidate_joins, *second.candidate_joins, *seam_joins),
    )


def assemble_surface_hierarchy(
    grid: GridSpec,
    bounds: BlockBounds,
    patches: Iterable[ClippedPatch],
    *,
    maximum_leaf_shape_cells_xyz: Int3 = (8, 8, 8),
    settings: TraceMatchSettings | None = None,
) -> SurfaceBlock:
    """Recursively assemble regular leaves, then compose only cached seams."""

    maximum_leaf = tuple(int(value) for value in maximum_leaf_shape_cells_xyz)
    if len(maximum_leaf) != 3 or any(value <= 0 for value in maximum_leaf):
        raise ValueError("maximum leaf shape must be a positive XYZ triple")
    patch_values = tuple(patches)

    def recurse(
        current_bounds: BlockBounds, current_patches: tuple[ClippedPatch, ...]
    ) -> SurfaceBlock:
        shape = current_bounds.shape_cells_xyz
        ratios = [shape[axis] / maximum_leaf[axis] for axis in range(3)]
        axis = int(np.argmax(ratios))
        if ratios[axis] <= 1.0:
            return assemble_surface_block(
                grid, current_bounds, current_patches, settings
            )
        split = current_bounds.start_cell_xyz[axis] + shape[axis] // 2
        first_stop = list(current_bounds.stop_cell_xyz_exclusive)
        first_stop[axis] = split
        second_start = list(current_bounds.start_cell_xyz)
        second_start[axis] = split
        first_bounds = BlockBounds(
            current_bounds.start_cell_xyz, tuple(first_stop)
        )
        second_bounds = BlockBounds(
            tuple(second_start), current_bounds.stop_cell_xyz_exclusive
        )
        first_patches = tuple(
            value for value in current_patches if value.cell_xyz[axis] < split
        )
        second_patches = tuple(
            value for value in current_patches if value.cell_xyz[axis] >= split
        )
        return merge_surface_blocks(
            recurse(first_bounds, first_patches),
            recurse(second_bounds, second_patches),
            settings,
        )

    return recurse(bounds, patch_values)
