from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Hashable, Iterable, Mapping

import numpy as np

from .block import DeferredJoin, join_orientation_xor
from .geometry import ClippedPatch
from .matching import TraceMatch
from .topology import GridEdge, Int3


Feature = GridEdge | Int3
Observation = tuple[int, GridEdge]
JoinKey = tuple[int, int, int, Int3]


@dataclass(frozen=True, slots=True)
class FrozenComponent:
    component_id: int
    total_patch_count: int
    occupied_cells: tuple[Int3, ...]
    anchor_patch_ids: tuple[int, ...]
    anchor_orientation_parity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class FrozenCrossing:
    group_id: int
    feature: Feature
    observations: tuple[Observation, ...]
    owners: tuple[Observation, ...]


@dataclass(frozen=True, slots=True)
class FrozenFaceState:
    axis: int
    side: int
    depth_cells: int
    cut_coordinate: int
    frozen_component_count: int
    detached_component_count: int
    components: tuple[FrozenComponent, ...]
    crossings: tuple[FrozenCrossing, ...]

    @property
    def anchor_patch_ids(self) -> tuple[int, ...]:
        return tuple(
            patch_id
            for component in self.components
            for patch_id in component.anchor_patch_ids
        )


@dataclass(frozen=True, slots=True)
class FrozenRegionState:
    face_mask: int
    depth_cells: int
    frozen_component_count: int
    detached_component_count: int
    components: tuple[FrozenComponent, ...]
    crossings: tuple[FrozenCrossing, ...]

    @property
    def faces(self) -> tuple[tuple[int, int], ...]:
        return faces_from_mask(self.face_mask)

    @property
    def anchor_patch_ids(self) -> tuple[int, ...]:
        return tuple(
            patch_id
            for component in self.components
            for patch_id in component.anchor_patch_ids
        )


def face_mask(faces: Iterable[tuple[int, int]]) -> int:
    result = 0
    axes: set[int] = set()
    for axis, side in faces:
        if axis not in (0, 1, 2) or side not in (0, 1):
            raise ValueError("region faces require axis and side in range")
        if axis in axes:
            raise ValueError("a merge region cannot use both sides of one axis")
        axes.add(axis)
        result |= 1 << (2 * axis + side)
    if result == 0:
        raise ValueError("a frozen region requires at least one mutable face")
    return result


def faces_from_mask(value: int) -> tuple[tuple[int, int], ...]:
    if value <= 0 or value >= 1 << 6:
        raise ValueError("frozen region face mask is invalid")
    result = tuple(
        (axis, side)
        for axis in range(3)
        for side in range(2)
        if value & (1 << (2 * axis + side))
    )
    if len({axis for axis, _ in result}) != len(result):
        raise ValueError("frozen region uses both sides of one axis")
    return result


def compatible_face_masks() -> tuple[int, ...]:
    masks: list[int] = []
    for x_side in (-1, 0, 1):
        for y_side in (-1, 0, 1):
            for z_side in (-1, 0, 1):
                faces = tuple(
                    (axis, side)
                    for axis, side in enumerate((x_side, y_side, z_side))
                    if side >= 0
                )
                if faces:
                    masks.append(face_mask(faces))
    return tuple(sorted(masks))


@dataclass(frozen=True, slots=True)
class ComponentSeed:
    key: Hashable
    total_patch_count: int
    occupied_cells: tuple[Int3, ...]
    anchor_patch_ids: tuple[int, ...]
    anchor_orientation_parity: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class CrossingSeed:
    key: Hashable
    feature: Feature
    observations: tuple[Observation, ...]
    owners: tuple[tuple[Hashable, GridEdge], ...] = ()


@dataclass(frozen=True, slots=True)
class FrozenTopologySeed:
    components: tuple[ComponentSeed, ...]
    crossings: tuple[CrossingSeed, ...]
    detached_component_count: int


@dataclass(frozen=True, slots=True)
class TopologySelection:
    joins: tuple[TraceMatch, ...]
    deferred_joins: tuple[DeferredJoin, ...]
    component_by_patch: tuple[tuple[int, int], ...]
    component_count: int
    detached_component_count: int


class _DisjointSet:
    def __init__(self, values: Iterable[Hashable]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: Hashable) -> Hashable:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: Hashable, second: Hashable) -> Hashable:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return first_root
        if repr(first_root) > repr(second_root):
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        return first_root


class _ParityDisjointSet:
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
                raise ValueError("orientation seed contradicts retained parity")
            return
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
            first_parity, second_parity = second_parity, first_parity
        self.parent[second_root] = first_root
        self.parity[second_root] = (
            first_parity ^ second_parity ^ int(required_xor)
        )
        self.size[first_root] += self.size.pop(second_root)


def _combine_features(first: Feature, second: Feature) -> Feature | None:
    if isinstance(first, GridEdge) and isinstance(second, GridEdge):
        if first == second:
            return first
        shared = set(first.endpoint_vertices()) & set(second.endpoint_vertices())
        return next(iter(shared)) if len(shared) == 1 else None
    if isinstance(first, GridEdge):
        return second if second in first.endpoint_vertices() else None
    if isinstance(second, GridEdge):
        return first if first in second.endpoint_vertices() else None
    return first if first == second else None


class _CrossingState:
    def __init__(self, observations: Iterable[Observation]) -> None:
        values = tuple(observations)
        self.parent: dict[Observation, Observation] = {
            value: value for value in values
        }
        self.feature: dict[Observation, Feature] = {
            value: value[1] for value in values
        }
        self.owners: dict[Observation, dict[Hashable, GridEdge]] = {
            value: {value[0]: value[1]} for value in values
        }

    def find(self, value: Observation) -> Observation:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def _merged_state(
        self,
        roots: Iterable[Observation],
    ) -> tuple[Feature, dict[Hashable, GridEdge]] | None:
        values = tuple(roots)
        feature = self.feature[values[0]]
        owners: dict[Hashable, GridEdge] = {}
        for root in values:
            combined = _combine_features(feature, self.feature[root])
            if combined is None:
                return None
            feature = combined
            for patch_id, edge in self.owners[root].items():
                if patch_id in owners and owners[patch_id] != edge:
                    return None
                owners[patch_id] = edge
        return feature, owners

    def feasible(self, pairs: Iterable[tuple[Observation, Observation]]) -> bool:
        pairs = tuple(pairs)
        roots = {self.find(value) for pair in pairs for value in pair}
        local = _DisjointSet(roots)
        for first, second in pairs:
            local.union(self.find(first), self.find(second))
        groups: dict[Hashable, list[Observation]] = defaultdict(list)
        for root in roots:
            groups[local.find(root)].append(root)
        return all(self._merged_state(group) is not None for group in groups.values())

    def union_pairs(
        self,
        pairs: Iterable[tuple[Observation, Observation]],
    ) -> None:
        pairs = tuple(pairs)
        if not self.feasible(pairs):
            raise ValueError("crossing union is infeasible")
        for first, second in pairs:
            first_root = self.find(first)
            second_root = self.find(second)
            if first_root == second_root:
                continue
            merged = self._merged_state((first_root, second_root))
            if merged is None:
                raise RuntimeError("validated crossing union became infeasible")
            if repr(first_root) > repr(second_root):
                first_root, second_root = second_root, first_root
            self.parent[second_root] = first_root
            self.feature[first_root], self.owners[first_root] = merged
            del self.feature[second_root]
            del self.owners[second_root]

    def apply_seed(self, seed: CrossingSeed) -> None:
        observations = tuple(seed.observations)
        if not observations:
            raise ValueError("frozen crossing seed cannot be empty")
        self.union_pairs(
            (observations[0], value) for value in observations[1:]
        )
        root = self.find(observations[0])
        for value in observations:
            if _combine_features(seed.feature, value[1]) is None:
                raise ValueError("frozen crossing feature excludes an anchor edge")
        self.feature[root] = seed.feature
        owners = self.owners[root]
        for owner, edge in seed.owners:
            if owner in owners and owners[owner] != edge:
                raise ValueError("frozen crossing owner uses multiple cube edges")
            owners[owner] = edge


def _join_key(value: TraceMatch) -> JoinKey:
    return (
        value.first_patch_id,
        value.second_patch_id,
        value.face.axis,
        value.face.anchor_xyz,
    )


def _crossing_pairs(match: TraceMatch) -> tuple[tuple[Observation, Observation], ...]:
    return tuple(
        (
            (match.first_patch_id, value.first_edge),
            (match.second_patch_id, value.second_edge),
        )
        for value in match.endpoint_agreements
    )


def select_joins_with_frozen_topology(
    patches: Iterable[ClippedPatch],
    candidates: Iterable[TraceMatch],
    seed: FrozenTopologySeed,
    *,
    fixed_join_keys: frozenset[JoinKey] = frozenset(),
) -> TopologySelection:
    """Select local joins against immutable component/crossing/orientation state."""

    patch_values = tuple(sorted(patches, key=lambda value: value.patch_id))
    patch_by_id = {value.patch_id: value for value in patch_values}
    if len(patch_by_id) != len(patch_values):
        raise ValueError("joint topology patches require unique IDs")
    component_set = _DisjointSet(patch_by_id)
    orientation_set = _ParityDisjointSet(patch_by_id)
    component_cells: dict[Hashable, set[Int3]] = {
        patch.patch_id: {patch.cell_xyz} for patch in patch_values
    }
    seeded_patch_ids: set[int] = set()
    for component in seed.components:
        anchors = component.anchor_patch_ids
        parity = component.anchor_orientation_parity
        if not anchors or len(anchors) != len(parity):
            raise ValueError("frozen component anchor/parity arrays are invalid")
        if any(value not in patch_by_id for value in anchors):
            raise ValueError("frozen component references an absent anchor patch")
        if seeded_patch_ids & set(anchors):
            raise ValueError("an anchor patch belongs to multiple frozen components")
        seeded_patch_ids.update(anchors)
        first = anchors[0]
        first_parity = int(parity[0])
        for patch_id, relative in zip(anchors[1:], parity[1:]):
            first_root = component_set.find(first)
            second_root = component_set.find(patch_id)
            cells = component_cells.pop(first_root) | component_cells.pop(second_root)
            root = component_set.union(first_root, second_root)
            component_cells[root] = cells
            orientation_set.union(
                first,
                patch_id,
                bool(first_parity ^ int(relative)),
            )
        root = component_set.find(first)
        occupied = set(component.occupied_cells)
        if not component_cells[root] <= occupied:
            raise ValueError("frozen component occupancy omits an anchor cell")
        component_cells[root] = occupied

    observations = tuple(
        (patch.patch_id, vertex.edge)
        for patch in patch_values
        for vertex in patch.vertices
    )
    crossing_state = _CrossingState(observations)
    seeded_observations: set[Observation] = set()
    for crossing in seed.crossings:
        if any(value not in crossing_state.parent for value in crossing.observations):
            raise ValueError("frozen crossing references an absent anchor vertex")
        if seeded_observations & set(crossing.observations):
            raise ValueError("an anchor vertex belongs to multiple frozen crossings")
        seeded_observations.update(crossing.observations)
        crossing_state.apply_seed(crossing)

    candidate_values = tuple(candidates)
    candidate_keys = {_join_key(value) for value in candidate_values}
    if len(candidate_keys) != len(candidate_values):
        raise ValueError("joint topology candidates contain duplicate joins")
    missing_fixed = fixed_join_keys - candidate_keys
    if missing_fixed:
        raise ValueError(f"fixed topology joins are absent: {sorted(missing_fixed)[:2]}")
    retained: list[TraceMatch] = []
    deferred: list[DeferredJoin] = []
    for match in sorted(
        candidate_values,
        key=lambda value: (
            0 if _join_key(value) in fixed_join_keys else 1,
            -value.score,
            value.negative_log_likelihood,
            value.face.axis,
            value.face.anchor_xyz,
            value.first_patch_id,
            value.second_patch_id,
        ),
    ):
        reason: str | None = None
        if not match.accepted:
            reason = "pair-gate"
        else:
            first_root = component_set.find(match.first_patch_id)
            second_root = component_set.find(match.second_patch_id)
            if first_root != second_root and (
                component_cells[first_root] & component_cells[second_root]
            ):
                reason = "component-cell-collision"
            elif not crossing_state.feasible(_crossing_pairs(match)):
                reason = "crossing-topology-cycle"
            else:
                required_xor = join_orientation_xor(patch_by_id, match)
                if not orientation_set.compatible(
                    match.first_patch_id,
                    match.second_patch_id,
                    required_xor,
                ):
                    reason = "orientation-parity-cycle"
        if reason is not None:
            if _join_key(match) in fixed_join_keys:
                raise ValueError(f"fixed topology join became infeasible: {reason}")
            deferred.append(DeferredJoin(match, reason))
            continue
        first_root = component_set.find(match.first_patch_id)
        second_root = component_set.find(match.second_patch_id)
        crossing_state.union_pairs(_crossing_pairs(match))
        orientation_set.union(
            match.first_patch_id,
            match.second_patch_id,
            join_orientation_xor(patch_by_id, match),
        )
        if first_root != second_root:
            cells = component_cells.pop(first_root) | component_cells.pop(second_root)
            root = component_set.union(first_root, second_root)
            component_cells[root] = cells
        retained.append(match)

    retained.sort(key=_join_key)
    deferred.sort(key=lambda value: _join_key(value.match))
    members: dict[Hashable, list[int]] = defaultdict(list)
    for patch_id in patch_by_id:
        members[component_set.find(patch_id)].append(patch_id)
    component_by_patch = {
        patch_id: min(values)
        for values in members.values()
        for patch_id in values
    }
    return TopologySelection(
        tuple(retained),
        tuple(deferred),
        tuple(sorted(component_by_patch.items())),
        len(members) + seed.detached_component_count,
        seed.detached_component_count,
    )


def _build_frozen_region_state(
    patches: Iterable[ClippedPatch],
    joins: Iterable[TraceMatch],
    shape_cells_xyz: Int3,
    depth_cells: int,
    region_face_mask: int,
) -> FrozenRegionState:
    faces = faces_from_mask(region_face_mask)
    cuts = {
        (axis, depth_cells if side == 0 else shape_cells_xyz[axis] - depth_cells)
        for axis, side in faces
    }
    patch_values = tuple(patches)
    patch_by_id = {value.patch_id: value for value in patch_values}
    join_values = tuple(joins)

    def mutable(patch: ClippedPatch) -> bool:
        return any(
            patch.cell_xyz[axis] < depth_cells
            if side == 0
            else patch.cell_xyz[axis] >= shape_cells_xyz[axis] - depth_cells
            for axis, side in faces
        )

    def cut_trace(patch_id: int, edge: GridEdge | None = None) -> bool:
        return any(
            (trace.face.axis, trace.face.anchor_xyz[trace.face.axis]) in cuts
            and (edge is None or edge in trace.endpoint_edges)
            for trace in patch_by_id[patch_id].traces
        )

    frozen_ids = {
        patch.patch_id for patch in patch_values if not mutable(patch)
    }
    frozen_joins = tuple(
        value
        for value in join_values
        if value.first_patch_id in frozen_ids
        and value.second_patch_id in frozen_ids
    )
    component_set = _DisjointSet(frozen_ids)
    orientation_set = _ParityDisjointSet(frozen_ids)
    for join in frozen_joins:
        component_set.union(join.first_patch_id, join.second_patch_id)
        orientation_set.union(
            join.first_patch_id,
            join.second_patch_id,
            join_orientation_xor(patch_by_id, join),
        )
    members: dict[Hashable, list[int]] = defaultdict(list)
    for patch_id in frozen_ids:
        members[component_set.find(patch_id)].append(patch_id)
    component_id_by_patch = {
        patch_id: min(component_members)
        for component_members in members.values()
        for patch_id in component_members
    }
    anchor_ids = {
        patch_id for patch_id in frozen_ids if cut_trace(patch_id)
    }
    participating = {
        component_id_by_patch[patch_id] for patch_id in anchor_ids
    }
    member_by_component = {
        min(values): tuple(sorted(values)) for values in members.values()
    }
    components: list[FrozenComponent] = []
    for component_id in sorted(participating):
        values = member_by_component[component_id]
        anchors = tuple(sorted(anchor_ids & set(values)))
        representative = anchors[0]
        _, representative_parity = orientation_set.find(representative)
        parity = tuple(
            orientation_set.find(value)[1] ^ representative_parity
            for value in anchors
        )
        cells = tuple(
            sorted({patch_by_id[value].cell_xyz for value in values})
        )
        components.append(
            FrozenComponent(
                component_id,
                len(values),
                cells,
                anchors,
                parity,
            )
        )

    observations = tuple(
        (patch_id, vertex.edge)
        for patch_id in frozen_ids
        for vertex in patch_by_id[patch_id].vertices
    )
    crossing_set = _DisjointSet(observations)
    for join in frozen_joins:
        for agreement in join.endpoint_agreements:
            crossing_set.union(
                (join.first_patch_id, agreement.first_edge),
                (join.second_patch_id, agreement.second_edge),
            )
    crossing_members: dict[Hashable, list[Observation]] = defaultdict(list)
    for observation in observations:
        crossing_members[crossing_set.find(observation)].append(observation)
    referenced = {
        crossing_set.find((patch_id, vertex.edge))
        for patch_id in anchor_ids
        for trace in patch_by_id[patch_id].traces
        if (trace.face.axis, trace.face.anchor_xyz[trace.face.axis]) in cuts
        for vertex in (trace.first, trace.second)
    }
    ordered_crossings = sorted(
        referenced,
        key=lambda value: min(crossing_members[value]),
    )
    crossings: list[FrozenCrossing] = []
    for group_id, root in enumerate(ordered_crossings):
        all_observations = crossing_members[root]
        feature: Feature = all_observations[0][1]
        owner_edges: dict[int, GridEdge] = {}
        for _, edge in all_observations[1:]:
            combined = _combine_features(feature, edge)
            if combined is None:
                raise ValueError(
                    "retained frozen topology has no common crossing feature"
                )
            feature = combined
        for patch_id, edge in all_observations:
            if patch_id in owner_edges and owner_edges[patch_id] != edge:
                raise ValueError(
                    "retained frozen crossing assigns one patch to multiple cube edges"
                )
            owner_edges[patch_id] = edge
        anchor_observations = tuple(
            sorted(
                value
                for value in all_observations
                if value[0] in anchor_ids and cut_trace(value[0], value[1])
            )
        )
        crossings.append(
            FrozenCrossing(
                group_id,
                feature,
                anchor_observations,
                tuple(sorted(owner_edges.items())),
            )
        )
    return FrozenRegionState(
        region_face_mask,
        depth_cells,
        len(members),
        len(members) - len(participating),
        tuple(components),
        tuple(crossings),
    )


def build_frozen_region_states(
    patches: Iterable[ClippedPatch],
    joins: Iterable[TraceMatch],
    shape_cells_xyz: Int3,
    depth_cells: int,
    *,
    face_masks: Iterable[int] | None = None,
) -> tuple[FrozenRegionState, ...]:
    """Summarize immutable topology after removing compatible outer bands."""

    if depth_cells <= 0 or any(2 * depth_cells >= value for value in shape_cells_xyz):
        raise ValueError("frozen region states require a nonempty interior")
    patch_values = tuple(patches)
    join_values = tuple(joins)
    masks = tuple(sorted(set(face_masks or compatible_face_masks())))
    return tuple(
        _build_frozen_region_state(
            patch_values,
            join_values,
            shape_cells_xyz,
            depth_cells,
            value,
        )
        for value in masks
    )


def build_frozen_face_states(
    patches: Iterable[ClippedPatch],
    joins: Iterable[TraceMatch],
    shape_cells_xyz: Int3,
    depth_cells: int,
) -> tuple[FrozenFaceState, ...]:
    """Cut each outer face band away and summarize the immutable remainder."""

    regions = build_frozen_region_states(
        patches,
        joins,
        shape_cells_xyz,
        depth_cells,
        face_masks=(1 << value for value in range(6)),
    )
    result: list[FrozenFaceState] = []
    for region in regions:
        ((axis, side),) = region.faces
        cut = depth_cells if side == 0 else shape_cells_xyz[axis] - depth_cells
        result.append(
            FrozenFaceState(
                axis,
                side,
                depth_cells,
                cut,
                region.frozen_component_count,
                region.detached_component_count,
                region.components,
                region.crossings,
            )
        )
    return tuple(result)


def frozen_topology_arrays(
    states: Iterable[FrozenFaceState],
) -> dict[str, np.ndarray]:
    values = tuple(sorted(states, key=lambda value: (value.axis, value.side)))
    component_values = tuple(
        component for state in values for component in state.components
    )
    crossing_values = tuple(
        crossing for state in values for crossing in state.crossings
    )
    face_component_offset = np.zeros(len(values) + 1, dtype=np.uint64)
    face_crossing_offset = np.zeros(len(values) + 1, dtype=np.uint64)
    for index, state in enumerate(values):
        face_component_offset[index + 1] = (
            face_component_offset[index] + len(state.components)
        )
        face_crossing_offset[index + 1] = (
            face_crossing_offset[index] + len(state.crossings)
        )
    component_cell_offset = np.zeros(len(component_values) + 1, dtype=np.uint64)
    component_anchor_offset = np.zeros(len(component_values) + 1, dtype=np.uint64)
    cells: list[Int3] = []
    anchor_ids: list[int] = []
    anchor_parity: list[int] = []
    for index, component in enumerate(component_values):
        cells.extend(component.occupied_cells)
        anchor_ids.extend(component.anchor_patch_ids)
        anchor_parity.extend(component.anchor_orientation_parity)
        component_cell_offset[index + 1] = len(cells)
        component_anchor_offset[index + 1] = len(anchor_ids)
    crossing_observation_offset = np.zeros(
        len(crossing_values) + 1, dtype=np.uint64
    )
    crossing_owner_offset = np.zeros(
        len(crossing_values) + 1, dtype=np.uint64
    )
    observations: list[Observation] = []
    owners: list[Observation] = []
    for index, crossing in enumerate(crossing_values):
        observations.extend(crossing.observations)
        owners.extend(crossing.owners)
        crossing_observation_offset[index + 1] = len(observations)
        crossing_owner_offset[index + 1] = len(owners)
    return {
        "faceAxis": np.asarray([value.axis for value in values], dtype=np.int8),
        "faceSide": np.asarray([value.side for value in values], dtype=np.int8),
        "faceDepthCells": np.asarray(
            [value.depth_cells for value in values], dtype=np.uint16
        ),
        "faceCutCoordinate": np.asarray(
            [value.cut_coordinate for value in values], dtype=np.int32
        ),
        "faceFrozenComponentCount": np.asarray(
            [value.frozen_component_count for value in values], dtype=np.uint64
        ),
        "faceDetachedComponentCount": np.asarray(
            [value.detached_component_count for value in values], dtype=np.uint64
        ),
        "faceComponentOffset": face_component_offset,
        "componentId": np.asarray(
            [value.component_id for value in component_values], dtype=np.uint64
        ),
        "componentTotalPatchCount": np.asarray(
            [value.total_patch_count for value in component_values], dtype=np.uint64
        ),
        "componentCellOffset": component_cell_offset,
        "componentCellXYZ": np.asarray(cells, dtype=np.int32).reshape(len(cells), 3),
        "componentAnchorOffset": component_anchor_offset,
        "anchorPatchId": np.asarray(anchor_ids, dtype=np.uint64),
        "anchorOrientationParity": np.asarray(anchor_parity, dtype=np.uint8),
        "faceCrossingOffset": face_crossing_offset,
        "crossingGroupId": np.asarray(
            [value.group_id for value in crossing_values], dtype=np.uint64
        ),
        "crossingFeatureKind": np.asarray(
            [0 if isinstance(value.feature, GridEdge) else 1 for value in crossing_values],
            dtype=np.uint8,
        ),
        "crossingFeatureEdgeAxis": np.asarray(
            [
                value.feature.axis if isinstance(value.feature, GridEdge) else -1
                for value in crossing_values
            ],
            dtype=np.int8,
        ),
        "crossingFeatureAnchorXYZ": np.asarray(
            [
                value.feature.anchor_xyz
                if isinstance(value.feature, GridEdge)
                else value.feature
                for value in crossing_values
            ],
            dtype=np.int32,
        ).reshape(len(crossing_values), 3),
        "crossingObservationOffset": crossing_observation_offset,
        "observationPatchId": np.asarray(
            [value[0] for value in observations], dtype=np.uint64
        ),
        "observationEdgeAxis": np.asarray(
            [value[1].axis for value in observations], dtype=np.int8
        ),
        "observationEdgeAnchorXYZ": np.asarray(
            [value[1].anchor_xyz for value in observations], dtype=np.int32
        ).reshape(len(observations), 3),
        "crossingOwnerOffset": crossing_owner_offset,
        "ownerPatchId": np.asarray(
            [value[0] for value in owners], dtype=np.uint64
        ),
        "ownerEdgeAxis": np.asarray(
            [value[1].axis for value in owners], dtype=np.int8
        ),
        "ownerEdgeAnchorXYZ": np.asarray(
            [value[1].anchor_xyz for value in owners], dtype=np.int32
        ).reshape(len(owners), 3),
    }


def write_frozen_topology_artifact(
    path: str | Path,
    states: Iterable[FrozenFaceState],
) -> Path:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **frozen_topology_arrays(states))
    temporary.replace(output)
    return output


def frozen_region_arrays(
    states: Iterable[FrozenRegionState],
) -> dict[str, np.ndarray]:
    """Serialize every compatible multi-face frozen-region certificate."""

    values = tuple(sorted(states, key=lambda value: value.face_mask))
    if len({value.face_mask for value in values}) != len(values):
        raise ValueError("frozen region artifact contains duplicate face masks")
    common = frozen_topology_arrays(
        FrozenFaceState(
            0,
            0,
            value.depth_cells,
            0,
            value.frozen_component_count,
            value.detached_component_count,
            value.components,
            value.crossings,
        )
        for value in values
    )
    for name in (
        "faceAxis",
        "faceSide",
        "faceDepthCells",
        "faceCutCoordinate",
        "faceFrozenComponentCount",
        "faceDetachedComponentCount",
    ):
        del common[name]
    common["regionComponentOffset"] = common.pop("faceComponentOffset")
    common["regionCrossingOffset"] = common.pop("faceCrossingOffset")
    return {
        "regionFaceMask": np.asarray(
            [value.face_mask for value in values], dtype=np.uint8
        ),
        "regionDepthCells": np.asarray(
            [value.depth_cells for value in values], dtype=np.uint16
        ),
        "regionFrozenComponentCount": np.asarray(
            [value.frozen_component_count for value in values], dtype=np.uint64
        ),
        "regionDetachedComponentCount": np.asarray(
            [value.detached_component_count for value in values], dtype=np.uint64
        ),
        **common,
    }


def write_frozen_region_artifact(
    path: str | Path,
    states: Iterable[FrozenRegionState],
) -> Path:
    output = Path(path)
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **frozen_region_arrays(states))
    temporary.replace(output)
    return output


def _read_frozen_payload(
    values: Mapping[str, np.ndarray],
    record_index: int,
    *,
    component_offset_name: str,
    crossing_offset_name: str,
) -> tuple[tuple[FrozenComponent, ...], tuple[FrozenCrossing, ...]]:
    component_low = int(values[component_offset_name][record_index])
    component_high = int(values[component_offset_name][record_index + 1])
    components: list[FrozenComponent] = []
    for component_index in range(component_low, component_high):
        cell_low = int(values["componentCellOffset"][component_index])
        cell_high = int(values["componentCellOffset"][component_index + 1])
        anchor_low = int(values["componentAnchorOffset"][component_index])
        anchor_high = int(values["componentAnchorOffset"][component_index + 1])
        components.append(
            FrozenComponent(
                int(values["componentId"][component_index]),
                int(values["componentTotalPatchCount"][component_index]),
                tuple(
                    tuple(int(value) for value in row)
                    for row in values["componentCellXYZ"][cell_low:cell_high]
                ),
                tuple(
                    int(value)
                    for value in values["anchorPatchId"][anchor_low:anchor_high]
                ),
                tuple(
                    int(value)
                    for value in values["anchorOrientationParity"][
                        anchor_low:anchor_high
                    ]
                ),
            )
        )
    crossing_low = int(values[crossing_offset_name][record_index])
    crossing_high = int(values[crossing_offset_name][record_index + 1])
    crossings: list[FrozenCrossing] = []
    for crossing_index in range(crossing_low, crossing_high):
        observation_low = int(values["crossingObservationOffset"][crossing_index])
        observation_high = int(
            values["crossingObservationOffset"][crossing_index + 1]
        )
        owner_low = int(values["crossingOwnerOffset"][crossing_index])
        owner_high = int(values["crossingOwnerOffset"][crossing_index + 1])
        anchor = tuple(
            int(value)
            for value in values["crossingFeatureAnchorXYZ"][crossing_index]
        )
        feature: Feature
        if int(values["crossingFeatureKind"][crossing_index]) == 0:
            feature = GridEdge(
                int(values["crossingFeatureEdgeAxis"][crossing_index]),
                anchor,
            )
        else:
            feature = anchor
        crossings.append(
            FrozenCrossing(
                int(values["crossingGroupId"][crossing_index]),
                feature,
                tuple(
                    (
                        int(patch_id),
                        GridEdge(
                            int(edge_axis),
                            tuple(int(value) for value in edge_anchor),
                        ),
                    )
                    for patch_id, edge_axis, edge_anchor in zip(
                        values["observationPatchId"][
                            observation_low:observation_high
                        ],
                        values["observationEdgeAxis"][
                            observation_low:observation_high
                        ],
                        values["observationEdgeAnchorXYZ"][
                            observation_low:observation_high
                        ],
                    )
                ),
                tuple(
                    (
                        int(patch_id),
                        GridEdge(
                            int(edge_axis),
                            tuple(int(value) for value in edge_anchor),
                        ),
                    )
                    for patch_id, edge_axis, edge_anchor in zip(
                        values["ownerPatchId"][owner_low:owner_high],
                        values["ownerEdgeAxis"][owner_low:owner_high],
                        values["ownerEdgeAnchorXYZ"][owner_low:owner_high],
                    )
                ),
            )
        )
    return tuple(components), tuple(crossings)


def read_frozen_face_state(
    path: str | Path,
    axis: int,
    side: int,
) -> FrozenFaceState:
    with np.load(Path(path)) as values:
        records = np.flatnonzero(
            (values["faceAxis"] == axis) & (values["faceSide"] == side)
        )
        if len(records) != 1:
            raise ValueError("frozen topology artifact lacks one requested face")
        face_index = int(records[0])
        components, crossings = _read_frozen_payload(
            values,
            face_index,
            component_offset_name="faceComponentOffset",
            crossing_offset_name="faceCrossingOffset",
        )
        return FrozenFaceState(
            axis,
            side,
            int(values["faceDepthCells"][face_index]),
            int(values["faceCutCoordinate"][face_index]),
            int(values["faceFrozenComponentCount"][face_index]),
            int(values["faceDetachedComponentCount"][face_index]),
            tuple(components),
            tuple(crossings),
        )


def read_frozen_region_state(
    path: str | Path,
    region_face_mask: int,
) -> FrozenRegionState:
    faces_from_mask(region_face_mask)
    with np.load(Path(path)) as values:
        records = np.flatnonzero(values["regionFaceMask"] == region_face_mask)
        if len(records) != 1:
            raise ValueError("frozen region artifact lacks one requested face mask")
        region_index = int(records[0])
        components, crossings = _read_frozen_payload(
            values,
            region_index,
            component_offset_name="regionComponentOffset",
            crossing_offset_name="regionCrossingOffset",
        )
        return FrozenRegionState(
            region_face_mask,
            int(values["regionDepthCells"][region_index]),
            int(values["regionFrozenComponentCount"][region_index]),
            int(values["regionDetachedComponentCount"][region_index]),
            components,
            crossings,
        )
