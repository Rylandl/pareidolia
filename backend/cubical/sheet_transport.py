from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from itertools import product
from typing import Any, Hashable, Iterable, Mapping

from .geometry import ClippedPatch
from .topology import GridFace, Int3


CandidateKey = Hashable


@dataclass(frozen=True, slots=True)
class StackContinuationEvidence:
    """One continuation edge with its whole-face marginal probability."""

    key: CandidateKey
    first_patch_id: int
    second_patch_id: int
    face: GridFace
    probability: float

    def __post_init__(self) -> None:
        if not math.isfinite(self.probability) or not 0.0 <= self.probability <= 1.0:
            raise ValueError("stack continuation probability must lie in [0, 1]")


@dataclass(frozen=True, slots=True)
class FaceTransportEvidence:
    """The dominant integer layer transport between two adjacent cell stacks."""

    face: GridFace
    lower_cell_xyz: Int3
    upper_cell_xyz: Int3
    shift: int
    weight: float
    dominant_support: float
    alternative_support: float
    support_by_shift: tuple[tuple[int, float], ...]

    @property
    def confidence(self) -> float:
        return 1.0 - math.exp(-self.weight)

    def record(self) -> dict[str, Any]:
        return {
            "faceAxis": self.face.axis,
            "faceAnchorXYZ": list(self.face.anchor_xyz),
            "lowerCellXYZ": list(self.lower_cell_xyz),
            "upperCellXYZ": list(self.upper_cell_xyz),
            "shift": self.shift,
            "weight": round(self.weight, 6),
            "confidence": round(self.confidence, 6),
            "dominantSupport": round(self.dominant_support, 6),
            "alternativeSupport": round(self.alternative_support, 6),
            "supportByShift": [
                {"shift": shift, "support": round(support, 6)}
                for shift, support in self.support_by_shift
            ],
        }


@dataclass(frozen=True, slots=True)
class CandidateTransportResidual:
    key: CandidateKey
    observed_shift: int
    synchronized_shift: int | None
    residual_layers: int
    confidence: float


@dataclass(frozen=True, slots=True)
class CandidateCycleConsistency:
    key: CandidateKey
    plaquette_count: int
    cycle_regret: float
    maximum_plaquette_regret: float

    @property
    def consistency_probability(self) -> float:
        return math.exp(-self.cycle_regret)


@dataclass(frozen=True, slots=True)
class StackCycleConsistency:
    plaquette_count: int
    candidates: tuple[CandidateCycleConsistency, ...]

    def by_key(self) -> dict[CandidateKey, CandidateCycleConsistency]:
        return {value.key: value for value in self.candidates}

    def record(self) -> dict[str, Any]:
        return {
            "method": (
                "exact convolution of neighboring face-shift marginals "
                "around every elementary cell plaquette"
            ),
            "plaquettes": self.plaquette_count,
            "candidates": len(self.candidates),
            "candidatesWithCycleContext": sum(
                value.plaquette_count > 0 for value in self.candidates
            ),
            "candidatesWithCycleRegret": sum(
                value.cycle_regret > 1.0e-12 for value in self.candidates
            ),
            "maximumCycleRegret": round(
                max(
                    (value.cycle_regret for value in self.candidates),
                    default=0.0,
                ),
                6,
            ),
        }


@dataclass(frozen=True, slots=True)
class StackTransportModel:
    """A path-independent integer layer gauge over the occupied cell graph."""

    gauge_by_cell: Mapping[Int3, int]
    component_by_cell: Mapping[Int3, int]
    faces: tuple[FaceTransportEvidence, ...]
    candidate_residuals: tuple[CandidateTransportResidual, ...]
    sweeps: int
    initial_weighted_absolute_residual: float
    final_weighted_absolute_residual: float
    elementary_cycle_count: int
    frustrated_elementary_cycle_count: int
    elementary_cycle_holonomy: Mapping[int, int]

    def residual_by_key(self) -> dict[CandidateKey, CandidateTransportResidual]:
        return {value.key: value for value in self.candidate_residuals}

    def record(self) -> dict[str, Any]:
        face_residuals = []
        for value in self.faces:
            lower = self.gauge_by_cell.get(value.lower_cell_xyz)
            upper = self.gauge_by_cell.get(value.upper_cell_xyz)
            if lower is None or upper is None:
                continue
            face_residuals.append(abs((upper - lower) - value.shift))
        candidate_residuals = [
            value.residual_layers
            for value in self.candidate_residuals
            if value.synchronized_shift is not None
        ]
        return {
            "method": (
                "maximum-support face transport followed by deterministic "
                "weighted-L1 integer graph synchronization"
            ),
            "cells": len(self.gauge_by_cell),
            "components": len(set(self.component_by_cell.values())),
            "faces": len(self.faces),
            "sweeps": self.sweeps,
            "initialWeightedAbsoluteResidual": round(
                self.initial_weighted_absolute_residual, 6
            ),
            "finalWeightedAbsoluteResidual": round(
                self.final_weighted_absolute_residual, 6
            ),
            "frustratedFaces": sum(value > 0 for value in face_residuals),
            "maximumFaceResidualLayers": max(face_residuals, default=0),
            "candidateResiduals": len(candidate_residuals),
            "candidatesWithNonzeroResidual": sum(
                value > 0 for value in candidate_residuals
            ),
            "maximumCandidateResidualLayers": max(
                candidate_residuals, default=0
            ),
            "elementaryCycles": self.elementary_cycle_count,
            "frustratedElementaryCycles": (
                self.frustrated_elementary_cycle_count
            ),
            "elementaryCycleHolonomy": {
                str(key): int(value)
                for key, value in sorted(self.elementary_cycle_holonomy.items())
            },
        }


class _DisjointSet:
    def __init__(self, values: Iterable[Int3]) -> None:
        self.parent = {value: value for value in values}
        self.size = {value: 1 for value in self.parent}

    def find(self, value: Int3) -> Int3:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            following = self.parent[value]
            self.parent[value] = root
            value = following
        return root

    def union(self, first: Int3, second: Int3) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size.pop(second_root)
        return True


def patch_stack_ranks(
    patches: tuple[ClippedPatch, ...],
) -> tuple[dict[int, Int3], dict[int, int]]:
    by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in patches:
        by_cell[patch.cell_xyz].append(patch)
    cell_by_patch: dict[int, Int3] = {}
    rank_by_patch: dict[int, int] = {}
    for cell, values in by_cell.items():
        values.sort(
            key=lambda value: (
                value.estimate.height_from_cell_center,
                value.patch_id,
            )
        )
        for rank, patch in enumerate(values):
            cell_by_patch[patch.patch_id] = cell
            rank_by_patch[patch.patch_id] = rank
    return cell_by_patch, rank_by_patch


def _oriented_shift(
    value: StackContinuationEvidence,
    cell_by_patch: Mapping[int, Int3],
    rank_by_patch: Mapping[int, int],
) -> int:
    lower, upper = value.face.adjacent_cells()
    first_cell = cell_by_patch[value.first_patch_id]
    second_cell = cell_by_patch[value.second_patch_id]
    if first_cell == lower and second_cell == upper:
        lower_patch = value.first_patch_id
        upper_patch = value.second_patch_id
    elif second_cell == lower and first_cell == upper:
        lower_patch = value.second_patch_id
        upper_patch = value.first_patch_id
    else:
        raise ValueError("stack continuation does not cross its declared face")
    # A common global layer label is rank + cell gauge.  Matching the two
    # ranks therefore observes gauge(upper) - gauge(lower).
    return rank_by_patch[lower_patch] - rank_by_patch[upper_patch]


def _face_evidence(
    values: tuple[StackContinuationEvidence, ...],
    cell_by_patch: Mapping[int, Int3],
    rank_by_patch: Mapping[int, int],
) -> tuple[FaceTransportEvidence, ...]:
    by_face: dict[GridFace, list[StackContinuationEvidence]] = defaultdict(list)
    for value in values:
        by_face[value.face].append(value)
    result: list[FaceTransportEvidence] = []
    for face, candidates in by_face.items():
        support: dict[int, float] = defaultdict(float)
        for value in candidates:
            support[_oriented_shift(value, cell_by_patch, rank_by_patch)] += (
                value.probability
            )
        ordered = sorted(
            support.items(),
            key=lambda value: (-value[1], abs(value[0]), value[0]),
        )
        dominant_shift, dominant_support = ordered[0]
        alternative_support = ordered[1][1] if len(ordered) > 1 else 0.0
        lower, upper = face.adjacent_cells()
        result.append(
            FaceTransportEvidence(
                face,
                lower,
                upper,
                dominant_shift,
                max(dominant_support - alternative_support, 0.0),
                dominant_support,
                alternative_support,
                tuple(sorted(support.items())),
            )
        )
    result.sort(
        key=lambda value: (
            value.face.axis,
            value.face.anchor_xyz,
        )
    )
    return tuple(result)


def _face_from_lower(axis: int, cell: Int3) -> GridFace:
    return GridFace(
        axis,
        tuple(
            value + int(index == axis)
            for index, value in enumerate(cell)
        ),
    )


def _elementary_plaquettes(
    cells: Iterable[Int3], faces: set[GridFace]
) -> tuple[tuple[GridFace, GridFace, GridFace, GridFace], ...]:
    values = []
    for cell in sorted(set(cells)):
        for first_axis in range(3):
            for second_axis in range(first_axis + 1, 3):
                after_first = tuple(
                    value + int(index == first_axis)
                    for index, value in enumerate(cell)
                )
                after_second = tuple(
                    value + int(index == second_axis)
                    for index, value in enumerate(cell)
                )
                plaquette = (
                    _face_from_lower(first_axis, cell),
                    _face_from_lower(second_axis, after_first),
                    _face_from_lower(first_axis, after_second),
                    _face_from_lower(second_axis, cell),
                )
                if all(value in faces for value in plaquette):
                    values.append(plaquette)
    return tuple(values)


def stack_cycle_consistency(
    patches: Iterable[ClippedPatch],
    evidence: Iterable[StackContinuationEvidence],
) -> StackCycleConsistency:
    """Score continuation shifts against every neighboring three-face path.

    A cell stack owns an arbitrary integer gauge, but differences of that gauge
    must have zero curl.  For each face of an elementary plaquette, the other
    three faces induce a complete probability distribution for its required
    layer shift.  Candidate regret is the exact log probability ratio between
    the best path-consistent shift and the candidate's shift.  No direction
    sign is attached to a physical normal or fiber; signs here are only the
    oriented algebra of the cell adjacency graph.
    """

    patch_values = tuple(patches)
    continuation_values = tuple(evidence)
    if not continuation_values:
        return StackCycleConsistency(0, tuple())
    cell_by_patch, rank_by_patch = patch_stack_ranks(patch_values)
    faces = _face_evidence(
        continuation_values, cell_by_patch, rank_by_patch
    )
    distribution_by_face: dict[GridFace, dict[int, float]] = {}
    for value in faces:
        total = sum(support for _, support in value.support_by_shift)
        distribution_by_face[value.face] = {
            shift: support / max(total, 1.0e-300)
            for shift, support in value.support_by_shift
        }
    plaquettes = _elementary_plaquettes(
        cell_by_patch.values(), set(distribution_by_face)
    )
    candidates_by_face: dict[
        GridFace, list[tuple[StackContinuationEvidence, int]]
    ] = defaultdict(list)
    for value in continuation_values:
        candidates_by_face[value.face].append(
            (
                value,
                _oriented_shift(value, cell_by_patch, rank_by_patch),
            )
        )
    regrets: dict[CandidateKey, list[float]] = defaultdict(list)
    signs = (1, 1, -1, -1)
    probability_floor = 1.0e-12
    for plaquette in plaquettes:
        distributions = [distribution_by_face[value] for value in plaquette]
        for target_index, target_face in enumerate(plaquette):
            other_indices = tuple(
                index for index in range(4) if index != target_index
            )
            path_distribution: dict[int, float] = defaultdict(float)
            entries = [
                tuple(distributions[index].items())
                for index in other_indices
            ]
            for combination in product(*entries):
                weighted_sum = sum(
                    signs[index] * shift
                    for index, (shift, _) in zip(other_indices, combination)
                )
                predicted = -weighted_sum // signs[target_index]
                probability = math.prod(
                    value for _, value in combination
                )
                path_distribution[predicted] += probability
            maximum_probability = max(path_distribution.values())
            for candidate, observed_shift in candidates_by_face[target_face]:
                probability = path_distribution.get(observed_shift, 0.0)
                regret = math.log(
                    max(maximum_probability, probability_floor)
                    / max(probability, probability_floor)
                )
                regrets[candidate.key].append(max(regret, 0.0))
    result = []
    for value in continuation_values:
        candidate_regrets = regrets.get(value.key, ())
        result.append(
            CandidateCycleConsistency(
                value.key,
                len(candidate_regrets),
                float(sum(candidate_regrets)),
                max(candidate_regrets, default=0.0),
            )
        )
    result.sort(key=lambda value: repr(value.key))
    return StackCycleConsistency(len(plaquettes), tuple(result))


def _maximum_spanning_forest(
    cells: tuple[Int3, ...],
    faces: tuple[FaceTransportEvidence, ...],
) -> tuple[dict[Int3, int], dict[Int3, int]]:
    disjoint = _DisjointSet(cells)
    adjacency: dict[Int3, list[tuple[Int3, int]]] = defaultdict(list)
    for value in sorted(
        faces,
        key=lambda item: (
            -item.weight,
            item.face.axis,
            item.face.anchor_xyz,
        ),
    ):
        if value.weight <= 0.0 or not disjoint.union(
            value.lower_cell_xyz, value.upper_cell_xyz
        ):
            continue
        adjacency[value.lower_cell_xyz].append(
            (value.upper_cell_xyz, value.shift)
        )
        adjacency[value.upper_cell_xyz].append(
            (value.lower_cell_xyz, -value.shift)
        )
    gauge: dict[Int3, int] = {}
    component_by_cell: dict[Int3, int] = {}
    for component_id, seed in enumerate(cells):
        if seed in gauge:
            continue
        gauge[seed] = 0
        component_by_cell[seed] = component_id
        queue: deque[Int3] = deque((seed,))
        while queue:
            first = queue.popleft()
            for second, delta in adjacency.get(first, ()):
                if second in gauge:
                    continue
                gauge[second] = gauge[first] + delta
                component_by_cell[second] = component_id
                queue.append(second)
    return gauge, component_by_cell


def _weighted_median(
    proposals: Iterable[tuple[int, float]], current: int
) -> int:
    combined: dict[int, float] = defaultdict(float)
    for value, weight in proposals:
        if weight > 0.0:
            combined[value] += weight
    if not combined:
        return current
    ordered = sorted(combined.items())
    total = sum(weight for _, weight in ordered)
    cumulative = 0.0
    lower = ordered[0][0]
    upper = ordered[-1][0]
    for value, weight in ordered:
        cumulative += weight
        if cumulative + 1.0e-12 >= total / 2.0:
            lower = value
            break
    cumulative = 0.0
    for value, weight in reversed(ordered):
        cumulative += weight
        if cumulative + 1.0e-12 >= total / 2.0:
            upper = value
            break
    return min(max(current, lower), upper)


def _weighted_residual(
    gauge: Mapping[Int3, int],
    faces: Iterable[FaceTransportEvidence],
) -> float:
    return sum(
        value.weight
        * abs(
            (gauge[value.upper_cell_xyz] - gauge[value.lower_cell_xyz])
            - value.shift
        )
        for value in faces
    )


def _synchronize_gauge(
    cells: tuple[Int3, ...],
    faces: tuple[FaceTransportEvidence, ...],
    maximum_sweeps: int,
) -> tuple[dict[Int3, int], dict[Int3, int], int, float, float]:
    gauge, component_by_cell = _maximum_spanning_forest(cells, faces)
    initial = _weighted_residual(gauge, faces)
    adjacency: dict[
        Int3, list[tuple[Int3, int, bool, float]]
    ] = defaultdict(list)
    for value in faces:
        if value.weight <= 0.0:
            continue
        adjacency[value.lower_cell_xyz].append(
            (value.upper_cell_xyz, value.shift, True, value.weight)
        )
        adjacency[value.upper_cell_xyz].append(
            (value.lower_cell_xyz, value.shift, False, value.weight)
        )
    anchor_by_component: dict[int, Int3] = {}
    for cell, component_id in component_by_cell.items():
        anchor_by_component[component_id] = min(
            anchor_by_component.get(component_id, cell), cell
        )
    anchors = set(anchor_by_component.values())
    sweeps = 0
    for sweep in range(maximum_sweeps):
        changed = 0
        order = cells if sweep % 2 == 0 else tuple(reversed(cells))
        for cell in order:
            if cell in anchors:
                continue
            proposals = []
            for neighbor, shift, cell_is_lower, weight in adjacency.get(cell, ()):
                proposal = (
                    gauge[neighbor] - shift
                    if cell_is_lower
                    else gauge[neighbor] + shift
                )
                proposals.append((proposal, weight))
            candidate = _weighted_median(proposals, gauge[cell])
            if candidate == gauge[cell]:
                continue
            before = sum(
                weight * abs(gauge[cell] - proposal)
                for proposal, weight in proposals
            )
            after = sum(
                weight * abs(candidate - proposal)
                for proposal, weight in proposals
            )
            if after + 1.0e-12 < before:
                gauge[cell] = candidate
                changed += 1
        sweeps = sweep + 1
        if changed == 0:
            break
    return (
        gauge,
        component_by_cell,
        sweeps,
        initial,
        _weighted_residual(gauge, faces),
    )


def _elementary_cycle_holonomy(
    faces: tuple[FaceTransportEvidence, ...],
) -> tuple[int, int, dict[int, int]]:
    shift = {value.face: value.shift for value in faces if value.weight > 0.0}
    counts: dict[int, int] = defaultdict(int)
    for face, first_shift in shift.items():
        first_axis = face.axis
        lower, _ = face.adjacent_cells()
        for second_axis in range(first_axis + 1, 3):
            upper_first = tuple(
                value + int(axis == first_axis)
                for axis, value in enumerate(lower)
            )
            upper_second = tuple(
                value + int(axis == second_axis)
                for axis, value in enumerate(lower)
            )
            def face_from_lower(axis: int, cell: Int3) -> GridFace:
                return GridFace(
                    axis,
                    tuple(
                        value + int(index == axis)
                        for index, value in enumerate(cell)
                    ),
                )

            first_after_second = face_from_lower(
                first_axis, upper_second
            )
            second_before_first = face_from_lower(second_axis, lower)
            second_after_first = face_from_lower(
                second_axis, upper_first
            )
            values = (
                face,
                second_after_first,
                first_after_second,
                second_before_first,
            )
            if not all(value in shift for value in values):
                continue
            holonomy = (
                first_shift
                + shift[second_after_first]
                - shift[first_after_second]
                - shift[second_before_first]
            )
            counts[holonomy] += 1
    total = sum(counts.values())
    return total, total - counts.get(0, 0), dict(counts)


def synchronize_stack_transport(
    patches: Iterable[ClippedPatch],
    evidence: Iterable[StackContinuationEvidence],
    *,
    maximum_sweeps: int = 128,
) -> StackTransportModel:
    """Infer one globally path-independent integer layer gauge.

    Every cell owns an ordered local stack.  A face alignment observes the
    integer difference between the gauges of its two cells.  Weighted-L1 graph
    synchronization resolves mutually inconsistent loop observations without
    assuming any component identity, and every continuation can then be scored
    by how many physical layers it departs from that common transport field.
    """

    if maximum_sweeps < 1:
        raise ValueError("stack transport synchronization requires a sweep")
    patch_values = tuple(patches)
    continuation_values = tuple(evidence)
    if not patch_values:
        return StackTransportModel(
            {}, {}, tuple(), tuple(), 0, 0.0, 0.0, 0, 0, {}
        )
    cell_by_patch, rank_by_patch = patch_stack_ranks(patch_values)
    patch_ids = set(cell_by_patch)
    unknown = {
        patch_id
        for value in continuation_values
        for patch_id in (value.first_patch_id, value.second_patch_id)
        if patch_id not in patch_ids
    }
    if unknown:
        raise ValueError(
            f"stack transport references absent patches: {sorted(unknown)[:2]}"
        )
    faces = _face_evidence(
        continuation_values, cell_by_patch, rank_by_patch
    )
    cells = tuple(sorted(set(cell_by_patch.values())))
    gauge, components, sweeps, initial, final = _synchronize_gauge(
        cells, faces, maximum_sweeps
    )
    face_by_key = {value.face: value for value in faces}
    candidate_residuals = []
    for value in continuation_values:
        observed = _oriented_shift(value, cell_by_patch, rank_by_patch)
        lower, upper = value.face.adjacent_cells()
        synchronized: int | None = None
        if components.get(lower) == components.get(upper):
            synchronized = gauge[upper] - gauge[lower]
        face_value = face_by_key[value.face]
        candidate_residuals.append(
            CandidateTransportResidual(
                value.key,
                observed,
                synchronized,
                (
                    abs(observed - synchronized)
                    if synchronized is not None
                    else 0
                ),
                face_value.confidence,
            )
        )
    candidate_residuals.sort(key=lambda value: repr(value.key))
    cycle_count, frustrated_cycles, holonomy = _elementary_cycle_holonomy(faces)
    return StackTransportModel(
        dict(gauge),
        dict(components),
        faces,
        tuple(candidate_residuals),
        sweeps,
        initial,
        final,
        cycle_count,
        frustrated_cycles,
        holonomy,
    )
