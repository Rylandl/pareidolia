from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Iterable

import numpy as np

from .geometry import (
    ClippedPatch,
    FaceTrace,
    PlaneEstimate,
    axial_angle_radians,
    canonical_axis,
)
from .topology import GridEdge, GridFace, GridSpec, Int3


@dataclass(frozen=True, slots=True)
class TraceMatchSettings:
    """Dimensionless statistical gates for shared-face trace matching."""

    crossing_standard_deviation_floor: float = 0.01
    normal_standard_deviation_floor_radians: float = math.radians(1.0)
    fiber_standard_deviation_floor_radians: float = math.radians(2.0)
    maximum_endpoint_z: float = 4.5
    maximum_normal_z: float = 4.5
    maximum_fiber_z: float = 4.5
    maximum_absolute_normal_angle_radians: float = 0.5 * math.pi
    maximum_absolute_fiber_residual_radians: float = 0.5 * math.pi
    maximum_reduced_chi_square: float = 8.0
    unmatched_negative_log_likelihood: float = 7.0
    orthogonal_fiber_equivalence: bool = False

    def __post_init__(self) -> None:
        positive = (
            self.crossing_standard_deviation_floor,
            self.normal_standard_deviation_floor_radians,
            self.fiber_standard_deviation_floor_radians,
            self.maximum_endpoint_z,
            self.maximum_normal_z,
            self.maximum_fiber_z,
            self.maximum_absolute_normal_angle_radians,
            self.maximum_absolute_fiber_residual_radians,
            self.maximum_reduced_chi_square,
            self.unmatched_negative_log_likelihood,
        )
        if any(not np.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("trace matching settings must be finite and positive")
        if not isinstance(self.orthogonal_fiber_equivalence, bool):
            raise ValueError("orthogonal fiber equivalence must be boolean")
        if (
            self.maximum_absolute_normal_angle_radians > 0.5 * math.pi
            or self.maximum_absolute_fiber_residual_radians > 0.5 * math.pi
        ):
            raise ValueError("absolute angular gates cannot exceed 90 degrees")


@dataclass(frozen=True, slots=True)
class EndpointAgreement:
    first_edge: GridEdge
    second_edge: GridEdge
    z: float
    mode: str
    shared_vertex_xyz: Int3 | None


@dataclass(frozen=True, slots=True)
class TraceMatch:
    first_patch_id: int
    second_patch_id: int
    face: GridFace
    accepted: bool
    failure_reasons: tuple[str, ...]
    endpoint_agreements: tuple[EndpointAgreement, ...]
    normal_angle_radians: float
    normal_z: float
    fiber_angle_radians: float | None
    fiber_z: float | None
    fiber_quarter_turn: bool | None
    reduced_chi_square: float
    negative_log_likelihood: float
    score: float


@dataclass(frozen=True, slots=True)
class FaceAlignment:
    face: GridFace
    matches: tuple[TraceMatch, ...]
    unmatched_first_patch_ids: tuple[int, ...]
    unmatched_second_patch_ids: tuple[int, ...]
    negative_log_likelihood: float
    order_axis_xyz: tuple[float, float, float] | None


def trace_tangent_side_offsets(
    first: ClippedPatch,
    second: ClippedPatch,
    match: TraceMatch,
) -> tuple[float, float] | None:
    """Return signed sheet-interior offsets from one matched trace.

    The sign is expressed in the common tangent plane, perpendicular to the
    matched trace.  It is invariant to the arbitrary signs of the axial normal
    and trace direction because either flip changes both returned signs.  A
    regular local surface crossing has opposite signs; equal signs identify a
    fold-back adjacency whose two interiors leave the shared edge on the same
    side in tangent coordinates.
    """

    if (
        first.patch_id != match.first_patch_id
        or second.patch_id != match.second_patch_id
    ):
        raise ValueError("tangent-side patches do not match the join ordering")
    first_trace = first.trace_on(match.face)
    second_trace = second.trace_on(match.face)
    if first_trace is None or second_trace is None:
        raise ValueError("tangent-side join is absent from one patch")
    first_by_edge = {
        value.edge: value
        for value in (first_trace.first, first_trace.second)
    }
    second_by_edge = {
        value.edge: value
        for value in (second_trace.first, second_trace.second)
    }
    paired_points = []
    for agreement in match.endpoint_agreements:
        if (
            agreement.first_edge not in first_by_edge
            or agreement.second_edge not in second_by_edge
        ):
            raise ValueError("tangent-side endpoint agreement is absent")
        paired_points.append(
            0.5
            * (
                np.asarray(
                    first_by_edge[agreement.first_edge].point_xyz,
                    dtype=np.float64,
                )
                + np.asarray(
                    second_by_edge[agreement.second_edge].point_xyz,
                    dtype=np.float64,
                )
            )
        )
    if len(paired_points) != 2:
        return None
    first_normal = np.asarray(first.estimate.normal_xyz, dtype=np.float64)
    second_normal = np.asarray(second.estimate.normal_xyz, dtype=np.float64)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
    normal = first_normal + second_normal
    normal_length = float(np.linalg.norm(normal))
    if normal_length <= 1.0e-10:
        return None
    normal /= normal_length
    trace_direction = paired_points[1] - paired_points[0]
    trace_direction -= normal * float(np.dot(trace_direction, normal))
    trace_length = float(np.linalg.norm(trace_direction))
    if trace_length <= 1.0e-10:
        return None
    trace_direction /= trace_length
    conormal = np.cross(normal, trace_direction)
    conormal /= max(float(np.linalg.norm(conormal)), 1.0e-12)
    midpoint = 0.5 * (paired_points[0] + paired_points[1])
    first_centroid = np.mean(
        np.asarray([value.point_xyz for value in first.vertices]), axis=0
    )
    second_centroid = np.mean(
        np.asarray([value.point_xyz for value in second.vertices]), axis=0
    )
    return (
        float(np.dot(first_centroid - midpoint, conormal)),
        float(np.dot(second_centroid - midpoint, conormal)),
    )


def _transported_fiber_angle(
    first: PlaneEstimate, second: PlaneEstimate
) -> float | None:
    if first.fiber_xyz is None or second.fiber_xyz is None:
        return None
    first_normal = np.asarray(first.normal_xyz, dtype=np.float64)
    second_normal = np.asarray(second.normal_xyz, dtype=np.float64)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
    first_fiber = np.asarray(first.fiber_xyz, dtype=np.float64)
    second_fiber = np.asarray(second.fiber_xyz, dtype=np.float64)
    rotation_axis = np.cross(first_normal, second_normal)
    sine = float(np.linalg.norm(rotation_axis))
    cosine = float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))
    if sine <= 1.0e-10:
        transported = first_fiber
    else:
        rotation_axis /= sine
        transported = (
            first_fiber * cosine
            + np.cross(rotation_axis, first_fiber) * sine
            + rotation_axis
            * float(np.dot(rotation_axis, first_fiber))
            * (1.0 - cosine)
        )
    transported -= second_normal * float(np.dot(transported, second_normal))
    transported /= max(float(np.linalg.norm(transported)), 1.0e-12)
    second_fiber -= second_normal * float(np.dot(second_fiber, second_normal))
    second_fiber /= max(float(np.linalg.norm(second_fiber)), 1.0e-12)
    return axial_angle_radians(transported, second_fiber)


def _trace_by_edge(trace: FaceTrace) -> dict[GridEdge, object]:
    if trace.first.edge == trace.second.edge:
        raise ValueError("face trace endpoints cannot occupy the same grid edge")
    return {trace.first.edge: trace.first, trace.second.edge: trace.second}


def _shared_vertex(first: GridEdge, second: GridEdge) -> Int3 | None:
    if first == second:
        return None
    shared = set(first.endpoint_vertices()) & set(second.endpoint_vertices())
    return next(iter(shared)) if len(shared) == 1 else None


def _endpoint_agreement(
    first: object,
    second: object,
    settings: TraceMatchSettings,
    grid: GridSpec | None,
) -> EndpointAgreement | None:
    if first.edge == second.edge:
        variance = (
            float(first.variance)
            + float(second.variance)
            + settings.crossing_standard_deviation_floor**2
        )
        z_value = abs(float(first.t) - float(second.t)) / math.sqrt(variance)
        return EndpointAgreement(
            first.edge, second.edge, z_value, "same-edge", None
        )
    vertex = _shared_vertex(first.edge, second.edge)
    if vertex is None or grid is None:
        return None

    def corner_z(crossing: object) -> float:
        start, stop = crossing.edge.endpoint_vertices()
        if vertex == start:
            coordinate = float(crossing.t)
        elif vertex == stop:
            coordinate = 1.0 - float(crossing.t)
        else:
            raise RuntimeError("candidate corner is not incident to its edge")
        edge_length = float(grid.cell_size_xyz[crossing.edge.axis])
        distance = coordinate * edge_length
        variance = edge_length**2 * (
            float(crossing.variance)
            + settings.crossing_standard_deviation_floor**2
        )
        return distance / math.sqrt(variance)

    z_value = math.hypot(corner_z(first), corner_z(second))
    return EndpointAgreement(
        first.edge, second.edge, z_value, "shared-corner", vertex
    )


def match_face_traces(
    first_trace: FaceTrace,
    first_estimate: PlaneEstimate,
    second_trace: FaceTrace,
    second_estimate: PlaneEstimate,
    settings: TraceMatchSettings | None = None,
    *,
    grid: GridSpec | None = None,
) -> TraceMatch:
    """Compare two observations of a trace on one canonical grid face."""

    resolved = settings or TraceMatchSettings()
    failures: list[str] = []
    if first_trace.face != second_trace.face:
        raise ValueError("trace matching requires one shared canonical face")
    _trace_by_edge(first_trace)
    _trace_by_edge(second_trace)
    first_endpoints = (first_trace.first, first_trace.second)
    second_endpoints = (second_trace.first, second_trace.second)
    endpoint_values: tuple[EndpointAgreement, ...] = ()
    chi_square_terms: list[float] = []
    endpoint_pairings: list[tuple[float, tuple[EndpointAgreement, ...]]] = []
    for second_order in (second_endpoints, tuple(reversed(second_endpoints))):
        agreements = tuple(
            _endpoint_agreement(first, second, resolved, grid)
            for first, second in zip(first_endpoints, second_order)
        )
        if all(value is not None for value in agreements):
            valid = tuple(value for value in agreements if value is not None)
            endpoint_pairings.append(
                (sum(value.z**2 for value in valid), valid)
            )
    if not endpoint_pairings:
        failures.append("edge-topology")
    else:
        _, endpoint_values = min(
            endpoint_pairings,
            key=lambda value: (
                value[0],
                tuple(
                    (
                        endpoint.mode,
                        endpoint.first_edge,
                        endpoint.second_edge,
                    )
                    for endpoint in value[1]
                ),
            ),
        )
        for endpoint in endpoint_values:
            chi_square_terms.append(endpoint.z**2)
            if endpoint.z > resolved.maximum_endpoint_z:
                failures.append("endpoint")

    normal_angle = axial_angle_radians(
        first_estimate.normal_xyz, second_estimate.normal_xyz
    )
    normal_std = math.sqrt(
        first_estimate.maximum_angular_std_radians**2
        + second_estimate.maximum_angular_std_radians**2
        + resolved.normal_standard_deviation_floor_radians**2
    )
    normal_z = normal_angle / normal_std
    chi_square_terms.append(normal_z**2)
    if normal_z > resolved.maximum_normal_z:
        failures.append("normal")
    if normal_angle > resolved.maximum_absolute_normal_angle_radians:
        failures.append("normal-angle")

    fiber_angle = _transported_fiber_angle(first_estimate, second_estimate)
    fiber_z: float | None = None
    fiber_quarter_turn: bool | None = None
    if fiber_angle is not None:
        fiber_quarter_turn = False
        if resolved.orthogonal_fiber_equivalence:
            orthogonal_residual = abs(0.5 * math.pi - fiber_angle)
            if orthogonal_residual < fiber_angle:
                fiber_angle = orthogonal_residual
                fiber_quarter_turn = True
        first_std = first_estimate.fiber_angular_std_radians or 0.0
        second_std = second_estimate.fiber_angular_std_radians or 0.0
        fiber_std = math.sqrt(
            first_std**2
            + second_std**2
            + resolved.fiber_standard_deviation_floor_radians**2
        )
        fiber_z = fiber_angle / fiber_std
        chi_square_terms.append(fiber_z**2)
        if fiber_z > resolved.maximum_fiber_z:
            failures.append("fiber")
        if fiber_angle > resolved.maximum_absolute_fiber_residual_radians:
            failures.append("fiber-angle")

    reduced_chi_square = float(np.mean(chi_square_terms)) if chi_square_terms else math.inf
    if reduced_chi_square > resolved.maximum_reduced_chi_square:
        failures.append("joint")
    confidence_cost = -math.log(
        max(first_estimate.confidence * second_estimate.confidence, 1.0e-8)
    )
    negative_log_likelihood = 0.5 * float(np.sum(chi_square_terms)) + confidence_cost
    score = math.exp(
        -negative_log_likelihood / max(float(len(chi_square_terms)), 1.0)
    )
    return TraceMatch(
        first_patch_id=first_trace.patch_id,
        second_patch_id=second_trace.patch_id,
        face=first_trace.face,
        accepted=not failures,
        failure_reasons=tuple(sorted(set(failures))),
        endpoint_agreements=endpoint_values,
        normal_angle_radians=normal_angle,
        normal_z=normal_z,
        fiber_angle_radians=fiber_angle,
        fiber_z=fiber_z,
        fiber_quarter_turn=fiber_quarter_turn,
        reduced_chi_square=reduced_chi_square,
        negative_log_likelihood=negative_log_likelihood,
        score=score,
    )


def _face_order_axis(
    traces_and_estimates: Iterable[tuple[FaceTrace, PlaneEstimate]],
    face: GridFace,
) -> np.ndarray:
    face_normal = np.zeros(3, dtype=np.float64)
    face_normal[face.axis] = 1.0
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for _, estimate in traces_and_estimates:
        normal = np.asarray(estimate.normal_xyz, dtype=np.float64)
        projected = normal - face_normal * float(np.dot(normal, face_normal))
        length = float(np.linalg.norm(projected))
        if length > 1.0e-8:
            projected /= length
            accumulator += np.outer(projected, projected)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    if float(eigenvalues[-1]) <= 1.0e-10:
        raise ValueError("shared-face traces do not define a stable order axis")
    return canonical_axis(eigenvectors[:, -1])


def face_patch_ranks(
    first_patches: Iterable[ClippedPatch],
    second_patches: Iterable[ClippedPatch],
    face: GridFace,
) -> tuple[dict[int, int], dict[int, int], tuple[float, float, float] | None]:
    """Return the canonical trace order on both sides of one grid face.

    A sheet-level solver may consider more than the single optimal alignment
    returned by :func:`align_face_patches`.  These ranks are the invariant it
    must preserve when combining those alternative pair correspondences.
    """

    first_values = [
        (trace, patch.estimate)
        for patch in first_patches
        if (trace := patch.trace_on(face)) is not None
    ]
    second_values = [
        (trace, patch.estimate)
        for patch in second_patches
        if (trace := patch.trace_on(face)) is not None
    ]
    if not first_values and not second_values:
        return {}, {}, None
    order_axis = _face_order_axis(first_values + second_values, face)

    def ranks(
        values: list[tuple[FaceTrace, PlaneEstimate]],
    ) -> dict[int, int]:
        ordered = sorted(
            values,
            key=lambda value: (
                float(np.dot(value[0].midpoint_xyz, order_axis)),
                value[0].patch_id,
            ),
        )
        return {
            trace.patch_id: index
            for index, (trace, _) in enumerate(ordered)
        }

    return (
        ranks(first_values),
        ranks(second_values),
        tuple(float(value) for value in order_axis),
    )


def align_face_patches(
    first_patches: Iterable[ClippedPatch],
    second_patches: Iterable[ClippedPatch],
    face: GridFace,
    settings: TraceMatchSettings | None = None,
    *,
    grid: GridSpec | None = None,
    _match_cache: dict[tuple[object, ...], TraceMatch] | None = None,
) -> FaceAlignment:
    """Order-preserving partial alignment of all patch traces on one face."""

    resolved = settings or TraceMatchSettings()
    first_values = [
        (trace, patch.estimate)
        for patch in first_patches
        if (trace := patch.trace_on(face)) is not None
    ]
    second_values = [
        (trace, patch.estimate)
        for patch in second_patches
        if (trace := patch.trace_on(face)) is not None
    ]
    if not first_values and not second_values:
        return FaceAlignment(face, (), (), (), 0.0, None)
    order_axis = _face_order_axis(first_values + second_values, face)
    first_values.sort(
        key=lambda value: (
            float(np.dot(value[0].midpoint_xyz, order_axis)),
            value[0].patch_id,
        )
    )
    second_values.sort(
        key=lambda value: (
            float(np.dot(value[0].midpoint_xyz, order_axis)),
            value[0].patch_id,
        )
    )
    first_count = len(first_values)
    second_count = len(second_values)
    pair_matches: dict[tuple[int, int], TraceMatch] = {}
    for first_index, (first_trace, first_estimate) in enumerate(first_values):
        for second_index, (second_trace, second_estimate) in enumerate(second_values):
            cache_key = (
                face,
                first_trace.first.edge,
                first_trace.first.t,
                first_trace.first.variance,
                first_trace.second.edge,
                first_trace.second.t,
                first_trace.second.variance,
                first_estimate,
                second_trace.first.edge,
                second_trace.first.t,
                second_trace.first.variance,
                second_trace.second.edge,
                second_trace.second.t,
                second_trace.second.variance,
                second_estimate,
                None if grid is None else grid.cell_size_xyz,
            )
            match = None if _match_cache is None else _match_cache.get(cache_key)
            if match is None:
                match = match_face_traces(
                    first_trace,
                    first_estimate,
                    second_trace,
                    second_estimate,
                    resolved,
                    grid=grid,
                )
                if _match_cache is not None:
                    _match_cache[cache_key] = match
            elif (
                match.first_patch_id != first_trace.patch_id
                or match.second_patch_id != second_trace.patch_id
            ):
                match = replace(
                    match,
                    first_patch_id=first_trace.patch_id,
                    second_patch_id=second_trace.patch_id,
                )
            pair_matches[(first_index, second_index)] = match

    costs = np.full((first_count + 1, second_count + 1), np.inf, dtype=np.float64)
    operations = np.full((first_count + 1, second_count + 1), -1, dtype=np.int8)
    costs[0, 0] = 0.0
    for first_index in range(1, first_count + 1):
        costs[first_index, 0] = (
            costs[first_index - 1, 0] + resolved.unmatched_negative_log_likelihood
        )
        operations[first_index, 0] = 1
    for second_index in range(1, second_count + 1):
        costs[0, second_index] = (
            costs[0, second_index - 1] + resolved.unmatched_negative_log_likelihood
        )
        operations[0, second_index] = 2
    for first_index in range(1, first_count + 1):
        for second_index in range(1, second_count + 1):
            match = pair_matches[(first_index - 1, second_index - 1)]
            candidates = [
                (
                    costs[first_index - 1, second_index - 1]
                    + (match.negative_log_likelihood if match.accepted else math.inf),
                    0,
                ),
                (
                    costs[first_index - 1, second_index]
                    + resolved.unmatched_negative_log_likelihood,
                    1,
                ),
                (
                    costs[first_index, second_index - 1]
                    + resolved.unmatched_negative_log_likelihood,
                    2,
                ),
            ]
            best_cost, operation = min(candidates, key=lambda value: (value[0], value[1]))
            costs[first_index, second_index] = best_cost
            operations[first_index, second_index] = operation

    selected_matches: list[TraceMatch] = []
    unmatched_first: list[int] = []
    unmatched_second: list[int] = []
    first_index, second_index = first_count, second_count
    while first_index or second_index:
        operation = int(operations[first_index, second_index])
        if operation == 0:
            selected_matches.append(pair_matches[(first_index - 1, second_index - 1)])
            first_index -= 1
            second_index -= 1
        elif operation == 1:
            unmatched_first.append(first_values[first_index - 1][0].patch_id)
            first_index -= 1
        elif operation == 2:
            unmatched_second.append(second_values[second_index - 1][0].patch_id)
            second_index -= 1
        else:
            raise RuntimeError("face alignment backtrace is incomplete")
    selected_matches.reverse()
    unmatched_first.reverse()
    unmatched_second.reverse()
    return FaceAlignment(
        face=face,
        matches=tuple(selected_matches),
        unmatched_first_patch_ids=tuple(unmatched_first),
        unmatched_second_patch_ids=tuple(unmatched_second),
        negative_log_likelihood=float(costs[first_count, second_count]),
        order_axis_xyz=tuple(float(value) for value in order_axis),
    )
