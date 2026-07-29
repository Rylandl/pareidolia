from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .topology import (
    Float3,
    GridEdge,
    GridFace,
    GridSpec,
    Int3,
    cell_edges,
    edge_supporting_faces,
)


class DegeneratePlaneIntersection(ValueError):
    """A plane contains a cube vertex/edge and needs an explicit alternative."""


def canonical_axis(vector: Iterable[float]) -> np.ndarray:
    """Normalize an unsigned axis into a deterministic sign gauge."""

    values = np.asarray(tuple(vector), dtype=np.float64)
    if values.shape != (3,) or not np.all(np.isfinite(values)):
        raise ValueError("axis must contain three finite values")
    length = float(np.linalg.norm(values))
    if length <= 1.0e-12:
        raise ValueError("axis must be nonzero")
    values = values / length
    dominant = int(np.argmax(np.abs(values)))
    if values[dominant] < 0.0:
        values = -values
    return values


def plane_basis(normal: Iterable[float]) -> tuple[np.ndarray, np.ndarray]:
    """Return a deterministic right-handed tangent basis for an axial normal."""

    normal_axis = canonical_axis(normal)
    helper = np.zeros(3, dtype=np.float64)
    helper[int(np.argmin(np.abs(normal_axis)))] = 1.0
    u_axis = np.cross(normal_axis, helper)
    u_axis /= max(float(np.linalg.norm(u_axis)), 1.0e-12)
    u_axis = canonical_axis(u_axis)
    v_axis = np.cross(normal_axis, u_axis)
    v_axis /= max(float(np.linalg.norm(v_axis)), 1.0e-12)
    return u_axis, v_axis


def axial_angle_radians(first: Iterable[float], second: Iterable[float]) -> float:
    first_axis = canonical_axis(first)
    second_axis = canonical_axis(second)
    cosine = float(np.clip(abs(np.dot(first_axis, second_axis)), 0.0, 1.0))
    return math.acos(cosine)


def _covariance(values: Iterable[Iterable[float]]) -> np.ndarray:
    covariance = np.asarray(values, dtype=np.float64)
    if covariance.shape != (3, 3) or not np.all(np.isfinite(covariance)):
        raise ValueError("plane covariance must be a finite 3 x 3 matrix")
    if not np.allclose(covariance, covariance.T, atol=1.0e-10):
        raise ValueError("plane covariance must be symmetric")
    eigenvalues = np.linalg.eigvalsh(covariance)
    if float(np.min(eigenvalues)) < -1.0e-10:
        raise ValueError("plane covariance must be positive semidefinite")
    return covariance


@dataclass(frozen=True, slots=True)
class PlaneEstimate:
    """A cell-centered axial plane posterior.

    Covariance coordinates are ``(tilt_u radians, tilt_v radians, height)`` in
    the deterministic tangent basis returned by :func:`plane_basis`.
    """

    normal_xyz: Float3
    height_from_cell_center: float
    covariance: tuple[tuple[float, float, float], ...]
    fiber_xyz: Float3 | None = None
    fiber_angular_std_radians: float | None = None
    confidence: float = 1.0

    def __post_init__(self) -> None:
        raw_normal = np.asarray(self.normal_xyz, dtype=np.float64)
        normal = canonical_axis(raw_normal)
        raw_unit = raw_normal / max(float(np.linalg.norm(raw_normal)), 1.0e-12)
        height = float(self.height_from_cell_center)
        if float(np.dot(raw_unit, normal)) < 0.0:
            height = -height
        if not np.isfinite(height):
            raise ValueError("plane height must be finite")
        covariance = _covariance(self.covariance)
        confidence = float(self.confidence)
        if not np.isfinite(confidence) or not 0.0 <= confidence <= 1.0:
            raise ValueError("confidence must lie in [0, 1]")
        fiber: tuple[float, float, float] | None = None
        if self.fiber_xyz is not None:
            fiber_values = canonical_axis(self.fiber_xyz)
            fiber_values -= normal * float(np.dot(fiber_values, normal))
            if float(np.linalg.norm(fiber_values)) <= 1.0e-8:
                raise ValueError("fiber axis must have a component in the plane")
            fiber_values = canonical_axis(fiber_values)
            fiber = tuple(float(value) for value in fiber_values)
        fiber_std = self.fiber_angular_std_radians
        if fiber_std is not None and (not np.isfinite(fiber_std) or fiber_std < 0.0):
            raise ValueError("fiber angular standard deviation must be nonnegative")
        object.__setattr__(self, "normal_xyz", tuple(float(value) for value in normal))
        object.__setattr__(self, "height_from_cell_center", height)
        object.__setattr__(
            self,
            "covariance",
            tuple(tuple(float(value) for value in row) for row in covariance),
        )
        object.__setattr__(self, "fiber_xyz", fiber)
        object.__setattr__(self, "confidence", confidence)

    @classmethod
    def isotropic(
        cls,
        normal_xyz: Iterable[float],
        height_from_cell_center: float,
        angular_std_radians: float,
        height_std: float,
        *,
        fiber_xyz: Iterable[float] | None = None,
        fiber_angular_std_radians: float | None = None,
        confidence: float = 1.0,
    ) -> "PlaneEstimate":
        if angular_std_radians < 0.0 or height_std < 0.0:
            raise ValueError("posterior standard deviations must be nonnegative")
        covariance = np.diag(
            [angular_std_radians**2, angular_std_radians**2, height_std**2]
        )
        return cls(
            tuple(float(value) for value in normal_xyz),
            height_from_cell_center,
            tuple(tuple(float(value) for value in row) for row in covariance),
            tuple(float(value) for value in fiber_xyz) if fiber_xyz is not None else None,
            fiber_angular_std_radians,
            confidence,
        )

    @property
    def covariance_matrix(self) -> np.ndarray:
        return np.asarray(self.covariance, dtype=np.float64)

    @property
    def maximum_angular_std_radians(self) -> float:
        angular = self.covariance_matrix[:2, :2]
        return math.sqrt(max(float(np.max(np.linalg.eigvalsh(angular))), 0.0))


@dataclass(frozen=True, slots=True)
class EdgeCrossing:
    edge: GridEdge
    t: float
    variance: float
    point_xyz: Float3

    @property
    def standard_deviation(self) -> float:
        return math.sqrt(max(self.variance, 0.0))


@dataclass(frozen=True, slots=True)
class FaceTrace:
    face: GridFace
    first: EdgeCrossing
    second: EdgeCrossing
    patch_id: int

    @property
    def endpoint_edges(self) -> frozenset[GridEdge]:
        return frozenset((self.first.edge, self.second.edge))

    @property
    def midpoint_xyz(self) -> np.ndarray:
        return 0.5 * (
            np.asarray(self.first.point_xyz) + np.asarray(self.second.point_xyz)
        )


@dataclass(frozen=True, slots=True)
class ClippedPatch:
    patch_id: int
    cell_xyz: Int3
    estimate: PlaneEstimate
    vertices: tuple[EdgeCrossing, ...]
    traces: tuple[FaceTrace, ...]

    def trace_on(self, face: GridFace) -> FaceTrace | None:
        matches = [trace for trace in self.traces if trace.face == face]
        if len(matches) > 1:
            raise RuntimeError("one planar patch produced multiple traces on a face")
        return matches[0] if matches else None


def _edge_crossing_variance(
    grid: GridSpec,
    cell_xyz: Int3,
    edge: GridEdge,
    estimate: PlaneEstimate,
    t: float,
) -> float:
    normal = np.asarray(estimate.normal_xyz, dtype=np.float64)
    u_axis, v_axis = plane_basis(normal)
    center = grid.cell_center_world(cell_xyz)
    first_vertex, second_vertex = edge.endpoint_vertices()
    first = grid.vertex_world(first_vertex) - center
    direction = grid.vertex_world(second_vertex) - grid.vertex_world(first_vertex)
    denominator = float(np.dot(normal, direction))
    if abs(denominator) <= 1.0e-14:
        return math.inf
    numerator = float(estimate.height_from_cell_center - np.dot(normal, first))
    derivatives = []
    for tangent in (u_axis, v_axis):
        derivatives.append(
            (
                -float(np.dot(tangent, first)) * denominator
                - numerator * float(np.dot(tangent, direction))
            )
            / denominator**2
        )
    derivatives.append(1.0 / denominator)
    jacobian = np.asarray(derivatives, dtype=np.float64)
    variance = float(jacobian @ estimate.covariance_matrix @ jacobian)
    if variance < -1.0e-10:
        raise RuntimeError("edge variance became negative")
    return max(variance, 0.0)


def clip_plane_to_cell(
    grid: GridSpec,
    cell_xyz: Int3,
    estimate: PlaneEstimate,
    *,
    patch_id: int = 0,
    tolerance: float | None = None,
) -> ClippedPatch | None:
    """Clip one generic plane estimate to a cell.

    ``None`` means the plane misses the cell. Exact vertex/edge containment is
    reported as :class:`DegeneratePlaneIntersection` so callers can preserve an
    alternative instead of accepting a platform-dependent topology.
    """

    cell = grid.require_cell(cell_xyz)
    cell_center = grid.cell_center_world(cell)
    normal = np.asarray(estimate.normal_xyz, dtype=np.float64)
    scale = max(grid.cell_size_xyz)
    epsilon = float(tolerance) if tolerance is not None else scale * 1.0e-8
    if epsilon <= 0.0 or not np.isfinite(epsilon):
        raise ValueError("clipping tolerance must be finite and positive")

    crossings: list[EdgeCrossing] = []
    for edge in cell_edges(cell):
        start_vertex, stop_vertex = edge.endpoint_vertices()
        start = grid.vertex_world(start_vertex)
        stop = grid.vertex_world(stop_vertex)
        start_distance = float(
            np.dot(normal, start - cell_center) - estimate.height_from_cell_center
        )
        stop_distance = float(
            np.dot(normal, stop - cell_center) - estimate.height_from_cell_center
        )
        if abs(start_distance) <= epsilon or abs(stop_distance) <= epsilon:
            raise DegeneratePlaneIntersection(
                f"plane for patch {patch_id} contains a cube vertex within tolerance"
            )
        if start_distance * stop_distance >= 0.0:
            continue
        denominator = start_distance - stop_distance
        t = start_distance / denominator
        if not epsilon / scale < t < 1.0 - epsilon / scale:
            raise DegeneratePlaneIntersection(
                f"plane for patch {patch_id} crosses too close to a cube vertex"
            )
        point = (1.0 - t) * start + t * stop
        variance = _edge_crossing_variance(grid, cell, edge, estimate, t)
        crossings.append(
            EdgeCrossing(
                edge=edge,
                t=float(t),
                variance=variance,
                point_xyz=tuple(float(value) for value in point),
            )
        )

    if not crossings:
        return None
    if not 3 <= len(crossings) <= 6:
        raise DegeneratePlaneIntersection(
            f"generic plane/cube intersection has {len(crossings)} vertices"
        )

    points = np.asarray([crossing.point_xyz for crossing in crossings])
    centroid = np.mean(points, axis=0)
    u_axis, v_axis = plane_basis(normal)
    relative = points - centroid
    angles = np.arctan2(relative @ v_axis, relative @ u_axis)
    ordered = [crossings[int(index)] for index in np.argsort(angles)]
    first_index = min(
        range(len(ordered)),
        key=lambda index: (
            ordered[index].edge.axis,
            ordered[index].edge.anchor_xyz,
            ordered[index].t,
        ),
    )
    ordered = ordered[first_index:] + ordered[:first_index]

    traces: list[FaceTrace] = []
    for index, first in enumerate(ordered):
        second = ordered[(index + 1) % len(ordered)]
        common_faces = edge_supporting_faces(first.edge, cell) & edge_supporting_faces(
            second.edge, cell
        )
        if len(common_faces) != 1:
            raise DegeneratePlaneIntersection(
                "consecutive polygon vertices do not define one unique cube face"
            )
        face = next(iter(common_faces))
        traces.append(FaceTrace(face, first, second, int(patch_id)))

    return ClippedPatch(
        patch_id=int(patch_id),
        cell_xyz=cell,
        estimate=estimate,
        vertices=tuple(ordered),
        traces=tuple(traces),
    )
