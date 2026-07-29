from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Iterable

import numpy as np


Int3 = tuple[int, int, int]
Float3 = tuple[float, float, float]


def _int3(values: Iterable[int]) -> Int3:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected three integer coordinates")
    return result  # type: ignore[return-value]


def _float3(values: Iterable[float]) -> Float3:
    result = tuple(float(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected three floating-point coordinates")
    return result  # type: ignore[return-value]


@dataclass(frozen=True, slots=True)
class GridSpec:
    """An axis-aligned finite cell grid in an explicit world coordinate unit."""

    shape_cells_xyz: Int3
    cell_size_xyz: Float3 = (1.0, 1.0, 1.0)
    origin_xyz: Float3 = (0.0, 0.0, 0.0)
    coordinate_unit: str = "cell"

    def __post_init__(self) -> None:
        shape = _int3(self.shape_cells_xyz)
        cell_size = _float3(self.cell_size_xyz)
        origin = _float3(self.origin_xyz)
        if any(value <= 0 for value in shape):
            raise ValueError("grid shape must be positive on every axis")
        if any(not np.isfinite(value) or value <= 0.0 for value in cell_size):
            raise ValueError("cell size must be finite and positive on every axis")
        if any(not np.isfinite(value) for value in origin):
            raise ValueError("grid origin must be finite")
        if not self.coordinate_unit:
            raise ValueError("coordinate unit must be non-empty")
        object.__setattr__(self, "shape_cells_xyz", shape)
        object.__setattr__(self, "cell_size_xyz", cell_size)
        object.__setattr__(self, "origin_xyz", origin)

    def contains_cell(self, cell_xyz: Int3) -> bool:
        cell = _int3(cell_xyz)
        return all(0 <= cell[axis] < self.shape_cells_xyz[axis] for axis in range(3))

    def require_cell(self, cell_xyz: Int3) -> Int3:
        cell = _int3(cell_xyz)
        if not self.contains_cell(cell):
            raise ValueError(f"cell {cell} lies outside grid {self.shape_cells_xyz}")
        return cell

    def vertex_world(self, vertex_xyz: Iterable[float]) -> np.ndarray:
        vertex = np.asarray(tuple(vertex_xyz), dtype=np.float64)
        if vertex.shape != (3,):
            raise ValueError("expected one XYZ grid vertex")
        return np.asarray(self.origin_xyz) + vertex * np.asarray(self.cell_size_xyz)

    def cell_center_world(self, cell_xyz: Int3) -> np.ndarray:
        cell = np.asarray(self.require_cell(cell_xyz), dtype=np.float64)
        return self.vertex_world(cell + 0.5)

    def cell_bounds_world(self, cell_xyz: Int3) -> tuple[np.ndarray, np.ndarray]:
        cell = np.asarray(self.require_cell(cell_xyz), dtype=np.float64)
        return self.vertex_world(cell), self.vertex_world(cell + 1.0)


@dataclass(frozen=True, order=True, slots=True)
class GridEdge:
    """A canonical unit grid edge identified independently of an incident cell."""

    axis: int
    anchor_xyz: Int3

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("edge axis must be X, Y, or Z")
        object.__setattr__(self, "anchor_xyz", _int3(self.anchor_xyz))

    def endpoint_vertices(self) -> tuple[Int3, Int3]:
        stop = list(self.anchor_xyz)
        stop[self.axis] += 1
        return self.anchor_xyz, _int3(stop)

    def point_world(self, grid: GridSpec, t: float) -> np.ndarray:
        if not np.isfinite(t):
            raise ValueError("edge coordinate must be finite")
        start, stop = self.endpoint_vertices()
        return (1.0 - t) * grid.vertex_world(start) + t * grid.vertex_world(stop)


@dataclass(frozen=True, order=True, slots=True)
class GridFace:
    """A canonical unit grid face; axis is its normal coordinate axis."""

    axis: int
    anchor_xyz: Int3

    def __post_init__(self) -> None:
        if self.axis not in (0, 1, 2):
            raise ValueError("face axis must be X, Y, or Z")
        object.__setattr__(self, "anchor_xyz", _int3(self.anchor_xyz))

    def adjacent_cells(self) -> tuple[Int3, Int3]:
        upper = list(self.anchor_xyz)
        lower = list(self.anchor_xyz)
        lower[self.axis] -= 1
        return _int3(lower), _int3(upper)


def cell_edges(cell_xyz: Int3) -> tuple[GridEdge, ...]:
    cell = _int3(cell_xyz)
    edges: list[GridEdge] = []
    for axis in range(3):
        other_axes = [value for value in range(3) if value != axis]
        for first, second in product((0, 1), repeat=2):
            anchor = list(cell)
            anchor[other_axes[0]] += first
            anchor[other_axes[1]] += second
            edges.append(GridEdge(axis, _int3(anchor)))
    return tuple(edges)


def cell_face(cell_xyz: Int3, axis: int, side: int) -> GridFace:
    if axis not in (0, 1, 2) or side not in (0, 1):
        raise ValueError("cell face requires an XYZ axis and binary side")
    anchor = list(_int3(cell_xyz))
    anchor[axis] += side
    return GridFace(axis, _int3(anchor))


def cell_faces(cell_xyz: Int3) -> tuple[GridFace, ...]:
    return tuple(cell_face(cell_xyz, axis, side) for axis in range(3) for side in (0, 1))


def edge_supporting_faces(edge: GridEdge, cell_xyz: Int3) -> frozenset[GridFace]:
    """Return the two faces of ``cell_xyz`` containing its local ``edge``."""

    cell = _int3(cell_xyz)
    result: set[GridFace] = set()
    for axis in range(3):
        if axis == edge.axis:
            continue
        delta = edge.anchor_xyz[axis] - cell[axis]
        if delta not in (0, 1):
            raise ValueError(f"edge {edge} is not incident to cell {cell}")
        result.add(cell_face(cell, axis, delta))
    edge_start, edge_stop = edge.endpoint_vertices()
    for axis in range(3):
        low = cell[axis]
        high = cell[axis] + 1
        if not (low <= edge_start[axis] <= high and low <= edge_stop[axis] <= high):
            raise ValueError(f"edge {edge} is not incident to cell {cell}")
    return frozenset(result)
