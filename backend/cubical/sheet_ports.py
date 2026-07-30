from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np

from .geometry import ClippedPatch, FaceTrace
from .topology import GridFace


PatchPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class FaceTraceCrossing:
    """A proper interior crossing of two sheetlet ports on one cell face."""

    first_patch_id: int
    second_patch_id: int
    face: GridFace
    intersection_xyz: tuple[float, float, float]
    first_fraction: float
    second_fraction: float
    crossing_angle_degrees: float

    @property
    def pair(self) -> PatchPair:
        return (
            min(self.first_patch_id, self.second_patch_id),
            max(self.first_patch_id, self.second_patch_id),
        )

    @property
    def severity(self) -> float:
        return math.sin(math.radians(self.crossing_angle_degrees))

    def record(self) -> dict[str, Any]:
        return {
            "firstPatchId": self.first_patch_id,
            "secondPatchId": self.second_patch_id,
            "faceAxis": self.face.axis,
            "faceAnchorXYZ": list(self.face.anchor_xyz),
            "intersectionXYZ": [
                round(value, 6) for value in self.intersection_xyz
            ],
            "firstFraction": round(self.first_fraction, 6),
            "secondFraction": round(self.second_fraction, 6),
            "crossingAngleDegrees": round(self.crossing_angle_degrees, 6),
            "severity": round(self.severity, 6),
        }


def _proper_trace_crossing(
    first: FaceTrace,
    second: FaceTrace,
) -> tuple[float, float, np.ndarray, float] | None:
    if first.face != second.face:
        raise ValueError("port crossing requires one canonical shared face")
    keep = tuple(axis for axis in range(3) if axis != first.face.axis)
    first_start_xyz = np.asarray(first.first.point_xyz, dtype=np.float64)
    first_stop_xyz = np.asarray(first.second.point_xyz, dtype=np.float64)
    second_start_xyz = np.asarray(second.first.point_xyz, dtype=np.float64)
    second_stop_xyz = np.asarray(second.second.point_xyz, dtype=np.float64)
    first_start = first_start_xyz[list(keep)]
    first_delta = first_stop_xyz[list(keep)] - first_start
    second_start = second_start_xyz[list(keep)]
    second_delta = second_stop_xyz[list(keep)] - second_start

    def cross(first_value: np.ndarray, second_value: np.ndarray) -> float:
        return float(
            first_value[0] * second_value[1]
            - first_value[1] * second_value[0]
        )

    denominator = cross(first_delta, second_delta)
    first_length = float(np.linalg.norm(first_delta))
    second_length = float(np.linalg.norm(second_delta))
    scale = max(first_length * second_length, 1.0)
    if abs(denominator) <= 1.0e-10 * scale:
        return None
    offset = second_start - first_start
    first_fraction = cross(offset, second_delta) / denominator
    second_fraction = cross(offset, first_delta) / denominator
    epsilon = 1.0e-8
    if not (
        epsilon < first_fraction < 1.0 - epsilon
        and epsilon < second_fraction < 1.0 - epsilon
    ):
        return None
    intersection = first_start_xyz + first_fraction * (
        first_stop_xyz - first_start_xyz
    )
    cosine = abs(float(np.dot(first_delta, second_delta))) / max(
        first_length * second_length, 1.0e-12
    )
    angle = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
    return first_fraction, second_fraction, intersection, angle


def enumerate_face_trace_crossings(
    patches: Iterable[ClippedPatch],
) -> tuple[FaceTraceCrossing, ...]:
    """Enumerate port crossings independently of a retained component graph."""

    by_face: dict[GridFace, list[tuple[ClippedPatch, FaceTrace]]] = defaultdict(
        list
    )
    for patch in patches:
        for trace in patch.traces:
            by_face[trace.face].append((patch, trace))
    crossings: list[FaceTraceCrossing] = []
    for face, values in by_face.items():
        lower_cell, upper_cell = face.adjacent_cells()
        lower = tuple(value for value in values if value[0].cell_xyz == lower_cell)
        upper = tuple(value for value in values if value[0].cell_xyz == upper_cell)
        for first_patch, first_trace in lower:
            for second_patch, second_trace in upper:
                measured = _proper_trace_crossing(first_trace, second_trace)
                if measured is None:
                    continue
                first_fraction, second_fraction, intersection, angle = measured
                crossings.append(
                    FaceTraceCrossing(
                        min(first_patch.patch_id, second_patch.patch_id),
                        max(first_patch.patch_id, second_patch.patch_id),
                        face,
                        tuple(float(value) for value in intersection),
                        first_fraction,
                        second_fraction,
                        angle,
                    )
                )
    crossings.sort(
        key=lambda value: (
            -value.severity,
            value.face.axis,
            value.face.anchor_xyz,
            value.first_patch_id,
            value.second_patch_id,
        )
    )
    return tuple(crossings)
