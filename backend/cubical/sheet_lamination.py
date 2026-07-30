from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .geometry import (
    ClippedPatch,
    axial_angle_radians,
    canonical_axis,
    plane_basis,
)
from .matching import TraceMatch


PatchPair = tuple[int, int]


@dataclass(frozen=True, slots=True)
class LayerExclusion:
    """A geometric reason two sheetlets cannot share one local surface chart."""

    first_patch_id: int
    second_patch_id: int
    reference_normal_xyz: tuple[float, float, float]
    overlap_fraction: float
    overlap_area: float
    normal_separation_cells: float
    normal_angle_degrees: float
    fiber_angle_degrees: float | None
    cell_delta_xyz: tuple[int, int, int]

    @property
    def pair(self) -> PatchPair:
        return (
            min(self.first_patch_id, self.second_patch_id),
            max(self.first_patch_id, self.second_patch_id),
        )

    @property
    def severity(self) -> float:
        return self.overlap_fraction * self.normal_separation_cells

    def record(self) -> dict[str, Any]:
        return {
            "firstPatchId": self.first_patch_id,
            "secondPatchId": self.second_patch_id,
            "referenceNormalXYZ": [
                round(value, 8) for value in self.reference_normal_xyz
            ],
            "cellDeltaXYZ": list(self.cell_delta_xyz),
            "overlapFraction": round(self.overlap_fraction, 6),
            "overlapArea": round(self.overlap_area, 6),
            "normalSeparationCells": round(
                self.normal_separation_cells, 6
            ),
            "normalAngleDegrees": round(self.normal_angle_degrees, 6),
            "fiberAngleDegrees": (
                round(self.fiber_angle_degrees, 6)
                if self.fiber_angle_degrees is not None
                else None
            ),
            "severity": round(self.severity, 6),
        }


@dataclass(frozen=True, slots=True)
class LaminationConflict:
    component_id: int
    exclusion: LayerExclusion
    parallel_path_length: int

    @property
    def severity(self) -> float:
        return self.exclusion.severity

    def record(self) -> dict[str, Any]:
        return {
            "componentId": self.component_id,
            **self.exclusion.record(),
            "parallelPathLength": self.parallel_path_length,
        }


@dataclass(frozen=True, slots=True)
class SheetLaminationAnalysis:
    conflicts: tuple[LaminationConflict, ...]
    component_records: tuple[dict[str, Any], ...]
    candidate_exclusion_count: int
    maximum_parallel_normal_angle_degrees: float
    proximity_radius_cells: int
    minimum_overlap_fraction: float
    minimum_normal_separation_cells: float

    def record(self, *, maximum_conflicts: int = 128) -> dict[str, Any]:
        return {
            "method": {
                "directions": "axial/unsigned",
                "collision": (
                    "interior overlap of local tangent projections at distinct "
                    "normal heights"
                ),
                "pathRequirement": (
                    "patches remain connected through a normal-coherent subgraph"
                ),
                "interpretation": (
                    "one retained identity is locally multi-valued along its "
                    "own normal"
                ),
            },
            "settings": {
                "maximumParallelNormalAngleDegrees": round(
                    self.maximum_parallel_normal_angle_degrees, 6
                ),
                "proximityRadiusCells": self.proximity_radius_cells,
                "minimumOverlapFraction": self.minimum_overlap_fraction,
                "minimumNormalSeparationCells": (
                    self.minimum_normal_separation_cells
                ),
            },
            "candidateExclusionCount": self.candidate_exclusion_count,
            "conflictCount": len(self.conflicts),
            "componentsWithConflicts": sum(
                int(value["conflictCount"]) > 0
                for value in self.component_records
            ),
            "components": list(self.component_records),
            "conflicts": [
                value.record() for value in self.conflicts[:maximum_conflicts]
            ],
        }


def _component_partition(
    patch_ids: Iterable[int],
    joins: tuple[TraceMatch, ...],
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    parent = {int(value): int(value) for value in patch_ids}
    size = {value: 1 for value in parent}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            following = parent[value]
            parent[value] = root
            value = following
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size.pop(second_root)

    for value in joins:
        union(value.first_patch_id, value.second_patch_id)
    groups: dict[int, list[int]] = defaultdict(list)
    for patch_id in parent:
        groups[find(patch_id)].append(patch_id)
    members = {
        min(values): tuple(sorted(values)) for values in groups.values()
    }
    component_by_patch = {
        patch_id: component_id
        for component_id, values in members.items()
        for patch_id in values
    }
    return component_by_patch, members


def _signed_area(points: np.ndarray) -> float:
    if len(points) < 3:
        return 0.0
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(
            points[:, 0] * following[:, 1]
            - following[:, 0] * points[:, 1]
        )
    )


def _counterclockwise(points: np.ndarray) -> np.ndarray:
    return points if _signed_area(points) >= 0.0 else points[::-1].copy()


def _cross_2d(first: np.ndarray, second: np.ndarray) -> float:
    return float(first[0] * second[1] - first[1] * second[0])


def _convex_intersection(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    """Return the convex polygon common to two counterclockwise polygons."""

    output = _counterclockwise(first)
    clip = _counterclockwise(second)
    epsilon = 1.0e-9
    for edge_start, edge_stop in zip(clip, np.roll(clip, -1, axis=0)):
        if len(output) == 0:
            break
        edge = edge_stop - edge_start
        input_values = output
        clipped: list[np.ndarray] = []
        prior = input_values[-1]
        prior_distance = _cross_2d(edge, prior - edge_start)
        prior_inside = prior_distance >= -epsilon
        for current in input_values:
            current_distance = _cross_2d(edge, current - edge_start)
            current_inside = current_distance >= -epsilon
            if current_inside != prior_inside:
                denominator = prior_distance - current_distance
                fraction = (
                    prior_distance / denominator
                    if abs(denominator) > epsilon
                    else 0.5
                )
                clipped.append(prior + fraction * (current - prior))
            if current_inside:
                clipped.append(current)
            prior = current
            prior_distance = current_distance
            prior_inside = current_inside
        output = (
            np.asarray(clipped, dtype=np.float64).reshape(-1, 2)
            if clipped
            else np.empty((0, 2), dtype=np.float64)
        )
    return output


def _polygon_centroid(points: np.ndarray) -> np.ndarray:
    area = _signed_area(points)
    if len(points) < 3 or abs(area) <= 1.0e-12:
        return np.mean(points, axis=0)
    following = np.roll(points, -1, axis=0)
    cross = (
        points[:, 0] * following[:, 1]
        - following[:, 0] * points[:, 1]
    )
    return np.asarray(
        (
            np.sum((points[:, 0] + following[:, 0]) * cross),
            np.sum((points[:, 1] + following[:, 1]) * cross),
        ),
        dtype=np.float64,
    ) / (6.0 * area)


def _project_patch(
    patch: ClippedPatch,
    first_axis: np.ndarray,
    second_axis: np.ndarray,
    normal: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    points = np.asarray(
        [value.point_xyz for value in patch.vertices], dtype=np.float64
    )
    projected = np.column_stack(
        (points @ first_axis, points @ second_axis, points @ normal)
    )
    return projected[:, :2], projected


def _height_plane(projected: np.ndarray) -> np.ndarray:
    design = np.column_stack(
        (projected[:, 0], projected[:, 1], np.ones(len(projected)))
    )
    coefficients, _, _, _ = np.linalg.lstsq(
        design, projected[:, 2], rcond=None
    )
    return coefficients


def _fiber_angle_degrees(
    first: ClippedPatch, second: ClippedPatch
) -> float | None:
    if first.estimate.fiber_xyz is None or second.estimate.fiber_xyz is None:
        return None
    return math.degrees(
        axial_angle_radians(
            first.estimate.fiber_xyz,
            second.estimate.fiber_xyz,
        )
    )


def _validate_settings(
    cell_size_xyz: Iterable[float],
    maximum_parallel_normal_angle_degrees: float,
    proximity_radius_cells: int,
    minimum_overlap_fraction: float,
    minimum_normal_separation_cells: float,
) -> np.ndarray:
    cell_size = np.asarray(tuple(cell_size_xyz), dtype=np.float64)
    if cell_size.shape != (3,) or np.any(~np.isfinite(cell_size)) or np.any(
        cell_size <= 0.0
    ):
        raise ValueError("lamination analysis requires three positive cell sizes")
    if (
        not math.isfinite(maximum_parallel_normal_angle_degrees)
        or not 0.0 < maximum_parallel_normal_angle_degrees <= 90.0
    ):
        raise ValueError("parallel-normal limit must lie in (0, 90]")
    if proximity_radius_cells < 1:
        raise ValueError("lamination proximity radius must be positive")
    if not 0.0 <= minimum_overlap_fraction <= 1.0:
        raise ValueError("minimum overlap fraction must lie in [0, 1]")
    if (
        not math.isfinite(minimum_normal_separation_cells)
        or minimum_normal_separation_cells < 0.0
    ):
        raise ValueError("minimum normal separation must be finite and nonnegative")
    return cell_size


def _measure_layer_exclusion(
    first: ClippedPatch,
    second: ClippedPatch,
    *,
    cell_size: np.ndarray,
    maximum_parallel_normal_angle_degrees: float,
    minimum_overlap_fraction: float,
    minimum_normal_separation_cells: float,
) -> LayerExclusion | None:
    normal_angle = math.degrees(
        axial_angle_radians(
            first.estimate.normal_xyz,
            second.estimate.normal_xyz,
        )
    )
    if normal_angle > maximum_parallel_normal_angle_degrees:
        return None
    first_normal = canonical_axis(first.estimate.normal_xyz)
    second_normal = canonical_axis(second.estimate.normal_xyz)
    tensor = np.outer(first_normal, first_normal) + np.outer(
        second_normal, second_normal
    )
    _, eigenvectors = np.linalg.eigh(tensor)
    normal = canonical_axis(eigenvectors[:, -1])
    first_axis, second_axis = plane_basis(normal)
    first_polygon, first_projected = _project_patch(
        first, first_axis, second_axis, normal
    )
    second_polygon, second_projected = _project_patch(
        second, first_axis, second_axis, normal
    )
    first_area = abs(_signed_area(first_polygon))
    second_area = abs(_signed_area(second_polygon))
    if min(first_area, second_area) <= 1.0e-9:
        return None
    intersection = _convex_intersection(first_polygon, second_polygon)
    if len(intersection) < 3:
        return None
    overlap_area = abs(_signed_area(intersection))
    overlap_fraction = overlap_area / min(first_area, second_area)
    if overlap_fraction < minimum_overlap_fraction:
        return None
    overlap_center = _polygon_centroid(intersection)
    center_homogeneous = np.asarray(
        (overlap_center[0], overlap_center[1], 1.0), dtype=np.float64
    )
    first_height = float(
        np.dot(center_homogeneous, _height_plane(first_projected))
    )
    second_height = float(
        np.dot(center_homogeneous, _height_plane(second_projected))
    )
    normal_cell_size = float(np.linalg.norm(normal * cell_size))
    separation = abs(first_height - second_height) / max(
        normal_cell_size, 1.0e-12
    )
    if separation < minimum_normal_separation_cells:
        return None
    delta = tuple(
        abs(first.cell_xyz[axis] - second.cell_xyz[axis])
        for axis in range(3)
    )
    return LayerExclusion(
        min(first.patch_id, second.patch_id),
        max(first.patch_id, second.patch_id),
        tuple(float(value) for value in normal),
        overlap_fraction,
        overlap_area,
        separation,
        normal_angle,
        _fiber_angle_degrees(first, second),
        delta,
    )


def enumerate_layer_exclusions(
    patches: Iterable[ClippedPatch],
    *,
    cell_size_xyz: Iterable[float],
    maximum_parallel_normal_angle_degrees: float,
    proximity_radius_cells: int = 2,
    minimum_overlap_fraction: float = 0.25,
    minimum_normal_separation_cells: float = 0.35,
) -> tuple[LayerExclusion, ...]:
    """Enumerate typed repulsive edges independent of a current partition."""

    cell_size = _validate_settings(
        cell_size_xyz,
        maximum_parallel_normal_angle_degrees,
        proximity_radius_cells,
        minimum_overlap_fraction,
        minimum_normal_separation_cells,
    )
    patch_values = tuple(sorted(patches, key=lambda value: value.patch_id))
    by_cell: dict[tuple[int, int, int], list[ClippedPatch]] = defaultdict(list)
    for patch in patch_values:
        by_cell[patch.cell_xyz].append(patch)
    exclusions: list[LayerExclusion] = []
    for first in patch_values:
        cell = first.cell_xyz
        for z_offset in range(-proximity_radius_cells, proximity_radius_cells + 1):
            for y_offset in range(
                -proximity_radius_cells, proximity_radius_cells + 1
            ):
                for x_offset in range(
                    -proximity_radius_cells, proximity_radius_cells + 1
                ):
                    neighbor = (
                        cell[0] + x_offset,
                        cell[1] + y_offset,
                        cell[2] + z_offset,
                    )
                    for second in by_cell.get(neighbor, ()):
                        if second.patch_id <= first.patch_id:
                            continue
                        exclusion = _measure_layer_exclusion(
                            first,
                            second,
                            cell_size=cell_size,
                            maximum_parallel_normal_angle_degrees=(
                                maximum_parallel_normal_angle_degrees
                            ),
                            minimum_overlap_fraction=minimum_overlap_fraction,
                            minimum_normal_separation_cells=(
                                minimum_normal_separation_cells
                            ),
                        )
                        if exclusion is not None:
                            exclusions.append(exclusion)
    exclusions.sort(
        key=lambda value: (
            -value.severity,
            -value.overlap_fraction,
            value.first_patch_id,
            value.second_patch_id,
        )
    )
    return tuple(exclusions)


def _parallel_path_length(
    source: int,
    target: int,
    adjacency: Mapping[int, tuple[int, ...]],
    patch_by_id: Mapping[int, ClippedPatch],
    reference_normal: np.ndarray,
    maximum_angle_degrees: float,
) -> int | None:
    distance = {source: 0}
    queue: deque[int] = deque((source,))
    while queue:
        first = queue.popleft()
        if first == target:
            return distance[first]
        for second in adjacency.get(first, ()):
            if second in distance:
                continue
            deviation = math.degrees(
                axial_angle_radians(
                    patch_by_id[second].estimate.normal_xyz,
                    reference_normal,
                )
            )
            if deviation > maximum_angle_degrees + 1.0e-9:
                continue
            distance[second] = distance[first] + 1
            queue.append(second)
    return None


def analyze_sheet_lamination(
    patches: Iterable[ClippedPatch],
    joins: Iterable[TraceMatch],
    *,
    cell_size_xyz: Iterable[float],
    maximum_parallel_normal_angle_degrees: float,
    proximity_radius_cells: int = 2,
    minimum_overlap_fraction: float = 0.25,
    minimum_normal_separation_cells: float = 0.35,
    exclusions: Iterable[LayerExclusion] | None = None,
) -> SheetLaminationAnalysis:
    """Find locally multi-valued identities made from near-parallel layers.

    A conflict requires two excluded sheetlets to remain connected through a
    path whose normals stay in the same local cone. A legitimate large-scale
    wrap whose connecting path rotates out of that cone is therefore not called
    a fused layer merely because it returns to the same spatial column.
    """

    patch_values = tuple(patches)
    join_values = tuple(joins)
    patch_by_id = {value.patch_id: value for value in patch_values}
    component_by_patch, members = _component_partition(
        patch_by_id, join_values
    )
    exclusion_values = (
        tuple(exclusions)
        if exclusions is not None
        else enumerate_layer_exclusions(
            patch_values,
            cell_size_xyz=cell_size_xyz,
            maximum_parallel_normal_angle_degrees=(
                maximum_parallel_normal_angle_degrees
            ),
            proximity_radius_cells=proximity_radius_cells,
            minimum_overlap_fraction=minimum_overlap_fraction,
            minimum_normal_separation_cells=(
                minimum_normal_separation_cells
            ),
        )
    )
    adjacency_values: dict[int, set[int]] = defaultdict(set)
    for value in join_values:
        adjacency_values[value.first_patch_id].add(value.second_patch_id)
        adjacency_values[value.second_patch_id].add(value.first_patch_id)
    adjacency = {
        patch_id: tuple(sorted(values))
        for patch_id, values in adjacency_values.items()
    }
    conflicts: list[LaminationConflict] = []
    conflicts_by_component: dict[int, list[LaminationConflict]] = defaultdict(list)
    for exclusion in exclusion_values:
        component_id = component_by_patch[exclusion.first_patch_id]
        if component_by_patch[exclusion.second_patch_id] != component_id:
            continue
        path_length = _parallel_path_length(
            exclusion.first_patch_id,
            exclusion.second_patch_id,
            adjacency,
            patch_by_id,
            np.asarray(exclusion.reference_normal_xyz, dtype=np.float64),
            maximum_parallel_normal_angle_degrees,
        )
        if path_length is None:
            continue
        conflict = LaminationConflict(component_id, exclusion, path_length)
        conflicts.append(conflict)
        conflicts_by_component[component_id].append(conflict)
    conflicts.sort(
        key=lambda value: (
            -value.severity,
            value.component_id,
            value.exclusion.first_patch_id,
            value.exclusion.second_patch_id,
        )
    )
    component_records = []
    for component_id, member_ids in members.items():
        values = conflicts_by_component.get(component_id, ())
        component_records.append(
            {
                "componentId": component_id,
                "patchCount": len(member_ids),
                "conflictCount": len(values),
                "maximumSeverity": round(
                    max((value.severity for value in values), default=0.0), 6
                ),
                "maximumOverlapFraction": round(
                    max(
                        (
                            value.exclusion.overlap_fraction
                            for value in values
                        ),
                        default=0.0,
                    ),
                    6,
                ),
                "maximumNormalSeparationCells": round(
                    max(
                        (
                            value.exclusion.normal_separation_cells
                            for value in values
                        ),
                        default=0.0,
                    ),
                    6,
                ),
            }
        )
    component_records.sort(
        key=lambda value: (-int(value["patchCount"]), int(value["componentId"]))
    )
    return SheetLaminationAnalysis(
        tuple(conflicts),
        tuple(component_records),
        len(exclusion_values),
        maximum_parallel_normal_angle_degrees,
        proximity_radius_cells,
        minimum_overlap_fraction,
        minimum_normal_separation_cells,
    )
