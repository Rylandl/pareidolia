from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import AbstractSet, Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .needle_surface import _triangle_geometry
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_depth_fields import _profile_fields
from .physical_ribbon_patch_holes import (
    PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
    PhysicalRibbonPatchHoleSettings,
    _point_in_polygon,
    _sample_normal_profiles,
    extract_surface_boundary_loops,
)
from .physical_ribbon_patch_states import (
    _resolve_depth_field_manifest,
    _surface_view,
)
from .physical_ribbon_surface_holes import PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA


PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA = (
    "pareidolia.physical-ribbon-dense-completion"
)
PHYSICAL_RIBBON_DENSE_COMPLETION_VERSION = 1
PHYSICAL_RIBBON_DENSE_COMPLETION_STEM = "physical-ribbon-dense-completion-v1"


def _resolve_holes_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema")
            in {
                PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
                PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA,
            }
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "holes root must identify one complete patch or surface-hole artifact"
        )
    return matches[0]


def _validate_depth_field(
    holes_path: Path,
    holes_manifest: Mapping[str, Any],
    holes: Mapping[str, np.ndarray],
    depth_manifest: Mapping[str, Any],
    depth: Mapping[str, np.ndarray],
) -> None:
    reference = depth_manifest.get("identity", {}).get("holes", {})
    if (
        reference.get("manifestPath") != str(holes_path)
        or reference.get("manifestSha256") != sha256_file(holes_path)
        or reference.get("dataSha256") != holes_manifest["data"]["sha256"]
    ):
        raise ValueError("depth field was not measured on the requested holes")
    if not np.array_equal(depth["holePatchOffset"], holes["patchOffset"]):
        raise ValueError("depth field and hole patch rasters differ")
    if not np.array_equal(depth["holeLoopIndex"], holes["scoredLoopIndex"]):
        raise ValueError("depth field and scored hole order differ")
    if "patchCandidateOffset" in holes and "candidateDepthOffset" in depth:
        if not np.array_equal(
            depth["candidateDepthOffset"], holes["patchCandidateOffset"]
        ):
            raise ValueError("depth field and optional candidate order differ")


@dataclass(frozen=True, slots=True)
class PhysicalRibbonDenseCompletionSettings:
    """Promote a complete CT depth field to one exact surface patch.

    Dense pixels are geometric samples, not independent growth decisions.  A
    completion is accepted or rejected as one closed-loop state after every
    boundary edge, mesh, CT, and competing-layer invariant is evaluated.
    """

    minimum_depth_field_supported_fraction: float = 0.90
    minimum_reconstructed_supported_fraction: float = 0.85
    minimum_median_profile_correlation: float = 0.70
    minimum_median_far_layer_margin: float = 0.02
    minimum_interior_boundary_separation_voxels: float = 0.20
    minimum_triangle_area_voxels_squared: float = 0.03
    maximum_triangle_edge_voxels: float = 6.0
    high_triangle_normal_residual_degrees: float = 45.0
    maximum_triangle_normal_residual_degrees: float = 85.0
    intersection_tolerance_voxels: float = 0.05
    maximum_edge_flip_iterations: int = 16_384
    maximum_completed_holes: int = 64
    profile_depth_fractions: tuple[float, ...] = (
        -0.85,
        -0.65,
        -0.35,
        0.0,
        0.35,
        0.65,
        0.85,
    )
    competing_shift_thicknesses: tuple[float, ...] = (
        -1.0,
        -0.5,
        0.0,
        0.5,
        1.0,
    )

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_depth_field_supported_fraction,
            self.minimum_reconstructed_supported_fraction,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in fractions
        ):
            raise ValueError("dense-completion fractions must lie in [0, 1]")
        if not -1.0 <= self.minimum_median_profile_correlation <= 1.0:
            raise ValueError("profile-correlation gate must lie in [-1, 1]")
        positive = (
            self.minimum_interior_boundary_separation_voxels,
            self.minimum_triangle_area_voxels_squared,
            self.maximum_triangle_edge_voxels,
            self.high_triangle_normal_residual_degrees,
            self.maximum_triangle_normal_residual_degrees,
            self.intersection_tolerance_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("dense-completion geometric scales must be positive")
        if (
            self.maximum_triangle_normal_residual_degrees >= 90.0
            or self.high_triangle_normal_residual_degrees
            >= self.maximum_triangle_normal_residual_degrees
        ):
            raise ValueError("triangle-normal residual gates must be ordered below 90 degrees")
        if self.maximum_edge_flip_iterations < 1 or self.maximum_completed_holes < 1:
            raise ValueError("dense-completion iteration counts must be positive")
        depths = tuple(float(value) for value in self.profile_depth_fractions)
        shifts = tuple(float(value) for value in self.competing_shift_thicknesses)
        if len(depths) < 5 or tuple(sorted(depths)) != depths or 0.0 not in depths:
            raise ValueError("profile depths must be sorted and include zero")
        if len(shifts) < 3 or tuple(sorted(shifts)) != shifts or 0.0 not in shifts:
            raise ValueError("competing shifts must be sorted and include zero")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _cross_2d(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    left = second - first
    right = third - first
    return float(left[0] * right[1] - left[1] * right[0])


def _signed_polygon_area(polygon: np.ndarray) -> float:
    points = np.asarray(polygon, dtype=np.float64)
    return 0.5 * float(
        np.sum(
            points[:, 0] * np.roll(points[:, 1], -1)
            - points[:, 1] * np.roll(points[:, 0], -1)
        )
    )


def decompose_weak_boundary_cycles(boundary: np.ndarray) -> tuple[np.ndarray, ...]:
    """Split a closed boundary walk at repeated pinch vertices.

    A triangle mesh can have several boundary fans touching at one vertex.  Its
    combinatorial loop is then weakly simple and ordinary polygon triangulation
    would cut across a real fold.  Stack decomposition returns the simple disk
    boundaries while preserving the original boundary-edge multiset exactly.
    """

    values = [int(value) for value in np.asarray(boundary, dtype=np.int64)]
    if len(values) < 3 or len(set(values)) < 3:
        raise ValueError("a dense completion requires a nontrivial boundary")
    stack: list[int] = []
    position: dict[int, int] = {}
    cycles: list[np.ndarray] = []
    for vertex in (*values, values[0]):
        previous = position.get(vertex)
        if previous is None:
            position[vertex] = len(stack)
            stack.append(vertex)
            continue
        cycle = stack[previous:]
        if len(cycle) >= 3:
            cycles.append(np.asarray(cycle, dtype=np.int32))
        for removed in stack[previous + 1 :]:
            position.pop(removed, None)
        stack = stack[: previous + 1]
    if not cycles:
        raise ValueError("boundary walk did not contain a closed disk cycle")

    original_edges = Counter(
        tuple(sorted((values[index], values[(index + 1) % len(values)])))
        for index in range(len(values))
    )
    cycle_edges: Counter[tuple[int, int]] = Counter()
    for cycle in cycles:
        cycle_edges.update(
            tuple(sorted((int(cycle[index]), int(cycle[(index + 1) % len(cycle)]))))
            for index in range(len(cycle))
        )
    if cycle_edges != original_edges:
        raise ValueError("pinch decomposition did not preserve boundary edges")
    if any(count != 1 for count in original_edges.values()):
        raise ValueError("boundary traverses the same mesh edge more than once")
    return tuple(cycles)


def _point_in_or_on_triangle(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    *,
    tolerance: float,
) -> bool:
    return min(
        _cross_2d(first, second, point),
        _cross_2d(second, third, point),
        _cross_2d(third, first, point),
    ) >= -tolerance


def _orient_chart_triangle(
    triangle: tuple[int, int, int], chart_uv: np.ndarray
) -> tuple[int, int, int]:
    if _cross_2d(
        chart_uv[triangle[0]], chart_uv[triangle[1]], chart_uv[triangle[2]]
    ) < 0.0:
        return triangle[0], triangle[2], triangle[1]
    return triangle


def _ear_clip_cycle(cycle: np.ndarray, chart_uv: np.ndarray) -> list[tuple[int, int, int]]:
    nodes = [int(value) for value in cycle]
    if _signed_polygon_area(chart_uv[nodes]) < 0.0:
        nodes.reverse()
    triangles: list[tuple[int, int, int]] = []
    while len(nodes) > 3:
        for position, middle in enumerate(nodes):
            first = nodes[position - 1]
            third = nodes[(position + 1) % len(nodes)]
            if _cross_2d(
                chart_uv[first], chart_uv[middle], chart_uv[third]
            ) <= 1.0e-10:
                continue
            blocked = any(
                _point_in_or_on_triangle(
                    chart_uv[other],
                    chart_uv[first],
                    chart_uv[middle],
                    chart_uv[third],
                    tolerance=1.0e-9,
                )
                for other in nodes
                if other not in {first, middle, third}
            )
            if blocked:
                continue
            triangles.append((first, middle, third))
            nodes.pop(position)
            break
        else:
            raise ValueError("simple boundary cycle cannot be ear triangulated")
    triangles.append(tuple(nodes))
    return triangles


def _point_segment_distance(
    point: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    edge = second - first
    parameter = float(np.dot(point - first, edge)) / max(float(np.dot(edge, edge)), 1.0e-12)
    projected = first + np.clip(parameter, 0.0, 1.0) * edge
    return float(np.linalg.norm(point - projected))


def _insert_chart_point(
    triangles: list[tuple[int, int, int]],
    chart_uv: np.ndarray,
    point_index: int,
) -> list[tuple[int, int, int]]:
    point = chart_uv[point_index]
    edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, first in enumerate(triangle):
            second = triangle[(edge_index + 1) % 3]
            edge_triangle[(min(first, second), max(first, second))].append(triangle_index)
    for (first, second), incident in edge_triangle.items():
        edge_length = max(float(np.linalg.norm(chart_uv[second] - chart_uv[first])), 1.0)
        if (
            abs(_cross_2d(chart_uv[first], chart_uv[second], point))
            <= 1.0e-9 * edge_length
            and float(np.dot(point - chart_uv[first], point - chart_uv[second]))
            <= 1.0e-9
        ):
            replacements: list[tuple[int, int, int]] = []
            for triangle_index in incident:
                third = next(
                    value
                    for value in triangles[triangle_index]
                    if value not in {first, second}
                )
                replacements.extend(
                    (
                        _orient_chart_triangle(
                            (first, point_index, third), chart_uv
                        ),
                        _orient_chart_triangle(
                            (point_index, second, third), chart_uv
                        ),
                    )
                )
            removed = set(incident)
            return [
                triangle
                for triangle_index, triangle in enumerate(triangles)
                if triangle_index not in removed
            ] + replacements
    for triangle_index, triangle in enumerate(triangles):
        if not _point_in_or_on_triangle(
            point,
            chart_uv[triangle[0]],
            chart_uv[triangle[1]],
            chart_uv[triangle[2]],
            tolerance=1.0e-7,
        ):
            continue
        first, second, third = triangle
        replacements = [
            _orient_chart_triangle((first, second, point_index), chart_uv),
            _orient_chart_triangle((second, third, point_index), chart_uv),
            _orient_chart_triangle((third, first, point_index), chart_uv),
        ]
        return triangles[:triangle_index] + triangles[triangle_index + 1 :] + replacements
    raise ValueError("dense field point falls outside its assigned boundary cycle")


def _minimum_triangle_angle(
    triangle: tuple[int, int, int], chart_uv: np.ndarray
) -> float:
    points = chart_uv[np.asarray(triangle, dtype=np.int32)]
    edge = np.asarray(
        [
            np.linalg.norm(points[1] - points[0]),
            np.linalg.norm(points[2] - points[1]),
            np.linalg.norm(points[0] - points[2]),
        ],
        dtype=np.float64,
    )
    if float(np.min(edge)) <= 1.0e-10:
        return 0.0
    first, second, third = (float(value) for value in edge)
    cosine = (
        (first * first + third * third - second * second) / (2.0 * first * third),
        (first * first + second * second - third * third) / (2.0 * first * second),
        (second * second + third * third - first * first) / (2.0 * second * third),
    )
    return min(math.acos(np.clip(value, -1.0, 1.0)) for value in cosine)


def _improve_chart_triangulation(
    triangles: list[tuple[int, int, int]],
    chart_uv: np.ndarray,
    constrained_edges: set[tuple[int, int]],
    *,
    maximum_iterations: int,
) -> tuple[list[tuple[int, int, int]], int]:
    """Lawson flips improve element quality without changing the boundary."""

    for iteration in range(maximum_iterations):
        edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
        for triangle_index, triangle in enumerate(triangles):
            for edge_index, first in enumerate(triangle):
                second = triangle[(edge_index + 1) % 3]
                edge_triangle[(min(first, second), max(first, second))].append(
                    triangle_index
                )
        changed = False
        for edge, incident in sorted(edge_triangle.items()):
            if len(incident) != 2 or edge in constrained_edges:
                continue
            first, second = edge
            left_index, right_index = incident
            left_other = next(
                value
                for value in triangles[left_index]
                if value not in {first, second}
            )
            right_other = next(
                value
                for value in triangles[right_index]
                if value not in {first, second}
            )
            replacement_edge = (min(left_other, right_other), max(left_other, right_other))
            if replacement_edge in edge_triangle:
                continue
            if (
                _cross_2d(
                    chart_uv[first], chart_uv[second], chart_uv[left_other]
                )
                * _cross_2d(
                    chart_uv[first], chart_uv[second], chart_uv[right_other]
                )
                >= -1.0e-10
                or _cross_2d(
                    chart_uv[left_other], chart_uv[right_other], chart_uv[first]
                )
                * _cross_2d(
                    chart_uv[left_other], chart_uv[right_other], chart_uv[second]
                )
                >= -1.0e-10
            ):
                continue
            replacement = (
                _orient_chart_triangle(
                    (left_other, right_other, first), chart_uv
                ),
                _orient_chart_triangle(
                    (right_other, left_other, second), chart_uv
                ),
            )
            old_quality = min(
                _minimum_triangle_angle(triangles[left_index], chart_uv),
                _minimum_triangle_angle(triangles[right_index], chart_uv),
            )
            new_quality = min(
                _minimum_triangle_angle(replacement[0], chart_uv),
                _minimum_triangle_angle(replacement[1], chart_uv),
            )
            if new_quality <= old_quality + 1.0e-9:
                continue
            triangles[left_index], triangles[right_index] = replacement
            changed = True
            break
        if not changed:
            return triangles, iteration
    raise ValueError("constrained chart edge flips did not converge")


def _improve_physical_triangulation(
    triangles: np.ndarray,
    chart_uv: np.ndarray,
    midpoint_xyz: np.ndarray,
    reference_normal_xyz: np.ndarray,
    *,
    maximum_iterations: int,
) -> tuple[np.ndarray, int]:
    """Choose interior diagonals from the complete realized surface.

    Intrinsic Delaunay quality alone can select the wrong diagonal across a
    tightly folded three-dimensional quadrilateral.  This second Lawson pass
    leaves the exact chart boundary untouched while minimizing the
    area-weighted axial disagreement between realized triangle normals and the
    complete fitted normal field.  Native CT and collision gates still decide
    whether the resulting whole patch is physically admissible.
    """

    values = [tuple(int(value) for value in row) for row in np.asarray(triangles)]
    uv = np.asarray(chart_uv, dtype=np.float64)

    def pair_penalty(pair: tuple[tuple[int, int, int], ...]) -> float:
        penalty = 0.0
        for triangle in pair:
            _, area, residual, _ = _triangle_geometry(
                triangle, midpoint_xyz, reference_normal_xyz
            )
            # A fourth-power tail makes one layer-crossing triangle more
            # expensive than several small, noisy disagreements.  The audit
            # below uses the same high-residual-area interpretation.
            penalty += area * (residual / 45.0) ** 4
        return penalty

    for iteration in range(maximum_iterations):
        edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
        for triangle_index, triangle in enumerate(values):
            for edge_index, first in enumerate(triangle):
                second = triangle[(edge_index + 1) % 3]
                edge_triangle[(min(first, second), max(first, second))].append(
                    triangle_index
                )
        constrained = {
            edge for edge, incident in edge_triangle.items() if len(incident) == 1
        }
        changed = False
        for edge, incident in sorted(edge_triangle.items()):
            if len(incident) != 2 or edge in constrained:
                continue
            first, second = edge
            left_index, right_index = incident
            left_other = next(
                value
                for value in values[left_index]
                if value not in {first, second}
            )
            right_other = next(
                value
                for value in values[right_index]
                if value not in {first, second}
            )
            replacement_edge = (
                min(left_other, right_other),
                max(left_other, right_other),
            )
            if replacement_edge in edge_triangle:
                continue
            if (
                _cross_2d(uv[first], uv[second], uv[left_other])
                * _cross_2d(uv[first], uv[second], uv[right_other])
                >= -1.0e-10
                or _cross_2d(uv[left_other], uv[right_other], uv[first])
                * _cross_2d(uv[left_other], uv[right_other], uv[second])
                >= -1.0e-10
            ):
                continue
            replacement = (
                _orient_chart_triangle((left_other, right_other, first), uv),
                _orient_chart_triangle((right_other, left_other, second), uv),
            )
            old_pair = (values[left_index], values[right_index])
            old_quality = min(
                _minimum_triangle_angle(old_pair[0], uv),
                _minimum_triangle_angle(old_pair[1], uv),
            )
            new_quality = min(
                _minimum_triangle_angle(replacement[0], uv),
                _minimum_triangle_angle(replacement[1], uv),
            )
            # Never trade the normal fit for a nearly collapsed chart element.
            if new_quality < math.radians(2.0) or new_quality < 0.35 * old_quality:
                continue
            old_penalty = pair_penalty(old_pair)
            new_penalty = pair_penalty(replacement)
            if new_penalty >= old_penalty - 1.0e-9:
                continue
            values[left_index], values[right_index] = replacement
            changed = True
            break
        if not changed:
            return np.asarray(values, dtype=np.int32), iteration
    raise ValueError("physical completion edge flips did not converge")


def triangulate_weak_boundary_field(
    boundary: np.ndarray,
    boundary_uv: np.ndarray,
    field_uv: np.ndarray,
    *,
    minimum_boundary_separation: float,
    maximum_edge_flip_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Triangulate all positive cycles of a weakly simple boundary.

    Returned triangle indices address ``pointUV``.  Boundary points precede
    retained field points, and ``pointSourceIndex`` maps them back to the
    original surface node or depth-field pixel respectively.
    """

    boundary = np.asarray(boundary, dtype=np.int32)
    boundary_uv = np.asarray(boundary_uv, dtype=np.float64)
    field_uv = np.asarray(field_uv, dtype=np.float64)
    if len(boundary) != len(boundary_uv):
        raise ValueError("boundary nodes and chart coordinates differ")
    cycles_global = decompose_weak_boundary_cycles(boundary)
    boundary_lookup = {int(value): row for row, value in enumerate(boundary)}
    unique_boundary = np.asarray(
        list(dict.fromkeys(int(value) for value in boundary)), dtype=np.int32
    )
    unique_uv = np.asarray(
        [boundary_uv[boundary_lookup[int(value)]] for value in unique_boundary],
        dtype=np.float64,
    )
    local_boundary = {
        int(value): index for index, value in enumerate(unique_boundary)
    }
    cycles = tuple(
        np.asarray([local_boundary[int(value)] for value in cycle], dtype=np.int32)
        for cycle in cycles_global
    )
    cycle_area = np.asarray(
        [_signed_polygon_area(unique_uv[cycle]) for cycle in cycles],
        dtype=np.float64,
    )
    dominant_sign = 1.0 if float(np.sum(cycle_area)) >= 0.0 else -1.0
    if np.any(dominant_sign * cycle_area <= 1.0e-8):
        raise ValueError(
            "weak boundary contains an opposite-orientation exclusion cycle"
        )

    boundary_distance = np.asarray(
        [
            min(
                _point_segment_distance(
                    point, boundary_uv[index], boundary_uv[(index + 1) % len(boundary_uv)]
                )
                for index in range(len(boundary_uv))
            )
            for point in field_uv
        ],
        dtype=np.float64,
    )
    retained_field = np.flatnonzero(
        boundary_distance >= minimum_boundary_separation
    ).astype(np.int32)
    point_uv = np.vstack((unique_uv, field_uv[retained_field]))
    field_local = {
        int(pixel): len(unique_boundary) + row
        for row, pixel in enumerate(retained_field)
    }
    cycle_field: list[list[int]] = [[] for _ in cycles]
    for pixel in retained_field:
        membership = [
            cycle_index
            for cycle_index, cycle in enumerate(cycles)
            if _point_in_polygon(field_uv[pixel], unique_uv[cycle])
        ]
        if len(membership) != 1:
            raise ValueError(
                "retained dense-field point does not belong to exactly one boundary disk"
            )
        cycle_field[membership[0]].append(int(pixel))

    triangles: list[tuple[int, int, int]] = []
    flip_count = 0
    for cycle, pixels in zip(cycles, cycle_field):
        local_triangles = _ear_clip_cycle(cycle, point_uv)
        insertion_order = sorted(
            pixels,
            key=lambda pixel: (
                float(field_uv[pixel, 0]),
                float(field_uv[pixel, 1]),
                pixel,
            ),
        )
        for pixel in insertion_order:
            local_triangles = _insert_chart_point(
                local_triangles, point_uv, field_local[pixel]
            )
        constrained = {
            tuple(sorted((int(cycle[index]), int(cycle[(index + 1) % len(cycle)]))))
            for index in range(len(cycle))
        }
        local_triangles, flips = _improve_chart_triangulation(
            local_triangles,
            point_uv,
            constrained,
            maximum_iterations=maximum_edge_flip_iterations,
        )
        flip_count += flips
        triangles.extend(local_triangles)

    triangle = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    edge_count: Counter[tuple[int, int]] = Counter()
    for values in triangle:
        edge_count.update(
            tuple(sorted((int(values[index]), int(values[(index + 1) % 3]))))
            for index in range(3)
        )
    expected_boundary = Counter()
    for cycle in cycles:
        expected_boundary.update(
            tuple(sorted((int(cycle[index]), int(cycle[(index + 1) % len(cycle)]))))
            for index in range(len(cycle))
        )
    actual_boundary = Counter(edge for edge, count in edge_count.items() if count == 1)
    if actual_boundary != expected_boundary:
        raise ValueError("constrained field mesh does not preserve its exact boundary")
    if any(count > 2 for count in edge_count.values()):
        raise ValueError("constrained field mesh contains a non-manifold edge")
    point_source = np.concatenate((unique_boundary, retained_field)).astype(np.int32)
    point_kind = np.concatenate(
        (
            np.zeros(len(unique_boundary), dtype=np.uint8),
            np.ones(len(retained_field), dtype=np.uint8),
        )
    )
    return {
        "pointUV": point_uv.astype(np.float32),
        "pointKind": point_kind,
        "pointSourceIndex": point_source,
        "trianglePointIndex": triangle,
        "retainedFieldPixel": retained_field,
        "fieldBoundaryDistanceVoxels": boundary_distance.astype(np.float32),
    }, {
        "boundaryWalkVertexCount": int(len(boundary)),
        "uniqueBoundaryVertexCount": int(len(unique_boundary)),
        "pinchCycleCount": int(len(cycles)),
        "pinchVertexReuseCount": int(len(boundary) - len(unique_boundary)),
        "fieldPixelCount": int(len(field_uv)),
        "retainedFieldPixelCount": int(len(retained_field)),
        "retainedFieldPixelFraction": round(
            float(len(retained_field)) / max(len(field_uv), 1), 6
        ),
        "triangleCount": int(len(triangle)),
        "chartEdgeFlipIterations": int(flip_count),
        "exactBoundaryPreserved": True,
    }


def _edge_incidence(triangles: np.ndarray) -> Counter[tuple[int, int]]:
    result: Counter[tuple[int, int]] = Counter()
    for triangle in np.asarray(triangles, dtype=np.int32):
        result.update(
            tuple(
                sorted(
                    (
                        int(triangle[index]),
                        int(triangle[(index + 1) % 3]),
                    )
                )
            )
            for index in range(3)
        )
    return result


def _triangle_region_labels(triangles: np.ndarray) -> np.ndarray:
    triangles = np.asarray(triangles, dtype=np.int32)
    parent = np.arange(len(triangles), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    edge_triangle: dict[tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            previous = edge_triangle.get(edge)
            if previous is None:
                edge_triangle[edge] = triangle_index
                continue
            first_root, second_root = find(previous), find(triangle_index)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)
    root = np.asarray([find(index) for index in range(len(triangles))], dtype=np.int32)
    _, labels = np.unique(root, return_inverse=True)
    return labels.astype(np.int32)


def _component_region_count(surface: Mapping[str, np.ndarray]) -> dict[int, int]:
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    if not len(triangle):
        return {}
    component = np.asarray(surface["component"], dtype=np.int32)
    region = _triangle_region_labels(triangle)
    result: dict[int, set[int]] = defaultdict(set)
    for triangle_index, values in enumerate(triangle):
        components = np.unique(component[values])
        if len(components) == 1:
            result[int(components[0])].add(int(region[triangle_index]))
    return {key: len(value) for key, value in result.items()}


def _strict_point_in_chart_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    signs = np.asarray(
        [
            _cross_2d(triangle[index], triangle[(index + 1) % 3], point)
            for index in range(3)
        ]
    )
    return bool(np.min(signs) > 1.0e-7 or np.max(signs) < -1.0e-7)


def _proper_segments_intersect(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
) -> bool:
    first_side = (
        _cross_2d(first_start, first_stop, second_start),
        _cross_2d(first_start, first_stop, second_stop),
    )
    second_side = (
        _cross_2d(second_start, second_stop, first_start),
        _cross_2d(second_start, second_stop, first_stop),
    )
    return (
        first_side[0] * first_side[1] < -1.0e-12
        and second_side[0] * second_side[1] < -1.0e-12
    )


def _chart_overlap_count(
    baseline_triangle: np.ndarray,
    patch_triangle: np.ndarray,
    chart_uv: np.ndarray,
) -> int:
    baseline_points = chart_uv[np.asarray(baseline_triangle, dtype=np.int32)]
    patch_points = chart_uv[np.asarray(patch_triangle, dtype=np.int32)]
    baseline_low = np.min(baseline_points, axis=1)
    baseline_high = np.max(baseline_points, axis=1)
    count = 0
    for patch_index, patch in enumerate(patch_points):
        low, high = np.min(patch, axis=0), np.max(patch, axis=0)
        possible = np.flatnonzero(
            np.all(baseline_high >= low - 1.0e-7, axis=1)
            & np.all(baseline_low <= high + 1.0e-7, axis=1)
        )
        patch_nodes = set(int(value) for value in patch_triangle[patch_index])
        for baseline_index in possible:
            baseline = baseline_points[baseline_index]
            baseline_nodes = set(int(value) for value in baseline_triangle[baseline_index])
            if any(
                _proper_segments_intersect(
                    patch[first],
                    patch[(first + 1) % 3],
                    baseline[second],
                    baseline[(second + 1) % 3],
                )
                for first in range(3)
                for second in range(3)
            ):
                count += 1
                break
            if any(
                int(node) not in baseline_nodes
                and _strict_point_in_chart_triangle(chart_uv[node], baseline)
                for node in patch_triangle[patch_index]
            ) or any(
                int(node) not in patch_nodes
                and _strict_point_in_chart_triangle(chart_uv[node], patch)
                for node in baseline_triangle[baseline_index]
            ):
                count += 1
                break
    return count


def _mesh_vertex_normals(
    triangles: np.ndarray,
    midpoint_xyz: np.ndarray,
    reference_normal_xyz: np.ndarray,
) -> np.ndarray:
    result = np.zeros_like(midpoint_xyz, dtype=np.float64)
    for triangle in np.asarray(triangles, dtype=np.int32):
        points = midpoint_xyz[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-12:
            continue
        normal /= length
        reference = np.sum(reference_normal_xyz[triangle], axis=0)
        if float(np.dot(normal, reference)) < 0.0:
            normal *= -1.0
        for vertex in triangle:
            result[int(vertex)] += normal
    length = np.linalg.norm(result, axis=1, keepdims=True)
    unresolved = length[:, 0] <= 1.0e-10
    result[~unresolved] /= length[~unresolved]
    result[unresolved] = reference_normal_xyz[unresolved]
    result[
        np.einsum("ij,ij->i", result, reference_normal_xyz) < 0.0
    ] *= -1.0
    return result.astype(np.float32)


def _normal_frame(normal_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    normal = np.asarray(normal_xyz, dtype=np.float64)
    tangent_u = np.empty_like(normal)
    for row, value in enumerate(normal):
        axis = np.eye(3)[int(np.argmin(np.abs(value)))]
        tangent_u[row] = np.cross(value, axis)
    tangent_u /= np.maximum(np.linalg.norm(tangent_u, axis=1, keepdims=True), 1.0e-12)
    tangent_v = np.cross(normal, tangent_u)
    tangent_v /= np.maximum(np.linalg.norm(tangent_v, axis=1, keepdims=True), 1.0e-12)
    return tangent_u.astype(np.float32), tangent_v.astype(np.float32)


def _other_component_clearance(
    point_xyz: np.ndarray,
    surface: Mapping[str, np.ndarray],
    component_id: int,
    thickness: float,
) -> dict[str, float | int]:
    selected = np.asarray(surface["selected"], dtype=np.uint8) > 0
    component = np.asarray(surface["component"], dtype=np.int32)
    other = selected & (component >= 0) & (component != component_id)
    other_xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)[other]
    if not len(other_xyz) or not len(point_xyz):
        return {
            "otherComponentNodeCount": int(len(other_xyz)),
            "minimumOtherComponentClearanceVoxels": float("inf"),
            "minimumOtherComponentClearanceThicknesses": float("inf"),
        }
    minimum = np.full(len(point_xyz), np.inf, dtype=np.float32)
    for start in range(0, len(other_xyz), 4096):
        distance = np.linalg.norm(
            point_xyz[:, None, :] - other_xyz[None, start : start + 4096, :],
            axis=2,
        )
        minimum = np.minimum(minimum, np.min(distance, axis=1))
    value = float(np.min(minimum))
    return {
        "otherComponentNodeCount": int(len(other_xyz)),
        "minimumOtherComponentClearanceVoxels": round(value, 6),
        "minimumOtherComponentClearanceThicknesses": round(
            value / max(thickness, 1.0e-6), 6
        ),
    }


def _other_component_triangle_intersections(
    baseline_surface: Mapping[str, np.ndarray],
    augmented_surface: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    component_id: int,
    *,
    tolerance: float,
) -> dict[str, int]:
    """Audit literal crossings against every other already-selected surface."""

    from ..slab_association_integrity import _triangle_intersection

    baseline_triangle = np.asarray(
        baseline_surface["triangleFrontierIndex"], dtype=np.int32
    )
    component = np.asarray(baseline_surface["component"], dtype=np.int32)
    other = baseline_triangle[
        ~np.all(component[baseline_triangle] == component_id, axis=1)
    ]
    if not len(other) or not len(patch_triangle):
        return {
            "otherComponentTriangleCount": int(len(other)),
            "broadPhaseTrianglePairCount": 0,
            "intersectingTrianglePairCount": 0,
        }
    xyz = np.asarray(augmented_surface["midpointXYZ"], dtype=np.float32)
    other_point = xyz[other]
    other_low = np.min(other_point, axis=1) - tolerance
    other_high = np.max(other_point, axis=1) + tolerance
    broad_count = 0
    intersection_count = 0
    for triangle in np.asarray(patch_triangle, dtype=np.int32):
        point = xyz[triangle]
        low = np.min(point, axis=0) - tolerance
        high = np.max(point, axis=0) + tolerance
        possible = np.flatnonzero(
            np.all(other_high >= low, axis=1) & np.all(other_low <= high, axis=1)
        )
        broad_count += len(possible)
        for other_index in possible:
            intersection, _ = _triangle_intersection(
                point, other_point[int(other_index)], tolerance
            )
            if intersection is not None:
                intersection_count += 1
    return {
        "otherComponentTriangleCount": int(len(other)),
        "broadPhaseTrianglePairCount": int(broad_count),
        "intersectingTrianglePairCount": int(intersection_count),
    }


def _completion_native_ct_audit(
    source: VolumeSource,
    volume: np.ndarray,
    point_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    thickness: float,
    context_profile: np.ndarray,
    context_physical_score: float,
    intensity_scale: float,
    *,
    settings: PhysicalRibbonDenseCompletionSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    depth = np.asarray(settings.profile_depth_fractions, dtype=np.float32)
    shifts = np.asarray(settings.competing_shift_thicknesses, dtype=np.float32)
    profiles = _sample_normal_profiles(
        source,
        volume,
        point_xyz,
        normal_xyz,
        np.full(len(point_xyz), thickness, dtype=np.float32),
        depth,
        shifts,
    ).transpose(1, 0, 2)
    physical, correlation = _profile_fields(
        profiles,
        np.asarray(context_profile, dtype=np.float32),
        depth,
        intensity_scale,
    )
    score = physical + 0.35 * correlation
    zero = int(np.flatnonzero(shifts == 0.0)[0])
    chosen_physical = physical[:, zero]
    chosen_correlation = correlation[:, zero]
    competitor = np.abs(shifts) >= 0.5
    far_margin = score[:, zero] - np.nanmax(score[:, competitor], axis=1)
    supported = (
        np.isfinite(chosen_physical)
        & np.isfinite(chosen_correlation)
        & (
            chosen_physical
            >= 0.50 * max(float(context_physical_score), 0.10)
        )
        & (chosen_correlation >= 0.35)
    )
    return {
        "normalProfile": profiles.astype(np.float32),
        "physicalScore": chosen_physical.astype(np.float32),
        "profileCorrelation": chosen_correlation.astype(np.float32),
        "farLayerMargin": far_margin.astype(np.float32),
        "ctSupported": supported.astype(np.uint8),
    }, {
        "sampleCount": int(len(point_xyz)),
        "supportedFraction": round(float(np.mean(supported)), 6) if len(supported) else 0.0,
        "medianPhysicalScore": round(float(np.nanmedian(chosen_physical)), 6),
        "medianProfileCorrelation": round(float(np.nanmedian(chosen_correlation)), 6),
        "minimumProfileCorrelation": round(float(np.nanmin(chosen_correlation)), 6),
        "medianFarLayerMargin": round(float(np.nanmedian(far_margin)), 6),
        "minimumFarLayerMargin": round(float(np.nanmin(far_margin)), 6),
    }


def _append_completion_surface(
    surface: Mapping[str, np.ndarray],
    mesh: Mapping[str, np.ndarray],
    field_xyz: np.ndarray,
    field_reference_normal: np.ndarray,
    component_id: int,
    thickness: float,
    *,
    triangle_area: np.ndarray,
    triangle_normal_residual: np.ndarray,
    existing_surface_edges: AbstractSet[tuple[int, int]],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    point_kind = np.asarray(mesh["pointKind"], dtype=np.uint8)
    point_source = np.asarray(mesh["pointSourceIndex"], dtype=np.int32)
    synthetic_local = np.flatnonzero(point_kind == 1)
    synthetic_pixel = point_source[synthetic_local]
    node_offset = len(np.asarray(surface["midpointXYZ"]))
    synthetic_global = node_offset + np.arange(len(synthetic_local), dtype=np.int32)
    local_to_global = np.empty(len(point_kind), dtype=np.int32)
    boundary_local = np.flatnonzero(point_kind == 0)
    local_to_global[boundary_local] = point_source[boundary_local]
    local_to_global[synthetic_local] = synthetic_global
    patch_triangle = local_to_global[
        np.asarray(mesh["trianglePointIndex"], dtype=np.int32)
    ]

    local_xyz = np.empty((len(point_kind), 3), dtype=np.float32)
    local_reference = np.empty((len(point_kind), 3), dtype=np.float32)
    local_xyz[boundary_local] = np.asarray(surface["midpointXYZ"])[
        point_source[boundary_local]
    ]
    local_reference[boundary_local] = np.asarray(surface["signedNormalXYZ"])[
        point_source[boundary_local]
    ]
    local_xyz[synthetic_local] = field_xyz[synthetic_pixel]
    local_reference[synthetic_local] = field_reference_normal[synthetic_pixel]
    local_normal = _mesh_vertex_normals(
        np.asarray(mesh["trianglePointIndex"], dtype=np.int32),
        local_xyz,
        local_reference,
    )
    tangent_u, tangent_v = _normal_frame(local_normal[synthetic_local])

    # Every mutated array is replaced below, so the invariant arrays can be
    # shared with the rejected trial.  In particular, the continuation graph
    # can contain millions of non-surface edges and must not be copied once per
    # tiny completion.
    result = {key: np.asarray(value) for key, value in surface.items()}
    result["selected"] = np.concatenate(
        (
            np.asarray(surface["selected"], dtype=np.uint8),
            np.ones(len(synthetic_global), dtype=np.uint8),
        )
    )
    result["component"] = np.concatenate(
        (
            np.asarray(surface["component"], dtype=np.int32),
            np.full(len(synthetic_global), component_id, dtype=np.int32),
        )
    )
    result["signedNormalXYZ"] = np.vstack(
        (
            np.asarray(surface["signedNormalXYZ"], dtype=np.float32),
            local_normal[synthetic_local],
        )
    )
    result["tangentUxyz"] = np.vstack(
        (np.asarray(surface["tangentUxyz"], dtype=np.float32), tangent_u)
    )
    result["tangentVxyz"] = np.vstack(
        (np.asarray(surface["tangentVxyz"], dtype=np.float32), tangent_v)
    )
    result["chartUV"] = np.vstack(
        (
            np.asarray(surface["chartUV"], dtype=np.float32),
            np.asarray(mesh["pointUV"], dtype=np.float32)[synthetic_local],
        )
    )
    result["integrationResidualVoxels"] = np.concatenate(
        (
            np.asarray(surface["integrationResidualVoxels"], dtype=np.float32),
            np.zeros(len(synthetic_global), dtype=np.float32),
        )
    )
    result["midpointXYZ"] = np.vstack(
        (np.asarray(surface["midpointXYZ"], dtype=np.float32), local_xyz[synthetic_local])
    )
    result["thicknessVoxels"] = np.concatenate(
        (
            np.asarray(surface["thicknessVoxels"], dtype=np.float32),
            np.full(len(synthetic_global), thickness, dtype=np.float32),
        )
    )
    result["triangleFrontierIndex"] = np.vstack(
        (np.asarray(surface["triangleFrontierIndex"], dtype=np.int32), patch_triangle)
    )
    result["triangleAreaVoxelsSquared"] = np.concatenate(
        (
            np.asarray(surface["triangleAreaVoxelsSquared"], dtype=np.float32),
            np.asarray(triangle_area, dtype=np.float32),
        )
    )
    result["triangleNormalResidualDegrees"] = np.concatenate(
        (
            np.asarray(surface["triangleNormalResidualDegrees"], dtype=np.float32),
            np.asarray(triangle_normal_residual, dtype=np.float32),
        )
    )

    patch_edge = sorted(_edge_incidence(patch_triangle))
    added_edge = np.asarray(
        [edge for edge in patch_edge if edge not in existing_surface_edges],
        dtype=np.int32,
    ).reshape((-1, 2))
    # Edge topology and component sizes are accumulated once after all exact
    # completion decisions.  No decision below consults these denormalized
    # arrays; surface incidence comes directly from the triangles.
    return result, synthetic_global, patch_triangle, added_edge


def _loop_counts(loops: Mapping[str, np.ndarray]) -> dict[str, int]:
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    macro = np.asarray(loops["loopMacroEligible"], dtype=np.uint8) > 0
    return {
        "interiorHoleCount": int(np.count_nonzero(kind == 1)),
        "macroHoleCount": int(np.count_nonzero(macro)),
    }


def build_physical_ribbon_dense_completion(
    holes: Mapping[str, np.ndarray],
    depth_field: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    hole_settings: PhysicalRibbonPatchHoleSettings,
    settings: PhysicalRibbonDenseCompletionSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    current = _surface_view(holes)
    baseline_node_count = len(np.asarray(current["midpointXYZ"]))
    baseline_triangle_count = len(np.asarray(current["triangleFrontierIndex"]))
    loops = {
        key: np.asarray(holes[key])
        for key in (
            "loopOffset",
            "loopVertexFrontierIndex",
            "loopTriangleRegion",
            "loopTopologyComponent",
            "loopKind",
            "loopAreaChartVoxelsSquared",
            "loopPerimeterChartVoxels",
            "loopDiameterChartVoxels",
            "loopMedianThicknessVoxels",
            "loopMeanBoundaryEdgeVoxels",
            "loopMacroEligible",
        )
    }
    loop_count_before = _loop_counts(loops)
    _, initial_loop_stats = extract_surface_boundary_loops(
        current, settings=hole_settings
    )
    current_incidence = _edge_incidence(current["triangleFrontierIndex"])
    current_region_count = _component_region_count(current)
    predicted_loop_count = dict(loop_count_before)
    volume = source.memmap()
    patch_offset = np.asarray(depth_field["holePatchOffset"], dtype=np.int64)
    loop_index = np.asarray(depth_field["holeLoopIndex"], dtype=np.int32)
    shifts = np.asarray(depth_field["shiftThicknesses"], dtype=np.float32)
    labels = np.asarray(depth_field["pixelCollectiveLabel"], dtype=np.int16)
    field_supported = np.asarray(depth_field["pixelCtSupported"], dtype=np.uint8) > 0

    proposal_offset = [0]
    proposal_node: list[int] = []
    proposal_accepted: list[int] = []
    completion_triangle_offset = [0]
    completion_triangle: list[int] = []
    completion_field_offset = [0]
    completion_field_pixel: list[int] = []
    completion_profile: list[np.ndarray] = []
    completion_physical: list[float] = []
    completion_correlation: list[float] = []
    completion_margin: list[float] = []
    completion_supported: list[int] = []
    accepted_added_edge: list[tuple[int, int]] = []
    records: list[dict[str, Any]] = []

    ranked_rows = sorted(
        range(len(loop_index)),
        key=lambda row: (
            -float(holes["loopAreaChartVoxelsSquared"][loop_index[row]]),
            row,
        ),
    )[: settings.maximum_completed_holes]
    for row in ranked_rows:
        loop = int(loop_index[row])
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        pixel_slice = slice(start, stop)
        boundary_offset = np.asarray(holes["loopOffset"], dtype=np.int64)
        boundary = np.asarray(holes["loopVertexFrontierIndex"], dtype=np.int32)[
            int(boundary_offset[loop]) : int(boundary_offset[loop + 1])
        ]
        component_id = int(holes["loopTopologyComponent"][loop])
        thickness = float(holes["loopMedianThicknessVoxels"][loop])
        field_uv = np.asarray(depth_field["patchUV"], dtype=np.float32)[pixel_slice]
        field_reference = np.asarray(
            depth_field["patchNormalXYZ"], dtype=np.float32
        )[pixel_slice]
        chosen_shift = shifts[labels[pixel_slice]]
        field_xyz = np.asarray(depth_field["patchXYZ"], dtype=np.float32)[pixel_slice] + (
            field_reference * (chosen_shift * thickness)[:, None]
        )
        reasons: list[str] = []
        depth_support_fraction = (
            float(np.mean(field_supported[pixel_slice])) if stop > start else 0.0
        )
        if depth_support_fraction < settings.minimum_depth_field_supported_fraction:
            reasons.append("insufficient collective depth-field CT support")
        try:
            mesh, mesh_stats = triangulate_weak_boundary_field(
                boundary,
                np.asarray(current["chartUV"], dtype=np.float32)[boundary],
                field_uv,
                minimum_boundary_separation=(
                    settings.minimum_interior_boundary_separation_voxels
                ),
                maximum_edge_flip_iterations=settings.maximum_edge_flip_iterations,
            )
        except ValueError as error:
            records.append(
                {
                    "holeRow": row,
                    "loopIndex": loop,
                    "component": component_id,
                    "accepted": False,
                    "rejectionReasons": [*reasons, str(error)],
                }
            )
            proposal_offset.append(proposal_offset[-1])
            completion_triangle_offset.append(completion_triangle_offset[-1])
            completion_field_offset.append(completion_field_offset[-1])
            proposal_accepted.append(0)
            continue
        retained_pixel = np.asarray(mesh["retainedFieldPixel"], dtype=np.int32)
        # Raster centers within the boundary-separation band are intentionally
        # omitted to avoid skinny triangles.  Their *fraction* is a
        # perimeter-to-area artifact, especially for three-to-five-edge
        # holes, rather than missing physical support.  Sampling density is
        # instead bounded by triangle edge length and the realized triangles
        # are audited from native CT at both retained vertices and centroids.

        point_kind = np.asarray(mesh["pointKind"], dtype=np.uint8)
        point_source = np.asarray(mesh["pointSourceIndex"], dtype=np.int32)
        boundary_local = np.flatnonzero(point_kind == 0)
        field_local = np.flatnonzero(point_kind == 1)
        local_xyz = np.empty((len(point_kind), 3), dtype=np.float32)
        local_reference = np.empty((len(point_kind), 3), dtype=np.float32)
        local_xyz[boundary_local] = np.asarray(current["midpointXYZ"])[
            point_source[boundary_local]
        ]
        local_reference[boundary_local] = np.asarray(current["signedNormalXYZ"])[
            point_source[boundary_local]
        ]
        local_xyz[field_local] = field_xyz[point_source[field_local]]
        local_reference[field_local] = field_reference[point_source[field_local]]
        local_triangle = np.asarray(mesh["trianglePointIndex"], dtype=np.int32)
        local_triangle, physical_flips = _improve_physical_triangulation(
            local_triangle,
            np.asarray(mesh["pointUV"], dtype=np.float32),
            local_xyz,
            local_reference,
            maximum_iterations=settings.maximum_edge_flip_iterations,
        )
        mesh_stats = {
            **mesh_stats,
            "physicalEdgeFlipIterations": int(physical_flips),
        }
        oriented: list[tuple[int, int, int]] = []
        triangle_area: list[float] = []
        triangle_residual: list[float] = []
        triangle_edge: list[float] = []
        for triangle in local_triangle:
            values, area, residual, maximum_edge = _triangle_geometry(
                tuple(int(value) for value in triangle),
                local_xyz,
                local_reference,
            )
            oriented.append(values)
            triangle_area.append(area)
            triangle_residual.append(residual)
            triangle_edge.append(maximum_edge)
        mesh = dict(mesh)
        mesh["trianglePointIndex"] = np.asarray(oriented, dtype=np.int32)
        if min(triangle_area, default=0.0) < settings.minimum_triangle_area_voxels_squared:
            reasons.append("completion contains a physically degenerate triangle")
        if max(triangle_edge, default=float("inf")) > settings.maximum_triangle_edge_voxels:
            reasons.append("completion contains an overlong triangle edge")
        triangle_area_array = np.asarray(triangle_area, dtype=np.float64)
        triangle_residual_array = np.asarray(triangle_residual, dtype=np.float64)
        high_normal_area_fraction = float(
            np.sum(
                triangle_area_array[
                    triangle_residual_array
                    > settings.high_triangle_normal_residual_degrees
                ]
            )
            / max(float(np.sum(triangle_area_array)), 1.0e-12)
        )
        if (
            max(triangle_residual, default=float("inf"))
            > settings.maximum_triangle_normal_residual_degrees
        ):
            reasons.append("completion triangle contradicts the local CT normal field")

        local_normal = _mesh_vertex_normals(
            np.asarray(mesh["trianglePointIndex"], dtype=np.int32),
            local_xyz,
            local_reference,
        )
        triangle_point = local_xyz[
            np.asarray(mesh["trianglePointIndex"], dtype=np.int32)
        ]
        triangle_audit_xyz = np.mean(triangle_point, axis=1)
        triangle_audit_normal = np.cross(
            triangle_point[:, 1] - triangle_point[:, 0],
            triangle_point[:, 2] - triangle_point[:, 0],
        )
        triangle_audit_normal /= np.maximum(
            np.linalg.norm(triangle_audit_normal, axis=1, keepdims=True),
            1.0e-12,
        )
        audit_xyz = np.vstack(
            (local_xyz[field_local], triangle_audit_xyz)
        ).astype(np.float32)
        audit_normal = np.vstack(
            (local_normal[field_local], triangle_audit_normal)
        ).astype(np.float32)
        native_arrays, native_stats = _completion_native_ct_audit(
            source,
            volume,
            audit_xyz,
            audit_normal,
            thickness,
            np.asarray(holes["contextMedianProfile"], dtype=np.float32)[row],
            float(holes["contextPhysicalScore"][row]),
            float(holes["localIntensityScale"][row]),
            settings=settings,
        )
        native_stats["fieldVertexSampleCount"] = int(len(field_local))
        native_stats["triangleCentroidSampleCount"] = int(
            len(triangle_audit_xyz)
        )
        if (
            float(native_stats["supportedFraction"])
            < settings.minimum_reconstructed_supported_fraction
        ):
            reasons.append("constructed mesh normals lose whole-patch CT support")
        if (
            float(native_stats["medianProfileCorrelation"])
            < settings.minimum_median_profile_correlation
        ):
            reasons.append("constructed surface profile does not match its boundary context")
        if (
            float(native_stats["medianFarLayerMargin"])
            < settings.minimum_median_far_layer_margin
        ):
            reasons.append("a displaced competing layer explains the constructed surface")
        clearance = _other_component_clearance(
            audit_xyz, current, component_id, thickness
        )

        trial, synthetic_node, patch_triangle, added_edge = _append_completion_surface(
            current,
            mesh,
            field_xyz,
            field_reference,
            component_id,
            thickness,
            triangle_area=np.asarray(triangle_area, dtype=np.float32),
            triangle_normal_residual=np.asarray(triangle_residual, dtype=np.float32),
            existing_surface_edges=current_incidence.keys(),
        )
        base_incidence = current_incidence
        trial_incidence = base_incidence.copy()
        trial_incidence.update(_edge_incidence(patch_triangle))
        boundary_edges = {
            tuple(sorted((int(boundary[index]), int(boundary[(index + 1) % len(boundary)]))))
            for index in range(len(boundary))
        }
        boundary_before_once = all(base_incidence[edge] == 1 for edge in boundary_edges)
        boundary_after_twice = all(trial_incidence[edge] == 2 for edge in boundary_edges)
        nonmanifold = sum(value > 2 for value in trial_incidence.values())
        if not boundary_before_once:
            reasons.append("target loop is no longer an exact open boundary")
        if not boundary_after_twice:
            reasons.append("completion does not attach to every target boundary edge once")
        if nonmanifold:
            reasons.append("completion creates a non-manifold mesh edge")

        intersections = _other_component_triangle_intersections(
            current,
            trial,
            patch_triangle,
            component_id,
            tolerance=settings.intersection_tolerance_voxels,
        )
        if int(intersections["intersectingTrianglePairCount"]):
            reasons.append("completion intersects another selected surface")

        base_triangle = np.asarray(current["triangleFrontierIndex"], dtype=np.int32)
        base_component = np.asarray(current["component"], dtype=np.int32)
        same_component_triangle = base_triangle[
            np.all(base_component[base_triangle] == component_id, axis=1)
        ]
        chart_overlap = _chart_overlap_count(
            same_component_triangle,
            patch_triangle,
            np.asarray(trial["chartUV"], dtype=np.float32),
        )
        if chart_overlap:
            reasons.append("completion overlaps an existing triangle in its intrinsic chart")

        target_was_macro = bool(
            np.asarray(holes["loopMacroEligible"], dtype=np.uint8)[loop]
        )
        topology_exact = bool(
            boundary_before_once and boundary_after_twice and not nonmanifold
        )
        current_loop_count = dict(predicted_loop_count)
        trial_loop_count = dict(current_loop_count)
        if topology_exact:
            trial_loop_count["interiorHoleCount"] -= 1
            trial_loop_count["macroHoleCount"] -= int(target_was_macro)
        trial_region = current_region_count

        accepted = not reasons
        if accepted:
            current = trial
            current_incidence = trial_incidence
            predicted_loop_count = trial_loop_count
            accepted_added_edge.extend(
                (int(value[0]), int(value[1])) for value in added_edge
            )
            proposal_node.extend(int(value) for value in synthetic_node)
            completion_triangle.extend(int(value) for value in patch_triangle.ravel())
            completion_field_pixel.extend(int(value) for value in retained_pixel)
            field_audit_count = len(field_local)
            completion_profile.extend(
                native_arrays["normalProfile"][:field_audit_count]
            )
            completion_physical.extend(
                float(value)
                for value in native_arrays["physicalScore"][:field_audit_count]
            )
            completion_correlation.extend(
                float(value)
                for value in native_arrays["profileCorrelation"][
                    :field_audit_count
                ]
            )
            completion_margin.extend(
                float(value)
                for value in native_arrays["farLayerMargin"][:field_audit_count]
            )
            completion_supported.extend(
                int(value)
                for value in native_arrays["ctSupported"][:field_audit_count]
            )
        proposal_offset.append(len(proposal_node))
        completion_triangle_offset.append(len(completion_triangle) // 3)
        completion_field_offset.append(len(completion_field_pixel))
        proposal_accepted.append(int(accepted))
        records.append(
            {
                "holeRow": row,
                "loopIndex": loop,
                "component": component_id,
                "accepted": accepted,
                "rejectionReasons": reasons,
                "depthFieldSupportedFraction": round(depth_support_fraction, 6),
                "mesh": mesh_stats,
                "geometry": {
                    "minimumTriangleAreaVoxelsSquared": round(float(min(triangle_area)), 6),
                    "maximumTriangleEdgeVoxels": round(float(max(triangle_edge)), 6),
                    "medianTriangleNormalResidualDegrees": round(
                        float(np.median(triangle_residual)), 6
                    ),
                    "p90TriangleNormalResidualDegrees": round(
                        float(np.percentile(triangle_residual, 90)), 6
                    ),
                    "maximumTriangleNormalResidualDegrees": round(
                        float(max(triangle_residual)), 6
                    ),
                    "highNormalResidualAreaFraction": round(
                        high_normal_area_fraction, 6
                    ),
                    "intrinsicChartOverlapCount": int(chart_overlap),
                    "nonManifoldEdgeCount": int(nonmanifold),
                    "everyBoundaryEdgeOpenBefore": bool(boundary_before_once),
                    "everyBoundaryEdgeClosedAfter": bool(boundary_after_twice),
                    **clearance,
                    **intersections,
                },
                "nativeCt": native_stats,
                "topology": {
                    "interiorHoleCountBefore": current_loop_count["interiorHoleCount"],
                    "interiorHoleCountAfter": trial_loop_count["interiorHoleCount"],
                    "macroHoleCountBefore": current_loop_count["macroHoleCount"],
                    "macroHoleCountAfter": trial_loop_count["macroHoleCount"],
                    "targetWasMacroEligible": target_was_macro,
                    "componentTriangleRegionCountBefore": current_region_count.get(component_id, 0),
                    "componentTriangleRegionCountAfter": trial_region.get(component_id, 0),
                },
            }
        )

    accepted_count = int(np.count_nonzero(proposal_accepted))
    final_loops, final_loop_stats = extract_surface_boundary_loops(
        current, settings=hole_settings
    )
    final_loop_count = _loop_counts(final_loops)
    if final_loop_count != predicted_loop_count:
        raise ValueError(
            "local completion incidence and final whole-surface topology differ: "
            f"predicted={predicted_loop_count}, actual={final_loop_count}"
        )
    if int(final_loop_stats["unresolvedBoundaryFanCount"]) > int(
        initial_loop_stats["unresolvedBoundaryFanCount"]
    ):
        raise ValueError("dense completion introduced an unresolved boundary fan")
    added_edge_array = np.asarray(accepted_added_edge, dtype=np.int32).reshape((-1, 2))
    current["edgeFirstFrontierIndex"] = np.concatenate(
        (
            np.asarray(current["edgeFirstFrontierIndex"], dtype=np.int32),
            added_edge_array[:, 0]
            if len(added_edge_array)
            else np.empty(0, dtype=np.int32),
        )
    )
    current["edgeSecondFrontierIndex"] = np.concatenate(
        (
            np.asarray(current["edgeSecondFrontierIndex"], dtype=np.int32),
            added_edge_array[:, 1]
            if len(added_edge_array)
            else np.empty(0, dtype=np.int32),
        )
    )
    current["edgeSelected"] = np.concatenate(
        (
            np.asarray(current["edgeSelected"], dtype=np.uint8),
            np.ones(len(added_edge_array), dtype=np.uint8),
        )
    )
    component = np.asarray(current["component"], dtype=np.int32)
    component_size = np.bincount(component[component >= 0]).astype(np.int32)
    node_component_size = np.zeros(len(component), dtype=np.int32)
    valid_component = component >= 0
    node_component_size[valid_component] = component_size[
        component[valid_component]
    ]
    current["componentSize"] = node_component_size
    arrays: dict[str, np.ndarray] = {
        **{key: np.asarray(value) for key, value in current.items()},
        "proposalOffset": np.asarray(proposal_offset, dtype=np.int64),
        "proposalFrontierIndex": np.asarray(proposal_node, dtype=np.int32),
        "proposalAccepted": np.asarray(proposal_accepted, dtype=np.uint8),
        "proposalHoleRow": np.asarray(ranked_rows, dtype=np.int32),
        "completionTriangleOffset": np.asarray(completion_triangle_offset, dtype=np.int64),
        "completionTriangleFrontierIndex": np.asarray(
            completion_triangle, dtype=np.int32
        ).reshape((-1, 3)),
        "completionFieldOffset": np.asarray(completion_field_offset, dtype=np.int64),
        "completionFieldPixel": np.asarray(completion_field_pixel, dtype=np.int32),
        "completionNormalProfile": np.asarray(completion_profile, dtype=np.float32).reshape(
            (-1, len(settings.competing_shift_thicknesses), len(settings.profile_depth_fractions))
        ),
        "completionPhysicalScore": np.asarray(completion_physical, dtype=np.float32),
        "completionProfileCorrelation": np.asarray(
            completion_correlation, dtype=np.float32
        ),
        "completionFarLayerMargin": np.asarray(completion_margin, dtype=np.float32),
        "completionCtSupported": np.asarray(completion_supported, dtype=np.uint8),
        "completionProfileDepthFractions": np.asarray(
            settings.profile_depth_fractions, dtype=np.float32
        ),
        "completionCompetingShiftThicknesses": np.asarray(
            settings.competing_shift_thicknesses, dtype=np.float32
        ),
        "baseNodeCount": np.asarray([baseline_node_count], dtype=np.int64),
        "baseTriangleCount": np.asarray([baseline_triangle_count], dtype=np.int64),
        "loopOffset": np.asarray(final_loops["loopOffset"], dtype=np.int64),
        "loopVertexFrontierIndex": np.asarray(
            final_loops["loopVertexFrontierIndex"], dtype=np.int32
        ),
        "loopKind": np.asarray(final_loops["loopKind"], dtype=np.uint8),
        "loopMacroEligible": np.asarray(
            final_loops["loopMacroEligible"], dtype=np.uint8
        ),
    }
    return arrays, records, {
        "attemptedHoleCount": len(ranked_rows),
        "acceptedHoleCount": accepted_count,
        "addedDenseNodeCount": int(len(proposal_node)),
        "addedTriangleCount": int(len(completion_triangle) // 3),
        "nodeCountBefore": baseline_node_count,
        "nodeCountAfter": int(len(current["midpointXYZ"])),
        "triangleCountBefore": baseline_triangle_count,
        "triangleCountAfter": int(len(current["triangleFrontierIndex"])),
        "macroHoleCountBefore": loop_count_before["macroHoleCount"],
        "macroHoleCountAfter": final_loop_count["macroHoleCount"],
        "interiorHoleCountBefore": loop_count_before["interiorHoleCount"],
        "interiorHoleCountAfter": final_loop_count["interiorHoleCount"],
        "triangleRegionCountBefore": int(
            len(
                np.unique(
                    _triangle_region_labels(
                        _surface_view(holes)["triangleFrontierIndex"]
                    )
                )
            )
        ),
        "triangleRegionCountAfter": int(
            len(
                np.unique(
                    _triangle_region_labels(current["triangleFrontierIndex"])
                )
            )
        ),
        "finalBoundaryAudit": final_loop_stats,
        "decisionUnit": (
            "one complete weakly-simple closed boundary and its collective "
            "CT depth field"
        ),
        "singlePixelGrowth": False,
        "ribbonCandidatesRequiredForInterior": False,
        "fittedNormalTailIsDiagnostic": True,
        "compressedSheetClearanceIsDiagnostic": True,
        "identityLabelsUsed": False,
    }


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def run_physical_ribbon_dense_completion(
    holes_root: str | Path,
    depth_field_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonDenseCompletionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonDenseCompletionSettings()
    holes_path, holes_manifest = _resolve_holes_manifest(holes_root)
    holes = _load_npz(
        holes_path.parent / str(holes_manifest["data"]["path"]),
        holes_manifest["data"]["sha256"],
    )
    depth_path, depth_manifest = _resolve_depth_field_manifest(depth_field_root)
    depth_field = _load_npz(
        depth_path.parent / str(depth_manifest["data"]["path"]),
        depth_manifest["data"]["sha256"],
    )
    _validate_depth_field(
        holes_path,
        holes_manifest,
        holes,
        depth_manifest,
        depth_field,
    )
    source_record = holes_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    hole_setting_values = dict(
        holes_manifest.get("identity", {}).get("settings", {})
    )
    allowed_hole_settings = {
        value.name for value in fields(PhysicalRibbonPatchHoleSettings)
    }
    hole_setting_values = {
        key: value
        for key, value in hole_setting_values.items()
        if key in allowed_hole_settings
    }
    for name in ("profile_depth_fractions", "competing_shift_thicknesses"):
        if name in hole_setting_values:
            hole_setting_values[name] = tuple(hole_setting_values[name])
    hole_settings = PhysicalRibbonPatchHoleSettings(**hole_setting_values)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA,
        "version": PHYSICAL_RIBBON_DENSE_COMPLETION_VERSION,
        "holes": _reference(holes_path, holes_manifest),
        "depthField": _reference(depth_path, depth_manifest),
        "topologyContinuity": holes_manifest["identity"]["topologyContinuity"],
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    arrays, records, statistics = build_physical_ribbon_dense_completion(
        holes,
        depth_field,
        source,
        hole_settings=hole_settings,
        settings=resolved,
    )
    built = time.monotonic()
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA,
        "version": PHYSICAL_RIBBON_DENSE_COMPLETION_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": holes_manifest.get("geometry", {}),
        "analysis": statistics,
        "completions": records,
        "timingSeconds": {
            "collectiveSurfaceCompletion": round(built - started, 6),
            "writing": round(finished - built, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "decisionUnit": (
                "one complete closed boundary and collective dense CT "
                "normal-depth field"
            ),
            "boundaryGeometry": (
                "weakly-simple loops are decomposed at pinch vertices into "
                "exact edge-preserving disk cycles"
            ),
            "surfaceRepresentation": (
                "constrained intrinsic triangles through dense CT field "
                "samples; ribbon-bank nodes are boundary and collision "
                "evidence only"
            ),
            "nativeCtAudit": (
                "the realized triangles are re-sampled at retained dense "
                "field vertices and triangle centroids using their mesh "
                "normals against boundary context and displaced competing "
                "layers"
            ),
            "defectHandling": (
                "fitted-normal tails and thickness-normalized proximity are "
                "reported diagnostics rather than vetoes; tight bends and "
                "compressed or delaminated sheets remain admissible when "
                "their realized triangles have direct CT support and do not "
                "literally intersect another surface"
            ),
            "topologyAudit": (
                "every prior boundary edge becomes exactly two-incident, no "
                "edge exceeds two faces, the target macro/interior loop "
                "disappears, and no triangle region is created"
            ),
            "mutation": (
                "accepted completions augment a versioned surface artifact; "
                "source ribbon configuration and depth field remain unchanged"
            ),
            "singlePixelGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
