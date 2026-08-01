from __future__ import annotations

import heapq
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
from .physical_ribbon_flattened_audit import (
    PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
)
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
from .physical_ribbon_open_bays import PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA


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
                PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA,
            }
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "holes root must identify one complete patch, surface-hole, or "
            "open-bay artifact"
        )
    return matches[0]


def _resolve_texture_audit_manifest(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "texture-audit root must identify one complete flattened audit"
        )
    return matches[0]


def _texture_compatible_hole_rows(
    completion_manifest: Mapping[str, Any],
    audit_manifest: Mapping[str, Any],
) -> frozenset[int]:
    """Require an exhaustive, disjoint texture verdict for every completion."""

    accepted_values = [
        int(record["holeRow"])
        for record in completion_manifest.get("completions", ())
        if bool(record.get("accepted"))
    ]
    accepted = set(accepted_values)
    if len(accepted) != len(accepted_values):
        raise ValueError("accepted completion hole rows are not unique")
    audit = audit_manifest.get("audit", {})
    compatible = {
        int(value)
        for value in audit.get("boundaryTextureCompatibleCompletionHoleRows", ())
    }
    incompatible = {
        int(value)
        for value in audit.get("boundaryTextureIncompatibleCompletionHoleRows", ())
    }
    unmeasured = {
        int(value)
        for value in audit.get("boundaryTextureUnmeasuredCompletionHoleRows", ())
    }
    if compatible & incompatible or compatible & unmeasured or incompatible & unmeasured:
        raise ValueError("flattened completion texture verdicts overlap")
    if compatible | incompatible | unmeasured != accepted:
        raise ValueError(
            "flattened completion texture verdicts do not exhaust accepted rows"
        )
    declared_count = int(audit.get("flattenedCompletionProposalCount", -1))
    if declared_count != len(accepted):
        raise ValueError("flattened completion proposal count differs from source")
    return frozenset(compatible)


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
    minimum_reconstructed_supported_fraction: float = 0.70
    minimum_median_profile_correlation: float = 0.65
    minimum_median_far_layer_margin: float = 0.02
    interior_boundary_separation_hypotheses_voxels: tuple[float, ...] = (
        0.20,
        0.30,
        0.50,
        0.75,
        1.00,
        1.50,
    )
    minimum_triangle_area_voxels_squared: float = 0.03
    maximum_triangle_edge_voxels: float = 6.0
    high_triangle_normal_residual_degrees: float = 45.0
    maximum_triangle_normal_residual_degrees: float = 85.0
    maximum_native_ct_quadrature_edge_voxels: float = 1.0
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
            self.minimum_triangle_area_voxels_squared,
            self.maximum_triangle_edge_voxels,
            self.high_triangle_normal_residual_degrees,
            self.maximum_triangle_normal_residual_degrees,
            self.maximum_native_ct_quadrature_edge_voxels,
            self.intersection_tolerance_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("dense-completion geometric scales must be positive")
        separations = tuple(
            float(value)
            for value in self.interior_boundary_separation_hypotheses_voxels
        )
        if (
            not separations
            or any(not math.isfinite(value) or value <= 0.0 for value in separations)
            or tuple(sorted(set(separations))) != separations
        ):
            raise ValueError(
                "boundary-separation hypotheses must be unique, positive, and sorted"
            )
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


def _triangle_edges(triangle: tuple[int, int, int]) -> tuple[tuple[int, int], ...]:
    return tuple(
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for index in range(3)
    )


def _edge_triangle_sets(
    triangles: list[tuple[int, int, int]],
) -> dict[tuple[int, int], set[int]]:
    result: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triangle_index, triangle in enumerate(triangles):
        for edge in _triangle_edges(triangle):
            result[edge].add(triangle_index)
    return result


def _replace_triangle_pair(
    triangles: list[tuple[int, int, int]],
    edge_triangle: dict[tuple[int, int], set[int]],
    first_index: int,
    second_index: int,
    replacement: tuple[tuple[int, int, int], tuple[int, int, int]],
) -> set[tuple[int, int]]:
    affected: set[tuple[int, int]] = set()
    for triangle_index in (first_index, second_index):
        for edge in _triangle_edges(triangles[triangle_index]):
            affected.add(edge)
            incident = edge_triangle[edge]
            incident.remove(triangle_index)
            if not incident:
                del edge_triangle[edge]
    triangles[first_index], triangles[second_index] = replacement
    for triangle_index in (first_index, second_index):
        for edge in _triangle_edges(triangles[triangle_index]):
            affected.add(edge)
            edge_triangle.setdefault(edge, set()).add(triangle_index)
    return affected


def _improve_chart_triangulation(
    triangles: list[tuple[int, int, int]],
    chart_uv: np.ndarray,
    constrained_edges: set[tuple[int, int]],
    *,
    maximum_iterations: int,
) -> tuple[list[tuple[int, int, int]], int]:
    """Lawson flips improve element quality without changing the boundary."""

    edge_triangle = _edge_triangle_sets(triangles)
    pending = {
        edge
        for edge, incident in edge_triangle.items()
        if len(incident) == 2 and edge not in constrained_edges
    }
    queue = list(pending)
    heapq.heapify(queue)
    flip_count = 0
    while queue:
        edge = heapq.heappop(queue)
        pending.discard(edge)
        incident = edge_triangle.get(edge, set())
        if len(incident) != 2 or edge in constrained_edges:
            continue
        first, second = edge
        left_index, right_index = sorted(incident)
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
        replacement_edge = (
            min(left_other, right_other),
            max(left_other, right_other),
        )
        if replacement_edge in edge_triangle:
            continue
        if (
            _cross_2d(chart_uv[first], chart_uv[second], chart_uv[left_other])
            * _cross_2d(chart_uv[first], chart_uv[second], chart_uv[right_other])
            >= -1.0e-10
            or _cross_2d(chart_uv[left_other], chart_uv[right_other], chart_uv[first])
            * _cross_2d(chart_uv[left_other], chart_uv[right_other], chart_uv[second])
            >= -1.0e-10
        ):
            continue
        replacement = (
            _orient_chart_triangle((left_other, right_other, first), chart_uv),
            _orient_chart_triangle((right_other, left_other, second), chart_uv),
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
        flip_count += 1
        if flip_count > maximum_iterations:
            raise ValueError("constrained chart edge flips did not converge")
        affected = _replace_triangle_pair(
            triangles,
            edge_triangle,
            left_index,
            right_index,
            replacement,
        )
        for candidate in affected:
            if (
                candidate not in pending
                and candidate not in constrained_edges
                and len(edge_triangle.get(candidate, ())) == 2
            ):
                heapq.heappush(queue, candidate)
                pending.add(candidate)
    return triangles, flip_count


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

    edge_triangle = _edge_triangle_sets(values)
    constrained = {
        edge for edge, incident in edge_triangle.items() if len(incident) == 1
    }
    pending = {
        edge
        for edge, incident in edge_triangle.items()
        if len(incident) == 2 and edge not in constrained
    }
    queue = list(pending)
    heapq.heapify(queue)
    flip_count = 0
    while queue:
        edge = heapq.heappop(queue)
        pending.discard(edge)
        incident = edge_triangle.get(edge, set())
        if len(incident) != 2 or edge in constrained:
            continue
        first, second = edge
        left_index, right_index = sorted(incident)
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
        flip_count += 1
        if flip_count > maximum_iterations:
            raise ValueError("physical completion edge flips did not converge")
        affected = _replace_triangle_pair(
            values,
            edge_triangle,
            left_index,
            right_index,
            replacement,
        )
        for candidate in affected:
            if (
                candidate not in pending
                and candidate not in constrained
                and len(edge_triangle.get(candidate, ())) == 2
            ):
                heapq.heappush(queue, candidate)
                pending.add(candidate)
    return np.asarray(values, dtype=np.int32), flip_count


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


def _mesh_edge_length_audit(
    triangles: np.ndarray,
    point_xyz: np.ndarray,
    point_kind: np.ndarray,
    point_source_index: np.ndarray,
    new_frontier_edges: AbstractSet[tuple[int, int]],
) -> dict[str, float]:
    """Separate an open replacement mouth from CT-supported mesh edges."""

    values = np.asarray(triangles, dtype=np.int32)
    xyz = np.asarray(point_xyz, dtype=np.float64)
    kind = np.asarray(point_kind, dtype=np.uint8)
    source = np.asarray(point_source_index, dtype=np.int32)
    normalized_frontier = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in new_frontier_edges
    }
    unique_edges = {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in values
        for index in range(3)
    }
    all_length: list[float] = []
    frontier_length: list[float] = []
    supported_length: list[float] = []
    for first, second in unique_edges:
        length = float(np.linalg.norm(xyz[second] - xyz[first]))
        all_length.append(length)
        global_edge = (
            tuple(sorted((int(source[first]), int(source[second]))))
            if kind[first] == 0 and kind[second] == 0
            else None
        )
        if global_edge in normalized_frontier:
            frontier_length.append(length)
        else:
            supported_length.append(length)
    return {
        "maximumTriangleEdgeVoxels": max(all_length, default=0.0),
        "maximumCtSupportedTriangleEdgeVoxels": max(
            supported_length, default=0.0
        ),
        "maximumOpenFrontierEdgeVoxels": max(frontier_length, default=0.0),
    }


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


def _surface_field_integrability(
    coordinates: np.ndarray,
    point_xyz: np.ndarray,
    reference_normal_xyz: np.ndarray,
    *,
    high_residual_degrees: float,
) -> dict[str, float | int]:
    """Measure whether independently supported depth samples form a surface.

    The collective depth solver assigns one displacement to each raster point.
    Pointwise CT support does not guarantee that neighboring assignments can be
    joined into a sheet.  Complete 2x2 raster cells provide a candidate-free,
    triangulation-independent integrability test before the expensive exact
    constrained mesh solve.
    """

    coordinate = np.asarray(coordinates, dtype=np.int32)
    xyz = np.asarray(point_xyz, dtype=np.float64)
    reference = np.asarray(reference_normal_xyz, dtype=np.float64)
    if not len(coordinate):
        return {
            "gridCellCount": 0,
            "gridTriangleCount": 0,
            "medianTriangleNormalResidualDegrees": 90.0,
            "p90TriangleNormalResidualDegrees": 90.0,
            "highNormalResidualAreaFraction": 1.0,
            "maximumGridEdgeVoxels": 0.0,
            "p90GridEdgeNormalDepartureDegrees": 90.0,
            "surfaceCoherenceScore": 0.0,
        }
    lookup = {
        (int(value[0]), int(value[1])): index
        for index, value in enumerate(coordinate)
    }
    cells: list[tuple[int, int, int, int]] = []
    edges: set[tuple[int, int]] = set()
    for first, value in enumerate(coordinate):
        u, v = int(value[0]), int(value[1])
        for neighbor_coordinate in ((u + 1, v), (u, v + 1)):
            neighbor = lookup.get(neighbor_coordinate)
            if neighbor is not None:
                edges.add(tuple(sorted((first, neighbor))))
        second = lookup.get((u + 1, v))
        third = lookup.get((u, v + 1))
        fourth = lookup.get((u + 1, v + 1))
        if second is not None and third is not None and fourth is not None:
            cells.append((first, second, third, fourth))

    triangle_area: list[float] = []
    triangle_residual: list[float] = []
    for first, second, third, fourth in cells:
        alternatives = (
            ((first, second, fourth), (first, fourth, third)),
            ((first, second, third), (second, fourth, third)),
        )
        evaluated: list[tuple[tuple[float, float], list[tuple[float, float]]]] = []
        for alternative in alternatives:
            values: list[tuple[float, float]] = []
            for triangle in alternative:
                _, area, residual, _ = _triangle_geometry(
                    triangle, xyz, reference
                )
                values.append((float(area), float(residual)))
            objective = (
                max(value[1] for value in values),
                sum(value[0] * value[1] for value in values)
                / max(sum(value[0] for value in values), 1.0e-12),
            )
            evaluated.append((objective, values))
        _, selected = min(evaluated, key=lambda value: value[0])
        triangle_area.extend(value[0] for value in selected)
        triangle_residual.extend(value[1] for value in selected)

    edge_length: list[float] = []
    edge_departure: list[float] = []
    for first, second in edges:
        tangent = xyz[second] - xyz[first]
        length = float(np.linalg.norm(tangent))
        if length <= 1.0e-12:
            continue
        first_normal = reference[first]
        second_normal = reference[second]
        if float(np.dot(first_normal, second_normal)) < 0.0:
            second_normal = -second_normal
        normal = first_normal + second_normal
        normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1.0e-12:
            normal = first_normal
            normal_length = float(np.linalg.norm(normal))
        if normal_length <= 1.0e-12:
            edge_length.append(length)
            edge_departure.append(90.0)
            continue
        departure = math.degrees(
            math.asin(
                float(
                    np.clip(
                        abs(float(np.dot(tangent / length, normal / normal_length))),
                        0.0,
                        1.0,
                    )
                )
            )
        )
        edge_length.append(length)
        edge_departure.append(departure)

    area = np.asarray(triangle_area, dtype=np.float64)
    residual = np.asarray(triangle_residual, dtype=np.float64)
    total_area = float(np.sum(area))
    high_fraction = (
        float(np.sum(area[residual > high_residual_degrees]) / total_area)
        if total_area > 1.0e-12
        else 1.0
    )
    p90_residual = (
        float(np.percentile(residual, 90)) if len(residual) else 90.0
    )
    coherence = (
        max(0.0, math.cos(math.radians(min(p90_residual, 90.0))))
        * max(0.0, 1.0 - high_fraction)
        if math.isfinite(p90_residual)
        else 0.0
    )
    return {
        "gridCellCount": int(len(cells)),
        "gridTriangleCount": int(len(residual)),
        "medianTriangleNormalResidualDegrees": round(
            float(np.median(residual)) if len(residual) else 90.0, 6
        ),
        "p90TriangleNormalResidualDegrees": round(p90_residual, 6),
        "highNormalResidualAreaFraction": round(high_fraction, 6),
        "maximumGridEdgeVoxels": round(
            max(edge_length, default=0.0), 6
        ),
        "p90GridEdgeNormalDepartureDegrees": round(
            float(np.percentile(edge_departure, 90))
            if edge_departure
            else 90.0,
            6,
        ),
        "surfaceCoherenceScore": round(coherence, 6),
    }


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


def _triangle_quadrature_samples(
    triangle_xyz: np.ndarray,
    triangle_normal_xyz: np.ndarray,
    *,
    maximum_edge_voxels: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Uniformly cover every realized facet with subtriangle centroids.

    A single centroid can alias a planar facet across a tightly curved sheet.
    Dividing each triangle until every subtriangle edge is at most the
    declared physical spacing makes native-CT support an area measurement.
    The returned triangle index preserves provenance for diagnostics.
    """

    triangles = np.asarray(triangle_xyz, dtype=np.float64)
    normals = np.asarray(triangle_normal_xyz, dtype=np.float64)
    if triangles.ndim != 3 or triangles.shape[1:] != (3, 3):
        raise ValueError("triangle quadrature expects T x 3 x 3 positions")
    if normals.shape != (len(triangles), 3):
        raise ValueError("triangle quadrature normals differ from triangles")
    if not math.isfinite(maximum_edge_voxels) or maximum_edge_voxels <= 0.0:
        raise ValueError("triangle quadrature spacing must be positive")
    point_values: list[np.ndarray] = []
    normal_values: list[np.ndarray] = []
    triangle_values: list[int] = []
    for triangle_index, points in enumerate(triangles):
        edge = (
            float(np.linalg.norm(points[1] - points[0])),
            float(np.linalg.norm(points[2] - points[1])),
            float(np.linalg.norm(points[0] - points[2])),
        )
        divisions = max(int(math.ceil(max(edge) / maximum_edge_voxels)), 1)
        for first in range(divisions):
            for second in range(divisions - first):
                # Centroid of (i,j), (i+1,j), (i,j+1).
                first_weight = (first + 1.0 / 3.0) / divisions
                second_weight = (second + 1.0 / 3.0) / divisions
                point_values.append(
                    (1.0 - first_weight - second_weight) * points[0]
                    + first_weight * points[1]
                    + second_weight * points[2]
                )
                normal_values.append(normals[triangle_index])
                triangle_values.append(triangle_index)
                if second >= divisions - first - 1:
                    continue
                # Centroid of (i+1,j), (i+1,j+1), (i,j+1).
                first_weight = (first + 2.0 / 3.0) / divisions
                second_weight = (second + 2.0 / 3.0) / divisions
                point_values.append(
                    (1.0 - first_weight - second_weight) * points[0]
                    + first_weight * points[1]
                    + second_weight * points[2]
                )
                normal_values.append(normals[triangle_index])
                triangle_values.append(triangle_index)
    return (
        np.asarray(point_values, dtype=np.float32).reshape((-1, 3)),
        np.asarray(normal_values, dtype=np.float32).reshape((-1, 3)),
        np.asarray(triangle_values, dtype=np.int32),
    )


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


@dataclass(frozen=True, slots=True)
class _TriangleSpatialIndex:
    """A conservative uniform-grid index over immutable surface triangles."""

    triangle: np.ndarray
    point: np.ndarray
    low: np.ndarray
    high: np.ndarray
    cell_size: float
    bucket: Mapping[tuple[int, int, int], tuple[int, ...]]


def _triangle_spatial_index(
    surface: Mapping[str, np.ndarray],
    *,
    tolerance: float,
    cell_size: float,
) -> _TriangleSpatialIndex:
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    point = xyz[triangle]
    low = np.min(point, axis=1) - tolerance
    high = np.max(point, axis=1) + tolerance
    scale = max(float(cell_size), 2.0 * float(tolerance), 1.0e-3)
    first_cell = np.floor(low / scale).astype(np.int32)
    last_cell = np.floor(high / scale).astype(np.int32)
    mutable_bucket: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for triangle_index, (first, last) in enumerate(zip(first_cell, last_cell)):
        for first_axis in range(int(first[0]), int(last[0]) + 1):
            for second_axis in range(int(first[1]), int(last[1]) + 1):
                for third_axis in range(int(first[2]), int(last[2]) + 1):
                    mutable_bucket[(first_axis, second_axis, third_axis)].append(
                        triangle_index
                    )
    return _TriangleSpatialIndex(
        triangle=triangle,
        point=point,
        low=low,
        high=high,
        cell_size=scale,
        bucket={key: tuple(value) for key, value in mutable_bucket.items()},
    )


def _spatial_triangle_candidates(
    index: _TriangleSpatialIndex,
    low: np.ndarray,
    high: np.ndarray,
) -> np.ndarray:
    first_cell = np.floor(np.asarray(low) / index.cell_size).astype(np.int32)
    last_cell = np.floor(np.asarray(high) / index.cell_size).astype(np.int32)
    possible: set[int] = set()
    for first_axis in range(int(first_cell[0]), int(last_cell[0]) + 1):
        for second_axis in range(int(first_cell[1]), int(last_cell[1]) + 1):
            for third_axis in range(int(first_cell[2]), int(last_cell[2]) + 1):
                possible.update(
                    index.bucket.get((first_axis, second_axis, third_axis), ())
                )
    if not possible:
        return np.empty(0, dtype=np.int32)
    candidate = np.fromiter(sorted(possible), dtype=np.int32)
    overlaps = np.all(index.high[candidate] >= low, axis=1) & np.all(
        index.low[candidate] <= high, axis=1
    )
    return candidate[overlaps]


def _other_component_triangle_intersections(
    baseline_surface: Mapping[str, np.ndarray],
    augmented_surface: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    component_id: int,
    *,
    tolerance: float,
    spatial_index: _TriangleSpatialIndex | None = None,
) -> dict[str, int]:
    """Audit crossings against every nonincident selected surface triangle."""

    from ..slab_association_integrity import _triangle_intersection

    index = spatial_index or _triangle_spatial_index(
        baseline_surface,
        tolerance=tolerance,
        cell_size=4.0,
    )
    baseline_triangle = index.triangle
    component = np.asarray(baseline_surface["component"], dtype=np.int32)
    other_component_mask = ~np.all(
        component[baseline_triangle] == component_id, axis=1
    )
    if not len(baseline_triangle) or not len(patch_triangle):
        return {
            "otherComponentTriangleCount": int(
                np.count_nonzero(other_component_mask)
            ),
            "sameComponentTriangleCount": int(
                np.count_nonzero(~other_component_mask)
            ),
            "broadPhaseTrianglePairCount": 0,
            "intersectingTrianglePairCount": 0,
        }
    xyz = np.asarray(augmented_surface["midpointXYZ"], dtype=np.float32)
    other_point = index.point
    broad_count = 0
    intersection_count = 0
    for triangle in np.asarray(patch_triangle, dtype=np.int32):
        point = xyz[triangle]
        low = np.min(point, axis=0) - tolerance
        high = np.max(point, axis=0) + tolerance
        possible = _spatial_triangle_candidates(index, low, high)
        broad_count += len(possible)
        for other_index in possible:
            # Shared boundary edges and pinch vertices are the intended exact
            # attachment, not a crossing.  Every nonincident triangle,
            # including another region carrying the same component label, is
            # still screened.  This prevents a bay from folding through the
            # back of its own reconstructed sheet.
            baseline_nodes = baseline_triangle[int(other_index)]
            if np.any(triangle[:, None] == baseline_nodes[None, :]):
                continue
            intersection, _ = _triangle_intersection(
                point, other_point[int(other_index)], tolerance
            )
            if intersection is not None:
                intersection_count += 1
    return {
        "otherComponentTriangleCount": int(
            np.count_nonzero(other_component_mask)
        ),
        "sameComponentTriangleCount": int(
            np.count_nonzero(~other_component_mask)
        ),
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


def _evaluate_dense_completion_variant(
    current: Mapping[str, np.ndarray],
    current_incidence: Counter[tuple[int, int]],
    holes: Mapping[str, np.ndarray],
    source: VolumeSource,
    volume: np.ndarray,
    *,
    row: int,
    component_id: int,
    thickness: float,
    boundary: np.ndarray,
    field_uv: np.ndarray,
    field_xyz: np.ndarray,
    field_reference: np.ndarray,
    boundary_separation_voxels: float,
    new_frontier_edges: AbstractSet[tuple[int, int]] = frozenset(),
    collision_index: _TriangleSpatialIndex | None = None,
    settings: PhysicalRibbonDenseCompletionSettings,
) -> dict[str, Any]:
    """Construct and audit one complete mesh-density hypothesis."""

    try:
        mesh, mesh_stats = triangulate_weak_boundary_field(
            boundary,
            np.asarray(current["chartUV"], dtype=np.float32)[boundary],
            field_uv,
            minimum_boundary_separation=boundary_separation_voxels,
            maximum_edge_flip_iterations=settings.maximum_edge_flip_iterations,
        )
    except ValueError as error:
        return {
            "constructed": False,
            "accepted": False,
            "rejectionReasons": [str(error)],
            "boundarySeparationVoxels": boundary_separation_voxels,
        }

    reasons: list[str] = []
    retained_pixel = np.asarray(mesh["retainedFieldPixel"], dtype=np.int32)
    point_kind = np.asarray(mesh["pointKind"], dtype=np.uint8)
    point_source = np.asarray(mesh["pointSourceIndex"], dtype=np.int32)
    boundary_local = np.flatnonzero(point_kind == 0)
    field_local = np.flatnonzero(point_kind == 1)
    normalized_new_frontier_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in new_frontier_edges
    }
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
    local_triangle, physical_flips = _improve_physical_triangulation(
        np.asarray(mesh["trianglePointIndex"], dtype=np.int32),
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
    for triangle in local_triangle:
        values, area, residual, maximum_edge = _triangle_geometry(
            tuple(int(value) for value in triangle),
            local_xyz,
            local_reference,
        )
        oriented.append(values)
        triangle_area.append(area)
        triangle_residual.append(residual)
    mesh = dict(mesh)
    mesh["trianglePointIndex"] = np.asarray(oriented, dtype=np.int32)
    minimum_area = min(triangle_area, default=0.0)
    edge_audit = _mesh_edge_length_audit(
        np.asarray(oriented, dtype=np.int32),
        local_xyz,
        point_kind,
        point_source,
        normalized_new_frontier_edges,
    )
    maximum_edge = float(edge_audit["maximumTriangleEdgeVoxels"])
    maximum_supported_edge = float(
        edge_audit["maximumCtSupportedTriangleEdgeVoxels"]
    )
    maximum_frontier_edge = float(
        edge_audit["maximumOpenFrontierEdgeVoxels"]
    )
    maximum_residual = max(triangle_residual, default=float("inf"))
    if minimum_area < settings.minimum_triangle_area_voxels_squared:
        reasons.append("completion contains a physically degenerate triangle")
    if maximum_supported_edge > settings.maximum_triangle_edge_voxels:
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
    if maximum_residual > settings.maximum_triangle_normal_residual_degrees:
        reasons.append("completion triangle contradicts the local CT normal field")

    local_normal = _mesh_vertex_normals(
        np.asarray(mesh["trianglePointIndex"], dtype=np.int32),
        local_xyz,
        local_reference,
    )
    triangle_point = local_xyz[
        np.asarray(mesh["trianglePointIndex"], dtype=np.int32)
    ]
    triangle_centroid_xyz = np.mean(triangle_point, axis=1)
    triangle_normal = np.cross(
        triangle_point[:, 1] - triangle_point[:, 0],
        triangle_point[:, 2] - triangle_point[:, 0],
    )
    triangle_normal /= np.maximum(
        np.linalg.norm(triangle_normal, axis=1, keepdims=True), 1.0e-12
    )
    quadrature_xyz, quadrature_normal, quadrature_triangle = (
        _triangle_quadrature_samples(
            triangle_point,
            triangle_normal,
            maximum_edge_voxels=(
                settings.maximum_native_ct_quadrature_edge_voxels
            ),
        )
    )
    audit_xyz = np.vstack((local_xyz[field_local], quadrature_xyz)).astype(
        np.float32
    )
    audit_normal = np.vstack(
        (local_normal[field_local], quadrature_normal)
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
    native_stats["triangleQuadratureSampleCount"] = int(len(quadrature_xyz))
    native_stats["quadratureTriangleCount"] = int(
        len(np.unique(quadrature_triangle))
    )
    native_stats["maximumQuadratureEdgeVoxels"] = round(
        settings.maximum_native_ct_quadrature_edge_voxels, 6
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
        reasons.append(
            "constructed surface profile does not match its boundary context"
        )
    if (
        float(native_stats["medianFarLayerMargin"])
        < settings.minimum_median_far_layer_margin
    ):
        reasons.append("a displaced competing layer explains the constructed surface")
    clearance = _other_component_clearance(
        np.vstack((local_xyz[field_local], triangle_centroid_xyz)),
        current,
        component_id,
        thickness,
    )

    trial, synthetic_node, patch_triangle, added_edge = _append_completion_surface(
        current,
        mesh,
        field_xyz,
        field_reference,
        component_id,
        thickness,
        triangle_area=np.asarray(triangle_area, dtype=np.float32),
        triangle_normal_residual=np.asarray(
            triangle_residual, dtype=np.float32
        ),
        existing_surface_edges=current_incidence.keys(),
    )
    trial_incidence = current_incidence.copy()
    trial_incidence.update(_edge_incidence(patch_triangle))
    boundary_edges = {
        tuple(
            sorted(
                (
                    int(boundary[index]),
                    int(boundary[(index + 1) % len(boundary)]),
                )
            )
        )
        for index in range(len(boundary))
    }
    if not normalized_new_frontier_edges.issubset(boundary_edges):
        reasons.append("new frontier edge is not on the completion boundary")
    attachment_edges = boundary_edges - normalized_new_frontier_edges
    boundary_before_once = all(
        current_incidence[edge] == 1 for edge in attachment_edges
    )
    boundary_after_twice = all(
        trial_incidence[edge] == 2 for edge in attachment_edges
    )
    new_frontier_absent_before = all(
        current_incidence[edge] == 0 for edge in normalized_new_frontier_edges
    )
    new_frontier_once_after = all(
        trial_incidence[edge] == 1 for edge in normalized_new_frontier_edges
    )
    nonmanifold = sum(value > 2 for value in trial_incidence.values())
    if not boundary_before_once:
        reasons.append("target loop is no longer an exact open boundary")
    if not boundary_after_twice:
        reasons.append("completion does not attach to every target boundary edge once")
    if not new_frontier_absent_before:
        reasons.append("proposed open-bay mouth already exists on the surface")
    if not new_frontier_once_after:
        reasons.append("completion does not leave exactly one open face at its mouth")
    if nonmanifold:
        reasons.append("completion creates a non-manifold mesh edge")

    intersections = _other_component_triangle_intersections(
        current,
        trial,
        patch_triangle,
        component_id,
        tolerance=settings.intersection_tolerance_voxels,
        spatial_index=collision_index,
    )
    if int(intersections["intersectingTrianglePairCount"]):
        reasons.append("completion intersects an existing selected surface")
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
    topology_exact = bool(
        boundary_before_once
        and boundary_after_twice
        and new_frontier_absent_before
        and new_frontier_once_after
        and not nonmanifold
    )
    geometry = {
        "minimumTriangleAreaVoxelsSquared": round(float(minimum_area), 6),
        "maximumTriangleEdgeVoxels": round(float(maximum_edge), 6),
        "maximumCtSupportedTriangleEdgeVoxels": round(
            float(maximum_supported_edge), 6
        ),
        "maximumOpenFrontierEdgeVoxels": round(
            float(maximum_frontier_edge), 6
        ),
        "medianTriangleNormalResidualDegrees": round(
            float(np.median(triangle_residual)), 6
        ),
        "p90TriangleNormalResidualDegrees": round(
            float(np.percentile(triangle_residual, 90)), 6
        ),
        "maximumTriangleNormalResidualDegrees": round(
            float(maximum_residual), 6
        ),
        "highNormalResidualAreaFraction": round(
            high_normal_area_fraction, 6
        ),
        "intrinsicChartOverlapCount": int(chart_overlap),
        "nonManifoldEdgeCount": int(nonmanifold),
        "everyBoundaryEdgeOpenBefore": bool(boundary_before_once),
        "everyBoundaryEdgeClosedAfter": bool(boundary_after_twice),
        "attachmentBoundaryEdgeCount": int(len(attachment_edges)),
        "newFrontierEdgeCount": int(len(normalized_new_frontier_edges)),
        "everyNewFrontierEdgeAbsentBefore": bool(
            new_frontier_absent_before
        ),
        "everyNewFrontierEdgeOpenAfter": bool(new_frontier_once_after),
        **clearance,
        **intersections,
    }
    return {
        "constructed": True,
        "accepted": not reasons,
        "rejectionReasons": reasons,
        "boundarySeparationVoxels": boundary_separation_voxels,
        "mesh": mesh,
        "meshStatistics": mesh_stats,
        "geometry": geometry,
        "nativeArrays": native_arrays,
        "nativeStatistics": native_stats,
        "trial": trial,
        "syntheticNode": synthetic_node,
        "patchTriangle": patch_triangle,
        "addedEdge": added_edge,
        "trialIncidence": trial_incidence,
        "retainedFieldPixel": retained_pixel,
        "fieldAuditCount": int(len(field_local)),
        "topologyExact": topology_exact,
    }


def build_physical_ribbon_dense_completion(
    holes: Mapping[str, np.ndarray],
    depth_field: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    hole_settings: PhysicalRibbonPatchHoleSettings,
    settings: PhysicalRibbonDenseCompletionSettings,
    eligible_hole_rows: AbstractSet[int] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    current = _surface_view(holes)
    baseline_node_count = len(np.asarray(current["midpointXYZ"]))
    baseline_triangle_count = len(np.asarray(current["triangleFrontierIndex"]))
    baseline_loops, initial_loop_stats = extract_surface_boundary_loops(
        current, settings=hole_settings
    )
    loop_count_before = _loop_counts(baseline_loops)
    boundary_edge_count_before = int(
        len(np.asarray(baseline_loops["loopVertexFrontierIndex"]))
    )
    predicted_boundary_edge_count = boundary_edge_count_before
    open_bay_mode = {
        "bayMouthFirstFrontierIndex",
        "bayMouthSecondFrontierIndex",
    }.issubset(holes)
    current_incidence = _edge_incidence(current["triangleFrontierIndex"])
    current_region_count = _component_region_count(current)
    predicted_loop_count = dict(loop_count_before)
    volume = source.memmap()
    patch_offset = np.asarray(depth_field["holePatchOffset"], dtype=np.int64)
    loop_index = np.asarray(depth_field["holeLoopIndex"], dtype=np.int32)
    shifts = np.asarray(depth_field["shiftThicknesses"], dtype=np.float32)
    labels = np.asarray(depth_field["pixelCollectiveLabel"], dtype=np.int16)
    field_supported = np.asarray(depth_field["pixelCtSupported"], dtype=np.uint8) > 0
    field_coordinates = np.asarray(
        depth_field["patchGridCoordinateUV"], dtype=np.int32
    )
    field_correlation = np.asarray(
        depth_field["pixelCollectiveProfileCorrelation"], dtype=np.float32
    )
    field_margin = np.asarray(
        depth_field["pixelCollectiveFarLayerMargin"], dtype=np.float32
    )
    if eligible_hole_rows is not None and any(
        row < 0 or row >= len(loop_index) for row in eligible_hole_rows
    ):
        raise ValueError("texture-eligible hole row is outside the depth field")

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

    field_integrability: list[dict[str, float | int]] = []
    for row, loop_value in enumerate(loop_index):
        loop = int(loop_value)
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        pixel_slice = slice(start, stop)
        thickness = float(holes["loopMedianThicknessVoxels"][loop])
        chosen_shift = shifts[labels[pixel_slice]]
        point_xyz = np.asarray(
            depth_field["patchXYZ"], dtype=np.float32
        )[pixel_slice] + (
            np.asarray(depth_field["patchNormalXYZ"], dtype=np.float32)[
                pixel_slice
            ]
            * (chosen_shift * thickness)[:, None]
        )
        metrics = _surface_field_integrability(
            field_coordinates[pixel_slice],
            point_xyz,
            np.asarray(depth_field["patchNormalXYZ"], dtype=np.float32)[
                pixel_slice
            ],
            high_residual_degrees=(
                settings.high_triangle_normal_residual_degrees
            ),
        )
        metrics["ctSupportedFraction"] = round(
            float(np.mean(field_supported[pixel_slice])) if stop > start else 0.0,
            6,
        )
        metrics["medianProfileCorrelation"] = round(
            float(np.median(field_correlation[pixel_slice]))
            if stop > start
            else -1.0,
            6,
        )
        finite_margin = field_margin[pixel_slice][
            np.isfinite(field_margin[pixel_slice])
        ]
        metrics["medianFarLayerMargin"] = round(
            float(np.median(finite_margin)) if len(finite_margin) else -1.0e6,
            6,
        )
        field_integrability.append(metrics)

    def ranking_key(row: int) -> tuple[float | int, ...]:
        if not open_bay_mode:
            return (
                -float(holes["loopAreaChartVoxelsSquared"][loop_index[row]]),
                row,
            )
        return (
            -int(
                float(field_integrability[row]["ctSupportedFraction"])
                >= settings.minimum_depth_field_supported_fraction
            ),
            -float(field_integrability[row]["surfaceCoherenceScore"]),
            -float(field_integrability[row]["ctSupportedFraction"]),
            -float(field_integrability[row]["medianProfileCorrelation"]),
            -float(field_integrability[row]["medianFarLayerMargin"]),
            -float(holes["bayGeometryObjective"][loop_index[row]]),
            row,
        )

    candidate_rows = (
        range(len(loop_index))
        if eligible_hole_rows is None
        else (
            row
            for row in range(len(loop_index))
            if row in eligible_hole_rows
        )
    )
    ranked_rows = sorted(candidate_rows, key=ranking_key)[
        : settings.maximum_completed_holes
    ]
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
        depth_support_fraction = (
            float(np.mean(field_supported[pixel_slice])) if stop > start else 0.0
        )
        depth_reasons: list[str] = []
        if depth_support_fraction < settings.minimum_depth_field_supported_fraction:
            depth_reasons.append("insufficient collective depth-field CT support")

        hypothesis_records: list[dict[str, Any]] = []
        selected_variant: dict[str, Any] | None = None
        selected_hypothesis_index: int | None = None
        selected_separation: float | None = None
        last_variant: dict[str, Any] | None = None
        collision_index = _triangle_spatial_index(
            current,
            tolerance=settings.intersection_tolerance_voxels,
            cell_size=settings.maximum_triangle_edge_voxels,
        )
        separation_hypotheses = (
            settings.interior_boundary_separation_hypotheses_voxels[:1]
            if open_bay_mode
            else settings.interior_boundary_separation_hypotheses_voxels
        )
        for hypothesis_index, separation in enumerate(separation_hypotheses):
            variant = _evaluate_dense_completion_variant(
                current,
                current_incidence,
                holes,
                source,
                volume,
                row=row,
                component_id=component_id,
                thickness=thickness,
                boundary=boundary,
                field_uv=field_uv,
                field_xyz=field_xyz,
                field_reference=field_reference,
                boundary_separation_voxels=float(separation),
                new_frontier_edges=(
                    {
                        tuple(
                            sorted(
                                (
                                    int(holes["bayMouthFirstFrontierIndex"][loop]),
                                    int(holes["bayMouthSecondFrontierIndex"][loop]),
                                )
                            )
                        )
                    }
                    if open_bay_mode
                    else frozenset()
                ),
                collision_index=collision_index,
                settings=settings,
            )
            last_variant = variant
            variant_reasons = [
                *depth_reasons,
                *variant["rejectionReasons"],
            ]
            hypothesis_records.append(
                {
                    "hypothesisIndex": hypothesis_index,
                    "boundarySeparationVoxels": float(separation),
                    "accepted": not variant_reasons,
                    "rejectionReasons": variant_reasons,
                    "mesh": variant.get("meshStatistics"),
                    "geometry": variant.get("geometry"),
                    "nativeCt": variant.get("nativeStatistics"),
                }
            )
            if not variant_reasons:
                selected_variant = variant
                selected_hypothesis_index = hypothesis_index
                selected_separation = float(separation)
                break
            # Mesh density cannot repair a depth field that already lacks
            # collective CT support, so do not manufacture redundant variants.
            if depth_reasons:
                break

        target_was_macro = bool(
            np.asarray(holes["loopMacroEligible"], dtype=np.uint8)[loop]
        ) and not open_bay_mode
        current_loop_count = dict(predicted_loop_count)
        if selected_variant is None:
            reasons = list(hypothesis_records[-1]["rejectionReasons"])
            records.append(
                {
                    "holeRow": row,
                    "loopIndex": loop,
                    "component": component_id,
                    "accepted": False,
                    "rejectionReasons": reasons,
                    "selectedHypothesisIndex": None,
                    "selectedBoundarySeparationVoxels": None,
                    "hypotheses": hypothesis_records,
                    "depthFieldSupportedFraction": round(
                        depth_support_fraction, 6
                    ),
                    "depthFieldIntegrability": field_integrability[row],
                    "mesh": (
                        last_variant.get("meshStatistics")
                        if last_variant is not None
                        else None
                    ),
                    "geometry": (
                        last_variant.get("geometry")
                        if last_variant is not None
                        else None
                    ),
                    "nativeCt": (
                        last_variant.get("nativeStatistics")
                        if last_variant is not None
                        else None
                    ),
                    "topology": {
                        "interiorHoleCountBefore": current_loop_count[
                            "interiorHoleCount"
                        ],
                        "interiorHoleCountAfter": current_loop_count[
                            "interiorHoleCount"
                        ],
                        "macroHoleCountBefore": current_loop_count[
                            "macroHoleCount"
                        ],
                        "macroHoleCountAfter": current_loop_count[
                            "macroHoleCount"
                        ],
                        "targetWasMacroEligible": target_was_macro,
                        "targetWasOpenBay": open_bay_mode,
                        "componentTriangleRegionCountBefore": (
                            current_region_count.get(component_id, 0)
                        ),
                        "componentTriangleRegionCountAfter": (
                            current_region_count.get(component_id, 0)
                        ),
                    },
                }
            )
            proposal_offset.append(proposal_offset[-1])
            completion_triangle_offset.append(completion_triangle_offset[-1])
            completion_field_offset.append(completion_field_offset[-1])
            proposal_accepted.append(0)
            continue

        if selected_hypothesis_index is None or selected_separation is None:
            raise RuntimeError("accepted completion lacks hypothesis provenance")
        if not bool(selected_variant["topologyExact"]):
            raise RuntimeError("accepted completion lacks exact boundary topology")

        trial = selected_variant["trial"]
        synthetic_node = np.asarray(
            selected_variant["syntheticNode"], dtype=np.int32
        )
        patch_triangle = np.asarray(
            selected_variant["patchTriangle"], dtype=np.int32
        )
        added_edge = np.asarray(selected_variant["addedEdge"], dtype=np.int32)
        trial_incidence = selected_variant["trialIncidence"]
        retained_pixel = np.asarray(
            selected_variant["retainedFieldPixel"], dtype=np.int32
        )
        field_audit_count = int(selected_variant["fieldAuditCount"])
        native_arrays = selected_variant["nativeArrays"]
        native_stats = selected_variant["nativeStatistics"]
        mesh_stats = selected_variant["meshStatistics"]
        geometry_record = selected_variant["geometry"]

        trial_loop_count = dict(current_loop_count)
        trial_boundary_edge_count = predicted_boundary_edge_count
        if open_bay_mode:
            # K existing arc edges are replaced by one new mouth edge.  The
            # boundary array has K+1 vertices because the mouth is implicit.
            trial_boundary_edge_count -= len(boundary) - 2
        else:
            trial_loop_count["interiorHoleCount"] -= 1
            trial_loop_count["macroHoleCount"] -= int(target_was_macro)
            trial_boundary_edge_count -= len(boundary)
        trial_region = current_region_count

        current = trial
        current_incidence = trial_incidence
        predicted_loop_count = trial_loop_count
        predicted_boundary_edge_count = trial_boundary_edge_count
        accepted_added_edge.extend(
            (int(value[0]), int(value[1])) for value in added_edge
        )
        proposal_node.extend(int(value) for value in synthetic_node)
        completion_triangle.extend(int(value) for value in patch_triangle.ravel())
        completion_field_pixel.extend(int(value) for value in retained_pixel)
        completion_profile.extend(
            native_arrays["normalProfile"][:field_audit_count]
        )
        completion_physical.extend(
            float(value)
            for value in native_arrays["physicalScore"][:field_audit_count]
        )
        completion_correlation.extend(
            float(value)
            for value in native_arrays["profileCorrelation"][:field_audit_count]
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
        proposal_accepted.append(1)
        records.append(
            {
                "holeRow": row,
                "loopIndex": loop,
                "component": component_id,
                "accepted": True,
                "rejectionReasons": [],
                "selectedHypothesisIndex": selected_hypothesis_index,
                "selectedBoundarySeparationVoxels": selected_separation,
                "hypotheses": hypothesis_records,
                "depthFieldSupportedFraction": round(depth_support_fraction, 6),
                "depthFieldIntegrability": field_integrability[row],
                "mesh": mesh_stats,
                "geometry": geometry_record,
                "nativeCt": native_stats,
                "topology": {
                    "interiorHoleCountBefore": current_loop_count[
                        "interiorHoleCount"
                    ],
                    "interiorHoleCountAfter": trial_loop_count[
                        "interiorHoleCount"
                    ],
                    "macroHoleCountBefore": current_loop_count["macroHoleCount"],
                    "macroHoleCountAfter": trial_loop_count["macroHoleCount"],
                    "targetWasMacroEligible": target_was_macro,
                    "targetWasOpenBay": open_bay_mode,
                    "boundaryEdgeCountBefore": (
                        predicted_boundary_edge_count
                        + (len(boundary) - 2 if open_bay_mode else len(boundary))
                    ),
                    "boundaryEdgeCountAfter": trial_boundary_edge_count,
                    "componentTriangleRegionCountBefore": (
                        current_region_count.get(component_id, 0)
                    ),
                    "componentTriangleRegionCountAfter": (
                        trial_region.get(component_id, 0)
                    ),
                },
            }
        )

    accepted_count = int(np.count_nonzero(proposal_accepted))
    final_loops, final_loop_stats = extract_surface_boundary_loops(
        current, settings=hole_settings
    )
    final_loop_count = _loop_counts(final_loops)
    final_boundary_edge_count = int(
        len(np.asarray(final_loops["loopVertexFrontierIndex"]))
    )
    if final_loop_count != predicted_loop_count:
        raise ValueError(
            "local completion incidence and final whole-surface topology differ: "
            f"predicted={predicted_loop_count}, actual={final_loop_count}"
        )
    if final_boundary_edge_count != predicted_boundary_edge_count:
        raise ValueError(
            "local completion boundary-edge accounting and final whole-surface "
            "topology differ: "
            f"predicted={predicted_boundary_edge_count}, "
            f"actual={final_boundary_edge_count}"
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
    accepted_records = [record for record in records if record["accepted"]]
    selected_separation_count = Counter(
        str(record["selectedBoundarySeparationVoxels"])
        for record in accepted_records
    )
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
        "boundaryEdgeCountBefore": boundary_edge_count_before,
        "boundaryEdgeCountAfter": final_boundary_edge_count,
        "boundaryEdgeReduction": (
            boundary_edge_count_before - final_boundary_edge_count
        ),
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
        "adaptiveMeshHypotheses": not open_bay_mode,
        "openBayFullResolutionOnly": open_bay_mode,
        "openBayRanking": (
            "depth-field readiness, surface integrability, CT support, "
            "profile correlation, far-layer margin, then geometry"
            if open_bay_mode
            else None
        ),
        "textureGatedReplay": eligible_hole_rows is not None,
        "textureEligibleHoleRowCount": (
            len(eligible_hole_rows)
            if eligible_hole_rows is not None
            else None
        ),
        "evaluatedMeshHypothesisCount": int(
            sum(len(record.get("hypotheses", ())) for record in records)
        ),
        "fallbackAcceptedHoleCount": int(
            sum(
                int(record.get("selectedHypothesisIndex", 0) > 0)
                for record in accepted_records
            )
        ),
        "selectedBoundarySeparationCount": dict(
            sorted(selected_separation_count.items(), key=lambda item: float(item[0]))
        ),
        "finalBoundaryAudit": final_loop_stats,
        "decisionUnit": (
            "one complete outer-boundary arc and replacement mouth"
            if open_bay_mode
            else "one complete weakly-simple closed boundary and its collective "
            "CT depth field"
        ),
        "openBayMode": open_bay_mode,
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
    texture_audit_root: str | Path | None = None,
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
    eligible_hole_rows: frozenset[int] | None = None
    texture_audit_reference: dict[str, Any] | None = None
    texture_gate_statistics: dict[str, int] | None = None
    if texture_audit_root is not None:
        texture_path, texture_manifest = _resolve_texture_audit_manifest(
            texture_audit_root
        )
        surface_reference = texture_manifest.get("identity", {}).get("surface", {})
        completion_path_value = surface_reference.get("manifestPath")
        if not completion_path_value:
            raise ValueError("flattened audit does not identify its source completion")
        completion_path = Path(str(completion_path_value))
        if (
            not completion_path.is_file()
            or sha256_file(completion_path)
            != surface_reference.get("manifestSha256")
        ):
            raise ValueError("flattened audit source completion changed")
        completion_manifest = json.loads(completion_path.read_text())
        if (
            completion_manifest.get("schema")
            != PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA
            or completion_manifest.get("state") != "complete"
            or completion_manifest.get("data", {}).get("sha256")
            != surface_reference.get("dataSha256")
        ):
            raise ValueError("flattened audit source is not a dense completion")
        if completion_manifest.get("identity", {}).get("holes") != _reference(
            holes_path, holes_manifest
        ):
            raise ValueError("texture audit was not measured on these hole states")
        if completion_manifest.get("identity", {}).get("depthField") != _reference(
            depth_path, depth_manifest
        ):
            raise ValueError("texture audit was not measured on this depth field")
        if canonical_json_hash(
            completion_manifest.get("identity", {}).get("settings", {})
        ) != canonical_json_hash(resolved.record()):
            raise ValueError("texture audit completion settings differ")
        eligible_hole_rows = _texture_compatible_hole_rows(
            completion_manifest, texture_manifest
        )
        accepted_source = sum(
            int(bool(record.get("accepted")))
            for record in completion_manifest.get("completions", ())
        )
        texture_gate_statistics = {
            "textureSourceAcceptedHoleCount": accepted_source,
            "textureCompatibleHoleCount": len(eligible_hole_rows),
            "textureRejectedOrUnmeasuredHoleCount": (
                accepted_source - len(eligible_hole_rows)
            ),
        }
        texture_audit_reference = {
            "manifestPath": str(texture_path),
            "manifestSha256": sha256_file(texture_path),
            "sourceCompletion": surface_reference,
        }
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
        "textureAudit": texture_audit_reference,
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
        eligible_hole_rows=eligible_hole_rows,
    )
    if texture_gate_statistics is not None:
        statistics.update(texture_gate_statistics)
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
                "one complete outer-boundary arc and one replacement mouth"
                if statistics["openBayMode"]
                else "one complete closed boundary and collective dense CT "
                "normal-depth field"
            ),
            "boundaryGeometry": (
                "an open bay closes every inherited arc edge while leaving "
                "its one new mouth edge open"
                if statistics["openBayMode"]
                else "weakly-simple loops are decomposed at pinch vertices "
                "into exact edge-preserving disk cycles"
            ),
            "surfaceRepresentation": (
                "constrained intrinsic triangles through dense CT field "
                "samples; ribbon-bank nodes are boundary and collision "
                "evidence only"
            ),
            "adaptiveMeshDensity": (
                "open bays retain the full depth-field resolution; a coarser "
                "retry cannot hide a non-integrable or unsupported frontier "
                "expansion"
                if statistics["openBayMode"]
                else "each full boundary is tested densest-first at declared "
                "boundary-separation scales; the first complete mesh whose "
                "entire realized area passes native CT and exact topology is "
                "selected, rather than growing individual pixels or cells"
            ),
            "depthFieldIntegrability": (
                "complete 2x2 raster cells test whether pointwise-supported "
                "depth assignments form a realizable surface and rank open "
                "bays before constrained meshing"
                if statistics["openBayMode"]
                else "reported as a diagnostic; closed-hole ordering and "
                "adaptive reconstruction remain unchanged"
            ),
            "nativeCtAudit": (
                "the realized triangles are re-sampled at retained dense "
                "field vertices and uniform subtriangle quadrature points "
                "using their mesh normals against boundary context and "
                "displaced competing layers"
            ),
            "defectHandling": (
                "fitted-normal tails and thickness-normalized proximity are "
                "reported diagnostics rather than vetoes; tight bends and "
                "compressed or delaminated sheets remain admissible when "
                "their realized triangles have direct CT support and do not "
                "literally intersect another surface"
            ),
            "topologyAudit": (
                "every inherited arc edge becomes exactly two-incident, the "
                "new mouth is exactly one-incident, total boundary length "
                "falls, loop counts remain fixed, and no triangle region is "
                "created"
                if statistics["openBayMode"]
                else "every prior boundary edge becomes exactly two-incident, "
                "no edge exceeds two faces, the target macro/interior loop "
                "disappears, and no triangle region is created"
            ),
            "openFrontierScale": (
                "the declared maximum triangle edge applies to CT-supported "
                "interior and attachment edges; the replacement mouth is an "
                "open frontier, is reported separately, and remains covered "
                "by uniform native-CT area quadrature"
                if statistics["openBayMode"]
                else "all completion edges use the declared maximum"
            ),
            "mutation": (
                "accepted completions augment a versioned surface artifact; "
                "source ribbon configuration and depth field remain unchanged"
            ),
            "textureGate": (
                "only proposal-local flattened-fiber-compatible hole rows "
                "from the audited source completion are reconstructed; all "
                "geometry, CT, collision, and topology gates are rerun"
                if texture_audit_reference is not None
                else "not applied in this proposal solve"
            ),
            "singlePixelGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
