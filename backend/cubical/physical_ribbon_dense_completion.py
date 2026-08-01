from __future__ import annotations

import heapq
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import AbstractSet, Any, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .flatten import ComponentMesh, conformal_chart
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
from .physical_ribbon_surface_corridors import (
    PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA,
    surface_corridor_completion_view,
)
from .surface_topology import triangle_edge_region_labels


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
                PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA,
            }
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "holes root must identify one complete patch, surface-hole, or "
            "open-bay, or surface-corridor artifact"
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
    verdict_prefix = (
        "nativeSeamFiber"
        if "nativeSeamFiberCompatibleCompletionHoleRows" in audit
        else "boundaryTexture"
    )
    compatible = {
        int(value)
        for value in audit.get(
            f"{verdict_prefix}CompatibleCompletionHoleRows", ()
        )
    }
    incompatible = {
        int(value)
        for value in audit.get(
            f"{verdict_prefix}IncompatibleCompletionHoleRows", ()
        )
    }
    unmeasured = {
        int(value)
        for value in audit.get(
            f"{verdict_prefix}UnmeasuredCompletionHoleRows", ()
        )
    }
    if compatible & incompatible or compatible & unmeasured or incompatible & unmeasured:
        raise ValueError("completion fiber-texture verdicts overlap")
    if compatible | incompatible | unmeasured != accepted:
        raise ValueError(
            "completion fiber-texture verdicts do not exhaust accepted rows"
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
    minimum_dense_triangle_shape_ratio: float = 0.05
    maximum_triangle_edge_voxels: float = 6.0
    high_triangle_normal_residual_degrees: float = 45.0
    maximum_triangle_normal_residual_degrees: float = 85.0
    maximum_high_normal_residual_area_fraction: float = 0.25
    maximum_native_ct_quadrature_edge_voxels: float = 1.0
    intersection_tolerance_voxels: float = 0.05
    attachment_collar_transition_voxels: float = 1.0
    attachment_collar_outward_tangent_ratio_hypotheses: tuple[float, ...] = (
        0.20,
        0.40,
        0.60,
        0.80,
        1.00,
    )
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
            self.maximum_high_normal_residual_area_fraction,
        )
        if any(
            not math.isfinite(value) or not 0.0 <= value <= 1.0
            for value in fractions
        ):
            raise ValueError("dense-completion fractions must lie in [0, 1]")
        if not -1.0 <= self.minimum_median_profile_correlation <= 1.0:
            raise ValueError("profile-correlation gate must lie in [-1, 1]")
        if (
            not math.isfinite(self.minimum_dense_triangle_shape_ratio)
            or not 0.0 < self.minimum_dense_triangle_shape_ratio <= 1.0
        ):
            raise ValueError(
                "dense-triangle shape ratio must lie in (0, 1]"
            )
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
        if (
            not math.isfinite(self.attachment_collar_transition_voxels)
            or self.attachment_collar_transition_voxels < 0.0
        ):
            raise ValueError("attachment collar transition must be nonnegative")
        tangent_ratios = tuple(
            float(value)
            for value in self.attachment_collar_outward_tangent_ratio_hypotheses
        )
        if (
            not tangent_ratios
            or any(
                not math.isfinite(value) or value <= 0.0
                for value in tangent_ratios
            )
            or tuple(sorted(set(tangent_ratios))) != tangent_ratios
        ):
            raise ValueError(
                "attachment-collar tangent ratios must be unique, positive, "
                "and sorted"
            )
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
    minimum_triangle_area_voxels_squared: float = 0.0,
    maximum_triangle_edge_voxels: float = math.inf,
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
        replacement_geometry = [
            _triangle_geometry(
                triangle, midpoint_xyz, reference_normal_xyz
            )
            for triangle in replacement
        ]
        if any(
            area < minimum_triangle_area_voxels_squared
            or maximum_edge > maximum_triangle_edge_voxels
            for _, area, _, maximum_edge in replacement_geometry
        ):
            continue
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


def triangulate_two_frontier_strip_field(
    boundary: np.ndarray,
    boundary_uv: np.ndarray,
    field_uv: np.ndarray,
    field_coordinates: np.ndarray,
    *,
    first_arc_edge_count: int,
    second_arc_edge_count: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build one structured strip with exact side arcs and sampled mouths.

    The dense raster supplies every interior column, including the two end
    rows.  Its outer columns are replaced by the inherited mesh arcs.  This
    avoids a single large cap triangle at either open mouth while retaining
    the proposal as one complete, jointly audited state.
    """

    boundary = np.asarray(boundary, dtype=np.int32)
    boundary_uv = np.asarray(boundary_uv, dtype=np.float64)
    field_uv = np.asarray(field_uv, dtype=np.float64)
    coordinate = np.asarray(field_coordinates, dtype=np.int32)
    first_vertex_count = int(first_arc_edge_count) + 1
    second_vertex_count = int(second_arc_edge_count) + 1
    if first_vertex_count + second_vertex_count != len(boundary):
        raise ValueError("structured corridor arc counts differ from its boundary")
    if len(np.unique(boundary)) != len(boundary):
        raise ValueError(
            "structured corridor cannot reuse a vertex across its two arcs"
        )
    if len(field_uv) != len(coordinate) or not len(field_uv):
        raise ValueError("structured corridor field coordinates differ")
    minimum = np.min(coordinate, axis=0)
    maximum = np.max(coordinate, axis=0)
    column_values = np.arange(int(minimum[0]), int(maximum[0]) + 1)
    row_values = np.arange(int(minimum[1]), int(maximum[1]) + 1)
    lookup = {
        (int(value[0]), int(value[1])): index
        for index, value in enumerate(coordinate)
    }
    if len(lookup) != len(coordinate) or len(lookup) != len(column_values) * len(
        row_values
    ):
        raise ValueError("structured corridor raster is not one complete rectangle")
    interior_columns = column_values[1:-1]
    if not len(interior_columns):
        raise ValueError("structured corridor needs an interior raster column")
    retained_pixel = np.asarray(
        [
            lookup[(int(column), int(row))]
            for row in row_values
            for column in interior_columns
        ],
        dtype=np.int32,
    )
    unique_boundary_count = len(boundary)
    field_local = {
        int(pixel): unique_boundary_count + index
        for index, pixel in enumerate(retained_pixel)
    }
    point_uv = np.vstack((boundary_uv, field_uv[retained_pixel]))
    first_arc = np.arange(first_vertex_count, dtype=np.int32)
    second_arc = np.arange(
        first_vertex_count,
        first_vertex_count + second_vertex_count,
        dtype=np.int32,
    )[::-1]

    def rail(column: int) -> np.ndarray:
        return np.asarray(
            [field_local[lookup[(int(column), int(row))]] for row in row_values],
            dtype=np.int32,
        )

    def zipper(first: np.ndarray, second: np.ndarray) -> list[tuple[int, int, int]]:
        first_position = 0
        second_position = 0
        result: list[tuple[int, int, int]] = []
        while (
            first_position < len(first) - 1
            or second_position < len(second) - 1
        ):
            if first_position == len(first) - 1:
                advance_first = False
            elif second_position == len(second) - 1:
                advance_first = True
            else:
                first_next = float(point_uv[first[first_position + 1], 1])
                second_next = float(point_uv[second[second_position + 1], 1])
                advance_first = first_next <= second_next
            if advance_first:
                triangle = (
                    int(first[first_position]),
                    int(first[first_position + 1]),
                    int(second[second_position]),
                )
                first_position += 1
            else:
                triangle = (
                    int(first[first_position]),
                    int(second[second_position + 1]),
                    int(second[second_position]),
                )
                second_position += 1
            result.append(_orient_chart_triangle(triangle, point_uv))
        return result

    left_rail = rail(int(interior_columns[0]))
    right_rail = rail(int(interior_columns[-1]))
    triangles: list[tuple[int, int, int]] = []
    triangles.extend(zipper(first_arc, left_rail))
    if len(interior_columns) > 1:
        for row_index in range(len(row_values) - 1):
            for column_index in range(len(interior_columns) - 1):
                lower_left = field_local[
                    lookup[
                        (
                            int(interior_columns[column_index]),
                            int(row_values[row_index]),
                        )
                    ]
                ]
                lower_right = field_local[
                    lookup[
                        (
                            int(interior_columns[column_index + 1]),
                            int(row_values[row_index]),
                        )
                    ]
                ]
                upper_left = field_local[
                    lookup[
                        (
                            int(interior_columns[column_index]),
                            int(row_values[row_index + 1]),
                        )
                    ]
                ]
                upper_right = field_local[
                    lookup[
                        (
                            int(interior_columns[column_index + 1]),
                            int(row_values[row_index + 1]),
                        )
                    ]
                ]
                triangles.extend(
                    (
                        _orient_chart_triangle(
                            (lower_left, lower_right, upper_right), point_uv
                        ),
                        _orient_chart_triangle(
                            (lower_left, upper_right, upper_left), point_uv
                        ),
                    )
                )
    triangles.extend(zipper(right_rail, second_arc))

    bottom_rail = np.asarray(
        [
            field_local[lookup[(int(column), int(row_values[0]))]]
            for column in interior_columns
        ],
        dtype=np.int32,
    )
    top_rail = np.asarray(
        [
            field_local[lookup[(int(column), int(row_values[-1]))]]
            for column in interior_columns
        ],
        dtype=np.int32,
    )
    mouth_walks = (
        np.concatenate(([first_arc[0]], bottom_rail, [second_arc[0]])),
        np.concatenate(([first_arc[-1]], top_rail, [second_arc[-1]])),
    )
    new_frontier_local_edge = np.asarray(
        [
            (int(first), int(second))
            for walk in mouth_walks
            for first, second in zip(walk[:-1], walk[1:])
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    attachment_local_edge = np.asarray(
        [
            (int(first), int(second))
            for walk in (first_arc, second_arc)
            for first, second in zip(walk[:-1], walk[1:])
        ],
        dtype=np.int32,
    ).reshape((-1, 2))
    triangle_array = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    incidence = _edge_incidence(triangle_array)
    actual_boundary = {
        edge for edge, count in incidence.items() if count == 1
    }
    expected_boundary = {
        tuple(sorted((int(first), int(second))))
        for first, second in np.vstack(
            (attachment_local_edge, new_frontier_local_edge)
        )
    }
    if actual_boundary != expected_boundary or any(
        count > 2 for count in incidence.values()
    ):
        raise ValueError("structured corridor does not preserve its exact frontier")
    point_kind = np.concatenate(
        (
            np.zeros(len(boundary), dtype=np.uint8),
            np.ones(len(retained_pixel), dtype=np.uint8),
        )
    )
    point_source = np.concatenate((boundary, retained_pixel)).astype(np.int32)
    return {
        "pointUV": point_uv.astype(np.float32),
        "pointKind": point_kind,
        "pointSourceIndex": point_source,
        "trianglePointIndex": triangle_array,
        "retainedFieldPixel": retained_pixel,
        "fieldBoundaryDistanceVoxels": np.zeros(len(field_uv), dtype=np.float32),
        "newFrontierLocalEdge": new_frontier_local_edge,
    }, {
        "boundaryWalkVertexCount": int(len(boundary)),
        "uniqueBoundaryVertexCount": int(len(boundary)),
        "pinchCycleCount": 1,
        "pinchVertexReuseCount": 0,
        "fieldPixelCount": int(len(field_uv)),
        "retainedFieldPixelCount": int(len(retained_pixel)),
        "retainedFieldPixelFraction": round(
            float(len(retained_pixel)) / max(len(field_uv), 1), 6
        ),
        "triangleCount": int(len(triangle_array)),
        "chartEdgeFlipIterations": 0,
        "exactBoundaryPreserved": True,
        "structuredTwoFrontierStrip": True,
        "newFrontierEdgeCount": int(len(new_frontier_local_edge)),
    }


def triangulate_mixed_boundary_field(
    boundary_kind: np.ndarray,
    boundary_source_index: np.ndarray,
    boundary_uv: np.ndarray,
    field_uv: np.ndarray,
    *,
    new_frontier_boundary_edge: np.ndarray,
    minimum_boundary_separation: float,
    maximum_edge_flip_iterations: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Triangulate one disk whose frontier mixes inherited and CT samples.

    Ordinary hole completion has an entirely inherited boundary, while an
    open surface sector can have a densely sampled new mouth.  ``kind == 0``
    addresses an existing surface node and ``kind == 1`` addresses a pixel in
    the supplied CT field.  Every boundary decision is still made as one disk;
    synthetic mouth vertices are not independent growth candidates.
    """

    kind = np.asarray(boundary_kind, dtype=np.uint8)
    source = np.asarray(boundary_source_index, dtype=np.int32)
    polygon = np.asarray(boundary_uv, dtype=np.float64)
    field = np.asarray(field_uv, dtype=np.float64)
    mouth = np.asarray(new_frontier_boundary_edge, dtype=np.int32).reshape(
        (-1, 2)
    )
    if (
        len(kind) != len(source)
        or len(kind) != len(polygon)
        or len(kind) < 3
    ):
        raise ValueError("mixed completion boundary fields differ")
    if np.any((kind != 0) & (kind != 1)):
        raise ValueError("mixed completion boundary has an unknown point kind")
    if len({(int(k), int(s)) for k, s in zip(kind, source)}) != len(kind):
        raise ValueError("mixed completion boundary repeats a source point")
    if np.any(source < 0):
        raise ValueError("mixed completion boundary has a negative source index")
    if np.any(kind == 1) and int(np.max(source[kind == 1])) >= len(field):
        raise ValueError("mixed completion mouth references a missing CT pixel")
    if abs(_signed_polygon_area(polygon)) <= 1.0e-8:
        raise ValueError("mixed completion boundary has no intrinsic area")

    boundary_local_edge = {
        tuple(sorted((index, (index + 1) % len(kind))))
        for index in range(len(kind))
    }
    normalized_mouth = {
        tuple(sorted((int(first), int(second)))) for first, second in mouth
    }
    if not normalized_mouth or not normalized_mouth.issubset(
        boundary_local_edge
    ):
        raise ValueError("mixed completion mouth is not one exact boundary arc")
    mouth_degree: Counter[int] = Counter(
        value for edge in normalized_mouth for value in edge
    )
    if sum(value == 1 for value in mouth_degree.values()) != 2 or any(
        value > 2 for value in mouth_degree.values()
    ):
        raise ValueError("mixed completion mouth is not one simple edge path")

    boundary_distance = np.asarray(
        [
            min(
                _point_segment_distance(
                    point,
                    polygon[index],
                    polygon[(index + 1) % len(polygon)],
                )
                for index in range(len(polygon))
            )
            for point in field
        ],
        dtype=np.float64,
    )
    excluded = np.zeros(len(field), dtype=bool)
    excluded[source[kind == 1]] = True
    inside = np.asarray(
        [_point_in_polygon(point, polygon) for point in field], dtype=bool
    )
    interior_pixel = np.flatnonzero(
        inside
        & ~excluded
        & (boundary_distance >= minimum_boundary_separation)
    ).astype(np.int32)

    point_uv = np.vstack((polygon, field[interior_pixel]))
    point_kind = np.concatenate(
        (kind, np.ones(len(interior_pixel), dtype=np.uint8))
    )
    point_source = np.concatenate((source, interior_pixel)).astype(np.int32)
    triangles = _ear_clip_cycle(
        np.arange(len(polygon), dtype=np.int32), point_uv
    )
    for point_index in sorted(
        range(len(polygon), len(point_uv)),
        key=lambda index: (
            float(point_uv[index, 0]),
            float(point_uv[index, 1]),
            int(point_source[index]),
        ),
    ):
        triangles = _insert_chart_point(triangles, point_uv, point_index)
    triangles, flips = _improve_chart_triangulation(
        triangles,
        point_uv,
        boundary_local_edge,
        maximum_iterations=maximum_edge_flip_iterations,
    )
    triangle = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    incidence = _edge_incidence(triangle)
    actual_boundary = {
        edge for edge, count in incidence.items() if count == 1
    }
    if actual_boundary != boundary_local_edge or any(
        count > 2 for count in incidence.values()
    ):
        raise ValueError("mixed field mesh does not preserve its exact boundary")

    synthetic_pixel = point_source[point_kind == 1]
    return {
        "pointUV": point_uv.astype(np.float32),
        "pointKind": point_kind,
        "pointSourceIndex": point_source,
        "trianglePointIndex": triangle,
        "retainedFieldPixel": synthetic_pixel.astype(np.int32),
        "fieldBoundaryDistanceVoxels": boundary_distance.astype(np.float32),
        "newFrontierLocalEdge": mouth,
    }, {
        "boundaryWalkVertexCount": int(len(polygon)),
        "uniqueBoundaryVertexCount": int(len(polygon)),
        "inheritedBoundaryVertexCount": int(np.count_nonzero(kind == 0)),
        "syntheticMouthVertexCount": int(np.count_nonzero(kind == 1)),
        "fieldPixelCount": int(len(field)),
        "retainedInteriorFieldPixelCount": int(len(interior_pixel)),
        "retainedFieldPixelCount": int(len(synthetic_pixel)),
        "retainedFieldPixelFraction": round(
            float(len(synthetic_pixel)) / max(len(field), 1), 6
        ),
        "triangleCount": int(len(triangle)),
        "chartEdgeFlipIterations": int(flips),
        "exactBoundaryPreserved": True,
        "mixedInheritedSyntheticBoundary": True,
        "newFrontierEdgeCount": int(len(mouth)),
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


def _weld_surface_node_pairs(
    surface: Mapping[str, np.ndarray],
    pairs: Sequence[tuple[int, int]],
    *,
    coordinate_tolerance: float = 1.0e-5,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Undo selected topology-normalization splits without moving geometry.

    The first node of each pair is replaced by the second in every triangle.
    A weld is legal only for nodes carrying the same physical weld-group and
    the same coordinates.  The abandoned node record remains address-stable
    but is deselected and removed from the exact surface edge graph.
    """

    result = {key: np.asarray(value).copy() for key, value in surface.items()}
    xyz = np.asarray(result["midpointXYZ"], dtype=np.float64)
    weld_group = np.asarray(
        result.get("surfaceNodeWeldGroup", np.arange(len(xyz))),
        dtype=np.int64,
    )
    replacement: dict[int, int] = {}
    for source, target in pairs:
        first, second = int(source), int(target)
        if first == second:
            continue
        if not (0 <= first < len(xyz) and 0 <= second < len(xyz)):
            raise ValueError("surface weld references a missing node")
        if int(weld_group[first]) != int(weld_group[second]):
            raise ValueError("surface weld joins different physical weld groups")
        if float(np.linalg.norm(xyz[first] - xyz[second])) > coordinate_tolerance:
            raise ValueError("surface weld moves a topology-normalized node")
        previous = replacement.get(first)
        if previous is not None and previous != second:
            raise ValueError("surface weld gives one node two targets")
        if second in replacement:
            raise ValueError("surface weld target is itself being replaced")
        replacement[first] = second
    if not replacement:
        raise ValueError("surface weld has no distinct node pair")

    triangle = np.asarray(result["triangleFrontierIndex"], dtype=np.int32).copy()
    referenced_before = set(int(value) for value in np.unique(triangle))
    for source, target in replacement.items():
        triangle[triangle == source] = target
    if np.any(
        (triangle[:, 0] == triangle[:, 1])
        | (triangle[:, 1] == triangle[:, 2])
        | (triangle[:, 2] == triangle[:, 0])
    ):
        raise ValueError("surface weld collapses an existing triangle")
    canonical_triangle = np.sort(triangle, axis=1)
    if len(np.unique(canonical_triangle, axis=0)) != len(canonical_triangle):
        raise ValueError("surface weld duplicates an existing triangle")
    incidence = _edge_incidence(triangle)
    if any(count > 2 for count in incidence.values()):
        raise ValueError("surface weld creates a non-manifold edge")
    result["triangleFrontierIndex"] = triangle

    referenced_after = set(int(value) for value in np.unique(triangle))
    orphan = sorted(
        source
        for source in replacement
        if source in referenced_before and source not in referenced_after
    )
    if "selected" in result:
        result["selected"][orphan] = 0
    if "component" in result:
        result["component"][orphan] = -1
    if "componentSize" in result:
        result["componentSize"][orphan] = 0

    edge = np.asarray(sorted(incidence), dtype=np.int32).reshape((-1, 2))
    result["edgeFirstFrontierIndex"] = edge[:, 0]
    result["edgeSecondFrontierIndex"] = edge[:, 1]
    result["edgeSelected"] = np.ones(len(edge), dtype=np.uint8)
    result["integrationResidualVoxels"] = np.zeros(
        len(edge), dtype=np.float32
    )
    if "component" in result and "componentSize" in result:
        component = np.asarray(result["component"], dtype=np.int32)
        present = component[component >= 0]
        sizes = (
            np.bincount(present).astype(np.int32)
            if len(present)
            else np.empty(0, dtype=np.int32)
        )
        node_size = np.zeros(len(component), dtype=np.int32)
        valid = component >= 0
        node_size[valid] = sizes[component[valid]]
        result["componentSize"] = node_size
    return result, {
        "weldPairCount": int(len(replacement)),
        "weldPairs": [
            {"replacedNode": int(first), "canonicalNode": int(second)}
            for first, second in sorted(replacement.items())
        ],
        "orphanedNodeCount": int(len(orphan)),
        "orphanedNodes": orphan,
        "geometryMoved": False,
        "triangleCountChanged": False,
        "triangleRegionCountChanged": False,
        "exactSurfaceEdgeCount": int(len(edge)),
    }


def _triangle_region_boundary_loop(
    surface: Mapping[str, np.ndarray], region_id: int
) -> np.ndarray:
    """Return the one simple open frontier of an edge-connected triangle island."""

    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    labels = _triangle_region_labels(triangle)
    selected = triangle[labels == int(region_id)]
    if not len(selected):
        raise ValueError("triangle island region is empty")
    incidence = _edge_incidence(selected)
    boundary = [edge for edge, count in incidence.items() if count == 1]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in boundary:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    if not adjacency or any(len(value) != 2 for value in adjacency.values()):
        raise ValueError("triangle island does not have one simple boundary cycle")
    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        options = sorted(value for value in adjacency[current] if value != previous)
        if not options:
            raise ValueError("triangle island boundary walk terminates early")
        following = options[0]
        if following == start:
            break
        if following in loop or len(loop) > len(adjacency):
            raise ValueError("triangle island has multiple boundary cycles")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("triangle island has multiple boundary cycles")
    return np.asarray(loop, dtype=np.int32)


def _closest_triangle_barycentric(
    point: np.ndarray, triangle: np.ndarray
) -> tuple[np.ndarray, float]:
    """Return barycentric coordinates of the closest point on one 3D triangle."""

    query = np.asarray(point, dtype=np.float64)
    first, second, third = np.asarray(triangle, dtype=np.float64)
    ab, ac, ap = second - first, third - first, query - first
    d1, d2 = float(np.dot(ab, ap)), float(np.dot(ac, ap))
    if d1 <= 0.0 and d2 <= 0.0:
        bary = np.asarray((1.0, 0.0, 0.0))
    else:
        bp = query - second
        d3, d4 = float(np.dot(ab, bp)), float(np.dot(ac, bp))
        if d3 >= 0.0 and d4 <= d3:
            bary = np.asarray((0.0, 1.0, 0.0))
        else:
            vc = d1 * d4 - d3 * d2
            if vc <= 0.0 and d1 >= 0.0 and d3 <= 0.0:
                value = d1 / max(d1 - d3, 1.0e-12)
                bary = np.asarray((1.0 - value, value, 0.0))
            else:
                cp = query - third
                d5, d6 = float(np.dot(ab, cp)), float(np.dot(ac, cp))
                if d6 >= 0.0 and d5 <= d6:
                    bary = np.asarray((0.0, 0.0, 1.0))
                else:
                    vb = d5 * d2 - d1 * d6
                    if vb <= 0.0 and d2 >= 0.0 and d6 <= 0.0:
                        value = d2 / max(d2 - d6, 1.0e-12)
                        bary = np.asarray((1.0 - value, 0.0, value))
                    else:
                        va = d3 * d6 - d5 * d4
                        if va <= 0.0 and (d4 - d3) >= 0.0 and (d5 - d6) >= 0.0:
                            value = (d4 - d3) / max(
                                (d4 - d3) + (d5 - d6), 1.0e-12
                            )
                            bary = np.asarray((0.0, 1.0 - value, value))
                        else:
                            denominator = max(va + vb + vc, 1.0e-12)
                            value_b = vb / denominator
                            value_c = vc / denominator
                            bary = np.asarray(
                                (1.0 - value_b - value_c, value_b, value_c)
                            )
    closest = bary @ np.asarray(triangle, dtype=np.float64)
    return bary, float(np.linalg.norm(query - closest))


def _project_nodes_to_triangle_parameter(
    point_xyz: np.ndarray,
    mesh_xyz: np.ndarray,
    mesh_triangle: np.ndarray,
    mesh_uv: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Project complete surface-island vertices into a realized strip chart."""

    triangle = np.asarray(mesh_triangle, dtype=np.int32)
    triangle_xyz = np.asarray(mesh_xyz, dtype=np.float64)[triangle]
    triangle_uv = np.asarray(mesh_uv, dtype=np.float64)[triangle]
    projected: list[np.ndarray] = []
    distance: list[float] = []
    for point in np.asarray(point_xyz, dtype=np.float64):
        best_distance = math.inf
        best_uv: np.ndarray | None = None
        for xyz, uv in zip(triangle_xyz, triangle_uv):
            barycentric, value = _closest_triangle_barycentric(point, xyz)
            if value < best_distance:
                best_distance = value
                best_uv = barycentric @ uv
        if best_uv is None:
            raise ValueError("cannot project an island into an empty strip mesh")
        projected.append(best_uv)
        distance.append(best_distance)
    return (
        np.asarray(projected, dtype=np.float32),
        np.asarray(distance, dtype=np.float32),
    )


def _cyclic_paths_between(
    loop: np.ndarray, first: int, second: int
) -> tuple[np.ndarray, np.ndarray]:
    values = [int(value) for value in np.asarray(loop, dtype=np.int32)]
    if values.count(int(first)) != 1 or values.count(int(second)) != 1:
        raise ValueError("island attachment endpoint is not unique on its boundary")
    first_position, second_position = values.index(int(first)), values.index(int(second))
    forward: list[int] = [int(first)]
    position = first_position
    while position != second_position:
        position = (position + 1) % len(values)
        forward.append(values[position])
    backward: list[int] = [int(first)]
    position = first_position
    while position != second_position:
        position = (position - 1) % len(values)
        backward.append(values[position])
    if set(forward[1:-1]) & set(backward[1:-1]):
        raise ValueError("island boundary paths overlap away from their endpoints")
    return np.asarray(forward, dtype=np.int32), np.asarray(backward, dtype=np.int32)


def _mesh_edge_length_audit(
    triangles: np.ndarray,
    point_xyz: np.ndarray,
    point_kind: np.ndarray,
    point_source_index: np.ndarray,
    new_frontier_edges: AbstractSet[tuple[int, int]],
    *,
    new_frontier_local_edges: AbstractSet[tuple[int, int]] = frozenset(),
    existing_surface_edges: AbstractSet[tuple[int, int]] = frozenset(),
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
    normalized_local_frontier = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in new_frontier_local_edges
    }
    normalized_existing = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in existing_surface_edges
    }
    unique_edges = {
        tuple(sorted((int(triangle[index]), int(triangle[(index + 1) % 3]))))
        for triangle in values
        for index in range(3)
    }
    all_length: list[float] = []
    frontier_length: list[float] = []
    supported_length: list[float] = []
    inherited_length: list[float] = []
    for first, second in unique_edges:
        length = float(np.linalg.norm(xyz[second] - xyz[first]))
        all_length.append(length)
        global_edge = (
            tuple(sorted((int(source[first]), int(source[second]))))
            if kind[first] == 0 and kind[second] == 0
            else None
        )
        if (
            (first, second) in normalized_local_frontier
            or global_edge in normalized_frontier
        ):
            frontier_length.append(length)
        elif global_edge in normalized_existing:
            inherited_length.append(length)
        else:
            supported_length.append(length)
    return {
        "maximumTriangleEdgeVoxels": max(all_length, default=0.0),
        "maximumCtSupportedTriangleEdgeVoxels": max(
            supported_length, default=0.0
        ),
        "maximumOpenFrontierEdgeVoxels": max(frontier_length, default=0.0),
        "maximumInheritedBoundaryEdgeVoxels": max(
            inherited_length, default=0.0
        ),
    }


def _triangle_region_labels(triangles: np.ndarray) -> np.ndarray:
    return triangle_edge_region_labels(triangles)


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


def _boundary_triangle_region_catalog(
    surface: Mapping[str, np.ndarray],
) -> dict[tuple[int, int], int]:
    """Map each exactly open mesh edge to its current triangle region."""

    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    region = _triangle_region_labels(triangles)
    edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for index in range(3):
            edge = tuple(
                sorted(
                    (
                        int(triangle[index]),
                        int(triangle[(index + 1) % 3]),
                    )
                )
            )
            edge_triangle[edge].append(triangle_index)
    return {
        edge: int(region[indices[0]])
        for edge, indices in edge_triangle.items()
        if len(indices) == 1
    }


def _corridor_region_merge_audit(
    patch_triangle: np.ndarray,
    attachment_edges: AbstractSet[tuple[int, int]],
    boundary_triangle_region: Mapping[tuple[int, int], int],
    *,
    required_region_reduction: int = 1,
) -> dict[str, Any]:
    """Prove that one connected patch realizes its declared region merge."""

    if required_region_reduction < 1:
        raise ValueError("corridor region reduction must be positive")

    triangles = np.asarray(patch_triangle, dtype=np.int32).reshape((-1, 3))
    patch_region = _triangle_region_labels(triangles)
    patch_edge_region: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triangle_index, triangle in enumerate(triangles):
        for index in range(3):
            edge = tuple(
                sorted(
                    (
                        int(triangle[index]),
                        int(triangle[(index + 1) % 3]),
                    )
                )
            )
            patch_edge_region[edge].add(int(patch_region[triangle_index]))
    missing = 0
    baseline_regions: set[int] = set()
    attached_patch_regions: set[int] = set()
    for edge in attachment_edges:
        baseline = boundary_triangle_region.get(edge)
        patch_values = patch_edge_region.get(edge, set())
        if baseline is None or len(patch_values) != 1:
            missing += 1
            continue
        baseline_regions.add(int(baseline))
        attached_patch_regions.update(patch_values)
    patch_region_count = int(len(np.unique(patch_region))) if len(triangles) else 0
    reduction = (
        len(baseline_regions) - 1
        if patch_region_count == 1
        and attached_patch_regions == {0}
        and not missing
        else 0
    )
    exact = bool(
        patch_region_count == 1
        and attached_patch_regions == {0}
        and len(baseline_regions) == required_region_reduction + 1
        and reduction == required_region_reduction
        and not missing
    )
    return {
        "patchTriangleRegionCount": patch_region_count,
        "attachedBaselineTriangleRegionCount": len(baseline_regions),
        "attachedBaselineTriangleRegionIds": sorted(baseline_regions),
        "unresolvedAttachmentRegionEdgeCount": missing,
        "triangleRegionReduction": reduction,
        "requiredTriangleRegionReduction": required_region_reduction,
        "mergesRequiredTriangleRegions": exact,
        "mergesExactlyTwoTriangleRegions": bool(
            exact and required_region_reduction == 1
        ),
    }


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


def _rigid_chart_alignment(
    source_uv: np.ndarray,
    target_uv: np.ndarray,
    *,
    reflected: bool,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Fit one scale-preserving 2D gauge transform."""

    source = np.asarray(source_uv, dtype=np.float64)
    target = np.asarray(target_uv, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 2:
        raise ValueError("chart alignment point sets differ")
    if len(source) < 2:
        raise ValueError("chart alignment requires a multi-vertex arc")
    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    first, _singular, second = np.linalg.svd(
        source_centered.T @ target_centered
    )
    transform = first @ second
    wants_negative = bool(reflected)
    if (float(np.linalg.det(transform)) < 0.0) != wants_negative:
        first[:, -1] *= -1.0
        transform = first @ second
    translation = target_center - source_center @ transform
    residual = source @ transform + translation - target
    rms = math.sqrt(float(np.mean(np.sum(residual * residual, axis=1))))
    return transform, translation, rms


def _interpolate_chart_arc(
    parameter: np.ndarray,
    chart_uv: np.ndarray,
    query: np.ndarray,
) -> np.ndarray:
    order = np.argsort(parameter, kind="stable")
    position = np.asarray(parameter, dtype=np.float64)[order]
    values = np.asarray(chart_uv, dtype=np.float64)[order]
    unique, unique_index = np.unique(position, return_index=True)
    if len(unique) < 2:
        raise ValueError("corridor chart arc has no parameter span")
    values = values[unique_index]
    query_value = np.asarray(query, dtype=np.float64)
    return np.column_stack(
        (
            np.interp(query_value, unique, values[:, 0]),
            np.interp(query_value, unique, values[:, 1]),
        )
    )


def _conformal_joined_chart_option(
    current: Mapping[str, np.ndarray],
    trial: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    region_triangles: list[np.ndarray],
    region_nodes: list[np.ndarray],
    *,
    component_id: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Reparameterize only the edge-connected region created by a strip."""

    if len(region_triangles) < 2 or len(region_triangles) != len(region_nodes):
        raise ValueError("joined chart needs matching prior triangle regions")
    joined_triangle = np.vstack(
        tuple(np.asarray(value, dtype=np.int32) for value in region_triangles)
        + (np.asarray(patch_triangle, dtype=np.int32),)
    )
    joined_node = np.unique(joined_triangle)
    local_triangle = np.searchsorted(joined_node, joined_triangle).astype(
        np.int32
    )
    xyz = np.asarray(trial["midpointXYZ"], dtype=np.float64)[joined_node]
    reference = np.asarray(trial["signedNormalXYZ"], dtype=np.float64)[
        joined_node
    ]
    triangle_point = xyz[local_triangle]
    triangle_normal = np.cross(
        triangle_point[:, 1] - triangle_point[:, 0],
        triangle_point[:, 2] - triangle_point[:, 0],
    )
    triangle_normal /= np.maximum(
        np.linalg.norm(triangle_normal, axis=1, keepdims=True), 1.0e-12
    )
    triangle_reference = np.mean(reference[local_triangle], axis=1)
    triangle_normal[
        np.einsum("ij,ij->i", triangle_normal, triangle_reference) < 0.0
    ] *= -1.0
    mesh = ComponentMesh(
        component_id=component_id,
        patch_ids=(component_id,),
        vertex_xyz=xyz,
        polygons=(),
        polygon_patch_ids=np.empty(0, dtype=np.uint64),
        triangles=local_triangle,
        triangle_patch_ids=np.full(
            len(local_triangle), component_id + 1, dtype=np.uint64
        ),
        triangle_normal_xyz=triangle_normal,
        statistics={},
    )
    try:
        chart = conformal_chart(mesh)
    except (ValueError, np.linalg.LinAlgError) as error:
        return None, {
            "mode": "conformal-joined-region",
            "failure": str(error),
            "evaluatedGaugeCount": 0,
        }
    if int(chart.statistics.get("chartPieces", 0)) != 1:
        return None, {
            "mode": "conformal-joined-region",
            "failure": "joined strip chart is not edge connected",
            "chart": chart.statistics,
            "evaluatedGaugeCount": 0,
        }

    component = np.asarray(current["component"], dtype=np.int32)
    baseline_triangle = np.asarray(
        current["triangleFrontierIndex"], dtype=np.int32
    )
    same_component_triangle = baseline_triangle[
        np.all(component[baseline_triangle] == component_id, axis=1)
    ]
    stationary_triangle = same_component_triangle[
        ~np.isin(same_component_triangle, np.concatenate(region_nodes)).any(
            axis=1
        )
    ]
    old_chart = np.asarray(current["chartUV"], dtype=np.float64)
    options: list[tuple[tuple[float | int, ...], dict[str, Any]]] = []
    for anchor_index in range(len(region_nodes)):
        anchor_node = np.asarray(region_nodes[anchor_index], dtype=np.int32)
        anchor_local = np.searchsorted(joined_node, anchor_node)
        for reflected in (False, True):
            transform, translation, rms = _rigid_chart_alignment(
                chart.uv[anchor_local],
                old_chart[anchor_node],
                reflected=reflected,
            )
            joined_uv = chart.uv @ transform + translation
            trial_uv = np.asarray(trial["chartUV"], dtype=np.float64).copy()
            trial_uv[joined_node] = joined_uv
            self_overlap = _chart_overlap_count(
                joined_triangle,
                joined_triangle,
                trial_uv,
            )
            external_overlap = _chart_overlap_count(
                stationary_triangle,
                joined_triangle,
                trial_uv,
            )
            triangle_uv = trial_uv[joined_triangle]
            signed_area = 0.5 * (
                (triangle_uv[:, 1, 0] - triangle_uv[:, 0, 0])
                * (triangle_uv[:, 2, 1] - triangle_uv[:, 0, 1])
                - (triangle_uv[:, 1, 1] - triangle_uv[:, 0, 1])
                * (triangle_uv[:, 2, 0] - triangle_uv[:, 0, 0])
            )
            collapsed = int(np.count_nonzero(np.abs(signed_area) <= 1.0e-8))
            overlap = int(self_overlap + external_overlap)
            options.append(
                (
                    (
                        overlap,
                        collapsed,
                        int(chart.statistics.get("flippedTriangles", 0)),
                        round(rms, 9),
                        -len(anchor_node),
                        anchor_index,
                        int(reflected),
                    ),
                    {
                        "chartUV": trial_uv.astype(np.float32),
                        "mode": "conformal-joined-region",
                        "anchorArc": anchor_index,
                        "anchorReflection": reflected,
                        "anchorAlignmentRmsVoxels": round(rms, 6),
                        "joinedRegionNodeCount": int(len(joined_node)),
                        "joinedRegionTriangleCount": int(len(joined_triangle)),
                        "joinedRegionSelfOverlapCount": int(self_overlap),
                        "stationaryRegionOverlapCount": int(external_overlap),
                        "collapsedJoinedTriangleCount": collapsed,
                        "totalOverlapCount": overlap,
                        "chart": chart.statistics,
                    },
                )
            )
    _key, best = min(options, key=lambda value: value[0])
    statistics = {
        key: value for key, value in best.items() if key != "chartUV"
    }
    statistics["evaluatedGaugeCount"] = len(options)
    statistics["existingRegionScaleWarped"] = True
    if (
        int(best["totalOverlapCount"])
        or int(best["collapsedJoinedTriangleCount"])
        or int(best["chart"].get("flippedTriangles", 0))
    ):
        return None, statistics
    result = {key: np.asarray(value) for key, value in trial.items()}
    result["chartUV"] = np.asarray(best["chartUV"], dtype=np.float32)
    return result, statistics


def _single_disk_boundary_loop(triangles: np.ndarray) -> tuple[np.ndarray, dict[str, int]]:
    """Prove disk topology and return its one ordered boundary cycle."""

    triangle = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    incidence = _edge_incidence(triangle)
    if any(count > 2 for count in incidence.values()):
        raise ValueError("joined chart contains a non-manifold edge")
    boundary_edge = [edge for edge, count in incidence.items() if count == 1]
    adjacency: dict[int, list[int]] = defaultdict(list)
    for first, second in boundary_edge:
        adjacency[int(first)].append(int(second))
        adjacency[int(second)].append(int(first))
    if not adjacency or any(len(value) != 2 for value in adjacency.values()):
        raise ValueError("joined chart does not have one simple boundary")
    start = min(adjacency)
    loop = [start]
    previous = -1
    current = start
    while True:
        following_values = sorted(
            value for value in adjacency[current] if value != previous
        )
        if not following_values:
            raise ValueError("joined chart boundary terminates early")
        following = following_values[0]
        if following == start:
            break
        if following in loop or len(loop) > len(adjacency):
            raise ValueError("joined chart boundary is not one cycle")
        loop.append(following)
        previous, current = current, following
    if len(loop) != len(adjacency):
        raise ValueError("joined chart has more than one boundary cycle")
    vertex_count = int(len(np.unique(triangle)))
    euler = vertex_count - len(incidence) + len(triangle)
    if euler != 1:
        raise ValueError("joined chart is not a topological disk")
    return np.asarray(loop, dtype=np.int32), {
        "vertexCount": vertex_count,
        "edgeCount": int(len(incidence)),
        "triangleCount": int(len(triangle)),
        "boundaryVertexCount": int(len(loop)),
        "eulerCharacteristic": int(euler),
    }


def _uniform_harmonic_disk_chart(
    xyz: np.ndarray, triangles: np.ndarray
) -> tuple[np.ndarray, dict[str, Any]]:
    """Embed a triangulated disk with a convex metric boundary.

    Uniform positive barycentric weights and a strictly convex boundary give
    an injective Tutte embedding.  This is an overlap-free fallback when a
    lower-distortion free-boundary conformal solve folds globally.
    """

    point = np.asarray(xyz, dtype=np.float64)
    triangle = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    boundary, topology = _single_disk_boundary_loop(triangle)
    edge_length = np.linalg.norm(
        point[np.roll(boundary, -1)] - point[boundary], axis=1
    )
    perimeter = float(np.sum(edge_length))
    if not math.isfinite(perimeter) or perimeter <= 1.0e-8:
        raise ValueError("joined chart boundary has no physical perimeter")
    cumulative = np.concatenate(
        (np.zeros(1), np.cumsum(edge_length[:-1], dtype=np.float64))
    )
    angle = 2.0 * math.pi * cumulative / perimeter
    radius = perimeter / (2.0 * math.pi)
    uv = np.zeros((len(point), 2), dtype=np.float64)
    uv[boundary] = radius * np.column_stack((np.cos(angle), np.sin(angle)))

    edge = set(_edge_incidence(triangle))
    neighbor: list[set[int]] = [set() for _ in range(len(point))]
    for first, second in edge:
        neighbor[int(first)].add(int(second))
        neighbor[int(second)].add(int(first))
    boundary_mask = np.zeros(len(point), dtype=bool)
    boundary_mask[boundary] = True
    interior = np.flatnonzero(~boundary_mask).astype(np.int32)
    interior_row = {int(node): row for row, node in enumerate(interior)}
    degree = np.asarray(
        [len(neighbor[int(node)]) for node in interior], dtype=np.float64
    )
    if np.any(degree <= 0.0):
        raise ValueError("joined chart contains an isolated interior vertex")
    interior_neighbor = [
        np.asarray(
            [
                interior_row[value]
                for value in neighbor[int(node)]
                if value in interior_row
            ],
            dtype=np.int32,
        )
        for node in interior
    ]
    right_hand = np.asarray(
        [
            np.sum(
                uv[
                    [
                        value
                        for value in neighbor[int(node)]
                        if boundary_mask[value]
                    ]
                ],
                axis=0,
            )
            for node in interior
        ],
        dtype=np.float64,
    ).reshape((-1, 2))

    def apply(values: np.ndarray) -> np.ndarray:
        result = degree * values
        for row, adjacent in enumerate(interior_neighbor):
            if len(adjacent):
                result[row] -= float(np.sum(values[adjacent]))
        return result

    iterations: list[int] = []
    residuals: list[float] = []
    for axis in range(2):
        right = right_hand[:, axis]
        solution = np.zeros(len(interior), dtype=np.float64)
        residual = right - apply(solution)
        preconditioned = residual / degree
        direction = preconditioned.copy()
        residual_product = float(np.dot(residual, preconditioned))
        tolerance = 1.0e-11 * max(float(np.linalg.norm(right)), 1.0)
        maximum_iterations = max(64, 20 * len(interior))
        used = 0
        for used in range(1, maximum_iterations + 1):
            applied = apply(direction)
            denominator = float(np.dot(direction, applied))
            if denominator <= 1.0e-20:
                break
            step = residual_product / denominator
            solution += step * direction
            residual -= step * applied
            residual_norm = float(np.linalg.norm(residual))
            if residual_norm <= tolerance:
                break
            new_preconditioned = residual / degree
            new_product = float(np.dot(residual, new_preconditioned))
            direction = new_preconditioned + (
                new_product / max(residual_product, 1.0e-30)
            ) * direction
            preconditioned = new_preconditioned
            residual_product = new_product
        final_residual = float(np.linalg.norm(right - apply(solution)))
        if final_residual > 1.0e-8 * max(float(np.linalg.norm(right)), 1.0):
            raise ValueError("harmonic joined chart did not converge")
        uv[interior, axis] = solution
        iterations.append(int(used))
        residuals.append(final_residual)

    signed_area = np.asarray(
        [
            0.5
            * _cross_2d(uv[value[0]], uv[value[1]], uv[value[2]])
            for value in triangle
        ],
        dtype=np.float64,
    )
    nonzero = np.abs(signed_area) > 1.0e-10
    if not np.all(nonzero):
        raise ValueError("harmonic joined chart collapses a triangle")
    return uv.astype(np.float32), {
        "solver": "convex-boundary uniform harmonic disk embedding",
        **topology,
        "boundaryPerimeterVoxels": round(perimeter, 6),
        "boundaryCircleRadiusVoxels": round(radius, 6),
        "conjugateGradientIterations": iterations,
        "conjugateGradientResidual": [round(value, 12) for value in residuals],
        # Triangle winding in the materialized 3D mesh is local-normal based
        # and need not be globally consistent through a hairpin.  Global chart
        # injectivity is therefore audited geometrically below, not inferred
        # from these input index orders.
        "positiveWindingTriangles": int(np.count_nonzero(signed_area > 0.0)),
        "negativeWindingTriangles": int(np.count_nonzero(signed_area < 0.0)),
        "collapsedTriangles": 0,
    }


def _harmonic_joined_chart_option(
    current: Mapping[str, np.ndarray],
    trial: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    region_triangles: list[np.ndarray],
    region_nodes: list[np.ndarray],
    *,
    component_id: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Find an injective convex-boundary chart for one multi-region disk."""

    joined_triangle = np.vstack(
        tuple(np.asarray(value, dtype=np.int32) for value in region_triangles)
        + (np.asarray(patch_triangle, dtype=np.int32),)
    )
    joined_node = np.unique(joined_triangle)
    local_triangle = np.searchsorted(joined_node, joined_triangle).astype(np.int32)
    try:
        harmonic_uv, harmonic_statistics = _uniform_harmonic_disk_chart(
            np.asarray(trial["midpointXYZ"], dtype=np.float64)[joined_node],
            local_triangle,
        )
    except ValueError as error:
        return None, {
            "mode": "harmonic-joined-region",
            "failure": str(error),
            "evaluatedGaugeCount": 0,
        }

    component = np.asarray(current["component"], dtype=np.int32)
    baseline_triangle = np.asarray(
        current["triangleFrontierIndex"], dtype=np.int32
    )
    same_component_triangle = baseline_triangle[
        np.all(component[baseline_triangle] == component_id, axis=1)
    ]
    stationary_triangle = same_component_triangle[
        ~np.isin(same_component_triangle, np.concatenate(region_nodes)).any(axis=1)
    ]
    joined_baseline_triangle_count = sum(
        len(value) for value in region_triangles
    )
    if (
        len(same_component_triangle) - len(stationary_triangle)
        != joined_baseline_triangle_count
    ):
        return None, {
            "mode": "harmonic-joined-region",
            "failure": (
                "joined chart regions share a vertex with another atlas page"
            ),
            "evaluatedGaugeCount": 0,
        }
    old_chart = np.asarray(current["chartUV"], dtype=np.float64)
    options: list[tuple[tuple[float | int, ...], dict[str, Any]]] = []
    for anchor_index, anchor_node in enumerate(region_nodes):
        anchor = np.asarray(anchor_node, dtype=np.int32)
        anchor_local = np.searchsorted(joined_node, anchor)
        for reflected in (False, True):
            transform, translation, rms = _rigid_chart_alignment(
                harmonic_uv[anchor_local],
                old_chart[anchor],
                reflected=reflected,
            )
            joined_uv = harmonic_uv @ transform + translation
            trial_uv = np.asarray(trial["chartUV"], dtype=np.float64).copy()
            trial_uv[joined_node] = joined_uv
            self_overlap = _chart_overlap_count(
                joined_triangle, joined_triangle, trial_uv
            )
            external_overlap = _chart_overlap_count(
                stationary_triangle, joined_triangle, trial_uv
            )
            overlap = int(self_overlap)
            options.append(
                (
                    (
                        overlap,
                        int(external_overlap),
                        round(rms, 9),
                        -len(anchor),
                        anchor_index,
                        int(reflected),
                    ),
                    {
                        "chartUV": trial_uv.astype(np.float32),
                        "mode": "harmonic-joined-region",
                        "anchorArc": anchor_index,
                        "anchorReflection": reflected,
                        "anchorAlignmentRmsVoxels": round(rms, 6),
                        "joinedRegionNodeCount": int(len(joined_node)),
                        "joinedRegionTriangleCount": int(len(joined_triangle)),
                        "joinedRegionSelfOverlapCount": int(self_overlap),
                        "stationaryRegionOverlapCount": int(external_overlap),
                        "crossAtlasOverlapCount": int(external_overlap),
                        "collapsedJoinedTriangleCount": 0,
                        "totalOverlapCount": overlap,
                        "atlasPageScope": "edge-connected-triangle-region",
                        "chart": harmonic_statistics,
                    },
                )
            )
    if not options:
        return None, {
            "mode": "harmonic-joined-region",
            "failure": "harmonic chart has no prior-region gauge anchor",
            "evaluatedGaugeCount": 0,
        }
    _key, best = min(options, key=lambda value: value[0])
    statistics = {key: value for key, value in best.items() if key != "chartUV"}
    statistics["evaluatedGaugeCount"] = len(options)
    statistics["existingRegionScaleWarped"] = True
    if int(best["totalOverlapCount"]):
        return None, statistics
    result = {key: np.asarray(value) for key, value in trial.items()}
    result["chartUV"] = np.asarray(best["chartUV"], dtype=np.float32)
    return result, statistics


def _reintegrate_multi_region_chart(
    current: Mapping[str, np.ndarray],
    trial: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    region_ids: tuple[int, ...],
    *,
    component_id: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Conformally chart one patch joining any declared prior regions."""

    if len(region_ids) < 2 or len(set(region_ids)) != len(region_ids):
        raise ValueError("multi-region chart needs distinct prior regions")
    baseline_triangle = np.asarray(
        current["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_region = _triangle_region_labels(baseline_triangle)
    region_triangles = [
        baseline_triangle[baseline_region == int(region)]
        for region in region_ids
    ]
    if any(not len(value) for value in region_triangles):
        raise ValueError("multi-region chart references an empty region")
    region_nodes = [np.unique(value) for value in region_triangles]
    conformal_result, conformal_statistics = _conformal_joined_chart_option(
        current,
        trial,
        patch_triangle,
        region_triangles,
        region_nodes,
        component_id=component_id,
    )
    if conformal_result is not None:
        return conformal_result, conformal_statistics
    harmonic_result, harmonic_statistics = _harmonic_joined_chart_option(
        current,
        trial,
        patch_triangle,
        region_triangles,
        region_nodes,
        component_id=component_id,
    )
    return harmonic_result, {
        **harmonic_statistics,
        "freeBoundaryConformalAttempt": conformal_statistics,
    }


def _reintegrate_corridor_chart_gauge(
    current: Mapping[str, np.ndarray],
    trial: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    boundary: np.ndarray,
    mesh: Mapping[str, np.ndarray],
    local_to_global: np.ndarray,
    boundary_triangle_region: Mapping[tuple[int, int], int],
    *,
    first_arc_edge_count: int,
    second_arc_edge_count: int,
) -> tuple[dict[str, np.ndarray] | None, dict[str, Any]]:
    """Place two disconnected charts only after a physical strip joins them.

    A disconnected triangle region has an arbitrary rigid 2D gauge.  The old
    relative chart placement is therefore not evidence for or against a
    proposed physical bridge.  We hold either attached region fixed, rigidly
    align the other through the complete strip parameterization, and accept a
    gauge only when the patch and moved region remain injective against the
    unchanged chart.  Existing region scale and shape are never warped.
    """

    boundary = np.asarray(boundary, dtype=np.int32)
    first_count = int(first_arc_edge_count) + 1
    second_count = int(second_arc_edge_count) + 1
    if first_count + second_count != len(boundary):
        raise ValueError("corridor chart arc counts differ from its boundary")
    arcs = (boundary[:first_count], boundary[first_count:])
    region_values: list[int] = []
    for arc in arcs:
        values = {
            int(boundary_triangle_region[tuple(sorted((int(first), int(second))))])
            for first, second in zip(arc[:-1], arc[1:])
            if tuple(sorted((int(first), int(second))))
            in boundary_triangle_region
        }
        if len(values) != 1:
            raise ValueError("corridor chart arc lacks one triangle-region gauge")
        region_values.append(next(iter(values)))
    if region_values[0] == region_values[1]:
        raise ValueError("corridor chart arcs already share one triangle region")

    baseline_triangle = np.asarray(
        current["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_region = _triangle_region_labels(baseline_triangle)
    region_triangles = [
        baseline_triangle[baseline_region == region]
        for region in region_values
    ]
    region_nodes = [np.unique(values) for values in region_triangles]
    if np.intersect1d(region_nodes[0], region_nodes[1]).size:
        raise ValueError(
            "corridor chart regions still share a non-manifold vertex"
        )

    point_parameter = np.asarray(mesh["pointUV"], dtype=np.float64)
    point_kind = np.asarray(mesh["pointKind"], dtype=np.uint8)
    local_boundary = np.flatnonzero(point_kind == 0)
    if len(local_boundary) != len(boundary):
        raise ValueError("corridor chart mesh boundary differs from exact arcs")
    if not np.array_equal(
        np.asarray(mesh["pointSourceIndex"], dtype=np.int32)[local_boundary],
        boundary,
    ):
        raise ValueError("corridor chart mesh reordered its exact boundary")
    boundary_parameter = point_parameter[local_boundary]
    synthetic_local = np.flatnonzero(point_kind == 1)
    synthetic_global = np.asarray(local_to_global, dtype=np.int32)[synthetic_local]
    parameter_low = np.min(point_parameter, axis=0)
    parameter_high = np.max(point_parameter, axis=0)
    parameter_width = float(parameter_high[0] - parameter_low[0])
    if parameter_width <= 1.0e-8:
        raise ValueError("corridor chart parameterization has zero width")

    component = np.asarray(current["component"], dtype=np.int32)
    component_id = int(component[boundary[0]])
    same_component_triangle = baseline_triangle[
        np.all(component[baseline_triangle] == component_id, axis=1)
    ]
    base_chart = np.asarray(current["chartUV"], dtype=np.float64)
    options: list[tuple[tuple[float | int, ...], dict[str, Any]]] = []
    for fixed_index in (0, 1):
        moving_index = 1 - fixed_index
        fixed_arc = arcs[fixed_index]
        moving_arc = arcs[moving_index]
        fixed_local = (
            np.arange(first_count, dtype=np.int32)
            if fixed_index == 0
            else np.arange(first_count, len(boundary), dtype=np.int32)
        )
        moving_local = (
            np.arange(first_count, dtype=np.int32)
            if moving_index == 0
            else np.arange(first_count, len(boundary), dtype=np.int32)
        )
        moving_triangle = region_triangles[moving_index]
        moving_nodes = region_nodes[moving_index]
        stationary_triangle = same_component_triangle[
            ~np.isin(same_component_triangle, moving_nodes).any(axis=1)
        ]
        for fixed_reflection in (False, True):
            fixed_transform, fixed_translation, fixed_rms = (
                _rigid_chart_alignment(
                    boundary_parameter[fixed_local],
                    base_chart[fixed_arc],
                    reflected=fixed_reflection,
                )
            )
            parameter_target = (
                boundary_parameter @ fixed_transform + fixed_translation
            )
            for moving_reflection in (False, True):
                moving_transform, moving_translation, moving_rms = (
                    _rigid_chart_alignment(
                        base_chart[moving_arc],
                        parameter_target[moving_local],
                        reflected=moving_reflection,
                    )
                )
                chart = np.asarray(trial["chartUV"], dtype=np.float64).copy()
                chart[moving_nodes] = (
                    base_chart[moving_nodes] @ moving_transform
                    + moving_translation
                )

                first_parameter = boundary_parameter[:first_count, 1]
                second_parameter = boundary_parameter[first_count:, 1]
                synthetic_parameter = point_parameter[synthetic_local]
                first_uv = _interpolate_chart_arc(
                    first_parameter,
                    chart[arcs[0]],
                    synthetic_parameter[:, 1],
                )
                second_uv = _interpolate_chart_arc(
                    second_parameter,
                    chart[arcs[1]],
                    synthetic_parameter[:, 1],
                )
                blend = np.clip(
                    (synthetic_parameter[:, 0] - parameter_low[0])
                    / parameter_width,
                    0.0,
                    1.0,
                )
                chart[synthetic_global] = (
                    (1.0 - blend)[:, None] * first_uv
                    + blend[:, None] * second_uv
                )

                patch_overlap = _chart_overlap_count(
                    same_component_triangle,
                    patch_triangle,
                    chart,
                )
                moved_overlap = _chart_overlap_count(
                    stationary_triangle,
                    moving_triangle,
                    chart,
                )
                patch_self_overlap = _chart_overlap_count(
                    patch_triangle,
                    patch_triangle,
                    chart,
                )
                patch_uv = chart[np.asarray(patch_triangle, dtype=np.int32)]
                patch_signed_area = 0.5 * (
                    (patch_uv[:, 1, 0] - patch_uv[:, 0, 0])
                    * (patch_uv[:, 2, 1] - patch_uv[:, 0, 1])
                    - (patch_uv[:, 1, 1] - patch_uv[:, 0, 1])
                    * (patch_uv[:, 2, 0] - patch_uv[:, 0, 0])
                )
                collapsed = int(
                    np.count_nonzero(np.abs(patch_signed_area) <= 1.0e-8)
                )
                overlap = patch_overlap + moved_overlap + patch_self_overlap
                key = (
                    overlap,
                    collapsed,
                    round(max(fixed_rms, moving_rms), 9),
                    len(moving_nodes),
                    fixed_index,
                    int(fixed_reflection),
                    int(moving_reflection),
                )
                options.append(
                    (
                        key,
                        {
                            "chart": chart.astype(np.float32),
                            "fixedArc": fixed_index,
                            "movingArc": moving_index,
                            "fixedReflection": fixed_reflection,
                            "movingReflection": moving_reflection,
                            "fixedAlignmentRmsVoxels": round(fixed_rms, 6),
                            "movingAlignmentRmsVoxels": round(moving_rms, 6),
                            "movedRegionNodeCount": int(len(moving_nodes)),
                            "patchBaselineOverlapCount": int(patch_overlap),
                            "movedRegionOverlapCount": int(moved_overlap),
                            "patchSelfOverlapCount": int(patch_self_overlap),
                            "collapsedPatchTriangleCount": collapsed,
                            "totalOverlapCount": int(overlap),
                        },
                    )
                )
    if not options:
        raise ValueError("corridor chart integration produced no gauge option")
    _key, best = min(options, key=lambda value: value[0])
    statistics = {
        key: value for key, value in best.items() if key != "chart"
    }
    statistics["evaluatedGaugeCount"] = len(options)
    statistics["existingRegionScaleWarped"] = False
    if int(best["totalOverlapCount"]) or int(
        best["collapsedPatchTriangleCount"]
    ):
        conformal_trial, raw_conformal_statistics = (
            _conformal_joined_chart_option(
                current,
                trial,
                patch_triangle,
                region_triangles,
                region_nodes,
                component_id=component_id,
            )
        )
        conformal_statistics = {
            **raw_conformal_statistics,
            "rigidGaugeBest": statistics,
            "evaluatedGaugeCount": int(
                raw_conformal_statistics.get("evaluatedGaugeCount", 0)
            )
            + len(options),
        }
        if conformal_trial is not None:
            return conformal_trial, conformal_statistics
        harmonic_trial, raw_harmonic_statistics = (
            _harmonic_joined_chart_option(
                current,
                trial,
                patch_triangle,
                region_triangles,
                region_nodes,
                component_id=component_id,
            )
        )
        harmonic_statistics = {
            **raw_harmonic_statistics,
            "freeBoundaryConformalAttempt": conformal_statistics,
            "rigidGaugeBest": statistics,
            "evaluatedGaugeCount": int(
                raw_harmonic_statistics.get("evaluatedGaugeCount", 0)
            )
            + int(conformal_statistics["evaluatedGaugeCount"]),
        }
        return harmonic_trial, harmonic_statistics
    result = {key: np.asarray(value) for key, value in trial.items()}
    result["chartUV"] = np.asarray(best["chart"], dtype=np.float32)
    return result, statistics


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
    component: np.ndarray
    region: np.ndarray
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
    node_component = np.asarray(surface["component"], dtype=np.int32)
    triangle_component = np.full(len(triangle), -1, dtype=np.int32)
    if len(triangle):
        triangle_node_component = node_component[triangle]
        uniform = np.all(
            triangle_node_component
            == triangle_node_component[:, :1],
            axis=1,
        )
        triangle_component[uniform] = triangle_node_component[uniform, 0]
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
        component=triangle_component,
        region=_triangle_region_labels(triangle),
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
    attachment_edges: AbstractSet[tuple[int, int]] = frozenset(),
    spatial_index: _TriangleSpatialIndex | None = None,
    maximum_stored_intersections: int = 64,
) -> dict[str, Any]:
    """Audit crossings against every nonincident selected surface triangle."""

    from ..slab_association_integrity import _triangle_intersection

    index = spatial_index or _triangle_spatial_index(
        baseline_surface,
        tolerance=tolerance,
        cell_size=4.0,
    )
    baseline_triangle = index.triangle
    triangle_component = np.asarray(index.component, dtype=np.int32)
    triangle_region = np.asarray(index.region, dtype=np.int32)
    other_component_mask = triangle_component != component_id
    empty_diagnostics = {
        "broadPhaseTrianglePairCount": 0,
        "intersectingTrianglePairCount": 0,
        "intersectingSameComponentTrianglePairCount": 0,
        "intersectingOtherComponentTrianglePairCount": 0,
        "coplanarIntersectingTrianglePairCount": 0,
        "intersectingBaselineTriangleCount": 0,
        "intersectingPatchTriangleCount": 0,
        "intersectingBaselineTriangleRegionCount": 0,
        "intersectingBaselineComponentCount": 0,
        "intersectingBaselineTriangleIndices": [],
        "intersectingPatchTriangleIndices": [],
        "intersectingBaselineTriangleRegionIds": [],
        "intersectingBaselineComponentIds": [],
        "storedIntersectionPointXYZ": [],
        "storedIntersectionPairs": [],
        "storedIntersectionCount": 0,
        "storedIntersectionLimit": int(maximum_stored_intersections),
    }
    return _finish_triangle_intersection_audit(
        baseline_surface,
        augmented_surface,
        patch_triangle,
        component_id,
        tolerance=tolerance,
        attachment_edges=attachment_edges,
        index=index,
        baseline_triangle=baseline_triangle,
        triangle_component=triangle_component,
        triangle_region=triangle_region,
        other_component_mask=other_component_mask,
        empty_diagnostics=empty_diagnostics,
        maximum_stored_intersections=maximum_stored_intersections,
    )


def _derive_attachment_collar_domain(
    variant: Mapping[str, Any],
    field_coordinates: np.ndarray,
    *,
    raster_step_voxels: float,
    transition_voxels: float,
    intersection_tolerance_voxels: float,
    allowed_additional_reasons: AbstractSet[str] = frozenset(),
    minimum_attached_region_count: int = 2,
    require_injective_chart: bool = True,
) -> dict[str, Any] | None:
    """Locate a bounded collar only from an exhaustive attachment collision.

    A collar is not a generic collision escape hatch.  It is available only
    when an otherwise valid two-frontier strip intersects its own two exact
    attachment regions, every pair was retained, every crossing lies within
    one raster step of an attachment edge, and the joined intrinsic chart is
    already injective.  The affected raster depth is read from the actual
    intersecting patch triangles; one physical voxel is then reserved for a
    transition back to the unchanged CT depth field.
    """

    collision_reason = "completion intersects an existing selected surface"
    reasons = set(str(value) for value in variant.get("rejectionReasons", ()))
    if collision_reason not in reasons or not (
        reasons - {collision_reason}
    ).issubset(set(allowed_additional_reasons)):
        return None
    geometry = variant.get("geometry")
    mesh = variant.get("mesh")
    if not isinstance(geometry, Mapping) or not isinstance(mesh, Mapping):
        return None
    pair_count = int(geometry.get("intersectingTrianglePairCount", 0))
    stored_count = int(geometry.get("storedIntersectionCount", 0))
    pairs = geometry.get("storedIntersectionPairs", ())
    if (
        pair_count < 1
        or stored_count != pair_count
        or not isinstance(pairs, (list, tuple))
        or len(pairs) != pair_count
        or int(geometry.get("intersectingOtherComponentTrianglePairCount", 0))
        or (
            require_injective_chart
            and int(geometry.get("intrinsicChartOverlapCount", 1))
        )
    ):
        return None
    attached_regions = {
        int(value)
        for value in geometry.get("attachedBaselineTriangleRegionIds", ())
    }
    intersected_regions = {
        int(value)
        for value in geometry.get("intersectingBaselineTriangleRegionIds", ())
    }
    if (
        len(attached_regions) < int(minimum_attached_region_count)
        or not intersected_regions.issubset(attached_regions)
    ):
        return None
    maximum_seam_distance = max(
        float(raster_step_voxels),
        4.0 * float(intersection_tolerance_voxels),
    )
    if any(
        not bool(pair.get("sameComponent"))
        or pair.get("minimumAttachmentEdgeDistanceVoxels") is None
        or float(pair["minimumAttachmentEdgeDistanceVoxels"])
        > maximum_seam_distance
        for pair in pairs
        if isinstance(pair, Mapping)
    ) or any(not isinstance(pair, Mapping) for pair in pairs):
        return None

    coordinate = np.asarray(field_coordinates, dtype=np.int32)
    if not len(coordinate):
        return None
    minimum_u = int(np.min(coordinate[:, 0]))
    maximum_u = int(np.max(coordinate[:, 0]))
    interior_column_count = maximum_u - minimum_u - 1
    if interior_column_count < 1:
        return None
    point_kind = np.asarray(mesh["pointKind"], dtype=np.uint8)
    point_source = np.asarray(mesh["pointSourceIndex"], dtype=np.int32)
    triangle = np.asarray(mesh["trianglePointIndex"], dtype=np.int32)
    collision_depth = {"left": 0, "right": 0}
    for pair in pairs:
        patch_index = int(pair["patchTriangleIndex"])
        if not 0 <= patch_index < len(triangle):
            return None
        local = triangle[patch_index]
        field_local = local[point_kind[local] == 1]
        if not len(field_local):
            return None
        source = point_source[field_local]
        if np.any(source < 0) or np.any(source >= len(coordinate)):
            return None
        u_value = coordinate[source, 0]
        left_depth = u_value - minimum_u
        right_depth = maximum_u - u_value
        nearest_left = int(np.min(left_depth))
        nearest_right = int(np.min(right_depth))
        if nearest_left == nearest_right:
            return None
        side = "left" if nearest_left < nearest_right else "right"
        depth = left_depth if side == "left" else right_depth
        collision_depth[side] = max(
            collision_depth[side], int(np.max(depth))
        )

    transition_columns = int(
        math.ceil(float(transition_voxels) / float(raster_step_voxels))
    )
    column_count = {
        side: (
            collision_depth[side] + transition_columns
            if collision_depth[side]
            else 0
        )
        for side in ("left", "right")
    }
    if (
        column_count["left"] + column_count["right"]
        >= interior_column_count
        or max(column_count.values()) > interior_column_count
    ):
        return None
    return {
        "mode": "attachment-halfspace-collar",
        "leftCollisionDepthColumns": collision_depth["left"],
        "rightCollisionDepthColumns": collision_depth["right"],
        "leftColumnCount": column_count["left"],
        "rightColumnCount": column_count["right"],
        "transitionColumnCount": transition_columns,
        "transitionVoxels": round(float(transition_voxels), 6),
        "sourceCollisionPairCount": pair_count,
        "sourceIntersectedTriangleRegionIds": sorted(intersected_regions),
        "maximumSourceAttachmentDistanceVoxels": round(
            max(
                float(pair["minimumAttachmentEdgeDistanceVoxels"])
                for pair in pairs
            ),
            6,
        ),
        "maximumEligibleAttachmentDistanceVoxels": round(
            maximum_seam_distance, 6
        ),
        "attachedTriangleRegionCount": int(len(attached_regions)),
        "injectiveChartRequiredForEligibility": bool(require_injective_chart),
        "allowedAdditionalRejectionReasons": sorted(
            str(value) for value in allowed_additional_reasons
        ),
    }


def _apply_attachment_halfspace_collar(
    surface: Mapping[str, np.ndarray],
    boundary: np.ndarray,
    boundary_parameter_uv: np.ndarray,
    field_parameter_uv: np.ndarray,
    field_coordinates: np.ndarray,
    field_xyz: np.ndarray,
    domain: Mapping[str, Any],
    *,
    first_arc_edge_count: int,
    second_arc_edge_count: int,
    minimum_outward_tangent_ratio: float,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Minimally unfold a strip's CT samples away from its attachment faces.

    For each affected raster point, the adjacent existing triangle defines the
    inward half-space across the exact attachment edge.  Only points whose
    continuation turns back into that half-space move, and they move by the
    minimum tangent-plane displacement needed for the declared exit slope.
    All other strip samples remain the original collective-depth solution.
    The resulting complete strip is still subjected to every native CT,
    intersection, topology, normal-tail, and intrinsic-chart gate.
    """

    boundary = np.asarray(boundary, dtype=np.int32)
    boundary_parameter = np.asarray(boundary_parameter_uv, dtype=np.float64)
    field_parameter = np.asarray(field_parameter_uv, dtype=np.float64)
    coordinate = np.asarray(field_coordinates, dtype=np.int32)
    original = np.asarray(field_xyz, dtype=np.float32)
    if len(field_parameter) != len(coordinate) or len(original) != len(coordinate):
        raise ValueError("attachment collar field arrays differ")
    first_count = int(first_arc_edge_count) + 1
    second_count = int(second_arc_edge_count) + 1
    if first_count + second_count != len(boundary):
        raise ValueError("attachment collar arc counts differ")
    lookup = {
        (int(value[0]), int(value[1])): index
        for index, value in enumerate(coordinate)
    }
    minimum = np.min(coordinate, axis=0)
    maximum = np.max(coordinate, axis=0)
    if len(lookup) != (
        int(maximum[0] - minimum[0] + 1)
        * int(maximum[1] - minimum[1] + 1)
    ):
        raise ValueError("attachment collar needs one complete field raster")

    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge_index in range(3):
            edge = tuple(
                sorted(
                    (
                        int(triangle[edge_index]),
                        int(triangle[(edge_index + 1) % 3]),
                    )
                )
            )
            edge_triangle[edge].append(triangle_index)

    adjusted = original.copy()
    displacement: list[float] = []
    inward_violation: list[float] = []
    side_adjusted = {"left": 0, "right": 0}
    first_arc = boundary[:first_count]
    second_arc = boundary[first_count:][::-1]
    first_parameter = boundary_parameter[:first_count]
    second_parameter = boundary_parameter[first_count:][::-1]
    sides = (
        (
            "left",
            first_arc,
            first_parameter,
            range(
                int(minimum[0]) + 1,
                int(minimum[0]) + int(domain["leftColumnCount"]) + 1,
            ),
        ),
        (
            "right",
            second_arc,
            second_parameter,
            range(
                int(maximum[0]) - 1,
                int(maximum[0]) - int(domain["rightColumnCount"]) - 1,
                -1,
            ),
        ),
    )
    for side, arc, parameter, columns in sides:
        if len(parameter) < 2 or np.any(np.diff(parameter[:, 1]) < -1.0e-7):
            raise ValueError("attachment collar arc parameter is not monotone")
        arc_u = float(np.median(parameter[:, 0]))
        for column in columns:
            for row_value in range(int(minimum[1]), int(maximum[1]) + 1):
                pixel = lookup[(int(column), int(row_value))]
                longitudinal = float(field_parameter[pixel, 1])
                segment = int(
                    np.searchsorted(parameter[:, 1], longitudinal, side="right")
                    - 1
                )
                segment = max(0, min(len(arc) - 2, segment))
                denominator = max(
                    float(parameter[segment + 1, 1] - parameter[segment, 1]),
                    1.0e-9,
                )
                fraction = float(
                    np.clip(
                        (longitudinal - float(parameter[segment, 1]))
                        / denominator,
                        0.0,
                        1.0,
                    )
                )
                first = int(arc[segment])
                second = int(arc[segment + 1])
                boundary_xyz = (
                    (1.0 - fraction) * xyz[first] + fraction * xyz[second]
                ).astype(np.float64)
                edge = tuple(sorted((first, second)))
                incident = edge_triangle.get(edge, ())
                if len(incident) != 1:
                    raise ValueError(
                        "attachment collar edge is not an exact open boundary"
                    )
                triangle = triangles[int(incident[0])]
                third_values = [
                    int(value)
                    for value in triangle
                    if int(value) not in {first, second}
                ]
                if len(third_values) != 1:
                    raise ValueError("attachment collar triangle is degenerate")
                tangent = xyz[second].astype(np.float64) - xyz[first]
                tangent_length = float(np.linalg.norm(tangent))
                if tangent_length <= 1.0e-9:
                    raise ValueError("attachment collar edge has zero length")
                tangent /= tangent_length
                inward = xyz[third_values[0]].astype(np.float64) - boundary_xyz
                inward -= tangent * float(np.dot(inward, tangent))
                inward_length = float(np.linalg.norm(inward))
                if inward_length <= 1.0e-9:
                    raise ValueError(
                        "attachment collar triangle has zero transverse extent"
                    )
                inward /= inward_length
                signed_inward = float(
                    np.dot(adjusted[pixel].astype(np.float64) - boundary_xyz, inward)
                )
                physical_depth = abs(
                    float(field_parameter[pixel, 0]) - arc_u
                )
                target = -float(minimum_outward_tangent_ratio) * physical_depth
                violation = signed_inward - target
                if violation > 0.0:
                    adjusted[pixel] = (
                        adjusted[pixel].astype(np.float64) - inward * violation
                    ).astype(np.float32)
                    displacement.append(violation)
                    inward_violation.append(max(signed_inward, 0.0))
                    side_adjusted[side] += 1

    statistics = {
        **dict(domain),
        "minimumOutwardTangentRatio": round(
            float(minimum_outward_tangent_ratio), 6
        ),
        "minimumOutwardAngleDegrees": round(
            math.degrees(math.atan(float(minimum_outward_tangent_ratio))), 6
        ),
        "adjustedFieldPointCount": len(displacement),
        "leftAdjustedFieldPointCount": side_adjusted["left"],
        "rightAdjustedFieldPointCount": side_adjusted["right"],
        "maximumFieldPointDisplacementVoxels": round(
            max(displacement, default=0.0), 6
        ),
        "medianFieldPointDisplacementVoxels": round(
            float(np.median(displacement)) if displacement else 0.0, 6
        ),
        "maximumOriginalInwardViolationVoxels": round(
            max(inward_violation, default=0.0), 6
        ),
        "unmodifiedFieldPointCount": int(len(original) - len(displacement)),
    }
    return adjusted, statistics


def _finish_triangle_intersection_audit(
    baseline_surface: Mapping[str, np.ndarray],
    augmented_surface: Mapping[str, np.ndarray],
    patch_triangle: np.ndarray,
    component_id: int,
    *,
    tolerance: float,
    attachment_edges: AbstractSet[tuple[int, int]],
    index: _TriangleSpatialIndex,
    baseline_triangle: np.ndarray,
    triangle_component: np.ndarray,
    triangle_region: np.ndarray,
    other_component_mask: np.ndarray,
    empty_diagnostics: Mapping[str, Any],
    maximum_stored_intersections: int,
) -> dict[str, Any]:
    """Complete the indexed pair audit after reusable collar helpers."""

    from ..slab_association_integrity import _triangle_intersection

    if not len(baseline_triangle) or not len(patch_triangle):
        return {
            "otherComponentTriangleCount": int(
                np.count_nonzero(other_component_mask)
            ),
            "sameComponentTriangleCount": int(
                np.count_nonzero(~other_component_mask)
            ),
            **empty_diagnostics,
        }
    xyz = np.asarray(augmented_surface["midpointXYZ"], dtype=np.float32)
    other_point = index.point
    augmented_weld = np.asarray(
        augmented_surface.get(
            "surfaceNodeWeldGroup",
            np.arange(len(xyz), dtype=np.int64),
        ),
        dtype=np.int64,
    )
    baseline_xyz = np.asarray(
        baseline_surface["midpointXYZ"], dtype=np.float32
    )
    baseline_weld = np.asarray(
        baseline_surface.get(
            "surfaceNodeWeldGroup",
            np.arange(len(baseline_xyz), dtype=np.int64),
        ),
        dtype=np.int64,
    )
    normalized_attachment = tuple(
        sorted(
            {
                tuple(sorted((int(first), int(second))))
                for first, second in attachment_edges
            }
        )
    )
    attachment_segment = tuple(
        (xyz[first], xyz[second]) for first, second in normalized_attachment
    )
    broad_count = 0
    intersection_count = 0
    same_component_intersection_count = 0
    other_component_intersection_count = 0
    coplanar_intersection_count = 0
    intersecting_baseline: set[int] = set()
    intersecting_patch: set[int] = set()
    intersection_points: list[list[float]] = []
    intersection_pairs: list[dict[str, Any]] = []
    for patch_index, triangle in enumerate(
        np.asarray(patch_triangle, dtype=np.int32)
    ):
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
            if np.any(
                augmented_weld[triangle, None]
                == baseline_weld[baseline_nodes][None, :]
            ):
                continue
            intersection, coplanar = _triangle_intersection(
                point, other_point[int(other_index)], tolerance
            )
            if intersection is not None:
                intersection_count += 1
                intersecting_baseline.add(int(other_index))
                intersecting_patch.add(int(patch_index))
                if triangle_component[int(other_index)] == component_id:
                    same_component_intersection_count += 1
                else:
                    other_component_intersection_count += 1
                coplanar_intersection_count += int(coplanar)
                if len(intersection_points) < maximum_stored_intersections:
                    rounded_intersection = [
                        round(float(value), 6) for value in intersection
                    ]
                    intersection_points.append(rounded_intersection)
                    patch_normal = np.cross(
                        point[1] - point[0], point[2] - point[0]
                    ).astype(np.float64)
                    baseline_point = other_point[int(other_index)]
                    baseline_normal = np.cross(
                        baseline_point[1] - baseline_point[0],
                        baseline_point[2] - baseline_point[0],
                    ).astype(np.float64)
                    patch_length = float(np.linalg.norm(patch_normal))
                    baseline_length = float(np.linalg.norm(baseline_normal))
                    if patch_length > 1.0e-12 and baseline_length > 1.0e-12:
                        patch_normal /= patch_length
                        baseline_normal /= baseline_length
                        axial_angle: float | None = math.degrees(
                            math.acos(
                                float(
                                    np.clip(
                                        abs(float(np.dot(patch_normal, baseline_normal))),
                                        0.0,
                                        1.0,
                                    )
                                )
                            )
                        )
                        patch_plane_distance: float | None = abs(
                            float(
                                np.dot(
                                    np.mean(baseline_point, axis=0) - point[0],
                                    patch_normal,
                                )
                            )
                        )
                        baseline_plane_distance: float | None = abs(
                            float(
                                np.dot(
                                    np.mean(point, axis=0) - baseline_point[0],
                                    baseline_normal,
                                )
                            )
                        )
                    else:
                        axial_angle = None
                        patch_plane_distance = None
                        baseline_plane_distance = None
                    attachment_distance = (
                        min(
                            _point_segment_distance(
                                np.asarray(intersection, dtype=np.float64),
                                np.asarray(first, dtype=np.float64),
                                np.asarray(second, dtype=np.float64),
                            )
                            for first, second in attachment_segment
                        )
                        if attachment_segment
                        else None
                    )
                    intersection_pairs.append(
                        {
                            "baselineTriangleIndex": int(other_index),
                            "patchTriangleIndex": int(patch_index),
                            "baselineTriangleRegionId": int(
                                triangle_region[int(other_index)]
                            ),
                            "baselineComponentId": int(
                                triangle_component[int(other_index)]
                            ),
                            "sameComponent": bool(
                                triangle_component[int(other_index)] == component_id
                            ),
                            "coplanar": bool(coplanar),
                            "intersectionPointXYZ": rounded_intersection,
                            "minimumAttachmentEdgeDistanceVoxels": (
                                round(float(attachment_distance), 6)
                                if attachment_distance is not None
                                else None
                            ),
                            "axialNormalDisagreementDegrees": (
                                round(float(axial_angle), 6)
                                if axial_angle is not None
                                else None
                            ),
                            "baselineCentroidToPatchPlaneVoxels": (
                                round(float(patch_plane_distance), 6)
                                if patch_plane_distance is not None
                                else None
                            ),
                            "patchCentroidToBaselinePlaneVoxels": (
                                round(float(baseline_plane_distance), 6)
                                if baseline_plane_distance is not None
                                else None
                            ),
                        }
                    )
    baseline_indices = sorted(intersecting_baseline)
    patch_indices = sorted(intersecting_patch)
    intersecting_regions = sorted(
        set(int(triangle_region[index]) for index in baseline_indices)
    )
    intersecting_components = sorted(
        set(int(triangle_component[index]) for index in baseline_indices)
    )
    return {
        "otherComponentTriangleCount": int(
            np.count_nonzero(other_component_mask)
        ),
        "sameComponentTriangleCount": int(
            np.count_nonzero(~other_component_mask)
        ),
        "broadPhaseTrianglePairCount": int(broad_count),
        "intersectingTrianglePairCount": int(intersection_count),
        "intersectingSameComponentTrianglePairCount": int(
            same_component_intersection_count
        ),
        "intersectingOtherComponentTrianglePairCount": int(
            other_component_intersection_count
        ),
        "coplanarIntersectingTrianglePairCount": int(
            coplanar_intersection_count
        ),
        "intersectingBaselineTriangleCount": len(baseline_indices),
        "intersectingPatchTriangleCount": len(patch_indices),
        "intersectingBaselineTriangleRegionCount": len(intersecting_regions),
        "intersectingBaselineComponentCount": len(intersecting_components),
        "intersectingBaselineTriangleIndices": baseline_indices[
            :maximum_stored_intersections
        ],
        "intersectingPatchTriangleIndices": patch_indices[
            :maximum_stored_intersections
        ],
        "intersectingBaselineTriangleRegionIds": intersecting_regions[
            :maximum_stored_intersections
        ],
        "intersectingBaselineComponentIds": intersecting_components[
            :maximum_stored_intersections
        ],
        "storedIntersectionPointXYZ": intersection_points,
        "storedIntersectionPairs": intersection_pairs,
        "storedIntersectionCount": len(intersection_points),
        "storedIntersectionLimit": int(maximum_stored_intersections),
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
) -> tuple[
    dict[str, np.ndarray],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
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
            np.asarray(
                mesh.get("outputPointUV", mesh["pointUV"]), dtype=np.float32
            )[synthetic_local],
        )
    )
    result["integrationResidualVoxels"] = np.asarray(
        surface["integrationResidualVoxels"], dtype=np.float32
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
    if "surfaceNodeWeldGroup" in surface:
        weld_group = np.asarray(
            surface["surfaceNodeWeldGroup"], dtype=np.int64
        )
        first_synthetic_group = (
            int(np.max(weld_group)) + 1 if len(weld_group) else 0
        )
        result["surfaceNodeWeldGroup"] = np.concatenate(
            (
                weld_group,
                first_synthetic_group
                + np.arange(len(synthetic_global), dtype=np.int64),
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
    return result, synthetic_global, patch_triangle, added_edge, local_to_global


def _loop_counts(loops: Mapping[str, np.ndarray]) -> dict[str, int]:
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    macro = np.asarray(loops["loopMacroEligible"], dtype=np.uint8) > 0
    return {
        "outerLoopCount": int(np.count_nonzero(kind == 0)),
        "interiorHoleCount": int(np.count_nonzero(kind == 1)),
        "macroHoleCount": int(np.count_nonzero(macro)),
    }


def _derive_transverse_island_sector_domains(
    current: Mapping[str, np.ndarray],
    base_variant: Mapping[str, Any],
    *,
    component_id: int,
    boundary: np.ndarray,
    boundary_parameter_uv: np.ndarray,
    field_parameter_uv: np.ndarray,
    field_coordinates: np.ndarray,
    field_xyz: np.ndarray,
    first_arc_edge_count: int,
    second_arc_edge_count: int,
    minimum_boundary_separation: float,
    maximum_edge_flip_iterations: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Enumerate whole sectors supported by a transverse surface island.

    A topology-normalized island is usable only when one boundary vertex is an
    exact physical clone of each established corridor front.  Re-welding those
    two vertices does not itself join triangle fans.  Each returned hypothesis
    fills one complete side of one island path and therefore has enough
    topology to join all three prior regions as one manifold surface.
    """

    diagnostics: list[dict[str, Any]] = []
    mesh_value = base_variant.get("mesh")
    geometry = base_variant.get("geometry")
    if not isinstance(mesh_value, Mapping) or not isinstance(geometry, Mapping):
        return [], [{"eligible": False, "reason": "base corridor was not constructed"}]
    if int(geometry.get("intersectingOtherComponentTrianglePairCount", 0)):
        return [], [{
            "eligible": False,
            "reason": "base corridor intersects a different surface component",
        }]
    if int(geometry.get("intersectingSameComponentTrianglePairCount", 0)) != int(
        geometry.get("intersectingTrianglePairCount", 0)
    ):
        return [], [{
            "eligible": False,
            "reason": "base corridor collision provenance is not same-component exhaustive",
        }]
    attached_region = tuple(
        int(value)
        for value in geometry.get("attachedBaselineTriangleRegionIds", ())
    )
    if len(attached_region) != 2:
        return [], [{
            "eligible": False,
            "reason": "base corridor does not attach exactly two triangle regions",
        }]
    collision_region = {
        int(value)
        for value in geometry.get("intersectingBaselineTriangleRegionIds", ())
    }
    island_region = sorted(collision_region - set(attached_region))
    if not island_region:
        return [], [{
            "eligible": False,
            "reason": "base corridor has no third colliding triangle region",
        }]

    boundary_node = np.asarray(boundary, dtype=np.int32)
    boundary_parameter = np.asarray(boundary_parameter_uv, dtype=np.float64)
    coordinate = np.asarray(field_coordinates, dtype=np.int32)
    field_parameter = np.asarray(field_parameter_uv, dtype=np.float64)
    first_count = int(first_arc_edge_count) + 1
    second_count = int(second_arc_edge_count) + 1
    if first_count + second_count != len(boundary_node):
        raise ValueError("transverse-island corridor arc counts differ")
    first_arc = boundary_node[:first_count]
    first_parameter = boundary_parameter[:first_count]
    second_arc = boundary_node[first_count:][::-1]
    second_parameter = boundary_parameter[first_count:][::-1]
    if len(coordinate) != len(field_parameter):
        raise ValueError("transverse-island CT field coordinates differ")
    if not len(coordinate):
        raise ValueError("transverse-island CT field is empty")
    coordinate_lookup = {
        (int(value[0]), int(value[1])): index
        for index, value in enumerate(coordinate)
    }
    if len(coordinate_lookup) != len(coordinate):
        raise ValueError("transverse-island CT field repeats a grid coordinate")
    minimum_coordinate = np.min(coordinate, axis=0)
    maximum_coordinate = np.max(coordinate, axis=0)
    interior_column = list(
        range(int(minimum_coordinate[0]) + 1, int(maximum_coordinate[0]))
    )

    base_mesh = {key: np.asarray(value) for key, value in mesh_value.items()}
    base_kind = np.asarray(base_mesh["pointKind"], dtype=np.uint8)
    base_source = np.asarray(base_mesh["pointSourceIndex"], dtype=np.int32)
    base_xyz = np.empty((len(base_kind), 3), dtype=np.float32)
    base_xyz[base_kind == 0] = np.asarray(
        current["midpointXYZ"], dtype=np.float32
    )[base_source[base_kind == 0]]
    base_xyz[base_kind == 1] = np.asarray(field_xyz, dtype=np.float32)[
        base_source[base_kind == 1]
    ]
    base_triangle = np.asarray(base_mesh["trianglePointIndex"], dtype=np.int32)
    base_parameter = np.asarray(base_mesh["pointUV"], dtype=np.float32)

    triangle = np.asarray(current["triangleFrontierIndex"], dtype=np.int32)
    triangle_region = _triangle_region_labels(triangle)
    node_component = np.asarray(current["component"], dtype=np.int32)
    weld_group = np.asarray(
        current.get("surfaceNodeWeldGroup", np.arange(len(node_component))),
        dtype=np.int64,
    )
    xyz = np.asarray(current["midpointXYZ"], dtype=np.float64)

    def endpoint_match(
        island_loop: np.ndarray, rail: np.ndarray
    ) -> tuple[int, int]:
        matches = {
            (int(island), int(front))
            for island in island_loop
            for front in rail
            if int(weld_group[int(island)]) == int(weld_group[int(front)])
            and float(np.linalg.norm(xyz[int(island)] - xyz[int(front)]))
            <= 1.0e-5
        }
        if len(matches) != 1:
            raise ValueError(
                "transverse island does not have one exact endpoint on each front"
            )
        return next(iter(matches))

    domains: list[dict[str, Any]] = []
    for region_id in island_region:
        region_record: dict[str, Any] = {
            "eligible": False,
            "islandTriangleRegion": int(region_id),
        }
        try:
            region_triangle = triangle[triangle_region == region_id]
            if not len(region_triangle) or not np.all(
                node_component[region_triangle] == component_id
            ):
                raise ValueError(
                    "third collision region is not wholly in the corridor component"
                )
            island_loop = _triangle_region_boundary_loop(current, region_id)
            left_island, left_front = endpoint_match(island_loop, first_arc)
            right_island, right_front = endpoint_match(island_loop, second_arc)
            if left_island == right_island or left_front == right_front:
                raise ValueError("transverse island endpoints collapse")
            welded, weld_statistics = _weld_surface_node_pairs(
                current,
                ((left_island, left_front), (right_island, right_front)),
            )
            welded_loop = np.asarray(
                [
                    left_front
                    if int(value) == left_island
                    else right_front
                    if int(value) == right_island
                    else int(value)
                    for value in island_loop
                ],
                dtype=np.int32,
            )
            if len(np.unique(welded_loop)) != len(welded_loop):
                raise ValueError("transverse-island weld pinches its boundary")

            left_position = np.flatnonzero(first_arc == left_front)
            right_position = np.flatnonzero(second_arc == right_front)
            if len(left_position) != 1 or len(right_position) != 1:
                raise ValueError("transverse-island endpoint is not unique on a front")
            left_index, right_index = int(left_position[0]), int(right_position[0])
            projected, projection_distance = _project_nodes_to_triangle_parameter(
                np.asarray(welded["midpointXYZ"], dtype=np.float32)[welded_loop],
                base_xyz,
                base_triangle,
                base_parameter,
            )
            projected_lookup = {
                int(node): projected[index].astype(np.float64)
                for index, node in enumerate(welded_loop)
            }
            projected_lookup[left_front] = first_parameter[left_index]
            projected_lookup[right_front] = second_parameter[right_index]
            island_paths = _cyclic_paths_between(
                welded_loop, left_front, right_front
            )
            region_area = float(
                np.sum(
                    np.asarray(
                        current["triangleAreaVoxelsSquared"], dtype=np.float64
                    )[triangle_region == region_id]
                )
            )
            region_record.update(
                {
                    "eligible": True,
                    "islandTriangleCount": int(len(region_triangle)),
                    "islandBoundaryVertexCount": int(len(island_loop)),
                    "islandAreaVoxelsSquared": round(region_area, 6),
                    "projectionDistanceVoxels": {
                        "median": round(float(np.median(projection_distance)), 6),
                        "maximum": round(float(np.max(projection_distance)), 6),
                    },
                    "weld": weld_statistics,
                }
            )

            for side in ("lower", "upper"):
                if side == "upper":
                    first_side = first_arc[left_index:]
                    first_side_uv = first_parameter[left_index:]
                    second_side = second_arc[right_index:][::-1]
                    second_side_uv = second_parameter[right_index:][::-1]
                    mouth_row = int(maximum_coordinate[1])
                else:
                    first_side = first_arc[: left_index + 1][::-1]
                    first_side_uv = first_parameter[: left_index + 1][::-1]
                    second_side = second_arc[: right_index + 1]
                    second_side_uv = second_parameter[: right_index + 1]
                    mouth_row = int(minimum_coordinate[1])
                try:
                    mouth_pixel = np.asarray(
                        [
                            coordinate_lookup[(column, mouth_row)]
                            for column in interior_column
                        ],
                        dtype=np.int32,
                    )
                except KeyError as error:
                    raise ValueError(
                        "transverse-island sector lacks a complete CT mouth row"
                    ) from error

                for path_index, path in enumerate(island_paths):
                    reverse_path = path[::-1]
                    island_interior = reverse_path[1:-1]
                    first_length = len(first_side)
                    mouth_length = len(mouth_pixel)
                    boundary_source = np.concatenate(
                        (
                            first_side,
                            mouth_pixel,
                            second_side,
                            island_interior,
                        )
                    ).astype(np.int32)
                    boundary_kind = np.concatenate(
                        (
                            np.zeros(first_length, dtype=np.uint8),
                            np.ones(mouth_length, dtype=np.uint8),
                            np.zeros(
                                len(second_side) + len(island_interior),
                                dtype=np.uint8,
                            ),
                        )
                    )
                    boundary_uv = np.vstack(
                        (
                            first_side_uv,
                            field_parameter[mouth_pixel],
                            second_side_uv,
                            np.asarray(
                                [projected_lookup[int(node)] for node in island_interior],
                                dtype=np.float64,
                            ).reshape((-1, 2)),
                        )
                    )
                    right_mouth_position = first_length + mouth_length
                    mouth_walk = np.arange(
                        first_length - 1,
                        right_mouth_position + 1,
                        dtype=np.int32,
                    )
                    mouth_edge = np.column_stack(
                        (mouth_walk[:-1], mouth_walk[1:])
                    ).astype(np.int32)
                    variant_record = {
                        **region_record,
                        "sectorSide": side,
                        "islandBoundaryPath": int(path_index),
                        "islandPathVertexCount": int(len(path)),
                        "islandPathMeanLongitudinalParameterVoxels": round(
                            float(
                                np.mean(
                                    [
                                        projected_lookup[int(node)][1]
                                        for node in path
                                    ]
                                )
                            ),
                            6,
                        ),
                        "sectorBoundaryParameterAreaVoxelsSquared": round(
                            abs(_signed_polygon_area(boundary_uv)), 6
                        ),
                    }
                    try:
                        sector_mesh, sector_mesh_stats = (
                            triangulate_mixed_boundary_field(
                                boundary_kind,
                                boundary_source,
                                boundary_uv,
                                field_parameter,
                                new_frontier_boundary_edge=mouth_edge,
                                minimum_boundary_separation=(
                                    minimum_boundary_separation
                                ),
                                maximum_edge_flip_iterations=(
                                    maximum_edge_flip_iterations
                                ),
                            )
                        )
                    except ValueError as error:
                        diagnostics.append(
                            {
                                **variant_record,
                                "constructed": False,
                                "reason": str(error),
                            }
                        )
                        continue
                    domains.append(
                        {
                            "surface": welded,
                            "mesh": sector_mesh,
                            "meshStatistics": {
                                **sector_mesh_stats,
                                "transverseIslandSector": variant_record,
                            },
                            "statistics": variant_record,
                        }
                    )
        except ValueError as error:
            region_record["reason"] = str(error)
        diagnostics.append(region_record)
    return domains, diagnostics


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
    boundary_parameter_uv: np.ndarray | None = None,
    field_parameter_uv: np.ndarray | None = None,
    field_coordinates: np.ndarray | None = None,
    first_arc_edge_count: int | None = None,
    second_arc_edge_count: int | None = None,
    new_frontier_edges: AbstractSet[tuple[int, int]] = frozenset(),
    boundary_triangle_region: Mapping[tuple[int, int], int] | None = None,
    required_triangle_region_reduction: int = 0,
    collision_index: _TriangleSpatialIndex | None = None,
    mesh_override: tuple[Mapping[str, np.ndarray], Mapping[str, Any]] | None = None,
    settings: PhysicalRibbonDenseCompletionSettings,
) -> dict[str, Any]:
    """Construct and audit one complete mesh-density hypothesis."""

    try:
        if mesh_override is not None:
            if first_arc_edge_count is not None or second_arc_edge_count is not None:
                raise ValueError(
                    "explicit completion mesh cannot also request a structured strip"
                )
            mesh = {
                key: np.asarray(value).copy()
                for key, value in mesh_override[0].items()
            }
            mesh_stats = dict(mesh_override[1])
        elif first_arc_edge_count is not None or second_arc_edge_count is not None:
            if (
                first_arc_edge_count is None
                or second_arc_edge_count is None
                or boundary_parameter_uv is None
                or field_parameter_uv is None
                or field_coordinates is None
            ):
                raise ValueError("structured corridor mesh inputs are incomplete")
            mesh, mesh_stats = triangulate_two_frontier_strip_field(
                boundary,
                boundary_parameter_uv,
                field_parameter_uv,
                field_coordinates,
                first_arc_edge_count=first_arc_edge_count,
                second_arc_edge_count=second_arc_edge_count,
            )
        else:
            mesh, mesh_stats = triangulate_weak_boundary_field(
                boundary,
                (
                    np.asarray(boundary_parameter_uv, dtype=np.float32)
                    if boundary_parameter_uv is not None
                    else np.asarray(current["chartUV"], dtype=np.float32)[boundary]
                ),
                (
                    np.asarray(field_parameter_uv, dtype=np.float32)
                    if field_parameter_uv is not None
                    else field_uv
                ),
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
    output_point_uv = np.empty((len(point_kind), 2), dtype=np.float32)
    output_point_uv[boundary_local] = np.asarray(
        current["chartUV"], dtype=np.float32
    )[point_source[boundary_local]]
    output_point_uv[field_local] = np.asarray(field_uv, dtype=np.float32)[
        point_source[field_local]
    ]
    mesh = dict(mesh)
    mesh["outputPointUV"] = output_point_uv
    normalized_new_frontier_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in new_frontier_edges
    }
    new_frontier_local_edges = {
        tuple(sorted((int(edge[0]), int(edge[1]))))
        for edge in np.asarray(
            mesh.get("newFrontierLocalEdge", np.empty((0, 2), dtype=np.int32)),
            dtype=np.int32,
        ).reshape((-1, 2))
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
        minimum_triangle_area_voxels_squared=(
            settings.minimum_triangle_area_voxels_squared
        ),
        maximum_triangle_edge_voxels=settings.maximum_triangle_edge_voxels,
    )
    mesh_stats = {
        **mesh_stats,
        "physicalEdgeFlipIterations": int(physical_flips),
    }
    oriented: list[tuple[int, int, int]] = []
    triangle_area: list[float] = []
    triangle_residual: list[float] = []
    triangle_maximum_edge: list[float] = []
    for triangle in local_triangle:
        values, area, residual, maximum_edge = _triangle_geometry(
            tuple(int(value) for value in triangle),
            local_xyz,
            local_reference,
        )
        oriented.append(values)
        triangle_area.append(area)
        triangle_residual.append(residual)
        triangle_maximum_edge.append(maximum_edge)
    mesh["trianglePointIndex"] = np.asarray(oriented, dtype=np.int32)
    minimum_area = min(triangle_area, default=0.0)
    minimum_area_triangle_index = (
        int(np.argmin(np.asarray(triangle_area, dtype=np.float64)))
        if triangle_area
        else -1
    )
    minimum_area_local = (
        np.asarray(oriented[minimum_area_triangle_index], dtype=np.int32)
        if minimum_area_triangle_index >= 0
        else np.empty(0, dtype=np.int32)
    )
    edge_audit = _mesh_edge_length_audit(
        np.asarray(oriented, dtype=np.int32),
        local_xyz,
        point_kind,
        point_source,
        normalized_new_frontier_edges,
        new_frontier_local_edges=new_frontier_local_edges,
        existing_surface_edges=current_incidence.keys(),
    )
    maximum_edge = float(edge_audit["maximumTriangleEdgeVoxels"])
    maximum_supported_edge = float(
        edge_audit["maximumCtSupportedTriangleEdgeVoxels"]
    )
    maximum_frontier_edge = float(
        edge_audit["maximumOpenFrontierEdgeVoxels"]
    )
    maximum_inherited_edge = float(
        edge_audit["maximumInheritedBoundaryEdgeVoxels"]
    )
    maximum_residual = max(triangle_residual, default=float("inf"))
    triangle_area_array = np.asarray(triangle_area, dtype=np.float64)
    triangle_residual_array = np.asarray(triangle_residual, dtype=np.float64)
    triangle_maximum_edge_array = np.asarray(
        triangle_maximum_edge, dtype=np.float64
    )
    local_triangle_array = np.asarray(oriented, dtype=np.int32)
    dense_ct_triangle = (
        np.all(point_kind[local_triangle_array] == 1, axis=1)
        if len(local_triangle_array)
        else np.empty(0, dtype=bool)
    )
    dense_shape_ratio = (
        2.0
        * triangle_area_array
        / np.maximum(triangle_maximum_edge_array**2, 1.0e-12)
    )
    subthreshold_area = (
        triangle_area_array < settings.minimum_triangle_area_voxels_squared
    )
    resolved_dense_subthreshold = (
        subthreshold_area
        & dense_ct_triangle
        & (
            triangle_maximum_edge_array
            <= settings.maximum_native_ct_quadrature_edge_voxels + 1.0e-7
        )
        & (dense_shape_ratio >= settings.minimum_dense_triangle_shape_ratio)
        & (
            triangle_residual_array
            <= settings.high_triangle_normal_residual_degrees
        )
    )
    physically_degenerate = subthreshold_area & ~resolved_dense_subthreshold
    if np.any(physically_degenerate):
        reasons.append("completion contains a physically degenerate triangle")
    if maximum_supported_edge > settings.maximum_triangle_edge_voxels:
        reasons.append("completion contains an overlong triangle edge")
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
        maximum_residual > settings.maximum_triangle_normal_residual_degrees
        and high_normal_area_fraction
        > settings.maximum_high_normal_residual_area_fraction
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

    (
        trial,
        synthetic_node,
        patch_triangle,
        added_edge,
        local_to_global,
    ) = _append_completion_surface(
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
    if new_frontier_local_edges:
        normalized_new_frontier_edges = {
            tuple(
                sorted(
                    (
                        int(local_to_global[first]),
                        int(local_to_global[second]),
                    )
                )
            )
            for first, second in new_frontier_local_edges
        }
    trial_incidence = current_incidence.copy()
    patch_incidence = _edge_incidence(patch_triangle)
    trial_incidence.update(patch_incidence)
    boundary_edges = {
        edge for edge, count in patch_incidence.items() if count == 1
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
        reasons.append("proposed frontier mouth already exists on the surface")
    if not new_frontier_once_after:
        reasons.append("completion does not leave every mouth edge exactly open")
    if nonmanifold:
        reasons.append("completion creates a non-manifold mesh edge")

    region_merge = {
        "patchTriangleRegionCount": int(
            len(np.unique(_triangle_region_labels(patch_triangle)))
        ),
        "attachedBaselineTriangleRegionCount": 0,
        "attachedBaselineTriangleRegionIds": [],
        "unresolvedAttachmentRegionEdgeCount": 0,
        "triangleRegionReduction": 0,
        "requiredTriangleRegionReduction": int(
            required_triangle_region_reduction
        ),
        "mergesRequiredTriangleRegions": False,
        "mergesExactlyTwoTriangleRegions": False,
    }
    if required_triangle_region_reduction:
        if boundary_triangle_region is None:
            raise ValueError(
                "corridor completion requires current boundary-region provenance"
            )
        region_merge = _corridor_region_merge_audit(
            patch_triangle,
            attachment_edges,
            boundary_triangle_region,
            required_region_reduction=required_triangle_region_reduction,
        )
        if (
            int(region_merge["triangleRegionReduction"])
            != required_triangle_region_reduction
            or not bool(region_merge["mergesRequiredTriangleRegions"])
        ):
            reasons.append(
                "corridor patch does not realize its required triangle-region merge"
            )

    intersections = _other_component_triangle_intersections(
        current,
        trial,
        patch_triangle,
        component_id,
        tolerance=settings.intersection_tolerance_voxels,
        attachment_edges=attachment_edges,
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
    stale_chart_overlap = int(chart_overlap)
    chart_gauge: dict[str, Any] = {
        "mode": "inherited",
        "evaluatedGaugeCount": 0,
        "totalOverlapCount": stale_chart_overlap,
    }
    collision_reason = "completion intersects an existing selected surface"
    corridor_chart_blockers = [
        reason for reason in reasons if reason != collision_reason
    ]
    if required_triangle_region_reduction and not corridor_chart_blockers and (
        chart_overlap or required_triangle_region_reduction > 1
    ):
        if boundary_triangle_region is None:
            raise ValueError("corridor chart integration lacks region provenance")
        try:
            if (
                required_triangle_region_reduction == 1
                and first_arc_edge_count is not None
                and second_arc_edge_count is not None
            ):
                recharted_trial, chart_gauge = (
                    _reintegrate_corridor_chart_gauge(
                        current,
                        trial,
                        patch_triangle,
                        boundary,
                        mesh,
                        local_to_global,
                        boundary_triangle_region,
                        first_arc_edge_count=first_arc_edge_count,
                        second_arc_edge_count=second_arc_edge_count,
                    )
                )
            else:
                recharted_trial, chart_gauge = _reintegrate_multi_region_chart(
                    current,
                    trial,
                    patch_triangle,
                    tuple(
                        int(value)
                        for value in region_merge[
                            "attachedBaselineTriangleRegionIds"
                        ]
                    ),
                    component_id=component_id,
                )
        except ValueError as error:
            recharted_trial = None
            chart_gauge = {
                "mode": "failed",
                "failure": str(error),
                "evaluatedGaugeCount": 0,
                "totalOverlapCount": stale_chart_overlap,
            }
        if recharted_trial is None:
            reasons.append("corridor has no injective intrinsic chart gauge")
        else:
            trial = recharted_trial
            chart_overlap = int(chart_gauge["totalOverlapCount"])
            chart_gauge = {
                "mode": (
                    "rigid-region-integration"
                    if required_triangle_region_reduction == 1
                    and first_arc_edge_count is not None
                    and second_arc_edge_count is not None
                    else "conformal-multi-region-integration"
                ),
                **chart_gauge,
            }
            chart_gauge["evaluatedWithPhysicalCollision"] = bool(
                int(intersections["intersectingTrianglePairCount"])
            )
    elif chart_overlap and not required_triangle_region_reduction:
        reasons.append(
            "completion overlaps an existing triangle in its intrinsic chart"
        )
    elif chart_overlap:
        chart_gauge["mode"] = "not-evaluated-after-physical-rejection"
    topology_exact = bool(
        boundary_before_once
        and boundary_after_twice
        and new_frontier_absent_before
        and new_frontier_once_after
        and not nonmanifold
        and (
            not required_triangle_region_reduction
            or bool(region_merge["mergesRequiredTriangleRegions"])
        )
    )
    geometry = {
        "minimumTriangleAreaVoxelsSquared": round(float(minimum_area), 6),
        "minimumDenseTriangleShapeRatio": round(
            float(np.min(dense_shape_ratio[dense_ct_triangle]))
            if np.any(dense_ct_triangle)
            else float("inf"),
            6,
        ),
        "subthresholdTriangleCount": int(np.count_nonzero(subthreshold_area)),
        "resolvedDenseSubthresholdTriangleCount": int(
            np.count_nonzero(resolved_dense_subthreshold)
        ),
        "physicallyDegenerateTriangleCount": int(
            np.count_nonzero(physically_degenerate)
        ),
        "minimumAreaTriangleIndex": minimum_area_triangle_index,
        "minimumAreaTrianglePointKind": (
            [int(value) for value in point_kind[minimum_area_local]]
            if len(minimum_area_local)
            else []
        ),
        "minimumAreaTriangleSourceIndex": (
            [int(value) for value in point_source[minimum_area_local]]
            if len(minimum_area_local)
            else []
        ),
        "minimumAreaTriangleParameterUV": (
            [
                [round(float(coordinate), 6) for coordinate in value]
                for value in np.asarray(mesh["pointUV"], dtype=np.float64)[
                    minimum_area_local
                ]
            ]
            if len(minimum_area_local)
            else []
        ),
        "minimumAreaTriangleXYZ": (
            [
                [round(float(coordinate), 6) for coordinate in value]
                for value in local_xyz[minimum_area_local]
            ]
            if len(minimum_area_local)
            else []
        ),
        "maximumTriangleEdgeVoxels": round(float(maximum_edge), 6),
        "maximumCtSupportedTriangleEdgeVoxels": round(
            float(maximum_supported_edge), 6
        ),
        "maximumOpenFrontierEdgeVoxels": round(
            float(maximum_frontier_edge), 6
        ),
        "maximumInheritedBoundaryEdgeVoxels": round(
            float(maximum_inherited_edge), 6
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
        "staleIntrinsicChartOverlapCount": stale_chart_overlap,
        "chartGaugeIntegration": chart_gauge,
        "nonManifoldEdgeCount": int(nonmanifold),
        "everyBoundaryEdgeOpenBefore": bool(boundary_before_once),
        "everyBoundaryEdgeClosedAfter": bool(boundary_after_twice),
        "attachmentBoundaryEdgeCount": int(len(attachment_edges)),
        "newFrontierEdgeCount": int(len(normalized_new_frontier_edges)),
        "everyNewFrontierEdgeAbsentBefore": bool(
            new_frontier_absent_before
        ),
        "everyNewFrontierEdgeOpenAfter": bool(new_frontier_once_after),
        **region_merge,
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
        "newFrontierEdge": np.asarray(
            sorted(normalized_new_frontier_edges), dtype=np.int32
        ).reshape((-1, 2)),
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
    corridor_mode = bool(
        np.asarray(
            holes.get("denseCorridorMode", np.zeros(1, dtype=np.uint8)),
            dtype=np.uint8,
        )[0]
    )
    if open_bay_mode and corridor_mode:
        raise ValueError("completion cannot be both an open bay and a corridor")
    current_incidence = _edge_incidence(current["triangleFrontierIndex"])
    current_region_count = _component_region_count(current)
    current_boundary_region = (
        _boundary_triangle_region_catalog(current) if corridor_mode else None
    )
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
        if not open_bay_mode and not corridor_mode:
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
            -float(
                holes[
                    "denseCorridorRankScore"
                    if corridor_mode
                    else "bayGeometryObjective"
                ][loop_index[row]]
            ),
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
    collision_index = _triangle_spatial_index(
        current,
        tolerance=settings.intersection_tolerance_voxels,
        cell_size=settings.maximum_triangle_edge_voxels,
    )
    for row in ranked_rows:
        loop = int(loop_index[row])
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        pixel_slice = slice(start, stop)
        boundary_offset = np.asarray(holes["loopOffset"], dtype=np.int64)
        boundary_start = int(boundary_offset[loop])
        boundary_stop = int(boundary_offset[loop + 1])
        boundary = np.asarray(holes["loopVertexFrontierIndex"], dtype=np.int32)[
            boundary_start:boundary_stop
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
        transverse_island_diagnostics: list[dict[str, Any]] = []
        selected_variant: dict[str, Any] | None = None
        selected_hypothesis_index: int | None = None
        selected_separation: float | None = None
        last_variant: dict[str, Any] | None = None
        separation_hypotheses = (
            settings.interior_boundary_separation_hypotheses_voxels[:1]
            if open_bay_mode or corridor_mode
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
                boundary_parameter_uv=(
                    np.asarray(
                        holes["denseCorridorBoundaryParameterUV"],
                        dtype=np.float32,
                    )[boundary_start:boundary_stop]
                    if corridor_mode
                    else None
                ),
                field_parameter_uv=(
                    field_coordinates[pixel_slice].astype(np.float32)
                    * float(holes["rasterStepVoxels"][row])
                    if corridor_mode
                    else None
                ),
                field_coordinates=(
                    field_coordinates[pixel_slice]
                    if corridor_mode
                    else None
                ),
                first_arc_edge_count=(
                    int(holes["denseCorridorFirstArcEdgeCount"][loop])
                    if corridor_mode
                    else None
                ),
                second_arc_edge_count=(
                    int(holes["denseCorridorSecondArcEdgeCount"][loop])
                    if corridor_mode
                    else None
                ),
                new_frontier_edges=(
                    {
                        tuple(sorted((int(value[0]), int(value[1]))))
                        for value in np.asarray(
                            holes["denseCorridorMouthFrontierIndex"],
                            dtype=np.int32,
                        )[loop]
                    }
                    if corridor_mode
                    else (
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
                    )
                ),
                boundary_triangle_region=current_boundary_region,
                required_triangle_region_reduction=(1 if corridor_mode else 0),
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
                    "attachmentCollar": variant.get("attachmentCollar"),
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

        # Preserve the unmodified, complete two-frontier construction as the
        # geometric reference for any third-region sector hypothesis.  A
        # later attachment collar changes only one seam and must not redefine
        # which transverse island the original strip encountered.
        corridor_base_variant = last_variant if corridor_mode else None

        if (
            corridor_mode
            and selected_variant is None
            and not depth_reasons
            and last_variant is not None
        ):
            row_coordinates = field_coordinates[pixel_slice]
            raster_step = float(holes["rasterStepVoxels"][row])
            collar_domain = _derive_attachment_collar_domain(
                last_variant,
                row_coordinates,
                raster_step_voxels=raster_step,
                transition_voxels=settings.attachment_collar_transition_voxels,
                intersection_tolerance_voxels=(
                    settings.intersection_tolerance_voxels
                ),
            )
            if collar_domain is not None:
                boundary_parameter = np.asarray(
                    holes["denseCorridorBoundaryParameterUV"], dtype=np.float32
                )[boundary_start:boundary_stop]
                field_parameter = row_coordinates.astype(np.float32) * raster_step
                first_arc_edge_count = int(
                    holes["denseCorridorFirstArcEdgeCount"][loop]
                )
                second_arc_edge_count = int(
                    holes["denseCorridorSecondArcEdgeCount"][loop]
                )
                mouth_edges = {
                    tuple(sorted((int(value[0]), int(value[1]))))
                    for value in np.asarray(
                        holes["denseCorridorMouthFrontierIndex"], dtype=np.int32
                    )[loop]
                }
                for tangent_ratio in (
                    settings.attachment_collar_outward_tangent_ratio_hypotheses
                ):
                    collar_xyz, collar_statistics = (
                        _apply_attachment_halfspace_collar(
                            current,
                            boundary,
                            boundary_parameter,
                            field_parameter,
                            row_coordinates,
                            field_xyz,
                            collar_domain,
                            first_arc_edge_count=first_arc_edge_count,
                            second_arc_edge_count=second_arc_edge_count,
                            minimum_outward_tangent_ratio=float(tangent_ratio),
                        )
                    )
                    hypothesis_index = len(hypothesis_records)
                    separation = float(separation_hypotheses[0])
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
                        field_xyz=collar_xyz,
                        field_reference=field_reference,
                        boundary_separation_voxels=separation,
                        boundary_parameter_uv=boundary_parameter,
                        field_parameter_uv=field_parameter,
                        field_coordinates=row_coordinates,
                        first_arc_edge_count=first_arc_edge_count,
                        second_arc_edge_count=second_arc_edge_count,
                        new_frontier_edges=mouth_edges,
                        boundary_triangle_region=current_boundary_region,
                        required_triangle_region_reduction=1,
                        collision_index=collision_index,
                        settings=settings,
                    )
                    variant["attachmentCollar"] = collar_statistics
                    if variant.get("meshStatistics") is not None:
                        variant["meshStatistics"] = {
                            **variant["meshStatistics"],
                            "attachmentCollar": collar_statistics,
                        }
                    if variant.get("geometry") is not None:
                        variant["geometry"] = {
                            **variant["geometry"],
                            "attachmentCollar": collar_statistics,
                        }
                    last_variant = variant
                    variant_reasons = list(variant["rejectionReasons"])
                    hypothesis_records.append(
                        {
                            "hypothesisIndex": hypothesis_index,
                            "boundarySeparationVoxels": separation,
                            "accepted": not variant_reasons,
                            "rejectionReasons": variant_reasons,
                            "mesh": variant.get("meshStatistics"),
                            "geometry": variant.get("geometry"),
                            "nativeCt": variant.get("nativeStatistics"),
                            "attachmentCollar": collar_statistics,
                        }
                    )
                    if not variant_reasons:
                        selected_variant = variant
                        selected_hypothesis_index = hypothesis_index
                        selected_separation = separation
                        break

        if (
            corridor_mode
            and selected_variant is None
            and not depth_reasons
            and corridor_base_variant is not None
            and corridor_base_variant.get("constructed")
        ):
            row_coordinates = field_coordinates[pixel_slice]
            raster_step = float(holes["rasterStepVoxels"][row])
            boundary_parameter = np.asarray(
                holes["denseCorridorBoundaryParameterUV"], dtype=np.float32
            )[boundary_start:boundary_stop]
            field_parameter = row_coordinates.astype(np.float32) * raster_step
            sector_domains, transverse_island_diagnostics = (
                _derive_transverse_island_sector_domains(
                    current,
                    corridor_base_variant,
                    component_id=component_id,
                    boundary=boundary,
                    boundary_parameter_uv=boundary_parameter,
                    field_parameter_uv=field_parameter,
                    field_coordinates=row_coordinates,
                    field_xyz=field_xyz,
                    first_arc_edge_count=int(
                        holes["denseCorridorFirstArcEdgeCount"][loop]
                    ),
                    second_arc_edge_count=int(
                        holes["denseCorridorSecondArcEdgeCount"][loop]
                    ),
                    minimum_boundary_separation=float(
                        separation_hypotheses[0]
                    ),
                    maximum_edge_flip_iterations=(
                        settings.maximum_edge_flip_iterations
                    ),
                )
            )
            for sector_domain in sector_domains:
                sector_surface = sector_domain["surface"]
                sector_incidence = _edge_incidence(
                    sector_surface["triangleFrontierIndex"]
                )
                sector_boundary_region = _boundary_triangle_region_catalog(
                    sector_surface
                )
                sector_collision_index = _triangle_spatial_index(
                    sector_surface,
                    tolerance=settings.intersection_tolerance_voxels,
                    cell_size=settings.maximum_triangle_edge_voxels,
                )
                hypothesis_index = len(hypothesis_records)
                separation = float(separation_hypotheses[0])
                variant = _evaluate_dense_completion_variant(
                    sector_surface,
                    sector_incidence,
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
                    boundary_separation_voxels=separation,
                    new_frontier_edges=frozenset(),
                    boundary_triangle_region=sector_boundary_region,
                    required_triangle_region_reduction=2,
                    collision_index=sector_collision_index,
                    mesh_override=(
                        sector_domain["mesh"],
                        sector_domain["meshStatistics"],
                    ),
                    settings=settings,
                )
                variant["multiRegionSector"] = sector_domain["statistics"]
                if variant.get("meshStatistics") is not None:
                    variant["meshStatistics"] = {
                        **variant["meshStatistics"],
                        "multiRegionSector": sector_domain["statistics"],
                    }
                if variant.get("geometry") is not None:
                    variant["geometry"] = {
                        **variant["geometry"],
                        "multiRegionSector": sector_domain["statistics"],
                    }
                last_variant = variant
                variant_reasons = list(variant["rejectionReasons"])
                hypothesis_records.append(
                    {
                        "hypothesisIndex": hypothesis_index,
                        "boundarySeparationVoxels": separation,
                        "accepted": not variant_reasons,
                        "rejectionReasons": variant_reasons,
                        "mesh": variant.get("meshStatistics"),
                        "geometry": variant.get("geometry"),
                        "nativeCt": variant.get("nativeStatistics"),
                        "attachmentCollar": None,
                        "multiRegionSector": sector_domain["statistics"],
                    }
                )
                if not variant_reasons:
                    selected_variant = variant
                    selected_hypothesis_index = hypothesis_index
                    selected_separation = separation
                    break
                sector_collar_domain = _derive_attachment_collar_domain(
                    variant,
                    row_coordinates,
                    raster_step_voxels=raster_step,
                    transition_voxels=(
                        settings.attachment_collar_transition_voxels
                    ),
                    intersection_tolerance_voxels=(
                        settings.intersection_tolerance_voxels
                    ),
                    allowed_additional_reasons={
                        "corridor has no injective intrinsic chart gauge"
                    },
                    minimum_attached_region_count=3,
                    require_injective_chart=False,
                )
                if sector_collar_domain is None:
                    continue
                sector_collar_domain = {
                    **sector_collar_domain,
                    "mode": "multi-region-sector-attachment-collar",
                }
                for tangent_ratio in (
                    settings.attachment_collar_outward_tangent_ratio_hypotheses
                ):
                    collar_xyz, collar_statistics = (
                        _apply_attachment_halfspace_collar(
                            sector_surface,
                            boundary,
                            boundary_parameter,
                            field_parameter,
                            row_coordinates,
                            field_xyz,
                            sector_collar_domain,
                            first_arc_edge_count=int(
                                holes["denseCorridorFirstArcEdgeCount"][loop]
                            ),
                            second_arc_edge_count=int(
                                holes["denseCorridorSecondArcEdgeCount"][loop]
                            ),
                            minimum_outward_tangent_ratio=float(tangent_ratio),
                        )
                    )
                    collar_hypothesis_index = len(hypothesis_records)
                    collar_variant = _evaluate_dense_completion_variant(
                        sector_surface,
                        sector_incidence,
                        holes,
                        source,
                        volume,
                        row=row,
                        component_id=component_id,
                        thickness=thickness,
                        boundary=boundary,
                        field_uv=field_uv,
                        field_xyz=collar_xyz,
                        field_reference=field_reference,
                        boundary_separation_voxels=separation,
                        new_frontier_edges=frozenset(),
                        boundary_triangle_region=sector_boundary_region,
                        required_triangle_region_reduction=2,
                        collision_index=sector_collision_index,
                        mesh_override=(
                            sector_domain["mesh"],
                            sector_domain["meshStatistics"],
                        ),
                        settings=settings,
                    )
                    collar_variant["attachmentCollar"] = collar_statistics
                    collar_variant["multiRegionSector"] = sector_domain[
                        "statistics"
                    ]
                    if collar_variant.get("meshStatistics") is not None:
                        collar_variant["meshStatistics"] = {
                            **collar_variant["meshStatistics"],
                            "attachmentCollar": collar_statistics,
                            "multiRegionSector": sector_domain["statistics"],
                        }
                    if collar_variant.get("geometry") is not None:
                        collar_variant["geometry"] = {
                            **collar_variant["geometry"],
                            "attachmentCollar": collar_statistics,
                            "multiRegionSector": sector_domain["statistics"],
                        }
                    last_variant = collar_variant
                    collar_reasons = list(
                        collar_variant["rejectionReasons"]
                    )
                    hypothesis_records.append(
                        {
                            "hypothesisIndex": collar_hypothesis_index,
                            "boundarySeparationVoxels": separation,
                            "accepted": not collar_reasons,
                            "rejectionReasons": collar_reasons,
                            "mesh": collar_variant.get("meshStatistics"),
                            "geometry": collar_variant.get("geometry"),
                            "nativeCt": collar_variant.get(
                                "nativeStatistics"
                            ),
                            "attachmentCollar": collar_statistics,
                            "multiRegionSector": sector_domain["statistics"],
                        }
                    )
                    if not collar_reasons:
                        selected_variant = collar_variant
                        selected_hypothesis_index = collar_hypothesis_index
                        selected_separation = separation
                        break
                if selected_variant is not None:
                    break

        target_was_macro = bool(
            np.asarray(holes["loopMacroEligible"], dtype=np.uint8)[loop]
        ) and not open_bay_mode and not corridor_mode
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
                    "selectedAttachmentCollar": None,
                    "selectedMultiRegionSector": None,
                    "hypotheses": hypothesis_records,
                    "transverseIslandDiagnostics": (
                        transverse_island_diagnostics
                    ),
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
                        "outerLoopCountBefore": current_loop_count[
                            "outerLoopCount"
                        ],
                        "outerLoopCountAfter": current_loop_count[
                            "outerLoopCount"
                        ],
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
                        "targetWasTwoFrontierCorridor": corridor_mode,
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
        new_frontier_edge = np.asarray(
            selected_variant["newFrontierEdge"], dtype=np.int32
        ).reshape((-1, 2))
        trial_incidence = selected_variant["trialIncidence"]
        retained_pixel = np.asarray(
            selected_variant["retainedFieldPixel"], dtype=np.int32
        )
        field_audit_count = int(selected_variant["fieldAuditCount"])
        native_arrays = selected_variant["nativeArrays"]
        native_stats = selected_variant["nativeStatistics"]
        mesh_stats = selected_variant["meshStatistics"]
        geometry_record = selected_variant["geometry"]
        required_region_reduction = int(
            geometry_record.get(
                "requiredTriangleRegionReduction", 1 if corridor_mode else 0
            )
        )

        trial_loop_count = dict(current_loop_count)
        trial_boundary_edge_count = predicted_boundary_edge_count
        if corridor_mode:
            attachment_count = int(
                geometry_record["attachmentBoundaryEdgeCount"]
            )
            trial_loop_count["outerLoopCount"] -= required_region_reduction
            trial_boundary_edge_count += len(new_frontier_edge) - attachment_count
        elif open_bay_mode:
            # K existing arc edges are replaced by one new mouth edge.  The
            # boundary array has K+1 vertices because the mouth is implicit.
            trial_boundary_edge_count -= len(boundary) - 2
        else:
            trial_loop_count["interiorHoleCount"] -= 1
            trial_loop_count["macroHoleCount"] -= int(target_was_macro)
            trial_boundary_edge_count -= len(boundary)
        component_region_count_before = current_region_count.get(
            component_id, 0
        )
        trial_region = (
            _component_region_count(trial)
            if corridor_mode
            else current_region_count
        )
        if corridor_mode and (
            component_region_count_before
            - trial_region.get(component_id, 0)
            != required_region_reduction
        ):
            raise ValueError(
                "accepted corridor local region audit differs from the exact "
                "materialized surface"
            )

        current = trial
        current_incidence = trial_incidence
        current_region_count = trial_region
        collision_index = _triangle_spatial_index(
            current,
            tolerance=settings.intersection_tolerance_voxels,
            cell_size=settings.maximum_triangle_edge_voxels,
        )
        if corridor_mode:
            current_boundary_region = _boundary_triangle_region_catalog(current)
        predicted_loop_count = trial_loop_count
        predicted_boundary_edge_count = trial_boundary_edge_count
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
                "selectedAttachmentCollar": selected_variant.get(
                    "attachmentCollar"
                ),
                "selectedMultiRegionSector": selected_variant.get(
                    "multiRegionSector"
                ),
                "hypotheses": hypothesis_records,
                "transverseIslandDiagnostics": transverse_island_diagnostics,
                "depthFieldSupportedFraction": round(depth_support_fraction, 6),
                "depthFieldIntegrability": field_integrability[row],
                "mesh": mesh_stats,
                "geometry": geometry_record,
                "nativeCt": native_stats,
                "topology": {
                    "outerLoopCountBefore": current_loop_count[
                        "outerLoopCount"
                    ],
                    "outerLoopCountAfter": trial_loop_count[
                        "outerLoopCount"
                    ],
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
                    "targetWasTwoFrontierCorridor": corridor_mode,
                    "boundaryEdgeCountBefore": (
                        predicted_boundary_edge_count
                        + (
                            attachment_count - len(new_frontier_edge)
                            if corridor_mode
                            else (
                                len(boundary) - 2
                                if open_bay_mode
                                else len(boundary)
                            )
                        )
                    ),
                    "boundaryEdgeCountAfter": trial_boundary_edge_count,
                    "componentTriangleRegionCountBefore": (
                        component_region_count_before
                    ),
                    "componentTriangleRegionCountAfter": (
                        trial_region.get(component_id, 0)
                    ),
                    "requiredTriangleRegionReduction": (
                        required_region_reduction
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
    # Rebuild the exact 1-skeleton once from the accepted triangle complex.
    # This is essential after a topology-normalized island is re-welded: its
    # abandoned clone edges must disappear, while every accepted dense edge is
    # represented exactly once.
    exact_edge = np.asarray(
        sorted(_edge_incidence(current["triangleFrontierIndex"])),
        dtype=np.int32,
    ).reshape((-1, 2))
    current["edgeFirstFrontierIndex"] = exact_edge[:, 0]
    current["edgeSecondFrontierIndex"] = exact_edge[:, 1]
    current["edgeSelected"] = np.ones(len(exact_edge), dtype=np.uint8)
    current["integrationResidualVoxels"] = np.zeros(
        len(exact_edge), dtype=np.float32
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
    collar_hypotheses = [
        hypothesis
        for record in records
        for hypothesis in record.get("hypotheses", ())
        if hypothesis.get("attachmentCollar") is not None
    ]
    multi_region_sector_hypotheses = [
        hypothesis
        for record in records
        for hypothesis in record.get("hypotheses", ())
        if hypothesis.get("multiRegionSector") is not None
    ]
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
        "outerLoopCountBefore": loop_count_before["outerLoopCount"],
        "outerLoopCountAfter": final_loop_count["outerLoopCount"],
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
        "adaptiveMeshHypotheses": not open_bay_mode and not corridor_mode,
        "openBayFullResolutionOnly": open_bay_mode,
        "surfaceCorridorFullResolutionOnly": corridor_mode,
        "frontierStateRanking": (
            "depth-field readiness, surface integrability, CT support, "
            "profile correlation, far-layer margin, then geometry"
            if open_bay_mode or corridor_mode
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
        "attachmentCollarEligibleHoleCount": int(
            sum(
                any(
                    hypothesis.get("attachmentCollar") is not None
                    for hypothesis in record.get("hypotheses", ())
                )
                for record in records
            )
        ),
        "attachmentCollarHypothesisCount": len(collar_hypotheses),
        "attachmentCollarAcceptedHoleCount": int(
            sum(
                record.get("selectedAttachmentCollar") is not None
                for record in accepted_records
            )
        ),
        "multiRegionSectorEligibleHoleCount": int(
            sum(
                any(
                    hypothesis.get("multiRegionSector") is not None
                    for hypothesis in record.get("hypotheses", ())
                )
                for record in records
            )
        ),
        "multiRegionSectorHypothesisCount": len(
            multi_region_sector_hypotheses
        ),
        "multiRegionSectorAcceptedHoleCount": int(
            sum(
                record.get("selectedMultiRegionSector") is not None
                for record in accepted_records
            )
        ),
        "selectedBoundarySeparationCount": dict(
            sorted(selected_separation_count.items(), key=lambda item: float(item[0]))
        ),
        "finalBoundaryAudit": final_loop_stats,
        "decisionUnit": (
            "two complete outer-boundary arcs and both replacement mouths; "
            "when a transverse island exactly terminates on both fronts, one "
            "whole island sector is an indivisible three-region alternative"
            if corridor_mode
            else (
                "one complete outer-boundary arc and replacement mouth"
                if open_bay_mode
                else "one complete weakly-simple closed boundary and its collective "
                "CT depth field"
            )
        ),
        "openBayMode": open_bay_mode,
        "surfaceCorridorMode": corridor_mode,
        "singlePixelGrowth": False,
        "ribbonCandidatesRequiredForInterior": False,
        "fittedNormalTailIsDiagnosticOnly": False,
        "fittedNormalContradictionGate": (
            "reject only when an extreme residual exceeds the hard angle "
            "and the softer high-residual tail exceeds its area fraction"
        ),
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
    if holes_manifest.get("schema") == PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA:
        holes = surface_corridor_completion_view(holes)
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
    if holes_manifest.get("schema") == PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA:
        nested_corridor_settings = hole_setting_values.get("corridors", {})
        hole_setting_values = (
            dict(nested_corridor_settings)
            if isinstance(nested_corridor_settings, Mapping)
            else {}
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
        "implementationSha256": {
            "denseCompletion": sha256_file(Path(__file__)),
            "conformalChart": sha256_file(Path(conformal_chart.__code__.co_filename)),
            "surfaceView": sha256_file(Path(_surface_view.__code__.co_filename)),
            "surfaceTopology": sha256_file(
                Path(triangle_edge_region_labels.__code__.co_filename)
            ),
            **(
                {
                    "surfaceCorridorAdapter": sha256_file(
                        Path(surface_corridor_completion_view.__code__.co_filename)
                    )
                }
                if holes_manifest.get("schema")
                == PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA
                else {}
            ),
        },
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
                "one complete paired-frontier domain; an eligible transverse "
                "island is enumerated only as a complete three-region sector"
                if statistics["surfaceCorridorMode"]
                else (
                    "one complete outer-boundary arc and one replacement mouth"
                    if statistics["openBayMode"]
                    else "one complete closed boundary and collective dense CT "
                    "normal-depth field"
                )
            ),
            "boundaryGeometry": (
                "a corridor closes both inherited multi-edge arcs while "
                "leaving sampled end mouths open; an exact transverse island "
                "endpoint match may replace one mouth-side boundary by either "
                "complete island path, never by local cell growth"
                if statistics["surfaceCorridorMode"]
                else (
                    "an open bay closes every inherited arc edge while leaving "
                    "its one new mouth edge open"
                    if statistics["openBayMode"]
                    else "weakly-simple loops are decomposed at pinch vertices "
                    "into exact edge-preserving disk cycles"
                )
            ),
            "surfaceRepresentation": (
                "constrained intrinsic triangles through dense CT field "
                "samples; ribbon-bank nodes are boundary and collision "
                "evidence only"
            ),
            "denseTriangleQuality": (
                "the absolute area floor applies to inherited or long-edge "
                "triangles; sub-floor all-CT triangles are admissible only at "
                "the native quadrature scale when their dimensionless shape "
                "ratio and local-normal residual pass explicit gates"
            ),
            "adaptiveMeshDensity": (
                "frontier strips retain the full depth-field resolution; a coarser "
                "retry cannot hide a non-integrable or unsupported frontier "
                "expansion"
                if statistics["openBayMode"]
                or statistics["surfaceCorridorMode"]
                else "each full boundary is tested densest-first at declared "
                "boundary-separation scales; the first complete mesh whose "
                "entire realized area passes native CT and exact topology is "
                "selected, rather than growing individual pixels or cells"
            ),
            "depthFieldIntegrability": (
                "complete 2x2 raster cells test whether pointwise-supported "
                "depth assignments form a realizable surface and rank complete "
                "frontier states before constrained meshing"
                if statistics["openBayMode"]
                or statistics["surfaceCorridorMode"]
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
                "fitted-normal disagreement is a bounded catastrophic gate: "
                "an extreme hard-angle residual vetoes a patch only when the "
                "softer high-residual tail also covers too much realized "
                "area; thickness-normalized proximity remains diagnostic, "
                "so tight bends and compressed or delaminated sheets remain "
                "admissible when direct CT support and exact intersections "
                "agree"
            ),
            "topologyAudit": (
                "every declared inherited attachment becomes two-incident, "
                "every replacement mouth remains one-incident, ordinary "
                "corridors prove a 2-to-1 region merge, transverse-island "
                "sectors prove a 3-to-1 merge, and no edge exceeds two faces"
                if statistics["surfaceCorridorMode"]
                else (
                    "every inherited arc edge becomes exactly two-incident, the "
                    "new mouth is exactly one-incident, total boundary length "
                    "falls, loop counts remain fixed, and no triangle region is "
                    "created"
                    if statistics["openBayMode"]
                    else "every prior boundary edge becomes exactly two-incident, "
                    "no edge exceeds two faces, the target macro/interior loop "
                    "disappears, and no triangle region is created"
                )
            ),
            "openFrontierScale": (
                "the declared maximum triangle edge applies to CT-supported "
                "interior and attachment edges; the replacement mouth is an "
                "open frontier, is reported separately, and remains covered "
                "by uniform native-CT area quadrature"
                if statistics["openBayMode"]
                or statistics["surfaceCorridorMode"]
                else "all completion edges use the declared maximum"
            ),
            "attachmentCollar": (
                "an otherwise valid corridor whose exhaustive crossings are "
                "confined to its own declared attachment regions within one "
                "raster step may retry as a complete half-space collar; collision "
                "depth sets the collar width, one physical voxel supplies the "
                "transition, progressively stronger exit slopes are audited, "
                "and every CT, topology, collision, chart, and texture gate "
                "remains unchanged"
                if statistics["surfaceCorridorMode"]
                else "not used outside complete paired-frontier strips"
            ),
            "multiRegionSector": (
                "a third colliding triangle region is considered only when it "
                "is one complete disk island with one exact weld-group endpoint "
                "on each front; both island boundary paths and both complete "
                "strip sectors are enumerated, topology-only clone welds move "
                "no geometry, and admission requires an exact collision-free "
                "3-to-1 merge with an injective joined chart"
                if statistics["surfaceCorridorMode"]
                else "not used outside complete paired-frontier strips"
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
