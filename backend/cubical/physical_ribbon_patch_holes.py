from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from ..rectify import _trilinear
from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import _draw_text
from .needle_surface import (
    integrate_intrinsic_surface_charts,
    triangulate_intrinsic_surface_charts,
)
from .physical_ribbon_bridging import _load_inputs, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_continuity import _draw_line


PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA = "pareidolia.physical-ribbon-patch-holes"
PHYSICAL_RIBBON_PATCH_HOLES_VERSION = 1
PHYSICAL_RIBBON_PATCH_HOLES_STEM = "physical-ribbon-patch-holes-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonPatchHoleSettings:
    minimum_component_ribbon_count: int = 32
    minimum_chart_separation_voxels: float = 0.20
    maximum_triangle_edge_voxels: float = 5.75
    maximum_triangle_normal_residual_degrees: float = 45.0
    minimum_triangle_area_voxels_squared: float = 0.20
    minimum_hole_boundary_vertex_count: int = 6
    minimum_hole_diameter_boundary_edges: float = 1.50
    minimum_hole_area_boundary_edges_squared: float = 1.00
    context_graph_hops: int = 2
    patch_pixel_step_voxels: float = 1.0
    maximum_patch_pixels: int = 4096
    maximum_scored_holes: int = 64
    maximum_preview_holes: int = 12
    quadratic_ridge: float = 0.015
    maximum_candidate_height_thicknesses: float = 0.25
    maximum_candidate_tangent_raster_steps: float = 1.75
    maximum_candidate_normal_degrees: float = 35.0
    maximum_candidate_thickness_ratio: float = 1.65
    surface_alignment_weight: float = 0.35
    configuration_beam_width: int = 16384
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
        if self.minimum_component_ribbon_count < 3:
            raise ValueError("patch holes require a nontrivial seed component")
        positive = (
            self.minimum_chart_separation_voxels,
            self.maximum_triangle_edge_voxels,
            self.maximum_triangle_normal_residual_degrees,
            self.minimum_triangle_area_voxels_squared,
            self.minimum_hole_diameter_boundary_edges,
            self.minimum_hole_area_boundary_edges_squared,
            self.patch_pixel_step_voxels,
            self.quadratic_ridge,
            self.maximum_candidate_height_thicknesses,
            self.maximum_candidate_tangent_raster_steps,
            self.maximum_candidate_normal_degrees,
            self.maximum_candidate_thickness_ratio,
            self.surface_alignment_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("patch-hole geometric scales must be finite and positive")
        if self.maximum_triangle_normal_residual_degrees >= 90.0:
            raise ValueError("triangle normal residual must be below 90 degrees")
        if self.minimum_hole_boundary_vertex_count < 3:
            raise ValueError("hole loops require at least three vertices")
        if self.context_graph_hops < 1:
            raise ValueError("patch context must include at least one graph hop")
        if self.maximum_patch_pixels < 64:
            raise ValueError("patch raster cap is too small")
        if self.maximum_scored_holes < 1 or self.maximum_preview_holes < 1:
            raise ValueError("patch-hole output counts must be positive")
        if self.maximum_candidate_normal_degrees >= 90.0:
            raise ValueError("candidate normal gate must be below 90 degrees")
        if self.maximum_candidate_thickness_ratio <= 1.0:
            raise ValueError("candidate thickness ratio must exceed one")
        if self.configuration_beam_width < 256:
            raise ValueError("configuration beam is too narrow for joint moves")
        if (
            len(self.profile_depth_fractions) < 5
            or tuple(sorted(self.profile_depth_fractions))
            != self.profile_depth_fractions
            or 0.0 not in self.profile_depth_fractions
        ):
            raise ValueError("profile depths must be sorted and include zero")
        if (
            len(self.competing_shift_thicknesses) < 3
            or tuple(sorted(self.competing_shift_thicknesses))
            != self.competing_shift_thicknesses
            or 0.0 not in self.competing_shift_thicknesses
        ):
            raise ValueError("competing shifts must be sorted and include zero")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _component_size(component: np.ndarray) -> np.ndarray:
    present = component[component >= 0]
    if not len(present):
        return np.empty(0, dtype=np.int32)
    return np.bincount(present).astype(np.int32)


def _orient_normals_and_frames(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    selected: np.ndarray,
    component: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    active_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    count = len(center_xyz)
    signed_normal = np.asarray(normal_xyz, dtype=np.float64).copy()
    tangent_u = np.full((count, 3), np.nan, dtype=np.float64)
    tangent_v = np.full((count, 3), np.nan, dtype=np.float64)
    adjacency: dict[int, list[int]] = defaultdict(list)
    for left, right in zip(first[active_edge], second[active_edge]):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    normal_flips = 0
    transported_nodes = 0
    disconnected_roots = 0
    for component_id in np.unique(component[selected]):
        if component_id < 0:
            continue
        nodes = set(int(value) for value in np.flatnonzero(component == component_id))
        while nodes:
            root = min(nodes)
            disconnected_roots += 1
            normal = signed_normal[root]
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            signed_normal[root] = normal
            axis = np.eye(3)[int(np.argmin(np.abs(normal)))]
            u_value = np.cross(normal, axis)
            u_value /= max(float(np.linalg.norm(u_value)), 1.0e-12)
            tangent_u[root] = u_value
            tangent_v[root] = np.cross(normal, u_value)
            queue: deque[int] = deque((root,))
            nodes.remove(root)
            transported_nodes += 1
            while queue:
                current = queue.popleft()
                for neighbor in adjacency.get(current, ()):
                    if component[neighbor] != component_id or neighbor not in nodes:
                        continue
                    candidate_normal = signed_normal[neighbor]
                    candidate_normal /= max(
                        float(np.linalg.norm(candidate_normal)), 1.0e-12
                    )
                    if float(np.dot(candidate_normal, signed_normal[current])) < 0.0:
                        candidate_normal *= -1.0
                        normal_flips += 1
                    signed_normal[neighbor] = candidate_normal
                    transported = tangent_u[current] - float(
                        np.dot(tangent_u[current], candidate_normal)
                    ) * candidate_normal
                    length = float(np.linalg.norm(transported))
                    if length <= 1.0e-8:
                        transported = tangent_v[current] - float(
                            np.dot(tangent_v[current], candidate_normal)
                        ) * candidate_normal
                        length = float(np.linalg.norm(transported))
                    if length <= 1.0e-8:
                        axis = np.eye(3)[int(np.argmin(np.abs(candidate_normal)))]
                        transported = np.cross(candidate_normal, axis)
                        length = float(np.linalg.norm(transported))
                    transported /= max(length, 1.0e-12)
                    tangent_u[neighbor] = transported
                    tangent_v[neighbor] = np.cross(candidate_normal, transported)
                    nodes.remove(neighbor)
                    queue.append(neighbor)
                    transported_nodes += 1
    return signed_normal, tangent_u, tangent_v, {
        "normalSignFlips": normal_flips,
        "transportedNodeCount": transported_nodes,
        "frameConnectedRootCount": disconnected_roots,
    }


def build_physical_ribbon_surface_complex(
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    center = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    normal = np.asarray(ribbon["normalXYZ"], dtype=np.float32)[frontier]
    thickness = np.asarray(ribbon["thicknessVoxels"], dtype=np.float32)[frontier]
    selected = np.asarray(configuration["selected"]) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    low = np.minimum(first, second)
    high = np.maximum(first, second)
    first, second = low, high
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    active_edge = (
        selected[first]
        & selected[second]
        & (component[first] >= 0)
        & (component[first] == component[second])
    )
    component_size_values = _component_size(component)
    node_component_size = np.zeros(len(frontier), dtype=np.int32)
    valid_component = component >= 0
    node_component_size[valid_component] = component_size_values[
        component[valid_component]
    ]
    signed_normal, tangent_u, tangent_v, frame_stats = (
        _orient_normals_and_frames(
            center,
            normal,
            selected,
            component,
            first,
            second,
            active_edge,
        )
    )
    # Only selected topology edges have transported frames.  Keeping the
    # inactive edge records finite avoids allowing NaNs from unrelated ribbon
    # hypotheses to leak into the graph-coordinate solve.
    delta = center[second] - center[first]
    along = np.zeros(len(first), dtype=np.float32)
    across = np.zeros(len(first), dtype=np.float32)
    active_index = np.flatnonzero(active_edge)
    average_u = tangent_u[first[active_index]] + tangent_u[second[active_index]]
    average_u /= np.maximum(
        np.linalg.norm(average_u, axis=1, keepdims=True), 1.0e-12
    )
    average_normal = (
        signed_normal[first[active_index]] + signed_normal[second[active_index]]
    )
    average_normal /= np.maximum(
        np.linalg.norm(average_normal, axis=1, keepdims=True), 1.0e-12
    )
    average_v = np.cross(average_normal, average_u)
    average_v /= np.maximum(
        np.linalg.norm(average_v, axis=1, keepdims=True), 1.0e-12
    )
    along[active_index] = np.einsum(
        "ij,ij->i", delta[active_index], average_u
    ).astype(np.float32)
    across[active_index] = np.einsum(
        "ij,ij->i", delta[active_index], average_v
    ).astype(np.float32)
    chart_uv, integration_residual, chart_stats = integrate_intrinsic_surface_charts(
        center,
        first,
        second,
        edge_score,
        active_edge,
        component,
        node_component_size,
        along,
        across,
        minimum_component_needles=settings.minimum_component_ribbon_count,
    )
    triangles, triangle_area, triangle_normal_residual, triangle_stats = (
        triangulate_intrinsic_surface_charts(
            center,
            signed_normal,
            chart_uv,
            first,
            second,
            edge_score,
            active_edge,
            active_edge,
            component,
            node_component_size,
            minimum_component_needles=settings.minimum_component_ribbon_count,
            minimum_chart_separation_voxels=(
                settings.minimum_chart_separation_voxels
            ),
            maximum_edge_voxels=settings.maximum_triangle_edge_voxels,
            maximum_normal_residual_degrees=(
                settings.maximum_triangle_normal_residual_degrees
            ),
            minimum_area_voxels_squared=(
                settings.minimum_triangle_area_voxels_squared
            ),
        )
    )
    arrays = {
        "frontierRibbonCandidate": frontier,
        "selected": selected.astype(np.uint8),
        "component": component,
        "componentSize": node_component_size,
        "signedNormalXYZ": signed_normal.astype(np.float32),
        "tangentUxyz": tangent_u.astype(np.float32),
        "tangentVxyz": tangent_v.astype(np.float32),
        "chartUV": chart_uv.astype(np.float32),
        "integrationResidualVoxels": integration_residual,
        "edgeFirstFrontierIndex": first,
        "edgeSecondFrontierIndex": second,
        "edgeSelected": active_edge.astype(np.uint8),
        "triangleFrontierIndex": triangles,
        "triangleAreaVoxelsSquared": triangle_area,
        "triangleNormalResidualDegrees": triangle_normal_residual,
        "midpointXYZ": center,
        "thicknessVoxels": thickness,
    }
    return arrays, {
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "eligibleComponentCount": int(
            np.count_nonzero(component_size_values >= settings.minimum_component_ribbon_count)
        ),
        **frame_stats,
        "chartIntegration": chart_stats,
        "triangulation": triangle_stats,
        "identityLabelsUsed": False,
    }


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool:
    inside = False
    previous = polygon[-1]
    for current in polygon:
        crosses = (current[1] > point[1]) != (previous[1] > point[1])
        if crosses:
            # ``crosses`` guarantees a non-zero signed denominator.  Its sign
            # is geometrically meaningful, so it must not be clamped positive.
            x_value = (previous[0] - current[0]) * (
                point[1] - current[1]
            ) / (previous[1] - current[1]) + current[0]
            if point[0] < x_value:
                inside = not inside
        previous = current
    return inside


def _polygon_area(polygon: np.ndarray) -> float:
    return 0.5 * float(
        np.sum(
            polygon[:, 0] * np.roll(polygon[:, 1], -1)
            - polygon[:, 1] * np.roll(polygon[:, 0], -1)
        )
    )


def extract_surface_boundary_loops(
    surface: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    component = np.asarray(surface["component"], dtype=np.int32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, left_value in enumerate(triangle):
            right_value = triangle[(edge_index + 1) % 3]
            edge = (min(int(left_value), int(right_value)), max(int(left_value), int(right_value)))
            edge_triangle[edge].append(triangle_index)
    triangle_parent = np.arange(len(triangles), dtype=np.int32)

    def find(value: int) -> int:
        while triangle_parent[value] != value:
            triangle_parent[value] = triangle_parent[triangle_parent[value]]
            value = int(triangle_parent[value])
        return value

    for records in edge_triangle.values():
        if len(records) != 2:
            continue
        first_root = find(records[0])
        second_root = find(records[1])
        if first_root != second_root:
            triangle_parent[max(first_root, second_root)] = min(first_root, second_root)
    boundary_by_region: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for edge, records in edge_triangle.items():
        if len(records) == 1:
            boundary_by_region[find(records[0])].append(edge)
    loop_values: list[list[int]] = []
    loop_region: list[int] = []
    ambiguous_boundary_components = 0
    for region, edges in boundary_by_region.items():
        graph: dict[int, list[int]] = defaultdict(list)
        for left, right in edges:
            graph[left].append(right)
            graph[right].append(left)
        unseen = set(graph)
        while unseen:
            seed = min(unseen)
            group = {seed}
            queue = [seed]
            unseen.remove(seed)
            while queue:
                current = queue.pop()
                for neighbor in graph[current]:
                    if neighbor in unseen:
                        unseen.remove(neighbor)
                        group.add(neighbor)
                        queue.append(neighbor)
            if not all(len(graph[value]) == 2 for value in group):
                ambiguous_boundary_components += 1
                continue
            start = min(group)
            previous = -1
            current = start
            loop = []
            while True:
                loop.append(current)
                candidates = sorted(value for value in graph[current] if value != previous)
                if not candidates:
                    break
                following = candidates[0]
                previous, current = current, following
                if current == start:
                    break
                if len(loop) > len(group):
                    break
            if current == start and len(loop) == len(group):
                loop_values.append(loop)
                loop_region.append(region)
            else:
                ambiguous_boundary_components += 1
    kind = np.full(len(loop_values), 2, dtype=np.uint8)
    for region in sorted(set(loop_region)):
        indices = [index for index, value in enumerate(loop_region) if value == region]
        if not indices:
            continue
        area = [abs(_polygon_area(chart_uv[loop_values[index]])) for index in indices]
        outer_index = indices[int(np.argmax(area))]
        kind[outer_index] = 0
        outer_polygon = chart_uv[loop_values[outer_index]]
        for index in indices:
            if index == outer_index:
                continue
            centroid = np.mean(chart_uv[loop_values[index]], axis=0)
            if _point_in_polygon(centroid, outer_polygon):
                kind[index] = 1
    loop_offset = [0]
    loop_vertex: list[int] = []
    loop_area: list[float] = []
    loop_perimeter: list[float] = []
    loop_diameter: list[float] = []
    loop_thickness: list[float] = []
    loop_component: list[int] = []
    macro_eligible: list[int] = []
    for index, loop in enumerate(loop_values):
        nodes = np.asarray(loop, dtype=np.int32)
        points = chart_uv[nodes]
        area = abs(_polygon_area(points))
        perimeter = float(
            np.sum(np.linalg.norm(points - np.roll(points, -1, axis=0), axis=1))
        )
        if len(points) > 1:
            diameter = float(
                np.max(np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2))
            )
        else:
            diameter = 0.0
        median_thickness = float(np.median(thickness[nodes]))
        mean_boundary_edge = perimeter / max(len(nodes), 1)
        eligible = (
            kind[index] == 1
            and len(nodes) >= settings.minimum_hole_boundary_vertex_count
            and diameter
            >= settings.minimum_hole_diameter_boundary_edges
            * mean_boundary_edge
            and area
            >= settings.minimum_hole_area_boundary_edges_squared
            * mean_boundary_edge**2
        )
        loop_vertex.extend(int(value) for value in nodes)
        loop_offset.append(len(loop_vertex))
        loop_area.append(area)
        loop_perimeter.append(perimeter)
        loop_diameter.append(diameter)
        loop_thickness.append(median_thickness)
        loop_component.append(int(component[nodes[0]]))
        macro_eligible.append(int(eligible))
    arrays = {
        "loopOffset": np.asarray(loop_offset, dtype=np.int64),
        "loopVertexFrontierIndex": np.asarray(loop_vertex, dtype=np.int32),
        "loopTriangleRegion": np.asarray(loop_region, dtype=np.int32),
        "loopTopologyComponent": np.asarray(loop_component, dtype=np.int32),
        "loopKind": kind,
        "loopAreaChartVoxelsSquared": np.asarray(loop_area, dtype=np.float32),
        "loopPerimeterChartVoxels": np.asarray(loop_perimeter, dtype=np.float32),
        "loopDiameterChartVoxels": np.asarray(loop_diameter, dtype=np.float32),
        "loopMedianThicknessVoxels": np.asarray(loop_thickness, dtype=np.float32),
        "loopMeanBoundaryEdgeVoxels": np.asarray(loop_perimeter, dtype=np.float32)
        / np.maximum(
            np.diff(np.asarray(loop_offset, dtype=np.int64)), 1
        ),
        "loopMacroEligible": np.asarray(macro_eligible, dtype=np.uint8),
    }
    return arrays, {
        "triangleRegionCount": int(len(set(loop_region))),
        "closedBoundaryLoopCount": int(len(loop_values)),
        "outerBoundaryLoopCount": int(np.count_nonzero(kind == 0)),
        "interiorHoleLoopCount": int(np.count_nonzero(kind == 1)),
        "ambiguousLoopCount": int(np.count_nonzero(kind == 2)),
        "macroEligibleHoleCount": int(np.count_nonzero(macro_eligible)),
        "nonCycleBoundaryComponentCount": ambiguous_boundary_components,
    }


def _selected_surface_adjacency(
    surface: Mapping[str, np.ndarray],
) -> list[list[int]]:
    count = len(np.asarray(surface["midpointXYZ"]))
    adjacency: list[list[int]] = [[] for _ in range(count)]
    first = np.asarray(surface["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(surface["edgeSecondFrontierIndex"], dtype=np.int32)
    active = np.asarray(surface["edgeSelected"]) > 0
    for left, right in zip(first[active], second[active]):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    return adjacency


def _loop_vertices(
    loops: Mapping[str, np.ndarray], loop_index: int
) -> np.ndarray:
    offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    values = np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32)
    return values[int(offset[loop_index]) : int(offset[loop_index + 1])]


def _context_vertices(
    boundary: np.ndarray,
    adjacency: list[list[int]],
    component: np.ndarray,
    *,
    graph_hops: int,
) -> np.ndarray:
    component_id = int(component[int(boundary[0])])
    visited = {int(value) for value in boundary}
    frontier = set(visited)
    for _ in range(graph_hops):
        following: set[int] = set()
        for current in frontier:
            for neighbor in adjacency[current]:
                if component[neighbor] == component_id and neighbor not in visited:
                    following.add(neighbor)
        visited.update(following)
        frontier = following
        if not frontier:
            break
    return np.asarray(sorted(visited), dtype=np.int32)


def _surface_design(
    uv: np.ndarray, center_uv: np.ndarray, scale: float
) -> np.ndarray:
    local = (np.asarray(uv, dtype=np.float64) - center_uv) / scale
    u_value, v_value = local[:, 0], local[:, 1]
    return np.column_stack(
        (
            np.ones(len(local)),
            u_value,
            v_value,
            u_value * u_value,
            u_value * v_value,
            v_value * v_value,
        )
    )


def _fit_patch_models(
    chart_uv: np.ndarray,
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    boundary: np.ndarray,
    context: np.ndarray,
    *,
    quadratic_ridge: float,
) -> dict[str, np.ndarray | float]:
    center_uv = np.mean(chart_uv[boundary], axis=0, dtype=np.float64)
    radial = np.linalg.norm(chart_uv[context] - center_uv, axis=1)
    scale = max(float(np.percentile(radial, 75.0)), 1.0)
    design = _surface_design(chart_uv[context], center_uv, scale)
    target = np.asarray(center_xyz[context], dtype=np.float64)
    affine, *_ = np.linalg.lstsq(design[:, :3], target, rcond=None)
    affine_coefficients = np.zeros((6, 3), dtype=np.float64)
    affine_coefficients[:3] = affine
    ridge = np.diag((1.0e-10, 1.0e-10, 1.0e-10, 1.0, 1.0, 1.0))
    quadratic_coefficients = np.linalg.solve(
        design.T @ design + quadratic_ridge * ridge,
        design.T @ target,
    )
    coefficients = np.stack((affine_coefficients, quadratic_coefficients))
    prediction = np.einsum("nd,mdc->mnc", design, coefficients)
    residual = np.linalg.norm(prediction - target[None, :, :], axis=2)
    boundary_design = _surface_design(chart_uv[boundary], center_uv, scale)
    boundary_prediction = np.einsum(
        "nd,mdc->mnc", boundary_design, coefficients
    )
    boundary_residual = np.linalg.norm(
        boundary_prediction - center_xyz[boundary][None, :, :], axis=2
    )
    context_normal_residual: list[np.ndarray] = []
    for model_coefficients in coefficients:
        predicted_normal = _patch_normals(
            chart_uv[context],
            center_uv,
            scale,
            model_coefficients,
            signed_normal_xyz[boundary],
        )
        cosine = np.clip(
            np.abs(
                np.einsum(
                    "ij,ij->i", predicted_normal, signed_normal_xyz[context]
                )
            ),
            0.0,
            1.0,
        )
        context_normal_residual.append(np.degrees(np.arccos(cosine)))
    return {
        "centerUV": center_uv.astype(np.float32),
        "scale": scale,
        "coefficientsXYZ": coefficients.astype(np.float32),
        "contextResidualVoxels": np.median(residual, axis=1).astype(np.float32),
        "boundaryResidualVoxels": np.median(boundary_residual, axis=1).astype(
            np.float32
        ),
        "contextNormalResidualDegrees": np.asarray(
            [np.median(value) for value in context_normal_residual],
            dtype=np.float32,
        ),
    }


def _patch_points(
    uv: np.ndarray,
    center_uv: np.ndarray,
    scale: float,
    coefficients_xyz: np.ndarray,
) -> np.ndarray:
    return (
        _surface_design(uv, center_uv, scale) @ coefficients_xyz
    ).astype(np.float32)


def _patch_normals(
    uv: np.ndarray,
    center_uv: np.ndarray,
    scale: float,
    coefficients_xyz: np.ndarray,
    reference_normal_xyz: np.ndarray,
) -> np.ndarray:
    local = (np.asarray(uv, dtype=np.float64) - center_uv) / scale
    u_value, v_value = local[:, 0], local[:, 1]
    coefficients = np.asarray(coefficients_xyz, dtype=np.float64)
    derivative_u = (
        coefficients[1][None, :]
        + 2.0 * u_value[:, None] * coefficients[3][None, :]
        + v_value[:, None] * coefficients[4][None, :]
    ) / scale
    derivative_v = (
        coefficients[2][None, :]
        + u_value[:, None] * coefficients[4][None, :]
        + 2.0 * v_value[:, None] * coefficients[5][None, :]
    ) / scale
    normal = np.cross(derivative_u, derivative_v)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    reference = np.mean(reference_normal_xyz, axis=0)
    reference /= max(float(np.linalg.norm(reference)), 1.0e-12)
    normal[np.einsum("ij,j->i", normal, reference) < 0.0] *= -1.0
    return normal.astype(np.float32)


def _rasterize_polygon(
    polygon_uv: np.ndarray,
    *,
    requested_step: float,
    maximum_pixels: int,
) -> tuple[np.ndarray, float]:
    low = np.min(polygon_uv, axis=0)
    high = np.max(polygon_uv, axis=0)
    step = float(requested_step)
    extent = np.maximum(high - low, step)
    estimated = float(np.prod(np.ceil(extent / step)))
    if estimated > maximum_pixels:
        step *= math.sqrt(estimated / maximum_pixels)
    u_values = np.arange(low[0] + 0.5 * step, high[0], step)
    v_values = np.arange(low[1] + 0.5 * step, high[1], step)
    if not len(u_values) or not len(v_values):
        return np.mean(polygon_uv, axis=0, keepdims=True), step
    grid_u, grid_v = np.meshgrid(u_values, v_values)
    candidates = np.column_stack((grid_u.ravel(), grid_v.ravel()))
    retained = np.asarray(
        [_point_in_polygon(value, polygon_uv) for value in candidates],
        dtype=bool,
    )
    result = candidates[retained]
    if not len(result):
        result = np.mean(polygon_uv, axis=0, keepdims=True)
    return result.astype(np.float32), step


def _sample_volume_points(
    source: VolumeSource,
    volume: np.ndarray,
    world_xyz: np.ndarray,
) -> np.ndarray:
    points = np.asarray(world_xyz, dtype=np.float32)
    local = points.reshape((-1, 3)) - np.asarray(source.origin_xyz, dtype=np.float32)
    shape_xyz = np.asarray(source.shape_xyz, dtype=np.int64)
    low = np.floor(np.min(local, axis=0)).astype(np.int64) - 1
    high = np.ceil(np.max(local, axis=0)).astype(np.int64) + 2
    low = np.maximum(low, 0)
    high = np.minimum(high, shape_xyz)
    if np.any(high <= low):
        return np.full(points.shape[:-1], np.nan, dtype=np.float32)
    subvolume = np.asarray(
        volume[
            int(low[2]) : int(high[2]),
            int(low[1]) : int(high[1]),
            int(low[0]) : int(high[0]),
        ],
        dtype=np.float32,
    )
    sampled = _trilinear(
        subvolume,
        (local - low[None, :]).reshape(points.shape),
        outside=np.nan,
    )
    return np.asarray(sampled, dtype=np.float32)


def _sample_normal_profiles(
    source: VolumeSource,
    volume: np.ndarray,
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    thickness_voxels: np.ndarray,
    depth_fractions: np.ndarray,
    shift_thicknesses: np.ndarray,
) -> np.ndarray:
    center = np.asarray(center_xyz, dtype=np.float32)
    normal = np.asarray(normal_xyz, dtype=np.float32)
    thickness = np.asarray(thickness_voxels, dtype=np.float32).reshape((-1, 1))
    offsets = (
        shift_thicknesses[:, None, None]
        + depth_fractions[None, None, :]
    ) * thickness[None, :, :]
    points = (
        center[None, :, None, :]
        + normal[None, :, None, :] * offsets[:, :, :, None]
    )
    return _sample_volume_points(source, volume, points)


def _profile_score(
    profiles: np.ndarray,
    depth_fractions: np.ndarray,
    *,
    intensity_scale: float,
) -> np.ndarray:
    inside = np.abs(depth_fractions) <= 0.36
    outside = np.abs(depth_fractions) >= 0.64
    contrast = np.nanmean(profiles[..., inside], axis=-1) - np.nanmean(
        profiles[..., outside], axis=-1
    )
    return np.nanmedian(contrast, axis=-1) / max(float(intensity_scale), 1.0)


def _profile_correlation(first: np.ndarray, second: np.ndarray) -> float:
    valid = np.isfinite(first) & np.isfinite(second)
    if np.count_nonzero(valid) < 3:
        return float("nan")
    left = np.asarray(first[valid], dtype=np.float64)
    right = np.asarray(second[valid], dtype=np.float64)
    left -= np.mean(left)
    right -= np.mean(right)
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= 1.0e-12:
        return 0.0
    return float(np.dot(left, right) / denominator)


def score_surface_patch_holes(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    eligible = np.flatnonzero(np.asarray(loops["loopMacroEligible"]) > 0)
    if len(eligible):
        area = np.asarray(loops["loopAreaChartVoxelsSquared"])
        eligible = eligible[
            np.argsort(area[eligible], kind="stable")[::-1]
        ][: settings.maximum_scored_holes]
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    center_xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    signed_normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    component = np.asarray(surface["component"], dtype=np.int32)
    adjacency = _selected_surface_adjacency(surface)
    depth = np.asarray(settings.profile_depth_fractions, dtype=np.float32)
    shifts = np.asarray(settings.competing_shift_thicknesses, dtype=np.float32)
    zero_shift = int(np.flatnonzero(shifts == 0.0)[0])
    volume = source.memmap()

    context_offset = [0]
    context_vertex: list[int] = []
    patch_offset = [0]
    patch_uv_values: list[np.ndarray] = []
    patch_xyz_values: list[np.ndarray] = []
    patch_normal_values: list[np.ndarray] = []
    center_uv_values: list[np.ndarray] = []
    scale_values: list[float] = []
    coefficients_values: list[np.ndarray] = []
    context_residual_values: list[np.ndarray] = []
    boundary_residual_values: list[np.ndarray] = []
    normal_residual_values: list[np.ndarray] = []
    raster_step_values: list[float] = []
    model_shift_score_values: list[np.ndarray] = []
    model_shift_profile_values: list[np.ndarray] = []
    context_profile_values: list[np.ndarray] = []
    context_score_values: list[float] = []
    profile_correlation_values: list[np.ndarray] = []
    zero_shift_margin_values: list[np.ndarray] = []
    model_rank_score_values: list[np.ndarray] = []
    selected_model_values: list[int] = []
    intensity_scale_values: list[float] = []
    for loop_index in eligible:
        boundary = _loop_vertices(loops, int(loop_index))
        context = _context_vertices(
            boundary,
            adjacency,
            component,
            graph_hops=settings.context_graph_hops,
        )
        fitted = _fit_patch_models(
            chart_uv,
            center_xyz,
            signed_normal,
            boundary,
            context,
            quadratic_ridge=settings.quadratic_ridge,
        )
        center_uv = np.asarray(fitted["centerUV"], dtype=np.float32)
        scale = float(fitted["scale"])
        coefficients = np.asarray(fitted["coefficientsXYZ"], dtype=np.float32)
        patch_uv, raster_step = _rasterize_polygon(
            chart_uv[boundary],
            requested_step=settings.patch_pixel_step_voxels,
            maximum_pixels=settings.maximum_patch_pixels,
        )
        median_thickness = float(np.median(thickness[context]))
        context_profiles = _sample_normal_profiles(
            source,
            volume,
            center_xyz[context],
            signed_normal[context],
            thickness[context],
            depth,
            np.asarray((0.0,), dtype=np.float32),
        )[0]
        model_profiles: list[np.ndarray] = []
        model_points: list[np.ndarray] = []
        model_normals: list[np.ndarray] = []
        for model_coefficients in coefficients:
            points = _patch_points(
                patch_uv, center_uv, scale, model_coefficients
            )
            normals = _patch_normals(
                patch_uv,
                center_uv,
                scale,
                model_coefficients,
                signed_normal[boundary],
            )
            profiles = _sample_normal_profiles(
                source,
                volume,
                points,
                normals,
                np.full(len(points), median_thickness, dtype=np.float32),
                depth,
                shifts,
            )
            model_points.append(points)
            model_normals.append(normals)
            model_profiles.append(profiles)
        finite_intensity = np.concatenate(
            (
                context_profiles[np.isfinite(context_profiles)],
                *(
                    value[np.isfinite(value)]
                    for value in model_profiles
                ),
            )
        )
        intensity_scale = (
            float(np.percentile(finite_intensity, 90.0))
            - float(np.percentile(finite_intensity, 10.0))
            if len(finite_intensity)
            else 1.0
        )
        context_score = float(
            _profile_score(
                context_profiles[None, :, :],
                depth,
                intensity_scale=intensity_scale,
            )[0]
        )
        context_mean_profile = np.nanmedian(context_profiles, axis=0)
        shift_scores = np.stack(
            [
                _profile_score(
                    value, depth, intensity_scale=intensity_scale
                )
                for value in model_profiles
            ]
        )
        shift_profiles = np.stack(
            [np.nanmedian(value, axis=1) for value in model_profiles]
        )
        correlations = np.asarray(
            [
                _profile_correlation(
                    value[zero_shift], context_mean_profile
                )
                for value in shift_profiles
            ],
            dtype=np.float32,
        )
        competing = np.delete(shift_scores, zero_shift, axis=1)
        margin = shift_scores[:, zero_shift] - np.max(competing, axis=1)
        geometry_penalty = (
            np.asarray(fitted["boundaryResidualVoxels"], dtype=np.float32)
            / max(float(np.mean(np.diff(chart_uv[boundary], axis=0) ** 2) ** 0.5), 1.0)
        )
        rank_score = (
            shift_scores[:, zero_shift]
            + 0.35 * correlations
            + 0.35 * margin
            - 0.15 * geometry_penalty
        )
        selected_model = int(np.nanargmax(rank_score))

        context_vertex.extend(int(value) for value in context)
        context_offset.append(len(context_vertex))
        patch_uv_values.append(patch_uv)
        patch_xyz_values.append(model_points[selected_model])
        patch_normal_values.append(model_normals[selected_model])
        patch_offset.append(patch_offset[-1] + len(patch_uv))
        center_uv_values.append(center_uv)
        scale_values.append(scale)
        coefficients_values.append(coefficients)
        context_residual_values.append(
            np.asarray(fitted["contextResidualVoxels"], dtype=np.float32)
        )
        boundary_residual_values.append(
            np.asarray(fitted["boundaryResidualVoxels"], dtype=np.float32)
        )
        normal_residual_values.append(
            np.asarray(fitted["contextNormalResidualDegrees"], dtype=np.float32)
        )
        raster_step_values.append(raster_step)
        model_shift_score_values.append(shift_scores)
        model_shift_profile_values.append(shift_profiles)
        context_profile_values.append(context_mean_profile)
        context_score_values.append(context_score)
        profile_correlation_values.append(correlations)
        zero_shift_margin_values.append(margin)
        model_rank_score_values.append(rank_score)
        selected_model_values.append(selected_model)
        intensity_scale_values.append(intensity_scale)

    hole_count = len(eligible)
    model_count = 2
    profile_shape = (hole_count, model_count, len(shifts), len(depth))
    arrays = {
        "scoredLoopIndex": eligible.astype(np.int32),
        "contextOffset": np.asarray(context_offset, dtype=np.int64),
        "contextVertexFrontierIndex": np.asarray(context_vertex, dtype=np.int32),
        "patchOffset": np.asarray(patch_offset, dtype=np.int64),
        "patchUV": np.concatenate(patch_uv_values).astype(np.float32)
        if patch_uv_values
        else np.empty((0, 2), dtype=np.float32),
        "patchXYZ": np.concatenate(patch_xyz_values).astype(np.float32)
        if patch_xyz_values
        else np.empty((0, 3), dtype=np.float32),
        "patchNormalXYZ": np.concatenate(patch_normal_values).astype(np.float32)
        if patch_normal_values
        else np.empty((0, 3), dtype=np.float32),
        "fitCenterUV": np.asarray(center_uv_values, dtype=np.float32).reshape((-1, 2)),
        "fitScaleVoxels": np.asarray(scale_values, dtype=np.float32),
        "fitCoefficientsXYZ": np.asarray(coefficients_values, dtype=np.float32).reshape(
            (-1, model_count, 6, 3)
        ),
        "fitContextResidualVoxels": np.asarray(
            context_residual_values, dtype=np.float32
        ).reshape((-1, model_count)),
        "fitBoundaryResidualVoxels": np.asarray(
            boundary_residual_values, dtype=np.float32
        ).reshape((-1, model_count)),
        "fitContextNormalResidualDegrees": np.asarray(
            normal_residual_values, dtype=np.float32
        ).reshape((-1, model_count)),
        "rasterStepVoxels": np.asarray(raster_step_values, dtype=np.float32),
        "profileDepthFractions": depth,
        "competingShiftThicknesses": shifts,
        "modelShiftPhysicalScore": np.asarray(
            model_shift_score_values, dtype=np.float32
        ).reshape((hole_count, model_count, len(shifts))),
        "modelShiftMedianProfile": np.asarray(
            model_shift_profile_values, dtype=np.float32
        ).reshape(profile_shape),
        "contextMedianProfile": np.asarray(
            context_profile_values, dtype=np.float32
        ).reshape((hole_count, len(depth))),
        "contextPhysicalScore": np.asarray(context_score_values, dtype=np.float32),
        "zeroShiftContextProfileCorrelation": np.asarray(
            profile_correlation_values, dtype=np.float32
        ).reshape((hole_count, model_count)),
        "zeroShiftCompetingMargin": np.asarray(
            zero_shift_margin_values, dtype=np.float32
        ).reshape((hole_count, model_count)),
        "modelRankScore": np.asarray(
            model_rank_score_values, dtype=np.float32
        ).reshape((hole_count, model_count)),
        "selectedModel": np.asarray(selected_model_values, dtype=np.uint8),
        "localIntensityScale": np.asarray(intensity_scale_values, dtype=np.float32),
    }
    return arrays, {
        "scoredHoleCount": hole_count,
        "decisionUnit": "closed multi-ribbon surface boundary loop",
        "surfaceModels": ["affine", "quadratic"],
        "competingHypotheses": "same fitted patch translated along its normal by local ribbon thickness",
        "rawCtEvidence": "whole-patch air-material-air normal profiles compared with the surrounding mesh context",
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def _geometric_patch_candidates(
    loop_row: int,
    loop_index: int,
    loops: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> dict[str, np.ndarray]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    center = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    normal = np.asarray(ribbon["normalXYZ"], dtype=np.float32)[frontier]
    thickness = np.asarray(ribbon["thicknessVoxels"], dtype=np.float32)[frontier]
    selected = np.asarray(configuration["selected"]) > 0
    patch_offset = np.asarray(scored["patchOffset"], dtype=np.int64)
    patch = np.asarray(scored["patchXYZ"], dtype=np.float32)[
        int(patch_offset[loop_row]) : int(patch_offset[loop_row + 1])
    ]
    patch_normal = np.asarray(scored["patchNormalXYZ"], dtype=np.float32)[
        int(patch_offset[loop_row]) : int(patch_offset[loop_row + 1])
    ]
    median_thickness = float(loops["loopMedianThicknessVoxels"][loop_index])
    raster_step = float(scored["rasterStepVoxels"][loop_row])
    padding = max(4.0, 0.4 * median_thickness)
    in_box = (~selected) & np.all(
        (center >= np.min(patch, axis=0) - padding)
        & (center <= np.max(patch, axis=0) + padding),
        axis=1,
    )
    candidate = np.flatnonzero(in_box)
    if not len(candidate):
        empty_float = np.empty(0, dtype=np.float32)
        return {
            "frontierIndex": np.empty(0, dtype=np.int32),
            "nearestPatchPixel": np.empty(0, dtype=np.int32),
            "heightResidualVoxels": empty_float,
            "tangentResidualVoxels": empty_float,
            "normalResidualDegrees": empty_float,
            "thicknessRatio": empty_float,
            "surfaceAlignment": empty_float,
        }
    delta = center[candidate, None, :] - patch[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
    nearest = np.argmin(distance_squared, axis=1)
    residual = delta[np.arange(len(candidate)), nearest]
    nearest_normal = patch_normal[nearest]
    height = np.abs(np.einsum("ij,ij->i", residual, nearest_normal))
    tangent = np.sqrt(
        np.maximum(np.einsum("ij,ij->i", residual, residual) - height**2, 0.0)
    )
    cosine = np.clip(
        np.abs(np.einsum("ij,ij->i", normal[candidate], nearest_normal)),
        0.0,
        1.0,
    )
    normal_degrees = np.degrees(np.arccos(cosine))
    thickness_ratio = np.maximum(
        thickness[candidate] / median_thickness,
        median_thickness / thickness[candidate],
    )
    height_scale = max(
        2.5,
        settings.maximum_candidate_height_thicknesses * median_thickness,
    )
    tangent_scale = (
        settings.maximum_candidate_tangent_raster_steps * raster_step
    )
    retained = (
        (height <= height_scale)
        & (tangent <= tangent_scale)
        & (normal_degrees <= settings.maximum_candidate_normal_degrees)
        & (thickness_ratio <= settings.maximum_candidate_thickness_ratio)
    )
    alignment = (
        np.exp(-0.5 * (height / max(height_scale, 1.0e-6)) ** 2)
        * np.exp(-0.5 * (tangent / max(tangent_scale, 1.0e-6)) ** 2)
        * cosine
        * np.exp(-np.abs(np.log(thickness_ratio)))
    )
    return {
        "frontierIndex": candidate[retained].astype(np.int32),
        "nearestPatchPixel": nearest[retained].astype(np.int32),
        "heightResidualVoxels": height[retained].astype(np.float32),
        "tangentResidualVoxels": tangent[retained].astype(np.float32),
        "normalResidualDegrees": normal_degrees[retained].astype(np.float32),
        "thicknessRatio": thickness_ratio[retained].astype(np.float32),
        "surfaceAlignment": alignment[retained].astype(np.float32),
    }


def _objective_for_mask(
    mask: int,
    node_weight: np.ndarray,
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    pair_weight: np.ndarray,
) -> float:
    selected = np.asarray(
        [bool(mask & (1 << index)) for index in range(len(node_weight))]
    )
    value = float(np.sum(node_weight[selected]))
    if len(pair_first):
        value += float(
            np.sum(pair_weight[selected[pair_first] & selected[pair_second]])
        )
    return value


def _beam_set_packing(
    node_weight: np.ndarray,
    conflict_mask: list[int],
    pair_first: np.ndarray,
    pair_second: np.ndarray,
    pair_weight: np.ndarray,
    baseline_mask: int,
    *,
    beam_width: int,
) -> tuple[list[tuple[float, int]], dict[str, Any]]:
    pair_neighbor: list[list[tuple[int, float]]] = [
        [] for _ in range(len(node_weight))
    ]
    for first, second, weight in zip(pair_first, pair_second, pair_weight):
        pair_neighbor[int(first)].append((int(second), float(weight)))
        pair_neighbor[int(second)].append((int(first), float(weight)))
    states: list[tuple[float, int]] = [(0.0, 0)]
    maximum_state_count = 1
    baseline_prefix_mask = 0
    for option in range(len(node_weight)):
        following = list(states)
        bit = 1 << option
        for score, mask in states:
            if mask & conflict_mask[option]:
                continue
            pair_gain = sum(
                weight
                for neighbor, weight in pair_neighbor[option]
                if mask & (1 << neighbor)
            )
            following.append((score + float(node_weight[option]) + pair_gain, mask | bit))
        if baseline_mask & bit:
            baseline_prefix_mask |= bit
        baseline_prefix_score = _objective_for_mask(
            baseline_prefix_mask,
            node_weight[: option + 1],
            pair_first[(pair_first <= option) & (pair_second <= option)],
            pair_second[(pair_first <= option) & (pair_second <= option)],
            pair_weight[(pair_first <= option) & (pair_second <= option)],
        )
        following.sort(key=lambda value: (value[0], value[1]), reverse=True)
        if len(following) > beam_width:
            following = following[:beam_width]
        if not any(mask == baseline_prefix_mask for _, mask in following):
            following.append((baseline_prefix_score, baseline_prefix_mask))
        states = following
        maximum_state_count = max(maximum_state_count, len(states))
    states.sort(key=lambda value: (value[0], value[1]), reverse=True)
    return states, {
        "algorithm": "deterministic joint set-packing beam with exact local factor scores",
        "beamWidth": beam_width,
        "maximumRetainedStateCount": maximum_state_count,
        "finalStateCount": len(states),
        "baselineStatePreserved": any(mask == baseline_mask for _, mask in states),
    }


def solve_patch_hole_reconfigurations(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(configuration["selected"]) > 0
    source_interface = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target_interface = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    interface_owner = np.full(
        len(np.asarray(ribbon["interfaceCandidateDegree"])), -1, dtype=np.int32
    )
    selected_index = np.flatnonzero(selected)
    interface_owner[source_interface[selected_index]] = selected_index
    interface_owner[target_interface[selected_index]] = selected_index
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    node_unary = np.asarray(configuration["nodeUnaryScore"], dtype=np.float32)
    loop_index_values = np.asarray(scored["scoredLoopIndex"], dtype=np.int32)
    patch_offset = np.asarray(scored["patchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(scored["patchXYZ"], dtype=np.float32)

    candidate_offset = [0]
    candidate_values: list[int] = []
    candidate_nearest: list[int] = []
    candidate_height: list[float] = []
    candidate_tangent: list[float] = []
    candidate_normal: list[float] = []
    candidate_thickness_ratio: list[float] = []
    candidate_alignment: list[float] = []
    option_offset = [0]
    option_values: list[int] = []
    option_was_selected: list[int] = []
    option_is_candidate: list[int] = []
    option_weight_values: list[float] = []
    proposal_added_offset = [0]
    proposal_added: list[int] = []
    proposal_removed_offset = [0]
    proposal_removed: list[int] = []
    baseline_objective: list[float] = []
    proposal_objective: list[float] = []
    objective_delta: list[float] = []
    proposal_coverage: list[float] = []
    retained_boundary_fraction: list[float] = []
    boundary_anchor_count: list[int] = []
    option_conflict_counts: list[int] = []
    beam_records: list[dict[str, Any]] = []
    for loop_row, loop_index in enumerate(loop_index_values):
        geometric = _geometric_patch_candidates(
            loop_row,
            int(loop_index),
            loops,
            scored,
            ribbon,
            topology,
            configuration,
            settings=settings,
        )
        candidate = np.asarray(geometric["frontierIndex"], dtype=np.int32)
        candidate_set = set(int(value) for value in candidate)
        incumbent = set(
            int(value)
            for value in np.concatenate(
                (
                    interface_owner[source_interface[candidate]],
                    interface_owner[target_interface[candidate]],
                )
            )
            if value >= 0
        )
        candidate_lookup = {int(value): index for index, value in enumerate(candidate)}
        for left, right in zip(crossing_first, crossing_second):
            left_value, right_value = int(left), int(right)
            if left_value in candidate_set and selected[right_value]:
                incumbent.add(right_value)
            if right_value in candidate_set and selected[left_value]:
                incumbent.add(left_value)
        options = np.asarray(
            sorted(incumbent) + sorted(candidate_set), dtype=np.int32
        )
        local_index = {int(value): index for index, value in enumerate(options)}
        local_set = set(local_index)
        fixed_selected = selected.copy()
        if incumbent:
            fixed_selected[np.asarray(sorted(incumbent), dtype=np.int32)] = False
        node_weight = node_unary[options].astype(np.float64)
        fixed_support = np.zeros(len(options), dtype=np.float64)
        pair_support: dict[tuple[int, int], float] = defaultdict(float)
        for left, right, value in zip(edge_first, edge_second, edge_score):
            left_value, right_value = int(left), int(right)
            left_local = local_index.get(left_value)
            right_local = local_index.get(right_value)
            if left_local is not None and right_local is not None:
                pair = (min(left_local, right_local), max(left_local, right_local))
                pair_support[pair] += continuity_weight * float(value)
            elif left_local is not None and fixed_selected[right_value]:
                fixed_support[left_local] += continuity_weight * float(value)
            elif right_local is not None and fixed_selected[left_value]:
                fixed_support[right_local] += continuity_weight * float(value)
        node_weight += fixed_support
        for option_value, option_index in local_index.items():
            candidate_index = candidate_lookup.get(option_value)
            if candidate_index is not None:
                node_weight[option_index] += (
                    settings.surface_alignment_weight
                    * float(geometric["surfaceAlignment"][candidate_index])
                )
        conflict_mask = [0 for _ in range(len(options))]
        interface_groups: dict[int, list[int]] = defaultdict(list)
        for option_index, option_value in enumerate(options):
            interface_groups[int(source_interface[option_value])].append(option_index)
            interface_groups[int(target_interface[option_value])].append(option_index)
        for group in interface_groups.values():
            for first_index, left in enumerate(group):
                for right in group[first_index + 1 :]:
                    conflict_mask[left] |= 1 << right
                    conflict_mask[right] |= 1 << left
        for left, right in zip(crossing_first, crossing_second):
            left_local = local_index.get(int(left))
            right_local = local_index.get(int(right))
            if left_local is not None and right_local is not None:
                conflict_mask[left_local] |= 1 << right_local
                conflict_mask[right_local] |= 1 << left_local
        pair_items = sorted(pair_support.items())
        pair_first = np.asarray([key[0] for key, _ in pair_items], dtype=np.int32)
        pair_second = np.asarray([key[1] for key, _ in pair_items], dtype=np.int32)
        pair_weight = np.asarray([value for _, value in pair_items], dtype=np.float32)
        baseline_mask = 0
        for option_index, option_value in enumerate(options):
            if selected[option_value]:
                baseline_mask |= 1 << option_index
        states, beam_stats = _beam_set_packing(
            node_weight,
            conflict_mask,
            pair_first,
            pair_second,
            pair_weight,
            baseline_mask,
            beam_width=settings.configuration_beam_width,
        )
        proposal_state = next(
            (
                (score, mask)
                for score, mask in states
                if any(
                    mask & (1 << local_index[int(value)]) for value in candidate
                )
            ),
            (_objective_for_mask(
                baseline_mask, node_weight, pair_first, pair_second, pair_weight
            ), baseline_mask),
        )
        baseline_score = _objective_for_mask(
            baseline_mask, node_weight, pair_first, pair_second, pair_weight
        )
        proposed_mask = proposal_state[1]
        proposed_nodes = {
            int(options[index])
            for index in range(len(options))
            if proposed_mask & (1 << index)
        }
        added = sorted(proposed_nodes & candidate_set)
        removed = sorted(incumbent - proposed_nodes)
        boundary = _loop_vertices(loops, int(loop_index))
        retained_boundary = sum(int(value) in proposed_nodes for value in boundary)
        boundary_set = set(int(value) for value in boundary)
        anchors: set[int] = set()
        added_set = set(added)
        for left, right in zip(edge_first, edge_second):
            left_value, right_value = int(left), int(right)
            if left_value in added_set and right_value in boundary_set:
                anchors.add(right_value)
            if right_value in added_set and left_value in boundary_set:
                anchors.add(left_value)
        start, stop = int(patch_offset[loop_row]), int(patch_offset[loop_row + 1])
        patch = patch_xyz[start:stop]
        if added:
            added_candidate_index = np.asarray(
                [candidate_lookup[value] for value in added], dtype=np.int32
            )
            projected = patch[
                np.asarray(geometric["nearestPatchPixel"], dtype=np.int32)[
                    added_candidate_index
                ]
            ]
            coverage_radius = float(loops["loopMeanBoundaryEdgeVoxels"][loop_index])
            covered = np.any(
                np.linalg.norm(patch[:, None, :] - projected[None, :, :], axis=2)
                <= coverage_radius,
                axis=1,
            )
            coverage = float(np.mean(covered))
        else:
            coverage = 0.0

        candidate_values.extend(int(value) for value in candidate)
        candidate_nearest.extend(int(value) for value in geometric["nearestPatchPixel"])
        candidate_height.extend(float(value) for value in geometric["heightResidualVoxels"])
        candidate_tangent.extend(float(value) for value in geometric["tangentResidualVoxels"])
        candidate_normal.extend(float(value) for value in geometric["normalResidualDegrees"])
        candidate_thickness_ratio.extend(float(value) for value in geometric["thicknessRatio"])
        candidate_alignment.extend(float(value) for value in geometric["surfaceAlignment"])
        candidate_offset.append(len(candidate_values))
        option_values.extend(int(value) for value in options)
        option_was_selected.extend(int(selected[value]) for value in options)
        option_is_candidate.extend(int(value in candidate_set) for value in options)
        option_weight_values.extend(float(value) for value in node_weight)
        option_offset.append(len(option_values))
        proposal_added.extend(added)
        proposal_added_offset.append(len(proposal_added))
        proposal_removed.extend(removed)
        proposal_removed_offset.append(len(proposal_removed))
        baseline_objective.append(baseline_score)
        proposal_objective.append(float(proposal_state[0]))
        objective_delta.append(float(proposal_state[0] - baseline_score))
        proposal_coverage.append(coverage)
        retained_boundary_fraction.append(retained_boundary / max(len(boundary), 1))
        boundary_anchor_count.append(len(anchors))
        option_conflict_counts.append(sum(value.bit_count() for value in conflict_mask) // 2)
        beam_records.append(beam_stats)
    arrays = {
        "reconfigurationLoopIndex": loop_index_values,
        "patchCandidateOffset": np.asarray(candidate_offset, dtype=np.int64),
        "patchCandidateFrontierIndex": np.asarray(candidate_values, dtype=np.int32),
        "patchCandidateNearestPixel": np.asarray(candidate_nearest, dtype=np.int32),
        "patchCandidateHeightResidualVoxels": np.asarray(candidate_height, dtype=np.float32),
        "patchCandidateTangentResidualVoxels": np.asarray(candidate_tangent, dtype=np.float32),
        "patchCandidateNormalResidualDegrees": np.asarray(candidate_normal, dtype=np.float32),
        "patchCandidateThicknessRatio": np.asarray(candidate_thickness_ratio, dtype=np.float32),
        "patchCandidateSurfaceAlignment": np.asarray(candidate_alignment, dtype=np.float32),
        "reconfigurationOptionOffset": np.asarray(option_offset, dtype=np.int64),
        "reconfigurationOptionFrontierIndex": np.asarray(option_values, dtype=np.int32),
        "reconfigurationOptionWasSelected": np.asarray(option_was_selected, dtype=np.uint8),
        "reconfigurationOptionIsPatchCandidate": np.asarray(option_is_candidate, dtype=np.uint8),
        "reconfigurationOptionNodeWeight": np.asarray(option_weight_values, dtype=np.float32),
        "proposalAddedOffset": np.asarray(proposal_added_offset, dtype=np.int64),
        "proposalAddedFrontierIndex": np.asarray(proposal_added, dtype=np.int32),
        "proposalRemovedOffset": np.asarray(proposal_removed_offset, dtype=np.int64),
        "proposalRemovedFrontierIndex": np.asarray(proposal_removed, dtype=np.int32),
        "baselineLocalObjective": np.asarray(baseline_objective, dtype=np.float32),
        "proposalLocalObjective": np.asarray(proposal_objective, dtype=np.float32),
        "proposalObjectiveDelta": np.asarray(objective_delta, dtype=np.float32),
        "proposalPatchCoverage": np.asarray(proposal_coverage, dtype=np.float32),
        "proposalRetainedBoundaryFraction": np.asarray(retained_boundary_fraction, dtype=np.float32),
        "proposalBoundaryAnchorCount": np.asarray(boundary_anchor_count, dtype=np.int32),
        "reconfigurationHardConflictCount": np.asarray(option_conflict_counts, dtype=np.int32),
    }
    return arrays, {
        "holeCount": len(loop_index_values),
        "geometricCandidateCount": len(candidate_values),
        "proposalAddedRibbonCount": len(proposal_added),
        "proposalRemovedRibbonCount": len(proposal_removed),
        "positiveObjectiveProposalCount": int(np.count_nonzero(np.asarray(objective_delta) > 0.0)),
        "beam": beam_records,
        "factorGraph": "ribbons are variables; shared interfaces and exact crossings are hard exclusions; strict continuity and fixed-neighbor support are pair factors",
        "jointMove": "every proposal simultaneously removes incumbent interface pairings and selects one compatible patch-covering matching",
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def replay_patch_hole_reconfigurations(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    reconfiguration: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    selected = np.asarray(configuration["selected"]).astype(bool).copy()
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    added_offset = np.asarray(reconfiguration["proposalAddedOffset"], dtype=np.int64)
    added_values = np.asarray(
        reconfiguration["proposalAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        reconfiguration["proposalRemovedOffset"], dtype=np.int64
    )
    removed_values = np.asarray(
        reconfiguration["proposalRemovedFrontierIndex"], dtype=np.int32
    )
    delta = np.asarray(reconfiguration["proposalObjectiveDelta"], dtype=np.float32)
    applied = np.zeros(len(delta), dtype=np.uint8)

    def conflict_counts(value: np.ndarray) -> tuple[int, int]:
        node = np.flatnonzero(value)
        interfaces = np.concatenate((source[node], target[node]))
        interface_conflicts = len(interfaces) - len(np.unique(interfaces))
        crossing_conflicts = int(
            np.count_nonzero(value[crossing_first] & value[crossing_second])
        )
        return interface_conflicts, crossing_conflicts

    # Hole proposals are independently solved local factor graphs.  Compose
    # them in descending evidence gain, accepting only proposals whose complete
    # alternating re-pairing remains globally feasible with earlier moves.
    for row in np.argsort(-delta, kind="stable"):
        if delta[row] <= 0.0:
            continue
        added = added_values[int(added_offset[row]) : int(added_offset[row + 1])]
        removed = removed_values[
            int(removed_offset[row]) : int(removed_offset[row + 1])
        ]
        trial = selected.copy()
        trial[removed] = False
        trial[added] = True
        if conflict_counts(trial) != (0, 0):
            continue
        selected = trial
        applied[row] = 1
    interface_conflicts, crossing_conflicts = conflict_counts(selected)
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    component, component_size = _component_labels(selected, first, second)
    replay_configuration = dict(configuration)
    replay_configuration["selected"] = selected.astype(np.uint8)
    replay_configuration["component"] = component
    replay_surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        topology,
        replay_configuration,
        settings=settings,
    )
    replay_loops, loop_stats = extract_surface_boundary_loops(
        replay_surface, settings=settings
    )
    old_component = np.asarray(configuration["component"], dtype=np.int32)
    cross_component_fusions = 0
    maximum_old_components_per_replay_component = 0
    for component_id in np.unique(component[selected]):
        nodes = np.flatnonzero(component == component_id)
        inherited = np.unique(old_component[nodes][old_component[nodes] >= 0])
        maximum_old_components_per_replay_component = max(
            maximum_old_components_per_replay_component, len(inherited)
        )
        cross_component_fusions += int(len(inherited) > 1)

    midpoint = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    original_loop_index = np.asarray(
        reconfiguration["reconfigurationLoopIndex"], dtype=np.int32
    )
    original_center = np.asarray(
        [
            np.mean(midpoint[_loop_vertices(loops, int(loop_index))], axis=0)
            for loop_index in original_loop_index
        ],
        dtype=np.float32,
    ).reshape((-1, 3))
    replay_midpoint = np.asarray(replay_surface["midpointXYZ"], dtype=np.float32)
    replay_eligible = np.flatnonzero(
        np.asarray(replay_loops["loopMacroEligible"]) > 0
    )
    replay_center = np.asarray(
        [
            np.mean(
                replay_midpoint[_loop_vertices(replay_loops, int(loop_index))],
                axis=0,
            )
            for loop_index in replay_eligible
        ],
        dtype=np.float32,
    ).reshape((-1, 3))
    nearest_distance = np.full(len(original_center), np.inf, dtype=np.float32)
    nearest_loop = np.full(len(original_center), -1, dtype=np.int32)
    still_open = np.zeros(len(original_center), dtype=np.uint8)
    if len(replay_center):
        distances = np.linalg.norm(
            original_center[:, None, :] - replay_center[None, :, :], axis=2
        )
        nearest = np.argmin(distances, axis=1)
        nearest_distance = distances[np.arange(len(original_center)), nearest].astype(
            np.float32
        )
        nearest_loop = replay_eligible[nearest].astype(np.int32)
        match_radius = 1.5 * np.asarray(
            loops["loopMeanBoundaryEdgeVoxels"], dtype=np.float32
        )[original_loop_index]
        still_open = (nearest_distance <= match_radius).astype(np.uint8)
    arrays = {
        "replayProposalApplied": applied,
        "replaySelected": selected.astype(np.uint8),
        "replayComponent": component,
        "replayComponentSize": component_size.astype(np.int32),
        "replayChartUV": np.asarray(replay_surface["chartUV"], dtype=np.float32),
        "replayTriangleFrontierIndex": np.asarray(
            replay_surface["triangleFrontierIndex"], dtype=np.int32
        ),
        "replayTriangleAreaVoxelsSquared": np.asarray(
            replay_surface["triangleAreaVoxelsSquared"], dtype=np.float32
        ),
        "replayLoopOffset": np.asarray(replay_loops["loopOffset"], dtype=np.int64),
        "replayLoopVertexFrontierIndex": np.asarray(
            replay_loops["loopVertexFrontierIndex"], dtype=np.int32
        ),
        "replayLoopKind": np.asarray(replay_loops["loopKind"], dtype=np.uint8),
        "replayLoopMacroEligible": np.asarray(
            replay_loops["loopMacroEligible"], dtype=np.uint8
        ),
        "replayOriginalHoleNearestMacroLoop": nearest_loop,
        "replayOriginalHoleNearestMacroDistanceVoxels": nearest_distance,
        "replayOriginalHoleStillOpen": still_open,
    }
    return arrays, {
        "candidateProposalCount": len(applied),
        "appliedPositiveNonconflictingProposalCount": int(np.count_nonzero(applied)),
        "selectedRibbonCountBefore": int(
            np.count_nonzero(configuration["selected"])
        ),
        "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
        "interfaceConflictCount": interface_conflicts,
        "crossingConflictCount": crossing_conflicts,
        "componentCountAfter": len(component_size),
        "largestComponentRibbonCountsAfter": [
            int(value) for value in component_size[:32]
        ],
        "crossPriorComponentFusionCount": cross_component_fusions,
        "maximumPriorComponentsPerReplayComponent": (
            maximum_old_components_per_replay_component
        ),
        "surface": surface_stats,
        "loops": loop_stats,
        "originalMacroHoleStillOpenCount": int(np.count_nonzero(still_open)),
        "selectionMutated": False,
        "replayMeaning": "counterfactual in-memory rebuild only; the source configuration artifact is unchanged",
        "identityLabelsUsed": False,
    }


def write_patch_hole_montage(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    maximum_holes: int,
) -> Path:
    loop_index = np.asarray(scored["scoredLoopIndex"], dtype=np.int32)
    count = min(len(loop_index), maximum_holes)
    panel_height = 250
    width = 920
    canvas = np.full(
        (max(count, 1) * panel_height, width, 3),
        (8, 12, 17),
        dtype=np.uint8,
    )
    if not count:
        _draw_text(canvas, 16, 16, "NO CLOSED PATCH HOLES", (220, 225, 232), scale=2)
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    first = np.asarray(surface["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(surface["edgeSecondFrontierIndex"], dtype=np.int32)
    active_edge = np.asarray(surface["edgeSelected"]) > 0
    context_offset = np.asarray(scored["contextOffset"], dtype=np.int64)
    context_vertex = np.asarray(
        scored["contextVertexFrontierIndex"], dtype=np.int32
    )
    patch_offset = np.asarray(scored["patchOffset"], dtype=np.int64)
    patch_uv = np.asarray(scored["patchUV"], dtype=np.float32)
    depth = np.asarray(scored["profileDepthFractions"], dtype=np.float32)
    shifts = np.asarray(scored["competingShiftThicknesses"], dtype=np.float32)
    zero_shift = int(np.flatnonzero(shifts == 0.0)[0])
    profiles = np.asarray(scored["modelShiftMedianProfile"], dtype=np.float32)
    context_profiles = np.asarray(scored["contextMedianProfile"], dtype=np.float32)
    scores = np.asarray(scored["modelShiftPhysicalScore"], dtype=np.float32)
    selected_model = np.asarray(scored["selectedModel"], dtype=np.int32)
    correlations = np.asarray(
        scored["zeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    margins = np.asarray(scored["zeroShiftCompetingMargin"], dtype=np.float32)
    for row in range(count):
        y_base = row * panel_height
        current_loop = int(loop_index[row])
        boundary = _loop_vertices(loops, current_loop)
        context = context_vertex[
            int(context_offset[row]) : int(context_offset[row + 1])
        ]
        current_patch = patch_uv[
            int(patch_offset[row]) : int(patch_offset[row + 1])
        ]
        selected = int(selected_model[row])
        _draw_text(
            canvas,
            12,
            y_base + 10,
            f"C {int(loops['loopTopologyComponent'][current_loop])} N {len(boundary)}",
            (225, 230, 238),
            scale=2,
        )
        # Intrinsic chart: the yellow closed loop is the decision boundary;
        # green samples are the jointly fitted missing patch.
        chart_left, chart_top, chart_width, chart_height = 12, y_base + 40, 286, 194
        all_uv = np.vstack((chart_uv[context], current_patch))
        low = np.min(all_uv, axis=0)
        high = np.max(all_uv, axis=0)
        span = np.maximum(high - low, 1.0)
        scale = min((chart_width - 16) / span[0], (chart_height - 16) / span[1])

        def chart_point(value: np.ndarray) -> np.ndarray:
            return np.asarray(
                (
                    chart_left + 8 + (value[0] - low[0]) * scale,
                    chart_top + chart_height - 8 - (value[1] - low[1]) * scale,
                ),
                dtype=np.float32,
            )

        context_mask = np.zeros(len(chart_uv), dtype=bool)
        context_mask[context] = True
        for left, right in zip(first[active_edge], second[active_edge]):
            if context_mask[left] and context_mask[right]:
                _draw_line(
                    canvas,
                    chart_point(chart_uv[left]),
                    chart_point(chart_uv[right]),
                    (48, 68, 82),
                )
        for value in current_patch:
            point = np.rint(chart_point(value)).astype(int)
            canvas[
                max(point[1] - 1, 0) : point[1] + 2,
                max(point[0] - 1, 0) : point[0] + 2,
            ] = (45, 190, 126)
        for index, left in enumerate(boundary):
            right = boundary[(index + 1) % len(boundary)]
            _draw_line(
                canvas,
                chart_point(chart_uv[left]),
                chart_point(chart_uv[right]),
                (255, 205, 67),
            )

        # Whole-loop CT normal profile, overlaid with the surrounding context.
        profile_left, profile_top = 330, y_base + 52
        profile_width, profile_height = 250, 166
        shown = np.vstack(
            (
                context_profiles[row][None, :],
                profiles[row, 0, zero_shift][None, :],
                profiles[row, 1, zero_shift][None, :],
            )
        )
        low_value = float(np.nanmin(shown))
        high_value = float(np.nanmax(shown))
        value_span = max(high_value - low_value, 1.0)
        colors = ((225, 230, 238), (75, 184, 246), (229, 98, 197))
        for values, color in zip(shown, colors):
            points = []
            for depth_value, intensity in zip(depth, values):
                points.append(
                    np.asarray(
                        (
                            profile_left
                            + (depth_value - depth[0])
                            / max(float(depth[-1] - depth[0]), 1.0e-6)
                            * profile_width,
                            profile_top
                            + profile_height
                            - (intensity - low_value) / value_span * profile_height,
                        ),
                        dtype=np.float32,
                    )
                )
            for left, right in zip(points, points[1:]):
                _draw_line(canvas, left, right, color)
        _draw_text(canvas, profile_left, y_base + 30, "CT NORMAL PROFILE", (180, 190, 202))

        # Competing layer translations: the center bar is the predicted sheet.
        score_left, score_top = 620, y_base + 52
        score_width, score_height = 270, 166
        score_low = min(float(np.nanmin(scores[row])), -0.05)
        score_high = max(float(np.nanmax(scores[row])), 0.05)
        score_span = max(score_high - score_low, 1.0e-6)
        for model, color in enumerate(((75, 184, 246), (229, 98, 197))):
            points = []
            for shift, value in zip(shifts, scores[row, model]):
                points.append(
                    np.asarray(
                        (
                            score_left
                            + (shift - shifts[0])
                            / max(float(shifts[-1] - shifts[0]), 1.0e-6)
                            * score_width,
                            score_top
                            + score_height
                            - (value - score_low) / score_span * score_height,
                        ),
                        dtype=np.float32,
                    )
                )
            for left, right in zip(points, points[1:]):
                _draw_line(canvas, left, right, color)
            for point in points:
                x_value, y_value = np.rint(point).astype(int)
                canvas[y_value - 2 : y_value + 3, x_value - 2 : x_value + 3] = color
        _draw_text(canvas, score_left, y_base + 30, "SHIFT SCORE", (180, 190, 202))
        _draw_text(
            canvas,
            score_left,
            y_base + 224,
            f"R {correlations[row, selected]:.2f} D {margins[row, selected]:.2f}",
            (225, 230, 238),
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rgb_png(canvas))
    return output


def run_physical_ribbon_patch_holes(
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonPatchHoleSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonPatchHoleSettings()
    (
        configuration_path,
        configuration_manifest,
        configuration,
        continuity_path,
        continuity_manifest,
        topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_root)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_HOLES_VERSION,
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "topologyContinuity": {
            "manifestPath": str(continuity_path),
            "manifestSha256": sha256_file(continuity_path),
            "dataSha256": continuity_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_PATCH_HOLES_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_PATCH_HOLES_STEM}.npz"
    preview_path = output / "physical-ribbon-patch-holes.png"
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
    surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon, topology, configuration, settings=resolved
    )
    surfaced = time.monotonic()
    loops, loop_stats = extract_surface_boundary_loops(surface, settings=resolved)
    looped = time.monotonic()
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    scored, scoring_stats = score_surface_patch_holes(
        surface, loops, source, settings=resolved
    )
    scored_at = time.monotonic()
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    reconfiguration, reconfiguration_stats = solve_patch_hole_reconfigurations(
        surface,
        loops,
        scored,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        settings=resolved,
    )
    reconfigured_at = time.monotonic()
    replay, replay_stats = replay_patch_hole_reconfigurations(
        surface,
        loops,
        reconfiguration,
        ribbon,
        topology,
        configuration,
        settings=resolved,
    )
    replayed_at = time.monotonic()
    arrays = {**surface, **loops, **scored, **reconfiguration, **replay}
    _write_npz(data_path, arrays)
    write_patch_hole_montage(
        surface,
        loops,
        scored,
        preview_path,
        maximum_holes=resolved.maximum_preview_holes,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_HOLES_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": configuration_manifest.get("geometry", {}),
        "surface": surface_stats,
        "loops": loop_stats,
        "scoring": scoring_stats,
        "reconfiguration": reconfiguration_stats,
        "counterfactualReplay": replay_stats,
        "timingSeconds": {
            "surfaceComplex": round(surfaced - started, 6),
            "boundaryLoops": round(looped - surfaced, 6),
            "rawCtScoring": round(scored_at - looped, 6),
            "jointReconfiguration": round(reconfigured_at - scored_at, 6),
            "counterfactualReplay": round(replayed_at - reconfigured_at, 6),
            "writingAndPreview": round(finished - replayed_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"patchHoleMontage": preview_path.name},
        "method": {
            "decisionUnit": "one closed boundary loop supported by an entire local surface complex",
            "geometry": "affine and ridge-regularized quadratic patches fitted jointly to all boundary and two-hop context ribbons",
            "physicalTest": "native CT air-material-air profiles over the full missing patch versus normal-offset competing layers",
            "reconfiguration": "joint interface-matching factor graphs propose complete alternating swaps, not independent ribbon additions",
            "mutation": "diagnostic counterfactual only; no ribbon or component membership is changed",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
