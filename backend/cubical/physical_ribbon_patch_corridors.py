from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import (
    ComponentMesh,
    SurfaceChart,
    _draw_text,
    rasterize_chart,
    sample_depth_stack,
)
from .physical_ribbon_bridging import _load_inputs, _write_npz
from .physical_ribbon_continuity import _draw_line
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    _context_vertices,
    _objective_for_mask,
    _profile_correlation,
    _profile_score,
    _beam_set_packing,
    _sample_normal_profiles,
    _sample_volume_points,
    _selected_surface_adjacency,
    build_physical_ribbon_surface_complex,
    extract_surface_boundary_loops,
)
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_cumulative_replay import (
    load_materialized_cumulative_surface,
)


PHYSICAL_RIBBON_PATCH_CORRIDORS_SCHEMA = (
    "pareidolia.physical-ribbon-patch-corridors"
)
PHYSICAL_RIBBON_PATCH_CORRIDORS_VERSION = 1
PHYSICAL_RIBBON_PATCH_CORRIDORS_STEM = "physical-ribbon-patch-corridors-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonPatchCorridorSettings:
    minimum_component_ribbon_count: int = 32
    minimum_chart_separation_voxels: float = 0.20
    maximum_triangle_edge_voxels: float = 5.75
    maximum_triangle_normal_residual_degrees: float = 45.0
    minimum_triangle_area_voxels_squared: float = 0.20
    minimum_paired_boundary_edges: int = 3
    maximum_pair_sequence_gap_edges: int = 4
    maximum_reciprocal_pair_rank: int = 3
    minimum_sequence_anchor_density: float = 0.50
    minimum_corridor_width_boundary_edges: float = 1.25
    maximum_corridor_width_thicknesses: float = 2.0
    minimum_facing_cosine: float = 0.40
    maximum_endpoint_normal_residual_degrees: float = 85.0
    endpoint_chord_slack_degrees: float = 25.0
    minimum_curvature_radius_thicknesses: float = 0.45
    context_graph_hops: int = 2
    patch_pixel_step_voxels: float = 0.5
    maximum_patch_pixels: int = 8192
    flatten_context_width_boundary_edges: float = 2.0
    hermite_tensions: tuple[float, ...] = (0.5, 1.0, 1.5)
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
    maximum_scored_corridors: int = 128
    maximum_preview_corridors: int = 16
    minimum_context_profile_correlation: float = 0.85
    minimum_boundary_trace_correlation: float = 0.35
    minimum_zero_shift_margin: float = 0.20
    maximum_candidate_height_thicknesses: float = 0.25
    maximum_candidate_tangent_raster_steps: float = 1.75
    maximum_candidate_normal_degrees: float = 35.0
    maximum_candidate_thickness_ratio: float = 1.65
    surface_alignment_weight: float = 0.35
    configuration_beam_width: int = 16384
    minimum_replay_arc_region_fraction: float = 0.50
    maximum_replay_arc_triangle_distance_edges: float = 2.0
    minimum_replay_surface_area_retention: float = 0.98

    def __post_init__(self) -> None:
        positive = (
            self.minimum_chart_separation_voxels,
            self.maximum_triangle_edge_voxels,
            self.maximum_triangle_normal_residual_degrees,
            self.minimum_triangle_area_voxels_squared,
            self.minimum_corridor_width_boundary_edges,
            self.minimum_sequence_anchor_density,
            self.maximum_corridor_width_thicknesses,
            self.minimum_facing_cosine,
            self.maximum_endpoint_normal_residual_degrees,
            self.endpoint_chord_slack_degrees,
            self.minimum_curvature_radius_thicknesses,
            self.patch_pixel_step_voxels,
            self.flatten_context_width_boundary_edges,
            self.minimum_context_profile_correlation,
            self.minimum_zero_shift_margin,
            self.maximum_candidate_height_thicknesses,
            self.maximum_candidate_tangent_raster_steps,
            self.maximum_candidate_normal_degrees,
            self.maximum_candidate_thickness_ratio,
            self.surface_alignment_weight,
            self.minimum_replay_arc_region_fraction,
            self.maximum_replay_arc_triangle_distance_edges,
            self.minimum_replay_surface_area_retention,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("corridor scales must be finite and positive")
        if self.minimum_component_ribbon_count < 3:
            raise ValueError("corridor surfaces require nontrivial components")
        if self.minimum_paired_boundary_edges < 3:
            raise ValueError("corridors require at least three paired boundary edges")
        if self.maximum_pair_sequence_gap_edges < 1:
            raise ValueError("corridor sequence gap must be positive")
        if self.maximum_reciprocal_pair_rank < 1:
            raise ValueError("reciprocal pair rank must be positive")
        if self.minimum_sequence_anchor_density > 1.0:
            raise ValueError("sequence anchor density cannot exceed one")
        if self.minimum_replay_arc_region_fraction > 1.0:
            raise ValueError("replay arc-region fraction cannot exceed one")
        if self.minimum_replay_surface_area_retention > 1.0:
            raise ValueError("surface-area retention cannot exceed one")
        if self.minimum_facing_cosine >= 1.0:
            raise ValueError("facing cosine must be below one")
        if self.maximum_endpoint_normal_residual_degrees >= 90.0:
            raise ValueError("endpoint normal residual must be below 90 degrees")
        if self.context_graph_hops < 1:
            raise ValueError("corridor context must include at least one graph hop")
        if self.maximum_patch_pixels < 64:
            raise ValueError("corridor patch raster cap is too small")
        if not self.hermite_tensions or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.hermite_tensions
        ):
            raise ValueError("Hermite tensions must be finite and positive")
        if (
            tuple(sorted(self.profile_depth_fractions))
            != self.profile_depth_fractions
            or 0.0 not in self.profile_depth_fractions
        ):
            raise ValueError("profile depths must be sorted and include zero")
        if (
            tuple(sorted(self.competing_shift_thicknesses))
            != self.competing_shift_thicknesses
            or 0.0 not in self.competing_shift_thicknesses
        ):
            raise ValueError("competing shifts must be sorted and include zero")
        if self.maximum_scored_corridors < 1 or self.maximum_preview_corridors < 1:
            raise ValueError("corridor output counts must be positive")
        if not -1.0 < self.minimum_boundary_trace_correlation < 1.0:
            raise ValueError("boundary trace correlation gate must be in (-1, 1)")
        if self.minimum_context_profile_correlation >= 1.0:
            raise ValueError("context profile correlation gate must be below one")
        if self.maximum_candidate_normal_degrees >= 90.0:
            raise ValueError("candidate normal gate must be below 90 degrees")
        if self.maximum_candidate_thickness_ratio <= 1.0:
            raise ValueError("candidate thickness ratio must exceed one")
        if self.configuration_beam_width < 256:
            raise ValueError("configuration beam is too narrow for joint moves")

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def surface_settings(self) -> PhysicalRibbonPatchHoleSettings:
        return PhysicalRibbonPatchHoleSettings(
            minimum_component_ribbon_count=self.minimum_component_ribbon_count,
            minimum_chart_separation_voxels=self.minimum_chart_separation_voxels,
            maximum_triangle_edge_voxels=self.maximum_triangle_edge_voxels,
            maximum_triangle_normal_residual_degrees=(
                self.maximum_triangle_normal_residual_degrees
            ),
            minimum_triangle_area_voxels_squared=(
                self.minimum_triangle_area_voxels_squared
            ),
        )


def _boundary_edge_catalog(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    midpoint_xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    normal_xyz = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    component = np.asarray(surface["component"], dtype=np.int32)
    loop_offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    loop_vertex = np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32)
    loop_kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    loop_region = np.asarray(loops["loopTriangleRegion"], dtype=np.int32)
    edge_triangle: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, left_value in enumerate(triangle):
            right_value = int(triangle[(edge_index + 1) % 3])
            edge = (
                min(int(left_value), right_value),
                max(int(left_value), right_value),
            )
            edge_triangle[edge].append(triangle_index)
    loop_values: list[int] = []
    position_values: list[int] = []
    loop_count_values: list[int] = []
    region_values: list[int] = []
    component_values: list[int] = []
    first_values: list[int] = []
    second_values: list[int] = []
    midpoint_uv_values: list[np.ndarray] = []
    outward_uv_values: list[np.ndarray] = []
    midpoint_xyz_values: list[np.ndarray] = []
    normal_xyz_values: list[np.ndarray] = []
    thickness_values: list[float] = []
    edge_length_values: list[float] = []
    for loop_index in np.flatnonzero(loop_kind == 0):
        nodes = loop_vertex[
            int(loop_offset[loop_index]) : int(loop_offset[loop_index + 1])
        ]
        for position, first_node_value in enumerate(nodes):
            first_node = int(first_node_value)
            second_node = int(nodes[(position + 1) % len(nodes)])
            key = (min(first_node, second_node), max(first_node, second_node))
            incident = edge_triangle.get(key, ())
            if len(incident) != 1:
                continue
            triangle = triangles[int(incident[0])]
            midpoint_uv = 0.5 * (
                chart_uv[first_node] + chart_uv[second_node]
            )
            triangle_center = np.mean(chart_uv[triangle], axis=0)
            outward_uv = midpoint_uv - triangle_center
            outward_length = float(np.linalg.norm(outward_uv))
            if outward_length <= 1.0e-8:
                continue
            outward_uv /= outward_length
            normal = normal_xyz[first_node] + normal_xyz[second_node]
            normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
            loop_values.append(int(loop_index))
            position_values.append(position)
            loop_count_values.append(len(nodes))
            region_values.append(int(loop_region[loop_index]))
            component_values.append(int(component[first_node]))
            first_values.append(first_node)
            second_values.append(second_node)
            midpoint_uv_values.append(midpoint_uv)
            outward_uv_values.append(outward_uv)
            midpoint_xyz_values.append(
                0.5 * (midpoint_xyz[first_node] + midpoint_xyz[second_node])
            )
            normal_xyz_values.append(normal)
            thickness_values.append(
                0.5 * float(thickness[first_node] + thickness[second_node])
            )
            edge_length_values.append(
                float(
                    np.linalg.norm(
                        chart_uv[first_node] - chart_uv[second_node]
                    )
                )
            )
    return {
        "boundaryEdgeLoopIndex": np.asarray(loop_values, dtype=np.int32),
        "boundaryEdgeLoopPosition": np.asarray(position_values, dtype=np.int32),
        "boundaryEdgeLoopCount": np.asarray(loop_count_values, dtype=np.int32),
        "boundaryEdgeTriangleRegion": np.asarray(region_values, dtype=np.int32),
        "boundaryEdgeTopologyComponent": np.asarray(
            component_values, dtype=np.int32
        ),
        "boundaryEdgeFirstFrontierIndex": np.asarray(first_values, dtype=np.int32),
        "boundaryEdgeSecondFrontierIndex": np.asarray(
            second_values, dtype=np.int32
        ),
        "boundaryEdgeMidpointUV": np.asarray(
            midpoint_uv_values, dtype=np.float32
        ).reshape((-1, 2)),
        "boundaryEdgeOutwardUV": np.asarray(
            outward_uv_values, dtype=np.float32
        ).reshape((-1, 2)),
        "boundaryEdgeMidpointXYZ": np.asarray(
            midpoint_xyz_values, dtype=np.float32
        ).reshape((-1, 3)),
        "boundaryEdgeNormalXYZ": np.asarray(
            normal_xyz_values, dtype=np.float32
        ).reshape((-1, 3)),
        "boundaryEdgeThicknessVoxels": np.asarray(
            thickness_values, dtype=np.float32
        ),
        "boundaryEdgeLengthVoxels": np.asarray(
            edge_length_values, dtype=np.float32
        ),
    }


def _cyclic_distance(first: int, second: int, count: int) -> int:
    direct = abs(first - second)
    return min(direct, count - direct)


def _corridor_pair_metrics(
    first_index: int,
    second_index: int,
    boundary: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> tuple[float, float, float] | None:
    loop = np.asarray(boundary["boundaryEdgeLoopIndex"], dtype=np.int32)
    if loop[first_index] == loop[second_index]:
        return None
    uv = np.asarray(boundary["boundaryEdgeMidpointUV"], dtype=np.float32)
    outward = np.asarray(boundary["boundaryEdgeOutwardUV"], dtype=np.float32)
    edge_length = np.asarray(
        boundary["boundaryEdgeLengthVoxels"], dtype=np.float32
    )
    thickness = np.asarray(
        boundary["boundaryEdgeThicknessVoxels"], dtype=np.float32
    )
    delta_uv = uv[second_index] - uv[first_index]
    distance_uv = float(np.linalg.norm(delta_uv))
    mean_edge = 0.5 * float(edge_length[first_index] + edge_length[second_index])
    mean_thickness = 0.5 * float(
        thickness[first_index] + thickness[second_index]
    )
    if (
        distance_uv
        < settings.minimum_corridor_width_boundary_edges * mean_edge
        or distance_uv
        > max(16.0, settings.maximum_corridor_width_thicknesses * mean_thickness)
    ):
        return None
    direction = delta_uv / max(distance_uv, 1.0e-12)
    if (
        float(np.dot(outward[first_index], direction))
        < settings.minimum_facing_cosine
        or float(np.dot(outward[second_index], -direction))
        < settings.minimum_facing_cosine
    ):
        return None
    normal = np.asarray(boundary["boundaryEdgeNormalXYZ"], dtype=np.float32)
    signed_cosine = float(np.dot(normal[first_index], normal[second_index]))
    normal_degrees = math.degrees(
        math.acos(min(max(abs(signed_cosine), 0.0), 1.0))
    )
    if normal_degrees > settings.maximum_endpoint_normal_residual_degrees:
        return None
    xyz = np.asarray(boundary["boundaryEdgeMidpointXYZ"], dtype=np.float32)
    delta_xyz = xyz[second_index] - xyz[first_index]
    chord = float(np.linalg.norm(delta_xyz))
    if chord <= 1.0e-8:
        return None
    first_height_fraction = abs(
        float(np.dot(delta_xyz, normal[first_index]))
    ) / chord
    second_height_fraction = abs(
        float(np.dot(delta_xyz, normal[second_index]))
    ) / chord
    allowed_height_fraction = math.sin(
        math.radians(
            min(
                normal_degrees + settings.endpoint_chord_slack_degrees,
                89.0,
            )
        )
    )
    if max(first_height_fraction, second_height_fraction) > max(
        0.25, allowed_height_fraction
    ):
        return None
    theta = max(math.radians(normal_degrees), 1.0e-3)
    curvature_radius = chord / max(2.0 * math.sin(0.5 * theta), 1.0e-3)
    radius_thicknesses = curvature_radius / max(mean_thickness, 1.0e-6)
    if radius_thicknesses < settings.minimum_curvature_radius_thicknesses:
        return None
    return distance_uv, normal_degrees, radius_thicknesses


def _best_monotone_corridor_chain(
    oriented: list[tuple[int, int, float, float, float]],
    position: np.ndarray,
    loop_count: np.ndarray,
    *,
    maximum_gap: int,
) -> tuple[list[tuple[int, int, float, float, float]], int, float]:
    """Select one dense, order-preserving arc alignment from local alternatives."""

    if not oriented:
        return [], 0, 0.0
    first_count = int(loop_count[oriented[0][0]])
    second_count = int(loop_count[oriented[0][1]])
    first_position = np.asarray(
        [position[value[0]] for value in oriented], dtype=np.int32
    )
    ordered_position = np.sort(np.unique(first_position))
    first_gap = np.diff(
        np.concatenate((ordered_position, ordered_position[:1] + first_count))
    )
    first_cut = int(
        ordered_position[(int(np.argmax(first_gap)) + 1) % len(ordered_position)]
    )
    first_unwrapped = first_position.copy()
    first_unwrapped[first_unwrapped < first_cut] += first_count
    second_position = np.asarray(
        [position[value[1]] for value in oriented], dtype=np.int32
    )
    best_chain: list[int] = []
    best_direction = 0
    best_density = 0.0
    best_key: tuple[float, ...] | None = None
    for direction in (-1, 1):
        for anchor in np.unique(second_position):
            second_progress = (
                direction * (second_position - int(anchor))
            ) % second_count
            order = sorted(
                range(len(oriented)),
                key=lambda value: (
                    int(first_unwrapped[value]),
                    int(second_progress[value]),
                    value,
                ),
            )
            paths: list[list[int]] = [[] for _ in oriented]
            for order_index, current in enumerate(order):
                current_path = [current]
                for previous in order[:order_index]:
                    first_delta = int(
                        first_unwrapped[current] - first_unwrapped[previous]
                    )
                    second_delta = int(
                        second_progress[current] - second_progress[previous]
                    )
                    if not (
                        0 < first_delta <= maximum_gap
                        and 0 < second_delta <= maximum_gap
                    ):
                        continue
                    candidate_path = paths[previous] + [current]
                    if len(candidate_path) > len(current_path):
                        current_path = candidate_path
                    elif len(candidate_path) == len(current_path):
                        candidate_distance = sum(
                            oriented[value][2] for value in candidate_path
                        )
                        current_distance = sum(
                            oriented[value][2] for value in current_path
                        )
                        if candidate_distance < current_distance:
                            current_path = candidate_path
                paths[current] = current_path
            for chain in paths:
                first_span = int(
                    first_unwrapped[chain[-1]] - first_unwrapped[chain[0]]
                )
                second_span = int(
                    second_progress[chain[-1]] - second_progress[chain[0]]
                )
                density = len(chain) / max(max(first_span, second_span) + 1, 1)
                median_distance = float(
                    np.median([oriented[value][2] for value in chain])
                )
                key = (
                    float(len(chain)),
                    density,
                    -median_distance,
                    -float(direction < 0),
                    -float(min(chain)),
                )
                if best_key is None or key > best_key:
                    best_key = key
                    best_chain = chain
                    best_direction = direction
                    best_density = density
    return (
        [oriented[value] for value in best_chain],
        best_direction,
        best_density,
    )


def extract_surface_patch_corridors(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    boundary = _boundary_edge_catalog(surface, loops)
    component = np.asarray(
        boundary["boundaryEdgeTopologyComponent"], dtype=np.int32
    )
    loop = np.asarray(boundary["boundaryEdgeLoopIndex"], dtype=np.int32)
    position = np.asarray(boundary["boundaryEdgeLoopPosition"], dtype=np.int32)
    loop_count = np.asarray(boundary["boundaryEdgeLoopCount"], dtype=np.int32)
    by_component: dict[int, list[int]] = defaultdict(list)
    for index, component_id in enumerate(component):
        by_component[int(component_id)].append(index)
    ranked_pair: dict[int, list[tuple[int, float, float, float]]] = {}
    compatible_directed_count = 0
    for indices in by_component.values():
        for first_index in indices:
            compatible: list[tuple[int, float, float, float]] = []
            for second_index in indices:
                if first_index == second_index:
                    continue
                metrics = _corridor_pair_metrics(
                    first_index,
                    second_index,
                    boundary,
                    settings=settings,
                )
                if metrics is None:
                    continue
                compatible_directed_count += 1
                distance, normal_degrees, radius_thicknesses = metrics
                compatible.append(
                    (
                        second_index,
                        distance,
                        normal_degrees,
                        radius_thicknesses,
                    )
                )
            if compatible:
                compatible.sort(key=lambda value: (value[1], value[0]))
                ranked_pair[first_index] = compatible[
                    : settings.maximum_reciprocal_pair_rank
                ]
    reciprocal_by_key: dict[
        tuple[int, int], tuple[int, int, float, float, float]
    ] = {}
    mutual_first_choice_count = 0
    for first_index, choices in ranked_pair.items():
        for first_rank, values in enumerate(choices):
            second_index, distance, normal_degrees, radius_thicknesses = values
            reverse_choices = ranked_pair.get(second_index, ())
            reverse_rank = next(
                (
                    rank
                    for rank, reverse in enumerate(reverse_choices)
                    if reverse[0] == first_index
                ),
                None,
            )
            if reverse_rank is None:
                continue
            if first_rank == 0 and reverse_rank == 0 and first_index < second_index:
                mutual_first_choice_count += 1
            key = (min(first_index, second_index), max(first_index, second_index))
            reciprocal_by_key.setdefault(
                key,
                (
                    key[0],
                    key[1],
                    distance,
                    normal_degrees,
                    radius_thicknesses,
                ),
            )
    reciprocal = [reciprocal_by_key[key] for key in sorted(reciprocal_by_key)]
    parent = np.arange(len(reciprocal), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    def union(first_value: int, second_value: int) -> None:
        first_root = find(first_value)
        second_root = find(second_value)
        if first_root != second_root:
            parent[max(first_root, second_root)] = min(first_root, second_root)

    for first_pair_index, first_pair in enumerate(reciprocal):
        first_edges = first_pair[:2]
        first_key = tuple(sorted(int(loop[value]) for value in first_edges))
        first_by_loop = {int(loop[value]): value for value in first_edges}
        for second_pair_index in range(first_pair_index + 1, len(reciprocal)):
            second_edges = reciprocal[second_pair_index][:2]
            second_key = tuple(sorted(int(loop[value]) for value in second_edges))
            if second_key != first_key:
                continue
            second_by_loop = {int(loop[value]): value for value in second_edges}
            adjacent = True
            for loop_index in first_key:
                first_edge = first_by_loop[loop_index]
                second_edge = second_by_loop[loop_index]
                if _cyclic_distance(
                    int(position[first_edge]),
                    int(position[second_edge]),
                    int(loop_count[first_edge]),
                ) > settings.maximum_pair_sequence_gap_edges:
                    adjacent = False
                    break
            if adjacent:
                union(first_pair_index, second_pair_index)
    groups: dict[int, list[int]] = defaultdict(list)
    for pair_index in range(len(reciprocal)):
        groups[find(pair_index)].append(pair_index)
    retained_groups: list[
        tuple[list[tuple[int, int, float, float, float]], int, float]
    ] = []
    for values in groups.values():
        if len(values) < settings.minimum_paired_boundary_edges:
            continue
        records = [reciprocal[value] for value in values]
        first_loop = min(int(loop[records[0][0]]), int(loop[records[0][1]]))
        oriented: list[tuple[int, int, float, float, float]] = []
        for record in records:
            first_edge, second_edge = record[:2]
            if int(loop[first_edge]) != first_loop:
                first_edge, second_edge = second_edge, first_edge
            oriented.append((first_edge, second_edge, *record[2:]))
        chain, direction, density = _best_monotone_corridor_chain(
            oriented,
            position,
            loop_count,
            maximum_gap=settings.maximum_pair_sequence_gap_edges,
        )
        if (
            len(chain) >= settings.minimum_paired_boundary_edges
            and density >= settings.minimum_sequence_anchor_density
        ):
            retained_groups.append((chain, direction, density))
    retained_groups.sort(
        key=lambda value: (
            -len(value[0]),
            -value[2],
            float(np.median([record[2] for record in value[0]])),
            value[0][0][0],
        )
    )
    corridor_pair_offset = [0]
    corridor_first_edge: list[int] = []
    corridor_second_edge: list[int] = []
    corridor_first_loop: list[int] = []
    corridor_second_loop: list[int] = []
    corridor_component: list[int] = []
    corridor_width: list[float] = []
    corridor_normal: list[float] = []
    corridor_radius: list[float] = []
    corridor_direction: list[int] = []
    corridor_density: list[float] = []
    for oriented, direction, density in retained_groups:
        records = oriented
        first_loop = min(int(loop[records[0][0]]), int(loop[records[0][1]]))
        second_loop = max(int(loop[records[0][0]]), int(loop[records[0][1]]))
        corridor_first_edge.extend(value[0] for value in oriented)
        corridor_second_edge.extend(value[1] for value in oriented)
        corridor_pair_offset.append(len(corridor_first_edge))
        corridor_first_loop.append(first_loop)
        corridor_second_loop.append(second_loop)
        corridor_component.append(int(component[oriented[0][0]]))
        corridor_width.append(float(np.median([value[2] for value in oriented])))
        corridor_normal.append(float(np.median([value[3] for value in oriented])))
        corridor_radius.append(float(np.median([value[4] for value in oriented])))
        corridor_direction.append(direction)
        corridor_density.append(density)
    arrays = {
        **boundary,
        "corridorPairOffset": np.asarray(corridor_pair_offset, dtype=np.int64),
        "corridorFirstBoundaryEdge": np.asarray(
            corridor_first_edge, dtype=np.int32
        ),
        "corridorSecondBoundaryEdge": np.asarray(
            corridor_second_edge, dtype=np.int32
        ),
        "corridorFirstLoopIndex": np.asarray(corridor_first_loop, dtype=np.int32),
        "corridorSecondLoopIndex": np.asarray(
            corridor_second_loop, dtype=np.int32
        ),
        "corridorTopologyComponent": np.asarray(
            corridor_component, dtype=np.int32
        ),
        "corridorMedianWidthVoxels": np.asarray(corridor_width, dtype=np.float32),
        "corridorMedianNormalResidualDegrees": np.asarray(
            corridor_normal, dtype=np.float32
        ),
        "corridorMedianCurvatureRadiusThicknesses": np.asarray(
            corridor_radius, dtype=np.float32
        ),
        "corridorPairingDirection": np.asarray(
            corridor_direction, dtype=np.int8
        ),
        "corridorSequenceAnchorDensity": np.asarray(
            corridor_density, dtype=np.float32
        ),
    }
    return arrays, {
        "outerBoundaryEdgeCount": len(loop),
        "compatibleDirectedEdgePairCount": compatible_directed_count,
        "rankedFacingEdgeCount": len(ranked_pair),
        "mutualFirstChoiceEdgePairCount": mutual_first_choice_count,
        "reciprocalRankedEdgePairCount": len(reciprocal),
        "maximumReciprocalPairRank": settings.maximum_reciprocal_pair_rank,
        "multiAnchorCorridorCount": len(retained_groups),
        "corridorPairedBoundaryEdgeCount": len(corridor_first_edge),
        "minimumSequenceAnchorDensity": settings.minimum_sequence_anchor_density,
        "identityLabelsUsed": False,
    }


def _interpolate_rows(
    coordinate: np.ndarray,
    values: np.ndarray,
    target: np.ndarray,
) -> np.ndarray:
    source = np.asarray(values, dtype=np.float64)
    if source.ndim == 1:
        return np.interp(target, coordinate, source)
    return np.column_stack(
        [np.interp(target, coordinate, source[:, axis]) for axis in range(source.shape[1])]
    )


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result / np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)


def _corridor_model_grid(
    corridor_index: int,
    corridors: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> dict[str, np.ndarray | float | int]:
    offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    first_edge = np.asarray(corridors["corridorFirstBoundaryEdge"], dtype=np.int32)[
        int(offset[corridor_index]) : int(offset[corridor_index + 1])
    ]
    second_edge = np.asarray(corridors["corridorSecondBoundaryEdge"], dtype=np.int32)[
        int(offset[corridor_index]) : int(offset[corridor_index + 1])
    ]
    midpoint_xyz = np.asarray(
        corridors["boundaryEdgeMidpointXYZ"], dtype=np.float32
    )
    midpoint_uv = np.asarray(corridors["boundaryEdgeMidpointUV"], dtype=np.float32)
    normal_xyz = np.asarray(corridors["boundaryEdgeNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(
        corridors["boundaryEdgeThicknessVoxels"], dtype=np.float32
    )
    first_xyz = midpoint_xyz[first_edge].astype(np.float64)
    second_xyz = midpoint_xyz[second_edge].astype(np.float64)
    first_uv = midpoint_uv[first_edge].astype(np.float64)
    second_uv = midpoint_uv[second_edge].astype(np.float64)
    first_normal = _normalize_rows(normal_xyz[first_edge])
    second_normal = _normalize_rows(normal_xyz[second_edge])
    second_normal[
        np.einsum("ij,ij->i", first_normal, second_normal) < 0.0
    ] *= -1.0
    first_thickness = thickness[first_edge].astype(np.float64)
    second_thickness = thickness[second_edge].astype(np.float64)
    if len(first_xyz) < settings.minimum_paired_boundary_edges:
        raise ValueError("corridor lost its multi-anchor support")
    along_step = 0.5 * (
        np.linalg.norm(np.diff(first_xyz, axis=0), axis=1)
        + np.linalg.norm(np.diff(second_xyz, axis=0), axis=1)
    )
    coordinate = np.concatenate(([0.0], np.cumsum(np.maximum(along_step, 1.0e-3))))
    total_length = float(coordinate[-1])
    straight_width = np.linalg.norm(second_xyz - first_xyz, axis=1)
    maximum_width = float(np.max(straight_width))
    step = float(settings.patch_pixel_step_voxels)
    estimated_rows = max(int(math.ceil(total_length / step)) + 1, 3)
    estimated_columns = max(int(math.ceil(maximum_width / step)) + 1, 3)
    estimated_pixels = estimated_rows * estimated_columns
    if estimated_pixels > settings.maximum_patch_pixels:
        step *= math.sqrt(estimated_pixels / settings.maximum_patch_pixels)
    row_count = max(int(math.ceil(total_length / step)) + 1, 3)
    column_count = max(int(math.ceil(maximum_width / step)) + 1, 3)
    row_coordinate = np.linspace(0.0, total_length, row_count)
    cross_coordinate = np.linspace(0.0, 1.0, column_count)
    first_xyz = _interpolate_rows(coordinate, first_xyz, row_coordinate)
    second_xyz = _interpolate_rows(coordinate, second_xyz, row_coordinate)
    first_uv = _interpolate_rows(coordinate, first_uv, row_coordinate)
    second_uv = _interpolate_rows(coordinate, second_uv, row_coordinate)
    first_normal = _normalize_rows(
        _interpolate_rows(coordinate, first_normal, row_coordinate)
    )
    second_normal = _normalize_rows(
        _interpolate_rows(coordinate, second_normal, row_coordinate)
    )
    second_normal[
        np.einsum("ij,ij->i", first_normal, second_normal) < 0.0
    ] *= -1.0
    first_thickness = _interpolate_rows(
        coordinate, first_thickness, row_coordinate
    )
    second_thickness = _interpolate_rows(
        coordinate, second_thickness, row_coordinate
    )
    chord = second_xyz - first_xyz
    chord_length = np.linalg.norm(chord, axis=1)
    first_projection = chord - np.einsum(
        "ij,ij->i", chord, first_normal
    )[:, None] * first_normal
    second_projection = chord - np.einsum(
        "ij,ij->i", chord, second_normal
    )[:, None] * second_normal
    first_along = np.gradient(first_xyz, row_coordinate, axis=0)
    second_along = np.gradient(second_xyz, row_coordinate, axis=0)
    first_fallback = np.cross(first_along, first_normal)
    second_fallback = np.cross(second_along, second_normal)
    first_fallback[
        np.einsum("ij,ij->i", first_fallback, chord) < 0.0
    ] *= -1.0
    second_fallback[
        np.einsum("ij,ij->i", second_fallback, chord) < 0.0
    ] *= -1.0
    first_projection_length = np.linalg.norm(first_projection, axis=1)
    second_projection_length = np.linalg.norm(second_projection, axis=1)
    first_projection[first_projection_length < 0.2 * chord_length] = first_fallback[
        first_projection_length < 0.2 * chord_length
    ]
    second_projection[
        second_projection_length < 0.2 * chord_length
    ] = second_fallback[second_projection_length < 0.2 * chord_length]
    first_direction = _normalize_rows(first_projection)
    second_direction = _normalize_rows(second_projection)
    model_count = 1 + len(settings.hermite_tensions)
    points = np.empty(
        (model_count, row_count, column_count, 3), dtype=np.float64
    )
    derivative_cross = np.empty_like(points)
    t_value = cross_coordinate[None, :, None]
    points[0] = (
        (1.0 - t_value) * first_xyz[:, None, :]
        + t_value * second_xyz[:, None, :]
    )
    derivative_cross[0] = chord[:, None, :]
    t_squared = t_value * t_value
    t_cubed = t_squared * t_value
    h00 = 2.0 * t_cubed - 3.0 * t_squared + 1.0
    h10 = t_cubed - 2.0 * t_squared + t_value
    h01 = -2.0 * t_cubed + 3.0 * t_squared
    h11 = t_cubed - t_squared
    dh00 = 6.0 * t_squared - 6.0 * t_value
    dh10 = 3.0 * t_squared - 4.0 * t_value + 1.0
    dh01 = -6.0 * t_squared + 6.0 * t_value
    dh11 = 3.0 * t_squared - 2.0 * t_value
    for model_index, tension in enumerate(settings.hermite_tensions, start=1):
        first_derivative = (
            float(tension) * chord_length[:, None] * first_direction
        )
        second_derivative = (
            float(tension) * chord_length[:, None] * second_direction
        )
        points[model_index] = (
            h00 * first_xyz[:, None, :]
            + h10 * first_derivative[:, None, :]
            + h01 * second_xyz[:, None, :]
            + h11 * second_derivative[:, None, :]
        )
        derivative_cross[model_index] = (
            dh00 * first_xyz[:, None, :]
            + dh10 * first_derivative[:, None, :]
            + dh01 * second_xyz[:, None, :]
            + dh11 * second_derivative[:, None, :]
        )
    reference_normal = _normalize_rows(
        (1.0 - t_value) * first_normal[:, None, :]
        + t_value * second_normal[:, None, :]
    )
    normals = np.empty_like(points)
    arc_length_ratio = np.empty(model_count, dtype=np.float32)
    minimum_radius_thicknesses = np.empty(model_count, dtype=np.float32)
    thickness_grid = (
        (1.0 - cross_coordinate[None, :]) * first_thickness[:, None]
        + cross_coordinate[None, :] * second_thickness[:, None]
    )
    for model_index in range(model_count):
        derivative_along = np.gradient(
            points[model_index], row_coordinate, axis=0
        )
        model_normal = _normalize_rows(
            np.cross(derivative_along, derivative_cross[model_index])
        )
        model_normal[
            np.einsum("ijk,ijk->ij", model_normal, reference_normal) < 0.0
        ] *= -1.0
        normals[model_index] = model_normal
        across_length = np.sum(
            np.linalg.norm(np.diff(points[model_index], axis=1), axis=2), axis=1
        )
        arc_length_ratio[model_index] = float(
            np.median(across_length / np.maximum(chord_length, 1.0e-6))
        )
        adjacent_cosine = np.clip(
            np.abs(
                np.einsum(
                    "ijk,ijk->ij",
                    model_normal[:, :-1],
                    model_normal[:, 1:],
                )
            ),
            0.0,
            1.0,
        )
        adjacent_angle = np.arccos(adjacent_cosine)
        segment_length = np.linalg.norm(
            np.diff(points[model_index], axis=1), axis=2
        )
        radius = segment_length / np.maximum(adjacent_angle, 1.0e-3)
        minimum_radius_thicknesses[model_index] = float(
            np.percentile(
                radius / np.maximum(thickness_grid[:, :-1], 1.0e-6), 10.0
            )
        )
    chart_grid = (
        (1.0 - t_value[..., :2]) * first_uv[:, None, :]
        + t_value[..., :2] * second_uv[:, None, :]
    )
    return {
        "pointsXYZ": points.astype(np.float32),
        "normalXYZ": normals.astype(np.float32),
        "thicknessVoxels": thickness_grid.astype(np.float32),
        "chartUV": chart_grid.astype(np.float32),
        "rowCoordinateVoxels": row_coordinate.astype(np.float32),
        "crossCoordinate": cross_coordinate.astype(np.float32),
        "rowCount": row_count,
        "columnCount": column_count,
        "pixelStepVoxels": step,
        "arcLengthRatio": arc_length_ratio,
        "minimumCurvatureRadiusThicknesses": minimum_radius_thicknesses,
        "firstBoundaryEdge": first_edge,
        "secondBoundaryEdge": second_edge,
        "firstBoundaryXYZ": first_xyz.astype(np.float32),
        "secondBoundaryXYZ": second_xyz.astype(np.float32),
        "firstBoundaryNormalXYZ": first_normal.astype(np.float32),
        "secondBoundaryNormalXYZ": second_normal.astype(np.float32),
        "firstBoundaryThicknessVoxels": first_thickness.astype(np.float32),
        "secondBoundaryThicknessVoxels": second_thickness.astype(np.float32),
        "firstOutwardDirectionXYZ": first_direction.astype(np.float32),
        "secondInteriorDirectionXYZ": second_direction.astype(np.float32),
    }


def _shifted_trace_correlation(first: np.ndarray, second: np.ndarray) -> float:
    def high_pass(value: np.ndarray) -> np.ndarray:
        source = np.asarray(value, dtype=np.float32)
        if len(source) < 5:
            return source - float(np.mean(source))
        padded = np.pad(source, (2, 2), mode="edge")
        smooth = np.convolve(padded, np.ones(5) / 5.0, mode="valid")
        return source - smooth

    first_value = high_pass(first)
    second_value = high_pass(second)
    result = -1.0
    maximum_shift = min(3, max(len(first_value) // 3, 0))
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            left = first_value[-shift:]
            right = second_value[: len(second_value) + shift]
        elif shift > 0:
            left = first_value[: len(first_value) - shift]
            right = second_value[shift:]
        else:
            left, right = first_value, second_value
        result = max(result, _profile_correlation(left, right))
    return float(result)


def _structure_tensor(values: np.ndarray) -> tuple[float, float]:
    image = np.asarray(values, dtype=np.float64)
    if min(image.shape) < 2:
        return 0.0, 0.0
    gradient_row, gradient_column = np.gradient(image)
    tensor = np.asarray(
        (
            (np.mean(gradient_column**2), np.mean(gradient_column * gradient_row)),
            (np.mean(gradient_column * gradient_row), np.mean(gradient_row**2)),
        )
    )
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    order = np.argsort(eigenvalue)[::-1]
    high, low = float(eigenvalue[order[0]]), float(eigenvalue[order[1]])
    direction = eigenvector[:, order[0]]
    angle = math.degrees(math.atan2(float(direction[1]), float(direction[0]))) % 180.0
    anisotropy = (high - low) / max(high + low, 1.0e-12)
    return angle, anisotropy


def _texture_strength(values: np.ndarray) -> float:
    image = np.asarray(values, dtype=np.float64)
    if min(image.shape) < 2:
        return 0.0
    gradient = np.gradient(image)
    gradient_energy = math.sqrt(
        float(np.mean(gradient[0] ** 2 + gradient[1] ** 2))
    )
    _, anisotropy = _structure_tensor(image)
    return gradient_energy * (0.25 + 0.75 * anisotropy)


def score_surface_patch_corridors(
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    corridor_count = len(np.asarray(corridors["corridorFirstLoopIndex"]))
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    pair_count = np.diff(pair_offset)
    width = np.asarray(corridors["corridorMedianWidthVoxels"], dtype=np.float32)
    priority = np.lexsort((width, -pair_count))
    scored_corridor = priority[: settings.maximum_scored_corridors].astype(np.int32)
    adjacency = _selected_surface_adjacency(surface)
    component = np.asarray(surface["component"], dtype=np.int32)
    center_xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    signed_normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    first_node = np.asarray(
        corridors["boundaryEdgeFirstFrontierIndex"], dtype=np.int32
    )
    second_node = np.asarray(
        corridors["boundaryEdgeSecondFrontierIndex"], dtype=np.int32
    )
    first_edge_values = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )
    second_edge_values = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )
    depth = np.asarray(settings.profile_depth_fractions, dtype=np.float32)
    shifts = np.asarray(settings.competing_shift_thicknesses, dtype=np.float32)
    zero_shift = int(np.flatnonzero(shifts == 0.0)[0])
    volume = source.memmap()
    model_count = 1 + len(settings.hermite_tensions)

    context_offset = [0]
    context_vertex: list[int] = []
    patch_offset = [0]
    patch_xyz: list[np.ndarray] = []
    patch_normal: list[np.ndarray] = []
    patch_thickness: list[np.ndarray] = []
    patch_uv: list[np.ndarray] = []
    image_offset = [0]
    image_values: list[np.ndarray] = []
    row_count_values: list[int] = []
    column_count_values: list[int] = []
    image_column_count_values: list[int] = []
    image_first_boundary_column_values: list[int] = []
    image_second_boundary_column_values: list[int] = []
    pixel_step_values: list[float] = []
    model_shift_scores: list[np.ndarray] = []
    model_shift_profiles: list[np.ndarray] = []
    context_profiles: list[np.ndarray] = []
    context_scores: list[float] = []
    profile_correlations: list[np.ndarray] = []
    shift_margins: list[np.ndarray] = []
    trace_correlations: list[np.ndarray] = []
    texture_angles: list[np.ndarray] = []
    texture_anisotropies: list[np.ndarray] = []
    texture_depth_values: list[np.ndarray] = []
    arc_length_ratios: list[np.ndarray] = []
    curvature_radius_values: list[np.ndarray] = []
    rank_scores: list[np.ndarray] = []
    selected_models: list[int] = []
    intensity_scales: list[float] = []
    for corridor_index in scored_corridor:
        model = _corridor_model_grid(
            int(corridor_index), corridors, settings=settings
        )
        start, stop = (
            int(pair_offset[corridor_index]),
            int(pair_offset[corridor_index + 1]),
        )
        boundary_edge = np.concatenate(
            (
                first_edge_values[start:stop],
                second_edge_values[start:stop],
            )
        )
        boundary_vertex = np.unique(
            np.concatenate(
                (
                    first_node[boundary_edge],
                    second_node[boundary_edge],
                )
            )
        )
        context = _context_vertices(
            boundary_vertex,
            adjacency,
            component,
            graph_hops=settings.context_graph_hops,
        )
        context_profile = _sample_normal_profiles(
            source,
            volume,
            center_xyz[context],
            signed_normal[context],
            thickness[context],
            depth,
            np.asarray((0.0,), dtype=np.float32),
        )[0]
        points = np.asarray(model["pointsXYZ"], dtype=np.float32)
        normals = np.asarray(model["normalXYZ"], dtype=np.float32)
        thickness_grid = np.asarray(model["thicknessVoxels"], dtype=np.float32)
        row_count = int(model["rowCount"])
        column_count = int(model["columnCount"])
        per_model_profile: list[np.ndarray] = []
        per_model_image: list[np.ndarray] = []
        per_model_texture_depth: list[float] = []
        boundary_edge_length = np.asarray(
            corridors["boundaryEdgeLengthVoxels"], dtype=np.float32
        )[boundary_edge]
        context_width = settings.flatten_context_width_boundary_edges * float(
            np.median(boundary_edge_length)
        )
        context_column_count = max(
            int(math.ceil(context_width / float(model["pixelStepVoxels"]))), 2
        )
        context_distance = (
            np.arange(context_column_count, 0, -1, dtype=np.float32)
            * float(model["pixelStepVoxels"])
        )
        first_boundary_xyz = np.asarray(model["firstBoundaryXYZ"], dtype=np.float32)
        second_boundary_xyz = np.asarray(model["secondBoundaryXYZ"], dtype=np.float32)
        first_boundary_normal = np.asarray(
            model["firstBoundaryNormalXYZ"], dtype=np.float32
        )
        second_boundary_normal = np.asarray(
            model["secondBoundaryNormalXYZ"], dtype=np.float32
        )
        first_boundary_thickness = np.asarray(
            model["firstBoundaryThicknessVoxels"], dtype=np.float32
        )
        second_boundary_thickness = np.asarray(
            model["secondBoundaryThicknessVoxels"], dtype=np.float32
        )
        first_outward = np.asarray(
            model["firstOutwardDirectionXYZ"], dtype=np.float32
        )
        second_interior = np.asarray(
            model["secondInteriorDirectionXYZ"], dtype=np.float32
        )
        left_context_xyz = (
            first_boundary_xyz[:, None, :]
            - context_distance[None, :, None] * first_outward[:, None, :]
        )
        right_context_xyz = (
            second_boundary_xyz[:, None, :]
            + context_distance[::-1][None, :, None]
            * second_interior[:, None, :]
        )
        for model_index in range(model_count):
            profile = _sample_normal_profiles(
                source,
                volume,
                points[model_index].reshape((-1, 3)),
                normals[model_index].reshape((-1, 3)),
                thickness_grid.reshape(-1),
                depth,
                shifts,
            )
            per_model_profile.append(profile)
            profile_grid = profile[zero_shift].reshape(
                (row_count, column_count, len(depth))
            )
            interior_depth = np.flatnonzero(np.abs(depth) <= 0.36)
            context_plane: list[tuple[np.ndarray, np.ndarray]] = []
            for depth_index in interior_depth:
                depth_fraction = float(depth[depth_index])
                left_points = (
                    left_context_xyz
                    + depth_fraction
                    * first_boundary_thickness[:, None, None]
                    * first_boundary_normal[:, None, :]
                )
                right_points = (
                    right_context_xyz
                    + depth_fraction
                    * second_boundary_thickness[:, None, None]
                    * second_boundary_normal[:, None, :]
                )
                context_plane.append(
                    (
                        _sample_volume_points(source, volume, left_points),
                        _sample_volume_points(source, volume, right_points),
                    )
                )
            strength = np.asarray(
                [
                    0.5
                    * (
                        _texture_strength(value[0])
                        + _texture_strength(value[1])
                    )
                    for value in context_plane
                ]
            )
            selected_depth_row = int(np.argmax(strength))
            texture_depth_index = int(interior_depth[selected_depth_row])
            per_model_texture_depth.append(float(depth[texture_depth_index]))
            per_model_image.append(
                np.concatenate(
                    (
                        context_plane[selected_depth_row][0],
                        profile_grid[:, :, texture_depth_index],
                        context_plane[selected_depth_row][1],
                    ),
                    axis=1,
                ).astype(np.float32)
            )
        finite_intensity = np.concatenate(
            (
                context_profile[np.isfinite(context_profile)],
                *(value[np.isfinite(value)] for value in per_model_profile),
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
                context_profile[None, :, :],
                depth,
                intensity_scale=intensity_scale,
            )[0]
        )
        context_median_profile = np.nanmedian(context_profile, axis=0)
        shift_score = np.stack(
            [
                _profile_score(value, depth, intensity_scale=intensity_scale)
                for value in per_model_profile
            ]
        )
        shift_profile = np.stack(
            [np.nanmedian(value, axis=1) for value in per_model_profile]
        )
        profile_correlation = np.asarray(
            [
                _profile_correlation(
                    value[zero_shift], context_median_profile
                )
                for value in shift_profile
            ],
            dtype=np.float32,
        )
        competing = np.delete(shift_score, zero_shift, axis=1)
        shift_margin = shift_score[:, zero_shift] - np.max(competing, axis=1)
        trace_correlation = np.asarray(
            [
                _shifted_trace_correlation(
                    np.mean(
                        value[
                            :,
                            max(context_column_count - 2, 0) : context_column_count,
                        ],
                        axis=1,
                    ),
                    np.mean(
                        value[
                            :,
                            context_column_count
                            + column_count : context_column_count
                            + column_count
                            + min(2, context_column_count),
                        ],
                        axis=1,
                    ),
                )
                for value in per_model_image
            ],
            dtype=np.float32,
        )
        texture = np.asarray(
            [_structure_tensor(value) for value in per_model_image],
            dtype=np.float32,
        )
        radius = np.asarray(
            model["minimumCurvatureRadiusThicknesses"], dtype=np.float32
        )
        arc_ratio = np.asarray(model["arcLengthRatio"], dtype=np.float32)
        curvature_penalty = np.maximum(
            settings.minimum_curvature_radius_thicknesses - radius, 0.0
        )
        rank_score = (
            shift_score[:, zero_shift]
            + 0.35 * profile_correlation
            + 0.25 * shift_margin
            + 0.20 * np.maximum(trace_correlation, 0.0)
            - 0.35 * curvature_penalty
        )
        selected_model = int(np.nanargmax(rank_score))
        selected_image = per_model_image[selected_model]
        context_vertex.extend(int(value) for value in context)
        context_offset.append(len(context_vertex))
        patch_xyz.append(points[selected_model].reshape((-1, 3)))
        patch_normal.append(normals[selected_model].reshape((-1, 3)))
        patch_thickness.append(thickness_grid.reshape(-1))
        patch_uv.append(
            np.asarray(model["chartUV"], dtype=np.float32).reshape((-1, 2))
        )
        patch_offset.append(patch_offset[-1] + row_count * column_count)
        image_values.append(selected_image.reshape(-1))
        image_offset.append(image_offset[-1] + selected_image.size)
        row_count_values.append(row_count)
        column_count_values.append(column_count)
        image_column_count_values.append(
            column_count + 2 * context_column_count
        )
        image_first_boundary_column_values.append(context_column_count)
        image_second_boundary_column_values.append(
            context_column_count + column_count - 1
        )
        pixel_step_values.append(float(model["pixelStepVoxels"]))
        model_shift_scores.append(shift_score)
        model_shift_profiles.append(shift_profile)
        context_profiles.append(context_median_profile)
        context_scores.append(context_score)
        profile_correlations.append(profile_correlation)
        shift_margins.append(shift_margin)
        trace_correlations.append(trace_correlation)
        texture_angles.append(texture[:, 0])
        texture_anisotropies.append(texture[:, 1])
        texture_depth_values.append(
            np.asarray(per_model_texture_depth, dtype=np.float32)
        )
        arc_length_ratios.append(arc_ratio)
        curvature_radius_values.append(radius)
        rank_scores.append(rank_score)
        selected_models.append(selected_model)
        intensity_scales.append(intensity_scale)
    count = len(scored_corridor)
    profile_shape = (count, model_count, len(shifts), len(depth))
    arrays = {
        "scoredCorridorIndex": scored_corridor,
        "corridorContextOffset": np.asarray(context_offset, dtype=np.int64),
        "corridorContextVertexFrontierIndex": np.asarray(
            context_vertex, dtype=np.int32
        ),
        "corridorPatchOffset": np.asarray(patch_offset, dtype=np.int64),
        "corridorPatchXYZ": np.concatenate(patch_xyz).astype(np.float32)
        if patch_xyz
        else np.empty((0, 3), dtype=np.float32),
        "corridorPatchNormalXYZ": np.concatenate(patch_normal).astype(np.float32)
        if patch_normal
        else np.empty((0, 3), dtype=np.float32),
        "corridorPatchThicknessVoxels": np.concatenate(patch_thickness).astype(
            np.float32
        )
        if patch_thickness
        else np.empty(0, dtype=np.float32),
        "corridorPatchUV": np.concatenate(patch_uv).astype(np.float32)
        if patch_uv
        else np.empty((0, 2), dtype=np.float32),
        "corridorImageOffset": np.asarray(image_offset, dtype=np.int64),
        "corridorFlattenedCt": np.concatenate(image_values).astype(np.float32)
        if image_values
        else np.empty(0, dtype=np.float32),
        "corridorPatchRows": np.asarray(row_count_values, dtype=np.int32),
        "corridorPatchColumns": np.asarray(column_count_values, dtype=np.int32),
        "corridorImageRows": np.asarray(row_count_values, dtype=np.int32),
        "corridorImageColumns": np.asarray(
            image_column_count_values, dtype=np.int32
        ),
        "corridorImageFirstBoundaryColumn": np.asarray(
            image_first_boundary_column_values, dtype=np.int32
        ),
        "corridorImageSecondBoundaryColumn": np.asarray(
            image_second_boundary_column_values, dtype=np.int32
        ),
        "corridorRasterStepVoxels": np.asarray(pixel_step_values, dtype=np.float32),
        "corridorProfileDepthFractions": depth,
        "corridorCompetingShiftThicknesses": shifts,
        "corridorModelShiftPhysicalScore": np.asarray(
            model_shift_scores, dtype=np.float32
        ).reshape((count, model_count, len(shifts))),
        "corridorModelShiftMedianProfile": np.asarray(
            model_shift_profiles, dtype=np.float32
        ).reshape(profile_shape),
        "corridorContextMedianProfile": np.asarray(
            context_profiles, dtype=np.float32
        ).reshape((count, len(depth))),
        "corridorContextPhysicalScore": np.asarray(
            context_scores, dtype=np.float32
        ),
        "corridorZeroShiftContextProfileCorrelation": np.asarray(
            profile_correlations, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorZeroShiftCompetingMargin": np.asarray(
            shift_margins, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorBoundaryTraceCorrelation": np.asarray(
            trace_correlations, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorTextureAxisDegrees": np.asarray(
            texture_angles, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorTextureAnisotropy": np.asarray(
            texture_anisotropies, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorTextureDepthFraction": np.asarray(
            texture_depth_values, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorModelArcLengthRatio": np.asarray(
            arc_length_ratios, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorModelMinimumCurvatureRadiusThicknesses": np.asarray(
            curvature_radius_values, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorModelRankScore": np.asarray(
            rank_scores, dtype=np.float32
        ).reshape((count, model_count)),
        "corridorSelectedModel": np.asarray(selected_models, dtype=np.uint8),
        "corridorLocalIntensityScale": np.asarray(
            intensity_scales, dtype=np.float32
        ),
    }
    return arrays, {
        "scoredCorridorCount": count,
        "surfaceModels": [
            "ruled",
            *(f"hermite-tension-{value:g}" for value in settings.hermite_tensions),
        ],
        "decisionUnit": "two mutually facing multi-edge boundary arcs",
        "bendModel": "cubic Hermite cross-sections constrained by both endpoint tangent planes",
        "rawCtEvidence": "whole-corridor normal profiles and flattened center-plane texture",
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def _undirected_adjacency(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    source = np.concatenate((first, second)).astype(np.int32, copy=False)
    neighbor = np.concatenate((second, first)).astype(np.int32, copy=False)
    if weight is None:
        edge_weight = np.ones(len(source), dtype=np.float32)
    else:
        edge_weight = np.concatenate((weight, weight)).astype(np.float32, copy=False)
    order = np.argsort(source, kind="stable")
    degree = np.bincount(source, minlength=count)
    offset = np.concatenate(([0], np.cumsum(degree))).astype(np.int64)
    return offset, neighbor[order], edge_weight[order]


def _corridor_geometric_candidates(
    scored_row: int,
    scored: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> dict[str, np.ndarray]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    center = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    normal = np.asarray(ribbon["normalXYZ"], dtype=np.float32)[frontier]
    thickness = np.asarray(ribbon["thicknessVoxels"], dtype=np.float32)[frontier]
    selected = np.asarray(configuration["selected"]) > 0
    patch_offset = np.asarray(scored["corridorPatchOffset"], dtype=np.int64)
    start, stop = int(patch_offset[scored_row]), int(patch_offset[scored_row + 1])
    patch = np.asarray(scored["corridorPatchXYZ"], dtype=np.float32)[start:stop]
    patch_normal = np.asarray(
        scored["corridorPatchNormalXYZ"], dtype=np.float32
    )[start:stop]
    patch_thickness = np.asarray(
        scored["corridorPatchThicknessVoxels"], dtype=np.float32
    )[start:stop]
    median_thickness = float(np.median(patch_thickness))
    padding = max(4.0, 0.4 * median_thickness)
    in_box = (~selected) & np.all(
        (center >= np.min(patch, axis=0) - padding)
        & (center <= np.max(patch, axis=0) + padding),
        axis=1,
    )
    candidate = np.flatnonzero(in_box)
    if not len(candidate):
        empty = np.empty(0, dtype=np.float32)
        return {
            "frontierIndex": np.empty(0, dtype=np.int32),
            "nearestPatchPixel": np.empty(0, dtype=np.int32),
            "heightResidualVoxels": empty,
            "tangentResidualVoxels": empty,
            "normalResidualDegrees": empty,
            "thicknessRatio": empty,
            "surfaceAlignment": empty,
        }
    delta = center[candidate, None, :] - patch[None, :, :]
    distance_squared = np.einsum("ijk,ijk->ij", delta, delta)
    nearest = np.argmin(distance_squared, axis=1)
    residual = delta[np.arange(len(candidate)), nearest]
    nearest_normal = patch_normal[nearest]
    nearest_thickness = patch_thickness[nearest]
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
        thickness[candidate] / nearest_thickness,
        nearest_thickness / thickness[candidate],
    )
    height_scale = np.maximum(
        2.5,
        settings.maximum_candidate_height_thicknesses * nearest_thickness,
    )
    tangent_scale = (
        settings.maximum_candidate_tangent_raster_steps
        * float(scored["corridorRasterStepVoxels"][scored_row])
    )
    retained = (
        (height <= height_scale)
        & (tangent <= tangent_scale)
        & (normal_degrees <= settings.maximum_candidate_normal_degrees)
        & (thickness_ratio <= settings.maximum_candidate_thickness_ratio)
    )
    alignment = (
        np.exp(-0.5 * (height / np.maximum(height_scale, 1.0e-6)) ** 2)
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


def solve_patch_corridor_reconfigurations(
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    node_count = len(frontier)
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
    edge_offset, edge_neighbor, edge_neighbor_score = _undirected_adjacency(
        node_count, edge_first, edge_second, edge_score
    )
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_offset, crossing_neighbor, _ = _undirected_adjacency(
        node_count, crossing_first, crossing_second
    )
    node_unary = np.asarray(configuration["nodeUnaryScore"], dtype=np.float32)
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    selected_model = np.asarray(scored["corridorSelectedModel"], dtype=np.int32)
    profile_correlation = np.asarray(
        scored["corridorZeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    trace_correlation = np.asarray(
        scored["corridorBoundaryTraceCorrelation"], dtype=np.float32
    )
    shift_margin = np.asarray(
        scored["corridorZeroShiftCompetingMargin"], dtype=np.float32
    )
    evidence_eligible = np.asarray(
        [
            profile_correlation[row, model] >= settings.minimum_context_profile_correlation
            and trace_correlation[row, model]
            >= settings.minimum_boundary_trace_correlation
            and shift_margin[row, model] >= settings.minimum_zero_shift_margin
            for row, model in enumerate(selected_model)
        ],
        dtype=np.uint8,
    )
    boundary_first_node = np.asarray(
        corridors["boundaryEdgeFirstFrontierIndex"], dtype=np.int32
    )
    boundary_second_node = np.asarray(
        corridors["boundaryEdgeSecondFrontierIndex"], dtype=np.int32
    )
    first_boundary_edge = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )
    second_boundary_edge = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    patch_offset = np.asarray(scored["corridorPatchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(scored["corridorPatchXYZ"], dtype=np.float32)
    edge_length = np.asarray(
        corridors["boundaryEdgeLengthVoxels"], dtype=np.float32
    )

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
    added_offset = [0]
    added_values: list[int] = []
    removed_offset = [0]
    removed_values: list[int] = []
    baseline_objective: list[float] = []
    proposal_objective: list[float] = []
    objective_delta: list[float] = []
    proposal_coverage: list[float] = []
    retained_boundary_fraction: list[float] = []
    boundary_anchor_count: list[int] = []
    hard_conflict_count: list[int] = []
    beam_records: list[dict[str, Any]] = []
    for scored_row, corridor_index in enumerate(scored_corridor):
        pair_start, pair_stop = (
            int(pair_offset[corridor_index]),
            int(pair_offset[corridor_index + 1]),
        )
        corridor_boundary_edge = np.concatenate(
            (
                first_boundary_edge[pair_start:pair_stop],
                second_boundary_edge[pair_start:pair_stop],
            )
        )
        boundary = np.unique(
            np.concatenate(
                (
                    boundary_first_node[corridor_boundary_edge],
                    boundary_second_node[corridor_boundary_edge],
                )
            )
        )
        if not evidence_eligible[scored_row]:
            candidate_offset.append(len(candidate_values))
            option_offset.append(len(option_values))
            added_offset.append(len(added_values))
            removed_offset.append(len(removed_values))
            baseline_objective.append(0.0)
            proposal_objective.append(0.0)
            objective_delta.append(0.0)
            proposal_coverage.append(0.0)
            retained_boundary_fraction.append(1.0)
            boundary_anchor_count.append(0)
            hard_conflict_count.append(0)
            beam_records.append({"skippedByEvidenceGate": True})
            continue
        geometric = _corridor_geometric_candidates(
            scored_row,
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
        for candidate_value in candidate:
            start, stop = (
                int(crossing_offset[candidate_value]),
                int(crossing_offset[candidate_value + 1]),
            )
            incumbent.update(
                int(value)
                for value in crossing_neighbor[start:stop]
                if selected[value]
            )
        options = np.asarray(sorted(incumbent) + sorted(candidate_set), dtype=np.int32)
        local_index = {int(value): index for index, value in enumerate(options)}
        candidate_lookup = {int(value): index for index, value in enumerate(candidate)}
        node_weight = node_unary[options].astype(np.float64)
        pair_support: dict[tuple[int, int], float] = defaultdict(float)
        fixed_selected = selected.copy()
        if incumbent:
            fixed_selected[np.asarray(sorted(incumbent), dtype=np.int32)] = False
        for option_index, option_value in enumerate(options):
            start, stop = int(edge_offset[option_value]), int(edge_offset[option_value + 1])
            for neighbor, weight in zip(
                edge_neighbor[start:stop], edge_neighbor_score[start:stop]
            ):
                neighbor_local = local_index.get(int(neighbor))
                if neighbor_local is not None:
                    if option_index < neighbor_local:
                        pair_support[(option_index, neighbor_local)] += (
                            continuity_weight * float(weight)
                        )
                elif fixed_selected[neighbor]:
                    node_weight[option_index] += continuity_weight * float(weight)
            candidate_index = candidate_lookup.get(int(option_value))
            if candidate_index is not None:
                node_weight[option_index] += (
                    settings.surface_alignment_weight
                    * float(geometric["surfaceAlignment"][candidate_index])
                )
        conflict_mask = [0 for _ in range(len(options))]
        interface_group: dict[int, list[int]] = defaultdict(list)
        for option_index, option_value in enumerate(options):
            interface_group[int(source_interface[option_value])].append(option_index)
            interface_group[int(target_interface[option_value])].append(option_index)
        for group in interface_group.values():
            for first_index, left in enumerate(group):
                for right in group[first_index + 1 :]:
                    conflict_mask[left] |= 1 << right
                    conflict_mask[right] |= 1 << left
        for option_index, option_value in enumerate(options):
            start, stop = (
                int(crossing_offset[option_value]),
                int(crossing_offset[option_value + 1]),
            )
            for neighbor in crossing_neighbor[start:stop]:
                neighbor_local = local_index.get(int(neighbor))
                if neighbor_local is not None:
                    conflict_mask[option_index] |= 1 << neighbor_local
        pair_items = sorted(pair_support.items())
        pair_first = np.asarray([value[0][0] for value in pair_items], dtype=np.int32)
        pair_second = np.asarray([value[0][1] for value in pair_items], dtype=np.int32)
        pair_weight = np.asarray([value[1] for value in pair_items], dtype=np.float32)
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
        proposal_score, proposal_mask = next(
            (
                (score, mask)
                for score, mask in states
                if any(mask & (1 << local_index[value]) for value in candidate_set)
            ),
            (
                _objective_for_mask(
                    baseline_mask,
                    node_weight,
                    pair_first,
                    pair_second,
                    pair_weight,
                ),
                baseline_mask,
            ),
        )
        baseline_score = _objective_for_mask(
            baseline_mask, node_weight, pair_first, pair_second, pair_weight
        )
        proposed = {
            int(options[index])
            for index in range(len(options))
            if proposal_mask & (1 << index)
        }
        added = sorted(proposed & candidate_set)
        removed = sorted(incumbent - proposed)
        boundary_set = set(int(value) for value in boundary)
        anchors: set[int] = set()
        for value in added:
            start, stop = int(edge_offset[value]), int(edge_offset[value + 1])
            anchors.update(
                int(neighbor)
                for neighbor in edge_neighbor[start:stop]
                if int(neighbor) in boundary_set
            )
        patch_start, patch_stop = (
            int(patch_offset[scored_row]),
            int(patch_offset[scored_row + 1]),
        )
        patch = patch_xyz[patch_start:patch_stop]
        if added:
            added_candidate_index = np.asarray(
                [candidate_lookup[value] for value in added], dtype=np.int32
            )
            projected = patch[
                np.asarray(geometric["nearestPatchPixel"], dtype=np.int32)[
                    added_candidate_index
                ]
            ]
            coverage_radius = float(np.median(edge_length[corridor_boundary_edge]))
            coverage = float(
                np.mean(
                    np.any(
                        np.linalg.norm(
                            patch[:, None, :] - projected[None, :, :], axis=2
                        )
                        <= coverage_radius,
                        axis=1,
                    )
                )
            )
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
        added_values.extend(added)
        added_offset.append(len(added_values))
        removed_values.extend(removed)
        removed_offset.append(len(removed_values))
        baseline_objective.append(baseline_score)
        proposal_objective.append(float(proposal_score))
        objective_delta.append(float(proposal_score - baseline_score))
        proposal_coverage.append(coverage)
        retained_boundary_fraction.append(
            sum(int(value) in proposed for value in boundary) / max(len(boundary), 1)
        )
        boundary_anchor_count.append(len(anchors))
        hard_conflict_count.append(sum(value.bit_count() for value in conflict_mask) // 2)
        beam_records.append(beam_stats)
    arrays = {
        "corridorEvidenceEligible": evidence_eligible,
        "corridorCandidateOffset": np.asarray(candidate_offset, dtype=np.int64),
        "corridorCandidateFrontierIndex": np.asarray(candidate_values, dtype=np.int32),
        "corridorCandidateNearestPatchPixel": np.asarray(candidate_nearest, dtype=np.int32),
        "corridorCandidateHeightResidualVoxels": np.asarray(candidate_height, dtype=np.float32),
        "corridorCandidateTangentResidualVoxels": np.asarray(candidate_tangent, dtype=np.float32),
        "corridorCandidateNormalResidualDegrees": np.asarray(candidate_normal, dtype=np.float32),
        "corridorCandidateThicknessRatio": np.asarray(candidate_thickness_ratio, dtype=np.float32),
        "corridorCandidateSurfaceAlignment": np.asarray(candidate_alignment, dtype=np.float32),
        "corridorOptionOffset": np.asarray(option_offset, dtype=np.int64),
        "corridorOptionFrontierIndex": np.asarray(option_values, dtype=np.int32),
        "corridorOptionWasSelected": np.asarray(option_was_selected, dtype=np.uint8),
        "corridorOptionIsPatchCandidate": np.asarray(option_is_candidate, dtype=np.uint8),
        "corridorOptionNodeWeight": np.asarray(option_weight_values, dtype=np.float32),
        "corridorProposalAddedOffset": np.asarray(added_offset, dtype=np.int64),
        "corridorProposalAddedFrontierIndex": np.asarray(added_values, dtype=np.int32),
        "corridorProposalRemovedOffset": np.asarray(removed_offset, dtype=np.int64),
        "corridorProposalRemovedFrontierIndex": np.asarray(removed_values, dtype=np.int32),
        "corridorBaselineLocalObjective": np.asarray(baseline_objective, dtype=np.float32),
        "corridorProposalLocalObjective": np.asarray(proposal_objective, dtype=np.float32),
        "corridorProposalObjectiveDelta": np.asarray(objective_delta, dtype=np.float32),
        "corridorProposalPatchCoverage": np.asarray(proposal_coverage, dtype=np.float32),
        "corridorProposalRetainedBoundaryFraction": np.asarray(
            retained_boundary_fraction, dtype=np.float32
        ),
        "corridorProposalBoundaryAnchorCount": np.asarray(
            boundary_anchor_count, dtype=np.int32
        ),
        "corridorReconfigurationHardConflictCount": np.asarray(
            hard_conflict_count, dtype=np.int32
        ),
    }
    return arrays, {
        "corridorCount": len(scored_corridor),
        "evidenceEligibleCorridorCount": int(np.count_nonzero(evidence_eligible)),
        "geometricCandidateCount": len(candidate_values),
        "proposalAddedRibbonCount": len(added_values),
        "proposalRemovedRibbonCount": len(removed_values),
        "positiveObjectiveProposalCount": int(
            np.count_nonzero(np.asarray(objective_delta) > 0.0)
        ),
        "beam": beam_records,
        "factorGraph": "complete alternating interface re-pairings with strict continuation factors and exact physical conflicts",
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def _triangle_region_labels(triangles: np.ndarray) -> np.ndarray:
    parent = np.arange(len(triangles), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    edge_triangle: dict[tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, first_value in enumerate(triangle):
            second_value = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first_value), second_value), max(int(first_value), second_value))
            previous = edge_triangle.get(edge)
            if previous is None:
                edge_triangle[edge] = triangle_index
                continue
            first_root, second_root = find(previous), find(triangle_index)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(first_root, second_root)
    root = np.asarray([find(value) for value in range(len(triangles))], dtype=np.int32)
    _, label = np.unique(root, return_inverse=True)
    return label.astype(np.int32)


def _evaluate_corridor_connections(
    replay_surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    *,
    minimum_arc_region_fraction: float = 0.50,
    maximum_arc_triangle_distance_edges: float = 2.0,
) -> dict[str, np.ndarray]:
    triangles = np.asarray(replay_surface["triangleFrontierIndex"], dtype=np.int32)
    triangle_region = _triangle_region_labels(triangles)
    center = np.asarray(replay_surface["midpointXYZ"], dtype=np.float32)
    triangle_center = np.mean(center[triangles], axis=1)
    surface_normal = replay_surface.get("signedNormalXYZ")
    if surface_normal is not None:
        triangle_normal = _normalize_rows(
            np.mean(np.asarray(surface_normal, dtype=np.float32)[triangles], axis=1)
        )
    else:
        triangle_normal = None
    edge_region: dict[tuple[int, int], set[int]] = defaultdict(set)
    for triangle_index, triangle in enumerate(triangles):
        region_id = int(triangle_region[triangle_index])
        for edge_index, first_node in enumerate(triangle):
            second_node = int(triangle[(edge_index + 1) % 3])
            edge_region[
                (min(int(first_node), second_node), max(int(first_node), second_node))
            ].add(region_id)
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    first_boundary_edge = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )
    second_boundary_edge = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )
    boundary_first_node = np.asarray(
        corridors["boundaryEdgeFirstFrontierIndex"], dtype=np.int32
    )
    boundary_second_node = np.asarray(
        corridors["boundaryEdgeSecondFrontierIndex"], dtype=np.int32
    )
    boundary_xyz = np.asarray(
        corridors["boundaryEdgeMidpointXYZ"], dtype=np.float32
    )
    boundary_normal = np.asarray(
        corridors["boundaryEdgeNormalXYZ"], dtype=np.float32
    )
    boundary_edge_length = np.asarray(
        corridors["boundaryEdgeLengthVoxels"], dtype=np.float32
    )
    baseline_edge_region = np.asarray(
        corridors["boundaryEdgeTriangleRegion"], dtype=np.int32
    )
    patch_offset = np.asarray(scored["corridorPatchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(scored["corridorPatchXYZ"], dtype=np.float32)
    connected = np.zeros(len(scored_corridor), dtype=np.uint8)
    connecting_region = np.full(len(scored_corridor), -1, dtype=np.int32)
    shared_region_fraction = np.zeros(len(scored_corridor), dtype=np.float32)
    baseline_distinct = np.zeros(len(scored_corridor), dtype=np.uint8)
    patch_triangle_coverage = np.zeros(len(scored_corridor), dtype=np.float32)
    for row, corridor_index in enumerate(scored_corridor):
        start, stop = (
            int(pair_offset[corridor_index]),
            int(pair_offset[corridor_index + 1]),
        )
        first_arc_edge = first_boundary_edge[start:stop]
        second_arc_edge = second_boundary_edge[start:stop]
        first_arc = boundary_xyz[first_arc_edge]
        second_arc = boundary_xyz[second_arc_edge]
        baseline_first = set(int(value) for value in baseline_edge_region[first_arc_edge])
        baseline_second = set(int(value) for value in baseline_edge_region[second_arc_edge])
        baseline_distinct[row] = int(not (baseline_first & baseline_second))
        first_region_count: dict[int, int] = defaultdict(int)
        second_region_count: dict[int, int] = defaultdict(int)

        def replay_regions(edge_index: int) -> set[int]:
            key = (
                min(
                    int(boundary_first_node[edge_index]),
                    int(boundary_second_node[edge_index]),
                ),
                max(
                    int(boundary_first_node[edge_index]),
                    int(boundary_second_node[edge_index]),
                ),
            )
            exact = edge_region.get(key)
            if exact:
                return exact
            distance = np.linalg.norm(
                triangle_center - boundary_xyz[edge_index], axis=1
            )
            valid = distance <= (
                maximum_arc_triangle_distance_edges
                * float(boundary_edge_length[edge_index])
            )
            if triangle_normal is not None:
                valid &= np.abs(
                    triangle_normal @ boundary_normal[edge_index]
                ) >= math.cos(math.radians(60.0))
            candidate = np.flatnonzero(valid)
            if not len(candidate):
                return set()
            nearest_triangle = int(candidate[int(np.argmin(distance[candidate]))])
            return {int(triangle_region[nearest_triangle])}

        for edge_index in first_arc_edge:
            for region_id in replay_regions(int(edge_index)):
                first_region_count[region_id] += 1
        for edge_index in second_arc_edge:
            for region_id in replay_regions(int(edge_index)):
                second_region_count[region_id] += 1
        common_region = sorted(
            set(first_region_count) & set(second_region_count),
            key=lambda region_id: (
                -min(first_region_count[region_id], second_region_count[region_id]),
                region_id,
            ),
        )
        if common_region:
            best_region = int(common_region[0])
            fraction = min(
                first_region_count[best_region] / max(len(first_arc_edge), 1),
                second_region_count[best_region] / max(len(second_arc_edge), 1),
            )
            shared_region_fraction[row] = fraction
        if (
            baseline_distinct[row]
            and common_region
            and shared_region_fraction[row] >= minimum_arc_region_fraction
        ):
            connected[row] = 1
            connecting_region[row] = int(common_region[0])
        scale = 1.5 * float(
            np.median(
                boundary_edge_length[
                    np.concatenate((first_arc_edge, second_arc_edge))
                ]
            )
        )
        local_low = (
            np.minimum(np.min(first_arc, axis=0), np.min(second_arc, axis=0))
            - scale
        )
        local_high = (
            np.maximum(np.max(first_arc, axis=0), np.max(second_arc, axis=0))
            + scale
        )
        local_triangle = np.flatnonzero(
            np.all(
                (triangle_center >= local_low) & (triangle_center <= local_high),
                axis=1,
            )
        )
        patch_start, patch_stop = (
            int(patch_offset[row]),
            int(patch_offset[row + 1]),
        )
        patch = patch_xyz[patch_start:patch_stop]
        if len(local_triangle):
            nearest = np.min(
                np.linalg.norm(
                    patch[:, None, :] - triangle_center[local_triangle][None, :, :],
                    axis=2,
                ),
                axis=1,
            )
            patch_triangle_coverage[row] = float(np.mean(nearest <= scale))
    return {
        "triangleFrontierIndex": triangles,
        "triangleRegion": triangle_region,
        "boundaryArcsConnected": connected,
        "connectingTriangleRegion": connecting_region,
        "boundaryArcSharedRegionFraction": shared_region_fraction,
        "baselineBoundaryArcsDistinct": baseline_distinct,
        "patchTriangleCoverage": patch_triangle_coverage,
    }


def _select_density_preserving_corridor_subsets(
    preliminary: np.ndarray,
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    reconfiguration: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Choose exact local proposal subsets without regressing sheet density."""

    retained = np.zeros(len(preliminary), dtype=np.uint8)
    rejected = np.zeros(len(preliminary), dtype=np.uint8)
    candidate_row = np.flatnonzero(preliminary)
    if not len(candidate_row):
        return retained, rejected, []
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    corridor_component = np.asarray(
        corridors["corridorTopologyComponent"], dtype=np.int32
    )
    rows_by_component: dict[int, list[int]] = defaultdict(list)
    for row in candidate_row:
        rows_by_component[
            int(corridor_component[int(scored_corridor[row])])
        ].append(int(row))
    added_offset = np.asarray(
        reconfiguration["corridorProposalAddedOffset"], dtype=np.int64
    )
    added_value = np.asarray(
        reconfiguration["corridorProposalAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        reconfiguration["corridorProposalRemovedOffset"], dtype=np.int64
    )
    removed_value = np.asarray(
        reconfiguration["corridorProposalRemovedFrontierIndex"], dtype=np.int32
    )
    objective_delta = np.asarray(
        reconfiguration["corridorProposalObjectiveDelta"], dtype=np.float32
    )
    baseline_selected = np.asarray(configuration["selected"]) > 0
    original_component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source_interface = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target_interface = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    baseline_triangle = np.asarray(
        surface["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_area = np.asarray(
        surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    baseline_region = _triangle_region_labels(baseline_triangle)
    records: list[dict[str, Any]] = []

    def has_global_conflict(selected: np.ndarray) -> bool:
        node = np.flatnonzero(selected)
        interface = np.concatenate(
            (source_interface[node], target_interface[node])
        )
        return bool(
            len(interface) != len(np.unique(interface))
            or np.any(selected[crossing_first] & selected[crossing_second])
        )

    for component_id, rows in sorted(rows_by_component.items()):
        baseline_triangle_index = np.flatnonzero(
            np.all(original_component[baseline_triangle] == component_id, axis=1)
        )
        baseline_region_count = len(
            np.unique(baseline_region[baseline_triangle_index])
        )
        baseline_area_value = float(np.sum(baseline_area[baseline_triangle_index]))
        best_key: tuple[float, ...] = (
            0.0,
            0.0,
            0.0,
            0.0,
        )
        best_rows: tuple[int, ...] = ()
        evaluated = 0
        feasible = 1
        for mask in range(1, 1 << len(rows)):
            subset = tuple(
                rows[index]
                for index in range(len(rows))
                if mask & (1 << index)
            )
            selected = baseline_selected.copy()
            additions: list[int] = []
            for row in subset:
                added = added_value[
                    int(added_offset[row]) : int(added_offset[row + 1])
                ]
                removed = removed_value[
                    int(removed_offset[row]) : int(removed_offset[row + 1])
                ]
                selected[removed] = False
                selected[added] = True
                additions.extend(int(value) for value in added)
            if has_global_conflict(selected):
                continue
            local_selected = selected & (original_component == component_id)
            if additions:
                local_selected[np.asarray(additions, dtype=np.int32)] = selected[
                    np.asarray(additions, dtype=np.int32)
                ]
            local_component, _ = _component_labels(
                local_selected, first, second
            )
            retained_original = local_selected & (
                original_component == component_id
            )
            retained_label = np.unique(
                local_component[retained_original][
                    local_component[retained_original] >= 0
                ]
            )
            if len(retained_label) != 1:
                continue
            local_configuration = dict(configuration)
            local_configuration["selected"] = local_selected.astype(np.uint8)
            local_configuration["component"] = local_component
            local_surface, _ = build_physical_ribbon_surface_complex(
                ribbon,
                topology,
                local_configuration,
                settings=settings.surface_settings(),
            )
            local_triangle = np.asarray(
                local_surface["triangleFrontierIndex"], dtype=np.int32
            )
            local_region_count = (
                len(np.unique(_triangle_region_labels(local_triangle)))
                if len(local_triangle)
                else 0
            )
            local_area_value = float(
                np.sum(local_surface["triangleAreaVoxelsSquared"])
            )
            evaluated += 1
            local_connections = _evaluate_corridor_connections(
                local_surface,
                corridors,
                scored,
                minimum_arc_region_fraction=(
                    settings.minimum_replay_arc_region_fraction
                ),
                maximum_arc_triangle_distance_edges=(
                    settings.maximum_replay_arc_triangle_distance_edges
                ),
            )
            connected = np.asarray(
                local_connections["boundaryArcsConnected"]
            ) > 0
            if not all(connected[row] for row in subset):
                continue
            area_retention = local_area_value / max(baseline_area_value, 1.0e-6)
            if (
                local_region_count > baseline_region_count
                or area_retention < settings.minimum_replay_surface_area_retention
            ):
                continue
            feasible += 1
            key = (
                float(baseline_region_count - local_region_count),
                local_area_value - baseline_area_value,
                float(sum(objective_delta[row] for row in subset)),
                float(len(subset)),
            )
            if key > best_key:
                best_key = key
                best_rows = subset
        retained[np.asarray(best_rows, dtype=np.int32)] = 1
        rejected[np.asarray(rows, dtype=np.int32)] = 1
        rejected[np.asarray(best_rows, dtype=np.int32)] = 0
        records.append(
            {
                "componentId": component_id,
                "candidateRows": rows,
                "evaluatedNonemptySubsets": evaluated,
                "feasibleSubsetCountIncludingEmpty": feasible,
                "selectedRows": list(best_rows),
                "baselineTriangleRegionCount": baseline_region_count,
                "baselineTriangleAreaVoxelsSquared": round(
                    baseline_area_value, 4
                ),
                "selectionKey": [round(value, 6) for value in best_key],
            }
        )
    return retained, rejected, records


def replay_patch_corridor_reconfigurations(
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    reconfiguration: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchCorridorSettings,
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
    added_offset = np.asarray(
        reconfiguration["corridorProposalAddedOffset"], dtype=np.int64
    )
    added_values = np.asarray(
        reconfiguration["corridorProposalAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        reconfiguration["corridorProposalRemovedOffset"], dtype=np.int64
    )
    removed_values = np.asarray(
        reconfiguration["corridorProposalRemovedFrontierIndex"], dtype=np.int32
    )
    objective_delta = np.asarray(
        reconfiguration["corridorProposalObjectiveDelta"], dtype=np.float32
    )
    evidence_eligible = np.asarray(
        reconfiguration["corridorEvidenceEligible"]
    ) > 0
    applied = np.zeros(len(objective_delta), dtype=np.uint8)
    split_rejected = np.zeros(len(objective_delta), dtype=np.uint8)
    original_component = np.asarray(configuration["component"], dtype=np.int32)

    def conflict_counts(value: np.ndarray) -> tuple[int, int]:
        nodes = np.flatnonzero(value)
        interfaces = np.concatenate((source[nodes], target[nodes]))
        interface_conflicts = len(interfaces) - len(np.unique(interfaces))
        crossing_conflicts = int(
            np.count_nonzero(value[crossing_first] & value[crossing_second])
        )
        return interface_conflicts, crossing_conflicts

    for row in np.argsort(-objective_delta, kind="stable"):
        if not evidence_eligible[row] or objective_delta[row] <= 0.0:
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
        trial_component, _ = _component_labels(
            trial,
            np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32),
            np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32),
        )
        fragmented = False
        for old_component_id in np.unique(
            original_component[removed][original_component[removed] >= 0]
        ):
            retained = trial & (original_component == old_component_id)
            if len(np.unique(trial_component[retained])) > 1:
                fragmented = True
                break
        if fragmented:
            split_rejected[row] = 1
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
        settings=settings.surface_settings(),
    )
    replay_loops, loop_stats = extract_surface_boundary_loops(
        replay_surface, settings=settings.surface_settings()
    )
    trial_surface_stats = surface_stats
    trial_loop_stats = loop_stats
    trial_metrics = _evaluate_corridor_connections(
        replay_surface,
        corridors,
        scored,
        minimum_arc_region_fraction=settings.minimum_replay_arc_region_fraction,
        maximum_arc_triangle_distance_edges=(
            settings.maximum_replay_arc_triangle_distance_edges
        ),
    )
    trial_connected = np.asarray(trial_metrics["boundaryArcsConnected"]) > 0
    preliminary_success = (applied > 0) & trial_connected
    failed_replay = (applied > 0) & ~trial_connected
    (
        density_success,
        rejected_surface_regression,
        density_selection_records,
    ) = _select_density_preserving_corridor_subsets(
        preliminary_success,
        surface,
        corridors,
        scored,
        reconfiguration,
        ribbon,
        topology,
        configuration,
        settings=settings,
    )
    successful = density_success > 0
    additional_rebuild = bool(np.any((applied > 0) & ~successful))
    if additional_rebuild:
        selected = np.asarray(configuration["selected"]).astype(bool).copy()
        for row in np.flatnonzero(successful):
            added = added_values[
                int(added_offset[row]) : int(added_offset[row + 1])
            ]
            removed = removed_values[
                int(removed_offset[row]) : int(removed_offset[row + 1])
            ]
            selected[removed] = False
            selected[added] = True
        interface_conflicts, crossing_conflicts = conflict_counts(selected)
        component, component_size = _component_labels(selected, first, second)
        replay_configuration = dict(configuration)
        replay_configuration["selected"] = selected.astype(np.uint8)
        replay_configuration["component"] = component
        replay_surface, surface_stats = build_physical_ribbon_surface_complex(
            ribbon,
            topology,
            replay_configuration,
            settings=settings.surface_settings(),
        )
        replay_loops, loop_stats = extract_surface_boundary_loops(
            replay_surface, settings=settings.surface_settings()
        )
    final_metrics = _evaluate_corridor_connections(
        replay_surface,
        corridors,
        scored,
        minimum_arc_region_fraction=settings.minimum_replay_arc_region_fraction,
        maximum_arc_triangle_distance_edges=(
            settings.maximum_replay_arc_triangle_distance_edges
        ),
    )
    final_connected = np.asarray(final_metrics["boundaryArcsConnected"]) > 0
    successful &= final_connected
    old_component = original_component
    cross_component_fusions = 0
    maximum_old_components = 0
    for component_id in np.unique(component[selected]):
        nodes = np.flatnonzero(component == component_id)
        inherited = np.unique(old_component[nodes][old_component[nodes] >= 0])
        cross_component_fusions += int(len(inherited) > 1)
        maximum_old_components = max(maximum_old_components, len(inherited))
    triangles = np.asarray(final_metrics["triangleFrontierIndex"], dtype=np.int32)
    triangle_region = np.asarray(final_metrics["triangleRegion"], dtype=np.int32)
    connected = np.asarray(final_metrics["boundaryArcsConnected"], dtype=np.uint8)
    connecting_region = np.asarray(
        final_metrics["connectingTriangleRegion"], dtype=np.int32
    )
    patch_triangle_coverage = np.asarray(
        final_metrics["patchTriangleCoverage"], dtype=np.float32
    )
    shared_region_fraction = np.asarray(
        final_metrics["boundaryArcSharedRegionFraction"], dtype=np.float32
    )
    baseline_distinct = np.asarray(
        final_metrics["baselineBoundaryArcsDistinct"], dtype=np.uint8
    )
    arrays = {
        "corridorReplayProposalTrialApplied": applied,
        "corridorReplayProposalSuccessful": successful.astype(np.uint8),
        "corridorReplayProposalRejectedNoConnection": failed_replay.astype(
            np.uint8
        ),
        "corridorReplayProposalRejectedSurfaceRegression": (
            rejected_surface_regression
        ),
        "corridorReplayProposalRejectedComponentSplit": split_rejected,
        "corridorReplaySelected": selected.astype(np.uint8),
        "corridorReplayComponent": component,
        "corridorReplayComponentSize": component_size.astype(np.int32),
        "corridorReplayChartUV": np.asarray(
            replay_surface["chartUV"], dtype=np.float32
        ),
        "corridorReplayTriangleFrontierIndex": triangles,
        "corridorReplayTriangleAreaVoxelsSquared": np.asarray(
            replay_surface["triangleAreaVoxelsSquared"], dtype=np.float32
        ),
        "corridorReplayTriangleRegion": triangle_region,
        "corridorReplayBoundaryArcsConnected": connected,
        "corridorReplayConnectingTriangleRegion": connecting_region,
        "corridorReplayBoundaryArcSharedRegionFraction": shared_region_fraction,
        "corridorReplayBaselineBoundaryArcsDistinct": baseline_distinct,
        "corridorReplayPatchTriangleCoverage": patch_triangle_coverage,
        "corridorReplayLoopOffset": np.asarray(
            replay_loops["loopOffset"], dtype=np.int64
        ),
        "corridorReplayLoopVertexFrontierIndex": np.asarray(
            replay_loops["loopVertexFrontierIndex"], dtype=np.int32
        ),
        "corridorReplayLoopKind": np.asarray(
            replay_loops["loopKind"], dtype=np.uint8
        ),
        "corridorReplayLoopTriangleRegion": np.asarray(
            replay_loops["loopTriangleRegion"], dtype=np.int32
        ),
    }
    return arrays, {
        "candidateProposalCount": len(applied),
        "trialAppliedPositiveEvidenceProposalCount": int(np.count_nonzero(applied)),
        "componentSplitRejectedProposalCount": int(
            np.count_nonzero(split_rejected)
        ),
        "replaySuccessfulCorridorCount": int(np.count_nonzero(successful)),
        "replayRejectedNoConnectionCount": int(
            np.count_nonzero(failed_replay)
        ),
        "replayRejectedSurfaceRegressionCount": int(
            np.count_nonzero(rejected_surface_regression)
        ),
        "additionalSuccessOnlySurfaceRebuild": additional_rebuild,
        "selectedRibbonCountBefore": int(np.count_nonzero(configuration["selected"])),
        "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
        "interfaceConflictCount": interface_conflicts,
        "crossingConflictCount": crossing_conflicts,
        "componentCountAfter": len(component_size),
        "largestComponentRibbonCountsAfter": [int(value) for value in component_size[:32]],
        "crossPriorComponentFusionCount": cross_component_fusions,
        "maximumPriorComponentsPerReplayComponent": maximum_old_components,
        "surface": surface_stats,
        "loops": loop_stats,
        "trialSurface": trial_surface_stats,
        "trialLoops": trial_loop_stats,
        "densityPreservingSubsetSelection": density_selection_records,
        "selectionMutated": False,
        "replayMeaning": "counterfactual full topology and triangle rebuild only",
        "identityLabelsUsed": False,
    }


def write_patch_corridor_montage(
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    maximum_corridors: int,
    reconfiguration: Mapping[str, np.ndarray] | None = None,
    replay: Mapping[str, np.ndarray] | None = None,
) -> Path:
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    selected_model = np.asarray(scored["corridorSelectedModel"], dtype=np.int32)
    rank = np.asarray(scored["corridorModelRankScore"], dtype=np.float32)
    selected_rank = rank[np.arange(len(scored_corridor)), selected_model]
    order = np.argsort(-selected_rank, kind="stable")[:maximum_corridors]
    columns = 4
    tile_width, tile_height = 320, 230
    rows = max(int(math.ceil(len(order) / columns)), 1)
    canvas = np.full(
        (rows * tile_height, columns * tile_width, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    image_offset = np.asarray(scored["corridorImageOffset"], dtype=np.int64)
    image_value = np.asarray(scored["corridorFlattenedCt"], dtype=np.float32)
    image_rows = np.asarray(scored["corridorImageRows"], dtype=np.int32)
    image_columns = np.asarray(scored["corridorImageColumns"], dtype=np.int32)
    first_boundary_column = np.asarray(
        scored["corridorImageFirstBoundaryColumn"], dtype=np.int32
    )
    second_boundary_column = np.asarray(
        scored["corridorImageSecondBoundaryColumn"], dtype=np.int32
    )
    shifts = np.asarray(
        scored["corridorCompetingShiftThicknesses"], dtype=np.float32
    )
    scores = np.asarray(scored["corridorModelShiftPhysicalScore"], dtype=np.float32)
    profile_correlation = np.asarray(
        scored["corridorZeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    trace_correlation = np.asarray(
        scored["corridorBoundaryTraceCorrelation"], dtype=np.float32
    )
    margin = np.asarray(
        scored["corridorZeroShiftCompetingMargin"], dtype=np.float32
    )
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    evidence_eligible = (
        np.asarray(reconfiguration["corridorEvidenceEligible"]) > 0
        if reconfiguration is not None
        else np.ones(len(scored_corridor), dtype=bool)
    )
    successful = (
        np.asarray(replay["corridorReplayProposalSuccessful"]) > 0
        if replay is not None
        else np.zeros(len(scored_corridor), dtype=bool)
    )
    rejected_no_connection = (
        np.asarray(replay["corridorReplayProposalRejectedNoConnection"]) > 0
        if replay is not None
        else np.zeros(len(scored_corridor), dtype=bool)
    )
    rejected_component_split = (
        np.asarray(replay["corridorReplayProposalRejectedComponentSplit"]) > 0
        if replay is not None
        else np.zeros(len(scored_corridor), dtype=bool)
    )
    rejected_surface_regression = (
        np.asarray(replay["corridorReplayProposalRejectedSurfaceRegression"]) > 0
        if replay is not None
        else np.zeros(len(scored_corridor), dtype=bool)
    )
    for display_index, scored_row in enumerate(order):
        corridor_index = int(scored_corridor[scored_row])
        model_index = int(selected_model[scored_row])
        tile_x = (display_index % columns) * tile_width
        tile_y = (display_index // columns) * tile_height
        pair_count = int(
            pair_offset[corridor_index + 1] - pair_offset[corridor_index]
        )
        component = int(corridors["corridorTopologyComponent"][corridor_index])
        _draw_text(
            canvas,
            tile_x + 8,
            tile_y + 8,
            f"C {component} N {pair_count} R {profile_correlation[scored_row, model_index]:.2f} D {margin[scored_row, model_index]:.2f}",
            (224, 231, 239),
            scale=1,
        )
        if successful[scored_row]:
            border = (65, 210, 133)
        elif rejected_surface_regression[scored_row]:
            border = (181, 102, 218)
        elif rejected_component_split[scored_row]:
            border = (244, 103, 95)
        elif rejected_no_connection[scored_row]:
            border = (245, 166, 73)
        elif not evidence_eligible[scored_row]:
            border = (86, 96, 110)
        else:
            border = (95, 145, 210)
        canvas[tile_y : tile_y + 3, tile_x : tile_x + tile_width] = border
        canvas[
            tile_y + tile_height - 3 : tile_y + tile_height,
            tile_x : tile_x + tile_width,
        ] = border
        canvas[tile_y : tile_y + tile_height, tile_x : tile_x + 3] = border
        canvas[
            tile_y : tile_y + tile_height,
            tile_x + tile_width - 3 : tile_x + tile_width,
        ] = border
        start, stop = (
            int(image_offset[scored_row]),
            int(image_offset[scored_row + 1]),
        )
        image = image_value[start:stop].reshape(
            (int(image_rows[scored_row]), int(image_columns[scored_row]))
        )
        low, high = np.percentile(image[np.isfinite(image)], (2.0, 98.0))
        normalized = np.clip((image - low) / max(float(high - low), 1.0), 0.0, 1.0)
        grayscale = np.rint(255.0 * normalized).astype(np.uint8)
        target_height, target_width = 180, 196
        row_index = np.minimum(
            (np.arange(target_height) * image.shape[0] / target_height).astype(int),
            image.shape[0] - 1,
        )
        column_index = np.minimum(
            (np.arange(target_width) * image.shape[1] / target_width).astype(int),
            image.shape[1] - 1,
        )
        enlarged = grayscale[row_index[:, None], column_index[None, :]]
        rgb = np.repeat(enlarged[:, :, None], 3, axis=2)
        image_x, image_y = tile_x + 8, tile_y + 34
        canvas[
            image_y : image_y + target_height,
            image_x : image_x + target_width,
        ] = rgb
        first_boundary_x = image_x + int(
            round(
                float(first_boundary_column[scored_row])
                / max(image.shape[1] - 1, 1)
                * (target_width - 1)
            )
        )
        second_boundary_x = image_x + int(
            round(
                float(second_boundary_column[scored_row])
                / max(image.shape[1] - 1, 1)
                * (target_width - 1)
            )
        )
        canvas[
            image_y : image_y + target_height,
            first_boundary_x : first_boundary_x + 2,
        ] = (
            255,
            199,
            62,
        )
        canvas[
            image_y : image_y + target_height,
            second_boundary_x - 1 : second_boundary_x + 1,
        ] = (63, 203, 224)
        plot_x, plot_y = tile_x + 220, tile_y + 54
        plot_width, plot_height = 88, 120
        values = scores[scored_row, model_index]
        low_score = min(float(np.min(values)), -0.05)
        high_score = max(float(np.max(values)), 0.05)
        points: list[np.ndarray] = []
        for shift, value in zip(shifts, values):
            points.append(
                np.asarray(
                    (
                        plot_x
                        + (shift - shifts[0])
                        / max(float(shifts[-1] - shifts[0]), 1.0e-6)
                        * plot_width,
                        plot_y
                        + plot_height
                        - (value - low_score)
                        / max(high_score - low_score, 1.0e-6)
                        * plot_height,
                    ),
                    dtype=np.float32,
                )
            )
        for first, second in zip(points, points[1:]):
            _draw_line(canvas, first, second, (225, 98, 190))
        for point in points:
            x_value, y_value = np.rint(point).astype(int)
            canvas[y_value - 2 : y_value + 3, x_value - 2 : x_value + 3] = (
                225,
                98,
                190,
            )
        _draw_text(
            canvas,
            plot_x,
            tile_y + 188,
            f"R {trace_correlation[scored_row, model_index]:.2f}",
            (224, 231, 239),
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rgb_png(canvas))
    return output


def write_replayed_corridor_fragment_montage(
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    replay: Mapping[str, np.ndarray],
    source: VolumeSource,
    path: str | Path,
    *,
    maximum_components: int = 8,
    pixel_step_voxels: float = 0.5,
    maximum_raster_pixels: int = 768,
) -> tuple[Path, dict[str, Any]]:
    """Flatten complete replay components containing accepted corridor repairs."""

    successful_row = np.flatnonzero(
        np.asarray(replay["corridorReplayProposalSuccessful"]) > 0
    )
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    selected_model = np.asarray(scored["corridorSelectedModel"], dtype=np.int32)
    texture_depth_fraction = np.asarray(
        scored["corridorTextureDepthFraction"], dtype=np.float32
    )
    component = np.asarray(replay["corridorReplayComponent"], dtype=np.int32)
    original_component = np.asarray(surface["component"], dtype=np.int32)
    corridor_component = np.asarray(
        corridors["corridorTopologyComponent"], dtype=np.int32
    )
    triangles = np.asarray(
        replay["corridorReplayTriangleFrontierIndex"], dtype=np.int32
    )
    chart_uv = np.asarray(replay["corridorReplayChartUV"], dtype=np.float32)
    midpoint = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    first_boundary_edge = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )
    second_boundary_edge = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )
    edge_first_node = np.asarray(
        corridors["boundaryEdgeFirstFrontierIndex"], dtype=np.int32
    )
    edge_second_node = np.asarray(
        corridors["boundaryEdgeSecondFrontierIndex"], dtype=np.int32
    )
    rows_by_component: dict[int, list[int]] = defaultdict(list)
    for row in successful_row:
        corridor_index = int(scored_corridor[row])
        prior_component = int(corridor_component[corridor_index])
        inherited_label = component[
            (original_component == prior_component) & (component >= 0)
        ]
        if not len(inherited_label):
            continue
        value, count = np.unique(inherited_label, return_counts=True)
        replay_component = int(value[int(np.argmax(count))])
        if replay_component >= 0:
            rows_by_component[replay_component].append(int(row))
    ranked_components = sorted(
        rows_by_component,
        key=lambda value: (
            -int(np.count_nonzero(component == value)),
            value,
        ),
    )[:maximum_components]
    columns = 2
    tile_width, tile_height = 650, 680
    canvas_rows = max(int(math.ceil(len(ranked_components) / columns)), 1)
    canvas = np.full(
        (canvas_rows * tile_height, columns * tile_width, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    if not ranked_components:
        _draw_text(canvas, 20, 20, "NO REPLAYED CORRIDOR", (224, 231, 239), scale=2)
    records: list[dict[str, Any]] = []
    for display_index, component_id in enumerate(ranked_components):
        component_triangle = triangles[
            np.all(component[triangles] == component_id, axis=1)
        ]
        vertex = np.unique(component_triangle)
        local_triangle = np.searchsorted(vertex, component_triangle).astype(np.int32)
        triangle_normal = _normalize_rows(
            np.mean(normal[component_triangle], axis=1)
        ).astype(np.float32)
        mesh = ComponentMesh(
            component_id=component_id,
            patch_ids=(component_id,),
            vertex_xyz=midpoint[vertex].astype(np.float64),
            polygons=(),
            polygon_patch_ids=np.empty(0, dtype=np.uint64),
            triangles=local_triangle,
            triangle_patch_ids=np.full(
                len(local_triangle), component_id + 1, dtype=np.uint64
            ),
            triangle_normal_xyz=triangle_normal.astype(np.float64),
            statistics={},
        )
        chart = SurfaceChart(
            uv=chart_uv[vertex].astype(np.float64),
            anchor_vertices=(),
            statistics={},
        )
        raster = rasterize_chart(
            mesh,
            chart,
            pixel_step_voxels=pixel_step_voxels,
            maximum_pixels=maximum_raster_pixels,
            padding_pixels=5,
        )
        component_rows = rows_by_component[component_id]
        representative_row = max(
            component_rows,
            key=lambda value: float(
                scored["corridorModelRankScore"][
                    value, selected_model[value]
                ]
            ),
        )
        model_index = int(selected_model[representative_row])
        depth_fraction = float(
            texture_depth_fraction[representative_row, model_index]
        )
        median_thickness = float(np.median(thickness[vertex]))
        depth_offset = depth_fraction * median_thickness
        stack, sampling_stats = sample_depth_stack(
            source, raster, (depth_offset,)
        )
        plane = stack[0].astype(np.float32)
        values = plane[raster.mask]
        low, high = (
            np.percentile(values, (1.0, 99.0))
            if len(values)
            else (0.0, 1.0)
        )
        normalized = np.clip(
            (plane - float(low)) / max(float(high - low), 1.0), 0.0, 1.0
        )
        grayscale = np.rint(12.0 + 243.0 * normalized).astype(np.uint8)
        grayscale[~raster.mask] = 0
        image = np.repeat(grayscale[:, :, None], 3, axis=2)
        image[raster.overlap_mask] = (255, 58, 58)
        chart_low = np.min(chart.uv, axis=0)

        def chart_pixel(global_node: int) -> np.ndarray:
            value = (
                (chart_uv[global_node] - chart_low)
                / raster.pixel_step_voxels
                + 5.0
            )
            return np.asarray((value[0], value[1]), dtype=np.float32)

        for row in component_rows:
            corridor_index = int(scored_corridor[row])
            start, stop = (
                int(pair_offset[corridor_index]),
                int(pair_offset[corridor_index + 1]),
            )
            for edge in first_boundary_edge[start:stop]:
                first_pixel = chart_pixel(int(edge_first_node[edge]))
                second_pixel = chart_pixel(int(edge_second_node[edge]))
                if not np.all(np.isfinite((first_pixel, second_pixel))):
                    continue
                _draw_line(
                    image,
                    first_pixel,
                    second_pixel,
                    (255, 199, 62),
                )
            for edge in second_boundary_edge[start:stop]:
                first_pixel = chart_pixel(int(edge_first_node[edge]))
                second_pixel = chart_pixel(int(edge_second_node[edge]))
                if not np.all(np.isfinite((first_pixel, second_pixel))):
                    continue
                _draw_line(
                    image,
                    first_pixel,
                    second_pixel,
                    (63, 203, 224),
                )
        scale = min(
            620.0 / max(image.shape[1], 1),
            620.0 / max(image.shape[0], 1),
        )
        target_width = max(int(round(image.shape[1] * scale)), 1)
        target_height = max(int(round(image.shape[0] * scale)), 1)
        row_index = np.minimum(
            (np.arange(target_height) * image.shape[0] / target_height).astype(int),
            image.shape[0] - 1,
        )
        column_index = np.minimum(
            (np.arange(target_width) * image.shape[1] / target_width).astype(int),
            image.shape[1] - 1,
        )
        fitted = image[row_index[:, None], column_index[None, :]]
        tile_x = (display_index % columns) * tile_width
        tile_y = (display_index // columns) * tile_height
        image_x = tile_x + (tile_width - target_width) // 2
        image_y = tile_y + 42 + (620 - target_height) // 2
        canvas[
            image_y : image_y + target_height,
            image_x : image_x + target_width,
        ] = fitted
        canvas[tile_y : tile_y + 3, tile_x : tile_x + tile_width] = (
            65,
            210,
            133,
        )
        _draw_text(
            canvas,
            tile_x + 10,
            tile_y + 14,
            f"C {component_id} N {len(vertex)} R {len(component_rows)} D {depth_offset:.1f}",
            (224, 231, 239),
        )
        records.append(
            {
                "componentId": component_id,
                "successfulCorridorRows": component_rows,
                "ribbonCount": len(vertex),
                "triangleCount": len(component_triangle),
                "depthFraction": round(depth_fraction, 4),
                "depthOffsetVoxels": round(depth_offset, 4),
                "raster": raster.statistics,
                "sampling": sampling_stats,
            }
        )
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rgb_png(canvas))
    return output, {
        "successfulCorridorCount": len(successful_row),
        "flattenedReplayComponentCount": len(ranked_components),
        "components": records,
        "yellowAndCyan": "the two original open boundary arcs closed by replay",
        "red": "nonadjacent chart overlap",
    }


def _baseline_cumulative_corridor_replay(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Expose an immutable cumulative baseline through the replay contract.

    The ordinary strict counterfactual replay cannot safely reconstruct and
    preserve previously accepted CT faces.  A cumulative census therefore
    proposes no local mutation.  Downstream whole-strip optimization receives
    every unresolved CT corridor and the exact selected state, then makes the
    first permitted change as a complete multi-cell decision.
    """

    scored_count = len(np.asarray(scored["scoredCorridorIndex"]))
    selected = np.asarray(configuration["selected"], dtype=np.uint8)
    component = np.asarray(configuration["component"], dtype=np.int32)
    selected_component = component[selected > 0]
    component_size = (
        np.bincount(selected_component).astype(np.int32)
        if len(selected_component)
        else np.empty(0, dtype=np.int32)
    )
    connections = _evaluate_corridor_connections(surface, corridors, scored)
    zeros = np.zeros(scored_count, dtype=np.uint8)
    arrays = {
        "corridorReplayProposalTrialApplied": zeros.copy(),
        "corridorReplayProposalSuccessful": zeros.copy(),
        "corridorReplayProposalRejectedNoConnection": zeros.copy(),
        "corridorReplayProposalRejectedSurfaceRegression": zeros.copy(),
        "corridorReplayProposalRejectedComponentSplit": zeros.copy(),
        "corridorReplaySelected": selected.copy(),
        "corridorReplayComponent": component.copy(),
        "corridorReplayComponentSize": component_size,
        "corridorReplayChartUV": np.asarray(
            surface["chartUV"], dtype=np.float32
        ),
        "corridorReplayTriangleFrontierIndex": np.asarray(
            connections["triangleFrontierIndex"], dtype=np.int32
        ),
        "corridorReplayTriangleAreaVoxelsSquared": np.asarray(
            surface["triangleAreaVoxelsSquared"], dtype=np.float32
        ),
        "corridorReplayTriangleRegion": np.asarray(
            connections["triangleRegion"], dtype=np.int32
        ),
        "corridorReplayBoundaryArcsConnected": np.asarray(
            connections["boundaryArcsConnected"], dtype=np.uint8
        ),
        "corridorReplayConnectingTriangleRegion": np.asarray(
            connections["connectingTriangleRegion"], dtype=np.int32
        ),
        "corridorReplayBoundaryArcSharedRegionFraction": np.asarray(
            connections["boundaryArcSharedRegionFraction"], dtype=np.float32
        ),
        "corridorReplayBaselineBoundaryArcsDistinct": np.asarray(
            connections["baselineBoundaryArcsDistinct"], dtype=np.uint8
        ),
        "corridorReplayPatchTriangleCoverage": np.asarray(
            connections["patchTriangleCoverage"], dtype=np.float32
        ),
        "corridorReplayLoopOffset": np.asarray(loops["loopOffset"], dtype=np.int64),
        "corridorReplayLoopVertexFrontierIndex": np.asarray(
            loops["loopVertexFrontierIndex"], dtype=np.int32
        ),
        "corridorReplayLoopKind": np.asarray(loops["loopKind"], dtype=np.uint8),
        "corridorReplayLoopTriangleRegion": np.asarray(
            loops["loopTriangleRegion"], dtype=np.int32
        ),
    }
    return arrays, {
        "candidateProposalCount": scored_count,
        "trialAppliedPositiveEvidenceProposalCount": 0,
        "replaySuccessfulCorridorCount": 0,
        "selectedRibbonCountBefore": int(np.count_nonzero(selected)),
        "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
        "componentCountAfter": len(component_size),
        "selectionMutated": False,
        "replayMeaning": (
            "immutable cumulative baseline; mutation is deferred to exact "
            "whole-strip optimization"
        ),
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_patch_corridors(
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    surface_replay_root: str | Path | None = None,
    settings: PhysicalRibbonPatchCorridorSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonPatchCorridorSettings()
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
    cumulative_surface: dict[str, np.ndarray] | None = None
    cumulative_surface_stats: dict[str, Any] | None = None
    cumulative_reference: dict[str, Any] | None = None
    if surface_replay_root is not None:
        (
            cumulative_path,
            cumulative_manifest,
            cumulative_surface,
            cumulative_surface_stats,
        ) = load_materialized_cumulative_surface(
            surface_replay_root,
            configuration_manifest,
            configuration,
            topology,
        )
        cumulative_reference = {
            "manifestPath": str(cumulative_path),
            "manifestSha256": sha256_file(cumulative_path),
            "dataSha256": cumulative_manifest["data"]["sha256"],
        }
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_CORRIDORS_VERSION,
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
    if cumulative_reference is not None:
        identity["cumulativeSurfaceReplay"] = cumulative_reference
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_PATCH_CORRIDORS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_PATCH_CORRIDORS_STEM}.npz"
    preview_path = output / "physical-ribbon-patch-corridors.png"
    fragment_preview_path = output / "physical-ribbon-patch-corridor-fragments.png"
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
    if cumulative_surface is None:
        surface, surface_stats = build_physical_ribbon_surface_complex(
            ribbon,
            topology,
            configuration,
            settings=resolved.surface_settings(),
        )
    else:
        surface = cumulative_surface
        surface_stats = dict(cumulative_surface_stats or {})
    surfaced = time.monotonic()
    loops, loop_stats = extract_surface_boundary_loops(
        surface, settings=resolved.surface_settings()
    )
    looped = time.monotonic()
    corridors, corridor_stats = extract_surface_patch_corridors(
        surface, loops, settings=resolved
    )
    extracted = time.monotonic()
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    scored, scoring_stats = score_surface_patch_corridors(
        surface, corridors, source, settings=resolved
    )
    scored_at = time.monotonic()
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    reconfiguration, reconfiguration_stats = solve_patch_corridor_reconfigurations(
        corridors,
        scored,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        settings=resolved,
    )
    reconfigured_at = time.monotonic()
    if cumulative_surface is None:
        replay, replay_stats = replay_patch_corridor_reconfigurations(
            surface,
            corridors,
            scored,
            reconfiguration,
            ribbon,
            topology,
            configuration,
            settings=resolved,
        )
    else:
        replay, replay_stats = _baseline_cumulative_corridor_replay(
            surface, loops, corridors, scored, configuration
        )
    replayed_at = time.monotonic()
    arrays = {
        **surface,
        **loops,
        **corridors,
        **scored,
        **reconfiguration,
        **replay,
    }
    _write_npz(data_path, arrays)
    write_patch_corridor_montage(
        corridors,
        scored,
        preview_path,
        maximum_corridors=resolved.maximum_preview_corridors,
        reconfiguration=reconfiguration,
        replay=replay,
    )
    _, fragment_preview_stats = write_replayed_corridor_fragment_montage(
        surface,
        corridors,
        scored,
        replay,
        source,
        fragment_preview_path,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_CORRIDORS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": configuration_manifest.get("geometry", {}),
        "surface": surface_stats,
        "loops": loop_stats,
        "corridors": corridor_stats,
        "scoring": scoring_stats,
        "reconfiguration": reconfiguration_stats,
        "counterfactualReplay": replay_stats,
        "flattenedReplayFragments": fragment_preview_stats,
        "timingSeconds": {
            "surfaceComplex": round(surfaced - started, 6),
            "boundaryLoops": round(looped - surfaced, 6),
            "corridorExtraction": round(extracted - looped, 6),
            "rawCtAndTextureScoring": round(scored_at - extracted, 6),
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
        "artifacts": {
            "corridorMontage": preview_path.name,
            "flattenedReplayFragments": fragment_preview_path.name,
        },
        "method": {
            "decisionUnit": "mutually facing sequences of at least three boundary edges",
            "bendHandling": "ruled and tangent-plane-constrained cubic Hermite strips compete directly against native CT",
            "defectHandling": "profile ambiguity and failed boundary texture continuation remain unresolved rather than forcing a bridge",
            "reconfiguration": "complete alternating interface swaps are selected jointly under strict continuation, interface-exclusivity, and profile-crossing factors",
            "replayAcceptance": (
                "cumulative census preserves the exact augmented surface and "
                "defers mutation to complete-strip optimization"
                if cumulative_surface is not None
                else "a proposal survives only when its two original boundary arcs share one rebuilt triangle region; failed or component-splitting swaps are removed"
            ),
            "mutation": "counterfactual only; the source ribbon configuration is unchanged",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
