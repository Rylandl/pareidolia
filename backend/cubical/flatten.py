from __future__ import annotations

import heapq
import json
import math
import time
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from ..rectify import _trilinear
from .block import BlockBounds, SurfaceBlock, assemble_surface_hierarchy
from .contracts import (
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    resolve_pipeline_manifest,
    sha256_file,
)
from .continuity import apply_join_continuity_refinement
from .export import rgb_png
from .stratigraphic_continuity import (
    apply_stratigraphic_continuity_refinement,
)
from .surface_graph import read_surface_graph
from .tables import read_patch_shard


FLATTENING_SCHEMA = "pareidolia.cubical-component-flattening"
FLATTENING_VERSION = 1


@dataclass(frozen=True, slots=True)
class ComponentMesh:
    """One welded cubical component triangulated without smoothing its planes."""

    component_id: int
    patch_ids: tuple[int, ...]
    vertex_xyz: np.ndarray
    polygons: tuple[tuple[int, ...], ...]
    polygon_patch_ids: np.ndarray
    triangles: np.ndarray
    triangle_patch_ids: np.ndarray
    triangle_normal_xyz: np.ndarray
    statistics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class SurfaceChart:
    """One auditable UV atlas for an exact component mesh."""

    uv: np.ndarray
    anchor_vertices: tuple[tuple[int, int], ...]
    statistics: dict[str, Any]


@dataclass(frozen=True, slots=True)
class ChartRaster:
    """Exact piecewise-planar surface samples on a regular chart raster."""

    surface_xyz: np.ndarray
    normal_xyz: np.ndarray
    patch_id: np.ndarray
    mask: np.ndarray
    patch_boundary_mask: np.ndarray
    overlap_mask: np.ndarray
    pixel_step_voxels: float
    statistics: dict[str, Any]


def _axial_normal_statistics(normals: np.ndarray) -> dict[str, float]:
    values = np.asarray(normals, dtype=np.float64)
    reference = values[0]
    signed = values.copy()
    signed[signed @ reference < 0.0] *= -1.0
    for _ in range(4):
        mean = np.sum(signed, axis=0)
        mean /= max(float(np.linalg.norm(mean)), 1.0e-12)
        signed[sign := (signed @ mean < 0.0)] *= -1.0
        if not np.any(sign):
            break
    angles = np.degrees(
        np.arccos(np.clip(np.abs(signed @ mean), 0.0, 1.0))
    )
    return {
        "medianDegrees": round(float(np.median(angles)), 4),
        "p90Degrees": round(float(np.percentile(angles, 90)), 4),
        "maximumDegrees": round(float(np.max(angles)), 4),
    }


def _boundary_statistics(
    edge_records: dict[tuple[int, int], list[tuple[int, bool]]],
    vertex_count: int,
    face_count: int,
) -> dict[str, Any]:
    boundary = [edge for edge, records in edge_records.items() if len(records) == 1]
    boundary_graph: dict[int, list[int]] = defaultdict(list)
    for first, second in boundary:
        boundary_graph[first].append(second)
        boundary_graph[second].append(first)
    seen: set[int] = set()
    loops = 0
    chains = 0
    for seed in boundary_graph:
        if seed in seen:
            continue
        group = {seed}
        queue = [seed]
        seen.add(seed)
        while queue:
            current = queue.pop()
            for neighbor in boundary_graph[current]:
                if neighbor not in seen:
                    seen.add(neighbor)
                    group.add(neighbor)
                    queue.append(neighbor)
        if all(len(boundary_graph[value]) == 2 for value in group):
            loops += 1
        else:
            chains += 1
    return {
        "vertices": vertex_count,
        "edges": len(edge_records),
        "faces": face_count,
        "eulerCharacteristic": vertex_count - len(edge_records) + face_count,
        "boundaryEdges": len(boundary),
        "boundaryLoops": loops,
        "boundaryChains": chains,
        "nonmanifoldEdges": sum(
            len(records) > 2 for records in edge_records.values()
        ),
        "boundaryVertexDegree": {
            str(degree): count
            for degree, count in sorted(
                Counter(len(values) for values in boundary_graph.values()).items()
            )
        },
    }


def component_mesh(
    block: SurfaceBlock,
    component_id: int,
    *,
    maximum_chart_normal_deviation_degrees: float = 40.0,
) -> ComponentMesh:
    """Extract and consistently orient one welded polygon component."""

    if not 0.0 < maximum_chart_normal_deviation_degrees < 89.0:
        raise ValueError("chart normal deviation must lie strictly between 0 and 89 degrees")

    summary = next(
        (value for value in block.components if value.component_id == component_id),
        None,
    )
    if summary is None:
        raise KeyError(f"component {component_id} is absent from this block")
    patch_by_id = {value.patch_id: value for value in block.patches}
    observation_vertex = {
        observation: crossing_index
        for crossing_index, crossing in enumerate(block.welded_crossings)
        for observation in crossing.observations
    }
    global_vertices: set[int] = set()
    global_polygons: list[tuple[int, ...]] = []
    patch_ids = tuple(sorted(summary.patch_ids))
    for patch_id in patch_ids:
        patch = patch_by_id[patch_id]
        polygon = tuple(
            observation_vertex[(patch_id, value.edge)] for value in patch.vertices
        )
        if len(set(polygon)) != len(polygon):
            raise ValueError(f"patch {patch_id} has a repeated welded vertex")
        global_polygons.append(polygon)
        global_vertices.update(polygon)
    ordered_global = sorted(global_vertices)
    local_vertex = {value: index for index, value in enumerate(ordered_global)}
    polygons = [
        tuple(local_vertex[value] for value in polygon)
        for polygon in global_polygons
    ]
    vertex_xyz = np.asarray(
        [block.welded_crossings[value].point_xyz for value in ordered_global],
        dtype=np.float64,
    )

    raw_edge_records: dict[
        tuple[int, int], list[tuple[int, bool]]
    ] = defaultdict(list)
    for polygon_index, polygon in enumerate(polygons):
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            edge = (min(first, second), max(first, second))
            raw_edge_records[edge].append((polygon_index, first == edge[0]))
    zero_length_edges = {
        edge
        for edge in raw_edge_records
        if float(np.linalg.norm(vertex_xyz[edge[0]] - vertex_xyz[edge[1]]))
        <= 1.0e-8
    }
    # Corner welding can leave two distinct crossing identities at exactly one
    # cube vertex. Their connecting segment has no physical extent and cannot
    # be a manifold edge; retain it as a degeneracy diagnostic, not topology.
    edge_records = {
        edge: records
        for edge, records in raw_edge_records.items()
        if edge not in zero_length_edges
    }

    adjacency: dict[int, list[tuple[int, bool, tuple[int, int]]]] = defaultdict(list)
    for edge, records in edge_records.items():
        if len(records) != 2:
            continue
        (first, first_direction), (second, second_direction) = records
        # Shared edges must run in opposite directions after optional loop flips.
        required_xor = first_direction == second_direction
        adjacency[first].append((second, required_xor, edge))
        adjacency[second].append((first, required_xor, edge))
    flips: dict[int, bool] = {}
    retained_edges: set[tuple[int, int]] = set()
    conflicting_edges: set[tuple[int, int]] = set()
    for seed in range(len(polygons)):
        if seed in flips:
            continue
        flips[seed] = False
        queue: deque[int] = deque([seed])
        while queue:
            current = queue.popleft()
            for neighbor, required_xor, edge in adjacency[current]:
                expected = flips[current] ^ required_xor
                if neighbor not in flips:
                    flips[neighbor] = expected
                    retained_edges.add(edge)
                    queue.append(neighbor)
                elif flips[neighbor] != expected:
                    conflicting_edges.add(edge)
                    retained_edges.discard(edge)
                elif edge not in conflicting_edges:
                    retained_edges.add(edge)

    # A mesh with holes has no guaranteed injective free-boundary chart. Build
    # a maximum-score spanning forest of patch adjacencies for visualization:
    # redundant cycles become chart seams, while the strongest physical joins
    # remain continuous. This changes only UV topology, never the 3D surface.
    join_score = {
        frozenset((value.first_patch_id, value.second_patch_id)): value.score
        for value in block.joins
    }
    polygon_parent = list(range(len(polygons)))

    def polygon_find(value: int) -> int:
        root = polygon_parent[value]
        if root != value:
            polygon_parent[value] = polygon_find(root)
        return polygon_parent[value]

    forest_edges: set[tuple[int, int]] = set()
    for edge in sorted(
        retained_edges,
        key=lambda value: (
            -join_score.get(
                frozenset(
                    patch_ids[record[0]] for record in edge_records[value]
                ),
                0.0,
            ),
            value,
        ),
    ):
        records = edge_records[edge]
        if len(records) != 2:
            continue
        first = polygon_find(records[0][0])
        second = polygon_find(records[1][0])
        if first == second:
            continue
        if first < second:
            polygon_parent[second] = first
        else:
            polygon_parent[first] = second
        forest_edges.add(edge)
    cycle_seam_edges = retained_edges - forest_edges
    retained_edges = forest_edges

    forest_adjacency: dict[
        int, list[tuple[int, tuple[int, int]]]
    ] = defaultdict(list)
    for edge in retained_edges:
        records = edge_records[edge]
        first = records[0][0]
        second = records[1][0]
        forest_adjacency[first].append((second, edge))
        forest_adjacency[second].append((first, edge))
    polygon_normals = []
    for polygon_index, polygon in enumerate(polygons):
        ordered = tuple(reversed(polygon)) if flips[polygon_index] else polygon
        points = vertex_xyz[np.asarray(ordered)]
        normal = np.zeros(3, dtype=np.float64)
        for index in range(1, len(points) - 1):
            normal += np.cross(points[index] - points[0], points[index + 1] - points[0])
        length = float(np.linalg.norm(normal))
        if length <= 1.0e-10:
            normal = np.asarray(
                patch_by_id[patch_ids[polygon_index]].estimate.normal_xyz,
                dtype=np.float64,
            )
        else:
            normal /= length
        polygon_normals.append(normal)
    polygon_normals = np.asarray(polygon_normals)
    chart_cluster: dict[int, int] = {}
    cluster_normal_sum: dict[int, np.ndarray] = {}
    next_cluster = 0
    threshold = math.radians(maximum_chart_normal_deviation_degrees)
    for seed in range(len(polygons)):
        if seed in chart_cluster:
            continue
        chart_cluster[seed] = next_cluster
        cluster_normal_sum[next_cluster] = polygon_normals[seed].copy()
        queue: deque[int] = deque([seed])
        next_cluster += 1
        while queue:
            current = queue.popleft()
            current_cluster = chart_cluster[current]
            for neighbor, _ in forest_adjacency[current]:
                if neighbor in chart_cluster:
                    continue
                mean = cluster_normal_sum[current_cluster]
                mean /= max(float(np.linalg.norm(mean)), 1.0e-12)
                candidate = polygon_normals[neighbor].copy()
                angle = math.acos(
                    float(np.clip(np.dot(candidate, mean), -1.0, 1.0))
                )
                if angle <= threshold:
                    chart_cluster[neighbor] = current_cluster
                    cluster_normal_sum[current_cluster] += candidate
                else:
                    chart_cluster[neighbor] = next_cluster
                    cluster_normal_sum[next_cluster] = candidate
                    next_cluster += 1
                queue.append(neighbor)
    normal_seam_edges = {
        edge
        for edge in retained_edges
        if chart_cluster[edge_records[edge][0][0]]
        != chart_cluster[edge_records[edge][1][0]]
    }
    retained_edges -= normal_seam_edges

    # A contradictory or nonmanifold edge becomes an explicit chart seam. The
    # 3D surface remains exact; only its UV topology is cut. Unioning polygon
    # observations across retained edges duplicates exactly the seam vertices.
    observations = [
        (polygon_index, vertex)
        for polygon_index, polygon in enumerate(polygons)
        for vertex in polygon
    ]
    parent = {value: value for value in observations}

    def find(value: tuple[int, int]) -> tuple[int, int]:
        root = parent[value]
        if root != value:
            parent[value] = find(root)
        return parent[value]

    def union(first: tuple[int, int], second: tuple[int, int]) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if first_root <= second_root:
            parent[second_root] = first_root
        else:
            parent[first_root] = second_root

    for edge in retained_edges:
        records = edge_records[edge]
        if len(records) != 2:
            continue
        first_polygon = records[0][0]
        second_polygon = records[1][0]
        for vertex in edge:
            union((first_polygon, vertex), (second_polygon, vertex))
    roots = sorted({find(value) for value in observations})
    root_index = {value: index for index, value in enumerate(roots)}
    cut_vertex_xyz = np.asarray(
        [vertex_xyz[value[1]] for value in roots], dtype=np.float64
    )
    cut_polygons = [
        tuple(root_index[find((polygon_index, vertex))] for vertex in polygon)
        for polygon_index, polygon in enumerate(polygons)
    ]
    cleaned_polygons = []
    collapsed_corner_vertices = 0
    for polygon in cut_polygons:
        cleaned: list[int] = []
        for vertex in polygon:
            if cleaned and np.linalg.norm(
                cut_vertex_xyz[vertex] - cut_vertex_xyz[cleaned[-1]]
            ) <= 1.0e-8:
                collapsed_corner_vertices += 1
                continue
            cleaned.append(vertex)
        if len(cleaned) > 1 and np.linalg.norm(
            cut_vertex_xyz[cleaned[0]] - cut_vertex_xyz[cleaned[-1]]
        ) <= 1.0e-8:
            cleaned.pop()
            collapsed_corner_vertices += 1
        cleaned_polygons.append(tuple(cleaned))
    oriented = tuple(
        tuple(reversed(polygon)) if flips[index] else polygon
        for index, polygon in enumerate(cleaned_polygons)
    )

    cut_edge_records: dict[
        tuple[int, int], list[tuple[int, bool]]
    ] = defaultdict(list)
    for polygon_index, polygon in enumerate(oriented):
        for index, first in enumerate(polygon):
            second = polygon[(index + 1) % len(polygon)]
            edge = (min(first, second), max(first, second))
            cut_edge_records[edge].append((polygon_index, first == edge[0]))

    # Center fans avoid artificial degeneracies when a valid clipped polygon
    # contains an exactly collinear grid-corner transition.
    polygon_centers = np.asarray(
        [np.mean(cut_vertex_xyz[np.asarray(polygon)], axis=0) for polygon in oriented],
        dtype=np.float64,
    )
    vertex_xyz = np.concatenate((cut_vertex_xyz, polygon_centers), axis=0)

    triangles: list[tuple[int, int, int]] = []
    triangle_patch_ids: list[int] = []
    triangle_normals: list[np.ndarray] = []
    degenerate_triangles = 0
    for polygon_index, (polygon, patch_id) in enumerate(zip(oriented, patch_ids)):
        patch_normal = np.asarray(
            patch_by_id[patch_id].estimate.normal_xyz, dtype=np.float64
        )
        center = len(cut_vertex_xyz) + polygon_index
        for index, first in enumerate(polygon):
            triangle = (center, first, polygon[(index + 1) % len(polygon)])
            points = vertex_xyz[np.asarray(triangle)]
            geometric = np.cross(points[1] - points[0], points[2] - points[0])
            length = float(np.linalg.norm(geometric))
            if length <= 1.0e-8:
                degenerate_triangles += 1
                continue
            geometric /= length
            normal = patch_normal.copy()
            if float(np.dot(normal, geometric)) < 0.0:
                normal *= -1.0
            triangles.append(triangle)
            triangle_patch_ids.append(patch_id)
            triangle_normals.append(normal)
    if not triangles:
        raise ValueError(f"component {component_id} has no nondegenerate triangles")

    used_vertices = sorted({value for triangle in triangles for value in triangle})
    if len(used_vertices) != len(vertex_xyz):
        remapped = {value: index for index, value in enumerate(used_vertices)}
        vertex_xyz = vertex_xyz[np.asarray(used_vertices)]
        triangles = [
            tuple(remapped[value] for value in triangle) for triangle in triangles
        ]
        oriented = tuple(
            tuple(remapped[value] for value in polygon if value in remapped)
            for polygon in oriented
        )

    topology = _boundary_statistics(edge_records, len(ordered_global), len(oriented))
    topology["coincidentZeroLengthEdges"] = len(zero_length_edges)
    topology["coincidentZeroLengthEdgeIncidences"] = sum(
        len(raw_edge_records[edge]) for edge in zero_length_edges
    )
    topology["orientationConflicts"] = len(conflicting_edges)
    topology["chartConflictSeamEdges"] = len(conflicting_edges) + sum(
        len(records) > 2 for records in edge_records.values()
    )
    topology["chartCycleSeamEdges"] = len(cycle_seam_edges)
    topology["chartNormalSeamEdges"] = len(normal_seam_edges)
    topology["maximumChartNormalDeviationDegrees"] = (
        maximum_chart_normal_deviation_degrees
    )
    topology["chartSeamEdges"] = (
        topology["chartConflictSeamEdges"]
        + len(cycle_seam_edges)
        + len(normal_seam_edges)
    )
    topology["chartTopology"] = _boundary_statistics(
        cut_edge_records, len(cut_vertex_xyz), len(oriented)
    )
    topology["degenerateTriangles"] = degenerate_triangles
    topology["collapsedCoincidentCornerVertices"] = collapsed_corner_vertices
    topology["triangles"] = len(triangles)
    topology["normalDeviation"] = _axial_normal_statistics(
        np.asarray([patch_by_id[value].estimate.normal_xyz for value in patch_ids])
    )
    return ComponentMesh(
        component_id,
        patch_ids,
        vertex_xyz,
        oriented,
        np.asarray(patch_ids, dtype=np.uint64),
        np.asarray(triangles, dtype=np.int32),
        np.asarray(triangle_patch_ids, dtype=np.uint64),
        np.asarray(triangle_normals, dtype=np.float64),
        topology,
    )


def _mesh_graph(mesh: ComponentMesh) -> dict[int, dict[int, float]]:
    graph: dict[int, dict[int, float]] = {
        index: {} for index in range(len(mesh.vertex_xyz))
    }
    for triangle in mesh.triangles:
        for index, first in enumerate(triangle):
            second = int(triangle[(index + 1) % 3])
            first = int(first)
            distance = float(
                np.linalg.norm(mesh.vertex_xyz[first] - mesh.vertex_xyz[second])
            )
            previous = graph[first].get(second)
            if previous is None or distance < previous:
                graph[first][second] = distance
                graph[second][first] = distance
    return graph


def _geodesic_distances(
    graph: dict[int, dict[int, float]], seed: int
) -> np.ndarray:
    distances = np.full(len(graph), np.inf, dtype=np.float64)
    distances[seed] = 0.0
    queue: list[tuple[float, int]] = [(0.0, seed)]
    while queue:
        distance, current = heapq.heappop(queue)
        if distance != distances[current]:
            continue
        for neighbor, weight in graph[current].items():
            proposed = distance + weight
            if proposed < distances[neighbor]:
                distances[neighbor] = proposed
                heapq.heappush(queue, (proposed, neighbor))
    return distances


def _graph_components(graph: dict[int, dict[int, float]]) -> tuple[tuple[int, ...], ...]:
    remaining = set(graph)
    result = []
    while remaining:
        seed = min(remaining)
        group = {seed}
        queue = [seed]
        while queue:
            current = queue.pop()
            for neighbor in graph[current]:
                if neighbor not in group:
                    group.add(neighbor)
                    queue.append(neighbor)
        remaining -= group
        result.append(tuple(sorted(group)))
    return tuple(result)


def _chart_anchors(
    graph: dict[int, dict[int, float]], vertices: tuple[int, ...]
) -> tuple[int, int, float]:
    seed = vertices[0]
    first_distances = _geodesic_distances(graph, seed)
    first = max(vertices, key=lambda value: first_distances[value])
    second_distances = _geodesic_distances(graph, first)
    second = max(vertices, key=lambda value: second_distances[value])
    distance = float(second_distances[second])
    if not math.isfinite(distance) or distance <= 0.0:
        raise ValueError("component mesh is disconnected or collapsed")
    return first, second, distance


def conformal_chart(mesh: ComponentMesh) -> SurfaceChart:
    """Solve a dense free-boundary LSCM chart with two geodesic anchors."""

    vertex_count = len(mesh.vertex_xyz)
    triangle_count = len(mesh.triangles)
    matrix = np.zeros((2 * triangle_count, 2 * vertex_count), dtype=np.float64)
    triangle_local: list[np.ndarray] = []
    triangle_area: list[float] = []
    for triangle_index, triangle in enumerate(mesh.triangles):
        points = mesh.vertex_xyz[triangle]
        first = points[1] - points[0]
        second = points[2] - points[0]
        length = float(np.linalg.norm(first))
        axis = first / length
        x2 = float(np.dot(second, axis))
        y2 = float(np.linalg.norm(second - x2 * axis))
        if length <= 1.0e-10 or y2 <= 1.0e-10:
            raise ValueError("component triangulation contains a collapsed face")
        local = np.asarray(((0.0, 0.0), (length, 0.0), (x2, y2)))
        triangle_local.append(local)
        doubled_area = length * y2
        triangle_area.append(0.5 * doubled_area)
        gradient_x = np.asarray(
            (
                local[1, 1] - local[2, 1],
                local[2, 1] - local[0, 1],
                local[0, 1] - local[1, 1],
            )
        ) / doubled_area
        gradient_y = np.asarray(
            (
                local[2, 0] - local[1, 0],
                local[0, 0] - local[2, 0],
                local[1, 0] - local[0, 0],
            )
        ) / doubled_area
        weight = math.sqrt(0.5 * doubled_area)
        first_row = 2 * triangle_index
        second_row = first_row + 1
        matrix[first_row, triangle] = weight * gradient_x
        matrix[first_row, vertex_count + triangle] = -weight * gradient_y
        matrix[second_row, triangle] = weight * gradient_y
        matrix[second_row, vertex_count + triangle] = weight * gradient_x

    graph = _mesh_graph(mesh)
    graph_components = _graph_components(graph)
    anchors = tuple(
        _chart_anchors(graph, vertices) for vertices in graph_components
    )
    fixed_indices = np.asarray(
        [
            fixed
            for first_anchor, second_anchor, _ in anchors
            for fixed in (
                first_anchor,
                second_anchor,
                vertex_count + first_anchor,
                vertex_count + second_anchor,
            )
        ],
        dtype=np.int64,
    )
    fixed_values = np.asarray(
        [
            value
            for _, _, anchor_distance in anchors
            for value in (0.0, anchor_distance, 0.0, 0.0)
        ]
    )
    free_mask = np.ones(2 * vertex_count, dtype=bool)
    free_mask[fixed_indices] = False
    right = -(matrix[:, fixed_indices] @ fixed_values)
    solved, residuals, rank, singular = np.linalg.lstsq(
        matrix[:, free_mask], right, rcond=1.0e-11
    )
    values = np.zeros(2 * vertex_count, dtype=np.float64)
    values[fixed_indices] = fixed_values
    values[free_mask] = solved
    uv = np.column_stack((values[:vertex_count], values[vertex_count:]))

    triangle_area_values = np.asarray(triangle_area)
    cursor_x = 0.0
    chart_gap = 16.0
    for vertices in graph_components:
        vertex_indices = np.asarray(vertices, dtype=np.int32)
        included = np.zeros(vertex_count, dtype=bool)
        included[vertex_indices] = True
        selected_triangles = np.all(included[mesh.triangles], axis=1)
        first_delta = (
            uv[mesh.triangles[selected_triangles, 1]]
            - uv[mesh.triangles[selected_triangles, 0]]
        )
        second_delta = (
            uv[mesh.triangles[selected_triangles, 2]]
            - uv[mesh.triangles[selected_triangles, 0]]
        )
        signed = 0.5 * (
            first_delta[:, 0] * second_delta[:, 1]
            - first_delta[:, 1] * second_delta[:, 0]
        )
        if float(np.sum(signed)) < 0.0:
            uv[vertex_indices, 1] *= -1.0
            signed *= -1.0
        surface_area = float(np.sum(triangle_area_values[selected_triangles]))
        chart_area = float(np.sum(np.abs(signed)))
        if chart_area <= 1.0e-10:
            raise ValueError("conformal chart piece collapsed to zero area")
        uv[vertex_indices] *= math.sqrt(surface_area / chart_area)
        piece_low = np.min(uv[vertex_indices], axis=0)
        uv[vertex_indices] -= piece_low
        uv[vertex_indices, 0] += cursor_x
        cursor_x = float(np.max(uv[vertex_indices, 0])) + chart_gap

    first_delta = uv[mesh.triangles[:, 1]] - uv[mesh.triangles[:, 0]]
    second_delta = uv[mesh.triangles[:, 2]] - uv[mesh.triangles[:, 0]]
    signed_area = 0.5 * (
        first_delta[:, 0] * second_delta[:, 1]
        - first_delta[:, 1] * second_delta[:, 0]
    )
    conformal_distortion = []
    area_stretch = []
    for triangle, local in zip(mesh.triangles, triangle_local):
        source_basis = np.column_stack((local[1] - local[0], local[2] - local[0]))
        target_basis = np.column_stack(
            (uv[triangle[1]] - uv[triangle[0]], uv[triangle[2]] - uv[triangle[0]])
        )
        jacobian = target_basis @ np.linalg.inv(source_basis)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        conformal_distortion.append(
            float(singular_values[0] / max(singular_values[-1], 1.0e-12))
        )
        area_stretch.append(abs(float(np.linalg.det(jacobian))))
    distortion = np.asarray(conformal_distortion)
    stretch = np.asarray(area_stretch)
    residual = matrix @ values
    return SurfaceChart(
        uv,
        tuple((first, second) for first, second, _ in anchors),
        {
            "solver": "free-boundary dense least-squares conformal map",
            "chartPieces": len(graph_components),
            "anchorVertices": [
                [first, second] for first, second, _ in anchors
            ],
            "anchorGeodesicDistanceVoxels": [
                round(distance, 4) for _, _, distance in anchors
            ],
            "equations": int(matrix.shape[0]),
            "unknowns": int(matrix.shape[1]),
            "freeRank": int(rank),
            "freeUnknowns": int(np.count_nonzero(free_mask)),
            "rmsConformalResidual": round(
                float(np.linalg.norm(residual) / math.sqrt(max(len(residual), 1))),
                8,
            ),
            "flippedTriangles": int(np.count_nonzero(signed_area <= 1.0e-10)),
            "flippedTriangleFraction": round(
                float(np.mean(signed_area <= 1.0e-10)), 6
            ),
            "conformalDistortion": {
                "median": round(float(np.median(distortion)), 5),
                "p90": round(float(np.percentile(distortion, 90)), 5),
                "maximum": round(float(np.max(distortion)), 5),
            },
            "areaStretch": {
                "median": round(float(np.median(stretch)), 5),
                "p10": round(float(np.percentile(stretch, 10)), 5),
                "p90": round(float(np.percentile(stretch, 90)), 5),
            },
        },
    )


def tangent_atlas_chart(mesh: ComponentMesh) -> SurfaceChart:
    """Project bounded-normal chart pieces into their own tangent frames.

    ``component_mesh`` has already cut contradictory, nonmanifold, redundant
    cycle, and excessive-curvature adjacencies. Each remaining connected piece
    is therefore a locally graph-like surface with a known projection-distortion
    ceiling. Pieces are packed without changing their physical voxel scale.
    """

    graph = _mesh_graph(mesh)
    components = _graph_components(graph)
    vertex_count = len(mesh.vertex_xyz)
    uv = np.zeros((vertex_count, 2), dtype=np.float64)
    local_piece: list[tuple[np.ndarray, np.ndarray, dict[str, Any]]] = []
    for component_index, vertices in enumerate(components):
        vertex_indices = np.asarray(vertices, dtype=np.int32)
        included = np.zeros(vertex_count, dtype=bool)
        included[vertex_indices] = True
        selected_triangles = np.flatnonzero(
            np.all(included[mesh.triangles], axis=1)
        )
        normals = mesh.triangle_normal_xyz[selected_triangles].copy()
        normal = np.sum(normals, axis=0)
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        points = mesh.vertex_xyz[vertex_indices]
        center = np.mean(points, axis=0)
        centered = points - center
        tangent_points = centered - (centered @ normal)[:, None] * normal
        covariance = tangent_points.T @ tangent_points
        eigenvalues, eigenvectors = np.linalg.eigh(covariance)
        first_axis = eigenvectors[:, int(np.argmax(eigenvalues))]
        first_axis -= float(np.dot(first_axis, normal)) * normal
        if np.linalg.norm(first_axis) <= 1.0e-10:
            basis = np.eye(3)[int(np.argmin(np.abs(normal)))]
            first_axis = np.cross(normal, basis)
        first_axis /= max(float(np.linalg.norm(first_axis)), 1.0e-12)
        second_axis = np.cross(normal, first_axis)
        second_axis /= max(float(np.linalg.norm(second_axis)), 1.0e-12)
        piece_uv = np.column_stack(
            (centered @ first_axis, centered @ second_axis)
        )
        local_index = {value: index for index, value in enumerate(vertices)}
        local_triangles = np.asarray(
            [
                tuple(local_index[int(value)] for value in mesh.triangles[index])
                for index in selected_triangles
            ],
            dtype=np.int32,
        )
        first_delta = (
            piece_uv[local_triangles[:, 1]] - piece_uv[local_triangles[:, 0]]
        )
        second_delta = (
            piece_uv[local_triangles[:, 2]] - piece_uv[local_triangles[:, 0]]
        )
        signed = 0.5 * (
            first_delta[:, 0] * second_delta[:, 1]
            - first_delta[:, 1] * second_delta[:, 0]
        )
        if float(np.sum(signed)) < 0.0:
            piece_uv[:, 1] *= -1.0
            second_axis *= -1.0
        piece_uv -= np.min(piece_uv, axis=0, keepdims=True)
        local_piece.append(
            (
                vertex_indices,
                piece_uv,
                {
                    "piece": component_index,
                    "vertices": len(vertex_indices),
                    "triangles": len(selected_triangles),
                    "normalXYZ": [round(float(value), 7) for value in normal],
                    "uAxisXYZ": [round(float(value), 7) for value in first_axis],
                    "vAxisXYZ": [round(float(value), 7) for value in second_axis],
                },
            )
        )

    gap = 16.0
    piece_extents = [np.max(value[1], axis=0) for value in local_piece]
    total_box_area = sum(
        max(float(extent[0]), 1.0) * max(float(extent[1]), 1.0)
        for extent in piece_extents
    )
    target_width = max(
        max(float(value[0]) for value in piece_extents),
        math.sqrt(total_box_area) * 1.35,
    )
    cursor_x = 0.0
    cursor_y = 0.0
    row_height = 0.0
    piece_records = []
    for (vertex_indices, piece_uv, record), extent in zip(local_piece, piece_extents):
        width = float(extent[0])
        height = float(extent[1])
        if cursor_x > 0.0 and cursor_x + width > target_width:
            cursor_x = 0.0
            cursor_y += row_height + gap
            row_height = 0.0
        piece_uv[:, 0] += cursor_x
        piece_uv[:, 1] += cursor_y
        uv[vertex_indices] = piece_uv
        record["atlasOffsetUV"] = [round(cursor_x, 4), round(cursor_y, 4)]
        piece_records.append(record)
        cursor_x += width + gap
        row_height = max(row_height, height)

    first_delta = uv[mesh.triangles[:, 1]] - uv[mesh.triangles[:, 0]]
    second_delta = uv[mesh.triangles[:, 2]] - uv[mesh.triangles[:, 0]]
    signed_area = 0.5 * (
        first_delta[:, 0] * second_delta[:, 1]
        - first_delta[:, 1] * second_delta[:, 0]
    )
    distortion = []
    stretch = []
    for triangle in mesh.triangles:
        points = mesh.vertex_xyz[triangle]
        first = points[1] - points[0]
        second = points[2] - points[0]
        length = float(np.linalg.norm(first))
        axis = first / max(length, 1.0e-12)
        x2 = float(np.dot(second, axis))
        y2 = float(np.linalg.norm(second - x2 * axis))
        source_basis = np.asarray(((length, x2), (0.0, y2)))
        target_basis = np.column_stack(
            (uv[triangle[1]] - uv[triangle[0]], uv[triangle[2]] - uv[triangle[0]])
        )
        jacobian = target_basis @ np.linalg.inv(source_basis)
        singular_values = np.linalg.svd(jacobian, compute_uv=False)
        distortion.append(
            float(singular_values[0] / max(singular_values[-1], 1.0e-12))
        )
        stretch.append(abs(float(np.linalg.det(jacobian))))
    distortion_values = np.asarray(distortion)
    stretch_values = np.asarray(stretch)
    return SurfaceChart(
        uv,
        (),
        {
            "solver": "bounded-normal tangent atlas",
            "chartPieces": len(components),
            "pieces": piece_records,
            "flippedTriangles": int(np.count_nonzero(signed_area <= 1.0e-10)),
            "flippedTriangleFraction": round(
                float(np.mean(signed_area <= 1.0e-10)), 6
            ),
            "linearDistortion": {
                "median": round(float(np.median(distortion_values)), 5),
                "p90": round(float(np.percentile(distortion_values, 90)), 5),
                "maximum": round(float(np.max(distortion_values)), 5),
            },
            "areaStretch": {
                "median": round(float(np.median(stretch_values)), 5),
                "p10": round(float(np.percentile(stretch_values, 10)), 5),
                "p90": round(float(np.percentile(stretch_values, 90)), 5),
            },
        },
    )


def _patch_boundaries(patch_id: np.ndarray, mask: np.ndarray) -> np.ndarray:
    boundary = np.zeros(mask.shape, dtype=bool)
    for axis in (0, 1):
        first = [slice(None), slice(None)]
        second = [slice(None), slice(None)]
        first[axis] = slice(0, -1)
        second[axis] = slice(1, None)
        first_key = tuple(first)
        second_key = tuple(second)
        different = (
            mask[first_key]
            & mask[second_key]
            & (patch_id[first_key] != patch_id[second_key])
        )
        boundary[first_key] |= different
        boundary[second_key] |= different
    return boundary


def rasterize_chart(
    mesh: ComponentMesh,
    chart: SurfaceChart,
    *,
    pixel_step_voxels: float = 2.0,
    maximum_pixels: int = 768,
    padding_pixels: int = 5,
) -> ChartRaster:
    """Rasterize triangles barycentrically while exposing nonadjacent overlap."""

    if pixel_step_voxels <= 0.0 or maximum_pixels < 64 or padding_pixels < 1:
        raise ValueError("chart raster settings are invalid")
    low = np.min(chart.uv, axis=0)
    high = np.max(chart.uv, axis=0)
    extent = np.maximum(high - low, 1.0e-6)
    available = maximum_pixels - 2 * padding_pixels
    effective_step = max(
        float(pixel_step_voxels),
        float(extent[0] / max(available - 1, 1)),
        float(extent[1] / max(available - 1, 1)),
    )
    width = int(math.ceil(extent[0] / effective_step)) + 1 + 2 * padding_pixels
    height = int(math.ceil(extent[1] / effective_step)) + 1 + 2 * padding_pixels
    uv_pixels = (chart.uv - low) / effective_step + padding_pixels
    surface = np.full((height, width, 3), np.nan, dtype=np.float32)
    normal = np.full((height, width, 3), np.nan, dtype=np.float32)
    # Stable patch identities are full uint64 content hashes.  Keep the raster
    # in that domain as well; signed int64 silently excludes half of the valid
    # identity space.  The mask, rather than a negative sentinel, declares
    # whether a pixel owns a patch.
    patch_id = np.full(
        (height, width), np.iinfo(np.uint64).max, dtype=np.uint64
    )
    triangle_owner = np.full((height, width), -1, dtype=np.int32)
    interior_owner = np.full((height, width), -1, dtype=np.int32)
    overlap = np.zeros((height, width), dtype=bool)
    triangle_vertex_sets = [set(int(value) for value in triangle) for triangle in mesh.triangles]

    for triangle_index, triangle in enumerate(mesh.triangles):
        uv = uv_pixels[triangle]
        minimum = np.maximum(np.floor(np.min(uv, axis=0)).astype(int), 0)
        maximum = np.minimum(
            np.ceil(np.max(uv, axis=0)).astype(int),
            np.asarray((width - 1, height - 1)),
        )
        if np.any(maximum < minimum):
            continue
        x_values = np.arange(minimum[0], maximum[0] + 1)
        y_values = np.arange(minimum[1], maximum[1] + 1)
        x_grid, y_grid = np.meshgrid(x_values, y_values)
        points = np.column_stack((x_grid.ravel(), y_grid.ravel()))
        denominator = float(
            (uv[1, 1] - uv[2, 1]) * (uv[0, 0] - uv[2, 0])
            + (uv[2, 0] - uv[1, 0]) * (uv[0, 1] - uv[2, 1])
        )
        if abs(denominator) <= 1.0e-12:
            continue
        first = (
            (uv[1, 1] - uv[2, 1]) * (points[:, 0] - uv[2, 0])
            + (uv[2, 0] - uv[1, 0]) * (points[:, 1] - uv[2, 1])
        ) / denominator
        second = (
            (uv[2, 1] - uv[0, 1]) * (points[:, 0] - uv[2, 0])
            + (uv[0, 0] - uv[2, 0]) * (points[:, 1] - uv[2, 1])
        ) / denominator
        third = 1.0 - first - second
        weights = np.column_stack((first, second, third))
        inside = np.all(weights >= -1.0e-6, axis=1)
        if not np.any(inside):
            continue
        selected_points = points[inside].astype(np.int64)
        selected_weights = weights[inside]
        rows = selected_points[:, 1]
        columns = selected_points[:, 0]
        previous = triangle_owner[rows, columns]
        for previous_triangle in np.unique(previous[previous >= 0]):
            locations = previous == previous_triangle
            if len(
                triangle_vertex_sets[triangle_index]
                & triangle_vertex_sets[int(previous_triangle)]
            ) < 2:
                overlap[rows[locations], columns[locations]] = True
        empty = previous < 0
        if np.any(empty):
            target_rows = rows[empty]
            target_columns = columns[empty]
            barycentric = selected_weights[empty]
            surface[target_rows, target_columns] = (
                barycentric @ mesh.vertex_xyz[triangle]
            ).astype(np.float32)
            normal[target_rows, target_columns] = mesh.triangle_normal_xyz[
                triangle_index
            ].astype(np.float32)
            patch_id[target_rows, target_columns] = mesh.triangle_patch_ids[
                triangle_index
            ]
            triangle_owner[target_rows, target_columns] = triangle_index
        interior = np.min(selected_weights, axis=1) > 0.02
        prior_interior = interior_owner[rows[interior], columns[interior]]
        if np.any(prior_interior >= 0):
            overlap[rows[interior][prior_interior >= 0], columns[interior][prior_interior >= 0]] = True
        interior_owner[rows[interior], columns[interior]] = triangle_index

    mask = triangle_owner >= 0
    boundary = _patch_boundaries(patch_id, mask)
    overlap &= mask
    return ChartRaster(
        surface,
        normal,
        patch_id,
        mask,
        boundary,
        overlap,
        effective_step,
        {
            "shapeYX": [height, width],
            "requestedPixelStepVoxels": float(pixel_step_voxels),
            "effectivePixelStepVoxels": round(effective_step, 6),
            "supportedPixels": int(np.count_nonzero(mask)),
            "supportedFraction": round(float(np.mean(mask)), 6),
            "patchBoundaryPixels": int(np.count_nonzero(boundary)),
            "nonadjacentOverlapPixels": int(np.count_nonzero(overlap)),
            "nonadjacentOverlapFraction": round(
                float(np.count_nonzero(overlap) / max(np.count_nonzero(mask), 1)),
                8,
            ),
        },
    )


def sample_depth_stack(
    source: VolumeSource,
    raster: ChartRaster,
    depth_offsets_voxels: Iterable[float],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Sample native CT at fixed offsets from the unsmoothed cubical surface."""

    offsets = np.asarray(tuple(float(value) for value in depth_offsets_voxels))
    if offsets.ndim != 1 or not len(offsets) or np.any(~np.isfinite(offsets)):
        raise ValueError("depth offsets must be one finite sequence")
    valid_surface = raster.surface_xyz[raster.mask].astype(np.float32)
    valid_normal = raster.normal_xyz[raster.mask].astype(np.float32)
    local_surface = valid_surface - np.asarray(source.origin_xyz, dtype=np.float32)
    extreme = np.concatenate(
        (
            local_surface + float(np.min(offsets)) * valid_normal,
            local_surface + float(np.max(offsets)) * valid_normal,
        ),
        axis=0,
    )
    low = np.floor(np.min(extreme, axis=0) - 2.0).astype(int)
    high = np.ceil(np.max(extreme, axis=0) + 3.0).astype(int)
    shape = np.asarray(source.shape_xyz)
    if np.any(low < 0) or np.any(high > shape):
        raise ValueError("flattened component depth stack leaves the native CT source")
    x0, y0, z0 = (int(value) for value in low)
    x1, y1, z1 = (int(value) for value in high)
    source_array = source.memmap()
    subvolume = np.asarray(source_array[z0:z1, y0:y1, x0:x1], dtype=np.uint8)
    local_origin = np.asarray((x0, y0, z0), dtype=np.float32)
    stack = np.zeros((len(offsets), *raster.mask.shape), dtype=np.uint8)
    for index, depth in enumerate(offsets):
        points = local_surface + float(depth) * valid_normal - local_origin
        sampled = _trilinear(subvolume, points)
        plane = np.zeros(raster.mask.shape, dtype=np.float32)
        plane[raster.mask] = sampled
        stack[index] = np.clip(np.rint(plane), 0.0, 255.0).astype(np.uint8)
    return stack, {
        "depthOffsetsVoxels": [float(value) for value in offsets],
        "sourceBoundsXYZ": [x0, x1, y0, y1, z0, z1],
        "sourceSubvolumeShapeZYX": [z1 - z0, y1 - y0, x1 - x0],
        "sourceSubvolumeMiB": round(float(subvolume.nbytes / (1024**2)), 3),
    }


_FONT_5X7 = {
    " ": ("00000",) * 7,
    "+": ("00000", "00100", "00100", "11111", "00100", "00100", "00000"),
    "-": ("00000", "00000", "00000", "11111", "00000", "00000", "00000"),
    "0": ("01110", "10001", "10011", "10101", "11001", "10001", "01110"),
    "1": ("00100", "01100", "00100", "00100", "00100", "00100", "01110"),
    "2": ("01110", "10001", "00001", "00010", "00100", "01000", "11111"),
    "3": ("11110", "00001", "00001", "01110", "00001", "00001", "11110"),
    "4": ("00010", "00110", "01010", "10010", "11111", "00010", "00010"),
    "5": ("11111", "10000", "10000", "11110", "00001", "00001", "11110"),
    "6": ("01110", "10000", "10000", "11110", "10001", "10001", "01110"),
    "7": ("11111", "00001", "00010", "00100", "01000", "01000", "01000"),
    "8": ("01110", "10001", "10001", "01110", "10001", "10001", "01110"),
    "9": ("01110", "10001", "10001", "01111", "00001", "00001", "01110"),
    "C": ("01110", "10001", "10000", "10000", "10000", "10001", "01110"),
    "D": ("11110", "10001", "10001", "10001", "10001", "10001", "11110"),
    "N": ("10001", "11001", "10101", "10011", "10001", "10001", "10001"),
    "R": ("11110", "10001", "10001", "11110", "10100", "10010", "10001"),
    "X": ("10001", "10001", "01010", "00100", "01010", "10001", "10001"),
    "Y": ("10001", "10001", "01010", "00100", "00100", "00100", "00100"),
}


def _draw_text(
    image: np.ndarray,
    x: int,
    y: int,
    value: str,
    color: tuple[int, int, int],
    *,
    scale: int = 1,
) -> None:
    cursor = x
    for character in value.upper():
        glyph = _FONT_5X7.get(character, _FONT_5X7[" "])
        for row, pattern in enumerate(glyph):
            for column, bit in enumerate(pattern):
                if bit != "1":
                    continue
                y0 = y + row * scale
                x0 = cursor + column * scale
                image[y0 : y0 + scale, x0 : x0 + scale] = color
        cursor += 6 * scale


def _contrast_limits(stack: np.ndarray, mask: np.ndarray) -> tuple[float, float]:
    values = np.asarray(stack[:, mask], dtype=np.float32)
    if not values.size:
        return 0.0, 1.0
    low, high = np.percentile(values, (1.0, 99.5))
    if high <= low + 1.0e-6:
        high = low + 1.0
    return float(low), float(high)


def _contrast_plane(
    plane: np.ndarray,
    mask: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    low, high = limits
    scaled = np.clip((plane.astype(np.float32) - low) / (high - low), 0.0, 1.0)
    image = np.rint(12.0 + 243.0 * scaled).astype(np.uint8)
    image[~mask] = 0
    return image


def _overlay(
    grayscale: np.ndarray,
    raster: ChartRaster,
    *,
    include_boundaries: bool,
) -> np.ndarray:
    image = np.repeat(grayscale[..., None], 3, axis=2)
    if include_boundaries:
        image[raster.patch_boundary_mask] = (0, 210, 235)
    image[raster.overlap_mask] = (255, 48, 48)
    return image


def _nearest_resize(image: np.ndarray, width: int, height: int) -> np.ndarray:
    if width <= 0 or height <= 0:
        raise ValueError("image resize target must be positive")
    rows = np.minimum(
        np.floor(np.arange(height) * image.shape[0] / height).astype(int),
        image.shape[0] - 1,
    )
    columns = np.minimum(
        np.floor(np.arange(width) * image.shape[1] / width).astype(int),
        image.shape[1] - 1,
    )
    return image[rows[:, None], columns[None, :]]


def _fit_image(image: np.ndarray, maximum_width: int, maximum_height: int) -> np.ndarray:
    scale = min(
        maximum_width / max(image.shape[1], 1),
        maximum_height / max(image.shape[0], 1),
        1.0,
    )
    return _nearest_resize(
        image,
        max(1, int(round(image.shape[1] * scale))),
        max(1, int(round(image.shape[0] * scale))),
    )


def _depth_montage(
    stack: np.ndarray,
    raster: ChartRaster,
    offsets: np.ndarray,
    limits: tuple[float, float],
) -> np.ndarray:
    columns = 5
    rows = int(math.ceil(len(stack) / columns))
    rendered = [
        _fit_image(
            _overlay(
                _contrast_plane(plane, raster.mask, limits),
                raster,
                include_boundaries=True,
            ),
            260,
            230,
        )
        for plane in stack
    ]
    panel_height = max(value.shape[0] for value in rendered) + 18
    panel_width = max(value.shape[1] for value in rendered) + 4
    canvas = np.full(
        (rows * panel_height, columns * panel_width, 3),
        (7, 10, 15),
        dtype=np.uint8,
    )
    for index, (image, depth) in enumerate(zip(rendered, offsets)):
        row = index // columns
        column = index % columns
        x = column * panel_width + (panel_width - image.shape[1]) // 2
        y = row * panel_height + 16
        canvas[y : y + image.shape[0], x : x + image.shape[1]] = image
        label = f"D{depth:+g}"
        _draw_text(canvas, column * panel_width + 3, row * panel_height + 4, label, (225, 230, 238))
    return canvas


def _depth_crossings(
    stack: np.ndarray,
    raster: ChartRaster,
    limits: tuple[float, float],
) -> tuple[np.ndarray, dict[str, int]]:
    row = int(np.argmax(np.sum(raster.mask, axis=1)))
    column = int(np.argmax(np.sum(raster.mask, axis=0)))
    horizontal_mask = np.broadcast_to(raster.mask[row][None, :], (len(stack), stack.shape[2]))
    vertical_mask = np.broadcast_to(raster.mask[:, column][None, :], (len(stack), stack.shape[1]))
    horizontal = _contrast_plane(stack[:, row, :], horizontal_mask, limits)
    vertical = _contrast_plane(stack[:, :, column], vertical_mask, limits)
    depth_scale = max(4, int(math.ceil(180 / max(len(stack), 1))))
    horizontal = np.repeat(horizontal, depth_scale, axis=0)
    vertical = np.repeat(vertical, depth_scale, axis=0)
    horizontal_rgb = np.repeat(horizontal[..., None], 3, axis=2)
    vertical_rgb = np.repeat(vertical[..., None], 3, axis=2)
    horizontal_boundaries = np.repeat(
        raster.patch_boundary_mask[row][None, :], horizontal_rgb.shape[0], axis=0
    )
    vertical_boundaries = np.repeat(
        raster.patch_boundary_mask[:, column][None, :], vertical_rgb.shape[0], axis=0
    )
    horizontal_rgb[horizontal_boundaries] = (0, 210, 235)
    vertical_rgb[vertical_boundaries] = (0, 210, 235)
    maximum_width = max(horizontal_rgb.shape[1], vertical_rgb.shape[1])
    canvas = np.full(
        (horizontal_rgb.shape[0] + vertical_rgb.shape[0] + 42, maximum_width, 3),
        (7, 10, 15),
        dtype=np.uint8,
    )
    _draw_text(canvas, 3, 4, f"X R{row}", (225, 230, 238))
    first_y = 17
    canvas[first_y : first_y + horizontal_rgb.shape[0], : horizontal_rgb.shape[1]] = horizontal_rgb
    second_label_y = first_y + horizontal_rgb.shape[0] + 5
    _draw_text(canvas, 3, second_label_y, f"Y C{column}", (225, 230, 238))
    second_y = second_label_y + 13
    canvas[second_y : second_y + vertical_rgb.shape[0], : vertical_rgb.shape[1]] = vertical_rgb
    return canvas, {"horizontalRow": row, "verticalColumn": column}


def _write_bytes(path: Path, payload: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_stack(
    path: Path,
    stack: np.ndarray,
    offsets: np.ndarray,
    raster: ChartRaster,
    mesh: ComponentMesh,
    chart: SurfaceChart,
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            stack=stack,
            depthOffsetsVoxels=offsets.astype(np.float32),
            mask=raster.mask,
            surfaceXYZ=raster.surface_xyz,
            normalXYZ=raster.normal_xyz,
            patchId=raster.patch_id,
            patchBoundaryMask=raster.patch_boundary_mask,
            overlapMask=raster.overlap_mask,
            meshVertexXYZ=mesh.vertex_xyz.astype(np.float32),
            meshTriangle=mesh.triangles,
            meshTrianglePatchId=mesh.triangle_patch_ids,
            chartUV=chart.uv.astype(np.float32),
        )
    temporary.replace(path)


def _overview(
    values: list[tuple[int, int, int, np.ndarray]],
    *,
    tile_size: int = 500,
) -> np.ndarray:
    columns = 2
    rows = int(math.ceil(len(values) / columns))
    header = 24
    canvas = np.full(
        (rows * (tile_size + header), columns * tile_size, 3),
        (7, 10, 15),
        dtype=np.uint8,
    )
    for index, (rank, component_id, patch_count, image) in enumerate(values):
        row = index // columns
        column = index % columns
        fitted = _fit_image(image, tile_size - 8, tile_size - header - 8)
        x = column * tile_size + (tile_size - fitted.shape[1]) // 2
        y = row * (tile_size + header) + header + (tile_size - header - fitted.shape[0]) // 2
        canvas[y : y + fitted.shape[0], x : x + fitted.shape[1]] = fitted
        _draw_text(
            canvas,
            column * tile_size + 6,
            row * (tile_size + header) + 6,
            f"R{rank} C{component_id} N{patch_count}",
            (225, 230, 238),
            scale=2,
        )
    return canvas


def _resolve_source(root: Path) -> tuple[Path, dict[str, Any], VolumeSource]:
    pipeline_root, pipeline = resolve_pipeline_manifest(root)
    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    if source.source_identity["identitySha256"] != source_values["identitySha256"]:
        raise ValueError("native CT source identity changed since reconstruction")
    return pipeline_root, pipeline, source


def _identity(
    root: Path,
    source_resolution_root: Path,
    source: VolumeSource,
    component_ranks: tuple[int, ...],
    depth_offsets: np.ndarray,
    pixel_step_voxels: float,
    maximum_pixels: int,
    maximum_chart_normal_deviation_degrees: float,
    surface_graph_root: Path | None,
    join_refinement_root: Path | None,
    stratigraphic_refinement_root: Path | None,
) -> dict[str, Any]:
    implementation_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": FLATTENING_SCHEMA,
        "version": FLATTENING_VERSION,
        "inputRoot": str(root),
        "sourceResolutionRoot": str(source_resolution_root),
        "inputPatchManifestSha256": sha256_file(root / "selected-patches-v1.json"),
        "inputPatchDataSha256": sha256_file(root / "selected-patches-v1.npz"),
        "sourceIdentitySha256": source.source_identity["identitySha256"],
        "settings": {
            "componentRanks": list(component_ranks),
            "depthOffsetsVoxels": [float(value) for value in depth_offsets],
            "pixelStepVoxels": pixel_step_voxels,
            "maximumPixels": maximum_pixels,
            "maximumChartNormalDeviationDegrees": (
                maximum_chart_normal_deviation_degrees
            ),
            "surface": "exact piecewise-planar cubical patches",
            "depthAlignment": "one fixed offset shared by the complete component",
            "surfaceGraphRoot": (
                str(surface_graph_root) if surface_graph_root is not None else None
            ),
            "joinRefinementRoot": (
                str(join_refinement_root) if join_refinement_root is not None else None
            ),
            "stratigraphicRefinementRoot": (
                str(stratigraphic_refinement_root)
                if stratigraphic_refinement_root is not None
                else None
            ),
        },
        "implementationSha256": {
            "flatten.py": sha256_file(implementation_root / "flatten.py"),
            "block.py": sha256_file(implementation_root / "block.py"),
            "continuity.py": sha256_file(implementation_root / "continuity.py"),
            "stratigraphic_continuity.py": sha256_file(
                implementation_root / "stratigraphic_continuity.py"
            ),
            "geometry.py": sha256_file(implementation_root / "geometry.py"),
            "tables.py": sha256_file(implementation_root / "tables.py"),
            "export.py": sha256_file(implementation_root / "export.py"),
            "surface_graph.py": sha256_file(
                implementation_root / "surface_graph.py"
            ),
            "rectify.py": sha256_file(implementation_root.parent / "rectify.py"),
        },
    }
    if surface_graph_root is not None:
        payload["surfaceGraphSha256"] = {
            "manifest": sha256_file(surface_graph_root / "surface-graph-v1.json"),
            "data": sha256_file(surface_graph_root / "surface-graph-v1.npz"),
        }
    if join_refinement_root is not None:
        payload["joinRefinementSha256"] = {
            "manifest": sha256_file(join_refinement_root / "refinement.json"),
            "table": sha256_file(join_refinement_root / "join-continuity-v1.npz"),
        }
    if stratigraphic_refinement_root is not None:
        payload["stratigraphicRefinementSha256"] = {
            "manifest": sha256_file(
                stratigraphic_refinement_root / "stratigraphic-refinement.json"
            ),
            "table": sha256_file(
                stratigraphic_refinement_root
                / "join-stratigraphic-continuity-v1.npz"
            ),
        }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def run_component_flattening(
    input_root: str | Path,
    output_root: str | Path,
    *,
    component_ranks: Iterable[int] = (1, 2, 3, 7),
    depth_offsets_voxels: Iterable[float] = tuple(range(-12, 13)),
    pixel_step_voxels: float = 2.0,
    maximum_pixels: int = 768,
    maximum_chart_normal_deviation_degrees: float = 40.0,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    source_root: str | Path | None = None,
    surface_graph_root: str | Path | None = None,
    join_refinement_root: str | Path | None = None,
    stratigraphic_refinement_root: str | Path | None = None,
    force: bool = False,
    progress: Any | None = None,
) -> dict[str, Any]:
    """Flatten selected components and sample an honest fixed-depth CT stack."""

    started = time.monotonic()
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("flattening output must differ from reconstruction input")
    ranks = tuple(int(value) for value in component_ranks)
    if not ranks or any(value <= 0 for value in ranks) or len(set(ranks)) != len(ranks):
        raise ValueError("component ranks must be distinct positive integers")
    offsets = np.asarray(tuple(float(value) for value in depth_offsets_voxels))
    if offsets.ndim != 1 or not len(offsets) or np.any(np.diff(offsets) <= 0.0):
        raise ValueError("depth offsets must be a strictly increasing sequence")
    source_resolution_root = (
        Path(source_root).resolve() if source_root is not None else root
    )
    _, pipeline, source = _resolve_source(source_resolution_root)
    refinement_root = (
        Path(join_refinement_root).resolve()
        if join_refinement_root is not None
        else None
    )
    stratigraphic_root = (
        Path(stratigraphic_refinement_root).resolve()
        if stratigraphic_refinement_root is not None
        else None
    )
    graph_root = (
        Path(surface_graph_root).resolve()
        if surface_graph_root is not None
        else root
        if (root / "surface-graph-v1.json").is_file()
        else None
    )
    identity = _identity(
        root,
        source_resolution_root,
        source,
        ranks,
        offsets,
        pixel_step_voxels,
        maximum_pixels,
        maximum_chart_normal_deviation_degrees,
        graph_root,
        refinement_root,
        stratigraphic_root,
    )
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "flattening.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("flattening output belongs to another input or settings identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": FLATTENING_SCHEMA,
        "version": FLATTENING_VERSION,
        "state": "assembling",
        "identity": identity,
        "inputRoot": str(root),
        "source": pipeline["identity"]["source"],
    }
    atomic_json(manifest_path, manifest)

    table = read_patch_shard(root / "selected-patches-v1", verify=True)
    block = (
        read_surface_graph(graph_root, table=table, verify=True)
        if graph_root is not None
        else assemble_surface_hierarchy(
            table.grid,
            BlockBounds((0, 0, 0), table.grid.shape_cells_xyz),
            table.to_patches(),
            maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
        )
    )
    if refinement_root is not None:
        block = apply_join_continuity_refinement(block, refinement_root)
    if stratigraphic_root is not None:
        block = apply_stratigraphic_continuity_refinement(
            block, stratigraphic_root
        )
    ordered_components = sorted(
        block.components, key=lambda value: (-len(value.patch_ids), value.component_id)
    )
    if max(ranks) > len(ordered_components):
        raise ValueError(
            f"component rank {max(ranks)} exceeds {len(ordered_components)} components"
        )

    records = []
    overview_values: list[tuple[int, int, int, np.ndarray]] = []
    clean_overview_values: list[tuple[int, int, int, np.ndarray]] = []
    manifest["state"] = "flattening"
    atomic_json(manifest_path, manifest)
    for index, rank in enumerate(ranks, start=1):
        component = ordered_components[rank - 1]
        if progress is not None:
            progress(index, len(ranks), rank, component.component_id, "chart")
        mesh = component_mesh(
            block,
            component.component_id,
            maximum_chart_normal_deviation_degrees=(
                maximum_chart_normal_deviation_degrees
            ),
        )
        chart = tangent_atlas_chart(mesh)
        raster = rasterize_chart(
            mesh,
            chart,
            pixel_step_voxels=pixel_step_voxels,
            maximum_pixels=maximum_pixels,
        )
        if progress is not None:
            progress(index, len(ranks), rank, component.component_id, "sampling")
        stack, sampling = sample_depth_stack(source, raster, offsets)
        limits = _contrast_limits(stack, raster.mask)
        center_index = int(np.argmin(np.abs(offsets)))
        center_gray = _contrast_plane(stack[center_index], raster.mask, limits)
        center_raw = np.repeat(center_gray[..., None], 3, axis=2)
        center = _overlay(center_gray, raster, include_boundaries=False)
        center_cells = _overlay(center_gray, raster, include_boundaries=True)
        montage = _depth_montage(stack, raster, offsets, limits)
        crossings, crossing_selection = _depth_crossings(stack, raster, limits)

        component_root = output / f"rank-{rank:03d}-component-{component.component_id}"
        component_root.mkdir(parents=True, exist_ok=True)
        _write_stack(component_root / "depth-stack-v1.npz", stack, offsets, raster, mesh, chart)
        _write_bytes(component_root / "center-raw.png", rgb_png(center_raw))
        _write_bytes(component_root / "center.png", rgb_png(center))
        _write_bytes(component_root / "center-with-cells.png", rgb_png(center_cells))
        _write_bytes(component_root / "depth-montage.png", rgb_png(montage))
        _write_bytes(component_root / "depth-crossings.png", rgb_png(crossings))
        record = {
            "rank": rank,
            "componentId": component.component_id,
            "patchCount": len(component.patch_ids),
            "mesh": mesh.statistics,
            "chart": chart.statistics,
            "raster": raster.statistics,
            "sampling": sampling,
            "contrast": {"low": round(limits[0], 4), "high": round(limits[1], 4)},
            "crossingSelection": crossing_selection,
            "artifacts": {
                "stack": str((component_root / "depth-stack-v1.npz").relative_to(output)),
                "centerRaw": str(
                    (component_root / "center-raw.png").relative_to(output)
                ),
                "center": str((component_root / "center.png").relative_to(output)),
                "centerWithCells": str(
                    (component_root / "center-with-cells.png").relative_to(output)
                ),
                "depthMontage": str(
                    (component_root / "depth-montage.png").relative_to(output)
                ),
                "depthCrossings": str(
                    (component_root / "depth-crossings.png").relative_to(output)
                ),
            },
        }
        atomic_json(component_root / "manifest.json", record)
        records.append(record)
        overview_values.append(
            (rank, component.component_id, len(component.patch_ids), center_cells)
        )
        clean_overview_values.append(
            (rank, component.component_id, len(component.patch_ids), center_raw)
        )
    overview = _overview(overview_values)
    clean_overview = _overview(clean_overview_values)
    _write_bytes(output / "overview.png", rgb_png(overview))
    _write_bytes(output / "overview-clean.png", rgb_png(clean_overview))
    summary = {
        "schema": "pareidolia.cubical-component-flattening-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(root),
        "surfaceGraphRoot": str(graph_root) if graph_root is not None else None,
        "joinRefinementRoot": (
            str(refinement_root) if refinement_root is not None else None
        ),
        "stratigraphicRefinementRoot": (
            str(stratigraphic_root) if stratigraphic_root is not None else None
        ),
        "directions": "all surface normals remain axial; depth montage sign is a chart gauge",
        "surfaceSampling": "exact piecewise-planar patches with one fixed component-wide depth offset",
        "components": records,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "overview": "overview.png",
            "cleanOverview": "overview-clean.png",
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
