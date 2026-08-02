from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .needle_surface import (
    integrate_intrinsic_surface_charts,
    triangulate_intrinsic_surface_charts,
)
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
)


PHYSICAL_MID_SURFACE_MESH_SCHEMA = "pareidolia.physical-mid-surface-mesh"
PHYSICAL_MID_SURFACE_MESH_VERSION = 1
PHYSICAL_MID_SURFACE_MESH_STEM = "physical-mid-surface-mesh-v1"


@dataclass(frozen=True, slots=True)
class PhysicalMidSurfaceMeshSettings:
    """Dataset-independent controls for meshing physical midpoint graphs."""

    minimum_source_component_nodes: int = 32
    maximum_source_components: int = 24
    minimum_mesh_component_nodes: int = 16
    maximum_mesh_components: int = 32
    maximum_oriented_neighbor_normal_degrees: float = 35.0
    robust_chart_iterations: int = 3
    chart_huber_delta_voxels: float = 1.0
    maximum_mesh_edge_residual_voxels: float = 1.5
    minimum_chart_separation_voxels: float = 0.15
    # Match the three-sampling-step continuation radius used by the physical
    # boundary graph.  The previous shorter triangulation radius converted
    # valid graph continuity into artificial holes in the rendered sheet.
    maximum_local_closure_edge_voxels: float = 6.0
    maximum_local_closure_height_voxels: float = 1.5
    maximum_local_closure_normal_degrees: float = 25.0
    maximum_triangle_edge_voxels: float = 7.0
    maximum_triangle_normal_residual_degrees: float = 35.0
    minimum_triangle_area_voxels_squared: float = 0.25
    chart_solver_relative_tolerance: float = 1.0e-7
    chart_solver_maximum_iterations: int = 2048
    triangulation_mode: str = "graph-cliques"

    def __post_init__(self) -> None:
        if self.triangulation_mode not in (
            "graph-cliques",
            "chart-delaunay",
            "local-fans",
        ):
            raise ValueError(
                "mid-surface triangulation mode must be graph-cliques, "
                "chart-delaunay, or local-fans"
            )
        integer_values = (
            self.minimum_source_component_nodes,
            self.maximum_source_components,
            self.minimum_mesh_component_nodes,
            self.maximum_mesh_components,
            self.robust_chart_iterations,
            self.chart_solver_maximum_iterations,
        )
        if any(value < 1 for value in integer_values):
            raise ValueError("mid-surface mesh integer settings must be positive")
        if not 0.0 < self.maximum_oriented_neighbor_normal_degrees < 89.0:
            raise ValueError("oriented normal gate must lie in (0, 89) degrees")
        if not 0.0 < self.maximum_triangle_normal_residual_degrees < 89.0:
            raise ValueError("triangle normal gate must lie in (0, 89) degrees")
        if not 0.0 < self.maximum_local_closure_normal_degrees < 89.0:
            raise ValueError("local closure normal gate must lie in (0, 89) degrees")
        positive = (
            self.chart_huber_delta_voxels,
            self.maximum_mesh_edge_residual_voxels,
            self.minimum_chart_separation_voxels,
            self.maximum_local_closure_edge_voxels,
            self.maximum_local_closure_height_voxels,
            self.maximum_triangle_edge_voxels,
            self.minimum_triangle_area_voxels_squared,
            self.chart_solver_relative_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("mid-surface mesh scales must be finite and positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            following = int(self.parent[value])
            self.parent[value] = root
            value = following
        return root

    def union(self, first: int, second: int) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return False
        if self.size[first_root] < self.size[second_root]:
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.size[first_root] += self.size[second_root]
        return True


def _normalized(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)


def _percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    quantiles = np.percentile(finite, (0, 50, 90, 99, 100))
    return {
        "count": int(len(finite)),
        **{
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "median", "p90", "p99", "maximum"), quantiles
            )
        },
    }


def _resolve_mid_surface(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{PHYSICAL_MID_SURFACE_STEM}.json"


def _load_mid_surface(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = _resolve_mid_surface(root)
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_MID_SURFACE_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("meshing requires a complete physical mid-surface catalog")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("physical mid-surface data hash differs from its manifest")
    required = (
        "midpointXYZ",
        "normalXYZ",
        "physicalSheetLabel",
        "componentId",
        "edgeFirstNode",
        "edgeSecondNode",
        "edgeScore",
    )
    with np.load(data_path, allow_pickle=False) as stored:
        missing = set(required) - set(stored.files)
        if missing:
            raise ValueError(f"physical mid-surface is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return manifest_path, manifest, arrays


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _minimum_rotation(
    value: np.ndarray,
    first_normal: np.ndarray,
    second_normal: np.ndarray,
) -> np.ndarray:
    """Parallel transport a tangent vector by the shortest normal rotation."""

    cosine = float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))
    cross = np.cross(first_normal, second_normal)
    if cosine >= 1.0 - 1.0e-12:
        rotated = np.asarray(value, dtype=np.float64).copy()
    elif cosine <= -1.0 + 1.0e-8:
        # This branch should not occur after the unsigned normal gauge has been
        # resolved, but a deterministic tangent projection is safer than NaNs.
        rotated = value - float(np.dot(value, second_normal)) * second_normal
    else:
        rotated = (
            value
            + np.cross(cross, value)
            + np.cross(cross, np.cross(cross, value)) / (1.0 + cosine)
        )
    rotated -= float(np.dot(rotated, second_normal)) * second_normal
    length = float(np.linalg.norm(rotated))
    if length <= 1.0e-10:
        axis = np.eye(3)[int(np.argmin(np.abs(second_normal)))]
        rotated = axis - float(np.dot(axis, second_normal)) * second_normal
        length = float(np.linalg.norm(rotated))
    return rotated / max(length, 1.0e-12)


def transport_mid_surface_frames(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    selected_node: np.ndarray,
) -> dict[str, np.ndarray]:
    """Resolve unsigned normals and transport one tangent gauge per fragment.

    A maximum-quality spanning forest defines gauge transport only. All graph
    edges remain independent observations and are later checked against the
    transported gauge, so a contradictory cycle cannot silently twist a chart.
    """

    center = np.asarray(center_xyz, dtype=np.float64)
    normal = _normalized(normal_xyz)
    edge_first = np.asarray(first, dtype=np.int32)
    edge_second = np.asarray(second, dtype=np.int32)
    score = np.asarray(edge_score, dtype=np.float64)
    selected = np.asarray(selected_node, dtype=bool)
    if center.shape != normal.shape or center.ndim != 2 or center.shape[1] != 3:
        raise ValueError("mid-surface centers and normals must have shape (N, 3)")
    if selected.shape != (len(center),):
        raise ValueError("selected-node mask must match midpoint nodes")
    if any(len(value) != len(edge_first) for value in (edge_second, score)):
        raise ValueError("mid-surface edge arrays are not aligned")
    if np.any(edge_first < 0) or np.any(edge_second >= len(center)):
        raise ValueError("mid-surface edge endpoints leave the node table")

    selected_edge = selected[edge_first] & selected[edge_second]
    raw_cosine = np.einsum("ij,ij->i", normal[edge_first], normal[edge_second])
    distance = np.linalg.norm(center[edge_second] - center[edge_first], axis=1)
    quality = score * (0.25 + 0.75 * np.abs(raw_cosine)) / np.maximum(distance, 0.25)
    eligible = np.flatnonzero(selected_edge)
    order = eligible[
        np.lexsort((eligible, edge_second[eligible], edge_first[eligible], -quality[eligible]))
    ]
    forest = _DisjointSet(len(center))
    tree_edge = np.zeros(len(edge_first), dtype=bool)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(len(center))]
    for edge_index in order:
        left = int(edge_first[edge_index])
        right = int(edge_second[edge_index])
        if not forest.union(left, right):
            continue
        tree_edge[edge_index] = True
        adjacency[left].append((right, int(edge_index)))
        adjacency[right].append((left, int(edge_index)))

    normal_sign = np.zeros(len(center), dtype=np.int8)
    tangent_u = np.full((len(center), 3), np.nan, dtype=np.float64)
    tangent_v = np.full((len(center), 3), np.nan, dtype=np.float64)
    for root in np.flatnonzero(selected):
        if normal_sign[root] != 0:
            continue
        normal_sign[root] = 1
        root_normal = normal[root]
        axis = np.eye(3)[int(np.argmin(np.abs(root_normal)))]
        root_u = axis - float(np.dot(axis, root_normal)) * root_normal
        root_u /= max(float(np.linalg.norm(root_u)), 1.0e-12)
        tangent_u[root] = root_u
        tangent_v[root] = np.cross(root_normal, root_u)
        queue = [int(root)]
        cursor = 0
        while cursor < len(queue):
            parent = queue[cursor]
            cursor += 1
            parent_normal = normal[parent] * float(normal_sign[parent])
            for child, edge_index in adjacency[parent]:
                if normal_sign[child] != 0:
                    continue
                relation = 1 if raw_cosine[edge_index] >= 0.0 else -1
                normal_sign[child] = np.int8(int(normal_sign[parent]) * relation)
                child_normal = normal[child] * float(normal_sign[child])
                child_u = _minimum_rotation(
                    tangent_u[parent], parent_normal, child_normal
                )
                tangent_u[child] = child_u
                tangent_v[child] = np.cross(child_normal, child_u)
                tangent_v[child] /= max(
                    float(np.linalg.norm(tangent_v[child])), 1.0e-12
                )
                queue.append(child)

    signed_normal = normal * normal_sign[:, None]
    signed_cosine = np.full(len(edge_first), np.nan, dtype=np.float64)
    signed_cosine[selected_edge] = np.einsum(
        "ij,ij->i",
        signed_normal[edge_first[selected_edge]],
        signed_normal[edge_second[selected_edge]],
    )
    return {
        "normalSign": normal_sign,
        "signedNormalXYZ": signed_normal.astype(np.float32),
        "tangentUXYZ": tangent_u.astype(np.float32),
        "tangentVXYZ": tangent_v.astype(np.float32),
        "treeEdge": tree_edge,
        "selectedEdge": selected_edge,
        "rawNormalCosine": raw_cosine.astype(np.float32),
        "signedNormalCosine": signed_cosine.astype(np.float32),
    }


def _ranked_node_components(
    selected_node: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    selected_edge: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    selected = np.asarray(selected_node, dtype=bool)
    edge_mask = np.asarray(selected_edge, dtype=bool)
    forest = _DisjointSet(len(selected))
    for left, right in zip(first[edge_mask], second[edge_mask]):
        forest.union(int(left), int(right))
    nodes = np.flatnonzero(selected)
    roots = np.asarray([forest.find(int(node)) for node in nodes], dtype=np.int32)
    root_value, root_count = np.unique(roots, return_counts=True)
    order = np.lexsort((root_value, -root_count))
    rank_by_root = np.full(len(selected), -1, dtype=np.int32)
    rank_by_root[root_value[order]] = np.arange(len(root_value), dtype=np.int32)
    component = np.full(len(selected), -1, dtype=np.int32)
    component[nodes] = rank_by_root[roots]
    component_size = np.zeros(len(selected), dtype=np.int32)
    component_size[nodes] = root_count[order][component[nodes]]
    return component, component_size, root_count[order].astype(np.int32)


def _triangle_components(
    triangles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, int]]:
    triangle_values = np.asarray(triangles, dtype=np.int32)
    if not len(triangle_values):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            {"meshEdgeCount": 0, "boundaryEdgeCount": 0, "nonmanifoldEdgeCount": 0},
        )
    forest = _DisjointSet(len(triangle_values))
    owner: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(triangle_values):
        for index, left in enumerate(triangle):
            right = int(triangle[(index + 1) % 3])
            edge = (min(int(left), right), max(int(left), right))
            owner.setdefault(edge, []).append(triangle_index)
    for values in owner.values():
        for following in values[1:]:
            forest.union(values[0], following)
    roots = np.asarray(
        [forest.find(index) for index in range(len(triangle_values))],
        dtype=np.int32,
    )
    root_value, root_count = np.unique(roots, return_counts=True)
    order = np.lexsort((root_value, -root_count))
    rank_by_root = np.full(len(triangle_values), -1, dtype=np.int32)
    rank_by_root[root_value[order]] = np.arange(len(root_value), dtype=np.int32)
    component = rank_by_root[roots]
    return component, root_count[order].astype(np.int32), {
        "meshEdgeCount": len(owner),
        "boundaryEdgeCount": sum(len(values) == 1 for values in owner.values()),
        "nonmanifoldEdgeCount": sum(len(values) > 2 for values in owner.values()),
    }


def _chart_orientation_filter(
    triangles: np.ndarray,
    chart_uv: np.ndarray,
    source_component: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    triangle_values = np.asarray(triangles, dtype=np.int32)
    if not len(triangle_values):
        return np.empty(0, dtype=bool), np.empty(0, dtype=np.float32)
    uv = np.asarray(chart_uv, dtype=np.float64)
    first = uv[triangle_values[:, 1]] - uv[triangle_values[:, 0]]
    second = uv[triangle_values[:, 2]] - uv[triangle_values[:, 0]]
    signed_area = 0.5 * (first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0])
    retained = np.zeros(len(triangle_values), dtype=bool)
    for component in np.unique(source_component):
        member = np.flatnonzero(source_component == component)
        orientation = 1.0 if float(np.sum(signed_area[member])) >= 0.0 else -1.0
        retained[member] = orientation * signed_area[member] > 1.0e-8
    return retained, signed_area.astype(np.float32)


def _triangle_geometry(
    nodes: tuple[int, int, int],
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
) -> tuple[tuple[int, int, int], float, float, float]:
    points = center_xyz[np.asarray(nodes)]
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    norm = float(np.linalg.norm(cross))
    if norm <= 1.0e-12:
        return nodes, 0.0, 90.0, float("inf")
    triangle_normal = cross / norm
    reference = np.sum(signed_normal_xyz[np.asarray(nodes)], axis=0)
    reference /= max(float(np.linalg.norm(reference)), 1.0e-12)
    oriented = nodes
    if float(np.dot(triangle_normal, reference)) < 0.0:
        oriented = (nodes[0], nodes[2], nodes[1])
        triangle_normal *= -1.0
    residual = math.degrees(
        math.acos(
            np.clip(float(np.dot(triangle_normal, reference)), -1.0, 1.0)
        )
    )
    edges = (
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[0] - points[2])),
    )
    return oriented, 0.5 * norm, residual, max(edges)


def _separated_chart_nodes(
    chart_uv: np.ndarray,
    component_id: np.ndarray,
    component_size: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    edge_mask: np.ndarray,
    *,
    minimum_component_nodes: int,
    minimum_separation_voxels: float,
) -> tuple[np.ndarray, int]:
    """Keep the strongest representative of near-coincident chart samples."""

    uv = np.asarray(chart_uv, dtype=np.float64)
    component = np.asarray(component_id, dtype=np.int32)
    size = np.asarray(component_size, dtype=np.int32)
    active = (
        (component >= 0)
        & (size >= minimum_component_nodes)
        & np.all(np.isfinite(uv), axis=1)
    )
    support = np.bincount(
        np.concatenate((first[edge_mask], second[edge_mask])),
        weights=np.concatenate((edge_score[edge_mask], edge_score[edge_mask])),
        minlength=len(uv),
    )
    retained = np.zeros(len(uv), dtype=bool)
    rejected = 0
    for value in np.unique(component[active]):
        nodes = np.flatnonzero(active & (component == value))
        order = nodes[np.lexsort((nodes, -support[nodes]))]
        bins: dict[tuple[int, int], list[int]] = {}
        for node in order:
            key = tuple(
                np.floor(uv[node] / minimum_separation_voxels).astype(int)
            )
            duplicate = False
            for offset_x in (-1, 0, 1):
                for offset_y in (-1, 0, 1):
                    for prior in bins.get(
                        (key[0] + offset_x, key[1] + offset_y), ()
                    ):
                        if (
                            float(np.linalg.norm(uv[node] - uv[prior]))
                            < minimum_separation_voxels
                        ):
                            duplicate = True
                            break
                    if duplicate:
                        break
                if duplicate:
                    break
            if duplicate:
                rejected += 1
                continue
            retained[node] = True
            bins.setdefault(key, []).append(int(node))
    return retained, rejected


def _orientation(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    return float(
        (second[0] - first[0]) * (third[1] - first[1])
        - (second[1] - first[1]) * (third[0] - first[0])
    )


def _proper_segment_crossing(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
) -> bool:
    first_left = _orientation(first_start, first_stop, second_start)
    first_right = _orientation(first_start, first_stop, second_stop)
    second_left = _orientation(second_start, second_stop, first_start)
    second_right = _orientation(second_start, second_stop, first_stop)
    epsilon = 1.0e-9
    return (
        first_left * first_right < -epsilon
        and second_left * second_right < -epsilon
    )


def _strictly_inside_triangle(point: np.ndarray, triangle: np.ndarray) -> bool:
    orientation = np.asarray(
        [
            _orientation(triangle[index], triangle[(index + 1) % 3], point)
            for index in range(3)
        ]
    )
    epsilon = 1.0e-9
    return bool(np.all(orientation > epsilon) or np.all(orientation < -epsilon))


def _positive_area_overlap(
    first_nodes: tuple[int, int, int],
    second_nodes: tuple[int, int, int],
    chart_uv: np.ndarray,
) -> bool:
    shared = set(first_nodes) & set(second_nodes)
    first_triangle = chart_uv[np.asarray(first_nodes)]
    second_triangle = chart_uv[np.asarray(second_nodes)]
    for first_index, first_node in enumerate(first_nodes):
        first_following = first_nodes[(first_index + 1) % 3]
        for second_index, second_node in enumerate(second_nodes):
            second_following = second_nodes[(second_index + 1) % 3]
            if {first_node, first_following} & {second_node, second_following}:
                continue
            if _proper_segment_crossing(
                first_triangle[first_index],
                first_triangle[(first_index + 1) % 3],
                second_triangle[second_index],
                second_triangle[(second_index + 1) % 3],
            ):
                return True
    for index, node in enumerate(first_nodes):
        if node not in shared and _strictly_inside_triangle(
            first_triangle[index], second_triangle
        ):
            return True
    for index, node in enumerate(second_nodes):
        if node not in shared and _strictly_inside_triangle(
            second_triangle[index], first_triangle
        ):
            return True
    return False


def triangulate_graph_supported_charts(
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    chart_uv: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    edge_mask: np.ndarray,
    component_id: np.ndarray,
    component_size: np.ndarray,
    *,
    minimum_component_nodes: int,
    minimum_chart_separation_voxels: float,
    maximum_local_closure_edge_voxels: float,
    maximum_local_closure_height_voxels: float,
    maximum_local_closure_normal_degrees: float,
    maximum_edge_voxels: float,
    maximum_normal_residual_degrees: float,
    minimum_area_voxels_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Select a planar mesh from local three-edge graph cliques.

    Small supported triangles are considered before larger ones. A chart-space
    spatial index rejects positive-area overlap, so the selected mesh remains
    planar and manifold without an all-pairs Delaunay construction.
    """

    center = np.asarray(center_xyz, dtype=np.float64)
    normal = np.asarray(signed_normal_xyz, dtype=np.float64)
    uv = np.asarray(chart_uv, dtype=np.float64)
    edge_first = np.asarray(first, dtype=np.int32)
    edge_second = np.asarray(second, dtype=np.int32)
    score = np.asarray(edge_score, dtype=np.float64)
    selected_edge = np.asarray(edge_mask, dtype=bool)
    component = np.asarray(component_id, dtype=np.int32)
    size = np.asarray(component_size, dtype=np.int32)
    retained_node, rejected_duplicate = _separated_chart_nodes(
        uv,
        component,
        size,
        edge_first,
        edge_second,
        score,
        selected_edge,
        minimum_component_nodes=minimum_component_nodes,
        minimum_separation_voxels=minimum_chart_separation_voxels,
    )
    selected_edge &= retained_node[edge_first] & retained_node[edge_second]
    score_by_pair: dict[tuple[int, int], float] = {}
    neighbor: dict[int, set[int]] = {}
    for edge_index in np.flatnonzero(selected_edge):
        left = int(edge_first[edge_index])
        right = int(edge_second[edge_index])
        if left > right:
            left, right = right, left
        pair = (left, right)
        prior_score = score_by_pair.get(pair)
        if prior_score is not None and prior_score >= score[edge_index]:
            continue
        score_by_pair[pair] = float(score[edge_index])
        neighbor.setdefault(left, set()).add(right)
        neighbor.setdefault(right, set()).add(left)

    # Persisted graph edges establish fragment identity. Once that identity and
    # its intrinsic chart are fixed, short missing triangle sides may be closed
    # without changing connectivity. Both chart and 3D geometry must agree.
    closure_edge_count = 0
    closure_bin_size = max(maximum_local_closure_edge_voxels, 1.0e-6)
    closure_cosine = math.cos(math.radians(maximum_local_closure_normal_degrees))
    active_nodes = np.flatnonzero(retained_node)
    bins: dict[tuple[int, int, int], list[int]] = {}
    for node in active_nodes:
        key_xy = np.floor(uv[node] / closure_bin_size).astype(int)
        key = (int(component[node]), int(key_xy[0]), int(key_xy[1]))
        bins.setdefault(key, []).append(int(node))
    for left in active_nodes:
        key_xy = np.floor(uv[left] / closure_bin_size).astype(int)
        component_value = int(component[left])
        candidates: set[int] = set()
        for offset_x in (-1, 0, 1):
            for offset_y in (-1, 0, 1):
                candidates.update(
                    bins.get(
                        (
                            component_value,
                            int(key_xy[0]) + offset_x,
                            int(key_xy[1]) + offset_y,
                        ),
                        (),
                    )
                )
        for right in sorted(value for value in candidates if value > left):
            pair = (int(left), int(right))
            if pair in score_by_pair:
                continue
            chart_distance = float(np.linalg.norm(uv[right] - uv[left]))
            if chart_distance > maximum_local_closure_edge_voxels:
                continue
            delta = center[right] - center[left]
            physical_distance = float(np.linalg.norm(delta))
            if physical_distance > maximum_local_closure_edge_voxels:
                continue
            normal_cosine = float(np.dot(normal[left], normal[right]))
            if normal_cosine < closure_cosine:
                continue
            common_normal = normal[left] + normal[right]
            common_normal /= max(float(np.linalg.norm(common_normal)), 1.0e-12)
            height = abs(float(np.dot(delta, common_normal)))
            if height > maximum_local_closure_height_voxels:
                continue
            closure_score = math.exp(
                -0.5 * (height / maximum_local_closure_height_voxels) ** 2
                -0.5
                * (chart_distance / maximum_local_closure_edge_voxels) ** 2
            )
            score_by_pair[pair] = closure_score
            neighbor.setdefault(int(left), set()).add(int(right))
            neighbor.setdefault(int(right), set()).add(int(left))
            closure_edge_count += 1

    candidate: list[dict[str, Any]] = []
    rejected_geometry = 0
    for left, middle in score_by_pair:
        if component[left] != component[middle]:
            continue
        common = neighbor.get(left, set()) & neighbor.get(middle, set())
        for right in sorted(value for value in common if value > middle):
            second_score = score_by_pair.get((left, right))
            third_score = score_by_pair.get((middle, right))
            if second_score is None or third_score is None:
                continue
            nodes, area, residual, maximum_edge = _triangle_geometry(
                (left, middle, right), center, normal
            )
            if (
                area < minimum_area_voxels_squared
                or residual > maximum_normal_residual_degrees
                or maximum_edge > maximum_edge_voxels
            ):
                rejected_geometry += 1
                continue
            points_uv = uv[np.asarray(nodes)]
            signed_chart_area = 0.5 * _orientation(
                points_uv[0], points_uv[1], points_uv[2]
            )
            if abs(signed_chart_area) <= 1.0e-8:
                rejected_geometry += 1
                continue
            edge_lengths = (
                float(np.linalg.norm(center[middle] - center[left])),
                float(np.linalg.norm(center[right] - center[left])),
                float(np.linalg.norm(center[right] - center[middle])),
            )
            circumradius = (
                edge_lengths[0]
                * edge_lengths[1]
                * edge_lengths[2]
                / max(4.0 * area, 1.0e-12)
            )
            minimum_score = min(
                float(score_by_pair[(left, middle)]),
                float(second_score),
                float(third_score),
            )
            candidate.append(
                {
                    "nodes": nodes,
                    "area": area,
                    "normalResidual": residual,
                    "chartSignedArea": signed_chart_area,
                    "component": int(component[left]),
                    "circumradius": circumradius,
                    "minimumEdgeScore": minimum_score,
                }
            )

    orientation_by_component: dict[int, float] = {}
    for value in {int(item["component"]) for item in candidate}:
        signed_area = sum(
            float(item["chartSignedArea"])
            for item in candidate
            if int(item["component"]) == value
        )
        orientation_by_component[value] = 1.0 if signed_area >= 0.0 else -1.0
    orientation_candidate = [
        item
        for item in candidate
        if orientation_by_component[int(item["component"])]
        * float(item["chartSignedArea"])
        > 0.0
    ]
    rejected_orientation = len(candidate) - len(orientation_candidate)
    ordered = sorted(
        orientation_candidate,
        key=lambda item: (
            float(item["circumradius"]),
            -float(item["minimumEdgeScore"]),
            -float(item["area"]),
            tuple(int(value) for value in item["nodes"]),
        ),
    )

    accepted: list[dict[str, Any]] = []
    accepted_by_bin: dict[tuple[int, int, int], list[int]] = {}
    edge_incidence: dict[tuple[int, int], int] = {}
    bin_size = max(float(maximum_edge_voxels), 1.0)
    rejected_overlap = 0
    rejected_manifold = 0
    for item in ordered:
        nodes = tuple(int(value) for value in item["nodes"])
        component_value = int(item["component"])
        edges = [
            (min(nodes[index], nodes[(index + 1) % 3]), max(nodes[index], nodes[(index + 1) % 3]))
            for index in range(3)
        ]
        if any(edge_incidence.get(edge, 0) >= 2 for edge in edges):
            rejected_manifold += 1
            continue
        points = uv[np.asarray(nodes)]
        low = np.floor(np.min(points, axis=0) / bin_size).astype(int)
        high = np.floor(np.max(points, axis=0) / bin_size).astype(int)
        nearby: set[int] = set()
        for x_value in range(int(low[0]), int(high[0]) + 1):
            for y_value in range(int(low[1]), int(high[1]) + 1):
                nearby.update(
                    accepted_by_bin.get(
                        (component_value, x_value, y_value), ()
                    )
                )
        if any(
            _positive_area_overlap(nodes, accepted[index]["nodes"], uv)
            for index in nearby
        ):
            rejected_overlap += 1
            continue
        accepted_index = len(accepted)
        item["nodes"] = nodes
        accepted.append(item)
        for edge in edges:
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
        for x_value in range(int(low[0]), int(high[0]) + 1):
            for y_value in range(int(low[1]), int(high[1]) + 1):
                accepted_by_bin.setdefault(
                    (component_value, x_value, y_value), []
                ).append(accepted_index)

    triangles = np.asarray(
        [item["nodes"] for item in accepted], dtype=np.int32
    ).reshape((-1, 3))
    return (
        triangles,
        np.asarray([item["area"] for item in accepted], dtype=np.float32),
        np.asarray(
            [item["normalResidual"] for item in accepted], dtype=np.float32
        ),
        np.asarray(
            [item["chartSignedArea"] for item in accepted], dtype=np.float32
        ),
        {
            "components": int(
                len(np.unique(component[retained_node]))
                if np.any(retained_node)
                else 0
            ),
            "retainedChartNodes": int(np.count_nonzero(retained_node)),
            "rejectedNearDuplicateChartNodes": rejected_duplicate,
            "localClosureEdgeCount": closure_edge_count,
            "triangulationEdgeCount": len(score_by_pair),
            "supportedThreeEdgeCliques": len(candidate) + rejected_geometry,
            "rejectedTriangleGeometry": rejected_geometry,
            "rejectedChartOrientation": rejected_orientation,
            "rejectedPlanarOverlap": rejected_overlap,
            "rejectedManifoldIncidence": rejected_manifold,
            "retainedTriangles": len(accepted),
        },
    )


def triangulate_local_chart_fans(
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    chart_uv: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    edge_mask: np.ndarray,
    component_id: np.ndarray,
    component_size: np.ndarray,
    *,
    minimum_component_nodes: int,
    minimum_chart_separation_voxels: float,
    maximum_edge_voxels: float,
    maximum_normal_residual_degrees: float,
    minimum_area_voxels_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Triangulate dense local graphs from consecutive chart-space neighbors.

    Boundary-track graphs have many redundant short chords. Enumerating every
    three-edge clique is unnecessarily cubic in local degree, while a generic
    incremental Delaunay solve ignores the acquisition lattice and scales
    poorly in pure Python. Around a sampled surface point, only consecutive
    neighbors in angular chart order can bound a local face. Requiring the
    closing graph edge retains the measured topology; a final planar/manifold
    selection removes chords that are inconsistent with that embedding.
    """

    center = np.asarray(center_xyz, dtype=np.float64)
    normal = np.asarray(signed_normal_xyz, dtype=np.float64)
    uv = np.asarray(chart_uv, dtype=np.float64)
    edge_first = np.asarray(first, dtype=np.int32)
    edge_second = np.asarray(second, dtype=np.int32)
    score = np.asarray(edge_score, dtype=np.float64)
    selected_edge = np.asarray(edge_mask, dtype=bool)
    component = np.asarray(component_id, dtype=np.int32)
    size = np.asarray(component_size, dtype=np.int32)
    retained_node, rejected_duplicate = _separated_chart_nodes(
        uv,
        component,
        size,
        edge_first,
        edge_second,
        score,
        selected_edge,
        minimum_component_nodes=minimum_component_nodes,
        minimum_separation_voxels=minimum_chart_separation_voxels,
    )
    selected_edge &= retained_node[edge_first] & retained_node[edge_second]
    score_by_pair: dict[tuple[int, int], float] = {}
    neighbor: dict[int, set[int]] = {}
    for edge_index in np.flatnonzero(selected_edge):
        left = int(edge_first[edge_index])
        right = int(edge_second[edge_index])
        if left > right:
            left, right = right, left
        pair = (left, right)
        prior = score_by_pair.get(pair)
        if prior is not None and prior >= score[edge_index]:
            continue
        score_by_pair[pair] = float(score[edge_index])
        neighbor.setdefault(left, set()).add(right)
        neighbor.setdefault(right, set()).add(left)

    triangle_keys: set[tuple[int, int, int]] = set()
    angular_wedge_count = 0
    rejected_missing_closure = 0
    rejected_wide_wedge = 0
    for middle, adjacent_set in neighbor.items():
        adjacent = np.asarray(sorted(adjacent_set), dtype=np.int32)
        if len(adjacent) < 2:
            continue
        delta = uv[adjacent] - uv[middle]
        angle = np.arctan2(delta[:, 1], delta[:, 0])
        order = np.lexsort((adjacent, angle))
        adjacent = adjacent[order]
        angle = angle[order]
        for index, left_value in enumerate(adjacent):
            right_value = int(adjacent[(index + 1) % len(adjacent)])
            left = int(left_value)
            gap = float(
                (angle[(index + 1) % len(angle)] - angle[index])
                % (2.0 * math.pi)
            )
            angular_wedge_count += 1
            if gap >= math.pi - 1.0e-9:
                rejected_wide_wedge += 1
                continue
            closing = (min(left, right_value), max(left, right_value))
            if closing not in score_by_pair:
                rejected_missing_closure += 1
                continue
            triangle_keys.add(tuple(sorted((middle, left, right_value))))

    candidate: list[dict[str, Any]] = []
    rejected_geometry = 0
    for nodes_key in sorted(triangle_keys):
        nodes, area, residual, maximum_edge = _triangle_geometry(
            nodes_key, center, normal
        )
        if (
            area < minimum_area_voxels_squared
            or residual > maximum_normal_residual_degrees
            or maximum_edge > maximum_edge_voxels
        ):
            rejected_geometry += 1
            continue
        points_uv = uv[np.asarray(nodes)]
        signed_chart_area = 0.5 * _orientation(
            points_uv[0], points_uv[1], points_uv[2]
        )
        if abs(signed_chart_area) <= 1.0e-8:
            rejected_geometry += 1
            continue
        edge_lengths = (
            float(np.linalg.norm(center[nodes[1]] - center[nodes[0]])),
            float(np.linalg.norm(center[nodes[2]] - center[nodes[1]])),
            float(np.linalg.norm(center[nodes[0]] - center[nodes[2]])),
        )
        circumradius = (
            edge_lengths[0]
            * edge_lengths[1]
            * edge_lengths[2]
            / max(4.0 * area, 1.0e-12)
        )
        minimum_score = min(
            score_by_pair[(min(nodes[0], nodes[1]), max(nodes[0], nodes[1]))],
            score_by_pair[(min(nodes[1], nodes[2]), max(nodes[1], nodes[2]))],
            score_by_pair[(min(nodes[2], nodes[0]), max(nodes[2], nodes[0]))],
        )
        candidate.append(
            {
                "nodes": nodes,
                "area": area,
                "normalResidual": residual,
                "chartSignedArea": signed_chart_area,
                "component": int(component[nodes[0]]),
                "circumradius": circumradius,
                "minimumEdgeScore": minimum_score,
            }
        )

    orientation_by_component: dict[int, float] = {}
    for value in {int(item["component"]) for item in candidate}:
        signed_area = sum(
            float(item["chartSignedArea"])
            for item in candidate
            if int(item["component"]) == value
        )
        orientation_by_component[value] = 1.0 if signed_area >= 0.0 else -1.0
    oriented = [
        item
        for item in candidate
        if orientation_by_component[int(item["component"])]
        * float(item["chartSignedArea"])
        > 0.0
    ]
    rejected_orientation = len(candidate) - len(oriented)
    ordered = sorted(
        oriented,
        key=lambda item: (
            float(item["circumradius"]),
            -float(item["minimumEdgeScore"]),
            -float(item["area"]),
            tuple(int(value) for value in item["nodes"]),
        ),
    )

    accepted: list[dict[str, Any]] = []
    accepted_by_bin: dict[tuple[int, int, int], list[int]] = {}
    edge_incidence: dict[tuple[int, int], int] = {}
    bin_size = max(float(maximum_edge_voxels), 1.0)
    rejected_overlap = 0
    rejected_manifold = 0
    for item in ordered:
        nodes = tuple(int(value) for value in item["nodes"])
        component_value = int(item["component"])
        edges = [
            (
                min(nodes[index], nodes[(index + 1) % 3]),
                max(nodes[index], nodes[(index + 1) % 3]),
            )
            for index in range(3)
        ]
        if any(edge_incidence.get(edge, 0) >= 2 for edge in edges):
            rejected_manifold += 1
            continue
        points = uv[np.asarray(nodes)]
        low = np.floor(np.min(points, axis=0) / bin_size).astype(int)
        high = np.floor(np.max(points, axis=0) / bin_size).astype(int)
        nearby: set[int] = set()
        for x_value in range(int(low[0]), int(high[0]) + 1):
            for y_value in range(int(low[1]), int(high[1]) + 1):
                nearby.update(
                    accepted_by_bin.get(
                        (component_value, x_value, y_value), ()
                    )
                )
        if any(
            _positive_area_overlap(nodes, accepted[index]["nodes"], uv)
            for index in nearby
        ):
            rejected_overlap += 1
            continue
        accepted_index = len(accepted)
        accepted.append(item)
        for edge in edges:
            edge_incidence[edge] = edge_incidence.get(edge, 0) + 1
        for x_value in range(int(low[0]), int(high[0]) + 1):
            for y_value in range(int(low[1]), int(high[1]) + 1):
                accepted_by_bin.setdefault(
                    (component_value, x_value, y_value), []
                ).append(accepted_index)

    triangles = np.asarray(
        [item["nodes"] for item in accepted], dtype=np.int32
    ).reshape((-1, 3))
    return (
        triangles,
        np.asarray([item["area"] for item in accepted], dtype=np.float32),
        np.asarray(
            [item["normalResidual"] for item in accepted], dtype=np.float32
        ),
        np.asarray(
            [item["chartSignedArea"] for item in accepted], dtype=np.float32
        ),
        {
            "components": int(
                len(np.unique(component[retained_node]))
                if np.any(retained_node)
                else 0
            ),
            "retainedChartNodes": int(np.count_nonzero(retained_node)),
            "rejectedNearDuplicateChartNodes": rejected_duplicate,
            "angularWedgeCount": angular_wedge_count,
            "uniqueLocalFanTriangles": len(triangle_keys),
            "rejectedWideAngularWedge": rejected_wide_wedge,
            "rejectedMissingClosingGraphEdge": rejected_missing_closure,
            "rejectedTriangleGeometry": rejected_geometry,
            "rejectedChartOrientation": rejected_orientation,
            "rejectedPlanarOverlap": rejected_overlap,
            "rejectedManifoldIncidence": rejected_manifold,
            "retainedTriangles": len(accepted),
        },
    )


def build_physical_mid_surface_mesh(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    physical_sheet_label: np.ndarray,
    source_component_id: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    *,
    settings: PhysicalMidSurfaceMeshSettings | None = None,
    geometry_scale: float = 1.0,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Orient, intrinsically chart, and triangulate physical midpoint graphs."""

    resolved = settings or PhysicalMidSurfaceMeshSettings()
    if not math.isfinite(geometry_scale) or geometry_scale <= 0.0:
        raise ValueError("mid-surface mesh geometry scale must be positive")
    linear_scale = float(geometry_scale)
    area_scale = linear_scale * linear_scale
    center = np.asarray(center_xyz, dtype=np.float64)
    normal = _normalized(normal_xyz)
    label = np.asarray(physical_sheet_label, dtype=np.int32)
    source_component = np.asarray(source_component_id, dtype=np.int32)
    edge_first = np.asarray(first, dtype=np.int32)
    edge_second = np.asarray(second, dtype=np.int32)
    score = np.asarray(edge_score, dtype=np.float32)
    if any(len(value) != len(center) for value in (normal, label, source_component)):
        raise ValueError("physical midpoint node arrays are not aligned")
    if any(len(value) != len(edge_first) for value in (edge_second, score)):
        raise ValueError("physical midpoint edge arrays are not aligned")
    if np.any(label[edge_first] != label[edge_second]):
        raise ValueError("physical midpoint graph contains cross-sheet edges")

    component_value, component_count = np.unique(
        source_component[source_component >= 0], return_counts=True
    )
    component_order = np.lexsort((component_value, -component_count))
    eligible_component = component_value[
        component_order[component_count[component_order] >= resolved.minimum_source_component_nodes]
    ][: resolved.maximum_source_components]
    selected_node = np.isin(source_component, eligible_component)
    frames = transport_mid_surface_frames(
        center,
        normal,
        edge_first,
        edge_second,
        score,
        selected_node,
    )
    signed_normal = np.asarray(frames["signedNormalXYZ"], dtype=np.float64)
    tangent_u = np.asarray(frames["tangentUXYZ"], dtype=np.float64)
    tangent_v = np.asarray(frames["tangentVXYZ"], dtype=np.float64)
    oriented_cosine = np.asarray(frames["signedNormalCosine"], dtype=np.float64)
    chart_edge = np.asarray(frames["selectedEdge"], dtype=bool) & (
        oriented_cosine
        >= math.cos(math.radians(resolved.maximum_oriented_neighbor_normal_degrees))
    )
    displacement = center[edge_second] - center[edge_first]
    along = 0.5 * (
        np.einsum("ij,ij->i", displacement, tangent_u[edge_first])
        + np.einsum("ij,ij->i", displacement, tangent_u[edge_second])
    )
    cross = 0.5 * (
        np.einsum("ij,ij->i", displacement, tangent_v[edge_first])
        + np.einsum("ij,ij->i", displacement, tangent_v[edge_second])
    )
    source_size = np.zeros(len(center), dtype=np.int32)
    eligible_component_set = set(int(item) for item in eligible_component)
    for value, count in zip(component_value, component_count):
        if int(value) in eligible_component_set:
            source_size[source_component == value] = int(count)

    robust_weight = score.astype(np.float64)
    chart_uv = np.full((len(center), 2), np.nan, dtype=np.float32)
    chart_residual = np.full(len(edge_first), np.nan, dtype=np.float32)
    integration_passes: list[dict[str, Any]] = []
    for iteration in range(resolved.robust_chart_iterations):
        chart_uv, chart_residual, chart_summary = integrate_intrinsic_surface_charts(
            center,
            edge_first,
            edge_second,
            robust_weight,
            chart_edge,
            source_component,
            source_size,
            along,
            cross,
            minimum_component_needles=resolved.minimum_source_component_nodes,
            solver_relative_tolerance=resolved.chart_solver_relative_tolerance,
            solver_maximum_iterations=resolved.chart_solver_maximum_iterations,
        )
        integration_passes.append(chart_summary)
        if iteration + 1 == resolved.robust_chart_iterations:
            break
        finite = chart_edge & np.isfinite(chart_residual)
        robust_scale = np.ones(len(score), dtype=np.float64)
        robust_scale[finite] = np.minimum(
            1.0,
            resolved.chart_huber_delta_voxels * linear_scale
            / np.maximum(chart_residual[finite], 1.0e-8),
        )
        robust_weight = score.astype(np.float64) * robust_scale

    mesh_edge = (
        chart_edge
        & np.isfinite(chart_residual)
        & (
            chart_residual
            <= resolved.maximum_mesh_edge_residual_voxels * linear_scale
        )
    )
    mesh_component, mesh_component_size, mesh_component_counts = _ranked_node_components(
        selected_node, edge_first, edge_second, mesh_edge
    )
    mesh_component_eligible = (
        (mesh_component >= 0)
        & (mesh_component < resolved.maximum_mesh_components)
        & (mesh_component_size >= resolved.minimum_mesh_component_nodes)
    )
    triangulation_size = np.where(
        mesh_component_eligible, mesh_component_size, 0
    ).astype(np.int32)
    triangulation_edge = (
        mesh_edge
        & mesh_component_eligible[edge_first]
        & mesh_component_eligible[edge_second]
        & (mesh_component[edge_first] == mesh_component[edge_second])
    )
    if resolved.triangulation_mode == "chart-delaunay":
        (
            triangles,
            triangle_area,
            triangle_normal_residual,
            triangle_summary,
        ) = triangulate_intrinsic_surface_charts(
            center,
            signed_normal,
            chart_uv,
            edge_first,
            edge_second,
            score,
            triangulation_edge,
            triangulation_edge,
            mesh_component,
            triangulation_size,
            minimum_component_needles=resolved.minimum_mesh_component_nodes,
            minimum_chart_separation_voxels=(
                resolved.minimum_chart_separation_voxels * linear_scale
            ),
            maximum_edge_voxels=(
                resolved.maximum_triangle_edge_voxels * linear_scale
            ),
            maximum_normal_residual_degrees=(
                resolved.maximum_triangle_normal_residual_degrees
            ),
            minimum_area_voxels_squared=(
                resolved.minimum_triangle_area_voxels_squared * area_scale
            ),
            maximum_local_closure_edge_voxels=(
                resolved.maximum_local_closure_edge_voxels * linear_scale
            ),
            maximum_local_closure_height_voxels=(
                resolved.maximum_local_closure_height_voxels * linear_scale
            ),
            maximum_local_closure_normal_degrees=(
                resolved.maximum_local_closure_normal_degrees
            ),
        )
        triangle_source = (
            mesh_component[triangles[:, 0]]
            if len(triangles)
            else np.empty(0, dtype=np.int32)
        )
        oriented, triangle_chart_signed_area = _chart_orientation_filter(
            triangles,
            chart_uv,
            triangle_source,
        )
        rejected_orientation = int(np.count_nonzero(~oriented))
        triangles = triangles[oriented]
        triangle_area = triangle_area[oriented]
        triangle_normal_residual = triangle_normal_residual[oriented]
        triangle_chart_signed_area = triangle_chart_signed_area[oriented]
        triangle_summary = {
            **triangle_summary,
            "rejectedChartOrientation": rejected_orientation,
            "retainedTriangles": int(len(triangles)),
        }
    elif resolved.triangulation_mode == "local-fans":
        (
            triangles,
            triangle_area,
            triangle_normal_residual,
            triangle_chart_signed_area,
            triangle_summary,
        ) = triangulate_local_chart_fans(
            center,
            signed_normal,
            chart_uv,
            edge_first,
            edge_second,
            score,
            triangulation_edge,
            mesh_component,
            triangulation_size,
            minimum_component_nodes=resolved.minimum_mesh_component_nodes,
            minimum_chart_separation_voxels=(
                resolved.minimum_chart_separation_voxels * linear_scale
            ),
            maximum_edge_voxels=(
                resolved.maximum_triangle_edge_voxels * linear_scale
            ),
            maximum_normal_residual_degrees=(
                resolved.maximum_triangle_normal_residual_degrees
            ),
            minimum_area_voxels_squared=(
                resolved.minimum_triangle_area_voxels_squared * area_scale
            ),
        )
    else:
        (
            triangles,
            triangle_area,
            triangle_normal_residual,
            triangle_chart_signed_area,
            triangle_summary,
        ) = triangulate_graph_supported_charts(
            center,
            signed_normal,
            chart_uv,
            edge_first,
            edge_second,
            score,
            triangulation_edge,
            mesh_component,
            triangulation_size,
            minimum_component_nodes=resolved.minimum_mesh_component_nodes,
            minimum_chart_separation_voxels=(
                resolved.minimum_chart_separation_voxels * linear_scale
            ),
            maximum_local_closure_edge_voxels=(
                resolved.maximum_local_closure_edge_voxels * linear_scale
            ),
            maximum_local_closure_height_voxels=(
                resolved.maximum_local_closure_height_voxels * linear_scale
            ),
            maximum_local_closure_normal_degrees=(
                resolved.maximum_local_closure_normal_degrees
            ),
            maximum_edge_voxels=(
                resolved.maximum_triangle_edge_voxels * linear_scale
            ),
            maximum_normal_residual_degrees=(
                resolved.maximum_triangle_normal_residual_degrees
            ),
            minimum_area_voxels_squared=(
                resolved.minimum_triangle_area_voxels_squared * area_scale
            ),
        )
    triangle_source_component = (
        mesh_component[triangles[:, 0]]
        if len(triangles)
        else np.empty(0, dtype=np.int32)
    )
    triangle_component, triangle_component_counts, manifold_summary = (
        _triangle_components(triangles)
    )
    if len(triangles):
        triangle_label = label[triangles]
        if np.any(triangle_label != triangle_label[:, :1]):
            raise RuntimeError("one midpoint triangle crosses physical sheet identities")
    else:
        triangle_label = np.empty((0, 3), dtype=np.int32)

    arrays = {
        "selectedNode": selected_node,
        "normalSign": np.asarray(frames["normalSign"], dtype=np.int8),
        "signedNormalXYZ": np.asarray(frames["signedNormalXYZ"], dtype=np.float32),
        "tangentUXYZ": np.asarray(frames["tangentUXYZ"], dtype=np.float32),
        "tangentVXYZ": np.asarray(frames["tangentVXYZ"], dtype=np.float32),
        "chartUV": chart_uv.astype(np.float32),
        "sourceComponentId": source_component.astype(np.int32),
        "sourceComponentSize": source_size,
        "meshComponentId": mesh_component.astype(np.int32),
        "meshComponentSize": mesh_component_size.astype(np.int32),
        "edgeTreeGauge": np.asarray(frames["treeEdge"], dtype=bool),
        "edgeChartEligible": chart_edge,
        "edgeMeshEligible": mesh_edge,
        "edgeRawNormalCosine": np.asarray(frames["rawNormalCosine"], dtype=np.float32),
        "edgeSignedNormalCosine": np.asarray(
            frames["signedNormalCosine"], dtype=np.float32
        ),
        "edgeAlongVoxels": along.astype(np.float32),
        "edgeCrossVoxels": cross.astype(np.float32),
        "edgeChartResidualVoxels": chart_residual.astype(np.float32),
        "triangleNode": triangles.astype(np.int32),
        "triangleSourceComponentId": triangle_source_component.astype(np.int32),
        "triangleMeshComponentId": triangle_component.astype(np.int32),
        "triangleMeshComponentSize": (
            triangle_component_counts[triangle_component]
            if len(triangle_component)
            else np.empty(0, dtype=np.int32)
        ),
        "trianglePhysicalSheetLabel": (
            triangle_label[:, 0].astype(np.int32)
            if len(triangle_label)
            else np.empty(0, dtype=np.int32)
        ),
        "triangleAreaVoxelsSquared": triangle_area.astype(np.float32),
        "triangleNormalResidualDegrees": triangle_normal_residual.astype(np.float32),
        "triangleChartSignedAreaVoxelsSquared": triangle_chart_signed_area.astype(
            np.float32
        ),
    }
    summary: dict[str, Any] = {
        "geometryScale": round(linear_scale, 8),
        "counts": {
            "inputNodeCount": int(len(center)),
            "inputEdgeCount": int(len(edge_first)),
            "selectedSourceComponentCount": int(len(eligible_component)),
            "selectedNodeCount": int(np.count_nonzero(selected_node)),
            "selectedEdgeCount": int(np.count_nonzero(frames["selectedEdge"])),
            "normalGaugeTreeEdgeCount": int(np.count_nonzero(frames["treeEdge"])),
            "normalGaugeContradictoryEdgeCount": int(
                np.count_nonzero(np.asarray(frames["selectedEdge"]) & ~chart_edge)
            ),
            "chartEligibleEdgeCount": int(np.count_nonzero(chart_edge)),
            "meshEligibleEdgeCount": int(np.count_nonzero(mesh_edge)),
            "meshNodeComponentCount": int(len(mesh_component_counts)),
            "triangulatedNodeComponentCount": int(
                len(np.unique(mesh_component[mesh_component_eligible]))
            ),
            "triangleCountBeforeChartOrientation": int(
                len(triangles)
                + int(triangle_summary["rejectedChartOrientation"])
            ),
            "rejectedChartOrientationTriangleCount": int(
                triangle_summary["rejectedChartOrientation"]
            ),
            "triangleCount": int(len(triangles)),
            "triangleMeshComponentCount": int(len(triangle_component_counts)),
            "crossPhysicalSheetTriangleCount": int(
                np.count_nonzero(
                    np.any(triangle_label != triangle_label[:, :1], axis=1)
                )
                if len(triangle_label)
                else 0
            ),
            **manifold_summary,
        },
        "selectedSourceComponentIds": [int(value) for value in eligible_component],
        "largestMeshNodeComponentSizes": [
            int(value) for value in mesh_component_counts[:32]
        ],
        "largestTriangleMeshComponentSizes": [
            int(value) for value in triangle_component_counts[:32]
        ],
        "distributions": {
            "selectedRawNormalDegrees": _percentiles(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.abs(
                                np.asarray(frames["rawNormalCosine"])[
                                    np.asarray(frames["selectedEdge"], dtype=bool)
                                ]
                            ),
                            0.0,
                            1.0,
                        )
                    )
                )
            ),
            "orientedNormalDegrees": _percentiles(
                np.degrees(
                    np.arccos(
                        np.clip(
                            np.asarray(frames["signedNormalCosine"])[chart_edge],
                            0.0,
                            1.0,
                        )
                    )
                )
            ),
            "chartEdgeResidualVoxels": _percentiles(chart_residual[chart_edge]),
            "retainedMeshEdgeResidualVoxels": _percentiles(
                chart_residual[mesh_edge]
            ),
            "triangleAreaVoxelsSquared": _percentiles(triangle_area),
            "triangleNormalResidualDegrees": _percentiles(
                triangle_normal_residual
            ),
        },
        "chartIntegrationPasses": integration_passes,
        "triangulation": triangle_summary,
        "triangulationMode": resolved.triangulation_mode,
    }
    return summary, arrays


def run_physical_mid_surface_mesh(
    mid_surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalMidSurfaceMeshSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Persist intrinsic charts and manifold meshes for midpoint fragments."""

    started = time.monotonic()
    manifest_path, manifest, source_arrays = _load_mid_surface(mid_surface_root)
    resolved = settings or PhysicalMidSurfaceMeshSettings()
    sampling_stride = float(
        manifest.get("geometry", {}).get("samplingStrideVoxels", 2.0)
    )
    if not math.isfinite(sampling_stride) or sampling_stride <= 0.0:
        raise ValueError("physical mid-surface sampling stride must be positive")
    geometry_scale = sampling_stride / 2.0
    identity: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_MESH_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_MESH_VERSION,
        "midSurface": {
            "manifestPath": str(manifest_path),
            "manifestSha256": sha256_file(manifest_path),
            "dataSha256": manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "geometryScale": geometry_scale,
        "referenceSamplingStrideVoxels": 2.0,
        "implementationSha256": sha256_file(Path(__file__)),
        "chartImplementationSha256": sha256_file(
            Path(integrate_intrinsic_surface_charts.__code__.co_filename)
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_manifest = output / f"{PHYSICAL_MID_SURFACE_MESH_STEM}.json"
    data_path = output / f"{PHYSICAL_MID_SURFACE_MESH_STEM}.npz"
    if not force and output_manifest.is_file() and data_path.is_file():
        cached = json.loads(output_manifest.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    summary, arrays = build_physical_mid_surface_mesh(
        np.asarray(source_arrays["midpointXYZ"]),
        np.asarray(source_arrays["normalXYZ"]),
        np.asarray(source_arrays["physicalSheetLabel"]),
        np.asarray(source_arrays["componentId"]),
        np.asarray(source_arrays["edgeFirstNode"]),
        np.asarray(source_arrays["edgeSecondNode"]),
        np.asarray(source_arrays["edgeScore"]),
        settings=resolved,
        geometry_scale=geometry_scale,
    )
    _write_npz(data_path, arrays)
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_MESH_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_MESH_VERSION,
        "state": "complete",
        "identity": identity,
        "source": manifest["source"],
        "geometry": manifest["geometry"],
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "normalGauge": (
                "maximum-quality spanning-forest unsigned gauge with every "
                "non-tree edge retained as an independent consistency check"
            ),
            "chart": (
                "parallel-transported tangent chords integrated by robust "
                "weighted graph least squares"
            ),
            "mesh": (
                "planar graph-supported intrinsic triangles with short local "
                "geometry closures, gated by 3D surface geometry and a "
                "consistent local chart orientation"
            ),
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output_manifest, payload)
    return payload
