from __future__ import annotations

from collections import defaultdict
from typing import Any, Mapping

import numpy as np


SURFACE_NODE_FIELDS = (
    "selected",
    "component",
    "componentSize",
    "signedNormalXYZ",
    "tangentUxyz",
    "tangentVxyz",
    "chartUV",
    "midpointXYZ",
    "thicknessVoxels",
)


def triangle_edge_region_labels(triangles: np.ndarray) -> np.ndarray:
    """Label triangle atlas pages joined through complete mesh edges.

    Vertex-only contact does not join chart pages.  Callers that require a
    single-valued node chart must first split non-manifold vertex fans with
    ``split_nonmanifold_surface_vertices``.
    """

    triangle = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    parent = np.arange(len(triangle), dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    edge_triangle: dict[tuple[int, int], int] = {}
    for triangle_index, values in enumerate(triangle):
        for edge_index, first in enumerate(values):
            second = int(values[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            previous = edge_triangle.get(edge)
            if previous is None:
                edge_triangle[edge] = triangle_index
                continue
            first_root, second_root = find(previous), find(triangle_index)
            if first_root != second_root:
                parent[max(first_root, second_root)] = min(
                    first_root, second_root
                )
    root = np.asarray(
        [find(index) for index in range(len(triangle))], dtype=np.int32
    )
    _, labels = np.unique(root, return_inverse=True)
    return labels.astype(np.int32)


def split_nonmanifold_surface_vertices(
    surface: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Split triangle fans that touch only through one surface vertex.

    Edge incidence alone is insufficient to make a triangle complex a
    two-manifold.  Two otherwise independent sheet regions may reuse one Acus
    node, leaving one vertex with several disconnected incident fans.  Such a
    pinch has no single valid intrinsic coordinate.  This routine duplicates
    the complete node record once per extra local fan without moving geometry
    or changing any triangle edge connectivity.

    The returned edge graph is the exact 1-skeleton of the normalized triangle
    complex.  Downstream sheet-level operations therefore cannot accidentally
    traverse an old candidate-graph edge that is absent from the materialized
    surface.
    """

    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    if triangles.ndim != 2 or triangles.shape[1:] != (3,):
        raise ValueError("surface triangles must have shape (N, 3)")
    node_count = len(np.asarray(surface["midpointXYZ"]))
    if len(triangles) and (
        int(np.min(triangles)) < 0 or int(np.max(triangles)) >= node_count
    ):
        raise ValueError("surface triangle references a missing node")

    incident: dict[int, list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangles):
        for node in triangle:
            incident[int(node)].append(triangle_index)

    normalized_triangle = triangles.copy()
    duplicate_source: list[int] = []
    split_vertex_count = 0
    fan_count_before = 0
    fan_count_after = 0
    maximum_fans = 1
    for node, triangle_indices in sorted(incident.items()):
        if len(triangle_indices) < 2:
            continue
        parent = {
            int(triangle_index): int(triangle_index)
            for triangle_index in triangle_indices
        }

        def find(value: int) -> int:
            while parent[value] != value:
                parent[value] = parent[parent[value]]
                value = parent[value]
            return value

        by_neighbor: dict[int, list[int]] = defaultdict(list)
        for triangle_index in triangle_indices:
            for neighbor in triangles[triangle_index]:
                neighbor_value = int(neighbor)
                if neighbor_value != node:
                    by_neighbor[neighbor_value].append(int(triangle_index))
        for values in by_neighbor.values():
            first_root = find(values[0])
            for value in values[1:]:
                second_root = find(value)
                if first_root != second_root:
                    parent[max(first_root, second_root)] = min(
                        first_root, second_root
                    )
                    first_root = find(first_root)
        groups: dict[int, list[int]] = defaultdict(list)
        for triangle_index in triangle_indices:
            groups[find(int(triangle_index))].append(int(triangle_index))
        ordered_groups = sorted(groups.values(), key=lambda values: min(values))
        fan_count = len(ordered_groups)
        if fan_count <= 1:
            continue
        split_vertex_count += 1
        fan_count_before += 1
        fan_count_after += fan_count
        maximum_fans = max(maximum_fans, fan_count)
        for group in ordered_groups[1:]:
            duplicate = node_count + len(duplicate_source)
            duplicate_source.append(node)
            for triangle_index in group:
                corner = normalized_triangle[triangle_index] == node
                if int(np.count_nonzero(corner)) != 1:
                    raise ValueError("surface triangle does not contain its fan node once")
                normalized_triangle[triangle_index, corner] = duplicate

    duplicate_index = np.asarray(duplicate_source, dtype=np.int32)
    result = {key: np.asarray(value) for key, value in surface.items()}
    for name in SURFACE_NODE_FIELDS:
        if name not in surface:
            raise ValueError(f"surface is missing node field {name}")
        values = np.asarray(surface[name])
        if len(values) != node_count:
            raise ValueError(f"surface node field {name} has the wrong length")
        result[name] = (
            np.concatenate((values, values[duplicate_index]), axis=0)
            if len(duplicate_index)
            else values.copy()
        )
    result["triangleFrontierIndex"] = normalized_triangle
    weld_group = np.asarray(
        surface.get("surfaceNodeWeldGroup", np.arange(node_count)),
        dtype=np.int64,
    )
    if len(weld_group) != node_count:
        raise ValueError("surface weld-group field has the wrong length")
    result["surfaceNodeWeldGroup"] = (
        np.concatenate((weld_group, weld_group[duplicate_index]))
        if len(duplicate_index)
        else weld_group.copy()
    )

    if len(normalized_triangle):
        surface_edge = np.sort(
            np.concatenate(
                (
                    normalized_triangle[:, (0, 1)],
                    normalized_triangle[:, (1, 2)],
                    normalized_triangle[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        )
        surface_edge = np.unique(surface_edge, axis=0).astype(np.int32)
    else:
        surface_edge = np.empty((0, 2), dtype=np.int32)
    result["edgeFirstFrontierIndex"] = surface_edge[:, 0]
    result["edgeSecondFrontierIndex"] = surface_edge[:, 1]
    result["edgeSelected"] = np.ones(len(surface_edge), dtype=np.uint8)
    result["integrationResidualVoxels"] = np.zeros(
        len(surface_edge), dtype=np.float32
    )

    component = np.asarray(result["component"], dtype=np.int32)
    present = component[component >= 0]
    component_size = (
        np.bincount(present).astype(np.int32)
        if len(present)
        else np.empty(0, dtype=np.int32)
    )
    node_component_size = np.zeros(len(component), dtype=np.int32)
    valid = component >= 0
    node_component_size[valid] = component_size[component[valid]]
    result["componentSize"] = node_component_size

    return result, {
        "nodeCountBefore": node_count,
        "nodeCountAfter": int(node_count + len(duplicate_index)),
        "splitVertexCount": split_vertex_count,
        "duplicatedVertexCount": int(len(duplicate_index)),
        "extraIncidentFanCount": int(fan_count_after - fan_count_before),
        "maximumIncidentFansAtOneVertex": maximum_fans,
        "exactSurfaceEdgeCount": int(len(surface_edge)),
        "geometryMoved": False,
        "triangleConnectivityChanged": False,
        "oldCandidateGraphRetained": False,
    }
