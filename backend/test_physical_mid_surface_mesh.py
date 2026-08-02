from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.physical_mid_surface_mesh import (
    PhysicalMidSurfaceMeshSettings,
    build_physical_mid_surface_mesh,
    transport_mid_surface_frames,
)


def _curved_grid(
    width: int = 10,
    height: int = 8,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    coordinates = np.asarray(
        [(x, y) for y in range(height) for x in range(width)],
        dtype=np.float64,
    )
    radius = 20.0
    theta = coordinates[:, 0] / radius
    center = np.column_stack(
        (
            radius * np.sin(theta),
            coordinates[:, 1],
            radius * (1.0 - np.cos(theta)),
        )
    )
    normal = np.column_stack((-np.sin(theta), np.zeros(len(theta)), np.cos(theta)))
    normal[::3] *= -1.0
    edge: list[tuple[int, int]] = []
    for first in range(len(coordinates)):
        delta = np.abs(coordinates[first + 1 :] - coordinates[first])
        for relative in np.flatnonzero(
            (np.max(delta, axis=1) <= 1.0) & (np.max(delta, axis=1) > 0.0)
        ):
            edge.append((first, first + 1 + int(relative)))
    edge_values = np.asarray(edge, dtype=np.int32)
    return center, normal, edge_values[:, 0], edge_values[:, 1]


class PhysicalMidSurfaceMeshTests(unittest.TestCase):
    def test_unsigned_curved_normals_transport_one_tangent_gauge(self) -> None:
        center, normal, first, second = _curved_grid()
        result = transport_mid_surface_frames(
            center,
            normal,
            first,
            second,
            np.ones(len(first), dtype=np.float32),
            np.ones(len(center), dtype=bool),
        )
        signed = np.asarray(result["signedNormalXYZ"])
        tangent_u = np.asarray(result["tangentUXYZ"])
        tangent_v = np.asarray(result["tangentVXYZ"])
        self.assertTrue(np.all(np.asarray(result["normalSign"]) != 0))
        self.assertGreater(float(np.min(np.sum(signed[first] * signed[second], axis=1))), 0.99)
        np.testing.assert_allclose(np.sum(signed * tangent_u, axis=1), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(np.sum(signed * tangent_v, axis=1), 0.0, atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(tangent_u, axis=1), 1.0, atol=1.0e-6)
        np.testing.assert_allclose(np.linalg.norm(tangent_v, axis=1), 1.0, atol=1.0e-6)

    def test_curved_midpoint_grid_forms_one_manifold_mesh(self) -> None:
        center, normal, first, second = _curved_grid()
        summary, arrays = build_physical_mid_surface_mesh(
            center,
            normal,
            np.full(len(center), 7, dtype=np.int32),
            np.zeros(len(center), dtype=np.int32),
            first,
            second,
            np.ones(len(first), dtype=np.float32),
            settings=PhysicalMidSurfaceMeshSettings(
                minimum_source_component_nodes=8,
                maximum_source_components=1,
                minimum_mesh_component_nodes=4,
                maximum_mesh_components=2,
                maximum_oriented_neighbor_normal_degrees=15.0,
                robust_chart_iterations=2,
                chart_huber_delta_voxels=0.5,
                maximum_mesh_edge_residual_voxels=0.5,
                minimum_chart_separation_voxels=0.01,
                maximum_triangle_edge_voxels=2.0,
                maximum_triangle_normal_residual_degrees=15.0,
                minimum_triangle_area_voxels_squared=0.1,
            ),
        )
        self.assertGreater(summary["counts"]["triangleCount"], 100)
        self.assertEqual(summary["counts"]["triangleMeshComponentCount"], 1)
        self.assertEqual(summary["counts"]["nonmanifoldEdgeCount"], 0)
        self.assertEqual(summary["counts"]["crossPhysicalSheetTriangleCount"], 0)
        self.assertTrue(
            np.all(np.asarray(arrays["trianglePhysicalSheetLabel"]) == 7)
        )
        self.assertLess(
            summary["distributions"]["chartEdgeResidualVoxels"]["p90"],
            0.1,
        )

    def test_chart_delaunay_forms_one_manifold_mesh(self) -> None:
        center, normal, first, second = _curved_grid()
        summary, arrays = build_physical_mid_surface_mesh(
            center,
            normal,
            np.full(len(center), 7, dtype=np.int32),
            np.zeros(len(center), dtype=np.int32),
            first,
            second,
            np.ones(len(first), dtype=np.float32),
            settings=PhysicalMidSurfaceMeshSettings(
                minimum_source_component_nodes=8,
                maximum_source_components=1,
                minimum_mesh_component_nodes=4,
                maximum_mesh_components=2,
                maximum_oriented_neighbor_normal_degrees=15.0,
                robust_chart_iterations=1,
                chart_huber_delta_voxels=0.5,
                maximum_mesh_edge_residual_voxels=0.5,
                minimum_chart_separation_voxels=0.01,
                maximum_triangle_edge_voxels=2.0,
                maximum_triangle_normal_residual_degrees=15.0,
                minimum_triangle_area_voxels_squared=0.1,
                triangulation_mode="chart-delaunay",
            ),
        )
        self.assertGreater(summary["counts"]["triangleCount"], 100)
        self.assertEqual(summary["counts"]["triangleMeshComponentCount"], 1)
        self.assertEqual(summary["counts"]["nonmanifoldEdgeCount"], 0)
        self.assertEqual(summary["triangulationMode"], "chart-delaunay")
        self.assertTrue(
            np.all(np.asarray(arrays["trianglePhysicalSheetLabel"]) == 7)
        )

    def test_local_fans_form_one_manifold_mesh(self) -> None:
        center, normal, first, second = _curved_grid()
        summary, arrays = build_physical_mid_surface_mesh(
            center,
            normal,
            np.full(len(center), 7, dtype=np.int32),
            np.zeros(len(center), dtype=np.int32),
            first,
            second,
            np.ones(len(first), dtype=np.float32),
            settings=PhysicalMidSurfaceMeshSettings(
                minimum_source_component_nodes=8,
                maximum_source_components=1,
                minimum_mesh_component_nodes=4,
                maximum_mesh_components=2,
                maximum_oriented_neighbor_normal_degrees=15.0,
                robust_chart_iterations=1,
                chart_huber_delta_voxels=0.5,
                maximum_mesh_edge_residual_voxels=0.5,
                minimum_chart_separation_voxels=0.01,
                maximum_triangle_edge_voxels=2.0,
                maximum_triangle_normal_residual_degrees=15.0,
                minimum_triangle_area_voxels_squared=0.1,
                triangulation_mode="local-fans",
            ),
        )
        self.assertGreater(summary["counts"]["triangleCount"], 100)
        self.assertEqual(summary["counts"]["triangleMeshComponentCount"], 1)
        self.assertEqual(summary["counts"]["nonmanifoldEdgeCount"], 0)
        self.assertEqual(summary["triangulationMode"], "local-fans")
        self.assertTrue(
            np.all(np.asarray(arrays["trianglePhysicalSheetLabel"]) == 7)
        )

    def test_independent_components_may_overlap_in_chart_coordinates(self) -> None:
        center, normal, first, second = _curved_grid()
        count = len(center)
        summary, arrays = build_physical_mid_surface_mesh(
            np.vstack((center, center + np.asarray((0.0, 0.0, 50.0)))),
            np.vstack((normal, normal)),
            np.concatenate(
                (
                    np.full(count, 7, dtype=np.int32),
                    np.full(count, 8, dtype=np.int32),
                )
            ),
            np.concatenate(
                (
                    np.zeros(count, dtype=np.int32),
                    np.ones(count, dtype=np.int32),
                )
            ),
            np.concatenate((first, first + count)),
            np.concatenate((second, second + count)),
            np.ones(2 * len(first), dtype=np.float32),
            settings=PhysicalMidSurfaceMeshSettings(
                minimum_source_component_nodes=8,
                maximum_source_components=2,
                minimum_mesh_component_nodes=4,
                maximum_mesh_components=4,
                maximum_oriented_neighbor_normal_degrees=15.0,
                robust_chart_iterations=2,
                chart_huber_delta_voxels=0.5,
                maximum_mesh_edge_residual_voxels=0.5,
                minimum_chart_separation_voxels=0.01,
                maximum_local_closure_edge_voxels=2.0,
                maximum_local_closure_height_voxels=0.5,
                maximum_local_closure_normal_degrees=15.0,
                maximum_triangle_edge_voxels=2.0,
                maximum_triangle_normal_residual_degrees=15.0,
                minimum_triangle_area_voxels_squared=0.1,
            ),
        )
        self.assertGreater(summary["counts"]["triangleCount"], 200)
        self.assertEqual(summary["counts"]["triangleMeshComponentCount"], 2)
        self.assertEqual(
            set(np.asarray(arrays["trianglePhysicalSheetLabel"]).tolist()),
            {7, 8},
        )


if __name__ == "__main__":
    unittest.main()
