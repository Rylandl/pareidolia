from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.paired_boundary_surface import (
    PairedBoundarySurfaceSettings,
    build_paired_boundary_surface,
)


def _parallel_papyrus_patch(width: int = 10, height: int = 8) -> dict[str, np.ndarray]:
    profile_count = width * height
    midpoint: list[tuple[float, float, float]] = []
    lower: list[tuple[float, float, float]] = []
    upper: list[tuple[float, float, float]] = []
    for y_value in range(height):
        for x_value in range(width):
            bend = 0.01 * float(x_value * x_value)
            lower.append((float(x_value), float(y_value), bend))
            upper.append((float(x_value), float(y_value), bend + 10.0))
            midpoint.append((float(x_value), float(y_value), bend + 5.0))
    lower_xyz = np.asarray(lower, dtype=np.float32)
    upper_xyz = np.asarray(upper, dtype=np.float32)
    endpoint_xyz = np.empty((2 * profile_count, 3), dtype=np.float32)
    endpoint_xyz[0::2] = lower_xyz
    endpoint_xyz[1::2] = upper_xyz
    edge: list[tuple[int, int]] = []
    for y_value in range(height):
        for x_value in range(width):
            first = y_value * width + x_value
            for dy, dx in ((0, 1), (1, -1), (1, 0), (1, 1)):
                following_y = y_value + dy
                following_x = x_value + dx
                if not (0 <= following_y < height and 0 <= following_x < width):
                    continue
                second = following_y * width + following_x
                edge.append((2 * first, 2 * second))
                edge.append((2 * first + 1, 2 * second + 1))
    edge_values = np.asarray(edge, dtype=np.int32)
    support = np.bincount(edge_values.ravel(), minlength=2 * profile_count)
    return {
        "midpointXYZ": np.asarray(midpoint, dtype=np.float32),
        "normalXYZ": np.tile(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32),
            (profile_count, 1),
        ),
        "thicknessVoxels": np.full(profile_count, 10.0, dtype=np.float32),
        "lowerLocalEvidenceScore": np.full(profile_count, 0.9, dtype=np.float32),
        "boundaryTrackEndpointXYZ": endpoint_xyz,
        "boundaryTrackEndpointProfileNode": np.repeat(
            np.arange(profile_count, dtype=np.int32), 2
        ),
        "boundaryTrackEndpointSide": np.tile(
            np.asarray((0, 1), dtype=np.uint8), profile_count
        ),
        "boundaryTrackComponentId": np.tile(
            np.asarray((0, 1), dtype=np.int32), profile_count
        ),
        "boundaryTrackLocalSupportDegree": support.astype(np.int32),
        "boundaryTrackEdgeFirstEndpoint": edge_values[:, 0],
        "boundaryTrackEdgeSecondEndpoint": edge_values[:, 1],
        "boundaryTrackEdgeAffinity": np.ones(len(edge_values), dtype=np.float32),
    }


class PairedBoundarySurfaceTests(unittest.TestCase):
    def test_parallel_boundary_tracks_certify_one_papyrus_patch(self) -> None:
        source = _parallel_papyrus_patch()
        arrays, summary = build_paired_boundary_surface(
            source,
            sampling_stride_voxels=1.0,
            settings=PairedBoundarySurfaceSettings(
                minimum_boundary_track_endpoints=8,
                maximum_boundary_tracks=4,
                minimum_mesh_component_endpoints=4,
                maximum_mesh_components=8,
                robust_chart_iterations=1,
                minimum_local_support_degree=2,
                minimum_certified_patch_triangles=8,
            ),
        )
        self.assertGreater(summary["counts"]["boundaryTriangleCount"], 100)
        self.assertGreater(
            summary["counts"]["certifiedBoundaryFaceTriangleCount"], 100
        )
        self.assertEqual(
            summary["counts"]["certifiedBoundaryFaceComponentCount"], 2
        )
        self.assertGreater(summary["counts"]["certifiedPairedTriangleCount"], 100)
        self.assertEqual(summary["counts"]["certifiedPatchCount"], 1)
        self.assertEqual(summary["counts"]["certifiedSubstantialPatchCount"], 1)
        self.assertEqual(summary["counts"]["nonmanifoldEdgeCount"], 0)
        pairs = np.asarray(arrays["certifiedTriangleBoundaryTrackPair"])
        np.testing.assert_array_equal(
            np.unique(pairs, axis=0), np.asarray(((0, 1),), dtype=np.int32)
        )
        self.assertTrue(
            np.all(np.asarray(arrays["certifiedTriangleSubstantialPatch"]))
        )

    def test_companion_track_disagreement_is_not_certified(self) -> None:
        source = _parallel_papyrus_patch()
        component = np.asarray(source["boundaryTrackComponentId"]).copy()
        # Split half of the upper boundary into a second track. Boundary
        # triangles crossing that physical discontinuity cannot certify an
        # interior even though their primary lower face remains smooth.
        profiles = np.asarray(source["boundaryTrackEndpointProfileNode"])
        sides = np.asarray(source["boundaryTrackEndpointSide"])
        split = (
            (sides == 1)
            & ((profiles % 10) >= 5)
        )
        component[split] = 2
        source["boundaryTrackComponentId"] = component
        edge_first = np.asarray(source["boundaryTrackEdgeFirstEndpoint"])
        edge_second = np.asarray(source["boundaryTrackEdgeSecondEndpoint"])
        retained_edge = component[edge_first] == component[edge_second]
        source["boundaryTrackEdgeFirstEndpoint"] = edge_first[retained_edge]
        source["boundaryTrackEdgeSecondEndpoint"] = edge_second[retained_edge]
        source["boundaryTrackEdgeAffinity"] = np.asarray(
            source["boundaryTrackEdgeAffinity"]
        )[retained_edge]
        arrays, summary = build_paired_boundary_surface(
            source,
            sampling_stride_voxels=1.0,
            settings=PairedBoundarySurfaceSettings(
                minimum_boundary_track_endpoints=8,
                maximum_boundary_tracks=4,
                minimum_mesh_component_endpoints=4,
                maximum_mesh_components=8,
                robust_chart_iterations=1,
                minimum_local_support_degree=2,
                minimum_certified_patch_triangles=4,
            ),
        )
        certified = int(summary["counts"]["certifiedPairedTriangleCount"])
        boundary = int(summary["counts"]["boundaryTriangleCount"])
        self.assertGreater(certified, 0)
        self.assertLess(certified, boundary // 2)
        self.assertLess(
            summary["progressiveCertificateCounts"]["singleCompanionTrack"],
            boundary,
        )
        # The opposite face split invalidates interior prisms, not either
        # directly observed signed physical boundary.
        self.assertGreater(
            summary["counts"]["certifiedBoundaryFaceTriangleCount"],
            certified,
        )
        self.assertGreaterEqual(
            summary["patchAssociation"]["acceptedPatchAssociationCount"], 1
        )
        self.assertLess(
            summary["counts"]["certifiedAssemblyCount"],
            summary["counts"]["certifiedPatchCount"],
        )
        pair = np.asarray(arrays["certifiedTriangleBoundaryTrackPair"])
        self.assertTrue(np.all(pair[:, 0] != pair[:, 1]))


if __name__ == "__main__":
    unittest.main()
