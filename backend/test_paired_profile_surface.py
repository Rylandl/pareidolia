import unittest

import numpy as np

from backend.cubical.paired_profile_surface import (
    PairedProfileSurfaceSettings,
    _frontier_bundle_connectivity,
    build_direct_paired_profile_surface,
)


class PairedProfileSurfaceTests(unittest.TestCase):
    def test_geometric_closure_swaps_boundaries_for_unsigned_normal_flip(self) -> None:
        bank = {
            "midpointXYZ": np.asarray([[0, 0, 5], [2, 0, 5]], np.float32),
            "normalXYZ": np.asarray([[0, 0, 1], [0, 0, -1]], np.float32),
            "boundaryLowerXYZ": np.asarray([[0, 0, 0], [2, 0, 10]], np.float32),
            "boundaryUpperXYZ": np.asarray([[0, 0, 10], [2, 0, 0]], np.float32),
            "thicknessVoxels": np.full(2, 10, np.float32),
            "spatialKeyXYZ": np.asarray([[0, 0, 0], [1, 0, 0]], np.int32),
            "localEvidenceScore": np.full(2, 0.9, np.float32),
            "opposingNormalCosine": np.ones(2, np.float32),
            "isolatedConservative": np.ones(2, bool),
            "seedComponentId": np.full(2, -1, np.int32),
        }
        growth = {
            "selected": np.ones(2, bool),
            "selectedLabel": np.asarray([0, 1], np.int32),
            "edgeFirstCandidate": np.empty(0, np.int32),
            "edgeSecondCandidate": np.empty(0, np.int32),
            "edgeAffinity": np.empty(0, np.float32),
            "edgeNormalDegrees": np.empty(0, np.float32),
            "edgeMidpointHeightSamplingSteps": np.empty(0, np.float32),
            "edgeBoundaryHeightSamplingSteps": np.empty(0, np.float32),
            "edgeThicknessDifferenceSamplingSteps": np.empty(0, np.float32),
        }
        macro_manifest = {"identity": {"resolvedScale": {"supportSamplingSteps": 4}}}
        macro = {
            "binKeyXYZ": np.asarray([[0, 0, 0]], np.int32),
            "normalXYZ": np.asarray([[0, 0, 1]], np.float32),
            "orientationConfidence": np.asarray([1.0], np.float32),
            "trusted": np.asarray([True]),
        }

        arrays, summary = build_direct_paired_profile_surface(
            bank,
            growth,
            macro_manifest,
            macro,
            settings=PairedProfileSurfaceSettings(
                component_solver_mode="connected-components",
                enable_tangent_column_guard=False,
                maximum_closure_distance_sampling_steps=2.0,
            ),
        )

        self.assertEqual(summary["counts"]["retainedGeometricClosureEdgeCount"], 1)
        np.testing.assert_array_equal(arrays["componentId"], (0, 0))

    def test_frontier_bundle_requires_repeated_spatial_support(self) -> None:
        midpoint = np.asarray(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 4, 0],
                [2, 0, 0],
                [2, 2, 0],
                [2, 4, 0],
            ],
            np.float32,
        )
        # Strong vertical edges establish two independent local cores.  The
        # three parallel horizontal proposals jointly describe their shared
        # frontier.
        first = np.asarray([0, 1, 3, 4, 0, 1, 2], np.int32)
        second = np.asarray([1, 2, 4, 5, 3, 4, 5], np.int32)
        score = np.asarray([0.9, 0.9, 0.9, 0.9, 0.6, 0.6, 0.6], np.float32)
        kind = np.asarray([6, 6, 6, 6, 7, 7, 7], np.uint8)
        settings = PairedProfileSurfaceSettings(
            enable_tangent_column_guard=False,
            minimum_frontier_span_sampling_steps=1.5,
        )

        selected, summary = _frontier_bundle_connectivity(
            midpoint,
            first,
            second,
            score,
            kind,
            sampling_stride_voxels=1.0,
            settings=settings,
        )

        np.testing.assert_array_equal(selected, np.ones(len(first), dtype=bool))
        self.assertEqual(summary["coreComponentCount"], 2)
        self.assertEqual(summary["acceptedBundleCount"], 1)

        sparse_selected, sparse_summary = _frontier_bundle_connectivity(
            midpoint,
            first[:5],
            second[:5],
            score[:5],
            kind[:5],
            sampling_stride_voxels=1.0,
            settings=settings,
        )
        self.assertFalse(bool(sparse_selected[-1]))
        self.assertEqual(sparse_summary["acceptedBundleCount"], 0)

    def test_reconnects_geometry_across_inherited_labels(self) -> None:
        count = 3
        bank = {
            "midpointXYZ": np.asarray([[0, 0, 5], [2, 0, 5], [4, 0, 5]], np.float32),
            "normalXYZ": np.tile((0, 0, 1), (count, 1)).astype(np.float32),
            "boundaryLowerXYZ": np.asarray([[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float32),
            "boundaryUpperXYZ": np.asarray([[0, 0, 10], [2, 0, 10], [4, 0, 10]], np.float32),
            "thicknessVoxels": np.full(count, 10, np.float32),
            "spatialKeyXYZ": np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.int32),
            "localEvidenceScore": np.full(count, 0.9, np.float32),
            "opposingNormalCosine": np.ones(count, np.float32),
            "isolatedConservative": np.ones(count, bool),
            "seedComponentId": np.arange(count, dtype=np.int32),
        }
        growth = {
            "selected": np.ones(count, bool),
            "selectedLabel": np.asarray([0, 1, 2], np.int32),
            "edgeFirstCandidate": np.asarray([0, 1], np.int32),
            "edgeSecondCandidate": np.asarray([1, 2], np.int32),
            "edgeAffinity": np.asarray([0.9, 0.9], np.float32),
            "edgeNormalDegrees": np.zeros(2, np.float32),
            "edgeMidpointHeightSamplingSteps": np.zeros(2, np.float32),
            "edgeBoundaryHeightSamplingSteps": np.zeros(2, np.float32),
            "edgeThicknessDifferenceSamplingSteps": np.zeros(2, np.float32),
        }
        macro_manifest = {"identity": {"resolvedScale": {"supportSamplingSteps": 4}}}
        macro = {
            "binKeyXYZ": np.asarray([[0, 0, 0]], np.int32),
            "normalXYZ": np.asarray([[0, 0, 1]], np.float32),
            "orientationConfidence": np.asarray([1.0], np.float32),
            "trusted": np.asarray([True]),
        }

        arrays, summary = build_direct_paired_profile_surface(
            bank,
            growth,
            macro_manifest,
            macro,
            settings=PairedProfileSurfaceSettings(
                minimum_edge_affinity=0.5,
                enable_geometric_closure=False,
                enable_tangent_column_guard=False,
            ),
        )

        self.assertEqual(summary["counts"]["componentCount"], 1)
        self.assertEqual(summary["counts"]["crossOriginalGrowthLabelEdgeCount"], 2)
        np.testing.assert_array_equal(arrays["componentId"], (0, 0, 0))


if __name__ == "__main__":
    unittest.main()
