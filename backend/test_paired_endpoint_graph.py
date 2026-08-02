import unittest

import numpy as np

from backend.cubical.paired_endpoint_graph import (
    build_paired_endpoint_continuity_graph,
)
from backend.cubical.paired_profile_surface import (
    PairedProfileSurfaceSettings,
    build_direct_paired_profile_surface,
)


class PairedEndpointGraphTests(unittest.TestCase):
    def test_preserves_one_boundary_when_the_opposite_crossing_changes(self) -> None:
        bank = {
            "spatialKeyXYZ": np.asarray([[0, 0, 0], [1, 0, 0]], np.int32),
            "normalXYZ": np.asarray([[0, 0, 1], [0, 0, 1]], np.float32),
            "boundaryLowerXYZ": np.asarray([[0, 0, 0], [2, 0, 0]], np.float32),
            "boundaryUpperXYZ": np.asarray([[0, 0, 10], [2, 0, 16]], np.float32),
        }

        graph, summary = build_paired_endpoint_continuity_graph(
            bank,
            np.ones(2, dtype=bool),
            sampling_stride_voxels=2.0,
            link_radius_sampling_steps=2.0,
            maximum_normal_degrees=30.0,
            maximum_endpoint_distance_sampling_steps=4.0,
            maximum_endpoint_height_sampling_steps=1.5,
            normal_scale_degrees=20.0,
            endpoint_height_scale_sampling_steps=0.75,
            endpoint_distance_scale_sampling_steps=3.0,
        )

        self.assertEqual(summary["endpointPairCount"], 1)
        self.assertGreater(float(graph["lowerAffinity"][0]), 0.0)
        self.assertEqual(float(graph["upperAffinity"][0]), 0.0)
        self.assertEqual(summary["oneEndpointOnlyMatchCount"], 1)

    def test_endpoint_context_selects_the_two_face_hypothesis(self) -> None:
        midpoint = np.asarray(
            [[0, 0, 5], [2, 0, 8], [2, 0, 5], [4, 0, 5]], np.float32
        )
        bank = {
            "midpointXYZ": midpoint,
            "normalXYZ": np.tile((0, 0, 1), (4, 1)).astype(np.float32),
            "boundaryLowerXYZ": np.asarray(
                [[0, 0, 0], [2, 0, 0], [2, 0, 0], [4, 0, 0]], np.float32
            ),
            "boundaryUpperXYZ": np.asarray(
                [[0, 0, 10], [2, 0, 16], [2, 0, 10], [4, 0, 10]], np.float32
            ),
            "thicknessVoxels": np.asarray([10, 16, 10, 10], np.float32),
            "spatialKeyXYZ": np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 0, 0], [2, 0, 0]], np.int32
            ),
            "localEvidenceScore": np.asarray([0.9, 0.99, 0.8, 0.9], np.float32),
            "opposingNormalCosine": np.ones(4, np.float32),
            "isolatedConservative": np.ones(4, bool),
            "seedComponentId": np.full(4, -1, np.int32),
        }
        growth = {
            "selected": np.asarray([True, True, False, True]),
            "selectedLabel": np.asarray([0, 0, -1, 0], np.int32),
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
            "centerXYZ": np.asarray([[0, 0, 0]], np.float32),
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
                candidate_selection_mode="endpoint-coordinate-ascent",
                component_solver_mode="connected-components",
                enable_geometric_closure=False,
                enable_tangent_column_guard=False,
            ),
        )

        np.testing.assert_array_equal(arrays["profileCandidateIndex"], (0, 2, 3))
        self.assertEqual(summary["selection"]["mode"], "endpoint-coordinate-ascent")
        self.assertGreater(summary["counts"]["selectedTwoEndpointContinuityEdgeCount"], 0)


if __name__ == "__main__":
    unittest.main()
