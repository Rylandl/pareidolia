from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.clear_ribbon import (
    ClearRibbonSettings,
    build_clear_ribbon_graph,
    build_clear_ribbons,
    label_clear_ribbon_components,
)


class ClearRibbonTests(unittest.TestCase):
    def test_duplicate_face_pair_prefers_selected_immutable_profile(self) -> None:
        interface = {
            "positionXYZ": np.asarray(
                ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)), dtype=np.float32
            ),
            "signedNormalXYZ": np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float32
            ),
            "processingKeyXYZ": np.asarray(
                ((0, 0, 0), (0, 0, 2)), dtype=np.int32
            ),
        }
        paired_bank = {
            "normalXYZ": np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float32
            ),
            "boundaryLowerXYZ": np.asarray(
                ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)), dtype=np.float32
            ),
            "boundaryUpperXYZ": np.asarray(
                ((0.0, 0.0, 2.0), (0.0, 0.0, 2.0)), dtype=np.float32
            ),
            "midpointXYZ": np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float32
            ),
            "thicknessVoxels": np.asarray((2.0, 2.0), dtype=np.float32),
            "spatialKeyXYZ": np.asarray(
                ((0, 0, 1), (0, 0, 1)), dtype=np.int32
            ),
            "localEvidenceScore": np.asarray((1.0, 0.8), dtype=np.float32),
            "seedComponentId": np.asarray((-1, 4), dtype=np.int32),
        }
        paired_growth = {
            "selected": np.asarray((0, 1), dtype=np.uint8),
            "selectedLabel": np.asarray((-1, 7), dtype=np.int32),
        }
        one_sided_growth = {
            "interfaceContinuityComponent": np.asarray((0, 1), dtype=np.int32),
            "componentSeedLabelCount": np.asarray((1, 1), dtype=np.uint16),
            "surfaceLabel": np.asarray((7,), dtype=np.int32),
            "surfaceAssemblyLabel": np.asarray((7,), dtype=np.int32),
        }
        ribbons, stats = build_clear_ribbons(
            interface,
            one_sided_growth,
            paired_bank,
            paired_growth,
            processing_start_xyz=np.zeros(3, dtype=np.float32),
            source_origin_xyz=np.zeros(3, dtype=np.float32),
            processing_shape_sampling_xyz=(1, 1, 3),
            stride=1,
            settings=ClearRibbonSettings(),
        )
        self.assertEqual(stats["uniqueRibbonCount"], 1)
        self.assertEqual(int(ribbons["pairedCandidateIndex"][0]), 1)
        self.assertEqual(int(ribbons["alternativeProfileCount"][0]), 2)
        self.assertEqual(int(ribbons["selectedAssemblyLabel"][0]), 7)

    def test_graph_collapse_keeps_strongest_duplicate_continuity_edge(self) -> None:
        graph, stats = build_clear_ribbon_graph(
            {"pairedCandidateToRibbon": np.asarray((0, 0, 1), dtype=np.int32)},
            {
                "edgeFirstCandidate": np.asarray((0, 1), dtype=np.int32),
                "edgeSecondCandidate": np.asarray((2, 2), dtype=np.int32),
                "edgeAffinity": np.asarray((0.5, 0.9), dtype=np.float32),
                "edgeNormalDegrees": np.asarray((10.0, 2.0), dtype=np.float32),
            },
            settings=ClearRibbonSettings(),
        )
        self.assertEqual(stats["continuityEdgeCount"], 1)
        self.assertAlmostEqual(float(graph["edgeAffinity"][0]), 0.9)
        self.assertAlmostEqual(float(graph["edgeNormalDegrees"][0]), 2.0)

    def test_component_census_preserves_key_collisions_and_unseeded_state(
        self,
    ) -> None:
        components, stats = label_clear_ribbon_components(
            {
                "pairedCandidateIndex": np.arange(4, dtype=np.int32),
                "selectedAssemblyLabel": np.asarray((7, 7, -1, -1)),
                "lockedPairedSeed": np.asarray((1, 1, 0, 0), dtype=np.uint8),
                "spatialKeyXYZ": np.asarray(
                    ((0, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0)),
                    dtype=np.int32,
                ),
            },
            {
                "edgeFirstRibbon": np.asarray((0, 1), dtype=np.int32),
                "edgeSecondRibbon": np.asarray((1, 2), dtype=np.int32),
            },
            processing_shape_sampling_xyz=(3, 1, 1),
            settings=ClearRibbonSettings(),
        )
        self.assertEqual(stats["componentCount"], 2)
        self.assertEqual(stats["states"]["singleAssembly"]["componentCount"], 1)
        self.assertEqual(stats["states"]["unseeded"]["componentCount"], 1)
        component = int(components["ribbonComponent"][0])
        self.assertEqual(
            int(components["componentSpatialKeyCollisionCount"][component]), 1
        )


if __name__ == "__main__":
    unittest.main()
