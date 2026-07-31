from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.paired_surface_growth import (
    PairedSurfaceGrowthSettings,
    _continuity_metrics,
    associate_seed_components,
    discover_seed_component_support,
)


class PairedSurfaceGrowthTests(unittest.TestCase):
    def test_opposite_normal_sign_preserves_both_physical_boundaries(self) -> None:
        bank = {
            "midpointXYZ": np.asarray(((0.0, 0.0, 1.0), (1.0, 0.0, 1.0))),
            "normalXYZ": np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0))),
            "boundaryLowerXYZ": np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 2.0))
            ),
            "boundaryUpperXYZ": np.asarray(
                ((0.0, 0.0, 2.0), (1.0, 0.0, 0.0))
            ),
            "thicknessVoxels": np.asarray((2.0, 2.0)),
        }
        metrics = _continuity_metrics(
            np.asarray((0,)), np.asarray((1,)), bank, stride=1
        )
        self.assertAlmostEqual(float(metrics["normalDegrees"][0]), 0.0)
        self.assertAlmostEqual(
            float(metrics["midpointHeightSamplingSteps"][0]), 0.0
        )
        self.assertAlmostEqual(
            float(metrics["lowerBoundaryHeightSamplingSteps"][0]), 0.0
        )
        self.assertAlmostEqual(
            float(metrics["upperBoundaryHeightSamplingSteps"][0]), 0.0
        )

    def test_foreign_support_reaches_but_does_not_cross_locked_seed(self) -> None:
        bank = {
            "localEvidenceScore": np.ones(4, dtype=np.float32),
            "seedComponentId": np.asarray((0, -1, 1, -1), dtype=np.int32),
        }
        graph = {
            "edgeFirstCandidate": np.asarray((0, 1, 2), dtype=np.int32),
            "edgeSecondCandidate": np.asarray((1, 2, 3), dtype=np.int32),
            "edgeAffinity": np.ones(3, dtype=np.float32),
        }
        support, _stats = discover_seed_component_support(
            bank,
            graph,
            locked_seed=np.asarray((True, False, True, False)),
            settings=PairedSurfaceGrowthSettings(),
        )
        labels = support["supportSeedComponent"]
        self.assertEqual(set(int(value) for value in labels[2]), {0, 1})
        self.assertEqual(int(labels[3, 0]), 1)
        self.assertEqual(int(labels[3, 1]), -1)

    def test_seed_association_requires_reciprocal_seed_reach(self) -> None:
        node_count = 10
        bank = {
            "seedComponentId": np.asarray(
                (0, 0, 1, 1, -1, -1, -1, -1, -1, -1), dtype=np.int32
            ),
            "midpointXYZ": np.column_stack(
                (
                    np.arange(node_count, dtype=np.float32),
                    np.zeros(node_count, dtype=np.float32),
                    np.zeros(node_count, dtype=np.float32),
                )
            ),
        }
        labels = np.tile(np.asarray((0, 1), dtype=np.int32), (node_count, 1))
        labels[2:4] = (1, 0)
        support = {
            "supportSeedComponent": labels,
            "supportPathBottleneck": np.tile(
                np.asarray((1.0, 0.8), dtype=np.float32), (node_count, 1)
            ),
        }
        settings = PairedSurfaceGrowthSettings(
            minimum_seed_component_size=1,
            minimum_seed_association_shared_candidates=4,
            minimum_seed_association_median_bottleneck=0.6,
            minimum_seed_association_reciprocal_candidates=1,
            minimum_seed_association_reciprocal_fraction=0.4,
            minimum_seed_association_extent_sampling_steps=0.1,
        )
        locked = np.asarray(
            (True, True, True, True, False, False, False, False, False, False)
        )
        assembly, arrays, stats = associate_seed_components(
            bank,
            support,
            locked_seed=locked,
            component_value=np.asarray((0, 1), dtype=np.int32),
            component_size=np.asarray((2, 2), dtype=np.int64),
            stride=1,
            settings=settings,
        )
        self.assertEqual(stats["acceptedPairCount"], 1)
        self.assertEqual(stats["seedAssemblyCount"], 1)
        self.assertTrue(np.all(assembly[locked] == 0))
        self.assertEqual(int(arrays["associationAccepted"][0]), 1)

        one_way_labels = labels.copy()
        one_way_labels[:2, 1] = -1
        one_way_support = {
            **support,
            "supportSeedComponent": one_way_labels,
        }
        assembly, arrays, stats = associate_seed_components(
            bank,
            one_way_support,
            locked_seed=locked,
            component_value=np.asarray((0, 1), dtype=np.int32),
            component_size=np.asarray((2, 2), dtype=np.int64),
            stride=1,
            settings=settings,
        )
        self.assertEqual(stats["acceptedPairCount"], 0)
        self.assertEqual(stats["seedAssemblyCount"], 2)
        self.assertEqual(set(int(value) for value in assembly[locked]), {0, 1})
        self.assertEqual(int(arrays["associationAccepted"][0]), 0)


if __name__ == "__main__":
    unittest.main()
