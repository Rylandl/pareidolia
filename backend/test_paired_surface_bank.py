from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.paired_surface_bank import (
    PairedSurfaceBankSettings,
    _match_isolated_samples,
    suppress_reciprocal_profile_duplicates,
)


class PairedSurfaceBankTests(unittest.TestCase):
    def test_isolated_match_searches_neighbor_when_nominal_key_is_unrelated(self) -> None:
        matched, stats = _match_isolated_samples(
            raw_midpoint_world=np.asarray(
                ((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)), dtype=np.float64
            ),
            raw_normal=np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 0.0, 1.0)), dtype=np.float64
            ),
            raw_thickness_voxels=np.asarray((8.0, 2.0), dtype=np.float64),
            raw_key=np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.int32),
            slab={
                "midpointXYZ": np.asarray(((0.5, 0.5, 0.5),)),
                "normalXYZ": np.asarray(((0.0, 0.0, 1.0),)),
                "thicknessVoxels": np.asarray((2.0,)),
            },
            processing_shape_sampling_xyz=(2, 1, 1),
            processing_start_xyz=np.zeros(3),
            source_origin_xyz=np.zeros(3),
            stride=2,
        )
        np.testing.assert_array_equal(matched, (-1, 0))
        self.assertEqual(stats["matchedIsolatedSampleCount"], 1)
        self.assertAlmostEqual(float(stats["maximumMatchCost"]), 0.0)

    def test_anchor_wins_duplicate_suppression_but_distinct_hypothesis_survives(
        self,
    ) -> None:
        midpoint = np.zeros((3, 3), dtype=np.float32)
        normal = np.asarray(
            ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0), (1.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        lower = np.asarray(
            ((0.0, 0.0, -1.0), (0.0, 0.0, -1.0), (-1.0, 0.0, 0.0)),
            dtype=np.float32,
        )
        upper = -lower
        retained, alternative, stats = suppress_reciprocal_profile_duplicates(
            midpoint,
            normal,
            lower,
            upper,
            np.full(3, 2.0, dtype=np.float32),
            np.zeros((3, 3), dtype=np.int32),
            np.asarray((0.1, 0.9, 0.8), dtype=np.float32),
            np.ones(3, dtype=np.uint8),
            np.asarray((0, -1, -1), dtype=np.int32),
            stride=1,
            processing_shape_sampling_xyz=(1, 1, 1),
            settings=PairedSurfaceBankSettings(),
        )
        np.testing.assert_array_equal(retained, (0, 2))
        np.testing.assert_array_equal(alternative, (0, 1))
        self.assertEqual(stats["reciprocalDuplicateCount"], 1)
        self.assertEqual(stats["keysWithMultipleRetainedHypotheses"], 1)


if __name__ == "__main__":
    unittest.main()
