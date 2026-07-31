from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.physical_ribbon_bank import (
    PhysicalRibbonBankSettings,
    build_physical_ribbon_bank,
)
from backend.cubical.physical_ribbon_configuration import (
    PhysicalRibbonConfigurationSettings,
    build_profile_crossing_conflicts,
    optimize_physical_ribbon_configuration,
)


class PhysicalRibbonBankTests(unittest.TestCase):
    def test_opposing_first_hits_form_one_label_free_ribbon(self) -> None:
        interfaces = {
            "positionXYZ": np.asarray(
                ((5.0, 5.0, 5.0), (15.0, 5.0, 5.0)),
                dtype=np.float32,
            ),
            "signedNormalXYZ": np.asarray(
                ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
                dtype=np.float32,
            ),
            "processingKeyXYZ": np.asarray(
                ((5, 5, 5), (15, 5, 5)), dtype=np.int32
            ),
            "localEvidenceScore": np.ones(2, dtype=np.float32),
        }
        ribbons, stats = build_physical_ribbon_bank(
            interfaces,
            processing_shape_sampling_xyz=(24, 16, 16),
            processing_world_start_xyz=np.zeros(3, dtype=np.float32),
            sampling_stride_voxels=1,
            voxel_size_microns=1.0,
            settings=PhysicalRibbonBankSettings(
                minimum_sheet_thickness_microns=8.0,
                maximum_sheet_thickness_microns=12.0,
                ray_search_radius_sampling_steps=1,
                batch_interface_count=2,
            ),
        )
        self.assertEqual(stats["undirectedCandidateCount"], 1)
        self.assertEqual(stats["mutualFirstHitCount"], 1)
        self.assertFalse(stats["identityLabelsUsed"])
        self.assertEqual(int(ribbons["mutualFirstHit"][0]), 1)


class PhysicalRibbonConfigurationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.interfaces = {
            "positionXYZ": np.asarray(
                (
                    (0.0, 0.0, 0.0),
                    (10.0, 0.0, 0.0),
                    (5.0, -5.0, 0.0),
                    (5.0, 5.0, 0.0),
                ),
                dtype=np.float32,
            )
        }
        self.ribbon = {
            "sourceInterface": np.asarray((0, 2), dtype=np.int32),
            "targetInterface": np.asarray((1, 3), dtype=np.int32),
            "physicalEvidenceScore": np.asarray((0.9, 0.7), dtype=np.float32),
            "sourceRayRank": np.zeros(2, dtype=np.int16),
            "targetRayRank": np.zeros(2, dtype=np.int16),
            "mutualFirstHit": np.zeros(2, dtype=np.uint8),
            "midpointXYZ": np.asarray(
                ((5.0, 0.0, 0.0), (5.0, 0.0, 0.0)), dtype=np.float32
            ),
        }
        self.continuity = {
            "frontierRibbonCandidate": np.asarray((0, 1), dtype=np.int32),
            "edgeFirstFrontierIndex": np.empty(0, dtype=np.int32),
            "edgeSecondFrontierIndex": np.empty(0, dtype=np.int32),
            "edgeScore": np.empty(0, dtype=np.float32),
            "selected": np.ones(2, dtype=np.uint8),
        }

    def test_exact_interior_profile_crossing_is_detected(self) -> None:
        crossings, stats = build_profile_crossing_conflicts(
            self.ribbon,
            self.interfaces,
            self.continuity,
            processing_world_start_xyz=np.asarray((-10.0, -10.0, -10.0)),
            processing_shape_sampling_xyz=(32, 32, 32),
            sampling_stride_voxels=1,
            settings=PhysicalRibbonConfigurationSettings(),
        )
        self.assertEqual(stats["exactInteriorCrossingConflictCount"], 1)
        np.testing.assert_array_equal(
            crossings["crossingFirstFrontierIndex"], (0,)
        )
        np.testing.assert_array_equal(
            crossings["crossingSecondFrontierIndex"], (1,)
        )

    def test_crossing_configuration_keeps_only_the_better_profile(self) -> None:
        crossings = {
            "crossingFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "crossingSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
            "crossingDistanceVoxels": np.zeros(1, dtype=np.float32),
            "crossingFirstParameter": np.full(1, 0.5, dtype=np.float32),
            "crossingSecondParameter": np.full(1, 0.5, dtype=np.float32),
        }
        solved, stats = optimize_physical_ribbon_configuration(
            self.ribbon,
            self.interfaces,
            self.continuity,
            crossings,
            settings=PhysicalRibbonConfigurationSettings(
                maximum_optimization_sweeps=1,
                maximum_hole_growth_sweeps=1,
            ),
        )
        np.testing.assert_array_equal(solved["selected"], (1, 0))
        self.assertEqual(stats["selectedCrossingConflictCount"], 0)
        self.assertEqual(stats["selectedInterfaceCount"], 2)


if __name__ == "__main__":
    unittest.main()
