from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.laminar_ribbon import (
    LaminarRibbonSettings,
    _deduplicate_pair_keys,
    discover_laminar_ribbons,
)


class LaminarRibbonTests(unittest.TestCase):
    def test_pair_key_deduplication_prefers_conservative_then_evidence(self) -> None:
        candidate = np.asarray((0, 1, 2, 3), dtype=np.int32)
        pair = np.asarray((5, 5, 5, 7), dtype=np.uint64)
        key = np.asarray(((2, 3, 4), (2, 3, 4), (2, 3, 4), (2, 3, 4)))
        conservative = np.asarray((0, 1, 1, 0), dtype=bool)
        evidence = np.asarray((0.99, 0.7, 0.8, 0.6), dtype=np.float32)
        selected = _deduplicate_pair_keys(
            candidate, pair, key, conservative, evidence
        )
        self.assertEqual(set(selected.tolist()), {2, 3})

    def test_repeated_macro_aligned_profiles_define_one_face_pair(self) -> None:
        count = 4
        y = np.arange(count, dtype=np.float32)
        lower = np.column_stack((np.ones(count), y, np.ones(count))).astype(
            np.float32
        )
        upper = lower.copy()
        upper[:, 0] = 3.0
        midpoint = 0.5 * (lower + upper)
        profile_normal = np.tile((1.0, 0.0, 0.0), (count, 1)).astype(
            np.float32
        )
        bank = {
            "midpointXYZ": midpoint,
            "normalXYZ": profile_normal,
            "boundaryLowerXYZ": lower,
            "boundaryUpperXYZ": upper,
            "thicknessVoxels": np.full(count, 2.0, dtype=np.float32),
            "spatialKeyXYZ": np.rint(midpoint).astype(np.int32),
            "localEvidenceScore": np.full(count, 0.9, dtype=np.float32),
            "isolatedConservative": np.ones(count, dtype=np.uint8),
            "seedComponentId": np.zeros(count, dtype=np.int32),
        }
        interface_position = np.concatenate((lower, upper), axis=0)
        interface_normal = np.concatenate(
            (profile_normal, -profile_normal), axis=0
        )
        interfaces = {
            "positionXYZ": interface_position,
            "signedNormalXYZ": interface_normal,
            "processingKeyXYZ": np.rint(interface_position).astype(np.int32),
        }
        surface = {
            "interfaceIndex": np.arange(2 * count, dtype=np.int32),
            "componentId": np.concatenate(
                (
                    np.zeros(count, dtype=np.int32),
                    np.ones(count, dtype=np.int32),
                )
            ),
        }
        macro_manifest = {
            "identity": {"resolvedScale": {"supportSamplingSteps": 8}}
        }
        macro = {
            "binKeyXYZ": np.asarray(((0, 0, 0),), dtype=np.int32),
            "normalXYZ": np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            "orientationConfidence": np.asarray((0.9,), dtype=np.float32),
            "trusted": np.asarray((1,), dtype=np.uint8),
        }
        discovery, records, summary = discover_laminar_ribbons(
            bank,
            surface,
            interfaces,
            macro_manifest,
            macro,
            processing_start_xyz=np.zeros(3),
            source_origin_xyz=np.zeros(3),
            processing_shape_sampling_xyz=(6, 8, 4),
            sampling_stride_voxels=1,
            settings=LaminarRibbonSettings(
                minimum_support_profiles=4,
                minimum_conservative_support_profiles=4,
                minimum_support_extent_sampling_steps=2.0,
            ),
        )
        self.assertEqual(summary["selectedRibbonCount"], 1)
        self.assertEqual(summary["selectedSupportProfileCount"], 4)
        self.assertEqual(records[0]["faceComponentFirst"], 0)
        self.assertEqual(records[0]["faceComponentSecond"], 1)
        np.testing.assert_array_equal(
            discovery["supportRibbonLabel"], np.zeros(count, dtype=np.int32)
        )

    def test_transverse_profiles_fail_the_generic_macro_guard(self) -> None:
        lower = np.asarray(((1.0, 1.0, 1.0),), dtype=np.float32)
        upper = np.asarray(((1.0, 3.0, 1.0),), dtype=np.float32)
        normal = np.asarray(((0.0, 1.0, 0.0),), dtype=np.float32)
        bank = {
            "midpointXYZ": 0.5 * (lower + upper),
            "normalXYZ": normal,
            "boundaryLowerXYZ": lower,
            "boundaryUpperXYZ": upper,
            "thicknessVoxels": np.asarray((2.0,), dtype=np.float32),
            "spatialKeyXYZ": np.asarray(((1, 2, 1),), dtype=np.int32),
            "localEvidenceScore": np.asarray((0.95,), dtype=np.float32),
            "isolatedConservative": np.asarray((1,), dtype=np.uint8),
            "seedComponentId": np.asarray((0,), dtype=np.int32),
        }
        interfaces = {
            "positionXYZ": np.concatenate((lower, upper)),
            "signedNormalXYZ": np.concatenate((normal, -normal)),
            "processingKeyXYZ": np.asarray(((1, 1, 1), (1, 3, 1))),
        }
        surface = {
            "interfaceIndex": np.asarray((0, 1), dtype=np.int32),
            "componentId": np.asarray((0, 1), dtype=np.int32),
        }
        macro_manifest = {
            "identity": {"resolvedScale": {"supportSamplingSteps": 8}}
        }
        macro = {
            "binKeyXYZ": np.asarray(((0, 0, 0),), dtype=np.int32),
            "normalXYZ": np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            "orientationConfidence": np.asarray((1.0,), dtype=np.float32),
            "trusted": np.asarray((1,), dtype=np.uint8),
        }
        _, records, summary = discover_laminar_ribbons(
            bank,
            surface,
            interfaces,
            macro_manifest,
            macro,
            processing_start_xyz=np.zeros(3),
            source_origin_xyz=np.zeros(3),
            processing_shape_sampling_xyz=(5, 5, 3),
            sampling_stride_voxels=1,
            settings=LaminarRibbonSettings(
                minimum_support_profiles=1,
                minimum_conservative_support_profiles=1,
                minimum_support_extent_sampling_steps=0.1,
            ),
        )
        self.assertEqual(records, [])
        self.assertEqual(summary["macroEligibleProfileCount"], 0)


if __name__ == "__main__":
    unittest.main()
