from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.clear_ribbon_feedback import (
    SEED_SOURCE_BASELINE,
    SEED_SOURCE_NEW_CLEAR_CORE,
    audit_baseline_preservation,
    build_clear_ribbon_feedback_seeds,
)
from backend.cubical.clear_ribbon_selection import (
    SELECTION_CLASS_NEW_CLEAR_CORE,
    SELECTION_CLASS_UPSTREAM_ANCHOR,
)


class ClearRibbonFeedbackTests(unittest.TestCase):
    def test_feedback_uses_only_new_cores_and_defers_endpoint_conflict(
        self,
    ) -> None:
        baseline = {
            "seedAssemblyLabel": np.asarray((7, -1, -1, -1, -1), dtype=np.int32)
        }
        ribbons = {
            "lowerInterface": np.asarray((0, 2, 3), dtype=np.int32),
            "upperInterface": np.asarray((1, 3, 4), dtype=np.int32),
            "lowerComponentSeedLabelCount": np.zeros(3, dtype=np.uint16),
            "upperComponentSeedLabelCount": np.zeros(3, dtype=np.uint16),
        }
        selection = {
            "selected": np.ones(3, dtype=np.uint8),
            "selectionClass": np.asarray(
                (
                    SELECTION_CLASS_UPSTREAM_ANCHOR,
                    SELECTION_CLASS_NEW_CLEAR_CORE,
                    SELECTION_CLASS_NEW_CLEAR_CORE,
                ),
                dtype=np.uint8,
            ),
            "selectedAssemblyLabel": np.asarray((7, 10, 20), dtype=np.int32),
        }
        seeds, stats = build_clear_ribbon_feedback_seeds(
            baseline, ribbons, selection
        )
        np.testing.assert_array_equal(
            seeds["effectiveSeedLabel"], (7, -1, 10, -1, 20)
        )
        self.assertEqual(int(seeds["seedSource"][0]), SEED_SOURCE_BASELINE)
        self.assertEqual(
            int(seeds["seedSource"][2]), SEED_SOURCE_NEW_CLEAR_CORE
        )
        self.assertEqual(int(seeds["ribbonSeedConflict"][3]), 1)
        self.assertEqual(stats["acceptedRibbonSeedInterfaceCount"], 2)
        self.assertEqual(stats["conflictingRibbonSeedInterfaceCount"], 1)

    def test_feedback_never_overwrites_a_baseline_seed(self) -> None:
        baseline = {
            "seedAssemblyLabel": np.asarray((7, -1), dtype=np.int32)
        }
        ribbons = {
            "lowerInterface": np.asarray((0,), dtype=np.int32),
            "upperInterface": np.asarray((1,), dtype=np.int32),
            "lowerComponentSeedLabelCount": np.zeros(1, dtype=np.uint16),
            "upperComponentSeedLabelCount": np.zeros(1, dtype=np.uint16),
        }
        selection = {
            "selected": np.ones(1, dtype=np.uint8),
            "selectionClass": np.asarray(
                (SELECTION_CLASS_NEW_CLEAR_CORE,), dtype=np.uint8
            ),
            "selectedAssemblyLabel": np.asarray((10,), dtype=np.int32),
        }
        seeds, stats = build_clear_ribbon_feedback_seeds(
            baseline, ribbons, selection
        )
        np.testing.assert_array_equal(seeds["effectiveSeedLabel"], (7, 10))
        self.assertEqual(int(seeds["ribbonSeedConflict"][0]), 1)
        self.assertEqual(stats["acceptedRibbonSeedInterfaceCount"], 1)

    def test_feedback_rejects_a_false_new_seeded_component(self) -> None:
        baseline = {"seedAssemblyLabel": np.full(2, -1, dtype=np.int32)}
        ribbons = {
            "lowerInterface": np.asarray((0,), dtype=np.int32),
            "upperInterface": np.asarray((1,), dtype=np.int32),
            "lowerComponentSeedLabelCount": np.zeros(1, dtype=np.uint16),
            "upperComponentSeedLabelCount": np.ones(1, dtype=np.uint16),
        }
        selection = {
            "selected": np.ones(1, dtype=np.uint8),
            "selectionClass": np.asarray(
                (SELECTION_CLASS_NEW_CLEAR_CORE,), dtype=np.uint8
            ),
            "selectedAssemblyLabel": np.asarray((10,), dtype=np.int32),
        }
        with self.assertRaises(ValueError):
            build_clear_ribbon_feedback_seeds(baseline, ribbons, selection)

    def test_baseline_preservation_detects_loss_or_label_change(self) -> None:
        baseline = {
            "selected": np.asarray((1, 1, 0), dtype=np.uint8),
            "selectedLabel": np.asarray((7, 8, -1), dtype=np.int32),
        }
        refined = {
            "selected": np.asarray((1, 1, 1), dtype=np.uint8),
            "selectedLabel": np.asarray((7, 8, 10), dtype=np.int32),
        }
        stats = audit_baseline_preservation(baseline, refined)
        self.assertEqual(stats["newlySelectedInterfaceCount"], 1)

        refined["selectedLabel"][1] = 9
        with self.assertRaises(RuntimeError):
            audit_baseline_preservation(baseline, refined)

        refined["selectedLabel"][1] = 8
        refined["selected"][1] = 0
        with self.assertRaises(RuntimeError):
            audit_baseline_preservation(baseline, refined)


if __name__ == "__main__":
    unittest.main()
