from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.clear_core_interface_refinement import (
    SEED_SOURCE_FROZEN_INTERFACE,
    SEED_SOURCE_PAIRED_ENDPOINT,
    build_paired_endpoint_interface_seeds,
)
from backend.cubical.clear_ribbon_paired_feedback import (
    BOUNDARY_MATCH_NONE,
    BOUNDARY_MATCH_UNOWNED,
    SELECTION_CLASS_NEW_CLEAR_CORE,
    SELECTION_CLASS_PAIRED_GROWTH,
)


def _interfaces(count: int) -> dict[str, np.ndarray]:
    return {"seedConflict": np.zeros(count, dtype=np.uint8)}


def _baseline(
    count: int, selected: tuple[int, ...] = ()
) -> dict[str, np.ndarray]:
    selected_mask = np.zeros(count, dtype=np.uint8)
    selected_label = np.full(count, -1, dtype=np.int32)
    for interface, label in enumerate(selected):
        selected_mask[interface] = 1
        selected_label[interface] = label
    return {"selected": selected_mask, "selectedLabel": selected_label}


class ClearCoreInterfaceRefinementTests(unittest.TestCase):
    def test_safe_endpoint_observations_deduplicate_into_new_seeds(self) -> None:
        paired = {
            "selectionClass": np.asarray(
                (
                    SELECTION_CLASS_NEW_CLEAR_CORE,
                    SELECTION_CLASS_PAIRED_GROWTH,
                    0,
                ),
                dtype=np.uint8,
            ),
            "selected": np.asarray((1, 1, 0), dtype=np.uint8),
            "selectedLabel": np.asarray((10, 10, -1), dtype=np.int32),
            "lowerMatchedInterface": np.asarray((1, 1, -1), dtype=np.int32),
            "upperMatchedInterface": np.asarray((2, 3, -1), dtype=np.int32),
            "lowerBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_UNOWNED, BOUNDARY_MATCH_UNOWNED, 0),
                dtype=np.uint8,
            ),
            "upperBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_UNOWNED, BOUNDARY_MATCH_UNOWNED, 0),
                dtype=np.uint8,
            ),
        }
        seeds, stats = build_paired_endpoint_interface_seeds(
            paired,
            _baseline(4, (7,)),
            _interfaces(4),
        )
        self.assertEqual(stats["pairedEndpointObservationCount"], 4)
        self.assertEqual(stats["uniquePairedEndpointInterfaceCount"], 3)
        self.assertEqual(stats["acceptedPairedEndpointSeedCount"], 3)
        np.testing.assert_array_equal(
            seeds["effectiveSeedLabel"], (7, 10, 10, 10)
        )
        np.testing.assert_array_equal(
            seeds["refinementSeedSource"],
            (
                SEED_SOURCE_FROZEN_INTERFACE,
                SEED_SOURCE_PAIRED_ENDPOINT,
                SEED_SOURCE_PAIRED_ENDPOINT,
                SEED_SOURCE_PAIRED_ENDPOINT,
            ),
        )

    def test_two_labels_for_one_interface_are_deferred_as_a_conflict(self) -> None:
        paired = {
            "selectionClass": np.asarray(
                (
                    SELECTION_CLASS_NEW_CLEAR_CORE,
                    SELECTION_CLASS_NEW_CLEAR_CORE,
                ),
                dtype=np.uint8,
            ),
            "selected": np.ones(2, dtype=np.uint8),
            "selectedLabel": np.asarray((10, 20), dtype=np.int32),
            "lowerMatchedInterface": np.asarray((1, 1), dtype=np.int32),
            "upperMatchedInterface": np.asarray((-1, -1), dtype=np.int32),
            "lowerBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_UNOWNED, BOUNDARY_MATCH_UNOWNED),
                dtype=np.uint8,
            ),
            "upperBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_NONE, BOUNDARY_MATCH_NONE), dtype=np.uint8
            ),
        }
        seeds, stats = build_paired_endpoint_interface_seeds(
            paired,
            _baseline(2),
            _interfaces(2),
        )
        self.assertEqual(stats["acceptedPairedEndpointSeedCount"], 0)
        self.assertEqual(stats["conflictingPairedEndpointSeedCount"], 1)
        self.assertEqual(int(seeds["effectiveSeedLabel"][1]), -1)
        self.assertEqual(int(seeds["pairedEndpointSeedConflict"][1]), 1)

    def test_opposite_sides_at_one_interface_keep_side_ambiguous(self) -> None:
        paired = {
            "selectionClass": np.asarray(
                (SELECTION_CLASS_NEW_CLEAR_CORE,), dtype=np.uint8
            ),
            "selected": np.ones(1, dtype=np.uint8),
            "selectedLabel": np.asarray((10,), dtype=np.int32),
            "lowerMatchedInterface": np.asarray((1,), dtype=np.int32),
            "upperMatchedInterface": np.asarray((1,), dtype=np.int32),
            "lowerBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_UNOWNED,), dtype=np.uint8
            ),
            "upperBoundaryOwnershipClass": np.asarray(
                (BOUNDARY_MATCH_UNOWNED,), dtype=np.uint8
            ),
        }
        seeds, stats = build_paired_endpoint_interface_seeds(
            paired,
            _baseline(2),
            _interfaces(2),
        )
        self.assertEqual(stats["acceptedPairedEndpointSeedCount"], 1)
        self.assertEqual(stats["ambiguousBoundarySideSeedCount"], 1)
        self.assertEqual(
            int(seeds["pairedEndpointSeedBoundarySide"][1]), 255
        )


if __name__ == "__main__":
    unittest.main()
