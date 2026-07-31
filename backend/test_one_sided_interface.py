from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.one_sided_interface import (
    OneSidedInterfaceSettings,
    match_paired_surface_boundaries,
)


class OneSidedInterfaceTests(unittest.TestCase):
    def test_both_signed_faces_anchor_to_the_same_surface_identity(self) -> None:
        seed, stats = match_paired_surface_boundaries(
            interface_position_world=np.asarray(
                ((0.0, 0.0, 0.0), (0.0, 0.0, 2.0)), dtype=np.float32
            ),
            interface_normal=np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float32
            ),
            interface_processing_key=np.asarray(
                ((0, 0, 0), (0, 0, 2)), dtype=np.int32
            ),
            bank={
                "boundaryLowerXYZ": np.asarray(
                    ((0.0, 0.0, 0.0),), dtype=np.float32
                ),
                "boundaryUpperXYZ": np.asarray(
                    ((0.0, 0.0, 2.0),), dtype=np.float32
                ),
                "normalXYZ": np.asarray(
                    ((0.0, 0.0, 1.0),), dtype=np.float32
                ),
            },
            growth={
                "selected": np.asarray((1,), dtype=np.uint8),
                "selectedLabel": np.asarray((7,), dtype=np.int32),
            },
            processing_start_xyz=np.zeros(3, dtype=np.float32),
            source_origin_xyz=np.zeros(3, dtype=np.float32),
            processing_shape_sampling_xyz=(1, 1, 3),
            stride=1,
            settings=OneSidedInterfaceSettings(),
        )
        np.testing.assert_array_equal(seed["seedSurfaceLabel"], (7, 7))
        np.testing.assert_array_equal(seed["seedBoundarySide"], (0, 1))
        self.assertEqual(stats["matchedPairedBoundaryEndpointCount"], 2)

    def test_opposite_signed_normal_does_not_anchor(self) -> None:
        seed, stats = match_paired_surface_boundaries(
            interface_position_world=np.asarray(
                ((0.0, 0.0, 0.0),), dtype=np.float32
            ),
            interface_normal=np.asarray(
                ((0.0, 0.0, -1.0),), dtype=np.float32
            ),
            interface_processing_key=np.asarray(((0, 0, 0),), dtype=np.int32),
            bank={
                "boundaryLowerXYZ": np.asarray(
                    ((0.0, 0.0, 0.0),), dtype=np.float32
                ),
                "boundaryUpperXYZ": np.asarray(
                    ((0.0, 0.0, 2.0),), dtype=np.float32
                ),
                "normalXYZ": np.asarray(
                    ((0.0, 0.0, 1.0),), dtype=np.float32
                ),
            },
            growth={
                "selected": np.asarray((1,), dtype=np.uint8),
                "selectedLabel": np.asarray((7,), dtype=np.int32),
            },
            processing_start_xyz=np.zeros(3, dtype=np.float32),
            source_origin_xyz=np.zeros(3, dtype=np.float32),
            processing_shape_sampling_xyz=(1, 1, 3),
            stride=1,
            settings=OneSidedInterfaceSettings(),
        )
        self.assertEqual(int(seed["seedSurfaceLabel"][0]), -1)
        self.assertEqual(stats["matchedPairedBoundaryEndpointCount"], 0)


if __name__ == "__main__":
    unittest.main()
