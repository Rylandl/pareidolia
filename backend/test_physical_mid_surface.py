from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.physical_mid_surface import (
    PhysicalMidSurfaceSettings,
    pair_physical_boundary_faces,
)


class PhysicalMidSurfaceTests(unittest.TestCase):
    def test_parallel_boundary_grids_pair_one_to_one(self) -> None:
        x, y = np.meshgrid(np.arange(4) * 2.0, np.arange(4) * 2.0)
        xy = np.column_stack((x.reshape(-1), y.reshape(-1)))
        lower = np.column_stack((xy, np.zeros(len(xy))))
        upper = np.column_stack((xy, np.full(len(xy), 10.0)))
        position = np.concatenate((lower, upper))
        normal = np.concatenate(
            (
                np.tile((0.0, 0.0, 1.0), (len(lower), 1)),
                np.tile((0.0, 0.0, -1.0), (len(upper), 1)),
            )
        )
        label = np.full(len(position), 7, dtype=np.int32)
        side = np.concatenate(
            (
                np.zeros(len(lower), dtype=np.uint8),
                np.ones(len(upper), dtype=np.uint8),
            )
        )
        result = pair_physical_boundary_faces(
            position,
            normal,
            label,
            side,
            {7: np.asarray((9.8, 10.0, 10.2))},
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(),
        )
        self.assertEqual(len(result["lowerNode"]), 16)
        np.testing.assert_array_equal(
            result["upperNode"], result["lowerNode"] + 16
        )
        np.testing.assert_allclose(result["thicknessVoxels"], 10.0)
        np.testing.assert_allclose(result["tangentResidualSamplingSteps"], 0.0)

    def test_same_sheet_requires_opposite_boundary_normal(self) -> None:
        result = pair_physical_boundary_faces(
            np.asarray(((0.0, 0.0, 0.0), (0.0, 0.0, 10.0))),
            np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0))),
            np.asarray((3, 3), dtype=np.int32),
            np.asarray((0, 1), dtype=np.uint8),
            {3: np.asarray((10.0,))},
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(),
        )
        self.assertEqual(len(result["lowerNode"]), 0)

    def test_local_profile_thickness_rejects_a_nearer_fold_branch(self) -> None:
        position = np.asarray(
            ((0.0, 0.0, 0.0), (0.0, 0.0, 6.0), (0.0, 0.0, 10.0))
        )
        result = pair_physical_boundary_faces(
            position,
            np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, -1.0))
            ),
            np.asarray((5, 5, 5), dtype=np.int32),
            np.asarray((0, 1, 1), dtype=np.uint8),
            {5: np.asarray((6.0, 10.0))},
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(),
            local_thickness_prior=np.asarray((10.0, 6.0, 10.0)),
        )
        np.testing.assert_array_equal(result["lowerNode"], (0,))
        np.testing.assert_array_equal(result["upperNode"], (2,))
        np.testing.assert_allclose(result["thicknessVoxels"], 10.0)


if __name__ == "__main__":
    unittest.main()
