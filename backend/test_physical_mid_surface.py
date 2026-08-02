from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.contextual_profile_adoption import (
    adopt_contextual_profiles,
)
from backend.cubical.physical_mid_surface import (
    PhysicalMidSurfaceSettings,
    one_sided_mid_surface_proxies,
    pair_physical_boundary_faces,
    propagate_profile_thickness_prior,
)


class PhysicalMidSurfaceTests(unittest.TestCase):
    def test_one_sided_proxies_offset_inward_and_skip_dense_pairs(self) -> None:
        result = one_sided_mid_surface_proxies(
            np.asarray(
                ((0.0, 0.0, 0.0), (2.0, 0.0, 10.0), (4.0, 0.0, 0.0))
            ),
            np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0), (0.0, 0.0, 1.0))
            ),
            np.asarray(
                ((0.0, 0.0, -1.0), (0.0, 0.0, 1.0), (0.0, 0.0, 1.0))
            ),
            np.asarray((3, 3, 3), dtype=np.int32),
            np.asarray((0, 1, 0), dtype=np.uint8),
            np.full(3, 10.0),
            np.asarray((2.0, 3.0, 1.0)),
            np.asarray((False, False, True)),
            settings=PhysicalMidSurfaceSettings(),
        )
        np.testing.assert_array_equal(result["sourceSurfaceNode"], (0, 1))
        np.testing.assert_allclose(
            result["midpointXYZ"], ((0.0, 0.0, 5.0), (2.0, 0.0, 5.0))
        )
        np.testing.assert_allclose(
            result["boundaryLowerXYZ"], ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0))
        )
        np.testing.assert_allclose(
            result["boundaryUpperXYZ"], ((0.0, 0.0, 10.0), (2.0, 0.0, 10.0))
        )

    def test_contextual_profiles_use_face_identity_orientation_and_key_exclusion(
        self,
    ) -> None:
        interface_key = []
        interface_position = []
        interface_normal = []
        for x in range(5):
            interface_key.extend(((x, 0, 0), (x, 0, 2)))
            interface_position.extend(
                ((2.0 * x + 0.5, 0.5, 0.5), (2.0 * x + 0.5, 0.5, 4.5))
            )
            interface_normal.extend(((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)))
        interface_key_array = np.asarray(interface_key, dtype=np.int32)
        interface_position_array = np.asarray(interface_position, dtype=np.float32)
        interface_normal_array = np.asarray(interface_normal, dtype=np.float32)
        surface_interface = np.asarray(
            [value for value in range(10) if value != 7], dtype=np.int32
        )
        surface_side = (surface_interface % 2).astype(np.uint8)
        surface_label = np.full(len(surface_interface), 7, dtype=np.int32)
        surface_label[surface_interface == 9] = 8

        lower = np.asarray(
            (
                (0.5, 0.5, 0.5),
                (2.5, 0.5, 0.5),
                (4.5, 0.5, 4.5),
                (6.5, 0.5, 0.5),
                (8.5, 0.5, 0.5),
                (2.5, 0.5, 0.5),
            ),
            dtype=np.float32,
        )
        upper = np.asarray(
            (
                (0.5, 0.5, 4.5),
                (2.5, 0.5, 4.5),
                (4.5, 0.5, 0.5),
                (6.5, 0.5, 4.5),
                (8.5, 0.5, 4.5),
                (2.5, 0.5, 4.5),
            ),
            dtype=np.float32,
        )
        normal = np.tile((0.0, 0.0, 1.0), (6, 1)).astype(np.float32)
        normal[2] *= -1.0
        result = adopt_contextual_profiles(
            {
                "boundaryLowerXYZ": lower,
                "boundaryUpperXYZ": upper,
                "normalXYZ": normal,
                "thicknessVoxels": np.full(6, 4.0, dtype=np.float32),
                "localEvidenceScore": np.asarray(
                    (0.9, 0.8, 0.75, 0.7, 0.9, 0.5), dtype=np.float32
                ),
                "spatialKeyXYZ": np.asarray(
                    ((0, 0, 1), (1, 0, 1), (2, 0, 1), (3, 0, 1), (4, 0, 1), (1, 0, 1)),
                    dtype=np.int32,
                ),
            },
            {
                "selected": np.asarray((1, 0, 0, 0, 0, 0), dtype=bool),
                "selectedLabel": np.asarray((7, -1, -1, -1, -1, -1), dtype=np.int32),
            },
            {
                "interfaceIndex": surface_interface,
                "physicalSheetLabel": surface_label,
                "physicalBoundarySide": surface_side,
            },
            {
                "positionXYZ": interface_position_array,
                "signedNormalXYZ": interface_normal_array,
                "processingKeyXYZ": interface_key_array,
            },
            processing_start_xyz=np.zeros(3),
            source_origin_xyz=np.zeros(3),
            processing_shape_sampling_xyz=(5, 1, 3),
            sampling_stride_voxels=2,
        )
        np.testing.assert_array_equal(
            np.flatnonzero(result["selected"]), (0, 1, 2, 3)
        )
        np.testing.assert_array_equal(
            result["canonicalOrientation"][:4], (1, 1, -1, 1)
        )
        np.testing.assert_array_equal(
            result["endpointSupportCount"][:4], (2, 2, 2, 1)
        )
        self.assertFalse(bool(result["selected"][4]))
        self.assertFalse(bool(result["selected"][5]))
        self.assertEqual(result["summary"]["adoptedContextualProfileCount"], 3)

    def test_thickness_prior_propagates_only_along_one_physical_face(self) -> None:
        position = np.column_stack(
            (
                np.arange(6, dtype=np.float64) * 2.0,
                np.zeros(6),
                np.zeros(6),
            )
        )
        expected = np.full(6, np.nan, dtype=np.float32)
        distance = np.full(6, np.nan, dtype=np.float32)
        candidate = np.full(6, -1, dtype=np.int32)
        expected[0] = 10.0
        distance[0] = 0.0
        candidate[0] = 4
        result = propagate_profile_thickness_prior(
            position,
            np.tile((0.0, 0.0, 1.0), (6, 1)),
            np.asarray((5, 5, 5, 5, 6, 6), dtype=np.int32),
            np.zeros(6, dtype=np.uint8),
            np.arange(5, dtype=np.int32),
            np.arange(1, 6, dtype=np.int32),
            {
                "expectedThicknessVoxels": expected,
                "profileDistanceSamplingSteps": distance,
                "profileCandidateIndex": candidate,
            },
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(
                maximum_geodesic_profile_distance_sampling_steps=3.1
            ),
        )
        np.testing.assert_array_equal(
            np.isfinite(result["expectedThicknessVoxels"]),
            (True, True, True, True, False, False),
        )
        np.testing.assert_allclose(
            result["expectedThicknessVoxels"][:4], 10.0
        )
        np.testing.assert_array_equal(result["profileCandidateIndex"][:4], 4)
        np.testing.assert_allclose(
            result["profileGeodesicDistanceSamplingSteps"][:4],
            (0.0, 1.0, 2.0, 3.0),
        )

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

    def test_reciprocal_surfel_support_allows_staggered_boundary_samples(self) -> None:
        lower = np.asarray(
            ((0.0, 0.0, 0.0), (2.0, 0.0, 0.0), (4.0, 0.0, 0.0))
        )
        upper = np.asarray(((1.0, 0.0, 10.0), (3.0, 0.0, 10.0)))
        position = np.vstack((lower, upper))
        result = pair_physical_boundary_faces(
            position,
            np.vstack(
                (
                    np.tile((0.0, 0.0, 1.0), (3, 1)),
                    np.tile((0.0, 0.0, -1.0), (2, 1)),
                )
            ),
            np.full(5, 9, dtype=np.int32),
            np.asarray((0, 0, 0, 1, 1), dtype=np.uint8),
            {9: np.asarray((10.0,))},
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(),
            local_thickness_prior=np.full(5, 10.0),
        )
        self.assertEqual(len(result["lowerNode"]), 3)
        self.assertEqual(
            result["census"]["lowerNodesReachingStage"][
                "reciprocalSurfelSupport"
            ],
            3,
        )
        self.assertLess(
            result["census"]["exactMutualOneToOneCount"], 3
        )

    def test_reciprocal_surfel_capacity_prevents_a_dense_fan(self) -> None:
        lower = np.column_stack(
            (
                np.arange(6, dtype=np.float64),
                np.zeros(6),
                np.zeros(6),
            )
        )
        position = np.vstack((lower, np.asarray(((2.5, 0.0, 10.0),))))
        result = pair_physical_boundary_faces(
            position,
            np.vstack(
                (
                    np.tile((0.0, 0.0, 1.0), (6, 1)),
                    np.asarray(((0.0, 0.0, -1.0),)),
                )
            ),
            np.full(7, 2, dtype=np.int32),
            np.asarray((0, 0, 0, 0, 0, 0, 1), dtype=np.uint8),
            {2: np.asarray((10.0,))},
            sampling_stride_voxels=2,
            settings=PhysicalMidSurfaceSettings(
                maximum_lower_correspondences_per_upper_surfel=4
            ),
            local_thickness_prior=np.full(7, 10.0),
        )
        self.assertEqual(
            result["census"]["lowerNodesReachingStage"][
                "reciprocalSurfelSupport"
            ],
            6,
        )
        self.assertEqual(len(result["lowerNode"]), 4)
        self.assertEqual(
            result["census"]["lowerNodesReachingStage"][
                "upperSurfelCapacity"
            ],
            4,
        )

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
