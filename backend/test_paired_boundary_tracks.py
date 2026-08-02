import unittest

import numpy as np

from backend.cubical.paired_boundary_tracks import (
    PairedBoundaryTrackSettings,
    build_paired_boundary_tracks,
)


def _macro_geometry(count: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.zeros(count, dtype=np.int32),
        np.zeros((count, 3), dtype=np.float32),
        np.tile((0.0, 0.0, 1.0), (count, 1)).astype(np.float32),
    )


class PairedBoundaryTrackTests(unittest.TestCase):
    def test_recovers_two_independent_physical_faces(self) -> None:
        lower = np.asarray([[0, 0, 0], [2, 0, 0], [4, 0, 0]], np.float32)
        upper = lower + (0, 0, 10)
        macro_bin, macro_center, macro_normal = _macro_geometry(3)

        arrays, summary = build_paired_boundary_tracks(
            lower,
            upper,
            np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]], np.int32),
            macro_bin,
            macro_center,
            macro_normal,
            np.asarray([0, 1], np.int32),
            np.asarray([1, 2], np.int32),
            np.ones(2, np.int8),
            np.full(2, 0.9, np.float32),
            np.full(2, 0.9, np.float32),
            sampling_stride_voxels=2.0,
        )

        component = arrays["endpointComponentId"]
        self.assertTrue(np.all(component[0::2] == component[0]))
        self.assertTrue(np.all(component[1::2] == component[1]))
        self.assertNotEqual(int(component[0]), int(component[1]))
        self.assertEqual(summary["counts"]["componentCount"], 2)
        self.assertEqual(summary["counts"]["profileFaceCollisionCount"], 0)

    def test_unsigned_orientation_swaps_the_second_profile_faces(self) -> None:
        macro_bin, macro_center, macro_normal = _macro_geometry(2)
        arrays, _summary = build_paired_boundary_tracks(
            np.asarray([[0, 0, 0], [2, 0, 10]], np.float32),
            np.asarray([[0, 0, 10], [2, 0, 0]], np.float32),
            np.asarray([[0, 0, 0], [1, 0, 0]], np.int32),
            macro_bin,
            macro_center,
            macro_normal,
            np.asarray([0], np.int32),
            np.asarray([1], np.int32),
            np.asarray([-1], np.int8),
            np.asarray([0.9], np.float32),
            np.asarray([0.9], np.float32),
            sampling_stride_voxels=2.0,
        )

        component = arrays["endpointComponentId"]
        self.assertEqual(int(component[0]), int(component[3]))
        self.assertEqual(int(component[1]), int(component[2]))
        self.assertNotEqual(int(component[0]), int(component[1]))

    def test_long_gap_requires_dense_local_support_at_both_ends(self) -> None:
        key = np.asarray(
            [
                [1, 0, 0],
                [0, 0, 0],
                [1, 1, 0],
                [4, 0, 0],
                [5, 0, 0],
                [4, 1, 0],
            ],
            np.int32,
        )
        lower = key.astype(np.float32) * 2.0
        upper = lower + (0, 0, 10)
        macro_bin, macro_center, macro_normal = _macro_geometry(len(key))
        first = np.asarray([0, 0, 3, 3, 0], np.int32)
        second = np.asarray([1, 2, 4, 5, 3], np.int32)
        affinity = np.full(len(first), 0.9, np.float32)

        arrays, summary = build_paired_boundary_tracks(
            lower,
            upper,
            key,
            macro_bin,
            macro_center,
            macro_normal,
            first,
            second,
            np.ones(len(first), np.int8),
            affinity,
            affinity,
            sampling_stride_voxels=2.0,
            settings=PairedBoundaryTrackSettings(
                minimum_local_support_degree=2
            ),
        )

        self.assertEqual(summary["counts"]["supportedLongProposalCount"], 2)
        self.assertEqual(int(np.count_nonzero(arrays["edgeKind"] == 1)), 2)
        component = arrays["endpointComponentId"]
        self.assertEqual(int(component[0]), int(component[6]))
        self.assertEqual(int(component[1]), int(component[7]))

        strict_arrays, strict_summary = build_paired_boundary_tracks(
            lower,
            upper,
            key,
            macro_bin,
            macro_center,
            macro_normal,
            first,
            second,
            np.ones(len(first), np.int8),
            affinity,
            affinity,
            sampling_stride_voxels=2.0,
            settings=PairedBoundaryTrackSettings(
                minimum_local_support_degree=3
            ),
        )
        self.assertEqual(strict_summary["counts"]["supportedLongProposalCount"], 0)
        strict_component = strict_arrays["endpointComponentId"]
        self.assertNotEqual(int(strict_component[0]), int(strict_component[6]))


if __name__ == "__main__":
    unittest.main()
