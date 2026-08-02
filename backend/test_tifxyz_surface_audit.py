import unittest

import numpy as np

from backend.cubical.tifxyz_surface_audit import (
    TifxyzSurfaceAuditSettings,
    audit_mid_surfaces_against_truth,
)


class TifxyzSurfaceAuditTests(unittest.TestCase):
    def test_certified_patch_representation_is_explicitly_supported(self) -> None:
        settings = TifxyzSurfaceAuditSettings(
            component_representation="certified-patches"
        )
        self.assertEqual(settings.component_representation, "certified-patches")

    def test_certified_boundary_face_representation_is_explicitly_supported(self) -> None:
        settings = TifxyzSurfaceAuditSettings(
            component_representation="certified-boundary-faces"
        )
        self.assertEqual(
            settings.component_representation, "certified-boundary-faces"
        )

    def test_recovers_boundary_aligned_component_and_rejects_transverse_one(self) -> None:
        x, y = np.meshgrid(np.arange(0.0, 20.0, 2.0), np.arange(0.0, 20.0, 2.0))
        truth = np.column_stack((x.ravel(), y.ravel(), np.zeros(x.size)))
        truth_normal = np.tile((0.0, 0.0, 1.0), (len(truth), 1))
        good_midpoint = truth + (0.0, 0.0, 5.0)
        good_lower = truth.copy()
        good_upper = truth + (0.0, 0.0, 10.0)
        bad_midpoint = truth + (0.0, 0.0, 24.0)
        bad_lower = bad_midpoint - (5.0, 0.0, 0.0)
        bad_upper = bad_midpoint + (5.0, 0.0, 0.0)
        midpoint = np.concatenate((good_midpoint, bad_midpoint))
        lower = np.concatenate((good_lower, bad_lower))
        upper = np.concatenate((good_upper, bad_upper))
        normal = np.concatenate(
            (
                np.tile((0.0, 0.0, 1.0), (len(truth), 1)),
                np.tile((1.0, 0.0, 0.0), (len(truth), 1)),
            )
        )
        component = np.repeat((7, 9), len(truth))

        summary, arrays = audit_mid_surfaces_against_truth(
            midpoint,
            lower,
            upper,
            normal,
            component,
            truth,
            truth_normal,
            settings=TifxyzSurfaceAuditSettings(minimum_component_nodes=4),
        )

        self.assertEqual(summary["components"][0]["componentId"], 7)
        self.assertEqual(summary["coverage"]["knownSurfaceSampleFraction"], 1.0)
        self.assertEqual(
            summary["components"][0]["boundaryHeightResidualVoxels"]["maximum"],
            0.0,
        )
        self.assertEqual(
            summary["components"][0]["normalResidualDegrees"]["maximum"], 0.0
        )
        self.assertTrue(np.all(arrays["nodeMatchesTruth"][: len(truth)]))
        self.assertFalse(np.any(arrays["nodeMatchesTruth"][len(truth) :]))


if __name__ == "__main__":
    unittest.main()
