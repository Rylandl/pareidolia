from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.one_sided_growth import (
    OneSidedGrowthSettings,
    _continuity_metrics,
    _retained_component_membership,
    associate_surface_labels_from_boundary_components,
    grow_one_sided_interfaces,
)


class OneSidedGrowthTests(unittest.TestCase):
    def test_interface_normal_is_signed(self) -> None:
        bank = {
            "positionXYZ": np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32
            ),
            "signedNormalXYZ": np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float32
            ),
        }
        metrics = _continuity_metrics(
            np.asarray((0,)), np.asarray((1,)), bank, stride=1
        )
        self.assertAlmostEqual(float(metrics["signedNormalDegrees"][0]), 180.0)

    def test_contested_component_is_deferred_instead_of_partitioned(self) -> None:
        bank = {
            "positionXYZ": np.asarray(
                ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (2.0, 0.0, 0.0)),
                dtype=np.float32,
            ),
            "localEvidenceScore": np.ones(3, dtype=np.float32),
            "seedSurfaceLabel": np.asarray((10, -1, 20), dtype=np.int32),
            "seedConflict": np.zeros(3, dtype=np.uint8),
        }
        graph = {
            "edgeFirstInterface": np.asarray((0, 1), dtype=np.int32),
            "edgeSecondInterface": np.asarray((1, 2), dtype=np.int32),
            "edgeAffinity": np.ones(2, dtype=np.float32),
        }
        settings = OneSidedGrowthSettings()
        membership = _retained_component_membership(
            bank, graph, settings=settings
        )
        selection, stats = grow_one_sided_interfaces(
            bank, graph, settings=settings, membership=membership
        )
        np.testing.assert_array_equal(selection["selected"], (1, 0, 1))
        self.assertEqual(stats["deferredContestedInterfaceCount"], 1)

    def test_surface_association_requires_bilateral_component_support(self) -> None:
        seed_label = np.tile(np.asarray((10, 20), dtype=np.int32), 4)
        seed_side = np.repeat(np.asarray((0, 0, 1, 1), dtype=np.uint8), 2)
        component = np.repeat(np.arange(4, dtype=np.int32), 2)
        membership = {
            "eligibleInterface": np.ones(8, dtype=np.uint8),
            "interfaceContinuityComponent": component,
            "componentInterfaceCount": np.full(4, 2, dtype=np.int32),
            "componentSeedInterfaceCount": np.full(4, 2, dtype=np.int32),
            "componentSeedLabelCount": np.full(4, 2, dtype=np.uint16),
            "componentSoleSeedLabel": np.full(4, -1, dtype=np.int32),
        }
        settings = OneSidedGrowthSettings(
            minimum_association_side_component_count=2,
            minimum_association_balanced_seed_support=2,
        )
        arrays, stats = associate_surface_labels_from_boundary_components(
            {
                "seedSurfaceLabel": seed_label,
                "seedBoundarySide": seed_side,
            },
            membership,
            settings=settings,
        )
        self.assertEqual(stats["acceptedPairCount"], 1)
        np.testing.assert_array_equal(
            arrays["surfaceAssemblyLabel"], (10, 10)
        )

        one_face_membership = {
            name: values[:4] if len(values) == 8 else values[:2]
            for name, values in membership.items()
        }
        arrays, stats = associate_surface_labels_from_boundary_components(
            {
                "seedSurfaceLabel": seed_label[:4],
                "seedBoundarySide": seed_side[:4],
            },
            one_face_membership,
            settings=settings,
        )
        self.assertEqual(stats["acceptedPairCount"], 0)
        np.testing.assert_array_equal(
            arrays["surfaceAssemblyLabel"], (10, 20)
        )


if __name__ == "__main__":
    unittest.main()
