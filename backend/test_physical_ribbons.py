from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.physical_ribbon_bank import (
    PhysicalRibbonBankSettings,
    build_physical_ribbon_bank,
)
from backend.cubical.physical_ribbon_bridging import (
    PhysicalRibbonBridgingSettings,
    build_bridge_path_bundles,
    select_bridge_bundles,
)
from backend.cubical.physical_ribbon_configuration import (
    PhysicalRibbonConfigurationSettings,
    build_profile_crossing_conflicts,
    optimize_physical_ribbon_configuration,
)
from backend.cubical.physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    _beam_set_packing,
    _fit_patch_models,
    _point_in_polygon,
    extract_surface_boundary_loops,
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

    def test_broad_support_does_not_define_component_identity(self) -> None:
        support = {
            **self.continuity,
            "edgeFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
            "edgeScore": np.ones(1, dtype=np.float32),
        }
        topology = {
            **self.continuity,
            "edgeFirstFrontierIndex": np.empty(0, dtype=np.int32),
            "edgeSecondFrontierIndex": np.empty(0, dtype=np.int32),
            "edgeScore": np.empty(0, dtype=np.float32),
        }
        crossings = {
            "crossingFirstFrontierIndex": np.empty(0, dtype=np.int32),
            "crossingSecondFrontierIndex": np.empty(0, dtype=np.int32),
            "crossingDistanceVoxels": np.empty(0, dtype=np.float32),
            "crossingFirstParameter": np.empty(0, dtype=np.float32),
            "crossingSecondParameter": np.empty(0, dtype=np.float32),
        }
        solved, stats = optimize_physical_ribbon_configuration(
            self.ribbon,
            self.interfaces,
            support,
            crossings,
            settings=PhysicalRibbonConfigurationSettings(
                maximum_optimization_sweeps=1,
                maximum_hole_growth_sweeps=1,
            ),
            topology_continuity=topology,
        )
        self.assertEqual(stats["selectedSupportEdgeCount"], 1)
        self.assertEqual(stats["selectedContinuityEdgeCount"], 0)
        self.assertEqual(stats["componentCount"], 2)
        np.testing.assert_array_equal(solved["edgeSelected"], (0,))


class PhysicalRibbonBridgingTests(unittest.TestCase):
    @staticmethod
    def _ribbons(count: int) -> dict[str, np.ndarray]:
        return {
            "sourceInterface": np.arange(0, 2 * count, 2, dtype=np.int32),
            "targetInterface": np.arange(1, 2 * count, 2, dtype=np.int32),
            "physicalEvidenceScore": np.full(count, 0.9, dtype=np.float32),
            "interfaceCandidateDegree": np.ones(2 * count, dtype=np.int32),
        }

    @staticmethod
    def _empty_crossings() -> dict[str, np.ndarray]:
        return {
            "crossingFirstFrontierIndex": np.empty(0, dtype=np.int32),
            "crossingSecondFrontierIndex": np.empty(0, dtype=np.int32),
        }

    def test_adaptive_edge_cannot_silently_join_existing_components(self) -> None:
        ribbon = self._ribbons(2)
        continuity = {
            "frontierRibbonCandidate": np.arange(2, dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
            "edgeScore": np.ones(1, dtype=np.float32),
            "edgeBaseline": np.zeros(1, dtype=np.uint8),
        }
        configuration = {
            "selected": np.ones(2, dtype=np.uint8),
            "component": np.asarray((0, 1), dtype=np.int32),
            **self._empty_crossings(),
        }
        bundles = {
            "bundleComponentFirst": np.empty(0, dtype=np.int32),
            "bundleComponentSecond": np.empty(0, dtype=np.int32),
            "bundleCandidateOffset": np.zeros(1, dtype=np.int64),
            "bundleCandidateFrontierIndex": np.empty(0, dtype=np.int32),
            "bundleScore": np.empty(0, dtype=np.float32),
            "bundleTangentRatio": np.empty(0, dtype=np.float32),
            "bundleFirstAnchorCount": np.empty(0, dtype=np.int32),
            "bundleSecondAnchorCount": np.empty(0, dtype=np.int32),
            "bundleKind": np.empty(0, dtype=np.uint8),
            "bundlePathCount": np.empty(0, dtype=np.uint8),
            "bundleSharedCandidateCount": np.empty(0, dtype=np.uint8),
        }
        solved, stats = select_bridge_bundles(
            ribbon,
            continuity,
            configuration,
            bundles,
            settings=PhysicalRibbonBridgingSettings(
                minimum_bundle_candidate_count=2,
                minimum_anchor_count_per_component=1,
            ),
        )
        self.assertEqual(stats["componentCount"], 2)
        self.assertEqual(stats["selectedAdaptiveContinuationEdgeCount"], 0)
        np.testing.assert_array_equal(solved["edgeSelected"], (0,))

    def test_guarded_bundle_activates_adaptive_edges_through_new_nodes(self) -> None:
        ribbon = self._ribbons(4)
        continuity = {
            "frontierRibbonCandidate": np.arange(4, dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray(
                (0, 2, 0, 3), dtype=np.int32
            ),
            "edgeSecondFrontierIndex": np.asarray(
                (2, 1, 3, 1), dtype=np.int32
            ),
            "edgeScore": np.ones(4, dtype=np.float32),
            "edgeBaseline": np.zeros(4, dtype=np.uint8),
        }
        configuration = {
            "selected": np.asarray((1, 1, 0, 0), dtype=np.uint8),
            "component": np.asarray((0, 1, -1, -1), dtype=np.int32),
            **self._empty_crossings(),
        }
        bundles = {
            "bundleComponentFirst": np.asarray((0,), dtype=np.int32),
            "bundleComponentSecond": np.asarray((1,), dtype=np.int32),
            "bundleCandidateOffset": np.asarray((0, 2), dtype=np.int64),
            "bundleCandidateFrontierIndex": np.asarray((2, 3), dtype=np.int32),
            "bundleScore": np.ones(1, dtype=np.float32),
            "bundleTangentRatio": np.ones(1, dtype=np.float32),
            "bundleFirstAnchorCount": np.asarray((1,), dtype=np.int32),
            "bundleSecondAnchorCount": np.asarray((1,), dtype=np.int32),
            "bundleKind": np.zeros(1, dtype=np.uint8),
            "bundlePathCount": np.ones(1, dtype=np.uint8),
            "bundleSharedCandidateCount": np.zeros(1, dtype=np.uint8),
        }
        solved, stats = select_bridge_bundles(
            ribbon,
            continuity,
            configuration,
            bundles,
            settings=PhysicalRibbonBridgingSettings(
                minimum_bundle_candidate_count=2,
                minimum_anchor_count_per_component=1,
            ),
        )
        self.assertEqual(stats["selectedBundleCount"], 1)
        self.assertEqual(stats["addedBridgeRibbonCount"], 2)
        self.assertEqual(stats["componentCount"], 1)
        self.assertEqual(stats["selectedAdaptiveContinuationEdgeCount"], 4)
        np.testing.assert_array_equal(solved["selected"], (1, 1, 1, 1))

    def test_two_lanes_may_share_one_interior_fold_apex(self) -> None:
        ribbon = self._ribbons(9)
        first = np.asarray((0, 4, 6, 7, 1, 5, 6, 8), dtype=np.int32)
        second = np.asarray((4, 6, 7, 2, 5, 6, 8, 3), dtype=np.int32)
        continuity = {
            "frontierRibbonCandidate": np.arange(9, dtype=np.int32),
            "edgeFirstFrontierIndex": first,
            "edgeSecondFrontierIndex": second,
            "edgeScore": np.full(len(first), 0.9, dtype=np.float32),
        }
        configuration = {
            "selected": np.asarray((1, 1, 1, 1, 0, 0, 0, 0, 0)),
            "component": np.asarray((0, 0, 1, 1, -1, -1, -1, -1, -1)),
            **self._empty_crossings(),
        }
        bundles, stats = build_bridge_path_bundles(
            ribbon,
            continuity,
            configuration,
            settings=PhysicalRibbonBridgingSettings(),
        )
        self.assertEqual(stats["qualifiedSharedApexBundleCount"], 1)
        np.testing.assert_array_equal(bundles["bundleKind"], (2,))
        np.testing.assert_array_equal(
            bundles["bundleSharedCandidateCount"], (1,)
        )
        self.assertEqual(len(bundles["bundleCandidateFrontierIndex"]), 5)


class PhysicalRibbonPatchHoleTests(unittest.TestCase):
    def test_polygon_test_preserves_signed_edge_denominator(self) -> None:
        counterclockwise = np.asarray(
            ((0.0, 0.0), (3.0, 0.0), (3.0, 2.0), (0.0, 2.0))
        )
        self.assertTrue(_point_in_polygon(np.asarray((1.0, 1.0)), counterclockwise))
        self.assertTrue(
            _point_in_polygon(np.asarray((1.0, 1.0)), counterclockwise[::-1])
        )
        self.assertFalse(_point_in_polygon(np.asarray((4.0, 1.0)), counterclockwise))

    def test_closed_triangle_annulus_produces_one_interior_loop(self) -> None:
        chart = np.asarray(
            (
                (-2.0, -2.0),
                (2.0, -2.0),
                (2.0, 2.0),
                (-2.0, 2.0),
                (-0.75, -0.75),
                (0.75, -0.75),
                (0.75, 0.75),
                (-0.75, 0.75),
            ),
            dtype=np.float32,
        )
        triangles = np.asarray(
            (
                (0, 1, 5),
                (0, 5, 4),
                (1, 2, 6),
                (1, 6, 5),
                (2, 3, 7),
                (2, 7, 6),
                (3, 0, 4),
                (3, 4, 7),
            ),
            dtype=np.int32,
        )
        loops, stats = extract_surface_boundary_loops(
            {
                "triangleFrontierIndex": triangles,
                "chartUV": chart,
                "component": np.zeros(8, dtype=np.int32),
                "thicknessVoxels": np.ones(8, dtype=np.float32),
            },
            settings=PhysicalRibbonPatchHoleSettings(
                minimum_hole_boundary_vertex_count=4,
                minimum_hole_diameter_boundary_edges=1.3,
                minimum_hole_area_boundary_edges_squared=0.5,
            ),
        )
        self.assertEqual(stats["outerBoundaryLoopCount"], 1)
        self.assertEqual(stats["interiorHoleLoopCount"], 1)
        self.assertEqual(stats["macroEligibleHoleCount"], 1)
        self.assertEqual(int(np.sum(loops["loopMacroEligible"])), 1)

    def test_quadratic_patch_is_fit_from_the_whole_context(self) -> None:
        uv = np.asarray(
            [
                (u_value, v_value)
                for u_value in (-1.0, 0.0, 1.0)
                for v_value in (-1.0, 0.0, 1.0)
            ],
            dtype=np.float32,
        )
        xyz = np.column_stack((uv, 0.25 * np.sum(uv * uv, axis=1))).astype(
            np.float32
        )
        normal = np.tile((0.0, 0.0, 1.0), (len(uv), 1)).astype(np.float32)
        boundary = np.asarray((0, 1, 2, 3, 5, 6, 7, 8), dtype=np.int32)
        fitted = _fit_patch_models(
            uv,
            xyz,
            normal,
            boundary,
            np.arange(len(uv), dtype=np.int32),
            quadratic_ridge=1.0e-5,
        )
        residual = np.asarray(fitted["contextResidualVoxels"])
        self.assertLess(float(residual[1]), float(residual[0]) * 0.05)

    def test_joint_beam_can_replace_two_incumbents_as_one_move(self) -> None:
        # Options 0/1 are the incumbent matching.  Each candidate alone is
        # worse, but the strict continuation factor between 2/3 makes their
        # simultaneous alternating swap the best state.
        states, stats = _beam_set_packing(
            np.asarray((0.8, 0.8, 0.4, 0.4), dtype=np.float32),
            [1 << 2, 1 << 3, 1 << 0, 1 << 1],
            np.asarray((2,), dtype=np.int32),
            np.asarray((3,), dtype=np.int32),
            np.asarray((2.0,), dtype=np.float32),
            (1 << 0) | (1 << 1),
            beam_width=256,
        )
        self.assertEqual(states[0][1], (1 << 2) | (1 << 3))
        self.assertTrue(stats["baselineStatePreserved"])


if __name__ == "__main__":
    unittest.main()
