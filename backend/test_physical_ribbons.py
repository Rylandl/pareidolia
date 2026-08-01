from __future__ import annotations

import unittest
from unittest.mock import patch

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
from backend.cubical.physical_ribbon_continuity import (
    PhysicalRibbonContinuitySettings,
    build_paired_boundary_continuity,
)
from backend.cubical.physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    _beam_set_packing,
    _fit_patch_models,
    _point_in_polygon,
    extract_surface_boundary_loops,
)
from backend.cubical.physical_ribbon_patch_corridors import (
    PhysicalRibbonPatchCorridorSettings,
    _best_monotone_corridor_chain,
    _corridor_model_grid,
    _evaluate_corridor_connections,
    _shifted_trace_correlation,
    _triangle_region_labels,
)
from backend.cubical.physical_ribbon_corridor_variants import (
    compile_exact_variant_reconfiguration,
)
from backend.cubical.physical_ribbon_corridor_sets import (
    _choose_global_variant_states,
)
from backend.cubical.physical_ribbon_corridor_extension import (
    _delta_variant_arrays,
    _variant_signature,
)
from backend.cubical.physical_ribbon_corridor_dormant import (
    _combine_compiled_reconfigurations,
    _condition_crossings_on_immutable_baseline,
    _union_crossing_continuity,
    _variant_dormant_addition_count,
)
from backend.cubical.physical_ribbon_corridor_one_sided import (
    _variant_one_sided_addition_count,
)
from backend.cubical.physical_ribbon_corridor_frontier import (
    _map_frontier_by_bank,
)
from backend.cubical.physical_ribbon_corridor_saturation import (
    _assess_saturation,
)
from backend.cubical.physical_ribbon_corridor_deficits import (
    _largest_four_connected_fraction,
    _nearest_distance_and_index,
)
from backend.cubical.physical_ribbon_corridor_faces import (
    PhysicalRibbonCorridorFaceSettings,
    _attached_candidate_closure,
    _face_is_physical,
    _minimum_face_path,
    _retained_chart_nodes,
)
from backend.cubical.physical_ribbon_corridor_face_replay import (
    _edge_manifold_audit,
    _optimize_candidate_state,
)
from backend.cubical.physical_ribbon_complete_strips import (
    _residual_corridor_rows,
    _selection_key as _complete_strip_selection_key,
    _split_audit,
    _strict_surface,
)
from backend.cubical.physical_ribbon_complete_strip_replay import (
    _grouped_candidate_states,
)
from backend.cubical.physical_ribbon_lineage_strips import (
    _affected_lineages_preserved,
    _lineage_preserved,
    _lineage_target_rows,
    _selected_component_sizes,
)
from backend.cubical.physical_ribbon_lineage_strip_replay import (
    _complete_inheritance_audit,
    _cumulative_supplemental_face_arrays,
    _prior_face_rows,
)
from backend.cubical.physical_ribbon_replay_configuration import (
    _materialize_replay_arrays,
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
    def test_materialize_replay_configuration_preserves_exact_state(self) -> None:
        replay = {
            "corridorReplaySelected": np.asarray((1, 1, 0), dtype=np.uint8),
            "corridorReplayComponent": np.asarray((0, 0, -1), dtype=np.int32),
        }
        topology = {
            "frontierRibbonCandidate": np.asarray((0, 1, 2), dtype=np.int32),
            "frontierMidpointKeyXYZ": np.zeros((3, 3), dtype=np.int32),
            "continuitySupportDegree": np.asarray((1, 1, 0), dtype=np.int32),
            "continuitySupportScore": np.asarray((1.0, 1.0, 0.0), dtype=np.float32),
            "tangentRankRatio": np.asarray((1.0, 1.0, 0.0), dtype=np.float32),
            "selectionObjective": np.asarray((1.0, 1.0, 0.0), dtype=np.float32),
            "selected": np.asarray((0, 0, 1), dtype=np.uint8),
            "component": np.asarray((-1, -1, 0), dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray((0, 1), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((1, 2), dtype=np.int32),
            "edgeScore": np.asarray((0.8, 0.7), dtype=np.float32),
            "edgeSelected": np.asarray((0, 0), dtype=np.uint8),
            "edgeNormalDegrees": np.zeros(2, dtype=np.float32),
            "edgeMidpointHeightResidualVoxels": np.zeros(2, dtype=np.float32),
            "edgeBoundaryHeightResidualVoxels": np.zeros(2, dtype=np.float32),
            "edgeThicknessChangeVoxels": np.zeros(2, dtype=np.float32),
            "edgeBoundaryShiftDifferenceVoxels": np.zeros(2, dtype=np.float32),
            "crossingFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "crossingSecondFrontierIndex": np.asarray((2,), dtype=np.int32),
            "crossingDistanceVoxels": np.asarray((0.5,), dtype=np.float32),
            "crossingFirstParameter": np.asarray((0.5,), dtype=np.float32),
            "crossingSecondParameter": np.asarray((0.5,), dtype=np.float32),
            "nodeUnaryScore": np.asarray((0.8, 0.7, 0.6), dtype=np.float32),
        }
        ribbon = {
            "sourceInterface": np.asarray((0, 2, 4), dtype=np.int32),
            "targetInterface": np.asarray((1, 3, 5), dtype=np.int32),
        }
        materialized, configuration, stats = _materialize_replay_arrays(
            replay, topology, ribbon
        )
        np.testing.assert_array_equal(materialized["selected"], (1, 1, 0))
        np.testing.assert_array_equal(materialized["edgeSelected"], (1, 0))
        np.testing.assert_array_equal(configuration["selected"], (1, 1, 0))
        self.assertEqual(stats["selectedRibbonCount"], 2)
        self.assertEqual(stats["componentCount"], 1)
        self.assertEqual(stats["selectedCrossingConflictCount"], 0)

    def test_explicit_continuity_frontier_can_include_one_sided_ribbon(self) -> None:
        ribbon = {
            "sourceInterface": np.asarray((0,), dtype=np.int32),
            "targetInterface": np.asarray((1,), dtype=np.int32),
            "sourceRayRank": np.asarray((0,), dtype=np.int16),
            "targetRayRank": np.asarray((-1,), dtype=np.int16),
            "physicalEvidenceScore": np.asarray((0.9,), dtype=np.float32),
            "mutualFirstHit": np.asarray((0,), dtype=np.uint8),
            "bidirectional": np.asarray((0,), dtype=np.uint8),
            "midpointXYZ": np.asarray(((5.0, 5.0, 5.0),), dtype=np.float32),
            "normalXYZ": np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            "thicknessVoxels": np.asarray((10.0,), dtype=np.float32),
        }
        interfaces = {
            "positionXYZ": np.asarray(
                ((0.0, 5.0, 5.0), (10.0, 5.0, 5.0)), dtype=np.float32
            ),
            "signedNormalXYZ": np.asarray(
                ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32
            ),
        }
        topology, stats = build_paired_boundary_continuity(
            ribbon,
            interfaces,
            processing_world_start_xyz=np.zeros(3, dtype=np.float32),
            sampling_stride_voxels=1,
            settings=PhysicalRibbonContinuitySettings(
                minimum_support_degree=1,
                minimum_selected_degree=0,
                peeling_sweeps=0,
            ),
            frontier_bank_index=np.asarray((0,), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            topology["frontierRibbonCandidate"], (0,)
        )
        self.assertEqual(stats["unidirectionalFrontierCount"], 1)
        self.assertLess(float(topology["selectionObjective"][0]), 0.9)

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


class PhysicalRibbonPatchCorridorTests(unittest.TestCase):
    def test_lineage_strip_targets_rows_with_split_beam_states(self) -> None:
        manifest = {
            "screen": {
                "rows": [
                    {
                        "corridorRow": 2,
                        "variantCount": 16,
                        "componentPreservingVariantCount": 0,
                        "eligibleVariantCount": 0,
                    },
                    {
                        "corridorRow": 4,
                        "variantCount": 16,
                        "componentPreservingVariantCount": 3,
                        "eligibleVariantCount": 0,
                    },
                    {
                        "corridorRow": 6,
                        "variantCount": 16,
                        "componentPreservingVariantCount": 16,
                        "eligibleVariantCount": 0,
                    },
                    {
                        "corridorRow": 8,
                        "variantCount": 16,
                        "componentPreservingVariantCount": 8,
                        "eligibleVariantCount": 2,
                    },
                ]
            }
        }
        np.testing.assert_array_equal(
            _lineage_target_rows(manifest, already_replayed_rows=(4,)), (2,)
        )

    def test_lineage_constraint_uses_entire_induced_sheet_graph(self) -> None:
        edge_offset = np.asarray((0, 1, 3, 5, 7, 8), dtype=np.int64)
        edge_neighbor = np.asarray((1, 0, 2, 1, 3, 2, 4, 3), dtype=np.int32)
        count, sizes = _selected_component_sizes(
            {0, 1, 3, 4}, edge_offset, edge_neighbor
        )
        self.assertEqual(count, 2)
        self.assertEqual(sizes, (2, 2))
        preserved, sizes = _lineage_preserved(
            {0, 1, 2, 3, 4},
            added=(),
            removed=(2,),
            edge_offset=edge_offset,
            edge_neighbor=edge_neighbor,
        )
        self.assertFalse(preserved)
        self.assertEqual(sizes, (2, 2))
        preserved, sizes = _lineage_preserved(
            {0, 1, 2, 3, 4},
            added=(),
            removed=(),
            edge_offset=edge_offset,
            edge_neighbor=edge_neighbor,
        )
        self.assertTrue(preserved)
        self.assertEqual(sizes, (5,))

    def test_lineage_constraint_rejects_deleted_neighbor_component(self) -> None:
        edge_offset = np.asarray((0, 1, 2), dtype=np.int64)
        edge_neighbor = np.asarray((1, 0), dtype=np.int32)
        preserved, audit = _affected_lineages_preserved(
            {4: {0}, 7: {1}},
            np.asarray((4, 7), dtype=np.int32),
            target_component=4,
            added=(),
            removed=(1,),
            edge_offset=edge_offset,
            edge_neighbor=edge_neighbor,
        )
        self.assertFalse(preserved)
        self.assertEqual(audit["deletedComponents"], (7,))

    def test_lineage_replay_recovers_all_cumulative_face_rows(self) -> None:
        manifest = {
            "surface": {
                "pathRecords": [
                    {"corridorRow": 8},
                    {"corridorRow": 3},
                    {"corridorRow": 8},
                ]
            }
        }
        self.assertEqual(_prior_face_rows(manifest), (3, 8))

    def test_lineage_replay_accepts_zero_face_strict_connection(self) -> None:
        empty_supplemental = {
            "supplementalTriangleFrontierIndex": np.empty((0, 3), dtype=np.int32)
        }
        connections = {
            "boundaryArcsConnected": np.asarray((0, 1, 0), dtype=np.uint8),
            "boundaryArcSharedRegionFraction": np.asarray(
                (0.0, 0.75, 0.0), dtype=np.float32
            ),
        }
        face_record = {"corridorRow": 2, "eligible": True}
        with patch(
            "backend.cubical.physical_ribbon_lineage_strip_replay._evaluate_corridor_connections",
            return_value=connections,
        ), patch(
            "backend.cubical.physical_ribbon_lineage_strip_replay._supplemental_face_arrays",
            return_value=(empty_supplemental, [face_record]),
        ) as supplemental:
            arrays, records, strict_rows = _cumulative_supplemental_face_arrays(
                (1, 2),
                {},
                {},
                surface_settings=object(),
                face_settings=PhysicalRibbonCorridorFaceSettings(),
            )
        self.assertIs(arrays, empty_supplemental)
        self.assertEqual(strict_rows, (1,))
        self.assertTrue(records[0]["eligible"])
        self.assertEqual(records[0]["physicalPathFaceCount"], 0)
        self.assertIs(records[1], face_record)
        self.assertEqual(supplemental.call_args.args[0], (2,))

    def test_lineage_replay_rejects_deleted_prior_component(self) -> None:
        audit = _complete_inheritance_audit(
            np.asarray((1, 1, 1), dtype=bool),
            np.asarray((0, 0, 1), dtype=np.int32),
            np.asarray((1, 1, 0), dtype=bool),
            np.asarray((0, 0, -1), dtype=np.int32),
            minimum_substantial_ribbon_count=2,
        )
        self.assertEqual(audit["deletedPriorComponentCount"], 1)
        self.assertEqual(audit["deletedPriorComponents"], [1])
        self.assertEqual(audit["orphanFinalComponentCount"], 0)

    def test_lineage_replay_only_absorbs_provisional_components(self) -> None:
        allowed = _complete_inheritance_audit(
            np.asarray((1, 1, 1), dtype=bool),
            np.asarray((0, 0, 1), dtype=np.int32),
            np.asarray((1, 1, 1), dtype=bool),
            np.asarray((0, 0, 0), dtype=np.int32),
            minimum_substantial_ribbon_count=2,
        )
        self.assertEqual(allowed["crossPriorComponentFusionCount"], 1)
        self.assertEqual(allowed["allowedProvisionalAbsorptionCount"], 1)
        self.assertEqual(allowed["forbiddenSubstantialFusionCount"], 0)
        forbidden = _complete_inheritance_audit(
            np.asarray((1, 1, 1, 1), dtype=bool),
            np.asarray((0, 0, 1, 1), dtype=np.int32),
            np.asarray((1, 1, 1, 1), dtype=bool),
            np.asarray((0, 0, 0, 0), dtype=np.int32),
            minimum_substantial_ribbon_count=2,
        )
        self.assertEqual(forbidden["forbiddenSubstantialFusionCount"], 1)
        self.assertEqual(forbidden["allowedProvisionalAbsorptionCount"], 0)

    def test_complete_strip_residuals_ignore_candidate_provenance(self) -> None:
        prior = {
            "corridorEvidenceEligible": np.asarray((1, 1, 1, 0), dtype=np.uint8),
            "corridorReplayProposalSuccessful": np.asarray(
                (0, 1, 0, 0), dtype=np.uint8
            ),
        }
        manifest = {"optimization": {"chosenCorridorRows": [2]}}
        np.testing.assert_array_equal(
            _residual_corridor_rows(manifest, prior), (0,)
        )

    def test_complete_strip_strict_surface_excludes_ct_closure_faces(self) -> None:
        replay = {
            "baseStrictTriangleCount": np.asarray((1,), dtype=np.int64),
            "triangleFrontierIndex": np.asarray(
                ((0, 1, 2), (1, 2, 3)), dtype=np.int32
            ),
            "triangleAreaVoxelsSquared": np.asarray((2.0, 3.0)),
            "triangleNormalResidualDegrees": np.asarray((4.0, 5.0)),
            "selected": np.asarray((1, 1, 1, 1), dtype=np.uint8),
        }
        strict = _strict_surface(replay)
        self.assertEqual(len(strict["triangleFrontierIndex"]), 1)
        self.assertEqual(len(strict["triangleAreaVoxelsSquared"]), 1)
        self.assertEqual(len(strict["selected"]), 4)

    def test_complete_strip_split_audit_measures_detached_lineage(self) -> None:
        audit = _split_audit(
            np.asarray((1, 1, 1, 1, 0), dtype=bool),
            np.asarray((0, 0, 1, 1, -1), dtype=np.int32),
            np.asarray((7, 7, 7, 7, 7), dtype=np.int32),
            7,
        )
        self.assertTrue(audit["split"])
        self.assertEqual(audit["descendantRibbonCounts"], [2, 2])
        self.assertEqual(audit["detachedRibbonCount"], 2)
        self.assertAlmostEqual(audit["largestDescendantFraction"], 0.5)

    def test_complete_strip_selection_prefers_strict_then_face_debt(self) -> None:
        common = {
            "eligible": True,
            "triangleRegionCountBefore": 2,
            "triangleRegionCountAfter": 1,
            "strictAreaRetention": 1.0,
            "localObjectiveDelta": 1.0,
            "patchCoverage": 0.8,
            "variantRank": 0,
        }
        strict = {
            **common,
            "strictConnected": True,
            "physicalPathCost": 0.0,
            "physicalPathFaceCount": 0,
        }
        faced = {
            **common,
            "strictConnected": False,
            "physicalPathCost": 1.0,
            "physicalPathFaceCount": 1,
        }
        self.assertGreater(
            _complete_strip_selection_key(strict),
            _complete_strip_selection_key(faced),
        )

    def test_complete_strip_replay_chooses_one_variant_per_corridor(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray(
                (0, 1, 2, 3), dtype=np.int64
            ),
            "corridorVariantAddedFrontierIndex": np.asarray(
                (0, 1, 2), dtype=np.int32
            ),
            "corridorVariantRemovedOffset": np.asarray(
                (0, 0, 0, 0), dtype=np.int64
            ),
            "corridorVariantRemovedFrontierIndex": np.empty(0, dtype=np.int32),
        }
        candidates = [
            {
                "corridorRow": row,
                "variantIndex": index,
                "physicalPathCost": cost,
                "physicalPathFaceCount": 1,
                "triangleRegionCountBefore": 2,
                "triangleRegionCountAfter": 1,
                "strictAreaRetention": 1.0,
                "localObjectiveDelta": 1.0,
                "patchCoverage": 0.8,
            }
            for row, index, cost in ((5, 0, 2.0), (5, 1, 1.0), (7, 2, 1.0))
        ]
        states, stats = _grouped_candidate_states(
            candidates,
            variants,
            valid_modifications=lambda added, _removed: not ({1, 2} <= added),
            beam_width=32,
        )
        self.assertEqual(states[0]["rows"], (5, 7))
        self.assertEqual(states[0]["variantIndices"], (0, 2))
        self.assertEqual(stats["corridorCount"], 2)

    def test_corridor_deficit_support_metrics_are_spatial(self) -> None:
        distance, index = _nearest_distance_and_index(
            np.asarray(((0.0, 0.0, 0.0), (4.0, 0.0, 0.0))),
            np.asarray(((1.0, 0.0, 0.0), (5.0, 0.0, 0.0))),
        )
        np.testing.assert_allclose(distance, (1.0, 1.0))
        np.testing.assert_array_equal(index, (0, 1))
        self.assertAlmostEqual(
            _largest_four_connected_fraction(
                np.asarray(
                    (
                        (1, 1, 0, 0),
                        (1, 0, 0, 1),
                        (0, 0, 1, 1),
                    ),
                    dtype=bool,
                )
            ),
            3.0 / 12.0,
        )

    def test_corridor_face_path_bridges_triangle_regions_only_by_faces(self) -> None:
        existing = np.asarray(((0, 1, 2), (3, 4, 5)), dtype=np.int32)
        region = np.asarray((0, 1), dtype=np.int32)
        candidate = np.asarray(((1, 2, 3), (2, 3, 4)), dtype=np.int32)
        path, cost = _minimum_face_path(
            existing,
            region,
            candidate,
            np.asarray((1.25, 1.5), dtype=np.float32),
            first_region=0,
            second_region=1,
        )
        np.testing.assert_array_equal(path, (0, 1))
        self.assertAlmostEqual(cost, 2.75)
        missing, missing_cost = _minimum_face_path(
            existing,
            region,
            candidate,
            np.asarray((1.25, 1.5), dtype=np.float32),
            first_region=0,
            second_region=1,
            candidate_eligible=np.asarray((1, 0), dtype=bool),
        )
        self.assertEqual(len(missing), 0)
        self.assertTrue(np.isinf(missing_cost))

    def test_corridor_face_closure_excludes_isolated_candidates(self) -> None:
        existing = np.asarray(((0, 1, 2),), dtype=np.int32)
        candidates = np.asarray(
            ((1, 2, 3), (2, 3, 4), (6, 7, 8)), dtype=np.int32
        )
        np.testing.assert_array_equal(
            _attached_candidate_closure(
                existing,
                candidates,
                np.asarray((1, 1, 1), dtype=bool),
            ),
            (1, 1, 0),
        )

    def test_corridor_face_gate_uses_local_physical_units(self) -> None:
        settings = PhysicalRibbonCorridorFaceSettings()
        metrics = {
            "areaVoxelsSquared": 1.0,
            "centerDistanceThicknesses": 0.10,
            "centerHeightThicknesses": 0.05,
            "centerTangentRasterSteps": 1.0,
            "ctNormalResidualDegrees": 30.0,
            "maximumEdgeThicknesses": 1.0,
        }
        self.assertTrue(_face_is_physical(metrics, settings=settings))
        self.assertFalse(
            _face_is_physical(
                {**metrics, "maximumEdgeThicknesses": 1.5},
                settings=settings,
            )
        )

    def test_corridor_face_catalog_can_target_one_strict_component(self) -> None:
        surface = {
            "component": np.asarray((0, 0, 0, 1, 1, 1), dtype=np.int32),
            "componentSize": np.full(6, 3, dtype=np.int32),
            "chartUV": np.asarray(
                (
                    (0.0, 0.0),
                    (1.0, 0.0),
                    (0.0, 1.0),
                    (3.0, 0.0),
                    (4.0, 0.0),
                    (3.0, 1.0),
                ),
                dtype=np.float32,
            ),
        }
        retained = _retained_chart_nodes(
            surface,
            minimum_component_ribbon_count=3,
            minimum_chart_separation_voxels=0.1,
            component_ids={1},
        )
        self.assertEqual(len(retained), 1)
        np.testing.assert_array_equal(np.sort(retained[0]), (3, 4, 5))

    def test_corridor_face_replay_prefers_lower_face_debt_on_conflict(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray((0, 1, 2, 3), dtype=np.int64),
            "corridorVariantAddedFrontierIndex": np.asarray((0, 1, 2), dtype=np.int32),
            "corridorVariantRemovedOffset": np.asarray((0, 0, 0, 0), dtype=np.int64),
            "corridorVariantRemovedFrontierIndex": np.empty(0, dtype=np.int32),
        }
        candidates = [
            {
                "corridorRow": row,
                "bestFailedVariantIndex": row,
                "physicalPathCost": cost,
                "physicalPathFaceCount": 1,
                "sharedArcRegionFraction": 1.0,
            }
            for row, cost in ((0, 1.0), (1, 2.0), (2, 1.0))
        ]
        chosen, stats = _optimize_candidate_state(
            candidates,
            variants,
            valid_modifications=lambda added, _removed: not ({0, 1} <= added),
            beam_width=16,
        )
        self.assertEqual(chosen["rows"], (0, 2))
        self.assertEqual(stats["chosenCount"], 2)

    def test_supplemental_faces_preserve_edge_manifoldness(self) -> None:
        clean = _edge_manifold_audit(
            np.asarray(((0, 1, 2), (1, 0, 3)), dtype=np.int32)
        )
        self.assertEqual(clean["nonManifoldEdgeCount"], 0)
        self.assertEqual(clean["maximumTriangleIncidencePerEdge"], 2)
        broken = _edge_manifold_audit(
            np.asarray(
                ((0, 1, 2), (1, 0, 3), (0, 1, 4)), dtype=np.int32
            )
        )
        self.assertEqual(broken["nonManifoldEdgeCount"], 1)

    def test_corridor_saturation_reuses_untouched_exact_failures(self) -> None:
        prior_corridor = {
            "corridorEvidenceEligible": np.asarray((1, 1), dtype=np.uint8),
            "corridorPatchOffset": np.asarray((0, 1, 2), dtype=np.int64),
            "corridorPatchXYZ": np.asarray(
                ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0)), dtype=np.float32
            ),
            "corridorPatchNormalXYZ": np.asarray(
                ((1.0, 0.0, 0.0), (0.0, 1.0, 0.0)), dtype=np.float32
            ),
            "corridorPatchThicknessVoxels": np.asarray(
                (8.0, 9.0), dtype=np.float32
            ),
            "corridorTopologyComponent": np.asarray((0, 1), dtype=np.int32),
        }
        current_corridor = {
            name: np.asarray(value[:1]).copy()
            for name, value in prior_corridor.items()
            if name != "corridorPatchOffset"
        }
        current_corridor["corridorPatchOffset"] = np.asarray(
            (0, 1), dtype=np.int64
        )
        prior_frontier = {
            "frontierRibbonCandidate": np.asarray((7, 8), dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
            "selected": np.asarray((1, 0), dtype=np.uint8),
            "component": np.asarray((0, -1), dtype=np.int32),
            "targetCorridorRow": np.asarray((0, 1), dtype=np.int32),
            "targetCorridorCandidateOffset": np.asarray(
                (0, 1, 2), dtype=np.int64
            ),
            "targetCorridorCandidateBankIndex": np.asarray(
                (7, 8), dtype=np.int32
            ),
        }
        current_frontier = {
            **prior_frontier,
            "targetCorridorRow": np.asarray((0,), dtype=np.int32),
            "targetCorridorCandidateOffset": np.asarray((0, 1), dtype=np.int64),
            "targetCorridorCandidateBankIndex": np.asarray((7,), dtype=np.int32),
        }
        prior_replay = {
            "corridorReplaySelected": np.asarray((1, 1), dtype=np.uint8),
            "corridorReplayComponent": np.asarray((0, 1), dtype=np.int32),
            "corridorChosenExactVariant": np.asarray((-1, 0), dtype=np.int32),
        }
        result = _assess_saturation(
            prior_corridor,
            prior_frontier,
            prior_replay,
            current_corridor,
            current_frontier,
        )
        self.assertTrue(result["candidateClassSaturated"])
        self.assertEqual(result["sharedPriorExactFailureCount"], 1)
        self.assertEqual(result["remainingCorridorsInChangedComponents"], [])

    def test_frontier_map_uses_ribbon_bank_identity(self) -> None:
        np.testing.assert_array_equal(
            _map_frontier_by_bank(
                np.asarray((7, 2, 9), dtype=np.int32),
                np.asarray((2, 4, 7, 9), dtype=np.int32),
                ribbon_bank_count=10,
            ),
            (2, 0, 3),
        )

    def test_one_sided_variant_count_uses_bank_identity(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray(
                (0, 2, 3), dtype=np.int64
            ),
            "corridorVariantAddedFrontierIndex": np.asarray(
                (0, 2, 1), dtype=np.int32
            ),
        }
        frontier = np.asarray((7, 4, 9), dtype=np.int32)
        one_sided = np.zeros(10, dtype=bool)
        one_sided[[7, 9]] = True
        np.testing.assert_array_equal(
            _variant_one_sided_addition_count(
                variants, frontier, one_sided
            ),
            (2, 0),
        )

    def test_crossing_screen_unions_strict_and_support_edges(self) -> None:
        expanded = {
            "frontierRibbonCandidate": np.asarray((1, 2, 3), dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
        }
        support = {
            "frontierRibbonCandidate": np.asarray((3, 1, 2), dtype=np.int32),
            "edgeFirstFrontierIndex": np.asarray((0, 1), dtype=np.int32),
            "edgeSecondFrontierIndex": np.asarray((2, 2), dtype=np.int32),
        }
        union, stats = _union_crossing_continuity(
            expanded, support, ribbon_bank_count=4
        )
        np.testing.assert_array_equal(
            union["edgeFirstFrontierIndex"], (0, 1)
        )
        np.testing.assert_array_equal(
            union["edgeSecondFrontierIndex"], (1, 2)
        )
        self.assertEqual(stats["unionEdgeCount"], 2)

    def test_prior_and_dormant_exact_states_combine_in_expanded_space(self) -> None:
        def compiled(
            eligible: tuple[int, ...],
            added_offset: tuple[int, ...],
            added: tuple[int, ...],
            removed_offset: tuple[int, ...],
            removed: tuple[int, ...],
            value: float,
        ) -> dict[str, np.ndarray]:
            result = {
                "corridorEvidenceEligible": np.asarray(
                    eligible, dtype=np.uint8
                ),
                "corridorProposalAddedOffset": np.asarray(
                    added_offset, dtype=np.int64
                ),
                "corridorProposalAddedFrontierIndex": np.asarray(
                    added, dtype=np.int32
                ),
                "corridorProposalRemovedOffset": np.asarray(
                    removed_offset, dtype=np.int64
                ),
                "corridorProposalRemovedFrontierIndex": np.asarray(
                    removed, dtype=np.int32
                ),
            }
            for name in (
                "corridorProposalLocalObjective",
                "corridorProposalObjectiveDelta",
                "corridorProposalPatchCoverage",
                "corridorProposalRetainedBoundaryFraction",
                "corridorProposalBoundaryAnchorCount",
            ):
                result[name] = np.full(3, value, dtype=np.float32)
            return result

        prior = compiled((1, 0, 0), (0, 1, 1, 1), (1,), (0, 1, 1, 1), (2,), 1.0)
        dormant = compiled((0, 1, 0), (0, 0, 1, 1), (5,), (0, 0, 0, 0), (), 2.0)
        combined, audit = _combine_compiled_reconfigurations(
            {},
            prior,
            dormant,
            prior_frontier_to_expanded=np.asarray((10, 11, 12), dtype=np.int32),
        )
        np.testing.assert_array_equal(
            combined["corridorEvidenceEligible"], (1, 1, 0)
        )
        np.testing.assert_array_equal(
            combined["corridorProposalAddedOffset"], (0, 1, 2, 2)
        )
        np.testing.assert_array_equal(
            combined["corridorProposalAddedFrontierIndex"], (11, 5)
        )
        np.testing.assert_array_equal(
            combined["corridorProposalRemovedFrontierIndex"], (12,)
        )
        np.testing.assert_array_equal(
            combined["corridorProposalObjectiveDelta"], (1.0, 2.0, 1.0)
        )
        np.testing.assert_array_equal(
            audit["combinedCorridorDecisionSource"], (1, 2, 0)
        )

    def test_dormant_screen_excludes_only_inherited_crossing_debt(self) -> None:
        crossings = {
            "crossingFirstFrontierIndex": np.asarray(
                (0, 0, 1, 2), dtype=np.int32
            ),
            "crossingSecondFrontierIndex": np.asarray(
                (1, 2, 3, 3), dtype=np.int32
            ),
            "crossingDistanceVoxels": np.asarray(
                (0.1, 0.2, 0.3, 0.4), dtype=np.float32
            ),
        }
        conditioned, stats = _condition_crossings_on_immutable_baseline(
            crossings, np.asarray((1, 1, 0, 0), dtype=bool)
        )
        np.testing.assert_array_equal(
            conditioned["crossingFirstFrontierIndex"], (0, 1, 2)
        )
        np.testing.assert_array_equal(
            conditioned["crossingSecondFrontierIndex"], (2, 3, 3)
        )
        self.assertEqual(stats["inheritedBaselineCrossingPairCount"], 1)
        self.assertEqual(stats["enforcedCounterfactualCrossingPairCount"], 3)

    def test_dormant_additions_are_counted_in_bank_identity_space(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray(
                (0, 3, 5, 5), dtype=np.int64
            ),
            "corridorVariantAddedFrontierIndex": np.asarray(
                (0, 1, 3, 2, 4), dtype=np.int32
            ),
        }
        expanded_frontier = np.asarray((7, 2, 9, 5, 4), dtype=np.int32)
        base_bank_mask = np.zeros(10, dtype=bool)
        base_bank_mask[[2, 4, 7]] = True
        np.testing.assert_array_equal(
            _variant_dormant_addition_count(
                variants, expanded_frontier, base_bank_mask
            ),
            (1, 1, 0),
        )

    def test_corridor_extension_preserves_ragged_variant_identity(self) -> None:
        full = {
            "corridorVariantRow": np.asarray((0, 0, 1), dtype=np.int32),
            "corridorVariantRank": np.asarray((0, 1, 0), dtype=np.int16),
            "corridorVariantAddedOffset": np.asarray(
                (0, 1, 3, 4), dtype=np.int64
            ),
            "corridorVariantAddedFrontierIndex": np.asarray(
                (10, 11, 12, 13), dtype=np.int32
            ),
            "corridorVariantRemovedOffset": np.asarray(
                (0, 1, 2, 4), dtype=np.int64
            ),
            "corridorVariantRemovedFrontierIndex": np.asarray(
                (20, 21, 22, 23), dtype=np.int32
            ),
        }
        for name, dtype in (
            ("corridorVariantLocalObjective", np.float32),
            ("corridorVariantLocalObjectiveDelta", np.float32),
            ("corridorVariantPatchCoverage", np.float32),
            ("corridorVariantFirstArcAnchorCount", np.int16),
            ("corridorVariantSecondArcAnchorCount", np.int16),
            ("corridorVariantRetainedBoundaryFraction", np.float32),
        ):
            full[name] = np.arange(3, dtype=dtype)
        delta, source_index = _delta_variant_arrays(
            full, np.asarray((2, 1), dtype=np.int32), corridor_count=2
        )
        np.testing.assert_array_equal(source_index, (1, 2))
        np.testing.assert_array_equal(delta["corridorVariantOffset"], (0, 1, 2))
        np.testing.assert_array_equal(
            delta["corridorVariantAddedFrontierIndex"], (11, 12, 13)
        )
        self.assertEqual(
            _variant_signature(delta, 0),
            _variant_signature(full, 1),
        )

    def test_global_corridor_set_uses_compatible_lower_local_choice(self) -> None:
        empty_key = (0.0, 0.0, 0.0, 0.0, 0.0)
        states = (
            (
                {
                    "stateIndex": 0,
                    "variantIndices": (),
                    "added": frozenset(),
                    "removed": frozenset(),
                    "key": empty_key,
                },
                {
                    "stateIndex": 1,
                    "variantIndices": (10,),
                    "added": frozenset((1,)),
                    "removed": frozenset(),
                    "key": (2.0, 0.0, 1.0, 0.0, 0.0),
                },
                {
                    "stateIndex": 2,
                    "variantIndices": (11,),
                    "added": frozenset((3,)),
                    "removed": frozenset(),
                    "key": (1.0, 0.0, 1.0, 0.0, 0.0),
                },
            ),
            (
                {
                    "stateIndex": 3,
                    "variantIndices": (),
                    "added": frozenset(),
                    "removed": frozenset(),
                    "key": empty_key,
                },
                {
                    "stateIndex": 4,
                    "variantIndices": (20,),
                    "added": frozenset((2,)),
                    "removed": frozenset(),
                    "key": (2.0, 0.0, 1.0, 0.0, 0.0),
                },
            ),
        )
        chosen = _choose_global_variant_states(
            states,
            valid_modifications=lambda added, _removed: not {1, 2} <= added,
            beam_width=32,
        )
        self.assertEqual(chosen["variantIndices"], (11, 20))

    def test_exact_variant_compiles_back_to_replay_contract(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray((0, 2, 3), dtype=np.int64),
            "corridorVariantAddedFrontierIndex": np.asarray(
                (4, 5, 7), dtype=np.int32
            ),
            "corridorVariantRemovedOffset": np.asarray((0, 1, 3), dtype=np.int64),
            "corridorVariantRemovedFrontierIndex": np.asarray(
                (1, 2, 3), dtype=np.int32
            ),
            "corridorVariantLocalObjective": np.asarray(
                (2.0, 1.8), dtype=np.float32
            ),
            "corridorVariantLocalObjectiveDelta": np.asarray(
                (0.4, -0.1), dtype=np.float32
            ),
            "corridorVariantPatchCoverage": np.asarray(
                (0.7, 0.9), dtype=np.float32
            ),
            "corridorVariantFirstArcAnchorCount": np.asarray(
                (2, 3), dtype=np.int16
            ),
            "corridorVariantSecondArcAnchorCount": np.asarray(
                (1, 2), dtype=np.int16
            ),
            "corridorVariantRetainedBoundaryFraction": np.asarray(
                (0.8, 0.6), dtype=np.float32
            ),
        }
        exact = {
            "corridorChosenExactVariant": np.asarray((1, -1), dtype=np.int32),
            "corridorVariantTriangleRegionCountBefore": np.asarray(
                (3, 4), dtype=np.int32
            ),
            "corridorVariantTriangleRegionCountAfter": np.asarray(
                (2, 3), dtype=np.int32
            ),
            "corridorVariantTriangleAreaBefore": np.asarray(
                (10.0, 20.0), dtype=np.float32
            ),
            "corridorVariantTriangleAreaAfter": np.asarray(
                (12.0, 21.0), dtype=np.float32
            ),
        }
        compiled = compile_exact_variant_reconfiguration({}, variants, exact)
        np.testing.assert_array_equal(
            compiled["corridorProposalAddedFrontierIndex"], (7,)
        )
        np.testing.assert_array_equal(
            compiled["corridorProposalRemovedFrontierIndex"], (2, 3)
        )
        np.testing.assert_array_equal(
            compiled["corridorProposalAddedOffset"], (0, 1, 1)
        )
        np.testing.assert_array_equal(
            compiled["corridorEvidenceEligible"], (1, 0)
        )
        self.assertGreater(
            float(compiled["corridorProposalObjectiveDelta"][0]), 100.0
        )

    def test_arc_alignment_discards_crossing_edge_hypothesis(self) -> None:
        records = [
            (0, 4, 2.0, 0.0, 2.0),
            (1, 5, 2.0, 0.0, 2.0),
            (2, 7, 2.0, 0.0, 2.0),
            (3, 6, 2.0, 0.0, 2.0),
        ]
        position = np.asarray((0, 1, 2, 3, 0, 1, 2, 3), dtype=np.int32)
        loop_count = np.full(8, 8, dtype=np.int32)
        chain, _, density = _best_monotone_corridor_chain(
            records, position, loop_count, maximum_gap=4
        )
        self.assertEqual(len(chain), 3)
        self.assertGreaterEqual(density, 0.5)

    def test_vertex_touch_does_not_count_as_corridor_closure(self) -> None:
        center = np.asarray(
            ((0, 0, 0), (-1, 0, 0), (0, 1, 0), (1, 0, 0), (0, -1, 0)),
            dtype=np.float32,
        )
        surface = {
            "triangleFrontierIndex": np.asarray(((0, 1, 2), (0, 3, 4)), dtype=np.int32),
            "midpointXYZ": center,
            "signedNormalXYZ": np.tile((0, 0, 1), (5, 1)).astype(np.float32),
        }
        corridors = {
            "corridorPairOffset": np.asarray((0, 1), dtype=np.int64),
            "corridorFirstBoundaryEdge": np.asarray((0,), dtype=np.int32),
            "corridorSecondBoundaryEdge": np.asarray((1,), dtype=np.int32),
            "boundaryEdgeFirstFrontierIndex": np.asarray((1, 3), dtype=np.int32),
            "boundaryEdgeSecondFrontierIndex": np.asarray((2, 4), dtype=np.int32),
            "boundaryEdgeMidpointXYZ": np.asarray(((-0.5, 0.5, 0), (0.5, -0.5, 0)), dtype=np.float32),
            "boundaryEdgeNormalXYZ": np.tile((0, 0, 1), (2, 1)).astype(np.float32),
            "boundaryEdgeLengthVoxels": np.full(2, np.sqrt(2), dtype=np.float32),
            "boundaryEdgeTriangleRegion": np.asarray((0, 1), dtype=np.int32),
        }
        scored = {
            "scoredCorridorIndex": np.asarray((0,), dtype=np.int32),
            "corridorPatchOffset": np.asarray((0, 1), dtype=np.int64),
            "corridorPatchXYZ": np.zeros((1, 3), dtype=np.float32),
        }
        disconnected = _evaluate_corridor_connections(surface, corridors, scored)
        self.assertEqual(int(disconnected["boundaryArcsConnected"][0]), 0)
        surface["triangleFrontierIndex"] = np.asarray(
            ((0, 1, 2), (0, 3, 4), (1, 2, 3), (2, 3, 4)), dtype=np.int32
        )
        connected = _evaluate_corridor_connections(surface, corridors, scored)
        self.assertEqual(int(connected["boundaryArcsConnected"][0]), 1)

    def test_hermite_corridor_obeys_both_endpoint_tangent_planes(self) -> None:
        along = np.asarray((0.0, 1.0, 2.0), dtype=np.float32)
        first = np.column_stack((np.zeros(3), along, np.zeros(3)))
        second = np.column_stack((np.full(3, 4.0), along, np.ones(3)))
        corridors = {
            "corridorPairOffset": np.asarray((0, 3), dtype=np.int64),
            "corridorFirstBoundaryEdge": np.asarray((0, 1, 2), dtype=np.int32),
            "corridorSecondBoundaryEdge": np.asarray((3, 4, 5), dtype=np.int32),
            "boundaryEdgeMidpointXYZ": np.vstack((first, second)).astype(np.float32),
            "boundaryEdgeMidpointUV": np.vstack(
                (
                    np.column_stack((np.zeros(3), along)),
                    np.column_stack((np.full(3, 4.0), along)),
                )
            ).astype(np.float32),
            "boundaryEdgeNormalXYZ": np.vstack(
                (
                    np.tile((0.0, 0.0, 1.0), (3, 1)),
                    np.tile((1.0, 0.0, 0.0), (3, 1)),
                )
            ).astype(np.float32),
            "boundaryEdgeThicknessVoxels": np.ones(6, dtype=np.float32),
        }
        model = _corridor_model_grid(
            0,
            corridors,
            settings=PhysicalRibbonPatchCorridorSettings(
                hermite_tensions=(1.0,), patch_pixel_step_voxels=0.25
            ),
        )
        points = np.asarray(model["pointsXYZ"])[1]
        normals = np.asarray(model["normalXYZ"])[1]
        np.testing.assert_allclose(points[:, 0], model["firstBoundaryXYZ"], atol=1e-5)
        np.testing.assert_allclose(points[:, -1], model["secondBoundaryXYZ"], atol=1e-5)
        self.assertGreater(float(np.min(np.abs(normals[:, 0, 2]))), 0.98)
        self.assertGreater(float(np.min(np.abs(normals[:, -1, 0]))), 0.98)

    def test_boundary_texture_correlation_allows_small_phase_shift(self) -> None:
        trace = np.asarray((0, 3, -2, 5, -4, 2, 1, -3, 4, -1), dtype=np.float32)
        shifted = np.concatenate((np.zeros(1, dtype=np.float32), trace[:-1]))
        unrelated = np.asarray((1, 1, -1, -1, 1, 1, -1, -1, 1, 1), dtype=np.float32)
        self.assertGreater(_shifted_trace_correlation(trace, shifted), 0.95)
        self.assertLess(_shifted_trace_correlation(trace, unrelated), 0.75)

    def test_triangle_region_requires_shared_mesh_edges(self) -> None:
        triangles = np.asarray(
            ((0, 1, 2), (2, 1, 3), (4, 5, 6)), dtype=np.int32
        )
        region = _triangle_region_labels(triangles)
        self.assertEqual(int(region[0]), int(region[1]))
        self.assertNotEqual(int(region[0]), int(region[2]))


if __name__ == "__main__":
    unittest.main()
