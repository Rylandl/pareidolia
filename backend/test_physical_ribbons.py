from __future__ import annotations

import unittest
from pathlib import Path
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
from backend.cubical.physical_ribbon_collective import (
    PhysicalRibbonCollectiveSettings,
    _filter_unrealized_surface_proposals,
    optimize_collective_patch,
)
from backend.cubical.physical_ribbon_flattened_audit import (
    _rank_exact_variant_rows,
    boundary_texture_compatibility,
    flattened_texture_structure,
)
from backend.cubical.physical_ribbon_depth_fields import (
    _coherent_supported_fraction,
    _grid_edges,
    _profile_fields,
    solve_collective_depth_labels,
)
from backend.cubical.physical_ribbon_patch_states import (
    PhysicalRibbonPatchStateSettings,
    _lineage_audit,
    _patch_state_groups,
    _patch_state_proxy_screen,
    _prepare_component_exact_graph,
    _selection_conflicts,
    optimize_collective_patch_coverage,
    optimize_collective_patch_coverage_binary,
    optimize_collective_patch_coverage_ensemble,
)
from backend.cubical.physical_ribbon_texture_gate import (
    texture_patch_decisions,
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
from backend.cubical.physical_ribbon_cumulative_hole_replay import (
    _proposal_state_for_rows,
)
from backend.cubical.physical_ribbon_corridor_dormant import (
    _combine_compiled_reconfigurations,
    _condition_crossings_on_immutable_baseline,
    _union_crossing_continuity,
    _variant_dormant_addition_count,
)
from backend.cubical.physical_ribbon_corridor_one_sided import (
    _array_mapping_sha256,
    _variant_one_sided_addition_count,
)
from backend.cubical.physical_ribbon_corridor_frontier import (
    _map_frontier_by_bank,
    _successful_corridor_mask,
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
    _preserves_prior_component_anchors,
)
from backend.cubical.physical_ribbon_complete_strips import (
    PhysicalRibbonCompleteStripSettings,
    _area_retention_decision,
    _residual_corridor_rows,
    _selection_key as _complete_strip_selection_key,
    _split_audit,
    _strict_surface,
)
from backend.cubical.physical_ribbon_complete_strip_replay import (
    _grouped_candidate_states,
)
from backend.cubical.physical_ribbon_cumulative_replay import (
    cumulative_face_replay_reference,
    cumulative_prior_exact_reference,
)
from backend.cubical.physical_ribbon_lineage_strips import (
    _affected_lineages_preserved,
    _lineage_preserved,
    _lineage_target_rows,
    _select_lineage_variant_candidates,
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
from backend.cubical.physical_ribbon_cumulative_corridor_replay import (
    _candidate_records as _cumulative_candidate_records,
    _merge_connection_catalogs,
    _merge_supplemental_faces,
)
from backend.cubical.physical_ribbon_cumulative_hole_replay import (
    PhysicalRibbonCumulativeHoleReplaySettings,
    _eligible_hole_proposals,
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

    def test_materialize_chained_replay_loads_constraints_from_configuration(
        self,
    ) -> None:
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
        }
        constraints = {
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
        _, configuration, stats = _materialize_replay_arrays(
            replay,
            topology,
            ribbon,
            constraint_configuration=constraints,
        )
        np.testing.assert_array_equal(configuration["selected"], (1, 1, 0))
        np.testing.assert_array_equal(
            configuration["crossingFirstFrontierIndex"], (0,)
        )
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


class PhysicalRibbonCollectiveTests(unittest.TestCase):
    def test_patch_state_settings_allow_disabling_forced_exclusions(self) -> None:
        settings = PhysicalRibbonPatchStateSettings(
            maximum_forced_exclusion_trials=0,
            maximum_individual_hole_scopes=0,
        )
        self.assertEqual(settings.maximum_forced_exclusion_trials, 0)
        self.assertEqual(settings.maximum_individual_hole_scopes, 0)

    def test_patch_state_groups_include_complete_single_loop_scopes(self) -> None:
        groups = _patch_state_groups(
            {4: [0, 1, 2]},
            {"patchOffset": np.asarray((0, 10, 15, 35), dtype=np.int64)},
            maximum_individual_hole_scopes=2,
        )
        self.assertEqual(
            [
                (component, rows.tolist(), scope, hole_row)
                for component, rows, scope, hole_row in groups
            ],
            [
                (4, [0, 1, 2], "component", -1),
                (4, [2], "hole", 2),
                (4, [0], "hole", 0),
            ],
        )

    def test_ct_supported_counterexample_reaches_exact_screen(self) -> None:
        settings = PhysicalRibbonPatchStateSettings()
        screened, objective, counterexample = _patch_state_proxy_screen(
            -8.0,
            1.0,
            depth_field_present=True,
            settings=settings,
        )
        self.assertTrue(screened)
        self.assertFalse(objective)
        self.assertTrue(counterexample)
        screened, _, counterexample = _patch_state_proxy_screen(
            -8.0,
            0.5,
            depth_field_present=True,
            settings=settings,
        )
        self.assertFalse(screened)
        self.assertFalse(counterexample)

    def test_collective_patch_crosses_negative_single_node_barrier(self) -> None:
        selected, objective = optimize_collective_patch(
            np.full(3, -0.6, dtype=np.float32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1, 2), dtype=np.int32),
            np.ones(2, dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
        )
        np.testing.assert_array_equal(selected, (1, 1, 1))
        self.assertAlmostEqual(objective, 0.2, places=5)

    def test_collective_patch_enforces_joint_hard_conflicts(self) -> None:
        selected, objective = optimize_collective_patch(
            np.asarray((-0.2, -0.2, -0.2), dtype=np.float32),
            np.asarray((0, 1, 0), dtype=np.int32),
            np.asarray((2, 2, 1), dtype=np.int32),
            np.asarray((0.8, 0.8, 0.1), dtype=np.float32),
            np.asarray((0,), dtype=np.int32),
            np.asarray((1,), dtype=np.int32),
        )
        self.assertFalse(bool(selected[0] and selected[1]))
        self.assertTrue(bool(selected[2]))
        self.assertGreater(objective, 0.0)

    def test_collective_patch_accepts_feasible_incumbent_seed(self) -> None:
        selected, objective = optimize_collective_patch(
            np.asarray((-0.7, -0.7, -0.7), dtype=np.float32),
            np.asarray((0, 1), dtype=np.int32),
            np.asarray((1, 2), dtype=np.int32),
            np.asarray((1.2, 1.2), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            initial_selection=np.ones(3, dtype=bool),
        )
        np.testing.assert_array_equal(selected, (1, 1, 1))
        self.assertAlmostEqual(objective, 0.3, places=5)

    def test_patch_state_lineage_detects_split_and_fusion(self) -> None:
        baseline_selected = np.ones(4, dtype=bool)
        baseline_component = np.asarray((0, 0, 1, 1), dtype=np.int32)
        _, split = _lineage_audit(
            baseline_selected,
            baseline_component,
            baseline_selected,
            np.asarray((0, 2, 1, 1), dtype=np.int32),
        )
        self.assertEqual(split["splitPriorComponentCount"], 1)
        _, fusion = _lineage_audit(
            baseline_selected,
            baseline_component,
            baseline_selected,
            np.zeros(4, dtype=np.int32),
        )
        self.assertEqual(fusion["crossPriorComponentFusionCount"], 1)

    def test_patch_state_hard_conflicts_are_global(self) -> None:
        selected = np.asarray((1, 1, 0), dtype=bool)
        self.assertEqual(
            _selection_conflicts(
                selected,
                np.asarray((0, 2, 0), dtype=np.int32),
                np.asarray((1, 3, 4), dtype=np.int32),
                np.asarray((0,), dtype=np.int32),
                np.asarray((1,), dtype=np.int32),
            ),
            (0, 1),
        )

    def test_whole_patch_coverage_avoids_duplicate_candidate_cluster(self) -> None:
        selected, stats = optimize_collective_patch_coverage(
            np.asarray((0.4, 0.3, 0.0), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.asarray((0,), dtype=np.int32),
            np.asarray((2,), dtype=np.int32),
            np.asarray(
                (
                    (1, 0),
                    (1, 0),
                    (0, 1),
                ),
                dtype=bool,
            ),
            np.ones(2, dtype=np.float32),
            maximum_sweeps=4,
            initial_selection=np.asarray((1, 0, 0), dtype=bool),
        )
        np.testing.assert_array_equal(selected, (0, 1, 1))
        self.assertEqual(stats["coveredPixelFraction"], 1.0)
        self.assertFalse(stats["singleCellGrowth"])

    def test_patch_state_ensemble_retains_complete_matching_alternatives(self) -> None:
        states, stats = optimize_collective_patch_coverage_ensemble(
            np.asarray((0.4, 0.35, 0.1, 0.09), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.asarray((0, 2), dtype=np.int32),
            np.asarray((1, 3), dtype=np.int32),
            np.asarray(
                (
                    (1, 0),
                    (1, 0),
                    (0, 1),
                    (0, 1),
                ),
                dtype=bool,
            ),
            np.ones(2, dtype=np.float32),
            maximum_sweeps=4,
            initial_selection=np.asarray((1, 0, 0, 0), dtype=bool),
            maximum_states=4,
            maximum_forced_exclusion_trials=4,
            minimum_hamming_fraction=0.01,
        )
        self.assertGreaterEqual(len(states), 2)
        self.assertGreaterEqual(stats["discoveredStateCount"], 2)
        for selected in states:
            self.assertFalse(bool(selected[0] and selected[1]))
            self.assertFalse(bool(selected[2] and selected[3]))
            self.assertTrue(bool(np.any(selected[:2])))
            self.assertTrue(bool(np.any(selected[2:])))

    def test_binary_patch_optimizer_exposes_coverage_lexicographic_state(self) -> None:
        records, statistics = optimize_collective_patch_coverage_binary(
            np.asarray((10.0, 0.0, 0.0), dtype=np.float32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
            np.asarray((0,), dtype=np.int32),
            np.asarray((1,), dtype=np.int32),
            np.asarray(
                (
                    (1, 0),
                    (0, 1),
                    (1, 0),
                ),
                dtype=bool,
            ),
            np.ones(2, dtype=np.float32),
            maximum_states=2,
            time_limit_seconds=10.0,
        )
        self.assertGreater(int(statistics["provenOptimalStateCount"]), 0)
        self.assertTrue(
            any(float(record["coveredPixelFraction"]) == 1.0 for record in records)
        )
        for record in records:
            selected = np.asarray(record["selected"], dtype=bool)
            self.assertFalse(bool(selected[0] and selected[1]))

    def test_component_exact_graph_marks_selected_external_neighbors(self) -> None:
        graph = _prepare_component_exact_graph(
            4,
            [
                {
                    "added": np.asarray((2,), dtype=np.int32),
                    "removed": np.empty(0, dtype=np.int32),
                }
            ],
            [0],
            np.asarray((1, 1, 0, 0, 1), dtype=bool),
            np.asarray((4, 4, -1, -1, 7), dtype=np.int32),
            {
                "edgeFirstFrontierIndex": np.asarray(
                    (0, 1, 2, 0), dtype=np.int32
                ),
                "edgeSecondFrontierIndex": np.asarray(
                    (1, 2, 4, 3), dtype=np.int32
                ),
                "edgeScore": np.ones(4, dtype=np.float32),
            },
        )
        np.testing.assert_array_equal(graph.local_to_global, (0, 1, 2))
        np.testing.assert_array_equal(graph.baseline_selected, (1, 1, 0))
        np.testing.assert_array_equal(graph.edge_first, (0, 1))
        np.testing.assert_array_equal(graph.edge_second, (1, 2))
        np.testing.assert_array_equal(
            graph.external_selected_neighbor, (0, 0, 1)
        )

    def test_flattened_texture_audit_recovers_coherent_axial_pattern(self) -> None:
        image = np.tile(np.arange(32, dtype=np.float32), (32, 1))
        _, statistics = flattened_texture_structure(
            image,
            np.ones_like(image, dtype=bool),
            window_radius=2,
            minimum_coherence=0.15,
        )
        self.assertGreater(float(statistics["medianCoherence"]), 0.99)
        self.assertLess(
            float(statistics["adjacentAxialDisagreementDegrees"]["p90"]),
            0.01,
        )

    def test_texture_compatibility_scales_with_same_surface_control(self) -> None:
        compatible = boundary_texture_compatibility(
            {"count": 20, "median": 27.0},
            {"count": 200, "median": 17.0, "p90": 67.0},
            minimum_measurements=12,
            median_excess_floor_degrees=5.0,
            control_spread_fraction=0.25,
        )
        self.assertTrue(bool(compatible["compatible"]))
        incompatible = boundary_texture_compatibility(
            {"count": 20, "median": 32.0},
            {"count": 200, "median": 17.0, "p90": 67.0},
            minimum_measurements=12,
            median_excess_floor_degrees=5.0,
            control_spread_fraction=0.25,
        )
        self.assertFalse(bool(incompatible["compatible"]))

    def test_texture_gate_maps_patch_through_final_component(self) -> None:
        geometry, measured, compatible, accepted = texture_patch_decisions(
            {
                "patchAccepted": np.asarray((1, 1, 0), dtype=np.uint8),
                "patchAddedOffset": np.asarray((0, 2, 3, 3), dtype=np.int64),
                "patchAddedFrontierIndex": np.asarray((1, 2, 4), dtype=np.int32),
                "component": np.asarray((0, 5, 5, 1, 7), dtype=np.int32),
            },
            {
                "audit": {
                    "components": [
                        {"componentId": 5, "boundaryTextureCompatible": True},
                        {"componentId": 7, "boundaryTextureCompatible": False},
                    ]
                }
            },
        )
        np.testing.assert_array_equal(geometry, (1, 1, 0))
        np.testing.assert_array_equal(measured, (1, 1, 0))
        np.testing.assert_array_equal(compatible, (1, 0, 0))
        np.testing.assert_array_equal(accepted, (1, 0, 0))

    def test_texture_gate_tries_exact_alternative_after_ct_rejection(self) -> None:
        def exact(row: int, component: int, area: float) -> dict[str, object]:
            return {
                "patchRow": row,
                "priorComponent": component,
                "accepted": True,
                "before": {
                    "macroHoleCount": 1,
                    "interiorHoleCount": 2,
                    "triangleRegionCount": 3,
                    "triangleCount": 20,
                },
                "after": {
                    "macroHoleCount": 0,
                    "interiorHoleCount": 2,
                    "triangleRegionCount": 3,
                    "triangleCount": 20,
                },
                "triangleAreaRetention": area,
            }

        geometry, measured, compatible, accepted = texture_patch_decisions(
            {
                "patchAccepted": np.asarray((1, 0, 0), dtype=np.uint8),
                "patchTargetPriorComponent": np.asarray((5, 5, 7), dtype=np.int32),
            },
            {
                "audit": {
                    "variants": [
                        {
                            "patchRow": 0,
                            "componentId": 5,
                            "variantRank": 0,
                            "objectiveGain": 4.0,
                            "boundaryTextureCompatible": False,
                            "exactAudit": exact(0, 5, 1.02),
                        },
                        {
                            "patchRow": 1,
                            "componentId": 5,
                            "variantRank": 1,
                            "objectiveGain": 3.0,
                            "boundaryTextureCompatible": True,
                            "exactAudit": exact(1, 5, 1.01),
                        },
                        {
                            "patchRow": 2,
                            "componentId": 7,
                            "variantRank": 0,
                            "objectiveGain": -1.0,
                            "boundaryTextureCompatible": True,
                            "exactAudit": exact(2, 7, 1.00),
                        },
                    ]
                }
            },
        )
        np.testing.assert_array_equal(geometry, (1, 1, 1))
        np.testing.assert_array_equal(measured, (1, 1, 1))
        np.testing.assert_array_equal(compatible, (0, 1, 1))
        np.testing.assert_array_equal(accepted, (0, 1, 1))

    def test_flattened_variant_budget_is_shared_across_components(self) -> None:
        def record(row: int, component: int, area: float) -> dict[str, object]:
            return {
                "patchRow": row,
                "priorComponent": component,
                "variantRank": row,
                "before": {
                    "macroHoleCount": 1,
                    "interiorHoleCount": 1,
                    "triangleRegionCount": 1,
                    "triangleCount": 10,
                },
                "after": {
                    "macroHoleCount": 0,
                    "interiorHoleCount": 1,
                    "triangleRegionCount": 1,
                    "triangleCount": 10,
                },
                "triangleAreaRetention": area,
            }

        ranked = _rank_exact_variant_rows(
            [
                record(0, 0, 1.03),
                record(1, 0, 1.02),
                record(2, 0, 1.01),
                record(3, 12, 1.04),
                record(4, 12, 1.00),
            ],
            3,
        )
        self.assertEqual([int(value["patchRow"]) for value in ranked], [0, 3, 1])

    def test_collective_patch_requires_realized_mesh_attachment(self) -> None:
        arrays = {
            "proposalOffset": np.asarray((0, 1), dtype=np.int64),
            "proposalFrontierIndex": np.asarray((1,), dtype=np.int32),
            "proposalAccepted": np.asarray((1,), dtype=np.uint8),
            "proposalRejectionReason": np.asarray((0,), dtype=np.uint8),
            "proposalAnchorComponent": np.asarray((0,), dtype=np.int32),
            "selected": np.asarray((1, 1), dtype=np.uint8),
            "component": np.asarray((0, 0), dtype=np.int32),
        }
        statistics = {
            "acceptedProposalCount": 1,
            "acceptedRibbonCount": 1,
        }
        filtered = _filter_unrealized_surface_proposals(
            arrays,
            statistics,
            {"triangleFrontierIndex": np.empty((0, 3), dtype=np.int32)},
            {
                "sourceInterface": np.asarray((0, 2), dtype=np.int32),
                "targetInterface": np.asarray((1, 3), dtype=np.int32),
                "interfaceCandidateDegree": np.ones(4, dtype=np.int32),
            },
            {
                "frontierRibbonCandidate": np.asarray((0, 1), dtype=np.int32),
                "edgeFirstFrontierIndex": np.asarray((0,), dtype=np.int32),
                "edgeSecondFrontierIndex": np.asarray((1,), dtype=np.int32),
            },
            {
                "selected": np.asarray((1, 0), dtype=np.uint8),
                "component": np.asarray((0, -1), dtype=np.int32),
            },
            {"triangleFrontierIndex": np.empty((0, 3), dtype=np.int32)},
            settings=PhysicalRibbonCollectiveSettings(),
        )
        self.assertTrue(filtered)
        np.testing.assert_array_equal(arrays["selected"], (1, 0))
        np.testing.assert_array_equal(arrays["proposalAccepted"], (0,))
        self.assertEqual(statistics["surfaceRejectedProposalCount"], 1)


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
    def test_counterexample_trim_rebuilds_complete_hole_state(self) -> None:
        candidates = [
            {
                "row": row,
                "eligible": True,
                "added": frozenset((10 + row,)),
                "removed": frozenset((20 + row,)),
                "patchCoverage": 0.8 + 0.05 * row,
                "profileCorrelation": 0.9,
                "competingLayerMargin": 0.3,
                "localObjectiveDelta": 1.0 + row,
            }
            for row in range(3)
        ]
        state = _proposal_state_for_rows(
            candidates,
            (0, 2),
            valid_modifications=lambda added, removed: (
                len(added) == len(removed)
            ),
        )
        self.assertIsNotNone(state)
        assert state is not None
        self.assertEqual(state["rows"], (0, 2))
        self.assertEqual(state["added"], frozenset((10, 12)))
        self.assertEqual(state["removed"], frozenset((20, 22)))
        self.assertEqual(state["key"][0], 2.0)

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

    def test_touching_boundary_fans_are_traced_without_discarding_loop(self) -> None:
        # Two triangular boundary lobes share vertex zero, while their
        # triangles remain one edge-connected surface through the center
        # strip.  The undirected boundary has degree four at zero; its triangle
        # link nevertheless contains two independent fans.  Pairing through
        # those fans recovers the one abstract boundary circle, with the shared
        # embedded vertex visited twice.
        chart = np.asarray(
            (
                (0.0, 0.0),
                (-2.0, -1.0),
                (-2.0, 1.0),
                (2.0, -1.0),
                (2.0, 1.0),
            ),
            dtype=np.float32,
        )
        triangles = np.asarray(
            (
                (0, 1, 2),
                (1, 3, 2),
                (2, 3, 4),
                (0, 3, 4),
            ),
            dtype=np.int32,
        )
        loops, stats = extract_surface_boundary_loops(
            {
                "triangleFrontierIndex": triangles,
                "chartUV": chart,
                "component": np.zeros(5, dtype=np.int32),
                "thicknessVoxels": np.ones(5, dtype=np.float32),
            },
            settings=PhysicalRibbonPatchHoleSettings(),
        )
        self.assertEqual(stats["closedBoundaryLoopCount"], 1)
        self.assertEqual(stats["pinchedBoundaryComponentCount"], 1)
        self.assertEqual(stats["splitBoundaryFanCount"], 1)
        self.assertEqual(stats["nonCycleBoundaryComponentCount"], 0)
        values = loops["loopVertexFrontierIndex"]
        self.assertEqual(len(values), 6)
        self.assertEqual(set(values.tolist()), {0, 1, 2, 3, 4})
        self.assertEqual(int(np.count_nonzero(values == 0)), 2)

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


class PhysicalRibbonDepthFieldTests(unittest.TestCase):
    def test_collective_line_solve_crosses_a_unary_barrier(self) -> None:
        coordinates = np.column_stack(
            (np.arange(7, dtype=np.int32), np.zeros(7, dtype=np.int32))
        )
        shifts = np.asarray((0.0, 1.0), dtype=np.float32)
        # The all-one state is locally stable to every single-pixel change:
        # each +1 unary improvement would create at least one pairwise cut.
        # A complete row solve can make the globally superior all-zero move.
        unary = np.column_stack(
            (np.ones(7, dtype=np.float32), np.zeros(7, dtype=np.float32))
        )
        labels, stats = solve_collective_depth_labels(
            unary,
            coordinates,
            shifts,
            smoothness_weight=2.0,
            truncation_thicknesses=1.0,
            maximum_sweeps=4,
        )
        np.testing.assert_array_equal(labels, np.zeros(7, dtype=np.int16))
        self.assertFalse(stats["singlePixelGrowth"])

    def test_truncated_smoothness_preserves_coherent_delamination(self) -> None:
        coordinates = np.asarray(
            [(u_value, v_value) for v_value in range(3) for u_value in range(8)],
            dtype=np.int32,
        )
        shifts = np.asarray((-1.0, 0.0, 1.0), dtype=np.float32)
        unary = np.zeros((len(coordinates), len(shifts)), dtype=np.float32)
        unary[coordinates[:, 0] < 4, 0] = 2.0
        unary[coordinates[:, 0] >= 4, 2] = 2.0
        labels, _ = solve_collective_depth_labels(
            unary,
            coordinates,
            shifts,
            smoothness_weight=0.4,
            truncation_thicknesses=0.5,
            maximum_sweeps=6,
        )
        np.testing.assert_array_equal(labels[coordinates[:, 0] < 4], 0)
        np.testing.assert_array_equal(labels[coordinates[:, 0] >= 4], 2)

    def test_patch_grid_edges_are_face_local(self) -> None:
        coordinates = np.asarray(
            ((0, 0), (1, 0), (1, 1), (3, 0)), dtype=np.int32
        )
        first, second = _grid_edges(coordinates)
        self.assertEqual(
            {tuple(sorted(value)) for value in zip(first.tolist(), second.tolist())},
            {(0, 1), (1, 2)},
        )

    def test_disconnected_patch_is_not_mistaken_for_incoherent_ct(self) -> None:
        first = np.asarray((0, 2), dtype=np.int32)
        second = np.asarray((1, 3), dtype=np.int32)
        self.assertEqual(
            _coherent_supported_fraction(
                np.ones(4, dtype=bool), first, second
            ),
            1.0,
        )
        self.assertEqual(
            _coherent_supported_fraction(
                np.asarray((1, 0, 1, 1), dtype=bool), first, second
            ),
            0.75,
        )

    def test_profile_field_rewards_air_material_air_context(self) -> None:
        depth = np.asarray((-0.8, -0.35, 0.0, 0.35, 0.8), dtype=np.float32)
        context = np.asarray((2.0, 8.0, 10.0, 8.0, 2.0), dtype=np.float32)
        profiles = np.asarray(
            (((2.0, 8.0, 10.0, 8.0, 2.0), (8.0, 3.0, 2.0, 3.0, 8.0)),),
            dtype=np.float32,
        )
        physical, correlation = _profile_fields(profiles, context, depth, 8.0)
        self.assertGreater(float(physical[0, 0]), 0.5)
        self.assertGreater(float(correlation[0, 0]), 0.99)
        self.assertLess(float(physical[0, 1]), 0.0)
        self.assertLess(float(correlation[0, 1]), -0.9)


class PhysicalRibbonPatchCorridorTests(unittest.TestCase):
    def test_cumulative_hole_gate_requires_dense_macro_support(self) -> None:
        arrays = {
            "reconfigurationLoopIndex": np.asarray((0, 1), dtype=np.int32),
            "selectedModel": np.asarray((0, 0), dtype=np.int32),
            "zeroShiftContextProfileCorrelation": np.asarray(
                ((0.99,), (0.62,)), dtype=np.float32
            ),
            "zeroShiftCompetingMargin": np.asarray(
                ((0.42,), (0.14,)), dtype=np.float32
            ),
            "proposalPatchCoverage": np.asarray((1.0, 0.35), dtype=np.float32),
            "proposalRetainedBoundaryFraction": np.asarray(
                (0.9, 0.6), dtype=np.float32
            ),
            "proposalBoundaryAnchorCount": np.asarray((6, 5), dtype=np.int32),
            "proposalObjectiveDelta": np.asarray((4.0, 8.0), dtype=np.float32),
            "loopOffset": np.asarray((0, 6, 16), dtype=np.int64),
            "proposalAddedOffset": np.asarray((0, 1, 2), dtype=np.int64),
            "proposalAddedFrontierIndex": np.asarray((3, 4), dtype=np.int32),
            "proposalRemovedOffset": np.asarray((0, 0, 0), dtype=np.int64),
            "proposalRemovedFrontierIndex": np.empty(0, dtype=np.int32),
        }
        proposals = _eligible_hole_proposals(
            arrays, settings=PhysicalRibbonCumulativeHoleReplaySettings()
        )
        self.assertTrue(proposals[0]["eligible"])
        self.assertFalse(proposals[1]["eligible"])

    def test_cumulative_candidate_beam_uses_only_exact_surface_states(self) -> None:
        arrays = {
            "corridorVariantSurfaceEligible": np.asarray((0, 1), dtype=np.uint8),
            "corridorVariantRow": np.asarray((3, 3), dtype=np.int32),
            "corridorVariantRank": np.asarray((0, 1), dtype=np.int32),
            "corridorVariantTriangleRegionCountBefore": np.asarray(
                (2, 2), dtype=np.int32
            ),
            "corridorVariantTriangleRegionCountAfter": np.asarray(
                (2, 1), dtype=np.int32
            ),
            "corridorVariantTriangleAreaBefore": np.asarray(
                (10.0, 10.0), dtype=np.float32
            ),
            "corridorVariantTriangleAreaAfter": np.asarray(
                (10.0, 11.0), dtype=np.float32
            ),
            "corridorVariantPatchCoverage": np.asarray(
                (0.5, 0.8), dtype=np.float32
            ),
            "corridorVariantLocalObjectiveDelta": np.asarray(
                (1.0, 2.0), dtype=np.float32
            ),
        }
        records = _cumulative_candidate_records(arrays)
        self.assertEqual(len(records), 1)
        self.assertEqual(records[0]["variantIndex"], 1)
        self.assertAlmostEqual(records[0]["strictAreaRetention"], 1.1)

    def test_cumulative_catalog_merge_preserves_prior_and_new_rows(self) -> None:
        settings = PhysicalRibbonCorridorFaceSettings()
        reference = {
            "manifestPath": "corridors.json",
            "manifestSha256": "manifest",
            "dataSha256": "data",
        }
        merged = _merge_connection_catalogs(
            (
                {
                    "corridors": reference,
                    "rows": (2, 4),
                    "faceSettings": settings.record(),
                },
            ),
            reference,
            (4, 7),
            settings,
        )
        self.assertEqual(len(merged), 1)
        self.assertEqual(merged[0]["rows"], [2, 4, 7])

    def test_cumulative_face_merge_deduplicates_across_catalogs(self) -> None:
        def faces(row: int, residual: float) -> dict[str, np.ndarray]:
            return {
                "supplementalTriangleFrontierIndex": np.asarray(
                    ((2, 0, 1),), dtype=np.int32
                ),
                "supplementalTrianglePrimaryCorridorRow": np.asarray(
                    (row,), dtype=np.int32
                ),
                "supplementalTriangleMinimumPath": np.asarray(
                    (row == 4,), dtype=np.uint8
                ),
                "supplementalTriangleAreaVoxelsSquared": np.asarray(
                    (1.0,), dtype=np.float32
                ),
                "supplementalTriangleNodeNormalResidualDegrees": np.asarray(
                    (2.0,), dtype=np.float32
                ),
                "supplementalTriangleCtNormalResidualDegrees": np.asarray(
                    (residual,), dtype=np.float32
                ),
                "supplementalTriangleCenterDistanceThicknesses": np.asarray(
                    (0.1,), dtype=np.float32
                ),
                "supplementalTriangleCenterHeightThicknesses": np.asarray(
                    (0.1,), dtype=np.float32
                ),
                "supplementalTriangleMaximumEdgeThicknesses": np.asarray(
                    (0.5,), dtype=np.float32
                ),
            }

        merged = _merge_supplemental_faces(((0, faces(2, 10.0)), (1, faces(4, 5.0))))
        self.assertEqual(len(merged["supplementalTriangleFrontierIndex"]), 1)
        self.assertEqual(int(merged["supplementalTriangleCatalogIndex"][0]), 1)
        self.assertEqual(int(merged["supplementalTriangleMinimumPath"][0]), 1)

    def test_ct_closure_may_replace_small_strict_area_loss(self) -> None:
        settings = PhysicalRibbonCompleteStripSettings()
        self.assertEqual(
            _area_retention_decision(0.99, 0.99, settings),
            "strict-area-sufficient",
        )
        self.assertEqual(
            _area_retention_decision(0.97, 1.01, settings),
            "ct-closure-replaces-area",
        )
        self.assertEqual(
            _area_retention_decision(0.94, 1.10, settings),
            "insufficient-area",
        )
        self.assertEqual(
            _area_retention_decision(0.97, 0.97, settings),
            "insufficient-area",
        )

    def test_lineage_variant_budget_retains_dense_strip_states(self) -> None:
        candidates = [
            {
                "beamRank": rank,
                "coverage": coverage,
                "firstAnchorCount": 1,
                "secondAnchorCount": 1,
                "retainedBoundaryFraction": 0.5,
                "objective": 10.0 - rank,
            }
            for rank, coverage in enumerate((0.50, 0.55, 0.90, 0.95, 0.80))
        ]
        retained = _select_lineage_variant_candidates(
            candidates,
            maximum_count=4,
            coverage_priority_count=2,
        )
        self.assertEqual(
            [value["beamRank"] for value in retained], [0, 1, 3, 2]
        )
        self.assertEqual(
            [value["selectionClass"] for value in retained], [0, 0, 1, 1]
        )

    def test_cumulative_replay_resolves_source_references_by_schema(self) -> None:
        face_reference = {"manifestPath": "face.json"}
        exact_reference = {"manifestPath": "exact.json"}
        complete = {
            "schema": "pareidolia.physical-ribbon-complete-strip-replay",
            "identity": {"replay": face_reference},
        }
        face = {"identity": {"replay": exact_reference}}
        self.assertIs(cumulative_face_replay_reference(complete), face_reference)
        self.assertIs(
            cumulative_prior_exact_reference(complete, face), exact_reference
        )
        iterative_face = {"manifestPath": "iterative-face.json"}
        iterative_exact = {"manifestPath": "iterative-exact.json"}
        iterative = {
            "schema": "pareidolia.physical-ribbon-lineage-strip-replay",
            "identity": {
                "faceReplay": iterative_face,
                "priorExactReplay": iterative_exact,
            },
        }
        self.assertIs(
            cumulative_face_replay_reference(iterative), iterative_face
        )
        self.assertIs(
            cumulative_prior_exact_reference(iterative, face), iterative_exact
        )

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
        self.assertEqual(audit["forbiddenPriorComponentDeletionCount"], 1)
        self.assertEqual(audit["orphanFinalComponentCount"], 0)

    def test_lineage_replay_allows_fully_replaced_provisional_component(
        self,
    ) -> None:
        audit = _complete_inheritance_audit(
            np.asarray((1, 1, 1, 0, 0), dtype=bool),
            np.asarray((0, 0, 1, -1, -1), dtype=np.int32),
            np.asarray((1, 1, 0, 1, 1), dtype=bool),
            np.asarray((0, 0, -1, 0, 0), dtype=np.int32),
            minimum_substantial_ribbon_count=2,
            source_interface=np.asarray((0, 2, 4, 4, 6), dtype=np.int32),
            target_interface=np.asarray((1, 3, 5, 7, 5), dtype=np.int32),
        )
        self.assertEqual(audit["deletedPriorComponentCount"], 1)
        self.assertEqual(audit["forbiddenPriorComponentDeletionCount"], 0)
        self.assertEqual(audit["allowedProvisionalReplacementCount"], 1)
        self.assertTrue(audit["deletedPriorComponentRecords"][0]["fullyReplaced"])

    def test_lineage_replay_rejects_partially_replaced_provisional_component(
        self,
    ) -> None:
        audit = _complete_inheritance_audit(
            np.asarray((1, 1, 1, 0), dtype=bool),
            np.asarray((0, 0, 1, -1), dtype=np.int32),
            np.asarray((1, 1, 0, 1), dtype=bool),
            np.asarray((0, 0, -1, 0), dtype=np.int32),
            minimum_substantial_ribbon_count=2,
            source_interface=np.asarray((0, 2, 4, 4), dtype=np.int32),
            target_interface=np.asarray((1, 3, 5, 7), dtype=np.int32),
        )
        self.assertEqual(audit["forbiddenPriorComponentDeletionCount"], 1)
        self.assertEqual(audit["allowedProvisionalReplacementCount"], 0)

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

    def test_complete_strip_replay_prioritizes_retained_dense_surface(self) -> None:
        variants = {
            "corridorVariantAddedOffset": np.asarray((0, 1, 2), dtype=np.int64),
            "corridorVariantAddedFrontierIndex": np.asarray((0, 1), dtype=np.int32),
            "corridorVariantRemovedOffset": np.asarray((0, 0, 0), dtype=np.int64),
            "corridorVariantRemovedFrontierIndex": np.empty(0, dtype=np.int32),
        }
        common = {
            "corridorRow": 5,
            "physicalPathFaceCount": 2,
            "triangleRegionCountBefore": 2,
            "triangleRegionCountAfter": 1,
            "localObjectiveDelta": 1.0,
        }
        candidates = [
            {
                **common,
                "variantIndex": 0,
                "physicalPathCost": 1.0,
                "strictAreaRetention": 0.96,
                "patchCoverage": 0.8,
                "augmentedAreaRetention": 1.1,
            },
            {
                **common,
                "variantIndex": 1,
                "physicalPathCost": 1.2,
                "strictAreaRetention": 0.98,
                "patchCoverage": 1.0,
                "augmentedAreaRetention": 1.05,
            },
        ]
        states, _ = _grouped_candidate_states(
            candidates,
            variants,
            valid_modifications=lambda _added, _removed: True,
            beam_width=8,
        )
        self.assertEqual(states[0]["variantIndices"], (1,))

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

    def test_assignment_beam_preserves_one_anchor_per_prior_component(self) -> None:
        selected = np.asarray((1, 1, 1, 1), dtype=bool)
        component = np.asarray((0, 0, 1, 2), dtype=np.int32)
        self.assertTrue(
            _preserves_prior_component_anchors(
                frozenset(),
                frozenset((0,)),
                baseline_selected=selected,
                baseline_component=component,
            )
        )
        self.assertFalse(
            _preserves_prior_component_anchors(
                frozenset(),
                frozenset((2,)),
                baseline_selected=selected,
                baseline_component=component,
            )
        )
        self.assertTrue(
            _preserves_prior_component_anchors(
                frozenset((2,)),
                frozenset((2,)),
                baseline_selected=selected,
                baseline_component=component,
            )
        )

    def test_new_corridor_catalog_does_not_reuse_prior_success_rows(self) -> None:
        prior_manifest = {
            "identity": {"corridors": {"dataSha256": "old"}},
            "data": {"sha256": "replay"},
        }
        current_manifest = {"data": {"sha256": "new"}}
        result = _successful_corridor_mask(
            Path("/tmp/prior.json"),
            prior_manifest,
            {},
            Path("/tmp/current.json"),
            current_manifest,
            3,
        )
        np.testing.assert_array_equal(result, (False, False, False))

    def test_same_corridor_catalog_preserves_prior_success_rows(self) -> None:
        prior_manifest = {
            "identity": {"corridors": {"dataSha256": "same"}},
            "data": {"sha256": "replay"},
        }
        current_manifest = {"data": {"sha256": "same"}}
        result = _successful_corridor_mask(
            Path("/tmp/prior.json"),
            prior_manifest,
            {"corridorReplayProposalSuccessful": np.asarray((1, 0, 1))},
            Path("/tmp/current.json"),
            current_manifest,
            3,
        )
        np.testing.assert_array_equal(result, (True, False, True))

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

    def test_exact_screen_array_fingerprint_is_order_independent(self) -> None:
        first = {
            "b": np.asarray((3.0, 4.0), dtype=np.float32),
            "a": np.asarray(((1, 2),), dtype=np.int32),
        }
        second = {"a": first["a"].copy(), "b": first["b"].copy()}
        self.assertEqual(
            _array_mapping_sha256(first), _array_mapping_sha256(second)
        )
        second["a"][0, 1] = 7
        self.assertNotEqual(
            _array_mapping_sha256(first), _array_mapping_sha256(second)
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
            "triangleFrontierIndex": np.asarray(
                ((0, 1, 2), (0, 3, 4)), dtype=np.int32
            ),
            "midpointXYZ": center,
            "signedNormalXYZ": np.tile((0, 0, 1), (5, 1)).astype(np.float32),
        }
        corridors = {
            "corridorPairOffset": np.asarray((0, 1), dtype=np.int64),
            "corridorFirstBoundaryEdge": np.asarray((0,), dtype=np.int32),
            "corridorSecondBoundaryEdge": np.asarray((1,), dtype=np.int32),
            "boundaryEdgeFirstFrontierIndex": np.asarray((1, 3), dtype=np.int32),
            "boundaryEdgeSecondFrontierIndex": np.asarray((2, 4), dtype=np.int32),
            "boundaryEdgeMidpointXYZ": np.asarray(
                ((-0.5, 0.5, 0), (0.5, -0.5, 0)), dtype=np.float32
            ),
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
