from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from backend.acus import (
    _refine_needle,
    _refine_needles_batch,
    fit_acus,
    fit_acus_audit,
    fit_acus_field,
    fit_acus_padding_audit,
)
from backend.rectify import VolumeData, fit_local_chart, grayscale_png, synthetic_scroll
from backend.region import fit_acus_region
from backend.slab_flake_audit import slab_flake_audit
from backend.slab_flake_holdout import _mutual_cell_matches, _stable_fold
from backend.slab_flakes import slab_flake_plane
from backend.slab_sheetlets import _match_3d
from backend.slab_sheetlet_explore import (
    _candidate_catalog,
    _components_without_cell_collisions,
    _direction_edge_links,
)
from backend.slab_sheetlet_carriers import (
    _carrier_yield,
    _mls_carrier,
    _plane_texture,
    _sample_stack,
    _texture_profile,
)
from backend.slab_carrier_assembly import _carrier_boundary, _score_point_pairs
from backend.slab_carrier_growth import _flake_arrays, _score_growth_candidates
from backend.slab_carrier_iteration import _merge_states
from backend.slab_carrier_bridges import (
    _bridge_endpoint_scores,
    _ct_bridge_evidence,
)
from backend.slab_carrier_gaps import (
    _axial_angle_difference,
    _internal_gap_components,
)
from backend.slab_gap_reanalysis import (
    _accepted_candidates,
    _cell_normal,
    _mask_covering_points,
)
from backend.slab_gap_census import _cropped_gap_context, _ct_gate
from backend.slab_normal_families import (
    NORMAL_FAMILY_DTYPE,
    _infer_cell_normal_families,
    _normal_family_assignment,
    _union_components,
)
from backend.slab_material_intervals import (
    LABEL_AIR,
    LABEL_CONTESTED_MATERIAL,
    LABEL_SINGLY_CLAIMED_MATERIAL,
    _annotate_profile,
    _smoothed_material_mask,
)
from backend.slab_monotone_layers import (
    _axial_monotone_match,
    _box_sums,
    _monotone_partial_match,
    _parity_consistent_links,
)
from backend.slab_branch_association import (
    CANDIDATE_DTYPE,
    DECISION_BELOW_THRESHOLD,
    DECISION_EXACT_GROUP_PRUNED,
    DECISION_EXACT_PAIR_DEFERRED as LOCAL_DECISION_EXACT_PAIR_DEFERRED,
    DECISION_ORDER_AMBIGUOUS,
    DECISION_ORDER_BLOCKED,
    DECISION_RETAINED,
    DEFAULT_SETTINGS,
    _audit_exact_candidate_pairs,
    _order_condensation,
    _score_endpoint_hits,
    _solve_candidates,
    _solve_with_exact_geometry,
)
from backend.slab_association_integrity import (
    _shared_cell_order,
    _triangle_intersection,
)
from backend.slab_window_reconciliation import (
    _cell_regions,
    _partition_overlap_stats,
)
from backend.slab_window_scheduler import (
    _aggregate_pairs,
    _axis_origins,
    _integrity_path,
    _integrity_quarantine_mask,
    _neighbor_pairs,
    _window_components,
)
from backend.slab_global_branch_association import (
    GRAPH_CELL_COLLISION,
    GRAPH_REDUNDANT,
    GRAPH_RETAINED,
    DECISION_EXACT_PAIR_DEFERRED,
    DECISION_INPUT_BRANCH_CARRIER_DEFERRED,
    DECISION_RETAINED as GLOBAL_DECISION_RETAINED,
    PROVENANCE_DIRECTIONAL_BOUNDARY,
    PROVENANCE_CONTEXT_DISPUTED,
    PROVENANCE_LOCAL_ORDER_RESOLVED_BOUNDARY,
    PROVENANCE_OVERLAP_VALIDATED,
    PROVENANCE_SINGLE_WINDOW,
    _aggregate_observations,
    _candidate_order,
    _candidate_tier_selection,
    _pair_gate_state,
    _solve_candidate_graph,
    _weakest_integrity_candidates,
    _weakest_retained_candidate,
)
from backend.slab_global_branch_candidates import (
    SOURCE_LOCAL_EXACT_DEFERRED,
    SOURCE_SUBWINDOW_UNRESOLVED,
    _candidate_evidence_source,
)
from backend.slab_global_boundary_candidates import (
    LOCAL_ORDER_BLOCKED,
    LOCAL_ORDER_CYCLIC,
    LOCAL_ORDER_FEASIBLE,
    LOCAL_ORDER_SAME_BRANCH,
    _boundary_geometry_arrays,
    _boundary_outward,
    _expanded_cell_pairs,
    _local_order_reconciliation,
    _score_boundary_pair_arrays,
    _stream_boundary_hits,
)
from backend.slab_fragment_termination_census import (
    CATEGORY_CONTINUED as TERMINATION_CONTINUED,
    CATEGORY_GEOMETRY_REJECTED as TERMINATION_GEOMETRY_REJECTED,
    CATEGORY_ORDER_UNRESOLVED as TERMINATION_ORDER_UNRESOLVED,
    CATEGORY_OVERLAP_UNRESOLVED as TERMINATION_OVERLAP_UNRESOLVED,
    _cluster_termination_regions,
    _global_candidate_category,
    _local_candidate_category,
)
from backend.slab_termination_reanalysis import (
    _classify_comparison as _classify_termination_reanalysis,
    _coalesce_target_crops,
    _resolve_open_targets,
)
from backend.slab_analysis import (
    CELL_DTYPE,
    NEEDLE_DTYPE,
    _macro_radial_fit,
    run_slab_analysis,
    slab_overview,
)


class RectifierTests(unittest.TestCase):
    def test_dense_termination_comparison_separates_new_and_stored_modes(
        self,
    ) -> None:
        failed = {"passed": False, "failureReasons": ["score"]}
        passed = {"passed": True, "failureReasons": []}
        self.assertEqual(
            _classify_termination_reanalysis(failed, passed, False),
            "recovered-new-dense-mode",
        )
        self.assertEqual(
            _classify_termination_reanalysis(failed, passed, True),
            "recovered-stored-cell-mode",
        )
        self.assertEqual(
            _classify_termination_reanalysis(passed, passed),
            "corroborated-coarse-evidence",
        )

    def test_dense_termination_targets_use_an_open_member_endpoint(self) -> None:
        resolved, skipped = _resolve_open_targets(
            [
                {
                    "clusterIndex": 7,
                    "associationId": 2,
                    "associationNodeCount": 30,
                    "endpointCount": 2,
                    "denseAcusPriority": 4.0,
                }
            ],
            np.asarray([7, 7], dtype=np.int32),
            np.asarray([0, 1], dtype=np.uint32),
            np.asarray([10, 11], dtype=np.uint64),
            np.asarray([[0.0, 0.0, 0.0], [32.0, 0.0, 0.0]]),
            np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.asarray([0.5, 0.8]),
            np.asarray([0.2, 0.3]),
            np.asarray([2, 2], dtype=np.uint32),
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
            (
                np.asarray([0.0, 32.0, 64.0]),
                np.asarray([0.0]),
                np.asarray([0.0]),
            ),
            32.0,
        )
        self.assertFalse(skipped)
        self.assertEqual(len(resolved), 1)
        self.assertEqual(resolved[0]["sourceEndpointIndex"], 1)
        self.assertEqual(resolved[0]["targetCellIndex"], (2, 0, 0))

    def test_dense_termination_crops_merge_only_bounded_overlap(self) -> None:
        crops = _coalesce_target_crops(
            np.asarray(
                [
                    [150.0, 150.0, 50.0],
                    [190.0, 150.0, 50.0],
                    [400.0, 400.0, 50.0],
                ]
            ),
            64,
            16,
            (500, 500, 100),
            4_000_000,
            32,
        )
        self.assertEqual(len(crops), 2)
        self.assertEqual(crops[0]["targetIndices"], [0, 1])
        self.assertEqual(crops[1]["targetIndices"], [2])
        self.assertLessEqual(crops[0]["voxelCount"], 4_000_000)
        self.assertEqual((int(crops[0]["lowXYZ"][0]) + 16) % 32, 0)

    def test_termination_evidence_preserves_failure_stage(self) -> None:
        self.assertEqual(
            _local_candidate_category(
                DECISION_ORDER_AMBIGUOUS,
                DECISION_BELOW_THRESHOLD,
                False,
            ),
            TERMINATION_ORDER_UNRESOLVED,
        )
        self.assertEqual(
            _local_candidate_category(
                DECISION_RETAINED,
                DECISION_BELOW_THRESHOLD,
                False,
            ),
            TERMINATION_OVERLAP_UNRESOLVED,
        )
        self.assertEqual(
            _global_candidate_category(DECISION_EXACT_PAIR_DEFERRED),
            TERMINATION_GEOMETRY_REJECTED,
        )
        self.assertEqual(
            _global_candidate_category(GLOBAL_DECISION_RETAINED),
            TERMINATION_CONTINUED,
        )

    def test_termination_clusters_require_association_and_direction_agreement(
        self,
    ) -> None:
        cluster = _cluster_termination_regions(
            np.asarray([4, 4, 4, 5, 4]),
            np.asarray(
                [[0, 0, 0], [1, 0, 0], [1, 1, 0], [1, 0, 0], [2, 0, 0]]
            ),
            np.asarray(
                [
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [-1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                    [1.0, 0.0, 0.0],
                ]
            ),
            np.asarray([True, True, True, True, False]),
            1,
            0.5,
        )
        self.assertEqual(int(cluster[0]), int(cluster[1]))
        self.assertNotEqual(int(cluster[1]), int(cluster[2]))
        self.assertNotEqual(int(cluster[1]), int(cluster[3]))
        self.assertEqual(int(cluster[4]), -1)

    def test_boundary_exposure_rejects_an_already_occupied_open_cone(self) -> None:
        usable, outward, concentration, neighbor_count, maximum_forward = (
            _boundary_outward(
                [{"normal": [0.0, 0.0, 1.0]}],
                np.asarray([0], dtype=np.uint32),
                np.asarray(
                    [
                        [0.0, 0.0, 0.0],
                        [1.0, 0.0, 0.0],
                        [-1.0, 0.0, 0.0],
                        [-2.0, 0.0, 0.0],
                        [-3.0, 0.0, 0.0],
                    ],
                    dtype=np.float32,
                ),
                np.asarray([0, 0, 0, 0], dtype=np.uint32),
                np.asarray([1, 2, 3, 4], dtype=np.uint32),
                np.ones(4, dtype=bool),
            )
        )
        self.assertTrue(bool(usable[0]))
        self.assertEqual(int(neighbor_count[0]), 4)
        self.assertAlmostEqual(float(concentration[0]), 0.5)
        np.testing.assert_allclose(outward[0], [1.0, 0.0, 0.0])
        self.assertAlmostEqual(float(maximum_forward[0]), 1.0)

    def test_streamed_boundary_pair_expansion_preserves_cell_cross_product(
        self,
    ) -> None:
        source, target = _expanded_cell_pairs(
            np.asarray([0, 1, 2, 3, 4], dtype=np.uint32),
            np.asarray([0, 2]),
            np.asarray([2, 3]),
            np.asarray([0]),
            np.asarray([1]),
        )
        np.testing.assert_array_equal(source, [0, 0, 0, 1, 1, 1])
        np.testing.assert_array_equal(target, [2, 3, 4, 2, 3, 4])

    def test_directional_boundary_candidates_cannot_displace_local_evidence(
        self,
    ) -> None:
        order = _candidate_order(
            np.asarray([0.9, 0.5]),
            np.asarray([0, 0]),
            np.asarray([1, 2]),
            np.asarray(
                [PROVENANCE_DIRECTIONAL_BOUNDARY, PROVENANCE_SINGLE_WINDOW]
            ),
        )
        np.testing.assert_array_equal(order, [1, 0])
        weakest = _weakest_retained_candidate(
            np.asarray([0, 1]),
            np.asarray([0.9, 0.5]),
            np.asarray([[10, 11], [12, 13]], dtype=np.uint64),
            np.asarray(
                [PROVENANCE_DIRECTIONAL_BOUNDARY, PROVENANCE_SINGLE_WINDOW]
            ),
        )
        self.assertEqual(weakest, 0)

    def test_locally_order_resolved_boundaries_are_the_weakest_tier(self) -> None:
        provenance = np.asarray(
            [
                PROVENANCE_LOCAL_ORDER_RESOLVED_BOUNDARY,
                PROVENANCE_DIRECTIONAL_BOUNDARY,
                PROVENANCE_SINGLE_WINDOW,
            ]
        )
        order = _candidate_order(
            np.asarray([0.99, 0.90, 0.50]),
            np.asarray([0, 0, 0]),
            np.asarray([1, 2, 3]),
            provenance,
        )
        np.testing.assert_array_equal(order, [2, 1, 0])
        weakest = _weakest_retained_candidate(
            np.asarray([0, 1, 2]),
            np.asarray([0.99, 0.90, 0.50]),
            np.asarray([[10, 11], [12, 13], [14, 15]], dtype=np.uint64),
            provenance,
        )
        self.assertEqual(weakest, 0)

    def test_local_order_recovery_requires_unanimous_overlap(self) -> None:
        reconciliation = _local_order_reconciliation(
            5,
            np.asarray([0, 0, 1, 1, 2, 3, 3, 4, 4], dtype=np.uint32),
            np.asarray(
                [
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_BLOCKED,
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_SAME_BRANCH,
                    LOCAL_ORDER_FEASIBLE,
                    LOCAL_ORDER_CYCLIC,
                ],
                dtype=np.uint8,
            ),
            2,
        )
        np.testing.assert_array_equal(
            reconciliation["resolved"], [True, False, False, True, False]
        )
        np.testing.assert_array_equal(
            reconciliation["observationCount"], [2, 2, 1, 2, 2]
        )
        self.assertEqual(int(reconciliation["blockedCount"][1]), 1)
        self.assertEqual(int(reconciliation["cyclicCount"][4]), 1)

    def test_streamed_boundary_search_scores_each_spatial_pair_once(self) -> None:
        flakes = [
            {
                "normal": [0.0, 0.0, 1.0],
                "fiber": [0.0, 1.0, 0.0],
                "crossFiber": [-1.0, 0.0, 0.0],
                "quality": 0.4,
                "normalFamily": 0,
                "radiusFiber": 20.0,
                "radiusCrossFiber": 20.0,
            }
            for _ in range(2)
        ]
        hits, stats = _stream_boundary_hits(
            np.asarray([[0, 0, 0], [1, 0, 0]], dtype=np.int32),
            np.asarray([0, 1], dtype=np.int32),
            np.asarray([[0.0, 0.0, 0.0], [20.0, 0.0, 0.5]], dtype=np.float32),
            np.asarray(
                [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
            ),
            _boundary_geometry_arrays(flakes),
            np.asarray([True, True]),
            np.asarray([False, False]),
            np.asarray([2, 1, 1]),
            3,
            128.0,
            0.1,
            8.0,
            0.08,
            None,
        )
        self.assertEqual(stats["rawSpatialPairCount"], 1)
        self.assertEqual(stats["spatialFacingBoundaryPairCount"], 1)
        self.assertEqual(len(hits), 1)

    def test_vectorized_global_boundary_score_matches_local_endpoint_score(
        self,
    ) -> None:
        flakes = [
            {
                "center": [0.0, 0.0, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "fiber": [0.0, 1.0, 0.0],
                "crossFiber": [-1.0, 0.0, 0.0],
                "quality": 0.4,
                "normalFamily": 0,
                "radiusFiber": 20.0,
                "radiusCrossFiber": 20.0,
            },
            {
                "center": [20.0, 0.0, 0.5],
                "normal": [0.0, 0.0, 1.0],
                "fiber": [0.0, 1.0, 0.0],
                "crossFiber": [-1.0, 0.0, 0.0],
                "quality": 0.4,
                "normalFamily": 0,
                "radiusFiber": 20.0,
                "radiusCrossFiber": 20.0,
            },
        ]
        centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
        outward = np.asarray(
            [[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32
        )
        branch = np.asarray([0, 1], dtype=np.int32)
        supported = np.asarray([True, True])
        contested = np.asarray([False, False])
        legacy = _score_endpoint_hits(
            flakes,
            [(0, 1, 0)],
            branch,
            centers,
            outward,
            supported,
            contested,
            8.0,
            0.08,
        )
        vectorized = _score_boundary_pair_arrays(
            np.asarray([0]),
            np.asarray([1]),
            np.asarray([0]),
            branch,
            centers,
            outward,
            _boundary_geometry_arrays(flakes),
            supported,
            contested,
            8.0,
            0.08,
        )
        self.assertEqual(len(legacy), 1)
        self.assertEqual(len(vectorized), 1)
        for field in (
            "score",
            "geometryScore",
            "facing",
            "distanceVoxels",
            "edgeResidualVoxels",
            "fiberAngleDeg",
            "normalBendDeg",
            "reachRatio",
        ):
            self.assertAlmostEqual(
                float(vectorized[field][0]), float(legacy[field][0]), places=5
            )

    def test_global_rescue_candidate_source_preserves_local_deferral(self) -> None:
        source, selected = _candidate_evidence_source(
            np.asarray([DECISION_RETAINED, 0]),
            np.asarray([0, 0]),
        )
        self.assertEqual(source, SOURCE_SUBWINDOW_UNRESOLVED)
        np.testing.assert_array_equal(selected, [True, False])

        source, selected = _candidate_evidence_source(
            np.asarray([DECISION_RETAINED, DECISION_RETAINED]),
            np.asarray([LOCAL_DECISION_EXACT_PAIR_DEFERRED, 0]),
        )
        self.assertEqual(source, SOURCE_LOCAL_EXACT_DEFERRED)
        np.testing.assert_array_equal(selected, [True, False])

        source, selected = _candidate_evidence_source(
            np.asarray([DECISION_RETAINED]),
            np.asarray([DECISION_RETAINED]),
        )
        self.assertEqual(source, 0)
        np.testing.assert_array_equal(selected, [False])

    def test_global_pair_gate_lets_joined_carrier_stabilize_sparse_input(
        self,
    ) -> None:
        eligible, decisions = _pair_gate_state(
            np.asarray([True, False, False]),
            np.asarray([False, False, True]),
        )
        np.testing.assert_array_equal(eligible, [True, False, False])
        self.assertEqual(int(decisions[0]), 0)
        self.assertEqual(
            int(decisions[1]), DECISION_INPUT_BRANCH_CARRIER_DEFERRED
        )
        self.assertEqual(int(decisions[2]), DECISION_EXACT_PAIR_DEFERRED)

    def test_global_branch_join_tiers_preserve_weaker_provenance(self) -> None:
        selected, provenance = _candidate_tier_selection(
            np.asarray([True, True, False, True, False]),
            np.asarray([True, False, True, True, True]),
            np.asarray([2, 1, 2, 1, 2]),
            np.asarray([2, 1, 1, 1, 0]),
            True,
            True,
            2,
        )
        np.testing.assert_array_equal(
            selected, [True, True, True, False, False]
        )
        np.testing.assert_array_equal(
            provenance,
            [
                PROVENANCE_OVERLAP_VALIDATED,
                PROVENANCE_SINGLE_WINDOW,
                PROVENANCE_CONTEXT_DISPUTED,
            ],
        )

    def test_global_branch_join_tiers_can_exclude_disputed_context(self) -> None:
        selected, provenance = _candidate_tier_selection(
            np.asarray([True, True, False]),
            np.asarray([True, False, True]),
            np.asarray([2, 1, 2]),
            np.asarray([2, 1, 1]),
            True,
            False,
            2,
        )
        np.testing.assert_array_equal(selected, [True, True, False])
        np.testing.assert_array_equal(
            provenance,
            [PROVENANCE_OVERLAP_VALIDATED, PROVENANCE_SINGLE_WINDOW],
        )

    def test_global_branch_join_aggregation_uses_conservative_score(self) -> None:
        aggregate = _aggregate_observations(
            {
                "candidateIndex": np.asarray([0, 0, 1], dtype=np.uint32),
                "score": np.asarray([0.7, 0.5, 0.8], dtype=np.float32),
                "medianHeightResidualVoxels": np.asarray(
                    [1.0, 2.0, 0.5], dtype=np.float32
                ),
                "medianNormalResidualDeg": np.asarray(
                    [3.0, 5.0, 2.0], dtype=np.float32
                ),
            },
            2,
        )
        np.testing.assert_array_equal(aggregate["count"], [2, 1])
        np.testing.assert_allclose(aggregate["minimumScore"], [0.5, 0.8])
        np.testing.assert_allclose(aggregate["meanScore"], [0.6, 0.8])
        np.testing.assert_allclose(
            aggregate["maximumLocalMedianHeightResidualVoxels"], [2.0, 0.5]
        )
        np.testing.assert_allclose(
            aggregate["maximumLocalMedianNormalResidualDeg"], [5.0, 2.0]
        )

    def test_global_branch_join_aggregation_ignores_missing_local_geometry(
        self,
    ) -> None:
        aggregate = _aggregate_observations(
            {
                "candidateIndex": np.asarray([0], dtype=np.uint32),
                "score": np.asarray([0.6], dtype=np.float32),
                "medianHeightResidualVoxels": np.asarray(
                    [np.nan], dtype=np.float32
                ),
                "medianNormalResidualDeg": np.asarray(
                    [np.nan], dtype=np.float32
                ),
            },
            1,
        )
        self.assertTrue(
            np.isneginf(aggregate["maximumLocalMedianHeightResidualVoxels"][0])
        )
        self.assertTrue(
            np.isneginf(aggregate["maximumLocalMedianNormalResidualDeg"][0])
        )

    def test_global_branch_join_solver_rejects_transitive_cell_collision(
        self,
    ) -> None:
        result = _solve_candidate_graph(
            3,
            np.asarray([0, 1, 0]),
            np.asarray([1, 2, 2]),
            np.asarray([0.9, 0.8, 0.7]),
            np.ones(3, dtype=bool),
            {0: {10}, 1: {11}, 2: {10}},
        )
        np.testing.assert_array_equal(
            result["decisions"],
            [GRAPH_RETAINED, GRAPH_CELL_COLLISION, GRAPH_CELL_COLLISION],
        )
        self.assertEqual(
            result["branchAssociation"][0], result["branchAssociation"][1]
        )
        self.assertNotEqual(
            result["branchAssociation"][0], result["branchAssociation"][2]
        )

    def test_global_branch_join_solver_marks_redundant_cycle_support(self) -> None:
        result = _solve_candidate_graph(
            3,
            np.asarray([0, 1, 0]),
            np.asarray([1, 2, 2]),
            np.asarray([0.9, 0.8, 0.7]),
            np.ones(3, dtype=bool),
            {0: {10}, 1: {11}, 2: {12}},
        )
        np.testing.assert_array_equal(
            result["decisions"],
            [GRAPH_RETAINED, GRAPH_RETAINED, GRAPH_REDUNDANT],
        )
        self.assertEqual(len(np.unique(result["branchAssociation"])), 1)

    def test_global_branch_join_score_tie_prefers_overlap_provenance(self) -> None:
        result = _solve_candidate_graph(
            3,
            np.asarray([0, 1]),
            np.asarray([1, 2]),
            np.asarray([0.7, 0.7]),
            np.ones(2, dtype=bool),
            {0: {10}, 1: {11}, 2: {10}},
            np.asarray(
                [PROVENANCE_SINGLE_WINDOW, PROVENANCE_OVERLAP_VALIDATED]
            ),
        )
        np.testing.assert_array_equal(
            result["decisions"], [GRAPH_CELL_COLLISION, GRAPH_RETAINED]
        )

    def test_global_integrity_pruning_removes_only_weakest_construction_edge(
        self,
    ) -> None:
        selected = _weakest_integrity_candidates(
            [
                {
                    "associationSource": 0,
                    "associationTarget": 2,
                }
            ],
            np.asarray([0, 0, 2, 2]),
            np.asarray([[0, 1], [2, 3]]),
            np.asarray([GRAPH_RETAINED, GRAPH_RETAINED]),
            np.ones(2, dtype=bool),
            np.asarray([0.61, 0.54]),
            np.asarray([[10, 11], [20, 21]], dtype=np.uint64),
            np.asarray(
                [PROVENANCE_CONTEXT_DISPUTED, PROVENANCE_CONTEXT_DISPUTED]
            ),
        )
        np.testing.assert_array_equal(selected, [1])

    def test_global_integrity_pruning_prefers_weaker_provenance_on_score_tie(
        self,
    ) -> None:
        selected = _weakest_integrity_candidates(
            [{"associationSource": 0, "associationTarget": 2}],
            np.asarray([0, 0, 2, 2]),
            np.asarray([[0, 1], [2, 3]]),
            np.asarray([GRAPH_RETAINED, GRAPH_RETAINED]),
            np.ones(2, dtype=bool),
            np.asarray([0.54, 0.54]),
            np.asarray([[10, 11], [20, 21]], dtype=np.uint64),
            np.asarray(
                [PROVENANCE_SINGLE_WINDOW, PROVENANCE_CONTEXT_DISPUTED]
            ),
        )
        np.testing.assert_array_equal(selected, [1])

    def test_window_scheduler_end_aligns_axis_coverage(self) -> None:
        origins = _axis_origins(242, 32, 24)
        self.assertEqual(origins, [0, 24, 48, 72, 96, 120, 144, 168, 192, 210])
        covered = np.zeros(242, dtype=bool)
        for origin in origins:
            covered[origin : origin + 32] = True
        self.assertTrue(np.all(covered))

    def test_window_scheduler_only_reconciles_face_neighbors(self) -> None:
        windows = [
            {
                "originCellXYZ": origin,
                "stopCellXYZExclusive": [value + 32 for value in origin],
            }
            for origin in ([0, 0, 0], [24, 0, 0], [0, 24, 0], [24, 24, 0])
        ]
        pairs = _neighbor_pairs(windows)
        self.assertEqual(len(pairs), 4)
        self.assertNotIn(([0, 0, 0], [24, 24, 0]), pairs)

    def test_window_scheduler_defers_nonunanimous_overlap_pairs(self) -> None:
        node_identity = np.asarray([10, 20, 30], dtype=np.uint64)
        node_window_mask = np.asarray([3, 3, 1], dtype=np.uint64)
        pairs = _aggregate_pairs(
            [
                np.asarray([[10, 20], [10, 30]], dtype=np.uint64),
                np.empty((0, 2), dtype=np.uint64),
            ],
            node_identity,
            node_window_mask,
        )
        np.testing.assert_array_equal(pairs["observationCount"], [2, 1])
        np.testing.assert_array_equal(pairs["acceptanceCount"], [1, 1])
        np.testing.assert_array_equal(pairs["unanimous"], [False, True])
        np.testing.assert_array_equal(pairs["overlapValidated"], [True, False])

    def test_window_scheduler_defers_relative_parity_disagreement(self) -> None:
        pairs = _aggregate_pairs(
            [
                np.asarray([[10, 20]], dtype=np.uint64),
                np.asarray([[10, 20]], dtype=np.uint64),
            ],
            np.asarray([10, 20], dtype=np.uint64),
            np.asarray([3, 3], dtype=np.uint64),
            [
                np.asarray([1], dtype=np.int8),
                np.asarray([-1], dtype=np.int8),
            ],
        )
        np.testing.assert_array_equal(pairs["consensusParity"], [0])
        np.testing.assert_array_equal(pairs["parityUnanimous"], [False])

    def test_window_scheduler_quarantines_intersecting_association_joins(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            origin = [24, 48, 0]
            np.savez_compressed(
                _integrity_path(root, origin),
                associationSource=np.asarray([7], dtype=np.uint32),
                associationTarget=np.asarray([9], dtype=np.uint32),
                intersectingTrianglePairCount=np.asarray([3], dtype=np.uint32),
                evidenceCoreIntersectingTrianglePairCount=np.asarray(
                    [1], dtype=np.uint32
                ),
            )
            quarantine, stats = _integrity_quarantine_mask(
                root,
                origin,
                {"component": np.asarray([0, 1, 2, 3], dtype=np.uint32)},
                {
                    "candidateNodeSource": np.asarray([0, 2], dtype=np.uint32),
                    "candidateNodeTarget": np.asarray([1, 3], dtype=np.uint32),
                    "branchAssociation": np.asarray([7, 7, 8, 8], dtype=np.uint32),
                },
                np.asarray([True, True]),
            )
        np.testing.assert_array_equal(quarantine, [True, False])
        self.assertEqual(stats["violatingAssociationCount"], 2)
        self.assertEqual(stats["evidenceCoreViolatingAssociationCount"], 2)
        self.assertEqual(stats["quarantinedJoinCount"], 1)

    def test_window_scheduler_connects_reconciled_windows(self) -> None:
        windows = [
            {"originCellXYZ": [0, 0, 0]},
            {"originCellXYZ": [24, 0, 0]},
            {"originCellXYZ": [48, 0, 0]},
        ]
        reconciliations = [
            {
                "sourceWindow": {"originCellXYZ": [0, 0, 0]},
                "targetWindow": {"originCellXYZ": [24, 0, 0]},
                "stats": {"sharedNodeCount": 100},
            },
            {
                "sourceWindow": {"originCellXYZ": [24, 0, 0]},
                "targetWindow": {"originCellXYZ": [48, 0, 0]},
                "stats": {"sharedNodeCount": 80},
            },
        ]
        component = _window_components(windows, reconciliations)
        np.testing.assert_array_equal(component, [0, 0, 0])

    def test_material_mask_closes_only_the_declared_air_gap(self) -> None:
        intensity = np.asarray([0, 20, 0, 20, 0, 0], dtype=np.uint8)
        material = _smoothed_material_mask(
            intensity,
            10.0,
            smoothing_kernel=[1.0],
            maximum_bridged_air_gap_samples=1,
        )
        np.testing.assert_array_equal(
            material,
            np.asarray([False, True, True, True, False, False]),
        )

    def test_material_claim_overlay_separates_assignment_from_contact(self) -> None:
        depth = np.arange(-4.0, 5.0, dtype=np.float32)
        material = np.asarray(
            [False, True, True, True, False, True, True, True, False]
        )
        annotated = _annotate_profile(
            material,
            depth,
            np.asarray([-2.0, 1.0, 3.0, 12.0], dtype=np.float32),
            claim_support_tolerance=1.0,
            claim_cluster_gap=1.0,
        )
        labels = annotated["labels"]
        self.assertEqual(int(labels[0]), LABEL_AIR)
        self.assertTrue(
            np.all(labels[1:4] == LABEL_SINGLY_CLAIMED_MATERIAL)
        )
        self.assertTrue(np.all(labels[5:8] == LABEL_CONTESTED_MATERIAL))
        self.assertEqual(
            annotated["claimIntervalIndex"].tolist(), [0, 1, 1, -1]
        )
        self.assertEqual(
            annotated["claimClusterIndex"].tolist(), [0, 0, 1, -1]
        )
        self.assertEqual(
            annotated["intervals"][0]["apparentCtThicknessVoxels"], 3.0
        )
        self.assertTrue(
            np.isnan(
                annotated["intervals"][1]["apparentCtThicknessVoxels"]
            )
        )

    def test_boundary_truncated_material_has_no_apparent_thickness(self) -> None:
        annotated = _annotate_profile(
            np.asarray([True, True, False, False]),
            np.arange(4, dtype=np.float32),
            np.asarray([0.5], dtype=np.float32),
        )
        self.assertTrue(annotated["intervals"][0]["boundaryTruncated"])
        self.assertTrue(
            np.isnan(
                annotated["intervals"][0]["apparentCtThicknessVoxels"]
            )
        )

    def test_monotone_partial_match_allows_birth_without_crossing(self) -> None:
        compatibility = {
            (0, 10): (0.82,),
            (0, 11): (0.95,),
            (1, 10): (0.95,),
            (1, 11): (0.82,),
            (2, 11): (0.84,),
        }
        matched = _monotone_partial_match(
            [0, 1, 2], [10, 11], compatibility, 0.60, 0.02
        )
        self.assertEqual(matched, [(1, 10), (2, 11)])

    def test_axial_monotone_match_is_invariant_to_depth_axis_flip(self) -> None:
        compatibility = {
            (0, 10): (0.9,),
            (0, 11): (0.65,),
            (1, 10): (0.65,),
            (1, 11): (0.9,),
        }
        original = _axial_monotone_match(
            [0, 1], [10, 11], compatibility, 0.6, 0.02, 1
        )
        flipped = _axial_monotone_match(
            [0, 1], [11, 10], compatibility, 0.6, 0.02, -1
        )
        self.assertEqual(original["matches"], [(0, 10), (1, 11)])
        self.assertEqual(flipped["matches"], original["matches"])
        self.assertEqual(original["relativeParity"], 1)
        self.assertEqual(flipped["relativeParity"], -1)

    def test_axial_monotone_tie_defers_orientation_specific_links(self) -> None:
        compatibility = {
            (0, 10): (0.9,),
            (0, 11): (0.9,),
        }
        matched = _axial_monotone_match(
            [0], [10, 11], compatibility, 0.6, 0.02, 1
        )
        self.assertEqual(matched["decision"], "geometry-tie")
        self.assertEqual(matched["matches"], [])
        self.assertEqual(matched["deferredAlternativeLinkCount"], 2)

    def test_parity_cycle_rejects_weakest_inconsistent_link(self) -> None:
        retained, node_parity, frustrated, stats = _parity_consistent_links(
            3,
            np.asarray([0, 1, 0], dtype=np.uint32),
            np.asarray([1, 2, 2], dtype=np.uint32),
            np.asarray([1, 1, -1], dtype=np.int8),
            np.asarray([0.9, 0.8, 0.7], dtype=np.float32),
            np.ones(3, dtype=bool),
        )
        np.testing.assert_array_equal(retained, [True, True, False])
        np.testing.assert_array_equal(frustrated, [False, False, True])
        np.testing.assert_array_equal(node_parity, [1, 1, 1])
        self.assertEqual(stats["frustratedCycleLinkCount"], 1)

    def test_dense_window_box_sum_uses_exact_extents(self) -> None:
        values = np.zeros((2, 4, 5), dtype=np.uint8)
        values[:, 1:3, 2:5] = 1
        sums = _box_sums(values, (2, 2, 3))
        self.assertEqual(sums.shape, (1, 3, 3))
        self.assertEqual(int(np.max(sums)), 12)

    def test_branch_association_defers_transitive_layer_order(self) -> None:
        cell = np.asarray(
            [[0, 0, 0], [0, 0, 0], [1, 0, 0], [1, 0, 0]],
            dtype=np.int32,
        )
        depth = np.asarray([0.0, 1.0, 0.0, 1.0], dtype=np.float32)
        branch = np.asarray([0, 1, 1, 2], dtype=np.int32)
        order = _order_condensation(cell, depth, branch)
        self.assertEqual(order["stats"]["sccCount"], 3)
        self.assertEqual(order["stats"]["cyclicSccCount"], 0)

        candidates = np.zeros(1, dtype=CANDIDATE_DTYPE)
        candidates["branchSource"] = 0
        candidates["branchTarget"] = 2
        candidates["score"] = 0.9
        candidates["endpointSupportedCount"] = 2
        solved = _solve_candidates(
            candidates, 0.45, order, [{1}, {2}, {3}]
        )
        self.assertEqual(
            int(solved["decisions"][0]), DECISION_ORDER_BLOCKED
        )

    def test_branch_order_cycles_remain_explicit_ambiguity(self) -> None:
        cell = np.asarray(
            [
                [0, 0, 0],
                [0, 0, 0],
                [1, 0, 0],
                [1, 0, 0],
                [2, 0, 0],
                [2, 0, 0],
            ],
            dtype=np.int32,
        )
        depth = np.asarray([0, 1, 0, 1, 0, 1], dtype=np.float32)
        branch = np.asarray([0, 1, 1, 2, 2, 0], dtype=np.int32)
        order = _order_condensation(cell, depth, branch)
        self.assertEqual(order["stats"]["cyclicSccCount"], 1)
        self.assertEqual(order["stats"]["largestCyclicSccSize"], 3)
        solved = _solve_candidates(
            np.empty(0, dtype=CANDIDATE_DTYPE),
            0.45,
            order,
            [{1}, {2}, {3}],
        )
        self.assertEqual(len(np.unique(solved["branchGroup"])), 3)

    def test_exact_pair_gate_defers_separated_parallel_surfaces(self) -> None:
        flakes = []
        branch = []
        for branch_index, height in enumerate((0.0, 10.0)):
            for x in (0.0, 16.0):
                flakes.append(
                    {
                        "center": [x, 0.0, height],
                        "normal": [0.0, 0.0, 1.0],
                        "fiber": [1.0, 0.0, 0.0],
                        "quality": 0.4,
                    }
                )
                branch.append(branch_index)
        candidates = np.zeros(1, dtype=CANDIDATE_DTYPE)
        candidates["branchSource"] = 0
        candidates["branchTarget"] = 1
        audit = _audit_exact_candidate_pairs(
            candidates,
            np.ones(1, dtype=bool),
            flakes,
            np.asarray(branch, dtype=np.int32),
            DEFAULT_SETTINGS,
        )
        self.assertGreater(float(audit["medianHeightResidualVoxels"][0]), 3.0)
        self.assertFalse(bool(audit["passed"][0]))

    def test_exact_group_pruning_removes_weakest_transitive_join(self) -> None:
        flakes = []
        branch = []
        for branch_index in range(3):
            for x in (float(branch_index * 16), float(branch_index * 16 + 8)):
                flakes.append(
                    {
                        "center": [x, 0.0, 0.0],
                        "normal": [0.0, 0.0, 1.0],
                        "fiber": [1.0, 0.0, 0.0],
                        "quality": 0.4,
                    }
                )
                branch.append(branch_index)
        candidates = np.zeros(2, dtype=CANDIDATE_DTYPE)
        candidates["branchSource"] = [0, 1]
        candidates["branchTarget"] = [1, 2]
        candidates["score"] = [0.9, 0.8]
        candidates["endpointSupportedCount"] = 2
        order = {
            "branchScc": np.arange(3, dtype=np.int32),
            "sccSizes": np.ones(3, dtype=np.uint32),
            "dag": [[], [], []],
        }

        def group_audit(branch_group, *_args):
            counts = np.bincount(branch_group)
            return [
                {
                    "associationId": int(group_index),
                    "branchCount": int(counts[group_index]),
                    "flakeCount": 2 * int(counts[group_index]),
                    "medianHeightResidualVoxels": 0.0,
                    "p90HeightResidualVoxels": 0.0,
                    "medianNormalResidualDeg": 0.0,
                    "p90NormalResidualDeg": 0.0,
                    "gatesPass": bool(counts[group_index] <= 2),
                }
                for group_index in np.flatnonzero(counts >= 2)
            ]

        with patch(
            "backend.slab_branch_association._audit_exact_association_groups",
            side_effect=group_audit,
        ):
            solved = _solve_with_exact_geometry(
                candidates,
                np.ones(2, dtype=bool),
                flakes,
                np.asarray(branch, dtype=np.int32),
                0.45,
                order,
                [{0}, {1}, {2}],
                DEFAULT_SETTINGS,
            )
        self.assertEqual(int(solved["decisions"][0]), DECISION_RETAINED)
        self.assertEqual(
            int(solved["decisions"][1]), DECISION_EXACT_GROUP_PRUNED
        )
        self.assertEqual(solved["stats"]["exactPruningRoundCount"], 1)
        self.assertEqual(solved["stats"]["finalExactFailureCount"], 0)

    def test_association_integrity_detects_crossing_not_parallel_separation(self) -> None:
        horizontal = np.asarray(
            [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0]]
        )
        crossing = np.asarray(
            [[0.5, -1.0, -1.0], [0.5, 2.0, -1.0], [0.5, 0.0, 1.0]]
        )
        point, coplanar = _triangle_intersection(horizontal, crossing, 0.05)
        self.assertIsNotNone(point)
        self.assertFalse(coplanar)

        separated = horizontal + np.asarray([0.0, 0.0, 1.0])
        point, _ = _triangle_intersection(horizontal, separated, 0.05)
        self.assertIsNone(point)

        overlapping = horizontal + np.asarray([0.25, 0.25, 0.0])
        point, coplanar = _triangle_intersection(horizontal, overlapping, 0.05)
        self.assertIsNotNone(point)
        self.assertTrue(coplanar)

        disjoint = horizontal + np.asarray([4.0, 0.0, 0.0])
        point, _ = _triangle_intersection(horizontal, disjoint, 0.05)
        self.assertIsNone(point)

    def test_association_integrity_exposes_shared_cell_order_inversion(self) -> None:
        first = {
            "cellDepth": {
                (0, 0, 0): [0.0],
                (1, 0, 0): [2.0],
            }
        }
        second = {
            "cellDepth": {
                (0, 0, 0): [1.0],
                (1, 0, 0): [1.0],
            }
        }
        order = _shared_cell_order(first, second, 0.5)
        self.assertEqual(order["sharedCellCount"], 2)
        self.assertEqual(order["negativeOrderCount"], 1)
        self.assertEqual(order["positiveOrderCount"], 1)
        self.assertTrue(order["orderInversion"])

    def test_window_partition_reconciliation_counts_context_splits(self) -> None:
        stats = _partition_overlap_stats(
            np.asarray([0, 0, 1, 1], dtype=np.uint32),
            np.asarray([0, 0, 0, 1], dtype=np.uint32),
        )
        self.assertEqual(stats["sourceCoassignedPairCount"], 2)
        self.assertEqual(stats["targetCoassignedPairCount"], 3)
        self.assertEqual(stats["jointCoassignedPairCount"], 1)
        self.assertEqual(stats["coassignmentUnionPairCount"], 4)
        self.assertEqual(stats["coassignmentDisagreementPairCount"], 3)
        self.assertEqual(stats["sourceSplitGroupCount"], 1)
        self.assertEqual(stats["targetSplitGroupCount"], 1)
        self.assertEqual(stats["nodeCountInContextSplit"], 4)

        regions = _cell_regions(
            np.asarray(
                [[0, 0, 0], [1, 0, 0], [4, 3, 2]], dtype=np.int32
            )
        )
        self.assertEqual([value["cellCount"] for value in regions], [2, 1])
        self.assertEqual(regions[0]["originCellXYZ"], [0, 0, 0])
        self.assertEqual(regions[0]["stopCellXYZExclusive"], [2, 1, 1])

    def test_secondary_normal_family_is_standalone_partitioned_and_spatial(self) -> None:
        rng = np.random.default_rng(7)
        directions = []
        for index in range(100):
            base = (
                np.asarray([1.0, 0.0, 0.0])
                if index % 2 == 0
                else np.asarray([0.0, 1.0, 0.0])
            )
            direction = base + rng.normal(0.0, 0.025, 3)
            directions.append(direction / np.linalg.norm(direction))
        angle = np.radians(32.0)
        for index in range(60):
            base = (
                np.asarray([np.cos(angle), 0.0, np.sin(angle)])
                if index % 2 == 0
                else np.asarray([-np.sin(angle), 0.0, np.cos(angle)])
            )
            direction = base + rng.normal(0.0, 0.025, 3)
            directions.append(direction / np.linalg.norm(direction))
        records = np.zeros(len(directions), dtype=NEEDLE_DTYPE)
        records["direction"] = directions
        records["score"] = 1.0
        records["axialCoverage"] = 1.0
        records["supportScore"] = 1.0
        inferred = _infer_cell_normal_families(
            records, np.asarray([0.0, 0.0, 1.0], dtype=np.float32), 0.8
        )
        self.assertTrue(inferred["secondaryFitted"])
        self.assertTrue(inferred["standaloneCandidate"])
        self.assertGreater(inferred["secondaryStandaloneConfidence"], 0.8)
        self.assertGreater(inferred["secondaryCoverage"], 0.3)
        self.assertGreater(inferred["ambiguousFraction"], 0.2)
        self.assertGreater(inferred["normalAngleDeg"], 85.0)

        primary, secondary, ambiguous, overlap, unassigned = _normal_family_assignment(
            records["direction"],
            inferred["primaryNormal"],
            inferred["secondaryNormal"],
            inferred["primaryInlierLimitDeg"],
            inferred["secondaryInlierLimitDeg"],
            3.0,
        )
        self.assertFalse(np.any(primary & secondary))
        self.assertFalse(np.any(secondary & ambiguous))
        self.assertTrue(np.all(ambiguous <= overlap))
        self.assertEqual(
            int(np.count_nonzero(primary | secondary | unassigned)),
            len(records),
        )

        families = np.zeros((1, 3, 3), dtype=NORMAL_FAMILY_DTYPE)
        families["componentId"] = -1
        for y, x in ((0, 0), (0, 1), (0, 2), (2, 2)):
            families[0, y, x]["standaloneCandidate"] = 1
            families[0, y, x]["secondaryNormal"] = [0.0, 1.0, 0.0]
        component_count, edge_count = _union_components(families, 12.0)
        self.assertEqual(component_count, 2)
        self.assertEqual(edge_count, 2)
        self.assertEqual(int(families[0, 0, 0]["componentSize"]), 3)
        self.assertEqual(int(families[0, 2, 2]["componentSize"]), 1)

    def test_gap_census_gate_and_block_aligned_crop(self) -> None:
        evidence = {
            "depthAlignedTextureScore": 0.55,
            "materialFraction": 0.9,
            "bestDepthOffsetVoxels": -2.0,
            "fiberAngleResidualDeg": 3.0,
        }
        gate = _ct_gate(evidence, 0.5, 0.35, 4.0, 12.0)
        self.assertTrue(gate["queuedForDenseAcus"])
        self.assertAlmostEqual(
            gate["thresholdSlack"]["depthAlignedTextureScore"], 0.05
        )
        evidence["bestDepthOffsetVoxels"] = -5.0
        gate = _ct_gate(evidence, 0.5, 0.35, 4.0, 12.0)
        self.assertFalse(gate["queuedForDenseAcus"])
        self.assertIn("depth", gate["rejectionReasons"])

        support = np.ones((30, 30), dtype=bool)
        gap_mask = np.zeros_like(support)
        gap_mask[5:14, 9:20] = True
        carrier = {
            "surfaceXYZ": np.zeros((30, 30, 3), dtype=np.float32),
            "normalXYZ": np.zeros((30, 30, 3), dtype=np.float32),
            "fiberXYZ": np.zeros((30, 30, 3), dtype=np.float32),
            "supportMask": support,
            "frame": {},
        }
        cropped, cropped_gap = _cropped_gap_context(
            carrier, {"bboxYX": [5, 14, 9, 20], "mask": gap_mask, "gapId": 1}
        )
        self.assertEqual(cropped["supportMask"].shape, (16, 16))
        self.assertEqual(
            int(np.count_nonzero(cropped_gap["mask"])),
            int(np.count_nonzero(gap_mask)),
        )

    def test_targeted_gap_acceptance_requires_ct_and_global_ownership(self) -> None:
        candidate = {
            "targetRank": 1,
            "targetGapId": 1,
            "sampleId": 0,
            "center": [4.0, 5.0, 6.0],
            "_needleIds": {1, 2, 3},
            "depthAlignedTextureScore": 0.49,
        }
        scores = np.asarray([[0.56], [0.0]], dtype=np.float32)
        metrics = {
            "heightResidual": np.asarray([3.7], dtype=np.float32),
            "normalAngle": np.asarray([5.5], dtype=np.float32),
            "fiberAngle": np.asarray([2.0], dtype=np.float32),
            "nearestPlanarDistance": np.asarray([20.0], dtype=np.float32),
        }
        accepted, diagnostics = _accepted_candidates(
            [candidate], scores, metrics, 0.55, 0.04, 0.04, 0.5, 8.0
        )
        self.assertEqual(accepted, [])
        self.assertIn("ct-evidence", diagnostics[0]["rejectionReasons"])
        self.assertAlmostEqual(
            diagnostics[0]["thresholdSlack"]["depthAlignedTextureScore"], -0.01
        )
        candidate["depthAlignedTextureScore"] = 0.51
        accepted, diagnostics = _accepted_candidates(
            [candidate], scores, metrics, 0.55, 0.04, 0.04, 0.5, 8.0
        )
        self.assertEqual(accepted, [0])
        self.assertTrue(diagnostics[0]["accepted"])

    def test_gap_covering_and_independent_cell_normal(self) -> None:
        mask = np.zeros((28, 32), dtype=bool)
        mask[3:25, 4:28] = True
        mask[12:25, 16:28] = False
        covering = _mask_covering_points(mask, spacing_pixels=4.0)
        self.assertGreater(len(covering), 1)
        pixels = np.argwhere(mask)
        distance2 = np.sum(
            (pixels[:, None, :] - covering[None, :, :]) ** 2, axis=2
        )
        self.assertLessEqual(float(np.sqrt(np.max(np.min(distance2, axis=1)))), 4.0)

        records = np.zeros(16, dtype=NEEDLE_DTYPE)
        angles = np.linspace(0.0, 2.0 * np.pi, len(records), endpoint=False)
        records["direction"][:, 0] = np.cos(angles)
        records["direction"][:, 1] = np.sin(angles)
        records["score"] = 1.0
        normal, confidence, stats = _cell_normal(records)
        self.assertIsNotNone(normal)
        assert normal is not None
        self.assertGreater(abs(float(normal[2])), 0.99)
        self.assertGreater(confidence, 0.9)
        self.assertLess(stats["medianPlaneResidualDeg"], 0.01)

    def test_carrier_gap_components_exclude_open_boundaries(self) -> None:
        mask = np.zeros((24, 28), dtype=bool)
        mask[2:22, 3:25] = True
        mask[8:14, 10:18] = False
        mask[2:12, 20:25] = False
        gaps = _internal_gap_components(
            mask, pixel_step=2.0, minimum_area_square_voxels=32.0
        )
        self.assertEqual(len(gaps), 1)
        self.assertEqual(gaps[0]["pixelCount"], 48)
        self.assertEqual(gaps[0]["areaSquareVoxels"], 192.0)
        self.assertEqual(gaps[0]["bboxYX"], [8, 14, 10, 18])
        self.assertAlmostEqual(_axial_angle_difference(178.0, 2.0), 4.0)

    def test_vectorized_needle_refinement_matches_scalar_path(self) -> None:
        z, y, _ = np.indices((32, 32, 32), dtype=np.float32)
        score = (
            0.8 * np.exp(-((z - 16.0) ** 2 + (y - 16.0) ** 2) / 3.0)
        ).astype(np.float32)
        direction = np.zeros((*score.shape, 3), dtype=np.float32)
        direction[..., 0] = 1.0
        candidates = np.asarray(
            [[score[16, 16, x], 16, 16, x] for x in (10, 16, 22)],
            dtype=np.float32,
        )
        scalar = [
            _refine_needle(
                score,
                direction,
                (
                    float(candidate[0]),
                    int(candidate[1]),
                    int(candidate[2]),
                    int(candidate[3]),
                ),
                radius=4,
                needle_length=16.0,
                cross_section_radius=2.0,
            )
            for candidate in candidates
        ]
        batched = _refine_needles_batch(
            score,
            direction,
            candidates,
            radius=4,
            needle_length=16.0,
            cross_section_radius=2.0,
            batch_size=2,
        )
        self.assertTrue(all(value is not None for value in scalar))
        self.assertEqual([value["candidateIndex"] for value in batched], [0, 1, 2])
        for expected, actual in zip(scalar, batched):
            assert expected is not None
            np.testing.assert_allclose(actual["center"], expected["center"], atol=3.0e-5)
            np.testing.assert_allclose(
                actual["direction"], expected["direction"], atol=3.0e-5
            )
            for key in (
                "linearity",
                "score",
                "axialCoverage",
                "longestAxialRun",
                "supportScore",
            ):
                self.assertAlmostEqual(actual[key], expected[key], places=5)

    @classmethod
    def setUpClass(cls) -> None:
        cls.volume = VolumeData(synthetic_scroll((72, 88, 88)), "Synthetic test scroll")

    def test_png_encoder(self) -> None:
        encoded = grayscale_png(np.arange(20, dtype=np.uint8).reshape(4, 5))
        self.assertTrue(encoded.startswith(b"\x89PNG\r\n\x1a\n"))

    def test_fit_is_seed_anchored_and_finite(self) -> None:
        seed = self.volume.suggested_seed
        result = fit_local_chart(
            self.volume,
            {
                "seed": seed,
                "patchSize": 22,
                "gridSize": 13,
                "depth": 10,
                "depthSamples": 9,
                "fieldRadius": 3,
                "iterations": 8,
            },
        )
        mesh = np.asarray(result["mesh"]["positions"], dtype=np.float32).reshape(13, 13, 3)
        np.testing.assert_allclose(mesh[6, 6], [seed["x"], seed["y"], seed["z"]], atol=1.0e-4)
        self.assertTrue(np.isfinite(mesh).all())
        self.assertTrue(result["rectified"]["center"].startswith("data:image/png;base64,"))
        self.assertEqual(result["stats"]["claim"], "phase-neutral carrier chart; no layer identity assigned")

    def test_air_seed_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "appears to be air"):
            fit_local_chart(self.volume, {"seed": {"x": 3, "y": 3, "z": 3}})

    def test_npy_sidecar_preserves_scan_coordinates(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "crop.npy"
            np.save(path, synthetic_scroll((32, 36, 40)))
            path.with_suffix(".json").write_text(
                json.dumps(
                    {
                        "name": "Real test cuboid",
                        "originXYZ": [1000, 2000, 3000],
                        "voxelSizeMicrons": 9.362,
                        "sourceKind": "zarr-cuboid",
                        "sourceShapeZYX": [9000, 8000, 7000],
                        "suggestedSeed": {"x": 20, "y": 18, "z": 16},
                    }
                )
            )
            loaded = VolumeData.load(path)
            metadata = loaded.metadata()
            self.assertEqual(metadata["name"], "Real test cuboid")
            self.assertEqual(metadata["origin"], {"x": 1000, "y": 2000, "z": 3000})
            self.assertEqual(metadata["suggestedSeed"], {"x": 20, "y": 18, "z": 16})
            self.assertEqual(
                metadata["globalSuggestedSeed"], {"x": 1020, "y": 2018, "z": 3016}
            )
            self.assertEqual(metadata["voxelUnit"], "µm")

    def test_centered_cube_has_exact_shape_and_padding(self) -> None:
        cube, origin = self.volume.cube((2, 3, 4), 16)
        self.assertEqual(cube.shape, (16, 16, 16))
        self.assertEqual(origin, (-6, -5, -4))
        self.assertTrue(np.all(cube[:4] == 0))
        with self.assertRaisesRegex(ValueError, "between 16 and 128"):
            self.volume.cube((20, 20, 20), 129)
        with self.assertRaisesRegex(ValueError, "padded Acus context"):
            self.volume.context_cube((4, 4, 4), 32)

    def test_acus_recovers_shared_normal_from_crossed_needles(self) -> None:
        volume_size = 112
        cube_size = 48
        center = 56
        z, y, x = np.indices((volume_size, volume_size, volume_size), dtype=np.float32)
        phantom = np.full((volume_size, volume_size, volume_size), 8.0, dtype=np.float32)
        center_z = float(center)
        for center_y in (42.0, 51.0, 61.0, 70.0):
            phantom += 155.0 * np.exp(
                -((z - center_z) ** 2 + (y - center_y) ** 2) / (2.0 * 1.35**2)
            )
        for center_x in (45.0, 56.0, 67.0):
            phantom += 145.0 * np.exp(
                -((z - center_z - 1.5) ** 2 + (x - center_x) ** 2) / (2.0 * 1.35**2)
            )
        phantom += 2.0 * np.sin(x * 0.31 + y * 0.19 + z * 0.13)
        volume = VolumeData(np.clip(phantom, 0, 255).astype(np.uint8), "Acus phantom")
        result = fit_acus(
            volume,
            {
                "seed": {"x": center, "y": center, "z": center},
                "cubeSize": cube_size,
                "scale": 1.15,
                "spacing": 4,
                "maxNeedles": 120,
            },
        )

        normal = np.asarray(result["normal"], dtype=np.float32)
        self.assertGreaterEqual(result["stats"]["needleCount"], 6)
        self.assertGreater(float(normal[2]), 0.9)
        self.assertAlmostEqual(float(np.linalg.norm(normal)), 1.0, places=4)
        self.assertEqual(
            result["normalSignConvention"],
            "largest-magnitude XYZ component is positive",
        )
        self.assertEqual(
            result["stats"]["constraint"],
            "shared unsigned normal; no surface or sheet identity fitted",
        )
        self.assertEqual(len(result["normalLine"]["start"]), 3)
        profile = result["orientationProfile"]
        self.assertEqual(len(profile["depthCenters"]), len(profile["density"]))
        self.assertEqual(len(profile["orientationCentersDeg"]), 36)
        self.assertTrue(
            all(len(row) == len(profile["orientationCentersDeg"]) for row in profile["density"])
        )
        self.assertGreater(profile["stats"]["meanTwoModeCoverage"], 0.45)
        self.assertEqual(result["settings"]["needleLength"], 16.0)
        self.assertIn(result["stats"]["computeBackend"], {"cpu", "gpu"})
        self.assertGreater(result["stats"]["lineFieldMs"], 0.0)
        self.assertEqual(
            result["settings"]["requestedPadding"],
            result["settings"]["needleLength"],
        )
        self.assertGreaterEqual(
            result["settings"]["effectivePadding"], result["settings"]["minimumPadding"]
        )
        self.assertGreater(result["stats"]["medianAxialCoverage"], 0.4)
        self.assertTrue(all(needle["length"] == 16.0 for needle in result["needles"]))

        field = fit_acus_field(
            volume,
            {
                "seed": {"x": center, "y": center, "z": center},
                "cubeSize": cube_size,
                "scale": 1.15,
                "spacing": 4,
                "maxNeedles": 80,
                "gridSize": 3,
                "fieldSpacing": 4,
            },
        )
        self.assertEqual(len(field["cells"]), 9)
        self.assertGreaterEqual(field["stats"]["validCellCount"], 7)
        self.assertEqual(field["stats"]["lineFieldBatchSize"], 8)
        self.assertFalse(field["stats"]["anchorReused"])
        anchor = next(cell for cell in field["cells"] if cell["isAnchor"])
        self.assertEqual(anchor["normalAngleDeg"], 0.0)
        self.assertEqual(anchor["profileCorrelation"], 1.0)
        self.assertIn("overlapping cubes", field["stats"]["warning"])

        audit = fit_acus_audit(
            volume,
            {
                "seed": {"x": center, "y": center, "z": center},
                "cubeSize": cube_size,
                "scale": 1.15,
                "spacing": 4,
                "maxNeedles": 72,
                "fieldSpacings": [4, 12],
                "bootstrapRepetitions": 12,
                "nullRepetitions": 12,
            },
        )
        self.assertEqual(audit["spacings"], [4, 12])
        self.assertEqual(len(audit["sweeps"]), 2)
        self.assertIn(audit["stats"]["computeBackend"], {"cpu", "gpu"})
        self.assertGreater(
            audit["sweeps"][0]["medianOverlapFraction"],
            audit["sweeps"][1]["medianOverlapFraction"],
        )
        self.assertIsNotNone(audit["sweeps"][0]["medianProfileNull"])
        self.assertIsNotNone(audit["sweeps"][0]["medianNormalBootstrapP90Deg"])

        padding_audit = fit_acus_padding_audit(
            volume,
            {
                "seed": {"x": center, "y": center, "z": center},
                "cubeSize": cube_size,
                "scale": 1.15,
                "spacing": 4,
                "maxNeedles": 72,
                "needleLength": 16,
                "paddingValues": [0, 16, 24],
            },
        )
        self.assertEqual([sweep["requestedPadding"] for sweep in padding_audit["sweeps"]], [0, 16, 24])
        self.assertFalse(padding_audit["sweeps"][0]["paddingSufficient"])
        self.assertTrue(padding_audit["sweeps"][-1]["paddingSufficient"])
        self.assertEqual(padding_audit["sweeps"][-1]["normalAngleToReferenceDeg"], 0.0)

    def test_region_bake_builds_cached_neighbor_evidence_field(self) -> None:
        size = 64
        z, y, x = np.indices((size, size, size), dtype=np.float32)
        phantom = np.full((size, size, size), 8.0, dtype=np.float32)
        for center_y in (22.0, 32.0, 42.0):
            phantom += 155.0 * np.exp(
                -((z - 32.0) ** 2 + (y - center_y) ** 2) / (2.0 * 1.2**2)
            )
        for center_x in (22.0, 32.0, 42.0):
            phantom += 145.0 * np.exp(
                -((z - 33.5) ** 2 + (x - center_x) ** 2) / (2.0 * 1.2**2)
            )
        volume = VolumeData(
            np.clip(phantom, 0, 255).astype(np.uint8), "Region test phantom"
        )
        request = {
            "cubeSize": 24,
            "scale": 1.0,
            "spacing": 4,
            "maxNeedles": 64,
            "needleLength": 6,
            "contextPadding": 8,
            "gridStride": 12,
            "tileCore": 32,
            "catalogBinSize": 16,
            "maxNeedlesPerBin": 8,
            "force": True,
        }
        result = fit_acus_region(volume, request)
        self.assertEqual(result["shape"], {"x": 64, "y": 64, "z": 64})
        self.assertEqual(len(result["cells"]), 27)
        self.assertGreaterEqual(result["stats"]["validCellCount"], 18)
        self.assertGreater(result["stats"]["needleCount"], 20)
        self.assertIsNotNone(result["stats"]["medianNeighborNormalDeg"])
        self.assertIsNotNone(result["stats"]["medianNeighborPattern"])
        self.assertEqual(
            result["stats"]["constraint"],
            "region evidence field; no sheet identity or surface connectivity assigned",
        )
        cached = fit_acus_region(volume, {**request, "force": False})
        self.assertTrue(cached["stats"]["cacheHit"])
        self.assertEqual(cached["stats"]["elapsedMs"], 0.0)

    def test_chunked_slab_analysis_builds_resumable_normal_field(self) -> None:
        size = 64
        z, y, x = np.indices((size, size, size), dtype=np.float32)
        phantom = np.full((size, size, size), 8.0, dtype=np.float32)
        for center_y in (22.0, 32.0, 42.0):
            phantom += 155.0 * np.exp(
                -((z - 32.0) ** 2 + (y - center_y) ** 2) / (2.0 * 1.2**2)
            )
        for center_x in (22.0, 32.0, 42.0):
            phantom += 145.0 * np.exp(
                -((z - 33.5) ** 2 + (x - center_x) ** 2) / (2.0 * 1.2**2)
            )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "slab.npy"
            np.save(source, np.clip(phantom, 0, 255).astype(np.uint8))
            source.with_suffix(".json").write_text(
                json.dumps({"originXYZ": [100, 200, 300]})
            )
            output = Path(directory) / "analysis"
            result = run_slab_analysis(
                source,
                output,
                {
                    "cubeSize": 24,
                    "scale": 1.0,
                    "spacing": 4,
                    "needleLength": 6,
                    "halo": 8,
                    "gridStride": 12,
                    "tileCore": 64,
                    "binSize": 16,
                    "maxNeedlesPerBin": 8,
                    "maxNeedles": 64,
                    "calibrationTiles": 16,
                    "strengthScale": 0.07197796553373337,
                },
            )
            self.assertEqual(result["state"], "complete")
            self.assertEqual(result["strengthScale"], 0.07197796553373337)
            self.assertEqual(
                result["identity"]["settings"]["fixedStrengthScale"],
                0.07197796553373337,
            )
            self.assertGreater(result["needleCount"], 20)
            self.assertGreaterEqual(result["validCellCount"], 18)
            cells = np.load(output / "cells.npy")
            self.assertEqual(cells.shape, (3, 3, 3))
            normals = cells["normal"][cells["valid"].astype(bool)]
            self.assertGreater(float(np.median(np.abs(normals[:, 2]))), 0.85)
            sliced = slab_overview(output, maximum_cells=1000, z_index=1)
            self.assertEqual(sliced["view"]["mode"], "slice")
            self.assertEqual(sliced["view"]["zIndex"], 1)
            self.assertEqual(sliced["grid"]["z"], [32])
            self.assertEqual(sliced["grid"]["availableZ"], [20, 32, 44])
            self.assertEqual(sliced["stats"]["cellCount"], 9)
            self.assertEqual(len(sliced["cells"]), 9)
            with self.assertRaisesRegex(ValueError, "zIndex must be between"):
                slab_overview(output, z_index=3)
            flakes = slab_flake_plane(output, z_index=1, maximum_flakes=3, force=True)
            self.assertEqual(flakes["view"], {"mode": "slice", "zIndex": 1, "z": 32})
            self.assertGreater(flakes["stats"]["flakeCount"], 0)
            self.assertGreater(flakes["stats"]["fittedCellCount"], 0)
            self.assertTrue(
                all(flake["needleCount"] >= 5 for flake in flakes["flakes"])
            )
            self.assertTrue(
                all(abs(float(np.dot(flake["normal"], flake["fiber"]))) < 1.0e-3 for flake in flakes["flakes"])
            )
            cached_flakes = slab_flake_plane(output, z_index=1)
            self.assertTrue(cached_flakes["stats"]["cacheHit"])
            audit = slab_flake_audit(output, z_index=1, repetitions=2, force=True)
            self.assertEqual(
                [sweep["spacingVoxels"] for sweep in audit["sweeps"]],
                [12, 24, 36],
            )
            self.assertFalse(audit["sweeps"][0]["independentWindows"])
            self.assertTrue(audit["sweeps"][1]["independentWindows"])
            self.assertEqual(audit["sweeps"][1]["overlapFraction"], 0.0)
            self.assertEqual(audit["sweeps"][0]["nulls"]["fiber"]["repetitions"], 2)
            self.assertIn("links", audit["sweeps"][0])
            cached_audit = slab_flake_audit(output, z_index=1, repetitions=2)
            self.assertTrue(cached_audit["stats"]["cacheHit"])
            with self.assertRaisesRegex(ValueError, "zIndex must be between"):
                slab_flake_plane(output, z_index=3)

    def test_macro_radial_fit_recovers_unsigned_normal_center(self) -> None:
        grid_x = list(range(32, 513, 32))
        grid_y = list(range(32, 513, 32))
        cells = np.zeros((3, len(grid_y), len(grid_x)), dtype=CELL_DTYPE)
        expected_center = np.asarray([272.0, 240.0], dtype=np.float32)
        for iz in range(cells.shape[0]):
            for iy, center_y in enumerate(grid_y):
                for ix, center_x in enumerate(grid_x):
                    radial = np.asarray([center_x, center_y], dtype=np.float32) - expected_center
                    if float(np.linalg.norm(radial)) < 8.0:
                        continue
                    radial /= np.linalg.norm(radial)
                    cells[iz, iy, ix]["valid"] = 1
                    cells[iz, iy, ix]["normal"] = [radial[0], radial[1], 0.0]
                    cells[iz, iy, ix]["needleCount"] = 48
                    cells[iz, iy, ix]["normalConfidence"] = 0.8
                    cells[iz, iy, ix]["coplanarity"] = 0.95
        result = _macro_radial_fit(cells, grid_x, grid_y)
        np.testing.assert_allclose(result["macroRadialCenterXY"], expected_center, atol=0.1)
        self.assertLess(result["medianMacroRadialResidualDeg"], 0.05)
        self.assertEqual(result["macroRadialFitCellCount"], int(np.count_nonzero(cells["valid"])))

    def test_holdout_split_and_sheetlet_matching_are_independent_and_axial(self) -> None:
        record_ids = np.arange(256, dtype=np.uint64)
        first = _stable_fold(record_ids, 358)
        second = _stable_fold(record_ids, 358)
        np.testing.assert_array_equal(first, second)
        self.assertGreater(int(np.count_nonzero(first)), 90)
        self.assertGreater(int(np.count_nonzero(~first)), 90)

        def flake(cell: list[int], center: list[float], fiber: list[float]) -> dict[str, object]:
            return {
                "cellIndex": cell,
                "cellCenter": center,
                "center": center,
                "normal": [0.0, 1.0, 0.0],
                "fiber": fiber,
                "depthOffset": 0.0,
                "quality": 0.5,
                "validationScore": 0.2,
                "effectiveSupport": 8.0,
                "needleCount": 10,
            }

        fold_matches = _mutual_cell_matches(
            [flake([0, 0, 0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
            [flake([0, 0, 0], [0.0, 0.4, 0.0], [-1.0, 0.0, 0.0])],
        )
        self.assertEqual(len(fold_matches), 1)
        self.assertLess(fold_matches[0]["fiberAngle"], 0.01)
        other_family = flake(
            [0, 0, 0], [0.0, 0.4, 0.0], [-1.0, 0.0, 0.0]
        )
        other_family["normalFamily"] = 1
        self.assertEqual(
            _mutual_cell_matches(
                [flake([0, 0, 0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0])],
                [other_family],
            ),
            [],
        )
        links = _match_3d(
            [
                flake([0, 0, 0], [0.0, 0.0, 0.0], [1.0, 0.0, 0.0]),
                flake([0, 0, 2], [0.0, 0.0, 64.0], [1.0, 0.0, 0.0]),
            ],
            2,
        )
        self.assertEqual(len(links), 1)
        self.assertEqual(links[0]["axis"], "z")

    def test_exploratory_sheetlets_allow_curvature_but_reject_cell_collisions(self) -> None:
        def flake(
            cell: list[int],
            center: list[float],
            normal: list[float],
            normal_family: int = 0,
        ) -> dict[str, object]:
            fiber = np.asarray([0.0, 0.0, 1.0], dtype=np.float32)
            normal_array = np.asarray(normal, dtype=np.float32)
            cross = np.cross(normal_array, fiber)
            cross /= np.linalg.norm(cross)
            return {
                "cellIndex": cell,
                "normalFamily": normal_family,
                "center": center,
                "normal": normal,
                "fiber": fiber.tolist(),
                "crossFiber": cross.tolist(),
                "quality": 0.3,
                "radiusFiber": 20.0,
                "radiusCrossFiber": 12.0,
            }

        curved = _direction_edge_links(
            [
                flake([0, 0, 0], [0.0, 0.0, 0.0], [0.0, 1.0, 0.0]),
                flake([1, 0, 0], [32.0, 2.0, 0.0], [-0.124, 0.992, 0.0]),
            ],
            cell_step=1,
            edge_padding=8.0,
        )
        self.assertEqual(len(curved["score"]), 1)
        self.assertLess(float(curved["edgeResidual"][0]), 0.02)
        self.assertGreater(float(curved["normalBend"][0]), 7.0)

        cross_family_close = _direction_edge_links(
            [
                flake(
                    [0, 0, 0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    normal_family=0,
                ),
                flake(
                    [1, 0, 0],
                    [32.0, 4.213, 0.0],
                    [-0.258819, 0.965926, 0.0],
                    normal_family=1,
                ),
            ],
            cell_step=1,
            edge_padding=8.0,
        )
        self.assertEqual(len(cross_family_close["score"]), 0)
        cross_family_jump = _direction_edge_links(
            [
                flake(
                    [0, 0, 0],
                    [0.0, 0.0, 0.0],
                    [0.0, 1.0, 0.0],
                    normal_family=0,
                ),
                flake(
                    [1, 0, 0],
                    [32.0, 8.574, 0.0],
                    [-0.5, 0.866025, 0.0],
                    normal_family=1,
                ),
            ],
            cell_step=1,
            edge_padding=8.0,
        )
        self.assertEqual(len(cross_family_jump["score"]), 0)

        component, _, _, retained = _components_without_cell_collisions(
            4,
            np.asarray([0, 1, 2, 0], dtype=np.int64),
            np.asarray([0, 1, 2], dtype=np.uint32),
            np.asarray([1, 2, 3], dtype=np.uint32),
            np.asarray([0.95, 0.9, 0.8], dtype=np.float32),
        )
        self.assertTrue(retained[0])
        self.assertTrue(retained[1])
        self.assertFalse(retained[2])
        self.assertNotEqual(int(component[0]), int(component[3]))

    def test_pure_secondary_fragments_have_a_smaller_seed_threshold(self) -> None:
        flakes = [
            {
                "cellIndex": [index, 0, 0],
                "center": [float(index * 32), 0.0, 0.0],
                "normal": [0.0, 1.0, 0.0],
                "fiber": [1.0, 0.0, 0.0],
                "quality": 0.2,
                "normalFamily": 1,
            }
            for index in range(5)
        ]
        empty_links = {
            "source": np.empty(0, dtype=np.uint32),
            "axis": np.empty(0, dtype=np.uint8),
            "edgeResidual": np.empty(0, dtype=np.float32),
            "fiberAngle": np.empty(0, dtype=np.float32),
            "normalBend": np.empty(0, dtype=np.float32),
        }
        candidates = _candidate_catalog(
            flakes,
            empty_links,
            np.empty(0, dtype=bool),
            np.zeros(5, dtype=np.int32),
            np.asarray([5], dtype=np.int32),
            np.zeros(5, dtype=np.uint8),
        )
        self.assertEqual(len(candidates), 1)
        self.assertEqual(candidates[0]["candidateClass"], "secondary-seed")
        large_secondary = [dict(flakes[index % len(flakes)]) for index in range(20)]
        for index, flake in enumerate(large_secondary):
            flake["cellIndex"] = [index, 0, 0]
        large_candidates = _candidate_catalog(
            large_secondary,
            empty_links,
            np.empty(0, dtype=bool),
            np.zeros(20, dtype=np.int32),
            np.asarray([20], dtype=np.int32),
            np.zeros(20, dtype=np.uint8),
        )
        self.assertEqual(large_candidates[0]["candidateClass"], "secondary-seed")
        for flake in flakes:
            flake["normalFamily"] = 0
        self.assertEqual(
            _candidate_catalog(
                flakes,
                empty_links,
                np.empty(0, dtype=bool),
                np.zeros(5, dtype=np.int32),
                np.asarray([5], dtype=np.int32),
                np.zeros(5, dtype=np.uint8),
            ),
            [],
        )

    def test_sheetlet_carrier_follows_curved_normals_and_samples_depth(self) -> None:
        flakes = []
        for y in (-16.0, 0.0, 16.0):
            for x in (-16.0, 0.0, 16.0):
                height = 0.006 * x * x
                normal = np.asarray([-0.012 * x, 0.0, 1.0], dtype=np.float32)
                normal /= np.linalg.norm(normal)
                flakes.append(
                    {
                        "center": [x + 32.0, y + 32.0, height + 24.0],
                        "normal": normal.tolist(),
                        "fiber": [0.0, 1.0, 0.0],
                        "quality": 0.4,
                    }
                )
        carrier = _mls_carrier(
            flakes,
            pixel_step=2.0,
            bandwidth=24.0,
            support_radius=24.0,
            maximum_pixels=64,
        )
        self.assertEqual(len(carrier["nodeHeightResidualVoxels"]), len(flakes))
        self.assertEqual(len(carrier["nodeNormalResidualDeg"]), len(flakes))
        self.assertEqual(carrier["stats"]["normalFamilies"]["0"]["flakeCount"], len(flakes))
        self.assertGreater(carrier["stats"]["supportedPixelFraction"], 0.5)
        self.assertLess(carrier["stats"]["medianNodeHeightResidualVoxels"], 2.0)
        self.assertLess(carrier["stats"]["medianNodeNormalResidualDeg"], 10.0)
        boundary = _carrier_boundary(carrier, spacing=8.0, maximum_points=64)
        self.assertGreater(len(boundary["point"]), 8)
        np.testing.assert_allclose(
            np.linalg.norm(boundary["outward"], axis=1), 1.0, atol=1.0e-4
        )

        z, _, _ = np.indices((64, 64, 64), dtype=np.float32)
        sampled, _ = _sample_stack(
            z,
            carrier,
            np.asarray([-2.0, 0.0, 2.0], dtype=np.float32),
        )
        mask = carrier["supportMask"]
        means = [float(np.mean(plane[mask])) for plane in sampled]
        self.assertGreater(means[1], means[0])
        self.assertGreater(means[2], means[1])

    def test_carrier_texture_score_selects_directional_depth_plane(self) -> None:
        size = 96
        coordinate = np.arange(size, dtype=np.float32)
        stripes = np.tile(100.0 + 40.0 * np.sin(0.7 * coordinate), (size, 1))
        random = np.random.default_rng(17).normal(100.0, 40.0, (size, size)).astype(
            np.float32
        )
        mask = np.ones((size, size), dtype=bool)

        stripe_texture = _plane_texture(stripes, mask)
        random_texture = _plane_texture(random, mask)
        self.assertGreater(
            float(stripe_texture["textureScore"]),
            float(random_texture["textureScore"]) + 0.5,
        )

        profile = _texture_profile(
            np.stack([random, stripes, random]),
            mask,
            np.asarray([-1.0, 0.0, 1.0], dtype=np.float32),
        )
        self.assertEqual(profile["bestDepthOffsetVoxels"], 0.0)
        self.assertAlmostEqual(
            float(profile["bestPlane"]["dominantFiberAngleDeg"]), 90.0, delta=1.0
        )

    def test_carrier_yield_rewards_area_and_penalizes_bad_fit(self) -> None:
        texture = {"bestTextureScore": 0.4}
        clean = {
            "supportedPixelCount": 1000,
            "pixelStepVoxels": 2.0,
            "medianNodeHeightResidualVoxels": 1.0,
            "medianNodeNormalResidualDeg": 3.0,
        }
        folded = {
            **clean,
            "medianNodeHeightResidualVoxels": 9.0,
            "medianNodeNormalResidualDeg": 14.0,
        }
        large = {**clean, "supportedPixelCount": 2000}
        self.assertGreater(
            _carrier_yield(clean, texture)["constructionYieldScore"],
            _carrier_yield(folded, texture)["constructionYieldScore"],
        )
        self.assertAlmostEqual(
            _carrier_yield(large, texture)["constructionYieldScore"],
            2.0 * _carrier_yield(clean, texture)["constructionYieldScore"],
            delta=0.02,
        )

    def test_carrier_boundary_match_requires_facing_tangent_edges(self) -> None:
        arrays = {
            "point": np.asarray([[0.0, 0.0, 0.0], [10.0, 0.0, 0.0]], dtype=np.float32),
            "normal": np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 1.0]], dtype=np.float32),
            "fiber": np.asarray([[0.0, 1.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float32),
            "outward": np.asarray([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]], dtype=np.float32),
        }
        matched = _score_point_pairs(np.asarray([0]), np.asarray([1]), arrays)
        self.assertTrue(bool(matched["valid"][0]))
        arrays["outward"][1] *= -1.0
        rejected = _score_point_pairs(np.asarray([0]), np.asarray([1]), arrays)
        self.assertFalse(bool(rejected["valid"][0]))

    def test_guided_growth_prefers_coplanar_fiber_continuation(self) -> None:
        flakes = [
            {
                "center": [x, y, 0.0],
                "normal": [0.0, 0.0, 1.0],
                "fiber": [1.0, 0.0, 0.0],
                "quality": 0.4,
                "cellIndex": [index, 0, 0],
            }
            for index, (x, y) in enumerate(
                ((0.0, -12.0), (0.0, 12.0), (24.0, -12.0), (24.0, 12.0))
            )
        ]
        flakes.extend(
            [
                {
                    "center": [48.0, 0.0, 0.5],
                    "normal": [0.0, 0.0, 1.0],
                    "fiber": [1.0, 0.0, 0.0],
                    "quality": 0.4,
                    "cellIndex": [4, 0, 0],
                },
                {
                    "center": [48.0, 0.0, 9.0],
                    "normal": [0.0, 1.0, 0.0],
                    "fiber": [1.0, 0.0, 0.0],
                    "quality": 0.4,
                    "cellIndex": [4, 0, 0],
                },
            ]
        )
        score = _score_growth_candidates(
            np.arange(4), np.asarray([4, 5]), _flake_arrays(flakes), flakes
        )
        self.assertGreater(float(score["score"][0]), 0.8)
        self.assertGreater(float(score["score"][0]), float(score["score"][1]) + 0.7)

        cross_family = dict(flakes[4])
        cross_family["normalFamily"] = 1
        separated = flakes[:4] + [cross_family]
        family_score = _score_growth_candidates(
            np.arange(4), np.asarray([4]), _flake_arrays(separated), separated
        )
        self.assertEqual(float(family_score["score"][0]), 0.0)

    def test_iterative_carrier_merge_rejects_transitive_cell_collision(self) -> None:
        states = [
            {
                "members": {index},
                "occupiedCells": cells,
                "assemblyComponentIds": [index],
                "sourceRanks": [index + 1],
                "initialFlakeCount": 1,
            }
            for index, cells in enumerate(({1}, {2}, {1}))
        ]
        edges = [
            {"source": 0, "target": 1, "score": 0.9},
            {"source": 1, "target": 2, "score": 0.8},
        ]
        merged, retained, conflicts = _merge_states(states, edges, 0.45)
        self.assertEqual(len(merged), 2)
        self.assertEqual(len(retained), 1)
        self.assertEqual(conflicts, 1)

    def test_long_range_bridge_requires_facing_edges_and_internal_ct_material(self) -> None:
        first = {
            "point": np.asarray([[8.0, 32.0, 32.0]], dtype=np.float32),
            "normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "fiber": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "outward": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
        }
        second = {
            "point": np.asarray([[56.0, 32.0, 32.0]], dtype=np.float32),
            "normal": np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            "fiber": np.asarray([[0.0, 1.0, 0.0]], dtype=np.float32),
            "outward": np.asarray([[-1.0, 0.0, 0.0]], dtype=np.float32),
        }
        endpoint = _bridge_endpoint_scores(first, second)
        self.assertTrue(bool(endpoint["valid"][0]))
        self.assertGreater(float(endpoint["score"][0]), 0.8)

        volume = np.zeros((64, 64, 64), dtype=np.float32)
        volume[32, :, :] = 100.0
        points = np.asarray(
            [[x, 32.0, 32.0] for x in np.linspace(16.0, 48.0, 5)],
            dtype=np.float32,
        )
        normals = np.tile([0.0, 0.0, 1.0], (len(points), 1)).astype(np.float32)
        evidence = _ct_bridge_evidence(volume, points, normals)
        self.assertEqual(evidence["materialFraction"], 1.0)
        self.assertEqual(evidence["ridgeFraction"], 1.0)


if __name__ == "__main__":
    unittest.main()
