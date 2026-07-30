from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cubical.block import (
    BlockBounds,
    _ParityDisjointSet,
    augment_surface_block,
    assemble_surface_hierarchy,
    assemble_surface_block,
    assemble_surface_block_from_candidates,
    extend_surface_block_joins,
    merge_surface_blocks,
    rebuild_surface_block,
)
from backend.cubical.boundary_band import (
    BoundaryBandSettings,
    run_boundary_band_export,
)
from backend.cubical.boundary_audit import run_cluster_reference_audit
from backend.cubical.boundary_merge import (
    _ordered_packet_alignment,
    run_boundary_band_merge,
)
from backend.cubical.boundary_reselection import run_boundary_band_reselection
from backend.cubical.cluster_reselection import run_boundary_cluster_reselection
from backend.cubical.cluster_materialization import run_cluster_materialization
from backend.cubical.cell_refinement import CellRefinementSettings
from backend.cubical.cell_refinement_targets import (
    _percentile_ranks,
    _spatially_separated,
)
from backend.cubical.boundary_topology import (
    build_frozen_face_states,
    build_frozen_region_states,
    compatible_face_masks,
    face_mask,
    freeze_topology_outside_patches,
    read_frozen_face_state,
    read_frozen_region_state,
    select_joins_with_frozen_topology,
    write_frozen_region_artifact,
    write_frozen_topology_artifact,
)
from backend.cubical.continuity import score_join_continuity
from backend.cubical.contextual_growth import (
    ContextualGrowthSettings,
    discover_contextual_growth_candidates,
)
from backend.cubical.geometry import (
    DegeneratePlaneIntersection,
    PlaneEstimate,
    axial_angle_radians,
    clip_plane_to_cell,
)
from backend.cubical.matching import (
    TraceMatchSettings,
    align_face_patches,
    match_face_traces,
)
from backend.cubical.multiseam import run_multiseam_audit
from backend.cubical.continuation import discover_mode_continuations
from backend.cubical.gaps import analyze_component_gaps
from backend.cubical.flatten import (
    component_mesh,
    rasterize_chart,
    sample_depth_stack,
    tangent_atlas_chart,
)
from backend.cubical.contracts import RawAcusSettings, VolumeSource, sha256_file
from backend.cubical.repair import evaluate_single_cell_gap_repairs
from backend.cubical.saturation import classify_cell_structural_evidence
from backend.cubical.saturation_reselection import (
    SaturationReselectionSettings,
    configuration_evidence_log_score,
    enumerate_cell_saturation_configurations,
)
from backend.cubical.saturation_selection import (
    load_saturation_candidates,
    reweight_saturation_candidates,
)
from backend.cubical.sheet_packets import run_dual_axis_packet_connectivity
from backend.cubical.sheet_evidence import (
    SheetEvidenceInput,
    compile_block_sheet_evidence,
    read_block_sheet_evidence,
)
from backend.cubical.sheet_correspondence import enumerate_mode_correspondences
from backend.cubical.sheet_configuration_solver import (
    SheetConfigurationSolverSettings,
    _max_sum_configuration_seed,
)
from backend.cubical.sheet_factors import _ordered_alignment_factor
from backend.cubical.sheet_ownership import (
    crop_surface_graph_to_owned_block,
    extract_sheet_evidence_subblock,
    finalize_sheet_halo_experiment,
)
from backend.cubical.sheet_stitching import (
    SheetMatchingPolicy,
    SheetStitchingSettings,
    _minimum_undirected_join_cut,
    enumerate_sheet_join_catalog,
    restitch_sheet_graph,
)
from backend.cubical.sheet_topology_refinement import (
    SheetTopologyEvaluation,
    choose_improving_topology_evaluation,
    frozen_exterior_join_keys,
)
from backend.cubical.surface_graph import read_surface_graph, write_surface_graph
from backend.cubical.selection import (
    ConfigurationOption,
    optimize_configurations,
    pairwise_reward_energy,
)
from backend.cubical.stratigraphic_continuity import (
    PatchFingerprintTable,
    StratigraphicContinuitySettings,
    _calibrate_records,
    build_patch_fingerprints,
    score_patch_fingerprints,
)
from backend.cubical.stratigraphy import ConfigurationTable, LayerModeTable
from backend.cubical.subblock import extract_selected_patch_subblock
from backend.cubical.topology import GridSpec, cell_edges, cell_face
from backend.cubical.tables import PatchTable, read_patch_shard, write_patch_shard
from backend.cubical.synthetic import (
    SyntheticStackSettings,
    generate_synthetic_stack,
)


class CubicalGeometryTests(unittest.TestCase):
    @staticmethod
    def _horizontal_patch(
        grid: GridSpec,
        cell: tuple[int, int, int],
        height: float,
        patch_id: int,
        *,
        height_std: float = 0.02,
    ):
        patch = clip_plane_to_cell(
            grid,
            cell,
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                height,
                angular_std_radians=math.radians(1.0),
                height_std=height_std,
                fiber_xyz=(1.0, 0.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=patch_id,
        )
        assert patch is not None
        return patch

    def _write_analytic_candidate_boundary(
        self,
        root: Path,
        grid: GridSpec,
        *,
        patch_id_start: int,
    ) -> Path:
        selected = root / "selected"
        boundary = root / "boundary"
        cells = np.asarray(
            [
                (x, y, z)
                for z in range(grid.shape_cells_xyz[2])
                for y in range(grid.shape_cells_xyz[1])
                for x in range(grid.shape_cells_xyz[0])
            ],
            dtype=np.int32,
        )
        count = len(cells)
        patches = tuple(
            self._horizontal_patch(
                grid,
                tuple(int(value) for value in cell),
                0.0,
                patch_id_start + index,
            )
            for index, cell in enumerate(cells)
        )
        write_patch_shard(
            selected / "selected-patches-v1",
            PatchTable.from_patches(grid, patches),
        )
        configurations = ConfigurationTable(
            cells,
            np.arange(count + 1, dtype=np.uint64),
            np.zeros(count, dtype=np.uint16),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.int8),
            np.arange(count + 1, dtype=np.uint64),
            np.tile((0.0, 0.0, 1.0), (count, 1)).astype(np.float32),
            np.zeros(count, dtype=np.float32),
            np.tile(
                (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                (count, 1),
            ).astype(np.float32),
            np.tile((1.0, 0.0, 0.0), (count, 1)).astype(np.float32),
            np.full(count, math.radians(2.0), dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
        )
        candidate_path = selected / "saturation-configurations-v1.npz"
        metadata = {
            name: np.ones(count, dtype=np.float32)
            for name in (
                "evidenceLogScore",
                "physicalLogScore",
                "totalLogScore",
                "coveredEvidenceMass",
                "totalEvidenceMass",
            )
        }
        metadata["isCurrent"] = np.ones(count, dtype=np.uint8)
        with candidate_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                **configurations.arrays(),
                **metadata,
            )
        candidate_sha256 = sha256_file(candidate_path)
        (selected / "saturation-configurations-v1.json").write_text(
            json.dumps(
                {
                    "schema": "pareidolia.cubical-saturation-configurations",
                    "version": 1,
                    "identitySha256": f"analytic-{patch_id_start}",
                    "data": {
                        "path": candidate_path.name,
                        "bytes": candidate_path.stat().st_size,
                        "sha256": candidate_sha256,
                    },
                }
            )
        )
        selection_path = selected / "selection-v1.npz"
        with selection_path.open("wb") as handle:
            np.savez_compressed(
                handle,
                cellXYZ=cells,
                optionId=np.arange(count, dtype=np.uint64),
                sourceTableIndex=np.zeros(count, dtype=np.uint32),
                sourceConfigurationIndex=np.arange(count, dtype=np.uint32),
                localConfigurationId=np.zeros(count, dtype=np.uint16),
                configurationLogWeight=np.zeros(count, dtype=np.float32),
                selectedLayerCount=np.ones(count, dtype=np.uint16),
            )
        selection_sha256 = sha256_file(selection_path)
        (selected / "selection-v1.json").write_text(
            json.dumps(
                {
                    "schema": "pareidolia.raw-acus-configuration-selection",
                    "version": 1,
                    "data": {
                        "path": selection_path.name,
                        "bytes": selection_path.stat().st_size,
                        "sha256": selection_sha256,
                    },
                }
            )
        )
        (selected / "variant.json").write_text(
            json.dumps(
                {"identity": {"candidateDataSha256": candidate_sha256}}
            )
        )
        run_boundary_band_export(
            selected,
            boundary,
            candidate_root=selected,
            settings=BoundaryBandSettings(depth_cells=1),
        )
        return boundary

    def test_cell_topology_has_canonical_shared_features(self) -> None:
        self.assertEqual(len(set(cell_edges((2, 3, 4)))), 12)
        self.assertEqual(cell_face((2, 3, 4), 0, 1), cell_face((3, 3, 4), 0, 0))

    def test_orientation_parity_rejects_a_contradictory_cycle(self) -> None:
        orientation = _ParityDisjointSet((1, 2, 3))
        orientation.union(1, 2, False)
        orientation.union(2, 3, False)
        self.assertTrue(orientation.compatible(1, 3, False))
        self.assertFalse(orientation.compatible(1, 3, True))

    def test_structural_saturation_uses_unsigned_fiber_and_residual_gating(self) -> None:
        assignment = classify_cell_structural_evidence(
            np.asarray(
                (
                    (0.0, 0.0, 0.5),
                    (0.0, 0.0, 7.0),
                    (0.0, 0.0, 0.5),
                ),
                dtype=np.float32,
            ),
            np.asarray(
                (
                    (-1.0, 0.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (0.0, 1.0, 0.0),
                ),
                dtype=np.float32,
            ),
            cell_center_xyz=np.zeros(3, dtype=np.float32),
            patch_normals_xyz=np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
            patch_heights=np.asarray((0.0,), dtype=np.float32),
            patch_fibers_xyz=np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            patch_fiber_std_degrees=np.asarray((2.0,), dtype=np.float32),
            patch_confidence=np.asarray((0.7,), dtype=np.float32),
            depth_sigma_voxels=2.5,
            orientation_kernel_degrees=9.0,
        )
        self.assertLess(assignment.best_joint_residual[0], 0.25)
        self.assertGreater(assignment.best_joint_residual[1], 2.5)
        self.assertGreater(assignment.best_joint_residual[2], 8.0)
        self.assertLess(assignment.best_orthogonal_joint_residual[2], 0.25)
        np.testing.assert_allclose(assignment.best_assignment_share, 1.0)

    def test_structural_saturation_reports_competing_layer_ambiguity(self) -> None:
        assignment = classify_cell_structural_evidence(
            np.asarray(((0.0, 0.0, 0.0),), dtype=np.float32),
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            cell_center_xyz=np.zeros(3, dtype=np.float32),
            patch_normals_xyz=np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float32
            ),
            patch_heights=np.asarray((-0.25, 0.25), dtype=np.float32),
            patch_fibers_xyz=np.asarray(
                ((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), dtype=np.float32
            ),
            patch_fiber_std_degrees=np.asarray((2.0, 2.0), dtype=np.float32),
            patch_confidence=np.asarray((0.7, 0.7), dtype=np.float32),
            depth_sigma_voxels=2.5,
            orientation_kernel_degrees=9.0,
        )
        self.assertLess(assignment.best_joint_residual[0], 0.2)
        self.assertAlmostEqual(float(assignment.best_assignment_share[0]), 0.5)

    def test_saturation_mixture_normalization_does_not_reward_duplicate_modes(self) -> None:
        weights = np.asarray((1.0, 2.0), dtype=np.float32)
        one = configuration_evidence_log_score(
            np.asarray((0.8, 0.2)),
            0.8,
            weights,
            background_likelihood=0.05,
        )
        duplicate = configuration_evidence_log_score(
            np.asarray((1.6, 0.4)),
            1.6,
            weights,
            background_likelihood=0.05,
        )
        self.assertAlmostEqual(one, duplicate)

    def test_saturation_reselection_enumerates_physical_multilayer_coverage(self) -> None:
        modes = LayerModeTable(
            np.asarray(((0, 0, 0),), dtype=np.int32),
            np.asarray((0, 2), dtype=np.uint64),
            np.asarray((0, 0), dtype=np.int8),
            np.asarray(((0.0, 0.0, 1.0), (0.0, 0.0, 1.0)), dtype=np.float32),
            np.asarray((-5.0, 5.0), dtype=np.float32),
            np.zeros((2, 6), dtype=np.float32),
            np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
            np.radians(np.asarray((2.0, 2.0), dtype=np.float32)),
            np.asarray((0.8, 0.8), dtype=np.float32),
            np.asarray((-5.0, 5.0), dtype=np.float32),
            np.asarray((0.0, 0.0), dtype=np.float32),
            np.asarray((0.7, 0.7), dtype=np.float32),
            np.asarray((0.9, 0.9), dtype=np.float32),
            np.asarray((6.0, 6.0), dtype=np.float32),
        )
        modes.validate()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "source.npy"
            metadata_path = Path(directory) / "source.json"
            np.save(source_path, np.zeros((4, 4, 4), dtype=np.uint8))
            metadata_path.write_text('{"voxelSizeMicrons": 10.0}\n')
            source = VolumeSource.open(source_path, metadata_path)
            configurations, statistics = enumerate_cell_saturation_configurations(
                (0, 0, 0),
                "synthetic",
                modes,
                0,
                np.asarray(((0.0, 0.0, -5.0), (0.0, 0.0, 5.0)), dtype=np.float32),
                np.asarray(((1.0, 0.0, 0.0), (1.0, 0.0, 0.0)), dtype=np.float32),
                np.ones(2, dtype=np.float32),
                cell_center_xyz=np.zeros(3, dtype=np.float32),
                normal_confidence=np.asarray((0.8, 0.0), dtype=np.float32),
                current_mode_indices=(0,),
                source=source,
                raw_settings=RawAcusSettings(),
                settings=SaturationReselectionSettings(),
            )
        self.assertEqual(statistics["currentCoveredEvidenceMass"], 1.0)
        self.assertEqual(statistics["oracleCoveredEvidenceMass"], 2.0)
        self.assertTrue(any(value.mode_indices == (0, 1) for value in configurations))
        self.assertTrue(any(value.is_current for value in configurations))

    def test_saturation_candidate_coverage_reward_is_cell_normalized(self) -> None:
        table = ConfigurationTable(
            np.asarray(((0, 0, 0),), dtype=np.int32),
            np.asarray((0, 2), dtype=np.uint64),
            np.asarray((0, 1), dtype=np.uint16),
            np.asarray((0.0, 0.0), dtype=np.float32),
            np.asarray((-1, 0), dtype=np.int8),
            np.asarray((0, 0, 1), dtype=np.uint64),
            np.asarray(((0.0, 0.0, 1.0),), dtype=np.float32),
            np.asarray((0.0,), dtype=np.float32),
            np.zeros((1, 6), dtype=np.float32),
            np.asarray(((1.0, 0.0, 0.0),), dtype=np.float32),
            np.asarray((0.1,), dtype=np.float32),
            np.asarray((0.8,), dtype=np.float32),
            np.asarray((0.7,), dtype=np.float32),
            np.asarray((0.9,), dtype=np.float32),
            np.asarray((4.0,), dtype=np.float32),
        )
        table.validate()
        reweighted = reweight_saturation_candidates(
            table,
            {
                "totalLogScore": np.asarray((0.0, 0.0), dtype=np.float32),
                "coveredEvidenceMass": np.asarray((0.0, 2.0), dtype=np.float32),
            },
            coverage_reward_scale=0.5,
        )
        self.assertAlmostEqual(
            float(
                reweighted.configuration_log_weight[1]
                - reweighted.configuration_log_weight[0]
            ),
            1.0,
            places=6,
        )

    def test_stratigraphic_fingerprint_resolves_axial_depth_gauge(self) -> None:
        depths = np.arange(-10.0, 11.0, 5.0, dtype=np.float32)
        first_density = np.asarray((0.0, 1.0, 0.25, 0.5, 0.0), dtype=np.float32)
        first_moment = first_density * np.asarray(
            (0.0, -1.0, 1.0, -0.5, 0.0), dtype=np.float32
        )
        table = PatchFingerprintTable(
            patch_id=np.asarray((1, 2), dtype=np.uint64),
            anchor_valid=np.asarray((True, True)),
            anchor_shard_index=np.asarray((0, 0), dtype=np.int16),
            anchor_mode_index=np.asarray((1, 1), dtype=np.int32),
            anchor_height_residual_voxels=np.zeros(2, dtype=np.float32),
            anchor_normal_residual_degrees=np.zeros(2, dtype=np.float32),
            anchor_fiber_residual_degrees=np.zeros(2, dtype=np.float32),
            context_mode_count=np.asarray((3, 3), dtype=np.uint16),
            support_low_voxels=np.asarray((-10.0, -10.0), dtype=np.float32),
            support_high_voxels=np.asarray((10.0, 10.0), dtype=np.float32),
            normal_xyz=np.asarray(
                ((0.0, 0.0, 1.0), (0.0, 0.0, -1.0)), dtype=np.float32
            ),
            depth_offsets_voxels=depths,
            density=np.vstack((first_density, first_density[::-1])),
            orientation_moment=np.vstack(
                (first_moment, first_moment[::-1])
            ),
        )
        table.validate()
        score = score_patch_fingerprints(
            table,
            0,
            1,
            StratigraphicContinuitySettings(
                minimum_common_depth_span_voxels=5.0
            ),
        )
        self.assertEqual(score["status"], "scored")
        self.assertTrue(score["normalGaugeReversed"])
        self.assertAlmostEqual(score["mismatch"], 0.0, places=6)

    def test_full_mode_fingerprint_anchors_selected_plane_exactly(self) -> None:
        grid = GridSpec(
            (1, 1, 1),
            cell_size_xyz=(32.0, 32.0, 32.0),
            coordinate_unit="source-voxel",
        )
        estimate = PlaneEstimate.isotropic(
            (0.0, 0.0, 1.0),
            0.0,
            angular_std_radians=math.radians(1.0),
            height_std=0.5,
            fiber_xyz=(1.0, 0.0, 0.0),
            fiber_angular_std_radians=math.radians(2.0),
            confidence=0.9,
        )
        patch = clip_plane_to_cell(grid, (0, 0, 0), estimate, patch_id=7)
        assert patch is not None
        patches = PatchTable.from_patches(
            grid, (patch,), normal_family={7: 0}
        )
        covariance = np.tile(
            np.asarray(
                (
                    math.radians(1.0) ** 2,
                    0.0,
                    0.0,
                    math.radians(1.0) ** 2,
                    0.0,
                    0.25,
                ),
                dtype=np.float32,
            ),
            (3, 1),
        )
        modes = LayerModeTable(
            cell_xyz=np.asarray(((0, 0, 0),), dtype=np.int32),
            mode_offset=np.asarray((0, 3), dtype=np.uint64),
            normal_hypothesis=np.zeros(3, dtype=np.int8),
            normal_xyz=np.tile((0.0, 0.0, 1.0), (3, 1)).astype(np.float32),
            height=np.asarray((-8.0, 0.0, 8.0), dtype=np.float32),
            covariance=covariance,
            fiber_xyz=np.asarray(
                ((0.0, 1.0, 0.0), (1.0, 0.0, 0.0), (0.0, 1.0, 0.0)),
                dtype=np.float32,
            ),
            fiber_angular_std_radians=np.full(
                3, math.radians(2.0), dtype=np.float32
            ),
            confidence=np.full(3, 0.9, dtype=np.float32),
            source_depth_voxels=np.asarray((-8.0, 0.0, 8.0), dtype=np.float32),
            source_orientation_degrees=np.asarray((90.0, 0.0, 90.0), dtype=np.float32),
            evidence_score=np.full(3, 0.9, dtype=np.float32),
            material_probability=np.full(3, 0.8, dtype=np.float32),
            effective_support=np.full(3, 8.0, dtype=np.float32),
        )
        fingerprints, statistics = build_patch_fingerprints(
            patches,
            {"x0000-y0000-z0000": modes},
            RawAcusSettings(),
        )
        self.assertEqual(statistics["anchoredPatches"], 1)
        self.assertEqual(int(fingerprints.anchor_mode_index[0]), 1)
        self.assertEqual(int(fingerprints.context_mode_count[0]), 2)
        for depth in (-8.0, 8.0):
            index = int(np.argmin(np.abs(fingerprints.depth_offsets_voxels - depth)))
            self.assertGreater(fingerprints.density[0, index], 0.5)
            self.assertLess(fingerprints.orientation_moment[0, index], -0.5)

    def test_stratigraphic_gate_requires_local_and_multicell_outliers(self) -> None:
        records = []
        for index in range(40):
            local = 0.1
            neighborhood = 0.1
            if index in (37, 39):
                local = 0.9
            if index in (38, 39):
                neighborhood = 0.9
            records.append(
                {
                    "key": (index + 1, index + 101, 0, (index, 0, 0)),
                    "local": {"status": "scored", "mismatch": local},
                    "neighborhood": {
                        "status": "scored",
                        "mismatch": neighborhood,
                    },
                }
            )
        calibration = _calibrate_records(
            records, StratigraphicContinuitySettings()
        )
        self.assertEqual(calibration["0"]["state"], "calibrated")
        self.assertFalse(records[37]["rejected"])
        self.assertFalse(records[38]["rejected"])
        self.assertTrue(records[39]["rejected"])

    def test_axis_aligned_plane_clips_to_four_edge_loop(self) -> None:
        grid = GridSpec((2, 2, 2))
        estimate = PlaneEstimate.isotropic(
            (0.0, 0.0, 1.0),
            0.0,
            angular_std_radians=0.0,
            height_std=0.1,
            fiber_xyz=(1.0, 0.0, 0.0),
        )
        patch = clip_plane_to_cell(grid, (0, 0, 0), estimate, patch_id=7)
        self.assertIsNotNone(patch)
        assert patch is not None
        self.assertEqual(patch.patch_id, 7)
        self.assertEqual(len(patch.vertices), 4)
        self.assertEqual(len(patch.traces), 4)
        self.assertEqual({vertex.edge.axis for vertex in patch.vertices}, {2})
        np.testing.assert_allclose([vertex.t for vertex in patch.vertices], 0.5)
        np.testing.assert_allclose(
            [vertex.variance for vertex in patch.vertices], 0.01
        )
        self.assertEqual({trace.face.axis for trace in patch.traces}, {0, 1})

    def test_tangent_atlas_preserves_a_perforated_planar_component(self) -> None:
        grid = GridSpec((3, 3, 1))
        patches = tuple(
            self._horizontal_patch(grid, (x, y, 0), 0.0, 10 + 3 * y + x)
            for y in range(3)
            for x in range(3)
            if (x, y) != (1, 1)
        )
        block = assemble_surface_hierarchy(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            patches,
            maximum_leaf_shape_cells_xyz=(2, 2, 1),
        )
        component = max(block.components, key=lambda value: len(value.patch_ids))
        mesh = component_mesh(block, component.component_id)
        chart = tangent_atlas_chart(mesh)
        raster = rasterize_chart(
            mesh, chart, pixel_step_voxels=0.05, maximum_pixels=256
        )
        self.assertEqual(len(component.patch_ids), 8)
        self.assertEqual(mesh.statistics["orientationConflicts"], 0)
        self.assertGreaterEqual(mesh.statistics["chartCycleSeamEdges"], 1)
        self.assertEqual(chart.statistics["flippedTriangles"], 0)
        self.assertEqual(raster.statistics["nonadjacentOverlapPixels"], 0)
        self.assertEqual(set(np.unique(raster.patch_id[raster.mask])), set(component.patch_ids))

    def test_flattened_depth_stack_uses_one_fixed_native_ct_offset(self) -> None:
        grid = GridSpec(
            (1, 1, 1),
            cell_size_xyz=(8.0, 8.0, 8.0),
            origin_xyz=(8.0, 8.0, 12.0),
        )
        patch = self._horizontal_patch(grid, (0, 0, 0), 0.0, 1)
        block = assemble_surface_hierarchy(
            grid,
            BlockBounds((0, 0, 0), (1, 1, 1)),
            (patch,),
            maximum_leaf_shape_cells_xyz=(1, 1, 1),
        )
        mesh = component_mesh(block, block.components[0].component_id)
        chart = tangent_atlas_chart(mesh)
        raster = rasterize_chart(
            mesh, chart, pixel_step_voxels=0.5, maximum_pixels=128
        )
        volume = np.broadcast_to(
            (5 * np.arange(32, dtype=np.uint8))[:, None, None],
            (32, 32, 32),
        ).copy()
        with tempfile.TemporaryDirectory() as directory:
            source_path = Path(directory) / "volume.npy"
            np.save(source_path, volume)
            source = VolumeSource.open(source_path)
            stack, statistics = sample_depth_stack(source, raster, (-1.0, 0.0, 1.0))
        medians = sorted(
            float(np.median(plane[raster.mask])) for plane in stack
        )
        np.testing.assert_allclose(medians, (75.0, 80.0, 85.0), atol=0.5)
        self.assertEqual(statistics["depthOffsetsVoxels"], [-1.0, 0.0, 1.0])

    def test_join_continuity_compares_against_equal_span_controls(self) -> None:
        grid = GridSpec(
            (2, 1, 1),
            cell_size_xyz=(8.0, 8.0, 8.0),
            origin_xyz=(8.0, 8.0, 8.0),
        )
        patches = tuple(
            self._horizontal_patch(grid, (x, 0, 0), 0.0, x + 1)
            for x in range(2)
        )
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (2, 1, 1)), patches
        )
        z_index, _, x_index = np.indices((32, 32, 32))
        continuous = np.clip(20 + 2 * x_index + 3 * z_index, 0, 255).astype(
            np.uint8
        )
        discontinuous = np.clip(
            continuous.astype(np.int16) + 50 * (x_index >= 16), 0, 255
        ).astype(np.uint8)
        ratios = []
        texture_angles = []
        with tempfile.TemporaryDirectory() as directory:
            for name, volume in (
                ("continuous", continuous),
                ("discontinuous", discontinuous),
            ):
                source_path = Path(directory) / f"{name}.npy"
                np.save(source_path, volume)
                records = score_join_continuity(
                    block, VolumeSource.open(source_path)
                )
                self.assertEqual(len(records), 1)
                ratios.append(records[0]["mismatchRatio"])
                texture_angles.append(records[0]["surfaceTextureAngleDegrees"])
            profile = np.asarray(
                [
                    21, 32, 48, 79, 116, 143, 128, 88,
                    49, 37, 61, 105, 151, 173, 139, 82,
                    44, 53, 97, 148, 164, 121, 68, 35,
                    46, 91, 137, 155, 109, 57, 31, 42,
                ],
                dtype=np.uint8,
            )
            shifted_z = np.clip(z_index - 4, 0, len(profile) - 1)
            shifted_profile = np.where(
                x_index < 16,
                profile[z_index],
                profile[shifted_z],
            ).astype(np.uint8)
            source_path = Path(directory) / "shifted-profile.npy"
            np.save(source_path, shifted_profile)
            shifted_record = score_join_continuity(
                block, VolumeSource.open(source_path)
            )[0]
        self.assertAlmostEqual(ratios[0], 1.0, places=5)
        self.assertGreater(ratios[1], 8.0)
        np.testing.assert_allclose(texture_angles, (0.0, 0.0), atol=1.0e-5)
        self.assertAlmostEqual(
            abs(shifted_record["bestDepthShiftVoxels"]), 4.0
        )
        self.assertEqual(shifted_record["firstControlDepthShiftVoxels"], 0.0)
        self.assertEqual(shifted_record["secondControlDepthShiftVoxels"], 0.0)
        self.assertGreater(shifted_record["excessDepthShiftCorrelationGain"], 0.5)

    def test_oblique_plane_supports_triangle_and_hexagon_topologies(self) -> None:
        grid = GridSpec((1, 1, 1))
        triangle = clip_plane_to_cell(
            grid,
            (0, 0, 0),
            PlaneEstimate.isotropic(
                (1.0, 1.0, 1.0), 0.7, math.radians(1.0), 0.02
            ),
        )
        hexagon = clip_plane_to_cell(
            grid,
            (0, 0, 0),
            PlaneEstimate.isotropic(
                (1.0, 1.0, 1.0), 0.0, math.radians(1.0), 0.02
            ),
        )
        self.assertIsNotNone(triangle)
        self.assertIsNotNone(hexagon)
        assert triangle is not None and hexagon is not None
        self.assertEqual(len(triangle.vertices), 3)
        self.assertEqual(len(hexagon.vertices), 6)

    def test_cell_centered_plane_is_translation_invariant(self) -> None:
        first_grid = GridSpec(
            (4, 4, 4), cell_size_xyz=(2.0, 3.0, 5.0), origin_xyz=(0.0, 0.0, 0.0)
        )
        second_grid = GridSpec(
            (4, 4, 4),
            cell_size_xyz=(2.0, 3.0, 5.0),
            origin_xyz=(1000.0, -250.0, 88.0),
        )
        estimate = PlaneEstimate.isotropic(
            (0.2, 0.4, 0.9), 0.13, math.radians(2.0), 0.04
        )
        first = clip_plane_to_cell(first_grid, (2, 1, 0), estimate)
        second = clip_plane_to_cell(second_grid, (2, 1, 0), estimate)
        self.assertIsNotNone(first)
        self.assertIsNotNone(second)
        assert first is not None and second is not None
        self.assertEqual(
            [(value.edge, value.t) for value in first.vertices],
            [(value.edge, value.t) for value in second.vertices],
        )
        translation = np.asarray(second_grid.origin_xyz) - np.asarray(
            first_grid.origin_xyz
        )
        np.testing.assert_allclose(
            np.asarray([value.point_xyz for value in second.vertices])
            - np.asarray([value.point_xyz for value in first.vertices]),
            np.broadcast_to(translation, (len(first.vertices), 3)),
        )

    def test_unsigned_plane_gauge_flips_height_with_normal(self) -> None:
        positive = PlaneEstimate.isotropic(
            (0.0, 0.0, 1.0), 0.2, 0.01, 0.02
        )
        negative = PlaneEstimate.isotropic(
            (0.0, 0.0, -1.0), -0.2, 0.01, 0.02
        )
        self.assertEqual(positive.normal_xyz, negative.normal_xyz)
        self.assertAlmostEqual(
            positive.height_from_cell_center, negative.height_from_cell_center
        )
        self.assertAlmostEqual(
            axial_angle_radians((1.0, 0.0, 0.0), (-1.0, 0.0, 0.0)), 0.0
        )

    def test_boundary_coincident_plane_is_explicitly_degenerate(self) -> None:
        grid = GridSpec((1, 1, 1))
        with self.assertRaises(DegeneratePlaneIntersection):
            clip_plane_to_cell(
                grid,
                (0, 0, 0),
                PlaneEstimate.isotropic((0.0, 0.0, 1.0), 0.5, 0.01, 0.01),
            )

    def test_shared_face_trace_match_uses_canonical_edge_crossings(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(grid, (0, 0, 0), 0.1, 1)
        second = self._horizontal_patch(grid, (1, 0, 0), 0.1, 2)
        face = cell_face((0, 0, 0), 0, 1)
        result = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
        )
        self.assertTrue(result.accepted)
        self.assertFalse(result.failure_reasons)
        self.assertEqual(len(result.endpoint_agreements), 2)
        self.assertAlmostEqual(result.reduced_chi_square, 0.0)

    def test_shared_face_trace_match_rejects_distinct_layer_height(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(
            grid, (0, 0, 0), -0.2, 1, height_std=0.005
        )
        second = self._horizontal_patch(
            grid, (1, 0, 0), 0.2, 2, height_std=0.005
        )
        face = cell_face((0, 0, 0), 0, 1)
        result = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
        )
        self.assertFalse(result.accepted)
        self.assertIn("endpoint", result.failure_reasons)

    def test_orthogonal_fiber_equivalence_is_explicit_packet_semantics(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(grid, (0, 0, 0), 0.1, 1)
        second = clip_plane_to_cell(
            grid,
            (1, 0, 0),
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                0.1,
                angular_std_radians=math.radians(1.0),
                height_std=0.02,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=2,
        )
        assert second is not None
        face = cell_face((0, 0, 0), 0, 1)
        strict = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
            grid=grid,
        )
        packet = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
            TraceMatchSettings(orthogonal_fiber_equivalence=True),
            grid=grid,
        )
        self.assertFalse(strict.accepted)
        self.assertIn("fiber", strict.failure_reasons)
        self.assertFalse(strict.fiber_quarter_turn)
        self.assertTrue(packet.accepted)
        self.assertTrue(packet.fiber_quarter_turn)
        self.assertIsNotNone(packet.fiber_angle_radians)
        self.assertAlmostEqual(float(packet.fiber_angle_radians), 0.0)

    def test_packet_join_extension_preserves_strict_connectivity(self) -> None:
        grid = GridSpec((3, 1, 1))
        first = self._horizontal_patch(grid, (0, 0, 0), 0.1, 1)
        second = self._horizontal_patch(grid, (1, 0, 0), 0.1, 2)
        third = clip_plane_to_cell(
            grid,
            (2, 0, 0),
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                0.1,
                angular_std_radians=math.radians(1.0),
                height_std=0.02,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=3,
        )
        assert third is not None
        strict = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (3, 1, 1)), (first, second, third)
        )
        self.assertEqual(len(strict.joins), 1)
        face = cell_face((1, 0, 0), 0, 1)
        quarter_turn = match_face_traces(
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
            third.trace_on(face),  # type: ignore[arg-type]
            third.estimate,
            TraceMatchSettings(orthogonal_fiber_equivalence=True),
            grid=grid,
        )
        extended = extend_surface_block_joins(strict, (quarter_turn,))
        self.assertEqual(len(extended.joins), 2)
        self.assertEqual(len(extended.components), 1)
        self.assertTrue(
            {
                (value.first_patch_id, value.second_patch_id)
                for value in strict.joins
            }.issubset(
                {
                    (value.first_patch_id, value.second_patch_id)
                    for value in extended.joins
                }
            )
        )

    def test_face_alignment_is_ordered_and_permutation_invariant(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = [
            self._horizontal_patch(grid, (0, 0, 0), -0.25, 10),
            self._horizontal_patch(grid, (0, 0, 0), 0.0, 11),
            self._horizontal_patch(grid, (0, 0, 0), 0.25, 12),
        ]
        second = [
            self._horizontal_patch(grid, (1, 0, 0), 0.25, 22),
            self._horizontal_patch(grid, (1, 0, 0), -0.25, 20),
        ]
        result = align_face_patches(
            reversed(first),
            second,
            cell_face((0, 0, 0), 0, 1),
        )
        self.assertEqual(
            [(value.first_patch_id, value.second_patch_id) for value in result.matches],
            [(10, 20), (12, 22)],
        )
        self.assertEqual(result.unmatched_first_patch_ids, (11,))
        self.assertFalse(result.unmatched_second_patch_ids)

    def test_face_alignment_reuses_geometry_matches_without_leaking_patch_ids(self) -> None:
        grid = GridSpec((2, 1, 1))
        face = cell_face((0, 0, 0), 0, 1)
        cache = {}
        first = self._horizontal_patch(grid, (0, 0, 0), 0.1, 1)
        second = self._horizontal_patch(grid, (1, 0, 0), 0.1, 2)
        initial = align_face_patches(
            (first,), (second,), face, grid=grid, _match_cache=cache
        )
        cache_size = len(cache)
        repeated = align_face_patches(
            (self._horizontal_patch(grid, (0, 0, 0), 0.1, 101),),
            (self._horizontal_patch(grid, (1, 0, 0), 0.1, 102),),
            face,
            grid=grid,
            _match_cache=cache,
        )
        self.assertEqual(cache_size, 1)
        self.assertEqual(len(cache), cache_size)
        self.assertEqual(initial.matches[0].first_patch_id, 1)
        self.assertEqual(initial.matches[0].second_patch_id, 2)
        self.assertEqual(repeated.matches[0].first_patch_id, 101)
        self.assertEqual(repeated.matches[0].second_patch_id, 102)

    def test_trace_matching_rejects_different_face_topology(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(grid, (0, 0, 0), 0.0, 1)
        second = clip_plane_to_cell(
            grid,
            (1, 0, 0),
            PlaneEstimate.isotropic(
                (0.0, 1.0, 0.0),
                0.1,
                angular_std_radians=math.radians(1.0),
                height_std=0.02,
                fiber_xyz=(1.0, 0.0, 0.0),
            ),
            patch_id=2,
        )
        assert second is not None
        face = cell_face((0, 0, 0), 0, 1)
        result = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
        )
        self.assertFalse(result.accepted)
        self.assertIn("edge-topology", result.failure_reasons)

    def test_uncertain_corner_transition_welds_to_grid_vertex(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(
            grid, (0, 0, 0), 0.49, 1, height_std=0.02
        )
        slope = 0.01 / 0.98
        normal = np.asarray((0.0, -slope, 1.0), dtype=np.float64)
        normal /= np.linalg.norm(normal)
        point = np.asarray((1.0, 0.0, 0.99))
        second_center = grid.cell_center_world((1, 0, 0))
        second = clip_plane_to_cell(
            grid,
            (1, 0, 0),
            PlaneEstimate.isotropic(
                normal,
                float(np.dot(normal, point - second_center)),
                angular_std_radians=math.radians(1.0),
                height_std=0.02,
                fiber_xyz=(1.0, 0.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=2,
        )
        assert second is not None
        face = cell_face((0, 0, 0), 0, 1)
        match = match_face_traces(
            first.trace_on(face),  # type: ignore[arg-type]
            first.estimate,
            second.trace_on(face),  # type: ignore[arg-type]
            second.estimate,
            grid=grid,
        )
        self.assertTrue(match.accepted)
        self.assertEqual(
            sorted(value.mode for value in match.endpoint_agreements),
            ["same-edge", "shared-corner"],
        )
        block = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), (2, 1, 1)),
            (first, second),
        )
        self.assertEqual(len(block.components), 1)
        corner = [
            value
            for value in block.welded_crossings
            if value.grid_vertex_xyz == (1, 1, 1)
        ]
        self.assertEqual(len(corner), 1)
        self.assertIsNone(corner[0].edge)
        np.testing.assert_allclose(corner[0].point_xyz, (1.0, 1.0, 1.0))

    def test_block_assembly_welds_four_incident_cell_observations(self) -> None:
        grid = GridSpec((2, 2, 1))
        patches = [
            self._horizontal_patch(grid, (x_index, y_index, 0), 0.0, 2 * y_index + x_index)
            for y_index in range(2)
            for x_index in range(2)
        ]
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (2, 2, 1)), patches
        )
        self.assertEqual(len(block.joins), 4)
        self.assertEqual(len(block.components), 1)
        self.assertEqual(block.components[0].patch_ids, (0, 1, 2, 3))
        self.assertEqual(len(block.exterior_traces), 8)
        self.assertFalse(block.unresolved_interior_traces)
        self.assertEqual(len(block.welded_crossings), 9)
        center = [
            value
            for value in block.welded_crossings
            if value.edge is not None
            and value.edge.axis == 2
            and value.edge.anchor_xyz == (1, 1, 0)
        ]
        self.assertEqual(len(center), 1)
        self.assertEqual(len(center[0].observations), 4)
        self.assertAlmostEqual(center[0].t, 0.5)

    def test_block_assembly_preserves_two_ordered_surface_components(self) -> None:
        grid = GridSpec((2, 2, 1))
        patches = []
        for y_index in range(2):
            for x_index in range(2):
                cell_index = 2 * y_index + x_index
                patches.extend(
                    (
                        self._horizontal_patch(
                            grid, (x_index, y_index, 0), -0.2, 2 * cell_index
                        ),
                        self._horizontal_patch(
                            grid, (x_index, y_index, 0), 0.2, 2 * cell_index + 1
                        ),
                    )
                )
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (2, 2, 1)), reversed(patches)
        )
        self.assertEqual(len(block.components), 2)
        self.assertEqual([len(value.patch_ids) for value in block.components], [4, 4])
        self.assertEqual(len(block.joins), 8)
        self.assertEqual(len(block.welded_crossings), 18)
        self.assertFalse(block.unresolved_interior_traces)

    def test_post_assembly_refinement_only_removes_retained_joins(self) -> None:
        grid = GridSpec((3, 1, 1))
        patches = tuple(
            self._horizontal_patch(grid, (x, 0, 0), 0.0, x + 1)
            for x in range(3)
        )
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (3, 1, 1)), patches
        )
        refined = rebuild_surface_block(block, block.joins[:1])
        self.assertEqual(len(block.joins), 2)
        self.assertEqual(len(refined.joins), 1)
        self.assertEqual(
            sorted(len(value.patch_ids) for value in refined.components),
            [1, 2],
        )
        with self.assertRaises(ValueError):
            rebuild_surface_block(block, (*block.joins, block.joins[0]))

    def test_incompatible_neighbor_traces_remain_explicit_open_seams(self) -> None:
        grid = GridSpec((2, 1, 1))
        patches = (
            self._horizontal_patch(
                grid, (0, 0, 0), -0.2, 0, height_std=0.005
            ),
            self._horizontal_patch(
                grid, (1, 0, 0), 0.2, 1, height_std=0.005
            ),
        )
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (2, 1, 1)), patches
        )
        self.assertFalse(block.joins)
        self.assertEqual(len(block.components), 2)
        self.assertEqual(len(block.unresolved_interior_traces), 2)

    def test_gap_census_and_mode_bank_recover_an_explicit_missing_neighbor(self) -> None:
        grid = GridSpec((2, 1, 1))
        first = self._horizontal_patch(grid, (0, 0, 0), 0.1, 1)
        second = self._horizontal_patch(grid, (1, 0, 0), 0.1, 2)
        block = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (2, 1, 1)), (first,)
        )
        options = {
            (0, 0, 0): (
                ConfigurationOption(0, (0, 0, 0), 0, 0, 0, -0.1, (first,), 0),
            ),
            (1, 0, 0): (
                ConfigurationOption(1, (1, 0, 0), 1, 0, 0, -0.1, (), 0),
                ConfigurationOption(2, (1, 0, 0), 1, 1, 1, -0.2, (second,), 0),
            ),
        }
        selected = {(0, 0, 0): 0, (1, 0, 0): 1}
        census = analyze_component_gaps(block, options, selected)
        self.assertEqual(len(census.traces), 1)
        self.assertEqual(
            census.traces[0].classification, "recoverable-configuration-gap"
        )
        repair = evaluate_single_cell_gap_repairs(
            block,
            options,
            selected,
            census,
            maximum_leaf_shape_cells_xyz=(1, 1, 1),
        )
        self.assertEqual(len(repair.trials), 1)
        self.assertTrue(repair.trials[0].recommended)
        self.assertEqual(repair.trials[0].closed_gap_count, 1)

        estimate = second.estimate
        covariance = estimate.covariance_matrix
        mode_table = LayerModeTable(
            np.asarray([[1, 0, 0]], dtype=np.int32),
            np.asarray([0, 1], dtype=np.uint64),
            np.asarray([0], dtype=np.int8),
            np.asarray([estimate.normal_xyz], dtype=np.float32),
            np.asarray([estimate.height_from_cell_center], dtype=np.float32),
            np.asarray(
                [[
                    covariance[0, 0],
                    covariance[0, 1],
                    covariance[0, 2],
                    covariance[1, 1],
                    covariance[1, 2],
                    covariance[2, 2],
                ]],
                dtype=np.float32,
            ),
            np.asarray([estimate.fiber_xyz], dtype=np.float32),
            np.asarray([estimate.fiber_angular_std_radians], dtype=np.float32),
            np.asarray([estimate.confidence], dtype=np.float32),
            np.asarray([0.1], dtype=np.float32),
            np.asarray([2.5], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([8.0], dtype=np.float32),
        )
        mode_census = analyze_component_gaps(
            block,
            {
                (0, 0, 0): options[(0, 0, 0)],
                (1, 0, 0): (options[(1, 0, 0)][0],),
            },
            selected,
        )
        self.assertEqual(mode_census.traces[0].classification, "mode-gap")
        discovery = discover_mode_continuations(
            block, mode_census, {"test": mode_table}
        )
        self.assertEqual(discovery.mode_gap_count, 1)
        self.assertEqual(discovery.matched_gap_count, 1)
        self.assertEqual(len(discovery.candidates), 1)

    def test_contextual_growth_discovers_and_incrementally_adds_two_face_mode(self) -> None:
        grid = GridSpec((2, 2, 1))
        left = self._horizontal_patch(grid, (0, 1, 0), 0.1, 1)
        lower = self._horizontal_patch(grid, (1, 0, 0), 0.1, 2)
        target = self._horizontal_patch(grid, (1, 1, 0), 0.1, 30)
        baseline = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            (left, lower),
        )
        estimate = target.estimate
        covariance = estimate.covariance_matrix
        modes = LayerModeTable(
            np.asarray(((1, 1, 0),), dtype=np.int32),
            np.asarray((0, 1), dtype=np.uint64),
            np.asarray((0,), dtype=np.int8),
            np.asarray((estimate.normal_xyz,), dtype=np.float32),
            np.asarray((estimate.height_from_cell_center,), dtype=np.float32),
            np.asarray(
                ((
                    covariance[0, 0],
                    covariance[0, 1],
                    covariance[0, 2],
                    covariance[1, 1],
                    covariance[1, 2],
                    covariance[2, 2],
                ),),
                dtype=np.float32,
            ),
            np.asarray((estimate.fiber_xyz,), dtype=np.float32),
            np.asarray((estimate.fiber_angular_std_radians,), dtype=np.float32),
            np.asarray((estimate.confidence,), dtype=np.float32),
            np.asarray((0.1,), dtype=np.float32),
            np.asarray((2.5,), dtype=np.float32),
            np.asarray((0.8,), dtype=np.float32),
            np.asarray((0.9,), dtype=np.float32),
            np.asarray((8.0,), dtype=np.float32),
        )
        candidates, statistics = discover_contextual_growth_candidates(
            baseline,
            {"test": modes},
            ContextualGrowthSettings(),
        )
        self.assertEqual(statistics["multiFaceCandidateModes"], 1)
        self.assertEqual(len(candidates), 1)
        self.assertEqual(len(candidates[0].supports), 2)

        allowed = {
            (30, 1, cell_face((0, 1, 0), 0, 1)),
            (30, 2, cell_face((1, 0, 0), 1, 1)),
        }
        augmented = augment_surface_block(
            baseline, (target,), allowed_supports=allowed
        )
        direct = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            (left, lower, target),
        )
        self.assertEqual(len(augmented.joins), 2)
        self.assertEqual(
            {_join.face for _join in augmented.joins},
            {_join.face for _join in direct.joins},
        )
        self.assertEqual(len(augmented.components), 1)

    def test_hierarchical_block_merge_matches_direct_assembly(self) -> None:
        grid = GridSpec((4, 2, 1))
        patches = [
            self._horizontal_patch(
                grid, (x_index, y_index, 0), 0.05, 2 * x_index + y_index
            )
            for x_index in range(4)
            for y_index in range(2)
        ]
        left = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), (2, 2, 1)),
            [value for value in patches if value.cell_xyz[0] < 2],
        )
        right = assemble_surface_block(
            grid,
            BlockBounds((2, 0, 0), (4, 2, 1)),
            [value for value in patches if value.cell_xyz[0] >= 2],
        )
        merged = merge_surface_blocks(left, right)
        direct = assemble_surface_block(
            grid, BlockBounds((0, 0, 0), (4, 2, 1)), patches
        )
        self.assertEqual(len(merged.joins), len(direct.joins))
        self.assertEqual(len(merged.components), len(direct.components))
        self.assertEqual(
            len(merged.welded_crossings), len(direct.welded_crossings)
        )
        self.assertEqual(
            len(merged.exterior_traces), len(direct.exterior_traces)
        )
        self.assertEqual(
            merged.components[0].patch_ids, direct.components[0].patch_ids
        )
        self.assertFalse(merged.unresolved_interior_traces)

        recursive = assemble_surface_hierarchy(
            grid,
            BlockBounds((0, 0, 0), (4, 2, 1)),
            patches,
            maximum_leaf_shape_cells_xyz=(1, 1, 1),
        )
        self.assertEqual(len(recursive.joins), len(direct.joins))
        self.assertEqual(
            recursive.components[0].patch_ids, direct.components[0].patch_ids
        )
        self.assertEqual(
            len(recursive.welded_crossings), len(direct.welded_crossings)
        )

    def test_patch_table_round_trip_preserves_geometry_and_metadata(self) -> None:
        grid = GridSpec(
            (2, 1, 1),
            cell_size_xyz=(2.0, 3.0, 4.0),
            origin_xyz=(10.0, 20.0, 30.0),
            coordinate_unit="voxel",
        )
        patches = (
            self._horizontal_patch(grid, (0, 0, 0), -0.4, 7),
            self._horizontal_patch(grid, (1, 0, 0), 0.4, 11),
        )
        table = PatchTable.from_patches(
            grid,
            patches,
            configuration_id={7: 2, 11: 3},
            configuration_log_weight={7: -0.2, 11: -0.4},
            local_order={7: -1, 11: 1},
            normal_family={7: 0, 11: 2},
        )
        self.assertEqual(table.patch_count, 2)
        self.assertEqual(table.vertex_count, 8)
        self.assertEqual(table.vertex_offset.tolist(), [0, 4, 8])
        with tempfile.TemporaryDirectory() as directory:
            prefix = Path(directory) / "patch-shard-v1"
            manifest = write_patch_shard(
                prefix,
                table,
                settings={"source": "analytic-test"},
                provenance={"seed": 19},
            )
            restored = read_patch_shard(prefix)
        self.assertEqual(manifest["schema"], "pareidolia.cubical-patches")
        self.assertEqual(manifest["counts"]["patches"], 2)
        np.testing.assert_array_equal(restored.patch_id, [7, 11])
        np.testing.assert_array_equal(restored.configuration_id, [2, 3])
        np.testing.assert_array_equal(restored.local_order, [-1, 1])
        round_trip = restored.to_patches()
        self.assertEqual([value.patch_id for value in round_trip], [7, 11])
        for expected, actual in zip(patches, round_trip):
            self.assertEqual(
                [value.edge for value in expected.vertices],
                [value.edge for value in actual.vertices],
            )
            np.testing.assert_allclose(
                [value.t for value in expected.vertices],
                [value.t for value in actual.vertices],
                atol=2.0e-5,
            )

    def test_boundary_band_freezes_interior_and_preserves_exterior_identity(
        self,
    ) -> None:
        grid = GridSpec(
            (4, 4, 4),
            cell_size_xyz=(2.0, 3.0, 5.0),
            origin_xyz=(10.0, 20.0, 30.0),
            coordinate_unit="voxel",
        )
        patches = tuple(
            self._horizontal_patch(
                grid,
                (x, y, z),
                0.0,
                1 + x + 4 * y + 16 * z,
            )
            for z in range(4)
            for y in range(4)
            for x in range(4)
        )
        table = PatchTable.from_patches(grid, patches)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "selected"
            output = Path(directory) / "boundary"
            write_patch_shard(root / "selected-patches-v1", table)
            summary = run_boundary_band_export(
                root,
                output,
                settings=BoundaryBandSettings(depth_cells=1),
            )
            restored = read_patch_shard(output / "boundary-patches-v1")
            with np.load(output / "boundary-interface-v1.npz") as values:
                component_ids = np.asarray(values["componentId"])
                component_face_masks = np.asarray(
                    values["componentExteriorFaceMask"]
                )
                component_cell_count = len(values["componentCellXYZ"])
                crossing_feature_kind = np.asarray(
                    values["crossingGroupFeatureKind"]
                )
                trace_points = np.concatenate(
                    (
                        values["traceFirstPointWorldXYZ"],
                        values["traceSecondPointWorldXYZ"],
                    ),
                    axis=0,
                )
                trace_count = len(values["tracePatchId"])

        statistics = summary["statistics"]
        self.assertEqual(statistics["boundaryBandCells"], 56)
        self.assertEqual(statistics["boundaryCellsWithPatches"], 56)
        self.assertEqual(statistics["boundaryPatches"], 56)
        self.assertEqual(statistics["frozenInteriorCells"], 8)
        self.assertEqual(statistics["boundaryComponents"], 4)
        self.assertEqual(statistics["boundaryComponentOccupiedCells"], 64)
        self.assertEqual(statistics["boundaryCrossingGroups"], 64)
        self.assertEqual(statistics["exteriorTraces"], 64)
        self.assertIsNone(summary["artifacts"]["configurations"])
        self.assertEqual(restored.patch_count, 56)
        self.assertEqual(len(component_ids), 4)
        self.assertEqual(component_cell_count, 64)
        np.testing.assert_array_equal(component_face_masks, np.full(4, 15))
        np.testing.assert_array_equal(crossing_feature_kind, np.zeros(64))
        self.assertEqual(trace_count, 64)
        self.assertTrue(np.all(np.isfinite(trace_points)))
        self.assertGreaterEqual(float(np.min(trace_points[:, 0])), 10.0)
        self.assertGreaterEqual(float(np.min(trace_points[:, 1])), 20.0)
        self.assertGreaterEqual(float(np.min(trace_points[:, 2])), 30.0)

    def test_boundary_band_preserves_selected_physical_alternative_index(
        self,
    ) -> None:
        grid = GridSpec((4, 4, 4))
        cells = np.asarray(
            [
                (x, y, z)
                for z in range(4)
                for y in range(4)
                for x in range(4)
            ],
            dtype=np.int32,
        )
        count = len(cells)
        candidates = ConfigurationTable(
            cells,
            np.arange(count + 1, dtype=np.uint64),
            np.zeros(count, dtype=np.uint16),
            np.zeros(count, dtype=np.float32),
            np.zeros(count, dtype=np.int8),
            np.arange(count + 1, dtype=np.uint64),
            np.tile((0.0, 0.0, 1.0), (count, 1)).astype(np.float32),
            np.zeros(count, dtype=np.float32),
            np.tile(
                (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                (count, 1),
            ).astype(np.float32),
            np.tile((1.0, 0.0, 0.0), (count, 1)).astype(np.float32),
            np.full(count, math.radians(2.0), dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
            np.ones(count, dtype=np.float32),
        )
        candidates.validate()
        patches = tuple(
            self._horizontal_patch(
                grid,
                tuple(int(value) for value in cell),
                0.0,
                index + 1,
            )
            for index, cell in enumerate(cells)
        )
        with tempfile.TemporaryDirectory() as directory:
            selected_root = Path(directory) / "selected"
            candidate_root = Path(directory) / "candidates"
            output = Path(directory) / "boundary"
            write_patch_shard(
                selected_root / "selected-patches-v1",
                PatchTable.from_patches(grid, patches),
            )
            candidate_root.mkdir(parents=True)
            candidate_path = candidate_root / "saturation-configurations-v1.npz"
            metadata = {
                name: np.ones(count, dtype=np.float32)
                for name in (
                    "evidenceLogScore",
                    "physicalLogScore",
                    "totalLogScore",
                    "coveredEvidenceMass",
                    "totalEvidenceMass",
                )
            }
            metadata["isCurrent"] = np.ones(count, dtype=np.uint8)
            with candidate_path.open("wb") as handle:
                np.savez_compressed(handle, **candidates.arrays(), **metadata)
            candidate_sha = sha256_file(candidate_path)
            (candidate_root / "saturation-configurations-v1.json").write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.cubical-saturation-configurations",
                        "version": 1,
                        "identitySha256": "analytic-candidates",
                        "data": {
                            "path": candidate_path.name,
                            "bytes": candidate_path.stat().st_size,
                            "sha256": candidate_sha,
                        },
                    }
                )
            )
            selection_path = selected_root / "selection-v1.npz"
            with selection_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cellXYZ=cells,
                    optionId=np.arange(count, dtype=np.uint64),
                    sourceTableIndex=np.zeros(count, dtype=np.uint32),
                    sourceConfigurationIndex=np.arange(count, dtype=np.uint32),
                    localConfigurationId=np.zeros(count, dtype=np.uint16),
                    configurationLogWeight=np.zeros(count, dtype=np.float32),
                    selectedLayerCount=np.ones(count, dtype=np.uint16),
                )
            selection_sha = sha256_file(selection_path)
            (selected_root / "selection-v1.json").write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.raw-acus-configuration-selection",
                        "version": 1,
                        "data": {
                            "path": selection_path.name,
                            "bytes": selection_path.stat().st_size,
                            "sha256": selection_sha,
                        },
                    }
                )
            )
            (selected_root / "variant.json").write_text(
                json.dumps(
                    {
                        "identity": {"candidateDataSha256": candidate_sha}
                    }
                )
            )
            summary = run_boundary_band_export(
                selected_root,
                output,
                candidate_root=candidate_root,
                settings=BoundaryBandSettings(depth_cells=1),
            )
            with np.load(output / "boundary-configurations-v1.npz") as values:
                selected_index = np.asarray(
                    values["selectedConfigurationIndex"]
                )
                source_index = np.asarray(values["sourceConfigurationIndex"])
                selected_source = np.asarray(
                    values["selectedSourceConfigurationIndex"]
                )

        self.assertEqual(
            summary["artifacts"]["configurations"]["selectedCells"], 56
        )
        self.assertEqual(len(selected_index), 56)
        np.testing.assert_array_equal(
            source_index[selected_index], selected_source
        )

    def test_boundary_merge_rebases_world_adjacent_blocks_and_freezes_interiors(
        self,
    ) -> None:
        grids = (
            GridSpec((4, 4, 4), origin_xyz=(0.0, 0.0, 0.0)),
            GridSpec((4, 4, 4), origin_xyz=(4.0, 0.0, 0.0)),
        )
        with tempfile.TemporaryDirectory() as directory:
            roots = []
            for block_index, grid in enumerate(grids):
                patches = tuple(
                    self._horizontal_patch(
                        grid,
                        (x, y, z),
                        0.0,
                        1 + x + 4 * y + 16 * z,
                    )
                    for z in range(4)
                    for y in range(4)
                    for x in range(4)
                )
                selected = Path(directory) / f"selected-{block_index}"
                boundary = Path(directory) / f"boundary-{block_index}"
                write_patch_shard(
                    selected / "selected-patches-v1",
                    PatchTable.from_patches(grid, patches),
                )
                run_boundary_band_export(
                    selected,
                    boundary,
                    settings=BoundaryBandSettings(depth_cells=1),
                )
                roots.append(boundary)
            output = Path(directory) / "merge"
            summary = run_boundary_band_merge(roots[0], roots[1], output)
            with np.load(output / "boundary-merge-v1.npz") as values:
                pair_support = np.asarray(values["pairSupportMatchCount"])
                pair_disposition = np.asarray(values["pairDisposition"])
                selected_bridges = np.asarray(values["matchSelectedBridge"])

        self.assertEqual(summary["grid"]["shapeCellsXYZ"], [8, 4, 4])
        self.assertEqual(summary["adjacency"]["axis"], 0)
        self.assertEqual(summary["statistics"]["seamUnitFaces"], 16)
        self.assertEqual(summary["statistics"]["alignedMatches"], 16)
        self.assertEqual(summary["statistics"]["unmatchedTraces"], 0)
        self.assertEqual(summary["statistics"]["componentPairHypotheses"], 4)
        self.assertEqual(summary["statistics"]["retainedComponentBridges"], 4)
        np.testing.assert_array_equal(pair_support, np.full(4, 4))
        np.testing.assert_array_equal(
            pair_disposition,
            np.full(4, "retained-forest-bridge"),
        )
        self.assertEqual(int(np.sum(selected_bridges)), 4)

    def test_boundary_reselection_uses_candidates_and_frozen_anchor_topology(
        self,
    ) -> None:
        grids = (
            GridSpec((4, 4, 4), origin_xyz=(0.0, 0.0, 0.0)),
            GridSpec((4, 4, 4), origin_xyz=(4.0, 0.0, 0.0)),
        )
        with tempfile.TemporaryDirectory() as directory:
            boundary_roots: list[Path] = []
            for side, grid in enumerate(grids):
                cells = np.asarray(
                    [
                        (x, y, z)
                        for z in range(4)
                        for y in range(4)
                        for x in range(4)
                    ],
                    dtype=np.int32,
                )
                count = len(cells)
                patch_start = 1 + side * count
                patches = tuple(
                    self._horizontal_patch(
                        grid,
                        tuple(int(value) for value in cell),
                        0.0,
                        patch_start + index,
                    )
                    for index, cell in enumerate(cells)
                )
                selected = Path(directory) / f"selected-{side}"
                boundary = Path(directory) / f"boundary-{side}"
                write_patch_shard(
                    selected / "selected-patches-v1",
                    PatchTable.from_patches(grid, patches),
                )
                candidates = ConfigurationTable(
                    cells,
                    np.arange(count + 1, dtype=np.uint64),
                    np.zeros(count, dtype=np.uint16),
                    np.zeros(count, dtype=np.float32),
                    np.zeros(count, dtype=np.int8),
                    np.arange(count + 1, dtype=np.uint64),
                    np.tile((0.0, 0.0, 1.0), (count, 1)).astype(np.float32),
                    np.zeros(count, dtype=np.float32),
                    np.tile(
                        (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                        (count, 1),
                    ).astype(np.float32),
                    np.tile((1.0, 0.0, 0.0), (count, 1)).astype(np.float32),
                    np.full(count, math.radians(2.0), dtype=np.float32),
                    np.ones(count, dtype=np.float32),
                    np.ones(count, dtype=np.float32),
                    np.ones(count, dtype=np.float32),
                    np.ones(count, dtype=np.float32),
                )
                candidate_path = selected / "saturation-configurations-v1.npz"
                metadata = {
                    name: np.ones(count, dtype=np.float32)
                    for name in (
                        "evidenceLogScore",
                        "physicalLogScore",
                        "totalLogScore",
                        "coveredEvidenceMass",
                        "totalEvidenceMass",
                    )
                }
                metadata["isCurrent"] = np.ones(count, dtype=np.uint8)
                with candidate_path.open("wb") as handle:
                    np.savez_compressed(handle, **candidates.arrays(), **metadata)
                candidate_sha = sha256_file(candidate_path)
                (selected / "saturation-configurations-v1.json").write_text(
                    json.dumps(
                        {
                            "schema": "pareidolia.cubical-saturation-configurations",
                            "version": 1,
                            "identitySha256": f"analytic-{side}",
                            "data": {
                                "path": candidate_path.name,
                                "bytes": candidate_path.stat().st_size,
                                "sha256": candidate_sha,
                            },
                        }
                    )
                )
                selection_path = selected / "selection-v1.npz"
                with selection_path.open("wb") as handle:
                    np.savez_compressed(
                        handle,
                        cellXYZ=cells,
                        optionId=np.arange(count, dtype=np.uint64),
                        sourceTableIndex=np.zeros(count, dtype=np.uint32),
                        sourceConfigurationIndex=np.arange(count, dtype=np.uint32),
                        localConfigurationId=np.zeros(count, dtype=np.uint16),
                        configurationLogWeight=np.zeros(count, dtype=np.float32),
                        selectedLayerCount=np.ones(count, dtype=np.uint16),
                    )
                selection_sha = sha256_file(selection_path)
                (selected / "selection-v1.json").write_text(
                    json.dumps(
                        {
                            "schema": "pareidolia.raw-acus-configuration-selection",
                            "version": 1,
                            "data": {
                                "path": selection_path.name,
                                "bytes": selection_path.stat().st_size,
                                "sha256": selection_sha,
                            },
                        }
                    )
                )
                (selected / "variant.json").write_text(
                    json.dumps(
                        {"identity": {"candidateDataSha256": candidate_sha}}
                    )
                )
                run_boundary_band_export(
                    selected,
                    boundary,
                    candidate_root=selected,
                    settings=BoundaryBandSettings(depth_cells=1),
                )
                boundary_roots.append(boundary)
            output = Path(directory) / "reselection"
            summary = run_boundary_band_reselection(
                boundary_roots[0], boundary_roots[1], output
            )

        statistics = summary["statistics"]
        self.assertEqual(statistics["mutableCells"], 32)
        self.assertEqual(statistics["changedConfigurations"], 0)
        self.assertEqual(statistics["strictCandidateJoins"], 72)
        self.assertEqual(statistics["retainedBandJoins"], 72)
        self.assertEqual(statistics["quarterTurnCandidateJoins"], 0)
        self.assertEqual(statistics["recomposedComponents"], 4)

    def test_multiseam_audit_detects_consistent_four_block_corner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundaries: dict[tuple[int, int], Path] = {}
            for y in range(2):
                for x in range(2):
                    boundaries[(x, y)] = self._write_analytic_candidate_boundary(
                        root / f"block-{x}-{y}",
                        GridSpec(
                            (4, 4, 4),
                            origin_xyz=(4.0 * x, 4.0 * y, 0.0),
                        ),
                        patch_id_start=1 + 64 * (x + 2 * y),
                    )
            seam_pairs = (
                ((0, 0), (1, 0)),
                ((0, 1), (1, 1)),
                ((0, 0), (0, 1)),
                ((1, 0), (1, 1)),
            )
            reselections: list[Path] = []
            for index, (first, second) in enumerate(seam_pairs):
                output = root / f"reselection-{index}"
                run_boundary_band_reselection(
                    boundaries[first],
                    boundaries[second],
                    output,
                )
                reselections.append(output)
            cluster = root / "cluster"
            run_boundary_cluster_reselection(
                tuple(boundaries[key] for key in sorted(boundaries)),
                cluster,
            )
            audit = run_multiseam_audit(
                tuple(reselections),
                root / "multiseam-audit.json",
                cluster_root=cluster,
            )

        configuration = audit["configurationConsistency"]
        topology = audit["topologyConsistency"]
        self.assertEqual(audit["layout"]["blocks"], 4)
        self.assertEqual(audit["layout"]["pairwiseSolutions"], 4)
        self.assertEqual(audit["layout"]["shapeCellsXYZ"], [8, 8, 4])
        self.assertEqual(configuration["overlapCells"], 16)
        self.assertEqual(configuration["disagreementCells"], 0)
        self.assertEqual(topology["crossingSeamPairs"], 4)
        self.assertTrue(topology["allCommonComponentPartitionsAgree"])
        self.assertGreater(audit["storage"]["componentCellRecords"], 0)
        cluster_comparison = audit["clusterComparison"]
        self.assertEqual(
            cluster_comparison["configuration"]["pairwiseDisagreementCells"],
            0,
        )
        self.assertTrue(
            all(
                value["componentPartitionsAgree"]
                for value in cluster_comparison["topology"]["comparisons"]
            )
        )

    def test_cluster_reselection_solves_four_block_corner_once(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundaries = tuple(
                self._write_analytic_candidate_boundary(
                    root / f"block-{x}-{y}",
                    GridSpec(
                        (4, 4, 4),
                        origin_xyz=(4.0 * x, 4.0 * y, 0.0),
                    ),
                    patch_id_start=1 + 64 * (x + 2 * y),
                )
                for y in range(2)
                for x in range(2)
            )
            output = root / "cluster-reselection"
            summary = run_boundary_cluster_reselection(boundaries, output)
            with np.load(output / "cluster-reselection-v1.npz") as values:
                combined_cells = np.asarray(values["selectedCellCombinedXYZ"])
            materialized_root = root / "cluster-materialized"
            materialized = run_cluster_materialization(
                output,
                materialized_root,
                boundary_roots=boundaries,
            )
            materialized_block = read_surface_graph(materialized_root)
            self._write_analytic_candidate_boundary(
                root / "full",
                GridSpec((8, 8, 4)),
                patch_id_start=1000,
            )
            packet_root = root / "full-packets"
            run_dual_axis_packet_connectivity(
                root / "full" / "selected", packet_root
            )
            reference = run_cluster_reference_audit(
                packet_root,
                output,
                root / "cluster-reference.json",
                full_selected_root=root / "full" / "selected",
            )

        statistics = summary["statistics"]
        self.assertEqual(summary["grid"]["shapeCellsXYZ"], [8, 8, 4])
        self.assertEqual(summary["layout"]["blocks"], 4)
        self.assertEqual(statistics["mutableCells"], 112)
        self.assertEqual(statistics["immutableAnchorShellCells"], 80)
        self.assertEqual(statistics["changedConfigurations"], 0)
        self.assertEqual(statistics["selectedMutablePatches"], 112)
        self.assertEqual(statistics["anchorPatches"], 80)
        self.assertEqual(statistics["recomposedComponents"], 4)
        self.assertEqual(len({tuple(value) for value in combined_cells}), 112)
        self.assertEqual(len(materialized_block.patches), 256)
        self.assertEqual(len(materialized_block.joins), 448)
        self.assertEqual(len(materialized_block.components), 4)
        self.assertEqual(
            materialized["independentChildBaseline"][
                "largestOccupiedCellCount"
            ],
            16,
        )
        self.assertEqual(
            materialized["materializedCluster"]["largestOccupiedCellCount"],
            64,
        )
        self.assertEqual(
            materialized["growth"]["componentsSpanningMultipleChildren"], 4
        )
        self.assertEqual(
            reference["configurationAgreement"]["allMutable"]["clusterExact"],
            112,
        )
        self.assertEqual(reference["joinAgreement"]["joinJaccard"], 1.0)
        self.assertEqual(reference["componentAgreement"]["delta"], 0)

    def test_cluster_reselection_supports_eight_block_corner(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            boundaries = tuple(
                self._write_analytic_candidate_boundary(
                    root / f"block-{x}-{y}-{z}",
                    GridSpec(
                        (4, 4, 4),
                        origin_xyz=(4.0 * x, 4.0 * y, 4.0 * z),
                    ),
                    patch_id_start=1 + 64 * (x + 2 * y + 4 * z),
                )
                for z in range(2)
                for y in range(2)
                for x in range(2)
            )
            summary = run_boundary_cluster_reselection(
                boundaries, root / "cluster"
            )

        statistics = summary["statistics"]
        self.assertEqual(summary["grid"]["shapeCellsXYZ"], [8, 8, 8])
        self.assertEqual(summary["layout"]["blocks"], 8)
        self.assertTrue(
            all(
                len(value["internalFaces"]) == 3
                for value in summary["layout"]["inputs"]
            )
        )
        self.assertEqual(statistics["mutableCells"], 296)
        self.assertEqual(statistics["immutableAnchorShellCells"], 152)
        self.assertEqual(statistics["changedConfigurations"], 0)
        self.assertEqual(statistics["recomposedComponents"], 8)

    def test_boundary_packet_policy_caps_only_quarter_turn_additions(self) -> None:
        grid = GridSpec((2, 1, 1))
        face = cell_face((0, 0, 0), 0, 1)
        angle = math.radians(16.0)
        first = clip_plane_to_cell(
            grid,
            (0, 0, 0),
            PlaneEstimate.isotropic(
                (math.sin(angle), 0.0, math.cos(angle)),
                0.5 * math.sin(angle),
                angular_std_radians=math.radians(10.0),
                height_std=0.1,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=1,
        )
        second = clip_plane_to_cell(
            grid,
            (1, 0, 0),
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                0.0,
                angular_std_radians=math.radians(10.0),
                height_std=0.1,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=2,
        )
        assert first is not None and second is not None
        policy = {
            "normalAndFiberPolarity": "axial/unsigned",
            "parallelMatching": {
                "orthogonalFiberEquivalence": False,
                "maximumNormalAngleDegrees": 90.0,
                "maximumFiberResidualDegrees": 90.0,
            },
            "quarterTurnAdmission": {
                "enabled": True,
                "maximumNormalAngleDegrees": 15.0,
                "maximumFiberFrameResidualDegrees": 15.0,
            },
            "strictParallelMatchesHavePriority": True,
        }
        parallel, _, _, _, _ = _ordered_packet_alignment(
            (first,), (second,), face, policy, grid
        )
        self.assertEqual(len(parallel), 1)
        self.assertGreater(
            math.degrees(parallel[0].normal_angle_radians), 15.0
        )
        self.assertFalse(parallel[0].fiber_quarter_turn)

        orthogonal_second = clip_plane_to_cell(
            grid,
            (1, 0, 0),
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                0.0,
                angular_std_radians=math.radians(1.0),
                height_std=0.1,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(2.0),
            ),
            patch_id=4,
        )
        axial_first = self._horizontal_patch(
            grid,
            (0, 0, 0),
            0.0,
            3,
            height_std=0.1,
        )
        assert orthogonal_second is not None
        quarter, _, _, _, _ = _ordered_packet_alignment(
            (axial_first,), (orthogonal_second,), face, policy, grid
        )
        self.assertEqual(len(quarter), 1)
        self.assertTrue(quarter[0].fiber_quarter_turn)

    def test_selected_subblock_preserves_world_geometry_and_patch_identity(
        self,
    ) -> None:
        grid = GridSpec(
            (4, 2, 2),
            cell_size_xyz=(2.0, 3.0, 5.0),
            origin_xyz=(10.0, 20.0, 30.0),
            coordinate_unit="voxel",
        )
        patches = tuple(
            self._horizontal_patch(
                grid,
                (x, y, z),
                0.0,
                1 + x + 4 * y + 8 * z,
            )
            for z in range(2)
            for y in range(2)
            for x in range(4)
        )
        source_by_id = {value.patch_id: value for value in patches}
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selected"
            output = Path(directory) / "subblock"
            write_patch_shard(
                source / "selected-patches-v1",
                PatchTable.from_patches(grid, patches),
            )
            summary = extract_selected_patch_subblock(
                source,
                output,
                start_cell_xyz=(2, 0, 0),
                stop_cell_xyz_exclusive=(4, 2, 2),
            )
            restored = read_patch_shard(output / "selected-patches-v1")

        self.assertEqual(summary["patches"], 8)
        self.assertEqual(restored.grid.shape_cells_xyz, (2, 2, 2))
        self.assertEqual(restored.grid.origin_xyz, (14.0, 20.0, 30.0))
        self.assertEqual(set(int(value) for value in restored.patch_id), {
            value.patch_id for value in patches if value.cell_xyz[0] >= 2
        })
        for patch in restored.to_patches():
            source_patch = source_by_id[patch.patch_id]
            np.testing.assert_allclose(
                sorted(value.point_xyz for value in patch.vertices),
                sorted(value.point_xyz for value in source_patch.vertices),
                atol=1.0e-6,
            )

    def test_owned_sheet_crop_rebases_geometry_and_clips_connectivity(self) -> None:
        grid = GridSpec(
            (4, 2, 1),
            cell_size_xyz=(2.0, 3.0, 5.0),
            origin_xyz=(10.0, 20.0, 30.0),
            coordinate_unit="voxel",
        )
        patches = tuple(
            self._horizontal_patch(grid, (x, y, 0), 0.0, 1 + x + 4 * y)
            for y in range(2)
            for x in range(4)
        )
        source_by_id = {value.patch_id: value for value in patches}
        block = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            patches,
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source"
            owned = Path(directory) / "owned"
            write_patch_shard(
                source / "selected-patches-v1",
                PatchTable.from_patches(grid, patches),
            )
            write_surface_graph(
                source,
                block,
                semantics="synthetic ownership crop source",
            )
            selected_cells = np.asarray(
                [(x, y, 0) for y in range(2) for x in range(4)],
                dtype=np.int32,
            )
            np.savez_compressed(
                source / "selected-configurations-v1.npz",
                cellXYZ=selected_cells,
                configurationIndex=np.arange(8, dtype=np.uint32),
                configurationId=np.arange(100, 108, dtype=np.uint64),
                inputIndex=np.zeros(8, dtype=np.uint16),
                sourceConfigurationIndex=np.arange(8, dtype=np.uint32),
            )
            summary = crop_surface_graph_to_owned_block(
                source,
                owned,
                start_cell_xyz=(1, 0, 0),
                stop_cell_xyz_exclusive=(3, 2, 1),
            )
            restored = read_surface_graph(owned)
            with np.load(owned / "owned-configurations-v1.npz") as values:
                owned_cells = np.asarray(values["cellXYZ"])
                owned_configuration_ids = np.asarray(values["configurationId"])

        self.assertEqual(restored.grid.shape_cells_xyz, (2, 2, 1))
        self.assertEqual(restored.grid.origin_xyz, (12.0, 20.0, 30.0))
        self.assertEqual(len(restored.patches), 4)
        self.assertEqual(len(restored.joins), 4)
        self.assertEqual(len(restored.components), 1)
        self.assertFalse(restored.unresolved_interior_traces)
        self.assertEqual(summary["summary"]["componentDisposition"]["clipped"], 1)
        self.assertEqual(
            summary["summary"]["componentDisposition"]["splitAfterCrop"], 0
        )
        np.testing.assert_array_equal(
            owned_cells,
            ((0, 0, 0), (1, 0, 0), (0, 1, 0), (1, 1, 0)),
        )
        np.testing.assert_array_equal(
            owned_configuration_ids,
            (101, 102, 105, 106),
        )
        for patch in restored.patches:
            source_patch = source_by_id[patch.patch_id]
            np.testing.assert_allclose(
                sorted(value.point_xyz for value in patch.vertices),
                sorted(value.point_xyz for value in source_patch.vertices),
                atol=1.0e-6,
            )

    def test_cached_sheet_halo_final_audit_is_attached_to_experiment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest_path = root / "sheet-halo-experiment-v1.json"
            audit_path = root / "sheet-halo-final-audit-v1.json"
            manifest_path.write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "identity": {"identitySha256": "synthetic-halo"},
                    }
                )
            )
            audit_path.write_text(
                json.dumps(
                    {
                        "experimentIdentitySha256": "synthetic-halo",
                        "configurationContext": {"referenceHaloCells": 2},
                    }
                )
            )

            restored = finalize_sheet_halo_experiment(root)
            manifest = json.loads(manifest_path.read_text())
            audit_sha256 = sha256_file(audit_path)

        self.assertEqual(
            restored["experimentIdentitySha256"], "synthetic-halo"
        )
        self.assertEqual(manifest["finalAudit"]["referenceHaloCells"], 2)
        self.assertEqual(
            manifest["finalAudit"]["sha256"], audit_sha256
        )

    def test_selected_subblock_partitions_and_rebases_physical_candidates(
        self,
    ) -> None:
        grid = GridSpec((4, 1, 1), origin_xyz=(12.0, 20.0, 30.0))
        cells = np.asarray(
            [(x, 0, 0) for x in range(4)],
            dtype=np.int32,
        )
        configurations = ConfigurationTable(
            cells,
            np.asarray((0, 2, 4, 6, 8), dtype=np.uint64),
            np.tile((0, 1), 4).astype(np.uint16),
            np.tile((-1.0, 0.0), 4).astype(np.float32),
            np.zeros(8, dtype=np.int8),
            np.asarray((0, 0, 1, 1, 2, 2, 3, 3, 4), dtype=np.uint64),
            np.tile((0.0, 0.0, 1.0), (4, 1)).astype(np.float32),
            np.zeros(4, dtype=np.float32),
            np.tile(
                (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                (4, 1),
            ).astype(np.float32),
            np.tile((1.0, 0.0, 0.0), (4, 1)).astype(np.float32),
            np.full(4, math.radians(2.0), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "selected"
            output = Path(directory) / "subblock"
            patches = tuple(
                self._horizontal_patch(grid, (x, 0, 0), 0.0, x + 1)
                for x in range(4)
            )
            write_patch_shard(
                source / "selected-patches-v1",
                PatchTable.from_patches(grid, patches),
            )
            candidate_path = source / "saturation-configurations-v1.npz"
            metadata = {
                name: np.ones(8, dtype=np.float32)
                for name in (
                    "evidenceLogScore",
                    "physicalLogScore",
                    "totalLogScore",
                    "coveredEvidenceMass",
                    "totalEvidenceMass",
                )
            }
            metadata["isCurrent"] = np.tile((0, 1), 4).astype(np.uint8)
            with candidate_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    **configurations.arrays(),
                    **metadata,
                )
            candidate_sha256 = sha256_file(candidate_path)
            (source / "saturation-configurations-v1.json").write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.cubical-saturation-configurations",
                        "version": 1,
                        "identitySha256": "analytic-candidates",
                        "data": {
                            "path": candidate_path.name,
                            "bytes": candidate_path.stat().st_size,
                            "sha256": candidate_sha256,
                        },
                    }
                )
            )
            selection_path = source / "selection-v1.npz"
            with selection_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    cellXYZ=cells,
                    optionId=np.arange(4, dtype=np.uint64),
                    sourceTableIndex=np.zeros(4, dtype=np.uint32),
                    sourceConfigurationIndex=np.asarray(
                        (1, 3, 5, 7), dtype=np.uint32
                    ),
                    localConfigurationId=np.ones(4, dtype=np.uint16),
                    configurationLogWeight=np.zeros(4, dtype=np.float32),
                    selectedLayerCount=np.ones(4, dtype=np.uint16),
                )
            selection_sha256 = sha256_file(selection_path)
            (source / "selection-v1.json").write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.raw-acus-configuration-selection",
                        "version": 1,
                        "data": {
                            "path": selection_path.name,
                            "bytes": selection_path.stat().st_size,
                            "sha256": selection_sha256,
                        },
                    }
                )
            )
            (source / "variant.json").write_text(
                json.dumps(
                    {"identity": {"candidateDataSha256": candidate_sha256}}
                )
            )
            summary = extract_selected_patch_subblock(
                source,
                output,
                start_cell_xyz=(2, 0, 0),
                stop_cell_xyz_exclusive=(4, 1, 1),
            )
            subset, _, candidate_manifest = load_saturation_candidates(output)
            with np.load(output / "selection-v1.npz") as values:
                selected_source = np.asarray(values["sourceConfigurationIndex"])

        self.assertEqual(summary["configurations"]["cells"], 2)
        self.assertEqual(subset.cell_count, 2)
        self.assertEqual(subset.configuration_count, 4)
        self.assertEqual(subset.layer_count, 2)
        np.testing.assert_array_equal(subset.cell_xyz, ((0, 0, 0), (1, 0, 0)))
        np.testing.assert_array_equal(subset.configuration_offset, (0, 2, 4))
        np.testing.assert_array_equal(selected_source, (1, 3))
        self.assertEqual(
            summary["configurations"]["candidateDataSha256"],
            candidate_manifest["data"]["sha256"],
        )

    def test_configuration_selection_keeps_frozen_warm_start(self) -> None:
        grid = GridSpec((2, 1, 1))
        table = ConfigurationTable(
            np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.int32),
            np.asarray((0, 2, 4), dtype=np.uint64),
            np.asarray((0, 1, 0, 1), dtype=np.uint16),
            np.asarray((0.0, -10.0, 0.0, -10.0), dtype=np.float32),
            np.zeros(4, dtype=np.int8),
            np.arange(5, dtype=np.uint64),
            np.tile((0.0, 0.0, 1.0), (4, 1)).astype(np.float32),
            np.asarray((-0.2, 0.2, -0.2, 0.2), dtype=np.float32),
            np.tile(
                (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                (4, 1),
            ).astype(np.float32),
            np.tile((1.0, 0.0, 0.0), (4, 1)).astype(np.float32),
            np.full(4, math.radians(2.0), dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
            np.ones(4, dtype=np.float32),
        )
        selection = optimize_configurations(
            grid,
            (table,),
            initial_configuration_indices={
                (0, 0, 0): (0, 1),
                (1, 0, 0): (0, 2),
            },
            mutable_cells={(1, 0, 0)},
        )
        selected = {
            value.cell_xyz: value.source_configuration_index
            for value in selection.selected_options
        }
        self.assertEqual(selected[(0, 0, 0)], 1)
        self.assertEqual(selected[(1, 0, 0)], 2)

        sparse = optimize_configurations(
            GridSpec((3, 1, 1)),
            (table,),
            active_cells={(0, 0, 0), (1, 0, 0)},
        )
        self.assertEqual(len(sparse.selected_options), 2)
        self.assertGreater(sparse.pairwise_evaluation_count, 0)

    def test_trace_mean_pairwise_reward_separates_quality_from_stack_size(
        self,
    ) -> None:
        raw_one = pairwise_reward_energy(
            -12.0,
            1,
            1,
            pairwise_scale=0.2,
            normalization="none",
        )
        raw_three = pairwise_reward_energy(
            -36.0,
            3,
            3,
            pairwise_scale=0.2,
            normalization="none",
        )
        mean_one = pairwise_reward_energy(
            -12.0,
            1,
            1,
            pairwise_scale=0.2,
            normalization="trace-mean",
        )
        mean_three = pairwise_reward_energy(
            -36.0,
            3,
            3,
            pairwise_scale=0.2,
            normalization="trace-mean",
        )
        self.assertAlmostEqual(raw_three, 3.0 * raw_one)
        self.assertAlmostEqual(mean_three, mean_one)
        settings = CellRefinementSettings(pairwise_scale=0.2)
        self.assertAlmostEqual(
            settings.resolved_unmatched_trace_penalty(TraceMatchSettings()),
            1.4,
        )

    def test_cell_refinement_target_ranks_preserve_ties_and_separate_cubes(
        self,
    ) -> None:
        np.testing.assert_allclose(
            _percentile_ranks(np.asarray((0.0, 0.0, 2.0, 4.0))),
            np.asarray((0.25, 0.25, 0.625, 0.875)),
        )
        records = [
            {"cellXYZ": [0, 0, 0]},
            {"cellXYZ": [2, 0, 0]},
            {"cellXYZ": [3, 0, 0]},
            {"cellXYZ": [6, 0, 0]},
        ]
        separated = _spatially_separated(
            records,
            radius_cells=1,
            maximum_targets=3,
        )
        self.assertEqual(
            [value["cellXYZ"] for value in separated],
            [[0, 0, 0], [3, 0, 0], [6, 0, 0]],
        )

    def test_frozen_face_topology_round_trip_preserves_anchor_certificates(
        self,
    ) -> None:
        grid = GridSpec((4, 4, 4))
        patches = tuple(
            self._horizontal_patch(
                grid,
                (x, y, z),
                0.0,
                1 + x + 4 * y + 16 * z,
            )
            for z in range(4)
            for y in range(4)
            for x in range(4)
        )
        block = assemble_surface_hierarchy(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            patches,
            maximum_leaf_shape_cells_xyz=(2, 2, 2),
        )
        states = build_frozen_face_states(
            patches,
            block.joins,
            grid.shape_cells_xyz,
            1,
        )
        expected = next(
            value for value in states if value.axis == 0 and value.side == 1
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frozen.npz"
            write_frozen_topology_artifact(path, states)
            restored = read_frozen_face_state(path, 0, 1)

        self.assertEqual(restored, expected)
        self.assertEqual(restored.frozen_component_count, 4)
        self.assertEqual(restored.detached_component_count, 0)
        self.assertEqual(len(restored.anchor_patch_ids), 16)
        self.assertTrue(restored.crossings)
        self.assertTrue(all(value.owners for value in restored.crossings))

        region_states = build_frozen_region_states(
            patches,
            block.joins,
            grid.shape_cells_xyz,
            1,
        )
        self.assertEqual(len(region_states), 26)
        self.assertEqual(
            {value.face_mask for value in region_states},
            set(compatible_face_masks()),
        )
        expected_region = next(
            value
            for value in region_states
            if value.face_mask == face_mask(((0, 1), (1, 0)))
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "regions.npz"
            write_frozen_region_artifact(path, region_states)
            restored_region = read_frozen_region_state(
                path, expected_region.face_mask
            )
        self.assertEqual(restored_region, expected_region)

    def test_arbitrary_frozen_cut_replays_local_joins_exactly(self) -> None:
        grid = GridSpec((4, 1, 1))
        patches = tuple(
            self._horizontal_patch(grid, (x, 0, 0), 0.0, x + 1)
            for x in range(4)
        )
        block = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            patches,
        )
        mutable_patch_ids = {2}
        local_joins = tuple(
            value
            for value in block.joins
            if value.first_patch_id in mutable_patch_ids
            or value.second_patch_id in mutable_patch_ids
        )
        anchor_observations = {
            (patch_id, edge)
            for join in local_joins
            for agreement in join.endpoint_agreements
            for patch_id, edge in (
                (join.first_patch_id, agreement.first_edge),
                (join.second_patch_id, agreement.second_edge),
            )
            if patch_id not in mutable_patch_ids
        }
        cut = freeze_topology_outside_patches(
            patches,
            block.joins,
            mutable_patch_ids,
            anchor_observations,
        )
        local_patch_ids = {*cut.anchor_patch_ids, *mutable_patch_ids}
        selection = select_joins_with_frozen_topology(
            tuple(
                value for value in patches if value.patch_id in local_patch_ids
            ),
            local_joins,
            cut.seed,
        )

        self.assertEqual(selection.joins, local_joins)
        self.assertFalse(selection.deferred_joins)
        self.assertEqual(selection.component_count, len(block.components))
        self.assertEqual(cut.seed.detached_component_count, 0)
        self.assertEqual(len(cut.frozen_join_keys), 1)

    def test_sheet_catalog_preserves_alternative_face_correspondences(self) -> None:
        grid = GridSpec((2, 1, 1))
        patches = (
            self._horizontal_patch(
                grid, (0, 0, 0), -0.2, 1, height_std=0.2
            ),
            self._horizontal_patch(
                grid, (0, 0, 0), 0.2, 2, height_std=0.2
            ),
            self._horizontal_patch(
                grid, (1, 0, 0), -0.2, 3, height_std=0.2
            ),
            self._horizontal_patch(
                grid, (1, 0, 0), 0.2, 4, height_std=0.2
            ),
        )
        bounds = BlockBounds((0, 0, 0), grid.shape_cells_xyz)
        baseline = assemble_surface_block(grid, bounds, patches)
        policy = SheetMatchingPolicy(
            TraceMatchSettings(
                maximum_endpoint_z=10.0,
                maximum_reduced_chi_square=100.0,
            ),
            False,
            15.0,
            15.0,
        )
        settings = SheetStitchingSettings(restart_count=3)
        catalog = enumerate_sheet_join_catalog(
            baseline,
            policy,
            settings=settings,
        )
        self.assertEqual(len(catalog.candidates), 4)
        cut, capacity = _minimum_undirected_join_cut(
            (1, 2, 3, 4),
            catalog.candidates,
            {value.key: 1.0 for value in catalog.candidates},
            1,
            2,
        )
        self.assertEqual(len(cut), 2)
        self.assertEqual(capacity, 2.0)

        crossing_priority = {
            value.key: (
                10.0
                if (value.match.first_patch_id, value.match.second_patch_id)
                in ((1, 4), (2, 3))
                else 1.0
            )
            for value in catalog.candidates
        }
        crossed = assemble_surface_block_from_candidates(
            grid,
            bounds,
            patches,
            (value.match for value in catalog.candidates),
            candidate_priorities=crossing_priority,
        )
        self.assertEqual(len(crossed.joins), 1)
        self.assertIn(
            "face-order-crossing",
            {value.reason for value in crossed.deferred_joins},
        )
        self.assertIn(
            "trace-occupancy",
            {value.reason for value in crossed.deferred_joins},
        )

        result = restitch_sheet_graph(
            baseline,
            catalog,
            policy,
            settings=settings,
        )
        self.assertEqual(len(result.block.joins), 2)
        self.assertEqual(len(result.block.components), 2)
        self.assertEqual(
            result.summary["delta"]["unresolvedInteriorTraceEndpoints"],
            0.0,
        )
        self.assertGreaterEqual(
            result.summary["best"]["totalJoinBenefit"],
            result.summary["baseline"]["totalJoinBenefit"],
        )

    def test_sheet_factor_uses_exact_noncrossing_augmenting_alignment(self) -> None:
        benefit, matched, quarter = _ordered_alignment_factor(
            (1, 2),
            (3, 4),
            {
                (1, 3): (5.0, False),
                (2, 4): (6.0, True),
                (1, 4): (10.0, False),
                (2, 3): (10.0, False),
            },
        )
        self.assertEqual(benefit, 11.0)
        self.assertEqual(matched, 2)
        self.assertEqual(quarter, 1)

    def test_sheet_configuration_belief_propagation_coordinates_cells(self) -> None:
        factors = {
            "firstCellIndex": np.asarray((0,), dtype=np.uint32),
            "secondCellIndex": np.asarray((1,), dtype=np.uint32),
            "firstConfigurationStart": np.asarray((0,), dtype=np.uint32),
            "firstConfigurationCount": np.asarray((2,), dtype=np.uint16),
            "secondConfigurationStart": np.asarray((2,), dtype=np.uint32),
            "secondConfigurationCount": np.asarray((2,), dtype=np.uint16),
            "pairOffset": np.asarray((0, 4), dtype=np.uint64),
            "pairJoinBenefit": np.asarray((0.0, 0.0, 0.0, 10.0)),
            "pairMatchedTraceCount": np.zeros(4, dtype=np.uint16),
            "pairUnmatchedTraceCount": np.zeros(4, dtype=np.uint16),
        }
        selected, record = _max_sum_configuration_seed(
            np.asarray((0, 2, 4), dtype=np.uint64),
            factors,
            np.asarray((3.0, 0.0, 3.0, 0.0)),
            SheetConfigurationSolverSettings(
                pairwise_scale=1.0,
                belief_propagation_iterations=8,
                belief_propagation_damping=0.0,
            ),
        )

        self.assertEqual(selected, (1, 3))
        self.assertTrue(record["beliefPropagationConverged"])

    def test_sheet_topology_acceptance_uses_evidence_objective_not_size(self) -> None:
        def evaluation(
            objective: float,
            components: tuple[tuple[int, tuple[int, ...]], ...],
            proposal: str,
        ) -> SheetTopologyEvaluation:
            patch_count = sum(len(values) for _, values in components)
            return SheetTopologyEvaluation(
                (0,),
                tuple(),
                tuple(),
                objective,
                objective,
                0.0,
                0.0,
                0,
                tuple(
                    (patch_id, component_id)
                    for component_id, values in components
                    for patch_id in values
                ),
                components,
                patch_count,
                0,
                0,
                0,
                proposal,
                0,
            )

        current = evaluation(10.0, ((1, (1,)), (2, (2,))), "current")
        larger_but_weaker = evaluation(9.9, ((1, (1, 2)),), "size-only")
        tied_larger = evaluation(10.0, ((1, (1, 2)),), "tie-size-only")
        supported = evaluation(10.2, ((1, (1,)), (2, (2,))), "supported")

        self.assertIs(
            choose_improving_topology_evaluation(
                current, (larger_but_weaker, tied_larger, supported)
            ),
            supported,
        )
        self.assertIsNone(
            choose_improving_topology_evaluation(
                current, (larger_but_weaker, tied_larger)
            )
        )
        self.assertIsNone(
            choose_improving_topology_evaluation(
                current, (supported,), minimum_objective_gain=0.25
            )
        )

    def test_sheet_topology_reopens_one_component_and_freezes_the_other(
        self,
    ) -> None:
        grid = GridSpec((3, 1, 1))
        patches = tuple(
            self._horizontal_patch(
                grid,
                (x, 0, 0),
                height,
                1 + 2 * x + layer,
            )
            for x in range(3)
            for layer, height in enumerate((-0.25, 0.25))
        )
        block = assemble_surface_block(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            patches,
        )
        component_by_patch = dict(block.component_by_patch)
        mutable_component = component_by_patch[1]
        mutable_patch_ids = {
            patch_id
            for patch_id, component_id in block.component_by_patch
            if component_id == mutable_component
        }
        fixed = frozen_exterior_join_keys(
            block.joins,
            {value.patch_id for value in patches},
            mutable_patch_ids,
        )

        self.assertEqual(len(block.components), 2)
        self.assertEqual(len(block.joins), 4)
        self.assertEqual(len(fixed), 2)
        self.assertTrue(
            all(
                first not in mutable_patch_ids and second not in mutable_patch_ids
                for first, second, _, _ in fixed
            )
        )

    def test_sheet_evidence_deduplicates_modes_and_preserves_stack_hyperedges(
        self,
    ) -> None:
        grid = GridSpec((2, 1, 1), origin_xyz=(10.0, 20.0, 30.0))
        cells = np.asarray(((0, 0, 0), (1, 0, 0)), dtype=np.int32)
        table = ConfigurationTable(
            cells,
            np.asarray((0, 2, 4), dtype=np.uint64),
            np.asarray((0, 1, 0, 1), dtype=np.uint16),
            np.asarray((0.0, -1.0, 0.0, -0.5), dtype=np.float32),
            np.asarray((0, -1, 0, 0), dtype=np.int8),
            np.asarray((0, 1, 1, 2, 3), dtype=np.uint64),
            np.tile((0.0, 0.0, 1.0), (3, 1)).astype(np.float32),
            np.asarray((0.0, 0.0, 0.2), dtype=np.float32),
            np.tile(
                (0.001, 0.0, 0.0, 0.001, 0.0, 0.001),
                (3, 1),
            ).astype(np.float32),
            np.tile((1.0, 0.0, 0.0), (3, 1)).astype(np.float32),
            np.full(3, math.radians(2.0), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.asarray((2.0, 3.0, 4.0), dtype=np.float32),
            np.ones(3, dtype=np.float32),
            np.ones(3, dtype=np.float32),
        )
        table.validate()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            raw = root / "raw"
            candidate = root / "candidate"
            mode_bank = root / "mode-bank"
            output = root / "sheet-evidence"
            current_patches = tuple(
                self._horizontal_patch(grid, tuple(cell), 0.0, index + 1)
                for index, cell in enumerate(cells.tolist())
            )
            write_patch_shard(
                raw / "selected-patches-v1",
                PatchTable.from_patches(grid, current_patches),
            )
            mode_bank.mkdir(parents=True)
            (mode_bank / "mode-bank.json").write_text(
                json.dumps(
                    {
                        "identity": {
                            "identitySha256": "synthetic-mode-bank-identity"
                        }
                    }
                )
            )
            candidate.mkdir(parents=True)
            data_path = candidate / "saturation-configurations-v1.npz"
            metadata = {
                name: np.asarray((1.0, 0.0, 1.5, 1.0), dtype=np.float32)
                for name in (
                    "evidenceLogScore",
                    "physicalLogScore",
                    "totalLogScore",
                    "coveredEvidenceMass",
                    "totalEvidenceMass",
                )
            }
            metadata["isCurrent"] = np.asarray((1, 0, 1, 0), dtype=np.uint8)
            with data_path.open("wb") as handle:
                np.savez_compressed(
                    handle,
                    **table.arrays(),
                    **metadata,
                    sourceShardIndex=np.zeros(4, dtype=np.int16),
                    sourceModeOffset=table.layer_offset,
                    sourceModeIndex=np.asarray((0, 1, 2), dtype=np.int32),
                    shardNames=np.asarray(("x0000-y0000-z0000",)),
                )
            data_sha = sha256_file(data_path)
            (candidate / "saturation-configurations-v1.json").write_text(
                json.dumps(
                    {
                        "schema": "pareidolia.cubical-saturation-configurations",
                        "version": 1,
                        "identitySha256": "synthetic-saturation-identity",
                        "data": {
                            "path": data_path.name,
                            "bytes": data_path.stat().st_size,
                            "sha256": data_sha,
                        },
                    }
                )
            )
            (candidate / "variant.json").write_text(
                json.dumps(
                    {
                        "state": "complete",
                        "inputRoot": str(raw),
                        "modeBankRoot": str(mode_bank),
                    }
                )
            )

            summary = compile_block_sheet_evidence(
                (SheetEvidenceInput(candidate),),
                output,
            )
            restored = read_block_sheet_evidence(output)
            source_modes = {
                value.patch_id: value for value in restored.mode_patches.to_patches()
            }
            subset_root = root / "sheet-evidence-subblock"
            subset_summary = extract_sheet_evidence_subblock(
                output,
                subset_root,
                start_cell_xyz=(1, 0, 0),
                stop_cell_xyz_exclusive=(2, 1, 1),
            )
            subset = read_block_sheet_evidence(subset_root)

        self.assertEqual(summary["statistics"]["ownedCells"], 2)
        self.assertEqual(summary["statistics"]["uniqueAcusModes"], 3)
        self.assertEqual(summary["statistics"]["physicalConfigurations"], 4)
        self.assertEqual(restored.mode_count, 3)
        self.assertEqual(restored.configuration_count, 4)
        self.assertEqual(restored.mode_patches.patch_count, 3)
        self.assertEqual(subset_summary["statistics"]["ownedCells"], 1)
        self.assertEqual(subset.grid.shape_cells_xyz, (1, 1, 1))
        self.assertEqual(subset.grid.origin_xyz, (11.0, 20.0, 30.0))
        self.assertEqual(subset.mode_count, 2)
        self.assertEqual(subset.configuration_count, 2)
        self.assertEqual(
            set(int(value) for value in subset.arrays["modeId"]),
            {
                patch_id
                for patch_id, patch in source_modes.items()
                if patch.cell_xyz == (1, 0, 0)
            },
        )
        for patch in subset.mode_patches.to_patches():
            np.testing.assert_allclose(
                sorted(value.point_xyz for value in patch.vertices),
                sorted(
                    value.point_xyz for value in source_modes[patch.patch_id].vertices
                ),
                atol=1.0e-6,
            )
        np.testing.assert_array_equal(
            np.diff(restored.arrays["configurationModeOffset"]),
            (1, 0, 1, 1),
        )
        correspondences = enumerate_mode_correspondences(
            restored,
            SheetMatchingPolicy(
                TraceMatchSettings(
                    maximum_endpoint_z=10.0,
                    maximum_reduced_chi_square=100.0,
                ),
                False,
                15.0,
                15.0,
            ),
        )
        self.assertEqual(len(correspondences.candidates), 2)
        self.assertEqual(
            sum(value.currently_active_endpoints for value in correspondences.candidates),
            1,
        )

    def test_noisy_synthetic_stack_remains_pure_and_connected(self) -> None:
        grid = GridSpec((8, 8, 5))
        scene = generate_synthetic_stack(
            grid,
            SyntheticStackSettings(
                sheet_count=2,
                curvature_amplitude_cells=0.12,
                observation_noise_scale=0.25,
                random_seed=7,
            ),
        )
        block = assemble_surface_hierarchy(
            grid,
            BlockBounds((0, 0, 0), grid.shape_cells_xyz),
            scene.patches,
            maximum_leaf_shape_cells_xyz=(2, 2, 2),
        )
        truth = scene.truth_map
        component_truth: dict[int, set[int]] = {}
        for patch_id, component in block.component_by_patch:
            component_truth.setdefault(component, set()).add(truth[patch_id])
        self.assertEqual(len(block.components), 2)
        self.assertTrue(all(len(value) == 1 for value in component_truth.values()))
        self.assertFalse(block.unresolved_interior_traces)
        self.assertFalse(block.deferred_joins)


if __name__ == "__main__":
    unittest.main()
