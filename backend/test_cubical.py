from __future__ import annotations

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
    merge_surface_blocks,
    rebuild_surface_block,
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
from backend.cubical.matching import align_face_patches, match_face_traces
from backend.cubical.continuation import discover_mode_continuations
from backend.cubical.gaps import analyze_component_gaps
from backend.cubical.flatten import (
    component_mesh,
    rasterize_chart,
    sample_depth_stack,
    tangent_atlas_chart,
)
from backend.cubical.contracts import RawAcusSettings, VolumeSource
from backend.cubical.repair import evaluate_single_cell_gap_repairs
from backend.cubical.saturation import classify_cell_structural_evidence
from backend.cubical.selection import ConfigurationOption
from backend.cubical.stratigraphic_continuity import (
    PatchFingerprintTable,
    StratigraphicContinuitySettings,
    _calibrate_records,
    build_patch_fingerprints,
    score_patch_fingerprints,
)
from backend.cubical.stratigraphy import LayerModeTable
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
