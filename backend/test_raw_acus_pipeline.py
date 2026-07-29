from __future__ import annotations

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cubical.contracts import (
    RawAcusSettings,
    ReconstructionWindow,
    ShardSpec,
    VolumeSource,
    VoxelBounds,
    extraction_tiles_for_shard,
    pipeline_identity,
    plan_extraction_tiles,
    plan_shards,
)
from backend.cubical.evidence import CellEvidenceTable, _normal_hypotheses
from backend.cubical.geometry import PlaneEstimate
from backend.cubical.raw_acus import (
    NeedleTable,
    read_needle_artifact,
    write_needle_artifact,
)
from backend.cubical.selection import optimize_configurations
from backend.cubical.stratigraphy import (
    ConfigurationTable,
    LayerMode,
    _transition_reward,
    build_stratigraphies,
)
from backend.cubical.topology import GridSpec


class RawAcusContractTests(unittest.TestCase):
    def test_regular_shards_have_disjoint_ownership_and_overlapping_support(self) -> None:
        settings = RawAcusSettings(calibration_samples=1)
        window = ReconstructionWindow((64, 64, 64), (5, 4, 3))
        shards = plan_shards(window, settings, (2, 2, 2))
        self.assertEqual(len(shards), 12)
        owned_cells = set()
        for shard in shards:
            for iz in range(shard.start_cell_xyz[2], shard.stop_cell_xyz_exclusive[2]):
                for iy in range(shard.start_cell_xyz[1], shard.stop_cell_xyz_exclusive[1]):
                    for ix in range(shard.start_cell_xyz[0], shard.stop_cell_xyz_exclusive[0]):
                        self.assertNotIn((ix, iy, iz), owned_cells)
                        owned_cells.add((ix, iy, iz))
            self.assertEqual(
                shard.raw_voxel_bounds.start_xyz,
                tuple(value - 32 for value in shard.owned_voxel_bounds.start_xyz),
            )
            self.assertEqual(
                shard.raw_voxel_bounds.stop_xyz_exclusive,
                tuple(
                    value + 32
                    for value in shard.owned_voxel_bounds.stop_xyz_exclusive
                ),
            )
        self.assertEqual(len(owned_cells), 5 * 4 * 3)

    def test_extraction_tiles_are_source_anchored_not_shard_anchored(self) -> None:
        source = VolumeSource(
            Path("/unused/native.npy"),
            None,
            (512, 512, 512),
            (0, 0, 0),
            9.362,
            {},
        )
        settings = RawAcusSettings(calibration_samples=1)
        window = ReconstructionWindow((128, 128, 128), (8, 8, 3))
        whole = plan_shards(window, settings, (8, 8, 3))
        partitioned = plan_shards(window, settings, (4, 4, 3))
        processing = VoxelBounds((112, 112, 112), (400, 400, 240))
        tiles = plan_extraction_tiles(source, processing, settings)

        whole_ids = {
            tile.tile_id
            for shard in whole
            for tile in extraction_tiles_for_shard(tiles, shard)
        }
        partitioned_ids = {
            tile.tile_id
            for shard in partitioned
            for tile in extraction_tiles_for_shard(tiles, shard)
        }
        self.assertEqual(whole_ids, partitioned_ids)
        self.assertEqual(len(whole_ids), 18)
        for tile in tiles:
            self.assertTrue(
                all(
                    (value - settings.extraction_halo_voxels)
                    % settings.extraction_tile_core_voxels
                    == 0
                    for value in tile.core_voxel_bounds.start_xyz
                )
            )
            self.assertEqual(
                tile.raw_voxel_bounds,
                tile.core_voxel_bounds.expand(settings.extraction_halo_voxels),
            )

    def test_volume_identity_comes_from_native_array_and_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_path = root / "volume.npy"
            np.save(source_path, np.zeros((128, 128, 128), dtype=np.uint8))
            metadata_path = root / "volume.json"
            metadata_path.write_text(
                json.dumps(
                    {"originXYZ": [7, 8, 9], "voxelSizeMicrons": 9.5}
                )
            )
            source = VolumeSource.open(source_path)
            settings = RawAcusSettings(calibration_samples=1)
            window = ReconstructionWindow((32, 32, 32), (2, 2, 2))
            window.validate(source, settings)
            identity = pipeline_identity(source, window, settings, (1, 1, 1))
            self.assertEqual(source.origin_xyz, (7, 8, 9))
            self.assertEqual(source.shape_xyz, (128, 128, 128))
            self.assertEqual(identity["source"]["shapeZYX"], [128, 128, 128])
            self.assertIn("metadataSha256", identity["source"])

    def test_needle_artifact_is_identity_checked(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            table = NeedleTable(
                np.asarray([[10.0, 11.0, 12.0]], dtype=np.float32),
                np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
                np.asarray([0.7], dtype=np.float32),
                np.asarray([0.8], dtype=np.float32),
                np.asarray([0.9], dtype=np.float32),
            )
            owned = VoxelBounds((8, 8, 8), (40, 40, 40))
            shard = ShardSpec(
                (0, 0, 0),
                (0, 0, 0),
                (1, 1, 1),
                owned,
                owned.expand(16),
                owned.expand(32),
            )
            write_needle_artifact(
                root / "needles-v1",
                table,
                identity_sha256="abc",
                shard=shard,
                compute_metadata={"backend": "test"},
            )
            restored = read_needle_artifact(
                root / "needles-v1", identity_sha256="abc"
            )
            np.testing.assert_array_equal(restored.center_xyz, table.center_xyz)
            with self.assertRaises(ValueError):
                read_needle_artifact(
                    root / "needles-v1", identity_sha256="different"
                )


class RawAcusInferenceTests(unittest.TestCase):
    def test_axial_needle_families_recover_common_normal_without_signs(self) -> None:
        rng = np.random.default_rng(11)
        directions = np.vstack(
            [
                np.tile((1.0, 0.0, 0.0), (24, 1)),
                np.tile((0.0, -1.0, 0.0), (24, 1)),
            ]
        )
        directions += rng.normal(0.0, 0.025, directions.shape)
        directions /= np.linalg.norm(directions, axis=1, keepdims=True)
        directions[::3] *= -1.0
        hypotheses = _normal_hypotheses(
            directions.astype(np.float32),
            np.ones(len(directions), dtype=np.float32),
            maximum=2,
        )
        self.assertTrue(hypotheses)
        self.assertGreater(abs(float(hypotheses[0][0][2])), 0.995)
        self.assertGreater(hypotheses[0][1], 0.5)

    @staticmethod
    def _two_layer_scene() -> tuple[VolumeSource, ShardSpec, NeedleTable, CellEvidenceTable, RawAcusSettings]:
        rng = np.random.default_rng(5)
        center = np.asarray([48.0, 48.0, 48.0], dtype=np.float32)
        centers = []
        directions = []
        for depth, direction in ((-7.0, (1.0, 0.0, 0.0)), (7.0, (0.0, 1.0, 0.0))):
            for _ in range(20):
                centers.append(
                    center
                    + np.asarray(
                        (
                            rng.uniform(-12.0, 12.0),
                            rng.uniform(-12.0, 12.0),
                            depth + rng.normal(0.0, 0.4),
                        ),
                        dtype=np.float32,
                    )
                )
                vector = np.asarray(direction) + rng.normal(0.0, 0.025, 3)
                vector /= np.linalg.norm(vector)
                directions.append(vector)
        count = len(centers)
        needles = NeedleTable(
            np.asarray(centers, dtype=np.float32),
            np.asarray(directions, dtype=np.float32),
            np.full(count, 0.8, dtype=np.float32),
            np.full(count, 0.9, dtype=np.float32),
            np.full(count, 0.9, dtype=np.float32),
        )
        depth = np.arange(-32, 33, dtype=np.float32)
        orientation = np.arange(36, dtype=np.float32) * 5.0 + 2.5
        density = np.zeros((1, 1, 65, 36), dtype=np.float16)
        support = np.zeros((1, 1, 65), dtype=np.float16)
        for height, angle_index in ((-7, 0), (7, 18)):
            depth_index = int(np.flatnonzero(depth == height)[0])
            density[0, 0, depth_index, angle_index] = 1.0
            support[0, 0, depth_index] = 1.0
        zeros = np.zeros((1, 1, 65), dtype=np.float16)
        evidence = CellEvidenceTable(
            np.asarray([[0, 0, 0]], dtype=np.int32),
            center[None],
            depth,
            orientation,
            np.ones((1, 1), dtype=np.uint8),
            np.asarray([[[0.0, 0.0, 1.0]]], dtype=np.float32),
            np.asarray([[0.8]], dtype=np.float32),
            np.asarray([[math.radians(2.0)]], dtype=np.float32),
            np.asarray([[count]], dtype=np.uint16),
            np.asarray([[20.0]], dtype=np.float32),
            np.ones((1, 1), dtype=np.float32),
            density,
            support,
            zeros.copy(),
            zeros.copy(),
            np.full((1, 1, 65), 0.8, dtype=np.float16),
        )
        settings = RawAcusSettings(
            maximum_normal_hypotheses=1, calibration_samples=1
        )
        owned = VoxelBounds((32, 32, 32), (64, 64, 64))
        shard = ShardSpec(
            (0, 0, 0),
            (0, 0, 0),
            (1, 1, 1),
            owned,
            owned.expand(16),
            owned.expand(32),
        )
        source = VolumeSource(
            Path("/unused/native.npy"),
            None,
            (128, 128, 128),
            (0, 0, 0),
            9.362,
            {},
        )
        return source, shard, needles, evidence, settings

    def test_top_m_stratigraphy_retains_physical_two_ply_solution_and_empty(self) -> None:
        source, shard, needles, evidence, settings = self._two_layer_scene()
        table, statistics = build_stratigraphies(
            source, shard, needles, evidence, settings
        )
        layer_counts = [
            len(table.estimates_for_configuration(index))
            for index in table.configurations_for_cell(0)
        ]
        self.assertEqual(max(layer_counts), 2)
        self.assertIn(0, layer_counts)
        self.assertGreaterEqual(statistics["candidateModeCount"], 2)
        best = table.estimates_for_configuration(0)
        self.assertEqual(len(best), 2)
        self.assertGreater(
            math.degrees(
                math.acos(
                    np.clip(abs(np.dot(best[0].fiber_xyz, best[1].fiber_xyz)), 0.0, 1.0)
                )
            ),
            80.0,
        )

    def test_physical_transition_allows_parallel_layers_but_rejects_crossing_planes(self) -> None:
        source = VolumeSource(
            Path("/unused/native.npy"), None, (128, 128, 128), (0, 0, 0), 9.362, {}
        )
        settings = RawAcusSettings(calibration_samples=1)
        first = LayerMode(
            0,
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                -5.0,
                math.radians(2.0),
                0.8,
                fiber_xyz=(1.0, 0.0, 0.0),
                fiber_angular_std_radians=math.radians(4.0),
            ),
            -5.0,
            0.0,
            0.8,
            0.8,
            8.0,
        )
        parallel = LayerMode(
            0,
            PlaneEstimate.isotropic(
                (0.0, 0.0, 1.0),
                5.0,
                math.radians(2.0),
                0.8,
                fiber_xyz=(1.0, 0.0, 0.0),
                fiber_angular_std_radians=math.radians(4.0),
            ),
            5.0,
            0.0,
            0.8,
            0.8,
            8.0,
        )
        self.assertIsNotNone(_transition_reward(first, parallel, source, settings))
        tilted_normal = np.asarray((-0.8, 0.0, 1.0), dtype=np.float64)
        tilted_normal /= np.linalg.norm(tilted_normal)
        crossing = LayerMode(
            0,
            PlaneEstimate.isotropic(
                tilted_normal,
                5.0 / float(np.linalg.norm((-0.8, 0.0, 1.0))),
                math.radians(2.0),
                0.8,
                fiber_xyz=(0.0, 1.0, 0.0),
                fiber_angular_std_radians=math.radians(4.0),
            ),
            5.0,
            90.0,
            0.8,
            0.8,
            8.0,
        )
        self.assertIsNone(_transition_reward(first, crossing, source, settings))

    @staticmethod
    def _single_plane_options(cell: tuple[int, int, int]) -> ConfigurationTable:
        angular_variance = math.radians(2.0) ** 2
        covariance = np.asarray(
            [[angular_variance, 0.0, 0.0, angular_variance, 0.0, 0.8**2]],
            dtype=np.float32,
        )
        return ConfigurationTable(
            np.asarray([cell], dtype=np.int32),
            np.asarray([0, 2], dtype=np.uint64),
            np.asarray([0, 1], dtype=np.uint16),
            np.asarray([-0.1, -2.0], dtype=np.float32),
            np.asarray([0, -1], dtype=np.int8),
            np.asarray([0, 1, 1], dtype=np.uint64),
            np.asarray([[0.0, 0.0, 1.0]], dtype=np.float32),
            np.asarray([0.1], dtype=np.float32),
            covariance,
            np.asarray([[1.0, 0.0, 0.0]], dtype=np.float32),
            np.asarray([math.radians(3.0)], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            np.asarray([0.9], dtype=np.float32),
            np.asarray([0.8], dtype=np.float32),
            np.asarray([8.0], dtype=np.float32),
        )

    def test_face_optimization_selects_agreeing_nonempty_configurations(self) -> None:
        grid = GridSpec((2, 1, 1), cell_size_xyz=(32.0, 32.0, 32.0))
        selection = optimize_configurations(
            grid,
            [
                self._single_plane_options((0, 0, 0)),
                self._single_plane_options((1, 0, 0)),
            ],
        )
        self.assertEqual(len(selection.patches), 2)
        self.assertTrue(all(value.patches for value in selection.selected_options))
        self.assertLess(selection.pairwise_energy, 0.0)
        self.assertEqual(selection.changed_last_sweep, 0)


if __name__ == "__main__":
    unittest.main()
