from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cubical.contracts import sha256_file
from backend.cubical.macro_orientation import (
    MACRO_ORIENTATION_SCHEMA,
    MacroOrientationSettings,
    _physical_orientation_modes,
    run_macro_orientation_field,
)
from backend.cubical.material_interface import MATERIAL_INTERFACE_SCHEMA
from backend.cubical.material_surface_graph import (
    MATERIAL_SURFACE_GRAPH_SCHEMA,
    _collision_safe_components,
    run_material_surface_graph,
)


class MaterialSurfaceGraphTests(unittest.TestCase):
    def test_physical_orientation_modes_preserve_a_local_hairpin(self) -> None:
        tilt = np.deg2rad(2.0)
        normal = np.asarray(
            (
                (0.0, 0.0, 1.0),
                (np.sin(tilt), 0.0, np.cos(tilt)),
                (0.0, 0.0, -1.0),
                (-np.sin(tilt), 0.0, np.cos(tilt)),
                (1.0, 0.0, 0.0),
                (np.cos(tilt), 0.0, np.sin(tilt)),
                (-1.0, 0.0, 0.0),
                (np.cos(tilt), 0.0, -np.sin(tilt)),
            ),
            dtype=np.float64,
        )
        modes = _physical_orientation_modes(
            np.zeros(len(normal), dtype=np.int32),
            np.column_stack(
                (np.arange(len(normal)), np.zeros((len(normal), 2)))
            ),
            normal,
            np.full(len(normal), 0.95, dtype=np.float64),
            bin_count=1,
            settings=MacroOrientationSettings(
                minimum_physical_mode_samples=4,
                physical_mode_radius_degrees=15.0,
            ),
        )
        self.assertEqual(len(modes), 2)
        axes = np.asarray([mode["normalXYZ"] for mode in modes])
        self.assertGreater(float(np.max(np.abs(axes @ (0.0, 0.0, 1.0)))), 0.99)
        self.assertGreater(float(np.max(np.abs(axes @ (1.0, 0.0, 0.0)))), 0.99)

    def test_tangent_column_conflict_breaks_a_transitive_layer_loop(self) -> None:
        component, size, retained, summary = _collision_safe_components(
            np.zeros(5, dtype=np.int32),
            np.asarray((0, 1, 2, 3), dtype=np.int32),
            np.asarray((1, 2, 3, 4), dtype=np.int32),
            np.asarray((1.0, 0.9, 0.8, 0.7), dtype=np.float32),
            np.asarray((0, 1, 2, 1, 0), dtype=np.int32),
            np.asarray((0.0, 0.0, 1.0, 3.0, 3.0), dtype=np.float32),
            maximum_depth_range=2.25,
        )
        self.assertEqual(size.tolist(), [3, 2])
        self.assertEqual(len(np.unique(component)), 2)
        self.assertEqual(retained.tolist(), [1, 1, 0, 1])
        self.assertEqual(summary["columnConflictRejectedEdgeCount"], 1)

    def test_opposite_physical_boundary_sides_split_a_geometric_component(self) -> None:
        component, size, retained, summary = _collision_safe_components(
            np.zeros(5, dtype=np.int32),
            np.asarray((0, 1, 2, 3), dtype=np.int32),
            np.asarray((1, 2, 3, 4), dtype=np.int32),
            np.asarray((1.0, 0.9, 0.8, 0.7), dtype=np.float32),
            np.arange(5, dtype=np.int32),
            np.zeros(5, dtype=np.float32),
            maximum_depth_range=2.25,
            # Encoded identities 20 and 21 are the two sides of sheet 10.
            immutable_seed_label=np.asarray((20, -1, -1, -1, 21)),
        )
        self.assertEqual(size.tolist(), [4, 1])
        self.assertEqual(len(np.unique(component)), 2)
        self.assertEqual(retained.tolist(), [1, 1, 1, 0])
        self.assertEqual(summary["physicalSeedConflictRejectedEdgeCount"], 1)

    def test_macro_tensor_then_tangent_graph_keeps_parallel_faces_separate(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            volume_path = root / "volume.npy"
            metadata_path = root / "volume.json"
            np.save(volume_path, np.full((6, 8, 8), 128, dtype=np.uint8))
            metadata_path.write_text(
                json.dumps({"originXYZ": [0, 0, 0], "voxelSizeMicrons": 10.0})
            )
            x, y = np.meshgrid(np.arange(1, 7), np.arange(1, 7))
            xy = np.column_stack((x.reshape(-1), y.reshape(-1)))
            key = np.concatenate(
                (
                    np.column_stack((xy, np.ones(len(xy), dtype=np.int32))),
                    np.column_stack((xy, np.full(len(xy), 3, dtype=np.int32))),
                )
            ).astype(np.int32)
            position = key.astype(np.float32)
            tilt = np.deg2rad(20.0)
            normal = np.empty_like(position)
            normal[:, 0] = np.where(np.arange(len(position)) % 2, np.sin(tilt), -np.sin(tilt))
            normal[:, 1] = 0.0
            normal[:, 2] = np.cos(tilt)
            interface_root = root / "interfaces"
            interface_root.mkdir()
            data_path = interface_root / "material-interface-field-v1.npz"
            np.savez_compressed(
                data_path,
                positionXYZ=position,
                signedNormalXYZ=normal,
                processingKeyXYZ=key,
                localEvidenceScore=np.ones(len(position), dtype=np.float32),
            )
            manifest = {
                "schema": MATERIAL_INTERFACE_SCHEMA,
                "version": 1,
                "state": "complete",
                "identity": {"settings": {"sampling_stride_voxels": 1}},
                "source": {
                    "path": str(volume_path),
                    "metadataPath": str(metadata_path),
                    "shapeZYX": [6, 8, 8],
                    "sourceOriginXYZ": [0, 0, 0],
                    "voxelSizeMicrons": 10.0,
                },
                "geometry": {
                    "ownedVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [8, 8, 6],
                    },
                    "ownedWorldBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [8, 8, 6],
                    },
                    "processingVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [8, 8, 6],
                    },
                    "processingShapeSamplingXYZ": [8, 8, 6],
                    "coordinateUnit": "source-voxel",
                },
                "calibration": {"displayHighRaw": 255.0},
                "data": {
                    "path": data_path.name,
                    "sha256": sha256_file(data_path),
                },
            }
            (interface_root / "material-interface-field-v1.json").write_text(
                json.dumps(manifest)
            )

            macro_root = root / "macro"
            macro = run_macro_orientation_field(interface_root, macro_root)
            self.assertEqual(macro["schema"], MACRO_ORIENTATION_SCHEMA)
            with np.load(macro_root / "macro-sheet-orientation-field-v1.npz") as stored:
                macro_normal = stored["normalXYZ"]
                error = np.degrees(
                    np.arccos(np.clip(np.abs(macro_normal[:, 2]), 0.0, 1.0))
                )
                self.assertLess(float(np.max(error)), 1.0)

            graph_root = root / "graph"
            graph = run_material_surface_graph(
                interface_root, macro_root, graph_root
            )
            self.assertEqual(graph["schema"], MATERIAL_SURFACE_GRAPH_SCHEMA)
            self.assertEqual(graph["counts"]["largestComponentSizes"][:2], [36, 36])
            with np.load(
                graph_root / "material-interface-surface-graph-v1.npz"
            ) as stored:
                first = stored["edgeFirstNode"]
                second = stored["edgeSecondNode"]
                position = stored["positionXYZ"]
                np.testing.assert_allclose(position[first, 2], position[second, 2])


if __name__ == "__main__":
    unittest.main()
