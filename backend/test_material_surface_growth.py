from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cubical.contracts import sha256_file
from backend.cubical.macro_orientation import run_macro_orientation_field
from backend.cubical.material_interface import MATERIAL_INTERFACE_SCHEMA
from backend.cubical.material_surface_graph import run_material_surface_graph
from backend.cubical.material_surface_bridging import (
    MATERIAL_SURFACE_BRIDGING_SCHEMA,
    _merge_physical_face_identity,
    run_material_surface_bridging,
)
from backend.cubical.material_surface_growth import (
    MATERIAL_SURFACE_GROWTH_SCHEMA,
    _support_geometry,
    run_material_surface_growth,
)
from backend.cubical.material_surface_fixed_point import (
    MATERIAL_SURFACE_FIXED_POINT_SCHEMA,
    run_material_surface_fixed_point,
)


class MaterialSurfaceGrowthTests(unittest.TestCase):
    def test_bridge_identity_never_merges_distinct_physical_faces(self) -> None:
        self.assertEqual(
            _merge_physical_face_identity(-1, 255, -1, 255),
            (True, -1, 255),
        )
        self.assertEqual(
            _merge_physical_face_identity(12, 0, -1, 255),
            (True, 12, 0),
        )
        self.assertEqual(
            _merge_physical_face_identity(-1, 255, 12, 1),
            (True, 12, 1),
        )
        self.assertEqual(
            _merge_physical_face_identity(12, 0, 12, 0),
            (True, 12, 0),
        )
        self.assertEqual(
            _merge_physical_face_identity(12, 0, 12, 1),
            (False, -1, 255),
        )
        self.assertEqual(
            _merge_physical_face_identity(12, 0, 13, 0),
            (False, -1, 255),
        )

    def test_open_one_sided_support_is_not_an_interior_hole(self) -> None:
        geometry = _support_geometry(
            np.zeros(3),
            np.asarray((0.0, 0.0, 1.0)),
            np.asarray(
                (
                    (1.0, -1.0, 0.0),
                    (1.0, 0.0, 0.0),
                    (1.0, 1.0, 0.0),
                    (2.0, 0.5, 0.0),
                )
            ),
            np.tile((0.0, 0.0, 1.0), (4, 1)),
            np.tile((0.0, 0.0, 1.0), (4, 1)),
            sampling_stride_voxels=1,
        )
        self.assertGreater(
            geometry["maximumSupportAngularGapDegrees"], 180.0
        )

    def test_enclosed_weak_interface_is_recovered_without_merging_planes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            volume_path = root / "volume.npy"
            metadata_path = root / "volume.json"
            np.save(volume_path, np.full((7, 9, 9), 128, dtype=np.uint8))
            metadata_path.write_text(
                json.dumps({"originXYZ": [0, 0, 0], "voxelSizeMicrons": 10.0})
            )
            x, y = np.meshgrid(np.arange(1, 8), np.arange(1, 8))
            xy = np.column_stack((x.reshape(-1), y.reshape(-1)))
            lower = np.column_stack(
                (xy, np.ones(len(xy), dtype=np.int32))
            )
            upper = np.column_stack(
                (xy, np.full(len(xy), 5, dtype=np.int32))
            )
            key = np.concatenate((lower, upper)).astype(np.int32)
            position = key.astype(np.float32)
            normal = np.tile((0.0, 0.0, 1.0), (len(key), 1)).astype(
                np.float32
            )
            evidence = np.ones(len(key), dtype=np.float32)
            hole = int(np.flatnonzero(np.all(key == (4, 4, 1), axis=1))[0])
            evidence[hole] = 0.2
            interface_root = root / "interfaces"
            interface_root.mkdir()
            interface_data_path = (
                interface_root / "material-interface-field-v1.npz"
            )
            np.savez_compressed(
                interface_data_path,
                positionXYZ=position,
                signedNormalXYZ=normal,
                processingKeyXYZ=key,
                localEvidenceScore=evidence,
            )
            interface_manifest = {
                "schema": MATERIAL_INTERFACE_SCHEMA,
                "version": 1,
                "state": "complete",
                "identity": {"settings": {"sampling_stride_voxels": 1}},
                "source": {
                    "path": str(volume_path),
                    "metadataPath": str(metadata_path),
                    "shapeZYX": [7, 9, 9],
                    "sourceOriginXYZ": [0, 0, 0],
                    "voxelSizeMicrons": 10.0,
                },
                "geometry": {
                    "ownedVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "ownedWorldBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "processingVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "processingShapeSamplingXYZ": [9, 9, 7],
                    "coordinateUnit": "source-voxel",
                },
                "calibration": {"displayHighRaw": 255.0},
                "data": {
                    "path": interface_data_path.name,
                    "sha256": sha256_file(interface_data_path),
                },
            }
            (interface_root / "material-interface-field-v1.json").write_text(
                json.dumps(interface_manifest)
            )

            macro_root = root / "macro"
            run_macro_orientation_field(interface_root, macro_root)
            graph_root = root / "graph"
            graph = run_material_surface_graph(
                interface_root, macro_root, graph_root
            )
            self.assertEqual(graph["counts"]["largestComponentSizes"][:2], [49, 48])

            growth_root = root / "growth"
            growth = run_material_surface_growth(
                interface_root, macro_root, graph_root, growth_root
            )
            self.assertEqual(growth["schema"], MATERIAL_SURFACE_GROWTH_SCHEMA)
            self.assertEqual(growth["counts"]["grownNodeCount"], 1)
            self.assertEqual(growth["counts"]["largestComponentSizes"][:2], [49, 49])
            with np.load(
                growth_root / "material-interface-interior-growth-v1.npz"
            ) as stored:
                interface_index = stored["interfaceIndex"]
                node = int(np.flatnonzero(interface_index == hole)[0])
                self.assertEqual(int(stored["growthRound"][node]), 1)
                self.assertGreaterEqual(int(stored["growthSupportCount"][node]), 4)

    def test_repeated_gap_faces_bridge_coplanar_fragments_only(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            volume_path = root / "volume.npy"
            metadata_path = root / "volume.json"
            np.save(volume_path, np.full((7, 9, 9), 128, dtype=np.uint8))
            metadata_path.write_text(
                json.dumps({"originXYZ": [0, 0, 0], "voxelSizeMicrons": 10.0})
            )
            x, y = np.meshgrid(np.arange(1, 8), np.arange(1, 8))
            xy = np.column_stack((x.reshape(-1), y.reshape(-1)))
            lower = np.column_stack(
                (xy, np.ones(len(xy), dtype=np.int32))
            )
            upper = np.column_stack(
                (xy, np.full(len(xy), 5, dtype=np.int32))
            )
            key = np.concatenate((lower, upper)).astype(np.int32)
            position = key.astype(np.float32)
            normal = np.tile((0.0, 0.0, 1.0), (len(key), 1)).astype(
                np.float32
            )
            evidence = np.ones(len(key), dtype=np.float32)
            gap = (key[:, 0] == 4) & (key[:, 2] == 1)
            evidence[gap] = 0.2
            interface_root = root / "interfaces"
            interface_root.mkdir()
            interface_data_path = (
                interface_root / "material-interface-field-v1.npz"
            )
            np.savez_compressed(
                interface_data_path,
                positionXYZ=position,
                signedNormalXYZ=normal,
                processingKeyXYZ=key,
                localEvidenceScore=evidence,
            )
            interface_manifest = {
                "schema": MATERIAL_INTERFACE_SCHEMA,
                "version": 1,
                "state": "complete",
                "identity": {"settings": {"sampling_stride_voxels": 1}},
                "source": {
                    "path": str(volume_path),
                    "metadataPath": str(metadata_path),
                    "shapeZYX": [7, 9, 9],
                    "sourceOriginXYZ": [0, 0, 0],
                    "voxelSizeMicrons": 10.0,
                },
                "geometry": {
                    "ownedVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "ownedWorldBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "processingVoxelBounds": {
                        "startXYZ": [0, 0, 0],
                        "stopXYZExclusive": [9, 9, 7],
                    },
                    "processingShapeSamplingXYZ": [9, 9, 7],
                    "coordinateUnit": "source-voxel",
                },
                "calibration": {"displayHighRaw": 255.0},
                "data": {
                    "path": interface_data_path.name,
                    "sha256": sha256_file(interface_data_path),
                },
            }
            (interface_root / "material-interface-field-v1.json").write_text(
                json.dumps(interface_manifest)
            )
            macro_root = root / "macro"
            run_macro_orientation_field(interface_root, macro_root)
            graph_root = root / "graph"
            graph = run_material_surface_graph(
                interface_root, macro_root, graph_root
            )
            self.assertEqual(
                graph["counts"]["largestComponentSizes"][:3], [49, 21, 21]
            )
            growth_root = root / "growth"
            growth = run_material_surface_growth(
                interface_root, macro_root, graph_root, growth_root
            )
            self.assertEqual(growth["counts"]["grownNodeCount"], 0)
            bridge_root = root / "bridge"
            bridge = run_material_surface_bridging(
                interface_root, macro_root, growth_root, bridge_root
            )
            self.assertEqual(bridge["schema"], MATERIAL_SURFACE_BRIDGING_SCHEMA)
            self.assertEqual(bridge["counts"]["componentCount"], 2)
            self.assertEqual(bridge["counts"]["componentMergeCount"], 1)
            self.assertEqual(bridge["counts"]["bridgeCandidateNodeCount"], 7)
            self.assertEqual(
                bridge["counts"]["largestComponentSizes"][:2], [49, 49]
            )
            second_growth_root = root / "growth-after-bridge"
            second_growth = run_material_surface_growth(
                interface_root,
                macro_root,
                bridge_root,
                second_growth_root,
            )
            self.assertEqual(second_growth["counts"]["grownNodeCount"], 0)
            self.assertEqual(second_growth["counts"]["componentCount"], 2)
            self.assertEqual(
                second_growth["identity"]["seedSurface"]["schema"],
                MATERIAL_SURFACE_BRIDGING_SCHEMA,
            )
            with np.load(
                second_growth_root
                / "material-interface-interior-growth-v1.npz"
            ) as stored:
                self.assertTrue(np.any(stored["edgeKind"] == 2))

            fixed_point_root = root / "fixed-point"
            fixed_point = run_material_surface_fixed_point(
                interface_root,
                macro_root,
                graph_root,
                fixed_point_root,
            )
            self.assertEqual(
                fixed_point["schema"], MATERIAL_SURFACE_FIXED_POINT_SCHEMA
            )
            self.assertTrue(fixed_point["convergence"]["converged"])
            self.assertEqual(fixed_point["convergence"]["completedCycles"], 2)
            self.assertEqual(
                [
                    (
                        cycle["grownNodeCount"],
                        cycle["bridgeCandidateNodeCount"],
                        cycle["componentMergeCount"],
                    )
                    for cycle in fixed_point["cycles"]
                ],
                [(0, 7, 1), (0, 0, 0)],
            )


if __name__ == "__main__":
    unittest.main()
