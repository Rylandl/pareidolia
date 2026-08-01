from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from backend.cubical.isolated_slab import (
    ISOLATED_SLAB_SCHEMA,
    IsolatedSlabSettings,
    otsu_material_calibration,
    run_isolated_slab_detection,
)
from backend.cubical.isolated_slab_audit import nearest_needle_and_segment


class IsolatedSlabTests(unittest.TestCase):
    def test_default_sampling_scale_is_physical_across_voxel_sizes(self) -> None:
        scroll = IsolatedSlabSettings.at_physical_scale(9.362)
        truth = IsolatedSlabSettings.at_physical_scale(3.24)
        self.assertEqual(scroll.sampling_stride_voxels, 2)
        self.assertEqual(truth.sampling_stride_voxels, 6)
        self.assertAlmostEqual(
            scroll.sampling_stride_voxels * 9.362,
            truth.sampling_stride_voxels * 3.24,
            delta=1.0,
        )

    def test_finite_needle_distance_clamps_to_segment_endpoints(self) -> None:
        result = nearest_needle_and_segment(
            np.asarray(((3.0, 2.0, 0.0), (8.0, 0.0, 0.0))),
            np.asarray(((0.0, 0.0, 0.0),)),
            np.asarray(((1.0, 0.0, 0.0),)),
            needle_half_length_voxels=4.0,
            maximum_search_radius_voxels=8.0,
        )
        np.testing.assert_allclose(
            result["nearestSegmentDistanceVoxels"], (2.0, 4.0)
        )
        np.testing.assert_allclose(
            result["nearestSegmentAxialOffsetVoxels"], (3.0, 4.0)
        )

    def test_otsu_calibration_preserves_two_raw_classes(self) -> None:
        values = np.concatenate(
            (
                np.full(2_000, 42, dtype=np.uint8),
                np.full(3_000, 146, dtype=np.uint8),
            )
        )
        calibration = otsu_material_calibration(values)
        self.assertGreater(calibration["materialThresholdRaw"], 41.0)
        self.assertLess(calibration["materialThresholdRaw"], 146.0)
        self.assertAlmostEqual(calibration["airMeanRaw"], 42.0)
        self.assertAlmostEqual(calibration["materialMeanRaw"], 146.0)

    def test_curved_air_material_air_slab_forms_one_dense_seed_patch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            z, y, x = np.indices((96, 96, 96), dtype=np.float32)
            center = 48.0 + 0.08 * (x - 48.0) + 0.04 * (y - 48.0)
            raw = np.full((96, 96, 96), 48.0, dtype=np.float32)
            inside = np.abs(z - center) <= 6.0
            raw[inside] = 145.0 + 8.0 * np.sin(x[inside] * 0.4)
            raw += 2.0 * np.sin(x * 0.23 + y * 0.17 + z * 0.11)
            source_path = root / "volume.npy"
            metadata_path = root / "volume.json"
            np.save(source_path, np.clip(np.rint(raw), 0, 255).astype(np.uint8))
            metadata_path.write_text(
                json.dumps(
                    {
                        "originXYZ": [100, 200, 300],
                        "voxelSizeMicrons": 10.0,
                    }
                )
            )
            settings = IsolatedSlabSettings(
                maximum_sheet_thickness_microns=200.0,
                minimum_air_clearance_microns=40.0,
            )
            output = root / "output"
            manifest = run_isolated_slab_detection(
                source_path,
                output,
                world_start_xyz=(116, 216, 316),
                world_stop_xyz_exclusive=(180, 280, 380),
                metadata_path=metadata_path,
                settings=settings,
            )
            self.assertEqual(manifest["schema"], ISOLATED_SLAB_SCHEMA)
            self.assertGreater(manifest["counts"]["highConfidenceSeedCount"], 900)
            self.assertEqual(manifest["components"]["componentCount"], 1)
            self.assertGreater(
                manifest["components"]["largestComponentSizes"][0], 900
            )
            thickness = manifest["distributions"]["seedThicknessVoxels"]
            self.assertGreater(thickness["median"], 11.0)
            self.assertLess(thickness["median"], 15.0)
            with np.load(output / "isolated-slabs-v1.npz") as arrays:
                midpoint = arrays["midpointXYZ"]
                self.assertTrue(np.all(midpoint >= np.asarray((116, 216, 316))))
                self.assertTrue(np.all(midpoint < np.asarray((180, 280, 380))))
                self.assertEqual(len(np.unique(arrays["componentId"])), 1)
            cached = run_isolated_slab_detection(
                source_path,
                output,
                world_start_xyz=(116, 216, 316),
                world_stop_xyz_exclusive=(180, 280, 380),
                metadata_path=metadata_path,
                settings=settings,
            )
            self.assertEqual(
                cached["identity"]["identitySha256"],
                manifest["identity"]["identitySha256"],
            )


if __name__ == "__main__":
    unittest.main()
