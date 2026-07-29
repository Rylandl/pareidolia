from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

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
from backend.slab_analysis import (
    CELL_DTYPE,
    NEEDLE_DTYPE,
    _macro_radial_fit,
    run_slab_analysis,
    slab_overview,
)


class RectifierTests(unittest.TestCase):
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
