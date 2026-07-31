from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.clear_ribbon_paired_feedback import (
    SELECTION_CLASS_NEW_CLEAR_CORE,
    SELECTION_CLASS_PAIRED_GROWTH,
    build_paired_feedback_seeds,
    grow_clear_cores_in_paired_graph,
    label_free_paired_components,
)
from backend.cubical.clear_ribbon_selection import (
    SELECTION_CLASS_NEW_CLEAR_CORE as RIBBON_CLASS_NEW_CLEAR_CORE,
)
from backend.cubical.paired_surface_growth import PairedSurfaceGrowthSettings


def _baseline(count: int) -> dict[str, np.ndarray]:
    selected = np.zeros(count, dtype=np.uint8)
    selected[0] = 1
    label = np.full(count, -1, dtype=np.int32)
    label[0] = 7
    path = np.zeros(count, dtype=np.float32)
    path[0] = 1.0
    return {
        "selected": selected,
        "selectedLabel": label,
        "pathBottleneck": path,
        "parentCandidate": np.full(count, -1, dtype=np.int32),
        "parentContinuityEdge": np.full(count, -1, dtype=np.int32),
    }


def _bank(keys: tuple[tuple[int, int, int], ...]) -> dict[str, np.ndarray]:
    return {
        "localEvidenceScore": np.ones(len(keys), dtype=np.float32),
        "spatialKeyXYZ": np.asarray(keys, dtype=np.int32),
    }


def _graph(
    edges: tuple[tuple[int, int, float], ...]
) -> dict[str, np.ndarray]:
    return {
        "edgeFirstCandidate": np.asarray(
            tuple(value[0] for value in edges), dtype=np.int32
        ),
        "edgeSecondCandidate": np.asarray(
            tuple(value[1] for value in edges), dtype=np.int32
        ),
        "edgeAffinity": np.asarray(
            tuple(value[2] for value in edges), dtype=np.float32
        ),
    }


class ClearRibbonPairedFeedbackTests(unittest.TestCase):
    def test_empty_free_graph_preserves_the_baseline(self) -> None:
        bank = _bank(((0, 0, 0),))
        baseline = _baseline(1)
        graph = _graph(())
        feedback_seed = {
            "newSeedCandidate": np.empty(0, dtype=np.int32),
            "newSeedLabel": np.asarray((-1,), dtype=np.int32),
        }
        settings = PairedSurfaceGrowthSettings()
        membership, stats = label_free_paired_components(
            bank,
            baseline,
            graph,
            feedback_seed,
            processing_shape_sampling_xyz=(1, 1, 1),
            settings=settings,
        )
        self.assertEqual(stats["freeComponentCount"], 0)
        selection, growth = grow_clear_cores_in_paired_graph(
            bank,
            baseline,
            graph,
            feedback_seed,
            membership,
            processing_shape_sampling_xyz=(1, 1, 1),
            settings=settings,
        )
        np.testing.assert_array_equal(selection["selected"], (1,))
        self.assertEqual(growth["grownPairedCandidateCount"], 0)

    def test_seed_builder_rejects_a_baseline_key_collision(self) -> None:
        bank = _bank(((0, 0, 0), (1, 0, 0)))
        baseline = _baseline(2)
        ribbons = {"pairedCandidateIndex": np.asarray((1,), dtype=np.int32)}
        selection = {
            "selected": np.asarray((1,), dtype=np.uint8),
            "selectionClass": np.asarray(
                (RIBBON_CLASS_NEW_CLEAR_CORE,), dtype=np.uint8
            ),
            "selectedAssemblyLabel": np.asarray((10,), dtype=np.int32),
        }
        seeds, stats = build_paired_feedback_seeds(
            bank,
            baseline,
            ribbons,
            selection,
            processing_shape_sampling_xyz=(2, 1, 1),
            settings=PairedSurfaceGrowthSettings(),
        )
        self.assertEqual(stats["newSeedCandidateCount"], 1)
        self.assertEqual(int(seeds["newSeedLabel"][1]), 10)

        bank["spatialKeyXYZ"][1] = (0, 0, 0)
        with self.assertRaises(ValueError):
            build_paired_feedback_seeds(
                bank,
                baseline,
                ribbons,
                selection,
                processing_shape_sampling_xyz=(2, 1, 1),
                settings=PairedSurfaceGrowthSettings(),
            )

    def test_multi_label_free_component_defers_its_interior(self) -> None:
        bank = _bank(
            ((0, 0, 0), (1, 0, 0), (2, 0, 0), (3, 0, 0))
        )
        baseline = _baseline(4)
        graph = _graph(((1, 2, 1.0), (2, 3, 1.0)))
        seed_label = np.asarray((-1, 10, -1, 20), dtype=np.int32)
        feedback_seed = {
            "newSeedCandidate": np.asarray((1, 3), dtype=np.int32),
            "newSeedLabel": seed_label,
        }
        settings = PairedSurfaceGrowthSettings()
        membership, stats = label_free_paired_components(
            bank,
            baseline,
            graph,
            feedback_seed,
            processing_shape_sampling_xyz=(4, 1, 1),
            settings=settings,
        )
        self.assertEqual(stats["contestedComponentCount"], 1)
        selection, growth = grow_clear_cores_in_paired_graph(
            bank,
            baseline,
            graph,
            feedback_seed,
            membership,
            processing_shape_sampling_xyz=(4, 1, 1),
            settings=settings,
        )
        np.testing.assert_array_equal(selection["selected"], (1, 1, 0, 1))
        self.assertEqual(growth["grownPairedCandidateCount"], 0)

    def test_single_label_growth_keeps_one_candidate_per_key(self) -> None:
        bank = _bank(
            ((0, 0, 0), (1, 0, 0), (2, 0, 0), (2, 0, 0))
        )
        baseline = _baseline(4)
        graph = _graph(((1, 2, 0.8), (1, 3, 0.9)))
        feedback_seed = {
            "newSeedCandidate": np.asarray((1,), dtype=np.int32),
            "newSeedLabel": np.asarray((-1, 10, -1, -1), dtype=np.int32),
        }
        settings = PairedSurfaceGrowthSettings()
        membership, _stats = label_free_paired_components(
            bank,
            baseline,
            graph,
            feedback_seed,
            processing_shape_sampling_xyz=(3, 1, 1),
            settings=settings,
        )
        selection, growth = grow_clear_cores_in_paired_graph(
            bank,
            baseline,
            graph,
            feedback_seed,
            membership,
            processing_shape_sampling_xyz=(3, 1, 1),
            settings=settings,
        )
        np.testing.assert_array_equal(selection["selected"], (1, 1, 0, 1))
        self.assertEqual(
            int(selection["selectionClass"][1]),
            SELECTION_CLASS_NEW_CLEAR_CORE,
        )
        self.assertEqual(
            int(selection["selectionClass"][3]),
            SELECTION_CLASS_PAIRED_GROWTH,
        )
        self.assertEqual(growth["collisionRejectedCandidateCount"], 1)


if __name__ == "__main__":
    unittest.main()
