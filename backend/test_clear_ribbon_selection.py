from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.clear_ribbon_selection import (
    ClearRibbonSelectionSettings,
    SELECTION_CLASS_ANCHORED_GROWTH,
    SELECTION_CLASS_NEW_CLEAR_CORE,
    SELECTION_CLASS_UNSELECTED,
    SELECTION_CLASS_UPSTREAM_ANCHOR,
    select_clear_ribbons,
)


def _bank(
    *,
    key: tuple[tuple[int, int, int], ...],
    component: tuple[int, ...],
    component_size: tuple[int, ...],
    component_assembly_count: tuple[int, ...],
    component_sole_assembly: tuple[int, ...],
    selected: tuple[int, ...],
    label: tuple[int, ...],
    edges: tuple[tuple[int, int, float], ...],
    evidence: tuple[float, ...] | None = None,
) -> dict[str, np.ndarray]:
    ribbon_count = len(key)
    if evidence is None:
        evidence = (1.0,) * ribbon_count
    return {
        "pairedCandidateIndex": np.arange(ribbon_count, dtype=np.int32),
        "ribbonComponent": np.asarray(component, dtype=np.int32),
        "componentRibbonCount": np.asarray(component_size, dtype=np.int32),
        "componentAssemblyCount": np.asarray(
            component_assembly_count, dtype=np.uint16
        ),
        "componentSoleAssemblyLabel": np.asarray(
            component_sole_assembly, dtype=np.int32
        ),
        "spatialKeyXYZ": np.asarray(key, dtype=np.int32),
        "localEvidenceScore": np.asarray(evidence, dtype=np.float32),
        "selectedPairedSurface": np.asarray(selected, dtype=np.uint8),
        "selectedAssemblyLabel": np.asarray(label, dtype=np.int32),
        "edgeFirstRibbon": np.asarray(
            tuple(edge[0] for edge in edges), dtype=np.int32
        ),
        "edgeSecondRibbon": np.asarray(
            tuple(edge[1] for edge in edges), dtype=np.int32
        ),
        "edgeAffinity": np.asarray(
            tuple(edge[2] for edge in edges), dtype=np.float32
        ),
    }


class ClearRibbonSelectionTests(unittest.TestCase):
    def test_anchor_growth_selects_only_one_alternative_per_spatial_key(
        self,
    ) -> None:
        bank = _bank(
            key=((0, 0, 0), (1, 0, 0), (1, 0, 0)),
            component=(0, 0, 0),
            component_size=(3,),
            component_assembly_count=(1,),
            component_sole_assembly=(7,),
            selected=(1, 0, 0),
            label=(7, -1, -1),
            edges=((0, 1, 0.8), (0, 2, 0.9)),
        )
        selection, stats = select_clear_ribbons(
            bank,
            processing_shape_sampling_xyz=(2, 1, 1),
            settings=ClearRibbonSelectionSettings(),
        )
        np.testing.assert_array_equal(
            selection["selectedAssemblyLabel"], (7, -1, 7)
        )
        np.testing.assert_array_equal(
            selection["selectionClass"],
            (
                SELECTION_CLASS_UPSTREAM_ANCHOR,
                SELECTION_CLASS_UNSELECTED,
                SELECTION_CLASS_ANCHORED_GROWTH,
            ),
        )
        self.assertEqual(stats["collisionRejectedRibbonCount"], 1)

    def test_new_core_must_survive_exclusivity_at_minimum_size(self) -> None:
        first_key = tuple((x, 0, 0) for x in range(8))
        second_key = tuple((8 + min(x, 6), 0, 0) for x in range(8))
        edges = tuple((x, x + 1, 0.9) for x in range(7)) + tuple(
            (8 + x, 9 + x, 0.9) for x in range(7)
        )
        bank = _bank(
            key=first_key + second_key,
            component=(0,) * 8 + (1,) * 8,
            component_size=(8, 8),
            component_assembly_count=(0, 0),
            component_sole_assembly=(-1, -1),
            selected=(0,) * 16,
            label=(-1,) * 16,
            edges=edges,
        )
        selection, stats = select_clear_ribbons(
            bank,
            processing_shape_sampling_xyz=(15, 1, 1),
            settings=ClearRibbonSelectionSettings(
                minimum_new_component_ribbons=8
            ),
        )
        self.assertEqual(stats["newClearCoreCount"], 1)
        self.assertEqual(stats["newClearCoreRibbonCount"], 8)
        self.assertEqual(stats["rejectedNewComponentCount"], 1)
        np.testing.assert_array_equal(
            selection["selectionClass"][:8],
            np.full(8, SELECTION_CLASS_NEW_CLEAR_CORE),
        )
        np.testing.assert_array_equal(
            selection["selectionClass"][8:],
            np.full(8, SELECTION_CLASS_UNSELECTED),
        )

    def test_contested_component_preserves_anchor_and_defers_interior(
        self,
    ) -> None:
        bank = _bank(
            key=((0, 0, 0), (1, 0, 0)),
            component=(0, 0),
            component_size=(2,),
            component_assembly_count=(2,),
            component_sole_assembly=(-1,),
            selected=(1, 0),
            label=(7, -1),
            edges=((0, 1, 1.0),),
        )
        selection, stats = select_clear_ribbons(
            bank,
            processing_shape_sampling_xyz=(2, 1, 1),
            settings=ClearRibbonSelectionSettings(),
        )
        np.testing.assert_array_equal(selection["selected"], (1, 0))
        np.testing.assert_array_equal(
            selection["selectionClass"],
            (
                SELECTION_CLASS_UPSTREAM_ANCHOR,
                SELECTION_CLASS_UNSELECTED,
            ),
        )
        self.assertEqual(stats["anchoredGrowthCount"], 0)


if __name__ == "__main__":
    unittest.main()
