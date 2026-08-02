from __future__ import annotations

import unittest

import numpy as np

from backend.cubical.laminar_boundary_matching import (
    LaminarBoundaryMatchingSettings,
    compile_laminar_boundary_problem,
    solve_laminar_boundary_problem,
)


def _source_arrays() -> dict[str, np.ndarray]:
    # Candidate 1 and candidate 2 compete for boundary surfel 1.  Candidate 1
    # completes a sustained two-face chain, while candidate 2 is isolated.
    lower = np.asarray((0, 1, 1, 2), dtype=np.int32)
    upper = np.asarray((3, 4, 5, 6), dtype=np.int32)
    midpoint = np.asarray(
        ((0.0, 0.0, 0.0), (1.0, 0.0, 0.0), (1.0, 3.0, 0.0), (2.0, 0.0, 0.0)),
        dtype=np.float32,
    )
    count = len(lower)
    return {
        "nodeKind": np.ones(count, dtype=np.uint8),
        "lowerSurfaceNode": lower,
        "upperSurfaceNode": upper,
        "midpointXYZ": midpoint,
        "normalXYZ": np.tile(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32), (count, 1)
        ),
        "pairCost": np.ones(count, dtype=np.float32),
        "lowerLocalEvidenceScore": np.ones(count, dtype=np.float32),
        "upperLocalEvidenceScore": np.ones(count, dtype=np.float32),
        "thicknessResidualVoxels": np.zeros(count, dtype=np.float32),
        "thicknessVoxels": np.full(count, 8.0, dtype=np.float32),
    }


def _surface_arrays(*, both_faces: bool = True) -> dict[str, np.ndarray]:
    edge = [(0, 1), (1, 2)]
    if both_faces:
        edge.extend(((3, 4), (4, 6)))
    return {
        "positionXYZ": np.zeros((7, 3), dtype=np.float32),
        "signedNormalXYZ": np.tile(
            np.asarray((0.0, 0.0, 1.0), dtype=np.float32), (7, 1)
        ),
        "edgeFirstNode": np.asarray([value[0] for value in edge], dtype=np.int32),
        "edgeSecondNode": np.asarray([value[1] for value in edge], dtype=np.int32),
        "edgeScore": np.ones(len(edge), dtype=np.float32),
    }


class LaminarBoundaryMatchingTests(unittest.TestCase):
    def test_continuity_requires_both_physical_faces(self) -> None:
        problem, summary = compile_laminar_boundary_problem(
            _source_arrays(),
            _surface_arrays(both_faces=False),
            sampling_stride_voxels=1.0,
            settings=LaminarBoundaryMatchingSettings(
                enable_one_face_geometric_closure=False
            ),
        )
        self.assertEqual(summary["exactTwoFaceContinuityCount"], 0)
        self.assertEqual(len(problem["continuityFirstCandidate"]), 0)

    def test_one_face_edge_can_anchor_independent_opposite_face_closure(self) -> None:
        problem, summary = compile_laminar_boundary_problem(
            _source_arrays(),
            _surface_arrays(both_faces=False),
            sampling_stride_voxels=1.0,
            settings=LaminarBoundaryMatchingSettings(),
        )
        self.assertEqual(summary["exactTwoFaceContinuityCount"], 0)
        self.assertEqual(summary["oneFaceAnchoredClosureCount"], 2)
        np.testing.assert_array_equal(
            problem["continuityKind"], np.ones(2, dtype=np.uint8)
        )

    def test_global_matching_prefers_the_coherent_chain(self) -> None:
        settings = LaminarBoundaryMatchingSettings(maximum_solver_seconds=10.0)
        problem, summary = compile_laminar_boundary_problem(
            _source_arrays(),
            _surface_arrays(),
            sampling_stride_voxels=1.0,
            settings=settings,
        )
        self.assertEqual(summary["exactTwoFaceContinuityCount"], 2)
        selection, solve_summary = solve_laminar_boundary_problem(
            problem, settings=settings
        )
        np.testing.assert_array_equal(
            selection["selectedCandidate"],
            np.asarray((True, True, False, True)),
        )
        self.assertEqual(solve_summary["maximumSelectedMatesPerBoundarySurfel"], 1)
        self.assertEqual(solve_summary["selectedContinuityCount"], 2)


if __name__ == "__main__":
    unittest.main()
