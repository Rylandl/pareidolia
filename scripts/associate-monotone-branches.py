#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_branch_association import (
    DEFAULT_SETTINGS,
    associate_monotone_branches,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Associate facing monotone-branch boundaries under material, order, "
            "collision, and overlapping-window constraints."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--maximum-gap-cells", type=int, default=3)
    parser.add_argument("--selected-threshold", type=float, default=0.45)
    parser.add_argument(
        "--threshold-sweep",
        type=float,
        nargs="+",
        default=[0.25, 0.35, 0.45, 0.55],
    )
    parser.add_argument("--subwindow-cells-xy", type=int, nargs=2, default=[24, 24])
    parser.add_argument("--window-origin-cell-xyz", type=int, nargs=3)
    parser.add_argument(
        "--maximum-exact-median-height-residual-voxels",
        type=float,
        default=DEFAULT_SETTINGS["maximumExactMedianHeightResidualVoxels"],
    )
    parser.add_argument(
        "--maximum-exact-median-normal-residual-deg",
        type=float,
        default=DEFAULT_SETTINGS["maximumExactMedianNormalResidualDeg"],
    )
    args = parser.parse_args()
    result = associate_monotone_branches(
        args.root,
        force=args.force,
        settings={
            "maximumGapCells": args.maximum_gap_cells,
            "selectedThreshold": args.selected_threshold,
            "thresholdSweep": args.threshold_sweep,
            "subwindowCellsXY": args.subwindow_cells_xy,
            "maximumExactMedianHeightResidualVoxels": (
                args.maximum_exact_median_height_residual_voxels
            ),
            "maximumExactMedianNormalResidualDeg": (
                args.maximum_exact_median_normal_residual_deg
            ),
        },
        window_origin_cell_xyz=args.window_origin_cell_xyz,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "sweeps": result["sweeps"],
                "selected": result["selected"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
