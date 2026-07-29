#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_association_integrity import association_integrity_audit


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Audit accepted branch-association MLS surfaces for intersections "
            "and sampled evidence-core clearance."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-origin-cell-xyz", type=int, nargs=3)
    parser.add_argument(
        "--maximum-evidence-core-distance-voxels", type=float, default=24.0
    )
    parser.add_argument(
        "--maximum-sampled-clearance-voxels", type=float, default=12.0
    )
    args = parser.parse_args()
    clearance_sweep = [
        value
        for value in (2.0, 4.0, 6.0, 8.0, 12.0)
        if value <= args.maximum_sampled_clearance_voxels
    ]
    if (
        not clearance_sweep
        or clearance_sweep[-1] < args.maximum_sampled_clearance_voxels
    ):
        clearance_sweep.append(args.maximum_sampled_clearance_voxels)
    result = association_integrity_audit(
        args.root,
        force=args.force,
        settings={
            "maximumEvidenceCoreDistanceVoxels": (
                args.maximum_evidence_core_distance_voxels
            ),
            "maximumSampledClearanceVoxels": (
                args.maximum_sampled_clearance_voxels
            ),
            "clearanceSweepVoxels": clearance_sweep,
        },
        window_origin_cell_xyz=args.window_origin_cell_xyz,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "topSpatialPairs": result["topSpatialPairs"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
