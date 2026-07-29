#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_material_intervals import build_material_intervals


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Sample local-normal CT profiles and build hypothesis-independent "
            "material intervals with a separate flake-claim overlay."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--depth-minimum", type=float, default=-32.0)
    parser.add_argument("--depth-maximum", type=float, default=32.0)
    parser.add_argument("--depth-step", type=float, default=1.0)
    parser.add_argument("--claim-support-tolerance", type=float, default=1.5)
    parser.add_argument("--claim-cluster-gap", type=float, default=3.0)
    parser.add_argument("--tile-cells-zyx", type=int, nargs=3, default=[7, 16, 16])
    args = parser.parse_args()
    result = build_material_intervals(
        args.root,
        force=args.force,
        settings={
            "depthMinimumVoxels": args.depth_minimum,
            "depthMaximumVoxels": args.depth_maximum,
            "depthStepVoxels": args.depth_step,
            "claimSupportToleranceVoxels": args.claim_support_tolerance,
            "claimClusterGapVoxels": args.claim_cluster_gap,
            "tileCellsZYX": args.tile_cells_zyx,
        },
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "artifacts": result["artifacts"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
