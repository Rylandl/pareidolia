#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_monotone_layers import prototype_monotone_layers


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Prototype order-preserving partial layer matches in the densest "
            "primary-family cell window."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--window-cells-xyz", type=int, nargs=3, default=[32, 32, 14])
    parser.add_argument("--window-origin-cell-xyz", type=int, nargs=3)
    parser.add_argument("--minimum-link-score", type=float, default=0.60)
    parser.add_argument("--gap-penalty", type=float, default=0.02)
    args = parser.parse_args()
    result = prototype_monotone_layers(
        args.root,
        force=args.force,
        settings={
            "windowCellsXYZ": args.window_cells_xyz,
            "minimumLinkScore": args.minimum_link_score,
            "gapPenalty": args.gap_penalty,
        },
        window_origin_cell_xyz=args.window_origin_cell_xyz,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "window": result["window"],
                "stats": result["stats"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
