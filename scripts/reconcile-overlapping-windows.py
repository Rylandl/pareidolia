#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_window_reconciliation import reconcile_overlapping_windows


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Reconcile stable flake, monotone-edge, partition, and branch-join "
            "decisions in two independently solved overlapping windows."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--source-window-origin-cell-xyz", type=int, nargs=3)
    parser.add_argument(
        "--target-window-origin-cell-xyz", type=int, nargs=3, required=True
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = reconcile_overlapping_windows(
        args.root,
        target_origin_cell_xyz=args.target_window_origin_cell_xyz,
        source_origin_cell_xyz=args.source_window_origin_cell_xyz,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "overlap": result["overlap"],
                "classification": result["classification"],
                "stats": result["stats"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
