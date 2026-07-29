#!/usr/bin/env python3
"""Audit enclosed gaps across every final sheetlet carrier."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.slab_gap_census import census_carrier_gaps  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure CT evidence inside every enclosed final-carrier gap."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-gap-area", type=float, default=256.0)
    parser.add_argument("--minimum-ct-score", type=float, default=0.5)
    parser.add_argument("--minimum-material", type=float, default=0.35)
    parser.add_argument("--maximum-depth-offset", type=float, default=4.0)
    parser.add_argument("--maximum-fiber-residual", type=float, default=12.0)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = census_carrier_gaps(
        args.root,
        minimum_gap_area_square_voxels=args.minimum_gap_area,
        minimum_ct_score=args.minimum_ct_score,
        minimum_material_fraction=args.minimum_material,
        maximum_depth_offset=args.maximum_depth_offset,
        maximum_fiber_residual=args.maximum_fiber_residual,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "stats": result["stats"],
                "queue": result["queue"],
                "queued": [
                    {
                        "rank": state["rank"],
                        "flakeCount": state["flakeCount"],
                        "queuedGapCount": state["queuedGapCount"],
                        "gaps": [
                            {
                                "gapId": gap["gapId"],
                                "areaSquareVoxels": gap["areaSquareVoxels"],
                                "ctEvidence": gap["ctEvidence"],
                                "gate": gap["gate"],
                            }
                            for gap in state["gaps"]
                            if gap["gate"]["queuedForDenseAcus"]
                        ],
                    }
                    for state in result["states"]
                    if state["queuedGapCount"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
