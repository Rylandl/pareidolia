#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_gaps import (  # noqa: E402
    build_gap_fill_previews,
    fill_carrier_gaps,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fill enclosed carrier gaps with compatible unclaimed Acus flakes."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--top", type=int, default=24)
    parser.add_argument("--preview-top", type=int, default=12)
    parser.add_argument("--maximum-rounds", type=int, default=4)
    parser.add_argument("--minimum-gap-area", type=float, default=256.0)
    parser.add_argument("--score-threshold", type=float, default=0.62)
    parser.add_argument("--minimum-margin", type=float, default=0.04)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    gaps = fill_carrier_gaps(
        args.root,
        top_count=args.top,
        maximum_rounds=args.maximum_rounds,
        minimum_gap_area_square_voxels=args.minimum_gap_area,
        score_threshold=args.score_threshold,
        minimum_margin=args.minimum_margin,
        force=args.force,
    )
    previews = None
    if not args.no_previews:
        previews = build_gap_fill_previews(
            args.root, min(args.preview_top, args.top), force=args.force
        )
    print(
        json.dumps(
            {
                "stats": gaps["stats"],
                "rounds": gaps["rounds"],
                "changedStates": [
                    {
                        "rank": state["rank"],
                        "initialFlakeCount": state["initialFlakeCount"],
                        "flakeCount": state["flakeCount"],
                        "gapAddedFlakeCount": state["gapAddedFlakeCount"],
                        "finalGapCount": state.get("gapFill", {}).get("finalGapCount"),
                        "finalGapAreaSquareVoxels": state.get("gapFill", {}).get(
                            "finalGapAreaSquareVoxels"
                        ),
                        **(
                            {
                                "carrier": previews["candidates"][index]["carrier"],
                                "gapPixels": previews["candidates"][index]["gapPixels"],
                                "gapMap": previews["candidates"][index]["artifacts"][
                                    "gapMap"
                                ],
                            }
                            if previews is not None and index < len(previews["candidates"])
                            else {}
                        ),
                    }
                    for index, state in enumerate(gaps["states"][: args.top])
                    if int(state["gapAddedFlakeCount"]) > 0
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
