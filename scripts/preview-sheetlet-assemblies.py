#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_assembly import build_assembly_previews


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rebuild leading boundary assemblies as exact flattened carriers."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_assembly_previews(args.root, args.top, force=args.force)
    print(
        json.dumps(
            {
                "stats": result["stats"],
                "candidates": [
                    {
                        "rank": value["rank"],
                        "sourceRanks": value["assembly"]["sourceRanks"],
                        "carrierCount": value["assembly"]["carrierCount"],
                        "flakeCount": value["assembly"]["flakeCount"],
                        "boundaryScore": value["assembly"]["medianBoundaryScore"],
                        "joinCost": value["joinCost"],
                        "yield": value["yield"],
                        "texture": value["texture"]["bestTextureScore"],
                        "image": value["artifacts"]["bestTextureImage"],
                    }
                    for value in result["candidates"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
