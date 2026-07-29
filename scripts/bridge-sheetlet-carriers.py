#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_bridges import (
    build_bridge_previews,
    build_long_range_bridges,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Bridge fixed-point carriers across longer CT-supported gaps."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    bridges = build_long_range_bridges(args.root, force=args.force)
    previews = None
    if not args.no_previews:
        previews = build_bridge_previews(args.root, args.top, force=args.force)
    print(
        json.dumps(
            {
                "stats": bridges["stats"],
                "sweep": bridges["sweep"],
                "topBridges": bridges["bridges"][:20],
                "topStates": [
                    {
                        **state,
                        **(
                            {
                                "carrier": previews["candidates"][index]["carrier"],
                                "texture": previews["candidates"][index]["texture"][
                                    "bestTextureScore"
                                ],
                                "image": previews["candidates"][index]["artifacts"][
                                    "bestTextureImage"
                                ],
                            }
                            if previews is not None and index < len(previews["candidates"])
                            else {}
                        ),
                    }
                    for index, state in enumerate(bridges["states"][: args.top])
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
