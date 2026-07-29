#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_growth import (
    build_growth_previews,
    grow_carrier_hypotheses,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Grow merged carriers through compatible neighboring Acus cells."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    growth = grow_carrier_hypotheses(args.root, force=args.force)
    result = growth if args.no_previews else build_growth_previews(args.root, force=args.force)
    print(
        json.dumps(
            {
                "stats": result["stats"],
                "rounds": growth["rounds"],
                "seeds": [
                    {
                        **seed,
                        **(
                            {
                                "growthCost": result["candidates"][index]["growthCost"],
                                "carrier": result["candidates"][index]["carrier"],
                                "texture": result["candidates"][index]["texture"][
                                    "bestTextureScore"
                                ],
                                "image": result["candidates"][index]["artifacts"][
                                    "bestTextureImage"
                                ],
                            }
                            if not args.no_previews
                            else {}
                        ),
                    }
                    for index, seed in enumerate(growth["seeds"])
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
