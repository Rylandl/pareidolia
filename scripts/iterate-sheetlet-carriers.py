#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_iteration import (
    build_iteration_previews,
    iterate_carrier_hypotheses,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Alternate guided carrier growth and boundary assembly to convergence."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--top", type=int, default=12)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--no-previews", action="store_true")
    args = parser.parse_args()
    iteration = iterate_carrier_hypotheses(args.root, force=args.force)
    previews = None
    if not args.no_previews:
        previews = build_iteration_previews(args.root, args.top, force=args.force)
    print(
        json.dumps(
            {
                "stats": iteration["stats"],
                "cycles": iteration["cycles"],
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
                    for index, state in enumerate(iteration["states"][: args.top])
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
