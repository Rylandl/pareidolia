#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_sheetlet_carriers import (
    CARRIER_SCREEN_VERSION,
    build_sheetlet_carriers,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Fit and sample continuous carriers for ranked sheetlets."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--top", type=int, default=3)
    parser.add_argument(
        "--screened-top",
        type=int,
        help="materialize this many winners from the completed coarse screen",
    )
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    source_ranks = None
    summary_label = None
    if args.screened_top is not None:
        screen_path = (
            Path(args.root)
            / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json"
        )
        screen = json.loads(screen_path.read_text())
        count = max(1, min(int(args.screened_top), 64))
        source_ranks = [
            int(value["sourceRank"]) for value in screen["yieldRanking"][:count]
        ]
        summary_label = f"screened-top{count}"
    result = build_sheetlet_carriers(
        args.root,
        args.top,
        force=args.force,
        source_ranks=source_ranks,
        summary_label=summary_label,
    )
    print(
        json.dumps(
            {
                "settings": result["settings"],
                "stats": result["stats"],
                "yieldRanking": result["yieldRanking"][:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
