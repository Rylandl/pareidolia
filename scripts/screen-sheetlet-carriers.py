#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_sheetlet_carriers import screen_sheetlet_carriers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Coarsely flatten and rank the full substantial-sheetlet catalog."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def report(completed: int, total: int, elapsed_ms: float) -> None:
        rate = completed / max(elapsed_ms / 1000.0, 1.0e-6)
        remaining = (total - completed) / max(rate, 1.0e-6)
        print(
            f"screened {completed:,}/{total:,} · {rate:.1f} carriers/s · "
            f"~{remaining:.0f}s remaining",
            flush=True,
        )

    result = screen_sheetlet_carriers(
        args.root,
        force=args.force,
        candidate_limit=args.limit,
        progress=report,
    )
    print(
        json.dumps(
            {
                "settings": result["settings"],
                "stats": result["stats"],
                "yieldRanking": result["yieldRanking"][:20],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
