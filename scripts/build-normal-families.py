#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_normal_families import build_normal_families


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Fit standalone secondary normal candidates and include only "
            "spatially supported families."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def report(completed: int, total: int, elapsed_ms: float) -> None:
        rate = completed / max(elapsed_ms / 1000.0, 1.0e-6)
        remaining = (total - completed) / max(rate, 1.0e-6)
        print(
            f"normal families {completed:,}/{total:,} · {rate:.1f} cells/s · "
            f"~{remaining:.0f}s remaining",
            flush=True,
        )

    result = build_normal_families(
        args.root, force=args.force, progress=report
    )
    print(
        json.dumps(
            {
                "settings": result["settings"],
                "stats": result["stats"],
                "largestComponents": result["components"][:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
