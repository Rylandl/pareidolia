#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_carrier_assembly import assemble_carriers


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Match nearby carrier boundaries and assemble sheet hypotheses."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def report(completed: int, total: int, elapsed_ms: float) -> None:
        print(
            f"boundary assembly {completed:,}/{total:,} · {elapsed_ms / 1000.0:.1f}s",
            flush=True,
        )

    result = assemble_carriers(args.root, force=args.force, progress=report)
    print(
        json.dumps(
            {
                "stats": result["stats"],
                "sweep": [
                    {key: value for key, value in sweep.items() if key != "topComponents"}
                    for sweep in result["sweep"]
                ],
                "topComponents": result["selected"]["topComponents"][:12],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
