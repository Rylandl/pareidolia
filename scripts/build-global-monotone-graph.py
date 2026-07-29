#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_global_monotone_graph import build_global_monotone_graph


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build one collision- and parity-safe sparse graph from unanimous "
            "whole-volume tiled monotone evidence."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_global_monotone_graph(args.root, force=args.force)
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
