#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_sheetlet_explore import analyze_sheetlets_exploratory


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the exploratory direction-and-edge sheetlet graph."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = analyze_sheetlets_exploratory(args.root, force=args.force)
    print(json.dumps(result["stats"], indent=2))


if __name__ == "__main__":
    main()
