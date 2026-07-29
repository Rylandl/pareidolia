#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.slab_global_branch_candidates import build_global_branch_candidates


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Build whole-volume rescue candidates from locally deferred branch-pair "
            "evidence without accepting any joins."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = build_global_branch_candidates(args.root, force=args.force)
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "topCandidates": result["topCandidates"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
