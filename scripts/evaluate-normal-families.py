#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_normal_family_evaluation import evaluate_normal_families


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the additive multi-normal graph and carrier residuals "
            "against a preserved single-normal baseline."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis")
    parser.add_argument("--baseline-flake-version", type=int, default=3)
    parser.add_argument("--baseline-explore-version", type=int, default=1)
    parser.add_argument("--baseline-screen-version", type=int, default=1)
    args = parser.parse_args()
    result = evaluate_normal_families(
        args.root,
        baseline_flake_version=args.baseline_flake_version,
        baseline_explore_version=args.baseline_explore_version,
        baseline_screen_version=args.baseline_screen_version,
    )
    print(
        json.dumps(
            {
                "declaredCriteria": result["declaredCriteria"],
                "stats": result["stats"],
                "graphComparison": result["graphComparison"],
                "screenComparison": result["screenComparison"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
