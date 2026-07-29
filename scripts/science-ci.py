#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_science_benchmark import (
    compare_science_benchmark,
    freeze_science_benchmark,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Freeze or compare the bounded cross-scroll science benchmark."
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument(
        "--baseline",
        default="benchmarks/cross-scroll-z512-multinormal-v1.json",
    )
    parser.add_argument("--freeze", action="store_true")
    parser.add_argument("--verify-artifacts", action="store_true")
    args = parser.parse_args()
    if args.freeze:
        result = freeze_science_benchmark(args.root, args.baseline)
        output = {
            "identity": result["identity"],
            "scope": result["scope"],
            "stats": result["stats"],
        }
    else:
        result = compare_science_benchmark(
            args.root,
            args.baseline,
            verify_artifacts=args.verify_artifacts,
        )
        output = {
            "stats": result["stats"],
            "guards": result["guards"],
            "metricDifferences": result["metricDifferences"],
            "artifactAudit": result["artifactAudit"],
        }
    print(json.dumps(output, indent=2))
    if not args.freeze and not bool(result["stats"]["allGuardsPass"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
