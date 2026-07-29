#!/usr/bin/env python3
"""Run targeted fine Acus extraction inside selected carrier gaps."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from backend.slab_gap_reanalysis import reanalyze_carrier_gaps  # noqa: E402


def _ranks(value: str) -> tuple[int, ...]:
    try:
        parsed = tuple(int(item.strip()) for item in value.split(",") if item.strip())
    except ValueError as error:
        raise argparse.ArgumentTypeError("ranks must be comma-separated integers") from error
    if not parsed:
        raise argparse.ArgumentTypeError("at least one rank is required")
    return parsed


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Re-extract dense Acus needles only inside selected carrier gaps."
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--ranks", type=_ranks, default=(11, 12))
    parser.add_argument("--fine-stride", type=float, default=8.0)
    parser.add_argument("--candidate-spacing", type=int, default=2)
    parser.add_argument("--maximum-per-bin", type=int, default=256)
    parser.add_argument("--maximum-cell-needles", type=int, default=640)
    parser.add_argument("--score-threshold", type=float, default=0.55)
    parser.add_argument("--minimum-mode-margin", type=float, default=0.04)
    parser.add_argument("--minimum-ownership-margin", type=float, default=0.04)
    parser.add_argument("--minimum-ct-evidence", type=float, default=0.5)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()
    result = reanalyze_carrier_gaps(
        args.root,
        ranks=args.ranks,
        fine_stride=args.fine_stride,
        candidate_spacing=args.candidate_spacing,
        maximum_per_bin=args.maximum_per_bin,
        maximum_cell_needles=args.maximum_cell_needles,
        score_threshold=args.score_threshold,
        minimum_mode_margin=args.minimum_mode_margin,
        minimum_ownership_margin=args.minimum_ownership_margin,
        minimum_ct_evidence=args.minimum_ct_evidence,
        force=args.force,
    )
    print(
        json.dumps(
            {
                "stats": result["stats"],
                "ranks": [
                    {
                        "rank": value["rank"],
                        "candidateModeCount": value["candidateModeCount"],
                        "acceptedFineFlakeCount": value["acceptedFineFlakeCount"],
                        "gapPixels": value["gapPixels"],
                        "failureProfile": value["failureProfile"],
                        "initialCarrier": value["initialCarrier"],
                        "finalCarrier": value["finalCarrier"],
                        "artifacts": value["artifacts"],
                    }
                    for value in result["ranks"]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
