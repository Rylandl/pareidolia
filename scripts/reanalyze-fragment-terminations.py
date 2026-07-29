#!/usr/bin/env python3
"""Run targeted dense Acus at ranked, genuinely open v6 fragment termini."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.slab_termination_reanalysis import (  # noqa: E402
    reanalyze_fragment_terminations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Compare the existing coarse Acus catalog with targeted dense "
            "re-extraction at ranked open fragment termini."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--maximum-targets", type=int, default=128)
    parser.add_argument("--candidate-spacing", type=int, default=2)
    parser.add_argument("--maximum-crop-voxels", type=int, default=4_000_000)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def progress(
        stage: str, completed: int, total: int, details: dict[str, object]
    ) -> None:
        if stage == "crops":
            print(
                f"crops {completed}/{total} · targets {details['targetCount']} · "
                f"needles {details['deduplicatedNeedleCount']}",
                file=sys.stderr,
                flush=True,
            )
        elif stage == "targets" and (
            completed % 8 == 0 or completed == total
        ):
            print(
                f"targets {completed}/{total} · {details['classification']}",
                file=sys.stderr,
                flush=True,
            )

    result = reanalyze_fragment_terminations(
        args.root,
        settings={
            "maximumTargets": args.maximum_targets,
            "candidateSpacingVoxels": args.candidate_spacing,
            "maximumCropVoxels": args.maximum_crop_voxels,
        },
        force=args.force,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "summaryPath": str(
                    args.root / "fragment-termination-reanalysis-v1.json"
                ),
                "topRecoveredTargets": [
                    {
                        "queueRank": value["queueRank"],
                        "classification": value["classification"],
                        "clusterIndex": value["clusterIndex"],
                        "associationId": value["associationId"],
                        "targetXYZ": value["targetXYZ"],
                        "denseScore": value["dense"]["bestAssociationScore"],
                        "coarseScore": value["coarse"]["bestAssociationScore"],
                        "ownershipMargin": value["dense"]["ownershipMargin"],
                    }
                    for value in result["recoveredTargets"][:20]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
