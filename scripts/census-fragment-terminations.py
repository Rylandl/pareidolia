#!/usr/bin/env python3
"""Classify and rank definite termini of the final global fragment catalog."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.slab_fragment_termination_census import (  # noqa: E402
    census_fragment_terminations,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Classify degree-one ends on substantial final fragments and rank "
            "dense-Acus, order, and geometry follow-up targets."
        )
    )
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--minimum-fragment-size", type=int, default=25)
    parser.add_argument("--maximum-dense-targets", type=int, default=128)
    parser.add_argument("--maximum-ct-samples", type=int, default=512)
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def progress(
        stage: str, completed: int, total: int, details: dict[str, object]
    ) -> None:
        if stage == "local-evidence":
            print(
                f"local evidence {completed}/{total} · "
                f"candidates {details['candidateCount']}",
                file=sys.stderr,
                flush=True,
            )
        elif stage == "geometry":
            print(
                f"geometry {completed}/{total} · z {details['zIndex']}",
                file=sys.stderr,
                flush=True,
            )
        elif stage == "ct-targets":
            print(
                f"CT targets {completed}/{total}",
                file=sys.stderr,
                flush=True,
            )

    result = census_fragment_terminations(
        args.root,
        settings={
            "minimumAssociationNodeCount": args.minimum_fragment_size,
            "maximumDenseAcusTargetCount": args.maximum_dense_targets,
            "maximumCtSampledClusterCount": args.maximum_ct_samples,
        },
        force=args.force,
        progress=progress,
    )

    def compact(record: dict[str, object]) -> dict[str, object]:
        target_ct = record["targetCt"]
        evidence = record["bestEvidence"]
        assert isinstance(target_ct, dict)
        assert isinstance(evidence, dict)
        return {
            "clusterIndex": record["clusterIndex"],
            "category": record["category"],
            "associationId": record["associationId"],
            "associationNodeCount": record["associationNodeCount"],
            "endpointCount": record["endpointCount"],
            "targetXYZ": record["targetXYZ"],
            "targetMaterialFraction": target_ct["materialFraction"],
            "evidenceSource": evidence["source"],
            "priority": record["denseAcusPriority"],
        }

    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "artifact": result["artifact"],
                "summaryPath": str(
                    args.root / "fragment-termination-census-v1.json"
                ),
                "topDenseAcusTargets": [
                    compact(value)
                    for value in result["queues"]["denseAcus"][:10]
                ],
                "topOrderReviewTargets": [
                    compact(value)
                    for value in result["queues"]["orderReview"][:5]
                ],
                "topGeometryReviewTargets": [
                    compact(value)
                    for value in result["queues"]["geometryReview"][:5]
                ],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
