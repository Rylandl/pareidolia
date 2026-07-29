#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from backend.slab_global_boundary_candidates import (  # noqa: E402
    build_global_boundary_candidates,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Discover conservative whole-volume branch-boundary candidates that "
            "were absent from every local-window candidate list."
        )
    )
    parser.add_argument("--root", required=True, help="slab analysis directory")
    parser.add_argument("--force", action="store_true", help="ignore a valid cache")
    args = parser.parse_args()

    def progress(
        stage: str, completed: int, total: int, details: dict[str, object]
    ) -> None:
        print(
            json.dumps(
                {
                    "stage": stage,
                    "completed": completed,
                    "total": total,
                    **details,
                }
            ),
            flush=True,
        )

    summary = build_global_boundary_candidates(
        args.root, force=args.force, progress=progress
    )
    print(json.dumps(summary["stats"], indent=2))


if __name__ == "__main__":
    main()
