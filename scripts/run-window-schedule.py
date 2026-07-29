#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_window_scheduler import run_window_schedule


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run and reconcile sparse overlapping monotone/association windows "
            "across the occupied Acus grid."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--force-windows", action="store_true")
    parser.add_argument("--window-cells-xyz", type=int, nargs=3, default=[32, 32, 14])
    parser.add_argument("--stride-cells-xyz", type=int, nargs=3, default=[24, 24, 14])
    parser.add_argument("--minimum-primary-flake-claims", type=int, default=1)
    parser.add_argument("--maximum-workers", type=int, default=4)
    parser.add_argument("--no-integrity-audit", action="store_true")
    parser.add_argument("--no-integrity-quarantine", action="store_true")
    args = parser.parse_args()

    def progress(
        stage: str, completed: int, total: int, value: dict[str, object]
    ) -> None:
        if stage == "window":
            origin = value["originCellXYZ"]
            association = value["association"]
            detail = (
                f"origin {origin} · merges {association['retainedMergeCount']}"
            )
        else:
            detail = str(value["classification"])
        print(
            f"{stage} {completed}/{total} · {detail}",
            file=sys.stderr,
            flush=True,
        )

    result = run_window_schedule(
        args.root,
        force=args.force,
        force_windows=args.force_windows,
        settings={
            "windowCellsXYZ": args.window_cells_xyz,
            "strideCellsXYZ": args.stride_cells_xyz,
            "minimumPrimaryFlakeClaimCount": args.minimum_primary_flake_claims,
            "maximumWorkers": args.maximum_workers,
            "runIntegrityAudit": not args.no_integrity_audit,
            "quarantineIntegrityViolations": (
                not args.no_integrity_quarantine
            ),
        },
        progress=progress,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "coverage": result["coverage"],
                "aggregate": result["aggregate"],
                "stats": result["stats"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
