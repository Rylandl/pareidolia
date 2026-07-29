#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_global_branch_association import associate_global_branches


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Apply overlap-validated joins to the global sparse branch graph with "
            "collision, exact-carrier, and mesh-integrity gates."
        )
    )
    parser.add_argument("--root", default="work/cross-scroll-analysis-z512")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    def progress(
        stage: str, completed: int, total: int, value: dict[str, object]
    ) -> None:
        if stage == "pair-exact":
            print(f"pair exact {completed}/{total}", file=sys.stderr, flush=True)
        else:
            print(
                f"global round {completed} · "
                f"associations {value['mergedAssociationCount']} · "
                f"exact failures {value['failingExactAssociationCount']}",
                file=sys.stderr,
                flush=True,
            )

    result = associate_global_branches(
        args.root,
        force=args.force,
        progress=progress,
    )
    print(
        json.dumps(
            {
                "contract": result["contract"],
                "stats": result["stats"],
                "topAssociations": result["topAssociations"],
                "artifact": result["artifact"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
