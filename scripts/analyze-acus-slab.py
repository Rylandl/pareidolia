#!/usr/bin/env python3
"""Run or resume the chunked Acus analysis for a downloaded cross-scroll slab."""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from backend.slab_analysis import run_slab_analysis, slab_status  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--grid-stride", type=int, default=32)
    parser.add_argument("--tile-core", type=int, default=128)
    parser.add_argument("--calibration-tiles", type=int, default=96)
    parser.add_argument(
        "--strength-scale",
        type=float,
        help="reuse a positive global Hessian strength scale from a nested slab",
    )
    parser.add_argument("--limit-tiles", type=int)
    parser.add_argument("--limit-cells", type=int)
    parser.add_argument(
        "--compute",
        choices=("auto", "cpu", "gpu"),
        default="auto",
        help="select the line-field backend; gpu fails closed instead of falling back",
    )
    parser.add_argument("--status", action="store_true")
    args = parser.parse_args()
    if args.status:
        print(json.dumps(slab_status(args.output), indent=2))
        return
    os.environ["ACUS_COMPUTE"] = args.compute
    request = {
        "cubeSize": 64,
        "scale": 1.25,
        "spacing": 4,
        "needleLength": 16,
        "halo": 16,
        "gridStride": args.grid_stride,
        "tileCore": args.tile_core,
        "binSize": 32,
        "maxNeedlesPerBin": 32,
        "maxNeedles": 160,
        "calibrationTiles": args.calibration_tiles,
    }
    if args.strength_scale is not None:
        request["strengthScale"] = args.strength_scale
    if args.limit_tiles is not None:
        request["limitTiles"] = args.limit_tiles
    if args.limit_cells is not None:
        request["limitCells"] = args.limit_cells
    result = run_slab_analysis(args.source, args.output, request)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
