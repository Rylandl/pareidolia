#!/usr/bin/env python3
"""Fetch a bounded cuboid from an uncompressed Zarr v2 array.

The official Vesuvius masked scroll volumes use uint8, C-order, 128^3 raw
chunks. Keeping this importer deliberately narrow makes the local data contract
auditable and prevents the web API from becoming an arbitrary remote reader.
"""

from __future__ import annotations

import argparse
import json
import urllib.request
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - explicit CLI URL
        return json.load(response)


def read_bytes(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as response:  # noqa: S310 - explicit CLI URL
        return response.read()


def fetch_cuboid(
    base_url: str,
    level: int,
    origin_zyx: tuple[int, int, int],
    shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    base = base_url.rstrip("/")
    metadata = read_json(f"{base}/{level}/.zarray")
    if metadata.get("zarr_format") != 2:
        raise ValueError("only Zarr v2 arrays are supported")
    if metadata.get("compressor") is not None or metadata.get("filters") is not None:
        raise ValueError("only raw, unfiltered Zarr chunks are supported")
    if metadata.get("order") != "C" or metadata.get("dimension_separator") != "/":
        raise ValueError("expected C-order chunks with '/' dimension separators")

    array_shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(metadata["dtype"])
    if len(array_shape) != 3 or len(chunks) != 3:
        raise ValueError("expected a 3D ZYX array")
    if any(size <= 0 for size in shape_zyx):
        raise ValueError("cuboid shape must be positive")
    if any(
        start < 0 or start + size > limit
        for start, size, limit in zip(origin_zyx, shape_zyx, array_shape)
    ):
        raise ValueError(f"cuboid {origin_zyx}+{shape_zyx} is outside array shape {array_shape}")

    output = np.full(shape_zyx, metadata.get("fill_value", 0), dtype=dtype)
    chunk_first = tuple(start // size for start, size in zip(origin_zyx, chunks))
    chunk_last = tuple(
        (start + size - 1) // chunk
        for start, size, chunk in zip(origin_zyx, shape_zyx, chunks)
    )

    for chunk_index in product(
        *(range(first, last + 1) for first, last in zip(chunk_first, chunk_last))
    ):
        chunk_url = f"{base}/{level}/" + "/".join(str(value) for value in chunk_index)
        payload = read_bytes(chunk_url)
        expected = int(np.prod(chunks)) * dtype.itemsize
        if len(payload) != expected:
            raise ValueError(f"chunk {chunk_index} has {len(payload)} bytes; expected {expected}")
        chunk_array = np.frombuffer(payload, dtype=dtype).reshape(chunks)

        chunk_origin = tuple(index * size for index, size in zip(chunk_index, chunks))
        global_lo = tuple(
            max(start, chunk_start) for start, chunk_start in zip(origin_zyx, chunk_origin)
        )
        global_hi = tuple(
            min(start + size, chunk_start + chunk_size)
            for start, size, chunk_start, chunk_size in zip(
                origin_zyx, shape_zyx, chunk_origin, chunks
            )
        )
        output_slices = tuple(
            slice(lo - start, hi - start)
            for lo, hi, start in zip(global_lo, global_hi, origin_zyx)
        )
        chunk_slices = tuple(
            slice(lo - chunk_start, hi - chunk_start)
            for lo, hi, chunk_start in zip(global_lo, global_hi, chunk_origin)
        )
        output[output_slices] = chunk_array[chunk_slices]

    return output, metadata


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Zarr group URL")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument(
        "--origin-zyx", type=int, nargs=3, required=True, metavar=("Z", "Y", "X")
    )
    parser.add_argument(
        "--shape-zyx", type=int, nargs=3, required=True, metavar=("Z", "Y", "X")
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--voxel-size-microns", type=float, required=True)
    parser.add_argument(
        "--suggested-seed-xyz", type=int, nargs=3, metavar=("X", "Y", "Z")
    )
    args = parser.parse_args()

    origin = tuple(args.origin_zyx)
    shape = tuple(args.shape_zyx)
    array, zarr_metadata = fetch_cuboid(args.url, args.level, origin, shape)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.save(args.output, array)

    scale = 2**args.level
    sidecar = {
        "name": args.name,
        "sourceKind": "zarr-cuboid",
        "sourceUrl": args.url,
        "sourceLevel": args.level,
        "sourceShapeZYX": zarr_metadata["shape"],
        "originXYZ": [origin[2] * scale, origin[1] * scale, origin[0] * scale],
        "voxelSizeMicrons": args.voxel_size_microns * scale,
    }
    if args.suggested_seed_xyz:
        sidecar["suggestedSeed"] = dict(zip(("x", "y", "z"), args.suggested_seed_xyz))
    args.output.with_suffix(".json").write_text(json.dumps(sidecar, indent=2) + "\n")
    print(
        f"saved {args.name}: shape={array.shape}, "
        f"range={int(array.min())}..{int(array.max())}, "
        f"nonzero={float(np.count_nonzero(array)) / array.size:.3f}"
    )


if __name__ == "__main__":
    main()
