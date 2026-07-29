#!/usr/bin/env python3
"""Seed a larger Zarr slab fetch from a completed nested slab."""

from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _bounds(origin: tuple[int, int, int], shape: tuple[int, int, int]) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
    return origin, tuple(start + size for start, size in zip(origin, shape))


def _contains(
    outer_low: tuple[int, int, int],
    outer_high: tuple[int, int, int],
    inner_low: tuple[int, int, int],
    inner_high: tuple[int, int, int],
) -> bool:
    return all(
        outer_start <= inner_start and inner_stop <= outer_stop
        for outer_start, outer_stop, inner_start, inner_stop in zip(
            outer_low, outer_high, inner_low, inner_high
        )
    )


def _intersection(
    first_low: tuple[int, int, int],
    first_high: tuple[int, int, int],
    second_low: tuple[int, int, int],
    second_high: tuple[int, int, int],
) -> tuple[tuple[int, int, int], tuple[int, int, int]] | None:
    low = tuple(max(left, right) for left, right in zip(first_low, second_low))
    high = tuple(min(left, right) for left, right in zip(first_high, second_high))
    return (low, high) if all(start < stop for start, stop in zip(low, high)) else None


def _local_slices(
    low: tuple[int, int, int],
    high: tuple[int, int, int],
    origin: tuple[int, int, int],
) -> tuple[slice, slice, slice]:
    return tuple(
        slice(start - axis_origin, stop - axis_origin)
        for start, stop, axis_origin in zip(low, high, origin)
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--origin-zyx", type=int, nargs=3, required=True)
    parser.add_argument("--shape-zyx", type=int, nargs=3, required=True)
    args = parser.parse_args()

    source_manifest_path = args.source.with_suffix(".fetch.json")
    if not source_manifest_path.is_file():
        raise ValueError(f"missing source fetch manifest: {source_manifest_path}")
    source_manifest = json.loads(source_manifest_path.read_text())
    if source_manifest.get("state") != "complete":
        raise ValueError("source fetch must be complete before it can seed an extension")
    source_identity = source_manifest["identity"]
    source_origin = tuple(int(value) for value in source_identity["originZYX"])
    source_shape = tuple(int(value) for value in source_identity["shapeZYX"])
    target_origin = tuple(int(value) for value in args.origin_zyx)
    target_shape = tuple(int(value) for value in args.shape_zyx)
    source_low, source_high = _bounds(source_origin, source_shape)
    target_low, target_high = _bounds(target_origin, target_shape)
    if not _contains(target_low, target_high, source_low, source_high):
        raise ValueError("target slab must completely contain the source slab")

    target_manifest_path = args.output.with_suffix(".fetch.json")
    if args.output.exists() or target_manifest_path.exists():
        raise ValueError("target output or fetch manifest already exists")
    dtype = np.dtype(source_identity["dtype"])
    chunks = tuple(int(value) for value in source_identity["chunksZYX"])
    source = np.load(args.source, mmap_mode="r")
    if source.shape != source_shape or source.dtype != dtype:
        raise ValueError("source array does not match its fetch manifest")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    target = np.lib.format.open_memmap(
        args.output, mode="w+", dtype=dtype, shape=target_shape
    )
    completed: list[str] = []
    missing: list[str] = []
    source_missing = set(source_manifest.get("missingChunks", []))
    copied_bytes = 0
    for key in source_manifest.get("completedChunks", []):
        index = tuple(int(value) for value in key.split("/"))
        chunk_low = tuple(axis * size for axis, size in zip(index, chunks))
        chunk_high = tuple(start + size for start, size in zip(chunk_low, chunks))
        target_overlap = _intersection(chunk_low, chunk_high, target_low, target_high)
        if target_overlap is None:
            continue
        overlap_low, overlap_high = target_overlap
        if not _contains(source_low, source_high, overlap_low, overlap_high):
            continue
        completed.append(key)
        if key in source_missing:
            missing.append(key)
            continue
        source_slices = _local_slices(overlap_low, overlap_high, source_origin)
        target_slices = _local_slices(overlap_low, overlap_high, target_origin)
        values = source[source_slices]
        target[target_slices] = values
        copied_bytes += int(values.nbytes)
    target.flush()

    source_url = source_identity["sourceUrl"]
    source_level = int(source_identity["sourceLevel"])
    target_identity = {
        "sourceUrl": source_url,
        "sourceLevel": source_level,
        "sourceShapeZYX": source_identity["sourceShapeZYX"],
        "originZYX": list(target_origin),
        "shapeZYX": list(target_shape),
        "chunksZYX": list(chunks),
        "dtype": dtype.str,
        "output": str(args.output),
    }
    chunk_first = tuple(start // chunk for start, chunk in zip(target_origin, chunks))
    chunk_last = tuple(
        (start + extent - 1) // chunk
        for start, extent, chunk in zip(target_origin, target_shape, chunks)
    )
    total_count = int(np.prod([last - first + 1 for first, last in zip(chunk_first, chunk_last)]))
    manifest = {
        "version": 1,
        "identity": target_identity,
        "state": "partial",
        "completedChunks": sorted(completed),
        "missingChunks": sorted(missing),
        "completedCount": len(completed),
        "totalCount": total_count,
        "remainingCount": total_count - len(completed),
        "downloadedBytesThisRun": 0,
        "elapsedSecondsThisRun": 0.0,
        "downloadMiBPerSecond": 0.0,
        "seededFrom": str(args.source),
        "copiedBytes": copied_bytes,
        "updatedAt": datetime.now(UTC).isoformat(),
    }
    _atomic_json(target_manifest_path, manifest)
    print(
        json.dumps(
            {
                "state": manifest["state"],
                "completedCount": manifest["completedCount"],
                "totalCount": manifest["totalCount"],
                "remainingCount": manifest["remainingCount"],
                "copiedBytes": copied_bytes,
                "missingChunks": len(missing),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
