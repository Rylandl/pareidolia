#!/usr/bin/env python3
"""Resumably stream a bounded raw Zarr v2 slab into a sparse NumPy memmap."""

from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import UTC, datetime
from itertools import product
from pathlib import Path
from typing import Any

import numpy as np


def _read_json(url: str) -> dict[str, Any]:
    with urllib.request.urlopen(url, timeout=30) as response:  # noqa: S310 - explicit CLI source
        return json.load(response)


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _chunk_key(index: tuple[int, int, int]) -> str:
    return "/".join(str(value) for value in index)


def _download_chunk(url: str, expected: int, retries: int) -> tuple[bytes | None, int]:
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=90) as response:  # noqa: S310 - explicit CLI source
                payload = response.read()
            if len(payload) != expected:
                raise ValueError(f"chunk returned {len(payload)} bytes; expected {expected}")
            return payload, len(payload)
        except urllib.error.HTTPError as error:
            if error.code == 404:
                return None, 0
            if attempt >= retries:
                raise
        except (TimeoutError, urllib.error.URLError, ValueError):
            if attempt >= retries:
                raise
        time.sleep(2**attempt)
    raise RuntimeError("unreachable download retry state")


def _chunk_overlap(
    chunk_index: tuple[int, int, int],
    chunks: tuple[int, int, int],
    origin: tuple[int, int, int],
    shape: tuple[int, int, int],
) -> tuple[tuple[slice, slice, slice], tuple[slice, slice, slice]]:
    chunk_origin = tuple(index * size for index, size in zip(chunk_index, chunks))
    global_low = tuple(max(start, chunk_start) for start, chunk_start in zip(origin, chunk_origin))
    global_high = tuple(
        min(start + extent, chunk_start + chunk_extent)
        for start, extent, chunk_start, chunk_extent in zip(origin, shape, chunk_origin, chunks)
    )
    output_slices = tuple(
        slice(low - start, high - start)
        for low, high, start in zip(global_low, global_high, origin)
    )
    chunk_slices = tuple(
        slice(low - chunk_start, high - chunk_start)
        for low, high, chunk_start in zip(global_low, global_high, chunk_origin)
    )
    return output_slices, chunk_slices


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True, help="Zarr group URL")
    parser.add_argument("--level", type=int, default=0)
    parser.add_argument("--origin-zyx", type=int, nargs=3, required=True)
    parser.add_argument("--shape-zyx", type=int, nargs=3, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--voxel-size-microns", type=float, required=True)
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--retries", type=int, default=4)
    parser.add_argument("--limit-chunks", type=int, help="Validation-only cap")
    args = parser.parse_args()

    base_url = args.url.rstrip("/")
    metadata = _read_json(f"{base_url}/{args.level}/.zarray")
    if metadata.get("zarr_format") != 2:
        raise ValueError("only Zarr v2 arrays are supported")
    if metadata.get("compressor") is not None or metadata.get("filters") is not None:
        raise ValueError("only raw, unfiltered arrays are supported")
    if metadata.get("order") != "C" or metadata.get("dimension_separator") != "/":
        raise ValueError("expected C-order chunks with '/' separators")
    source_shape = tuple(int(value) for value in metadata["shape"])
    chunks = tuple(int(value) for value in metadata["chunks"])
    dtype = np.dtype(metadata["dtype"])
    origin = tuple(int(value) for value in args.origin_zyx)
    shape = tuple(int(value) for value in args.shape_zyx)
    if any(
        start < 0 or start + extent > limit
        for start, extent, limit in zip(origin, shape, source_shape)
    ):
        raise ValueError(f"requested slab {origin}+{shape} is outside {source_shape}")

    chunk_first = tuple(start // chunk for start, chunk in zip(origin, chunks))
    chunk_last = tuple(
        (start + extent - 1) // chunk
        for start, extent, chunk in zip(origin, shape, chunks)
    )
    chunk_indices = list(
        product(*(range(first, last + 1) for first, last in zip(chunk_first, chunk_last)))
    )
    manifest_path = args.output.with_suffix(".fetch.json")
    identity = {
        "sourceUrl": base_url,
        "sourceLevel": args.level,
        "sourceShapeZYX": list(source_shape),
        "originZYX": list(origin),
        "shapeZYX": list(shape),
        "chunksZYX": list(chunks),
        "dtype": dtype.str,
        "output": str(args.output),
    }
    completed: set[str] = set()
    missing: set[str] = set()
    downloaded_bytes = 0
    started = time.monotonic()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("identity") != identity:
            raise ValueError("existing fetch manifest does not match the requested slab")
        completed = set(manifest.get("completedChunks", []))
        missing = set(manifest.get("missingChunks", []))
        output = np.load(args.output, mmap_mode="r+")
        if output.shape != shape or output.dtype != dtype:
            raise ValueError("existing slab array does not match its fetch manifest")
    else:
        if args.output.exists():
            raise ValueError("output exists without a matching fetch manifest; choose a new path")
        output = np.lib.format.open_memmap(args.output, mode="w+", dtype=dtype, shape=shape)

    pending = [index for index in chunk_indices if _chunk_key(index) not in completed]
    if args.limit_chunks is not None:
        pending = pending[: max(0, args.limit_chunks)]
    expected_chunk_bytes = int(np.prod(chunks)) * dtype.itemsize

    def status_payload(state: str) -> dict[str, Any]:
        elapsed = max(time.monotonic() - started, 1.0e-6)
        return {
            "version": 1,
            "identity": identity,
            "state": state,
            "completedChunks": sorted(completed),
            "missingChunks": sorted(missing),
            "completedCount": len(completed),
            "totalCount": len(chunk_indices),
            "remainingCount": len(chunk_indices) - len(completed),
            "downloadedBytesThisRun": downloaded_bytes,
            "elapsedSecondsThisRun": round(elapsed, 3),
            "downloadMiBPerSecond": round(downloaded_bytes / elapsed / 2**20, 2),
            "updatedAt": datetime.now(UTC).isoformat(),
        }

    try:
        with ThreadPoolExecutor(max_workers=max(1, args.workers)) as executor:
            futures = {
                executor.submit(
                    _download_chunk,
                    f"{base_url}/{args.level}/{_chunk_key(index)}",
                    expected_chunk_bytes,
                    max(0, args.retries),
                ): index
                for index in pending
            }
            for count, future in enumerate(as_completed(futures), start=1):
                index = futures[future]
                payload, received = future.result()
                key = _chunk_key(index)
                if payload is None:
                    missing.add(key)
                else:
                    chunk = np.frombuffer(payload, dtype=dtype).reshape(chunks)
                    output_slices, chunk_slices = _chunk_overlap(index, chunks, origin, shape)
                    output[output_slices] = chunk[chunk_slices]
                    downloaded_bytes += received
                completed.add(key)
                if count % 32 == 0 or count == len(pending):
                    output.flush()
                    state = "complete" if len(completed) == len(chunk_indices) else "running"
                    progress = status_payload(state)
                    _atomic_json(manifest_path, progress)
                    print(
                        f"{progress['completedCount']}/{progress['totalCount']} chunks · "
                        f"{progress['downloadMiBPerSecond']:.1f} MiB/s · "
                        f"{len(missing)} fill chunks",
                        flush=True,
                    )
    except BaseException:
        output.flush()
        _atomic_json(manifest_path, status_payload("interrupted"))
        raise

    output.flush()
    final_state = "complete" if len(completed) == len(chunk_indices) else "partial"
    _atomic_json(manifest_path, status_payload(final_state))
    sidecar = {
        "name": args.name,
        "originXYZ": [origin[2] * 2**args.level, origin[1] * 2**args.level, origin[0] * 2**args.level],
        "voxelSizeMicrons": args.voxel_size_microns * 2**args.level,
        "sourceKind": "zarr-slab",
        "sourceUrl": base_url,
        "sourceLevel": args.level,
        "sourceShapeZYX": source_shape,
        "slabShapeZYX": shape,
        "fetchManifest": str(manifest_path),
    }
    _atomic_json(args.output.with_suffix(".json"), sidecar)
    final_status = status_payload(final_state)
    print(
        json.dumps(
            {
                key: final_status[key]
                for key in (
                    "state",
                    "completedCount",
                    "totalCount",
                    "remainingCount",
                    "downloadedBytesThisRun",
                    "elapsedSecondsThisRun",
                    "downloadMiBPerSecond",
                    "updatedAt",
                )
            },
            indent=2,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
