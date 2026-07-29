from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rectify import _trilinear
from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_normal_families import NORMAL_FAMILY_VERSION


MATERIAL_INTERVAL_VERSION = 1

LABEL_AIR = 0
LABEL_UNASSIGNED_MATERIAL = 1
LABEL_SINGLY_CLAIMED_MATERIAL = 2
LABEL_CONTESTED_MATERIAL = 3
LABEL_NAMES = {
    LABEL_AIR: "air",
    LABEL_UNASSIGNED_MATERIAL: "unassigned-material",
    LABEL_SINGLY_CLAIMED_MATERIAL: "singly-claimed-material",
    LABEL_CONTESTED_MATERIAL: "contested-material",
}

PROFILE_DTYPE = np.dtype(
    [
        ("cellIndex", "<u2", (3,)),
        ("normalFamily", "u1"),
        ("centerXYZ", "<f4", (3,)),
        ("normalXYZ", "<f4", (3,)),
        ("normalConfidence", "<f4"),
        ("familyComponentId", "<i4"),
        ("familyComponentSize", "<u4"),
        ("claimOffset", "<u4"),
        ("claimCount", "u1"),
        ("supportedClaimCount", "u1"),
        ("intervalOffset", "<u4"),
        ("intervalCount", "u1"),
        ("airSampleCount", "u1"),
        ("materialSampleCount", "u1"),
        ("unassignedSampleCount", "u1"),
        ("singlyClaimedSampleCount", "u1"),
        ("contestedSampleCount", "u1"),
        ("unassignedIntervalCount", "u1"),
        ("singlyClaimedIntervalCount", "u1"),
        ("contestedIntervalCount", "u1"),
    ]
)

INTERVAL_DTYPE = np.dtype(
    [
        ("profileIndex", "<u4"),
        ("localIntervalIndex", "u1"),
        ("startSampleIndex", "u1"),
        ("stopSampleIndex", "u1"),
        ("startDepthVoxels", "<f4"),
        ("stopDepthVoxels", "<f4"),
        ("sampleCount", "u1"),
        ("state", "u1"),
        ("claimCount", "u1"),
        ("claimClusterCount", "u1"),
        ("boundaryTruncated", "u1"),
        ("observedThicknessVoxels", "<f4"),
        ("apparentCtThicknessVoxels", "<f4"),
    ]
)

CLAIM_DTYPE = np.dtype(
    [
        ("profileIndex", "<u4"),
        ("sourceZIndex", "u1"),
        ("sourceFlakeId", "<u4"),
        ("normalFamily", "u1"),
        ("depthVoxels", "<f4"),
        ("sampleIndex", "<i2"),
        ("quality", "<f4"),
        ("intervalIndex", "<i4"),
        ("clusterIndex", "<i2"),
        ("supported", "u1"),
    ]
)


DEFAULT_SETTINGS: dict[str, Any] = {
    "depthMinimumVoxels": -32.0,
    "depthMaximumVoxels": 32.0,
    "depthStepVoxels": 1.0,
    "smoothingKernel": [1.0, 4.0, 6.0, 4.0, 1.0],
    "maximumBridgedAirGapSamples": 1,
    "claimSupportToleranceVoxels": 1.5,
    "claimClusterGapVoxels": 3.0,
    "tileCellsZYX": [7, 16, 16],
    "thresholdSensitivityRawValues": [24.0, 32.0, 40.0, 48.0, 64.0, 80.0, 96.0, 112.0, 128.0],
    "thresholdSensitivityProfileCount": 20000,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npy(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, array, allow_pickle=False)
    temporary.replace(path)


def _content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _quantiles(values: np.ndarray, digits: int = 4) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    if not len(values):
        return {name: None for name in names}
    percentiles = (0, 10, 25, 50, 75, 90, 100)
    return {
        name: round(float(value), digits)
        for name, value in zip(names, np.percentile(values, percentiles))
    }


def _smoothed_material_mask(
    intensities: np.ndarray,
    air_threshold: float,
    smoothing_kernel: list[float] | np.ndarray | None = None,
    maximum_bridged_air_gap_samples: int = 1,
) -> np.ndarray:
    """Classify CT material independently of any flake or sheet hypothesis."""
    values = np.asarray(intensities, dtype=np.float32)
    one_dimensional = values.ndim == 1
    if one_dimensional:
        values = values[None, :]
    if values.ndim != 2:
        raise ValueError("intensities must have shape (profiles, depth) or (depth,)")
    kernel = np.asarray(
        smoothing_kernel if smoothing_kernel is not None else [1, 4, 6, 4, 1],
        dtype=np.float32,
    )
    if not len(kernel) or len(kernel) % 2 == 0 or float(np.sum(kernel)) <= 0.0:
        raise ValueError("smoothing kernel must have positive odd length")
    kernel /= float(np.sum(kernel))
    radius = len(kernel) // 2
    padded = np.pad(values, ((0, 0), (radius, radius)), mode="edge")
    smoothed = np.zeros_like(values, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        smoothed += float(weight) * padded[:, offset : offset + values.shape[1]]
    material = smoothed > float(air_threshold)
    maximum_gap = max(0, int(maximum_bridged_air_gap_samples))
    for gap_size in range(1, maximum_gap + 1):
        if values.shape[1] <= gap_size + 1:
            break
        bounded = material[:, : -(gap_size + 1)] & material[:, gap_size + 1 :]
        for offset in range(1, gap_size + 1):
            material[:, offset : offset + bounded.shape[1]] |= bounded
    return material[0] if one_dimensional else material


def _material_runs(material: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(material, dtype=bool)
    if mask.ndim != 1:
        raise ValueError("material mask must be one dimensional")
    padded = np.pad(mask.astype(np.int8), (1, 1))
    transitions = np.diff(padded)
    starts = np.flatnonzero(transitions == 1)
    stops = np.flatnonzero(transitions == -1) - 1
    return [(int(start), int(stop)) for start, stop in zip(starts, stops)]


def _annotate_profile(
    material: np.ndarray,
    depth_offsets: np.ndarray,
    claim_depths: np.ndarray,
    claim_support_tolerance: float = 1.5,
    claim_cluster_gap: float = 3.0,
) -> dict[str, Any]:
    """Overlay local flake claims without assigning a physical sheet identity."""
    material = np.asarray(material, dtype=bool)
    depths = np.asarray(depth_offsets, dtype=np.float32)
    claims = np.asarray(claim_depths, dtype=np.float32)
    if material.ndim != 1 or depths.ndim != 1 or len(material) != len(depths):
        raise ValueError("material and depth offsets must be equal-length vectors")
    if len(depths) < 2 or not np.all(np.diff(depths) > 0.0):
        raise ValueError("depth offsets must be strictly increasing")
    step = float(np.median(np.diff(depths)))
    runs = _material_runs(material)
    labels = np.full(len(material), LABEL_AIR, dtype=np.uint8)
    labels[material] = LABEL_UNASSIGNED_MATERIAL
    claim_interval = np.full(len(claims), -1, dtype=np.int32)
    claim_cluster = np.full(len(claims), -1, dtype=np.int16)
    run_claims: list[list[int]] = [[] for _ in runs]
    for claim_index, claim_depth in enumerate(claims):
        if not np.isfinite(claim_depth) or not runs:
            continue
        distances = []
        for start, stop in runs:
            low = float(depths[start]) - 0.5 * step
            high = float(depths[stop]) + 0.5 * step
            distances.append(max(low - float(claim_depth), float(claim_depth) - high, 0.0))
        run_index = int(np.argmin(distances))
        if float(distances[run_index]) <= float(claim_support_tolerance):
            claim_interval[claim_index] = run_index
            run_claims[run_index].append(claim_index)

    intervals = []
    for run_index, (start, stop) in enumerate(runs):
        indices = sorted(run_claims[run_index], key=lambda index: float(claims[index]))
        cluster_count = 0
        previous_depth = None
        for claim_index in indices:
            claim_depth = float(claims[claim_index])
            if previous_depth is None or claim_depth - previous_depth > float(claim_cluster_gap):
                cluster_count += 1
            claim_cluster[claim_index] = cluster_count - 1
            previous_depth = claim_depth
        if cluster_count == 1:
            state = LABEL_SINGLY_CLAIMED_MATERIAL
        elif cluster_count >= 2:
            state = LABEL_CONTESTED_MATERIAL
        else:
            state = LABEL_UNASSIGNED_MATERIAL
        labels[start : stop + 1] = state
        boundary_truncated = start == 0 or stop == len(material) - 1
        observed_thickness = (stop - start + 1) * step
        intervals.append(
            {
                "startSampleIndex": start,
                "stopSampleIndex": stop,
                "startDepthVoxels": float(depths[start]),
                "stopDepthVoxels": float(depths[stop]),
                "sampleCount": stop - start + 1,
                "state": state,
                "claimCount": len(indices),
                "claimClusterCount": cluster_count,
                "boundaryTruncated": boundary_truncated,
                "observedThicknessVoxels": observed_thickness,
                "apparentCtThicknessVoxels": (
                    observed_thickness
                    if state == LABEL_SINGLY_CLAIMED_MATERIAL and not boundary_truncated
                    else math.nan
                ),
            }
        )
    return {
        "labels": labels,
        "intervals": intervals,
        "claimIntervalIndex": claim_interval,
        "claimClusterIndex": claim_cluster,
    }


def _input_artifacts(root: Path, plane_count: int) -> list[dict[str, Any]]:
    paths = [
        root / "analysis.json",
        root / "grid.json",
        root / "cells.npy",
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.json",
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.npy",
    ]
    paths.extend(
        root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        for z_index in range(plane_count)
    )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError("material interval inputs are incomplete: " + ", ".join(missing))
    return [_content_identity(path) for path in paths]


def _profile_catalog(
    cells: np.ndarray,
    families: np.ndarray,
    grid: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.asarray(cells["valid"], dtype=bool)
    secondary = valid & np.asarray(families["included"], dtype=bool)
    primary_zyx = np.argwhere(valid)
    secondary_zyx = np.argwhere(secondary)
    profile_count = len(primary_zyx) + len(secondary_zyx)
    profiles = np.zeros(profile_count, dtype=PROFILE_DTYPE)
    lookup = np.full((2, *valid.shape), -1, dtype=np.int32)
    grid_x = np.asarray(grid["x"], dtype=np.float32)
    grid_y = np.asarray(grid["y"], dtype=np.float32)
    grid_z = np.asarray(grid["z"], dtype=np.float32)

    def populate(
        positions: np.ndarray,
        offset: int,
        normal_family: int,
    ) -> None:
        if not len(positions):
            return
        indices = np.arange(offset, offset + len(positions), dtype=np.int32)
        iz, iy, ix = positions.T
        profiles["cellIndex"][indices] = np.stack([ix, iy, iz], axis=1)
        profiles["normalFamily"][indices] = normal_family
        profiles["centerXYZ"][indices] = np.stack(
            [grid_x[ix], grid_y[iy], grid_z[iz]], axis=1
        )
        if normal_family == 0:
            profiles["normalXYZ"][indices] = cells["normal"][iz, iy, ix]
            profiles["normalConfidence"][indices] = cells["normalConfidence"][iz, iy, ix]
            profiles["familyComponentId"][indices] = -1
        else:
            profiles["normalXYZ"][indices] = families["secondaryNormal"][iz, iy, ix]
            profiles["normalConfidence"][indices] = families["secondaryConfidence"][iz, iy, ix]
            profiles["familyComponentId"][indices] = families["componentId"][iz, iy, ix]
            profiles["familyComponentSize"][indices] = families["componentSize"][iz, iy, ix]
        lookup[normal_family, iz, iy, ix] = indices

    populate(primary_zyx, 0, 0)
    populate(secondary_zyx, len(primary_zyx), 1)
    norms = np.linalg.norm(profiles["normalXYZ"], axis=1)
    if np.any(norms < 0.5):
        raise ValueError("profile catalog contains a degenerate normal")
    profiles["normalXYZ"] /= norms[:, None]
    return profiles, lookup


def _sample_profiles(
    source: np.ndarray,
    profiles: np.ndarray,
    depth_offsets: np.ndarray,
    tile_cells_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, dict[str, Any]]:
    started = time.monotonic()
    profile_count = len(profiles)
    intensities = np.empty((profile_count, len(depth_offsets)), dtype=np.uint8)
    cell_indices = np.asarray(profiles["cellIndex"], dtype=np.int32)
    tile_z, tile_y, tile_x = (max(1, int(value)) for value in tile_cells_zyx)
    grid_shape_xyz = np.max(cell_indices, axis=0) + 1
    tile_shape_xyz = np.asarray([tile_x, tile_y, tile_z], dtype=np.int32)
    tile_counts_xyz = np.ceil(grid_shape_xyz / tile_shape_xyz).astype(np.int32)
    tile_xyz = cell_indices // tile_shape_xyz
    tile_ids = (
        (tile_xyz[:, 2] * tile_counts_xyz[1] + tile_xyz[:, 1])
        * tile_counts_xyz[0]
        + tile_xyz[:, 0]
    )
    order = np.argsort(tile_ids, kind="stable")
    ordered_tiles = tile_ids[order]
    boundaries = np.flatnonzero(np.r_[True, ordered_tiles[1:] != ordered_tiles[:-1], True])
    source_shape_xyz = np.asarray([source.shape[2], source.shape[1], source.shape[0]])
    sampled_count = 0
    input_bytes = 0
    maximum_subvolume_bytes = 0
    last_report = started
    for group_index in range(len(boundaries) - 1):
        indices = order[boundaries[group_index] : boundaries[group_index + 1]]
        points = (
            profiles["centerXYZ"][indices, None, :]
            + profiles["normalXYZ"][indices, None, :]
            * depth_offsets[None, :, None]
        )
        low = np.floor(np.min(points, axis=(0, 1)) - 1.0).astype(np.int32)
        high = np.ceil(np.max(points, axis=(0, 1)) + 2.0).astype(np.int32)
        low = np.maximum(low, 0)
        high = np.minimum(high, source_shape_xyz)
        x0, y0, z0 = (int(value) for value in low)
        x1, y1, z1 = (int(value) for value in high)
        subvolume = np.array(source[z0:z1, y0:y1, x0:x1], copy=True)
        origin = np.asarray([x0, y0, z0], dtype=np.float32)
        sampled = _trilinear(subvolume, points - origin)
        intensities[indices] = np.clip(np.rint(sampled), 0.0, 255.0).astype(np.uint8)
        sampled_count += len(indices)
        input_bytes += subvolume.nbytes
        maximum_subvolume_bytes = max(maximum_subvolume_bytes, subvolume.nbytes)
        now = time.monotonic()
        if now - last_report >= 10.0 or group_index == len(boundaries) - 2:
            elapsed = max(now - started, 1.0e-6)
            print(
                f"material profiles {sampled_count}/{profile_count} · "
                f"{sampled_count / elapsed:.1f} profiles/s · "
                f"{input_bytes / (1024**3):.1f} GiB read",
                flush=True,
            )
            last_report = now
    return intensities, {
        "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        "tileCount": len(boundaries) - 1,
        "sampledProfileCount": sampled_count,
        "sourceBytesReadIncludingTileOverlap": int(input_bytes),
        "maximumSourceSubvolumeBytes": int(maximum_subvolume_bytes),
    }


def _load_claims(
    root: Path,
    plane_count: int,
    profile_lookup: np.ndarray,
    profile_count: int,
    depth_offsets: np.ndarray,
) -> np.ndarray:
    claims = np.empty(profile_count * 3, dtype=CLAIM_DTYPE)
    claims["intervalIndex"] = -1
    claims["clusterIndex"] = -1
    cursor = 0
    depth_minimum = float(depth_offsets[0])
    depth_step = float(np.median(np.diff(depth_offsets)))
    for z_index in range(plane_count):
        payload = json.loads(
            (root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json").read_text()
        )
        flakes = payload["flakes"]
        count = len(flakes)
        if cursor + count > len(claims):
            raise ValueError("more than three flake claims were found per profile")
        output = claims[cursor : cursor + count]
        cell = np.asarray([value["cellIndex"] for value in flakes], dtype=np.int32)
        family = np.fromiter(
            (int(value.get("normalFamily", 0)) for value in flakes),
            dtype=np.uint8,
            count=count,
        )
        profile_index = profile_lookup[family, cell[:, 2], cell[:, 1], cell[:, 0]]
        if np.any(profile_index < 0):
            raise ValueError(f"plane {z_index} contains a flake without a profile")
        claim_depth = np.fromiter(
            (float(value["depthOffset"]) for value in flakes),
            dtype=np.float32,
            count=count,
        )
        output["profileIndex"] = profile_index
        output["sourceZIndex"] = z_index
        output["sourceFlakeId"] = np.fromiter(
            (int(value["id"]) for value in flakes), dtype=np.uint32, count=count
        )
        output["normalFamily"] = family
        output["depthVoxels"] = claim_depth
        output["sampleIndex"] = np.rint(
            (claim_depth - depth_minimum) / depth_step
        ).astype(np.int16)
        output["quality"] = np.fromiter(
            (float(value["quality"]) for value in flakes),
            dtype=np.float32,
            count=count,
        )
        output["supported"] = 0
        cursor += count
    claims = claims[:cursor].copy()
    order = np.lexsort(
        (claims["sourceFlakeId"], claims["depthVoxels"], claims["profileIndex"])
    )
    return claims[order]


def _annotate_catalog(
    profiles: np.ndarray,
    material: np.ndarray,
    depth_offsets: np.ndarray,
    claims: np.ndarray,
    claim_support_tolerance: float,
    claim_cluster_gap: float,
) -> tuple[np.ndarray, np.ndarray]:
    claim_counts = np.bincount(
        claims["profileIndex"], minlength=len(profiles)
    ).astype(np.int32)
    claim_offsets = np.r_[0, np.cumsum(claim_counts, dtype=np.int64)]
    profiles["claimOffset"] = claim_offsets[:-1]
    profiles["claimCount"] = claim_counts
    run_starts = material & ~np.pad(material[:, :-1], ((0, 0), (1, 0)))
    interval_counts = np.count_nonzero(run_starts, axis=1).astype(np.int32)
    interval_offsets = np.r_[0, np.cumsum(interval_counts, dtype=np.int64)]
    profiles["intervalOffset"] = interval_offsets[:-1]
    profiles["intervalCount"] = interval_counts
    intervals = np.empty(int(interval_offsets[-1]), dtype=INTERVAL_DTYPE)
    intervals["apparentCtThicknessVoxels"] = np.nan
    labels = np.empty_like(material, dtype=np.uint8)
    started = time.monotonic()
    last_report = started
    for profile_index in range(len(profiles)):
        claim_low = int(claim_offsets[profile_index])
        claim_high = int(claim_offsets[profile_index + 1])
        annotation = _annotate_profile(
            material[profile_index],
            depth_offsets,
            claims["depthVoxels"][claim_low:claim_high],
            claim_support_tolerance,
            claim_cluster_gap,
        )
        labels[profile_index] = annotation["labels"]
        interval_low = int(interval_offsets[profile_index])
        for local_index, value in enumerate(annotation["intervals"]):
            interval_index = interval_low + local_index
            record = intervals[interval_index]
            record["profileIndex"] = profile_index
            record["localIntervalIndex"] = local_index
            for key in (
                "startSampleIndex",
                "stopSampleIndex",
                "startDepthVoxels",
                "stopDepthVoxels",
                "sampleCount",
                "state",
                "claimCount",
                "claimClusterCount",
                "boundaryTruncated",
                "observedThicknessVoxels",
                "apparentCtThicknessVoxels",
            ):
                record[key] = value[key]
        local_claim_intervals = annotation["claimIntervalIndex"]
        supported = local_claim_intervals >= 0
        claims["supported"][claim_low:claim_high] = supported
        claims["intervalIndex"][claim_low:claim_high] = np.where(
            supported, interval_low + local_claim_intervals, -1
        )
        claims["clusterIndex"][claim_low:claim_high] = annotation[
            "claimClusterIndex"
        ]
        profile_labels = labels[profile_index]
        profiles["supportedClaimCount"][profile_index] = int(np.count_nonzero(supported))
        profiles["airSampleCount"][profile_index] = int(
            np.count_nonzero(profile_labels == LABEL_AIR)
        )
        profiles["materialSampleCount"][profile_index] = int(
            np.count_nonzero(profile_labels != LABEL_AIR)
        )
        profiles["unassignedSampleCount"][profile_index] = int(
            np.count_nonzero(profile_labels == LABEL_UNASSIGNED_MATERIAL)
        )
        profiles["singlyClaimedSampleCount"][profile_index] = int(
            np.count_nonzero(profile_labels == LABEL_SINGLY_CLAIMED_MATERIAL)
        )
        profiles["contestedSampleCount"][profile_index] = int(
            np.count_nonzero(profile_labels == LABEL_CONTESTED_MATERIAL)
        )
        states = np.asarray([value["state"] for value in annotation["intervals"]])
        profiles["unassignedIntervalCount"][profile_index] = int(
            np.count_nonzero(states == LABEL_UNASSIGNED_MATERIAL)
        )
        profiles["singlyClaimedIntervalCount"][profile_index] = int(
            np.count_nonzero(states == LABEL_SINGLY_CLAIMED_MATERIAL)
        )
        profiles["contestedIntervalCount"][profile_index] = int(
            np.count_nonzero(states == LABEL_CONTESTED_MATERIAL)
        )
        now = time.monotonic()
        if now - last_report >= 10.0 or profile_index == len(profiles) - 1:
            elapsed = max(now - started, 1.0e-6)
            print(
                f"material ownership {profile_index + 1}/{len(profiles)} · "
                f"{(profile_index + 1) / elapsed:.1f} profiles/s",
                flush=True,
            )
            last_report = now
    return labels, intervals


def _family_stats(
    family: int,
    profiles: np.ndarray,
    claims: np.ndarray,
    labels: np.ndarray,
) -> dict[str, Any]:
    selected_profiles = profiles["normalFamily"] == family
    selected_claims = claims["normalFamily"] == family
    profile_count = int(np.count_nonzero(selected_profiles))
    sample_count = profile_count * labels.shape[1]
    return {
        "profileCount": profile_count,
        "claimCount": int(np.count_nonzero(selected_claims)),
        "supportedClaimCount": int(np.count_nonzero(claims["supported"][selected_claims])),
        "materialSampleFraction": round(
            float(np.count_nonzero(labels[selected_profiles] != LABEL_AIR))
            / max(sample_count, 1),
            6,
        ),
        "contestedMaterialSampleFraction": round(
            float(
                np.count_nonzero(
                    labels[selected_profiles] == LABEL_CONTESTED_MATERIAL
                )
            )
            / max(np.count_nonzero(labels[selected_profiles] != LABEL_AIR), 1),
            6,
        ),
    }


def _threshold_sensitivity(
    profiles: np.ndarray,
    intensity: np.ndarray,
    claims: np.ndarray,
    depth_offsets: np.ndarray,
    analysis_air_threshold: float,
    resolved: dict[str, Any],
) -> dict[str, Any]:
    primary = np.flatnonzero(profiles["normalFamily"] == 0)
    sample_count = min(
        len(primary), int(resolved["thresholdSensitivityProfileCount"])
    )
    rng = np.random.default_rng(358)
    sample = np.sort(rng.choice(primary, size=sample_count, replace=False))
    thresholds = [float(analysis_air_threshold)]
    thresholds.extend(
        float(value) for value in resolved["thresholdSensitivityRawValues"]
    )
    thresholds = list(dict.fromkeys(thresholds))
    outputs = []
    for threshold in thresholds:
        mask = _smoothed_material_mask(
            np.asarray(intensity[sample]),
            threshold,
            resolved["smoothingKernel"],
            int(resolved["maximumBridgedAirGapSamples"]),
        )
        supported_claim_count = 0
        claim_count = 0
        interval_count = 0
        contested_profile_count = 0
        claims_separated_profile_count = 0
        for local_index, profile_index in enumerate(sample):
            low = int(profiles["claimOffset"][profile_index])
            high = low + int(profiles["claimCount"][profile_index])
            annotated = _annotate_profile(
                mask[local_index],
                depth_offsets,
                claims["depthVoxels"][low:high],
                float(resolved["claimSupportToleranceVoxels"]),
                float(resolved["claimClusterGapVoxels"]),
            )
            assignment = annotated["claimIntervalIndex"]
            supported_claim_count += int(np.count_nonzero(assignment >= 0))
            claim_count += len(assignment)
            interval_count += len(annotated["intervals"])
            contested_profile_count += int(
                any(
                    value["state"] == LABEL_CONTESTED_MATERIAL
                    for value in annotated["intervals"]
                )
            )
            claims_separated_profile_count += int(
                len({int(value) for value in assignment if value >= 0}) >= 2
            )
        outputs.append(
            {
                "rawThreshold": threshold,
                "isAnalysisAirThreshold": threshold == float(analysis_air_threshold),
                "materialSampleFraction": round(float(np.mean(mask)), 6),
                "wholeWindowMaterialProfileFraction": round(
                    float(np.mean(np.all(mask, axis=1))), 6
                ),
                "meanMaterialIntervalsPerProfile": round(
                    interval_count / max(sample_count, 1), 4
                ),
                "supportedClaimFraction": round(
                    supported_claim_count / max(claim_count, 1), 6
                ),
                "contestedProfileFraction": round(
                    contested_profile_count / max(sample_count, 1), 6
                ),
                "claimsSeparatedAcrossRunsProfileFraction": round(
                    claims_separated_profile_count / max(sample_count, 1), 6
                ),
            }
        )
    return {
        "sample": {
            "normalFamily": 0,
            "profileCount": sample_count,
            "selection": "deterministic uniform sample without replacement; RNG seed 358",
        },
        "interpretation": (
            "thresholds above the declared analysis air cutoff are sensitivity "
            "probes, not alternate material definitions"
        ),
        "sweep": outputs,
    }


def build_material_intervals(
    output_root: str | Path,
    force: bool = False,
    settings: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Build 1D CT material intervals and a separate flake-claim overlay."""
    root = Path(output_root)
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    depth_minimum = float(resolved["depthMinimumVoxels"])
    depth_maximum = float(resolved["depthMaximumVoxels"])
    depth_step = float(resolved["depthStepVoxels"])
    if depth_step <= 0.0 or depth_maximum <= depth_minimum:
        raise ValueError("material interval depth range is invalid")
    depth_offsets = np.arange(
        depth_minimum, depth_maximum + 0.5 * depth_step, depth_step, dtype=np.float32
    )
    analysis = json.loads((root / "analysis.json").read_text())
    grid = json.loads((root / "grid.json").read_text())
    plane_count = len(grid["z"])
    input_artifacts = _input_artifacts(root, plane_count)
    identity = {
        "version": MATERIAL_INTERVAL_VERSION,
        "analysisIdentity": analysis["identity"],
        "normalFamilyVersion": NORMAL_FAMILY_VERSION,
        "flakeVersion": FLAKE_CACHE_VERSION,
        "settings": resolved,
        "inputArtifacts": input_artifacts,
    }
    stem = f"material-intervals-v{MATERIAL_INTERVAL_VERSION}"
    summary_path = root / f"{stem}.json"
    artifact_paths = {
        "profiles": root / f"{stem}-profiles.npy",
        "intensity": root / f"{stem}-intensity.npy",
        "material": root / f"{stem}-material.npy",
        "labels": root / f"{stem}-labels.npy",
        "intervals": root / f"{stem}-intervals.npy",
        "claims": root / f"{stem}-claims.npy",
    }
    if summary_path.is_file() and all(path.is_file() for path in artifact_paths.values()) and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    cells = np.load(root / "cells.npy", mmap_mode="r")
    families = np.load(
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.npy", mmap_mode="r"
    )
    profiles, profile_lookup = _profile_catalog(cells, families, grid)
    source = np.load(analysis["identity"]["source"], mmap_mode="r")
    intensity, sampling_stats = _sample_profiles(
        source,
        profiles,
        depth_offsets,
        tuple(int(value) for value in resolved["tileCellsZYX"]),
    )
    material = _smoothed_material_mask(
        intensity,
        float(analysis["normalization"]["airThreshold"]),
        resolved["smoothingKernel"],
        int(resolved["maximumBridgedAirGapSamples"]),
    )
    claims = _load_claims(
        root, plane_count, profile_lookup, len(profiles), depth_offsets
    )
    labels, intervals = _annotate_catalog(
        profiles,
        material,
        depth_offsets,
        claims,
        float(resolved["claimSupportToleranceVoxels"]),
        float(resolved["claimClusterGapVoxels"]),
    )
    for key, array in (
        ("profiles", profiles),
        ("intensity", intensity),
        ("material", material.astype(np.uint8)),
        ("labels", labels),
        ("intervals", intervals),
        ("claims", claims),
    ):
        _atomic_npy(artifact_paths[key], array)
    output_artifacts = {
        key: _content_identity(path) for key, path in artifact_paths.items()
    }
    material_sample_count = int(np.count_nonzero(material))
    state_interval_counts = {
        LABEL_NAMES[state]: int(np.count_nonzero(intervals["state"] == state))
        for state in (
            LABEL_UNASSIGNED_MATERIAL,
            LABEL_SINGLY_CLAIMED_MATERIAL,
            LABEL_CONTESTED_MATERIAL,
        )
    }
    state_sample_counts = {
        LABEL_NAMES[state]: int(np.count_nonzero(labels == state))
        for state in LABEL_NAMES
    }
    apparent = intervals["apparentCtThicknessVoxels"]
    contested_thickness = intervals["observedThicknessVoxels"][
        intervals["state"] == LABEL_CONTESTED_MATERIAL
    ]
    threshold_sensitivity = _threshold_sensitivity(
        profiles,
        intensity,
        claims,
        depth_offsets,
        float(analysis["normalization"]["airThreshold"]),
        resolved,
    )
    result = {
        "identity": identity,
        "contract": {
            "materialMask": (
                "smoothed native CT thresholding with one-sample air-gap closing; "
                "independent of flakes, carriers, and sheet hypotheses"
            ),
            "claimOverlay": (
                "air, unassigned material, singly claimed material, or contested "
                "material; claims are local flake depth clusters and never sheet IDs"
            ),
            "contestedMeaning": (
                "multiple separated local flake-depth clusters occupy one unresolved "
                "CT material run; this is a contact/segmentation ambiguity, not proof "
                "of multiple physical sheets"
            ),
            "thicknessMeaning": (
                "apparent CT run thickness is reported only for singly claimed, "
                "non-boundary-truncated intervals and is not the fitted flake thickness"
            ),
        },
        "depthOffsetsVoxels": depth_offsets.tolist(),
        "labelValues": {name: value for value, name in LABEL_NAMES.items()},
        "artifacts": output_artifacts,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "profileCount": len(profiles),
            "primaryProfileCount": int(np.count_nonzero(profiles["normalFamily"] == 0)),
            "secondaryProfileCount": int(np.count_nonzero(profiles["normalFamily"] == 1)),
            "depthSampleCount": len(depth_offsets),
            "profileSampleCount": int(labels.size),
            "materialSampleCount": material_sample_count,
            "materialSampleFraction": round(material_sample_count / max(labels.size, 1), 6),
            "wholeWindowMaterialProfileCount": int(np.count_nonzero(np.all(material, axis=1))),
            "fullyAirProfileCount": int(np.count_nonzero(~np.any(material, axis=1))),
            "materialIntervalCount": len(intervals),
            "boundaryTruncatedIntervalCount": int(np.count_nonzero(intervals["boundaryTruncated"])),
            "apparentCtThicknessIntervalCount": int(np.count_nonzero(np.isfinite(apparent))),
            "intervalCountsByState": state_interval_counts,
            "sampleCountsByState": state_sample_counts,
            "claimCount": len(claims),
            "supportedClaimCount": int(np.count_nonzero(claims["supported"])),
            "unsupportedClaimCount": int(np.count_nonzero(~claims["supported"].astype(bool))),
            "supportedClaimFraction": round(float(np.mean(claims["supported"])), 6),
            "profilesWithNoClaimCount": int(np.count_nonzero(profiles["claimCount"] == 0)),
            "profilesWithContestedMaterialCount": int(np.count_nonzero(profiles["contestedIntervalCount"] > 0)),
            "apparentCtThicknessVoxels": _quantiles(apparent),
            "contestedObservedThicknessVoxels": _quantiles(contested_thickness),
            "family": {
                "primary": _family_stats(0, profiles, claims, labels),
                "secondary": _family_stats(1, profiles, claims, labels),
            },
            "sampling": sampling_stats,
            "thresholdSensitivity": threshold_sensitivity,
        },
    }
    _atomic_json(summary_path, result)
    return result
