from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from backend.acus import _refine_needles_batch
from backend.acus_compute import compute_status, extract_needles_gpu, hessian_line_fields

from .contracts import (
    ExtractionTileSpec,
    RawAcusSettings,
    ShardSpec,
    VolumeSource,
    VoxelBounds,
    atomic_json,
    sha256_file,
)


NEEDLE_ARTIFACT_SCHEMA = "pareidolia.raw-acus-needles"
NEEDLE_ARTIFACT_VERSION = 1
CALIBRATION_SCHEMA = "pareidolia.raw-acus-calibration"
CALIBRATION_VERSION = 1


def normalize_ct(raw: np.ndarray, low: float, high: float) -> np.ndarray:
    if high <= low:
        raise ValueError("normalization high value must exceed low value")
    return np.clip(
        (np.asarray(raw, dtype=np.float32) - float(low)) / float(high - low),
        0.0,
        1.0,
    )


@dataclass(frozen=True, slots=True)
class AcusCalibration:
    low: float
    high: float
    air_threshold_raw: float
    strength_scale: float
    compute_backend: str
    compute_device: str | None
    sample_bounds: tuple[VoxelBounds, ...]

    def __post_init__(self) -> None:
        values = (self.low, self.high, self.air_threshold_raw, self.strength_scale)
        if any(not math.isfinite(value) for value in values):
            raise ValueError("calibration values must be finite")
        if self.high <= self.low or self.strength_scale <= 0.0:
            raise ValueError("calibration has a degenerate scale")

    def record(self) -> dict[str, Any]:
        return {
            "low": self.low,
            "high": self.high,
            "airThresholdRaw": self.air_threshold_raw,
            "strengthScale": self.strength_scale,
            "computeBackend": self.compute_backend,
            "computeDevice": self.compute_device,
            "sampleBounds": [value.record() for value in self.sample_bounds],
        }


@dataclass(slots=True)
class NeedleTable:
    center_xyz: np.ndarray
    direction_xyz: np.ndarray
    score: np.ndarray
    axial_coverage: np.ndarray
    support_score: np.ndarray

    @classmethod
    def empty(cls) -> "NeedleTable":
        return cls(
            np.empty((0, 3), dtype=np.float32),
            np.empty((0, 3), dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
            np.empty(0, dtype=np.float32),
        )

    @property
    def count(self) -> int:
        return int(len(self.score))

    def validate(self) -> None:
        count = self.count
        expected = {
            "center_xyz": (count, 3),
            "direction_xyz": (count, 3),
            "score": (count,),
            "axial_coverage": (count,),
            "support_score": (count,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if count:
            lengths = np.linalg.norm(self.direction_xyz, axis=1)
            if np.any(np.abs(lengths - 1.0) > 2.0e-4):
                raise ValueError("needle directions must be unit axes")
            if np.any((self.score < 0.0) | (self.score > 1.0)):
                raise ValueError("needle scores must lie in [0, 1]")
            if np.any((self.axial_coverage < 0.0) | (self.axial_coverage > 1.0)):
                raise ValueError("needle axial coverage must lie in [0, 1]")
            if np.any((self.support_score < 0.0) | (self.support_score > 1.0)):
                raise ValueError("needle support scores must lie in [0, 1]")

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "centerXYZ": np.asarray(self.center_xyz, dtype=np.float32),
            "directionXYZ": np.asarray(self.direction_xyz, dtype=np.float32),
            "score": np.asarray(self.score, dtype=np.float32),
            "axialCoverage": np.asarray(self.axial_coverage, dtype=np.float32),
            "supportScore": np.asarray(self.support_score, dtype=np.float32),
        }


def _sample_bounds(
    bounds: VoxelBounds, count: int, cube_size: int
) -> tuple[VoxelBounds, ...]:
    available = np.asarray(bounds.shape_xyz, dtype=np.int64)
    side = int(min(cube_size, *available.tolist()))
    if side < 24:
        raise ValueError("calibration region is too small for an Acus Hessian sample")
    slack = np.maximum(available - side, 0)
    # Three incommensurate progressions spread samples over the volume without
    # depending on an old analysis grid or random process state.
    fractions = (0.61803398875, 0.41421356237, 0.73205080757)
    result: list[VoxelBounds] = []
    seen: set[tuple[int, int, int]] = set()
    for index in range(count):
        start = tuple(
            int(
                bounds.start_xyz[axis]
                + round(
                    float(slack[axis])
                    * ((0.5 + (index + 1) * fractions[axis]) % 1.0)
                )
            )
            for axis in range(3)
        )
        if start not in seen:
            result.append(
                VoxelBounds(start, tuple(value + side for value in start))
            )
            seen.add(start)
    return tuple(result)


def calibrate_raw_acus(
    source: VolumeSource,
    processing_bounds: VoxelBounds,
    settings: RawAcusSettings,
) -> AcusCalibration:
    """Derive intensity and Hessian scales only from the native CT window."""

    volume = source.memmap()
    shape = np.asarray(processing_bounds.shape_xyz, dtype=np.int64)
    target_samples = 2_000_000
    sampling_stride = max(1, int(round((np.prod(shape) / target_samples) ** (1.0 / 3.0))))
    raw_sample = np.asarray(
        volume[
            processing_bounds.slices_zyx[0].start : processing_bounds.slices_zyx[0].stop : sampling_stride,
            processing_bounds.slices_zyx[1].start : processing_bounds.slices_zyx[1].stop : sampling_stride,
            processing_bounds.slices_zyx[2].start : processing_bounds.slices_zyx[2].stop : sampling_stride,
        ]
    )
    positive = raw_sample[raw_sample > 0]
    if not len(positive):
        raise ValueError("raw calibration window contains no nonzero CT samples")
    low = 0.0
    high = float(np.percentile(positive, 99.5))
    if high <= low:
        raise ValueError("raw calibration intensity range is degenerate")
    air_threshold = max(4.0, high * 0.08)

    candidates = _sample_bounds(
        processing_bounds,
        max(settings.calibration_samples * 3, settings.calibration_samples),
        settings.calibration_cube_voxels,
    )
    material_bounds: list[VoxelBounds] = []
    normalized: list[np.ndarray] = []
    for bounds in candidates:
        raw = volume[bounds.slices_zyx]
        if int(np.max(raw, initial=0)) <= air_threshold:
            continue
        material_bounds.append(bounds)
        normalized.append(normalize_ct(raw, low, high))
        if len(normalized) >= settings.calibration_samples:
            break
    if not normalized:
        raise ValueError("no material-bearing raw Acus calibration cubes were found")
    _, metadata = hessian_line_fields(
        normalized, settings.hessian_scale_voxels, strength_scale=None
    )
    scales = np.asarray(metadata["strengthScales"], dtype=np.float64)
    scales = scales[np.isfinite(scales) & (scales > 0.0)]
    if not len(scales):
        raise ValueError("Hessian calibration did not produce a positive strength scale")
    return AcusCalibration(
        low,
        high,
        air_threshold,
        float(np.median(scales)),
        str(metadata["backend"]),
        metadata.get("device"),
        tuple(material_bounds),
    )


def write_calibration(
    path: str | Path,
    calibration: AcusCalibration,
    *,
    identity_sha256: str,
) -> dict[str, Any]:
    payload = {
        "schema": CALIBRATION_SCHEMA,
        "version": CALIBRATION_VERSION,
        "identitySha256": identity_sha256,
        "calibration": calibration.record(),
    }
    atomic_json(path, payload)
    return payload


def read_calibration(
    path: str | Path, *, identity_sha256: str
) -> AcusCalibration:
    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema") != CALIBRATION_SCHEMA
        or int(payload.get("version", -1)) != CALIBRATION_VERSION
        or payload.get("identitySha256") != identity_sha256
    ):
        raise ValueError("raw Acus calibration does not match this pipeline identity")
    return _calibration_from_record(payload["calibration"])


def _calibration_from_record(record: Mapping[str, Any]) -> AcusCalibration:
    return AcusCalibration(
        float(record["low"]),
        float(record["high"]),
        float(record["airThresholdRaw"]),
        float(record["strengthScale"]),
        str(record["computeBackend"]),
        record.get("computeDevice"),
        tuple(
            VoxelBounds(tuple(value["startXYZ"]), tuple(value["stopXYZExclusive"]))
            for value in record["sampleBounds"]
        ),
    )


def read_calibration_reference(path: str | Path) -> AcusCalibration:
    """Load a source-level calibration without adopting its pipeline identity."""

    payload = json.loads(Path(path).read_text())
    if (
        payload.get("schema") != CALIBRATION_SCHEMA
        or int(payload.get("version", -1)) != CALIBRATION_VERSION
        or not isinstance(payload.get("calibration"), Mapping)
    ):
        raise ValueError("raw Acus calibration reference is invalid")
    return _calibration_from_record(payload["calibration"])


def _cpu_block_candidates(
    score: np.ndarray,
    core_local_zyx: tuple[int, int, int, int, int, int],
    spacing: int,
    bin_size: int,
    maximum_per_bin: int,
    threshold: float,
) -> np.ndarray:
    z0, z1, y0, y1, x0, x1 = core_local_zyx
    core = score[z0:z1, y0:y1, x0:x1]
    original_shape = core.shape
    block_shape = tuple(int(math.ceil(size / spacing)) for size in original_shape)
    padded_shape = tuple(count * spacing for count in block_shape)
    if padded_shape != original_shape:
        padded = np.full(padded_shape, -np.inf, dtype=np.float32)
        padded[: original_shape[0], : original_shape[1], : original_shape[2]] = core
    else:
        padded = core
    blocks = (
        padded.reshape(
            block_shape[0], spacing,
            block_shape[1], spacing,
            block_shape[2], spacing,
        )
        .transpose(0, 2, 4, 1, 3, 5)
        .reshape(*block_shape, spacing**3)
    )
    ranking_blocks = np.rint(blocks * 10_000.0)
    flat = np.argmax(ranking_blocks, axis=-1)
    values = np.take_along_axis(blocks, flat[..., None], axis=-1)[..., 0]
    dz, dy, dx = np.unravel_index(flat, (spacing, spacing, spacing))
    bz, by, bx = np.indices(block_shape, dtype=np.int32)
    local_z = z0 + bz * spacing + dz
    local_y = y0 + by * spacing + dy
    local_x = x0 + bx * spacing + dx
    valid = (
        (values >= threshold)
        & (local_z < z1)
        & (local_y < y1)
        & (local_x < x1)
    )
    if not np.any(valid):
        return np.empty((0, 4), dtype=np.float32)
    values = values[valid].astype(np.float32)
    points = np.stack([local_z[valid], local_y[valid], local_x[valid]], axis=1)
    block_starts = np.stack(
        [bz[valid] * spacing, by[valid] * spacing, bx[valid] * spacing], axis=1
    )
    bin_zyx = block_starts // bin_size
    bin_shape = tuple(int(math.ceil(size / bin_size)) for size in original_shape)
    bin_ids = (
        (bin_zyx[:, 0] * bin_shape[1] + bin_zyx[:, 1]) * bin_shape[2]
        + bin_zyx[:, 2]
    )
    ordering_response = np.rint(values * 10_000.0)
    order = np.lexsort(
        (
            points[:, 2],
            points[:, 1],
            points[:, 0],
            -ordering_response,
            bin_ids,
        )
    )
    ordered_bins = bin_ids[order]
    starts = np.flatnonzero(np.r_[True, ordered_bins[1:] != ordered_bins[:-1]])
    selected: list[np.ndarray] = []
    for group, start in enumerate(starts):
        stop = starts[group + 1] if group + 1 < len(starts) else len(order)
        selected.append(order[start : min(stop, start + maximum_per_bin)])
    chosen = np.concatenate(selected) if selected else np.empty(0, dtype=np.int64)
    return np.column_stack([values[chosen], points[chosen]]).astype(np.float32)


def _needle_priority_key(table: NeedleTable, index: int) -> tuple[float, ...]:
    center = table.center_xyz[index]
    direction = table.direction_xyz[index]
    return (
        -float(table.score[index]),
        float(center[2]),
        float(center[1]),
        float(center[0]),
        float(direction[2]),
        float(direction[1]),
        float(direction[0]),
    )


def deduplicate_needle_indices(
    table: NeedleTable,
    indices: Iterable[int],
    minimum_distance: float,
) -> np.ndarray:
    values = [int(value) for value in indices]
    if len(values) < 2:
        return np.asarray(values, dtype=np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = {}
    accepted: list[int] = []
    distance2 = minimum_distance**2
    for index in sorted(values, key=lambda value: _needle_priority_key(table, value)):
        center = table.center_xyz[index]
        bucket = tuple(int(math.floor(float(value) / minimum_distance)) for value in center)
        duplicate = False
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    neighbor = (bucket[0] + dx, bucket[1] + dy, bucket[2] + dz)
                    if any(
                        float(np.sum((center - table.center_xyz[other]) ** 2)) < distance2
                        for other in buckets.get(neighbor, ())
                    ):
                        duplicate = True
                        break
                if duplicate:
                    break
            if duplicate:
                break
        if duplicate:
            continue
        accepted.append(int(index))
        buckets.setdefault(bucket, []).append(int(index))
    return np.asarray(accepted, dtype=np.int64)


def select_cell_needle_indices(
    table: NeedleTable,
    center_xyz: np.ndarray,
    half_extent_voxels: float,
    settings: RawAcusSettings,
) -> np.ndarray:
    relative = table.center_xyz - np.asarray(center_xyz, dtype=np.float32)[None, :]
    inside = np.all(
        (relative >= -half_extent_voxels) & (relative < half_extent_voxels),
        axis=1,
    )
    indices = deduplicate_needle_indices(
        table,
        np.flatnonzero(inside),
        float(max(2, settings.candidate_spacing_voxels - 1)),
    )
    if len(indices) > settings.maximum_needles_per_cell:
        indices = np.asarray(
            sorted(
                indices.tolist(),
                key=lambda value: _needle_priority_key(table, value),
            )[: settings.maximum_needles_per_cell],
            dtype=np.int64,
        )
    return indices


def _sort_needle_table(table: NeedleTable) -> NeedleTable:
    chosen = np.asarray(
        sorted(range(table.count), key=lambda value: _needle_priority_key(table, value)),
        dtype=np.int64,
    )
    return NeedleTable(
        table.center_xyz[chosen],
        table.direction_xyz[chosen],
        table.score[chosen],
        table.axial_coverage[chosen],
        table.support_score[chosen],
    )


def _canonicalize_needle_precision(table: NeedleTable) -> NeedleTable:
    """Remove backend/array-shape roundoff before spatial ownership decisions."""

    if not table.count:
        return table
    centers = (
        np.rint(np.asarray(table.center_xyz, dtype=np.float64) * 256.0) / 256.0
    ).astype(np.float32)
    directions = (
        np.rint(np.asarray(table.direction_xyz, dtype=np.float64) * 100_000.0)
        / 100_000.0
    )
    directions /= np.maximum(np.linalg.norm(directions, axis=1, keepdims=True), 1.0e-12)
    dominant = np.argmax(np.abs(directions), axis=1)
    signs = np.where(directions[np.arange(len(directions)), dominant] < 0.0, -1.0, 1.0)
    directions *= signs[:, None]

    def scalar(values: np.ndarray) -> np.ndarray:
        return (
            np.rint(np.asarray(values, dtype=np.float64) * 10_000.0) / 10_000.0
        ).astype(np.float32)

    result = NeedleTable(
        centers,
        directions.astype(np.float32),
        scalar(table.score),
        scalar(table.axial_coverage),
        scalar(table.support_score),
    )
    result.validate()
    return result


def extract_needle_region(
    source: VolumeSource,
    core_bounds: VoxelBounds,
    raw_bounds: VoxelBounds,
    calibration: AcusCalibration,
    settings: RawAcusSettings,
) -> tuple[NeedleTable, dict[str, Any]]:
    """Run Hessian ridge extraction for one explicitly owned voxel core."""

    started = time.monotonic()
    raw = source.memmap()[raw_bounds.slices_zyx]
    data = normalize_ct(raw, calibration.low, calibration.high)
    raw_start = np.asarray(raw_bounds.start_xyz, dtype=np.float32)
    core_start_xyz = np.asarray(core_bounds.start_xyz) - np.asarray(
        raw_bounds.start_xyz
    )
    core_stop_xyz = np.asarray(core_bounds.stop_xyz_exclusive) - np.asarray(
        raw_bounds.start_xyz
    )
    core_local_zyx = (
        int(core_start_xyz[2]), int(core_stop_xyz[2]),
        int(core_start_xyz[1]), int(core_stop_xyz[1]),
        int(core_start_xyz[0]), int(core_stop_xyz[0]),
    )
    backend = compute_status()
    metadata: dict[str, Any]
    if backend["backend"] == "gpu":
        core_shape_zyx = tuple(
            core_local_zyx[index + 1] - core_local_zyx[index]
            for index in (0, 2, 4)
        )
        bin_shape = tuple(
            int(math.ceil(value / settings.candidate_bin_voxels))
            for value in core_shape_zyx
        )
        values, metadata = extract_needles_gpu(
            data,
            sigma=settings.hessian_scale_voxels,
            strength_scale=calibration.strength_scale,
            core_local_zyx=core_local_zyx,
            core_global_start_zyx=(0, 0, 0),
            spacing=settings.candidate_spacing_voxels,
            halo=0,
            bin_size=settings.candidate_bin_voxels,
            bin_shape_zyx=bin_shape,
            maximum_per_bin=settings.maximum_needles_per_bin,
            radius=settings.refinement_radius_voxels,
            needle_length=settings.needle_length_voxels,
            cross_section_radius=settings.cross_section_radius_voxels,
            threshold=settings.candidate_threshold,
        )
        table = NeedleTable(
            np.asarray(values["center"], dtype=np.float32) + raw_start[None, :],
            np.asarray(values["direction"], dtype=np.float32),
            np.asarray(values["score"], dtype=np.float32),
            np.asarray(values["axialCoverage"], dtype=np.float32),
            np.asarray(values["supportScore"], dtype=np.float32),
        )
    else:
        fields, field_metadata = hessian_line_fields(
            [data], settings.hessian_scale_voxels, calibration.strength_scale
        )
        score, direction = fields[0]
        candidates = _cpu_block_candidates(
            score,
            core_local_zyx,
            settings.candidate_spacing_voxels,
            settings.candidate_bin_voxels,
            settings.maximum_needles_per_bin,
            settings.candidate_threshold,
        )
        refined = _refine_needles_batch(
            score,
            direction,
            candidates,
            settings.refinement_radius_voxels,
            settings.needle_length_voxels,
            settings.cross_section_radius_voxels,
        )
        table = NeedleTable(
            np.asarray([value["center"] for value in refined], dtype=np.float32).reshape(-1, 3)
            + raw_start[None, :],
            np.asarray([value["direction"] for value in refined], dtype=np.float32).reshape(-1, 3),
            np.asarray([value["score"] for value in refined], dtype=np.float32),
            np.asarray([value["axialCoverage"] for value in refined], dtype=np.float32),
            np.asarray([value["supportScore"] for value in refined], dtype=np.float32),
        )
        metadata = {
            "backend": field_metadata["backend"],
            "device": field_metadata.get("device"),
            "fallbackReason": field_metadata.get("fallbackReason"),
            "candidateCount": int(len(candidates)),
            "refinedCount": table.count,
            "timingsMs": {"lineField": field_metadata["elapsedMs"]},
        }
    table = _canonicalize_needle_precision(table)
    evidence_start = np.asarray(core_bounds.start_xyz, dtype=np.float32)
    evidence_stop = np.asarray(core_bounds.stop_xyz_exclusive, dtype=np.float32)
    owned = np.all(
        (table.center_xyz >= evidence_start[None, :])
        & (table.center_xyz < evidence_stop[None, :]),
        axis=1,
    )
    table = NeedleTable(
        table.center_xyz[owned],
        table.direction_xyz[owned],
        table.score[owned],
        table.axial_coverage[owned],
        table.support_score[owned],
    )
    table = _sort_needle_table(table)
    table.validate()
    metadata = dict(metadata)
    metadata["retainedCandidateCount"] = table.count
    metadata["spatialOwnership"] = "canonical source-anchored extraction tile core"
    metadata["deduplication"] = "deferred to each half-open cell analysis context"
    metadata["canonicalPrecision"] = {
        "centerVoxel": 1.0 / 256.0,
        "direction": 1.0 / 100_000.0,
        "scoreAndSupport": 1.0 / 10_000.0,
    }
    metadata["elapsedSeconds"] = round(time.monotonic() - started, 6)
    return table, metadata


def extract_shard_needles(
    source: VolumeSource,
    shard: ShardSpec,
    calibration: AcusCalibration,
    settings: RawAcusSettings,
) -> tuple[NeedleTable, dict[str, Any]]:
    """Compatibility wrapper for one evidence shard-sized extraction."""

    return extract_needle_region(
        source,
        shard.evidence_voxel_bounds,
        shard.raw_voxel_bounds,
        calibration,
        settings,
    )


def extract_tile_needles(
    source: VolumeSource,
    tile: ExtractionTileSpec,
    calibration: AcusCalibration,
    settings: RawAcusSettings,
) -> tuple[NeedleTable, dict[str, Any]]:
    return extract_needle_region(
        source,
        tile.core_voxel_bounds,
        tile.raw_voxel_bounds,
        calibration,
        settings,
    )


def gather_needle_tables(
    tables: Iterable[NeedleTable], bounds: VoxelBounds
) -> NeedleTable:
    values = list(tables)
    if not values:
        return NeedleTable.empty()
    combined = NeedleTable(
        np.concatenate([value.center_xyz for value in values], axis=0),
        np.concatenate([value.direction_xyz for value in values], axis=0),
        np.concatenate([value.score for value in values]),
        np.concatenate([value.axial_coverage for value in values]),
        np.concatenate([value.support_score for value in values]),
    )
    start = np.asarray(bounds.start_xyz, dtype=np.float32)
    stop = np.asarray(bounds.stop_xyz_exclusive, dtype=np.float32)
    owned = np.all(
        (combined.center_xyz >= start[None, :])
        & (combined.center_xyz < stop[None, :]),
        axis=1,
    )
    result = _sort_needle_table(
        NeedleTable(
            combined.center_xyz[owned],
            combined.direction_xyz[owned],
            combined.score[owned],
            combined.axial_coverage[owned],
            combined.support_score[owned],
        )
    )
    result.validate()
    return result


def write_needle_artifact(
    prefix: str | Path,
    table: NeedleTable,
    *,
    identity_sha256: str,
    shard: ShardSpec | ExtractionTileSpec,
    compute_metadata: Mapping[str, Any],
) -> dict[str, Any]:
    base = Path(prefix)
    data_path = base.with_suffix(".npz")
    manifest_path = base.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, **table.arrays())
    temporary.replace(data_path)
    payload = {
        "schema": NEEDLE_ARTIFACT_SCHEMA,
        "version": NEEDLE_ARTIFACT_VERSION,
        "identitySha256": identity_sha256,
        "region": shard.record(),
        "counts": {"needles": table.count},
        "compute": dict(compute_metadata),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(manifest_path, payload)
    return payload


def read_needle_artifact(
    prefix: str | Path,
    *,
    identity_sha256: str,
    verify: bool = True,
) -> NeedleTable:
    base = Path(prefix)
    manifest = json.loads(base.with_suffix(".json").read_text())
    data_path = base.with_suffix(".npz")
    if (
        manifest.get("schema") != NEEDLE_ARTIFACT_SCHEMA
        or int(manifest.get("version", -1)) != NEEDLE_ARTIFACT_VERSION
        or manifest.get("identitySha256") != identity_sha256
    ):
        raise ValueError("needle artifact does not match this raw Acus pipeline")
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("needle artifact content hash mismatch")
    with np.load(data_path) as values:
        table = NeedleTable(
            np.asarray(values["centerXYZ"], dtype=np.float32),
            np.asarray(values["directionXYZ"], dtype=np.float32),
            np.asarray(values["score"], dtype=np.float32),
            np.asarray(values["axialCoverage"], dtype=np.float32),
            np.asarray(values["supportScore"], dtype=np.float32),
        )
    table.validate()
    return table
