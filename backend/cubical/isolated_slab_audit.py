from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .isolated_slab import ISOLATED_SLAB_SCHEMA, _percentile_record
from .needle_topology import _load_field_artifact, _raw_settings_from_field


ISOLATED_SLAB_ACUS_AUDIT_SCHEMA = "pareidolia.isolated-slab-acus-audit"
ISOLATED_SLAB_ACUS_AUDIT_VERSION = 1
ISOLATED_SLAB_ACUS_AUDIT_STEM = "isolated-slab-acus-audit-v1"


@dataclass(frozen=True, slots=True)
class IsolatedSlabAcusAuditSettings:
    minimum_seed_confidence: float | None = None
    center_coverage_radii_voxels: tuple[float, ...] = (
        2.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
    )
    segment_coverage_radii_voxels: tuple[float, ...] = (
        1.0,
        2.0,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
    )
    nominal_segment_coverage_radius_voxels: float = 4.0
    nominal_center_coverage_radius_voxels: float = 8.0
    minimum_component_seed_count: int = 32
    maximum_reported_components: int = 128

    def __post_init__(self) -> None:
        if self.minimum_seed_confidence is not None and not (
            0.0 <= self.minimum_seed_confidence <= 1.0
        ):
            raise ValueError("minimum seed confidence must lie in [0, 1]")
        radii = (
            *self.center_coverage_radii_voxels,
            *self.segment_coverage_radii_voxels,
            self.nominal_segment_coverage_radius_voxels,
            self.nominal_center_coverage_radius_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in radii):
            raise ValueError("Acus audit radii must be finite and positive")
        if tuple(sorted(set(self.center_coverage_radii_voxels))) != tuple(
            self.center_coverage_radii_voxels
        ):
            raise ValueError("center coverage radii must be unique and increasing")
        if tuple(sorted(set(self.segment_coverage_radii_voxels))) != tuple(
            self.segment_coverage_radii_voxels
        ):
            raise ValueError("segment coverage radii must be unique and increasing")
        if self.minimum_component_seed_count < 1 or self.maximum_reported_components < 1:
            raise ValueError("component audit counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def nearest_needle_and_segment(
    point_xyz: np.ndarray,
    needle_center_xyz: np.ndarray,
    needle_direction_xyz: np.ndarray,
    *,
    needle_half_length_voxels: float,
    maximum_search_radius_voxels: float,
) -> dict[str, np.ndarray]:
    """Find nearest finite Acus segment and center with a spatial hash."""

    point = np.asarray(point_xyz, dtype=np.float64)
    center = np.asarray(needle_center_xyz, dtype=np.float64)
    direction = np.asarray(needle_direction_xyz, dtype=np.float64)
    if point.ndim != 2 or point.shape[1] != 3:
        raise ValueError("audit points must have shape (N, 3)")
    if center.shape != direction.shape or center.ndim != 2 or center.shape[1] != 3:
        raise ValueError("needle centers and directions must have shape (M, 3)")
    bucket_size = maximum_search_radius_voxels + needle_half_length_voxels
    bucket_xyz = np.floor(center / bucket_size).astype(np.int64)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(bucket_xyz):
        buckets[tuple(int(item) for item in value)].append(index)
    center_distance = np.full(len(point), np.inf, dtype=np.float32)
    center_index = np.full(len(point), -1, dtype=np.int32)
    segment_distance = np.full(len(point), np.inf, dtype=np.float32)
    segment_index = np.full(len(point), -1, dtype=np.int32)
    segment_axial_offset = np.full(len(point), np.nan, dtype=np.float32)
    maximum_candidate_center_distance = (
        maximum_search_radius_voxels + needle_half_length_voxels
    )
    for point_index, value in enumerate(point):
        base = np.floor(value / bucket_size).astype(np.int64)
        candidates: list[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidates.extend(
                        buckets.get(
                            (
                                int(base[0]) + dx,
                                int(base[1]) + dy,
                                int(base[2]) + dz,
                            ),
                            (),
                        )
                    )
        if not candidates:
            continue
        candidate = np.asarray(candidates, dtype=np.int32)
        delta = value[None, :] - center[candidate]
        candidate_center_distance = np.linalg.norm(delta, axis=1)
        nearby = candidate_center_distance <= maximum_candidate_center_distance
        candidate = candidate[nearby]
        delta = delta[nearby]
        candidate_center_distance = candidate_center_distance[nearby]
        if not len(candidate):
            continue
        closest_center = int(np.argmin(candidate_center_distance))
        center_distance[point_index] = candidate_center_distance[closest_center]
        center_index[point_index] = candidate[closest_center]
        axial = np.clip(
            np.einsum("ij,ij->i", delta, direction[candidate]),
            -needle_half_length_voxels,
            needle_half_length_voxels,
        )
        residual = delta - axial[:, None] * direction[candidate]
        candidate_segment_distance = np.linalg.norm(residual, axis=1)
        closest_segment = int(np.argmin(candidate_segment_distance))
        if candidate_segment_distance[closest_segment] <= maximum_search_radius_voxels:
            segment_distance[point_index] = candidate_segment_distance[closest_segment]
            segment_index[point_index] = candidate[closest_segment]
            segment_axial_offset[point_index] = axial[closest_segment]
    return {
        "nearestCenterNeedleIndex": center_index,
        "nearestCenterDistanceVoxels": center_distance,
        "nearestSegmentNeedleIndex": segment_index,
        "nearestSegmentDistanceVoxels": segment_distance,
        "nearestSegmentAxialOffsetVoxels": segment_axial_offset,
    }


def _angle_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    dot = np.abs(np.einsum("ij,ij->i", first, second))
    return np.degrees(np.arccos(np.clip(dot, 0.0, 1.0))).astype(np.float32)


def _fiber_plane_error_degrees(normal: np.ndarray, fiber: np.ndarray) -> np.ndarray:
    dot = np.abs(np.einsum("ij,ij->i", normal, fiber))
    return np.degrees(np.arcsin(np.clip(dot, 0.0, 1.0))).astype(np.float32)


def _coverage_record(distance: np.ndarray, radii: tuple[float, ...]) -> dict[str, float]:
    return {
        f"{radius:g}": round(float(np.mean(distance <= radius)), 6)
        for radius in radii
    }


def _finite_percentile_record(values: np.ndarray) -> dict[str, float | int | None]:
    return _percentile_record(np.asarray(values)[np.isfinite(values)])


def _component_records(
    component_id: np.ndarray,
    center_distance: np.ndarray,
    segment_distance: np.ndarray,
    normal_angle: np.ndarray,
    fiber_plane_error: np.ndarray,
    settings: IsolatedSlabAcusAuditSettings,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    def finite_median(values: np.ndarray) -> float | None:
        finite = np.asarray(values, dtype=np.float64)
        finite = finite[np.isfinite(finite)]
        return round(float(np.median(finite)), 6) if len(finite) else None

    records: list[dict[str, Any]] = []
    for component in sorted(int(value) for value in np.unique(component_id) if value >= 0):
        selected = component_id == component
        count = int(np.count_nonzero(selected))
        if count < settings.minimum_component_seed_count:
            continue
        records.append(
            {
                "componentId": component,
                "seedCount": count,
                "nominalCenterCoverage": round(
                    float(
                        np.mean(
                            center_distance[selected]
                            <= settings.nominal_center_coverage_radius_voxels
                        )
                    ),
                    6,
                ),
                "nominalSegmentCoverage": round(
                    float(
                        np.mean(
                            segment_distance[selected]
                            <= settings.nominal_segment_coverage_radius_voxels
                        )
                    ),
                    6,
                ),
                "medianCenterDistanceVoxels": finite_median(
                    center_distance[selected]
                ),
                "medianSegmentDistanceVoxels": finite_median(
                    segment_distance[selected]
                ),
                "medianNeedleNormalAngleDegrees": finite_median(
                    normal_angle[selected]
                ),
                "medianFiberPlaneErrorDegrees": finite_median(
                    fiber_plane_error[selected]
                ),
            }
        )
    records.sort(key=lambda value: (-value["seedCount"], value["componentId"]))
    segment_coverage = np.asarray(
        [value["nominalSegmentCoverage"] for value in records], dtype=np.float64
    )
    center_coverage = np.asarray(
        [value["nominalCenterCoverage"] for value in records], dtype=np.float64
    )
    summary = {
        "componentCount": len(records),
        "nominalCenterCoverage": _percentile_record(center_coverage),
        "nominalSegmentCoverage": _percentile_record(segment_coverage),
        "zeroNominalSegmentCoverageComponents": int(
            np.count_nonzero(segment_coverage == 0.0)
        ),
    }
    return records[: settings.maximum_reported_components], summary


def write_acus_coverage_projection(
    midpoint_xyz: np.ndarray,
    segment_distance: np.ndarray,
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    settings: IsolatedSlabAcusAuditSettings,
    path: str | Path,
    *,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    covered = segment_distance <= settings.nominal_segment_coverage_radius_voxels
    projections = ((0, 1), (0, 2), (1, 2))
    for panel, axes in enumerate(projections):
        offset = panel * panel_size
        normalized = (
            midpoint_xyz[:, list(axes)] - world_start_xyz[None, list(axes)]
        ) / width[None, list(axes)]
        x = np.rint(
            offset + margin + normalized[:, 0] * (panel_size - 2 * margin)
        ).astype(np.int32)
        y = np.rint(
            panel_size - margin - normalized[:, 1] * (panel_size - 2 * margin)
        ).astype(np.int32)
        valid = (
            (x >= offset)
            & (x < offset + panel_size)
            & (y >= 0)
            & (y < panel_size)
        )
        # Draw uncovered evidence first and covered evidence last.  The latter
        # is cyan; the former is coral, making coherent missed ribbons visible.
        for mask, color in (
            (valid & ~covered, (246, 91, 88)),
            (valid & covered, (42, 238, 202)),
        ):
            canvas[y[mask], x[mask]] = color
        canvas[margin, offset + margin : offset + panel_size - margin] = (64, 72, 84)
        canvas[panel_size - margin, offset + margin : offset + panel_size - margin] = (
            64,
            72,
            84,
        )
        canvas[margin : panel_size - margin, offset + margin] = (64, 72, 84)
        canvas[
            margin : panel_size - margin, offset + panel_size - margin
        ] = (64, 72, 84)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def run_isolated_slab_acus_audit(
    slab_root: str | Path,
    field_root: str | Path,
    output_root: str | Path,
    *,
    settings: IsolatedSlabAcusAuditSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or IsolatedSlabAcusAuditSettings()
    slab_root_path = Path(slab_root).resolve()
    slab_manifest_path = (
        slab_root_path
        if slab_root_path.is_file()
        else slab_root_path / "isolated-slabs-v1.json"
    )
    slab_manifest = json.loads(slab_manifest_path.read_text())
    if (
        slab_manifest.get("schema") != ISOLATED_SLAB_SCHEMA
        or slab_manifest.get("state") != "complete"
    ):
        raise ValueError("Acus audit requires a complete isolated-slab artifact")
    slab_data_path = slab_manifest_path.parent / str(slab_manifest["data"]["path"])
    if sha256_file(slab_data_path) != slab_manifest["data"]["sha256"]:
        raise ValueError("isolated-slab data hash differs from its manifest")
    with np.load(slab_data_path) as values:
        slab = {name: np.asarray(values[name]) for name in values.files}
    field_manifest_path, field_manifest, field = _load_field_artifact(field_root)
    raw_settings, raw_provenance = _raw_settings_from_field(field_manifest)
    needle_half_length = 0.5 * raw_settings.needle_length_voxels
    slab_threshold = float(
        slab_manifest["identity"]["settings"]["minimum_seed_confidence"]
    )
    seed_threshold = (
        slab_threshold
        if resolved.minimum_seed_confidence is None
        else resolved.minimum_seed_confidence
    )
    if seed_threshold < slab_threshold:
        raise ValueError(
            "audit seed confidence cannot be lower than the slab component "
            "threshold without recomputing its descriptive component graph"
        )
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    identity: dict[str, Any] = {
        "schema": ISOLATED_SLAB_ACUS_AUDIT_SCHEMA,
        "version": ISOLATED_SLAB_ACUS_AUDIT_VERSION,
        "slabs": {
            "manifestPath": str(slab_manifest_path),
            "manifestSha256": sha256_file(slab_manifest_path),
            "dataSha256": slab_manifest["data"]["sha256"],
        },
        "needleField": {
            "manifestPath": str(field_manifest_path),
            "manifestSha256": sha256_file(field_manifest_path),
            "dataSha256": field_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "resolvedSeedConfidence": seed_threshold,
        "needleLengthVoxels": raw_settings.needle_length_voxels,
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    manifest_path = output / f"{ISOLATED_SLAB_ACUS_AUDIT_STEM}.json"
    data_path = output / f"{ISOLATED_SLAB_ACUS_AUDIT_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    seed_index = np.flatnonzero(slab["confidence"] >= seed_threshold)
    midpoint = slab["midpointXYZ"][seed_index].astype(np.float64)
    slab_normal = slab["normalXYZ"][seed_index].astype(np.float64)
    component_id = slab["componentId"][seed_index].astype(np.int32)
    maximum_radius = max(resolved.segment_coverage_radii_voxels)
    nearest = nearest_needle_and_segment(
        midpoint,
        field["centerXYZ"],
        field["directionXYZ"],
        needle_half_length_voxels=needle_half_length,
        maximum_search_radius_voxels=maximum_radius,
    )
    segment_index = nearest["nearestSegmentNeedleIndex"]
    valid_segment = segment_index >= 0
    safe_segment_index = np.maximum(segment_index, 0)
    normal_angle = np.full(len(seed_index), np.nan, dtype=np.float32)
    fiber_plane_error = np.full(len(seed_index), np.nan, dtype=np.float32)
    normal_angle[valid_segment] = _angle_degrees(
        slab_normal[valid_segment], field["normalXYZ"][safe_segment_index[valid_segment]]
    )
    fiber_plane_error[valid_segment] = _fiber_plane_error_degrees(
        slab_normal[valid_segment],
        field["directionXYZ"][safe_segment_index[valid_segment]],
    )
    audit_arrays = {
        "slabSeedIndex": seed_index.astype(np.int32),
        **nearest,
        "nearestSegmentNeedleNormalAngleDegrees": normal_angle,
        "nearestSegmentFiberPlaneErrorDegrees": fiber_plane_error,
    }
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **audit_arrays)
    temporary.replace(data_path)

    component_records, component_summary = _component_records(
        component_id,
        nearest["nearestCenterDistanceVoxels"],
        nearest["nearestSegmentDistanceVoxels"],
        normal_angle,
        fiber_plane_error,
        resolved,
    )
    world_bounds = slab_manifest["geometry"]["ownedWorldBounds"]
    projection_path = write_acus_coverage_projection(
        midpoint,
        nearest["nearestSegmentDistanceVoxels"],
        np.asarray(world_bounds["startXYZ"], dtype=np.float64),
        np.asarray(world_bounds["stopXYZExclusive"], dtype=np.float64),
        resolved,
        output / "acus-segment-coverage.png",
    )
    payload: dict[str, Any] = {
        "schema": ISOLATED_SLAB_ACUS_AUDIT_SCHEMA,
        "version": ISOLATED_SLAB_ACUS_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "counts": {
            "isolatedSlabSeedCount": int(len(seed_index)),
            "acusNeedleCount": int(len(field["centerXYZ"])),
            "matchedFiniteSegmentCount": int(np.count_nonzero(valid_segment)),
        },
        "coverage": {
            "nearestCenterByRadiusVoxels": _coverage_record(
                nearest["nearestCenterDistanceVoxels"],
                resolved.center_coverage_radii_voxels,
            ),
            "nearestFiniteSegmentByRadiusVoxels": _coverage_record(
                nearest["nearestSegmentDistanceVoxels"],
                resolved.segment_coverage_radii_voxels,
            ),
            "nominalInterpretation": {
                "centerRadiusVoxels": resolved.nominal_center_coverage_radius_voxels,
                "segmentRadiusVoxels": resolved.nominal_segment_coverage_radius_voxels,
                "finiteNeedleHalfLengthVoxels": needle_half_length,
            },
        },
        "distributions": {
            "nearestCenterDistanceVoxels": _finite_percentile_record(
                nearest["nearestCenterDistanceVoxels"]
            ),
            "nearestFiniteSegmentDistanceVoxels": _finite_percentile_record(
                nearest["nearestSegmentDistanceVoxels"]
            ),
            "needleNormalAngleDegrees": _finite_percentile_record(normal_angle),
            "fiberPlaneErrorDegrees": _finite_percentile_record(fiber_plane_error),
        },
        "substantialComponents": {
            **component_summary,
            "minimumSeedCount": resolved.minimum_component_seed_count,
            "records": component_records,
        },
        "rawAcusProvenance": list(raw_provenance),
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(audit_arrays),
        },
        "artifacts": {"segmentCoverageProjection": projection_path.name},
        "interpretation": {
            "coverageOnly": True,
            "changesEitherReconstruction": False,
            "segmentDistance": (
                "Euclidean distance to the closest point on a physically finite "
                "Acus needle, not merely its center"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
