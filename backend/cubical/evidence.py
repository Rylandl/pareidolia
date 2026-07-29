from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.acus import _plane_basis, _robust_common_normal

from .contracts import (
    RawAcusSettings,
    ReconstructionWindow,
    ShardSpec,
    VolumeSource,
    atomic_json,
    sha256_file,
)
from .geometry import axial_angle_radians, canonical_axis
from .raw_acus import (
    AcusCalibration,
    NeedleTable,
    normalize_ct,
    select_cell_needle_indices,
)


EVIDENCE_ARTIFACT_SCHEMA = "pareidolia.raw-acus-cell-evidence"
EVIDENCE_ARTIFACT_VERSION = 1


def _canonical_axis_array(values: np.ndarray) -> np.ndarray:
    return canonical_axis(values).astype(np.float32)


def _angular_distance_degrees(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta = np.abs(first - second)
    return np.minimum(delta, 180.0 - delta)


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    if not len(values):
        return math.nan
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = np.maximum(weights[order], 0.0)
    cumulative = np.cumsum(ordered_weights)
    if float(cumulative[-1]) <= 1.0e-12:
        return float(np.median(values))
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


def _normal_hypotheses(
    directions: np.ndarray,
    weights: np.ndarray,
    maximum: int,
) -> list[tuple[np.ndarray, float, float, np.ndarray]]:
    """Find distinct axial page normals without assigning needle signs."""

    if len(directions) < 3:
        return []
    order = np.argsort(weights)[::-1][: min(24, len(weights))]
    seeds: list[np.ndarray] = []
    global_normal, _, _ = _robust_common_normal(directions, weights)
    seeds.append(_canonical_axis_array(global_normal))
    selected_directions = directions[order]
    for first_index in range(len(selected_directions)):
        first = selected_directions[first_index]
        for second in selected_directions[first_index + 1 :]:
            cross = np.cross(first, second)
            length = float(np.linalg.norm(cross))
            if length < math.sin(math.radians(18.0)):
                continue
            seeds.append(_canonical_axis_array(cross / length))

    scored: list[tuple[float, np.ndarray]] = []
    for seed in seeds:
        residual = np.degrees(
            np.arcsin(np.clip(np.abs(directions @ seed), 0.0, 1.0))
        )
        score = float(np.sum(weights * np.exp(-0.5 * (residual / 12.0) ** 2)))
        scored.append((score, seed))
    scored.sort(key=lambda value: value[0], reverse=True)

    result: list[tuple[np.ndarray, float, float, np.ndarray]] = []
    total_weight = max(float(np.sum(weights)), 1.0e-8)
    for _, seed in scored:
        if any(axial_angle_radians(seed, previous[0]) < math.radians(10.0) for previous in result):
            continue
        residual = np.degrees(
            np.arcsin(np.clip(np.abs(directions @ seed), 0.0, 1.0))
        )
        subset = residual <= 27.0
        if int(np.count_nonzero(subset)) < 4:
            continue
        refined, eigenvalues, robust = _robust_common_normal(
            directions[subset], weights[subset]
        )
        refined = _canonical_axis_array(refined)
        if any(axial_angle_radians(refined, previous[0]) < math.radians(10.0) for previous in result):
            continue
        refined_residual = np.degrees(
            np.arcsin(np.clip(np.abs(directions @ refined), 0.0, 1.0))
        )
        median = _weighted_quantile(refined_residual, weights, 0.5)
        inlier = refined_residual <= max(10.0, min(25.0, median * 2.8))
        support = float(np.sum(weights[inlier]) / total_weight)
        eigen_total = max(float(np.sum(eigenvalues)), 1.0e-8)
        separation = float(
            np.clip((eigenvalues[1] - eigenvalues[0]) / eigen_total, 0.0, 1.0)
        )
        confidence = float(np.clip(math.sqrt(support * separation), 0.0, 1.0))
        angular_std = math.radians(
            max(1.0, _weighted_quantile(refined_residual[inlier], weights[inlier], 0.68))
        )
        all_robust = 1.0 / (
            1.0 + (refined_residual / max(4.0, median * 1.8)) ** 4
        )
        result.append(
            (refined, confidence, angular_std, all_robust.astype(np.float32))
        )
        if len(result) >= maximum:
            break
    return result


def _trilinear(array: np.ndarray, points_xyz: np.ndarray) -> np.ndarray:
    shape = points_xyz.shape[:-1]
    flat = np.asarray(points_xyz, dtype=np.float32).reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    valid = (
        (x >= 0.0)
        & (x <= array.shape[2] - 1)
        & (y >= 0.0)
        & (y <= array.shape[1] - 1)
        & (z >= 0.0)
        & (z <= array.shape[0] - 1)
    )
    x0 = np.clip(np.floor(x).astype(np.int32), 0, array.shape[2] - 1)
    y0 = np.clip(np.floor(y).astype(np.int32), 0, array.shape[1] - 1)
    z0 = np.clip(np.floor(z).astype(np.int32), 0, array.shape[0] - 1)
    x1 = np.minimum(x0 + 1, array.shape[2] - 1)
    y1 = np.minimum(y0 + 1, array.shape[1] - 1)
    z1 = np.minimum(z0 + 1, array.shape[0] - 1)
    fx, fy, fz = x - x0, y - y0, z - z0
    c000 = array[z0, y0, x0]
    c001 = array[z0, y0, x1]
    c010 = array[z0, y1, x0]
    c011 = array[z0, y1, x1]
    c100 = array[z1, y0, x0]
    c101 = array[z1, y0, x1]
    c110 = array[z1, y1, x0]
    c111 = array[z1, y1, x1]
    c00 = c000 * (1.0 - fx) + c001 * fx
    c01 = c010 * (1.0 - fx) + c011 * fx
    c10 = c100 * (1.0 - fx) + c101 * fx
    c11 = c110 * (1.0 - fx) + c111 * fx
    values = (c00 * (1.0 - fy) + c01 * fy) * (1.0 - fz) + (
        c10 * (1.0 - fy) + c11 * fy
    ) * fz
    return np.where(valid, values, 0.0).reshape(shape)


@dataclass(slots=True)
class CellEvidenceTable:
    cell_xyz: np.ndarray
    cell_center_source_xyz: np.ndarray
    depth_centers_voxels: np.ndarray
    orientation_centers_degrees: np.ndarray
    normal_valid: np.ndarray
    normal_xyz: np.ndarray
    normal_confidence: np.ndarray
    normal_angular_std_radians: np.ndarray
    needle_count: np.ndarray
    total_needle_weight: np.ndarray
    density_scale: np.ndarray
    depth_orientation_density: np.ndarray
    depth_support: np.ndarray
    ct_mean: np.ndarray
    ct_std: np.ndarray
    ct_material_fraction: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(len(self.cell_xyz))

    @property
    def hypothesis_count(self) -> int:
        return int(self.normal_valid.shape[1])

    def validate(self) -> None:
        cells = self.cell_count
        hypotheses = self.hypothesis_count
        depths = len(self.depth_centers_voxels)
        orientations = len(self.orientation_centers_degrees)
        expected = {
            "cell_xyz": (cells, 3),
            "cell_center_source_xyz": (cells, 3),
            "normal_valid": (cells, hypotheses),
            "normal_xyz": (cells, hypotheses, 3),
            "normal_confidence": (cells, hypotheses),
            "normal_angular_std_radians": (cells, hypotheses),
            "needle_count": (cells, hypotheses),
            "total_needle_weight": (cells, hypotheses),
            "density_scale": (cells, hypotheses),
            "depth_orientation_density": (cells, hypotheses, depths, orientations),
            "depth_support": (cells, hypotheses, depths),
            "ct_mean": (cells, hypotheses, depths),
            "ct_std": (cells, hypotheses, depths),
            "ct_material_fraction": (cells, hypotheses, depths),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if np.issubdtype(value.dtype, np.floating) and np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if np.any(np.diff(self.depth_centers_voxels) <= 0.0):
            raise ValueError("depth coordinates must be strictly increasing")
        if np.any(np.diff(self.orientation_centers_degrees) <= 0.0):
            raise ValueError("orientation coordinates must be strictly increasing")
        if np.any((self.normal_confidence < 0.0) | (self.normal_confidence > 1.0)):
            raise ValueError("normal confidence lies outside [0, 1]")
        if np.any((self.ct_material_fraction < 0.0) | (self.ct_material_fraction > 1.0)):
            raise ValueError("CT material fractions lie outside [0, 1]")
        valid_normals = self.normal_xyz[self.normal_valid.astype(bool)]
        if len(valid_normals) and np.any(
            np.abs(np.linalg.norm(valid_normals, axis=1) - 1.0) > 2.0e-4
        ):
            raise ValueError("valid normal hypotheses must be unit axes")

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "cellXYZ": self.cell_xyz.astype(np.int32, copy=False),
            "cellCenterSourceXYZ": self.cell_center_source_xyz.astype(np.float32, copy=False),
            "depthCentersVoxels": self.depth_centers_voxels.astype(np.float32, copy=False),
            "orientationCentersDegrees": self.orientation_centers_degrees.astype(np.float32, copy=False),
            "normalValid": self.normal_valid.astype(np.uint8, copy=False),
            "normalXYZ": self.normal_xyz.astype(np.float32, copy=False),
            "normalConfidence": self.normal_confidence.astype(np.float32, copy=False),
            "normalAngularStdRadians": self.normal_angular_std_radians.astype(np.float32, copy=False),
            "needleCount": self.needle_count.astype(np.uint16, copy=False),
            "totalNeedleWeight": self.total_needle_weight.astype(np.float32, copy=False),
            "densityScale": self.density_scale.astype(np.float32, copy=False),
            "depthOrientationDensity": self.depth_orientation_density.astype(np.float16, copy=False),
            "depthSupport": self.depth_support.astype(np.float16, copy=False),
            "ctMean": self.ct_mean.astype(np.float16, copy=False),
            "ctStd": self.ct_std.astype(np.float16, copy=False),
            "ctMaterialFraction": self.ct_material_fraction.astype(np.float16, copy=False),
        }


def build_cell_evidence(
    source: VolumeSource,
    window: ReconstructionWindow,
    shard: ShardSpec,
    needles: NeedleTable,
    calibration: AcusCalibration,
    settings: RawAcusSettings,
) -> tuple[CellEvidenceTable, dict[str, Any]]:
    started = time.monotonic()
    stride = settings.cell_stride_voxels
    half_cube = settings.analysis_cube_voxels * 0.5
    depth_centers = np.arange(
        -half_cube,
        half_cube + settings.depth_bin_voxels * 0.5,
        settings.depth_bin_voxels,
        dtype=np.float32,
    )
    orientation_centers = (
        np.arange(settings.orientation_bins, dtype=np.float32)
        + 0.5
    ) * (180.0 / settings.orientation_bins)
    cell_indices = np.asarray(
        [
            (ix, iy, iz)
            for iz in range(shard.start_cell_xyz[2], shard.stop_cell_xyz_exclusive[2])
            for iy in range(shard.start_cell_xyz[1], shard.stop_cell_xyz_exclusive[1])
            for ix in range(shard.start_cell_xyz[0], shard.stop_cell_xyz_exclusive[0])
        ],
        dtype=np.int32,
    )
    centers = (
        np.asarray(window.origin_voxel_xyz, dtype=np.float32)[None, :]
        + (cell_indices.astype(np.float32) + 0.5) * stride
    )
    cells = len(cell_indices)
    hypotheses = settings.maximum_normal_hypotheses
    depths = len(depth_centers)
    orientations = len(orientation_centers)
    normal_valid = np.zeros((cells, hypotheses), dtype=np.uint8)
    normal_xyz = np.zeros((cells, hypotheses, 3), dtype=np.float32)
    normal_confidence = np.zeros((cells, hypotheses), dtype=np.float32)
    normal_std = np.zeros((cells, hypotheses), dtype=np.float32)
    needle_count = np.zeros((cells, hypotheses), dtype=np.uint16)
    total_weight = np.zeros((cells, hypotheses), dtype=np.float32)
    density_scale = np.zeros((cells, hypotheses), dtype=np.float32)
    density = np.zeros((cells, hypotheses, depths, orientations), dtype=np.float16)
    support = np.zeros((cells, hypotheses, depths), dtype=np.float16)
    ct_mean = np.zeros((cells, hypotheses, depths), dtype=np.float16)
    ct_std = np.zeros((cells, hypotheses, depths), dtype=np.float16)
    ct_material = np.zeros((cells, hypotheses, depths), dtype=np.float16)

    raw = source.memmap()[shard.raw_voxel_bounds.slices_zyx]
    normalized_raw = normalize_ct(raw, calibration.low, calibration.high)
    raw_start = np.asarray(shard.raw_voxel_bounds.start_xyz, dtype=np.float32)
    material_threshold = float(
        np.clip(
            (calibration.air_threshold_raw - calibration.low)
            / (calibration.high - calibration.low),
            0.0,
            1.0,
        )
    )
    tangent_positions = np.linspace(
        -0.45 * stride,
        0.45 * stride,
        settings.ct_profile_tangent_samples,
        dtype=np.float32,
    )
    aa, bb = np.meshgrid(tangent_positions, tangent_positions, indexing="ij")

    valid_cell_count = 0
    for cell_index, center in enumerate(centers):
        records = select_cell_needle_indices(
            needles, center, half_cube, settings
        )
        if len(records) < settings.minimum_needles_per_cell:
            continue
        directions = needles.direction_xyz[records]
        base_weights = (
            needles.score[records]
            * np.sqrt(
                np.maximum(
                    needles.axial_coverage[records] * needles.support_score[records],
                    0.0,
                )
            )
        ).astype(np.float32)
        normal_values = _normal_hypotheses(
            directions, base_weights, settings.maximum_normal_hypotheses
        )
        if normal_values:
            valid_cell_count += 1
        for hypothesis_index, (normal, confidence, angular_std, robust) in enumerate(normal_values):
            u_axis, v_axis = _plane_basis(normal)
            offsets = needles.center_xyz[records] - center[None, :]
            depth = offsets @ normal
            plane_residual = np.degrees(
                np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0))
            )
            projected = directions - (directions @ normal)[:, None] * normal[None, :]
            projected_length = np.linalg.norm(projected, axis=1)
            usable = projected_length >= 0.2
            projected = np.divide(
                projected,
                np.maximum(projected_length[:, None], 1.0e-7),
            )
            angles = np.degrees(
                np.arctan2(projected @ v_axis, projected @ u_axis)
            ) % 180.0
            weights = (
                base_weights
                * robust
                * np.exp(-0.5 * (plane_residual / 14.0) ** 2)
                * usable.astype(np.float32)
            )
            depth_kernel = np.exp(
                -0.5
                * (
                    (depth_centers[:, None] - depth[None, :])
                    / settings.depth_kernel_voxels
                )
                ** 2
            ).astype(np.float32)
            angle_delta = _angular_distance_degrees(
                orientation_centers[:, None], angles[None, :]
            )
            angle_kernel = np.exp(
                -0.5 * (angle_delta / settings.orientation_kernel_degrees) ** 2
            ).astype(np.float32)
            weighted_depth = depth_kernel * weights[None, :]
            raw_density = weighted_depth @ angle_kernel.T
            raw_support = np.sum(weighted_depth, axis=1)
            scale = max(float(np.max(raw_density, initial=0.0)), 1.0e-8)

            normal_valid[cell_index, hypothesis_index] = 1
            normal_xyz[cell_index, hypothesis_index] = normal
            normal_confidence[cell_index, hypothesis_index] = confidence
            normal_std[cell_index, hypothesis_index] = angular_std
            needle_count[cell_index, hypothesis_index] = int(np.count_nonzero(weights > 0.01))
            total_weight[cell_index, hypothesis_index] = float(np.sum(weights))
            density_scale[cell_index, hypothesis_index] = scale
            density[cell_index, hypothesis_index] = (raw_density / scale).astype(np.float16)
            support_scale = max(float(np.max(raw_support, initial=0.0)), 1.0e-8)
            support[cell_index, hypothesis_index] = (raw_support / support_scale).astype(np.float16)

            tangent_offsets = (
                aa[..., None] * u_axis[None, None, :]
                + bb[..., None] * v_axis[None, None, :]
            )
            sample_points = (
                center[None, None, None, :]
                + depth_centers[:, None, None, None] * normal[None, None, None, :]
                + tangent_offsets[None, :, :, :]
                - raw_start[None, None, None, :]
            )
            values = _trilinear(normalized_raw, sample_points)
            ct_mean[cell_index, hypothesis_index] = np.mean(values, axis=(1, 2)).astype(np.float16)
            ct_std[cell_index, hypothesis_index] = np.std(values, axis=(1, 2)).astype(np.float16)
            ct_material[cell_index, hypothesis_index] = np.mean(
                values >= material_threshold, axis=(1, 2)
            ).astype(np.float16)

    table = CellEvidenceTable(
        cell_indices,
        centers,
        depth_centers,
        orientation_centers,
        normal_valid,
        normal_xyz,
        normal_confidence,
        normal_std,
        needle_count,
        total_weight,
        density_scale,
        density,
        support,
        ct_mean,
        ct_std,
        ct_material,
    )
    table.validate()
    return table, {
        "cellCount": cells,
        "validCellCount": valid_cell_count,
        "normalHypothesisCount": int(np.count_nonzero(normal_valid)),
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }


def write_evidence_artifact(
    prefix: str | Path,
    table: CellEvidenceTable,
    *,
    identity_sha256: str,
    shard: ShardSpec,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    base = Path(prefix)
    data_path = base.with_suffix(".npz")
    manifest_path = base.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **table.arrays())
    temporary.replace(data_path)
    payload = {
        "schema": EVIDENCE_ARTIFACT_SCHEMA,
        "version": EVIDENCE_ARTIFACT_VERSION,
        "identitySha256": identity_sha256,
        "shard": shard.record(),
        "statistics": dict(statistics),
        "axes": {
            "depth": "signed voxels from the cell center along each axial normal",
            "orientation": "unsigned tangent angle in [0, 180) degrees",
            "density": "per-hypothesis density divided by densityScale",
            "ct": "native CT sampled on tangent squares at each signed depth",
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(manifest_path, payload)
    return payload


def read_evidence_artifact(
    prefix: str | Path,
    *,
    identity_sha256: str,
    verify: bool = True,
) -> CellEvidenceTable:
    base = Path(prefix)
    manifest = json.loads(base.with_suffix(".json").read_text())
    data_path = base.with_suffix(".npz")
    if (
        manifest.get("schema") != EVIDENCE_ARTIFACT_SCHEMA
        or int(manifest.get("version", -1)) != EVIDENCE_ARTIFACT_VERSION
        or manifest.get("identitySha256") != identity_sha256
    ):
        raise ValueError("cell evidence does not match this raw Acus pipeline")
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("cell evidence content hash mismatch")
    with np.load(data_path) as values:
        table = CellEvidenceTable(
            np.asarray(values["cellXYZ"], dtype=np.int32),
            np.asarray(values["cellCenterSourceXYZ"], dtype=np.float32),
            np.asarray(values["depthCentersVoxels"], dtype=np.float32),
            np.asarray(values["orientationCentersDegrees"], dtype=np.float32),
            np.asarray(values["normalValid"], dtype=np.uint8),
            np.asarray(values["normalXYZ"], dtype=np.float32),
            np.asarray(values["normalConfidence"], dtype=np.float32),
            np.asarray(values["normalAngularStdRadians"], dtype=np.float32),
            np.asarray(values["needleCount"], dtype=np.uint16),
            np.asarray(values["totalNeedleWeight"], dtype=np.float32),
            np.asarray(values["densityScale"], dtype=np.float32),
            np.asarray(values["depthOrientationDensity"], dtype=np.float16),
            np.asarray(values["depthSupport"], dtype=np.float16),
            np.asarray(values["ctMean"], dtype=np.float16),
            np.asarray(values["ctStd"], dtype=np.float16),
            np.asarray(values["ctMaterialFraction"], dtype=np.float16),
        )
    table.validate()
    return table
