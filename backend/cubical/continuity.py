from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable

import numpy as np

from ..rectify import _trilinear
from .block import (
    BlockBounds,
    SurfaceBlock,
    assemble_surface_hierarchy,
    rebuild_surface_block,
)
from .contracts import (
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .geometry import ClippedPatch
from .matching import TraceMatch
from .tables import read_patch_shard


JOIN_CONTINUITY_SCHEMA = "pareidolia.cubical-join-continuity"
JOIN_CONTINUITY_VERSION = 1

JoinKey = tuple[int, int, int, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class JoinContinuitySettings:
    """Dataset-independent sampling and robust-tail settings for one join."""

    trace_samples: int = 7
    trace_endpoint_margin: float = 0.15
    depth_radius_voxels: float = 8.0
    depth_step_voxels: float = 2.0
    maximum_profile_shift_voxels: float = 6.0
    near_inset_voxels: float = 1.5
    comparison_span_voxels: float = 3.0
    tile_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3)
    outlier_standard_deviations: float = 4.0
    minimum_mismatch_ratio: float = 1.5
    minimum_log_scale: float = 0.12

    def __post_init__(self) -> None:
        if self.trace_samples < 3 or not self.trace_samples % 2:
            raise ValueError("trace samples must be odd and at least three")
        if not 0.0 < self.trace_endpoint_margin < 0.5:
            raise ValueError("trace endpoint margin must lie in (0, 0.5)")
        positive = (
            self.depth_radius_voxels,
            self.depth_step_voxels,
            self.maximum_profile_shift_voxels,
            self.near_inset_voxels,
            self.comparison_span_voxels,
            self.outlier_standard_deviations,
            self.minimum_mismatch_ratio,
            self.minimum_log_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("join-continuity settings must be finite and positive")
        if self.maximum_profile_shift_voxels >= 2.0 * self.depth_radius_voxels:
            raise ValueError("profile shift must be smaller than the sampled depth span")
        tile = tuple(int(value) for value in self.tile_shape_cells_xyz)
        if len(tile) != 3 or any(value <= 0 for value in tile):
            raise ValueError("continuity tile shape must be a positive XYZ triple")
        object.__setattr__(self, "tile_shape_cells_xyz", tile)

    @property
    def depth_offsets_voxels(self) -> np.ndarray:
        count = int(math.floor(self.depth_radius_voxels / self.depth_step_voxels))
        return np.arange(-count, count + 1, dtype=np.float64) * (
            self.depth_step_voxels
        )

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["tile_shape_cells_xyz"] = list(self.tile_shape_cells_xyz)
        values["depth_offsets_voxels"] = [
            float(value) for value in self.depth_offsets_voxels
        ]
        return values


def join_key(match: TraceMatch) -> JoinKey:
    return (
        int(match.first_patch_id),
        int(match.second_patch_id),
        int(match.face.axis),
        tuple(int(value) for value in match.face.anchor_xyz),
    )


def _resolve_source(root: Path) -> tuple[dict[str, Any], VolumeSource]:
    pipeline_path = root / "pipeline.json"
    if not pipeline_path.is_file():
        variant_path = root / "variant.json"
        if not variant_path.is_file():
            raise ValueError("continuity root has neither pipeline.json nor variant.json")
        variant = json.loads(variant_path.read_text())
        pipeline_path = Path(variant["inputRoot"]).resolve() / "pipeline.json"
    pipeline = json.loads(pipeline_path.read_text())
    if pipeline.get("state") != "complete":
        raise ValueError("native CT continuity requires a complete input pipeline")
    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    if source.source_identity["identitySha256"] != source_values["identitySha256"]:
        raise ValueError("native CT source identity changed since reconstruction")
    return pipeline, source


def _trace_points_and_normal(
    block: SurfaceBlock,
    patch_by_id: dict[int, ClippedPatch],
    match: TraceMatch,
    settings: JoinContinuitySettings,
) -> tuple[np.ndarray, dict[str, float]] | None:
    first_patch = patch_by_id[match.first_patch_id]
    second_patch = patch_by_id[match.second_patch_id]
    first_trace = first_patch.trace_on(match.face)
    second_trace = second_patch.trace_on(match.face)
    if first_trace is None or second_trace is None:
        raise ValueError("retained join is missing one face trace")
    first_crossing = {
        first_trace.first.edge: first_trace.first,
        first_trace.second.edge: first_trace.second,
    }
    second_crossing = {
        second_trace.first.edge: second_trace.first,
        second_trace.second.edge: second_trace.second,
    }
    mapping = {
        value.first_edge: value.second_edge for value in match.endpoint_agreements
    }
    if set(mapping) != set(first_crossing) or set(mapping.values()) != set(
        second_crossing
    ):
        raise ValueError("retained join has an incomplete endpoint map")
    first_endpoints = np.asarray(
        (
            first_crossing[first_trace.first.edge].point_xyz,
            first_crossing[first_trace.second.edge].point_xyz,
        ),
        dtype=np.float64,
    )
    second_endpoints = np.asarray(
        (
            second_crossing[mapping[first_trace.first.edge]].point_xyz,
            second_crossing[mapping[first_trace.second.edge]].point_xyz,
        ),
        dtype=np.float64,
    )
    first_normal = np.asarray(first_patch.estimate.normal_xyz, dtype=np.float64)
    second_normal = np.asarray(second_patch.estimate.normal_xyz, dtype=np.float64)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal *= -1.0

    def basis(
        endpoints: np.ndarray,
        normal: np.ndarray,
        cell_xyz: tuple[int, int, int],
    ) -> tuple[np.ndarray, np.ndarray] | None:
        tangent = endpoints[1] - endpoints[0]
        tangent -= normal * float(np.dot(tangent, normal))
        length = float(np.linalg.norm(tangent))
        if length <= 1.0e-6:
            return None
        tangent /= length
        conormal = np.cross(normal, tangent)
        conormal /= max(float(np.linalg.norm(conormal)), 1.0e-12)
        target = block.grid.cell_center_world(cell_xyz) - np.mean(endpoints, axis=0)
        if float(np.dot(conormal, target)) < 0.0:
            conormal *= -1.0
        return tangent, conormal

    first_basis = basis(first_endpoints, first_normal, first_patch.cell_xyz)
    second_basis = basis(second_endpoints, second_normal, second_patch.cell_xyz)
    if first_basis is None or second_basis is None:
        return None
    _, first_conormal = first_basis
    _, second_conormal = second_basis
    parameters = np.linspace(
        settings.trace_endpoint_margin,
        1.0 - settings.trace_endpoint_margin,
        settings.trace_samples,
        dtype=np.float64,
    )
    first_seam = (
        first_endpoints[0][None, :] * (1.0 - parameters[:, None])
        + first_endpoints[1][None, :] * parameters[:, None]
    )
    second_seam = (
        second_endpoints[0][None, :] * (1.0 - parameters[:, None])
        + second_endpoints[1][None, :] * parameters[:, None]
    )
    depths = settings.depth_offsets_voxels

    def samples(
        seam: np.ndarray, conormal: np.ndarray, normal: np.ndarray, inset: float
    ) -> np.ndarray:
        surface = seam + inset * conormal[None, :]
        return surface[:, None, :] + depths[None, :, None] * normal[None, None, :]

    far = settings.near_inset_voxels + settings.comparison_span_voxels
    points = np.stack(
        (
            samples(first_seam, first_conormal, first_normal, settings.near_inset_voxels),
            samples(first_seam, first_conormal, first_normal, far),
            samples(second_seam, second_conormal, second_normal, settings.near_inset_voxels),
            samples(second_seam, second_conormal, second_normal, far),
        )
    )
    seam_residual = np.linalg.norm(first_seam - second_seam, axis=1)
    return points, {
        "traceLengthVoxels": float(
            min(
                np.linalg.norm(first_endpoints[1] - first_endpoints[0]),
                np.linalg.norm(second_endpoints[1] - second_endpoints[0]),
            )
        ),
        "medianSeamResidualVoxels": float(np.median(seam_residual)),
    }


def _correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_values = np.asarray(first, dtype=np.float64).ravel()
    second_values = np.asarray(second, dtype=np.float64).ravel()
    first_values -= np.mean(first_values)
    second_values -= np.mean(second_values)
    denominator = float(np.linalg.norm(first_values) * np.linalg.norm(second_values))
    if denominator <= 1.0e-8:
        return 0.0
    return float(np.clip(np.dot(first_values, second_values) / denominator, -1.0, 1.0))


def _surface_texture_axis(
    near: np.ndarray,
    far: np.ndarray,
    trace_length_voxels: float,
    settings: JoinContinuitySettings,
) -> tuple[np.ndarray, float]:
    trace_span = trace_length_voxels * (1.0 - 2.0 * settings.trace_endpoint_margin)
    trace_step = trace_span / max(settings.trace_samples - 1, 1)
    center = 0.5 * (np.asarray(near, dtype=np.float64) + far)
    trace_gradient = np.gradient(center, trace_step, axis=0)
    inward_gradient = (np.asarray(far, dtype=np.float64) - near) / (
        settings.comparison_span_voxels
    )
    gradients = np.column_stack(
        (trace_gradient.ravel(), inward_gradient.ravel())
    )
    tensor = gradients.T @ gradients / max(len(gradients), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    total = float(np.sum(eigenvalues))
    coherence = (
        float((eigenvalues[-1] - eigenvalues[0]) / total)
        if total > 1.0e-10
        else 0.0
    )
    axis = eigenvectors[:, -1]
    axis /= max(float(np.linalg.norm(axis)), 1.0e-12)
    return axis, coherence


def _profile_shift(
    first: np.ndarray,
    second: np.ndarray,
    settings: JoinContinuitySettings,
) -> tuple[int, float, float]:
    maximum_steps = int(
        math.floor(
            settings.maximum_profile_shift_voxels / settings.depth_step_voxels
        )
    )
    values: list[tuple[float, int]] = []
    for shift in range(-maximum_steps, maximum_steps + 1):
        if shift < 0:
            first_values = first[:, :shift]
            second_values = second[:, -shift:]
        elif shift > 0:
            first_values = first[:, shift:]
            second_values = second[:, :-shift]
        else:
            first_values = first
            second_values = second
        values.append((_correlation(first_values, second_values), shift))
    zero = next(value for value, shift in values if shift == 0)
    best, best_shift = max(
        values,
        key=lambda value: (value[0], -abs(value[1]), -value[1]),
    )
    return best_shift, zero, best


def _score_values(
    values: np.ndarray,
    geometry: dict[str, float],
    settings: JoinContinuitySettings,
) -> dict[str, float]:
    first_near, first_far, second_near, second_far = np.asarray(
        values, dtype=np.float64
    )
    across_difference = np.abs(first_near - second_near)
    first_control = np.abs(first_near - first_far)
    second_control = np.abs(second_near - second_far)
    control_difference = np.concatenate((first_control.ravel(), second_control.ravel()))
    across_mismatch = float(np.median(across_difference))
    control_mismatch = float(np.median(control_difference))
    stabilizer = 0.5
    ratio = (across_mismatch + stabilizer) / (control_mismatch + stabilizer)
    first_control_correlation = _correlation(first_near, first_far)
    second_control_correlation = _correlation(second_near, second_far)
    control_correlation = 0.5 * (
        first_control_correlation + second_control_correlation
    )
    across_correlation = _correlation(first_near, second_near)
    texture_standard_deviation = float(
        np.median(
            [np.std(value, dtype=np.float64) for value in np.asarray(values)]
        )
    )
    first_texture_axis, first_texture_coherence = _surface_texture_axis(
        first_near,
        first_far,
        geometry["traceLengthVoxels"],
        settings,
    )
    second_texture_axis, second_texture_coherence = _surface_texture_axis(
        second_near,
        second_far,
        geometry["traceLengthVoxels"],
        settings,
    )
    # Each conormal points inward to its own cell, hence the second component
    # changes sign when both axes are expressed in one across-seam chart.
    second_texture_axis = second_texture_axis * np.asarray((1.0, -1.0))
    texture_cosine = float(
        np.clip(abs(np.dot(first_texture_axis, second_texture_axis)), 0.0, 1.0)
    )
    texture_angle = math.degrees(math.acos(texture_cosine))
    across_shift, across_zero, across_best = _profile_shift(
        first_near, second_near, settings
    )
    first_control_shift, first_control_zero, first_control_best = _profile_shift(
        first_near, first_far, settings
    )
    second_control_shift, second_control_zero, second_control_best = _profile_shift(
        second_near, second_far, settings
    )
    control_shift_gain = 0.5 * (
        (first_control_best - first_control_zero)
        + (second_control_best - second_control_zero)
    )
    across_shift_gain = across_best - across_zero
    return {
        **geometry,
        "acrossMismatch": across_mismatch,
        "controlMismatch": control_mismatch,
        "mismatchRatio": ratio,
        "logMismatchRatio": math.log(max(ratio, 1.0e-12)),
        "acrossCorrelation": across_correlation,
        "controlCorrelation": control_correlation,
        "correlationDeficit": control_correlation - across_correlation,
        "textureStandardDeviation": texture_standard_deviation,
        "surfaceTextureAngleDegrees": texture_angle,
        "firstTextureCoherence": first_texture_coherence,
        "secondTextureCoherence": second_texture_coherence,
        "minimumTextureCoherence": min(
            first_texture_coherence, second_texture_coherence
        ),
        "bestDepthShiftVoxels": (
            across_shift * settings.depth_step_voxels
        ),
        "zeroDepthProfileCorrelation": across_zero,
        "bestShiftedDepthProfileCorrelation": across_best,
        "depthShiftCorrelationGain": across_shift_gain,
        "controlDepthShiftCorrelationGain": control_shift_gain,
        "excessDepthShiftCorrelationGain": (
            across_shift_gain - control_shift_gain
        ),
        "firstControlDepthShiftVoxels": (
            first_control_shift * settings.depth_step_voxels
        ),
        "secondControlDepthShiftVoxels": (
            second_control_shift * settings.depth_step_voxels
        ),
    }


def score_join_continuity(
    block: SurfaceBlock,
    source: VolumeSource,
    settings: JoinContinuitySettings | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Score retained joins against equal-span within-patch CT controls."""

    resolved = settings or JoinContinuitySettings()
    patch_by_id = {value.patch_id: value for value in block.patches}
    tile_shape = np.asarray(resolved.tile_shape_cells_xyz, dtype=np.int64)
    grouped: dict[tuple[int, int, int], list[TraceMatch]] = defaultdict(list)
    for match in block.joins:
        anchor = np.asarray(match.face.anchor_xyz, dtype=np.int64)
        grouped[tuple(int(value) for value in anchor // tile_shape)].append(match)
    source_array = source.memmap()
    records: list[dict[str, Any]] = []
    completed = 0
    total = len(block.joins)
    for tile in sorted(grouped):
        valid: list[tuple[TraceMatch, np.ndarray, dict[str, float]]] = []
        for match in grouped[tile]:
            geometry = _trace_points_and_normal(
                block, patch_by_id, match, resolved
            )
            if geometry is None:
                records.append(
                    {
                        "key": join_key(match),
                        "geometryScore": float(match.score),
                        "normalAngleDegrees": math.degrees(
                            match.normal_angle_radians
                        ),
                        "geometricFiberAngleDegrees": (
                            math.degrees(match.fiber_angle_radians)
                            if match.fiber_angle_radians is not None
                            else None
                        ),
                        "status": "degenerate-trace",
                    }
                )
                continue
            points, statistics = geometry
            valid.append((match, points, statistics))
        if valid:
            all_points = np.concatenate(
                [value[1].reshape(-1, 3) for value in valid], axis=0
            )
            local_points = all_points - np.asarray(source.origin_xyz, dtype=np.float64)
            low = np.floor(np.min(local_points, axis=0) - 2.0).astype(np.int64)
            high = np.ceil(np.max(local_points, axis=0) + 3.0).astype(np.int64)
            if np.any(low < 0) or np.any(high > np.asarray(source.shape_xyz)):
                raise ValueError("join-continuity samples leave the native CT source")
            x0, y0, z0 = (int(value) for value in low)
            x1, y1, z1 = (int(value) for value in high)
            subvolume = np.asarray(
                source_array[z0:z1, y0:y1, x0:x1], dtype=np.uint8
            )
            offset = np.asarray((x0, y0, z0), dtype=np.float64)
            cursor = 0
            for match, points, geometry in valid:
                count = int(np.prod(points.shape[:-1]))
                selected = local_points[cursor : cursor + count] - offset
                cursor += count
                sampled = _trilinear(subvolume, selected).reshape(points.shape[:-1])
                records.append(
                    {
                        "key": join_key(match),
                        "geometryScore": float(match.score),
                        "normalAngleDegrees": math.degrees(
                            match.normal_angle_radians
                        ),
                        "geometricFiberAngleDegrees": (
                            math.degrees(match.fiber_angle_radians)
                            if match.fiber_angle_radians is not None
                            else None
                        ),
                        "status": "scored",
                        **_score_values(sampled, geometry, resolved),
                    }
                )
        completed += len(grouped[tile])
        if progress is not None:
            progress(completed, total)
    records.sort(key=lambda value: value["key"])
    return records


def _calibrate_rejections(
    records: list[dict[str, Any]], settings: JoinContinuitySettings
) -> dict[str, Any]:
    by_axis: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        if record["status"] == "scored":
            by_axis[int(record["key"][2])].append(record)
    calibration: dict[str, Any] = {}
    for axis, values in sorted(by_axis.items()):
        log_ratios = np.asarray(
            [value["logMismatchRatio"] for value in values], dtype=np.float64
        )
        center = float(np.median(log_ratios))
        mad = float(np.median(np.abs(log_ratios - center)))
        raw_scale = 1.4826 * mad
        effective_scale = max(raw_scale, settings.minimum_log_scale)
        threshold = center + settings.outlier_standard_deviations * effective_scale
        calibration[str(axis)] = {
            "count": len(values),
            "medianLogMismatchRatio": round(center, 7),
            "rawRobustScale": round(raw_scale, 7),
            "effectiveRobustScale": round(effective_scale, 7),
            "outlierThresholdLogRatio": round(threshold, 7),
            "outlierThresholdRatio": round(math.exp(threshold), 7),
        }
        for value in values:
            value["robustOutlierZ"] = (
                value["logMismatchRatio"] - center
            ) / effective_scale
            value["rejected"] = bool(
                value["robustOutlierZ"] > settings.outlier_standard_deviations
                and value["mismatchRatio"] >= settings.minimum_mismatch_ratio
            )
            value["rejectionReason"] = (
                "fixed-depth-mismatch-outlier" if value["rejected"] else None
            )
    for record in records:
        if record["status"] != "scored":
            record["robustOutlierZ"] = None
            record["rejected"] = False
            record["rejectionReason"] = None
    return calibration


def _component_summary(block: SurfaceBlock, limit: int = 20) -> dict[str, Any]:
    ordered = sorted(
        block.components, key=lambda value: (-len(value.patch_ids), value.component_id)
    )
    return {
        "componentCount": len(ordered),
        "top": [
            {
                "rank": rank,
                "componentId": value.component_id,
                "patchCount": len(value.patch_ids),
            }
            for rank, value in enumerate(ordered[:limit], start=1)
        ],
    }


def _write_table(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    keys = [value["key"] for value in records]
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray([value[0] for value in keys], dtype=np.uint64),
            secondPatchId=np.asarray([value[1] for value in keys], dtype=np.uint64),
            faceAxis=np.asarray([value[2] for value in keys], dtype=np.int8),
            faceAnchorXYZ=np.asarray([value[3] for value in keys], dtype=np.int32),
            geometryScore=np.asarray(
                [value["geometryScore"] for value in records], dtype=np.float32
            ),
            normalAngleDegrees=np.asarray(
                [value["normalAngleDegrees"] for value in records],
                dtype=np.float32,
            ),
            geometricFiberAngleDegrees=np.asarray(
                [
                    value["geometricFiberAngleDegrees"]
                    if value["geometricFiberAngleDegrees"] is not None
                    else np.nan
                    for value in records
                ],
                dtype=np.float32,
            ),
            scored=np.asarray(
                [value["status"] == "scored" for value in records], dtype=bool
            ),
            traceLengthVoxels=np.asarray(
                [value.get("traceLengthVoxels", np.nan) for value in records],
                dtype=np.float32,
            ),
            medianSeamResidualVoxels=np.asarray(
                [value.get("medianSeamResidualVoxels", np.nan) for value in records],
                dtype=np.float32,
            ),
            acrossMismatch=np.asarray(
                [value.get("acrossMismatch", np.nan) for value in records],
                dtype=np.float32,
            ),
            controlMismatch=np.asarray(
                [value.get("controlMismatch", np.nan) for value in records],
                dtype=np.float32,
            ),
            mismatchRatio=np.asarray(
                [value.get("mismatchRatio", np.nan) for value in records],
                dtype=np.float32,
            ),
            acrossCorrelation=np.asarray(
                [value.get("acrossCorrelation", np.nan) for value in records],
                dtype=np.float32,
            ),
            controlCorrelation=np.asarray(
                [value.get("controlCorrelation", np.nan) for value in records],
                dtype=np.float32,
            ),
            correlationDeficit=np.asarray(
                [value.get("correlationDeficit", np.nan) for value in records],
                dtype=np.float32,
            ),
            textureStandardDeviation=np.asarray(
                [value.get("textureStandardDeviation", np.nan) for value in records],
                dtype=np.float32,
            ),
            surfaceTextureAngleDegrees=np.asarray(
                [
                    value.get("surfaceTextureAngleDegrees", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            firstTextureCoherence=np.asarray(
                [value.get("firstTextureCoherence", np.nan) for value in records],
                dtype=np.float32,
            ),
            secondTextureCoherence=np.asarray(
                [value.get("secondTextureCoherence", np.nan) for value in records],
                dtype=np.float32,
            ),
            minimumTextureCoherence=np.asarray(
                [value.get("minimumTextureCoherence", np.nan) for value in records],
                dtype=np.float32,
            ),
            bestDepthShiftVoxels=np.asarray(
                [value.get("bestDepthShiftVoxels", np.nan) for value in records],
                dtype=np.float32,
            ),
            zeroDepthProfileCorrelation=np.asarray(
                [
                    value.get("zeroDepthProfileCorrelation", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            bestShiftedDepthProfileCorrelation=np.asarray(
                [
                    value.get("bestShiftedDepthProfileCorrelation", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            depthShiftCorrelationGain=np.asarray(
                [
                    value.get("depthShiftCorrelationGain", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            controlDepthShiftCorrelationGain=np.asarray(
                [
                    value.get("controlDepthShiftCorrelationGain", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            excessDepthShiftCorrelationGain=np.asarray(
                [
                    value.get("excessDepthShiftCorrelationGain", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            firstControlDepthShiftVoxels=np.asarray(
                [
                    value.get("firstControlDepthShiftVoxels", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            secondControlDepthShiftVoxels=np.asarray(
                [
                    value.get("secondControlDepthShiftVoxels", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            robustOutlierZ=np.asarray(
                [
                    value["robustOutlierZ"]
                    if value["robustOutlierZ"] is not None
                    else np.nan
                    for value in records
                ],
                dtype=np.float32,
            ),
            retained=np.asarray(
                [not value["rejected"] for value in records], dtype=bool
            ),
        )
    temporary.replace(path)


def _percentiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    return {
        f"p{percentile}": round(float(np.percentile(array, percentile)), 6)
        for percentile in (50, 90, 95, 99, 100)
    }


def _diagnostic_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    scored = [value for value in records if value["status"] == "scored"]
    shifted = Counter(
        str(round(float(value["bestDepthShiftVoxels"]), 6)) for value in scored
    )
    texture_by_axis: dict[str, Any] = {}
    for axis in range(3):
        values = [
            value
            for value in scored
            if value["key"][2] == axis
            and value["minimumTextureCoherence"] >= 0.4
            and value["geometricFiberAngleDegrees"] is not None
        ]
        measured = np.asarray(
            [value["surfaceTextureAngleDegrees"] for value in values]
        )
        geometric = np.asarray(
            [value["geometricFiberAngleDegrees"] for value in values]
        )
        correlation = (
            float(np.corrcoef(measured, geometric)[0, 1])
            if len(values) >= 3
            and float(np.std(measured)) > 0.0
            and float(np.std(geometric)) > 0.0
            else 0.0
        )
        texture_by_axis[str(axis)] = {
            "count": len(values),
            "rawVsGeometricFiberPearson": round(correlation, 6),
            "angleDegrees": _percentiles(measured),
        }
    return {
        "gate": (
            "only the per-axis robust fixed-depth mismatch tail changes connectivity"
        ),
        "surfaceTextureOrientation": {
            "state": "diagnostic-only",
            "minimumCoherenceForComparison": 0.4,
            "byFaceAxis": texture_by_axis,
        },
        "normalProfileShift": {
            "state": "diagnostic-only",
            "bestShiftCountsVoxels": dict(sorted(shifted.items(), key=lambda value: float(value[0]))),
            "excessCorrelationGain": _percentiles(
                value["excessDepthShiftCorrelationGain"] for value in scored
            ),
            "absoluteShiftAtLeastFourVoxels": sum(
                abs(value["bestDepthShiftVoxels"]) >= 4.0 for value in scored
            ),
        },
    }


def _identity(
    root: Path,
    source: VolumeSource,
    settings: JoinContinuitySettings,
) -> dict[str, Any]:
    implementation_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": JOIN_CONTINUITY_SCHEMA,
        "version": JOIN_CONTINUITY_VERSION,
        "inputRoot": str(root),
        "inputPatchManifestSha256": sha256_file(root / "selected-patches-v1.json"),
        "inputPatchDataSha256": sha256_file(root / "selected-patches-v1.npz"),
        "sourceIdentitySha256": source.source_identity["identitySha256"],
        "settings": settings.record(),
        "implementationSha256": {
            "continuity.py": sha256_file(implementation_root / "continuity.py"),
            "block.py": sha256_file(implementation_root / "block.py"),
            "contracts.py": sha256_file(implementation_root / "contracts.py"),
            "geometry.py": sha256_file(implementation_root / "geometry.py"),
            "matching.py": sha256_file(implementation_root / "matching.py"),
            "tables.py": sha256_file(implementation_root / "tables.py"),
            "rectify.py": sha256_file(implementation_root.parent / "rectify.py"),
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def run_join_continuity_refinement(
    input_root: str | Path,
    output_root: str | Path,
    *,
    settings: JoinContinuitySettings | None = None,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Score every retained join, conservatively split CT-discontinuous tails."""

    started = time.monotonic()
    resolved = settings or JoinContinuitySettings()
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if root == output:
        raise ValueError("join-continuity output must differ from its input")
    pipeline, source = _resolve_source(root)
    identity = _identity(root, source, resolved)
    manifest_path = output / "refinement.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity["identitySha256"]:
            raise ValueError("join-continuity output belongs to another identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": JOIN_CONTINUITY_SCHEMA,
        "version": JOIN_CONTINUITY_VERSION,
        "state": "assembling",
        "identity": identity,
        "inputRoot": str(root),
        "source": pipeline["identity"]["source"],
    }
    atomic_json(manifest_path, manifest)
    table = read_patch_shard(root / "selected-patches-v1", verify=True)
    baseline = assemble_surface_hierarchy(
        table.grid,
        BlockBounds((0, 0, 0), table.grid.shape_cells_xyz),
        table.to_patches(),
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
    )
    manifest["state"] = "scoring"
    atomic_json(manifest_path, manifest)
    records = score_join_continuity(
        baseline, source, resolved, progress=progress
    )
    calibration = _calibrate_rejections(records, resolved)
    record_by_key = {value["key"]: value for value in records}
    retained = [
        match for match in baseline.joins if not record_by_key[join_key(match)]["rejected"]
    ]
    refined = rebuild_surface_block(baseline, retained)
    table_path = output / "join-continuity-v1.npz"
    _write_table(table_path, records)
    baseline_component = dict(baseline.component_by_patch)
    refined_component = dict(refined.component_by_patch)
    rejected_by_component = Counter(
        baseline_component[value["key"][0]]
        for value in records
        if value["rejected"]
    )
    baseline_to_refined: dict[int, set[int]] = defaultdict(set)
    for patch_id, component_id in baseline.component_by_patch:
        baseline_to_refined[component_id].add(refined_component[patch_id])
    split_components = {
        str(component_id): {
            "rejectedJoins": rejected_by_component[component_id],
            "refinedComponentCount": len(parts),
            "refinedComponentIds": sorted(parts),
        }
        for component_id, parts in sorted(baseline_to_refined.items())
        if len(parts) > 1
    }
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-join-continuity-summary",
        "version": 1,
        "identitySha256": identity["identitySha256"],
        "inputRoot": str(root),
        "method": (
            "fixed-depth native-CT mismatch across each join versus equal-span "
            "within-patch controls; per-face-axis robust outer-tail rejection; "
            "texture orientation and profile shift are diagnostic-only"
        ),
        "counts": {
            "baselineJoins": len(baseline.joins),
            "scoredJoins": sum(value["status"] == "scored" for value in records),
            "degenerateTraceJoins": sum(
                value["status"] != "scored" for value in records
            ),
            "rejectedJoins": sum(value["rejected"] for value in records),
            "retainedJoins": len(refined.joins),
            "splitBaselineComponents": len(split_components),
        },
        "calibration": calibration,
        "diagnostics": _diagnostic_summary(records),
        "baseline": _component_summary(baseline),
        "refined": _component_summary(refined),
        "splitComponents": split_components,
        "rejectedByBaselineComponent": {
            str(key): value for key, value in sorted(rejected_by_component.items())
        },
        "artifacts": {
            "table": table_path.name,
            "tableSha256": sha256_file(table_path),
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["table"] = table_path.name
    manifest["tableSha256"] = summary["artifacts"]["tableSha256"]
    atomic_json(manifest_path, manifest)
    return summary


def apply_join_continuity_refinement(
    block: SurfaceBlock, refinement_root: str | Path
) -> SurfaceBlock:
    """Apply one complete continuity table to the exact baseline join set."""

    root = Path(refinement_root)
    manifest = json.loads((root / "refinement.json").read_text())
    if manifest.get("state") != "complete":
        raise ValueError("join-continuity refinement is not complete")
    table_path = root / manifest["table"]
    if sha256_file(table_path) != manifest["tableSha256"]:
        raise ValueError("join-continuity table hash mismatch")
    with np.load(table_path, allow_pickle=False) as values:
        keys = [
            (
                int(first),
                int(second),
                int(axis),
                tuple(int(value) for value in anchor),
            )
            for first, second, axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            )
        ]
        retained_flags = np.asarray(values["retained"], dtype=bool)
    retained_by_key = dict(zip(keys, retained_flags.tolist()))
    baseline_keys = {join_key(value) for value in block.joins}
    if set(retained_by_key) != baseline_keys:
        raise ValueError("join-continuity table does not match the baseline join set")
    return rebuild_surface_block(
        block,
        [value for value in block.joins if retained_by_key[join_key(value)]],
    )
