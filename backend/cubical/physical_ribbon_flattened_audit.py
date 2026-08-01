from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import AbstractSet, Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import (
    ChartRaster,
    ComponentMesh,
    SurfaceChart,
    _draw_text,
    rasterize_chart,
    sample_depth_stack,
)
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_continuity import _draw_line
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    _sample_volume_points,
)
from .physical_ribbon_patch_states import (
    PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
    _prepare_component_exact_graph,
    _reconstruct_component_graph_state,
)
from .surface_topology import triangle_edge_region_labels


PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA = (
    "pareidolia.physical-ribbon-flattened-audit"
)
PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION = 1
PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM = "physical-ribbon-flattened-audit-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonFlattenedAuditSettings:
    maximum_components: int = 8
    pixel_step_voxels: float = 0.5
    maximum_raster_pixels: int = 768
    depth_fractions: tuple[float, ...] = (-0.35, 0.0, 0.35)
    structure_window_radius_pixels: int = 4
    minimum_orientation_coherence: float = 0.15
    minimum_boundary_edge_measurements: int = 6
    maximum_median_excess_floor_degrees: float = 5.0
    maximum_control_spread_fraction: float = 0.25
    native_seam_along_radius_voxels: float = 4.0
    native_seam_inward_range_voxels: tuple[float, float] = (1.5, 5.5)
    native_seam_sample_step_voxels: float = 1.0
    native_seam_edge_parameters: tuple[float, ...] = (0.25, 0.50, 0.75)
    native_seam_scale_hypotheses: tuple[float, ...] = (0.50, 0.75, 1.00, 1.50)
    native_seam_minimum_orientation_coherence: float = 0.15
    native_seam_control_edge_multiplier: int = 3
    native_seam_minimum_measurements: int = 6

    def __post_init__(self) -> None:
        if self.maximum_components < 1 or self.maximum_raster_pixels < 32:
            raise ValueError("flattened audit dimensions must be positive")
        if not math.isfinite(self.pixel_step_voxels) or self.pixel_step_voxels <= 0:
            raise ValueError("flattened audit pixel step must be positive")
        if not self.depth_fractions or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.depth_fractions
        ):
            raise ValueError("depth fractions must be finite and lie in [-1, 1]")
        if self.structure_window_radius_pixels < 1:
            raise ValueError("structure window radius must be positive")
        if self.minimum_boundary_edge_measurements < 3:
            raise ValueError("texture compatibility requires several boundary edges")
        if not 0.0 <= self.minimum_orientation_coherence <= 1.0:
            raise ValueError("orientation coherence gate must lie in [0, 1]")
        if (
            not math.isfinite(self.maximum_median_excess_floor_degrees)
            or self.maximum_median_excess_floor_degrees < 0.0
        ):
            raise ValueError("texture median-excess floor must be finite and nonnegative")
        if (
            not math.isfinite(self.maximum_control_spread_fraction)
            or not 0.0 <= self.maximum_control_spread_fraction <= 1.0
        ):
            raise ValueError("texture control-spread fraction must lie in [0, 1]")
        if (
            not math.isfinite(self.native_seam_along_radius_voxels)
            or self.native_seam_along_radius_voxels <= 0.0
            or not math.isfinite(self.native_seam_sample_step_voxels)
            or self.native_seam_sample_step_voxels <= 0.0
        ):
            raise ValueError("native seam sampling scales must be finite and positive")
        inward_start, inward_stop = self.native_seam_inward_range_voxels
        if (
            not math.isfinite(inward_start)
            or not math.isfinite(inward_stop)
            or not 0.0 < inward_start < inward_stop
        ):
            raise ValueError("native seam inward range must be increasing and positive")
        if not self.native_seam_edge_parameters or any(
            not math.isfinite(value) or not 0.0 < value < 1.0
            for value in self.native_seam_edge_parameters
        ):
            raise ValueError("native seam edge parameters must lie strictly in (0, 1)")
        if not self.native_seam_scale_hypotheses or any(
            not math.isfinite(value) or value <= 0.0
            for value in self.native_seam_scale_hypotheses
        ):
            raise ValueError("native seam scale hypotheses must be finite and positive")
        if not 0.0 <= self.native_seam_minimum_orientation_coherence <= 1.0:
            raise ValueError("native seam coherence gate must lie in [0, 1]")
        if self.native_seam_control_edge_multiplier < 1:
            raise ValueError("native seam control multiplier must be positive")
        if self.native_seam_minimum_measurements < 3:
            raise ValueError("native seam comparison requires several measurements")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    width = 2 * radius + 1
    padded = np.pad(
        np.asarray(values, dtype=np.float32),
        ((radius, radius), (radius, radius)),
        mode="constant",
    )
    vertical_integral = np.pad(padded, ((1, 0), (0, 0))).cumsum(axis=0)
    vertical = vertical_integral[width:] - vertical_integral[:-width]
    horizontal_integral = np.pad(vertical, ((0, 0), (1, 0))).cumsum(axis=1)
    return horizontal_integral[:, width:] - horizontal_integral[:, :-width]


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def boundary_texture_compatibility(
    added_statistics: Mapping[str, float | int],
    baseline_statistics: Mapping[str, float | int],
    *,
    minimum_measurements: int,
    median_excess_floor_degrees: float,
    control_spread_fraction: float,
) -> dict[str, float | bool | None]:
    measured = (
        int(added_statistics.get("count", 0)) >= minimum_measurements
        and int(baseline_statistics.get("count", 0)) >= minimum_measurements
        and "median" in added_statistics
        and "median" in baseline_statistics
        and "p90" in baseline_statistics
    )
    if not measured:
        return {
            "measured": False,
            "compatible": None,
        }
    baseline_median = float(baseline_statistics["median"])
    baseline_spread = max(
        float(baseline_statistics["p90"]) - baseline_median,
        0.0,
    )
    allowance = max(
        median_excess_floor_degrees,
        control_spread_fraction * baseline_spread,
    )
    excess = float(added_statistics["median"]) - baseline_median
    return {
        "measured": True,
        "compatible": bool(excess <= allowance),
        "medianExcessDegrees": round(excess, 6),
        "medianExcessAllowanceDegrees": round(allowance, 6),
    }


def _summarize_boundary_texture_depths(
    depth_records: list[dict[str, Any]],
    settings: PhysicalRibbonFlattenedAuditSettings,
) -> dict[str, int | bool | None]:
    """Evaluate one realized boundary independently at every fixed depth."""

    compatible_depth_count = 0
    measured_depth_count = 0
    for record in depth_records:
        compatibility = boundary_texture_compatibility(
            record["addedBoundaryAxialDisagreementDegrees"],
            record["baselineMeshAxialDisagreementDegrees"],
            minimum_measurements=settings.minimum_boundary_edge_measurements,
            median_excess_floor_degrees=(
                settings.maximum_median_excess_floor_degrees
            ),
            control_spread_fraction=settings.maximum_control_spread_fraction,
        )
        if not compatibility["measured"]:
            record["boundaryTextureCompatibilityMeasured"] = False
            record["boundaryTextureCompatible"] = None
            continue
        measured_depth_count += 1
        compatible = bool(compatibility["compatible"])
        compatible_depth_count += int(compatible)
        record["boundaryTextureCompatibilityMeasured"] = True
        record["boundaryTextureMedianExcessDegrees"] = compatibility[
            "medianExcessDegrees"
        ]
        record["boundaryTextureMedianExcessAllowanceDegrees"] = compatibility[
            "medianExcessAllowanceDegrees"
        ]
        record["boundaryTextureCompatible"] = compatible
    return {
        "boundaryTextureMeasuredDepthCount": measured_depth_count,
        "boundaryTextureCompatibleDepthCount": compatible_depth_count,
        "boundaryTextureCompatible": (
            bool(compatible_depth_count) if measured_depth_count else None
        ),
    }


def flattened_texture_structure(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    window_radius: int,
    minimum_coherence: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure local axial texture continuity on one flattened CT plane."""

    values = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(mask, dtype=bool)
    gradient_y = np.zeros_like(values)
    gradient_x = np.zeros_like(values)
    gradient_y[1:-1] = 0.5 * (values[2:] - values[:-2])
    gradient_x[:, 1:-1] = 0.5 * (values[:, 2:] - values[:, :-2])
    weight = valid_mask.astype(np.float32)
    count = _box_sum(weight, window_radius)
    xx = _box_sum(gradient_x * gradient_x * weight, window_radius)
    yy = _box_sum(gradient_y * gradient_y * weight, window_radius)
    xy = _box_sum(gradient_x * gradient_y * weight, window_radius)
    trace = xx + yy
    anisotropy = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    coherence = anisotropy / np.maximum(trace, 1.0e-6)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    full_window = float((2 * window_radius + 1) ** 2)
    oriented = (
        valid_mask
        & (count >= 0.60 * full_window)
        & (trace > 1.0e-4)
        & np.isfinite(coherence)
    )
    reliable = oriented & (coherence >= minimum_coherence)
    disagreement: list[np.ndarray] = []
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair = reliable[first] & reliable[second]
        if not np.any(pair):
            continue
        difference = 0.5 * np.degrees(
            np.arccos(
                np.clip(
                    np.cos(2.0 * (angle[first][pair] - angle[second][pair])),
                    -1.0,
                    1.0,
                )
            )
        )
        disagreement.append(difference)
    adjacent_disagreement = (
        np.concatenate(disagreement)
        if disagreement
        else np.empty(0, dtype=np.float32)
    )
    masked_values = values[valid_mask]
    contrast = (
        float(np.percentile(masked_values, 90) - np.percentile(masked_values, 10))
        if len(masked_values)
        else 0.0
    )
    reliable_fraction = float(np.count_nonzero(reliable)) / max(
        float(np.count_nonzero(valid_mask)), 1.0
    )
    median_coherence = (
        float(np.median(coherence[oriented])) if np.any(oriented) else 0.0
    )
    statistics = {
        "supportedPixelCount": int(np.count_nonzero(valid_mask)),
        "orientedPixelCount": int(np.count_nonzero(oriented)),
        "reliableOrientationPixelCount": int(np.count_nonzero(reliable)),
        "reliableOrientationFraction": round(reliable_fraction, 6),
        "medianCoherence": round(median_coherence, 6),
        "coherence": _percentiles(coherence[oriented]),
        "adjacentAxialDisagreementDegrees": _percentiles(adjacent_disagreement),
        "intensityP90MinusP10": round(contrast, 6),
        "structureScore": round(
            median_coherence * math.sqrt(max(reliable_fraction, 0.0)), 6
        ),
    }
    return {
        "angleRadians": angle.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "reliable": reliable.astype(np.uint8),
    }, statistics


def _native_axial_structure_axis(
    image: np.ndarray,
    *,
    minimum_coherence: float,
) -> tuple[np.ndarray | None, dict[str, float | int | bool]]:
    """Recover one unsigned line axis from a native tangent-strip sample.

    The two returned coordinates are coefficients along the physical strip's
    along-edge and inward-tangent basis.  The lower-eigenvalue structure-tensor
    axis follows line texture; its sign is intentionally left ambiguous.
    """

    values = np.asarray(image, dtype=np.float32)
    if values.ndim != 2 or min(values.shape) < 3:
        return None, {
            "supportedSampleCount": 0,
            "coherence": 0.0,
            "reliable": False,
        }
    valid = np.isfinite(values)
    gradient_u = np.zeros_like(values)
    gradient_v = np.zeros_like(values)
    gradient_u[:, 1:-1] = 0.5 * (values[:, 2:] - values[:, :-2])
    gradient_v[1:-1] = 0.5 * (values[2:] - values[:-2])
    supported = valid.copy()
    supported[:, 0] = False
    supported[:, -1] = False
    supported[0] = False
    supported[-1] = False
    supported[:, 1:-1] &= valid[:, :-2] & valid[:, 2:]
    supported[1:-1] &= valid[:-2] & valid[2:]
    supported_count = int(np.count_nonzero(supported))
    interior_count = max(
        (values.shape[0] - 2) * (values.shape[1] - 2), 1
    )
    minimum_count = max(6, int(math.ceil(0.40 * interior_count)))
    if supported_count < minimum_count:
        return None, {
            "supportedSampleCount": supported_count,
            "coherence": 0.0,
            "reliable": False,
        }
    u = gradient_u[supported].astype(np.float64)
    v = gradient_v[supported].astype(np.float64)
    tensor = np.asarray(
        (
            (float(np.dot(u, u)), float(np.dot(u, v))),
            (float(np.dot(u, v)), float(np.dot(v, v))),
        ),
        dtype=np.float64,
    )
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    trace = float(np.sum(eigenvalue))
    coherence = (
        float((eigenvalue[1] - eigenvalue[0]) / trace)
        if trace > 1.0e-6
        else 0.0
    )
    reliable = bool(
        trace > 1.0e-6
        and math.isfinite(coherence)
        and coherence >= minimum_coherence
    )
    axis = np.asarray(eigenvector[:, 0], dtype=np.float64) if reliable else None
    finite = values[valid]
    contrast = (
        float(np.percentile(finite, 90) - np.percentile(finite, 10))
        if len(finite)
        else 0.0
    )
    return axis, {
        "supportedSampleCount": supported_count,
        "coherence": round(coherence, 6),
        "intensityP90MinusP10": round(contrast, 6),
        "reliable": reliable,
    }


def _triangle_frame_at_edge(
    midpoint_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    triangle: np.ndarray,
    edge: tuple[int, int],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return shared-edge tangent, signed face normal, and inward tangent."""

    points = np.asarray(midpoint_xyz, dtype=np.float64)
    triangle_nodes = np.asarray(triangle, dtype=np.int32)
    edge_start = points[int(edge[0])]
    edge_stop = points[int(edge[1])]
    tangent = edge_stop - edge_start
    tangent_norm = float(np.linalg.norm(tangent))
    if tangent_norm <= 1.0e-8:
        raise ValueError("native seam edge has zero physical length")
    tangent /= tangent_norm
    triangle_points = points[triangle_nodes]
    face_normal = np.cross(
        triangle_points[1] - triangle_points[0],
        triangle_points[2] - triangle_points[0],
    )
    normal_norm = float(np.linalg.norm(face_normal))
    if normal_norm <= 1.0e-8:
        raise ValueError("native seam triangle is physically degenerate")
    face_normal /= normal_norm
    fitted_normal = np.mean(
        np.asarray(signed_normal_xyz, dtype=np.float64)[triangle_nodes], axis=0
    )
    if float(np.dot(face_normal, fitted_normal)) < 0.0:
        face_normal *= -1.0
    edge_midpoint = 0.5 * (edge_start + edge_stop)
    inward = np.mean(triangle_points, axis=0) - edge_midpoint
    inward -= float(np.dot(inward, tangent)) * tangent
    inward_norm = float(np.linalg.norm(inward))
    if inward_norm <= 1.0e-8:
        raise ValueError("native seam triangle has no edge-transverse extent")
    inward /= inward_norm
    return tangent, face_normal, inward


def _rotate_about_axis(
    vector: np.ndarray,
    axis: np.ndarray,
    angle_radians: float,
) -> np.ndarray:
    unit_axis = np.asarray(axis, dtype=np.float64)
    unit_axis /= max(float(np.linalg.norm(unit_axis)), 1.0e-12)
    value = np.asarray(vector, dtype=np.float64)
    cosine = math.cos(angle_radians)
    sine = math.sin(angle_radians)
    return (
        cosine * value
        + sine * np.cross(unit_axis, value)
        + (1.0 - cosine) * float(np.dot(unit_axis, value)) * unit_axis
    )


def _transported_axial_disagreement_degrees(
    first_axis_xyz: np.ndarray,
    second_axis_xyz: np.ndarray,
    edge_tangent_xyz: np.ndarray,
    first_normal_xyz: np.ndarray,
    second_normal_xyz: np.ndarray,
) -> tuple[float, float]:
    """Compare unsigned tangent axes after hinge transport around the edge."""

    edge = np.asarray(edge_tangent_xyz, dtype=np.float64)
    edge /= max(float(np.linalg.norm(edge)), 1.0e-12)
    first_normal = np.asarray(first_normal_xyz, dtype=np.float64)
    second_normal = np.asarray(second_normal_xyz, dtype=np.float64)
    first_normal -= float(np.dot(first_normal, edge)) * edge
    second_normal -= float(np.dot(second_normal, edge)) * edge
    first_normal /= max(float(np.linalg.norm(first_normal)), 1.0e-12)
    second_normal /= max(float(np.linalg.norm(second_normal)), 1.0e-12)
    sine = float(np.dot(edge, np.cross(first_normal, second_normal)))
    cosine = float(np.clip(np.dot(first_normal, second_normal), -1.0, 1.0))
    hinge_angle = math.atan2(sine, cosine)
    transported = _rotate_about_axis(first_axis_xyz, edge, hinge_angle)
    transported /= max(float(np.linalg.norm(transported)), 1.0e-12)
    second = np.asarray(second_axis_xyz, dtype=np.float64)
    second /= max(float(np.linalg.norm(second)), 1.0e-12)
    disagreement = math.degrees(
        math.acos(float(np.clip(abs(np.dot(transported, second)), 0.0, 1.0)))
    )
    return disagreement, abs(math.degrees(hinge_angle))


def _completion_triangle_native_edges(
    triangle_index: np.ndarray,
    triangles: np.ndarray,
    *,
    base_triangle_count: int,
    target_triangle_indices: AbstractSet[int],
) -> dict[str, list[tuple[tuple[int, int], int, int]]]:
    """Classify exact proposal seams and uncontaminated local controls."""

    indices = np.asarray(triangle_index, dtype=np.int64)
    triangle_values = np.asarray(triangles, dtype=np.int32)
    target = {int(value) for value in target_triangle_indices}
    edge_triangle: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        triangle = triangle_values[int(index)]
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            edge_triangle.setdefault(edge, []).append(int(index))
    seam: list[tuple[tuple[int, int], int, int]] = []
    baseline: list[tuple[tuple[int, int], int, int]] = []
    added: list[tuple[tuple[int, int], int, int]] = []
    for edge, incident in edge_triangle.items():
        if len(incident) != 2:
            continue
        first, second = incident
        first_target, second_target = first in target, second in target
        if first_target != second_target:
            baseline_triangle = second if first_target else first
            added_triangle = first if first_target else second
            if baseline_triangle < base_triangle_count:
                seam.append((edge, baseline_triangle, added_triangle))
            continue
        if first_target:
            added.append((edge, first, second))
        elif first < base_triangle_count and second < base_triangle_count:
            baseline.append((edge, first, second))
    seam.sort()
    baseline.sort()
    added.sort()
    return {"seam": seam, "baseline": baseline, "added": added}


def _nearby_native_control_edges(
    seam_edges: list[tuple[tuple[int, int], int, int]],
    control_edges: list[tuple[tuple[int, int], int, int]],
    midpoint_xyz: np.ndarray,
    *,
    multiplier: int,
) -> tuple[list[tuple[tuple[int, int], int, int]], np.ndarray]:
    if not seam_edges or not control_edges:
        return [], np.empty(0, dtype=np.float32)
    points = np.asarray(midpoint_xyz, dtype=np.float64)
    seam_midpoint = np.asarray(
        [0.5 * (points[edge[0]] + points[edge[1]]) for edge, _, _ in seam_edges]
    )
    ranked: list[tuple[float, tuple[int, int], int, int]] = []
    for edge, first, second in control_edges:
        value = 0.5 * (points[edge[0]] + points[edge[1]])
        distance = float(np.min(np.linalg.norm(seam_midpoint - value, axis=1)))
        ranked.append((distance, edge, first, second))
    ranked.sort(key=lambda value: (value[0], value[1], value[2], value[3]))
    selected = ranked[: min(len(ranked), multiplier * len(seam_edges))]
    return (
        [(edge, first, second) for _, edge, first, second in selected],
        np.asarray([distance for distance, _, _, _ in selected], dtype=np.float32),
    )


def _native_edge_depth_measurements(
    edges: list[tuple[tuple[int, int], int, int]],
    triangles: np.ndarray,
    midpoint_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    thickness_voxels: np.ndarray,
    source: VolumeSource,
    volume: np.ndarray,
    *,
    depth_fraction: float,
    scale: float,
    settings: PhysicalRibbonFlattenedAuditSettings,
) -> dict[str, Any]:
    sample_step = scale * settings.native_seam_sample_step_voxels
    along = np.arange(
        -scale * settings.native_seam_along_radius_voxels,
        scale * settings.native_seam_along_radius_voxels + 0.5 * sample_step,
        sample_step,
        dtype=np.float64,
    )
    inward_start, inward_stop = settings.native_seam_inward_range_voxels
    inward_distance = np.arange(
        scale * inward_start,
        scale * inward_stop + 0.5 * sample_step,
        sample_step,
        dtype=np.float64,
    )
    along_grid, inward_grid = np.meshgrid(along, inward_distance)
    point_values = np.asarray(midpoint_xyz, dtype=np.float64)
    triangle_values = np.asarray(triangles, dtype=np.int32)
    thickness = np.asarray(thickness_voxels, dtype=np.float64)
    disagreements: list[float] = []
    hinge_angles: list[float] = []
    coherence: list[float] = []
    attempted = 0
    for edge, first_triangle, second_triangle in edges:
        first_tangent, first_normal, first_inward = _triangle_frame_at_edge(
            point_values,
            signed_normal_xyz,
            triangle_values[first_triangle],
            edge,
        )
        second_tangent, second_normal, second_inward = _triangle_frame_at_edge(
            point_values,
            signed_normal_xyz,
            triangle_values[second_triangle],
            edge,
        )
        if float(np.dot(first_tangent, second_tangent)) < 0.0:
            second_tangent *= -1.0
        edge_start = point_values[edge[0]]
        edge_stop = point_values[edge[1]]
        side_data = (
            (
                first_tangent,
                first_normal,
                first_inward,
                float(np.mean(thickness[triangle_values[first_triangle]])),
            ),
            (
                second_tangent,
                second_normal,
                second_inward,
                float(np.mean(thickness[triangle_values[second_triangle]])),
            ),
        )
        sample_points: list[np.ndarray] = []
        for tangent, normal, inward, local_thickness in side_data:
            side_points: list[np.ndarray] = []
            for parameter in settings.native_seam_edge_parameters:
                edge_point = (1.0 - parameter) * edge_start + parameter * edge_stop
                side_points.append(
                    edge_point[None, None, :]
                    + along_grid[:, :, None] * tangent[None, None, :]
                    + inward_grid[:, :, None] * inward[None, None, :]
                    + depth_fraction * local_thickness * normal[None, None, :]
                )
            sample_points.append(np.asarray(side_points, dtype=np.float32))
        samples = _sample_volume_points(
            source,
            volume,
            np.asarray(sample_points, dtype=np.float32),
        )
        for parameter_index in range(len(settings.native_seam_edge_parameters)):
            attempted += 1
            first_axis, first_stats = _native_axial_structure_axis(
                samples[0, parameter_index],
                minimum_coherence=(
                    settings.native_seam_minimum_orientation_coherence
                ),
            )
            second_axis, second_stats = _native_axial_structure_axis(
                samples[1, parameter_index],
                minimum_coherence=(
                    settings.native_seam_minimum_orientation_coherence
                ),
            )
            if first_axis is None or second_axis is None:
                continue
            first_axis_xyz = (
                first_axis[0] * first_tangent + first_axis[1] * first_inward
            )
            second_axis_xyz = (
                second_axis[0] * second_tangent + second_axis[1] * second_inward
            )
            disagreement, hinge_angle = _transported_axial_disagreement_degrees(
                first_axis_xyz,
                second_axis_xyz,
                first_tangent,
                first_normal,
                second_normal,
            )
            disagreements.append(disagreement)
            hinge_angles.append(hinge_angle)
            coherence.extend(
                (float(first_stats["coherence"]), float(second_stats["coherence"]))
            )
    return {
        "scale": round(float(scale), 6),
        "attemptedMeasurementCount": attempted,
        "axialDisagreementDegrees": _percentiles(
            np.asarray(disagreements, dtype=np.float32)
        ),
        "hingeAngleDegrees": _percentiles(np.asarray(hinge_angles, dtype=np.float32)),
        "orientationCoherence": _percentiles(
            np.asarray(coherence, dtype=np.float32)
        ),
    }


def _native_completion_seam_audit(
    triangle_index: np.ndarray,
    triangles: np.ndarray,
    target_triangle_indices: AbstractSet[int],
    midpoint_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    thickness_voxels: np.ndarray,
    source: VolumeSource,
    volume: np.ndarray,
    *,
    base_triangle_count: int,
    settings: PhysicalRibbonFlattenedAuditSettings,
) -> dict[str, Any]:
    edge_sets = _completion_triangle_native_edges(
        triangle_index,
        triangles,
        base_triangle_count=base_triangle_count,
        target_triangle_indices=target_triangle_indices,
    )
    controls, control_distance = _nearby_native_control_edges(
        edge_sets["seam"],
        edge_sets["baseline"],
        midpoint_xyz,
        multiplier=settings.native_seam_control_edge_multiplier,
    )
    depth_records: list[dict[str, Any]] = []
    compatible_depth_count = 0
    measured_depth_count = 0
    for depth_index, depth_fraction in enumerate(settings.depth_fractions):
        scale_records: list[dict[str, Any]] = []
        for scale in settings.native_seam_scale_hypotheses:
            seam = _native_edge_depth_measurements(
                edge_sets["seam"],
                triangles,
                midpoint_xyz,
                signed_normal_xyz,
                thickness_voxels,
                source,
                volume,
                depth_fraction=float(depth_fraction),
                scale=float(scale),
                settings=settings,
            )
            control = _native_edge_depth_measurements(
                controls,
                triangles,
                midpoint_xyz,
                signed_normal_xyz,
                thickness_voxels,
                source,
                volume,
                depth_fraction=float(depth_fraction),
                scale=float(scale),
                settings=settings,
            )
            scale_records.append(
                {
                    "scale": round(float(scale), 6),
                    "seam": seam,
                    "nearbyBaselineControl": control,
                }
            )
        measurable_scale_records = [
            record
            for record in scale_records
            if int(
                record["nearbyBaselineControl"][
                    "axialDisagreementDegrees"
                ].get("count", 0)
            )
            >= settings.native_seam_minimum_measurements
        ]
        ranked_scale_records = (
            measurable_scale_records if measurable_scale_records else scale_records
        )
        selected_scale_record = min(
            ranked_scale_records,
            key=lambda record: (
                float(
                    record["nearbyBaselineControl"][
                        "axialDisagreementDegrees"
                    ].get("median", float("inf"))
                ),
                -float(
                    record["nearbyBaselineControl"][
                        "orientationCoherence"
                    ].get("median", 0.0)
                ),
                abs(float(record["scale"]) - 1.0),
            ),
        )
        seam = selected_scale_record["seam"]
        control = selected_scale_record["nearbyBaselineControl"]
        compatibility = boundary_texture_compatibility(
            seam["axialDisagreementDegrees"],
            control["axialDisagreementDegrees"],
            minimum_measurements=settings.native_seam_minimum_measurements,
            median_excess_floor_degrees=(
                settings.maximum_median_excess_floor_degrees
            ),
            control_spread_fraction=settings.maximum_control_spread_fraction,
        )
        measured = bool(compatibility["measured"])
        compatible = (
            bool(compatibility["compatible"])
            if measured
            else None
        )
        measured_depth_count += int(measured)
        compatible_depth_count += int(compatible is True)
        depth_record: dict[str, Any] = {
            "depthIndex": depth_index,
            "depthFraction": round(float(depth_fraction), 6),
            "selectedScale": selected_scale_record["scale"],
            "seam": seam,
            "nearbyBaselineControl": control,
            "scaleHypotheses": scale_records,
            "nativeSeamFiberCompatibilityMeasured": measured,
            "nativeSeamFiberCompatible": compatible,
        }
        if measured:
            depth_record["nativeSeamMedianExcessDegrees"] = compatibility[
                "medianExcessDegrees"
            ]
            depth_record["nativeSeamMedianExcessAllowanceDegrees"] = compatibility[
                "medianExcessAllowanceDegrees"
            ]
        depth_records.append(depth_record)
    return {
        "seamEdgeCount": len(edge_sets["seam"]),
        "candidateBaselineControlEdgeCount": len(edge_sets["baseline"]),
        "selectedBaselineControlEdgeCount": len(controls),
        "addedInteriorEdgeCount": len(edge_sets["added"]),
        "selectedControlDistanceVoxels": _percentiles(control_distance),
        "stripAlongRadiusVoxels": round(
            settings.native_seam_along_radius_voxels, 6
        ),
        "stripInwardRangeVoxels": [
            round(float(value), 6)
            for value in settings.native_seam_inward_range_voxels
        ],
        "stripSampleStepVoxels": round(
            settings.native_seam_sample_step_voxels, 6
        ),
        "edgeParameters": [
            round(float(value), 6)
            for value in settings.native_seam_edge_parameters
        ],
        "scaleHypotheses": [
            round(float(value), 6)
            for value in settings.native_seam_scale_hypotheses
        ],
        "nativeSeamFiberMeasuredDepthCount": measured_depth_count,
        "nativeSeamFiberCompatibleDepthCount": compatible_depth_count,
        "nativeSeamFiberCompatible": (
            bool(compatible_depth_count) if measured_depth_count else None
        ),
        "depths": depth_records,
    }


def _resolve_surface_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    if value.is_file():
        candidates = (value,)
    else:
        candidates = tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    required = {
        "selected",
        "component",
        "chartUV",
        "triangleFrontierIndex",
        "signedNormalXYZ",
        "midpointXYZ",
        "thicknessVoxels",
    }
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("state") == "complete"
            and required.issubset(manifest.get("data", {}).get("fields", ()))
            and manifest.get("method", {}).get("identityLabelsUsed") is False
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("surface root must identify exactly one flattened surface artifact")
    return matches[0]


def _load_topology(
    surface_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    reference = surface_manifest.get("identity", {}).get("topologyContinuity")
    if reference is None:
        reference = surface_manifest.get("identity", {}).get("frontier")
    if reference is None:
        raise ValueError("surface does not identify its continuation topology")
    path = Path(reference["manifestPath"])
    if sha256_file(path) != reference["manifestSha256"]:
        raise ValueError("surface topology manifest changed")
    manifest = json.loads(path.read_text())
    if manifest["data"]["sha256"] != reference["dataSha256"]:
        raise ValueError("surface topology data identity differs")
    data_path = path.parent / str(manifest["data"]["path"])
    return path, manifest, _load_npz(data_path, reference["dataSha256"])


def _added_nodes(
    surface_manifest: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
) -> np.ndarray:
    selected = np.asarray(surface["selected"], dtype=np.uint8) > 0
    proposal_offset = surface.get("proposalOffset")
    proposal_node = surface.get("proposalFrontierIndex")
    proposal_accepted = surface.get("proposalAccepted")
    added = np.zeros(len(selected), dtype=bool)
    if (
        proposal_offset is not None
        and proposal_node is not None
        and proposal_accepted is not None
    ):
        offset = np.asarray(proposal_offset, dtype=np.int64)
        node = np.asarray(proposal_node, dtype=np.int32)
        for row in np.flatnonzero(np.asarray(proposal_accepted) > 0):
            added[node[offset[row] : offset[row + 1]]] = True
        return added & selected
    configuration_reference = surface_manifest.get("identity", {}).get(
        "configuration"
    )
    if configuration_reference is None:
        return added
    path = Path(configuration_reference["manifestPath"])
    if sha256_file(path) != configuration_reference["manifestSha256"]:
        raise ValueError("baseline configuration manifest changed")
    manifest = json.loads(path.read_text())
    data_path = path.parent / str(manifest["data"]["path"])
    baseline = _load_npz(data_path, configuration_reference["dataSha256"])
    baseline_selected = np.asarray(baseline["selected"], dtype=np.uint8) > 0
    if len(baseline_selected) != len(selected):
        raise ValueError("surface and baseline frontiers differ")
    return selected & ~baseline_selected


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-6)


def _node_pixel(
    chart_uv: np.ndarray,
    node: int,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
) -> tuple[int, int]:
    value = (chart_uv[node] - chart_low) / pixel_step + padding
    return int(round(float(value[1]))), int(round(float(value[0])))


def _sample_axial_angle(
    angle: np.ndarray,
    coherence: np.ndarray,
    point_uv: np.ndarray,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
    minimum_coherence: float,
    sample_radius_pixels: int,
) -> float | None:
    value = (np.asarray(point_uv) - chart_low) / pixel_step + padding
    x_value, y_value = int(round(float(value[0]))), int(round(float(value[1])))
    if not (0 <= y_value < angle.shape[0] and 0 <= x_value < angle.shape[1]):
        return None
    y_start = max(y_value - sample_radius_pixels, 0)
    y_stop = min(y_value + sample_radius_pixels + 1, angle.shape[0])
    x_start = max(x_value - sample_radius_pixels, 0)
    x_stop = min(x_value + sample_radius_pixels + 1, angle.shape[1])
    local_coherence = coherence[y_start:y_stop, x_start:x_stop]
    local_angle = angle[y_start:y_stop, x_start:x_stop]
    member = local_coherence >= minimum_coherence
    if np.count_nonzero(member) < 3:
        return None
    weight = np.square(local_coherence[member].astype(np.float64))
    doubled = 2.0 * local_angle[member].astype(np.float64)
    vector_x = float(np.sum(weight * np.cos(doubled)))
    vector_y = float(np.sum(weight * np.sin(doubled)))
    if math.hypot(vector_x, vector_y) < 0.15 * float(np.sum(weight)):
        return None
    return 0.5 * math.atan2(vector_y, vector_x)


def _point_pair_orientation_disagreement(
    angle: np.ndarray,
    coherence: np.ndarray,
    first_uv: np.ndarray,
    second_uv: np.ndarray,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
    minimum_coherence: float,
    sample_radius_pixels: int = 2,
) -> dict[str, float | int]:
    values: list[float] = []
    for first, second in zip(np.asarray(first_uv), np.asarray(second_uv)):
        first_angle = _sample_axial_angle(
            angle,
            coherence,
            first,
            chart_low,
            pixel_step,
            padding,
            minimum_coherence,
            sample_radius_pixels,
        )
        second_angle = _sample_axial_angle(
            angle,
            coherence,
            second,
            chart_low,
            pixel_step,
            padding,
            minimum_coherence,
            sample_radius_pixels,
        )
        if first_angle is None or second_angle is None:
            continue
        values.append(
            0.5
            * math.degrees(
                math.acos(
                    float(
                        np.clip(
                            math.cos(2.0 * (first_angle - second_angle)),
                            -1.0,
                            1.0,
                        )
                    )
                )
            )
        )
    return _percentiles(np.asarray(values, dtype=np.float32))


def _sample_chart_jacobian_fiber_axis(
    angle: np.ndarray,
    coherence: np.ndarray,
    raster: ChartRaster,
    point_uv: np.ndarray,
    expected_local_triangle: int,
    triangle: np.ndarray,
    chart_uv: np.ndarray,
    midpoint_xyz: np.ndarray,
    chart_low: np.ndarray,
    padding: float,
    minimum_coherence: float,
    *,
    search_radius_pixels: int = 4,
    structure_radius_pixels: int = 2,
) -> tuple[np.ndarray | None, float]:
    """Map one local raster fiber axis back through its triangle Jacobian."""

    value = (
        (np.asarray(point_uv, dtype=np.float64) - chart_low)
        / raster.pixel_step_voxels
        + padding
    )
    target_x, target_y = float(value[0]), float(value[1])
    rounded_x, rounded_y = int(round(target_x)), int(round(target_y))
    y_start = max(rounded_y - search_radius_pixels, 0)
    y_stop = min(rounded_y + search_radius_pixels + 1, angle.shape[0])
    x_start = max(rounded_x - search_radius_pixels, 0)
    x_stop = min(rounded_x + search_radius_pixels + 1, angle.shape[1])
    owner = raster.triangle_index[y_start:y_stop, x_start:x_stop]
    owner_y, owner_x = np.nonzero(owner == expected_local_triangle)
    if not len(owner_y):
        return None, 0.0
    global_y = owner_y + y_start
    global_x = owner_x + x_start
    distance_squared = (global_x - target_x) ** 2 + (global_y - target_y) ** 2
    closest = int(np.argmin(distance_squared))
    center_y, center_x = int(global_y[closest]), int(global_x[closest])
    local_y_start = max(center_y - structure_radius_pixels, 0)
    local_y_stop = min(center_y + structure_radius_pixels + 1, angle.shape[0])
    local_x_start = max(center_x - structure_radius_pixels, 0)
    local_x_stop = min(center_x + structure_radius_pixels + 1, angle.shape[1])
    local_coherence = coherence[
        local_y_start:local_y_stop, local_x_start:local_x_stop
    ]
    local_angle = angle[local_y_start:local_y_stop, local_x_start:local_x_stop]
    member = np.isfinite(local_angle) & (local_coherence >= minimum_coherence)
    if np.count_nonzero(member) < 3:
        return None, 0.0
    weight = np.square(local_coherence[member].astype(np.float64))
    doubled = 2.0 * local_angle[member].astype(np.float64)
    vector_x = float(np.sum(weight * np.cos(doubled)))
    vector_y = float(np.sum(weight * np.sin(doubled)))
    if math.hypot(vector_x, vector_y) < 0.15 * float(np.sum(weight)):
        return None, 0.0
    gradient_angle = 0.5 * math.atan2(vector_y, vector_x)
    # The structure tensor reports the dominant gradient covector.  Its
    # perpendicular is the chart-space tangent to the same intensity line.
    fiber_uv = np.asarray(
        (-math.sin(gradient_angle), math.cos(gradient_angle)),
        dtype=np.float64,
    )
    triangle_nodes = np.asarray(triangle, dtype=np.int32)
    uv = np.asarray(chart_uv, dtype=np.float64)[triangle_nodes]
    xyz = np.asarray(midpoint_xyz, dtype=np.float64)[triangle_nodes]
    uv_basis = np.column_stack((uv[1] - uv[0], uv[2] - uv[0]))
    determinant = float(np.linalg.det(uv_basis))
    if abs(determinant) <= 1.0e-10:
        return None, 0.0
    xyz_basis = np.column_stack((xyz[1] - xyz[0], xyz[2] - xyz[0]))
    axis_xyz = xyz_basis @ np.linalg.solve(uv_basis, fiber_uv)
    axis_norm = float(np.linalg.norm(axis_xyz))
    if axis_norm <= 1.0e-10:
        return None, 0.0
    return axis_xyz / axis_norm, round(float(np.median(local_coherence[member])), 6)


def _chart_jacobian_edge_measurements(
    edges: list[tuple[tuple[int, int], int, int]],
    component_triangle_index: np.ndarray,
    triangles: np.ndarray,
    chart_uv: np.ndarray,
    midpoint_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    structure: Mapping[str, np.ndarray],
    raster: ChartRaster,
    chart_low: np.ndarray,
    *,
    minimum_coherence: float,
) -> dict[str, Any]:
    global_to_local = {
        int(global_index): local_index
        for local_index, global_index in enumerate(
            np.asarray(component_triangle_index, dtype=np.int64)
        )
    }
    triangle_values = np.asarray(triangles, dtype=np.int32)
    uv = np.asarray(chart_uv, dtype=np.float64)
    disagreements: list[float] = []
    hinge_angles: list[float] = []
    coherence_values: list[float] = []
    attempted = 0
    for edge, first_triangle, second_triangle in edges:
        if first_triangle not in global_to_local or second_triangle not in global_to_local:
            continue
        first_tangent, first_normal, _ = _triangle_frame_at_edge(
            midpoint_xyz,
            signed_normal_xyz,
            triangle_values[first_triangle],
            edge,
        )
        _, second_normal, _ = _triangle_frame_at_edge(
            midpoint_xyz,
            signed_normal_xyz,
            triangle_values[second_triangle],
            edge,
        )
        edge_start, edge_stop = uv[np.asarray(edge, dtype=np.int32)]
        for parameter in (0.25, 0.50, 0.75):
            attempted += 1
            edge_point = (1.0 - parameter) * edge_start + parameter * edge_stop
            axes: list[np.ndarray] = []
            coherences: list[float] = []
            for triangle_index in (first_triangle, second_triangle):
                centroid = np.mean(uv[triangle_values[triangle_index]], axis=0)
                sample_uv = edge_point + 0.40 * (centroid - edge_point)
                axis, local_coherence = _sample_chart_jacobian_fiber_axis(
                    structure["angleRadians"],
                    structure["coherence"],
                    raster,
                    sample_uv,
                    global_to_local[triangle_index],
                    triangle_values[triangle_index],
                    uv,
                    midpoint_xyz,
                    chart_low,
                    5.0,
                    minimum_coherence,
                )
                if axis is None:
                    break
                axes.append(axis)
                coherences.append(local_coherence)
            if len(axes) != 2:
                continue
            disagreement, hinge_angle = _transported_axial_disagreement_degrees(
                axes[0],
                axes[1],
                first_tangent,
                first_normal,
                second_normal,
            )
            disagreements.append(disagreement)
            hinge_angles.append(hinge_angle)
            coherence_values.extend(coherences)
    return {
        "attemptedMeasurementCount": attempted,
        "axialDisagreementDegrees": _percentiles(
            np.asarray(disagreements, dtype=np.float32)
        ),
        "hingeAngleDegrees": _percentiles(np.asarray(hinge_angles, dtype=np.float32)),
        "orientationCoherence": _percentiles(
            np.asarray(coherence_values, dtype=np.float32)
        ),
    }


def _edge_orientation_disagreement(
    angle: np.ndarray,
    coherence: np.ndarray,
    chart_uv: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_mask: np.ndarray,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
    minimum_coherence: float,
    sample_radius_pixels: int = 2,
) -> dict[str, float | int]:
    return _point_pair_orientation_disagreement(
        angle,
        coherence,
        chart_uv[np.asarray(edge_first)[edge_mask]],
        chart_uv[np.asarray(edge_second)[edge_mask]],
        chart_low,
        pixel_step,
        padding,
        minimum_coherence,
        sample_radius_pixels,
    )


def _completion_triangle_texture_pairs(
    triangle_index: np.ndarray,
    triangles: np.ndarray,
    chart_uv: np.ndarray,
    *,
    base_triangle_count: int,
    pixel_step: float,
    target_triangle_indices: AbstractSet[int] | None = None,
) -> dict[str, np.ndarray]:
    """Build exact seam and control samples from triangle provenance.

    With no target, every triangle appended after ``base_triangle_count`` is
    treated as one collective completion.  A target isolates one proposal so
    a locally bad closure cannot be hidden by other good closures in the same
    final component.
    """

    indices = np.asarray(triangle_index, dtype=np.int64)
    triangle_values = np.asarray(triangles, dtype=np.int32)
    centroids = np.mean(chart_uv[triangle_values], axis=1)
    edge_triangle: dict[tuple[int, int], list[int]] = {}
    for index in indices:
        triangle = triangle_values[int(index)]
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            edge_triangle.setdefault(edge, []).append(int(index))

    seam_edge: list[tuple[int, int]] = []
    seam_first: list[np.ndarray] = []
    seam_second: list[np.ndarray] = []
    added_edge: list[tuple[int, int]] = []
    added_first: list[np.ndarray] = []
    added_second: list[np.ndarray] = []
    baseline_edge: list[tuple[int, int]] = []
    baseline_first: list[np.ndarray] = []
    baseline_second: list[np.ndarray] = []

    def is_target(index: int) -> bool:
        return (
            index >= base_triangle_count
            if target_triangle_indices is None
            else index in target_triangle_indices
        )

    for edge, incident in edge_triangle.items():
        if len(incident) != 2:
            continue
        first_triangle, second_triangle = incident
        first_added = is_target(first_triangle)
        second_added = is_target(second_triangle)
        if first_added != second_added:
            baseline_triangle = second_triangle if first_added else first_triangle
            added_triangle = first_triangle if first_added else second_triangle
            # When one proposal is isolated, another completion is neither its
            # original boundary context nor a valid same-surface control.
            if baseline_triangle >= base_triangle_count:
                continue
            seam_edge.append(edge)
            edge_start, edge_stop = chart_uv[np.asarray(edge, dtype=np.int32)]
            for parameter in (0.25, 0.50, 0.75):
                edge_point = (
                    (1.0 - parameter) * edge_start + parameter * edge_stop
                )

                def inward(triangle_id: int) -> np.ndarray:
                    centroid = centroids[triangle_id]
                    distance = float(np.linalg.norm(centroid - edge_point))
                    fraction = min(
                        0.75,
                        pixel_step / max(distance, 1.0e-6),
                    )
                    return edge_point + fraction * (centroid - edge_point)

                seam_first.append(inward(baseline_triangle))
                seam_second.append(inward(added_triangle))
            continue
        if first_added:
            added_edge.append(edge)
            added_first.append(centroids[first_triangle])
            added_second.append(centroids[second_triangle])
        elif first_triangle < base_triangle_count and second_triangle < base_triangle_count:
            baseline_edge.append(edge)
            baseline_first.append(centroids[first_triangle])
            baseline_second.append(centroids[second_triangle])

    def points(values: list[np.ndarray]) -> np.ndarray:
        return np.asarray(values, dtype=np.float32).reshape((-1, 2))

    def edges(values: list[tuple[int, int]]) -> np.ndarray:
        return np.asarray(values, dtype=np.int32).reshape((-1, 2))

    return {
        "seamEdge": edges(seam_edge),
        "seamFirstUV": points(seam_first),
        "seamSecondUV": points(seam_second),
        "addedInteriorEdge": edges(added_edge),
        "addedFirstUV": points(added_first),
        "addedSecondUV": points(added_second),
        "baselineInteriorEdge": edges(baseline_edge),
        "baselineFirstUV": points(baseline_first),
        "baselineSecondUV": points(baseline_second),
    }


def _completion_proposals_by_component(
    surface: Mapping[str, np.ndarray],
    triangles: np.ndarray,
    component: np.ndarray,
    *,
    base_triangle_count: int,
) -> dict[int, list[dict[str, Any]]]:
    """Recover each accepted completion's exact final-surface triangles."""

    required = {
        "proposalAccepted",
        "proposalHoleRow",
        "completionTriangleOffset",
        "completionTriangleFrontierIndex",
    }
    if not required.issubset(surface):
        return {}
    accepted = np.asarray(surface["proposalAccepted"], dtype=np.uint8) > 0
    hole_row = np.asarray(surface["proposalHoleRow"], dtype=np.int32)
    offset = np.asarray(surface["completionTriangleOffset"], dtype=np.int64)
    catalog = np.asarray(
        surface["completionTriangleFrontierIndex"], dtype=np.int32
    ).reshape((-1, 3))
    triangle_values = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    triangle_region = triangle_edge_region_labels(triangle_values)
    if len(hole_row) != len(accepted) or len(offset) != len(accepted) + 1:
        raise ValueError("completion proposal provenance offsets differ")
    if offset[0] != 0 or np.any(np.diff(offset) < 0) or offset[-1] != len(catalog):
        raise ValueError("completion triangle provenance offsets are invalid")
    if base_triangle_count + len(catalog) != len(triangle_values):
        raise ValueError("completion catalog does not span the final triangle tail")
    if not np.array_equal(triangle_values[base_triangle_count:], catalog):
        raise ValueError("completion catalog and final triangle tail differ")

    result: dict[int, list[dict[str, Any]]] = {}
    for proposal_index in range(len(accepted)):
        start, stop = int(offset[proposal_index]), int(offset[proposal_index + 1])
        if not accepted[proposal_index]:
            if start != stop:
                raise ValueError("rejected completion owns realized triangles")
            continue
        if start == stop:
            raise ValueError("accepted completion has no realized triangles")
        final_indices = np.arange(
            base_triangle_count + start,
            base_triangle_count + stop,
            dtype=np.int32,
        )
        vertex = np.unique(triangle_values[final_indices])
        component_values = np.unique(component[vertex])
        if len(component_values) != 1 or int(component_values[0]) < 0:
            raise ValueError("completion triangles do not belong to one component")
        component_id = int(component_values[0])
        region_values = np.unique(triangle_region[final_indices])
        if len(region_values) != 1:
            raise ValueError(
                "one completion proposal spans multiple chart atlas pages"
            )
        result.setdefault(component_id, []).append(
            {
                "proposalIndex": proposal_index,
                "holeRow": int(hole_row[proposal_index]),
                "triangleIndex": final_indices,
                "triangleCount": int(stop - start),
                "triangleRegionId": int(region_values[0]),
            }
        )
    return result


def _variant_topology_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    before = record.get("before", {})
    after = record.get("after", {})
    return (
        float(
            int(before.get("macroHoleCount", 0))
            - int(after.get("macroHoleCount", 0))
        ),
        float(
            int(before.get("interiorHoleCount", 0))
            - int(after.get("interiorHoleCount", 0))
        ),
        float(
            int(before.get("triangleRegionCount", 0))
            - int(after.get("triangleRegionCount", 0))
        ),
        float(
            int(after.get("triangleCount", 0))
            - int(before.get("triangleCount", 0))
        ),
        float(record.get("triangleAreaRetention", 0.0)),
        -float(record.get("variantRank", 0)),
    )


def _rank_exact_variant_rows(
    exact_records: list[Mapping[str, Any]], maximum_count: int
) -> list[Mapping[str, Any]]:
    """Share a bounded texture budget across affected sheet components."""

    by_component: dict[int, list[Mapping[str, Any]]] = {}
    seen_rows: set[int] = set()
    for record in exact_records:
        row = int(record["patchRow"])
        if row in seen_rows:
            continue
        seen_rows.add(row)
        by_component.setdefault(int(record["priorComponent"]), []).append(record)
    for records in by_component.values():
        records.sort(key=_variant_topology_key, reverse=True)
    ranked: list[Mapping[str, Any]] = []
    depth = 0
    while len(ranked) < maximum_count:
        added = False
        for component_id in sorted(by_component):
            records = by_component[component_id]
            if depth >= len(records):
                continue
            ranked.append(records[depth])
            added = True
            if len(ranked) >= maximum_count:
                break
        if not added:
            break
        depth += 1
    return ranked


def _proposal_records_from_arrays(
    surface: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    added_offset = np.asarray(surface["patchAddedOffset"], dtype=np.int64)
    added_node = np.asarray(
        surface["patchAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(surface["patchRemovedOffset"], dtype=np.int64)
    removed_node = np.asarray(
        surface["patchRemovedFrontierIndex"], dtype=np.int32
    )
    return [
        {
            "added": added_node[
                int(added_offset[row]) : int(added_offset[row + 1])
            ],
            "removed": removed_node[
                int(removed_offset[row]) : int(removed_offset[row + 1])
            ],
        }
        for row in range(len(added_offset) - 1)
    ]


def _write_variant_surface(
    output: Path,
    patch_state_path: Path,
    surface_manifest: Mapping[str, Any],
    local_surface: Mapping[str, np.ndarray],
    component_global: np.ndarray,
    proposal: Mapping[str, Any],
    *,
    patch_row: int,
    prior_component: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "physical-ribbon-patch-variant-surface-v1.npz"
    manifest_path = output / "physical-ribbon-patch-variant-surface-v1.json"
    added = set(int(value) for value in proposal["added"])
    local_added = np.asarray(
        [index for index, value in enumerate(component_global) if int(value) in added],
        dtype=np.int32,
    )
    arrays = {
        name: np.asarray(value)
        for name, value in local_surface.items()
    }
    arrays["proposalOffset"] = np.asarray((0, len(local_added)), dtype=np.int64)
    arrays["proposalFrontierIndex"] = local_added
    arrays["proposalAccepted"] = np.ones(1, dtype=np.uint8)
    arrays["sourceFrontierIndex"] = np.asarray(component_global, dtype=np.int32)
    _write_npz(data_path, arrays)
    identity = {
        "patchState": {
            "manifestPath": str(patch_state_path),
            "manifestSha256": sha256_file(patch_state_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": surface_manifest["identity"][
            "topologyContinuity"
        ],
        "patchRow": int(patch_row),
        "priorComponent": int(prior_component),
    }
    payload = {
        "schema": "pareidolia.physical-ribbon-patch-variant-surface",
        "version": 1,
        "state": "complete",
        "identity": identity,
        "source": surface_manifest["source"],
        "geometry": surface_manifest["geometry"],
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "decisionUnit": "one exact-valid complete patch matching",
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return manifest_path


def _run_patch_variant_audit(
    surface_path: Path,
    surface_manifest: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
    output: Path,
    montage_path: Path,
    identity: Mapping[str, Any],
    *,
    settings: PhysicalRibbonFlattenedAuditSettings,
    force: bool,
    started: float,
) -> dict[str, Any]:
    exact_records = [
        record
        for record in surface_manifest.get("patchStates", {}).get(
            "exactPatchAudits", ()
        )
        if bool(record.get("accepted"))
    ]
    selected_records = _rank_exact_variant_rows(
        exact_records, settings.maximum_components
    )
    configuration_reference = surface_manifest["identity"]["configuration"]
    (
        _,
        _,
        configuration,
        _,
        _,
        topology,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    proposals = _proposal_records_from_arrays(surface)
    baseline_selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    baseline_component = np.asarray(configuration["component"], dtype=np.int32)
    rows_by_component: dict[int, list[int]] = {}
    for record in selected_records:
        rows_by_component.setdefault(int(record["priorComponent"]), []).append(
            int(record["patchRow"])
        )
    graph_by_component = {
        component_id: _prepare_component_exact_graph(
            component_id,
            proposals,
            rows,
            baseline_selected,
            baseline_component,
            topology,
        )
        for component_id, rows in sorted(rows_by_component.items())
    }
    variant_records: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    variant_root = output / "variants"
    for record in selected_records:
        row = int(record["patchRow"])
        component_id = int(record["priorComponent"])
        ok, local_surface, _, component_global = _reconstruct_component_graph_state(
            graph_by_component[component_id],
            proposals[row],
            ribbon,
            topology,
            settings=PhysicalRibbonPatchHoleSettings(),
        )
        if not ok or local_surface is None:
            raise RuntimeError("exact-valid patch variant no longer reconstructs")
        local_root = variant_root / f"patch-{row:06d}"
        local_manifest_path = _write_variant_surface(
            local_root / "surface",
            surface_path,
            surface_manifest,
            local_surface,
            component_global,
            proposals[row],
            patch_row=row,
            prior_component=component_id,
        )
        audit_root = local_root / "audit"
        local_audit = run_physical_ribbon_flattened_audit(
            local_manifest_path,
            audit_root,
            settings=settings,
            force=force,
        )
        components = local_audit.get("audit", {}).get("components", ())
        if len(components) != 1:
            raise RuntimeError("patch variant audit did not yield one component")
        variant = {
            **components[0],
            "componentId": component_id,
            "patchRow": row,
            "variantRank": int(record.get("variantRank", 0)),
            "variantProfile": str(record.get("variantProfile", "unknown")),
            "objectiveGain": round(
                float(np.asarray(surface["patchObjectiveGain"])[row]), 6
            ),
            "exactAudit": dict(record),
        }
        if "patchScopeKind" in surface:
            variant["scopeKind"] = str(surface["patchScopeKind"][row])
            variant["scopeHoleRow"] = int(surface["patchScopeHoleRow"][row])
        variant_records.append(variant)
        artifact_records.append(
            {
                "patchRow": row,
                "surfaceManifest": str(local_manifest_path),
                "auditManifest": str(
                    audit_root
                    / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
                ),
                "montage": str(audit_root / "physical-ribbon-flattened-audit.png"),
            }
        )

    canvas = np.full(
        (max(80 + 34 * len(variant_records), 120), 1100, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    _draw_text(
        canvas,
        16,
        14,
        "EXACT PATCH VARIANTS / FLATTENED NATIVE CT",
        (224, 231, 239),
        scale=2,
    )
    for index, variant in enumerate(variant_records):
        compatible = variant["boundaryTextureCompatible"]
        color = (
            (102, 227, 159)
            if compatible is True
            else (255, 105, 120)
            if compatible is False
            else (174, 184, 199)
        )
        depth_count = int(variant["boundaryTextureCompatibleDepthCount"])
        _draw_text(
            canvas,
            18,
            64 + 34 * index,
            (
                f"P {variant['patchRow']:>3}  C {variant['componentId']:>4}  "
                f"V {variant['variantRank']:>2}  {variant['variantProfile']:<20} "
                f"CT {depth_count}/{variant['boundaryTextureMeasuredDepthCount']}"
            ),
            color,
        )
    montage_path.write_bytes(rgb_png(canvas))
    finished = time.monotonic()
    compatible_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is True
    ]
    incompatible_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is False
    ]
    unmeasured_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is None
    ]
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "state": "complete",
        "identity": dict(identity),
        "source": surface_manifest["source"],
        "audit": {
            "mode": "all exact-valid complete patch variants",
            "exactGeometryVariantCount": len(exact_records),
            "flattenedVariantCount": len(variant_records),
            "omittedVariantCount": len(exact_records) - len(variant_records),
            "boundaryTextureCompatibleVariantCount": len(compatible_rows),
            "boundaryTextureIncompatibleVariantCount": len(incompatible_rows),
            "boundaryTextureUnmeasuredVariantCount": len(unmeasured_rows),
            "boundaryTextureCompatiblePatchRows": compatible_rows,
            "boundaryTextureIncompatiblePatchRows": incompatible_rows,
            "boundaryTextureUnmeasuredPatchRows": unmeasured_rows,
            "variants": variant_records,
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "artifacts": {
            "montage": montage_path.name,
            "variantAudits": artifact_records,
        },
        "method": {
            "measurement": (
                "every retained exact-valid complete matching is rebuilt, "
                "flattened, and sampled from native CT at fixed depths"
            ),
            "compatibility": (
                "variant boundaries are compared with same-surface control "
                "edges before one compatible state is chosen per component"
            ),
            "acceptanceMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(
        output / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json", payload
    )
    return payload


def run_physical_ribbon_flattened_audit(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonFlattenedAuditSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonFlattenedAuditSettings()
    surface_path, surface_manifest = _resolve_surface_manifest(surface_root)
    surface_data_path = surface_path.parent / str(surface_manifest["data"]["path"])
    surface = _load_npz(surface_data_path, surface_manifest["data"]["sha256"])
    topology_path, topology_manifest, topology = _load_topology(surface_manifest)
    source_record = surface_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    identity = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "surface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": {
            "manifestPath": str(topology_path),
            "manifestSha256": sha256_file(topology_path),
            "dataSha256": topology_manifest["data"]["sha256"],
        },
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": {
            "flattenedAudit": sha256_file(Path(__file__)),
            "surfaceTopology": sha256_file(
                Path(triangle_edge_region_labels.__code__.co_filename)
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
    montage_path = output / "physical-ribbon-flattened-audit.png"
    if not force and manifest_path.is_file() and montage_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("identity", {}).get("identitySha256") == identity[
            "identitySha256"
        ]:
            return cached

    started = time.monotonic()
    if (
        surface_manifest.get("schema") == PHYSICAL_RIBBON_PATCH_STATE_SCHEMA
        and any(
            bool(record.get("accepted"))
            for record in surface_manifest.get("patchStates", {}).get(
                "exactPatchAudits", ()
            )
        )
    ):
        return _run_patch_variant_audit(
            surface_path,
            surface_manifest,
            surface,
            output,
            montage_path,
            identity,
            settings=resolved,
            force=force,
            started=started,
        )

    selected = np.asarray(surface["selected"], dtype=np.uint8) > 0
    component = np.asarray(surface["component"], dtype=np.int32)
    added = _added_nodes(surface_manifest, surface)
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    base_triangle_count = int(
        np.asarray(
            surface.get("baseTriangleCount", np.asarray([len(triangle)])),
            dtype=np.int64,
        )[0]
    )
    completion_triangle_provenance = bool(
        0 <= base_triangle_count < len(triangle)
        and "completionTriangleFrontierIndex" in surface
    )
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    midpoint = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    triangle_region = triangle_edge_region_labels(triangle)
    completion_proposals_by_component = (
        _completion_proposals_by_component(
            surface,
            triangle,
            component,
            base_triangle_count=base_triangle_count,
        )
        if completion_triangle_provenance
        else {}
    )
    if completion_triangle_provenance:
        added_triangle_index = np.arange(
            base_triangle_count, len(triangle), dtype=np.int32
        )
        changed_region, changed_count = np.unique(
            triangle_region[added_triangle_index],
            return_counts=True,
        )
        page_candidates: list[tuple[int, int, int, int]] = []
        for region_id, added_count in zip(changed_region, changed_count):
            page_triangle_index = np.flatnonzero(
                triangle_region == int(region_id)
            )
            page_vertex = np.unique(triangle[page_triangle_index])
            component_values = np.unique(component[page_vertex])
            if len(component_values) != 1 or int(component_values[0]) < 0:
                raise ValueError(
                    "one edge-connected chart page spans surface components"
                )
            page_candidates.append(
                (
                    int(component_values[0]),
                    int(region_id),
                    int(added_count),
                    int(len(page_triangle_index)),
                )
            )
        page_candidates.sort(
            key=lambda value: (-value[2], -value[3], value[0], value[1])
        )
        ranked: list[tuple[int, int | None]] = [
            (component_id, region_id)
            for component_id, region_id, _added_count, _triangle_count in (
                page_candidates[: resolved.maximum_components]
            )
        ]
        changed_component = np.asarray(
            sorted({value[0] for value in page_candidates}), dtype=np.int32
        )
        changed_atlas_page_count = len(page_candidates)
    else:
        changed_component, changed_count = np.unique(
            component[added & (component >= 0)], return_counts=True
        )
        changed_atlas_page_count = int(len(changed_component))
        if not len(changed_component):
            labels, counts = np.unique(component[selected], return_counts=True)
            order = np.argsort(-counts)
            ranked = [
                (int(value), None)
                for value in labels[order[: resolved.maximum_components]]
            ]
        else:
            size = np.bincount(component[selected])
            order = sorted(
                range(len(changed_component)),
                key=lambda index: (
                    -int(changed_count[index]),
                    -int(size[changed_component[index]]),
                    int(changed_component[index]),
                ),
            )
            ranked = [
                (int(changed_component[index]), None)
                for index in order[: resolved.maximum_components]
            ]

    columns = 2
    tile_width, tile_height = 650, 560
    canvas_rows = max(int(math.ceil(len(ranked) / columns)), 1)
    canvas = np.full(
        (canvas_rows * tile_height, columns * tile_width, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    records: list[dict[str, Any]] = []
    completion_records: list[dict[str, Any]] = []
    displayed = 0
    for component_id, atlas_region_id in ranked:
        component_triangle_index = (
            np.flatnonzero(triangle_region == atlas_region_id)
            if atlas_region_id is not None
            else np.flatnonzero(
                np.all(component[triangle] == component_id, axis=1)
            )
        )
        component_triangle = triangle[component_triangle_index]
        if not len(component_triangle):
            continue
        vertex = np.unique(component_triangle)
        if not np.all(component[vertex] == component_id):
            raise ValueError("chart atlas page spans component identities")
        if not np.all(np.isfinite(chart_uv[vertex])):
            continue
        local_triangle = np.searchsorted(vertex, component_triangle).astype(np.int32)
        triangle_normal = _normalize_rows(
            np.mean(normal[component_triangle], axis=1)
        )
        mesh = ComponentMesh(
            component_id=component_id,
            patch_ids=(component_id,),
            vertex_xyz=midpoint[vertex].astype(np.float64),
            polygons=(),
            polygon_patch_ids=np.empty(0, dtype=np.uint64),
            triangles=local_triangle,
            triangle_patch_ids=np.full(
                len(local_triangle), component_id + 1, dtype=np.uint64
            ),
            triangle_normal_xyz=triangle_normal.astype(np.float64),
            statistics={},
        )
        chart = SurfaceChart(
            uv=chart_uv[vertex].astype(np.float64),
            anchor_vertices=(),
            statistics={},
        )
        raster = rasterize_chart(
            mesh,
            chart,
            pixel_step_voxels=resolved.pixel_step_voxels,
            maximum_pixels=resolved.maximum_raster_pixels,
            padding_pixels=5,
        )
        median_thickness = float(np.median(thickness[vertex]))
        depth_offsets = tuple(
            float(value) * median_thickness for value in resolved.depth_fractions
        )
        stack, sampling_stats = sample_depth_stack(source, raster, depth_offsets)
        depth_records: list[dict[str, Any]] = []
        structures: list[dict[str, np.ndarray]] = []
        chart_low = np.min(chart.uv, axis=0)
        component_added = np.zeros_like(added)
        component_added[vertex] = added[vertex]
        component_added_triangle_count = int(
            np.count_nonzero(component_triangle_index >= base_triangle_count)
            if completion_triangle_provenance
            else 0
        )
        mesh_edge = np.sort(
            np.concatenate(
                (
                    component_triangle[:, (0, 1)],
                    component_triangle[:, (1, 2)],
                    component_triangle[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        )
        mesh_edge = np.unique(mesh_edge, axis=0)
        mesh_first = mesh_edge[:, 0]
        mesh_second = mesh_edge[:, 1]
        texture_pairs: dict[str, np.ndarray] | None = None
        component_completion_records: list[dict[str, Any]] = []
        if completion_triangle_provenance:
            texture_pairs = _completion_triangle_texture_pairs(
                component_triangle_index,
                triangle,
                chart_uv,
                base_triangle_count=base_triangle_count,
                pixel_step=raster.pixel_step_voxels,
            )

            def provenance_mask(name: str) -> np.ndarray:
                values = {
                    tuple(int(value) for value in row)
                    for row in np.asarray(texture_pairs[name], dtype=np.int32)
                }
                return np.asarray(
                    [tuple(int(value) for value in row) in values for row in mesh_edge],
                    dtype=bool,
                )

            boundary_edge = provenance_mask("seamEdge")
            interior_added_edge = provenance_mask("addedInteriorEdge")
            baseline_edge = provenance_mask("baselineInteriorEdge")
            page_proposals = [
                proposal
                for proposal in completion_proposals_by_component.get(
                    component_id, ()
                )
                if atlas_region_id is None
                or int(proposal["triangleRegionId"]) == atlas_region_id
            ]
            for proposal in page_proposals:
                proposal_triangle_indices = {
                    int(value) for value in proposal["triangleIndex"]
                }
                proposal_pairs = _completion_triangle_texture_pairs(
                    component_triangle_index,
                    triangle,
                    chart_uv,
                    base_triangle_count=base_triangle_count,
                    pixel_step=raster.pixel_step_voxels,
                    target_triangle_indices=proposal_triangle_indices,
                )
                chart_edge_sets = _completion_triangle_native_edges(
                    component_triangle_index,
                    triangle,
                    base_triangle_count=base_triangle_count,
                    target_triangle_indices=proposal_triangle_indices,
                )
                chart_control_edges, chart_control_distance = (
                    _nearby_native_control_edges(
                        chart_edge_sets["seam"],
                        chart_edge_sets["baseline"],
                        midpoint,
                        multiplier=resolved.native_seam_control_edge_multiplier,
                    )
                )
                component_completion_records.append(
                    {
                        "proposalIndex": int(proposal["proposalIndex"]),
                        "holeRow": int(proposal["holeRow"]),
                        "componentId": component_id,
                        "triangleRegionId": int(
                            proposal["triangleRegionId"]
                        ),
                        "chartAtlasPage": (
                            f"{component_id}:{proposal['triangleRegionId']}"
                        ),
                        "triangleCount": int(proposal["triangleCount"]),
                        "boundaryEdgeCount": int(
                            len(proposal_pairs["seamEdge"])
                        ),
                        "interiorEdgeCount": int(
                            len(proposal_pairs["addedInteriorEdge"])
                        ),
                        "nativeSeamFiber": {
                            "seamEdgeCount": len(chart_edge_sets["seam"]),
                            "candidateBaselineControlEdgeCount": len(
                                chart_edge_sets["baseline"]
                            ),
                            "selectedBaselineControlEdgeCount": len(
                                chart_control_edges
                            ),
                            "selectedControlDistanceVoxels": _percentiles(
                                chart_control_distance
                            ),
                            "depths": [],
                        },
                        "_nativeSeamEdges": chart_edge_sets["seam"],
                        "_nativeControlEdges": chart_control_edges,
                        "texturePairs": proposal_pairs,
                        "depths": [],
                    }
                )
        else:
            boundary_edge = added[mesh_first] ^ added[mesh_second]
            interior_added_edge = added[mesh_first] & added[mesh_second]
            baseline_edge = ~added[mesh_first] & ~added[mesh_second]
        for depth_index, (depth_fraction, depth_offset) in enumerate(
            zip(resolved.depth_fractions, depth_offsets)
        ):
            structure, structure_stats = flattened_texture_structure(
                stack[depth_index],
                raster.mask,
                window_radius=resolved.structure_window_radius_pixels,
                minimum_coherence=resolved.minimum_orientation_coherence,
            )
            structure_stats["depthFraction"] = round(float(depth_fraction), 6)
            structure_stats["depthOffsetVoxels"] = round(float(depth_offset), 6)
            if texture_pairs is None:
                boundary_disagreement = _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    boundary_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
                added_disagreement = _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    interior_added_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
                baseline_disagreement = _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    baseline_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
            else:
                boundary_disagreement = _point_pair_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    texture_pairs["seamFirstUV"],
                    texture_pairs["seamSecondUV"],
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
                added_disagreement = _point_pair_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    texture_pairs["addedFirstUV"],
                    texture_pairs["addedSecondUV"],
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
                baseline_disagreement = _point_pair_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    texture_pairs["baselineFirstUV"],
                    texture_pairs["baselineSecondUV"],
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
            structure_stats["addedBoundaryAxialDisagreementDegrees"] = (
                boundary_disagreement
            )
            structure_stats["addedInteriorAxialDisagreementDegrees"] = (
                added_disagreement
            )
            structure_stats["baselineMeshAxialDisagreementDegrees"] = (
                baseline_disagreement
            )
            for proposal_record in component_completion_records:
                proposal_pairs = proposal_record["texturePairs"]
                proposal_boundary_disagreement = (
                    _point_pair_orientation_disagreement(
                        structure["angleRadians"],
                        structure["coherence"],
                        proposal_pairs["seamFirstUV"],
                        proposal_pairs["seamSecondUV"],
                        chart_low,
                        raster.pixel_step_voxels,
                        5.0,
                        resolved.minimum_orientation_coherence,
                    )
                )
                chart_jacobian_seam = _chart_jacobian_edge_measurements(
                    proposal_record["_nativeSeamEdges"],
                    component_triangle_index,
                    triangle,
                    chart_uv,
                    midpoint,
                    normal,
                    structure,
                    raster,
                    chart_low,
                    minimum_coherence=resolved.minimum_orientation_coherence,
                )
                chart_jacobian_control = _chart_jacobian_edge_measurements(
                    proposal_record["_nativeControlEdges"],
                    component_triangle_index,
                    triangle,
                    chart_uv,
                    midpoint,
                    normal,
                    structure,
                    raster,
                    chart_low,
                    minimum_coherence=resolved.minimum_orientation_coherence,
                )
                chart_jacobian_compatibility = boundary_texture_compatibility(
                    chart_jacobian_seam["axialDisagreementDegrees"],
                    chart_jacobian_control["axialDisagreementDegrees"],
                    minimum_measurements=resolved.native_seam_minimum_measurements,
                    median_excess_floor_degrees=(
                        resolved.maximum_median_excess_floor_degrees
                    ),
                    control_spread_fraction=(
                        resolved.maximum_control_spread_fraction
                    ),
                )
                chart_jacobian_depth: dict[str, Any] = {
                    "depthIndex": depth_index,
                    "depthFraction": round(float(depth_fraction), 6),
                    "depthOffsetVoxels": round(float(depth_offset), 6),
                    "seam": chart_jacobian_seam,
                    "nearbyBaselineControl": chart_jacobian_control,
                    "nativeSeamFiberCompatibilityMeasured": bool(
                        chart_jacobian_compatibility["measured"]
                    ),
                    "nativeSeamFiberCompatible": (
                        bool(chart_jacobian_compatibility["compatible"])
                        if chart_jacobian_compatibility["measured"]
                        else None
                    ),
                }
                if chart_jacobian_compatibility["measured"]:
                    chart_jacobian_depth[
                        "nativeSeamMedianExcessDegrees"
                    ] = chart_jacobian_compatibility["medianExcessDegrees"]
                    chart_jacobian_depth[
                        "nativeSeamMedianExcessAllowanceDegrees"
                    ] = chart_jacobian_compatibility[
                        "medianExcessAllowanceDegrees"
                    ]
                proposal_record["nativeSeamFiber"]["depths"].append(
                    chart_jacobian_depth
                )
                proposal_record["depths"].append(
                    {
                        "depthIndex": depth_index,
                        "depthFraction": round(float(depth_fraction), 6),
                        "depthOffsetVoxels": round(float(depth_offset), 6),
                        "addedBoundaryAxialDisagreementDegrees": (
                            proposal_boundary_disagreement
                        ),
                        "baselineMeshAxialDisagreementDegrees": (
                            baseline_disagreement
                        ),
                    }
                )
            structures.append(structure)
            depth_records.append(structure_stats)
        component_texture_summary = _summarize_boundary_texture_depths(
            depth_records, resolved
        )
        compatible_depth_count = int(
            component_texture_summary["boundaryTextureCompatibleDepthCount"]
        )
        measured_depth_count = int(
            component_texture_summary["boundaryTextureMeasuredDepthCount"]
        )
        texture_compatible = component_texture_summary[
            "boundaryTextureCompatible"
        ]
        for proposal_record in component_completion_records:
            proposal_summary = _summarize_boundary_texture_depths(
                proposal_record["depths"], resolved
            )
            proposal_record.update(proposal_summary)
            chart_jacobian_depths = proposal_record[
                "nativeSeamFiber"
            ]["depths"]
            chart_jacobian_measured_count = sum(
                bool(record["nativeSeamFiberCompatibilityMeasured"])
                for record in chart_jacobian_depths
            )
            chart_jacobian_compatible_count = sum(
                record["nativeSeamFiberCompatible"] is True
                for record in chart_jacobian_depths
            )
            proposal_record["nativeSeamFiber"].update(
                {
                    "nativeSeamFiberMeasuredDepthCount": (
                        chart_jacobian_measured_count
                    ),
                    "nativeSeamFiberCompatibleDepthCount": (
                        chart_jacobian_compatible_count
                    ),
                    "nativeSeamFiberCompatible": (
                        bool(chart_jacobian_compatible_count)
                        if chart_jacobian_measured_count
                        else None
                    ),
                }
            )
            proposal_record.pop("texturePairs")
            proposal_record.pop("_nativeSeamEdges")
            proposal_record.pop("_nativeControlEdges")
            completion_records.append(proposal_record)
        chosen_depth = max(
            range(len(depth_records)),
            key=lambda index: (
                float(depth_records[index]["structureScore"]),
                float(depth_records[index]["intensityP90MinusP10"]),
                -abs(float(resolved.depth_fractions[index])),
            ),
        )
        plane = stack[chosen_depth].astype(np.float32)
        values = plane[raster.mask]
        low, high = (
            np.percentile(values, (1.0, 99.0)) if len(values) else (0.0, 1.0)
        )
        normalized = np.clip(
            (plane - float(low)) / max(float(high - low), 1.0), 0.0, 1.0
        )
        grayscale = np.rint(12.0 + 243.0 * normalized).astype(np.uint8)
        grayscale[~raster.mask] = 0
        image = np.repeat(grayscale[:, :, None], 3, axis=2)
        image[raster.overlap_mask] = (255, 58, 58)
        for node in np.flatnonzero(component_added):
            y_value, x_value = _node_pixel(
                chart_uv,
                int(node),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            if not (
                1 <= y_value < image.shape[0] - 1
                and 1 <= x_value < image.shape[1] - 1
            ):
                continue
            image[y_value - 1 : y_value + 2, x_value] = (255, 83, 201)
            image[y_value, x_value - 1 : x_value + 2] = (255, 83, 201)
        for first, second in zip(mesh_first[boundary_edge], mesh_second[boundary_edge]):
            first_y, first_x = _node_pixel(
                chart_uv,
                int(first),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            second_y, second_x = _node_pixel(
                chart_uv,
                int(second),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            _draw_line(
                image,
                np.asarray((first_x, first_y), dtype=np.float32),
                np.asarray((second_x, second_y), dtype=np.float32),
                (255, 184, 72),
            )
        scale = min(
            620.0 / max(image.shape[1], 1),
            490.0 / max(image.shape[0], 1),
        )
        target_width = max(int(round(image.shape[1] * scale)), 1)
        target_height = max(int(round(image.shape[0] * scale)), 1)
        row_index = np.minimum(
            (np.arange(target_height) * image.shape[0] / target_height).astype(int),
            image.shape[0] - 1,
        )
        column_index = np.minimum(
            (np.arange(target_width) * image.shape[1] / target_width).astype(int),
            image.shape[1] - 1,
        )
        fitted = image[row_index[:, None], column_index[None, :]]
        tile_x = (displayed % columns) * tile_width
        tile_y = (displayed // columns) * tile_height
        image_x = tile_x + (tile_width - target_width) // 2
        image_y = tile_y + 54 + (490 - target_height) // 2
        canvas[
            image_y : image_y + target_height,
            image_x : image_x + target_width,
        ] = fitted
        canvas[tile_y : tile_y + 3, tile_x : tile_x + tile_width] = (
            255,
            83,
            201,
        )
        chosen_record = depth_records[chosen_depth]
        _draw_text(
            canvas,
            tile_x + 10,
            tile_y + 12,
            (
                f"C {component_id}"
                f"{f' R {atlas_region_id}' if atlas_region_id is not None else ''} "
                f"N {len(vertex)} +{np.count_nonzero(component_added)} "
                f"T+{component_added_triangle_count} "
                f"D {resolved.depth_fractions[chosen_depth]:+.2f} "
                f"Q {chosen_record['medianCoherence']:.2f}"
            ),
            (224, 231, 239),
        )
        records.append(
            {
                "componentId": component_id,
                "triangleRegionId": (
                    int(atlas_region_id)
                    if atlas_region_id is not None
                    else None
                ),
                "chartAtlasPage": (
                    f"{component_id}:{atlas_region_id}"
                    if atlas_region_id is not None
                    else str(component_id)
                ),
                "ribbonCount": int(np.count_nonzero(component == component_id)),
                "atlasPageVertexCount": int(len(vertex)),
                "surfaceVertexCount": int(len(vertex)),
                "triangleCount": int(len(component_triangle)),
                "addedRibbonCount": int(np.count_nonzero(component_added)),
                "addedSurfaceTriangleCount": component_added_triangle_count,
                "addedBoundaryEdgeCount": int(np.count_nonzero(boundary_edge)),
                "addedInteriorEdgeCount": int(np.count_nonzero(interior_added_edge)),
                "completionProposalCount": len(component_completion_records),
                "completionHoleRows": [
                    int(record["holeRow"])
                    for record in component_completion_records
                ],
                "nativeSeamFiberMeasuredCompletionCount": int(
                    sum(
                        record["nativeSeamFiber"][
                            "nativeSeamFiberCompatible"
                        ]
                        is not None
                        for record in component_completion_records
                    )
                ),
                "nativeSeamFiberCompatibleCompletionCount": int(
                    sum(
                        record["nativeSeamFiber"][
                            "nativeSeamFiberCompatible"
                        ]
                        is True
                        for record in component_completion_records
                    )
                ),
                "chosenDepthIndex": int(chosen_depth),
                "boundaryTextureMeasuredDepthCount": measured_depth_count,
                "boundaryTextureCompatibleDepthCount": compatible_depth_count,
                "boundaryTextureCompatible": texture_compatible,
                "raster": raster.statistics,
                "sampling": sampling_stats,
                "depths": depth_records,
            }
        )
        displayed += 1
    if not displayed:
        _draw_text(canvas, 20, 20, "NO ELIGIBLE CHANGED SURFACE", (224, 231, 239), scale=2)
    montage_path.write_bytes(rgb_png(canvas))
    finished = time.monotonic()
    compatible_component = sorted({
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is True
    })
    incompatible_component = sorted({
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is False
    })
    unmeasured_component = sorted({
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is None
    })
    compatible_page = [
        str(record["chartAtlasPage"])
        for record in records
        if record["boundaryTextureCompatible"] is True
    ]
    incompatible_page = [
        str(record["chartAtlasPage"])
        for record in records
        if record["boundaryTextureCompatible"] is False
    ]
    unmeasured_page = [
        str(record["chartAtlasPage"])
        for record in records
        if record["boundaryTextureCompatible"] is None
    ]
    compatible_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["boundaryTextureCompatible"] is True
    ]
    incompatible_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["boundaryTextureCompatible"] is False
    ]
    unmeasured_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["boundaryTextureCompatible"] is None
    ]
    native_compatible_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["nativeSeamFiber"]["nativeSeamFiberCompatible"] is True
    ]
    native_incompatible_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["nativeSeamFiber"]["nativeSeamFiberCompatible"] is False
    ]
    native_unmeasured_completion = [
        int(record["holeRow"])
        for record in completion_records
        if record["nativeSeamFiber"]["nativeSeamFiberCompatible"] is None
    ]
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "audit": {
            "addedRibbonCount": int(np.count_nonzero(added)),
            "addedSurfaceTriangleCount": int(
                len(triangle) - base_triangle_count
                if completion_triangle_provenance
                else 0
            ),
            "changedComponentCount": int(len(changed_component)),
            "changedAtlasPageCount": int(changed_atlas_page_count),
            "flattenedComponentCount": int(
                len({int(record["componentId"]) for record in records})
            ),
            "flattenedAtlasPageCount": int(displayed),
            "boundaryTextureCompatibleComponentCount": len(
                compatible_component
            ),
            "boundaryTextureIncompatibleComponentCount": len(
                incompatible_component
            ),
            "boundaryTextureUnmeasuredComponentCount": len(
                unmeasured_component
            ),
            "boundaryTextureCompatibleComponents": compatible_component,
            "boundaryTextureIncompatibleComponents": incompatible_component,
            "boundaryTextureUnmeasuredComponents": unmeasured_component,
            "boundaryTextureCompatibleAtlasPageCount": len(compatible_page),
            "boundaryTextureIncompatibleAtlasPageCount": len(
                incompatible_page
            ),
            "boundaryTextureUnmeasuredAtlasPageCount": len(unmeasured_page),
            "boundaryTextureCompatibleAtlasPages": compatible_page,
            "boundaryTextureIncompatibleAtlasPages": incompatible_page,
            "boundaryTextureUnmeasuredAtlasPages": unmeasured_page,
            "flattenedCompletionProposalCount": len(completion_records),
            "boundaryTextureCompatibleCompletionCount": len(
                compatible_completion
            ),
            "boundaryTextureIncompatibleCompletionCount": len(
                incompatible_completion
            ),
            "boundaryTextureUnmeasuredCompletionCount": len(
                unmeasured_completion
            ),
            "boundaryTextureCompatibleCompletionHoleRows": (
                compatible_completion
            ),
            "boundaryTextureIncompatibleCompletionHoleRows": (
                incompatible_completion
            ),
            "boundaryTextureUnmeasuredCompletionHoleRows": (
                unmeasured_completion
            ),
            "nativeSeamFiberCompatibleCompletionCount": len(
                native_compatible_completion
            ),
            "nativeSeamFiberIncompatibleCompletionCount": len(
                native_incompatible_completion
            ),
            "nativeSeamFiberUnmeasuredCompletionCount": len(
                native_unmeasured_completion
            ),
            "nativeSeamFiberCompatibleCompletionHoleRows": (
                native_compatible_completion
            ),
            "nativeSeamFiberIncompatibleCompletionHoleRows": (
                native_incompatible_completion
            ),
            "nativeSeamFiberUnmeasuredCompletionHoleRows": (
                native_unmeasured_completion
            ),
            "completionProposals": completion_records,
            "components": records,
            "depthChoice": (
                "display choice maximizes a fixed-depth local structure score; "
                "all fixed depth metrics remain reported"
            ),
            "magenta": "new collectively admitted ribbon centers",
            "yellow": (
                "exact shared edges between completion and baseline triangles"
                if completion_triangle_provenance
                else "strict continuation edges from new ribbons to the prior sheet"
            ),
            "red": "nonadjacent chart overlap",
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "artifacts": {"montage": montage_path.name},
        "method": {
            "measurement": (
                "native CT sampled on exact intrinsic charts at fixed physical "
                "depth fractions with local axial structure-tensor continuity; "
                "edge-disconnected triangle regions are independent chart "
                "atlas pages, so arbitrary cross-page UV overlap cannot veto "
                "or contaminate a physical join; "
                "dense completions use exact baseline-versus-added triangle "
                "provenance so boundary-only patches remain measurable, and "
                "each proposal is scored independently so component averages "
                "cannot conceal a locally incompatible closure; completion "
                "seams additionally map local fiber axes through each owning "
                "triangle's chart-to-physical Jacobian and compare them after "
                "physical hinge transport, making their verdict independent "
                "of global atlas distortion"
            ),
            "compatibility": (
                "a changed boundary is compatible when at least one fixed "
                "depth has a median axial disagreement no farther above its "
                "same-surface control than the larger of a fixed noise floor "
                "and a declared fraction of the control median-to-p90 spread"
            ),
            "acceptanceMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
