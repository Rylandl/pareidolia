from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.acus import _plane_basis

from .contracts import RawAcusSettings, ShardSpec, VolumeSource, atomic_json, sha256_file
from .evidence import CellEvidenceTable, _angular_distance_degrees
from .geometry import PlaneEstimate, axial_angle_radians, canonical_axis
from .raw_acus import NeedleTable, select_cell_needle_indices


CONFIGURATION_ARTIFACT_SCHEMA = "pareidolia.raw-acus-stratigraphies"
CONFIGURATION_ARTIFACT_VERSION = 1
MODE_ARTIFACT_SCHEMA = "pareidolia.raw-acus-layer-modes"
MODE_ARTIFACT_VERSION = 1


def _pack_covariance(covariance: np.ndarray) -> np.ndarray:
    return covariance[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def _unpack_covariance(values: np.ndarray) -> np.ndarray:
    result = np.zeros((3, 3), dtype=np.float64)
    result[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)] = values
    result[(1, 2, 2), (0, 0, 1)] = values[[1, 2, 4]]
    return result


def _weighted_quantile(values: np.ndarray, weights: np.ndarray, quantile: float) -> float:
    order = np.argsort(values)
    ordered_values = values[order]
    ordered_weights = np.maximum(weights[order], 0.0)
    cumulative = np.cumsum(ordered_weights)
    if not len(cumulative) or float(cumulative[-1]) <= 1.0e-10:
        return float(np.median(values)) if len(values) else math.nan
    index = int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left"))
    return float(ordered_values[min(index, len(ordered_values) - 1)])


@dataclass(frozen=True, slots=True)
class LayerMode:
    normal_hypothesis: int
    estimate: PlaneEstimate
    source_depth_voxels: float
    source_orientation_degrees: float
    evidence_score: float
    material_probability: float
    effective_support: float


@dataclass(slots=True)
class LayerModeTable:
    """All independently fitted local Acus modes before configuration pruning."""

    cell_xyz: np.ndarray
    mode_offset: np.ndarray
    normal_hypothesis: np.ndarray
    normal_xyz: np.ndarray
    height: np.ndarray
    covariance: np.ndarray
    fiber_xyz: np.ndarray
    fiber_angular_std_radians: np.ndarray
    confidence: np.ndarray
    source_depth_voxels: np.ndarray
    source_orientation_degrees: np.ndarray
    evidence_score: np.ndarray
    material_probability: np.ndarray
    effective_support: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(len(self.cell_xyz))

    @property
    def mode_count(self) -> int:
        return int(len(self.height))

    def validate(self) -> None:
        cells = self.cell_count
        modes = self.mode_count
        expected = {
            "cell_xyz": (cells, 3),
            "mode_offset": (cells + 1,),
            "normal_hypothesis": (modes,),
            "normal_xyz": (modes, 3),
            "height": (modes,),
            "covariance": (modes, 6),
            "fiber_xyz": (modes, 3),
            "fiber_angular_std_radians": (modes,),
            "confidence": (modes,),
            "source_depth_voxels": (modes,),
            "source_orientation_degrees": (modes,),
            "evidence_score": (modes,),
            "material_probability": (modes,),
            "effective_support": (modes,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if np.issubdtype(value.dtype, np.floating) and np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if (
            int(self.mode_offset[0]) != 0
            or int(self.mode_offset[-1]) != modes
            or np.any(np.diff(self.mode_offset) < 0)
        ):
            raise ValueError("mode offsets do not span the mode table")
        if modes:
            if np.any(np.abs(np.linalg.norm(self.normal_xyz, axis=1) - 1.0) > 2.0e-4):
                raise ValueError("mode normals must be unit axes")
            if np.any(np.abs(np.linalg.norm(self.fiber_xyz, axis=1) - 1.0) > 2.0e-4):
                raise ValueError("mode fibers must be unit axes")
            if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
                raise ValueError("mode confidence lies outside [0, 1]")
            if np.any((self.material_probability < 0.0) | (self.material_probability > 1.0)):
                raise ValueError("mode material probability lies outside [0, 1]")

    def mode_indices_for_cell(self, cell_index: int) -> range:
        return range(
            int(self.mode_offset[cell_index]),
            int(self.mode_offset[cell_index + 1]),
        )

    def mode(self, mode_index: int) -> LayerMode:
        return LayerMode(
            int(self.normal_hypothesis[mode_index]),
            PlaneEstimate(
                tuple(float(value) for value in self.normal_xyz[mode_index]),
                float(self.height[mode_index]),
                tuple(
                    tuple(float(value) for value in row)
                    for row in _unpack_covariance(self.covariance[mode_index])
                ),
                tuple(float(value) for value in self.fiber_xyz[mode_index]),
                float(self.fiber_angular_std_radians[mode_index]),
                float(self.confidence[mode_index]),
            ),
            float(self.source_depth_voxels[mode_index]),
            float(self.source_orientation_degrees[mode_index]),
            float(self.evidence_score[mode_index]),
            float(self.material_probability[mode_index]),
            float(self.effective_support[mode_index]),
        )

    def modes_for_cell(self, cell_index: int) -> tuple[LayerMode, ...]:
        return tuple(self.mode(index) for index in self.mode_indices_for_cell(cell_index))

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "cellXYZ": self.cell_xyz.astype(np.int32, copy=False),
            "modeOffset": self.mode_offset.astype(np.uint64, copy=False),
            "normalHypothesis": self.normal_hypothesis.astype(np.int8, copy=False),
            "normalXYZ": self.normal_xyz.astype(np.float32, copy=False),
            "height": self.height.astype(np.float32, copy=False),
            "covariance": self.covariance.astype(np.float32, copy=False),
            "fiberXYZ": self.fiber_xyz.astype(np.float32, copy=False),
            "fiberAngularStdRadians": self.fiber_angular_std_radians.astype(np.float32, copy=False),
            "confidence": self.confidence.astype(np.float32, copy=False),
            "sourceDepthVoxels": self.source_depth_voxels.astype(np.float32, copy=False),
            "sourceOrientationDegrees": self.source_orientation_degrees.astype(np.float32, copy=False),
            "evidenceScore": self.evidence_score.astype(np.float32, copy=False),
            "materialProbability": self.material_probability.astype(np.float32, copy=False),
            "effectiveSupport": self.effective_support.astype(np.float32, copy=False),
        }


@dataclass(frozen=True, slots=True)
class CellStratigraphy:
    normal_hypothesis: int
    score: float
    layers: tuple[LayerMode, ...]


@dataclass(slots=True)
class ConfigurationTable:
    cell_xyz: np.ndarray
    configuration_offset: np.ndarray
    configuration_id: np.ndarray
    configuration_log_weight: np.ndarray
    normal_hypothesis: np.ndarray
    layer_offset: np.ndarray
    layer_normal_xyz: np.ndarray
    layer_height: np.ndarray
    layer_covariance: np.ndarray
    layer_fiber_xyz: np.ndarray
    layer_fiber_angular_std_radians: np.ndarray
    layer_confidence: np.ndarray
    layer_evidence_score: np.ndarray
    layer_material_probability: np.ndarray
    layer_effective_support: np.ndarray

    @property
    def cell_count(self) -> int:
        return int(len(self.cell_xyz))

    @property
    def configuration_count(self) -> int:
        return int(len(self.configuration_id))

    @property
    def layer_count(self) -> int:
        return int(len(self.layer_height))

    def validate(self) -> None:
        cells = self.cell_count
        configurations = self.configuration_count
        layers = self.layer_count
        expected = {
            "cell_xyz": (cells, 3),
            "configuration_offset": (cells + 1,),
            "configuration_id": (configurations,),
            "configuration_log_weight": (configurations,),
            "normal_hypothesis": (configurations,),
            "layer_offset": (configurations + 1,),
            "layer_normal_xyz": (layers, 3),
            "layer_height": (layers,),
            "layer_covariance": (layers, 6),
            "layer_fiber_xyz": (layers, 3),
            "layer_fiber_angular_std_radians": (layers,),
            "layer_confidence": (layers,),
            "layer_evidence_score": (layers,),
            "layer_material_probability": (layers,),
            "layer_effective_support": (layers,),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if np.issubdtype(value.dtype, np.floating) and np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if int(self.configuration_offset[0]) != 0 or int(self.configuration_offset[-1]) != configurations:
            raise ValueError("configuration offsets do not span the configuration table")
        if int(self.layer_offset[0]) != 0 or int(self.layer_offset[-1]) != layers:
            raise ValueError("layer offsets do not span the layer table")
        if np.any(np.diff(self.configuration_offset) < 1):
            raise ValueError("every cell must retain at least one local stratigraphy")
        if np.any(np.diff(self.layer_offset) < 0):
            raise ValueError("layer offsets must be monotonic")
        if layers:
            if np.any(np.abs(np.linalg.norm(self.layer_normal_xyz, axis=1) - 1.0) > 2.0e-4):
                raise ValueError("layer normals must be unit axes")
            if np.any(np.abs(np.linalg.norm(self.layer_fiber_xyz, axis=1) - 1.0) > 2.0e-4):
                raise ValueError("layer fibers must be unit axes")
            if np.any((self.layer_confidence < 0.0) | (self.layer_confidence > 1.0)):
                raise ValueError("layer confidence lies outside [0, 1]")

    def configurations_for_cell(self, cell_index: int) -> range:
        return range(
            int(self.configuration_offset[cell_index]),
            int(self.configuration_offset[cell_index + 1]),
        )

    def estimates_for_configuration(self, configuration_index: int) -> tuple[PlaneEstimate, ...]:
        start = int(self.layer_offset[configuration_index])
        stop = int(self.layer_offset[configuration_index + 1])
        result: list[PlaneEstimate] = []
        for index in range(start, stop):
            result.append(
                PlaneEstimate(
                    tuple(float(value) for value in self.layer_normal_xyz[index]),
                    float(self.layer_height[index]),
                    tuple(
                        tuple(float(value) for value in row)
                        for row in _unpack_covariance(self.layer_covariance[index])
                    ),
                    tuple(float(value) for value in self.layer_fiber_xyz[index]),
                    float(self.layer_fiber_angular_std_radians[index]),
                    float(self.layer_confidence[index]),
                )
            )
        return tuple(result)

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "cellXYZ": self.cell_xyz.astype(np.int32, copy=False),
            "configurationOffset": self.configuration_offset.astype(np.uint64, copy=False),
            "configurationId": self.configuration_id.astype(np.uint16, copy=False),
            "configurationLogWeight": self.configuration_log_weight.astype(np.float32, copy=False),
            "normalHypothesis": self.normal_hypothesis.astype(np.int8, copy=False),
            "layerOffset": self.layer_offset.astype(np.uint64, copy=False),
            "layerNormalXYZ": self.layer_normal_xyz.astype(np.float32, copy=False),
            "layerHeight": self.layer_height.astype(np.float32, copy=False),
            "layerCovariance": self.layer_covariance.astype(np.float32, copy=False),
            "layerFiberXYZ": self.layer_fiber_xyz.astype(np.float32, copy=False),
            "layerFiberAngularStdRadians": self.layer_fiber_angular_std_radians.astype(np.float32, copy=False),
            "layerConfidence": self.layer_confidence.astype(np.float32, copy=False),
            "layerEvidenceScore": self.layer_evidence_score.astype(np.float32, copy=False),
            "layerMaterialProbability": self.layer_material_probability.astype(np.float32, copy=False),
            "layerEffectiveSupport": self.layer_effective_support.astype(np.float32, copy=False),
        }


def _mode_peaks(
    density: np.ndarray,
    support: np.ndarray,
    depth_centers: np.ndarray,
    orientation_centers: np.ndarray,
    normal: np.ndarray,
    stride: int,
    settings: RawAcusSettings,
) -> list[tuple[int, int, float]]:
    if not np.any(density > 0.0):
        return []
    owned_depth_radius = 0.5 * stride * float(np.sum(np.abs(normal)))
    valid_depth = np.abs(depth_centers) < owned_depth_radius + settings.depth_kernel_voxels
    values = np.where(valid_depth[:, None], density, 0.0)
    order = np.argsort(values.ravel())[::-1]
    peak_limit = max(float(values.ravel()[order[0]]) * 0.11, 0.035)
    selected: list[tuple[int, int, float]] = []
    for flat_index in order:
        value = float(values.ravel()[flat_index])
        if value < peak_limit or len(selected) >= settings.maximum_layer_modes:
            break
        depth_index, orientation_index = np.unravel_index(flat_index, values.shape)
        if float(support[depth_index]) < 0.04:
            continue
        separated = True
        for previous_depth, previous_orientation, _ in selected:
            depth_delta = abs(
                float(depth_centers[depth_index] - depth_centers[previous_depth])
            ) / max(settings.depth_kernel_voxels * 1.6, 1.0e-6)
            angle_delta = float(
                _angular_distance_degrees(
                    np.asarray(orientation_centers[orientation_index]),
                    np.asarray(orientation_centers[previous_orientation]),
                )
            ) / max(settings.orientation_kernel_degrees * 1.6, 1.0e-6)
            if depth_delta**2 + angle_delta**2 < 1.0:
                separated = False
                break
        if separated:
            selected.append((depth_index, orientation_index, value))
    return selected


def _fit_layer_mode(
    needles: NeedleTable,
    records: np.ndarray,
    cell_center: np.ndarray,
    base_normal: np.ndarray,
    base_normal_std: float,
    base_normal_confidence: float,
    target_depth: float,
    target_orientation: float,
    peak_value: float,
    material_probability: float,
    normal_hypothesis: int,
    settings: RawAcusSettings,
) -> LayerMode | None:
    directions = needles.direction_xyz[records]
    centers = needles.center_xyz[records]
    offsets = centers - cell_center[None, :]
    u_axis, v_axis = _plane_basis(base_normal)
    depth = offsets @ base_normal
    projected = directions - (directions @ base_normal)[:, None] * base_normal[None, :]
    lengths = np.linalg.norm(projected, axis=1)
    projected /= np.maximum(lengths[:, None], 1.0e-7)
    angles = np.degrees(np.arctan2(projected @ v_axis, projected @ u_axis)) % 180.0
    angle_delta = _angular_distance_degrees(angles, np.asarray(target_orientation))
    residual = np.degrees(
        np.arcsin(np.clip(np.abs(directions @ base_normal), 0.0, 1.0))
    )
    weights = (
        needles.score[records]
        * np.sqrt(
            np.maximum(
                needles.axial_coverage[records] * needles.support_score[records],
                0.0,
            )
        )
        * np.exp(-0.5 * ((depth - target_depth) / settings.depth_kernel_voxels) ** 2)
        * np.exp(-0.5 * (angle_delta / settings.orientation_kernel_degrees) ** 2)
        * np.exp(-0.5 * (residual / 14.0) ** 2)
        * (lengths >= 0.2)
    ).astype(np.float64)
    useful = weights >= max(float(np.max(weights, initial=0.0)) * 0.05, 1.0e-5)
    if int(np.count_nonzero(useful)) < 3:
        return None
    offsets = offsets[useful]
    directions = directions[useful]
    projected = projected[useful]
    weights = weights[useful]
    depth = depth[useful]
    coordinates = np.column_stack(
        [np.ones(len(offsets)), offsets @ u_axis, offsets @ v_axis]
    )
    robust = np.ones(len(offsets), dtype=np.float64)
    beta = np.asarray([target_depth, 0.0, 0.0], dtype=np.float64)
    inverse = np.eye(3, dtype=np.float64)
    for _ in range(4):
        combined = np.maximum(weights * robust, 1.0e-8)
        normal_matrix = coordinates.T @ (combined[:, None] * coordinates)
        inverse = np.linalg.pinv(normal_matrix, rcond=1.0e-6)
        beta = inverse @ (coordinates.T @ (combined * depth))
        fit_residual = depth - coordinates @ beta
        median = _weighted_quantile(np.abs(fit_residual), combined, 0.5)
        scale = max(0.65, 1.4826 * median)
        robust = 1.0 / (1.0 + (fit_residual / (2.5 * scale)) ** 4)
    combined = np.maximum(weights * robust, 1.0e-8)
    weight_sum = float(np.sum(combined))
    effective_support = weight_sum**2 / max(float(np.sum(combined**2)), 1.0e-8)
    if effective_support < 2.5:
        return None
    fit_residual = depth - coordinates @ beta
    residual_scale = max(
        0.35,
        1.4826 * _weighted_quantile(np.abs(fit_residual), combined, 0.5),
    )
    unnormalized_normal = base_normal - beta[1] * u_axis - beta[2] * v_axis
    normal_length = max(float(np.linalg.norm(unnormalized_normal)), 1.0e-8)
    fitted_normal = canonical_axis(unnormalized_normal)
    fitted_height = float(beta[0] / normal_length)
    if float(np.dot(fitted_normal, unnormalized_normal)) < 0.0:
        fitted_height = -fitted_height

    tangent_directions = directions - (directions @ fitted_normal)[:, None] * fitted_normal[None, :]
    tangent_lengths = np.linalg.norm(tangent_directions, axis=1)
    tangent_directions /= np.maximum(tangent_lengths[:, None], 1.0e-7)
    axial_matrix = np.einsum(
        "n,ni,nj->ij", combined, tangent_directions, tangent_directions
    )
    _, eigenvectors = np.linalg.eigh(axial_matrix)
    fiber = canonical_axis(eigenvectors[:, -1])
    fiber -= fitted_normal * float(np.dot(fiber, fitted_normal))
    fiber = canonical_axis(fiber)
    cross_fiber = np.cross(fitted_normal, fiber)
    cross_fiber /= max(float(np.linalg.norm(cross_fiber)), 1.0e-8)
    signed_fiber_angle = np.arctan2(
        tangent_directions @ cross_fiber,
        tangent_directions @ fiber,
    )
    fiber_residual = np.degrees(
        np.arccos(np.clip(np.abs(tangent_directions @ fiber), 0.0, 1.0))
    )
    fiber_std = math.radians(
        max(2.0, _weighted_quantile(fiber_residual, combined, 0.68))
    )

    beta_covariance = inverse * residual_scale**2
    slope_std = math.sqrt(max(float(beta_covariance[1, 1]), float(beta_covariance[2, 2]), 0.0))
    angular_std = max(float(base_normal_std), math.atan(slope_std), math.radians(1.0))
    height_std = max(
        0.45,
        math.sqrt(max(float(beta_covariance[0, 0]), 0.0)),
        residual_scale / math.sqrt(max(effective_support, 1.0)),
    )
    covariance = np.diag([angular_std**2, angular_std**2, height_std**2])
    concentration = float(
        np.clip(
            abs(np.sum(combined * np.exp(2.0j * signed_fiber_angle)))
            / max(weight_sum, 1.0e-8),
            0.0,
            1.0,
        )
    )
    confidence = float(
        np.clip(
            math.sqrt(max(base_normal_confidence, 0.0) * max(concentration, 0.0))
            * min(1.0, effective_support / 8.0)
            * math.sqrt(max(peak_value, 0.0)),
            0.0,
            1.0,
        )
    )
    if confidence < 0.025:
        return None
    owned_depth_radius = (
        0.5
        * settings.cell_stride_voxels
        * float(np.sum(np.abs(fitted_normal)))
    )
    if abs(fitted_height) >= owned_depth_radius * (1.0 - 1.0e-6):
        return None
    estimate = PlaneEstimate(
        tuple(float(value) for value in fitted_normal),
        fitted_height,
        tuple(tuple(float(value) for value in row) for row in covariance),
        tuple(float(value) for value in fiber),
        fiber_std,
        confidence,
    )
    return LayerMode(
        normal_hypothesis,
        estimate,
        target_depth,
        target_orientation,
        float(peak_value),
        float(material_probability),
        float(effective_support),
    )


def _layer_reward(mode: LayerMode) -> float:
    evidence = float(np.clip(mode.evidence_score, 0.0, 1.0))
    material = float(np.clip(mode.material_probability, 0.0, 1.0))
    support = min(1.0, mode.effective_support / 8.0)
    return (
        3.2 * math.sqrt(evidence)
        + 0.45 * material
        + 0.65 * support
        + math.log(max(mode.estimate.confidence, 0.03))
        - 2.0
    )


def _transition_reward(
    first: LayerMode,
    second: LayerMode,
    source: VolumeSource,
    settings: RawAcusSettings,
) -> float | None:
    first_normal = np.asarray(first.estimate.normal_xyz, dtype=np.float64)
    second_normal = np.asarray(second.estimate.normal_xyz, dtype=np.float64)
    second_height = float(second.estimate.height_from_cell_center)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
        second_height = -second_height
    reference = first_normal + second_normal
    reference /= max(float(np.linalg.norm(reference)), 1.0e-8)
    first_denominator = float(np.dot(first_normal, reference))
    second_denominator = float(np.dot(second_normal, reference))
    if min(first_denominator, second_denominator) <= 0.1:
        return None
    center_first = first.estimate.height_from_cell_center / first_denominator
    center_second = second_height / second_denominator
    order_sign = 1.0 if center_second >= center_first else -1.0
    half_cell = settings.cell_stride_voxels * 0.5
    minimum_separation = math.inf
    for signs in product((-1.0, 1.0), repeat=3):
        corner = half_cell * np.asarray(signs, dtype=np.float64)
        tangent = corner - reference * float(np.dot(corner, reference))
        first_position = (
            first.estimate.height_from_cell_center
            - float(np.dot(first_normal, tangent))
        ) / first_denominator
        second_position = (
            second_height - float(np.dot(second_normal, tangent))
        ) / second_denominator
        minimum_separation = min(
            minimum_separation,
            order_sign * (second_position - first_position),
        )
    if minimum_separation <= 0.0:
        return None
    spacing_microns = abs(
        second.estimate.height_from_cell_center
        - first.estimate.height_from_cell_center
    ) * source.voxel_size_microns
    if spacing_microns < settings.minimum_layer_spacing_microns:
        return None
    fiber_angle = math.degrees(
        axial_angle_radians(first.estimate.fiber_xyz, second.estimate.fiber_xyz)  # type: ignore[arg-type]
    )
    thickness_low, thickness_high = settings.plausible_sheet_thickness_microns
    center = 0.5 * (thickness_low + thickness_high)
    width = max(0.5 * (thickness_high - thickness_low), 1.0)
    thickness_affinity = math.exp(-0.5 * ((spacing_microns - center) / width) ** 2)
    orthogonal_affinity = math.exp(
        -0.5
        * ((90.0 - fiber_angle) / settings.orthogonal_ply_std_degrees) ** 2
    )
    # This is a reward, not an alternation constraint. Consecutive parallel
    # modes remain legal for folds, contacts, missing plies, and weak evidence.
    return 0.9 * thickness_affinity * orthogonal_affinity


def enumerate_stratigraphies(
    modes: list[LayerMode],
    source: VolumeSource,
    settings: RawAcusSettings,
    normal_hypothesis: int,
    normal_confidence: float,
    *,
    required_mode: LayerMode | None = None,
) -> list[CellStratigraphy]:
    values = sorted(modes, key=lambda value: value.estimate.height_from_cell_center)
    beams: list[tuple[float, tuple[LayerMode, ...]]] = [(0.0, ())]
    beam_limit = max(64, settings.maximum_configurations_per_cell * 16)
    for mode in values:
        expanded = list(beams)
        reward = _layer_reward(mode)
        for score, layers in beams:
            transition = 0.0
            if layers:
                resolved = _transition_reward(layers[-1], mode, source, settings)
                if resolved is None:
                    continue
                transition = resolved
            expanded.append((score + reward + transition, layers + (mode,)))
        expanded.sort(key=lambda value: (value[0], len(value[1])), reverse=True)
        unique: dict[
            tuple[bool, tuple[int, ...]], tuple[float, tuple[LayerMode, ...]]
        ] = {}
        retained_by_requirement = {False: 0, True: 0}
        for score, layers in expanded:
            contains_required = required_mode is not None and any(
                value is required_mode for value in layers
            )
            key = (
                contains_required,
                tuple(
                    round(layer.estimate.height_from_cell_center * 2.0)
                    for layer in layers
                ),
            )
            if retained_by_requirement[contains_required] >= beam_limit:
                continue
            if key not in unique:
                unique[key] = (score, layers)
                retained_by_requirement[contains_required] += 1
            if required_mode is None and len(unique) >= beam_limit:
                break
            if required_mode is not None and all(
                value >= beam_limit for value in retained_by_requirement.values()
            ):
                break
        beams = list(unique.values())
    normal_term = math.log(max(normal_confidence, 0.03))
    result = [
        CellStratigraphy(normal_hypothesis, score + normal_term, layers)
        for score, layers in beams
        if layers
    ]
    result.sort(key=lambda value: (value.score, len(value.layers)), reverse=True)
    return result


def build_layer_modes(
    needles: NeedleTable,
    evidence: CellEvidenceTable,
    settings: RawAcusSettings,
) -> tuple[LayerModeTable, dict[str, Any]]:
    """Fit every accepted local depth-orientation peak exactly once."""

    started = time.monotonic()
    half_cube = settings.analysis_cube_voxels * 0.5
    modes_by_cell: list[list[LayerMode]] = []
    for cell_index, cell_center in enumerate(evidence.cell_center_source_xyz):
        records = select_cell_needle_indices(
            needles, cell_center, half_cube, settings
        )
        cell_modes: list[LayerMode] = []
        for hypothesis in range(evidence.hypothesis_count):
            if not evidence.normal_valid[cell_index, hypothesis]:
                continue
            normal = evidence.normal_xyz[cell_index, hypothesis]
            density = np.asarray(
                evidence.depth_orientation_density[cell_index, hypothesis],
                dtype=np.float32,
            )
            support = np.asarray(
                evidence.depth_support[cell_index, hypothesis], dtype=np.float32
            )
            peaks = _mode_peaks(
                density,
                support,
                evidence.depth_centers_voxels,
                evidence.orientation_centers_degrees,
                normal,
                settings.cell_stride_voxels,
                settings,
            )
            for depth_index, orientation_index, peak_value in peaks:
                mode = _fit_layer_mode(
                    needles,
                    records,
                    cell_center,
                    normal,
                    float(evidence.normal_angular_std_radians[cell_index, hypothesis]),
                    float(evidence.normal_confidence[cell_index, hypothesis]),
                    float(evidence.depth_centers_voxels[depth_index]),
                    float(evidence.orientation_centers_degrees[orientation_index]),
                    peak_value,
                    float(evidence.ct_material_fraction[cell_index, hypothesis, depth_index]),
                    hypothesis,
                    settings,
                )
                if mode is not None:
                    cell_modes.append(mode)
        modes_by_cell.append(cell_modes)

    mode_offset = np.zeros(evidence.cell_count + 1, dtype=np.uint64)
    for index, values in enumerate(modes_by_cell):
        mode_offset[index + 1] = mode_offset[index] + len(values)
    flattened = [value for values in modes_by_cell for value in values]
    table = LayerModeTable(
        evidence.cell_xyz.copy(),
        mode_offset,
        np.asarray(
            [value.normal_hypothesis for value in flattened], dtype=np.int8
        ),
        np.asarray(
            [value.estimate.normal_xyz for value in flattened], dtype=np.float32
        ).reshape(-1, 3),
        np.asarray(
            [value.estimate.height_from_cell_center for value in flattened],
            dtype=np.float32,
        ),
        np.asarray(
            [_pack_covariance(value.estimate.covariance_matrix) for value in flattened],
            dtype=np.float32,
        ).reshape(-1, 6),
        np.asarray(
            [value.estimate.fiber_xyz for value in flattened], dtype=np.float32
        ).reshape(-1, 3),
        np.asarray(
            [value.estimate.fiber_angular_std_radians for value in flattened],
            dtype=np.float32,
        ),
        np.asarray(
            [value.estimate.confidence for value in flattened], dtype=np.float32
        ),
        np.asarray(
            [value.source_depth_voxels for value in flattened], dtype=np.float32
        ),
        np.asarray(
            [value.source_orientation_degrees for value in flattened],
            dtype=np.float32,
        ),
        np.asarray(
            [value.evidence_score for value in flattened], dtype=np.float32
        ),
        np.asarray(
            [value.material_probability for value in flattened], dtype=np.float32
        ),
        np.asarray(
            [value.effective_support for value in flattened], dtype=np.float32
        ),
    )
    table.validate()
    counts = np.diff(table.mode_offset.astype(np.int64))
    return table, {
        "cellCount": table.cell_count,
        "candidateModeCount": table.mode_count,
        "cellsWithModes": int(np.count_nonzero(counts)),
        "maximumModesPerCell": int(np.max(counts, initial=0)),
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }


def build_configurations_from_modes(
    source: VolumeSource,
    modes: LayerModeTable,
    evidence: CellEvidenceTable,
    settings: RawAcusSettings,
) -> tuple[ConfigurationTable, dict[str, Any]]:
    """Compress a persistent mode bank into top-M physical stratigraphies."""

    started = time.monotonic()
    modes.validate()
    evidence.validate()
    if not np.array_equal(modes.cell_xyz, evidence.cell_xyz):
        raise ValueError("mode and evidence tables must contain the same ordered cells")
    all_configurations: list[list[CellStratigraphy]] = []
    for cell_index in range(evidence.cell_count):
        candidates: list[CellStratigraphy] = []
        cell_modes = modes.modes_for_cell(cell_index)
        for hypothesis in range(evidence.hypothesis_count):
            hypothesis_modes = [
                value
                for value in cell_modes
                if value.normal_hypothesis == hypothesis
            ]
            if not hypothesis_modes:
                continue
            candidates.extend(
                enumerate_stratigraphies(
                    hypothesis_modes,
                    source,
                    settings,
                    hypothesis,
                    float(evidence.normal_confidence[cell_index, hypothesis]),
                )
            )

        candidates.sort(key=lambda value: (value.score, len(value.layers)), reverse=True)
        retained = candidates[: max(0, settings.maximum_configurations_per_cell - 1)]
        empty_score = -0.35 * math.log1p(
            float(np.max(evidence.total_needle_weight[cell_index], initial=0.0))
        )
        retained.append(CellStratigraphy(-1, empty_score, ()))
        retained.sort(key=lambda value: (value.score, len(value.layers)), reverse=True)
        scores = np.asarray([value.score for value in retained], dtype=np.float64)
        maximum = float(np.max(scores))
        log_normalizer = maximum + math.log(float(np.sum(np.exp(scores - maximum))))
        all_configurations.append(
            [
                CellStratigraphy(value.normal_hypothesis, value.score - log_normalizer, value.layers)
                for value in retained
            ]
        )

    configuration_offset = np.zeros(evidence.cell_count + 1, dtype=np.uint64)
    for index, values in enumerate(all_configurations):
        configuration_offset[index + 1] = configuration_offset[index] + len(values)
    flattened = [value for values in all_configurations for value in values]
    layer_offset = np.zeros(len(flattened) + 1, dtype=np.uint64)
    for index, value in enumerate(flattened):
        layer_offset[index + 1] = layer_offset[index] + len(value.layers)
    layers = [layer for value in flattened for layer in value.layers]
    configuration_id = np.concatenate(
        [np.arange(len(values), dtype=np.uint16) for values in all_configurations]
    )
    table = ConfigurationTable(
        evidence.cell_xyz.copy(),
        configuration_offset,
        configuration_id,
        np.asarray([value.score for value in flattened], dtype=np.float32),
        np.asarray([value.normal_hypothesis for value in flattened], dtype=np.int8),
        layer_offset,
        np.asarray([layer.estimate.normal_xyz for layer in layers], dtype=np.float32).reshape(-1, 3),
        np.asarray([layer.estimate.height_from_cell_center for layer in layers], dtype=np.float32),
        np.asarray(
            [_pack_covariance(layer.estimate.covariance_matrix) for layer in layers],
            dtype=np.float32,
        ).reshape(-1, 6),
        np.asarray([layer.estimate.fiber_xyz for layer in layers], dtype=np.float32).reshape(-1, 3),
        np.asarray([layer.estimate.fiber_angular_std_radians for layer in layers], dtype=np.float32),
        np.asarray([layer.estimate.confidence for layer in layers], dtype=np.float32),
        np.asarray([layer.evidence_score for layer in layers], dtype=np.float32),
        np.asarray([layer.material_probability for layer in layers], dtype=np.float32),
        np.asarray([layer.effective_support for layer in layers], dtype=np.float32),
    )
    table.validate()
    layer_counts = np.diff(table.layer_offset.astype(np.int64))
    return table, {
        "cellCount": table.cell_count,
        "candidateModeCount": modes.mode_count,
        "configurationCount": table.configuration_count,
        "layerAlternativeCount": table.layer_count,
        "nonemptyConfigurationCount": int(np.count_nonzero(layer_counts)),
        "maximumLayersInConfiguration": int(np.max(layer_counts, initial=0)),
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }


def build_stratigraphies(
    source: VolumeSource,
    shard: ShardSpec,
    needles: NeedleTable,
    evidence: CellEvidenceTable,
    settings: RawAcusSettings,
) -> tuple[ConfigurationTable, dict[str, Any]]:
    """Compatibility wrapper for callers that do not persist the mode bank."""

    del shard
    started = time.monotonic()
    modes, _ = build_layer_modes(needles, evidence, settings)
    table, statistics = build_configurations_from_modes(
        source, modes, evidence, settings
    )
    statistics["elapsedSeconds"] = round(time.monotonic() - started, 6)
    return table, statistics


# Kept for downstream experiments written against the initial private helper.
_top_stratigraphies = enumerate_stratigraphies


def write_mode_artifact(
    prefix: str | Path,
    table: LayerModeTable,
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
        "schema": MODE_ARTIFACT_SCHEMA,
        "version": MODE_ARTIFACT_VERSION,
        "identitySha256": identity_sha256,
        "shard": shard.record(),
        "statistics": dict(statistics),
        "model": {
            "scope": "all fitted local Acus peaks before configuration pruning",
            "directions": "normal and fiber are axial/unsigned",
            "ownership": "modes belong to disjoint cubical cells",
            "purpose": "persistent evidence bank for stratigraphy and contextual continuation",
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(manifest_path, payload)
    return payload


def read_mode_artifact(
    prefix: str | Path,
    *,
    identity_sha256: str,
    verify: bool = True,
) -> LayerModeTable:
    base = Path(prefix)
    manifest = json.loads(base.with_suffix(".json").read_text())
    data_path = base.with_suffix(".npz")
    if (
        manifest.get("schema") != MODE_ARTIFACT_SCHEMA
        or int(manifest.get("version", -1)) != MODE_ARTIFACT_VERSION
        or manifest.get("identitySha256") != identity_sha256
    ):
        raise ValueError("layer-mode artifact does not match this inference identity")
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("layer-mode artifact content hash mismatch")
    with np.load(data_path) as values:
        table = LayerModeTable(
            np.asarray(values["cellXYZ"], dtype=np.int32),
            np.asarray(values["modeOffset"], dtype=np.uint64),
            np.asarray(values["normalHypothesis"], dtype=np.int8),
            np.asarray(values["normalXYZ"], dtype=np.float32),
            np.asarray(values["height"], dtype=np.float32),
            np.asarray(values["covariance"], dtype=np.float32),
            np.asarray(values["fiberXYZ"], dtype=np.float32),
            np.asarray(values["fiberAngularStdRadians"], dtype=np.float32),
            np.asarray(values["confidence"], dtype=np.float32),
            np.asarray(values["sourceDepthVoxels"], dtype=np.float32),
            np.asarray(values["sourceOrientationDegrees"], dtype=np.float32),
            np.asarray(values["evidenceScore"], dtype=np.float32),
            np.asarray(values["materialProbability"], dtype=np.float32),
            np.asarray(values["effectiveSupport"], dtype=np.float32),
        )
    table.validate()
    return table


def write_configuration_artifact(
    prefix: str | Path,
    table: ConfigurationTable,
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
        "schema": CONFIGURATION_ARTIFACT_SCHEMA,
        "version": CONFIGURATION_ARTIFACT_VERSION,
        "identitySha256": identity_sha256,
        "shard": shard.record(),
        "statistics": dict(statistics),
        "model": {
            "normal": "one of the cell's raw unsigned Acus normal hypotheses",
            "layers": "ordered subsets of fitted depth-orientation modes",
            "physicalPrior": "minimum spacing plus a soft, non-mandatory orthogonal-ply reward",
            "sameFiberSuccessors": "legal",
            "withinCellOrdering": "successive fitted planes may not cross inside the owning cube",
            "emptyConfiguration": "always retained",
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(manifest_path, payload)
    return payload


def read_configuration_artifact(
    prefix: str | Path,
    *,
    identity_sha256: str,
    verify: bool = True,
) -> ConfigurationTable:
    base = Path(prefix)
    manifest = json.loads(base.with_suffix(".json").read_text())
    data_path = base.with_suffix(".npz")
    if (
        manifest.get("schema") != CONFIGURATION_ARTIFACT_SCHEMA
        or int(manifest.get("version", -1)) != CONFIGURATION_ARTIFACT_VERSION
        or manifest.get("identitySha256") != identity_sha256
    ):
        raise ValueError("stratigraphy artifact does not match this raw Acus pipeline")
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("stratigraphy artifact content hash mismatch")
    with np.load(data_path) as values:
        table = ConfigurationTable(
            np.asarray(values["cellXYZ"], dtype=np.int32),
            np.asarray(values["configurationOffset"], dtype=np.uint64),
            np.asarray(values["configurationId"], dtype=np.uint16),
            np.asarray(values["configurationLogWeight"], dtype=np.float32),
            np.asarray(values["normalHypothesis"], dtype=np.int8),
            np.asarray(values["layerOffset"], dtype=np.uint64),
            np.asarray(values["layerNormalXYZ"], dtype=np.float32),
            np.asarray(values["layerHeight"], dtype=np.float32),
            np.asarray(values["layerCovariance"], dtype=np.float32),
            np.asarray(values["layerFiberXYZ"], dtype=np.float32),
            np.asarray(values["layerFiberAngularStdRadians"], dtype=np.float32),
            np.asarray(values["layerConfidence"], dtype=np.float32),
            np.asarray(values["layerEvidenceScore"], dtype=np.float32),
            np.asarray(values["layerMaterialProbability"], dtype=np.float32),
            np.asarray(values["layerEffectiveSupport"], dtype=np.float32),
        )
    table.validate()
    return table
