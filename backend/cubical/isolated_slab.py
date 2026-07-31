from __future__ import annotations

import colorsys
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.rectify import _trilinear, gaussian_blur_3d

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .export import rgb_png


ISOLATED_SLAB_SCHEMA = "pareidolia.dense-isolated-slabs"
ISOLATED_SLAB_VERSION = 1
ISOLATED_SLAB_STEM = "isolated-slabs-v1"


@dataclass(frozen=True, slots=True)
class IsolatedSlabSettings:
    """Physical and dimensionless controls for dense isolated-sheet seeding.

    The detector deliberately makes no use of Acus needles.  It looks for a
    material interval bounded by two opposing interfaces with independently
    clear air beyond both faces.  Low-confidence pairs remain in the artifact;
    ``minimum_seed_confidence`` only controls the conservative connected seed
    patches and can therefore be changed downstream without re-reading CT.
    """

    sampling_stride_voxels: int = 2
    smoothing_sigma_voxels: float = 2.0
    minimum_sheet_thickness_microns: float = 80.0
    maximum_sheet_thickness_microns: float = 400.0
    minimum_air_clearance_microns: float = 50.0
    material_threshold_raw: float | None = None
    minimum_boundary_gradient_class_fraction: float = 0.25
    minimum_profile_margin_class_fraction: float = 0.03
    full_confidence_profile_margin_class_fraction: float = 0.35
    minimum_air_sample_fraction: float = 0.8
    maximum_opposing_normal_degrees: float = 35.0
    minimum_seed_confidence: float = 0.5
    component_link_radius_sampling_steps: float = math.sqrt(5.0)
    component_maximum_normal_degrees: float = 30.0
    component_maximum_height_sampling_steps: float = 1.15
    component_maximum_relative_thickness_difference: float = 0.35
    component_minimum_thickness_tolerance_sampling_steps: float = 2.0
    profile_batch_size: int = 65_536
    maximum_preview_components: int = 96

    def __post_init__(self) -> None:
        if self.sampling_stride_voxels < 1:
            raise ValueError("sampling stride must be positive")
        if self.profile_batch_size < 1 or self.maximum_preview_components < 1:
            raise ValueError("batch and preview sizes must be positive")
        positive = (
            self.smoothing_sigma_voxels,
            self.minimum_sheet_thickness_microns,
            self.maximum_sheet_thickness_microns,
            self.minimum_air_clearance_microns,
            self.minimum_boundary_gradient_class_fraction,
            self.minimum_profile_margin_class_fraction,
            self.full_confidence_profile_margin_class_fraction,
            self.component_link_radius_sampling_steps,
            self.component_maximum_height_sampling_steps,
            self.component_maximum_relative_thickness_difference,
            self.component_minimum_thickness_tolerance_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("isolated-slab scales must be finite and positive")
        if self.maximum_sheet_thickness_microns <= self.minimum_sheet_thickness_microns:
            raise ValueError("maximum sheet thickness must exceed the minimum")
        if (
            self.full_confidence_profile_margin_class_fraction
            <= self.minimum_profile_margin_class_fraction
        ):
            raise ValueError("full-confidence profile margin must exceed the minimum")
        if not 0.5 <= self.minimum_air_sample_fraction <= 1.0:
            raise ValueError("minimum air sample fraction must lie in [0.5, 1]")
        if not 0.0 <= self.minimum_seed_confidence <= 1.0:
            raise ValueError("minimum seed confidence must lie in [0, 1]")
        for value, name in (
            (self.maximum_opposing_normal_degrees, "opposing-normal angle"),
            (self.component_maximum_normal_degrees, "component-normal angle"),
        ):
            if not 0.0 < value < 90.0:
                raise ValueError(f"{name} must lie strictly between 0 and 90 degrees")
        if self.material_threshold_raw is not None and (
            not math.isfinite(self.material_threshold_raw)
            or not 0.0 < self.material_threshold_raw < 255.0
        ):
            raise ValueError("explicit material threshold must lie in (0, 255)")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class IsolatedSlabTable:
    midpoint_xyz: np.ndarray
    normal_xyz: np.ndarray
    boundary_first_xyz: np.ndarray
    boundary_second_xyz: np.ndarray
    thickness_voxels: np.ndarray
    confidence: np.ndarray
    air_margin_fraction: np.ndarray
    material_margin_fraction: np.ndarray
    opposing_normal_cosine: np.ndarray
    component_id: np.ndarray

    @property
    def count(self) -> int:
        return int(len(self.confidence))

    def validate(self) -> None:
        count = self.count
        shapes = {
            "midpoint_xyz": (count, 3),
            "normal_xyz": (count, 3),
            "boundary_first_xyz": (count, 3),
            "boundary_second_xyz": (count, 3),
            "thickness_voxels": (count,),
            "confidence": (count,),
            "air_margin_fraction": (count,),
            "material_margin_fraction": (count,),
            "opposing_normal_cosine": (count,),
            "component_id": (count,),
        }
        for name, shape in shapes.items():
            value = np.asarray(getattr(self, name))
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if name != "component_id" and np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if count:
            length = np.linalg.norm(self.normal_xyz, axis=1)
            if np.any(np.abs(length - 1.0) > 2.0e-4):
                raise ValueError("isolated-slab normals must be unit vectors")
            if np.any(self.thickness_voxels <= 0.0):
                raise ValueError("isolated-slab thickness must be positive")
            if np.any((self.confidence < 0.0) | (self.confidence > 1.0)):
                raise ValueError("isolated-slab confidence must lie in [0, 1]")

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "midpointXYZ": np.asarray(self.midpoint_xyz, dtype=np.float32),
            "normalXYZ": np.asarray(self.normal_xyz, dtype=np.float32),
            "boundaryFirstXYZ": np.asarray(
                self.boundary_first_xyz, dtype=np.float32
            ),
            "boundarySecondXYZ": np.asarray(
                self.boundary_second_xyz, dtype=np.float32
            ),
            "thicknessVoxels": np.asarray(
                self.thickness_voxels, dtype=np.float32
            ),
            "confidence": np.asarray(self.confidence, dtype=np.float32),
            "airMarginClassFraction": np.asarray(
                self.air_margin_fraction, dtype=np.float32
            ),
            "materialMarginClassFraction": np.asarray(
                self.material_margin_fraction, dtype=np.float32
            ),
            "opposingNormalCosine": np.asarray(
                self.opposing_normal_cosine, dtype=np.float32
            ),
            "componentId": np.asarray(self.component_id, dtype=np.int32),
        }


def otsu_material_calibration(values: np.ndarray) -> dict[str, float]:
    """Return a deterministic two-class calibration for one raw uint8 block."""

    sample = np.asarray(values)
    if not sample.size:
        raise ValueError("cannot calibrate an empty CT sample")
    rounded = np.clip(np.rint(sample), 0, 255).astype(np.uint8, copy=False)
    histogram = np.bincount(rounded.reshape(-1), minlength=256).astype(np.float64)
    probability = histogram / np.maximum(np.sum(histogram), 1.0)
    bins = np.arange(256, dtype=np.float64)
    weight = np.cumsum(probability)
    moment = np.cumsum(probability * bins)
    denominator = weight * (1.0 - weight)
    between = np.zeros(256, dtype=np.float64)
    valid = denominator > 1.0e-12
    between[valid] = (
        (moment[-1] * weight[valid] - moment[valid]) ** 2
        / denominator[valid]
    )
    threshold = int(np.argmax(between[:-1]))
    air_count = np.sum(histogram[: threshold + 1])
    material_count = np.sum(histogram[threshold + 1 :])
    if air_count <= 0.0 or material_count <= 0.0:
        raise ValueError("CT intensity histogram cannot be split into air and material")
    air_mean = float(
        np.sum(bins[: threshold + 1] * histogram[: threshold + 1]) / air_count
    )
    material_mean = float(
        np.sum(bins[threshold + 1 :] * histogram[threshold + 1 :])
        / material_count
    )
    contrast = material_mean - air_mean
    if contrast <= 1.0e-6:
        raise ValueError("air/material calibration has no positive class contrast")
    return {
        "materialThresholdRaw": float(threshold),
        "airMeanRaw": air_mean,
        "materialMeanRaw": material_mean,
        "classContrastRaw": contrast,
        "method": "otsu-on-smoothed-owned-block",
    }


def _downsample_mean_zyx(volume: np.ndarray, stride: int) -> np.ndarray:
    """Block-average a memmapped ZYX array while bounding transient memory."""

    usable = tuple((int(size) // stride) * stride for size in volume.shape)
    if any(size < stride for size in usable):
        raise ValueError("processing volume is smaller than one sampling cell")
    output_shape = tuple(size // stride for size in usable)
    output = np.empty(output_shape, dtype=np.float32)
    for output_z in range(output_shape[0]):
        source_z = output_z * stride
        slab = np.asarray(
            volume[
                source_z : source_z + stride,
                : usable[1],
                : usable[2],
            ],
            dtype=np.float32,
        )
        output[output_z] = slab.reshape(
            stride,
            output_shape[1],
            stride,
            output_shape[2],
            stride,
        ).mean(axis=(0, 2, 4))
    return output


def _owned_sample_slices(
    owned: VoxelBounds, processing: VoxelBounds, stride: int
) -> tuple[slice, slice, slice]:
    start = tuple(
        max(0, int(math.floor((owned.start_xyz[axis] - processing.start_xyz[axis]) / stride)))
        for axis in range(3)
    )
    stop = tuple(
        int(math.ceil((owned.stop_xyz_exclusive[axis] - processing.start_xyz[axis]) / stride))
        for axis in range(3)
    )
    return tuple(slice(start[axis], stop[axis]) for axis in (2, 1, 0))  # type: ignore[return-value]


def _candidate_boundary_field(
    smoothed: np.ndarray,
    threshold: float,
    class_contrast: float,
    settings: IsolatedSlabSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    gradient_z, gradient_y, gradient_x = np.gradient(smoothed)
    gradient_magnitude = np.sqrt(
        gradient_x * gradient_x
        + gradient_y * gradient_y
        + gradient_z * gradient_z
    )
    sigma_steps = max(
        0.55,
        settings.smoothing_sigma_voxels / settings.sampling_stride_voxels,
    )
    material = smoothed >= threshold
    boundary = material & (
        (~np.roll(material, 1, axis=0))
        | (~np.roll(material, -1, axis=0))
        | (~np.roll(material, 1, axis=1))
        | (~np.roll(material, -1, axis=1))
        | (~np.roll(material, 1, axis=2))
        | (~np.roll(material, -1, axis=2))
    )
    boundary[[0, -1], :, :] = False
    boundary[:, [0, -1], :] = False
    boundary[:, :, [0, -1]] = False
    boundary &= (
        gradient_magnitude * sigma_steps
        >= settings.minimum_boundary_gradient_class_fraction * class_contrast
    )
    return boundary, gradient_x, gradient_y, gradient_z


def _crossing_position(
    first_value: np.ndarray,
    second_value: np.ndarray,
    first_distance: np.ndarray,
    threshold: float,
) -> np.ndarray:
    denominator = second_value - first_value
    fraction = np.divide(
        threshold - first_value,
        denominator,
        out=np.full_like(first_value, 0.5, dtype=np.float32),
        where=np.abs(denominator) > 1.0e-6,
    )
    return first_distance + np.clip(fraction, 0.0, 1.0)


def detect_isolated_slab_pairs(
    smoothed: np.ndarray,
    *,
    threshold: float,
    class_contrast: float,
    voxel_size_microns: float,
    settings: IsolatedSlabSettings,
    retain_physical_profiles: bool = False,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Pair opposing air/material interfaces on one downsampled CT field.

    Coordinates returned here are in downsampled array-index XYZ space.  The
    caller is responsible for converting them into immutable source/world
    coordinates after de-duplication and ownership cropping.
    """

    boundary, gradient_x, gradient_y, gradient_z = _candidate_boundary_field(
        smoothed, threshold, class_contrast, settings
    )
    candidate_zyx = np.column_stack(np.nonzero(boundary))
    candidate_xyz = candidate_zyx[:, ::-1].astype(np.float32)
    normal = np.column_stack(
        (gradient_x[boundary], gradient_y[boundary], gradient_z[boundary])
    ).astype(np.float32)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-6)

    step_microns = voxel_size_microns * settings.sampling_stride_voxels
    minimum_thickness_steps = settings.minimum_sheet_thickness_microns / step_microns
    maximum_thickness_steps = settings.maximum_sheet_thickness_microns / step_microns
    clearance_steps = max(
        2, int(math.ceil(settings.minimum_air_clearance_microns / step_microns))
    )
    minimum_exit = max(1, int(math.floor(minimum_thickness_steps)) - 1)
    maximum_exit = int(math.ceil(maximum_thickness_steps)) + 2
    profile_distances = np.arange(
        -clearance_steps,
        maximum_exit + clearance_steps + 3,
        dtype=np.float32,
    )
    zero_index = clearance_steps
    opposing_limit = math.cos(math.radians(settings.maximum_opposing_normal_degrees))

    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "midpoint",
            "normal",
            "boundary_first",
            "boundary_second",
            "thickness_steps",
            "confidence",
            "air_margin",
            "air_sample_fraction",
            "material_margin",
            "opposing_cosine",
            "conservative",
        )
    }
    counts = {
        "boundaryCandidateCount": int(len(candidate_xyz)),
        "hasEntryAndExitCrossingCount": 0,
        "physicalThicknessCount": 0,
        "clearAirCount": 0,
        "materialInteriorCount": 0,
        "opposingInterfaceCount": 0,
        "pairedProfileCountBeforeDeduplication": 0,
    }

    for low in range(0, len(candidate_xyz), settings.profile_batch_size):
        point = candidate_xyz[low : low + settings.profile_batch_size]
        axis = normal[low : low + settings.profile_batch_size]
        profile_points = (
            point[:, None, :]
            + axis[:, None, :] * profile_distances[None, :, None]
        )
        profile = _trilinear(smoothed, profile_points, outside=255.0)

        entry_pairs = (
            (profile[:, :zero_index] < threshold)
            & (profile[:, 1 : zero_index + 1] >= threshold)
        )
        has_entry = np.any(entry_pairs, axis=1)
        reverse_entry = np.argmax(entry_pairs[:, ::-1], axis=1)
        entry_index = zero_index - 1 - reverse_entry

        high_start = zero_index + minimum_exit - 1
        high_stop = zero_index + maximum_exit
        exit_pairs = (
            (profile[:, high_start:high_stop] >= threshold)
            & (profile[:, high_start + 1 : high_stop + 1] < threshold)
            & (profile[:, high_start + 2 : high_stop + 2] < threshold)
        )
        has_exit = np.any(exit_pairs, axis=1)
        exit_index = np.argmax(exit_pairs, axis=1) + high_start
        has_crossings = has_entry & has_exit
        counts["hasEntryAndExitCrossingCount"] += int(np.count_nonzero(has_crossings))

        row = np.arange(len(point), dtype=np.int64)
        entry_first = profile[row, entry_index]
        entry_second = profile[row, entry_index + 1]
        entry_distance = _crossing_position(
            entry_first,
            entry_second,
            profile_distances[entry_index],
            threshold,
        )
        exit_first = profile[row, exit_index]
        exit_second = profile[row, exit_index + 1]
        exit_distance = _crossing_position(
            exit_first,
            exit_second,
            profile_distances[exit_index],
            threshold,
        )
        thickness_steps = exit_distance - entry_distance
        physical = (
            has_crossings
            & (thickness_steps >= minimum_thickness_steps)
            & (thickness_steps <= maximum_thickness_steps)
        )
        counts["physicalThicknessCount"] += int(np.count_nonzero(physical))

        air_offsets = np.arange(1, clearance_steps + 1, dtype=np.float32)
        interior_fraction = np.asarray((0.25, 0.5, 0.75), dtype=np.float32)
        query_distance = np.concatenate(
            (
                entry_distance[:, None] - air_offsets[None, :],
                entry_distance[:, None]
                + thickness_steps[:, None] * interior_fraction[None, :],
                exit_distance[:, None] + air_offsets[None, :],
            ),
            axis=1,
        )
        query_points = point[:, None, :] + axis[:, None, :] * query_distance[:, :, None]
        query = _trilinear(smoothed, query_points, outside=255.0)
        first_air = query[:, :clearance_steps]
        interior = query[:, clearance_steps : clearance_steps + 3]
        second_air = query[:, clearance_steps + 3 :]
        first_air_mean = np.mean(first_air, axis=1)
        second_air_mean = np.mean(second_air, axis=1)
        interior_mean = np.mean(interior, axis=1)
        air_fraction = np.minimum(
            np.mean(first_air < threshold, axis=1),
            np.mean(second_air < threshold, axis=1),
        )
        air_margin = (
            threshold - np.maximum(first_air_mean, second_air_mean)
        ) / class_contrast
        material_margin = (interior_mean - threshold) / class_contrast
        clear_air = (
            physical
            & (air_fraction >= settings.minimum_air_sample_fraction)
            & (air_margin >= settings.minimum_profile_margin_class_fraction)
        )
        material_inside = (
            clear_air
            & (material_margin >= settings.minimum_profile_margin_class_fraction)
        )
        counts["clearAirCount"] += int(np.count_nonzero(clear_air))
        counts["materialInteriorCount"] += int(np.count_nonzero(material_inside))

        exit_point = point + axis * exit_distance[:, None]
        exit_gradient = np.column_stack(
            (
                _trilinear(gradient_x, exit_point),
                _trilinear(gradient_y, exit_point),
                _trilinear(gradient_z, exit_point),
            )
        )
        exit_gradient_length = np.linalg.norm(exit_gradient, axis=1)
        exit_gradient /= np.maximum(exit_gradient_length[:, None], 1.0e-6)
        opposing_cosine = -np.einsum("ij,ij->i", axis, exit_gradient)
        opposing = material_inside & (opposing_cosine >= opposing_limit)
        counts["opposingInterfaceCount"] += int(np.count_nonzero(opposing))

        profile_margin = np.minimum(air_margin, material_margin)
        profile_score = np.clip(
            (
                profile_margin
                - settings.minimum_profile_margin_class_fraction
            )
            / (
                settings.full_confidence_profile_margin_class_fraction
                - settings.minimum_profile_margin_class_fraction
            ),
            0.0,
            1.0,
        )
        opposing_score = np.clip(
            (opposing_cosine - opposing_limit) / (1.0 - opposing_limit),
            0.0,
            1.0,
        )
        confidence = profile_score * (0.5 + 0.5 * opposing_score)
        selected = np.flatnonzero(
            physical if retain_physical_profiles else opposing
        )
        counts["pairedProfileCountBeforeDeduplication"] += int(len(selected))
        if not len(selected):
            continue
        first = point[selected] + axis[selected] * entry_distance[selected, None]
        second = point[selected] + axis[selected] * exit_distance[selected, None]
        result["midpoint"].append(0.5 * (first + second))
        result["normal"].append(axis[selected])
        result["boundary_first"].append(first)
        result["boundary_second"].append(second)
        result["thickness_steps"].append(thickness_steps[selected])
        result["confidence"].append(confidence[selected])
        result["air_margin"].append(air_margin[selected])
        result["air_sample_fraction"].append(air_fraction[selected])
        result["material_margin"].append(material_margin[selected])
        result["opposing_cosine"].append(opposing_cosine[selected])
        result["conservative"].append(opposing[selected].astype(np.uint8))

    arrays = {
        name: (
            np.concatenate(values).astype(
                np.uint8 if name == "conservative" else np.float32,
                copy=False,
            )
            if values
            else np.empty(
                (
                    (0, 3)
                    if name
                    in {
                        "midpoint",
                        "normal",
                        "boundary_first",
                        "boundary_second",
                    }
                    else (0,)
                ),
                dtype=(np.uint8 if name == "conservative" else np.float32),
            )
        )
        for name, values in result.items()
    }
    return counts, arrays


def _deduplicate_and_crop_pairs(
    arrays: Mapping[str, np.ndarray],
    *,
    processing: VoxelBounds,
    owned: VoxelBounds,
    source_origin_xyz: tuple[int, int, int],
    settings: IsolatedSlabSettings,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    stride = settings.sampling_stride_voxels
    midpoint = np.asarray(arrays["midpoint"], dtype=np.float32)
    if not len(midpoint):
        empty = {name: np.asarray(value) for name, value in arrays.items()}
        return empty, np.empty((0, 3), dtype=np.int32)
    key = np.rint(midpoint).astype(np.int32)
    shape_xyz = tuple(
        (processing.shape_xyz[axis] // stride) for axis in range(3)
    )
    clipped = np.column_stack(
        tuple(
            np.clip(key[:, axis], 0, shape_xyz[axis] - 1)
            for axis in range(3)
        )
    )
    flat = np.ravel_multi_index(
        (clipped[:, 2], clipped[:, 1], clipped[:, 0]),
        shape_xyz[::-1],
    )
    confidence = np.asarray(arrays["confidence"])
    order = np.lexsort((-confidence, flat))
    sorted_flat = flat[order]
    keep = np.concatenate(
        (np.ones(1, dtype=bool), sorted_flat[1:] != sorted_flat[:-1])
    )
    selected = order[keep]

    half = 0.5 * (stride - 1)
    local_offset = np.asarray(processing.start_xyz, dtype=np.float32) + half
    source_shift = np.asarray(source_origin_xyz, dtype=np.float32)

    def world(points: np.ndarray) -> np.ndarray:
        return (
            np.asarray(points, dtype=np.float32) * stride
            + local_offset[None, :]
            + source_shift[None, :]
        )

    midpoint_world = world(midpoint[selected])
    owned_world_start = (
        np.asarray(owned.start_xyz, dtype=np.float32) + source_shift
    )
    owned_world_stop = (
        np.asarray(owned.stop_xyz_exclusive, dtype=np.float32) + source_shift
    )
    owned_mask = np.all(
        (midpoint_world >= owned_world_start[None, :])
        & (midpoint_world < owned_world_stop[None, :]),
        axis=1,
    )
    selected = selected[owned_mask]
    midpoint_world = midpoint_world[owned_mask]
    output = {
        "midpoint": midpoint_world,
        "normal": np.asarray(arrays["normal"])[selected],
        "boundary_first": world(np.asarray(arrays["boundary_first"])[selected]),
        "boundary_second": world(np.asarray(arrays["boundary_second"])[selected]),
        "thickness_steps": np.asarray(arrays["thickness_steps"])[selected],
        "confidence": np.asarray(arrays["confidence"])[selected],
        "air_margin": np.asarray(arrays["air_margin"])[selected],
        "material_margin": np.asarray(arrays["material_margin"])[selected],
        "opposing_cosine": np.asarray(arrays["opposing_cosine"])[selected],
    }
    return output, midpoint[selected]


def connect_isolated_slab_seeds(
    midpoint_sampling_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    thickness_steps: np.ndarray,
    confidence: np.ndarray,
    *,
    processing_shape_xyz: tuple[int, int, int],
    settings: IsolatedSlabSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Connect only locally coplanar high-confidence slab pairs.

    This graph is descriptive rather than reconstructive: it does not bridge a
    missing CT profile, choose among ambiguous layers, or alter any interface.
    """

    count = len(confidence)
    component_id = np.full(count, -1, dtype=np.int32)
    seed_index = np.flatnonzero(confidence >= settings.minimum_seed_confidence)
    if not len(seed_index):
        return component_id, {
            "seedCount": 0,
            "edgeCount": 0,
            "componentCount": 0,
            "componentSize": _percentile_record(np.empty(0)),
            "largestComponentSizes": [],
        }
    point = np.asarray(midpoint_sampling_xyz[seed_index], dtype=np.float64)
    normal = np.asarray(normal_xyz[seed_index], dtype=np.float64)
    thickness = np.asarray(thickness_steps[seed_index], dtype=np.float64)
    key = np.rint(point).astype(np.int32)
    shape = processing_shape_xyz[::-1]
    grid = np.full(shape, -1, dtype=np.int32)
    grid[key[:, 2], key[:, 1], key[:, 0]] = np.arange(
        len(seed_index), dtype=np.int32
    )

    radius = settings.component_link_radius_sampling_steps
    reach = int(math.ceil(radius))
    edge_first: list[np.ndarray] = []
    edge_second: list[np.ndarray] = []
    normal_limit = math.cos(
        math.radians(settings.component_maximum_normal_degrees)
    )
    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                if dx * dx + dy * dy + dz * dz > radius * radius + 1.0e-9:
                    continue
                valid = (
                    (key[:, 0] + dx >= 0)
                    & (key[:, 0] + dx < processing_shape_xyz[0])
                    & (key[:, 1] + dy >= 0)
                    & (key[:, 1] + dy < processing_shape_xyz[1])
                    & (key[:, 2] + dz >= 0)
                    & (key[:, 2] + dz < processing_shape_xyz[2])
                )
                first = np.flatnonzero(valid)
                first_key = key[first]
                second = grid[
                    first_key[:, 2] + dz,
                    first_key[:, 1] + dy,
                    first_key[:, 0] + dx,
                ]
                exists = second >= 0
                first = first[exists]
                second = second[exists]
                if not len(first):
                    continue
                dot = np.einsum("ij,ij->i", normal[first], normal[second])
                sign = np.where(dot >= 0.0, 1.0, -1.0)
                average_normal = normal[first] + sign[:, None] * normal[second]
                average_normal /= np.maximum(
                    np.linalg.norm(average_normal, axis=1, keepdims=True), 1.0e-9
                )
                displacement = point[second] - point[first]
                height = np.abs(
                    np.einsum("ij,ij->i", displacement, average_normal)
                )
                thickness_difference = np.abs(
                    thickness[first] - thickness[second]
                )
                thickness_tolerance = np.maximum(
                    settings.component_minimum_thickness_tolerance_sampling_steps,
                    settings.component_maximum_relative_thickness_difference
                    * 0.5
                    * (thickness[first] + thickness[second]),
                )
                accepted = (
                    (np.abs(dot) >= normal_limit)
                    & (
                        height
                        <= settings.component_maximum_height_sampling_steps
                    )
                    & (thickness_difference <= thickness_tolerance)
                )
                edge_first.append(first[accepted])
                edge_second.append(second[accepted])

    if edge_first:
        first = np.concatenate(edge_first)
        second = np.concatenate(edge_second)
    else:
        first = np.empty(0, dtype=np.int32)
        second = np.empty(0, dtype=np.int32)
    labels = np.arange(len(seed_index), dtype=np.int32)
    iteration_count = 0
    for iteration_count in range(1, 129):
        prior = labels.copy()
        if len(first):
            minimum = np.minimum(labels[first], labels[second])
            np.minimum.at(labels, first, minimum)
            np.minimum.at(labels, second, minimum)
        for _ in range(8):
            labels = labels[labels]
        if np.array_equal(labels, prior):
            break
    else:
        raise RuntimeError("isolated-slab component labeling did not converge")
    root, size = np.unique(labels, return_counts=True)
    rank_order = np.lexsort((root, -size))
    ranked_root = root[rank_order]
    ranked_size = size[rank_order]
    root_to_component = np.full(len(seed_index), -1, dtype=np.int32)
    root_to_component[ranked_root] = np.arange(len(ranked_root), dtype=np.int32)
    component_id[seed_index] = root_to_component[labels]
    return component_id, {
        "seedCount": int(len(seed_index)),
        "edgeCount": int(len(first)),
        "componentCount": int(len(ranked_size)),
        "componentsAtLeast8Seeds": int(np.count_nonzero(ranked_size >= 8)),
        "componentsAtLeast32Seeds": int(np.count_nonzero(ranked_size >= 32)),
        "labelPropagationIterations": iteration_count,
        "componentSize": _percentile_record(ranked_size),
        "largestComponentSizes": [
            int(value) for value in ranked_size[:32]
        ],
    }


def _percentile_record(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    quantile = np.percentile(finite, (0, 50, 90, 99, 100))
    return {
        "count": int(len(finite)),
        **{
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "median", "p90", "p99", "maximum"), quantile
            )
        },
    }


def _component_colors(component_id: np.ndarray, maximum: int) -> dict[int, tuple[int, int, int]]:
    present = sorted(
        int(value) for value in np.unique(component_id) if 0 <= value < maximum
    )
    return {
        component: tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.09 + 0.61803398875 * component) % 1.0, 0.68, 0.98
            )
        )
        for component in present
    }


def write_isolated_slab_cross_sections(
    source: VolumeSource,
    owned: VoxelBounds,
    table: IsolatedSlabTable,
    settings: IsolatedSlabSettings,
    path: str | Path,
    *,
    display_high_raw: float,
) -> Path:
    output = Path(path)
    volume = source.memmap()
    world_start = np.asarray(owned.start_xyz) + np.asarray(source.origin_xyz)
    world_stop = np.asarray(owned.stop_xyz_exclusive) + np.asarray(source.origin_xyz)
    z_values = np.linspace(owned.start_xyz[2], owned.stop_xyz_exclusive[2] - 1, 3)
    y_values = np.linspace(owned.start_xyz[1], owned.stop_xyz_exclusive[1] - 1, 3)
    views = [("z", int(round(value))) for value in z_values] + [
        ("y", int(round(value))) for value in y_values
    ]
    panel_width = owned.shape_xyz[0]
    panel_height = max(owned.shape_xyz[1], owned.shape_xyz[2])
    canvas = np.full(
        (2 * panel_height, 3 * panel_width, 3), (7, 10, 14), dtype=np.uint8
    )
    seed = table.confidence >= settings.minimum_seed_confidence
    midpoint = table.midpoint_xyz[seed]
    confidence = table.confidence[seed]
    high = max(float(display_high_raw), 1.0)
    tolerance = max(1.0, settings.sampling_stride_voxels)
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            world_index = source_index + source.origin_xyz[2]
            selected = np.abs(midpoint[:, 2] - world_index) <= tolerance
            x = np.rint(midpoint[selected, 0] - world_start[0]).astype(np.int32)
            y = np.rint(midpoint[selected, 1] - world_start[1]).astype(np.int32)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            world_index = source_index + source.origin_xyz[1]
            selected = np.abs(midpoint[:, 1] - world_index) <= tolerance
            x = np.rint(midpoint[selected, 0] - world_start[0]).astype(np.int32)
            y = np.rint(midpoint[selected, 2] - world_start[2]).astype(np.int32)
        gray = np.clip(np.asarray(raw, dtype=np.float32) / high * 255.0, 0, 255).astype(
            np.uint8
        )
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        score = confidence[selected]
        valid = (
            (x >= 1)
            & (x < panel.shape[1] - 1)
            & (y >= 1)
            & (y < panel.shape[0] - 1)
        )
        x = x[valid]
        y = y[valid]
        score = score[valid]
        color = np.column_stack(
            (
                38.0 + 55.0 * (1.0 - score),
                205.0 + 50.0 * score,
                120.0 + 115.0 * score,
            )
        ).astype(np.uint8)
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                panel[y + dy, x + dx] = color
        row = view_index // 3
        column = view_index % 3
        y0 = row * panel_height + (panel_height - panel.shape[0]) // 2
        x0 = column * panel_width
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def write_isolated_slab_projections(
    table: IsolatedSlabTable,
    owned_world_start_xyz: np.ndarray,
    owned_world_stop_xyz: np.ndarray,
    settings: IsolatedSlabSettings,
    path: str | Path,
    *,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    colors = _component_colors(
        table.component_id, settings.maximum_preview_components
    )
    margin = max(12, panel_size // 30)
    projections = ((0, 1), (0, 2), (1, 2))
    width = np.maximum(
        np.asarray(owned_world_stop_xyz, dtype=np.float64)
        - np.asarray(owned_world_start_xyz, dtype=np.float64),
        1.0,
    )
    for panel_index, axes in enumerate(projections):
        offset = panel_index * panel_size
        for component in sorted(colors, reverse=True):
            selected = table.component_id == component
            point = table.midpoint_xyz[selected]
            if not len(point):
                continue
            normalized = (
                point[:, list(axes)]
                - np.asarray(owned_world_start_xyz)[None, list(axes)]
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
            canvas[y[valid], x[valid]] = colors[component]
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


def write_isolated_slab_ply(
    table: IsolatedSlabTable,
    settings: IsolatedSlabSettings,
    path: str | Path,
) -> Path:
    output = Path(path)
    seed = table.confidence >= settings.minimum_seed_confidence
    colors = _component_colors(table.component_id, settings.maximum_preview_components)
    selected = np.flatnonzero(seed)
    dtype = np.dtype(
        [
            ("x", "<f4"),
            ("y", "<f4"),
            ("z", "<f4"),
            ("nx", "<f4"),
            ("ny", "<f4"),
            ("nz", "<f4"),
            ("thickness", "<f4"),
            ("confidence", "<f4"),
            ("component", "<i4"),
            ("red", "u1"),
            ("green", "u1"),
            ("blue", "u1"),
        ]
    )
    vertices = np.empty(len(selected), dtype=dtype)
    for axis, name in enumerate(("x", "y", "z")):
        vertices[name] = table.midpoint_xyz[selected, axis]
    for axis, name in enumerate(("nx", "ny", "nz")):
        vertices[name] = table.normal_xyz[selected, axis]
    vertices["thickness"] = table.thickness_voxels[selected]
    vertices["confidence"] = table.confidence[selected]
    vertices["component"] = table.component_id[selected]
    rgb = np.full((len(selected), 3), (125, 132, 145), dtype=np.uint8)
    for component, color in colors.items():
        rgb[table.component_id[selected] == component] = color
    vertices["red"], vertices["green"], vertices["blue"] = rgb.T
    header = (
        "ply\n"
        "format binary_little_endian 1.0\n"
        "comment Pareidolia high-confidence isolated slab seeds\n"
        f"element vertex {len(vertices)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property float nx\nproperty float ny\nproperty float nz\n"
        "property float thickness\nproperty float confidence\n"
        "property int component\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\n"
        "end_header\n"
    ).encode("ascii")
    temporary = output.with_suffix(output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        handle.write(header)
        handle.write(vertices.tobytes())
    temporary.replace(output)
    return output


def _processing_bounds(
    source: VolumeSource,
    owned: VoxelBounds,
    settings: IsolatedSlabSettings,
) -> VoxelBounds:
    thickness_voxels = settings.maximum_sheet_thickness_microns / source.voxel_size_microns
    clearance_voxels = settings.minimum_air_clearance_microns / source.voxel_size_microns
    halo = int(
        math.ceil(
            thickness_voxels
            + clearance_voxels
            + 2.5 * settings.smoothing_sigma_voxels
            + settings.sampling_stride_voxels
        )
    )
    stride = settings.sampling_stride_voxels
    start = tuple(
        max(0, ((owned.start_xyz[axis] - halo) // stride) * stride)
        for axis in range(3)
    )
    stop = tuple(
        min(
            source.shape_xyz[axis],
            int(math.ceil((owned.stop_xyz_exclusive[axis] + halo) / stride)) * stride,
        )
        for axis in range(3)
    )
    stop = tuple(
        start[axis] + ((stop[axis] - start[axis]) // stride) * stride
        for axis in range(3)
    )
    return VoxelBounds(start, stop)


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_isolated_slab_detection(
    source_path: str | Path,
    output_path: str | Path,
    *,
    world_start_xyz: tuple[int, int, int],
    world_stop_xyz_exclusive: tuple[int, int, int],
    metadata_path: str | Path | None = None,
    settings: IsolatedSlabSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    settings = settings or IsolatedSlabSettings()
    source = VolumeSource.open(source_path, metadata_path)
    source_origin = np.asarray(source.origin_xyz, dtype=np.int64)
    owned = VoxelBounds(
        tuple((np.asarray(world_start_xyz, dtype=np.int64) - source_origin).tolist()),
        tuple(
            (
                np.asarray(world_stop_xyz_exclusive, dtype=np.int64)
                - source_origin
            ).tolist()
        ),
    )
    if any(owned.start_xyz[axis] < 0 for axis in range(3)) or any(
        owned.stop_xyz_exclusive[axis] > source.shape_xyz[axis]
        for axis in range(3)
    ):
        raise ValueError("owned world bounds lie outside the CT source")
    processing = _processing_bounds(source, owned, settings)
    output = Path(output_path)
    output.mkdir(parents=True, exist_ok=True)
    implementation = {
        "isolated_slab.py": sha256_file(Path(__file__)),
        "contracts.py": sha256_file(Path(__file__).with_name("contracts.py")),
        "export.py": sha256_file(Path(__file__).with_name("export.py")),
        "backend/rectify.py": sha256_file(Path(__file__).parent.parent / "rectify.py"),
    }
    identity = {
        "schema": ISOLATED_SLAB_SCHEMA,
        "version": ISOLATED_SLAB_VERSION,
        "source": dict(source.source_identity),
        "worldBounds": {
            "startXYZ": list(world_start_xyz),
            "stopXYZExclusive": list(world_stop_xyz_exclusive),
        },
        "settings": settings.record(),
        "implementationSha256": implementation,
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    manifest_path = output / f"{ISOLATED_SLAB_STEM}.json"
    data_path = output / f"{ISOLATED_SLAB_STEM}.npz"
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
    timing: dict[str, float] = {}
    volume = source.memmap()[processing.slices_zyx]
    stage = time.monotonic()
    sampled = _downsample_mean_zyx(volume, settings.sampling_stride_voxels)
    timing["blockAverage"] = time.monotonic() - stage
    stage = time.monotonic()
    smoothed = gaussian_blur_3d(
        sampled,
        settings.smoothing_sigma_voxels / settings.sampling_stride_voxels,
    )
    timing["smoothing"] = time.monotonic() - stage
    owned_sample_slices = _owned_sample_slices(
        owned, processing, settings.sampling_stride_voxels
    )
    owned_sample = smoothed[owned_sample_slices]
    calibration = otsu_material_calibration(owned_sample)
    calibration["displayHighRaw"] = float(
        np.percentile(sampled[owned_sample_slices], 99.5)
    )
    if settings.material_threshold_raw is not None:
        calibration["materialThresholdRaw"] = float(settings.material_threshold_raw)
        calibration["method"] = "explicit-threshold-with-otsu-class-scale"

    stage = time.monotonic()
    profile_counts, raw_arrays = detect_isolated_slab_pairs(
        smoothed,
        threshold=float(calibration["materialThresholdRaw"]),
        class_contrast=float(calibration["classContrastRaw"]),
        voxel_size_microns=source.voxel_size_microns,
        settings=settings,
    )
    timing["profilePairing"] = time.monotonic() - stage
    stage = time.monotonic()
    arrays, sampling_midpoint = _deduplicate_and_crop_pairs(
        raw_arrays,
        processing=processing,
        owned=owned,
        source_origin_xyz=source.origin_xyz,
        settings=settings,
    )
    processing_shape_sampling_xyz = tuple(
        value // settings.sampling_stride_voxels
        for value in processing.shape_xyz
    )
    component_id, component_stats = connect_isolated_slab_seeds(
        sampling_midpoint,
        arrays["normal"],
        arrays["thickness_steps"],
        arrays["confidence"],
        processing_shape_xyz=processing_shape_sampling_xyz,
        settings=settings,
    )
    timing["deduplicationAndComponents"] = time.monotonic() - stage
    table = IsolatedSlabTable(
        arrays["midpoint"],
        arrays["normal"],
        arrays["boundary_first"],
        arrays["boundary_second"],
        arrays["thickness_steps"] * settings.sampling_stride_voxels,
        arrays["confidence"],
        arrays["air_margin"],
        arrays["material_margin"],
        arrays["opposing_cosine"],
        component_id,
    )
    table.validate()
    _write_npz(data_path, table.arrays())

    stage = time.monotonic()
    cross_sections_path = write_isolated_slab_cross_sections(
        source,
        owned,
        table,
        settings,
        output / "cross-sections.png",
        display_high_raw=float(calibration["displayHighRaw"]),
    )
    world_start = np.asarray(world_start_xyz, dtype=np.float64)
    world_stop = np.asarray(world_stop_xyz_exclusive, dtype=np.float64)
    projections_path = write_isolated_slab_projections(
        table,
        world_start,
        world_stop,
        settings,
        output / "component-projections.png",
    )
    ply_path = write_isolated_slab_ply(
        table, settings, output / "isolated-slab-seeds.ply"
    )
    timing["artifacts"] = time.monotonic() - stage
    timing["total"] = time.monotonic() - started

    seed = table.confidence >= settings.minimum_seed_confidence
    profile_counts["uniqueOwnedPairCount"] = table.count
    profile_counts["highConfidenceSeedCount"] = int(np.count_nonzero(seed))
    manifest: dict[str, Any] = {
        "schema": ISOLATED_SLAB_SCHEMA,
        "version": ISOLATED_SLAB_VERSION,
        "state": "complete",
        "identity": identity,
        "source": {
            "path": str(source.path),
            "metadataPath": (
                str(source.metadata_path) if source.metadata_path is not None else None
            ),
            "shapeZYX": list(source.shape_zyx),
            "sourceOriginXYZ": list(source.origin_xyz),
            "voxelSizeMicrons": source.voxel_size_microns,
        },
        "geometry": {
            "ownedVoxelBounds": owned.record(),
            "ownedWorldBounds": {
                "startXYZ": list(world_start_xyz),
                "stopXYZExclusive": list(world_stop_xyz_exclusive),
            },
            "processingVoxelBounds": processing.record(),
            "processingShapeSamplingXYZ": list(processing_shape_sampling_xyz),
            "coordinateUnit": "source-voxel",
        },
        "calibration": calibration,
        "counts": profile_counts,
        "components": component_stats,
        "distributions": {
            "thicknessVoxels": _percentile_record(table.thickness_voxels),
            "thicknessMicrons": _percentile_record(
                table.thickness_voxels * source.voxel_size_microns
            ),
            "confidence": _percentile_record(table.confidence),
            "seedThicknessVoxels": _percentile_record(table.thickness_voxels[seed]),
            "seedConfidence": _percentile_record(table.confidence[seed]),
            "airMarginClassFraction": _percentile_record(table.air_margin_fraction),
            "materialMarginClassFraction": _percentile_record(
                table.material_margin_fraction
            ),
            "opposingNormalAngleDegrees": _percentile_record(
                np.degrees(
                    np.arccos(np.clip(table.opposing_normal_cosine, -1.0, 1.0))
                )
            ),
        },
        "timingSeconds": {
            name: round(value, 6) for name, value in timing.items()
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(table.arrays()),
        },
        "artifacts": {
            "crossSections": cross_sections_path.name,
            "componentProjections": projections_path.name,
            "seedPointCloud": ply_path.name,
        },
        "method": {
            "representation": (
                "dense paired opposing CT interfaces with explicit thickness, "
                "air clearance, and confidence"
            ),
            "acusIndependent": True,
            "componentGraphChangesGeometry": False,
            "ambiguityPolicy": (
                "retain profile evidence but seed components only above the "
                "confidence threshold; never bridge a missing profile"
            ),
        },
    }
    atomic_json(manifest_path, manifest)
    return manifest
