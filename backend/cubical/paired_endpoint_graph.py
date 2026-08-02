from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np

from .isolated_slab import _percentile_record


def build_paired_endpoint_continuity_graph(
    bank: Mapping[str, np.ndarray],
    eligible_candidate: np.ndarray,
    *,
    sampling_stride_voxels: float,
    link_radius_sampling_steps: float,
    maximum_normal_degrees: float,
    maximum_endpoint_distance_sampling_steps: float,
    maximum_endpoint_height_sampling_steps: float,
    normal_scale_degrees: float,
    endpoint_height_scale_sampling_steps: float,
    endpoint_distance_scale_sampling_steps: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build independent lower/upper continuation evidence over profile pairs.

    A paired profile can preserve one physical boundary while proposing the
    wrong exit crossing on its other side.  The ordinary paired graph drops
    that observation because it requires both boundaries to agree at once.
    This graph preserves each endpoint affinity separately, while retaining
    the candidate and axial orientation needed to keep the two faces coupled
    during the subsequent one-profile-per-key optimization.
    """

    key_all = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    normal_all = np.asarray(bank["normalXYZ"], dtype=np.float64)
    lower_all = np.asarray(bank["boundaryLowerXYZ"], dtype=np.float64)
    upper_all = np.asarray(bank["boundaryUpperXYZ"], dtype=np.float64)
    candidate_count = len(key_all)
    eligible = np.asarray(eligible_candidate, dtype=bool)
    if eligible.shape != (candidate_count,):
        raise ValueError("endpoint eligibility must align with the paired bank")
    if any(
        len(value) != candidate_count
        for value in (normal_all, lower_all, upper_all)
    ):
        raise ValueError("paired endpoint geometry arrays are not aligned")
    eligible_index = np.flatnonzero(eligible).astype(np.int32)
    if not len(eligible_index):
        empty = np.empty(0, dtype=np.int32)
        arrays = {
            "firstCandidate": empty,
            "secondCandidate": empty.copy(),
            "secondOrientation": np.empty(0, dtype=np.int8),
            "lowerAffinity": np.empty(0, dtype=np.float32),
            "upperAffinity": np.empty(0, dtype=np.float32),
            "normalDegrees": np.empty(0, dtype=np.float32),
            "lowerDistanceSamplingSteps": np.empty(0, dtype=np.float32),
            "upperDistanceSamplingSteps": np.empty(0, dtype=np.float32),
            "lowerHeightSamplingSteps": np.empty(0, dtype=np.float32),
            "upperHeightSamplingSteps": np.empty(0, dtype=np.float32),
        }
        return arrays, {
            "eligibleCandidateCount": 0,
            "eligibleSpatialKeyCount": 0,
            "consideredCandidatePairCount": 0,
            "endpointPairCount": 0,
            "lowerEndpointMatchCount": 0,
            "upperEndpointMatchCount": 0,
            "twoEndpointMatchCount": 0,
            "oneEndpointOnlyMatchCount": 0,
        }

    key = key_all[eligible_index]
    low_key = np.min(key, axis=0)
    shifted_key = key - low_key[None, :]
    shape_xyz = tuple(int(value) for value in (np.max(shifted_key, axis=0) + 1))
    flat = np.ravel_multi_index(
        (shifted_key[:, 2], shifted_key[:, 1], shifted_key[:, 0]),
        shape_xyz[::-1],
    )
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    _unique_flat, first, count = np.unique(
        sorted_flat, return_index=True, return_counts=True
    )
    group_key = shifted_key[order[first]]
    group_grid = np.full(shape_xyz[::-1], -1, dtype=np.int32)
    group_grid[group_key[:, 2], group_key[:, 1], group_key[:, 0]] = np.arange(
        len(group_key), dtype=np.int32
    )
    maximum_alternatives = int(np.max(count, initial=0))
    reach = int(math.ceil(link_radius_sampling_steps))
    stride = float(sampling_stride_voxels)

    first_parts: list[np.ndarray] = []
    second_parts: list[np.ndarray] = []
    orientation_parts: list[np.ndarray] = []
    lower_affinity_parts: list[np.ndarray] = []
    upper_affinity_parts: list[np.ndarray] = []
    normal_parts: list[np.ndarray] = []
    lower_distance_parts: list[np.ndarray] = []
    upper_distance_parts: list[np.ndarray] = []
    lower_height_parts: list[np.ndarray] = []
    upper_height_parts: list[np.ndarray] = []
    considered = 0

    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                if (
                    dx * dx + dy * dy + dz * dz
                    > link_radius_sampling_steps**2 + 1.0e-9
                ):
                    continue
                valid_group = (
                    (group_key[:, 0] + dx >= 0)
                    & (group_key[:, 0] + dx < shape_xyz[0])
                    & (group_key[:, 1] + dy >= 0)
                    & (group_key[:, 1] + dy < shape_xyz[1])
                    & (group_key[:, 2] + dz >= 0)
                    & (group_key[:, 2] + dz < shape_xyz[2])
                )
                source_group = np.flatnonzero(valid_group)
                source_key = group_key[source_group]
                target_group = group_grid[
                    source_key[:, 2] + dz,
                    source_key[:, 1] + dy,
                    source_key[:, 0] + dx,
                ]
                exists = target_group >= 0
                source_group = source_group[exists]
                target_group = target_group[exists]
                if not len(source_group):
                    continue
                for source_alternative in range(maximum_alternatives):
                    source_valid = count[source_group] > source_alternative
                    if not np.any(source_valid):
                        continue
                    selected_source_group = source_group[source_valid]
                    selected_target_group = target_group[source_valid]
                    source_sorted = first[selected_source_group] + source_alternative
                    source_candidate = eligible_index[order[source_sorted]]
                    for target_alternative in range(maximum_alternatives):
                        pair_valid = count[selected_target_group] > target_alternative
                        if not np.any(pair_valid):
                            continue
                        left = source_candidate[pair_valid]
                        right_group = selected_target_group[pair_valid]
                        right_sorted = first[right_group] + target_alternative
                        right = eligible_index[order[right_sorted]]
                        considered += len(left)

                        first_normal = normal_all[left]
                        second_normal = normal_all[right]
                        dot = np.einsum("ij,ij->i", first_normal, second_normal)
                        orientation = np.where(dot >= 0.0, 1, -1).astype(np.int8)
                        normal_degrees = np.degrees(
                            np.arccos(np.clip(np.abs(dot), 0.0, 1.0))
                        )
                        normal_ok = normal_degrees <= maximum_normal_degrees
                        if not np.any(normal_ok):
                            continue
                        left = left[normal_ok]
                        right = right[normal_ok]
                        dot = dot[normal_ok]
                        orientation = orientation[normal_ok]
                        normal_degrees = normal_degrees[normal_ok]
                        first_normal = first_normal[normal_ok]
                        aligned_second_normal = second_normal[normal_ok] * orientation[:, None]
                        average_normal = first_normal + aligned_second_normal
                        average_normal /= np.maximum(
                            np.linalg.norm(average_normal, axis=1, keepdims=True),
                            1.0e-9,
                        )
                        second_lower = np.where(
                            orientation[:, None] > 0,
                            lower_all[right],
                            upper_all[right],
                        )
                        second_upper = np.where(
                            orientation[:, None] > 0,
                            upper_all[right],
                            lower_all[right],
                        )
                        lower_delta = second_lower - lower_all[left]
                        upper_delta = second_upper - upper_all[left]
                        lower_distance = np.linalg.norm(lower_delta, axis=1) / stride
                        upper_distance = np.linalg.norm(upper_delta, axis=1) / stride
                        lower_height = np.abs(
                            np.einsum("ij,ij->i", lower_delta, average_normal)
                        ) / stride
                        upper_height = np.abs(
                            np.einsum("ij,ij->i", upper_delta, average_normal)
                        ) / stride
                        lower_match = (
                            (lower_distance <= maximum_endpoint_distance_sampling_steps)
                            & (lower_height <= maximum_endpoint_height_sampling_steps)
                        )
                        upper_match = (
                            (upper_distance <= maximum_endpoint_distance_sampling_steps)
                            & (upper_height <= maximum_endpoint_height_sampling_steps)
                        )
                        retained = lower_match | upper_match
                        if not np.any(retained):
                            continue
                        left = left[retained]
                        right = right[retained]
                        orientation = orientation[retained]
                        normal_degrees = normal_degrees[retained]
                        lower_distance = lower_distance[retained]
                        upper_distance = upper_distance[retained]
                        lower_height = lower_height[retained]
                        upper_height = upper_height[retained]
                        lower_match = lower_match[retained]
                        upper_match = upper_match[retained]
                        normal_quality = np.exp(
                            -0.5 * (normal_degrees / normal_scale_degrees) ** 2
                        )
                        lower_affinity = np.where(
                            lower_match,
                            normal_quality
                            * np.exp(
                                -0.5
                                * (
                                    lower_height
                                    / endpoint_height_scale_sampling_steps
                                )
                                ** 2
                                -0.5
                                * (
                                    lower_distance
                                    / endpoint_distance_scale_sampling_steps
                                )
                                ** 2
                            ),
                            0.0,
                        )
                        upper_affinity = np.where(
                            upper_match,
                            normal_quality
                            * np.exp(
                                -0.5
                                * (
                                    upper_height
                                    / endpoint_height_scale_sampling_steps
                                )
                                ** 2
                                -0.5
                                * (
                                    upper_distance
                                    / endpoint_distance_scale_sampling_steps
                                )
                                ** 2
                            ),
                            0.0,
                        )
                        first_parts.append(left.astype(np.int32))
                        second_parts.append(right.astype(np.int32))
                        orientation_parts.append(orientation)
                        lower_affinity_parts.append(lower_affinity.astype(np.float32))
                        upper_affinity_parts.append(upper_affinity.astype(np.float32))
                        normal_parts.append(normal_degrees.astype(np.float32))
                        lower_distance_parts.append(lower_distance.astype(np.float32))
                        upper_distance_parts.append(upper_distance.astype(np.float32))
                        lower_height_parts.append(lower_height.astype(np.float32))
                        upper_height_parts.append(upper_height.astype(np.float32))

    def concatenate(parts: list[np.ndarray], dtype: np.dtype[Any]) -> np.ndarray:
        return np.concatenate(parts).astype(dtype, copy=False) if parts else np.empty(0, dtype=dtype)

    arrays = {
        "firstCandidate": concatenate(first_parts, np.int32),
        "secondCandidate": concatenate(second_parts, np.int32),
        "secondOrientation": concatenate(orientation_parts, np.int8),
        "lowerAffinity": concatenate(lower_affinity_parts, np.float32),
        "upperAffinity": concatenate(upper_affinity_parts, np.float32),
        "normalDegrees": concatenate(normal_parts, np.float32),
        "lowerDistanceSamplingSteps": concatenate(lower_distance_parts, np.float32),
        "upperDistanceSamplingSteps": concatenate(upper_distance_parts, np.float32),
        "lowerHeightSamplingSteps": concatenate(lower_height_parts, np.float32),
        "upperHeightSamplingSteps": concatenate(upper_height_parts, np.float32),
    }
    lower_match = arrays["lowerAffinity"] > 0.0
    upper_match = arrays["upperAffinity"] > 0.0
    summary = {
        "eligibleCandidateCount": int(len(eligible_index)),
        "eligibleSpatialKeyCount": int(len(group_key)),
        "consideredCandidatePairCount": int(considered),
        "endpointPairCount": int(len(arrays["firstCandidate"])),
        "lowerEndpointMatchCount": int(np.count_nonzero(lower_match)),
        "upperEndpointMatchCount": int(np.count_nonzero(upper_match)),
        "twoEndpointMatchCount": int(np.count_nonzero(lower_match & upper_match)),
        "oneEndpointOnlyMatchCount": int(np.count_nonzero(lower_match ^ upper_match)),
        "lowerAffinity": _percentile_record(arrays["lowerAffinity"][lower_match]),
        "upperAffinity": _percentile_record(arrays["upperAffinity"][upper_match]),
        "normalDegrees": _percentile_record(arrays["normalDegrees"]),
        "lowerHeightSamplingSteps": _percentile_record(
            arrays["lowerHeightSamplingSteps"][lower_match]
        ),
        "upperHeightSamplingSteps": _percentile_record(
            arrays["upperHeightSamplingSteps"][upper_match]
        ),
    }
    return arrays, summary
