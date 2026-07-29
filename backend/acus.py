from __future__ import annotations

import math
import time
from typing import Any

import numpy as np

from .rectify import VolumeData, _trilinear
from .acus_compute import hessian_line_fields


def _block_candidates(score: np.ndarray, margin: int, spacing: int) -> list[tuple[float, int, int, int]]:
    interior = score[margin : score.shape[0] - margin, margin : score.shape[1] - margin, margin : score.shape[2] - margin]
    positive = interior[interior > 0.0]
    if not positive.size:
        return []
    threshold = max(0.015, float(np.percentile(positive, 76.0)))
    candidates: list[tuple[float, int, int, int]] = []
    for z0 in range(margin, score.shape[0] - margin, spacing):
        for y0 in range(margin, score.shape[1] - margin, spacing):
            for x0 in range(margin, score.shape[2] - margin, spacing):
                block = score[
                    z0 : min(z0 + spacing, score.shape[0] - margin),
                    y0 : min(y0 + spacing, score.shape[1] - margin),
                    x0 : min(x0 + spacing, score.shape[2] - margin),
                ]
                if not block.size:
                    continue
                flat_index = int(np.argmax(block))
                value = float(block.flat[flat_index])
                if value < threshold:
                    continue
                dz, dy, dx = np.unravel_index(flat_index, block.shape)
                candidates.append((value, z0 + dz, y0 + dy, x0 + dx))
    candidates.sort(reverse=True)
    return candidates


def _refine_needle(
    score: np.ndarray,
    direction_field: np.ndarray,
    candidate: tuple[float, int, int, int],
    radius: int,
    needle_length: float,
    cross_section_radius: float,
) -> dict[str, Any] | None:
    response, z, y, x = candidate
    z0, z1 = z - radius, z + radius + 1
    y0, y1 = y - radius, y + radius + 1
    x0, x1 = x - radius, x + radius + 1
    local_score = score[z0:z1, y0:y1, x0:x1]
    zz, yy, xx = np.indices(local_score.shape, dtype=np.float32)
    coordinates = np.stack(
        [xx + x0, yy + y0, zz + z0], axis=-1
    ).reshape(-1, 3)
    distance2 = (xx - radius) ** 2 + (yy - radius) ** 2 + (zz - radius) ** 2
    weights = (
        local_score
        * np.exp(-distance2 / max(2.0 * (radius * 0.78) ** 2, 1.0))
        * (distance2 <= radius * radius)
    ).reshape(-1)
    weight_sum = float(weights.sum())
    if weight_sum <= 1.0e-6:
        return None
    center = np.sum(coordinates * weights[:, None], axis=0) / weight_sum
    centered = coordinates - center
    covariance = (centered * weights[:, None]).T @ centered / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    direction = eigenvectors[:, 2]
    reference = direction_field[z, y, x]
    if float(np.dot(direction, reference)) < 0.0:
        direction = -direction
    total = max(float(eigenvalues.sum()), 1.0e-6)
    linearity = float(np.clip((eigenvalues[2] - eigenvalues[1]) / total, 0.0, 1.0))
    if linearity < 0.035:
        return None
    axial_samples = max(9, int(math.ceil(needle_length)) + 1)
    offsets = np.linspace(-needle_length * 0.5, needle_length * 0.5, axial_samples)
    u_axis, v_axis = _plane_basis(direction)
    lateral = cross_section_radius * 0.55
    sample_offsets = np.stack(
        [
            np.zeros(3, dtype=np.float32),
            u_axis * lateral,
            -u_axis * lateral,
            v_axis * lateral,
            -v_axis * lateral,
        ]
    )
    axial_points = center[None, None, :] + offsets[:, None, None] * direction[None, None, :]
    support_points = axial_points + sample_offsets[None, :, :]
    axial_response = np.max(_trilinear(score, support_points, outside=0.0), axis=1)
    support_threshold = max(0.008, response * 0.12)
    supported = axial_response >= support_threshold
    axial_coverage = float(np.mean(supported))
    longest_run = 0
    current_run = 0
    for value in supported:
        current_run = current_run + 1 if value else 0
        longest_run = max(longest_run, current_run)
    longest_fraction = float(longest_run / len(supported))
    support_score = float(
        np.mean(np.clip(axial_response / max(response * 0.65, 1.0e-6), 0.0, 1.0))
    )
    if axial_coverage < 0.42 or longest_fraction < 0.34 or support_score < 0.18:
        return None
    confidence = float(
        np.clip(
            response
            * (0.38 + 1.4 * linearity)
            * (0.48 + 0.52 * support_score),
            0.0,
            1.0,
        )
    )
    return {
        "center": center.astype(np.float32),
        "direction": direction.astype(np.float32),
        "linearity": linearity,
        "score": confidence,
        "length": float(needle_length),
        "axialCoverage": axial_coverage,
        "longestAxialRun": longest_fraction,
        "supportScore": support_score,
    }


def _refine_needles_batch(
    score: np.ndarray,
    direction_field: np.ndarray,
    candidates: np.ndarray,
    radius: int,
    needle_length: float,
    cross_section_radius: float,
    batch_size: int = 2048,
) -> list[dict[str, Any]]:
    """Vectorized equivalent of ``_refine_needle`` for slab-scale candidates."""
    candidate_values = np.asarray(candidates, dtype=np.float32)
    if candidate_values.size == 0:
        return []
    if candidate_values.ndim != 2 or candidate_values.shape[1] != 4:
        raise ValueError("candidates must have response,z,y,x columns")

    local_axis = np.arange(-radius, radius + 1, dtype=np.int32)
    dz, dy, dx = np.meshgrid(local_axis, local_axis, local_axis, indexing="ij")
    distance2 = dx * dx + dy * dy + dz * dz
    inside = distance2 <= radius * radius
    offsets_zyx = np.stack([dz, dy, dx], axis=-1).reshape(-1, 3)
    offsets_xyz = offsets_zyx[:, ::-1].astype(np.float32)
    spatial_weight = np.exp(
        -distance2.reshape(-1).astype(np.float32)
        / max(2.0 * (radius * 0.78) ** 2, 1.0)
    ).astype(np.float32) * inside.reshape(-1).astype(np.float32)
    axial_samples = max(9, int(math.ceil(needle_length)) + 1)
    axial_offsets = np.linspace(
        -needle_length * 0.5, needle_length * 0.5, axial_samples
    )
    lateral = cross_section_radius * 0.55
    refined: list[dict[str, Any]] = []

    for start in range(0, len(candidate_values), max(1, int(batch_size))):
        stop = min(start + max(1, int(batch_size)), len(candidate_values))
        batch = candidate_values[start:stop]
        response = batch[:, 0]
        points_zyx = np.rint(batch[:, 1:4]).astype(np.int32)
        local_score = score[
            points_zyx[:, None, 0] + offsets_zyx[None, :, 0],
            points_zyx[:, None, 1] + offsets_zyx[None, :, 1],
            points_zyx[:, None, 2] + offsets_zyx[None, :, 2],
        ]
        weights = local_score * spatial_weight[None, :]
        weight_sum = np.sum(weights, axis=1)
        valid_weight = weight_sum > 1.0e-6
        safe_weight = np.maximum(weight_sum, 1.0e-6)
        coordinates = (
            points_zyx[:, None, ::-1].astype(np.float32)
            + offsets_xyz[None, :, :]
        )
        center = np.sum(coordinates * weights[:, :, None], axis=1) / safe_weight[:, None]
        centered = coordinates - center[:, None, :]
        weighted_centered = centered * weights[:, :, None]
        covariance = np.matmul(
            np.transpose(weighted_centered, (0, 2, 1)), centered
        ) / safe_weight[:, None, None]
        eigenvalues = np.zeros((len(batch), 3), dtype=np.float32)
        eigenvectors = np.zeros((len(batch), 3, 3), dtype=np.float32)
        # The 3x3 solves are cheap relative to neighborhood construction. Keeping
        # them per-candidate preserves the scalar path's exact tie ordering while
        # all high-volume sampling remains vectorized.
        for local_index in np.flatnonzero(valid_weight):
            values, vectors = np.linalg.eigh(covariance[local_index])
            eigenvalues[local_index] = values
            eigenvectors[local_index] = vectors
        direction = eigenvectors[:, :, 2]
        reference = direction_field[
            points_zyx[:, 0], points_zyx[:, 1], points_zyx[:, 2]
        ]
        direction *= np.where(
            np.sum(direction * reference, axis=1) < 0.0, -1.0, 1.0
        )[:, None]
        total = np.maximum(np.sum(eigenvalues, axis=1), 1.0e-6)
        linearity = np.clip(
            (eigenvalues[:, 2] - eigenvalues[:, 1]) / total, 0.0, 1.0
        )
        geometry_valid = valid_weight & (linearity >= 0.035)
        if not np.any(geometry_valid):
            continue

        local_indices = np.flatnonzero(geometry_valid)
        selected_direction = direction[local_indices]
        selected_center = center[local_indices]
        basis_axis = np.zeros_like(selected_direction)
        basis_axis[:, 2] = 1.0
        near_z = np.abs(selected_direction[:, 2]) > 0.86
        basis_axis[near_z] = np.asarray([0.0, 1.0, 0.0], dtype=np.float32)
        u_axis = np.cross(selected_direction, basis_axis)
        u_axis /= np.maximum(
            np.linalg.norm(u_axis, axis=1, keepdims=True), 1.0e-7
        )
        v_axis = np.cross(selected_direction, u_axis)
        v_axis /= np.maximum(
            np.linalg.norm(v_axis, axis=1, keepdims=True), 1.0e-7
        )
        sample_offsets = np.stack(
            [
                np.zeros_like(u_axis),
                u_axis * lateral,
                -u_axis * lateral,
                v_axis * lateral,
                -v_axis * lateral,
            ],
            axis=1,
        )
        axial_points = (
            selected_center[:, None, None, :]
            + axial_offsets[None, :, None, None]
            * selected_direction[:, None, None, :]
        )
        support_points = axial_points + sample_offsets[:, None, :, :]
        axial_response = np.max(
            _trilinear(score, support_points, outside=0.0), axis=2
        )
        selected_response = response[local_indices]
        supported = axial_response >= np.maximum(
            0.008, selected_response[:, None] * 0.12
        )
        axial_coverage = np.mean(supported, axis=1)
        current_run = np.zeros(len(local_indices), dtype=np.int16)
        longest_run = np.zeros(len(local_indices), dtype=np.int16)
        for column in supported.T:
            current_run = np.where(column, current_run + 1, 0)
            longest_run = np.maximum(longest_run, current_run)
        longest_fraction = longest_run.astype(np.float32) / supported.shape[1]
        support_score = np.mean(
            np.clip(
                axial_response
                / np.maximum(selected_response[:, None] * 0.65, 1.0e-6),
                0.0,
                1.0,
            ),
            axis=1,
        )
        support_valid = (
            (axial_coverage >= 0.42)
            & (longest_fraction >= 0.34)
            & (support_score >= 0.18)
        )
        confidence = np.clip(
            selected_response
            * (0.38 + 1.4 * linearity[local_indices])
            * (0.48 + 0.52 * support_score),
            0.0,
            1.0,
        )
        for position in np.flatnonzero(support_valid):
            candidate_index = int(start + local_indices[position])
            refined.append(
                {
                    "candidateIndex": candidate_index,
                    "center": selected_center[position].astype(np.float32),
                    "direction": selected_direction[position].astype(np.float32),
                    "linearity": float(linearity[local_indices[position]]),
                    "score": float(confidence[position]),
                    "length": float(needle_length),
                    "axialCoverage": float(axial_coverage[position]),
                    "longestAxialRun": float(longest_fraction[position]),
                    "supportScore": float(support_score[position]),
                }
            )
    return refined


def _robust_common_normal(directions: np.ndarray, weights: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    robust = np.ones_like(weights, dtype=np.float32)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    eigenvalues = np.zeros(3, dtype=np.float32)
    for _ in range(7):
        combined = np.maximum(weights * robust, 1.0e-7)
        matrix = (directions * combined[:, None]).T @ directions / float(combined.sum())
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        normal = eigenvectors[:, 0].astype(np.float32)
        residual = np.abs(directions @ normal)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(0.025, median + 2.5 * 1.4826 * mad)
        robust = (1.0 / (1.0 + (residual / scale) ** 4)).astype(np.float32)
    return normal, eigenvalues.astype(np.float32), robust


def _plane_basis(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    if abs(float(np.dot(axis, normal))) > 0.86:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
    u = np.cross(normal, axis)
    u /= max(float(np.linalg.norm(u)), 1.0e-7)
    v = np.cross(normal, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-7)
    return u, v


def _orientation_profile(
    normal_coordinates: np.ndarray,
    family_angles: np.ndarray,
    weights: np.ndarray,
    inliers: np.ndarray,
    cube_size: int,
) -> dict[str, Any]:
    """Estimate circular orientation density as a function of signed normal position."""
    depth_sigma = max(2.0, cube_size / 24.0)
    angle_sigma = 9.0
    depth_count = int(np.clip(round(cube_size / 2.0), 16, 48))
    depth_centers = np.linspace(
        -cube_size * 0.5 + depth_sigma,
        cube_size * 0.5 - depth_sigma,
        depth_count,
        dtype=np.float32,
    )
    orientation_centers = np.linspace(2.5, 177.5, 36, dtype=np.float32)
    base_weights = weights * inliers.astype(np.float32)
    depth_kernel = np.exp(
        -0.5
        * ((depth_centers[:, None] - normal_coordinates[None, :]) / depth_sigma) ** 2
    )
    angle_delta = np.abs(
        orientation_centers[None, :] - family_angles[:, None]
    )
    angle_delta = np.minimum(angle_delta, 180.0 - angle_delta)
    angle_kernel = np.exp(-0.5 * (angle_delta / angle_sigma) ** 2)
    weighted_depth = depth_kernel * base_weights[None, :]
    raw_density = weighted_depth @ angle_kernel
    support = weighted_depth.sum(axis=1)
    maximum = max(float(raw_density.max()), 1.0e-8)
    density = raw_density / maximum

    slices: list[dict[str, Any]] = []
    concentrations: list[float] = []
    two_mode_coverages: list[float] = []
    concentration_weights: list[float] = []
    support_limit = max(float(support.max()) * 0.12, 1.0e-6)
    for index, row in enumerate(raw_density):
        row_sum = float(row.sum())
        concentration = 0.0
        two_mode_coverage = 0.0
        dominant: list[dict[str, float]] = []
        if row_sum > 1.0e-8:
            probability = row / row_sum
            entropy = -float(
                np.sum(probability * np.log(np.maximum(probability, 1.0e-12)))
            ) / math.log(len(probability))
            concentration = float(np.clip(1.0 - entropy, 0.0, 1.0))
            peak_indices = [
                angle_index
                for angle_index in range(len(row))
                if row[angle_index] >= row[(angle_index - 1) % len(row)]
                and row[angle_index] >= row[(angle_index + 1) % len(row)]
            ]
            peak_indices.sort(key=lambda angle_index: float(row[angle_index]), reverse=True)
            for angle_index in peak_indices:
                relative_strength = float(row[angle_index] / max(float(row.max()), 1.0e-8))
                angle = float(orientation_centers[angle_index])
                if relative_strength < 0.34:
                    continue
                if any(
                    min(abs(angle - peak["angleDeg"]), 180.0 - abs(angle - peak["angleDeg"]))
                    < 24.0
                    for peak in dominant
                ):
                    continue
                dominant.append(
                    {
                        "angleDeg": round(angle, 2),
                        "relativeStrength": round(relative_strength, 4),
                    }
                )
                if len(dominant) == 2:
                    break
            if dominant:
                covered = np.zeros(len(probability), dtype=bool)
                for peak in dominant:
                    distance = np.abs(orientation_centers - peak["angleDeg"])
                    distance = np.minimum(distance, 180.0 - distance)
                    covered |= distance <= 17.5
                two_mode_coverage = float(probability[covered].sum())
        if float(support[index]) >= support_limit:
            concentrations.append(concentration)
            two_mode_coverages.append(two_mode_coverage)
            concentration_weights.append(float(support[index]))
        slices.append(
            {
                "normalCoordinate": round(float(depth_centers[index]), 3),
                "support": round(float(support[index]), 4),
                "concentration": round(concentration, 4),
                "twoModeCoverage": round(two_mode_coverage, 4),
                "dominantAngles": dominant,
            }
        )

    if concentration_weights:
        mean_concentration = float(
            np.average(
                np.asarray(concentrations, dtype=np.float32),
                weights=np.asarray(concentration_weights, dtype=np.float32),
            )
        )
    else:
        mean_concentration = 0.0
    if concentration_weights:
        mean_two_mode_coverage = float(
            np.average(
                np.asarray(two_mode_coverages, dtype=np.float32),
                weights=np.asarray(concentration_weights, dtype=np.float32),
            )
        )
    else:
        mean_two_mode_coverage = 0.0
    covered_fraction = float(np.mean(support >= support_limit)) if support.size else 0.0
    return {
        "normalCoordinateRange": [round(-cube_size * 0.5, 3), round(cube_size * 0.5, 3)],
        "depthCenters": np.round(depth_centers, 3).tolist(),
        "orientationCentersDeg": np.round(orientation_centers, 2).tolist(),
        "density": np.round(density, 4).tolist(),
        "slices": slices,
        "stats": {
            "meanOrientationConcentration": round(mean_concentration, 4),
            "meanTwoModeCoverage": round(mean_two_mode_coverage, 4),
            "coveredDepthFraction": round(covered_fraction, 4),
            "normalBandwidthVoxels": round(depth_sigma, 3),
            "orientationBandwidthDeg": angle_sigma,
            "interpretation": "orientation density along signed n; clusters are exploratory, not sheets",
        },
    }


def _prepare_acus_context(
    volume: VolumeData, request: dict[str, Any]
) -> tuple[dict[str, Any], np.ndarray]:
    seed_object = request.get("seed") or {}
    seed = tuple(
        int(round(float(seed_object.get(axis, -1)))) for axis in ("x", "y", "z")
    )
    cube_size = int(np.clip(request.get("cubeSize", 64), 16, 128))
    scale = float(np.clip(request.get("scale", 1.25), 0.7, 3.0))
    spacing = int(np.clip(request.get("spacing", 4), 3, 10))
    max_needles = int(np.clip(request.get("maxNeedles", 160), 24, 320))
    needle_length = float(np.clip(request.get("needleLength", 16.0), 6.0, 48.0))
    requested_padding = int(
        np.clip(request.get("contextPadding", math.ceil(needle_length)), 0, 48)
    )
    gaussian_radius = max(1, int(math.ceil(2.5 * scale)))
    cross_section_radius = max(2.0, float(math.ceil(scale * 1.5)))
    minimum_padding = int(
        math.ceil(gaussian_radius + needle_length * 0.5 + cross_section_radius)
    )
    minimum_padding = int(math.ceil(minimum_padding / 4.0) * 4)
    allow_insufficient_padding = bool(request.get("allowInsufficientPadding", False))
    context_padding = (
        requested_padding
        if allow_insufficient_padding
        else max(requested_padding, minimum_padding)
    )
    context_size = cube_size + 2 * context_padding
    cube, context_origin_xyz = volume.context_cube(seed, context_size)
    origin_xyz = tuple(value + context_padding for value in context_origin_xyz)
    return {
        "seed": seed,
        "cubeSize": cube_size,
        "scale": scale,
        "spacing": spacing,
        "maxNeedles": max_needles,
        "needleLength": needle_length,
        "requestedPadding": requested_padding,
        "minimumPadding": minimum_padding,
        "contextPadding": context_padding,
        "contextSize": context_size,
        "crossSectionRadius": cross_section_radius,
        "originXYZ": origin_xyz,
    }, cube.astype(np.float32) / 255.0


def fit_acus(
    volume: VolumeData,
    request: dict[str, Any],
    *,
    _precomputed: tuple[np.ndarray, np.ndarray, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    started = time.perf_counter()
    prepared, data = _prepare_acus_context(volume, request)
    seed = prepared["seed"]
    cube_size = prepared["cubeSize"]
    scale = prepared["scale"]
    spacing = prepared["spacing"]
    max_needles = prepared["maxNeedles"]
    needle_length = prepared["needleLength"]
    requested_padding = prepared["requestedPadding"]
    minimum_padding = prepared["minimumPadding"]
    context_padding = prepared["contextPadding"]
    context_size = prepared["contextSize"]
    cross_section_radius = prepared["crossSectionRadius"]
    origin_xyz = prepared["originXYZ"]

    if _precomputed is None:
        line_fields, compute_meta = hessian_line_fields([data], scale)
        full_score, direction_field = line_fields[0]
    else:
        full_score, direction_field, compute_meta = _precomputed
    radius = max(3, int(math.ceil(scale * 2.5)))
    margin = radius + 2
    full_score[:margin] = full_score[-margin:] = 0.0
    full_score[:, :margin] = full_score[:, -margin:] = 0.0
    full_score[:, :, :margin] = full_score[:, :, -margin:] = 0.0
    candidate_score = np.zeros_like(full_score)
    inner = slice(context_padding, context_padding + cube_size)
    candidate_score[inner, inner, inner] = full_score[inner, inner, inner]
    candidates = _block_candidates(candidate_score, margin, spacing)

    needles: list[dict[str, Any]] = []
    accepted_centers: list[np.ndarray] = []
    minimum_distance2 = float(max(2, spacing - 1) ** 2)
    for candidate in candidates:
        center_zyx = np.array([candidate[1], candidate[2], candidate[3]], dtype=np.float32)
        if any(float(np.sum((center_zyx - existing) ** 2)) < minimum_distance2 for existing in accepted_centers):
            continue
        needle = _refine_needle(
            full_score,
            direction_field,
            candidate,
            radius,
            needle_length,
            cross_section_radius,
        )
        if needle is None:
            continue
        accepted_centers.append(center_zyx)
        needles.append(needle)
        if len(needles) >= max_needles:
            break

    if len(needles) < 6:
        raise ValueError("not enough coherent line-like regions were found in this cube")

    directions = np.stack([needle["direction"] for needle in needles])
    weights = np.asarray([needle["score"] for needle in needles], dtype=np.float32)
    normal, eigenvalues, robust_weights = _robust_common_normal(directions, weights)
    dominant_axis = int(np.argmax(np.abs(normal)))
    if normal[dominant_axis] < 0.0:
        normal = -normal
    residual_degrees = np.degrees(
        np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0))
    )
    median_residual = float(np.median(residual_degrees))
    inlier_limit = max(8.0, min(22.0, median_residual * 2.8))
    inliers = residual_degrees <= inlier_limit
    total_eigenvalue = max(float(eigenvalues.sum()), 1.0e-7)
    normal_confidence = float(
        np.clip(
            ((eigenvalues[1] - eigenvalues[0]) / max(eigenvalues[2], 1.0e-7))
            * float(np.mean(inliers)),
            0.0,
            1.0,
        )
    )
    coplanarity = float(np.clip(1.0 - eigenvalues[0] / total_eigenvalue, 0.0, 1.0))
    u_axis, v_axis = _plane_basis(normal)
    centers = np.stack([needle["center"] for needle in needles]) - context_padding
    normal_coordinates = (centers - np.full(3, cube_size * 0.5, dtype=np.float32)) @ normal
    family_angles = np.degrees(
        np.arctan2(directions @ v_axis, directions @ u_axis)
    ) % 180.0
    profile = _orientation_profile(
        normal_coordinates,
        family_angles,
        weights * robust_weights,
        inliers,
        cube_size,
    )

    ox, oy, oz = origin_xyz
    serialized: list[dict[str, Any]] = []
    for index, needle in enumerate(needles):
        center = needle["center"] - context_padding
        direction = needle["direction"]
        half_length = needle["length"] * 0.5
        start_point = center - direction * half_length
        end_point = center + direction * half_length
        family_angle = float(family_angles[index])
        serialized.append(
            {
                "center": np.round(center, 3).tolist(),
                "start": np.round(start_point, 3).tolist(),
                "end": np.round(end_point, 3).tolist(),
                "direction": np.round(direction, 5).tolist(),
                "length": round(float(needle["length"]), 3),
                "score": round(float(needle["score"]), 4),
                "linearity": round(float(needle["linearity"]), 4),
                "axialCoverage": round(float(needle["axialCoverage"]), 4),
                "longestAxialRun": round(float(needle["longestAxialRun"]), 4),
                "supportScore": round(float(needle["supportScore"]), 4),
                "robustWeight": round(float(robust_weights[index]), 4),
                "normalCoordinate": round(float(normal_coordinates[index]), 3),
                "planeResidualDeg": round(float(residual_degrees[index]), 3),
                "familyAngleDeg": round(family_angle, 2),
                "inlier": bool(inliers[index]),
            }
        )

    global_normal = normal
    boundary_band = needle_length * 0.5 + cross_section_radius
    boundary_tangential: list[bool] = []
    for needle, center in zip(needles, centers):
        face_distances = np.minimum(center, cube_size - center)
        nearest_axis = int(np.argmin(face_distances))
        if float(face_distances[nearest_axis]) <= boundary_band:
            boundary_tangential.append(
                abs(float(needle["direction"][nearest_axis])) < math.sin(math.radians(15.0))
            )
    elapsed = (time.perf_counter() - started) * 1000.0
    return {
        "seed": {"x": seed[0], "y": seed[1], "z": seed[2]},
        "globalSeed": {
            "x": seed[0] + volume.origin_xyz[0],
            "y": seed[1] + volume.origin_xyz[1],
            "z": seed[2] + volume.origin_xyz[2],
        },
        "cube": {
            "size": cube_size,
            "origin": {"x": ox, "y": oy, "z": oz},
            "order": "ZYX",
        },
        "settings": {
            "needleLength": round(needle_length, 3),
            "requestedPadding": requested_padding,
            "effectivePadding": context_padding,
            "minimumPadding": minimum_padding,
            "contextSize": context_size,
            "crossSectionRadius": round(cross_section_radius, 3),
            "paddingSufficient": context_padding >= minimum_padding,
        },
        "needles": serialized,
        "normal": np.round(global_normal, 6).tolist(),
        "normalSignConvention": "largest-magnitude XYZ component is positive",
        "normalLine": {
            "start": np.round(np.full(3, cube_size * 0.5) - global_normal * cube_size * 0.42, 3).tolist(),
            "end": np.round(np.full(3, cube_size * 0.5) + global_normal * cube_size * 0.42, 3).tolist(),
        },
        "orientationProfile": profile,
        "stats": {
            "elapsedMs": round(elapsed, 1),
            "computeBackend": compute_meta["backend"],
            "computeDevice": compute_meta.get("device"),
            "lineFieldMs": compute_meta["elapsedMs"],
            "lineFieldBatchSize": compute_meta["itemCount"],
            "lineFieldBatchLaunches": compute_meta["batchLaunches"],
            "computeFallbackReason": compute_meta.get("fallbackReason"),
            "candidateCount": len(candidates),
            "needleCount": len(needles),
            "inlierCount": int(np.count_nonzero(inliers)),
            "inlierFraction": round(float(np.mean(inliers)), 4),
            "medianPlaneResidualDeg": round(median_residual, 3),
            "p90PlaneResidualDeg": round(float(np.percentile(residual_degrees, 90.0)), 3),
            "normalConfidence": round(normal_confidence, 4),
            "coplanarity": round(coplanarity, 4),
            "orientationEigenvalues": np.round(eigenvalues, 6).tolist(),
            "boundaryNeedleCount": len(boundary_tangential),
            "boundaryTangentialFraction": round(float(np.mean(boundary_tangential)), 4)
            if boundary_tangential
            else None,
            "medianAxialCoverage": round(
                float(np.median([needle["axialCoverage"] for needle in needles])), 4
            ),
            "constraint": "shared unsigned normal; no surface or sheet identity fitted",
        },
    }


def _minimal_rotation(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Return the smallest 3D rotation carrying one unit vector to another."""
    source = source / max(float(np.linalg.norm(source)), 1.0e-8)
    target = target / max(float(np.linalg.norm(target)), 1.0e-8)
    cross = np.cross(source, target)
    sine = float(np.linalg.norm(cross))
    cosine = float(np.clip(np.dot(source, target), -1.0, 1.0))
    if sine < 1.0e-7:
        return np.eye(3, dtype=np.float32)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ],
        dtype=np.float32,
    )
    return (
        np.eye(3, dtype=np.float32)
        + skew
        + (skew @ skew) * ((1.0 - cosine) / max(sine * sine, 1.0e-8))
    )


def _aligned_density(
    result: dict[str, Any],
    anchor_normal: np.ndarray,
    anchor_u: np.ndarray,
    anchor_v: np.ndarray,
    depth_centers: np.ndarray,
    orientation_centers: np.ndarray,
    depth_sigma: float,
    angle_sigma: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Transport a fit into the anchor tangent frame and form its depth profile."""
    result_normal = np.asarray(result["normal"], dtype=np.float32)
    coordinate_sign = 1.0
    if float(np.dot(result_normal, anchor_normal)) < 0.0:
        result_normal = -result_normal
        coordinate_sign = -1.0
    rotation = _minimal_rotation(result_normal, anchor_normal)
    needles = result["needles"]
    directions = np.asarray([needle["direction"] for needle in needles], dtype=np.float32)
    aligned_directions = (rotation @ directions.T).T
    family_angles = np.degrees(
        np.arctan2(aligned_directions @ anchor_v, aligned_directions @ anchor_u)
    ) % 180.0
    normal_coordinates = (
        np.asarray([needle["normalCoordinate"] for needle in needles], dtype=np.float32)
        * coordinate_sign
    )
    weights = np.asarray(
        [
            needle["score"] * needle["robustWeight"] if needle["inlier"] else 0.0
            for needle in needles
        ],
        dtype=np.float32,
    )
    depth_kernel = np.exp(
        -0.5
        * ((depth_centers[:, None] - normal_coordinates[None, :]) / depth_sigma) ** 2
    )
    angle_delta = np.abs(orientation_centers[None, :] - family_angles[:, None])
    angle_delta = np.minimum(angle_delta, 180.0 - angle_delta)
    angle_kernel = np.exp(-0.5 * (angle_delta / angle_sigma) ** 2)
    weighted_depth = depth_kernel * weights[None, :]
    return (weighted_depth @ angle_kernel).astype(np.float32), weighted_depth.sum(axis=1)


def _best_profile_alignment(
    anchor_density: np.ndarray,
    anchor_support: np.ndarray,
    neighbor_density: np.ndarray,
    neighbor_support: np.ndarray,
    depth_step: float,
    maximum_lag: float,
) -> tuple[float, float]:
    anchor_norm = np.linalg.norm(anchor_density, axis=1)
    neighbor_norm = np.linalg.norm(neighbor_density, axis=1)
    anchor_rows = anchor_density / np.maximum(anchor_norm[:, None], 1.0e-8)
    neighbor_rows = neighbor_density / np.maximum(neighbor_norm[:, None], 1.0e-8)
    maximum_shift = min(
        max(1, int(round(maximum_lag / max(depth_step, 1.0e-6)))),
        max(1, len(anchor_rows) // 3),
    )
    candidates: list[tuple[float, float]] = []
    for shift in range(-maximum_shift, maximum_shift + 1):
        if shift < 0:
            anchor_slice = slice(-shift, len(anchor_rows))
            neighbor_slice = slice(0, len(neighbor_rows) + shift)
        elif shift > 0:
            anchor_slice = slice(0, len(anchor_rows) - shift)
            neighbor_slice = slice(shift, len(neighbor_rows))
        else:
            anchor_slice = slice(None)
            neighbor_slice = slice(None)
        similarities = np.sum(
            anchor_rows[anchor_slice] * neighbor_rows[neighbor_slice], axis=1
        )
        support = np.sqrt(
            anchor_support[anchor_slice] * neighbor_support[neighbor_slice]
        )
        support_limit = max(float(support.max()) * 0.08, 1.0e-8) if support.size else 1.0
        valid = support >= support_limit
        if not np.any(valid):
            score = 0.0
        else:
            score = float(np.average(similarities[valid], weights=support[valid]))
        candidates.append((score, shift * depth_step))
    score, lag = max(candidates, key=lambda item: (item[0], -abs(item[1])))
    return float(np.clip(score, 0.0, 1.0)), float(lag)


def _bootstrap_normal_uncertainty(
    result: dict[str, Any], repetitions: int, rng: np.random.Generator
) -> dict[str, float | int | None]:
    if repetitions <= 0:
        return {"medianDeg": None, "p90Deg": None, "samples": 0}
    usable = [needle for needle in result["needles"] if needle["inlier"]]
    if len(usable) < 6:
        return {"medianDeg": None, "p90Deg": None, "samples": 0}
    centers = np.asarray([needle["center"] for needle in usable], dtype=np.float32)
    directions = np.asarray([needle["direction"] for needle in usable], dtype=np.float32)
    weights = np.asarray(
        [needle["score"] * max(needle["robustWeight"], 0.05) for needle in usable],
        dtype=np.float32,
    )
    cube_size = float(result["cube"]["size"])
    block_coordinates = np.clip((centers / max(cube_size, 1.0) * 3.0).astype(np.int32), 0, 2)
    block_ids = block_coordinates[:, 0] + 3 * block_coordinates[:, 1] + 9 * block_coordinates[:, 2]
    blocks = [np.where(block_ids == block_id)[0] for block_id in np.unique(block_ids)]
    base_normal = np.asarray(result["normal"], dtype=np.float32)
    deviations: list[float] = []
    for _ in range(repetitions):
        chosen = rng.integers(0, len(blocks), size=len(blocks))
        indices = np.concatenate([blocks[int(block_index)] for block_index in chosen])
        if len(indices) < 6:
            continue
        sampled_normal, _, _ = _robust_common_normal(directions[indices], weights[indices])
        cosine = float(np.clip(abs(np.dot(sampled_normal, base_normal)), 0.0, 1.0))
        deviations.append(math.degrees(math.acos(cosine)))
    if not deviations:
        return {"medianDeg": None, "p90Deg": None, "samples": 0}
    values = np.asarray(deviations, dtype=np.float32)
    return {
        "medianDeg": round(float(np.median(values)), 3),
        "p90Deg": round(float(np.percentile(values, 90.0)), 3),
        "samples": len(deviations),
    }


def _profile_null_summary(
    anchor_density: np.ndarray,
    anchor_support: np.ndarray,
    neighbor_density: np.ndarray,
    neighbor_support: np.ndarray,
    observed_correlation: float,
    depth_step: float,
    maximum_lag: float,
    repetitions: int,
    rng: np.random.Generator,
) -> dict[str, float | int | None]:
    if repetitions <= 0:
        return {"median": None, "p95": None, "pValue": None, "samples": 0}
    null_scores: list[float] = []
    for _ in range(repetitions):
        order = rng.permutation(len(neighbor_density))
        score, _ = _best_profile_alignment(
            anchor_density,
            anchor_support,
            neighbor_density[order],
            neighbor_support[order],
            depth_step,
            maximum_lag,
        )
        null_scores.append(score)
    values = np.asarray(null_scores, dtype=np.float32)
    return {
        "median": round(float(np.median(values)), 4),
        "p95": round(float(np.percentile(values, 95.0)), 4),
        "pValue": round(
            float((1 + np.count_nonzero(values >= observed_correlation)) / (len(values) + 1)),
            4,
        ),
        "samples": len(values),
    }


def fit_acus_field(
    volume: VolumeData,
    request: dict[str, Any],
    *,
    _anchor: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fit a tangent-plane neighborhood and compare normals and depth profiles."""
    started = time.perf_counter()
    seed_object = request.get("seed") or {}
    anchor_seed = np.asarray(
        [int(round(float(seed_object.get(axis, -1)))) for axis in ("x", "y", "z")],
        dtype=np.int32,
    )
    spacing = int(np.clip(request.get("fieldSpacing", 8), 2, 48))
    grid_size = int(request.get("gridSize", 3))
    bootstrap_repetitions = int(np.clip(request.get("bootstrapRepetitions", 0), 0, 128))
    null_repetitions = int(np.clip(request.get("nullRepetitions", 0), 0, 128))
    if grid_size != 3:
        raise ValueError("the Acus field pilot currently uses a 3 by 3 neighborhood")
    fit_needle_length = float(
        np.clip(request.get("needleLength", 16.0), 6.0, 48.0)
    )
    fit_request = {
        "seed": {"x": int(anchor_seed[0]), "y": int(anchor_seed[1]), "z": int(anchor_seed[2])},
        "cubeSize": int(np.clip(request.get("cubeSize", 64), 16, 128)),
        "scale": float(np.clip(request.get("scale", 1.25), 0.7, 3.0)),
        "spacing": int(np.clip(request.get("spacing", 4), 3, 10)),
        "maxNeedles": int(np.clip(request.get("maxNeedles", 160), 24, 320)),
        "needleLength": fit_needle_length,
        "contextPadding": int(
            np.clip(
                request.get("contextPadding", math.ceil(fit_needle_length)), 0, 48
            )
        ),
    }
    anchor = _anchor if _anchor is not None else fit_acus(volume, fit_request)
    random_seed = int(
        (
            int(anchor_seed[0]) * 73856093
            + int(anchor_seed[1]) * 19349663
            + int(anchor_seed[2]) * 83492791
            + spacing * 2654435761
            + fit_request["cubeSize"] * 97531
        )
        % (2**32)
    )
    rng = np.random.default_rng(random_seed)
    anchor_bootstrap = _bootstrap_normal_uncertainty(
        anchor, bootstrap_repetitions, rng
    )
    anchor_normal = np.asarray(anchor["normal"], dtype=np.float32)
    anchor_u, anchor_v = _plane_basis(anchor_normal)
    profile = anchor["orientationProfile"]
    depth_centers = np.asarray(profile["depthCenters"], dtype=np.float32)
    orientation_centers = np.asarray(profile["orientationCentersDeg"], dtype=np.float32)
    depth_sigma = float(profile["stats"]["normalBandwidthVoxels"])
    angle_sigma = float(profile["stats"]["orientationBandwidthDeg"])
    anchor_density, anchor_support = _aligned_density(
        anchor,
        anchor_normal,
        anchor_u,
        anchor_v,
        depth_centers,
        orientation_centers,
        depth_sigma,
        angle_sigma,
    )
    depth_step = (
        float(abs(depth_centers[1] - depth_centers[0])) if len(depth_centers) > 1 else 1.0
    )
    maximum_lag = min(float(fit_request["cubeSize"]) * 0.22, 16.0)
    shape_xyz = np.asarray(volume.shape_xyz, dtype=np.int32)

    cell_specs: list[dict[str, Any]] = []
    for row, offset_v in enumerate((-spacing, 0, spacing)):
        for column, offset_u in enumerate((-spacing, 0, spacing)):
            requested_seed = anchor_seed + np.rint(
                anchor_u * float(offset_u) + anchor_v * float(offset_v)
            ).astype(np.int32)
            local_seed = np.clip(requested_seed, 0, shape_xyz - 1)
            is_anchor = offset_u == 0 and offset_v == 0
            cell_base = {
                "row": row,
                "column": column,
                "offsetU": offset_u,
                "offsetV": offset_v,
                "seed": {
                    "x": int(local_seed[0]),
                    "y": int(local_seed[1]),
                    "z": int(local_seed[2]),
                },
                "anchorLocalCenter": np.round(
                    np.full(3, fit_request["cubeSize"] * 0.5)
                    + local_seed.astype(np.float32)
                    - anchor_seed.astype(np.float32),
                    3,
                ).tolist(),
                "isAnchor": is_anchor,
                "overlapFraction": round(
                    float(
                        np.prod(
                            np.maximum(
                                0,
                                fit_request["cubeSize"]
                                - np.abs(local_seed - anchor_seed),
                            )
                        )
                        / float(fit_request["cubeSize"] ** 3)
                    ),
                    4,
                ),
            }
            cell_specs.append(cell_base)

    pending_contexts: list[tuple[int, np.ndarray]] = []
    precompute_errors: dict[int, str] = {}
    for index, cell_base in enumerate(cell_specs):
        if cell_base["isAnchor"]:
            continue
        try:
            _, context_data = _prepare_acus_context(
                volume, {**fit_request, "seed": cell_base["seed"]}
            )
            pending_contexts.append((index, context_data))
        except ValueError as error:
            precompute_errors[index] = str(error)

    precomputed_fields: dict[
        int, tuple[np.ndarray, np.ndarray, dict[str, Any]]
    ] = {}
    neighbor_compute_meta: dict[str, Any] | None = None
    if pending_contexts:
        fields, neighbor_compute_meta = hessian_line_fields(
            [context for _, context in pending_contexts], fit_request["scale"]
        )
        for (index, _), (score, direction) in zip(pending_contexts, fields):
            precomputed_fields[index] = (score, direction, neighbor_compute_meta)

    cells: list[dict[str, Any]] = []
    for index, cell_base in enumerate(cell_specs):
        try:
            if index in precompute_errors:
                raise ValueError(precompute_errors[index])
            result = (
                anchor
                if cell_base["isAnchor"]
                else fit_acus(
                    volume,
                    {**fit_request, "seed": cell_base["seed"]},
                    _precomputed=precomputed_fields[index],
                )
            )
            is_anchor = bool(cell_base["isAnchor"])
            normal = np.asarray(result["normal"], dtype=np.float32)
            if float(np.dot(normal, anchor_normal)) < 0.0:
                normal = -normal
            normal_angle = math.degrees(
                math.acos(float(np.clip(abs(np.dot(normal, anchor_normal)), 0.0, 1.0)))
            )
            if is_anchor:
                normal_angle, correlation, lag = 0.0, 1.0, 0.0
                normal_bootstrap = anchor_bootstrap
                null_summary = {
                    "median": None,
                    "p95": None,
                    "pValue": None,
                    "samples": 0,
                }
            else:
                neighbor_density, neighbor_support = _aligned_density(
                    result,
                    anchor_normal,
                    anchor_u,
                    anchor_v,
                    depth_centers,
                    orientation_centers,
                    depth_sigma,
                    angle_sigma,
                )
                correlation, lag = _best_profile_alignment(
                    anchor_density,
                    anchor_support,
                    neighbor_density,
                    neighbor_support,
                    depth_step,
                    maximum_lag,
                )
                normal_bootstrap = _bootstrap_normal_uncertainty(
                    result, bootstrap_repetitions, rng
                )
                null_summary = _profile_null_summary(
                    anchor_density,
                    anchor_support,
                    neighbor_density,
                    neighbor_support,
                    correlation,
                    depth_step,
                    maximum_lag,
                    null_repetitions,
                    rng,
                )
            combined_uncertainty = None
            if (
                anchor_bootstrap["p90Deg"] is not None
                and normal_bootstrap["p90Deg"] is not None
            ):
                combined_uncertainty = math.sqrt(
                    float(anchor_bootstrap["p90Deg"]) ** 2
                    + float(normal_bootstrap["p90Deg"]) ** 2
                )
            null_median = null_summary["median"]
            glyph_half_length = fit_request["cubeSize"] * 0.115
            glyph_center = np.asarray(cell_base["anchorLocalCenter"], dtype=np.float32)
            cells.append(
                {
                    **cell_base,
                    "valid": True,
                    "normal": np.round(normal, 6).tolist(),
                    "normalLine": {
                        "start": np.round(
                            glyph_center - normal * glyph_half_length, 3
                        ).tolist(),
                        "end": np.round(
                            glyph_center + normal * glyph_half_length, 3
                        ).tolist(),
                    },
                    "normalAngleDeg": round(normal_angle, 3),
                    "profileCorrelation": round(correlation, 4),
                    "bestDepthLagVoxels": round(lag, 3),
                    "normalBootstrap": normal_bootstrap,
                    "normalToUncertaintyRatio": round(
                        normal_angle / max(combined_uncertainty, 1.0e-6), 3
                    )
                    if combined_uncertainty is not None
                    else None,
                    "profileNull": null_summary,
                    "profileExcess": round(correlation - float(null_median), 4)
                    if null_median is not None
                    else None,
                    "normalConfidence": result["stats"]["normalConfidence"],
                    "twoModeCoverage": result["orientationProfile"]["stats"][
                        "meanTwoModeCoverage"
                    ],
                    "needleCount": result["stats"]["needleCount"],
                }
            )
        except ValueError as error:
            cells.append({**cell_base, "valid": False, "error": str(error)})

    neighbors = [cell for cell in cells if cell.get("valid") and not cell["isAnchor"]]
    normal_angles = np.asarray(
        [cell["normalAngleDeg"] for cell in neighbors], dtype=np.float32
    )
    correlations = np.asarray(
        [cell["profileCorrelation"] for cell in neighbors], dtype=np.float32
    )
    lags = np.asarray(
        [cell["bestDepthLagVoxels"] for cell in neighbors], dtype=np.float32
    )
    null_medians = np.asarray(
        [cell["profileNull"]["median"] for cell in neighbors if cell["profileNull"]["median"] is not None],
        dtype=np.float32,
    )
    profile_excess = np.asarray(
        [cell["profileExcess"] for cell in neighbors if cell["profileExcess"] is not None],
        dtype=np.float32,
    )
    uncertainty = np.asarray(
        [cell["normalBootstrap"]["p90Deg"] for cell in neighbors if cell["normalBootstrap"]["p90Deg"] is not None],
        dtype=np.float32,
    )
    p_values = np.asarray(
        [cell["profileNull"]["pValue"] for cell in neighbors if cell["profileNull"]["pValue"] is not None],
        dtype=np.float32,
    )
    return {
        "seed": anchor["seed"],
        "cube": anchor["cube"],
        "anchor": anchor,
        "grid": {
            "size": grid_size,
            "spacingVoxels": spacing,
            "basisU": np.round(anchor_u, 6).tolist(),
            "basisV": np.round(anchor_v, 6).tolist(),
            "normal": np.round(anchor_normal, 6).tolist(),
            "layout": "rows are -V to +V; columns are -U to +U",
        },
        "cells": cells,
        "stats": {
            "elapsedMs": round((time.perf_counter() - started) * 1000.0, 1),
            "computeBackend": (
                neighbor_compute_meta["backend"]
                if neighbor_compute_meta is not None
                else anchor["stats"]["computeBackend"]
            ),
            "computeDevice": (
                neighbor_compute_meta.get("device")
                if neighbor_compute_meta is not None
                else anchor["stats"].get("computeDevice")
            ),
            "lineFieldMs": round(
                (0.0 if _anchor is not None else float(anchor["stats"]["lineFieldMs"]))
                + (
                    float(neighbor_compute_meta["elapsedMs"])
                    if neighbor_compute_meta is not None
                    else 0.0
                ),
                1,
            ),
            "lineFieldBatchSize": (
                neighbor_compute_meta["itemCount"]
                if neighbor_compute_meta is not None
                else 0
            ),
            "lineFieldBatchLaunches": (
                (0 if _anchor is not None else int(anchor["stats"]["lineFieldBatchLaunches"]))
                + (
                    int(neighbor_compute_meta["batchLaunches"])
                    if neighbor_compute_meta is not None
                    else 0
                )
            ),
            "computeFallbackReason": (
                neighbor_compute_meta.get("fallbackReason")
                if neighbor_compute_meta is not None
                else anchor["stats"].get("computeFallbackReason")
            ),
            "anchorReused": _anchor is not None,
            "validCellCount": len(neighbors) + 1,
            "medianNormalAngleDeg": round(float(np.median(normal_angles)), 3)
            if normal_angles.size
            else None,
            "p90NormalAngleDeg": round(float(np.percentile(normal_angles, 90.0)), 3)
            if normal_angles.size
            else None,
            "medianProfileCorrelation": round(float(np.median(correlations)), 4)
            if correlations.size
            else None,
            "medianAbsoluteDepthLagVoxels": round(float(np.median(np.abs(lags))), 3)
            if lags.size
            else None,
            "medianProfileNull": round(float(np.median(null_medians)), 4)
            if null_medians.size
            else None,
            "medianProfileExcess": round(float(np.median(profile_excess)), 4)
            if profile_excess.size
            else None,
            "significantProfileFraction": round(float(np.mean(p_values <= 0.05)), 4)
            if p_values.size
            else None,
            "medianNormalBootstrapP90Deg": round(float(np.median(uncertainty)), 3)
            if uncertainty.size
            else None,
            "bootstrapRepetitions": bootstrap_repetitions,
            "nullRepetitions": null_repetitions,
            "warning": "overlapping cubes are not independent; sweep spacing before interpreting coherence",
        },
    }


def fit_acus_audit(volume: VolumeData, request: dict[str, Any]) -> dict[str, Any]:
    """Sweep neighborhood spacing and compare coherence with uncertainty and a depth null."""
    started = time.perf_counter()
    requested_spacings = request.get("fieldSpacings", [4, 8, 16, 24, 32])
    if not isinstance(requested_spacings, list):
        raise ValueError("fieldSpacings must be a list")
    spacings = sorted({int(np.clip(value, 4, 48)) for value in requested_spacings})
    if not spacings or len(spacings) > 6:
        raise ValueError("the audit requires between one and six unique field spacings")
    bootstrap_repetitions = int(np.clip(request.get("bootstrapRepetitions", 48), 12, 128))
    null_repetitions = int(np.clip(request.get("nullRepetitions", 32), 12, 128))
    fields: list[dict[str, Any]] = []
    sweeps: list[dict[str, Any]] = []
    shared_anchor: dict[str, Any] | None = None
    for spacing in spacings:
        field = fit_acus_field(
            volume,
            {
                **request,
                "gridSize": 3,
                "fieldSpacing": spacing,
                "bootstrapRepetitions": bootstrap_repetitions,
                "nullRepetitions": null_repetitions,
            },
            _anchor=shared_anchor,
        )
        if shared_anchor is None:
            shared_anchor = field["anchor"]
        fields.append(field)
        neighbors = [
            cell for cell in field["cells"] if cell.get("valid") and not cell["isAnchor"]
        ]

        def median_value(key: str) -> float | None:
            values = [cell[key] for cell in neighbors if cell.get(key) is not None]
            return round(float(np.median(np.asarray(values, dtype=np.float32))), 4) if values else None

        overlap = median_value("overlapFraction")
        sweeps.append(
            {
                "spacingVoxels": spacing,
                "validNeighborCount": len(neighbors),
                "medianOverlapFraction": overlap,
                "medianNormalAngleDeg": field["stats"]["medianNormalAngleDeg"],
                "p90NormalAngleDeg": field["stats"]["p90NormalAngleDeg"],
                "medianNormalBootstrapP90Deg": field["stats"][
                    "medianNormalBootstrapP90Deg"
                ],
                "medianNormalToUncertaintyRatio": median_value("normalToUncertaintyRatio"),
                "medianProfileCorrelation": field["stats"]["medianProfileCorrelation"],
                "medianProfileNull": field["stats"]["medianProfileNull"],
                "medianProfileExcess": field["stats"]["medianProfileExcess"],
                "significantProfileFraction": field["stats"]["significantProfileFraction"],
                "medianAbsoluteDepthLagVoxels": field["stats"][
                    "medianAbsoluteDepthLagVoxels"
                ],
                "elapsedMs": field["stats"]["elapsedMs"],
            }
        )
    return {
        "seed": fields[0]["seed"],
        "cube": fields[0]["cube"],
        "anchor": fields[0]["anchor"],
        "spacings": spacings,
        "sweeps": sweeps,
        "stats": {
            "elapsedMs": round((time.perf_counter() - started) * 1000.0, 1),
            "computeBackend": fields[0]["stats"]["computeBackend"],
            "computeDevice": fields[0]["stats"].get("computeDevice"),
            "lineFieldMs": round(
                sum(float(field["stats"]["lineFieldMs"]) for field in fields), 1
            ),
            "lineFieldBatchLaunches": sum(
                int(field["stats"]["lineFieldBatchLaunches"]) for field in fields
            ),
            "bootstrapRepetitions": bootstrap_repetitions,
            "nullRepetitions": null_repetitions,
            "nullHypothesis": "neighbor depth rows are exchangeable after local-frame transport",
            "warning": "spacing changes both overlap and sampled location; this is an exploratory coherence audit",
        },
    }


def fit_acus_padding_audit(volume: VolumeData, request: dict[str, Any]) -> dict[str, Any]:
    """Measure face bias and fit stability while increasing the real-data halo."""
    started = time.perf_counter()
    audit_needle_length = int(
        math.ceil(float(np.clip(request.get("needleLength", 16.0), 6.0, 48.0)))
    )
    half_length_halo = int(math.ceil((audit_needle_length * 0.5) / 4.0) * 4)
    larger_halo = int(
        np.clip(math.ceil((audit_needle_length * 1.5) / 4.0) * 4, 0, 48)
    )
    requested_values = request.get(
        "paddingValues", [0, half_length_halo, audit_needle_length, larger_halo]
    )
    if not isinstance(requested_values, list):
        raise ValueError("paddingValues must be a list")
    padding_values = sorted({int(np.clip(value, 0, 48)) for value in requested_values})
    if not padding_values or len(padding_values) > 6:
        raise ValueError("the padding audit requires between one and six unique halo sizes")
    results: list[tuple[int, dict[str, Any]]] = []
    failures: list[dict[str, Any]] = []
    for padding in padding_values:
        try:
            result = fit_acus(
                volume,
                {
                    **request,
                    "contextPadding": padding,
                    "allowInsufficientPadding": True,
                },
            )
            results.append((padding, result))
        except ValueError as error:
            failures.append({"requestedPadding": padding, "error": str(error)})
    if not results:
        raise ValueError("none of the requested halo sizes fit inside the loaded volume")
    reference_padding, reference = max(results, key=lambda item: item[0])
    reference_normal = np.asarray(reference["normal"], dtype=np.float32)
    reference_u, reference_v = _plane_basis(reference_normal)
    reference_profile = reference["orientationProfile"]
    depth_centers = np.asarray(reference_profile["depthCenters"], dtype=np.float32)
    orientation_centers = np.asarray(
        reference_profile["orientationCentersDeg"], dtype=np.float32
    )
    depth_sigma = float(reference_profile["stats"]["normalBandwidthVoxels"])
    angle_sigma = float(reference_profile["stats"]["orientationBandwidthDeg"])
    reference_density, reference_support = _aligned_density(
        reference,
        reference_normal,
        reference_u,
        reference_v,
        depth_centers,
        orientation_centers,
        depth_sigma,
        angle_sigma,
    )
    depth_step = (
        float(abs(depth_centers[1] - depth_centers[0])) if len(depth_centers) > 1 else 1.0
    )
    maximum_lag = min(float(reference["cube"]["size"]) * 0.22, 16.0)
    sweeps: list[dict[str, Any]] = []
    for padding, result in results:
        normal = np.asarray(result["normal"], dtype=np.float32)
        normal_angle = math.degrees(
            math.acos(float(np.clip(abs(np.dot(normal, reference_normal)), 0.0, 1.0)))
        )
        if padding == reference_padding:
            normal_angle, correlation, lag = 0.0, 1.0, 0.0
        else:
            density, support = _aligned_density(
                result,
                reference_normal,
                reference_u,
                reference_v,
                depth_centers,
                orientation_centers,
                depth_sigma,
                angle_sigma,
            )
            correlation, lag = _best_profile_alignment(
                reference_density,
                reference_support,
                density,
                support,
                depth_step,
                maximum_lag,
            )
        sweeps.append(
            {
                "requestedPadding": padding,
                "effectivePadding": result["settings"]["effectivePadding"],
                "minimumPadding": result["settings"]["minimumPadding"],
                "paddingSufficient": result["settings"]["paddingSufficient"],
                "contextSize": result["settings"]["contextSize"],
                "needleCount": result["stats"]["needleCount"],
                "boundaryNeedleCount": result["stats"]["boundaryNeedleCount"],
                "boundaryTangentialFraction": result["stats"][
                    "boundaryTangentialFraction"
                ],
                "medianAxialCoverage": result["stats"]["medianAxialCoverage"],
                "normalAngleToReferenceDeg": round(normal_angle, 3),
                "profileCorrelationToReference": round(correlation, 4),
                "bestDepthLagToReferenceVoxels": round(lag, 3),
                "normalConfidence": result["stats"]["normalConfidence"],
                "elapsedMs": result["stats"]["elapsedMs"],
            }
        )
    return {
        "seed": reference["seed"],
        "cube": reference["cube"],
        "referencePadding": reference_padding,
        "needleLength": reference["settings"]["needleLength"],
        "sweeps": sweeps,
        "failures": failures,
        "stats": {
            "elapsedMs": round((time.perf_counter() - started) * 1000.0, 1),
            "criterion": "boundary-near needle tangent to its nearest inner-cube face within 15 degrees",
            "interpretation": "stability across sufficient halos argues against crop-face orientation bias",
        },
    }
