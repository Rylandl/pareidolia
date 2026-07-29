from __future__ import annotations

import math
import os
import threading
import time
from typing import Any

import numpy as np

from .rectify import gaussian_blur_3d

try:  # Optional: the normal system Python remains a complete CPU fallback.
    import cupy as cp
except ImportError:  # pragma: no cover - exercised by the CPU-only test runtime
    cp = None


_GPU_LOCK = threading.Lock()
_GPU_FAILURE: str | None = None


def _hessian_line_field_cpu(
    data: np.ndarray, sigma: float, strength_scale: float | None = None
) -> tuple[np.ndarray, np.ndarray, float]:
    """Return a polarity-agnostic 3D line score and unsigned XYZ direction."""
    smoothed = gaussian_blur_3d(data.astype(np.float32, copy=False), sigma)
    gz, gy, gx = np.gradient(smoothed)
    hxx = np.gradient(gx, axis=2)
    hyy = np.gradient(gy, axis=1)
    hzz = np.gradient(gz, axis=0)
    hxy = 0.5 * (np.gradient(gx, axis=1) + np.gradient(gy, axis=2))
    hxz = 0.5 * (np.gradient(gx, axis=0) + np.gradient(gz, axis=2))
    hyz = 0.5 * (np.gradient(gy, axis=0) + np.gradient(gz, axis=1))

    hessian = np.empty(data.shape + (3, 3), dtype=np.float32)
    hessian[..., 0, 0] = hxx
    hessian[..., 0, 1] = hessian[..., 1, 0] = hxy
    hessian[..., 0, 2] = hessian[..., 2, 0] = hxz
    hessian[..., 1, 1] = hyy
    hessian[..., 1, 2] = hessian[..., 2, 1] = hyz
    hessian[..., 2, 2] = hzz

    eigenvalues, eigenvectors = np.linalg.eigh(hessian)
    order = np.argsort(np.abs(eigenvalues), axis=-1)
    ordered = np.take_along_axis(eigenvalues, order, axis=-1)
    direction_index = order[..., 0]
    direction = np.take_along_axis(
        eigenvectors, direction_index[..., None, None], axis=-1
    )[..., 0]
    score, resolved_scale = _line_score_numpy(ordered, strength_scale)
    return score, direction.astype(np.float32), resolved_scale


def _line_score_numpy(
    ordered: np.ndarray, strength_scale: float | None = None
) -> tuple[np.ndarray, float]:
    absolute = np.abs(ordered)
    l1, l2, l3 = absolute[..., 0], absolute[..., 1], absolute[..., 2]
    epsilon = 1.0e-7
    cross_section_balance = l2 / np.maximum(l3, epsilon)
    axial_leakage = l1 / np.sqrt(np.maximum(l2 * l3, epsilon))
    strength = np.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
    nonzero_strength = strength[strength > epsilon]
    if strength_scale is None:
        strength_scale = (
            float(np.percentile(nonzero_strength, 82.0))
            if nonzero_strength.size
            else 1.0
        )
    strength_scale = max(float(strength_scale), epsilon)
    same_polarity = ordered[..., 1] * ordered[..., 2] > 0.0
    score = (
        (1.0 - np.exp(-(cross_section_balance**2) / (2.0 * 0.52**2)))
        * np.exp(-(axial_leakage**2) / (2.0 * 0.48**2))
        * (1.0 - np.exp(-(strength**2) / (2.0 * strength_scale**2)))
        * same_polarity
    )
    return score.astype(np.float32), strength_scale


def _gaussian_blur_gpu(data: Any, sigma: float) -> Any:
    radius = max(1, int(math.ceil(2.5 * sigma)))
    coordinates = cp.arange(-radius, radius + 1, dtype=cp.float32)
    kernel = cp.exp(-(coordinates**2) / (2.0 * sigma * sigma))
    kernel /= cp.sum(kernel)
    result = data.astype(cp.float32, copy=False)
    for axis in (1, 2, 3):
        padding = [(0, 0)] * result.ndim
        padding[axis] = (radius, radius)
        padded = cp.pad(result, padding, mode="reflect")
        blurred = cp.zeros_like(result)
        width = result.shape[axis]
        for offset, weight in enumerate(kernel):
            source = [slice(None)] * result.ndim
            source[axis] = slice(offset, offset + width)
            blurred += weight * padded[tuple(source)]
        result = blurred
    return result


def _hessian_line_field_gpu_device(
    batch: np.ndarray, sigma: float, strength_scale: float | None = None
) -> tuple[Any, Any, list[float]]:
    data = cp.asarray(batch, dtype=cp.float32)
    smoothed = _gaussian_blur_gpu(data, sigma)
    gz = cp.gradient(smoothed, axis=1)
    gy = cp.gradient(smoothed, axis=2)
    gx = cp.gradient(smoothed, axis=3)
    hxx = cp.gradient(gx, axis=3)
    hyy = cp.gradient(gy, axis=2)
    hzz = cp.gradient(gz, axis=1)
    hxy = 0.5 * (cp.gradient(gx, axis=2) + cp.gradient(gy, axis=3))
    hxz = 0.5 * (cp.gradient(gx, axis=1) + cp.gradient(gz, axis=3))
    hyz = 0.5 * (cp.gradient(gy, axis=1) + cp.gradient(gz, axis=2))

    # An analytic symmetric 3x3 solve avoids cuSOLVER's multi-gigabyte batched
    # workspace. This is what lets all eight neighbor contexts share one upload.
    trace_third = (hxx + hyy + hzz) / 3.0
    axx = hxx - trace_third
    ayy = hyy - trace_third
    azz = hzz - trace_third
    p = cp.sqrt(
        cp.maximum(
            (
                axx * axx
                + ayy * ayy
                + azz * azz
                + 2.0 * (hxy * hxy + hxz * hxz + hyz * hyz)
            )
            / 6.0,
            0.0,
        )
    )
    p_safe = cp.maximum(p, 1.0e-12)
    determinant = (
        axx * (ayy * azz - hyz * hyz)
        - hxy * (hxy * azz - hyz * hxz)
        + hxz * (hxy * hyz - ayy * hxz)
    )
    phase = cp.arccos(cp.clip(determinant / (2.0 * p_safe**3), -1.0, 1.0)) / 3.0
    largest = trace_third + 2.0 * p * cp.cos(phase)
    smallest = trace_third + 2.0 * p * cp.cos(phase + 2.0 * math.pi / 3.0)
    middle = 3.0 * trace_third - smallest - largest
    eigenvalues = cp.stack([smallest, middle, largest], axis=-1).astype(cp.float32)
    order = cp.argsort(cp.abs(eigenvalues), axis=-1)
    ordered = cp.take_along_axis(eigenvalues, order, axis=-1)
    selected = ordered[..., 0]

    r0x, r0y, r0z = hxx - selected, hxy, hxz
    r1x, r1y, r1z = hxy, hyy - selected, hyz
    r2x, r2y, r2z = hxz, hyz, hzz - selected
    cross01 = cp.stack(
        [r0y * r1z - r0z * r1y, r0z * r1x - r0x * r1z, r0x * r1y - r0y * r1x],
        axis=-1,
    )
    cross02 = cp.stack(
        [r0y * r2z - r0z * r2y, r0z * r2x - r0x * r2z, r0x * r2y - r0y * r2x],
        axis=-1,
    )
    cross12 = cp.stack(
        [r1y * r2z - r1z * r2y, r1z * r2x - r1x * r2z, r1x * r2y - r1y * r2x],
        axis=-1,
    )
    candidates = cp.stack([cross01, cross02, cross12], axis=-2)
    candidate_norm2 = cp.sum(candidates * candidates, axis=-1)
    best = cp.argmax(candidate_norm2, axis=-1)
    direction = cp.take_along_axis(
        candidates, best[..., None, None], axis=-2
    )[..., 0, :]
    norm = cp.sqrt(cp.sum(direction * direction, axis=-1, keepdims=True))
    fallback = cp.zeros_like(direction)
    fallback[..., 0] = 1.0
    direction = cp.where(norm > 1.0e-12, direction / cp.maximum(norm, 1.0e-12), fallback)

    scores = []
    resolved_scales: list[float] = []
    epsilon = 1.0e-7
    for index in range(batch.shape[0]):
        values = ordered[index]
        absolute = cp.abs(values)
        l1, l2, l3 = absolute[..., 0], absolute[..., 1], absolute[..., 2]
        cross_section_balance = l2 / cp.maximum(l3, epsilon)
        axial_leakage = l1 / cp.sqrt(cp.maximum(l2 * l3, epsilon))
        strength = cp.sqrt(l1 * l1 + l2 * l2 + l3 * l3)
        nonzero_strength = strength[strength > epsilon]
        if strength_scale is None:
            local_scale = (
                cp.percentile(nonzero_strength, 82.0)
                if nonzero_strength.size
                else cp.asarray(1.0, dtype=cp.float32)
            )
            local_scale = cp.maximum(local_scale, epsilon)
            resolved_scales.append(float(local_scale.get()))
        else:
            local_scale = cp.asarray(max(float(strength_scale), epsilon), dtype=cp.float32)
            resolved_scales.append(float(strength_scale))
        same_polarity = values[..., 1] * values[..., 2] > 0.0
        score = (
            (1.0 - cp.exp(-(cross_section_balance**2) / (2.0 * 0.52**2)))
            * cp.exp(-(axial_leakage**2) / (2.0 * 0.48**2))
            * (1.0 - cp.exp(-(strength**2) / (2.0 * local_scale**2)))
            * same_polarity
        )
        scores.append(score.astype(cp.float32))
    score_batch = cp.stack(scores, axis=0)
    return score_batch, direction.astype(cp.float32), resolved_scales


def _hessian_line_field_gpu_batch(
    batch: np.ndarray, sigma: float, strength_scale: float | None = None
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    score_batch, direction_batch, resolved_scales = _hessian_line_field_gpu_device(
        batch, sigma, strength_scale
    )
    cp.cuda.get_current_stream().synchronize()
    return cp.asnumpy(score_batch), cp.asnumpy(direction_batch), resolved_scales


def _trilinear_gpu(array: Any, xyz: Any) -> Any:
    """Sample a ZYX CuPy scalar volume at (..., 3) XYZ coordinates."""
    shape = xyz.shape[:-1]
    flat = xyz.reshape(-1, 3).astype(cp.float32, copy=False)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    valid = (
        (x >= 0.0)
        & (x <= array.shape[2] - 1)
        & (y >= 0.0)
        & (y <= array.shape[1] - 1)
        & (z >= 0.0)
        & (z <= array.shape[0] - 1)
    )
    x0 = cp.clip(cp.floor(x).astype(cp.int32), 0, array.shape[2] - 1)
    y0 = cp.clip(cp.floor(y).astype(cp.int32), 0, array.shape[1] - 1)
    z0 = cp.clip(cp.floor(z).astype(cp.int32), 0, array.shape[0] - 1)
    x1 = cp.minimum(x0 + 1, array.shape[2] - 1)
    y1 = cp.minimum(y0 + 1, array.shape[1] - 1)
    z1 = cp.minimum(z0 + 1, array.shape[0] - 1)
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
    return cp.where(valid, values, 0.0).reshape(shape)


def _gpu_block_candidates(
    score: Any,
    core_local_zyx: tuple[int, int, int, int, int, int],
    core_global_start_zyx: tuple[int, int, int],
    spacing: int,
    halo: int,
    bin_size: int,
    bin_shape_zyx: tuple[int, int, int],
    maximum_per_bin: int,
    threshold: float,
) -> tuple[Any, Any, Any]:
    lz0, lz1, ly0, ly1, lx0, lx1 = core_local_zyx
    core = score[lz0:lz1, ly0:ly1, lx0:lx1]
    original_shape = tuple(int(value) for value in core.shape)
    block_shape = tuple(int(math.ceil(value / spacing)) for value in original_shape)
    padded_shape = tuple(value * spacing for value in block_shape)
    if padded_shape != original_shape:
        padded = cp.full(padded_shape, -cp.inf, dtype=cp.float32)
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
    flat = cp.argmax(blocks, axis=-1)
    values = cp.take_along_axis(blocks, flat[..., None], axis=-1)[..., 0]
    dz = flat // (spacing * spacing)
    dy = (flat // spacing) % spacing
    dx = flat % spacing
    bz, by, bx = cp.indices(block_shape, dtype=cp.int32)
    global_start = cp.asarray(core_global_start_zyx, dtype=cp.int32)
    global_z = global_start[0] + bz * spacing + dz
    global_y = global_start[1] + by * spacing + dy
    global_x = global_start[2] + bx * spacing + dx
    valid = (
        (values >= threshold)
        & (global_z < global_start[0] + original_shape[0])
        & (global_y < global_start[1] + original_shape[1])
        & (global_x < global_start[2] + original_shape[2])
    )
    values = values[valid].astype(cp.float32)
    global_points = cp.stack(
        [global_z[valid], global_y[valid], global_x[valid]], axis=1
    ).astype(cp.int32)
    local_points = global_points - global_start[None, :] + cp.asarray(
        [lz0, ly0, lx0], dtype=cp.int32
    )[None, :]
    block_starts = cp.stack(
        [
            global_start[0] + bz[valid] * spacing,
            global_start[1] + by[valid] * spacing,
            global_start[2] + bx[valid] * spacing,
        ],
        axis=1,
    )
    bin_zyx = (block_starts - halo) // bin_size
    bin_ids = (
        (bin_zyx[:, 0] * bin_shape_zyx[1] + bin_zyx[:, 1])
        * bin_shape_zyx[2]
        + bin_zyx[:, 2]
    ).astype(cp.int64)
    if not int(values.size):
        return values, local_points, bin_ids
    order = cp.lexsort((-values, bin_ids))
    ordered_bins = bin_ids[order]
    group_start = cp.concatenate(
        [cp.ones(1, dtype=cp.bool_), ordered_bins[1:] != ordered_bins[:-1]]
    )
    positions = cp.arange(len(order), dtype=cp.int64)
    starts = cp.maximum.accumulate(cp.where(group_start, positions, 0))
    selected = order[(positions - starts) < maximum_per_bin]
    return values[selected], local_points[selected], bin_ids[selected]


def _gpu_refine_needles(
    score: Any,
    direction_field: Any,
    values: Any,
    points_zyx: Any,
    radius: int,
    needle_length: float,
    cross_section_radius: float,
) -> dict[str, Any]:
    if not int(values.size):
        return {
            "indices": cp.empty(0, dtype=cp.int64),
            "center": cp.empty((0, 3), dtype=cp.float32),
            "direction": cp.empty((0, 3), dtype=cp.float32),
            "score": cp.empty(0, dtype=cp.float32),
            "axialCoverage": cp.empty(0, dtype=cp.float32),
            "supportScore": cp.empty(0, dtype=cp.float32),
        }
    axis = cp.arange(-radius, radius + 1, dtype=cp.int32)
    dz, dy, dx = cp.meshgrid(axis, axis, axis, indexing="ij")
    distance2 = dx * dx + dy * dy + dz * dz
    offsets_zyx = cp.stack([dz, dy, dx], axis=-1).reshape(-1, 3)
    offsets_xyz = offsets_zyx[:, ::-1].astype(cp.float32)
    spatial_weight = cp.exp(
        -distance2.reshape(-1).astype(cp.float32)
        / max(2.0 * (radius * 0.78) ** 2, 1.0)
    ) * (distance2.reshape(-1) <= radius * radius)
    local_score = score[
        points_zyx[:, None, 0] + offsets_zyx[None, :, 0],
        points_zyx[:, None, 1] + offsets_zyx[None, :, 1],
        points_zyx[:, None, 2] + offsets_zyx[None, :, 2],
    ]
    weights = local_score * spatial_weight[None, :]
    weight_sum = cp.sum(weights, axis=1)
    safe_weight = cp.maximum(weight_sum, 1.0e-6)
    coordinates = points_zyx[:, None, ::-1].astype(cp.float32) + offsets_xyz[None]
    center = cp.sum(coordinates * weights[:, :, None], axis=1) / safe_weight[:, None]
    centered = coordinates - center[:, None]
    covariance = cp.matmul(
        cp.transpose(centered * weights[:, :, None], (0, 2, 1)), centered
    ) / safe_weight[:, None, None]
    eigenvalues, eigenvectors = cp.linalg.eigh(covariance)
    direction = eigenvectors[:, :, 2]
    reference = direction_field[
        points_zyx[:, 0], points_zyx[:, 1], points_zyx[:, 2]
    ]
    direction *= cp.where(cp.sum(direction * reference, axis=1) < 0.0, -1.0, 1.0)[:, None]
    total = cp.maximum(cp.sum(eigenvalues, axis=1), 1.0e-6)
    linearity = cp.clip((eigenvalues[:, 2] - eigenvalues[:, 1]) / total, 0.0, 1.0)
    geometry_valid = (weight_sum > 1.0e-6) & (linearity >= 0.035)
    geometry_indices = cp.flatnonzero(geometry_valid)
    selected_direction = direction[geometry_indices]
    selected_center = center[geometry_indices]
    selected_response = values[geometry_indices]
    basis_axis = cp.zeros_like(selected_direction)
    basis_axis[:, 2] = 1.0
    basis_axis[cp.abs(selected_direction[:, 2]) > 0.86] = cp.asarray(
        [0.0, 1.0, 0.0], dtype=cp.float32
    )
    u_axis = cp.cross(selected_direction, basis_axis)
    u_axis /= cp.maximum(cp.linalg.norm(u_axis, axis=1, keepdims=True), 1.0e-7)
    v_axis = cp.cross(selected_direction, u_axis)
    v_axis /= cp.maximum(cp.linalg.norm(v_axis, axis=1, keepdims=True), 1.0e-7)
    lateral = cross_section_radius * 0.55
    sample_offsets = cp.stack(
        [cp.zeros_like(u_axis), u_axis * lateral, -u_axis * lateral, v_axis * lateral, -v_axis * lateral],
        axis=1,
    )
    axial_count = max(9, int(math.ceil(needle_length)) + 1)
    axial_offsets = cp.linspace(
        -needle_length * 0.5, needle_length * 0.5, axial_count, dtype=cp.float32
    )
    axial_points = selected_center[:, None, None] + (
        axial_offsets[None, :, None, None] * selected_direction[:, None, None]
    )
    axial_response = cp.max(
        _trilinear_gpu(score, axial_points + sample_offsets[:, None]), axis=2
    )
    supported = axial_response >= cp.maximum(0.008, selected_response[:, None] * 0.12)
    axial_coverage = cp.mean(supported, axis=1)
    current_run = cp.zeros(len(geometry_indices), dtype=cp.int16)
    longest_run = cp.zeros(len(geometry_indices), dtype=cp.int16)
    for column in supported.T:
        current_run = cp.where(column, current_run + 1, 0)
        longest_run = cp.maximum(longest_run, current_run)
    longest_fraction = longest_run.astype(cp.float32) / supported.shape[1]
    support_score = cp.mean(
        cp.clip(
            axial_response / cp.maximum(selected_response[:, None] * 0.65, 1.0e-6),
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
    confidence = cp.clip(
        selected_response
        * (0.38 + 1.4 * linearity[geometry_indices])
        * (0.48 + 0.52 * support_score),
        0.0,
        1.0,
    )
    keep = cp.flatnonzero(support_valid)
    return {
        "indices": geometry_indices[keep],
        "center": selected_center[keep].astype(cp.float32),
        "direction": selected_direction[keep].astype(cp.float32),
        "score": confidence[keep].astype(cp.float32),
        "axialCoverage": axial_coverage[keep].astype(cp.float32),
        "supportScore": support_score[keep].astype(cp.float32),
    }


def extract_needles_gpu(
    data: np.ndarray,
    *,
    sigma: float,
    strength_scale: float,
    core_local_zyx: tuple[int, int, int, int, int, int],
    core_global_start_zyx: tuple[int, int, int],
    spacing: int,
    halo: int,
    bin_size: int,
    bin_shape_zyx: tuple[int, int, int],
    maximum_per_bin: int,
    radius: int,
    needle_length: float,
    cross_section_radius: float,
    threshold: float = 0.015,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Keep a slab tile resident on CUDA through candidate and needle extraction."""
    ready, device_name, fallback_reason = _gpu_ready()
    if not ready:
        raise RuntimeError(f"GPU needle extraction unavailable: {fallback_reason}")
    timings: dict[str, float] = {}
    started = time.perf_counter()
    with _GPU_LOCK:
        field_started = time.perf_counter()
        score_batch, direction_batch, _ = _hessian_line_field_gpu_device(
            data[None], sigma, strength_scale
        )
        score = score_batch[0]
        direction = direction_batch[0]
        cp.cuda.get_current_stream().synchronize()
        timings["lineFieldMs"] = (time.perf_counter() - field_started) * 1000.0
        candidate_started = time.perf_counter()
        values, points, bin_ids = _gpu_block_candidates(
            score,
            core_local_zyx,
            core_global_start_zyx,
            spacing,
            halo,
            bin_size,
            bin_shape_zyx,
            maximum_per_bin,
            threshold,
        )
        refined = _gpu_refine_needles(
            score,
            direction,
            values,
            points,
            radius,
            needle_length,
            cross_section_radius,
        )
        cp.cuda.get_current_stream().synchronize()
        timings["candidateAndRefineMs"] = (time.perf_counter() - candidate_started) * 1000.0
        transfer_started = time.perf_counter()
        selected_indices = refined.pop("indices")
        output = {key: cp.asnumpy(value) for key, value in refined.items()}
        output["binId"] = cp.asnumpy(bin_ids[selected_indices])
        timings["resultTransferMs"] = (time.perf_counter() - transfer_started) * 1000.0
        del score_batch, direction_batch, score, direction
    timings["totalMs"] = (time.perf_counter() - started) * 1000.0
    return output, {
        "backend": "gpu-resident",
        "device": device_name,
        "fallbackReason": None,
        "candidateCount": int(len(values)),
        "refinedCount": int(len(output["score"])),
        "timingsMs": {key: round(value, 3) for key, value in timings.items()},
    }


def _gpu_ready() -> tuple[bool, str | None, str | None]:
    global _GPU_FAILURE
    mode = os.environ.get("ACUS_COMPUTE", "auto").strip().lower()
    if mode == "cpu":
        return False, None, "disabled by ACUS_COMPUTE=cpu"
    if cp is None:
        return False, None, "CuPy is not installed"
    if _GPU_FAILURE is not None and mode != "gpu":
        return False, None, _GPU_FAILURE
    try:
        if cp.cuda.runtime.getDeviceCount() < 1:
            return False, None, "no CUDA device is available"
        device_name = cp.cuda.runtime.getDeviceProperties(0)["name"]
        if isinstance(device_name, bytes):
            device_name = device_name.decode("utf-8", errors="replace")
        return True, str(device_name), None
    except Exception as error:  # pragma: no cover - depends on host driver state
        _GPU_FAILURE = f"CUDA initialization failed: {error}"
        return False, None, _GPU_FAILURE


def hessian_line_fields(
    data_cubes: list[np.ndarray], sigma: float, strength_scale: float | None = None
) -> tuple[list[tuple[np.ndarray, np.ndarray]], dict[str, Any]]:
    """Compute one or more equal-shaped line fields with GPU batching when available."""
    if not data_cubes:
        return [], {
            "backend": "cpu",
            "device": None,
            "itemCount": 0,
            "batchLaunches": 0,
            "elapsedMs": 0.0,
            "fallbackReason": None,
            "strengthScales": [],
        }
    shape = data_cubes[0].shape
    if any(cube.shape != shape for cube in data_cubes):
        raise ValueError("batched Acus contexts must have the same shape")
    started = time.perf_counter()
    ready, device_name, fallback_reason = _gpu_ready()
    if ready:
        maximum_voxels = max(
            int(os.environ.get("ACUS_GPU_BATCH_VOXELS", "8000000")),
            int(np.prod(shape)),
        )
        items_per_batch = max(1, maximum_voxels // int(np.prod(shape)))
        scores: list[np.ndarray] = []
        directions: list[np.ndarray] = []
        strength_scales: list[float] = []
        launches = 0
        try:
            with _GPU_LOCK:
                for start in range(0, len(data_cubes), items_per_batch):
                    chunk = np.stack(
                        data_cubes[start : start + items_per_batch], axis=0
                    ).astype(np.float32, copy=False)
                    score_batch, direction_batch, chunk_scales = _hessian_line_field_gpu_batch(
                        chunk, sigma, strength_scale
                    )
                    scores.extend(score_batch)
                    directions.extend(direction_batch)
                    strength_scales.extend(chunk_scales)
                    launches += 1
                    cp.get_default_memory_pool().free_all_blocks()
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            return list(zip(scores, directions)), {
                "backend": "gpu",
                "device": device_name,
                "itemCount": len(data_cubes),
                "batchLaunches": launches,
                "elapsedMs": round(elapsed_ms, 1),
                "fallbackReason": None,
                "strengthScales": strength_scales,
            }
        except Exception as error:  # pragma: no cover - host/GPU dependent
            global _GPU_FAILURE
            _GPU_FAILURE = f"GPU line field failed: {error}"
            fallback_reason = _GPU_FAILURE
            try:
                cp.get_default_memory_pool().free_all_blocks()
            except Exception:
                pass
            if os.environ.get("ACUS_COMPUTE", "auto").strip().lower() == "gpu":
                raise RuntimeError(_GPU_FAILURE) from error

    cpu_results = [
        _hessian_line_field_cpu(cube, sigma, strength_scale) for cube in data_cubes
    ]
    fields = [(score, direction) for score, direction, _ in cpu_results]
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return fields, {
        "backend": "cpu",
        "device": None,
        "itemCount": len(data_cubes),
        "batchLaunches": len(data_cubes),
        "elapsedMs": round(elapsed_ms, 1),
        "fallbackReason": fallback_reason,
        "strengthScales": [scale for _, _, scale in cpu_results],
    }


def compute_status() -> dict[str, Any]:
    ready, device_name, reason = _gpu_ready()
    return {
        "backend": "gpu" if ready else "cpu",
        "device": device_name,
        "fallbackReason": reason,
    }
