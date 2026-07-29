from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .rectify import _trilinear, grayscale_png
from .slab_flakes import slab_flake_plane
from .slab_sheetlet_explore import SHEETLET_EXPLORE_VERSION


CARRIER_VERSION = 5
CARRIER_SCREEN_VERSION = 5


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _axial_reference(vectors: np.ndarray) -> np.ndarray:
    matrix = np.einsum("ni,nj->ij", vectors, vectors)
    _, eigenvectors = np.linalg.eigh(matrix)
    reference = np.asarray(eigenvectors[:, -1], dtype=np.float32)
    dominant = int(np.argmax(np.abs(reference)))
    if reference[dominant] < 0.0:
        reference = -reference
    return reference / max(float(np.linalg.norm(reference)), 1.0e-8)


def _carrier_frame(flakes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    normals = np.asarray([flake["normal"] for flake in flakes], dtype=np.float32)
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    normal = _axial_reference(normals)
    signed = normals.copy()
    signed[(signed @ normal) < 0.0] *= -1.0
    fiber = _axial_reference(fibers)
    fiber -= normal * float(np.dot(fiber, normal))
    if float(np.linalg.norm(fiber)) < 0.2:
        centered = centers - np.mean(centers, axis=0)
        tangent = centered - (centered @ normal)[:, None] * normal[None, :]
        _, _, vh = np.linalg.svd(tangent, full_matrices=False)
        fiber = np.asarray(vh[0], dtype=np.float32)
        fiber -= normal * float(np.dot(fiber, normal))
    u_axis = fiber / max(float(np.linalg.norm(fiber)), 1.0e-8)
    v_axis = np.cross(normal, u_axis)
    v_axis /= max(float(np.linalg.norm(v_axis)), 1.0e-8)
    origin = np.average(
        centers,
        axis=0,
        weights=np.maximum(
            np.asarray([flake["quality"] for flake in flakes], dtype=np.float32),
            0.02,
        ),
    ).astype(np.float32)
    return {
        "origin": origin,
        "normal": normal,
        "uAxis": u_axis,
        "vAxis": v_axis,
        "signedNormals": signed,
    }


def _mls_carrier(
    flakes: list[dict[str, Any]],
    pixel_step: float = 2.0,
    bandwidth: float = 48.0,
    support_radius: float = 48.0,
    maximum_pixels: int = 512,
) -> dict[str, Any]:
    frame = _carrier_frame(flakes)
    centers = np.asarray([flake["center"] for flake in flakes], dtype=np.float32)
    quality = np.maximum(
        np.asarray([flake["quality"] for flake in flakes], dtype=np.float32), 0.02
    )
    signed_normals = frame["signedNormals"]
    fibers = np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32)
    signed_fibers = fibers.copy()
    signed_fibers[(signed_fibers @ frame["uAxis"]) < 0.0] *= -1.0
    relative = centers - frame["origin"]
    node_u = relative @ frame["uAxis"]
    node_v = relative @ frame["vAxis"]
    node_h = relative @ frame["normal"]
    normal_denominator = np.maximum(signed_normals @ frame["normal"], 0.2)
    gradient_u = -(signed_normals @ frame["uAxis"]) / normal_denominator
    gradient_v = -(signed_normals @ frame["vAxis"]) / normal_denominator
    margin = 8.0
    u_low, u_high = float(np.min(node_u) - margin), float(np.max(node_u) + margin)
    v_low, v_high = float(np.min(node_v) - margin), float(np.max(node_v) + margin)
    step = max(
        float(pixel_step),
        (u_high - u_low) / max(maximum_pixels - 1, 1),
        (v_high - v_low) / max(maximum_pixels - 1, 1),
    )
    u_values = np.arange(u_low, u_high + step * 0.5, step, dtype=np.float32)
    v_values = np.arange(v_low, v_high + step * 0.5, step, dtype=np.float32)
    grid_u, grid_v = np.meshgrid(u_values, v_values)
    query_u = grid_u.ravel()
    query_v = grid_v.ravel()
    height = np.empty(len(query_u), dtype=np.float32)
    normals = np.empty((len(query_u), 3), dtype=np.float32)
    fiber_grid = np.empty((len(query_u), 3), dtype=np.float32)
    minimum_distance = np.empty(len(query_u), dtype=np.float32)
    total_weight = np.empty(len(query_u), dtype=np.float32)
    for start in range(0, len(query_u), 4096):
        stop = min(start + 4096, len(query_u))
        delta_u = query_u[start:stop, None] - node_u[None, :]
        delta_v = query_v[start:stop, None] - node_v[None, :]
        distance2 = delta_u**2 + delta_v**2
        weights = np.exp(-0.5 * distance2 / (bandwidth**2)) * quality[None, :]
        weights[distance2 > (2.75 * bandwidth) ** 2] = 0.0
        sums = np.maximum(np.sum(weights, axis=1), 1.0e-8)
        predicted_height = (
            node_h[None, :]
            + delta_u * gradient_u[None, :]
            + delta_v * gradient_v[None, :]
        )
        height[start:stop] = np.sum(weights * predicted_height, axis=1) / sums
        blended = weights @ signed_normals
        blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1.0e-8)
        normals[start:stop] = blended
        blended_fiber = weights @ signed_fibers
        blended_fiber -= (
            np.sum(blended_fiber * blended, axis=1, keepdims=True) * blended
        )
        blended_fiber /= np.maximum(
            np.linalg.norm(blended_fiber, axis=1, keepdims=True), 1.0e-8
        )
        fiber_grid[start:stop] = blended_fiber
        minimum_distance[start:stop] = np.sqrt(np.min(distance2, axis=1))
        total_weight[start:stop] = sums
    mask = (minimum_distance <= support_radius) & (total_weight > 1.0e-5)
    surface = (
        frame["origin"][None, :]
        + query_u[:, None] * frame["uAxis"][None, :]
        + query_v[:, None] * frame["vAxis"][None, :]
        + height[:, None] * frame["normal"][None, :]
    )
    shape = grid_u.shape
    surface = surface.reshape(*shape, 3)
    normal_grid = normals.reshape(*shape, 3)
    fiber_grid = fiber_grid.reshape(*shape, 3)
    surface[~mask.reshape(shape)] = np.nan
    normal_grid[~mask.reshape(shape)] = np.nan
    fiber_grid[~mask.reshape(shape)] = np.nan

    # Evaluate the same carrier at its supporting flakes to expose construction
    # error without turning this exploratory stage into another holdout test.
    fitted_node_height = np.empty(len(flakes), dtype=np.float32)
    fitted_node_normal = np.empty((len(flakes), 3), dtype=np.float32)
    for start in range(0, len(flakes), 512):
        stop = min(start + 512, len(flakes))
        delta_u = node_u[start:stop, None] - node_u[None, :]
        delta_v = node_v[start:stop, None] - node_v[None, :]
        distance2 = delta_u**2 + delta_v**2
        weights = np.exp(-0.5 * distance2 / (bandwidth**2)) * quality[None, :]
        weights[distance2 > (2.75 * bandwidth) ** 2] = 0.0
        sums = np.maximum(np.sum(weights, axis=1), 1.0e-8)
        prediction = (
            node_h[None, :]
            + delta_u * gradient_u[None, :]
            + delta_v * gradient_v[None, :]
        )
        fitted_node_height[start:stop] = np.sum(weights * prediction, axis=1) / sums
        blended = weights @ signed_normals
        blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1.0e-8)
        fitted_node_normal[start:stop] = blended
    height_residual = np.abs(fitted_node_height - node_h)
    normal_residual = np.degrees(
        np.arccos(
            np.clip(
                np.abs(np.sum(fitted_node_normal * signed_normals, axis=1)), 0.0, 1.0
            )
        )
    )
    normal_families = np.asarray(
        [int(flake.get("normalFamily", 0)) for flake in flakes], dtype=np.uint8
    )
    family_stats = {}
    for family_index in np.unique(normal_families):
        family_mask = normal_families == family_index
        family_stats[str(int(family_index))] = {
            "flakeCount": int(np.count_nonzero(family_mask)),
            "medianQuality": round(float(np.median(quality[family_mask])), 4),
            "medianNodeHeightResidualVoxels": round(
                float(np.median(height_residual[family_mask])), 3
            ),
            "p90NodeHeightResidualVoxels": round(
                float(np.percentile(height_residual[family_mask], 90)), 3
            ),
            "medianNodeNormalResidualDeg": round(
                float(np.median(normal_residual[family_mask])), 3
            ),
            "p90NodeNormalResidualDeg": round(
                float(np.percentile(normal_residual[family_mask], 90)), 3
            ),
        }
    return {
        "uValues": u_values,
        "vValues": v_values,
        "surfaceXYZ": surface.astype(np.float32),
        "normalXYZ": normal_grid.astype(np.float32),
        "fiberXYZ": fiber_grid.astype(np.float32),
        "supportMask": mask.reshape(shape),
        "frame": frame,
        "nodeHeightResidualVoxels": height_residual.astype(np.float32),
        "nodeNormalResidualDeg": normal_residual.astype(np.float32),
        "stats": {
            "pixelStepVoxels": round(step, 4),
            "shapeYX": [int(shape[0]), int(shape[1])],
            "supportedPixelCount": int(np.count_nonzero(mask)),
            "supportedPixelFraction": round(float(np.mean(mask)), 4),
            "medianNodeHeightResidualVoxels": round(float(np.median(height_residual)), 3),
            "p90NodeHeightResidualVoxels": round(float(np.percentile(height_residual, 90)), 3),
            "medianNodeNormalResidualDeg": round(float(np.median(normal_residual)), 3),
            "p90NodeNormalResidualDeg": round(float(np.percentile(normal_residual, 90)), 3),
            "normalFamilies": family_stats,
        },
    }


def build_mls_carrier(
    flakes: list[dict[str, Any]],
    pixel_step: float = 2.0,
    bandwidth: float = 48.0,
    support_radius: float = 48.0,
    maximum_pixels: int = 512,
) -> dict[str, Any]:
    """Build the active moving-tangent-plane carrier for supplied flakes."""
    return _mls_carrier(
        flakes,
        pixel_step=pixel_step,
        bandwidth=bandwidth,
        support_radius=support_radius,
        maximum_pixels=maximum_pixels,
    )


def _contrast(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    values = np.asarray(image, dtype=np.float32)
    valid = values[mask]
    if not len(valid):
        return np.zeros(values.shape, dtype=np.uint8)
    low, high = np.percentile(valid, [1.0, 99.5])
    if high <= low + 1.0e-6:
        high = low + 1.0
    output = np.clip((values - low) / (high - low) * 255.0, 0.0, 255.0)
    output[~mask] = 0.0
    return output.astype(np.uint8)


def _plane_texture(
    image: np.ndarray, mask: np.ndarray, block_size: int = 16
) -> dict[str, float | int | None]:
    values = np.asarray(image, dtype=np.float32)
    valid_values = values[mask]
    if len(valid_values) < 64:
        return {
            "blockCount": 0,
            "medianLocalCoherence": 0.0,
            "p90LocalCoherence": 0.0,
            "orientationConcentration": 0.0,
            "dominantFiberAngleDeg": None,
            "normalizedGradientEnergy": 0.0,
            "textureScore": 0.0,
        }
    mean = float(np.mean(valid_values))
    deviation = max(float(np.std(valid_values)), 1.0)
    normalized = (values - mean) / deviation
    gradient_x = np.zeros_like(normalized)
    gradient_y = np.zeros_like(normalized)
    gradient_x[:, 1:-1] = 0.5 * (normalized[:, 2:] - normalized[:, :-2])
    gradient_y[1:-1, :] = 0.5 * (normalized[2:, :] - normalized[:-2, :])
    gradient_mask = mask.copy()
    gradient_mask[:, 0] = gradient_mask[:, -1] = False
    gradient_mask[0, :] = gradient_mask[-1, :] = False
    gradient_mask[:, 1:-1] &= mask[:, :-2] & mask[:, 2:]
    gradient_mask[1:-1, :] &= mask[:-2, :] & mask[2:, :]
    coherence_values: list[float] = []
    energy_values: list[float] = []
    angles: list[float] = []
    for y0 in range(0, values.shape[0], block_size):
        for x0 in range(0, values.shape[1], block_size):
            block_mask = gradient_mask[
                y0 : y0 + block_size, x0 : x0 + block_size
            ]
            if int(np.count_nonzero(block_mask)) < max(24, block_mask.size // 4):
                continue
            gx = gradient_x[y0 : y0 + block_size, x0 : x0 + block_size][block_mask]
            gy = gradient_y[y0 : y0 + block_size, x0 : x0 + block_size][block_mask]
            jxx = float(np.mean(gx * gx))
            jyy = float(np.mean(gy * gy))
            jxy = float(np.mean(gx * gy))
            energy = jxx + jyy
            if energy <= 1.0e-8:
                continue
            coherence = math.sqrt((jxx - jyy) ** 2 + 4.0 * jxy**2) / energy
            gradient_angle = 0.5 * math.atan2(2.0 * jxy, jxx - jyy)
            coherence_values.append(float(np.clip(coherence, 0.0, 1.0)))
            energy_values.append(energy)
            angles.append(gradient_angle)
    if not coherence_values:
        return {
            "blockCount": 0,
            "medianLocalCoherence": 0.0,
            "p90LocalCoherence": 0.0,
            "orientationConcentration": 0.0,
            "dominantFiberAngleDeg": None,
            "normalizedGradientEnergy": 0.0,
            "textureScore": 0.0,
        }
    coherence_array = np.asarray(coherence_values, dtype=np.float32)
    energy_array = np.asarray(energy_values, dtype=np.float32)
    angle_array = np.asarray(angles, dtype=np.float32)
    axial = np.sum(energy_array * np.exp(2.0j * angle_array))
    concentration = float(abs(axial) / max(float(np.sum(energy_array)), 1.0e-8))
    dominant_gradient = 0.5 * math.atan2(float(np.imag(axial)), float(np.real(axial)))
    dominant_fiber = (math.degrees(dominant_gradient) + 90.0) % 180.0
    median_coherence = float(np.median(coherence_array))
    texture_score = median_coherence * (0.75 + 0.25 * concentration)
    return {
        "blockCount": len(coherence_values),
        "medianLocalCoherence": round(median_coherence, 4),
        "p90LocalCoherence": round(float(np.percentile(coherence_array, 90)), 4),
        "orientationConcentration": round(concentration, 4),
        "dominantFiberAngleDeg": round(dominant_fiber, 3),
        "normalizedGradientEnergy": round(float(np.median(energy_array)), 4),
        "textureScore": round(texture_score, 4),
    }


def _texture_profile(
    stack: np.ndarray, mask: np.ndarray, depth_offsets: np.ndarray
) -> dict[str, Any]:
    planes = []
    for depth, image in zip(depth_offsets, stack):
        planes.append({"depthOffsetVoxels": float(depth), **_plane_texture(image, mask)})
    scores = np.asarray([float(value["textureScore"]) for value in planes])
    best_index = int(np.argmax(scores))
    center_index = int(np.argmin(np.abs(depth_offsets)))
    return {
        "bestDepthOffsetVoxels": float(depth_offsets[best_index]),
        "bestTextureScore": round(float(scores[best_index]), 4),
        "centerTextureScore": round(float(scores[center_index]), 4),
        "medianTextureScoreAcrossDepth": round(float(np.median(scores)), 4),
        "depthPeakSharpness": round(float(scores[best_index] - np.median(scores)), 4),
        "bestPlane": planes[best_index],
        "centerPlane": planes[center_index],
        "planes": planes,
    }


def _sample_stack(
    source: np.ndarray,
    carrier: dict[str, Any],
    depth_offsets: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    surface = carrier["surfaceXYZ"]
    normals = carrier["normalXYZ"]
    mask = carrier["supportMask"]
    valid_surface = surface[mask]
    valid_normals = normals[mask]
    extreme = np.concatenate(
        [
            valid_surface + float(depth_offsets[0]) * valid_normals,
            valid_surface + float(depth_offsets[-1]) * valid_normals,
        ],
        axis=0,
    )
    low = np.floor(np.min(extreme, axis=0) - 2.0).astype(int)
    high = np.ceil(np.max(extreme, axis=0) + 3.0).astype(int)
    low = np.maximum(low, 0)
    high = np.minimum(high, np.asarray([source.shape[2], source.shape[1], source.shape[0]]))
    x0, y0, z0 = (int(value) for value in low)
    x1, y1, z1 = (int(value) for value in high)
    subvolume = np.asarray(source[z0:z1, y0:y1, x0:x1], dtype=np.float32)
    local_origin = np.asarray([x0, y0, z0], dtype=np.float32)
    stack = np.zeros((len(depth_offsets), *mask.shape), dtype=np.uint8)
    for depth_index, depth in enumerate(depth_offsets):
        points = valid_surface + float(depth) * valid_normals - local_origin
        values = _trilinear(subvolume, points)
        plane = np.zeros(mask.shape, dtype=np.float32)
        plane[mask] = values
        stack[depth_index] = np.clip(np.rint(plane), 0.0, 255.0).astype(np.uint8)
    return stack, {
        "sourceBoundsXYZ": [x0, x1, y0, y1, z0, z1],
        "sourceSubvolumeShapeZYX": [z1 - z0, y1 - y0, x1 - x0],
        "sourceSubvolumeMiB": round(float(subvolume.nbytes / (1024**2)), 2),
    }


def _carrier_yield(
    carrier_stats: dict[str, Any], texture: dict[str, Any]
) -> dict[str, float]:
    supported_area = (
        int(carrier_stats["supportedPixelCount"])
        * float(carrier_stats["pixelStepVoxels"]) ** 2
    )
    fit_factor = math.exp(
        -0.5
        * (
            (float(carrier_stats["medianNodeHeightResidualVoxels"]) / 4.0) ** 2
            + (float(carrier_stats["medianNodeNormalResidualDeg"]) / 10.0) ** 2
        )
    )
    construction_yield = (
        supported_area
        * fit_factor
        * (0.35 + 0.65 * float(texture["bestTextureScore"]))
    )
    return {
        "supportedAreaSquareVoxels": round(supported_area, 2),
        "fitFactor": round(fit_factor, 4),
        "constructionYieldScore": round(construction_yield, 2),
    }


def _load_carrier_catalog(
    root: Path,
) -> tuple[Path, dict[str, Any], np.ndarray, list[dict[str, Any]]]:
    analysis = json.loads((root / "analysis.json").read_text())
    source_path = Path(analysis["identity"]["source"])
    candidate_payload = json.loads(
        (
            root
            / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-candidates.json"
        ).read_text()
    )
    with np.load(
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz"
    ) as payload:
        component = np.asarray(payload["component"], dtype=np.uint32)
    grid = json.loads((root / "grid.json").read_text())
    planes = [
        slab_flake_plane(root, z_index, 3)
        for z_index in range(len(grid["z"]))
    ]
    flakes = [
        flake
        for plane in planes
        for flake in plane["flakes"]
        if float(flake["quality"]) >= 0.08
    ]
    if len(flakes) != len(component):
        raise ValueError("the exploratory component catalog does not match the flake planes")
    return source_path, candidate_payload, component, flakes


def _compact_texture(texture: dict[str, Any]) -> dict[str, Any]:
    return {
        key: texture[key]
        for key in (
            "bestDepthOffsetVoxels",
            "bestTextureScore",
            "centerTextureScore",
            "medianTextureScoreAcrossDepth",
            "depthPeakSharpness",
            "bestPlane",
            "centerPlane",
        )
    }


def screen_sheetlet_carriers(
    output_root: str | Path,
    force: bool = False,
    candidate_limit: int | None = None,
    checkpoint_every: int = 50,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    """Coarsely flatten and score every substantial sheetlet without artifacts."""
    root = Path(output_root)
    source_path, candidate_payload, component, flakes = _load_carrier_catalog(root)
    all_candidates = candidate_payload["candidates"]
    resolved_count = len(all_candidates)
    if candidate_limit is not None:
        resolved_count = int(np.clip(candidate_limit, 1, resolved_count))
    candidates = all_candidates[:resolved_count]
    settings = {
        "candidateCount": resolved_count,
        "carrier": "coarse quality-weighted moving tangent-plane blend",
        "pixelStepVoxels": 4.0,
        "bandwidthVoxels": 48.0,
        "supportRadiusVoxels": 48.0,
        "maximumPixelsPerAxis": 192,
        "normalDepthRangeVoxels": [-12.0, 12.0],
        "normalDepthStepVoxels": 2.0,
    }
    identity = {
        "version": CARRIER_SCREEN_VERSION,
        "sheetletIdentity": candidate_payload.get("identity"),
        "sheetletThreshold": candidate_payload["settings"]["selectedThreshold"],
        "source": str(source_path),
        "sourceMtimeNs": source_path.stat().st_mtime_ns,
    }
    screen_path = root / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json"
    checkpoint_path = root / (
        f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}-checkpoint.json"
    )
    if screen_path.is_file() and not force:
        cached = json.loads(screen_path.read_text())
        if cached.get("identity") == identity and cached.get("settings") == settings:
            return cached

    outputs: list[dict[str, Any]] = []
    elapsed_before = 0.0
    if checkpoint_path.is_file() and not force:
        checkpoint = json.loads(checkpoint_path.read_text())
        if checkpoint.get("identity") == identity and checkpoint.get("settings") == settings:
            outputs = checkpoint.get("candidates", [])
            elapsed_before = float(checkpoint.get("stats", {}).get("elapsedMs", 0.0))
    if len(outputs) > resolved_count:
        outputs = []
        elapsed_before = 0.0

    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 2.0, dtype=np.float32)
    started = time.monotonic()

    def elapsed_ms() -> float:
        return elapsed_before + (time.monotonic() - started) * 1000.0

    def checkpoint_payload(complete: bool) -> dict[str, Any]:
        return {
            "identity": identity,
            "settings": settings,
            "stats": {
                "elapsedMs": round(elapsed_ms(), 2),
                "candidateCount": len(outputs),
                "complete": complete,
            },
            "candidates": outputs,
        }

    for source_index in range(len(outputs), resolved_count):
        candidate = candidates[source_index]
        component_id = int(candidate["componentId"])
        member_indices = np.flatnonzero(component == component_id)
        member_flakes = [flakes[int(index)] for index in member_indices]
        carrier = _mls_carrier(
            member_flakes,
            pixel_step=4.0,
            bandwidth=48.0,
            support_radius=48.0,
            maximum_pixels=192,
        )
        stack, _ = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        yield_stats = _carrier_yield(carrier["stats"], texture)
        outputs.append(
            {
                "sourceRank": source_index + 1,
                "componentId": component_id,
                "memberCount": len(member_flakes),
                "planeCount": int(candidate["planeCount"]),
                "extentXYZ": candidate["extentXYZ"],
                "carrier": carrier["stats"],
                "texture": _compact_texture(texture),
                "yield": yield_stats,
            }
        )
        completed = source_index + 1
        if completed % max(checkpoint_every, 1) == 0 or completed == resolved_count:
            _atomic_json(checkpoint_path, checkpoint_payload(False))
            if progress is not None:
                progress(completed, resolved_count, elapsed_ms())

    yield_ranking = sorted(
        (
            {
                "sourceRank": int(value["sourceRank"]),
                "componentId": int(value["componentId"]),
                "memberCount": int(value["memberCount"]),
                "planeCount": int(value["planeCount"]),
                "constructionYieldScore": float(
                    value["yield"]["constructionYieldScore"]
                ),
                "supportedAreaSquareVoxels": float(
                    value["yield"]["supportedAreaSquareVoxels"]
                ),
                "medianHeightResidualVoxels": float(
                    value["carrier"]["medianNodeHeightResidualVoxels"]
                ),
                "medianNormalResidualDeg": float(
                    value["carrier"]["medianNodeNormalResidualDeg"]
                ),
                "bestDepthOffsetVoxels": float(
                    value["texture"]["bestDepthOffsetVoxels"]
                ),
                "bestTextureScore": float(value["texture"]["bestTextureScore"]),
            }
            for value in outputs
        ),
        key=lambda value: float(value["constructionYieldScore"]),
        reverse=True,
    )
    for yield_rank, value in enumerate(yield_ranking, start=1):
        value["yieldRank"] = yield_rank
    result = checkpoint_payload(True)
    result["yieldRanking"] = yield_ranking
    _atomic_json(screen_path, result)
    _atomic_json(checkpoint_path, result)
    return result


def build_sheetlet_carriers(
    output_root: str | Path,
    top_count: int = 3,
    force: bool = False,
    source_ranks: list[int] | None = None,
    summary_label: str | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    source_path, candidate_payload, component, flakes = _load_carrier_catalog(root)
    all_candidates = candidate_payload["candidates"]
    if source_ranks is None:
        top_count = int(np.clip(top_count, 1, 64))
        selected = list(enumerate(all_candidates[:top_count], start=1))
        summary_label = summary_label or f"top{top_count}"
    else:
        normalized_ranks = list(dict.fromkeys(int(value) for value in source_ranks))
        if not normalized_ranks or len(normalized_ranks) > 64:
            raise ValueError("source_ranks must contain between 1 and 64 unique ranks")
        if min(normalized_ranks) < 1 or max(normalized_ranks) > len(all_candidates):
            raise ValueError("source_ranks contains a rank outside the candidate catalog")
        selected = [(rank, all_candidates[rank - 1]) for rank in normalized_ranks]
        top_count = len(selected)
        summary_label = summary_label or f"selected-{top_count}"
    artifact_root = root / f"sheetlet-carriers-v{CARRIER_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"summary-{summary_label}.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    started = time.monotonic()
    outputs = []
    for rank, candidate in selected:
        component_id = int(candidate["componentId"])
        member_indices = np.flatnonzero(component == component_id)
        member_flakes = [flakes[int(index)] for index in member_indices]
        carrier = _mls_carrier(member_flakes)
        stack, sampling_stats = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        candidate_root = artifact_root / f"rank-{rank:02d}-component-{component_id}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        geometry_path = candidate_root / "carrier.npz"
        geometry_temporary = geometry_path.with_suffix(".npz.tmp")
        with geometry_temporary.open("wb") as handle:
            np.savez_compressed(
                handle,
                uValues=carrier["uValues"],
                vValues=carrier["vValues"],
                surfaceXYZ=carrier["surfaceXYZ"],
                normalXYZ=carrier["normalXYZ"],
                fiberXYZ=carrier["fiberXYZ"],
                supportMask=carrier["supportMask"],
                memberIndex=member_indices.astype(np.uint32),
            )
        geometry_temporary.replace(geometry_path)
        stack_path = candidate_root / "depth-stack.npz"
        stack_temporary = stack_path.with_suffix(".npz.tmp")
        with stack_temporary.open("wb") as handle:
            np.savez_compressed(handle, depthOffsets=depth_offsets, intensity=stack)
        stack_temporary.replace(stack_path)
        mask = carrier["supportMask"]
        center_index = int(np.argmin(np.abs(depth_offsets)))
        center_image = _contrast(stack[center_index], mask)
        maximum_image = _contrast(np.max(stack, axis=0), mask)
        best_index = int(np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"])))
        best_image = _contrast(stack[best_index], mask)
        selected_depths = [4, 8, 12, 16, 20]
        montage = np.concatenate(
            [_contrast(stack[index], mask) for index in selected_depths], axis=1
        )
        (candidate_root / "center.png").write_bytes(grayscale_png(center_image))
        (candidate_root / "maximum.png").write_bytes(grayscale_png(maximum_image))
        (candidate_root / "best-texture.png").write_bytes(grayscale_png(best_image))
        (candidate_root / "depth-montage.png").write_bytes(grayscale_png(montage))
        yield_stats = _carrier_yield(carrier["stats"], texture)
        output = {
            "rank": rank,
            "componentId": component_id,
            "memberCount": len(member_flakes),
            "candidate": candidate,
            "carrier": carrier["stats"],
            "sampling": {
                **sampling_stats,
                "depthOffsetsVoxels": depth_offsets.tolist(),
                "centerNonzeroMean": round(float(np.mean(stack[center_index][mask])), 3),
                "maximumNonzeroMean": round(float(np.mean(np.max(stack, axis=0)[mask])), 3),
            },
            "texture": texture,
            "yield": yield_stats,
            "frame": {
                key: np.round(carrier["frame"][key], 6).tolist()
                for key in ("origin", "normal", "uAxis", "vAxis")
            },
            "artifacts": {
                "geometry": str(geometry_path.relative_to(root)),
                "depthStack": str(stack_path.relative_to(root)),
                "centerImage": str((candidate_root / "center.png").relative_to(root)),
                "maximumImage": str((candidate_root / "maximum.png").relative_to(root)),
                "bestTextureImage": str((candidate_root / "best-texture.png").relative_to(root)),
                "depthMontage": str((candidate_root / "depth-montage.png").relative_to(root)),
            },
        }
        _atomic_json(candidate_root / "summary.json", output)
        outputs.append(output)
        del stack
    yield_ranking = sorted(
        (
            {
                "sourceRank": int(value["rank"]),
                "componentId": int(value["componentId"]),
                "memberCount": int(value["memberCount"]),
                "constructionYieldScore": float(value["yield"]["constructionYieldScore"]),
                "supportedAreaSquareVoxels": float(value["yield"]["supportedAreaSquareVoxels"]),
                "medianHeightResidualVoxels": float(
                    value["carrier"]["medianNodeHeightResidualVoxels"]
                ),
                "medianNormalResidualDeg": float(
                    value["carrier"]["medianNodeNormalResidualDeg"]
                ),
                "bestDepthOffsetVoxels": float(value["texture"]["bestDepthOffsetVoxels"]),
                "bestTextureScore": float(value["texture"]["bestTextureScore"]),
                "bestTextureImage": value["artifacts"]["bestTextureImage"],
            }
            for value in outputs
        ),
        key=lambda value: float(value["constructionYieldScore"]),
        reverse=True,
    )
    for rank, value in enumerate(yield_ranking, start=1):
        value["yieldRank"] = rank
    result = {
        "identity": {
            "version": CARRIER_VERSION,
            "sheetletIdentity": candidate_payload.get("identity"),
            "sheetletThreshold": candidate_payload["settings"]["selectedThreshold"],
            "source": str(source_path),
            "sourceMtimeNs": source_path.stat().st_mtime_ns,
        },
        "settings": {
            "topCount": top_count,
            "sourceRanks": [rank for rank, _ in selected],
            "carrier": "quality-weighted moving blend of local tangent-plane predictions",
            "pixelStepVoxels": 2.0,
            "bandwidthVoxels": 48.0,
            "supportRadiusVoxels": 48.0,
            "normalDepthRangeVoxels": [-12.0, 12.0],
            "normalDepthStepVoxels": 1.0,
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "candidateCount": len(outputs),
        },
        "yieldRanking": yield_ranking,
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
