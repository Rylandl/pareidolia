from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import (
    ComponentMesh,
    SurfaceChart,
    rasterize_chart,
    sample_depth_stack,
)
from .needle_surface import (
    BLOCK_NEEDLE_SURFACE_SCHEMA,
    BLOCK_NEEDLE_SURFACE_STEM,
)
from .needle_topology import _load_field_artifact


BLOCK_NEEDLE_FLATTENING_SCHEMA = "pareidolia.block-acus-needle-flattening"
BLOCK_NEEDLE_FLATTENING_VERSION = 1
BLOCK_NEEDLE_FLATTENING_STEM = "block-needle-flattening-v1"


def _load_surface_artifact(
    surface_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    root = Path(surface_root).resolve()
    manifest_path = root if root.is_file() else root / f"{BLOCK_NEEDLE_SURFACE_STEM}.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != BLOCK_NEEDLE_SURFACE_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("needle flattening requires complete block needle surfaces")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("needle-surface data hash differs from its manifest")
    required = (
        "normalSign",
        "triangleNeedle",
        "triangleSurfaceComponentId",
        "surfaceChartUV",
    )
    with np.load(data_path) as values:
        missing = set(required) - set(values.files)
        if missing:
            raise ValueError(f"needle surfaces are missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(values[name]) for name in required}
    if arrays["triangleNeedle"].ndim != 2 or arrays["triangleNeedle"].shape[1] != 3:
        raise ValueError("needle-surface triangles must have shape (N, 3)")
    if len(arrays["triangleSurfaceComponentId"]) != len(arrays["triangleNeedle"]):
        raise ValueError("surface component IDs must match triangles")
    return manifest_path, manifest, arrays


def _resolve_native_source(field_manifest: Mapping[str, Any]) -> VolumeSource:
    records: list[tuple[str, str | None, str]] = []
    for value in field_manifest["source"]["rawRoots"]:
        pipeline_path = Path(value["root"]) / "pipeline.json"
        pipeline = json.loads(pipeline_path.read_text())
        if pipeline.get("state") != "complete":
            raise ValueError("needle field references an incomplete raw-Acus root")
        source = pipeline["identity"]["source"]
        records.append(
            (
                str(source["path"]),
                source.get("metadataPath"),
                str(source["identitySha256"]),
            )
        )
    if not records or len(set(records)) != 1:
        raise ValueError("needle field raw roots do not share one native CT source")
    path, metadata_path, _identity = records[0]
    return VolumeSource.open(path, metadata_path)


def _component_mesh_and_chart(
    component_id: int,
    field_arrays: Mapping[str, np.ndarray],
    surface_arrays: Mapping[str, np.ndarray],
) -> tuple[ComponentMesh, SurfaceChart, np.ndarray, np.ndarray]:
    triangles_global = np.asarray(surface_arrays["triangleNeedle"], dtype=np.int32)
    triangle_component = np.asarray(
        surface_arrays["triangleSurfaceComponentId"], dtype=np.int32
    )
    selected = np.flatnonzero(triangle_component == component_id)
    if not len(selected):
        raise KeyError(f"surface component {component_id} is absent")
    selected_triangles = triangles_global[selected]
    nodes = np.unique(selected_triangles)
    local = {int(node): index for index, node in enumerate(nodes)}
    triangles = np.asarray(
        [tuple(local[int(node)] for node in triangle) for triangle in selected_triangles],
        dtype=np.int32,
    )
    vertex_xyz = np.asarray(field_arrays["centerXYZ"][nodes], dtype=np.float64)
    signed_normal = (
        np.asarray(field_arrays["normalXYZ"][nodes], dtype=np.float64)
        * np.asarray(surface_arrays["normalSign"][nodes], dtype=np.float64)[:, None]
    )
    triangle_normal: list[np.ndarray] = []
    for triangle in triangles:
        points = vertex_xyz[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        reference = np.sum(signed_normal[triangle], axis=0)
        if float(np.dot(normal, reference)) < 0.0:
            normal *= -1.0
        triangle_normal.append(normal)
    triangle_normal_xyz = np.asarray(triangle_normal, dtype=np.float64)
    triangle_ids = np.arange(len(triangles), dtype=np.uint64)
    mesh = ComponentMesh(
        component_id=component_id,
        patch_ids=tuple(int(value) for value in triangle_ids),
        vertex_xyz=vertex_xyz,
        polygons=tuple(tuple(int(node) for node in value) for value in triangles),
        polygon_patch_ids=triangle_ids,
        triangles=triangles,
        triangle_patch_ids=triangle_ids,
        triangle_normal_xyz=triangle_normal_xyz,
        statistics={"source": "block-global Acus needle surface"},
    )
    chart = SurfaceChart(
        np.asarray(surface_arrays["surfaceChartUV"][nodes], dtype=np.float64),
        (),
        {"solver": "block-global weighted intrinsic chord integration"},
    )
    return mesh, chart, nodes, selected


def _texture_score(image: np.ndarray, mask: np.ndarray) -> float:
    values = image[mask].astype(np.float64)
    if len(values) < 8:
        return 0.0
    contrast = float(np.percentile(values, 95) - np.percentile(values, 5))
    horizontal = mask[:, 1:] & mask[:, :-1]
    vertical = mask[1:] & mask[:-1]
    gradients: list[np.ndarray] = []
    if np.any(horizontal):
        gradients.append(
            np.abs(image[:, 1:].astype(np.float64) - image[:, :-1])[horizontal]
        )
    if np.any(vertical):
        gradients.append(
            np.abs(image[1:].astype(np.float64) - image[:-1].astype(np.float64))[vertical]
        )
    gradient = (
        float(np.mean(np.concatenate(gradients))) if gradients else 0.0
    )
    return contrast * (1.0 + gradient)


def _contrast_rgb(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    output = np.full((*image.shape, 3), (8, 12, 18), dtype=np.uint8)
    values = image[mask].astype(np.float64)
    if not len(values):
        return output
    low, high = np.percentile(values, (2, 98))
    normalized = np.clip(
        (image.astype(np.float64) - low) / max(float(high - low), 1.0),
        0.0,
        1.0,
    )
    grayscale = np.rint(255.0 * normalized).astype(np.uint8)
    output[mask] = grayscale[mask, None]
    boundary = np.zeros(mask.shape, dtype=bool)
    boundary[1:] |= mask[1:] != mask[:-1]
    boundary[:-1] |= mask[:-1] != mask[1:]
    boundary[:, 1:] |= mask[:, 1:] != mask[:, :-1]
    boundary[:, :-1] |= mask[:, :-1] != mask[:, 1:]
    output[boundary & mask] = (62, 218, 205)
    return output


def _crop_to_mask(image: np.ndarray, mask: np.ndarray, padding: int = 3) -> np.ndarray:
    rows, columns = np.nonzero(mask)
    if not len(rows):
        return image
    low_row = max(int(np.min(rows)) - padding, 0)
    high_row = min(int(np.max(rows)) + padding + 1, image.shape[0])
    low_column = max(int(np.min(columns)) - padding, 0)
    high_column = min(int(np.max(columns)) + padding + 1, image.shape[1])
    return image[low_row:high_row, low_column:high_column]


def _fit_tile(image: np.ndarray, tile_size: int) -> np.ndarray:
    output = np.full((tile_size, tile_size, 3), (8, 12, 18), dtype=np.uint8)
    scale = min(tile_size / image.shape[1], tile_size / image.shape[0])
    width = max(1, int(round(image.shape[1] * scale)))
    height = max(1, int(round(image.shape[0] * scale)))
    columns = np.minimum(
        np.floor(np.arange(width) * image.shape[1] / width).astype(int),
        image.shape[1] - 1,
    )
    rows = np.minimum(
        np.floor(np.arange(height) * image.shape[0] / height).astype(int),
        image.shape[0] - 1,
    )
    resized = image[rows[:, None], columns[None, :]]
    low_row = (tile_size - height) // 2
    low_column = (tile_size - width) // 2
    output[low_row : low_row + height, low_column : low_column + width] = resized
    return output


def _write_montage(images: list[np.ndarray], path: Path, *, columns: int = 3) -> Path:
    tile_size = 320
    gap = 8
    rows = max(1, math.ceil(len(images) / columns))
    montage = np.full(
        (
            rows * tile_size + (rows + 1) * gap,
            columns * tile_size + (columns + 1) * gap,
            3,
        ),
        (4, 8, 13),
        dtype=np.uint8,
    )
    for index, image in enumerate(images):
        row = index // columns
        column = index % columns
        low_row = gap + row * (tile_size + gap)
        low_column = gap + column * (tile_size + gap)
        montage[
            low_row : low_row + tile_size,
            low_column : low_column + tile_size,
        ] = _fit_tile(image, tile_size)
        montage[low_row : low_row + 4, low_column : low_column + tile_size] = (
            56,
            191,
            179,
        )
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(rgb_png(montage))
    temporary.replace(path)
    return path


def run_block_needle_flattening(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    maximum_components: int = 12,
    pixel_step_voxels: float = 1.0,
    maximum_pixels: int = 512,
    depth_min_voxels: float = -12.0,
    depth_max_voxels: float = 12.0,
    depth_step_voxels: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    """Flatten and native-CT sample leading block-global needle surfaces."""

    if maximum_components < 1 or pixel_step_voxels <= 0.0 or maximum_pixels < 64:
        raise ValueError("needle flattening raster settings are invalid")
    if depth_step_voxels <= 0.0 or depth_max_voxels < depth_min_voxels:
        raise ValueError("needle flattening depth range is invalid")
    started = time.monotonic()
    surface_path, surface_manifest, surface_arrays = _load_surface_artifact(
        surface_root
    )
    field_path = Path(surface_manifest["source"]["fieldManifest"])
    field_manifest_path, field_manifest, field_arrays = _load_field_artifact(field_path)
    if (
        field_manifest["identity"]["identitySha256"]
        != surface_manifest["source"]["fieldIdentitySha256"]
    ):
        raise ValueError("needle surfaces reference another block needle field")
    source = _resolve_native_source(field_manifest)
    depth_offsets = np.arange(
        depth_min_voxels,
        depth_max_voxels + 0.5 * depth_step_voxels,
        depth_step_voxels,
        dtype=np.float32,
    )
    triangle_component = surface_arrays["triangleSurfaceComponentId"]
    component_values, component_sizes = np.unique(
        triangle_component, return_counts=True
    )
    ranking = np.lexsort((component_values, -component_sizes))[:maximum_components]
    selected_components = component_values[ranking]
    identity: dict[str, Any] = {
        "schema": BLOCK_NEEDLE_FLATTENING_SCHEMA,
        "version": BLOCK_NEEDLE_FLATTENING_VERSION,
        "surface": {
            "path": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
            "identitySha256": surface_manifest["identity"]["identitySha256"],
        },
        "settings": {
            "maximumComponents": maximum_components,
            "pixelStepVoxels": pixel_step_voxels,
            "maximumPixels": maximum_pixels,
            "depthOffsetsVoxels": [float(value) for value in depth_offsets],
        },
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    manifest_path = output / f"{BLOCK_NEEDLE_FLATTENING_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity["identitySha256"]:
            raise ValueError("needle-flattening output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    montage_images: list[np.ndarray] = []
    for rank, component in enumerate(selected_components, start=1):
        mesh, chart, nodes, selected_triangles = _component_mesh_and_chart(
            int(component), field_arrays, surface_arrays
        )
        raster = rasterize_chart(
            mesh,
            chart,
            pixel_step_voxels=pixel_step_voxels,
            maximum_pixels=maximum_pixels,
        )
        stack, sampling = sample_depth_stack(source, raster, depth_offsets)
        scores = np.asarray(
            [_texture_score(image, raster.mask) for image in stack],
            dtype=np.float64,
        )
        best_index = int(np.argmax(scores))
        best_rgb = _contrast_rgb(stack[best_index], raster.mask)
        cropped = _crop_to_mask(best_rgb, raster.mask)
        preview_path = output / f"rank-{rank:02d}-component-{int(component)}.png"
        temporary_preview = preview_path.with_suffix(preview_path.suffix + ".tmp")
        temporary_preview.write_bytes(rgb_png(cropped))
        temporary_preview.replace(preview_path)
        data_path = output / f"rank-{rank:02d}-component-{int(component)}.npz"
        temporary_data = data_path.with_suffix(data_path.suffix + ".tmp")
        with temporary_data.open("wb") as handle:
            np.savez_compressed(
                handle,
                depthOffsetsVoxels=depth_offsets,
                stack=stack,
                mask=raster.mask.astype(np.uint8),
                surfaceXYZ=raster.surface_xyz,
                normalXYZ=raster.normal_xyz,
                sourceNeedle=nodes,
                sourceTriangle=selected_triangles,
                chartUV=chart.uv.astype(np.float32),
                textureScore=scores.astype(np.float32),
            )
        temporary_data.replace(data_path)
        montage_images.append(cropped)
        records.append(
            {
                "rank": rank,
                "componentId": int(component),
                "needles": len(nodes),
                "triangles": len(mesh.triangles),
                "bestDepthOffsetVoxels": float(depth_offsets[best_index]),
                "bestTextureScore": round(float(scores[best_index]), 6),
                "raster": raster.statistics,
                "sampling": sampling,
                "preview": preview_path.name,
                "data": {
                    "path": data_path.name,
                    "bytes": data_path.stat().st_size,
                    "sha256": sha256_file(data_path),
                },
            }
        )
    montage_path = _write_montage(montage_images, output / "top-surfaces-flattened.png")
    payload = {
        "schema": BLOCK_NEEDLE_FLATTENING_SCHEMA,
        "version": BLOCK_NEEDLE_FLATTENING_VERSION,
        "state": "complete",
        "identity": identity,
        "source": {
            "surfaceManifest": str(surface_path),
            "fieldManifest": str(field_manifest_path),
            "nativeCT": source.source_identity,
        },
        "components": records,
        "montage": montage_path.name,
        "timingSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(manifest_path, payload)
    return payload
