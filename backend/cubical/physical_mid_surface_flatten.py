from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import ComponentMesh, SurfaceChart, rasterize_chart, sample_depth_stack
from .needle_flatten import (
    _contrast_rgb,
    _crop_to_mask,
    _texture_score,
    _write_montage,
)
from .physical_mid_surface import PHYSICAL_MID_SURFACE_SCHEMA
from .physical_mid_surface_mesh import (
    PHYSICAL_MID_SURFACE_MESH_SCHEMA,
    PHYSICAL_MID_SURFACE_MESH_STEM,
)


PHYSICAL_MID_SURFACE_FLATTEN_SCHEMA = "pareidolia.physical-mid-surface-flattening"
PHYSICAL_MID_SURFACE_FLATTEN_VERSION = 1
PHYSICAL_MID_SURFACE_FLATTEN_STEM = "physical-mid-surface-flattening-v1"


def _resolve_mesh(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{PHYSICAL_MID_SURFACE_MESH_STEM}.json"


def _load_npz(path: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"data hash differs from manifest for {path}")
    with np.load(data_path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


def _load_mesh_and_mid_surface(
    root: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    mesh_path = _resolve_mesh(root)
    mesh = json.loads(mesh_path.read_text())
    if (
        mesh.get("schema") != PHYSICAL_MID_SURFACE_MESH_SCHEMA
        or mesh.get("state") != "complete"
    ):
        raise ValueError("flattening requires a complete physical mid-surface mesh")
    mesh_arrays = _load_npz(mesh_path, mesh)
    mid_record = mesh["identity"]["midSurface"]
    mid_path = Path(str(mid_record["manifestPath"])).resolve()
    if sha256_file(mid_path) != mid_record["manifestSha256"]:
        raise ValueError("mid-surface manifest changed after meshing")
    mid = json.loads(mid_path.read_text())
    if (
        mid.get("schema") != PHYSICAL_MID_SURFACE_SCHEMA
        or mid.get("state") != "complete"
        or mid["data"]["sha256"] != mid_record["dataSha256"]
    ):
        raise ValueError("mesh and physical mid-surface identities disagree")
    mid_arrays = _load_npz(mid_path, mid)
    required_mesh = (
        "signedNormalXYZ",
        "chartUV",
        "triangleNode",
        "triangleMeshComponentId",
        "trianglePhysicalSheetLabel",
    )
    required_mid = ("midpointXYZ", "thicknessVoxels", "nodeKind")
    missing = set(required_mesh) - set(mesh_arrays)
    missing |= set(required_mid) - set(mid_arrays)
    if missing:
        raise ValueError(f"mid-surface flattening is missing arrays: {sorted(missing)}")
    return mesh_path, mesh, mesh_arrays, mid_path, mid, mid_arrays


def _percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0, "median": None, "p90": None, "maximum": None}
    quantiles = np.percentile(finite, (50, 90, 100))
    return {
        "count": int(len(finite)),
        **{
            name: round(float(value), 6)
            for name, value in zip(("median", "p90", "maximum"), quantiles)
        },
    }


def _mesh_and_chart(
    group_id: int,
    selected_triangle: np.ndarray,
    midpoint_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    chart_uv: np.ndarray,
    triangle_node: np.ndarray,
) -> tuple[ComponentMesh, SurfaceChart, np.ndarray]:
    global_triangles = np.asarray(triangle_node, dtype=np.int32)[selected_triangle]
    nodes = np.unique(global_triangles)
    local_triangles = np.searchsorted(nodes, global_triangles).astype(np.int32)
    vertex_xyz = np.asarray(midpoint_xyz, dtype=np.float64)[nodes]
    signed_normal = np.asarray(signed_normal_xyz, dtype=np.float64)[nodes]
    triangle_normal: list[np.ndarray] = []
    for triangle in local_triangles:
        points = vertex_xyz[triangle]
        normal = np.cross(points[1] - points[0], points[2] - points[0])
        normal /= max(float(np.linalg.norm(normal)), 1.0e-12)
        reference = np.sum(signed_normal[triangle], axis=0)
        if float(np.dot(normal, reference)) < 0.0:
            normal *= -1.0
        triangle_normal.append(normal)
    triangle_ids = np.asarray(selected_triangle, dtype=np.uint64)
    mesh = ComponentMesh(
        component_id=int(group_id),
        patch_ids=tuple(int(value) for value in triangle_ids),
        vertex_xyz=vertex_xyz,
        polygons=tuple(tuple(int(node) for node in value) for value in local_triangles),
        polygon_patch_ids=triangle_ids,
        triangles=local_triangles,
        triangle_patch_ids=triangle_ids,
        triangle_normal_xyz=np.asarray(triangle_normal, dtype=np.float64),
        statistics={"source": "physical air-papyrus-air midpoint mesh"},
    )
    chart = SurfaceChart(
        np.asarray(chart_uv, dtype=np.float64)[nodes],
        (),
        {"solver": "robust parallel-transported midpoint graph chart"},
    )
    return mesh, chart, nodes


def _chart_metric_distortion(
    mesh: ComponentMesh,
    chart: SurfaceChart,
) -> dict[str, Any]:
    length_ratio: list[float] = []
    for triangle in mesh.triangles:
        for index, first in enumerate(triangle):
            second = triangle[(index + 1) % 3]
            xyz_length = float(
                np.linalg.norm(mesh.vertex_xyz[second] - mesh.vertex_xyz[first])
            )
            uv_length = float(np.linalg.norm(chart.uv[second] - chart.uv[first]))
            length_ratio.append(uv_length / max(xyz_length, 1.0e-12))
    first_uv = chart.uv[mesh.triangles[:, 1]] - chart.uv[mesh.triangles[:, 0]]
    second_uv = chart.uv[mesh.triangles[:, 2]] - chart.uv[mesh.triangles[:, 0]]
    chart_area = 0.5 * np.abs(
        first_uv[:, 0] * second_uv[:, 1]
        - first_uv[:, 1] * second_uv[:, 0]
    )
    first_xyz = (
        mesh.vertex_xyz[mesh.triangles[:, 1]]
        - mesh.vertex_xyz[mesh.triangles[:, 0]]
    )
    second_xyz = (
        mesh.vertex_xyz[mesh.triangles[:, 2]]
        - mesh.vertex_xyz[mesh.triangles[:, 0]]
    )
    physical_area = 0.5 * np.linalg.norm(np.cross(first_xyz, second_xyz), axis=1)
    return {
        "chartToPhysicalEdgeLengthRatio": _percentiles(
            np.asarray(length_ratio)
        ),
        "chartToPhysicalAreaRatio": _percentiles(
            chart_area / np.maximum(physical_area, 1.0e-12)
        ),
    }


def run_physical_mid_surface_flattening(
    mesh_root: str | Path,
    output_root: str | Path,
    *,
    grouping: str = "source-component",
    maximum_components: int = 12,
    pixel_step_voxels: float = 0.75,
    maximum_pixels: int = 640,
    depth_min_voxels: float = -8.0,
    depth_max_voxels: float = 8.0,
    depth_step_voxels: float = 1.0,
    force: bool = False,
) -> dict[str, Any]:
    """Flatten leading midpoint meshes and sample their native CT texture."""

    if grouping not in ("source-component", "mesh-component"):
        raise ValueError(
            "mid-surface grouping must be source-component or mesh-component"
        )
    if maximum_components < 1 or pixel_step_voxels <= 0.0 or maximum_pixels < 64:
        raise ValueError("mid-surface flattening raster settings are invalid")
    if depth_step_voxels <= 0.0 or depth_max_voxels < depth_min_voxels:
        raise ValueError("mid-surface flattening depth range is invalid")
    started = time.monotonic()
    (
        mesh_path,
        mesh_manifest,
        mesh_arrays,
        mid_path,
        mid_manifest,
        mid_arrays,
    ) = _load_mesh_and_mid_surface(mesh_root)
    depth_offsets = np.arange(
        depth_min_voxels,
        depth_max_voxels + 0.5 * depth_step_voxels,
        depth_step_voxels,
        dtype=np.float32,
    )
    triangle_mesh_component = np.asarray(
        mesh_arrays["triangleMeshComponentId"], dtype=np.int32
    )
    triangle_source_component = np.asarray(
        mesh_arrays["triangleSourceComponentId"], dtype=np.int32
    )
    triangle_component = (
        triangle_source_component
        if grouping == "source-component"
        else triangle_mesh_component
    )
    component_value, component_count = np.unique(
        triangle_component, return_counts=True
    )
    order = np.lexsort((component_value, -component_count))
    selected_components = component_value[order][:maximum_components]
    identity: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_FLATTEN_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_FLATTEN_VERSION,
        "mesh": {
            "manifestPath": str(mesh_path),
            "manifestSha256": sha256_file(mesh_path),
            "dataSha256": mesh_manifest["data"]["sha256"],
        },
        "settings": {
            "grouping": grouping,
            "maximumComponents": maximum_components,
            "pixelStepVoxels": pixel_step_voxels,
            "maximumPixels": maximum_pixels,
            "depthOffsetsVoxels": [float(value) for value in depth_offsets],
            "interiorDepthLimit": "0.45 times component median physical thickness",
        },
        "implementationSha256": sha256_file(Path(__file__)),
        "rasterImplementationSha256": sha256_file(
            Path(rasterize_chart.__code__.co_filename)
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_manifest = output / f"{PHYSICAL_MID_SURFACE_FLATTEN_STEM}.json"
    if output_manifest.is_file() and not force:
        cached = json.loads(output_manifest.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
        ):
            return cached

    source = VolumeSource.open(
        mid_manifest["source"]["path"],
        mid_manifest["source"].get("metadataPath"),
    )
    midpoint = np.asarray(mid_arrays["midpointXYZ"], dtype=np.float64)
    thickness = np.asarray(mid_arrays["thicknessVoxels"], dtype=np.float64)
    node_kind = np.asarray(mid_arrays["nodeKind"], dtype=np.uint8)
    signed_normal = np.asarray(mesh_arrays["signedNormalXYZ"], dtype=np.float64)
    chart_uv = np.asarray(mesh_arrays["chartUV"], dtype=np.float64)
    triangle_node = np.asarray(mesh_arrays["triangleNode"], dtype=np.int32)
    triangle_label = np.asarray(
        mesh_arrays["trianglePhysicalSheetLabel"], dtype=np.int32
    )
    records: list[dict[str, Any]] = []
    montage_images: list[np.ndarray] = []
    for rank, component in enumerate(selected_components, start=1):
        selected_triangle = np.flatnonzero(triangle_component == component)
        mesh, chart, nodes = _mesh_and_chart(
            int(component),
            selected_triangle,
            midpoint,
            signed_normal,
            chart_uv,
            triangle_node,
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
        median_thickness = float(np.median(thickness[nodes]))
        interior = np.abs(depth_offsets) <= 0.45 * median_thickness
        if not np.any(interior):
            interior[np.argmin(np.abs(depth_offsets))] = True
        interior_index = np.flatnonzero(interior)
        best_index = int(interior_index[np.argmax(scores[interior])])
        best_rgb = _contrast_rgb(stack[best_index], raster.mask)
        cropped = _crop_to_mask(best_rgb, raster.mask)
        file_group = grouping.replace("-", "_")
        preview_path = (
            output / f"rank-{rank:02d}-{file_group}-{int(component)}.png"
        )
        temporary_preview = preview_path.with_suffix(preview_path.suffix + ".tmp")
        temporary_preview.write_bytes(rgb_png(cropped))
        temporary_preview.replace(preview_path)
        data_path = (
            output / f"rank-{rank:02d}-{file_group}-{int(component)}.npz"
        )
        temporary_data = data_path.with_suffix(data_path.suffix + ".tmp")
        with temporary_data.open("wb") as handle:
            np.savez_compressed(
                handle,
                depthOffsetsVoxels=depth_offsets,
                stack=stack,
                mask=raster.mask.astype(np.uint8),
                overlapMask=raster.overlap_mask.astype(np.uint8),
                surfaceXYZ=raster.surface_xyz,
                normalXYZ=raster.normal_xyz,
                sourceMidpointNode=nodes,
                sourceTriangle=selected_triangle,
                chartUV=chart.uv.astype(np.float32),
                textureScore=scores.astype(np.float32),
            )
        temporary_data.replace(data_path)
        montage_images.append(cropped)
        physical_labels = np.unique(triangle_label[selected_triangle])
        if len(physical_labels) != 1:
            raise RuntimeError("one flattened mesh spans physical sheet identities")
        source_values = np.unique(triangle_source_component[selected_triangle])
        records.append(
            {
                "rank": rank,
                "grouping": grouping,
                "groupId": int(component),
                "sourceComponentId": (
                    int(source_values[0]) if len(source_values) == 1 else None
                ),
                "meshIslandCount": int(
                    len(np.unique(triangle_mesh_component[selected_triangle]))
                ),
                "physicalSheetLabel": int(physical_labels[0]),
                "nodes": int(len(nodes)),
                "directProfileNodes": int(np.count_nonzero(node_kind[nodes] == 0)),
                "contextualAdoptedProfileNodes": int(
                    np.count_nonzero(node_kind[nodes] == 2)
                ),
                "oneSidedThicknessProxyNodes": int(
                    np.count_nonzero(node_kind[nodes] == 3)
                ),
                "denseBoundaryPairNodes": int(
                    np.count_nonzero(node_kind[nodes] == 1)
                ),
                "triangles": int(len(selected_triangle)),
                "medianPhysicalThicknessVoxels": round(median_thickness, 6),
                "bestInteriorDepthOffsetVoxels": float(depth_offsets[best_index]),
                "bestInteriorTextureScore": round(float(scores[best_index]), 6),
                "raster": raster.statistics,
                "metricDistortion": _chart_metric_distortion(mesh, chart),
                "sampling": sampling,
                "preview": preview_path.name,
                "data": {
                    "path": data_path.name,
                    "bytes": data_path.stat().st_size,
                    "sha256": sha256_file(data_path),
                },
            }
        )
    montage_path = _write_montage(
        montage_images,
        output / "top-physical-mid-surface-flattened.png",
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_FLATTEN_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_FLATTEN_VERSION,
        "state": "complete",
        "identity": identity,
        "source": {
            "meshManifest": str(mesh_path),
            "midSurfaceManifest": str(mid_path),
            "nativeCT": source.source_identity,
        },
        "counts": {
            "grouping": grouping,
            "availableGroupCount": int(len(component_value)),
            "flattenedGroupCount": int(len(records)),
            "flattenedTriangleCount": int(
                sum(int(record["triangles"]) for record in records)
            ),
            "flattenedNodeCount": int(sum(int(record["nodes"]) for record in records)),
            "overlappingRasterCount": int(
                sum(record["raster"]["nonadjacentOverlapPixels"] > 0 for record in records)
            ),
        },
        "components": records,
        "artifacts": {"montage": montage_path.name},
        "method": (
            "sample native CT at fixed offsets from graph-supported intrinsic "
            "mid-surface meshes; choose previews only from the measured papyrus interior"
        ),
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output_manifest, payload)
    return payload
