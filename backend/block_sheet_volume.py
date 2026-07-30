from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHEET_ROOT = (
    PROJECT_ROOT
    / "work/multiseam-2x2-b00c03c/sheet-halo-core-12x12x10-v1/halo-1"
    / "owned-bp32-c10-u010-collision-cut-curvature-v1"
)
DEFAULT_VOLUME_PATH = Path(
    "/mnt/t5/acus-cross-scroll/pherc0358-z7168-d512-yfull-xfull.npy"
)


def configured_sheet_root() -> Path:
    return Path(os.environ.get("PAREIDOLIA_BLOCK_SHEET_ROOT", DEFAULT_SHEET_ROOT))


def configured_volume_path() -> Path:
    return Path(os.environ.get("PAREIDOLIA_BLOCK_VOLUME", DEFAULT_VOLUME_PATH))


def _required(values: Any, name: str) -> np.ndarray:
    if name not in values:
        raise ValueError(f"block sheet artifact is missing {name}")
    return np.asarray(values[name])


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


@lru_cache(maxsize=4)
def _load_block_sheet_payload(root_value: str) -> dict[str, Any]:
    root = Path(root_value)
    patch_manifest_path = root / "selected-patches-v1.json"
    patch_data_path = root / "selected-patches-v1.npz"
    graph_data_path = root / "surface-graph-v1.npz"
    summary_path = root / "summary.json"
    if not patch_manifest_path.is_file() or not patch_data_path.is_file():
        raise FileNotFoundError(f"block sheet geometry is unavailable at {root}")
    if not graph_data_path.is_file():
        raise FileNotFoundError(f"block sheet graph is unavailable at {graph_data_path}")

    manifest = json.loads(patch_manifest_path.read_text())
    with np.load(patch_data_path, allow_pickle=False) as stored:
        patch_id = _required(stored, "patchId").astype(np.uint64, copy=False)
        cell_xyz = _required(stored, "cellXYZ").astype(np.int32, copy=False)
        confidence = _required(stored, "confidence").astype(np.float64, copy=False)
        normal_xyz = _required(stored, "normalXYZ").astype(np.float64, copy=False)
        vertex_offset = _required(stored, "vertexOffset").astype(np.int64, copy=False)
        vertex_axis = _required(stored, "vertexEdgeAxis").astype(np.int64, copy=False)
        vertex_anchor = _required(stored, "vertexEdgeAnchor").astype(np.float64, copy=False)
        vertex_t = _required(stored, "vertexT").astype(np.float64, copy=False)

    with np.load(graph_data_path, allow_pickle=False) as stored:
        graph_patch_id = _required(stored, "patchId").astype(np.uint64, copy=False)
        graph_component_id = _required(stored, "componentId").astype(np.uint64, copy=False)
        retained_join_count = len(_required(stored, "firstPatchId"))

    if len(patch_id) != len(cell_xyz) or len(vertex_offset) != len(patch_id) + 1:
        raise ValueError("block sheet patch arrays have inconsistent lengths")
    if len(graph_patch_id) != len(graph_component_id):
        raise ValueError("block sheet component arrays have inconsistent lengths")

    component_by_patch = {
        int(current_patch): int(current_component)
        for current_patch, current_component in zip(graph_patch_id, graph_component_id)
    }
    try:
        patch_component = np.asarray(
            [component_by_patch[int(value)] for value in patch_id], dtype=np.uint64
        )
    except KeyError as exc:
        raise ValueError(f"surface graph omits selected patch {exc.args[0]}") from exc

    component_values, component_counts = np.unique(patch_component, return_counts=True)
    ranked_components = sorted(
        (
            (int(component), int(count))
            for component, count in zip(component_values, component_counts)
        ),
        key=lambda value: (-value[1], value[0]),
    )
    rank_by_component = {
        component: rank for rank, (component, _count) in enumerate(ranked_components, 1)
    }

    grid = manifest.get("grid", {})
    shape_cells = np.asarray(grid.get("shapeCellsXYZ", ()), dtype=np.int64)
    cell_size = np.asarray(grid.get("cellSizeXYZ", ()), dtype=np.float64)
    origin = np.asarray(grid.get("originXYZ", ()), dtype=np.float64)
    if shape_cells.shape != (3,) or cell_size.shape != (3,) or origin.shape != (3,):
        raise ValueError("block sheet grid must define three-dimensional shape, size, and origin")
    extent = shape_cells.astype(np.float64) * cell_size

    component_accumulators: dict[int, dict[str, Any]] = {
        component: {
            "rank": rank_by_component[component],
            "stableId": str(component),
            "patchCount": count,
            "confidenceTotal": 0.0,
            "boundsMinimum": np.full(3, np.inf, dtype=np.float64),
            "boundsMaximum": np.full(3, -np.inf, dtype=np.float64),
        }
        for component, count in ranked_components
    }
    patches: list[dict[str, Any]] = []
    for index, current_patch_id in enumerate(patch_id):
        low = int(vertex_offset[index])
        high = int(vertex_offset[index + 1])
        vertices_grid = vertex_anchor[low:high].copy()
        for vertex_index, axis in enumerate(vertex_axis[low:high]):
            vertices_grid[vertex_index, int(axis)] += vertex_t[low + vertex_index]
        vertices_local = vertices_grid * cell_size
        component = int(patch_component[index])
        accumulator = component_accumulators[component]
        accumulator["confidenceTotal"] += float(confidence[index])
        accumulator["boundsMinimum"] = np.minimum(
            accumulator["boundsMinimum"], np.min(vertices_local, axis=0)
        )
        accumulator["boundsMaximum"] = np.maximum(
            accumulator["boundsMaximum"], np.max(vertices_local, axis=0)
        )
        patches.append(
            {
                "id": str(int(current_patch_id)),
                "component": int(accumulator["rank"]),
                "componentSize": int(accumulator["patchCount"]),
                "cell": [int(value) for value in cell_xyz[index]],
                "confidence": round(float(confidence[index]), 6),
                "normal": _round_list(normal_xyz[index], 6),
                "vertices": [_round_list(vertex) for vertex in vertices_local],
            }
        )

    components: list[dict[str, Any]] = []
    for component, _count in ranked_components:
        accumulator = component_accumulators[component]
        patch_count = int(accumulator["patchCount"])
        components.append(
            {
                "rank": int(accumulator["rank"]),
                "stableId": accumulator["stableId"],
                "patchCount": patch_count,
                "meanConfidence": round(
                    float(accumulator["confidenceTotal"]) / max(patch_count, 1), 6
                ),
                "boundsMinimumXYZ": _round_list(accumulator["boundsMinimum"]),
                "boundsMaximumXYZ": _round_list(accumulator["boundsMaximum"]),
            }
        )

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    best = summary.get("restitch", {}).get("best", {})
    curvature = summary.get("restitch", {}).get("sheetCurvatureRefinement", {})
    curvature_by_component = {
        str(value["componentId"]): value
        for value in curvature.get("after", {}).get("components", ())
    }
    for component in components:
        record = curvature_by_component.get(component["stableId"])
        if record is None:
            continue
        component["curvature"] = {
            "flaggedJoins": int(record.get("flaggedJoins", 0)),
            "maximumPressure": float(record.get("maximumPressure", 0.0)),
            "directBendP90Degrees": float(
                record.get("directBendDegrees", {}).get("p90") or 0.0
            ),
            "branchContrastP90Degrees": float(
                record.get("branchContrastDegrees", {}).get("p90") or 0.0
            ),
            "normalConeP90DegreesDiagnosticOnly": float(
                record.get("globalNormalConeDegreesDiagnosticOnly", {}).get("p90")
                or 0.0
            ),
        }
    return {
        "schema": "pareidolia.block-sheet-volume",
        "version": 1,
        "variant": root.name,
        "grid": {
            "shapeCellsXYZ": [int(value) for value in shape_cells],
            "cellSizeXYZ": _round_list(cell_size),
            "originXYZ": _round_list(origin),
            "extentXYZ": _round_list(extent),
            "coordinateUnit": str(grid.get("coordinateUnit", "source-voxel")),
        },
        "stats": {
            "patchCount": len(patches),
            "componentCount": len(components),
            "retainedJoinCount": int(retained_join_count),
            "largestComponentPatchCount": int(
                best.get(
                    "largestComponentPatchCount",
                    components[0]["patchCount"] if components else 0,
                )
            ),
            "unresolvedInteriorTraceEndpoints": int(
                best.get("unresolvedInteriorTraceEndpoints", 0)
            ),
            "retainedInteriorTraceFraction": float(
                best.get("retainedInteriorTraceFraction", 0.0)
            ),
            "curvatureFlaggedJoinsBefore": int(
                curvature.get("before", {}).get("flaggedJoins", 0)
            ),
            "curvatureFlaggedJoinsAfter": int(
                curvature.get("after", {}).get("flaggedJoins", 0)
            ),
        },
        "components": components,
        "patches": patches,
    }


def load_block_sheet_payload(root: str | Path | None = None) -> dict[str, Any]:
    selected_root = Path(root) if root is not None else configured_sheet_root()
    return _load_block_sheet_payload(str(selected_root.resolve()))


@lru_cache(maxsize=8)
def _load_block_volume(
    sheet_root_value: str,
    volume_path_value: str,
    stride: int,
) -> tuple[bytes, dict[str, Any]]:
    payload = _load_block_sheet_payload(sheet_root_value)
    grid = payload["grid"]
    block_origin = np.asarray(grid["originXYZ"], dtype=np.int64)
    extent = np.asarray(grid["extentXYZ"], dtype=np.int64)
    if np.any(extent <= 0):
        raise ValueError("block extent must be positive")

    volume_path = Path(volume_path_value)
    if not volume_path.is_file():
        raise FileNotFoundError(f"source block volume is unavailable at {volume_path}")
    sidecar_path = volume_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    source_origin = np.asarray(sidecar.get("originXYZ", (0, 0, 0)), dtype=np.int64)
    if source_origin.shape != (3,):
        raise ValueError("source volume origin must contain three coordinates")

    source = np.load(volume_path, mmap_mode="r")
    if source.ndim != 3:
        raise ValueError("source block volume must be a three-dimensional ZYX array")
    low_xyz = block_origin - source_origin
    high_xyz = low_xyz + extent
    shape_xyz = np.asarray((source.shape[2], source.shape[1], source.shape[0]))
    if np.any(low_xyz < 0) or np.any(high_xyz > shape_xyz):
        raise ValueError(
            "sheet block lies outside the configured source volume: "
            f"local bounds {low_xyz.tolist()}–{high_xyz.tolist()}, "
            f"source shape {shape_xyz.tolist()}"
        )

    x0, y0, z0 = (int(value) for value in low_xyz)
    x1, y1, z1 = (int(value) for value in high_xyz)
    sampled = np.asarray(source[z0:z1:stride, y0:y1:stride, x0:x1:stride])
    if sampled.dtype != np.uint8:
        values = sampled.astype(np.float32)
        sample = values[::4, ::4, ::4]
        low, high = np.percentile(sample, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        sampled_u8 = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(
            np.uint8
        )
    else:
        sampled_u8 = np.ascontiguousarray(sampled)

    distribution_sample = sampled_u8[::4, ::4, ::4]
    percentiles = np.percentile(distribution_sample, (1.0, 50.0, 90.0, 99.0))
    shape_sampled_xyz = [
        int(sampled_u8.shape[2]),
        int(sampled_u8.shape[1]),
        int(sampled_u8.shape[0]),
    ]
    metadata = {
        "shapeXYZ": shape_sampled_xyz,
        "stride": int(stride),
        "originXYZ": [int(value) for value in block_origin],
        "extentXYZ": [int(value) for value in extent],
        "percentiles": _round_list(percentiles, 2),
        "source": str(sidecar.get("name", volume_path.name)),
    }
    return sampled_u8.tobytes(order="C"), metadata


def load_block_volume(
    *,
    sheet_root: str | Path | None = None,
    volume_path: str | Path | None = None,
    stride: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    stride = int(stride)
    if stride not in (1, 2, 3, 4):
        raise ValueError("block volume stride must be one of 1, 2, 3, or 4")
    selected_sheet_root = Path(sheet_root) if sheet_root is not None else configured_sheet_root()
    selected_volume_path = Path(volume_path) if volume_path is not None else configured_volume_path()
    return _load_block_volume(
        str(selected_sheet_root.resolve()),
        str(selected_volume_path.resolve()),
        stride,
    )
