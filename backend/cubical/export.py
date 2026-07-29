from __future__ import annotations

import colorsys
import math
import struct
import zlib
from pathlib import Path

import numpy as np

from .block import SurfaceBlock


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def _rgb_png(image: np.ndarray) -> bytes:
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 3 or pixels.shape[2] != 3:
        raise ValueError("RGB PNG encoding requires an H x W x 3 array")
    height, width, _ = pixels.shape
    rows = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, 6))
        + _png_chunk(b"IEND", b"")
    )


def _draw_line(
    image: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = first
    x1, y1 = second
    count = max(int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))), 1) + 1
    x = np.rint(np.linspace(x0, x1, count)).astype(np.int32)
    y = np.rint(np.linspace(y0, y1, count)).astype(np.int32)
    valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
    image[y[valid], x[valid]] = color


def write_block_obj(block: SurfaceBlock, path: str | Path) -> Path:
    """Write the welded polygon complex as a component-grouped OBJ mesh."""

    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    observation_vertex: dict[tuple[int, object], int] = {}
    lines = [
        "# Pareidolia cubical surface block",
        f"# coordinate unit: {block.grid.coordinate_unit}",
    ]
    for vertex_index, crossing in enumerate(block.welded_crossings, start=1):
        point = crossing.point_xyz
        lines.append(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}")
        for observation in crossing.observations:
            observation_vertex[observation] = vertex_index
    component_by_patch = dict(block.component_by_patch)
    current_component: int | None = None
    for patch in sorted(
        block.patches,
        key=lambda value: (component_by_patch[value.patch_id], value.patch_id),
    ):
        component = component_by_patch[patch.patch_id]
        if component != current_component:
            lines.append(f"g component_{component}")
            current_component = component
        indices = [
            observation_vertex[(patch.patch_id, vertex.edge)]
            for vertex in patch.vertices
        ]
        lines.append("f " + " ".join(str(value) for value in indices))
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)
    return output


def write_block_projection_png(
    block: SurfaceBlock,
    path: str | Path,
    *,
    panel_size: int = 640,
    maximum_components: int = 96,
) -> Path:
    """Write deterministic XY, XZ, and YZ wireframe projections."""

    if panel_size < 128 or maximum_components <= 0:
        raise ValueError("projection size and component count must be positive")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    image = np.full((panel_size, 3 * panel_size, 3), (9, 12, 18), dtype=np.uint8)
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in block.components
    }
    selected = {
        component
        for component, _ in sorted(
            component_sizes.items(), key=lambda value: (-value[1], value[0])
        )[:maximum_components]
    }
    component_rank = {
        component: index
        for index, component in enumerate(
            sorted(selected, key=lambda value: (-component_sizes[value], value))
        )
    }
    colors = {
        component: tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.11 + 0.61803398875 * rank) % 1.0, 0.66, 0.96
            )
        )
        for component, rank in component_rank.items()
    }
    observation_point = {
        observation: np.asarray(crossing.point_xyz, dtype=np.float64)
        for crossing in block.welded_crossings
        for observation in crossing.observations
    }
    low = np.asarray(block.grid.origin_xyz, dtype=np.float64) + np.asarray(
        block.bounds.start_cell_xyz
    ) * np.asarray(block.grid.cell_size_xyz)
    high = np.asarray(block.grid.origin_xyz, dtype=np.float64) + np.asarray(
        block.bounds.stop_cell_xyz_exclusive
    ) * np.asarray(block.grid.cell_size_xyz)
    projections = ((0, 1), (0, 2), (1, 2))
    margin = max(12, panel_size // 30)

    def project(
        point: np.ndarray, panel: int, axes: tuple[int, int]
    ) -> tuple[float, float]:
        widths = np.maximum(high[list(axes)] - low[list(axes)], 1.0e-12)
        normalized = (point[list(axes)] - low[list(axes)]) / widths
        return (
            panel * panel_size + margin + normalized[0] * (panel_size - 2 * margin),
            panel_size - margin - normalized[1] * (panel_size - 2 * margin),
        )

    component_by_patch = dict(block.component_by_patch)
    for panel, axes in enumerate(projections):
        offset = panel * panel_size
        image[margin, offset + margin : offset + panel_size - margin] = (55, 62, 74)
        image[panel_size - margin, offset + margin : offset + panel_size - margin] = (
            55,
            62,
            74,
        )
        image[margin : panel_size - margin, offset + margin] = (55, 62, 74)
        image[margin : panel_size - margin, offset + panel_size - margin] = (
            55,
            62,
            74,
        )
        for patch in block.patches:
            component = component_by_patch[patch.patch_id]
            if component not in selected:
                continue
            points = [
                observation_point[(patch.patch_id, vertex.edge)]
                for vertex in patch.vertices
            ]
            for index, first in enumerate(points):
                second = points[(index + 1) % len(points)]
                _draw_line(
                    image,
                    project(first, panel, axes),
                    project(second, panel, axes),
                    colors[component],
                )
    image[:, panel_size - 1 : panel_size + 1] = (90, 98, 112)
    image[:, 2 * panel_size - 1 : 2 * panel_size + 1] = (90, 98, 112)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(_rgb_png(image))
    temporary.replace(output)
    return output
