from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from ..slab_flakes import FLAKE_CACHE_VERSION
from .geometry import (
    ClippedPatch,
    DegeneratePlaneIntersection,
    PlaneEstimate,
    clip_plane_to_cell,
)
from .topology import GridSpec, Int3


@dataclass(frozen=True, slots=True)
class AcusAdapterSettings:
    minimum_quality: float = 0.08
    normal_family: int = 0
    minimum_normal_standard_deviation_degrees: float = 1.0
    minimum_height_standard_deviation_voxels: float = 0.75
    minimum_fiber_standard_deviation_degrees: float = 1.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_quality <= 1.0:
            raise ValueError("minimum quality must lie in [0, 1]")
        if self.normal_family < 0:
            raise ValueError("normal family must be nonnegative")
        if (
            self.minimum_normal_standard_deviation_degrees <= 0.0
            or self.minimum_height_standard_deviation_voxels <= 0.0
            or self.minimum_fiber_standard_deviation_degrees <= 0.0
        ):
            raise ValueError("adapter uncertainty floors must be positive")


@dataclass(frozen=True, slots=True)
class AcusWindowScene:
    grid: GridSpec
    patches: tuple[ClippedPatch, ...]
    local_order_by_patch: tuple[tuple[int, int], ...]
    source_identity: dict[str, Any]
    stats: dict[str, int | float]

    @property
    def local_order_map(self) -> dict[int, int]:
        return dict(self.local_order_by_patch)


def _window_grid(
    root: Path, origin_cell_xyz: Int3, shape_cells_xyz: Int3, stride: int
) -> tuple[GridSpec, dict[str, list[int]]]:
    grid_axes = json.loads((root / "grid.json").read_text())
    selected: dict[str, list[int]] = {}
    for axis, name in enumerate(("x", "y", "z")):
        values = [int(value) for value in grid_axes[name]]
        low = int(origin_cell_xyz[axis])
        high = low + int(shape_cells_xyz[axis])
        if low < 0 or high > len(values):
            raise ValueError("Acus window lies outside the persisted grid")
        selected[name] = values[low:high]
        if any(
            selected[name][index + 1] - selected[name][index] != stride
            for index in range(len(selected[name]) - 1)
        ):
            raise ValueError(
                "selected Acus window includes an irregular terminal grid interval"
            )
    origin = tuple(float(selected[name][0] - stride * 0.5) for name in ("x", "y", "z"))
    return (
        GridSpec(
            shape_cells_xyz,
            cell_size_xyz=(float(stride),) * 3,
            origin_xyz=origin,
            coordinate_unit="voxel",
        ),
        selected,
    )


def load_acus_flake_window(
    root: str | Path,
    origin_cell_xyz: Int3,
    shape_cells_xyz: Int3,
    settings: AcusAdapterSettings | None = None,
) -> AcusWindowScene:
    """Adapt persisted Acus depth/orientation modes into geometric patch evidence.

    This is a plumbing adapter, not a declaration that legacy flakes are physical
    layers. It derives uncertainty from each mode's effective support and its
    source cell residual, clips only planes that cross the non-overlapping core
    cell, and records every exclusion.
    """

    output_root = Path(root)
    resolved = settings or AcusAdapterSettings()
    if resolved.normal_family != 0:
        raise ValueError(
            "Acus flake proxy v1 supports only the primary normal family; "
            "secondary-family uncertainty needs its own evidence adapter"
        )
    analysis = json.loads((output_root / "analysis.json").read_text())
    stride = int(analysis["identity"]["settings"]["gridStride"])
    grid, selected_axes = _window_grid(
        output_root, origin_cell_xyz, shape_cells_xyz, stride
    )
    cells = np.load(output_root / "cells.npy", mmap_mode="r")
    expected_shape = tuple(int(value) for value in analysis["gridShapeZYX"])
    if cells.shape != expected_shape:
        raise ValueError("Acus cell artifact shape disagrees with its manifest")
    depth_bandwidth: float | None = None
    source_identity: dict[str, Any] | None = None
    candidates = 0
    quality_excluded = 0
    family_excluded = 0
    misses_core = 0
    degenerate = 0
    patches: list[ClippedPatch] = []
    family_by_patch: dict[int, int] = {}
    for global_z in range(
        origin_cell_xyz[2], origin_cell_xyz[2] + shape_cells_xyz[2]
    ):
        flake_path = (
            output_root
            / f"flakes-v{FLAKE_CACHE_VERSION}-z{global_z}-k3.json"
        )
        payload = json.loads(flake_path.read_text())
        if depth_bandwidth is None:
            depth_bandwidth = float(payload["identity"]["depthBandwidthVoxels"])
            source_identity = payload["identity"]
        elif depth_bandwidth != float(payload["identity"]["depthBandwidthVoxels"]):
            raise ValueError("flake shards disagree on their depth bandwidth")
        for flake in payload["flakes"]:
            global_cell = tuple(int(value) for value in flake["cellIndex"])
            if not all(
                origin_cell_xyz[axis]
                <= global_cell[axis]
                < origin_cell_xyz[axis] + shape_cells_xyz[axis]
                for axis in range(3)
            ):
                continue
            candidates += 1
            family = int(flake.get("normalFamily", 0))
            if family != resolved.normal_family:
                family_excluded += 1
                continue
            quality = float(flake["quality"])
            if quality < resolved.minimum_quality:
                quality_excluded += 1
                continue
            local_cell = tuple(
                global_cell[axis] - origin_cell_xyz[axis] for axis in range(3)
            )
            cell_record = cells[global_cell[2], global_cell[1], global_cell[0]]
            effective_support = max(float(flake["effectiveSupport"]), 1.0)
            normal_residual = max(float(cell_record["medianPlaneResidualDeg"]), 0.0)
            normal_std_degrees = max(
                resolved.minimum_normal_standard_deviation_degrees,
                normal_residual / math.sqrt(effective_support),
            )
            height_std = max(
                resolved.minimum_height_standard_deviation_voxels,
                float(depth_bandwidth) / math.sqrt(effective_support),
                float(flake["thickness"]),
            )
            fiber_std_degrees = max(
                resolved.minimum_fiber_standard_deviation_degrees,
                float(flake["medianFiberResidualDeg"])
                / math.sqrt(effective_support),
            )
            raw_normal = np.asarray(flake["normal"], dtype=np.float64)
            raw_normal /= max(float(np.linalg.norm(raw_normal)), 1.0e-12)
            local_center = grid.cell_center_world(local_cell)
            source_cell_center = np.asarray(flake["cellCenter"], dtype=np.float64)
            if float(np.max(np.abs(source_cell_center - local_center))) > 1.0e-3:
                raise ValueError("flake cell center disagrees with persisted grid axes")
            flake_center = np.asarray(flake["center"], dtype=np.float64)
            height = float(np.dot(raw_normal, flake_center - local_center))
            normal_confidence = float(cell_record["normalConfidence"])
            confidence = float(
                np.clip(math.sqrt(max(quality * normal_confidence, 0.0)), 0.0, 1.0)
            )
            estimate = PlaneEstimate.isotropic(
                raw_normal,
                height,
                math.radians(normal_std_degrees),
                height_std,
                fiber_xyz=flake["fiber"],
                fiber_angular_std_radians=math.radians(fiber_std_degrees),
                confidence=confidence,
            )
            patch_id = (global_z << 32) | int(flake["id"])
            try:
                patch = clip_plane_to_cell(
                    grid, local_cell, estimate, patch_id=patch_id
                )
            except DegeneratePlaneIntersection:
                degenerate += 1
                continue
            if patch is None:
                misses_core += 1
                continue
            patches.append(patch)
            family_by_patch[patch_id] = family
    if depth_bandwidth is None or source_identity is None:
        raise ValueError("Acus window contains no flake shards")
    local_order: dict[int, int] = {}
    by_cell_family: dict[tuple[Int3, int], list[ClippedPatch]] = {}
    for patch in patches:
        key = (patch.cell_xyz, family_by_patch[patch.patch_id])
        by_cell_family.setdefault(key, []).append(patch)
    for values in by_cell_family.values():
        values.sort(
            key=lambda value: (
                value.estimate.height_from_cell_center,
                value.patch_id,
            )
        )
        for order, patch in enumerate(values):
            local_order[patch.patch_id] = order
    return AcusWindowScene(
        grid=grid,
        patches=tuple(sorted(patches, key=lambda value: value.patch_id)),
        local_order_by_patch=tuple(sorted(local_order.items())),
        source_identity={
            "analysis": analysis["identity"],
            "flakes": source_identity,
            "originCellXYZ": list(origin_cell_xyz),
            "shapeCellsXYZ": list(shape_cells_xyz),
            "selectedGridCentersXYZ": selected_axes,
        },
        stats={
            "candidateModeCount": candidates,
            "familyExcludedCount": family_excluded,
            "qualityExcludedCount": quality_excluded,
            "coreIntersectingPatchCount": len(patches),
            "coreMissCount": misses_core,
            "degenerateIntersectionCount": degenerate,
            "occupiedCellCount": len({value.cell_xyz for value in patches}),
        },
    )
