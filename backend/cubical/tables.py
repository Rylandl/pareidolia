from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .geometry import ClippedPatch, PlaneEstimate, clip_plane_to_cell
from .topology import GridEdge, GridSpec, cell_edges


PATCH_ARTIFACT_SCHEMA = "pareidolia.cubical-patches"
PATCH_ARTIFACT_VERSION = 1


def _packed_covariance(covariance: np.ndarray) -> np.ndarray:
    return covariance[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)]


def _unpacked_covariance(values: np.ndarray) -> np.ndarray:
    result = np.zeros((3, 3), dtype=np.float64)
    result[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)] = values
    result[(1, 2, 2), (0, 0, 1)] = values[[1, 2, 4]]
    return result


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


@dataclass(slots=True)
class PatchTable:
    """Structure-of-arrays storage for one independently processable patch shard."""

    grid: GridSpec
    patch_id: np.ndarray
    cell_xyz: np.ndarray
    configuration_id: np.ndarray
    configuration_log_weight: np.ndarray
    local_order: np.ndarray
    normal_family: np.ndarray
    normal_xyz: np.ndarray
    height: np.ndarray
    plane_covariance: np.ndarray
    fiber_xyz: np.ndarray
    fiber_angular_std_radians: np.ndarray
    confidence: np.ndarray
    vertex_offset: np.ndarray
    vertex_edge_axis: np.ndarray
    vertex_edge_anchor: np.ndarray
    vertex_t: np.ndarray
    vertex_variance: np.ndarray

    @property
    def patch_count(self) -> int:
        return int(len(self.patch_id))

    @property
    def vertex_count(self) -> int:
        return int(len(self.vertex_t))

    @classmethod
    def from_patches(
        cls,
        grid: GridSpec,
        patches: list[ClippedPatch] | tuple[ClippedPatch, ...],
        *,
        configuration_id: Mapping[int, int] | None = None,
        configuration_log_weight: Mapping[int, float] | None = None,
        local_order: Mapping[int, int] | None = None,
        normal_family: Mapping[int, int] | None = None,
    ) -> "PatchTable":
        values = sorted(patches, key=lambda value: value.patch_id)
        patch_count = len(values)
        vertex_offset = np.zeros(patch_count + 1, dtype=np.uint64)
        for index, patch in enumerate(values):
            vertex_offset[index + 1] = vertex_offset[index] + len(patch.vertices)
        vertex_count = int(vertex_offset[-1])
        fiber = np.full((patch_count, 3), np.nan, dtype=np.float32)
        fiber_std = np.full(patch_count, np.nan, dtype=np.float32)
        table = cls(
            grid=grid,
            patch_id=np.asarray([value.patch_id for value in values], dtype=np.uint64),
            cell_xyz=np.asarray([value.cell_xyz for value in values], dtype=np.int32).reshape(
                patch_count, 3
            ),
            configuration_id=np.asarray(
                [
                    (configuration_id or {}).get(value.patch_id, 0)
                    for value in values
                ],
                dtype=np.uint32,
            ),
            configuration_log_weight=np.asarray(
                [
                    (configuration_log_weight or {}).get(value.patch_id, 0.0)
                    for value in values
                ],
                dtype=np.float32,
            ),
            local_order=np.asarray(
                [(local_order or {}).get(value.patch_id, 0) for value in values],
                dtype=np.int16,
            ),
            normal_family=np.asarray(
                [(normal_family or {}).get(value.patch_id, 0) for value in values],
                dtype=np.int16,
            ),
            normal_xyz=np.asarray(
                [value.estimate.normal_xyz for value in values], dtype=np.float32
            ).reshape(patch_count, 3),
            height=np.asarray(
                [value.estimate.height_from_cell_center for value in values],
                dtype=np.float32,
            ),
            plane_covariance=np.asarray(
                [
                    _packed_covariance(value.estimate.covariance_matrix)
                    for value in values
                ],
                dtype=np.float32,
            ).reshape(patch_count, 6),
            fiber_xyz=fiber,
            fiber_angular_std_radians=fiber_std,
            confidence=np.asarray(
                [value.estimate.confidence for value in values], dtype=np.float32
            ),
            vertex_offset=vertex_offset,
            vertex_edge_axis=np.empty(vertex_count, dtype=np.uint8),
            vertex_edge_anchor=np.empty((vertex_count, 3), dtype=np.int32),
            vertex_t=np.empty(vertex_count, dtype=np.float32),
            vertex_variance=np.empty(vertex_count, dtype=np.float32),
        )
        for patch_index, patch in enumerate(values):
            if patch.estimate.fiber_xyz is not None:
                table.fiber_xyz[patch_index] = patch.estimate.fiber_xyz
            if patch.estimate.fiber_angular_std_radians is not None:
                table.fiber_angular_std_radians[patch_index] = (
                    patch.estimate.fiber_angular_std_radians
                )
            low = int(vertex_offset[patch_index])
            high = int(vertex_offset[patch_index + 1])
            table.vertex_edge_axis[low:high] = [
                value.edge.axis for value in patch.vertices
            ]
            table.vertex_edge_anchor[low:high] = [
                value.edge.anchor_xyz for value in patch.vertices
            ]
            table.vertex_t[low:high] = [value.t for value in patch.vertices]
            table.vertex_variance[low:high] = [
                value.variance for value in patch.vertices
            ]
        table.validate()
        return table

    def arrays(self) -> dict[str, np.ndarray]:
        return {
            "patchId": self.patch_id,
            "cellXYZ": self.cell_xyz,
            "configurationId": self.configuration_id,
            "configurationLogWeight": self.configuration_log_weight,
            "localOrder": self.local_order,
            "normalFamily": self.normal_family,
            "normalXYZ": self.normal_xyz,
            "height": self.height,
            "planeCovariance": self.plane_covariance,
            "fiberXYZ": self.fiber_xyz,
            "fiberAngularStdRadians": self.fiber_angular_std_radians,
            "confidence": self.confidence,
            "vertexOffset": self.vertex_offset,
            "vertexEdgeAxis": self.vertex_edge_axis,
            "vertexEdgeAnchor": self.vertex_edge_anchor,
            "vertexT": self.vertex_t,
            "vertexVariance": self.vertex_variance,
        }

    def validate(self) -> None:
        patch_count = self.patch_count
        vertex_count = self.vertex_count
        expected = {
            "patch_id": (patch_count,),
            "cell_xyz": (patch_count, 3),
            "configuration_id": (patch_count,),
            "configuration_log_weight": (patch_count,),
            "local_order": (patch_count,),
            "normal_family": (patch_count,),
            "normal_xyz": (patch_count, 3),
            "height": (patch_count,),
            "plane_covariance": (patch_count, 6),
            "fiber_xyz": (patch_count, 3),
            "fiber_angular_std_radians": (patch_count,),
            "confidence": (patch_count,),
            "vertex_offset": (patch_count + 1,),
            "vertex_edge_axis": (vertex_count,),
            "vertex_edge_anchor": (vertex_count, 3),
            "vertex_t": (vertex_count,),
            "vertex_variance": (vertex_count,),
        }
        for name, shape in expected.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} has shape {getattr(self, name).shape}, expected {shape}")
        if len(np.unique(self.patch_id)) != patch_count:
            raise ValueError("patch IDs must be unique within a shard")
        if patch_count and not np.all(
            (self.cell_xyz >= 0)
            & (self.cell_xyz < np.asarray(self.grid.shape_cells_xyz))
        ):
            raise ValueError("patch table contains a cell outside its grid")
        if int(self.vertex_offset[0]) != 0 or not np.array_equal(
            self.vertex_offset,
            np.maximum.accumulate(self.vertex_offset),
        ) or int(self.vertex_offset[-1]) != vertex_count:
            raise ValueError("polygon vertex offsets are invalid")
        counts = np.diff(self.vertex_offset.astype(np.int64))
        if np.any((counts < 3) | (counts > 6)):
            raise ValueError("generic clipped patches require three to six vertices")
        if vertex_count and (
            np.any(~np.isfinite(self.vertex_t))
            or np.any(self.vertex_t <= 0.0)
            or np.any(self.vertex_t >= 1.0)
            or np.any(~np.isfinite(self.vertex_variance))
            or np.any(self.vertex_variance < 0.0)
        ):
            raise ValueError("edge crossing arrays contain invalid values")
        if np.any(~np.isfinite(self.normal_xyz)) or np.any(
            np.abs(np.linalg.norm(self.normal_xyz, axis=1) - 1.0) > 2.0e-5
        ):
            raise ValueError("patch normals must be finite unit axes")
        if np.any(~np.isfinite(self.confidence)) or np.any(
            (self.confidence < 0.0) | (self.confidence > 1.0)
        ):
            raise ValueError("patch confidence must lie in [0, 1]")
        for patch_index in range(patch_count):
            allowed = set(cell_edges(tuple(int(value) for value in self.cell_xyz[patch_index])))
            low = int(self.vertex_offset[patch_index])
            high = int(self.vertex_offset[patch_index + 1])
            for vertex_index in range(low, high):
                edge = GridEdge(
                    int(self.vertex_edge_axis[vertex_index]),
                    tuple(int(value) for value in self.vertex_edge_anchor[vertex_index]),
                )
                if edge not in allowed:
                    raise ValueError("polygon vertex references a non-incident grid edge")

    def to_patches(self) -> tuple[ClippedPatch, ...]:
        self.validate()
        patches: list[ClippedPatch] = []
        for patch_index in range(self.patch_count):
            fiber_values = self.fiber_xyz[patch_index]
            estimate = PlaneEstimate(
                tuple(float(value) for value in self.normal_xyz[patch_index]),
                float(self.height[patch_index]),
                tuple(
                    tuple(float(value) for value in row)
                    for row in _unpacked_covariance(
                        self.plane_covariance[patch_index]
                    )
                ),
                tuple(float(value) for value in fiber_values)
                if np.all(np.isfinite(fiber_values))
                else None,
                float(self.fiber_angular_std_radians[patch_index])
                if np.isfinite(self.fiber_angular_std_radians[patch_index])
                else None,
                float(self.confidence[patch_index]),
            )
            patch = clip_plane_to_cell(
                self.grid,
                tuple(int(value) for value in self.cell_xyz[patch_index]),
                estimate,
                patch_id=int(self.patch_id[patch_index]),
            )
            if patch is None:
                raise ValueError("stored plane no longer intersects its owning cell")
            low = int(self.vertex_offset[patch_index])
            high = int(self.vertex_offset[patch_index + 1])
            stored = {
                GridEdge(
                    int(self.vertex_edge_axis[index]),
                    tuple(int(value) for value in self.vertex_edge_anchor[index]),
                ): float(self.vertex_t[index])
                for index in range(low, high)
            }
            derived = {value.edge: value.t for value in patch.vertices}
            if stored.keys() != derived.keys() or any(
                abs(stored[edge] - derived[edge]) > 2.0e-5 for edge in stored
            ):
                raise ValueError("stored clipped polygon disagrees with its plane")
            patches.append(patch)
        return tuple(patches)


def write_patch_shard(
    prefix: str | Path,
    table: PatchTable,
    *,
    settings: Mapping[str, Any] | None = None,
    provenance: Mapping[str, Any] | None = None,
    compressed: bool = False,
) -> dict[str, Any]:
    """Atomically write one independently invalidatable patch shard."""

    table.validate()
    base = Path(prefix)
    data_path = base.with_suffix(".npz")
    manifest_path = base.with_suffix(".json")
    data_path.parent.mkdir(parents=True, exist_ok=True)
    temporary_data = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary_data.open("wb") as handle:
        writer = np.savez_compressed if compressed else np.savez
        writer(handle, **table.arrays())
    temporary_data.replace(data_path)
    arrays = table.arrays()
    manifest = {
        "schema": PATCH_ARTIFACT_SCHEMA,
        "version": PATCH_ARTIFACT_VERSION,
        "grid": {
            "shapeCellsXYZ": list(table.grid.shape_cells_xyz),
            "cellSizeXYZ": list(table.grid.cell_size_xyz),
            "originXYZ": list(table.grid.origin_xyz),
            "coordinateUnit": table.grid.coordinate_unit,
        },
        "counts": {
            "patches": table.patch_count,
            "vertices": table.vertex_count,
            "configurations": int(
                len(
                    {
                        (*cell, int(configuration))
                        for cell, configuration in zip(
                            table.cell_xyz.tolist(), table.configuration_id
                        )
                    }
                )
            ),
        },
        "arrays": {
            name: {"shape": list(value.shape), "dtype": value.dtype.str}
            for name, value in arrays.items()
        },
        "settings": dict(settings or {}),
        "provenance": dict(provenance or {}),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": _sha256(data_path),
            "compressed": bool(compressed),
        },
    }
    temporary_manifest = manifest_path.with_suffix(manifest_path.suffix + ".tmp")
    temporary_manifest.write_text(json.dumps(manifest, indent=2) + "\n")
    temporary_manifest.replace(manifest_path)
    return manifest


def read_patch_shard(prefix: str | Path, *, verify: bool = True) -> PatchTable:
    base = Path(prefix)
    data_path = base.with_suffix(".npz")
    manifest_path = base.with_suffix(".json")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PATCH_ARTIFACT_SCHEMA
        or int(manifest.get("version", -1)) != PATCH_ARTIFACT_VERSION
    ):
        raise ValueError("unsupported cubical patch artifact")
    if verify and _sha256(data_path) != manifest["data"]["sha256"]:
        raise ValueError("cubical patch artifact content hash mismatch")
    grid_record = manifest["grid"]
    grid = GridSpec(
        tuple(grid_record["shapeCellsXYZ"]),
        tuple(grid_record["cellSizeXYZ"]),
        tuple(grid_record["originXYZ"]),
        grid_record["coordinateUnit"],
    )
    with np.load(data_path) as values:
        table = PatchTable(
            grid=grid,
            patch_id=np.asarray(values["patchId"], dtype=np.uint64),
            cell_xyz=np.asarray(values["cellXYZ"], dtype=np.int32),
            configuration_id=np.asarray(values["configurationId"], dtype=np.uint32),
            configuration_log_weight=np.asarray(
                values["configurationLogWeight"], dtype=np.float32
            ),
            local_order=np.asarray(values["localOrder"], dtype=np.int16),
            normal_family=np.asarray(values["normalFamily"], dtype=np.int16),
            normal_xyz=np.asarray(values["normalXYZ"], dtype=np.float32),
            height=np.asarray(values["height"], dtype=np.float32),
            plane_covariance=np.asarray(values["planeCovariance"], dtype=np.float32),
            fiber_xyz=np.asarray(values["fiberXYZ"], dtype=np.float32),
            fiber_angular_std_radians=np.asarray(
                values["fiberAngularStdRadians"], dtype=np.float32
            ),
            confidence=np.asarray(values["confidence"], dtype=np.float32),
            vertex_offset=np.asarray(values["vertexOffset"], dtype=np.uint64),
            vertex_edge_axis=np.asarray(values["vertexEdgeAxis"], dtype=np.uint8),
            vertex_edge_anchor=np.asarray(values["vertexEdgeAnchor"], dtype=np.int32),
            vertex_t=np.asarray(values["vertexT"], dtype=np.float32),
            vertex_variance=np.asarray(values["vertexVariance"], dtype=np.float32),
        )
    table.validate()
    return table
