from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


RAW_ACUS_PIPELINE_SCHEMA = "pareidolia.raw-acus-cubical"
RAW_ACUS_PIPELINE_VERSION = 1


Int3 = tuple[int, int, int]
Float2 = tuple[float, float]


def _int3(values: Iterable[int], name: str) -> Int3:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{name} must contain three values")
    return result  # type: ignore[return-value]


def canonical_json_hash(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path: str | Path, payload: Mapping[str, Any]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(target)


def resolve_pipeline_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    """Follow an immutable variant chain to its originating raw-Acus pipeline.

    Variants intentionally own a new selected-patch artifact while retaining an
    ``inputRoot`` link to the geometry they refined.  Following that link makes
    variants composable instead of limiting downstream tools to one generation.
    """

    current = Path(root).resolve()
    visited: set[Path] = set()
    while True:
        if current in visited:
            raise ValueError("variant inputRoot chain contains a cycle")
        visited.add(current)
        pipeline_path = current / "pipeline.json"
        if pipeline_path.is_file():
            pipeline = json.loads(pipeline_path.read_text())
            if pipeline.get("state") != "complete":
                raise ValueError("raw-Acus pipeline is not complete")
            return current, pipeline
        variant_path = current / "variant.json"
        if not variant_path.is_file():
            raise ValueError(
                f"{current} has neither pipeline.json nor variant.json"
            )
        variant = json.loads(variant_path.read_text())
        if variant.get("state") != "complete":
            raise ValueError(f"variant at {current} is not complete")
        input_root = variant.get("inputRoot")
        if not input_root:
            raise ValueError(f"variant at {current} has no inputRoot")
        current = Path(input_root).resolve()


@dataclass(frozen=True, slots=True)
class VolumeSource:
    """One native uint8 ZYX CT array and its scanner-space metadata."""

    path: Path
    metadata_path: Path | None
    shape_zyx: Int3
    origin_xyz: Int3
    voxel_size_microns: float
    source_identity: Mapping[str, Any]

    @classmethod
    def open(
        cls, path: str | Path, metadata_path: str | Path | None = None
    ) -> "VolumeSource":
        source_path = Path(path).resolve()
        if metadata_path is None:
            candidate = source_path.with_suffix(".json")
            resolved_metadata = candidate if candidate.is_file() else None
        else:
            resolved_metadata = Path(metadata_path).resolve()
        array = np.load(source_path, mmap_mode="r")
        if array.ndim != 3 or array.dtype != np.uint8:
            raise ValueError("raw Acus requires one uint8 ZYX .npy CT volume")
        metadata: dict[str, Any] = {}
        if resolved_metadata is not None:
            metadata = json.loads(resolved_metadata.read_text())
        origin = _int3(metadata.get("originXYZ", (0, 0, 0)), "originXYZ")
        voxel_size = float(metadata.get("voxelSizeMicrons", 1.0))
        if not math.isfinite(voxel_size) or voxel_size <= 0.0:
            raise ValueError("voxelSizeMicrons must be finite and positive")
        stat = source_path.stat()
        identity: dict[str, Any] = {
            "path": str(source_path),
            "bytes": stat.st_size,
            "mtimeNs": stat.st_mtime_ns,
            "shapeZYX": [int(value) for value in array.shape],
            "dtype": array.dtype.str,
        }
        if resolved_metadata is not None:
            identity["metadataPath"] = str(resolved_metadata)
            identity["metadataSha256"] = sha256_file(resolved_metadata)
        identity["identitySha256"] = canonical_json_hash(identity)
        return cls(
            source_path,
            resolved_metadata,
            tuple(int(value) for value in array.shape),
            origin,
            voxel_size,
            identity,
        )

    @property
    def shape_xyz(self) -> Int3:
        return self.shape_zyx[::-1]

    def memmap(self) -> np.ndarray:
        return np.load(self.path, mmap_mode="r")


@dataclass(frozen=True, slots=True)
class RawAcusSettings:
    """Dataset-independent numerical and physical priors for raw Acus."""

    cell_stride_voxels: int = 32
    analysis_cube_voxels: int = 64
    hessian_scale_voxels: float = 1.25
    candidate_spacing_voxels: int = 4
    needle_length_voxels: float = 16.0
    extraction_halo_voxels: int = 16
    candidate_threshold: float = 0.015
    candidate_bin_voxels: int = 32
    extraction_tile_core_voxels: int = 128
    maximum_needles_per_bin: int = 32
    maximum_needles_per_cell: int = 256
    minimum_needles_per_cell: int = 8
    depth_bin_voxels: float = 1.0
    orientation_bins: int = 36
    depth_kernel_voxels: float = 2.5
    orientation_kernel_degrees: float = 9.0
    maximum_normal_hypotheses: int = 2
    maximum_layer_modes: int = 14
    maximum_configurations_per_cell: int = 8
    minimum_layer_spacing_microns: float = 35.0
    plausible_sheet_thickness_microns: Float2 = (80.0, 400.0)
    orthogonal_ply_std_degrees: float = 22.0
    ct_profile_tangent_samples: int = 7
    calibration_samples: int = 4
    calibration_cube_voxels: int = 96

    def __post_init__(self) -> None:
        integer_positive = (
            self.cell_stride_voxels,
            self.analysis_cube_voxels,
            self.candidate_spacing_voxels,
            self.extraction_halo_voxels,
            self.candidate_bin_voxels,
            self.extraction_tile_core_voxels,
            self.maximum_needles_per_bin,
            self.maximum_needles_per_cell,
            self.minimum_needles_per_cell,
            self.orientation_bins,
            self.maximum_normal_hypotheses,
            self.maximum_layer_modes,
            self.maximum_configurations_per_cell,
            self.ct_profile_tangent_samples,
            self.calibration_samples,
            self.calibration_cube_voxels,
        )
        if any(value <= 0 for value in integer_positive):
            raise ValueError("raw Acus integer settings must be positive")
        finite_positive = (
            self.hessian_scale_voxels,
            self.needle_length_voxels,
            self.candidate_threshold,
            self.depth_bin_voxels,
            self.depth_kernel_voxels,
            self.orientation_kernel_degrees,
            self.minimum_layer_spacing_microns,
            self.orthogonal_ply_std_degrees,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("raw Acus floating settings must be finite and positive")
        if self.analysis_cube_voxels < self.cell_stride_voxels:
            raise ValueError("analysis cube cannot be smaller than its owned cell")
        if (self.analysis_cube_voxels - self.cell_stride_voxels) % 2:
            raise ValueError("analysis cube and cell stride must have the same parity")
        if self.extraction_halo_voxels < math.ceil(self.needle_length_voxels):
            raise ValueError("extraction halo must be at least one needle length")
        if self.orientation_bins < 12 or self.orientation_bins % 2:
            raise ValueError("orientation bins must be an even value of at least 12")
        if self.ct_profile_tangent_samples < 3 or not self.ct_profile_tangent_samples % 2:
            raise ValueError("CT tangent sample count must be odd and at least three")
        low, high = (float(value) for value in self.plausible_sheet_thickness_microns)
        if not 0.0 < low < high:
            raise ValueError("plausible sheet thickness must be an increasing positive range")
        object.__setattr__(self, "plausible_sheet_thickness_microns", (low, high))

    @property
    def evidence_margin_voxels(self) -> int:
        return (self.analysis_cube_voxels - self.cell_stride_voxels) // 2

    @property
    def refinement_radius_voxels(self) -> int:
        return max(3, int(math.ceil(self.hessian_scale_voxels * 2.5)))

    @property
    def cross_section_radius_voxels(self) -> float:
        return max(2.0, float(math.ceil(self.hessian_scale_voxels * 1.5)))

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["plausible_sheet_thickness_microns"] = list(
            self.plausible_sheet_thickness_microns
        )
        return values


@dataclass(frozen=True, slots=True)
class ReconstructionWindow:
    """A regular cubical grid expressed in source-local voxel coordinates."""

    origin_voxel_xyz: Int3
    shape_cells_xyz: Int3

    def __post_init__(self) -> None:
        origin = _int3(self.origin_voxel_xyz, "origin_voxel_xyz")
        shape = _int3(self.shape_cells_xyz, "shape_cells_xyz")
        if any(value < 0 for value in origin):
            raise ValueError("window voxel origin must be nonnegative")
        if any(value <= 0 for value in shape):
            raise ValueError("window cell shape must be positive")
        object.__setattr__(self, "origin_voxel_xyz", origin)
        object.__setattr__(self, "shape_cells_xyz", shape)

    def stop_voxel_xyz(self, settings: RawAcusSettings) -> Int3:
        return tuple(
            self.origin_voxel_xyz[axis]
            + self.shape_cells_xyz[axis] * settings.cell_stride_voxels
            for axis in range(3)
        )  # type: ignore[return-value]

    def validate(self, source: VolumeSource, settings: RawAcusSettings) -> None:
        stop = self.stop_voxel_xyz(settings)
        required_margin = settings.evidence_margin_voxels + settings.extraction_halo_voxels
        for axis in range(3):
            if self.origin_voxel_xyz[axis] < required_margin:
                raise ValueError(
                    f"window begins too close to source axis {axis} boundary; "
                    f"requires {required_margin} voxels of raw support"
                )
            if stop[axis] + required_margin > source.shape_xyz[axis]:
                raise ValueError(
                    f"window ends too close to source axis {axis} boundary; "
                    f"requires {required_margin} voxels of raw support"
                )


@dataclass(frozen=True, slots=True)
class VoxelBounds:
    start_xyz: Int3
    stop_xyz_exclusive: Int3

    def __post_init__(self) -> None:
        start = _int3(self.start_xyz, "start_xyz")
        stop = _int3(self.stop_xyz_exclusive, "stop_xyz_exclusive")
        if any(stop[axis] <= start[axis] for axis in range(3)):
            raise ValueError("voxel bounds must have positive extent")
        object.__setattr__(self, "start_xyz", start)
        object.__setattr__(self, "stop_xyz_exclusive", stop)

    @property
    def shape_xyz(self) -> Int3:
        return tuple(
            self.stop_xyz_exclusive[axis] - self.start_xyz[axis]
            for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def slices_zyx(self) -> tuple[slice, slice, slice]:
        return tuple(
            slice(self.start_xyz[axis], self.stop_xyz_exclusive[axis])
            for axis in (2, 1, 0)
        )  # type: ignore[return-value]

    def expand(self, margin: int) -> "VoxelBounds":
        return VoxelBounds(
            tuple(value - margin for value in self.start_xyz),
            tuple(value + margin for value in self.stop_xyz_exclusive),
        )

    def record(self) -> dict[str, list[int]]:
        return {
            "startXYZ": list(self.start_xyz),
            "stopXYZExclusive": list(self.stop_xyz_exclusive),
        }


@dataclass(frozen=True, slots=True)
class ShardSpec:
    index_xyz: Int3
    start_cell_xyz: Int3
    stop_cell_xyz_exclusive: Int3
    owned_voxel_bounds: VoxelBounds
    evidence_voxel_bounds: VoxelBounds
    raw_voxel_bounds: VoxelBounds

    @property
    def shape_cells_xyz(self) -> Int3:
        return tuple(
            self.stop_cell_xyz_exclusive[axis] - self.start_cell_xyz[axis]
            for axis in range(3)
        )  # type: ignore[return-value]

    @property
    def shard_id(self) -> str:
        return "x%04d-y%04d-z%04d" % self.index_xyz

    def record(self) -> dict[str, Any]:
        return {
            "id": self.shard_id,
            "indexXYZ": list(self.index_xyz),
            "startCellXYZ": list(self.start_cell_xyz),
            "stopCellXYZExclusive": list(self.stop_cell_xyz_exclusive),
            "shapeCellsXYZ": list(self.shape_cells_xyz),
            "ownedVoxelBounds": self.owned_voxel_bounds.record(),
            "evidenceVoxelBounds": self.evidence_voxel_bounds.record(),
            "rawVoxelBounds": self.raw_voxel_bounds.record(),
        }


@dataclass(frozen=True, slots=True)
class ExtractionTileSpec:
    """One canonical source-anchored GPU compute tile, independent of cells."""

    index_xyz: Int3
    core_voxel_bounds: VoxelBounds
    raw_voxel_bounds: VoxelBounds

    @property
    def tile_id(self) -> str:
        return "x%04d-y%04d-z%04d" % self.index_xyz

    def record(self) -> dict[str, Any]:
        return {
            "kind": "canonical-acus-extraction-tile",
            "id": self.tile_id,
            "indexXYZ": list(self.index_xyz),
            "coreVoxelBounds": self.core_voxel_bounds.record(),
            "rawVoxelBounds": self.raw_voxel_bounds.record(),
        }


def _bounds_intersect(first: VoxelBounds, second: VoxelBounds) -> bool:
    return all(
        first.start_xyz[axis] < second.stop_xyz_exclusive[axis]
        and second.start_xyz[axis] < first.stop_xyz_exclusive[axis]
        for axis in range(3)
    )


def plan_extraction_tiles(
    source: VolumeSource,
    processing_bounds: VoxelBounds,
    settings: RawAcusSettings,
) -> tuple[ExtractionTileSpec, ...]:
    """Plan fixed source-anchored cores intersecting one evidence region."""

    halo = settings.extraction_halo_voxels
    core_size = settings.extraction_tile_core_voxels
    starts_by_axis: list[list[int]] = []
    for axis in range(3):
        high = source.shape_xyz[axis] - halo
        starts_by_axis.append(list(range(halo, high, core_size)))
    result: list[ExtractionTileSpec] = []
    for iz, start_z in enumerate(starts_by_axis[2]):
        for iy, start_y in enumerate(starts_by_axis[1]):
            for ix, start_x in enumerate(starts_by_axis[0]):
                start = (start_x, start_y, start_z)
                stop = tuple(
                    min(
                        start[axis] + core_size,
                        source.shape_xyz[axis] - halo,
                    )
                    for axis in range(3)
                )
                core = VoxelBounds(start, stop)
                if not _bounds_intersect(core, processing_bounds):
                    continue
                result.append(
                    ExtractionTileSpec(
                        (ix, iy, iz),
                        core,
                        core.expand(halo),
                    )
                )
    return tuple(result)


def extraction_tiles_for_shard(
    tiles: Iterable[ExtractionTileSpec], shard: ShardSpec
) -> tuple[ExtractionTileSpec, ...]:
    return tuple(
        tile
        for tile in tiles
        if _bounds_intersect(tile.core_voxel_bounds, shard.evidence_voxel_bounds)
    )


def plan_shards(
    window: ReconstructionWindow,
    settings: RawAcusSettings,
    shard_shape_cells_xyz: Iterable[int],
) -> tuple[ShardSpec, ...]:
    shard_shape = _int3(shard_shape_cells_xyz, "shard_shape_cells_xyz")
    if any(value <= 0 for value in shard_shape):
        raise ValueError("shard cell shape must be positive")
    result: list[ShardSpec] = []
    counts = tuple(
        int(math.ceil(window.shape_cells_xyz[axis] / shard_shape[axis]))
        for axis in range(3)
    )
    stride = settings.cell_stride_voxels
    for iz in range(counts[2]):
        for iy in range(counts[1]):
            for ix in range(counts[0]):
                index = (ix, iy, iz)
                start = tuple(index[axis] * shard_shape[axis] for axis in range(3))
                stop = tuple(
                    min(start[axis] + shard_shape[axis], window.shape_cells_xyz[axis])
                    for axis in range(3)
                )
                owned_start = tuple(
                    window.origin_voxel_xyz[axis] + start[axis] * stride
                    for axis in range(3)
                )
                owned_stop = tuple(
                    window.origin_voxel_xyz[axis] + stop[axis] * stride
                    for axis in range(3)
                )
                owned = VoxelBounds(owned_start, owned_stop)
                evidence = owned.expand(settings.evidence_margin_voxels)
                raw = evidence.expand(settings.extraction_halo_voxels)
                result.append(ShardSpec(index, start, stop, owned, evidence, raw))
    return tuple(result)


def pipeline_identity(
    source: VolumeSource,
    window: ReconstructionWindow,
    settings: RawAcusSettings,
    shard_shape_cells_xyz: Iterable[int],
) -> dict[str, Any]:
    payload: dict[str, Any] = {
        "schema": RAW_ACUS_PIPELINE_SCHEMA,
        "version": RAW_ACUS_PIPELINE_VERSION,
        "source": dict(source.source_identity),
        "window": {
            "originVoxelXYZ": list(window.origin_voxel_xyz),
            "shapeCellsXYZ": list(window.shape_cells_xyz),
        },
        "settings": settings.record(),
        "shardShapeCellsXYZ": list(
            _int3(shard_shape_cells_xyz, "shard_shape_cells_xyz")
        ),
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload
