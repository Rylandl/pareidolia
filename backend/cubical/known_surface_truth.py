from __future__ import annotations

import json
import math
import struct
import time
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .isolated_slab import ISOLATED_SLAB_SCHEMA, ISOLATED_SLAB_STEM
from .material_interface import MATERIAL_INTERFACE_SCHEMA, MATERIAL_INTERFACE_STEM


KNOWN_SURFACE_CROP_SCHEMA = "pareidolia.known-surface-truth-crop"
KNOWN_SURFACE_CROP_VERSION = 1
KNOWN_SURFACE_CROP_STEM = "known-surface-truth-v1"
KNOWN_SURFACE_AUDIT_SCHEMA = "pareidolia.known-surface-ribbon-audit"
KNOWN_SURFACE_AUDIT_VERSION = 1
KNOWN_SURFACE_AUDIT_STEM = "known-surface-ribbon-audit-v1"
KNOWN_SURFACE_INTERFACE_AUDIT_SCHEMA = (
    "pareidolia.known-surface-material-interface-audit"
)
KNOWN_SURFACE_INTERFACE_AUDIT_VERSION = 1
KNOWN_SURFACE_INTERFACE_AUDIT_STEM = "known-surface-material-interface-audit-v1"


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "median": 0.0,
            "p90": 0.0,
            "p99": 0.0,
            "maximum": 0.0,
        }
    return {
        "count": int(len(finite)),
        "median": round(float(np.percentile(finite, 50)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "p99": round(float(np.percentile(finite, 99)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def _surface_volume_root(fragment_root: str | Path) -> Path:
    root = Path(fragment_root).resolve()
    candidate = root / "surface_volume"
    return candidate if candidate.is_dir() else root


def _surface_metadata(root: Path) -> dict[str, Any]:
    metadata_path = root / "meta.json"
    metadata = json.loads(metadata_path.read_text())
    required = ("width", "height", "slices", "voxelsize")
    if any(name not in metadata for name in required):
        raise ValueError(f"known surface metadata is incomplete: {metadata_path}")
    width = int(metadata["width"])
    height = int(metadata["height"])
    slices = int(metadata["slices"])
    voxel_size = float(metadata["voxelsize"])
    if width <= 0 or height <= 0 or slices <= 0:
        raise ValueError("known surface dimensions must be positive")
    if not math.isfinite(voxel_size) or voxel_size <= 0.0:
        raise ValueError("known surface voxel size must be finite and positive")
    return {
        **metadata,
        "width": width,
        "height": height,
        "slices": slices,
        "voxelsize": voxel_size,
        "metadataPath": metadata_path,
    }


def legacy_surface_tiff(
    path: str | Path,
    *,
    height: int,
    width: int,
) -> np.ndarray:
    """Memory-map one legacy uncompressed little-endian uint16 surface TIFF.

    Vesuvius legacy fragment volumes place one contiguous pixel strip directly
    after the eight-byte TIFF header and the IFD directly after that strip.  A
    strict check keeps this dependency-free reader from silently interpreting
    a compressed or tiled TIFF as pixels.
    """

    source = Path(path)
    with source.open("rb") as handle:
        header = handle.read(8)
    if len(header) != 8:
        raise ValueError(f"surface TIFF header is truncated: {source}")
    byte_order, magic, ifd_offset = struct.unpack("<2sHI", header)
    expected_ifd = 8 + int(height) * int(width) * 2
    if byte_order != b"II" or magic != 42 or ifd_offset != expected_ifd:
        raise ValueError(
            "known-surface reader requires a contiguous little-endian uint16 "
            f"legacy TIFF: {source}"
        )
    if source.stat().st_size < expected_ifd:
        raise ValueError(f"surface TIFF pixel strip is truncated: {source}")
    return np.memmap(
        source,
        dtype="<u2",
        mode="r",
        offset=8,
        shape=(int(height), int(width)),
    )


def _atomic_npy(path: Path, values: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.save(handle, values, allow_pickle=False)
    temporary.replace(path)


def _display_gray(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    pixels = np.asarray(values, dtype=np.float32)
    sample = pixels[valid]
    if len(sample):
        low, high = np.percentile(sample, (1.0, 99.0))
    else:
        low, high = 0.0, 1.0
    if high <= low:
        high = low + 1.0
    return np.clip((pixels - low) * (255.0 / (high - low)), 0, 255).astype(
        np.uint8
    )


def prepare_known_surface_crop(
    fragment_root: str | Path,
    output_path: str | Path,
    *,
    crop_origin_xy: tuple[int, int],
    crop_shape_xy: tuple[int, int],
    depth_start: int = 0,
    depth_stop_exclusive: int | None = None,
    force: bool = False,
) -> dict[str, Any]:
    root = _surface_volume_root(fragment_root)
    metadata = _surface_metadata(root)
    x0, y0 = (int(value) for value in crop_origin_xy)
    width, height = (int(value) for value in crop_shape_xy)
    z0 = int(depth_start)
    z1 = (
        int(metadata["slices"])
        if depth_stop_exclusive is None
        else int(depth_stop_exclusive)
    )
    if width <= 0 or height <= 0 or z1 <= z0:
        raise ValueError("known surface crop dimensions must be positive")
    if (
        x0 < 0
        or y0 < 0
        or z0 < 0
        or x0 + width > int(metadata["width"])
        or y0 + height > int(metadata["height"])
        or z1 > int(metadata["slices"])
    ):
        raise ValueError("known surface crop lies outside the source volume")

    slice_paths = [root / f"{index:02d}.tif" for index in range(z0, z1)]
    missing = [path for path in slice_paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"known surface slice is unavailable: {missing[0]}")
    source_identity = {
        "metadataPath": str(metadata["metadataPath"]),
        "metadataSha256": sha256_file(metadata["metadataPath"]),
        "slices": [
            {
                "path": str(path),
                "bytes": path.stat().st_size,
                "mtimeNs": path.stat().st_mtime_ns,
            }
            for path in slice_paths
        ],
    }
    identity: dict[str, Any] = {
        "schema": KNOWN_SURFACE_CROP_SCHEMA,
        "version": KNOWN_SURFACE_CROP_VERSION,
        "source": source_identity,
        "crop": {
            "originXYZ": [x0, y0, z0],
            "shapeXYZ": [width, height, z1 - z0],
        },
        "normalization": "round(uint16 / 257) to uint8",
    }
    identity["identitySha256"] = canonical_json_hash(identity)

    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{KNOWN_SURFACE_CROP_STEM}.json"
    volume_path = output / "known-surface-volume.npy"
    volume_metadata_path = output / "known-surface-volume.json"
    preview_path = output / "known-surface-preview.png"
    if not force and manifest_path.is_file() and volume_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(volume_path)
        ):
            return cached

    started = time.monotonic()
    volume = np.empty((z1 - z0, height, width), dtype=np.uint8)
    for output_z, path in enumerate(slice_paths):
        source = legacy_surface_tiff(
            path,
            height=int(metadata["height"]),
            width=int(metadata["width"]),
        )
        raw = np.asarray(source[y0 : y0 + height, x0 : x0 + width], dtype=np.uint32)
        volume[output_z] = ((raw + 128) // 257).astype(np.uint8)
    valid = np.max(volume, axis=0) > 0
    _atomic_npy(volume_path, volume)
    atomic_json(
        volume_metadata_path,
        {
            "name": f"known unrolled surface crop from {root.parent.name}",
            "originXYZ": [x0, y0, z0],
            "voxelSizeMicrons": float(metadata["voxelsize"]),
            "sourceKind": "known-unrolled-surface-volume",
            "expectedNormalXYZ": [0.0, 0.0, 1.0],
            "expectedSheetCountPerValidColumn": 1,
        },
    )
    center = volume[len(volume) // 2]
    gray = _display_gray(center, valid)
    preview = np.repeat(gray[:, :, None], 3, axis=2)
    preview[~valid] = (8, 11, 15)
    preview_path.write_bytes(rgb_png(preview))

    manifest: dict[str, Any] = {
        "schema": KNOWN_SURFACE_CROP_SCHEMA,
        "version": KNOWN_SURFACE_CROP_VERSION,
        "state": "complete",
        "identity": identity,
        "source": {
            "fragmentRoot": str(root.parent),
            "surfaceVolumeRoot": str(root),
            "voxelSizeMicrons": float(metadata["voxelsize"]),
            "sourceShapeZYX": [
                int(metadata["slices"]),
                int(metadata["height"]),
                int(metadata["width"]),
            ],
        },
        "geometry": {
            "originXYZ": [x0, y0, z0],
            "shapeZYX": list(volume.shape),
            "coordinateUnit": "known-surface-chart-voxel",
            "expectedNormalXYZ": [0.0, 0.0, 1.0],
            "expectedSheetCountPerValidColumn": 1,
        },
        "coverage": {
            "validColumnCount": int(np.count_nonzero(valid)),
            "columnCount": int(valid.size),
            "validColumnFraction": round(float(np.mean(valid)), 6),
        },
        "data": {
            "path": volume_path.name,
            "metadataPath": volume_metadata_path.name,
            "bytes": volume_path.stat().st_size,
            "sha256": sha256_file(volume_path),
        },
        "artifacts": {"preview": preview_path.name},
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _resolve_manifest(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _coverage_grid(valid: np.ndarray, stride: int) -> np.ndarray:
    height, width = valid.shape
    shape = (math.ceil(height / stride), math.ceil(width / stride))
    result = np.zeros(shape, dtype=bool)
    y, x = np.nonzero(valid)
    result[y // stride, x // stride] = True
    return result


def _best_per_column(
    linear_key: np.ndarray,
    confidence: np.ndarray,
) -> np.ndarray:
    if not len(linear_key):
        return np.empty(0, dtype=np.int64)
    order = np.lexsort((-confidence, linear_key))
    key = linear_key[order]
    keep = np.concatenate((np.ones(1, dtype=bool), key[1:] != key[:-1]))
    return order[keep]


def run_known_surface_ribbon_audit(
    truth_root: str | Path,
    slab_root: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    truth_path = _resolve_manifest(truth_root, KNOWN_SURFACE_CROP_STEM)
    slab_path = _resolve_manifest(slab_root, ISOLATED_SLAB_STEM)
    truth = json.loads(truth_path.read_text())
    slabs = json.loads(slab_path.read_text())
    if truth.get("schema") != KNOWN_SURFACE_CROP_SCHEMA or truth.get("state") != "complete":
        raise ValueError("known-surface truth crop must be complete")
    if slabs.get("schema") != ISOLATED_SLAB_SCHEMA or slabs.get("state") != "complete":
        raise ValueError("isolated-slab artifact must be complete")
    truth_data_path = truth_path.parent / str(truth["data"]["path"])
    slab_data_path = slab_path.parent / str(slabs["data"]["path"])
    if sha256_file(truth_data_path) != truth["data"]["sha256"]:
        raise ValueError("known-surface truth data changed after preparation")
    if sha256_file(slab_data_path) != slabs["data"]["sha256"]:
        raise ValueError("isolated-slab data changed after detection")

    identity: dict[str, Any] = {
        "schema": KNOWN_SURFACE_AUDIT_SCHEMA,
        "version": KNOWN_SURFACE_AUDIT_VERSION,
        "truthManifestPath": str(truth_path),
        "truthManifestSha256": sha256_file(truth_path),
        "slabManifestPath": str(slab_path),
        "slabManifestSha256": sha256_file(slab_path),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{KNOWN_SURFACE_AUDIT_STEM}.json"
    preview_path = output / "known-surface-ribbon-audit.png"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
        ):
            return cached

    started = time.monotonic()
    volume = np.load(truth_data_path, mmap_mode="r")
    valid = np.max(volume, axis=0) > 0
    with np.load(slab_data_path, allow_pickle=False) as stored:
        midpoint = np.asarray(stored["midpointXYZ"], dtype=np.float64)
        normal = np.asarray(stored["normalXYZ"], dtype=np.float64)
        thickness = np.asarray(stored["thicknessVoxels"], dtype=np.float64)
        confidence = np.asarray(stored["confidence"], dtype=np.float64)
        component = np.asarray(stored["componentId"], dtype=np.int64)

    origin = np.asarray(truth["geometry"]["originXYZ"], dtype=np.float64)
    local = midpoint - origin[None, :]
    width = int(volume.shape[2])
    height = int(volume.shape[1])
    depth = int(volume.shape[0])
    inside = (
        (local[:, 0] >= 0.0)
        & (local[:, 0] < width)
        & (local[:, 1] >= 0.0)
        & (local[:, 1] < height)
        & (local[:, 2] >= 0.0)
        & (local[:, 2] < depth)
    )
    pixel_x = np.clip(np.floor(local[:, 0]).astype(np.int64), 0, width - 1)
    pixel_y = np.clip(np.floor(local[:, 1]).astype(np.int64), 0, height - 1)
    on_valid_column = inside & valid[pixel_y, pixel_x]
    seed_threshold = float(
        slabs["identity"]["settings"].get("minimum_seed_confidence", 0.5)
    )
    seed = on_valid_column & (confidence >= seed_threshold) & (component >= 0)
    stride = int(slabs["identity"]["settings"]["sampling_stride_voxels"])
    valid_grid = _coverage_grid(valid, stride)
    grid_height, grid_width = valid_grid.shape
    grid_x = np.clip((local[:, 0] // stride).astype(np.int64), 0, grid_width - 1)
    grid_y = np.clip((local[:, 1] // stride).astype(np.int64), 0, grid_height - 1)
    linear = grid_y * grid_width + grid_x
    all_count = np.zeros(valid_grid.size, dtype=np.int32)
    seed_count = np.zeros(valid_grid.size, dtype=np.int32)
    np.add.at(all_count, linear[on_valid_column], 1)
    np.add.at(seed_count, linear[seed], 1)
    all_count = all_count.reshape(valid_grid.shape)
    seed_count = seed_count.reshape(valid_grid.shape)
    valid_count = max(int(np.count_nonzero(valid_grid)), 1)
    recovered = valid_grid & (seed_count > 0)

    axial_angle = np.degrees(
        np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0))
    )
    best = _best_per_column(linear[seed], confidence[seed])
    seed_index = np.flatnonzero(seed)
    best_index = seed_index[best]
    if len(best_index) >= 3:
        design = np.column_stack(
            (local[best_index, 0], local[best_index, 1], np.ones(len(best_index)))
        )
        coefficient, *_ = np.linalg.lstsq(
            design, local[best_index, 2], rcond=None
        )
        plane_residual = np.abs(
            local[best_index, 2] - design @ coefficient
        )
    else:
        coefficient = np.zeros(3, dtype=np.float64)
        plane_residual = np.empty(0, dtype=np.float64)

    component_values, component_counts = np.unique(
        component[seed], return_counts=True
    )
    largest_component_fraction = (
        float(np.max(component_counts)) / max(int(np.sum(component_counts)), 1)
        if len(component_counts)
        else 0.0
    )
    recovered_count = max(int(np.count_nonzero(recovered)), 1)
    metrics = {
        "validSamplingColumnCount": int(np.count_nonzero(valid_grid)),
        "recoveredSamplingColumnCount": int(np.count_nonzero(recovered)),
        "highConfidenceColumnCoverage": round(
            float(np.count_nonzero(recovered)) / valid_count, 6
        ),
        "anyCandidateColumnCoverage": round(
            float(np.count_nonzero(valid_grid & (all_count > 0))) / valid_count,
            6,
        ),
        "singleSeedRecoveredColumnFraction": round(
            float(np.count_nonzero(recovered & (seed_count == 1))) / recovered_count,
            6,
        ),
        "multipleSeedRecoveredColumnFraction": round(
            float(np.count_nonzero(recovered & (seed_count > 1))) / recovered_count,
            6,
        ),
        "seedCount": int(np.count_nonzero(seed)),
        "componentCount": int(len(component_values)),
        "largestComponentSeedFraction": round(largest_component_fraction, 6),
        "axialNormalErrorDegrees": _percentiles(axial_angle[seed]),
        "sheetCenterPlaneResidualVoxels": _percentiles(plane_residual),
        "sheetCenterPlaneXYZ": [round(float(value), 8) for value in coefficient],
        "thicknessVoxels": _percentiles(thickness[seed]),
    }
    gates = {
        "coverageAtLeast70Percent": metrics["highConfidenceColumnCoverage"] >= 0.7,
        "singleSheetColumnsAtLeast95Percent": metrics[
            "multipleSeedRecoveredColumnFraction"
        ]
        <= 0.05,
        "medianNormalErrorAtMost10Degrees": metrics[
            "axialNormalErrorDegrees"
        ]["median"]
        <= 10.0,
        "p90NormalErrorAtMost20Degrees": metrics[
            "axialNormalErrorDegrees"
        ]["p90"]
        <= 20.0,
        "largestComponentAtLeast70Percent": largest_component_fraction >= 0.7,
        "p90PlaneResidualAtMostTwoSamples": metrics[
            "sheetCenterPlaneResidualVoxels"
        ]["p90"]
        <= 2.0 * stride,
    }
    gates["passed"] = bool(all(gates.values()))

    center = np.asarray(volume[depth // 2])
    gray = _display_gray(center, valid)
    first_panel = np.repeat(gray[:, :, None], 3, axis=2)
    first_panel[~valid] = (7, 10, 14)
    for index in best_index:
        x = int(np.clip(round(local[index, 0]), 0, width - 1))
        y = int(np.clip(round(local[index, 1]), 0, height - 1))
        color = (68, 232, 187) if axial_angle[index] <= 15.0 else (255, 151, 74)
        first_panel[max(0, y - 1) : min(height, y + 2), max(0, x - 1) : min(width, x + 2)] = color
    coverage_panel = np.full_like(first_panel, (7, 10, 14))
    valid_expanded = np.repeat(
        np.repeat(valid_grid, stride, axis=0), stride, axis=1
    )[:height, :width]
    recovered_expanded = np.repeat(
        np.repeat(recovered, stride, axis=0), stride, axis=1
    )[:height, :width]
    coverage_panel[valid_expanded] = (106, 48, 52)
    coverage_panel[recovered_expanded] = (45, 177, 143)
    preview_path.write_bytes(rgb_png(np.concatenate((first_panel, coverage_panel), axis=1)))

    manifest = {
        "schema": KNOWN_SURFACE_AUDIT_SCHEMA,
        "version": KNOWN_SURFACE_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "expectation": {
            "description": (
                "one known-unrolled papyrus surface: one ribbon per valid chart "
                "column, chart-depth normal, and one dominant connected sheet"
            ),
            "expectedNormalXYZ": [0.0, 0.0, 1.0],
            "expectedSheetCountPerValidColumn": 1,
        },
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"coverageAndNormalPreview": preview_path.name},
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, manifest)
    return manifest


def run_known_surface_interface_audit(
    truth_root: str | Path,
    interface_root: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Audit raw face evidence without pretending it is already a sheet."""

    truth_path = _resolve_manifest(truth_root, KNOWN_SURFACE_CROP_STEM)
    interface_path = _resolve_manifest(interface_root, MATERIAL_INTERFACE_STEM)
    truth = json.loads(truth_path.read_text())
    interfaces = json.loads(interface_path.read_text())
    if truth.get("schema") != KNOWN_SURFACE_CROP_SCHEMA or truth.get("state") != "complete":
        raise ValueError("known-surface truth crop must be complete")
    if (
        interfaces.get("schema") != MATERIAL_INTERFACE_SCHEMA
        or interfaces.get("state") != "complete"
    ):
        raise ValueError("material-interface field must be complete")
    truth_data_path = truth_path.parent / str(truth["data"]["path"])
    interface_data_path = interface_path.parent / str(interfaces["data"]["path"])
    if sha256_file(truth_data_path) != truth["data"]["sha256"]:
        raise ValueError("known-surface truth data changed after preparation")
    if sha256_file(interface_data_path) != interfaces["data"]["sha256"]:
        raise ValueError("material-interface data changed after detection")

    identity: dict[str, Any] = {
        "schema": KNOWN_SURFACE_INTERFACE_AUDIT_SCHEMA,
        "version": KNOWN_SURFACE_INTERFACE_AUDIT_VERSION,
        "truthManifestPath": str(truth_path),
        "truthManifestSha256": sha256_file(truth_path),
        "interfaceManifestPath": str(interface_path),
        "interfaceManifestSha256": sha256_file(interface_path),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{KNOWN_SURFACE_INTERFACE_AUDIT_STEM}.json"
    preview_path = output / "known-surface-material-interface-audit.png"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
        ):
            return cached

    started = time.monotonic()
    volume = np.load(truth_data_path, mmap_mode="r")
    valid = np.max(volume, axis=0) > 0
    with np.load(interface_data_path, allow_pickle=False) as stored:
        position = np.asarray(stored["positionXYZ"], dtype=np.float64)
        normal = np.asarray(stored["signedNormalXYZ"], dtype=np.float64)
        key = np.asarray(stored["processingKeyXYZ"], dtype=np.int64)
        evidence = np.asarray(stored["localEvidenceScore"], dtype=np.float64)

    origin = np.asarray(truth["geometry"]["originXYZ"], dtype=np.float64)
    local = position - origin[None, :]
    depth, height, width = (int(value) for value in volume.shape)
    expected_depth = 0.5 * (depth - 1)
    stride = int(interfaces["identity"]["settings"]["sampling_stride_voxels"])
    valid_grid = _coverage_grid(valid, stride)
    grid_height, grid_width = valid_grid.shape
    grid_x = np.clip(key[:, 0], 0, grid_width - 1)
    grid_y = np.clip(key[:, 1], 0, grid_height - 1)
    linear = grid_y * grid_width + grid_x
    depth_residual = np.abs(local[:, 2] - expected_depth)
    axial_angle = np.degrees(
        np.arccos(np.clip(np.abs(normal[:, 2]), 0.0, 1.0))
    )
    cost = (
        depth_residual / max(stride, 1)
        + axial_angle / 30.0
        - 0.15 * evidence
    )
    best = _best_per_column(linear, -cost)
    best_linear = linear[best]
    on_valid = valid_grid.reshape(-1)[best_linear]
    best = best[on_valid]
    best_linear = best_linear[on_valid]
    near = depth_residual[best] <= 2.0 * stride
    recovered_linear = best_linear[near]
    recovered = np.zeros(valid_grid.size, dtype=bool)
    recovered[recovered_linear] = True
    recovered = recovered.reshape(valid_grid.shape) & valid_grid
    valid_count = max(int(np.count_nonzero(valid_grid)), 1)
    selected = best[near]
    metrics = {
        "validSamplingColumnCount": int(np.count_nonzero(valid_grid)),
        "candidateSamplingColumnCount": int(len(best)),
        "nearSurfaceSamplingColumnCount": int(len(selected)),
        "candidateColumnCoverage": round(len(best) / valid_count, 6),
        "nearSurfaceColumnCoverage": round(len(selected) / valid_count, 6),
        "expectedSurfaceDepthIndex": expected_depth,
        "selectedDepthResidualVoxels": _percentiles(depth_residual[selected]),
        "selectedRawAxialNormalErrorDegrees": _percentiles(
            axial_angle[selected]
        ),
        "selectedLocalEvidenceScore": _percentiles(evidence[selected]),
    }
    gates = {
        "nearSurfaceCoverageAtLeast80Percent": (
            metrics["nearSurfaceColumnCoverage"] >= 0.8
        ),
        "p90DepthResidualAtMostTwoSamples": (
            metrics["selectedDepthResidualVoxels"]["p90"] <= 2.0 * stride
        ),
        "rawNormalP90AtMost20Degrees": (
            metrics["selectedRawAxialNormalErrorDegrees"]["p90"] <= 20.0
        ),
    }
    gates["passed"] = bool(all(gates.values()))

    center = np.asarray(volume[int(round(expected_depth))])
    gray = _display_gray(center, valid)
    evidence_panel = np.repeat(gray[:, :, None], 3, axis=2)
    evidence_panel[~valid] = (7, 10, 14)
    for index in selected:
        x = int(np.clip(round(local[index, 0]), 0, width - 1))
        y = int(np.clip(round(local[index, 1]), 0, height - 1))
        color = (68, 232, 187) if axial_angle[index] <= 20.0 else (255, 151, 74)
        evidence_panel[
            max(0, y - 1) : min(height, y + 2),
            max(0, x - 1) : min(width, x + 2),
        ] = color
    coverage_panel = np.full_like(evidence_panel, (7, 10, 14))
    valid_expanded = np.repeat(
        np.repeat(valid_grid, stride, axis=0), stride, axis=1
    )[:height, :width]
    recovered_expanded = np.repeat(
        np.repeat(recovered, stride, axis=0), stride, axis=1
    )[:height, :width]
    coverage_panel[valid_expanded] = (106, 48, 52)
    coverage_panel[recovered_expanded] = (45, 177, 143)
    preview_path.write_bytes(
        rgb_png(np.concatenate((evidence_panel, coverage_panel), axis=1))
    )

    manifest = {
        "schema": KNOWN_SURFACE_INTERFACE_AUDIT_SCHEMA,
        "version": KNOWN_SURFACE_INTERFACE_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "expectation": {
            "description": (
                "the known chart center is a surface location; this audit tests "
                "face-position availability separately from orientation quality"
            ),
            "expectedNormalXYZ": [0.0, 0.0, 1.0],
            "expectedSurfaceDepthIndex": expected_depth,
        },
        "metrics": metrics,
        "gates": gates,
        "artifacts": {"coverageAndRawNormalPreview": preview_path.name},
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, manifest)
    return manifest
