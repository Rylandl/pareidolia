from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.rectify import gaussian_blur_3d

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import (
    IsolatedSlabSettings,
    _downsample_mean_zyx,
    _owned_sample_slices,
    _percentile_record,
    _processing_bounds,
    otsu_material_calibration,
)
from .one_sided_interface import (
    extract_one_sided_interfaces,
    write_one_sided_projection,
)


MATERIAL_INTERFACE_SCHEMA = "pareidolia.dense-material-interface-field"
MATERIAL_INTERFACE_VERSION = 1
MATERIAL_INTERFACE_STEM = "material-interface-field-v1"


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_material_interface_field(
    source_path: str | Path,
    output_path: str | Path,
    *,
    world_start_xyz: tuple[int, int, int],
    world_stop_xyz_exclusive: tuple[int, int, int],
    metadata_path: str | Path | None = None,
    settings: IsolatedSlabSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Extract dense signed air-to-material faces without requiring a mate."""

    source = VolumeSource.open(source_path, metadata_path)
    resolved = settings or IsolatedSlabSettings.at_physical_scale(
        source.voxel_size_microns
    )
    source_origin = np.asarray(source.origin_xyz, dtype=np.int64)
    owned = VoxelBounds(
        tuple((np.asarray(world_start_xyz, dtype=np.int64) - source_origin).tolist()),
        tuple(
            (
                np.asarray(world_stop_xyz_exclusive, dtype=np.int64)
                - source_origin
            ).tolist()
        ),
    )
    if any(owned.start_xyz[axis] < 0 for axis in range(3)) or any(
        owned.stop_xyz_exclusive[axis] > source.shape_xyz[axis]
        for axis in range(3)
    ):
        raise ValueError("owned world bounds lie outside the CT source")
    processing = _processing_bounds(source, owned, resolved)
    implementation = {
        "material_interface.py": sha256_file(Path(__file__)),
        "one_sided_interface.py": sha256_file(
            Path(__file__).with_name("one_sided_interface.py")
        ),
        "isolated_slab.py": sha256_file(Path(__file__).with_name("isolated_slab.py")),
    }
    identity: dict[str, Any] = {
        "schema": MATERIAL_INTERFACE_SCHEMA,
        "version": MATERIAL_INTERFACE_VERSION,
        "source": dict(source.source_identity),
        "worldBounds": {
            "startXYZ": list(world_start_xyz),
            "stopXYZExclusive": list(world_stop_xyz_exclusive),
        },
        "settings": resolved.record(),
        "implementationSha256": implementation,
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_INTERFACE_STEM}.json"
    data_path = output / f"{MATERIAL_INTERFACE_STEM}.npz"
    preview_path = output / "material-interface-projection.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    timing: dict[str, float] = {}
    stage = time.monotonic()
    stride = resolved.sampling_stride_voxels
    sampled = _downsample_mean_zyx(
        source.memmap()[processing.slices_zyx], stride
    )
    smoothed = gaussian_blur_3d(
        sampled, resolved.smoothing_sigma_voxels / stride
    )
    owned_slices = _owned_sample_slices(owned, processing, stride)
    calibration = otsu_material_calibration(smoothed[owned_slices])
    calibration["displayHighRaw"] = float(
        np.percentile(sampled[owned_slices], 99.5)
    )
    if resolved.material_threshold_raw is not None:
        calibration["materialThresholdRaw"] = float(
            resolved.material_threshold_raw
        )
        calibration["method"] = "explicit-threshold-with-otsu-class-scale"
    timing["ctPreparation"] = time.monotonic() - stage

    stage = time.monotonic()
    counts, raw = extract_one_sided_interfaces(
        smoothed,
        threshold=float(calibration["materialThresholdRaw"]),
        class_contrast=float(calibration["classContrastRaw"]),
        voxel_size_microns=source.voxel_size_microns,
        isolated_settings=resolved,
    )
    half = 0.5 * (stride - 1)
    processing_start = np.asarray(processing.start_xyz, dtype=np.float32)
    source_shift = np.asarray(source.origin_xyz, dtype=np.float32)
    position_world = (
        np.asarray(raw["position"], dtype=np.float32) * stride
        + processing_start[None, :]
        + source_shift[None, :]
        + half
    )
    owned_start = np.asarray(world_start_xyz, dtype=np.float32)
    owned_stop = np.asarray(world_stop_xyz_exclusive, dtype=np.float32)
    inside = np.all(
        (position_world >= owned_start[None, :])
        & (position_world < owned_stop[None, :]),
        axis=1,
    )
    selected = np.flatnonzero(inside)
    counts["ownedInterfaceCount"] = int(len(selected))
    arrays = {
        "positionXYZ": position_world[selected].astype(np.float32),
        "signedNormalXYZ": np.asarray(raw["normal"])[selected].astype(np.float32),
        "processingKeyXYZ": np.asarray(raw["processing_key"])[selected].astype(
            np.int32
        ),
        "localEvidenceScore": np.asarray(raw["confidence"])[selected].astype(
            np.float32
        ),
        "airMarginClassFraction": np.asarray(raw["air_margin"])[selected].astype(
            np.float32
        ),
        "airSampleFraction": np.asarray(raw["air_sample_fraction"])[selected].astype(
            np.float32
        ),
        "materialMarginClassFraction": np.asarray(raw["material_margin"])[
            selected
        ].astype(np.float32),
        "gradientClassFraction": np.asarray(raw["gradient_class_fraction"])[
            selected
        ].astype(np.float32),
    }
    timing["interfaceExtraction"] = time.monotonic() - stage
    stage = time.monotonic()
    _write_npz(data_path, arrays)

    preview_arrays = {
        **arrays,
        "seedSurfaceLabel": np.full(len(selected), -1, dtype=np.int32),
    }
    write_one_sided_projection(
        preview_arrays,
        owned_start,
        owned_stop,
        preview_path,
    )
    timing["writingAndArtifact"] = time.monotonic() - stage
    timing["total"] = time.monotonic() - started

    processing_shape_sampling_xyz = [
        int(value // stride) for value in processing.shape_xyz
    ]
    payload: dict[str, Any] = {
        "schema": MATERIAL_INTERFACE_SCHEMA,
        "version": MATERIAL_INTERFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": {
            "path": str(source.path),
            "metadataPath": (
                str(source.metadata_path) if source.metadata_path is not None else None
            ),
            "shapeZYX": list(source.shape_zyx),
            "sourceOriginXYZ": list(source.origin_xyz),
            "voxelSizeMicrons": source.voxel_size_microns,
        },
        "geometry": {
            "ownedVoxelBounds": owned.record(),
            "ownedWorldBounds": {
                "startXYZ": list(world_start_xyz),
                "stopXYZExclusive": list(world_stop_xyz_exclusive),
            },
            "processingVoxelBounds": processing.record(),
            "processingShapeSamplingXYZ": processing_shape_sampling_xyz,
            "coordinateUnit": "source-voxel",
        },
        "calibration": calibration,
        "counts": counts,
        "distributions": {
            "localEvidenceScore": _percentile_record(arrays["localEvidenceScore"]),
            "airMarginClassFraction": _percentile_record(
                arrays["airMarginClassFraction"]
            ),
            "materialMarginClassFraction": _percentile_record(
                arrays["materialMarginClassFraction"]
            ),
            "gradientClassFraction": _percentile_record(
                arrays["gradientClassFraction"]
            ),
        },
        "timingSeconds": {
            name: round(value, 6) for name, value in timing.items()
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"projection": preview_path.name},
        "method": {
            "representation": "signed air-to-material face samples",
            "requiresOppositeFace": False,
            "collapsesPairedFacesToMidpoint": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
