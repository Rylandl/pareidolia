from __future__ import annotations

import colorsys
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .material_interface import MATERIAL_INTERFACE_SCHEMA, MATERIAL_INTERFACE_STEM


MACRO_ORIENTATION_SCHEMA = "pareidolia.macro-sheet-orientation-field"
MACRO_ORIENTATION_VERSION = 1
MACRO_ORIENTATION_STEM = "macro-sheet-orientation-field-v1"


@dataclass(frozen=True, slots=True)
class MacroOrientationSettings:
    # Three minimum ply thicknesses: large enough to average fiber texture, but
    # still small relative to the scroll's macroscopic curvature.
    support_microns: float = 240.0
    minimum_interface_samples: int = 12
    minimum_orientation_confidence: float = 0.35
    minimum_sample_weight: float = 0.05

    def __post_init__(self) -> None:
        if not math.isfinite(self.support_microns) or self.support_microns <= 0.0:
            raise ValueError("macro orientation support must be finite and positive")
        if self.minimum_interface_samples < 3:
            raise ValueError("macro orientation requires at least three samples")
        if not 0.0 <= self.minimum_orientation_confidence <= 1.0:
            raise ValueError("macro orientation confidence must lie in [0, 1]")
        if not 0.0 < self.minimum_sample_weight <= 1.0:
            raise ValueError("macro sample weight must lie in (0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_interface_manifest(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{MATERIAL_INTERFACE_STEM}.json"


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _percentiles(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "minimum": None,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    quantile = np.percentile(finite, (0, 50, 90, 99, 100))
    return {
        "count": int(len(finite)),
        **{
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "median", "p90", "p99", "maximum"), quantile
            )
        },
    }


def _canonical_axial(normals: np.ndarray) -> np.ndarray:
    result = np.asarray(normals, dtype=np.float64).copy()
    dominant = np.argmax(np.abs(result), axis=1)
    sign = np.where(result[np.arange(len(result)), dominant] >= 0.0, 1.0, -1.0)
    result *= sign[:, None]
    return result


def _draw_segment(
    canvas: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    length = max(abs(second[0] - first[0]), abs(second[1] - first[1]))
    steps = max(2, int(math.ceil(length)) + 1)
    x = np.rint(np.linspace(first[0], second[0], steps)).astype(np.int32)
    y = np.rint(np.linspace(first[1], second[1], steps)).astype(np.int32)
    valid = (
        (x >= 0)
        & (x < canvas.shape[1])
        & (y >= 0)
        & (y < canvas.shape[0])
    )
    canvas[y[valid], x[valid]] = color


def write_macro_orientation_projection(
    arrays: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    path: str | Path,
    *,
    support_voxels: float,
    panel_size: int = 640,
) -> Path:
    """Render in-slice sheet tangents, never the through-thickness normal."""

    output = Path(path)
    center = np.asarray(arrays["centerXYZ"], dtype=np.float64)
    normal = np.asarray(arrays["normalXYZ"], dtype=np.float64)
    confidence = np.asarray(arrays["orientationConfidence"], dtype=np.float64)
    trusted = np.asarray(arrays["trusted"], dtype=bool)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    middle = 0.5 * (world_start_xyz + world_stop_xyz)
    views = ((0, 1, 2), (0, 2, 1), (1, 2, 0))
    for panel, (first_axis, second_axis, slice_axis) in enumerate(views):
        offset = panel * panel_size
        selected = np.flatnonzero(
            np.abs(center[:, slice_axis] - middle[slice_axis])
            <= 0.75 * support_voxels
        )
        for index in selected:
            projected_normal = normal[index, [first_axis, second_axis]]
            projected_length = float(np.linalg.norm(projected_normal))
            # When the normal points almost out of the slice, the sheet is
            # nearly parallel to the slice and its intersection direction is
            # numerically undefined. Drawing an arbitrary long line here is
            # exactly the misleading perpendicular-cluster visual we want to
            # avoid.
            if projected_length < 0.25:
                continue
            tangent = np.asarray(
                (-projected_normal[1], projected_normal[0]), dtype=np.float64
            ) / projected_length
            normalized = (
                center[index, [first_axis, second_axis]]
                - world_start_xyz[[first_axis, second_axis]]
            ) / width[[first_axis, second_axis]]
            point = np.asarray(
                (
                    offset
                    + margin
                    + normalized[0] * (panel_size - 2 * margin),
                    panel_size
                    - margin
                    - normalized[1] * (panel_size - 2 * margin),
                )
            )
            pixel_scale = np.asarray(
                (
                    (panel_size - 2 * margin) / width[first_axis],
                    (panel_size - 2 * margin) / width[second_axis],
                )
            )
            delta = tangent * pixel_scale * 0.38 * support_voxels
            hue = 0.48 if trusted[index] else 0.08
            saturation = 0.72
            value = 0.55 + 0.43 * confidence[index]
            color = tuple(
                int(round(255.0 * channel))
                for channel in colorsys.hsv_to_rgb(hue, saturation, value)
            )
            _draw_segment(canvas, tuple(point - delta), tuple(point + delta), color)
        border = (64, 72, 84)
        canvas[margin, offset + margin : offset + panel_size - margin] = border
        canvas[
            panel_size - margin,
            offset + margin : offset + panel_size - margin,
        ] = border
        canvas[margin : panel_size - margin, offset + margin] = border
        canvas[
            margin : panel_size - margin, offset + panel_size - margin
        ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def run_macro_orientation_field(
    interface_root: str | Path,
    output_path: str | Path,
    *,
    settings: MacroOrientationSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve_interface_manifest(interface_root)
    interface_manifest = json.loads(interface_path.read_text())
    if (
        interface_manifest.get("schema") != MATERIAL_INTERFACE_SCHEMA
        or interface_manifest.get("state") != "complete"
    ):
        raise ValueError("macro orientation requires a complete interface field")
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    if sha256_file(interface_data_path) != interface_manifest["data"]["sha256"]:
        raise ValueError("material-interface data changed after extraction")
    resolved = settings or MacroOrientationSettings()
    voxel_size = float(interface_manifest["source"]["voxelSizeMicrons"])
    stride = int(
        interface_manifest["identity"]["settings"]["sampling_stride_voxels"]
    )
    sampling_microns = voxel_size * stride
    support_sampling_steps = max(
        2, int(round(resolved.support_microns / sampling_microns))
    )
    support_voxels = support_sampling_steps * stride
    identity: dict[str, Any] = {
        "schema": MACRO_ORIENTATION_SCHEMA,
        "version": MACRO_ORIENTATION_VERSION,
        "interfaces": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "resolvedScale": {
            "samplingStrideVoxels": stride,
            "samplingStepMicrons": sampling_microns,
            "supportSamplingSteps": support_sampling_steps,
            "supportVoxels": support_voxels,
            "actualSupportMicrons": support_voxels * voxel_size,
        },
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MACRO_ORIENTATION_STEM}.json"
    data_path = output / f"{MACRO_ORIENTATION_STEM}.npz"
    preview_path = output / "macro-sheet-tangents.png"
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
    with np.load(interface_data_path, allow_pickle=False) as stored:
        position = np.asarray(stored["positionXYZ"], dtype=np.float64)
        normal = np.asarray(stored["signedNormalXYZ"], dtype=np.float64)
        key = np.asarray(stored["processingKeyXYZ"], dtype=np.int64)
        evidence = np.asarray(stored["localEvidenceScore"], dtype=np.float64)
    if not len(position):
        raise ValueError("material-interface field contains no samples")
    bin_key = key // support_sampling_steps
    unique_key, sample_bin_index = np.unique(
        bin_key, axis=0, return_inverse=True
    )
    bin_count = len(unique_key)
    weight = np.clip(evidence, resolved.minimum_sample_weight, 1.0)
    total_weight = np.bincount(
        sample_bin_index, weights=weight, minlength=bin_count
    )
    sample_count = np.bincount(sample_bin_index, minlength=bin_count)
    center_total = np.zeros((bin_count, 3), dtype=np.float64)
    for axis in range(3):
        np.add.at(center_total[:, axis], sample_bin_index, weight * position[:, axis])
    center = center_total / np.maximum(total_weight[:, None], 1.0e-12)
    tensor = np.zeros((bin_count, 3, 3), dtype=np.float64)
    for first in range(3):
        for second in range(first, 3):
            total = np.bincount(
                sample_bin_index,
                weights=weight * normal[:, first] * normal[:, second],
                minlength=bin_count,
            )
            tensor[:, first, second] = total
            tensor[:, second, first] = total
    tensor /= np.maximum(total_weight[:, None, None], 1.0e-12)
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    macro_normal = _canonical_axial(eigenvector[:, :, -1])
    confidence = (eigenvalue[:, 2] - eigenvalue[:, 1]) / np.maximum(
        eigenvalue[:, 2], 1.0e-12
    )
    trusted = (
        (sample_count >= resolved.minimum_interface_samples)
        & (confidence >= resolved.minimum_orientation_confidence)
    )
    sample_residual = np.degrees(
        np.arccos(
            np.clip(
                np.abs(
                    np.einsum(
                        "ij,ij->i", normal, macro_normal[sample_bin_index]
                    )
                ),
                0.0,
                1.0,
            )
        )
    )
    arrays = {
        "binKeyXYZ": unique_key.astype(np.int32),
        "centerXYZ": center.astype(np.float32),
        "normalXYZ": macro_normal.astype(np.float32),
        "eigenvaluesAscending": eigenvalue.astype(np.float32),
        "orientationConfidence": confidence.astype(np.float32),
        "sampleCount": sample_count.astype(np.int32),
        "evidenceWeight": total_weight.astype(np.float32),
        "trusted": trusted.astype(np.uint8),
        "sampleMacroBinIndex": sample_bin_index.astype(np.int32),
        "sampleNormalResidualDegrees": sample_residual.astype(np.float32),
    }
    _write_npz(data_path, arrays)
    owned = interface_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(owned["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(owned["stopXYZExclusive"], dtype=np.float64)
    write_macro_orientation_projection(
        arrays,
        world_start,
        world_stop,
        preview_path,
        support_voxels=support_voxels,
    )
    payload: dict[str, Any] = {
        "schema": MACRO_ORIENTATION_SCHEMA,
        "version": MACRO_ORIENTATION_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interface_manifest["source"],
        "geometry": {
            **interface_manifest["geometry"],
            "supportSamplingSteps": support_sampling_steps,
            "supportVoxels": support_voxels,
            "actualSupportMicrons": support_voxels * voxel_size,
        },
        "counts": {
            "interfaceSampleCount": int(len(position)),
            "orientationBinCount": int(bin_count),
            "trustedOrientationBinCount": int(np.count_nonzero(trusted)),
            "trustedOrientationBinFraction": round(float(np.mean(trusted)), 6),
            "samplesInTrustedBins": int(np.count_nonzero(trusted[sample_bin_index])),
        },
        "distributions": {
            "sampleCountPerBin": _percentiles(sample_count),
            "orientationConfidence": _percentiles(confidence),
            "rawSampleToMacroNormalResidualDegrees": _percentiles(sample_residual),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"tangentProjection": preview_path.name},
        "method": {
            "input": "dense signed material-interface normals",
            "orientationGauge": "unsigned outer-product tensor",
            "normalChoice": "principal eigenvector",
            "multimodalitySignal": "principal-versus-secondary eigenvalue gap",
            "visualizedVector": "intersection of tangent plane with slice plane",
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
