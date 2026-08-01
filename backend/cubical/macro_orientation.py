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
from .isolated_slab import ISOLATED_SLAB_SCHEMA, ISOLATED_SLAB_STEM
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
    minimum_physical_profile_confidence: float = 0.5
    minimum_physical_mode_samples: int = 4
    minimum_physical_mode_confidence: float = 0.5
    physical_mode_radius_degrees: float = 25.0
    maximum_sample_to_physical_mode_degrees: float = 50.0
    maximum_physical_modes_per_bin: int = 4

    def __post_init__(self) -> None:
        if not math.isfinite(self.support_microns) or self.support_microns <= 0.0:
            raise ValueError("macro orientation support must be finite and positive")
        if self.minimum_interface_samples < 3:
            raise ValueError("macro orientation requires at least three samples")
        if not 0.0 <= self.minimum_orientation_confidence <= 1.0:
            raise ValueError("macro orientation confidence must lie in [0, 1]")
        if not 0.0 < self.minimum_sample_weight <= 1.0:
            raise ValueError("macro sample weight must lie in (0, 1]")
        if not 0.0 <= self.minimum_physical_profile_confidence <= 1.0:
            raise ValueError("physical profile confidence must lie in [0, 1]")
        if self.minimum_physical_mode_samples < 2:
            raise ValueError("physical orientation modes require repeated profiles")
        if not 0.0 <= self.minimum_physical_mode_confidence <= 1.0:
            raise ValueError("physical mode confidence must lie in [0, 1]")
        for value in (
            self.physical_mode_radius_degrees,
            self.maximum_sample_to_physical_mode_degrees,
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise ValueError("physical orientation angles must lie in (0, 90)")
        if self.maximum_physical_modes_per_bin < 1:
            raise ValueError("at least one physical mode per bin must be permitted")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_interface_manifest(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{MATERIAL_INTERFACE_STEM}.json"


def _resolve_slab_manifest(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{ISOLATED_SLAB_STEM}.json"


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


def _axial_mode(
    normals: np.ndarray,
    weights: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float, np.ndarray]:
    """Fit one unsigned normal mode and return its residuals."""

    values = np.asarray(normals, dtype=np.float64)
    weight = np.asarray(weights, dtype=np.float64)
    tensor = np.einsum("i,ij,ik->jk", weight, values, values)
    tensor /= max(float(np.sum(weight)), 1.0e-12)
    eigenvalue, eigenvector = np.linalg.eigh(tensor)
    normal = _canonical_axial(eigenvector[:, -1][None, :])[0]
    confidence = float(
        (eigenvalue[-1] - eigenvalue[-2]) / max(eigenvalue[-1], 1.0e-12)
    )
    residual = np.degrees(
        np.arccos(np.clip(np.abs(values @ normal), 0.0, 1.0))
    )
    return normal, eigenvalue, confidence, residual


def _physical_orientation_modes(
    bin_index: np.ndarray,
    position: np.ndarray,
    normal: np.ndarray,
    confidence: np.ndarray,
    *,
    bin_count: int,
    settings: MacroOrientationSettings,
) -> list[dict[str, Any]]:
    """Extract repeated axial modes without collapsing a local hairpin.

    A macro bin may contain more than one legitimate page normal.  Modes are
    peeled from the strongest remaining physical profile, refit as an axial
    tensor, and accepted only when repeated profiles remain coherent.
    """

    accepted_profile = confidence >= settings.minimum_physical_profile_confidence
    order = np.argsort(bin_index, kind="stable")
    sorted_bin = bin_index[order]
    boundary = np.flatnonzero(
        np.concatenate(
            (
                np.ones(1, dtype=bool),
                sorted_bin[1:] != sorted_bin[:-1],
                np.ones(1, dtype=bool),
            )
        )
    )
    modes: list[dict[str, Any]] = []
    radius_cosine = math.cos(math.radians(settings.physical_mode_radius_degrees))
    for start, stop in zip(boundary[:-1], boundary[1:]):
        rows = order[start:stop]
        rows = rows[accepted_profile[rows]]
        if len(rows) < settings.minimum_physical_mode_samples:
            continue
        remaining = rows.copy()
        bin_modes: list[dict[str, Any]] = []
        while (
            len(remaining) >= settings.minimum_physical_mode_samples
            and len(bin_modes) < settings.maximum_physical_modes_per_bin
        ):
            seed = int(remaining[np.argmax(confidence[remaining])])
            axis = normal[seed]
            member_mask = np.abs(normal[remaining] @ axis) >= radius_cosine
            members = remaining[member_mask]
            if len(members) < settings.minimum_physical_mode_samples:
                remaining = remaining[~member_mask]
                if not np.any(member_mask):
                    remaining = remaining[remaining != seed]
                continue
            # Refit and reselect twice so the result is centered on the mode,
            # not on whichever profile happened to seed it.
            for _ in range(2):
                axis, _, _, _ = _axial_mode(
                    normal[members], confidence[members]
                )
                member_mask = np.abs(normal[remaining] @ axis) >= radius_cosine
                members = remaining[member_mask]
                if len(members) < settings.minimum_physical_mode_samples:
                    break
            if len(members) < settings.minimum_physical_mode_samples:
                remaining = remaining[remaining != seed]
                continue
            axis, eigenvalue, mode_confidence, residual = _axial_mode(
                normal[members], confidence[members]
            )
            coherent = (
                mode_confidence >= settings.minimum_physical_mode_confidence
                and float(np.percentile(residual, 90))
                <= settings.physical_mode_radius_degrees
            )
            remaining = remaining[~member_mask]
            if not coherent:
                continue
            weight = confidence[members]
            center = np.average(position[members], axis=0, weights=weight)
            bin_modes.append(
                {
                    "binIndex": int(bin_index[members[0]]),
                    "centerXYZ": center,
                    "normalXYZ": axis,
                    "eigenvaluesAscending": eigenvalue,
                    "orientationConfidence": mode_confidence,
                    "sampleCount": int(len(members)),
                    "evidenceWeight": float(np.sum(weight)),
                    "normalResidualP90Degrees": float(
                        np.percentile(residual, 90)
                    ),
                }
            )
        bin_modes.sort(
            key=lambda value: (
                -float(value["evidenceWeight"]),
                -int(value["sampleCount"]),
            )
        )
        modes.extend(bin_modes)
    return modes


def sample_orientation_field(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Resolve either a legacy single-mode or guided multimodal field."""

    sample_bin = np.asarray(arrays["sampleMacroBinIndex"], dtype=np.int64)
    if "sampleOrientationXYZ" not in arrays:
        normal = np.asarray(arrays["normalXYZ"], dtype=np.float64)[sample_bin]
        confidence = np.asarray(
            arrays["orientationConfidence"], dtype=np.float64
        )[sample_bin]
        trusted = np.asarray(arrays["trusted"], dtype=bool)[sample_bin]
        center = np.asarray(arrays["centerXYZ"], dtype=np.float64)[sample_bin]
        return {
            "sampleBinIndex": sample_bin,
            "sampleGroupIndex": sample_bin,
            "sampleNormalXYZ": normal,
            "sampleOrientationConfidence": confidence,
            "sampleTrusted": trusted,
            "sampleGroupCenterXYZ": center,
            "sampleOrientationSource": np.zeros(len(sample_bin), dtype=np.uint8),
        }
    group = np.asarray(arrays["sampleOrientationGroupIndex"], dtype=np.int64)
    group_center = np.asarray(
        arrays["orientationGroupCenterXYZ"], dtype=np.float64
    )
    return {
        "sampleBinIndex": sample_bin,
        "sampleGroupIndex": group,
        "sampleNormalXYZ": np.asarray(
            arrays["sampleOrientationXYZ"], dtype=np.float64
        ),
        "sampleOrientationConfidence": np.asarray(
            arrays["sampleOrientationConfidence"], dtype=np.float64
        ),
        "sampleTrusted": np.asarray(arrays["sampleOrientationTrusted"], dtype=bool),
        "sampleGroupCenterXYZ": group_center[group],
        "sampleOrientationSource": np.asarray(
            arrays["sampleOrientationSource"], dtype=np.uint8
        ),
    }


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
    guided = "orientationGroupCenterXYZ" in arrays
    center = np.asarray(
        arrays[
            "orientationGroupCenterXYZ" if guided else "centerXYZ"
        ],
        dtype=np.float64,
    )
    normal = np.asarray(
        arrays["orientationGroupNormalXYZ" if guided else "normalXYZ"],
        dtype=np.float64,
    )
    confidence = np.asarray(
        arrays[
            "orientationGroupConfidence"
            if guided
            else "orientationConfidence"
        ],
        dtype=np.float64,
    )
    trusted = np.asarray(
        arrays["orientationGroupTrusted" if guided else "trusted"], dtype=bool
    )
    source = (
        np.asarray(arrays["orientationGroupSource"], dtype=np.uint8)
        if guided
        else np.zeros(len(center), dtype=np.uint8)
    )
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
            if source[index] == 1:
                hue = 0.46
                saturation = 0.9
                value = 0.68 + 0.30 * confidence[index]
            elif trusted[index]:
                hue = 0.56
                saturation = 0.46
                value = 0.40 + 0.35 * confidence[index]
            else:
                hue = 0.08
                saturation = 0.72
                value = 0.38 + 0.28 * confidence[index]
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
    isolated_slab_root: str | Path | None = None,
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
    slab_path: Path | None = None
    slab_manifest: dict[str, Any] | None = None
    slab_data_path: Path | None = None
    if isolated_slab_root is not None:
        slab_path = _resolve_slab_manifest(isolated_slab_root)
        slab_manifest = json.loads(slab_path.read_text())
        if (
            slab_manifest.get("schema") != ISOLATED_SLAB_SCHEMA
            or slab_manifest.get("state") != "complete"
        ):
            raise ValueError("physical guidance requires complete isolated slabs")
        slab_data_path = slab_path.parent / str(slab_manifest["data"]["path"])
        if sha256_file(slab_data_path) != slab_manifest["data"]["sha256"]:
            raise ValueError("isolated-slab data changed after extraction")
        for field in ("source", "geometry"):
            if field not in slab_manifest:
                raise ValueError("isolated-slab geometry is incomplete")
        interface_source = interface_manifest["source"]
        slab_source = slab_manifest["source"]
        if (
            Path(str(interface_source["path"])).resolve()
            != Path(str(slab_source["path"])).resolve()
            or tuple(interface_source["sourceOriginXYZ"])
            != tuple(slab_source["sourceOriginXYZ"])
            or not math.isclose(
                float(interface_source["voxelSizeMicrons"]),
                float(slab_source["voxelSizeMicrons"]),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
            or interface_manifest["geometry"]["ownedWorldBounds"]
            != slab_manifest["geometry"]["ownedWorldBounds"]
            or interface_manifest["geometry"]["processingVoxelBounds"]
            != slab_manifest["geometry"]["processingVoxelBounds"]
        ):
            raise ValueError(
                "isolated slabs and material interfaces must share source-aligned geometry"
            )
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
        "physicalGuidance": (
            {
                "manifestPath": str(slab_path),
                "manifestSha256": sha256_file(slab_path),
                "dataSha256": slab_manifest["data"]["sha256"],
            }
            if slab_path is not None and slab_manifest is not None
            else None
        ),
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
    generic_sample_residual = np.degrees(
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
    physical_profile_count = 0
    mapped_physical_profile_count = 0
    physical_modes: list[dict[str, Any]] = []
    if slab_data_path is not None and slab_manifest is not None:
        with np.load(slab_data_path, allow_pickle=False) as stored:
            slab_position = np.asarray(stored["midpointXYZ"], dtype=np.float64)
            slab_normal = np.asarray(stored["normalXYZ"], dtype=np.float64)
            slab_confidence = np.asarray(stored["confidence"], dtype=np.float64)
            slab_component = np.asarray(stored["componentId"], dtype=np.int64)
        physical_profile_count = len(slab_position)
        processing_start = np.asarray(
            interface_manifest["geometry"]["processingVoxelBounds"]["startXYZ"],
            dtype=np.float64,
        ) + np.asarray(
            interface_manifest["source"]["sourceOriginXYZ"], dtype=np.float64
        )
        half = 0.5 * (stride - 1)
        slab_key = np.rint(
            (slab_position - processing_start[None, :] - half) / stride
        ).astype(np.int64)
        processing_shape = np.asarray(
            interface_manifest["geometry"]["processingShapeSamplingXYZ"],
            dtype=np.int64,
        )
        inside = np.all(
            (slab_key >= 0) & (slab_key < processing_shape[None, :]), axis=1
        )
        slab_bin_key = slab_key // support_sampling_steps
        bin_lookup = {
            tuple(int(value) for value in value): index
            for index, value in enumerate(unique_key)
        }
        slab_bin_index = np.asarray(
            [
                bin_lookup.get(tuple(int(value) for value in value), -1)
                if valid
                else -1
                for value, valid in zip(slab_bin_key, inside)
            ],
            dtype=np.int32,
        )
        mapped = (slab_bin_index >= 0) & (slab_component >= 0)
        mapped_physical_profile_count = int(np.count_nonzero(mapped))
        if np.any(mapped):
            physical_modes = _physical_orientation_modes(
                slab_bin_index[mapped],
                slab_position[mapped],
                slab_normal[mapped],
                slab_confidence[mapped],
                bin_count=bin_count,
                settings=resolved,
            )

    sample_group = sample_bin_index.astype(np.int32, copy=True)
    sample_orientation = macro_normal[sample_bin_index].copy()
    sample_orientation_confidence = confidence[sample_bin_index].copy()
    sample_orientation_trusted = trusted[sample_bin_index].copy()
    sample_orientation_source = np.zeros(len(position), dtype=np.uint8)
    sample_residual = generic_sample_residual.copy()
    mode_bin = np.asarray(
        [int(mode["binIndex"]) for mode in physical_modes], dtype=np.int32
    )
    mode_center = np.asarray(
        [mode["centerXYZ"] for mode in physical_modes], dtype=np.float64
    ).reshape(-1, 3)
    mode_normal = np.asarray(
        [mode["normalXYZ"] for mode in physical_modes], dtype=np.float64
    ).reshape(-1, 3)
    mode_eigenvalue = np.asarray(
        [mode["eigenvaluesAscending"] for mode in physical_modes],
        dtype=np.float64,
    ).reshape(-1, 3)
    mode_confidence = np.asarray(
        [float(mode["orientationConfidence"]) for mode in physical_modes],
        dtype=np.float64,
    )
    mode_profile_count = np.asarray(
        [int(mode["sampleCount"]) for mode in physical_modes], dtype=np.int32
    )
    mode_weight = np.asarray(
        [float(mode["evidenceWeight"]) for mode in physical_modes],
        dtype=np.float64,
    )
    mode_p90 = np.asarray(
        [float(mode["normalResidualP90Degrees"]) for mode in physical_modes],
        dtype=np.float64,
    )
    modes_by_bin: dict[int, list[int]] = {}
    for mode_index, bin_index_value in enumerate(mode_bin):
        modes_by_bin.setdefault(int(bin_index_value), []).append(mode_index)
    guided_bin = np.zeros(bin_count, dtype=bool)
    maximum_mode_angle = resolved.maximum_sample_to_physical_mode_degrees
    for bin_index_value, mode_indices in modes_by_bin.items():
        guided_bin[bin_index_value] = True
        rows = np.flatnonzero(sample_bin_index == bin_index_value)
        candidates = np.asarray(mode_indices, dtype=np.int64)
        cosine = np.abs(normal[rows] @ mode_normal[candidates].T)
        best_local = np.argmax(cosine, axis=1)
        best_mode = candidates[best_local]
        best_angle = np.degrees(
            np.arccos(np.clip(cosine[np.arange(len(rows)), best_local], 0.0, 1.0))
        )
        sample_group[rows] = (bin_count + best_mode).astype(np.int32)
        sample_orientation[rows] = mode_normal[best_mode]
        sample_orientation_confidence[rows] = mode_confidence[best_mode]
        sample_residual[rows] = best_angle
        matched = best_angle <= maximum_mode_angle
        sample_orientation_trusted[rows] = matched
        sample_orientation_source[rows] = np.where(matched, 1, 2).astype(
            np.uint8
        )

    group_center = np.concatenate((center, mode_center), axis=0)
    group_normal = np.concatenate((macro_normal, mode_normal), axis=0)
    group_confidence = np.concatenate((confidence, mode_confidence), axis=0)
    group_trusted = np.concatenate(
        (trusted & ~guided_bin, np.ones(len(physical_modes), dtype=bool))
    )
    group_source = np.concatenate(
        (
            np.zeros(bin_count, dtype=np.uint8),
            np.ones(len(physical_modes), dtype=np.uint8),
        )
    )
    group_bin = np.concatenate(
        (np.arange(bin_count, dtype=np.int32), mode_bin), axis=0
    )
    group_assigned_count = np.bincount(
        sample_group, minlength=len(group_center)
    ).astype(np.int32)
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
        "sampleGenericNormalResidualDegrees": generic_sample_residual.astype(
            np.float32
        ),
        "sampleOrientationGroupIndex": sample_group,
        "sampleOrientationXYZ": sample_orientation.astype(np.float32),
        "sampleOrientationConfidence": sample_orientation_confidence.astype(
            np.float32
        ),
        "sampleOrientationTrusted": sample_orientation_trusted.astype(np.uint8),
        "sampleOrientationSource": sample_orientation_source,
        "orientationGroupBinIndex": group_bin,
        "orientationGroupCenterXYZ": group_center.astype(np.float32),
        "orientationGroupNormalXYZ": group_normal.astype(np.float32),
        "orientationGroupConfidence": group_confidence.astype(np.float32),
        "orientationGroupTrusted": group_trusted.astype(np.uint8),
        "orientationGroupSource": group_source,
        "orientationGroupAssignedSampleCount": group_assigned_count,
        "physicalModeBinIndex": mode_bin,
        "physicalModeCenterXYZ": mode_center.astype(np.float32),
        "physicalModeNormalXYZ": mode_normal.astype(np.float32),
        "physicalModeEigenvaluesAscending": mode_eigenvalue.astype(np.float32),
        "physicalModeConfidence": mode_confidence.astype(np.float32),
        "physicalModeProfileCount": mode_profile_count,
        "physicalModeEvidenceWeight": mode_weight.astype(np.float32),
        "physicalModeNormalResidualP90Degrees": mode_p90.astype(np.float32),
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
            "physicalProfileCount": int(physical_profile_count),
            "mappedPhysicalProfileCount": int(mapped_physical_profile_count),
            "physicalModeCount": int(len(physical_modes)),
            "physicallyGuidedBinCount": int(np.count_nonzero(guided_bin)),
            "physicallyGuidedSampleCount": int(
                np.count_nonzero(sample_orientation_source == 1)
            ),
            "physicallyRejectedSampleCount": int(
                np.count_nonzero(sample_orientation_source == 2)
            ),
            "trustedSampleCount": int(np.count_nonzero(sample_orientation_trusted)),
        },
        "distributions": {
            "sampleCountPerBin": _percentiles(sample_count),
            "orientationConfidence": _percentiles(confidence),
            "rawSampleToGenericMacroNormalResidualDegrees": _percentiles(
                generic_sample_residual
            ),
            "rawSampleToSelectedNormalResidualDegrees": _percentiles(
                sample_residual
            ),
            "physicalModeProfileCount": _percentiles(mode_profile_count),
            "physicalModeConfidence": _percentiles(mode_confidence),
            "physicalModeNormalResidualP90Degrees": _percentiles(mode_p90),
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
            "physicalGuidance": (
                "repeated air-papyrus-air profile normals form up to four axial "
                "modes per physical macro bin"
                if slab_manifest is not None
                else "not supplied; generic tensor fallback only"
            ),
            "guidedSampleRule": (
                "choose the nearest physical mode; defer a raw interface whose "
                "normal is more than the declared cap from every physical mode"
            ),
            "visualizedVector": "intersection of tangent plane with slice plane",
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
