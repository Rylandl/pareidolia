from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
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
from .export import rgb_png
from .isolated_slab import (
    IsolatedSlabSettings,
    _candidate_boundary_field,
    _crossing_position,
    _downsample_mean_zyx,
    _percentile_record,
    _trilinear,
)
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PAIRED_SURFACE_GROWTH_STEM,
)


ONE_SIDED_INTERFACE_SCHEMA = "pareidolia.one-sided-material-interface-bank"
ONE_SIDED_INTERFACE_VERSION = 1
ONE_SIDED_INTERFACE_STEM = "one-sided-interface-bank-v1"


@dataclass(frozen=True, slots=True)
class OneSidedInterfaceSettings:
    maximum_seed_position_residual_sampling_steps: float = 0.75
    maximum_seed_normal_degrees: float = 15.0
    seed_match_normal_scale_degrees: float = 15.0
    maximum_preview_seed_labels: int = 96

    def __post_init__(self) -> None:
        positive = (
            self.maximum_seed_position_residual_sampling_steps,
            self.maximum_seed_normal_degrees,
            self.seed_match_normal_scale_degrees,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("one-sided interface scales must be finite and positive")
        if not 0.0 < self.maximum_seed_normal_degrees < 90.0:
            raise ValueError("one-sided seed normal cap must lie in (0, 90)")
        if self.maximum_preview_seed_labels < 1:
            raise ValueError("one-sided preview label count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_growth(
    root: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    value = Path(root).resolve()
    growth_path = (
        value if value.is_file() else value / f"{PAIRED_SURFACE_GROWTH_STEM}.json"
    )
    growth_manifest = json.loads(growth_path.read_text())
    if (
        growth_manifest.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA
        or growth_manifest.get("state") != "complete"
    ):
        raise ValueError("one-sided interfaces require complete paired growth")
    growth_data_path = growth_path.parent / str(growth_manifest["data"]["path"])
    if sha256_file(growth_data_path) != growth_manifest["data"]["sha256"]:
        raise ValueError("paired-growth data hash differs from its manifest")
    with np.load(growth_data_path) as values:
        growth = {name: np.asarray(values[name]) for name in values.files}

    bank_path = Path(
        growth_manifest["identity"]["candidateBank"]["manifestPath"]
    )
    if (
        sha256_file(bank_path)
        != growth_manifest["identity"]["candidateBank"]["manifestSha256"]
    ):
        raise ValueError("paired-bank manifest changed after contextual growth")
    bank_manifest = json.loads(bank_path.read_text())
    if (
        bank_manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA
        or bank_manifest.get("state") != "complete"
    ):
        raise ValueError("paired growth references an incomplete candidate bank")
    bank_data_path = bank_path.parent / str(bank_manifest["data"]["path"])
    if sha256_file(bank_data_path) != bank_manifest["data"]["sha256"]:
        raise ValueError("paired bank data hash differs from its manifest")
    if (
        bank_manifest["data"]["sha256"]
        != growth_manifest["identity"]["candidateBank"]["dataSha256"]
    ):
        raise ValueError("paired growth and paired bank identities disagree")
    with np.load(bank_data_path) as values:
        bank = {name: np.asarray(values[name]) for name in values.files}
    return (
        growth_path,
        growth_manifest,
        growth,
        bank_path,
        bank_manifest,
        bank,
    )


def _isolated_settings(bank_manifest: Mapping[str, Any]) -> IsolatedSlabSettings:
    slab_path = Path(bank_manifest["identity"]["isolatedSlabs"]["manifestPath"])
    slab_manifest = json.loads(slab_path.read_text())
    return IsolatedSlabSettings(**slab_manifest["identity"]["settings"])


def extract_one_sided_interfaces(
    smoothed: np.ndarray,
    *,
    threshold: float,
    class_contrast: float,
    voxel_size_microns: float,
    isolated_settings: IsolatedSlabSettings,
) -> tuple[dict[str, int], dict[str, np.ndarray]]:
    """Extract signed air-to-material interfaces without requiring an exit."""

    boundary, gradient_x, gradient_y, gradient_z = _candidate_boundary_field(
        smoothed,
        threshold,
        class_contrast,
        isolated_settings,
    )
    candidate_zyx = np.column_stack(np.nonzero(boundary))
    candidate_xyz = candidate_zyx[:, ::-1].astype(np.float32)
    normal = np.column_stack(
        (gradient_x[boundary], gradient_y[boundary], gradient_z[boundary])
    ).astype(np.float32)
    gradient_length = np.linalg.norm(normal, axis=1)
    normal /= np.maximum(gradient_length[:, None], 1.0e-6)

    stride = isolated_settings.sampling_stride_voxels
    step_microns = voxel_size_microns * stride
    clearance_steps = max(
        2,
        int(
            math.ceil(
                isolated_settings.minimum_air_clearance_microns / step_microns
            )
        ),
    )
    minimum_thickness_steps = (
        isolated_settings.minimum_sheet_thickness_microns / step_microns
    )
    interior_steps = min(
        clearance_steps,
        max(2, int(math.floor(minimum_thickness_steps)) - 1),
    )
    profile_distances = np.arange(
        -clearance_steps,
        interior_steps + 2,
        dtype=np.float32,
    )
    zero_index = clearance_steps
    counts = {
        "boundaryCandidateCount": int(len(candidate_xyz)),
        "hasAirToMaterialCrossingCount": 0,
        "clearExteriorAirCount": 0,
        "materialInteriorCount": 0,
    }
    result: dict[str, list[np.ndarray]] = {
        name: []
        for name in (
            "position",
            "normal",
            "processing_key",
            "confidence",
            "air_margin",
            "air_sample_fraction",
            "material_margin",
            "gradient_class_fraction",
        )
    }
    air_offsets = np.arange(1, clearance_steps + 1, dtype=np.float32)
    material_offsets = np.arange(1, interior_steps + 1, dtype=np.float32)
    for low in range(
        0, len(candidate_xyz), isolated_settings.profile_batch_size
    ):
        point = candidate_xyz[low : low + isolated_settings.profile_batch_size]
        axis = normal[low : low + isolated_settings.profile_batch_size]
        profile = _trilinear(
            smoothed,
            point[:, None, :]
            + axis[:, None, :] * profile_distances[None, :, None],
            outside=255.0,
        )
        entry_pairs = (
            (profile[:, :zero_index] < threshold)
            & (profile[:, 1 : zero_index + 1] >= threshold)
        )
        has_entry = np.any(entry_pairs, axis=1)
        reverse_entry = np.argmax(entry_pairs[:, ::-1], axis=1)
        entry_index = zero_index - 1 - reverse_entry
        row = np.arange(len(point), dtype=np.int64)
        entry_distance = _crossing_position(
            profile[row, entry_index],
            profile[row, entry_index + 1],
            profile_distances[entry_index],
            threshold,
        )
        counts["hasAirToMaterialCrossingCount"] += int(
            np.count_nonzero(has_entry)
        )

        query_distance = np.concatenate(
            (
                entry_distance[:, None] - air_offsets[None, :],
                entry_distance[:, None] + material_offsets[None, :],
            ),
            axis=1,
        )
        query = _trilinear(
            smoothed,
            point[:, None, :] + axis[:, None, :] * query_distance[:, :, None],
            outside=255.0,
        )
        air = query[:, :clearance_steps]
        material = query[:, clearance_steps:]
        air_fraction = np.mean(air < threshold, axis=1)
        air_margin = (threshold - np.mean(air, axis=1)) / class_contrast
        material_margin = (np.mean(material, axis=1) - threshold) / class_contrast
        clear_air = (
            has_entry
            & (
                air_fraction
                >= isolated_settings.minimum_air_sample_fraction
            )
            & (
                air_margin
                >= isolated_settings.minimum_profile_margin_class_fraction
            )
        )
        material_inside = clear_air & (
            material_margin
            >= isolated_settings.minimum_profile_margin_class_fraction
        )
        counts["clearExteriorAirCount"] += int(np.count_nonzero(clear_air))
        counts["materialInteriorCount"] += int(
            np.count_nonzero(material_inside)
        )
        selected = np.flatnonzero(material_inside)
        if not len(selected):
            continue
        position = (
            point[selected]
            + axis[selected] * entry_distance[selected, None]
        )
        profile_margin = np.minimum(
            air_margin[selected], material_margin[selected]
        )
        confidence = np.clip(
            (
                profile_margin
                - isolated_settings.minimum_profile_margin_class_fraction
            )
            / (
                isolated_settings.full_confidence_profile_margin_class_fraction
                - isolated_settings.minimum_profile_margin_class_fraction
            ),
            0.0,
            1.0,
        )
        result["position"].append(position.astype(np.float32))
        result["normal"].append(axis[selected].astype(np.float32))
        result["processing_key"].append(point[selected].astype(np.int32))
        result["confidence"].append(confidence.astype(np.float32))
        result["air_margin"].append(air_margin[selected].astype(np.float32))
        result["air_sample_fraction"].append(
            air_fraction[selected].astype(np.float32)
        )
        result["material_margin"].append(
            material_margin[selected].astype(np.float32)
        )
        result["gradient_class_fraction"].append(
            (
                gradient_length[
                    low : low + isolated_settings.profile_batch_size
                ][selected]
                / class_contrast
            ).astype(np.float32)
        )
    arrays = {
        name: (
            np.concatenate(values)
            if values
            else np.empty(
                (0, 3)
                if name in {"position", "normal", "processing_key"}
                else (0,),
                dtype=(np.int32 if name == "processing_key" else np.float32),
            )
        )
        for name, values in result.items()
    }
    return counts, arrays


def match_signed_interface_endpoints(
    endpoint_position_world: np.ndarray,
    endpoint_normal: np.ndarray,
    interface_position_world: np.ndarray,
    interface_normal: np.ndarray,
    interface_processing_key: np.ndarray,
    *,
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: OneSidedInterfaceSettings,
) -> dict[str, np.ndarray]:
    """Match arbitrary signed physical endpoints to the dense interface bank."""

    half = 0.5 * (stride - 1)
    endpoint_sampling = (
        endpoint_position_world
        - source_origin_xyz[None, :]
        - processing_start_xyz[None, :]
        - half
    ) / stride
    endpoint_key = np.rint(endpoint_sampling).astype(np.int32)
    group_grid = np.full(
        processing_shape_sampling_xyz[::-1], -1, dtype=np.int32
    )
    group_grid[
        interface_processing_key[:, 2],
        interface_processing_key[:, 1],
        interface_processing_key[:, 0],
    ] = np.arange(len(interface_processing_key), dtype=np.int32)
    best_interface = np.full(len(endpoint_position_world), -1, dtype=np.int32)
    best_cost = np.full(len(endpoint_position_world), np.inf, dtype=np.float32)
    best_position = np.full(
        len(endpoint_position_world), np.inf, dtype=np.float32
    )
    best_normal = np.full(
        len(endpoint_position_world), np.inf, dtype=np.float32
    )
    shape = np.asarray(processing_shape_sampling_xyz, dtype=np.int32)
    for dz in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                query = endpoint_key + np.asarray((dx, dy, dz), dtype=np.int32)
                valid = np.all((query >= 0) & (query < shape[None, :]), axis=1)
                endpoint_index = np.flatnonzero(valid)
                candidate = group_grid[
                    query[valid, 2], query[valid, 1], query[valid, 0]
                ]
                exists = candidate >= 0
                endpoint_index = endpoint_index[exists]
                candidate = candidate[exists]
                if not len(candidate):
                    continue
                position_residual = np.linalg.norm(
                    interface_position_world[candidate]
                    - endpoint_position_world[endpoint_index],
                    axis=1,
                ) / stride
                normal_degrees = np.degrees(
                    np.arccos(
                        np.clip(
                            np.einsum(
                                "ij,ij->i",
                                interface_normal[candidate],
                                endpoint_normal[endpoint_index],
                            ),
                            -1.0,
                            1.0,
                        )
                    )
                )
                cost = (
                    position_residual
                    + normal_degrees / settings.seed_match_normal_scale_degrees
                )
                accepted = (
                    position_residual
                    <= settings.maximum_seed_position_residual_sampling_steps
                ) & (normal_degrees <= settings.maximum_seed_normal_degrees)
                improved = accepted & (cost < best_cost[endpoint_index])
                endpoint_index = endpoint_index[improved]
                candidate = candidate[improved]
                if not len(candidate):
                    continue
                best_interface[endpoint_index] = candidate
                best_cost[endpoint_index] = cost[improved]
                best_position[endpoint_index] = position_residual[improved]
                best_normal[endpoint_index] = normal_degrees[improved]
    return {
        "interfaceIndex": best_interface,
        "matchCost": best_cost,
        "positionResidualSamplingSteps": best_position,
        "normalResidualDegrees": best_normal,
    }


def match_paired_surface_boundaries(
    interface_position_world: np.ndarray,
    interface_normal: np.ndarray,
    interface_processing_key: np.ndarray,
    bank: Mapping[str, np.ndarray],
    growth: Mapping[str, np.ndarray],
    *,
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: OneSidedInterfaceSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Match both signed faces of selected paired slabs to interface samples."""

    selected_paired = np.flatnonzero(np.asarray(growth["selected"]) > 0)
    paired_label = np.asarray(growth["selectedLabel"], dtype=np.int32)
    paired_normal = np.asarray(bank["normalXYZ"], dtype=np.float32)
    endpoint_position = np.concatenate(
        (
            np.asarray(bank["boundaryLowerXYZ"])[selected_paired],
            np.asarray(bank["boundaryUpperXYZ"])[selected_paired],
        )
    ).astype(np.float32)
    endpoint_normal = np.concatenate(
        (paired_normal[selected_paired], -paired_normal[selected_paired])
    ).astype(np.float32)
    endpoint_paired_candidate = np.tile(selected_paired, 2).astype(np.int32)
    endpoint_surface_label = np.tile(
        paired_label[selected_paired], 2
    ).astype(np.int32)
    endpoint_side = np.repeat(
        np.asarray((0, 1), dtype=np.uint8), len(selected_paired)
    )

    endpoint_match = match_signed_interface_endpoints(
        endpoint_position,
        endpoint_normal,
        interface_position_world,
        interface_normal,
        interface_processing_key,
        processing_start_xyz=processing_start_xyz,
        source_origin_xyz=source_origin_xyz,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        stride=stride,
        settings=settings,
    )
    best_interface = endpoint_match["interfaceIndex"]
    best_cost = endpoint_match["matchCost"]
    best_position = endpoint_match["positionResidualSamplingSteps"]
    best_normal = endpoint_match["normalResidualDegrees"]

    seed_surface_label = np.full(len(interface_position_world), -1, dtype=np.int32)
    seed_paired_candidate = np.full(
        len(interface_position_world), -1, dtype=np.int32
    )
    seed_boundary_side = np.full(
        len(interface_position_world), 255, dtype=np.uint8
    )
    seed_match_cost = np.full(
        len(interface_position_world), np.inf, dtype=np.float32
    )
    seed_conflict = np.zeros(len(interface_position_world), dtype=np.uint8)
    matched_endpoint = np.flatnonzero(best_interface >= 0)
    order = np.lexsort(
        (
            matched_endpoint,
            best_cost[matched_endpoint],
            best_interface[matched_endpoint],
        )
    )
    sorted_endpoint = matched_endpoint[order]
    start = 0
    conflicting_interfaces = 0
    while start < len(sorted_endpoint):
        end = start + 1
        interface = int(best_interface[sorted_endpoint[start]])
        while (
            end < len(sorted_endpoint)
            and int(best_interface[sorted_endpoint[end]]) == interface
        ):
            end += 1
        endpoint_group = sorted_endpoint[start:end]
        labels = np.unique(endpoint_surface_label[endpoint_group])
        if len(labels) == 1:
            chosen = int(endpoint_group[0])
            seed_surface_label[interface] = int(labels[0])
            seed_paired_candidate[interface] = int(
                endpoint_paired_candidate[chosen]
            )
            seed_boundary_side[interface] = endpoint_side[chosen]
            seed_match_cost[interface] = best_cost[chosen]
        else:
            seed_conflict[interface] = 1
            conflicting_interfaces += 1
        start = end
    side_records = []
    for side, name in ((0, "canonicalLower"), (1, "canonicalUpper")):
        side_mask = endpoint_side == side
        side_matched = side_mask & (best_interface >= 0)
        side_index = np.flatnonzero(side_matched)
        side_records.append(
            {
                "side": name,
                "endpointCount": int(np.count_nonzero(side_mask)),
                "matchedEndpointCount": int(len(side_index)),
                "matchedEndpointFraction": round(
                    len(side_index) / max(np.count_nonzero(side_mask), 1), 6
                ),
                "positionResidualSamplingSteps": _percentile_record(
                    best_position[side_index]
                ),
                "normalResidualDegrees": _percentile_record(
                    best_normal[side_index]
                ),
            }
        )
    return {
        "seedSurfaceLabel": seed_surface_label,
        "seedPairedCandidateIndex": seed_paired_candidate,
        "seedBoundarySide": seed_boundary_side,
        "seedMatchCost": seed_match_cost,
        "seedConflict": seed_conflict,
    }, {
        "selectedPairedCandidateCount": int(len(selected_paired)),
        "pairedBoundaryEndpointCount": int(len(endpoint_position)),
        "matchedPairedBoundaryEndpointCount": int(len(matched_endpoint)),
        "matchedPairedBoundaryEndpointFraction": round(
            len(matched_endpoint) / max(len(endpoint_position), 1), 6
        ),
        "uniqueSeedInterfaceCount": int(
            np.count_nonzero(seed_surface_label >= 0)
        ),
        "conflictingSeedInterfaceCount": conflicting_interfaces,
        "boundarySides": side_records,
        "matchedPositionResidualSamplingSteps": _percentile_record(
            best_position[matched_endpoint]
        ),
        "matchedNormalResidualDegrees": _percentile_record(
            best_normal[matched_endpoint]
        ),
        "matchedCost": _percentile_record(best_cost[matched_endpoint]),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def write_one_sided_projection(
    arrays: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    path: str | Path,
    *,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    point = np.asarray(arrays["positionXYZ"])
    seeded = np.asarray(arrays["seedSurfaceLabel"]) >= 0
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        normalized = (
            point[:, list(axes)] - world_start_xyz[None, list(axes)]
        ) / width[None, list(axes)]
        x = np.rint(
            offset + margin + normalized[:, 0] * (panel_size - 2 * margin)
        ).astype(np.int32)
        y = np.rint(
            panel_size - margin - normalized[:, 1] * (panel_size - 2 * margin)
        ).astype(np.int32)
        valid = (
            (x >= offset)
            & (x < offset + panel_size)
            & (y >= 0)
            & (y < panel_size)
        )
        canvas[y[valid], x[valid]] = (54, 101, 112)
        seed_valid = valid & seeded
        canvas[y[seed_valid], x[seed_valid]] = (255, 177, 72)
        border = (64, 72, 84)
        canvas[margin, offset + margin : offset + panel_size - margin] = border
        canvas[
            panel_size - margin,
            offset + margin : offset + panel_size - margin,
        ] = border
        canvas[margin : panel_size - margin, offset + margin] = border
        canvas[
            margin : panel_size - margin,
            offset + panel_size - margin,
        ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def run_one_sided_interface_bank(
    growth_root: str | Path,
    output_root: str | Path,
    *,
    settings: OneSidedInterfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or OneSidedInterfaceSettings()
    (
        growth_path,
        growth_manifest,
        growth,
        bank_path,
        bank_manifest,
        bank,
    ) = _load_growth(growth_root)
    isolated_settings = _isolated_settings(bank_manifest)
    source = VolumeSource.open(
        bank_manifest["source"]["path"],
        bank_manifest["source"]["metadataPath"],
    )
    processing_record = bank_manifest["geometry"]["processingVoxelBounds"]
    processing = VoxelBounds(
        tuple(processing_record["startXYZ"]),
        tuple(processing_record["stopXYZExclusive"]),
    )
    stride = isolated_settings.sampling_stride_voxels
    processing_shape_sampling_xyz = tuple(
        value // stride for value in processing.shape_xyz
    )
    identity: dict[str, Any] = {
        "schema": ONE_SIDED_INTERFACE_SCHEMA,
        "version": ONE_SIDED_INTERFACE_VERSION,
        "pairedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "pairedBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{ONE_SIDED_INTERFACE_STEM}.json"
    data_path = output / f"{ONE_SIDED_INTERFACE_STEM}.npz"
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
    stage = time.monotonic()
    sampled = _downsample_mean_zyx(
        source.memmap()[processing.slices_zyx], stride
    )
    smoothed = gaussian_blur_3d(
        sampled, isolated_settings.smoothing_sigma_voxels / stride
    )
    prepared = time.monotonic()
    counts, raw = extract_one_sided_interfaces(
        smoothed,
        threshold=float(bank_manifest["calibration"]["materialThresholdRaw"]),
        class_contrast=float(bank_manifest["calibration"]["classContrastRaw"]),
        voxel_size_microns=source.voxel_size_microns,
        isolated_settings=isolated_settings,
    )
    extracted = time.monotonic()

    half = 0.5 * (stride - 1)
    processing_start = np.asarray(processing.start_xyz, dtype=np.float32)
    source_origin = np.asarray(source.origin_xyz, dtype=np.float32)
    position_world = (
        np.asarray(raw["position"], dtype=np.float32) * stride
        + processing_start[None, :]
        + source_origin[None, :]
        + half
    )
    owned_record = bank_manifest["geometry"]["ownedWorldBounds"]
    owned_start = np.asarray(owned_record["startXYZ"], dtype=np.float32)
    owned_stop = np.asarray(owned_record["stopXYZExclusive"], dtype=np.float32)
    owned = np.all(
        (position_world >= owned_start[None, :])
        & (position_world < owned_stop[None, :]),
        axis=1,
    )
    selected = np.flatnonzero(owned)
    counts["ownedInterfaceCount"] = int(len(selected))
    position_world = position_world[selected]
    interface_normal = np.asarray(raw["normal"])[selected]
    interface_key = np.asarray(raw["processing_key"])[selected]
    seed, match_stats = match_paired_surface_boundaries(
        position_world,
        interface_normal,
        interface_key,
        bank,
        growth,
        processing_start_xyz=processing_start,
        source_origin_xyz=source_origin,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        stride=stride,
        settings=resolved,
    )
    matched = time.monotonic()
    arrays = {
        "positionXYZ": position_world.astype(np.float32),
        "signedNormalXYZ": interface_normal.astype(np.float32),
        "processingKeyXYZ": interface_key.astype(np.int32),
        "localEvidenceScore": np.asarray(raw["confidence"])[selected].astype(
            np.float32
        ),
        "airMarginClassFraction": np.asarray(raw["air_margin"])[selected].astype(
            np.float32
        ),
        "airSampleFraction": np.asarray(raw["air_sample_fraction"])[
            selected
        ].astype(np.float32),
        "materialMarginClassFraction": np.asarray(raw["material_margin"])[
            selected
        ].astype(np.float32),
        "gradientClassFraction": np.asarray(raw["gradient_class_fraction"])[
            selected
        ].astype(np.float32),
        **seed,
    }
    _write_npz(data_path, arrays)
    projection = write_one_sided_projection(
        arrays,
        owned_start,
        owned_stop,
        output / "one-sided-interface-projection.png",
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": ONE_SIDED_INTERFACE_SCHEMA,
        "version": ONE_SIDED_INTERFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": bank_manifest["geometry"],
        "calibration": bank_manifest["calibration"],
        "counts": counts,
        "matching": match_stats,
        "distributions": {
            "localEvidenceScore": _percentile_record(
                arrays["localEvidenceScore"]
            ),
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
            "ctPreparation": round(prepared - stage, 6),
            "interfaceExtraction": round(extracted - prepared, 6),
            "ownershipAndSeedMatching": round(matched - extracted, 6),
            "writingAndArtifact": round(finished - matched, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"projection": projection.name},
        "method": {
            "interfaceOrientation": "signed air-to-material CT gradient",
            "requiresOppositeFace": False,
            "changesPairedSurfaceGeometry": False,
            "seedSemantics": (
                "the two exact signed faces of selected paired surfaces; "
                "conflicting surface identities are never assigned"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
