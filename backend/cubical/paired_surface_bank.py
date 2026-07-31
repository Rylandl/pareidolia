from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from backend.rectify import gaussian_blur_3d

from .contracts import VolumeSource, VoxelBounds, atomic_json, canonical_json_hash, sha256_file
from .isolated_slab import (
    ISOLATED_SLAB_SCHEMA,
    IsolatedSlabSettings,
    _downsample_mean_zyx,
    _percentile_record,
    detect_isolated_slab_pairs,
)


PAIRED_SURFACE_BANK_SCHEMA = "pareidolia.paired-surface-candidate-bank"
PAIRED_SURFACE_BANK_VERSION = 1
PAIRED_SURFACE_BANK_STEM = "paired-surface-bank-v1"


@dataclass(frozen=True, slots=True)
class PairedSurfaceBankSettings:
    """Controls for preserving distinct paired-interface hypotheses.

    Candidate generation retains every profile satisfying only the immutable
    physical thickness and two-crossing requirements.  These settings perform
    reciprocal duplicate suppression; they never decide which candidate is a
    sheet or connect candidates across spatial keys.
    """

    maximum_hypotheses_per_spatial_key: int = 4
    duplicate_normal_degrees: float = 15.0
    duplicate_midpoint_distance_sampling_steps: float = 1.25
    duplicate_boundary_distance_sampling_steps: float = 1.5
    duplicate_thickness_tolerance_sampling_steps: float = 1.5
    evidence_air_margin_scale: float = 0.18
    evidence_material_margin_scale: float = 0.18
    evidence_opposing_cosine_scale: float = 0.12
    evidence_air_weight: float = 0.3
    evidence_material_weight: float = 0.25
    evidence_opposing_weight: float = 0.3
    evidence_clearance_weight: float = 0.15

    def __post_init__(self) -> None:
        if self.maximum_hypotheses_per_spatial_key < 1:
            raise ValueError("candidate bank must retain at least one hypothesis per key")
        positive = (
            self.duplicate_normal_degrees,
            self.duplicate_midpoint_distance_sampling_steps,
            self.duplicate_boundary_distance_sampling_steps,
            self.duplicate_thickness_tolerance_sampling_steps,
            self.evidence_air_margin_scale,
            self.evidence_material_margin_scale,
            self.evidence_opposing_cosine_scale,
            self.evidence_air_weight,
            self.evidence_material_weight,
            self.evidence_opposing_weight,
            self.evidence_clearance_weight,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("candidate-bank scales and weights must be finite and positive")
        if not 0.0 < self.duplicate_normal_degrees < 90.0:
            raise ValueError("duplicate normal angle must lie in (0, 90) degrees")
        weight_sum = (
            self.evidence_air_weight
            + self.evidence_material_weight
            + self.evidence_opposing_weight
            + self.evidence_clearance_weight
        )
        if abs(weight_sum - 1.0) > 1.0e-6:
            raise ValueError("candidate-bank evidence weights must sum to one")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_isolated_slabs(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = value if value.is_file() else value / "isolated-slabs-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != ISOLATED_SLAB_SCHEMA or manifest.get("state") != "complete":
        raise ValueError("paired-surface bank requires complete isolated slabs")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("isolated-slab data hash differs from its manifest")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return manifest_path, manifest, arrays


def _canonical_profile_geometry(
    midpoint_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    boundary_first_xyz: np.ndarray,
    boundary_second_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    normal = np.asarray(normal_xyz, dtype=np.float32).copy()
    first = np.asarray(boundary_first_xyz, dtype=np.float32).copy()
    second = np.asarray(boundary_second_xyz, dtype=np.float32).copy()
    dominant = np.argmax(np.abs(normal), axis=1)
    sign = np.where(normal[np.arange(len(normal)), dominant] < 0.0, -1.0, 1.0)
    flip = sign < 0.0
    normal *= sign[:, None]
    lower = first.copy()
    upper = second.copy()
    lower[flip] = second[flip]
    upper[flip] = first[flip]
    return (
        np.asarray(midpoint_xyz, dtype=np.float32),
        normal,
        lower,
        upper,
    )


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(np.asarray(value, dtype=np.float64), -30.0, 30.0)
    return (1.0 / (1.0 + np.exp(-clipped))).astype(np.float32)


def candidate_local_evidence_score(
    air_margin: np.ndarray,
    material_margin: np.ndarray,
    opposing_cosine: np.ndarray,
    air_sample_fraction: np.ndarray,
    *,
    isolated_settings: IsolatedSlabSettings,
    settings: PairedSurfaceBankSettings,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    air = _sigmoid(
        (air_margin - isolated_settings.minimum_profile_margin_class_fraction)
        / settings.evidence_air_margin_scale
    )
    material = _sigmoid(
        (material_margin - isolated_settings.minimum_profile_margin_class_fraction)
        / settings.evidence_material_margin_scale
    )
    opposing_limit = math.cos(
        math.radians(isolated_settings.maximum_opposing_normal_degrees)
    )
    opposing = _sigmoid(
        (opposing_cosine - opposing_limit)
        / settings.evidence_opposing_cosine_scale
    )
    clearance = np.clip(np.asarray(air_sample_fraction, dtype=np.float32), 0.0, 1.0)
    score = (
        settings.evidence_air_weight * air
        + settings.evidence_material_weight * material
        + settings.evidence_opposing_weight * opposing
        + settings.evidence_clearance_weight * clearance
    )
    return score.astype(np.float32), {
        "airEvidence": air,
        "materialEvidence": material,
        "opposingEvidence": opposing,
        "clearanceEvidence": clearance,
    }


def _match_isolated_samples(
    raw_midpoint_world: np.ndarray,
    raw_normal: np.ndarray,
    raw_thickness_voxels: np.ndarray,
    raw_key: np.ndarray,
    slab: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    stride: int,
) -> tuple[np.ndarray, dict[str, float | int]]:
    slab_midpoint = np.asarray(slab["midpointXYZ"], dtype=np.float64)
    half = 0.5 * (stride - 1)
    slab_sampling = (
        slab_midpoint
        - source_origin_xyz[None, :]
        - processing_start_xyz[None, :]
        - half
    ) / stride
    slab_key = np.rint(slab_sampling).astype(np.int32)
    shape_zyx = processing_shape_sampling_xyz[::-1]
    raw_flat = np.ravel_multi_index(
        (raw_key[:, 2], raw_key[:, 1], raw_key[:, 0]), shape_zyx
    )
    slab_flat = np.ravel_multi_index(
        (slab_key[:, 2], slab_key[:, 1], slab_key[:, 0]), shape_zyx
    )
    order = np.argsort(raw_flat, kind="stable")
    sorted_flat = raw_flat[order]
    matched = np.full(len(raw_midpoint_world), -1, dtype=np.int32)
    match_cost = np.full(len(slab_midpoint), np.inf, dtype=np.float64)

    def cost_for(candidate: np.ndarray, slab_index: int) -> np.ndarray:
        midpoint_distance = np.linalg.norm(
            raw_midpoint_world[candidate] - slab_midpoint[slab_index], axis=1
        ) / stride
        normal_dot = np.abs(
            np.einsum(
                "ij,j->i",
                raw_normal[candidate],
                np.asarray(slab["normalXYZ"][slab_index], dtype=np.float64),
            )
        )
        normal_residual = np.degrees(
            np.arccos(np.clip(normal_dot, 0.0, 1.0))
        ) / 15.0
        thickness_residual = np.abs(
            raw_thickness_voxels[candidate]
            - float(slab["thicknessVoxels"][slab_index])
        ) / stride
        return midpoint_distance + normal_residual + thickness_residual

    for slab_index, flat in enumerate(slab_flat):
        low = int(np.searchsorted(sorted_flat, flat, side="left"))
        high = int(np.searchsorted(sorted_flat, flat, side="right"))
        candidate_groups = [order[low:high]] if low < high else []
        # Float32 world-coordinate round trips can move a midpoint exactly onto
        # the neighboring source-aligned key.  An unrelated profile can occupy
        # the nominal key, so a geometrically non-exact hit must also pay for
        # this deterministic 26-neighbor fallback.  Exact matches avoid it.
        search_neighbors = not candidate_groups
        if candidate_groups:
            exact_candidate = candidate_groups[0]
            search_neighbors = bool(
                np.min(cost_for(exact_candidate, slab_index), initial=np.inf)
                > 0.05
            )
        if search_neighbors:
            for dz in (-1, 0, 1):
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        if (dx, dy, dz) == (0, 0, 0):
                            continue
                        neighbor = slab_key[slab_index] + np.asarray((dx, dy, dz))
                        if np.any(neighbor < 0) or np.any(
                            neighbor >= np.asarray(processing_shape_sampling_xyz)
                        ):
                            continue
                        neighbor_flat = np.ravel_multi_index(
                            (neighbor[2], neighbor[1], neighbor[0]), shape_zyx
                        )
                        low = int(
                            np.searchsorted(sorted_flat, neighbor_flat, side="left")
                        )
                        high = int(
                            np.searchsorted(sorted_flat, neighbor_flat, side="right")
                        )
                        if low < high:
                            candidate_groups.append(order[low:high])
        if not candidate_groups:
            continue
        candidate = np.concatenate(candidate_groups)
        cost = cost_for(candidate, slab_index)
        best = int(np.argmin(cost))
        selected = int(candidate[best])
        best_cost = float(cost[best])
        if best_cost > 4.0:
            continue
        if matched[selected] < 0 or best_cost < match_cost[matched[selected]]:
            matched[selected] = slab_index
            match_cost[slab_index] = best_cost
    matched_slab = matched[matched >= 0]
    return matched, {
        "isolatedSampleCount": int(len(slab_midpoint)),
        "matchedIsolatedSampleCount": int(len(np.unique(matched_slab))),
        "maximumMatchCost": (
            round(float(np.max(match_cost[np.isfinite(match_cost)])), 6)
            if np.any(np.isfinite(match_cost))
            else None
        ),
    }


def suppress_reciprocal_profile_duplicates(
    midpoint_world: np.ndarray,
    normal_xyz: np.ndarray,
    boundary_lower_world: np.ndarray,
    boundary_upper_world: np.ndarray,
    thickness_voxels: np.ndarray,
    spatial_key_xyz: np.ndarray,
    local_evidence_score: np.ndarray,
    conservative: np.ndarray,
    isolated_sample_index: np.ndarray,
    *,
    stride: int,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: PairedSurfaceBankSettings,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    shape_zyx = processing_shape_sampling_xyz[::-1]
    flat = np.ravel_multi_index(
        (
            spatial_key_xyz[:, 2],
            spatial_key_xyz[:, 1],
            spatial_key_xyz[:, 0],
        ),
        shape_zyx,
    )
    # Existing isolated-slab samples are immutable anchors and therefore win
    # their key before any merely local score.  Conservative unselected pairs
    # then precede relaxed physical candidates.
    anchor = isolated_sample_index >= 0
    order = np.lexsort(
        (
            np.arange(len(flat), dtype=np.int64),
            -local_evidence_score,
            -conservative.astype(np.int8),
            -anchor.astype(np.int8),
            flat,
        )
    )
    normal_limit = math.cos(math.radians(settings.duplicate_normal_degrees))
    kept: list[int] = []
    alternative_index: list[int] = []
    duplicate_count = 0
    capped_count = 0
    start = 0
    while start < len(order):
        end = start + 1
        current_flat = flat[order[start]]
        while end < len(order) and flat[order[end]] == current_flat:
            end += 1
        selected: list[int] = []
        for candidate in order[start:end]:
            duplicate = False
            for prior in selected:
                normal_dot = abs(float(np.dot(normal_xyz[candidate], normal_xyz[prior])))
                midpoint_distance = float(
                    np.linalg.norm(midpoint_world[candidate] - midpoint_world[prior])
                    / stride
                )
                boundary_distance = 0.5 * float(
                    np.linalg.norm(
                        boundary_lower_world[candidate] - boundary_lower_world[prior]
                    )
                    + np.linalg.norm(
                        boundary_upper_world[candidate] - boundary_upper_world[prior]
                    )
                ) / stride
                thickness_difference = (
                    abs(float(thickness_voxels[candidate] - thickness_voxels[prior]))
                    / stride
                )
                if (
                    normal_dot >= normal_limit
                    and midpoint_distance
                    <= settings.duplicate_midpoint_distance_sampling_steps
                    and boundary_distance
                    <= settings.duplicate_boundary_distance_sampling_steps
                    and thickness_difference
                    <= settings.duplicate_thickness_tolerance_sampling_steps
                ):
                    duplicate = True
                    break
            if duplicate:
                duplicate_count += 1
                continue
            if len(selected) >= settings.maximum_hypotheses_per_spatial_key:
                capped_count += 1
                continue
            selected.append(int(candidate))
        kept.extend(selected)
        alternative_index.extend(range(len(selected)))
        start = end
    kept_array = np.asarray(kept, dtype=np.int32)
    _, retained_per_key = np.unique(flat[kept_array], return_counts=True)
    return kept_array, np.asarray(alternative_index, dtype=np.uint8), {
        "inputProfileCount": int(len(flat)),
        "retainedProfileCount": int(len(kept_array)),
        "reciprocalDuplicateCount": duplicate_count,
        "hypothesisCapDiscardCount": capped_count,
        "spatialKeyCount": int(len(np.unique(flat))),
        "keysWithMultipleRetainedHypotheses": int(
            np.count_nonzero(retained_per_key > 1)
        ),
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_paired_surface_bank(
    slab_root: str | Path,
    output_root: str | Path,
    *,
    settings: PairedSurfaceBankSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PairedSurfaceBankSettings()
    slab_manifest_path, slab_manifest, slab = _load_isolated_slabs(slab_root)
    isolated_settings = IsolatedSlabSettings(
        **slab_manifest["identity"]["settings"]
    )
    source = VolumeSource.open(
        slab_manifest["source"]["path"], slab_manifest["source"]["metadataPath"]
    )
    processing_record = slab_manifest["geometry"]["processingVoxelBounds"]
    processing = VoxelBounds(
        tuple(processing_record["startXYZ"]),
        tuple(processing_record["stopXYZExclusive"]),
    )
    stride = isolated_settings.sampling_stride_voxels
    processing_shape_sampling_xyz = tuple(
        value // stride for value in processing.shape_xyz
    )
    identity: dict[str, Any] = {
        "schema": PAIRED_SURFACE_BANK_SCHEMA,
        "version": PAIRED_SURFACE_BANK_VERSION,
        "isolatedSlabs": {
            "manifestPath": str(slab_manifest_path),
            "manifestSha256": sha256_file(slab_manifest_path),
            "dataSha256": slab_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": {
            "paired_surface_bank.py": sha256_file(Path(__file__)),
            "isolated_slab.py": sha256_file(Path(__file__).with_name("isolated_slab.py")),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PAIRED_SURFACE_BANK_STEM}.json"
    data_path = output / f"{PAIRED_SURFACE_BANK_STEM}.npz"
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
    sampled = _downsample_mean_zyx(
        source.memmap()[processing.slices_zyx], stride
    )
    smoothed = gaussian_blur_3d(
        sampled, isolated_settings.smoothing_sigma_voxels / stride
    )
    timing["ctPreparation"] = time.monotonic() - stage
    stage = time.monotonic()
    profile_counts, raw = detect_isolated_slab_pairs(
        smoothed,
        threshold=float(slab_manifest["calibration"]["materialThresholdRaw"]),
        class_contrast=float(slab_manifest["calibration"]["classContrastRaw"]),
        voxel_size_microns=source.voxel_size_microns,
        settings=isolated_settings,
        retain_physical_profiles=True,
    )
    timing["physicalProfileExtraction"] = time.monotonic() - stage

    stage = time.monotonic()
    midpoint_sampling = np.asarray(raw["midpoint"], dtype=np.float32)
    half = 0.5 * (stride - 1)
    source_shift = np.asarray(source.origin_xyz, dtype=np.float32)
    processing_start = np.asarray(processing.start_xyz, dtype=np.float32)

    def world(points: np.ndarray) -> np.ndarray:
        return (
            np.asarray(points, dtype=np.float32) * stride
            + processing_start[None, :]
            + source_shift[None, :]
            + half
        )

    midpoint_world = world(midpoint_sampling)
    first_world = world(raw["boundary_first"])
    second_world = world(raw["boundary_second"])
    midpoint_world, normal, lower_world, upper_world = _canonical_profile_geometry(
        midpoint_world, raw["normal"], first_world, second_world
    )
    owned_world = slab_manifest["geometry"]["ownedWorldBounds"]
    owned_start = np.asarray(owned_world["startXYZ"], dtype=np.float32)
    owned_stop = np.asarray(owned_world["stopXYZExclusive"], dtype=np.float32)
    owned_mask = np.all(
        (midpoint_world >= owned_start[None, :])
        & (midpoint_world < owned_stop[None, :]),
        axis=1,
    )
    selected = np.flatnonzero(owned_mask)
    midpoint_sampling = midpoint_sampling[selected]
    midpoint_world = midpoint_world[selected]
    normal = normal[selected]
    lower_world = lower_world[selected]
    upper_world = upper_world[selected]
    thickness_voxels = raw["thickness_steps"][selected] * stride
    spatial_key = np.rint(midpoint_sampling).astype(np.int32)
    evidence_score, evidence_parts = candidate_local_evidence_score(
        raw["air_margin"][selected],
        raw["material_margin"][selected],
        raw["opposing_cosine"][selected],
        raw["air_sample_fraction"][selected],
        isolated_settings=isolated_settings,
        settings=resolved,
    )
    isolated_sample_index, match_stats = _match_isolated_samples(
        midpoint_world,
        normal,
        thickness_voxels,
        spatial_key,
        slab,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        processing_start_xyz=processing_start.astype(np.float64),
        source_origin_xyz=source_shift.astype(np.float64),
        stride=stride,
    )
    retained, alternative_index, suppression_stats = suppress_reciprocal_profile_duplicates(
        midpoint_world,
        normal,
        lower_world,
        upper_world,
        thickness_voxels,
        spatial_key,
        evidence_score,
        raw["conservative"][selected],
        isolated_sample_index,
        stride=stride,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        settings=resolved,
    )
    timing["ownershipMatchingAndSuppression"] = time.monotonic() - stage

    isolated_index = isolated_sample_index[retained]
    seed_component = np.full(len(retained), -1, dtype=np.int32)
    seed_confidence = np.zeros(len(retained), dtype=np.float32)
    matched = isolated_index >= 0
    seed_component[matched] = slab["componentId"][isolated_index[matched]]
    seed_confidence[matched] = slab["confidence"][isolated_index[matched]]
    arrays = {
        "midpointXYZ": midpoint_world[retained].astype(np.float32),
        "normalXYZ": normal[retained].astype(np.float32),
        "boundaryLowerXYZ": lower_world[retained].astype(np.float32),
        "boundaryUpperXYZ": upper_world[retained].astype(np.float32),
        "thicknessVoxels": thickness_voxels[retained].astype(np.float32),
        "spatialKeyXYZ": spatial_key[retained].astype(np.int32),
        "alternativeIndex": alternative_index,
        "localEvidenceScore": evidence_score[retained].astype(np.float32),
        "airMarginClassFraction": raw["air_margin"][selected][retained].astype(np.float32),
        "airSampleFraction": raw["air_sample_fraction"][selected][retained].astype(np.float32),
        "materialMarginClassFraction": raw["material_margin"][selected][
            retained
        ].astype(np.float32),
        "opposingNormalCosine": raw["opposing_cosine"][selected][retained].astype(np.float32),
        "isolatedConservative": raw["conservative"][selected][retained].astype(np.uint8),
        "isolatedSampleIndex": isolated_index.astype(np.int32),
        "seedComponentId": seed_component,
        "seedConfidence": seed_confidence,
        "sourcePhysicalProfileIndex": selected[retained].astype(np.int32),
        **{
            name: values[retained].astype(np.float32)
            for name, values in evidence_parts.items()
        },
    }
    _write_npz(data_path, arrays)
    timing["total"] = time.monotonic() - started
    key_count = suppression_stats["spatialKeyCount"]
    payload: dict[str, Any] = {
        "schema": PAIRED_SURFACE_BANK_SCHEMA,
        "version": PAIRED_SURFACE_BANK_VERSION,
        "state": "complete",
        "identity": identity,
        "source": slab_manifest["source"],
        "geometry": slab_manifest["geometry"],
        "calibration": slab_manifest["calibration"],
        "counts": {
            **profile_counts,
            "ownedPhysicalProfileCountBeforeSuppression": int(len(selected)),
            "retainedCandidateCount": int(len(retained)),
            "retainedConservativeCandidateCount": int(
                np.count_nonzero(arrays["isolatedConservative"])
            ),
            "retainedSeedCandidateCount": int(
                np.count_nonzero(arrays["seedComponentId"] >= 0)
            ),
            "spatialKeyCount": key_count,
        },
        "matching": match_stats,
        "suppression": suppression_stats,
        "distributions": {
            "hypothesesPerSpatialKey": _percentile_record(
                np.unique(
                    np.ravel_multi_index(
                        (
                            arrays["spatialKeyXYZ"][:, 2],
                            arrays["spatialKeyXYZ"][:, 1],
                            arrays["spatialKeyXYZ"][:, 0],
                        ),
                        processing_shape_sampling_xyz[::-1],
                    ),
                    return_counts=True,
                )[1]
            ),
            "localEvidenceScore": _percentile_record(arrays["localEvidenceScore"]),
            "thicknessVoxels": _percentile_record(arrays["thicknessVoxels"]),
            "airMarginClassFraction": _percentile_record(
                arrays["airMarginClassFraction"]
            ),
            "materialMarginClassFraction": _percentile_record(
                arrays["materialMarginClassFraction"]
            ),
            "opposingNormalAngleDegrees": _percentile_record(
                np.degrees(
                    np.arccos(
                        np.clip(arrays["opposingNormalCosine"], -1.0, 1.0)
                    )
                )
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
        "method": {
            "changesGeometry": False,
            "selectsSheetIdentity": False,
            "candidateMinimum": "two CT crossings plus physical thickness only",
            "mutualExclusionUnit": "source-aligned sampling-lattice spatial key",
            "ambiguityPolicy": (
                "retain distinct boundary-pair hypotheses up to the declared per-key cap"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
