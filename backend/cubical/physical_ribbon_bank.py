from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .one_sided_interface import ONE_SIDED_INTERFACE_SCHEMA


PHYSICAL_RIBBON_BANK_SCHEMA = "pareidolia.physical-ribbon-bank"
PHYSICAL_RIBBON_BANK_VERSION = 1
PHYSICAL_RIBBON_BANK_STEM = "physical-ribbon-bank-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonBankSettings:
    """Physical priors used to propose, but never own, papyrus ribbons."""

    minimum_sheet_thickness_microns: float = 80.0
    maximum_sheet_thickness_microns: float = 400.0
    maximum_opposing_normal_degrees: float = 45.0
    maximum_inward_ray_degrees: float = 45.0
    ray_search_radius_sampling_steps: int = 2
    batch_interface_count: int = 4096

    def __post_init__(self) -> None:
        if not (
            math.isfinite(self.minimum_sheet_thickness_microns)
            and math.isfinite(self.maximum_sheet_thickness_microns)
            and 0.0 < self.minimum_sheet_thickness_microns
            < self.maximum_sheet_thickness_microns
        ):
            raise ValueError("sheet thickness bounds must be finite and ordered")
        for value in (
            self.maximum_opposing_normal_degrees,
            self.maximum_inward_ray_degrees,
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise ValueError("angular caps must lie in (0, 90) degrees")
        if self.ray_search_radius_sampling_steps < 0:
            raise ValueError("ray search radius must be nonnegative")
        if self.batch_interface_count < 1:
            raise ValueError("batch interface count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": round(float(np.min(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "p99": round(float(np.percentile(finite, 99)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def _ray_offsets(radius: int) -> np.ndarray:
    values = [
        (x, y, z)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
        if x * x + y * y + z * z <= radius * radius
    ]
    return np.asarray(
        sorted(values, key=lambda value: (sum(v * v for v in value), value)),
        dtype=np.int16,
    )


def _deduplicate_directed(
    source: np.ndarray,
    target: np.ndarray,
    score: np.ndarray,
    fields: Mapping[str, np.ndarray],
    interface_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    key = source.astype(np.int64) * interface_count + target
    order = np.lexsort((-score, key))
    ordered_key = key[order]
    keep = np.ones(len(order), dtype=bool)
    keep[1:] = ordered_key[1:] != ordered_key[:-1]
    chosen = order[keep]
    chosen_key = key[chosen]
    final_order = np.argsort(chosen_key)
    chosen = chosen[final_order]
    return (
        source[chosen],
        target[chosen],
        score[chosen],
        {name: np.asarray(value)[chosen] for name, value in fields.items()},
    )


def build_physical_ribbon_bank(
    interfaces: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    processing_world_start_xyz: np.ndarray,
    sampling_stride_voxels: int,
    voxel_size_microns: float,
    settings: PhysicalRibbonBankSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Pair dense signed CT interfaces without using propagated identities."""

    position = np.asarray(interfaces["positionXYZ"], dtype=np.float32)
    normal = np.asarray(interfaces["signedNormalXYZ"], dtype=np.float32)
    key = np.asarray(interfaces["processingKeyXYZ"], dtype=np.int32)
    evidence = np.asarray(interfaces["localEvidenceScore"], dtype=np.float32)
    interface_count = len(position)
    shape = tuple(int(value) for value in processing_shape_sampling_xyz)
    flat_size = int(np.prod(shape))
    lattice = np.full(flat_size, -1, dtype=np.int32)
    flat_key = np.ravel_multi_index(key.T, shape)
    if len(np.unique(flat_key)) != interface_count:
        raise ValueError("one-sided interface lattice contains duplicate keys")
    lattice[flat_key] = np.arange(interface_count, dtype=np.int32)

    stride = int(sampling_stride_voxels)
    half = 0.5 * (stride - 1)
    sampling_position = (
        position
        - np.asarray(processing_world_start_xyz, dtype=np.float32)[None, :]
        - half
    ) / stride
    minimum_voxels = settings.minimum_sheet_thickness_microns / voxel_size_microns
    maximum_voxels = settings.maximum_sheet_thickness_microns / voxel_size_microns
    radius = settings.ray_search_radius_sampling_steps
    minimum_step = max(1, int(math.floor(minimum_voxels / stride)) - radius)
    maximum_step = int(math.ceil(maximum_voxels / stride)) + radius
    ray_steps = np.arange(minimum_step, maximum_step + 1, dtype=np.float32)
    offsets = _ray_offsets(radius)
    queries_per_source = len(ray_steps) * len(offsets)
    facing_cosine = math.cos(math.radians(settings.maximum_inward_ray_degrees))
    opposing_cosine = math.cos(
        math.radians(settings.maximum_opposing_normal_degrees)
    )

    accumulated: dict[str, list[np.ndarray]] = {
        "source": [],
        "target": [],
        "score": [],
        "distance": [],
        "projected": [],
        "lateral": [],
        "opposing": [],
        "sourceFacing": [],
        "targetFacing": [],
    }
    for begin in range(0, interface_count, settings.batch_interface_count):
        end = min(begin + settings.batch_interface_count, interface_count)
        source_index = np.arange(begin, end, dtype=np.int32)
        predicted = np.rint(
            sampling_position[source_index, None, :]
            + ray_steps[None, :, None] * normal[source_index, None, :]
        ).astype(np.int32)
        query = predicted[:, :, None, :] + offsets[None, None, :, :]
        query = query.reshape(-1, 3)
        source = np.repeat(source_index, queries_per_source)
        inside = np.all(
            (query >= 0) & (query < np.asarray(shape)[None, :]), axis=1
        )
        query = query[inside]
        source = source[inside]
        target = lattice[np.ravel_multi_index(query.T, shape)]
        present = (target >= 0) & (target != source)
        source = source[present]
        target = target[present]
        delta = position[target] - position[source]
        distance = np.linalg.norm(delta, axis=1)
        unit = delta / np.maximum(distance[:, None], 1.0e-6)
        source_facing = np.einsum("ij,ij->i", unit, normal[source])
        target_facing = np.einsum("ij,ij->i", -unit, normal[target])
        opposing = -np.einsum("ij,ij->i", normal[source], normal[target])
        projected = np.einsum("ij,ij->i", delta, normal[source])
        lateral = np.sqrt(np.maximum(distance**2 - projected**2, 0.0))
        physical = (
            (distance >= minimum_voxels)
            & (distance <= maximum_voxels)
            & (source_facing >= facing_cosine)
            & (target_facing >= facing_cosine)
            & (opposing >= opposing_cosine)
        )
        source = source[physical]
        target = target[physical]
        distance = distance[physical]
        projected = projected[physical]
        lateral = lateral[physical]
        opposing = opposing[physical]
        source_facing = source_facing[physical]
        target_facing = target_facing[physical]
        boundary_evidence = np.sqrt(evidence[source] * evidence[target])
        angular_quality = np.sqrt(
            np.clip(opposing, 0.0, 1.0)
            * np.clip(source_facing, 0.0, 1.0)
            * np.clip(target_facing, 0.0, 1.0)
        )
        lateral_quality = np.exp(
            -0.5 * (lateral / max(2.0 * stride, 1.0)) ** 2
        )
        score = boundary_evidence * angular_quality * lateral_quality
        source, target, score, batch_fields = _deduplicate_directed(
            source,
            target,
            score,
            {
                "distance": distance,
                "projected": projected,
                "lateral": lateral,
                "opposing": opposing,
                "sourceFacing": source_facing,
                "targetFacing": target_facing,
            },
            interface_count,
        )
        accumulated["source"].append(source)
        accumulated["target"].append(target)
        accumulated["score"].append(score)
        for name, value in batch_fields.items():
            accumulated[name].append(value)

    joined = {
        name: np.concatenate(values) if values else np.empty(0)
        for name, values in accumulated.items()
    }
    source, target, score, directed_fields = _deduplicate_directed(
        joined.pop("source").astype(np.int32),
        joined.pop("target").astype(np.int32),
        joined.pop("score").astype(np.float32),
        joined,
        interface_count,
    )
    directed_key = source.astype(np.int64) * interface_count + target
    order = np.lexsort((-score, directed_fields["projected"], source))
    ordered_source = source[order]
    start = np.flatnonzero(
        np.r_[True, ordered_source[1:] != ordered_source[:-1]]
    )
    rank = np.empty(len(source), dtype=np.int32)
    rank[order] = np.arange(len(order)) - np.repeat(
        start, np.diff(np.r_[start, len(order)])
    )

    reverse_key = target.astype(np.int64) * interface_count + source
    reverse_index = np.searchsorted(directed_key, reverse_key)
    reverse_exists = reverse_index < len(directed_key)
    reverse_exists[reverse_exists] &= (
        directed_key[reverse_index[reverse_exists]]
        == reverse_key[reverse_exists]
    )
    reverse_rank = np.full(len(source), -1, dtype=np.int32)
    reverse_rank[reverse_exists] = rank[reverse_index[reverse_exists]]

    low = np.minimum(source, target)
    high = np.maximum(source, target)
    pair_key = low.astype(np.int64) * interface_count + high
    pair_order = np.lexsort((-score, pair_key))
    ordered_pair_key = pair_key[pair_order]
    keep = np.ones(len(pair_order), dtype=bool)
    keep[1:] = ordered_pair_key[1:] != ordered_pair_key[:-1]
    chosen = pair_order[keep]
    source = source[chosen]
    target = target[chosen]
    score = score[chosen]
    reverse_exists = reverse_exists[chosen]
    reverse_rank = reverse_rank[chosen]
    rank = rank[chosen]
    pair_fields = {name: value[chosen] for name, value in directed_fields.items()}
    midpoint = 0.5 * (position[source] + position[target])
    sheet_normal = normal[source] - normal[target]
    sheet_normal /= np.maximum(
        np.linalg.norm(sheet_normal, axis=1, keepdims=True), 1.0e-6
    )
    mutual_first = reverse_exists & (rank == 0) & (reverse_rank == 0)
    degree = np.bincount(
        np.concatenate((source, target)), minlength=interface_count
    )
    arrays = {
        "sourceInterface": source.astype(np.int32),
        "targetInterface": target.astype(np.int32),
        "midpointXYZ": midpoint.astype(np.float32),
        "normalXYZ": sheet_normal.astype(np.float32),
        "thicknessVoxels": pair_fields["distance"].astype(np.float32),
        "projectedThicknessVoxels": pair_fields["projected"].astype(np.float32),
        "lateralResidualVoxels": pair_fields["lateral"].astype(np.float32),
        "opposingNormalCosine": pair_fields["opposing"].astype(np.float32),
        "sourceInwardCosine": pair_fields["sourceFacing"].astype(np.float32),
        "targetInwardCosine": pair_fields["targetFacing"].astype(np.float32),
        "physicalEvidenceScore": score.astype(np.float32),
        "sourceRayRank": rank.astype(np.int16),
        "targetRayRank": reverse_rank.astype(np.int16),
        "bidirectional": reverse_exists.astype(np.uint8),
        "mutualFirstHit": mutual_first.astype(np.uint8),
        "interfaceCandidateDegree": degree.astype(np.int32),
    }
    stats = {
        "interfaceCount": int(interface_count),
        "rayStepCount": int(len(ray_steps)),
        "rayOffsetCount": int(len(offsets)),
        "latticeQueriesPerInterface": int(queries_per_source),
        "directedCandidateCount": int(len(directed_key)),
        "undirectedCandidateCount": int(len(source)),
        "bidirectionalCandidateCount": int(np.count_nonzero(reverse_exists)),
        "mutualFirstHitCount": int(np.count_nonzero(mutual_first)),
        "interfaceWithCandidateCount": int(np.count_nonzero(degree)),
        "interfaceWithCandidateFraction": round(
            float(np.count_nonzero(degree)) / max(interface_count, 1), 6
        ),
        "candidateDegree": _percentiles(degree[degree > 0]),
        "thicknessVoxels": _percentiles(pair_fields["distance"]),
        "lateralResidualVoxels": _percentiles(pair_fields["lateral"]),
        "physicalEvidenceScore": _percentiles(score),
        "sourceRayRank": _percentiles(rank),
        "identityLabelsUsed": False,
    }
    return arrays, stats


def _load_interface_bank(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value if value.is_file() else value / "one-sided-interface-bank-v1.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("physical ribbon bank requires a complete interface bank")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("interface bank data hash differs from its manifest")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return manifest_path, manifest, arrays


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_physical_ribbon_bank(
    interface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonBankSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonBankSettings()
    manifest_path, manifest, interfaces = _load_interface_bank(interface_root)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_BANK_SCHEMA,
        "version": PHYSICAL_RIBBON_BANK_VERSION,
        "interfaceBank": {
            "manifestPath": str(manifest_path),
            "manifestSha256": sha256_file(manifest_path),
            "dataSha256": manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    output_manifest = output / f"{PHYSICAL_RIBBON_BANK_STEM}.json"
    output_data = output / f"{PHYSICAL_RIBBON_BANK_STEM}.npz"
    if not force and output_manifest.is_file() and output_data.is_file():
        cached = json.loads(output_manifest.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(output_data)
        ):
            return cached

    geometry = manifest["geometry"]
    source_origin = np.asarray(manifest["source"]["sourceOriginXYZ"], dtype=np.float32)
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    processing_stop = np.asarray(
        geometry["processingVoxelBounds"]["stopXYZExclusive"],
        dtype=np.float32,
    )
    processing_shape = np.asarray(
        geometry["processingShapeSamplingXYZ"], dtype=np.int32
    )
    stride_xyz = (processing_stop - processing_start) / processing_shape
    if not (
        np.allclose(stride_xyz, stride_xyz[0])
        and math.isclose(float(stride_xyz[0]), round(float(stride_xyz[0])))
    ):
        raise ValueError("physical ribbon bank requires an isotropic integer stride")
    stride = int(round(float(stride_xyz[0])))
    started = time.monotonic()
    arrays, stats = build_physical_ribbon_bank(
        interfaces,
        processing_shape_sampling_xyz=tuple(int(value) for value in processing_shape),
        processing_world_start_xyz=source_origin + processing_start,
        sampling_stride_voxels=stride,
        voxel_size_microns=float(manifest["source"]["voxelSizeMicrons"]),
        settings=resolved,
    )
    built = time.monotonic()
    _write_npz(output_data, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_BANK_SCHEMA,
        "version": PHYSICAL_RIBBON_BANK_VERSION,
        "state": "complete",
        "identity": identity,
        "source": manifest["source"],
        "geometry": geometry,
        "counts": stats,
        "timingSeconds": {
            "physicalPairing": round(built - started, 6),
            "writing": round(finished - built, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": output_data.name,
            "bytes": output_data.stat().st_size,
            "sha256": sha256_file(output_data),
            "fields": list(arrays),
        },
        "method": {
            "observation": "dense signed air-to-material CT interfaces",
            "papyrusPrior": (
                "a sheet is a bounded-thickness material ribbon between two "
                "opposing interfaces, each pointing inward toward the other"
            ),
            "firstHit": (
                "mutual first inward-ray hits are high-confidence cores; all "
                "later physically plausible hits remain explicit alternatives"
            ),
            "identityLabelsUsed": False,
            "selection": "candidate bank only; no sheet identity or ownership assigned",
        },
    }
    atomic_json(output_manifest, payload)
    return payload
