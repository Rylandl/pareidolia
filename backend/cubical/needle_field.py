from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .raw_acus import NeedleTable, read_needle_artifact


BLOCK_NEEDLE_FIELD_SCHEMA = "pareidolia.block-acus-needle-field"
BLOCK_NEEDLE_FIELD_VERSION = 1
BLOCK_NEEDLE_FIELD_STEM = "block-needle-field-v1"


Float3 = tuple[float, float, float]


@dataclass(frozen=True, slots=True)
class BlockNeedleFieldSettings:
    """One coupled axial-normal field over every canonical block needle."""

    neighbor_radius_voxels: float = 24.0
    maximum_neighbors: int = 24
    spatial_kernel_voxels: float = 14.0
    smoothing_weight: float = 0.8
    robust_smoothing_angle_degrees: float = 32.0
    iteration_count: int = 16
    damping: float = 0.65
    maximum_normal_hypotheses: int = 3
    minimum_candidate_cross_angle_degrees: float = 18.0
    candidate_residual_kernel_degrees: float = 12.0
    candidate_separation_degrees: float = 10.0
    tangent_compatibility_sigma_voxels: float = 3.0
    mixture_pairwise_weight: float = 10.0
    mixture_initial_temperature: float = 0.25
    mixture_temperature: float = 0.03
    mixture_damping: float = 0.65
    mixture_iteration_count: int = 32
    mixture_annealing_iterations: int = 16
    carrier_minimum_affinity: float = 0.7
    minimum_curvature_radius_voxels: float = 4.0
    compute: str = "auto"

    def __post_init__(self) -> None:
        positive = (
            self.neighbor_radius_voxels,
            self.spatial_kernel_voxels,
            self.robust_smoothing_angle_degrees,
            self.minimum_candidate_cross_angle_degrees,
            self.candidate_residual_kernel_degrees,
            self.candidate_separation_degrees,
            self.tangent_compatibility_sigma_voxels,
            self.mixture_initial_temperature,
            self.mixture_temperature,
            self.minimum_curvature_radius_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("needle-field scales must be finite and positive")
        if self.maximum_neighbors < 4:
            raise ValueError("needle field requires at least four neighbors")
        if self.iteration_count < 1:
            raise ValueError("needle-field iteration count must be positive")
        if self.maximum_normal_hypotheses < 1:
            raise ValueError("needle field requires at least one normal hypothesis")
        if self.mixture_iteration_count < 1:
            raise ValueError("needle mixture iteration count must be positive")
        if not 1 <= self.mixture_annealing_iterations <= self.mixture_iteration_count:
            raise ValueError(
                "needle mixture annealing iterations must lie inside the solve"
            )
        if not math.isfinite(self.smoothing_weight) or self.smoothing_weight < 0.0:
            raise ValueError("needle-field smoothing weight must be nonnegative")
        if not math.isfinite(self.damping) or not 0.0 < self.damping <= 1.0:
            raise ValueError("needle-field damping must lie in (0, 1]")
        if (
            not math.isfinite(self.mixture_pairwise_weight)
            or self.mixture_pairwise_weight < 0.0
        ):
            raise ValueError("needle mixture pairwise weight must be nonnegative")
        if (
            not math.isfinite(self.mixture_damping)
            or not 0.0 < self.mixture_damping <= 1.0
        ):
            raise ValueError("needle mixture damping must lie in (0, 1]")
        if self.mixture_initial_temperature < self.mixture_temperature:
            raise ValueError(
                "needle mixture initial temperature cannot be below its final temperature"
            )
        if not 0.0 < self.carrier_minimum_affinity < 1.0:
            raise ValueError("needle carrier affinity must lie in (0, 1)")
        if self.compute not in ("auto", "gpu", "cpu"):
            raise ValueError("needle-field compute must be auto, gpu, or cpu")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class NeedleNeighborGraph:
    neighbor_index: np.ndarray
    neighbor_distance_voxels: np.ndarray
    neighbor_weight: np.ndarray
    neighbor_count: np.ndarray

    def validate(self, needle_count: int) -> None:
        shape = self.neighbor_index.shape
        if len(shape) != 2 or shape[0] != needle_count:
            raise ValueError("needle neighbor index has invalid shape")
        if self.neighbor_distance_voxels.shape != shape:
            raise ValueError("needle neighbor distance shape differs")
        if self.neighbor_weight.shape != shape:
            raise ValueError("needle neighbor weight shape differs")
        if self.neighbor_count.shape != (needle_count,):
            raise ValueError("needle neighbor count has invalid shape")
        valid = self.neighbor_index >= 0
        if np.any(self.neighbor_index[valid] >= needle_count):
            raise ValueError("needle graph references an absent node")
        if np.any(~np.isfinite(self.neighbor_distance_voxels[valid])):
            raise ValueError("needle graph distances are non-finite")
        if np.any(~np.isfinite(self.neighbor_weight[valid])):
            raise ValueError("needle graph weights are non-finite")
        if np.any(self.neighbor_weight[valid] <= 0.0):
            raise ValueError("needle graph weights must be positive")


def _float3(values: Iterable[float], name: str) -> Float3:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or any(not math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain three finite values")
    return result  # type: ignore[return-value]


def _bounds_intersect(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
) -> bool:
    return bool(np.all(first_start < second_stop) and np.all(second_start < first_stop))


def load_canonical_block_needles(
    raw_roots: Iterable[str | Path],
    world_start_xyz: Iterable[float],
    world_stop_xyz_exclusive: Iterable[float],
    *,
    verify: bool = True,
) -> tuple[NeedleTable, dict[str, Any]]:
    """Union source-anchored extraction tiles and crop unique needle centers."""

    roots = tuple(Path(value).resolve() for value in raw_roots)
    if not roots:
        raise ValueError("canonical needle loading requires at least one raw root")
    start = np.asarray(_float3(world_start_xyz, "world start"), dtype=np.float64)
    stop = np.asarray(
        _float3(world_stop_xyz_exclusive, "world stop"), dtype=np.float64
    )
    if np.any(stop <= start):
        raise ValueError("needle-field world bounds must have positive extent")

    source_identity: Mapping[str, Any] | None = None
    source_origin: tuple[int, int, int] | None = None
    voxel_size_microns: float | None = None
    tile_by_id: dict[str, tuple[Path, dict[str, Any], str]] = {}
    duplicate_tiles = 0
    root_records: list[dict[str, Any]] = []
    for root in roots:
        pipeline_path = root / "pipeline.json"
        pipeline = json.loads(pipeline_path.read_text())
        if pipeline.get("state") != "complete":
            raise ValueError(f"raw Acus root is not complete: {root}")
        identity = pipeline.get("identity", {})
        current_source = identity.get("source")
        geometry = pipeline.get("sourceGeometry", {})
        current_origin = tuple(int(value) for value in geometry["sourceOriginXYZ"])
        current_voxel_size = float(geometry["voxelSizeMicrons"])
        if source_identity is None:
            source_identity = current_source
            source_origin = current_origin
            voxel_size_microns = current_voxel_size
        elif (
            current_source != source_identity
            or current_origin != source_origin
            or not math.isclose(
                current_voxel_size,
                float(voxel_size_microns),
                rel_tol=0.0,
                abs_tol=1.0e-9,
            )
        ):
            raise ValueError("raw Acus roots do not describe one common source")
        root_records.append(
            {
                "root": str(root),
                "pipelineSha256": sha256_file(pipeline_path),
                "identitySha256": identity.get("identitySha256"),
            }
        )
        for manifest_path in sorted(
            (root / "extraction-tiles").glob("*/needles-v1.json")
        ):
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("schema") != "pareidolia.raw-acus-needles":
                raise ValueError("unsupported canonical needle artifact")
            region = manifest.get("region", {})
            if region.get("kind") != "canonical-acus-extraction-tile":
                raise ValueError("raw extraction tile lacks canonical ownership")
            tile_id = str(region["id"])
            data_sha256 = str(manifest["data"]["sha256"])
            prior = tile_by_id.get(tile_id)
            if prior is not None:
                duplicate_tiles += 1
                if data_sha256 != prior[2]:
                    raise ValueError(
                        f"canonical tile {tile_id} differs between raw roots"
                    )
                continue
            tile_by_id[tile_id] = (manifest_path, manifest, data_sha256)
    if source_origin is None or voxel_size_microns is None:
        raise RuntimeError("raw source geometry was not resolved")
    source_shift = np.asarray(source_origin, dtype=np.float64)

    tables: list[NeedleTable] = []
    selected_tile_records: list[dict[str, Any]] = []
    canonical_candidates = 0
    for tile_id, (manifest_path, manifest, data_sha256) in sorted(tile_by_id.items()):
        bounds = manifest["region"]["coreVoxelBounds"]
        tile_start = np.asarray(bounds["startXYZ"], dtype=np.float64) + source_shift
        tile_stop = (
            np.asarray(bounds["stopXYZExclusive"], dtype=np.float64) + source_shift
        )
        if not _bounds_intersect(tile_start, tile_stop, start, stop):
            continue
        prefix = manifest_path.with_suffix("")
        table = read_needle_artifact(
            prefix,
            identity_sha256=str(manifest["identitySha256"]),
            verify=verify,
        )
        centers = table.center_xyz.astype(np.float64) + source_shift[None, :]
        owned = np.all((centers >= start[None, :]) & (centers < stop[None, :]), axis=1)
        canonical_candidates += table.count
        if np.any(owned):
            tables.append(
                NeedleTable(
                    centers[owned].astype(np.float32),
                    table.direction_xyz[owned],
                    table.score[owned],
                    table.axial_coverage[owned],
                    table.support_score[owned],
                )
            )
        selected_tile_records.append(
            {
                "id": tile_id,
                "dataSha256": data_sha256,
                "candidateCount": table.count,
                "retainedInBounds": int(np.count_nonzero(owned)),
            }
        )
    if not tables:
        raise ValueError("needle-field bounds contain no canonical Acus needles")
    center = np.concatenate([value.center_xyz for value in tables])
    direction = np.concatenate([value.direction_xyz for value in tables])
    score = np.concatenate([value.score for value in tables])
    axial = np.concatenate([value.axial_coverage for value in tables])
    support = np.concatenate([value.support_score for value in tables])
    order = np.lexsort(
        (
            direction[:, 2],
            direction[:, 1],
            direction[:, 0],
            center[:, 2],
            center[:, 1],
            center[:, 0],
        )
    )
    result = NeedleTable(
        center[order],
        direction[order],
        score[order],
        axial[order],
        support[order],
    )
    result.validate()
    return result, {
        "rawRoots": root_records,
        "worldBounds": {
            "startXYZ": start.tolist(),
            "stopXYZExclusive": stop.tolist(),
            "coordinateUnit": "source-voxel",
        },
        "sourceOriginXYZ": list(source_origin),
        "voxelSizeMicrons": voxel_size_microns,
        "uniqueCanonicalTilesAvailable": len(tile_by_id),
        "duplicateCanonicalTileOccurrences": duplicate_tiles,
        "selectedCanonicalTiles": len(selected_tile_records),
        "selectedTileCandidateCountBeforeBoundsCrop": canonical_candidates,
        "uniqueNeedlesInBounds": result.count,
        "tiles": selected_tile_records,
    }


def build_needle_neighbor_graph(
    needles: NeedleTable,
    settings: BlockNeedleFieldSettings,
) -> NeedleNeighborGraph:
    """Build deterministic bounded-radius nearest-neighbor rows by spatial hash."""

    needles.validate()
    centers = np.asarray(needles.center_xyz, dtype=np.float64)
    count = needles.count
    radius = settings.neighbor_radius_voxels
    radius2 = radius**2
    bucket_size = radius
    bucket_xyz = np.floor(centers / bucket_size).astype(np.int32)
    buckets: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for index, value in enumerate(bucket_xyz):
        buckets[tuple(int(item) for item in value)].append(index)
    neighbor_index = np.full(
        (count, settings.maximum_neighbors), -1, dtype=np.int32
    )
    neighbor_distance = np.full(
        neighbor_index.shape, np.nan, dtype=np.float32
    )
    neighbor_weight = np.zeros(neighbor_index.shape, dtype=np.float32)
    neighbor_count = np.zeros(count, dtype=np.uint16)
    quality = (
        needles.score
        * np.sqrt(np.maximum(needles.axial_coverage * needles.support_score, 0.0))
    ).astype(np.float64)
    sigma2 = settings.spatial_kernel_voxels**2
    for index in range(count):
        base = bucket_xyz[index]
        candidate_values: list[int] = []
        for dz in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for dx in (-1, 0, 1):
                    candidate_values.extend(
                        buckets.get(
                            (
                                int(base[0]) + dx,
                                int(base[1]) + dy,
                                int(base[2]) + dz,
                            ),
                            (),
                        )
                    )
        candidate = np.asarray(candidate_values, dtype=np.int32)
        if not len(candidate):
            continue
        delta = centers[candidate] - centers[index]
        distance2 = np.einsum("ij,ij->i", delta, delta)
        keep = (candidate != index) & (distance2 <= radius2)
        candidate = candidate[keep]
        distance2 = distance2[keep]
        if not len(candidate):
            continue
        order = np.lexsort((candidate, distance2))[: settings.maximum_neighbors]
        candidate = candidate[order]
        distance2 = distance2[order]
        length = len(candidate)
        neighbor_index[index, :length] = candidate
        neighbor_distance[index, :length] = np.sqrt(distance2).astype(np.float32)
        neighbor_weight[index, :length] = (
            np.exp(-0.5 * distance2 / sigma2)
            * np.sqrt(np.maximum(quality[index] * quality[candidate], 1.0e-8))
        ).astype(np.float32)
        neighbor_count[index] = length
    graph = NeedleNeighborGraph(
        neighbor_index,
        neighbor_distance,
        neighbor_weight,
        neighbor_count,
    )
    graph.validate(count)
    return graph


def _initial_normal_field(
    needles: NeedleTable,
    graph: NeedleNeighborGraph,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count, width = graph.neighbor_index.shape
    index = np.maximum(graph.neighbor_index, 0)
    valid = graph.neighbor_index >= 0
    weights = graph.neighbor_weight.astype(np.float64) * valid
    directions = needles.direction_xyz[index].astype(np.float64)
    tensor = np.einsum(
        "nki,nkj,nk->nij", directions, directions, weights, optimize=True
    )
    self_quality = (
        needles.score
        * np.sqrt(np.maximum(needles.axial_coverage * needles.support_score, 0.0))
    ).astype(np.float64)
    tensor += np.einsum(
        "ni,nj,n->nij",
        needles.direction_xyz.astype(np.float64),
        needles.direction_xyz.astype(np.float64),
        self_quality,
        optimize=True,
    )
    trace = np.trace(tensor, axis1=1, axis2=2)
    tensor /= np.maximum(trace[:, None, None], 1.0e-12)
    eigenvalues, eigenvectors = np.linalg.eigh(tensor)
    normals = eigenvectors[:, :, 0]
    dominant = np.argmax(np.abs(normals), axis=1)
    sign = np.where(normals[np.arange(count), dominant] < 0.0, -1.0, 1.0)
    normals *= sign[:, None]
    confidence = np.clip(
        (eigenvalues[:, 1] - eigenvalues[:, 0])
        / np.maximum(np.sum(eigenvalues, axis=1), 1.0e-12),
        0.0,
        1.0,
    )
    return (
        tensor.astype(np.float32),
        eigenvalues.astype(np.float32),
        normals.astype(np.float32),
        confidence.astype(np.float32),
    )


def _canonicalize_axis_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64).copy()
    result /= np.maximum(np.linalg.norm(result, axis=-1, keepdims=True), 1.0e-12)
    flat = result.reshape(-1, 3)
    dominant = np.argmax(np.abs(flat), axis=1)
    signs = np.where(flat[np.arange(len(flat)), dominant] < 0.0, -1.0, 1.0)
    flat *= signs[:, None]
    return result.astype(np.float32)


def build_normal_hypothesis_bank(
    needles: NeedleTable,
    graph: NeedleNeighborGraph,
    seed_normals: np.ndarray,
    settings: BlockNeedleFieldSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Generate multiple locally supported normals for every needle node.

    Crossing one needle with nearby nonparallel fibers produces candidate page
    normals without assigning signs.  Robust support over the full spatial
    neighborhood ranks those candidates, while angular nonmaximum suppression
    preserves genuinely distinct folded populations.
    """

    count, width = graph.neighbor_index.shape
    hypotheses = settings.maximum_normal_hypotheses
    neighbor_index = np.maximum(graph.neighbor_index, 0)
    valid_neighbor = graph.neighbor_index >= 0
    neighbor_direction = needles.direction_xyz[neighbor_index].astype(np.float64)
    own_direction = needles.direction_xyz.astype(np.float64)
    base_weight = graph.neighbor_weight.astype(np.float64) * valid_neighbor
    own_quality = (
        needles.score
        * np.sqrt(np.maximum(needles.axial_coverage * needles.support_score, 0.0))
    ).astype(np.float64)
    candidates = np.zeros((count, hypotheses, 3), dtype=np.float32)
    support = np.zeros((count, hypotheses), dtype=np.float32)
    candidate_valid = np.zeros((count, hypotheses), dtype=np.uint8)
    minimum_cross = math.sin(
        math.radians(settings.minimum_candidate_cross_angle_degrees)
    )
    kernel = math.radians(settings.candidate_residual_kernel_degrees)
    separation_cosine = math.cos(math.radians(settings.candidate_separation_degrees))
    chunk_size = 2048
    for low in range(0, count, chunk_size):
        high = min(low + chunk_size, count)
        own = own_direction[low:high]
        neighbor = neighbor_direction[low:high]
        cross = np.cross(own[:, None, :], neighbor)
        cross_length = np.linalg.norm(cross, axis=-1)
        cross /= np.maximum(cross_length[..., None], 1.0e-12)
        seeds = np.concatenate(
            (seed_normals[low:high, None, :].astype(np.float64), cross), axis=1
        )
        seed_valid = np.concatenate(
            (
                np.ones((high - low, 1), dtype=bool),
                valid_neighbor[low:high] & (cross_length >= minimum_cross),
            ),
            axis=1,
        )
        dot = np.einsum("nsi,nki->nsk", seeds, neighbor, optimize=True)
        residual = np.arcsin(np.clip(np.abs(dot), 0.0, 1.0))
        score = np.sum(
            base_weight[low:high, None, :]
            * np.exp(-0.5 * (residual / kernel) ** 2),
            axis=-1,
        )
        own_dot = np.abs(np.einsum("nsi,ni->ns", seeds, own, optimize=True))
        own_residual = np.arcsin(np.clip(own_dot, 0.0, 1.0))
        score += own_quality[low:high, None] * np.exp(
            -0.5 * (own_residual / kernel) ** 2
        )
        score[~seed_valid] = -np.inf
        for local_index in range(high - low):
            chosen: list[np.ndarray] = []
            for seed_index in np.argsort(score[local_index])[::-1]:
                if not np.isfinite(score[local_index, seed_index]):
                    continue
                seed = seeds[local_index, seed_index]
                if any(
                    abs(float(np.dot(seed, prior))) >= separation_cosine
                    for prior in chosen
                ):
                    continue
                output_index = len(chosen)
                candidates[low + local_index, output_index] = seed
                support[low + local_index, output_index] = score[
                    local_index, seed_index
                ]
                candidate_valid[low + local_index, output_index] = 1
                chosen.append(seed)
                if len(chosen) >= hypotheses:
                    break

    # One robust tensor refit improves each seed, then projection against the
    # node's own fiber enforces the defining page-normal constraint exactly.
    for low in range(0, count, chunk_size):
        high = min(low + chunk_size, count)
        candidate = candidates[low:high].astype(np.float64)
        neighbor = neighbor_direction[low:high]
        dot = np.einsum("nhi,nki->nhk", candidate, neighbor, optimize=True)
        residual = np.arcsin(np.clip(np.abs(dot), 0.0, 1.0))
        weights = (
            base_weight[low:high, None, :]
            * np.exp(-0.5 * (residual / kernel) ** 2)
            * candidate_valid[low:high, :, None]
        )
        tensor = np.einsum(
            "nki,nkj,nhk->nhij", neighbor, neighbor, weights, optimize=True
        )
        own = own_direction[low:high]
        tensor += np.einsum(
            "ni,nj,nh->nhij",
            own,
            own,
            own_quality[low:high, None] * candidate_valid[low:high],
            optimize=True,
        )
        _, eigenvectors = np.linalg.eigh(tensor)
        refined = eigenvectors[..., 0]
        refined -= own[:, None, :] * np.sum(
            refined * own[:, None, :], axis=-1, keepdims=True
        )
        refined_length = np.linalg.norm(refined, axis=-1)
        refined = np.where(
            (refined_length > 1.0e-8)[..., None],
            refined / np.maximum(refined_length[..., None], 1.0e-12),
            candidate,
        )
        refined_dot = np.einsum("nhi,nki->nhk", refined, neighbor, optimize=True)
        refined_residual = np.arcsin(
            np.clip(np.abs(refined_dot), 0.0, 1.0)
        )
        refined_support = np.sum(
            base_weight[low:high, None, :]
            * np.exp(-0.5 * (refined_residual / kernel) ** 2),
            axis=-1,
        ) + own_quality[low:high, None]
        refined_support *= candidate_valid[low:high]
        candidates[low:high] = refined.astype(np.float32)
        support[low:high] = refined_support.astype(np.float32)
    candidates = _canonicalize_axis_rows(candidates)
    total_support = (
        np.sum(base_weight, axis=1) + own_quality
    )[:, None]
    support_fraction = support / np.maximum(total_support, 1.0e-8)
    support_fraction *= candidate_valid
    return candidates, support_fraction.astype(np.float32), candidate_valid


def _smallest_symmetric_eigenvector(
    matrices: Any,
    xp: Any,
) -> tuple[Any, Any]:
    """Analytic smallest eigenpair for a batch of symmetric 3x3 matrices."""

    xx = matrices[..., 0, 0]
    xy = matrices[..., 0, 1]
    xz = matrices[..., 0, 2]
    yy = matrices[..., 1, 1]
    yz = matrices[..., 1, 2]
    zz = matrices[..., 2, 2]
    trace_third = (xx + yy + zz) / 3.0
    axx = xx - trace_third
    ayy = yy - trace_third
    azz = zz - trace_third
    p = xp.sqrt(
        xp.maximum(
            (
                axx * axx
                + ayy * ayy
                + azz * azz
                + 2.0 * (xy * xy + xz * xz + yz * yz)
            )
            / 6.0,
            0.0,
        )
    )
    p_safe = xp.maximum(p, 1.0e-12)
    determinant = (
        axx * (ayy * azz - yz * yz)
        - xy * (xy * azz - yz * xz)
        + xz * (xy * yz - ayy * xz)
    )
    phase = xp.arccos(
        xp.clip(determinant / (2.0 * p_safe**3), -1.0, 1.0)
    ) / 3.0
    smallest = trace_third + 2.0 * p * xp.cos(
        phase + 2.0 * math.pi / 3.0
    )
    r0 = xp.stack((xx - smallest, xy, xz), axis=-1)
    r1 = xp.stack((xy, yy - smallest, yz), axis=-1)
    r2 = xp.stack((xz, yz, zz - smallest), axis=-1)
    candidates = xp.stack(
        (xp.cross(r0, r1), xp.cross(r0, r2), xp.cross(r1, r2)), axis=-2
    )
    norm2 = xp.sum(candidates * candidates, axis=-1)
    best = xp.argmax(norm2, axis=-1)
    vector = xp.take_along_axis(
        candidates, best[..., None, None], axis=-2
    )[..., 0, :]
    length = xp.sqrt(xp.sum(vector * vector, axis=-1, keepdims=True))
    vector = vector / xp.maximum(length, 1.0e-12)
    valid = xp.max(norm2, axis=-1) > 1.0e-18
    return smallest, xp.where(valid[..., None], vector, 0.0)


def optimize_normal_hypothesis_mixture(
    center_xyz: np.ndarray,
    candidates: np.ndarray,
    support_fraction: np.ndarray,
    candidate_valid: np.ndarray,
    graph: NeedleNeighborGraph,
    settings: BlockNeedleFieldSettings,
    *,
    compute_module: Any = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Globally cluster spatially varying normal modes by damped mean field."""

    xp = compute_module if compute_module is not None else np
    normal = xp.asarray(candidates, dtype=xp.float32)
    support = xp.asarray(support_fraction, dtype=xp.float32)
    valid_candidate = xp.asarray(candidate_valid.astype(bool))
    neighbor_index = xp.asarray(
        np.maximum(graph.neighbor_index, 0), dtype=xp.int32
    )
    valid_neighbor = xp.asarray(graph.neighbor_index >= 0)
    base_weight = xp.asarray(graph.neighbor_weight, dtype=xp.float32) * valid_neighbor
    weight_sum = xp.maximum(xp.sum(base_weight, axis=1), 1.0e-12)
    center = xp.asarray(center_xyz, dtype=xp.float32)
    displacement = center[neighbor_index] - center[:, None, :]
    unary = -xp.log(xp.maximum(support, 1.0e-5))
    unary = xp.where(valid_candidate, unary, 40.0)

    def softmax_negative(cost: Any, temperature: float) -> Any:
        logits = -cost / temperature
        logits = xp.where(valid_candidate, logits, -1.0e9)
        logits -= xp.max(logits, axis=1, keepdims=True)
        probability = xp.exp(logits) * valid_candidate
        return probability / xp.maximum(
            xp.sum(probability, axis=1, keepdims=True), 1.0e-12
        )

    probability = softmax_negative(unary, settings.mixture_initial_temperature)
    records: list[dict[str, Any]] = []
    for iteration in range(settings.mixture_iteration_count):
        fraction = (
            min(iteration, settings.mixture_annealing_iterations - 1)
            / max(settings.mixture_annealing_iterations - 1, 1)
        )
        temperature = settings.mixture_initial_temperature * (
            settings.mixture_temperature
            / settings.mixture_initial_temperature
        ) ** fraction
        neighbor_normal = normal[neighbor_index]
        neighbor_probability = probability[neighbor_index]
        signed_cosine = xp.einsum(
            "nhi,nkli->nhkl", normal, neighbor_normal, optimize=True
        )
        axial_cosine = xp.clip(xp.abs(signed_cosine), 0.0, 1.0)
        axial_similarity = axial_cosine * axial_cosine
        source_height = xp.einsum(
            "nki,nhi->nhk", displacement, normal, optimize=True
        )
        target_height = xp.einsum(
            "nki,nkli->nkl", displacement, neighbor_normal, optimize=True
        )
        aligned_target_height = (
            target_height[:, None, :, :]
            * xp.where(signed_cosine < 0.0, -1.0, 1.0)
        )
        expanded_source_height = source_height[..., None]
        midpoint_residual = 0.5 * xp.abs(
            expanded_source_height + aligned_target_height
        )
        observed_sag = 0.5 * xp.abs(
            expanded_source_height - aligned_target_height
        )
        normal_angle = xp.arccos(axial_cosine)
        chord_length = xp.sqrt(xp.sum(displacement * displacement, axis=-1))
        expected_sag = (
            chord_length[:, None, :, None] * xp.sin(0.5 * normal_angle)
        )
        bend_residual = xp.abs(observed_sag - expected_sag)
        curvature_radius = chord_length[:, None, :, None] / xp.maximum(
            2.0 * xp.sin(0.5 * normal_angle), 1.0e-6
        )
        resolved_curvature = (
            (normal_angle <= 1.0e-4)
            | (curvature_radius >= settings.minimum_curvature_radius_voxels)
        )
        tangent_sigma = settings.tangent_compatibility_sigma_voxels
        tangent_affinity = (
            xp.exp(-0.5 * (midpoint_residual / tangent_sigma) ** 2)
            * xp.exp(-0.5 * (bend_residual / tangent_sigma) ** 2)
            * resolved_curvature
        )
        pair_affinity = axial_similarity * tangent_affinity
        expected_reward = xp.sum(
            pair_affinity
            * neighbor_probability[:, None, :, :]
            * base_weight[:, None, :, None],
            axis=(2, 3),
        ) / weight_sum[:, None]
        proposal = softmax_negative(
            unary - settings.mixture_pairwise_weight * expected_reward,
            temperature,
        )
        updated = (
            (1.0 - settings.mixture_damping) * probability
            + settings.mixture_damping * proposal
        )
        updated /= xp.maximum(xp.sum(updated, axis=1, keepdims=True), 1.0e-12)
        change = xp.max(xp.abs(updated - probability), axis=1)
        if compute_module is not None:
            change_values = compute_module.asnumpy(change)
        else:
            change_values = np.asarray(change)
        records.append(
            {
                "iteration": iteration + 1,
                "temperature": round(float(temperature), 8),
                "medianMaximumProbabilityChange": round(
                    float(np.median(change_values)), 8
                ),
                "p90MaximumProbabilityChange": round(
                    float(np.percentile(change_values, 90)), 8
                ),
                "maximumProbabilityChange": round(float(np.max(change_values)), 8),
            }
        )
        probability = updated
    label = xp.argmax(probability, axis=1)
    selected = xp.take_along_axis(
        normal, label[:, None, None], axis=1
    )[:, 0, :]
    if compute_module is not None:
        return (
            compute_module.asnumpy(selected).astype(np.float32),
            compute_module.asnumpy(probability).astype(np.float32),
            compute_module.asnumpy(label).astype(np.uint8),
            records,
        )
    return (
        np.asarray(selected, dtype=np.float32),
        np.asarray(probability, dtype=np.float32),
        np.asarray(label, dtype=np.uint8),
        records,
    )


def _percentile_record(values: np.ndarray) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {
            "count": 0,
            "median": None,
            "p90": None,
            "p99": None,
            "maximum": None,
        }
    quantiles = np.percentile(finite, (50, 90, 99, 100))
    return {
        "count": len(finite),
        **{
            name: round(float(value), 6)
            for name, value in zip(("median", "p90", "p99", "maximum"), quantiles)
        },
    }


def _field_metrics(
    needles: NeedleTable,
    graph: NeedleNeighborGraph,
    normals: np.ndarray,
    initial_normals: np.ndarray,
    confidence: np.ndarray,
) -> dict[str, Any]:
    plane_residual = np.degrees(
        np.arcsin(
            np.clip(
                np.abs(np.einsum("ij,ij->i", normals, needles.direction_xyz)),
                0.0,
                1.0,
            )
        )
    )
    valid = graph.neighbor_index >= 0
    source = np.broadcast_to(
        np.arange(needles.count, dtype=np.int32)[:, None], valid.shape
    )[valid]
    target = graph.neighbor_index[valid]
    neighbor_angle = np.degrees(
        np.arccos(
            np.clip(
                np.abs(np.einsum("ij,ij->i", normals[source], normals[target])),
                0.0,
                1.0,
            )
        )
    )
    change = np.degrees(
        np.arccos(
            np.clip(
                np.abs(np.einsum("ij,ij->i", normals, initial_normals)),
                0.0,
                1.0,
            )
        )
    )
    return {
        "needlePlaneResidualDegrees": _percentile_record(plane_residual),
        "neighborAxialNormalAngleDegrees": _percentile_record(neighbor_angle),
        "changeFromLocalTensorNormalDegrees": _percentile_record(change),
        "localTensorConfidence": _percentile_record(confidence),
        "nodesWithFewerThanFourNeighbors": int(
            np.count_nonzero(graph.neighbor_count < 4)
        ),
        "nodesWithConfidenceBelow005": int(np.count_nonzero(confidence < 0.05)),
    }


def curvature_aware_tangent_metrics(
    displacement_xyz: np.ndarray,
    first_normal_xyz: np.ndarray,
    second_normal_xyz: np.ndarray,
    *,
    compatibility_sigma_voxels: float,
    minimum_curvature_radius_voxels: float,
) -> dict[str, np.ndarray]:
    """Score one smooth Hermite chord without flattening real curvature.

    After aligning the unsigned endpoint normals, a locally smooth chord has
    equal-and-opposite signed tangent-plane offsets.  Their mean is therefore
    a layer-shift residual.  Their antisymmetric magnitude is compared with
    the circular-arc sag implied by the endpoint normal angle.  A parallel
    layer jump has same-sign offsets and is rejected even when its normals are
    identical; a resolved bend is not penalized merely for being curved.
    """

    displacement = np.asarray(displacement_xyz, dtype=np.float64)
    first = np.asarray(first_normal_xyz, dtype=np.float64)
    second = np.asarray(second_normal_xyz, dtype=np.float64)
    if displacement.ndim != 2 or displacement.shape[1:] != (3,):
        raise ValueError("Hermite displacement must have shape (N, 3)")
    if first.shape != displacement.shape or second.shape != displacement.shape:
        raise ValueError("Hermite normals must match displacement shape")
    if compatibility_sigma_voxels <= 0.0:
        raise ValueError("Hermite compatibility sigma must be positive")
    if minimum_curvature_radius_voxels <= 0.0:
        raise ValueError("minimum curvature radius must be positive")

    signed_cosine = np.einsum("ij,ij->i", first, second)
    alignment = np.where(signed_cosine < 0.0, -1.0, 1.0)
    aligned_second = second * alignment[:, None]
    axial_cosine = np.clip(np.abs(signed_cosine), 0.0, 1.0)
    normal_angle = np.arccos(axial_cosine)
    first_height = np.einsum("ij,ij->i", displacement, first)
    second_height = np.einsum("ij,ij->i", displacement, aligned_second)
    midpoint_residual = 0.5 * np.abs(first_height + second_height)
    observed_sag = 0.5 * np.abs(first_height - second_height)
    chord_length = np.linalg.norm(displacement, axis=1)
    sine_half_angle = np.sin(0.5 * normal_angle)
    expected_sag = chord_length * sine_half_angle
    bend_residual = np.abs(observed_sag - expected_sag)
    curvature_radius = np.divide(
        chord_length,
        2.0 * sine_half_angle,
        out=np.full_like(chord_length, np.inf),
        where=sine_half_angle > 1.0e-6,
    )
    resolved = (normal_angle <= 1.0e-4) | (
        curvature_radius >= minimum_curvature_radius_voxels
    )
    sigma = float(compatibility_sigma_voxels)
    affinity = (
        axial_cosine**2
        * np.exp(-0.5 * (midpoint_residual / sigma) ** 2)
        * np.exp(-0.5 * (bend_residual / sigma) ** 2)
        * resolved
    ).astype(np.float32)
    return {
        "affinity": affinity,
        "normalAngleDegrees": np.degrees(normal_angle).astype(np.float32),
        "midpointLayerShiftVoxels": midpoint_residual.astype(np.float32),
        "bendModelResidualVoxels": bend_residual.astype(np.float32),
        "curvatureRadiusVoxels": curvature_radius.astype(np.float32),
    }


def analyze_needle_carriers(
    needles: NeedleTable,
    graph: NeedleNeighborGraph,
    normals: np.ndarray,
    settings: BlockNeedleFieldSettings,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Materialize conservative tangential carriers from selected normal modes."""

    index = graph.neighbor_index
    valid = index >= 0
    source = np.broadcast_to(
        np.arange(needles.count, dtype=np.int32)[:, None], index.shape
    )[valid]
    target = index[valid]
    displacement = needles.center_xyz[target] - needles.center_xyz[source]
    geometry = curvature_aware_tangent_metrics(
        displacement,
        normals[source],
        normals[target],
        compatibility_sigma_voxels=settings.tangent_compatibility_sigma_voxels,
        minimum_curvature_radius_voxels=settings.minimum_curvature_radius_voxels,
    )
    affinity = geometry["affinity"]
    normal_angle = geometry["normalAngleDegrees"]
    retained = affinity >= settings.carrier_minimum_affinity
    first = np.minimum(source[retained], target[retained])
    second = np.maximum(source[retained], target[retained])
    values = affinity[retained]
    if len(first):
        order = np.lexsort((second, first))
        first = first[order]
        second = second[order]
        values = values[order]
        boundary = np.concatenate(
            (
                np.asarray((True,)),
                (first[1:] != first[:-1]) | (second[1:] != second[:-1]),
            )
        )
        starts = np.flatnonzero(boundary)
        carrier_first = first[starts]
        carrier_second = second[starts]
        carrier_affinity = np.maximum.reduceat(values, starts)
    else:
        carrier_first = np.empty(0, dtype=np.int32)
        carrier_second = np.empty(0, dtype=np.int32)
        carrier_affinity = np.empty(0, dtype=np.float32)

    parent = np.arange(needles.count, dtype=np.int32)
    size = np.ones(needles.count, dtype=np.int32)

    def find(value: int) -> int:
        while int(parent[value]) != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first_node, second_node in zip(carrier_first, carrier_second):
        first_root = find(int(first_node))
        second_root = find(int(second_node))
        if first_root == second_root:
            continue
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
    root = np.asarray([find(value) for value in range(needles.count)], dtype=np.int32)
    stable_component = np.full(needles.count, needles.count, dtype=np.int32)
    np.minimum.at(stable_component, root, np.arange(needles.count, dtype=np.int32))
    component_id = stable_component[root]
    component_values, component_sizes = np.unique(
        component_id, return_counts=True
    )
    ranking = np.lexsort((component_values, -component_sizes))
    component_values = component_values[ranking]
    component_sizes = component_sizes[ranking]
    top: list[dict[str, Any]] = []
    for rank, (value, count) in enumerate(
        zip(component_values[:64], component_sizes[:64]), start=1
    ):
        members = np.flatnonzero(component_id == value)
        points = needles.center_xyz[members]
        member_normals = normals[members].astype(np.float64)
        projector = np.einsum(
            "ni,nj->ij", member_normals, member_normals, optimize=True
        )
        _, eigenvectors = np.linalg.eigh(projector)
        reference = eigenvectors[:, -1]
        cone = np.degrees(
            np.arccos(
                np.clip(np.abs(member_normals @ reference), 0.0, 1.0)
            )
        )
        fiber = needles.direction_xyz[members].astype(np.float64)
        fiber_projector = np.einsum("ni,nj->ij", fiber, fiber, optimize=True)
        fiber_eigenvalues = np.linalg.eigvalsh(fiber_projector)
        fiber_eigenvalues /= max(float(np.sum(fiber_eigenvalues)), 1.0e-12)
        top.append(
            {
                "rank": rank,
                "componentId": int(value),
                "needles": int(count),
                "worldStartXYZ": [
                    round(float(item), 6) for item in np.min(points, axis=0)
                ],
                "worldStopXYZ": [
                    round(float(item), 6) for item in np.max(points, axis=0)
                ],
                "extentVoxelsXYZ": [
                    round(float(item), 6) for item in np.ptp(points, axis=0)
                ],
                "normalConeDegrees": {
                    "median": round(float(np.median(cone)), 6),
                    "p90": round(float(np.percentile(cone, 90)), 6),
                    "maximum": round(float(np.max(cone)), 6),
                },
                "fiberProjectorEigenvalueFractions": [
                    round(float(item), 6) for item in fiber_eigenvalues
                ],
            }
        )
    retained_directed_angle = normal_angle[retained]
    retained_midpoint = geometry["midpointLayerShiftVoxels"][retained]
    retained_bend = geometry["bendModelResidualVoxels"][retained]
    retained_radius = geometry["curvatureRadiusVoxels"][retained]
    summary = {
        "method": {
            "edgeAffinity": (
                "axial-normal cosine squared times curvature-aware Hermite "
                "midpoint-layer-shift and circular-bend compatibility"
            ),
            "component": (
                "connected component of unique undirected edges at the declared "
                "conservative affinity threshold; diagnostic carrier, not final sheet"
            ),
        },
        "minimumAffinity": settings.carrier_minimum_affinity,
        "directedAffinity": _percentile_record(affinity),
        "directedEdgesAtLeastAffinity": {
            str(value): int(np.count_nonzero(affinity >= value))
            for value in (0.25, 0.5, 0.7, 0.85, 0.93)
        },
        "counts": {
            "uniqueCarrierEdges": len(carrier_first),
            "components": len(component_values),
            "isolatedNeedles": int(np.count_nonzero(component_sizes == 1)),
            "componentsAtLeastNeedles": {
                str(value): int(np.count_nonzero(component_sizes >= value))
                for value in (8, 16, 32, 64, 128, 256, 512, 1024)
            },
            "largestComponentNeedles": int(component_sizes[0]),
        },
        "retainedEdgeNormalAngleDegrees": _percentile_record(
            retained_directed_angle
        ),
        "retainedEdgeMidpointLayerShiftVoxels": _percentile_record(
            retained_midpoint
        ),
        "retainedEdgeBendModelResidualVoxels": _percentile_record(retained_bend),
        "retainedEdgeCurvatureRadiusVoxels": _percentile_record(retained_radius),
        "topComponents": top,
    }
    arrays = {
        "carrierComponentId": component_id,
        "carrierEdgeFirstNeedle": carrier_first.astype(np.int32, copy=False),
        "carrierEdgeSecondNeedle": carrier_second.astype(np.int32, copy=False),
        "carrierEdgeAffinity": carrier_affinity.astype(np.float32, copy=False),
    }
    return summary, arrays


def optimize_block_needle_field(
    needles: NeedleTable,
    graph: NeedleNeighborGraph,
    *,
    settings: BlockNeedleFieldSettings | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Solve a robust unsigned normal field jointly over the entire graph."""

    resolved = settings or BlockNeedleFieldSettings()
    graph.validate(needles.count)
    data_tensor, local_eigenvalues, initial, confidence = _initial_normal_field(
        needles, graph
    )
    backend = "cpu"
    device: str | None = None
    fallback_reason: str | None = None
    cp = None
    if resolved.compute != "cpu":
        try:
            import cupy as cp_module

            device_count = int(cp_module.cuda.runtime.getDeviceCount())
            if device_count < 1:
                raise RuntimeError("no CUDA device")
            cp = cp_module
            backend = "gpu"
            properties = cp.cuda.runtime.getDeviceProperties(0)
            raw_name = properties["name"]
            device = raw_name.decode() if isinstance(raw_name, bytes) else str(raw_name)
        except Exception as error:  # pragma: no cover - host dependent
            fallback_reason = f"CUDA initialization failed: {error}"
            if resolved.compute == "gpu":
                raise RuntimeError(fallback_reason) from error
            cp = None
    xp = cp if cp is not None else np
    neighbor_index = xp.asarray(np.maximum(graph.neighbor_index, 0), dtype=xp.int32)
    valid = xp.asarray(graph.neighbor_index >= 0)
    base_weight = xp.asarray(graph.neighbor_weight, dtype=xp.float32)
    tensor = xp.asarray(data_tensor, dtype=xp.float32)
    data_strength = xp.asarray(0.2 + 1.8 * confidence, dtype=xp.float32)
    normals = xp.asarray(initial, dtype=xp.float32)
    robust_scale = math.radians(resolved.robust_smoothing_angle_degrees)
    iteration_records: list[dict[str, Any]] = []
    solve_started = time.monotonic()
    for iteration in range(resolved.iteration_count):
        neighbor_normal = normals[neighbor_index]
        cosine = xp.clip(
            xp.abs(xp.sum(normals[:, None, :] * neighbor_normal, axis=-1)),
            0.0,
            1.0,
        )
        angle = xp.arccos(cosine)
        robust = 1.0 / xp.sqrt(1.0 + (angle / robust_scale) ** 4)
        weight = base_weight * robust * valid
        weight_sum = xp.maximum(xp.sum(weight, axis=1), 1.0e-12)
        smooth_tensor = xp.einsum(
            "nki,nkj,nk->nij",
            neighbor_normal,
            neighbor_normal,
            weight,
            optimize=True,
        ) / weight_sum[:, None, None]
        update_matrix = (
            data_strength[:, None, None] * tensor
            - resolved.smoothing_weight * smooth_tensor
        )
        _, candidate = _smallest_symmetric_eigenvector(update_matrix, xp)
        candidate_length = xp.sqrt(xp.sum(candidate * candidate, axis=1))
        candidate = xp.where(
            (candidate_length > 0.5)[:, None], candidate, normals
        )
        candidate = xp.where(
            (xp.sum(candidate * normals, axis=1) < 0.0)[:, None],
            -candidate,
            candidate,
        )
        updated = (
            (1.0 - resolved.damping) * normals
            + resolved.damping * candidate
        )
        updated /= xp.maximum(
            xp.sqrt(xp.sum(updated * updated, axis=1, keepdims=True)), 1.0e-12
        )
        change = xp.degrees(
            xp.arccos(
                xp.clip(xp.abs(xp.sum(updated * normals, axis=1)), 0.0, 1.0)
            )
        )
        if cp is not None:
            change_values = cp.asnumpy(change)
        else:
            change_values = np.asarray(change)
        iteration_records.append(
            {
                "iteration": iteration + 1,
                "medianChangeDegrees": round(float(np.median(change_values)), 6),
                "p90ChangeDegrees": round(
                    float(np.percentile(change_values, 90)), 6
                ),
                "maximumChangeDegrees": round(float(np.max(change_values)), 6),
            }
        )
        normals = updated
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()
        solved_normals = cp.asnumpy(normals).astype(np.float32)
    else:
        solved_normals = np.asarray(normals, dtype=np.float32)
    dominant = np.argmax(np.abs(solved_normals), axis=1)
    signs = np.where(
        solved_normals[np.arange(needles.count), dominant] < 0.0, -1.0, 1.0
    )
    solved_normals *= signs[:, None]
    continuous_finished = time.monotonic()
    candidate_normals, candidate_support, candidate_valid = (
        build_normal_hypothesis_bank(
            needles,
            graph,
            solved_normals,
            resolved,
        )
    )
    candidates_finished = time.monotonic()
    (
        clustered_normals,
        candidate_probability,
        candidate_label,
        mixture_records,
    ) = optimize_normal_hypothesis_mixture(
        needles.center_xyz,
        candidate_normals,
        candidate_support,
        candidate_valid,
        graph,
        resolved,
        compute_module=cp,
    )
    if cp is not None:
        cp.cuda.get_current_stream().synchronize()
        cp.get_default_memory_pool().free_all_blocks()
    mixture_finished = time.monotonic()
    continuous_metrics = _field_metrics(
        needles, graph, solved_normals, initial, confidence
    )
    clustered_metrics = _field_metrics(
        needles, graph, clustered_normals, initial, confidence
    )
    selected_support = candidate_support[
        np.arange(needles.count), candidate_label
    ]
    maximum_probability = np.max(candidate_probability, axis=1)
    entropy = -np.sum(
        candidate_probability
        * np.log(np.maximum(candidate_probability, 1.0e-12)),
        axis=1,
    )
    valid_count = np.sum(candidate_valid, axis=1)
    normalized_entropy = entropy / np.maximum(np.log(np.maximum(valid_count, 2)), 1.0e-12)
    clustered_metrics.update(
        {
            "selectedCandidateSupportFraction": _percentile_record(
                selected_support
            ),
            "maximumPosteriorProbability": _percentile_record(
                maximum_probability
            ),
            "normalizedPosteriorEntropy": _percentile_record(
                normalized_entropy
            ),
            "nodesWithMultipleNormalHypotheses": int(
                np.count_nonzero(valid_count >= 2)
            ),
        }
    )
    carrier_summary, carrier_arrays = analyze_needle_carriers(
        needles,
        graph,
        clustered_normals,
        resolved,
    )
    arrays = {
        "centerXYZ": needles.center_xyz.astype(np.float32, copy=False),
        "directionXYZ": needles.direction_xyz.astype(np.float32, copy=False),
        "score": needles.score.astype(np.float32, copy=False),
        "axialCoverage": needles.axial_coverage.astype(np.float32, copy=False),
        "supportScore": needles.support_score.astype(np.float32, copy=False),
        "normalXYZ": clustered_normals,
        "continuousNormalXYZ": solved_normals,
        "initialNormalXYZ": initial,
        "candidateNormalXYZ": candidate_normals,
        "candidateSupportFraction": candidate_support,
        "candidateValid": candidate_valid,
        "candidateProbability": candidate_probability,
        "candidateLabel": candidate_label,
        "localTensorEigenvalues": local_eigenvalues,
        "localTensorConfidence": confidence,
        "neighborIndex": graph.neighbor_index,
        "neighborDistanceVoxels": graph.neighbor_distance_voxels,
        "neighborWeight": graph.neighbor_weight,
        "neighborCount": graph.neighbor_count,
        **carrier_arrays,
    }
    summary = {
        "method": {
            "nodes": "every unique canonical Acus needle in the block",
            "directions": "needle directions and solved page normals are axial/unsigned",
            "dataTerm": (
                "weighted local needle-direction structure tensor; page normal "
                "minimizes squared projection onto observed fibers"
            ),
            "continuousCoupling": (
                "one robust projector-valued smoothness optimization over the "
                "entire spatial needle graph"
            ),
            "normalMixture": (
                "node-plus-neighbor fiber cross products generate distinct normal "
                "hypotheses; one block-global mean-field solve clusters their "
                "spatially varying posterior labels"
            ),
            "cells": "not used by fitting; retained only for later storage and meshing",
            "scope": (
                "normal-mode clustering stage; layered phase and individual sheet "
                "identity remain a subsequent block-global optimization"
            ),
        },
        "counts": {
            "needles": needles.count,
            "directedNeighborEdges": int(np.count_nonzero(graph.neighbor_index >= 0)),
            "validNormalHypotheses": int(np.count_nonzero(candidate_valid)),
            "needlesWithMultipleNormalHypotheses": int(
                np.count_nonzero(valid_count >= 2)
            ),
        },
        "neighborCount": _percentile_record(graph.neighbor_count),
        "continuousFieldMetrics": continuous_metrics,
        "clusteredFieldMetrics": clustered_metrics,
        "continuousIterations": iteration_records,
        "mixtureIterations": mixture_records,
        "carriers": carrier_summary,
        "compute": {
            "requested": resolved.compute,
            "backend": backend,
            "device": device,
            "fallbackReason": fallback_reason,
            "continuousSolveSeconds": round(
                continuous_finished - solve_started, 6
            ),
            "candidateGenerationSeconds": round(
                candidates_finished - continuous_finished, 6
            ),
            "mixtureSolveSeconds": round(
                mixture_finished - candidates_finished, 6
            ),
            "solveSeconds": round(mixture_finished - solve_started, 6),
        },
    }
    return summary, arrays


def run_block_needle_field(
    raw_roots: Iterable[str | Path],
    output_root: str | Path,
    *,
    world_start_xyz: Iterable[float],
    world_stop_xyz_exclusive: Iterable[float],
    settings: BlockNeedleFieldSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Load, graph, and optimize one immutable block-global needle field."""

    started = time.monotonic()
    resolved = settings or BlockNeedleFieldSettings()
    roots = tuple(Path(value).resolve() for value in raw_roots)
    output = Path(output_root).resolve()
    start = _float3(world_start_xyz, "world start")
    stop = _float3(world_stop_xyz_exclusive, "world stop")
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": BLOCK_NEEDLE_FIELD_SCHEMA,
        "version": BLOCK_NEEDLE_FIELD_VERSION,
        "rawRoots": [
            {
                "path": str(value),
                "pipelineSha256": sha256_file(value / "pipeline.json"),
            }
            for value in roots
        ],
        "worldBounds": {
            "startXYZ": list(start),
            "stopXYZExclusive": list(stop),
        },
        "settings": resolved.record(),
        "implementationSha256": {
            "needle_field.py": sha256_file(Path(__file__)),
            "raw_acus.py": sha256_file(module_root / "raw_acus.py"),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    manifest_path = output / f"{BLOCK_NEEDLE_FIELD_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("block needle-field output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    needles, source = load_canonical_block_needles(roots, start, stop)
    loaded = time.monotonic()
    graph = build_needle_neighbor_graph(needles, resolved)
    graphed = time.monotonic()
    summary, arrays = optimize_block_needle_field(
        needles, graph, settings=resolved
    )
    solved = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{BLOCK_NEEDLE_FIELD_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(data_path)
    payload = {
        "schema": BLOCK_NEEDLE_FIELD_SCHEMA,
        "version": BLOCK_NEEDLE_FIELD_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source,
        "settings": resolved.record(),
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "neighborGraph": round(graphed - loaded, 6),
            "optimizationAndMetrics": round(solved - graphed, 6),
            "writing": round(time.monotonic() - solved, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
