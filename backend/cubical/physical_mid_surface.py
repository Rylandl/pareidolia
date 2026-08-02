from __future__ import annotations

import heapq
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from . import contextual_profile_adoption
from .contracts import VolumeSource, VoxelBounds, atomic_json, canonical_json_hash, sha256_file
from .contextual_profile_adoption import ContextualProfileAdoptionSettings
from .material_surface_bridging import (
    MATERIAL_SURFACE_BRIDGING_SCHEMA,
    MATERIAL_SURFACE_BRIDGING_STEM,
)
from .material_surface_fixed_point import (
    MATERIAL_SURFACE_FIXED_POINT_SCHEMA,
    MATERIAL_SURFACE_FIXED_POINT_STEM,
)
from .material_surface_graph import (
    MATERIAL_SURFACE_GRAPH_SCHEMA,
    MATERIAL_SURFACE_GRAPH_STEM,
    MaterialSurfaceGraphSettings,
    write_material_surface_cross_sections,
)
from .material_surface_growth import (
    MATERIAL_SURFACE_GROWTH_SCHEMA,
    MATERIAL_SURFACE_GROWTH_STEM,
)
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA, PAIRED_SURFACE_BANK_STEM
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PAIRED_SURFACE_GROWTH_STEM,
)


PHYSICAL_MID_SURFACE_SCHEMA = "pareidolia.physical-mid-surface-catalog"
PHYSICAL_MID_SURFACE_VERSION = 1
PHYSICAL_MID_SURFACE_STEM = "physical-mid-surface-catalog-v1"


@dataclass(frozen=True, slots=True)
class PhysicalMidSurfaceSettings:
    adopt_contextual_profiles: bool = True
    maximum_contextual_endpoint_position_residual_sampling_steps: float = 0.75
    maximum_contextual_endpoint_normal_degrees: float = 15.0
    contextual_endpoint_normal_cost_scale_degrees: float = 15.0
    minimum_contextual_profile_evidence: float = 0.35
    contextual_thickness_tolerance_sampling_steps: float = 0.5
    maximum_endpoint_direction_degrees: float = 35.0
    maximum_opposing_boundary_normal_degrees: float = 35.0
    maximum_endpoint_tangent_residual_sampling_steps: float = 3.0
    maximum_reciprocal_boundary_offset_sampling_steps: float = 3.0
    maximum_reciprocal_boundary_normal_degrees: float = 35.0
    maximum_lower_correspondences_per_upper_surfel: int = 4
    thickness_prior_tolerance_sampling_steps: float = 0.5
    thickness_prior_lower_percentile: float = 1.0
    thickness_prior_upper_percentile: float = 99.0
    pairing_batch_size: int = 128
    maximum_local_profile_distance_sampling_steps: float = 3.0
    maximum_local_profile_normal_degrees: float = 35.0
    maximum_geodesic_profile_distance_sampling_steps: float = 16.0
    maximum_geodesic_profile_normal_degrees: float = 35.0
    maximum_one_sided_proxy_profile_distance_sampling_steps: float = 20.0
    local_thickness_tolerance_sampling_steps: float = 1.5
    require_local_profile_prior: bool = True
    maximum_neighbor_midpoint_distance_sampling_steps: float = 3.0
    maximum_neighbor_normal_degrees: float = 35.0
    maximum_neighbor_midpoint_height_sampling_steps: float = 1.5

    def __post_init__(self) -> None:
        for value in (
            self.maximum_endpoint_direction_degrees,
            self.maximum_opposing_boundary_normal_degrees,
            self.maximum_reciprocal_boundary_normal_degrees,
            self.maximum_neighbor_normal_degrees,
            self.maximum_local_profile_normal_degrees,
            self.maximum_geodesic_profile_normal_degrees,
            self.maximum_contextual_endpoint_normal_degrees,
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise ValueError("mid-surface angular gates must lie in (0, 90)")
        for value in (
            self.maximum_endpoint_tangent_residual_sampling_steps,
            self.maximum_reciprocal_boundary_offset_sampling_steps,
            self.thickness_prior_tolerance_sampling_steps,
            self.maximum_neighbor_midpoint_distance_sampling_steps,
            self.maximum_neighbor_midpoint_height_sampling_steps,
            self.maximum_local_profile_distance_sampling_steps,
            self.maximum_geodesic_profile_distance_sampling_steps,
            self.maximum_one_sided_proxy_profile_distance_sampling_steps,
            self.local_thickness_tolerance_sampling_steps,
            self.maximum_contextual_endpoint_position_residual_sampling_steps,
            self.contextual_endpoint_normal_cost_scale_degrees,
            self.contextual_thickness_tolerance_sampling_steps,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("mid-surface distance gates must be positive")
        if not (
            0.0
            <= self.thickness_prior_lower_percentile
            < self.thickness_prior_upper_percentile
            <= 100.0
        ):
            raise ValueError("mid-surface thickness percentiles are invalid")
        if self.pairing_batch_size < 1:
            raise ValueError("mid-surface pairing batch size must be positive")
        if self.maximum_lower_correspondences_per_upper_surfel < 1:
            raise ValueError("upper-surface correspondence capacity must be positive")
        if not math.isfinite(self.minimum_contextual_profile_evidence):
            raise ValueError("contextual profile evidence must be finite")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _resolve_surface(root: str | Path) -> Path:
    value = Path(root).resolve()
    candidates = (
        value,
        value / f"{MATERIAL_SURFACE_FIXED_POINT_STEM}.json",
        value / f"{MATERIAL_SURFACE_BRIDGING_STEM}.json",
        value / f"{MATERIAL_SURFACE_GROWTH_STEM}.json",
        value / f"{MATERIAL_SURFACE_GRAPH_STEM}.json",
    )
    for candidate in candidates:
        if not candidate.is_file():
            continue
        manifest = json.loads(candidate.read_text())
        if manifest.get("schema") != MATERIAL_SURFACE_FIXED_POINT_SCHEMA:
            return candidate
        final = manifest.get("finalSurface", {})
        final_path = Path(str(final.get("manifestPath", ""))).resolve()
        if not final_path.is_file() or sha256_file(final_path) != final.get(
            "manifestSha256"
        ):
            raise ValueError("fixed-point final surface is unavailable or changed")
        return final_path
    raise FileNotFoundError(f"material surface is unavailable at {value}")


def _load_npz(path: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"data hash differs from manifest for {path}")
    with np.load(data_path, allow_pickle=False) as stored:
        return {name: np.asarray(stored[name]) for name in stored.files}


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


def _normalized(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float64)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-12)


def local_profile_thickness_prior(
    position: np.ndarray,
    signed_normal: np.ndarray,
    physical_sheet_label: np.ndarray,
    physical_boundary_side: np.ndarray,
    profile_lower: np.ndarray,
    profile_upper: np.ndarray,
    profile_normal: np.ndarray,
    profile_thickness: np.ndarray,
    profile_label: np.ndarray,
    profile_candidate_index: np.ndarray,
    *,
    sampling_stride_voxels: int,
    settings: PhysicalMidSurfaceSettings,
) -> dict[str, np.ndarray]:
    """Transport directly measured profile thickness onto nearby face nodes."""

    point = np.asarray(position, dtype=np.float64)
    normal = _normalized(signed_normal)
    label = np.asarray(physical_sheet_label, dtype=np.int32)
    side = np.asarray(physical_boundary_side, dtype=np.uint8)
    lower = np.asarray(profile_lower, dtype=np.float64)
    upper = np.asarray(profile_upper, dtype=np.float64)
    profile_axis = _normalized(profile_normal)
    thickness = np.asarray(profile_thickness, dtype=np.float64)
    profile_sheet = np.asarray(profile_label, dtype=np.int32)
    profile_index = np.asarray(profile_candidate_index, dtype=np.int32)
    if any(
        len(value) != len(thickness)
        for value in (lower, upper, profile_axis, profile_sheet, profile_index)
    ):
        raise ValueError("local thickness profile arrays are not aligned")
    expected = np.full(len(point), np.nan, dtype=np.float32)
    distance_result = np.full(len(point), np.nan, dtype=np.float32)
    candidate_result = np.full(len(point), -1, dtype=np.int32)
    maximum_distance = (
        settings.maximum_local_profile_distance_sampling_steps
        * float(sampling_stride_voxels)
    )
    cosine_limit = math.cos(
        math.radians(settings.maximum_local_profile_normal_degrees)
    )
    stride = float(sampling_stride_voxels)
    for sheet_label in np.unique(label[label >= 0]):
        profile_member = np.flatnonzero(profile_sheet == sheet_label)
        if not len(profile_member):
            continue
        for boundary_side in (0, 1):
            node = np.flatnonzero(
                (label == sheet_label) & (side == boundary_side)
            )
            if not len(node):
                continue
            boundary = lower if boundary_side == 0 else upper
            boundary_normal = profile_axis if boundary_side == 0 else -profile_axis
            profile_point = boundary[profile_member]
            profile_direction = boundary_normal[profile_member]
            best_cost = np.full(len(node), np.inf, dtype=np.float64)
            best_profile = np.full(len(node), -1, dtype=np.int32)
            best_distance = np.full(len(node), np.inf, dtype=np.float64)
            for start in range(0, len(node), settings.pairing_batch_size):
                stop = min(start + settings.pairing_batch_size, len(node))
                batch = node[start:stop]
                delta = profile_point[None, :, :] - point[batch, None, :]
                distance = np.linalg.norm(delta, axis=2)
                cosine = normal[batch] @ profile_direction.T
                valid = (distance <= maximum_distance) & (cosine >= cosine_limit)
                angular_scale = max(1.0 - cosine_limit, 1.0e-9)
                cost = distance / stride + 0.5 * (1.0 - cosine) / angular_scale
                cost[~valid] = np.inf
                choice = np.argmin(cost, axis=1)
                value = cost[np.arange(len(batch)), choice]
                finite = np.isfinite(value)
                local_row = np.flatnonzero(finite)
                best_cost[start + local_row] = value[finite]
                best_profile[start + local_row] = choice[finite]
                best_distance[start + local_row] = distance[
                    local_row, choice[finite]
                ]
            valid_node = best_profile >= 0
            selected_node = node[valid_node]
            selected_profile = profile_member[best_profile[valid_node]]
            expected[selected_node] = thickness[selected_profile].astype(np.float32)
            distance_result[selected_node] = (
                best_distance[valid_node] / stride
            ).astype(np.float32)
            candidate_result[selected_node] = profile_index[selected_profile]
    return {
        "expectedThicknessVoxels": expected,
        "profileDistanceSamplingSteps": distance_result,
        "profileCandidateIndex": candidate_result,
    }


def propagate_profile_thickness_prior(
    position: np.ndarray,
    signed_normal: np.ndarray,
    physical_sheet_label: np.ndarray,
    physical_boundary_side: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    direct_prior: Mapping[str, np.ndarray],
    *,
    sampling_stride_voxels: int,
    settings: PhysicalMidSurfaceSettings,
) -> dict[str, np.ndarray]:
    """Transport measured thickness along one validated physical boundary.

    Euclidean nearest profiles can jump between adjacent fold branches. This
    multi-source geodesic propagation traverses only persisted graph edges with
    the same immutable sheet and boundary-side identity. The path length is
    bounded, and pairing still has to pass the original two-sided thickness,
    direction, normal, tangent, and mutual-best gates.
    """

    point = np.asarray(position, dtype=np.float64)
    normal = _normalized(signed_normal)
    label = np.asarray(physical_sheet_label, dtype=np.int32)
    side = np.asarray(physical_boundary_side, dtype=np.uint8)
    first = np.asarray(edge_first, dtype=np.int32)
    second = np.asarray(edge_second, dtype=np.int32)
    expected_seed = np.asarray(
        direct_prior["expectedThicknessVoxels"], dtype=np.float64
    )
    direct_distance = np.asarray(
        direct_prior["profileDistanceSamplingSteps"], dtype=np.float64
    )
    candidate_seed = np.asarray(
        direct_prior["profileCandidateIndex"], dtype=np.int32
    )
    if point.shape != normal.shape or point.ndim != 2 or point.shape[1] != 3:
        raise ValueError("geodesic prior positions and normals must have shape (N, 3)")
    if any(len(value) != len(point) for value in (label, side, expected_seed, direct_distance, candidate_seed)):
        raise ValueError("geodesic profile-prior node arrays are not aligned")
    if len(first) != len(second):
        raise ValueError("geodesic profile-prior edge arrays are not aligned")
    if np.any(first < 0) or np.any(second >= len(point)):
        raise ValueError("geodesic profile-prior edge leaves the node table")

    cosine_limit = math.cos(
        math.radians(settings.maximum_geodesic_profile_normal_degrees)
    )
    edge_cosine = np.einsum("ij,ij->i", normal[first], normal[second])
    valid_edge = (
        (label[first] >= 0)
        & (label[first] == label[second])
        & (side[first] <= 1)
        & (side[first] == side[second])
        & (edge_cosine >= cosine_limit)
    )
    first = first[valid_edge]
    second = second[valid_edge]
    weight = (
        np.linalg.norm(point[second] - point[first], axis=1)
        / float(sampling_stride_voxels)
    )
    directed_source = np.concatenate((first, second))
    directed_target = np.concatenate((second, first))
    directed_weight = np.concatenate((weight, weight))
    order = np.argsort(directed_source, kind="stable")
    directed_source = directed_source[order]
    directed_target = directed_target[order]
    directed_weight = directed_weight[order]
    degree = np.bincount(directed_source, minlength=len(point))
    offset = np.zeros(len(point) + 1, dtype=np.int64)
    np.cumsum(degree, out=offset[1:])

    seed = np.flatnonzero(
        np.isfinite(expected_seed)
        & (expected_seed > 0.0)
        & (candidate_seed >= 0)
    )
    path_distance = np.full(len(point), np.inf, dtype=np.float64)
    source_seed = np.full(len(point), -1, dtype=np.int32)
    path_distance[seed] = 0.0
    source_seed[seed] = seed.astype(np.int32)
    queue = [(0.0, int(node), int(node)) for node in seed]
    heapq.heapify(queue)
    maximum_distance = settings.maximum_geodesic_profile_distance_sampling_steps
    while queue:
        distance, node, source = heapq.heappop(queue)
        if distance > maximum_distance:
            break
        if distance > path_distance[node] + 1.0e-10 or source != source_seed[node]:
            continue
        for edge_index in range(int(offset[node]), int(offset[node + 1])):
            neighbor = int(directed_target[edge_index])
            candidate_distance = distance + float(directed_weight[edge_index])
            if candidate_distance > maximum_distance:
                continue
            prior_distance = path_distance[neighbor]
            improve = candidate_distance < prior_distance - 1.0e-10
            tie = (
                abs(candidate_distance - prior_distance) <= 1.0e-10
                and (
                    source_seed[neighbor] < 0
                    or candidate_seed[source]
                    < candidate_seed[source_seed[neighbor]]
                )
            )
            if not improve and not tie:
                continue
            path_distance[neighbor] = candidate_distance
            source_seed[neighbor] = source
            heapq.heappush(queue, (candidate_distance, neighbor, source))

    reached = source_seed >= 0
    expected = expected_seed.copy()
    candidate = candidate_seed.copy()
    total_distance = direct_distance.copy()
    propagated = reached & ~np.isfinite(expected_seed)
    expected[propagated] = expected_seed[source_seed[propagated]]
    candidate[propagated] = candidate_seed[source_seed[propagated]]
    total_distance[propagated] = (
        direct_distance[source_seed[propagated]] + path_distance[propagated]
    )
    return {
        "expectedThicknessVoxels": expected.astype(np.float32),
        "profileDistanceSamplingSteps": total_distance.astype(np.float32),
        "profileCandidateIndex": candidate.astype(np.int32),
        "profileGeodesicDistanceSamplingSteps": path_distance.astype(np.float32),
        "directProfilePrior": np.isfinite(expected_seed),
        "geodesicallyPropagatedProfilePrior": propagated,
    }


def one_sided_mid_surface_proxies(
    position: np.ndarray,
    signed_normal: np.ndarray,
    sheet_normal: np.ndarray,
    physical_sheet_label: np.ndarray,
    physical_boundary_side: np.ndarray,
    expected_thickness_voxels: np.ndarray,
    profile_distance_sampling_steps: np.ndarray,
    used_by_dense_pair: np.ndarray,
    *,
    settings: PhysicalMidSurfaceSettings,
    source_start_xyz: np.ndarray | None = None,
    source_stop_xyz_exclusive: np.ndarray | None = None,
) -> dict[str, np.ndarray]:
    """Offset a measured physical face by half a locally measured thickness.

    The dense material-surface graph already represents one signed physical
    boundary. Its smoother sheet normal is sign-aligned to the air-to-material
    normal, so the offset always points into papyrus. A face used by an actual
    lower/upper correspondence never receives a proxy. The construction does
    not connect opposite boundary sides; continuity remains inherited from one
    immutable signed-face graph until direct profiles or dense pairs join it.
    """

    point = np.asarray(position, dtype=np.float64)
    raw_normal = _normalized(signed_normal)
    smooth_normal = _normalized(sheet_normal)
    label = np.asarray(physical_sheet_label, dtype=np.int32)
    side = np.asarray(physical_boundary_side, dtype=np.uint8)
    thickness = np.asarray(expected_thickness_voxels, dtype=np.float64)
    profile_distance = np.asarray(
        profile_distance_sampling_steps, dtype=np.float64
    )
    dense = np.asarray(used_by_dense_pair, dtype=bool)
    if point.ndim != 2 or point.shape[1] != 3:
        raise ValueError("one-sided proxy positions must have shape (N, 3)")
    if raw_normal.shape != point.shape or smooth_normal.shape != point.shape:
        raise ValueError("one-sided proxy normals must match positions")
    if any(
        len(value) != len(point)
        for value in (label, side, thickness, profile_distance, dense)
    ):
        raise ValueError("one-sided proxy arrays are not node aligned")
    inward_sign = np.where(
        np.einsum("ij,ij->i", smooth_normal, raw_normal) >= 0.0,
        1.0,
        -1.0,
    )
    inward = smooth_normal * inward_sign[:, None]
    inferred_opposite_all = point + thickness[:, None] * inward
    selected = (
        (label >= 0)
        & (side <= 1)
        & ~dense
        & np.isfinite(thickness)
        & (thickness > 0.0)
        & np.isfinite(profile_distance)
        & (
            profile_distance
            <= settings.maximum_one_sided_proxy_profile_distance_sampling_steps
        )
    )
    if (source_start_xyz is None) != (source_stop_xyz_exclusive is None):
        raise ValueError("one-sided proxy source bounds must be supplied together")
    if source_start_xyz is not None and source_stop_xyz_exclusive is not None:
        source_start = np.asarray(source_start_xyz, dtype=np.float64)
        source_stop = np.asarray(source_stop_xyz_exclusive, dtype=np.float64)
        if (
            source_start.shape != (3,)
            or source_stop.shape != (3,)
            or np.any(source_stop <= source_start)
        ):
            raise ValueError("one-sided proxy source bounds are invalid")
        selected &= np.all(
            (inferred_opposite_all >= source_start[None, :])
            & (inferred_opposite_all < source_stop[None, :]),
            axis=1,
        )
    source_node = np.flatnonzero(selected).astype(np.int32)
    selected_point = point[selected]
    selected_inward = inward[selected]
    selected_thickness = thickness[selected]
    midpoint = selected_point + 0.5 * selected_thickness[:, None] * selected_inward
    inferred_opposite = inferred_opposite_all[selected]
    selected_side = side[selected]
    lower = np.where(
        (selected_side == 0)[:, None], selected_point, inferred_opposite
    )
    upper = np.where(
        (selected_side == 1)[:, None], selected_point, inferred_opposite
    )
    return {
        "sourceSurfaceNode": source_node,
        "midpointXYZ": midpoint.astype(np.float32),
        "normalXYZ": selected_inward.astype(np.float32),
        "boundaryLowerXYZ": lower.astype(np.float32),
        "boundaryUpperXYZ": upper.astype(np.float32),
        "physicalSheetLabel": label[selected].astype(np.int32),
        "physicalBoundarySide": selected_side.astype(np.uint8),
        "thicknessVoxels": selected_thickness.astype(np.float32),
        "profileDistanceSamplingSteps": profile_distance[selected].astype(
            np.float32
        ),
    }


def pair_physical_boundary_faces(
    position: np.ndarray,
    signed_normal: np.ndarray,
    physical_sheet_label: np.ndarray,
    physical_boundary_side: np.ndarray,
    thickness_by_label: Mapping[int, np.ndarray],
    *,
    sampling_stride_voxels: int,
    settings: PhysicalMidSurfaceSettings,
    local_thickness_prior: np.ndarray | None = None,
) -> dict[str, Any]:
    """Pair lower/upper signed faces using reciprocal local surface support.

    The sheet label limits correspondence to one paired air-material-air
    hypothesis. Direction, opposing normals, tangent residual, and the
    per-sheet measured thickness distribution then define the only admissible
    lower/upper matches. A lower face is retained when the upper face's best
    lower match lies on the same small surface patch. Bounded upper-face
    capacity preserves staggered sampling without allowing broad many-to-one
    collapses.
    """

    point = np.asarray(position, dtype=np.float64)
    normal = _normalized(signed_normal)
    label = np.asarray(physical_sheet_label, dtype=np.int32)
    side = np.asarray(physical_boundary_side, dtype=np.uint8)
    if point.ndim != 2 or point.shape[1] != 3 or normal.shape != point.shape:
        raise ValueError("boundary positions and normals must have shape (node, 3)")
    if len(label) != len(point) or len(side) != len(point):
        raise ValueError("boundary identities are not node aligned")
    if np.any((label >= 0) != (side <= 1)):
        raise ValueError("physical sheet and boundary-side identities disagree")
    local_thickness = (
        np.asarray(local_thickness_prior, dtype=np.float64)
        if local_thickness_prior is not None
        else np.full(len(point), np.nan, dtype=np.float64)
    )
    if len(local_thickness) != len(point):
        raise ValueError("local thickness prior is not node aligned")
    stride = float(sampling_stride_voxels)
    direction_cosine = math.cos(
        math.radians(settings.maximum_endpoint_direction_degrees)
    )
    opposition_cosine = math.cos(
        math.radians(settings.maximum_opposing_boundary_normal_degrees)
    )
    tangent_cap = settings.maximum_endpoint_tangent_residual_sampling_steps * stride
    reciprocal_offset_cap = (
        settings.maximum_reciprocal_boundary_offset_sampling_steps * stride
    )
    reciprocal_normal_cosine = math.cos(
        math.radians(settings.maximum_reciprocal_boundary_normal_degrees)
    )
    tolerance = settings.thickness_prior_tolerance_sampling_steps * stride

    lower_result: list[np.ndarray] = []
    upper_result: list[np.ndarray] = []
    cost_result: list[np.ndarray] = []
    tangent_result: list[np.ndarray] = []
    lower_angle_result: list[np.ndarray] = []
    upper_angle_result: list[np.ndarray] = []
    opposition_result: list[np.ndarray] = []
    thickness_result: list[np.ndarray] = []
    stage_names = (
        "sameSheetHasOppositeSide",
        "globalThicknessRange",
        "localThicknessAgreement",
        "endpointDirection",
        "opposingBoundaryNormal",
        "tangentResidual",
        "reciprocalSurfelSupport",
        "upperSurfelCapacity",
    )
    stage_count = {name: 0 for name in stage_names}
    total_lower_nodes = 0
    maximum_one_to_one_pairs = 0
    labels_with_both_sides = 0
    labels_with_thickness_prior = 0
    exact_mutual_count = 0
    for sheet_label in np.unique(label[label >= 0]):
        lower = np.flatnonzero((label == sheet_label) & (side == 0))
        upper = np.flatnonzero((label == sheet_label) & (side == 1))
        total_lower_nodes += len(lower)
        maximum_one_to_one_pairs += min(len(lower), len(upper))
        if len(lower) and len(upper):
            labels_with_both_sides += 1
            stage_count["sameSheetHasOppositeSide"] += len(lower)
        prior = np.asarray(thickness_by_label.get(int(sheet_label), ()), dtype=np.float64)
        prior = prior[np.isfinite(prior) & (prior > 0.0)]
        if not len(lower) or not len(upper) or not len(prior):
            continue
        labels_with_thickness_prior += 1
        lower_thickness, center_thickness, upper_thickness = np.percentile(
            prior,
            (
                settings.thickness_prior_lower_percentile,
                50.0,
                settings.thickness_prior_upper_percentile,
            ),
        )
        lower_thickness = max(0.0, float(lower_thickness) - tolerance)
        upper_thickness = float(upper_thickness) + tolerance
        thickness_scale = max(
            0.5 * (upper_thickness - lower_thickness), stride
        )
        best_upper = np.full(len(lower), -1, dtype=np.int32)
        best_upper_cost = np.full(len(lower), np.inf, dtype=np.float64)
        best_lower = np.full(len(upper), -1, dtype=np.int32)
        best_lower_cost = np.full(len(upper), np.inf, dtype=np.float64)
        reached = {
            name: np.zeros(len(lower), dtype=bool)
            for name in stage_names[1:-2]
        }
        upper_point = point[upper]
        upper_normal = normal[upper]
        for start in range(0, len(lower), settings.pairing_batch_size):
            stop = min(start + settings.pairing_batch_size, len(lower))
            batch = lower[start:stop]
            delta = upper_point[None, :, :] - point[batch, None, :]
            distance_squared = np.einsum("buj,buj->bu", delta, delta)
            distance = np.sqrt(np.maximum(distance_squared, 1.0e-18))
            lower_height = np.einsum(
                "buj,bj->bu", delta, normal[batch]
            )
            upper_height = np.einsum(
                "buj,uj->bu", delta, -upper_normal
            )
            lower_direction = lower_height / distance
            upper_direction = upper_height / distance
            lower_tangent = np.sqrt(
                np.maximum(distance_squared - lower_height * lower_height, 0.0)
            )
            upper_tangent = np.sqrt(
                np.maximum(distance_squared - upper_height * upper_height, 0.0)
            )
            tangent = np.maximum(lower_tangent, upper_tangent)
            opposing = normal[batch] @ upper_normal.T
            valid = (distance >= lower_thickness) & (distance <= upper_thickness)
            reached["globalThicknessRange"][start:stop] |= np.any(valid, axis=1)
            lower_local_thickness = local_thickness[batch]
            upper_local_thickness = local_thickness[upper]
            local_available = np.isfinite(lower_local_thickness)[:, None] & np.isfinite(
                upper_local_thickness
            )[None, :]
            local_center = 0.5 * (
                lower_local_thickness[:, None] + upper_local_thickness[None, :]
            )
            local_tolerance = (
                settings.local_thickness_tolerance_sampling_steps * stride
            )
            local_agreement = (
                np.abs(distance - lower_local_thickness[:, None]) <= local_tolerance
            ) & (
                np.abs(distance - upper_local_thickness[None, :]) <= local_tolerance
            )
            if settings.require_local_profile_prior and local_thickness_prior is not None:
                valid &= local_available & local_agreement
            reached["localThicknessAgreement"][start:stop] |= np.any(
                valid, axis=1
            )
            valid &= (
                (lower_direction >= direction_cosine)
                & (upper_direction >= direction_cosine)
            )
            reached["endpointDirection"][start:stop] |= np.any(valid, axis=1)
            valid &= opposing <= -opposition_cosine
            reached["opposingBoundaryNormal"][start:stop] |= np.any(
                valid, axis=1
            )
            valid &= tangent <= tangent_cap
            reached["tangentResidual"][start:stop] |= np.any(valid, axis=1)
            angular_scale = max(1.0 - direction_cosine, 1.0e-9)
            opposition_scale = max(1.0 - opposition_cosine, 1.0e-9)
            cost = (
                tangent / stride
                + 0.25 * (1.0 - lower_direction) / angular_scale
                + 0.25 * (1.0 - upper_direction) / angular_scale
                + 0.25 * (1.0 + opposing) / opposition_scale
                + 0.15
                * np.abs(distance - center_thickness)
                / thickness_scale
            )
            if local_thickness_prior is not None:
                local_residual = np.divide(
                    np.abs(distance - local_center),
                    max(local_tolerance, 1.0e-9),
                    out=np.zeros_like(distance),
                    where=local_available,
                )
                cost += np.where(local_available, 0.75 * local_residual, 0.75)
            cost[~valid] = np.inf
            row_choice = np.argmin(cost, axis=1)
            row_cost = cost[np.arange(len(batch)), row_choice]
            row_valid = np.isfinite(row_cost)
            row_index = start + np.flatnonzero(row_valid)
            best_upper[row_index] = row_choice[row_valid]
            best_upper_cost[row_index] = row_cost[row_valid]
            column_choice = np.argmin(cost, axis=0)
            column_cost = cost[column_choice, np.arange(len(upper))]
            improve = column_cost < best_lower_cost
            best_lower[improve] = start + column_choice[improve]
            best_lower_cost[improve] = column_cost[improve]
        lower_local = np.flatnonzero(best_upper >= 0)
        upper_local = best_upper[lower_local]
        reciprocal_lower = best_lower[upper_local]
        reciprocal_available = reciprocal_lower >= 0
        exact_mutual = reciprocal_available & (reciprocal_lower == lower_local)
        exact_mutual_count += int(np.count_nonzero(exact_mutual))
        reciprocal_offset = np.full(len(lower_local), np.inf, dtype=np.float64)
        reciprocal_cosine = np.full(len(lower_local), -1.0, dtype=np.float64)
        valid_reciprocal = np.flatnonzero(reciprocal_available)
        if len(valid_reciprocal):
            current_lower_node = lower[lower_local[valid_reciprocal]]
            reciprocal_lower_node = lower[
                reciprocal_lower[valid_reciprocal]
            ]
            reciprocal_offset[valid_reciprocal] = np.linalg.norm(
                point[current_lower_node] - point[reciprocal_lower_node],
                axis=1,
            )
            reciprocal_cosine[valid_reciprocal] = np.einsum(
                "ij,ij->i",
                normal[current_lower_node],
                normal[reciprocal_lower_node],
            )
        reciprocal = (
            reciprocal_available
            & (reciprocal_offset <= reciprocal_offset_cap)
            & (reciprocal_cosine >= reciprocal_normal_cosine)
        )
        lower_local = lower_local[reciprocal]
        upper_local = upper_local[reciprocal]
        for name, values in reached.items():
            stage_count[name] += int(np.count_nonzero(values))
        stage_count["reciprocalSurfelSupport"] += len(lower_local)
        if len(lower_local):
            reciprocal_exact = exact_mutual[reciprocal]
            reciprocal_cost = best_upper_cost[lower_local]
            keep_capacity = np.zeros(len(lower_local), dtype=bool)
            for upper_value in np.unique(upper_local):
                member = np.flatnonzero(upper_local == upper_value)
                ranking = np.lexsort(
                    (
                        lower_local[member],
                        reciprocal_cost[member],
                        ~reciprocal_exact[member],
                    )
                )
                retained = member[
                    ranking[
                        : settings.maximum_lower_correspondences_per_upper_surfel
                    ]
                ]
                keep_capacity[retained] = True
            lower_local = lower_local[keep_capacity]
            upper_local = upper_local[keep_capacity]
        stage_count["upperSurfelCapacity"] += len(lower_local)
        if not len(lower_local):
            continue
        lower_node = lower[lower_local]
        upper_node = upper[upper_local]
        delta = point[upper_node] - point[lower_node]
        distance = np.linalg.norm(delta, axis=1)
        direction = delta / np.maximum(distance[:, None], 1.0e-12)
        lower_cosine = np.einsum("ij,ij->i", direction, normal[lower_node])
        upper_cosine = np.einsum("ij,ij->i", direction, -normal[upper_node])
        opposing = np.einsum("ij,ij->i", normal[lower_node], normal[upper_node])
        lower_tangent = np.sqrt(
            np.maximum(
                distance * distance
                - np.einsum("ij,ij->i", delta, normal[lower_node]) ** 2,
                0.0,
            )
        )
        upper_tangent = np.sqrt(
            np.maximum(
                distance * distance
                - np.einsum("ij,ij->i", delta, -normal[upper_node]) ** 2,
                0.0,
            )
        )
        lower_result.append(lower_node.astype(np.int32))
        upper_result.append(upper_node.astype(np.int32))
        cost_result.append(best_upper_cost[lower_local].astype(np.float32))
        tangent_result.append(
            (np.maximum(lower_tangent, upper_tangent) / stride).astype(np.float32)
        )
        lower_angle_result.append(
            np.degrees(np.arccos(np.clip(lower_cosine, -1.0, 1.0))).astype(
                np.float32
            )
        )
        upper_angle_result.append(
            np.degrees(np.arccos(np.clip(upper_cosine, -1.0, 1.0))).astype(
                np.float32
            )
        )
        opposition_result.append(
            np.degrees(np.arccos(np.clip(-opposing, -1.0, 1.0))).astype(
                np.float32
            )
        )
        thickness_result.append(distance.astype(np.float32))
    empty_int = np.empty(0, dtype=np.int32)
    empty_float = np.empty(0, dtype=np.float32)
    stage_loss: dict[str, int] = {}
    previous = total_lower_nodes
    for name in stage_names:
        value = stage_count[name]
        stage_loss[name] = previous - value
        previous = value
    return {
        "lowerNode": np.concatenate(lower_result) if lower_result else empty_int,
        "upperNode": np.concatenate(upper_result) if upper_result else empty_int,
        "pairCost": np.concatenate(cost_result) if cost_result else empty_float,
        "tangentResidualSamplingSteps": (
            np.concatenate(tangent_result) if tangent_result else empty_float
        ),
        "lowerDirectionDegrees": (
            np.concatenate(lower_angle_result) if lower_angle_result else empty_float
        ),
        "upperDirectionDegrees": (
            np.concatenate(upper_angle_result) if upper_angle_result else empty_float
        ),
        "opposingNormalDegrees": (
            np.concatenate(opposition_result) if opposition_result else empty_float
        ),
        "thicknessVoxels": (
            np.concatenate(thickness_result) if thickness_result else empty_float
        ),
        "census": {
            "totalLowerBoundaryNodeCount": int(total_lower_nodes),
            "maximumOneToOnePairCount": int(maximum_one_to_one_pairs),
            "labelsWithBothBoundarySides": int(labels_with_both_sides),
            "labelsWithThicknessPrior": int(labels_with_thickness_prior),
            "exactMutualOneToOneCount": int(exact_mutual_count),
            "lowerNodesReachingStage": {
                name: int(stage_count[name]) for name in stage_names
            },
            "lowerNodesLostAtStage": stage_loss,
        },
    }


def _midpoint_edges(
    pair_by_surface_node: np.ndarray,
    surface_edge_first: np.ndarray,
    surface_edge_second: np.ndarray,
    surface_side: np.ndarray,
    midpoint: np.ndarray,
    normal: np.ndarray,
    label: np.ndarray,
    *,
    sampling_stride_voxels: int,
    settings: PhysicalMidSurfaceSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    first_pair = pair_by_surface_node[surface_edge_first]
    second_pair = pair_by_surface_node[surface_edge_second]
    valid = (first_pair >= 0) & (second_pair >= 0) & (first_pair != second_pair)
    first_pair = first_pair[valid]
    second_pair = second_pair[valid]
    edge_side = surface_side[surface_edge_first[valid]]
    valid_side = (edge_side <= 1) & (
        edge_side == surface_side[surface_edge_second[valid]]
    )
    first_pair = first_pair[valid_side]
    second_pair = second_pair[valid_side]
    edge_side = edge_side[valid_side]
    if not len(first_pair):
        empty_int = np.empty(0, dtype=np.int32)
        return (
            empty_int,
            empty_int,
            np.empty(0, dtype=np.uint8),
            np.empty(0, dtype=np.float32),
            {
                "surfaceEdgeProjectionCount": 0,
                "uniqueProjectedPairEdgeCount": 0,
                "retainedMidSurfaceEdgeCount": 0,
                "rejectedMidSurfaceGeometryCount": 0,
            },
        )
    low = np.minimum(first_pair, second_pair).astype(np.int64)
    high = np.maximum(first_pair, second_pair).astype(np.int64)
    key = low * max(len(midpoint), 1) + high
    order = np.argsort(key, kind="stable")
    key = key[order]
    low = low[order]
    high = high[order]
    bit = np.left_shift(np.uint8(1), edge_side[order]).astype(np.uint8)
    start = np.flatnonzero(
        np.concatenate((np.ones(1, dtype=bool), key[1:] != key[:-1]))
    )
    first = low[start].astype(np.int32)
    second = high[start].astype(np.int32)
    support = np.bitwise_or.reduceat(bit, start).astype(np.uint8)
    delta = midpoint[second] - midpoint[first]
    distance = np.linalg.norm(delta, axis=1)
    dot = np.einsum("ij,ij->i", normal[first], normal[second])
    sign = np.where(dot >= 0.0, 1.0, -1.0)
    common = normal[first] + sign[:, None] * normal[second]
    common = _normalized(common)
    height = np.abs(np.einsum("ij,ij->i", delta, common))
    stride = float(sampling_stride_voxels)
    retained = (
        (label[first] == label[second])
        & (distance <= settings.maximum_neighbor_midpoint_distance_sampling_steps * stride)
        & (
            np.abs(dot)
            >= math.cos(math.radians(settings.maximum_neighbor_normal_degrees))
        )
        & (
            height
            <= settings.maximum_neighbor_midpoint_height_sampling_steps * stride
        )
    )
    score = (
        np.exp(-0.5 * (height / max(stride, 1.0e-9)) ** 2)
        * np.where(support == 3, 1.0, 0.75)
    ).astype(np.float32)
    return (
        first[retained],
        second[retained],
        support[retained],
        score[retained],
        {
            "surfaceEdgeProjectionCount": int(np.count_nonzero(valid)),
            "uniqueProjectedPairEdgeCount": int(len(first)),
            "retainedMidSurfaceEdgeCount": int(np.count_nonzero(retained)),
            "rejectedMidSurfaceGeometryCount": int(np.count_nonzero(~retained)),
        },
    )


def _components(
    node_count: int,
    first: np.ndarray,
    second: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    for first_node, second_node in zip(first, second):
        first_root = find(int(first_node))
        second_root = find(int(second_node))
        if first_root == second_root:
            continue
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
    root = np.asarray([find(value) for value in range(node_count)], dtype=np.int32)
    root_value, root_count = np.unique(root, return_counts=True)
    order = np.lexsort((root_value, -root_count))
    rank = np.full(node_count, -1, dtype=np.int32)
    rank[root_value[order]] = np.arange(len(root_value), dtype=np.int32)
    return rank[root], root_count[order]


def _profile_attachment_edges(
    dense_midpoint: np.ndarray,
    dense_normal: np.ndarray,
    dense_label: np.ndarray,
    lower_profile_candidate: np.ndarray,
    upper_profile_candidate: np.ndarray,
    candidate_to_profile_node: np.ndarray,
    profile_midpoint: np.ndarray,
    profile_normal: np.ndarray,
    profile_label: np.ndarray,
    *,
    sampling_stride_voxels: int,
    settings: PhysicalMidSurfaceSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    dense_count = len(dense_midpoint)
    dense_index = np.concatenate(
        (np.arange(dense_count, dtype=np.int32), np.arange(dense_count, dtype=np.int32))
    )
    candidate = np.concatenate(
        (
            np.asarray(lower_profile_candidate, dtype=np.int32),
            np.asarray(upper_profile_candidate, dtype=np.int32),
        )
    )
    valid = (candidate >= 0) & (candidate < len(candidate_to_profile_node))
    profile_node = np.full(len(candidate), -1, dtype=np.int32)
    profile_node[valid] = candidate_to_profile_node[candidate[valid]]
    valid &= profile_node >= 0
    dense_index = dense_index[valid]
    profile_node = profile_node[valid]
    if not len(dense_index):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    key = profile_node.astype(np.int64) * max(dense_count, 1) + dense_index
    order = np.argsort(key, kind="stable")
    unique = np.concatenate(
        (
            np.ones(1, dtype=bool),
            key[order][1:] != key[order][:-1],
        )
    )
    dense_index = dense_index[order][unique]
    profile_node = profile_node[order][unique]
    if not len(dense_index):
        return (
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.int32),
            np.empty(0, dtype=np.float32),
        )
    delta = dense_midpoint[dense_index] - profile_midpoint[profile_node]
    distance = np.linalg.norm(delta, axis=1)
    dot = np.einsum(
        "ij,ij->i", dense_normal[dense_index], profile_normal[profile_node]
    )
    sign = np.where(dot >= 0.0, 1.0, -1.0)
    common = _normalized(
        dense_normal[dense_index] + sign[:, None] * profile_normal[profile_node]
    )
    height = np.abs(np.einsum("ij,ij->i", delta, common))
    stride = float(sampling_stride_voxels)
    retained = (
        (dense_label[dense_index] == profile_label[profile_node])
        & (
            distance
            <= settings.maximum_neighbor_midpoint_distance_sampling_steps * stride
        )
        & (
            np.abs(dot)
            >= math.cos(math.radians(settings.maximum_neighbor_normal_degrees))
        )
        & (
            height
            <= settings.maximum_neighbor_midpoint_height_sampling_steps * stride
        )
    )
    score = np.exp(-0.5 * (height[retained] / max(stride, 1.0e-9)) ** 2)
    return (
        profile_node[retained].astype(np.int32),
        dense_index[retained].astype(np.int32),
        score.astype(np.float32),
    )


def run_physical_mid_surface_catalog(
    paired_bank_root: str | Path,
    paired_growth_root: str | Path,
    material_surface_root: str | Path,
    output_path: str | Path,
    *,
    settings: PhysicalMidSurfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    bank_path = _resolve(paired_bank_root, PAIRED_SURFACE_BANK_STEM)
    growth_path = _resolve(paired_growth_root, PAIRED_SURFACE_GROWTH_STEM)
    surface_path = _resolve_surface(material_surface_root)
    bank = json.loads(bank_path.read_text())
    growth = json.loads(growth_path.read_text())
    surface = json.loads(surface_path.read_text())
    if bank.get("schema") != PAIRED_SURFACE_BANK_SCHEMA or bank.get("state") != "complete":
        raise ValueError("mid-surface catalog requires a complete paired bank")
    if growth.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA or growth.get("state") != "complete":
        raise ValueError("mid-surface catalog requires complete paired growth")
    if surface.get("schema") not in {
        MATERIAL_SURFACE_GRAPH_SCHEMA,
        MATERIAL_SURFACE_GROWTH_SCHEMA,
        MATERIAL_SURFACE_BRIDGING_SCHEMA,
    } or surface.get("state") != "complete":
        raise ValueError("mid-surface catalog requires a complete side-aware surface")
    candidate_identity = growth["identity"]["candidateBank"]
    if (
        candidate_identity["manifestSha256"] != sha256_file(bank_path)
        or candidate_identity["dataSha256"] != bank["data"]["sha256"]
    ):
        raise ValueError("paired growth and bank identities disagree")
    if bank.get("source") != surface.get("source") or bank.get("geometry") != surface.get("geometry"):
        raise ValueError("paired profiles and material surfaces must share geometry")
    resolved = settings or PhysicalMidSurfaceSettings()
    identity: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_VERSION,
        "pairedBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank["data"]["sha256"],
        },
        "pairedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth["data"]["sha256"],
        },
        "materialSurface": {
            "schema": surface["schema"],
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "profileAdoptionImplementationSha256": sha256_file(
            Path(contextual_profile_adoption.__file__)
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.json"
    data_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.npz"
    preview_path = output / "physical-mid-surface-cross-sections.png"
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
    bank_arrays = _load_npz(bank_path, bank)
    growth_arrays = _load_npz(growth_path, growth)
    surface_arrays = _load_npz(surface_path, surface)
    required = (
        "physicalSheetLabel",
        "physicalBoundarySide",
        "physicalSeedAnchor",
    )
    if any(name not in surface_arrays for name in required):
        raise ValueError("material surface lacks side-aware physical identities")
    position = np.asarray(surface_arrays["positionXYZ"], dtype=np.float64)
    signed_normal = np.asarray(surface_arrays["signedNormalXYZ"], dtype=np.float64)
    sheet_label = np.asarray(surface_arrays["physicalSheetLabel"], dtype=np.int32)
    boundary_side = np.asarray(surface_arrays["physicalBoundarySide"], dtype=np.uint8)
    local_evidence = np.asarray(surface_arrays["localEvidenceScore"], dtype=np.float64)
    originally_selected = np.asarray(growth_arrays["selected"], dtype=bool)
    originally_selected_label = np.asarray(
        growth_arrays["selectedLabel"], dtype=np.int32
    )
    bank_thickness = np.asarray(bank_arrays["thicknessVoxels"], dtype=np.float64)
    thickness_by_label = {
        int(value): bank_thickness[
            originally_selected & (originally_selected_label == value)
        ]
        for value in np.unique(originally_selected_label[originally_selected])
    }
    interface_identity = surface["identity"]["interfaces"]
    interface_path = Path(str(interface_identity["manifestPath"])).resolve()
    interface_manifest = json.loads(interface_path.read_text())
    interface_arrays = _load_npz(interface_path, interface_manifest)
    stride = int(interface_manifest["identity"]["settings"]["sampling_stride_voxels"])

    if resolved.adopt_contextual_profiles:
        profile_selection = contextual_profile_adoption.adopt_contextual_profiles(
            bank_arrays,
            growth_arrays,
            surface_arrays,
            interface_arrays,
            processing_start_xyz=np.asarray(
                interface_manifest["geometry"]["processingVoxelBounds"]["startXYZ"],
                dtype=np.float64,
            ),
            source_origin_xyz=np.asarray(
                interface_manifest["source"]["sourceOriginXYZ"], dtype=np.float64
            ),
            processing_shape_sampling_xyz=tuple(
                int(value)
                for value in interface_manifest["geometry"][
                    "processingShapeSamplingXYZ"
                ]
            ),
            sampling_stride_voxels=stride,
            settings=ContextualProfileAdoptionSettings(
                maximum_endpoint_position_residual_sampling_steps=(
                    resolved.maximum_contextual_endpoint_position_residual_sampling_steps
                ),
                maximum_endpoint_normal_degrees=(
                    resolved.maximum_contextual_endpoint_normal_degrees
                ),
                endpoint_normal_cost_scale_degrees=(
                    resolved.contextual_endpoint_normal_cost_scale_degrees
                ),
                minimum_local_evidence=(
                    resolved.minimum_contextual_profile_evidence
                ),
                thickness_tolerance_sampling_steps=(
                    resolved.contextual_thickness_tolerance_sampling_steps
                ),
            ),
        )
    else:
        profile_selection = {
            "selected": originally_selected,
            "originallySelected": originally_selected,
            "adopted": np.zeros(len(originally_selected), dtype=bool),
            "physicalSheetLabel": originally_selected_label,
            "canonicalOrientation": np.where(
                originally_selected, 1, 0
            ).astype(np.int8),
            "endpointSupportCount": np.zeros(
                len(originally_selected), dtype=np.uint8
            ),
            "summary": {
                "candidateCount": int(len(originally_selected)),
                "originalSelectedProfileCount": int(
                    np.count_nonzero(originally_selected)
                ),
                "contextuallyAdmissibleProfileCount": 0,
                "contextualEndpointIdentityConflictCount": 0,
                "selectedProfileCount": int(
                    np.count_nonzero(originally_selected)
                ),
                "adoptedContextualProfileCount": 0,
                "adoptedWithTwoMatchedEndpointsCount": 0,
                "adoptedWithOneMatchedEndpointCount": 0,
                "adoptedCanonicalOrientationCount": 0,
                "adoptedReversedOrientationCount": 0,
            },
        }
    selected = np.asarray(profile_selection["selected"], dtype=bool)
    selected_label = np.asarray(
        profile_selection["physicalSheetLabel"], dtype=np.int32
    )
    canonical_orientation = np.asarray(
        profile_selection["canonicalOrientation"], dtype=np.int8
    )
    raw_profile_lower = np.asarray(
        bank_arrays["boundaryLowerXYZ"], dtype=np.float64
    )
    raw_profile_upper = np.asarray(
        bank_arrays["boundaryUpperXYZ"], dtype=np.float64
    )
    raw_profile_normal = _normalized(
        np.asarray(bank_arrays["normalXYZ"], dtype=np.float64)
    )
    canonical_profile_lower = np.where(
        (canonical_orientation >= 0)[:, None],
        raw_profile_lower,
        raw_profile_upper,
    )
    canonical_profile_upper = np.where(
        (canonical_orientation >= 0)[:, None],
        raw_profile_upper,
        raw_profile_lower,
    )
    canonical_profile_normal = (
        raw_profile_normal * canonical_orientation[:, None]
    )

    selected_candidate = np.flatnonzero(selected).astype(np.int32)
    direct_profile_prior = local_profile_thickness_prior(
        position,
        signed_normal,
        sheet_label,
        boundary_side,
        canonical_profile_lower[selected],
        canonical_profile_upper[selected],
        canonical_profile_normal[selected],
        bank_thickness[selected],
        selected_label[selected],
        selected_candidate,
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    profile_prior = propagate_profile_thickness_prior(
        position,
        signed_normal,
        sheet_label,
        boundary_side,
        np.asarray(surface_arrays["edgeFirstNode"], dtype=np.int32),
        np.asarray(surface_arrays["edgeSecondNode"], dtype=np.int32),
        direct_profile_prior,
        sampling_stride_voxels=stride,
        settings=resolved,
    )

    pairing = pair_physical_boundary_faces(
        position,
        signed_normal,
        sheet_label,
        boundary_side,
        thickness_by_label,
        sampling_stride_voxels=stride,
        settings=resolved,
        local_thickness_prior=profile_prior["expectedThicknessVoxels"],
    )
    lower_node = pairing["lowerNode"]
    upper_node = pairing["upperNode"]
    dense_midpoint = 0.5 * (position[lower_node] + position[upper_node])
    dense_normal = _normalized(signed_normal[lower_node] - signed_normal[upper_node])
    dense_label = sheet_label[lower_node]
    if not np.array_equal(dense_label, sheet_label[upper_node]):
        raise RuntimeError("paired boundary nodes cross physical sheet identities")
    used_boundary_face_count = int(
        len(np.unique(np.concatenate((lower_node, upper_node))))
    )
    used_by_dense_pair = np.zeros(len(position), dtype=bool)
    used_by_dense_pair[lower_node] = True
    used_by_dense_pair[upper_node] = True
    proxy = one_sided_mid_surface_proxies(
        position,
        signed_normal,
        np.asarray(
            surface_arrays.get(
                "sheetNormalXYZ",
                surface_arrays.get("macroNormalXYZ", signed_normal),
            ),
            dtype=np.float64,
        ),
        sheet_label,
        boundary_side,
        profile_prior["expectedThicknessVoxels"],
        profile_prior["profileDistanceSamplingSteps"],
        used_by_dense_pair,
        settings=resolved,
        source_start_xyz=np.asarray(
            bank["source"]["sourceOriginXYZ"], dtype=np.float64
        ),
        source_stop_xyz_exclusive=(
            np.asarray(bank["source"]["sourceOriginXYZ"], dtype=np.float64)
            + np.asarray(bank["source"]["shapeZYX"], dtype=np.float64)[::-1]
        ),
    )
    proxy_midpoint = np.asarray(proxy["midpointXYZ"], dtype=np.float64)
    proxy_normal = np.asarray(proxy["normalXYZ"], dtype=np.float64)
    proxy_label = np.asarray(proxy["physicalSheetLabel"], dtype=np.int32)
    proxy_source_node = np.asarray(proxy["sourceSurfaceNode"], dtype=np.int32)
    dense_count = len(dense_midpoint)
    proxy_count = len(proxy_midpoint)
    support_midpoint = np.concatenate((dense_midpoint, proxy_midpoint), axis=0)
    support_normal = np.concatenate((dense_normal, proxy_normal), axis=0)
    support_label = np.concatenate((dense_label, proxy_label)).astype(np.int32)
    midpoint_by_surface_node = np.full(len(position), -1, dtype=np.int32)
    pair_index = np.arange(len(lower_node), dtype=np.int32)
    midpoint_by_surface_node[lower_node] = pair_index
    if len(upper_node):
        # Lower surfels are unique, whereas reciprocal patch support permits a
        # bounded number of staggered lower samples to share one upper surfel.
        # Use the lowest-cost pair as that upper surfel's deterministic graph
        # representative instead of relying on duplicate advanced assignment.
        pair_cost = np.asarray(pairing["pairCost"], dtype=np.float64)
        order = np.lexsort((pair_index, pair_cost, upper_node))
        ordered_upper = upper_node[order]
        first_for_upper = np.concatenate(
            (
                np.ones(1, dtype=bool),
                ordered_upper[1:] != ordered_upper[:-1],
            )
        )
        midpoint_by_surface_node[ordered_upper[first_for_upper]] = pair_index[
            order[first_for_upper]
        ]
    midpoint_by_surface_node[proxy_source_node] = (
        dense_count + np.arange(proxy_count, dtype=np.int32)
    )
    (
        surface_edge_first,
        surface_edge_second,
        surface_edge_support,
        surface_edge_score,
        edge_summary,
    ) = _midpoint_edges(
        midpoint_by_surface_node,
        np.asarray(surface_arrays["edgeFirstNode"], dtype=np.int64),
        np.asarray(surface_arrays["edgeSecondNode"], dtype=np.int64),
        boundary_side,
        support_midpoint,
        support_normal,
        support_label,
        sampling_stride_voxels=stride,
        settings=resolved,
    )

    profile_midpoint = np.asarray(bank_arrays["midpointXYZ"], dtype=np.float64)[
        selected
    ]
    profile_normal = canonical_profile_normal[selected]
    profile_label = selected_label[selected]
    profile_adopted = np.asarray(
        profile_selection["adopted"], dtype=bool
    )[selected]
    profile_endpoint_support = np.asarray(
        profile_selection["endpointSupportCount"], dtype=np.uint8
    )[selected]
    profile_orientation = canonical_orientation[selected]
    profile_node_kind = np.where(profile_adopted, 2, 0).astype(np.uint8)
    profile_count = len(profile_midpoint)
    candidate_to_profile_node = np.full(len(selected), -1, dtype=np.int32)
    candidate_to_profile_node[selected_candidate] = np.arange(
        profile_count, dtype=np.int32
    )
    growth_edge_first = np.asarray(
        growth_arrays["edgeFirstCandidate"], dtype=np.int64
    )
    growth_edge_second = np.asarray(
        growth_arrays["edgeSecondCandidate"], dtype=np.int64
    )
    profile_edge_valid = (
        selected[growth_edge_first]
        & selected[growth_edge_second]
        & (selected_label[growth_edge_first] == selected_label[growth_edge_second])
    )
    profile_edge_first = candidate_to_profile_node[
        growth_edge_first[profile_edge_valid]
    ]
    profile_edge_second = candidate_to_profile_node[
        growth_edge_second[profile_edge_valid]
    ]
    profile_edge_score = np.asarray(growth_arrays["edgeAffinity"], dtype=np.float32)[
        profile_edge_valid
    ]
    lower_profile_candidate = profile_prior["profileCandidateIndex"][lower_node]
    upper_profile_candidate = profile_prior["profileCandidateIndex"][upper_node]
    proxy_profile_candidate = profile_prior["profileCandidateIndex"][
        proxy_source_node
    ]
    support_lower_profile_candidate = np.concatenate(
        (lower_profile_candidate, proxy_profile_candidate)
    )
    support_upper_profile_candidate = np.concatenate(
        (upper_profile_candidate, proxy_profile_candidate)
    )
    (
        attachment_profile_node,
        attachment_support_node,
        attachment_score,
    ) = _profile_attachment_edges(
        support_midpoint,
        support_normal,
        support_label,
        support_lower_profile_candidate,
        support_upper_profile_candidate,
        candidate_to_profile_node,
        profile_midpoint,
        profile_normal,
        profile_label,
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    midpoint = np.concatenate((profile_midpoint, support_midpoint), axis=0)
    mid_normal = np.concatenate((profile_normal, support_normal), axis=0)
    mid_label = np.concatenate((profile_label, support_label)).astype(np.int32)
    edge_first = np.concatenate(
        (
            profile_edge_first.astype(np.int32),
            profile_count + surface_edge_first,
            attachment_profile_node,
        )
    )
    edge_second = np.concatenate(
        (
            profile_edge_second.astype(np.int32),
            profile_count + surface_edge_second,
            profile_count + attachment_support_node,
        )
    )
    edge_support = np.concatenate(
        (
            np.zeros(len(profile_edge_first), dtype=np.uint8),
            surface_edge_support,
            np.full(len(attachment_profile_node), 3, dtype=np.uint8),
        )
    )
    edge_score = np.concatenate(
        (profile_edge_score, surface_edge_score, attachment_score)
    ).astype(np.float32)
    edge_kind = np.concatenate(
        (
            np.zeros(len(profile_edge_first), dtype=np.uint8),
            np.ones(len(surface_edge_first), dtype=np.uint8),
            np.full(len(attachment_profile_node), 2, dtype=np.uint8),
        )
    )
    component, component_size = _components(
        len(midpoint), edge_first, edge_second
    )
    cross_label_edge_count = int(
        np.count_nonzero(mid_label[edge_first] != mid_label[edge_second])
    )
    physical_identity_violation_components = 0
    for component_id in np.unique(component):
        physical_identity_violation_components += int(
            len(np.unique(mid_label[component == component_id])) > 1
        )
    if cross_label_edge_count or physical_identity_violation_components:
        raise RuntimeError("physical mid-surface graph crossed a sheet identity")
    profile_float_fill = np.full(profile_count, np.nan, dtype=np.float32)
    profile_int_fill = np.full(profile_count, -1, dtype=np.int32)
    profile_evidence = np.asarray(
        bank_arrays["localEvidenceScore"], dtype=np.float32
    )[selected]
    dense_expected_thickness = 0.5 * (
        profile_prior["expectedThicknessVoxels"][lower_node]
        + profile_prior["expectedThicknessVoxels"][upper_node]
    )
    dense_thickness_residual = (
        pairing["thicknessVoxels"] - dense_expected_thickness
    ).astype(np.float32)
    proxy_float_fill = np.full(proxy_count, np.nan, dtype=np.float32)
    proxy_side = np.asarray(proxy["physicalBoundarySide"], dtype=np.uint8)
    proxy_evidence = local_evidence[proxy_source_node].astype(np.float32)
    proxy_component = np.asarray(surface_arrays["componentId"], dtype=np.int32)[
        proxy_source_node
    ]
    proxy_lower_surface = np.where(
        proxy_side == 0, proxy_source_node, -1
    ).astype(np.int32)
    proxy_upper_surface = np.where(
        proxy_side == 1, proxy_source_node, -1
    ).astype(np.int32)
    arrays = {
        "midpointXYZ": midpoint.astype(np.float32),
        "normalXYZ": mid_normal.astype(np.float32),
        "boundaryLowerXYZ": np.concatenate(
            (
                canonical_profile_lower[selected],
                position[lower_node],
                np.asarray(proxy["boundaryLowerXYZ"], dtype=np.float32),
            )
        ).astype(np.float32),
        "boundaryUpperXYZ": np.concatenate(
            (
                canonical_profile_upper[selected],
                position[upper_node],
                np.asarray(proxy["boundaryUpperXYZ"], dtype=np.float32),
            )
        ).astype(np.float32),
        "nodeKind": np.concatenate(
            (
                profile_node_kind,
                np.ones(dense_count, dtype=np.uint8),
                np.full(proxy_count, 3, dtype=np.uint8),
            )
        ),
        "profileAdopted": np.concatenate(
            (
                profile_adopted,
                np.zeros(dense_count + proxy_count, dtype=bool),
            )
        ),
        "profileEndpointSupportCount": np.concatenate(
            (
                profile_endpoint_support,
                np.zeros(dense_count + proxy_count, dtype=np.uint8),
            )
        ),
        "profileCanonicalOrientation": np.concatenate(
            (
                profile_orientation,
                np.zeros(dense_count + proxy_count, dtype=np.int8),
            )
        ),
        "profileCandidateIndex": np.concatenate(
            (
                selected_candidate,
                np.full(dense_count + proxy_count, -1, dtype=np.int32),
            )
        ),
        "lowerSurfaceNode": np.concatenate(
            (profile_int_fill, lower_node.astype(np.int32), proxy_lower_surface)
        ),
        "upperSurfaceNode": np.concatenate(
            (profile_int_fill, upper_node.astype(np.int32), proxy_upper_surface)
        ),
        "lowerFaceComponentId": np.concatenate(
            (
                profile_int_fill,
                np.asarray(surface_arrays["componentId"])[lower_node].astype(
                    np.int32
                ),
                np.where(proxy_side == 0, proxy_component, -1).astype(np.int32),
            )
        ),
        "upperFaceComponentId": np.concatenate(
            (
                profile_int_fill,
                np.asarray(surface_arrays["componentId"])[upper_node].astype(
                    np.int32
                ),
                np.where(proxy_side == 1, proxy_component, -1).astype(np.int32),
            )
        ),
        "physicalSheetLabel": mid_label.astype(np.int32),
        "componentId": component.astype(np.int32),
        "thicknessVoxels": np.concatenate(
            (
                bank_thickness[selected].astype(np.float32),
                pairing["thicknessVoxels"],
                np.asarray(proxy["thicknessVoxels"], dtype=np.float32),
            )
        ),
        "pairCost": np.concatenate(
            (profile_float_fill, pairing["pairCost"], proxy_float_fill)
        ),
        "thicknessResidualVoxels": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                dense_thickness_residual,
                proxy_float_fill,
            )
        ),
        "tangentResidualSamplingSteps": np.concatenate(
            (
                profile_float_fill,
                pairing["tangentResidualSamplingSteps"],
                proxy_float_fill,
            )
        ),
        "lowerDirectionDegrees": np.concatenate(
            (profile_float_fill, pairing["lowerDirectionDegrees"], proxy_float_fill)
        ),
        "upperDirectionDegrees": np.concatenate(
            (profile_float_fill, pairing["upperDirectionDegrees"], proxy_float_fill)
        ),
        "opposingNormalDegrees": np.concatenate(
            (profile_float_fill, pairing["opposingNormalDegrees"], proxy_float_fill)
        ),
        "lowerLocalEvidenceScore": np.concatenate(
            (
                profile_evidence,
                local_evidence[lower_node].astype(np.float32),
                proxy_evidence,
            )
        ),
        "upperLocalEvidenceScore": np.concatenate(
            (
                profile_evidence,
                local_evidence[upper_node].astype(np.float32),
                proxy_evidence,
            )
        ),
        "lowerExpectedThicknessVoxels": np.concatenate(
            (
                bank_thickness[selected].astype(np.float32),
                profile_prior["expectedThicknessVoxels"][lower_node],
                np.asarray(proxy["thicknessVoxels"], dtype=np.float32),
            )
        ),
        "upperExpectedThicknessVoxels": np.concatenate(
            (
                bank_thickness[selected].astype(np.float32),
                profile_prior["expectedThicknessVoxels"][upper_node],
                np.asarray(proxy["thicknessVoxels"], dtype=np.float32),
            )
        ),
        "lowerProfileDistanceSamplingSteps": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                profile_prior["profileDistanceSamplingSteps"][lower_node],
                np.asarray(
                    proxy["profileDistanceSamplingSteps"], dtype=np.float32
                ),
            )
        ),
        "upperProfileDistanceSamplingSteps": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                profile_prior["profileDistanceSamplingSteps"][upper_node],
                np.asarray(
                    proxy["profileDistanceSamplingSteps"], dtype=np.float32
                ),
            )
        ),
        "lowerProfileCandidateIndex": np.concatenate(
            (selected_candidate, lower_profile_candidate, proxy_profile_candidate)
        ),
        "upperProfileCandidateIndex": np.concatenate(
            (selected_candidate, upper_profile_candidate, proxy_profile_candidate)
        ),
        "oneSidedSourceSurfaceNode": np.concatenate(
            (profile_int_fill, np.full(dense_count, -1, dtype=np.int32), proxy_source_node)
        ),
        "oneSidedPhysicalBoundarySide": np.concatenate(
            (
                np.full(profile_count + dense_count, 255, dtype=np.uint8),
                proxy_side,
            )
        ),
        "edgeFirstNode": edge_first.astype(np.int32),
        "edgeSecondNode": edge_second.astype(np.int32),
        "edgeBoundarySupportMask": edge_support.astype(np.uint8),
        "edgeKind": edge_kind,
        "edgeScore": edge_score.astype(np.float32),
    }
    _write_npz(data_path, arrays)
    source = VolumeSource.open(bank["source"]["path"], bank["source"].get("metadataPath"))
    owned_record = bank["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(int(value) for value in owned_record["startXYZ"]),
        tuple(int(value) for value in owned_record["stopXYZExclusive"]),
    )
    write_material_surface_cross_sections(
        source,
        owned,
        midpoint,
        component,
        component_size,
        preview_path,
        display_high_raw=float(bank["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=MaterialSurfaceGraphSettings(),
    )
    labels_with_pairs = np.unique(dense_label)
    substantial = component_size >= 8
    dense_dense_surface_edge = (
        (surface_edge_first < dense_count)
        & (surface_edge_second < dense_count)
    )
    proxy_proxy_surface_edge = (
        (surface_edge_first >= dense_count)
        & (surface_edge_second >= dense_count)
    )
    mixed_surface_edge = ~(dense_dense_surface_edge | proxy_proxy_surface_edge)
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank["source"],
        "geometry": {
            **bank["geometry"],
            "samplingStrideVoxels": int(stride),
        },
        "counts": {
            "inputMaterialFaceNodeCount": int(len(position)),
            "physicallyLabeledFaceNodeCount": int(np.count_nonzero(sheet_label >= 0)),
            "lowerLabeledFaceNodeCount": int(np.count_nonzero(boundary_side == 0)),
            "upperLabeledFaceNodeCount": int(np.count_nonzero(boundary_side == 1)),
            "physicalSheetLabelCount": int(len(np.unique(sheet_label[sheet_label >= 0]))),
            "faceNodeWithLocalProfilePriorCount": int(
                np.count_nonzero(
                    np.isfinite(profile_prior["expectedThicknessVoxels"])
                )
            ),
            "faceNodeWithDirectProfilePriorCount": int(
                np.count_nonzero(profile_prior["directProfilePrior"])
            ),
            "faceNodeWithGeodesicallyPropagatedProfilePriorCount": int(
                np.count_nonzero(
                    profile_prior["geodesicallyPropagatedProfilePrior"]
                )
            ),
            "pairedPhysicalSheetLabelCount": int(len(labels_with_pairs)),
            "representedPhysicalSheetLabelCount": int(len(np.unique(mid_label))),
            "midSurfaceNodeCount": int(len(midpoint)),
            "directProfileNodeCount": int(profile_count),
            "trustedSelectedProfileNodeCount": int(
                np.count_nonzero(~profile_adopted)
            ),
            "contextualAdoptedProfileNodeCount": int(
                np.count_nonzero(profile_adopted)
            ),
            "denseBoundaryPairNodeCount": int(dense_count),
            "oneSidedThicknessProxyNodeCount": int(proxy_count),
            "usedBoundaryFaceNodeCount": used_boundary_face_count,
            "usedBoundaryFaceIncidenceCount": int(2 * dense_count),
            "usedBoundaryFaceNodeFraction": round(
                used_boundary_face_count
                / max(np.count_nonzero(sheet_label >= 0), 1),
                6,
            ),
            "representedBoundaryFaceNodeCount": int(
                used_boundary_face_count + proxy_count
            ),
            "representedBoundaryFaceNodeFraction": round(
                (used_boundary_face_count + proxy_count)
                / max(np.count_nonzero(sheet_label >= 0), 1),
                6,
            ),
            "retainedEdgeCount": int(len(edge_first)),
            "profileContinuityEdgeCount": int(len(profile_edge_first)),
            "surfaceProjectedContinuityEdgeCount": int(len(surface_edge_first)),
            "denseBoundaryContinuityEdgeCount": int(
                np.count_nonzero(dense_dense_surface_edge)
            ),
            "oneSidedProxyContinuityEdgeCount": int(
                np.count_nonzero(proxy_proxy_surface_edge)
            ),
            "denseProxyTransitionEdgeCount": int(
                np.count_nonzero(mixed_surface_edge)
            ),
            "profileAttachmentEdgeCount": int(len(attachment_profile_node)),
            "twoBoundarySupportedDenseEdgeCount": int(
                np.count_nonzero(
                    dense_dense_surface_edge & (surface_edge_support == 3)
                )
            ),
            "oneBoundarySupportedDenseEdgeCount": int(
                np.count_nonzero(
                    dense_dense_surface_edge & (surface_edge_support != 3)
                )
            ),
            "componentCount": int(len(component_size)),
            "crossPhysicalSheetEdgeCount": cross_label_edge_count,
            "physicalIdentityViolationComponentCount": int(
                physical_identity_violation_components
            ),
            "componentsAtLeast8Nodes": int(np.count_nonzero(substantial)),
            "componentsAtLeast32Nodes": int(np.count_nonzero(component_size >= 32)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_size >= 128)),
            "nodesInComponentsAtLeast8": int(np.sum(component_size[substantial])),
            "largestComponentSizes": [int(value) for value in component_size[:32]],
            **edge_summary,
        },
        "distributions": {
            "componentSize": _percentiles(component_size),
            "directProfileThicknessVoxels": _percentiles(
                bank_thickness[selected]
            ),
            "denseBoundaryPairThicknessVoxels": _percentiles(
                pairing["thicknessVoxels"]
            ),
            "pairCost": _percentiles(pairing["pairCost"]),
            "denseThicknessResidualVoxels": _percentiles(
                dense_thickness_residual
            ),
            "tangentResidualSamplingSteps": _percentiles(
                pairing["tangentResidualSamplingSteps"]
            ),
            "lowerDirectionDegrees": _percentiles(
                pairing["lowerDirectionDegrees"]
            ),
            "upperDirectionDegrees": _percentiles(
                pairing["upperDirectionDegrees"]
            ),
            "opposingNormalDegrees": _percentiles(
                pairing["opposingNormalDegrees"]
            ),
            "localProfileDistanceSamplingSteps": _percentiles(
                profile_prior["profileDistanceSamplingSteps"]
            ),
            "geodesicProfileDistanceSamplingSteps": _percentiles(
                profile_prior["profileGeodesicDistanceSamplingSteps"][
                    profile_prior["geodesicallyPropagatedProfilePrior"]
                ]
            ),
        },
        "profileAdoption": profile_selection["summary"],
        "pairingCensus": pairing["census"],
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"crossSections": preview_path.name},
        "method": {
            "node": (
                "trusted or face-supported air-papyrus-air profile, or a reciprocal "
                "local-patch correspondence between propagated lower/upper boundary faces"
            ),
            "identity": (
                "dense pairs require one immutable physical sheet on both faces; "
                "contextual profiles require at least one exact signed-face match "
                "and reject every two-face identity or side conflict"
            ),
            "profileAdoption": (
                "unused two-crossing profiles may fill an unowned lattice key only "
                "when signed endpoint support and trusted per-sheet thickness agree"
            ),
            "thicknessPrior": (
                "trusted selected per-sheet profile distribution, locally augmented "
                "only by identity-compatible contextual profiles"
            ),
            "thicknessTransport": (
                "bounded multi-source geodesic propagation on one immutable "
                "physical sheet and boundary side"
            ),
            "pairingGates": (
                "physical thickness, opposing signed normals, endpoint direction, "
                "tangent residual, reciprocal patch support, and bounded upper-"
                "surfel correspondence capacity"
            ),
            "edge": (
                "persisted physical-profile continuity, locally sheet-like dense "
                "boundary continuity, or a geometry-checked profile attachment"
            ),
            "oppositeFacesCollapsed": True,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
