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
from .isolated_slab import _percentile_record
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
    _components,
    _write_npz,
)
from .physical_mid_surface_mesh import (
    PhysicalMidSurfaceMeshSettings,
    _triangle_components,
    build_physical_mid_surface_mesh,
)


PAIRED_BOUNDARY_SURFACE_SCHEMA = "pareidolia.paired-boundary-surface"
PAIRED_BOUNDARY_SURFACE_VERSION = 1
PAIRED_BOUNDARY_SURFACE_STEM = "paired-boundary-surface-v1"


@dataclass(frozen=True, slots=True)
class PairedBoundarySurfaceSettings:
    """Mesh physical boundary tracks and certify local papyrus interiors.

    A boundary triangle is not called papyrus merely because it is smooth.
    Its three source profiles must point to one continuous companion boundary,
    both faces must have dense local support, and the resulting triangular
    prism must preserve thickness, orientation, and edge scale.  The accepted
    midpoint triangle is therefore an explicit local air--papyrus--air
    certificate rather than a connectivity extrapolation.
    """

    minimum_boundary_track_endpoints: int = 32
    maximum_boundary_tracks: int = 256
    minimum_mesh_component_endpoints: int = 16
    maximum_mesh_components: int = 512
    robust_chart_iterations: int = 2
    chart_solver_maximum_iterations: int = 1024
    minimum_local_support_degree: int = 2
    minimum_profile_evidence: float = 0.25
    maximum_thickness_ratio: float = 1.75
    maximum_corresponding_edge_ratio: float = 4.0
    maximum_inward_direction_spread_degrees: float = 35.0
    maximum_companion_triangle_normal_residual_degrees: float = 35.0
    maximum_midpoint_triangle_normal_residual_degrees: float = 35.0
    minimum_patch_frontier_edges: int = 3
    minimum_patch_frontier_profiles_per_side: int = 2
    minimum_patch_frontier_span_sampling_steps: float = 2.0
    minimum_patch_frontier_compatible_fraction: float = 0.6
    maximum_frontier_companion_distance_sampling_steps: float = 6.0
    maximum_frontier_companion_height_sampling_steps: float = 1.5
    maximum_frontier_companion_normal_degrees: float = 30.0
    maximum_frontier_thickness_ratio: float = 1.5
    minimum_certified_face_triangles: int = 8
    minimum_certified_patch_triangles: int = 8
    maximum_preview_patches: int = 48
    preview_size: int = 512

    def __post_init__(self) -> None:
        integers = (
            self.minimum_boundary_track_endpoints,
            self.maximum_boundary_tracks,
            self.minimum_mesh_component_endpoints,
            self.maximum_mesh_components,
            self.robust_chart_iterations,
            self.chart_solver_maximum_iterations,
            self.minimum_local_support_degree,
            self.minimum_patch_frontier_edges,
            self.minimum_patch_frontier_profiles_per_side,
            self.minimum_certified_face_triangles,
            self.minimum_certified_patch_triangles,
            self.maximum_preview_patches,
            self.preview_size,
        )
        if any(value < 1 for value in integers):
            raise ValueError("paired-boundary surface counts must be positive")
        if not 0.0 <= self.minimum_profile_evidence <= 1.0:
            raise ValueError("paired-boundary profile evidence must lie in [0, 1]")
        ratios = (
            self.maximum_thickness_ratio,
            self.maximum_corresponding_edge_ratio,
        )
        if any(not math.isfinite(value) or value <= 1.0 for value in ratios):
            raise ValueError("paired-boundary ratios must be finite and exceed one")
        angles = (
            self.maximum_inward_direction_spread_degrees,
            self.maximum_companion_triangle_normal_residual_degrees,
            self.maximum_midpoint_triangle_normal_residual_degrees,
            self.maximum_frontier_companion_normal_degrees,
        )
        if any(not 0.0 < value < 90.0 for value in angles):
            raise ValueError("paired-boundary angle caps must lie in (0, 90)")
        positive = (
            self.minimum_patch_frontier_span_sampling_steps,
            self.maximum_frontier_companion_distance_sampling_steps,
            self.maximum_frontier_companion_height_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("paired-boundary frontier distances must be positive")
        if not 0.0 < self.minimum_patch_frontier_compatible_fraction <= 1.0:
            raise ValueError("frontier compatible fraction must lie in (0, 1]")
        if self.maximum_frontier_thickness_ratio <= 1.0:
            raise ValueError("frontier thickness ratio must exceed one")

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def mesh_settings(self) -> PhysicalMidSurfaceMeshSettings:
        return PhysicalMidSurfaceMeshSettings(
            minimum_source_component_nodes=self.minimum_boundary_track_endpoints,
            maximum_source_components=self.maximum_boundary_tracks,
            minimum_mesh_component_nodes=self.minimum_mesh_component_endpoints,
            maximum_mesh_components=self.maximum_mesh_components,
            maximum_oriented_neighbor_normal_degrees=35.0,
            robust_chart_iterations=self.robust_chart_iterations,
            chart_huber_delta_voxels=1.0,
            maximum_mesh_edge_residual_voxels=1.5,
            minimum_chart_separation_voxels=0.15,
            maximum_local_closure_edge_voxels=6.0,
            maximum_local_closure_height_voxels=1.5,
            maximum_local_closure_normal_degrees=25.0,
            maximum_triangle_edge_voxels=7.0,
            maximum_triangle_normal_residual_degrees=35.0,
            minimum_triangle_area_voxels_squared=0.25,
            chart_solver_relative_tolerance=1.0e-7,
            chart_solver_maximum_iterations=self.chart_solver_maximum_iterations,
            triangulation_mode="chart-delaunay",
        )


def _resolve(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{PHYSICAL_MID_SURFACE_STEM}.json"


def _load_direct_surface(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    path = _resolve(root)
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_MID_SURFACE_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("paired-boundary meshing requires a complete surface catalog")
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("paired-boundary source data differs from its manifest")
    required = {
        "midpointXYZ",
        "normalXYZ",
        "thicknessVoxels",
        "lowerLocalEvidenceScore",
        "boundaryTrackEndpointXYZ",
        "boundaryTrackEndpointProfileNode",
        "boundaryTrackEndpointSide",
        "boundaryTrackComponentId",
        "boundaryTrackLocalSupportDegree",
        "boundaryTrackEdgeFirstEndpoint",
        "boundaryTrackEdgeSecondEndpoint",
        "boundaryTrackEdgeAffinity",
    }
    with np.load(data_path, allow_pickle=False) as stored:
        missing = required - set(stored.files)
        if missing:
            raise ValueError(
                f"surface catalog has no boundary-track fields: {sorted(missing)}"
            )
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return path, manifest, arrays


def _triangle_geometry(
    vertex_xyz: np.ndarray,
    reference_normal_xyz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(vertex_xyz, dtype=np.float64)
    reference = np.asarray(reference_normal_xyz, dtype=np.float64)
    cross = np.cross(points[:, 1] - points[:, 0], points[:, 2] - points[:, 0])
    doubled_area = np.linalg.norm(cross, axis=1)
    triangle_normal = cross / np.maximum(doubled_area[:, None], 1.0e-12)
    average_reference = np.sum(reference, axis=1)
    average_reference /= np.maximum(
        np.linalg.norm(average_reference, axis=1, keepdims=True), 1.0e-12
    )
    cosine = np.abs(np.einsum("ij,ij->i", triangle_normal, average_reference))
    residual = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
    edges = np.stack(
        (
            np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
        ),
        axis=1,
    )
    return 0.5 * doubled_area, residual, edges, triangle_normal


def _endpoint_companions(
    endpoint_profile: np.ndarray,
    endpoint_side: np.ndarray,
    profile_count: int,
) -> np.ndarray:
    profile = np.asarray(endpoint_profile, dtype=np.int32)
    side = np.asarray(endpoint_side, dtype=np.uint8)
    if len(profile) != 2 * profile_count or np.any(side > 1):
        raise ValueError("paired-boundary endpoints are not two faces per profile")
    by_profile_side = np.full((profile_count, 2), -1, dtype=np.int32)
    for endpoint, (profile_value, side_value) in enumerate(zip(profile, side)):
        if by_profile_side[int(profile_value), int(side_value)] >= 0:
            raise ValueError("paired-boundary profile has duplicate physical face")
        by_profile_side[int(profile_value), int(side_value)] = endpoint
    if np.any(by_profile_side < 0):
        raise ValueError("paired-boundary profile is missing one physical face")
    return by_profile_side[profile, 1 - side]


def associate_certified_profile_patches(
    endpoint_xyz: np.ndarray,
    endpoint_profile: np.ndarray,
    endpoint_component: np.ndarray,
    endpoint_companion: np.ndarray,
    endpoint_inward_normal_xyz: np.ndarray,
    edge_first_endpoint: np.ndarray,
    edge_second_endpoint: np.ndarray,
    edge_affinity: np.ndarray,
    profile_thickness_voxels: np.ndarray,
    profile_pair_component: np.ndarray,
    profile_patch_component: np.ndarray,
    *,
    sampling_stride_voxels: float,
    settings: PairedBoundarySurfaceSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Associate exact track-pair patches across one shared face frontier.

    The shared boundary is already one collision-safe physical track.  A
    bundle can associate its two adjacent exact-pair patches only when the
    unshared companion endpoints also continue as a signed surface with
    compatible thickness.  Single-edge bridges are never considered.
    """

    endpoint = np.asarray(endpoint_xyz, dtype=np.float64)
    endpoint_profile_value = np.asarray(endpoint_profile, dtype=np.int32)
    endpoint_component_value = np.asarray(endpoint_component, dtype=np.int32)
    companion = np.asarray(endpoint_companion, dtype=np.int32)
    inward = np.asarray(endpoint_inward_normal_xyz, dtype=np.float64)
    first_endpoint = np.asarray(edge_first_endpoint, dtype=np.int32)
    second_endpoint = np.asarray(edge_second_endpoint, dtype=np.int32)
    affinity = np.asarray(edge_affinity, dtype=np.float64)
    thickness = np.asarray(profile_thickness_voxels, dtype=np.float64)
    profile_pair = np.asarray(profile_pair_component, dtype=np.int32)
    profile_patch = np.asarray(profile_patch_component, dtype=np.int32)
    patch_count = int(np.max(profile_patch, initial=-1)) + 1
    empty_arrays = {
        "assemblyFirstPatch": np.empty(0, dtype=np.int32),
        "assemblySecondPatch": np.empty(0, dtype=np.int32),
        "assemblySharedBoundaryTrack": np.empty(0, dtype=np.int32),
        "assemblyFrontierEdgeCount": np.empty(0, dtype=np.int32),
        "assemblyCompatibleEdgeCount": np.empty(0, dtype=np.int32),
        "assemblyScore": np.empty(0, dtype=np.float32),
        "profileAssemblyComponentId": profile_patch.copy(),
        "profileAssemblyComponentSize": np.zeros(
            len(profile_patch), dtype=np.int32
        ),
    }
    if not patch_count or not len(first_endpoint):
        return empty_arrays, {
            "candidatePatchFrontierCount": 0,
            "acceptedPatchFrontierCount": 0,
            "acceptedPatchAssociationCount": 0,
            "assemblyCount": patch_count,
        }

    first_profile = endpoint_profile_value[first_endpoint]
    second_profile = endpoint_profile_value[second_endpoint]
    first_patch = profile_patch[first_profile]
    second_patch = profile_patch[second_profile]
    shared_track = endpoint_component_value[first_endpoint]
    same_boundary_track = shared_track == endpoint_component_value[second_endpoint]
    first_uses_shared = np.any(
        profile_pair[first_profile] == shared_track[:, None], axis=1
    )
    second_uses_shared = np.any(
        profile_pair[second_profile] == shared_track[:, None], axis=1
    )
    distinct_pair = np.any(
        profile_pair[first_profile] != profile_pair[second_profile], axis=1
    )
    frontier = (
        (first_patch >= 0)
        & (second_patch >= 0)
        & (first_patch != second_patch)
        & same_boundary_track
        & first_uses_shared
        & second_uses_shared
        & distinct_pair
    )
    frontier_index = np.flatnonzero(frontier)
    if not len(frontier_index):
        return empty_arrays, {
            "candidatePatchFrontierCount": 0,
            "acceptedPatchFrontierCount": 0,
            "acceptedPatchAssociationCount": 0,
            "assemblyCount": patch_count,
        }

    left_patch = np.minimum(first_patch[frontier_index], second_patch[frontier_index])
    right_patch = np.maximum(first_patch[frontier_index], second_patch[frontier_index])
    left_is_first = first_patch[frontier_index] == left_patch
    left_profile = np.where(
        left_is_first,
        first_profile[frontier_index],
        second_profile[frontier_index],
    )
    right_profile = np.where(
        left_is_first,
        second_profile[frontier_index],
        first_profile[frontier_index],
    )
    other_first = companion[first_endpoint[frontier_index]]
    other_second = companion[second_endpoint[frontier_index]]
    delta = endpoint[other_second] - endpoint[other_first]
    distance = np.linalg.norm(delta, axis=1) / float(sampling_stride_voxels)
    average_normal = inward[other_first] + inward[other_second]
    average_normal /= np.maximum(
        np.linalg.norm(average_normal, axis=1, keepdims=True), 1.0e-12
    )
    height = np.abs(np.einsum("ij,ij->i", delta, average_normal)) / float(
        sampling_stride_voxels
    )
    normal_cosine = np.einsum("ij,ij->i", inward[other_first], inward[other_second])
    normal_degrees = np.degrees(
        np.arccos(np.clip(normal_cosine, -1.0, 1.0))
    )
    thickness_ratio = np.maximum(
        thickness[first_profile[frontier_index]]
        / np.maximum(thickness[second_profile[frontier_index]], 1.0e-6),
        thickness[second_profile[frontier_index]]
        / np.maximum(thickness[first_profile[frontier_index]], 1.0e-6),
    )
    compatible = (
        (distance <= settings.maximum_frontier_companion_distance_sampling_steps)
        & (height <= settings.maximum_frontier_companion_height_sampling_steps)
        & (normal_degrees <= settings.maximum_frontier_companion_normal_degrees)
        & (thickness_ratio <= settings.maximum_frontier_thickness_ratio)
    )
    shared_center = 0.5 * (
        endpoint[first_endpoint[frontier_index]]
        + endpoint[second_endpoint[frontier_index]]
    )

    order = np.lexsort((right_patch, left_patch))
    ordered_pair = np.column_stack((left_patch[order], right_patch[order]))
    start = np.ones(len(order), dtype=bool)
    start[1:] = np.any(ordered_pair[1:] != ordered_pair[:-1], axis=1)
    starts = np.flatnonzero(start)
    stops = np.r_[starts[1:], len(order)]
    accepted_first: list[int] = []
    accepted_second: list[int] = []
    accepted_shared: list[int] = []
    accepted_edge_count: list[int] = []
    accepted_compatible_count: list[int] = []
    accepted_score: list[float] = []
    rejection = {
        "edgeSupport": 0,
        "compatibleFraction": 0,
        "uniqueProfiles": 0,
        "spatialSpan": 0,
        "inconsistentSharedTrack": 0,
    }
    for start_value, stop_value in zip(starts, stops):
        member = order[start_value:stop_value]
        member_compatible = member[compatible[member]]
        if (
            len(member) < settings.minimum_patch_frontier_edges
            or len(member_compatible) < settings.minimum_patch_frontier_edges
        ):
            rejection["edgeSupport"] += 1
            continue
        compatible_fraction = len(member_compatible) / len(member)
        if compatible_fraction < settings.minimum_patch_frontier_compatible_fraction:
            rejection["compatibleFraction"] += 1
            continue
        if (
            len(np.unique(left_profile[member_compatible]))
            < settings.minimum_patch_frontier_profiles_per_side
            or len(np.unique(right_profile[member_compatible]))
            < settings.minimum_patch_frontier_profiles_per_side
        ):
            rejection["uniqueProfiles"] += 1
            continue
        track_values = np.unique(shared_track[frontier_index][member_compatible])
        if len(track_values) != 1:
            rejection["inconsistentSharedTrack"] += 1
            continue
        span = (
            float(
                np.linalg.norm(
                    np.ptp(shared_center[member_compatible], axis=0)
                )
            )
            / float(sampling_stride_voxels)
        )
        if span < settings.minimum_patch_frontier_span_sampling_steps:
            rejection["spatialSpan"] += 1
            continue
        quality = (
            float(np.median(affinity[frontier_index][member_compatible]))
            * compatible_fraction
            * math.exp(
                -0.5
                * (
                    float(np.median(height[member_compatible]))
                    / settings.maximum_frontier_companion_height_sampling_steps
                )
                ** 2
                -0.5
                * (
                    float(np.median(normal_degrees[member_compatible]))
                    / settings.maximum_frontier_companion_normal_degrees
                )
                ** 2
            )
        )
        accepted_first.append(int(left_patch[member[0]]))
        accepted_second.append(int(right_patch[member[0]]))
        accepted_shared.append(int(track_values[0]))
        accepted_edge_count.append(len(member))
        accepted_compatible_count.append(len(member_compatible))
        accepted_score.append(quality)

    accepted_first_array = np.asarray(accepted_first, dtype=np.int32)
    accepted_second_array = np.asarray(accepted_second, dtype=np.int32)
    raw_assembly, _raw_count = _components(
        patch_count, accepted_first_array, accepted_second_array
    )
    used_profile = profile_patch >= 0
    raw_profile_assembly = np.full(len(profile_patch), -1, dtype=np.int32)
    raw_profile_assembly[used_profile] = raw_assembly[profile_patch[used_profile]]
    raw_value, raw_profile_count = np.unique(
        raw_profile_assembly[used_profile], return_counts=True
    )
    rank_order = np.lexsort((raw_value, -raw_profile_count))
    rank_by_raw = np.full(patch_count, -1, dtype=np.int32)
    rank_by_raw[raw_value[rank_order]] = np.arange(len(raw_value), dtype=np.int32)
    profile_assembly = np.full(len(profile_patch), -1, dtype=np.int32)
    profile_assembly[used_profile] = rank_by_raw[
        raw_profile_assembly[used_profile]
    ]
    assembly_profile_count = raw_profile_count[rank_order]
    arrays = {
        "assemblyFirstPatch": accepted_first_array,
        "assemblySecondPatch": accepted_second_array,
        "assemblySharedBoundaryTrack": np.asarray(accepted_shared, dtype=np.int32),
        "assemblyFrontierEdgeCount": np.asarray(accepted_edge_count, dtype=np.int32),
        "assemblyCompatibleEdgeCount": np.asarray(
            accepted_compatible_count, dtype=np.int32
        ),
        "assemblyScore": np.asarray(accepted_score, dtype=np.float32),
        "profileAssemblyComponentId": profile_assembly,
        "profileAssemblyComponentSize": (
            np.where(
                profile_assembly >= 0,
                assembly_profile_count[np.maximum(profile_assembly, 0)],
                0,
            ).astype(np.int32)
            if len(assembly_profile_count)
            else np.zeros(len(profile_patch), dtype=np.int32)
        ),
    }
    summary = {
        "candidatePatchFrontierCount": int(len(starts)),
        "acceptedPatchFrontierCount": int(len(accepted_first_array)),
        "acceptedPatchAssociationCount": int(len(accepted_first_array)),
        "assemblyCount": int(len(assembly_profile_count)),
        "largestAssemblyProfileCounts": [
            int(value) for value in assembly_profile_count[:32]
        ],
        "rejectedPatchFrontierCount": rejection,
        "frontierCompanionDistanceSamplingSteps": _percentile_record(distance),
        "frontierCompanionHeightSamplingSteps": _percentile_record(height),
        "frontierCompanionNormalDegrees": _percentile_record(normal_degrees),
        "acceptedAssociationScore": _percentile_record(
            np.asarray(accepted_score, dtype=np.float64)
        ),
    }
    return arrays, summary


def build_paired_boundary_surface(
    source: Mapping[str, np.ndarray],
    *,
    sampling_stride_voxels: float,
    settings: PairedBoundarySurfaceSettings | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build boundary meshes and strict local two-face papyrus certificates."""

    resolved = settings or PairedBoundarySurfaceSettings()
    endpoint_xyz = np.asarray(source["boundaryTrackEndpointXYZ"], dtype=np.float64)
    endpoint_profile = np.asarray(
        source["boundaryTrackEndpointProfileNode"], dtype=np.int32
    )
    endpoint_side = np.asarray(source["boundaryTrackEndpointSide"], dtype=np.uint8)
    endpoint_component = np.asarray(
        source["boundaryTrackComponentId"], dtype=np.int32
    )
    endpoint_support = np.asarray(
        source["boundaryTrackLocalSupportDegree"], dtype=np.int32
    )
    profile_count = len(source["midpointXYZ"])
    profile_normal = np.asarray(source["normalXYZ"], dtype=np.float64)
    profile_normal /= np.maximum(
        np.linalg.norm(profile_normal, axis=1, keepdims=True), 1.0e-12
    )
    inward_normal = profile_normal[endpoint_profile].copy()
    inward_normal[endpoint_side == 1] *= -1.0
    companion_endpoint = _endpoint_companions(
        endpoint_profile, endpoint_side, profile_count
    )

    mesh_summary, mesh = build_physical_mid_surface_mesh(
        endpoint_xyz,
        inward_normal,
        endpoint_component,
        endpoint_component,
        np.asarray(source["boundaryTrackEdgeFirstEndpoint"], dtype=np.int32),
        np.asarray(source["boundaryTrackEdgeSecondEndpoint"], dtype=np.int32),
        np.asarray(source["boundaryTrackEdgeAffinity"], dtype=np.float32),
        settings=resolved.mesh_settings(),
        geometry_scale=float(sampling_stride_voxels) / 2.0,
    )
    boundary_triangle = np.asarray(mesh["triangleNode"], dtype=np.int32)
    boundary_triangle_count = len(boundary_triangle)
    primary_component = (
        endpoint_component[boundary_triangle[:, 0]]
        if boundary_triangle_count
        else np.empty(0, dtype=np.int32)
    )
    companion_triangle = companion_endpoint[boundary_triangle]
    companion_component = endpoint_component[companion_triangle]
    single_companion = (
        np.all(companion_component == companion_component[:, :1], axis=1)
        if boundary_triangle_count
        else np.empty(0, dtype=bool)
    )
    partner_component = (
        companion_component[:, 0]
        if boundary_triangle_count
        else np.empty(0, dtype=np.int32)
    )
    track_size = np.bincount(
        endpoint_component,
        minlength=int(np.max(endpoint_component, initial=-1)) + 1,
    )
    canonical_primary = single_companion & (
        (track_size[primary_component] > track_size[partner_component])
        | (
            (track_size[primary_component] == track_size[partner_component])
            & (primary_component < partner_component)
        )
    )
    triangle_profile = endpoint_profile[boundary_triangle]
    unique_profiles = (
        (triangle_profile[:, 0] != triangle_profile[:, 1])
        & (triangle_profile[:, 0] != triangle_profile[:, 2])
        & (triangle_profile[:, 1] != triangle_profile[:, 2])
        if boundary_triangle_count
        else np.empty(0, dtype=bool)
    )
    dense_support = (
        np.all(
            endpoint_support[boundary_triangle]
            >= resolved.minimum_local_support_degree,
            axis=1,
        )
        & np.all(
            endpoint_support[companion_triangle]
            >= resolved.minimum_local_support_degree,
            axis=1,
        )
        if boundary_triangle_count
        else np.empty(0, dtype=bool)
    )
    evidence = np.asarray(source["lowerLocalEvidenceScore"], dtype=np.float64)
    evidence_ok = (
        np.min(evidence[triangle_profile], axis=1)
        >= resolved.minimum_profile_evidence
        if boundary_triangle_count
        else np.empty(0, dtype=bool)
    )
    # A physical boundary remains a valid flattenable surface when the CT
    # evidence on the other side becomes ambiguous.  Keep this signed face
    # certificate separate from the stricter two-face interior certificate
    # below: requiring one immutable paired profile at every vertex establishes
    # local air--material support, while the collision-safe boundary track and
    # signed chart establish surface identity.  Companion-track identity is
    # deliberately *not* part of this face decision.
    face_candidate = (
        unique_profiles
        & np.all(
            endpoint_support[boundary_triangle]
            >= resolved.minimum_local_support_degree,
            axis=1,
        )
        & evidence_ok
    )
    face_source_component_count = np.bincount(
        primary_component[face_candidate],
        minlength=int(np.max(endpoint_component, initial=-1)) + 1,
    ).astype(np.int32)
    retained_face_source = np.flatnonzero(
        face_source_component_count >= resolved.minimum_certified_face_triangles
    ).astype(np.int32)
    face_source_order = retained_face_source[
        np.lexsort(
            (
                retained_face_source,
                -face_source_component_count[retained_face_source],
            )
        )
    ]
    face_rank_by_source = np.full(len(face_source_component_count), -1, dtype=np.int32)
    face_rank_by_source[face_source_order] = np.arange(
        len(face_source_order), dtype=np.int32
    )
    face_certified = face_candidate & (face_rank_by_source[primary_component] >= 0)
    face_triangle_index = np.flatnonzero(face_certified).astype(np.int32)
    face_triangle_endpoint = boundary_triangle[face_certified].astype(np.int32)
    face_triangle_component = face_rank_by_source[
        primary_component[face_certified]
    ].astype(np.int32)
    face_component_triangle_count = face_source_component_count[
        face_source_order
    ].astype(np.int32)
    endpoint_face_component = np.full(len(endpoint_xyz), -1, dtype=np.int32)
    if len(face_triangle_endpoint):
        used_face_endpoint = np.unique(face_triangle_endpoint)
        endpoint_face_component[used_face_endpoint] = face_rank_by_source[
            endpoint_component[used_face_endpoint]
        ]
    thickness = np.asarray(source["thicknessVoxels"], dtype=np.float64)
    triangle_thickness = thickness[triangle_profile]
    thickness_ratio = (
        np.max(triangle_thickness, axis=1)
        / np.maximum(np.min(triangle_thickness, axis=1), 1.0e-6)
        if boundary_triangle_count
        else np.empty(0, dtype=np.float64)
    )
    thickness_ok = thickness_ratio <= resolved.maximum_thickness_ratio

    primary_xyz = endpoint_xyz[boundary_triangle]
    companion_xyz = endpoint_xyz[companion_triangle]
    midpoint_xyz = 0.5 * (primary_xyz + companion_xyz)
    primary_inward = inward_normal[boundary_triangle]
    companion_inward = inward_normal[companion_triangle]
    _, _, primary_edges, _ = _triangle_geometry(primary_xyz, primary_inward)
    companion_area, companion_normal_residual, companion_edges, _ = (
        _triangle_geometry(companion_xyz, companion_inward)
    )
    midpoint_area, midpoint_normal_residual, _, _ = _triangle_geometry(
        midpoint_xyz, primary_inward
    )
    edge_ratio = (
        np.maximum(
            primary_edges / np.maximum(companion_edges, 1.0e-6),
            companion_edges / np.maximum(primary_edges, 1.0e-6),
        ).max(axis=1)
        if boundary_triangle_count
        else np.empty(0, dtype=np.float64)
    )
    edge_ok = edge_ratio <= resolved.maximum_corresponding_edge_ratio
    companion_geometry_ok = (
        (companion_area > 1.0e-6)
        & (
            companion_normal_residual
            <= resolved.maximum_companion_triangle_normal_residual_degrees
        )
    )
    midpoint_geometry_ok = (
        (midpoint_area > 1.0e-6)
        & (
            midpoint_normal_residual
            <= resolved.maximum_midpoint_triangle_normal_residual_degrees
        )
    )
    inward_direction = companion_xyz - primary_xyz
    inward_direction /= np.maximum(
        np.linalg.norm(inward_direction, axis=2, keepdims=True), 1.0e-12
    )
    average_inward = np.sum(inward_direction, axis=1)
    average_inward /= np.maximum(
        np.linalg.norm(average_inward, axis=1, keepdims=True), 1.0e-12
    )
    inward_cosine = np.min(
        np.einsum("tvi,ti->tv", inward_direction, average_inward), axis=1
    )
    direction_ok = inward_cosine >= math.cos(
        math.radians(resolved.maximum_inward_direction_spread_degrees)
    )

    gates = {
        "singleCompanionTrack": single_companion,
        "canonicalPrimaryFace": canonical_primary,
        "uniqueSourceProfiles": unique_profiles,
        "denseTwoFaceSupport": dense_support,
        "profileEvidence": evidence_ok,
        "smoothThickness": thickness_ok,
        "coherentInwardDirection": direction_ok,
        "correspondingEdgeScale": edge_ok,
        "companionTriangleGeometry": companion_geometry_ok,
        "midpointTriangleGeometry": midpoint_geometry_ok,
    }
    certified = np.ones(boundary_triangle_count, dtype=bool)
    progressive: dict[str, int] = {}
    for name, gate in gates.items():
        certified &= gate
        progressive[name] = int(np.count_nonzero(certified))
    certified_index = np.flatnonzero(certified)
    certified_profile_triangle = triangle_profile[certified]
    seed_profile = np.zeros(profile_count, dtype=bool)
    seed_profile[np.unique(certified_profile_triangle)] = True
    triangle_union = (
        np.column_stack(
            (
                certified_profile_triangle[:, (0, 1, 2)].ravel(),
                certified_profile_triangle[:, (1, 2, 0)].ravel(),
            )
        )
        if len(certified_profile_triangle)
        else np.empty((0, 2), dtype=np.int32)
    )
    source_edge_first = np.asarray(
        source["boundaryTrackEdgeFirstEndpoint"], dtype=np.int32
    )
    source_edge_second = np.asarray(
        source["boundaryTrackEdgeSecondEndpoint"], dtype=np.int32
    )
    source_edge_first_profile = endpoint_profile[source_edge_first]
    source_edge_second_profile = endpoint_profile[source_edge_second]
    profile_lower_component = np.full(profile_count, -1, dtype=np.int32)
    profile_upper_component = np.full(profile_count, -1, dtype=np.int32)
    lower_endpoint = endpoint_side == 0
    upper_endpoint = endpoint_side == 1
    profile_lower_component[endpoint_profile[lower_endpoint]] = endpoint_component[
        lower_endpoint
    ]
    profile_upper_component[endpoint_profile[upper_endpoint]] = endpoint_component[
        upper_endpoint
    ]
    profile_pair = np.sort(
        np.column_stack((profile_lower_component, profile_upper_component)), axis=1
    )
    profile_support = np.zeros((profile_count, 2), dtype=np.int32)
    profile_support[endpoint_profile, endpoint_side] = endpoint_support
    eligible_profile = (
        np.all(
            profile_support >= resolved.minimum_local_support_degree,
            axis=1,
        )
        & (evidence >= resolved.minimum_profile_evidence)
        & (profile_lower_component >= 0)
        & (profile_upper_component >= 0)
        & (profile_lower_component != profile_upper_component)
    )
    eligible_profile_index = np.flatnonzero(eligible_profile)
    local_by_profile = np.full(profile_count, -1, dtype=np.int32)
    local_by_profile[eligible_profile_index] = np.arange(
        len(eligible_profile_index), dtype=np.int32
    )
    source_edge_same_pair = np.all(
        profile_pair[source_edge_first_profile]
        == profile_pair[source_edge_second_profile],
        axis=1,
    )
    source_edge_used = (
        eligible_profile[source_edge_first_profile]
        & eligible_profile[source_edge_second_profile]
        & source_edge_same_pair
    )
    continuity_union = np.column_stack(
        (
            source_edge_first_profile[source_edge_used],
            source_edge_second_profile[source_edge_used],
        )
    )
    union_edge = np.vstack((triangle_union, continuity_union)).astype(np.int32)
    if len(eligible_profile_index):
        raw_component, raw_component_size = _components(
            len(eligible_profile_index),
            local_by_profile[union_edge[:, 0]],
            local_by_profile[union_edge[:, 1]],
        )
        seed_raw_component = np.unique(
            raw_component[local_by_profile[np.flatnonzero(seed_profile)]]
        )
        seed_raw_size = raw_component_size[seed_raw_component]
        seed_order = np.lexsort((seed_raw_component, -seed_raw_size))
        ranked_raw_component = seed_raw_component[seed_order]
        profile_patch_size = seed_raw_size[seed_order].astype(np.int32)
        rank_by_raw = np.full(len(raw_component_size), -1, dtype=np.int32)
        rank_by_raw[ranked_raw_component] = np.arange(
            len(ranked_raw_component), dtype=np.int32
        )
        profile_patch = np.full(profile_count, -1, dtype=np.int32)
        eligible_rank = rank_by_raw[raw_component]
        retained_eligible = eligible_rank >= 0
        profile_patch[
            eligible_profile_index[retained_eligible]
        ] = eligible_rank[retained_eligible]
        triangle_patch = profile_patch[certified_profile_triangle]
        if np.any(triangle_patch != triangle_patch[:, :1]):
            raise RuntimeError("one certified triangle crosses paired profile patches")
        patch_component = triangle_patch[:, 0]
        patch_triangle_count = np.bincount(
            patch_component, minlength=len(profile_patch_size)
        ).astype(np.int32)
    else:
        profile_patch = np.full(profile_count, -1, dtype=np.int32)
        profile_patch_size = np.empty(0, dtype=np.int32)
        patch_component = np.empty(0, dtype=np.int32)
        patch_triangle_count = np.empty(0, dtype=np.int32)
    # Triangle manifoldness remains a mesh diagnostic, independent of the
    # broader measured-continuity identity used for paired patches.
    _triangle_component, _triangle_size, patch_manifold = _triangle_components(
        certified_profile_triangle
    )
    substantial_patch = (
        patch_triangle_count[patch_component]
        >= resolved.minimum_certified_patch_triangles
        if len(patch_component)
        else np.empty(0, dtype=bool)
    )
    association_arrays, association_summary = associate_certified_profile_patches(
        endpoint_xyz,
        endpoint_profile,
        endpoint_component,
        companion_endpoint,
        inward_normal,
        np.asarray(source["boundaryTrackEdgeFirstEndpoint"], dtype=np.int32),
        np.asarray(source["boundaryTrackEdgeSecondEndpoint"], dtype=np.int32),
        np.asarray(source["boundaryTrackEdgeAffinity"], dtype=np.float32),
        thickness,
        profile_pair,
        profile_patch,
        sampling_stride_voxels=sampling_stride_voxels,
        settings=resolved,
    )
    profile_assembly = np.asarray(
        association_arrays["profileAssemblyComponentId"], dtype=np.int32
    )
    triangle_assembly = (
        profile_assembly[certified_profile_triangle]
        if len(certified_profile_triangle)
        else np.empty((0, 3), dtype=np.int32)
    )
    if len(triangle_assembly) and np.any(
        triangle_assembly != triangle_assembly[:, :1]
    ):
        raise RuntimeError("one certified triangle crosses paired assemblies")
    certified_triangle_assembly = (
        triangle_assembly[:, 0]
        if len(triangle_assembly)
        else np.empty(0, dtype=np.int32)
    )
    assembly_triangle_count = np.bincount(
        certified_triangle_assembly,
        minlength=int(np.max(certified_triangle_assembly, initial=-1)) + 1,
    ).astype(np.int32)
    pair = np.column_stack(
        (primary_component[certified], partner_component[certified])
    ).astype(np.int32)
    arrays = {
        "endpointXYZ": endpoint_xyz.astype(np.float32),
        "endpointProfileNode": endpoint_profile,
        "endpointSide": endpoint_side,
        "endpointComponentId": endpoint_component,
        "endpointCompanion": companion_endpoint,
        "endpointInwardNormalXYZ": inward_normal.astype(np.float32),
        "endpointLocalSupportDegree": endpoint_support,
        **{f"mesh{name[0].upper()}{name[1:]}": value for name, value in mesh.items()},
        "boundaryTrianglePairedCertified": certified,
        "boundaryTriangleFaceCertified": face_certified,
        "boundaryTriangleCompanionEndpoint": companion_triangle.astype(np.int32),
        "boundaryTriangleCompanionComponent": partner_component.astype(np.int32),
        "boundaryTriangleThicknessRatio": thickness_ratio.astype(np.float32),
        "boundaryTriangleCorrespondingEdgeRatio": edge_ratio.astype(np.float32),
        "boundaryTriangleInwardDirectionCosine": inward_cosine.astype(np.float32),
        "boundaryTriangleCompanionNormalResidualDegrees": (
            companion_normal_residual.astype(np.float32)
        ),
        "boundaryTriangleMidpointNormalResidualDegrees": (
            midpoint_normal_residual.astype(np.float32)
        ),
        "certifiedFaceBoundaryTriangleIndex": face_triangle_index,
        "certifiedFaceTriangleEndpoint": face_triangle_endpoint,
        "certifiedFaceTriangleProfileNode": endpoint_profile[
            face_triangle_endpoint
        ].astype(np.int32),
        "certifiedFaceTriangleSourceBoundaryTrack": primary_component[
            face_certified
        ].astype(np.int32),
        "certifiedFaceTriangleComponentId": face_triangle_component,
        "certifiedFaceTriangleComponentSize": (
            face_component_triangle_count[face_triangle_component]
            if len(face_triangle_component)
            else np.empty(0, dtype=np.int32)
        ),
        "endpointCertifiedFaceComponentId": endpoint_face_component,
        "certifiedBoundaryTriangleIndex": certified_index.astype(np.int32),
        "certifiedTriangleProfileNode": certified_profile_triangle.astype(np.int32),
        "certifiedTrianglePrimaryEndpoint": boundary_triangle[certified].astype(np.int32),
        "certifiedTriangleCompanionEndpoint": companion_triangle[certified].astype(np.int32),
        "certifiedTriangleBoundaryTrackPair": pair,
        "certifiedTrianglePatchComponentId": patch_component.astype(np.int32),
        "certifiedTrianglePatchComponentSize": (
            patch_triangle_count[patch_component].astype(np.int32)
            if len(patch_component)
            else np.empty(0, dtype=np.int32)
        ),
        "certifiedProfilePatchComponentId": profile_patch,
        "certifiedProfilePatchComponentSize": (
            np.where(
                profile_patch >= 0,
                profile_patch_size[np.maximum(profile_patch, 0)],
                0,
            ).astype(np.int32)
            if len(profile_patch_size)
            else np.zeros(profile_count, dtype=np.int32)
        ),
        **association_arrays,
        "certifiedTriangleAssemblyComponentId": certified_triangle_assembly,
        "certifiedTriangleAssemblyComponentSize": (
            assembly_triangle_count[certified_triangle_assembly]
            if len(certified_triangle_assembly)
            else np.empty(0, dtype=np.int32)
        ),
        "certifiedTriangleSubstantialPatch": substantial_patch,
    }
    profile_used = np.unique(certified_profile_triangle) if len(certified) else np.empty(0)
    endpoint_used = (
        np.unique(
            np.concatenate(
                (boundary_triangle[certified].ravel(), companion_triangle[certified].ravel())
            )
        )
        if np.any(certified)
        else np.empty(0, dtype=np.int32)
    )
    summary = {
        "mesh": mesh_summary,
        "counts": {
            "profileCount": int(profile_count),
            "boundaryEndpointCount": int(len(endpoint_xyz)),
            "meshedBoundaryTrackCount": int(
                mesh_summary["counts"]["selectedSourceComponentCount"]
            ),
            "boundaryTriangleCount": int(boundary_triangle_count),
            "certifiedBoundaryFaceTriangleCount": int(len(face_triangle_index)),
            "certifiedBoundaryFaceEndpointCount": int(
                np.count_nonzero(endpoint_face_component >= 0)
            ),
            "certifiedBoundaryFaceComponentCount": int(
                len(face_component_triangle_count)
            ),
            "largestCertifiedBoundaryFaceTriangleCounts": [
                int(value) for value in face_component_triangle_count[:32]
            ],
            "certifiedPairedTriangleCount": int(np.count_nonzero(certified)),
            "certifiedProfileCount": int(len(profile_used)),
            "certifiedEndpointCount": int(len(endpoint_used)),
            "certifiedPatchCount": int(len(profile_patch_size)),
            "certifiedSubstantialPatchCount": int(
                np.count_nonzero(
                    patch_triangle_count
                    >= resolved.minimum_certified_patch_triangles
                )
            ),
            "largestCertifiedPatchProfileCounts": [
                int(value) for value in profile_patch_size[:32]
            ],
            "largestCertifiedPatchTriangleCounts": [
                int(value)
                for value in sorted(patch_triangle_count, reverse=True)[:32]
            ],
            "certifiedAssemblyCount": int(
                association_summary["assemblyCount"]
            ),
            "largestCertifiedAssemblyTriangleCounts": [
                int(value)
                for value in sorted(assembly_triangle_count, reverse=True)[:32]
            ],
            **patch_manifold,
        },
        "patchAssociation": association_summary,
        "progressiveCertificateCounts": progressive,
        "distributions": {
            "certifiedThicknessVoxels": _percentile_record(
                triangle_thickness[certified].ravel()
            ),
            "certifiedThicknessRatio": _percentile_record(thickness_ratio[certified]),
            "certifiedCorrespondingEdgeRatio": _percentile_record(edge_ratio[certified]),
            "certifiedCompanionNormalResidualDegrees": _percentile_record(
                companion_normal_residual[certified]
            ),
            "certifiedMidpointNormalResidualDegrees": _percentile_record(
                midpoint_normal_residual[certified]
            ),
        },
    }
    return arrays, summary


def _draw_line(
    image: np.ndarray,
    first: tuple[int, int],
    second: tuple[int, int],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = first
    x1, y1 = second
    steps = max(abs(x1 - x0), abs(y1 - y0), 1)
    x = np.rint(np.linspace(x0, x1, steps + 1)).astype(np.int32)
    y = np.rint(np.linspace(y0, y1, steps + 1)).astype(np.int32)
    valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
    image[y[valid], x[valid]] = color


def write_paired_boundary_surface_preview(
    path: Path,
    source: Mapping[str, np.ndarray],
    arrays: Mapping[str, np.ndarray],
    *,
    maximum_patches: int,
    size: int,
) -> None:
    canvas = np.full((size, 3 * size, 3), (8, 13, 17), dtype=np.uint8)
    endpoint = np.asarray(arrays["endpointXYZ"], dtype=np.float64)
    triangle = np.asarray(
        arrays["certifiedFaceTriangleEndpoint"], dtype=np.int32
    )
    patch = np.asarray(
        arrays["certifiedFaceTriangleComponentId"], dtype=np.int32
    )
    patch_size = np.asarray(
        arrays["certifiedFaceTriangleComponentSize"], dtype=np.int32
    )
    if not len(triangle):
        path.write_bytes(rgb_png(canvas))
        return
    low = np.min(endpoint, axis=0)
    high = np.max(endpoint, axis=0)
    span = np.maximum(high - low, 1.0)
    selected_patch = np.unique(patch[patch_size > 0])[:maximum_patches]
    palette: dict[int, tuple[int, int, int]] = {}
    for rank, value in enumerate(selected_patch):
        red, green, blue = colorsys.hsv_to_rgb(
            (rank * 0.61803398875) % 1.0, 0.62, 1.0
        )
        palette[int(value)] = (
            int(round(255 * red)),
            int(round(255 * green)),
            int(round(255 * blue)),
        )
    projections = ((0, 1), (0, 2), (1, 2))
    for panel, axes in enumerate(projections):
        projected = np.rint(
            (endpoint[:, axes] - low[list(axes)])[None, :, :]
        )[0]
        projected[:, 0] = (
            (endpoint[:, axes[0]] - low[axes[0]]) / span[axes[0]] * (size - 1)
        )
        projected[:, 1] = (
            (1.0 - (endpoint[:, axes[1]] - low[axes[1]]) / span[axes[1]])
            * (size - 1)
        )
        projected = np.rint(projected).astype(np.int32)
        for triangle_index, nodes in enumerate(triangle):
            color = palette.get(int(patch[triangle_index]))
            if color is None:
                continue
            points = projected[nodes]
            for edge in range(3):
                first = points[edge]
                second = points[(edge + 1) % 3]
                _draw_line(
                    canvas[:, panel * size : (panel + 1) * size],
                    (int(first[0]), int(first[1])),
                    (int(second[0]), int(second[1])),
                    color,
                )
    path.write_bytes(rgb_png(canvas))


def run_paired_boundary_surface(
    direct_surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PairedBoundarySurfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    source_path, source_manifest, source = _load_direct_surface(direct_surface_root)
    resolved = settings or PairedBoundarySurfaceSettings()
    sampling_stride = float(
        source_manifest.get("geometry", {}).get("samplingStrideVoxels", 2.0)
    )
    try:
        import scipy  # type: ignore[import-not-found]
    except ImportError:
        scipy_version: str | None = None
    else:
        scipy_version = str(scipy.__version__)
    identity: dict[str, Any] = {
        "schema": PAIRED_BOUNDARY_SURFACE_SCHEMA,
        "version": PAIRED_BOUNDARY_SURFACE_VERSION,
        "directSurface": {
            "manifestPath": str(source_path),
            "manifestSha256": sha256_file(source_path),
            "dataSha256": source_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "samplingStrideVoxels": sampling_stride,
        "scipyVersion": scipy_version,
        "implementationSha256": sha256_file(Path(__file__)),
        "meshImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_mid_surface_mesh.py")
        ),
        "delaunayImplementationSha256": sha256_file(
            Path(__file__).with_name("needle_surface.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PAIRED_BOUNDARY_SURFACE_STEM}.json"
    data_path = output / f"{PAIRED_BOUNDARY_SURFACE_STEM}.npz"
    preview_path = output / "paired-boundary-surface-projections.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    arrays, summary = build_paired_boundary_surface(
        source,
        sampling_stride_voxels=sampling_stride,
        settings=resolved,
    )
    _write_npz(data_path, arrays)
    write_paired_boundary_surface_preview(
        preview_path,
        source,
        arrays,
        maximum_patches=resolved.maximum_preview_patches,
        size=resolved.preview_size,
    )
    payload: dict[str, Any] = {
        "schema": PAIRED_BOUNDARY_SURFACE_SCHEMA,
        "version": PAIRED_BOUNDARY_SURFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_manifest["source"],
        "geometry": source_manifest["geometry"],
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"projection": preview_path.name},
        "method": {
            "boundarySurface": (
                "collision-safe physical boundary tracks integrated into "
                "intrinsic charts and triangulated by local Delaunay geometry"
            ),
            "flattenableFaceCertificate": (
                "one signed physical boundary triangle with dense local track "
                "support and immutable air-material profile evidence; opposite-"
                "face identity is retained as evidence but does not define the "
                "continuity of the observed boundary"
            ),
            "papyrusCertificate": (
                "one boundary triangle plus the same three immutable profiles "
                "on one companion track, gated by two-face support, physical "
                "thickness smoothness, orientation, and corresponding edges"
            ),
            "truthLabelsUsed": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
