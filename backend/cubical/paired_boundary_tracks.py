from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np

from .isolated_slab import _percentile_record
from .material_surface_graph import _collision_safe_components, _tangent_columns
from .physical_mid_surface import _components


@dataclass(frozen=True, slots=True)
class PairedBoundaryTrackSettings:
    """Recover physical boundary tracks from selected paired profiles.

    A profile contains two observed air/material boundaries, but the two faces
    need not remain paired everywhere: an ambiguous exit crossing can change
    while the nearer physical boundary remains continuous.  Short links form
    the local support census.  A longer link is admitted only when both of its
    endpoint observations already have multiple independent local
    continuations.  Final unions are constrained to one narrow depth interval
    per macro-tangent column so a transitive shear jump cannot silently fuse
    neighboring layers.
    """

    minimum_track_affinity: float = 0.45
    local_support_radius_sampling_steps: float = math.sqrt(5.0)
    minimum_local_support_affinity: float = 0.2
    minimum_local_support_degree: int = 2
    tangent_column_width_sampling_steps: float = 1.5
    maximum_column_depth_range_sampling_steps: float = 2.25

    def __post_init__(self) -> None:
        for value, name in (
            (self.minimum_track_affinity, "track affinity"),
            (self.minimum_local_support_affinity, "local support affinity"),
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"boundary {name} must lie in [0, 1]")
        positive = (
            self.local_support_radius_sampling_steps,
            self.tangent_column_width_sampling_steps,
            self.maximum_column_depth_range_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("boundary-track distances must be finite and positive")
        if self.minimum_local_support_degree < 1:
            raise ValueError("boundary-track local support degree must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def build_paired_boundary_tracks(
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    spatial_key_xyz: np.ndarray,
    macro_bin_index: np.ndarray,
    macro_center_xyz: np.ndarray,
    macro_normal_xyz: np.ndarray,
    edge_first_profile: np.ndarray,
    edge_second_profile: np.ndarray,
    edge_second_orientation: np.ndarray,
    edge_lower_affinity: np.ndarray,
    edge_upper_affinity: np.ndarray,
    *,
    sampling_stride_voxels: float,
    settings: PairedBoundaryTrackSettings | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Construct collision-safe components over individual profile faces."""

    resolved = settings or PairedBoundaryTrackSettings()
    lower = np.asarray(lower_xyz, dtype=np.float64)
    upper = np.asarray(upper_xyz, dtype=np.float64)
    key = np.asarray(spatial_key_xyz, dtype=np.int64)
    macro_bin = np.asarray(macro_bin_index, dtype=np.int32)
    macro_center = np.asarray(macro_center_xyz, dtype=np.float64)
    macro_normal = np.asarray(macro_normal_xyz, dtype=np.float64)
    profile_count = len(lower)
    if any(
        len(value) != profile_count
        for value in (upper, key, macro_bin, macro_center, macro_normal)
    ):
        raise ValueError("boundary-track profile geometry is not aligned")
    first_profile = np.asarray(edge_first_profile, dtype=np.int32)
    second_profile = np.asarray(edge_second_profile, dtype=np.int32)
    orientation = np.asarray(edge_second_orientation, dtype=np.int8)
    lower_affinity = np.asarray(edge_lower_affinity, dtype=np.float32)
    upper_affinity = np.asarray(edge_upper_affinity, dtype=np.float32)
    pair_edge_count = len(first_profile)
    if any(
        len(value) != pair_edge_count
        for value in (
            second_profile,
            orientation,
            lower_affinity,
            upper_affinity,
        )
    ):
        raise ValueError("boundary-track endpoint edges are not aligned")
    if pair_edge_count and (
        np.min(first_profile) < 0
        or np.min(second_profile) < 0
        or np.max(first_profile) >= profile_count
        or np.max(second_profile) >= profile_count
    ):
        raise ValueError("boundary-track endpoint edge is out of range")
    if np.any((orientation != 1) & (orientation != -1)):
        raise ValueError("boundary-track orientations must be +1 or -1")

    endpoint_xyz = np.empty((2 * profile_count, 3), dtype=np.float64)
    endpoint_xyz[0::2] = lower
    endpoint_xyz[1::2] = upper
    endpoint_profile = np.repeat(np.arange(profile_count, dtype=np.int32), 2)
    endpoint_side = np.tile(np.asarray((0, 1), dtype=np.uint8), profile_count)

    # The first half describes the first profile's lower face.  When the
    # second profile's axial normal flips, that continuation lands on its
    # stored upper face.  The second half performs the complementary mapping.
    first_endpoint = np.concatenate((2 * first_profile, 2 * first_profile + 1))
    second_endpoint = np.concatenate(
        (
            2 * second_profile + (orientation < 0).astype(np.int32),
            2 * second_profile + (orientation > 0).astype(np.int32),
        )
    )
    affinity = np.concatenate((lower_affinity, upper_affinity)).astype(
        np.float32, copy=False
    )
    key_distance = np.linalg.norm(
        key[second_profile] - key[first_profile], axis=1
    )
    key_distance = np.concatenate((key_distance, key_distance))
    local_link = (
        key_distance <= resolved.local_support_radius_sampling_steps + 1.0e-9
    )
    local_support_edge = local_link & (
        affinity >= resolved.minimum_local_support_affinity
    )
    local_degree = np.bincount(
        np.concatenate(
            (
                first_endpoint[local_support_edge],
                second_endpoint[local_support_edge],
            )
        ),
        minlength=2 * profile_count,
    ).astype(np.int32)
    strong_edge = affinity >= resolved.minimum_track_affinity
    supported_long_edge = (
        strong_edge
        & ~local_link
        & (local_degree[first_endpoint] >= resolved.minimum_local_support_degree)
        & (local_degree[second_endpoint] >= resolved.minimum_local_support_degree)
    )
    proposed_edge = (strong_edge & local_link) | supported_long_edge

    proposed_first = first_endpoint[proposed_edge].astype(np.int32, copy=False)
    proposed_second = second_endpoint[proposed_edge].astype(np.int32, copy=False)
    proposed_affinity = affinity[proposed_edge]
    pre_component, _pre_count = _components(
        2 * profile_count, proposed_first, proposed_second
    )
    endpoint_macro_bin = np.repeat(macro_bin, 2)
    endpoint_macro_center = np.repeat(macro_center, 2, axis=0)
    endpoint_macro_normal = np.repeat(macro_normal, 2, axis=0)
    tangent_column, normal_depth = _tangent_columns(
        endpoint_xyz,
        endpoint_macro_bin,
        endpoint_macro_center,
        endpoint_macro_normal,
        stride=max(int(round(sampling_stride_voxels)), 1),
        width_sampling_steps=resolved.tangent_column_width_sampling_steps,
    )
    component, component_count, retained_proposed, collision_summary = (
        _collision_safe_components(
            pre_component,
            proposed_first,
            proposed_second,
            proposed_affinity,
            tangent_column,
            normal_depth,
            maximum_depth_range=(
                resolved.maximum_column_depth_range_sampling_steps
            ),
        )
    )
    retained_edge = np.zeros(len(first_endpoint), dtype=bool)
    retained_edge[np.flatnonzero(proposed_edge)] = retained_proposed.astype(bool)
    profile_collision = component[0::2] == component[1::2]
    if np.any(profile_collision):
        raise RuntimeError(
            "one boundary track contains both physical faces of a paired profile"
        )

    arrays = {
        "endpointXYZ": endpoint_xyz.astype(np.float32),
        "endpointProfileNode": endpoint_profile,
        "endpointSide": endpoint_side,
        "endpointComponentId": component.astype(np.int32),
        "endpointLocalSupportDegree": local_degree,
        "edgeFirstEndpoint": first_endpoint[retained_edge].astype(np.int32),
        "edgeSecondEndpoint": second_endpoint[retained_edge].astype(np.int32),
        "edgeAffinity": affinity[retained_edge],
        "edgeKind": np.where(
            local_link[retained_edge], 0, 1
        ).astype(np.uint8),
        "pairEdgeLowerRetained": retained_edge[:pair_edge_count],
        "pairEdgeUpperRetained": retained_edge[pair_edge_count:],
    }
    retained_local = retained_edge & local_link
    retained_long = retained_edge & ~local_link
    summary = {
        "settings": resolved.record(),
        "counts": {
            "profileCount": int(profile_count),
            "endpointCount": int(2 * profile_count),
            "inputPairEdgeCount": int(pair_edge_count),
            "inputEndpointEdgeCount": int(len(first_endpoint)),
            "localSupportEdgeCount": int(np.count_nonzero(local_support_edge)),
            "strongLocalProposalCount": int(
                np.count_nonzero(strong_edge & local_link)
            ),
            "supportedLongProposalCount": int(
                np.count_nonzero(supported_long_edge)
            ),
            "retainedEndpointEdgeCount": int(np.count_nonzero(retained_edge)),
            "retainedLocalEdgeCount": int(np.count_nonzero(retained_local)),
            "retainedSupportedLongEdgeCount": int(
                np.count_nonzero(retained_long)
            ),
            "componentCount": int(len(component_count)),
            "componentsAtLeast32Endpoints": int(
                np.count_nonzero(component_count >= 32)
            ),
            "componentsAtLeast128Endpoints": int(
                np.count_nonzero(component_count >= 128)
            ),
            "largestComponentSizes": [int(value) for value in component_count[:32]],
            "profileFaceCollisionCount": int(np.count_nonzero(profile_collision)),
        },
        "localSupportDegree": _percentile_record(local_degree),
        "retainedEdgeAffinity": _percentile_record(affinity[retained_edge]),
        "collisionGuard": collision_summary,
    }
    return arrays, summary
