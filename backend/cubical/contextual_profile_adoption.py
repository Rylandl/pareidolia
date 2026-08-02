from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np

from .one_sided_interface import (
    OneSidedInterfaceSettings,
    match_signed_interface_endpoints,
)


@dataclass(frozen=True, slots=True)
class ContextualProfileAdoptionSettings:
    """Controls for assigning unused physical profiles to known sheet faces."""

    maximum_endpoint_position_residual_sampling_steps: float = 0.75
    maximum_endpoint_normal_degrees: float = 15.0
    endpoint_normal_cost_scale_degrees: float = 15.0
    minimum_local_evidence: float = 0.35
    thickness_tolerance_sampling_steps: float = 0.5

    def __post_init__(self) -> None:
        positive = (
            self.maximum_endpoint_position_residual_sampling_steps,
            self.maximum_endpoint_normal_degrees,
            self.endpoint_normal_cost_scale_degrees,
            self.thickness_tolerance_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("contextual profile scales must be finite and positive")
        if not 0.0 < self.maximum_endpoint_normal_degrees < 90.0:
            raise ValueError("contextual endpoint normal cap must lie in (0, 90)")
        if not math.isfinite(self.minimum_local_evidence):
            raise ValueError("contextual profile evidence must be finite")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _required(
    values: Mapping[str, np.ndarray],
    name: str,
) -> np.ndarray:
    if name not in values:
        raise ValueError(f"contextual profile adoption requires {name}")
    return np.asarray(values[name])


def adopt_contextual_profiles(
    bank: Mapping[str, np.ndarray],
    growth: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    *,
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    sampling_stride_voxels: int,
    settings: ContextualProfileAdoptionSettings | None = None,
) -> dict[str, Any]:
    """Adopt unused two-crossing profiles from exact physical-face support.

    Contextual profiles never create a sheet identity. At least one signed CT
    endpoint must match a face that already carries an immutable physical
    ``(sheet, side)`` identity. If both endpoints match, their identities and
    canonical orientation must agree. The candidate must also lie inside the
    trusted selected profiles' per-sheet thickness range. Finally, the
    original selected profile always owns its sampling-lattice key; otherwise
    one highest-supported contextual candidate may own the key.
    """

    resolved = settings or ContextualProfileAdoptionSettings()
    lower_xyz = np.asarray(_required(bank, "boundaryLowerXYZ"), dtype=np.float32)
    upper_xyz = np.asarray(_required(bank, "boundaryUpperXYZ"), dtype=np.float32)
    profile_normal = np.asarray(_required(bank, "normalXYZ"), dtype=np.float32)
    thickness = np.asarray(_required(bank, "thicknessVoxels"), dtype=np.float64)
    evidence = np.asarray(_required(bank, "localEvidenceScore"), dtype=np.float64)
    spatial_key = np.asarray(_required(bank, "spatialKeyXYZ"), dtype=np.int64)
    candidate_count = len(lower_xyz)
    if any(
        len(value) != candidate_count
        for value in (
            upper_xyz,
            profile_normal,
            thickness,
            evidence,
            spatial_key,
        )
    ):
        raise ValueError("contextual profile candidate arrays are not aligned")
    if spatial_key.shape != (candidate_count, 3):
        raise ValueError("contextual profile spatial keys must have shape (N, 3)")

    originally_selected = np.asarray(
        _required(growth, "selected"), dtype=bool
    )
    selected_label = np.asarray(
        _required(growth, "selectedLabel"), dtype=np.int32
    )
    if len(originally_selected) != candidate_count or len(selected_label) != candidate_count:
        raise ValueError("contextual profile growth arrays are not candidate aligned")
    selected_keys = spatial_key[originally_selected]
    if len(selected_keys) and len(np.unique(selected_keys, axis=0)) != len(selected_keys):
        raise ValueError("selected physical profiles must be key exclusive")

    interface_position = np.asarray(
        _required(interfaces, "positionXYZ"), dtype=np.float32
    )
    interface_normal = np.asarray(
        _required(interfaces, "signedNormalXYZ"), dtype=np.float32
    )
    interface_key = np.asarray(
        _required(interfaces, "processingKeyXYZ"), dtype=np.int32
    )
    surface_interface = np.asarray(
        _required(surface, "interfaceIndex"), dtype=np.int64
    )
    surface_label = np.asarray(
        _required(surface, "physicalSheetLabel"), dtype=np.int32
    )
    surface_side = np.asarray(
        _required(surface, "physicalBoundarySide"), dtype=np.uint8
    )
    if any(
        len(value) != len(surface_interface)
        for value in (surface_label, surface_side)
    ):
        raise ValueError("contextual material-surface identities are not aligned")
    if np.any(surface_interface < 0) or np.any(surface_interface >= len(interface_position)):
        raise ValueError("material surface references an unavailable interface")
    if len(np.unique(surface_interface)) != len(surface_interface):
        raise ValueError("material surface contains a repeated interface sample")

    matcher_settings = OneSidedInterfaceSettings(
        maximum_seed_position_residual_sampling_steps=(
            resolved.maximum_endpoint_position_residual_sampling_steps
        ),
        maximum_seed_normal_degrees=resolved.maximum_endpoint_normal_degrees,
        seed_match_normal_scale_degrees=(
            resolved.endpoint_normal_cost_scale_degrees
        ),
    )
    endpoint_match = match_signed_interface_endpoints(
        np.concatenate((lower_xyz, upper_xyz), axis=0),
        np.concatenate((profile_normal, -profile_normal), axis=0),
        interface_position,
        interface_normal,
        interface_key,
        processing_start_xyz=np.asarray(processing_start_xyz, dtype=np.float64),
        source_origin_xyz=np.asarray(source_origin_xyz, dtype=np.float64),
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        stride=int(sampling_stride_voxels),
        settings=matcher_settings,
    )
    interface_to_surface = np.full(len(interface_position), -1, dtype=np.int32)
    interface_to_surface[surface_interface] = np.arange(
        len(surface_interface), dtype=np.int32
    )
    endpoint_interface = np.asarray(
        endpoint_match["interfaceIndex"], dtype=np.int32
    )
    endpoint_surface = np.full(len(endpoint_interface), -1, dtype=np.int32)
    matched_interface = endpoint_interface >= 0
    endpoint_surface[matched_interface] = interface_to_surface[
        endpoint_interface[matched_interface]
    ]
    lower_surface = endpoint_surface[:candidate_count]
    upper_surface = endpoint_surface[candidate_count:]
    lower_cost = np.asarray(endpoint_match["matchCost"], dtype=np.float32)[
        :candidate_count
    ]
    upper_cost = np.asarray(endpoint_match["matchCost"], dtype=np.float32)[
        candidate_count:
    ]

    def endpoint_identity(
        node: np.ndarray,
        *,
        canonical_side: int,
    ) -> tuple[np.ndarray, np.ndarray]:
        label = np.full(candidate_count, -1, dtype=np.int32)
        orientation = np.zeros(candidate_count, dtype=np.int8)
        available = node >= 0
        safe = np.maximum(node, 0)
        available &= surface_label[safe] >= 0
        label[available] = surface_label[safe[available]]
        orientation[available] = np.where(
            surface_side[safe[available]] == canonical_side,
            1,
            -1,
        ).astype(np.int8)
        return label, orientation

    lower_label, lower_orientation = endpoint_identity(
        lower_surface, canonical_side=0
    )
    upper_label, upper_orientation = endpoint_identity(
        upper_surface, canonical_side=1
    )
    lower_available = lower_label >= 0
    upper_available = upper_label >= 0
    conflict = lower_available & upper_available & (
        (lower_label != upper_label) | (lower_orientation != upper_orientation)
    )
    contextual_label = np.where(
        lower_available, lower_label, upper_label
    ).astype(np.int32)
    contextual_orientation = np.where(
        lower_available, lower_orientation, upper_orientation
    ).astype(np.int8)
    contextual_label[conflict] = -1
    contextual_orientation[conflict] = 0
    endpoint_support = (
        lower_available.astype(np.uint8) + upper_available.astype(np.uint8)
    )

    thickness_supported = np.zeros(candidate_count, dtype=bool)
    tolerance = (
        resolved.thickness_tolerance_sampling_steps
        * float(sampling_stride_voxels)
    )
    selected_values = np.unique(selected_label[originally_selected])
    for sheet_label in selected_values[selected_values >= 0]:
        trusted = thickness[
            originally_selected & (selected_label == sheet_label)
        ]
        trusted = trusted[np.isfinite(trusted) & (trusted > 0.0)]
        if not len(trusted):
            continue
        lower_bound, upper_bound = np.percentile(trusted, (1.0, 99.0))
        thickness_supported |= (
            (contextual_label == sheet_label)
            & (thickness >= max(0.0, lower_bound - tolerance))
            & (thickness <= upper_bound + tolerance)
        )
    contextual = (
        (contextual_label >= 0)
        & ~conflict
        & thickness_supported
        & np.isfinite(evidence)
        & (evidence >= resolved.minimum_local_evidence)
    )

    eligible = originally_selected | contextual
    eligible_index = np.flatnonzero(eligible)
    chosen = np.zeros(candidate_count, dtype=bool)
    if len(eligible_index):
        support_cost = np.where(
            lower_available & upper_available,
            0.5 * (lower_cost + upper_cost),
            np.where(lower_available, lower_cost, upper_cost),
        )
        key = spatial_key[eligible_index]
        order = np.lexsort(
            (
                eligible_index,
                support_cost[eligible_index],
                -evidence[eligible_index],
                -endpoint_support[eligible_index].astype(np.int16),
                ~originally_selected[eligible_index],
                key[:, 2],
                key[:, 1],
                key[:, 0],
            )
        )
        ranked = eligible_index[order]
        ranked_key = spatial_key[ranked]
        first = np.concatenate(
            (
                np.ones(1, dtype=bool),
                np.any(ranked_key[1:] != ranked_key[:-1], axis=1),
            )
        )
        chosen[ranked[first]] = True

    physical_label = np.full(candidate_count, -1, dtype=np.int32)
    canonical_orientation = np.zeros(candidate_count, dtype=np.int8)
    physical_label[chosen] = contextual_label[chosen]
    canonical_orientation[chosen] = contextual_orientation[chosen]
    trusted_choice = chosen & originally_selected
    physical_label[trusted_choice] = selected_label[trusted_choice]
    canonical_orientation[trusted_choice] = 1
    adopted = chosen & ~originally_selected
    canonical_lower_surface = np.where(
        canonical_orientation >= 0, lower_surface, upper_surface
    ).astype(np.int32)
    canonical_upper_surface = np.where(
        canonical_orientation >= 0, upper_surface, lower_surface
    ).astype(np.int32)

    return {
        "selected": chosen,
        "originallySelected": originally_selected,
        "adopted": adopted,
        "physicalSheetLabel": physical_label,
        "canonicalOrientation": canonical_orientation,
        "endpointSupportCount": endpoint_support,
        "canonicalLowerSurfaceNode": canonical_lower_surface,
        "canonicalUpperSurfaceNode": canonical_upper_surface,
        "lowerEndpointMatchCost": lower_cost,
        "upperEndpointMatchCost": upper_cost,
        "summary": {
            "candidateCount": int(candidate_count),
            "originalSelectedProfileCount": int(
                np.count_nonzero(originally_selected)
            ),
            "contextuallyAdmissibleProfileCount": int(
                np.count_nonzero(contextual)
            ),
            "contextualEndpointIdentityConflictCount": int(
                np.count_nonzero(conflict)
            ),
            "selectedProfileCount": int(np.count_nonzero(chosen)),
            "adoptedContextualProfileCount": int(np.count_nonzero(adopted)),
            "adoptedWithTwoMatchedEndpointsCount": int(
                np.count_nonzero(adopted & (endpoint_support == 2))
            ),
            "adoptedWithOneMatchedEndpointCount": int(
                np.count_nonzero(adopted & (endpoint_support == 1))
            ),
            "adoptedCanonicalOrientationCount": int(
                np.count_nonzero(adopted & (canonical_orientation == 1))
            ),
            "adoptedReversedOrientationCount": int(
                np.count_nonzero(adopted & (canonical_orientation == -1))
            ),
        },
    }
