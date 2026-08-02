from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import _percentile_record
from .laminar_ribbon import _macro_profile_assignment
from .macro_orientation import MACRO_ORIENTATION_SCHEMA, MACRO_ORIENTATION_STEM
from .material_surface_graph import (
    MaterialSurfaceGraphSettings,
    _collision_safe_components,
    _tangent_columns,
    write_material_surface_cross_sections,
)
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA, PAIRED_SURFACE_BANK_STEM
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PAIRED_SURFACE_GROWTH_STEM,
)
from .paired_boundary_tracks import (
    PairedBoundaryTrackSettings,
    build_paired_boundary_tracks,
)
from .paired_endpoint_graph import build_paired_endpoint_continuity_graph
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
    PHYSICAL_MID_SURFACE_VERSION,
    _components,
    _write_npz,
)


PAIRED_PROFILE_SURFACE_SCHEMA = "pareidolia.direct-paired-profile-surface"
PAIRED_PROFILE_SURFACE_VERSION = 1


@dataclass(frozen=True, slots=True)
class PairedProfileSurfaceSettings:
    """Build sheets directly from immutable two-boundary CT profiles.

    Candidate ownership remains the key-exclusive decision made by contextual
    paired growth.  This stage deliberately discards its inherited seed labels
    and reconnects selected profiles from their physical two-face geometry.
    The generic macro tensor is an independent hard guard against coherent but
    transverse profile families.
    """

    candidate_selection_mode: str = "endpoint-coordinate-ascent"
    minimum_macro_confidence: float = 0.35
    maximum_profile_to_macro_normal_degrees: float = 25.0
    minimum_edge_affinity: float = 0.4
    maximum_edge_normal_degrees: float = 30.0
    maximum_midpoint_height_sampling_steps: float = 1.0
    maximum_boundary_height_sampling_steps: float = 1.5
    maximum_thickness_difference_sampling_steps: float = 1.5
    enable_geometric_closure: bool = True
    maximum_closure_distance_sampling_steps: float = 5.0
    minimum_closure_affinity: float = 0.35
    candidate_base_reward: float = 0.25
    candidate_local_evidence_reward: float = 1.0
    candidate_conservative_bonus: float = 0.35
    candidate_seed_bonus: float = 0.15
    continuity_reward: float = 1.5
    maximum_selection_sweeps: int = 20
    endpoint_link_radius_sampling_steps: float = 5.0
    maximum_endpoint_normal_degrees: float = 30.0
    maximum_endpoint_distance_sampling_steps: float = 6.0
    maximum_endpoint_height_sampling_steps: float = 1.5
    endpoint_normal_scale_degrees: float = 20.0
    endpoint_height_scale_sampling_steps: float = 0.75
    endpoint_distance_scale_sampling_steps: float = 3.0
    minimum_endpoint_affinity: float = 0.1
    endpoint_strong_face_weight: float = 0.45
    endpoint_weak_face_weight: float = 0.25
    endpoint_paired_face_weight: float = 0.3
    enable_boundary_tracks: bool = True
    boundary_track_minimum_affinity: float = 0.45
    boundary_track_local_support_radius_sampling_steps: float = math.sqrt(5.0)
    boundary_track_minimum_local_support_affinity: float = 0.2
    boundary_track_minimum_local_support_degree: int = 2
    boundary_track_tangent_column_width_sampling_steps: float = 1.5
    boundary_track_maximum_column_depth_range_sampling_steps: float = 2.25
    # Frontier bundling remains an explicit experimental solver.  The current
    # truth control favors sign-correct connected geometry, so production does
    # not silently opt into the more fragmented experimental partition.
    component_solver_mode: str = "connected-components"
    minimum_core_edge_affinity: float = 0.7
    minimum_frontier_bundle_edges: int = 3
    minimum_frontier_unique_endpoints_per_side: int = 2
    minimum_frontier_span_sampling_steps: float = 2.0
    minimum_frontier_direction_coherence: float = 0.45
    minimum_frontier_median_affinity: float = 0.45
    enable_tangent_column_guard: bool = True
    tangent_column_width_sampling_steps: float = 1.5
    maximum_column_depth_range_sampling_steps: float = 2.25
    minimum_component_nodes_for_preview: int = 8
    maximum_preview_components: int = 128

    def __post_init__(self) -> None:
        if self.candidate_selection_mode not in (
            "growth",
            "coordinate-ascent",
            "endpoint-coordinate-ascent",
        ):
            raise ValueError(
                "candidate selection mode must be growth, coordinate-ascent, "
                "or endpoint-coordinate-ascent"
            )
        if self.component_solver_mode not in ("connected-components", "frontier-bundles"):
            raise ValueError(
                "component solver mode must be connected-components or frontier-bundles"
            )
        for value, name in (
            (self.minimum_macro_confidence, "macro confidence"),
            (self.minimum_edge_affinity, "edge affinity"),
            (self.minimum_closure_affinity, "closure affinity"),
            (self.minimum_endpoint_affinity, "minimum endpoint affinity"),
            (
                self.boundary_track_minimum_affinity,
                "boundary track affinity",
            ),
            (
                self.boundary_track_minimum_local_support_affinity,
                "boundary local support affinity",
            ),
            (
                self.minimum_core_edge_affinity,
                "minimum core edge affinity",
            ),
            (
                self.minimum_frontier_direction_coherence,
                "minimum frontier direction coherence",
            ),
            (
                self.minimum_frontier_median_affinity,
                "minimum frontier median affinity",
            ),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for value in (
            self.maximum_profile_to_macro_normal_degrees,
            self.maximum_edge_normal_degrees,
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise ValueError("paired-profile angle gates must lie in (0, 90)")
        for value in (
            self.maximum_midpoint_height_sampling_steps,
            self.maximum_boundary_height_sampling_steps,
            self.maximum_thickness_difference_sampling_steps,
            self.maximum_closure_distance_sampling_steps,
            self.candidate_base_reward,
            self.candidate_local_evidence_reward,
            self.continuity_reward,
            self.tangent_column_width_sampling_steps,
            self.maximum_column_depth_range_sampling_steps,
            self.minimum_frontier_span_sampling_steps,
            self.endpoint_link_radius_sampling_steps,
            self.maximum_endpoint_distance_sampling_steps,
            self.maximum_endpoint_height_sampling_steps,
            self.endpoint_normal_scale_degrees,
            self.endpoint_height_scale_sampling_steps,
            self.endpoint_distance_scale_sampling_steps,
            self.boundary_track_local_support_radius_sampling_steps,
            self.boundary_track_tangent_column_width_sampling_steps,
            self.boundary_track_maximum_column_depth_range_sampling_steps,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("paired-profile geometry gates must be positive")
        if self.candidate_conservative_bonus < 0.0 or self.candidate_seed_bonus < 0.0:
            raise ValueError("paired-profile candidate bonuses must be nonnegative")
        if not 0.0 < self.maximum_endpoint_normal_degrees < 90.0:
            raise ValueError("endpoint normal gate must lie in (0, 90)")
        endpoint_weight = (
            self.endpoint_strong_face_weight,
            self.endpoint_weak_face_weight,
            self.endpoint_paired_face_weight,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in endpoint_weight):
            raise ValueError("endpoint selection weights must be finite and nonnegative")
        if sum(endpoint_weight) <= 0.0:
            raise ValueError("at least one endpoint selection weight must be positive")
        if self.boundary_track_minimum_local_support_degree < 1:
            raise ValueError("boundary local support degree must be positive")
        if (
            self.minimum_component_nodes_for_preview < 1
            or self.maximum_preview_components < 1
            or self.maximum_selection_sweeps < 1
            or self.minimum_frontier_bundle_edges < 2
            or self.minimum_frontier_unique_endpoints_per_side < 2
        ):
            raise ValueError("paired-profile preview counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _frontier_bundle_connectivity(
    midpoint_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    score: np.ndarray,
    edge_kind: np.ndarray,
    *,
    sampling_stride_voxels: float,
    settings: PairedProfileSurfaceSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Select component-changing edges only from correlated frontiers.

    A dense sheet can contain many individually plausible local continuations,
    but one such edge is not evidence that two already coherent fragments are
    the same physical layer.  Strong, observed short-range edges establish
    local cores.  We then aggregate every weaker/longer proposal by the pair of
    cores it would join and require repeated support across a spatially
    extended frontier before allowing that join.
    """

    node_count = len(midpoint_xyz)
    edge_count = len(first)
    if not edge_count:
        return np.zeros(0, dtype=bool), {
            "mode": "frontier-bundles",
            "coreComponentCount": node_count,
            "coreEdgeCount": 0,
            "candidateBundleCount": 0,
            "acceptedBundleCount": 0,
            "acceptedBundleEdgeCount": 0,
        }
    values = np.asarray(score, dtype=np.float64)
    kind = np.asarray(edge_kind, dtype=np.uint8)
    # Kind 6 is an observed short-range continuation from the original paired
    # bank.  Kind 7 is a geometric gap proposal and can only act through a
    # supported frontier bundle.
    core_edge = (kind == 6) & (values >= settings.minimum_core_edge_affinity)
    core_component, core_count = _components(
        node_count, first[core_edge], second[core_edge]
    )
    first_component = core_component[first]
    second_component = core_component[second]
    cross_core = first_component != second_component
    selected = core_edge.copy()
    if not np.any(cross_core):
        # Keep redundant geometry inside the already established cores.
        selected |= first_component == second_component
        return selected, {
            "mode": "frontier-bundles",
            "coreComponentCount": int(len(core_count)),
            "coreEdgeCount": int(np.count_nonzero(core_edge)),
            "candidateBundleCount": 0,
            "acceptedBundleCount": 0,
            "acceptedBundleEdgeCount": 0,
        }

    cross_index = np.flatnonzero(cross_core)
    left_component = np.minimum(
        first_component[cross_index], second_component[cross_index]
    )
    right_component = np.maximum(
        first_component[cross_index], second_component[cross_index]
    )
    pair = np.column_stack((left_component, right_component))
    order = np.lexsort((pair[:, 1], pair[:, 0]))
    ordered_pair = pair[order]
    start_mask = np.ones(len(order), dtype=bool)
    start_mask[1:] = np.any(ordered_pair[1:] != ordered_pair[:-1], axis=1)
    starts = np.flatnonzero(start_mask)
    stops = np.r_[starts[1:], len(order)]
    midpoint = np.asarray(midpoint_xyz, dtype=np.float64)
    stride = float(sampling_stride_voxels)
    accepted_bundle_count = 0
    accepted_bundle_edges = 0
    rejected_support = 0
    rejected_span = 0
    rejected_direction = 0
    rejected_affinity = 0
    accepted_records: list[dict[str, Any]] = []
    for start, stop in zip(starts, stops):
        member = cross_index[order[start:stop]]
        pair_left = int(ordered_pair[start, 0])
        # Orient every bridge consistently from the lower-ranked core to the
        # higher-ranked core before measuring its directional agreement.
        first_is_left = first_component[member] == pair_left
        left_node = np.where(first_is_left, first[member], second[member])
        right_node = np.where(first_is_left, second[member], first[member])
        unique_left = np.unique(left_node)
        unique_right = np.unique(right_node)
        if (
            len(member) < settings.minimum_frontier_bundle_edges
            or len(unique_left)
            < settings.minimum_frontier_unique_endpoints_per_side
            or len(unique_right)
            < settings.minimum_frontier_unique_endpoints_per_side
        ):
            rejected_support += 1
            continue
        bridge_center = 0.5 * (midpoint[left_node] + midpoint[right_node])
        span = float(np.linalg.norm(np.ptp(bridge_center, axis=0))) / stride
        if span < settings.minimum_frontier_span_sampling_steps:
            rejected_span += 1
            continue
        direction = midpoint[right_node] - midpoint[left_node]
        direction /= np.maximum(
            np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-9
        )
        coherence = float(np.linalg.norm(np.mean(direction, axis=0)))
        if coherence < settings.minimum_frontier_direction_coherence:
            rejected_direction += 1
            continue
        median_affinity = float(np.median(values[member]))
        if median_affinity < settings.minimum_frontier_median_affinity:
            rejected_affinity += 1
            continue
        selected[member] = True
        accepted_bundle_count += 1
        accepted_bundle_edges += len(member)
        if len(accepted_records) < 32:
            accepted_records.append(
                {
                    "firstCoreComponent": pair_left,
                    "secondCoreComponent": int(ordered_pair[start, 1]),
                    "edgeCount": int(len(member)),
                    "uniqueFirstEndpointCount": int(len(unique_left)),
                    "uniqueSecondEndpointCount": int(len(unique_right)),
                    "spanSamplingSteps": round(span, 6),
                    "directionCoherence": round(coherence, 6),
                    "medianAffinity": round(median_affinity, 6),
                }
            )
    # Redundant edges inside a core are safe and useful to downstream mesh
    # construction even when they were not strong enough to establish it.
    selected |= first_component == second_component
    return selected, {
        "mode": "frontier-bundles",
        "coreComponentCount": int(len(core_count)),
        "coreEdgeCount": int(np.count_nonzero(core_edge)),
        "candidateBundleCount": int(len(starts)),
        "acceptedBundleCount": int(accepted_bundle_count),
        "acceptedBundleEdgeCount": int(accepted_bundle_edges),
        "rejectedBundleCount": {
            "support": int(rejected_support),
            "span": int(rejected_span),
            "direction": int(rejected_direction),
            "affinity": int(rejected_affinity),
        },
        "acceptedBundleExamples": accepted_records,
    }


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _load(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(path.read_text())
    if manifest.get("state") != "complete":
        raise ValueError(f"paired-profile input is incomplete: {path}")
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"paired-profile input data changed: {path}")
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return manifest, arrays


def _direct_geometric_closures(
    midpoint_xyz: np.ndarray,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    thickness_voxels: np.ndarray,
    spatial_key_xyz: np.ndarray,
    existing_first: np.ndarray,
    existing_second: np.ndarray,
    *,
    sampling_stride_voxels: float,
    settings: PairedProfileSurfaceSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Bridge missing lattice samples only when both physical faces continue."""

    if not settings.enable_geometric_closure or not len(midpoint_xyz):
        empty = np.empty(0, dtype=np.int32)
        return empty, empty.copy(), np.empty(0, dtype=np.float32), {
            "considered": 0,
            "retained": 0,
        }
    midpoint = np.asarray(midpoint_xyz, dtype=np.float64)
    lower = np.asarray(lower_xyz, dtype=np.float64)
    upper = np.asarray(upper_xyz, dtype=np.float64)
    normal = np.asarray(normal_xyz, dtype=np.float64)
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    thickness = np.asarray(thickness_voxels, dtype=np.float64)
    key = np.asarray(spatial_key_xyz, dtype=np.int64)
    key_to_node = {tuple(int(item) for item in value): index for index, value in enumerate(key)}
    maximum_offset = int(math.ceil(settings.maximum_closure_distance_sampling_steps))
    offsets: list[tuple[int, int, int]] = []
    for dx in range(-maximum_offset, maximum_offset + 1):
        for dy in range(-maximum_offset, maximum_offset + 1):
            for dz in range(-maximum_offset, maximum_offset + 1):
                if (dx, dy, dz) <= (0, 0, 0):
                    continue
                if math.sqrt(dx * dx + dy * dy + dz * dz) > (
                    settings.maximum_closure_distance_sampling_steps + 1.0e-8
                ):
                    continue
                offsets.append((dx, dy, dz))
    existing = {
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in zip(existing_first, existing_second)
    }
    closure_first: list[int] = []
    closure_second: list[int] = []
    closure_score: list[float] = []
    considered = 0
    stride = float(sampling_stride_voxels)
    for first_node, first_key in enumerate(key):
        for offset in offsets:
            second_node = key_to_node.get(
                (
                    int(first_key[0]) + offset[0],
                    int(first_key[1]) + offset[1],
                    int(first_key[2]) + offset[2],
                )
            )
            if second_node is None:
                continue
            pair = (
                min(first_node, second_node),
                max(first_node, second_node),
            )
            if pair in existing:
                continue
            considered += 1
            signed_cosine = float(np.dot(normal[first_node], normal[second_node]))
            cosine = abs(signed_cosine)
            normal_degrees = math.degrees(math.acos(float(np.clip(cosine, 0.0, 1.0))))
            if normal_degrees > settings.maximum_edge_normal_degrees:
                continue
            aligned_second_normal = (
                normal[second_node]
                if signed_cosine >= 0.0
                else -normal[second_node]
            )
            average_normal = normal[first_node] + aligned_second_normal
            average_normal /= max(float(np.linalg.norm(average_normal)), 1.0e-12)
            delta = midpoint[second_node] - midpoint[first_node]
            distance = float(np.linalg.norm(delta)) / stride
            if distance > settings.maximum_closure_distance_sampling_steps:
                continue
            midpoint_height = abs(float(np.dot(delta, average_normal))) / stride
            if midpoint_height > settings.maximum_midpoint_height_sampling_steps:
                continue
            second_lower = (
                lower[second_node] if signed_cosine >= 0.0 else upper[second_node]
            )
            second_upper = (
                upper[second_node] if signed_cosine >= 0.0 else lower[second_node]
            )
            lower_delta = second_lower - lower[first_node]
            upper_delta = second_upper - upper[first_node]
            boundary_height = max(
                abs(float(np.dot(lower_delta, average_normal))),
                abs(float(np.dot(upper_delta, average_normal))),
            ) / stride
            if boundary_height > settings.maximum_boundary_height_sampling_steps:
                continue
            thickness_difference = abs(thickness[second_node] - thickness[first_node]) / stride
            if (
                thickness_difference
                > settings.maximum_thickness_difference_sampling_steps
            ):
                continue
            quality = (
                math.exp(-((normal_degrees / 15.0) ** 2))
                * math.exp(-((midpoint_height / 0.75) ** 2))
                * math.exp(-((boundary_height / 1.0) ** 2))
                * math.exp(-((thickness_difference / 1.5) ** 2))
            ) ** 0.25
            quality *= math.exp(
                -0.25
                * (
                    distance
                    / settings.maximum_closure_distance_sampling_steps
                )
                ** 2
            )
            if quality < settings.minimum_closure_affinity:
                continue
            closure_first.append(pair[0])
            closure_second.append(pair[1])
            closure_score.append(quality)
    return (
        np.asarray(closure_first, dtype=np.int32),
        np.asarray(closure_second, dtype=np.int32),
        np.asarray(closure_score, dtype=np.float32),
        {"considered": int(considered), "retained": int(len(closure_first))},
    )


def _optimize_candidate_choices(
    bank: Mapping[str, np.ndarray],
    eligible_candidate: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_affinity: np.ndarray,
    *,
    settings: PairedProfileSurfaceSettings,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Coordinate-ascent selection with one candidate per spatial key.

    Only ambiguous keys need iterative work.  Each update evaluates the exact
    change in unary plus incident pairwise rewards, so the objective is
    monotone and the method scales linearly with the retained continuity graph.
    """

    eligible = np.asarray(eligible_candidate, dtype=bool)
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int64)
    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float64)
    conservative = np.asarray(bank["isolatedConservative"], dtype=bool)
    seeded = np.asarray(bank["seedComponentId"], dtype=np.int32) >= 0
    candidate_count = len(key)
    unary = (
        settings.candidate_base_reward
        + settings.candidate_local_evidence_reward * np.clip(evidence, 0.0, 1.0)
        + settings.candidate_conservative_bonus * conservative.astype(np.float64)
        + settings.candidate_seed_bonus * seeded.astype(np.float64)
    )
    unary[~eligible] = -np.inf
    order = np.lexsort((np.arange(candidate_count), key[:, 2], key[:, 1], key[:, 0]))
    ordered_key = key[order]
    start_mask = np.ones(candidate_count, dtype=bool)
    start_mask[1:] = np.any(ordered_key[1:] != ordered_key[:-1], axis=1)
    group_start = np.flatnonzero(start_mask)
    group_stop = np.r_[group_start[1:], candidate_count]
    groups = [order[start:stop] for start, stop in zip(group_start, group_stop)]
    eligible_group = [group[np.isfinite(unary[group])] for group in groups]

    directed_first = np.concatenate((edge_first, edge_second)).astype(np.int32)
    directed_second = np.concatenate((edge_second, edge_first)).astype(np.int32)
    directed_weight = np.concatenate((edge_affinity, edge_affinity)).astype(np.float64)
    adjacency_order = np.argsort(directed_first, kind="stable")
    directed_first = directed_first[adjacency_order]
    directed_second = directed_second[adjacency_order]
    directed_weight = directed_weight[adjacency_order]
    adjacency_count = np.bincount(directed_first, minlength=candidate_count)
    adjacency_offset = np.r_[0, np.cumsum(adjacency_count)]

    selected = np.zeros(candidate_count, dtype=bool)
    selected_candidate = np.full(len(groups), -1, dtype=np.int32)
    ambiguous_group: list[int] = []
    for group_index, group in enumerate(eligible_group):
        if not len(group):
            continue
        best = int(group[np.argmax(unary[group])])
        selected[best] = True
        selected_candidate[group_index] = best
        if len(group) > 1:
            ambiguous_group.append(group_index)

    change_per_sweep: list[int] = []
    for _sweep in range(settings.maximum_selection_sweeps):
        changes = 0
        for group_index in ambiguous_group:
            group = eligible_group[group_index]
            current = int(selected_candidate[group_index])
            best = current
            best_score = -math.inf
            for candidate in group:
                start = int(adjacency_offset[candidate])
                stop = int(adjacency_offset[candidate + 1])
                support = float(
                    np.sum(
                        directed_weight[start:stop]
                        * selected[directed_second[start:stop]]
                    )
                )
                score = float(unary[candidate]) + settings.continuity_reward * support
                if score > best_score + 1.0e-12 or (
                    abs(score - best_score) <= 1.0e-12 and int(candidate) < best
                ):
                    best = int(candidate)
                    best_score = score
            if best == current:
                continue
            selected[current] = False
            selected[best] = True
            selected_candidate[group_index] = best
            changes += 1
        change_per_sweep.append(changes)
        if not changes:
            break

    selected_edge = selected[edge_first] & selected[edge_second]
    objective = float(np.sum(unary[selected])) + settings.continuity_reward * float(
        np.sum(edge_affinity[selected_edge])
    )
    return selected, {
        "mode": "coordinate-ascent",
        "spatialKeyCount": int(len(groups)),
        "eligibleSpatialKeyCount": int(np.count_nonzero(selected_candidate >= 0)),
        "ambiguousSpatialKeyCount": int(len(ambiguous_group)),
        "selectedCandidateCount": int(np.count_nonzero(selected)),
        "sweeps": int(len(change_per_sweep)),
        "changesPerSweep": change_per_sweep,
        "converged": bool(change_per_sweep and change_per_sweep[-1] == 0),
        "objective": round(objective, 8),
        "selectedContinuityEdgeCount": int(np.count_nonzero(selected_edge)),
    }


def build_direct_paired_profile_surface(
    bank: Mapping[str, np.ndarray],
    growth: Mapping[str, np.ndarray],
    macro_manifest: Mapping[str, Any],
    macro: Mapping[str, np.ndarray],
    *,
    sampling_stride_voxels: float = 2.0,
    settings: PairedProfileSurfaceSettings | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Reconnect selected physical profiles without inherited identities."""

    resolved = settings or PairedProfileSurfaceSettings()
    candidate_count = len(bank["midpointXYZ"])
    growth_selected = np.asarray(growth["selected"], dtype=bool)
    if len(growth_selected) != candidate_count:
        raise ValueError("paired bank and growth selection are not aligned")
    macro_assignment = _macro_profile_assignment(
        np.asarray(bank["spatialKeyXYZ"]),
        np.asarray(bank["normalXYZ"]),
        macro_manifest,
        macro,
    )
    macro_eligible = (
        np.asarray(macro_assignment["macroTrusted"], dtype=bool)
        & (
            np.asarray(macro_assignment["macroConfidence"], dtype=np.float64)
            >= resolved.minimum_macro_confidence
        )
        & (
            np.asarray(
                macro_assignment["macroNormalResidualDegrees"], dtype=np.float64
            )
            <= resolved.maximum_profile_to_macro_normal_degrees
        )
    )
    source_first = np.asarray(growth["edgeFirstCandidate"], dtype=np.int32)
    source_second = np.asarray(growth["edgeSecondCandidate"], dtype=np.int32)
    affinity = np.asarray(growth["edgeAffinity"], dtype=np.float32)
    edge_normal = np.asarray(growth["edgeNormalDegrees"], dtype=np.float32)
    midpoint_height = np.asarray(
        growth["edgeMidpointHeightSamplingSteps"], dtype=np.float32
    )
    boundary_height = np.asarray(
        growth["edgeBoundaryHeightSamplingSteps"], dtype=np.float32
    )
    thickness_difference = np.asarray(
        growth["edgeThicknessDifferenceSamplingSteps"], dtype=np.float32
    )
    eligible_edge = (
        macro_eligible[source_first]
        & macro_eligible[source_second]
        & (affinity >= resolved.minimum_edge_affinity)
        & (edge_normal <= resolved.maximum_edge_normal_degrees)
        & (
            midpoint_height
            <= resolved.maximum_midpoint_height_sampling_steps
        )
        & (boundary_height <= resolved.maximum_boundary_height_sampling_steps)
        & (
            thickness_difference
            <= resolved.maximum_thickness_difference_sampling_steps
        )
    )
    endpoint_graph: dict[str, np.ndarray] = {
        "firstCandidate": np.empty(0, dtype=np.int32),
        "secondCandidate": np.empty(0, dtype=np.int32),
        "secondOrientation": np.empty(0, dtype=np.int8),
        "lowerAffinity": np.empty(0, dtype=np.float32),
        "upperAffinity": np.empty(0, dtype=np.float32),
    }
    endpoint_summary: dict[str, Any] = {"state": "not-requested"}
    if (
        resolved.candidate_selection_mode == "endpoint-coordinate-ascent"
        or resolved.enable_boundary_tracks
    ):
        endpoint_graph, endpoint_summary = build_paired_endpoint_continuity_graph(
            bank,
            macro_eligible,
            sampling_stride_voxels=sampling_stride_voxels,
            link_radius_sampling_steps=resolved.endpoint_link_radius_sampling_steps,
            maximum_normal_degrees=resolved.maximum_endpoint_normal_degrees,
            maximum_endpoint_distance_sampling_steps=(
                resolved.maximum_endpoint_distance_sampling_steps
            ),
            maximum_endpoint_height_sampling_steps=(
                resolved.maximum_endpoint_height_sampling_steps
            ),
            normal_scale_degrees=resolved.endpoint_normal_scale_degrees,
            endpoint_height_scale_sampling_steps=(
                resolved.endpoint_height_scale_sampling_steps
            ),
            endpoint_distance_scale_sampling_steps=(
                resolved.endpoint_distance_scale_sampling_steps
            ),
        )
    if resolved.candidate_selection_mode == "endpoint-coordinate-ascent":
        lower_endpoint_affinity = np.where(
            np.asarray(endpoint_graph["lowerAffinity"], dtype=np.float64)
            >= resolved.minimum_endpoint_affinity,
            np.asarray(endpoint_graph["lowerAffinity"], dtype=np.float64),
            0.0,
        )
        upper_endpoint_affinity = np.where(
            np.asarray(endpoint_graph["upperAffinity"], dtype=np.float64)
            >= resolved.minimum_endpoint_affinity,
            np.asarray(endpoint_graph["upperAffinity"], dtype=np.float64),
            0.0,
        )
        strongest_endpoint = np.maximum(
            lower_endpoint_affinity, upper_endpoint_affinity
        )
        weakest_endpoint = np.minimum(
            lower_endpoint_affinity, upper_endpoint_affinity
        )
        endpoint_pair_weight = (
            resolved.endpoint_strong_face_weight * strongest_endpoint
            + resolved.endpoint_weak_face_weight * weakest_endpoint
            + resolved.endpoint_paired_face_weight
            * np.sqrt(lower_endpoint_affinity * upper_endpoint_affinity)
        )
        useful_endpoint_edge = endpoint_pair_weight > 0.0
        selected, selection_summary = _optimize_candidate_choices(
            bank,
            macro_eligible,
            np.asarray(endpoint_graph["firstCandidate"], dtype=np.int32)[
                useful_endpoint_edge
            ],
            np.asarray(endpoint_graph["secondCandidate"], dtype=np.int32)[
                useful_endpoint_edge
            ],
            endpoint_pair_weight[useful_endpoint_edge].astype(np.float32),
            settings=resolved,
        )
        selection_summary["mode"] = "endpoint-coordinate-ascent"
        selection_summary["endpointContinuityEdgeCount"] = int(
            np.count_nonzero(useful_endpoint_edge)
        )
        selection_summary["twoEndpointContinuityEdgeCount"] = int(
            np.count_nonzero(
                useful_endpoint_edge
                & (lower_endpoint_affinity > 0.0)
                & (upper_endpoint_affinity > 0.0)
            )
        )
        selection_summary["oneEndpointOnlyContinuityEdgeCount"] = int(
            np.count_nonzero(
                useful_endpoint_edge
                & ((lower_endpoint_affinity > 0.0) ^ (upper_endpoint_affinity > 0.0))
            )
        )
    if resolved.candidate_selection_mode == "growth":
        selected = growth_selected & macro_eligible
        selection_summary: dict[str, Any] = {
            "mode": "inherited-paired-growth",
            "selectedCandidateCount": int(np.count_nonzero(selected)),
        }
    elif resolved.candidate_selection_mode == "coordinate-ascent":
        selected, selection_summary = _optimize_candidate_choices(
            bank,
            macro_eligible,
            source_first[eligible_edge],
            source_second[eligible_edge],
            affinity[eligible_edge],
            settings=resolved,
        )
    retained_candidate = np.flatnonzero(selected).astype(np.int32)
    candidate_to_node = np.full(candidate_count, -1, dtype=np.int32)
    candidate_to_node[retained_candidate] = np.arange(
        len(retained_candidate), dtype=np.int32
    )
    endpoint_selected = (
        selected[np.asarray(endpoint_graph["firstCandidate"], dtype=np.int32)]
        & selected[np.asarray(endpoint_graph["secondCandidate"], dtype=np.int32)]
    )
    endpoint_first = candidate_to_node[
        np.asarray(endpoint_graph["firstCandidate"], dtype=np.int32)[
            endpoint_selected
        ]
    ]
    endpoint_second = candidate_to_node[
        np.asarray(endpoint_graph["secondCandidate"], dtype=np.int32)[
            endpoint_selected
        ]
    ]
    endpoint_orientation = np.asarray(
        endpoint_graph["secondOrientation"], dtype=np.int8
    )[endpoint_selected]
    endpoint_lower_affinity = np.asarray(
        endpoint_graph["lowerAffinity"], dtype=np.float32
    )[endpoint_selected]
    endpoint_upper_affinity = np.asarray(
        endpoint_graph["upperAffinity"], dtype=np.float32
    )[endpoint_selected]
    retained_edge = (
        eligible_edge & selected[source_first] & selected[source_second]
    )
    first = candidate_to_node[source_first[retained_edge]]
    second = candidate_to_node[source_second[retained_edge]]
    score = affinity[retained_edge]
    if len(first):
        pair = np.column_stack((np.minimum(first, second), np.maximum(first, second)))
        order = np.lexsort((pair[:, 1], pair[:, 0], -score))
        pair = pair[order]
        score = score[order]
        unique = np.ones(len(pair), dtype=bool)
        unique[1:] = np.any(pair[1:] != pair[:-1], axis=1)
        first = pair[unique, 0].astype(np.int32)
        second = pair[unique, 1].astype(np.int32)
        score = score[unique].astype(np.float32)
    retained_midpoint = np.asarray(bank["midpointXYZ"])[retained_candidate]
    retained_lower = np.asarray(bank["boundaryLowerXYZ"])[retained_candidate]
    retained_upper = np.asarray(bank["boundaryUpperXYZ"])[retained_candidate]
    retained_normal = np.asarray(bank["normalXYZ"])[retained_candidate]
    retained_thickness = np.asarray(bank["thicknessVoxels"], dtype=np.float32)[
        retained_candidate
    ]
    retained_key = np.asarray(bank["spatialKeyXYZ"])[retained_candidate]
    retained_macro_index = np.asarray(
        macro_assignment["macroBinIndex"], dtype=np.int32
    )[retained_candidate]
    boundary_track_arrays: dict[str, np.ndarray] = {
        "endpointXYZ": np.empty((0, 3), dtype=np.float32),
        "endpointProfileNode": np.empty(0, dtype=np.int32),
        "endpointSide": np.empty(0, dtype=np.uint8),
        "endpointComponentId": np.empty(0, dtype=np.int32),
        "endpointLocalSupportDegree": np.empty(0, dtype=np.int32),
        "edgeFirstEndpoint": np.empty(0, dtype=np.int32),
        "edgeSecondEndpoint": np.empty(0, dtype=np.int32),
        "edgeAffinity": np.empty(0, dtype=np.float32),
        "edgeKind": np.empty(0, dtype=np.uint8),
        "pairEdgeLowerRetained": np.zeros(len(endpoint_first), dtype=bool),
        "pairEdgeUpperRetained": np.zeros(len(endpoint_first), dtype=bool),
    }
    boundary_track_summary: dict[str, Any] = {"state": "not-requested"}
    lower_face_component = np.full(len(retained_candidate), -1, dtype=np.int32)
    upper_face_component = np.full(len(retained_candidate), -1, dtype=np.int32)
    if resolved.enable_boundary_tracks:
        boundary_track_arrays, boundary_track_summary = build_paired_boundary_tracks(
            retained_lower,
            retained_upper,
            retained_key,
            retained_macro_index,
            np.asarray(macro["centerXYZ"])[retained_macro_index],
            np.asarray(macro["normalXYZ"])[retained_macro_index],
            endpoint_first,
            endpoint_second,
            endpoint_orientation,
            endpoint_lower_affinity,
            endpoint_upper_affinity,
            sampling_stride_voxels=sampling_stride_voxels,
            settings=PairedBoundaryTrackSettings(
                minimum_track_affinity=(
                    resolved.boundary_track_minimum_affinity
                ),
                local_support_radius_sampling_steps=(
                    resolved.boundary_track_local_support_radius_sampling_steps
                ),
                minimum_local_support_affinity=(
                    resolved.boundary_track_minimum_local_support_affinity
                ),
                minimum_local_support_degree=(
                    resolved.boundary_track_minimum_local_support_degree
                ),
                tangent_column_width_sampling_steps=(
                    resolved.boundary_track_tangent_column_width_sampling_steps
                ),
                maximum_column_depth_range_sampling_steps=(
                    resolved.boundary_track_maximum_column_depth_range_sampling_steps
                ),
            ),
        )
        endpoint_component = np.asarray(
            boundary_track_arrays["endpointComponentId"], dtype=np.int32
        )
        lower_face_component = endpoint_component[0::2]
        upper_face_component = endpoint_component[1::2]
    closure_first, closure_second, closure_score, closure_summary = (
        _direct_geometric_closures(
            retained_midpoint,
            retained_lower,
            retained_upper,
            retained_normal,
            retained_thickness,
            retained_key,
            first,
            second,
            sampling_stride_voxels=sampling_stride_voxels,
            settings=resolved,
        )
    )
    source_base_edge_count = len(first)
    all_first = np.concatenate((first, closure_first))
    all_second = np.concatenate((second, closure_second))
    all_score = np.concatenate((score, closure_score))
    all_edge_kind = np.concatenate(
        (
            np.full(source_base_edge_count, 6, dtype=np.uint8),
            np.full(len(closure_first), 7, dtype=np.uint8),
        )
    )
    if resolved.component_solver_mode == "frontier-bundles":
        connectivity_edge, component_solver_summary = _frontier_bundle_connectivity(
            retained_midpoint,
            all_first,
            all_second,
            all_score,
            all_edge_kind,
            sampling_stride_voxels=sampling_stride_voxels,
            settings=resolved,
        )
    else:
        connectivity_edge = np.ones(len(all_first), dtype=bool)
        component_solver_summary = {
            "mode": "connected-components",
            "connectivityEdgeCount": int(len(all_first)),
        }
    connectivity_first = all_first[connectivity_edge]
    connectivity_second = all_second[connectivity_edge]
    connectivity_score = all_score[connectivity_edge]
    pre_component, _pre_component_count = _components(
        len(retained_candidate), connectivity_first, connectivity_second
    )
    column_summary: dict[str, int] = {
        "preCollisionComponentCount": int(len(_pre_component_count)),
        "postCollisionComponentCount": int(len(_pre_component_count)),
        "columnConflictRejectedEdgeCount": 0,
    }
    if resolved.enable_tangent_column_guard and len(connectivity_first):
        tangent_column, normal_depth = _tangent_columns(
            retained_midpoint,
            retained_macro_index,
            np.asarray(macro["centerXYZ"])[retained_macro_index],
            np.asarray(macro["normalXYZ"])[retained_macro_index],
            stride=int(round(sampling_stride_voxels)),
            width_sampling_steps=resolved.tangent_column_width_sampling_steps,
        )
        component, component_count, column_retained, column_summary = (
            _collision_safe_components(
                pre_component,
                connectivity_first,
                connectivity_second,
                connectivity_score,
                tangent_column,
                normal_depth,
                maximum_depth_range=(
                    resolved.maximum_column_depth_range_sampling_steps
                ),
            )
        )
    else:
        component = pre_component
        component_count = _pre_component_count
    # Component construction is deliberately more conservative than topology
    # retention.  Once identities are fixed, preserve every plausible edge
    # whose endpoints lie inside the same accepted sheet; downstream meshing
    # benefits from those redundant cycles without gaining a new merge path.
    final_edge = component[all_first] == component[all_second]
    first = all_first[final_edge]
    second = all_second[final_edge]
    score = all_score[final_edge]
    edge_kind = all_edge_kind[final_edge]
    component_size = component_count[component]
    original_label = np.asarray(growth["selectedLabel"], dtype=np.int32)[
        retained_candidate
    ]
    cross_original_label = original_label[first] != original_label[second]
    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)[
        retained_candidate
    ]
    thickness = retained_thickness
    opposing_cosine = np.asarray(bank["opposingNormalCosine"], dtype=np.float64)[
        retained_candidate
    ]
    opposing_degrees = np.degrees(
        np.arccos(np.clip(opposing_cosine, -1.0, 1.0))
    ).astype(np.float32)
    node_count = len(retained_candidate)
    arrays = {
        "midpointXYZ": retained_midpoint,
        "normalXYZ": retained_normal,
        "boundaryLowerXYZ": retained_lower,
        "boundaryUpperXYZ": retained_upper,
        # Every retained node is itself an observed two-boundary profile.  Kind
        # 1 is reserved by the viewer contract for a derived dense confirmation.
        "nodeKind": np.zeros(node_count, dtype=np.uint8),
        "profileAdopted": np.zeros(node_count, dtype=bool),
        "profileEndpointSupportCount": np.full(node_count, 2, dtype=np.uint8),
        "profileCanonicalOrientation": np.ones(node_count, dtype=np.int8),
        "profileCandidateIndex": retained_candidate,
        "spatialKeyXYZ": retained_key.astype(np.int32),
        "lowerSurfaceNode": np.full(node_count, -1, dtype=np.int32),
        "upperSurfaceNode": np.full(node_count, -1, dtype=np.int32),
        "lowerFaceComponentId": lower_face_component,
        "upperFaceComponentId": upper_face_component,
        "physicalSheetLabel": component.astype(np.int32),
        "componentId": component.astype(np.int32),
        "thicknessVoxels": thickness,
        "pairCost": (1.0 - evidence).astype(np.float32),
        "thicknessResidualVoxels": np.zeros(node_count, dtype=np.float32),
        "tangentResidualSamplingSteps": np.zeros(node_count, dtype=np.float32),
        "lowerDirectionDegrees": np.zeros(node_count, dtype=np.float32),
        "upperDirectionDegrees": np.zeros(node_count, dtype=np.float32),
        "opposingNormalDegrees": opposing_degrees,
        "lowerLocalEvidenceScore": evidence,
        "upperLocalEvidenceScore": evidence,
        "lowerExpectedThicknessVoxels": thickness,
        "upperExpectedThicknessVoxels": thickness,
        "lowerProfileDistanceSamplingSteps": np.zeros(node_count, dtype=np.float32),
        "upperProfileDistanceSamplingSteps": np.zeros(node_count, dtype=np.float32),
        "lowerProfileCandidateIndex": retained_candidate,
        "upperProfileCandidateIndex": retained_candidate,
        "oneSidedSourceSurfaceNode": np.full(node_count, -1, dtype=np.int32),
        "oneSidedPhysicalBoundarySide": np.full(node_count, 255, dtype=np.uint8),
        "sourceGrowthLabel": original_label,
        "macroNormalXYZ": np.asarray(macro_assignment["macroNormalXYZ"])[
            retained_candidate
        ],
        "macroNormalResidualDegrees": np.asarray(
            macro_assignment["macroNormalResidualDegrees"]
        )[retained_candidate],
        "edgeFirstNode": first,
        "edgeSecondNode": second,
        "edgeBoundarySupportMask": np.full(len(first), 3, dtype=np.uint8),
        "edgeKind": edge_kind,
        "edgeScore": score,
        "endpointEdgeFirstNode": endpoint_first.astype(np.int32),
        "endpointEdgeSecondNode": endpoint_second.astype(np.int32),
        "endpointEdgeSecondOrientation": endpoint_orientation,
        "endpointEdgeLowerAffinity": endpoint_lower_affinity,
        "endpointEdgeUpperAffinity": endpoint_upper_affinity,
        "endpointEdgeLowerTrackRetained": boundary_track_arrays[
            "pairEdgeLowerRetained"
        ],
        "endpointEdgeUpperTrackRetained": boundary_track_arrays[
            "pairEdgeUpperRetained"
        ],
        "boundaryTrackEndpointXYZ": boundary_track_arrays["endpointXYZ"],
        "boundaryTrackEndpointProfileNode": boundary_track_arrays[
            "endpointProfileNode"
        ],
        "boundaryTrackEndpointSide": boundary_track_arrays["endpointSide"],
        "boundaryTrackComponentId": boundary_track_arrays[
            "endpointComponentId"
        ],
        "boundaryTrackLocalSupportDegree": boundary_track_arrays[
            "endpointLocalSupportDegree"
        ],
        "boundaryTrackEdgeFirstEndpoint": boundary_track_arrays[
            "edgeFirstEndpoint"
        ],
        "boundaryTrackEdgeSecondEndpoint": boundary_track_arrays[
            "edgeSecondEndpoint"
        ],
        "boundaryTrackEdgeAffinity": boundary_track_arrays["edgeAffinity"],
        "boundaryTrackEdgeKind": boundary_track_arrays["edgeKind"],
    }
    summary = {
        "counts": {
            "candidateCount": int(candidate_count),
            "growthSelectedCandidateCount": int(np.count_nonzero(growth_selected)),
            "macroEligibleGrowthSelectedCandidateCount": int(
                np.count_nonzero(growth_selected & macro_eligible)
            ),
            "selectedCandidateCount": int(len(retained_candidate)),
            "selectedEndpointContinuityEdgeCount": int(len(endpoint_first)),
            "selectedTwoEndpointContinuityEdgeCount": int(
                np.count_nonzero(
                    (endpoint_lower_affinity >= resolved.minimum_endpoint_affinity)
                    & (endpoint_upper_affinity >= resolved.minimum_endpoint_affinity)
                )
            ),
            "selectedOneEndpointOnlyContinuityEdgeCount": int(
                np.count_nonzero(
                    (endpoint_lower_affinity >= resolved.minimum_endpoint_affinity)
                    ^ (endpoint_upper_affinity >= resolved.minimum_endpoint_affinity)
                )
            ),
            "macroRejectedSelectedCandidateCount": int(
                np.count_nonzero(growth_selected & ~macro_eligible)
            ),
            "inputContinuityEdgeCount": int(len(source_first)),
            "retainedGrowthContinuityEdgeCount": int(
                np.count_nonzero(edge_kind == 6)
            ),
            "consideredGeometricClosureEdgeCount": closure_summary["considered"],
            "retainedGeometricClosureEdgeCount": int(
                np.count_nonzero(edge_kind == 7)
            ),
            "retainedContinuityEdgeCount": int(len(first)),
            "crossOriginalGrowthLabelEdgeCount": int(
                np.count_nonzero(cross_original_label)
            ),
            "componentCount": int(len(component_count)),
            "componentsAtLeast32Nodes": int(np.count_nonzero(component_count >= 32)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_count >= 128)),
            "largestComponentSizes": [int(value) for value in component_count[:32]],
        },
        "selection": selection_summary,
        "endpointGraph": endpoint_summary,
        "boundaryTracks": boundary_track_summary,
        "componentSolver": component_solver_summary,
        "tangentColumnGuard": column_summary,
        "distributions": {
            "retainedMacroNormalResidualDegrees": _percentile_record(
                arrays["macroNormalResidualDegrees"]
            ),
            "retainedEdgeAffinity": _percentile_record(score),
            "retainedEdgeNormalDegrees": _percentile_record(
                edge_normal[retained_edge]
            ),
            "retainedEdgeBoundaryHeightSamplingSteps": _percentile_record(
                boundary_height[retained_edge]
            ),
        },
    }
    return arrays, summary


def run_direct_paired_profile_surface(
    paired_bank_root: str | Path,
    paired_growth_root: str | Path,
    macro_orientation_root: str | Path,
    output_root: str | Path,
    *,
    settings: PairedProfileSurfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Persist a macro-gated graph directly over selected paired profiles."""

    started = time.monotonic()
    resolved = settings or PairedProfileSurfaceSettings()
    bank_path = _resolve(paired_bank_root, PAIRED_SURFACE_BANK_STEM)
    growth_path = _resolve(paired_growth_root, PAIRED_SURFACE_GROWTH_STEM)
    macro_path = _resolve(macro_orientation_root, MACRO_ORIENTATION_STEM)
    bank_manifest, bank = _load(bank_path)
    growth_manifest, growth = _load(growth_path)
    macro_manifest, macro = _load(macro_path)
    if bank_manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA:
        raise ValueError("direct profile surfaces require a paired candidate bank")
    if growth_manifest.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA:
        raise ValueError("direct profile surfaces require paired growth selection")
    if macro_manifest.get("schema") != MACRO_ORIENTATION_SCHEMA:
        raise ValueError("direct profile surfaces require macro orientation")
    identity: dict[str, Any] = {
        "schema": PAIRED_PROFILE_SURFACE_SCHEMA,
        "version": PAIRED_PROFILE_SURFACE_VERSION,
        "pairedBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "pairedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "macroOrientation": {
            "manifestPath": str(macro_path),
            "manifestSha256": sha256_file(macro_path),
            "dataSha256": macro_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "boundaryTrackImplementationSha256": sha256_file(
            Path(__file__).with_name("paired_boundary_tracks.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.json"
    data_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.npz"
    preview_path = output / "direct-paired-profile-cross-sections.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    slab_path = Path(bank_manifest["identity"]["isolatedSlabs"]["manifestPath"])
    slab_manifest = json.loads(slab_path.read_text())
    stride = int(slab_manifest["identity"]["settings"]["sampling_stride_voxels"])
    arrays, summary = build_direct_paired_profile_surface(
        bank,
        growth,
        macro_manifest,
        macro,
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    _write_npz(data_path, arrays)
    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"].get("metadataPath")
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]), tuple(owned_record["stopXYZExclusive"])
    )
    component_count = np.bincount(np.asarray(arrays["componentId"], dtype=np.int32))
    component_size = component_count[np.asarray(arrays["componentId"], dtype=np.int32)]
    write_material_surface_cross_sections(
        source,
        owned,
        arrays["midpointXYZ"],
        arrays["componentId"],
        component_size,
        preview_path,
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=MaterialSurfaceGraphSettings(
            minimum_component_samples_for_preview=(
                resolved.minimum_component_nodes_for_preview
            ),
            maximum_preview_components=resolved.maximum_preview_components,
        ),
    )
    geometry = dict(bank_manifest["geometry"])
    geometry["samplingStrideVoxels"] = stride
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_VERSION,
        "constructionSchema": PAIRED_PROFILE_SURFACE_SCHEMA,
        "constructionVersion": PAIRED_PROFILE_SURFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": geometry,
        "calibration": bank_manifest["calibration"],
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "node": "one selected immutable air-papyrus-air profile",
            "edge": "both observed profile boundaries continue geometrically",
            "identity": "connected components after discarding inherited seed labels",
            "boundaryTrack": (
                "independent physical faces joined by locally supported endpoint "
                "continuity with collision-safe macro-tangent ordering"
                if resolved.enable_boundary_tracks
                else "not requested"
            ),
            "macroGuard": "independent generic unsigned laminar tensor",
            "oneCandidatePerSpatialKey": True,
            "acusRole": "none",
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
