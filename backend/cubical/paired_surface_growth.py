from __future__ import annotations

import colorsys
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, VoxelBounds, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .isolated_slab import _percentile_record
from .paired_surface_bank import (
    PAIRED_SURFACE_BANK_SCHEMA,
    PAIRED_SURFACE_BANK_STEM,
)


PAIRED_SURFACE_GROWTH_SCHEMA = "pareidolia.paired-surface-contextual-growth"
PAIRED_SURFACE_GROWTH_VERSION = 1
PAIRED_SURFACE_GROWTH_STEM = "paired-surface-growth-v1"


@dataclass(frozen=True, slots=True)
class PairedSurfaceGrowthSettings:
    minimum_seed_component_size: int = 8
    minimum_candidate_evidence: float = 0.35
    minimum_growth_bottleneck: float = 0.35
    link_radius_sampling_steps: float = math.sqrt(5.0)
    maximum_midpoint_distance_sampling_steps: float = 3.0
    maximum_normal_degrees: float = 35.0
    maximum_midpoint_height_sampling_steps: float = 1.5
    maximum_boundary_height_sampling_steps: float = 2.0
    maximum_thickness_difference_sampling_steps: float = 3.0
    affinity_normal_sigma_degrees: float = 15.0
    affinity_midpoint_height_sigma_steps: float = 0.75
    affinity_boundary_height_sigma_steps: float = 1.0
    affinity_thickness_sigma_steps: float = 1.5
    minimum_seed_association_bottleneck: float = 0.5
    minimum_seed_association_shared_candidates: int = 24
    minimum_seed_association_median_bottleneck: float = 0.55
    minimum_seed_association_reciprocal_candidates: int = 5
    minimum_seed_association_reciprocal_fraction: float = 0.05
    minimum_seed_association_extent_sampling_steps: float = 10.0
    maximum_preview_labels: int = 128

    def __post_init__(self) -> None:
        if (
            self.minimum_seed_component_size < 1
            or self.minimum_seed_association_shared_candidates < 1
            or self.minimum_seed_association_reciprocal_candidates < 1
            or self.maximum_preview_labels < 1
        ):
            raise ValueError("growth component and preview counts must be positive")
        for value, name in (
            (self.minimum_candidate_evidence, "minimum candidate evidence"),
            (self.minimum_growth_bottleneck, "minimum growth bottleneck"),
            (
                self.minimum_seed_association_bottleneck,
                "minimum seed-association bottleneck",
            ),
            (
                self.minimum_seed_association_median_bottleneck,
                "minimum median seed-association bottleneck",
            ),
            (
                self.minimum_seed_association_reciprocal_fraction,
                "minimum reciprocal seed-association fraction",
            ),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        positive = (
            self.link_radius_sampling_steps,
            self.maximum_midpoint_distance_sampling_steps,
            self.maximum_normal_degrees,
            self.maximum_midpoint_height_sampling_steps,
            self.maximum_boundary_height_sampling_steps,
            self.maximum_thickness_difference_sampling_steps,
            self.affinity_normal_sigma_degrees,
            self.affinity_midpoint_height_sigma_steps,
            self.affinity_boundary_height_sigma_steps,
            self.affinity_thickness_sigma_steps,
            self.minimum_seed_association_extent_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("growth geometry scales must be finite and positive")
        if not 0.0 < self.maximum_normal_degrees < 90.0:
            raise ValueError("maximum growth normal angle must lie in (0, 90)")
        if self.minimum_candidate_evidence > self.minimum_growth_bottleneck:
            raise ValueError(
                "candidate evidence gate cannot exceed the growth bottleneck"
            )
        if (
            self.minimum_seed_association_bottleneck
            < self.minimum_growth_bottleneck
        ):
            raise ValueError(
                "seed-association bottleneck cannot be below the growth bottleneck"
            )
        if (
            self.minimum_seed_association_median_bottleneck
            < self.minimum_seed_association_bottleneck
        ):
            raise ValueError(
                "median seed-association bottleneck cannot be below its sample gate"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_bank(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = value if value.is_file() else value / f"{PAIRED_SURFACE_BANK_STEM}.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("paired-surface growth requires a complete candidate bank")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("paired-surface bank data hash differs from its manifest")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return manifest_path, manifest, arrays


def _sampling_stride(bank_manifest: Mapping[str, Any]) -> int:
    slab_manifest_path = Path(
        bank_manifest["identity"]["isolatedSlabs"]["manifestPath"]
    )
    slab_manifest = json.loads(slab_manifest_path.read_text())
    return int(slab_manifest["identity"]["settings"]["sampling_stride_voxels"])


def _continuity_metrics(
    first: np.ndarray,
    second: np.ndarray,
    bank: Mapping[str, np.ndarray],
    *,
    stride: int,
) -> dict[str, np.ndarray]:
    first_normal = np.asarray(bank["normalXYZ"])[first].astype(np.float64)
    second_normal = np.asarray(bank["normalXYZ"])[second].astype(np.float64)
    dot = np.einsum("ij,ij->i", first_normal, second_normal)
    sign = np.where(dot >= 0.0, 1.0, -1.0)
    aligned_second_normal = second_normal * sign[:, None]
    average_normal = first_normal + aligned_second_normal
    average_normal /= np.maximum(
        np.linalg.norm(average_normal, axis=1, keepdims=True), 1.0e-9
    )
    midpoint_delta = (
        np.asarray(bank["midpointXYZ"])[second]
        - np.asarray(bank["midpointXYZ"])[first]
    ).astype(np.float64)
    midpoint_distance = np.linalg.norm(midpoint_delta, axis=1) / stride
    midpoint_height = np.abs(
        np.einsum("ij,ij->i", midpoint_delta, average_normal)
    ) / stride
    first_lower = np.asarray(bank["boundaryLowerXYZ"])[first].astype(np.float64)
    first_upper = np.asarray(bank["boundaryUpperXYZ"])[first].astype(np.float64)
    raw_second_lower = np.asarray(bank["boundaryLowerXYZ"])[second].astype(np.float64)
    raw_second_upper = np.asarray(bank["boundaryUpperXYZ"])[second].astype(np.float64)
    second_lower = np.where(sign[:, None] > 0.0, raw_second_lower, raw_second_upper)
    second_upper = np.where(sign[:, None] > 0.0, raw_second_upper, raw_second_lower)
    lower_height = np.abs(
        np.einsum("ij,ij->i", second_lower - first_lower, average_normal)
    ) / stride
    upper_height = np.abs(
        np.einsum("ij,ij->i", second_upper - first_upper, average_normal)
    ) / stride
    boundary_height = np.maximum(lower_height, upper_height)
    thickness_difference = np.abs(
        np.asarray(bank["thicknessVoxels"])[second]
        - np.asarray(bank["thicknessVoxels"])[first]
    ) / stride
    normal_degrees = np.degrees(np.arccos(np.clip(np.abs(dot), 0.0, 1.0)))
    return {
        "midpointDistanceSamplingSteps": midpoint_distance.astype(np.float32),
        "normalDegrees": normal_degrees.astype(np.float32),
        "midpointHeightSamplingSteps": midpoint_height.astype(np.float32),
        "lowerBoundaryHeightSamplingSteps": lower_height.astype(np.float32),
        "upperBoundaryHeightSamplingSteps": upper_height.astype(np.float32),
        "boundaryHeightSamplingSteps": boundary_height.astype(np.float32),
        "thicknessDifferenceSamplingSteps": thickness_difference.astype(np.float32),
    }


def build_paired_surface_continuity_graph(
    bank: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build strict two-boundary continuation edges over eligible candidates."""

    candidate_count = len(bank["localEvidenceScore"])
    seed = np.asarray(bank["seedComponentId"]) >= 0
    eligible = (
        np.asarray(bank["localEvidenceScore"])
        >= settings.minimum_candidate_evidence
    ) | seed
    eligible_index = np.flatnonzero(eligible).astype(np.int32)
    key = np.asarray(bank["spatialKeyXYZ"])[eligible_index].astype(np.int32)
    flat = np.ravel_multi_index(
        (key[:, 2], key[:, 1], key[:, 0]),
        processing_shape_sampling_xyz[::-1],
    )
    order = np.argsort(flat, kind="stable")
    sorted_flat = flat[order]
    unique_flat, first, count = np.unique(
        sorted_flat, return_index=True, return_counts=True
    )
    group_key = key[order[first]]
    group_grid = np.full(
        processing_shape_sampling_xyz[::-1], -1, dtype=np.int32
    )
    group_grid[group_key[:, 2], group_key[:, 1], group_key[:, 0]] = np.arange(
        len(unique_flat), dtype=np.int32
    )
    maximum_alternatives = int(np.max(count, initial=0))
    reach = int(math.ceil(settings.link_radius_sampling_steps))
    edge_first_parts: list[np.ndarray] = []
    edge_second_parts: list[np.ndarray] = []
    metric_parts: dict[str, list[np.ndarray]] = {}
    affinity_parts: list[np.ndarray] = []
    considered = 0
    gated_counts = {
        "midpointDistance": 0,
        "normal": 0,
        "midpointHeight": 0,
        "boundaryHeight": 0,
        "thickness": 0,
    }
    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                if (
                    dx * dx + dy * dy + dz * dz
                    > settings.link_radius_sampling_steps**2 + 1.0e-9
                ):
                    continue
                valid_group = (
                    (group_key[:, 0] + dx >= 0)
                    & (group_key[:, 0] + dx < processing_shape_sampling_xyz[0])
                    & (group_key[:, 1] + dy >= 0)
                    & (group_key[:, 1] + dy < processing_shape_sampling_xyz[1])
                    & (group_key[:, 2] + dz >= 0)
                    & (group_key[:, 2] + dz < processing_shape_sampling_xyz[2])
                )
                source_group = np.flatnonzero(valid_group)
                source_key = group_key[source_group]
                target_group = group_grid[
                    source_key[:, 2] + dz,
                    source_key[:, 1] + dy,
                    source_key[:, 0] + dx,
                ]
                exists = target_group >= 0
                source_group = source_group[exists]
                target_group = target_group[exists]
                if not len(source_group):
                    continue
                for source_alternative in range(maximum_alternatives):
                    source_valid = count[source_group] > source_alternative
                    if not np.any(source_valid):
                        continue
                    source_selected_group = source_group[source_valid]
                    target_selected_group = target_group[source_valid]
                    source_sorted_index = (
                        first[source_selected_group] + source_alternative
                    )
                    source_candidate = eligible_index[order[source_sorted_index]]
                    for target_alternative in range(maximum_alternatives):
                        pair_valid = count[target_selected_group] > target_alternative
                        if not np.any(pair_valid):
                            continue
                        left = source_candidate[pair_valid]
                        right_group = target_selected_group[pair_valid]
                        right_sorted_index = first[right_group] + target_alternative
                        right = eligible_index[order[right_sorted_index]]
                        considered += len(left)
                        metrics = _continuity_metrics(
                            left, right, bank, stride=stride
                        )
                        accepted = (
                            metrics["midpointDistanceSamplingSteps"]
                            <= settings.maximum_midpoint_distance_sampling_steps
                        )
                        gated_counts["midpointDistance"] += int(
                            np.count_nonzero(~accepted)
                        )
                        prior = accepted.copy()
                        accepted &= (
                            metrics["normalDegrees"]
                            <= settings.maximum_normal_degrees
                        )
                        gated_counts["normal"] += int(
                            np.count_nonzero(prior & ~accepted)
                        )
                        prior = accepted.copy()
                        accepted &= (
                            metrics["midpointHeightSamplingSteps"]
                            <= settings.maximum_midpoint_height_sampling_steps
                        )
                        gated_counts["midpointHeight"] += int(
                            np.count_nonzero(prior & ~accepted)
                        )
                        prior = accepted.copy()
                        accepted &= (
                            metrics["boundaryHeightSamplingSteps"]
                            <= settings.maximum_boundary_height_sampling_steps
                        )
                        gated_counts["boundaryHeight"] += int(
                            np.count_nonzero(prior & ~accepted)
                        )
                        prior = accepted.copy()
                        accepted &= (
                            metrics["thicknessDifferenceSamplingSteps"]
                            <= settings.maximum_thickness_difference_sampling_steps
                        )
                        gated_counts["thickness"] += int(
                            np.count_nonzero(prior & ~accepted)
                        )
                        if not np.any(accepted):
                            continue
                        selected_metrics = {
                            name: values[accepted]
                            for name, values in metrics.items()
                        }
                        exponent = (
                            (
                                selected_metrics["normalDegrees"]
                                / settings.affinity_normal_sigma_degrees
                            )
                            ** 2
                            + (
                                selected_metrics["midpointHeightSamplingSteps"]
                                / settings.affinity_midpoint_height_sigma_steps
                            )
                            ** 2
                            + (
                                selected_metrics["boundaryHeightSamplingSteps"]
                                / settings.affinity_boundary_height_sigma_steps
                            )
                            ** 2
                            + (
                                selected_metrics[
                                    "thicknessDifferenceSamplingSteps"
                                ]
                                / settings.affinity_thickness_sigma_steps
                            )
                            ** 2
                        )
                        affinity = np.exp(-0.5 * exponent).astype(np.float32)
                        edge_first_parts.append(left[accepted].astype(np.int32))
                        edge_second_parts.append(right[accepted].astype(np.int32))
                        affinity_parts.append(affinity)
                        for name, values in selected_metrics.items():
                            metric_parts.setdefault(name, []).append(
                                values.astype(np.float32)
                            )
    edge_first = (
        np.concatenate(edge_first_parts)
        if edge_first_parts
        else np.empty(0, dtype=np.int32)
    )
    edge_second = (
        np.concatenate(edge_second_parts)
        if edge_second_parts
        else np.empty(0, dtype=np.int32)
    )
    affinity = (
        np.concatenate(affinity_parts)
        if affinity_parts
        else np.empty(0, dtype=np.float32)
    )
    graph = {
        "edgeFirstCandidate": edge_first,
        "edgeSecondCandidate": edge_second,
        "edgeAffinity": affinity,
        **{
            f"edge{name[0].upper()}{name[1:]}": np.concatenate(parts)
            for name, parts in metric_parts.items()
        },
    }
    return graph, {
        "candidateCount": candidate_count,
        "eligibleCandidateCount": int(len(eligible_index)),
        "eligibleSpatialKeyCount": int(len(unique_flat)),
        "consideredCandidatePairCount": considered,
        "continuityEdgeCount": int(len(edge_first)),
        "gateRejectionCount": gated_counts,
        "edgeAffinity": _percentile_record(affinity),
        "edgeNormalDegrees": _percentile_record(
            graph.get("edgeNormalDegrees", np.empty(0))
        ),
        "edgeMidpointHeightSamplingSteps": _percentile_record(
            graph.get("edgeMidpointHeightSamplingSteps", np.empty(0))
        ),
        "edgeBoundaryHeightSamplingSteps": _percentile_record(
            graph.get("edgeBoundaryHeightSamplingSteps", np.empty(0))
        ),
    }


def _adjacency(
    node_count: int,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_affinity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.concatenate((edge_first, edge_second)).astype(np.int32)
    target = np.concatenate((edge_second, edge_first)).astype(np.int32)
    edge = np.concatenate(
        (
            np.arange(len(edge_first), dtype=np.int32),
            np.arange(len(edge_first), dtype=np.int32),
        )
    )
    affinity = np.concatenate((edge_affinity, edge_affinity)).astype(np.float32)
    order = np.argsort(source, kind="stable")
    count = np.bincount(source, minlength=node_count)
    offset = np.concatenate(((0,), np.cumsum(count, dtype=np.int64)))
    return offset, target[order], edge[order], affinity[order]


def _locked_seed_components(
    bank: Mapping[str, np.ndarray],
    settings: PairedSurfaceGrowthSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    seed_component = np.asarray(bank["seedComponentId"], dtype=np.int32)
    valid_seed = seed_component >= 0
    component_value, component_size = np.unique(
        seed_component[valid_seed], return_counts=True
    )
    eligible_component = component_value[
        component_size >= settings.minimum_seed_component_size
    ]
    locked_seed = valid_seed & np.isin(seed_component, eligible_component)
    return seed_component, component_value, component_size, locked_seed


def discover_seed_component_support(
    bank: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    locked_seed: np.ndarray,
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Find the two strongest immutable seed explanations for each profile.

    Foreign support may reach a clear seed profile so reciprocal evidence can
    be measured, but it may not propagate through that profile.  This makes a
    seed patch a frontier rather than a one-edge bridge into another sheet.
    """

    node_count = len(bank["localEvidenceScore"])
    seed_component = np.asarray(bank["seedComponentId"], dtype=np.int32)
    local_evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    edge_first = np.asarray(graph["edgeFirstCandidate"], dtype=np.int32)
    edge_second = np.asarray(graph["edgeSecondCandidate"], dtype=np.int32)
    edge_affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    offset, neighbor, _adjacency_edge, adjacency_affinity = _adjacency(
        node_count, edge_first, edge_second, edge_affinity
    )
    support_label = np.full((node_count, 2), -1, dtype=np.int32)
    support_score = np.zeros((node_count, 2), dtype=np.float32)
    queue: list[tuple[float, int, int]] = [
        (-1.0, int(node), int(seed_component[node]))
        for node in np.flatnonzero(locked_seed)
    ]
    heapq.heapify(queue)
    proposal_count = len(queue)
    finalized_state_count = 0
    foreign_seed_frontier_count = 0
    while queue:
        negative_score, node, proposed_label = heapq.heappop(queue)
        score = -negative_score
        if (
            proposed_label == int(support_label[node, 0])
            or proposed_label == int(support_label[node, 1])
            or support_label[node, 1] >= 0
        ):
            continue
        slot = 0 if support_label[node, 0] < 0 else 1
        support_label[node, slot] = proposed_label
        support_score[node, slot] = score
        finalized_state_count += 1
        if locked_seed[node] and int(seed_component[node]) != proposed_label:
            foreign_seed_frontier_count += 1
            continue
        for adjacency_index in range(int(offset[node]), int(offset[node + 1])):
            target = int(neighbor[adjacency_index])
            if (
                proposed_label == int(support_label[target, 0])
                or proposed_label == int(support_label[target, 1])
                or support_label[target, 1] >= 0
            ):
                continue
            candidate_score = min(
                score,
                float(adjacency_affinity[adjacency_index]),
                float(local_evidence[target]),
            )
            if candidate_score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue, (-candidate_score, target, proposed_label)
            )
            proposal_count += 1
    second = support_label[:, 1] >= 0
    return {
        "supportSeedComponent": support_label,
        "supportPathBottleneck": support_score,
    }, {
        "proposalCount": proposal_count,
        "finalizedStateCount": finalized_state_count,
        "candidateWithOneSeedExplanationCount": int(
            np.count_nonzero(support_label[:, 0] >= 0)
        ),
        "candidateWithTwoSeedExplanationsCount": int(np.count_nonzero(second)),
        "foreignSeedFrontierStateCount": foreign_seed_frontier_count,
        "secondPathBottleneck": _percentile_record(support_score[second, 1]),
    }


def associate_seed_components(
    bank: Mapping[str, np.ndarray],
    support: Mapping[str, np.ndarray],
    *,
    locked_seed: np.ndarray,
    component_value: np.ndarray,
    component_size: np.ndarray,
    stride: int,
    settings: PairedSurfaceGrowthSettings,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, Any]]:
    """Collapse seed identities only under broad reciprocal path support."""

    seed_component = np.asarray(bank["seedComponentId"], dtype=np.int32)
    support_label = np.asarray(support["supportSeedComponent"], dtype=np.int32)
    support_score = np.asarray(support["supportPathBottleneck"], dtype=np.float32)
    eligible_component = component_value[
        component_size >= settings.minimum_seed_component_size
    ].astype(np.int32)
    size_by_component = {
        int(value): int(size)
        for value, size in zip(component_value, component_size)
        if size >= settings.minimum_seed_component_size
    }
    shared = (
        (support_label[:, 1] >= 0)
        & (support_score[:, 1] >= settings.minimum_seed_association_bottleneck)
    )
    shared_index = np.flatnonzero(shared)
    shared_first = np.minimum(
        support_label[shared_index, 0], support_label[shared_index, 1]
    ).astype(np.int64)
    shared_second = np.maximum(
        support_label[shared_index, 0], support_label[shared_index, 1]
    ).astype(np.int64)
    shared_pair = (shared_first << 32) | shared_second
    shared_order = np.argsort(shared_pair, kind="stable")
    shared_unique, shared_start, shared_count = np.unique(
        shared_pair[shared_order], return_index=True, return_counts=True
    )

    # Count directional support that actually reaches the other immutable seed.
    frontier = locked_seed & (support_label[:, 1] >= 0)
    frontier_index = np.flatnonzero(frontier)
    target_component = seed_component[frontier_index].astype(np.int64)
    foreign_component = np.where(
        support_label[frontier_index, 0] == target_component,
        support_label[frontier_index, 1],
        support_label[frontier_index, 0],
    ).astype(np.int64)
    foreign_score = np.where(
        support_label[frontier_index, 0] == target_component,
        support_score[frontier_index, 1],
        support_score[frontier_index, 0],
    )
    frontier_valid = (
        (foreign_component != target_component)
        & (
            foreign_score
            >= settings.minimum_seed_association_bottleneck
        )
    )
    directional_pair = (
        (foreign_component[frontier_valid] << 32)
        | target_component[frontier_valid]
    )
    directional_unique, directional_count = np.unique(
        directional_pair, return_counts=True
    )
    directional = {
        (int(value >> 32), int(value & 0xFFFFFFFF)): int(count)
        for value, count in zip(directional_unique, directional_count)
    }

    midpoint = np.asarray(bank["midpointXYZ"], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for encoded, start, count in zip(
        shared_unique, shared_start, shared_count
    ):
        first = int(encoded >> 32)
        second = int(encoded & 0xFFFFFFFF)
        indices = shared_index[shared_order[start : start + count]]
        scores = support_score[indices, 1]
        extent = float(
            np.linalg.norm(np.ptp(midpoint[indices], axis=0)) / stride
        )
        first_to_second = directional.get((first, second), 0)
        second_to_first = directional.get((second, first), 0)
        first_to_second_fraction = (
            first_to_second / max(size_by_component.get(second, 0), 1)
        )
        second_to_first_fraction = (
            second_to_first / max(size_by_component.get(first, 0), 1)
        )
        median_score = float(np.median(scores))
        accepted = bool(
            count >= settings.minimum_seed_association_shared_candidates
            and median_score
            >= settings.minimum_seed_association_median_bottleneck
            and extent
            >= settings.minimum_seed_association_extent_sampling_steps
            and first_to_second
            >= settings.minimum_seed_association_reciprocal_candidates
            and second_to_first
            >= settings.minimum_seed_association_reciprocal_candidates
            and first_to_second_fraction
            >= settings.minimum_seed_association_reciprocal_fraction
            and second_to_first_fraction
            >= settings.minimum_seed_association_reciprocal_fraction
        )
        records.append(
            {
                "firstSeedComponent": first,
                "secondSeedComponent": second,
                "sharedCandidateCount": int(count),
                "medianPathBottleneck": median_score,
                "supportExtentSamplingSteps": extent,
                "firstToSecondSeedCount": first_to_second,
                "secondToFirstSeedCount": second_to_first,
                "firstToSecondSeedFraction": first_to_second_fraction,
                "secondToFirstSeedFraction": second_to_first_fraction,
                "accepted": accepted,
            }
        )

    parent = {int(value): int(value) for value in eligible_component}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            following = parent[value]
            parent[value] = root
            value = following
        return root

    accepted_records = [value for value in records if value["accepted"]]
    accepted_records.sort(
        key=lambda value: (
            -min(
                value["firstToSecondSeedFraction"],
                value["secondToFirstSeedFraction"],
            ),
            -value["medianPathBottleneck"],
            -value["sharedCandidateCount"],
            value["firstSeedComponent"],
            value["secondSeedComponent"],
        )
    )
    for value in accepted_records:
        first = find(int(value["firstSeedComponent"]))
        second = find(int(value["secondSeedComponent"]))
        if first == second:
            continue
        # The minimum original component ID is the stable assembly identity.
        low, high = sorted((first, second))
        parent[high] = low
    assembly_by_component = {
        int(value): find(int(value)) for value in eligible_component
    }
    seed_assembly_label = np.full(len(seed_component), -1, dtype=np.int32)
    for node in np.flatnonzero(locked_seed):
        seed_assembly_label[node] = assembly_by_component[int(seed_component[node])]
    members: dict[int, list[int]] = {}
    for component, assembly in assembly_by_component.items():
        members.setdefault(assembly, []).append(component)
    member_sizes = sorted((len(value) for value in members.values()), reverse=True)
    records.sort(
        key=lambda value: (
            not value["accepted"],
            -value["sharedCandidateCount"],
            -value["medianPathBottleneck"],
            value["firstSeedComponent"],
            value["secondSeedComponent"],
        )
    )
    association_arrays = {
        "associationFirstSeedComponent": np.asarray(
            [value["firstSeedComponent"] for value in records], dtype=np.int32
        ),
        "associationSecondSeedComponent": np.asarray(
            [value["secondSeedComponent"] for value in records], dtype=np.int32
        ),
        "associationSharedCandidateCount": np.asarray(
            [value["sharedCandidateCount"] for value in records], dtype=np.int32
        ),
        "associationMedianPathBottleneck": np.asarray(
            [value["medianPathBottleneck"] for value in records], dtype=np.float32
        ),
        "associationSupportExtentSamplingSteps": np.asarray(
            [value["supportExtentSamplingSteps"] for value in records],
            dtype=np.float32,
        ),
        "associationFirstToSecondSeedCount": np.asarray(
            [value["firstToSecondSeedCount"] for value in records], dtype=np.int32
        ),
        "associationSecondToFirstSeedCount": np.asarray(
            [value["secondToFirstSeedCount"] for value in records], dtype=np.int32
        ),
        "associationAccepted": np.asarray(
            [value["accepted"] for value in records], dtype=np.uint8
        ),
    }
    return seed_assembly_label, association_arrays, {
        "candidatePairCount": len(records),
        "acceptedPairCount": len(accepted_records),
        "inputSeedComponentCount": int(len(eligible_component)),
        "seedAssemblyCount": len(members),
        "multiComponentAssemblyCount": int(
            np.count_nonzero(np.asarray(member_sizes) > 1)
        ),
        "largestAssemblySeedComponentCount": member_sizes[0] if member_sizes else 0,
        "largestAssemblies": [
            {
                "assemblyLabel": int(assembly),
                "seedComponentCount": len(components),
                "seedComponents": sorted(components),
                "seedCandidateCount": int(
                    sum(size_by_component[value] for value in components)
                ),
            }
            for assembly, components in sorted(
                (
                    item
                    for item in members.items()
                    if len(item[1]) > 1
                ),
                key=lambda item: (-len(item[1]), item[0]),
            )[:128]
        ],
        "strongestAssociations": records[:128],
    }


def grow_paired_surfaces(
    bank: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Associate clear patches, then grow a key-exclusive bottleneck forest."""

    node_count = len(bank["localEvidenceScore"])
    (
        seed_component,
        component_value,
        component_size,
        locked_seed,
    ) = _locked_seed_components(bank, settings)
    eligible_component = component_value[
        component_size >= settings.minimum_seed_component_size
    ]
    support, support_stats = discover_seed_component_support(
        bank,
        graph,
        locked_seed=locked_seed,
        settings=settings,
    )
    (
        seed_assembly_label,
        association_arrays,
        association_stats,
    ) = associate_seed_components(
        bank,
        support,
        locked_seed=locked_seed,
        component_value=component_value,
        component_size=component_size,
        stride=stride,
        settings=settings,
    )
    edge_first = np.asarray(graph["edgeFirstCandidate"], dtype=np.int32)
    edge_second = np.asarray(graph["edgeSecondCandidate"], dtype=np.int32)
    edge_affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    offset, neighbor, adjacency_edge, adjacency_affinity = _adjacency(
        node_count, edge_first, edge_second, edge_affinity
    )
    label = np.full(node_count, -1, dtype=np.int32)
    path_score = np.zeros(node_count, dtype=np.float32)
    parent = np.full(node_count, -1, dtype=np.int32)
    parent_edge = np.full(node_count, -1, dtype=np.int32)
    selected_key_node = np.full(
        processing_shape_sampling_xyz[::-1], -1, dtype=np.int32
    )
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    local_evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    queue: list[tuple[float, int, int, int, int]] = []
    for node in np.flatnonzero(locked_seed):
        heapq.heappush(
            queue,
            (-1.0, int(node), int(seed_assembly_label[node]), -1, -1),
        )
    key_conflict_rejections = 0
    protected_seed_rejections = 0
    proposal_count = len(queue)
    while queue:
        negative_score, node, proposed_label, proposed_parent, proposed_edge = heapq.heappop(
            queue
        )
        score = -negative_score
        if label[node] >= 0:
            continue
        if score < settings.minimum_growth_bottleneck:
            break
        existing_key_node = int(
            selected_key_node[key[node, 2], key[node, 1], key[node, 0]]
        )
        if existing_key_node >= 0:
            key_conflict_rejections += 1
            continue
        own_seed_label = int(seed_assembly_label[node])
        if locked_seed[node] and own_seed_label != proposed_label:
            protected_seed_rejections += 1
            continue
        label[node] = proposed_label
        path_score[node] = score
        parent[node] = proposed_parent
        parent_edge[node] = proposed_edge
        selected_key_node[key[node, 2], key[node, 1], key[node, 0]] = node
        for adjacency_index in range(int(offset[node]), int(offset[node + 1])):
            target = int(neighbor[adjacency_index])
            if label[target] >= 0:
                continue
            if (
                locked_seed[target]
                and int(seed_assembly_label[target]) != proposed_label
            ):
                continue
            candidate_score = min(
                score,
                float(adjacency_affinity[adjacency_index]),
                float(local_evidence[target]),
            )
            if candidate_score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue,
                (
                    -candidate_score,
                    target,
                    proposed_label,
                    node,
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1
    selected = label >= 0
    grown = selected & ~locked_seed
    same_label_edge = (
        selected[edge_first]
        & selected[edge_second]
        & (label[edge_first] == label[edge_second])
        & (edge_affinity >= settings.minimum_growth_bottleneck)
    )
    same_label_neighbor_count = np.bincount(
        np.concatenate(
            (
                edge_first[same_label_edge],
                edge_second[same_label_edge],
            )
        ),
        minlength=node_count,
    ).astype(np.uint16)
    parent_hop_depth = np.full(node_count, -1, dtype=np.int16)
    parent_hop_depth[selected & locked_seed] = 0
    for node in np.flatnonzero(grown):
        if parent_hop_depth[node] >= 0:
            continue
        trail: list[int] = []
        current = int(node)
        visited: set[int] = set()
        while (
            current >= 0
            and parent_hop_depth[current] < 0
            and current not in visited
        ):
            visited.add(current)
            trail.append(current)
            current = int(parent[current])
        depth = int(parent_hop_depth[current]) if current >= 0 else -1
        for value in reversed(trail):
            depth += 1
            parent_hop_depth[value] = depth
    selected_labels, selected_size = np.unique(label[selected], return_counts=True)
    selected_seed_labels, selected_seed_size = np.unique(
        label[selected & locked_seed], return_counts=True
    )
    seed_size_by_label = {
        int(value): int(size)
        for value, size in zip(selected_seed_labels, selected_seed_size)
    }
    seed_component_count_by_label: dict[int, int] = {}
    for component in eligible_component:
        member = np.flatnonzero(locked_seed & (seed_component == component))
        if not len(member):
            continue
        assembly = int(seed_assembly_label[member[0]])
        seed_component_count_by_label[assembly] = (
            seed_component_count_by_label.get(assembly, 0) + 1
        )
    records = []
    for value, size in zip(selected_labels, selected_size):
        seed_size = seed_size_by_label.get(int(value), 0)
        records.append(
            {
                "label": int(value),
                "seedCount": seed_size,
                "seedComponentCount": seed_component_count_by_label.get(
                    int(value), 0
                ),
                "selectedCount": int(size),
                "grownCount": int(size) - seed_size,
                "growthFactor": round(float(size) / max(seed_size, 1), 6),
            }
        )
    records.sort(key=lambda value: (-value["selectedCount"], value["label"]))
    return {
        "selectedLabel": label,
        "pathBottleneck": path_score,
        "parentCandidate": parent,
        "parentContinuityEdge": parent_edge,
        "lockedSeed": locked_seed.astype(np.uint8),
        "seedAssemblyLabel": seed_assembly_label,
        "selected": selected.astype(np.uint8),
        "selectedSameLabelNeighborCount": same_label_neighbor_count,
        "parentHopDepth": parent_hop_depth,
        **support,
        **association_arrays,
    }, {
        "eligibleSeedComponentCount": int(len(eligible_component)),
        "lockedSeedCount": int(np.count_nonzero(locked_seed)),
        "selectedLockedSeedCount": int(np.count_nonzero(selected & locked_seed)),
        "seedSupport": support_stats,
        "seedAssociation": association_stats,
        "selectedCandidateCount": int(np.count_nonzero(selected)),
        "grownCandidateCount": int(np.count_nonzero(grown)),
        "selectedSpatialKeyCount": int(np.count_nonzero(selected_key_node >= 0)),
        "proposalCount": proposal_count,
        "keyConflictRejectionCount": key_conflict_rejections,
        "protectedSeedRejectionCount": protected_seed_rejections,
        "selectedLabelCount": int(len(selected_labels)),
        "pathBottleneck": _percentile_record(path_score[grown]),
        "grownLocalEvidence": _percentile_record(local_evidence[grown]),
        "grownParentHopDepth": _percentile_record(parent_hop_depth[grown]),
        "sameLabelNeighborMinimumAffinity": settings.minimum_growth_bottleneck,
        "grownSameLabelNeighborCount": _percentile_record(
            same_label_neighbor_count[grown]
        ),
        "grownWithAtLeastTwoSameLabelNeighborsFraction": round(
            float(np.mean(same_label_neighbor_count[grown] >= 2)), 6
        ) if np.any(grown) else None,
        "grownWithAtLeastThreeSameLabelNeighborsFraction": round(
            float(np.mean(same_label_neighbor_count[grown] >= 3)), 6
        ) if np.any(grown) else None,
        "largestLabels": records[:128],
    }


def _label_colors(labels: np.ndarray, maximum: int) -> dict[int, tuple[int, int, int]]:
    value, count = np.unique(labels[labels >= 0], return_counts=True)
    order = np.lexsort((value, -count))[:maximum]
    return {
        int(value[index]): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.09 + 0.61803398875 * rank) % 1.0, 0.68, 0.98
            )
        )
        for rank, index in enumerate(order)
    }


def write_growth_projection(
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    settings: PairedSurfaceGrowthSettings,
    path: str | Path,
    *,
    delta_only: bool = False,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    label = np.asarray(selection["selectedLabel"])
    locked = np.asarray(selection["lockedSeed"]) > 0
    selected = label >= 0
    point = np.asarray(bank["midpointXYZ"])
    colors = _label_colors(label, settings.maximum_preview_labels)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        for value in sorted(colors, reverse=True):
            mask = selected & (label == value)
            if delta_only:
                mask &= ~locked
            points = point[mask]
            if not len(points):
                continue
            normalized = (
                points[:, list(axes)] - world_start_xyz[None, list(axes)]
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
            canvas[y[valid], x[valid]] = colors[value]
        canvas[margin, offset + margin : offset + panel_size - margin] = (64, 72, 84)
        canvas[panel_size - margin, offset + margin : offset + panel_size - margin] = (
            64,
            72,
            84,
        )
        canvas[margin : panel_size - margin, offset + margin] = (64, 72, 84)
        canvas[
            margin : panel_size - margin, offset + panel_size - margin
        ] = (64, 72, 84)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def write_seed_association_audit(
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    maximum_assemblies: int = 4,
    tile_size: int = 400,
) -> tuple[Path, list[dict[str, Any]]]:
    """Show the largest multi-seed assemblies in all three projections."""

    output = Path(path)
    label = np.asarray(selection["selectedLabel"], dtype=np.int32)
    locked = np.asarray(selection["lockedSeed"]) > 0
    seed_assembly = np.asarray(selection["seedAssemblyLabel"], dtype=np.int32)
    seed_component = np.asarray(bank["seedComponentId"], dtype=np.int32)
    point = np.asarray(bank["midpointXYZ"], dtype=np.float32)
    assemblies: list[tuple[int, int, np.ndarray]] = []
    for value in np.unique(label[label >= 0]):
        components = np.unique(
            seed_component[locked & (seed_assembly == value)]
        )
        if len(components) > 1:
            assemblies.append(
                (int(np.count_nonzero(label == value)), int(value), components)
            )
    assemblies.sort(key=lambda value: (-value[0], value[1]))
    assemblies = assemblies[:maximum_assemblies]
    row_count = max(len(assemblies), 1)
    canvas = np.full(
        (row_count * tile_size, 3 * tile_size, 3),
        (7, 16, 20),
        dtype=np.uint8,
    )
    margin = max(16, tile_size // 20)
    axes_values = ((0, 1), (0, 2), (1, 2))
    records: list[dict[str, Any]] = []
    for row, (size, assembly, components) in enumerate(assemblies):
        selected_points = point[label == assembly]
        low = np.min(selected_points, axis=0)
        span = np.maximum(np.ptp(selected_points, axis=0), 1.0)
        records.append(
            {
                "row": row,
                "assemblyLabel": assembly,
                "selectedCandidateCount": size,
                "seedComponents": [int(value) for value in components],
            }
        )
        for column, axes in enumerate(axes_values):
            x0 = column * tile_size
            y0 = row * tile_size

            def project(points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
                normalized = (
                    points[:, list(axes)] - low[None, list(axes)]
                ) / span[None, list(axes)]
                x = np.rint(
                    x0
                    + margin
                    + normalized[:, 0] * (tile_size - 2 * margin - 1)
                ).astype(np.int32)
                y = np.rint(
                    y0
                    + tile_size
                    - margin
                    - normalized[:, 1] * (tile_size - 2 * margin - 1)
                ).astype(np.int32)
                return x, y

            x, y = project(selected_points)
            canvas[y, x] = (71, 85, 105)
            for component_rank, component in enumerate(components):
                seed_points = point[locked & (seed_component == component)]
                x, y = project(seed_points)
                color = tuple(
                    int(round(255.0 * channel))
                    for channel in colorsys.hsv_to_rgb(
                        (0.06 + component_rank * 0.37) % 1.0, 0.8, 1.0
                    )
                )
                for dy in (-1, 0, 1):
                    for dx in (-1, 0, 1):
                        canvas[
                            np.clip(y + dy, y0, y0 + tile_size - 1),
                            np.clip(x + dx, x0, x0 + tile_size - 1),
                        ] = color
                bar_start = x0 + 4 + component_rank * 28
                canvas[
                    y0 + 4 : y0 + 12,
                    bar_start : min(bar_start + 24, x0 + tile_size - 4),
                ] = color
            border = (51, 65, 85)
            canvas[y0 : y0 + 2, x0 : x0 + tile_size] = border
            canvas[
                y0 + tile_size - 2 : y0 + tile_size,
                x0 : x0 + tile_size,
            ] = border
            canvas[y0 : y0 + tile_size, x0 : x0 + 2] = border
            canvas[
                y0 : y0 + tile_size,
                x0 + tile_size - 2 : x0 + tile_size,
            ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output, records


def write_growth_cross_sections(
    source: VolumeSource,
    owned: VoxelBounds,
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    display_high_raw: float,
    sampling_stride: int,
) -> Path:
    output = Path(path)
    volume = source.memmap()
    label = np.asarray(selection["selectedLabel"])
    locked = np.asarray(selection["lockedSeed"]) > 0
    selected = label >= 0
    midpoint = np.asarray(bank["midpointXYZ"])
    world_start = np.asarray(owned.start_xyz) + np.asarray(source.origin_xyz)
    z_values = np.linspace(owned.start_xyz[2], owned.stop_xyz_exclusive[2] - 1, 3)
    y_values = np.linspace(owned.start_xyz[1], owned.stop_xyz_exclusive[1] - 1, 3)
    views = [("z", int(round(value))) for value in z_values] + [
        ("y", int(round(value))) for value in y_values
    ]
    panel_width = owned.shape_xyz[0]
    panel_height = max(owned.shape_xyz[1], owned.shape_xyz[2])
    canvas = np.full(
        (2 * panel_height, 3 * panel_width, 3), (7, 10, 14), dtype=np.uint8
    )
    tolerance = max(1.0, sampling_stride)
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[2]
            near = selected & (np.abs(midpoint[:, 2] - coordinate) <= tolerance)
            x = np.rint(midpoint[near, 0] - world_start[0]).astype(np.int32)
            y = np.rint(midpoint[near, 1] - world_start[1]).astype(np.int32)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[1]
            near = selected & (np.abs(midpoint[:, 1] - coordinate) <= tolerance)
            x = np.rint(midpoint[near, 0] - world_start[0]).astype(np.int32)
            y = np.rint(midpoint[near, 2] - world_start[2]).astype(np.int32)
        gray = np.clip(
            np.asarray(raw, dtype=np.float32) / max(display_high_raw, 1.0) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        locked_near = locked[near]
        valid = (
            (x >= 1)
            & (x < panel.shape[1] - 1)
            & (y >= 1)
            & (y < panel.shape[0] - 1)
        )
        x, y, locked_near = x[valid], y[valid], locked_near[valid]
        color = np.where(
            locked_near[:, None],
            np.asarray((38, 238, 202), dtype=np.uint8),
            np.asarray((255, 164, 62), dtype=np.uint8),
        )
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                panel[y + dy, x + dx] = color
        row, column = divmod(view_index, 3)
        y0 = row * panel_height + (panel_height - panel.shape[0]) // 2
        x0 = column * panel_width
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_paired_surface_growth(
    bank_root: str | Path,
    output_root: str | Path,
    *,
    settings: PairedSurfaceGrowthSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PairedSurfaceGrowthSettings()
    bank_manifest_path, bank_manifest, bank = _load_bank(bank_root)
    stride = _sampling_stride(bank_manifest)
    processing_shape = tuple(
        int(value)
        for value in bank_manifest["geometry"]["processingShapeSamplingXYZ"]
    )
    identity: dict[str, Any] = {
        "schema": PAIRED_SURFACE_GROWTH_SCHEMA,
        "version": PAIRED_SURFACE_GROWTH_VERSION,
        "candidateBank": {
            "manifestPath": str(bank_manifest_path),
            "manifestSha256": sha256_file(bank_manifest_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PAIRED_SURFACE_GROWTH_STEM}.json"
    data_path = output / f"{PAIRED_SURFACE_GROWTH_STEM}.npz"
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
    graph, graph_stats = build_paired_surface_continuity_graph(
        bank,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=resolved,
    )
    graphed = time.monotonic()
    selection, growth_stats = grow_paired_surfaces(
        bank,
        graph,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=resolved,
    )
    grown = time.monotonic()
    arrays = {**selection, **graph}
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"]["metadataPath"]
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]), tuple(owned_record["stopXYZExclusive"])
    )
    world_record = bank_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    projection = write_growth_projection(
        bank, selection, world_start, world_stop, resolved, output / "grown-surfaces.png"
    )
    delta_projection = write_growth_projection(
        bank,
        selection,
        world_start,
        world_stop,
        resolved,
        output / "growth-only.png",
        delta_only=True,
    )
    association_audit, audited_associations = write_seed_association_audit(
        bank,
        selection,
        output / "seed-association-audit.png",
    )
    cross_sections = write_growth_cross_sections(
        source,
        owned,
        bank,
        selection,
        output / "growth-cross-sections.png",
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=stride,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PAIRED_SURFACE_GROWTH_SCHEMA,
        "version": PAIRED_SURFACE_GROWTH_VERSION,
        "state": "complete",
        "identity": identity,
        "geometry": bank_manifest["geometry"],
        "graph": graph_stats,
        "growth": growth_stats,
        "timingSeconds": {
            "graph": round(graphed - stage, 6),
            "growth": round(grown - graphed, 6),
            "writingAndArtifacts": round(finished - grown, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "grownSurfaces": projection.name,
            "growthOnly": delta_projection.name,
            "seedAssociationAudit": association_audit.name,
            "crossSections": cross_sections.name,
        },
        "auditedSeedAssemblies": audited_associations,
        "method": {
            "geometryUnit": "paired lower and upper CT interfaces",
            "seedAssociation": (
                "two-best maximum-bottleneck support with reciprocal immutable-"
                "seed frontiers and spatially broad component-pair evidence"
            ),
            "propagation": (
                "key-exclusive maximum-bottleneck forest from associated clear "
                "seed patches"
            ),
            "changesCandidateGeometry": False,
            "mergesSeedIdentities": True,
            "ambiguityPolicy": (
                "only reciprocally associated seed patches share an identity; "
                "the highest supported candidate alternative then owns each key"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
