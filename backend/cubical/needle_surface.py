from __future__ import annotations

import colorsys
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .needle_field import curvature_aware_tangent_metrics
from .needle_topology import (
    BLOCK_NEEDLE_TOPOLOGY_SCHEMA,
    _load_field_artifact,
    _raw_settings_from_field,
    _transported_fiber_residual_degrees,
    score_stack_fingerprint_pairs,
)


BLOCK_NEEDLE_SURFACE_SCHEMA = "pareidolia.block-acus-needle-surfaces"
BLOCK_NEEDLE_SURFACE_VERSION = 1
BLOCK_NEEDLE_SURFACE_STEM = "block-needle-surfaces-v1"


@dataclass(frozen=True, slots=True)
class BlockNeedleSurfaceSettings:
    """Dataset-independent controls for ordered fiber-strip reconstruction."""

    minimum_topology_component_needles: int = 8
    maximum_longitudinal_deviation_degrees: float = 30.0
    minimum_trace_needles: int = 2
    minimum_crosslinks_per_strip: int = 1
    minimum_crosslink_side_consistency: float = 0.75
    minimum_crosslink_span_candidate_steps: float = 0.5
    maximum_mesh_edge_neighbor_radius_multiplier: float = 1.5
    maximum_triangle_normal_residual_degrees: float = 45.0
    minimum_triangle_area_voxels_squared: float = 0.25
    minimum_chart_separation_candidate_fraction: float = 0.05
    chart_solver_relative_tolerance: float = 1.0e-7
    chart_solver_maximum_iterations: int = 2048
    maximum_preview_components: int = 12

    def __post_init__(self) -> None:
        positive_integer = (
            self.minimum_topology_component_needles,
            self.minimum_trace_needles,
            self.minimum_crosslinks_per_strip,
            self.chart_solver_maximum_iterations,
            self.maximum_preview_components,
        )
        if any(value < 1 for value in positive_integer):
            raise ValueError("needle-surface integer settings must be positive")
        if not 0.0 < self.maximum_longitudinal_deviation_degrees < 89.0:
            raise ValueError("longitudinal deviation must lie in (0, 89) degrees")
        if not 0.5 <= self.minimum_crosslink_side_consistency <= 1.0:
            raise ValueError("crosslink side consistency must lie in [0.5, 1]")
        positive = (
            self.maximum_mesh_edge_neighbor_radius_multiplier,
            self.maximum_triangle_normal_residual_degrees,
            self.minimum_triangle_area_voxels_squared,
            self.minimum_chart_separation_candidate_fraction,
            self.chart_solver_relative_tolerance,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("needle-surface scales must be finite and positive")
        if (
            not math.isfinite(self.minimum_crosslink_span_candidate_steps)
            or self.minimum_crosslink_span_candidate_steps < 0.0
        ):
            raise ValueError("crosslink span must be finite and nonnegative")
        if self.maximum_triangle_normal_residual_degrees >= 90.0:
            raise ValueError("triangle normal residual must be below 90 degrees")

    def record(self) -> dict[str, Any]:
        return asdict(self)


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        root = value
        while int(self.parent[root]) != root:
            root = int(self.parent[root])
        while int(self.parent[value]) != value:
            following = int(self.parent[value])
            self.parent[value] = root
            value = following
        return root

    def union(self, first: int, second: int) -> bool:
        first = self.find(first)
        second = self.find(second)
        if first == second:
            return False
        if self.size[first] < self.size[second]:
            first, second = second, first
        self.parent[second] = first
        self.size[first] += self.size[second]
        return True

    def roots(self) -> np.ndarray:
        return np.asarray([self.find(value) for value in range(len(self.parent))])


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


def _load_topology_artifact(
    topology_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    root = Path(topology_root).resolve()
    manifest_path = root if root.is_file() else root / "block-needle-topology-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != BLOCK_NEEDLE_TOPOLOGY_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("needle surfaces require a complete needle-topology artifact")
    data_record = manifest["data"]
    data_path = manifest_path.parent / str(data_record["path"])
    if sha256_file(data_path) != data_record["sha256"]:
        raise ValueError("needle-topology data hash differs from its manifest")
    required = (
        "edgeFirstNeedle",
        "edgeSecondNeedle",
        "edgeScore",
        "edgeGrowthEligible",
        "edgeSelected",
        "plyComponentId",
        "plyComponentSize",
        "stackDensity",
        "stackOrientationMoment",
    )
    with np.load(data_path) as values:
        missing = set(required) - set(values.files)
        if missing:
            raise ValueError(f"needle topology is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(values[name]) for name in required}
    edge_count = len(arrays["edgeFirstNeedle"])
    if any(len(arrays[name]) != edge_count for name in required[:5]):
        raise ValueError("needle-topology edge arrays disagree in length")
    return manifest_path, manifest, arrays


def synchronize_axial_gauge(
    vectors_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_weight: np.ndarray,
    selected_edge: np.ndarray,
) -> tuple[np.ndarray, dict[str, int]]:
    """Choose deterministic signs from a maximum-support spanning forest.

    Vector sign is a gauge only.  The returned signs make local transport
    explicit for ordering and meshing; disagreement is reported, never used as
    physical evidence or as a reason to split a carrier.
    """

    vectors = np.asarray(vectors_xyz, dtype=np.float64)
    count = len(vectors)
    chosen = np.flatnonzero(selected_edge)
    strength = edge_weight[chosen] * np.abs(
        np.einsum("ij,ij->i", vectors[first[chosen]], vectors[second[chosen]])
    )
    order = chosen[np.lexsort((chosen, -strength))]
    forest = _DisjointSet(count)
    adjacency: list[list[tuple[int, int]]] = [[] for _ in range(count)]
    forest_edges = 0
    for edge_index in order:
        left = int(first[edge_index])
        right = int(second[edge_index])
        if not forest.union(left, right):
            continue
        relative = -1 if float(np.dot(vectors[left], vectors[right])) < 0.0 else 1
        adjacency[left].append((right, relative))
        adjacency[right].append((left, relative))
        forest_edges += 1
    sign = np.zeros(count, dtype=np.int8)
    for seed in range(count):
        if sign[seed]:
            continue
        sign[seed] = 1
        stack = [seed]
        while stack:
            node = stack.pop()
            for neighbor, relative in adjacency[node]:
                expected = int(sign[node]) * relative
                if not sign[neighbor]:
                    sign[neighbor] = expected
                    stack.append(neighbor)
    signed_dot = (
        sign[first[chosen]].astype(np.float64)
        * sign[second[chosen]].astype(np.float64)
        * np.einsum("ij,ij->i", vectors[first[chosen]], vectors[second[chosen]])
    )
    return sign, {
        "forestEdges": forest_edges,
        "selectedEdges": len(chosen),
        "cycleGaugeDisagreements": int(np.count_nonzero(signed_dot < 0.0)),
    }


def _edge_tangent_coordinates(
    center_xyz: np.ndarray,
    fiber_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    fiber_sign: np.ndarray,
    normal_sign: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> dict[str, np.ndarray]:
    center = np.asarray(center_xyz, dtype=np.float64)
    fiber = np.asarray(fiber_xyz, dtype=np.float64) * fiber_sign[:, None]
    normal = np.asarray(normal_xyz, dtype=np.float64) * normal_sign[:, None]
    displacement = center[second] - center[first]
    middle_normal = normal[first] + normal[second]
    middle_normal_length = np.linalg.norm(middle_normal, axis=1, keepdims=True)
    fallback = normal[first]
    middle_normal = np.where(
        middle_normal_length > 1.0e-8,
        middle_normal / np.maximum(middle_normal_length, 1.0e-8),
        fallback,
    )
    first_fiber = fiber[first] - (
        np.einsum("ij,ij->i", fiber[first], middle_normal)[:, None]
        * middle_normal
    )
    second_fiber = fiber[second] - (
        np.einsum("ij,ij->i", fiber[second], middle_normal)[:, None]
        * middle_normal
    )
    first_fiber /= np.maximum(
        np.linalg.norm(first_fiber, axis=1, keepdims=True), 1.0e-8
    )
    second_fiber /= np.maximum(
        np.linalg.norm(second_fiber, axis=1, keepdims=True), 1.0e-8
    )
    reverse = np.einsum("ij,ij->i", first_fiber, second_fiber) < 0.0
    second_fiber[reverse] *= -1.0
    middle_fiber = first_fiber + second_fiber
    middle_fiber /= np.maximum(
        np.linalg.norm(middle_fiber, axis=1, keepdims=True), 1.0e-8
    )
    middle_cross = np.cross(middle_normal, middle_fiber)
    middle_cross /= np.maximum(
        np.linalg.norm(middle_cross, axis=1, keepdims=True), 1.0e-8
    )
    along_signed = np.einsum("ij,ij->i", displacement, middle_fiber)
    cross_signed = np.einsum("ij,ij->i", displacement, middle_cross)
    height_signed = np.einsum("ij,ij->i", displacement, middle_normal)
    tangent_length = np.hypot(along_signed, cross_signed)
    return {
        "lengthVoxels": np.linalg.norm(displacement, axis=1).astype(np.float32),
        "alongSignedVoxels": along_signed.astype(np.float32),
        "crossSignedVoxels": cross_signed.astype(np.float32),
        "heightSignedVoxels": height_signed.astype(np.float32),
        "longitudinalFraction": np.divide(
            np.abs(along_signed),
            np.maximum(tangent_length, 1.0e-8),
        ).astype(np.float32),
    }


def build_complete_component_pair_graph(
    center_xyz: np.ndarray,
    fiber_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    stack_density: np.ndarray,
    stack_orientation_moment: np.ndarray,
    topology_component_id: np.ndarray,
    topology_component_size: np.ndarray,
    artifact_first: np.ndarray,
    artifact_second: np.ndarray,
    artifact_growth_edge: np.ndarray,
    artifact_selected_edge: np.ndarray,
    *,
    minimum_component_needles: int,
    neighbor_radius_voxels: float,
    tangent_compatibility_sigma_voxels: float,
    minimum_curvature_radius_voxels: float,
    minimum_curved_affinity: float,
    growth_caps: Mapping[str, float],
    fingerprint_chunk_edges: int = 32768,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Re-evaluate every within-radius pair inside frozen ply components."""

    center = np.asarray(center_xyz, dtype=np.float32)
    component = np.asarray(topology_component_id, dtype=np.int32)
    component_size = np.asarray(topology_component_size, dtype=np.int32)
    pair_first: list[np.ndarray] = []
    pair_second: list[np.ndarray] = []
    component_values = np.unique(
        component[component_size >= minimum_component_needles]
    )
    radius_squared = neighbor_radius_voxels**2
    enumerated_pairs = 0
    for value in component_values:
        nodes = np.flatnonzero(component == value).astype(np.int32)
        low, high = np.triu_indices(len(nodes), 1)
        enumerated_pairs += len(low)
        displacement = center[nodes[high]] - center[nodes[low]]
        inside = np.einsum("ij,ij->i", displacement, displacement) <= radius_squared
        if np.any(inside):
            pair_first.append(nodes[low[inside]])
            pair_second.append(nodes[high[inside]])
    if pair_first:
        first = np.concatenate(pair_first).astype(np.int32, copy=False)
        second = np.concatenate(pair_second).astype(np.int32, copy=False)
    else:
        first = np.empty(0, dtype=np.int32)
        second = np.empty(0, dtype=np.int32)
    displacement = center[second] - center[first]
    geometry = curvature_aware_tangent_metrics(
        displacement,
        normal_xyz[first],
        normal_xyz[second],
        compatibility_sigma_voxels=tangent_compatibility_sigma_voxels,
        minimum_curvature_radius_voxels=minimum_curvature_radius_voxels,
    )
    fiber_residual = _transported_fiber_residual_degrees(
        fiber_xyz, normal_xyz, first, second
    )
    fingerprint_mismatch = score_stack_fingerprint_pairs(
        stack_density,
        stack_orientation_moment,
        normal_xyz,
        first,
        second,
        chunk_edges=fingerprint_chunk_edges,
    )
    midpoint = geometry["midpointLayerShiftVoxels"]
    bend = geometry["bendModelResidualVoxels"]
    radius = geometry["curvatureRadiusVoxels"]
    eligible = (
        (geometry["affinity"] >= minimum_curved_affinity)
        & (midpoint <= float(growth_caps["midpointLayerShiftVoxels"]))
        & (bend <= float(growth_caps["bendModelResidualVoxels"]))
        & (fiber_residual <= float(growth_caps["fiberResidualDegrees"]))
        & (
            fingerprint_mismatch
            <= float(growth_caps["stackFingerprintMismatch"])
        )
        & (radius >= minimum_curvature_radius_voxels)
    )
    caps = {
        name: max(float(value), 1.0e-6)
        for name, value in growth_caps.items()
    }
    pair_score = (
        np.exp(-0.5 * (midpoint / caps["midpointLayerShiftVoxels"]) ** 2)
        * np.exp(-0.5 * (bend / caps["bendModelResidualVoxels"]) ** 2)
        * np.exp(-0.5 * (fiber_residual / caps["fiberResidualDegrees"]) ** 2)
        * np.exp(
            -0.5
            * (
                fingerprint_mismatch / caps["stackFingerprintMismatch"]
            )
            ** 2
        )
    ).astype(np.float32)
    count = len(center)
    pair_key = first.astype(np.int64) * count + second
    artifact_key = artifact_first.astype(np.int64) * count + artifact_second
    growth_key = np.sort(artifact_key[np.asarray(artifact_growth_edge, dtype=bool)])
    selected_key = np.sort(
        artifact_key[np.asarray(artifact_selected_edge, dtype=bool)]
    )
    existing_growth = np.isin(pair_key, growth_key, assume_unique=False)
    selected_anchor = np.isin(pair_key, selected_key, assume_unique=False)
    result = {
        "first": first[eligible],
        "second": second[eligible],
        "score": pair_score[eligible],
        "selectedAnchor": selected_anchor[eligible],
        "existingGrowthEdge": existing_growth[eligible],
    }
    return result, {
        "components": len(component_values),
        "allComponentPairs": enumerated_pairs,
        "pairsInsideNeighborRadius": len(first),
        "eligiblePairs": int(np.count_nonzero(eligible)),
        "existingGrowthPairs": int(np.count_nonzero(eligible & existing_growth)),
        "recoveredPairsOmittedByFixedDegreeGraph": int(
            np.count_nonzero(eligible & ~existing_growth)
        ),
        "selectedAnchorPairs": int(np.count_nonzero(eligible & selected_anchor)),
        "midpointLayerShiftVoxels": _percentile_record(midpoint[eligible]),
        "bendModelResidualVoxels": _percentile_record(bend[eligible]),
        "fiberResidualDegrees": _percentile_record(fiber_residual[eligible]),
        "stackFingerprintMismatch": _percentile_record(
            fingerprint_mismatch[eligible]
        ),
    }


def _ordered_trace_nodes(
    count: int,
    first: np.ndarray,
    second: np.ndarray,
    longitudinal_edge: np.ndarray,
    center_xyz: np.ndarray,
    signed_fiber_xyz: np.ndarray,
) -> tuple[list[np.ndarray], np.ndarray, np.ndarray, np.ndarray]:
    adjacency: list[list[int]] = [[] for _ in range(count)]
    for left, right in zip(first[longitudinal_edge], second[longitudinal_edge]):
        adjacency[int(left)].append(int(right))
        adjacency[int(right)].append(int(left))
    if any(len(values) > 2 for values in adjacency):
        raise ValueError("longitudinal trace selection produced a branch")
    visited = np.zeros(count, dtype=bool)
    traces: list[np.ndarray] = []
    trace_id = np.full(count, -1, dtype=np.int32)
    trace_position = np.zeros(count, dtype=np.int32)
    trace_size = np.ones(count, dtype=np.int32)
    for seed in range(count):
        if visited[seed]:
            continue
        group: list[int] = []
        pending = [seed]
        visited[seed] = True
        while pending:
            node = pending.pop()
            group.append(node)
            for neighbor in adjacency[node]:
                if not visited[neighbor]:
                    visited[neighbor] = True
                    pending.append(neighbor)
        if len(group) == 1:
            ordered = group
        else:
            endpoints = sorted(value for value in group if len(adjacency[value]) == 1)
            if len(endpoints) != 2:
                raise ValueError("longitudinal trace is not an open chain")
            start = endpoints[0]
            following = adjacency[start][0]
            if float(
                np.dot(center_xyz[following] - center_xyz[start], signed_fiber_xyz[start])
            ) < 0.0:
                start = endpoints[1]
            ordered = []
            previous = -1
            current = start
            while True:
                ordered.append(current)
                following_values = [
                    value for value in adjacency[current] if value != previous
                ]
                if not following_values:
                    break
                previous, current = current, following_values[0]
        trace_index = len(traces)
        values = np.asarray(ordered, dtype=np.int32)
        traces.append(values)
        trace_id[values] = trace_index
        trace_position[values] = np.arange(len(values), dtype=np.int32)
        trace_size[values] = len(values)
    return traces, trace_id, trace_position, trace_size


def select_ordered_fiber_traces(
    center_xyz: np.ndarray,
    fiber_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    selected_edge: np.ndarray,
    topology_component_size: np.ndarray,
    *,
    depth_kernel_voxels: float,
    settings: BlockNeedleSurfaceSettings,
) -> dict[str, Any]:
    """Select a branch-free predecessor/successor graph from ply edges."""

    count = len(center_xyz)
    fiber_sign, fiber_gauge = synchronize_axial_gauge(
        fiber_xyz, first, second, edge_score, selected_edge
    )
    normal_sign, normal_gauge = synchronize_axial_gauge(
        normal_xyz, first, second, edge_score, selected_edge
    )
    coordinates = _edge_tangent_coordinates(
        center_xyz,
        fiber_xyz,
        normal_xyz,
        fiber_sign,
        normal_sign,
        first,
        second,
    )
    minimum_component = settings.minimum_topology_component_needles
    eligible_node = topology_component_size >= minimum_component
    longitudinal_floor = math.cos(
        math.radians(settings.maximum_longitudinal_deviation_degrees)
    )
    eligible = (
        selected_edge
        & eligible_node[first]
        & eligible_node[second]
        & (coordinates["longitudinalFraction"] >= longitudinal_floor)
        & (np.abs(coordinates["alongSignedVoxels"]) > 1.0e-6)
    )
    height = np.abs(coordinates["heightSignedVoxels"])
    length = coordinates["lengthVoxels"]
    trace_score = (
        edge_score
        * coordinates["longitudinalFraction"] ** 2
        * np.exp(-0.5 * (height / max(depth_kernel_voxels, 1.0e-6)) ** 2)
        / np.sqrt(np.maximum(length, 0.25))
    )
    candidates = np.flatnonzero(eligible)
    order = candidates[np.lexsort((candidates, length[candidates], -trace_score[candidates]))]
    signed_fiber = np.asarray(fiber_xyz, dtype=np.float64) * fiber_sign[:, None]
    slot_used = np.zeros((count, 2), dtype=bool)
    trace_union = _DisjointSet(count)
    chosen = np.zeros(len(first), dtype=bool)
    slot_conflicts = 0
    cycle_conflicts = 0
    for edge_index in order:
        left = int(first[edge_index])
        right = int(second[edge_index])
        displacement = center_xyz[right] - center_xyz[left]
        left_side = 1 if float(np.dot(displacement, signed_fiber[left])) > 0.0 else 0
        right_side = 1 if float(np.dot(-displacement, signed_fiber[right])) > 0.0 else 0
        if slot_used[left, left_side] or slot_used[right, right_side]:
            slot_conflicts += 1
            continue
        if not trace_union.union(left, right):
            cycle_conflicts += 1
            continue
        slot_used[left, left_side] = True
        slot_used[right, right_side] = True
        chosen[edge_index] = True
    traces, trace_id, trace_position, trace_size = _ordered_trace_nodes(
        count,
        first,
        second,
        chosen,
        np.asarray(center_xyz, dtype=np.float64),
        signed_fiber,
    )
    return {
        "fiberSign": fiber_sign,
        "normalSign": normal_sign,
        "coordinates": coordinates,
        "longitudinalEdge": chosen,
        "traces": traces,
        "traceId": trace_id,
        "tracePosition": trace_position,
        "traceSize": trace_size,
        "summary": {
            "fiberGauge": fiber_gauge,
            "normalGauge": normal_gauge,
            "eligibleLongitudinalEdges": len(candidates),
            "selectedLongitudinalEdges": int(np.count_nonzero(chosen)),
            "slotConflicts": slot_conflicts,
            "cycleConflicts": cycle_conflicts,
            "traces": len(traces),
            "multiNeedleTraces": sum(len(value) >= 2 for value in traces),
            "traceNeedles": {
                str(size): int(sum(len(value) >= size for value in traces))
                for size in (2, 4, 8, 16, 32, 64)
            },
            "traceSize": _percentile_record(
                np.asarray([len(value) for value in traces], dtype=np.float64)
            ),
        },
    }


def _longest_monotone_crosslinks(
    edge_indices: np.ndarray,
    first_position: np.ndarray,
    second_position: np.ndarray,
    edge_score: np.ndarray,
) -> tuple[np.ndarray, bool]:
    if not len(edge_indices):
        return np.empty(0, dtype=np.int32), False
    if len(edge_indices) == 1:
        return edge_indices.astype(np.int32, copy=True), False
    positions_first = np.asarray(first_position, dtype=np.int32)
    positions_second = np.asarray(second_position, dtype=np.int32)
    weights = edge_score[edge_indices].astype(np.float64)
    if positions_first.shape != edge_indices.shape or positions_second.shape != edge_indices.shape:
        raise ValueError("crosslink positions must match their edge indices")
    total = max(float(np.sum(weights)), 1.0e-12)
    mean_first = float(np.sum(weights * positions_first) / total)
    mean_second = float(np.sum(weights * positions_second) / total)
    covariance = float(
        np.sum(
            weights
            * (positions_first - mean_first)
            * (positions_second - mean_second)
        )
    )
    reverse_second = covariance < 0.0
    mapped_second = -positions_second if reverse_second else positions_second
    best_by_pair: dict[tuple[int, int], int] = {}
    for local_index, edge_index in enumerate(edge_indices):
        key = (int(positions_first[local_index]), int(mapped_second[local_index]))
        prior_local = best_by_pair.get(key)
        if prior_local is None or weights[local_index] > weights[prior_local]:
            best_by_pair[key] = local_index
    ordered_local = np.asarray(
        sorted(
            best_by_pair.values(),
            key=lambda local_index: (
                int(positions_first[local_index]),
                -int(positions_second[local_index])
                if reverse_second
                else int(positions_second[local_index]),
                -float(weights[local_index]),
                int(edge_indices[local_index]),
            ),
        ),
        dtype=np.int32,
    )
    if len(ordered_local) < 2:
        return np.empty(0, dtype=np.int32), reverse_second
    x = positions_first[ordered_local]
    y = positions_second[ordered_local]
    if reverse_second:
        y = -y
    ordered_weight = weights[ordered_local]
    value = ordered_weight.copy()
    count = np.ones(len(ordered_local), dtype=np.int32)
    previous = np.full(len(ordered_local), -1, dtype=np.int32)
    for current in range(len(ordered_local)):
        valid = np.flatnonzero((x[:current] < x[current]) & (y[:current] < y[current]))
        if not len(valid):
            continue
        candidate_value = value[valid] + float(ordered_weight[current])
        candidate_count = count[valid] + 1
        ranking = np.lexsort((valid, -candidate_count, -candidate_value))
        best = int(valid[ranking[0]])
        value[current] = candidate_value[ranking[0]]
        count[current] = candidate_count[ranking[0]]
        previous[current] = best
    finish_order = np.lexsort((np.arange(len(ordered_local)), -count, -value))
    current = int(finish_order[0])
    chain: list[int] = []
    while current >= 0:
        chain.append(int(edge_indices[ordered_local[current]]))
        current = int(previous[current])
    chain.reverse()
    return np.asarray(chain, dtype=np.int32), reverse_second


def _triangle_geometry(
    nodes: tuple[int, int, int],
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
) -> tuple[tuple[int, int, int], float, float, float]:
    points = center_xyz[np.asarray(nodes)]
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    norm = float(np.linalg.norm(cross))
    if norm <= 1.0e-12:
        return nodes, 0.0, 90.0, float("inf")
    triangle_normal = cross / norm
    reference = np.sum(signed_normal_xyz[np.asarray(nodes)], axis=0)
    reference_length = float(np.linalg.norm(reference))
    if reference_length <= 1.0e-12:
        reference = signed_normal_xyz[nodes[0]]
    else:
        reference /= reference_length
    oriented = nodes
    if float(np.dot(triangle_normal, reference)) < 0.0:
        oriented = (nodes[0], nodes[2], nodes[1])
        triangle_normal *= -1.0
    residual = math.degrees(
        math.acos(np.clip(float(np.dot(triangle_normal, reference)), -1.0, 1.0))
    )
    edges = (
        float(np.linalg.norm(points[1] - points[0])),
        float(np.linalg.norm(points[2] - points[1])),
        float(np.linalg.norm(points[0] - points[2])),
    )
    return oriented, 0.5 * norm, residual, max(edges)


def _zipper_interval(
    first_nodes: np.ndarray,
    second_nodes: np.ndarray,
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    *,
    maximum_edge_voxels: float,
    maximum_normal_residual_degrees: float,
    minimum_area_voxels_squared: float,
) -> tuple[list[tuple[int, int, int]], list[float], list[float]]:
    first_position = 0
    second_position = 0
    triangles: list[tuple[int, int, int]] = []
    areas: list[float] = []
    residuals: list[float] = []
    while (
        first_position < len(first_nodes) - 1
        or second_position < len(second_nodes) - 1
    ):
        options: list[tuple[float, int, tuple[int, int, int]]] = []
        if first_position < len(first_nodes) - 1:
            nodes = (
                int(first_nodes[first_position]),
                int(first_nodes[first_position + 1]),
                int(second_nodes[second_position]),
            )
            diagonal = float(
                np.linalg.norm(
                    center_xyz[first_nodes[first_position + 1]]
                    - center_xyz[second_nodes[second_position]]
                )
            )
            options.append((diagonal, 0, nodes))
        if second_position < len(second_nodes) - 1:
            nodes = (
                int(first_nodes[first_position]),
                int(second_nodes[second_position + 1]),
                int(second_nodes[second_position]),
            )
            diagonal = float(
                np.linalg.norm(
                    center_xyz[first_nodes[first_position]]
                    - center_xyz[second_nodes[second_position + 1]]
                )
            )
            options.append((diagonal, 1, nodes))
        accepted: tuple[int, tuple[int, int, int], float, float] | None = None
        for _diagonal, advance, nodes in sorted(options):
            oriented, area, residual, maximum_edge = _triangle_geometry(
                nodes, center_xyz, signed_normal_xyz
            )
            if (
                area >= minimum_area_voxels_squared
                and residual <= maximum_normal_residual_degrees
                and maximum_edge <= maximum_edge_voxels
            ):
                accepted = advance, oriented, area, residual
                break
        if accepted is None:
            return [], [], []
        advance, triangle, area, residual = accepted
        triangles.append(triangle)
        areas.append(area)
        residuals.append(residual)
        if advance == 0:
            first_position += 1
        else:
            second_position += 1
    return triangles, areas, residuals


def _solve_weighted_graph_coordinate(
    node_count: int,
    first: np.ndarray,
    second: np.ndarray,
    weight: np.ndarray,
    right_hand_side: np.ndarray,
    *,
    relative_tolerance: float,
    maximum_iterations: int,
) -> tuple[np.ndarray, int, bool]:
    """Solve one anchored graph Laplacian with matrix-free preconditioned CG."""

    diagonal = np.bincount(
        np.concatenate((first, second)),
        weights=np.concatenate((weight, weight)),
        minlength=node_count,
    )
    reduced_diagonal = np.maximum(diagonal[1:], 1.0e-12)
    target = np.asarray(right_hand_side[1:], dtype=np.float64)
    target_norm = float(np.linalg.norm(target))
    solution = np.zeros(node_count - 1, dtype=np.float64)
    if target_norm <= 1.0e-14:
        return np.zeros(node_count, dtype=np.float64), 0, True

    def multiply(values: np.ndarray) -> np.ndarray:
        full = np.zeros(node_count, dtype=np.float64)
        full[1:] = values
        result = diagonal * full
        result -= np.bincount(
            first,
            weights=weight * full[second],
            minlength=node_count,
        )
        result -= np.bincount(
            second,
            weights=weight * full[first],
            minlength=node_count,
        )
        return result[1:]

    residual = target.copy()
    preconditioned = residual / reduced_diagonal
    direction = preconditioned.copy()
    residual_dot = float(np.dot(residual, preconditioned))
    threshold = relative_tolerance * max(target_norm, 1.0)
    converged = float(np.linalg.norm(residual)) <= threshold
    iteration = 0
    while not converged and iteration < maximum_iterations:
        product = multiply(direction)
        denominator = float(np.dot(direction, product))
        if denominator <= 1.0e-20:
            break
        step = residual_dot / denominator
        solution += step * direction
        residual -= step * product
        iteration += 1
        converged = float(np.linalg.norm(residual)) <= threshold
        if converged:
            break
        preconditioned = residual / reduced_diagonal
        following_dot = float(np.dot(residual, preconditioned))
        direction = preconditioned + (following_dot / residual_dot) * direction
        residual_dot = following_dot
    full_solution = np.zeros(node_count, dtype=np.float64)
    full_solution[1:] = solution
    return full_solution, iteration, converged


def integrate_intrinsic_surface_charts(
    center_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    geometric_candidate_edge: np.ndarray,
    topology_component_id: np.ndarray,
    topology_component_size: np.ndarray,
    along_signed_voxels: np.ndarray,
    cross_signed_voxels: np.ndarray,
    *,
    minimum_component_needles: int,
    solver_relative_tolerance: float = 1.0e-7,
    solver_maximum_iterations: int = 2048,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Integrate local fiber/cross-fiber chords into per-carrier 2D charts."""

    count = len(center_xyz)
    same_component = topology_component_id[first] == topology_component_id[second]
    chart_edge = (
        geometric_candidate_edge
        & same_component
        & (topology_component_size[first] >= minimum_component_needles)
        & (topology_component_size[second] >= minimum_component_needles)
    )
    chart_uv = np.full((count, 2), np.nan, dtype=np.float64)
    edge_residual = np.full(len(first), np.nan, dtype=np.float32)
    component_values = np.unique(
        topology_component_id[
            topology_component_size >= minimum_component_needles
        ]
    )
    solved_components = 0
    unconverged_components = 0
    solver_iterations: list[int] = []
    for component in component_values:
        nodes = np.flatnonzero(topology_component_id == component)
        component_edge = np.flatnonzero(
            chart_edge & (topology_component_id[first] == component)
        )
        if len(nodes) < 2 or not len(component_edge):
            continue
        local_index = {int(node): index for index, node in enumerate(nodes)}
        local_first = np.asarray(
            [local_index[int(value)] for value in first[component_edge]],
            dtype=np.int32,
        )
        local_second = np.asarray(
            [local_index[int(value)] for value in second[component_edge]],
            dtype=np.int32,
        )
        weight = np.maximum(
            np.asarray(edge_score[component_edge], dtype=np.float64), 1.0e-3
        )
        right_hand_side = np.zeros((len(nodes), 2), dtype=np.float64)
        delta = np.column_stack(
            (
                along_signed_voxels[component_edge],
                cross_signed_voxels[component_edge],
            )
        ).astype(np.float64)
        np.add.at(right_hand_side, local_first, -weight[:, None] * delta)
        np.add.at(right_hand_side, local_second, weight[:, None] * delta)
        values = np.zeros((len(nodes), 2), dtype=np.float64)
        component_converged = True
        component_iterations = 0
        for axis in range(2):
            solved, iterations, converged = _solve_weighted_graph_coordinate(
                len(nodes),
                local_first,
                local_second,
                weight,
                right_hand_side[:, axis],
                relative_tolerance=solver_relative_tolerance,
                maximum_iterations=solver_maximum_iterations,
            )
            values[:, axis] = solved
            component_iterations = max(component_iterations, iterations)
            component_converged &= converged
        if not component_converged:
            unconverged_components += 1
        solver_iterations.append(component_iterations)
        values -= np.mean(values, axis=0, keepdims=True)
        chart_uv[nodes] = values
        predicted = values[
            local_second
        ] - values[local_first]
        observed = np.column_stack(
            (
                along_signed_voxels[component_edge],
                cross_signed_voxels[component_edge],
            )
        )
        edge_residual[component_edge] = np.linalg.norm(
            predicted - observed, axis=1
        ).astype(np.float32)
        solved_components += 1
    finite_residual = edge_residual[np.isfinite(edge_residual)]
    return chart_uv.astype(np.float32), edge_residual, {
        "eligibleComponents": len(component_values),
        "solvedComponents": solved_components,
        "singularLeastSquaresComponents": unconverged_components,
        "unconvergedComponents": unconverged_components,
        "solver": "matrix-free Jacobi-preconditioned conjugate gradient",
        "solverRelativeTolerance": solver_relative_tolerance,
        "solverMaximumIterations": solver_maximum_iterations,
        "solverIterations": _percentile_record(
            np.asarray(solver_iterations, dtype=np.float64)
        ),
        "constraintEdges": int(np.count_nonzero(chart_edge)),
        "edgeIntegrationResidualVoxels": _percentile_record(finite_residual),
    }


def _circumcircle_contains(
    first: np.ndarray,
    second: np.ndarray,
    third: np.ndarray,
    point: np.ndarray,
) -> bool:
    denominator = 2.0 * (
        first[0] * (second[1] - third[1])
        + second[0] * (third[1] - first[1])
        + third[0] * (first[1] - second[1])
    )
    if abs(float(denominator)) <= 1.0e-12:
        return False
    first_squared = float(np.dot(first, first))
    second_squared = float(np.dot(second, second))
    third_squared = float(np.dot(third, third))
    center = np.asarray(
        (
            (
                first_squared * (second[1] - third[1])
                + second_squared * (third[1] - first[1])
                + third_squared * (first[1] - second[1])
            )
            / denominator,
            (
                first_squared * (third[0] - second[0])
                + second_squared * (first[0] - third[0])
                + third_squared * (second[0] - first[0])
            )
            / denominator,
        )
    )
    radius_squared = float(np.sum((center - first) ** 2))
    distance_squared = float(np.sum((center - point) ** 2))
    return distance_squared <= radius_squared + 1.0e-9 * max(radius_squared, 1.0)


def _delaunay_triangles(points_uv: np.ndarray) -> np.ndarray:
    """Return deterministic Bowyer-Watson triangles for unique 2D points."""

    points = np.asarray(points_uv, dtype=np.float64)
    if len(points) < 3:
        return np.empty((0, 3), dtype=np.int32)
    low = np.min(points, axis=0)
    high = np.max(points, axis=0)
    span = max(float(np.max(high - low)), 1.0)
    center = 0.5 * (low + high)
    super_points = np.asarray(
        (
            (center[0] - 32.0 * span, center[1] - 16.0 * span),
            (center[0], center[1] + 32.0 * span),
            (center[0] + 32.0 * span, center[1] - 16.0 * span),
        ),
        dtype=np.float64,
    )
    extended = np.vstack((points, super_points))
    super_triangle = (len(points), len(points) + 1, len(points) + 2)
    triangles: list[tuple[int, int, int]] = [super_triangle]
    insertion_order = np.lexsort(
        (np.arange(len(points)), points[:, 1], points[:, 0])
    )
    for point_index in insertion_order:
        bad: list[int] = []
        edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for triangle_index, triangle in enumerate(triangles):
            if _circumcircle_contains(
                extended[triangle[0]],
                extended[triangle[1]],
                extended[triangle[2]],
                extended[point_index],
            ):
                bad.append(triangle_index)
                for edge_index, left in enumerate(triangle):
                    right = triangle[(edge_index + 1) % 3]
                    edge_count[(min(left, right), max(left, right))] += 1
        if not bad:
            continue
        bad_set = set(bad)
        triangles = [
            triangle
            for triangle_index, triangle in enumerate(triangles)
            if triangle_index not in bad_set
        ]
        boundary = sorted(edge for edge, count in edge_count.items() if count == 1)
        triangles.extend((edge[0], edge[1], int(point_index)) for edge in boundary)
    result = {
        tuple(sorted(triangle))
        for triangle in triangles
        if all(value < len(points) for value in triangle)
        and len(set(triangle)) == 3
    }
    return np.asarray(sorted(result), dtype=np.int32).reshape((-1, 3))


def triangulate_intrinsic_surface_charts(
    center_xyz: np.ndarray,
    signed_normal_xyz: np.ndarray,
    chart_uv: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    geometric_candidate_edge: np.ndarray,
    selected_edge: np.ndarray,
    topology_component_id: np.ndarray,
    topology_component_size: np.ndarray,
    *,
    minimum_component_needles: int,
    minimum_chart_separation_voxels: float,
    maximum_edge_voxels: float,
    maximum_normal_residual_degrees: float,
    minimum_area_voxels_squared: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Triangulate each integrated carrier chart and map it back to 3D."""

    count = len(center_xyz)
    same_component = topology_component_id[first] == topology_component_id[second]
    eligible_edge = (
        geometric_candidate_edge
        & same_component
        & (topology_component_size[first] >= minimum_component_needles)
        & (topology_component_size[second] >= minimum_component_needles)
    )
    edge_by_key = {
        int(first[edge_index]) * count + int(second[edge_index]): int(edge_index)
        for edge_index in np.flatnonzero(eligible_edge)
    }
    triangles: list[tuple[int, int, int]] = []
    areas: list[float] = []
    residuals: list[float] = []
    raw_delaunay_triangles = 0
    rejected_duplicate_uv_nodes = 0
    rejected_unsupported_edge = 0
    rejected_without_selected_edge = 0
    rejected_geometry = 0
    component_values = np.unique(
        topology_component_id[topology_component_size >= minimum_component_needles]
    )
    for component in component_values:
        nodes = np.flatnonzero(
            (topology_component_id == component)
            & np.all(np.isfinite(chart_uv), axis=1)
        )
        if len(nodes) < 3:
            continue
        order = np.lexsort((nodes, chart_uv[nodes, 1], chart_uv[nodes, 0]))
        retained_nodes: list[int] = []
        for node in nodes[order]:
            if retained_nodes and np.any(
                np.linalg.norm(chart_uv[np.asarray(retained_nodes)] - chart_uv[node], axis=1)
                < minimum_chart_separation_voxels
            ):
                rejected_duplicate_uv_nodes += 1
                continue
            retained_nodes.append(int(node))
        if len(retained_nodes) < 3:
            continue
        retained = np.asarray(retained_nodes, dtype=np.int32)
        local_triangles = _delaunay_triangles(chart_uv[retained])
        raw_delaunay_triangles += len(local_triangles)
        for local_triangle in local_triangles:
            triangle = tuple(int(value) for value in retained[local_triangle])
            edge_indices: list[int] = []
            for edge_index, left in enumerate(triangle):
                right = triangle[(edge_index + 1) % 3]
                low = min(left, right)
                high = max(left, right)
                value = edge_by_key.get(low * count + high)
                if value is None:
                    break
                edge_indices.append(value)
            if len(edge_indices) != 3:
                rejected_unsupported_edge += 1
                continue
            if not any(selected_edge[value] for value in edge_indices):
                rejected_without_selected_edge += 1
                continue
            oriented, area, residual, maximum_edge = _triangle_geometry(
                triangle, center_xyz, signed_normal_xyz
            )
            if (
                area < minimum_area_voxels_squared
                or residual > maximum_normal_residual_degrees
                or maximum_edge > maximum_edge_voxels
            ):
                rejected_geometry += 1
                continue
            triangles.append(oriented)
            areas.append(area)
            residuals.append(residual)
    return (
        np.asarray(triangles, dtype=np.int32).reshape((-1, 3)),
        np.asarray(areas, dtype=np.float32),
        np.asarray(residuals, dtype=np.float32),
        {
            "components": len(component_values),
            "rawDelaunayTriangles": raw_delaunay_triangles,
            "retainedTriangles": len(triangles),
            "rejectedNearDuplicateChartNodes": rejected_duplicate_uv_nodes,
            "rejectedUnsupportedEdges": rejected_unsupported_edge,
            "rejectedWithoutSelectedEdge": rejected_without_selected_edge,
            "rejectedTriangleGeometry": rejected_geometry,
        },
    )


def reconstruct_needle_surface_strips(
    center_xyz: np.ndarray,
    fiber_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    edge_score: np.ndarray,
    selected_edge: np.ndarray,
    topology_component_id: np.ndarray,
    topology_component_size: np.ndarray,
    *,
    geometric_candidate_edge: np.ndarray | None = None,
    complete_pair_graph: Mapping[str, np.ndarray] | None = None,
    complete_pair_summary: Mapping[str, Any] | None = None,
    candidate_spacing_voxels: float,
    depth_kernel_voxels: float,
    neighbor_radius_voxels: float,
    settings: BlockNeedleSurfaceSettings | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Build manifold triangle strips from one block-global ply graph."""

    resolved = settings or BlockNeedleSurfaceSettings()
    center = np.asarray(center_xyz, dtype=np.float64)
    fiber = np.asarray(fiber_xyz, dtype=np.float64)
    normal = np.asarray(normal_xyz, dtype=np.float64)
    edge_first = np.asarray(first, dtype=np.int32)
    edge_second = np.asarray(second, dtype=np.int32)
    score = np.asarray(edge_score, dtype=np.float32)
    selected = np.asarray(selected_edge, dtype=bool)
    geometric_candidate = (
        selected
        if geometric_candidate_edge is None
        else np.asarray(geometric_candidate_edge, dtype=bool)
    )
    if geometric_candidate.shape != selected.shape:
        raise ValueError("geometric candidate and selected edges must have equal shape")
    trace_result = select_ordered_fiber_traces(
        center,
        fiber,
        normal,
        edge_first,
        edge_second,
        score,
        selected,
        np.asarray(topology_component_size, dtype=np.int32),
        depth_kernel_voxels=depth_kernel_voxels,
        settings=resolved,
    )
    traces: list[np.ndarray] = trace_result["traces"]
    trace_id = trace_result["traceId"]
    trace_position = trace_result["tracePosition"]
    trace_size = trace_result["traceSize"]
    fiber_sign = trace_result["fiberSign"]
    normal_sign = trace_result["normalSign"]
    coordinates = trace_result["coordinates"]
    longitudinal_edge = trace_result["longitudinalEdge"]
    signed_fiber = fiber * fiber_sign[:, None]
    signed_normal = normal * normal_sign[:, None]
    cross_axis = np.cross(signed_normal, signed_fiber)
    cross_axis /= np.maximum(np.linalg.norm(cross_axis, axis=1, keepdims=True), 1.0e-8)
    displacement = center[edge_second] - center[edge_first]
    side_first = (
        np.einsum("ij,ij->i", displacement, cross_axis[edge_first]) > 0.0
    ).astype(np.uint8)
    side_second = (
        np.einsum("ij,ij->i", -displacement, cross_axis[edge_second]) > 0.0
    ).astype(np.uint8)
    longitudinal_floor = math.cos(
        math.radians(resolved.maximum_longitudinal_deviation_degrees)
    )
    minimum_trace = resolved.minimum_trace_needles
    same_topology_component = (
        topology_component_id[edge_first] == topology_component_id[edge_second]
    )
    cross_candidate = (
        geometric_candidate
        & same_topology_component
        & (trace_id[edge_first] != trace_id[edge_second])
        & (trace_size[edge_first] >= minimum_trace)
        & (trace_size[edge_second] >= minimum_trace)
        & (coordinates["longitudinalFraction"] < longitudinal_floor)
    )
    pair_groups: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge_index in np.flatnonzero(cross_candidate):
        left_trace = int(trace_id[edge_first[edge_index]])
        right_trace = int(trace_id[edge_second[edge_index]])
        pair_groups[(min(left_trace, right_trace), max(left_trace, right_trace))].append(
            int(edge_index)
        )

    strip_candidates: list[dict[str, Any]] = []
    minimum_span = candidate_spacing_voxels * resolved.minimum_crosslink_span_candidate_steps
    rejected_short_packet = 0
    rejected_side = 0
    rejected_span = 0
    for (first_trace, second_trace), edge_values in pair_groups.items():
        edge_indices = np.asarray(edge_values, dtype=np.int32)
        canonical_first = np.where(
            trace_id[edge_first[edge_indices]] == first_trace,
            edge_first[edge_indices],
            edge_second[edge_indices],
        )
        canonical_second = np.where(
            trace_id[edge_first[edge_indices]] == first_trace,
            edge_second[edge_indices],
            edge_first[edge_indices],
        )
        local_first_position = trace_position[canonical_first]
        local_second_position = trace_position[canonical_second]
        anchors, reverse_second = _longest_monotone_crosslinks(
            edge_indices,
            local_first_position,
            local_second_position,
            score,
        )
        if len(anchors) < resolved.minimum_crosslinks_per_strip:
            rejected_short_packet += 1
            continue
        anchor_first = np.where(
            trace_id[edge_first[anchors]] == first_trace,
            edge_first[anchors],
            edge_second[anchors],
        )
        anchor_second = np.where(
            trace_id[edge_first[anchors]] == first_trace,
            edge_second[anchors],
            edge_first[anchors],
        )
        anchor_first_position = trace_position[anchor_first]
        anchor_second_position = trace_position[anchor_second]
        if reverse_second:
            anchor_second_mapped = len(traces[second_trace]) - 1 - anchor_second_position
        else:
            anchor_second_mapped = anchor_second_position
        order = np.lexsort((anchors, anchor_second_mapped, anchor_first_position))
        anchors = anchors[order]
        anchor_first = anchor_first[order]
        anchor_second = anchor_second[order]
        anchor_first_position = anchor_first_position[order]
        anchor_second_position = anchor_second_position[order]
        anchor_second_mapped = anchor_second_mapped[order]
        first_side = np.where(
            edge_first[anchors] == anchor_first,
            side_first[anchors],
            side_second[anchors],
        )
        second_side = np.where(
            edge_first[anchors] == anchor_second,
            side_first[anchors],
            side_second[anchors],
        )
        side_key = first_side.astype(np.int8) * 2 + second_side.astype(np.int8)
        support_by_side = np.bincount(
            side_key,
            weights=score[anchors],
            minlength=4,
        )
        dominant_side_key = int(np.argmax(support_by_side))
        side_consistency = float(
            support_by_side[dominant_side_key]
            / max(float(np.sum(support_by_side)), 1.0e-12)
        )
        if side_consistency < resolved.minimum_crosslink_side_consistency:
            rejected_side += 1
            continue
        keep = side_key == dominant_side_key
        anchors = anchors[keep]
        anchor_first_position = anchor_first_position[keep]
        anchor_second_position = anchor_second_position[keep]
        anchor_second_mapped = anchor_second_mapped[keep]
        if len(anchors) < resolved.minimum_crosslinks_per_strip:
            rejected_short_packet += 1
            continue
        mesh_first_positions = anchor_first_position
        mesh_second_positions = anchor_second_mapped
        if len(anchors) == 1:
            first_anchor = int(anchor_first_position[0])
            second_anchor = int(anchor_second_mapped[0])
            first_low = max(0, first_anchor - 1)
            first_high = min(len(traces[first_trace]) - 1, first_anchor + 1)
            second_low = max(0, second_anchor - 1)
            second_high = min(len(traces[second_trace]) - 1, second_anchor + 1)
            mesh_first_positions = np.asarray((first_low, first_high), dtype=np.int32)
            mesh_second_positions = np.asarray((second_low, second_high), dtype=np.int32)
        else:
            first_low = int(anchor_first_position[0])
            first_high = int(anchor_first_position[-1])
            second_low = int(anchor_second_mapped[0])
            second_high = int(anchor_second_mapped[-1])
        ordered_second_trace = (
            traces[second_trace][::-1]
            if reverse_second
            else traces[second_trace]
        )
        endpoint_midpoints = 0.5 * (
            center[traces[first_trace][mesh_first_positions[[0, -1]]]]
            + center[ordered_second_trace[mesh_second_positions[[0, -1]]]]
        )
        span = float(np.linalg.norm(endpoint_midpoints[1] - endpoint_midpoints[0]))
        if span < minimum_span:
            rejected_span += 1
            continue
        support = float(np.sum(score[anchors]))
        strip_candidates.append(
            {
                "firstTrace": first_trace,
                "secondTrace": second_trace,
                "firstSide": dominant_side_key // 2,
                "secondSide": dominant_side_key % 2,
                "firstInterval": (first_low, first_high),
                "secondInterval": (second_low, second_high),
                "reverseSecond": reverse_second,
                "anchors": anchors,
                "firstPositions": mesh_first_positions,
                "secondMappedPositions": mesh_second_positions,
                "support": support,
                "spanVoxels": span,
                "sideConsistency": side_consistency,
                "priority": support * math.log1p(span),
            }
        )

    occupied: dict[tuple[int, int], np.ndarray] = {}
    selected_strips: list[dict[str, Any]] = []
    interval_conflicts = 0
    for candidate in sorted(
        strip_candidates,
        key=lambda value: (
            -value["priority"],
            value["firstTrace"],
            value["secondTrace"],
        ),
    ):
        conflicts = False
        for trace_key, side_key, interval in (
            (
                candidate["firstTrace"],
                candidate["firstSide"],
                candidate["firstInterval"],
            ),
            (
                candidate["secondTrace"],
                candidate["secondSide"],
                candidate["secondInterval"],
            ),
        ):
            key = (trace_key, side_key)
            used = occupied.setdefault(key, np.zeros(len(traces[trace_key]), dtype=bool))
            low, high = interval
            if np.any(used[low : high + 1]):
                conflicts = True
                break
        if conflicts:
            interval_conflicts += 1
            continue
        candidate["stripId"] = len(selected_strips)
        selected_strips.append(candidate)
        for trace_key, side_key, interval in (
            (
                candidate["firstTrace"],
                candidate["firstSide"],
                candidate["firstInterval"],
            ),
            (
                candidate["secondTrace"],
                candidate["secondSide"],
                candidate["secondInterval"],
            ),
        ):
            low, high = interval
            occupied[(trace_key, side_key)][low : high + 1] = True

    maximum_edge = (
        neighbor_radius_voxels
        * resolved.maximum_mesh_edge_neighbor_radius_multiplier
    )
    triangles: list[tuple[int, int, int]] = []
    triangle_strip: list[int] = []
    triangle_area: list[float] = []
    triangle_normal_residual: list[float] = []
    retained_strips: list[dict[str, Any]] = []
    edge_face_count: dict[tuple[int, int], int] = defaultdict(int)
    geometry_rejections = 0
    manifold_rejections = 0
    for candidate in selected_strips:
        first_nodes = traces[candidate["firstTrace"]]
        second_nodes = traces[candidate["secondTrace"]]
        if candidate["reverseSecond"]:
            second_nodes = second_nodes[::-1]
        candidate_triangles: list[tuple[int, int, int]] = []
        candidate_areas: list[float] = []
        candidate_residuals: list[float] = []
        first_positions = candidate["firstPositions"]
        second_positions = candidate["secondMappedPositions"]
        valid = True
        for anchor_index in range(len(first_positions) - 1):
            first_low = int(first_positions[anchor_index])
            first_high = int(first_positions[anchor_index + 1])
            second_low = int(second_positions[anchor_index])
            second_high = int(second_positions[anchor_index + 1])
            interval_triangles, interval_areas, interval_residuals = _zipper_interval(
                first_nodes[first_low : first_high + 1],
                second_nodes[second_low : second_high + 1],
                center,
                signed_normal,
                maximum_edge_voxels=maximum_edge,
                maximum_normal_residual_degrees=(
                    resolved.maximum_triangle_normal_residual_degrees
                ),
                minimum_area_voxels_squared=(
                    resolved.minimum_triangle_area_voxels_squared
                ),
            )
            if not interval_triangles:
                valid = False
                break
            candidate_triangles.extend(interval_triangles)
            candidate_areas.extend(interval_areas)
            candidate_residuals.extend(interval_residuals)
        if not valid or not candidate_triangles:
            geometry_rejections += 1
            continue
        local_edge_count: dict[tuple[int, int], int] = defaultdict(int)
        for triangle in candidate_triangles:
            for index, left in enumerate(triangle):
                right = triangle[(index + 1) % 3]
                local_edge_count[(min(left, right), max(left, right))] += 1
        if any(
            edge_face_count[edge] + count > 2
            for edge, count in local_edge_count.items()
        ):
            manifold_rejections += 1
            continue
        output_strip_id = len(retained_strips)
        candidate["outputStripId"] = output_strip_id
        retained_strips.append(candidate)
        triangles.extend(candidate_triangles)
        triangle_strip.extend([output_strip_id] * len(candidate_triangles))
        triangle_area.extend(candidate_areas)
        triangle_normal_residual.extend(candidate_residuals)
        for edge, count in local_edge_count.items():
            edge_face_count[edge] += count

    zipper_triangle_array = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    zipper_strip_array = np.asarray(triangle_strip, dtype=np.int32)
    zipper_area_array = np.asarray(triangle_area, dtype=np.float32)
    zipper_normal_residual_array = np.asarray(
        triangle_normal_residual, dtype=np.float32
    )

    if complete_pair_graph is None:
        chart_first = edge_first
        chart_second = edge_second
        chart_score = score
        chart_geometric = geometric_candidate
        chart_selected = selected
        chart_coordinates = coordinates
    else:
        chart_first = np.asarray(complete_pair_graph["first"], dtype=np.int32)
        chart_second = np.asarray(complete_pair_graph["second"], dtype=np.int32)
        chart_score = np.asarray(complete_pair_graph["score"], dtype=np.float32)
        chart_geometric = np.ones(len(chart_first), dtype=bool)
        chart_selected = np.asarray(
            complete_pair_graph["selectedAnchor"], dtype=bool
        )
        chart_coordinates = _edge_tangent_coordinates(
            center,
            fiber,
            normal,
            fiber_sign,
            normal_sign,
            chart_first,
            chart_second,
        )
    chart_uv, chart_edge_residual, chart_summary = integrate_intrinsic_surface_charts(
        center,
        chart_first,
        chart_second,
        chart_score,
        chart_geometric,
        np.asarray(topology_component_id, dtype=np.int32),
        np.asarray(topology_component_size, dtype=np.int32),
        chart_coordinates["alongSignedVoxels"],
        chart_coordinates["crossSignedVoxels"],
        minimum_component_needles=resolved.minimum_topology_component_needles,
        solver_relative_tolerance=resolved.chart_solver_relative_tolerance,
        solver_maximum_iterations=resolved.chart_solver_maximum_iterations,
    )
    (
        chart_triangles,
        chart_triangle_area,
        chart_triangle_normal_residual,
        chart_mesh_summary,
    ) = triangulate_intrinsic_surface_charts(
        center,
        signed_normal,
        chart_uv,
        chart_first,
        chart_second,
        chart_score,
        chart_geometric,
        chart_selected,
        np.asarray(topology_component_id, dtype=np.int32),
        np.asarray(topology_component_size, dtype=np.int32),
        minimum_component_needles=resolved.minimum_topology_component_needles,
        minimum_chart_separation_voxels=(
            candidate_spacing_voxels
            * resolved.minimum_chart_separation_candidate_fraction
        ),
        maximum_edge_voxels=maximum_edge,
        maximum_normal_residual_degrees=(
            resolved.maximum_triangle_normal_residual_degrees
        ),
        minimum_area_voxels_squared=resolved.minimum_triangle_area_voxels_squared,
    )
    triangles = [tuple(int(value) for value in triangle) for triangle in chart_triangles]
    triangle_strip = [-2] * len(triangles)
    triangle_area = [float(value) for value in chart_triangle_area]
    triangle_normal_residual = [
        float(value) for value in chart_triangle_normal_residual
    ]
    edge_face_count = defaultdict(int)
    for triangle in triangles:
        for edge_index, left in enumerate(triangle):
            right = triangle[(edge_index + 1) % 3]
            edge_face_count[(min(left, right), max(left, right))] += 1

    triangle_array = np.asarray(triangles, dtype=np.int32).reshape((-1, 3))
    strip_array = np.asarray(triangle_strip, dtype=np.int32)
    area_array = np.asarray(triangle_area, dtype=np.float32)
    normal_residual_array = np.asarray(triangle_normal_residual, dtype=np.float32)
    referenced = np.zeros(len(center), dtype=bool)
    if len(triangle_array):
        referenced[np.unique(triangle_array)] = True
    triangle_union = _DisjointSet(len(triangle_array))
    triangle_by_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for triangle_index, triangle in enumerate(triangle_array):
        for edge_index, left in enumerate(triangle):
            right = int(triangle[(edge_index + 1) % 3])
            triangle_by_edge[(min(int(left), right), max(int(left), right))].append(
                triangle_index
            )
    for incident in triangle_by_edge.values():
        if len(incident) == 2:
            triangle_union.union(incident[0], incident[1])
    triangle_roots = triangle_union.roots()
    triangle_component_minimum = np.full(
        len(triangle_array), len(triangle_array), dtype=np.int32
    )
    np.minimum.at(
        triangle_component_minimum,
        triangle_roots,
        np.arange(len(triangle_array), dtype=np.int32),
    )
    triangle_component = triangle_component_minimum[triangle_roots]
    component_values, component_triangle_count = np.unique(
        triangle_component, return_counts=True
    )
    mesh_component = np.full(len(center), -1, dtype=np.int32)
    component_sizes = np.zeros(len(component_values), dtype=np.int32)
    component_nodes: list[np.ndarray] = []
    component_membership_count = np.zeros(len(center), dtype=np.uint16)
    for component_index, value in enumerate(component_values):
        component_triangle = triangle_component == value
        nodes = np.unique(triangle_array[component_triangle])
        component_nodes.append(nodes)
        component_sizes[component_index] = len(nodes)
        component_membership_count[nodes] += 1
        unassigned = mesh_component[nodes] < 0
        mesh_component[nodes[unassigned]] = int(value)
    ranking = np.lexsort((component_values, -component_triangle_count))
    top_components: list[dict[str, Any]] = []
    for rank, component_index in enumerate(ranking[:64], start=1):
        component = int(component_values[component_index])
        component_triangle = triangle_component == component
        nodes = component_nodes[component_index]
        points = center[nodes]
        component_edges: dict[tuple[int, int], int] = defaultdict(int)
        for triangle in triangle_array[component_triangle]:
            for edge_index, left in enumerate(triangle):
                right = int(triangle[(edge_index + 1) % 3])
                component_edges[(min(int(left), right), max(int(left), right))] += 1
        top_components.append(
            {
                "rank": rank,
                "componentId": component,
                "topologyPlyComponentId": int(topology_component_id[nodes[0]]),
                "needles": int(component_sizes[component_index]),
                "triangles": int(component_triangle_count[component_index]),
                "surfaceAreaVoxelsSquared": round(
                    float(np.sum(area_array[component_triangle])), 6
                ),
                "worldStartXYZ": [round(float(value), 6) for value in np.min(points, axis=0)],
                "worldStopXYZ": [round(float(value), 6) for value in np.max(points, axis=0)],
                "extentVoxelsXYZ": [round(float(value), 6) for value in np.ptp(points, axis=0)],
                "boundaryEdges": sum(value == 1 for value in component_edges.values()),
                "nonmanifoldEdges": sum(value > 2 for value in component_edges.values()),
                "triangleNormalResidualDegrees": _percentile_record(
                    normal_residual_array[component_triangle]
                ),
            }
        )

    trace_ids = np.arange(len(traces), dtype=np.int32)
    trace_offsets = np.zeros(len(traces) + 1, dtype=np.int64)
    if traces:
        trace_offsets[1:] = np.cumsum([len(value) for value in traces])
        trace_needles = np.concatenate(traces).astype(np.int32, copy=False)
    else:
        trace_needles = np.empty(0, dtype=np.int32)
    strip_trace_first = np.asarray(
        [value["firstTrace"] for value in retained_strips], dtype=np.int32
    )
    strip_trace_second = np.asarray(
        [value["secondTrace"] for value in retained_strips], dtype=np.int32
    )
    strip_side_first = np.asarray(
        [value["firstSide"] for value in retained_strips], dtype=np.uint8
    )
    strip_side_second = np.asarray(
        [value["secondSide"] for value in retained_strips], dtype=np.uint8
    )
    strip_support = np.asarray(
        [value["support"] for value in retained_strips], dtype=np.float32
    )
    strip_span = np.asarray(
        [value["spanVoxels"] for value in retained_strips], dtype=np.float32
    )
    strip_anchor_offset = np.zeros(len(retained_strips) + 1, dtype=np.int64)
    if retained_strips:
        strip_anchor_offset[1:] = np.cumsum(
            [len(value["anchors"]) for value in retained_strips]
        )
        strip_anchor_edge = np.concatenate(
            [value["anchors"] for value in retained_strips]
        ).astype(np.int32, copy=False)
    else:
        strip_anchor_edge = np.empty(0, dtype=np.int32)
    output_arrays = {
        "fiberSign": fiber_sign,
        "normalSign": normal_sign,
        "edgeLongitudinalFraction": coordinates["longitudinalFraction"],
        "edgeLongitudinalSelected": longitudinal_edge.astype(np.uint8),
        "fiberTraceId": trace_id,
        "fiberTracePosition": trace_position,
        "fiberTraceSize": trace_size,
        "traceId": trace_ids,
        "traceNeedleOffset": trace_offsets,
        "traceNeedle": trace_needles,
        "stripTraceFirst": strip_trace_first,
        "stripTraceSecond": strip_trace_second,
        "stripSideFirst": strip_side_first,
        "stripSideSecond": strip_side_second,
        "stripSupport": strip_support,
        "stripSpanVoxels": strip_span,
        "stripAnchorOffset": strip_anchor_offset,
        "stripAnchorEdge": strip_anchor_edge,
        "triangleNeedle": triangle_array,
        "triangleStrip": strip_array,
        "triangleAreaVoxelsSquared": area_array,
        "triangleNormalResidualDegrees": normal_residual_array,
        "triangleSurfaceComponentId": triangle_component,
        "surfaceComponentId": mesh_component,
        "needleSurfaceComponentMultiplicity": component_membership_count,
        "surfaceChartUV": chart_uv,
        "chartPairFirstNeedle": chart_first,
        "chartPairSecondNeedle": chart_second,
        "chartPairScore": chart_score,
        "chartPairSelectedAnchor": chart_selected.astype(np.uint8),
        "chartEdgeIntegrationResidualVoxels": chart_edge_residual,
        "zipperTriangleNeedle": zipper_triangle_array,
        "zipperTriangleStrip": zipper_strip_array,
        "zipperTriangleAreaVoxelsSquared": zipper_area_array,
        "zipperTriangleNormalResidualDegrees": zipper_normal_residual_array,
    }
    summary = {
        "method": {
            "scope": "one block-global manifold surface-strip reconstruction",
            "fiberGauge": (
                "unsigned fiber and normal axes receive deterministic transport "
                "signs for ordering only; sign is never treated as evidence"
            ),
            "traces": (
                "maximum-support edges consume one predecessor/successor slot per "
                "needle and cycles are forbidden, producing ordered open chains"
            ),
            "strips": (
                "neighboring traces require order-preserving crosslinks, "
                "a consistent side, and nonoverlapping trace-side intervals"
            ),
            "mesh": (
                "calibrated local fiber/cross-fiber chords are integrated into an "
                "intrinsic chart per frozen carrier; Delaunay faces retain exact "
                "needle centers and every edge must independently pass the physical "
                "continuation gates"
            ),
            "cells": "not used by reconstruction",
        },
        "derivedScales": {
            "candidateSpacingVoxels": candidate_spacing_voxels,
            "depthKernelVoxels": depth_kernel_voxels,
            "neighborRadiusVoxels": neighbor_radius_voxels,
            "minimumCrosslinkSpanVoxels": minimum_span,
            "maximumMeshEdgeVoxels": maximum_edge,
        },
        "traces": trace_result["summary"],
        "crosslinkSelection": {
            "candidateEdges": int(np.count_nonzero(cross_candidate)),
            "candidateTracePairs": len(pair_groups),
            "geometricStripCandidates": len(strip_candidates),
            "rejectedShortPackets": rejected_short_packet,
            "rejectedInconsistentSides": rejected_side,
            "rejectedShortSpan": rejected_span,
            "rejectedIntervalConflicts": interval_conflicts,
            "selectedBeforeMeshing": len(selected_strips),
            "rejectedByTriangleGeometry": geometry_rejections,
            "rejectedByManifoldConstraint": manifold_rejections,
            "retainedStrips": len(retained_strips),
            "zipperTriangles": len(zipper_triangle_array),
            "zipperTrianglesAreEvidenceOnly": True,
        },
        "intrinsicCharts": chart_summary,
        "completeComponentPairGraph": dict(complete_pair_summary or {}),
        "chartTriangulation": chart_mesh_summary,
        "mesh": {
            "triangles": len(triangle_array),
            "referencedNeedles": int(np.count_nonzero(referenced)),
            "components": len(component_values),
            "largestComponentNeedles": (
                int(np.max(component_sizes)) if len(component_sizes) else 0
            ),
            "largestComponentTriangles": (
                int(np.max(component_triangle_count))
                if len(component_triangle_count)
                else 0
            ),
            "surfaceAreaVoxelsSquared": round(float(np.sum(area_array)), 6),
            "boundaryEdges": sum(value == 1 for value in edge_face_count.values()),
            "nonmanifoldEdges": sum(value > 2 for value in edge_face_count.values()),
            "maximumIncidentTrianglesPerEdge": max(edge_face_count.values(), default=0),
            "needlesInMultipleEdgeConnectedComponents": int(
                np.count_nonzero(component_membership_count > 1)
            ),
            "triangleAreaVoxelsSquared": _percentile_record(area_array),
            "triangleNormalResidualDegrees": _percentile_record(
                normal_residual_array
            ),
        },
        "topComponents": top_components,
    }
    return summary, output_arrays


def write_needle_surface_obj(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    surface_arrays: Mapping[str, np.ndarray],
    path: str | Path,
) -> Path:
    triangles = np.asarray(surface_arrays["triangleNeedle"], dtype=np.int32)
    triangle_component = np.asarray(
        surface_arrays["triangleSurfaceComponentId"], dtype=np.int32
    )
    referenced = np.unique(triangles) if len(triangles) else np.empty(0, dtype=np.int32)
    local = {int(value): index + 1 for index, value in enumerate(referenced)}
    lines = [
        "# Pareidolia block-global Acus needle surfaces",
        "# coordinate unit: source voxel",
    ]
    for needle in referenced:
        point = center_xyz[needle]
        lines.append(f"v {point[0]:.9g} {point[1]:.9g} {point[2]:.9g}")
    for needle in referenced:
        value = normal_xyz[needle]
        lines.append(f"vn {value[0]:.9g} {value[1]:.9g} {value[2]:.9g}")
    current_component: int | None = None
    order = np.argsort(triangle_component) if len(triangles) else np.empty(0, dtype=int)
    for triangle_index in order:
        triangle = triangles[triangle_index]
        value = int(triangle_component[triangle_index])
        if value != current_component:
            lines.append(f"g surface_{value}")
            current_component = value
        indices = [local[int(node)] for node in triangle]
        lines.append("f " + " ".join(f"{index}//{index}" for index in indices))
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_text("\n".join(lines) + "\n")
    temporary.replace(output)
    return output


def _draw_line(
    image: np.ndarray,
    first: tuple[float, float],
    second: tuple[float, float],
    color: tuple[int, int, int],
) -> None:
    x0, y0 = first
    x1, y1 = second
    count = max(int(math.ceil(max(abs(x1 - x0), abs(y1 - y0)))), 1) + 1
    x = np.rint(np.linspace(x0, x1, count)).astype(np.int32)
    y = np.rint(np.linspace(y0, y1, count)).astype(np.int32)
    valid = (x >= 0) & (x < image.shape[1]) & (y >= 0) & (y < image.shape[0])
    image[y[valid], x[valid]] = color


def _fill_triangle(
    image: np.ndarray,
    points: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    low = np.maximum(np.floor(np.min(points, axis=0)).astype(int), 0)
    high = np.minimum(
        np.ceil(np.max(points, axis=0)).astype(int),
        np.asarray((image.shape[1] - 1, image.shape[0] - 1)),
    )
    if np.any(high < low):
        return
    x, y = np.meshgrid(
        np.arange(low[0], high[0] + 1),
        np.arange(low[1], high[1] + 1),
    )
    first, second, third = points
    denominator = (
        (second[1] - third[1]) * (first[0] - third[0])
        + (third[0] - second[0]) * (first[1] - third[1])
    )
    if abs(float(denominator)) <= 1.0e-8:
        return
    a = (
        (second[1] - third[1]) * (x - third[0])
        + (third[0] - second[0]) * (y - third[1])
    ) / denominator
    b = (
        (third[1] - first[1]) * (x - third[0])
        + (first[0] - third[0]) * (y - third[1])
    ) / denominator
    mask = (a >= -1.0e-6) & (b >= -1.0e-6) & (a + b <= 1.0 + 1.0e-6)
    target = image[low[1] : high[1] + 1, low[0] : high[0] + 1]
    target[mask] = np.rint(
        0.48 * target[mask].astype(np.float64)
        + 0.52 * np.asarray(color, dtype=np.float64)
    ).astype(np.uint8)


def write_needle_surface_projection_png(
    center_xyz: np.ndarray,
    surface_arrays: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    maximum_components: int,
    fit_selected: bool = False,
    panel_size: int = 640,
) -> Path:
    triangles = np.asarray(surface_arrays["triangleNeedle"], dtype=np.int32)
    triangle_component = np.asarray(
        surface_arrays["triangleSurfaceComponentId"], dtype=np.int32
    )
    center = np.asarray(center_xyz, dtype=np.float64)
    if not len(triangles):
        selected_components = np.empty(0, dtype=np.int32)
    else:
        values, sizes = np.unique(triangle_component, return_counts=True)
        selected_components = values[np.lexsort((values, -sizes))[:maximum_components]]
    selected_set = set(int(value) for value in selected_components)
    visible = np.asarray(
        [int(value) in selected_set for value in triangle_component],
        dtype=bool,
    ) if len(triangles) else np.empty(0, dtype=bool)
    low = np.asarray(world_start_xyz, dtype=np.float64).copy()
    high = np.asarray(world_stop_xyz, dtype=np.float64).copy()
    if fit_selected and np.any(visible):
        nodes = np.unique(triangles[visible])
        low = np.min(center[nodes], axis=0)
        high = np.max(center[nodes], axis=0)
        padding = np.maximum(0.05 * (high - low), 2.0)
        low -= padding
        high += padding
    image = np.full((panel_size, 3 * panel_size, 3), (8, 12, 18), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    projections = ((0, 1), (0, 2), (1, 2))
    colors = {
        int(value): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.08 + 0.61803398875 * rank) % 1.0, 0.64, 0.96
            )
        )
        for rank, value in enumerate(selected_components)
    }

    def project(point: np.ndarray, panel: int, axes: tuple[int, int]) -> np.ndarray:
        width = np.maximum(high[list(axes)] - low[list(axes)], 1.0e-8)
        normalized = (point[list(axes)] - low[list(axes)]) / width
        return np.asarray(
            (
                panel * panel_size + margin + normalized[0] * (panel_size - 2 * margin),
                panel_size - margin - normalized[1] * (panel_size - 2 * margin),
            )
        )

    for panel, axes in enumerate(projections):
        panel_offset = panel * panel_size
        for triangle_index in np.flatnonzero(visible):
            triangle = triangles[triangle_index]
            value = int(triangle_component[triangle_index])
            points = np.asarray([project(center[node], panel, axes) for node in triangle])
            _fill_triangle(image, points, colors[value])
        for triangle_index in np.flatnonzero(visible):
            triangle = triangles[triangle_index]
            value = int(triangle_component[triangle_index])
            points = [project(center[node], panel, axes) for node in triangle]
            edge_color = tuple(max(32, int(channel * 0.72)) for channel in colors[value])
            for index, first_point in enumerate(points):
                _draw_line(
                    image,
                    tuple(first_point),
                    tuple(points[(index + 1) % 3]),
                    edge_color,
                )
        image[margin, panel_offset + margin : panel_offset + panel_size - margin] = (52, 61, 72)
        image[
            panel_size - margin,
            panel_offset + margin : panel_offset + panel_size - margin,
        ] = (52, 61, 72)
        image[margin : panel_size - margin, panel_offset + margin] = (52, 61, 72)
        image[margin : panel_size - margin, panel_offset + panel_size - margin] = (52, 61, 72)
    image[:, panel_size - 1 : panel_size + 1] = (82, 92, 106)
    image[:, 2 * panel_size - 1 : 2 * panel_size + 1] = (82, 92, 106)
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(image))
    temporary.replace(output)
    return output


def run_block_needle_surfaces(
    topology_root: str | Path,
    output_root: str | Path,
    *,
    settings: BlockNeedleSurfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize ordered manifold strips from a block needle topology."""

    started = time.monotonic()
    resolved = settings or BlockNeedleSurfaceSettings()
    topology_path, topology_manifest, topology_arrays = _load_topology_artifact(
        topology_root
    )
    field_path = Path(topology_manifest["source"]["fieldManifest"])
    field_manifest_path, field_manifest, field_arrays = _load_field_artifact(field_path)
    if (
        field_manifest["identity"]["identitySha256"]
        != topology_manifest["source"]["fieldIdentitySha256"]
    ):
        raise ValueError("needle topology references another block needle field")
    raw_settings, _raw_sources = _raw_settings_from_field(field_manifest)
    loaded = time.monotonic()
    identity: dict[str, Any] = {
        "schema": BLOCK_NEEDLE_SURFACE_SCHEMA,
        "version": BLOCK_NEEDLE_SURFACE_VERSION,
        "topology": {
            "path": str(topology_path),
            "manifestSha256": sha256_file(topology_path),
            "dataSha256": topology_manifest["data"]["sha256"],
            "identitySha256": topology_manifest["identity"]["identitySha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    manifest_path = output / f"{BLOCK_NEEDLE_SURFACE_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity["identitySha256"]:
            raise ValueError("needle-surface output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    complete_pair_graph, complete_pair_summary = build_complete_component_pair_graph(
        field_arrays["centerXYZ"],
        field_arrays["directionXYZ"],
        field_arrays["normalXYZ"],
        topology_arrays["stackDensity"],
        topology_arrays["stackOrientationMoment"],
        topology_arrays["plyComponentId"],
        topology_arrays["plyComponentSize"],
        topology_arrays["edgeFirstNeedle"],
        topology_arrays["edgeSecondNeedle"],
        topology_arrays["edgeGrowthEligible"].astype(bool),
        topology_arrays["edgeSelected"].astype(bool),
        minimum_component_needles=resolved.minimum_topology_component_needles,
        neighbor_radius_voxels=float(
            field_manifest["settings"]["neighbor_radius_voxels"]
        ),
        tangent_compatibility_sigma_voxels=float(
            field_manifest["settings"]["tangent_compatibility_sigma_voxels"]
        ),
        minimum_curvature_radius_voxels=float(
            topology_manifest["derivedPhysicalSettings"][
                "minimumCurvatureRadiusVoxels"
            ]
        ),
        minimum_curved_affinity=float(
            topology_manifest["settings"]["growth_minimum_curved_affinity"]
        ),
        growth_caps=topology_manifest["growthCaps"],
        fingerprint_chunk_edges=int(
            topology_manifest["settings"]["edge_score_chunk"]
        ),
    )
    summary, output_arrays = reconstruct_needle_surface_strips(
        field_arrays["centerXYZ"],
        field_arrays["directionXYZ"],
        field_arrays["normalXYZ"],
        topology_arrays["edgeFirstNeedle"],
        topology_arrays["edgeSecondNeedle"],
        topology_arrays["edgeScore"],
        topology_arrays["edgeSelected"].astype(bool),
        topology_arrays["plyComponentId"],
        topology_arrays["plyComponentSize"],
        geometric_candidate_edge=topology_arrays["edgeGrowthEligible"].astype(bool),
        complete_pair_graph=complete_pair_graph,
        complete_pair_summary=complete_pair_summary,
        candidate_spacing_voxels=float(raw_settings.candidate_spacing_voxels),
        depth_kernel_voxels=float(raw_settings.depth_kernel_voxels),
        neighbor_radius_voxels=float(field_manifest["settings"]["neighbor_radius_voxels"]),
        settings=resolved,
    )
    reconstructed = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{BLOCK_NEEDLE_SURFACE_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    temporary.replace(data_path)
    signed_normal = field_arrays["normalXYZ"] * output_arrays["normalSign"][:, None]
    obj_path = write_needle_surface_obj(
        field_arrays["centerXYZ"], signed_normal, output_arrays, output / "surfaces.obj"
    )
    bounds = field_manifest["source"]["worldBounds"]
    world_start = np.asarray(bounds["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(bounds["stopXYZExclusive"], dtype=np.float64)
    overview_path = write_needle_surface_projection_png(
        field_arrays["centerXYZ"],
        output_arrays,
        output / "top-12-surfaces.png",
        world_start_xyz=world_start,
        world_stop_xyz=world_stop,
        maximum_components=resolved.maximum_preview_components,
    )
    largest_path = write_needle_surface_projection_png(
        field_arrays["centerXYZ"],
        output_arrays,
        output / "largest-surface.png",
        world_start_xyz=world_start,
        world_stop_xyz=world_stop,
        maximum_components=1,
        fit_selected=True,
    )
    payload = {
        "schema": BLOCK_NEEDLE_SURFACE_SCHEMA,
        "version": BLOCK_NEEDLE_SURFACE_VERSION,
        "state": "complete",
        "identity": identity,
        "settings": resolved.record(),
        "source": {
            "topologyManifest": str(topology_path),
            "topologyIdentitySha256": topology_manifest["identity"]["identitySha256"],
            "fieldManifest": str(field_manifest_path),
            "fieldIdentitySha256": field_manifest["identity"]["identitySha256"],
            "worldBounds": bounds,
        },
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "meshArtifact": {
            "path": obj_path.name,
            "bytes": obj_path.stat().st_size,
            "sha256": sha256_file(obj_path),
        },
        "previews": {
            "top12": overview_path.name,
            "largest": largest_path.name,
            "projectionOrder": ["XY", "XZ", "YZ"],
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "reconstruction": round(reconstructed - loaded, 6),
            "writing": round(time.monotonic() - reconstructed, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
