from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from itertools import combinations
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .needle_flatten import _load_surface_artifact
from .needle_surface import _load_topology_artifact
from .needle_topology import _load_field_artifact, _raw_settings_from_field


BLOCK_NEEDLE_BUNDLE_SCHEMA = "pareidolia.block-acus-needle-bundles"
BLOCK_NEEDLE_BUNDLE_VERSION = 1
BLOCK_NEEDLE_BUNDLE_STEM = "block-needle-bundles-v1"


@dataclass(frozen=True, slots=True)
class BlockNeedleBundleSettings:
    """Dimensionless controls for cross-ply association evidence."""

    maximum_edge_normal_sigma: float = 2.0
    maximum_edge_orthogonal_fiber_error_sigma: float = 2.0
    maximum_packet_normal_sigma: float = 1.0
    maximum_packet_orthogonal_fiber_error_sigma: float = 1.0
    minimum_packet_edges: int = 4
    minimum_packet_endpoints_per_side: int = 3
    minimum_packet_span_needle_fraction: float = 0.5
    minimum_height_side_consistency: float = 0.75
    maximum_separation_mad_depth_kernel: float = 2.0
    maximum_shadow_bridge_gap_needle_length: float = 1.0
    maximum_reported_packets: int = 64
    maximum_reported_bridges: int = 64

    def __post_init__(self) -> None:
        positive = (
            self.maximum_edge_normal_sigma,
            self.maximum_edge_orthogonal_fiber_error_sigma,
            self.maximum_packet_normal_sigma,
            self.maximum_packet_orthogonal_fiber_error_sigma,
            self.minimum_packet_span_needle_fraction,
            self.maximum_separation_mad_depth_kernel,
            self.maximum_shadow_bridge_gap_needle_length,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("needle-bundle scales must be finite and positive")
        integer_positive = (
            self.minimum_packet_edges,
            self.minimum_packet_endpoints_per_side,
            self.maximum_reported_packets,
            self.maximum_reported_bridges,
        )
        if any(value < 1 for value in integer_positive):
            raise ValueError("needle-bundle integer settings must be positive")
        if not 0.5 <= self.minimum_height_side_consistency <= 1.0:
            raise ValueError("height-side consistency must lie in [0.5, 1]")

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
        while value != root:
            following = int(self.parent[value])
            self.parent[value] = root
            value = following
        return root

    def union(self, first: int, second: int) -> None:
        left = self.find(first)
        right = self.find(second)
        if left == right:
            return
        if int(self.size[left]) < int(self.size[right]):
            left, right = right, left
        self.parent[right] = left
        self.size[left] += self.size[right]


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


def _surface_membership(
    triangle_needle: np.ndarray,
    triangle_component: np.ndarray,
    topology_component: np.ndarray,
    needle_count: int,
) -> dict[str, np.ndarray]:
    component_values, triangle_count = np.unique(
        triangle_component, return_counts=True
    )
    pair = np.column_stack(
        (
            triangle_needle.reshape(-1),
            np.repeat(triangle_component, 3),
        )
    )
    pair = np.unique(pair, axis=0)
    order = np.lexsort((pair[:, 1], pair[:, 0]))
    pair = pair[order]
    needle_component = pair[:, 1].astype(np.int32, copy=False)
    needle_count_by_index = np.bincount(
        pair[:, 0].astype(np.int32), minlength=needle_count
    )
    needle_offset = np.concatenate(
        ((0,), np.cumsum(needle_count_by_index, dtype=np.int64))
    )
    pair_component_index = np.searchsorted(component_values, pair[:, 1])
    component_needle_count = np.bincount(
        pair_component_index, minlength=len(component_values)
    ).astype(np.int32)
    pair_topology = topology_component[pair[:, 0].astype(np.int32)]
    component_topology_low = np.full(
        len(component_values), np.iinfo(np.int32).max, dtype=np.int32
    )
    component_topology_high = np.full(
        len(component_values), np.iinfo(np.int32).min, dtype=np.int32
    )
    np.minimum.at(component_topology_low, pair_component_index, pair_topology)
    np.maximum.at(component_topology_high, pair_component_index, pair_topology)
    if np.any(component_topology_low != component_topology_high):
        raise ValueError("one surface component crosses topology ply carriers")
    component_order = np.lexsort((pair[:, 0], pair_component_index))
    component_needle = pair[component_order, 0].astype(np.int32, copy=False)
    component_needle_offset = np.concatenate(
        ((0,), np.cumsum(component_needle_count, dtype=np.int64))
    )
    return {
        "componentValue": component_values.astype(np.int32, copy=False),
        "componentTriangleCount": triangle_count.astype(np.int32, copy=False),
        "componentNeedleCount": component_needle_count,
        "componentTopologyPlyId": component_topology_low,
        "componentNeedleOffset": component_needle_offset,
        "componentNeedle": component_needle,
        "needleComponentOffset": needle_offset,
        "needleComponentValue": needle_component,
    }


def _maximum_pair_distance(points_xyz: np.ndarray) -> float:
    if len(points_xyz) < 2:
        return 0.0
    difference = points_xyz[:, None, :] - points_xyz[None, :, :]
    return float(np.sqrt(np.max(np.einsum("ijk,ijk->ij", difference, difference))))


def _minimum_pair_distance(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return math.inf
    difference = first[:, None, :] - second[None, :, :]
    return float(np.sqrt(np.min(np.einsum("ijk,ijk->ij", difference, difference))))


def associate_orthogonal_surface_packets(
    center_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    normal_sign: np.ndarray,
    needle_quality: np.ndarray,
    topology_component_id: np.ndarray,
    triangle_needle: np.ndarray,
    triangle_surface_component: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_normal_angle_degrees: np.ndarray,
    edge_fiber_residual_degrees: np.ndarray,
    edge_stack_fingerprint_mismatch: np.ndarray,
    *,
    minimum_layer_separation_voxels: float,
    maximum_sheet_thickness_voxels: float,
    orthogonal_ply_sigma_degrees: float,
    needle_length_voxels: float,
    depth_kernel_voxels: float,
    settings: BlockNeedleBundleSettings,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Aggregate local cross-fiber edges into gauge-invariant ply packets."""

    center = np.asarray(center_xyz, dtype=np.float64)
    normal = np.asarray(normal_xyz, dtype=np.float64)
    signed_normal = normal * np.asarray(normal_sign, dtype=np.float64)[:, None]
    topology = np.asarray(topology_component_id, dtype=np.int32)
    triangle = np.asarray(triangle_needle, dtype=np.int32)
    triangle_component = np.asarray(triangle_surface_component, dtype=np.int32)
    first = np.asarray(edge_first, dtype=np.int32)
    second = np.asarray(edge_second, dtype=np.int32)
    normal_angle = np.asarray(edge_normal_angle_degrees, dtype=np.float64)
    fiber_angle = np.asarray(edge_fiber_residual_degrees, dtype=np.float64)
    fingerprint = np.asarray(edge_stack_fingerprint_mismatch, dtype=np.float64)
    quality = np.asarray(needle_quality, dtype=np.float64)
    membership = _surface_membership(
        triangle,
        triangle_component,
        topology,
        len(center),
    )
    component_to_topology = dict(
        zip(
            membership["componentValue"].tolist(),
            membership["componentTopologyPlyId"].tolist(),
        )
    )
    offset = membership["needleComponentOffset"]
    needle_component = membership["needleComponentValue"]
    has_surface = offset[1:] > offset[:-1]
    displacement = center[second] - center[first]
    first_height = np.einsum("ij,ij->i", displacement, signed_normal[first])
    second_height = np.einsum("ij,ij->i", -displacement, signed_normal[second])
    separation = 0.5 * (np.abs(first_height) + np.abs(second_height))
    edge_normal_cap = (
        settings.maximum_edge_normal_sigma * orthogonal_ply_sigma_degrees
    )
    edge_fiber_error_cap = (
        settings.maximum_edge_orthogonal_fiber_error_sigma
        * orthogonal_ply_sigma_degrees
    )
    candidate_edge = (
        has_surface[first]
        & has_surface[second]
        & (topology[first] != topology[second])
        & (normal_angle <= edge_normal_cap)
        & (np.abs(90.0 - fiber_angle) <= edge_fiber_error_cap)
        & (separation >= minimum_layer_separation_voxels)
        & (separation <= maximum_sheet_thickness_voxels)
    )
    packet_edge: dict[tuple[int, int], list[int]] = defaultdict(list)
    for edge in np.flatnonzero(candidate_edge):
        left_components = needle_component[offset[first[edge]] : offset[first[edge] + 1]]
        right_components = needle_component[
            offset[second[edge]] : offset[second[edge] + 1]
        ]
        for left in left_components:
            for right in right_components:
                if left == right:
                    continue
                key = (min(int(left), int(right)), max(int(left), int(right)))
                packet_edge[key].append(int(edge))

    minimum_span = (
        settings.minimum_packet_span_needle_fraction * needle_length_voxels
    )
    maximum_separation_mad = (
        settings.maximum_separation_mad_depth_kernel * depth_kernel_voxels
    )
    packet_first: list[int] = []
    packet_second: list[int] = []
    packet_edges: list[np.ndarray] = []
    packet_edge_count: list[int] = []
    packet_first_endpoints: list[int] = []
    packet_second_endpoints: list[int] = []
    packet_span: list[float] = []
    packet_side_consistency: list[float] = []
    packet_separation: list[float] = []
    packet_separation_mad: list[float] = []
    packet_normal_median: list[float] = []
    packet_normal_p90: list[float] = []
    packet_fiber_error_median: list[float] = []
    packet_fiber_error_p90: list[float] = []
    packet_fingerprint_median: list[float] = []
    packet_quality_median: list[float] = []
    packet_score: list[float] = []
    packet_evidence_mass: list[float] = []
    packet_selected: list[bool] = []
    for (left_component, right_component), edge_values in sorted(
        packet_edge.items()
    ):
        edges = np.unique(np.asarray(edge_values, dtype=np.int32))
        left_topology = component_to_topology[left_component]
        right_topology = component_to_topology[right_component]
        left_is_first = topology[first[edges]] == left_topology
        if np.any(
            np.where(left_is_first, topology[second[edges]], topology[first[edges]])
            != right_topology
        ):
            raise ValueError("cross-ply packet contains an unrelated topology edge")
        left_node = np.where(left_is_first, first[edges], second[edges])
        right_node = np.where(left_is_first, second[edges], first[edges])
        packet_displacement = center[right_node] - center[left_node]
        signed_height = np.einsum(
            "ij,ij->i", packet_displacement, signed_normal[left_node]
        )
        side_consistency = max(
            float(np.mean(signed_height >= 0.0)),
            float(np.mean(signed_height <= 0.0)),
        )
        edge_separation = 0.5 * (
            np.abs(signed_height)
            + np.abs(
                np.einsum(
                    "ij,ij->i", -packet_displacement, signed_normal[right_node]
                )
            )
        )
        median_separation = float(np.median(edge_separation))
        separation_mad = float(
            np.median(np.abs(edge_separation - median_separation))
        )
        midpoints = 0.5 * (center[left_node] + center[right_node])
        span = _maximum_pair_distance(midpoints)
        normal_median = float(np.median(normal_angle[edges]))
        normal_p90 = float(np.percentile(normal_angle[edges], 90))
        fiber_error = np.abs(90.0 - fiber_angle[edges])
        fiber_error_median = float(np.median(fiber_error))
        fiber_error_p90 = float(np.percentile(fiber_error, 90))
        first_endpoint_count = len(np.unique(left_node))
        second_endpoint_count = len(np.unique(right_node))
        evidence_factors = np.asarray(
            (
                math.exp(
                    -0.5 * (normal_median / orthogonal_ply_sigma_degrees) ** 2
                ),
                math.exp(
                    -0.5
                    * (fiber_error_median / orthogonal_ply_sigma_degrees) ** 2
                ),
                math.exp(-0.5 * (separation_mad / depth_kernel_voxels) ** 2),
                max(2.0 * side_consistency - 1.0, 0.0),
                min(span / minimum_span, 1.0),
                min(
                    min(first_endpoint_count, second_endpoint_count)
                    / settings.minimum_packet_endpoints_per_side,
                    1.0,
                ),
            ),
            dtype=np.float64,
        )
        score = float(
            np.exp(np.mean(np.log(np.maximum(evidence_factors, 1.0e-12))))
        )
        evidence_mass = float(
            score
            * math.sqrt(len(edges) / settings.minimum_packet_edges)
            * math.sqrt(
                min(first_endpoint_count, second_endpoint_count)
                / settings.minimum_packet_endpoints_per_side
            )
            * (span / minimum_span)
        )
        selected = bool(
            len(edges) >= settings.minimum_packet_edges
            and min(first_endpoint_count, second_endpoint_count)
            >= settings.minimum_packet_endpoints_per_side
            and span >= minimum_span
            and normal_median
            <= settings.maximum_packet_normal_sigma * orthogonal_ply_sigma_degrees
            and fiber_error_median
            <= settings.maximum_packet_orthogonal_fiber_error_sigma
            * orthogonal_ply_sigma_degrees
            and side_consistency >= settings.minimum_height_side_consistency
            and separation_mad <= maximum_separation_mad
        )
        packet_first.append(left_component)
        packet_second.append(right_component)
        packet_edges.append(edges)
        packet_edge_count.append(len(edges))
        packet_first_endpoints.append(first_endpoint_count)
        packet_second_endpoints.append(second_endpoint_count)
        packet_span.append(span)
        packet_side_consistency.append(side_consistency)
        packet_separation.append(median_separation)
        packet_separation_mad.append(separation_mad)
        packet_normal_median.append(normal_median)
        packet_normal_p90.append(normal_p90)
        packet_fiber_error_median.append(fiber_error_median)
        packet_fiber_error_p90.append(fiber_error_p90)
        packet_fingerprint_median.append(float(np.median(fingerprint[edges])))
        pair_quality = np.sqrt(quality[left_node] * quality[right_node])
        packet_quality_median.append(float(np.median(pair_quality)))
        packet_score.append(score)
        packet_evidence_mass.append(evidence_mass)
        packet_selected.append(selected)

    packet_offset = np.zeros(len(packet_edges) + 1, dtype=np.int64)
    if packet_edges:
        packet_offset[1:] = np.cumsum(
            np.asarray([len(value) for value in packet_edges]), dtype=np.int64
        )
        packet_topology_edge = np.concatenate(packet_edges).astype(
            np.int32, copy=False
        )
    else:
        packet_topology_edge = np.empty(0, dtype=np.int32)
    output = {
        **membership,
        "candidateTopologyEdge": np.flatnonzero(candidate_edge).astype(np.int32),
        "packetFirstSurfaceComponent": np.asarray(packet_first, dtype=np.int32),
        "packetSecondSurfaceComponent": np.asarray(packet_second, dtype=np.int32),
        "packetTopologyEdgeOffset": packet_offset,
        "packetTopologyEdge": packet_topology_edge,
        "packetEdgeCount": np.asarray(packet_edge_count, dtype=np.int32),
        "packetFirstEndpointCount": np.asarray(
            packet_first_endpoints, dtype=np.int32
        ),
        "packetSecondEndpointCount": np.asarray(
            packet_second_endpoints, dtype=np.int32
        ),
        "packetMidpointSpanVoxels": np.asarray(packet_span, dtype=np.float32),
        "packetHeightSideConsistency": np.asarray(
            packet_side_consistency, dtype=np.float32
        ),
        "packetSeparationVoxels": np.asarray(packet_separation, dtype=np.float32),
        "packetSeparationMadVoxels": np.asarray(
            packet_separation_mad, dtype=np.float32
        ),
        "packetNormalMedianDegrees": np.asarray(
            packet_normal_median, dtype=np.float32
        ),
        "packetNormalP90Degrees": np.asarray(packet_normal_p90, dtype=np.float32),
        "packetOrthogonalFiberErrorMedianDegrees": np.asarray(
            packet_fiber_error_median, dtype=np.float32
        ),
        "packetOrthogonalFiberErrorP90Degrees": np.asarray(
            packet_fiber_error_p90, dtype=np.float32
        ),
        "packetStackFingerprintMismatchMedian": np.asarray(
            packet_fingerprint_median, dtype=np.float32
        ),
        "packetNeedleQualityMedian": np.asarray(
            packet_quality_median, dtype=np.float32
        ),
        "packetAssociationScore": np.asarray(packet_score, dtype=np.float32),
        "packetEvidenceMass": np.asarray(
            packet_evidence_mass, dtype=np.float32
        ),
        "packetSelected": np.asarray(packet_selected, dtype=np.uint8),
    }
    selected_mask = output["packetSelected"].astype(bool)
    summary = {
        "surfaceComponents": len(membership["componentValue"]),
        "surfaceNeedles": int(np.count_nonzero(has_surface)),
        "crossTopologyEdgesConsidered": int(
            np.count_nonzero(has_surface[first] & has_surface[second])
        ),
        "candidateOrthogonalEdges": int(np.count_nonzero(candidate_edge)),
        "candidatePackets": len(packet_first),
        "selectedPackets": int(np.count_nonzero(selected_mask)),
        "selectedPacketEdges": _percentile_record(
            output["packetEdgeCount"][selected_mask]
        ),
        "selectedPacketSpanVoxels": _percentile_record(
            output["packetMidpointSpanVoxels"][selected_mask]
        ),
        "selectedPacketSeparationVoxels": _percentile_record(
            output["packetSeparationVoxels"][selected_mask]
        ),
        "selectedPacketSeparationMadVoxels": _percentile_record(
            output["packetSeparationMadVoxels"][selected_mask]
        ),
        "selectedPacketNormalDegrees": _percentile_record(
            output["packetNormalMedianDegrees"][selected_mask]
        ),
        "selectedPacketOrthogonalFiberErrorDegrees": _percentile_record(
            output["packetOrthogonalFiberErrorMedianDegrees"][selected_mask]
        ),
        "selectedPacketScore": _percentile_record(
            output["packetAssociationScore"][selected_mask]
        ),
        "selectedPacketEvidenceMass": _percentile_record(
            output["packetEvidenceMass"][selected_mask]
        ),
    }
    return summary, output


def build_shadow_bridge_evidence(
    center_xyz: np.ndarray,
    surface_chart_uv: np.ndarray,
    triangle_needle: np.ndarray,
    triangle_surface_component: np.ndarray,
    component_value: np.ndarray,
    component_topology_ply_id: np.ndarray,
    component_needle_offset: np.ndarray,
    component_needle: np.ndarray,
    packet_first_component: np.ndarray,
    packet_second_component: np.ndarray,
    packet_score: np.ndarray,
    packet_selected: np.ndarray,
    *,
    maximum_chart_gap_voxels: float,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Find same-ply islands co-supported by one crossed-fiber carrier."""

    center = np.asarray(center_xyz, dtype=np.float64)
    chart = np.asarray(surface_chart_uv, dtype=np.float64)
    triangle = np.asarray(triangle_needle, dtype=np.int32)
    triangle_component = np.asarray(triangle_surface_component, dtype=np.int32)
    component = np.asarray(component_value, dtype=np.int32)
    component_topology = np.asarray(component_topology_ply_id, dtype=np.int32)
    component_node_offset = np.asarray(component_needle_offset, dtype=np.int64)
    component_node = np.asarray(component_needle, dtype=np.int32)
    first = np.asarray(packet_first_component, dtype=np.int32)
    second = np.asarray(packet_second_component, dtype=np.int32)
    score = np.asarray(packet_score, dtype=np.float64)
    selected = np.asarray(packet_selected, dtype=bool)
    component_index = {int(value): index for index, value in enumerate(component)}
    topology_by_component = dict(
        zip(component.tolist(), component_topology.tolist())
    )
    nodes_by_component = {
        int(value): component_node[
            component_node_offset[index] : component_node_offset[index + 1]
        ]
        for index, value in enumerate(component)
    }
    directed: dict[tuple[int, int], list[tuple[int, int, int]]] = defaultdict(list)
    for packet in np.flatnonzero(selected):
        left = int(first[packet])
        right = int(second[packet])
        left_topology = topology_by_component[left]
        right_topology = topology_by_component[right]
        directed[(left_topology, right_topology)].append(
            (left, right, int(packet))
        )
        directed[(right_topology, left_topology)].append(
            (right, left, int(packet))
        )
    support: dict[
        tuple[int, int], list[tuple[int, int, int, int, int]]
    ] = defaultdict(list)
    for (_home_topology, partner_topology), records in sorted(directed.items()):
        for first_record, second_record in combinations(sorted(set(records)), 2):
            left, left_partner, left_packet = first_record
            right, right_partner, right_packet = second_record
            if left == right:
                continue
            key = (min(left, right), max(left, right))
            if key[0] != left:
                left_partner, right_partner = right_partner, left_partner
                left_packet, right_packet = right_packet, left_packet
            support[key].append(
                (
                    partner_topology,
                    left_partner,
                    right_partner,
                    left_packet,
                    right_packet,
                )
            )

    bridge_first: list[int] = []
    bridge_second: list[int] = []
    bridge_chart_gap: list[float] = []
    bridge_world_gap: list[float] = []
    bridge_score: list[float] = []
    bridge_selected: list[bool] = []
    bridge_support: list[list[tuple[int, int, int, int, int]]] = []
    for (left, right), records in sorted(support.items()):
        left_nodes = nodes_by_component[left]
        right_nodes = nodes_by_component[right]
        chart_gap = _minimum_pair_distance(chart[left_nodes], chart[right_nodes])
        world_gap = _minimum_pair_distance(center[left_nodes], center[right_nodes])
        support_score = max(
            min(float(score[left_packet]), float(score[right_packet]))
            for (
                _partner_topology,
                _left_partner,
                _right_partner,
                left_packet,
                right_packet,
            ) in records
        )
        bridge_first.append(left)
        bridge_second.append(right)
        bridge_chart_gap.append(chart_gap)
        bridge_world_gap.append(world_gap)
        bridge_score.append(support_score)
        bridge_selected.append(chart_gap <= maximum_chart_gap_voxels)
        bridge_support.append(sorted(set(records)))
    support_offset = np.zeros(len(bridge_support) + 1, dtype=np.int64)
    if bridge_support:
        support_offset[1:] = np.cumsum(
            np.asarray([len(value) for value in bridge_support]), dtype=np.int64
        )
        flattened_support = [value for values in bridge_support for value in values]
        support_partner_topology = np.asarray(
            [value[0] for value in flattened_support], dtype=np.int32
        )
        support_left_partner = np.asarray(
            [value[1] for value in flattened_support], dtype=np.int32
        )
        support_right_partner = np.asarray(
            [value[2] for value in flattened_support], dtype=np.int32
        )
        support_first_packet = np.asarray(
            [value[3] for value in flattened_support], dtype=np.int32
        )
        support_second_packet = np.asarray(
            [value[4] for value in flattened_support], dtype=np.int32
        )
    else:
        support_partner_topology = np.empty(0, dtype=np.int32)
        support_left_partner = np.empty(0, dtype=np.int32)
        support_right_partner = np.empty(0, dtype=np.int32)
        support_first_packet = np.empty(0, dtype=np.int32)
        support_second_packet = np.empty(0, dtype=np.int32)

    disjoint = _DisjointSet(len(component))
    selected_bridge = np.asarray(bridge_selected, dtype=bool)
    for bridge in np.flatnonzero(selected_bridge):
        disjoint.union(
            component_index[bridge_first[bridge]],
            component_index[bridge_second[bridge]],
        )
    root = np.asarray([disjoint.find(index) for index in range(len(component))])
    group_value_by_root: dict[int, int] = {}
    for index, value in enumerate(component):
        group_root = int(root[index])
        group_value_by_root[group_root] = min(
            group_value_by_root.get(group_root, int(value)), int(value)
        )
    support_group = np.asarray(
        [group_value_by_root[int(value)] for value in root], dtype=np.int32
    )
    group_records: list[tuple[int, int, int, int]] = []
    for group in np.unique(support_group):
        members = component[support_group == group]
        nodes = np.unique(
            np.concatenate([nodes_by_component[int(value)] for value in members])
        )
        triangles = int(
            np.count_nonzero(np.isin(triangle_component, members))
        )
        group_records.append((len(nodes), triangles, len(members), int(group)))
    group_records.sort(reverse=True)
    multi_groups = [value for value in group_records if value[2] > 1]
    output = {
        "shadowBridgeFirstSurfaceComponent": np.asarray(
            bridge_first, dtype=np.int32
        ),
        "shadowBridgeSecondSurfaceComponent": np.asarray(
            bridge_second, dtype=np.int32
        ),
        "shadowBridgeChartGapVoxels": np.asarray(
            bridge_chart_gap, dtype=np.float32
        ),
        "shadowBridgeWorldGapVoxels": np.asarray(
            bridge_world_gap, dtype=np.float32
        ),
        "shadowBridgeScore": np.asarray(bridge_score, dtype=np.float32),
        "shadowBridgeSelected": np.asarray(bridge_selected, dtype=np.uint8),
        "shadowBridgeSupportOffset": support_offset,
        "shadowBridgeSupportPartnerTopologyPlyId": support_partner_topology,
        "shadowBridgeSupportFirstPartnerSurfaceComponent": support_left_partner,
        "shadowBridgeSupportSecondPartnerSurfaceComponent": support_right_partner,
        "shadowBridgeSupportFirstPacket": support_first_packet,
        "shadowBridgeSupportSecondPacket": support_second_packet,
        "surfaceComponentSupportGroupId": support_group,
    }
    summary = {
        "candidateBridges": len(bridge_first),
        "selectedBridges": int(np.count_nonzero(selected_bridge)),
        "selectedBridgesWithOneIntactPartnerIsland": int(
            sum(
                bool(
                    np.any(
                        support_left_partner[support_offset[index] : support_offset[index + 1]]
                        == support_right_partner[
                            support_offset[index] : support_offset[index + 1]
                        ]
                    )
                )
                for index in np.flatnonzero(selected_bridge)
            )
        ),
        "selectedBridgeChartGapVoxels": _percentile_record(
            output["shadowBridgeChartGapVoxels"][selected_bridge]
        ),
        "selectedBridgeWorldGapVoxels": _percentile_record(
            output["shadowBridgeWorldGapVoxels"][selected_bridge]
        ),
        "selectedBridgeScore": _percentile_record(
            output["shadowBridgeScore"][selected_bridge]
        ),
        "supportGroups": len(group_records),
        "multiComponentSupportGroups": len(multi_groups),
        "largestSupportGroupNeedles": multi_groups[0][0] if multi_groups else 0,
        "largestSupportGroupTriangles": multi_groups[0][1] if multi_groups else 0,
        "largestSupportGroupComponents": multi_groups[0][2] if multi_groups else 0,
    }
    return summary, output


def _carrier_atlas_summary(
    triangle_needle: np.ndarray,
    triangle_surface_component: np.ndarray,
    topology_component_id: np.ndarray,
) -> tuple[dict[str, Any], list[dict[str, int]]]:
    triangle_topology = topology_component_id[triangle_needle]
    if np.any(triangle_topology != triangle_topology[:, :1]):
        raise ValueError("one surface triangle crosses topology ply carriers")
    records: list[dict[str, int]] = []
    for topology in np.unique(triangle_topology[:, 0]):
        selected = triangle_topology[:, 0] == topology
        records.append(
            {
                "topologyPlyComponentId": int(topology),
                "needles": len(np.unique(triangle_needle[selected])),
                "triangles": int(np.count_nonzero(selected)),
                "surfaceComponents": len(
                    np.unique(triangle_surface_component[selected])
                ),
            }
        )
    records.sort(
        key=lambda value: (
            -value["needles"],
            -value["triangles"],
            value["topologyPlyComponentId"],
        )
    )
    summary = {
        "carriersWithTriangles": len(records),
        "carrierNeedles": _percentile_record(
            np.asarray([value["needles"] for value in records])
        ),
        "carrierTriangles": _percentile_record(
            np.asarray([value["triangles"] for value in records])
        ),
        "carrierSurfaceComponents": _percentile_record(
            np.asarray([value["surfaceComponents"] for value in records])
        ),
        "largestCarrierNeedles": records[0]["needles"] if records else 0,
        "largestCarrierTriangles": records[0]["triangles"] if records else 0,
        "largestCarrierSurfaceComponents": (
            records[0]["surfaceComponents"] if records else 0
        ),
    }
    return summary, records


def _top_packet_records(
    arrays: Mapping[str, np.ndarray], maximum: int
) -> list[dict[str, Any]]:
    selected = np.flatnonzero(np.asarray(arrays["packetSelected"], dtype=bool))
    order = sorted(
        selected.tolist(),
        key=lambda index: (
            -float(arrays["packetEvidenceMass"][index]),
            -float(arrays["packetAssociationScore"][index]),
            -int(arrays["packetEdgeCount"][index]),
            int(arrays["packetFirstSurfaceComponent"][index]),
            int(arrays["packetSecondSurfaceComponent"][index]),
        ),
    )[:maximum]
    return [
        {
            "rank": rank,
            "packetId": index,
            "firstSurfaceComponent": int(
                arrays["packetFirstSurfaceComponent"][index]
            ),
            "secondSurfaceComponent": int(
                arrays["packetSecondSurfaceComponent"][index]
            ),
            "edges": int(arrays["packetEdgeCount"][index]),
            "firstEndpoints": int(arrays["packetFirstEndpointCount"][index]),
            "secondEndpoints": int(arrays["packetSecondEndpointCount"][index]),
            "midpointSpanVoxels": round(
                float(arrays["packetMidpointSpanVoxels"][index]), 6
            ),
            "separationVoxels": round(
                float(arrays["packetSeparationVoxels"][index]), 6
            ),
            "separationMadVoxels": round(
                float(arrays["packetSeparationMadVoxels"][index]), 6
            ),
            "normalMedianDegrees": round(
                float(arrays["packetNormalMedianDegrees"][index]), 6
            ),
            "orthogonalFiberErrorMedianDegrees": round(
                float(
                    arrays["packetOrthogonalFiberErrorMedianDegrees"][index]
                ),
                6,
            ),
            "heightSideConsistency": round(
                float(arrays["packetHeightSideConsistency"][index]), 6
            ),
            "associationScore": round(
                float(arrays["packetAssociationScore"][index]), 6
            ),
            "evidenceMass": round(
                float(arrays["packetEvidenceMass"][index]), 6
            ),
        }
        for rank, index in enumerate(order, start=1)
    ]


def _top_bridge_records(
    arrays: Mapping[str, np.ndarray], maximum: int
) -> list[dict[str, Any]]:
    selected = np.flatnonzero(
        np.asarray(arrays["shadowBridgeSelected"], dtype=bool)
    )
    order = sorted(
        selected.tolist(),
        key=lambda index: (
            -float(arrays["shadowBridgeScore"][index]),
            float(arrays["shadowBridgeChartGapVoxels"][index]),
            int(arrays["shadowBridgeFirstSurfaceComponent"][index]),
            int(arrays["shadowBridgeSecondSurfaceComponent"][index]),
        ),
    )[:maximum]
    support_offset = arrays["shadowBridgeSupportOffset"]
    partner_topology = arrays["shadowBridgeSupportPartnerTopologyPlyId"]
    left_partner = arrays["shadowBridgeSupportFirstPartnerSurfaceComponent"]
    right_partner = arrays["shadowBridgeSupportSecondPartnerSurfaceComponent"]
    records: list[dict[str, Any]] = []
    for rank, index in enumerate(order, start=1):
        low = int(support_offset[index])
        high = int(support_offset[index + 1])
        records.append(
            {
                "rank": rank,
                "bridgeId": index,
                "firstSurfaceComponent": int(
                    arrays["shadowBridgeFirstSurfaceComponent"][index]
                ),
                "secondSurfaceComponent": int(
                    arrays["shadowBridgeSecondSurfaceComponent"][index]
                ),
                "supportingOrthogonalCarriers": len(
                    np.unique(partner_topology[low:high])
                ),
                "supportingPacketPairs": high - low,
                "hasOneIntactPartnerIsland": bool(
                    np.any(left_partner[low:high] == right_partner[low:high])
                ),
                "chartGapVoxels": round(
                    float(arrays["shadowBridgeChartGapVoxels"][index]), 6
                ),
                "worldGapVoxels": round(
                    float(arrays["shadowBridgeWorldGapVoxels"][index]), 6
                ),
                "score": round(float(arrays["shadowBridgeScore"][index]), 6),
            }
        )
    return records


def run_block_needle_bundles(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: BlockNeedleBundleSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Associate crossed-fiber ply patches without changing their meshes."""

    started = time.monotonic()
    resolved = settings or BlockNeedleBundleSettings()
    surface_path, surface_manifest, surface_arrays = _load_surface_artifact(
        surface_root
    )
    identity: dict[str, Any] = {
        "schema": BLOCK_NEEDLE_BUNDLE_SCHEMA,
        "version": BLOCK_NEEDLE_BUNDLE_VERSION,
        "surface": {
            "path": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
            "identitySha256": surface_manifest["identity"]["identitySha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    manifest_path = output / f"{BLOCK_NEEDLE_BUNDLE_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("needle-bundle output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    topology_path, topology_manifest, topology_arrays = _load_topology_artifact(
        surface_manifest["source"]["topologyManifest"]
    )
    if (
        topology_manifest["identity"]["identitySha256"]
        != surface_manifest["source"]["topologyIdentitySha256"]
    ):
        raise ValueError("needle surfaces reference another block needle topology")
    topology_data_path = topology_path.parent / str(topology_manifest["data"]["path"])
    extra_topology_names = (
        "edgeNormalAngleDegrees",
        "edgeFiberResidualDegrees",
        "edgeStackFingerprintMismatch",
    )
    with np.load(topology_data_path) as values:
        missing = set(extra_topology_names) - set(values.files)
        if missing:
            raise ValueError(f"needle topology is missing arrays: {sorted(missing)}")
        topology_arrays.update(
            {name: np.asarray(values[name]) for name in extra_topology_names}
        )
    field_path, field_manifest, field_arrays = _load_field_artifact(
        surface_manifest["source"]["fieldManifest"]
    )
    if (
        field_manifest["identity"]["identitySha256"]
        != surface_manifest["source"]["fieldIdentitySha256"]
    ):
        raise ValueError("needle surfaces reference another block needle field")
    raw_settings, _raw_sources = _raw_settings_from_field(field_manifest)
    voxel_size_microns = float(field_manifest["source"]["voxelSizeMicrons"])
    minimum_layer_separation = (
        raw_settings.minimum_layer_spacing_microns / voxel_size_microns
    )
    maximum_sheet_thickness = (
        max(raw_settings.plausible_sheet_thickness_microns) / voxel_size_microns
    )
    maximum_shadow_gap = (
        resolved.maximum_shadow_bridge_gap_needle_length
        * float(raw_settings.needle_length_voxels)
    )
    loaded = time.monotonic()
    derived = {
        "voxelSizeMicrons": voxel_size_microns,
        "minimumLayerSeparationVoxels": minimum_layer_separation,
        "maximumSheetThicknessVoxels": maximum_sheet_thickness,
        "orthogonalPlySigmaDegrees": float(
            raw_settings.orthogonal_ply_std_degrees
        ),
        "minimumPacketSpanVoxels": (
            resolved.minimum_packet_span_needle_fraction
            * raw_settings.needle_length_voxels
        ),
        "maximumPacketSeparationMadVoxels": (
            resolved.maximum_separation_mad_depth_kernel
            * raw_settings.depth_kernel_voxels
        ),
        "maximumShadowBridgeChartGapVoxels": maximum_shadow_gap,
    }
    packet_summary, output_arrays = associate_orthogonal_surface_packets(
        field_arrays["centerXYZ"],
        field_arrays["normalXYZ"],
        surface_arrays["normalSign"],
        field_arrays["score"] * field_arrays["supportScore"],
        topology_arrays["plyComponentId"],
        surface_arrays["triangleNeedle"],
        surface_arrays["triangleSurfaceComponentId"],
        topology_arrays["edgeFirstNeedle"],
        topology_arrays["edgeSecondNeedle"],
        topology_arrays["edgeNormalAngleDegrees"],
        topology_arrays["edgeFiberResidualDegrees"],
        topology_arrays["edgeStackFingerprintMismatch"],
        minimum_layer_separation_voxels=minimum_layer_separation,
        maximum_sheet_thickness_voxels=maximum_sheet_thickness,
        orthogonal_ply_sigma_degrees=float(
            raw_settings.orthogonal_ply_std_degrees
        ),
        needle_length_voxels=float(raw_settings.needle_length_voxels),
        depth_kernel_voxels=float(raw_settings.depth_kernel_voxels),
        settings=resolved,
    )
    bridge_summary, bridge_arrays = build_shadow_bridge_evidence(
        field_arrays["centerXYZ"],
        surface_arrays["surfaceChartUV"],
        surface_arrays["triangleNeedle"],
        surface_arrays["triangleSurfaceComponentId"],
        output_arrays["componentValue"],
        output_arrays["componentTopologyPlyId"],
        output_arrays["componentNeedleOffset"],
        output_arrays["componentNeedle"],
        output_arrays["packetFirstSurfaceComponent"],
        output_arrays["packetSecondSurfaceComponent"],
        output_arrays["packetAssociationScore"],
        output_arrays["packetSelected"],
        maximum_chart_gap_voxels=maximum_shadow_gap,
    )
    output_arrays.update(bridge_arrays)
    carrier_summary, carrier_records = _carrier_atlas_summary(
        surface_arrays["triangleNeedle"],
        surface_arrays["triangleSurfaceComponentId"],
        topology_arrays["plyComponentId"],
    )
    associated = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{BLOCK_NEEDLE_BUNDLE_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    temporary.replace(data_path)
    top_packets = _top_packet_records(
        output_arrays, resolved.maximum_reported_packets
    )
    top_bridges = _top_bridge_records(
        output_arrays, resolved.maximum_reported_bridges
    )
    payload = {
        "schema": BLOCK_NEEDLE_BUNDLE_SCHEMA,
        "version": BLOCK_NEEDLE_BUNDLE_VERSION,
        "state": "complete",
        "identity": identity,
        "settings": resolved.record(),
        "derivedPhysicalSettings": derived,
        "source": {
            "surfaceManifest": str(surface_path),
            "surfaceIdentitySha256": surface_manifest["identity"]["identitySha256"],
            "topologyManifest": str(topology_path),
            "topologyIdentitySha256": topology_manifest["identity"][
                "identitySha256"
            ],
            "fieldManifest": str(field_path),
            "fieldIdentitySha256": field_manifest["identity"]["identitySha256"],
            "worldBounds": field_manifest["source"]["worldBounds"],
        },
        "method": {
            "scope": "block-global evidence-only association of reconstructed ply patches",
            "packets": (
                "nearby components require aligned normals, orthogonal unsigned fibers, "
                "physical layer separation, independent endpoints, spatial span, and "
                "a consistent normal side"
            ),
            "shadowBridges": (
                "two disconnected islands of one frozen ply carrier may be supported "
                "by one intact orthogonal island or by two islands in one frozen "
                "orthogonal carrier; this never adds an edge or triangle"
            ),
            "meshMutation": "none",
            "cells": "not used by association",
        },
        "carrierAtlas": carrier_summary,
        "orthogonalPlyPackets": packet_summary,
        "shadowBridges": bridge_summary,
        "topCarriers": carrier_records[: resolved.maximum_reported_packets],
        "topPackets": top_packets,
        "topBridges": top_bridges,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "association": round(associated - loaded, 6),
            "writing": round(time.monotonic() - associated, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
