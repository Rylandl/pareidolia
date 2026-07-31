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
from .physical_ribbon_bank import PHYSICAL_RIBBON_BANK_SCHEMA


PHYSICAL_RIBBON_CONTINUITY_SCHEMA = "pareidolia.physical-ribbon-continuity"
PHYSICAL_RIBBON_CONTINUITY_VERSION = 1
PHYSICAL_RIBBON_CONTINUITY_STEM = "physical-ribbon-continuity-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonContinuitySettings:
    maximum_bidirectional_ray_rank: int = 3
    neighbor_bucket_radius: int = 1
    maximum_neighbor_distance_voxels: float = 5.0
    maximum_normal_degrees: float = 35.0
    maximum_midpoint_height_residual_voxels: float = 2.5
    maximum_boundary_height_residual_voxels: float = 2.5
    maximum_thickness_change_voxels: float = 5.0
    maximum_boundary_shift_difference_voxels: float = 3.0
    minimum_support_degree: int = 3
    minimum_tangent_rank_ratio: float = 0.04
    minimum_selected_degree: int = 1
    peeling_sweeps: int = 2
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        if self.maximum_bidirectional_ray_rank < 0:
            raise ValueError("maximum ray rank must be nonnegative")
        if self.neighbor_bucket_radius < 1:
            raise ValueError("neighbor bucket radius must be positive")
        if self.minimum_support_degree < 1:
            raise ValueError("minimum support degree must be positive")
        if self.minimum_selected_degree < 0:
            raise ValueError("minimum selected degree must be nonnegative")
        if self.peeling_sweeps < 0:
            raise ValueError("peeling sweeps must be nonnegative")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")
        positive = (
            self.maximum_neighbor_distance_voxels,
            self.maximum_normal_degrees,
            self.maximum_midpoint_height_residual_voxels,
            self.maximum_boundary_height_residual_voxels,
            self.maximum_thickness_change_voxels,
            self.maximum_boundary_shift_difference_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("continuity caps must be finite and positive")
        if not 0.0 < self.maximum_normal_degrees < 90.0:
            raise ValueError("normal cap must lie in (0, 90) degrees")
        if not 0.0 <= self.minimum_tangent_rank_ratio <= 1.0:
            raise ValueError("tangent rank ratio must lie in [0, 1]")

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


def _candidate_bucket_edges(
    key: np.ndarray,
    radius: int,
) -> tuple[np.ndarray, np.ndarray]:
    bucket: dict[tuple[int, int, int], list[int]] = {}
    for index, value in enumerate(key):
        bucket.setdefault(tuple(int(v) for v in value), []).append(index)
    offsets = [
        (x, y, z)
        for x in range(-radius, radius + 1)
        for y in range(-radius, radius + 1)
        for z in range(-radius, radius + 1)
        if (x, y, z) > (0, 0, 0)
        and x * x + y * y + z * z <= 3 * radius * radius
    ]
    first: list[np.ndarray] = []
    second: list[np.ndarray] = []
    for value, members in bucket.items():
        current = np.asarray(members, dtype=np.int32)
        if len(current) > 1:
            row, column = np.triu_indices(len(current), k=1)
            first.append(current[row])
            second.append(current[column])
        for offset in offsets:
            adjacent = bucket.get(
                (value[0] + offset[0], value[1] + offset[1], value[2] + offset[2])
            )
            if adjacent is None:
                continue
            other = np.asarray(adjacent, dtype=np.int32)
            first.append(np.repeat(current, len(other)))
            second.append(np.tile(other, len(current)))
    if not first:
        return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
    return np.concatenate(first), np.concatenate(second)


def build_paired_boundary_continuity(
    ribbon: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    *,
    processing_world_start_xyz: np.ndarray,
    sampling_stride_voxels: int,
    settings: PhysicalRibbonContinuitySettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build and select a label-free paired-boundary continuation graph."""

    source_all = np.asarray(ribbon["sourceInterface"], dtype=np.int32)
    target_all = np.asarray(ribbon["targetInterface"], dtype=np.int32)
    rank_source_all = np.asarray(ribbon["sourceRayRank"], dtype=np.int32)
    rank_target_all = np.asarray(ribbon["targetRayRank"], dtype=np.int32)
    score_all = np.asarray(ribbon["physicalEvidenceScore"], dtype=np.float32)
    mutual_all = np.asarray(ribbon["mutualFirstHit"]) > 0
    frontier_mask = (
        (np.asarray(ribbon["bidirectional"]) > 0)
        & (rank_source_all <= settings.maximum_bidirectional_ray_rank)
        & (rank_target_all >= 0)
        & (rank_target_all <= settings.maximum_bidirectional_ray_rank)
    )
    bank_index = np.flatnonzero(frontier_mask).astype(np.int32)
    source = source_all[bank_index]
    target = target_all[bank_index]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[bank_index]
    sheet_normal = np.asarray(ribbon["normalXYZ"], dtype=np.float32)[bank_index]
    thickness = np.asarray(ribbon["thicknessVoxels"], dtype=np.float32)[bank_index]
    score = score_all[bank_index]
    rank_source = rank_source_all[bank_index]
    rank_target = rank_target_all[bank_index]
    mutual = mutual_all[bank_index]
    interface_position = np.asarray(interfaces["positionXYZ"], dtype=np.float32)
    interface_normal = np.asarray(interfaces["signedNormalXYZ"], dtype=np.float32)

    half = 0.5 * (sampling_stride_voxels - 1)
    midpoint_key = np.rint(
        (
            midpoint
            - np.asarray(processing_world_start_xyz, dtype=np.float32)[None, :]
            - half
        )
        / sampling_stride_voxels
    ).astype(np.int32)
    raw_first, raw_second = _candidate_bucket_edges(
        midpoint_key, settings.neighbor_bucket_radius
    )
    first_source = source[raw_first]
    first_target = target[raw_first]
    second_source = source[raw_second]
    second_target = target[raw_second]
    shared_interface = (
        (first_source == second_source)
        | (first_source == second_target)
        | (first_target == second_source)
        | (first_target == second_target)
    )
    delta = midpoint[raw_second] - midpoint[raw_first]
    distance = np.linalg.norm(delta, axis=1)
    normal_dot = np.einsum(
        "ij,ij->i", sheet_normal[raw_first], sheet_normal[raw_second]
    )
    aligned_second_normal = sheet_normal[raw_second] * np.where(
        normal_dot[:, None] >= 0.0, 1.0, -1.0
    )
    mean_normal = sheet_normal[raw_first] + aligned_second_normal
    mean_normal /= np.maximum(
        np.linalg.norm(mean_normal, axis=1, keepdims=True), 1.0e-6
    )
    midpoint_height = np.abs(np.einsum("ij,ij->i", delta, mean_normal))
    normal_degrees = np.degrees(
        np.arccos(np.clip(np.abs(normal_dot), -1.0, 1.0))
    )

    direct_alignment = (
        np.einsum(
            "ij,ij->i",
            interface_normal[first_source],
            interface_normal[second_source],
        )
        + np.einsum(
            "ij,ij->i",
            interface_normal[first_target],
            interface_normal[second_target],
        )
    )
    swapped_alignment = (
        np.einsum(
            "ij,ij->i",
            interface_normal[first_source],
            interface_normal[second_target],
        )
        + np.einsum(
            "ij,ij->i",
            interface_normal[first_target],
            interface_normal[second_source],
        )
    )
    swapped = swapped_alignment > direct_alignment
    corresponding_second_source = np.where(
        swapped, second_target, second_source
    )
    corresponding_second_target = np.where(
        swapped, second_source, second_target
    )
    source_shift = (
        interface_position[corresponding_second_source]
        - interface_position[first_source]
    )
    target_shift = (
        interface_position[corresponding_second_target]
        - interface_position[first_target]
    )
    source_shift_distance = np.linalg.norm(source_shift, axis=1)
    target_shift_distance = np.linalg.norm(target_shift, axis=1)
    shift_difference = np.abs(source_shift_distance - target_shift_distance)
    source_face_normal = (
        interface_normal[first_source]
        + interface_normal[corresponding_second_source]
    )
    target_face_normal = (
        interface_normal[first_target]
        + interface_normal[corresponding_second_target]
    )
    source_face_normal /= np.maximum(
        np.linalg.norm(source_face_normal, axis=1, keepdims=True), 1.0e-6
    )
    target_face_normal /= np.maximum(
        np.linalg.norm(target_face_normal, axis=1, keepdims=True), 1.0e-6
    )
    boundary_height = np.maximum(
        np.abs(np.einsum("ij,ij->i", source_shift, source_face_normal)),
        np.abs(np.einsum("ij,ij->i", target_shift, target_face_normal)),
    )
    thickness_change = np.abs(thickness[raw_first] - thickness[raw_second])
    compatible = (
        ~shared_interface
        & (distance >= 0.5)
        & (distance <= settings.maximum_neighbor_distance_voxels)
        & (normal_degrees <= settings.maximum_normal_degrees)
        & (
            midpoint_height
            <= settings.maximum_midpoint_height_residual_voxels
        )
        & (
            boundary_height
            <= settings.maximum_boundary_height_residual_voxels
        )
        & (thickness_change <= settings.maximum_thickness_change_voxels)
        & (
            shift_difference
            <= settings.maximum_boundary_shift_difference_voxels
        )
    )
    edge_first = raw_first[compatible]
    edge_second = raw_second[compatible]
    distance = distance[compatible]
    normal_degrees = normal_degrees[compatible]
    midpoint_height = midpoint_height[compatible]
    boundary_height = boundary_height[compatible]
    thickness_change = thickness_change[compatible]
    shift_difference = shift_difference[compatible]
    edge_score = np.sqrt(score[edge_first] * score[edge_second]) * np.exp(
        -0.5
        * (
            (normal_degrees / 22.5) ** 2
            + (midpoint_height / 1.5) ** 2
            + (boundary_height / 1.5) ** 2
            + (thickness_change / 3.0) ** 2
            + (shift_difference / 2.0) ** 2
        )
    )

    node_count = len(bank_index)
    degree = np.bincount(
        np.concatenate((edge_first, edge_second)), minlength=node_count
    ).astype(np.int32)
    direction = (
        midpoint[edge_second] - midpoint[edge_first]
    ) / np.maximum(distance[:, None], 1.0e-6)
    outer = direction[:, :, None] * direction[:, None, :]
    weighted_outer = outer * edge_score[:, None, None]
    covariance = np.zeros((node_count, 3, 3), dtype=np.float32)
    np.add.at(covariance, edge_first, weighted_outer)
    np.add.at(covariance, edge_second, weighted_outer)
    eigenvalue = np.linalg.eigvalsh(covariance)
    tangent_rank_ratio = eigenvalue[:, 1] / np.maximum(eigenvalue[:, 2], 1.0e-6)
    support = np.zeros(node_count, dtype=np.float32)
    np.add.at(support, edge_first, edge_score)
    np.add.at(support, edge_second, edge_score)

    eligible = (
        (degree >= settings.minimum_support_degree)
        & (tangent_rank_ratio >= settings.minimum_tangent_rank_ratio)
    ) | (mutual & (degree > 0))
    objective = (
        score
        + 0.16 * np.log1p(degree)
        + 0.30 * np.clip(tangent_rank_ratio, 0.0, 1.0)
        + 0.25 * mutual
        - 0.035 * (rank_source + rank_target)
    )
    order = np.argsort(-objective)
    used_interface = np.zeros(len(interface_position), dtype=bool)
    selected = np.zeros(node_count, dtype=bool)
    mutual_node = np.flatnonzero(mutual)
    mutual_endpoint = np.concatenate(
        (source[mutual_node], target[mutual_node])
    )
    if len(np.unique(mutual_endpoint)) != len(mutual_endpoint):
        raise RuntimeError("mutual first-hit ribbons are not interface-disjoint")
    selected[mutual_node] = True
    used_interface[source[mutual_node]] = True
    used_interface[target[mutual_node]] = True
    for node in order:
        if selected[node] or not eligible[node]:
            continue
        if used_interface[source[node]] or used_interface[target[node]]:
            continue
        selected[node] = True
        used_interface[source[node]] = True
        used_interface[target[node]] = True

    for _ in range(settings.peeling_sweeps):
        selected_edge = selected[edge_first] & selected[edge_second]
        selected_degree = np.bincount(
            np.concatenate(
                (edge_first[selected_edge], edge_second[selected_edge])
            ),
            minlength=node_count,
        )
        remove = (
            selected
            & ~mutual
            & (selected_degree < settings.minimum_selected_degree)
        )
        if not np.any(remove):
            break
        selected[remove] = False

    selected_edge = selected[edge_first] & selected[edge_second]
    kept_first = edge_first[selected_edge]
    kept_second = edge_second[selected_edge]
    parent = np.arange(node_count, dtype=np.int32)

    def find(value: int) -> int:
        while parent[value] != value:
            parent[value] = parent[parent[value]]
            value = int(parent[value])
        return value

    for first, second in zip(kept_first, kept_second):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            if first_root > second_root:
                first_root, second_root = second_root, first_root
            parent[second_root] = first_root
    selected_index = np.flatnonzero(selected)
    roots = np.asarray([find(int(value)) for value in selected_index])
    unique_root, inverse, component_size = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    component_order = np.argsort(-component_size)
    component_rank = np.empty(len(component_order), dtype=np.int32)
    component_rank[component_order] = np.arange(len(component_order))
    component = np.full(node_count, -1, dtype=np.int32)
    component[selected_index] = component_rank[inverse]
    final_selected_interfaces = np.unique(
        np.concatenate((source[selected], target[selected]))
    )

    arrays = {
        "frontierRibbonCandidate": bank_index,
        "frontierMidpointKeyXYZ": midpoint_key,
        "continuitySupportDegree": degree,
        "continuitySupportScore": support,
        "tangentRankRatio": tangent_rank_ratio.astype(np.float32),
        "selectionObjective": objective.astype(np.float32),
        "selected": selected.astype(np.uint8),
        "component": component,
        "edgeFirstFrontierIndex": edge_first,
        "edgeSecondFrontierIndex": edge_second,
        "edgeScore": edge_score.astype(np.float32),
        "edgeSelected": selected_edge.astype(np.uint8),
        "edgeNormalDegrees": normal_degrees.astype(np.float32),
        "edgeMidpointHeightResidualVoxels": midpoint_height.astype(np.float32),
        "edgeBoundaryHeightResidualVoxels": boundary_height.astype(np.float32),
        "edgeThicknessChangeVoxels": thickness_change.astype(np.float32),
        "edgeBoundaryShiftDifferenceVoxels": shift_difference.astype(np.float32),
    }
    stats = {
        "ribbonBankCandidateCount": int(len(source_all)),
        "bidirectionalFrontierCount": int(node_count),
        "rawNeighborPairCount": int(len(raw_first)),
        "compatibleContinuationEdgeCount": int(len(edge_first)),
        "eligibleTwoDimensionalCandidateCount": int(np.count_nonzero(eligible)),
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "selectedMutualFirstHitCount": int(np.count_nonzero(selected & mutual)),
        "selectedInterfaceCount": int(len(final_selected_interfaces)),
        "selectedContinuationEdgeCount": int(np.count_nonzero(selected_edge)),
        "componentCount": int(len(component_size)),
        "componentWithAtLeast8RibbonsCount": int(np.count_nonzero(component_size >= 8)),
        "componentWithAtLeast32RibbonsCount": int(
            np.count_nonzero(component_size >= 32)
        ),
        "largestComponentRibbonCounts": [
            int(value) for value in component_size[component_order[:32]]
        ],
        "supportDegree": _percentiles(degree[degree > 0]),
        "tangentRankRatio": _percentiles(tangent_rank_ratio[degree > 0]),
        "edgeScore": _percentiles(edge_score),
        "identityLabelsUsed": False,
    }
    return arrays, stats


def _component_colors(count: int) -> np.ndarray:
    return np.asarray(
        [
            tuple(
                int(round(255.0 * channel))
                for channel in colorsys.hsv_to_rgb(
                    (0.08 + 0.61803398875 * index) % 1.0,
                    0.72,
                    1.0,
                )
            )
            for index in range(count)
        ],
        dtype=np.uint8,
    )


def _draw_line(
    canvas: np.ndarray,
    first_xy: np.ndarray,
    second_xy: np.ndarray,
    color: tuple[int, int, int] | np.ndarray,
) -> None:
    steps = max(int(np.max(np.abs(second_xy - first_xy))) + 1, 1)
    points = np.rint(
        np.linspace(first_xy, second_xy, steps, dtype=np.float32)
    ).astype(np.int32)
    valid = (
        (points[:, 0] >= 0)
        & (points[:, 0] < canvas.shape[1])
        & (points[:, 1] >= 0)
        & (points[:, 1] < canvas.shape[0])
    )
    canvas[points[valid, 1], points[valid, 0]] = color


def write_continuity_overview(
    ribbon: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    path: str | Path,
    *,
    maximum_components: int,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    frontier = np.asarray(continuity["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(continuity["selected"]) > 0
    component = np.asarray(continuity["component"], dtype=np.int32)
    midpoint = np.asarray(ribbon["midpointXYZ"])[frontier]
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    colors = _component_colors(maximum_components)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        for component_id in range(maximum_components - 1, -1, -1):
            member = selected & (component == component_id)
            if not np.any(member):
                continue
            normalized = (
                midpoint[member][:, list(axes)]
                - world_start_xyz[None, list(axes)]
            ) / width[None, list(axes)]
            x = np.rint(
                offset + margin + normalized[:, 0] * (panel_size - 2 * margin)
            ).astype(np.int32)
            y = np.rint(
                panel_size
                - margin
                - normalized[:, 1] * (panel_size - 2 * margin)
            ).astype(np.int32)
            valid = (
                (x >= offset)
                & (x < offset + panel_size)
                & (y >= 0)
                & (y < panel_size)
            )
            canvas[y[valid], x[valid]] = colors[component_id]
        canvas[:, offset] = (54, 66, 75)
        canvas[:, offset + panel_size - 1] = (54, 66, 75)
    canvas[0] = (54, 66, 75)
    canvas[-1] = (54, 66, 75)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def write_largest_component_montage(
    ribbon: Mapping[str, np.ndarray],
    continuity: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    component_count: int = 12,
    columns: int = 4,
    panel_size: int = 300,
) -> Path:
    output = Path(path)
    rows = int(math.ceil(component_count / columns))
    canvas = np.full(
        (rows * panel_size, columns * panel_size, 3),
        (7, 10, 14),
        dtype=np.uint8,
    )
    frontier = np.asarray(continuity["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(continuity["selected"]) > 0
    component = np.asarray(continuity["component"], dtype=np.int32)
    midpoint = np.asarray(ribbon["midpointXYZ"])[frontier]
    thickness = np.asarray(ribbon["thicknessVoxels"])[frontier]
    edge_first = np.asarray(
        continuity["edgeFirstFrontierIndex"], dtype=np.int32
    )
    edge_second = np.asarray(
        continuity["edgeSecondFrontierIndex"], dtype=np.int32
    )
    edge_selected = np.asarray(continuity["edgeSelected"]) > 0
    for component_id in range(component_count):
        member = selected & (component == component_id)
        nodes = np.flatnonzero(member)
        if not len(nodes):
            continue
        row, column = divmod(component_id, columns)
        origin = np.asarray((column * panel_size, row * panel_size))
        local = midpoint[nodes].astype(np.float64)
        centered = local - np.mean(local, axis=0, keepdims=True)
        if len(local) >= 3:
            _, _, basis = np.linalg.svd(centered, full_matrices=False)
            projected = centered @ basis[:2].T
        else:
            projected = centered[:, :2]
        low = np.min(projected, axis=0)
        high = np.max(projected, axis=0)
        span = np.maximum(high - low, 1.0)
        margin = 18
        scale = min(
            (panel_size - 2 * margin) / span[0],
            (panel_size - 2 * margin) / span[1],
        )
        xy = np.rint(
            margin
            + 0.5 * ((panel_size - 2 * margin) - scale * span)[None, :]
            + (projected - low[None, :]) * scale
        ).astype(np.int32)
        xy[:, 1] = panel_size - 1 - xy[:, 1]
        xy += origin[None, :]
        local_index = np.full(len(frontier), -1, dtype=np.int32)
        local_index[nodes] = np.arange(len(nodes), dtype=np.int32)
        component_edge = (
            edge_selected
            & member[edge_first]
            & member[edge_second]
        )
        for first, second in zip(
            edge_first[component_edge], edge_second[component_edge]
        ):
            _draw_line(
                canvas,
                xy[local_index[first]],
                xy[local_index[second]],
                (40, 55, 62),
            )
        value = thickness[nodes]
        normalized = (value - np.percentile(value, 5)) / max(
            float(np.percentile(value, 95) - np.percentile(value, 5)),
            1.0e-6,
        )
        normalized = np.clip(normalized, 0.0, 1.0)
        color = np.column_stack(
            (
                55 + 200 * normalized,
                210 - 65 * normalized,
                235 - 180 * normalized,
            )
        ).astype(np.uint8)
        valid = (
            (xy[:, 0] >= 0)
            & (xy[:, 0] < canvas.shape[1])
            & (xy[:, 1] >= 0)
            & (xy[:, 1] < canvas.shape[0])
        )
        canvas[xy[valid, 1], xy[valid, 0]] = color[valid]
        x0, y0 = origin
        canvas[y0, x0 : x0 + panel_size] = (54, 66, 75)
        canvas[y0 + panel_size - 1, x0 : x0 + panel_size] = (54, 66, 75)
        canvas[y0 : y0 + panel_size, x0] = (54, 66, 75)
        canvas[y0 : y0 + panel_size, x0 + panel_size - 1] = (54, 66, 75)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact data hash differs from manifest: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _load_inputs(
    root: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    value = Path(root).resolve()
    ribbon_path = (
        value
        if value.is_file()
        else value / "physical-ribbon-bank-v1.json"
    )
    ribbon_manifest = json.loads(ribbon_path.read_text())
    if (
        ribbon_manifest.get("schema") != PHYSICAL_RIBBON_BANK_SCHEMA
        or ribbon_manifest.get("state") != "complete"
        or ribbon_manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("continuity requires a complete label-free ribbon bank")
    ribbon_data_path = ribbon_path.parent / str(ribbon_manifest["data"]["path"])
    ribbon = _load_npz(ribbon_data_path, ribbon_manifest["data"]["sha256"])
    interface_path = Path(
        ribbon_manifest["identity"]["interfaceBank"]["manifestPath"]
    )
    if (
        sha256_file(interface_path)
        != ribbon_manifest["identity"]["interfaceBank"]["manifestSha256"]
    ):
        raise ValueError("interface bank changed after ribbon pairing")
    interface_manifest = json.loads(interface_path.read_text())
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    interfaces = _load_npz(
        interface_data_path, interface_manifest["data"]["sha256"]
    )
    return (
        ribbon_path,
        ribbon_manifest,
        ribbon,
        interface_path,
        interface_manifest,
        interfaces,
    )


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_physical_ribbon_continuity(
    ribbon_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonContinuitySettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonContinuitySettings()
    (
        ribbon_path,
        ribbon_manifest,
        ribbon,
        interface_path,
        interface_manifest,
        interfaces,
    ) = _load_inputs(ribbon_root)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
        "version": PHYSICAL_RIBBON_CONTINUITY_VERSION,
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "interfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CONTINUITY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CONTINUITY_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    geometry = ribbon_manifest["geometry"]
    source_origin = np.asarray(
        ribbon_manifest["source"]["sourceOriginXYZ"], dtype=np.float32
    )
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    processing_stop = np.asarray(
        geometry["processingVoxelBounds"]["stopXYZExclusive"], dtype=np.float32
    )
    processing_shape = np.asarray(
        geometry["processingShapeSamplingXYZ"], dtype=np.int32
    )
    stride_xyz = (processing_stop - processing_start) / processing_shape
    if not np.allclose(stride_xyz, stride_xyz[0]):
        raise ValueError("continuity requires an isotropic sampling stride")
    stride = int(round(float(stride_xyz[0])))
    started = time.monotonic()
    arrays, stats = build_paired_boundary_continuity(
        ribbon,
        interfaces,
        processing_world_start_xyz=source_origin + processing_start,
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    solved = time.monotonic()
    _write_npz(data_path, arrays)
    world = geometry["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        arrays,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "selected-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        arrays,
        output / "largest-ribbon-components.png",
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
        "version": PHYSICAL_RIBBON_CONTINUITY_VERSION,
        "state": "complete",
        "identity": identity,
        "source": ribbon_manifest["source"],
        "geometry": geometry,
        "counts": stats,
        "timingSeconds": {
            "continuityAndSelection": round(solved - started, 6),
            "writingAndPreviews": round(finished - solved, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "componentOverview": overview.name,
            "largestComponentMontage": montage.name,
        },
        "method": {
            "state": "paired physical boundaries, not a propagated sheet label",
            "continuation": (
                "both ribbon faces translate tangentially together while "
                "thickness and the unsigned sheet normal remain continuous"
            ),
            "twoDimensionalSupport": (
                "compatible neighbor directions must span a local tangent plane"
            ),
            "exclusivity": (
                "one observed interface may bound at most one selected ribbon"
            ),
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
