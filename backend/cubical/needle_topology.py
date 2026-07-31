from __future__ import annotations

import colorsys
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    RawAcusSettings,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .needle_field import (
    BLOCK_NEEDLE_FIELD_SCHEMA,
    curvature_aware_tangent_metrics,
)
from .export import rgb_png


BLOCK_NEEDLE_TOPOLOGY_SCHEMA = "pareidolia.block-acus-needle-topology"
BLOCK_NEEDLE_TOPOLOGY_VERSION = 1
BLOCK_NEEDLE_TOPOLOGY_STEM = "block-needle-topology-v1"


@dataclass(frozen=True, slots=True)
class BlockNeedleTopologySettings:
    """Dataset-independent controls for one block-global ply graph.

    Physical scales such as layer spacing, plausible sheet thickness, needle
    length, and fiber spread are inherited from the immutable raw-Acus inputs.
    These settings control robust calibration and graph optimization only.
    """

    profile_depth_step_multiplier: float = 2.0
    profile_lateral_kernel_needle_fraction: float = 0.5
    profile_normal_kernel_degrees: float = 25.0
    calibration_affinity_quantile: float = 0.5
    calibration_outlier_standard_deviations: float = 3.0
    growth_scale_multiplier: float = 2.0
    growth_minimum_curved_affinity: float = 0.3
    minimum_bridge_edges: int = 2
    minimum_bridge_endpoints_per_side: int = 2
    minimum_bridge_midpoint_span_voxels: float = 2.0
    minimum_bridge_score: float = 0.5
    maximum_growth_iterations: int = 8
    fingerprint_chunk_needles: int = 2048
    edge_score_chunk: int = 32768

    def __post_init__(self) -> None:
        positive = (
            self.profile_depth_step_multiplier,
            self.profile_lateral_kernel_needle_fraction,
            self.profile_normal_kernel_degrees,
            self.calibration_outlier_standard_deviations,
            self.growth_scale_multiplier,
            self.minimum_bridge_midpoint_span_voxels,
            self.minimum_bridge_score,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("needle-topology scales must be finite and positive")
        if not 0.0 < self.calibration_affinity_quantile < 1.0:
            raise ValueError("calibration affinity quantile must lie in (0, 1)")
        if not 0.0 < self.growth_minimum_curved_affinity < 1.0:
            raise ValueError("growth affinity must lie in (0, 1)")
        integer_positive = (
            self.minimum_bridge_edges,
            self.minimum_bridge_endpoints_per_side,
            self.maximum_growth_iterations,
            self.fingerprint_chunk_needles,
            self.edge_score_chunk,
        )
        if any(value < 1 for value in integer_positive):
            raise ValueError("needle-topology integer settings must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


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


def _draw_projection_line(
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


def write_needle_topology_projection_png(
    field_arrays: Mapping[str, np.ndarray],
    topology_arrays: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    maximum_components: int,
    fit_selected: bool = False,
    panel_size: int = 640,
) -> Path:
    """Write deterministic XY/XZ/YZ projections of selected ply graph edges."""

    if maximum_components < 1 or panel_size < 128:
        raise ValueError("projection component count and size must be positive")
    center = np.asarray(field_arrays["centerXYZ"], dtype=np.float64)
    component = np.asarray(topology_arrays["plyComponentId"], dtype=np.int32)
    first = np.asarray(topology_arrays["edgeFirstNeedle"], dtype=np.int32)
    second = np.asarray(topology_arrays["edgeSecondNeedle"], dtype=np.int32)
    selected_edge = np.asarray(topology_arrays["edgeSelected"], dtype=bool)
    values, sizes = np.unique(component, return_counts=True)
    ranking = np.lexsort((values, -sizes))[:maximum_components]
    selected_components = values[ranking]
    selected_set = set(int(value) for value in selected_components)
    selected_node = np.asarray(
        [int(value) in selected_set for value in component], dtype=bool
    )
    low = np.asarray(world_start_xyz, dtype=np.float64).copy()
    high = np.asarray(world_stop_xyz, dtype=np.float64).copy()
    if fit_selected and np.any(selected_node):
        low = np.min(center[selected_node], axis=0)
        high = np.max(center[selected_node], axis=0)
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
                (0.08 + 0.61803398875 * rank) % 1.0, 0.68, 0.98
            )
        )
        for rank, value in enumerate(selected_components)
    }

    def project(
        point: np.ndarray, panel: int, axes: tuple[int, int]
    ) -> tuple[float, float]:
        width = np.maximum(high[list(axes)] - low[list(axes)], 1.0e-8)
        normalized = (point[list(axes)] - low[list(axes)]) / width
        return (
            panel * panel_size
            + margin
            + normalized[0] * (panel_size - 2 * margin),
            panel_size - margin - normalized[1] * (panel_size - 2 * margin),
        )

    visible_edge = selected_edge & selected_node[first] & selected_node[second]
    for panel, axes in enumerate(projections):
        panel_offset = panel * panel_size
        image[margin, panel_offset + margin : panel_offset + panel_size - margin] = (
            52,
            61,
            72,
        )
        image[
            panel_size - margin,
            panel_offset + margin : panel_offset + panel_size - margin,
        ] = (52, 61, 72)
        image[margin : panel_size - margin, panel_offset + margin] = (52, 61, 72)
        image[
            margin : panel_size - margin, panel_offset + panel_size - margin
        ] = (52, 61, 72)
        for edge_index in np.flatnonzero(visible_edge):
            left = int(first[edge_index])
            right = int(second[edge_index])
            value = int(component[left])
            color = tuple(max(20, int(channel * 0.58)) for channel in colors[value])
            _draw_projection_line(
                image,
                project(center[left], panel, axes),
                project(center[right], panel, axes),
                color,
            )
        for node in np.flatnonzero(selected_node):
            x_float, y_float = project(center[node], panel, axes)
            x = int(round(x_float))
            y = int(round(y_float))
            if 1 <= x < image.shape[1] - 1 and 1 <= y < image.shape[0] - 1:
                image[y - 1 : y + 2, x - 1 : x + 2] = colors[int(component[node])]
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(image))
    temporary.replace(output)
    return output


def _load_field_artifact(
    field_root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    root = Path(field_root).resolve()
    manifest_path = root if root.is_file() else root / "block-needle-field-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != BLOCK_NEEDLE_FIELD_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("needle topology requires a complete block needle field")
    root = manifest_path.parent
    data_record = manifest["data"]
    data_path = root / str(data_record["path"])
    if sha256_file(data_path) != data_record["sha256"]:
        raise ValueError("block needle-field data hash differs from its manifest")
    required = (
        "centerXYZ",
        "directionXYZ",
        "score",
        "axialCoverage",
        "supportScore",
        "normalXYZ",
        "candidateSupportFraction",
        "candidateProbability",
        "candidateLabel",
        "neighborIndex",
        "neighborWeight",
    )
    with np.load(data_path) as values:
        missing = set(required) - set(values.files)
        if missing:
            raise ValueError(f"needle field is missing arrays: {sorted(missing)}")
        arrays = {name: np.asarray(values[name]) for name in required}
    count = len(arrays["centerXYZ"])
    if arrays["centerXYZ"].shape != (count, 3):
        raise ValueError("needle centers have invalid shape")
    if arrays["directionXYZ"].shape != (count, 3):
        raise ValueError("needle directions have invalid shape")
    if arrays["normalXYZ"].shape != (count, 3):
        raise ValueError("needle normals have invalid shape")
    if arrays["neighborIndex"].shape[0] != count:
        raise ValueError("needle neighbor graph has invalid shape")
    return manifest_path, manifest, arrays


def _raw_settings_from_field(
    field_manifest: Mapping[str, Any],
) -> tuple[RawAcusSettings, tuple[dict[str, Any], ...]]:
    settings_records: list[dict[str, Any]] = []
    provenance: list[dict[str, Any]] = []
    for value in field_manifest["source"]["rawRoots"]:
        root = Path(str(value["root"])).resolve()
        pipeline_path = root / "pipeline.json"
        pipeline_sha256 = sha256_file(pipeline_path)
        if pipeline_sha256 != value["pipelineSha256"]:
            raise ValueError("raw-Acus pipeline changed after needle-field creation")
        pipeline = json.loads(pipeline_path.read_text())
        settings_record = dict(pipeline["identity"]["settings"])
        settings_records.append(settings_record)
        provenance.append(
            {
                "root": str(root),
                "pipelineSha256": pipeline_sha256,
                "settingsSha256": canonical_json_hash(settings_record),
            }
        )
    if not settings_records:
        raise ValueError("needle field identifies no raw-Acus source")
    reference = canonical_json_hash(settings_records[0])
    if any(canonical_json_hash(value) != reference for value in settings_records[1:]):
        raise ValueError("block needle field combines incompatible raw-Acus settings")
    return RawAcusSettings(**settings_records[0]), tuple(provenance)


def build_needle_stack_fingerprints(
    arrays: Mapping[str, np.ndarray],
    depth_offsets_voxels: np.ndarray,
    *,
    lateral_kernel_voxels: float,
    normal_kernel_degrees: float,
    depth_kernel_voxels: float,
    chunk_needles: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Build a local signed stack distribution around every dense needle."""

    center = np.asarray(arrays["centerXYZ"], dtype=np.float32)
    normal = np.asarray(arrays["normalXYZ"], dtype=np.float32)
    fiber = np.asarray(arrays["directionXYZ"], dtype=np.float32)
    neighbor = np.asarray(arrays["neighborIndex"], dtype=np.int32)
    neighbor_weight = np.asarray(arrays["neighborWeight"], dtype=np.float32)
    depths = np.asarray(depth_offsets_voxels, dtype=np.float32)
    count = len(center)
    density = np.zeros((count, len(depths)), dtype=np.float32)
    orientation_moment = np.zeros_like(density)
    lateral_variance = lateral_kernel_voxels**2
    for low in range(0, count, chunk_needles):
        high = min(low + chunk_needles, count)
        index = np.maximum(neighbor[low:high], 0)
        valid = neighbor[low:high] >= 0
        displacement = center[index] - center[low:high, None, :]
        source_normal = normal[low:high, None, :]
        height = np.einsum("nki,nki->nk", displacement, source_normal)
        distance_squared = np.einsum(
            "nki,nki->nk", displacement, displacement
        )
        lateral_squared = np.maximum(distance_squared - height * height, 0.0)
        normal_cosine = np.abs(
            np.einsum("nki,ni->nk", normal[index], normal[low:high])
        )
        normal_angle = np.degrees(
            np.arccos(np.clip(normal_cosine, 0.0, 1.0))
        )
        weight = (
            neighbor_weight[low:high]
            * valid
            * np.exp(-0.5 * lateral_squared / lateral_variance)
            * np.exp(-0.5 * (normal_angle / normal_kernel_degrees) ** 2)
        )
        target_fiber = fiber[index]
        target_fiber = target_fiber - (
            np.einsum("nki,nki->nk", target_fiber, source_normal)[..., None]
            * source_normal
        )
        target_fiber /= np.maximum(
            np.linalg.norm(target_fiber, axis=-1, keepdims=True), 1.0e-8
        )
        fiber_cosine = np.clip(
            np.abs(
                np.einsum(
                    "nki,nki->nk", fiber[low:high, None, :], target_fiber
                )
            ),
            0.0,
            1.0,
        )
        orientation = 2.0 * fiber_cosine**2 - 1.0
        kernel = (
            np.exp(
                -0.5
                * (
                    (depths[None, None, :] - height[..., None])
                    / depth_kernel_voxels
                )
                ** 2
            )
            * weight[..., None]
        )
        density[low:high] = np.sum(kernel, axis=1)
        orientation_moment[low:high] = np.sum(
            kernel * orientation[..., None], axis=1
        )
    return density, orientation_moment


def _unique_neighbor_pairs(
    neighbor_index: np.ndarray,
    neighbor_weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = len(neighbor_index)
    valid = neighbor_index >= 0
    source = np.broadcast_to(
        np.arange(count, dtype=np.int32)[:, None], neighbor_index.shape
    )[valid]
    target = neighbor_index[valid]
    weight = neighbor_weight[valid]
    first = np.minimum(source, target)
    second = np.maximum(source, target)
    key = first.astype(np.int64) * count + second
    order = np.argsort(key)
    key = key[order]
    first = first[order]
    second = second[order]
    weight = weight[order]
    boundary = np.concatenate(
        (np.asarray((True,)), key[1:] != key[:-1])
    )
    starts = np.flatnonzero(boundary)
    return (
        first[starts].astype(np.int32, copy=False),
        second[starts].astype(np.int32, copy=False),
        np.maximum.reduceat(weight, starts).astype(np.float32, copy=False),
    )


def _transported_fiber_residual_degrees(
    fiber_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> np.ndarray:
    signed_cosine = np.einsum(
        "ij,ij->i", normal_xyz[first], normal_xyz[second]
    )
    aligned_second = normal_xyz[second] * np.where(
        signed_cosine < 0.0, -1.0, 1.0
    )[:, None]
    middle = normal_xyz[first] + aligned_second
    middle /= np.maximum(np.linalg.norm(middle, axis=1, keepdims=True), 1.0e-8)
    first_fiber = fiber_xyz[first] - (
        np.einsum("ij,ij->i", fiber_xyz[first], middle)[:, None] * middle
    )
    second_fiber = fiber_xyz[second] - (
        np.einsum("ij,ij->i", fiber_xyz[second], middle)[:, None] * middle
    )
    first_length = np.linalg.norm(first_fiber, axis=1, keepdims=True)
    second_length = np.linalg.norm(second_fiber, axis=1, keepdims=True)
    valid = (first_length[:, 0] > 1.0e-6) & (second_length[:, 0] > 1.0e-6)
    first_fiber /= np.maximum(first_length, 1.0e-8)
    second_fiber /= np.maximum(second_length, 1.0e-8)
    cosine = np.clip(
        np.abs(np.einsum("ij,ij->i", first_fiber, second_fiber)), 0.0, 1.0
    )
    residual = np.degrees(np.arccos(cosine)).astype(np.float32)
    residual[~valid] = 90.0
    return residual


def score_stack_fingerprint_pairs(
    density: np.ndarray,
    orientation_moment: np.ndarray,
    normal_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    *,
    chunk_edges: int,
) -> np.ndarray:
    """Compare stack density and parallel/orthogonal phase in axial gauges."""

    mismatch = np.empty(len(first), dtype=np.float32)
    for low in range(0, len(first), chunk_edges):
        high = min(low + chunk_edges, len(first))
        left = first[low:high]
        right = second[low:high]
        reverse = (
            np.einsum("ij,ij->i", normal_xyz[left], normal_xyz[right]) < 0.0
        )
        first_density = density[left].astype(np.float64)
        second_density = density[right].astype(np.float64)
        second_moment = orientation_moment[right].astype(np.float64)
        second_density = np.where(
            reverse[:, None], second_density[:, ::-1], second_density
        )
        second_moment = np.where(
            reverse[:, None], second_moment[:, ::-1], second_moment
        )
        first_mass = np.sum(first_density, axis=1, keepdims=True)
        second_mass = np.sum(second_density, axis=1, keepdims=True)
        valid = (first_mass[:, 0] > 1.0e-8) & (second_mass[:, 0] > 1.0e-8)
        first_probability = first_density / np.maximum(first_mass, 1.0e-8)
        second_probability = second_density / np.maximum(second_mass, 1.0e-8)
        overlap = np.sqrt(first_probability * second_probability)
        density_mismatch = 1.0 - np.sum(overlap, axis=1)
        first_orientation = np.divide(
            orientation_moment[left],
            density[left],
            out=np.zeros_like(orientation_moment[left]),
            where=density[left] > 1.0e-8,
        )
        second_orientation = np.divide(
            second_moment,
            second_density,
            out=np.zeros_like(second_moment),
            where=second_density > 1.0e-8,
        )
        orientation_mismatch = np.sum(
            overlap
            * np.abs(first_orientation - second_orientation)
            * 0.5,
            axis=1,
        )
        values = np.clip(density_mismatch + orientation_mismatch, 0.0, 2.0)
        values[~valid] = 2.0
        mismatch[low:high] = values.astype(np.float32)
    return mismatch


def _robust_cap(
    values: np.ndarray,
    sample: np.ndarray,
    *,
    standard_deviations: float,
    floor: float,
) -> tuple[float, dict[str, Any]]:
    selected = np.asarray(values[sample], dtype=np.float64)
    if not len(selected):
        raise ValueError("needle-topology calibration sample is empty")
    median = float(np.median(selected))
    mad = float(np.median(np.abs(selected - median)))
    robust_standard_deviation = 1.4826 * mad
    cap = max(floor, median + standard_deviations * robust_standard_deviation)
    return cap, {
        "sampleCount": len(selected),
        "median": round(median, 6),
        "mad": round(mad, 6),
        "robustStandardDeviation": round(robust_standard_deviation, 6),
        "floor": round(float(floor), 6),
        "cap": round(float(cap), 6),
    }


class _DisjointSet:
    def __init__(self, count: int) -> None:
        self.parent = np.arange(count, dtype=np.int32)
        self.size = np.ones(count, dtype=np.int32)

    def find(self, value: int) -> int:
        while int(self.parent[value]) != value:
            self.parent[value] = self.parent[self.parent[value]]
            value = int(self.parent[value])
        return value

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
        result = np.arange(len(self.parent), dtype=np.int32)
        while True:
            updated = self.parent[result]
            if np.array_equal(updated, result):
                return result
            result = updated


def _fixed_point_packet_growth(
    center_xyz: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    seed_edge: np.ndarray,
    growth_edge: np.ndarray,
    edge_score: np.ndarray,
    settings: BlockNeedleTopologySettings,
) -> tuple[np.ndarray, np.ndarray, list[dict[str, Any]]]:
    """Grow conservative ply components only through independent edge packets."""

    count = len(center_xyz)
    disjoint = _DisjointSet(count)
    selected = seed_edge.copy()
    for left, right in zip(first[seed_edge], second[seed_edge]):
        disjoint.union(int(left), int(right))

    growth_indices = np.flatnonzero(growth_edge)
    growth_first = first[growth_indices]
    growth_second = second[growth_indices]
    growth_score = edge_score[growth_indices]
    midpoint = 0.5 * (
        center_xyz[growth_first] + center_xyz[growth_second]
    )
    records: list[dict[str, Any]] = []
    for iteration in range(settings.maximum_growth_iterations):
        roots = disjoint.roots()
        first_root = roots[growth_first]
        second_root = roots[growth_second]
        cross = first_root != second_root
        if not np.any(cross):
            records.append(
                {
                    "iteration": iteration + 1,
                    "candidateComponentPairs": 0,
                    "supportedComponentPairs": 0,
                    "mergedComponentPairs": 0,
                    "supportingEdges": 0,
                }
            )
            break
        cross_indices = np.flatnonzero(cross)
        root_size = np.bincount(roots, minlength=count)
        low_root = np.minimum(first_root[cross], second_root[cross])
        high_root = np.maximum(first_root[cross], second_root[cross])
        key = low_root.astype(np.int64) * count + high_root
        order = np.argsort(key)
        key = key[order]
        cross_indices = cross_indices[order]
        low_root = low_root[order]
        high_root = high_root[order]
        boundary = np.concatenate(
            (np.asarray((True,)), key[1:] != key[:-1])
        )
        starts = np.flatnonzero(boundary)
        stops = np.concatenate((starts[1:], np.asarray((len(key),))))
        supported: list[tuple[float, int, int, np.ndarray]] = []
        for start, stop in zip(starts, stops):
            indices = cross_indices[start:stop]
            if len(indices) < settings.minimum_bridge_edges:
                continue
            component_low = int(low_root[start])
            component_high = int(high_root[start])
            left_endpoint = np.where(
                first_root[indices] == component_low,
                growth_first[indices],
                growth_second[indices],
            )
            right_endpoint = np.where(
                first_root[indices] == component_low,
                growth_second[indices],
                growth_first[indices],
            )
            required_left = min(
                settings.minimum_bridge_endpoints_per_side,
                int(root_size[component_low]),
            )
            required_right = min(
                settings.minimum_bridge_endpoints_per_side,
                int(root_size[component_high]),
            )
            if (
                len(np.unique(left_endpoint)) < required_left
                or len(np.unique(right_endpoint)) < required_right
            ):
                continue
            span = float(np.linalg.norm(np.ptp(midpoint[indices], axis=0)))
            if span < settings.minimum_bridge_midpoint_span_voxels:
                continue
            support = float(np.sum(growth_score[indices]))
            if support < settings.minimum_bridge_score:
                continue
            supported.append((support, component_low, component_high, indices))
        merged = 0
        supporting_edges = 0
        for _support, component_low, component_high, indices in sorted(
            supported, key=lambda value: (-value[0], value[1], value[2])
        ):
            if disjoint.union(component_low, component_high):
                merged += 1
                supporting_edges += len(indices)
                selected[growth_indices[indices]] = True
        records.append(
            {
                "iteration": iteration + 1,
                "candidateComponentPairs": len(starts),
                "supportedComponentPairs": len(supported),
                "mergedComponentPairs": merged,
                "supportingEdges": supporting_edges,
            }
        )
        if not merged:
            break
    roots = disjoint.roots()
    stable = np.full(count, count, dtype=np.int32)
    np.minimum.at(stable, roots, np.arange(count, dtype=np.int32))
    return stable[roots], selected, records


def analyze_block_needle_topology(
    field_manifest: Mapping[str, Any],
    arrays: Mapping[str, np.ndarray],
    raw_settings: RawAcusSettings,
    settings: BlockNeedleTopologySettings | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Infer fiber-coherent ply carriers over every needle in one block."""

    resolved = settings or BlockNeedleTopologySettings()
    center = np.asarray(arrays["centerXYZ"], dtype=np.float32)
    fiber = np.asarray(arrays["directionXYZ"], dtype=np.float32)
    normal = np.asarray(arrays["normalXYZ"], dtype=np.float32)
    neighbor_index = np.asarray(arrays["neighborIndex"], dtype=np.int32)
    neighbor_weight = np.asarray(arrays["neighborWeight"], dtype=np.float32)
    count = len(center)
    voxel_size_microns = float(field_manifest["source"]["voxelSizeMicrons"])
    field_settings = field_manifest["settings"]
    neighbor_radius = float(field_settings["neighbor_radius_voxels"])
    field_carrier_affinity = float(field_settings["carrier_minimum_affinity"])
    minimum_curvature_radius = max(
        float(field_settings.get("minimum_curvature_radius_voxels", 0.0)),
        0.5
        * float(raw_settings.plausible_sheet_thickness_microns[0])
        / voxel_size_microns,
    )
    minimum_layer_spacing = (
        raw_settings.minimum_layer_spacing_microns / voxel_size_microns
    )
    depth_step = (
        raw_settings.depth_bin_voxels * resolved.profile_depth_step_multiplier
    )
    depth_steps = int(math.ceil(neighbor_radius / depth_step))
    depth_offsets = (
        np.arange(-depth_steps, depth_steps + 1, dtype=np.float32) * depth_step
    )
    lateral_kernel = (
        raw_settings.needle_length_voxels
        * resolved.profile_lateral_kernel_needle_fraction
    )
    density, orientation_moment = build_needle_stack_fingerprints(
        arrays,
        depth_offsets,
        lateral_kernel_voxels=lateral_kernel,
        normal_kernel_degrees=resolved.profile_normal_kernel_degrees,
        depth_kernel_voxels=raw_settings.depth_kernel_voxels,
        chunk_needles=resolved.fingerprint_chunk_needles,
    )

    first, second, pair_weight = _unique_neighbor_pairs(
        neighbor_index, neighbor_weight
    )
    displacement = center[second] - center[first]
    geometry = curvature_aware_tangent_metrics(
        displacement,
        normal[first],
        normal[second],
        compatibility_sigma_voxels=float(
            field_settings["tangent_compatibility_sigma_voxels"]
        ),
        minimum_curvature_radius_voxels=minimum_curvature_radius,
    )
    fiber_residual = _transported_fiber_residual_degrees(
        fiber, normal, first, second
    )
    fingerprint_mismatch = score_stack_fingerprint_pairs(
        density,
        orientation_moment,
        normal,
        first,
        second,
        chunk_edges=resolved.edge_score_chunk,
    )
    curved_affinity = geometry["affinity"]
    calibration_pool = (
        (curved_affinity >= field_carrier_affinity)
        & (fiber_residual <= raw_settings.orthogonal_ply_std_degrees)
    )
    if np.count_nonzero(calibration_pool) < 32:
        raise ValueError("needle topology has too few high-confidence ply edges")
    calibration_threshold = float(
        np.quantile(
            curved_affinity[calibration_pool],
            resolved.calibration_affinity_quantile,
        )
    )
    calibration_sample = calibration_pool & (
        curved_affinity >= calibration_threshold
    )
    caps: dict[str, float] = {}
    calibration: dict[str, Any] = {
        "curvedAffinityPoolCount": int(np.count_nonzero(calibration_pool)),
        "curvedAffinityQuantile": resolved.calibration_affinity_quantile,
        "curvedAffinityThreshold": round(calibration_threshold, 6),
        "sampleCount": int(np.count_nonzero(calibration_sample)),
    }
    cap_inputs = (
        (
            "midpointLayerShiftVoxels",
            geometry["midpointLayerShiftVoxels"],
            raw_settings.depth_kernel_voxels,
        ),
        (
            "bendModelResidualVoxels",
            geometry["bendModelResidualVoxels"],
            raw_settings.depth_kernel_voxels,
        ),
        (
            "fiberResidualDegrees",
            fiber_residual,
            raw_settings.orthogonal_ply_std_degrees,
        ),
        ("stackFingerprintMismatch", fingerprint_mismatch, 0.0),
    )
    for name, values, floor in cap_inputs:
        cap, record = _robust_cap(
            values,
            calibration_sample,
            standard_deviations=resolved.calibration_outlier_standard_deviations,
            floor=float(floor),
        )
        caps[name] = cap
        calibration[name] = record

    midpoint = geometry["midpointLayerShiftVoxels"]
    bend = geometry["bendModelResidualVoxels"]
    radius = geometry["curvatureRadiusVoxels"]
    seed_edge = (
        (curved_affinity >= field_carrier_affinity)
        & (midpoint <= caps["midpointLayerShiftVoxels"])
        & (bend <= caps["bendModelResidualVoxels"])
        & (fiber_residual <= caps["fiberResidualDegrees"])
        & (fingerprint_mismatch <= caps["stackFingerprintMismatch"])
        & (radius >= minimum_curvature_radius)
    )
    multiplier = resolved.growth_scale_multiplier
    growth_midpoint_cap = min(
        caps["midpointLayerShiftVoxels"] * multiplier,
        0.95 * minimum_layer_spacing,
    )
    growth_caps = {
        "midpointLayerShiftVoxels": growth_midpoint_cap,
        "bendModelResidualVoxels": caps["bendModelResidualVoxels"] * multiplier,
        "fiberResidualDegrees": caps["fiberResidualDegrees"] * multiplier,
        "stackFingerprintMismatch": caps["stackFingerprintMismatch"] * multiplier,
    }
    growth_edge = (
        (curved_affinity >= resolved.growth_minimum_curved_affinity)
        & (midpoint <= growth_caps["midpointLayerShiftVoxels"])
        & (bend <= growth_caps["bendModelResidualVoxels"])
        & (fiber_residual <= growth_caps["fiberResidualDegrees"])
        & (fingerprint_mismatch <= growth_caps["stackFingerprintMismatch"])
        & (radius >= minimum_curvature_radius)
    )
    edge_score = (
        np.exp(
            -0.5
            * (midpoint / max(caps["midpointLayerShiftVoxels"], 1.0e-6)) ** 2
        )
        * np.exp(
            -0.5
            * (bend / max(caps["bendModelResidualVoxels"], 1.0e-6)) ** 2
        )
        * np.exp(
            -0.5
            * (fiber_residual / max(caps["fiberResidualDegrees"], 1.0e-6)) ** 2
        )
        * np.exp(
            -0.5
            * (
                fingerprint_mismatch
                / max(caps["stackFingerprintMismatch"], 1.0e-6)
            )
            ** 2
        )
    ).astype(np.float32)
    component_id, selected_edge, growth_records = _fixed_point_packet_growth(
        center,
        first,
        second,
        seed_edge,
        growth_edge,
        edge_score,
        resolved,
    )

    component_values, component_sizes = np.unique(
        component_id, return_counts=True
    )
    ranking = np.lexsort((component_values, -component_sizes))
    component_values = component_values[ranking]
    component_sizes = component_sizes[ranking]
    size_by_id = np.zeros(count, dtype=np.int32)
    size_by_id[component_values] = component_sizes
    node_component_size = size_by_id[component_id]
    degree = np.bincount(
        np.concatenate((first[selected_edge], second[selected_edge])),
        minlength=count,
    ).astype(np.uint16)
    selected_support = arrays["candidateSupportFraction"][
        np.arange(count), arrays["candidateLabel"]
    ]
    maximum_probability = np.max(arrays["candidateProbability"], axis=1)
    quality = np.clip(
        np.asarray(arrays["supportScore"], dtype=np.float64)
        * np.asarray(arrays["axialCoverage"], dtype=np.float64)
        * selected_support
        * maximum_probability,
        0.0,
        1.0,
    ) ** 0.25
    quality_mass = max(float(np.sum(quality)), 1.0e-12)

    top: list[dict[str, Any]] = []
    for rank, (component, component_size) in enumerate(
        zip(component_values[:64], component_sizes[:64]), start=1
    ):
        members = np.flatnonzero(component_id == component)
        points = center[members]
        member_normals = normal[members].astype(np.float64)
        normal_projector = np.einsum(
            "ni,nj->ij", member_normals, member_normals, optimize=True
        )
        reference = np.linalg.eigh(normal_projector)[1][:, -1]
        cone = np.degrees(
            np.arccos(
                np.clip(np.abs(member_normals @ reference), 0.0, 1.0)
            )
        )
        member_fibers = fiber[members].astype(np.float64)
        fiber_projector = np.einsum(
            "ni,nj->ij", member_fibers, member_fibers, optimize=True
        )
        fiber_eigenvalues = np.linalg.eigvalsh(fiber_projector)
        fiber_eigenvalues /= max(float(np.sum(fiber_eigenvalues)), 1.0e-12)
        member_edges = selected_edge & (component_id[first] == component)
        top.append(
            {
                "rank": rank,
                "componentId": int(component),
                "needles": int(component_size),
                "selectedEdges": int(np.count_nonzero(member_edges)),
                "worldStartXYZ": [
                    round(float(value), 6) for value in np.min(points, axis=0)
                ],
                "worldStopXYZ": [
                    round(float(value), 6) for value in np.max(points, axis=0)
                ],
                "extentVoxelsXYZ": [
                    round(float(value), 6) for value in np.ptp(points, axis=0)
                ],
                "globalNormalConeDegreesDiagnosticOnly": {
                    "median": round(float(np.median(cone)), 6),
                    "p90": round(float(np.percentile(cone, 90)), 6),
                    "maximum": round(float(np.max(cone)), 6),
                },
                "fiberProjectorEigenvalueFractions": [
                    round(float(value), 6) for value in fiber_eigenvalues
                ],
                "localSelectedEdgeNormalAngleDegrees": _percentile_record(
                    geometry["normalAngleDegrees"][member_edges]
                ),
                "localSelectedEdgeStackMismatch": _percentile_record(
                    fingerprint_mismatch[member_edges]
                ),
            }
        )

    selected_stack_shortcut = selected_edge & (
        midpoint >= minimum_layer_spacing
    )
    retained_by_size = {
        str(value): int(np.count_nonzero(node_component_size >= value))
        for value in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    }
    quality_by_size = {
        str(value): round(
            float(np.sum(quality[node_component_size >= value])) / quality_mass,
            6,
        )
        for value in (2, 4, 8, 16, 32, 64, 128, 256, 512, 1024)
    }
    summary = {
        "method": {
            "scope": "one block-global fiber-coherent ply topology graph",
            "nodes": "every canonical Acus needle retained by the input field",
            "curvature": (
                "unsigned endpoint normals are aligned and scored as a cubic "
                "Hermite chord; smooth-bend offsets are antisymmetric while "
                "parallel layer shifts are symmetric"
            ),
            "stackFingerprint": (
                "local normal-depth density and cos(2 fiber-angle) moment built "
                "directly from dense needles, without cell modes"
            ),
            "optimization": (
                "robustly calibrated seed graph followed by fixed-point merges "
                "requiring independent spatially extended bridge packets"
            ),
            "componentSemantics": (
                "fiber-coherent ply carrier; orthogonal physical plies remain "
                "separate and may be associated into papyrus sheets later"
            ),
            "selfContact": (
                "normal-separated branches are not globally repulsive because a "
                "real hairpin may return beside itself; only direct layer-shift "
                "edges are excluded"
            ),
            "cells": "not used by inference",
        },
        "derivedPhysicalSettings": {
            "voxelSizeMicrons": voxel_size_microns,
            "minimumLayerSpacingVoxels": round(minimum_layer_spacing, 6),
            "minimumCurvatureRadiusVoxels": round(
                minimum_curvature_radius, 6
            ),
            "profileDepthExtentVoxels": float(depth_offsets[-1]),
            "profileDepthStepVoxels": depth_step,
            "profileLateralKernelVoxels": lateral_kernel,
            "profileDepthKernelVoxels": raw_settings.depth_kernel_voxels,
            "fiberCalibrationFloorDegrees": (
                raw_settings.orthogonal_ply_std_degrees
            ),
        },
        "calibration": calibration,
        "growthCaps": {
            name: round(float(value), 6) for name, value in growth_caps.items()
        },
        "counts": {
            "needles": count,
            "uniqueNeighborPairs": len(first),
            "seedEdges": int(np.count_nonzero(seed_edge)),
            "growthCandidateEdges": int(np.count_nonzero(growth_edge)),
            "selectedEdges": int(np.count_nonzero(selected_edge)),
            "selectedDirectLayerShiftEdges": int(
                np.count_nonzero(selected_stack_shortcut)
            ),
            "components": len(component_values),
            "isolatedNeedles": int(np.count_nonzero(component_sizes == 1)),
            "openDegreeOneNeedles": int(np.count_nonzero(degree == 1)),
            "componentsAtLeastNeedles": {
                str(value): int(np.count_nonzero(component_sizes >= value))
                for value in (8, 16, 32, 64, 128, 256, 512, 1024)
            },
            "largestComponentNeedles": int(component_sizes[0]),
            "needlesInComponentsAtLeast": retained_by_size,
        },
        "qualityMassFractionInComponentsAtLeast": quality_by_size,
        "selectedEdgeMetrics": {
            "normalAngleDegrees": _percentile_record(
                geometry["normalAngleDegrees"][selected_edge]
            ),
            "midpointLayerShiftVoxels": _percentile_record(
                midpoint[selected_edge]
            ),
            "bendModelResidualVoxels": _percentile_record(
                bend[selected_edge]
            ),
            "curvatureRadiusVoxels": _percentile_record(radius[selected_edge]),
            "fiberResidualDegrees": _percentile_record(
                fiber_residual[selected_edge]
            ),
            "stackFingerprintMismatch": _percentile_record(
                fingerprint_mismatch[selected_edge]
            ),
            "score": _percentile_record(edge_score[selected_edge]),
        },
        "fixedPointGrowth": growth_records,
        "inputNormalCarriers": field_manifest["carriers"]["counts"],
        "topComponents": top,
    }
    output_arrays = {
        "profileDepthVoxels": depth_offsets.astype(np.float32, copy=False),
        "stackDensity": density,
        "stackOrientationMoment": orientation_moment,
        "edgeFirstNeedle": first,
        "edgeSecondNeedle": second,
        "edgeNeighborWeight": pair_weight,
        "edgeNormalAngleDegrees": geometry["normalAngleDegrees"],
        "edgeMidpointLayerShiftVoxels": midpoint,
        "edgeBendModelResidualVoxels": bend,
        "edgeCurvatureRadiusVoxels": radius,
        "edgeFiberResidualDegrees": fiber_residual,
        "edgeStackFingerprintMismatch": fingerprint_mismatch,
        "edgeCurvedAffinity": curved_affinity,
        "edgeScore": edge_score,
        "edgeSeedEligible": seed_edge.astype(np.uint8),
        "edgeGrowthEligible": growth_edge.astype(np.uint8),
        "edgeSelected": selected_edge.astype(np.uint8),
        "plyComponentId": component_id,
        "plyComponentSize": node_component_size,
        "plyDegree": degree,
    }
    return summary, output_arrays


def run_block_needle_topology(
    field_root: str | Path,
    output_root: str | Path,
    *,
    settings: BlockNeedleTopologySettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize one immutable block-global Acus ply-topology artifact."""

    started = time.monotonic()
    resolved = settings or BlockNeedleTopologySettings()
    field_manifest_path, field_manifest, arrays = _load_field_artifact(field_root)
    raw_settings, raw_sources = _raw_settings_from_field(field_manifest)
    loaded = time.monotonic()
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": BLOCK_NEEDLE_TOPOLOGY_SCHEMA,
        "version": BLOCK_NEEDLE_TOPOLOGY_VERSION,
        "field": {
            "path": str(field_manifest_path),
            "manifestSha256": sha256_file(field_manifest_path),
            "dataSha256": field_manifest["data"]["sha256"],
            "identitySha256": field_manifest["identity"]["identitySha256"],
        },
        "rawSources": list(raw_sources),
        "settings": resolved.record(),
        "implementationSha256": {
            "needle_topology.py": sha256_file(Path(__file__)),
            "needle_field.py": sha256_file(module_root / "needle_field.py"),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    manifest_path = output / f"{BLOCK_NEEDLE_TOPOLOGY_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("needle-topology output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    summary, output_arrays = analyze_block_needle_topology(
        field_manifest,
        arrays,
        raw_settings,
        resolved,
    )
    analyzed = time.monotonic()
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{BLOCK_NEEDLE_TOPOLOGY_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **output_arrays)
    temporary.replace(data_path)
    world_bounds = field_manifest["source"]["worldBounds"]
    world_start = np.asarray(world_bounds["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_bounds["stopXYZExclusive"], dtype=np.float64)
    overview_path = write_needle_topology_projection_png(
        arrays,
        output_arrays,
        output / "top-12-ply-carriers.png",
        world_start_xyz=world_start,
        world_stop_xyz=world_stop,
        maximum_components=12,
    )
    largest_path = write_needle_topology_projection_png(
        arrays,
        output_arrays,
        output / "largest-ply-carrier.png",
        world_start_xyz=world_start,
        world_stop_xyz=world_stop,
        maximum_components=1,
        fit_selected=True,
    )
    payload = {
        "schema": BLOCK_NEEDLE_TOPOLOGY_SCHEMA,
        "version": BLOCK_NEEDLE_TOPOLOGY_VERSION,
        "state": "complete",
        "identity": identity,
        "settings": resolved.record(),
        "source": {
            "fieldManifest": str(field_manifest_path),
            "fieldIdentitySha256": field_manifest["identity"]["identitySha256"],
            "worldBounds": field_manifest["source"]["worldBounds"],
            "rawAcusSettings": raw_settings.record(),
        },
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "previews": {
            "top12": overview_path.name,
            "largest": largest_path.name,
            "projectionOrder": ["XY", "XZ", "YZ"],
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "analysis": round(analyzed - loaded, 6),
            "writing": round(time.monotonic() - analyzed, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
