from __future__ import annotations

import colorsys
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, VoxelBounds, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .macro_orientation import (
    MACRO_ORIENTATION_SCHEMA,
    MACRO_ORIENTATION_STEM,
    sample_orientation_field,
)
from .material_interface import MATERIAL_INTERFACE_SCHEMA, MATERIAL_INTERFACE_STEM
from .one_sided_interface import (
    ONE_SIDED_INTERFACE_SCHEMA,
    ONE_SIDED_INTERFACE_STEM,
)


MATERIAL_SURFACE_GRAPH_SCHEMA = "pareidolia.material-interface-surface-graph"
MATERIAL_SURFACE_GRAPH_VERSION = 1
MATERIAL_SURFACE_GRAPH_STEM = "material-interface-surface-graph-v1"


@dataclass(frozen=True, slots=True)
class MaterialSurfaceGraphSettings:
    minimum_local_evidence: float = 0.5
    minimum_macro_confidence: float = 0.35
    maximum_raw_to_macro_normal_degrees: float = 50.0
    maximum_neighbor_macro_normal_degrees: float = 25.0
    maximum_neighbor_signed_normal_degrees: float = 50.0
    maximum_normal_height_sampling_steps: float = 1.15
    maximum_tangent_deviation_degrees: float = 35.0
    neighbor_radius_sampling_steps: float = math.sqrt(3.0)
    tangent_column_width_sampling_steps: float = 1.5
    # About 44 microns at either calibrated dataset scale: above measured
    # within-face localization jitter, but below the 80-micron minimum ply
    # thickness used by the physical detector.
    maximum_column_depth_range_sampling_steps: float = 2.25
    maximum_physical_seed_position_residual_sampling_steps: float = 0.75
    maximum_physical_seed_signed_normal_degrees: float = 15.0
    minimum_physical_anchor_samples_for_priority: int = 4
    minimum_component_samples_for_preview: int = 8
    maximum_preview_components: int = 128

    def __post_init__(self) -> None:
        fractions = (self.minimum_local_evidence, self.minimum_macro_confidence)
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("surface graph confidence thresholds must lie in [0, 1]")
        angles = (
            self.maximum_raw_to_macro_normal_degrees,
            self.maximum_neighbor_macro_normal_degrees,
            self.maximum_neighbor_signed_normal_degrees,
            self.maximum_tangent_deviation_degrees,
        )
        if any(not 0.0 < value < 90.0 for value in angles):
            raise ValueError("surface graph angle limits must lie in (0, 90)")
        if self.maximum_normal_height_sampling_steps <= 0.0:
            raise ValueError("surface graph height cap must be positive")
        if self.neighbor_radius_sampling_steps < 1.0:
            raise ValueError("surface graph neighbor radius must be at least one sample")
        if self.tangent_column_width_sampling_steps <= 0.0:
            raise ValueError("surface graph tangent-column width must be positive")
        if self.maximum_column_depth_range_sampling_steps <= 0.0:
            raise ValueError("surface graph column depth range must be positive")
        if self.maximum_physical_seed_position_residual_sampling_steps <= 0.0:
            raise ValueError("physical seed position cap must be positive")
        if not 0.0 < self.maximum_physical_seed_signed_normal_degrees < 90.0:
            raise ValueError("physical seed normal cap must lie in (0, 90)")
        if self.minimum_physical_anchor_samples_for_priority < 1:
            raise ValueError("physical component priority requires anchors")
        if self.minimum_component_samples_for_preview < 1:
            raise ValueError("preview component size must be positive")
        if self.maximum_preview_components < 1:
            raise ValueError("maximum preview components must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _load_npz(manifest_path: Path, manifest: Mapping[str, Any]) -> dict[str, np.ndarray]:
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"artifact data changed after manifest creation: {data_path}")
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


def _component_labels(
    node_count: int, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray, int]:
    label = np.arange(node_count, dtype=np.int32)
    iteration = 0
    for iteration in range(1, 257):
        previous = label.copy()
        if len(first):
            minimum = np.minimum(label[first], label[second])
            np.minimum.at(label, first, minimum)
            np.minimum.at(label, second, minimum)
        for _ in range(8):
            label = label[label]
        if np.array_equal(label, previous):
            break
    else:
        raise RuntimeError("material surface component labeling did not converge")
    root, count = np.unique(label, return_counts=True)
    order = np.lexsort((root, -count))
    rank_by_root = np.full(node_count, -1, dtype=np.int32)
    rank_by_root[root[order]] = np.arange(len(root), dtype=np.int32)
    return rank_by_root[label], count[order], iteration


def _plane_basis(normals: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(normals, dtype=np.float64)
    reference_axis = np.argmin(np.abs(values), axis=1)
    reference = np.zeros_like(values)
    reference[np.arange(len(values)), reference_axis] = 1.0
    first = np.cross(values, reference)
    first /= np.maximum(np.linalg.norm(first, axis=1, keepdims=True), 1.0e-9)
    second = np.cross(values, first)
    second /= np.maximum(np.linalg.norm(second, axis=1, keepdims=True), 1.0e-9)
    return first, second


def _tangent_column_records(
    position: np.ndarray,
    sample_bin: np.ndarray,
    bin_center: np.ndarray,
    bin_normal: np.ndarray,
    *,
    stride: int,
    width_sampling_steps: float,
) -> tuple[np.ndarray, np.ndarray]:
    first, second = _plane_basis(bin_normal)
    delta = (position - bin_center) / float(stride)
    first_coordinate = np.einsum("ij,ij->i", delta, first)
    second_coordinate = np.einsum("ij,ij->i", delta, second)
    depth = np.einsum("ij,ij->i", delta, bin_normal)
    column_record = np.column_stack(
        (
            sample_bin,
            np.rint(first_coordinate / width_sampling_steps).astype(np.int64),
            np.rint(second_coordinate / width_sampling_steps).astype(np.int64),
        )
    )
    return column_record, depth.astype(np.float32)


def _tangent_columns(
    position: np.ndarray,
    sample_bin: np.ndarray,
    bin_center: np.ndarray,
    bin_normal: np.ndarray,
    *,
    stride: int,
    width_sampling_steps: float,
) -> tuple[np.ndarray, np.ndarray]:
    column_record, depth = _tangent_column_records(
        position,
        sample_bin,
        bin_center,
        bin_normal,
        stride=stride,
        width_sampling_steps=width_sampling_steps,
    )
    _, column = np.unique(column_record, axis=0, return_inverse=True)
    return column.astype(np.int32), depth


def _find(parent: np.ndarray, value: int) -> int:
    root = int(value)
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[value]) != value:
        following = int(parent[value])
        parent[value] = root
        value = following
    return root


def _collision_safe_components(
    pre_component: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    score: np.ndarray,
    tangent_column: np.ndarray,
    normal_depth: np.ndarray,
    *,
    maximum_depth_range: float,
    immutable_seed_label: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, int]]:
    """Maximum-score unions subject to one depth interval per tangent column."""

    node_count = len(pre_component)
    pre_count = int(np.max(pre_component)) + 1 if node_count else 0
    nodes_by_component = np.argsort(pre_component, kind="stable")
    node_count_by_component = np.bincount(pre_component, minlength=pre_count)
    node_offset = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(node_count_by_component))
    )
    edge_component = pre_component[first]
    if len(first) and not np.array_equal(edge_component, pre_component[second]):
        raise ValueError("pre-collision edge crosses component identities")
    edges_by_component = np.argsort(edge_component, kind="stable")
    edge_count_by_component = np.bincount(edge_component, minlength=pre_count)
    edge_offset = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(edge_count_by_component))
    )
    retained = np.zeros(len(first), dtype=np.uint8)
    provisional = np.full(node_count, -1, dtype=np.int32)
    local_by_node = np.full(node_count, -1, dtype=np.int32)
    provisional_count = 0
    rejected_conflicts = 0
    membership_unions = 0
    retained_cycle_edges = 0
    physical_seed_conflicts = 0
    seed_label = (
        np.asarray(immutable_seed_label, dtype=np.int64)
        if immutable_seed_label is not None
        else np.full(node_count, -1, dtype=np.int64)
    )
    if len(seed_label) != node_count:
        raise ValueError("immutable physical seed labels are not node aligned")
    for component_id in range(pre_count):
        nodes = nodes_by_component[
            node_offset[component_id] : node_offset[component_id + 1]
        ]
        if not len(nodes):
            continue
        local_by_node[nodes] = np.arange(len(nodes), dtype=np.int32)
        parent = np.arange(len(nodes), dtype=np.int32)
        column_state: list[dict[int, tuple[float, float]] | None] = [
            {
                int(tangent_column[node]): (
                    float(normal_depth[node]),
                    float(normal_depth[node]),
                )
            }
            for node in nodes
        ]
        label_state = seed_label[nodes].copy()
        edge_ids = edges_by_component[
            edge_offset[component_id] : edge_offset[component_id + 1]
        ]
        if len(edge_ids):
            edge_ids = edge_ids[
                np.argsort(-score[edge_ids], kind="stable")
            ]
        for edge_id in edge_ids:
            first_root = _find(parent, int(local_by_node[first[edge_id]]))
            second_root = _find(parent, int(local_by_node[second[edge_id]]))
            if first_root == second_root:
                retained[edge_id] = 1
                retained_cycle_edges += 1
                continue
            first_state = column_state[first_root]
            second_state = column_state[second_root]
            if first_state is None or second_state is None:
                raise RuntimeError("collision-safe component state is unavailable")
            first_label = int(label_state[first_root])
            second_label = int(label_state[second_root])
            if (
                first_label >= 0
                and second_label >= 0
                and first_label != second_label
            ):
                physical_seed_conflicts += 1
                continue
            if len(first_state) < len(second_state):
                first_root, second_root = second_root, first_root
                first_state, second_state = second_state, first_state
            compatible = True
            for column_id, (second_low, second_high) in second_state.items():
                existing = first_state.get(column_id)
                if existing is None:
                    continue
                if (
                    max(existing[1], second_high)
                    - min(existing[0], second_low)
                    > maximum_depth_range
                ):
                    compatible = False
                    break
            if not compatible:
                rejected_conflicts += 1
                continue
            parent[second_root] = first_root
            for column_id, interval in second_state.items():
                existing = first_state.get(column_id)
                if existing is None:
                    first_state[column_id] = interval
                else:
                    first_state[column_id] = (
                        min(existing[0], interval[0]),
                        max(existing[1], interval[1]),
                    )
            column_state[second_root] = None
            label_state[first_root] = max(first_label, second_label)
            label_state[second_root] = -1
            retained[edge_id] = 1
            membership_unions += 1
        root_to_label: dict[int, int] = {}
        for local_index, node in enumerate(nodes):
            root = _find(parent, local_index)
            label = root_to_label.get(root)
            if label is None:
                label = provisional_count
                provisional_count += 1
                root_to_label[root] = label
            provisional[node] = label
        local_by_node[nodes] = -1
    value, count = np.unique(provisional, return_counts=True)
    order = np.lexsort((value, -count))
    rank_by_value = np.full(provisional_count, -1, dtype=np.int32)
    rank_by_value[value[order]] = np.arange(len(value), dtype=np.int32)
    final_component = rank_by_value[provisional]
    return final_component, count[order], retained, {
        "preCollisionComponentCount": pre_count,
        "postCollisionComponentCount": int(len(count)),
        "membershipUnionCount": membership_unions,
        "retainedCycleEdgeCount": retained_cycle_edges,
        "columnConflictRejectedEdgeCount": rejected_conflicts,
        "physicalSeedConflictRejectedEdgeCount": physical_seed_conflicts,
    }


def _component_colors(
    component: np.ndarray, component_size: np.ndarray, settings: MaterialSurfaceGraphSettings
) -> dict[int, tuple[int, int, int]]:
    selected = np.flatnonzero(
        component_size >= settings.minimum_component_samples_for_preview
    )[: settings.maximum_preview_components]
    return {
        int(value): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.08 + 0.61803398875 * int(value)) % 1.0, 0.72, 0.98
            )
        )
        for value in selected
    }


def write_material_surface_cross_sections(
    source: VolumeSource,
    owned: VoxelBounds,
    position_world: np.ndarray,
    component: np.ndarray,
    component_size: np.ndarray,
    path: str | Path,
    *,
    display_high_raw: float,
    sampling_stride_voxels: int,
    settings: MaterialSurfaceGraphSettings,
) -> Path:
    output = Path(path)
    volume = source.memmap()
    source_origin = np.asarray(source.origin_xyz, dtype=np.int64)
    world_start = np.asarray(owned.start_xyz, dtype=np.int64) + source_origin
    colors = _component_colors(component, component_size, settings)
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
    high = max(float(display_high_raw), 1.0)
    tolerance = max(1.0, float(sampling_stride_voxels))
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            world_index = source_index + source.origin_xyz[2]
            selected = np.abs(position_world[:, 2] - world_index) <= tolerance
            x = np.rint(position_world[selected, 0] - world_start[0]).astype(np.int32)
            y = np.rint(position_world[selected, 1] - world_start[1]).astype(np.int32)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            world_index = source_index + source.origin_xyz[1]
            selected = np.abs(position_world[:, 1] - world_index) <= tolerance
            x = np.rint(position_world[selected, 0] - world_start[0]).astype(np.int32)
            y = np.rint(position_world[selected, 2] - world_start[2]).astype(np.int32)
        gray = np.clip(np.asarray(raw, dtype=np.float32) / high * 255.0, 0, 255).astype(
            np.uint8
        )
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        selected_component = component[selected]
        valid = (
            (x >= 1)
            & (x < panel.shape[1] - 1)
            & (y >= 1)
            & (y < panel.shape[0] - 1)
        )
        for x_value, y_value, component_value in zip(
            x[valid], y[valid], selected_component[valid]
        ):
            color = colors.get(int(component_value))
            if color is None:
                continue
            panel[y_value, x_value] = color
            panel[y_value - 1, x_value] = color
            panel[y_value + 1, x_value] = color
            panel[y_value, x_value - 1] = color
            panel[y_value, x_value + 1] = color
        row = view_index // 3
        column = view_index % 3
        y0 = row * panel_height + (panel_height - panel.shape[0]) // 2
        x0 = column * panel_width
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _match_physical_seed_labels(
    interface_arrays: Mapping[str, np.ndarray],
    interfaces: Mapping[str, Any],
    seed_path: Path,
    seed_manifest: Mapping[str, Any],
    *,
    settings: MaterialSurfaceGraphSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    """Map immutable paired-slab face identities onto dense interfaces."""

    seed_arrays = _load_npz(seed_path, seed_manifest)
    interface_key = np.asarray(
        interface_arrays["processingKeyXYZ"], dtype=np.int64
    )
    interface_position = np.asarray(
        interface_arrays["positionXYZ"], dtype=np.float64
    )
    interface_normal = np.asarray(
        interface_arrays["signedNormalXYZ"], dtype=np.float64
    )
    seed_key = np.asarray(seed_arrays["processingKeyXYZ"], dtype=np.int64)
    seed_position = np.asarray(seed_arrays["positionXYZ"], dtype=np.float64)
    seed_normal = np.asarray(seed_arrays["signedNormalXYZ"], dtype=np.float64)
    seed_label = np.asarray(seed_arrays["seedSurfaceLabel"], dtype=np.int64)
    seed_side = np.asarray(seed_arrays["seedBoundarySide"], dtype=np.uint8)
    seed_conflict = np.asarray(seed_arrays["seedConflict"], dtype=bool)
    shape = np.asarray(
        interfaces["geometry"]["processingShapeSamplingXYZ"], dtype=np.int64
    )
    grid = np.full(tuple(shape[::-1]), -1, dtype=np.int32)
    if np.any(grid[interface_key[:, 2], interface_key[:, 1], interface_key[:, 0]] >= 0):
        raise ValueError("material interfaces contain duplicate processing keys")
    grid[
        interface_key[:, 2], interface_key[:, 1], interface_key[:, 0]
    ] = np.arange(len(interface_key), dtype=np.int32)
    inside = np.all((seed_key >= 0) & (seed_key < shape[None, :]), axis=1)
    mapped = np.full(len(seed_key), -1, dtype=np.int32)
    mapped[inside] = grid[
        seed_key[inside, 2], seed_key[inside, 1], seed_key[inside, 0]
    ]
    exists = mapped >= 0
    stride = int(interfaces["identity"]["settings"]["sampling_stride_voxels"])
    position_residual = np.full(len(seed_key), np.inf, dtype=np.float64)
    normal_residual = np.full(len(seed_key), np.inf, dtype=np.float64)
    position_residual[exists] = np.linalg.norm(
        seed_position[exists] - interface_position[mapped[exists]], axis=1
    ) / stride
    normal_residual[exists] = np.degrees(
        np.arccos(
            np.clip(
                np.einsum(
                    "ij,ij->i",
                    seed_normal[exists],
                    interface_normal[mapped[exists]],
                ),
                -1.0,
                1.0,
            )
        )
    )
    accepted = (
        exists
        & (seed_label >= 0)
        & (seed_side <= 1)
        & ~seed_conflict
        & (
            position_residual
            <= settings.maximum_physical_seed_position_residual_sampling_steps
        )
        & (
            normal_residual
            <= settings.maximum_physical_seed_signed_normal_degrees
        )
    )
    label_by_interface = np.full(len(interface_key), -1, dtype=np.int32)
    side_by_interface = np.full(len(interface_key), 255, dtype=np.uint8)
    anchor_by_interface = np.zeros(len(interface_key), dtype=np.uint8)
    for seed_index in np.flatnonzero(accepted):
        interface_index = int(mapped[seed_index])
        label = int(seed_label[seed_index])
        side = int(seed_side[seed_index])
        existing_label = int(label_by_interface[interface_index])
        existing_side = int(side_by_interface[interface_index])
        if existing_label >= 0 and (
            existing_label != label or existing_side != side
        ):
            label_by_interface[interface_index] = -1
            side_by_interface[interface_index] = 255
            anchor_by_interface[interface_index] = 2
            continue
        if anchor_by_interface[interface_index] == 2:
            continue
        label_by_interface[interface_index] = label
        side_by_interface[interface_index] = side
        anchor_by_interface[interface_index] = 1
    ambiguous = anchor_by_interface == 2
    label_by_interface[ambiguous] = -1
    side_by_interface[ambiguous] = 255
    summary = {
        "seedInterfaceSampleCount": int(len(seed_key)),
        "seedLabeledSampleCount": int(np.count_nonzero(seed_label >= 0)),
        "mappedSeedSampleCount": int(np.count_nonzero(exists)),
        "acceptedSeedSampleCount": int(np.count_nonzero(accepted)),
        "uniquePhysicalAnchorInterfaceCount": int(
            np.count_nonzero(anchor_by_interface == 1)
        ),
        "ambiguousPhysicalAnchorInterfaceCount": int(
            np.count_nonzero(ambiguous)
        ),
        "physicalSheetIdentityCount": int(
            len(np.unique(label_by_interface[label_by_interface >= 0]))
        ),
        "physicalBoundaryFaceIdentityCount": int(
            len(
                np.unique(
                    2 * label_by_interface[label_by_interface >= 0]
                    + side_by_interface[label_by_interface >= 0]
                )
            )
        ),
        "acceptedCanonicalLowerAnchorCount": int(
            np.count_nonzero((anchor_by_interface == 1) & (side_by_interface == 0))
        ),
        "acceptedCanonicalUpperAnchorCount": int(
            np.count_nonzero((anchor_by_interface == 1) & (side_by_interface == 1))
        ),
        "positionResidualSamplingSteps": _percentiles(
            position_residual[accepted]
        ),
        "signedNormalResidualDegrees": _percentiles(normal_residual[accepted]),
    }
    return label_by_interface, side_by_interface, anchor_by_interface, summary


def run_material_surface_graph(
    interface_root: str | Path,
    macro_root: str | Path,
    output_path: str | Path,
    *,
    physical_seed_root: str | Path | None = None,
    settings: MaterialSurfaceGraphSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve(interface_root, MATERIAL_INTERFACE_STEM)
    macro_path = _resolve(macro_root, MACRO_ORIENTATION_STEM)
    interfaces = json.loads(interface_path.read_text())
    macro = json.loads(macro_path.read_text())
    if interfaces.get("schema") != MATERIAL_INTERFACE_SCHEMA or interfaces.get("state") != "complete":
        raise ValueError("surface graph requires complete material interfaces")
    if macro.get("schema") != MACRO_ORIENTATION_SCHEMA or macro.get("state") != "complete":
        raise ValueError("surface graph requires a complete macro orientation field")
    if macro["identity"]["interfaces"]["manifestSha256"] != sha256_file(interface_path):
        raise ValueError("macro field was not derived from the supplied interfaces")
    interface_arrays = _load_npz(interface_path, interfaces)
    macro_arrays = _load_npz(macro_path, macro)
    resolved = settings or MaterialSurfaceGraphSettings()
    physical_seed_path: Path | None = None
    physical_seed_manifest: dict[str, Any] | None = None
    if physical_seed_root is not None:
        physical_seed_path = _resolve(
            physical_seed_root, ONE_SIDED_INTERFACE_STEM
        )
        physical_seed_manifest = json.loads(physical_seed_path.read_text())
        if (
            physical_seed_manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA
            or physical_seed_manifest.get("state") != "complete"
        ):
            raise ValueError(
                "surface graph physical seeds require a complete one-sided interface bank"
            )
        if (
            physical_seed_manifest.get("source") != interfaces.get("source")
            or physical_seed_manifest.get("geometry") != interfaces.get("geometry")
        ):
            raise ValueError(
                "physical seeds and material interfaces must share source-aligned geometry"
            )
    identity: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_GRAPH_SCHEMA,
        "version": MATERIAL_SURFACE_GRAPH_VERSION,
        "interfaces": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interfaces["data"]["sha256"],
        },
        "macroOrientation": {
            "manifestPath": str(macro_path),
            "manifestSha256": sha256_file(macro_path),
            "dataSha256": macro["data"]["sha256"],
        },
        "physicalSeeds": (
            {
                "manifestPath": str(physical_seed_path),
                "manifestSha256": sha256_file(physical_seed_path),
                "dataSha256": physical_seed_manifest["data"]["sha256"],
            }
            if physical_seed_path is not None
            and physical_seed_manifest is not None
            else None
        ),
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_SURFACE_GRAPH_STEM}.json"
    data_path = output / f"{MATERIAL_SURFACE_GRAPH_STEM}.npz"
    preview_path = output / "material-surface-component-cross-sections.png"
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
    position = np.asarray(interface_arrays["positionXYZ"], dtype=np.float64)
    raw_normal = np.asarray(interface_arrays["signedNormalXYZ"], dtype=np.float64)
    key = np.asarray(interface_arrays["processingKeyXYZ"], dtype=np.int64)
    evidence = np.asarray(interface_arrays["localEvidenceScore"], dtype=np.float64)
    orientation = sample_orientation_field(macro_arrays)
    sample_bin = orientation["sampleBinIndex"]
    sample_group = orientation["sampleGroupIndex"]
    if len(sample_bin) != len(position):
        raise ValueError("macro sample index is not aligned with interface samples")
    macro_normal = orientation["sampleNormalXYZ"]
    macro_confidence = orientation["sampleOrientationConfidence"]
    trusted = orientation["sampleTrusted"]
    orientation_source = orientation["sampleOrientationSource"]
    orientation_center = orientation["sampleGroupCenterXYZ"]
    raw_macro_cosine = np.abs(np.einsum("ij,ij->i", raw_normal, macro_normal))
    raw_macro_angle = np.degrees(
        np.arccos(np.clip(raw_macro_cosine, 0.0, 1.0))
    )
    physical_seed_label = np.full(len(position), -1, dtype=np.int32)
    physical_seed_side = np.full(len(position), 255, dtype=np.uint8)
    physical_seed_anchor = np.zeros(len(position), dtype=np.uint8)
    physical_seed_summary: dict[str, Any] = {
        "seedInterfaceSampleCount": 0,
        "seedLabeledSampleCount": 0,
        "mappedSeedSampleCount": 0,
        "acceptedSeedSampleCount": 0,
        "uniquePhysicalAnchorInterfaceCount": 0,
        "ambiguousPhysicalAnchorInterfaceCount": 0,
        "physicalSheetIdentityCount": 0,
        "physicalBoundaryFaceIdentityCount": 0,
        "acceptedCanonicalLowerAnchorCount": 0,
        "acceptedCanonicalUpperAnchorCount": 0,
        "positionResidualSamplingSteps": _percentiles(np.empty(0)),
        "signedNormalResidualDegrees": _percentiles(np.empty(0)),
    }
    if physical_seed_path is not None and physical_seed_manifest is not None:
        (
            physical_seed_label,
            physical_seed_side,
            physical_seed_anchor,
            physical_seed_summary,
        ) = _match_physical_seed_labels(
            interface_arrays,
            interfaces,
            physical_seed_path,
            physical_seed_manifest,
            settings=resolved,
        )
    eligible = (
        trusted
        & (evidence >= resolved.minimum_local_evidence)
        & (macro_confidence >= resolved.minimum_macro_confidence)
        & (raw_macro_angle <= resolved.maximum_raw_to_macro_normal_degrees)
    )
    interface_index = np.flatnonzero(eligible)
    local_physical_seed_label = physical_seed_label[interface_index]
    local_physical_seed_side = physical_seed_side[interface_index]
    local_physical_face_identity = np.where(
        local_physical_seed_label >= 0,
        2 * local_physical_seed_label + local_physical_seed_side.astype(np.int32),
        -1,
    ).astype(np.int32)
    local_physical_seed_anchor = physical_seed_anchor[interface_index]
    local_by_interface = np.full(len(position), -1, dtype=np.int32)
    local_by_interface[interface_index] = np.arange(len(interface_index), dtype=np.int32)
    local_key = key[interface_index]
    processing_shape = tuple(
        int(value)
        for value in interfaces["geometry"]["processingShapeSamplingXYZ"]
    )
    grid = np.full(processing_shape[::-1], -1, dtype=np.int32)
    existing = grid[local_key[:, 2], local_key[:, 1], local_key[:, 0]]
    if np.any(existing >= 0):
        raise ValueError("material interfaces contain duplicate processing keys")
    grid[local_key[:, 2], local_key[:, 1], local_key[:, 0]] = np.arange(
        len(interface_index), dtype=np.int32
    )

    local_position = position[interface_index]
    local_raw_normal = raw_normal[interface_index]
    local_macro_normal = macro_normal[interface_index]
    local_macro_confidence = macro_confidence[interface_index]
    stride = int(interfaces["identity"]["settings"]["sampling_stride_voxels"])
    reach = int(math.ceil(resolved.neighbor_radius_sampling_steps))
    macro_cosine_limit = math.cos(
        math.radians(resolved.maximum_neighbor_macro_normal_degrees)
    )
    signed_cosine_limit = math.cos(
        math.radians(resolved.maximum_neighbor_signed_normal_degrees)
    )
    tangent_deviation_limit = math.radians(
        resolved.maximum_tangent_deviation_degrees
    )
    first_edges: list[np.ndarray] = []
    second_edges: list[np.ndarray] = []
    height_edges: list[np.ndarray] = []
    deviation_edges: list[np.ndarray] = []
    edge_score: list[np.ndarray] = []
    rejected = {
        "neighborCandidates": 0,
        "macroNormal": 0,
        "signedNormal": 0,
        "normalHeight": 0,
        "tangentDeviation": 0,
    }
    shape = np.asarray(processing_shape, dtype=np.int64)
    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                distance = math.sqrt(dx * dx + dy * dy + dz * dz)
                if distance > resolved.neighbor_radius_sampling_steps + 1.0e-9:
                    continue
                offset = np.asarray((dx, dy, dz), dtype=np.int64)
                valid = np.all(
                    (local_key + offset[None, :] >= 0)
                    & (local_key + offset[None, :] < shape[None, :]),
                    axis=1,
                )
                first = np.flatnonzero(valid)
                query = local_key[first] + offset[None, :]
                second = grid[query[:, 2], query[:, 1], query[:, 0]]
                exists = second >= 0
                first = first[exists]
                second = second[exists]
                if not len(first):
                    continue
                rejected["neighborCandidates"] += int(len(first))
                macro_dot = np.einsum(
                    "ij,ij->i", local_macro_normal[first], local_macro_normal[second]
                )
                macro_sign = np.where(macro_dot >= 0.0, 1.0, -1.0)
                common_normal = (
                    local_macro_normal[first]
                    + macro_sign[:, None] * local_macro_normal[second]
                )
                common_normal /= np.maximum(
                    np.linalg.norm(common_normal, axis=1, keepdims=True), 1.0e-9
                )
                signed_dot = np.einsum(
                    "ij,ij->i", local_raw_normal[first], local_raw_normal[second]
                )
                displacement = local_position[second] - local_position[first]
                signed_height = np.einsum("ij,ij->i", displacement, common_normal)
                height_sampling = np.abs(signed_height) / stride
                tangent = displacement - signed_height[:, None] * common_normal
                tangent_sampling = np.linalg.norm(tangent, axis=1) / stride
                deviation = np.arctan2(height_sampling, np.maximum(tangent_sampling, 1.0e-6))
                good_macro = np.abs(macro_dot) >= macro_cosine_limit
                good_signed = signed_dot >= signed_cosine_limit
                good_height = (
                    height_sampling
                    <= resolved.maximum_normal_height_sampling_steps
                )
                good_deviation = deviation <= tangent_deviation_limit
                rejected["macroNormal"] += int(np.count_nonzero(~good_macro))
                rejected["signedNormal"] += int(np.count_nonzero(good_macro & ~good_signed))
                rejected["normalHeight"] += int(
                    np.count_nonzero(good_macro & good_signed & ~good_height)
                )
                rejected["tangentDeviation"] += int(
                    np.count_nonzero(
                        good_macro & good_signed & good_height & ~good_deviation
                    )
                )
                accepted = good_macro & good_signed & good_height & good_deviation
                if not np.any(accepted):
                    continue
                first_edges.append(first[accepted].astype(np.int32))
                second_edges.append(second[accepted].astype(np.int32))
                height_edges.append(height_sampling[accepted].astype(np.float32))
                deviation_edges.append(
                    np.degrees(deviation[accepted]).astype(np.float32)
                )
                geometry_score = np.exp(
                    -0.5
                    * (
                        height_sampling[accepted]
                        / max(resolved.maximum_normal_height_sampling_steps, 1.0e-6)
                    )
                    ** 2
                )
                edge_score.append(
                    (
                        np.minimum(
                            evidence[interface_index[first[accepted]]],
                            evidence[interface_index[second[accepted]]],
                        )
                        * np.minimum(
                            local_macro_confidence[first[accepted]],
                            local_macro_confidence[second[accepted]],
                        )
                        * geometry_score
                    ).astype(np.float32)
                )
    if first_edges:
        first_edge = np.concatenate(first_edges)
        second_edge = np.concatenate(second_edges)
        edge_height = np.concatenate(height_edges)
        edge_deviation = np.concatenate(deviation_edges)
        score = np.concatenate(edge_score)
    else:
        first_edge = np.empty(0, dtype=np.int32)
        second_edge = np.empty(0, dtype=np.int32)
        edge_height = np.empty(0, dtype=np.float32)
        edge_deviation = np.empty(0, dtype=np.float32)
        score = np.empty(0, dtype=np.float32)
    pre_component, pre_component_size, label_iterations = _component_labels(
        len(interface_index), first_edge, second_edge
    )
    local_sample_group = sample_group[interface_index]
    tangent_column, normal_depth = _tangent_columns(
        local_position,
        local_sample_group,
        orientation_center[interface_index],
        local_macro_normal,
        stride=stride,
        width_sampling_steps=resolved.tangent_column_width_sampling_steps,
    )
    component, component_size, retained_edge, collision_summary = (
        _collision_safe_components(
            pre_component,
            first_edge,
            second_edge,
            score,
            tangent_column,
            normal_depth,
            maximum_depth_range=(
                resolved.maximum_column_depth_range_sampling_steps
            ),
            immutable_seed_label=local_physical_face_identity,
        )
    )
    component_count = len(component_size)
    component_anchor_count = np.bincount(
        component,
        weights=(local_physical_seed_anchor == 1).astype(np.int32),
        minlength=component_count,
    ).astype(np.int32)
    component_physical_label = np.full(component_count, -1, dtype=np.int32)
    component_physical_side = np.full(component_count, 255, dtype=np.uint8)
    for component_id in range(component_count):
        identities = np.unique(
            local_physical_face_identity[
                (component == component_id) & (local_physical_face_identity >= 0)
            ]
        )
        if len(identities) > 1:
            raise RuntimeError(
                "physical seed constraint left multiple boundary-face identities in one component"
            )
        if len(identities) == 1:
            identity_value = int(identities[0])
            component_physical_label[component_id] = identity_value // 2
            component_physical_side[component_id] = identity_value % 2
    priority = (
        component_anchor_count
        >= resolved.minimum_physical_anchor_samples_for_priority
    )
    component_order = np.lexsort(
        (
            np.arange(component_count, dtype=np.int32),
            -component_size,
            ~priority,
        )
    )
    rank_by_component = np.empty(component_count, dtype=np.int32)
    rank_by_component[component_order] = np.arange(
        component_count, dtype=np.int32
    )
    component = rank_by_component[component]
    component_size = component_size[component_order]
    component_anchor_count = component_anchor_count[component_order]
    component_physical_label = component_physical_label[component_order]
    component_physical_side = component_physical_side[component_order]
    node_physical_label = component_physical_label[component]
    node_physical_side = component_physical_side[component]
    retained_edge_mask = retained_edge > 0
    first_edge = first_edge[retained_edge_mask]
    second_edge = second_edge[retained_edge_mask]
    edge_height = edge_height[retained_edge_mask]
    edge_deviation = edge_deviation[retained_edge_mask]
    score = score[retained_edge_mask]
    arrays = {
        "interfaceIndex": interface_index.astype(np.int32),
        "positionXYZ": local_position.astype(np.float32),
        "signedNormalXYZ": local_raw_normal.astype(np.float32),
        "macroNormalXYZ": local_macro_normal.astype(np.float32),
        "macroOrientationConfidence": local_macro_confidence.astype(np.float32),
        "orientationSource": orientation_source[interface_index].astype(np.uint8),
        "physicalSeedAnchor": local_physical_seed_anchor.astype(np.uint8),
        "physicalSheetLabel": node_physical_label.astype(np.int32),
        "physicalBoundarySide": node_physical_side.astype(np.uint8),
        "componentPhysicalSheetLabel": component_physical_label,
        "componentPhysicalBoundarySide": component_physical_side,
        "componentPhysicalAnchorCount": component_anchor_count,
        "rawToMacroNormalDegrees": raw_macro_angle[interface_index].astype(np.float32),
        "localEvidenceScore": evidence[interface_index].astype(np.float32),
        "preCollisionComponentId": pre_component.astype(np.int32),
        "componentId": component.astype(np.int32),
        "tangentColumnId": tangent_column,
        "normalDepthSamplingSteps": normal_depth,
        "edgeFirstNode": first_edge,
        "edgeSecondNode": second_edge,
        "edgeNormalHeightSamplingSteps": edge_height,
        "edgeTangentDeviationDegrees": edge_deviation,
        "edgeScore": score,
    }
    _write_npz(data_path, arrays)
    source = VolumeSource.open(
        interfaces["source"]["path"], interfaces["source"].get("metadataPath")
    )
    owned_record = interfaces["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(int(value) for value in owned_record["startXYZ"]),
        tuple(int(value) for value in owned_record["stopXYZExclusive"]),
    )
    write_material_surface_cross_sections(
        source,
        owned,
        local_position,
        component,
        component_size,
        preview_path,
        display_high_raw=float(interfaces["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    substantial = component_size >= resolved.minimum_component_samples_for_preview
    payload: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_GRAPH_SCHEMA,
        "version": MATERIAL_SURFACE_GRAPH_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interfaces["source"],
        "geometry": interfaces["geometry"],
        "counts": {
            "interfaceSampleCount": int(len(position)),
            "eligibleNodeCount": int(len(interface_index)),
            "eligibleNodeFraction": round(len(interface_index) / max(len(position), 1), 6),
            "eligiblePhysicallyGuidedNodeCount": int(
                np.count_nonzero(orientation_source[interface_index] == 1)
            ),
            "physicallyRejectedInterfaceCount": int(
                np.count_nonzero(orientation_source == 2)
            ),
            "eligiblePhysicalAnchorNodeCount": int(
                np.count_nonzero(local_physical_seed_anchor == 1)
            ),
            "physicallyAnchoredComponentCount": int(
                np.count_nonzero(component_physical_label >= 0)
            ),
            "priorityPhysicalComponentCount": int(
                np.count_nonzero(
                    component_anchor_count
                    >= resolved.minimum_physical_anchor_samples_for_priority
                )
            ),
            "nodesInPhysicallyAnchoredComponents": int(
                np.sum(component_size[component_physical_label >= 0])
            ),
            "retainedEdgeCount": int(len(first_edge)),
            "preCollisionComponentCount": int(len(pre_component_size)),
            "componentCount": int(len(component_size)),
            "componentsAtLeast8Nodes": int(np.count_nonzero(component_size >= 8)),
            "componentsAtLeast32Nodes": int(np.count_nonzero(component_size >= 32)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_size >= 128)),
            "nodesInPreviewSizedComponents": int(np.sum(component_size[substantial])),
            "labelPropagationIterations": label_iterations,
            "preCollisionLargestComponentSizes": [
                int(value) for value in pre_component_size[:32]
            ],
            "largestComponentSizes": [int(value) for value in component_size[:32]],
            **collision_summary,
        },
        "edgeCandidateAccounting": {
            **rejected,
            "preCollisionRetainedEdgeCount": int(len(retained_edge)),
            "postCollisionRetainedEdgeCount": int(len(first_edge)),
        },
        "physicalSeedAccounting": physical_seed_summary,
        "distributions": {
            "componentSize": _percentiles(component_size),
            "rawToMacroNormalDegrees": _percentiles(raw_macro_angle[interface_index]),
            "edgeNormalHeightSamplingSteps": _percentiles(edge_height),
            "edgeTangentDeviationDegrees": _percentiles(edge_deviation),
            "edgeScore": _percentiles(score),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "node": "one signed air-to-material interface sample",
            "edge": "adjacent processing samples on the same signed face",
            "growthDirection": "local macro tangent plane only",
            "orientationSelection": (
                "nearest repeated physical air-papyrus-air mode when available; "
                "otherwise the generic unsigned interface tensor"
            ),
            "physicalIdentityConstraint": (
                "maximum-score unions may never combine different exact "
                "air-papyrus-air sheets or opposite boundary sides of one sheet"
                if physical_seed_manifest is not None
                else "not supplied"
            ),
            "crossLayerGuard": "absolute normal height plus signed-normal agreement",
            "transitiveLayerGuard": (
                "maximum-score unions with one bounded normal-depth interval "
                "per local macro tangent column"
            ),
            "oppositeFacesCollapsed": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
