from __future__ import annotations

import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch
from .matching import TraceMatch, TraceMatchSettings, match_face_traces
from .saturation_selection import load_saturation_candidates
from .stratigraphy import ConfigurationTable
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridEdge, GridFace, Int3


BOUNDARY_BAND_SCHEMA = "pareidolia.cubical-boundary-band"
BOUNDARY_BAND_VERSION = 1


@dataclass(frozen=True, slots=True)
class BoundaryBandSettings:
    depth_cells: int = 2
    leaf_shape_cells_xyz: Int3 = (4, 4, 3)

    def __post_init__(self) -> None:
        if self.depth_cells <= 0:
            raise ValueError("boundary band depth must be positive")
        leaf = tuple(int(value) for value in self.leaf_shape_cells_xyz)
        if len(leaf) != 3 or any(value <= 0 for value in leaf):
            raise ValueError("boundary leaf shape must be a positive XYZ triple")
        object.__setattr__(self, "leaf_shape_cells_xyz", leaf)

    def record(self) -> dict[str, Any]:
        return {
            "depth_cells": self.depth_cells,
            "leaf_shape_cells_xyz": list(self.leaf_shape_cells_xyz),
        }


@dataclass(frozen=True, slots=True)
class _GraphState:
    component_by_patch: dict[int, int]
    joins: tuple[TraceMatch, ...]
    semantics: str
    seam_policy: dict[str, Any]


class _UnionFind:
    def __init__(self, values: list[object]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: object) -> object:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, first: object, second: object) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if repr(first_root) <= repr(second_root):
            self.parent[second_root] = first_root
        else:
            self.parent[first_root] = second_root


def _cell_in_boundary_band(cell: Int3, shape: Int3, depth: int) -> bool:
    return any(
        cell[axis] < depth or cell[axis] >= shape[axis] - depth
        for axis in range(3)
    )


def _outer_face_side(face: GridFace, shape: Int3) -> int | None:
    coordinate = face.anchor_xyz[face.axis]
    if coordinate == 0:
        return 0
    if coordinate == shape[face.axis]:
        return 1
    return None


def _load_graph(
    selected_root: Path,
    patches: PatchTable,
    patch_values: tuple[ClippedPatch, ...],
    *,
    packet_root: Path | None,
    leaf_shape_cells_xyz: Int3,
) -> _GraphState:
    patch_by_id = {value.patch_id: value for value in patch_values}
    if packet_root is not None:
        manifest = json.loads((packet_root / "packets.json").read_text())
        identity = manifest.get("identity", {})
        if (
            manifest.get("schema") != "pareidolia.cubical-dual-axis-sheet-packets"
            or manifest.get("state") != "complete"
            or identity.get("inputPatchDataSha256")
            != sha256_file(selected_root / "selected-patches-v1.npz")
        ):
            raise ValueError("packet graph does not belong to the selected patch root")
        packet_settings = manifest.get("identity", {}).get("settings", {})
        maximum_normal_angle = float(
            packet_settings["maximum_normal_angle_degrees"]
        )
        maximum_fiber_residual = float(
            packet_settings["maximum_fiber_frame_residual_degrees"]
        )
        reconstruction_settings = TraceMatchSettings(
            orthogonal_fiber_equivalence=True,
        )
        joins: list[TraceMatch] = []
        with np.load(packet_root / "packet-graph-v1.npz") as values:
            component_by_patch = {
                int(patch_id): int(component_id)
                for patch_id, component_id in zip(
                    values["patchId"], values["componentId"]
                )
            }
            for first_id, second_id, axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            ):
                first_patch = patch_by_id[int(first_id)]
                second_patch = patch_by_id[int(second_id)]
                face = GridFace(
                    int(axis), tuple(int(value) for value in anchor)
                )
                first_trace = first_patch.trace_on(face)
                second_trace = second_patch.trace_on(face)
                if first_trace is None or second_trace is None:
                    raise ValueError("packet join does not cross its stored face")
                match = match_face_traces(
                    first_trace,
                    first_patch.estimate,
                    second_trace,
                    second_patch.estimate,
                    reconstruction_settings,
                    grid=patches.grid,
                )
                if len(match.endpoint_agreements) != 2:
                    raise ValueError("packet join lacks a reconstructible trace map")
                joins.append(match)
        semantics = "dual-axis-sheet-packet"
        seam_policy = {
            "normalAndFiberPolarity": "axial/unsigned",
            "parallelMatching": {
                "orthogonalFiberEquivalence": False,
                "maximumNormalAngleDegrees": 90.0,
                "maximumFiberResidualDegrees": 90.0,
            },
            "quarterTurnAdmission": {
                "enabled": True,
                "maximumNormalAngleDegrees": maximum_normal_angle,
                "maximumFiberFrameResidualDegrees": maximum_fiber_residual,
            },
            "strictParallelMatchesHavePriority": True,
        }
    else:
        block = assemble_surface_hierarchy(
            patches.grid,
            BlockBounds((0, 0, 0), patches.grid.shape_cells_xyz),
            patches.to_patches(),
            maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
        )
        component_by_patch = dict(block.component_by_patch)
        joins = list(block.joins)
        semantics = "strict-single-ply"
        seam_policy = {
            "normalAndFiberPolarity": "axial/unsigned",
            "parallelMatching": {
                "orthogonalFiberEquivalence": False,
                "maximumNormalAngleDegrees": 90.0,
                "maximumFiberResidualDegrees": 90.0,
            },
            "quarterTurnAdmission": {"enabled": False},
            "strictParallelMatchesHavePriority": True,
        }
    expected = {int(value) for value in patches.patch_id}
    if set(component_by_patch) != expected:
        raise ValueError(
            "boundary graph component ownership does not cover every patch"
        )
    return _GraphState(
        component_by_patch,
        tuple(joins),
        semantics,
        seam_policy,
    )


def _common_crossing_feature(edges: set[GridEdge]) -> GridEdge | Int3 | None:
    if len(edges) == 1:
        return next(iter(edges))
    shared_vertices = set.intersection(
        *(set(edge.endpoint_vertices()) for edge in edges)
    )
    return next(iter(shared_vertices)) if len(shared_vertices) == 1 else None


def _crossing_certificates(
    patches: tuple[ClippedPatch, ...],
    joins: tuple[TraceMatch, ...],
    boundary_patch_ids: set[int],
    shape_cells_xyz: Int3,
) -> tuple[
    dict[tuple[int, GridEdge], int],
    tuple[tuple[int, GridEdge | Int3], ...],
]:
    observations: list[object] = [
        (patch.patch_id, vertex.edge)
        for patch in patches
        for vertex in patch.vertices
    ]
    union = _UnionFind(observations)
    for match in joins:
        for agreement in match.endpoint_agreements:
            union.union(
                (match.first_patch_id, agreement.first_edge),
                (match.second_patch_id, agreement.second_edge),
            )
    members: dict[object, list[tuple[int, GridEdge]]] = defaultdict(list)
    for observation in observations:
        patch_id, edge = observation
        members[union.find(observation)].append((int(patch_id), edge))
    referenced_roots = {
        union.find((patch.patch_id, vertex.edge))
        for patch in patches
        if patch.patch_id in boundary_patch_ids
        for trace in patch.traces
        for vertex in (trace.first, trace.second)
        if trace.face.anchor_xyz[trace.face.axis]
        in (0, shape_cells_xyz[trace.face.axis])
    }
    ordered_roots = sorted(
        referenced_roots,
        key=lambda root: min(
            (patch_id, edge.axis, edge.anchor_xyz)
            for patch_id, edge in members[root]
        ),
    )
    group_by_root = {root: index for index, root in enumerate(ordered_roots)}
    group_by_observation: dict[tuple[int, GridEdge], int] = {}
    records: list[tuple[int, GridEdge | Int3]] = []
    for root in ordered_roots:
        edges = {edge for _, edge in members[root]}
        feature = _common_crossing_feature(edges)
        if feature is None:
            raise ValueError("retained graph contains an infeasible crossing group")
        group_id = group_by_root[root]
        records.append((group_id, feature))
        for observation in members[root]:
            group_by_observation[observation] = group_id
    return group_by_observation, tuple(records)


def _boundary_patch_table(
    source: PatchTable,
    rows: np.ndarray,
    patches_by_id: Mapping[int, ClippedPatch],
) -> PatchTable:
    patch_ids = [int(source.patch_id[row]) for row in rows]
    return PatchTable.from_patches(
        source.grid,
        tuple(patches_by_id[value] for value in patch_ids),
        configuration_id={
            int(source.patch_id[row]): int(source.configuration_id[row]) for row in rows
        },
        configuration_log_weight={
            int(source.patch_id[row]): float(source.configuration_log_weight[row])
            for row in rows
        },
        local_order={
            int(source.patch_id[row]): int(source.local_order[row]) for row in rows
        },
        normal_family={
            int(source.patch_id[row]): int(source.normal_family[row]) for row in rows
        },
    )


def _subset_configurations(
    table: ConfigurationTable,
    metadata: Mapping[str, np.ndarray],
    cells: set[Int3],
) -> tuple[ConfigurationTable, dict[str, np.ndarray]]:
    cell_indices = [
        index
        for index, values in enumerate(table.cell_xyz)
        if tuple(int(value) for value in values) in cells
    ]
    configuration_indices: list[int] = []
    configuration_offset = np.zeros(len(cell_indices) + 1, dtype=np.uint64)
    for output_index, source_index in enumerate(cell_indices):
        values = list(table.configurations_for_cell(source_index))
        configuration_indices.extend(values)
        configuration_offset[output_index + 1] = len(configuration_indices)
    configuration_index = np.asarray(configuration_indices, dtype=np.int64)
    layer_indices: list[int] = []
    layer_offset = np.zeros(len(configuration_indices) + 1, dtype=np.uint64)
    for output_index, source_index in enumerate(configuration_indices):
        low = int(table.layer_offset[source_index])
        high = int(table.layer_offset[source_index + 1])
        layer_indices.extend(range(low, high))
        layer_offset[output_index + 1] = len(layer_indices)
    layer_index = np.asarray(layer_indices, dtype=np.int64)
    subset = ConfigurationTable(
        table.cell_xyz[np.asarray(cell_indices, dtype=np.int64)],
        configuration_offset,
        table.configuration_id[configuration_index],
        table.configuration_log_weight[configuration_index],
        table.normal_hypothesis[configuration_index],
        layer_offset,
        table.layer_normal_xyz[layer_index],
        table.layer_height[layer_index],
        table.layer_covariance[layer_index],
        table.layer_fiber_xyz[layer_index],
        table.layer_fiber_angular_std_radians[layer_index],
        table.layer_confidence[layer_index],
        table.layer_evidence_score[layer_index],
        table.layer_material_probability[layer_index],
        table.layer_effective_support[layer_index],
    )
    subset.validate()
    subset_metadata = {
        name: np.asarray(values)[configuration_index]
        for name, values in metadata.items()
    }
    subset_metadata["sourceConfigurationIndex"] = configuration_index.astype(
        np.uint64
    )
    return subset, subset_metadata


def _write_configuration_band(
    output: Path,
    table: ConfigurationTable,
    metadata: Mapping[str, np.ndarray],
    *,
    candidate_root: Path,
    candidate_sha256: str,
    selected_arrays: Mapping[str, np.ndarray] | None = None,
) -> dict[str, Any]:
    data_path = output / "boundary-configurations-v1.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            **table.arrays(),
            **metadata,
            **dict(selected_arrays or {}),
        )
    temporary.replace(data_path)
    return {
        "sourceRoot": str(candidate_root),
        "sourceDataSha256": candidate_sha256,
        "cells": table.cell_count,
        "configurations": table.configuration_count,
        "layers": table.layer_count,
        "selectedCells": (
            0
            if selected_arrays is None
            else len(selected_arrays["selectedConfigurationIndex"])
        ),
        "path": data_path.name,
        "bytes": data_path.stat().st_size,
        "sha256": sha256_file(data_path),
    }


def _resolve_candidate_root(
    selected_root: Path, explicit: Path | None
) -> Path | None:
    if explicit is not None:
        return explicit.resolve()
    if (selected_root / "saturation-configurations-v1.json").is_file():
        return selected_root
    summary_path = selected_root / "summary.json"
    if summary_path.is_file():
        value = json.loads(summary_path.read_text()).get("candidateRoot")
        if value:
            return Path(value).resolve()
    return None


def _identity(
    selected_root: Path,
    packet_root: Path | None,
    candidate_root: Path | None,
    settings: BoundaryBandSettings,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": BOUNDARY_BAND_SCHEMA,
        "version": BOUNDARY_BAND_VERSION,
        "selectedRoot": str(selected_root),
        "selectedPatchManifestSha256": sha256_file(
            selected_root / "selected-patches-v1.json"
        ),
        "selectedPatchDataSha256": sha256_file(
            selected_root / "selected-patches-v1.npz"
        ),
        "packetRoot": None if packet_root is None else str(packet_root),
        "packetGraphSha256": (
            None
            if packet_root is None
            else sha256_file(packet_root / "packet-graph-v1.npz")
        ),
        "candidateRoot": None if candidate_root is None else str(candidate_root),
        "candidateDataSha256": (
            None
            if candidate_root is None
            else sha256_file(candidate_root / "saturation-configurations-v1.npz")
        ),
        "selectionManifestSha256": (
            sha256_file(selected_root / "selection-v1.json")
            if (selected_root / "selection-v1.json").is_file()
            else None
        ),
        "selectionDataSha256": (
            sha256_file(selected_root / "selection-v1.npz")
            if (selected_root / "selection-v1.npz").is_file()
            else None
        ),
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "boundary_band.py",
                "sheet_packets.py",
                "block.py",
                "matching.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _selected_configuration_band(
    selected_root: Path,
    candidate_table: ConfigurationTable,
    candidate_manifest: Mapping[str, Any],
    subset: ConfigurationTable,
    subset_metadata: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray] | None:
    selection_manifest_path = selected_root / "selection-v1.json"
    selection_data_path = selected_root / "selection-v1.npz"
    if not selection_manifest_path.is_file() or not selection_data_path.is_file():
        return None
    selection_manifest = json.loads(selection_manifest_path.read_text())
    if (
        selection_manifest.get("schema")
        != "pareidolia.raw-acus-configuration-selection"
        or int(selection_manifest.get("version", -1)) != 1
        or sha256_file(selection_data_path)
        != selection_manifest["data"]["sha256"]
    ):
        raise ValueError("selected configuration artifact is invalid")
    variant_path = selected_root / "variant.json"
    if variant_path.is_file():
        variant = json.loads(variant_path.read_text())
        expected_candidate_sha = (
            variant.get("identity", {}).get("candidateDataSha256")
        )
        if (
            expected_candidate_sha is not None
            and expected_candidate_sha != candidate_manifest["data"]["sha256"]
        ):
            raise ValueError(
                "candidate bank differs from the selected configuration source"
            )
    with np.load(selection_data_path) as values:
        selection = {name: np.asarray(values[name]) for name in values.files}
    selected_cells = np.asarray(selection["cellXYZ"], dtype=np.int32)
    selected_source = np.asarray(
        selection["sourceConfigurationIndex"], dtype=np.uint64
    )
    if len(selected_cells) != candidate_table.cell_count:
        raise ValueError("selected configuration cells do not span the candidate bank")
    configuration_cell_index = np.repeat(
        np.arange(candidate_table.cell_count, dtype=np.int64),
        np.diff(candidate_table.configuration_offset.astype(np.int64)),
    )
    if np.any(selected_source >= candidate_table.configuration_count):
        raise ValueError("selected configuration index lies outside candidate bank")
    source_cells = candidate_table.cell_xyz[
        configuration_cell_index[selected_source.astype(np.int64)]
    ]
    if not np.array_equal(source_cells, selected_cells):
        raise ValueError("selected configuration index belongs to another cell")
    selected_row_by_cell = {
        tuple(int(value) for value in cell): index
        for index, cell in enumerate(selected_cells)
    }
    subset_source = np.asarray(
        subset_metadata["sourceConfigurationIndex"], dtype=np.uint64
    )
    subset_index_by_source = {
        int(source_index): index
        for index, source_index in enumerate(subset_source)
    }
    selected_rows = np.asarray(
        [
            selected_row_by_cell[tuple(int(value) for value in cell)]
            for cell in subset.cell_xyz
        ],
        dtype=np.int64,
    )
    selected_subset_indices = np.asarray(
        [
            subset_index_by_source[int(selected_source[row])]
            for row in selected_rows
        ],
        dtype=np.uint64,
    )
    return {
        "selectedConfigurationIndex": selected_subset_indices,
        "selectedSourceConfigurationIndex": selected_source[selected_rows],
        "selectedOptionId": np.asarray(selection["optionId"])[selected_rows],
        "selectedLocalConfigurationId": np.asarray(
            selection["localConfigurationId"]
        )[selected_rows],
        "selectedConfigurationLogWeight": np.asarray(
            selection["configurationLogWeight"]
        )[selected_rows],
        "selectedLayerCount": np.asarray(selection["selectedLayerCount"])[
            selected_rows
        ],
    }


def run_boundary_band_export(
    selected_root: str | Path,
    output_root: str | Path,
    *,
    packet_root: str | Path | None = None,
    candidate_root: str | Path | None = None,
    settings: BoundaryBandSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Serialize the bounded state needed to compose one adjacent block."""

    resolved = settings or BoundaryBandSettings()
    selected = Path(selected_root).resolve()
    output = Path(output_root).resolve()
    packet = None if packet_root is None else Path(packet_root).resolve()
    candidates = _resolve_candidate_root(
        selected,
        None if candidate_root is None else Path(candidate_root),
    )
    if output in {selected, packet, candidates}:
        raise ValueError("boundary output must differ from every input root")
    patches = read_patch_shard(selected / "selected-patches-v1", verify=True)
    patch_values = patches.to_patches()
    patch_by_id = {value.patch_id: value for value in patch_values}
    if resolved.depth_cells * 2 >= min(patches.grid.shape_cells_xyz):
        raise ValueError("boundary depth leaves no independently frozen interior")
    identity = _identity(selected, packet, candidates, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "boundary-band-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("boundary output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": BOUNDARY_BAND_SCHEMA,
        "version": BOUNDARY_BAND_VERSION,
        "state": "building",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    shape = patches.grid.shape_cells_xyz
    band_cells = {
        (x, y, z)
        for z in range(shape[2])
        for y in range(shape[1])
        for x in range(shape[0])
        if _cell_in_boundary_band((x, y, z), shape, resolved.depth_cells)
    }
    boundary_rows = np.asarray(
        [
            row
            for row, values in enumerate(patches.cell_xyz)
            if _cell_in_boundary_band(
                tuple(int(value) for value in values),
                shape,
                resolved.depth_cells,
            )
        ],
        dtype=np.int64,
    )
    boundary_cells = {
        tuple(int(value) for value in values)
        for values in patches.cell_xyz[boundary_rows]
    }
    boundary_table = _boundary_patch_table(
        patches,
        boundary_rows,
        patch_by_id,
    )
    patch_manifest = write_patch_shard(
        output / "boundary-patches-v1",
        boundary_table,
        settings=resolved.record(),
        provenance={
            "boundaryIdentitySha256": identity_sha256,
            "selectedRoot": str(selected),
        },
        compressed=True,
    )
    graph = _load_graph(
        selected,
        patches,
        patch_values,
        packet_root=packet,
        leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    component_by_patch = graph.component_by_patch
    join_first = np.asarray(
        [value.first_patch_id for value in graph.joins], dtype=np.uint64
    )
    join_second = np.asarray(
        [value.second_patch_id for value in graph.joins], dtype=np.uint64
    )
    boundary_patch_ids = {int(value) for value in boundary_table.patch_id}
    join_mask = np.asarray(
        [
            int(first) in boundary_patch_ids or int(second) in boundary_patch_ids
            for first, second in zip(join_first, join_second)
        ],
        dtype=bool,
    )
    component_total = Counter(component_by_patch.values())
    component_boundary = Counter(
        component_by_patch[value] for value in boundary_patch_ids
    )
    crossing_group_by_observation, crossing_group_records = (
        _crossing_certificates(
            patch_values,
            graph.joins,
            boundary_patch_ids,
            shape,
        )
    )
    trace_records: list[tuple[Any, ...]] = []
    component_faces: dict[int, int] = defaultdict(int)
    for patch_id in sorted(boundary_patch_ids):
        patch_value = patch_by_id[patch_id]
        component_id = component_by_patch[patch_id]
        for trace in patch_value.traces:
            side = _outer_face_side(trace.face, shape)
            if side is None:
                continue
            component_faces[component_id] |= 1 << (2 * trace.face.axis + side)
            trace_records.append(
                (
                    trace.face.axis,
                    side,
                    trace.face.anchor_xyz,
                    patch_id,
                    component_id,
                    trace.first.edge.axis,
                    trace.first.edge.anchor_xyz,
                    trace.first.t,
                    trace.first.variance,
                    trace.first.point_xyz,
                    crossing_group_by_observation[(patch_id, trace.first.edge)],
                    trace.second.edge.axis,
                    trace.second.edge.anchor_xyz,
                    trace.second.t,
                    trace.second.variance,
                    trace.second.point_xyz,
                    crossing_group_by_observation[(patch_id, trace.second.edge)],
                )
            )
    trace_records.sort(key=lambda value: (value[0], value[1], value[2], value[3]))
    component_ids = np.asarray(sorted(component_boundary), dtype=np.uint64)
    component_cells: dict[int, set[Int3]] = defaultdict(set)
    boundary_component_ids = {int(value) for value in component_ids}
    for patch_id, cell in zip(patches.patch_id, patches.cell_xyz):
        component_id = component_by_patch[int(patch_id)]
        if component_id in boundary_component_ids:
            component_cells[component_id].add(
                tuple(int(value) for value in cell)
            )
    component_cell_offset = np.zeros(len(component_ids) + 1, dtype=np.uint64)
    flattened_component_cells: list[Int3] = []
    for index, component_id in enumerate(component_ids):
        flattened_component_cells.extend(
            sorted(component_cells[int(component_id)])
        )
        component_cell_offset[index + 1] = len(flattened_component_cells)
    crossing_group_ids = np.asarray(
        [value[0] for value in crossing_group_records], dtype=np.uint64
    )
    crossing_feature_kind = np.asarray(
        [
            0 if isinstance(value[1], GridEdge) else 1
            for value in crossing_group_records
        ],
        dtype=np.uint8,
    )
    crossing_feature_edge_axis = np.asarray(
        [
            value[1].axis if isinstance(value[1], GridEdge) else -1
            for value in crossing_group_records
        ],
        dtype=np.int8,
    )
    crossing_feature_anchor = np.asarray(
        [
            value[1].anchor_xyz
            if isinstance(value[1], GridEdge)
            else value[1]
            for value in crossing_group_records
        ],
        dtype=np.int32,
    ).reshape(len(crossing_group_records), 3)
    interface_path = output / "boundary-interface-v1.npz"
    temporary = interface_path.with_suffix(interface_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            boundaryPatchId=boundary_table.patch_id,
            boundaryPatchComponentId=np.asarray(
                [component_by_patch[int(value)] for value in boundary_table.patch_id],
                dtype=np.uint64,
            ),
            boundaryJoinFirstPatchId=join_first[join_mask],
            boundaryJoinSecondPatchId=join_second[join_mask],
            componentId=component_ids,
            componentTotalPatchCount=np.asarray(
                [component_total[int(value)] for value in component_ids],
                dtype=np.uint64,
            ),
            componentBoundaryPatchCount=np.asarray(
                [component_boundary[int(value)] for value in component_ids],
                dtype=np.uint64,
            ),
            componentExteriorFaceMask=np.asarray(
                [component_faces[int(value)] for value in component_ids],
                dtype=np.uint8,
            ),
            componentCellOffset=component_cell_offset,
            componentCellXYZ=np.asarray(
                flattened_component_cells, dtype=np.int32
            ).reshape(len(flattened_component_cells), 3),
            crossingGroupId=crossing_group_ids,
            crossingGroupFeatureKind=crossing_feature_kind,
            crossingGroupFeatureEdgeAxis=crossing_feature_edge_axis,
            crossingGroupFeatureAnchorXYZ=crossing_feature_anchor,
            traceFaceAxis=np.asarray(
                [value[0] for value in trace_records], dtype=np.int8
            ),
            traceFaceSide=np.asarray(
                [value[1] for value in trace_records], dtype=np.int8
            ),
            traceFaceAnchorXYZ=np.asarray(
                [value[2] for value in trace_records], dtype=np.int32
            ).reshape(len(trace_records), 3),
            tracePatchId=np.asarray(
                [value[3] for value in trace_records], dtype=np.uint64
            ),
            traceComponentId=np.asarray(
                [value[4] for value in trace_records], dtype=np.uint64
            ),
            traceFirstEdgeAxis=np.asarray(
                [value[5] for value in trace_records], dtype=np.int8
            ),
            traceFirstEdgeAnchorXYZ=np.asarray(
                [value[6] for value in trace_records], dtype=np.int32
            ).reshape(len(trace_records), 3),
            traceFirstT=np.asarray(
                [value[7] for value in trace_records], dtype=np.float32
            ),
            traceFirstVariance=np.asarray(
                [value[8] for value in trace_records], dtype=np.float32
            ),
            traceFirstPointWorldXYZ=np.asarray(
                [value[9] for value in trace_records], dtype=np.float32
            ).reshape(len(trace_records), 3),
            traceFirstCrossingGroupId=np.asarray(
                [value[10] for value in trace_records], dtype=np.uint64
            ),
            traceSecondEdgeAxis=np.asarray(
                [value[11] for value in trace_records], dtype=np.int8
            ),
            traceSecondEdgeAnchorXYZ=np.asarray(
                [value[12] for value in trace_records], dtype=np.int32
            ).reshape(len(trace_records), 3),
            traceSecondT=np.asarray(
                [value[13] for value in trace_records], dtype=np.float32
            ),
            traceSecondVariance=np.asarray(
                [value[14] for value in trace_records], dtype=np.float32
            ),
            traceSecondPointWorldXYZ=np.asarray(
                [value[15] for value in trace_records], dtype=np.float32
            ).reshape(len(trace_records), 3),
            traceSecondCrossingGroupId=np.asarray(
                [value[16] for value in trace_records], dtype=np.uint64
            ),
        )
    temporary.replace(interface_path)

    candidate_manifest = None
    if candidates is not None:
        candidate_table, candidate_metadata, source_manifest = (
            load_saturation_candidates(candidates)
        )
        expected_cells = {
            (x, y, z)
            for z in range(shape[2])
            for y in range(shape[1])
            for x in range(shape[0])
        }
        candidate_cells = {
            tuple(int(value) for value in row) for row in candidate_table.cell_xyz
        }
        if candidate_cells != expected_cells:
            raise ValueError("candidate bank does not span the selected block")
        candidate_subset, metadata_subset = _subset_configurations(
            candidate_table, candidate_metadata, band_cells
        )
        selected_arrays = _selected_configuration_band(
            selected,
            candidate_table,
            source_manifest,
            candidate_subset,
            metadata_subset,
        )
        candidate_manifest = _write_configuration_band(
            output,
            candidate_subset,
            metadata_subset,
            candidate_root=candidates,
            candidate_sha256=source_manifest["data"]["sha256"],
            selected_arrays=selected_arrays,
        )

    face_counts = Counter((int(value[0]), int(value[1])) for value in trace_records)
    manifest.update(
        {
            "state": "complete",
            "selectedRoot": str(selected),
            "packetRoot": None if packet is None else str(packet),
            "graphSemantics": graph.semantics,
            "seamMatchingPolicy": graph.seam_policy,
            "grid": patch_manifest["grid"],
            "settings": resolved.record(),
            "statistics": {
                "boundaryBandCells": len(band_cells),
                "boundaryCellsWithPatches": len(boundary_cells),
                "boundaryPatches": boundary_table.patch_count,
                "totalGraphComponents": len(set(component_by_patch.values())),
                "boundaryComponents": len(component_ids),
                "boundaryComponentOccupiedCells": len(flattened_component_cells),
                "incidentRetainedJoins": int(np.count_nonzero(join_mask)),
                "boundaryCrossingGroups": len(crossing_group_records),
                "exteriorTraces": len(trace_records),
                "exteriorTracesByFace": {
                    f"{axis}:{side}": face_counts[(axis, side)]
                    for axis in range(3)
                    for side in range(2)
                },
                "frozenInteriorCells": int(
                    np.prod(
                        np.maximum(
                            np.asarray(shape) - 2 * resolved.depth_cells,
                            0,
                        )
                    )
                ),
            },
            "artifacts": {
                "patches": {
                    "manifest": "boundary-patches-v1.json",
                    "data": "boundary-patches-v1.npz",
                    "sha256": patch_manifest["data"]["sha256"],
                },
                "interface": {
                    "path": interface_path.name,
                    "bytes": interface_path.stat().st_size,
                    "sha256": sha256_file(interface_path),
                },
                "configurations": candidate_manifest,
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
