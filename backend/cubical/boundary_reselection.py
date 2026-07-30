from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import BlockBounds, assemble_surface_block
from .boundary_merge import _BoundaryInput, _adjacency, _load_boundary
from .boundary_topology import (
    ComponentSeed,
    CrossingSeed,
    FrozenTopologySeed,
    JoinKey,
    TopologySelection,
    read_frozen_face_state,
    select_joins_with_frozen_topology,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch, clip_plane_to_cell, plane_basis
from .matching import TraceMatch, TraceMatchSettings
from .saturation_selection import _configuration_table_from_values
from .selection import ConfigurationSelection, optimize_configurations
from .stratigraphy import ConfigurationTable
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridEdge, GridSpec, Int3


BOUNDARY_RESELECTION_SCHEMA = "pareidolia.cubical-boundary-band-reselection"
BOUNDARY_RESELECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class BoundaryReselectionSettings:
    unary_scale: float = 1.0
    pairwise_scale: float = 0.2
    interior_unmatched_trace_penalty: float = 0.0
    maximum_sweeps: int = 12

    def __post_init__(self) -> None:
        if self.unary_scale <= 0.0 or self.pairwise_scale <= 0.0:
            raise ValueError("boundary reselection energy scales must be positive")
        if (
            not math.isfinite(self.interior_unmatched_trace_penalty)
            or self.interior_unmatched_trace_penalty < 0.0
        ):
            raise ValueError("boundary continuation penalty must be nonnegative")
        if self.maximum_sweeps <= 0:
            raise ValueError("boundary reselection sweeps must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class _BoundaryConfigurations:
    table: ConfigurationTable
    arrays: dict[str, np.ndarray]
    cell_index: dict[Int3, int]
    selected_index: dict[Int3, int]


@dataclass(frozen=True, slots=True)
class _ConfigurationReference:
    side: int
    source_configuration_index: int
    boundary_configuration_index: int
    current: bool


@dataclass(frozen=True, slots=True)
class _PatchReference:
    side: int
    source_patch_id: int | None
    source_configuration_index: int | None
    layer_index: int
    anchor: bool
    current_configuration: bool


def _offset_cell(cell: Int3, offset: Int3) -> Int3:
    return tuple(cell[axis] + offset[axis] for axis in range(3))  # type: ignore[return-value]


def _offset_edge(edge: GridEdge, offset: Int3) -> GridEdge:
    return GridEdge(edge.axis, _offset_cell(edge.anchor_xyz, offset))


def _offset_feature(feature: GridEdge | Int3, offset: Int3) -> GridEdge | Int3:
    return (
        _offset_edge(feature, offset)
        if isinstance(feature, GridEdge)
        else _offset_cell(feature, offset)
    )


def _load_configurations(boundary: _BoundaryInput) -> _BoundaryConfigurations:
    record = boundary.manifest["artifacts"].get("configurations")
    if not isinstance(record, dict):
        raise ValueError("joint boundary reselection requires physical candidates")
    path = boundary.root / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("boundary configuration content hash mismatch")
    with np.load(path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    required = {
        "selectedConfigurationIndex",
        "selectedSourceConfigurationIndex",
        "sourceConfigurationIndex",
    }
    if not required <= set(arrays):
        raise ValueError("boundary candidates lack current-selection provenance")
    table = _configuration_table_from_values(arrays)
    cell_index = {
        tuple(int(value) for value in cell): index
        for index, cell in enumerate(table.cell_xyz)
    }
    selected = np.asarray(arrays["selectedConfigurationIndex"], dtype=np.int64)
    if len(selected) != table.cell_count:
        raise ValueError("selected boundary configurations do not span candidate cells")
    selected_index = {
        tuple(int(value) for value in cell): int(selected[index])
        for index, cell in enumerate(table.cell_xyz)
    }
    for cell, index in selected_index.items():
        cell_row = cell_index[cell]
        if index not in table.configurations_for_cell(cell_row):
            raise ValueError("selected boundary configuration belongs to another cell")
    return _BoundaryConfigurations(table, arrays, cell_index, selected_index)


def _face_cell_is_mutable(
    cell: Int3,
    shape: Int3,
    axis: int,
    side: int,
    depth: int,
) -> bool:
    return (
        cell[axis] < depth
        if side == 0
        else cell[axis] >= shape[axis] - depth
    )


def _slab_grid(
    adjacency: Any,
    lower: _BoundaryInput,
    depth: int,
) -> tuple[GridSpec, Int3]:
    start = [0, 0, 0]
    start[adjacency.axis] = (
        adjacency.offsets[lower.side][adjacency.axis]
        + lower.patches.grid.shape_cells_xyz[adjacency.axis]
        - depth
        - 1
    )
    shape = list(adjacency.combined_grid.shape_cells_xyz)
    shape[adjacency.axis] = 2 * depth + 2
    origin = adjacency.combined_grid.vertex_world(tuple(start))
    return (
        GridSpec(
            tuple(shape),
            adjacency.combined_grid.cell_size_xyz,
            tuple(float(value) for value in origin),
            adjacency.combined_grid.coordinate_unit,
        ),
        tuple(start),
    )


def _append_layers(
    destination: dict[str, list[np.ndarray | float]],
    table: ConfigurationTable,
    configuration_index: int,
) -> None:
    low = int(table.layer_offset[configuration_index])
    high = int(table.layer_offset[configuration_index + 1])
    for index in range(low, high):
        destination["normal"].append(table.layer_normal_xyz[index])
        destination["height"].append(float(table.layer_height[index]))
        destination["covariance"].append(table.layer_covariance[index])
        destination["fiber"].append(table.layer_fiber_xyz[index])
        destination["fiber_std"].append(
            float(table.layer_fiber_angular_std_radians[index])
        )
        destination["confidence"].append(float(table.layer_confidence[index]))
        destination["evidence"].append(float(table.layer_evidence_score[index]))
        destination["material"].append(
            float(table.layer_material_probability[index])
        )
        destination["support"].append(float(table.layer_effective_support[index]))


def _anchor_rows_by_cell(table: PatchTable) -> dict[Int3, list[int]]:
    result: dict[Int3, list[int]] = defaultdict(list)
    for row, cell in enumerate(table.cell_xyz):
        result[tuple(int(value) for value in cell)].append(row)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                int(table.local_order[row]),
                float(table.height[row]),
                int(table.patch_id[row]),
            )
        )
    return result


def _combined_configuration_table(
    boundaries: tuple[_BoundaryInput, _BoundaryInput],
    configurations: tuple[_BoundaryConfigurations, _BoundaryConfigurations],
    anchor_tables: tuple[PatchTable, PatchTable],
    adjacency: Any,
    slab_grid: GridSpec,
    slab_start: Int3,
    facing_sides: tuple[int, int],
    depth: int,
) -> tuple[
    ConfigurationTable,
    dict[int, _ConfigurationReference],
    dict[Int3, tuple[int, int]],
    set[Int3],
]:
    anchor_rows = tuple(_anchor_rows_by_cell(value) for value in anchor_tables)
    cell_xyz: list[Int3] = []
    configuration_offset = [0]
    configuration_id: list[int] = []
    configuration_log_weight: list[float] = []
    normal_hypothesis: list[int] = []
    layer_offset = [0]
    layers: dict[str, list[np.ndarray | float]] = {
        name: []
        for name in (
            "normal",
            "height",
            "covariance",
            "fiber",
            "fiber_std",
            "confidence",
            "evidence",
            "material",
            "support",
        )
    }
    references: dict[int, _ConfigurationReference] = {}
    initial: dict[Int3, tuple[int, int]] = {}
    mutable_cells: set[Int3] = set()

    world_by_slab: dict[Int3, Int3] = {}
    for z in range(slab_grid.shape_cells_xyz[2]):
        for y in range(slab_grid.shape_cells_xyz[1]):
            for x in range(slab_grid.shape_cells_xyz[0]):
                slab_cell = (x, y, z)
                world_by_slab[slab_cell] = _offset_cell(slab_cell, slab_start)

    for slab_cell in sorted(world_by_slab, key=lambda value: (value[2], value[1], value[0])):
        world_cell = world_by_slab[slab_cell]
        owner: int | None = None
        local_cell: Int3 | None = None
        for side, boundary in enumerate(boundaries):
            offset = adjacency.offsets[side]
            candidate = tuple(world_cell[axis] - offset[axis] for axis in range(3))
            if boundary.patches.grid.contains_cell(candidate):
                owner = side
                local_cell = candidate
                break
        if owner is None or local_cell is None:
            raise ValueError("joint slab cell is not owned by either boundary block")
        cell_xyz.append(slab_cell)
        mutable = _face_cell_is_mutable(
            local_cell,
            boundaries[owner].patches.grid.shape_cells_xyz,
            adjacency.axis,
            facing_sides[owner],
            depth,
        )
        if mutable:
            mutable_cells.add(slab_cell)
            source = configurations[owner]
            if local_cell not in source.cell_index:
                raise ValueError("boundary candidate bank omits a mutable seam cell")
            cell_index = source.cell_index[local_cell]
            selected_boundary_index = source.selected_index[local_cell]
            selected_output_index: int | None = None
            for source_index in source.table.configurations_for_cell(cell_index):
                output_index = len(configuration_id)
                current = source_index == selected_boundary_index
                references[output_index] = _ConfigurationReference(
                    owner,
                    int(source.arrays["sourceConfigurationIndex"][source_index]),
                    source_index,
                    current,
                )
                configuration_id.append(int(source.table.configuration_id[source_index]))
                configuration_log_weight.append(
                    float(source.table.configuration_log_weight[source_index])
                )
                normal_hypothesis.append(
                    int(source.table.normal_hypothesis[source_index])
                )
                _append_layers(layers, source.table, source_index)
                layer_offset.append(len(layers["height"]))
                if current:
                    selected_output_index = output_index
            if selected_output_index is None:
                raise ValueError("current seam configuration is absent from candidates")
            initial[slab_cell] = (0, selected_output_index)
        else:
            rows = anchor_rows[owner].get(local_cell, [])
            output_index = len(configuration_id)
            references[output_index] = _ConfigurationReference(owner, -1, -1, True)
            configuration_id.append(0)
            configuration_log_weight.append(0.0)
            normal_hypothesis.append(
                int(anchor_tables[owner].normal_family[rows[0]]) if rows else 0
            )
            for row in rows:
                normal = np.asarray(anchor_tables[owner].normal_xyz[row], dtype=np.float32)
                fiber = np.asarray(anchor_tables[owner].fiber_xyz[row], dtype=np.float32)
                fiber_std = float(
                    anchor_tables[owner].fiber_angular_std_radians[row]
                )
                if not np.all(np.isfinite(fiber)):
                    fiber = np.asarray(plane_basis(normal)[0], dtype=np.float32)
                    fiber_std = math.pi
                layers["normal"].append(normal)
                layers["height"].append(float(anchor_tables[owner].height[row]))
                layers["covariance"].append(
                    anchor_tables[owner].plane_covariance[row]
                )
                layers["fiber"].append(fiber)
                layers["fiber_std"].append(fiber_std)
                layers["confidence"].append(
                    float(anchor_tables[owner].confidence[row])
                )
                layers["evidence"].append(1.0)
                layers["material"].append(1.0)
                layers["support"].append(1.0)
            layer_offset.append(len(layers["height"]))
            initial[slab_cell] = (0, output_index)
        configuration_offset.append(len(configuration_id))

    layer_count = len(layers["height"])
    table = ConfigurationTable(
        np.asarray(cell_xyz, dtype=np.int32),
        np.asarray(configuration_offset, dtype=np.uint64),
        np.asarray(configuration_id, dtype=np.uint16),
        np.asarray(configuration_log_weight, dtype=np.float32),
        np.asarray(normal_hypothesis, dtype=np.int8),
        np.asarray(layer_offset, dtype=np.uint64),
        np.asarray(layers["normal"], dtype=np.float32).reshape(layer_count, 3),
        np.asarray(layers["height"], dtype=np.float32),
        np.asarray(layers["covariance"], dtype=np.float32).reshape(layer_count, 6),
        np.asarray(layers["fiber"], dtype=np.float32).reshape(layer_count, 3),
        np.asarray(layers["fiber_std"], dtype=np.float32),
        np.asarray(layers["confidence"], dtype=np.float32),
        np.asarray(layers["evidence"], dtype=np.float32),
        np.asarray(layers["material"], dtype=np.float32),
        np.asarray(layers["support"], dtype=np.float32),
    )
    table.validate()
    return table, references, initial, mutable_cells


def _anchor_table(boundary: _BoundaryInput) -> PatchTable:
    record = boundary.manifest["artifacts"].get("anchors")
    if not isinstance(record, dict):
        raise ValueError("boundary artifact lacks immutable anchor patches")
    data_path = boundary.root / str(record["data"])
    if sha256_file(data_path) != record["sha256"]:
        raise ValueError("boundary anchor-patch content hash mismatch")
    return read_patch_shard(
        boundary.root / "boundary-anchor-patches-v1",
        verify=True,
    )


def _frozen_state(boundary: _BoundaryInput, axis: int, side: int) -> Any:
    record = boundary.manifest["artifacts"].get("frozenTopology")
    if not isinstance(record, dict):
        raise ValueError("boundary artifact lacks frozen topology certificates")
    path = boundary.root / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("boundary frozen-topology content hash mismatch")
    return read_frozen_face_state(path, axis, side)


def _rows_by_cell(table: PatchTable) -> dict[Int3, list[int]]:
    result: dict[Int3, list[int]] = defaultdict(list)
    for row, cell in enumerate(table.cell_xyz):
        result[tuple(int(value) for value in cell)].append(row)
    for rows in result.values():
        rows.sort(
            key=lambda row: (
                int(table.local_order[row]),
                float(table.height[row]),
                int(table.patch_id[row]),
            )
        )
    return result


def _realize_selected_patches(
    selection: ConfigurationSelection,
    configuration_references: Mapping[int, _ConfigurationReference],
    mutable_cells: set[Int3],
    slab_start: Int3,
    adjacency: Any,
    boundaries: tuple[_BoundaryInput, _BoundaryInput],
    anchor_tables: tuple[PatchTable, PatchTable],
    frozen_states: tuple[Any, Any],
) -> tuple[
    tuple[ClippedPatch, ...],
    dict[int, _PatchReference],
    dict[tuple[int, int], int],
    set[int],
]:
    combined_grid = adjacency.combined_grid
    patches: list[ClippedPatch] = []
    references: dict[int, _PatchReference] = {}
    anchor_temporary: dict[tuple[int, int], int] = {}
    next_patch_id = 1
    anchor_source = tuple(
        {patch.patch_id: patch for patch in table.to_patches()}
        for table in anchor_tables
    )
    for side, state in enumerate(frozen_states):
        offset = adjacency.offsets[side]
        for source_patch_id in sorted(state.anchor_patch_ids):
            source = anchor_source[side].get(source_patch_id)
            if source is None:
                raise ValueError("frozen topology references an absent anchor plane")
            cell = _offset_cell(source.cell_xyz, offset)
            patch = clip_plane_to_cell(
                combined_grid,
                cell,
                source.estimate,
                patch_id=next_patch_id,
            )
            if patch is None:
                raise ValueError("rebased anchor plane no longer intersects its cell")
            patches.append(patch)
            references[next_patch_id] = _PatchReference(
                side,
                source_patch_id,
                None,
                int(anchor_tables[side].local_order[
                    int(np.flatnonzero(anchor_tables[side].patch_id == source_patch_id)[0])
                ]),
                True,
                True,
            )
            anchor_temporary[(side, source_patch_id)] = next_patch_id
            next_patch_id += 1

    selected_rows = tuple(_rows_by_cell(value.patches) for value in boundaries)
    mutable_patch_ids: set[int] = set()
    for option in selection.selected_options:
        if option.cell_xyz not in mutable_cells:
            continue
        reference = configuration_references[option.source_configuration_index]
        side = reference.side
        world_cell = _offset_cell(option.cell_xyz, slab_start)
        local_cell = tuple(
            world_cell[axis] - adjacency.offsets[side][axis]
            for axis in range(3)
        )
        source_rows = selected_rows[side].get(local_cell, [])
        if reference.current and len(source_rows) != len(option.patches):
            raise ValueError("current seam configuration disagrees with selected patches")
        for layer_index, source_patch in enumerate(option.patches):
            source_patch_id = (
                int(boundaries[side].patches.patch_id[source_rows[layer_index]])
                if reference.current
                else None
            )
            patch = clip_plane_to_cell(
                combined_grid,
                world_cell,
                source_patch.estimate,
                patch_id=next_patch_id,
            )
            if patch is None:
                raise ValueError("selected seam plane no longer intersects its cell")
            patches.append(patch)
            references[next_patch_id] = _PatchReference(
                side,
                source_patch_id,
                reference.source_configuration_index,
                layer_index,
                False,
                reference.current,
            )
            mutable_patch_ids.add(next_patch_id)
            next_patch_id += 1
    return tuple(patches), references, anchor_temporary, mutable_patch_ids


def _topology_seed(
    frozen_states: tuple[Any, Any],
    anchor_temporary: Mapping[tuple[int, int], int],
    adjacency: Any,
) -> FrozenTopologySeed:
    components: list[ComponentSeed] = []
    crossings: list[CrossingSeed] = []
    detached = 0
    for side, state in enumerate(frozen_states):
        offset = adjacency.offsets[side]
        detached += int(state.detached_component_count)
        for component in state.components:
            components.append(
                ComponentSeed(
                    (side, component.component_id),
                    component.total_patch_count,
                    tuple(
                        _offset_cell(value, offset)
                        for value in component.occupied_cells
                    ),
                    tuple(
                        anchor_temporary[(side, value)]
                        for value in component.anchor_patch_ids
                    ),
                    component.anchor_orientation_parity,
                )
            )
        for crossing in state.crossings:
            crossings.append(
                CrossingSeed(
                    (side, crossing.group_id),
                    _offset_feature(crossing.feature, offset),
                    tuple(
                        (
                            anchor_temporary[(side, patch_id)],
                            _offset_edge(edge, offset),
                        )
                        for patch_id, edge in crossing.observations
                    ),
                    tuple(
                        (
                            (side, patch_id),
                            _offset_edge(edge, offset),
                        )
                        for patch_id, edge in crossing.owners
                    ),
                )
            )
    return FrozenTopologySeed(tuple(components), tuple(crossings), detached)


def _parallel_matching_settings(policy: Mapping[str, Any]) -> TraceMatchSettings:
    values = policy["parallelMatching"]
    return TraceMatchSettings(
        orthogonal_fiber_equivalence=bool(values["orthogonalFiberEquivalence"]),
        maximum_absolute_normal_angle_radians=math.radians(
            float(values["maximumNormalAngleDegrees"])
        ),
        maximum_absolute_fiber_residual_radians=math.radians(
            float(values["maximumFiberResidualDegrees"])
        ),
    )


def _join_key(value: TraceMatch) -> JoinKey:
    return (
        value.first_patch_id,
        value.second_patch_id,
        value.face.axis,
        value.face.anchor_xyz,
    )


def _joint_topology(
    patches: tuple[ClippedPatch, ...],
    mutable_patch_ids: set[int],
    seed: FrozenTopologySeed,
    adjacency: Any,
    slab_start: Int3,
    slab_grid: GridSpec,
    policy: Mapping[str, Any],
) -> tuple[TopologySelection, int, int]:
    stop = tuple(
        slab_start[axis] + slab_grid.shape_cells_xyz[axis]
        for axis in range(3)
    )
    bounds = BlockBounds(slab_start, stop)
    strict_block = assemble_surface_block(
        adjacency.combined_grid,
        bounds,
        patches,
        _parallel_matching_settings(policy),
    )
    strict_candidates = tuple(
        value
        for value in strict_block.candidate_joins
        if value.first_patch_id in mutable_patch_ids
        or value.second_patch_id in mutable_patch_ids
    )
    strict = select_joins_with_frozen_topology(
        patches,
        strict_candidates,
        seed,
    )
    quarter_policy = policy["quarterTurnAdmission"]
    if not bool(quarter_policy["enabled"]):
        return strict, len(strict_candidates), 0
    proposal_block = assemble_surface_block(
        adjacency.combined_grid,
        bounds,
        patches,
        TraceMatchSettings(orthogonal_fiber_equivalence=True),
    )
    strict_candidate_keys = {_join_key(value) for value in strict_candidates}
    quarter_candidates = tuple(
        value
        for value in proposal_block.candidate_joins
        if (
            value.first_patch_id in mutable_patch_ids
            or value.second_patch_id in mutable_patch_ids
        )
        and value.fiber_quarter_turn is True
        and _join_key(value) not in strict_candidate_keys
        and math.degrees(value.normal_angle_radians)
        <= float(quarter_policy["maximumNormalAngleDegrees"])
        and value.fiber_angle_radians is not None
        and math.degrees(value.fiber_angle_radians)
        <= float(quarter_policy["maximumFiberFrameResidualDegrees"])
    )
    fixed = frozenset(_join_key(value) for value in strict.joins)
    packets = select_joins_with_frozen_topology(
        patches,
        (*strict.joins, *quarter_candidates),
        seed,
        fixed_join_keys=fixed,
    )
    return packets, len(strict_candidates), len(quarter_candidates)


def _identity(
    boundaries: tuple[_BoundaryInput, _BoundaryInput],
    adjacency: Any,
    settings: BoundaryReselectionSettings,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": BOUNDARY_RESELECTION_SCHEMA,
        "version": BOUNDARY_RESELECTION_VERSION,
        "inputs": [
            {
                "root": str(value.root),
                "manifestSha256": sha256_file(value.root / "boundary-band-v1.json"),
                "boundaryIdentitySha256": value.manifest["identity"][
                    "identitySha256"
                ],
            }
            for value in boundaries
        ],
        "adjacency": {
            "axis": adjacency.axis,
            "lowerInput": adjacency.lower_side,
            "upperInput": adjacency.upper_side,
            "offsetsCellsXYZ": [list(value) for value in adjacency.offsets],
        },
        "settings": settings.record(),
        "seamMatchingPolicy": boundaries[0].manifest["seamMatchingPolicy"],
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "boundary_reselection.py",
                "boundary_topology.py",
                "boundary_band.py",
                "selection.py",
                "matching.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _write_reselection_artifacts(
    output: Path,
    identity_sha256: str,
    adjacency: Any,
    boundaries: tuple[_BoundaryInput, _BoundaryInput],
    slab_start: Int3,
    slab_grid: GridSpec,
    selection: ConfigurationSelection,
    configuration_references: Mapping[int, _ConfigurationReference],
    mutable_cells: set[Int3],
    patches: tuple[ClippedPatch, ...],
    patch_references: Mapping[int, _PatchReference],
    topology: TopologySelection,
) -> tuple[dict[str, Any], Path]:
    configuration_by_cell = {
        option.cell_xyz: configuration_references[option.source_configuration_index]
        for option in selection.selected_options
        if option.cell_xyz in mutable_cells
    }
    prior_by_cell: dict[tuple[int, Int3], int] = {}
    for side, boundary in enumerate(boundaries):
        configuration = _load_configurations(boundary)
        for local_cell, selected_index in configuration.selected_index.items():
            prior_by_cell[(side, local_cell)] = int(
                configuration.arrays["sourceConfigurationIndex"][selected_index]
            )
    selected_records: list[tuple[Any, ...]] = []
    for slab_cell, reference in sorted(configuration_by_cell.items()):
        world_cell = _offset_cell(slab_cell, slab_start)
        local_cell = tuple(
            world_cell[axis] - adjacency.offsets[reference.side][axis]
            for axis in range(3)
        )
        prior = prior_by_cell[(reference.side, local_cell)]
        selected_records.append(
            (
                reference.side,
                local_cell,
                world_cell,
                prior,
                reference.source_configuration_index,
                int(prior != reference.source_configuration_index),
            )
        )

    joins = tuple(topology.joins)
    artifact_path = output / "boundary-reselection-v1.npz"
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            selectedCellInput=np.asarray(
                [value[0] for value in selected_records], dtype=np.int8
            ),
            selectedCellLocalXYZ=np.asarray(
                [value[1] for value in selected_records], dtype=np.int32
            ).reshape(len(selected_records), 3),
            selectedCellCombinedXYZ=np.asarray(
                [value[2] for value in selected_records], dtype=np.int32
            ).reshape(len(selected_records), 3),
            priorSourceConfigurationIndex=np.asarray(
                [value[3] for value in selected_records], dtype=np.int64
            ),
            selectedSourceConfigurationIndex=np.asarray(
                [value[4] for value in selected_records], dtype=np.int64
            ),
            selectedConfigurationChanged=np.asarray(
                [value[5] for value in selected_records], dtype=np.uint8
            ),
            patchId=np.asarray([value.patch_id for value in patches], dtype=np.uint64),
            patchInput=np.asarray(
                [patch_references[value.patch_id].side for value in patches],
                dtype=np.int8,
            ),
            patchSourcePatchId=np.asarray(
                [
                    -1
                    if patch_references[value.patch_id].source_patch_id is None
                    else patch_references[value.patch_id].source_patch_id
                    for value in patches
                ],
                dtype=np.int64,
            ),
            patchSourceConfigurationIndex=np.asarray(
                [
                    -1
                    if patch_references[value.patch_id].source_configuration_index
                    is None
                    else patch_references[value.patch_id].source_configuration_index
                    for value in patches
                ],
                dtype=np.int64,
            ),
            patchLayerIndex=np.asarray(
                [patch_references[value.patch_id].layer_index for value in patches],
                dtype=np.int16,
            ),
            patchIsAnchor=np.asarray(
                [patch_references[value.patch_id].anchor for value in patches],
                dtype=np.uint8,
            ),
            patchUsesCurrentConfiguration=np.asarray(
                [
                    patch_references[value.patch_id].current_configuration
                    for value in patches
                ],
                dtype=np.uint8,
            ),
            joinFirstPatchId=np.asarray(
                [value.first_patch_id for value in joins], dtype=np.uint64
            ),
            joinSecondPatchId=np.asarray(
                [value.second_patch_id for value in joins], dtype=np.uint64
            ),
            joinFirstInput=np.asarray(
                [patch_references[value.first_patch_id].side for value in joins],
                dtype=np.int8,
            ),
            joinSecondInput=np.asarray(
                [patch_references[value.second_patch_id].side for value in joins],
                dtype=np.int8,
            ),
            joinFirstSourcePatchId=np.asarray(
                [
                    -1
                    if patch_references[value.first_patch_id].source_patch_id is None
                    else patch_references[value.first_patch_id].source_patch_id
                    for value in joins
                ],
                dtype=np.int64,
            ),
            joinSecondSourcePatchId=np.asarray(
                [
                    -1
                    if patch_references[value.second_patch_id].source_patch_id is None
                    else patch_references[value.second_patch_id].source_patch_id
                    for value in joins
                ],
                dtype=np.int64,
            ),
            joinFaceAxis=np.asarray([value.face.axis for value in joins], dtype=np.int8),
            joinFaceAnchorXYZ=np.asarray(
                [value.face.anchor_xyz for value in joins], dtype=np.int32
            ).reshape(len(joins), 3),
            joinScore=np.asarray([value.score for value in joins], dtype=np.float32),
            joinNegativeLogLikelihood=np.asarray(
                [value.negative_log_likelihood for value in joins], dtype=np.float32
            ),
            joinMaximumEndpointZ=np.asarray(
                [
                    max(
                        (endpoint.z for endpoint in value.endpoint_agreements),
                        default=np.nan,
                    )
                    for value in joins
                ],
                dtype=np.float32,
            ),
            joinNormalResidualDegrees=np.asarray(
                [math.degrees(value.normal_angle_radians) for value in joins],
                dtype=np.float32,
            ),
            joinFiberFrameResidualDegrees=np.asarray(
                [
                    np.nan
                    if value.fiber_angle_radians is None
                    else math.degrees(value.fiber_angle_radians)
                    for value in joins
                ],
                dtype=np.float32,
            ),
            joinFiberQuarterTurn=np.asarray(
                [
                    -1
                    if value.fiber_quarter_turn is None
                    else int(value.fiber_quarter_turn)
                    for value in joins
                ],
                dtype=np.int8,
            ),
            componentPatchId=np.asarray(
                [value[0] for value in topology.component_by_patch], dtype=np.uint64
            ),
            componentId=np.asarray(
                [value[1] for value in topology.component_by_patch], dtype=np.uint64
            ),
        )
    temporary.replace(artifact_path)

    configuration_id = {
        patch.patch_id: (
            0
            if patch_references[patch.patch_id].source_configuration_index is None
            else patch_references[patch.patch_id].source_configuration_index
        )
        for patch in patches
    }
    patch_manifest = write_patch_shard(
        output / "selected-band-patches-v1",
        PatchTable.from_patches(
            adjacency.combined_grid,
            patches,
            configuration_id=configuration_id,
            local_order={
                patch.patch_id: patch_references[patch.patch_id].layer_index
                for patch in patches
            },
        ),
        settings={
            "slabStartCellXYZ": list(slab_start),
            "slabShapeCellsXYZ": list(slab_grid.shape_cells_xyz),
            "interiorsImmutable": True,
        },
        provenance={
            "reselectionIdentitySha256": identity_sha256,
            "inputs": [str(value.root) for value in boundaries],
        },
        compressed=True,
    )
    return patch_manifest, artifact_path


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "maximum": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), (50, 90, 100))
    return {
        name: round(float(value), 7)
        for name, value in zip(("median", "p90", "maximum"), result)
    }


def run_boundary_band_reselection(
    first_root: str | Path,
    second_root: str | Path,
    output_root: str | Path,
    *,
    settings: BoundaryReselectionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Jointly reselect only the two serialized meeting bands and their graph."""

    started = time.monotonic()
    resolved = settings or BoundaryReselectionSettings()
    boundaries = (
        _load_boundary(first_root, 0),
        _load_boundary(second_root, 1),
    )
    if boundaries[0].root == boundaries[1].root:
        raise ValueError("boundary reselection requires two distinct inputs")
    if boundaries[0].manifest["graphSemantics"] != boundaries[1].manifest["graphSemantics"]:
        raise ValueError("boundary inputs use different graph semantics")
    if boundaries[0].manifest["seamMatchingPolicy"] != boundaries[1].manifest["seamMatchingPolicy"]:
        raise ValueError("boundary inputs use different seam matching policies")
    depths = tuple(
        int(value.manifest["settings"]["depth_cells"]) for value in boundaries
    )
    if depths[0] != depths[1]:
        raise ValueError("joint reselection requires equal boundary depths")
    depth = depths[0]
    adjacency = _adjacency(*boundaries)
    facing_sides = [0, 0]
    facing_sides[adjacency.lower_side] = 1
    facing_sides[adjacency.upper_side] = 0
    facing = tuple(facing_sides)
    output = Path(output_root).resolve()
    if output in {value.root for value in boundaries}:
        raise ValueError("boundary reselection output must differ from its inputs")
    identity = _identity(boundaries, adjacency, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "boundary-reselection-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("boundary reselection output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": BOUNDARY_RESELECTION_SCHEMA,
        "version": BOUNDARY_RESELECTION_VERSION,
        "state": "building",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    loading_finished = time.monotonic()
    configurations = tuple(_load_configurations(value) for value in boundaries)
    anchor_tables = tuple(_anchor_table(value) for value in boundaries)
    frozen_states = tuple(
        _frozen_state(value, adjacency.axis, facing[value.side])
        for value in boundaries
    )
    slab_grid, slab_start = _slab_grid(
        adjacency,
        boundaries[adjacency.lower_side],
        depth,
    )
    combined_table, configuration_references, initial, mutable_cells = (
        _combined_configuration_table(
            boundaries,
            configurations,
            anchor_tables,
            adjacency,
            slab_grid,
            slab_start,
            facing,
            depth,
        )
    )
    selection = optimize_configurations(
        slab_grid,
        (combined_table,),
        unary_scale=resolved.unary_scale,
        pairwise_scale=resolved.pairwise_scale,
        interior_unmatched_trace_penalty=resolved.interior_unmatched_trace_penalty,
        maximum_sweeps=resolved.maximum_sweeps,
        initial_configuration_indices=initial,
        mutable_cells=mutable_cells,
    )
    selection_finished = time.monotonic()
    patches, patch_references, anchor_temporary, mutable_patch_ids = (
        _realize_selected_patches(
            selection,
            configuration_references,
            mutable_cells,
            slab_start,
            adjacency,
            boundaries,
            anchor_tables,
            frozen_states,
        )
    )
    seed = _topology_seed(frozen_states, anchor_temporary, adjacency)
    topology, strict_candidates, quarter_candidates = _joint_topology(
        patches,
        mutable_patch_ids,
        seed,
        adjacency,
        slab_start,
        slab_grid,
        boundaries[0].manifest["seamMatchingPolicy"],
    )
    topology_finished = time.monotonic()
    patch_manifest, artifact_path = _write_reselection_artifacts(
        output,
        identity_sha256,
        adjacency,
        boundaries,
        slab_start,
        slab_grid,
        selection,
        configuration_references,
        mutable_cells,
        patches,
        patch_references,
        topology,
    )
    changed_cells = sum(
        not configuration_references[option.source_configuration_index].current
        for option in selection.selected_options
        if option.cell_xyz in mutable_cells
    )
    deferred = Counter(value.reason for value in topology.deferred_joins)
    joins = topology.joins
    finished = time.monotonic()
    manifest.update(
        {
            "state": "complete",
            "graphSemantics": boundaries[0].manifest["graphSemantics"],
            "method": {
                "interiorsImmutable": True,
                "mutableScope": (
                    "exactly the serialized depth-cell band on each meeting face"
                ),
                "configurationSelection": (
                    "warm-started conditional ICM with one immutable anchor layer"
                ),
                "topologySelection": (
                    "strict joins first; accepted strict graph then fixed while "
                    "gated quarter-turn packet joins are admitted"
                ),
                "frozenState": (
                    "complete component occupancy, cut-anchor orientation parity, "
                    "and welded crossing feature certificates"
                ),
            },
            "grid": {
                "shapeCellsXYZ": list(adjacency.combined_grid.shape_cells_xyz),
                "cellSizeXYZ": list(adjacency.combined_grid.cell_size_xyz),
                "originXYZ": list(adjacency.combined_grid.origin_xyz),
                "coordinateUnit": adjacency.combined_grid.coordinate_unit,
            },
            "adjacency": identity["adjacency"],
            "slab": {
                "startCellXYZ": list(slab_start),
                "shapeCellsXYZ": list(slab_grid.shape_cells_xyz),
                "depthCellsPerInput": depth,
                "anchorLayersPerInput": 1,
            },
            "settings": resolved.record(),
            "statistics": {
                "mutableCells": len(mutable_cells),
                "changedConfigurations": changed_cells,
                "unchangedConfigurations": len(mutable_cells) - changed_cells,
                "selectedMutablePatches": len(mutable_patch_ids),
                "anchorPatches": len(patches) - len(mutable_patch_ids),
                "strictCandidateJoins": strict_candidates,
                "quarterTurnCandidateJoins": quarter_candidates,
                "retainedBandJoins": len(joins),
                "retainedQuarterTurnJoins": sum(
                    value.fiber_quarter_turn is True for value in joins
                ),
                "deferredBandJoins": len(topology.deferred_joins),
                "deferredByReason": dict(sorted(deferred.items())),
                "frozenDetachedComponents": topology.detached_component_count,
                "recomposedComponents": topology.component_count,
                "joinResiduals": {
                    "endpointZ": _quantiles(
                        [
                            max(
                                endpoint.z for endpoint in value.endpoint_agreements
                            )
                            for value in joins
                        ]
                    ),
                    "normalDegrees": _quantiles(
                        [math.degrees(value.normal_angle_radians) for value in joins]
                    ),
                    "fiberFrameDegrees": _quantiles(
                        [
                            math.degrees(value.fiber_angle_radians)
                            for value in joins
                            if value.fiber_angle_radians is not None
                        ]
                    ),
                },
            },
            "timingSeconds": {
                "loading": round(loading_finished - started, 6),
                "configurationSelection": round(
                    selection_finished - loading_finished, 6
                ),
                "topologySelection": round(
                    topology_finished - selection_finished, 6
                ),
                "exports": round(finished - topology_finished, 6),
                "total": round(finished - started, 6),
            },
            "artifacts": {
                "reselection": {
                    "path": artifact_path.name,
                    "bytes": artifact_path.stat().st_size,
                    "sha256": sha256_file(artifact_path),
                },
                "selectedBandPatches": {
                    "manifest": "selected-band-patches-v1.json",
                    "data": "selected-band-patches-v1.npz",
                    "sha256": patch_manifest["data"]["sha256"],
                },
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
