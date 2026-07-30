from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from itertools import product
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .boundary_merge import _BoundaryInput, _load_boundary
from .boundary_reselection import (
    BoundaryReselectionSettings,
    _ConfigurationReference,
    _anchor_rows_by_cell,
    _anchor_table,
    _append_layers,
    _joint_topology,
    _load_configurations,
    _quantiles,
    _rows_by_cell,
)
from .boundary_topology import (
    ComponentSeed,
    CrossingSeed,
    FrozenRegionState,
    FrozenTopologySeed,
    TopologySelection,
    face_mask,
    read_frozen_region_state,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch, clip_plane_to_cell, plane_basis
from .selection import ConfigurationSelection, optimize_configurations
from .stratigraphy import ConfigurationTable
from .tables import PatchTable, write_patch_shard
from .topology import GridEdge, GridSpec, Int3


CLUSTER_RESELECTION_SCHEMA = "pareidolia.cubical-boundary-cluster-reselection"
CLUSTER_RESELECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ClusterLayout:
    grid: GridSpec
    offsets: tuple[Int3, ...]
    internal_faces: tuple[tuple[tuple[int, int], ...], ...]
    face_masks: tuple[int, ...]


@dataclass(frozen=True, slots=True)
class _ClusterPatchReference:
    block: int
    source_patch_id: int | None
    source_configuration_index: int | None
    layer_index: int
    anchor: bool
    current_configuration: bool


@dataclass(frozen=True, slots=True)
class _CombinedConfigurations:
    table: ConfigurationTable
    references: Mapping[int, _ConfigurationReference]
    initial: Mapping[Int3, tuple[int, int]]
    active_cells: frozenset[Int3]
    mutable_cells: frozenset[Int3]
    mutable_owner: Mapping[Int3, tuple[int, Int3]]


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


def _cluster_layout(boundaries: tuple[_BoundaryInput, ...]) -> _ClusterLayout:
    if len(boundaries) < 2 or len(boundaries) > 8:
        raise ValueError("a boundary cluster requires two to eight child blocks")
    grids = tuple(value.patches.grid for value in boundaries)
    if len({value.coordinate_unit for value in grids}) != 1:
        raise ValueError("cluster boundary grids use different coordinate units")
    cell_size = np.asarray(grids[0].cell_size_xyz, dtype=np.float64)
    if any(
        not np.allclose(
            np.asarray(value.cell_size_xyz, dtype=np.float64),
            cell_size,
            rtol=0.0,
            atol=1.0e-9,
        )
        for value in grids[1:]
    ):
        raise ValueError("cluster boundary grids use different cell sizes")
    global_origin = np.min(
        np.asarray([value.origin_xyz for value in grids], dtype=np.float64),
        axis=0,
    )
    offsets: list[Int3] = []
    for grid in grids:
        raw = (
            np.asarray(grid.origin_xyz, dtype=np.float64) - global_origin
        ) / cell_size
        rounded = np.rint(raw).astype(np.int64)
        if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-7):
            raise ValueError("cluster block origin is not cell aligned")
        offsets.append(tuple(int(value) for value in rounded))
    stops = tuple(
        tuple(
            offsets[index][axis] + grid.shape_cells_xyz[axis]
            for axis in range(3)
        )
        for index, grid in enumerate(grids)
    )
    global_shape: Int3 = tuple(
        max(value[axis] for value in stops) for axis in range(3)
    )  # type: ignore[assignment]
    child_shapes = {value.shape_cells_xyz for value in grids}
    if len(child_shapes) != 1:
        raise ValueError("a 2x2x2 cluster requires equal child-block shapes")
    child_shape = next(iter(child_shapes))
    positions = tuple(
        tuple(sorted({value[axis] for value in offsets})) for axis in range(3)
    )
    if any(
        len(values) > 2
        or values not in ((0,), (0, child_shape[axis]))
        for axis, values in enumerate(positions)
    ):
        raise ValueError("cluster children must occupy a regular 2x2x2 lattice")
    expected_offsets = set(product(*positions))
    if set(offsets) != expected_offsets:
        raise ValueError("cluster child positions do not form a Cartesian cuboid")
    for first in range(len(boundaries)):
        for second in range(first + 1, len(boundaries)):
            if all(
                max(offsets[first][axis], offsets[second][axis])
                < min(stops[first][axis], stops[second][axis])
                for axis in range(3)
            ):
                raise ValueError("cluster child blocks overlap")
    owned_volume = sum(math.prod(value.shape_cells_xyz) for value in grids)
    if owned_volume != math.prod(global_shape):
        raise ValueError("cluster child blocks do not fill one rectangular cuboid")

    internal_faces: list[tuple[tuple[int, int], ...]] = []
    masks: list[int] = []
    for index, grid in enumerate(grids):
        faces: list[tuple[int, int]] = []
        for axis in range(3):
            low_internal = offsets[index][axis] > 0
            high_internal = stops[index][axis] < global_shape[axis]
            if low_internal and high_internal:
                raise ValueError(
                    "a 2x2x2 child cannot have both sides of one axis internal"
                )
            if low_internal:
                faces.append((axis, 0))
            if high_internal:
                faces.append((axis, 1))
        if not faces:
            raise ValueError("every child must participate in the cluster boundary")
        internal_faces.append(tuple(faces))
        masks.append(face_mask(faces))
    return _ClusterLayout(
        GridSpec(
            global_shape,
            grids[0].cell_size_xyz,
            tuple(float(value) for value in global_origin),
            grids[0].coordinate_unit,
        ),
        tuple(offsets),
        tuple(internal_faces),
        tuple(masks),
    )


def _is_mutable(
    cell: Int3,
    shape: Int3,
    faces: tuple[tuple[int, int], ...],
    depth: int,
) -> bool:
    return any(
        cell[axis] < depth
        if side == 0
        else cell[axis] >= shape[axis] - depth
        for axis, side in faces
    )


def _is_anchor_shell(
    cell: Int3,
    shape: Int3,
    faces: tuple[tuple[int, int], ...],
    depth: int,
) -> bool:
    if _is_mutable(cell, shape, faces, depth):
        return False
    return any(
        cell[axis] == (depth if side == 0 else shape[axis] - depth - 1)
        for axis, side in faces
    )


def _combined_configuration_table(
    boundaries: tuple[_BoundaryInput, ...],
    layout: _ClusterLayout,
    depth: int,
) -> _CombinedConfigurations:
    configurations = tuple(_load_configurations(value) for value in boundaries)
    anchor_tables = tuple(_anchor_table(value) for value in boundaries)
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
    active_cells: set[Int3] = set()
    mutable_cells: set[Int3] = set()
    mutable_owner: dict[Int3, tuple[int, Int3]] = {}

    records: list[tuple[Int3, int, Int3, bool]] = []
    for block, boundary in enumerate(boundaries):
        shape = boundary.patches.grid.shape_cells_xyz
        faces = layout.internal_faces[block]
        for z in range(shape[2]):
            for y in range(shape[1]):
                for x in range(shape[0]):
                    local_cell = (x, y, z)
                    mutable = _is_mutable(local_cell, shape, faces, depth)
                    if not mutable and not _is_anchor_shell(
                        local_cell, shape, faces, depth
                    ):
                        continue
                    combined_cell = _offset_cell(local_cell, layout.offsets[block])
                    records.append((combined_cell, block, local_cell, mutable))
    records.sort(key=lambda value: (value[0][2], value[0][1], value[0][0]))
    if len({value[0] for value in records}) != len(records):
        raise ValueError("cluster active cells have multiple owners")

    for combined_cell, block, local_cell, mutable in records:
        cell_xyz.append(combined_cell)
        active_cells.add(combined_cell)
        if mutable:
            mutable_cells.add(combined_cell)
            mutable_owner[combined_cell] = (block, local_cell)
            source = configurations[block]
            if local_cell not in source.cell_index:
                raise ValueError("boundary candidates omit a mutable cluster cell")
            cell_index = source.cell_index[local_cell]
            selected_index = source.selected_index[local_cell]
            selected_output: int | None = None
            for source_index in source.table.configurations_for_cell(cell_index):
                output_index = len(configuration_id)
                current = source_index == selected_index
                references[output_index] = _ConfigurationReference(
                    block,
                    int(source.arrays["sourceConfigurationIndex"][source_index]),
                    source_index,
                    current,
                )
                configuration_id.append(
                    int(source.table.configuration_id[source_index])
                )
                configuration_log_weight.append(
                    float(source.table.configuration_log_weight[source_index])
                )
                normal_hypothesis.append(
                    int(source.table.normal_hypothesis[source_index])
                )
                _append_layers(layers, source.table, source_index)
                layer_offset.append(len(layers["height"]))
                if current:
                    selected_output = output_index
            if selected_output is None:
                raise ValueError("current cluster configuration is absent")
            initial[combined_cell] = (0, selected_output)
        else:
            rows = anchor_rows[block].get(local_cell, [])
            output_index = len(configuration_id)
            references[output_index] = _ConfigurationReference(
                block, -1, -1, True
            )
            configuration_id.append(0)
            configuration_log_weight.append(0.0)
            normal_hypothesis.append(
                int(anchor_tables[block].normal_family[rows[0]]) if rows else 0
            )
            for row in rows:
                normal = np.asarray(
                    anchor_tables[block].normal_xyz[row], dtype=np.float32
                )
                fiber = np.asarray(
                    anchor_tables[block].fiber_xyz[row], dtype=np.float32
                )
                fiber_std = float(
                    anchor_tables[block].fiber_angular_std_radians[row]
                )
                if not np.all(np.isfinite(fiber)):
                    fiber = np.asarray(plane_basis(normal)[0], dtype=np.float32)
                    fiber_std = math.pi
                layers["normal"].append(normal)
                layers["height"].append(float(anchor_tables[block].height[row]))
                layers["covariance"].append(
                    anchor_tables[block].plane_covariance[row]
                )
                layers["fiber"].append(fiber)
                layers["fiber_std"].append(fiber_std)
                layers["confidence"].append(
                    float(anchor_tables[block].confidence[row])
                )
                layers["evidence"].append(1.0)
                layers["material"].append(1.0)
                layers["support"].append(1.0)
            layer_offset.append(len(layers["height"]))
            initial[combined_cell] = (0, output_index)
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
    return _CombinedConfigurations(
        table,
        references,
        initial,
        frozenset(active_cells),
        frozenset(mutable_cells),
        mutable_owner,
    )


def _frozen_region(
    boundary: _BoundaryInput,
    region_face_mask: int,
) -> FrozenRegionState:
    record = boundary.manifest["artifacts"].get("frozenRegions")
    if not isinstance(record, dict):
        raise ValueError("boundary artifact lacks multi-face frozen regions")
    path = boundary.root / str(record["path"])
    if sha256_file(path) != record["sha256"]:
        raise ValueError("boundary frozen-region content hash mismatch")
    return read_frozen_region_state(path, region_face_mask)


def _realize_selected_patches(
    selection: ConfigurationSelection,
    combined: _CombinedConfigurations,
    boundaries: tuple[_BoundaryInput, ...],
    layout: _ClusterLayout,
    anchor_tables: tuple[PatchTable, ...],
    frozen_regions: tuple[FrozenRegionState, ...],
) -> tuple[
    tuple[ClippedPatch, ...],
    dict[int, _ClusterPatchReference],
    dict[tuple[int, int], int],
    set[int],
]:
    patches: list[ClippedPatch] = []
    references: dict[int, _ClusterPatchReference] = {}
    anchor_temporary: dict[tuple[int, int], int] = {}
    next_patch_id = 1
    anchor_sources = tuple(
        {patch.patch_id: patch for patch in table.to_patches()}
        for table in anchor_tables
    )
    anchor_rows = tuple(
        {int(patch_id): row for row, patch_id in enumerate(table.patch_id)}
        for table in anchor_tables
    )
    for block, state in enumerate(frozen_regions):
        offset = layout.offsets[block]
        for source_patch_id in sorted(state.anchor_patch_ids):
            source = anchor_sources[block].get(source_patch_id)
            if source is None:
                raise ValueError("frozen region references an absent anchor plane")
            cell = _offset_cell(source.cell_xyz, offset)
            patch = clip_plane_to_cell(
                layout.grid,
                cell,
                source.estimate,
                patch_id=next_patch_id,
            )
            if patch is None:
                raise ValueError("rebased cluster anchor no longer intersects its cell")
            row = anchor_rows[block][source_patch_id]
            patches.append(patch)
            references[next_patch_id] = _ClusterPatchReference(
                block,
                source_patch_id,
                None,
                int(anchor_tables[block].local_order[row]),
                True,
                True,
            )
            anchor_temporary[(block, source_patch_id)] = next_patch_id
            next_patch_id += 1

    selected_rows = tuple(_rows_by_cell(value.patches) for value in boundaries)
    mutable_patch_ids: set[int] = set()
    for option in selection.selected_options:
        if option.cell_xyz not in combined.mutable_cells:
            continue
        reference = combined.references[option.source_configuration_index]
        block, local_cell = combined.mutable_owner[option.cell_xyz]
        if reference.side != block:
            raise ValueError("selected cluster configuration changed block ownership")
        source_rows = selected_rows[block].get(local_cell, [])
        if reference.current and len(source_rows) != len(option.patches):
            raise ValueError("current cluster configuration disagrees with patches")
        for layer_index, source_patch in enumerate(option.patches):
            source_patch_id = (
                int(boundaries[block].patches.patch_id[source_rows[layer_index]])
                if reference.current
                else None
            )
            patch = clip_plane_to_cell(
                layout.grid,
                option.cell_xyz,
                source_patch.estimate,
                patch_id=next_patch_id,
            )
            if patch is None:
                raise ValueError("selected cluster plane no longer intersects its cell")
            patches.append(patch)
            references[next_patch_id] = _ClusterPatchReference(
                block,
                source_patch_id,
                reference.source_configuration_index,
                layer_index,
                False,
                reference.current,
            )
            mutable_patch_ids.add(next_patch_id)
            next_patch_id += 1
    return patches, references, anchor_temporary, mutable_patch_ids


def _topology_seed(
    regions: tuple[FrozenRegionState, ...],
    anchor_temporary: Mapping[tuple[int, int], int],
    layout: _ClusterLayout,
) -> FrozenTopologySeed:
    components: list[ComponentSeed] = []
    crossings: list[CrossingSeed] = []
    detached = 0
    for block, state in enumerate(regions):
        offset = layout.offsets[block]
        detached += state.detached_component_count
        for component in state.components:
            components.append(
                ComponentSeed(
                    (block, component.component_id),
                    component.total_patch_count,
                    tuple(
                        _offset_cell(value, offset)
                        for value in component.occupied_cells
                    ),
                    tuple(
                        anchor_temporary[(block, value)]
                        for value in component.anchor_patch_ids
                    ),
                    component.anchor_orientation_parity,
                )
            )
        for crossing in state.crossings:
            crossings.append(
                CrossingSeed(
                    (block, crossing.group_id),
                    _offset_feature(crossing.feature, offset),
                    tuple(
                        (
                            anchor_temporary[(block, patch_id)],
                            _offset_edge(edge, offset),
                        )
                        for patch_id, edge in crossing.observations
                    ),
                    tuple(
                        ((block, patch_id), _offset_edge(edge, offset))
                        for patch_id, edge in crossing.owners
                    ),
                )
            )
    return FrozenTopologySeed(tuple(components), tuple(crossings), detached)


def _identity(
    boundaries: tuple[_BoundaryInput, ...],
    layout: _ClusterLayout,
    settings: BoundaryReselectionSettings,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": CLUSTER_RESELECTION_SCHEMA,
        "version": CLUSTER_RESELECTION_VERSION,
        "inputs": [
            {
                "root": str(value.root),
                "manifestSha256": sha256_file(
                    value.root / "boundary-band-v1.json"
                ),
                "boundaryIdentitySha256": value.manifest["identity"][
                    "identitySha256"
                ],
                "offsetCellsXYZ": list(layout.offsets[index]),
                "internalFaces": [
                    list(face) for face in layout.internal_faces[index]
                ],
                "frozenRegionFaceMask": layout.face_masks[index],
            }
            for index, value in enumerate(boundaries)
        ],
        "grid": {
            "shapeCellsXYZ": list(layout.grid.shape_cells_xyz),
            "cellSizeXYZ": list(layout.grid.cell_size_xyz),
            "originXYZ": list(layout.grid.origin_xyz),
            "coordinateUnit": layout.grid.coordinate_unit,
        },
        "settings": settings.record(),
        "seamMatchingPolicy": boundaries[0].manifest["seamMatchingPolicy"],
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "cluster_reselection.py",
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


def _write_artifacts(
    output: Path,
    identity_sha256: str,
    boundaries: tuple[_BoundaryInput, ...],
    layout: _ClusterLayout,
    selection: ConfigurationSelection,
    combined: _CombinedConfigurations,
    patches: tuple[ClippedPatch, ...],
    patch_references: Mapping[int, _ClusterPatchReference],
    topology: TopologySelection,
) -> tuple[dict[str, Any], Path]:
    prior_configurations = tuple(_load_configurations(value) for value in boundaries)
    selected_records: list[tuple[Any, ...]] = []
    for option in selection.selected_options:
        if option.cell_xyz not in combined.mutable_cells:
            continue
        reference = combined.references[option.source_configuration_index]
        block, local_cell = combined.mutable_owner[option.cell_xyz]
        source = prior_configurations[block]
        prior_index = source.selected_index[local_cell]
        prior = int(source.arrays["sourceConfigurationIndex"][prior_index])
        selected_records.append(
            (
                block,
                local_cell,
                option.cell_xyz,
                prior,
                reference.source_configuration_index,
                int(prior != reference.source_configuration_index),
            )
        )

    joins = tuple(topology.joins)
    artifact_path = output / "cluster-reselection-v1.npz"
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
            patchId=np.asarray(
                [value.patch_id for value in patches], dtype=np.uint64
            ),
            patchInput=np.asarray(
                [patch_references[value.patch_id].block for value in patches],
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
                [
                    patch_references[value.first_patch_id].block
                    for value in joins
                ],
                dtype=np.int8,
            ),
            joinSecondInput=np.asarray(
                [
                    patch_references[value.second_patch_id].block
                    for value in joins
                ],
                dtype=np.int8,
            ),
            joinFaceAxis=np.asarray(
                [value.face.axis for value in joins], dtype=np.int8
            ),
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
                [value[0] for value in topology.component_by_patch],
                dtype=np.uint64,
            ),
            componentId=np.asarray(
                [value[1] for value in topology.component_by_patch],
                dtype=np.uint64,
            ),
        )
    temporary.replace(artifact_path)

    patch_manifest = write_patch_shard(
        output / "selected-cluster-patches-v1",
        PatchTable.from_patches(
            layout.grid,
            patches,
            configuration_id={
                patch.patch_id: (
                    0
                    if patch_references[
                        patch.patch_id
                    ].source_configuration_index
                    is None
                    else int(
                        patch_references[
                            patch.patch_id
                        ].source_configuration_index
                    )
                )
                for patch in patches
            },
            local_order={
                patch.patch_id: patch_references[patch.patch_id].layer_index
                for patch in patches
            },
        ),
        settings={
            "activeCells": len(combined.active_cells),
            "mutableCells": len(combined.mutable_cells),
            "interiorsImmutable": True,
        },
        provenance={
            "clusterReselectionIdentitySha256": identity_sha256,
            "inputs": [str(value.root) for value in boundaries],
        },
        compressed=True,
    )
    return patch_manifest, artifact_path


def run_boundary_cluster_reselection(
    boundary_roots: tuple[str | Path, ...],
    output_root: str | Path,
    *,
    settings: BoundaryReselectionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Jointly solve all meeting bands in one rectangular 2x2x2 child cluster."""

    started = time.monotonic()
    resolved = settings or BoundaryReselectionSettings()
    roots = tuple(Path(value).resolve() for value in boundary_roots)
    if len(set(roots)) != len(roots):
        raise ValueError("cluster boundary inputs must be distinct")
    boundaries = tuple(_load_boundary(root, index) for index, root in enumerate(roots))
    if len({value.manifest["graphSemantics"] for value in boundaries}) != 1:
        raise ValueError("cluster boundaries use different graph semantics")
    if len(
        {
            canonical_json_hash(value.manifest["seamMatchingPolicy"])
            for value in boundaries
        }
    ) != 1:
        raise ValueError("cluster boundaries use different seam policies")
    depths = {
        int(value.manifest["settings"]["depth_cells"])
        for value in boundaries
    }
    if len(depths) != 1:
        raise ValueError("cluster boundaries use different band depths")
    depth = next(iter(depths))
    layout = _cluster_layout(boundaries)
    output = Path(output_root).resolve()
    if output in roots:
        raise ValueError("cluster output must differ from every boundary input")
    identity = _identity(boundaries, layout, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "cluster-reselection-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("cluster output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": CLUSTER_RESELECTION_SCHEMA,
        "version": CLUSTER_RESELECTION_VERSION,
        "state": "building",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    loading_finished = time.monotonic()
    anchor_tables = tuple(_anchor_table(value) for value in boundaries)
    frozen_regions = tuple(
        _frozen_region(value, layout.face_masks[index])
        for index, value in enumerate(boundaries)
    )
    combined = _combined_configuration_table(boundaries, layout, depth)
    selection = optimize_configurations(
        layout.grid,
        (combined.table,),
        unary_scale=resolved.unary_scale,
        pairwise_scale=resolved.pairwise_scale,
        interior_unmatched_trace_penalty=resolved.interior_unmatched_trace_penalty,
        maximum_sweeps=resolved.maximum_sweeps,
        initial_configuration_indices=combined.initial,
        mutable_cells=combined.mutable_cells,
        active_cells=combined.active_cells,
    )
    selection_finished = time.monotonic()
    patches, patch_references, anchor_temporary, mutable_patch_ids = (
        _realize_selected_patches(
            selection,
            combined,
            boundaries,
            layout,
            anchor_tables,
            frozen_regions,
        )
    )
    seed = _topology_seed(frozen_regions, anchor_temporary, layout)

    @dataclass(frozen=True, slots=True)
    class _TopologyLayout:
        combined_grid: GridSpec

    topology, strict_candidates, quarter_candidates = _joint_topology(
        patches,
        mutable_patch_ids,
        seed,
        _TopologyLayout(layout.grid),
        (0, 0, 0),
        layout.grid,
        boundaries[0].manifest["seamMatchingPolicy"],
    )
    topology_finished = time.monotonic()
    patch_manifest, artifact_path = _write_artifacts(
        output,
        identity_sha256,
        boundaries,
        layout,
        selection,
        combined,
        patches,
        patch_references,
        topology,
    )
    changed_cells = sum(
        not combined.references[option.source_configuration_index].current
        for option in selection.selected_options
        if option.cell_xyz in combined.mutable_cells
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
                    "union of every participating depth-cell face band, solved once"
                ),
                "configurationSelection": (
                    "warm-started sparse conditional ICM with immutable cut shells"
                ),
                "topologySelection": (
                    "one strict-then-quarter-turn solve over the entire cluster"
                ),
                "frozenState": (
                    "one region certificate per child after removing all of its "
                    "participating mutually orthogonal face bands"
                ),
            },
            "grid": identity["grid"],
            "layout": {
                "blocks": len(boundaries),
                "depthCellsPerInput": depth,
                "inputs": identity["inputs"],
            },
            "settings": resolved.record(),
            "statistics": {
                "activeCells": len(combined.active_cells),
                "mutableCells": len(combined.mutable_cells),
                "immutableAnchorShellCells": (
                    len(combined.active_cells) - len(combined.mutable_cells)
                ),
                "changedConfigurations": changed_cells,
                "unchangedConfigurations": (
                    len(combined.mutable_cells) - changed_cells
                ),
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
                                endpoint.z
                                for endpoint in value.endpoint_agreements
                            )
                            for value in joins
                        ]
                    ),
                    "normalDegrees": _quantiles(
                        [
                            math.degrees(value.normal_angle_radians)
                            for value in joins
                        ]
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
                "selectedClusterPatches": {
                    "manifest": "selected-cluster-patches-v1.json",
                    "data": "selected-cluster-patches-v1.npz",
                    "sha256": patch_manifest["data"]["sha256"],
                },
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
