from __future__ import annotations

import json
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .block import (
    BlockBounds,
    SurfaceBlock,
    assemble_surface_hierarchy,
    surface_block_from_retained_joins,
)
from .boundary_merge import _BoundaryInput, _load_boundary
from .cluster_reselection import (
    CLUSTER_RESELECTION_SCHEMA,
    _cluster_layout,
    _is_mutable,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch, clip_plane_to_cell
from .matching import EndpointAgreement, TraceMatch
from .surface_graph import (
    component_statistics,
    read_legacy_packet_graph,
    reconstruct_retained_join,
    write_surface_graph,
)
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridEdge, GridFace, Int3


CLUSTER_MATERIALIZATION_SCHEMA = (
    "pareidolia.cubical-boundary-cluster-materialization"
)
CLUSTER_MATERIALIZATION_VERSION = 1


@dataclass(frozen=True, slots=True)
class _ChildGraph:
    boundary: _BoundaryInput
    selected_root: Path
    packet_root: Path | None
    table: PatchTable
    block: SurfaceBlock
    offset: Int3
    internal_faces: tuple[tuple[int, int], ...]
    immutable_patch_ids: frozenset[int]


def _offset_cell(cell: Int3, offset: Int3) -> Int3:
    return tuple(cell[axis] + offset[axis] for axis in range(3))  # type: ignore[return-value]


def _offset_edge(edge: GridEdge, offset: Int3) -> GridEdge:
    return GridEdge(edge.axis, _offset_cell(edge.anchor_xyz, offset))


def _offset_endpoint(
    value: EndpointAgreement, offset: Int3
) -> EndpointAgreement:
    return replace(
        value,
        first_edge=_offset_edge(value.first_edge, offset),
        second_edge=_offset_edge(value.second_edge, offset),
        shared_vertex_xyz=(
            None
            if value.shared_vertex_xyz is None
            else _offset_cell(value.shared_vertex_xyz, offset)
        ),
    )


def _offset_join(
    value: TraceMatch,
    offset: Int3,
    patch_id_map: Mapping[int, int],
) -> TraceMatch:
    return replace(
        value,
        first_patch_id=patch_id_map[value.first_patch_id],
        second_patch_id=patch_id_map[value.second_patch_id],
        face=GridFace(value.face.axis, _offset_cell(value.face.anchor_xyz, offset)),
        endpoint_agreements=tuple(
            _offset_endpoint(endpoint, offset)
            for endpoint in value.endpoint_agreements
        ),
    )


def _ordered_boundaries(
    manifest: Mapping[str, Any],
    boundary_roots: Iterable[str | Path] | None,
) -> tuple[_BoundaryInput, ...]:
    expected = tuple(manifest["layout"]["inputs"])
    roots = (
        tuple(Path(value).resolve() for value in boundary_roots)
        if boundary_roots is not None
        else tuple(Path(value["root"]).resolve() for value in expected)
    )
    if len(roots) != len(expected) or len(set(roots)) != len(roots):
        raise ValueError("materialization requires one distinct boundary root per child")
    loaded = tuple(_load_boundary(root, index) for index, root in enumerate(roots))
    by_identity = {
        value.manifest["identity"]["identitySha256"]: value for value in loaded
    }
    if len(by_identity) != len(loaded):
        raise ValueError("materialization boundary identities must be unique")
    ordered: list[_BoundaryInput] = []
    for index, record in enumerate(expected):
        identity = record["boundaryIdentitySha256"]
        if identity not in by_identity:
            raise ValueError(
                f"boundary override does not provide cluster input {index}"
            )
        value = by_identity[identity]
        ordered.append(replace(value, side=index))
    return tuple(ordered)


def _child_graphs(
    boundaries: tuple[_BoundaryInput, ...],
    manifest: Mapping[str, Any],
) -> tuple[_ChildGraph, ...]:
    depth = int(manifest["layout"]["depthCellsPerInput"])
    records = tuple(manifest["layout"]["inputs"])
    result: list[_ChildGraph] = []
    for index, (boundary, record) in enumerate(zip(boundaries, records)):
        identity = boundary.manifest["identity"]
        selected = Path(identity["selectedRoot"]).resolve()
        if sha256_file(selected / "selected-patches-v1.json") != identity[
            "selectedPatchManifestSha256"
        ] or sha256_file(selected / "selected-patches-v1.npz") != identity[
            "selectedPatchDataSha256"
        ]:
            raise ValueError(f"child {index} selected-patch identity changed")
        table = read_patch_shard(selected / "selected-patches-v1", verify=True)
        packet_value = identity.get("packetRoot")
        packet = Path(packet_value).resolve() if packet_value else None
        if packet is None:
            block = assemble_surface_hierarchy(
                table.grid,
                BlockBounds((0, 0, 0), table.grid.shape_cells_xyz),
                table.to_patches(),
                maximum_leaf_shape_cells_xyz=tuple(
                    int(value)
                    for value in identity["settings"]["leaf_shape_cells_xyz"]
                ),
            )
        else:
            block = read_legacy_packet_graph(
                selected, packet, table=table, verify=True
            )
        offset = tuple(int(value) for value in record["offsetCellsXYZ"])
        faces = tuple(
            (int(value[0]), int(value[1])) for value in record["internalFaces"]
        )
        immutable = frozenset(
            value.patch_id
            for value in block.patches
            if not _is_mutable(
                value.cell_xyz, table.grid.shape_cells_xyz, faces, depth
            )
        )
        result.append(
            _ChildGraph(
                boundary,
                selected,
                packet,
                table,
                block,
                offset,
                faces,
                immutable,
            )
        )
    return tuple(result)


def _identity(
    cluster: Path,
    cluster_manifest: Mapping[str, Any],
    children: tuple[_ChildGraph, ...],
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": CLUSTER_MATERIALIZATION_SCHEMA,
        "version": CLUSTER_MATERIALIZATION_VERSION,
        "clusterRoot": str(cluster),
        "clusterManifestSha256": sha256_file(
            cluster / "cluster-reselection-v1.json"
        ),
        "clusterReselectionSha256": sha256_file(
            cluster / "cluster-reselection-v1.npz"
        ),
        "clusterPatchManifestSha256": sha256_file(
            cluster / "selected-cluster-patches-v1.json"
        ),
        "clusterPatchDataSha256": sha256_file(
            cluster / "selected-cluster-patches-v1.npz"
        ),
        "clusterIdentitySha256": cluster_manifest["identity"]["identitySha256"],
        "children": [
            {
                "boundaryRoot": str(value.boundary.root),
                "boundaryManifestSha256": sha256_file(
                    value.boundary.root / "boundary-band-v1.json"
                ),
                "selectedRoot": str(value.selected_root),
                "selectedPatchDataSha256": sha256_file(
                    value.selected_root / "selected-patches-v1.npz"
                ),
                "packetRoot": (
                    str(value.packet_root) if value.packet_root is not None else None
                ),
                "packetGraphSha256": (
                    sha256_file(value.packet_root / "packet-graph-v1.npz")
                    if value.packet_root is not None
                    else None
                ),
                "offsetCellsXYZ": list(value.offset),
                "internalFaces": [list(face) for face in value.internal_faces],
            }
            for value in children
        ],
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "cluster_materialization.py",
                "cluster_reselection.py",
                "surface_graph.py",
                "block.py",
                "matching.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _baseline_statistics(children: tuple[_ChildGraph, ...]) -> dict[str, Any]:
    child_records = [
        component_statistics(value.block, maximum_records=8) for value in children
    ]
    sizes = sorted(
        (
            len(component.patch_ids)
            for value in children
            for component in value.block.components
        ),
        reverse=True,
    )
    thresholds = (8, 16, 32, 64, 128, 256, 512)
    return {
        "semantics": "four independently inferred child graphs before composition",
        "patches": sum(len(value.block.patches) for value in children),
        "retainedJoins": sum(len(value.block.joins) for value in children),
        "components": len(sizes),
        "largestOccupiedCellCount": max(sizes, default=0),
        "componentsAtLeastCells": {
            str(value): sum(size >= value for size in sizes) for value in thresholds
        },
        "children": [
            {
                "index": index,
                "offsetCellsXYZ": list(value.offset),
                "statistics": record,
            }
            for index, (value, record) in enumerate(zip(children, child_records))
        ],
    }


def run_cluster_materialization(
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    boundary_roots: Iterable[str | Path] | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Recompose a cluster solution into one complete selected surface graph.

    Mutable bands come from the joint solve.  Every child contributes only the
    complementary immutable geometry and retained joins certified during the
    solve.  The result is therefore a faithful materialization of the cluster,
    not a fresh global inference pass.
    """

    started = time.monotonic()
    cluster = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    if output == cluster:
        raise ValueError("cluster materialization output must differ from its input")
    cluster_manifest_path = cluster / "cluster-reselection-v1.json"
    cluster_manifest = json.loads(cluster_manifest_path.read_text())
    if (
        cluster_manifest.get("schema") != CLUSTER_RESELECTION_SCHEMA
        or cluster_manifest.get("state") != "complete"
    ):
        raise ValueError("cluster materialization requires a complete joint solve")
    reselection_record = cluster_manifest["artifacts"]["reselection"]
    if sha256_file(cluster / reselection_record["path"]) != reselection_record[
        "sha256"
    ]:
        raise ValueError("cluster reselection artifact identity changed")
    boundaries = _ordered_boundaries(cluster_manifest, boundary_roots)
    layout = _cluster_layout(boundaries)
    if list(layout.grid.shape_cells_xyz) != cluster_manifest["grid"][
        "shapeCellsXYZ"
    ]:
        raise ValueError("cluster boundary overrides reconstruct another grid")
    children = _child_graphs(boundaries, cluster_manifest)
    identity = _identity(cluster, cluster_manifest, children)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "cluster-materialization-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("cluster materialization output belongs to another identity")
        if (
            not force
            and prior.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": CLUSTER_MATERIALIZATION_SCHEMA,
        "version": CLUSTER_MATERIALIZATION_VERSION,
        "state": "materializing",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    cluster_table = read_patch_shard(
        cluster / "selected-cluster-patches-v1", verify=True
    )
    cluster_patches = {
        value.patch_id: value for value in cluster_table.to_patches()
    }
    cluster_rows = {
        int(patch_id): row for row, patch_id in enumerate(cluster_table.patch_id)
    }
    with np.load(cluster / reselection_record["path"]) as values:
        artifact = {name: np.asarray(values[name]) for name in values.files}
    references = {
        int(patch_id): {
            "input": int(block),
            "sourcePatchId": int(source),
            "isAnchor": bool(anchor),
        }
        for patch_id, block, source, anchor in zip(
            artifact["patchId"],
            artifact["patchInput"],
            artifact["patchSourcePatchId"],
            artifact["patchIsAnchor"],
        )
    }
    if set(references) != set(cluster_patches):
        raise ValueError("cluster patch table and reselection references disagree")

    final_patches: list[ClippedPatch] = []
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    source_to_final: dict[tuple[int, int], int] = {}
    cluster_to_final: dict[int, int] = {}
    provenance: list[tuple[int, int, int, int, int]] = []
    next_patch_id = 1

    for block_index, child in enumerate(children):
        row_by_id = {
            int(patch_id): row for row, patch_id in enumerate(child.table.patch_id)
        }
        for source_patch in child.block.patches:
            if source_patch.patch_id not in child.immutable_patch_ids:
                continue
            patch_id = next_patch_id
            next_patch_id += 1
            patch = clip_plane_to_cell(
                layout.grid,
                _offset_cell(source_patch.cell_xyz, child.offset),
                source_patch.estimate,
                patch_id=patch_id,
            )
            if patch is None:
                raise ValueError("rebased immutable patch no longer intersects its cell")
            row = row_by_id[source_patch.patch_id]
            final_patches.append(patch)
            source_to_final[(block_index, source_patch.patch_id)] = patch_id
            configuration_id[patch_id] = int(child.table.configuration_id[row])
            configuration_log_weight[patch_id] = float(
                child.table.configuration_log_weight[row]
            )
            local_order[patch_id] = int(child.table.local_order[row])
            normal_family[patch_id] = int(child.table.normal_family[row])
            provenance.append(
                (patch_id, block_index, source_patch.patch_id, -1, 0)
            )

    for cluster_patch_id in sorted(cluster_patches):
        reference = references[cluster_patch_id]
        if reference["isAnchor"]:
            key = (reference["input"], reference["sourcePatchId"])
            if key not in source_to_final:
                raise ValueError("cluster anchor does not map to immutable child geometry")
            cluster_to_final[cluster_patch_id] = source_to_final[key]
            continue
        patch_id = next_patch_id
        next_patch_id += 1
        source = cluster_patches[cluster_patch_id]
        patch = clip_plane_to_cell(
            layout.grid, source.cell_xyz, source.estimate, patch_id=patch_id
        )
        if patch is None:
            raise ValueError("selected mutable patch no longer intersects its cell")
        row = cluster_rows[cluster_patch_id]
        final_patches.append(patch)
        cluster_to_final[cluster_patch_id] = patch_id
        configuration_id[patch_id] = int(cluster_table.configuration_id[row])
        configuration_log_weight[patch_id] = float(
            cluster_table.configuration_log_weight[row]
        )
        local_order[patch_id] = int(cluster_table.local_order[row])
        normal_family[patch_id] = int(cluster_table.normal_family[row])
        provenance.append(
            (
                patch_id,
                reference["input"],
                reference["sourcePatchId"],
                cluster_patch_id,
                1,
            )
        )

    final_table = PatchTable.from_patches(
        layout.grid,
        final_patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        final_table,
        settings={
            "materialization": "joint mutable bands plus certified immutable interiors",
            "freshInference": False,
        },
        provenance={
            "clusterMaterializationIdentitySha256": identity_sha256,
            "clusterRoot": str(cluster),
        },
        compressed=True,
    )

    joins: list[TraceMatch] = []
    immutable_join_counts: list[int] = []
    for block_index, child in enumerate(children):
        patch_map = {
            source: target
            for (owner, source), target in source_to_final.items()
            if owner == block_index
        }
        retained = [
            value
            for value in child.block.joins
            if value.first_patch_id in child.immutable_patch_ids
            and value.second_patch_id in child.immutable_patch_ids
        ]
        immutable_join_counts.append(len(retained))
        joins.extend(
            _offset_join(value, child.offset, patch_map) for value in retained
        )

    final_by_id = {value.patch_id: value for value in final_patches}
    cluster_join_count = len(artifact["joinFirstPatchId"])
    for first, second, axis, anchor, quarter in zip(
        artifact["joinFirstPatchId"],
        artifact["joinSecondPatchId"],
        artifact["joinFaceAxis"],
        artifact["joinFaceAnchorXYZ"],
        artifact["joinFiberQuarterTurn"],
    ):
        quarter_value = int(quarter)
        joins.append(
            reconstruct_retained_join(
                final_by_id,
                cluster_to_final[int(first)],
                cluster_to_final[int(second)],
                GridFace(int(axis), tuple(int(value) for value in anchor)),
                None if quarter_value < 0 else bool(quarter_value),
                grid=layout.grid,
            )
        )

    block = surface_block_from_retained_joins(
        layout.grid,
        BlockBounds((0, 0, 0), layout.grid.shape_cells_xyz),
        final_patches,
        joins,
    )
    expected_components = int(
        cluster_manifest["statistics"]["recomposedComponents"]
    )
    if len(block.components) != expected_components:
        raise ValueError(
            "materialized graph component count disagrees with frozen cluster solve: "
            f"{len(block.components)} != {expected_components}"
        )

    graph_manifest = write_surface_graph(
        output,
        block,
        semantics=str(cluster_manifest["graphSemantics"]),
        provenance={
            "clusterMaterializationIdentitySha256": identity_sha256,
            "clusterRoot": str(cluster),
            "immutableJoinCountsByChild": immutable_join_counts,
            "clusterBandJoins": cluster_join_count,
        },
    )
    provenance_path = output / "patch-provenance-v1.npz"
    temporary = provenance_path.with_suffix(provenance_path.suffix + ".tmp")
    provenance.sort()
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            patchId=np.asarray([value[0] for value in provenance], dtype=np.uint64),
            inputIndex=np.asarray([value[1] for value in provenance], dtype=np.int8),
            sourcePatchId=np.asarray([value[2] for value in provenance], dtype=np.int64),
            clusterPatchId=np.asarray([value[3] for value in provenance], dtype=np.int64),
            mutableBand=np.asarray([value[4] for value in provenance], dtype=np.uint8),
        )
    temporary.replace(provenance_path)

    child_bounds = tuple(
        BlockBounds(
            value.offset,
            tuple(
                value.offset[axis] + value.table.grid.shape_cells_xyz[axis]
                for axis in range(3)
            ),
        )
        for value in children
    )
    materialized_statistics = component_statistics(
        block, child_bounds=child_bounds, maximum_records=32
    )
    baseline = _baseline_statistics(children)
    current_largest = int(materialized_statistics["largestOccupiedCellCount"])
    baseline_largest = int(baseline["largestOccupiedCellCount"])
    growth = {
        "largestFragmentCellGain": current_largest - baseline_largest,
        "largestFragmentFactor": round(
            current_largest / max(baseline_largest, 1), 6
        ),
        "componentReduction": int(baseline["components"])
        - int(materialized_statistics["components"]),
        "componentsSpanningMultipleChildren": sum(
            int(key) >= 2 and int(value)
            for key, value in materialized_statistics[
                "componentsByChildBlockCount"
            ].items()
        ),
    }
    summary = {
        "schema": "pareidolia.cubical-boundary-cluster-materialization-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "clusterRoot": str(cluster),
        "outputRoot": str(output),
        "method": {
            "mutableGeometry": "joint cluster selection",
            "immutableGeometry": "original child selections outside every internal band",
            "mutableTopology": "joint cluster retained joins",
            "immutableTopology": "original child joins with both endpoints immutable",
            "freshInference": False,
        },
        "counts": {
            "immutablePatches": sum(len(value.immutable_patch_ids) for value in children),
            "mutablePatches": sum(value[4] for value in provenance),
            "immutableJoins": sum(immutable_join_counts),
            "mutableBandJoins": cluster_join_count,
        },
        "independentChildBaseline": baseline,
        "materializedCluster": materialized_statistics,
        "growth": growth,
        "artifacts": {
            "selectedPatchManifest": "selected-patches-v1.json",
            "selectedPatchData": "selected-patches-v1.npz",
            "surfaceGraphManifest": "surface-graph-v1.json",
            "surfaceGraphData": "surface-graph-v1.npz",
            "patchProvenance": provenance_path.name,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output / "summary.json", summary)
    variant = {
        "schema": CLUSTER_MATERIALIZATION_SCHEMA,
        "version": CLUSTER_MATERIALIZATION_VERSION,
        "state": "complete",
        "inputRoot": str(children[0].selected_root),
        "identity": identity,
        "summary": "summary.json",
    }
    atomic_json(output / "variant.json", variant)
    manifest.update(
        {
            "state": "complete",
            "summary": "summary.json",
            "artifacts": {
                "selectedPatches": {
                    "manifest": "selected-patches-v1.json",
                    "data": "selected-patches-v1.npz",
                    "sha256": patch_manifest["data"]["sha256"],
                },
                "surfaceGraph": {
                    "manifest": "surface-graph-v1.json",
                    "data": "surface-graph-v1.npz",
                    "sha256": graph_manifest["data"]["sha256"],
                },
                "patchProvenance": {
                    "path": provenance_path.name,
                    "sha256": sha256_file(provenance_path),
                },
            },
            "elapsedSeconds": summary["timingSeconds"]["total"],
        }
    )
    atomic_json(manifest_path, manifest)
    return summary
