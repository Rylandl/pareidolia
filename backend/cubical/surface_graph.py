from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .block import (
    BlockBounds,
    SurfaceBlock,
    surface_block_from_retained_joins,
)
from .contracts import atomic_json, sha256_file
from .geometry import ClippedPatch
from .matching import TraceMatch, TraceMatchSettings, match_face_traces
from .tables import PatchTable, read_patch_shard
from .topology import GridFace, Int3


SURFACE_GRAPH_SCHEMA = "pareidolia.cubical-retained-surface-graph"
SURFACE_GRAPH_VERSION = 1
SURFACE_GRAPH_STEM = "surface-graph-v1"


def join_key(value: TraceMatch) -> tuple[int, int, int, Int3]:
    return (
        value.first_patch_id,
        value.second_patch_id,
        value.face.axis,
        value.face.anchor_xyz,
    )


def reconstruct_retained_join(
    patch_by_id: Mapping[int, ClippedPatch],
    first_patch_id: int,
    second_patch_id: int,
    face: GridFace,
    fiber_quarter_turn: bool | None,
    *,
    grid: Any,
) -> TraceMatch:
    try:
        first = patch_by_id[first_patch_id]
        second = patch_by_id[second_patch_id]
    except KeyError as error:
        raise ValueError("surface graph join references an absent patch") from error
    first_trace = first.trace_on(face)
    second_trace = second.trace_on(face)
    if first_trace is None or second_trace is None:
        raise ValueError("surface graph join does not cross its declared face")
    match = match_face_traces(
        first_trace,
        first.estimate,
        second_trace,
        second.estimate,
        TraceMatchSettings(
            orthogonal_fiber_equivalence=fiber_quarter_turn is True,
        ),
        grid=grid,
    )
    if not match.accepted or len(match.endpoint_agreements) != 2:
        raise ValueError(
            "surface graph join cannot be reconstructed from stored patch geometry"
        )
    if fiber_quarter_turn is True and match.fiber_quarter_turn is not True:
        raise ValueError("surface graph quarter-turn classification changed")
    if fiber_quarter_turn is False and match.fiber_quarter_turn is True:
        raise ValueError("surface graph strict join became a quarter-turn match")
    return match


def _component_map_from_arrays(
    patch_ids: np.ndarray,
    component_ids: np.ndarray,
) -> dict[int, int]:
    if patch_ids.shape != component_ids.shape or patch_ids.ndim != 1:
        raise ValueError("surface graph component arrays are invalid")
    result = {
        int(patch_id): int(component_id)
        for patch_id, component_id in zip(patch_ids, component_ids)
    }
    if len(result) != len(patch_ids):
        raise ValueError("surface graph component map contains duplicate patch IDs")
    return result


def _validate_component_map(
    block: SurfaceBlock,
    declared: Mapping[int, int],
) -> None:
    computed = dict(block.component_by_patch)
    if set(computed) != set(declared):
        raise ValueError("surface graph component map does not cover every patch")
    declared_groups: dict[int, set[int]] = {}
    computed_groups: dict[int, set[int]] = {}
    for patch_id, component_id in declared.items():
        declared_groups.setdefault(component_id, set()).add(patch_id)
    for patch_id, component_id in computed.items():
        computed_groups.setdefault(component_id, set()).add(patch_id)
    if {frozenset(value) for value in declared_groups.values()} != {
        frozenset(value) for value in computed_groups.values()
    }:
        raise ValueError("surface graph joins disagree with its component map")


def _block_from_graph_arrays(
    table: PatchTable,
    arrays: Mapping[str, np.ndarray],
) -> SurfaceBlock:
    patches = table.to_patches()
    patch_by_id = {value.patch_id: value for value in patches}
    required = (
        "firstPatchId",
        "secondPatchId",
        "faceAxis",
        "faceAnchorXYZ",
        "fiberQuarterTurn",
        "patchId",
        "componentId",
    )
    if any(name not in arrays for name in required):
        raise ValueError("surface graph artifact lacks required arrays")
    count = len(arrays["firstPatchId"])
    shapes = {
        "secondPatchId": (count,),
        "faceAxis": (count,),
        "faceAnchorXYZ": (count, 3),
        "fiberQuarterTurn": (count,),
    }
    for name, shape in shapes.items():
        if arrays[name].shape != shape:
            raise ValueError(f"surface graph array {name} has invalid shape")
    joins: list[TraceMatch] = []
    for first, second, axis, anchor, quarter in zip(
        arrays["firstPatchId"],
        arrays["secondPatchId"],
        arrays["faceAxis"],
        arrays["faceAnchorXYZ"],
        arrays["fiberQuarterTurn"],
    ):
        quarter_value = int(quarter)
        if quarter_value not in (-1, 0, 1):
            raise ValueError("surface graph quarter-turn values must be -1, 0, or 1")
        joins.append(
            reconstruct_retained_join(
                patch_by_id,
                int(first),
                int(second),
                GridFace(int(axis), tuple(int(value) for value in anchor)),
                None if quarter_value < 0 else bool(quarter_value),
                grid=table.grid,
            )
        )
    block = surface_block_from_retained_joins(
        table.grid,
        BlockBounds((0, 0, 0), table.grid.shape_cells_xyz),
        patches,
        joins,
    )
    declared = _component_map_from_arrays(
        arrays["patchId"], arrays["componentId"]
    )
    _validate_component_map(block, declared)
    return block


def read_surface_graph(
    root: str | Path,
    *,
    table: PatchTable | None = None,
    verify: bool = True,
) -> SurfaceBlock:
    """Load a complete retained graph and reconstruct its welded surface block."""

    source = Path(root).resolve()
    manifest_path = source / f"{SURFACE_GRAPH_STEM}.json"
    data_path = source / f"{SURFACE_GRAPH_STEM}.npz"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != SURFACE_GRAPH_SCHEMA
        or int(manifest.get("version", -1)) != SURFACE_GRAPH_VERSION
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete retained surface graph")
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("retained surface graph content hash mismatch")
    resolved_table = table or read_patch_shard(
        source / "selected-patches-v1", verify=verify
    )
    if verify:
        if sha256_file(source / "selected-patches-v1.json") != manifest[
            "selectedPatches"
        ]["manifestSha256"]:
            raise ValueError("retained graph patch-manifest identity changed")
        if sha256_file(source / "selected-patches-v1.npz") != manifest[
            "selectedPatches"
        ]["dataSha256"]:
            raise ValueError("retained graph patch-data identity changed")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return _block_from_graph_arrays(resolved_table, arrays)


def read_legacy_packet_graph(
    selected_root: str | Path,
    packet_root: str | Path,
    *,
    table: PatchTable | None = None,
    verify: bool = True,
) -> SurfaceBlock:
    """Load the established dual-axis packet graph through the common boundary."""

    selected = Path(selected_root).resolve()
    packet = Path(packet_root).resolve()
    manifest = json.loads((packet / "packets.json").read_text())
    identity = manifest.get("identity", {})
    if (
        manifest.get("schema") != "pareidolia.cubical-dual-axis-sheet-packets"
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete dual-axis packet graph")
    if verify:
        if identity.get("inputPatchManifestSha256") != sha256_file(
            selected / "selected-patches-v1.json"
        ):
            raise ValueError("packet graph patch-manifest identity changed")
        if identity.get("inputPatchDataSha256") != sha256_file(
            selected / "selected-patches-v1.npz"
        ):
            raise ValueError("packet graph patch-data identity changed")
    resolved_table = table or read_patch_shard(
        selected / "selected-patches-v1", verify=verify
    )
    with np.load(packet / "packet-graph-v1.npz") as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    # The legacy writer encoded missing fiber evidence as zero along with strict
    # matches.  Reconstructing those rows as strict retains their exact geometry.
    arrays["fiberQuarterTurn"] = np.asarray(
        arrays["fiberQuarterTurn"], dtype=np.int8
    )
    return _block_from_graph_arrays(resolved_table, arrays)


def _polygon_area(patch: ClippedPatch, grid: Any) -> float:
    points = np.asarray(
        [value.edge.point_world(grid, value.t) for value in patch.vertices],
        dtype=np.float64,
    )
    cross_sum = np.sum(np.cross(points, np.roll(points, -1, axis=0)), axis=0)
    return 0.5 * float(np.linalg.norm(cross_sum))


def component_statistics(
    block: SurfaceBlock,
    *,
    child_bounds: Iterable[BlockBounds] = (),
    maximum_records: int = 64,
) -> dict[str, Any]:
    """Return physical size and cluster-span statistics for retained fragments."""

    patch_by_id = {value.patch_id: value for value in block.patches}
    area_by_id = {
        value.patch_id: _polygon_area(value, block.grid) for value in block.patches
    }
    children = tuple(child_bounds)
    records: list[dict[str, Any]] = []
    for component in block.components:
        patches = [patch_by_id[value] for value in component.patch_ids]
        cells = np.asarray([value.cell_xyz for value in patches], dtype=np.int64)
        low = np.min(cells, axis=0)
        high = np.max(cells, axis=0) + 1
        child_ids = {
            index
            for index, bounds in enumerate(children)
            if any(bounds.contains_cell(value.cell_xyz) for value in patches)
        }
        records.append(
            {
                "componentId": component.component_id,
                "patchCount": len(component.patch_ids),
                "occupiedCellCount": len({value.cell_xyz for value in patches}),
                "surfaceAreaSquareVoxels": round(
                    sum(area_by_id[value] for value in component.patch_ids), 4
                ),
                "cellBounds": {
                    "startXYZ": [int(value) for value in low],
                    "stopXYZExclusive": [int(value) for value in high],
                    "spanXYZ": [int(value) for value in high - low],
                },
                "childBlockCount": len(child_ids) if children else None,
                "childBlocks": sorted(child_ids) if children else None,
                "exteriorTraceCount": component.exterior_trace_count,
                "unresolvedInteriorTraceCount": (
                    component.unresolved_interior_trace_count
                ),
            }
        )
    records.sort(
        key=lambda value: (
            -int(value["occupiedCellCount"]),
            -float(value["surfaceAreaSquareVoxels"]),
            int(value["componentId"]),
        )
    )
    sizes = np.asarray(
        [value["occupiedCellCount"] for value in records], dtype=np.int64
    )
    thresholds = (8, 16, 32, 64, 128, 256, 512)
    child_counts = Counter(
        int(value["childBlockCount"])
        for value in records
        if value["childBlockCount"] is not None
    )
    return {
        "patches": len(block.patches),
        "retainedJoins": len(block.joins),
        "components": len(block.components),
        "largestOccupiedCellCount": int(sizes[0]) if len(sizes) else 0,
        "medianOccupiedCellCount": (
            round(float(np.median(sizes)), 4) if len(sizes) else 0.0
        ),
        "componentsAtLeastCells": {
            str(value): int(np.sum(sizes >= value)) for value in thresholds
        },
        "componentsByChildBlockCount": {
            str(key): value for key, value in sorted(child_counts.items())
        },
        "topComponents": records[:maximum_records],
    }


def write_surface_graph(
    root: str | Path,
    block: SurfaceBlock,
    *,
    semantics: str,
    provenance: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Write one complete, independently verifiable retained surface graph."""

    output = Path(root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    patch_manifest_path = output / "selected-patches-v1.json"
    patch_data_path = output / "selected-patches-v1.npz"
    if not patch_manifest_path.is_file() or not patch_data_path.is_file():
        raise ValueError("surface graph output requires a selected-patch shard")
    joins = tuple(block.joins)
    component_by_patch = dict(block.component_by_patch)
    data_path = output / f"{SURFACE_GRAPH_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray(
                [value.first_patch_id for value in joins], dtype=np.uint64
            ),
            secondPatchId=np.asarray(
                [value.second_patch_id for value in joins], dtype=np.uint64
            ),
            faceAxis=np.asarray([value.face.axis for value in joins], dtype=np.int8),
            faceAnchorXYZ=np.asarray(
                [value.face.anchor_xyz for value in joins], dtype=np.int32
            ).reshape(len(joins), 3),
            fiberQuarterTurn=np.asarray(
                [
                    -1
                    if value.fiber_quarter_turn is None
                    else int(value.fiber_quarter_turn)
                    for value in joins
                ],
                dtype=np.int8,
            ),
            score=np.asarray([value.score for value in joins], dtype=np.float32),
            negativeLogLikelihood=np.asarray(
                [value.negative_log_likelihood for value in joins], dtype=np.float32
            ),
            maximumEndpointZ=np.asarray(
                [
                    max(endpoint.z for endpoint in value.endpoint_agreements)
                    for value in joins
                ],
                dtype=np.float32,
            ),
            normalResidualDegrees=np.asarray(
                [math.degrees(value.normal_angle_radians) for value in joins],
                dtype=np.float32,
            ),
            fiberFrameResidualDegrees=np.asarray(
                [
                    np.nan
                    if value.fiber_angle_radians is None
                    else math.degrees(value.fiber_angle_radians)
                    for value in joins
                ],
                dtype=np.float32,
            ),
            patchId=np.asarray(sorted(component_by_patch), dtype=np.uint64),
            componentId=np.asarray(
                [component_by_patch[value] for value in sorted(component_by_patch)],
                dtype=np.uint64,
            ),
        )
    temporary.replace(data_path)
    manifest = {
        "schema": SURFACE_GRAPH_SCHEMA,
        "version": SURFACE_GRAPH_VERSION,
        "state": "complete",
        "semantics": semantics,
        "selectedPatches": {
            "manifestSha256": sha256_file(patch_manifest_path),
            "dataSha256": sha256_file(patch_data_path),
        },
        "counts": {
            "patches": len(block.patches),
            "retainedJoins": len(block.joins),
            "components": len(block.components),
        },
        "provenance": dict(provenance or {}),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(output / f"{SURFACE_GRAPH_STEM}.json", manifest)
    return manifest
