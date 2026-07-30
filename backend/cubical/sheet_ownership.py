from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .block import BlockBounds, SurfaceBlock, surface_block_from_retained_joins
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import DegeneratePlaneIntersection, clip_plane_to_cell
from .sheet_configuration_solver import (
    SheetConfigurationSolverSettings,
    _factor_value,
    _read_factors,
    _unary_values,
    run_sheet_configuration_initialization,
)
from .sheet_correspondence import catalog_block_sheet_correspondences
from .sheet_evidence import (
    SHEET_EVIDENCE_SCHEMA,
    SHEET_EVIDENCE_STEM,
    SHEET_EVIDENCE_VERSION,
    BlockSheetEvidence,
    read_block_sheet_evidence,
)
from .sheet_factors import SheetFactorSettings, compile_sheet_configuration_factors
from .sheet_graph_solver import replay_joint_sheet_graph
from .sheet_stitching import SheetStitchingSettings, run_block_sheet_restitching
from .surface_graph import (
    join_key,
    read_surface_graph,
    reconstruct_retained_join,
    write_surface_graph,
)
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridFace, GridSpec, Int3


SHEET_EVIDENCE_SUBBLOCK_SCHEMA = "pareidolia.cubical-sheet-evidence-subblock"
SHEET_OWNERSHIP_CROP_SCHEMA = "pareidolia.cubical-sheet-ownership-crop"
SHEET_HALO_EXPERIMENT_SCHEMA = "pareidolia.cubical-sheet-halo-experiment"
SHEET_HALO_AUDIT_SCHEMA = "pareidolia.cubical-sheet-halo-audit"
SHEET_HALO_FINAL_AUDIT_SCHEMA = "pareidolia.cubical-sheet-halo-final-audit"
SHEET_OWNERSHIP_VERSION = 1


def _triple(values: Iterable[int], label: str) -> Int3:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{label} requires an XYZ triple")
    return result  # type: ignore[return-value]


def _validate_bounds(grid: GridSpec, start: Int3, stop: Int3) -> None:
    if any(
        start[axis] < 0
        or stop[axis] <= start[axis]
        or stop[axis] > grid.shape_cells_xyz[axis]
        for axis in range(3)
    ):
        raise ValueError("sheet subblock bounds must be a positive grid subset")


def _contains(cell: Iterable[int], start: Int3, stop: Int3) -> bool:
    values = tuple(int(value) for value in cell)
    return all(start[axis] <= values[axis] < stop[axis] for axis in range(3))


def _subgrid(grid: GridSpec, start: Int3, stop: Int3) -> GridSpec:
    return GridSpec(
        tuple(stop[axis] - start[axis] for axis in range(3)),
        grid.cell_size_xyz,
        tuple(float(value) for value in grid.vertex_world(start)),
        grid.coordinate_unit,
    )


def _grid_record(grid: GridSpec) -> dict[str, Any]:
    return {
        "shapeCellsXYZ": list(grid.shape_cells_xyz),
        "cellSizeXYZ": list(grid.cell_size_xyz),
        "originXYZ": list(grid.origin_xyz),
        "coordinateUnit": grid.coordinate_unit,
    }


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> dict[str, Any]:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _evidence_statistics(evidence: BlockSheetEvidence) -> dict[str, Any]:
    status = np.asarray(evidence.arrays["modeGeometryStatus"], dtype=np.uint8)
    valid_configurations = np.asarray(
        evidence.arrays["configurationGeometryValid"], dtype=np.uint8
    )
    current = np.asarray(
        evidence.arrays["configurationIsCurrent"], dtype=np.uint8
    )
    memberships = np.asarray(evidence.arrays["configurationModeId"])
    return {
        "ownedCells": evidence.cell_count,
        "uniqueAcusModes": evidence.mode_count,
        "validModePatches": evidence.mode_patches.patch_count,
        "modesMissingCell": int(np.sum(status == 1)),
        "degenerateModePatches": int(np.sum(status == 2)),
        "physicalConfigurations": evidence.configuration_count,
        "configurationModeMemberships": len(memberships),
        "geometryValidConfigurations": int(np.sum(valid_configurations)),
        "currentConfigurations": int(np.sum(current)),
        "meanUniqueModesPerCell": round(
            evidence.mode_count / max(evidence.cell_count, 1), 6
        ),
        "meanConfigurationsPerCell": round(
            evidence.configuration_count / max(evidence.cell_count, 1), 6
        ),
    }


def extract_sheet_evidence_subblock(
    evidence_root: str | Path,
    output_root: str | Path,
    *,
    start_cell_xyz: Int3,
    stop_cell_xyz_exclusive: Int3,
    force: bool = False,
) -> dict[str, Any]:
    """Extract and rebase an exact rectangular subset of an immutable Acus bake.

    Stable mode and physical-configuration IDs are preserved. Geometry is
    reclipped against a grid whose origin is translated to the requested cell
    corner, so world-space planes and crossings remain unchanged while local
    cell and face coordinates start at zero.
    """

    started = time.monotonic()
    source = Path(evidence_root).resolve()
    output = Path(output_root).resolve()
    if source == output:
        raise ValueError("sheet-evidence subblock output must differ from its source")
    evidence = read_block_sheet_evidence(source, verify=True)
    start = _triple(start_cell_xyz, "subblock start")
    stop = _triple(stop_cell_xyz_exclusive, "subblock stop")
    _validate_bounds(evidence.grid, start, stop)
    grid = _subgrid(evidence.grid, start, stop)
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_EVIDENCE_SUBBLOCK_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "sourceRoot": str(source),
        "sourceEvidenceManifestSha256": sha256_file(
            source / f"{SHEET_EVIDENCE_STEM}.json"
        ),
        "sourceEvidenceDataSha256": sha256_file(
            source / f"{SHEET_EVIDENCE_STEM}.npz"
        ),
        "sourceModePatchManifestSha256": sha256_file(
            source / "mode-patches-v1.json"
        ),
        "sourceModePatchDataSha256": sha256_file(
            source / "mode-patches-v1.npz"
        ),
        "startCellXYZ": list(start),
        "stopCellXYZExclusive": list(stop),
        "grid": _grid_record(grid),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_ownership.py",
                "sheet_evidence.py",
                "geometry.py",
                "tables.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_EVIDENCE_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet-evidence subblock output belongs to another identity")
        if not force and prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_EVIDENCE_SCHEMA,
            "version": SHEET_EVIDENCE_VERSION,
            "state": "extracting-subblock",
            "identity": identity,
        },
    )

    arrays = evidence.arrays
    cells = np.asarray(arrays["cellXYZ"], dtype=np.int32)
    cell_rows = np.flatnonzero(
        np.all(cells >= np.asarray(start), axis=1)
        & np.all(cells < np.asarray(stop), axis=1)
    )
    expected_cells = int(np.prod(grid.shape_cells_xyz))
    if len(cell_rows) != expected_cells:
        raise ValueError(
            "sheet-evidence subblock must contain every cell in its rectangular bounds"
        )

    configuration_offset = np.asarray(
        arrays["configurationOffset"], dtype=np.uint64
    )
    configuration_rows: list[int] = []
    subset_configuration_offset = np.zeros(len(cell_rows) + 1, dtype=np.uint64)
    for output_index, source_index_value in enumerate(cell_rows):
        source_index = int(source_index_value)
        low = int(configuration_offset[source_index])
        high = int(configuration_offset[source_index + 1])
        configuration_rows.extend(range(low, high))
        subset_configuration_offset[output_index + 1] = len(configuration_rows)
    configuration_index = np.asarray(configuration_rows, dtype=np.int64)

    mode_cells = np.asarray(arrays["modeCellXYZ"], dtype=np.int32)
    mode_rows = np.flatnonzero(
        np.all(mode_cells >= np.asarray(start), axis=1)
        & np.all(mode_cells < np.asarray(stop), axis=1)
    )
    mode_index = np.asarray(mode_rows, dtype=np.int64)
    selected_mode_ids = {
        int(value) for value in np.asarray(arrays["modeId"])[mode_index]
    }

    membership_offset = np.asarray(
        arrays["configurationModeOffset"], dtype=np.uint64
    )
    source_membership = np.asarray(arrays["configurationModeId"], dtype=np.uint64)
    subset_membership: list[int] = []
    subset_membership_offset = np.zeros(
        len(configuration_index) + 1, dtype=np.uint64
    )
    for output_index, source_index_value in enumerate(configuration_index):
        source_index = int(source_index_value)
        low = int(membership_offset[source_index])
        high = int(membership_offset[source_index + 1])
        values = [int(value) for value in source_membership[low:high]]
        if any(value not in selected_mode_ids for value in values):
            raise ValueError("one retained configuration references a mode outside its cell")
        subset_membership.extend(values)
        subset_membership_offset[output_index + 1] = len(subset_membership)

    subset_arrays: dict[str, np.ndarray] = {}
    for name, values in arrays.items():
        array = np.asarray(values)
        if name == "cellXYZ":
            subset_arrays[name] = cells[cell_rows] - np.asarray(start, dtype=np.int32)
        elif name == "configurationOffset":
            subset_arrays[name] = subset_configuration_offset
        elif name == "configurationModeOffset":
            subset_arrays[name] = subset_membership_offset
        elif name == "configurationModeId":
            subset_arrays[name] = np.asarray(subset_membership, dtype=np.uint64)
        elif name == "modeCellXYZ":
            subset_arrays[name] = (
                mode_cells[mode_index] - np.asarray(start, dtype=np.int32)
            )
        elif name.startswith("mode"):
            subset_arrays[name] = array[mode_index]
        elif name.startswith("configuration"):
            subset_arrays[name] = array[configuration_index]
        else:
            raise ValueError(f"unclassified sheet-evidence array: {name}")

    source_table = evidence.mode_patches
    source_patches = source_table.to_patches()
    mode_ids = {int(value) for value in subset_arrays["modeId"]}
    tolerance_scale = float(
        evidence.manifest.get("identity", {})
        .get("settings", {})
        .get("clipping_tolerance_scale", 1.0e-8)
    )
    tolerance = max(grid.cell_size_xyz) * tolerance_scale
    patches = []
    for source_patch in source_patches:
        if source_patch.patch_id not in mode_ids:
            continue
        cell = tuple(
            source_patch.cell_xyz[axis] - start[axis] for axis in range(3)
        )
        try:
            patch = clip_plane_to_cell(
                grid,
                cell,
                source_patch.estimate,
                patch_id=source_patch.patch_id,
                tolerance=tolerance,
            )
        except DegeneratePlaneIntersection as error:
            raise ValueError("valid source mode became degenerate after rebasing") from error
        if patch is None:
            raise ValueError("valid source mode misses its rebased cell")
        patches.append(patch)
    source_metadata = {
        int(patch_id): row for row, patch_id in enumerate(source_table.patch_id)
    }
    patch_table = PatchTable.from_patches(
        grid,
        tuple(patches),
        configuration_id={
            patch.patch_id: int(
                source_table.configuration_id[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
        configuration_log_weight={
            patch.patch_id: float(
                source_table.configuration_log_weight[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
        local_order={
            patch.patch_id: int(source_table.local_order[source_metadata[patch.patch_id]])
            for patch in patches
        },
        normal_family={
            patch.patch_id: int(
                source_table.normal_family[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
    )
    patch_manifest = write_patch_shard(
        output / "mode-patches-v1",
        patch_table,
        settings={
            "semantics": "exact rebased subset of immutable Acus layer modes",
            "startCellXYZ": list(start),
            "stopCellXYZExclusive": list(stop),
        },
        provenance={
            "sourceEvidenceRoot": str(source),
            "sheetEvidenceSubblockIdentitySha256": identity_sha256,
        },
        compressed=True,
    )
    data_record = _write_npz(
        output / f"{SHEET_EVIDENCE_STEM}.npz", subset_arrays
    )
    provisional_manifest = {
        "schema": SHEET_EVIDENCE_SCHEMA,
        "version": SHEET_EVIDENCE_VERSION,
        "state": "complete",
        "identity": identity,
        "summary": summary_path.name,
        "data": data_record,
        "modePatches": {
            "manifest": "mode-patches-v1.json",
            "data": "mode-patches-v1.npz",
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(manifest_path, provisional_manifest)
    restored = read_block_sheet_evidence(output, verify=True)
    statistics = _evidence_statistics(restored)
    summary = {
        "schema": "pareidolia.cubical-block-sheet-evidence-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": statistics,
        "artifacts": {
            "evidence": data_record,
            "modePatches": {
                "manifest": "mode-patches-v1.json",
                "manifestSha256": sha256_file(output / "mode-patches-v1.json"),
                "data": patch_manifest["data"],
            },
        },
        "contract": {
            "AcusEvidenceMutableDuringSheetSolve": False,
            "stableModeAndConfigurationIds": True,
            "sourceCellBounds": {
                "startCellXYZ": list(start),
                "stopCellXYZExclusive": list(stop),
            },
            "worldGeometryRebased": True,
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(summary_path, summary)
    provisional_manifest["statistics"] = statistics
    provisional_manifest["elapsedSeconds"] = summary["elapsedSeconds"]
    atomic_json(manifest_path, provisional_manifest)
    return summary


def _component_patch_membership(block: SurfaceBlock, maximum_size: int) -> int:
    sizes = {value.component_id: len(value.patch_ids) for value in block.components}
    return sum(
        sizes[component_id] <= maximum_size
        for _, component_id in block.component_by_patch
    )


def _block_metrics(block: SurfaceBlock) -> dict[str, Any]:
    component_sizes = sorted(
        (len(value.patch_ids) for value in block.components), reverse=True
    )
    interior_endpoints = len(block.unresolved_interior_traces) + 2 * len(block.joins)
    return {
        "patches": len(block.patches),
        "retainedJoins": len(block.joins),
        "components": len(block.components),
        "largestComponent": component_sizes[0] if component_sizes else 0,
        "componentsAtMost8": sum(value <= 8 for value in component_sizes),
        "patchesInComponentsAtMost8": _component_patch_membership(block, 8),
        "unresolvedInteriorTraceEndpoints": len(block.unresolved_interior_traces),
        "interiorTraceEndpoints": interior_endpoints,
        "openInteriorTraceFraction": round(
            len(block.unresolved_interior_traces) / max(interior_endpoints, 1), 6
        ),
        "exteriorTraceEndpoints": len(block.exterior_traces),
    }


def crop_surface_graph_to_owned_block(
    graph_root: str | Path,
    output_root: str | Path,
    *,
    start_cell_xyz: Int3,
    stop_cell_xyz_exclusive: Int3,
    force: bool = False,
) -> dict[str, Any]:
    """Publish one owned core from a larger solved graph.

    Patches and incident joins outside the ownership bounds are removed, kept
    planes are rebased without changing world geometry, and connectivity is
    recomputed. Traces on the new ownership boundary consequently become
    exterior traces rather than unresolved interior gaps.
    """

    started = time.monotonic()
    source = Path(graph_root).resolve()
    output = Path(output_root).resolve()
    if source == output:
        raise ValueError("owned sheet graph output must differ from its source")
    source_table = read_patch_shard(source / "selected-patches-v1", verify=True)
    source_block = read_surface_graph(source, table=source_table, verify=True)
    start = _triple(start_cell_xyz, "ownership start")
    stop = _triple(stop_cell_xyz_exclusive, "ownership stop")
    _validate_bounds(source_block.grid, start, stop)
    grid = _subgrid(source_block.grid, start, stop)
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_OWNERSHIP_CROP_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "sourceRoot": str(source),
        "sourceGraphManifestSha256": sha256_file(source / "surface-graph-v1.json"),
        "sourceGraphDataSha256": sha256_file(source / "surface-graph-v1.npz"),
        "sourcePatchManifestSha256": sha256_file(
            source / "selected-patches-v1.json"
        ),
        "sourcePatchDataSha256": sha256_file(source / "selected-patches-v1.npz"),
        "startCellXYZ": list(start),
        "stopCellXYZExclusive": list(stop),
        "grid": _grid_record(grid),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_ownership.py",
                "surface_graph.py",
                "block.py",
                "geometry.py",
                "tables.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "ownership-crop-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("owned sheet graph output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_OWNERSHIP_CROP_SCHEMA,
            "version": SHEET_OWNERSHIP_VERSION,
            "state": "cropping",
            "identity": identity,
        },
    )

    source_metadata = {
        int(patch_id): row for row, patch_id in enumerate(source_table.patch_id)
    }
    patches = []
    for source_patch in source_block.patches:
        if not _contains(source_patch.cell_xyz, start, stop):
            continue
        cell = tuple(
            source_patch.cell_xyz[axis] - start[axis] for axis in range(3)
        )
        try:
            patch = clip_plane_to_cell(
                grid,
                cell,
                source_patch.estimate,
                patch_id=source_patch.patch_id,
            )
        except DegeneratePlaneIntersection as error:
            raise ValueError("selected patch became degenerate after ownership crop") from error
        if patch is None:
            raise ValueError("selected patch misses its rebased ownership cell")
        patches.append(patch)
    patch_by_id = {value.patch_id: value for value in patches}
    joins = []
    for value in source_block.joins:
        if (
            value.first_patch_id not in patch_by_id
            or value.second_patch_id not in patch_by_id
        ):
            continue
        face = GridFace(
            value.face.axis,
            tuple(
                value.face.anchor_xyz[axis] - start[axis] for axis in range(3)
            ),
        )
        joins.append(
            reconstruct_retained_join(
                patch_by_id,
                value.first_patch_id,
                value.second_patch_id,
                face,
                value.fiber_quarter_turn,
                grid=grid,
            )
        )
    patch_table = PatchTable.from_patches(
        grid,
        tuple(patches),
        configuration_id={
            patch.patch_id: int(
                source_table.configuration_id[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
        configuration_log_weight={
            patch.patch_id: float(
                source_table.configuration_log_weight[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
        local_order={
            patch.patch_id: int(source_table.local_order[source_metadata[patch.patch_id]])
            for patch in patches
        },
        normal_family={
            patch.patch_id: int(
                source_table.normal_family[source_metadata[patch.patch_id]]
            )
            for patch in patches
        },
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        patch_table,
        settings={
            "semantics": "owned core cropped after complete expanded sheet solve",
            "startCellXYZ": list(start),
            "stopCellXYZExclusive": list(stop),
        },
        provenance={
            "sourceGraphRoot": str(source),
            "ownershipCropIdentitySha256": identity_sha256,
        },
        compressed=True,
    )
    block = surface_block_from_retained_joins(
        grid,
        BlockBounds((0, 0, 0), grid.shape_cells_xyz),
        tuple(patches),
        tuple(joins),
    )
    graph_manifest = write_surface_graph(
        output,
        block,
        semantics="owned core of an expanded sheet solve after halo removal",
        provenance={
            "sourceGraphRoot": str(source),
            "ownershipCropIdentitySha256": identity_sha256,
        },
    )

    source_component_by_patch = dict(source_block.component_by_patch)
    child_parent: dict[int, int] = {}
    for component in block.components:
        parents = {
            source_component_by_patch[patch_id] for patch_id in component.patch_ids
        }
        if len(parents) != 1:
            raise RuntimeError("ownership crop unexpectedly merged source components")
        child_parent[component.component_id] = next(iter(parents))
    children_by_parent: dict[int, set[int]] = defaultdict(set)
    for child, parent in child_parent.items():
        children_by_parent[parent].add(child)
    kept_ids = set(patch_by_id)
    disposition = Counter()
    for component in source_block.components:
        kept_count = sum(patch_id in kept_ids for patch_id in component.patch_ids)
        if kept_count == 0:
            disposition["pruned"] += 1
        elif kept_count == len(component.patch_ids):
            disposition["retained"] += 1
        else:
            disposition["clipped"] += 1
    lineage_record = _write_npz(
        output / "component-lineage-v1.npz",
        {
            "childComponentId": np.asarray(sorted(child_parent), dtype=np.uint64),
            "parentComponentId": np.asarray(
                [child_parent[value] for value in sorted(child_parent)],
                dtype=np.uint64,
            ),
        },
    )
    atomic_json(
        output / "component-lineage-v1.json",
        {
            "schema": "pareidolia.cubical-sheet-component-lineage",
            "version": 1,
            "identitySha256": identity_sha256,
            "data": lineage_record,
        },
    )

    configuration_record = None
    source_configuration_path = source / "selected-configurations-v1.npz"
    if source_configuration_path.is_file():
        with np.load(source_configuration_path) as values:
            source_configuration = {
                name: np.asarray(values[name]) for name in values.files
            }
        cells = np.asarray(source_configuration["cellXYZ"], dtype=np.int32)
        rows = np.flatnonzero(
            np.all(cells >= np.asarray(start), axis=1)
            & np.all(cells < np.asarray(stop), axis=1)
        )
        owned_configuration = {
            name: (
                values[rows] - np.asarray(start, dtype=np.int32)
                if name == "cellXYZ"
                else values[rows]
            )
            for name, values in source_configuration.items()
        }
        configuration_record = _write_npz(
            output / "owned-configurations-v1.npz", owned_configuration
        )

    verified = read_surface_graph(output, verify=True)
    if _block_metrics(verified) != _block_metrics(block):
        raise RuntimeError("owned sheet graph changed during verification")
    summary = {
        "source": _block_metrics(source_block),
        "owned": _block_metrics(block),
        "componentDisposition": {
            "pruned": disposition["pruned"],
            "clipped": disposition["clipped"],
            "retained": disposition["retained"],
            "splitAfterCrop": sum(
                len(children) > 1 for children in children_by_parent.values()
            ),
        },
        "artifacts": {
            "selectedPatches": {
                "manifest": "selected-patches-v1.json",
                "manifestSha256": sha256_file(output / "selected-patches-v1.json"),
                "data": patch_manifest["data"],
            },
            "surfaceGraph": {
                "manifest": "surface-graph-v1.json",
                "manifestSha256": sha256_file(output / "surface-graph-v1.json"),
                "data": graph_manifest["data"],
            },
            "componentLineage": lineage_record,
            "ownedConfigurations": configuration_record,
        },
    }
    manifest = {
        "schema": SHEET_OWNERSHIP_CROP_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "state": "complete",
        "identity": identity,
        "summary": summary,
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(manifest_path, manifest)
    return manifest


def _shell_depth(cell: Int3, shape: Int3) -> int:
    return min(
        min(cell[axis], shape[axis] - 1 - cell[axis]) for axis in range(3)
    )


def _shell_metrics(block: SurfaceBlock) -> dict[str, Any]:
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in block.components
    }
    component_by_patch = dict(block.component_by_patch)
    patch_by_id = {value.patch_id: value for value in block.patches}
    patches = Counter()
    small_patches = Counter()
    interior = Counter()
    opened = Counter()
    for patch in block.patches:
        depth = _shell_depth(patch.cell_xyz, block.grid.shape_cells_xyz)
        patches[depth] += 1
        if component_sizes[component_by_patch[patch.patch_id]] <= 8:
            small_patches[depth] += 1
        for trace in patch.traces:
            lower, upper = trace.face.adjacent_cells()
            if block.bounds.contains_cell(lower) and block.bounds.contains_cell(upper):
                interior[depth] += 1
    for value in block.unresolved_interior_traces:
        depth = _shell_depth(
            patch_by_id[value.patch_id].cell_xyz, block.grid.shape_cells_xyz
        )
        opened[depth] += 1
    rows = []
    for depth in range(max(patches, default=-1) + 1):
        rows.append(
            {
                "depth": depth,
                "patches": patches[depth],
                "patchesInComponentsAtMost8": small_patches[depth],
                "smallPatchFraction": round(
                    small_patches[depth] / max(patches[depth], 1), 6
                ),
                "interiorTraceEndpoints": interior[depth],
                "unresolvedInteriorTraceEndpoints": opened[depth],
                "openInteriorTraceFraction": round(
                    opened[depth] / max(interior[depth], 1), 6
                ),
            }
        )
    return {"byDepth": rows}


def _owned_configuration_map(root: Path) -> dict[Int3, int]:
    path = root / "owned-configurations-v1.npz"
    if not path.is_file():
        return {}
    with np.load(path) as values:
        cells = np.asarray(values["cellXYZ"], dtype=np.int32)
        configuration_id = np.asarray(values["configurationId"], dtype=np.uint64)
    return {
        tuple(int(value) for value in cell): int(config_id)
        for cell, config_id in zip(cells, configuration_id)
    }


def compare_sheet_halo_outputs(
    owned_roots_by_halo: Mapping[int, str | Path],
    output_path: str | Path,
    *,
    configuration_roots_by_halo: Mapping[int, str | Path] | None = None,
) -> dict[str, Any]:
    roots = {
        int(halo): Path(root).resolve()
        for halo, root in owned_roots_by_halo.items()
    }
    if not roots:
        raise ValueError("sheet halo audit requires at least one owned graph")
    blocks = {halo: read_surface_graph(root, verify=True) for halo, root in roots.items()}
    reference_grid = blocks[min(blocks)].grid
    if any(value.grid != reference_grid for value in blocks.values()):
        raise ValueError("sheet halo outputs do not describe the same owned core")
    configuration_roots = (
        roots
        if configuration_roots_by_halo is None
        else {
            int(halo): Path(root).resolve()
            for halo, root in configuration_roots_by_halo.items()
        }
    )
    if set(configuration_roots) != set(roots):
        raise ValueError("sheet halo configuration roots do not cover every graph")
    configurations = {
        halo: _owned_configuration_map(configuration_roots[halo])
        for halo in roots
    }
    metrics = {
        str(halo): {
            **_block_metrics(blocks[halo]),
            "shell": _shell_metrics(blocks[halo]),
        }
        for halo in sorted(blocks)
    }
    comparisons = []
    halo_values = sorted(blocks)
    for first_index, first_halo in enumerate(halo_values):
        for second_halo in halo_values[first_index + 1 :]:
            first = blocks[first_halo]
            second = blocks[second_halo]
            first_patch_ids = {value.patch_id for value in first.patches}
            second_patch_ids = {value.patch_id for value in second.patches}
            first_joins = {join_key(value) for value in first.joins}
            second_joins = {join_key(value) for value in second.joins}
            first_config = configurations[first_halo]
            second_config = configurations[second_halo]
            shared_cells = set(first_config) & set(second_config)
            changed_by_depth = Counter(
                _shell_depth(cell, reference_grid.shape_cells_xyz)
                for cell in shared_cells
                if first_config[cell] != second_config[cell]
            )
            comparisons.append(
                {
                    "firstHalo": first_halo,
                    "secondHalo": second_halo,
                    "configurationCellsCompared": len(shared_cells),
                    "changedConfigurations": sum(changed_by_depth.values()),
                    "changedConfigurationsByCoreShellDepth": {
                        str(depth): changed_by_depth[depth]
                        for depth in sorted(changed_by_depth)
                    },
                    "selectedPatchSymmetricDifference": len(
                        first_patch_ids ^ second_patch_ids
                    ),
                    "selectedPatchJaccard": round(
                        len(first_patch_ids & second_patch_ids)
                        / max(len(first_patch_ids | second_patch_ids), 1),
                        6,
                    ),
                    "retainedJoinSymmetricDifference": len(
                        first_joins ^ second_joins
                    ),
                    "retainedJoinJaccard": round(
                        len(first_joins & second_joins)
                        / max(len(first_joins | second_joins), 1),
                        6,
                    ),
                }
            )
    payload = {
        "schema": SHEET_HALO_AUDIT_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "ownedGrid": _grid_record(reference_grid),
        "runs": metrics,
        "comparisons": comparisons,
    }
    atomic_json(Path(output_path).resolve(), payload)
    return payload


def _inside_global_core(cell: Int3, start: Int3, stop: Int3) -> bool:
    return all(start[axis] <= cell[axis] < stop[axis] for axis in range(3))


def _configuration_context_terms(
    evidence: BlockSheetEvidence,
    factors: Mapping[str, np.ndarray],
    selection: np.ndarray,
    settings: SheetConfigurationSolverSettings,
    *,
    solve_start_cell_xyz: Int3,
    core_start_cell_xyz: Int3,
    core_stop_cell_xyz_exclusive: Int3,
) -> dict[str, Any]:
    cells = tuple(
        tuple(
            int(row[axis]) + solve_start_cell_xyz[axis] for axis in range(3)
        )
        for row in np.asarray(evidence.arrays["cellXYZ"], dtype=np.int64)
    )
    core = np.asarray(
        [
            _inside_global_core(
                cell, core_start_cell_xyz, core_stop_cell_xyz_exclusive
            )
            for cell in cells
        ],
        dtype=bool,
    )
    unary = _unary_values(evidence, settings)
    terms: dict[str, float | int] = {
        "coreUnaryObjective": 0.0,
        "contextUnaryObjective": 0.0,
        "coreInternalPairwiseObjective": 0.0,
        "coreBoundaryPairwiseObjective": 0.0,
        "contextInternalPairwiseObjective": 0.0,
        "coreInternalMatchedTraces": 0,
        "coreInternalUnmatchedTraceEndpoints": 0,
        "coreBoundaryMatchedTraces": 0,
        "coreBoundaryUnmatchedTraceEndpoints": 0,
    }
    for cell_index, configuration_index in enumerate(selection):
        key = "coreUnaryObjective" if core[cell_index] else "contextUnaryObjective"
        terms[key] = float(terms[key]) + float(unary[int(configuration_index)])
    for face_index, (first_value, second_value) in enumerate(
        zip(factors["firstCellIndex"], factors["secondCellIndex"])
    ):
        first = int(first_value)
        second = int(second_value)
        value, matched, unmatched = _factor_value(
            factors,
            face_index,
            int(selection[first]),
            int(selection[second]),
            settings,
        )
        if core[first] and core[second]:
            terms["coreInternalPairwiseObjective"] = (
                float(terms["coreInternalPairwiseObjective"]) + value
            )
            terms["coreInternalMatchedTraces"] = (
                int(terms["coreInternalMatchedTraces"]) + matched
            )
            terms["coreInternalUnmatchedTraceEndpoints"] = (
                int(terms["coreInternalUnmatchedTraceEndpoints"]) + unmatched
            )
        elif core[first] or core[second]:
            terms["coreBoundaryPairwiseObjective"] = (
                float(terms["coreBoundaryPairwiseObjective"]) + value
            )
            terms["coreBoundaryMatchedTraces"] = (
                int(terms["coreBoundaryMatchedTraces"]) + matched
            )
            terms["coreBoundaryUnmatchedTraceEndpoints"] = (
                int(terms["coreBoundaryUnmatchedTraceEndpoints"]) + unmatched
            )
        else:
            terms["contextInternalPairwiseObjective"] = (
                float(terms["contextInternalPairwiseObjective"]) + value
            )
    terms["coreOwnedObjective"] = (
        float(terms["coreUnaryObjective"])
        + float(terms["coreInternalPairwiseObjective"])
    )
    terms["fullContextObjective"] = sum(
        float(terms[name])
        for name in (
            "coreUnaryObjective",
            "contextUnaryObjective",
            "coreInternalPairwiseObjective",
            "coreBoundaryPairwiseObjective",
            "contextInternalPairwiseObjective",
        )
    )
    terms["coreInternalProjectedRetainedTraceFraction"] = (
        2 * int(terms["coreInternalMatchedTraces"])
        / max(
            2 * int(terms["coreInternalMatchedTraces"])
            + int(terms["coreInternalUnmatchedTraceEndpoints"]),
            1,
        )
    )
    terms["coreBoundaryProjectedRetainedTraceFraction"] = (
        2 * int(terms["coreBoundaryMatchedTraces"])
        / max(
            2 * int(terms["coreBoundaryMatchedTraces"])
            + int(terms["coreBoundaryUnmatchedTraceEndpoints"]),
            1,
        )
    )
    return {
        name: round(value, 6) if isinstance(value, float) else value
        for name, value in terms.items()
    }


def evaluate_sheet_halo_configuration_context(
    experiment_root: str | Path,
) -> dict[str, Any]:
    """Score every owned configuration choice in the largest common context.

    The largest-halo solve supplies the fixed exterior configuration state. For
    each smaller-halo run, only the owned-core choices are substituted. This
    exposes whether a boundary choice that looks good inside the cropped block
    loses more agreement across the artificial cut than it gains internally.
    """

    root = Path(experiment_root).resolve()
    experiment = json.loads((root / "sheet-halo-experiment-v1.json").read_text())
    if experiment.get("state") != "complete":
        raise ValueError("sheet halo experiment is incomplete")
    identity = experiment["identity"]
    settings = SheetConfigurationSolverSettings(
        **identity["configurationSettings"]
    )
    core_start = _triple(identity["coreStartCellXYZ"], "core start")
    core_stop = _triple(identity["coreStopCellXYZExclusive"], "core stop")
    records = {int(value["haloCells"]): value for value in experiment["runs"]}
    reference_halo = max(records)
    reference_record = records[reference_halo]
    reference_root = Path(reference_record["root"]).resolve()
    reference_evidence = read_block_sheet_evidence(
        reference_root / "evidence", verify=True
    )
    reference_factors, _ = _read_factors(reference_root / "factors")
    with np.load(
        reference_root
        / "configurations"
        / "sheet-configuration-selection-v1.npz"
    ) as values:
        reference_selection = np.asarray(
            values["configurationIndex"], dtype=np.int64
        )
    reference_solve_start = _triple(
        reference_record["solveStartCellXYZ"], "reference solve start"
    )
    reference_cells = tuple(
        tuple(
            int(row[axis]) + reference_solve_start[axis] for axis in range(3)
        )
        for row in np.asarray(reference_evidence.arrays["cellXYZ"], dtype=np.int64)
    )
    reference_cell_index = {
        cell: index for index, cell in enumerate(reference_cells)
    }
    configuration_ids = np.asarray(
        reference_evidence.arrays["configurationId"], dtype=np.uint64
    )
    configuration_index_by_id = {
        int(value): index for index, value in enumerate(configuration_ids)
    }
    configuration_offset = np.asarray(
        reference_evidence.arrays["configurationOffset"], dtype=np.uint64
    )
    configuration_cell_index = np.repeat(
        np.arange(reference_evidence.cell_count, dtype=np.int64),
        np.diff(configuration_offset).astype(np.int64),
    )
    results: dict[str, dict[str, Any]] = {}
    for halo, record in sorted(records.items()):
        run_root = Path(record["root"]).resolve()
        solve_start = _triple(record["solveStartCellXYZ"], "solve start")
        with np.load(
            run_root / "configurations" / "sheet-configuration-selection-v1.npz"
        ) as values:
            cells = np.asarray(values["cellXYZ"], dtype=np.int64)
            selected_ids = np.asarray(values["configurationId"], dtype=np.uint64)
        hybrid = reference_selection.copy()
        substituted = 0
        for local_cell, selected_id_value in zip(cells, selected_ids):
            global_cell = tuple(
                int(local_cell[axis]) + solve_start[axis] for axis in range(3)
            )
            if not _inside_global_core(global_cell, core_start, core_stop):
                continue
            try:
                reference_cell = reference_cell_index[global_cell]
                configuration_index = configuration_index_by_id[int(selected_id_value)]
            except KeyError as error:
                raise ValueError(
                    "a smaller-halo stable configuration is absent from the reference bake"
                ) from error
            if int(configuration_cell_index[configuration_index]) != reference_cell:
                raise ValueError("stable configuration ID moved to another global cell")
            hybrid[reference_cell] = configuration_index
            substituted += 1
        expected = int(np.prod(tuple(core_stop[a] - core_start[a] for a in range(3))))
        if substituted != expected:
            raise ValueError("one halo selection does not cover the complete owned core")
        results[str(halo)] = _configuration_context_terms(
            reference_evidence,
            reference_factors,
            hybrid,
            settings,
            solve_start_cell_xyz=reference_solve_start,
            core_start_cell_xyz=core_start,
            core_stop_cell_xyz_exclusive=core_stop,
        )
    reference = results[str(reference_halo)]
    for values in results.values():
        values["coreOwnedObjectiveDeltaFromLargestHalo"] = round(
            float(values["coreOwnedObjective"])
            - float(reference["coreOwnedObjective"]),
            6,
        )
        values["coreBoundaryPairwiseDeltaFromLargestHalo"] = round(
            float(values["coreBoundaryPairwiseObjective"])
            - float(reference["coreBoundaryPairwiseObjective"]),
            6,
        )
        values["fullContextObjectiveDeltaFromLargestHalo"] = round(
            float(values["fullContextObjective"])
            - float(reference["fullContextObjective"]),
            6,
        )
    return {
        "referenceHaloCells": reference_halo,
        "semantics": (
            "each run's owned-core configuration IDs substituted into the "
            "largest-halo exterior state and scored with the largest-halo factors"
        ),
        "runs": results,
    }


def finalize_sheet_halo_experiment(
    experiment_root: str | Path,
    *,
    cluster_root: str | Path | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Replay owned topology after halo removal and write the final audit."""

    root = Path(experiment_root).resolve()
    experiment = json.loads((root / "sheet-halo-experiment-v1.json").read_text())
    if experiment.get("state") != "complete":
        raise ValueError("sheet halo experiment is incomplete")
    identity = experiment["identity"]
    output_path = root / "sheet-halo-final-audit-v1.json"
    if output_path.is_file() and not force:
        payload = json.loads(output_path.read_text())
        if (
            payload.get("experimentIdentitySha256")
            != identity["identitySha256"]
        ):
            raise ValueError("sheet halo final audit belongs to another experiment")
        experiment["finalAudit"] = {
            "path": output_path.name,
            "sha256": sha256_file(output_path),
            "referenceHaloCells": payload["configurationContext"][
                "referenceHaloCells"
            ],
        }
        atomic_json(root / "sheet-halo-experiment-v1.json", experiment)
        return payload
    cluster = Path(cluster_root or identity["clusterRoot"]).resolve()
    settings = SheetStitchingSettings(**identity["stitchingSettings"])
    final_roots: dict[int, Path] = {}
    configuration_roots: dict[int, Path] = {}
    replay_records = []
    for record in sorted(experiment["runs"], key=lambda value: value["haloCells"]):
        halo = int(record["haloCells"])
        run_root = Path(record["root"]).resolve()
        owned = run_root / "owned"
        final = run_root / "owned-restitch"
        if progress is not None:
            progress(f"halo {halo}: replaying topology inside the owned crop")
        summary = run_block_sheet_restitching(
            cluster,
            owned,
            final,
            settings=settings,
            force=force,
        )
        final_roots[halo] = final
        configuration_roots[halo] = owned
        replay_records.append(
            {
                "haloCells": halo,
                "inputRoot": str(owned),
                "outputRoot": str(final),
                "before": summary["restitch"]["baseline"],
                "after": summary["restitch"]["best"],
                "delta": summary["restitch"]["delta"],
                "elapsedSeconds": summary["timingSeconds"]["total"],
            }
        )
    replay_audit_path = root / "sheet-halo-owned-replay-audit-v1.json"
    replay_audit = compare_sheet_halo_outputs(
        final_roots,
        replay_audit_path,
        configuration_roots_by_halo=configuration_roots,
    )
    configuration_context = evaluate_sheet_halo_configuration_context(root)
    cropped_audit_path = root / str(experiment["audit"]["path"])
    cropped_audit = json.loads(cropped_audit_path.read_text())
    payload = {
        "schema": SHEET_HALO_FINAL_AUDIT_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "experimentIdentitySha256": identity["identitySha256"],
        "croppedExpandedTopology": cropped_audit,
        "ownedTopologyReplay": replay_audit,
        "configurationContext": configuration_context,
        "topologyReplayRuns": replay_records,
        "artifacts": {
            "croppedAudit": {
                "path": cropped_audit_path.name,
                "sha256": sha256_file(cropped_audit_path),
            },
            "ownedReplayAudit": {
                "path": replay_audit_path.name,
                "sha256": sha256_file(replay_audit_path),
            },
        },
    }
    atomic_json(output_path, payload)
    experiment["finalAudit"] = {
        "path": output_path.name,
        "sha256": sha256_file(output_path),
        "referenceHaloCells": configuration_context["referenceHaloCells"],
    }
    atomic_json(root / "sheet-halo-experiment-v1.json", experiment)
    return payload


def run_sheet_halo_experiment(
    evidence_root: str | Path,
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    core_start_cell_xyz: Int3,
    core_stop_cell_xyz_exclusive: Int3,
    halo_cells: Iterable[int] = (0, 1, 2),
    configuration_settings: SheetConfigurationSolverSettings | None = None,
    stitching_settings: SheetStitchingSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    """Solve and re-stitch one core under several sheet-inference halos."""

    started = time.monotonic()
    source = Path(evidence_root).resolve()
    cluster = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    evidence = read_block_sheet_evidence(source, verify=True)
    core_start = _triple(core_start_cell_xyz, "core start")
    core_stop = _triple(core_stop_cell_xyz_exclusive, "core stop")
    _validate_bounds(evidence.grid, core_start, core_stop)
    halos = tuple(sorted({int(value) for value in halo_cells}))
    if not halos or any(value < 0 for value in halos):
        raise ValueError("sheet inference halos must be nonnegative")
    solve_bounds: dict[int, tuple[Int3, Int3]] = {}
    for halo in halos:
        start = tuple(core_start[axis] - halo for axis in range(3))
        stop = tuple(core_stop[axis] + halo for axis in range(3))
        _validate_bounds(evidence.grid, start, stop)
        solve_bounds[halo] = start, stop
    resolved_configuration = configuration_settings or SheetConfigurationSolverSettings()
    resolved_stitching = stitching_settings or SheetStitchingSettings(restart_count=4)
    identity: dict[str, Any] = {
        "schema": SHEET_HALO_EXPERIMENT_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "evidenceRoot": str(source),
        "evidenceManifestSha256": sha256_file(source / "sheet-evidence-v1.json"),
        "clusterRoot": str(cluster),
        "clusterManifestSha256": sha256_file(cluster / "cluster-reselection-v1.json"),
        "coreStartCellXYZ": list(core_start),
        "coreStopCellXYZExclusive": list(core_stop),
        "haloCells": list(halos),
        "configurationSettings": resolved_configuration.record(),
        "stitchingSettings": resolved_stitching.record(),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "sheet-halo-experiment-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet halo experiment output belongs to another identity")
        if not force and prior.get("state") == "complete":
            final_audit_path = output / "sheet-halo-final-audit-v1.json"
            if not final_audit_path.is_file():
                if progress is not None:
                    progress("finalizing owned-core topology and context audit")
            final_audit = finalize_sheet_halo_experiment(
                output,
                cluster_root=cluster,
                progress=progress,
            )
            prior["finalAudit"] = {
                "path": final_audit_path.name,
                "sha256": sha256_file(final_audit_path),
                "referenceHaloCells": final_audit["configurationContext"][
                    "referenceHaloCells"
                ],
            }
            atomic_json(manifest_path, prior)
            return prior
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_HALO_EXPERIMENT_SCHEMA,
            "version": SHEET_OWNERSHIP_VERSION,
            "state": "running",
            "identity": identity,
        },
    )

    owned_roots: dict[int, Path] = {}
    run_records = []
    for halo in halos:
        run_started = time.monotonic()
        solve_start, solve_stop = solve_bounds[halo]
        run_root = output / f"halo-{halo}"
        evidence_output = run_root / "evidence"
        correspondence_output = run_root / "correspondences"
        factor_output = run_root / "factors"
        configuration_output = run_root / "configurations"
        graph_output = run_root / "graph"
        owned_output = run_root / "owned"

        if progress is not None:
            progress(f"halo {halo}: extracting exact Acus evidence subblock")
        extract_sheet_evidence_subblock(
            source,
            evidence_output,
            start_cell_xyz=solve_start,
            stop_cell_xyz_exclusive=solve_stop,
            force=force,
        )
        if progress is not None:
            progress(f"halo {halo}: cataloging mode correspondences")
        catalog_block_sheet_correspondences(
            evidence_output,
            cluster,
            correspondence_output,
            force=force,
        )
        if progress is not None:
            progress(f"halo {halo}: compiling physical-stack face factors")
        compile_sheet_configuration_factors(
            evidence_output,
            correspondence_output,
            cluster,
            factor_output,
            settings=SheetFactorSettings(
                quarter_turn_penalty=resolved_stitching.quarter_turn_penalty
            ),
            force=force,
        )
        if progress is not None:
            progress(f"halo {halo}: optimizing cell configurations")
        run_sheet_configuration_initialization(
            evidence_output,
            factor_output,
            configuration_output,
            settings=resolved_configuration,
            force=force,
        )
        if progress is not None:
            progress(f"halo {halo}: replaying global sheet topology")
        replay_joint_sheet_graph(
            evidence_output,
            correspondence_output,
            configuration_output,
            cluster,
            graph_output,
            stitching_settings=resolved_stitching,
            force=force,
        )
        owned_start = tuple(core_start[axis] - solve_start[axis] for axis in range(3))
        owned_stop = tuple(core_stop[axis] - solve_start[axis] for axis in range(3))
        if progress is not None:
            progress(f"halo {halo}: cropping solved halo to the owned core")
        crop_summary = crop_surface_graph_to_owned_block(
            graph_output,
            owned_output,
            start_cell_xyz=owned_start,
            stop_cell_xyz_exclusive=owned_stop,
            force=force,
        )
        owned_roots[halo] = owned_output
        run_records.append(
            {
                "haloCells": halo,
                "solveStartCellXYZ": list(solve_start),
                "solveStopCellXYZExclusive": list(solve_stop),
                "solveShapeCellsXYZ": [
                    solve_stop[axis] - solve_start[axis] for axis in range(3)
                ],
                "owned": crop_summary["summary"]["owned"],
                "elapsedSeconds": round(time.monotonic() - run_started, 6),
                "root": str(run_root),
            }
        )

    audit_path = output / "sheet-halo-audit-v1.json"
    audit = compare_sheet_halo_outputs(owned_roots, audit_path)
    manifest = {
        "schema": SHEET_HALO_EXPERIMENT_SCHEMA,
        "version": SHEET_OWNERSHIP_VERSION,
        "state": "complete",
        "identity": identity,
        "runs": run_records,
        "audit": {
            "path": audit_path.name,
            "sha256": sha256_file(audit_path),
            "summary": audit,
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(manifest_path, manifest)
    if progress is not None:
        progress("finalizing owned-core topology and context audit")
    final_audit = finalize_sheet_halo_experiment(
        output,
        cluster_root=cluster,
        force=force,
        progress=progress,
    )
    final_audit_path = output / "sheet-halo-final-audit-v1.json"
    manifest["finalAudit"] = {
        "path": final_audit_path.name,
        "sha256": sha256_file(final_audit_path),
        "referenceHaloCells": final_audit["configurationContext"][
            "referenceHaloCells"
        ],
    }
    manifest["elapsedSeconds"] = round(time.monotonic() - started, 6)
    atomic_json(manifest_path, manifest)
    return manifest
