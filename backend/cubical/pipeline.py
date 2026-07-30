from __future__ import annotations

import json
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from backend.acus_compute import compute_status

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import (
    RAW_ACUS_PIPELINE_SCHEMA,
    RAW_ACUS_PIPELINE_VERSION,
    RawAcusSettings,
    ReconstructionWindow,
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    pipeline_identity,
    extraction_tiles_for_shard,
    plan_extraction_tiles,
    plan_shards,
    sha256_file,
)
from .evidence import (
    build_cell_evidence,
    read_evidence_artifact,
    write_evidence_artifact,
)
from .export import write_block_obj, write_block_projection_png
from .raw_acus import (
    calibrate_raw_acus,
    extract_tile_needles,
    gather_needle_tables,
    read_calibration,
    read_calibration_reference,
    read_needle_artifact,
    write_calibration,
    write_needle_artifact,
)
from .selection import optimize_configurations
from .stratigraphy import (
    build_configurations_from_modes,
    build_layer_modes,
    read_configuration_artifact,
    read_mode_artifact,
    write_configuration_artifact,
    write_mode_artifact,
)
from .tables import PatchTable, write_patch_shard
from .topology import GridSpec


def _implementation_record() -> dict[str, str]:
    root = Path(__file__).resolve().parent
    names = (
        "pipeline.py",
        "contracts.py",
        "raw_acus.py",
        "evidence.py",
        "stratigraphy.py",
        "selection.py",
        "geometry.py",
        "topology.py",
        "matching.py",
        "block.py",
        "export.py",
        "tables.py",
    )
    result = {name: sha256_file(root / name) for name in names}
    result["backend/acus.py"] = sha256_file(root.parent / "acus.py")
    result["backend/acus_compute.py"] = sha256_file(root.parent / "acus_compute.py")
    return result


def _resolved_identity(
    source: VolumeSource,
    window: ReconstructionWindow,
    settings: RawAcusSettings,
    shard_shape_cells_xyz: Iterable[int],
    compute: str,
    calibration_reference: Path | None = None,
) -> dict[str, Any]:
    identity = pipeline_identity(source, window, settings, shard_shape_cells_xyz)
    identity.pop("identitySha256", None)
    identity["compute"] = {
        "requested": compute,
        "statusAtStart": compute_status(),
    }
    identity["implementationSha256"] = _implementation_record()
    if calibration_reference is not None:
        identity["calibrationReference"] = {
            "path": str(calibration_reference.resolve()),
            "sha256": sha256_file(calibration_reference),
        }
    identity["identitySha256"] = canonical_json_hash(identity)
    return identity


def _processing_bounds(shards: Iterable[Any]) -> VoxelBounds:
    values = list(shards)
    return VoxelBounds(
        tuple(
            min(value.raw_voxel_bounds.start_xyz[axis] for value in values)
            for axis in range(3)
        ),
        tuple(
            max(value.raw_voxel_bounds.stop_xyz_exclusive[axis] for value in values)
            for axis in range(3)
        ),
    )


def _evidence_processing_bounds(shards: Iterable[Any]) -> VoxelBounds:
    values = list(shards)
    return VoxelBounds(
        tuple(
            min(value.evidence_voxel_bounds.start_xyz[axis] for value in values)
            for axis in range(3)
        ),
        tuple(
            max(value.evidence_voxel_bounds.stop_xyz_exclusive[axis] for value in values)
            for axis in range(3)
        ),
    )


def _artifact_ready(prefix: Path, identity_sha256: str, schema: str) -> bool:
    manifest_path = prefix.with_suffix(".json")
    data_path = prefix.with_suffix(".npz")
    if not manifest_path.is_file() or not data_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema") == schema
        and manifest.get("identitySha256") == identity_sha256
        and manifest.get("data", {}).get("sha256") == sha256_file(data_path)
    )


def write_selection_artifact(
    output: Path,
    selection: Any,
    identity_sha256: str,
) -> dict[str, Any]:
    data_path = output / "selection-v1.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    options = selection.selected_options
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=np.asarray([value.cell_xyz for value in options], dtype=np.int32),
            optionId=np.asarray([value.option_id for value in options], dtype=np.uint64),
            sourceTableIndex=np.asarray(
                [value.source_table_index for value in options], dtype=np.uint32
            ),
            sourceConfigurationIndex=np.asarray(
                [value.source_configuration_index for value in options], dtype=np.uint32
            ),
            localConfigurationId=np.asarray(
                [value.local_configuration_id for value in options], dtype=np.uint16
            ),
            configurationLogWeight=np.asarray(
                [value.log_weight for value in options], dtype=np.float32
            ),
            selectedLayerCount=np.asarray(
                [len(value.patches) for value in options], dtype=np.uint16
            ),
        )
    temporary.replace(data_path)
    manifest = {
        "schema": "pareidolia.raw-acus-configuration-selection",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": {
            "cellCount": len(options),
            "nonemptyCellCount": sum(bool(value.patches) for value in options),
            "selectedLayerCount": len(selection.patches),
            "sweeps": selection.sweeps,
            "changedLastSweep": selection.changed_last_sweep,
            "unaryEnergy": selection.unary_energy,
            "pairwiseEnergy": selection.pairwise_energy,
            "pairwiseReward": selection.pairwise_reward,
            "continuationEnergy": selection.continuation_energy,
            "totalEnergy": selection.total_energy,
            "pairwiseEvaluationCount": selection.pairwise_evaluation_count,
            "traceMatchEvaluationCount": selection.trace_match_evaluation_count,
            "interiorUnmatchedTraceCount": selection.interior_unmatched_trace_count,
            "degenerateLayerAlternatives": selection.degenerate_layer_count,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(output / "selection-v1.json", manifest)
    return manifest


def patch_table_from_options(
    grid: GridSpec,
    configuration_tables: list[Any],
    selected_options: Iterable[Any],
) -> PatchTable:
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    options = tuple(selected_options)
    patches = tuple(patch for option in options for patch in option.patches)
    for option in options:
        table = configuration_tables[option.source_table_index]
        family = int(table.normal_hypothesis[option.source_configuration_index])
        for order, patch in enumerate(option.patches):
            configuration_id[patch.patch_id] = option.option_id
            configuration_log_weight[patch.patch_id] = option.log_weight
            local_order[patch.patch_id] = order
            normal_family[patch.patch_id] = family
    return PatchTable.from_patches(
        grid,
        patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )


def patch_table_from_selection(
    grid: GridSpec,
    configuration_tables: list[Any],
    selection: Any,
) -> PatchTable:
    return patch_table_from_options(
        grid, configuration_tables, selection.selected_options
    )


def _quantiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p90": None, "maximum": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), (0, 50, 90, 100))
    return {
        name: round(float(value), 3)
        for name, value in zip(("minimum", "median", "p90", "maximum"), result)
    }


def _aggregate_completed_shards(shard_records: dict[str, Any]) -> dict[str, int]:
    completed = [
        value for value in shard_records.values() if value.get("state") == "complete"
    ]
    return {
        "completedShards": len(completed),
        "shardNeedleOccurrences": sum(
            int(value.get("needleCount", 0)) for value in completed
        ),
        "validCells": sum(int(value.get("validCellCount", 0)) for value in completed),
        "normalHypotheses": sum(
            int(value.get("normalHypothesisCount", 0)) for value in completed
        ),
        "candidateModes": sum(
            int(value.get("candidateModeCount", 0)) for value in completed
        ),
        "configurations": sum(
            int(value.get("configurationCount", 0)) for value in completed
        ),
        "layerAlternatives": sum(
            int(value.get("layerAlternativeCount", 0)) for value in completed
        ),
    }


def _aggregate_completed_tiles(tile_records: dict[str, Any]) -> dict[str, int]:
    completed = [
        value for value in tile_records.values() if value.get("state") == "complete"
    ]
    return {
        "completedExtractionTiles": len(completed),
        "canonicalTileCandidates": sum(
            int(value.get("needleCount", 0)) for value in completed
        ),
    }


def _aggregate_local(manifest: dict[str, Any]) -> dict[str, int]:
    return {
        **_aggregate_completed_tiles(manifest["extractionTiles"]),
        **_aggregate_completed_shards(manifest["shards"]),
    }


def _raw_acus_counts(aggregate: dict[str, int]) -> dict[str, int]:
    return {
        key: value
        for key, value in aggregate.items()
        if key not in ("completedExtractionTiles", "completedShards")
    }


def run_raw_acus_pipeline(
    source_path: str | Path,
    output_root: str | Path,
    window: ReconstructionWindow,
    *,
    metadata_path: str | Path | None = None,
    calibration_path: str | Path | None = None,
    settings: RawAcusSettings | None = None,
    shard_shape_cells_xyz: Iterable[int] = (4, 4, 4),
    leaf_shape_cells_xyz: Iterable[int] = (4, 4, 4),
    compute: str = "auto",
    force: bool = False,
    maximum_preview_components: int = 128,
    local_only: bool = False,
    only_shard_ids: Iterable[str] | None = None,
    limit_shards: int | None = None,
) -> dict[str, Any]:
    """Bake a cubical reconstruction from native CT voxels, end to end."""

    if compute not in ("auto", "cpu", "gpu"):
        raise ValueError("compute must be auto, cpu, or gpu")
    if maximum_preview_components <= 0:
        raise ValueError("maximum preview components must be positive")
    if limit_shards is not None and limit_shards <= 0:
        raise ValueError("limit_shards must be positive")
    os.environ["ACUS_COMPUTE"] = compute
    resolved_settings = settings or RawAcusSettings()
    calibration_reference = (
        None if calibration_path is None else Path(calibration_path).resolve()
    )
    source = VolumeSource.open(source_path, metadata_path)
    window.validate(source, resolved_settings)
    shards = plan_shards(window, resolved_settings, shard_shape_cells_xyz)
    extraction_tiles = plan_extraction_tiles(
        source,
        _evidence_processing_bounds(shards),
        resolved_settings,
    )
    identity = _resolved_identity(
        source,
        window,
        resolved_settings,
        shard_shape_cells_xyz,
        compute,
        calibration_reference,
    )
    identity_sha256 = str(identity["identitySha256"])
    output = Path(output_root)
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / "pipeline.json"
    if manifest_path.is_file():
        existing = json.loads(manifest_path.read_text())
        existing_identity = existing.get("identity", {}).get("identitySha256")
        if existing_identity != identity_sha256:
            raise ValueError(
                "output root belongs to a different source/configuration/code identity; "
                "choose a new output root"
            )
    manifest: dict[str, Any] = {
        "schema": RAW_ACUS_PIPELINE_SCHEMA,
        "version": RAW_ACUS_PIPELINE_VERSION,
        "identity": identity,
        "state": "created",
        "computeRequested": compute,
        "sourceGeometry": {
            "sourceOriginXYZ": list(source.origin_xyz),
            "voxelSizeMicrons": source.voxel_size_microns,
        },
        "shardCount": len(shards),
        "shards": {value.shard_id: {"state": "pending"} for value in shards},
        "extractionTileCount": len(extraction_tiles),
        "extractionTiles": {
            value.tile_id: {"state": "pending"} for value in extraction_tiles
        },
    }
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        manifest["shards"].update(previous.get("shards", {}))
        manifest["extractionTiles"].update(previous.get("extractionTiles", {}))
    atomic_json(manifest_path, manifest)
    pipeline_started = time.monotonic()

    requested_shards = set(only_shard_ids or ())
    known_shards = {value.shard_id for value in shards}
    unknown_shards = requested_shards - known_shards
    if unknown_shards:
        raise ValueError(f"unknown shard ids: {sorted(unknown_shards)}")
    targeted_mode = bool(requested_shards) or limit_shards is not None or local_only
    if requested_shards:
        target_shards = [value for value in shards if value.shard_id in requested_shards]
    elif targeted_mode:
        target_shards = [
            value
            for value in shards
            if force or manifest["shards"][value.shard_id].get("state") != "complete"
        ]
    else:
        target_shards = list(shards)
    if limit_shards is not None:
        target_shards = target_shards[:limit_shards]

    local_calibration_path = output / "calibration-v1.json"
    if local_calibration_path.is_file() and not force:
        calibration = read_calibration(
            local_calibration_path, identity_sha256=identity_sha256
        )
    elif calibration_reference is not None:
        calibration = read_calibration_reference(calibration_reference)
        write_calibration(
            local_calibration_path,
            calibration,
            identity_sha256=identity_sha256,
        )
    else:
        manifest["state"] = "calibrating"
        atomic_json(manifest_path, manifest)
        calibration = calibrate_raw_acus(
            source, _processing_bounds(shards), resolved_settings
        )
        write_calibration(
            local_calibration_path,
            calibration,
            identity_sha256=identity_sha256,
        )
    manifest["calibration"] = calibration.record()
    manifest["state"] = "extracting"
    atomic_json(manifest_path, manifest)

    shards_needing_needles = [
        shard
        for shard in target_shards
        if force
        or not _artifact_ready(
            output / "shards" / shard.shard_id / "needles-v1",
            identity_sha256,
            "pareidolia.raw-acus-needles",
        )
    ]
    required_tile_ids = {
        tile.tile_id
        for shard in shards_needing_needles
        for tile in extraction_tiles_for_shard(extraction_tiles, shard)
    }
    required_tiles = [
        tile for tile in extraction_tiles if tile.tile_id in required_tile_ids
    ]
    for tile_number, tile in enumerate(required_tiles, start=1):
        tile_prefix = output / "extraction-tiles" / tile.tile_id / "needles-v1"
        tile_state = manifest["extractionTiles"][tile.tile_id]
        if (
            not force
            and _artifact_ready(
                tile_prefix,
                identity_sha256,
                "pareidolia.raw-acus-needles",
            )
        ):
            tile_manifest = json.loads(tile_prefix.with_suffix(".json").read_text())
            tile_state["state"] = "complete"
            tile_state["needleCount"] = tile_manifest["counts"]["needles"]
            continue
        tile_state["state"] = "extracting"
        atomic_json(manifest_path, manifest)
        tile_needles, tile_compute = extract_tile_needles(
            source, tile, calibration, resolved_settings
        )
        write_needle_artifact(
            tile_prefix,
            tile_needles,
            identity_sha256=identity_sha256,
            shard=tile,
            compute_metadata=tile_compute,
        )
        tile_state.update(
            {
                "state": "complete",
                "needleCount": tile_needles.count,
                "computeBackend": tile_compute.get("backend"),
                "computeDevice": tile_compute.get("device"),
            }
        )
        manifest["completedExtractionTileCount"] = sum(
            value.get("state") == "complete"
            for value in manifest["extractionTiles"].values()
        )
        atomic_json(manifest_path, manifest)
        print(
            f"Acus extraction tile {tile_number}/{len(required_tiles)} {tile.tile_id} · "
            f"{tile_needles.count} candidates",
            flush=True,
        )

    configuration_tables = []
    for shard_number, shard in enumerate(target_shards, start=1):
        shard_root = output / "shards" / shard.shard_id
        shard_root.mkdir(parents=True, exist_ok=True)
        shard_state = manifest["shards"][shard.shard_id]
        needle_prefix = shard_root / "needles-v1"
        if (
            not force
            and _artifact_ready(
                needle_prefix,
                identity_sha256,
                "pareidolia.raw-acus-needles",
            )
        ):
            needles = read_needle_artifact(
                needle_prefix, identity_sha256=identity_sha256, verify=False
            )
            needle_metadata = json.loads(needle_prefix.with_suffix(".json").read_text())[
                "compute"
            ]
        else:
            shard_state["state"] = "extracting"
            atomic_json(manifest_path, manifest)
            shard_tiles = extraction_tiles_for_shard(extraction_tiles, shard)
            tile_tables = [
                read_needle_artifact(
                    output
                    / "extraction-tiles"
                    / tile.tile_id
                    / "needles-v1",
                    identity_sha256=identity_sha256,
                    verify=False,
                )
                for tile in shard_tiles
            ]
            needles = gather_needle_tables(
                tile_tables, shard.evidence_voxel_bounds
            )
            needle_metadata = {
                "backend": "canonical-tile-aggregate",
                "device": calibration.compute_device,
                "fallbackReason": None,
                "sourceTileIds": [tile.tile_id for tile in shard_tiles],
                "sourceTileNeedleCount": sum(value.count for value in tile_tables),
                "retainedCandidateCount": needles.count,
            }
            write_needle_artifact(
                needle_prefix,
                needles,
                identity_sha256=identity_sha256,
                shard=shard,
                compute_metadata=needle_metadata,
            )
        shard_state["needleCount"] = needles.count
        shard_state["computeBackend"] = needle_metadata.get("backend")
        shard_state["computeDevice"] = needle_metadata.get("device")

        evidence_prefix = shard_root / "evidence-v1"
        if (
            not force
            and _artifact_ready(
                evidence_prefix,
                identity_sha256,
                "pareidolia.raw-acus-cell-evidence",
            )
        ):
            evidence = read_evidence_artifact(
                evidence_prefix, identity_sha256=identity_sha256, verify=False
            )
            evidence_statistics = json.loads(
                evidence_prefix.with_suffix(".json").read_text()
            )["statistics"]
        else:
            shard_state["state"] = "evidence"
            atomic_json(manifest_path, manifest)
            evidence, evidence_statistics = build_cell_evidence(
                source,
                window,
                shard,
                needles,
                calibration,
                resolved_settings,
            )
            write_evidence_artifact(
                evidence_prefix,
                evidence,
                identity_sha256=identity_sha256,
                shard=shard,
                statistics=evidence_statistics,
            )
        shard_state["validCellCount"] = evidence_statistics["validCellCount"]
        shard_state["normalHypothesisCount"] = evidence_statistics[
            "normalHypothesisCount"
        ]

        mode_prefix = shard_root / "modes-v1"
        if (
            not force
            and _artifact_ready(
                mode_prefix,
                identity_sha256,
                "pareidolia.raw-acus-layer-modes",
            )
        ):
            modes = read_mode_artifact(
                mode_prefix, identity_sha256=identity_sha256, verify=False
            )
            mode_statistics = json.loads(
                mode_prefix.with_suffix(".json").read_text()
            )["statistics"]
        else:
            shard_state["state"] = "modes"
            atomic_json(manifest_path, manifest)
            modes, mode_statistics = build_layer_modes(
                needles, evidence, resolved_settings
            )
            write_mode_artifact(
                mode_prefix,
                modes,
                identity_sha256=identity_sha256,
                shard=shard,
                statistics=mode_statistics,
            )
        shard_state["candidateModeCount"] = mode_statistics[
            "candidateModeCount"
        ]

        configuration_prefix = shard_root / "stratigraphies-v1"
        if (
            not force
            and _artifact_ready(
                configuration_prefix,
                identity_sha256,
                "pareidolia.raw-acus-stratigraphies",
            )
        ):
            configurations = read_configuration_artifact(
                configuration_prefix,
                identity_sha256=identity_sha256,
                verify=False,
            )
            configuration_statistics = json.loads(
                configuration_prefix.with_suffix(".json").read_text()
            )["statistics"]
        else:
            shard_state["state"] = "stratigraphy"
            atomic_json(manifest_path, manifest)
            configurations, configuration_statistics = (
                build_configurations_from_modes(
                    source, modes, evidence, resolved_settings
                )
            )
            write_configuration_artifact(
                configuration_prefix,
                configurations,
                identity_sha256=identity_sha256,
                shard=shard,
                statistics=configuration_statistics,
            )
        configuration_tables.append(configurations)
        shard_state.update(
            {
                "state": "complete",
                "configurationCount": configuration_statistics[
                    "configurationCount"
                ],
                "layerAlternativeCount": configuration_statistics[
                    "layerAlternativeCount"
                ],
            }
        )
        aggregate = _aggregate_local(manifest)
        manifest["completedShardCount"] = aggregate["completedShards"]
        manifest["completedExtractionTileCount"] = aggregate[
            "completedExtractionTiles"
        ]
        manifest["aggregate"] = aggregate
        atomic_json(manifest_path, manifest)
        print(
            f"raw Acus shard {shard_number}/{len(target_shards)} {shard.shard_id} · "
            f"{needles.count} needles · {evidence_statistics['validCellCount']} valid cells · "
            f"{mode_statistics['candidateModeCount']} modes · "
            f"{configuration_statistics['layerAlternativeCount']} layer alternatives",
            flush=True,
        )

    aggregate = _aggregate_local(manifest)
    all_local_complete = aggregate["completedShards"] == len(shards)
    if targeted_mode:
        manifest["state"] = "local-complete" if all_local_complete else "local-partial"
        manifest["completedShardCount"] = aggregate["completedShards"]
        manifest["completedExtractionTileCount"] = aggregate[
            "completedExtractionTiles"
        ]
        manifest["aggregate"] = aggregate
        manifest["elapsedSecondsThisRun"] = round(
            time.monotonic() - pipeline_started, 6
        )
        atomic_json(manifest_path, manifest)
        local_summary: dict[str, Any] = {
            "schema": "pareidolia.raw-acus-local-bake-summary",
            "version": 1,
            "identitySha256": identity_sha256,
            "state": manifest["state"],
            "contract": {
                "input": "native uint8 CT voxels",
                "output": "independently resumable needle, evidence, mode-bank, and top-M stratigraphy shards",
                "globalSelectionPerformed": False,
            },
            "shards": {
                "completed": aggregate["completedShards"],
                "total": len(shards),
                "processedThisRun": len(target_shards),
                "targetIds": [value.shard_id for value in target_shards],
            },
            "extractionTiles": {
                "completed": aggregate["completedExtractionTiles"],
                "total": len(extraction_tiles),
            },
            "rawAcus": _raw_acus_counts(aggregate),
            "timingSeconds": {
                "pipelineThisRun": manifest["elapsedSecondsThisRun"]
            },
            "next": (
                "run without --local-only/--only-shard/--limit-shards to finalize a window"
                if all_local_complete
                else "continue the local bake until every shard is complete"
            ),
        }
        atomic_json(output / "local-summary.json", local_summary)
        return local_summary

    if not all_local_complete:
        raise RuntimeError("global selection requires every local shard to be complete")

    manifest["state"] = "selecting"
    atomic_json(manifest_path, manifest)
    grid = GridSpec(
        window.shape_cells_xyz,
        cell_size_xyz=(
            float(resolved_settings.cell_stride_voxels),
            float(resolved_settings.cell_stride_voxels),
            float(resolved_settings.cell_stride_voxels),
        ),
        origin_xyz=tuple(
            float(source.origin_xyz[axis] + window.origin_voxel_xyz[axis])
            for axis in range(3)
        ),
        coordinate_unit="source-voxel",
    )
    selection_started = time.monotonic()
    selection = optimize_configurations(grid, configuration_tables)
    selection_seconds = time.monotonic() - selection_started
    selection_manifest = write_selection_artifact(
        output, selection, identity_sha256
    )

    selected_table = patch_table_from_selection(
        grid, configuration_tables, selection
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        selected_table,
        settings={
            "source": "raw Acus top-M stratigraphy selection",
            "selection": "ICM over unary posterior and shared-face relative likelihood",
        },
        provenance={"pipelineIdentitySha256": identity_sha256},
        compressed=True,
    )

    manifest["state"] = "assembling"
    atomic_json(manifest_path, manifest)
    assembly_started = time.monotonic()
    block = assemble_surface_hierarchy(
        grid,
        BlockBounds((0, 0, 0), grid.shape_cells_xyz),
        selection.patches,
        maximum_leaf_shape_cells_xyz=tuple(
            int(value) for value in leaf_shape_cells_xyz
        ),
    )
    assembly_seconds = time.monotonic() - assembly_started
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block,
        output / "projections.png",
        maximum_components=maximum_preview_components,
    )
    largest_component_path = write_block_projection_png(
        block,
        output / "largest-component.png",
        maximum_components=1,
    )
    deferred = defaultdict(int)
    for value in block.deferred_joins:
        deferred[value.reason] += 1
    summary: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-reconstruction-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "contract": {
            "input": "native uint8 CT voxels",
            "forbiddenLegacyInputs": [
                "flakes-*.json",
                "legacy sheetlet graphs",
                "legacy component identities",
                "persisted legacy needle catalogs",
            ],
            "ownership": "disjoint cubical cells with overlapping raw evidence halos",
            "directions": "all normals and fibers are axial/unsigned",
            "localInference": "persistent fitted mode bank, then top-M physical stratigraphies",
            "globalInference": "configuration-aware shared-face selection, then topology-safe hierarchical assembly",
        },
        "grid": patch_manifest["grid"],
        "rawAcus": _raw_acus_counts(aggregate),
        "selection": selection_manifest["statistics"],
        "assembly": {
            "candidateJoins": len(block.candidate_joins),
            "retainedJoins": len(block.joins),
            "deferredJoins": len(block.deferred_joins),
            "deferredByReason": dict(sorted(deferred.items())),
            "components": len(block.components),
            "componentPatchCount": _quantiles(
                [len(value.patch_ids) for value in block.components]
            ),
            "largestComponentPatchCount": max(
                (len(value.patch_ids) for value in block.components), default=0
            ),
            "weldedCrossings": len(block.welded_crossings),
            "exteriorTraces": len(block.exterior_traces),
            "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
        },
        "timingSeconds": {
            "selection": round(selection_seconds, 6),
            "assembly": round(assembly_seconds, 6),
            "pipelineThisRun": round(time.monotonic() - pipeline_started, 6),
        },
        "artifacts": {
            "pipeline": manifest_path.name,
            "calibration": calibration_path.name,
            "selection": "selection-v1.npz",
            "selectedPatches": "selected-patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponentProjection": largest_component_path.name,
            "projectionOrder": ["XY", "XZ", "YZ"],
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSecondsThisRun"] = summary["timingSeconds"]["pipelineThisRun"]
    atomic_json(manifest_path, manifest)
    return summary
