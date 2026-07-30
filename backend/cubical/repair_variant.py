from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import write_block_obj, write_block_projection_png
from .gaps import analyze_component_gaps, write_gap_census
from .pipeline import patch_table_from_options
from .repair import apply_recommended_gap_repairs, read_gap_repair_search
from .selection import configuration_options
from .stratigraphy import read_configuration_artifact
from .tables import read_patch_shard, write_patch_shard


GAP_REPAIR_VARIANT_SCHEMA = "pareidolia.raw-acus-gap-repair-variant"
GAP_REPAIR_VARIANT_VERSION = 1


def _variant_identity(
    input_identity: str, applied: list[dict[str, Any]]
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": GAP_REPAIR_VARIANT_SCHEMA,
        "version": GAP_REPAIR_VARIANT_VERSION,
        "inputPipelineIdentitySha256": input_identity,
        "appliedRepairs": applied,
        "implementationSha256": {
            name: sha256_file(root / name)
            for name in (
                "repair_variant.py",
                "repair.py",
                "gaps.py",
                "selection.py",
                "matching.py",
                "block.py",
                "geometry.py",
                "tables.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _block_statistics(block: Any) -> dict[str, Any]:
    deferred = Counter(value.reason for value in block.deferred_joins)
    sizes = [len(value.patch_ids) for value in block.components]
    return {
        "selectedPatches": len(block.patches),
        "candidateJoins": len(block.candidate_joins),
        "retainedJoins": len(block.joins),
        "deferredJoins": len(block.deferred_joins),
        "deferredByReason": dict(sorted(deferred.items())),
        "components": len(block.components),
        "largestComponentPatchCount": max(sizes, default=0),
        "componentsAtLeast": {
            str(limit): sum(value >= limit for value in sizes)
            for limit in (10, 25, 50, 100, 150)
        },
        "exteriorTraces": len(block.exterior_traces),
        "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
    }


def _delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    names = (
        "selectedPatches",
        "candidateJoins",
        "retainedJoins",
        "deferredJoins",
        "components",
        "largestComponentPatchCount",
        "exteriorTraces",
        "unresolvedInteriorTraces",
    )
    return {name: int(after[name]) - int(before[name]) for name in names}


def run_gap_repair_variant(
    input_root: str | Path,
    search_path: str | Path,
    output_root: str | Path,
    *,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    force: bool = False,
) -> dict[str, Any]:
    """Apply conservative search winners and preserve a complete audit trail."""

    started = time.monotonic()
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output == root:
        raise ValueError("gap repair variant output must differ from its input")
    pipeline_manifest = json.loads((root / "pipeline.json").read_text())
    if pipeline_manifest.get("state") != "complete":
        raise ValueError("gap repair requires a completed raw Acus pipeline")
    input_identity = str(pipeline_manifest["identity"]["identitySha256"])
    search = read_gap_repair_search(search_path, identity_sha256=input_identity)
    planned = []
    planned_cells: set[tuple[int, int, int]] = set()
    for value in search.trials:
        if not value.recommended or value.target_cell_xyz in planned_cells:
            continue
        planned.append(value.record())
        planned_cells.add(value.target_cell_xyz)
    identity = _variant_identity(input_identity, planned)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("gap repair output belongs to another repair identity")
        if existing.get("state") == "complete" and (output / "summary.json").is_file():
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": GAP_REPAIR_VARIANT_SCHEMA,
        "version": GAP_REPAIR_VARIANT_VERSION,
        "state": "loading",
        "identity": identity,
        "inputRoot": str(root),
        "search": str(Path(search_path).resolve()),
    }
    atomic_json(manifest_path, manifest)
    tables = [
        read_configuration_artifact(
            root / "shards" / shard_id / "stratigraphies-v1",
            identity_sha256=input_identity,
        )
        for shard_id in pipeline_manifest["shards"]
    ]
    baseline_table = read_patch_shard(root / "selected-patches-v1")
    grid = baseline_table.grid
    baseline_block = assemble_surface_hierarchy(
        grid,
        BlockBounds((0, 0, 0), grid.shape_cells_xyz),
        baseline_table.to_patches(),
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
    )
    options_by_cell, _ = configuration_options(grid, tables)
    with np.load(root / "selection-v1.npz") as values:
        selected_option_ids = {
            tuple(int(item) for item in cell): int(option_id)
            for cell, option_id in zip(values["cellXYZ"], values["optionId"])
        }
    manifest["state"] = "applying"
    atomic_json(manifest_path, manifest)
    application = apply_recommended_gap_repairs(
        baseline_block,
        options_by_cell,
        selected_option_ids,
        search,
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
    )
    applied_by_cell = {
        value.target_cell_xyz: value.candidate_option_id
        for value in application.applied_trials
    }
    repaired_option_ids = dict(selected_option_ids)
    repaired_option_ids.update(applied_by_cell)
    selected_table = patch_table_from_options(
        grid, tables, application.selected_options
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        selected_table,
        settings={
            "source": "topology-validated conservative gap repair",
            "leafShapeCellsXYZ": list(leaf_shape_cells_xyz),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputPipelineIdentitySha256": input_identity,
        },
        compressed=True,
    )
    selection_path = output / "repair-selection-v1.npz"
    temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
    ordered_cells = sorted(
        repaired_option_ids, key=lambda value: (value[2], value[1], value[0])
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=np.asarray(ordered_cells, dtype=np.int32),
            optionId=np.asarray(
                [repaired_option_ids[value] for value in ordered_cells],
                dtype=np.uint64,
            ),
            repaired=np.asarray(
                [value in applied_by_cell for value in ordered_cells], dtype=np.uint8
            ),
        )
    temporary.replace(selection_path)
    block = application.block
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block, output / "projections.png", maximum_components=128
    )
    largest_path = write_block_projection_png(
        block, output / "largest-component.png", maximum_components=1
    )
    top_four_path = write_block_projection_png(
        block, output / "top-4-components.png", maximum_components=4
    )
    top_twelve_path = write_block_projection_png(
        block, output / "top-12-components.png", maximum_components=12
    )
    census = analyze_component_gaps(
        block, options_by_cell, repaired_option_ids
    )
    census_manifest = write_gap_census(
        output / "gap-census-v1.json",
        census,
        identity_sha256=identity_sha256,
        provenance={
            "inputPipelineIdentitySha256": input_identity,
            "repairVariantIdentitySha256": identity_sha256,
        },
    )
    before = _block_statistics(baseline_block)
    after = _block_statistics(block)
    summary: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-gap-repair-variant-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputPipelineIdentitySha256": input_identity,
        "grid": patch_manifest["grid"],
        "appliedRepairs": [value.record() for value in application.applied_trials],
        "verifiedClosedGapCount": application.verified_closed_gap_count,
        "baseline": before,
        "repaired": after,
        "delta": _delta(after, before),
        "largestComponentGapCensus": census_manifest["statistics"],
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "repairSelection": selection_path.name,
            "selectedPatches": "selected-patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponent": largest_path.name,
            "topFourComponents": top_four_path.name,
            "topTwelveComponents": top_twelve_path.name,
            "gapCensus": "gap-census-v1.json",
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
