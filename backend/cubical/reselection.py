from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import write_block_obj, write_block_projection_png
from .gaps import analyze_component_gaps, write_gap_census
from .pipeline import patch_table_from_selection, write_selection_artifact
from .selection import configuration_options, optimize_configurations
from .stratigraphy import ConfigurationTable, read_configuration_artifact
from .tables import read_patch_shard, write_patch_shard


SELECTION_VARIANT_SCHEMA = "pareidolia.raw-acus-selection-variant"
SELECTION_VARIANT_VERSION = 1


@dataclass(frozen=True, slots=True)
class SelectionVariantSettings:
    interior_unmatched_trace_penalty: float = 0.0
    unary_scale: float = 1.0
    pairwise_scale: float = 0.35
    maximum_sweeps: int = 12
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3)
    maximum_preview_components: int = 128

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.interior_unmatched_trace_penalty)
            or self.interior_unmatched_trace_penalty < 0.0
        ):
            raise ValueError("interior unmatched-trace penalty must be nonnegative")
        if self.unary_scale <= 0.0 or self.pairwise_scale <= 0.0:
            raise ValueError("selection energy scales must be positive")
        if self.maximum_sweeps <= 0 or self.maximum_preview_components <= 0:
            raise ValueError("selection iteration and preview limits must be positive")
        shape = tuple(int(value) for value in self.leaf_shape_cells_xyz)
        if len(shape) != 3 or any(value <= 0 for value in shape):
            raise ValueError("leaf cell shape must be a positive XYZ triple")
        object.__setattr__(self, "leaf_shape_cells_xyz", shape)

    def record(self) -> dict[str, Any]:
        result = asdict(self)
        result["leaf_shape_cells_xyz"] = list(self.leaf_shape_cells_xyz)
        return result


def _variant_identity(
    input_identity_sha256: str, settings: SelectionVariantSettings
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    implementation = {
        name: sha256_file(root / name)
        for name in (
            "reselection.py",
            "selection.py",
            "gaps.py",
            "matching.py",
            "block.py",
            "geometry.py",
            "tables.py",
        )
    }
    payload: dict[str, Any] = {
        "schema": SELECTION_VARIANT_SCHEMA,
        "version": SELECTION_VARIANT_VERSION,
        "inputPipelineIdentitySha256": input_identity_sha256,
        "settings": settings.record(),
        "implementationSha256": implementation,
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _load_configuration_tables(
    root: Path, pipeline_manifest: dict[str, Any], identity_sha256: str
) -> list[ConfigurationTable]:
    return [
        read_configuration_artifact(
            root / "shards" / shard_id / "stratigraphies-v1",
            identity_sha256=identity_sha256,
        )
        for shard_id in pipeline_manifest["shards"]
    ]


def _quantiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p90": None, "maximum": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), (0, 50, 90, 100))
    return {
        name: round(float(value), 3)
        for name, value in zip(("minimum", "median", "p90", "maximum"), result)
    }


def run_selection_variant(
    input_root: str | Path,
    output_root: str | Path,
    settings: SelectionVariantSettings,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Reselect immutable raw-Acus stratigraphies under one explicit prior."""

    started = time.monotonic()
    source_root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output == source_root:
        raise ValueError("selection variant output must differ from its input root")
    pipeline_manifest = json.loads((source_root / "pipeline.json").read_text())
    if pipeline_manifest.get("state") != "complete":
        raise ValueError("selection variants require a completed raw Acus pipeline")
    input_identity = str(pipeline_manifest["identity"]["identitySha256"])
    identity = _variant_identity(input_identity, settings)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file() and not force:
        existing = json.loads(manifest_path.read_text())
        if existing.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("selection variant root belongs to different settings/code")
        summary_path = output / "summary.json"
        if existing.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SELECTION_VARIANT_SCHEMA,
        "version": SELECTION_VARIANT_VERSION,
        "state": "loading",
        "identity": identity,
        "inputRoot": str(source_root),
    }
    atomic_json(manifest_path, manifest)

    tables = _load_configuration_tables(
        source_root, pipeline_manifest, input_identity
    )
    baseline_table = read_patch_shard(source_root / "selected-patches-v1")
    grid = baseline_table.grid
    manifest["state"] = "selecting"
    atomic_json(manifest_path, manifest)
    selection_started = time.monotonic()
    selection = optimize_configurations(
        grid,
        tables,
        unary_scale=settings.unary_scale,
        pairwise_scale=settings.pairwise_scale,
        interior_unmatched_trace_penalty=(
            settings.interior_unmatched_trace_penalty
        ),
        maximum_sweeps=settings.maximum_sweeps,
    )
    selection_seconds = time.monotonic() - selection_started
    selection_manifest = write_selection_artifact(
        output, selection, identity_sha256
    )
    selected_table = patch_table_from_selection(grid, tables, selection)
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        selected_table,
        settings={
            "source": "raw Acus top-M stratigraphy selection variant",
            **settings.record(),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputPipelineIdentitySha256": input_identity,
        },
        compressed=True,
    )

    manifest["state"] = "assembling"
    atomic_json(manifest_path, manifest)
    assembly_started = time.monotonic()
    block = assemble_surface_hierarchy(
        grid,
        BlockBounds((0, 0, 0), grid.shape_cells_xyz),
        selection.patches,
        maximum_leaf_shape_cells_xyz=settings.leaf_shape_cells_xyz,
    )
    assembly_seconds = time.monotonic() - assembly_started
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block,
        output / "projections.png",
        maximum_components=settings.maximum_preview_components,
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

    options_by_cell, _ = configuration_options(grid, tables)
    selected_option_ids = {
        value.cell_xyz: value.option_id for value in selection.selected_options
    }
    census = analyze_component_gaps(
        block, options_by_cell, selected_option_ids
    )
    gap_manifest = write_gap_census(
        output / "gap-census-v1.json",
        census,
        identity_sha256=identity_sha256,
        provenance={
            "inputPipelineIdentitySha256": input_identity,
            "selectionVariantIdentitySha256": identity_sha256,
        },
    )
    deferred = Counter(value.reason for value in block.deferred_joins)
    component_sizes = [len(value.patch_ids) for value in block.components]
    summary: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-selection-variant-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputPipelineIdentitySha256": input_identity,
        "settings": settings.record(),
        "grid": patch_manifest["grid"],
        "selection": selection_manifest["statistics"],
        "assembly": {
            "candidateJoins": len(block.candidate_joins),
            "retainedJoins": len(block.joins),
            "deferredJoins": len(block.deferred_joins),
            "deferredByReason": dict(sorted(deferred.items())),
            "components": len(block.components),
            "componentPatchCount": _quantiles(component_sizes),
            "componentsAtLeast": {
                str(limit): sum(value >= limit for value in component_sizes)
                for limit in (10, 25, 50, 100, 150)
            },
            "largestComponentPatchCount": max(component_sizes, default=0),
            "weldedCrossings": len(block.welded_crossings),
            "exteriorTraces": len(block.exterior_traces),
            "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
        },
        "largestComponentGapCensus": gap_manifest["statistics"],
        "timingSeconds": {
            "selection": round(selection_seconds, 6),
            "assembly": round(assembly_seconds, 6),
            "total": round(time.monotonic() - started, 6),
        },
        "artifacts": {
            "selection": "selection-v1.npz",
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
