from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acus_adapter import AcusAdapterSettings, load_acus_flake_window
from .block import BlockBounds, assemble_surface_block, assemble_surface_hierarchy
from .contracts import RawAcusSettings, ReconstructionWindow
from .continuation_search import run_continuation_search
from .continuation_variant import run_continuation_variant
from .continuity import JoinContinuitySettings, run_join_continuity_refinement
from .export import write_block_obj, write_block_projection_png
from .flatten import run_component_flattening
from .gaps import analyze_component_gaps, write_gap_census
from .mode_bank import run_mode_bank
from .pipeline import run_raw_acus_pipeline
from .reselection import SelectionVariantSettings, run_selection_variant
from .repair import evaluate_single_cell_gap_repairs, write_gap_repair_search
from .repair_variant import run_gap_repair_variant
from .selection import configuration_options
from .stratigraphy import read_configuration_artifact
from .synthetic import SyntheticStackSettings, generate_synthetic_stack
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridSpec


def _atomic_json(path: Path, payload: dict[str, object]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _synthetic(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    grid = GridSpec(
        tuple(args.shape),
        cell_size_xyz=tuple(args.cell_size),
        coordinate_unit=args.coordinate_unit,
    )
    settings = SyntheticStackSettings(
        sheet_count=args.sheets,
        curvature_amplitude_cells=args.curvature,
        observation_noise_scale=args.noise_scale,
        missing_patch_fraction=args.missing_fraction,
        random_seed=args.seed,
    )
    started = time.monotonic()
    scene = generate_synthetic_stack(grid, settings)
    generated_seconds = time.monotonic() - started
    truth_map = scene.truth_map
    table_started = time.monotonic()
    table = PatchTable.from_patches(
        grid,
        scene.patches,
        local_order=truth_map,
    )
    manifest = write_patch_shard(
        output / "patches-v1",
        table,
        settings={
            "generator": "analytic-smooth-stack",
            "sheetCount": args.sheets,
            "curvatureAmplitudeCells": args.curvature,
            "observationNoiseScale": args.noise_scale,
            "missingPatchFraction": args.missing_fraction,
        },
        provenance={"randomSeed": args.seed},
        compressed=args.compressed,
    )
    table_seconds = time.monotonic() - table_started
    bounds = BlockBounds((0, 0, 0), grid.shape_cells_xyz)
    assembly_started = time.monotonic()
    block = assemble_surface_hierarchy(
        grid,
        bounds,
        scene.patches,
        maximum_leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    assembly_seconds = time.monotonic() - assembly_started
    direct_seconds = None
    if args.verify_direct:
        direct_started = time.monotonic()
        direct = assemble_surface_block(grid, bounds, scene.patches)
        direct_seconds = time.monotonic() - direct_started
        signatures = (
            len(block.joins),
            len(block.components),
            len(block.welded_crossings),
            len(block.exterior_traces),
            len(block.unresolved_interior_traces),
        )
        direct_signatures = (
            len(direct.joins),
            len(direct.components),
            len(direct.welded_crossings),
            len(direct.exterior_traces),
            len(direct.unresolved_interior_traces),
        )
        if signatures != direct_signatures:
            raise RuntimeError(
                f"hierarchical/direct assembly disagreement: {signatures} != "
                f"{direct_signatures}"
            )
    component_truth: dict[int, set[int]] = defaultdict(set)
    fragments_by_truth: dict[int, set[int]] = defaultdict(set)
    for patch_id, component_id in block.component_by_patch:
        truth_sheet = truth_map[patch_id]
        component_truth[component_id].add(truth_sheet)
        fragments_by_truth[truth_sheet].add(component_id)
    mixed_components = [
        component for component, truths in component_truth.items() if len(truths) > 1
    ]
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(block, output / "projections.png")
    summary: dict[str, object] = {
        "contract": {
            "geometry": "analytic local tangent planes clipped to canonical grid features",
            "assembly": "uncertainty-normalized ordered shared-face matching",
            "hierarchy": "regular leaves composed only through cached boundary traces",
        },
        "grid": manifest["grid"],
        "settings": manifest["settings"],
        "counts": {
            "truthSheets": args.sheets,
            "patches": len(scene.patches),
            "degenerateCandidates": scene.degenerate_candidate_count,
            "joins": len(block.joins),
            "deferredJoins": len(block.deferred_joins),
            "crossingTopologyCycleDeferrals": sum(
                value.reason == "crossing-topology-cycle"
                for value in block.deferred_joins
            ),
            "componentCellCollisionDeferrals": sum(
                value.reason == "component-cell-collision"
                for value in block.deferred_joins
            ),
            "components": len(block.components),
            "mixedTruthComponents": len(mixed_components),
            "fragmentedTruthSheets": sum(
                len(components) > 1 for components in fragments_by_truth.values()
            ),
            "maximumFragmentsPerTruthSheet": max(
                (len(value) for value in fragments_by_truth.values()), default=0
            ),
            "weldedCrossings": len(block.welded_crossings),
            "exteriorTraces": len(block.exterior_traces),
            "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
        },
        "welding": {
            "maximumStandardizedResidual": max(
                (
                    value.maximum_standardized_residual
                    for value in block.welded_crossings
                ),
                default=0.0,
            )
        },
        "timingSeconds": {
            "generate": round(generated_seconds, 6),
            "tableAndWrite": round(table_seconds, 6),
            "hierarchicalAssembly": round(assembly_seconds, 6),
            "directAssembly": round(direct_seconds, 6)
            if direct_seconds is not None
            else None,
        },
        "artifacts": {
            "patchManifest": "patches-v1.json",
            "patchArrays": "patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "projectionOrder": ["XY", "XZ", "YZ"],
        },
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _quantiles(values: list[int]) -> dict[str, float | None]:
    if not values:
        return {"minimum": None, "median": None, "p90": None, "maximum": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), (0, 50, 90, 100))
    return {
        name: round(float(value), 3)
        for name, value in zip(("minimum", "median", "p90", "maximum"), result)
    }


def _acus_window(args: argparse.Namespace) -> None:
    output = Path(args.output)
    output.mkdir(parents=True, exist_ok=True)
    adapter_settings = AcusAdapterSettings(
        minimum_quality=args.minimum_quality,
        normal_family=args.normal_family,
    )
    started = time.monotonic()
    scene = load_acus_flake_window(
        args.root,
        tuple(args.origin),
        tuple(args.shape),
        adapter_settings,
    )
    adapter_seconds = time.monotonic() - started
    table_started = time.monotonic()
    normal_family = {value.patch_id: args.normal_family for value in scene.patches}
    table = PatchTable.from_patches(
        scene.grid,
        scene.patches,
        local_order=scene.local_order_map,
        normal_family=normal_family,
    )
    manifest = write_patch_shard(
        output / "patches-v1",
        table,
        settings={
            "adapter": "acus-flake-plane-proxy-v1",
            "minimumQuality": args.minimum_quality,
            "normalFamily": args.normal_family,
            "uncertainty": {
                "normal": "max(floor, cell residual / sqrt(effective support))",
                "height": "max(floor, source bandwidth / sqrt(effective support), mode thickness)",
                "fiber": "max(floor, mode residual / sqrt(effective support))",
            },
        },
        provenance=scene.source_identity,
        compressed=args.compressed,
    )
    table_seconds = time.monotonic() - table_started
    bounds = BlockBounds((0, 0, 0), scene.grid.shape_cells_xyz)
    assembly_started = time.monotonic()
    block = assemble_surface_hierarchy(
        scene.grid,
        bounds,
        scene.patches,
        maximum_leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    assembly_seconds = time.monotonic() - assembly_started
    export_started = time.monotonic()
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block,
        output / "projections.png",
        maximum_components=args.maximum_preview_components,
    )
    export_seconds = time.monotonic() - export_started
    deferred_counts: dict[str, int] = defaultdict(int)
    for value in block.deferred_joins:
        deferred_counts[value.reason] += 1
    summary: dict[str, object] = {
        "contract": {
            "status": (
                "geometry plumbing experiment; inherited Acus flake modes are "
                "plane evidence proxies, not physical layer labels"
            ),
            "ownership": "only planes intersecting the non-overlapping 32-voxel core are owned",
            "joining": "ordered uncertainty-aware shared-face traces with global topology vetoes",
        },
        "grid": manifest["grid"],
        "window": {
            "originCellXYZ": list(args.origin),
            "shapeCellsXYZ": list(args.shape),
        },
        "adapterStats": scene.stats,
        "assemblyStats": {
            "candidateJoins": len(block.candidate_joins),
            "retainedJoins": len(block.joins),
            "deferredJoins": len(block.deferred_joins),
            "deferredByReason": dict(sorted(deferred_counts.items())),
            "components": len(block.components),
            "componentPatchCount": _quantiles(
                [len(value.patch_ids) for value in block.components]
            ),
            "largestComponentPatchCount": max(
                (len(value.patch_ids) for value in block.components), default=0
            ),
            "weldedCrossings": len(block.welded_crossings),
            "cornerWeldedVertices": sum(
                value.grid_vertex_xyz is not None for value in block.welded_crossings
            ),
            "exteriorTraces": len(block.exterior_traces),
            "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
        },
        "timingSeconds": {
            "adapter": round(adapter_seconds, 6),
            "tableAndWrite": round(table_seconds, 6),
            "hierarchicalAssembly": round(assembly_seconds, 6),
            "export": round(export_seconds, 6),
            "total": round(time.monotonic() - started, 6),
        },
        "artifacts": {
            "patchManifest": "patches-v1.json",
            "patchArrays": "patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "projectionOrder": ["XY", "XZ", "YZ"],
        },
    }
    _atomic_json(output / "summary.json", summary)
    print(json.dumps(summary, indent=2))


def _full_acus(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    settings = RawAcusSettings(**settings_values)
    summary = run_raw_acus_pipeline(
        args.source,
        args.output,
        ReconstructionWindow(tuple(args.voxel_origin), tuple(args.shape)),
        metadata_path=args.metadata,
        settings=settings,
        shard_shape_cells_xyz=tuple(args.shard_shape),
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        compute=args.compute,
        force=args.force,
        maximum_preview_components=args.maximum_preview_components,
        local_only=args.local_only,
        only_shard_ids=args.only_shard,
        limit_shards=args.limit_shards,
    )
    print(json.dumps(summary, indent=2))


def _gap_census(args: argparse.Namespace) -> None:
    root = Path(args.root)
    pipeline_manifest = json.loads((root / "pipeline.json").read_text())
    if pipeline_manifest.get("state") != "complete":
        raise ValueError("gap census requires a completed raw Acus reconstruction")
    identity_sha256 = str(
        pipeline_manifest["identity"]["identitySha256"]
    )
    tables = [
        read_configuration_artifact(
            root / "shards" / shard_id / "stratigraphies-v1",
            identity_sha256=identity_sha256,
        )
        for shard_id in pipeline_manifest["shards"]
    ]
    selected_table = read_patch_shard(root / "selected-patches-v1")
    patches = selected_table.to_patches()
    block = assemble_surface_hierarchy(
        selected_table.grid,
        BlockBounds((0, 0, 0), selected_table.grid.shape_cells_xyz),
        patches,
        maximum_leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    options_by_cell, _ = configuration_options(selected_table.grid, tables)
    with np.load(root / "selection-v1.npz") as values:
        selected_option_ids = {
            tuple(int(item) for item in cell): int(option_id)
            for cell, option_id in zip(values["cellXYZ"], values["optionId"])
        }
    census = analyze_component_gaps(
        block,
        options_by_cell,
        selected_option_ids,
        component_id=args.component_id,
    )
    output = args.output or (root / "gap-census-v1.json")
    payload = write_gap_census(
        output,
        census,
        identity_sha256=identity_sha256,
        provenance={
            "inputRoot": str(root.resolve()),
            "selectedPatches": "selected-patches-v1.npz",
            "configurationSource": "shards/*/stratigraphies-v1.npz",
            "directions": "axial/unsigned",
        },
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "version": payload["version"],
                "identitySha256": identity_sha256,
                "component": payload["component"],
                "statistics": payload["statistics"],
                "artifact": str(Path(output).resolve()),
            },
            indent=2,
        )
    )


def _selection_variant(args: argparse.Namespace) -> None:
    settings = SelectionVariantSettings(
        interior_unmatched_trace_penalty=args.interior_unmatched_trace_penalty,
        unary_scale=args.unary_scale,
        pairwise_scale=args.pairwise_scale,
        maximum_sweeps=args.maximum_sweeps,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        maximum_preview_components=args.maximum_preview_components,
    )
    summary = run_selection_variant(
        args.root,
        args.output,
        settings,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _mode_bank(args: argparse.Namespace) -> None:
    def progress(index: int, total: int, shard_id: str, mode_count: int) -> None:
        print(
            f"mode bank shard {index}/{total} {shard_id} · {mode_count} modes",
            flush=True,
        )

    summary = run_mode_bank(
        args.root,
        args.output,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _mode_continuation_search(args: argparse.Namespace) -> None:
    def progress(
        index: int, total: int, candidate_id: int, configuration_rank: int
    ) -> None:
        print(
            f"continuation trial {index}/{total} · candidate {candidate_id} · "
            f"configuration {configuration_rank}",
            flush=True,
        )

    payload, _ = run_continuation_search(
        args.root,
        args.mode_bank,
        args.output,
        component_id=args.component_id,
        reuse_search_path=args.reuse_search,
        maximum_modes_per_gap=args.maximum_modes_per_gap,
        maximum_configurations_per_candidate=(
            args.maximum_configurations_per_candidate
        ),
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        progress=progress,
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "identitySha256": payload["identitySha256"],
                "discovery": {
                    key: payload["discovery"][key]
                    for key in (
                        "componentId",
                        "modeGapCount",
                        "matchedGapCount",
                        "candidateCount",
                    )
                },
                "statistics": payload["statistics"],
                "recommended": [
                    value
                    for value in payload["trials"]
                    if value["recommended"]
                ],
                "artifact": str(args.output.resolve()),
            },
            indent=2,
        )
    )


def _apply_mode_continuations(args: argparse.Namespace) -> None:
    summary = run_continuation_variant(
        args.root,
        args.mode_bank,
        args.search,
        args.output,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _flatten_components(args: argparse.Namespace) -> None:
    if args.depth_step <= 0.0 or args.depth_max < args.depth_min:
        raise ValueError("depth step must be positive and depth range increasing")
    offsets = np.arange(
        args.depth_min,
        args.depth_max + 0.5 * args.depth_step,
        args.depth_step,
        dtype=np.float64,
    )
    if not len(offsets) or offsets[-1] > args.depth_max + 1.0e-8:
        raise ValueError("depth range and step do not form a valid sequence")

    def progress(
        index: int,
        total: int,
        rank: int,
        component_id: int,
        stage: str,
    ) -> None:
        print(
            f"flatten component {index}/{total} · rank {rank} · "
            f"component {component_id} · {stage}",
            flush=True,
        )

    summary = run_component_flattening(
        args.root,
        args.output,
        component_ranks=tuple(args.component_ranks),
        depth_offsets_voxels=offsets,
        pixel_step_voxels=args.pixel_step,
        maximum_pixels=args.maximum_pixels,
        maximum_chart_normal_deviation_degrees=(
            args.maximum_chart_normal_deviation
        ),
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        join_refinement_root=args.join_refinement,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _refine_join_continuity(args: argparse.Namespace) -> None:
    settings = JoinContinuitySettings(
        trace_samples=args.trace_samples,
        trace_endpoint_margin=args.trace_endpoint_margin,
        depth_radius_voxels=args.depth_radius,
        depth_step_voxels=args.depth_step,
        maximum_profile_shift_voxels=args.maximum_profile_shift,
        near_inset_voxels=args.near_inset,
        comparison_span_voxels=args.comparison_span,
        tile_shape_cells_xyz=tuple(args.tile_shape),
        outlier_standard_deviations=args.outlier_standard_deviations,
        minimum_mismatch_ratio=args.minimum_mismatch_ratio,
    )
    last_report = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        bucket = completed // 500
        if bucket != last_report or completed == total:
            last_report = bucket
            print(
                f"join continuity {completed:,}/{total:,}",
                flush=True,
            )

    summary = run_join_continuity_refinement(
        args.root,
        args.output,
        settings=settings,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _gap_repair_search(args: argparse.Namespace) -> None:
    root = Path(args.root)
    pipeline_manifest = json.loads((root / "pipeline.json").read_text())
    if pipeline_manifest.get("state") != "complete":
        raise ValueError("gap repair requires a completed raw Acus reconstruction")
    identity_sha256 = str(
        pipeline_manifest["identity"]["identitySha256"]
    )
    tables = [
        read_configuration_artifact(
            root / "shards" / shard_id / "stratigraphies-v1",
            identity_sha256=identity_sha256,
        )
        for shard_id in pipeline_manifest["shards"]
    ]
    selected_table = read_patch_shard(root / "selected-patches-v1")
    block = assemble_surface_hierarchy(
        selected_table.grid,
        BlockBounds((0, 0, 0), selected_table.grid.shape_cells_xyz),
        selected_table.to_patches(),
        maximum_leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    options_by_cell, _ = configuration_options(selected_table.grid, tables)
    with np.load(root / "selection-v1.npz") as values:
        selected_option_ids = {
            tuple(int(item) for item in cell): int(option_id)
            for cell, option_id in zip(values["cellXYZ"], values["optionId"])
        }
    census = analyze_component_gaps(
        block,
        options_by_cell,
        selected_option_ids,
        component_id=args.component_id,
    )

    def progress(
        index: int, total: int, cell: tuple[int, int, int], option_id: int
    ) -> None:
        print(
            f"gap repair trial {index}/{total} · cell {cell} · option {option_id}",
            flush=True,
        )

    search = evaluate_single_cell_gap_repairs(
        block,
        options_by_cell,
        selected_option_ids,
        census,
        maximum_leaf_shape_cells_xyz=tuple(args.leaf_shape),
        progress=progress,
    )
    output = args.output or (root / "gap-repair-search-v1.json")
    payload = write_gap_repair_search(
        output,
        search,
        identity_sha256=identity_sha256,
        provenance={
            "inputRoot": str(root.resolve()),
            "selection": "selection-v1.npz",
            "gapCensus": "computed from immutable selected geometry",
            "acceptance": "full topology-safe hierarchical reassembly per trial",
        },
    )
    print(
        json.dumps(
            {
                "schema": payload["schema"],
                "version": payload["version"],
                "componentId": payload["componentId"],
                "statistics": payload["statistics"],
                "recommended": [
                    value
                    for value in payload["trials"]
                    if value["recommended"]
                ],
                "artifact": str(Path(output).resolve()),
            },
            indent=2,
        )
    )


def _apply_gap_repairs(args: argparse.Namespace) -> None:
    summary = run_gap_repair_variant(
        args.root,
        args.search,
        args.output,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Dataset-independent cubical surface reconstruction tools."
    )
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic = subparsers.add_parser(
        "synthetic", description="Generate and assemble an analytic sheet stack."
    )
    synthetic.add_argument("--shape", nargs=3, type=int, default=(16, 16, 8))
    synthetic.add_argument(
        "--cell-size", nargs=3, type=float, default=(1.0, 1.0, 1.0)
    )
    synthetic.add_argument("--coordinate-unit", default="cell")
    synthetic.add_argument("--sheets", type=int, default=4)
    synthetic.add_argument("--curvature", type=float, default=0.12)
    synthetic.add_argument("--noise-scale", type=float, default=0.25)
    synthetic.add_argument("--missing-fraction", type=float, default=0.0)
    synthetic.add_argument("--seed", type=int, default=7)
    synthetic.add_argument("--leaf-shape", nargs=3, type=int, default=(4, 4, 4))
    synthetic.add_argument(
        "--output", type=Path, default=Path("work/cubical-synthetic-v1")
    )
    synthetic.add_argument("--compressed", action="store_true")
    synthetic.add_argument("--verify-direct", action="store_true")
    synthetic.set_defaults(handler=_synthetic)
    acus = subparsers.add_parser(
        "acus-window",
        description="Adapt and assemble one persisted Acus cell window.",
    )
    acus.add_argument(
        "--root", type=Path, default=Path("work/cross-scroll-analysis-z512")
    )
    acus.add_argument("--origin", nargs=3, type=int, default=(109, 86, 0))
    acus.add_argument("--shape", nargs=3, type=int, default=(16, 16, 14))
    acus.add_argument("--leaf-shape", nargs=3, type=int, default=(4, 4, 4))
    acus.add_argument("--minimum-quality", type=float, default=0.08)
    acus.add_argument("--normal-family", type=int, default=0)
    acus.add_argument("--maximum-preview-components", type=int, default=96)
    acus.add_argument(
        "--output", type=Path, default=Path("work/cubical-acus-window-v1")
    )
    acus.add_argument("--compressed", action="store_true")
    acus.set_defaults(handler=_acus_window)
    full_acus = subparsers.add_parser(
        "full-acus",
        description=(
            "Run the cubical pipeline from native CT voxels. This command does "
            "not consume legacy needle, flake, graph, or component artifacts."
        ),
    )
    full_acus.add_argument(
        "--source",
        type=Path,
        default=Path(
            "/mnt/t5/acus-cross-scroll/"
            "pherc0358-z7168-d512-yfull-xfull.npy"
        ),
    )
    full_acus.add_argument("--metadata", type=Path)
    full_acus.add_argument(
        "--voxel-origin",
        nargs=3,
        type=int,
        default=(3520, 2784, 160),
        metavar=("X", "Y", "Z"),
        help="source-local voxel coordinate of the first cubical cell corner",
    )
    full_acus.add_argument(
        "--shape",
        nargs=3,
        type=int,
        default=(8, 8, 6),
        metavar=("X", "Y", "Z"),
        help="number of owned cubical cells",
    )
    full_acus.add_argument(
        "--shard-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    full_acus.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    full_acus.add_argument(
        "--compute", choices=("auto", "gpu", "cpu"), default="auto"
    )
    full_acus.add_argument(
        "--settings-json",
        type=Path,
        help="optional RawAcusSettings keyword object for reproducible experiments",
    )
    full_acus.add_argument(
        "--output", type=Path, default=Path("work/raw-acus-cubical-v1")
    )
    full_acus.add_argument("--maximum-preview-components", type=int, default=128)
    full_acus.add_argument(
        "--local-only",
        action="store_true",
        help="bake resumable raw evidence and stratigraphies without global selection",
    )
    full_acus.add_argument(
        "--only-shard",
        action="append",
        help="process one planned shard id (repeatable); implies local-only behavior",
    )
    full_acus.add_argument(
        "--limit-shards",
        type=int,
        help="process at most this many incomplete local shards in this invocation",
    )
    full_acus.add_argument(
        "--force",
        action="store_true",
        help="rebake this pipeline's own matching artifacts from native CT",
    )
    full_acus.set_defaults(handler=_full_acus)
    gap_census = subparsers.add_parser(
        "gap-census",
        description=(
            "Classify unresolved cubical traces as ordering decisions, topology "
            "vetoes, recoverable configuration gaps, or missing local modes."
        ),
    )
    gap_census.add_argument("--root", type=Path, required=True)
    gap_census.add_argument("--component-id", type=int)
    gap_census.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    gap_census.add_argument("--output", type=Path)
    gap_census.set_defaults(handler=_gap_census)
    selection_variant = subparsers.add_parser(
        "selection-variant",
        description=(
            "Reselect a completed raw-Acus stratigraphy bake under an explicit "
            "continuation prior without rerunning Acus or mutating the input."
        ),
    )
    selection_variant.add_argument("--root", type=Path, required=True)
    selection_variant.add_argument("--output", type=Path, required=True)
    selection_variant.add_argument(
        "--interior-unmatched-trace-penalty", type=float, default=0.0
    )
    selection_variant.add_argument("--unary-scale", type=float, default=1.0)
    selection_variant.add_argument("--pairwise-scale", type=float, default=0.35)
    selection_variant.add_argument("--maximum-sweeps", type=int, default=12)
    selection_variant.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    selection_variant.add_argument(
        "--maximum-preview-components", type=int, default=128
    )
    selection_variant.add_argument("--force", action="store_true")
    selection_variant.set_defaults(handler=_selection_variant)
    mode_bank = subparsers.add_parser(
        "mode-bank",
        description=(
            "Persist every fitted local Acus mode from an existing completed "
            "evidence bake, before top-M stratigraphy pruning."
        ),
    )
    mode_bank.add_argument("--root", type=Path, required=True)
    mode_bank.add_argument("--output", type=Path, required=True)
    mode_bank.add_argument("--force", action="store_true")
    mode_bank.set_defaults(handler=_mode_bank)
    mode_continuation = subparsers.add_parser(
        "mode-continuation-search",
        description=(
            "Recover pruned local modes at explicit component gaps and validate "
            "conditioned physical stratigraphies by complete reassembly."
        ),
    )
    mode_continuation.add_argument("--root", type=Path, required=True)
    mode_continuation.add_argument("--mode-bank", type=Path, required=True)
    mode_continuation.add_argument("--output", type=Path, required=True)
    mode_continuation.add_argument("--component-id", type=int)
    mode_continuation.add_argument(
        "--reuse-search",
        type=Path,
        help="reuse already evaluated candidate/configuration trials",
    )
    mode_continuation.add_argument("--maximum-modes-per-gap", type=int, default=3)
    mode_continuation.add_argument(
        "--maximum-configurations-per-candidate", type=int, default=3
    )
    mode_continuation.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    mode_continuation.set_defaults(handler=_mode_continuation_search)
    apply_mode_continuation = subparsers.add_parser(
        "apply-mode-continuations",
        description=(
            "Combine independently safe full-mode continuation trials and "
            "write an auditable reconstructed variant with previews."
        ),
    )
    apply_mode_continuation.add_argument("--root", type=Path, required=True)
    apply_mode_continuation.add_argument("--mode-bank", type=Path, required=True)
    apply_mode_continuation.add_argument("--search", type=Path, required=True)
    apply_mode_continuation.add_argument("--output", type=Path, required=True)
    apply_mode_continuation.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    apply_mode_continuation.add_argument("--force", action="store_true")
    apply_mode_continuation.set_defaults(handler=_apply_mode_continuations)
    flatten_components = subparsers.add_parser(
        "flatten-components",
        description=(
            "Flatten selected cubical components into a bounded-distortion atlas "
            "and sample fixed native-CT depth stacks without per-cell alignment."
        ),
    )
    flatten_components.add_argument("--root", type=Path, required=True)
    flatten_components.add_argument("--output", type=Path, required=True)
    flatten_components.add_argument(
        "--component-ranks", nargs="+", type=int, default=(1, 2, 3, 7)
    )
    flatten_components.add_argument("--depth-min", type=float, default=-12.0)
    flatten_components.add_argument("--depth-max", type=float, default=12.0)
    flatten_components.add_argument("--depth-step", type=float, default=1.0)
    flatten_components.add_argument("--pixel-step", type=float, default=2.0)
    flatten_components.add_argument("--maximum-pixels", type=int, default=768)
    flatten_components.add_argument(
        "--maximum-chart-normal-deviation", type=float, default=40.0
    )
    flatten_components.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    flatten_components.add_argument("--join-refinement", type=Path)
    flatten_components.add_argument("--force", action="store_true")
    flatten_components.set_defaults(handler=_flatten_components)
    refine_continuity = subparsers.add_parser(
        "refine-join-continuity",
        description=(
            "Score every accepted join against fixed-depth native CT and split "
            "only robust discontinuity outliers relative to within-patch controls."
        ),
    )
    refine_continuity.add_argument("--root", type=Path, required=True)
    refine_continuity.add_argument("--output", type=Path, required=True)
    refine_continuity.add_argument("--trace-samples", type=int, default=7)
    refine_continuity.add_argument(
        "--trace-endpoint-margin", type=float, default=0.15
    )
    refine_continuity.add_argument("--depth-radius", type=float, default=8.0)
    refine_continuity.add_argument("--depth-step", type=float, default=2.0)
    refine_continuity.add_argument(
        "--maximum-profile-shift", type=float, default=6.0
    )
    refine_continuity.add_argument("--near-inset", type=float, default=1.5)
    refine_continuity.add_argument("--comparison-span", type=float, default=3.0)
    refine_continuity.add_argument(
        "--tile-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    refine_continuity.add_argument(
        "--outlier-standard-deviations", type=float, default=4.0
    )
    refine_continuity.add_argument(
        "--minimum-mismatch-ratio", type=float, default=1.5
    )
    refine_continuity.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    refine_continuity.add_argument("--force", action="store_true")
    refine_continuity.set_defaults(handler=_refine_join_continuity)
    gap_repair = subparsers.add_parser(
        "gap-repair-search",
        description=(
            "Try retained single-cell stratigraphy substitutions at recoverable "
            "gaps, validating every trial by full topology-safe reassembly."
        ),
    )
    gap_repair.add_argument("--root", type=Path, required=True)
    gap_repair.add_argument("--component-id", type=int)
    gap_repair.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    gap_repair.add_argument("--output", type=Path)
    gap_repair.set_defaults(handler=_gap_repair_search)
    apply_repairs = subparsers.add_parser(
        "apply-gap-repairs",
        description=(
            "Apply the best nonconflicting conservative trials from a gap-repair "
            "search and write a separately auditable reconstruction variant."
        ),
    )
    apply_repairs.add_argument("--root", type=Path, required=True)
    apply_repairs.add_argument("--search", type=Path, required=True)
    apply_repairs.add_argument("--output", type=Path, required=True)
    apply_repairs.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    apply_repairs.add_argument("--force", action="store_true")
    apply_repairs.set_defaults(handler=_apply_gap_repairs)
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
