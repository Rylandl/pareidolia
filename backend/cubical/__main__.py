from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

from .acus_adapter import AcusAdapterSettings, load_acus_flake_window
from .block import BlockBounds, assemble_surface_block, assemble_surface_hierarchy
from .export import write_block_obj, write_block_projection_png
from .synthetic import SyntheticStackSettings, generate_synthetic_stack
from .tables import PatchTable, write_patch_shard
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
    args = parser.parse_args()
    args.handler(args)


if __name__ == "__main__":
    main()
