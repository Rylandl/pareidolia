from __future__ import annotations

import argparse
import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Mapping

import numpy as np

from .acus_adapter import AcusAdapterSettings, load_acus_flake_window
from .block import BlockBounds, assemble_surface_block, assemble_surface_hierarchy
from .boundary_audit import (
    run_cluster_reference_audit,
    run_independent_boundary_audit,
    run_boundary_reselection_split_audit,
    run_boundary_split_audit,
)
from .boundary_band import BoundaryBandSettings, run_boundary_band_export
from .boundary_merge import run_boundary_band_merge
from .boundary_reselection import (
    BoundaryReselectionSettings,
    run_boundary_band_reselection,
)
from .cluster_reselection import run_boundary_cluster_reselection
from .cluster_materialization import run_cluster_materialization
from .cell_refinement import (
    CellRefinementSettings,
    run_cell_refinement_diagnostic,
)
from .cell_refinement_variant import run_cell_refinement_materialization
from .cell_refinement_targets import (
    CellRefinementTargetSettings,
    run_cell_refinement_target_ranking,
)
from .contracts import RawAcusSettings, ReconstructionWindow
from .continuation_search import run_continuation_search
from .continuation_variant import run_continuation_variant
from .continuity import JoinContinuitySettings, run_join_continuity_refinement
from .contextual_growth import ContextualGrowthSettings, run_contextual_growth
from .clear_core_interface_refinement import (
    run_clear_core_interface_refinement,
)
from .clear_ribbon import ClearRibbonSettings, run_clear_ribbon_bank
from .clear_ribbon_feedback import run_clear_ribbon_feedback
from .clear_ribbon_paired_feedback import run_clear_ribbon_paired_feedback
from .clear_ribbon_selection import (
    ClearRibbonSelectionSettings,
    run_clear_ribbon_selection,
)
from .export import write_block_obj, write_block_projection_png
from .flatten import run_component_flattening
from .gaps import analyze_component_gaps, write_gap_census
from .isolated_slab import (
    IsolatedSlabSettings,
    run_isolated_slab_detection,
)
from .isolated_slab_audit import (
    IsolatedSlabAcusAuditSettings,
    run_isolated_slab_acus_audit,
)
from .mode_bank import run_mode_bank
from .multiseam import run_multiseam_audit
from .needle_field import (
    BlockNeedleFieldSettings,
    run_block_needle_field,
)
from .needle_bundle import (
    BlockNeedleBundleSettings,
    run_block_needle_bundles,
)
from .needle_flatten import run_block_needle_flattening
from .needle_surface import (
    BlockNeedleSurfaceSettings,
    run_block_needle_surfaces,
)
from .needle_topology import (
    BlockNeedleTopologySettings,
    run_block_needle_topology,
)
from .one_sided_interface import (
    OneSidedInterfaceSettings,
    run_one_sided_interface_bank,
)
from .one_sided_growth import (
    OneSidedGrowthSettings,
    run_one_sided_growth,
)
from .pipeline import run_raw_acus_pipeline
from .paired_surface_bank import (
    PairedSurfaceBankSettings,
    run_paired_surface_bank,
)
from .paired_surface_growth import (
    PairedSurfaceGrowthSettings,
    run_paired_surface_growth,
)
from .physical_ribbon_bank import (
    PhysicalRibbonBankSettings,
    run_physical_ribbon_bank,
)
from .physical_ribbon_continuity import (
    PhysicalRibbonContinuitySettings,
    run_physical_ribbon_continuity,
)
from .physical_ribbon_configuration import (
    PhysicalRibbonConfigurationSettings,
    run_physical_ribbon_configuration,
)
from .physical_ribbon_bridging import (
    PhysicalRibbonBridgingSettings,
    run_physical_ribbon_bridging,
)
from .physical_ribbon_collective import (
    PhysicalRibbonCollectiveSettings,
    run_physical_ribbon_collective,
)
from .physical_ribbon_flattened_audit import (
    PhysicalRibbonFlattenedAuditSettings,
    run_physical_ribbon_flattened_audit,
)
from .physical_ribbon_depth_fields import (
    PhysicalRibbonDepthFieldSettings,
    run_physical_ribbon_depth_fields,
)
from .physical_ribbon_surface_holes import (
    PhysicalRibbonSurfaceHoleSettings,
    run_physical_ribbon_surface_holes,
)
from .physical_ribbon_open_bays import (
    PhysicalRibbonOpenBaySettings,
    run_physical_ribbon_open_bays,
)
from .physical_ribbon_open_bay_saturation import (
    PhysicalRibbonOpenBaySaturationSettings,
    run_physical_ribbon_open_bay_saturation,
)
from .physical_ribbon_dense_completion import (
    PhysicalRibbonDenseCompletionSettings,
    run_physical_ribbon_dense_completion,
)
from .physical_ribbon_texture_gate import run_physical_ribbon_texture_gate
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    run_physical_ribbon_patch_holes,
)
from .physical_ribbon_patch_states import (
    PhysicalRibbonPatchStateSettings,
    run_physical_ribbon_patch_states,
)
from .physical_ribbon_patch_corridors import (
    PhysicalRibbonPatchCorridorSettings,
    run_physical_ribbon_patch_corridors,
)
from .physical_ribbon_corridor_variants import (
    PhysicalRibbonCorridorVariantSettings,
    run_physical_ribbon_corridor_variants,
)
from .physical_ribbon_corridor_sets import (
    PhysicalRibbonCorridorSetSettings,
    run_physical_ribbon_corridor_sets,
)
from .physical_ribbon_corridor_extension import (
    PhysicalRibbonCorridorExtensionSettings,
    run_physical_ribbon_corridor_extension,
)
from .physical_ribbon_corridor_dormant import (
    PhysicalRibbonDormantCorridorSettings,
    run_physical_ribbon_dormant_corridors,
)
from .physical_ribbon_corridor_frontier import (
    PhysicalRibbonCorridorFrontierSettings,
    run_physical_ribbon_corridor_frontier,
)
from .physical_ribbon_corridor_one_sided import (
    PhysicalRibbonOneSidedCorridorSettings,
    run_physical_ribbon_one_sided_corridors,
)
from .physical_ribbon_replay_configuration import (
    run_physical_ribbon_replay_configuration,
)
from .physical_ribbon_corridor_saturation import (
    run_physical_ribbon_corridor_saturation,
)
from .physical_ribbon_corridor_deficits import (
    run_physical_ribbon_corridor_deficits,
)
from .physical_ribbon_corridor_faces import (
    PhysicalRibbonCorridorFaceSettings,
    run_physical_ribbon_corridor_faces,
)
from .physical_ribbon_corridor_face_replay import (
    PhysicalRibbonCorridorFaceReplaySettings,
    run_physical_ribbon_corridor_face_replay,
)
from .physical_ribbon_complete_strips import (
    PhysicalRibbonCompleteStripSettings,
    run_physical_ribbon_complete_strips,
)
from .physical_ribbon_complete_strip_replay import (
    PhysicalRibbonCompleteStripReplaySettings,
    run_physical_ribbon_complete_strip_replay,
)
from .physical_ribbon_lineage_strips import (
    PhysicalRibbonLineageStripSettings,
    run_physical_ribbon_lineage_strips,
)
from .physical_ribbon_lineage_strip_replay import (
    PhysicalRibbonLineageStripReplaySettings,
    run_physical_ribbon_lineage_strip_replay,
)
from .physical_ribbon_cumulative_corridor_replay import (
    PhysicalRibbonCumulativeCorridorReplaySettings,
    run_physical_ribbon_cumulative_corridor_replay,
)
from .physical_ribbon_cumulative_hole_replay import (
    PhysicalRibbonCumulativeHoleReplaySettings,
    run_physical_ribbon_cumulative_hole_replay,
)
from .reselection import SelectionVariantSettings, run_selection_variant
from .repair import evaluate_single_cell_gap_repairs, write_gap_repair_search
from .repair_variant import run_gap_repair_variant
from .selection import configuration_options
from .sheet_packets import (
    DualAxisPacketSettings,
    run_dual_axis_packet_connectivity,
)
from .sheet_resolution import (
    SheetResolutionAuditSettings,
    run_sheet_resolution_audit,
)
from .sheet_evidence import (
    SheetEvidenceInput,
    SheetEvidenceSettings,
    compile_block_sheet_evidence,
)
from .sheet_correspondence import catalog_block_sheet_correspondences
from .sheet_factors import SheetFactorSettings, compile_sheet_configuration_factors
from .sheet_configuration_solver import (
    SheetConfigurationSolverSettings,
    run_sheet_configuration_initialization,
)
from .sheet_core_audit import audit_sheet_core
from .sheet_graph_solver import replay_joint_sheet_graph
from .sheet_ownership import (
    crop_surface_graph_to_owned_block,
    extract_sheet_evidence_subblock,
    finalize_sheet_halo_experiment,
    run_sheet_halo_experiment,
)
from .sheet_topology_refinement import (
    SheetTopologyRefinementSettings,
    run_sheet_topology_refinement,
)
from .sheet_stitching import (
    SheetStitchingSettings,
    run_block_sheet_restitching,
)
from .saturation import SheetSaturationSettings, run_sheet_saturation_audit
from .saturation_reselection import (
    SaturationReselectionSettings,
    run_saturation_reselection,
)
from .saturation_selection import (
    SaturationCandidateSelectionSettings,
    run_saturation_candidate_selection,
)
from .stratigraphic_continuity import (
    StratigraphicContinuitySettings,
    run_stratigraphic_continuity_refinement,
)
from .subblock import extract_selected_patch_subblock
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
        calibration_path=args.calibration,
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


def _block_needle_field(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    argument_settings = {
        "neighbor_radius": "neighbor_radius_voxels",
        "maximum_neighbors": "maximum_neighbors",
        "spatial_kernel": "spatial_kernel_voxels",
        "smoothing_weight": "smoothing_weight",
        "robust_smoothing_angle": "robust_smoothing_angle_degrees",
        "iterations": "iteration_count",
        "damping": "damping",
        "maximum_normal_hypotheses": "maximum_normal_hypotheses",
        "minimum_candidate_cross_angle": "minimum_candidate_cross_angle_degrees",
        "candidate_residual_kernel": "candidate_residual_kernel_degrees",
        "candidate_separation": "candidate_separation_degrees",
        "tangent_compatibility_sigma": "tangent_compatibility_sigma_voxels",
        "mixture_pairwise_weight": "mixture_pairwise_weight",
        "mixture_initial_temperature": "mixture_initial_temperature",
        "mixture_temperature": "mixture_temperature",
        "mixture_damping": "mixture_damping",
        "mixture_iterations": "mixture_iteration_count",
        "mixture_annealing_iterations": "mixture_annealing_iterations",
        "carrier_minimum_affinity": "carrier_minimum_affinity",
        "minimum_curvature_radius": "minimum_curvature_radius_voxels",
        "compute": "compute",
    }
    for argument_name, setting_name in argument_settings.items():
        value = getattr(args, argument_name, None)
        if value is not None:
            settings_values[setting_name] = value
    summary = run_block_needle_field(
        args.raw_root,
        args.output,
        world_start_xyz=tuple(args.world_start),
        world_stop_xyz_exclusive=tuple(args.world_stop),
        settings=BlockNeedleFieldSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _block_needle_topology(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_block_needle_topology(
        args.field,
        args.output,
        settings=BlockNeedleTopologySettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _block_needle_surfaces(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_block_needle_surfaces(
        args.topology,
        args.output,
        settings=BlockNeedleSurfaceSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _block_needle_flattening(args: argparse.Namespace) -> None:
    summary = run_block_needle_flattening(
        args.surfaces,
        args.output,
        grouping=args.grouping,
        maximum_components=args.maximum_components,
        pixel_step_voxels=args.pixel_step,
        maximum_pixels=args.maximum_pixels,
        depth_min_voxels=args.depth_min,
        depth_max_voxels=args.depth_max,
        depth_step_voxels=args.depth_step,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _block_needle_bundles(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_block_needle_bundles(
        args.surfaces,
        args.output,
        settings=BlockNeedleBundleSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _isolated_slabs(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_isolated_slab_detection(
        args.source,
        args.output,
        world_start_xyz=tuple(args.world_start),
        world_stop_xyz_exclusive=tuple(args.world_stop),
        metadata_path=args.metadata,
        settings=IsolatedSlabSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _audit_isolated_slab_acus(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_isolated_slab_acus_audit(
        args.slabs,
        args.field,
        args.output,
        settings=IsolatedSlabAcusAuditSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _paired_surface_bank(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_paired_surface_bank(
        args.slabs,
        args.output,
        settings=PairedSurfaceBankSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _paired_surface_growth(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_paired_surface_growth(
        args.bank,
        args.output,
        settings=PairedSurfaceGrowthSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _one_sided_interface_bank(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_one_sided_interface_bank(
        args.growth,
        args.output,
        settings=OneSidedInterfaceSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _one_sided_interface_growth(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_one_sided_growth(
        args.bank,
        args.output,
        settings=OneSidedGrowthSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _clear_ribbon_bank(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_clear_ribbon_bank(
        args.growth,
        args.output,
        settings=ClearRibbonSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _clear_ribbon_selection(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_clear_ribbon_selection(
        args.bank,
        args.output,
        settings=ClearRibbonSelectionSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _clear_ribbon_feedback(args: argparse.Namespace) -> None:
    summary = run_clear_ribbon_feedback(
        args.selection,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _clear_ribbon_paired_feedback(args: argparse.Namespace) -> None:
    summary = run_clear_ribbon_paired_feedback(
        args.feedback,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _clear_core_interface_refinement(args: argparse.Namespace) -> None:
    summary = run_clear_core_interface_refinement(
        args.paired_feedback,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_bank(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_bank(
        args.interfaces,
        args.output,
        settings=PhysicalRibbonBankSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_continuity(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_continuity(
        args.ribbons,
        args.output,
        settings=PhysicalRibbonContinuitySettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_configuration(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_configuration(
        args.continuity,
        args.output,
        topology_continuity_root=args.topology_continuity,
        settings=PhysicalRibbonConfigurationSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_bridging(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_bridging(
        args.configuration,
        args.output,
        bridge_continuity_root=args.bridge_continuity,
        settings=PhysicalRibbonBridgingSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_collective(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_collective(
        args.configuration,
        args.output,
        settings=PhysicalRibbonCollectiveSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_flattened_audit(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_flattened_audit(
        args.surface,
        args.output,
        settings=PhysicalRibbonFlattenedAuditSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_depth_fields(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_depth_fields(
        args.holes,
        args.output,
        settings=PhysicalRibbonDepthFieldSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_surface_holes(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    for name in ("profile_depth_fractions", "competing_shift_thicknesses"):
        if name in settings_values:
            settings_values[name] = tuple(settings_values[name])
    summary = run_physical_ribbon_surface_holes(
        args.surface,
        args.output,
        settings=PhysicalRibbonSurfaceHoleSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_open_bays(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    for name in ("profile_depth_fractions", "competing_shift_thicknesses"):
        if name in settings_values:
            settings_values[name] = tuple(settings_values[name])
    summary = run_physical_ribbon_open_bays(
        args.surface,
        args.output,
        settings=PhysicalRibbonOpenBaySettings(**settings_values),
        prior_completion_roots=tuple(args.prior_completion),
        prior_texture_audit_roots=tuple(args.prior_texture_audit),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_open_bay_saturation(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")

    def progress(
        round_index: int, stage: str, values: Mapping[str, object]
    ) -> None:
        details = " · ".join(
            f"{name} {value}"
            for name, value in values.items()
            if isinstance(value, (int, float, str, bool))
        )
        suffix = f" · {details}" if details else ""
        print(
            f"open-bay saturation round {round_index} · {stage}{suffix}",
            flush=True,
        )

    summary = run_physical_ribbon_open_bay_saturation(
        args.surface,
        args.output,
        settings=PhysicalRibbonOpenBaySaturationSettings.from_record(
            settings_values
        ),
        prior_completion_roots=tuple(args.prior_completion),
        prior_texture_audit_roots=tuple(args.prior_texture_audit),
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_dense_completion(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    for name in (
        "profile_depth_fractions",
        "competing_shift_thicknesses",
        "interior_boundary_separation_hypotheses_voxels",
    ):
        if name in settings_values:
            settings_values[name] = tuple(settings_values[name])
    summary = run_physical_ribbon_dense_completion(
        args.holes,
        args.depth_field,
        args.output,
        settings=PhysicalRibbonDenseCompletionSettings(**settings_values),
        texture_audit_root=args.texture_audit,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_patch_states(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_patch_states(
        args.holes,
        args.output,
        depth_field_root=args.depth_field,
        settings=PhysicalRibbonPatchStateSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_texture_gate(args: argparse.Namespace) -> None:
    summary = run_physical_ribbon_texture_gate(
        args.patch_state,
        args.texture_audit,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_patch_holes(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_patch_holes(
        args.configuration,
        args.output,
        surface_replay_root=args.surface_replay,
        settings=PhysicalRibbonPatchHoleSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_patch_corridors(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_patch_corridors(
        args.configuration,
        args.output,
        surface_replay_root=args.surface_replay,
        settings=PhysicalRibbonPatchCorridorSettings(**settings_values),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_variants(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_variants(
        args.corridors,
        args.configuration,
        args.output,
        settings=PhysicalRibbonCorridorVariantSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_sets(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_sets(
        args.variants,
        args.configuration,
        args.output,
        settings=PhysicalRibbonCorridorSetSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_extension(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_extension(
        args.variants,
        args.configuration,
        args.output,
        settings=PhysicalRibbonCorridorExtensionSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_dormant(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_dormant_corridors(
        args.corridors,
        args.variants,
        args.corridor_sets,
        args.configuration,
        args.expanded_continuity,
        args.output,
        settings=PhysicalRibbonDormantCorridorSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_frontier(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_frontier(
        args.corridors,
        args.prior_replay,
        args.configuration,
        args.bidirectional_continuity,
        args.output,
        settings=PhysicalRibbonCorridorFrontierSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_one_sided(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_one_sided_corridors(
        args.frontier,
        args.output,
        settings=PhysicalRibbonOneSidedCorridorSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_replay_configuration(args: argparse.Namespace) -> None:
    summary = run_physical_ribbon_replay_configuration(
        args.replay,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_saturation(args: argparse.Namespace) -> None:
    summary = run_physical_ribbon_corridor_saturation(
        args.prior_corridors,
        args.prior_frontier,
        args.prior_replay,
        args.current_corridors,
        args.current_frontier,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_deficits(args: argparse.Namespace) -> None:
    summary = run_physical_ribbon_corridor_deficits(
        args.replay,
        args.output,
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_faces(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_faces(
        args.replay,
        args.output,
        settings=PhysicalRibbonCorridorFaceSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_corridor_face_replay(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_corridor_face_replay(
        args.faces,
        args.output,
        settings=PhysicalRibbonCorridorFaceReplaySettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_complete_strips(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_complete_strips(
        args.replay,
        args.output,
        settings=PhysicalRibbonCompleteStripSettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_complete_strip_replay(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_complete_strip_replay(
        args.strips,
        args.output,
        settings=PhysicalRibbonCompleteStripReplaySettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_lineage_strips(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_lineage_strips(
        args.replay,
        args.output,
        settings=PhysicalRibbonLineageStripSettings(**settings_values),
        target_rows=args.corridor_row,
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_lineage_strip_replay(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_lineage_strip_replay(
        args.lineage_strips,
        args.output,
        settings=PhysicalRibbonLineageStripReplaySettings(**settings_values),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_cumulative_corridor_replay(
    args: argparse.Namespace,
) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_cumulative_corridor_replay(
        args.prior_replay,
        args.candidate_replay,
        args.output,
        settings=PhysicalRibbonCumulativeCorridorReplaySettings(
            **settings_values
        ),
        force=args.force,
        progress=print,
    )
    print(json.dumps(summary, indent=2))


def _physical_ribbon_cumulative_hole_replay(args: argparse.Namespace) -> None:
    settings_values: dict[str, object] = {}
    if args.settings_json is not None:
        settings_values = json.loads(Path(args.settings_json).read_text())
        if not isinstance(settings_values, dict):
            raise ValueError("settings JSON must contain one object")
    summary = run_physical_ribbon_cumulative_hole_replay(
        args.prior_replay,
        args.holes,
        args.output,
        settings=PhysicalRibbonCumulativeHoleReplaySettings(**settings_values),
        force=args.force,
        progress=print,
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
        source_root=args.source_root,
        surface_graph_root=args.surface_graph,
        join_refinement_root=args.join_refinement,
        stratigraphic_refinement_root=args.stratigraphic_refinement,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _materialize_boundary_cluster(args: argparse.Namespace) -> None:
    summary = run_cluster_materialization(
        args.cluster,
        args.output,
        boundary_roots=args.boundary,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _diagnose_cell_refinement(args: argparse.Namespace) -> None:
    summary = run_cell_refinement_diagnostic(
        args.cluster,
        args.materialized,
        args.output,
        cell_xyz=tuple(args.cell),
        component_id=args.component_id,
        neighborhood_radius_cells=args.neighborhood_radius,
        settings=CellRefinementSettings(
            unary_scale=args.unary_scale,
            pairwise_scale=args.pairwise_scale,
            pairwise_reward_normalization=args.pairwise_reward_normalization,
            unmatched_trace_penalty=args.unmatched_trace_penalty,
            coverage_reward_scale=args.coverage_reward_scale,
            minimum_oracle_coverage_fraction=(
                args.minimum_oracle_coverage_fraction
            ),
            maximum_cell_utilization_drop=args.maximum_cell_utilization_drop,
            minimum_evidence_mass_for_coverage_floor=(
                args.minimum_evidence_mass_for_coverage_floor
            ),
            maximum_sweeps=args.maximum_sweeps,
            maximum_pair_sweeps=args.maximum_pair_sweeps,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _materialize_cell_refinement(args: argparse.Namespace) -> None:
    summary = run_cell_refinement_materialization(
        args.cluster,
        args.materialized,
        args.diagnostic,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _rank_cell_refinement_targets(args: argparse.Namespace) -> None:
    summary = run_cell_refinement_target_ranking(
        args.cluster,
        args.materialized,
        args.output,
        settings=CellRefinementTargetSettings(
            neighborhood_radius_cells=args.neighborhood_radius,
            maximum_targets=args.maximum_targets,
            minimum_recoverable_evidence_mass=(
                args.minimum_recoverable_evidence_mass
            ),
            minimum_incident_open_trace_endpoints=(
                args.minimum_incident_open_trace_endpoints
            ),
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _audit_sheet_resolution(args: argparse.Namespace) -> None:
    summary = run_sheet_resolution_audit(
        args.graph,
        args.output,
        settings=SheetResolutionAuditSettings(
            normal_limit_degrees=args.normal_limit_degrees,
            robust_standard_deviations=args.robust_standard_deviations,
            minimum_normal_limit_degrees=args.minimum_normal_limit_degrees,
            minimum_coherent_layers=args.minimum_coherent_layers,
            maximum_refinement_factor=args.maximum_refinement_factor,
            voxel_size_microns=args.voxel_size_microns,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _restitch_block_sheets(args: argparse.Namespace) -> None:
    summary = run_block_sheet_restitching(
        args.cluster,
        args.materialized,
        args.output,
        settings=SheetStitchingSettings(
            minimum_join_benefit=args.minimum_join_benefit,
            quarter_turn_penalty=args.quarter_turn_penalty,
            unmatched_trace_penalty=args.unmatched_trace_penalty,
            restart_count=args.restarts,
            priority_jitter_fraction=args.priority_jitter_fraction,
            exchange_round_count=args.exchange_rounds,
            exchange_trials_per_round=args.exchange_trials_per_round,
            collision_cut_enabled=not args.no_collision_cut,
            collision_cut_limit=args.collision_cut_limit,
            collision_cut_order=args.collision_cut_order,
            curvature_refinement_enabled=args.curvature_refinement,
            curvature_neighborhood_radius=args.curvature_neighborhood_radius,
            curvature_minimum_branch_support=(
                args.curvature_minimum_branch_support
            ),
            curvature_robust_standard_deviations=(
                args.curvature_robust_standard_deviations
            ),
            curvature_minimum_calibration_joins=(
                args.curvature_minimum_calibration_joins
            ),
            curvature_round_count=args.curvature_rounds,
            curvature_cut_penalty_weight=args.curvature_cut_penalty_weight,
            strict_normal_angle_cap_degrees=(
                args.strict_normal_angle_cap_degrees
            ),
            strict_fiber_angle_cap_degrees=(
                args.strict_fiber_angle_cap_degrees
            ),
            angle_calibration_robust_standard_deviations=(
                args.angle_calibration_robust_standard_deviations
            ),
            angle_calibration_minimum_joins=(
                args.angle_calibration_minimum_joins
            ),
            layer_partition_enabled=args.layer_partition,
            stack_transport_enabled=args.stack_transport,
            signed_partition_enabled=not args.no_signed_partition,
            layer_repulsion_minimum_overlap_fraction=(
                args.layer_repulsion_minimum_overlap_fraction
            ),
            layer_repulsion_minimum_normal_separation_cells=(
                args.layer_repulsion_minimum_normal_separation_cells
            ),
            layer_repulsion_scale=args.layer_repulsion_scale,
            layer_exclusion_proximity_radius_cells=(
                args.layer_exclusion_proximity_radius
            ),
            layer_exclusion_minimum_overlap_fraction=(
                args.layer_exclusion_minimum_overlap_fraction
            ),
            layer_exclusion_minimum_normal_separation_cells=(
                args.layer_exclusion_minimum_normal_separation_cells
            ),
            layer_exclusion_maximum_normal_angle_degrees=(
                args.layer_exclusion_maximum_normal_angle_degrees
            ),
        ),
        stratigraphic_candidates_root=args.stratigraphic_candidates,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _compile_sheet_evidence(args: argparse.Namespace) -> None:
    inputs = tuple(
        SheetEvidenceInput(
            Path(value[0]),
            tuple(int(coordinate) for coordinate in value[1:]),
        )
        for value in args.input
    )
    summary = compile_block_sheet_evidence(
        inputs,
        args.output,
        settings=SheetEvidenceSettings(
            clipping_tolerance_scale=args.clipping_tolerance_scale,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _catalog_sheet_correspondences(args: argparse.Namespace) -> None:
    last_report = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        bucket = completed // 500
        if bucket != last_report or completed == total:
            last_report = bucket
            print(f"mode correspondences {completed:,}/{total:,}", flush=True)

    summary = catalog_block_sheet_correspondences(
        args.evidence,
        args.cluster,
        args.output,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _compile_sheet_factors(args: argparse.Namespace) -> None:
    last_report = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        bucket = completed // 500
        if bucket != last_report or completed == total:
            last_report = bucket
            print(f"configuration factors {completed:,}/{total:,}", flush=True)

    summary = compile_sheet_configuration_factors(
        args.evidence,
        args.correspondences,
        args.cluster,
        args.output,
        settings=SheetFactorSettings(
            quarter_turn_penalty=args.quarter_turn_penalty,
        ),
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _initialize_sheet_configurations(args: argparse.Namespace) -> None:
    summary = run_sheet_configuration_initialization(
        args.evidence,
        args.factors,
        args.output,
        initial_root=args.initial,
        settings=SheetConfigurationSolverSettings(
            unary_scale=args.unary_scale,
            pairwise_scale=args.pairwise_scale,
            coverage_reward_scale=args.coverage_reward_scale,
            unmatched_trace_penalty=args.unmatched_trace_penalty,
            pairwise_normalization=args.pairwise_normalization,
            maximum_sweeps=args.maximum_sweeps,
            belief_propagation_iterations=args.belief_propagation_iterations,
            belief_propagation_damping=args.belief_propagation_damping,
            belief_propagation_tolerance=args.belief_propagation_tolerance,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _replay_joint_sheet_graph(args: argparse.Namespace) -> None:
    last_report = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        bucket = completed // 5000
        if bucket != last_report or completed == total:
            last_report = bucket
            print(f"active correspondences {completed:,}/{total:,}", flush=True)

    summary = replay_joint_sheet_graph(
        args.evidence,
        args.correspondences,
        args.configurations,
        args.cluster,
        args.output,
        stitching_settings=SheetStitchingSettings(
            minimum_join_benefit=args.minimum_join_benefit,
            quarter_turn_penalty=args.quarter_turn_penalty,
            unmatched_trace_penalty=args.unmatched_trace_penalty,
            restart_count=args.restarts,
            priority_jitter_fraction=args.priority_jitter_fraction,
            exchange_round_count=args.exchange_rounds,
            exchange_trials_per_round=args.exchange_trials_per_round,
            collision_cut_enabled=not args.no_collision_cut,
            collision_cut_limit=args.collision_cut_limit,
            collision_cut_order=args.collision_cut_order,
        ),
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _extract_sheet_evidence_subblock(args: argparse.Namespace) -> None:
    summary = extract_sheet_evidence_subblock(
        args.evidence,
        args.output,
        start_cell_xyz=tuple(args.start),
        stop_cell_xyz_exclusive=tuple(args.stop),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _crop_owned_sheet_graph(args: argparse.Namespace) -> None:
    summary = crop_surface_graph_to_owned_block(
        args.graph,
        args.output,
        start_cell_xyz=tuple(args.start),
        stop_cell_xyz_exclusive=tuple(args.stop),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _sheet_halo_experiment(args: argparse.Namespace) -> None:
    def progress(message: str) -> None:
        print(message, flush=True)

    summary = run_sheet_halo_experiment(
        args.evidence,
        args.cluster,
        args.output,
        core_start_cell_xyz=tuple(args.core_start),
        core_stop_cell_xyz_exclusive=tuple(args.core_stop),
        halo_cells=tuple(args.halos),
        configuration_settings=SheetConfigurationSolverSettings(
            unary_scale=args.unary_scale,
            pairwise_scale=args.pairwise_scale,
            coverage_reward_scale=args.coverage_reward_scale,
            unmatched_trace_penalty=args.unmatched_trace_penalty,
            pairwise_normalization=args.pairwise_normalization,
            maximum_sweeps=args.maximum_sweeps,
            belief_propagation_iterations=args.belief_propagation_iterations,
            belief_propagation_damping=args.belief_propagation_damping,
            belief_propagation_tolerance=args.belief_propagation_tolerance,
        ),
        stitching_settings=SheetStitchingSettings(
            minimum_join_benefit=args.minimum_join_benefit,
            quarter_turn_penalty=args.quarter_turn_penalty,
            unmatched_trace_penalty=args.stitching_unmatched_trace_penalty,
            restart_count=args.restarts,
            priority_jitter_fraction=args.priority_jitter_fraction,
            exchange_round_count=args.exchange_rounds,
            exchange_trials_per_round=args.exchange_trials_per_round,
            collision_cut_enabled=not args.no_collision_cut,
            collision_cut_limit=args.collision_cut_limit,
            collision_cut_order=args.collision_cut_order,
        ),
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _finalize_sheet_halo_experiment(args: argparse.Namespace) -> None:
    def progress(message: str) -> None:
        print(message, flush=True)

    summary = finalize_sheet_halo_experiment(
        args.experiment,
        cluster_root=args.cluster,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _audit_sheet_core(args: argparse.Namespace) -> None:
    summary = audit_sheet_core(
        args.evidence,
        args.correspondences,
        args.factors,
        args.configurations,
        args.graph,
        args.output,
        core_start_cell_xyz=tuple(args.core_start),
        maximum_hotspots=args.maximum_hotspots,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _refine_sheet_topology(args: argparse.Namespace) -> None:
    def progress(round_index: int, completed: int, total: int, label: str) -> None:
        print(
            f"topology round {round_index} · trial {completed:,}/{total:,} · {label}",
            flush=True,
        )

    summary = run_sheet_topology_refinement(
        args.evidence,
        args.correspondences,
        args.factors,
        args.configurations,
        args.graph,
        args.cluster,
        args.output,
        settings=SheetTopologyRefinementSettings(
            maximum_rounds=args.maximum_rounds,
            maximum_trials_per_round=args.maximum_trials_per_round,
            maximum_seed_moves=args.maximum_seed_moves,
            alternatives_per_pressure_cell=(
                args.alternatives_per_pressure_cell
            ),
            relaxation_radius=args.relaxation_radius,
            relaxation_sweeps=args.relaxation_sweeps,
            minimum_objective_gain=args.minimum_objective_gain,
            minimum_join_benefit=args.minimum_join_benefit,
            quarter_turn_penalty=args.quarter_turn_penalty,
        ),
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


def _refine_stratigraphic_continuity(args: argparse.Namespace) -> None:
    settings = StratigraphicContinuitySettings(
        neighborhood_radius_hops=args.neighborhood_radius,
        minimum_side_patches=args.minimum_side_patches,
        minimum_context_modes=args.minimum_context_modes,
        minimum_coverage_fraction=args.minimum_coverage_fraction,
        minimum_common_depth_span_voxels=args.minimum_common_depth_span,
        outlier_standard_deviations=args.outlier_standard_deviations,
        minimum_log_scale=args.minimum_log_scale,
    )
    last_fingerprint_report = -1
    last_join_report = -1

    def fingerprint_progress(completed: int, total: int) -> None:
        nonlocal last_fingerprint_report
        bucket = completed // 1000
        if bucket != last_fingerprint_report or completed == total:
            last_fingerprint_report = bucket
            print(
                f"stratigraphic fingerprints {completed:,}/{total:,}",
                flush=True,
            )

    def join_progress(completed: int, total: int) -> None:
        nonlocal last_join_report
        bucket = completed // 500
        if bucket != last_join_report or completed == total:
            last_join_report = bucket
            print(
                f"stratigraphic joins {completed:,}/{total:,}",
                flush=True,
            )

    summary = run_stratigraphic_continuity_refinement(
        args.root,
        args.mode_bank,
        args.output,
        sheet_evidence_root=args.sheet_evidence,
        candidate_restitch_root=args.candidate_restitch,
        settings=settings,
        join_refinement_root=args.join_refinement,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        force=args.force,
        fingerprint_progress=fingerprint_progress,
        join_progress=join_progress,
    )
    print(json.dumps(summary, indent=2))


def _contextual_growth(args: argparse.Namespace) -> None:
    settings = ContextualGrowthSettings(
        minimum_support_faces=args.minimum_support_faces,
        maximum_admission_robust_z=args.maximum_admission_robust_z,
        maximum_modes_per_trace=args.maximum_modes_per_trace,
        maximum_trials=args.maximum_trials,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    last_discovery_report = -1
    last_scoring_report = -1

    def discovery_progress(completed: int, total: int) -> None:
        nonlocal last_discovery_report
        bucket = completed // 1000
        if bucket != last_discovery_report or completed == total:
            last_discovery_report = bucket
            print(f"growth discovery {completed:,}/{total:,}", flush=True)

    def scoring_progress(completed: int, total: int) -> None:
        nonlocal last_scoring_report
        bucket = completed // 500
        if bucket != last_scoring_report or completed == total:
            last_scoring_report = bucket
            print(f"growth fingerprints {completed:,}/{total:,}", flush=True)

    def trial_progress(completed: int, total: int, candidate_id: int) -> None:
        print(
            f"growth topology trial {completed:,}/{total:,} · candidate {candidate_id}",
            flush=True,
        )

    summary = run_contextual_growth(
        args.stratigraphic_refinement,
        args.output,
        settings=settings,
        force=args.force,
        discovery_progress=discovery_progress,
        scoring_progress=scoring_progress,
        trial_progress=trial_progress,
    )
    print(json.dumps(summary, indent=2))


def _sheet_saturation(args: argparse.Namespace) -> None:
    settings = SheetSaturationSettings(
        distance_radii_voxels=tuple(args.distance_radii),
        joint_residual_limits=tuple(args.joint_residuals),
        assignment_share_thresholds=tuple(args.assignment_shares),
        primary_joint_residual_limit=args.primary_joint_residual,
        primary_confident_share=args.primary_confident_share,
        cell_overview_scale=args.overview_scale,
    )
    last_report = -1

    def progress(completed: int, total: int) -> None:
        nonlocal last_report
        bucket = completed // 128
        if bucket != last_report or completed == total:
            last_report = bucket
            print(f"sheet saturation {completed:,}/{total:,} cells", flush=True)

    summary = run_sheet_saturation_audit(
        args.root,
        args.output,
        settings=settings,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _saturation_reselection(args: argparse.Namespace) -> None:
    settings = SaturationReselectionSettings(
        joint_residual_limit=args.joint_residual,
        maximum_configurations_per_cell=args.maximum_configurations,
        maximum_configurations_per_coverage=args.maximum_per_coverage,
        pairwise_scale=args.pairwise_scale,
        interior_unmatched_trace_penalty=args.interior_unmatched_trace_penalty,
        maximum_sweeps=args.maximum_sweeps,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
    )
    last_report = -1

    def progress(completed: int, total: int, cell: tuple[int, int, int], paths: int) -> None:
        nonlocal last_report
        bucket = completed // 64
        if bucket != last_report or completed == total:
            last_report = bucket
            print(
                f"saturation reselection {completed:,}/{total:,} cells · "
                f"{cell} · {paths:,} physical paths",
                flush=True,
            )

    summary = run_saturation_reselection(
        args.root,
        args.mode_bank,
        args.output,
        settings=settings,
        force=args.force,
        progress=progress,
    )
    print(json.dumps(summary, indent=2))


def _saturation_candidate_selection(args: argparse.Namespace) -> None:
    settings = SaturationCandidateSelectionSettings(
        coverage_reward_scale=args.coverage_reward_scale,
        unary_scale=args.unary_scale,
        pairwise_scale=args.pairwise_scale,
        interior_unmatched_trace_penalty=args.interior_unmatched_trace_penalty,
        maximum_sweeps=args.maximum_sweeps,
        leaf_shape_cells_xyz=tuple(args.leaf_shape),
        write_visuals=not args.no_visuals,
    )
    summary = run_saturation_candidate_selection(
        args.candidates,
        args.output,
        settings=settings,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _dual_axis_packets(args: argparse.Namespace) -> None:
    summary = run_dual_axis_packet_connectivity(
        args.root,
        args.output,
        settings=DualAxisPacketSettings(
            leaf_shape_cells_xyz=tuple(args.leaf_shape),
            maximum_preview_components=args.maximum_preview_components,
            maximum_normal_angle_degrees=args.maximum_normal_angle,
            maximum_fiber_frame_residual_degrees=args.maximum_fiber_residual,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_band(args: argparse.Namespace) -> None:
    summary = run_boundary_band_export(
        args.root,
        args.output,
        packet_root=args.packet_root,
        candidate_root=args.candidate_root,
        settings=BoundaryBandSettings(
            depth_cells=args.depth_cells,
            leaf_shape_cells_xyz=tuple(args.leaf_shape),
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_merge(args: argparse.Namespace) -> None:
    summary = run_boundary_band_merge(
        args.first,
        args.second,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_reselection(args: argparse.Namespace) -> None:
    summary = run_boundary_band_reselection(
        args.first,
        args.second,
        args.output,
        settings=BoundaryReselectionSettings(
            unary_scale=args.unary_scale,
            pairwise_scale=args.pairwise_scale,
            interior_unmatched_trace_penalty=(
                args.interior_unmatched_trace_penalty
            ),
            maximum_sweeps=args.maximum_sweeps,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_cluster_reselection(args: argparse.Namespace) -> None:
    summary = run_boundary_cluster_reselection(
        tuple(args.boundary),
        args.output,
        settings=BoundaryReselectionSettings(
            unary_scale=args.unary_scale,
            pairwise_scale=args.pairwise_scale,
            interior_unmatched_trace_penalty=(
                args.interior_unmatched_trace_penalty
            ),
            maximum_sweeps=args.maximum_sweeps,
        ),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _selected_subblock(args: argparse.Namespace) -> None:
    summary = extract_selected_patch_subblock(
        args.root,
        args.output,
        start_cell_xyz=tuple(args.start),
        stop_cell_xyz_exclusive=tuple(args.stop),
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_split_audit(args: argparse.Namespace) -> None:
    summary = run_boundary_split_audit(
        args.full_packet_root,
        args.merge_root,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _boundary_reselection_audit(args: argparse.Namespace) -> None:
    summary = run_boundary_reselection_split_audit(
        args.full_packet_root,
        args.reselection_root,
        args.output,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _independent_boundary_audit(args: argparse.Namespace) -> None:
    summary = run_independent_boundary_audit(
        args.full_packet_root,
        args.selected_merge_root,
        args.reselection_root,
        args.output,
        height_tolerance_voxels=args.height_tolerance,
        normal_tolerance_degrees=args.normal_tolerance,
        fiber_tolerance_degrees=args.fiber_tolerance,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _multiseam_audit(args: argparse.Namespace) -> None:
    summary = run_multiseam_audit(
        tuple(args.reselection),
        args.output,
        cluster_root=args.cluster_root,
        force=args.force,
    )
    print(json.dumps(summary, indent=2))


def _cluster_reference_audit(args: argparse.Namespace) -> None:
    summary = run_cluster_reference_audit(
        args.full_packet_root,
        args.cluster_root,
        args.output,
        full_selected_root=args.full_selected_root,
        height_tolerance_voxels=args.height_tolerance,
        normal_tolerance_degrees=args.normal_tolerance,
        fiber_tolerance_degrees=args.fiber_tolerance,
        force=args.force,
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
        "--calibration",
        type=Path,
        help=(
            "optional source-level calibration-v1.json to reuse across "
            "independently inferred adjacent blocks"
        ),
    )
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
    needle_field = subparsers.add_parser(
        "solve-block-needle-field",
        description=(
            "Load every unique canonical Acus needle in one world-space cuboid "
            "and jointly optimize a robust unsigned page-normal field without "
            "independent cell fitting."
        ),
    )
    needle_field.add_argument(
        "--raw-root",
        type=Path,
        action="append",
        required=True,
        help="complete raw-Acus pipeline root; repeat for adjacent source blocks",
    )
    needle_field.add_argument(
        "--world-start", nargs=3, type=float, required=True
    )
    needle_field.add_argument(
        "--world-stop", nargs=3, type=float, required=True
    )
    needle_field.add_argument("--output", type=Path, required=True)
    needle_field.add_argument(
        "--settings-json",
        type=Path,
        help=(
            "optional BlockNeedleFieldSettings keyword object; explicit command "
            "options override matching JSON values"
        ),
    )
    needle_field.add_argument("--neighbor-radius", type=float)
    needle_field.add_argument("--maximum-neighbors", type=int)
    needle_field.add_argument("--spatial-kernel", type=float)
    needle_field.add_argument("--smoothing-weight", type=float)
    needle_field.add_argument(
        "--robust-smoothing-angle", type=float
    )
    needle_field.add_argument("--iterations", type=int)
    needle_field.add_argument("--damping", type=float)
    needle_field.add_argument("--maximum-normal-hypotheses", type=int)
    needle_field.add_argument("--minimum-candidate-cross-angle", type=float)
    needle_field.add_argument("--candidate-residual-kernel", type=float)
    needle_field.add_argument("--candidate-separation", type=float)
    needle_field.add_argument("--tangent-compatibility-sigma", type=float)
    needle_field.add_argument("--mixture-pairwise-weight", type=float)
    needle_field.add_argument("--mixture-initial-temperature", type=float)
    needle_field.add_argument("--mixture-temperature", type=float)
    needle_field.add_argument("--mixture-damping", type=float)
    needle_field.add_argument("--mixture-iterations", type=int)
    needle_field.add_argument("--mixture-annealing-iterations", type=int)
    needle_field.add_argument("--carrier-minimum-affinity", type=float)
    needle_field.add_argument("--minimum-curvature-radius", type=float)
    needle_field.add_argument(
        "--compute", choices=("auto", "gpu", "cpu")
    )
    needle_field.add_argument("--force", action="store_true")
    needle_field.set_defaults(handler=_block_needle_field)
    needle_topology = subparsers.add_parser(
        "solve-block-needle-topology",
        description=(
            "Build one curvature-aware, stack-fingerprinted ply topology graph "
            "over every needle in a completed block-global field."
        ),
    )
    needle_topology.add_argument("--field", type=Path, required=True)
    needle_topology.add_argument("--output", type=Path, required=True)
    needle_topology.add_argument(
        "--settings-json",
        type=Path,
        help="optional BlockNeedleTopologySettings keyword object",
    )
    needle_topology.add_argument("--force", action="store_true")
    needle_topology.set_defaults(handler=_block_needle_topology)
    needle_surfaces = subparsers.add_parser(
        "solve-block-needle-surfaces",
        description=(
            "Integrate one block-global needle topology into ordered fiber traces, "
            "intrinsic carrier charts, and physically gated manifold triangle meshes."
        ),
    )
    needle_surfaces.add_argument("--topology", type=Path, required=True)
    needle_surfaces.add_argument("--output", type=Path, required=True)
    needle_surfaces.add_argument(
        "--settings-json",
        type=Path,
        help="optional BlockNeedleSurfaceSettings keyword object",
    )
    needle_surfaces.add_argument("--force", action="store_true")
    needle_surfaces.set_defaults(handler=_block_needle_surfaces)
    needle_flattening = subparsers.add_parser(
        "flatten-block-needle-surfaces",
        description=(
            "Rasterize leading intrinsic needle surfaces and sample their native "
            "CT depth stacks for qualitative physical validation."
        ),
    )
    needle_flattening.add_argument("--surfaces", type=Path, required=True)
    needle_flattening.add_argument("--output", type=Path, required=True)
    needle_flattening.add_argument(
        "--grouping",
        choices=("surface-component", "topology-carrier"),
        default="surface-component",
        help="flatten individual mesh islands or all islands in one ply carrier",
    )
    needle_flattening.add_argument("--maximum-components", type=int, default=12)
    needle_flattening.add_argument("--pixel-step", type=float, default=0.5)
    needle_flattening.add_argument("--maximum-pixels", type=int, default=768)
    needle_flattening.add_argument("--depth-min", type=float, default=-12.0)
    needle_flattening.add_argument("--depth-max", type=float, default=12.0)
    needle_flattening.add_argument("--depth-step", type=float, default=1.0)
    needle_flattening.add_argument("--force", action="store_true")
    needle_flattening.set_defaults(handler=_block_needle_flattening)
    needle_bundles = subparsers.add_parser(
        "associate-block-needle-surfaces",
        description=(
            "Build evidence-only orthogonal-ply packets and shadow bridges between "
            "disconnected surface islands without changing their meshes."
        ),
    )
    needle_bundles.add_argument("--surfaces", type=Path, required=True)
    needle_bundles.add_argument("--output", type=Path, required=True)
    needle_bundles.add_argument(
        "--settings-json",
        type=Path,
        help="optional BlockNeedleBundleSettings keyword object",
    )
    needle_bundles.add_argument("--force", action="store_true")
    needle_bundles.set_defaults(handler=_block_needle_bundles)
    isolated_slabs = subparsers.add_parser(
        "detect-isolated-slabs",
        description=(
            "Detect dense, high-confidence air-papyrus-air slabs directly from "
            "native CT by pairing opposing interfaces with physical thickness "
            "and external-air checks. This stage is intentionally independent "
            "of Acus needles."
        ),
    )
    isolated_slabs.add_argument("--source", type=Path, required=True)
    isolated_slabs.add_argument("--metadata", type=Path)
    isolated_slabs.add_argument("--world-start", nargs=3, type=int, required=True)
    isolated_slabs.add_argument("--world-stop", nargs=3, type=int, required=True)
    isolated_slabs.add_argument("--output", type=Path, required=True)
    isolated_slabs.add_argument(
        "--settings-json",
        type=Path,
        help="optional IsolatedSlabSettings keyword object",
    )
    isolated_slabs.add_argument("--force", action="store_true")
    isolated_slabs.set_defaults(handler=_isolated_slabs)
    isolated_slab_acus_audit = subparsers.add_parser(
        "audit-isolated-slabs-with-acus",
        description=(
            "Measure how densely the finite Acus needles sample conservative "
            "CT-derived isolated slabs, without modifying either artifact."
        ),
    )
    isolated_slab_acus_audit.add_argument("--slabs", type=Path, required=True)
    isolated_slab_acus_audit.add_argument("--field", type=Path, required=True)
    isolated_slab_acus_audit.add_argument("--output", type=Path, required=True)
    isolated_slab_acus_audit.add_argument(
        "--settings-json",
        type=Path,
        help="optional IsolatedSlabAcusAuditSettings keyword object",
    )
    isolated_slab_acus_audit.add_argument("--force", action="store_true")
    isolated_slab_acus_audit.set_defaults(handler=_audit_isolated_slab_acus)
    paired_surface_bank = subparsers.add_parser(
        "build-paired-surface-bank",
        description=(
            "Reify every physically bounded CT profile behind conservative "
            "isolated slabs, retaining distinct paired-interface alternatives "
            "without selecting sheet identities."
        ),
    )
    paired_surface_bank.add_argument("--slabs", type=Path, required=True)
    paired_surface_bank.add_argument("--output", type=Path, required=True)
    paired_surface_bank.add_argument(
        "--settings-json",
        type=Path,
        help="optional PairedSurfaceBankSettings keyword object",
    )
    paired_surface_bank.add_argument("--force", action="store_true")
    paired_surface_bank.set_defaults(handler=_paired_surface_bank)
    paired_surface_growth = subparsers.add_parser(
        "grow-paired-surfaces",
        description=(
            "Grow immutable isolated-slab seeds through the broader paired CT "
            "candidate bank using lower/upper boundary continuity, thickness, "
            "curvature, and source-lattice mutual exclusion."
        ),
    )
    paired_surface_growth.add_argument("--bank", type=Path, required=True)
    paired_surface_growth.add_argument("--output", type=Path, required=True)
    paired_surface_growth.add_argument(
        "--settings-json",
        type=Path,
        help="optional PairedSurfaceGrowthSettings keyword object",
    )
    paired_surface_growth.add_argument("--force", action="store_true")
    paired_surface_growth.set_defaults(handler=_paired_surface_growth)
    one_sided_interface = subparsers.add_parser(
        "build-one-sided-interface-bank",
        description=(
            "Extract signed air-to-material CT interfaces without requiring "
            "an opposite face, and anchor them to both exact faces of the "
            "selected paired-surface result."
        ),
    )
    one_sided_interface.add_argument("--growth", type=Path, required=True)
    one_sided_interface.add_argument("--output", type=Path, required=True)
    one_sided_interface.add_argument(
        "--settings-json",
        type=Path,
        help="optional OneSidedInterfaceSettings keyword object",
    )
    one_sided_interface.add_argument("--force", action="store_true")
    one_sided_interface.set_defaults(handler=_one_sided_interface_bank)
    one_sided_growth = subparsers.add_parser(
        "grow-one-sided-interfaces",
        description=(
            "Grow paired-surface identities over signed one-sided material "
            "interfaces, associate only bilaterally supported fragments, "
            "and preserve multi-identity components as unresolved."
        ),
    )
    one_sided_growth.add_argument("--bank", type=Path, required=True)
    one_sided_growth.add_argument("--output", type=Path, required=True)
    one_sided_growth.add_argument(
        "--settings-json",
        type=Path,
        help="optional OneSidedGrowthSettings keyword object",
    )
    one_sided_growth.add_argument("--force", action="store_true")
    one_sided_growth.set_defaults(handler=_one_sided_interface_growth)
    clear_ribbon_bank = subparsers.add_parser(
        "build-clear-ribbon-bank",
        description=(
            "Map both faces of every physically bounded paired profile onto "
            "strong signed interfaces, collapse reciprocal duplicates, and "
            "catalog strict two-boundary components without selecting new "
            "alternatives."
        ),
    )
    clear_ribbon_bank.add_argument("--growth", type=Path, required=True)
    clear_ribbon_bank.add_argument("--output", type=Path, required=True)
    clear_ribbon_bank.add_argument(
        "--settings-json",
        type=Path,
        help="optional ClearRibbonSettings keyword object",
    )
    clear_ribbon_bank.add_argument("--force", action="store_true")
    clear_ribbon_bank.set_defaults(handler=_clear_ribbon_bank)
    clear_ribbon_selection = subparsers.add_parser(
        "select-clear-ribbons",
        description=(
            "Select a collision-safe maximum-bottleneck forest from the exact "
            "two-face ribbon bank, preserving trusted anchors, deferring "
            "contested interiors, and assigning new identities only to "
            "substantial unseeded cores."
        ),
    )
    clear_ribbon_selection.add_argument("--bank", type=Path, required=True)
    clear_ribbon_selection.add_argument("--output", type=Path, required=True)
    clear_ribbon_selection.add_argument(
        "--settings-json",
        type=Path,
        help="optional ClearRibbonSelectionSettings keyword object",
    )
    clear_ribbon_selection.add_argument("--force", action="store_true")
    clear_ribbon_selection.set_defaults(handler=_clear_ribbon_selection)
    clear_ribbon_feedback = subparsers.add_parser(
        "grow-clear-ribbon-interfaces",
        description=(
            "Feed genuinely new two-face ribbon cores back into the dense "
            "signed-interface graph, rebuild seed eligibility, and grow only "
            "where all prior assignments remain unchanged."
        ),
    )
    clear_ribbon_feedback.add_argument(
        "--selection", type=Path, required=True
    )
    clear_ribbon_feedback.add_argument("--output", type=Path, required=True)
    clear_ribbon_feedback.add_argument("--force", action="store_true")
    clear_ribbon_feedback.set_defaults(handler=_clear_ribbon_feedback)
    clear_ribbon_paired_feedback = subparsers.add_parser(
        "grow-clear-ribbon-paired-profiles",
        description=(
            "Use interface-validated new clear cores as seeds in the full "
            "physical paired-profile graph while freezing every baseline "
            "selection and deferring multi-label free components."
        ),
    )
    clear_ribbon_paired_feedback.add_argument(
        "--feedback", type=Path, required=True
    )
    clear_ribbon_paired_feedback.add_argument(
        "--output", type=Path, required=True
    )
    clear_ribbon_paired_feedback.add_argument("--force", action="store_true")
    clear_ribbon_paired_feedback.set_defaults(
        handler=_clear_ribbon_paired_feedback
    )
    clear_core_interface_refinement = subparsers.add_parser(
        "grow-paired-feedback-interfaces",
        description=(
            "Freeze all current signed-interface assignments, add only safe "
            "unowned endpoints from paired-profile feedback, and grow "
            "unambiguous interface components."
        ),
    )
    clear_core_interface_refinement.add_argument(
        "--paired-feedback", type=Path, required=True
    )
    clear_core_interface_refinement.add_argument(
        "--output", type=Path, required=True
    )
    clear_core_interface_refinement.add_argument(
        "--force", action="store_true"
    )
    clear_core_interface_refinement.set_defaults(
        handler=_clear_core_interface_refinement
    )
    physical_ribbon_bank = subparsers.add_parser(
        "build-physical-ribbon-bank",
        description=(
            "Pair dense signed CT interfaces using only papyrus thickness and "
            "opposing-boundary geometry. Prior sheet labels are ignored and "
            "all physically plausible alternatives remain explicit."
        ),
    )
    physical_ribbon_bank.add_argument(
        "--interfaces", type=Path, required=True
    )
    physical_ribbon_bank.add_argument("--output", type=Path, required=True)
    physical_ribbon_bank.add_argument("--settings-json", type=Path)
    physical_ribbon_bank.add_argument("--force", action="store_true")
    physical_ribbon_bank.set_defaults(handler=_physical_ribbon_bank)
    physical_ribbon_continuity = subparsers.add_parser(
        "solve-physical-ribbon-continuity",
        description=(
            "Infer label-free papyrus components from simultaneous continuity "
            "of both physical ribbon boundaries, local two-dimensional support, "
            "and exclusive use of observed interfaces."
        ),
    )
    physical_ribbon_continuity.add_argument(
        "--ribbons", type=Path, required=True
    )
    physical_ribbon_continuity.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_continuity.add_argument("--settings-json", type=Path)
    physical_ribbon_continuity.add_argument("--force", action="store_true")
    physical_ribbon_continuity.set_defaults(
        handler=_physical_ribbon_continuity
    )
    physical_ribbon_configuration = subparsers.add_parser(
        "optimize-physical-ribbon-configuration",
        description=(
            "Choose a label-free ribbon configuration under one-interface, "
            "paired-boundary continuity, first-hit, and exact profile "
            "non-intersection factors while retaining rejected alternatives."
        ),
    )
    physical_ribbon_configuration.add_argument(
        "--continuity", type=Path, required=True
    )
    physical_ribbon_configuration.add_argument(
        "--topology-continuity",
        type=Path,
        help=(
            "Optional stricter continuity graph used to define component "
            "identity while --continuity supplies broader support votes."
        ),
    )
    physical_ribbon_configuration.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_configuration.add_argument("--settings-json", type=Path)
    physical_ribbon_configuration.add_argument("--force", action="store_true")
    physical_ribbon_configuration.set_defaults(
        handler=_physical_ribbon_configuration
    )
    physical_ribbon_collective = subparsers.add_parser(
        "optimize-physical-ribbon-collective-patches",
        description=(
            "Find coherent multi-ribbon residual regions and optimize each as "
            "one collective surface move, crossing unary energy barriers while "
            "preserving interface, crossing, and prior-sheet constraints."
        ),
    )
    physical_ribbon_collective.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_collective.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_collective.add_argument("--settings-json", type=Path)
    physical_ribbon_collective.add_argument("--force", action="store_true")
    physical_ribbon_collective.set_defaults(
        handler=_physical_ribbon_collective
    )
    physical_ribbon_flattened_audit = subparsers.add_parser(
        "audit-physical-ribbon-flat-texture",
        description=(
            "Flatten exact physical-ribbon components, sample native CT at "
            "fixed ply depths, and report local axial texture continuity around "
            "newly admitted surface patches."
        ),
    )
    physical_ribbon_flattened_audit.add_argument(
        "--surface", type=Path, required=True
    )
    physical_ribbon_flattened_audit.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_flattened_audit.add_argument("--settings-json", type=Path)
    physical_ribbon_flattened_audit.add_argument("--force", action="store_true")
    physical_ribbon_flattened_audit.set_defaults(
        handler=_physical_ribbon_flattened_audit
    )
    physical_ribbon_patch_states = subparsers.add_parser(
        "optimize-physical-ribbon-patch-states",
        description=(
            "Jointly reconfigure every CT-supported closed-hole frontier on "
            "one reconstructed surface component inside a fixed selected "
            "halo, enumerate heuristic and binary-optimized complete "
            "matchings, then retain only exact topology or mesh-density "
            "improvements."
        ),
    )
    physical_ribbon_patch_states.add_argument(
        "--holes", type=Path, required=True
    )
    physical_ribbon_patch_states.add_argument(
        "--depth-field",
        type=Path,
        help=(
            "Optional dense native-CT ordered-label field measured on the "
            "same holes; enables saturated whole-patch coverage optimization."
        ),
    )
    physical_ribbon_patch_states.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_patch_states.add_argument("--settings-json", type=Path)
    physical_ribbon_patch_states.add_argument("--force", action="store_true")
    physical_ribbon_patch_states.set_defaults(
        handler=_physical_ribbon_patch_states
    )
    physical_ribbon_texture_gate = subparsers.add_parser(
        "gate-physical-ribbon-patch-texture",
        description=(
            "Retain exact component-level patch states only when their "
            "flattened native-CT boundary texture is compatible with control "
            "edges from the same surface."
        ),
    )
    physical_ribbon_texture_gate.add_argument(
        "--patch-state", type=Path, required=True
    )
    physical_ribbon_texture_gate.add_argument(
        "--texture-audit", type=Path, required=True
    )
    physical_ribbon_texture_gate.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_texture_gate.add_argument("--force", action="store_true")
    physical_ribbon_texture_gate.set_defaults(
        handler=_physical_ribbon_texture_gate
    )
    physical_ribbon_bridging = subparsers.add_parser(
        "bridge-physical-ribbon-components",
        description=(
            "Merge label-free ribbon fragments only through connected bundles "
            "of multiple two-face bridge candidates while preserving interface "
            "exclusivity and exact profile non-intersection."
        ),
    )
    physical_ribbon_bridging.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_bridging.add_argument(
        "--bridge-continuity",
        type=Path,
        help=(
            "Optional broader continuation graph used only to propose guarded "
            "multi-path bridges."
        ),
    )
    physical_ribbon_bridging.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_bridging.add_argument("--settings-json", type=Path)
    physical_ribbon_bridging.add_argument("--force", action="store_true")
    physical_ribbon_bridging.set_defaults(
        handler=_physical_ribbon_bridging
    )
    physical_ribbon_patch_holes = subparsers.add_parser(
        "analyze-physical-ribbon-patch-holes",
        description=(
            "Construct intrinsic meshes for selected physical ribbons, identify "
            "closed multi-ribbon holes, and test jointly fitted surface patches "
            "against native CT and normal-offset competing layers."
        ),
    )
    physical_ribbon_patch_holes.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_patch_holes.add_argument(
        "--surface-replay",
        type=Path,
        help=(
            "Exact cumulative replay whose strict-plus-native-CT surface is "
            "used for the hole census. The configuration must be its "
            "materialized state."
        ),
    )
    physical_ribbon_patch_holes.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_patch_holes.add_argument("--settings-json", type=Path)
    physical_ribbon_patch_holes.add_argument("--force", action="store_true")
    physical_ribbon_patch_holes.set_defaults(
        handler=_physical_ribbon_patch_holes
    )
    physical_ribbon_surface_holes = subparsers.add_parser(
        "analyze-physical-ribbon-surface-holes",
        description=(
            "Find complete interior loops on any materialized label-free "
            "surface and score their full missing patches directly against "
            "native CT without requiring ribbon-bank candidates."
        ),
    )
    physical_ribbon_surface_holes.add_argument(
        "--surface", type=Path, required=True
    )
    physical_ribbon_surface_holes.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_surface_holes.add_argument("--settings-json", type=Path)
    physical_ribbon_surface_holes.add_argument("--force", action="store_true")
    physical_ribbon_surface_holes.set_defaults(
        handler=_physical_ribbon_surface_holes
    )
    physical_ribbon_open_bays = subparsers.add_parser(
        "analyze-physical-ribbon-open-bays",
        description=(
            "Find compact concave bays on outer surface frontiers, rank "
            "complete multi-edge arc replacements by area gain and boundary "
            "shortening, and score each whole bay against native CT."
        ),
    )
    physical_ribbon_open_bays.add_argument(
        "--surface", type=Path, required=True
    )
    physical_ribbon_open_bays.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_open_bays.add_argument("--settings-json", type=Path)
    physical_ribbon_open_bays.add_argument(
        "--prior-completion",
        action="append",
        type=Path,
        default=[],
        help=(
            "prior dense completion whose unchanged intrinsic failures should "
            "not consume the current scoring cap; repeat as needed"
        ),
    )
    physical_ribbon_open_bays.add_argument(
        "--prior-texture-audit",
        action="append",
        type=Path,
        default=[],
        help=(
            "prior flattened audit whose unchanged incompatible or unmeasured "
            "proposals should be excluded; repeat as needed"
        ),
    )
    physical_ribbon_open_bays.add_argument("--force", action="store_true")
    physical_ribbon_open_bays.set_defaults(
        handler=_physical_ribbon_open_bays
    )
    physical_ribbon_open_bay_saturation = subparsers.add_parser(
        "saturate-physical-ribbon-open-bays",
        description=(
            "Iterate complete multi-edge open-bay reconstruction, dense "
            "native-CT depth solving, exact surface admission, and flattened "
            "fiber gating until no uncached supported state remains."
        ),
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--surface", type=Path, required=True
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--prior-completion",
        action="append",
        type=Path,
        default=[],
        help=(
            "optional prior dense completion providing evidence-hashed exact "
            "failures; repeat as needed"
        ),
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--prior-texture-audit",
        action="append",
        type=Path,
        default=[],
        help=(
            "optional prior flattened audit providing proposal-local rejected "
            "fiber states; repeat as needed"
        ),
    )
    physical_ribbon_open_bay_saturation.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_open_bay_saturation.set_defaults(
        handler=_physical_ribbon_open_bay_saturation
    )
    physical_ribbon_depth_fields = subparsers.add_parser(
        "analyze-physical-ribbon-depth-fields",
        description=(
            "Sample dense native-CT normal-depth likelihoods across each "
            "complete residual hole or outer-frontier bay, then solve one "
            "coherent ordered-label field and classify its support."
        ),
    )
    physical_ribbon_depth_fields.add_argument(
        "--holes", type=Path, required=True
    )
    physical_ribbon_depth_fields.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_depth_fields.add_argument("--settings-json", type=Path)
    physical_ribbon_depth_fields.add_argument("--force", action="store_true")
    physical_ribbon_depth_fields.set_defaults(
        handler=_physical_ribbon_depth_fields
    )
    physical_ribbon_dense_completion = subparsers.add_parser(
        "complete-physical-ribbon-dense-surfaces",
        description=(
            "Promote each complete collective native-CT depth field to one "
            "boundary-exact constrained surface patch, retaining it only "
            "when whole-state manifold, chart, competing-layer, and CT "
            "audits pass."
        ),
    )
    physical_ribbon_dense_completion.add_argument(
        "--holes", type=Path, required=True
    )
    physical_ribbon_dense_completion.add_argument(
        "--depth-field", type=Path, required=True
    )
    physical_ribbon_dense_completion.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_dense_completion.add_argument(
        "--texture-audit",
        type=Path,
        help=(
            "optional flattened audit of an ungated completion; replay only "
            "proposal-local texture-compatible hole rows"
        ),
    )
    physical_ribbon_dense_completion.add_argument("--settings-json", type=Path)
    physical_ribbon_dense_completion.add_argument("--force", action="store_true")
    physical_ribbon_dense_completion.set_defaults(
        handler=_physical_ribbon_dense_completion
    )
    physical_ribbon_patch_corridors = subparsers.add_parser(
        "analyze-physical-ribbon-patch-corridors",
        description=(
            "Find mutually facing multi-edge arcs on open physical-ribbon "
            "boundaries, fit bend-aware corridor surfaces, test them against "
            "native CT and fiber continuation, and replay only exact closures."
        ),
    )
    physical_ribbon_patch_corridors.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_patch_corridors.add_argument(
        "--surface-replay",
        type=Path,
        help=(
            "Exact cumulative strip replay whose strict-plus-native-CT surface "
            "is preserved during the refreshed corridor census. The "
            "configuration must have been materialized from this replay."
        ),
    )
    physical_ribbon_patch_corridors.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_patch_corridors.add_argument("--settings-json", type=Path)
    physical_ribbon_patch_corridors.add_argument("--force", action="store_true")
    physical_ribbon_patch_corridors.set_defaults(
        handler=_physical_ribbon_patch_corridors
    )
    physical_ribbon_corridor_variants = subparsers.add_parser(
        "analyze-physical-ribbon-corridor-variants",
        description=(
            "Enumerate diverse complete interface matchings for CT-supported "
            "corridors, reconstruct each in its full sheet, and globally replay "
            "only exact density-preserving variants."
        ),
    )
    physical_ribbon_corridor_variants.add_argument(
        "--corridors", type=Path, required=True
    )
    physical_ribbon_corridor_variants.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_corridor_variants.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_variants.add_argument("--settings-json", type=Path)
    physical_ribbon_corridor_variants.add_argument("--force", action="store_true")
    physical_ribbon_corridor_variants.set_defaults(
        handler=_physical_ribbon_corridor_variants
    )
    physical_ribbon_corridor_sets = subparsers.add_parser(
        "optimize-physical-ribbon-corridor-sets",
        description=(
            "Jointly optimize exact corridor variants inside each physical "
            "sheet and across the block before counterfactual replay."
        ),
    )
    physical_ribbon_corridor_sets.add_argument(
        "--variants", type=Path, required=True
    )
    physical_ribbon_corridor_sets.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_corridor_sets.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_sets.add_argument("--settings-json", type=Path)
    physical_ribbon_corridor_sets.add_argument("--force", action="store_true")
    physical_ribbon_corridor_sets.set_defaults(
        handler=_physical_ribbon_corridor_sets
    )
    physical_ribbon_corridor_extension = subparsers.add_parser(
        "extend-physical-ribbon-corridor-variants",
        description=(
            "Extend a cached exact corridor bank to deeper complete matchings, "
            "screening only variants from previously unresolved corridors."
        ),
    )
    physical_ribbon_corridor_extension.add_argument(
        "--variants", type=Path, required=True
    )
    physical_ribbon_corridor_extension.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_corridor_extension.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_extension.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_extension.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_extension.set_defaults(
        handler=_physical_ribbon_corridor_extension
    )
    physical_ribbon_corridor_dormant = subparsers.add_parser(
        "analyze-physical-ribbon-dormant-corridors",
        description=(
            "Condition a deeper strict continuity frontier on an unchanged "
            "selected surface, then exact-screen only complete residual "
            "corridor states that add formerly dormant bidirectional ribbons."
        ),
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--corridors", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--variants", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--corridor-sets", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--expanded-continuity", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_dormant.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_dormant.set_defaults(
        handler=_physical_ribbon_corridor_dormant
    )
    physical_ribbon_corridor_frontier = subparsers.add_parser(
        "build-physical-ribbon-corridor-frontier",
        description=(
            "Add one-sided ribbon hypotheses only inside still-unresolved "
            "native-CT corridor strips, then build their strict continuation "
            "and crossing topology without mutating the cumulative replay."
        ),
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--corridors", type=Path, required=True
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--prior-replay", type=Path, required=True
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--configuration", type=Path, required=True
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--bidirectional-continuity", type=Path, required=True
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_frontier.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_frontier.set_defaults(
        handler=_physical_ribbon_corridor_frontier
    )
    physical_ribbon_corridor_one_sided = subparsers.add_parser(
        "analyze-physical-ribbon-one-sided-corridors",
        description=(
            "Enumerate complete residual-corridor matchings that use the "
            "targeted one-sided frontier, reconstruct them in their full "
            "sheets, optimize compatible states, and replay cumulatively."
        ),
    )
    physical_ribbon_corridor_one_sided.add_argument(
        "--frontier", type=Path, required=True
    )
    physical_ribbon_corridor_one_sided.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_one_sided.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_one_sided.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_one_sided.set_defaults(
        handler=_physical_ribbon_corridor_one_sided
    )
    physical_ribbon_replay_configuration = subparsers.add_parser(
        "materialize-physical-ribbon-replay-configuration",
        description=(
            "Materialize a cumulative exact corridor replay as an ordinary "
            "strict continuity and configuration pair for the next pipeline "
            "iteration."
        ),
    )
    physical_ribbon_replay_configuration.add_argument(
        "--replay", type=Path, required=True
    )
    physical_ribbon_replay_configuration.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_replay_configuration.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_replay_configuration.set_defaults(
        handler=_physical_ribbon_replay_configuration
    )
    physical_ribbon_corridor_saturation = subparsers.add_parser(
        "assess-physical-ribbon-corridor-saturation",
        description=(
            "Compare consecutive CT corridor iterations and prove when prior "
            "exact failures can be reused without another reconstruction pass."
        ),
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--prior-corridors", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--prior-frontier", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--prior-replay", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--current-corridors", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--current-frontier", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_saturation.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_saturation.set_defaults(
        handler=_physical_ribbon_corridor_saturation
    )
    physical_ribbon_corridor_deficits = subparsers.add_parser(
        "analyze-physical-ribbon-corridor-deficits",
        description=(
            "Reconstruct the best failed exact state for each residual CT "
            "strip and measure its dense support and triangulation deficits."
        ),
    )
    physical_ribbon_corridor_deficits.add_argument(
        "--replay", type=Path, required=True
    )
    physical_ribbon_corridor_deficits.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_deficits.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_deficits.set_defaults(
        handler=_physical_ribbon_corridor_deficits
    )
    physical_ribbon_corridor_faces = subparsers.add_parser(
        "analyze-physical-ribbon-corridor-faces",
        description=(
            "Repair residual CT corridors with physically gated faces from "
            "the existing chart Delaunay tessellation while leaving strict "
            "ribbon graph identity unchanged."
        ),
    )
    physical_ribbon_corridor_faces.add_argument(
        "--replay", type=Path, required=True
    )
    physical_ribbon_corridor_faces.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_faces.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_faces.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_faces.set_defaults(
        handler=_physical_ribbon_corridor_faces
    )
    physical_ribbon_corridor_face_replay = subparsers.add_parser(
        "replay-physical-ribbon-corridor-faces",
        description=(
            "Jointly choose compatible CT-backed corridor assignments, "
            "rebuild their strict sheets once, and attach supplemental mesh "
            "faces without promoting those faces to topology edges."
        ),
    )
    physical_ribbon_corridor_face_replay.add_argument(
        "--faces", type=Path, required=True
    )
    physical_ribbon_corridor_face_replay.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_corridor_face_replay.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_corridor_face_replay.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_corridor_face_replay.set_defaults(
        handler=_physical_ribbon_corridor_face_replay
    )
    physical_ribbon_complete_strips = subparsers.add_parser(
        "analyze-physical-ribbon-complete-strips",
        description=(
            "Enumerate complete both-arc matchings for residual native-CT "
            "strips without filtering by candidate provenance, then screen "
            "their topology and physically supported surface closure."
        ),
    )
    physical_ribbon_complete_strips.add_argument(
        "--replay", type=Path, required=True
    )
    physical_ribbon_complete_strips.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_complete_strips.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_complete_strips.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_complete_strips.set_defaults(
        handler=_physical_ribbon_complete_strips
    )
    physical_ribbon_complete_strip_replay = subparsers.add_parser(
        "replay-physical-ribbon-complete-strips",
        description=(
            "Jointly replay physically eligible complete-strip assignments, "
            "then recompute every prior and new CT-supported face closure in "
            "the final shared sheet charts."
        ),
    )
    physical_ribbon_complete_strip_replay.add_argument(
        "--strips", type=Path, required=True
    )
    physical_ribbon_complete_strip_replay.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_complete_strip_replay.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_complete_strip_replay.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_complete_strip_replay.set_defaults(
        handler=_physical_ribbon_complete_strip_replay
    )
    physical_ribbon_lineage_strips = subparsers.add_parser(
        "analyze-physical-ribbon-lineage-strips",
        description=(
            "Rescan unresolved complete native-CT strip assignments while "
            "requiring the entire inherited strict sheet lineage to remain "
            "connected, then apply the physical surface gates."
        ),
    )
    physical_ribbon_lineage_strips.add_argument(
        "--replay", type=Path, required=True
    )
    physical_ribbon_lineage_strips.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_lineage_strips.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_lineage_strips.add_argument(
        "--corridor-row",
        type=int,
        action="append",
        help=(
            "explicit scored corridor row to audit; repeat for multiple rows; "
            "defaults to prior split-lineage failures"
        ),
    )
    physical_ribbon_lineage_strips.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_lineage_strips.set_defaults(
        handler=_physical_ribbon_lineage_strips
    )
    physical_ribbon_lineage_strip_replay = subparsers.add_parser(
        "replay-physical-ribbon-lineage-strips",
        description=(
            "Jointly replay physically eligible whole-lineage strip "
            "assignments and recompute every cumulative native-CT face path."
        ),
    )
    physical_ribbon_lineage_strip_replay.add_argument(
        "--lineage-strips", type=Path, required=True
    )
    physical_ribbon_lineage_strip_replay.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_lineage_strip_replay.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_lineage_strip_replay.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_lineage_strip_replay.set_defaults(
        handler=_physical_ribbon_lineage_strip_replay
    )
    physical_ribbon_cumulative_corridor_replay = subparsers.add_parser(
        "replay-physical-ribbon-cumulative-corridors",
        description=(
            "Commit exact complete-strip candidates from a refreshed corridor "
            "catalog while rebuilding every inherited native-CT connection "
            "in the final shared sheet charts."
        ),
    )
    physical_ribbon_cumulative_corridor_replay.add_argument(
        "--prior-replay", type=Path, required=True
    )
    physical_ribbon_cumulative_corridor_replay.add_argument(
        "--candidate-replay", type=Path, required=True
    )
    physical_ribbon_cumulative_corridor_replay.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_cumulative_corridor_replay.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_cumulative_corridor_replay.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_cumulative_corridor_replay.set_defaults(
        handler=_physical_ribbon_cumulative_corridor_replay
    )
    physical_ribbon_cumulative_hole_replay = subparsers.add_parser(
        "replay-physical-ribbon-cumulative-holes",
        description=(
            "Commit dense CT-supported whole-hole assignments while "
            "rebuilding every inherited corridor connection and preserving "
            "sheet lineage."
        ),
    )
    physical_ribbon_cumulative_hole_replay.add_argument(
        "--prior-replay", type=Path, required=True
    )
    physical_ribbon_cumulative_hole_replay.add_argument(
        "--holes", type=Path, required=True
    )
    physical_ribbon_cumulative_hole_replay.add_argument(
        "--output", type=Path, required=True
    )
    physical_ribbon_cumulative_hole_replay.add_argument(
        "--settings-json", type=Path
    )
    physical_ribbon_cumulative_hole_replay.add_argument(
        "--force", action="store_true"
    )
    physical_ribbon_cumulative_hole_replay.set_defaults(
        handler=_physical_ribbon_cumulative_hole_replay
    )
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
    flatten_components.add_argument(
        "--source-root",
        type=Path,
        help=(
            "raw-Acus pipeline or variant chain used only to resolve native "
            "CT when the selected-patch root is a cropped or restitched graph"
        ),
    )
    flatten_components.add_argument(
        "--surface-graph",
        type=Path,
        help=(
            "complete retained surface-graph root; materialized inputs are "
            "detected automatically"
        ),
    )
    flatten_components.add_argument("--join-refinement", type=Path)
    flatten_components.add_argument("--stratigraphic-refinement", type=Path)
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
    refine_stratigraphy = subparsers.add_parser(
        "refine-stratigraphic-continuity",
        description=(
            "Compare full-mode depth/fiber fingerprints across each join and "
            "split only disagreements repeated over multi-cell neighborhoods."
        ),
    )
    refine_stratigraphy.add_argument("--root", type=Path, required=True)
    stratigraphic_mode_source = refine_stratigraphy.add_mutually_exclusive_group(
        required=True
    )
    stratigraphic_mode_source.add_argument(
        "--mode-bank",
        type=Path,
        help="complete single-pipeline Acus mode bank",
    )
    stratigraphic_mode_source.add_argument(
        "--sheet-evidence",
        type=Path,
        help=(
            "composed block sheet-evidence bank with stable mode IDs; supports "
            "owned crops and multi-block graphs"
        ),
    )
    refine_stratigraphy.add_argument("--output", type=Path, required=True)
    refine_stratigraphy.add_argument(
        "--candidate-restitch",
        type=Path,
        help=(
            "complete sheet-restitch candidate universe to score under the "
            "retained graph calibration"
        ),
    )
    refine_stratigraphy.add_argument(
        "--join-refinement",
        type=Path,
        help="optional native-CT join refinement to apply before this stage",
    )
    refine_stratigraphy.add_argument(
        "--neighborhood-radius", type=int, default=3
    )
    refine_stratigraphy.add_argument(
        "--minimum-side-patches", type=int, default=3
    )
    refine_stratigraphy.add_argument(
        "--minimum-context-modes", type=int, default=2
    )
    refine_stratigraphy.add_argument(
        "--minimum-coverage-fraction", type=float, default=0.5
    )
    refine_stratigraphy.add_argument(
        "--minimum-common-depth-span", type=float, default=8.0
    )
    refine_stratigraphy.add_argument(
        "--outlier-standard-deviations", type=float, default=4.0
    )
    refine_stratigraphy.add_argument(
        "--minimum-log-scale", type=float, default=0.12
    )
    refine_stratigraphy.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    refine_stratigraphy.add_argument("--force", action="store_true")
    refine_stratigraphy.set_defaults(handler=_refine_stratigraphic_continuity)
    contextual_growth = subparsers.add_parser(
        "contextual-grow",
        description=(
            "Add independently fitted full-bank modes only when two calibrated "
            "open-face graph contexts and complete physical/topological validation agree."
        ),
    )
    contextual_growth.add_argument(
        "--stratigraphic-refinement", type=Path, required=True
    )
    contextual_growth.add_argument("--output", type=Path, required=True)
    contextual_growth.add_argument(
        "--minimum-support-faces", type=int, default=2
    )
    contextual_growth.add_argument(
        "--maximum-admission-robust-z",
        type=float,
        default=1.0,
        help="require both local and graph-context scores within this retained-join robust Z",
    )
    contextual_growth.add_argument(
        "--maximum-modes-per-trace",
        type=int,
        default=0,
        help="operational cap after geometric ranking; zero retains every match",
    )
    contextual_growth.add_argument(
        "--maximum-trials",
        type=int,
        default=0,
        help="operational cap after contextual ranking; zero validates every candidate",
    )
    contextual_growth.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    contextual_growth.add_argument("--force", action="store_true")
    contextual_growth.set_defaults(handler=_contextual_growth)
    saturation = subparsers.add_parser(
        "audit-sheet-saturation",
        description=(
            "Measure calibrated Acus structural-evidence coverage, assignment "
            "ambiguity, and unexplained fiber evidence against selected layers."
        ),
    )
    saturation.add_argument("--root", type=Path, required=True)
    saturation.add_argument("--output", type=Path, required=True)
    saturation.add_argument(
        "--distance-radii",
        nargs="+",
        type=float,
        default=(1.0, 2.0, 2.5, 3.0, 4.0, 6.0, 8.0, 12.0, 16.0),
    )
    saturation.add_argument(
        "--joint-residuals",
        nargs="+",
        type=float,
        default=(1.0, 1.5, 2.0, 2.5, 3.0, 4.0),
    )
    saturation.add_argument(
        "--assignment-shares",
        nargs="+",
        type=float,
        default=(0.5, 0.67, 0.8, 0.9),
    )
    saturation.add_argument("--primary-joint-residual", type=float, default=2.5)
    saturation.add_argument("--primary-confident-share", type=float, default=0.8)
    saturation.add_argument("--overview-scale", type=int, default=24)
    saturation.add_argument("--force", action="store_true")
    saturation.set_defaults(handler=_sheet_saturation)
    saturation_reselection = subparsers.add_parser(
        "saturation-reselect",
        description=(
            "Enumerate physical full-bank cell stratigraphies, score them against "
            "owned Acus structural evidence, and globally reselect with face agreement."
        ),
    )
    saturation_reselection.add_argument("--root", type=Path, required=True)
    saturation_reselection.add_argument("--mode-bank", type=Path, required=True)
    saturation_reselection.add_argument("--output", type=Path, required=True)
    saturation_reselection.add_argument(
        "--joint-residual", type=float, default=2.5
    )
    saturation_reselection.add_argument(
        "--maximum-configurations", type=int, default=10
    )
    saturation_reselection.add_argument(
        "--maximum-per-coverage", type=int, default=2
    )
    saturation_reselection.add_argument("--pairwise-scale", type=float, default=0.35)
    saturation_reselection.add_argument(
        "--interior-unmatched-trace-penalty", type=float, default=0.0
    )
    saturation_reselection.add_argument("--maximum-sweeps", type=int, default=12)
    saturation_reselection.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    saturation_reselection.add_argument("--force", action="store_true")
    saturation_reselection.set_defaults(handler=_saturation_reselection)
    saturation_selection = subparsers.add_parser(
        "select-saturation-candidates",
        description=(
            "Reuse one immutable full-bank physical candidate artifact for fast "
            "global-selection and utilization-reward sweeps."
        ),
    )
    saturation_selection.add_argument("--candidates", type=Path, required=True)
    saturation_selection.add_argument("--output", type=Path, required=True)
    saturation_selection.add_argument(
        "--coverage-reward-scale", type=float, default=0.0
    )
    saturation_selection.add_argument("--unary-scale", type=float, default=1.0)
    saturation_selection.add_argument("--pairwise-scale", type=float, default=0.2)
    saturation_selection.add_argument(
        "--interior-unmatched-trace-penalty", type=float, default=0.0
    )
    saturation_selection.add_argument("--maximum-sweeps", type=int, default=12)
    saturation_selection.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    saturation_selection.add_argument("--no-visuals", action="store_true")
    saturation_selection.add_argument("--force", action="store_true")
    saturation_selection.set_defaults(handler=_saturation_candidate_selection)
    packet_connectivity = subparsers.add_parser(
        "dual-axis-packets",
        description=(
            "Build a separate sheet-level connectivity graph in which transported "
            "parallel and orthogonal fiber axes represent the same papyrus packet."
        ),
    )
    packet_connectivity.add_argument("--root", type=Path, required=True)
    packet_connectivity.add_argument("--output", type=Path, required=True)
    packet_connectivity.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    packet_connectivity.add_argument(
        "--maximum-preview-components", type=int, default=128
    )
    packet_connectivity.add_argument(
        "--maximum-normal-angle", type=float, default=15.0
    )
    packet_connectivity.add_argument(
        "--maximum-fiber-residual", type=float, default=15.0
    )
    packet_connectivity.add_argument("--force", action="store_true")
    packet_connectivity.set_defaults(handler=_dual_axis_packets)
    boundary_band = subparsers.add_parser(
        "export-boundary-band",
        description=(
            "Serialize bounded outer-cell geometry, graph ownership, exterior "
            "traces, and physical alternatives for adjacent-block composition."
        ),
    )
    boundary_band.add_argument("--root", type=Path, required=True)
    boundary_band.add_argument("--output", type=Path, required=True)
    boundary_band.add_argument("--packet-root", type=Path)
    boundary_band.add_argument("--candidate-root", type=Path)
    boundary_band.add_argument("--depth-cells", type=int, default=2)
    boundary_band.add_argument(
        "--leaf-shape", nargs=3, type=int, default=(4, 4, 3)
    )
    boundary_band.add_argument("--force", action="store_true")
    boundary_band.set_defaults(handler=_boundary_band)
    boundary_merge = subparsers.add_parser(
        "merge-boundary-bands",
        description=(
            "Compose two adjacent boundary artifacts into collision-safe, "
            "crossing-safe component bridges without reopening either interior."
        ),
    )
    boundary_merge.add_argument("--first", type=Path, required=True)
    boundary_merge.add_argument("--second", type=Path, required=True)
    boundary_merge.add_argument("--output", type=Path, required=True)
    boundary_merge.add_argument("--force", action="store_true")
    boundary_merge.set_defaults(handler=_boundary_merge)
    boundary_reselection = subparsers.add_parser(
        "reselect-boundary-bands",
        description=(
            "Jointly reselect physical configurations and strict/packet topology "
            "inside two meeting boundary bands while keeping both interiors frozen."
        ),
    )
    boundary_reselection.add_argument("--first", type=Path, required=True)
    boundary_reselection.add_argument("--second", type=Path, required=True)
    boundary_reselection.add_argument("--output", type=Path, required=True)
    boundary_reselection.add_argument("--unary-scale", type=float, default=1.0)
    boundary_reselection.add_argument("--pairwise-scale", type=float, default=0.2)
    boundary_reselection.add_argument(
        "--interior-unmatched-trace-penalty", type=float, default=0.0
    )
    boundary_reselection.add_argument("--maximum-sweeps", type=int, default=12)
    boundary_reselection.add_argument("--force", action="store_true")
    boundary_reselection.set_defaults(handler=_boundary_reselection)
    cluster_reselection = subparsers.add_parser(
        "reselect-boundary-cluster",
        description=(
            "Jointly reselect the union of all meeting face bands in one "
            "rectangular 2x2x2 child cluster while keeping child interiors frozen."
        ),
    )
    cluster_reselection.add_argument(
        "--boundary",
        action="append",
        type=Path,
        required=True,
        help="boundary-band root; repeat once for each child block",
    )
    cluster_reselection.add_argument("--output", type=Path, required=True)
    cluster_reselection.add_argument("--unary-scale", type=float, default=1.0)
    cluster_reselection.add_argument("--pairwise-scale", type=float, default=0.2)
    cluster_reselection.add_argument(
        "--interior-unmatched-trace-penalty", type=float, default=0.0
    )
    cluster_reselection.add_argument("--maximum-sweeps", type=int, default=12)
    cluster_reselection.add_argument("--force", action="store_true")
    cluster_reselection.set_defaults(handler=_boundary_cluster_reselection)
    cluster_materialization = subparsers.add_parser(
        "materialize-boundary-cluster",
        description=(
            "Recompose a joint cluster solution with its certified immutable "
            "child interiors into one complete retained surface graph."
        ),
    )
    cluster_materialization.add_argument("--cluster", type=Path, required=True)
    cluster_materialization.add_argument("--output", type=Path, required=True)
    cluster_materialization.add_argument(
        "--boundary",
        action="append",
        type=Path,
        help=(
            "optional relocated boundary root; repeat once per child and match "
            "by immutable boundary identity"
        ),
    )
    cluster_materialization.add_argument("--force", action="store_true")
    cluster_materialization.set_defaults(handler=_materialize_boundary_cluster)
    cell_refinement = subparsers.add_parser(
        "diagnose-cell-refinement",
        description=(
            "Trace one materialized cell from Acus evidence through retained "
            "physical candidates and face topology, then run bounded single-cell "
            "and adjacent-pair refinement rounds without mutating the graph."
        ),
    )
    cell_refinement.add_argument("--cluster", type=Path, required=True)
    cell_refinement.add_argument("--materialized", type=Path, required=True)
    cell_refinement.add_argument("--output", type=Path, required=True)
    cell_refinement.add_argument("--cell", nargs=3, type=int, required=True)
    cell_refinement.add_argument("--component-id", type=int)
    cell_refinement.add_argument("--neighborhood-radius", type=int, default=1)
    cell_refinement.add_argument("--unary-scale", type=float, default=1.0)
    cell_refinement.add_argument("--pairwise-scale", type=float, default=0.2)
    cell_refinement.add_argument(
        "--pairwise-reward-normalization",
        choices=("none", "trace-mean"),
        default="trace-mean",
    )
    cell_refinement.add_argument(
        "--unmatched-trace-penalty",
        type=float,
        help=(
            "continuation cost per unmatched trace; defaults to pairwise scale "
            "times the matcher unmatched likelihood"
        ),
    )
    cell_refinement.add_argument(
        "--coverage-reward-scale", type=float, default=0.5
    )
    cell_refinement.add_argument(
        "--minimum-oracle-coverage-fraction", type=float, default=0.5
    )
    cell_refinement.add_argument(
        "--maximum-cell-utilization-drop", type=float, default=0.05
    )
    cell_refinement.add_argument(
        "--minimum-evidence-mass-for-coverage-floor", type=float, default=1.0
    )
    cell_refinement.add_argument("--maximum-sweeps", type=int, default=4)
    cell_refinement.add_argument("--maximum-pair-sweeps", type=int, default=2)
    cell_refinement.add_argument("--force", action="store_true")
    cell_refinement.set_defaults(handler=_diagnose_cell_refinement)
    cell_refinement_materialization = subparsers.add_parser(
        "materialize-cell-refinement",
        description=(
            "Apply the accepted topology-safe changes from a cell diagnostic "
            "and write a complete, reloadable retained-graph variant."
        ),
    )
    cell_refinement_materialization.add_argument(
        "--cluster", type=Path, required=True
    )
    cell_refinement_materialization.add_argument(
        "--materialized", type=Path, required=True
    )
    cell_refinement_materialization.add_argument(
        "--diagnostic", type=Path, required=True
    )
    cell_refinement_materialization.add_argument(
        "--output", type=Path, required=True
    )
    cell_refinement_materialization.add_argument("--force", action="store_true")
    cell_refinement_materialization.set_defaults(
        handler=_materialize_cell_refinement
    )
    cell_refinement_targets = subparsers.add_parser(
        "rank-cell-refinement-targets",
        description=(
            "Rank spatially separated refinement neighborhoods using independent "
            "recoverable-Acus-evidence and unresolved-topology signals."
        ),
    )
    cell_refinement_targets.add_argument("--cluster", type=Path, required=True)
    cell_refinement_targets.add_argument(
        "--materialized", type=Path, required=True
    )
    cell_refinement_targets.add_argument("--output", type=Path, required=True)
    cell_refinement_targets.add_argument(
        "--neighborhood-radius", type=int, default=1
    )
    cell_refinement_targets.add_argument("--maximum-targets", type=int, default=128)
    cell_refinement_targets.add_argument(
        "--minimum-recoverable-evidence-mass", type=float, default=1.0e-6
    )
    cell_refinement_targets.add_argument(
        "--minimum-incident-open-trace-endpoints", type=int, default=1
    )
    cell_refinement_targets.add_argument("--force", action="store_true")
    cell_refinement_targets.set_defaults(handler=_rank_cell_refinement_targets)
    sheet_resolution = subparsers.add_parser(
        "audit-sheet-resolution",
        description=(
            "Detect coherent ordered-layer bends that are geometrically visible "
            "but under-resolved by one planar Acus sheetlet per cell, and emit "
            "power-of-two local raw-Acus refinement targets."
        ),
    )
    sheet_resolution.add_argument("--graph", type=Path, required=True)
    sheet_resolution.add_argument("--output", type=Path, required=True)
    sheet_resolution.add_argument(
        "--normal-limit-degrees",
        type=float,
        default=0.0,
        help="absolute locally-linear bend limit; zero derives it from retained joins",
    )
    sheet_resolution.add_argument(
        "--robust-standard-deviations", type=float, default=3.0
    )
    sheet_resolution.add_argument(
        "--minimum-normal-limit-degrees", type=float, default=15.0
    )
    sheet_resolution.add_argument(
        "--minimum-coherent-layers", type=int, default=2
    )
    sheet_resolution.add_argument(
        "--maximum-refinement-factor", type=int, default=4
    )
    sheet_resolution.add_argument(
        "--voxel-size-microns",
        type=float,
        help="source voxel pitch used only for physical-size reporting",
    )
    sheet_resolution.add_argument("--force", action="store_true")
    sheet_resolution.set_defaults(handler=_audit_sheet_resolution)
    sheet_restitch = subparsers.add_parser(
        "restitch-block-sheets",
        description=(
            "Enumerate every pair-gated selected-patch correspondence and "
            "rebuild the complete sheet graph from multiple whole-block "
            "topology-constrained proposals."
        ),
    )
    sheet_restitch.add_argument("--cluster", type=Path, required=True)
    sheet_restitch.add_argument("--materialized", type=Path, required=True)
    sheet_restitch.add_argument("--output", type=Path, required=True)
    sheet_restitch.add_argument(
        "--stratigraphic-candidates",
        type=Path,
        action="append",
        help=(
            "complete candidate stratigraphic refinement; jointly inconsistent "
            "local and multicell correspondences are removed before optimization; "
            "repeat only when an intentionally conservative intersection of "
            "independent gates is required"
        ),
    )
    sheet_restitch.add_argument(
        "--minimum-join-benefit", type=float, default=0.0
    )
    sheet_restitch.add_argument(
        "--quarter-turn-penalty", type=float, default=0.75
    )
    sheet_restitch.add_argument(
        "--unmatched-trace-penalty",
        type=float,
        default=0.0,
        help="extra cost per open endpoint in raw correspondence-benefit units",
    )
    sheet_restitch.add_argument("--restarts", type=int, default=12)
    sheet_restitch.add_argument(
        "--priority-jitter-fraction", type=float, default=0.35
    )
    sheet_restitch.add_argument("--exchange-rounds", type=int, default=2)
    sheet_restitch.add_argument(
        "--exchange-trials-per-round", type=int, default=24
    )
    sheet_restitch.add_argument(
        "--collision-cut-limit",
        type=int,
        default=0,
        help="maximum dense collision cuts; zero runs to physical completion",
    )
    sheet_restitch.add_argument("--no-collision-cut", action="store_true")
    sheet_restitch.add_argument(
        "--collision-cut-order",
        choices=("forward", "reverse", "both"),
        default="forward",
    )
    sheet_restitch.add_argument(
        "--curvature-refinement",
        action="store_true",
        help=(
            "refine abrupt axial-normal hinges with robust multiscale cuts and "
            "exact open-trace rematching"
        ),
    )
    sheet_restitch.add_argument(
        "--curvature-neighborhood-radius", type=int, default=3
    )
    sheet_restitch.add_argument(
        "--curvature-minimum-branch-support", type=int, default=3
    )
    sheet_restitch.add_argument(
        "--curvature-robust-standard-deviations", type=float, default=3.0
    )
    sheet_restitch.add_argument(
        "--curvature-minimum-calibration-joins", type=int, default=32
    )
    sheet_restitch.add_argument("--curvature-rounds", type=int, default=3)
    sheet_restitch.add_argument(
        "--curvature-cut-penalty-weight", type=float, default=1.0
    )
    sheet_restitch.add_argument(
        "--strict-normal-angle-cap-degrees",
        type=float,
        default=0.0,
        help=(
            "absolute axial normal cap for strict continuations; zero derives "
            "a robust cap from retained joins"
        ),
    )
    sheet_restitch.add_argument(
        "--strict-fiber-angle-cap-degrees",
        type=float,
        default=0.0,
        help=(
            "absolute transported axial fiber cap for strict continuations; "
            "zero derives a robust cap from retained strict joins"
        ),
    )
    sheet_restitch.add_argument(
        "--angle-calibration-robust-standard-deviations",
        type=float,
        default=3.0,
    )
    sheet_restitch.add_argument(
        "--angle-calibration-minimum-joins",
        type=int,
        default=32,
    )
    sheet_restitch.add_argument(
        "--layer-partition",
        action="store_true",
        help=(
            "partition the typed sheetlet graph with lifted tangent-overlap "
            "layer exclusions"
        ),
    )
    sheet_restitch.add_argument(
        "--stack-transport",
        action="store_true",
        help=(
            "experimentally require an independent path-consistent integer "
            "layer gauge per connected sheet; this is intentionally off "
            "because incomplete stacks require partial monotone transport"
        ),
    )
    sheet_restitch.add_argument(
        "--no-signed-partition",
        action="store_true",
        help="disable soft lifted correlation clustering",
    )
    sheet_restitch.add_argument(
        "--layer-repulsion-minimum-overlap-fraction",
        type=float,
        default=0.05,
    )
    sheet_restitch.add_argument(
        "--layer-repulsion-minimum-normal-separation-cells",
        type=float,
        default=0.15,
    )
    sheet_restitch.add_argument(
        "--layer-repulsion-scale",
        type=float,
        default=0.0,
        help=(
            "soft distinct-layer cost scale; zero uses one unmatched-trace "
            "negative log likelihood"
        ),
    )
    sheet_restitch.add_argument(
        "--layer-exclusion-proximity-radius", type=int, default=2
    )
    sheet_restitch.add_argument(
        "--layer-exclusion-minimum-overlap-fraction",
        type=float,
        default=0.25,
    )
    sheet_restitch.add_argument(
        "--layer-exclusion-minimum-normal-separation-cells",
        type=float,
        default=0.35,
    )
    sheet_restitch.add_argument(
        "--layer-exclusion-maximum-normal-angle-degrees",
        type=float,
        default=0.0,
        help=(
            "parallel-normal limit; zero derives it robustly from retained "
            "continuations"
        ),
    )
    sheet_restitch.add_argument("--force", action="store_true")
    sheet_restitch.set_defaults(handler=_restitch_block_sheets)
    sheet_evidence = subparsers.add_parser(
        "compile-sheet-evidence",
        description=(
            "Compile complete source-referenced Acus mode banks and physical "
            "within-cell stack alternatives into one immutable block-level "
            "sheet inference artifact. Each --input is ROOT OFFSET_X OFFSET_Y "
            "OFFSET_Z in the output block grid."
        ),
    )
    sheet_evidence.add_argument(
        "--input",
        nargs=4,
        action="append",
        required=True,
        metavar=("ROOT", "OFFSET_X", "OFFSET_Y", "OFFSET_Z"),
    )
    sheet_evidence.add_argument("--output", type=Path, required=True)
    sheet_evidence.add_argument(
        "--clipping-tolerance-scale", type=float, default=1.0e-8
    )
    sheet_evidence.add_argument("--force", action="store_true")
    sheet_evidence.set_defaults(handler=_compile_sheet_evidence)
    sheet_correspondences = subparsers.add_parser(
        "catalog-sheet-correspondences",
        description=(
            "Enumerate every pair-gated shared-face edge between immutable "
            "Acus mode nodes without selecting configurations or alignments."
        ),
    )
    sheet_correspondences.add_argument("--evidence", type=Path, required=True)
    sheet_correspondences.add_argument("--cluster", type=Path, required=True)
    sheet_correspondences.add_argument("--output", type=Path, required=True)
    sheet_correspondences.add_argument("--force", action="store_true")
    sheet_correspondences.set_defaults(handler=_catalog_sheet_correspondences)
    sheet_factors = subparsers.add_parser(
        "compile-sheet-factors",
        description=(
            "Compile exact order-preserving face factors for every neighboring "
            "pair of physical Acus stack configurations."
        ),
    )
    sheet_factors.add_argument("--evidence", type=Path, required=True)
    sheet_factors.add_argument("--correspondences", type=Path, required=True)
    sheet_factors.add_argument("--cluster", type=Path, required=True)
    sheet_factors.add_argument("--output", type=Path, required=True)
    sheet_factors.add_argument(
        "--quarter-turn-penalty", type=float, default=0.75
    )
    sheet_factors.add_argument("--force", action="store_true")
    sheet_factors.set_defaults(handler=_compile_sheet_factors)
    sheet_configuration = subparsers.add_parser(
        "initialize-sheet-configurations",
        description=(
            "Optimize the complete unary-plus-face factor graph as a reversible "
            "configuration initialization. This is not a globally valid sheet "
            "graph until topology replay."
        ),
    )
    sheet_configuration.add_argument("--evidence", type=Path, required=True)
    sheet_configuration.add_argument("--factors", type=Path, required=True)
    sheet_configuration.add_argument("--initial", type=Path)
    sheet_configuration.add_argument("--output", type=Path, required=True)
    sheet_configuration.add_argument("--unary-scale", type=float, default=1.0)
    sheet_configuration.add_argument("--pairwise-scale", type=float, default=0.2)
    sheet_configuration.add_argument(
        "--coverage-reward-scale", type=float, default=0.0
    )
    sheet_configuration.add_argument(
        "--unmatched-trace-penalty",
        type=float,
        default=0.0,
        help="extra local-factor cost per unmatched face-trace endpoint",
    )
    sheet_configuration.add_argument(
        "--pairwise-normalization",
        choices=("none", "trace-mean"),
        default="none",
    )
    sheet_configuration.add_argument("--maximum-sweeps", type=int, default=12)
    sheet_configuration.add_argument(
        "--belief-propagation-iterations", type=int, default=0
    )
    sheet_configuration.add_argument(
        "--belief-propagation-damping", type=float, default=0.5
    )
    sheet_configuration.add_argument(
        "--belief-propagation-tolerance", type=float, default=1.0e-4
    )
    sheet_configuration.add_argument("--force", action="store_true")
    sheet_configuration.set_defaults(handler=_initialize_sheet_configurations)
    joint_sheet_graph = subparsers.add_parser(
        "replay-joint-sheet-graph",
        description=(
            "Activate a complete physical stack selection, rebuild all available "
            "mode edges, and enforce global sheet topology before materializing "
            "a standard retained graph."
        ),
    )
    joint_sheet_graph.add_argument("--evidence", type=Path, required=True)
    joint_sheet_graph.add_argument("--correspondences", type=Path, required=True)
    joint_sheet_graph.add_argument("--configurations", type=Path, required=True)
    joint_sheet_graph.add_argument("--cluster", type=Path, required=True)
    joint_sheet_graph.add_argument("--output", type=Path, required=True)
    joint_sheet_graph.add_argument(
        "--minimum-join-benefit", type=float, default=0.0
    )
    joint_sheet_graph.add_argument(
        "--quarter-turn-penalty", type=float, default=0.75
    )
    joint_sheet_graph.add_argument(
        "--unmatched-trace-penalty",
        type=float,
        default=0.0,
        help="extra cost per open endpoint in raw correspondence-benefit units",
    )
    joint_sheet_graph.add_argument("--restarts", type=int, default=4)
    joint_sheet_graph.add_argument(
        "--priority-jitter-fraction", type=float, default=0.35
    )
    joint_sheet_graph.add_argument("--exchange-rounds", type=int, default=2)
    joint_sheet_graph.add_argument(
        "--exchange-trials-per-round", type=int, default=24
    )
    joint_sheet_graph.add_argument(
        "--collision-cut-limit",
        type=int,
        default=0,
        help="maximum dense collision cuts; zero runs to physical completion",
    )
    joint_sheet_graph.add_argument("--no-collision-cut", action="store_true")
    joint_sheet_graph.add_argument(
        "--collision-cut-order",
        choices=("forward", "reverse", "both"),
        default="forward",
    )
    joint_sheet_graph.add_argument("--force", action="store_true")
    joint_sheet_graph.set_defaults(handler=_replay_joint_sheet_graph)
    sheet_evidence_subblock = subparsers.add_parser(
        "extract-sheet-evidence-subblock",
        description=(
            "Extract and rebase an exact rectangular subset of an immutable "
            "block Acus evidence contract while preserving stable mode and "
            "physical-configuration IDs."
        ),
    )
    sheet_evidence_subblock.add_argument(
        "--evidence", type=Path, required=True
    )
    sheet_evidence_subblock.add_argument(
        "--start", nargs=3, type=int, required=True
    )
    sheet_evidence_subblock.add_argument(
        "--stop", nargs=3, type=int, required=True
    )
    sheet_evidence_subblock.add_argument("--output", type=Path, required=True)
    sheet_evidence_subblock.add_argument("--force", action="store_true")
    sheet_evidence_subblock.set_defaults(
        handler=_extract_sheet_evidence_subblock
    )
    owned_sheet_graph = subparsers.add_parser(
        "crop-owned-sheet-graph",
        description=(
            "Crop a completed expanded sheet solve to its owned core, prune "
            "outside components, and recompute clipped connectivity and boundary traces."
        ),
    )
    owned_sheet_graph.add_argument("--graph", type=Path, required=True)
    owned_sheet_graph.add_argument("--start", nargs=3, type=int, required=True)
    owned_sheet_graph.add_argument("--stop", nargs=3, type=int, required=True)
    owned_sheet_graph.add_argument("--output", type=Path, required=True)
    owned_sheet_graph.add_argument("--force", action="store_true")
    owned_sheet_graph.set_defaults(handler=_crop_owned_sheet_graph)
    sheet_halo = subparsers.add_parser(
        "audit-sheet-halos",
        description=(
            "Solve one fixed owned core independently with several cell halos, "
            "crop each result, re-stitch topology inside the owned core, and "
            "compare configuration and graph stability."
        ),
    )
    sheet_halo.add_argument("--evidence", type=Path, required=True)
    sheet_halo.add_argument("--cluster", type=Path, required=True)
    sheet_halo.add_argument(
        "--core-start", nargs=3, type=int, required=True
    )
    sheet_halo.add_argument(
        "--core-stop", nargs=3, type=int, required=True
    )
    sheet_halo.add_argument(
        "--halos", nargs="+", type=int, default=(0, 1, 2)
    )
    sheet_halo.add_argument("--output", type=Path, required=True)
    sheet_halo.add_argument("--unary-scale", type=float, default=1.0)
    sheet_halo.add_argument("--pairwise-scale", type=float, default=0.2)
    sheet_halo.add_argument("--coverage-reward-scale", type=float, default=0.0)
    sheet_halo.add_argument(
        "--unmatched-trace-penalty",
        type=float,
        default=0.0,
        help="extra local-factor cost per unmatched face-trace endpoint",
    )
    sheet_halo.add_argument(
        "--pairwise-normalization",
        choices=("none", "trace-mean"),
        default="none",
    )
    sheet_halo.add_argument("--maximum-sweeps", type=int, default=12)
    sheet_halo.add_argument(
        "--belief-propagation-iterations", type=int, default=0
    )
    sheet_halo.add_argument(
        "--belief-propagation-damping", type=float, default=0.5
    )
    sheet_halo.add_argument(
        "--belief-propagation-tolerance", type=float, default=1.0e-4
    )
    sheet_halo.add_argument("--minimum-join-benefit", type=float, default=0.0)
    sheet_halo.add_argument("--quarter-turn-penalty", type=float, default=0.75)
    sheet_halo.add_argument(
        "--stitching-unmatched-trace-penalty",
        type=float,
        default=0.0,
        help="extra owned-graph cost per open endpoint in raw benefit units",
    )
    sheet_halo.add_argument("--restarts", type=int, default=4)
    sheet_halo.add_argument(
        "--priority-jitter-fraction", type=float, default=0.35
    )
    sheet_halo.add_argument("--exchange-rounds", type=int, default=2)
    sheet_halo.add_argument(
        "--exchange-trials-per-round", type=int, default=24
    )
    sheet_halo.add_argument(
        "--collision-cut-limit",
        type=int,
        default=0,
        help="maximum dense collision cuts; zero runs to physical completion",
    )
    sheet_halo.add_argument("--no-collision-cut", action="store_true")
    sheet_halo.add_argument(
        "--collision-cut-order",
        choices=("forward", "reverse", "both"),
        default="forward",
    )
    sheet_halo.add_argument("--force", action="store_true")
    sheet_halo.set_defaults(handler=_sheet_halo_experiment)
    finalize_sheet_halo = subparsers.add_parser(
        "finalize-sheet-halo-audit",
        description=(
            "Re-stitch every cropped owned core with its halo-selected cell "
            "configurations, then compare final topology and score each core "
            "selection in the largest available halo context."
        ),
    )
    finalize_sheet_halo.add_argument(
        "--experiment", type=Path, required=True
    )
    finalize_sheet_halo.add_argument(
        "--cluster",
        type=Path,
        help="optional relocated cluster root; defaults to the experiment identity",
    )
    finalize_sheet_halo.add_argument("--force", action="store_true")
    finalize_sheet_halo.set_defaults(handler=_finalize_sheet_halo_experiment)
    sheet_core_audit = subparsers.add_parser(
        "audit-sheet-core",
        description=(
            "Classify every owned-core evidence deficit and unresolved interior "
            "trace against the complete immutable Acus mode, physical-stack, "
            "and correspondence banks."
        ),
    )
    sheet_core_audit.add_argument("--evidence", type=Path, required=True)
    sheet_core_audit.add_argument(
        "--correspondences", type=Path, required=True
    )
    sheet_core_audit.add_argument("--factors", type=Path, required=True)
    sheet_core_audit.add_argument(
        "--configurations", type=Path, required=True
    )
    sheet_core_audit.add_argument("--graph", type=Path, required=True)
    sheet_core_audit.add_argument(
        "--core-start", nargs=3, type=int, required=True
    )
    sheet_core_audit.add_argument("--output", type=Path, required=True)
    sheet_core_audit.add_argument("--maximum-hotspots", type=int, default=128)
    sheet_core_audit.add_argument("--force", action="store_true")
    sheet_core_audit.set_defaults(handler=_audit_sheet_core)
    sheet_topology_refinement = subparsers.add_parser(
        "refine-sheet-topology",
        description=(
            "Reopen complete Acus stack configurations in pressure-ranked "
            "sheet neighborhoods and accept changes only against the exact "
            "globally retained topology-safe graph."
        ),
    )
    sheet_topology_refinement.add_argument(
        "--evidence", type=Path, required=True
    )
    sheet_topology_refinement.add_argument(
        "--correspondences", type=Path, required=True
    )
    sheet_topology_refinement.add_argument(
        "--factors", type=Path, required=True
    )
    sheet_topology_refinement.add_argument(
        "--configurations", type=Path, required=True
    )
    sheet_topology_refinement.add_argument("--graph", type=Path, required=True)
    sheet_topology_refinement.add_argument(
        "--cluster", type=Path, required=True
    )
    sheet_topology_refinement.add_argument(
        "--output", type=Path, required=True
    )
    sheet_topology_refinement.add_argument(
        "--maximum-rounds", type=int, default=2
    )
    sheet_topology_refinement.add_argument(
        "--maximum-trials-per-round", type=int, default=36
    )
    sheet_topology_refinement.add_argument(
        "--maximum-seed-moves", type=int, default=48
    )
    sheet_topology_refinement.add_argument(
        "--alternatives-per-pressure-cell", type=int, default=3
    )
    sheet_topology_refinement.add_argument(
        "--relaxation-radius", type=int, default=1
    )
    sheet_topology_refinement.add_argument(
        "--relaxation-sweeps", type=int, default=3
    )
    sheet_topology_refinement.add_argument(
        "--minimum-objective-gain", type=float, default=1.0e-6
    )
    sheet_topology_refinement.add_argument(
        "--minimum-join-benefit", type=float, default=0.0
    )
    sheet_topology_refinement.add_argument(
        "--quarter-turn-penalty", type=float, default=0.75
    )
    sheet_topology_refinement.add_argument("--force", action="store_true")
    sheet_topology_refinement.set_defaults(handler=_refine_sheet_topology)
    selected_subblock = subparsers.add_parser(
        "extract-selected-subblock",
        description=(
            "Rebase a selected-patch subset for deterministic block-composition "
            "audits; this does not replace independent raw-CT inference."
        ),
    )
    selected_subblock.add_argument("--root", type=Path, required=True)
    selected_subblock.add_argument("--output", type=Path, required=True)
    selected_subblock.add_argument("--start", nargs=3, type=int, required=True)
    selected_subblock.add_argument("--stop", nargs=3, type=int, required=True)
    selected_subblock.add_argument("--force", action="store_true")
    selected_subblock.set_defaults(handler=_selected_subblock)
    split_audit = subparsers.add_parser(
        "audit-boundary-split",
        description=(
            "Compare a deterministic split/recompose seam with the retained "
            "joins in its unsplit full-block packet graph."
        ),
    )
    split_audit.add_argument("--full-packet-root", type=Path, required=True)
    split_audit.add_argument("--merge-root", type=Path, required=True)
    split_audit.add_argument("--output", type=Path, required=True)
    split_audit.add_argument("--force", action="store_true")
    split_audit.set_defaults(handler=_boundary_split_audit)
    reselection_audit = subparsers.add_parser(
        "audit-boundary-reselection",
        description=(
            "Compare deterministic narrow-band reselection with the complete "
            "unsplit packet graph, including exact retained-join identity."
        ),
    )
    reselection_audit.add_argument(
        "--full-packet-root", type=Path, required=True
    )
    reselection_audit.add_argument(
        "--reselection-root", type=Path, required=True
    )
    reselection_audit.add_argument("--output", type=Path, required=True)
    reselection_audit.add_argument("--force", action="store_true")
    reselection_audit.set_defaults(handler=_boundary_reselection_audit)
    independent_audit = subparsers.add_parser(
        "audit-independent-boundary",
        description=(
            "Compare selected-only and joint narrow-band composition of two "
            "independently inferred CT blocks with one full-context consistency "
            "reference."
        ),
    )
    independent_audit.add_argument(
        "--full-packet-root", type=Path, required=True
    )
    independent_audit.add_argument(
        "--selected-merge-root", type=Path, required=True
    )
    independent_audit.add_argument(
        "--reselection-root", type=Path, required=True
    )
    independent_audit.add_argument("--output", type=Path, required=True)
    independent_audit.add_argument(
        "--height-tolerance", type=float, default=1.0e-3
    )
    independent_audit.add_argument(
        "--normal-tolerance", type=float, default=0.01
    )
    independent_audit.add_argument(
        "--fiber-tolerance", type=float, default=0.01
    )
    independent_audit.add_argument("--force", action="store_true")
    independent_audit.set_defaults(handler=_independent_boundary_audit)
    multiseam_audit = subparsers.add_parser(
        "audit-multiseam",
        description=(
            "Audit configuration and topology agreement where a network of "
            "pairwise boundary-band solutions overlap at block corners."
        ),
    )
    multiseam_audit.add_argument(
        "--reselection",
        action="append",
        type=Path,
        required=True,
        help="pairwise boundary-reselection root; repeat for every seam",
    )
    multiseam_audit.add_argument("--output", type=Path, required=True)
    multiseam_audit.add_argument(
        "--cluster-root",
        type=Path,
        help="optional joint cluster solution to compare with pairwise overlaps",
    )
    multiseam_audit.add_argument("--force", action="store_true")
    multiseam_audit.set_defaults(handler=_multiseam_audit)
    cluster_reference_audit = subparsers.add_parser(
        "audit-boundary-cluster-reference",
        description=(
            "Compare one independent child-cluster solve with an unsplit "
            "full-context packet reconstruction over the same cuboid."
        ),
    )
    cluster_reference_audit.add_argument(
        "--full-packet-root", type=Path, required=True
    )
    cluster_reference_audit.add_argument("--full-selected-root", type=Path)
    cluster_reference_audit.add_argument("--cluster-root", type=Path, required=True)
    cluster_reference_audit.add_argument("--output", type=Path, required=True)
    cluster_reference_audit.add_argument(
        "--height-tolerance", type=float, default=1.0e-3
    )
    cluster_reference_audit.add_argument(
        "--normal-tolerance", type=float, default=0.01
    )
    cluster_reference_audit.add_argument(
        "--fiber-tolerance", type=float, default=0.01
    )
    cluster_reference_audit.add_argument("--force", action="store_true")
    cluster_reference_audit.set_defaults(handler=_cluster_reference_audit)
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
