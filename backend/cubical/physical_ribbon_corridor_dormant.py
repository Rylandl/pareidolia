from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import (
    PhysicalRibbonConfigurationSettings,
    _component_labels,
    _load_continuity_artifact,
    build_profile_crossing_conflicts,
)
from .physical_ribbon_corridor_extension import (
    _delta_variant_arrays,
    _load_variant_artifact,
)
from .physical_ribbon_corridor_variants import (
    PhysicalRibbonCorridorVariantSettings,
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
    compile_exact_variant_reconfiguration,
    enumerate_corridor_reconfiguration_variants,
    screen_exact_corridor_variants,
)
from .physical_ribbon_corridor_sets import (
    PHYSICAL_RIBBON_CORRIDOR_SETS_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_SETS_STEM,
)
from .physical_ribbon_patch_corridors import (
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    replay_patch_corridor_reconfigurations,
    solve_patch_corridor_reconfigurations,
    write_patch_corridor_montage,
    write_replayed_corridor_fragment_montage,
)


PHYSICAL_RIBBON_DORMANT_CORRIDORS_SCHEMA = (
    "pareidolia.physical-ribbon-dormant-corridors"
)
PHYSICAL_RIBBON_DORMANT_CORRIDORS_VERSION = 1
PHYSICAL_RIBBON_DORMANT_CORRIDORS_STEM = (
    "physical-ribbon-dormant-corridors-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonDormantCorridorSettings:
    maximum_variants_per_corridor: int = 8
    minimum_dormant_additions_per_variant: int = 1
    maximum_preview_components: int = 8

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("dormant corridor variant count must lie in [1, 16]")
        if self.minimum_dormant_additions_per_variant < 1:
            raise ValueError("dormant states must add at least one dormant ribbon")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_interfaces(
    ribbon_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    reference = ribbon_manifest["identity"]["interfaceBank"]
    manifest_path = Path(reference["manifestPath"])
    if sha256_file(manifest_path) != reference["manifestSha256"]:
        raise ValueError("ribbon interface bank has changed")
    manifest = json.loads(manifest_path.read_text())
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _load_corridor_sets(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_SETS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_SETS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "dormant corridors require complete label-free prior corridor sets"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _condition_crossings_on_immutable_baseline(
    crossings: Mapping[str, np.ndarray],
    selected: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """Keep the counterfactual screen from rejecting inherited graph debt.

    The expanded frontier can reveal profile intersections that the sparser
    frontier never rasterized.  The base selection is immutable in this stage,
    so those pairs are an audit finding rather than a reason to reject every
    possible change.  Exact trials must introduce no *new* crossing pair.
    """

    first = np.asarray(
        crossings["crossingFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        crossings["crossingSecondFrontierIndex"], dtype=np.int32
    )
    inherited = selected[first] & selected[second]
    conditioned = {
        name: np.asarray(value)[~inherited]
        for name, value in crossings.items()
    }
    return conditioned, {
        "detectedCrossingPairCount": len(first),
        "inheritedBaselineCrossingPairCount": int(
            np.count_nonzero(inherited)
        ),
        "enforcedCounterfactualCrossingPairCount": int(
            np.count_nonzero(~inherited)
        ),
    }


def _condition_configuration_on_expanded_frontier(
    ribbon: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    base_topology: Mapping[str, np.ndarray],
    expanded_topology: Mapping[str, np.ndarray],
    base_configuration: Mapping[str, np.ndarray],
    *,
    continuity_manifest: Mapping[str, Any],
    configuration_settings: PhysicalRibbonConfigurationSettings,
) -> tuple[dict[str, np.ndarray], np.ndarray, dict[str, Any]]:
    old_frontier = np.asarray(
        base_topology["frontierRibbonCandidate"], dtype=np.int32
    )
    expanded_frontier = np.asarray(
        expanded_topology["frontierRibbonCandidate"], dtype=np.int32
    )
    bank_count = len(np.asarray(ribbon["sourceInterface"]))
    bank_to_expanded = np.full(bank_count, -1, dtype=np.int32)
    bank_to_expanded[expanded_frontier] = np.arange(
        len(expanded_frontier), dtype=np.int32
    )
    old_to_expanded = bank_to_expanded[old_frontier]
    if np.any(old_to_expanded < 0):
        raise ValueError("expanded continuity does not contain the base frontier")
    selected = np.zeros(len(expanded_frontier), dtype=bool)
    selected[old_to_expanded] = (
        np.asarray(base_configuration["selected"], dtype=np.uint8) > 0
    )
    edge_first = np.asarray(
        expanded_topology["edgeFirstFrontierIndex"], dtype=np.int32
    )
    edge_second = np.asarray(
        expanded_topology["edgeSecondFrontierIndex"], dtype=np.int32
    )
    component, component_size = _component_labels(
        selected, edge_first, edge_second
    )
    geometry = continuity_manifest["geometry"]
    source_origin = np.asarray(
        continuity_manifest["source"]["sourceOriginXYZ"], dtype=np.float32
    )
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    processing_stop = np.asarray(
        geometry["processingVoxelBounds"]["stopXYZExclusive"], dtype=np.float32
    )
    processing_shape = np.asarray(
        geometry["processingShapeSamplingXYZ"], dtype=np.int32
    )
    stride_xyz = (processing_stop - processing_start) / processing_shape
    if not np.allclose(stride_xyz, stride_xyz[0]):
        raise ValueError("expanded corridor analysis requires isotropic sampling")
    stride = int(round(float(stride_xyz[0])))
    crossings, crossing_stats = build_profile_crossing_conflicts(
        ribbon,
        interfaces,
        expanded_topology,
        processing_world_start_xyz=source_origin + processing_start,
        processing_shape_sampling_xyz=tuple(int(value) for value in processing_shape),
        sampling_stride_voxels=stride,
        settings=configuration_settings,
    )
    crossings, crossing_conditioning_stats = (
        _condition_crossings_on_immutable_baseline(crossings, selected)
    )
    physical_score = np.asarray(
        ribbon["physicalEvidenceScore"], dtype=np.float32
    )[expanded_frontier]
    source_rank = np.asarray(ribbon["sourceRayRank"], dtype=np.int32)[
        expanded_frontier
    ]
    target_rank = np.asarray(ribbon["targetRayRank"], dtype=np.int32)[
        expanded_frontier
    ]
    mutual = np.asarray(ribbon["mutualFirstHit"])[expanded_frontier] > 0
    unary = (
        physical_score
        - configuration_settings.node_selection_cost
        - configuration_settings.ray_rank_penalty * (source_rank + target_rank)
        + configuration_settings.mutual_first_hit_bonus * mutual
    ).astype(np.float32)
    conditioned = {
        **crossings,
        "nodeUnaryScore": unary,
        "initialSelected": selected.astype(np.uint8),
        "selected": selected.astype(np.uint8),
        "component": component,
    }
    base_component_count = len(
        np.unique(
            np.asarray(base_configuration["component"])[
                np.asarray(base_configuration["component"]) >= 0
            ]
        )
    )
    if len(component_size) != base_component_count:
        raise ValueError("expanded frontier changed the selected base components")
    old_component = np.asarray(base_configuration["component"], dtype=np.int32)
    inherited_component = component[old_to_expanded]
    old_selected = np.asarray(base_configuration["selected"]) > 0
    inherited_to_old_component: dict[int, int] = {}
    for old_component_id in np.unique(old_component[old_selected]):
        if old_component_id < 0:
            continue
        inherited = np.unique(
            inherited_component[
                old_selected & (old_component == old_component_id)
            ]
        )
        inherited = inherited[inherited >= 0]
        if len(inherited) != 1:
            raise ValueError(
                "expanded frontier split a selected base component"
            )
        inherited_component_id = int(inherited[0])
        prior_old_component = inherited_to_old_component.setdefault(
            inherited_component_id, int(old_component_id)
        )
        if prior_old_component != int(old_component_id):
            raise ValueError(
                "expanded frontier fused selected base components"
            )
    return conditioned, old_to_expanded, {
        "baseFrontierCount": len(old_frontier),
        "expandedFrontierCount": len(expanded_frontier),
        "dormantFrontierCount": len(expanded_frontier) - len(old_frontier),
        "mappedSelectedRibbonCount": int(np.count_nonzero(selected)),
        "mappedComponentCount": len(component_size),
        "partitionPreservedExactly": True,
        "crossings": {
            **crossing_stats,
            **crossing_conditioning_stats,
            "acceptanceRule": (
                "the immutable baseline may retain newly revealed crossing "
                "debt; a counterfactual state may introduce no new pair"
            ),
        },
        "identityLabelsUsed": False,
    }


def _remap_corridor_surface(
    corridor: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    base_configuration: Mapping[str, np.ndarray],
    conditioned_configuration: Mapping[str, np.ndarray],
    old_to_expanded: np.ndarray,
) -> dict[str, np.ndarray]:
    result = {name: np.asarray(value) for name, value in corridor.items()}
    result.update({name: np.asarray(value) for name, value in surface.items()})
    first_boundary = old_to_expanded[
        np.asarray(corridor["boundaryEdgeFirstFrontierIndex"], dtype=np.int32)
    ]
    second_boundary = old_to_expanded[
        np.asarray(corridor["boundaryEdgeSecondFrontierIndex"], dtype=np.int32)
    ]
    result["boundaryEdgeFirstFrontierIndex"] = first_boundary
    result["boundaryEdgeSecondFrontierIndex"] = second_boundary
    old_component = np.asarray(base_configuration["component"], dtype=np.int32)
    expanded_component = np.asarray(
        conditioned_configuration["component"], dtype=np.int32
    )
    component_map: dict[int, int] = {}
    for component_id in np.unique(old_component[old_component >= 0]):
        nodes = np.flatnonzero(old_component == component_id)
        inherited = expanded_component[old_to_expanded[nodes]]
        inherited = inherited[inherited >= 0]
        if len(inherited):
            value = np.unique(inherited)
            if len(value) != 1:
                raise ValueError(
                    "cannot remap a corridor from a split base component"
                )
            component_map[int(component_id)] = int(value[0])
    result["corridorTopologyComponent"] = np.asarray(
        [
            component_map.get(int(value), -1)
            for value in corridor["corridorTopologyComponent"]
        ],
        dtype=np.int32,
    )
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    region = _triangle_region_labels(triangles)
    edge_region: dict[tuple[int, int], int] = {}
    for triangle_index, triangle in enumerate(triangles):
        for edge_index, left in enumerate(triangle):
            right = int(triangle[(edge_index + 1) % 3])
            edge_region[(min(int(left), right), max(int(left), right))] = int(
                region[triangle_index]
            )
    previous_region = np.asarray(
        corridor["boundaryEdgeTriangleRegion"], dtype=np.int32
    )
    result["boundaryEdgeTriangleRegion"] = np.asarray(
        [
            edge_region.get(
                (min(int(left), int(right)), max(int(left), int(right))),
                int(previous_region[index]),
            )
            for index, (left, right) in enumerate(
                zip(first_boundary, second_boundary)
            )
        ],
        dtype=np.int32,
    )
    return result


def _variant_dormant_addition_count(
    variants: Mapping[str, np.ndarray],
    expanded_frontier: np.ndarray,
    base_bank_mask: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(variants["corridorVariantAddedOffset"], dtype=np.int64)
    value = np.asarray(
        variants["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    result = np.zeros(len(offset) - 1, dtype=np.int16)
    for variant_index in range(len(result)):
        added = value[int(offset[variant_index]) : int(offset[variant_index + 1])]
        if len(added):
            result[variant_index] = int(
                np.count_nonzero(~base_bank_mask[expanded_frontier[added]])
            )
    return result


def _combine_compiled_reconfigurations(
    reconfiguration: Mapping[str, np.ndarray],
    prior: Mapping[str, np.ndarray],
    incremental: Mapping[str, np.ndarray],
    *,
    prior_frontier_to_expanded: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Combine disjoint exact decisions into one expanded-frontier replay."""

    prior_eligible = np.asarray(prior["corridorEvidenceEligible"]) > 0
    incremental_eligible = (
        np.asarray(incremental["corridorEvidenceEligible"]) > 0
    )
    if len(prior_eligible) != len(incremental_eligible):
        raise ValueError("prior and incremental corridor tables differ")
    if np.any(prior_eligible & incremental_eligible):
        raise ValueError("a corridor was selected by both exact sources")
    source = np.zeros(len(prior_eligible), dtype=np.uint8)
    source[prior_eligible] = 1
    source[incremental_eligible] = 2
    added_offset = [0]
    added_value: list[int] = []
    removed_offset = [0]
    removed_value: list[int] = []

    def append_ragged(
        table: Mapping[str, np.ndarray],
        row: int,
        prefix: str,
        mapping: np.ndarray | None,
        output: list[int],
    ) -> None:
        offset = np.asarray(
            table[f"corridorProposal{prefix}Offset"], dtype=np.int64
        )
        value = np.asarray(
            table[f"corridorProposal{prefix}FrontierIndex"], dtype=np.int32
        )[int(offset[row]) : int(offset[row + 1])]
        if mapping is not None and len(value):
            value = mapping[value]
            if np.any(value < 0):
                raise ValueError("prior exact state is absent from expanded frontier")
        output.extend(int(item) for item in value)

    for row, source_value in enumerate(source):
        if source_value == 1:
            append_ragged(
                prior,
                row,
                "Added",
                prior_frontier_to_expanded,
                added_value,
            )
            append_ragged(
                prior,
                row,
                "Removed",
                prior_frontier_to_expanded,
                removed_value,
            )
        elif source_value == 2:
            append_ragged(incremental, row, "Added", None, added_value)
            append_ragged(incremental, row, "Removed", None, removed_value)
        added_offset.append(len(added_value))
        removed_offset.append(len(removed_value))

    result = {name: np.asarray(value) for name, value in reconfiguration.items()}
    result["corridorEvidenceEligible"] = (source > 0).astype(np.uint8)
    result["corridorProposalAddedOffset"] = np.asarray(
        added_offset, dtype=np.int64
    )
    result["corridorProposalAddedFrontierIndex"] = np.asarray(
        added_value, dtype=np.int32
    )
    result["corridorProposalRemovedOffset"] = np.asarray(
        removed_offset, dtype=np.int64
    )
    result["corridorProposalRemovedFrontierIndex"] = np.asarray(
        removed_value, dtype=np.int32
    )
    scalar_fields = (
        "corridorProposalLocalObjective",
        "corridorProposalObjectiveDelta",
        "corridorProposalPatchCoverage",
        "corridorProposalRetainedBoundaryFraction",
        "corridorProposalBoundaryAnchorCount",
    )
    for name in scalar_fields:
        value = np.asarray(prior[name]).copy()
        value[incremental_eligible] = np.asarray(incremental[name])[
            incremental_eligible
        ]
        result[name] = value
    return result, {
        "combinedCorridorDecisionSource": source,
        "combinedPriorCorridorSelected": prior_eligible.astype(np.uint8),
        "combinedDormantCorridorSelected": incremental_eligible.astype(
            np.uint8
        ),
    }


def run_physical_ribbon_dormant_corridors(
    corridor_root: str | Path,
    prior_variant_root: str | Path,
    prior_set_root: str | Path,
    configuration_root: str | Path,
    expanded_continuity_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonDormantCorridorSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonDormantCorridorSettings()
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_root
    )
    prior_path, prior_manifest, prior_variants = _load_variant_artifact(
        prior_variant_root
    )
    prior_set_path, prior_set_manifest, prior_sets = _load_corridor_sets(
        prior_set_root
    )
    (
        configuration_path,
        configuration_manifest,
        base_configuration,
        base_topology_path,
        base_topology_manifest,
        base_topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_root)
    expanded_path, expanded_manifest, expanded_topology = (
        _load_continuity_artifact(expanded_continuity_root)
    )
    if (
        prior_manifest["identity"]["corridors"]["dataSha256"]
        != corridor_manifest["data"]["sha256"]
        or prior_manifest["identity"]["configuration"]["dataSha256"]
        != configuration_manifest["data"]["sha256"]
    ):
        raise ValueError("corridor, prior variant, and configuration inputs differ")
    if (
        prior_set_manifest["identity"]["variants"]["dataSha256"]
        != prior_manifest["data"]["sha256"]
        or prior_set_manifest["identity"]["configuration"]["dataSha256"]
        != configuration_manifest["data"]["sha256"]
    ):
        raise ValueError("prior corridor sets do not match variants/configuration")
    if (
        expanded_manifest["identity"]["ribbonBank"]["dataSha256"]
        != ribbon_manifest["data"]["sha256"]
    ):
        raise ValueError("expanded continuity uses a different ribbon bank")
    interface_path, interface_manifest, interfaces = _load_interfaces(
        ribbon_manifest
    )
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    configuration_settings = PhysicalRibbonConfigurationSettings(
        **configuration_manifest["identity"]["settings"]
    )
    variant_settings = PhysicalRibbonCorridorVariantSettings(
        maximum_variants_per_corridor=resolved.maximum_variants_per_corridor,
        minimum_variant_patch_coverage=0.45,
        minimum_anchor_count_per_arc=1,
        minimum_surface_area_retention=0.98,
        maximum_preview_components=resolved.maximum_preview_components,
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DORMANT_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_DORMANT_CORRIDORS_VERSION,
        "corridors": {
            "manifestPath": str(corridor_path),
            "manifestSha256": sha256_file(corridor_path),
            "dataSha256": corridor_manifest["data"]["sha256"],
        },
        "priorVariants": {
            "manifestPath": str(prior_path),
            "manifestSha256": sha256_file(prior_path),
            "dataSha256": prior_manifest["data"]["sha256"],
        },
        "priorCorridorSets": {
            "manifestPath": str(prior_set_path),
            "manifestSha256": sha256_file(prior_set_path),
            "dataSha256": prior_set_manifest["data"]["sha256"],
        },
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "baseTopology": {
            "manifestPath": str(base_topology_path),
            "manifestSha256": sha256_file(base_topology_path),
            "dataSha256": base_topology_manifest["data"]["sha256"],
        },
        "expandedContinuity": {
            "manifestPath": str(expanded_path),
            "manifestSha256": sha256_file(expanded_path),
            "dataSha256": expanded_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "interfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_DORMANT_CORRIDORS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_DORMANT_CORRIDORS_STEM}.npz"
    montage_path = output / "physical-ribbon-dormant-corridors.png"
    fragment_path = output / "physical-ribbon-dormant-fragments.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    if progress is not None:
        progress("conditioning the expanded frontier on the selected base surface")
    conditioned, old_to_expanded, conditioning_stats = (
        _condition_configuration_on_expanded_frontier(
            ribbon,
            interfaces,
            base_topology,
            expanded_topology,
            base_configuration,
            continuity_manifest=expanded_manifest,
            configuration_settings=configuration_settings,
        )
    )
    conditioned_at = time.monotonic()
    surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        expanded_topology,
        conditioned,
        settings=corridor_settings.surface_settings(),
    )
    remapped = _remap_corridor_surface(
        corridor,
        surface,
        base_configuration,
        conditioned,
        old_to_expanded,
    )
    surfaced_at = time.monotonic()
    if progress is not None:
        progress("solving residual corridors with dormant bidirectional ribbons")
    reconfiguration, reconfiguration_stats = solve_patch_corridor_reconfigurations(
        remapped,
        remapped,
        ribbon,
        expanded_topology,
        conditioned,
        continuity_weight=configuration_settings.continuity_weight,
        settings=corridor_settings,
    )
    reconfigured_at = time.monotonic()
    full_variants, enumeration_stats = enumerate_corridor_reconfiguration_variants(
        remapped,
        remapped,
        reconfiguration,
        ribbon,
        expanded_topology,
        conditioned,
        continuity_weight=configuration_settings.continuity_weight,
        corridor_settings=corridor_settings,
        settings=variant_settings,
    )
    base_frontier = np.asarray(
        base_topology["frontierRibbonCandidate"], dtype=np.int32
    )
    expanded_frontier = np.asarray(
        expanded_topology["frontierRibbonCandidate"], dtype=np.int32
    )
    base_bank_mask = np.zeros(len(np.asarray(ribbon["sourceInterface"])), dtype=bool)
    base_bank_mask[base_frontier] = True
    dormant_addition_count = _variant_dormant_addition_count(
        full_variants, expanded_frontier, base_bank_mask
    )
    prior_rows = set(
        int(value) for value in np.unique(prior_variants["corridorVariantRow"])
    )
    prior_resolved = set(
        int(value)
        for value in np.flatnonzero(
            np.asarray(prior_variants["corridorChosenExactVariant"]) >= 0
        )
    )
    residual_rows = prior_rows - prior_resolved
    full_row = np.asarray(full_variants["corridorVariantRow"], dtype=np.int32)
    target_variant = np.asarray(
        [
            index
            for index in range(len(full_row))
            if int(full_row[index]) in residual_rows
            and int(dormant_addition_count[index])
            >= resolved.minimum_dormant_additions_per_variant
        ],
        dtype=np.int32,
    )
    target_variants, target_to_full = _delta_variant_arrays(
        full_variants,
        target_variant,
        corridor_count=len(full_variants["corridorVariantOffset"]) - 1,
    )
    target_dormant_count = dormant_addition_count[target_to_full]
    enumerated_at = time.monotonic()
    if progress is not None:
        progress(
            f"exact-screening {len(target_to_full)} dormant-supported states "
            f"across {len(set(int(value) for value in target_variants['corridorVariantRow']))} corridors"
        )
    exact, exact_stats = screen_exact_corridor_variants(
        remapped,
        remapped,
        remapped,
        target_variants,
        ribbon,
        expanded_topology,
        conditioned,
        corridor_settings=corridor_settings,
        settings=variant_settings,
        progress=progress,
    )
    screened_at = time.monotonic()
    incremental_compiled = compile_exact_variant_reconfiguration(
        reconfiguration, target_variants, exact
    )
    if progress is not None:
        progress("replaying exact dormant-ribbon corridor states")
    incremental_replay, incremental_replay_stats = (
        replay_patch_corridor_reconfigurations(
            remapped,
            remapped,
            remapped,
            incremental_compiled,
            ribbon,
            expanded_topology,
            conditioned,
            settings=corridor_settings,
        )
    )
    incremental_replayed_at = time.monotonic()
    prior_exact_override = dict(prior_variants)
    prior_exact_override["corridorChosenExactVariant"] = np.asarray(
        prior_sets["corridorChosenGlobalVariant"], dtype=np.int32
    )
    prior_compiled = compile_exact_variant_reconfiguration(
        corridor, prior_variants, prior_exact_override
    )
    combined_compiled, combination_arrays = (
        _combine_compiled_reconfigurations(
            reconfiguration,
            prior_compiled,
            incremental_compiled,
            prior_frontier_to_expanded=old_to_expanded,
        )
    )
    if progress is not None:
        progress("replaying prior and dormant exact corridors as one surface")
    replay, replay_stats = replay_patch_corridor_reconfigurations(
        remapped,
        remapped,
        remapped,
        combined_compiled,
        ribbon,
        expanded_topology,
        conditioned,
        settings=corridor_settings,
    )
    replayed_at = time.monotonic()
    arrays = {
        **reconfiguration,
        **target_variants,
        **exact,
        **replay,
        **{
            f"incremental{name[0].upper()}{name[1:]}": value
            for name, value in incremental_replay.items()
        },
        **combination_arrays,
        "oldFrontierToExpandedFrontier": old_to_expanded,
        "targetVariantFullEnumerationIndex": target_to_full,
        "targetVariantDormantAdditionCount": target_dormant_count,
    }
    _write_npz(data_path, arrays)
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    write_patch_corridor_montage(
        remapped,
        remapped,
        montage_path,
        maximum_corridors=corridor_settings.maximum_preview_corridors,
        reconfiguration=combined_compiled,
        replay=replay,
    )
    _, fragment_stats = write_replayed_corridor_fragment_montage(
        remapped,
        remapped,
        remapped,
        replay,
        source,
        fragment_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    chosen = np.asarray(exact["corridorChosenExactVariant"], dtype=np.int32)
    chosen_rows = np.flatnonzero(chosen >= 0)
    incremental_successful_rows = np.flatnonzero(
        np.asarray(
            incremental_replay["corridorReplayProposalSuccessful"]
        )
        > 0
    )
    successful_rows = np.flatnonzero(
        np.asarray(replay["corridorReplayProposalSuccessful"]) > 0
    )
    prior_selected_rows = np.flatnonzero(
        np.asarray(prior_compiled["corridorEvidenceEligible"]) > 0
    )
    baseline_triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    final_triangle = np.asarray(
        replay["corridorReplayTriangleFrontierIndex"], dtype=np.int32
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DORMANT_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_DORMANT_CORRIDORS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": corridor_manifest.get("geometry", {}),
        "conditioning": conditioning_stats,
        "surface": surface_stats,
        "reconfiguration": reconfiguration_stats,
        "enumeration": enumeration_stats,
        "target": {
            "priorResolvedCorridorCount": len(prior_resolved),
            "residualCorridorCount": len(residual_rows),
            "dormantSupportedVariantCount": len(target_to_full),
            "corridorWithDormantSupportedVariantCount": len(
                set(int(value) for value in target_variants["corridorVariantRow"])
            ),
            "exactResolvedCorridorCount": len(chosen_rows),
            "exactResolvedCorridorRows": [int(value) for value in chosen_rows],
            "incrementalSuccessfulReplayCorridorCount": len(
                incremental_successful_rows
            ),
            "incrementalSuccessfulReplayCorridorRows": [
                int(value) for value in incremental_successful_rows
            ],
            "priorGloballySelectedCorridorCount": len(prior_selected_rows),
            "combinedSuccessfulReplayCorridorCount": len(successful_rows),
            "combinedSuccessfulReplayCorridorRows": [
                int(value) for value in successful_rows
            ],
        },
        "exactScreen": exact_stats,
        "incrementalCounterfactualReplay": incremental_replay_stats,
        "counterfactualReplay": replay_stats,
        "surfaceAudit": {
            "edgeConnectedTriangleRegionCountBefore": int(
                len(np.unique(_triangle_region_labels(baseline_triangle)))
            ),
            "edgeConnectedTriangleRegionCountAfter": int(
                len(np.unique(_triangle_region_labels(final_triangle)))
            ),
            "retainedTriangleCountBefore": len(baseline_triangle),
            "retainedTriangleCountAfter": len(final_triangle),
        },
        "flattenedReplayFragments": fragment_stats,
        "timingSeconds": {
            "conditioningAndCrossings": round(conditioned_at - started, 6),
            "baseSurfaceAndRemap": round(surfaced_at - conditioned_at, 6),
            "corridorReconfiguration": round(
                reconfigured_at - surfaced_at, 6
            ),
            "variantEnumeration": round(enumerated_at - reconfigured_at, 6),
            "exactScreen": round(screened_at - enumerated_at, 6),
            "incrementalReplay": round(
                incremental_replayed_at - screened_at, 6
            ),
            "combinedReplay": round(
                replayed_at - incremental_replayed_at, 6
            ),
            "writingAndPreviews": round(finished - replayed_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "corridorMontage": montage_path.name,
            "flattenedReplayFragments": fragment_path.name,
        },
        "method": {
            "decisionUnit": (
                "complete residual-corridor matchings containing at least one "
                "formerly dormant bidirectional physical ribbon"
            ),
            "baseSelection": "mapped unchanged into the expanded frontier",
            "acceptance": "exact full-sheet reconstruction and density-preserving replay",
            "cumulativeReplay": (
                "the prior global exact assignment and newly recovered "
                "corridors are replayed jointly on the expanded frontier"
            ),
            "mutation": "counterfactual only; all input artifacts remain unchanged",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
