from __future__ import annotations

import colorsys
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .export import rgb_png
from .isolated_slab import _percentile_record
from .one_sided_growth import (
    ONE_SIDED_GROWTH_SCHEMA,
    ONE_SIDED_GROWTH_STEM,
)
from .one_sided_interface import (
    ONE_SIDED_INTERFACE_SCHEMA,
    OneSidedInterfaceSettings,
    match_signed_interface_endpoints,
)
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA
from .paired_surface_growth import PAIRED_SURFACE_GROWTH_SCHEMA


CLEAR_RIBBON_SCHEMA = "pareidolia.clear-two-face-ribbon-bank"
CLEAR_RIBBON_VERSION = 1
CLEAR_RIBBON_STEM = "clear-ribbon-bank-v1"


@dataclass(frozen=True, slots=True)
class ClearRibbonSettings:
    maximum_endpoint_position_residual_sampling_steps: float = 0.75
    maximum_endpoint_normal_degrees: float = 15.0
    endpoint_match_normal_scale_degrees: float = 15.0
    minimum_continuity_affinity: float = 0.35
    minimum_substantial_component_size: int = 8
    maximum_preview_components: int = 128

    def __post_init__(self) -> None:
        positive = (
            self.maximum_endpoint_position_residual_sampling_steps,
            self.maximum_endpoint_normal_degrees,
            self.endpoint_match_normal_scale_degrees,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("clear-ribbon matching scales must be positive")
        if not 0.0 < self.maximum_endpoint_normal_degrees < 90.0:
            raise ValueError("clear-ribbon normal cap must lie in (0, 90)")
        if not 0.0 < self.minimum_continuity_affinity <= 1.0:
            raise ValueError("clear-ribbon continuity affinity must lie in (0, 1]")
        if (
            self.minimum_substantial_component_size < 1
            or self.maximum_preview_components < 1
        ):
            raise ValueError("clear-ribbon component counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact data hash differs from manifest: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _load_inputs(
    root: str | Path,
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    value = Path(root).resolve()
    growth_path = (
        value if value.is_file() else value / f"{ONE_SIDED_GROWTH_STEM}.json"
    )
    growth_manifest = json.loads(growth_path.read_text())
    if (
        growth_manifest.get("schema") != ONE_SIDED_GROWTH_SCHEMA
        or growth_manifest.get("state") != "complete"
    ):
        raise ValueError("clear ribbons require complete one-sided growth")
    growth_data_path = growth_path.parent / str(growth_manifest["data"]["path"])
    growth = _load_npz(growth_data_path, growth_manifest["data"]["sha256"])

    interface_path = Path(
        growth_manifest["identity"]["interfaceBank"]["manifestPath"]
    )
    if (
        sha256_file(interface_path)
        != growth_manifest["identity"]["interfaceBank"]["manifestSha256"]
    ):
        raise ValueError("one-sided interface manifest changed after growth")
    interface_manifest = json.loads(interface_path.read_text())
    if interface_manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA:
        raise ValueError("one-sided growth references the wrong interface schema")
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    interface = _load_npz(
        interface_data_path, interface_manifest["data"]["sha256"]
    )

    paired_growth_path = Path(
        interface_manifest["identity"]["pairedGrowth"]["manifestPath"]
    )
    if (
        sha256_file(paired_growth_path)
        != interface_manifest["identity"]["pairedGrowth"]["manifestSha256"]
    ):
        raise ValueError("paired-growth manifest changed after interface extraction")
    paired_growth_manifest = json.loads(paired_growth_path.read_text())
    if paired_growth_manifest.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA:
        raise ValueError("interface bank references the wrong paired-growth schema")
    paired_growth_data_path = paired_growth_path.parent / str(
        paired_growth_manifest["data"]["path"]
    )
    paired_growth = _load_npz(
        paired_growth_data_path, paired_growth_manifest["data"]["sha256"]
    )

    paired_bank_path = Path(
        interface_manifest["identity"]["pairedBank"]["manifestPath"]
    )
    if (
        sha256_file(paired_bank_path)
        != interface_manifest["identity"]["pairedBank"]["manifestSha256"]
    ):
        raise ValueError("paired-bank manifest changed after interface extraction")
    paired_bank_manifest = json.loads(paired_bank_path.read_text())
    if paired_bank_manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA:
        raise ValueError("interface bank references the wrong paired-bank schema")
    paired_bank_data_path = paired_bank_path.parent / str(
        paired_bank_manifest["data"]["path"]
    )
    paired_bank = _load_npz(
        paired_bank_data_path, paired_bank_manifest["data"]["sha256"]
    )
    return (
        growth_path,
        growth_manifest,
        growth,
        interface_path,
        interface_manifest,
        interface,
        paired_growth_path,
        paired_growth_manifest,
        paired_growth,
        paired_bank_path,
        paired_bank_manifest,
        paired_bank,
    )


def build_clear_ribbons(
    interface: Mapping[str, np.ndarray],
    one_sided_growth: Mapping[str, np.ndarray],
    paired_bank: Mapping[str, np.ndarray],
    paired_growth: Mapping[str, np.ndarray],
    *,
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: ClearRibbonSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Map both physical faces, then collapse reciprocal profile duplicates."""

    paired_normal = np.asarray(paired_bank["normalXYZ"], dtype=np.float32)
    candidate_count = len(paired_normal)
    endpoint_position = np.concatenate(
        (
            np.asarray(paired_bank["boundaryLowerXYZ"]),
            np.asarray(paired_bank["boundaryUpperXYZ"]),
        )
    ).astype(np.float32)
    endpoint_normal = np.concatenate((paired_normal, -paired_normal)).astype(
        np.float32
    )
    match_settings = OneSidedInterfaceSettings(
        maximum_seed_position_residual_sampling_steps=(
            settings.maximum_endpoint_position_residual_sampling_steps
        ),
        maximum_seed_normal_degrees=settings.maximum_endpoint_normal_degrees,
        seed_match_normal_scale_degrees=(
            settings.endpoint_match_normal_scale_degrees
        ),
    )
    endpoint_match = match_signed_interface_endpoints(
        endpoint_position,
        endpoint_normal,
        np.asarray(interface["positionXYZ"]),
        np.asarray(interface["signedNormalXYZ"]),
        np.asarray(interface["processingKeyXYZ"]),
        processing_start_xyz=processing_start_xyz,
        source_origin_xyz=source_origin_xyz,
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        stride=stride,
        settings=match_settings,
    )
    endpoint_interface = endpoint_match["interfaceIndex"]
    lower_interface = endpoint_interface[:candidate_count]
    upper_interface = endpoint_interface[candidate_count:]
    both_matched = (
        (lower_interface >= 0)
        & (upper_interface >= 0)
        & (lower_interface != upper_interface)
    )
    mapped_candidate = np.flatnonzero(both_matched).astype(np.int32)
    low = lower_interface[mapped_candidate]
    high = upper_interface[mapped_candidate]
    encoded_pair = (
        np.minimum(low, high).astype(np.int64) << 32
    ) | np.maximum(low, high).astype(np.int64)
    local_evidence = np.asarray(
        paired_bank["localEvidenceScore"], dtype=np.float32
    )
    selected = np.asarray(paired_growth["selected"]) > 0
    seed = np.asarray(paired_bank["seedComponentId"]) >= 0
    endpoint_cost = (
        endpoint_match["matchCost"][:candidate_count]
        + endpoint_match["matchCost"][candidate_count:]
    )
    order = np.lexsort(
        (
            mapped_candidate,
            endpoint_cost[mapped_candidate],
            -local_evidence[mapped_candidate],
            -selected[mapped_candidate].astype(np.int8),
            -seed[mapped_candidate].astype(np.int8),
            encoded_pair,
        )
    )
    sorted_pair = encoded_pair[order]
    first = np.concatenate(
        ((True,), sorted_pair[1:] != sorted_pair[:-1])
    )
    representative = mapped_candidate[order[first]]
    unique_pair, inverse, alternative_count = np.unique(
        encoded_pair, return_inverse=True, return_counts=True
    )
    ribbon_by_candidate = np.full(candidate_count, -1, dtype=np.int32)
    ribbon_by_candidate[mapped_candidate] = inverse.astype(np.int32)
    if not np.array_equal(unique_pair, sorted_pair[first]):
        raise RuntimeError("clear-ribbon pair indexing is not deterministic")

    lower_match_cost = endpoint_match["matchCost"][:candidate_count]
    upper_match_cost = endpoint_match["matchCost"][candidate_count:]
    lower_position = endpoint_match["positionResidualSamplingSteps"][
        :candidate_count
    ]
    upper_position = endpoint_match["positionResidualSamplingSteps"][
        candidate_count:
    ]
    lower_normal = endpoint_match["normalResidualDegrees"][:candidate_count]
    upper_normal = endpoint_match["normalResidualDegrees"][candidate_count:]
    selected_surface_label = np.asarray(
        paired_growth["selectedLabel"], dtype=np.int32
    )[representative]
    surface_label = np.asarray(
        one_sided_growth["surfaceLabel"], dtype=np.int32
    )
    surface_assembly = np.asarray(
        one_sided_growth["surfaceAssemblyLabel"], dtype=np.int32
    )
    assembly_lookup = {
        int(label): int(assembly)
        for label, assembly in zip(surface_label, surface_assembly)
    }
    selected_assembly = np.asarray(
        [
            assembly_lookup.get(int(label), -1) if label >= 0 else -1
            for label in selected_surface_label
        ],
        dtype=np.int32,
    )
    interface_component = np.asarray(
        one_sided_growth["interfaceContinuityComponent"], dtype=np.int32
    )
    component_seed_label_count = np.asarray(
        one_sided_growth["componentSeedLabelCount"]
    )
    ribbon_lower = lower_interface[representative]
    ribbon_upper = upper_interface[representative]
    lower_component = interface_component[ribbon_lower]
    upper_component = interface_component[ribbon_upper]
    arrays = {
        "pairedCandidateToRibbon": ribbon_by_candidate,
        "pairedCandidateIndex": representative.astype(np.int32),
        "lowerInterface": ribbon_lower.astype(np.int32),
        "upperInterface": ribbon_upper.astype(np.int32),
        "lowerInterfaceComponent": lower_component.astype(np.int32),
        "upperInterfaceComponent": upper_component.astype(np.int32),
        "lowerComponentSeedLabelCount": component_seed_label_count[
            lower_component
        ].astype(np.uint16),
        "upperComponentSeedLabelCount": component_seed_label_count[
            upper_component
        ].astype(np.uint16),
        "alternativeProfileCount": alternative_count.astype(np.uint8),
        "selectedPairedSurface": selected[representative].astype(np.uint8),
        "lockedPairedSeed": seed[representative].astype(np.uint8),
        "selectedSurfaceLabel": selected_surface_label,
        "selectedAssemblyLabel": selected_assembly,
        "spatialKeyXYZ": np.asarray(paired_bank["spatialKeyXYZ"])[
            representative
        ].astype(np.int32),
        "midpointXYZ": np.asarray(paired_bank["midpointXYZ"])[
            representative
        ].astype(np.float32),
        "normalXYZ": paired_normal[representative].astype(np.float32),
        "boundaryLowerXYZ": np.asarray(paired_bank["boundaryLowerXYZ"])[
            representative
        ].astype(np.float32),
        "boundaryUpperXYZ": np.asarray(paired_bank["boundaryUpperXYZ"])[
            representative
        ].astype(np.float32),
        "thicknessVoxels": np.asarray(paired_bank["thicknessVoxels"])[
            representative
        ].astype(np.float32),
        "localEvidenceScore": local_evidence[representative],
        "lowerEndpointMatchCost": lower_match_cost[representative],
        "upperEndpointMatchCost": upper_match_cost[representative],
        "lowerEndpointPositionResidualSamplingSteps": lower_position[
            representative
        ],
        "upperEndpointPositionResidualSamplingSteps": upper_position[
            representative
        ],
        "lowerEndpointNormalResidualDegrees": lower_normal[representative],
        "upperEndpointNormalResidualDegrees": upper_normal[representative],
    }
    return arrays, {
        "pairedCandidateCount": candidate_count,
        "lowerEndpointMatchedCandidateCount": int(
            np.count_nonzero(lower_interface >= 0)
        ),
        "upperEndpointMatchedCandidateCount": int(
            np.count_nonzero(upper_interface >= 0)
        ),
        "bothEndpointsMatchedCandidateCount": int(len(mapped_candidate)),
        "uniqueRibbonCount": int(len(representative)),
        "duplicateProfileCount": int(len(mapped_candidate) - len(representative)),
        "ribbonWithAlternativeProfilesCount": int(
            np.count_nonzero(alternative_count > 1)
        ),
        "maximumAlternativeProfileCount": int(
            np.max(alternative_count, initial=0)
        ),
        "selectedRibbonCount": int(
            np.count_nonzero(arrays["selectedPairedSurface"])
        ),
        "lockedSeedRibbonCount": int(
            np.count_nonzero(arrays["lockedPairedSeed"])
        ),
        "endpointPositionResidualSamplingSteps": _percentile_record(
            np.concatenate(
                (
                    lower_position[mapped_candidate],
                    upper_position[mapped_candidate],
                )
            )
        ),
        "endpointNormalResidualDegrees": _percentile_record(
            np.concatenate(
                (
                    lower_normal[mapped_candidate],
                    upper_normal[mapped_candidate],
                )
            )
        ),
    }


def build_clear_ribbon_graph(
    ribbons: Mapping[str, np.ndarray],
    paired_growth: Mapping[str, np.ndarray],
    *,
    settings: ClearRibbonSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Collapse the persisted strict paired-profile graph onto ribbons."""

    ribbon_by_candidate = np.asarray(
        ribbons["pairedCandidateToRibbon"], dtype=np.int32
    )
    first_candidate = np.asarray(
        paired_growth["edgeFirstCandidate"], dtype=np.int32
    )
    second_candidate = np.asarray(
        paired_growth["edgeSecondCandidate"], dtype=np.int32
    )
    affinity = np.asarray(paired_growth["edgeAffinity"], dtype=np.float32)
    first_ribbon = ribbon_by_candidate[first_candidate]
    second_ribbon = ribbon_by_candidate[second_candidate]
    valid = (
        (first_ribbon >= 0)
        & (second_ribbon >= 0)
        & (first_ribbon != second_ribbon)
        & (affinity >= settings.minimum_continuity_affinity)
    )
    original_edge = np.flatnonzero(valid).astype(np.int32)
    low = np.minimum(first_ribbon[valid], second_ribbon[valid]).astype(np.int64)
    high = np.maximum(first_ribbon[valid], second_ribbon[valid]).astype(np.int64)
    encoded = (low << 32) | high
    order = np.lexsort((-affinity[original_edge], encoded))
    sorted_encoded = encoded[order]
    unique = np.concatenate(
        ((True,), sorted_encoded[1:] != sorted_encoded[:-1])
    )
    chosen_edge = original_edge[order[unique]]
    edge_first = low[order[unique]].astype(np.int32)
    edge_second = high[order[unique]].astype(np.int32)
    graph: dict[str, np.ndarray] = {
        "edgeFirstRibbon": edge_first,
        "edgeSecondRibbon": edge_second,
        "edgePairedGrowthIndex": chosen_edge,
        "edgeAffinity": affinity[chosen_edge],
    }
    for name, values in paired_growth.items():
        if (
            name.startswith("edge")
            and name
            not in {"edgeFirstCandidate", "edgeSecondCandidate", "edgeAffinity"}
            and len(values) == len(affinity)
        ):
            graph[name] = np.asarray(values)[chosen_edge]
    return graph, {
        "mappedOriginalContinuityEdgeCount": int(len(original_edge)),
        "continuityEdgeCount": int(len(edge_first)),
        "duplicateContinuityEdgeCount": int(len(original_edge) - len(edge_first)),
        "edgeAffinity": _percentile_record(graph["edgeAffinity"]),
        "edgeNormalDegrees": _percentile_record(
            graph.get("edgeNormalDegrees", np.empty(0))
        ),
        "edgeBoundaryHeightSamplingSteps": _percentile_record(
            graph.get("edgeBoundaryHeightSamplingSteps", np.empty(0))
        ),
    }


def label_clear_ribbon_components(
    ribbons: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: ClearRibbonSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Inventory coherent ribbon components without selecting alternatives."""

    ribbon_count = len(ribbons["pairedCandidateIndex"])
    parent = np.arange(ribbon_count, dtype=np.int32)
    size = np.ones(ribbon_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    for first, second in zip(
        graph["edgeFirstRibbon"], graph["edgeSecondRibbon"]
    ):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root == second_root:
            continue
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size[second_root]
    roots = np.asarray([find(value) for value in range(ribbon_count)], dtype=np.int32)
    _root, component, component_size = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    component_count = len(component_size)
    selected_assembly = np.asarray(
        ribbons["selectedAssemblyLabel"], dtype=np.int32
    )
    component_assemblies: list[set[int]] = [set() for _ in range(component_count)]
    component_selected_count = np.zeros(component_count, dtype=np.int32)
    component_locked_count = np.zeros(component_count, dtype=np.int32)
    for ribbon in range(ribbon_count):
        value = int(component[ribbon])
        if selected_assembly[ribbon] >= 0:
            component_assemblies[value].add(int(selected_assembly[ribbon]))
            component_selected_count[value] += 1
        component_locked_count[value] += int(ribbons["lockedPairedSeed"][ribbon])
    component_assembly_count = np.asarray(
        [len(values) for values in component_assemblies], dtype=np.uint16
    )
    component_sole_assembly = np.asarray(
        [
            next(iter(values)) if len(values) == 1 else -1
            for values in component_assemblies
        ],
        dtype=np.int32,
    )

    flat_key = np.ravel_multi_index(
        np.asarray(ribbons["spatialKeyXYZ"], dtype=np.int32)[:, ::-1].T,
        processing_shape_sampling_xyz[::-1],
    )
    order = np.lexsort((flat_key, component))
    ordered_component = component[order]
    ordered_key = flat_key[order]
    duplicate = np.zeros(ribbon_count, dtype=bool)
    duplicate[1:] = (
        (ordered_component[1:] == ordered_component[:-1])
        & (ordered_key[1:] == ordered_key[:-1])
    )
    component_key_collision_count = np.bincount(
        ordered_component[duplicate], minlength=component_count
    ).astype(np.int32)
    substantial = component_size >= settings.minimum_substantial_component_size
    records = []
    for value in np.argsort(-component_size)[:128]:
        records.append(
            {
                "component": int(value),
                "ribbonCount": int(component_size[value]),
                "selectedRibbonCount": int(component_selected_count[value]),
                "lockedSeedRibbonCount": int(component_locked_count[value]),
                "assemblyCount": int(component_assembly_count[value]),
                "assemblies": sorted(component_assemblies[value])[:32],
                "spatialKeyCollisionCount": int(
                    component_key_collision_count[value]
                ),
            }
        )
    state_stats = {}
    for name, mask in (
        ("unseeded", component_assembly_count == 0),
        ("singleAssembly", component_assembly_count == 1),
        ("contested", component_assembly_count > 1),
    ):
        state_stats[name] = {
            "componentCount": int(np.count_nonzero(mask)),
            "ribbonCount": int(np.sum(component_size[mask])),
            "substantialComponentCount": int(np.count_nonzero(mask & substantial)),
            "componentSize": _percentile_record(component_size[mask]),
        }
    arrays = {
        "ribbonComponent": component.astype(np.int32),
        "componentRibbonCount": component_size.astype(np.int32),
        "componentSelectedRibbonCount": component_selected_count,
        "componentLockedSeedRibbonCount": component_locked_count,
        "componentAssemblyCount": component_assembly_count,
        "componentSoleAssemblyLabel": component_sole_assembly,
        "componentSpatialKeyCollisionCount": component_key_collision_count,
    }
    return arrays, {
        "componentCount": int(component_count),
        "componentSize": _percentile_record(component_size),
        "componentCountAtLeast8": int(np.count_nonzero(component_size >= 8)),
        "componentCountAtLeast32": int(np.count_nonzero(component_size >= 32)),
        "componentCountAtLeast128": int(np.count_nonzero(component_size >= 128)),
        "componentWithSpatialKeyCollisionCount": int(
            np.count_nonzero(component_key_collision_count > 0)
        ),
        "spatialKeyCollisionCount": int(np.sum(component_key_collision_count)),
        "states": state_stats,
        "largestComponents": records,
    }


def _component_colors(
    component: np.ndarray,
    size: np.ndarray,
    maximum: int,
) -> dict[int, tuple[int, int, int]]:
    order = np.argsort(-size)[:maximum]
    return {
        int(value): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.07 + 0.61803398875 * rank) % 1.0, 0.68, 0.98
            )
        )
        for rank, value in enumerate(order)
    }


def write_clear_ribbon_projection(
    ribbons: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    path: str | Path,
    *,
    maximum_components: int,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    point = np.asarray(ribbons["midpointXYZ"])
    component = np.asarray(components["ribbonComponent"], dtype=np.int32)
    size = np.asarray(components["componentRibbonCount"], dtype=np.int32)
    colors = _component_colors(component, size, maximum_components)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        for value, color in colors.items():
            points = point[component == value]
            normalized = (
                points[:, list(axes)] - world_start_xyz[None, list(axes)]
            ) / width[None, list(axes)]
            x = np.rint(
                offset + margin + normalized[:, 0] * (panel_size - 2 * margin)
            ).astype(np.int32)
            y = np.rint(
                panel_size - margin - normalized[:, 1] * (panel_size - 2 * margin)
            ).astype(np.int32)
            valid = (
                (x >= offset)
                & (x < offset + panel_size)
                & (y >= 0)
                & (y < panel_size)
            )
            canvas[y[valid], x[valid]] = color
        border = (64, 72, 84)
        canvas[margin, offset + margin : offset + panel_size - margin] = border
        canvas[
            panel_size - margin,
            offset + margin : offset + panel_size - margin,
        ] = border
        canvas[margin : panel_size - margin, offset + margin] = border
        canvas[
            margin : panel_size - margin, offset + panel_size - margin
        ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _draw_line(
    panel: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    color: tuple[int, int, int],
) -> None:
    length = int(max(np.max(np.abs(second - first)), 1)) + 1
    value = np.linspace(first, second, length)
    x = np.rint(value[:, 0]).astype(np.int32)
    y = np.rint(value[:, 1]).astype(np.int32)
    valid = (x >= 0) & (x < panel.shape[1]) & (y >= 0) & (y < panel.shape[0])
    panel[y[valid], x[valid]] = color


def write_clear_ribbon_cross_sections(
    source: VolumeSource,
    owned: VoxelBounds,
    ribbons: Mapping[str, np.ndarray],
    components: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    display_high_raw: float,
    sampling_stride: int,
) -> Path:
    output = Path(path)
    volume = source.memmap()
    midpoint = np.asarray(ribbons["midpointXYZ"])
    lower = np.asarray(ribbons["boundaryLowerXYZ"])
    upper = np.asarray(ribbons["boundaryUpperXYZ"])
    component = np.asarray(components["ribbonComponent"], dtype=np.int32)
    assembly_count = np.asarray(components["componentAssemblyCount"])
    world_start = np.asarray(owned.start_xyz) + np.asarray(source.origin_xyz)
    z_values = np.linspace(owned.start_xyz[2], owned.stop_xyz_exclusive[2] - 1, 3)
    y_values = np.linspace(owned.start_xyz[1], owned.stop_xyz_exclusive[1] - 1, 3)
    views = [("z", int(round(value))) for value in z_values] + [
        ("y", int(round(value))) for value in y_values
    ]
    panel_width = owned.shape_xyz[0]
    panel_height = max(owned.shape_xyz[1], owned.shape_xyz[2])
    canvas = np.full(
        (2 * panel_height, 3 * panel_width, 3), (7, 10, 14), dtype=np.uint8
    )
    tolerance = max(1.0, sampling_stride)
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[2]
            near = np.flatnonzero(np.abs(midpoint[:, 2] - coordinate) <= tolerance)
            axes = (0, 1)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[1]
            near = np.flatnonzero(np.abs(midpoint[:, 1] - coordinate) <= tolerance)
            axes = (0, 2)
        gray = np.clip(
            np.asarray(raw, dtype=np.float32) / max(display_high_raw, 1.0) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        for ribbon in near:
            state = int(assembly_count[component[ribbon]])
            color = (255, 92, 196) if state > 1 else (
                (43, 226, 190) if state == 1 else (255, 174, 62)
            )
            first = lower[ribbon, list(axes)] - world_start[list(axes)]
            second = upper[ribbon, list(axes)] - world_start[list(axes)]
            _draw_line(panel, first, second, color)
        row, column = divmod(view_index, 3)
        y0 = row * panel_height + (panel_height - panel.shape[0]) // 2
        x0 = column * panel_width
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_clear_ribbon_bank(
    growth_root: str | Path,
    output_root: str | Path,
    *,
    settings: ClearRibbonSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or ClearRibbonSettings()
    (
        growth_path,
        growth_manifest,
        growth,
        interface_path,
        interface_manifest,
        interface,
        paired_growth_path,
        paired_growth_manifest,
        paired_growth,
        paired_bank_path,
        paired_bank_manifest,
        paired_bank,
    ) = _load_inputs(growth_root)
    slab_path = Path(
        paired_bank_manifest["identity"]["isolatedSlabs"]["manifestPath"]
    )
    slab_manifest = json.loads(slab_path.read_text())
    stride = int(slab_manifest["identity"]["settings"]["sampling_stride_voxels"])
    geometry = interface_manifest["geometry"]
    processing_shape = tuple(
        int(value) for value in geometry["processingShapeSamplingXYZ"]
    )
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    source_origin = np.asarray(
        interface_manifest["source"]["sourceOriginXYZ"], dtype=np.float32
    )
    identity: dict[str, Any] = {
        "schema": CLEAR_RIBBON_SCHEMA,
        "version": CLEAR_RIBBON_VERSION,
        "oneSidedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "oneSidedInterfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "pairedGrowth": {
            "manifestPath": str(paired_growth_path),
            "manifestSha256": sha256_file(paired_growth_path),
            "dataSha256": paired_growth_manifest["data"]["sha256"],
        },
        "pairedBank": {
            "manifestPath": str(paired_bank_path),
            "manifestSha256": sha256_file(paired_bank_path),
            "dataSha256": paired_bank_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{CLEAR_RIBBON_STEM}.json"
    data_path = output / f"{CLEAR_RIBBON_STEM}.npz"
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
    ribbons, ribbon_stats = build_clear_ribbons(
        interface,
        growth,
        paired_bank,
        paired_growth,
        processing_start_xyz=processing_start,
        source_origin_xyz=source_origin,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=resolved,
    )
    mapped = time.monotonic()
    graph, graph_stats = build_clear_ribbon_graph(
        ribbons, paired_growth, settings=resolved
    )
    graphed = time.monotonic()
    components, component_stats = label_clear_ribbon_components(
        ribbons,
        graph,
        processing_shape_sampling_xyz=processing_shape,
        settings=resolved,
    )
    labeled = time.monotonic()
    arrays = {**ribbons, **graph, **components}
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        interface_manifest["source"]["path"],
        interface_manifest["source"]["metadataPath"],
    )
    owned_record = geometry["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    world_record = geometry["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    projection = write_clear_ribbon_projection(
        ribbons,
        components,
        world_start,
        world_stop,
        output / "clear-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    cross_sections = write_clear_ribbon_cross_sections(
        source,
        owned,
        ribbons,
        components,
        output / "clear-ribbon-cross-sections.png",
        display_high_raw=float(interface_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=stride,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": CLEAR_RIBBON_SCHEMA,
        "version": CLEAR_RIBBON_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interface_manifest["source"],
        "geometry": geometry,
        "calibration": interface_manifest["calibration"],
        "samplingStrideVoxels": stride,
        "ribbons": ribbon_stats,
        "graph": graph_stats,
        "components": component_stats,
        "timingSeconds": {
            "endpointMappingAndDeduplication": round(mapped - started, 6),
            "continuityGraph": round(graphed - mapped, 6),
            "componentCensus": round(labeled - graphed, 6),
            "writingAndArtifacts": round(finished - labeled, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "components": projection.name,
            "crossSections": cross_sections.name,
        },
        "method": {
            "geometryUnit": "two exact strong signed CT faces",
            "candidateSource": "physically bounded paired-profile bank",
            "continuitySource": "strict persisted two-boundary paired graph",
            "selectsNewRibbons": False,
            "preservesAlternatives": True,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
