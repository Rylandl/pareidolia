from __future__ import annotations

import json
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_variants import (
    PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM,
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
    compile_exact_variant_reconfiguration,
)
from .physical_ribbon_patch_corridors import (
    PhysicalRibbonPatchCorridorSettings,
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    replay_patch_corridor_reconfigurations,
    write_patch_corridor_montage,
    write_replayed_corridor_fragment_montage,
)


PHYSICAL_RIBBON_CORRIDOR_SETS_SCHEMA = "pareidolia.physical-ribbon-corridor-sets"
PHYSICAL_RIBBON_CORRIDOR_SETS_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_SETS_STEM = "physical-ribbon-corridor-sets-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorSetSettings:
    component_assignment_beam_width: int = 256
    maximum_retained_states_per_component: int = 16
    global_assignment_beam_width: int = 4096
    minimum_surface_area_retention: float = 0.98
    maximum_preview_components: int = 8

    def __post_init__(self) -> None:
        if self.component_assignment_beam_width < 1:
            raise ValueError("component assignment beam width must be positive")
        if self.maximum_retained_states_per_component < 2:
            raise ValueError(
                "at least two component states must be retained"
            )
        if self.global_assignment_beam_width < 1:
            raise ValueError("global assignment beam width must be positive")
        if not 0.0 < self.minimum_surface_area_retention <= 1.0:
            raise ValueError("surface-area retention must lie in (0, 1]")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_variant_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("global corridor sets require a complete label-free variant artifact")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _variant_values(
    variants: Mapping[str, np.ndarray],
    variant_index: int,
    prefix: str,
) -> np.ndarray:
    offset = np.asarray(variants[f"corridorVariant{prefix}Offset"], dtype=np.int64)
    value = np.asarray(
        variants[f"corridorVariant{prefix}FrontierIndex"], dtype=np.int32
    )
    return value[int(offset[variant_index]) : int(offset[variant_index + 1])]


def _sum_key(
    first: tuple[float, ...], second: tuple[float, ...]
) -> tuple[float, ...]:
    return tuple(left + right for left, right in zip(first, second))


def _modifications_valid(
    added: frozenset[int],
    removed: frozenset[int],
    *,
    baseline_selected: np.ndarray,
    baseline_interface_owner: Mapping[int, int],
    source_interface: np.ndarray,
    target_interface: np.ndarray,
    crossing_neighbor: Sequence[frozenset[int]],
) -> bool:
    if added & removed:
        return False
    added_interface_owner: dict[int, int] = {}
    for node_value in added:
        for interface in (
            int(source_interface[node_value]),
            int(target_interface[node_value]),
        ):
            baseline_owner = baseline_interface_owner.get(interface)
            if (
                baseline_owner is not None
                and baseline_owner != node_value
                and baseline_owner not in removed
            ):
                return False
            added_owner = added_interface_owner.get(interface)
            if added_owner is not None and added_owner != node_value:
                return False
            added_interface_owner[interface] = node_value
    return not any(
        any(
            neighbor in added
            or (baseline_selected[neighbor] and neighbor not in removed)
            for neighbor in crossing_neighbor[node_value]
            if neighbor != node_value
        )
        for node_value in added
    )


def _choose_global_variant_states(
    states_by_component: Sequence[Sequence[dict[str, Any]]],
    *,
    valid_modifications: Callable[[frozenset[int], frozenset[int]], bool],
    beam_width: int,
) -> dict[str, Any]:
    """Choose one exact state per component under block-wide hard conflicts."""

    beam: list[dict[str, Any]] = [
        {
            "key": (0.0, 0.0, 0.0, 0.0, 0.0),
            "stateIndices": (),
            "variantIndices": (),
            "added": frozenset(),
            "removed": frozenset(),
        }
    ]
    for component_states in states_by_component:
        best_by_signature: dict[
            tuple[frozenset[int], frozenset[int]], dict[str, Any]
        ] = {}
        for current in beam:
            for state in component_states:
                added = current["added"] | state["added"]
                removed = current["removed"] | state["removed"]
                signature = (added, removed)
                if not valid_modifications(added, removed):
                    continue
                candidate = {
                    "key": _sum_key(current["key"], state["key"]),
                    "stateIndices": current["stateIndices"]
                    + (int(state["stateIndex"]),),
                    "variantIndices": current["variantIndices"]
                    + tuple(int(value) for value in state["variantIndices"]),
                    "added": added,
                    "removed": removed,
                }
                previous = best_by_signature.get(signature)
                if previous is None or (
                    candidate["key"], candidate["variantIndices"]
                ) > (previous["key"], previous["variantIndices"]):
                    best_by_signature[signature] = candidate
        expanded = list(best_by_signature.values())
        expanded.sort(
            key=lambda value: (value["key"], value["variantIndices"]),
            reverse=True,
        )
        beam = expanded[:beam_width]
        if not beam:
            raise RuntimeError("global corridor state beam became empty")
    return beam[0]


def optimize_exact_corridor_variant_sets(
    surface: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    variants: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    corridor_settings: PhysicalRibbonPatchCorridorSettings,
    settings: PhysicalRibbonCorridorSetSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Optimize exact corridor alternatives jointly within and across sheets."""

    variant_row = np.asarray(variants["corridorVariantRow"], dtype=np.int32)
    variant_rank = np.asarray(variants["corridorVariantRank"], dtype=np.int16)
    eligible = np.asarray(variants["corridorVariantSurfaceEligible"]) > 0
    region_before = np.asarray(
        variants["corridorVariantTriangleRegionCountBefore"], dtype=np.int32
    )
    region_after = np.asarray(
        variants["corridorVariantTriangleRegionCountAfter"], dtype=np.int32
    )
    area_before = np.asarray(
        variants["corridorVariantTriangleAreaBefore"], dtype=np.float32
    )
    area_after = np.asarray(
        variants["corridorVariantTriangleAreaAfter"], dtype=np.float32
    )
    objective_delta = np.asarray(
        variants["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    coverage = np.asarray(
        variants["corridorVariantPatchCoverage"], dtype=np.float32
    )
    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    corridor_component = np.asarray(
        corridors["corridorTopologyComponent"], dtype=np.int32
    )
    baseline_selected = np.asarray(configuration["selected"]) > 0
    original_component = np.asarray(configuration["component"], dtype=np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source_interface = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[
        frontier
    ]
    target_interface = np.asarray(ribbon["targetInterface"], dtype=np.int32)[
        frontier
    ]
    baseline_interface_owner: dict[int, int] = {}
    for node in np.flatnonzero(baseline_selected):
        node_value = int(node)
        baseline_interface_owner[int(source_interface[node_value])] = node_value
        baseline_interface_owner[int(target_interface[node_value])] = node_value
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_neighbor: list[set[int]] = [set() for _ in range(len(frontier))]
    for left, right in zip(crossing_first, crossing_second):
        crossing_neighbor[int(left)].add(int(right))
        crossing_neighbor[int(right)].add(int(left))
    frozen_crossing_neighbor = tuple(
        frozenset(value) for value in crossing_neighbor
    )
    baseline_triangle = np.asarray(
        surface["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_triangle_area = np.asarray(
        surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    baseline_triangle_region = _triangle_region_labels(baseline_triangle)

    def modifications_valid(
        added: frozenset[int], removed: frozenset[int]
    ) -> bool:
        return _modifications_valid(
            added,
            removed,
            baseline_selected=baseline_selected,
            baseline_interface_owner=baseline_interface_owner,
            source_interface=source_interface,
            target_interface=target_interface,
            crossing_neighbor=frozen_crossing_neighbor,
        )

    eligible_by_row: dict[int, list[int]] = defaultdict(list)
    for variant_index in np.flatnonzero(eligible):
        eligible_by_row[int(variant_row[variant_index])].append(int(variant_index))
    rows_by_component: dict[int, list[int]] = defaultdict(list)
    for row in eligible_by_row:
        component_id = int(corridor_component[int(scored_corridor[row])])
        rows_by_component[component_id].append(row)

    all_states: list[dict[str, Any]] = []
    states_by_component: list[list[dict[str, Any]]] = []
    component_records: list[dict[str, Any]] = []
    reconstructed_assignment_count = 0
    rejected_assignment_count = 0
    for component_number, (component_id, rows) in enumerate(
        sorted(rows_by_component.items()), start=1
    ):
        rows.sort()
        assignment_beam: list[dict[str, Any]] = [
            {
                "variantIndices": (),
                "added": frozenset(),
                "removed": frozenset(),
                "estimate": (0.0, 0.0, 0.0, 0.0, 0.0),
            }
        ]
        for row in rows:
            best_by_signature: dict[
                tuple[frozenset[int], frozenset[int]], dict[str, Any]
            ] = {}
            options = [-1] + sorted(
                eligible_by_row[row],
                key=lambda value: (
                    int(region_before[value] - region_after[value]),
                    float(area_after[value] - area_before[value]),
                    float(objective_delta[value]),
                    float(coverage[value]),
                ),
                reverse=True,
            )
            for assignment in assignment_beam:
                for variant_index in options:
                    added = assignment["added"]
                    removed = assignment["removed"]
                    variant_indices = assignment["variantIndices"]
                    estimate = assignment["estimate"]
                    if variant_index >= 0:
                        added = added | frozenset(
                            int(value)
                            for value in _variant_values(
                                variants, variant_index, "Added"
                            )
                        )
                        removed = removed | frozenset(
                            int(value)
                            for value in _variant_values(
                                variants, variant_index, "Removed"
                            )
                        )
                        variant_indices = variant_indices + (variant_index,)
                        estimate = _sum_key(
                            estimate,
                            (
                                float(
                                    region_before[variant_index]
                                    - region_after[variant_index]
                                ),
                                float(
                                    area_after[variant_index]
                                    - area_before[variant_index]
                                ),
                                1.0,
                                float(objective_delta[variant_index]),
                                float(coverage[variant_index]),
                            ),
                        )
                    signature = (added, removed)
                    if not modifications_valid(added, removed):
                        continue
                    candidate = {
                        "variantIndices": variant_indices,
                        "added": added,
                        "removed": removed,
                        "estimate": estimate,
                    }
                    previous = best_by_signature.get(signature)
                    if previous is None or (
                        estimate,
                        variant_indices,
                    ) > (
                        previous["estimate"],
                        previous["variantIndices"],
                    ):
                        best_by_signature[signature] = candidate
            expanded = list(best_by_signature.values())
            expanded.sort(
                key=lambda value: (
                    value["estimate"], value["variantIndices"]
                ),
                reverse=True,
            )
            assignment_beam = expanded[
                : settings.component_assignment_beam_width
            ]

        component_triangle_index = np.flatnonzero(
            np.all(original_component[baseline_triangle] == component_id, axis=1)
        )
        base_region_count = len(
            np.unique(baseline_triangle_region[component_triangle_index])
        )
        base_area = float(
            np.sum(baseline_triangle_area[component_triangle_index])
        )
        feasible_states: list[dict[str, Any]] = []
        for assignment in assignment_beam:
            variant_indices = tuple(assignment["variantIndices"])
            if not variant_indices:
                key = (0.0, 0.0, 0.0, 0.0, 0.0)
                feasible = True
            elif len(variant_indices) == 1:
                variant_index = variant_indices[0]
                key = (
                    float(region_before[variant_index] - region_after[variant_index]),
                    float(area_after[variant_index] - area_before[variant_index]),
                    1.0,
                    float(objective_delta[variant_index]),
                    float(coverage[variant_index]),
                )
                feasible = key > (0.0, 0.0, 0.0, 0.0, 0.0)
            else:
                reconstructed_assignment_count += 1
                selected = baseline_selected.copy()
                if assignment["removed"]:
                    selected[
                        np.asarray(sorted(assignment["removed"]), dtype=np.int32)
                    ] = False
                if assignment["added"]:
                    selected[
                        np.asarray(sorted(assignment["added"]), dtype=np.int32)
                    ] = True
                local_selected = selected & (
                    original_component == component_id
                )
                if assignment["added"]:
                    added_array = np.asarray(
                        sorted(assignment["added"]), dtype=np.int32
                    )
                    local_selected[added_array] = selected[added_array]
                local_component, _ = _component_labels(
                    local_selected, first, second
                )
                retained = local_selected & (
                    original_component == component_id
                )
                retained_labels = np.unique(
                    local_component[retained][local_component[retained] >= 0]
                )
                feasible = len(retained_labels) == 1
                if feasible:
                    local_configuration = dict(configuration)
                    local_configuration["selected"] = local_selected.astype(
                        np.uint8
                    )
                    local_configuration["component"] = local_component
                    local_surface, _ = build_physical_ribbon_surface_complex(
                        ribbon,
                        topology,
                        local_configuration,
                        settings=corridor_settings.surface_settings(),
                    )
                    local_triangle = np.asarray(
                        local_surface["triangleFrontierIndex"], dtype=np.int32
                    )
                    local_region_count = (
                        len(np.unique(_triangle_region_labels(local_triangle)))
                        if len(local_triangle)
                        else 0
                    )
                    local_area = float(
                        np.sum(local_surface["triangleAreaVoxelsSquared"])
                    )
                    connections = _evaluate_corridor_connections(
                        local_surface,
                        corridors,
                        scored,
                        minimum_arc_region_fraction=(
                            corridor_settings.minimum_replay_arc_region_fraction
                        ),
                        maximum_arc_triangle_distance_edges=(
                            corridor_settings.maximum_replay_arc_triangle_distance_edges
                        ),
                    )
                    connected = np.asarray(
                        connections["boundaryArcsConnected"]
                    ) > 0
                    chosen_rows = [
                        int(variant_row[value]) for value in variant_indices
                    ]
                    feasible = (
                        all(connected[row] for row in chosen_rows)
                        and local_region_count <= base_region_count
                        and local_area / max(base_area, 1.0e-6)
                        >= settings.minimum_surface_area_retention
                    )
                    key = (
                        float(base_region_count - local_region_count),
                        local_area - base_area,
                        float(len(variant_indices)),
                        float(sum(objective_delta[value] for value in variant_indices)),
                        float(sum(coverage[value] for value in variant_indices)),
                    )
                    feasible &= key > (0.0, 0.0, 0.0, 0.0, 0.0)
                else:
                    key = (-1.0, -1.0, -1.0, -1.0, -1.0)
            if not feasible:
                rejected_assignment_count += 1
                continue
            feasible_states.append(
                {
                    "stateIndex": -1,
                    "componentId": component_id,
                    "variantIndices": variant_indices,
                    "added": assignment["added"],
                    "removed": assignment["removed"],
                    "key": key,
                }
            )
        feasible_states.sort(
            key=lambda value: (value["key"], value["variantIndices"]),
            reverse=True,
        )
        empty = next(
            value for value in feasible_states if not value["variantIndices"]
        )
        retained_states = feasible_states[
            : settings.maximum_retained_states_per_component
        ]
        if empty not in retained_states:
            retained_states[-1] = empty
        retained_states.sort(
            key=lambda value: (value["key"], value["variantIndices"]),
            reverse=True,
        )
        for state in retained_states:
            state["stateIndex"] = len(all_states)
            all_states.append(state)
        states_by_component.append(retained_states)
        component_records.append(
            {
                "componentId": component_id,
                "corridorRows": rows,
                "eligibleVariantCount": int(
                    sum(len(eligible_by_row[row]) for row in rows)
                ),
                "assignmentBeamCount": len(assignment_beam),
                "feasibleStateCount": len(feasible_states),
                "retainedStateIndices": [
                    int(value["stateIndex"]) for value in retained_states
                ],
                "bestStateVariantIndices": [
                    int(value) for value in retained_states[0]["variantIndices"]
                ],
                "bestStateKey": [
                    round(float(value), 6) for value in retained_states[0]["key"]
                ],
            }
        )
        if progress is not None and (
            component_number == len(rows_by_component)
            or component_number % 5 == 0
        ):
            progress(
                f"component corridor sets {component_number}/{len(rows_by_component)}"
            )

    global_state = _choose_global_variant_states(
        states_by_component,
        valid_modifications=modifications_valid,
        beam_width=settings.global_assignment_beam_width,
    )
    chosen_variant = np.full(len(scored_corridor), -1, dtype=np.int32)
    for variant_index in global_state["variantIndices"]:
        row = int(variant_row[variant_index])
        if chosen_variant[row] >= 0:
            raise RuntimeError("global solve selected two variants for one corridor")
        chosen_variant[row] = int(variant_index)

    state_variant_offset = [0]
    state_variant_value: list[int] = []
    state_key: list[tuple[float, ...]] = []
    state_component: list[int] = []
    for state in all_states:
        state_variant_value.extend(int(value) for value in state["variantIndices"])
        state_variant_offset.append(len(state_variant_value))
        state_key.append(tuple(float(value) for value in state["key"]))
        state_component.append(int(state["componentId"]))
    chosen_values = np.asarray(global_state["variantIndices"], dtype=np.int32)
    arrays = {
        "corridorChosenGlobalVariant": chosen_variant,
        "componentStateComponentId": np.asarray(state_component, dtype=np.int32),
        "componentStateVariantOffset": np.asarray(
            state_variant_offset, dtype=np.int64
        ),
        "componentStateVariantIndex": np.asarray(
            state_variant_value, dtype=np.int32
        ),
        "componentStateSelectionKey": np.asarray(state_key, dtype=np.float32),
        "globalChosenComponentStateIndex": np.asarray(
            global_state["stateIndices"], dtype=np.int32
        ),
        "globalChosenVariantIndex": chosen_values,
        "globalChosenVariantRank": variant_rank[chosen_values],
    }
    return arrays, {
        "componentCountWithExactVariants": len(states_by_component),
        "componentStateCount": len(all_states),
        "reconstructedMultiCorridorAssignmentCount": (
            reconstructed_assignment_count
        ),
        "rejectedAssignmentCount": rejected_assignment_count,
        "globallyChosenVariantCount": len(chosen_values),
        "globallyChosenCorridorRows": [
            int(variant_row[value]) for value in chosen_values
        ],
        "globallyChosenVariantRanks": [
            int(variant_rank[value]) for value in chosen_values
        ],
        "globalSelectionKey": [
            round(float(value), 6) for value in global_state["key"]
        ],
        "components": component_records,
        "decision": (
            "one exact component state per sheet under block-wide interface "
            "and crossing conflicts"
        ),
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_corridor_sets(
    variant_root: str | Path,
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorSetSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorSetSettings()
    variant_path, variant_manifest, variants = _load_variant_artifact(
        variant_root
    )
    corridor_reference = variant_manifest["identity"]["corridors"]
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_reference["manifestPath"]
    )
    if (
        sha256_file(corridor_path) != corridor_reference["manifestSha256"]
        or corridor_manifest["data"]["sha256"]
        != corridor_reference["dataSha256"]
    ):
        raise ValueError("variant artifact corridor input has changed")
    (
        configuration_path,
        configuration_manifest,
        configuration,
        _,
        _,
        topology,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_root)
    expected_configuration = variant_manifest["identity"]["configuration"]
    if (
        sha256_file(configuration_path)
        != expected_configuration["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != expected_configuration["dataSha256"]
    ):
        raise ValueError("variant artifact and configuration do not match")
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_SETS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_SETS_VERSION,
        "variants": {
            "manifestPath": str(variant_path),
            "manifestSha256": sha256_file(variant_path),
            "dataSha256": variant_manifest["data"]["sha256"],
        },
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_SETS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_SETS_STEM}.npz"
    corridor_preview_path = output / "physical-ribbon-corridor-sets.png"
    fragment_preview_path = output / "physical-ribbon-corridor-set-fragments.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256")
            == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    if progress is not None:
        progress("optimizing exact corridor variants jointly by component")
    selected_sets, set_stats = optimize_exact_corridor_variant_sets(
        corridor,
        corridor,
        corridor,
        variants,
        ribbon,
        topology,
        configuration,
        corridor_settings=corridor_settings,
        settings=resolved,
        progress=progress,
    )
    solved_at = time.monotonic()
    exact_override = dict(variants)
    exact_override["corridorChosenExactVariant"] = selected_sets[
        "corridorChosenGlobalVariant"
    ]
    compiled = compile_exact_variant_reconfiguration(
        corridor, variants, exact_override
    )
    if progress is not None:
        progress("replaying the block-level corridor assignment")
    replay, replay_stats = replay_patch_corridor_reconfigurations(
        corridor,
        corridor,
        corridor,
        compiled,
        ribbon,
        topology,
        configuration,
        settings=corridor_settings,
    )
    replayed_at = time.monotonic()
    baseline_triangle = np.asarray(
        corridor["triangleFrontierIndex"], dtype=np.int32
    )
    final_triangle = np.asarray(
        replay["corridorReplayTriangleFrontierIndex"], dtype=np.int32
    )
    surface_audit = {
        "edgeConnectedTriangleRegionCountBefore": int(
            len(np.unique(_triangle_region_labels(baseline_triangle)))
        ),
        "edgeConnectedTriangleRegionCountAfter": int(
            len(np.unique(_triangle_region_labels(final_triangle)))
        ),
        "retainedTriangleCountBefore": len(baseline_triangle),
        "retainedTriangleCountAfter": len(final_triangle),
        "closedBoundaryRegionCountBefore": int(
            corridor_manifest.get("loops", {}).get("triangleRegionCount", 0)
        ),
        "closedBoundaryRegionCountAfter": int(
            replay_stats["loops"]["triangleRegionCount"]
        ),
        "nonCycleBoundaryComponentCountBefore": int(
            corridor_manifest.get("loops", {}).get(
                "nonCycleBoundaryComponentCount", 0
            )
        ),
        "nonCycleBoundaryComponentCountAfter": int(
            replay_stats["loops"]["nonCycleBoundaryComponentCount"]
        ),
    }
    arrays = {**selected_sets, **replay}
    _write_npz(data_path, arrays)
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    write_patch_corridor_montage(
        corridor,
        corridor,
        corridor_preview_path,
        maximum_corridors=corridor_settings.maximum_preview_corridors,
        reconfiguration=compiled,
        replay=replay,
    )
    _, fragment_stats = write_replayed_corridor_fragment_montage(
        corridor,
        corridor,
        corridor,
        replay,
        source,
        fragment_preview_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_SETS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_SETS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": corridor_manifest.get("geometry", {}),
        "componentSetOptimization": set_stats,
        "counterfactualReplay": replay_stats,
        "surfaceAudit": surface_audit,
        "flattenedReplayFragments": fragment_stats,
        "timingSeconds": {
            "componentAndGlobalOptimization": round(solved_at - started, 6),
            "globalReplay": round(replayed_at - solved_at, 6),
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
            "corridorSetMontage": corridor_preview_path.name,
            "flattenedReplayFragments": fragment_preview_path.name,
        },
        "method": {
            "decisionUnit": "one exact multi-corridor state per physical-sheet component",
            "globalSelection": "block-wide hard-conflict beam over component states",
            "mutation": "counterfactual only; all source artifacts remain unchanged",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
