from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_faces import (
    PHYSICAL_RIBBON_CORRIDOR_FACES_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_FACES_STEM,
    PhysicalRibbonCorridorFaceSettings,
    _screen_corridor_face_path,
)
from .physical_ribbon_corridor_one_sided import (
    _load_frontier,
    _load_stage_inputs,
)
from .physical_ribbon_corridor_sets import (
    _modifications_valid,
    _variant_values,
)
from .physical_ribbon_corridor_variants import (
    _corridor_settings_from_manifest,
)
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    write_replayed_corridor_fragment_montage,
)
from .physical_ribbon_patch_holes import extract_surface_boundary_loops
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-face-replay"
)
PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_STEM = (
    "physical-ribbon-corridor-face-replay-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorFaceReplaySettings:
    global_assignment_beam_width: int = 4096
    maximum_preview_components: int = 8
    minimum_affected_component_area_retention: float = 0.95

    def __post_init__(self) -> None:
        if self.global_assignment_beam_width < 2:
            raise ValueError("corridor-face replay beam must retain alternatives")
        if self.maximum_preview_components < 1:
            raise ValueError("corridor-face replay must preview at least one component")
        if not 0.0 < self.minimum_affected_component_area_retention <= 1.0:
            raise ValueError("surface area retention must lie in (0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_face_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_FACES_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_FACES_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("corridor-face replay requires a complete label-free face audit")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _candidate_records(face_manifest: Mapping[str, Any]) -> list[dict[str, Any]]:
    result = [
        dict(value)
        for value in face_manifest["statistics"]["records"]
        if value.get("eligible") is True
    ]
    result.sort(key=lambda value: int(value["corridorRow"]))
    return result


def _selection_contract(
    topology: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    *,
    baseline_selected: np.ndarray | None = None,
) -> dict[str, Any]:
    if baseline_selected is None:
        baseline_selected = np.asarray(topology["selected"]) > 0
    else:
        baseline_selected = np.asarray(baseline_selected, dtype=bool)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source_interface = np.asarray(
        ribbon["sourceInterface"], dtype=np.int32
    )[frontier]
    target_interface = np.asarray(
        ribbon["targetInterface"], dtype=np.int32
    )[frontier]
    baseline_interface_owner: dict[int, int] = {}
    for node_value in np.flatnonzero(baseline_selected):
        node = int(node_value)
        baseline_interface_owner[int(source_interface[node])] = node
        baseline_interface_owner[int(target_interface[node])] = node
    crossing_neighbor: list[set[int]] = [set() for _ in range(len(frontier))]
    for first, second in zip(
        np.asarray(topology["crossingFirstFrontierIndex"], dtype=np.int32),
        np.asarray(topology["crossingSecondFrontierIndex"], dtype=np.int32),
    ):
        crossing_neighbor[int(first)].add(int(second))
        crossing_neighbor[int(second)].add(int(first))
    return {
        "baselineSelected": baseline_selected,
        "sourceInterface": source_interface,
        "targetInterface": target_interface,
        "baselineInterfaceOwner": baseline_interface_owner,
        "crossingNeighbor": tuple(
            frozenset(value) for value in crossing_neighbor
        ),
    }


def _optimize_candidate_state(
    candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, np.ndarray],
    *,
    valid_modifications: Callable[[frozenset[int], frozenset[int]], bool],
    beam_width: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Choose the most repairs, then the least CT-backed face debt."""

    beam: list[dict[str, Any]] = [
        {
            "key": (0.0, 0.0, 0.0, 0.0),
            "rows": (),
            "variantIndices": (),
            "added": frozenset(),
            "removed": frozenset(),
        }
    ]
    rejected_hard_conflicts = 0
    maximum_expanded = 1
    for candidate in candidates:
        row = int(candidate["corridorRow"])
        variant_index = int(candidate["bestFailedVariantIndex"])
        candidate_added = frozenset(
            int(value)
            for value in _variant_values(variants, variant_index, "Added")
        )
        candidate_removed = frozenset(
            int(value)
            for value in _variant_values(variants, variant_index, "Removed")
        )
        path_cost = float(candidate["physicalPathCost"])
        path_faces = float(candidate["physicalPathFaceCount"])
        shared_fraction = float(candidate["sharedArcRegionFraction"])
        best_by_signature: dict[
            tuple[frozenset[int], frozenset[int]], dict[str, Any]
        ] = {}
        for state in beam:
            for take in (False, True):
                added = state["added"]
                removed = state["removed"]
                rows = state["rows"]
                variant_indices = state["variantIndices"]
                key = state["key"]
                if take:
                    added = added | candidate_added
                    removed = removed | candidate_removed
                    if not valid_modifications(added, removed):
                        rejected_hard_conflicts += 1
                        continue
                    rows = rows + (row,)
                    variant_indices = variant_indices + (variant_index,)
                    key = (
                        key[0] + 1.0,
                        key[1] - path_cost,
                        key[2] + shared_fraction,
                        key[3] - path_faces,
                    )
                signature = (added, removed)
                value = {
                    "key": key,
                    "rows": rows,
                    "variantIndices": variant_indices,
                    "added": added,
                    "removed": removed,
                }
                previous = best_by_signature.get(signature)
                if previous is None or (
                    value["key"], value["rows"], value["variantIndices"]
                ) > (
                    previous["key"],
                    previous["rows"],
                    previous["variantIndices"],
                ):
                    best_by_signature[signature] = value
        expanded = list(best_by_signature.values())
        expanded.sort(
            key=lambda value: (
                value["key"], value["rows"], value["variantIndices"]
            ),
            reverse=True,
        )
        maximum_expanded = max(maximum_expanded, len(expanded))
        beam = expanded[:beam_width]
        if not beam:
            raise RuntimeError("corridor-face assignment beam became empty")
    chosen = beam[0]
    return chosen, {
        "candidateCount": len(candidates),
        "chosenCount": len(chosen["rows"]),
        "chosenCorridorRows": [int(value) for value in chosen["rows"]],
        "chosenVariantIndices": [
            int(value) for value in chosen["variantIndices"]
        ],
        "selectionKey": [round(float(value), 6) for value in chosen["key"]],
        "rejectedHardConflictTransitions": rejected_hard_conflicts,
        "maximumExpandedBeamStates": maximum_expanded,
        "finalBeamStates": len(beam),
        "decision": (
            "maximize exact CT corridor count, then minimize supplemental "
            "face cost and face count under interface and crossing conflicts"
        ),
        "identityLabelsUsed": False,
    }


def _hard_conflict_counts(
    selected: np.ndarray,
    topology: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
) -> tuple[int, int]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    interface = np.concatenate((source[selected], target[selected]))
    _, count = np.unique(interface, return_counts=True)
    interface_conflict = int(np.sum(np.maximum(count - 1, 0)))
    first = np.asarray(
        topology["crossingFirstFrontierIndex"], dtype=np.int32
    )
    second = np.asarray(
        topology["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_conflict = int(np.count_nonzero(selected[first] & selected[second]))
    return interface_conflict, crossing_conflict


def _component_inheritance_audit(
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    final_selected: np.ndarray,
    final_component: np.ndarray,
) -> dict[str, Any]:
    split_components: list[int] = []
    for component_id in np.unique(baseline_component[baseline_selected]):
        if component_id < 0:
            continue
        retained = (
            baseline_selected
            & final_selected
            & (baseline_component == component_id)
        )
        inherited = np.unique(final_component[retained])
        inherited = inherited[inherited >= 0]
        if len(inherited) > 1:
            split_components.append(int(component_id))
    fused_components: list[dict[str, Any]] = []
    for component_id in np.unique(final_component[final_selected]):
        inherited = np.unique(
            baseline_component[
                baseline_selected
                & final_selected
                & (final_component == component_id)
            ]
        )
        inherited = inherited[inherited >= 0]
        if len(inherited) > 1:
            fused_components.append(
                {
                    "finalComponent": int(component_id),
                    "priorComponents": [int(value) for value in inherited],
                }
            )
    return {
        "splitPriorComponentCount": len(split_components),
        "splitPriorComponents": split_components,
        "crossPriorComponentFusionCount": len(fused_components),
        "crossPriorComponentFusions": fused_components,
    }


def _affected_component_area_audit(
    baseline_surface: Mapping[str, np.ndarray],
    final_surface: Mapping[str, np.ndarray],
    affected_prior_components: Sequence[int],
) -> tuple[list[dict[str, Any]], float]:
    baseline_triangle = np.asarray(
        baseline_surface["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_area = np.asarray(
        baseline_surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    baseline_component = np.asarray(
        baseline_surface["component"], dtype=np.int32
    )
    final_triangle = np.asarray(
        final_surface["triangleFrontierIndex"], dtype=np.int32
    )
    final_area = np.asarray(
        final_surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    final_component = np.asarray(final_surface["component"], dtype=np.int32)
    records: list[dict[str, Any]] = []
    minimum_retention = math.inf
    for prior_component in sorted(set(int(value) for value in affected_prior_components)):
        baseline_mask = np.all(
            baseline_component[baseline_triangle] == prior_component, axis=1
        )
        inherited = final_component[
            (baseline_component == prior_component) & (final_component >= 0)
        ]
        if len(inherited):
            value, count = np.unique(inherited, return_counts=True)
            final_component_id = int(value[int(np.argmax(count))])
            final_mask = np.all(
                final_component[final_triangle] == final_component_id, axis=1
            )
        else:
            final_component_id = -1
            final_mask = np.zeros(len(final_triangle), dtype=bool)
        before = float(np.sum(baseline_area[baseline_mask]))
        after = float(np.sum(final_area[final_mask]))
        retention = after / max(before, 1.0e-6)
        minimum_retention = min(minimum_retention, retention)
        records.append(
            {
                "priorComponent": prior_component,
                "finalComponent": final_component_id,
                "triangleCountBefore": int(np.count_nonzero(baseline_mask)),
                "triangleCountAfterBeforeSupplementalFaces": int(
                    np.count_nonzero(final_mask)
                ),
                "areaBefore": round(before, 6),
                "areaAfterBeforeSupplementalFaces": round(after, 6),
                "areaRetention": round(retention, 6),
            }
        )
    return records, minimum_retention if records else 1.0


def _supplemental_face_arrays(
    chosen_rows: Sequence[int],
    surface: Mapping[str, np.ndarray],
    corridor: Mapping[str, np.ndarray],
    *,
    surface_settings: Any,
    face_settings: PhysicalRibbonCorridorFaceSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]]]:
    by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    records: list[dict[str, Any]] = []
    for row in chosen_rows:
        record, arrays = _screen_corridor_face_path(
            int(row),
            surface,
            corridor,
            surface_settings=surface_settings,
            settings=face_settings,
        )
        records.append(record)
        closure = np.flatnonzero(arrays["candidatePhysicalClosure"] > 0)
        for candidate_index in closure:
            triangle = np.asarray(
                arrays["candidateTriangleFrontierIndex"][candidate_index],
                dtype=np.int32,
            )
            key = tuple(sorted(int(value) for value in triangle))
            value = {
                "triangle": triangle,
                "corridorRows": {int(row)},
                "minimumPath": bool(
                    arrays["candidatePathSelected"][candidate_index]
                ),
                "area": float(
                    arrays["candidateAreaVoxelsSquared"][candidate_index]
                ),
                "nodeNormalResidual": float(
                    arrays["candidateNodeNormalResidualDegrees"][candidate_index]
                ),
                "ctNormalResidual": float(
                    arrays["candidateCtNormalResidualDegrees"][candidate_index]
                ),
                "distanceThicknesses": float(
                    arrays["candidateCenterDistanceThicknesses"][candidate_index]
                ),
                "heightThicknesses": float(
                    arrays["candidateCenterHeightThicknesses"][candidate_index]
                ),
                "edgeThicknesses": float(
                    arrays["candidateMaximumEdgeThicknesses"][candidate_index]
                ),
            }
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = value
            else:
                previous["corridorRows"].add(int(row))
                previous["minimumPath"] = bool(
                    previous["minimumPath"] or value["minimumPath"]
                )
    ordered = [by_key[key] for key in sorted(by_key)]
    triangle = np.asarray(
        [value["triangle"] for value in ordered], dtype=np.int32
    ).reshape((-1, 3))
    return {
        "supplementalTriangleFrontierIndex": triangle,
        "supplementalTrianglePrimaryCorridorRow": np.asarray(
            [min(value["corridorRows"]) for value in ordered], dtype=np.int32
        ),
        "supplementalTriangleMinimumPath": np.asarray(
            [value["minimumPath"] for value in ordered], dtype=np.uint8
        ),
        "supplementalTriangleAreaVoxelsSquared": np.asarray(
            [value["area"] for value in ordered], dtype=np.float32
        ),
        "supplementalTriangleNodeNormalResidualDegrees": np.asarray(
            [value["nodeNormalResidual"] for value in ordered], dtype=np.float32
        ),
        "supplementalTriangleCtNormalResidualDegrees": np.asarray(
            [value["ctNormalResidual"] for value in ordered], dtype=np.float32
        ),
        "supplementalTriangleCenterDistanceThicknesses": np.asarray(
            [value["distanceThicknesses"] for value in ordered], dtype=np.float32
        ),
        "supplementalTriangleCenterHeightThicknesses": np.asarray(
            [value["heightThicknesses"] for value in ordered], dtype=np.float32
        ),
        "supplementalTriangleMaximumEdgeThicknesses": np.asarray(
            [value["edgeThicknesses"] for value in ordered], dtype=np.float32
        ),
    }, records


def _edge_manifold_audit(triangles: np.ndarray) -> dict[str, int]:
    edge_count: Counter[tuple[int, int]] = Counter()
    for triangle in triangles:
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge_count[(min(int(first), second), max(int(first), second))] += 1
    return {
        "edgeCount": len(edge_count),
        "boundaryEdgeCount": sum(value == 1 for value in edge_count.values()),
        "interiorEdgeCount": sum(value == 2 for value in edge_count.values()),
        "nonManifoldEdgeCount": sum(value > 2 for value in edge_count.values()),
        "maximumTriangleIncidencePerEdge": max(edge_count.values(), default=0),
    }


def run_physical_ribbon_corridor_face_replay(
    face_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorFaceReplaySettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorFaceReplaySettings()
    face_path, face_manifest, _ = _load_face_artifact(face_root)
    if (
        face_manifest["identity"].get("implementationSha256")
        != sha256_file(Path(__file__).with_name("physical_ribbon_corridor_faces.py"))
    ):
        raise ValueError("corridor-face audit was built by a different implementation")
    replay_reference = face_manifest["identity"]["replay"]
    replay_path, replay_manifest, variants = _load_replay_artifact(
        replay_reference["manifestPath"]
    )
    if (
        sha256_file(replay_path) != replay_reference["manifestSha256"]
        or replay_manifest["data"]["sha256"] != replay_reference["dataSha256"]
    ):
        raise ValueError("corridor-face exact replay has changed")
    frontier_path, frontier_manifest, topology = _load_frontier(
        replay_manifest["identity"]["frontier"]["manifestPath"]
    )
    (
        corridor_path,
        corridor_manifest,
        corridor,
        _,
        _,
        _,
        configuration_path,
        configuration_manifest,
        base_configuration,
        ribbon,
    ) = _load_stage_inputs(frontier_manifest)
    face_settings = PhysicalRibbonCorridorFaceSettings(
        **face_manifest["identity"]["settings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_VERSION,
        "faces": {
            "manifestPath": str(face_path),
            "manifestSha256": sha256_file(face_path),
            "dataSha256": face_manifest["data"]["sha256"],
        },
        "replay": replay_reference,
        "frontier": {
            "manifestPath": str(frontier_path),
            "manifestSha256": sha256_file(frontier_path),
            "dataSha256": frontier_manifest["data"]["sha256"],
        },
        "corridors": {
            "manifestPath": str(corridor_path),
            "manifestSha256": sha256_file(corridor_path),
            "dataSha256": corridor_manifest["data"]["sha256"],
        },
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "faceSettings": face_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "faceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_faces.py")
        ),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_STEM}.npz"
    preview_path = output / "physical-ribbon-corridor-face-fragments.png"
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
    candidates = _candidate_records(face_manifest)
    exact_replay_selected = np.asarray(
        variants["corridorReplaySelected"], dtype=np.uint8
    ) > 0
    exact_replay_component = np.asarray(
        variants["corridorReplayComponent"], dtype=np.int32
    )
    contract = _selection_contract(
        topology,
        ribbon,
        baseline_selected=exact_replay_selected,
    )

    def valid_modifications(
        added: frozenset[int], removed: frozenset[int]
    ) -> bool:
        return _modifications_valid(
            added,
            removed,
            baseline_selected=contract["baselineSelected"],
            baseline_interface_owner=contract["baselineInterfaceOwner"],
            source_interface=contract["sourceInterface"],
            target_interface=contract["targetInterface"],
            crossing_neighbor=contract["crossingNeighbor"],
        )

    if progress is not None:
        progress("optimizing compatible CT-backed face repairs")
    chosen, optimization_stats = _optimize_candidate_state(
        candidates,
        variants,
        valid_modifications=valid_modifications,
        beam_width=resolved.global_assignment_beam_width,
    )
    optimized_at = time.monotonic()
    selected = contract["baselineSelected"].copy()
    unexpectedly_selected_additions = (
        int(
            np.count_nonzero(
                selected[np.asarray(sorted(chosen["added"]), dtype=np.int32)]
            )
        )
        if chosen["added"]
        else 0
    )
    unexpectedly_missing_removals = (
        int(
            np.count_nonzero(
                ~selected[
                    np.asarray(sorted(chosen["removed"]), dtype=np.int32)
                ]
            )
        )
        if chosen["removed"]
        else 0
    )
    if unexpectedly_selected_additions or unexpectedly_missing_removals:
        raise RuntimeError(
            "residual corridor deltas overlap the prior cumulative exact replay"
        )
    if chosen["removed"]:
        selected[np.asarray(sorted(chosen["removed"]), dtype=np.int32)] = False
    if chosen["added"]:
        selected[np.asarray(sorted(chosen["added"]), dtype=np.int32)] = True
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    component, component_size = _component_labels(selected, first, second)
    interface_conflicts, crossing_conflicts = _hard_conflict_counts(
        selected, topology, ribbon
    )
    inheritance = _component_inheritance_audit(
        contract["baselineSelected"],
        exact_replay_component,
        selected,
        component,
    )
    if (
        interface_conflicts
        or crossing_conflicts
        or inheritance["splitPriorComponentCount"]
        or inheritance["crossPriorComponentFusionCount"]
    ):
        raise RuntimeError("optimized corridor-face state violates strict topology")
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    surface_settings = corridor_settings.surface_settings()
    if progress is not None:
        progress("rebuilding the cumulative strict surface once")
    cumulative_baseline_configuration = dict(topology)
    cumulative_baseline_configuration["selected"] = (
        exact_replay_selected.astype(np.uint8)
    )
    cumulative_baseline_configuration["component"] = exact_replay_component
    final_configuration = dict(topology)
    final_configuration["selected"] = selected.astype(np.uint8)
    final_configuration["component"] = component
    final_surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        topology,
        final_configuration,
        settings=surface_settings,
    )
    baseline_surface, baseline_surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        topology,
        cumulative_baseline_configuration,
        settings=surface_settings,
    )
    corridor_reference_surface, corridor_reference_surface_stats = (
        build_physical_ribbon_surface_complex(
            ribbon,
            topology,
            topology,
            settings=surface_settings,
        )
    )
    remapped = _remap_corridor_surface(
        corridor,
        corridor_reference_surface,
        base_configuration,
        topology,
        np.asarray(topology["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    rebuilt_at = time.monotonic()
    if progress is not None:
        progress("recomputing physical face paths in the cumulative charts")
    supplemental, path_records = _supplemental_face_arrays(
        chosen["rows"],
        final_surface,
        remapped,
        surface_settings=surface_settings,
        face_settings=face_settings,
    )
    failed_rows = [
        int(value["corridorRow"])
        for value in path_records
        if value.get("eligible") is not True
    ]
    if failed_rows:
        raise RuntimeError(
            f"cumulative corridor charts invalidated face paths for rows {failed_rows}"
        )
    base_triangle = np.asarray(
        final_surface["triangleFrontierIndex"], dtype=np.int32
    )
    supplemental_triangle = supplemental["supplementalTriangleFrontierIndex"]
    augmented_triangle = np.vstack(
        (base_triangle, supplemental_triangle)
    ).astype(np.int32)
    augmented_area = np.concatenate(
        (
            np.asarray(
                final_surface["triangleAreaVoxelsSquared"], dtype=np.float32
            ),
            supplemental["supplementalTriangleAreaVoxelsSquared"],
        )
    )
    augmented_normal_residual = np.concatenate(
        (
            np.asarray(
                final_surface["triangleNormalResidualDegrees"],
                dtype=np.float32,
            ),
            supplemental["supplementalTriangleNodeNormalResidualDegrees"],
        )
    )
    augmented_surface = dict(final_surface)
    augmented_surface["triangleFrontierIndex"] = augmented_triangle
    augmented_surface["triangleAreaVoxelsSquared"] = augmented_area
    augmented_surface["triangleNormalResidualDegrees"] = (
        augmented_normal_residual
    )
    connection = _evaluate_corridor_connections(
        augmented_surface,
        remapped,
        remapped,
        minimum_arc_region_fraction=face_settings.minimum_arc_region_fraction,
        maximum_arc_triangle_distance_edges=(
            face_settings.maximum_arc_triangle_distance_edges
        ),
    )
    disconnected = [
        int(row)
        for row in chosen["rows"]
        if not connection["boundaryArcsConnected"][row]
    ]
    if disconnected:
        raise RuntimeError(
            f"supplemental face union failed corridor rows {disconnected}"
        )
    manifold = _edge_manifold_audit(augmented_triangle)
    if manifold["nonManifoldEdgeCount"]:
        raise RuntimeError("supplemental corridor faces created a non-manifold edge")
    loops, loop_stats = extract_surface_boundary_loops(
        augmented_surface, settings=surface_settings
    )
    affected_components = [
        int(value["topologyComponent"])
        for value in candidates
        if int(value["corridorRow"]) in set(chosen["rows"])
    ]
    component_area, minimum_area_retention = _affected_component_area_audit(
        corridor_reference_surface, final_surface, affected_components
    )
    if minimum_area_retention < resolved.minimum_affected_component_area_retention:
        raise RuntimeError(
            "cumulative face replay regressed an affected component's surface area"
        )
    repaired_at = time.monotonic()

    chosen_rows = np.asarray(chosen["rows"], dtype=np.int32)
    chosen_variants = np.asarray(chosen["variantIndices"], dtype=np.int32)
    triangle_supplemental = np.concatenate(
        (
            np.zeros(len(base_triangle), dtype=np.uint8),
            np.ones(len(supplemental_triangle), dtype=np.uint8),
        )
    )
    triangle_ct_normal = np.concatenate(
        (
            np.full(len(base_triangle), np.nan, dtype=np.float32),
            supplemental["supplementalTriangleCtNormalResidualDegrees"],
        )
    )
    triangle_minimum_path = np.concatenate(
        (
            np.zeros(len(base_triangle), dtype=np.uint8),
            supplemental["supplementalTriangleMinimumPath"],
        )
    )
    arrays: dict[str, np.ndarray] = {
        **{
            key: np.asarray(value)
            for key, value in final_surface.items()
            if key
            not in {
                "triangleFrontierIndex",
                "triangleAreaVoxelsSquared",
                "triangleNormalResidualDegrees",
            }
        },
        "selected": selected.astype(np.uint8),
        "component": component,
        "triangleFrontierIndex": augmented_triangle,
        "triangleAreaVoxelsSquared": augmented_area,
        "triangleNormalResidualDegrees": augmented_normal_residual,
        "triangleSupplementalCtFace": triangle_supplemental,
        "triangleMinimumCorridorPathFace": triangle_minimum_path,
        "triangleCtNormalResidualDegrees": triangle_ct_normal,
        "baseStrictTriangleCount": np.asarray([len(base_triangle)], dtype=np.int64),
        "chosenCorridorRow": chosen_rows,
        "chosenVariantIndex": chosen_variants,
        "triangleRegion": _triangle_region_labels(augmented_triangle),
        "loopOffset": np.asarray(loops["loopOffset"], dtype=np.int64),
        "loopVertexFrontierIndex": np.asarray(
            loops["loopVertexFrontierIndex"], dtype=np.int32
        ),
        "loopKind": np.asarray(loops["loopKind"], dtype=np.uint8),
        "loopTriangleRegion": np.asarray(
            loops["loopTriangleRegion"], dtype=np.int32
        ),
        **supplemental,
    }
    _write_npz(data_path, arrays)

    replay_preview = {
        "corridorReplayProposalSuccessful": np.isin(
            np.arange(len(remapped["scoredCorridorIndex"])), chosen_rows
        ).astype(np.uint8),
        "corridorReplayComponent": component,
        "corridorReplayChartUV": np.asarray(
            final_surface["chartUV"], dtype=np.float32
        ),
        "corridorReplayTriangleFrontierIndex": augmented_triangle,
    }
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    if progress is not None:
        progress("flattening repaired components from native CT")
    _, preview_stats = write_replayed_corridor_fragment_montage(
        remapped,
        remapped,
        remapped,
        replay_preview,
        source,
        preview_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "optimization": optimization_stats,
        "strictTopology": {
            "selectedRibbonCountBefore": int(
                np.count_nonzero(contract["baselineSelected"])
            ),
            "priorExactReplayAcceptedCorridorCount": int(
                np.count_nonzero(
                    np.asarray(
                        variants["corridorChosenExactVariant"], dtype=np.int32
                    )
                    >= 0
                )
            ),
            "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
            "addedRibbonCount": len(chosen["added"]),
            "removedRibbonCount": len(chosen["removed"]),
            "unexpectedlySelectedAdditionCount": unexpectedly_selected_additions,
            "unexpectedlyMissingRemovalCount": unexpectedly_missing_removals,
            "componentCountAfter": len(component_size),
            "interfaceConflictCount": interface_conflicts,
            "crossingConflictCount": crossing_conflicts,
            **inheritance,
        },
        "surface": {
            "corridorReferenceBeforePriorExactReplay": (
                corridor_reference_surface_stats
            ),
            "cumulativeBaseline": baseline_surface_stats,
            "cumulativeBeforeSupplementalFaces": surface_stats,
            "strictTriangleCount": len(base_triangle),
            "minimumPathCtFaceCount": int(
                np.count_nonzero(
                    supplemental["supplementalTriangleMinimumPath"]
                )
            ),
            "supplementalCtFaceCount": len(supplemental_triangle),
            "augmentedTriangleCount": len(augmented_triangle),
            "triangleRegionCountBeforeSupplementalFaces": int(
                len(np.unique(_triangle_region_labels(base_triangle)))
            ),
            "triangleRegionCountAfterSupplementalFaces": int(
                len(np.unique(_triangle_region_labels(augmented_triangle)))
            ),
            "chosenCorridorConnectionCount": int(
                np.count_nonzero(connection["boundaryArcsConnected"][chosen_rows])
            ),
            "affectedComponents": component_area,
            "minimumAffectedComponentAreaRetention": round(
                float(minimum_area_retention), 6
            ),
            "manifold": manifold,
            "loops": loop_stats,
            "pathRecords": path_records,
        },
        "flattenedFragments": preview_stats,
        "timingSeconds": {
            "optimization": round(optimized_at - started, 6),
            "surfaceRebuilds": round(rebuilt_at - optimized_at, 6),
            "cumulativeFaceRepair": round(repaired_at - rebuilt_at, 6),
            "writingAndPreview": round(finished - repaired_at, 6),
            "total": round(finished - started, 6),
        },
        "method": {
            "sheetIdentity": "strict ribbon graph only",
            "surfaceConnectivity": (
                "native-CT-gated chart-Delaunay faces stored separately from "
                "strict topology edges"
            ),
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "flattenedFragments": preview_path.name,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
