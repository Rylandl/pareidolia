from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_complete_strips import (
    PHYSICAL_RIBBON_COMPLETE_STRIPS_SCHEMA,
    PHYSICAL_RIBBON_COMPLETE_STRIPS_STEM,
    _load_face_replay_artifact,
    _strict_surface,
)
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_face_replay import (
    _affected_component_area_audit,
    _component_inheritance_audit,
    _edge_manifold_audit,
    _hard_conflict_counts,
    _selection_contract,
    _supplemental_face_arrays,
)
from .physical_ribbon_corridor_faces import PhysicalRibbonCorridorFaceSettings
from .physical_ribbon_corridor_one_sided import (
    _load_frontier,
    _load_stage_inputs,
)
from .physical_ribbon_corridor_sets import (
    _modifications_valid,
    _variant_values,
)
from .physical_ribbon_corridor_variants import _corridor_settings_from_manifest
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    write_replayed_corridor_fragment_montage,
)
from .physical_ribbon_patch_holes import extract_surface_boundary_loops
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_SCHEMA = (
    "pareidolia.physical-ribbon-complete-strip-replay"
)
PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_VERSION = 1
PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_STEM = (
    "physical-ribbon-complete-strip-replay-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCompleteStripReplaySettings:
    global_assignment_beam_width: int = 4096
    maximum_exact_states: int = 16
    maximum_preview_components: int = 8
    minimum_affected_component_area_retention: float = 0.95

    def __post_init__(self) -> None:
        if self.global_assignment_beam_width < 2:
            raise ValueError("complete-strip replay beam must retain alternatives")
        if self.maximum_exact_states < 1:
            raise ValueError("complete-strip replay must test at least one state")
        if self.maximum_preview_components < 1:
            raise ValueError("complete-strip replay must preview at least one sheet")
        if not 0.0 < self.minimum_affected_component_area_retention <= 1.0:
            raise ValueError("surface-area retention must lie in (0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_complete_strip_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_COMPLETE_STRIPS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_COMPLETE_STRIPS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "complete-strip replay requires a complete label-free strip audit"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return manifest_path, manifest, _load_npz(
        data_path, manifest["data"]["sha256"]
    )


def _eligible_candidates(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    result = [
        dict(value)
        for value in manifest["screen"]["variants"]
        if value.get("eligible") is True
    ]
    result.sort(
        key=lambda value: (
            int(value["corridorRow"]),
            int(value["variantRank"]),
            int(value["variantIndex"]),
        )
    )
    return result


def _candidate_delta(
    candidate: Mapping[str, Any],
    variants: Mapping[str, np.ndarray],
) -> tuple[frozenset[int], frozenset[int]]:
    index = int(candidate["variantIndex"])
    return (
        frozenset(int(value) for value in _variant_values(variants, index, "Added")),
        frozenset(
            int(value) for value in _variant_values(variants, index, "Removed")
        ),
    )


def _candidate_increment(candidate: Mapping[str, Any]) -> tuple[float, ...]:
    return (
        1.0,
        float(candidate["triangleRegionCountBefore"])
        - float(candidate["triangleRegionCountAfter"]),
        float(candidate["strictAreaRetention"]) - 1.0,
        float(candidate["patchCoverage"]),
        float(
            candidate.get(
                "augmentedAreaRetention", candidate["strictAreaRetention"]
            )
        )
        - 1.0,
        -float(candidate["physicalPathFaceCount"]),
        -float(candidate["physicalPathCost"] or 0.0),
        float(candidate["localObjectiveDelta"]),
    )


def _grouped_candidate_states(
    candidates: Sequence[Mapping[str, Any]],
    variants: Mapping[str, np.ndarray],
    *,
    valid_modifications: Callable[[frozenset[int], frozenset[int]], bool],
    beam_width: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Choose at most one complete matching per CT strip."""

    grouped: dict[int, list[Mapping[str, Any]]] = {}
    for candidate in candidates:
        grouped.setdefault(int(candidate["corridorRow"]), []).append(candidate)
    beam: list[dict[str, Any]] = [
        {
            "key": (0.0,) * 8,
            "rows": (),
            "variantIndices": (),
            "added": frozenset(),
            "removed": frozenset(),
        }
    ]
    rejected = 0
    maximum_expanded = 1
    for row in sorted(grouped):
        options: list[Mapping[str, Any] | None] = [None, *grouped[row]]
        by_signature: dict[
            tuple[frozenset[int], frozenset[int]], dict[str, Any]
        ] = {}
        for state in beam:
            for candidate in options:
                added = state["added"]
                removed = state["removed"]
                rows = state["rows"]
                indices = state["variantIndices"]
                key = state["key"]
                if candidate is not None:
                    candidate_added, candidate_removed = _candidate_delta(
                        candidate, variants
                    )
                    added = added | candidate_added
                    removed = removed | candidate_removed
                    if not valid_modifications(added, removed):
                        rejected += 1
                        continue
                    rows = rows + (row,)
                    indices = indices + (int(candidate["variantIndex"]),)
                    increment = _candidate_increment(candidate)
                    key = tuple(
                        float(first + second)
                        for first, second in zip(key, increment)
                    )
                signature = (added, removed)
                value = {
                    "key": key,
                    "rows": rows,
                    "variantIndices": indices,
                    "added": added,
                    "removed": removed,
                }
                previous = by_signature.get(signature)
                if previous is None or (
                    value["key"], value["rows"], value["variantIndices"]
                ) > (
                    previous["key"],
                    previous["rows"],
                    previous["variantIndices"],
                ):
                    by_signature[signature] = value
        expanded = list(by_signature.values())
        expanded.sort(
            key=lambda value: (
                value["key"], value["rows"], value["variantIndices"]
            ),
            reverse=True,
        )
        maximum_expanded = max(maximum_expanded, len(expanded))
        beam = expanded[:beam_width]
        if not beam:
            raise RuntimeError("complete-strip assignment beam became empty")
    return beam, {
        "candidateCount": len(candidates),
        "corridorCount": len(grouped),
        "candidateCountByCorridor": {
            str(row): len(grouped[row]) for row in sorted(grouped)
        },
        "rejectedHardConflictTransitions": rejected,
        "maximumExpandedBeamStates": maximum_expanded,
        "finalBeamStates": len(beam),
        "decision": (
            "maximize recovered complete CT strips, region reduction, strict "
            "area retention, whole-strip coverage, and augmented area; then "
            "minimize CT-face debt under one assignment per strip and exact "
            "physical conflicts"
        ),
        "identityLabelsUsed": False,
    }


def _apply_state(
    state: Mapping[str, Any], baseline_selected: np.ndarray
) -> np.ndarray:
    selected = np.asarray(baseline_selected, dtype=bool).copy()
    if state["removed"]:
        selected[np.asarray(sorted(state["removed"]), dtype=np.int32)] = False
    if state["added"]:
        selected[np.asarray(sorted(state["added"]), dtype=np.int32)] = True
    return selected


def run_physical_ribbon_complete_strip_replay(
    strips_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCompleteStripReplaySettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCompleteStripReplaySettings()
    strips_path, strips_manifest, strips = _load_complete_strip_artifact(
        strips_root
    )
    replay_reference = strips_manifest["identity"]["replay"]
    replay_path, replay_manifest, replay = _load_face_replay_artifact(
        replay_reference["manifestPath"]
    )
    if (
        sha256_file(replay_path) != replay_reference["manifestSha256"]
        or replay_manifest["data"]["sha256"] != replay_reference["dataSha256"]
    ):
        raise ValueError("complete-strip source replay has changed")
    prior_reference = replay_manifest["identity"]["replay"]
    _, _, prior_exact = _load_replay_artifact(prior_reference["manifestPath"])
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
        **strips_manifest["identity"]["faceSettings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_VERSION,
        "strips": {
            "manifestPath": str(strips_path),
            "manifestSha256": sha256_file(strips_path),
            "dataSha256": strips_manifest["data"]["sha256"],
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
        "stripImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_complete_strips.py")
        ),
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
    manifest_path = output / f"{PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_STEM}.npz"
    preview_path = output / "physical-ribbon-complete-strip-fragments.png"
    new_preview_path = output / "physical-ribbon-complete-strip-new-fragments.png"
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
    candidates = _eligible_candidates(strips_manifest)
    baseline_selected = np.asarray(replay["selected"]) > 0
    baseline_component = np.asarray(replay["component"], dtype=np.int32)
    contract = _selection_contract(
        topology, ribbon, baseline_selected=baseline_selected
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

    states, optimization_stats = _grouped_candidate_states(
        candidates,
        strips,
        valid_modifications=valid_modifications,
        beam_width=resolved.global_assignment_beam_width,
    )
    optimized_at = time.monotonic()
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    surface_settings = corridor_settings.surface_settings()
    if progress is not None:
        progress("building the original corridor reference surface once")
    reference_surface, reference_surface_stats = (
        build_physical_ribbon_surface_complex(
            ribbon, topology, topology, settings=surface_settings
        )
    )
    remapped = _remap_corridor_surface(
        corridor,
        reference_surface,
        base_configuration,
        topology,
        np.asarray(topology["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    referenced_at = time.monotonic()
    prior_face_rows = tuple(
        int(value)
        for value in replay_manifest["optimization"]["chosenCorridorRows"]
    )
    prior_exact_rows = tuple(
        int(value)
        for value in np.flatnonzero(
            np.asarray(prior_exact["corridorReplayProposalSuccessful"]) > 0
        )
    )
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    baseline_surface = _strict_surface(replay)
    chosen_result: dict[str, Any] | None = None
    exact_attempts: list[dict[str, Any]] = []
    for attempt, state in enumerate(states[: resolved.maximum_exact_states], start=1):
        selected = _apply_state(state, baseline_selected)
        component, component_size = _component_labels(selected, first, second)
        interface_conflict, crossing_conflict = _hard_conflict_counts(
            selected, topology, ribbon
        )
        inheritance = _component_inheritance_audit(
            baseline_selected,
            baseline_component,
            selected,
            component,
        )
        topology_valid = not (
            interface_conflict
            or crossing_conflict
            or inheritance["splitPriorComponentCount"]
            or inheritance["crossPriorComponentFusionCount"]
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "corridorRows": [int(value) for value in state["rows"]],
            "variantIndices": [int(value) for value in state["variantIndices"]],
            "selectionKey": [round(float(value), 6) for value in state["key"]],
            "topologyValid": topology_valid,
            "surfaceValid": False,
        }
        if not topology_valid:
            exact_attempts.append(attempt_record)
            continue
        configuration = dict(topology)
        configuration["selected"] = selected.astype(np.uint8)
        configuration["component"] = component
        if progress is not None:
            progress(
                f"exact complete-strip state {attempt}/{min(len(states), resolved.maximum_exact_states)} "
                f"· rows {list(state['rows'])}"
            )
        strict_surface, strict_stats = build_physical_ribbon_surface_complex(
            ribbon, topology, configuration, settings=surface_settings
        )
        required_face_rows = tuple(sorted(set(prior_face_rows) | set(state["rows"])))
        supplemental, path_records = _supplemental_face_arrays(
            required_face_rows,
            strict_surface,
            remapped,
            surface_settings=surface_settings,
            face_settings=face_settings,
        )
        failed_paths = [
            int(value["corridorRow"])
            for value in path_records
            if value.get("eligible") is not True
        ]
        base_triangle = np.asarray(
            strict_surface["triangleFrontierIndex"], dtype=np.int32
        )
        supplemental_triangle = supplemental[
            "supplementalTriangleFrontierIndex"
        ]
        augmented_triangle = np.vstack(
            (base_triangle, supplemental_triangle)
        ).astype(np.int32)
        augmented_surface = dict(strict_surface)
        augmented_surface["triangleFrontierIndex"] = augmented_triangle
        augmented_surface["triangleAreaVoxelsSquared"] = np.concatenate(
            (
                np.asarray(
                    strict_surface["triangleAreaVoxelsSquared"], dtype=np.float32
                ),
                supplemental["supplementalTriangleAreaVoxelsSquared"],
            )
        )
        augmented_surface["triangleNormalResidualDegrees"] = np.concatenate(
            (
                np.asarray(
                    strict_surface["triangleNormalResidualDegrees"],
                    dtype=np.float32,
                ),
                supplemental["supplementalTriangleNodeNormalResidualDegrees"],
            )
        )
        connections = _evaluate_corridor_connections(
            augmented_surface,
            remapped,
            remapped,
            minimum_arc_region_fraction=face_settings.minimum_arc_region_fraction,
            maximum_arc_triangle_distance_edges=(
                face_settings.maximum_arc_triangle_distance_edges
            ),
        )
        required_rows = tuple(
            sorted(set(required_face_rows) | set(prior_exact_rows))
        )
        disconnected = [
            int(row)
            for row in required_rows
            if not connections["boundaryArcsConnected"][row]
        ]
        manifold = _edge_manifold_audit(augmented_triangle)
        affected_components = [
            int(remapped["corridorTopologyComponent"][
                int(remapped["scoredCorridorIndex"][row])
            ])
            for row in state["rows"]
        ]
        component_area, minimum_retention = _affected_component_area_audit(
            baseline_surface, strict_surface, affected_components
        )
        surface_valid = not (
            failed_paths
            or disconnected
            or manifold["nonManifoldEdgeCount"]
            or minimum_retention
            < resolved.minimum_affected_component_area_retention
        )
        attempt_record.update(
            {
                "surfaceValid": surface_valid,
                "failedFacePathRows": failed_paths,
                "disconnectedRequiredRows": disconnected,
                "nonManifoldEdgeCount": manifold["nonManifoldEdgeCount"],
                "minimumAffectedComponentAreaRetention": round(
                    float(minimum_retention), 6
                ),
            }
        )
        exact_attempts.append(attempt_record)
        if surface_valid:
            chosen_result = {
                "state": state,
                "selected": selected,
                "component": component,
                "componentSizeById": component_size,
                "interfaceConflict": interface_conflict,
                "crossingConflict": crossing_conflict,
                "inheritance": inheritance,
                "strictSurface": strict_surface,
                "strictStats": strict_stats,
                "supplemental": supplemental,
                "pathRecords": path_records,
                "augmentedSurface": augmented_surface,
                "connections": connections,
                "manifold": manifold,
                "componentArea": component_area,
                "minimumAreaRetention": minimum_retention,
            }
            break
    if chosen_result is None:
        raise RuntimeError(
            "no exact complete-strip assignment preserved the cumulative CT surface"
        )
    exact_at = time.monotonic()

    state = chosen_result["state"]
    selected = chosen_result["selected"]
    component = chosen_result["component"]
    strict_surface = chosen_result["strictSurface"]
    supplemental = chosen_result["supplemental"]
    augmented_surface = chosen_result["augmentedSurface"]
    base_triangle = np.asarray(
        strict_surface["triangleFrontierIndex"], dtype=np.int32
    )
    augmented_triangle = np.asarray(
        augmented_surface["triangleFrontierIndex"], dtype=np.int32
    )
    supplemental_triangle = supplemental["supplementalTriangleFrontierIndex"]
    loops, loop_stats = extract_surface_boundary_loops(
        augmented_surface, settings=surface_settings
    )
    triangle_supplemental = np.concatenate(
        (
            np.zeros(len(base_triangle), dtype=np.uint8),
            np.ones(len(supplemental_triangle), dtype=np.uint8),
        )
    )
    triangle_minimum_path = np.concatenate(
        (
            np.zeros(len(base_triangle), dtype=np.uint8),
            supplemental["supplementalTriangleMinimumPath"],
        )
    )
    triangle_ct_normal = np.concatenate(
        (
            np.full(len(base_triangle), np.nan, dtype=np.float32),
            supplemental["supplementalTriangleCtNormalResidualDegrees"],
        )
    )
    arrays: dict[str, np.ndarray] = {
        **{
            key: np.asarray(value)
            for key, value in strict_surface.items()
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
        "triangleAreaVoxelsSquared": np.asarray(
            augmented_surface["triangleAreaVoxelsSquared"], dtype=np.float32
        ),
        "triangleNormalResidualDegrees": np.asarray(
            augmented_surface["triangleNormalResidualDegrees"], dtype=np.float32
        ),
        "triangleSupplementalCtFace": triangle_supplemental,
        "triangleMinimumCorridorPathFace": triangle_minimum_path,
        "triangleCtNormalResidualDegrees": triangle_ct_normal,
        "baseStrictTriangleCount": np.asarray([len(base_triangle)], dtype=np.int64),
        "chosenCorridorRow": np.asarray(state["rows"], dtype=np.int32),
        "chosenVariantIndex": np.asarray(
            state["variantIndices"], dtype=np.int32
        ),
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
    preview_rows = tuple(sorted(set(prior_face_rows) | set(state["rows"])))
    replay_preview = {
        "corridorReplayProposalSuccessful": np.isin(
            np.arange(len(remapped["scoredCorridorIndex"])), preview_rows
        ).astype(np.uint8),
        "corridorReplayComponent": component,
        "corridorReplayChartUV": np.asarray(
            strict_surface["chartUV"], dtype=np.float32
        ),
        "corridorReplayTriangleFrontierIndex": augmented_triangle,
    }
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    if progress is not None:
        progress("flattening complete-strip replay components from native CT")
    _, preview_stats = write_replayed_corridor_fragment_montage(
        remapped,
        remapped,
        remapped,
        replay_preview,
        source,
        preview_path,
        maximum_components=resolved.maximum_preview_components,
    )
    new_replay_preview = dict(replay_preview)
    new_replay_preview["corridorReplayProposalSuccessful"] = np.isin(
        np.arange(len(remapped["scoredCorridorIndex"])), state["rows"]
    ).astype(np.uint8)
    _, new_preview_stats = write_replayed_corridor_fragment_montage(
        remapped,
        remapped,
        remapped,
        new_replay_preview,
        source,
        new_preview_path,
        maximum_components=len(state["rows"]),
    )
    finished = time.monotonic()
    prior_strict_count = int(
        np.asarray(replay["baseStrictTriangleCount"]).reshape(-1)[0]
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_COMPLETE_STRIP_REPLAY_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "optimization": {
            **optimization_stats,
            "exactAttemptCount": len(exact_attempts),
            "exactAttempts": exact_attempts,
            "chosenCorridorRows": [int(value) for value in state["rows"]],
            "chosenVariantIndices": [
                int(value) for value in state["variantIndices"]
            ],
            "chosenSelectionKey": [
                round(float(value), 6) for value in state["key"]
            ],
        },
        "strictTopology": {
            "selectedRibbonCountBefore": int(np.count_nonzero(baseline_selected)),
            "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
            "addedRibbonCount": len(state["added"]),
            "removedRibbonCount": len(state["removed"]),
            "componentCountAfter": len(chosen_result["componentSizeById"]),
            "interfaceConflictCount": chosen_result["interfaceConflict"],
            "crossingConflictCount": chosen_result["crossingConflict"],
            **chosen_result["inheritance"],
        },
        "surface": {
            "originalCorridorReference": reference_surface_stats,
            "cumulativeBeforeSupplementalFaces": chosen_result["strictStats"],
            "strictTriangleCountBefore": prior_strict_count,
            "strictTriangleCountAfter": len(base_triangle),
            "supplementalCtFaceCountBefore": int(
                np.count_nonzero(replay["triangleSupplementalCtFace"])
            ),
            "supplementalCtFaceCountAfter": len(supplemental_triangle),
            "minimumPathCtFaceCountAfter": int(
                np.count_nonzero(supplemental["supplementalTriangleMinimumPath"])
            ),
            "augmentedTriangleCountBefore": len(replay["triangleFrontierIndex"]),
            "augmentedTriangleCountAfter": len(augmented_triangle),
            "triangleRegionCountBefore": int(
                len(np.unique(_triangle_region_labels(
                    np.asarray(replay["triangleFrontierIndex"], dtype=np.int32)
                )))
            ),
            "triangleRegionCountAfter": int(
                len(np.unique(_triangle_region_labels(augmented_triangle)))
            ),
            "preservedPriorFaceCorridorCount": len(prior_face_rows),
            "preservedPriorExactCorridorCount": len(prior_exact_rows),
            "newCorridorConnectionCount": len(state["rows"]),
            "affectedComponents": chosen_result["componentArea"],
            "minimumAffectedComponentAreaRetention": round(
                float(chosen_result["minimumAreaRetention"]), 6
            ),
            "manifold": chosen_result["manifold"],
            "loops": loop_stats,
            "pathRecords": chosen_result["pathRecords"],
        },
        "flattenedFragments": preview_stats,
        "flattenedNewFragments": new_preview_stats,
        "timingSeconds": {
            "optimization": round(optimized_at - started, 6),
            "referenceSurface": round(referenced_at - optimized_at, 6),
            "exactReplay": round(exact_at - referenced_at, 6),
            "loopsWritingAndPreview": round(finished - exact_at, 6),
            "total": round(finished - started, 6),
        },
        "method": {
            "sheetIdentity": "strict ribbon graph with inherited-component audit",
            "surfaceConnectivity": (
                "all prior and new native-CT-gated chart-Delaunay paths "
                "recomputed together in the final shared charts"
            ),
            "singleCellGrowth": False,
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
            "flattenedNewFragments": new_preview_path.name,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
