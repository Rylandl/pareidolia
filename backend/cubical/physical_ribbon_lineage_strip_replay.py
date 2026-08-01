from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_complete_strip_replay import (
    _apply_state,
    _eligible_candidates,
    _grouped_candidate_states,
)
from .physical_ribbon_complete_strips import (
    _load_face_replay_artifact,
    _strict_surface,
)
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_cumulative_replay import (
    cumulative_face_replay_reference,
    cumulative_prior_exact_reference,
    load_cumulative_strip_replay_artifact,
)
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
from .physical_ribbon_corridor_sets import _modifications_valid
from .physical_ribbon_corridor_variants import _corridor_settings_from_manifest
from .physical_ribbon_lineage_strips import (
    PHYSICAL_RIBBON_LINEAGE_STRIPS_SCHEMA,
    PHYSICAL_RIBBON_LINEAGE_STRIPS_STEM,
)
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    write_replayed_corridor_fragment_montage,
)
from .physical_ribbon_patch_holes import extract_surface_boundary_loops
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_SCHEMA = (
    "pareidolia.physical-ribbon-lineage-strip-replay"
)
PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_VERSION = 1
PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_STEM = (
    "physical-ribbon-lineage-strip-replay-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonLineageStripReplaySettings:
    global_assignment_beam_width: int = 4096
    maximum_exact_states: int = 16
    maximum_preview_components: int = 8
    minimum_affected_component_area_retention: float = 0.95
    minimum_affected_component_augmented_area_retention: float = 0.98

    def __post_init__(self) -> None:
        if self.global_assignment_beam_width < 2:
            raise ValueError("lineage replay beam must retain alternatives")
        if self.maximum_exact_states < 1:
            raise ValueError("lineage replay must test at least one state")
        if self.maximum_preview_components < 1:
            raise ValueError("lineage replay must preview at least one sheet")
        if not 0.0 < self.minimum_affected_component_area_retention <= 1.0:
            raise ValueError("surface-area retention must lie in (0, 1]")
        if not self.minimum_affected_component_area_retention <= (
            self.minimum_affected_component_augmented_area_retention
        ) <= 1.0:
            raise ValueError(
                "augmented area retention must lie between the preclosure "
                "floor and one"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _augmented_area_records(
    records: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Give the generic component-area audit surface-neutral field names."""

    return [
        {
            "priorComponent": int(value["priorComponent"]),
            "finalComponent": int(value["finalComponent"]),
            "triangleCountBefore": int(value["triangleCountBefore"]),
            "triangleCountAfter": int(
                value["triangleCountAfterBeforeSupplementalFaces"]
            ),
            "areaBefore": float(value["areaBefore"]),
            "areaAfter": float(value["areaAfterBeforeSupplementalFaces"]),
            "areaRetention": float(value["areaRetention"]),
        }
        for value in records
    ]


def _load_lineage_strip_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_LINEAGE_STRIPS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_LINEAGE_STRIPS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "lineage replay requires a complete label-free lineage-strip audit"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return manifest_path, manifest, _load_npz(
        data_path, manifest["data"]["sha256"]
    )


def _prior_face_rows(replay_manifest: Mapping[str, Any]) -> tuple[int, ...]:
    return tuple(
        sorted(
            {
                int(record["corridorRow"])
                for record in replay_manifest["surface"]["pathRecords"]
            }
        )
    )


def _complete_inheritance_audit(
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    final_selected: np.ndarray,
    final_component: np.ndarray,
    *,
    minimum_substantial_ribbon_count: int,
) -> dict[str, Any]:
    audit = _component_inheritance_audit(
        baseline_selected,
        baseline_component,
        final_selected,
        final_component,
    )
    deleted: list[int] = []
    for component_id in np.unique(baseline_component[baseline_selected]):
        if component_id < 0:
            continue
        retained = (
            baseline_selected
            & final_selected
            & (baseline_component == component_id)
        )
        if not np.any(retained):
            deleted.append(int(component_id))
    orphaned: list[int] = []
    for component_id in np.unique(final_component[final_selected]):
        if component_id < 0:
            continue
        inherited = (
            baseline_selected
            & final_selected
            & (final_component == component_id)
        )
        if not np.any(inherited):
            orphaned.append(int(component_id))
    prior_size = {
        int(component_id): int(
            np.count_nonzero(
                baseline_selected & (baseline_component == component_id)
            )
        )
        for component_id in np.unique(baseline_component[baseline_selected])
        if component_id >= 0
    }
    fusion_records: list[dict[str, Any]] = []
    forbidden_fusion_count = 0
    for fusion in audit["crossPriorComponentFusions"]:
        prior_components = [int(value) for value in fusion["priorComponents"]]
        substantial = [
            value
            for value in prior_components
            if prior_size[value] >= minimum_substantial_ribbon_count
        ]
        provisional = [
            value for value in prior_components if value not in substantial
        ]
        forbidden = len(substantial) > 1
        forbidden_fusion_count += int(forbidden)
        fusion_records.append(
            {
                **fusion,
                "priorComponentRibbonCounts": {
                    str(value): prior_size[value] for value in prior_components
                },
                "substantialPriorComponents": substantial,
                "provisionalPriorComponents": provisional,
                "forbidden": forbidden,
            }
        )
    return {
        **audit,
        "crossPriorComponentFusions": fusion_records,
        "minimumSubstantialRibbonCount": minimum_substantial_ribbon_count,
        "forbiddenSubstantialFusionCount": forbidden_fusion_count,
        "allowedProvisionalAbsorptionCount": (
            len(fusion_records) - forbidden_fusion_count
        ),
        "deletedPriorComponentCount": len(deleted),
        "deletedPriorComponents": deleted,
        "orphanFinalComponentCount": len(orphaned),
        "orphanFinalComponents": orphaned,
    }


def _cumulative_supplemental_face_arrays(
    chosen_rows: tuple[int, ...],
    surface: Mapping[str, np.ndarray],
    corridor: Mapping[str, np.ndarray],
    *,
    surface_settings: Any,
    face_settings: PhysicalRibbonCorridorFaceSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], tuple[int, ...]]:
    """Rebuild face paths while accepting rows already strict-connected."""

    strict_connections = _evaluate_corridor_connections(
        surface,
        corridor,
        corridor,
        minimum_arc_region_fraction=face_settings.minimum_arc_region_fraction,
        maximum_arc_triangle_distance_edges=(
            face_settings.maximum_arc_triangle_distance_edges
        ),
    )
    strict_rows = tuple(
        int(row)
        for row in chosen_rows
        if strict_connections["boundaryArcsConnected"][row]
    )
    strict_set = set(strict_rows)
    face_rows = tuple(int(row) for row in chosen_rows if row not in strict_set)
    supplemental, face_records = _supplemental_face_arrays(
        face_rows,
        surface,
        corridor,
        surface_settings=surface_settings,
        face_settings=face_settings,
    )
    by_row = {int(record["corridorRow"]): record for record in face_records}
    for row in strict_rows:
        by_row[row] = {
            "corridorRow": row,
            "physicalPathFaceCount": 0,
            "physicalPathCost": 0.0,
            "physicalCandidateFaceCount": 0,
            "attachedPhysicalClosureFaceCount": 0,
            "exactConnected": True,
            "sharedArcRegionFraction": round(
                float(
                    strict_connections["boundaryArcSharedRegionFraction"][row]
                ),
                6,
            ),
            "eligible": True,
            "strictBeforeSupplementalFaces": True,
        }
    return supplemental, [by_row[int(row)] for row in chosen_rows], strict_rows


def run_physical_ribbon_lineage_strip_replay(
    lineage_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonLineageStripReplaySettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonLineageStripReplaySettings()
    lineage_path, lineage_manifest, variants = _load_lineage_strip_artifact(
        lineage_root
    )
    replay_reference = lineage_manifest["identity"]["replay"]
    replay_path, replay_manifest, replay = load_cumulative_strip_replay_artifact(
        replay_reference["manifestPath"]
    )
    if (
        sha256_file(replay_path) != replay_reference["manifestSha256"]
        or replay_manifest["data"]["sha256"] != replay_reference["dataSha256"]
    ):
        raise ValueError("lineage-strip source replay has changed")
    face_reference = cumulative_face_replay_reference(replay_manifest)
    face_path, face_manifest, _ = _load_face_replay_artifact(
        face_reference["manifestPath"]
    )
    if (
        sha256_file(face_path) != face_reference["manifestSha256"]
        or face_manifest["data"]["sha256"] != face_reference["dataSha256"]
    ):
        raise ValueError("lineage-strip source face replay has changed")
    prior_exact_reference = cumulative_prior_exact_reference(
        replay_manifest, face_manifest
    )
    prior_exact_path, prior_exact_manifest, prior_exact = _load_replay_artifact(
        prior_exact_reference["manifestPath"]
    )
    if (
        sha256_file(prior_exact_path)
        != prior_exact_reference["manifestSha256"]
        or prior_exact_manifest["data"]["sha256"]
        != prior_exact_reference["dataSha256"]
    ):
        raise ValueError("lineage-strip prior exact replay has changed")
    frontier_path, frontier_manifest, topology = _load_frontier(
        lineage_manifest["identity"]["frontier"]["manifestPath"]
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
        **lineage_manifest["identity"]["faceSettings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_VERSION,
        "lineageStrips": {
            "manifestPath": str(lineage_path),
            "manifestSha256": sha256_file(lineage_path),
            "dataSha256": lineage_manifest["data"]["sha256"],
        },
        "replay": replay_reference,
        "faceReplay": face_reference,
        "priorExactReplay": prior_exact_reference,
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
        "assignmentImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_complete_strip_replay.py")
        ),
        "stripImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_lineage_strips.py")
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
    manifest_path = output / f"{PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_STEM}.npz"
    preview_path = output / "physical-ribbon-lineage-strip-fragments.png"
    new_preview_path = output / "physical-ribbon-lineage-strip-new-fragments.png"
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
    candidates = _eligible_candidates(lineage_manifest)
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
        variants,
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
    prior_face_rows = _prior_face_rows(replay_manifest)
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
    for attempt, state in enumerate(
        states[: resolved.maximum_exact_states], start=1
    ):
        selected = _apply_state(state, baseline_selected)
        component, component_size = _component_labels(selected, first, second)
        interface_conflict, crossing_conflict = _hard_conflict_counts(
            selected, topology, ribbon
        )
        inheritance = _complete_inheritance_audit(
            baseline_selected,
            baseline_component,
            selected,
            component,
            minimum_substantial_ribbon_count=(
                surface_settings.minimum_component_ribbon_count
            ),
        )
        topology_valid = not (
            interface_conflict
            or crossing_conflict
            or inheritance["splitPriorComponentCount"]
            or inheritance["forbiddenSubstantialFusionCount"]
            or inheritance["deletedPriorComponentCount"]
            or inheritance["orphanFinalComponentCount"]
        )
        attempt_record: dict[str, Any] = {
            "attempt": attempt,
            "corridorRows": [int(value) for value in state["rows"]],
            "variantIndices": [int(value) for value in state["variantIndices"]],
            "selectionKey": [round(float(value), 6) for value in state["key"]],
            "topologyValid": topology_valid,
            "surfaceValid": False,
            "inheritance": inheritance,
        }
        if not topology_valid:
            exact_attempts.append(attempt_record)
            continue
        configuration = dict(topology)
        configuration["selected"] = selected.astype(np.uint8)
        configuration["component"] = component
        if progress is not None:
            progress(
                f"exact lineage state {attempt}/{min(len(states), resolved.maximum_exact_states)} "
                f"· rows {list(state['rows'])}"
            )
        strict_surface, strict_stats = build_physical_ribbon_surface_complex(
            ribbon, topology, configuration, settings=surface_settings
        )
        required_face_rows = tuple(
            sorted(set(prior_face_rows) | set(state["rows"]))
        )
        supplemental, path_records, strict_face_rows = (
            _cumulative_supplemental_face_arrays(
                required_face_rows,
                strict_surface,
                remapped,
                surface_settings=surface_settings,
                face_settings=face_settings,
            )
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
                supplemental[
                    "supplementalTriangleNodeNormalResidualDegrees"
                ],
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
            int(
                remapped["corridorTopologyComponent"][
                    int(remapped["scoredCorridorIndex"][row])
                ]
            )
            for row in state["rows"]
        ]
        component_area, minimum_retention = _affected_component_area_audit(
            baseline_surface, strict_surface, affected_components
        )
        augmented_component_area_raw, minimum_augmented_retention = (
            _affected_component_area_audit(
                replay, augmented_surface, affected_components
            )
        )
        augmented_component_area = _augmented_area_records(
            augmented_component_area_raw
        )
        surface_valid = not (
            failed_paths
            or disconnected
            or manifold["nonManifoldEdgeCount"]
            or minimum_retention
            < resolved.minimum_affected_component_area_retention
            or minimum_augmented_retention
            < resolved.minimum_affected_component_augmented_area_retention
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
                "minimumAffectedComponentAugmentedAreaRetention": round(
                    float(minimum_augmented_retention), 6
                ),
                "strictBeforeSupplementalFaceRows": [
                    int(value) for value in strict_face_rows
                ],
            }
        )
        exact_attempts.append(attempt_record)
        if progress is not None and not surface_valid:
            progress(
                f"lineage state {attempt} rejected · failed paths {failed_paths} "
                f"· disconnected {disconnected} · manifold "
                f"{manifold['nonManifoldEdgeCount']} · strict area "
                f"{minimum_retention:.6f} · augmented area "
                f"{minimum_augmented_retention:.6f}"
            )
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
                "augmentedComponentArea": augmented_component_area,
                "minimumAugmentedAreaRetention": (
                    minimum_augmented_retention
                ),
            }
            break
    if chosen_result is None:
        failure_path = output / "physical-ribbon-lineage-strip-replay-failures.json"
        atomic_json(
            failure_path,
            {
                "schema": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_SCHEMA,
                "version": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_VERSION,
                "state": "no-valid-exact-state",
                "identity": identity,
                "optimization": {
                    **optimization_stats,
                    "exactAttemptCount": len(exact_attempts),
                    "exactAttempts": exact_attempts,
                },
                "timingSeconds": {
                    "optimization": round(optimized_at - started, 6),
                    "referenceSurface": round(referenced_at - optimized_at, 6),
                    "exactReplay": round(time.monotonic() - referenced_at, 6),
                },
            },
        )
        raise RuntimeError(
            "no exact lineage-strip assignment preserved the cumulative CT surface; "
            f"see {failure_path}"
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
        progress("flattening lineage-strip replay components from native CT")
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
        "schema": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_LINEAGE_STRIP_REPLAY_VERSION,
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
                len(
                    np.unique(
                        _triangle_region_labels(
                            np.asarray(
                                replay["triangleFrontierIndex"], dtype=np.int32
                            )
                        )
                    )
                )
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
            "affectedComponentsAugmented": chosen_result[
                "augmentedComponentArea"
            ],
            "minimumAffectedComponentAugmentedAreaRetention": round(
                float(chosen_result["minimumAugmentedAreaRetention"]), 6
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
            "sheetIdentity": (
                "strict ribbon graph with whole-component lineage preservation"
            ),
            "surfaceConnectivity": (
                "all prior and new native-CT-gated chart-Delaunay paths "
                "recomputed together in the final shared charts"
            ),
            "areaRetention": (
                "retain the preclosure strict surface above its safety floor "
                "and the complete prior-versus-final CT-augmented surface "
                "above the final density threshold"
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
