from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _write_npz
from .physical_ribbon_complete_strip_replay import (
    _apply_state,
    _grouped_candidate_states,
)
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_face_replay import (
    _affected_component_area_audit,
    _edge_manifold_audit,
    _hard_conflict_counts,
    _preserves_prior_component_anchors,
    _selection_contract,
)
from .physical_ribbon_corridor_faces import PhysicalRibbonCorridorFaceSettings
from .physical_ribbon_corridor_one_sided import _load_frontier, _load_stage_inputs
from .physical_ribbon_corridor_sets import _modifications_valid, _variant_values
from .physical_ribbon_corridor_variants import (
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
)
from .physical_ribbon_cumulative_replay import (
    load_cumulative_strip_replay_artifact,
)
from .physical_ribbon_lineage_strip_replay import (
    _augmented_area_records,
    _complete_inheritance_audit,
    _cumulative_supplemental_face_arrays,
)
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    write_replayed_corridor_fragment_montage,
)
from .physical_ribbon_patch_holes import extract_surface_boundary_loops
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_SCHEMA = (
    "pareidolia.physical-ribbon-cumulative-corridor-replay"
)
PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_VERSION = 1
PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_STEM = (
    "physical-ribbon-cumulative-corridor-replay-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCumulativeCorridorReplaySettings:
    global_assignment_beam_width: int = 4096
    maximum_exact_states: int = 8
    maximum_preview_components: int = 4
    minimum_affected_component_area_retention: float = 0.95
    minimum_affected_component_augmented_area_retention: float = 0.98

    def __post_init__(self) -> None:
        if self.global_assignment_beam_width < 2:
            raise ValueError("cumulative replay beam must retain alternatives")
        if self.maximum_exact_states < 1:
            raise ValueError("cumulative replay must test an exact state")
        if self.maximum_preview_components < 1:
            raise ValueError("cumulative replay must preview at least one sheet")
        if not 0.0 < self.minimum_affected_component_area_retention <= 1.0:
            raise ValueError("strict area retention must lie in (0, 1]")
        if not self.minimum_affected_component_area_retention <= (
            self.minimum_affected_component_augmented_area_retention
        ) <= 1.0:
            raise ValueError(
                "augmented retention must lie between the strict floor and one"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def _candidate_records(arrays: Mapping[str, np.ndarray]) -> list[dict[str, Any]]:
    """Translate exact one-sided variants into the generic strip-state beam."""

    eligible = np.asarray(arrays["corridorVariantSurfaceEligible"]) > 0
    rows = np.asarray(arrays["corridorVariantRow"], dtype=np.int32)
    ranks = np.asarray(arrays["corridorVariantRank"], dtype=np.int32)
    before_region = np.asarray(
        arrays["corridorVariantTriangleRegionCountBefore"], dtype=np.int32
    )
    after_region = np.asarray(
        arrays["corridorVariantTriangleRegionCountAfter"], dtype=np.int32
    )
    before_area = np.asarray(
        arrays["corridorVariantTriangleAreaBefore"], dtype=np.float32
    )
    after_area = np.asarray(
        arrays["corridorVariantTriangleAreaAfter"], dtype=np.float32
    )
    coverage = np.asarray(
        arrays["corridorVariantPatchCoverage"], dtype=np.float32
    )
    objective = np.asarray(
        arrays["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    records = []
    for index in np.flatnonzero(eligible):
        retention = float(after_area[index] / max(before_area[index], 1.0e-6))
        records.append(
            {
                "corridorRow": int(rows[index]),
                "variantRank": int(ranks[index]),
                "variantIndex": int(index),
                "triangleRegionCountBefore": int(before_region[index]),
                "triangleRegionCountAfter": int(after_region[index]),
                "strictAreaRetention": retention,
                "augmentedAreaRetention": retention,
                "patchCoverage": float(coverage[index]),
                "physicalPathFaceCount": 0,
                "physicalPathCost": 0.0,
                "localObjectiveDelta": float(objective[index]),
            }
        )
    records.sort(
        key=lambda value: (
            value["corridorRow"],
            value["variantRank"],
            value["variantIndex"],
        )
    )
    return records


def _map_frontier_by_bank(
    source_bank: np.ndarray,
    target_bank: np.ndarray,
    *,
    ribbon_count: int,
) -> np.ndarray:
    bank_to_target = np.full(ribbon_count, -1, dtype=np.int32)
    bank_to_target[np.asarray(target_bank, dtype=np.int32)] = np.arange(
        len(target_bank), dtype=np.int32
    )
    result = bank_to_target[np.asarray(source_bank, dtype=np.int32)]
    if np.any(result < 0):
        raise ValueError("cumulative frontier dropped an inherited ribbon mode")
    return result


def _legacy_connection_catalogs(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Recover every required old connection from pre-catalog replays."""

    rows = {
        int(record["corridorRow"])
        for record in manifest.get("surface", {}).get("pathRecords", ())
    }
    exact_reference = manifest.get("identity", {}).get("priorExactReplay")
    if exact_reference is not None:
        exact_path, exact_manifest, exact = _load_replay_artifact(
            exact_reference["manifestPath"]
        )
        if (
            sha256_file(exact_path) != exact_reference["manifestSha256"]
            or exact_manifest["data"]["sha256"] != exact_reference["dataSha256"]
        ):
            raise ValueError("prior exact corridor replay has changed")
        rows.update(
            int(value)
            for value in np.flatnonzero(
                np.asarray(exact["corridorReplayProposalSuccessful"]) > 0
            )
        )
    corridor_reference = manifest["identity"]["corridors"]
    return [
        {
            "corridors": dict(corridor_reference),
            "rows": sorted(rows),
            "faceSettings": dict(manifest["identity"]["faceSettings"]),
        }
    ]


def _connection_catalogs(
    manifest: Mapping[str, Any],
) -> list[dict[str, Any]]:
    catalogs = manifest.get("surface", {}).get("connectionCatalogs")
    if catalogs is None:
        return _legacy_connection_catalogs(manifest)
    return [
        {
            "corridors": dict(value["corridors"]),
            "rows": [int(row) for row in value["rows"]],
            "faceSettings": dict(value["faceSettings"]),
        }
        for value in catalogs
    ]


def _merge_connection_catalogs(
    prior: Sequence[Mapping[str, Any]],
    current_reference: Mapping[str, Any],
    current_rows: Sequence[int],
    face_settings: PhysicalRibbonCorridorFaceSettings,
) -> list[dict[str, Any]]:
    merged: dict[str, dict[str, Any]] = {}
    order: list[str] = []
    for value in [
        *prior,
        {
            "corridors": dict(current_reference),
            "rows": [int(row) for row in current_rows],
            "faceSettings": face_settings.record(),
        },
    ]:
        key = str(value["corridors"]["dataSha256"])
        if key not in merged:
            merged[key] = {
                "corridors": dict(value["corridors"]),
                "rows": set(),
                "faceSettings": dict(value["faceSettings"]),
            }
            order.append(key)
        elif merged[key]["faceSettings"] != dict(value["faceSettings"]):
            raise ValueError("one corridor catalog has inconsistent face settings")
        merged[key]["rows"].update(int(row) for row in value["rows"])
    return [
        {
            **merged[key],
            "rows": sorted(merged[key]["rows"]),
        }
        for key in order
        if merged[key]["rows"]
    ]


def _remap_connection_catalog(
    catalog: Mapping[str, Any],
    final_surface: Mapping[str, np.ndarray],
    final_configuration: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    reference = catalog["corridors"]
    path, manifest, corridor = _load_corridor_artifact(reference["manifestPath"])
    if (
        sha256_file(path) != reference["manifestSha256"]
        or manifest["data"]["sha256"] != reference["dataSha256"]
    ):
        raise ValueError("connection-catalog corridor artifact has changed")
    configuration_reference = manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        base_configuration,
        _,
        _,
        base_topology,
        _,
        ribbon_manifest,
        _,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path)
        != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("connection-catalog configuration has changed")
    current_bank_reference = np.asarray(
        topology["frontierRibbonCandidate"], dtype=np.int32
    )
    old_to_current = _map_frontier_by_bank(
        np.asarray(base_topology["frontierRibbonCandidate"], dtype=np.int32),
        current_bank_reference,
        ribbon_count=len(np.asarray(ribbon["sourceInterface"])),
    )
    if (
        ribbon_manifest["data"]["sha256"]
        != manifest["identity"]["ribbonBank"]["dataSha256"]
    ):
        raise ValueError("connection catalog and current replay use different banks")
    remapped = _remap_corridor_surface(
        corridor,
        final_surface,
        base_configuration,
        final_configuration,
        old_to_current,
    )
    return remapped, {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


_SUPPLEMENTAL_FIELDS = (
    "supplementalTrianglePrimaryCorridorRow",
    "supplementalTriangleMinimumPath",
    "supplementalTriangleAreaVoxelsSquared",
    "supplementalTriangleNodeNormalResidualDegrees",
    "supplementalTriangleCtNormalResidualDegrees",
    "supplementalTriangleCenterDistanceThicknesses",
    "supplementalTriangleCenterHeightThicknesses",
    "supplementalTriangleMaximumEdgeThicknesses",
)


def _merge_supplemental_faces(
    values: Sequence[tuple[int, Mapping[str, np.ndarray]]],
) -> dict[str, np.ndarray]:
    """Deduplicate physical faces reconstructed from multiple catalogs."""

    by_key: dict[tuple[int, int, int], dict[str, Any]] = {}
    for catalog_index, arrays in values:
        triangle = np.asarray(
            arrays["supplementalTriangleFrontierIndex"], dtype=np.int32
        )
        for index, face in enumerate(triangle):
            key = tuple(sorted(int(node) for node in face))
            record = {
                "triangle": face.copy(),
                "catalogIndex": int(catalog_index),
                **{
                    name: np.asarray(arrays[name])[index].item()
                    for name in _SUPPLEMENTAL_FIELDS
                },
            }
            previous = by_key.get(key)
            if previous is None:
                by_key[key] = record
            else:
                previous["supplementalTriangleMinimumPath"] = int(
                    bool(previous["supplementalTriangleMinimumPath"])
                    or bool(record["supplementalTriangleMinimumPath"])
                )
                if record["supplementalTriangleCtNormalResidualDegrees"] < (
                    previous["supplementalTriangleCtNormalResidualDegrees"]
                ):
                    minimum_path = previous["supplementalTriangleMinimumPath"]
                    by_key[key] = record
                    by_key[key]["supplementalTriangleMinimumPath"] = minimum_path
    ordered = [by_key[key] for key in sorted(by_key)]
    return {
        "supplementalTriangleFrontierIndex": np.asarray(
            [value["triangle"] for value in ordered], dtype=np.int32
        ).reshape((-1, 3)),
        **{
            name: np.asarray(
                [value[name] for value in ordered],
                dtype=(
                    np.uint8
                    if name == "supplementalTriangleMinimumPath"
                    else np.int32
                    if name == "supplementalTrianglePrimaryCorridorRow"
                    else np.float32
                ),
            )
            for name in _SUPPLEMENTAL_FIELDS
        },
        "supplementalTriangleCatalogIndex": np.asarray(
            [value["catalogIndex"] for value in ordered], dtype=np.int16
        ),
    }


def _mapped_prior_surface(
    replay: Mapping[str, np.ndarray],
    prior_to_current: np.ndarray,
    selected: np.ndarray,
    component: np.ndarray,
    *,
    strict: bool,
) -> dict[str, np.ndarray]:
    triangles = prior_to_current[
        np.asarray(replay["triangleFrontierIndex"], dtype=np.int32)
    ]
    areas = np.asarray(replay["triangleAreaVoxelsSquared"], dtype=np.float32)
    if strict:
        count = int(np.asarray(replay["baseStrictTriangleCount"]).reshape(-1)[0])
        triangles = triangles[:count]
        areas = areas[:count]
    return {
        "selected": selected.astype(np.uint8),
        "component": component,
        "triangleFrontierIndex": triangles,
        "triangleAreaVoxelsSquared": areas,
    }


def run_physical_ribbon_cumulative_corridor_replay(
    prior_replay_root: str | Path,
    candidate_replay_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCumulativeCorridorReplaySettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCumulativeCorridorReplaySettings()
    prior_path, prior_manifest, prior = load_cumulative_strip_replay_artifact(
        prior_replay_root
    )
    candidate_path, candidate_manifest, variants = _load_replay_artifact(
        candidate_replay_root
    )
    frontier_path, frontier_manifest, topology = _load_frontier(
        candidate_manifest["identity"]["frontier"]["manifestPath"]
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
    corridor_reference = _reference(corridor_path, corridor_manifest)
    face_settings = PhysicalRibbonCorridorFaceSettings(
        **prior_manifest["identity"]["faceSettings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_VERSION,
        "priorReplay": _reference(prior_path, prior_manifest),
        "candidateReplay": _reference(candidate_path, candidate_manifest),
        "frontier": _reference(frontier_path, frontier_manifest),
        "corridors": corridor_reference,
        "configuration": _reference(
            configuration_path, configuration_manifest
        ),
        "settings": resolved.record(),
        "faceSettings": face_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
        "faceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_faces.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_STEM}.npz"
    preview_path = output / "physical-ribbon-cumulative-corridor-new-fragments.png"
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
    prior_frontier = np.asarray(prior["frontierRibbonCandidate"], dtype=np.int32)
    target_frontier = np.asarray(
        topology["frontierRibbonCandidate"], dtype=np.int32
    )
    prior_to_current = _map_frontier_by_bank(
        prior_frontier,
        target_frontier,
        ribbon_count=len(np.asarray(ribbon["sourceInterface"])),
    )
    baseline_selected = np.zeros(len(target_frontier), dtype=bool)
    baseline_selected[prior_to_current] = np.asarray(prior["selected"]) > 0
    baseline_component = np.full(len(target_frontier), -1, dtype=np.int32)
    baseline_component[prior_to_current] = np.asarray(
        prior["component"], dtype=np.int32
    )
    if not np.array_equal(
        baseline_selected, np.asarray(topology["selected"]) > 0
    ):
        raise ValueError(
            "candidate frontier is not conditioned on the prior cumulative state"
        )
    contract = _selection_contract(
        topology, ribbon, baseline_selected=baseline_selected
    )

    def valid_modifications(
        added: frozenset[int], removed: frozenset[int]
    ) -> bool:
        return _preserves_prior_component_anchors(
            added,
            removed,
            baseline_selected=baseline_selected,
            baseline_component=baseline_component,
        ) and _modifications_valid(
            added,
            removed,
            baseline_selected=contract["baselineSelected"],
            baseline_interface_owner=contract["baselineInterfaceOwner"],
            source_interface=contract["sourceInterface"],
            target_interface=contract["targetInterface"],
            crossing_neighbor=contract["crossingNeighbor"],
        )

    candidates = _candidate_records(variants)
    if not candidates:
        raise RuntimeError("candidate replay contains no exact eligible strip state")
    states, optimization_stats = _grouped_candidate_states(
        candidates,
        variants,
        valid_modifications=valid_modifications,
        beam_width=resolved.global_assignment_beam_width,
    )
    states = [value for value in states if value["rows"]]
    optimized_at = time.monotonic()
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    surface_settings = corridor_settings.surface_settings()
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    prior_catalogs = _connection_catalogs(prior_manifest)
    prior_strict_surface = _mapped_prior_surface(
        prior,
        prior_to_current,
        baseline_selected,
        baseline_component,
        strict=True,
    )
    prior_augmented_surface = _mapped_prior_surface(
        prior,
        prior_to_current,
        baseline_selected,
        baseline_component,
        strict=False,
    )
    attempts: list[dict[str, Any]] = []
    chosen: dict[str, Any] | None = None
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
            source_interface=np.asarray(ribbon["sourceInterface"], dtype=np.int32)[
                target_frontier
            ],
            target_interface=np.asarray(ribbon["targetInterface"], dtype=np.int32)[
                target_frontier
            ],
        )
        topology_valid = not (
            interface_conflict
            or crossing_conflict
            or inheritance["splitPriorComponentCount"]
            or inheritance["forbiddenSubstantialFusionCount"]
            or inheritance["forbiddenPriorComponentDeletionCount"]
            or inheritance["orphanFinalComponentCount"]
        )
        record: dict[str, Any] = {
            "attempt": attempt,
            "corridorRows": [int(value) for value in state["rows"]],
            "variantIndices": [int(value) for value in state["variantIndices"]],
            "topologyValid": topology_valid,
            "surfaceValid": False,
            "inheritance": inheritance,
        }
        if not topology_valid:
            attempts.append(record)
            continue
        final_configuration = dict(topology)
        final_configuration["selected"] = selected.astype(np.uint8)
        final_configuration["component"] = component
        if progress is not None:
            progress(
                f"exact cumulative state {attempt}/{min(len(states), resolved.maximum_exact_states)} "
                f"· rows {list(state['rows'])}"
            )
        strict_surface, strict_stats = build_physical_ribbon_surface_complex(
            ribbon, topology, final_configuration, settings=surface_settings
        )
        catalogs = _merge_connection_catalogs(
            prior_catalogs,
            corridor_reference,
            state["rows"],
            face_settings,
        )
        supplemental_by_catalog = []
        remapped_catalogs = []
        path_records = []
        failed_paths: list[dict[str, int]] = []
        strict_rows_by_catalog: list[list[int]] = []
        for catalog_index, catalog in enumerate(catalogs):
            remapped, verified_reference = _remap_connection_catalog(
                catalog,
                strict_surface,
                final_configuration,
                topology,
                ribbon,
            )
            catalog["corridors"] = verified_reference
            catalog_face_settings = PhysicalRibbonCorridorFaceSettings(
                **catalog["faceSettings"]
            )
            supplemental, records, strict_rows = (
                _cumulative_supplemental_face_arrays(
                    tuple(catalog["rows"]),
                    strict_surface,
                    remapped,
                    surface_settings=surface_settings,
                    face_settings=catalog_face_settings,
                    require_baseline_distinct=False,
                )
            )
            supplemental_by_catalog.append((catalog_index, supplemental))
            remapped_catalogs.append(remapped)
            strict_rows_by_catalog.append([int(value) for value in strict_rows])
            for value in records:
                path_records.append({"catalogIndex": catalog_index, **value})
                if value.get("eligible") is not True:
                    failed_paths.append(
                        {
                            "catalogIndex": catalog_index,
                            "corridorRow": int(value["corridorRow"]),
                        }
                    )
        supplemental = _merge_supplemental_faces(supplemental_by_catalog)
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
        disconnected: list[dict[str, int]] = []
        for catalog_index, (catalog, remapped) in enumerate(
            zip(catalogs, remapped_catalogs)
        ):
            catalog_face_settings = PhysicalRibbonCorridorFaceSettings(
                **catalog["faceSettings"]
            )
            connections = _evaluate_corridor_connections(
                augmented_surface,
                remapped,
                remapped,
                minimum_arc_region_fraction=(
                    catalog_face_settings.minimum_arc_region_fraction
                ),
                maximum_arc_triangle_distance_edges=(
                    catalog_face_settings.maximum_arc_triangle_distance_edges
                ),
            )
            for row in catalog["rows"]:
                if (
                    connections["boundaryArcSharedRegionFraction"][int(row)]
                    < catalog_face_settings.minimum_arc_region_fraction
                ):
                    disconnected.append(
                        {"catalogIndex": catalog_index, "corridorRow": int(row)}
                    )
        manifold = _edge_manifold_audit(augmented_triangle)
        current_catalog_index = next(
            index
            for index, value in enumerate(catalogs)
            if value["corridors"]["dataSha256"]
            == corridor_reference["dataSha256"]
        )
        current_remapped = remapped_catalogs[current_catalog_index]
        affected_components = sorted(
            {
                int(
                    current_remapped["corridorTopologyComponent"][
                        int(current_remapped["scoredCorridorIndex"][row])
                    ]
                )
                for row in state["rows"]
            }
        )
        component_area, minimum_retention = _affected_component_area_audit(
            prior_strict_surface, strict_surface, affected_components
        )
        augmented_area_raw, minimum_augmented_retention = (
            _affected_component_area_audit(
                prior_augmented_surface,
                augmented_surface,
                affected_components,
            )
        )
        augmented_area = _augmented_area_records(augmented_area_raw)
        surface_valid = not (
            failed_paths
            or disconnected
            or manifold["nonManifoldEdgeCount"]
            or minimum_retention
            < resolved.minimum_affected_component_area_retention
            or minimum_augmented_retention
            < resolved.minimum_affected_component_augmented_area_retention
        )
        record.update(
            {
                "surfaceValid": surface_valid,
                "failedFacePaths": failed_paths,
                "disconnectedRequiredConnections": disconnected,
                "nonManifoldEdgeCount": manifold["nonManifoldEdgeCount"],
                "minimumAffectedComponentAreaRetention": round(
                    float(minimum_retention), 6
                ),
                "minimumAffectedComponentAugmentedAreaRetention": round(
                    float(minimum_augmented_retention), 6
                ),
                "strictRowsByCatalog": strict_rows_by_catalog,
            }
        )
        attempts.append(record)
        if surface_valid:
            chosen = {
                "state": state,
                "selected": selected,
                "component": component,
                "componentSize": component_size,
                "interfaceConflict": interface_conflict,
                "crossingConflict": crossing_conflict,
                "inheritance": inheritance,
                "strictSurface": strict_surface,
                "strictStats": strict_stats,
                "supplemental": supplemental,
                "augmentedSurface": augmented_surface,
                "catalogs": catalogs,
                "remappedCatalogs": remapped_catalogs,
                "pathRecords": path_records,
                "manifold": manifold,
                "componentArea": component_area,
                "minimumAreaRetention": minimum_retention,
                "augmentedComponentArea": augmented_area,
                "minimumAugmentedAreaRetention": minimum_augmented_retention,
            }
            break
        if progress is not None:
            progress(
                f"state {attempt} rejected · failed paths {len(failed_paths)} "
                f"· disconnected {len(disconnected)} · manifold "
                f"{manifold['nonManifoldEdgeCount']} · strict area "
                f"{minimum_retention:.6f} · augmented area "
                f"{minimum_augmented_retention:.6f}"
            )
    exact_at = time.monotonic()
    if chosen is None:
        failure_path = output / "physical-ribbon-cumulative-corridor-replay-failures.json"
        atomic_json(
            failure_path,
            {
                "schema": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_SCHEMA,
                "version": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_VERSION,
                "state": "no-valid-exact-state",
                "identity": identity,
                "optimization": {**optimization_stats, "exactAttempts": attempts},
            },
        )
        raise RuntimeError(
            "no candidate state preserved the complete cumulative surface; "
            f"see {failure_path}"
        )

    state = chosen["state"]
    strict_surface = chosen["strictSurface"]
    augmented_surface = chosen["augmentedSurface"]
    supplemental = chosen["supplemental"]
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
        "selected": np.asarray(chosen["selected"], dtype=np.uint8),
        "component": np.asarray(chosen["component"], dtype=np.int32),
        "triangleFrontierIndex": augmented_triangle,
        "triangleAreaVoxelsSquared": np.asarray(
            augmented_surface["triangleAreaVoxelsSquared"], dtype=np.float32
        ),
        "triangleNormalResidualDegrees": np.asarray(
            augmented_surface["triangleNormalResidualDegrees"], dtype=np.float32
        ),
        "triangleSupplementalCtFace": np.concatenate(
            (
                np.zeros(len(base_triangle), dtype=np.uint8),
                np.ones(len(supplemental_triangle), dtype=np.uint8),
            )
        ),
        "triangleMinimumCorridorPathFace": np.concatenate(
            (
                np.zeros(len(base_triangle), dtype=np.uint8),
                supplemental["supplementalTriangleMinimumPath"],
            )
        ),
        "triangleCtNormalResidualDegrees": np.concatenate(
            (
                np.full(len(base_triangle), np.nan, dtype=np.float32),
                supplemental["supplementalTriangleCtNormalResidualDegrees"],
            )
        ),
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
    current_catalog_index = next(
        index
        for index, value in enumerate(chosen["catalogs"])
        if value["corridors"]["dataSha256"] == corridor_reference["dataSha256"]
    )
    current_remapped = chosen["remappedCatalogs"][current_catalog_index]
    replay_preview = {
        "corridorReplayProposalSuccessful": np.isin(
            np.arange(len(current_remapped["scoredCorridorIndex"])), state["rows"]
        ).astype(np.uint8),
        "corridorReplayComponent": arrays["component"],
        "corridorReplayChartUV": np.asarray(
            strict_surface["chartUV"], dtype=np.float32
        ),
        "corridorReplayTriangleFrontierIndex": augmented_triangle,
    }
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    _, preview_stats = write_replayed_corridor_fragment_montage(
        current_remapped,
        current_remapped,
        current_remapped,
        replay_preview,
        source,
        preview_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CUMULATIVE_CORRIDOR_REPLAY_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "optimization": {
            **optimization_stats,
            "exactAttemptCount": len(attempts),
            "exactAttempts": attempts,
            "chosenCorridorRows": [int(value) for value in state["rows"]],
            "chosenVariantIndices": [
                int(value) for value in state["variantIndices"]
            ],
        },
        "strictTopology": {
            "selectedRibbonCountBefore": int(np.count_nonzero(baseline_selected)),
            "selectedRibbonCountAfter": int(np.count_nonzero(chosen["selected"])),
            "addedRibbonCount": len(state["added"]),
            "removedRibbonCount": len(state["removed"]),
            "componentCountAfter": len(chosen["componentSize"]),
            "interfaceConflictCount": chosen["interfaceConflict"],
            "crossingConflictCount": chosen["crossingConflict"],
            **chosen["inheritance"],
        },
        "surface": {
            "cumulativeBeforeSupplementalFaces": chosen["strictStats"],
            "strictTriangleCountBefore": int(
                np.asarray(prior["baseStrictTriangleCount"]).reshape(-1)[0]
            ),
            "strictTriangleCountAfter": len(base_triangle),
            "supplementalCtFaceCountBefore": int(
                np.count_nonzero(prior["triangleSupplementalCtFace"])
            ),
            "supplementalCtFaceCountAfter": len(supplemental_triangle),
            "augmentedTriangleCountBefore": len(prior["triangleFrontierIndex"]),
            "augmentedTriangleCountAfter": len(augmented_triangle),
            "triangleRegionCountBefore": len(
                np.unique(
                    _triangle_region_labels(
                        np.asarray(prior["triangleFrontierIndex"], dtype=np.int32)
                    )
                )
            ),
            "triangleRegionCountAfter": len(
                np.unique(_triangle_region_labels(augmented_triangle))
            ),
            "preservedPriorConnectionCount": sum(
                len(value["rows"]) for value in prior_catalogs
            ),
            "newCorridorConnectionCount": len(state["rows"]),
            "connectionCatalogs": chosen["catalogs"],
            "affectedComponents": chosen["componentArea"],
            "minimumAffectedComponentAreaRetention": round(
                float(chosen["minimumAreaRetention"]), 6
            ),
            "affectedComponentsAugmented": chosen["augmentedComponentArea"],
            "minimumAffectedComponentAugmentedAreaRetention": round(
                float(chosen["minimumAugmentedAreaRetention"]), 6
            ),
            "manifold": chosen["manifold"],
            "loops": loop_stats,
            "pathRecords": chosen["pathRecords"],
        },
        "flattenedNewFragments": preview_stats,
        "timingSeconds": {
            "optimization": round(optimized_at - started, 6),
            "exactCumulativeReplay": round(exact_at - optimized_at, 6),
            "loopsWritingAndPreview": round(finished - exact_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"flattenedNewFragments": preview_path.name},
        "method": {
            "decisionUnit": "complete CT-supported corridor assignments",
            "connectionMemory": (
                "every inherited and new corridor catalog is rebuilt against "
                "the final shared sheet charts"
            ),
            "singleCellGrowth": False,
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
