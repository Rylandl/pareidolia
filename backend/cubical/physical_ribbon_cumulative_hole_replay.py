from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_face_replay import (
    _affected_component_area_audit,
    _edge_manifold_audit,
    _hard_conflict_counts,
    _preserves_prior_component_anchors,
    _selection_contract,
)
from .physical_ribbon_corridor_faces import PhysicalRibbonCorridorFaceSettings
from .physical_ribbon_corridor_sets import _modifications_valid
from .physical_ribbon_cumulative_corridor_replay import (
    _connection_catalogs,
    _map_frontier_by_bank,
    _mapped_prior_surface,
    _merge_supplemental_faces,
    _reference,
    _remap_connection_catalog,
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
)
from .physical_ribbon_patch_holes import (
    PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
    PHYSICAL_RIBBON_PATCH_HOLES_STEM,
    PhysicalRibbonPatchHoleSettings,
    _loop_vertices,
    extract_surface_boundary_loops,
)


PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_SCHEMA = (
    "pareidolia.physical-ribbon-cumulative-hole-replay"
)
PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_VERSION = 1
PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_STEM = (
    "physical-ribbon-cumulative-hole-replay-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCumulativeHoleReplaySettings:
    global_assignment_beam_width: int = 256
    maximum_exact_states: int = 8
    minimum_context_profile_correlation: float = 0.85
    minimum_competing_layer_margin: float = 0.20
    minimum_patch_coverage: float = 0.75
    minimum_boundary_anchor_fraction: float = 0.75
    maximum_matching_hole_distance_boundary_edges: float = 1.5
    minimum_affected_component_area_retention: float = 0.95
    minimum_affected_component_augmented_area_retention: float = 0.98

    def __post_init__(self) -> None:
        if self.global_assignment_beam_width < 2:
            raise ValueError("cumulative hole beam must retain alternatives")
        if self.maximum_exact_states < 1:
            raise ValueError("cumulative hole replay must test an exact state")
        fractions = (
            self.minimum_context_profile_correlation,
            self.minimum_patch_coverage,
            self.minimum_boundary_anchor_fraction,
            self.minimum_affected_component_area_retention,
            self.minimum_affected_component_augmented_area_retention,
        )
        if any(not 0.0 < value <= 1.0 for value in fractions):
            raise ValueError("cumulative hole fractions must lie in (0, 1]")
        if self.minimum_competing_layer_margin <= 0.0:
            raise ValueError("competing-layer margin must be positive")
        if self.maximum_matching_hole_distance_boundary_edges <= 0.0:
            raise ValueError("hole matching distance must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_hole_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_PATCH_HOLES_STEM}.json"
    )
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
        or "cumulativeSurfaceReplay" not in manifest.get("identity", {})
    ):
        raise ValueError(
            "cumulative hole replay requires a cumulative label-free hole census"
        )
    data_path = path.parent / str(manifest["data"]["path"])
    return path, manifest, _load_npz(data_path, manifest["data"]["sha256"])


def _proposal_values(
    arrays: Mapping[str, np.ndarray], row: int, name: str
) -> frozenset[int]:
    offset = np.asarray(arrays[f"proposal{name}Offset"], dtype=np.int64)
    values = np.asarray(
        arrays[f"proposal{name}FrontierIndex"], dtype=np.int32
    )
    return frozenset(
        int(value) for value in values[int(offset[row]) : int(offset[row + 1])]
    )


def _eligible_hole_proposals(
    arrays: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonCumulativeHoleReplaySettings,
) -> list[dict[str, Any]]:
    loop_index = np.asarray(arrays["reconfigurationLoopIndex"], dtype=np.int32)
    selected_model = np.asarray(arrays["selectedModel"], dtype=np.int32)
    profile = np.asarray(
        arrays["zeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    margin = np.asarray(arrays["zeroShiftCompetingMargin"], dtype=np.float32)
    coverage = np.asarray(arrays["proposalPatchCoverage"], dtype=np.float32)
    retained = np.asarray(
        arrays["proposalRetainedBoundaryFraction"], dtype=np.float32
    )
    anchors = np.asarray(arrays["proposalBoundaryAnchorCount"], dtype=np.int32)
    objective = np.asarray(arrays["proposalObjectiveDelta"], dtype=np.float32)
    offset = np.asarray(arrays["loopOffset"], dtype=np.int64)
    result = []
    for row, loop in enumerate(loop_index):
        model = int(selected_model[row])
        boundary_count = int(offset[loop + 1] - offset[loop])
        anchor_fraction = float(anchors[row] / max(boundary_count, 1))
        record = {
            "row": row,
            "loopIndex": int(loop),
            "profileCorrelation": float(profile[row, model]),
            "competingLayerMargin": float(margin[row, model]),
            "patchCoverage": float(coverage[row]),
            "retainedBoundaryFraction": float(retained[row]),
            "boundaryAnchorFraction": anchor_fraction,
            "localObjectiveDelta": float(objective[row]),
            "added": _proposal_values(arrays, row, "Added"),
            "removed": _proposal_values(arrays, row, "Removed"),
        }
        record["eligible"] = bool(
            record["profileCorrelation"]
            >= settings.minimum_context_profile_correlation
            and record["competingLayerMargin"]
            >= settings.minimum_competing_layer_margin
            and record["patchCoverage"] >= settings.minimum_patch_coverage
            and record["retainedBoundaryFraction"]
            >= settings.minimum_boundary_anchor_fraction
            and record["boundaryAnchorFraction"]
            >= settings.minimum_boundary_anchor_fraction
            and record["localObjectiveDelta"] > 0.0
        )
        result.append(record)
    return result


def _proposal_states(
    candidates: list[dict[str, Any]],
    *,
    valid_modifications: Callable[[frozenset[int], frozenset[int]], bool],
    beam_width: int,
) -> list[dict[str, Any]]:
    beam = [
        {
            "key": (0.0,) * 5,
            "rows": (),
            "added": frozenset(),
            "removed": frozenset(),
        }
    ]
    for candidate in sorted(candidates, key=lambda value: int(value["row"])):
        if not candidate["eligible"]:
            continue
        by_signature: dict[tuple[frozenset[int], frozenset[int]], dict[str, Any]] = {}
        for state in beam:
            for include in (False, True):
                added = state["added"]
                removed = state["removed"]
                rows = state["rows"]
                key = state["key"]
                if include:
                    added = added | candidate["added"]
                    removed = removed | candidate["removed"]
                    if not valid_modifications(added, removed):
                        continue
                    rows = rows + (int(candidate["row"]),)
                    increment = (
                        1.0,
                        float(candidate["patchCoverage"]),
                        float(candidate["profileCorrelation"]),
                        float(candidate["competingLayerMargin"]),
                        float(candidate["localObjectiveDelta"]),
                    )
                    key = tuple(a + b for a, b in zip(key, increment))
                value = {
                    "key": key,
                    "rows": rows,
                    "added": added,
                    "removed": removed,
                }
                signature = (added, removed)
                previous = by_signature.get(signature)
                if previous is None or (value["key"], value["rows"]) > (
                    previous["key"],
                    previous["rows"],
                ):
                    by_signature[signature] = value
        beam = sorted(
            by_signature.values(),
            key=lambda value: (value["key"], value["rows"]),
            reverse=True,
        )[:beam_width]
    return [value for value in beam if value["rows"]]


def _proposal_state_for_rows(
    candidates: list[dict[str, Any]],
    rows: tuple[int, ...],
    *,
    valid_modifications: Callable[[frozenset[int], frozenset[int]], bool],
) -> dict[str, Any] | None:
    by_row = {int(value["row"]): value for value in candidates}
    added: frozenset[int] = frozenset()
    removed: frozenset[int] = frozenset()
    key = (0.0,) * 5
    for row in rows:
        candidate = by_row.get(int(row))
        if candidate is None or not candidate["eligible"]:
            return None
        added |= candidate["added"]
        removed |= candidate["removed"]
        increment = (
            1.0,
            float(candidate["patchCoverage"]),
            float(candidate["profileCorrelation"]),
            float(candidate["competingLayerMargin"]),
            float(candidate["localObjectiveDelta"]),
        )
        key = tuple(a + b for a, b in zip(key, increment))
    if not rows or not valid_modifications(added, removed):
        return None
    return {
        "key": key,
        "rows": tuple(int(value) for value in rows),
        "added": added,
        "removed": removed,
    }


def _proposal_record(value: Mapping[str, Any]) -> dict[str, Any]:
    return {
        **{
            key: item
            for key, item in value.items()
            if key not in {"added", "removed"}
        },
        "addedFrontierIndices": [int(item) for item in sorted(value["added"])],
        "removedFrontierIndices": [
            int(item) for item in sorted(value["removed"])
        ],
    }


def _hole_closure_records(
    chosen_rows: tuple[int, ...],
    original: Mapping[str, np.ndarray],
    final_surface: Mapping[str, np.ndarray],
    final_loops: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonCumulativeHoleReplaySettings,
) -> list[dict[str, Any]]:
    original_midpoint = np.asarray(original["midpointXYZ"], dtype=np.float32)
    final_midpoint = np.asarray(final_surface["midpointXYZ"], dtype=np.float32)
    final_kind = np.asarray(final_loops["loopKind"], dtype=np.uint8)
    final_component = np.asarray(
        final_loops["loopTopologyComponent"], dtype=np.int32
    )
    records = []
    for row in chosen_rows:
        loop = int(np.asarray(original["reconfigurationLoopIndex"])[row])
        nodes = _loop_vertices(original, loop)
        center = np.mean(original_midpoint[nodes], axis=0)
        component = int(np.asarray(original["loopTopologyComponent"])[loop])
        candidate_loop = np.flatnonzero(
            (final_kind == 1) & (final_component == component)
        )
        nearest_loop = -1
        nearest_distance = float("inf")
        for final_loop in candidate_loop:
            final_nodes = _loop_vertices(final_loops, int(final_loop))
            distance = float(
                np.linalg.norm(np.mean(final_midpoint[final_nodes], axis=0) - center)
            )
            if distance < nearest_distance:
                nearest_distance = distance
                nearest_loop = int(final_loop)
        match_distance = settings.maximum_matching_hole_distance_boundary_edges * float(
            np.asarray(original["loopMeanBoundaryEdgeVoxels"])[loop]
        )
        records.append(
            {
                "proposalRow": int(row),
                "originalLoopIndex": loop,
                "componentId": component,
                "nearestFinalInteriorHoleLoop": nearest_loop,
                "nearestFinalInteriorHoleDistanceVoxels": (
                    round(nearest_distance, 6)
                    if np.isfinite(nearest_distance)
                    else None
                ),
                "matchingDistanceVoxels": round(match_distance, 6),
                "stillOpen": bool(nearest_distance <= match_distance),
            }
        )
    return records


def run_physical_ribbon_cumulative_hole_replay(
    prior_replay_root: str | Path,
    hole_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCumulativeHoleReplaySettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCumulativeHoleReplaySettings()
    prior_path, prior_manifest, prior = load_cumulative_strip_replay_artifact(
        prior_replay_root
    )
    hole_path, hole_manifest, holes = _load_hole_artifact(hole_root)
    configuration_reference = hole_manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        configuration,
        topology_path,
        topology_manifest,
        topology,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path)
        != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("hole census configuration has changed")
    topology = dict(topology)
    for name in (
        "crossingFirstFrontierIndex",
        "crossingSecondFrontierIndex",
        "crossingDistanceVoxels",
        "crossingFirstParameter",
        "crossingSecondParameter",
    ):
        topology[name] = np.asarray(configuration[name])
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_VERSION,
        "priorReplay": _reference(prior_path, prior_manifest),
        "holes": _reference(hole_path, hole_manifest),
        "frontier": _reference(topology_path, topology_manifest),
        "configuration": _reference(
            configuration_path, configuration_manifest
        ),
        "settings": resolved.record(),
        "faceSettings": dict(prior_manifest["identity"]["faceSettings"]),
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
    manifest_path = output / f"{PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_STEM}.npz"
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
    prior_to_current = _map_frontier_by_bank(
        np.asarray(prior["frontierRibbonCandidate"], dtype=np.int32),
        np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32),
        ribbon_count=len(np.asarray(ribbon["sourceInterface"])),
    )
    baseline_selected = np.zeros(
        len(np.asarray(topology["frontierRibbonCandidate"])), dtype=bool
    )
    baseline_selected[prior_to_current] = np.asarray(prior["selected"]) > 0
    baseline_component = np.full(len(baseline_selected), -1, dtype=np.int32)
    baseline_component[prior_to_current] = np.asarray(
        prior["component"], dtype=np.int32
    )
    if not np.array_equal(baseline_selected, np.asarray(configuration["selected"]) > 0):
        raise ValueError("hole census is not materialized from the prior replay")
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

    proposals = _eligible_hole_proposals(holes, settings=resolved)
    states = _proposal_states(
        proposals,
        valid_modifications=valid_modifications,
        beam_width=resolved.global_assignment_beam_width,
    )
    if not states:
        raise RuntimeError("no whole-hole proposal passed the physical evidence gates")
    optimized_at = time.monotonic()
    surface_values = dict(hole_manifest["identity"]["settings"])
    for name in ("profile_depth_fractions", "competing_shift_thicknesses"):
        surface_values[name] = tuple(float(value) for value in surface_values[name])
    surface_settings = PhysicalRibbonPatchHoleSettings(**surface_values)
    face_settings = PhysicalRibbonCorridorFaceSettings(
        **prior_manifest["identity"]["faceSettings"]
    )
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
    attempts = []
    chosen: dict[str, Any] | None = None
    pending_states = list(states)
    attempted_rows: set[tuple[int, ...]] = set()
    while pending_states and len(attempts) < resolved.maximum_exact_states:
        state = pending_states.pop(0)
        row_signature = tuple(int(value) for value in state["rows"])
        if row_signature in attempted_rows:
            continue
        attempted_rows.add(row_signature)
        attempt = len(attempts) + 1
        selected = baseline_selected.copy()
        if state["removed"]:
            selected[np.asarray(sorted(state["removed"]), dtype=np.int32)] = False
        if state["added"]:
            selected[np.asarray(sorted(state["added"]), dtype=np.int32)] = True
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
                np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
            ],
            target_interface=np.asarray(ribbon["targetInterface"], dtype=np.int32)[
                np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
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
            "proposalRows": [int(value) for value in state["rows"]],
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
                f"exact cumulative hole state {attempt}/{resolved.maximum_exact_states} "
                f"· rows {list(state['rows'])}"
            )
        strict_surface, strict_stats = build_physical_ribbon_surface_complex(
            ribbon, topology, final_configuration, settings=surface_settings
        )
        supplemental_values = []
        remapped_catalogs = []
        path_records = []
        failed_paths = []
        for catalog_index, catalog in enumerate(prior_catalogs):
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
            supplemental, records, _ = _cumulative_supplemental_face_arrays(
                tuple(catalog["rows"]),
                strict_surface,
                remapped,
                surface_settings=surface_settings,
                face_settings=catalog_face_settings,
                require_baseline_distinct=False,
            )
            supplemental_values.append((catalog_index, supplemental))
            remapped_catalogs.append(remapped)
            for value in records:
                path_records.append({"catalogIndex": catalog_index, **value})
                if value.get("eligible") is not True:
                    failed_paths.append(
                        {"catalogIndex": catalog_index, "corridorRow": int(value["corridorRow"])}
                    )
        supplemental = _merge_supplemental_faces(supplemental_values)
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
                np.asarray(strict_surface["triangleAreaVoxelsSquared"], dtype=np.float32),
                supplemental["supplementalTriangleAreaVoxelsSquared"],
            )
        )
        augmented_surface["triangleNormalResidualDegrees"] = np.concatenate(
            (
                np.asarray(strict_surface["triangleNormalResidualDegrees"], dtype=np.float32),
                supplemental["supplementalTriangleNodeNormalResidualDegrees"],
            )
        )
        disconnected = []
        for catalog_index, (catalog, remapped) in enumerate(
            zip(prior_catalogs, remapped_catalogs)
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
        final_loops, loop_stats = extract_surface_boundary_loops(
            augmented_surface, settings=surface_settings
        )
        closure = _hole_closure_records(
            state["rows"], holes, augmented_surface, final_loops, settings=resolved
        )
        affected_components = sorted(
            {
                int(
                    np.asarray(holes["loopTopologyComponent"])[
                        int(np.asarray(holes["reconfigurationLoopIndex"])[row])
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
            or any(value["stillOpen"] for value in closure)
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
                "holeClosure": closure,
                "minimumAffectedComponentAreaRetention": round(float(minimum_retention), 6),
                "minimumAffectedComponentAugmentedAreaRetention": round(float(minimum_augmented_retention), 6),
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
                "loops": final_loops,
                "loopStats": loop_stats,
                "pathRecords": path_records,
                "closure": closure,
                "manifold": manifold,
                "componentArea": component_area,
                "minimumAreaRetention": minimum_retention,
                "augmentedComponentArea": augmented_area,
                "minimumAugmentedAreaRetention": minimum_augmented_retention,
            }
            break
        open_rows = tuple(
            int(value["proposalRow"])
            for value in closure
            if value["stillOpen"]
        )
        if open_rows:
            # A dense locally ranked state can contain several proposals that
            # never become a closed surface after exact triangulation.  Do not
            # spend the remaining exact budget on near-identical beam states.
            # First test the strongest subset that removes every observed
            # counterexample, while retaining the ordinary beam as fallback.
            trimmed_rows = tuple(
                row for row in row_signature if row not in set(open_rows)
            )
            trimmed = _proposal_state_for_rows(
                proposals,
                trimmed_rows,
                valid_modifications=valid_modifications,
            )
            if (
                trimmed is not None
                and trimmed["rows"] not in attempted_rows
            ):
                pending_states.insert(0, trimmed)
        if progress is not None:
            progress(
                f"hole state {attempt} rejected · open {sum(value['stillOpen'] for value in closure)} "
                f"rows {list(open_rows)} "
                f"· failed paths {len(failed_paths)} · disconnected {len(disconnected)} "
                f"· manifold {manifold['nonManifoldEdgeCount']}"
            )
    exact_at = time.monotonic()
    if chosen is None:
        failure_path = output / "physical-ribbon-cumulative-hole-replay-failures.json"
        atomic_json(
            failure_path,
            {
                "schema": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_SCHEMA,
                "version": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_VERSION,
                "state": "no-valid-exact-state",
                "identity": identity,
                "proposals": [_proposal_record(value) for value in proposals],
                "exactAttempts": attempts,
            },
        )
        raise RuntimeError(
            "no whole-hole state preserved the cumulative surface; "
            f"see {failure_path}"
        )

    state = chosen["state"]
    strict_surface = chosen["strictSurface"]
    augmented_surface = chosen["augmentedSurface"]
    supplemental = chosen["supplemental"]
    base_triangle = np.asarray(strict_surface["triangleFrontierIndex"], dtype=np.int32)
    augmented_triangle = np.asarray(augmented_surface["triangleFrontierIndex"], dtype=np.int32)
    supplemental_triangle = supplemental["supplementalTriangleFrontierIndex"]
    loops = chosen["loops"]
    arrays: dict[str, np.ndarray] = {
        **{
            key: np.asarray(value)
            for key, value in strict_surface.items()
            if key not in {"triangleFrontierIndex", "triangleAreaVoxelsSquared", "triangleNormalResidualDegrees"}
        },
        "selected": np.asarray(chosen["selected"], dtype=np.uint8),
        "component": np.asarray(chosen["component"], dtype=np.int32),
        "triangleFrontierIndex": augmented_triangle,
        "triangleAreaVoxelsSquared": np.asarray(augmented_surface["triangleAreaVoxelsSquared"], dtype=np.float32),
        "triangleNormalResidualDegrees": np.asarray(augmented_surface["triangleNormalResidualDegrees"], dtype=np.float32),
        "triangleSupplementalCtFace": np.concatenate((np.zeros(len(base_triangle), dtype=np.uint8), np.ones(len(supplemental_triangle), dtype=np.uint8))),
        "triangleMinimumCorridorPathFace": np.concatenate((np.zeros(len(base_triangle), dtype=np.uint8), supplemental["supplementalTriangleMinimumPath"])),
        "triangleCtNormalResidualDegrees": np.concatenate((np.full(len(base_triangle), np.nan, dtype=np.float32), supplemental["supplementalTriangleCtNormalResidualDegrees"])),
        "baseStrictTriangleCount": np.asarray([len(base_triangle)], dtype=np.int64),
        "chosenHoleProposalRow": np.asarray(state["rows"], dtype=np.int32),
        "chosenHoleLoopIndex": np.asarray([int(np.asarray(holes["reconfigurationLoopIndex"])[row]) for row in state["rows"]], dtype=np.int32),
        "triangleRegion": _triangle_region_labels(augmented_triangle),
        "loopOffset": np.asarray(loops["loopOffset"], dtype=np.int64),
        "loopVertexFrontierIndex": np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32),
        "loopKind": np.asarray(loops["loopKind"], dtype=np.uint8),
        "loopTriangleRegion": np.asarray(loops["loopTriangleRegion"], dtype=np.int32),
        **supplemental,
    }
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_SCHEMA,
        "version": PHYSICAL_RIBBON_CUMULATIVE_HOLE_REPLAY_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "proposalScreen": {
            "proposalCount": len(proposals),
            "eligibleProposalCount": sum(value["eligible"] for value in proposals),
            "proposals": [_proposal_record(value) for value in proposals],
        },
        "optimization": {
            "stateCount": len(states),
            "exactAttemptCount": len(attempts),
            "exactAttempts": attempts,
            "chosenProposalRows": [int(value) for value in state["rows"]],
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
            "strictTriangleCountBefore": int(np.asarray(prior["baseStrictTriangleCount"]).reshape(-1)[0]),
            "strictTriangleCountAfter": len(base_triangle),
            "supplementalCtFaceCountBefore": int(np.count_nonzero(prior["triangleSupplementalCtFace"])),
            "supplementalCtFaceCountAfter": len(supplemental_triangle),
            "augmentedTriangleCountBefore": len(prior["triangleFrontierIndex"]),
            "augmentedTriangleCountAfter": len(augmented_triangle),
            "triangleRegionCountBefore": len(np.unique(_triangle_region_labels(np.asarray(prior["triangleFrontierIndex"], dtype=np.int32)))),
            "triangleRegionCountAfter": len(np.unique(_triangle_region_labels(augmented_triangle))),
            "preservedPriorConnectionCount": sum(len(value["rows"]) for value in prior_catalogs),
            "connectionCatalogs": prior_catalogs,
            "holeClosure": chosen["closure"],
            "affectedComponents": chosen["componentArea"],
            "minimumAffectedComponentAreaRetention": round(float(chosen["minimumAreaRetention"]), 6),
            "affectedComponentsAugmented": chosen["augmentedComponentArea"],
            "minimumAffectedComponentAugmentedAreaRetention": round(float(chosen["minimumAugmentedAreaRetention"]), 6),
            "manifold": chosen["manifold"],
            "loops": chosen["loopStats"],
            "pathRecords": chosen["pathRecords"],
        },
        "timingSeconds": {
            "proposalOptimization": round(optimized_at - started, 6),
            "exactCumulativeReplay": round(exact_at - optimized_at, 6),
            "writing": round(finished - exact_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "decisionUnit": "one complete CT-supported closed boundary loop",
            "connectionMemory": "all inherited CT corridors are rebuilt in the final shared charts",
            "singleCellGrowth": False,
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
