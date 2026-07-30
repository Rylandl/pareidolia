from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .cell_refinement import (
    CELL_REFINEMENT_DIAGNOSTIC_SCHEMA,
    CELL_REFINEMENT_DIAGNOSTIC_VERSION,
    CELL_REFINEMENT_SELECTION_SCHEMA,
    CELL_REFINEMENT_SELECTION_STEM,
    CELL_REFINEMENT_SELECTION_VERSION,
    ClusterCellContext,
    TopologyReplay,
    evidence_utilization_summary,
    load_cluster_cell_context,
    replay_neighborhood_topology_state,
    topology_utilization_summary,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .surface_graph import (
    component_statistics,
    read_surface_graph,
    write_surface_graph,
)
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import Int3


CELL_REFINEMENT_VARIANT_SCHEMA = "pareidolia.cubical-cell-refinement-variant"
CELL_REFINEMENT_VARIANT_VERSION = 1
CELL_REFINEMENT_VARIANT_STEM = "cell-refinement-variant-v1"
PATCH_PROVENANCE_SCHEMA = "pareidolia.cubical-cell-refinement-patch-provenance"
PATCH_PROVENANCE_VERSION = 1


def _diagnostic_path(value: str | Path) -> Path:
    path = Path(value).resolve()
    if path.is_dir():
        path = path / "cell-refinement-diagnostic-v1.json"
    if not path.is_file():
        raise ValueError(f"cell-refinement diagnostic does not exist: {path}")
    return path


def _ordered_changes(values: Any) -> tuple[dict[str, Any], ...]:
    if not isinstance(values, list) or not values:
        raise ValueError("diagnostic has no accepted cell-refinement changes")
    changes: list[dict[str, Any]] = []
    seen: set[Int3] = set()
    for value in values:
        if not isinstance(value, Mapping):
            raise ValueError("accepted cell-refinement change is malformed")
        cell_values = value.get("cellXYZ")
        if not isinstance(cell_values, list) or len(cell_values) != 3:
            raise ValueError("accepted cell-refinement change lacks a cell")
        cell = tuple(int(item) for item in cell_values)
        if cell in seen:
            raise ValueError("accepted cell-refinement changes repeat a cell")
        seen.add(cell)
        prior = int(value["priorSourceConfigurationIndex"])
        selected = int(value["selectedSourceConfigurationIndex"])
        if prior == selected:
            raise ValueError("accepted cell-refinement change is a no-op")
        changes.append(
            {
                "cellXYZ": list(cell),
                "priorSourceConfigurationIndex": prior,
                "selectedSourceConfigurationIndex": selected,
            }
        )
    changes.sort(
        key=lambda value: (
            value["cellXYZ"][2],
            value["cellXYZ"][1],
            value["cellXYZ"][0],
        )
    )
    return tuple(changes)


def _write_npz(path: Path, **arrays: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _evidence_delta(
    context: ClusterCellContext,
    changes: tuple[dict[str, Any], ...],
) -> dict[str, Any]:
    before = 0.0
    after = 0.0
    total = 0.0
    cells: list[dict[str, Any]] = []
    for value in changes:
        cell = tuple(int(item) for item in value["cellXYZ"])
        prior = int(value["priorSourceConfigurationIndex"])
        selected = int(value["selectedSourceConfigurationIndex"])
        prior_covered, prior_total = context.evidence(cell, prior)
        selected_covered, selected_total = context.evidence(cell, selected)
        if not math.isclose(
            prior_total,
            selected_total,
            rel_tol=0.0,
            abs_tol=1.0e-6,
        ):
            raise ValueError("candidate evidence totals disagree within a cell")
        before += prior_covered
        after += selected_covered
        total += prior_total
        cells.append(
            {
                **value,
                "coveredEvidenceMassBefore": round(prior_covered, 6),
                "coveredEvidenceMassAfter": round(selected_covered, 6),
                "totalEvidenceMass": round(prior_total, 6),
                "evidenceUtilizationBefore": round(
                    prior_covered / max(prior_total, 1.0e-12), 6
                ),
                "evidenceUtilizationAfter": round(
                    selected_covered / max(selected_total, 1.0e-12), 6
                ),
            }
        )
    return {
        "coveredEvidenceMassBefore": round(before, 6),
        "coveredEvidenceMassAfter": round(after, 6),
        "coveredEvidenceMassDelta": round(after - before, 6),
        "totalEvidenceMass": round(total, 6),
        "evidenceUtilizationBefore": round(before / max(total, 1.0e-12), 6),
        "evidenceUtilizationAfter": round(after / max(total, 1.0e-12), 6),
        "cells": cells,
    }


def _accepted_replay_record(diagnostic: Mapping[str, Any]) -> Mapping[str, Any]:
    annealing = diagnostic["topologyAnnealing"]
    coordinated = annealing.get("coordinatedRound")
    if isinstance(coordinated, Mapping) and bool(coordinated.get("accepted")):
        return coordinated["topology"]
    round_two = annealing.get("roundTwo")
    if isinstance(round_two, Mapping) and bool(round_two.get("accepted")):
        return round_two["topology"]
    greedy = annealing.get("supportGreedyRound")
    if isinstance(greedy, Mapping) and bool(greedy.get("accepted")):
        return greedy["topology"]
    round_one = annealing.get("roundOne")
    if isinstance(round_one, Mapping) and bool(round_one.get("accepted")):
        return round_one["topology"]
    raise ValueError("diagnostic does not contain an accepted topology replay")


def _validate_replay(
    replay: TopologyReplay,
    expected: Mapping[str, Any],
    evidence: Mapping[str, Any],
) -> None:
    if replay.summary != dict(expected):
        raise ValueError(
            "accepted diagnostic replay changed under the current implementation; "
            "rerun the diagnostic before materializing it"
        )
    baseline = replay.summary["baseline"]
    refined = replay.summary["replayed"]
    acceptance = replay.summary["acceptance"]
    if not (
        acceptance["topologySafe"]
        and acceptance["allExteriorJoinsPreserved"]
        and acceptance["frozenExteriorConnectivityPreserved"]
        and refined["incidentUnresolvedTraceEndpoints"]
        <= baseline["incidentUnresolvedTraceEndpoints"]
        and refined["globalComponents"] <= baseline["globalComponents"]
    ):
        raise ValueError("accepted refinement no longer satisfies topology invariants")
    if float(evidence["coveredEvidenceMassDelta"]) <= 0.0:
        raise ValueError("accepted refinement does not improve aggregate Acus evidence")


def _patch_table_for_replay(
    context: ClusterCellContext,
    replay: TopologyReplay,
) -> tuple[PatchTable, dict[int, dict[str, int]]]:
    source = read_patch_shard(
        context.materialized_root / "selected-patches-v1",
        verify=True,
    )
    source_row = {
        int(patch_id): index for index, patch_id in enumerate(source.patch_id)
    }
    replacement: dict[int, dict[str, int]] = {}
    for cell in replay.active_cells:
        selected_index = int(replay.selected_by_cell[cell])
        bank, _ = context.owner(cell)
        option = context.option(cell, selected_index)
        family = int(bank.table.normal_hypothesis[selected_index])
        for layer_index, patch in enumerate(option.patches):
            replacement[patch.patch_id] = {
                "configurationId": selected_index,
                "configurationLogWeight": option.log_weight,
                "localOrder": layer_index,
                "normalFamily": family,
            }
    if set(replacement) != set(replay.replacement_patch_ids):
        raise ValueError("replacement patch provenance does not cover replay geometry")

    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    provenance: dict[int, dict[str, int]] = {}
    for patch in replay.block.patches:
        patch_id = patch.patch_id
        cell = patch.cell_xyz
        bank, local_cell = context.owner(cell)
        selected_index = int(replay.selected_by_cell[cell])
        if patch_id in source_row:
            row = source_row[patch_id]
            configuration_id[patch_id] = int(source.configuration_id[row])
            configuration_log_weight[patch_id] = float(
                source.configuration_log_weight[row]
            )
            local_order[patch_id] = int(source.local_order[row])
            normal_family[patch_id] = int(source.normal_family[row])
            predecessor_patch_id = patch_id
            changed = 0
        elif patch_id in replacement:
            record = replacement[patch_id]
            configuration_id[patch_id] = record["configurationId"]
            configuration_log_weight[patch_id] = float(
                record["configurationLogWeight"]
            )
            local_order[patch_id] = record["localOrder"]
            normal_family[patch_id] = record["normalFamily"]
            predecessor_patch_id = 0
            changed = 1
        else:
            raise ValueError("replayed patch has neither predecessor nor candidate provenance")
        provenance[patch_id] = {
            "inputIndex": bank.input_index,
            "localCellX": local_cell[0],
            "localCellY": local_cell[1],
            "localCellZ": local_cell[2],
            "sourceConfigurationIndex": selected_index,
            "layerIndex": local_order[patch_id],
            "replacementThisRound": changed,
            "predecessorPatchId": predecessor_patch_id,
        }
    table = PatchTable.from_patches(
        context.grid,
        replay.block.patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )
    return table, provenance


def _write_patch_provenance(
    output: Path,
    provenance: Mapping[int, Mapping[str, int]],
    *,
    identity_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(provenance)
    data_path = output / "patch-provenance-v1.npz"
    _write_npz(
        data_path,
        patchId=np.asarray(ordered, dtype=np.uint64),
        inputIndex=np.asarray(
            [provenance[value]["inputIndex"] for value in ordered],
            dtype=np.int16,
        ),
        localCellXYZ=np.asarray(
            [
                (
                    provenance[value]["localCellX"],
                    provenance[value]["localCellY"],
                    provenance[value]["localCellZ"],
                )
                for value in ordered
            ],
            dtype=np.int32,
        ).reshape(len(ordered), 3),
        sourceConfigurationIndex=np.asarray(
            [
                provenance[value]["sourceConfigurationIndex"]
                for value in ordered
            ],
            dtype=np.int64,
        ),
        layerIndex=np.asarray(
            [provenance[value]["layerIndex"] for value in ordered],
            dtype=np.int16,
        ),
        replacementThisRound=np.asarray(
            [provenance[value]["replacementThisRound"] for value in ordered],
            dtype=np.uint8,
        ),
        predecessorPatchId=np.asarray(
            [provenance[value]["predecessorPatchId"] for value in ordered],
            dtype=np.uint64,
        ),
    )
    manifest = {
        "schema": PATCH_PROVENANCE_SCHEMA,
        "version": PATCH_PROVENANCE_VERSION,
        "state": "complete",
        "variantIdentitySha256": identity_sha256,
        "counts": {
            "patches": len(ordered),
            "replacementsThisRound": sum(
                provenance[value]["replacementThisRound"] for value in ordered
            ),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(output / "patch-provenance-v1.json", manifest)
    return manifest


def _write_cell_selection(
    context: ClusterCellContext,
    replay: TopologyReplay,
    output: Path,
    *,
    identity_sha256: str,
    patch_manifest_sha256: str,
    patch_data_sha256: str,
    graph_manifest_sha256: str,
    graph_data_sha256: str,
) -> dict[str, Any]:
    ordered = sorted(
        context.owner_by_cell,
        key=lambda value: (value[2], value[1], value[0]),
    )
    changed = set(replay.active_cells)
    data_path = output / f"{CELL_REFINEMENT_SELECTION_STEM}.npz"
    _write_npz(
        data_path,
        cellXYZ=np.asarray(ordered, dtype=np.int32).reshape(len(ordered), 3),
        inputIndex=np.asarray(
            [context.owner_by_cell[value][0] for value in ordered],
            dtype=np.int16,
        ),
        localCellXYZ=np.asarray(
            [context.owner_by_cell[value][1] for value in ordered],
            dtype=np.int32,
        ).reshape(len(ordered), 3),
        predecessorSourceConfigurationIndex=np.asarray(
            [context.selected_by_cell[value] for value in ordered],
            dtype=np.int64,
        ),
        sourceConfigurationIndex=np.asarray(
            [replay.selected_by_cell[value] for value in ordered],
            dtype=np.int64,
        ),
        changedThisRound=np.asarray(
            [value in changed for value in ordered],
            dtype=np.uint8,
        ),
    )
    source_manifest_path = (
        context.materialized_root / f"{CELL_REFINEMENT_SELECTION_STEM}.json"
    )
    source_round = 0
    if source_manifest_path.is_file():
        source_round = int(
            json.loads(source_manifest_path.read_text()).get("round", 0)
        )
    manifest = {
        "schema": CELL_REFINEMENT_SELECTION_SCHEMA,
        "version": CELL_REFINEMENT_SELECTION_VERSION,
        "state": "complete",
        "variantIdentitySha256": identity_sha256,
        "round": source_round + 1,
        "counts": {
            "cells": len(ordered),
            "changedThisRound": len(changed),
        },
        "graphBindings": {
            "selectedPatchManifestSha256": patch_manifest_sha256,
            "selectedPatchDataSha256": patch_data_sha256,
            "surfaceGraphManifestSha256": graph_manifest_sha256,
            "surfaceGraphDataSha256": graph_data_sha256,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(
        output / f"{CELL_REFINEMENT_SELECTION_STEM}.json",
        manifest,
    )
    return manifest


def _numeric_delta(after: Mapping[str, Any], before: Mapping[str, Any]) -> dict[str, Any]:
    keys = (
        "selectedPatches",
        "retainedJoins",
        "components",
        "unresolvedInteriorTraceEndpoints",
        "retainedInteriorTraceFraction",
    )
    return {
        key: round(float(after[key]) - float(before[key]), 6)
        for key in keys
    }


def run_cell_refinement_materialization(
    cluster_root: str | Path,
    materialized_root: str | Path,
    diagnostic_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Materialize an accepted local refinement as a reloadable graph variant."""

    started = time.monotonic()
    cluster = Path(cluster_root).resolve()
    materialized = Path(materialized_root).resolve()
    diagnostic_path = _diagnostic_path(diagnostic_root)
    output = Path(output_root).resolve()
    if output == materialized:
        raise ValueError("cell-refinement output must differ from its input")
    diagnostic = json.loads(diagnostic_path.read_text())
    if (
        diagnostic.get("schema") != CELL_REFINEMENT_DIAGNOSTIC_SCHEMA
        or int(diagnostic.get("version", -1))
        != CELL_REFINEMENT_DIAGNOSTIC_VERSION
    ):
        raise ValueError("unsupported cell-refinement diagnostic")
    diagnostic_identity = diagnostic["identity"]
    input_hashes = {
        "clusterManifestSha256": sha256_file(
            cluster / "cluster-reselection-v1.json"
        ),
        "clusterDataSha256": sha256_file(cluster / "cluster-reselection-v1.npz"),
        "surfaceGraphManifestSha256": sha256_file(
            materialized / "surface-graph-v1.json"
        ),
        "surfaceGraphDataSha256": sha256_file(
            materialized / "surface-graph-v1.npz"
        ),
        "selectedPatchManifestSha256": sha256_file(
            materialized / "selected-patches-v1.json"
        ),
        "selectedPatchDataSha256": sha256_file(
            materialized / "selected-patches-v1.npz"
        ),
    }
    for name in (
        "clusterManifestSha256",
        "clusterDataSha256",
        "surfaceGraphManifestSha256",
        "surfaceGraphDataSha256",
    ):
        if diagnostic_identity.get(name) != input_hashes[name]:
            raise ValueError("diagnostic does not belong to the supplied graph identity")
    changes = _ordered_changes(
        diagnostic.get("topologyAnnealing", {}).get("acceptedChanges")
    )
    identity: dict[str, Any] = {
        "schema": CELL_REFINEMENT_VARIANT_SCHEMA,
        "version": CELL_REFINEMENT_VARIANT_VERSION,
        "clusterRoot": str(cluster),
        "materializedRoot": str(materialized),
        "diagnostic": str(diagnostic_path),
        "diagnosticSha256": sha256_file(diagnostic_path),
        **input_hashes,
        "acceptedChanges": list(changes),
        "implementationSha256": {
            "cell_refinement_variant.py": sha256_file(Path(__file__)),
            "cell_refinement.py": sha256_file(
                Path(__file__).resolve().parent / "cell_refinement.py"
            ),
            "surface_graph.py": sha256_file(
                Path(__file__).resolve().parent / "surface_graph.py"
            ),
            "tables.py": sha256_file(
                Path(__file__).resolve().parent / "tables.py"
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{CELL_REFINEMENT_VARIANT_STEM}.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("cell-refinement output belongs to another identity")
        if prior.get("state") == "complete" and (output / "summary.json").is_file():
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": CELL_REFINEMENT_VARIANT_SCHEMA,
        "version": CELL_REFINEMENT_VARIANT_VERSION,
        "state": "replaying",
        "identity": identity,
        "inputRoot": str(materialized),
    }
    atomic_json(manifest_path, manifest)

    context = load_cluster_cell_context(cluster, materialized)
    active: set[Int3] = set()
    for value in changes:
        cell = tuple(int(item) for item in value["cellXYZ"])
        if not context.grid.contains_cell(cell):
            raise ValueError("accepted refinement changes a cell outside the grid")
        if context.selected_by_cell[cell] != int(
            value["priorSourceConfigurationIndex"]
        ):
            raise ValueError(
                "accepted refinement prior no longer matches the materialized selection"
            )
        context.option(cell, int(value["selectedSourceConfigurationIndex"]))
        active.add(cell)
    center = tuple(int(value) for value in diagnostic_identity["cellXYZ"])
    if center not in active:
        # Match the annealer's fallback exactly: this label only controls the
        # center-component audit fields when the focal change was rejected.
        center = min(active)
    replay = replay_neighborhood_topology_state(
        context,
        {
            "centerCellXYZ": list(center),
            "radiusCells": int(diagnostic_identity["neighborhoodRadiusCells"]),
            "activeCellXYZ": [
                list(value)
                for value in sorted(active, key=lambda cell: (cell[2], cell[1], cell[0]))
            ],
            "netChanges": list(changes),
        },
    )
    evidence = _evidence_delta(context, changes)
    _validate_replay(
        replay,
        _accepted_replay_record(diagnostic),
        evidence,
    )

    manifest["state"] = "writing"
    atomic_json(manifest_path, manifest)
    table, patch_provenance = _patch_table_for_replay(context, replay)
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        table,
        settings={
            "source": "topology-safe iterative cell refinement",
            "freshInference": False,
            "changedCells": len(changes),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputMaterializedRoot": str(materialized),
            "diagnostic": str(diagnostic_path),
        },
        compressed=True,
    )
    source_graph_manifest = json.loads(
        (materialized / "surface-graph-v1.json").read_text()
    )
    graph_manifest = write_surface_graph(
        output,
        replay.block,
        semantics=str(source_graph_manifest["semantics"]),
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputMaterializedRoot": str(materialized),
            "inputSurfaceGraphManifestSha256": input_hashes[
                "surfaceGraphManifestSha256"
            ],
            "frozenExteriorJoins": replay.summary["frozenExteriorJoinCount"],
            "acceptedChanges": list(changes),
        },
    )
    patch_provenance_manifest = _write_patch_provenance(
        output,
        patch_provenance,
        identity_sha256=identity_sha256,
    )
    selection_manifest = _write_cell_selection(
        context,
        replay,
        output,
        identity_sha256=identity_sha256,
        patch_manifest_sha256=sha256_file(output / "selected-patches-v1.json"),
        patch_data_sha256=sha256_file(output / "selected-patches-v1.npz"),
        graph_manifest_sha256=sha256_file(output / "surface-graph-v1.json"),
        graph_data_sha256=sha256_file(output / "surface-graph-v1.npz"),
    )

    restored = read_surface_graph(output, verify=True)
    if (
        len(restored.patches) != len(replay.block.patches)
        or len(restored.joins) != len(replay.block.joins)
        or len(restored.components) != len(replay.block.components)
    ):
        raise RuntimeError("materialized cell-refinement graph failed round-trip validation")
    refined_context = load_cluster_cell_context(cluster, output)
    if refined_context.selected_by_cell != replay.selected_by_cell:
        raise RuntimeError("materialized cell selection failed round-trip validation")

    baseline_evidence = evidence_utilization_summary(context)
    refined_evidence = evidence_utilization_summary(refined_context)
    baseline_topology = topology_utilization_summary(context)
    refined_topology = topology_utilization_summary(refined_context)
    baseline_components = component_statistics(context.block, maximum_records=32)
    refined_components = component_statistics(restored, maximum_records=32)
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-cell-refinement-variant-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(materialized),
        "outputRoot": str(output),
        "round": selection_manifest["round"],
        "acceptedChangeCount": len(changes),
        "acceptedChanges": evidence["cells"],
        "exactLocalReplay": replay.summary,
        "baseline": {
            "evidence": baseline_evidence,
            "topology": baseline_topology,
            "components": baseline_components,
        },
        "refined": {
            "evidence": refined_evidence,
            "topology": refined_topology,
            "components": refined_components,
        },
        "delta": {
            "selectedCoveredEvidenceMass": round(
                float(refined_evidence["selectedCoveredEvidenceMass"])
                - float(baseline_evidence["selectedCoveredEvidenceMass"]),
                6,
            ),
            "selectedEvidenceUtilization": round(
                float(refined_evidence["selectedEvidenceUtilization"])
                - float(baseline_evidence["selectedEvidenceUtilization"]),
                6,
            ),
            "topology": _numeric_delta(refined_topology, baseline_topology),
            "largestOccupiedCellCount": int(
                refined_components["largestOccupiedCellCount"]
            )
            - int(baseline_components["largestOccupiedCellCount"]),
        },
        "artifacts": {
            "selectedPatchManifest": "selected-patches-v1.json",
            "selectedPatchData": "selected-patches-v1.npz",
            "surfaceGraphManifest": "surface-graph-v1.json",
            "surfaceGraphData": "surface-graph-v1.npz",
            "cellSelectionManifest": f"{CELL_REFINEMENT_SELECTION_STEM}.json",
            "cellSelectionData": f"{CELL_REFINEMENT_SELECTION_STEM}.npz",
            "patchProvenanceManifest": "patch-provenance-v1.json",
            "patchProvenanceData": "patch-provenance-v1.npz",
        },
        "artifactHashes": {
            "selectedPatches": patch_manifest["data"]["sha256"],
            "surfaceGraph": graph_manifest["data"]["sha256"],
            "cellSelection": selection_manifest["data"]["sha256"],
            "patchProvenance": patch_provenance_manifest["data"]["sha256"],
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output / "summary.json", summary)
    manifest.update(
        {
            "state": "complete",
            "summary": "summary.json",
            "round": selection_manifest["round"],
            "elapsedSeconds": summary["timingSeconds"]["total"],
        }
    )
    atomic_json(manifest_path, manifest)
    atomic_json(output / "variant.json", manifest)
    return summary
