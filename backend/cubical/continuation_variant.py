from __future__ import annotations

import json
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import (
    RawAcusSettings,
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .continuation import (
    apply_recommended_mode_continuations,
    read_continuation_search,
)
from .evidence import read_evidence_artifact
from .export import write_block_obj, write_block_projection_png
from .mode_bank import load_mode_bank
from .selection import configuration_options
from .stratigraphy import read_configuration_artifact
from .tables import PatchTable, read_patch_shard, write_patch_shard


CONTINUATION_VARIANT_SCHEMA = "pareidolia.raw-acus-mode-continuation-variant"
CONTINUATION_VARIANT_VERSION = 1


def _identity(
    input_identity: str,
    mode_bank_identity: str,
    search_sha256: str,
    applied: list[dict[str, Any]],
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": CONTINUATION_VARIANT_SCHEMA,
        "version": CONTINUATION_VARIANT_VERSION,
        "inputPipelineIdentitySha256": input_identity,
        "modeBankIdentitySha256": mode_bank_identity,
        "searchSha256": search_sha256,
        "applied": applied,
        "implementationSha256": {
            name: sha256_file(root / name)
            for name in (
                "continuation_variant.py",
                "continuation.py",
                "stratigraphy.py",
                "mode_bank.py",
                "matching.py",
                "block.py",
                "geometry.py",
                "tables.py",
                "export.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _block_statistics(block: Any) -> dict[str, Any]:
    deferred = Counter(value.reason for value in block.deferred_joins)
    sizes = [len(value.patch_ids) for value in block.components]
    return {
        "selectedPatches": len(block.patches),
        "candidateJoins": len(block.candidate_joins),
        "retainedJoins": len(block.joins),
        "deferredJoins": len(block.deferred_joins),
        "deferredByReason": dict(sorted(deferred.items())),
        "components": len(block.components),
        "largestComponentPatchCount": max(sizes, default=0),
        "componentsAtLeast": {
            str(limit): sum(value >= limit for value in sizes)
            for limit in (10, 25, 50, 100, 150)
        },
        "exteriorTraces": len(block.exterior_traces),
        "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
    }


def _delta(after: dict[str, Any], before: dict[str, Any]) -> dict[str, int]:
    names = (
        "selectedPatches",
        "candidateJoins",
        "retainedJoins",
        "deferredJoins",
        "components",
        "largestComponentPatchCount",
        "exteriorTraces",
        "unresolvedInteriorTraces",
    )
    return {name: int(after[name]) - int(before[name]) for name in names}


def run_continuation_variant(
    input_root: str | Path,
    mode_bank_root: str | Path,
    search_path: str | Path,
    output_root: str | Path,
    *,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    force: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output == root:
        raise ValueError("continuation variant output must differ from its input")
    pipeline = json.loads((root / "pipeline.json").read_text())
    if pipeline.get("state") != "complete":
        raise ValueError("continuation application requires a completed pipeline")
    input_identity = str(pipeline["identity"]["identitySha256"])
    bank_manifest, mode_tables = load_mode_bank(mode_bank_root, verify=False)
    if bank_manifest["identity"]["inputPipelineIdentitySha256"] != input_identity:
        raise ValueError("mode bank was not derived from the input pipeline")
    bank_identity = str(bank_manifest["identity"]["identitySha256"])
    search_file = Path(search_path).resolve()
    search_payload = json.loads(search_file.read_text())
    if (
        search_payload.get("provenance", {})
        .get("identity", {})
        .get("inputPipelineIdentitySha256")
        != input_identity
    ):
        raise ValueError("continuation search was not run on the input pipeline")
    if (
        search_payload.get("provenance", {})
        .get("identity", {})
        .get("modeBankIdentitySha256")
        != bank_identity
    ):
        raise ValueError("continuation search was not run on this mode bank")
    search = read_continuation_search(search_file)
    applied_records = [
        value.record() for value in search.trials if value.recommended
    ]
    identity = _identity(
        input_identity,
        bank_identity,
        sha256_file(search_file),
        applied_records,
    )
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("continuation output belongs to another identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": CONTINUATION_VARIANT_SCHEMA,
        "version": CONTINUATION_VARIANT_VERSION,
        "state": "loading",
        "identity": identity,
        "inputRoot": str(root),
        "modeBankRoot": str(Path(mode_bank_root).resolve()),
        "search": str(search_file),
    }
    atomic_json(manifest_path, manifest)

    tables = [
        read_configuration_artifact(
            root / "shards" / shard_id / "stratigraphies-v1",
            identity_sha256=input_identity,
            verify=False,
        )
        for shard_id in pipeline["shards"]
    ]
    evidence_tables = {
        shard_id: read_evidence_artifact(
            root / "shards" / shard_id / "evidence-v1",
            identity_sha256=input_identity,
            verify=False,
        )
        for shard_id in pipeline["shards"]
    }
    selected_table = read_patch_shard(root / "selected-patches-v1", verify=False)
    baseline = assemble_surface_hierarchy(
        selected_table.grid,
        BlockBounds((0, 0, 0), selected_table.grid.shape_cells_xyz),
        selected_table.to_patches(),
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
    )
    options_by_cell, _ = configuration_options(selected_table.grid, tables)
    with np.load(root / "selection-v1.npz") as values:
        selected_option_ids = {
            tuple(int(item) for item in cell): int(option_id)
            for cell, option_id in zip(values["cellXYZ"], values["optionId"])
        }
    selected_options = {
        cell: next(
            value
            for value in options_by_cell[cell]
            if value.option_id == option_id
        )
        for cell, option_id in selected_option_ids.items()
    }
    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    settings = RawAcusSettings(**pipeline["identity"]["settings"])
    manifest["state"] = "assembling"
    atomic_json(manifest_path, manifest)
    application = apply_recommended_mode_continuations(
        baseline,
        search,
        mode_tables,
        evidence_tables,
        source,
        settings,
        selected_options,
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
    )
    block = application.block

    baseline_index = {
        int(patch_id): index for index, patch_id in enumerate(selected_table.patch_id)
    }
    candidate_by_cell = {
        value.target_cell_xyz: value
        for value in search.discovery.candidates
        if value.candidate_id in application.required_patch_id_by_candidate
    }
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    trial_by_candidate = {
        value.candidate_id: value for value in application.applied_trials
    }
    for patch in block.patches:
        if patch.patch_id in baseline_index:
            index = baseline_index[patch.patch_id]
            configuration_id[patch.patch_id] = int(
                selected_table.configuration_id[index]
            )
            configuration_log_weight[patch.patch_id] = float(
                selected_table.configuration_log_weight[index]
            )
            local_order[patch.patch_id] = int(selected_table.local_order[index])
            normal_family[patch.patch_id] = int(
                selected_table.normal_family[index]
            )
            continue
        candidate = candidate_by_cell[patch.cell_xyz]
        trial = trial_by_candidate[candidate.candidate_id]
        cell_patches = application.replacement_patches_by_cell[patch.cell_xyz]
        configuration_id[patch.patch_id] = 0x80000000 + candidate.candidate_id
        configuration_log_weight[patch.patch_id] = trial.candidate_local_score
        local_order[patch.patch_id] = next(
            index
            for index, value in enumerate(cell_patches)
            if value.patch_id == patch.patch_id
        )
        normal_family[patch.patch_id] = candidate.normal_hypothesis
    table = PatchTable.from_patches(
        block.grid,
        block.patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        table,
        settings={
            "source": "full-mode-bank conditioned continuation",
            "leafShapeCellsXYZ": list(leaf_shape_cells_xyz),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputPipelineIdentitySha256": input_identity,
            "modeBankIdentitySha256": bank_identity,
            "searchSha256": sha256_file(search_file),
        },
        compressed=True,
    )
    selection_payload = {
        "schema": "pareidolia.raw-acus-mode-continuation-selection",
        "version": 1,
        "identitySha256": identity_sha256,
        "verifiedClosedGapCount": application.verified_closed_gap_count,
        "applied": [value.record() for value in application.applied_trials],
        "requiredPatchIdByCandidate": {
            str(key): value
            for key, value in application.required_patch_id_by_candidate.items()
        },
    }
    atomic_json(output / "continuation-selection-v1.json", selection_payload)
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block, output / "projections.png", maximum_components=128
    )
    largest_path = write_block_projection_png(
        block, output / "largest-component.png", maximum_components=1
    )
    top_four_path = write_block_projection_png(
        block, output / "top-4-components.png", maximum_components=4
    )
    top_twelve_path = write_block_projection_png(
        block, output / "top-12-components.png", maximum_components=12
    )
    before = _block_statistics(baseline)
    after = _block_statistics(block)
    delta = _delta(after, before)
    collision_delta = (
        after["deferredByReason"].get("component-cell-collision", 0)
        - before["deferredByReason"].get("component-cell-collision", 0)
    )
    topology_delta = (
        after["deferredByReason"].get("crossing-topology-cycle", 0)
        - before["deferredByReason"].get("crossing-topology-cycle", 0)
    )
    conservative = (
        delta["selectedPatches"] >= 0
        and delta["retainedJoins"] >= 0
        and delta["unresolvedInteriorTraces"] <= 0
        and collision_delta <= 0
        and topology_delta <= 0
    )
    if not conservative:
        raise RuntimeError("combined continuation variant failed conservative gates")
    summary: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-mode-continuation-variant-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputPipelineIdentitySha256": input_identity,
        "modeBankIdentitySha256": bank_identity,
        "grid": patch_manifest["grid"],
        "appliedContinuationCount": len(application.applied_trials),
        "verifiedClosedGapCount": application.verified_closed_gap_count,
        "baseline": before,
        "continued": after,
        "delta": delta,
        "deferredDeltaByReason": {
            "component-cell-collision": collision_delta,
            "crossing-topology-cycle": topology_delta,
        },
        "conservativeGatesPassed": conservative,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "selection": "continuation-selection-v1.json",
            "selectedPatches": "selected-patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponent": largest_path.name,
            "topFourComponents": top_four_path.name,
            "topTwelveComponents": top_twelve_path.name,
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
