from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_corridor_frontier import (
    PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM,
)
from .physical_ribbon_corridor_variants import _load_corridor_artifact
from .physical_ribbon_replay_configuration import _load_replay_artifact
from .physical_ribbon_bridging import _load_npz


PHYSICAL_RIBBON_CORRIDOR_SATURATION_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-saturation"
)
PHYSICAL_RIBBON_CORRIDOR_SATURATION_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_SATURATION_STEM = (
    "physical-ribbon-corridor-saturation-v1"
)


def _load_frontier_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("saturation audit requires a complete label-free frontier")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _corridor_fingerprint(
    corridor: Mapping[str, np.ndarray], row: int
) -> str:
    begin, end = (
        int(value)
        for value in np.asarray(corridor["corridorPatchOffset"])[row : row + 2]
    )
    digest = hashlib.sha256()
    for name in (
        "corridorPatchXYZ",
        "corridorPatchNormalXYZ",
        "corridorPatchThicknessVoxels",
    ):
        digest.update(
            np.ascontiguousarray(np.asarray(corridor[name])[begin:end]).tobytes()
        )
    return digest.hexdigest()


def _eligible_fingerprint_rows(
    corridor: Mapping[str, np.ndarray],
) -> dict[str, int]:
    rows = np.flatnonzero(np.asarray(corridor["corridorEvidenceEligible"]) > 0)
    result = {
        _corridor_fingerprint(corridor, int(row)): int(row) for row in rows
    }
    if len(result) != len(rows):
        raise ValueError("eligible corridors do not have unique physical patches")
    return result


def _target_candidate_bank_set(
    frontier: Mapping[str, np.ndarray], row: int
) -> set[int]:
    rows = np.asarray(frontier["targetCorridorRow"], dtype=np.int32)
    matches = np.flatnonzero(rows == row)
    if len(matches) != 1:
        raise ValueError("frontier does not contain one target record for corridor")
    index = int(matches[0])
    begin, end = (
        int(value)
        for value in np.asarray(frontier["targetCorridorCandidateOffset"])[
            index : index + 2
        ]
    )
    return set(
        int(value)
        for value in np.asarray(frontier["targetCorridorCandidateBankIndex"])[
            begin:end
        ]
    )


def _assess_saturation(
    prior_corridor: Mapping[str, np.ndarray],
    prior_frontier: Mapping[str, np.ndarray],
    prior_replay: Mapping[str, np.ndarray],
    current_corridor: Mapping[str, np.ndarray],
    current_frontier: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    prior_rows = _eligible_fingerprint_rows(prior_corridor)
    current_rows = _eligible_fingerprint_rows(current_corridor)
    prior_fingerprint = set(prior_rows)
    current_fingerprint = set(current_rows)
    shared = prior_fingerprint & current_fingerprint
    removed = prior_fingerprint - current_fingerprint
    added = current_fingerprint - prior_fingerprint
    prior_bank = np.asarray(
        prior_frontier["frontierRibbonCandidate"], dtype=np.int32
    )
    current_bank = np.asarray(
        current_frontier["frontierRibbonCandidate"], dtype=np.int32
    )
    frontier_bank_identical = bool(np.array_equal(prior_bank, current_bank))
    strict_edges_identical = bool(
        frontier_bank_identical
        and np.array_equal(
            prior_frontier["edgeFirstFrontierIndex"],
            current_frontier["edgeFirstFrontierIndex"],
        )
        and np.array_equal(
            prior_frontier["edgeSecondFrontierIndex"],
            current_frontier["edgeSecondFrontierIndex"],
        )
    )
    candidate_sets_identical = all(
        _target_candidate_bank_set(prior_frontier, prior_rows[fingerprint])
        == _target_candidate_bank_set(current_frontier, current_rows[fingerprint])
        for fingerprint in shared
    )
    before_selected = np.asarray(prior_frontier["selected"]) > 0
    after_selected = np.asarray(prior_replay["corridorReplaySelected"]) > 0
    if len(before_selected) != len(after_selected):
        raise ValueError("prior replay selection and frontier differ")
    changed_node = np.flatnonzero(before_selected != after_selected)
    before_component = np.asarray(prior_frontier["component"], dtype=np.int32)
    after_component = np.asarray(
        prior_replay["corridorReplayComponent"], dtype=np.int32
    )
    changed_component = set(
        int(value)
        for value in np.concatenate(
            (before_component[changed_node], after_component[changed_node])
        )
        if int(value) >= 0
    )
    current_component = set(
        int(value)
        for value in np.asarray(current_corridor["corridorTopologyComponent"])[
            list(current_rows.values())
        ]
    )
    touched_remaining_component = changed_component & current_component
    chosen_exact = np.asarray(
        prior_replay["corridorChosenExactVariant"], dtype=np.int32
    )
    shared_prior_failure = {
        fingerprint
        for fingerprint in shared
        if chosen_exact[prior_rows[fingerprint]] < 0
    }
    added_frontier_candidate_count = len(set(map(int, current_bank)) - set(map(int, prior_bank)))
    saturated = bool(
        not added
        and len(shared) == len(current_rows)
        and len(shared_prior_failure) == len(current_rows)
        and candidate_sets_identical
        and frontier_bank_identical
        and strict_edges_identical
        and not touched_remaining_component
        and added_frontier_candidate_count == 0
    )
    return {
        "priorEvidenceEligibleCorridorCount": len(prior_rows),
        "currentEvidenceEligibleCorridorCount": len(current_rows),
        "sharedExactPhysicalPatchCount": len(shared),
        "removedPhysicalPatchCount": len(removed),
        "removedPriorCorridorRows": sorted(prior_rows[value] for value in removed),
        "addedPhysicalPatchCount": len(added),
        "addedCurrentCorridorRows": sorted(current_rows[value] for value in added),
        "sharedPriorExactFailureCount": len(shared_prior_failure),
        "sharedCandidateBankSetsIdentical": candidate_sets_identical,
        "frontierBankIdentical": frontier_bank_identical,
        "strictContinuationEdgesIdentical": strict_edges_identical,
        "addedFrontierCandidateCount": added_frontier_candidate_count,
        "changedSelectionNodeCount": len(changed_node),
        "changedTopologyComponents": sorted(changed_component),
        "remainingCorridorsInChangedComponents": sorted(
            touched_remaining_component
        ),
        "candidateClassSaturated": saturated,
        "decision": (
            "skip redundant exact reconstruction; every remaining physical "
            "strip has the same failed exact state and lies outside the "
            "changed topology components"
            if saturated
            else "continue exact reconstruction; the residual problem changed"
        ),
        "identityLabelsUsed": False,
    }


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def run_physical_ribbon_corridor_saturation(
    prior_corridor_root: str | Path,
    prior_frontier_root: str | Path,
    prior_replay_root: str | Path,
    current_corridor_root: str | Path,
    current_frontier_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    prior_corridor_path, prior_corridor_manifest, prior_corridor = (
        _load_corridor_artifact(prior_corridor_root)
    )
    prior_frontier_path, prior_frontier_manifest, prior_frontier = (
        _load_frontier_artifact(prior_frontier_root)
    )
    prior_replay_path, prior_replay_manifest, prior_replay = (
        _load_replay_artifact(prior_replay_root)
    )
    current_corridor_path, current_corridor_manifest, current_corridor = (
        _load_corridor_artifact(current_corridor_root)
    )
    current_frontier_path, current_frontier_manifest, current_frontier = (
        _load_frontier_artifact(current_frontier_root)
    )
    if (
        prior_frontier_manifest["identity"]["corridors"]["dataSha256"]
        != prior_corridor_manifest["data"]["sha256"]
        or prior_replay_manifest["identity"]["frontier"]["dataSha256"]
        != prior_frontier_manifest["data"]["sha256"]
        or current_frontier_manifest["identity"]["corridors"]["dataSha256"]
        != current_corridor_manifest["data"]["sha256"]
    ):
        raise ValueError("saturation inputs do not form two consecutive iterations")
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_SATURATION_VERSION,
        "priorCorridors": _reference(prior_corridor_path, prior_corridor_manifest),
        "priorFrontier": _reference(prior_frontier_path, prior_frontier_manifest),
        "priorReplay": _reference(prior_replay_path, prior_replay_manifest),
        "currentCorridors": _reference(
            current_corridor_path, current_corridor_manifest
        ),
        "currentFrontier": _reference(
            current_frontier_path, current_frontier_manifest
        ),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_SATURATION_STEM}.json"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
        ):
            return cached
    statistics = _assess_saturation(
        prior_corridor,
        prior_frontier,
        prior_replay,
        current_corridor,
        current_frontier,
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_SATURATION_VERSION,
        "state": "complete",
        "identity": identity,
        "statistics": statistics,
        "method": {
            "corridorIdentity": (
                "SHA-256 over exact CT patch positions, normals, and thicknesses"
            ),
            "reuseRule": (
                "reuse a prior exact failure only when its candidate bank and "
                "strict topology are unchanged and its component was untouched"
            ),
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
