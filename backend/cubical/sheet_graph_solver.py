from __future__ import annotations

import json
import time
from collections import defaultdict
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .block import (
    BlockBounds,
    SurfaceJoinSelection,
    select_surface_joins,
    surface_block_from_retained_joins,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .matching import face_patch_ranks
from .sheet_configuration_solver import SHEET_CONFIGURATION_SOLVER_SCHEMA
from .sheet_evidence import read_block_sheet_evidence
from .sheet_stitching import (
    SheetJoinCandidate,
    SheetJoinCatalog,
    SheetMatchingPolicy,
    SheetStitchingSettings,
    match_sheet_join_candidate,
    restitch_sheet_graph,
)
from .surface_graph import join_key, read_surface_graph, write_surface_graph
from .tables import PatchTable, write_patch_shard
from .topology import GridFace, Int3


SHEET_GRAPH_SOLVER_SCHEMA = "pareidolia.cubical-joint-sheet-graph-replay"
SHEET_GRAPH_SOLVER_VERSION = 1
SHEET_GRAPH_SOLVER_STEM = "joint-sheet-graph-v1"


def _read_configuration_selection(
    root: Path,
    evidence_cell_count: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    manifest_path = root / "sheet-configuration-selection-v1.json"
    data_path = root / "sheet-configuration-selection-v1.npz"
    manifest = json.loads(manifest_path.read_text())
    supported_schemas = {
        SHEET_CONFIGURATION_SOLVER_SCHEMA,
        "pareidolia.cubical-sheet-topology-refinement",
    }
    if (
        manifest.get("schema") not in supported_schemas
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete sheet configuration selection")
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("sheet configuration selection hash mismatch")
    with np.load(data_path) as values:
        selected = np.asarray(values["configurationIndex"], dtype=np.uint32)
    if selected.shape != (evidence_cell_count,):
        raise ValueError("sheet configuration selection does not cover the evidence block")
    return selected, manifest


def _read_correspondences(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "sheet-mode-correspondences-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "complete":
        raise ValueError("mode correspondence catalog is incomplete")
    data_path = root / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("mode correspondence catalog hash mismatch")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    required = (
        "firstModeId",
        "secondModeId",
        "faceAxis",
        "faceAnchorXYZ",
        "family",
        "negativeLogLikelihood",
    )
    if any(name not in arrays for name in required):
        raise ValueError("mode correspondence catalog lacks replay arrays")
    return arrays, manifest


def _active_mode_ids(
    evidence: Any,
    selected: np.ndarray,
) -> tuple[tuple[int, ...], frozenset[int]]:
    offset = np.asarray(evidence.arrays["configurationModeOffset"], dtype=np.uint64)
    mode_id = np.asarray(evidence.arrays["configurationModeId"], dtype=np.uint64)
    by_cell = tuple(
        tuple(int(value) for value in mode_id[int(offset[index]):int(offset[index + 1])])
        for index in selected
    )
    flat = tuple(value for values in by_cell for value in values)
    if len(set(flat)) != len(flat):
        raise ValueError("selected mode IDs are not uniquely owned by cells")
    return by_cell, frozenset(flat)


def _active_join_catalog(
    evidence: Any,
    correspondence_arrays: dict[str, np.ndarray],
    active_ids: frozenset[int],
    policy: SheetMatchingPolicy,
    quarter_turn_penalty: float,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[tuple[Any, ...], SheetJoinCatalog]:
    all_patches = evidence.mode_patches.to_patches()
    patches = tuple(value for value in all_patches if value.patch_id in active_ids)
    patch_by_id = {value.patch_id: value for value in patches}
    first_ids = np.asarray(correspondence_arrays["firstModeId"], dtype=np.uint64)
    second_ids = np.asarray(correspondence_arrays["secondModeId"], dtype=np.uint64)
    active_values = np.asarray(tuple(active_ids), dtype=np.uint64)
    mask = np.isin(first_ids, active_values) & np.isin(second_ids, active_values)
    indices = np.flatnonzero(mask)
    by_cell: dict[Int3, list[Any]] = defaultdict(list)
    for patch in patches:
        by_cell[patch.cell_xyz].append(patch)
    rank_cache: dict[GridFace, tuple[dict[int, int], dict[int, int]]] = {}
    candidates: list[SheetJoinCandidate] = []
    for completed, index_value in enumerate(indices, start=1):
        index = int(index_value)
        first_id = int(first_ids[index])
        second_id = int(second_ids[index])
        face = GridFace(
            int(correspondence_arrays["faceAxis"][index]),
            tuple(int(value) for value in correspondence_arrays["faceAnchorXYZ"][index]),
        )
        family = (
            "quarter-turn"
            if bool(int(correspondence_arrays["family"][index]))
            else "strict"
        )
        matched = match_sheet_join_candidate(
            patch_by_id[first_id],
            patch_by_id[second_id],
            face,
            policy,
            grid=evidence.grid,
        )
        if matched is None or matched[1] != family:
            raise ValueError("persisted mode correspondence changed during replay")
        match, _ = matched
        stored_nll = float(correspondence_arrays["negativeLogLikelihood"][index])
        if abs(match.negative_log_likelihood - stored_nll) > 2.0e-4:
            raise ValueError("persisted mode correspondence likelihood changed")
        if face not in rank_cache:
            lower, upper = face.adjacent_cells()
            first_ranks, second_ranks, _ = face_patch_ranks(
                by_cell.get(lower, ()),
                by_cell.get(upper, ()),
                face,
            )
            rank_cache[face] = first_ranks, second_ranks
        first_ranks, second_ranks = rank_cache[face]
        benefit = (
            2.0 * policy.strict_settings.unmatched_negative_log_likelihood
            - match.negative_log_likelihood
            - (quarter_turn_penalty if family == "quarter-turn" else 0.0)
        )
        candidates.append(
            SheetJoinCandidate(
                match,
                family,
                first_ranks[first_id],
                second_ranks[second_id],
                float(benefit),
                False,
            )
        )
        if progress is not None and (
            completed == 1 or completed % 5000 == 0 or completed == len(indices)
        ):
            progress(completed, len(indices))
    candidates.sort(key=lambda value: value.key)
    faces = {value.match.face for value in candidates}
    return patches, SheetJoinCatalog(tuple(candidates), len(faces), 0)


def replay_joint_sheet_graph(
    evidence_root: str | Path,
    correspondence_root: str | Path,
    configuration_root: str | Path,
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    stitching_settings: SheetStitchingSettings | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    resolved = stitching_settings or SheetStitchingSettings()
    evidence_path = Path(evidence_root).resolve()
    correspondence_path = Path(correspondence_root).resolve()
    configuration_path = Path(configuration_root).resolve()
    cluster = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    selected, configuration_manifest = _read_configuration_selection(
        configuration_path,
        evidence.cell_count,
    )
    correspondence_arrays, correspondence_manifest = _read_correspondences(
        correspondence_path
    )
    policy = SheetMatchingPolicy.from_cluster_root(cluster)
    seed_graph_manifest_path = configuration_path / "surface-graph-v1.json"
    seed_graph_data_path = configuration_path / "surface-graph-v1.npz"
    if seed_graph_manifest_path.is_file() != seed_graph_data_path.is_file():
        raise ValueError("configuration root contains an incomplete seed graph")
    seed_graph_available = seed_graph_manifest_path.is_file()
    identity: dict[str, Any] = {
        "schema": SHEET_GRAPH_SOLVER_SCHEMA,
        "version": SHEET_GRAPH_SOLVER_VERSION,
        "evidenceManifestSha256": sha256_file(
            evidence_path / "sheet-evidence-v1.json"
        ),
        "correspondenceIdentitySha256": correspondence_manifest["identity"][
            "identitySha256"
        ],
        "correspondenceDataSha256": correspondence_manifest["data"]["sha256"],
        "configurationIdentitySha256": configuration_manifest["identity"][
            "identitySha256"
        ],
        "configurationDataSha256": configuration_manifest["data"]["sha256"],
        "clusterManifestSha256": sha256_file(
            cluster / "cluster-reselection-v1.json"
        ),
        "policy": policy.record(),
        "stitchingSettings": resolved.record(),
        "seedGraph": (
            {
                "manifestSha256": sha256_file(seed_graph_manifest_path),
                "dataSha256": sha256_file(seed_graph_data_path),
            }
            if seed_graph_available
            else None
        ),
        "implementationSha256": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in (
                "sheet_graph_solver.py",
                "sheet_stitching.py",
                "sheet_evidence.py",
                "sheet_correspondence.py",
                "block.py",
                "matching.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_GRAPH_SOLVER_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("joint sheet graph output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_GRAPH_SOLVER_SCHEMA,
            "version": SHEET_GRAPH_SOLVER_VERSION,
            "state": "cataloging-active-modes",
            "identity": identity,
        },
    )
    by_cell_mode_ids, active_ids = _active_mode_ids(evidence, selected)
    patches, provisional_catalog = _active_join_catalog(
        evidence,
        correspondence_arrays,
        active_ids,
        policy,
        resolved.quarter_turn_penalty,
        progress=progress,
    )
    cataloged = time.monotonic()
    priorities = {
        value.key: value.benefit
        for value in provisional_catalog.candidates
        if value.benefit > resolved.minimum_join_benefit
    }
    if seed_graph_available:
        seed_block = read_surface_graph(configuration_path, verify=True)
        patch_ids = {value.patch_id for value in patches}
        if {value.patch_id for value in seed_block.patches} != patch_ids:
            raise ValueError("configuration seed graph activates different patches")
        if seed_block.grid != evidence.grid:
            raise ValueError("configuration seed graph uses a different grid")
        seed_keys = {join_key(value) for value in seed_block.joins}
        missing_seed = seed_keys - set(priorities)
        if missing_seed:
            raise ValueError(
                "configuration seed graph references unavailable active joins"
            )
        baseline_selection = SurfaceJoinSelection(tuple(seed_block.joins), tuple())
    else:
        seed_block = None
        baseline_selection = select_surface_joins(
            patches,
            (
                value.match
                for value in provisional_catalog.candidates
                if value.key in priorities
            ),
            candidate_priorities=priorities,
        )
    baseline_keys = {join_key(value) for value in baseline_selection.joins}
    candidates = tuple(
        SheetJoinCandidate(
            value.match,
            value.family,
            value.first_rank,
            value.second_rank,
            value.benefit,
            value.key in baseline_keys,
        )
        for value in provisional_catalog.candidates
    )
    catalog = SheetJoinCatalog(
        candidates,
        provisional_catalog.interior_face_count,
        provisional_catalog.unstable_face_count,
    )
    baseline_block = (
        seed_block
        if seed_block is not None
        else surface_block_from_retained_joins(
            evidence.grid,
            BlockBounds((0, 0, 0), evidence.grid.shape_cells_xyz),
            patches,
            baseline_selection.joins,
        )
    )
    result = restitch_sheet_graph(
        baseline_block,
        catalog,
        policy,
        settings=resolved,
    )
    solved = time.monotonic()

    config_log_weight = np.asarray(
        evidence.arrays["configurationLogWeight"], dtype=np.float32
    )
    config_family = np.asarray(
        evidence.arrays["configurationNormalHypothesis"], dtype=np.int16
    )
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    for cell_index, mode_ids in enumerate(by_cell_mode_ids):
        configuration_index = int(selected[cell_index])
        for order, mode_id in enumerate(mode_ids):
            configuration_id[mode_id] = configuration_index
            configuration_log_weight[mode_id] = float(
                config_log_weight[configuration_index]
            )
            local_order[mode_id] = order
            normal_family[mode_id] = int(config_family[configuration_index])
    patch_table = PatchTable.from_patches(
        evidence.grid,
        result.block.patches,
        configuration_id=configuration_id,
        configuration_log_weight=configuration_log_weight,
        local_order=local_order,
        normal_family=normal_family,
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        patch_table,
        settings={
            "semantics": "selected immutable Acus modes after joint sheet replay"
        },
        provenance={
            "jointSheetGraphIdentitySha256": identity_sha256,
            "configurationSelectionRoot": str(configuration_path),
        },
        compressed=True,
    )
    graph_manifest = write_surface_graph(
        output,
        result.block,
        semantics=(
            "selected Acus configuration modes with complete face alternatives "
            "and topology-safe whole-sheet restitching"
        ),
        provenance={
            "jointSheetGraphIdentitySha256": identity_sha256,
            "configurationSelectionRoot": str(configuration_path),
            "correspondenceRoot": str(correspondence_path),
        },
    )
    selection_path = output / "selected-configurations-v1.npz"
    temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=np.asarray(evidence.arrays["cellXYZ"], dtype=np.int32),
            configurationIndex=selected,
            configurationId=np.asarray(evidence.arrays["configurationId"])[selected],
            inputIndex=np.asarray(evidence.arrays["configurationInputIndex"])[selected],
            sourceConfigurationIndex=np.asarray(
                evidence.arrays["configurationSourceIndex"]
            )[selected],
        )
    temporary.replace(selection_path)
    local_summary = json.loads((configuration_path / "summary.json").read_text())
    locally_matched = int(
        local_summary["selected"]["locallyMatchedFaceTraces"]
    )
    summary = {
        "schema": "pareidolia.cubical-joint-sheet-graph-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "configurationState": local_summary["selected"],
        "activeCatalog": catalog.statistics(),
        "globalRestitch": result.summary,
        "localToGlobalTopologyTax": {
            "locallyMatchedFaceTraces": locally_matched,
            "globallyRetainedJoins": len(result.block.joins),
            "rejectedLocalMatches": locally_matched - len(result.block.joins),
            "survivalFraction": round(
                len(result.block.joins) / max(locally_matched, 1), 6
            ),
        },
        "artifacts": {
            "selectedConfigurations": {
                "path": selection_path.name,
                "bytes": selection_path.stat().st_size,
                "sha256": sha256_file(selection_path),
            },
            "selectedPatches": {
                "manifest": "selected-patches-v1.json",
                "manifestSha256": sha256_file(output / "selected-patches-v1.json"),
                "data": patch_manifest["data"],
            },
            "surfaceGraph": {
                "manifest": "surface-graph-v1.json",
                "manifestSha256": sha256_file(output / "surface-graph-v1.json"),
                "data": graph_manifest["data"],
            },
        },
        "timingSeconds": {
            "activeCatalog": round(cataloged - started, 6),
            "globalSolve": round(solved - cataloged, 6),
            "writingAndVerification": round(time.monotonic() - solved, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_GRAPH_SOLVER_SCHEMA,
            "version": SHEET_GRAPH_SOLVER_VERSION,
            "state": "complete",
            "identity": identity,
            "summary": summary_path.name,
            "elapsedSeconds": summary["timingSeconds"]["total"],
        },
    )
    return summary
