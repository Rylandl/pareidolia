from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import (
    RawAcusSettings,
    VolumeSource,
    canonical_json_hash,
    sha256_file,
)
from .continuation import (
    ContinuationSearch,
    discover_mode_continuations,
    evaluate_mode_continuations,
    read_continuation_search,
    write_continuation_search,
)
from .evidence import read_evidence_artifact
from .gaps import analyze_component_gaps
from .mode_bank import load_mode_bank
from .selection import configuration_options
from .stratigraphy import read_configuration_artifact
from .tables import read_patch_shard


def _identity(
    input_identity: str,
    mode_bank_identity: str,
    *,
    maximum_modes_per_gap: int,
    maximum_configurations_per_candidate: int,
    leaf_shape_cells_xyz: tuple[int, int, int],
    reuse_search_sha256: str | None,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-mode-continuation-search-identity",
        "version": 1,
        "inputPipelineIdentitySha256": input_identity,
        "modeBankIdentitySha256": mode_bank_identity,
        "settings": {
            "maximumModesPerGap": maximum_modes_per_gap,
            "maximumConfigurationsPerCandidate": maximum_configurations_per_candidate,
            "leafShapeCellsXYZ": list(leaf_shape_cells_xyz),
        },
        "reuseSearchSha256": reuse_search_sha256,
        "implementationSha256": {
            name: sha256_file(root / name)
            for name in (
                "continuation_search.py",
                "continuation.py",
                "mode_bank.py",
                "stratigraphy.py",
                "gaps.py",
                "matching.py",
                "block.py",
                "geometry.py",
                "selection.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _mode_data_hashes(root: Path) -> dict[str, str]:
    manifest = json.loads((root / "mode-bank.json").read_text())
    return {
        shard_id: str(
            json.loads(
                (
                    root / "shards" / shard_id / "modes-v1.json"
                ).read_text()
            )["data"]["sha256"]
        )
        for shard_id in manifest["shards"]
    }


def run_continuation_search(
    input_root: str | Path,
    mode_bank_root: str | Path,
    output_path: str | Path,
    *,
    component_id: int | None = None,
    reuse_search_path: str | Path | None = None,
    maximum_modes_per_gap: int = 3,
    maximum_configurations_per_candidate: int = 3,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    progress: Callable[[int, int, int, int], None] | None = None,
) -> tuple[dict[str, Any], ContinuationSearch]:
    root = Path(input_root).resolve()
    pipeline = json.loads((root / "pipeline.json").read_text())
    if pipeline.get("state") != "complete":
        raise ValueError("continuation search requires a completed reconstruction")
    input_identity = str(pipeline["identity"]["identitySha256"])
    bank_manifest, mode_tables = load_mode_bank(mode_bank_root, verify=False)
    if (
        bank_manifest["identity"]["inputPipelineIdentitySha256"]
        != input_identity
    ):
        raise ValueError("mode bank was not derived from this reconstruction")
    mode_bank_identity = str(bank_manifest["identity"]["identitySha256"])
    reuse_path = (
        Path(reuse_search_path).resolve()
        if reuse_search_path is not None
        else None
    )
    reuse = None
    if reuse_path is not None:
        reuse_payload = json.loads(reuse_path.read_text())
        reuse_provenance = reuse_payload.get("provenance", {})
        reuse_identity = reuse_provenance.get("identity", {})
        if reuse_identity.get("inputPipelineIdentitySha256") != input_identity:
            raise ValueError("reused search belongs to another input pipeline")
        reuse_bank_root = Path(reuse_provenance["modeBankRoot"]).resolve()
        if _mode_data_hashes(reuse_bank_root) != _mode_data_hashes(
            Path(mode_bank_root).resolve()
        ):
            raise ValueError("reused search has different mode-bank arrays")
        reuse = read_continuation_search(reuse_path)
    identity = _identity(
        input_identity,
        mode_bank_identity,
        maximum_modes_per_gap=maximum_modes_per_gap,
        maximum_configurations_per_candidate=maximum_configurations_per_candidate,
        leaf_shape_cells_xyz=leaf_shape_cells_xyz,
        reuse_search_sha256=(sha256_file(reuse_path) if reuse_path is not None else None),
    )
    identity_sha256 = str(identity["identitySha256"])

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
    block = assemble_surface_hierarchy(
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
    census = analyze_component_gaps(
        block,
        options_by_cell,
        selected_option_ids,
        component_id=component_id,
    )
    discovery = discover_mode_continuations(
        block,
        census,
        mode_tables,
        maximum_modes_per_gap=maximum_modes_per_gap,
    )
    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    settings = RawAcusSettings(**pipeline["identity"]["settings"])
    search = evaluate_mode_continuations(
        block,
        discovery,
        mode_tables,
        evidence_tables,
        source,
        settings,
        selected_options,
        reuse=reuse,
        maximum_configurations_per_candidate=(
            maximum_configurations_per_candidate
        ),
        maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
        progress=progress,
    )
    payload = write_continuation_search(
        output_path,
        search,
        identity_sha256=identity_sha256,
        provenance={
            "identity": identity,
            "inputRoot": str(root),
            "modeBankRoot": str(Path(mode_bank_root).resolve()),
            "directions": "axial/unsigned",
            "candidateSource": "independently fitted full local mode bank",
            "validation": "complete topology-safe hierarchical reassembly per trial",
            "reuseSearch": str(reuse_path) if reuse_path is not None else None,
        },
    )
    return payload, search
