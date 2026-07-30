from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable

from .contracts import (
    RawAcusSettings,
    ReconstructionWindow,
    atomic_json,
    canonical_json_hash,
    plan_shards,
    sha256_file,
)
from .evidence import read_evidence_artifact
from .raw_acus import read_needle_artifact
from .stratigraphy import (
    MODE_ARTIFACT_SCHEMA,
    build_layer_modes,
    read_mode_artifact,
    write_mode_artifact,
)


MODE_BANK_SCHEMA = "pareidolia.raw-acus-mode-bank"
MODE_BANK_VERSION = 1


def _identity(input_identity_sha256: str) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": MODE_BANK_SCHEMA,
        "version": MODE_BANK_VERSION,
        "inputPipelineIdentitySha256": input_identity_sha256,
        "implementationSha256": {
            name: sha256_file(root / name)
            for name in (
                "mode_bank.py",
                "stratigraphy.py",
                "raw_acus.py",
                "evidence.py",
                "geometry.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _artifact_ready(prefix: Path, identity_sha256: str) -> bool:
    manifest_path = prefix.with_suffix(".json")
    data_path = prefix.with_suffix(".npz")
    if not manifest_path.is_file() or not data_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except (OSError, json.JSONDecodeError):
        return False
    return (
        manifest.get("schema") == MODE_ARTIFACT_SCHEMA
        and manifest.get("identitySha256") == identity_sha256
        and manifest.get("data", {}).get("sha256") == sha256_file(data_path)
    )


def run_mode_bank(
    input_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    progress: Callable[[int, int, str, int], None] | None = None,
) -> dict[str, Any]:
    """Persist all fitted modes from a completed raw-Acus evidence bake."""

    started = time.monotonic()
    source_root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if source_root == output:
        raise ValueError("mode-bank output must differ from the input pipeline")
    pipeline = json.loads((source_root / "pipeline.json").read_text())
    if pipeline.get("state") != "complete":
        raise ValueError("mode-bank construction requires a completed pipeline")
    input_identity = str(pipeline["identity"]["identitySha256"])
    identity = _identity(input_identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "mode-bank.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("mode-bank root belongs to another input/code identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    settings = RawAcusSettings(**pipeline["identity"]["settings"])
    window_values = pipeline["identity"]["window"]
    window = ReconstructionWindow(
        tuple(window_values["originVoxelXYZ"]),
        tuple(window_values["shapeCellsXYZ"]),
    )
    shards = plan_shards(
        window,
        settings,
        tuple(pipeline["identity"]["shardShapeCellsXYZ"]),
    )
    expected_ids = [value.shard_id for value in shards]
    if set(expected_ids) != set(pipeline["shards"]):
        raise ValueError("planned mode-bank shards disagree with the input pipeline")
    manifest: dict[str, Any] = {
        "schema": MODE_BANK_SCHEMA,
        "version": MODE_BANK_VERSION,
        "state": "building",
        "identity": identity,
        "inputRoot": str(source_root),
        "shards": {value: {"state": "pending"} for value in expected_ids},
    }
    if manifest_path.is_file() and not force:
        previous = json.loads(manifest_path.read_text())
        manifest["shards"].update(previous.get("shards", {}))
    atomic_json(manifest_path, manifest)

    total_modes = 0
    cells_with_modes = 0
    total_cells = 0
    for index, shard in enumerate(shards, start=1):
        prefix = output / "shards" / shard.shard_id / "modes-v1"
        if not force and _artifact_ready(prefix, identity_sha256):
            table = read_mode_artifact(
                prefix, identity_sha256=identity_sha256, verify=False
            )
            statistics = json.loads(prefix.with_suffix(".json").read_text())[
                "statistics"
            ]
        else:
            needles = read_needle_artifact(
                source_root / "shards" / shard.shard_id / "needles-v1",
                identity_sha256=input_identity,
                verify=False,
            )
            evidence = read_evidence_artifact(
                source_root / "shards" / shard.shard_id / "evidence-v1",
                identity_sha256=input_identity,
                verify=False,
            )
            table, statistics = build_layer_modes(needles, evidence, settings)
            write_mode_artifact(
                prefix,
                table,
                identity_sha256=identity_sha256,
                shard=shard,
                statistics=statistics,
            )
        total_modes += int(statistics["candidateModeCount"])
        cells_with_modes += int(statistics["cellsWithModes"])
        total_cells += int(statistics["cellCount"])
        manifest["shards"][shard.shard_id] = {
            "state": "complete",
            **statistics,
        }
        atomic_json(manifest_path, manifest)
        if progress is not None:
            progress(index, len(shards), shard.shard_id, table.mode_count)

    summary: dict[str, Any] = {
        "schema": "pareidolia.raw-acus-mode-bank-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputPipelineIdentitySha256": input_identity,
        "statistics": {
            "shards": len(shards),
            "cells": total_cells,
            "cellsWithModes": cells_with_modes,
            "candidateModes": total_modes,
            "meanModesPerCell": round(total_modes / max(total_cells, 1), 6),
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {"modeShards": "shards/*/modes-v1.npz"},
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    atomic_json(manifest_path, manifest)
    return summary


def load_mode_bank(
    root: str | Path,
    *,
    verify: bool = True,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Load a completed bank as shard-id to :class:`LayerModeTable`."""

    path = Path(root).resolve()
    manifest = json.loads((path / "mode-bank.json").read_text())
    if manifest.get("schema") != MODE_BANK_SCHEMA or manifest.get("state") != "complete":
        raise ValueError("mode bank is not complete")
    identity_sha256 = str(manifest["identity"]["identitySha256"])
    tables = {
        shard_id: read_mode_artifact(
            path / "shards" / shard_id / "modes-v1",
            identity_sha256=identity_sha256,
            verify=verify,
        )
        for shard_id in manifest["shards"]
    }
    return manifest, tables
