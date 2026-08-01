from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import sha256_file
from .physical_ribbon_bridging import _load_npz


_CUMULATIVE_REPLAY_STEMS = (
    "physical-ribbon-complete-strip-replay-v1",
    "physical-ribbon-lineage-strip-replay-v1",
)
_CUMULATIVE_REPLAY_SCHEMAS = {
    "pareidolia.physical-ribbon-complete-strip-replay",
    "pareidolia.physical-ribbon-lineage-strip-replay",
}


def load_cumulative_strip_replay_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    """Load any cumulative strict-plus-CT strip replay artifact."""

    value = Path(root).resolve()
    if value.is_file():
        manifest_path = value
    else:
        candidates = [
            value / f"{stem}.json" for stem in _CUMULATIVE_REPLAY_STEMS
        ]
        existing = [path for path in candidates if path.is_file()]
        if len(existing) != 1:
            raise ValueError(
                "cumulative replay root must contain exactly one supported manifest"
            )
        manifest_path = existing[0]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") not in _CUMULATIVE_REPLAY_SCHEMAS
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "cumulative strips require a complete label-free strip replay"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return manifest_path, manifest, _load_npz(
        data_path, manifest["data"]["sha256"]
    )


def cumulative_face_replay_reference(
    replay_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if replay_manifest["schema"] == (
        "pareidolia.physical-ribbon-complete-strip-replay"
    ):
        return replay_manifest["identity"]["replay"]
    return replay_manifest["identity"]["faceReplay"]


def cumulative_prior_exact_reference(
    replay_manifest: Mapping[str, Any],
    face_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if replay_manifest["schema"] == (
        "pareidolia.physical-ribbon-complete-strip-replay"
    ):
        return face_manifest["identity"]["replay"]
    return replay_manifest["identity"]["priorExactReplay"]


def cumulative_original_strip_reference(
    replay_manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    if replay_manifest["schema"] == (
        "pareidolia.physical-ribbon-complete-strip-replay"
    ):
        return replay_manifest["identity"]["strips"]
    lineage_reference = replay_manifest["identity"]["lineageStrips"]
    lineage_path = Path(lineage_reference["manifestPath"])
    if sha256_file(lineage_path) != lineage_reference["manifestSha256"]:
        raise ValueError("cumulative lineage-strip audit has changed")
    lineage_manifest = json.loads(lineage_path.read_text())
    if lineage_manifest["data"]["sha256"] != lineage_reference["dataSha256"]:
        raise ValueError("cumulative lineage-strip data identity changed")
    return lineage_manifest["identity"]["priorStrips"]
