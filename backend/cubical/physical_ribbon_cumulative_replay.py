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
    "physical-ribbon-cumulative-corridor-replay-v1",
    "physical-ribbon-cumulative-hole-replay-v1",
)
_CUMULATIVE_REPLAY_SCHEMAS = {
    "pareidolia.physical-ribbon-complete-strip-replay",
    "pareidolia.physical-ribbon-lineage-strip-replay",
    "pareidolia.physical-ribbon-cumulative-corridor-replay",
    "pareidolia.physical-ribbon-cumulative-hole-replay",
}

_CUMULATIVE_SURFACE_FIELDS = (
    "frontierRibbonCandidate",
    "selected",
    "component",
    "componentSize",
    "signedNormalXYZ",
    "tangentUxyz",
    "tangentVxyz",
    "chartUV",
    "integrationResidualVoxels",
    "edgeFirstFrontierIndex",
    "edgeSecondFrontierIndex",
    "edgeSelected",
    "midpointXYZ",
    "thicknessVoxels",
    "triangleFrontierIndex",
    "triangleAreaVoxelsSquared",
    "triangleNormalResidualDegrees",
)
_CUMULATIVE_OPTIONAL_SURFACE_FIELDS = (
    "triangleSupplementalCtFace",
    "triangleMinimumCorridorPathFace",
    "triangleCtNormalResidualDegrees",
    "baseStrictTriangleCount",
    "supplementalTriangleFrontierIndex",
    "supplementalTrianglePrimaryCorridorRow",
    "supplementalTriangleMinimumPath",
    "supplementalTriangleAreaVoxelsSquared",
    "supplementalTriangleNodeNormalResidualDegrees",
    "supplementalTriangleCtNormalResidualDegrees",
    "supplementalTriangleCenterDistanceThicknesses",
    "supplementalTriangleCenterHeightThicknesses",
    "supplementalTriangleMaximumEdgeThicknesses",
)


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


def load_materialized_cumulative_surface(
    replay_root: str | Path,
    configuration_manifest: Mapping[str, Any],
    configuration: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray], dict[str, Any]]:
    """Load an exact augmented surface coupled to its materialized state."""

    replay_path, replay_manifest, replay = load_cumulative_strip_replay_artifact(
        replay_root
    )
    replay_reference = configuration_manifest.get("identity", {}).get(
        "sourceReplay"
    )
    if replay_reference is None:
        raise ValueError(
            "a cumulative surface requires a configuration materialized from "
            "that exact replay"
        )
    if (
        replay_reference["manifestSha256"] != sha256_file(replay_path)
        or replay_reference["dataSha256"] != replay_manifest["data"]["sha256"]
    ):
        raise ValueError(
            "cumulative surface and materialized configuration identify "
            "different exact replays"
        )
    missing = [name for name in _CUMULATIVE_SURFACE_FIELDS if name not in replay]
    if missing:
        raise ValueError(f"cumulative surface is missing fields: {missing}")
    surface = {
        name: np.asarray(replay[name]).copy()
        for name in _CUMULATIVE_SURFACE_FIELDS
    }
    for name in _CUMULATIVE_OPTIONAL_SURFACE_FIELDS:
        if name in replay:
            surface[name] = np.asarray(replay[name]).copy()
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    if not np.array_equal(surface["frontierRibbonCandidate"], frontier):
        raise ValueError("cumulative surface and materialized frontier differ")
    selected = np.asarray(configuration["selected"], dtype=np.uint8)
    component = np.asarray(configuration["component"], dtype=np.int32)
    if not np.array_equal(surface["selected"], selected):
        raise ValueError("cumulative surface and materialized selection differ")
    if not np.array_equal(surface["component"], component):
        raise ValueError("cumulative surface and materialized components differ")
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    if len(triangles) and (
        np.any(~selected.astype(bool)[triangles])
        or np.any(component[triangles] != component[triangles[:, :1]])
    ):
        raise ValueError(
            "cumulative surface contains a triangle outside one selected sheet"
        )
    strict_count = int(
        np.asarray(surface.get("baseStrictTriangleCount", (len(triangles),)))
        .reshape(-1)[0]
    )
    supplemental_count = int(
        np.count_nonzero(
            np.asarray(
                surface.get(
                    "triangleSupplementalCtFace",
                    np.zeros(len(triangles), dtype=np.uint8),
                )
            )
        )
    )
    statistics = {
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "strictTriangleCount": strict_count,
        "supplementalCtFaceCount": supplemental_count,
        "augmentedTriangleCount": len(triangles),
        "sourceReplaySchema": replay_manifest["schema"],
        "surfaceSource": "cumulative strict-plus-native-CT replay",
        "identityLabelsUsed": False,
    }
    return replay_path, replay_manifest, surface, statistics


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
