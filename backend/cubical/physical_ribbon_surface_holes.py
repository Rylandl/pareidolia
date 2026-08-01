from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    extract_surface_boundary_loops,
    score_surface_patch_holes,
    write_patch_hole_montage,
)
from .physical_ribbon_patch_states import _surface_view


PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA = "pareidolia.physical-ribbon-surface-holes"
PHYSICAL_RIBBON_SURFACE_HOLES_VERSION = 1
PHYSICAL_RIBBON_SURFACE_HOLES_STEM = "physical-ribbon-surface-holes-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonSurfaceHoleSettings(PhysicalRibbonPatchHoleSettings):
    """Select and score complete holes on any materialized surface artifact."""

    maximum_scored_holes: int = 128
    maximum_preview_holes: int = 24
    minimum_scored_boundary_vertex_count: int = 3
    minimum_scored_area_chart_voxels_squared: float = 1.0
    minimum_scored_diameter_mean_boundary_edges: float = 1.0

    def __post_init__(self) -> None:
        PhysicalRibbonPatchHoleSettings.__post_init__(self)
        if self.minimum_scored_boundary_vertex_count < 3:
            raise ValueError("surface holes require at least three boundary vertices")
        if (
            not math.isfinite(self.minimum_scored_area_chart_voxels_squared)
            or self.minimum_scored_area_chart_voxels_squared <= 0.0
            or not math.isfinite(
                self.minimum_scored_diameter_mean_boundary_edges
            )
            or self.minimum_scored_diameter_mean_boundary_edges <= 0.0
        ):
            raise ValueError("surface-hole selection scales must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_surface_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    required = {
        "selected",
        "component",
        "chartUV",
        "triangleFrontierIndex",
        "signedNormalXYZ",
        "midpointXYZ",
        "thicknessVoxels",
    }
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("state") == "complete"
            and required.issubset(manifest.get("data", {}).get("fields", ()))
            and manifest.get("method", {}).get("identityLabelsUsed") is False
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "surface root must identify exactly one complete label-free mesh artifact"
        )
    return matches[0]


def _selected_interior_loops(
    loops: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonSurfaceHoleSettings,
) -> np.ndarray:
    offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    area = np.asarray(loops["loopAreaChartVoxelsSquared"], dtype=np.float32)
    diameter = np.asarray(loops["loopDiameterChartVoxels"], dtype=np.float32)
    mean_edge = np.asarray(loops["loopMeanBoundaryEdgeVoxels"], dtype=np.float32)
    vertex_count = np.diff(offset)
    retained = np.flatnonzero(
        (kind == 1)
        & (vertex_count >= settings.minimum_scored_boundary_vertex_count)
        & (area >= settings.minimum_scored_area_chart_voxels_squared)
        & (
            diameter
            >= settings.minimum_scored_diameter_mean_boundary_edges * mean_edge
        )
    )
    if len(retained):
        retained = retained[
            np.argsort(area[retained], kind="stable")[::-1]
        ][: settings.maximum_scored_holes]
    return retained.astype(np.int32)


def _empty_candidate_arrays(hole_count: int) -> dict[str, np.ndarray]:
    return {
        "patchCandidateOffset": np.zeros(hole_count + 1, dtype=np.int64),
        "patchCandidateFrontierIndex": np.empty(0, dtype=np.int32),
        "patchCandidateNearestPixel": np.empty(0, dtype=np.int32),
        "patchCandidateSurfaceAlignment": np.empty(0, dtype=np.float32),
    }


def build_physical_ribbon_surface_holes(
    surface: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    settings: PhysicalRibbonSurfaceHoleSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    surface_view = _surface_view(surface)
    loops, loop_statistics = extract_surface_boundary_loops(
        surface_view, settings=settings
    )
    selected = _selected_interior_loops(loops, settings=settings)
    scored, scoring_statistics = score_surface_patch_holes(
        surface_view,
        loops,
        source,
        settings=settings,
        loop_indices=selected,
    )
    arrays = {
        **surface_view,
        **loops,
        **scored,
        **_empty_candidate_arrays(len(selected)),
        "candidateBankUsed": np.zeros(1, dtype=np.uint8),
    }
    return arrays, {
        "loops": loop_statistics,
        "scoring": scoring_statistics,
        "selectedInteriorHoleCount": int(len(selected)),
        "candidateBankUsed": False,
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_surface_holes(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonSurfaceHoleSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonSurfaceHoleSettings()
    surface_path, surface_manifest = _resolve_surface_manifest(surface_root)
    surface = _load_npz(
        surface_path.parent / str(surface_manifest["data"]["path"]),
        surface_manifest["data"]["sha256"],
    )
    source_record = surface_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_HOLES_VERSION,
        "surface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": surface_manifest["identity"][
            "topologyContinuity"
        ],
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_SURFACE_HOLES_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_SURFACE_HOLES_STEM}.npz"
    preview_path = output / "physical-ribbon-surface-holes.png"
    if (
        not force
        and manifest_path.is_file()
        and data_path.is_file()
        and preview_path.is_file()
    ):
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    arrays, statistics = build_physical_ribbon_surface_holes(
        surface, source, settings=resolved
    )
    analyzed = time.monotonic()
    _write_npz(data_path, arrays)
    write_patch_hole_montage(
        arrays,
        arrays,
        arrays,
        preview_path,
        maximum_holes=resolved.maximum_preview_holes,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_HOLES_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": surface_manifest.get("geometry", {}),
        "analysis": statistics,
        "timingSeconds": {
            "surfaceHoleAnalysis": round(analyzed - started, 6),
            "writingAndPreview": round(finished - analyzed, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"surfaceHoleMontage": preview_path.name},
        "method": {
            "decisionUnit": "one complete interior boundary loop",
            "surfaceInput": "any materialized label-free intrinsic triangle surface",
            "candidateRole": "no ribbon-bank candidate is required or consulted",
            "rawCtEvidence": "whole-patch air-material-air profiles against same-surface boundary context",
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
