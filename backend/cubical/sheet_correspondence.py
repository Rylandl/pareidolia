from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch
from .matching import TraceMatch
from .sheet_evidence import BlockSheetEvidence, read_block_sheet_evidence
from .sheet_stitching import SheetMatchingPolicy, match_sheet_join_candidate
from .surface_graph import join_key
from .topology import GridFace, Int3, cell_face


MODE_CORRESPONDENCE_SCHEMA = "pareidolia.cubical-sheet-mode-correspondences"
MODE_CORRESPONDENCE_VERSION = 1
MODE_CORRESPONDENCE_STEM = "sheet-mode-correspondences-v1"


@dataclass(frozen=True, slots=True)
class ModeCorrespondence:
    match: TraceMatch
    family: str
    currently_active_endpoints: bool

    @property
    def key(self) -> tuple[int, int, int, Int3]:
        return join_key(self.match)


@dataclass(frozen=True, slots=True)
class ModeCorrespondenceCatalog:
    candidates: tuple[ModeCorrespondence, ...]
    interior_face_count: int

    def statistics(self) -> dict[str, Any]:
        degree: Counter[tuple[int, GridFace]] = Counter(
            trace
            for value in self.candidates
            for trace in (
                (value.match.first_patch_id, value.match.face),
                (value.match.second_patch_id, value.match.face),
            )
        )
        values = np.asarray(tuple(degree.values()), dtype=np.int64)
        return {
            "interiorFaces": self.interior_face_count,
            "candidates": len(self.candidates),
            "strictCandidates": sum(
                value.family == "strict" for value in self.candidates
            ),
            "quarterTurnCandidates": sum(
                value.family == "quarter-turn" for value in self.candidates
            ),
            "currentlyActiveEndpointCandidates": sum(
                value.currently_active_endpoints for value in self.candidates
            ),
            "traceResources": len(degree),
            "candidateDegreeQuantiles": {
                name: round(float(value), 4)
                for name, value in zip(
                    ("minimum", "median", "p90", "p99", "maximum"),
                    np.percentile(values, (0, 50, 90, 99, 100))
                    if len(values)
                    else (0.0, 0.0, 0.0, 0.0, 0.0),
                )
            },
        }


def active_mode_ids(evidence: BlockSheetEvidence) -> frozenset[int]:
    arrays = evidence.arrays
    configuration_offset = np.asarray(arrays["configurationOffset"], dtype=np.uint64)
    current = np.asarray(arrays["configurationIsCurrent"], dtype=np.uint8)
    mode_offset = np.asarray(arrays["configurationModeOffset"], dtype=np.uint64)
    mode_id = np.asarray(arrays["configurationModeId"], dtype=np.uint64)
    result: set[int] = set()
    for low, high in zip(configuration_offset[:-1], configuration_offset[1:]):
        indices = np.flatnonzero(current[int(low):int(high)]) + int(low)
        if len(indices) != 1:
            raise ValueError("each evidence cell must have one current configuration")
        index = int(indices[0])
        result.update(
            int(value)
            for value in mode_id[int(mode_offset[index]):int(mode_offset[index + 1])]
        )
    return frozenset(result)


def enumerate_mode_correspondences(
    evidence: BlockSheetEvidence,
    policy: SheetMatchingPolicy,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> ModeCorrespondenceCatalog:
    patches = evidence.mode_patches.to_patches()
    active = active_mode_ids(evidence)
    by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in patches:
        by_cell[patch.cell_xyz].append(patch)
    for values in by_cell.values():
        values.sort(
            key=lambda value: (
                value.estimate.height_from_cell_center,
                value.patch_id,
            )
        )
    face_work: list[tuple[GridFace, tuple[ClippedPatch, ...], tuple[ClippedPatch, ...]]] = []
    for lower in sorted(by_cell, key=lambda value: (value[2], value[1], value[0])):
        for axis in range(3):
            neighbor_values = list(lower)
            neighbor_values[axis] += 1
            upper = tuple(neighbor_values)
            if not evidence.grid.contains_cell(upper) or upper not in by_cell:
                continue
            face = cell_face(lower, axis, 1)
            first = tuple(
                value for value in by_cell[lower] if value.trace_on(face) is not None
            )
            second = tuple(
                value for value in by_cell[upper] if value.trace_on(face) is not None
            )
            if first and second:
                face_work.append((face, first, second))
    candidates: list[ModeCorrespondence] = []
    total = len(face_work)
    for face_index, (face, first_values, second_values) in enumerate(face_work):
        for first in first_values:
            for second in second_values:
                matched = match_sheet_join_candidate(
                    first,
                    second,
                    face,
                    policy,
                    grid=evidence.grid,
                )
                if matched is None:
                    continue
                match, family = matched
                candidates.append(
                    ModeCorrespondence(
                        match,
                        family,
                        first.patch_id in active and second.patch_id in active,
                    )
                )
        if progress is not None and (
            face_index == 0 or (face_index + 1) % 500 == 0 or face_index + 1 == total
        ):
            progress(face_index + 1, total)
    candidates.sort(key=lambda value: value.key)
    if len({value.key for value in candidates}) != len(candidates):
        raise RuntimeError("mode correspondence enumeration produced duplicate edges")
    return ModeCorrespondenceCatalog(tuple(candidates), total)


def _write_catalog(
    output: Path,
    catalog: ModeCorrespondenceCatalog,
) -> dict[str, Any]:
    path = output / f"{MODE_CORRESPONDENCE_STEM}.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    values = catalog.candidates
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstModeId=np.asarray(
                [value.match.first_patch_id for value in values], dtype=np.uint64
            ),
            secondModeId=np.asarray(
                [value.match.second_patch_id for value in values], dtype=np.uint64
            ),
            faceAxis=np.asarray(
                [value.match.face.axis for value in values], dtype=np.int8
            ),
            faceAnchorXYZ=np.asarray(
                [value.match.face.anchor_xyz for value in values], dtype=np.int32
            ).reshape(len(values), 3),
            family=np.asarray(
                [value.family == "quarter-turn" for value in values], dtype=np.uint8
            ),
            currentlyActiveEndpoints=np.asarray(
                [value.currently_active_endpoints for value in values], dtype=np.uint8
            ),
            negativeLogLikelihood=np.asarray(
                [value.match.negative_log_likelihood for value in values],
                dtype=np.float32,
            ),
            score=np.asarray([value.match.score for value in values], dtype=np.float32),
            maximumEndpointZ=np.asarray(
                [
                    max(agreement.z for agreement in value.match.endpoint_agreements)
                    for value in values
                ],
                dtype=np.float32,
            ),
            normalResidualDegrees=np.asarray(
                [math.degrees(value.match.normal_angle_radians) for value in values],
                dtype=np.float32,
            ),
            normalZ=np.asarray(
                [value.match.normal_z for value in values], dtype=np.float32
            ),
            fiberFrameResidualDegrees=np.asarray(
                [
                    np.nan
                    if value.match.fiber_angle_radians is None
                    else math.degrees(value.match.fiber_angle_radians)
                    for value in values
                ],
                dtype=np.float32,
            ),
            fiberZ=np.asarray(
                [
                    np.nan if value.match.fiber_z is None else value.match.fiber_z
                    for value in values
                ],
                dtype=np.float32,
            ),
            reducedChiSquare=np.asarray(
                [value.match.reduced_chi_square for value in values],
                dtype=np.float32,
            ),
        )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def catalog_block_sheet_correspondences(
    evidence_root: str | Path,
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    started = time.monotonic()
    evidence_path = Path(evidence_root).resolve()
    cluster = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    evidence_manifest_path = evidence_path / "sheet-evidence-v1.json"
    evidence_manifest = json.loads(evidence_manifest_path.read_text())
    policy = SheetMatchingPolicy.from_cluster_root(cluster)
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": MODE_CORRESPONDENCE_SCHEMA,
        "version": MODE_CORRESPONDENCE_VERSION,
        "evidenceRoot": str(evidence_path),
        "evidenceManifestSha256": sha256_file(evidence_manifest_path),
        "evidenceDataSha256": evidence_manifest["data"]["sha256"],
        "modePatchManifestSha256": sha256_file(
            evidence_path / "mode-patches-v1.json"
        ),
        "modePatchDataSha256": sha256_file(evidence_path / "mode-patches-v1.npz"),
        "policy": policy.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_correspondence.py",
                "sheet_stitching.py",
                "matching.py",
                "geometry.py",
                "sheet_evidence.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{MODE_CORRESPONDENCE_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("mode-correspondence output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": MODE_CORRESPONDENCE_SCHEMA,
            "version": MODE_CORRESPONDENCE_VERSION,
            "state": "enumerating",
            "identity": identity,
        },
    )
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    loaded = time.monotonic()
    catalog = enumerate_mode_correspondences(
        evidence,
        policy,
        progress=progress,
    )
    enumerated = time.monotonic()
    data = _write_catalog(output, catalog)
    statistics = catalog.statistics()
    summary = {
        "schema": "pareidolia.cubical-sheet-mode-correspondence-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": statistics,
        "data": data,
        "semantics": {
            "modeGeometryMutable": False,
            "directions": "axial/unsigned",
            "selection": (
                "none; this artifact retains every pair-gated mode edge and "
                "does not commit to a cell configuration or face alignment"
            ),
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "enumerating": round(enumerated - loaded, 6),
            "writing": round(time.monotonic() - enumerated, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            "schema": MODE_CORRESPONDENCE_SCHEMA,
            "version": MODE_CORRESPONDENCE_VERSION,
            "state": "complete",
            "identity": identity,
            "statistics": statistics,
            "data": data,
            "summary": summary_path.name,
            "elapsedSeconds": summary["timingSeconds"]["total"],
        },
    )
    return summary
