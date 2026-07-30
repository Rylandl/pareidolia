from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .sheet_correspondence import MODE_CORRESPONDENCE_SCHEMA
from .sheet_evidence import BlockSheetEvidence, read_block_sheet_evidence
from .sheet_stitching import SheetMatchingPolicy
from .topology import GridFace, cell_face


SHEET_FACTOR_SCHEMA = "pareidolia.cubical-sheet-configuration-factors"
SHEET_FACTOR_VERSION = 1
SHEET_FACTOR_STEM = "sheet-configuration-factors-v1"


@dataclass(frozen=True, slots=True)
class SheetFactorSettings:
    quarter_turn_penalty: float = 0.75

    def __post_init__(self) -> None:
        if not math.isfinite(self.quarter_turn_penalty) or self.quarter_turn_penalty < 0:
            raise ValueError("quarter-turn factor penalty must be finite and nonnegative")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _better_alignment(
    first: tuple[float, int, int],
    second: tuple[float, int, int],
) -> tuple[float, int, int]:
    if first[0] > second[0] + 1.0e-12:
        return first
    if second[0] > first[0] + 1.0e-12:
        return second
    if first[1] != second[1]:
        return first if first[1] > second[1] else second
    return first if first[2] <= second[2] else second


def _ordered_alignment_factor(
    first_modes: tuple[int, ...],
    second_modes: tuple[int, ...],
    edges: dict[tuple[int, int], tuple[float, bool]],
) -> tuple[float, int, int]:
    """Return maximum benefit, match count, and quarter-turn count."""

    first_count = len(first_modes)
    second_count = len(second_modes)
    benefit = np.zeros((first_count + 1, second_count + 1), dtype=np.float64)
    matched = np.zeros((first_count + 1, second_count + 1), dtype=np.uint16)
    quarter = np.zeros((first_count + 1, second_count + 1), dtype=np.uint16)
    for first_index in range(1, first_count + 1):
        for second_index in range(1, second_count + 1):
            chosen = _better_alignment(
                (
                    float(benefit[first_index - 1, second_index]),
                    int(matched[first_index - 1, second_index]),
                    int(quarter[first_index - 1, second_index]),
                ),
                (
                    float(benefit[first_index, second_index - 1]),
                    int(matched[first_index, second_index - 1]),
                    int(quarter[first_index, second_index - 1]),
                ),
            )
            edge = edges.get(
                (first_modes[first_index - 1], second_modes[second_index - 1])
            )
            if edge is not None:
                edge_benefit, edge_quarter = edge
                diagonal = (
                    float(benefit[first_index - 1, second_index - 1])
                    + edge_benefit,
                    int(matched[first_index - 1, second_index - 1]) + 1,
                    int(quarter[first_index - 1, second_index - 1])
                    + int(edge_quarter),
                )
                chosen = _better_alignment(chosen, diagonal)
            benefit[first_index, second_index] = chosen[0]
            matched[first_index, second_index] = chosen[1]
            quarter[first_index, second_index] = chosen[2]
    return (
        float(benefit[first_count, second_count]),
        int(matched[first_count, second_count]),
        int(quarter[first_count, second_count]),
    )


def _load_correspondence_arrays(
    root: Path,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "sheet-mode-correspondences-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != MODE_CORRESPONDENCE_SCHEMA
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete mode correspondence catalog")
    data_path = root / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("mode correspondence content hash mismatch")
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
        raise ValueError("mode correspondence catalog lacks factor arrays")
    count = len(arrays["firstModeId"])
    if any(
        arrays[name].shape
        != ((count, 3) if name == "faceAnchorXYZ" else (count,))
        for name in required[1:]
    ):
        raise ValueError("mode correspondence arrays are misaligned")
    return arrays, manifest


def _configuration_memberships(
    evidence: BlockSheetEvidence,
) -> tuple[tuple[int, ...], ...]:
    offset = np.asarray(evidence.arrays["configurationModeOffset"], dtype=np.uint64)
    mode_id = np.asarray(evidence.arrays["configurationModeId"], dtype=np.uint64)
    return tuple(
        tuple(int(value) for value in mode_id[int(low):int(high)])
        for low, high in zip(offset[:-1], offset[1:])
    )


def compile_sheet_configuration_factors(
    evidence_root: str | Path,
    correspondence_root: str | Path,
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    settings: SheetFactorSettings | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Compile exact per-face factors for every adjacent stack pair."""

    started = time.monotonic()
    resolved = settings or SheetFactorSettings()
    evidence_path = Path(evidence_root).resolve()
    correspondence_path = Path(correspondence_root).resolve()
    cluster = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    policy = SheetMatchingPolicy.from_cluster_root(cluster)
    correspondence_arrays, correspondence_manifest = _load_correspondence_arrays(
        correspondence_path
    )
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_FACTOR_SCHEMA,
        "version": SHEET_FACTOR_VERSION,
        "evidenceRoot": str(evidence_path),
        "evidenceManifestSha256": sha256_file(
            evidence_path / "sheet-evidence-v1.json"
        ),
        "correspondenceRoot": str(correspondence_path),
        "correspondenceIdentitySha256": correspondence_manifest["identity"][
            "identitySha256"
        ],
        "correspondenceDataSha256": correspondence_manifest["data"]["sha256"],
        "policy": policy.record(),
        "settings": resolved.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_factors.py",
                "sheet_evidence.py",
                "sheet_correspondence.py",
                "sheet_stitching.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_FACTOR_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet-factor output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_FACTOR_SCHEMA,
            "version": SHEET_FACTOR_VERSION,
            "state": "compiling",
            "identity": identity,
        },
    )
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    memberships = _configuration_memberships(evidence)
    cells = tuple(
        tuple(int(value) for value in row)
        for row in evidence.arrays["cellXYZ"]
    )
    cell_index = {cell: index for index, cell in enumerate(cells)}
    config_offset = np.asarray(evidence.arrays["configurationOffset"], dtype=np.uint64)
    current = np.asarray(evidence.arrays["configurationIsCurrent"], dtype=np.uint8)
    current_config = {
        cell_index_value: int(low) + int(np.flatnonzero(current[int(low):int(high)])[0])
        for cell_index_value, (low, high) in enumerate(
            zip(config_offset[:-1], config_offset[1:])
        )
    }
    patch_by_id = {
        value.patch_id: value for value in evidence.mode_patches.to_patches()
    }
    trace_faces = {
        patch_id: {trace.face for trace in patch.traces}
        for patch_id, patch in patch_by_id.items()
    }
    edges_by_face: dict[
        GridFace, dict[tuple[int, int], tuple[float, bool]]
    ] = defaultdict(dict)
    unmatched_cost = policy.strict_settings.unmatched_negative_log_likelihood
    for first, second, axis, anchor, family, negative_log_likelihood in zip(
        correspondence_arrays["firstModeId"],
        correspondence_arrays["secondModeId"],
        correspondence_arrays["faceAxis"],
        correspondence_arrays["faceAnchorXYZ"],
        correspondence_arrays["family"],
        correspondence_arrays["negativeLogLikelihood"],
    ):
        face = GridFace(int(axis), tuple(int(value) for value in anchor))
        quarter = bool(int(family))
        benefit = (
            2.0 * unmatched_cost
            - float(negative_log_likelihood)
            - (resolved.quarter_turn_penalty if quarter else 0.0)
        )
        edges_by_face[face][(int(first), int(second))] = (benefit, quarter)

    face_records: list[tuple[GridFace, int, int, int, int]] = []
    for lower in sorted(cell_index, key=lambda value: (value[2], value[1], value[0])):
        for axis in range(3):
            upper_values = list(lower)
            upper_values[axis] += 1
            upper = tuple(upper_values)
            if upper not in cell_index:
                continue
            first_cell_index = cell_index[lower]
            second_cell_index = cell_index[upper]
            first_start = int(config_offset[first_cell_index])
            first_count = int(config_offset[first_cell_index + 1]) - first_start
            second_start = int(config_offset[second_cell_index])
            second_count = int(config_offset[second_cell_index + 1]) - second_start
            face_records.append(
                (
                    cell_face(lower, axis, 1),
                    first_cell_index,
                    second_cell_index,
                    first_start,
                    second_start,
                )
            )
    pair_offset = np.zeros(len(face_records) + 1, dtype=np.uint64)
    pair_benefit: list[float] = []
    pair_matched: list[int] = []
    pair_unmatched: list[int] = []
    pair_quarter: list[int] = []
    current_benefit = 0.0
    current_matched = 0
    current_unmatched = 0
    for face_index, (
        face,
        first_cell_index,
        second_cell_index,
        first_start,
        second_start,
    ) in enumerate(face_records):
        first_stop = int(config_offset[first_cell_index + 1])
        second_stop = int(config_offset[second_cell_index + 1])
        edges = edges_by_face.get(face, {})
        first_sequences = {
            configuration_index: tuple(
                mode_id
                for mode_id in memberships[configuration_index]
                if face in trace_faces[mode_id]
            )
            for configuration_index in range(first_start, first_stop)
        }
        second_sequences = {
            configuration_index: tuple(
                mode_id
                for mode_id in memberships[configuration_index]
                if face in trace_faces[mode_id]
            )
            for configuration_index in range(second_start, second_stop)
        }
        for first_configuration in range(first_start, first_stop):
            first_modes = first_sequences[first_configuration]
            for second_configuration in range(second_start, second_stop):
                second_modes = second_sequences[second_configuration]
                benefit, matched, quarter = _ordered_alignment_factor(
                    first_modes,
                    second_modes,
                    edges,
                )
                unmatched = len(first_modes) + len(second_modes) - 2 * matched
                pair_benefit.append(benefit)
                pair_matched.append(matched)
                pair_unmatched.append(unmatched)
                pair_quarter.append(quarter)
                if (
                    first_configuration == current_config[first_cell_index]
                    and second_configuration == current_config[second_cell_index]
                ):
                    current_benefit += benefit
                    current_matched += matched
                    current_unmatched += unmatched
        pair_offset[face_index + 1] = len(pair_benefit)
        if progress is not None and (
            face_index == 0
            or (face_index + 1) % 500 == 0
            or face_index + 1 == len(face_records)
        ):
            progress(face_index + 1, len(face_records))
    compiled = time.monotonic()
    data_path = output / f"{SHEET_FACTOR_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            faceAxis=np.asarray([value[0].axis for value in face_records], dtype=np.int8),
            faceAnchorXYZ=np.asarray(
                [value[0].anchor_xyz for value in face_records], dtype=np.int32
            ).reshape(len(face_records), 3),
            firstCellIndex=np.asarray(
                [value[1] for value in face_records], dtype=np.uint32
            ),
            secondCellIndex=np.asarray(
                [value[2] for value in face_records], dtype=np.uint32
            ),
            firstConfigurationStart=np.asarray(
                [value[3] for value in face_records], dtype=np.uint32
            ),
            firstConfigurationCount=np.asarray(
                [
                    int(config_offset[value[1] + 1]) - value[3]
                    for value in face_records
                ],
                dtype=np.uint16,
            ),
            secondConfigurationStart=np.asarray(
                [value[4] for value in face_records], dtype=np.uint32
            ),
            secondConfigurationCount=np.asarray(
                [
                    int(config_offset[value[2] + 1]) - value[4]
                    for value in face_records
                ],
                dtype=np.uint16,
            ),
            pairOffset=pair_offset,
            pairJoinBenefit=np.asarray(pair_benefit, dtype=np.float32),
            pairMatchedTraceCount=np.asarray(pair_matched, dtype=np.uint16),
            pairUnmatchedTraceCount=np.asarray(pair_unmatched, dtype=np.uint16),
            pairQuarterTurnCount=np.asarray(pair_quarter, dtype=np.uint16),
        )
    temporary.replace(data_path)
    written = time.monotonic()
    benefits = np.asarray(pair_benefit, dtype=np.float64)
    statistics = {
        "interiorFaces": len(face_records),
        "configurationPairs": len(pair_benefit),
        "pairsWithMatches": int(np.sum(np.asarray(pair_matched) > 0)),
        "pairsWithPositiveBenefit": int(np.sum(benefits > 0.0)),
        "joinBenefitQuantiles": {
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "median", "p90", "p99", "maximum"),
                np.percentile(benefits, (0, 50, 90, 99, 100))
                if len(benefits)
                else (0.0, 0.0, 0.0, 0.0, 0.0),
            )
        },
        "sourceBankAnchorConfigurationState": {
            "totalJoinBenefit": round(current_benefit, 6),
            "matchedTraces": current_matched,
            "unmatchedTraceEndpoints": current_unmatched,
            "retainedTraceFraction": round(
                2 * current_matched / max(2 * current_matched + current_unmatched, 1),
                6,
            ),
        },
    }
    data = {
        "path": data_path.name,
        "bytes": data_path.stat().st_size,
        "sha256": sha256_file(data_path),
    }
    summary = {
        "schema": "pareidolia.cubical-sheet-configuration-factor-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": statistics,
        "data": data,
        "semantics": {
            "selection": "none",
            "factor": (
                "exact maximum-benefit order-preserving alignment for each "
                "adjacent pair of physical cell-stack configurations"
            ),
            "joinBenefit": (
                "two unmatched trace costs minus match NLL and optional "
                "quarter-turn penalty"
            ),
        },
        "timingSeconds": {
            "loadingAndIndexing": round(compiled - started, 6),
            "writing": round(written - compiled, 6),
            "total": round(written - started, 6),
        },
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_FACTOR_SCHEMA,
            "version": SHEET_FACTOR_VERSION,
            "state": "complete",
            "identity": identity,
            "statistics": statistics,
            "data": data,
            "summary": summary_path.name,
            "elapsedSeconds": summary["timingSeconds"]["total"],
        },
    )
    return summary
