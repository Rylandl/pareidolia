from __future__ import annotations

import json
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .sheet_correspondence import (
    MODE_CORRESPONDENCE_SCHEMA,
    MODE_CORRESPONDENCE_STEM,
    MODE_CORRESPONDENCE_VERSION,
)
from .sheet_configuration_solver import _read_factors
from .sheet_evidence import BlockSheetEvidence, read_block_sheet_evidence
from .surface_graph import read_surface_graph
from .topology import GridFace, Int3


SHEET_CORE_AUDIT_SCHEMA = "pareidolia.cubical-sheet-core-audit"
SHEET_CORE_AUDIT_VERSION = 1
SHEET_CORE_AUDIT_STEM = "sheet-core-audit-v1"


def _triple(values: Iterable[int], label: str) -> Int3:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError(f"{label} requires an XYZ triple")
    return result  # type: ignore[return-value]


def _quantiles(values: Iterable[float]) -> dict[str, float]:
    array = np.asarray(tuple(values), dtype=np.float64)
    labels = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    if len(array) == 0:
        return {label: 0.0 for label in labels}
    return {
        label: round(float(value), 6)
        for label, value in zip(
            labels,
            np.percentile(array, (0, 10, 25, 50, 75, 90, 100)),
        )
    }


def _shell_depth(cell: Int3, shape: Int3) -> int:
    return min(
        min(cell[axis], shape[axis] - 1 - cell[axis]) for axis in range(3)
    )


def _face_key(mode_id: int, face: GridFace) -> tuple[int, int, Int3]:
    return int(mode_id), int(face.axis), face.anchor_xyz


def _translated_face(face: GridFace, offset: Int3) -> GridFace:
    return GridFace(
        face.axis,
        tuple(face.anchor_xyz[axis] + offset[axis] for axis in range(3)),
    )


def _configuration_path(root: Path) -> Path:
    for name in (
        "owned-configurations-v1.npz",
        "selected-configurations-v1.npz",
        "sheet-configuration-selection-v1.npz",
    ):
        path = root / name
        if path.is_file():
            return path
    raise ValueError("configuration root lacks a supported selection ledger")


def _read_configuration_ledger(root: Path) -> tuple[Path, dict[Int3, int]]:
    path = _configuration_path(root)
    with np.load(path) as values:
        cells = np.asarray(values["cellXYZ"], dtype=np.int32)
        configuration_ids = np.asarray(values["configurationId"], dtype=np.uint64)
    if cells.shape != (len(configuration_ids), 3):
        raise ValueError("configuration selection cells and IDs are misaligned")
    ledger = {
        tuple(int(value) for value in cell): int(configuration_id)
        for cell, configuration_id in zip(cells, configuration_ids)
    }
    if len(ledger) != len(cells):
        raise ValueError("configuration selection contains duplicate cells")
    return path, ledger


def _read_correspondences(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / f"{MODE_CORRESPONDENCE_STEM}.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != MODE_CORRESPONDENCE_SCHEMA
        or int(manifest.get("version", -1)) != MODE_CORRESPONDENCE_VERSION
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete mode correspondence catalog")
    data_path = root / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("mode correspondence content hash mismatch")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    required = ("firstModeId", "secondModeId", "faceAxis", "faceAnchorXYZ")
    if any(name not in arrays for name in required):
        raise ValueError("mode correspondence catalog lacks required arrays")
    count = len(arrays["firstModeId"])
    if any(len(arrays[name]) != count for name in required[1:]):
        raise ValueError("mode correspondence arrays are misaligned")
    if arrays["faceAnchorXYZ"].shape != (count, 3):
        raise ValueError("mode correspondence face anchors are invalid")
    return arrays, manifest


def _selected_configuration_indices(
    evidence: BlockSheetEvidence,
    ledger: Mapping[Int3, int],
    *,
    core_start: Int3,
    core_shape: Int3,
) -> tuple[dict[Int3, int], dict[Int3, int]]:
    evidence_cells = tuple(
        tuple(int(value) for value in row) for row in evidence.arrays["cellXYZ"]
    )
    evidence_cell_index = {cell: index for index, cell in enumerate(evidence_cells)}
    configuration_ids = np.asarray(
        evidence.arrays["configurationId"], dtype=np.uint64
    )
    configuration_by_id = {
        int(value): index for index, value in enumerate(configuration_ids)
    }
    configuration_offset = np.asarray(
        evidence.arrays["configurationOffset"], dtype=np.uint64
    )
    selected: dict[Int3, int] = {}
    evidence_index_by_local: dict[Int3, int] = {}
    expected = int(np.prod(core_shape))
    if len(ledger) != expected:
        raise ValueError("configuration ledger does not cover the complete core")
    for local_cell, configuration_id in ledger.items():
        if any(
            local_cell[axis] < 0 or local_cell[axis] >= core_shape[axis]
            for axis in range(3)
        ):
            raise ValueError("configuration ledger contains a cell outside the core")
        evidence_cell = tuple(
            local_cell[axis] + core_start[axis] for axis in range(3)
        )
        try:
            cell_index = evidence_cell_index[evidence_cell]
            configuration_index = configuration_by_id[configuration_id]
        except KeyError as error:
            raise ValueError(
                "selected stable configuration is absent from the evidence core"
            ) from error
        if not (
            int(configuration_offset[cell_index])
            <= configuration_index
            < int(configuration_offset[cell_index + 1])
        ):
            raise ValueError("selected configuration belongs to another cell")
        selected[local_cell] = configuration_index
        evidence_index_by_local[local_cell] = cell_index
    return selected, evidence_index_by_local


def _configuration_memberships(
    evidence: BlockSheetEvidence,
) -> tuple[dict[int, tuple[int, ...]], dict[int, tuple[int, ...]]]:
    mode_offset = np.asarray(
        evidence.arrays["configurationModeOffset"], dtype=np.uint64
    )
    mode_id = np.asarray(evidence.arrays["configurationModeId"], dtype=np.uint64)
    modes_by_configuration: dict[int, tuple[int, ...]] = {}
    configurations_by_mode: dict[int, list[int]] = defaultdict(list)
    for configuration_index, (low, high) in enumerate(
        zip(mode_offset[:-1], mode_offset[1:])
    ):
        modes = tuple(int(value) for value in mode_id[int(low) : int(high)])
        modes_by_configuration[configuration_index] = modes
        for value in modes:
            configurations_by_mode[value].append(configuration_index)
    return modes_by_configuration, {
        mode: tuple(values) for mode, values in configurations_by_mode.items()
    }


def _coverage_record(
    selected_covered: float,
    oracle_covered: float,
    total: float,
) -> dict[str, float]:
    return {
        "selectedCoveredEvidenceMass": round(selected_covered, 6),
        "oracleCoveredEvidenceMass": round(oracle_covered, 6),
        "totalEvidenceMass": round(total, 6),
        "selectedEvidenceFraction": round(selected_covered / max(total, 1.0e-12), 6),
        "oracleEvidenceFraction": round(oracle_covered / max(total, 1.0e-12), 6),
        "recoverableEvidenceMass": round(max(oracle_covered - selected_covered, 0.0), 6),
        "recoverableEvidenceFraction": round(
            max(oracle_covered - selected_covered, 0.0) / max(total, 1.0e-12),
            6,
        ),
    }


def audit_sheet_core(
    evidence_root: str | Path,
    correspondence_root: str | Path,
    factor_root: str | Path,
    configuration_root: str | Path,
    graph_root: str | Path,
    output_root: str | Path,
    *,
    core_start_cell_xyz: Int3,
    maximum_hotspots: int = 128,
    force: bool = False,
) -> dict[str, Any]:
    """Measure recoverable evidence and topology holes inside one owned core."""

    started = time.monotonic()
    evidence_path = Path(evidence_root).resolve()
    correspondence_path = Path(correspondence_root).resolve()
    factor_path = Path(factor_root).resolve()
    configuration_path = Path(configuration_root).resolve()
    graph_path = Path(graph_root).resolve()
    output = Path(output_root).resolve()
    if maximum_hotspots <= 0:
        raise ValueError("maximum core-audit hotspots must be positive")
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    block = read_surface_graph(graph_path, verify=True)
    core_start = _triple(core_start_cell_xyz, "core start")
    core_shape = block.grid.shape_cells_xyz
    core_stop = tuple(core_start[axis] + core_shape[axis] for axis in range(3))
    if any(
        core_start[axis] < 0
        or core_stop[axis] > evidence.grid.shape_cells_xyz[axis]
        for axis in range(3)
    ):
        raise ValueError("owned core falls outside the evidence grid")
    if evidence.grid.cell_size_xyz != block.grid.cell_size_xyz:
        raise ValueError("evidence and owned graph cell sizes disagree")
    expected_origin = evidence.grid.vertex_world(core_start)
    if not np.allclose(expected_origin, block.grid.origin_xyz, atol=1.0e-6):
        raise ValueError("owned graph origin does not match its evidence subblock")
    selection_file, ledger = _read_configuration_ledger(configuration_path)
    selected, evidence_index_by_local = _selected_configuration_indices(
        evidence,
        ledger,
        core_start=core_start,
        core_shape=core_shape,
    )
    correspondence_arrays, correspondence_manifest = _read_correspondences(
        correspondence_path
    )
    factor_arrays, factor_manifest = _read_factors(factor_path)
    identity: dict[str, Any] = {
        "schema": SHEET_CORE_AUDIT_SCHEMA,
        "version": SHEET_CORE_AUDIT_VERSION,
        "evidenceRoot": str(evidence_path),
        "evidenceManifestSha256": sha256_file(
            evidence_path / "sheet-evidence-v1.json"
        ),
        "correspondenceRoot": str(correspondence_path),
        "correspondenceIdentitySha256": correspondence_manifest["identity"][
            "identitySha256"
        ],
        "factorRoot": str(factor_path),
        "factorIdentitySha256": factor_manifest["identity"]["identitySha256"],
        "configurationRoot": str(configuration_path),
        "configurationDataSha256": sha256_file(selection_file),
        "graphRoot": str(graph_path),
        "graphManifestSha256": sha256_file(graph_path / "surface-graph-v1.json"),
        "coreStartCellXYZ": list(core_start),
        "coreStopCellXYZExclusive": list(core_stop),
        "maximumHotspots": maximum_hotspots,
        "implementationSha256": sha256_file(Path(__file__).resolve()),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    output_path = output / f"{SHEET_CORE_AUDIT_STEM}.json"
    if output_path.is_file() and not force:
        prior = json.loads(output_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet core audit output belongs to another identity")
        return prior
    output.mkdir(parents=True, exist_ok=True)

    configuration_offset = np.asarray(
        evidence.arrays["configurationOffset"], dtype=np.uint64
    )
    covered_mass = np.asarray(
        evidence.arrays["configurationCoveredEvidenceMass"], dtype=np.float64
    )
    total_mass = np.asarray(
        evidence.arrays["configurationTotalEvidenceMass"], dtype=np.float64
    )
    log_weight = np.asarray(
        evidence.arrays["configurationLogWeight"], dtype=np.float64
    )
    modes_by_configuration, configurations_by_mode = _configuration_memberships(
        evidence
    )
    cell_records: dict[Int3, dict[str, Any]] = {}
    shell_coverage: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for local_cell in sorted(selected, key=lambda value: (value[2], value[1], value[0])):
        cell_index = evidence_index_by_local[local_cell]
        low = int(configuration_offset[cell_index])
        high = int(configuration_offset[cell_index + 1])
        selected_index = selected[local_cell]
        total = float(np.max(total_mass[low:high]))
        oracle = float(np.max(covered_mass[low:high]))
        chosen = float(covered_mass[selected_index])
        coverage = _coverage_record(chosen, oracle, total)
        depth = _shell_depth(local_cell, core_shape)
        record = {
            "cellXYZ": list(local_cell),
            "evidenceCellXYZ": [
                local_cell[axis] + core_start[axis] for axis in range(3)
            ],
            "shellDepth": depth,
            "configurationCount": high - low,
            "selectedConfigurationId": ledger[local_cell],
            "selectedModeCount": len(modes_by_configuration[selected_index]),
            **coverage,
            "unresolvedInteriorTraceEndpoints": 0,
            "gapClasses": {},
        }
        cell_records[local_cell] = record
        shell_coverage[depth].append(record)

    selected_patch_ids = {value.patch_id for value in block.patches}
    patch_by_id = {value.patch_id: value for value in block.patches}
    component_by_patch = dict(block.component_by_patch)
    component_size = {
        value.component_id: len(value.patch_ids) for value in block.components
    }
    joined_traces = {
        _face_key(patch_id, value.face)
        for value in block.joins
        for patch_id in (value.first_patch_id, value.second_patch_id)
    }
    candidates_by_trace: dict[tuple[int, int, Int3], list[int]] = defaultdict(list)
    for first, second, axis, anchor_values in zip(
        correspondence_arrays["firstModeId"],
        correspondence_arrays["secondModeId"],
        correspondence_arrays["faceAxis"],
        correspondence_arrays["faceAnchorXYZ"],
    ):
        anchor = tuple(int(value) for value in anchor_values)
        face = GridFace(int(axis), anchor)
        lower, upper = face.adjacent_cells()
        if not all(
            all(core_start[a] <= cell[a] < core_stop[a] for a in range(3))
            for cell in (lower, upper)
        ):
            continue
        first_id = int(first)
        second_id = int(second)
        candidates_by_trace[_face_key(first_id, face)].append(second_id)
        candidates_by_trace[_face_key(second_id, face)].append(first_id)

    evidence_modes_by_cell: dict[Int3, list[Any]] = defaultdict(list)
    for patch in evidence.mode_patches.to_patches():
        if all(core_start[a] <= patch.cell_xyz[a] < core_stop[a] for a in range(3)):
            evidence_modes_by_cell[patch.cell_xyz].append(patch)

    evidence_cells = tuple(
        tuple(int(value) for value in row) for row in evidence.arrays["cellXYZ"]
    )
    selected_by_evidence_cell_index = {
        cell_index: selected[local_cell]
        for local_cell, cell_index in evidence_index_by_local.items()
    }
    local_face_count = 0
    selected_local_matches = 0
    selected_local_unmatched = 0
    face_match_count_ceiling = 0
    faces_below_match_count_ceiling = 0
    selected_match_deficits: list[int] = []
    pair_offset = np.asarray(factor_arrays["pairOffset"], dtype=np.uint64)
    pair_matched = np.asarray(
        factor_arrays["pairMatchedTraceCount"], dtype=np.int64
    )
    pair_unmatched = np.asarray(
        factor_arrays["pairUnmatchedTraceCount"], dtype=np.int64
    )
    for face_index, (first_value, second_value) in enumerate(
        zip(factor_arrays["firstCellIndex"], factor_arrays["secondCellIndex"])
    ):
        first_cell_index = int(first_value)
        second_cell_index = int(second_value)
        first_cell = evidence_cells[first_cell_index]
        second_cell = evidence_cells[second_cell_index]
        if not all(
            all(core_start[axis] <= cell[axis] < core_stop[axis] for axis in range(3))
            for cell in (first_cell, second_cell)
        ):
            continue
        first_configuration = selected_by_evidence_cell_index[first_cell_index]
        second_configuration = selected_by_evidence_cell_index[second_cell_index]
        first_start = int(factor_arrays["firstConfigurationStart"][face_index])
        first_count = int(factor_arrays["firstConfigurationCount"][face_index])
        second_start = int(factor_arrays["secondConfigurationStart"][face_index])
        second_count = int(factor_arrays["secondConfigurationCount"][face_index])
        first_local = first_configuration - first_start
        second_local = second_configuration - second_start
        if not 0 <= first_local < first_count or not 0 <= second_local < second_count:
            raise ValueError("selected configuration falls outside its core face factor")
        low = int(pair_offset[face_index])
        high = int(pair_offset[face_index + 1])
        selected_pair = low + first_local * second_count + second_local
        matched = int(pair_matched[selected_pair])
        unmatched = int(pair_unmatched[selected_pair])
        ceiling = int(np.max(pair_matched[low:high]))
        deficit = ceiling - matched
        local_face_count += 1
        selected_local_matches += matched
        selected_local_unmatched += unmatched
        face_match_count_ceiling += ceiling
        faces_below_match_count_ceiling += deficit > 0
        selected_match_deficits.append(deficit)

    gap_counts: Counter[str] = Counter()
    gap_by_depth: dict[int, Counter[str]] = defaultdict(Counter)
    gap_by_component: dict[int, Counter[str]] = defaultdict(Counter)
    alternative_coverage_gain: list[float] = []
    alternative_log_weight_delta: list[float] = []
    configuration_recovery_without_coverage_loss = 0
    for boundary in block.unresolved_interior_traces:
        source_patch = patch_by_id[boundary.patch_id]
        local_face = boundary.trace.face
        evidence_face = _translated_face(local_face, core_start)
        compatible_ids = tuple(
            candidates_by_trace.get(_face_key(boundary.patch_id, evidence_face), ())
        )
        active_compatible = tuple(
            value for value in compatible_ids if value in selected_patch_ids
        )
        source_component = component_by_patch[boundary.patch_id]
        if active_compatible:
            if any(
                _face_key(value, local_face) in joined_traces
                for value in active_compatible
            ):
                classification = "active-compatible-occupied"
            elif any(
                component_by_patch[value] == source_component
                for value in active_compatible
            ):
                classification = "active-compatible-same-component"
            else:
                classification = "active-compatible-open-bridge"
        elif compatible_ids:
            alternative_configurations = {
                configuration_index
                for mode_id in compatible_ids
                for configuration_index in configurations_by_mode.get(mode_id, ())
            }
            if alternative_configurations:
                classification = "inactive-compatible-configuration"
                lower, upper = evidence_face.adjacent_cells()
                target_evidence_cell = (
                    upper if source_patch.cell_xyz == tuple(
                        lower[axis] - core_start[axis] for axis in range(3)
                    ) else lower
                )
                target_local_cell = tuple(
                    target_evidence_cell[axis] - core_start[axis]
                    for axis in range(3)
                )
                current_index = selected[target_local_cell]
                best_coverage = max(
                    float(covered_mass[value]) for value in alternative_configurations
                )
                best_weight = max(
                    float(log_weight[value]) for value in alternative_configurations
                )
                coverage_gain = best_coverage - float(covered_mass[current_index])
                weight_delta = best_weight - float(log_weight[current_index])
                alternative_coverage_gain.append(coverage_gain)
                alternative_log_weight_delta.append(weight_delta)
                if coverage_gain >= -1.0e-6:
                    configuration_recovery_without_coverage_loss += 1
            else:
                classification = "inactive-compatible-no-configuration"
        else:
            lower, upper = evidence_face.adjacent_cells()
            source_evidence_cell = tuple(
                source_patch.cell_xyz[axis] + core_start[axis] for axis in range(3)
            )
            target_cell = upper if source_evidence_cell == lower else lower
            target_crosses = any(
                value.trace_on(evidence_face) is not None
                for value in evidence_modes_by_cell.get(target_cell, ())
            )
            classification = (
                "bank-incompatible" if target_crosses else "bank-misses-face"
            )
        depth = _shell_depth(source_patch.cell_xyz, core_shape)
        gap_counts[classification] += 1
        gap_by_depth[depth][classification] += 1
        gap_by_component[source_component][classification] += 1
        record = cell_records[source_patch.cell_xyz]
        record["unresolvedInteriorTraceEndpoints"] += 1
        local_counts = Counter(record["gapClasses"])
        local_counts[classification] += 1
        record["gapClasses"] = dict(sorted(local_counts.items()))

    coverage_rows = list(cell_records.values())

    def coverage_summary(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
        values = tuple(rows)
        selected_mass = sum(float(value["selectedCoveredEvidenceMass"]) for value in values)
        oracle_mass = sum(float(value["oracleCoveredEvidenceMass"]) for value in values)
        total = sum(float(value["totalEvidenceMass"]) for value in values)
        evidence_cells = tuple(
            value for value in values if float(value["totalEvidenceMass"]) > 1.0e-6
        )
        return {
            "cells": len(values),
            "evidenceBearingCells": len(evidence_cells),
            "selectedCoveredEvidenceMass": round(selected_mass, 6),
            "oracleCoveredEvidenceMass": round(oracle_mass, 6),
            "totalEvidenceMass": round(total, 6),
            "selectedEvidenceFraction": round(selected_mass / max(total, 1.0e-12), 6),
            "oracleEvidenceFraction": round(oracle_mass / max(total, 1.0e-12), 6),
            "recoverableEvidenceMass": round(max(oracle_mass - selected_mass, 0.0), 6),
            "recoverableEvidenceFraction": round(
                max(oracle_mass - selected_mass, 0.0) / max(total, 1.0e-12), 6
            ),
            "emptySelectedEvidenceCells": sum(
                int(value["selectedModeCount"]) == 0 for value in evidence_cells
            ),
            "selectedEvidenceFractionQuantiles": _quantiles(
                float(value["selectedEvidenceFraction"]) for value in evidence_cells
            ),
            "recoverableEvidenceFractionQuantiles": _quantiles(
                float(value["recoverableEvidenceFraction"]) for value in evidence_cells
            ),
        }

    topology_total = sum(gap_counts.values())
    active_recoverable = sum(
        gap_counts[value]
        for value in (
            "active-compatible-occupied",
            "active-compatible-same-component",
            "active-compatible-open-bridge",
        )
    )
    configuration_recoverable = gap_counts["inactive-compatible-configuration"]
    component_rows = []
    for component_id, counts in gap_by_component.items():
        component_rows.append(
            {
                "componentId": component_id,
                "patches": component_size[component_id],
                "unresolvedInteriorTraceEndpoints": sum(counts.values()),
                "gapClasses": dict(sorted(counts.items())),
            }
        )
    component_rows.sort(
        key=lambda value: (
            -int(value["unresolvedInteriorTraceEndpoints"]),
            -int(value["patches"]),
            int(value["componentId"]),
        )
    )
    hotspots = sorted(
        coverage_rows,
        key=lambda value: (
            -int(value["unresolvedInteriorTraceEndpoints"]),
            -float(value["recoverableEvidenceMass"]),
            tuple(value["cellXYZ"]),
        ),
    )[:maximum_hotspots]
    payload = {
        "schema": SHEET_CORE_AUDIT_SCHEMA,
        "version": SHEET_CORE_AUDIT_VERSION,
        "identity": identity,
        "core": {
            "shapeCellsXYZ": list(core_shape),
            "originXYZ": list(block.grid.origin_xyz),
            "evidenceStartCellXYZ": list(core_start),
            "evidenceStopCellXYZExclusive": list(core_stop),
        },
        "evidenceUtilization": {
            "all": coverage_summary(coverage_rows),
            "byShellDepth": {
                str(depth): coverage_summary(rows)
                for depth, rows in sorted(shell_coverage.items())
            },
        },
        "topologyHoles": {
            "unresolvedInteriorTraceEndpoints": topology_total,
            "classification": dict(sorted(gap_counts.items())),
            "activeConfigurationRecoverableEndpoints": active_recoverable,
            "inactiveConfigurationRecoverableEndpoints": configuration_recoverable,
            "candidateBankLimitedEndpoints": (
                gap_counts["bank-incompatible"] + gap_counts["bank-misses-face"]
            ),
            "recoverableEndpointFraction": round(
                (active_recoverable + configuration_recoverable)
                / max(topology_total, 1),
                6,
            ),
            "configurationRecoveryWithoutCoverageLoss": (
                configuration_recovery_without_coverage_loss
            ),
            "configurationAlternativeCoverageGainQuantiles": _quantiles(
                alternative_coverage_gain
            ),
            "configurationAlternativeLogWeightDeltaQuantiles": _quantiles(
                alternative_log_weight_delta
            ),
            "byShellDepth": {
                str(depth): {
                    "unresolvedInteriorTraceEndpoints": sum(counts.values()),
                    "classification": dict(sorted(counts.items())),
                }
                for depth, counts in sorted(gap_by_depth.items())
            },
        },
        "localToGlobalContinuity": {
            "interiorFaces": local_face_count,
            "selectedConfigurationLocalMatches": selected_local_matches,
            "selectedConfigurationLocalUnmatchedEndpoints": selected_local_unmatched,
            "selectedConfigurationProjectedRetainedTraceFraction": round(
                2 * selected_local_matches
                / max(2 * selected_local_matches + selected_local_unmatched, 1),
                6,
            ),
            "globallyRetainedMatches": len(block.joins),
            "globallyUnresolvedEndpoints": len(block.unresolved_interior_traces),
            "globallyRetainedTraceFraction": round(
                2 * len(block.joins)
                / max(2 * len(block.joins) + len(block.unresolved_interior_traces), 1),
                6,
            ),
            "topologyTaxMatches": selected_local_matches - len(block.joins),
            "topologyTaxEndpoints": 2 * (selected_local_matches - len(block.joins)),
            "faceLocalMatchCountCeiling": face_match_count_ceiling,
            "additionalMatchesAtIndependentFaceCeiling": (
                face_match_count_ceiling - selected_local_matches
            ),
            "facesBelowIndependentMatchCountCeiling": faces_below_match_count_ceiling,
            "selectedMatchDeficitPerFaceQuantiles": _quantiles(
                selected_match_deficits
            ),
        },
        "hotspotCells": hotspots,
        "componentsWithMostOpenEndpoints": component_rows[:maximum_hotspots],
        "semantics": {
            "holeUnit": "one unresolved interior trace endpoint",
            "activeCompatible": (
                "a compatible continuation is already selected; topology or "
                "trace allocation prevents retention"
            ),
            "inactiveCompatibleConfiguration": (
                "the immutable bank contains a compatible continuation in at "
                "least one alternative physical stack"
            ),
            "bankIncompatible": (
                "alternative bank modes reach the target face but none passes "
                "the correspondence policy"
            ),
            "bankMissesFace": "no baked target-cell mode reaches the shared face",
            "faceLocalMatchCountCeiling": (
                "sum of each face's best physical-stack pair by match count; "
                "it is an optimistic diagnostic because neighboring faces may "
                "choose inconsistent configurations"
            ),
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(output_path, payload)
    return payload
