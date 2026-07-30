from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .sheet_evidence import BlockSheetEvidence, read_block_sheet_evidence
from .sheet_factors import SHEET_FACTOR_SCHEMA


SHEET_CONFIGURATION_SOLVER_SCHEMA = (
    "pareidolia.cubical-sheet-configuration-initialization"
)
SHEET_CONFIGURATION_SOLVER_VERSION = 1
SHEET_CONFIGURATION_SOLVER_STEM = "sheet-configuration-selection-v1"
PAIRWISE_NORMALIZATIONS = frozenset(("none", "trace-mean"))


@dataclass(frozen=True, slots=True)
class SheetConfigurationSolverSettings:
    unary_scale: float = 1.0
    pairwise_scale: float = 0.2
    coverage_reward_scale: float = 0.0
    unmatched_trace_penalty: float = 0.0
    pairwise_normalization: str = "none"
    maximum_sweeps: int = 12
    belief_propagation_iterations: int = 0
    belief_propagation_damping: float = 0.5
    belief_propagation_tolerance: float = 1.0e-4

    def __post_init__(self) -> None:
        if not math.isfinite(self.unary_scale) or self.unary_scale <= 0.0:
            raise ValueError("sheet-configuration unary scale must be positive")
        if not math.isfinite(self.pairwise_scale) or self.pairwise_scale <= 0.0:
            raise ValueError("sheet-configuration pairwise scale must be positive")
        if (
            not math.isfinite(self.coverage_reward_scale)
            or self.coverage_reward_scale < 0.0
        ):
            raise ValueError("coverage reward scale must be finite and nonnegative")
        if (
            not math.isfinite(self.unmatched_trace_penalty)
            or self.unmatched_trace_penalty < 0.0
        ):
            raise ValueError("unmatched trace penalty must be finite and nonnegative")
        if self.pairwise_normalization not in PAIRWISE_NORMALIZATIONS:
            raise ValueError(
                "pairwise normalization must be one of "
                f"{sorted(PAIRWISE_NORMALIZATIONS)}"
            )
        if self.maximum_sweeps <= 0:
            raise ValueError("maximum sweeps must be positive")
        if self.belief_propagation_iterations < 0:
            raise ValueError("belief-propagation iterations must be nonnegative")
        if (
            not math.isfinite(self.belief_propagation_damping)
            or not 0.0 <= self.belief_propagation_damping < 1.0
        ):
            raise ValueError("belief-propagation damping must lie in [0, 1)")
        if (
            not math.isfinite(self.belief_propagation_tolerance)
            or self.belief_propagation_tolerance <= 0.0
        ):
            raise ValueError("belief-propagation tolerance must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SheetConfigurationSelection:
    configuration_index_by_cell: tuple[int, ...]
    objective: float
    unary_objective: float
    pairwise_objective: float
    matched_trace_count: int
    unmatched_trace_endpoint_count: int
    sweeps: int
    changed_last_sweep: int
    initialization: str


def _read_factors(root: Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest_path = root / "sheet-configuration-factors-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != SHEET_FACTOR_SCHEMA
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete sheet configuration factors")
    data_path = root / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("sheet configuration factor content hash mismatch")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    required = (
        "firstCellIndex",
        "secondCellIndex",
        "firstConfigurationStart",
        "firstConfigurationCount",
        "secondConfigurationStart",
        "secondConfigurationCount",
        "pairOffset",
        "pairJoinBenefit",
        "pairMatchedTraceCount",
        "pairUnmatchedTraceCount",
    )
    if any(name not in arrays for name in required):
        raise ValueError("sheet configuration factors lack required arrays")
    face_count = len(arrays["firstCellIndex"])
    if any(len(arrays[name]) != face_count for name in required[1:6]):
        raise ValueError("sheet configuration face arrays are misaligned")
    pair_offset = np.asarray(arrays["pairOffset"], dtype=np.uint64)
    if (
        pair_offset.shape != (face_count + 1,)
        or int(pair_offset[0]) != 0
        or np.any(np.diff(pair_offset) < 0)
    ):
        raise ValueError("sheet configuration pair offsets are invalid")
    pair_count = int(pair_offset[-1])
    if any(len(arrays[name]) != pair_count for name in required[7:]):
        raise ValueError("sheet configuration pair arrays are misaligned")
    return arrays, manifest


def _source_anchor_selection(evidence: BlockSheetEvidence) -> tuple[int, ...]:
    offset = np.asarray(evidence.arrays["configurationOffset"], dtype=np.uint64)
    current = np.asarray(evidence.arrays["configurationIsCurrent"], dtype=np.uint8)
    result = []
    for low, high in zip(offset[:-1], offset[1:]):
        values = np.flatnonzero(current[int(low):int(high)])
        if len(values) != 1:
            raise ValueError("each evidence cell must identify one source anchor stack")
        result.append(int(low) + int(values[0]))
    return tuple(result)


def _load_initial_selection(
    evidence: BlockSheetEvidence,
    root: Path | None,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    if root is None:
        return _source_anchor_selection(evidence), {
            "kind": "source-bank-anchor",
            "root": None,
        }
    selection_path = root / "cell-refinement-selection-v1.npz"
    manifest_path = root / "cell-refinement-selection-v1.json"
    if not selection_path.is_file() or not manifest_path.is_file():
        raise ValueError(
            "initial sheet configuration root lacks a cell-refinement selection ledger"
        )
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "complete":
        raise ValueError("initial cell-refinement selection is incomplete")
    if sha256_file(selection_path) != manifest["data"]["sha256"]:
        raise ValueError("initial cell-refinement selection hash mismatch")
    with np.load(selection_path) as values:
        cells = np.asarray(values["cellXYZ"], dtype=np.int32)
        input_index = np.asarray(values["inputIndex"], dtype=np.int64)
        source_index = np.asarray(
            values["sourceConfigurationIndex"], dtype=np.int64
        )
    evidence_cells = tuple(
        tuple(int(value) for value in row) for row in evidence.arrays["cellXYZ"]
    )
    if cells.shape != (len(evidence_cells), 3):
        raise ValueError("initial configuration ledger does not cover the evidence block")
    ledger = {
        tuple(int(value) for value in cell): (int(input_value), int(source_value))
        for cell, input_value, source_value in zip(cells, input_index, source_index)
    }
    if len(ledger) != len(evidence_cells) or set(ledger) != set(evidence_cells):
        raise ValueError("initial configuration ledger has missing or duplicate cells")
    config_offset = np.asarray(evidence.arrays["configurationOffset"], dtype=np.uint64)
    config_input = np.asarray(
        evidence.arrays["configurationInputIndex"], dtype=np.int64
    )
    config_source = np.asarray(
        evidence.arrays["configurationSourceIndex"], dtype=np.int64
    )
    selected: list[int] = []
    for cell_index, cell in enumerate(evidence_cells):
        target = ledger[cell]
        low = int(config_offset[cell_index])
        high = int(config_offset[cell_index + 1])
        matches = [
            index
            for index in range(low, high)
            if (int(config_input[index]), int(config_source[index])) == target
        ]
        if len(matches) != 1:
            raise ValueError(
                f"initial cell {cell} references an absent physical configuration"
            )
        selected.append(matches[0])
    return tuple(selected), {
        "kind": "cell-refinement-selection",
        "root": str(root),
        "manifestSha256": sha256_file(manifest_path),
        "dataSha256": sha256_file(selection_path),
    }


def _factor_value(
    arrays: Mapping[str, np.ndarray],
    face_index: int,
    first_configuration: int,
    second_configuration: int,
    settings: SheetConfigurationSolverSettings,
) -> tuple[float, int, int]:
    first_start = int(arrays["firstConfigurationStart"][face_index])
    first_count = int(arrays["firstConfigurationCount"][face_index])
    second_start = int(arrays["secondConfigurationStart"][face_index])
    second_count = int(arrays["secondConfigurationCount"][face_index])
    first_local = first_configuration - first_start
    second_local = second_configuration - second_start
    if not 0 <= first_local < first_count or not 0 <= second_local < second_count:
        raise ValueError("configuration index falls outside its face factor")
    pair_index = (
        int(arrays["pairOffset"][face_index])
        + first_local * second_count
        + second_local
    )
    benefit = float(arrays["pairJoinBenefit"][pair_index])
    matched = int(arrays["pairMatchedTraceCount"][pair_index])
    unmatched = int(arrays["pairUnmatchedTraceCount"][pair_index])
    if settings.pairwise_normalization == "trace-mean":
        benefit /= max(2 * matched + unmatched, 1)
    value = (
        settings.pairwise_scale * benefit
        - settings.unmatched_trace_penalty * unmatched
    )
    return value, matched, unmatched


def _unary_values(
    evidence: BlockSheetEvidence,
    settings: SheetConfigurationSolverSettings,
) -> np.ndarray:
    log_weight = np.asarray(
        evidence.arrays["configurationLogWeight"], dtype=np.float64
    )
    covered = np.asarray(
        evidence.arrays["configurationCoveredEvidenceMass"], dtype=np.float64
    )
    valid = np.asarray(
        evidence.arrays["configurationGeometryValid"], dtype=np.uint8
    )
    values = settings.unary_scale * (
        log_weight + settings.coverage_reward_scale * covered
    )
    values[valid == 0] = -math.inf
    return values


def _pair_value_matrices(
    arrays: Mapping[str, np.ndarray],
    settings: SheetConfigurationSolverSettings,
) -> tuple[np.ndarray, ...]:
    pair_offset = np.asarray(arrays["pairOffset"], dtype=np.uint64)
    benefit = np.asarray(arrays["pairJoinBenefit"], dtype=np.float64)
    matched = np.asarray(arrays["pairMatchedTraceCount"], dtype=np.float64)
    unmatched = np.asarray(arrays["pairUnmatchedTraceCount"], dtype=np.float64)
    result = []
    for face_index, (first_count_value, second_count_value) in enumerate(
        zip(arrays["firstConfigurationCount"], arrays["secondConfigurationCount"])
    ):
        first_count = int(first_count_value)
        second_count = int(second_count_value)
        low = int(pair_offset[face_index])
        high = int(pair_offset[face_index + 1])
        local_benefit = benefit[low:high].copy()
        if settings.pairwise_normalization == "trace-mean":
            local_benefit /= np.maximum(
                2.0 * matched[low:high] + unmatched[low:high], 1.0
            )
        values = (
            settings.pairwise_scale * local_benefit
            - settings.unmatched_trace_penalty * unmatched[low:high]
        )
        result.append(values.reshape(first_count, second_count))
    return tuple(result)


def _max_sum_configuration_seed(
    configuration_offset: np.ndarray,
    factors: Mapping[str, np.ndarray],
    unary: np.ndarray,
    settings: SheetConfigurationSolverSettings,
) -> tuple[tuple[int, ...], dict[str, Any]]:
    """Decode a whole-block max-sum loopy-belief-propagation initialization."""

    cell_count = len(configuration_offset) - 1
    face_count = len(factors["firstCellIndex"])
    pair_values = _pair_value_matrices(factors, settings)
    first_to_second = [
        np.zeros(int(factors["secondConfigurationCount"][face]), dtype=np.float64)
        for face in range(face_count)
    ]
    second_to_first = [
        np.zeros(int(factors["firstConfigurationCount"][face]), dtype=np.float64)
        for face in range(face_count)
    ]
    iterations = 0
    maximum_delta = math.inf
    for iteration in range(settings.belief_propagation_iterations):
        totals = [
            np.asarray(unary[int(low) : int(high)], dtype=np.float64).copy()
            for low, high in zip(configuration_offset[:-1], configuration_offset[1:])
        ]
        for face_index, (first_value, second_value) in enumerate(
            zip(factors["firstCellIndex"], factors["secondCellIndex"])
        ):
            totals[int(first_value)] += second_to_first[face_index]
            totals[int(second_value)] += first_to_second[face_index]
        next_first_to_second: list[np.ndarray] = []
        next_second_to_first: list[np.ndarray] = []
        maximum_delta = 0.0
        for face_index, (first_value, second_value) in enumerate(
            zip(factors["firstCellIndex"], factors["secondCellIndex"])
        ):
            first = int(first_value)
            second = int(second_value)
            first_cavity = totals[first] - second_to_first[face_index]
            second_cavity = totals[second] - first_to_second[face_index]
            matrix = pair_values[face_index]
            proposed_first_to_second = np.max(
                first_cavity[:, np.newaxis] + matrix, axis=0
            )
            proposed_second_to_first = np.max(
                matrix + second_cavity[np.newaxis, :], axis=1
            )
            if not (
                np.all(np.isfinite(proposed_first_to_second))
                and np.all(np.isfinite(proposed_second_to_first))
            ):
                raise RuntimeError("belief propagation produced a non-finite message")
            proposed_first_to_second -= np.max(proposed_first_to_second)
            proposed_second_to_first -= np.max(proposed_second_to_first)
            updated_first_to_second = (
                settings.belief_propagation_damping
                * first_to_second[face_index]
                + (1.0 - settings.belief_propagation_damping)
                * proposed_first_to_second
            )
            updated_second_to_first = (
                settings.belief_propagation_damping
                * second_to_first[face_index]
                + (1.0 - settings.belief_propagation_damping)
                * proposed_second_to_first
            )
            maximum_delta = max(
                maximum_delta,
                float(
                    np.max(
                        np.abs(
                            updated_first_to_second
                            - first_to_second[face_index]
                        )
                    )
                ),
                float(
                    np.max(
                        np.abs(
                            updated_second_to_first
                            - second_to_first[face_index]
                        )
                    )
                ),
            )
            next_first_to_second.append(updated_first_to_second)
            next_second_to_first.append(updated_second_to_first)
        first_to_second = next_first_to_second
        second_to_first = next_second_to_first
        iterations = iteration + 1
        if maximum_delta <= settings.belief_propagation_tolerance:
            break

    beliefs = [
        np.asarray(unary[int(low) : int(high)], dtype=np.float64).copy()
        for low, high in zip(configuration_offset[:-1], configuration_offset[1:])
    ]
    for face_index, (first_value, second_value) in enumerate(
        zip(factors["firstCellIndex"], factors["secondCellIndex"])
    ):
        beliefs[int(first_value)] += second_to_first[face_index]
        beliefs[int(second_value)] += first_to_second[face_index]
    selected = tuple(
        int(configuration_offset[cell_index]) + int(np.argmax(beliefs[cell_index]))
        for cell_index in range(cell_count)
    )
    return selected, {
        "beliefPropagationIterations": iterations,
        "beliefPropagationMaximumMessageDelta": round(maximum_delta, 6),
        "beliefPropagationConverged": (
            maximum_delta <= settings.belief_propagation_tolerance
        ),
    }


def _selection_record(
    evidence: BlockSheetEvidence,
    factors: Mapping[str, np.ndarray],
    selected: tuple[int, ...],
    unary: np.ndarray,
    settings: SheetConfigurationSolverSettings,
    *,
    sweeps: int,
    changed_last_sweep: int,
    initialization: str,
) -> SheetConfigurationSelection:
    unary_objective = sum(float(unary[value]) for value in selected)
    pairwise_objective = 0.0
    matched = 0
    unmatched = 0
    for face_index, (first_cell, second_cell) in enumerate(
        zip(factors["firstCellIndex"], factors["secondCellIndex"])
    ):
        value, pair_matched, pair_unmatched = _factor_value(
            factors,
            face_index,
            selected[int(first_cell)],
            selected[int(second_cell)],
            settings,
        )
        pairwise_objective += value
        matched += pair_matched
        unmatched += pair_unmatched
    return SheetConfigurationSelection(
        selected,
        unary_objective + pairwise_objective,
        unary_objective,
        pairwise_objective,
        matched,
        unmatched,
        sweeps,
        changed_last_sweep,
        initialization,
    )


def optimize_sheet_configurations(
    evidence: BlockSheetEvidence,
    factors: Mapping[str, np.ndarray],
    initial: tuple[int, ...],
    *,
    settings: SheetConfigurationSolverSettings | None = None,
) -> tuple[SheetConfigurationSelection, tuple[dict[str, Any], ...]]:
    resolved = settings or SheetConfigurationSolverSettings()
    config_offset = np.asarray(evidence.arrays["configurationOffset"], dtype=np.uint64)
    if len(initial) != evidence.cell_count:
        raise ValueError("initial configuration state does not cover every cell")
    unary = _unary_values(evidence, resolved)
    neighbors: list[list[tuple[int, int, bool]]] = [
        [] for _ in range(evidence.cell_count)
    ]
    for face_index, (first, second) in enumerate(
        zip(factors["firstCellIndex"], factors["secondCellIndex"])
    ):
        first_cell = int(first)
        second_cell = int(second)
        neighbors[first_cell].append((face_index, second_cell, True))
        neighbors[second_cell].append((face_index, first_cell, False))
    unary_initial = tuple(
        min(
            range(int(low), int(high)),
            key=lambda index: (-float(unary[index]), index),
        )
        for low, high in zip(config_offset[:-1], config_offset[1:])
    )
    seeds: list[tuple[str, tuple[int, ...], dict[str, Any]]] = [
        ("declared-initial", initial, {}),
        ("unary-optimum", unary_initial, {}),
    ]
    if resolved.belief_propagation_iterations > 0:
        message_seed, message_record = _max_sum_configuration_seed(
            config_offset,
            factors,
            unary,
            resolved,
        )
        seeds.append(("max-sum-belief-propagation", message_seed, message_record))
    proposals: list[SheetConfigurationSelection] = []
    records: list[dict[str, Any]] = []
    for initialization, seed, seed_record in seeds:
        selected = list(seed)
        changed = 0
        sweeps = 0
        for sweep in range(resolved.maximum_sweeps):
            sweeps = sweep + 1
            changed = 0
            traversal = (
                range(evidence.cell_count)
                if sweep % 2 == 0
                else range(evidence.cell_count - 1, -1, -1)
            )
            for cell_index in traversal:
                low = int(config_offset[cell_index])
                high = int(config_offset[cell_index + 1])
                best_index = selected[cell_index]
                best_score = -math.inf
                for configuration_index in range(low, high):
                    score = float(unary[configuration_index])
                    if not math.isfinite(score):
                        continue
                    for face_index, neighbor, first_side in neighbors[cell_index]:
                        first_configuration = (
                            configuration_index
                            if first_side
                            else selected[neighbor]
                        )
                        second_configuration = (
                            selected[neighbor]
                            if first_side
                            else configuration_index
                        )
                        score += _factor_value(
                            factors,
                            face_index,
                            first_configuration,
                            second_configuration,
                            resolved,
                        )[0]
                    if (score, -configuration_index) > (best_score, -best_index):
                        best_score = score
                        best_index = configuration_index
                if best_index != selected[cell_index]:
                    selected[cell_index] = best_index
                    changed += 1
            if changed == 0:
                break
        result = _selection_record(
            evidence,
            factors,
            tuple(selected),
            unary,
            resolved,
            sweeps=sweeps,
            changed_last_sweep=changed,
            initialization=initialization,
        )
        proposals.append(result)
        records.append(
            {
                "initialization": initialization,
                "sweeps": sweeps,
                "changedLastSweep": changed,
                "objective": round(result.objective, 6),
                "unaryObjective": round(result.unary_objective, 6),
                "pairwiseObjective": round(result.pairwise_objective, 6),
                "matchedTraces": result.matched_trace_count,
                "unmatchedTraceEndpoints": result.unmatched_trace_endpoint_count,
                **seed_record,
            }
        )
    best = max(
        proposals,
        key=lambda value: (
            value.objective,
            value.matched_trace_count,
            -value.unmatched_trace_endpoint_count,
            tuple(-index for index in value.configuration_index_by_cell),
        ),
    )
    return best, tuple(records)


def _state_metrics(
    evidence: BlockSheetEvidence,
    selection: SheetConfigurationSelection,
    reference: tuple[int, ...],
) -> dict[str, Any]:
    selected = selection.configuration_index_by_cell
    covered = np.asarray(
        evidence.arrays["configurationCoveredEvidenceMass"], dtype=np.float64
    )
    total = np.asarray(
        evidence.arrays["configurationTotalEvidenceMass"], dtype=np.float64
    )
    mode_offset = np.asarray(evidence.arrays["configurationModeOffset"], dtype=np.uint64)
    selected_mode_count = sum(
        int(mode_offset[value + 1]) - int(mode_offset[value]) for value in selected
    )
    covered_mass = sum(float(covered[value]) for value in selected)
    total_mass = sum(float(total[value]) for value in selected)
    endpoints = 2 * selection.matched_trace_count + selection.unmatched_trace_endpoint_count
    return {
        "objective": round(selection.objective, 6),
        "unaryObjective": round(selection.unary_objective, 6),
        "pairwiseObjective": round(selection.pairwise_objective, 6),
        "changedConfigurations": sum(
            first != second for first, second in zip(selected, reference)
        ),
        "selectedModePatches": selected_mode_count,
        "coveredEvidenceMass": round(covered_mass, 6),
        "totalEvidenceMass": round(total_mass, 6),
        "coveredEvidenceFraction": round(covered_mass / max(total_mass, 1.0e-12), 6),
        "locallyMatchedFaceTraces": selection.matched_trace_count,
        "locallyUnmatchedFaceTraceEndpoints": selection.unmatched_trace_endpoint_count,
        "projectedLocalRetainedTraceFraction": round(
            2 * selection.matched_trace_count / max(endpoints, 1), 6
        ),
        "sweeps": selection.sweeps,
        "changedLastSweep": selection.changed_last_sweep,
        "initialization": selection.initialization,
    }


def run_sheet_configuration_initialization(
    evidence_root: str | Path,
    factor_root: str | Path,
    output_root: str | Path,
    *,
    initial_root: str | Path | None = None,
    settings: SheetConfigurationSolverSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    started = time.monotonic()
    resolved = settings or SheetConfigurationSolverSettings()
    evidence_path = Path(evidence_root).resolve()
    factors_path = Path(factor_root).resolve()
    initial_path = Path(initial_root).resolve() if initial_root is not None else None
    output = Path(output_root).resolve()
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    factors, factor_manifest = _read_factors(factors_path)
    initial, initial_identity = _load_initial_selection(evidence, initial_path)
    identity: dict[str, Any] = {
        "schema": SHEET_CONFIGURATION_SOLVER_SCHEMA,
        "version": SHEET_CONFIGURATION_SOLVER_VERSION,
        "evidenceRoot": str(evidence_path),
        "evidenceManifestSha256": sha256_file(
            evidence_path / "sheet-evidence-v1.json"
        ),
        "factorRoot": str(factors_path),
        "factorIdentitySha256": factor_manifest["identity"]["identitySha256"],
        "factorDataSha256": factor_manifest["data"]["sha256"],
        "initialState": initial_identity,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__).resolve()),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_CONFIGURATION_SOLVER_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet configuration output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    best, proposals = optimize_sheet_configurations(
        evidence,
        factors,
        initial,
        settings=resolved,
    )
    unary = _unary_values(evidence, resolved)
    baseline = _selection_record(
        evidence,
        factors,
        initial,
        unary,
        resolved,
        sweeps=0,
        changed_last_sweep=0,
        initialization="declared-initial",
    )
    cells = np.asarray(evidence.arrays["cellXYZ"], dtype=np.int32)
    config_id = np.asarray(evidence.arrays["configurationId"], dtype=np.uint64)
    input_index = np.asarray(
        evidence.arrays["configurationInputIndex"], dtype=np.uint16
    )
    source_index = np.asarray(
        evidence.arrays["configurationSourceIndex"], dtype=np.uint32
    )
    mode_offset = np.asarray(evidence.arrays["configurationModeOffset"], dtype=np.uint64)
    selected = np.asarray(best.configuration_index_by_cell, dtype=np.uint32)
    data_path = output / f"{SHEET_CONFIGURATION_SOLVER_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=cells,
            configurationIndex=selected,
            configurationId=config_id[selected],
            inputIndex=input_index[selected],
            sourceConfigurationIndex=source_index[selected],
            selectedModeCount=np.asarray(
                [int(mode_offset[value + 1]) - int(mode_offset[value]) for value in selected],
                dtype=np.uint16,
            ),
        )
    temporary.replace(data_path)
    baseline_metrics = _state_metrics(evidence, baseline, initial)
    best_metrics = _state_metrics(evidence, best, initial)
    summary = {
        "schema": "pareidolia.cubical-sheet-configuration-initialization-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "baseline": baseline_metrics,
        "selected": best_metrics,
        "delta": {
            key: round(float(best_metrics[key]) - float(baseline_metrics[key]), 6)
            for key in (
                "objective",
                "unaryObjective",
                "pairwiseObjective",
                "selectedModePatches",
                "coveredEvidenceMass",
                "coveredEvidenceFraction",
                "locallyMatchedFaceTraces",
                "locallyUnmatchedFaceTraceEndpoints",
                "projectedLocalRetainedTraceFraction",
            )
        },
        "proposals": list(proposals),
        "warning": (
            "face factors do not enforce transitive component/cell collision, "
            "crossing, or orientability constraints; this selection is an "
            "initialization and must be globally sheet-replayed before use"
        ),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_CONFIGURATION_SOLVER_SCHEMA,
            "version": SHEET_CONFIGURATION_SOLVER_VERSION,
            "state": "complete",
            "identity": identity,
            "summary": summary_path.name,
            "data": summary["data"],
            "elapsedSeconds": summary["elapsedSeconds"],
        },
    )
    return summary
