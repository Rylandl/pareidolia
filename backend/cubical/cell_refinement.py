from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import (
    BoundaryTrace,
    SurfaceBlock,
    extend_surface_block_joins,
    surface_block_from_retained_joins,
)
from .boundary_topology import (
    FrozenTopologyCut,
    freeze_topology_outside_patches,
    select_joins_with_frozen_topology,
)
from .cluster_reselection import CLUSTER_RESELECTION_SCHEMA
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch, clip_plane_to_cell
from .matching import (
    TraceMatch,
    TraceMatchSettings,
    align_face_patches,
    match_face_traces,
)
from .saturation_selection import load_saturation_candidates
from .selection import ConfigurationOption, pairwise_reward_energy
from .stratigraphy import ConfigurationTable
from .surface_graph import read_surface_graph
from .topology import GridFace, GridSpec, Int3, cell_face


CELL_REFINEMENT_DIAGNOSTIC_SCHEMA = (
    "pareidolia.cubical-cell-refinement-diagnostic"
)
CELL_REFINEMENT_DIAGNOSTIC_VERSION = 1
CELL_REFINEMENT_SELECTION_SCHEMA = (
    "pareidolia.cubical-cell-refinement-selection"
)
CELL_REFINEMENT_SELECTION_VERSION = 1
CELL_REFINEMENT_SELECTION_STEM = "cell-refinement-selection-v1"

# Candidate-bank configuration indices are stable across refinement rounds,
# while materialized patch IDs are not.  Reserve the upper half of uint64 for
# patches reconstructed from retained candidate configurations so iterative
# variants cannot collide with any existing sequential materialization IDs.
REFINEMENT_PATCH_ID_BASE = 1 << 63
REFINEMENT_PATCH_ID_STRIDE = 64


@dataclass(frozen=True, slots=True)
class CellRefinementSettings:
    """Evidence/continuity objective used for conditional cell refinement.

    Coverage is rewarded in evidence-mass units.  This naturally lets cells
    with substantial Acus support resist a contradictory neighborhood prior,
    while weakly observed cells can still inherit continuity from neighbors.
    """

    unary_scale: float = 1.0
    pairwise_scale: float = 0.2
    pairwise_reward_normalization: str = "trace-mean"
    unmatched_trace_penalty: float | None = None
    coverage_reward_scale: float = 0.5
    minimum_oracle_coverage_fraction: float = 0.5
    maximum_cell_utilization_drop: float = 0.05
    minimum_evidence_mass_for_coverage_floor: float = 1.0
    maximum_sweeps: int = 4
    maximum_pair_sweeps: int = 2

    def __post_init__(self) -> None:
        if not math.isfinite(self.unary_scale) or self.unary_scale <= 0.0:
            raise ValueError("unary scale must be finite and positive")
        if not math.isfinite(self.pairwise_scale) or self.pairwise_scale <= 0.0:
            raise ValueError("pairwise scale must be finite and positive")
        if self.pairwise_reward_normalization not in ("none", "trace-mean"):
            raise ValueError(
                "pairwise reward normalization must be 'none' or 'trace-mean'"
            )
        if self.unmatched_trace_penalty is not None and (
            not math.isfinite(self.unmatched_trace_penalty)
            or self.unmatched_trace_penalty < 0.0
        ):
            raise ValueError("unmatched trace penalty must be finite and nonnegative")
        if (
            not math.isfinite(self.coverage_reward_scale)
            or self.coverage_reward_scale < 0.0
        ):
            raise ValueError("coverage reward scale must be finite and nonnegative")
        unit_interval = (
            self.minimum_oracle_coverage_fraction,
            self.maximum_cell_utilization_drop,
        )
        if any(
            not math.isfinite(value) or value < 0.0 or value > 1.0
            for value in unit_interval
        ):
            raise ValueError("coverage fractions must lie in [0, 1]")
        if (
            not math.isfinite(self.minimum_evidence_mass_for_coverage_floor)
            or self.minimum_evidence_mass_for_coverage_floor < 0.0
        ):
            raise ValueError("minimum evidence mass must be finite and nonnegative")
        if self.maximum_sweeps <= 0 or self.maximum_pair_sweeps < 0:
            raise ValueError("cell sweeps must be positive and pair sweeps nonnegative")

    def record(self) -> dict[str, Any]:
        return asdict(self)

    def resolved_unmatched_trace_penalty(
        self, matching_settings: TraceMatchSettings
    ) -> float:
        """Continuation cost consistent with the unnormalized match baseline."""

        if self.unmatched_trace_penalty is not None:
            return self.unmatched_trace_penalty
        return (
            self.pairwise_scale
            * matching_settings.unmatched_negative_log_likelihood
        )


@dataclass(frozen=True, slots=True)
class FaceContinuityScore:
    axis: int
    side: int
    neighbor_cell_xyz: Int3
    first_trace_count: int
    second_trace_count: int
    match_count: int
    unmatched_trace_count: int
    baseline_negative_log_likelihood: float
    aligned_negative_log_likelihood: float
    relative_negative_log_likelihood: float
    raw_reward_energy: float
    trace_mean_reward_energy: float

    def record(self) -> dict[str, Any]:
        return {
            "axis": self.axis,
            "side": self.side,
            "neighborCellXYZ": list(self.neighbor_cell_xyz),
            "firstTraceCount": self.first_trace_count,
            "secondTraceCount": self.second_trace_count,
            "matchCount": self.match_count,
            "unmatchedTraceCount": self.unmatched_trace_count,
            "baselineNegativeLogLikelihood": round(
                self.baseline_negative_log_likelihood, 6
            ),
            "alignedNegativeLogLikelihood": round(
                self.aligned_negative_log_likelihood, 6
            ),
            "relativeNegativeLogLikelihood": round(
                self.relative_negative_log_likelihood, 6
            ),
            "rawRewardEnergy": round(self.raw_reward_energy, 6),
            "traceMeanRewardEnergy": round(
                self.trace_mean_reward_energy, 6
            ),
        }


@dataclass(frozen=True, slots=True)
class CellCandidateScore:
    source_configuration_index: int
    local_configuration_id: int
    selected: bool
    evidence_admissible: bool
    minimum_admissible_utilization: float
    layer_count: int
    log_weight: float
    covered_evidence_mass: float
    total_evidence_mass: float
    evidence_utilization: float
    unary_energy: float
    coverage_energy: float
    raw_continuity_energy: float
    trace_mean_continuity_energy: float
    unmatched_trace_energy: float
    objective_energy: float
    matched_trace_count: int
    unmatched_trace_count: int
    faces: tuple[FaceContinuityScore, ...]

    def record(self, *, include_faces: bool = False) -> dict[str, Any]:
        result: dict[str, Any] = {
            "sourceConfigurationIndex": self.source_configuration_index,
            "localConfigurationId": self.local_configuration_id,
            "selected": self.selected,
            "evidenceAdmissible": self.evidence_admissible,
            "minimumAdmissibleUtilization": round(
                self.minimum_admissible_utilization, 6
            ),
            "layerCount": self.layer_count,
            "logWeight": round(self.log_weight, 6),
            "coveredEvidenceMass": round(self.covered_evidence_mass, 6),
            "totalEvidenceMass": round(self.total_evidence_mass, 6),
            "evidenceUtilization": round(self.evidence_utilization, 6),
            "unaryEnergy": round(self.unary_energy, 6),
            "coverageEnergy": round(self.coverage_energy, 6),
            "rawContinuityEnergy": round(self.raw_continuity_energy, 6),
            "traceMeanContinuityEnergy": round(
                self.trace_mean_continuity_energy, 6
            ),
            "unmatchedTraceEnergy": round(self.unmatched_trace_energy, 6),
            "objectiveEnergy": round(self.objective_energy, 6),
            "matchedTraceCount": self.matched_trace_count,
            "unmatchedTraceCount": self.unmatched_trace_count,
        }
        if include_faces:
            result["faces"] = [value.record() for value in self.faces]
        return result


@dataclass(frozen=True, slots=True)
class _CandidateBank:
    input_index: int
    root: Path
    offset_cells_xyz: Int3
    table: ConfigurationTable
    metadata: Mapping[str, np.ndarray]
    selected_by_local_cell: Mapping[Int3, int]
    option_base: int


@dataclass(slots=True)
class ClusterCellContext:
    cluster_root: Path
    materialized_root: Path
    grid: GridSpec
    block: SurfaceBlock
    banks: tuple[_CandidateBank, ...]
    owner_by_cell: Mapping[Int3, tuple[int, Int3]]
    selected_by_cell: dict[Int3, int]
    _option_cache: dict[tuple[Int3, int], ConfigurationOption] = field(
        default_factory=dict
    )

    def owner(self, cell_xyz: Int3) -> tuple[_CandidateBank, Int3]:
        if cell_xyz not in self.owner_by_cell:
            raise KeyError(f"cell {cell_xyz} is outside the candidate cluster")
        bank_index, local_cell = self.owner_by_cell[cell_xyz]
        return self.banks[bank_index], local_cell

    def configuration_indices(self, cell_xyz: Int3) -> range:
        bank, local_cell = self.owner(cell_xyz)
        matches = np.flatnonzero(
            np.all(
                bank.table.cell_xyz
                == np.asarray(local_cell, dtype=np.int32)[None, :],
                axis=1,
            )
        )
        if len(matches) != 1:
            raise ValueError(f"candidate bank cell index is ambiguous for {cell_xyz}")
        return bank.table.configurations_for_cell(int(matches[0]))

    def option(self, cell_xyz: Int3, configuration_index: int) -> ConfigurationOption:
        key = (cell_xyz, int(configuration_index))
        cached = self._option_cache.get(key)
        if cached is not None:
            return cached
        bank, _ = self.owner(cell_xyz)
        valid = self.configuration_indices(cell_xyz)
        if configuration_index not in valid:
            raise ValueError(
                f"configuration {configuration_index} does not belong to {cell_xyz}"
            )
        option_id = bank.option_base + int(configuration_index)
        patches: list[ClippedPatch] = []
        degenerate = 0
        for layer_index, estimate in enumerate(
            bank.table.estimates_for_configuration(configuration_index)
        ):
            if layer_index >= REFINEMENT_PATCH_ID_STRIDE:
                raise ValueError(
                    "candidate configuration exceeds the refinement patch-ID stride"
                )
            patch = clip_plane_to_cell(
                self.grid,
                cell_xyz,
                estimate,
                patch_id=(
                    REFINEMENT_PATCH_ID_BASE
                    + option_id * REFINEMENT_PATCH_ID_STRIDE
                    + layer_index
                ),
            )
            if patch is None:
                degenerate += 1
            else:
                patches.append(patch)
        patches.sort(
            key=lambda value: (
                value.estimate.height_from_cell_center,
                value.patch_id,
            )
        )
        option = ConfigurationOption(
            option_id,
            cell_xyz,
            bank.input_index,
            int(configuration_index),
            int(bank.table.configuration_id[configuration_index]),
            float(bank.table.configuration_log_weight[configuration_index]),
            tuple(patches),
            degenerate,
        )
        self._option_cache[key] = option
        return option

    def selected_option(
        self,
        cell_xyz: Int3,
        selected_by_cell: Mapping[Int3, int] | None = None,
    ) -> ConfigurationOption:
        selected = self.selected_by_cell if selected_by_cell is None else selected_by_cell
        return self.option(cell_xyz, int(selected[cell_xyz]))

    def evidence(self, cell_xyz: Int3, configuration_index: int) -> tuple[float, float]:
        bank, _ = self.owner(cell_xyz)
        return (
            float(bank.metadata["coveredEvidenceMass"][configuration_index]),
            float(bank.metadata["totalEvidenceMass"][configuration_index]),
        )


@dataclass(frozen=True, slots=True)
class TopologyReplay:
    """Exact local replay state suitable for audit or materialization."""

    summary: dict[str, Any]
    block: SurfaceBlock
    selected_by_cell: dict[Int3, int]
    active_cells: frozenset[Int3]
    replaced_patch_ids: frozenset[int]
    replacement_patch_ids: frozenset[int]


def _load_child_selection(root: Path) -> dict[Int3, int]:
    path = root / "selection-v1.npz"
    if not path.is_file():
        raise ValueError(f"candidate root lacks selection artifact: {root}")
    manifest_path = root / "selection-v1.json"
    if manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        record = manifest.get("data", {})
        if record.get("sha256") and sha256_file(path) != record["sha256"]:
            raise ValueError("candidate selection content hash mismatch")
    with np.load(path) as values:
        cells = np.asarray(values["cellXYZ"], dtype=np.int32)
        indices = np.asarray(values["sourceConfigurationIndex"], dtype=np.int64)
    if len(cells) != len(indices):
        raise ValueError("candidate selection arrays do not align")
    result = {
        tuple(int(value) for value in cell): int(index)
        for cell, index in zip(cells, indices)
    }
    if len(result) != len(cells):
        raise ValueError("candidate selection contains duplicate cells")
    return result


def _load_refinement_selection(
    materialized: Path,
    *,
    owner_by_cell: Mapping[Int3, tuple[int, Int3]],
) -> dict[Int3, int] | None:
    """Load a complete, graph-bound selection ledger when one is present."""

    manifest_path = materialized / f"{CELL_REFINEMENT_SELECTION_STEM}.json"
    if not manifest_path.is_file():
        return None
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != CELL_REFINEMENT_SELECTION_SCHEMA
        or int(manifest.get("version", -1))
        != CELL_REFINEMENT_SELECTION_VERSION
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete cell-refinement selection")
    data_record = manifest.get("data", {})
    data_path = materialized / str(data_record.get("path", ""))
    if not data_path.is_file() or sha256_file(data_path) != data_record.get(
        "sha256"
    ):
        raise ValueError("cell-refinement selection content hash mismatch")
    bindings = manifest.get("graphBindings", {})
    expected_bindings = {
        "selectedPatchManifestSha256": materialized / "selected-patches-v1.json",
        "selectedPatchDataSha256": materialized / "selected-patches-v1.npz",
        "surfaceGraphManifestSha256": materialized / "surface-graph-v1.json",
        "surfaceGraphDataSha256": materialized / "surface-graph-v1.npz",
    }
    for name, path in expected_bindings.items():
        if not path.is_file() or sha256_file(path) != bindings.get(name):
            raise ValueError(
                "cell-refinement selection is not bound to this materialized graph"
            )
    with np.load(data_path) as values:
        cells = np.asarray(values["cellXYZ"], dtype=np.int32)
        inputs = np.asarray(values["inputIndex"], dtype=np.int64)
        local_cells = np.asarray(values["localCellXYZ"], dtype=np.int32)
        selected_indices = np.asarray(
            values["sourceConfigurationIndex"], dtype=np.int64
        )
    count = len(cells)
    if (
        cells.shape != (count, 3)
        or inputs.shape != (count,)
        or local_cells.shape != (count, 3)
        or selected_indices.shape != (count,)
    ):
        raise ValueError("cell-refinement selection arrays do not align")
    selected: dict[Int3, int] = {}
    for cell_values, input_index, local_values, selected_index in zip(
        cells, inputs, local_cells, selected_indices
    ):
        cell = tuple(int(value) for value in cell_values)
        local_cell = tuple(int(value) for value in local_values)
        if owner_by_cell.get(cell) != (int(input_index), local_cell):
            raise ValueError(
                "cell-refinement selection ownership disagrees with the cluster"
            )
        if cell in selected:
            raise ValueError("cell-refinement selection contains duplicate cells")
        selected[cell] = int(selected_index)
    if set(selected) != set(owner_by_cell):
        raise ValueError(
            "cell-refinement selection must cover every materialized cell"
        )
    return selected


def load_cluster_cell_context(
    cluster_root: str | Path,
    materialized_root: str | Path,
) -> ClusterCellContext:
    """Load complete per-cell candidate provenance for a materialized cluster."""

    cluster = Path(cluster_root).resolve()
    materialized = Path(materialized_root).resolve()
    manifest_path = cluster / "cluster-reselection-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != CLUSTER_RESELECTION_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("cell refinement requires a complete cluster solve")
    grid_record = manifest["grid"]
    grid = GridSpec(
        tuple(int(value) for value in grid_record["shapeCellsXYZ"]),
        tuple(float(value) for value in grid_record["cellSizeXYZ"]),
        tuple(float(value) for value in grid_record["originXYZ"]),
        str(grid_record["coordinateUnit"]),
    )
    block = read_surface_graph(materialized, verify=True)
    if block.grid != grid:
        raise ValueError("cluster and materialized graph grids disagree")

    banks: list[_CandidateBank] = []
    owner_by_cell: dict[Int3, tuple[int, Int3]] = {}
    selected_by_cell: dict[Int3, int] = {}
    option_base = 0
    for input_index, record in enumerate(manifest["layout"]["inputs"]):
        boundary = Path(record["root"]).resolve()
        boundary_manifest = json.loads(
            (boundary / "boundary-band-v1.json").read_text()
        )
        candidate_root = Path(
            boundary_manifest["identity"]["candidateRoot"]
        ).resolve()
        table, metadata, candidate_manifest = load_saturation_candidates(
            candidate_root
        )
        identity = boundary_manifest["identity"]
        if candidate_manifest["data"]["sha256"] != identity[
            "candidateDataSha256"
        ]:
            raise ValueError("boundary candidate-bank identity changed")
        if sha256_file(candidate_root / "selection-v1.json") != identity[
            "selectionManifestSha256"
        ] or sha256_file(candidate_root / "selection-v1.npz") != identity[
            "selectionDataSha256"
        ]:
            raise ValueError("boundary candidate selection identity changed")
        local_selection = _load_child_selection(candidate_root)
        offset = tuple(int(value) for value in record["offsetCellsXYZ"])
        bank = _CandidateBank(
            input_index,
            candidate_root,
            offset,
            table,
            metadata,
            local_selection,
            option_base,
        )
        banks.append(bank)
        option_base += table.configuration_count
        for local_values in table.cell_xyz:
            local_cell = tuple(int(value) for value in local_values)
            cell = tuple(
                local_cell[axis] + offset[axis] for axis in range(3)
            )
            if cell in owner_by_cell:
                raise ValueError(f"cluster candidate ownership overlaps at {cell}")
            owner_by_cell[cell] = (input_index, local_cell)
            selected_by_cell[cell] = int(local_selection[local_cell])

    expected_cell_count = int(np.prod(grid.shape_cells_xyz))
    if len(owner_by_cell) != expected_cell_count:
        raise ValueError(
            "cluster candidates cover "
            f"{len(owner_by_cell)}/{expected_cell_count} global cells"
        )

    reselection_record = manifest["artifacts"]["reselection"]
    reselection_path = cluster / reselection_record["path"]
    if sha256_file(reselection_path) != reselection_record["sha256"]:
        raise ValueError("cluster reselection content hash mismatch")
    with np.load(reselection_path) as values:
        inputs = np.asarray(values["selectedCellInput"], dtype=np.int64)
        local_cells = np.asarray(values["selectedCellLocalXYZ"], dtype=np.int32)
        global_cells = np.asarray(values["selectedCellCombinedXYZ"], dtype=np.int32)
        selected_indices = np.asarray(
            values["selectedSourceConfigurationIndex"], dtype=np.int64
        )
    for input_index, local_values, global_values, selected_index in zip(
        inputs, local_cells, global_cells, selected_indices
    ):
        local_cell = tuple(int(value) for value in local_values)
        cell = tuple(int(value) for value in global_values)
        if owner_by_cell.get(cell) != (int(input_index), local_cell):
            raise ValueError("cluster reselection cell provenance is inconsistent")
        selected_by_cell[cell] = int(selected_index)

    refined_selection = _load_refinement_selection(
        materialized,
        owner_by_cell=owner_by_cell,
    )
    if refined_selection is not None:
        selected_by_cell = refined_selection

    context = ClusterCellContext(
        cluster,
        materialized,
        grid,
        block,
        tuple(banks),
        owner_by_cell,
        selected_by_cell,
    )
    patches_per_cell = Counter(value.cell_xyz for value in block.patches)
    for cell, selected_index in selected_by_cell.items():
        realized = len(context.option(cell, selected_index).patches)
        if patches_per_cell[cell] != realized:
            raise ValueError(
                f"materialized layer count disagrees with selected candidate at {cell}"
            )
    return context


def _face_continuity(
    context: ClusterCellContext,
    cell_xyz: Int3,
    option: ConfigurationOption,
    neighbor_cell_xyz: Int3,
    neighbor: ConfigurationOption,
    axis: int,
    side: int,
    *,
    pairwise_scale: float,
    matching_settings: TraceMatchSettings,
) -> FaceContinuityScore:
    face = cell_face(cell_xyz, axis, side)
    first, second = (
        (neighbor.patches, option.patches)
        if side == 0
        else (option.patches, neighbor.patches)
    )
    first_count = sum(value.trace_on(face) is not None for value in first)
    second_count = sum(value.trace_on(face) is not None for value in second)
    baseline = matching_settings.unmatched_negative_log_likelihood * (
        first_count + second_count
    )
    if not first_count and not second_count:
        aligned = 0.0
        matches = 0
        unmatched = 0
    else:
        try:
            alignment = align_face_patches(
                first,
                second,
                face,
                matching_settings,
                grid=context.grid,
            )
            aligned = alignment.negative_log_likelihood
            matches = len(alignment.matches)
            unmatched = len(alignment.unmatched_first_patch_ids) + len(
                alignment.unmatched_second_patch_ids
            )
        except ValueError:
            aligned = baseline
            matches = 0
            unmatched = first_count + second_count
    relative = aligned - baseline
    return FaceContinuityScore(
        axis,
        side,
        neighbor_cell_xyz,
        first_count,
        second_count,
        matches,
        unmatched,
        float(baseline),
        float(aligned),
        float(relative),
        pairwise_reward_energy(
            relative,
            first_count,
            second_count,
            pairwise_scale=pairwise_scale,
            normalization="none",
        ),
        pairwise_reward_energy(
            relative,
            first_count,
            second_count,
            pairwise_scale=pairwise_scale,
            normalization="trace-mean",
        ),
    )


def score_cell_candidates(
    context: ClusterCellContext,
    cell_xyz: Int3,
    *,
    selected_by_cell: Mapping[Int3, int] | None = None,
    settings: CellRefinementSettings | None = None,
    matching_settings: TraceMatchSettings | None = None,
) -> tuple[CellCandidateScore, ...]:
    """Score every retained physical stack with all six neighbors fixed."""

    resolved = settings or CellRefinementSettings()
    matching = matching_settings or TraceMatchSettings()
    selected = context.selected_by_cell if selected_by_cell is None else selected_by_cell
    current_index = int(selected[cell_xyz])
    reference_index = int(context.selected_by_cell[cell_xyz])
    bank, _ = context.owner(cell_xyz)
    configuration_indices = tuple(context.configuration_indices(cell_xyz))
    evidence_values = {
        value: context.evidence(cell_xyz, value)
        for value in configuration_indices
    }
    current_covered, current_total = evidence_values[reference_index]
    current_utilization = (
        current_covered / current_total if current_total > 1.0e-12 else 0.0
    )
    oracle_utilization = max(
        (
            covered / total if total > 1.0e-12 else 0.0
            for covered, total in evidence_values.values()
        ),
        default=0.0,
    )
    minimum_utilization = (
        max(
            resolved.minimum_oracle_coverage_fraction * oracle_utilization,
            current_utilization - resolved.maximum_cell_utilization_drop,
        )
        if current_total >= resolved.minimum_evidence_mass_for_coverage_floor
        else 0.0
    )
    result: list[CellCandidateScore] = []
    for configuration_index in configuration_indices:
        option = context.option(cell_xyz, configuration_index)
        faces: list[FaceContinuityScore] = []
        for axis in range(3):
            for side in (0, 1):
                neighbor_values = list(cell_xyz)
                neighbor_values[axis] += -1 if side == 0 else 1
                neighbor_cell = tuple(neighbor_values)
                if not context.grid.contains_cell(neighbor_cell):
                    continue
                faces.append(
                    _face_continuity(
                        context,
                        cell_xyz,
                        option,
                        neighbor_cell,
                        context.selected_option(neighbor_cell, selected),
                        axis,
                        side,
                        pairwise_scale=resolved.pairwise_scale,
                        matching_settings=matching,
                    )
                )
        covered, total = evidence_values[configuration_index]
        utilization = covered / total if total > 1.0e-12 else 0.0
        admissible = utilization + 1.0e-9 >= minimum_utilization
        unary = resolved.unary_scale * -option.log_weight
        coverage = -resolved.coverage_reward_scale * covered
        raw = sum(value.raw_reward_energy for value in faces)
        trace_mean = sum(value.trace_mean_reward_energy for value in faces)
        unmatched_energy = resolved.resolved_unmatched_trace_penalty(matching) * sum(
            value.unmatched_trace_count for value in faces
        )
        continuity = (
            raw
            if resolved.pairwise_reward_normalization == "none"
            else trace_mean
        )
        result.append(
            CellCandidateScore(
                int(configuration_index),
                option.local_configuration_id,
                configuration_index == current_index,
                admissible,
                minimum_utilization,
                len(option.patches),
                option.log_weight,
                covered,
                total,
                utilization,
                unary,
                coverage,
                raw,
                trace_mean,
                unmatched_energy,
                unary + coverage + continuity + unmatched_energy,
                sum(value.match_count for value in faces),
                sum(value.unmatched_trace_count for value in faces),
                tuple(faces),
            )
        )
    return tuple(
        sorted(
            result,
            key=lambda value: (
                not value.evidence_admissible,
                value.objective_energy,
                -value.evidence_utilization,
                value.source_configuration_index,
            ),
        )
    )


def _quantiles(values: np.ndarray) -> dict[str, float]:
    if not len(values):
        return {name: 0.0 for name in ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")}
    points = np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
    return {
        name: round(float(value), 6)
        for name, value in zip(
            ("minimum", "p10", "p25", "median", "p75", "p90", "maximum"),
            points,
        )
    }


def evidence_utilization_summary(context: ClusterCellContext) -> dict[str, Any]:
    selected_mass = 0.0
    oracle_mass = 0.0
    total_mass = 0.0
    current_values: list[float] = []
    oracle_values: list[float] = []
    gap_values: list[float] = []
    selected_layer_counts: Counter[int] = Counter()
    for cell in sorted(context.owner_by_cell, key=lambda value: (value[2], value[1], value[0])):
        indices = tuple(context.configuration_indices(cell))
        selected_index = context.selected_by_cell[cell]
        covered = np.asarray(
            [context.evidence(cell, value)[0] for value in indices],
            dtype=np.float64,
        )
        total = context.evidence(cell, indices[0])[1]
        selected = context.evidence(cell, selected_index)[0]
        oracle = float(np.max(covered))
        selected_mass += selected
        oracle_mass += oracle
        total_mass += total
        current = selected / total if total > 1.0e-12 else 0.0
        best = oracle / total if total > 1.0e-12 else 0.0
        current_values.append(current)
        oracle_values.append(best)
        gap_values.append(best - current)
        selected_layer_counts[len(context.option(cell, selected_index).patches)] += 1
    current_array = np.asarray(current_values, dtype=np.float64)
    oracle_array = np.asarray(oracle_values, dtype=np.float64)
    gap_array = np.asarray(gap_values, dtype=np.float64)
    return {
        "cells": len(current_values),
        "totalEvidenceMass": round(total_mass, 6),
        "selectedCoveredEvidenceMass": round(selected_mass, 6),
        "oracleCoveredEvidenceMass": round(oracle_mass, 6),
        "selectedEvidenceUtilization": round(selected_mass / max(total_mass, 1.0e-12), 6),
        "retainedCandidateOracleUtilization": round(oracle_mass / max(total_mass, 1.0e-12), 6),
        "recoverableUtilization": round(
            (oracle_mass - selected_mass) / max(total_mass, 1.0e-12), 6
        ),
        "cellSelectedUtilizationQuantiles": _quantiles(current_array),
        "cellOracleUtilizationQuantiles": _quantiles(oracle_array),
        "cellRecoverableUtilizationQuantiles": _quantiles(gap_array),
        "cellsWithRecoverableUtilizationAtLeast": {
            str(value): int(np.count_nonzero(gap_array >= value))
            for value in (0.1, 0.2, 0.3, 0.5)
        },
        "selectedLayerCounts": {
            str(key): int(value) for key, value in sorted(selected_layer_counts.items())
        },
    }


def _open_trace_classification(
    context: ClusterCellContext,
    boundary: BoundaryTrace,
    *,
    matching_settings: TraceMatchSettings,
    patches_by_cell: Mapping[Int3, tuple[ClippedPatch, ...]],
    joined_endpoints: set[tuple[int, GridFace]],
    component_by_patch: Mapping[int, int],
    patch_by_id: Mapping[int, ClippedPatch],
) -> dict[str, Any]:
    source = patch_by_id[boundary.patch_id]
    face = boundary.trace.face
    lower, upper = face.adjacent_cells()
    target_cell = upper if source.cell_xyz == lower else lower
    targets = tuple(
        value
        for value in patches_by_cell.get(target_cell, ())
        if value.trace_on(face) is not None
    )
    compatible: list[dict[str, Any]] = []
    for target in targets:
        trace = target.trace_on(face)
        if trace is None:
            continue
        match = match_face_traces(
            boundary.trace,
            source.estimate,
            trace,
            target.estimate,
            matching_settings,
            grid=context.grid,
        )
        if not match.accepted:
            continue
        occupied = (target.patch_id, face) in joined_endpoints
        same = component_by_patch[target.patch_id] == boundary.component_id
        compatible.append(
            {
                "patchId": target.patch_id,
                "componentId": component_by_patch[target.patch_id],
                "occupied": occupied,
                "sameComponent": same,
                "score": round(match.score, 6),
                "reducedChiSquare": round(match.reduced_chi_square, 6),
            }
        )
    if any(value["occupied"] for value in compatible):
        classification = "compatible-occupied"
    elif any(value["sameComponent"] for value in compatible):
        classification = "compatible-open-same-component"
    elif compatible:
        classification = "compatible-open-bridge"
    elif targets:
        classification = "selected-incompatible"
    else:
        classification = "selected-misses-face"
    return {
        "patchId": boundary.patch_id,
        "componentId": boundary.component_id,
        "sourceCellXYZ": list(source.cell_xyz),
        "targetCellXYZ": list(target_cell),
        "face": {"axis": face.axis, "anchorXYZ": list(face.anchor_xyz)},
        "classification": classification,
        "compatibleTargets": compatible,
    }


def component_gap_summary(
    context: ClusterCellContext,
    component_id: int,
) -> dict[str, Any]:
    patches_by_cell_values: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in context.block.patches:
        patches_by_cell_values[patch.cell_xyz].append(patch)
    patches_by_cell = {
        key: tuple(value) for key, value in patches_by_cell_values.items()
    }
    joined_endpoints = {
        (patch_id, join.face)
        for join in context.block.joins
        for patch_id in (join.first_patch_id, join.second_patch_id)
    }
    component_by_patch = dict(context.block.component_by_patch)
    patch_by_id = {value.patch_id: value for value in context.block.patches}
    matching = TraceMatchSettings(orthogonal_fiber_equivalence=True)
    records = [
        _open_trace_classification(
            context,
            boundary,
            matching_settings=matching,
            patches_by_cell=patches_by_cell,
            joined_endpoints=joined_endpoints,
            component_by_patch=component_by_patch,
            patch_by_id=patch_by_id,
        )
        for boundary in context.block.unresolved_interior_traces
        if boundary.component_id == component_id
    ]
    counts = Counter(value["classification"] for value in records)
    component = [
        value for value in context.block.components if value.component_id == component_id
    ]
    if len(component) != 1:
        raise ValueError(f"unknown materialized component {component_id}")
    return {
        "componentId": component_id,
        "patchCount": len(component[0].patch_ids),
        "unresolvedInteriorTraceCount": len(records),
        "classifications": dict(sorted(counts.items())),
        "traces": records,
    }


def incident_gap_summary(
    context: ClusterCellContext,
    cell_xyz: Int3,
) -> dict[str, Any]:
    patches_by_cell_values: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in context.block.patches:
        patches_by_cell_values[patch.cell_xyz].append(patch)
    patches_by_cell = {
        key: tuple(value) for key, value in patches_by_cell_values.items()
    }
    joined_endpoints = {
        (patch_id, join.face)
        for join in context.block.joins
        for patch_id in (join.first_patch_id, join.second_patch_id)
    }
    component_by_patch = dict(context.block.component_by_patch)
    patch_by_id = {value.patch_id: value for value in context.block.patches}
    matching = TraceMatchSettings(orthogonal_fiber_equivalence=True)
    records = []
    for boundary in context.block.unresolved_interior_traces:
        source = patch_by_id[boundary.patch_id]
        lower, upper = boundary.trace.face.adjacent_cells()
        target = upper if source.cell_xyz == lower else lower
        if source.cell_xyz != cell_xyz and target != cell_xyz:
            continue
        records.append(
            _open_trace_classification(
                context,
                boundary,
                matching_settings=matching,
                patches_by_cell=patches_by_cell,
                joined_endpoints=joined_endpoints,
                component_by_patch=component_by_patch,
                patch_by_id=patch_by_id,
            )
        )
    counts = Counter(value["classification"] for value in records)
    return {
        "cellXYZ": list(cell_xyz),
        "incidentUnresolvedInteriorTraceCount": len(records),
        "classifications": dict(sorted(counts.items())),
        "traces": records,
    }


def topology_utilization_summary(context: ClusterCellContext) -> dict[str, Any]:
    """Separate selected geometry continuity from retained graph continuity."""

    interior_endpoints = (
        2 * len(context.block.joins)
        + len(context.block.unresolved_interior_traces)
    )
    patches_by_cell_values: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in context.block.patches:
        patches_by_cell_values[patch.cell_xyz].append(patch)
    patches_by_cell = {
        key: tuple(value) for key, value in patches_by_cell_values.items()
    }
    joined_endpoints = {
        (patch_id, join.face)
        for join in context.block.joins
        for patch_id in (join.first_patch_id, join.second_patch_id)
    }
    component_by_patch = dict(context.block.component_by_patch)
    patch_by_id = {value.patch_id: value for value in context.block.patches}
    matching = TraceMatchSettings(orthogonal_fiber_equivalence=True)
    classifications = Counter(
        _open_trace_classification(
            context,
            boundary,
            matching_settings=matching,
            patches_by_cell=patches_by_cell,
            joined_endpoints=joined_endpoints,
            component_by_patch=component_by_patch,
            patch_by_id=patch_by_id,
        )["classification"]
        for boundary in context.block.unresolved_interior_traces
    )
    cells_with_patches = len(patches_by_cell)
    return {
        "cells": len(context.owner_by_cell),
        "cellsWithSelectedPatches": cells_with_patches,
        "emptySelectedCells": len(context.owner_by_cell) - cells_with_patches,
        "selectedPatches": len(context.block.patches),
        "retainedJoins": len(context.block.joins),
        "components": len(context.block.components),
        "interiorTraceEndpoints": interior_endpoints,
        "retainedTraceEndpoints": 2 * len(context.block.joins),
        "unresolvedInteriorTraceEndpoints": len(
            context.block.unresolved_interior_traces
        ),
        "retainedInteriorTraceFraction": round(
            2 * len(context.block.joins) / max(interior_endpoints, 1), 6
        ),
        "unresolvedSelectedTraceClassifications": dict(
            sorted(classifications.items())
        ),
    }


def refine_cell_neighborhood(
    context: ClusterCellContext,
    center_cell_xyz: Int3,
    *,
    radius_cells: int = 1,
    settings: CellRefinementSettings | None = None,
) -> dict[str, Any]:
    """Run conditional ICM rounds in a bounded neighborhood.

    The result is a proposal only.  It deliberately does not mutate the
    materialized graph; candidates must subsequently pass retained-topology
    replay before becoming a reconstruction variant.
    """

    if radius_cells < 0:
        raise ValueError("neighborhood radius must be nonnegative")
    resolved = settings or CellRefinementSettings()
    selected = dict(context.selected_by_cell)
    active = {
        (x, y, z)
        for z in range(
            max(0, center_cell_xyz[2] - radius_cells),
            min(context.grid.shape_cells_xyz[2], center_cell_xyz[2] + radius_cells + 1),
        )
        for y in range(
            max(0, center_cell_xyz[1] - radius_cells),
            min(context.grid.shape_cells_xyz[1], center_cell_xyz[1] + radius_cells + 1),
        )
        for x in range(
            max(0, center_cell_xyz[0] - radius_cells),
            min(context.grid.shape_cells_xyz[0], center_cell_xyz[0] + radius_cells + 1),
        )
    }
    changes: list[dict[str, Any]] = []
    sweeps = 0
    for sweep in range(resolved.maximum_sweeps):
        sweeps = sweep + 1
        changed = 0
        traversal = sorted(active, key=lambda value: (value[2], value[1], value[0]))
        if sweep % 2:
            traversal.reverse()
        for cell in traversal:
            current_index = selected[cell]
            scores = score_cell_candidates(
                context,
                cell,
                selected_by_cell=selected,
                settings=resolved,
            )
            best = scores[0]
            current = next(
                value
                for value in scores
                if value.source_configuration_index == current_index
            )
            if best.source_configuration_index == current_index:
                continue
            selected[cell] = best.source_configuration_index
            changed += 1
            changes.append(
                {
                    "phase": "single-cell",
                    "sweep": sweeps,
                    "cellXYZ": list(cell),
                    "priorSourceConfigurationIndex": current_index,
                    "selectedSourceConfigurationIndex": best.source_configuration_index,
                    "objectiveEnergyDelta": round(
                        best.objective_energy - current.objective_energy, 6
                    ),
                    "evidenceUtilizationBefore": round(
                        current.evidence_utilization, 6
                    ),
                    "evidenceUtilizationAfter": round(
                        best.evidence_utilization, 6
                    ),
                    "layerCountBefore": current.layer_count,
                    "layerCountAfter": best.layer_count,
                }
            )
        if not changed:
            break

    admissible_cache: dict[Int3, tuple[int, ...]] = {}

    def admissible_indices(cell: Int3) -> tuple[int, ...]:
        cached = admissible_cache.get(cell)
        if cached is not None:
            return cached
        indices = tuple(context.configuration_indices(cell))
        evidence = {value: context.evidence(cell, value) for value in indices}
        reference = context.selected_by_cell[cell]
        reference_covered, total = evidence[reference]
        reference_utilization = (
            reference_covered / total if total > 1.0e-12 else 0.0
        )
        oracle = max(
            (
                covered / candidate_total
                if candidate_total > 1.0e-12
                else 0.0
                for covered, candidate_total in evidence.values()
            ),
            default=0.0,
        )
        minimum = (
            max(
                resolved.minimum_oracle_coverage_fraction * oracle,
                reference_utilization - resolved.maximum_cell_utilization_drop,
            )
            if total >= resolved.minimum_evidence_mass_for_coverage_floor
            else 0.0
        )
        cached = tuple(
            value
            for value in indices
            if (
                evidence[value][0] / evidence[value][1]
                if evidence[value][1] > 1.0e-12
                else 0.0
            )
            + 1.0e-9
            >= minimum
        )
        if not cached:
            raise RuntimeError(f"coverage envelope rejected every candidate at {cell}")
        admissible_cache[cell] = cached
        return cached

    unary_cache: dict[tuple[Int3, int], float] = {}

    def unary_energy(cell: Int3, configuration_index: int) -> float:
        key = (cell, configuration_index)
        if key not in unary_cache:
            option = context.option(cell, configuration_index)
            covered, _ = context.evidence(cell, configuration_index)
            unary_cache[key] = (
                resolved.unary_scale * -option.log_weight
                - resolved.coverage_reward_scale * covered
            )
        return unary_cache[key]

    face_energy_cache: dict[tuple[Int3, int, Int3, int, int], float] = {}

    def face_energy(
        lower: Int3,
        lower_index: int,
        upper: Int3,
        upper_index: int,
        axis: int,
    ) -> float:
        key = (lower, lower_index, upper, upper_index, axis)
        if key not in face_energy_cache:
            score = _face_continuity(
                context,
                lower,
                context.option(lower, lower_index),
                upper,
                context.option(upper, upper_index),
                axis,
                1,
                pairwise_scale=resolved.pairwise_scale,
                matching_settings=TraceMatchSettings(),
            )
            continuity = (
                score.raw_reward_energy
                if resolved.pairwise_reward_normalization == "none"
                else score.trace_mean_reward_energy
            )
            face_energy_cache[key] = (
                continuity
                + resolved.resolved_unmatched_trace_penalty(
                    TraceMatchSettings()
                )
                * score.unmatched_trace_count
            )
        return face_energy_cache[key]

    def two_cell_energy(
        first: Int3,
        first_index: int,
        second: Int3,
        second_index: int,
    ) -> float:
        assignment = {first: first_index, second: second_index}
        value = unary_energy(first, first_index) + unary_energy(
            second, second_index
        )
        faces: set[tuple[Int3, Int3, int]] = set()
        for cell in (first, second):
            for axis in range(3):
                for direction in (-1, 1):
                    neighbor_values = list(cell)
                    neighbor_values[axis] += direction
                    neighbor = tuple(neighbor_values)
                    if not context.grid.contains_cell(neighbor):
                        continue
                    lower, upper = (
                        (neighbor, cell) if direction < 0 else (cell, neighbor)
                    )
                    faces.add((lower, upper, axis))
        for lower, upper, axis in faces:
            value += face_energy(
                lower,
                assignment.get(lower, selected[lower]),
                upper,
                assignment.get(upper, selected[upper]),
                axis,
            )
        return value

    active_pairs = sorted(
        (
            (cell, neighbor, axis)
            for cell in active
            for axis in range(3)
            if (
                neighbor := tuple(
                    cell[value] + (1 if value == axis else 0)
                    for value in range(3)
                )
            )
            in active
        ),
        key=lambda value: (
            value[0][2],
            value[0][1],
            value[0][0],
            value[2],
        ),
    )
    pair_sweeps = 0
    for pair_sweep in range(resolved.maximum_pair_sweeps):
        pair_sweeps = pair_sweep + 1
        changed = 0
        traversal = active_pairs if pair_sweep % 2 == 0 else list(reversed(active_pairs))
        for first, second, _ in traversal:
            prior_first = selected[first]
            prior_second = selected[second]
            prior_energy = two_cell_energy(
                first, prior_first, second, prior_second
            )
            best = (prior_energy, prior_first, prior_second)
            for first_index in admissible_indices(first):
                for second_index in admissible_indices(second):
                    energy = two_cell_energy(
                        first, first_index, second, second_index
                    )
                    candidate = (energy, first_index, second_index)
                    if candidate < best:
                        best = candidate
            best_energy, best_first, best_second = best
            if (
                best_energy >= prior_energy - 1.0e-9
                or (best_first, best_second) == (prior_first, prior_second)
            ):
                continue
            selected[first] = best_first
            selected[second] = best_second
            changed += int(best_first != prior_first) + int(
                best_second != prior_second
            )
            for cell, prior, replacement in (
                (first, prior_first, best_first),
                (second, prior_second, best_second),
            ):
                if prior == replacement:
                    continue
                before_covered, before_total = context.evidence(cell, prior)
                after_covered, after_total = context.evidence(cell, replacement)
                changes.append(
                    {
                        "phase": "adjacent-pair",
                        "sweep": pair_sweeps,
                        "cellXYZ": list(cell),
                        "priorSourceConfigurationIndex": prior,
                        "selectedSourceConfigurationIndex": replacement,
                        "pairObjectiveEnergyDelta": round(
                            best_energy - prior_energy, 6
                        ),
                        "evidenceUtilizationBefore": round(
                            before_covered / max(before_total, 1.0e-12), 6
                        ),
                        "evidenceUtilizationAfter": round(
                            after_covered / max(after_total, 1.0e-12), 6
                        ),
                        "layerCountBefore": len(context.option(cell, prior).patches),
                        "layerCountAfter": len(
                            context.option(cell, replacement).patches
                        ),
                    }
                )
        if not changed:
            break

    def utilization(mapping: Mapping[Int3, int]) -> tuple[float, float]:
        covered = sum(context.evidence(cell, mapping[cell])[0] for cell in active)
        total = sum(
            context.evidence(cell, next(iter(context.configuration_indices(cell))))[1]
            for cell in active
        )
        return covered, total

    initial_covered, total = utilization(context.selected_by_cell)
    final_covered, _ = utilization(selected)
    final_changes = [
        {
            "cellXYZ": list(cell),
            "priorSourceConfigurationIndex": context.selected_by_cell[cell],
            "selectedSourceConfigurationIndex": selected[cell],
        }
        for cell in sorted(active, key=lambda value: (value[2], value[1], value[0]))
        if selected[cell] != context.selected_by_cell[cell]
    ]

    def face_statistics(mapping: Mapping[Int3, int]) -> dict[str, Any]:
        pairs: set[tuple[Int3, Int3, int]] = set()
        for cell in active:
            for axis in range(3):
                for direction in (-1, 1):
                    neighbor_values = list(cell)
                    neighbor_values[axis] += direction
                    neighbor = tuple(neighbor_values)
                    if not context.grid.contains_cell(neighbor):
                        continue
                    lower, upper = (
                        (neighbor, cell) if direction < 0 else (cell, neighbor)
                    )
                    pairs.add((lower, upper, axis))
        scored_pairs = [
            (
                lower in active and upper in active,
                _face_continuity(
                context,
                lower,
                context.selected_option(lower, mapping),
                upper,
                context.selected_option(upper, mapping),
                axis,
                1,
                pairwise_scale=resolved.pairwise_scale,
                matching_settings=TraceMatchSettings(),
                ),
            )
            for lower, upper, axis in sorted(
                pairs,
                key=lambda value: (
                    value[0][2],
                    value[0][1],
                    value[0][0],
                    value[2],
                ),
            )
        ]

        def summarize(scores: list[FaceContinuityScore]) -> dict[str, Any]:
            endpoints = sum(
                value.first_trace_count + value.second_trace_count
                for value in scores
            )
            matched_endpoints = 2 * sum(value.match_count for value in scores)
            return {
                "faceCount": len(scores),
                "traceEndpointCount": endpoints,
                "matchedTraceEndpointCount": matched_endpoints,
                "unmatchedTraceEndpointCount": sum(
                    value.unmatched_trace_count for value in scores
                ),
                "alignmentUtilization": round(
                    matched_endpoints / max(endpoints, 1), 6
                ),
                "rawContinuityEnergy": round(
                    sum(value.raw_reward_energy for value in scores), 6
                ),
                "traceMeanContinuityEnergy": round(
                    sum(value.trace_mean_reward_energy for value in scores), 6
                ),
            }

        all_scores = [value for _, value in scored_pairs]
        interior_scores = [value for interior, value in scored_pairs if interior]
        boundary_scores = [value for interior, value in scored_pairs if not interior]
        return {
            **summarize(all_scores),
            "interior": summarize(interior_scores),
            "boundary": summarize(boundary_scores),
        }

    initial_faces = face_statistics(context.selected_by_cell)
    final_faces = face_statistics(selected)
    initial_center = next(
        value
        for value in score_cell_candidates(
            context,
            center_cell_xyz,
            selected_by_cell=context.selected_by_cell,
            settings=resolved,
        )
        if value.source_configuration_index
        == context.selected_by_cell[center_cell_xyz]
    )
    final_center = next(
        value
        for value in score_cell_candidates(
            context,
            center_cell_xyz,
            selected_by_cell=selected,
            settings=resolved,
        )
        if value.source_configuration_index == selected[center_cell_xyz]
    )
    return {
        "status": "selection-proposal-requires-topology-replay",
        "centerCellXYZ": list(center_cell_xyz),
        "radiusCells": radius_cells,
        "activeCellCount": len(active),
        "completedSweeps": sweeps,
        "completedPairSweeps": pair_sweeps,
        "transitionCount": len(changes),
        "netChangedCellCount": len(final_changes),
        "initialCoveredEvidenceMass": round(initial_covered, 6),
        "finalCoveredEvidenceMass": round(final_covered, 6),
        "totalEvidenceMass": round(total, 6),
        "initialEvidenceUtilization": round(initial_covered / max(total, 1.0e-12), 6),
        "finalEvidenceUtilization": round(final_covered / max(total, 1.0e-12), 6),
        "initialFaceContinuity": initial_faces,
        "finalFaceContinuity": final_faces,
        "centerInitial": initial_center.record(),
        "centerFinal": final_center.record(),
        "netChanges": final_changes,
        "transitions": changes,
    }


@dataclass(frozen=True, slots=True)
class _ReplayMatchingPolicy:
    strict_settings: TraceMatchSettings
    quarter_turn_enabled: bool
    maximum_quarter_turn_normal_degrees: float
    maximum_quarter_turn_fiber_degrees: float
    quarter_turn_settings: TraceMatchSettings


def _replay_matching_policy(
    context: ClusterCellContext,
) -> _ReplayMatchingPolicy:
    cluster_manifest = json.loads(
        (context.cluster_root / "cluster-reselection-v1.json").read_text()
    )
    policy = cluster_manifest["identity"]["seamMatchingPolicy"]
    parallel = policy["parallelMatching"]
    quarter = policy["quarterTurnAdmission"]
    return _ReplayMatchingPolicy(
        TraceMatchSettings(
            orthogonal_fiber_equivalence=bool(
                parallel["orthogonalFiberEquivalence"]
            ),
            maximum_absolute_normal_angle_radians=math.radians(
                float(parallel["maximumNormalAngleDegrees"])
            ),
            maximum_absolute_fiber_residual_radians=math.radians(
                float(parallel["maximumFiberResidualDegrees"])
            ),
        ),
        bool(quarter["enabled"]),
        float(quarter["maximumNormalAngleDegrees"]),
        float(quarter["maximumFiberFrameResidualDegrees"]),
        TraceMatchSettings(orthogonal_fiber_equivalence=True),
    )


def _resolve_replay_proposal(
    context: ClusterCellContext,
    proposal: Mapping[str, Any],
) -> tuple[Int3, frozenset[Int3], dict[Int3, int]]:
    center = tuple(int(value) for value in proposal["centerCellXYZ"])
    radius = int(proposal["radiusCells"])
    if radius < 0:
        raise ValueError("topology replay radius must be nonnegative")
    active_values = proposal.get("activeCellXYZ")
    active = (
        {
            tuple(int(value) for value in cell)
            for cell in active_values
        }
        if active_values is not None
        else {
            (x, y, z)
            for z in range(
                max(0, center[2] - radius),
                min(context.grid.shape_cells_xyz[2], center[2] + radius + 1),
            )
            for y in range(
                max(0, center[1] - radius),
                min(context.grid.shape_cells_xyz[1], center[1] + radius + 1),
            )
            for x in range(
                max(0, center[0] - radius),
                min(context.grid.shape_cells_xyz[0], center[0] + radius + 1),
            )
        }
    )
    if not active or center not in active:
        raise ValueError("topology replay active cells must include its center")
    if any(not context.grid.contains_cell(value) for value in active):
        raise ValueError("topology replay active cell lies outside the grid")
    selected = dict(context.selected_by_cell)
    changed_cells: set[Int3] = set()
    for record in proposal["netChanges"]:
        cell = tuple(int(value) for value in record["cellXYZ"])
        if cell not in active:
            raise ValueError("refinement proposal changes a cell outside its cube")
        if cell in changed_cells:
            raise ValueError("refinement proposal changes one cell more than once")
        changed_cells.add(cell)
        selected[cell] = int(record["selectedSourceConfigurationIndex"])
    return center, frozenset(active), selected


def _active_faces(
    context: ClusterCellContext,
    active: frozenset[Int3],
) -> frozenset[tuple[Int3, Int3, int]]:
    faces: set[tuple[Int3, Int3, int]] = set()
    for cell in active:
        for axis in range(3):
            for direction in (-1, 1):
                neighbor_values = list(cell)
                neighbor_values[axis] += direction
                neighbor = tuple(neighbor_values)
                if not context.grid.contains_cell(neighbor):
                    continue
                lower, upper = (
                    (neighbor, cell) if direction < 0 else (cell, neighbor)
                )
                faces.add((lower, upper, axis))
    return frozenset(faces)


def _match_key(value: TraceMatch) -> tuple[int, int, int, Int3]:
    return (
        value.first_patch_id,
        value.second_patch_id,
        value.face.axis,
        value.face.anchor_xyz,
    )


def _local_replay_candidates(
    context: ClusterCellContext,
    patches_by_cell: Mapping[Int3, tuple[ClippedPatch, ...]],
    faces: frozenset[tuple[Int3, Int3, int]],
    policy: _ReplayMatchingPolicy,
) -> tuple[tuple[TraceMatch, ...], tuple[TraceMatch, ...]]:
    strict_candidates: list[TraceMatch] = []
    quarter_candidates: list[TraceMatch] = []
    for lower, upper, axis in sorted(
        faces,
        key=lambda value: (
            value[0][2],
            value[0][1],
            value[0][0],
            value[2],
        ),
    ):
        face = cell_face(lower, axis, 1)
        first = patches_by_cell.get(lower, ())
        second = patches_by_cell.get(upper, ())
        if not first and not second:
            continue
        strict_on_face: tuple[TraceMatch, ...] = ()
        try:
            strict = align_face_patches(
                first,
                second,
                face,
                policy.strict_settings,
                grid=context.grid,
            )
            strict_on_face = strict.matches
            strict_candidates.extend(strict_on_face)
        except ValueError:
            pass
        if not policy.quarter_turn_enabled:
            continue
        try:
            quarter = align_face_patches(
                first,
                second,
                face,
                policy.quarter_turn_settings,
                grid=context.grid,
            )
        except ValueError:
            continue
        strict_keys = {_match_key(value) for value in strict_on_face}
        quarter_candidates.extend(
            value
            for value in quarter.matches
            if _match_key(value) not in strict_keys
            and value.fiber_quarter_turn is True
            and math.degrees(value.normal_angle_radians)
            <= policy.maximum_quarter_turn_normal_degrees
            and value.fiber_angle_radians is not None
            and math.degrees(value.fiber_angle_radians)
            <= policy.maximum_quarter_turn_fiber_degrees
        )
    return tuple(strict_candidates), tuple(quarter_candidates)


def _region_statistics(
    block: SurfaceBlock,
    active: frozenset[Int3],
    center: Int3,
) -> dict[str, Any]:
    def face_touches_active(face: GridFace) -> bool:
        lower, upper = face.adjacent_cells()
        return lower in active or upper in active

    incident_joins = [
        value for value in block.joins if face_touches_active(value.face)
    ]
    incident_open = [
        value
        for value in block.unresolved_interior_traces
        if face_touches_active(value.trace.face)
    ]
    endpoints = 2 * len(incident_joins) + len(incident_open)
    component_by_patch = dict(block.component_by_patch)
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in block.components
    }
    center_components = sorted(
        {
            component_by_patch[value.patch_id]
            for value in block.patches
            if value.cell_xyz == center
        }
    )
    return {
        "patchesInActiveCells": sum(
            value.cell_xyz in active for value in block.patches
        ),
        "incidentRetainedJoins": len(incident_joins),
        "incidentUnresolvedTraceEndpoints": len(incident_open),
        "incidentTraceEndpoints": endpoints,
        "incidentRetainedTraceFraction": round(
            2 * len(incident_joins) / max(endpoints, 1), 6
        ),
        "globalComponents": len(block.components),
        "centerComponentIds": center_components,
        "centerComponentPatchCounts": [
            component_sizes[value] for value in center_components
        ],
    }


@dataclass(slots=True)
class FrozenReplayWorkspace:
    """Cache immutable graph cuts for exact local refinement trials."""

    context: ClusterCellContext
    _patch_by_id: dict[int, ClippedPatch] = field(init=False)
    _patches_by_cell: dict[Int3, tuple[ClippedPatch, ...]] = field(init=False)
    _baseline_component_by_patch: dict[int, int] = field(init=False)
    _policy: _ReplayMatchingPolicy = field(init=False)
    _cut_cache: dict[frozenset[Int3], FrozenTopologyCut] = field(
        init=False, default_factory=dict
    )

    def __post_init__(self) -> None:
        self._patch_by_id = {
            value.patch_id: value for value in self.context.block.patches
        }
        by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
        for patch in self.context.block.patches:
            by_cell[patch.cell_xyz].append(patch)
        self._patches_by_cell = {
            cell: tuple(
                sorted(
                    values,
                    key=lambda patch: (
                        patch.estimate.height_from_cell_center,
                        patch.patch_id,
                    ),
                )
            )
            for cell, values in by_cell.items()
        }
        self._baseline_component_by_patch = dict(
            self.context.block.component_by_patch
        )
        self._policy = _replay_matching_policy(self.context)

    def _frozen_cut(
        self,
        active: frozenset[Int3],
        faces: frozenset[tuple[Int3, Int3, int]],
    ) -> FrozenTopologyCut:
        cached = self._cut_cache.get(active)
        if cached is not None:
            return cached
        mutable_patch_ids = {
            patch.patch_id
            for cell in active
            for patch in self._patches_by_cell.get(cell, ())
        }
        anchor_observations: set[tuple[int, Any]] = set()
        for lower, upper, axis in faces:
            lower_active = lower in active
            upper_active = upper in active
            if lower_active == upper_active:
                continue
            outside_cell = upper if lower_active else lower
            face = cell_face(lower, axis, 1)
            for patch in self._patches_by_cell.get(outside_cell, ()):
                trace = patch.trace_on(face)
                if trace is None:
                    continue
                anchor_observations.add((patch.patch_id, trace.first.edge))
                anchor_observations.add((patch.patch_id, trace.second.edge))
        cut = freeze_topology_outside_patches(
            self.context.block.patches,
            self.context.block.joins,
            mutable_patch_ids,
            anchor_observations,
        )
        self._cut_cache[active] = cut
        return cut

    def replay(self, proposal: Mapping[str, Any]) -> dict[str, Any]:
        center, active, selected = _resolve_replay_proposal(
            self.context, proposal
        )
        faces = _active_faces(self.context, active)
        cut = self._frozen_cut(active, faces)
        replacement_patches = tuple(
            patch
            for cell in sorted(
                active, key=lambda value: (value[2], value[1], value[0])
            )
            for patch in self.context.option(cell, selected[cell]).patches
        )
        frozen_ids = set(cut.frozen_patch_ids)
        replacement_ids = {value.patch_id for value in replacement_patches}
        if frozen_ids & replacement_ids:
            raise ValueError("replacement patch IDs collide with frozen geometry")
        anchor_patches = tuple(
            self._patch_by_id[value] for value in cut.anchor_patch_ids
        )
        local_patches = (*anchor_patches, *replacement_patches)
        by_cell_values: dict[Int3, list[ClippedPatch]] = defaultdict(list)
        for patch in local_patches:
            by_cell_values[patch.cell_xyz].append(patch)
        patches_by_cell = {
            cell: tuple(
                sorted(
                    values,
                    key=lambda patch: (
                        patch.estimate.height_from_cell_center,
                        patch.patch_id,
                    ),
                )
            )
            for cell, values in by_cell_values.items()
        }
        strict_candidates, quarter_candidates = _local_replay_candidates(
            self.context,
            patches_by_cell,
            faces,
            self._policy,
        )
        strict = select_joins_with_frozen_topology(
            local_patches,
            strict_candidates,
            cut.seed,
        )
        strict_keys = frozenset(_match_key(value) for value in strict.joins)
        final = select_joins_with_frozen_topology(
            local_patches,
            (*strict_candidates, *quarter_candidates),
            cut.seed,
            fixed_join_keys=strict_keys,
        )

        local_component_by_patch = dict(final.component_by_patch)
        local_members: dict[int, list[int]] = defaultdict(list)
        for patch_id, component_id in final.component_by_patch:
            local_members[component_id].append(patch_id)
        cut_component_by_patch = dict(cut.component_by_patch)
        cut_component_counts = dict(cut.component_patch_counts)
        global_component_by_local: dict[int, int] = {}
        global_component_sizes: dict[int, int] = {}
        frozen_final_component: dict[int, int] = {}
        for local_component, members in local_members.items():
            frozen_components = {
                cut_component_by_patch[value]
                for value in members
                if value in frozen_ids
            }
            replacement_members = [
                value for value in members if value in replacement_ids
            ]
            candidates = [*frozen_components, *replacement_members]
            if not candidates:
                raise RuntimeError("local topology component has no physical members")
            global_component = min(candidates)
            global_component_by_local[local_component] = global_component
            global_component_sizes[global_component] = (
                sum(cut_component_counts[value] for value in frozen_components)
                + len(replacement_members)
            )
            for frozen_component in frozen_components:
                frozen_final_component[frozen_component] = global_component
        participating = {
            int(value.key) for value in cut.seed.components
        }
        for frozen_component in cut_component_counts:
            if frozen_component not in participating:
                frozen_final_component[frozen_component] = frozen_component

        baseline_frozen_components: dict[int, set[int]] = defaultdict(set)
        baseline_frozen_patch_counts: Counter[int] = Counter()
        baseline_by_frozen_component: dict[int, int] = {}
        for patch_id in cut.frozen_patch_ids:
            baseline_component = self._baseline_component_by_patch[patch_id]
            frozen_component = cut_component_by_patch[patch_id]
            baseline_frozen_components[baseline_component].add(frozen_component)
            baseline_frozen_patch_counts[baseline_component] += 1
            prior = baseline_by_frozen_component.setdefault(
                frozen_component, baseline_component
            )
            if prior != baseline_component:
                raise RuntimeError(
                    "one frozen component spans multiple baseline components"
                )
        split_components: list[dict[str, Any]] = []
        replayed_to_baseline: dict[int, set[int]] = defaultdict(set)
        for baseline_component, frozen_components in (
            baseline_frozen_components.items()
        ):
            replayed_components = {
                frozen_final_component[value] for value in frozen_components
            }
            for replayed_component in replayed_components:
                replayed_to_baseline[replayed_component].add(baseline_component)
            if len(replayed_components) > 1:
                split_components.append(
                    {
                        "baselineComponentId": baseline_component,
                        "frozenPatchCount": baseline_frozen_patch_counts[
                            baseline_component
                        ],
                        "replayedComponentIds": sorted(replayed_components),
                    }
                )
        split_components.sort(
            key=lambda value: (
                -len(value["replayedComponentIds"]),
                -value["frozenPatchCount"],
                value["baselineComponentId"],
            )
        )
        component_partition = {
            "frozenExteriorPatchCount": len(cut.frozen_patch_ids),
            "baselineComponentsWithFrozenPatches": len(
                baseline_frozen_components
            ),
            "splitBaselineComponentCount": len(split_components),
            "maximumSplitParts": max(
                (
                    len(value["replayedComponentIds"])
                    for value in split_components
                ),
                default=1,
            ),
            "mergedReplayComponentCount": sum(
                len(value) > 1 for value in replayed_to_baseline.values()
            ),
            "splitComponents": split_components,
        }

        def face_touches_active(face: GridFace) -> bool:
            lower, upper = face.adjacent_cells()
            return lower in active or upper in active

        joined_endpoints = {
            (patch_id, join.face)
            for join in final.joins
            for patch_id in (join.first_patch_id, join.second_patch_id)
        }
        incident_open = sum(
            not self.context.block.bounds.contains_face_on_boundary(trace.face)
            and face_touches_active(trace.face)
            and (patch.patch_id, trace.face) not in joined_endpoints
            for patch in local_patches
            for trace in patch.traces
        )
        incident_joins = len(final.joins)
        endpoints = 2 * incident_joins + incident_open
        center_components = sorted(
            {
                global_component_by_local[
                    local_component_by_patch[patch.patch_id]
                ]
                for patch in replacement_patches
                if patch.cell_xyz == center
            }
        )
        replayed = {
            "patchesInActiveCells": len(replacement_patches),
            "incidentRetainedJoins": incident_joins,
            "incidentUnresolvedTraceEndpoints": incident_open,
            "incidentTraceEndpoints": endpoints,
            "incidentRetainedTraceFraction": round(
                2 * incident_joins / max(endpoints, 1), 6
            ),
            "globalComponents": final.component_count,
            "centerComponentIds": center_components,
            "centerComponentPatchCounts": [
                global_component_sizes[value] for value in center_components
            ],
        }
        baseline = _region_statistics(self.context.block, active, center)
        deferred = Counter(value.reason for value in final.deferred_joins)
        summary = {
            "status": "topology-safe-local-replay",
            "activeCellCount": len(active),
            "replacedPatchCount": sum(
                len(self._patches_by_cell.get(value, ())) for value in active
            ),
            "replacementPatchCount": len(replacement_patches),
            "frozenExteriorJoinCount": len(cut.frozen_join_keys),
            "frozenExteriorJoinsPreserved": len(cut.frozen_join_keys),
            "strictCandidateJoinCount": len(strict_candidates),
            "quarterTurnCandidateJoinCount": len(quarter_candidates),
            "strictRetainedJoinCount": len(strict.joins),
            "finalLocalRetainedJoinCount": len(final.joins),
            "deferredLocalJoinsByReason": dict(sorted(deferred.items())),
            "frozenExteriorComponentPartition": component_partition,
            "baseline": baseline,
            "replayed": replayed,
            "delta": {
                key: replayed[key] - baseline[key]
                for key in (
                    "patchesInActiveCells",
                    "incidentRetainedJoins",
                    "incidentUnresolvedTraceEndpoints",
                    "globalComponents",
                )
            },
            "acceptance": {
                "topologySafe": True,
                "allExteriorJoinsPreserved": True,
                "frozenExteriorConnectivityPreserved": not split_components,
                "incidentTraceUtilizationNondecreasing": (
                    replayed["incidentRetainedTraceFraction"]
                    >= baseline["incidentRetainedTraceFraction"]
                ),
            },
        }
        return summary


def replay_neighborhood_topology_state(
    context: ClusterCellContext,
    proposal: Mapping[str, Any],
) -> TopologyReplay:
    """Reopen one bounded neighborhood while preserving every exterior join.

    All patches in the active cube are replaced from the proposed candidate
    selection.  Joins with both endpoints outside that cube remain immutable.
    Strict face matches are admitted first, then explicitly gated orthogonal
    fiber matches.  Both passes use the full collision, crossing, and
    orientability selector, making this an exact local topology replay rather
    than a face-score proxy.
    """

    center, active, selected = _resolve_replay_proposal(context, proposal)

    removed_patch_ids = {
        value.patch_id
        for value in context.block.patches
        if value.cell_xyz in active
    }
    outside_patches = tuple(
        value
        for value in context.block.patches
        if value.patch_id not in removed_patch_ids
    )
    replacement_patches = tuple(
        patch
        for cell in sorted(active, key=lambda value: (value[2], value[1], value[0]))
        for patch in context.option(cell, selected[cell]).patches
    )
    outside_ids = {value.patch_id for value in outside_patches}
    replacement_ids = {value.patch_id for value in replacement_patches}
    if outside_ids & replacement_ids:
        raise ValueError("replacement patch IDs collide with frozen geometry")
    outside_joins = tuple(
        value
        for value in context.block.joins
        if value.first_patch_id in outside_ids
        and value.second_patch_id in outside_ids
    )
    trial_patches = (*outside_patches, *replacement_patches)
    base = surface_block_from_retained_joins(
        context.grid,
        context.block.bounds,
        trial_patches,
        outside_joins,
    )

    patches_by_cell_values: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in trial_patches:
        patches_by_cell_values[patch.cell_xyz].append(patch)
    patches_by_cell = {
        key: tuple(
            sorted(
                value,
                key=lambda patch: (
                    patch.estimate.height_from_cell_center,
                    patch.patch_id,
                ),
            )
        )
        for key, value in patches_by_cell_values.items()
    }

    faces = _active_faces(context, active)
    strict_candidates, quarter_candidates = _local_replay_candidates(
        context,
        patches_by_cell,
        faces,
        _replay_matching_policy(context),
    )

    strict_block = extend_surface_block_joins(base, strict_candidates)
    final = extend_surface_block_joins(strict_block, quarter_candidates)
    outside_keys = {_match_key(value) for value in outside_joins}
    final_keys = {_match_key(value) for value in final.joins}
    if not outside_keys <= final_keys:
        raise RuntimeError("local topology replay lost a frozen exterior join")

    baseline = _region_statistics(context.block, active, center)
    replayed = _region_statistics(final, active, center)
    baseline_component_by_patch = dict(context.block.component_by_patch)
    replayed_component_by_patch = dict(final.component_by_patch)
    frozen_by_baseline_component: dict[int, list[int]] = defaultdict(list)
    for patch_id in outside_ids:
        frozen_by_baseline_component[baseline_component_by_patch[patch_id]].append(
            patch_id
        )
    split_components: list[dict[str, Any]] = []
    replayed_to_baseline: dict[int, set[int]] = defaultdict(set)
    for baseline_component, patch_ids in frozen_by_baseline_component.items():
        replayed_components = {
            replayed_component_by_patch[patch_id] for patch_id in patch_ids
        }
        for replayed_component in replayed_components:
            replayed_to_baseline[replayed_component].add(baseline_component)
        if len(replayed_components) > 1:
            split_components.append(
                {
                    "baselineComponentId": baseline_component,
                    "frozenPatchCount": len(patch_ids),
                    "replayedComponentIds": sorted(replayed_components),
                }
            )
    split_components.sort(
        key=lambda value: (
            -len(value["replayedComponentIds"]),
            -value["frozenPatchCount"],
            value["baselineComponentId"],
        )
    )
    component_partition = {
        "frozenExteriorPatchCount": len(outside_ids),
        "baselineComponentsWithFrozenPatches": len(frozen_by_baseline_component),
        "splitBaselineComponentCount": len(split_components),
        "maximumSplitParts": max(
            (len(value["replayedComponentIds"]) for value in split_components),
            default=1,
        ),
        "mergedReplayComponentCount": sum(
            len(value) > 1 for value in replayed_to_baseline.values()
        ),
        "splitComponents": split_components,
    }
    deferred = Counter(value.reason for value in final.deferred_joins)
    summary = {
        "status": "topology-safe-local-replay",
        "activeCellCount": len(active),
        "replacedPatchCount": len(removed_patch_ids),
        "replacementPatchCount": len(replacement_patches),
        "frozenExteriorJoinCount": len(outside_joins),
        "frozenExteriorJoinsPreserved": len(outside_keys),
        "strictCandidateJoinCount": len(strict_candidates),
        "quarterTurnCandidateJoinCount": len(quarter_candidates),
        "strictRetainedJoinCount": len(strict_block.joins) - len(outside_joins),
        "finalLocalRetainedJoinCount": len(final.joins) - len(outside_joins),
        "deferredLocalJoinsByReason": dict(sorted(deferred.items())),
        "frozenExteriorComponentPartition": component_partition,
        "baseline": baseline,
        "replayed": replayed,
        "delta": {
            key: replayed[key] - baseline[key]
            for key in (
                "patchesInActiveCells",
                "incidentRetainedJoins",
                "incidentUnresolvedTraceEndpoints",
                "globalComponents",
            )
        },
        "acceptance": {
            "topologySafe": True,
            "allExteriorJoinsPreserved": outside_keys <= final_keys,
            "frozenExteriorConnectivityPreserved": not split_components,
            "incidentTraceUtilizationNondecreasing": (
                replayed["incidentRetainedTraceFraction"]
                >= baseline["incidentRetainedTraceFraction"]
            ),
        },
    }
    return TopologyReplay(
        summary,
        final,
        selected,
        frozenset(active),
        frozenset(removed_patch_ids),
        frozenset(replacement_ids),
    )


def replay_neighborhood_topology(
    context: ClusterCellContext,
    proposal: Mapping[str, Any],
) -> dict[str, Any]:
    """Return the auditable summary of an exact local topology replay."""

    return replay_neighborhood_topology_state(context, proposal).summary


def anneal_topology_safe_refinement(
    context: ClusterCellContext,
    proposal: Mapping[str, Any],
    *,
    replay_workspace: FrozenReplayWorkspace | None = None,
) -> dict[str, Any]:
    """Convert a face-scored proposal into conservative exact replay rounds.

    Round one admits replacements that are individually non-worsening under
    exact topology.  Round two retries the focal replacement against that
    jointly replayed support set, allowing topology gains around the hole to
    fund a previously blocked evidence-backed stack while preserving the
    original neighborhood invariants.
    """

    center = tuple(int(value) for value in proposal["centerCellXYZ"])
    workspace = replay_workspace or FrozenReplayWorkspace(context)
    changes = [
        {
            "cellXYZ": [int(value) for value in record["cellXYZ"]],
            "priorSourceConfigurationIndex": int(
                record["priorSourceConfigurationIndex"]
            ),
            "selectedSourceConfigurationIndex": int(
                record["selectedSourceConfigurationIndex"]
            ),
        }
        for record in proposal["netChanges"]
    ]

    def evidence_record(records: list[dict[str, Any]]) -> dict[str, Any]:
        before = 0.0
        after = 0.0
        total = 0.0
        for record in records:
            cell = tuple(int(value) for value in record["cellXYZ"])
            prior = int(record["priorSourceConfigurationIndex"])
            replacement = int(record["selectedSourceConfigurationIndex"])
            prior_covered, prior_total = context.evidence(cell, prior)
            replacement_covered, replacement_total = context.evidence(
                cell, replacement
            )
            if not math.isclose(
                prior_total,
                replacement_total,
                rel_tol=0.0,
                abs_tol=1.0e-6,
            ):
                raise ValueError("candidate evidence totals disagree within a cell")
            before += prior_covered
            after += replacement_covered
            total += prior_total
        return {
            "coveredEvidenceMassBefore": round(before, 6),
            "coveredEvidenceMassAfter": round(after, 6),
            "coveredEvidenceMassDelta": round(after - before, 6),
            "totalEvidenceMass": round(total, 6),
            "evidenceUtilizationBefore": round(
                before / max(total, 1.0e-12), 6
            ),
            "evidenceUtilizationAfter": round(
                after / max(total, 1.0e-12), 6
            ),
        }

    def replay(
        records: list[dict[str, Any]],
        *,
        active: set[Int3] | None = None,
    ) -> dict[str, Any]:
        cells = (
            active
            if active is not None
            else {
                tuple(int(value) for value in record["cellXYZ"])
                for record in records
            }
        )
        if not cells:
            raise ValueError("topology annealing replay requires active cells")
        return workspace.replay(
            {
                "centerCellXYZ": list(center if center in cells else min(cells)),
                "radiusCells": int(proposal["radiusCells"]),
                "activeCellXYZ": [
                    list(value)
                    for value in sorted(
                        cells, key=lambda cell: (cell[2], cell[1], cell[0])
                    )
                ],
                "netChanges": records,
            },
        )

    def nonworsening(value: Mapping[str, Any]) -> bool:
        baseline = value["baseline"]
        replayed = value["replayed"]
        return bool(
            value["acceptance"]["topologySafe"]
            and value["acceptance"]["allExteriorJoinsPreserved"]
            and value["acceptance"]["frozenExteriorConnectivityPreserved"]
            and replayed["incidentUnresolvedTraceEndpoints"]
            <= baseline["incidentUnresolvedTraceEndpoints"]
            and replayed["globalComponents"] <= baseline["globalComponents"]
        )

    trials: list[dict[str, Any]] = []
    support: list[dict[str, Any]] = []
    center_change: dict[str, Any] | None = None
    for record in changes:
        cell = tuple(int(value) for value in record["cellXYZ"])
        trial = replay([record])
        evidence = evidence_record([record])
        safe = nonworsening(trial)
        trial_record = {
            **record,
            "evidence": evidence,
            "topology": trial,
            "individuallyTopologyNonworsening": safe,
        }
        trials.append(trial_record)
        if cell == center:
            center_change = record
        elif safe:
            support.append(record)

    round_one: dict[str, Any] | None = None
    round_one_accepted = False
    if support:
        replayed = replay(support)
        evidence = evidence_record(support)
        round_one_accepted = nonworsening(replayed) and (
            evidence["coveredEvidenceMassDelta"] >= 0.0
        )
        round_one = {
            "name": "individually-safe-support",
            "changes": support,
            "evidence": evidence,
            "topology": replayed,
            "accepted": round_one_accepted,
        }

    support_greedy_round: dict[str, Any] | None = None
    accepted = list(support) if round_one_accepted else []
    if support and not round_one_accepted:
        single_evidence_delta = {
            tuple(int(value) for value in trial["cellXYZ"]): float(
                trial["evidence"]["coveredEvidenceMassDelta"]
            )
            for trial in trials
        }
        remaining = sorted(
            support,
            key=lambda value: (
                -single_evidence_delta[
                    tuple(int(item) for item in value["cellXYZ"])
                ],
                value["cellXYZ"][2],
                value["cellXYZ"][1],
                value["cellXYZ"][0],
            ),
        )
        greedy_trials: list[dict[str, Any]] = []
        accepted_topology: dict[str, Any] | None = None
        accepted_evidence: dict[str, Any] | None = None
        for sweep in range(2):
            changed = False
            deferred: list[dict[str, Any]] = []
            for record in remaining:
                candidate = [*accepted, record]
                active = {
                    tuple(int(value) for value in item["cellXYZ"])
                    for item in candidate
                }
                replayed = replay(candidate, active=active)
                evidence = evidence_record(candidate)
                admitted = nonworsening(replayed) and (
                    evidence["coveredEvidenceMassDelta"] >= -1.0e-9
                )
                greedy_trials.append(
                    {
                        "sweep": sweep + 1,
                        "addedChange": record,
                        "candidateChangeCount": len(candidate),
                        "evidence": evidence,
                        "topology": replayed,
                        "accepted": admitted,
                    }
                )
                if admitted:
                    accepted = candidate
                    accepted_topology = replayed
                    accepted_evidence = evidence
                    changed = True
                else:
                    deferred.append(record)
            remaining = deferred
            if not changed or not remaining:
                break
        support_greedy_round = {
            "name": "collision-safe-greedy-support",
            "trials": greedy_trials,
            "changes": accepted,
            "evidence": accepted_evidence,
            "topology": accepted_topology,
            "deferredChanges": remaining,
            "accepted": bool(accepted),
        }
    round_two: dict[str, Any] | None = None
    if center_change is not None:
        candidate = [*accepted, center_change]
        active = {
            tuple(int(value) for value in record["cellXYZ"])
            for record in candidate
        }
        replayed = replay(candidate, active=active)
        evidence = evidence_record(candidate)
        accepted_center = nonworsening(replayed) and (
            evidence["coveredEvidenceMassDelta"] > 0.0
        )
        round_two = {
            "name": "focal-retry-with-safe-support",
            "changes": candidate,
            "evidence": evidence,
            "topology": replayed,
            "accepted": accepted_center,
        }
        if accepted_center:
            accepted = candidate

    coordinated_round: dict[str, Any] | None = None
    if changes and {tuple(value["cellXYZ"]) for value in changes} != {
        tuple(value["cellXYZ"]) for value in accepted
    }:
        active = {
            tuple(int(value) for value in record["cellXYZ"])
            for record in changes
        }
        replayed = replay(changes, active=active)
        evidence = evidence_record(changes)
        accepted_evidence_delta = (
            evidence_record(accepted)["coveredEvidenceMassDelta"]
            if accepted
            else 0.0
        )
        accepted_coordinated = nonworsening(replayed) and (
            evidence["coveredEvidenceMassDelta"] > accepted_evidence_delta
        )
        coordinated_round = {
            "name": "coordinated-net-changes",
            "changes": changes,
            "evidence": evidence,
            "topology": replayed,
            "acceptedEvidenceDeltaToBeat": accepted_evidence_delta,
            "accepted": accepted_coordinated,
        }
        if accepted_coordinated:
            accepted = changes

    return {
        "status": "topology-safe-annealing-proposal",
        "singleCellTrials": trials,
        "roundOne": round_one,
        "supportGreedyRound": support_greedy_round,
        "roundTwo": round_two,
        "coordinatedRound": coordinated_round,
        "acceptedChanges": accepted,
        "acceptedChangeCount": len(accepted),
        "focalCellAccepted": any(
            tuple(int(value) for value in record["cellXYZ"]) == center
            for record in accepted
        ),
    }


def run_cell_refinement_diagnostic(
    cluster_root: str | Path,
    materialized_root: str | Path,
    output_root: str | Path,
    *,
    cell_xyz: Int3,
    component_id: int | None = None,
    neighborhood_radius_cells: int = 1,
    settings: CellRefinementSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write an auditable single-cell diagnosis and conditional refinement proposal."""

    resolved = settings or CellRefinementSettings()
    cluster = Path(cluster_root).resolve()
    materialized = Path(materialized_root).resolve()
    output = Path(output_root).resolve()
    identity: dict[str, Any] = {
        "schema": CELL_REFINEMENT_DIAGNOSTIC_SCHEMA,
        "version": CELL_REFINEMENT_DIAGNOSTIC_VERSION,
        "clusterRoot": str(cluster),
        "clusterManifestSha256": sha256_file(cluster / "cluster-reselection-v1.json"),
        "clusterDataSha256": sha256_file(cluster / "cluster-reselection-v1.npz"),
        "materializedRoot": str(materialized),
        "surfaceGraphManifestSha256": sha256_file(materialized / "surface-graph-v1.json"),
        "surfaceGraphDataSha256": sha256_file(materialized / "surface-graph-v1.npz"),
        "cellXYZ": list(cell_xyz),
        "componentId": component_id,
        "neighborhoodRadiusCells": neighborhood_radius_cells,
        "settings": resolved.record(),
        "implementationSha256": {
            name: sha256_file(Path(__file__).resolve().parent / name)
            for name in (
                "cell_refinement.py",
                "boundary_topology.py",
                "selection.py",
                "matching.py",
                "cluster_reselection.py",
                "surface_graph.py",
                "saturation_selection.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    summary_path = output / "cell-refinement-diagnostic-v1.json"
    if summary_path.is_file() and not force:
        prior = json.loads(summary_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity["identitySha256"]:
            raise ValueError("cell diagnostic output belongs to another identity")
        return prior
    output.mkdir(parents=True, exist_ok=True)
    context = load_cluster_cell_context(cluster, materialized)
    if not context.grid.contains_cell(cell_xyz):
        raise ValueError(f"diagnostic cell {cell_xyz} lies outside the cluster")
    scores = score_cell_candidates(context, cell_xyz, settings=resolved)
    current = next(value for value in scores if value.selected)
    best_coverage = max(
        scores,
        key=lambda value: (
            value.covered_evidence_mass,
            -value.objective_energy,
        ),
    )
    neighborhood = refine_cell_neighborhood(
        context,
        cell_xyz,
        radius_cells=neighborhood_radius_cells,
        settings=resolved,
    )
    replay_workspace = FrozenReplayWorkspace(context)
    topology_replay = replay_workspace.replay(neighborhood)
    topology_annealing = anneal_topology_safe_refinement(
        context,
        neighborhood,
        replay_workspace=replay_workspace,
    )
    payload: dict[str, Any] = {
        "schema": CELL_REFINEMENT_DIAGNOSTIC_SCHEMA,
        "version": CELL_REFINEMENT_DIAGNOSTIC_VERSION,
        "identity": identity,
        "semantics": {
            "evidenceUtilization": "Acus evidence mass covered by the selected physical stack",
            "faceUtilization": "selected face traces admitted by ordered alignment",
            "topologyUtilization": (
                "compatible face matches retained in the collision-safe "
                "surface graph"
            ),
            "topologyAcceptance": (
                "frozen exterior joins and component connectivity are immutable; "
                "open endpoints and global component count may not increase"
            ),
            "proposalOnly": True,
            "resolvedUnmatchedTracePenalty": (
                resolved.resolved_unmatched_trace_penalty(
                    TraceMatchSettings()
                )
            ),
        },
        "globalEvidenceUtilization": evidence_utilization_summary(context),
        "globalTopologyUtilization": topology_utilization_summary(context),
        "cell": {
            "cellXYZ": list(cell_xyz),
            "inputIndex": context.owner_by_cell[cell_xyz][0],
            "localCellXYZ": list(context.owner_by_cell[cell_xyz][1]),
            "current": current.record(include_faces=True),
            "bestObjective": scores[0].record(include_faces=True),
            "bestCoverage": best_coverage.record(),
            "candidateRanking": [value.record() for value in scores],
            "incidentGaps": incident_gap_summary(context, cell_xyz),
        },
        "neighborhoodProposal": neighborhood,
        "topologyReplay": topology_replay,
        "topologyAnnealing": topology_annealing,
    }
    if component_id is not None:
        payload["componentGapSummary"] = component_gap_summary(context, component_id)
    atomic_json(summary_path, payload)
    return payload
