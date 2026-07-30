from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Mapping

import numpy as np

from .block import (
    BlockBounds,
    DeferredJoin,
    SurfaceJoinSelection,
    select_surface_joins,
    surface_block_from_retained_joins,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch
from .matching import TraceMatch
from .sheet_configuration_solver import (
    SheetConfigurationSolverSettings,
    _factor_value,
    _read_factors,
    _selection_record,
    _state_metrics,
    _unary_values,
)
from .sheet_graph_solver import (
    _read_configuration_selection,
    _read_correspondences,
)
from .sheet_evidence import read_block_sheet_evidence
from .sheet_stitching import (
    SheetMatchingPolicy,
    match_sheet_join_candidate,
)
from .surface_graph import join_key, read_surface_graph, write_surface_graph
from .tables import PatchTable, write_patch_shard
from .topology import GridFace, Int3


SHEET_TOPOLOGY_REFINEMENT_SCHEMA = (
    "pareidolia.cubical-sheet-topology-refinement"
)
SHEET_TOPOLOGY_REFINEMENT_VERSION = 1
SHEET_TOPOLOGY_REFINEMENT_STEM = "sheet-topology-refinement-v1"
SHEET_CONFIGURATION_SELECTION_STEM = "sheet-configuration-selection-v1"

JoinKey = tuple[int, int, int, Int3]
TraceKey = tuple[int, int, Int3]


@dataclass(frozen=True, slots=True)
class SheetTopologyRefinementSettings:
    """Search controls for exact whole-sheet configuration neighborhoods.

    These settings alter only how much of the discrete sheet state is searched.
    They do not change Acus measurements, pair gates, or topology constraints.
    """

    maximum_rounds: int = 2
    maximum_trials_per_round: int = 36
    maximum_seed_moves: int = 48
    alternatives_per_pressure_cell: int = 3
    relaxation_radius: int = 1
    relaxation_sweeps: int = 3
    minimum_objective_gain: float = 1.0e-6
    minimum_join_benefit: float = 0.0
    quarter_turn_penalty: float = 0.75

    def __post_init__(self) -> None:
        positive = (
            self.maximum_rounds,
            self.maximum_trials_per_round,
            self.maximum_seed_moves,
            self.alternatives_per_pressure_cell,
        )
        if any(value <= 0 for value in positive):
            raise ValueError("topology refinement search limits must be positive")
        if self.relaxation_radius < 0 or self.relaxation_sweeps < 0:
            raise ValueError("topology refinement relaxation controls are invalid")
        finite = (
            self.minimum_objective_gain,
            self.minimum_join_benefit,
            self.quarter_turn_penalty,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("topology refinement settings must be finite")
        if self.minimum_objective_gain < 0.0:
            raise ValueError("minimum topology objective gain cannot be negative")
        if self.quarter_turn_penalty < 0.0:
            raise ValueError("quarter-turn penalty cannot be negative")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class ConfigurationSeed:
    cell_index: int
    configuration_index: int
    search_score: float
    local_objective_delta: float
    recoverable_gap_count: int
    topology_pressure_count: int
    source_component_ids: tuple[int, ...]

    def record(self, cells: np.ndarray) -> dict[str, Any]:
        return {
            "cellIndex": self.cell_index,
            "cellXYZ": [int(value) for value in cells[self.cell_index]],
            "configurationIndex": self.configuration_index,
            "searchScore": round(self.search_score, 6),
            "localObjectiveDelta": round(self.local_objective_delta, 6),
            "recoverableGapCount": self.recoverable_gap_count,
            "topologyPressureCount": self.topology_pressure_count,
            "sourceComponentIds": list(self.source_component_ids),
        }


@dataclass(frozen=True, slots=True)
class SheetTopologyEvaluation:
    configuration_index_by_cell: tuple[int, ...]
    joins: tuple[TraceMatch, ...]
    deferred_joins: tuple[DeferredJoin, ...]
    objective: float
    unary_objective: float
    pairwise_objective: float
    total_join_benefit: float
    interior_trace_endpoint_count: int
    component_by_patch: tuple[tuple[int, int], ...]
    component_members: tuple[tuple[int, tuple[int, ...]], ...]
    active_patch_count: int
    active_candidate_count: int
    fixed_join_count: int
    mutable_component_count: int
    proposal: str
    reconstructed_correspondences: int

    @property
    def retained_join_count(self) -> int:
        return len(self.joins)

    @property
    def component_count(self) -> int:
        return len(self.component_members)

    @property
    def largest_component_patch_count(self) -> int:
        return max((len(value) for _, value in self.component_members), default=0)

    @property
    def unmatched_trace_endpoint_count(self) -> int:
        return self.interior_trace_endpoint_count - 2 * len(self.joins)

    def record(self) -> dict[str, Any]:
        endpoints = self.interior_trace_endpoint_count
        sizes = np.asarray(
            [len(value) for _, value in self.component_members], dtype=np.int64
        )
        return {
            "objective": round(self.objective, 6),
            "unaryObjective": round(self.unary_objective, 6),
            "globalPairwiseObjective": round(self.pairwise_objective, 6),
            "totalJoinBenefit": round(self.total_join_benefit, 6),
            "activePatches": self.active_patch_count,
            "activeCandidates": self.active_candidate_count,
            "retainedJoins": len(self.joins),
            "interiorTraceEndpoints": endpoints,
            "unmatchedInteriorTraceEndpoints": self.unmatched_trace_endpoint_count,
            "retainedInteriorTraceFraction": round(
                2 * len(self.joins) / max(endpoints, 1), 6
            ),
            "components": len(self.component_members),
            "largestComponentPatchCount": self.largest_component_patch_count,
            "componentsAtLeastCells": {
                str(threshold): int(np.sum(sizes >= threshold))
                for threshold in (8, 16, 32, 64, 128, 256, 512)
            },
            "fixedExteriorJoins": self.fixed_join_count,
            "mutableSheetComponents": self.mutable_component_count,
            "selectionProposal": self.proposal,
            "newlyReconstructedCorrespondences": (
                self.reconstructed_correspondences
            ),
            "deferredJoinsByReason": dict(
                sorted(Counter(value.reason for value in self.deferred_joins).items())
            ),
        }


def _component_partition(
    patch_ids: Iterable[int],
    joins: Iterable[TraceMatch],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, tuple[int, ...]], ...]]:
    """Return a deterministic graph partition without materializing geometry."""

    values = tuple(sorted(int(value) for value in patch_ids))
    parent = {value: value for value in values}
    size = {value: 1 for value in values}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            next_value = parent[value]
            parent[value] = root
            value = next_value
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size.pop(second_root)

    for value in joins:
        union(value.first_patch_id, value.second_patch_id)
    groups: dict[int, list[int]] = defaultdict(list)
    for patch_id in values:
        groups[find(patch_id)].append(patch_id)
    members = tuple(
        sorted(
            ((min(group), tuple(sorted(group))) for group in groups.values()),
            key=lambda value: value[0],
        )
    )
    component_by_patch = tuple(
        sorted(
            (patch_id, component_id)
            for component_id, group in members
            for patch_id in group
        )
    )
    return component_by_patch, members


def _evaluation_order(value: SheetTopologyEvaluation) -> tuple[Any, ...]:
    """Objective is primary; connectivity is only a deterministic tie-break."""

    return (
        value.objective,
        value.retained_join_count,
        -value.component_count,
        value.largest_component_patch_count,
        tuple(-index for index in value.configuration_index_by_cell),
    )


def choose_improving_topology_evaluation(
    current: SheetTopologyEvaluation,
    proposals: Iterable[SheetTopologyEvaluation],
    *,
    minimum_objective_gain: float = 0.0,
) -> SheetTopologyEvaluation | None:
    """Choose an exact globally scored move without rewarding size by itself."""

    values = tuple(proposals)
    if not values:
        return None
    best = max(values, key=_evaluation_order)
    if best.objective < current.objective + minimum_objective_gain:
        return None
    if best.objective <= current.objective + 1.0e-12:
        return None
    return best


def frozen_exterior_join_keys(
    joins: Iterable[TraceMatch],
    active_patch_ids: frozenset[int] | set[int],
    mutable_patch_ids: frozenset[int] | set[int],
) -> frozenset[JoinKey]:
    """Freeze only joins whose two endpoints remain outside reopened sheets."""

    return frozenset(
        join_key(value)
        for value in joins
        if value.first_patch_id in active_patch_ids
        and value.second_patch_id in active_patch_ids
        and value.first_patch_id not in mutable_patch_ids
        and value.second_patch_id not in mutable_patch_ids
    )


class SheetTopologyWorkspace:
    """Immutable evidence plus lazy geometric replay for many sheet states."""

    def __init__(
        self,
        evidence: Any,
        correspondence_arrays: Mapping[str, np.ndarray],
        factors: Mapping[str, np.ndarray],
        policy: SheetMatchingPolicy,
        solver_settings: SheetConfigurationSolverSettings,
        settings: SheetTopologyRefinementSettings,
        graph_bounds: BlockBounds,
    ) -> None:
        self.evidence = evidence
        self.correspondences = correspondence_arrays
        self.factors = factors
        self.policy = policy
        self.solver_settings = solver_settings
        self.settings = settings
        self.graph_bounds = graph_bounds
        self.cells = np.asarray(evidence.arrays["cellXYZ"], dtype=np.int32)
        self.cell_index = {
            tuple(int(value) for value in cell): index
            for index, cell in enumerate(self.cells)
        }
        self.config_offset = np.asarray(
            evidence.arrays["configurationOffset"], dtype=np.uint64
        )
        self.config_mode_offset = np.asarray(
            evidence.arrays["configurationModeOffset"], dtype=np.uint64
        )
        config_mode_id = np.asarray(
            evidence.arrays["configurationModeId"], dtype=np.uint64
        )
        self.config_modes = tuple(
            tuple(
                int(value)
                for value in config_mode_id[int(low):int(high)]
            )
            for low, high in zip(
                self.config_mode_offset[:-1], self.config_mode_offset[1:]
            )
        )
        self.patch_by_id = {
            value.patch_id: value for value in evidence.mode_patches.to_patches()
        }
        self.mode_cell_index = {
            patch_id: self.cell_index[patch.cell_xyz]
            for patch_id, patch in self.patch_by_id.items()
        }
        configurations_by_mode: dict[int, list[int]] = defaultdict(list)
        for configuration_index, mode_ids in enumerate(self.config_modes):
            for mode_id in mode_ids:
                configurations_by_mode[mode_id].append(configuration_index)
        self.configurations_by_mode = {
            mode_id: tuple(values)
            for mode_id, values in configurations_by_mode.items()
        }
        self.first_mode_id = np.asarray(
            correspondence_arrays["firstModeId"], dtype=np.uint64
        )
        self.second_mode_id = np.asarray(
            correspondence_arrays["secondModeId"], dtype=np.uint64
        )
        self.correspondence_benefit = (
            2.0 * policy.strict_settings.unmatched_negative_log_likelihood
            - np.asarray(
                correspondence_arrays["negativeLogLikelihood"], dtype=np.float64
            )
            - settings.quarter_turn_penalty
            * np.asarray(correspondence_arrays["family"], dtype=np.float64)
        )
        edges_by_mode: dict[int, list[int]] = defaultdict(list)
        self.edge_index_by_key: dict[JoinKey, int] = {}
        for index, (first, second, axis, anchor) in enumerate(
            zip(
                self.first_mode_id,
                self.second_mode_id,
                correspondence_arrays["faceAxis"],
                correspondence_arrays["faceAnchorXYZ"],
            )
        ):
            first_id = int(first)
            second_id = int(second)
            edges_by_mode[first_id].append(index)
            edges_by_mode[second_id].append(index)
            key = (
                first_id,
                second_id,
                int(axis),
                tuple(int(value) for value in anchor),
            )
            if key in self.edge_index_by_key:
                raise ValueError("mode correspondence artifact contains duplicate edges")
            self.edge_index_by_key[key] = index
        self.edges_by_mode = {
            mode_id: tuple(values) for mode_id, values in edges_by_mode.items()
        }
        self.match_cache: dict[int, TraceMatch] = {}
        self.unary = _unary_values(evidence, solver_settings)
        self.factor_neighbors: list[list[tuple[int, int, bool]]] = [
            [] for _ in range(evidence.cell_count)
        ]
        for face_index, (first, second) in enumerate(
            zip(factors["firstCellIndex"], factors["secondCellIndex"])
        ):
            first_cell = int(first)
            second_cell = int(second)
            self.factor_neighbors[first_cell].append(
                (face_index, second_cell, True)
            )
            self.factor_neighbors[second_cell].append(
                (face_index, first_cell, False)
            )

    def active_mode_ids(self, selected: tuple[int, ...]) -> frozenset[int]:
        flat = tuple(
            mode_id
            for configuration_index in selected
            for mode_id in self.config_modes[configuration_index]
        )
        if len(flat) != len(set(flat)):
            raise ValueError("one mode became active in more than one cell")
        return frozenset(flat)

    def active_patches(
        self, selected: tuple[int, ...]
    ) -> tuple[ClippedPatch, ...]:
        return tuple(
            sorted(
                (
                    self.patch_by_id[mode_id]
                    for mode_id in self.active_mode_ids(selected)
                ),
                key=lambda value: (
                    value.cell_xyz[2],
                    value.cell_xyz[1],
                    value.cell_xyz[0],
                    value.estimate.height_from_cell_center,
                    value.patch_id,
                ),
            )
        )

    def _face(self, edge_index: int) -> GridFace:
        return GridFace(
            int(self.correspondences["faceAxis"][edge_index]),
            tuple(
                int(value)
                for value in self.correspondences["faceAnchorXYZ"][edge_index]
            ),
        )

    def _match(self, edge_index: int) -> tuple[TraceMatch, bool]:
        cached = self.match_cache.get(edge_index)
        if cached is not None:
            return cached, False
        first_id = int(self.first_mode_id[edge_index])
        second_id = int(self.second_mode_id[edge_index])
        expected_family = (
            "quarter-turn"
            if bool(int(self.correspondences["family"][edge_index]))
            else "strict"
        )
        matched = match_sheet_join_candidate(
            self.patch_by_id[first_id],
            self.patch_by_id[second_id],
            self._face(edge_index),
            self.policy,
            grid=self.evidence.grid,
        )
        if matched is None or matched[1] != expected_family:
            raise ValueError("persisted mode correspondence changed during refinement")
        match, _ = matched
        stored = float(
            self.correspondences["negativeLogLikelihood"][edge_index]
        )
        if abs(match.negative_log_likelihood - stored) > 2.0e-4:
            raise ValueError("persisted correspondence likelihood changed")
        self.match_cache[edge_index] = match
        return match, True

    def active_candidates(
        self,
        selected: tuple[int, ...],
    ) -> tuple[
        tuple[ClippedPatch, ...],
        tuple[TraceMatch, ...],
        dict[JoinKey, float],
        tuple[int, ...],
        int,
    ]:
        active_ids = self.active_mode_ids(selected)
        active_values = np.fromiter(active_ids, dtype=np.uint64)
        mask = np.isin(self.first_mode_id, active_values) & np.isin(
            self.second_mode_id, active_values
        )
        indices = tuple(
            int(value)
            for value in np.flatnonzero(
                mask
                & (
                    self.correspondence_benefit
                    > self.settings.minimum_join_benefit
                )
            )
        )
        candidates: list[TraceMatch] = []
        priorities: dict[JoinKey, float] = {}
        reconstructed = 0
        for edge_index in indices:
            match, created = self._match(edge_index)
            reconstructed += int(created)
            candidates.append(match)
            priorities[join_key(match)] = float(
                self.correspondence_benefit[edge_index]
            )
        patches = tuple(
            sorted(
                (self.patch_by_id[value] for value in active_ids),
                key=lambda value: (
                    value.cell_xyz[2],
                    value.cell_xyz[1],
                    value.cell_xyz[0],
                    value.estimate.height_from_cell_center,
                    value.patch_id,
                ),
            )
        )
        return patches, tuple(candidates), priorities, indices, reconstructed

    def _local_score(
        self,
        cell_index: int,
        configuration_index: int,
        selected: list[int] | tuple[int, ...],
    ) -> float:
        score = float(self.unary[configuration_index])
        for face_index, neighbor, first_side in self.factor_neighbors[cell_index]:
            first_configuration = (
                configuration_index if first_side else int(selected[neighbor])
            )
            second_configuration = (
                int(selected[neighbor]) if first_side else configuration_index
            )
            score += _factor_value(
                self.factors,
                face_index,
                first_configuration,
                second_configuration,
                self.solver_settings,
            )[0]
        return score

    def relax(
        self,
        selected: tuple[int, ...],
        forced: Mapping[int, int],
    ) -> tuple[int, ...]:
        values = list(selected)
        for cell_index, configuration_index in forced.items():
            low = int(self.config_offset[cell_index])
            high = int(self.config_offset[cell_index + 1])
            if not low <= configuration_index < high:
                raise ValueError("forced configuration belongs to another cell")
            values[cell_index] = configuration_index
        neighborhood = set(forced)
        frontier = set(forced)
        for _ in range(self.settings.relaxation_radius):
            next_frontier: set[int] = set()
            for cell_index in frontier:
                next_frontier.update(
                    neighbor
                    for _, neighbor, _ in self.factor_neighbors[cell_index]
                    if neighbor not in neighborhood
                )
            neighborhood.update(next_frontier)
            frontier = next_frontier
        ordered = sorted(
            neighborhood,
            key=lambda index: (
                int(self.cells[index, 2]),
                int(self.cells[index, 1]),
                int(self.cells[index, 0]),
            ),
        )
        for sweep in range(self.settings.relaxation_sweeps):
            changed = 0
            traversal = ordered if sweep % 2 == 0 else list(reversed(ordered))
            for cell_index in traversal:
                if cell_index in forced:
                    continue
                low = int(self.config_offset[cell_index])
                high = int(self.config_offset[cell_index + 1])
                current = values[cell_index]
                best = max(
                    range(low, high),
                    key=lambda configuration_index: (
                        self._local_score(cell_index, configuration_index, values),
                        -configuration_index,
                    ),
                )
                if best != current:
                    values[cell_index] = best
                    changed += 1
            if changed == 0:
                break
        return tuple(values)

    def _pairwise_objective(
        self,
        patches: tuple[ClippedPatch, ...],
        joins: tuple[TraceMatch, ...],
        priorities: Mapping[JoinKey, float],
    ) -> tuple[float, float, int]:
        endpoint_count: Counter[GridFace] = Counter()
        for patch in patches:
            for trace in patch.traces:
                lower, upper = trace.face.adjacent_cells()
                if self.evidence.grid.contains_cell(
                    lower
                ) and self.evidence.grid.contains_cell(upper):
                    endpoint_count[trace.face] += 1
        benefit_by_face: Counter[GridFace] = Counter()
        matched_by_face: Counter[GridFace] = Counter()
        total_benefit = 0.0
        for value in joins:
            benefit = float(priorities[join_key(value)])
            total_benefit += benefit
            benefit_by_face[value.face] += benefit
            matched_by_face[value.face] += 1
        pairwise = 0.0
        for face, endpoints in endpoint_count.items():
            benefit = float(benefit_by_face[face])
            matched = int(matched_by_face[face])
            unmatched = endpoints - 2 * matched
            if unmatched < 0:
                raise RuntimeError("sheet selection overused a shared-face trace")
            if self.solver_settings.pairwise_normalization == "trace-mean":
                benefit /= max(endpoints, 1)
            pairwise += (
                self.solver_settings.pairwise_scale * benefit
                - self.solver_settings.unmatched_trace_penalty * unmatched
            )
        return pairwise, total_benefit, sum(endpoint_count.values())

    def declared_evaluation(
        self,
        selected: tuple[int, ...],
        joins: tuple[TraceMatch, ...],
        *,
        proposal: str,
    ) -> SheetTopologyEvaluation:
        patches, candidates, priorities, _, reconstructed = self.active_candidates(
            selected
        )
        candidate_keys = {join_key(value) for value in candidates}
        unknown = {join_key(value) for value in joins} - candidate_keys
        if unknown:
            raise ValueError(
                "declared surface graph references absent active correspondences"
            )
        pairwise, benefit, endpoints = self._pairwise_objective(
            patches, joins, priorities
        )
        unary = sum(float(self.unary[value]) for value in selected)
        component_by_patch, members = _component_partition(
            (value.patch_id for value in patches), joins
        )
        return SheetTopologyEvaluation(
            selected,
            joins,
            tuple(),
            unary + pairwise,
            unary,
            pairwise,
            benefit,
            endpoints,
            component_by_patch,
            members,
            len(patches),
            len(candidates),
            len(joins),
            0,
            proposal,
            reconstructed,
        )

    def _mutable_components(
        self,
        current: SheetTopologyEvaluation,
        proposed: tuple[int, ...],
    ) -> tuple[frozenset[int], frozenset[int]]:
        changed = {
            index
            for index, (first, second) in enumerate(
                zip(current.configuration_index_by_cell, proposed)
            )
            if first != second
        }
        component_by_patch = dict(current.component_by_patch)
        members = dict(current.component_members)
        active_current = self.active_mode_ids(current.configuration_index_by_cell)
        mutable_components = {
            component_by_patch[mode_id]
            for cell_index in changed
            for mode_id in self.config_modes[
                current.configuration_index_by_cell[cell_index]
            ]
        }
        proposed_modes = {
            mode_id
            for cell_index in changed
            for mode_id in self.config_modes[proposed[cell_index]]
        }
        for mode_id in proposed_modes:
            for edge_index in self.edges_by_mode.get(mode_id, ()):
                first = int(self.first_mode_id[edge_index])
                second = int(self.second_mode_id[edge_index])
                other = second if first == mode_id else first
                if other in active_current and other in component_by_patch:
                    mutable_components.add(component_by_patch[other])
        mutable_patch_ids = frozenset(
            patch_id
            for component_id in mutable_components
            for patch_id in members[component_id]
        )
        return frozenset(mutable_components), mutable_patch_ids

    def evaluate(
        self,
        proposed: tuple[int, ...],
        current: SheetTopologyEvaluation,
        *,
        proposal: str,
    ) -> SheetTopologyEvaluation:
        patches, candidates, priorities, _, reconstructed = self.active_candidates(
            proposed
        )
        active_ids = {value.patch_id for value in patches}
        mutable_components, mutable_patch_ids = self._mutable_components(
            current, proposed
        )
        current_keys = {join_key(value) for value in current.joins}
        fixed = frozen_exterior_join_keys(
            current.joins,
            active_ids,
            mutable_patch_ids,
        )
        benefit_selection = select_surface_joins(
            patches,
            candidates,
            fixed_join_keys=fixed,
            candidate_priorities=priorities,
        )
        priority_span = max(priorities.values(), default=1.0) + 1.0
        preserving_priorities = {
            key: value + (2.0 * priority_span if key in current_keys else 0.0)
            for key, value in priorities.items()
        }
        preserving_selection = select_surface_joins(
            patches,
            candidates,
            fixed_join_keys=fixed,
            candidate_priorities=preserving_priorities,
        )

        def selection_value(
            selection: SurfaceJoinSelection,
        ) -> tuple[float, float, int, int, tuple[JoinKey, ...]]:
            pairwise, benefit, _ = self._pairwise_objective(
                patches, tuple(selection.joins), priorities
            )
            _, members = _component_partition(
                (value.patch_id for value in patches), selection.joins
            )
            return (
                pairwise,
                float(benefit),
                len(selection.joins),
                -len(members),
                tuple(sorted(join_key(value) for value in selection.joins)),
            )

        selection, selection_name = max(
            (
                (benefit_selection, "join-benefit"),
                (preserving_selection, "preserve-surviving-sheet-joins"),
            ),
            key=lambda value: selection_value(value[0]),
        )
        joins = tuple(selection.joins)
        pairwise, benefit, endpoints = self._pairwise_objective(
            patches, joins, priorities
        )
        unary = sum(float(self.unary[value]) for value in proposed)
        component_by_patch, members = _component_partition(
            (value.patch_id for value in patches), joins
        )
        return SheetTopologyEvaluation(
            proposed,
            joins,
            tuple(selection.deferred_joins),
            unary + pairwise,
            unary,
            pairwise,
            benefit,
            endpoints,
            component_by_patch,
            members,
            len(patches),
            len(candidates),
            len(fixed),
            len(mutable_components),
            f"{proposal} / {selection_name}",
            reconstructed,
        )

    def _edge_trace_key(self, edge_index: int, mode_id: int) -> TraceKey:
        face = self._face(edge_index)
        return mode_id, face.axis, face.anchor_xyz

    def configuration_seeds(
        self,
        current: SheetTopologyEvaluation,
    ) -> tuple[ConfigurationSeed, ...]:
        selected = current.configuration_index_by_cell
        active_ids = self.active_mode_ids(selected)
        patch_by_id = {
            value.patch_id: value for value in self.active_patches(selected)
        }
        joined_traces = {
            (patch_id, value.face.axis, value.face.anchor_xyz)
            for value in current.joins
            for patch_id in (value.first_patch_id, value.second_patch_id)
        }
        component_by_patch = dict(current.component_by_patch)
        pressure: dict[tuple[int, int], dict[str, Any]] = {}
        cell_pressure: Counter[int] = Counter()

        def entry(cell_index: int, configuration_index: int) -> dict[str, Any]:
            return pressure.setdefault(
                (cell_index, configuration_index),
                {
                    "gap": {},
                    "topology": {},
                    "components": set(),
                },
            )

        for patch_id, patch in patch_by_id.items():
            source_component = component_by_patch[patch_id]
            for trace in patch.traces:
                lower, upper = trace.face.adjacent_cells()
                if not (
                    self.evidence.grid.contains_cell(lower)
                    and self.evidence.grid.contains_cell(upper)
                ):
                    continue
                trace_key = (patch_id, trace.face.axis, trace.face.anchor_xyz)
                if trace_key in joined_traces:
                    continue
                source_cell = self.mode_cell_index[patch_id]
                cell_pressure[source_cell] += 1
                for edge_index in self.edges_by_mode.get(patch_id, ()):
                    if self._face(edge_index) != trace.face:
                        continue
                    benefit = float(self.correspondence_benefit[edge_index])
                    if benefit <= self.settings.minimum_join_benefit:
                        continue
                    first = int(self.first_mode_id[edge_index])
                    second = int(self.second_mode_id[edge_index])
                    other = second if first == patch_id else first
                    if other in active_ids:
                        continue
                    target_cell = self.mode_cell_index[other]
                    for configuration_index in self.configurations_by_mode[other]:
                        if configuration_index == selected[target_cell]:
                            continue
                        value = entry(target_cell, configuration_index)
                        prior = value["gap"].get(trace_key, -math.inf)
                        value["gap"][trace_key] = max(prior, benefit)
                        value["components"].add(source_component)
                        cell_pressure[target_cell] += 1

        patches, candidates, priorities, _, _ = self.active_candidates(selected)
        fixed_keys = frozenset(join_key(value) for value in current.joins)
        audit = select_surface_joins(
            patches,
            candidates,
            fixed_join_keys=fixed_keys,
            candidate_priorities=priorities,
        )
        topology_reasons = {
            "component-cell-collision",
            "crossing-topology-cycle",
            "orientation-parity-cycle",
            "face-order-crossing",
        }
        for deferred in audit.deferred_joins:
            if deferred.reason not in topology_reasons:
                continue
            key = join_key(deferred.match)
            benefit = priorities[key]
            for mode_id in (
                deferred.match.first_patch_id,
                deferred.match.second_patch_id,
            ):
                cell_index = self.mode_cell_index[mode_id]
                cell_pressure[cell_index] += 1
                for configuration_index in self.configurations_by_mode[mode_id]:
                    if configuration_index == selected[cell_index]:
                        continue
                    value = entry(cell_index, configuration_index)
                    value["topology"][key] = max(
                        value["topology"].get(key, -math.inf), benefit
                    )
                    value["components"].add(component_by_patch[mode_id])

        maximum_pressure_cells = min(
            max(self.settings.maximum_seed_moves // 2, 1), len(cell_pressure)
        )
        for cell_index, _ in cell_pressure.most_common(maximum_pressure_cells):
            current_configuration = selected[cell_index]
            current_score = self._local_score(
                cell_index, current_configuration, selected
            )
            low = int(self.config_offset[cell_index])
            high = int(self.config_offset[cell_index + 1])
            alternatives = sorted(
                (
                    (
                        self._local_score(cell_index, configuration_index, selected),
                        configuration_index,
                    )
                    for configuration_index in range(low, high)
                    if configuration_index != current_configuration
                    and math.isfinite(float(self.unary[configuration_index]))
                ),
                key=lambda value: (-value[0], value[1]),
            )[: self.settings.alternatives_per_pressure_cell]
            for alternative_score, configuration_index in alternatives:
                value = entry(cell_index, configuration_index)
                value.setdefault("localDelta", alternative_score - current_score)

        seeds: list[ConfigurationSeed] = []
        for (cell_index, configuration_index), value in pressure.items():
            current_score = self._local_score(
                cell_index, selected[cell_index], selected
            )
            alternative_score = self._local_score(
                cell_index, configuration_index, selected
            )
            local_delta = alternative_score - current_score
            gap_benefit = sum(float(item) for item in value["gap"].values())
            topology_benefit = sum(
                float(item) for item in value["topology"].values()
            )
            search_score = local_delta + self.solver_settings.pairwise_scale * (
                gap_benefit + 0.5 * topology_benefit
            )
            seeds.append(
                ConfigurationSeed(
                    cell_index,
                    configuration_index,
                    search_score,
                    local_delta,
                    len(value["gap"]),
                    len(value["topology"]),
                    tuple(sorted(value["components"])),
                )
            )
        seeds.sort(
            key=lambda value: (
                -value.search_score,
                -value.recoverable_gap_count,
                -value.topology_pressure_count,
                value.cell_index,
                value.configuration_index,
            )
        )
        return tuple(seeds[: self.settings.maximum_seed_moves])

    def proposal_states(
        self,
        current: SheetTopologyEvaluation,
        seeds: tuple[ConfigurationSeed, ...],
    ) -> tuple[tuple[float, str, tuple[int, ...], tuple[ConfigurationSeed, ...]], ...]:
        selected = current.configuration_index_by_cell
        values: list[
            tuple[float, str, tuple[int, ...], tuple[ConfigurationSeed, ...]]
        ] = []
        for seed in seeds:
            direct = list(selected)
            direct[seed.cell_index] = seed.configuration_index
            forced = {seed.cell_index: seed.configuration_index}
            relaxed = self.relax(selected, forced)
            values.append(
                (
                    seed.search_score + 1.0e-9,
                    "single-seed relaxed sheet neighborhood",
                    relaxed,
                    (seed,),
                )
            )
            values.append(
                (
                    seed.search_score,
                    "single-seed direct stack exchange",
                    tuple(direct),
                    (seed,),
                )
            )
        pair_candidates: list[tuple[float, ConfigurationSeed, ConfigurationSeed]] = []
        leading = seeds[: min(16, len(seeds))]
        neighbor_sets = [
            {neighbor for _, neighbor, _ in self.factor_neighbors[index]}
            for index in range(self.evidence.cell_count)
        ]
        for first_index, first in enumerate(leading):
            for second in leading[first_index + 1 :]:
                if first.cell_index == second.cell_index:
                    continue
                related = (
                    second.cell_index in neighbor_sets[first.cell_index]
                    or bool(
                        set(first.source_component_ids)
                        & set(second.source_component_ids)
                    )
                )
                if related:
                    pair_candidates.append(
                        (first.search_score + second.search_score, first, second)
                    )
        pair_candidates.sort(
            key=lambda value: (
                -value[0],
                value[1].cell_index,
                value[2].cell_index,
            )
        )
        for score, first, second in pair_candidates[
            : max(self.settings.maximum_trials_per_round // 4, 1)
        ]:
            forced = {
                first.cell_index: first.configuration_index,
                second.cell_index: second.configuration_index,
            }
            values.append(
                (
                    score + 2.0e-9,
                    "paired-seed relaxed sheet neighborhood",
                    self.relax(selected, forced),
                    (first, second),
                )
            )
        unique: dict[tuple[int, ...], tuple[float, str, tuple[ConfigurationSeed, ...]]] = {}
        for score, label, state, state_seeds in values:
            if state == selected:
                continue
            prior = unique.get(state)
            candidate = (score, label, state_seeds)
            if prior is None or (score, label) > (prior[0], prior[1]):
                unique[state] = candidate
        ordered = sorted(
            (
                (score, label, state, state_seeds)
                for state, (score, label, state_seeds) in unique.items()
            ),
            key=lambda value: (-value[0], value[1], value[2]),
        )
        return tuple(ordered[: self.settings.maximum_trials_per_round])


def _selection_settings_from_manifest(
    manifest: Mapping[str, Any],
) -> SheetConfigurationSolverSettings:
    settings = manifest.get("identity", {}).get("settings")
    if not isinstance(settings, dict):
        raise ValueError("configuration selection does not record solver settings")
    return SheetConfigurationSolverSettings(**settings)


def _evaluation_delta(
    after: SheetTopologyEvaluation,
    before: SheetTopologyEvaluation,
) -> dict[str, Any]:
    return {
        "objective": round(after.objective - before.objective, 6),
        "unaryObjective": round(
            after.unary_objective - before.unary_objective, 6
        ),
        "globalPairwiseObjective": round(
            after.pairwise_objective - before.pairwise_objective, 6
        ),
        "totalJoinBenefit": round(
            after.total_join_benefit - before.total_join_benefit, 6
        ),
        "activePatches": after.active_patch_count - before.active_patch_count,
        "retainedJoins": after.retained_join_count - before.retained_join_count,
        "unmatchedInteriorTraceEndpoints": (
            after.unmatched_trace_endpoint_count
            - before.unmatched_trace_endpoint_count
        ),
        "components": after.component_count - before.component_count,
        "largestComponentPatchCount": (
            after.largest_component_patch_count
            - before.largest_component_patch_count
        ),
    }


def _write_selected_state(
    output: Path,
    workspace: SheetTopologyWorkspace,
    evaluation: SheetTopologyEvaluation,
    identity_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    selected = np.asarray(
        evaluation.configuration_index_by_cell, dtype=np.uint32
    )
    evidence = workspace.evidence
    config_id = np.asarray(evidence.arrays["configurationId"], dtype=np.uint64)
    input_index = np.asarray(
        evidence.arrays["configurationInputIndex"], dtype=np.uint16
    )
    source_index = np.asarray(
        evidence.arrays["configurationSourceIndex"], dtype=np.uint32
    )
    selection_path = output / f"{SHEET_CONFIGURATION_SELECTION_STEM}.npz"
    temporary = selection_path.with_suffix(selection_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=workspace.cells,
            configurationIndex=selected,
            configurationId=config_id[selected],
            inputIndex=input_index[selected],
            sourceConfigurationIndex=source_index[selected],
            selectedModeCount=np.asarray(
                [len(workspace.config_modes[value]) for value in selected],
                dtype=np.uint16,
            ),
        )
    temporary.replace(selection_path)
    patches = workspace.active_patches(evaluation.configuration_index_by_cell)
    block = surface_block_from_retained_joins(
        evidence.grid,
        workspace.graph_bounds,
        patches,
        evaluation.joins,
    )
    config_log_weight = np.asarray(
        evidence.arrays["configurationLogWeight"], dtype=np.float32
    )
    config_family = np.asarray(
        evidence.arrays["configurationNormalHypothesis"], dtype=np.int16
    )
    configuration_id: dict[int, int] = {}
    configuration_log_weight: dict[int, float] = {}
    local_order: dict[int, int] = {}
    normal_family: dict[int, int] = {}
    for cell_index, configuration_index in enumerate(selected):
        index = int(configuration_index)
        for order, mode_id in enumerate(workspace.config_modes[index]):
            configuration_id[mode_id] = index
            configuration_log_weight[mode_id] = float(config_log_weight[index])
            local_order[mode_id] = order
            normal_family[mode_id] = int(config_family[index])
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        PatchTable.from_patches(
            evidence.grid,
            block.patches,
            configuration_id=configuration_id,
            configuration_log_weight=configuration_log_weight,
            local_order=local_order,
            normal_family=normal_family,
        ),
        settings={
            "semantics": (
                "selected immutable Acus modes after topology-aware "
                "configuration refinement"
            )
        },
        provenance={"sheetTopologyRefinementIdentitySha256": identity_sha256},
        compressed=True,
    )
    graph_manifest = write_surface_graph(
        output,
        block,
        semantics=(
            "whole-block stack configurations accepted only against an exact "
            "topology-safe retained sheet graph"
        ),
        provenance={"sheetTopologyRefinementIdentitySha256": identity_sha256},
    )
    selection = {
        "path": selection_path.name,
        "bytes": selection_path.stat().st_size,
        "sha256": sha256_file(selection_path),
    }
    return selection, patch_manifest, graph_manifest


def run_sheet_topology_refinement(
    evidence_root: str | Path,
    correspondence_root: str | Path,
    factor_root: str | Path,
    configuration_root: str | Path,
    graph_root: str | Path,
    cluster_root: str | Path,
    output_root: str | Path,
    *,
    settings: SheetTopologyRefinementSettings | None = None,
    force: bool = False,
    progress: Callable[[int, int, int, str], None] | None = None,
) -> dict[str, Any]:
    """Refine physical stack choices against the exact retained sheet graph."""

    started = time.monotonic()
    resolved = settings or SheetTopologyRefinementSettings()
    evidence_path = Path(evidence_root).resolve()
    correspondence_path = Path(correspondence_root).resolve()
    factor_path = Path(factor_root).resolve()
    configuration_path = Path(configuration_root).resolve()
    graph_path = Path(graph_root).resolve()
    cluster_path = Path(cluster_root).resolve()
    output = Path(output_root).resolve()
    evidence = read_block_sheet_evidence(evidence_path, verify=True)
    selected_array, configuration_manifest = _read_configuration_selection(
        configuration_path, evidence.cell_count
    )
    selected = tuple(int(value) for value in selected_array)
    correspondence_arrays, correspondence_manifest = _read_correspondences(
        correspondence_path
    )
    factors, factor_manifest = _read_factors(factor_path)
    policy = SheetMatchingPolicy.from_cluster_root(cluster_path)
    solver_settings = _selection_settings_from_manifest(configuration_manifest)
    graph = read_surface_graph(graph_path, verify=True)
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_TOPOLOGY_REFINEMENT_SCHEMA,
        "version": SHEET_TOPOLOGY_REFINEMENT_VERSION,
        "evidenceManifestSha256": sha256_file(
            evidence_path / "sheet-evidence-v1.json"
        ),
        "correspondenceIdentitySha256": correspondence_manifest["identity"][
            "identitySha256"
        ],
        "correspondenceDataSha256": correspondence_manifest["data"]["sha256"],
        "factorIdentitySha256": factor_manifest["identity"]["identitySha256"],
        "factorDataSha256": factor_manifest["data"]["sha256"],
        "configurationIdentitySha256": configuration_manifest["identity"][
            "identitySha256"
        ],
        "configurationDataSha256": configuration_manifest["data"]["sha256"],
        "inputGraphManifestSha256": sha256_file(
            graph_path / "surface-graph-v1.json"
        ),
        "inputGraphDataSha256": sha256_file(graph_path / "surface-graph-v1.npz"),
        "clusterManifestSha256": sha256_file(
            cluster_path / "cluster-reselection-v1.json"
        ),
        "solverSettings": solver_settings.record(),
        "searchSettings": resolved.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_topology_refinement.py",
                "sheet_graph_solver.py",
                "sheet_configuration_solver.py",
                "sheet_stitching.py",
                "block.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_TOPOLOGY_REFINEMENT_STEM}.json"
    selection_manifest_path = output / f"{SHEET_CONFIGURATION_SELECTION_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet-topology output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_TOPOLOGY_REFINEMENT_SCHEMA,
            "version": SHEET_TOPOLOGY_REFINEMENT_VERSION,
            "state": "loading-and-replaying",
            "identity": identity,
        },
    )
    workspace = SheetTopologyWorkspace(
        evidence,
        correspondence_arrays,
        factors,
        policy,
        solver_settings,
        resolved,
        graph.bounds,
    )
    expected_patch_ids = workspace.active_mode_ids(selected)
    graph_patch_ids = frozenset(value.patch_id for value in graph.patches)
    if graph_patch_ids != expected_patch_ids:
        raise ValueError(
            "input graph and configuration selection activate different modes"
        )
    initial = workspace.declared_evaluation(
        selected, tuple(graph.joins), proposal="declared input graph"
    )
    normalized = workspace.evaluate(selected, initial, proposal="graph completion")
    normalization_accepted = normalized.objective > initial.objective + 1.0e-12
    current = normalized
    if not normalization_accepted:
        current = initial
    trial_records: list[dict[str, Any]] = []
    round_records: list[dict[str, Any]] = []
    accepted_rounds = 0
    for round_index in range(resolved.maximum_rounds):
        seeds = workspace.configuration_seeds(current)
        proposal_states = workspace.proposal_states(current, seeds)
        evaluations: list[SheetTopologyEvaluation] = []
        round_trials: list[dict[str, Any]] = []
        total = len(proposal_states)
        for trial_index, (_, label, state, state_seeds) in enumerate(
            proposal_states, start=1
        ):
            evaluation = workspace.evaluate(state, current, proposal=label)
            evaluations.append(evaluation)
            record = {
                "round": round_index + 1,
                "trial": trial_index,
                "label": label,
                "changedConfigurations": sum(
                    first != second
                    for first, second in zip(
                        state, current.configuration_index_by_cell
                    )
                ),
                "seedMoves": [seed.record(workspace.cells) for seed in state_seeds],
                "evaluation": evaluation.record(),
                "deltaFromRoundStart": _evaluation_delta(evaluation, current),
            }
            round_trials.append(record)
            trial_records.append(record)
            if progress is not None:
                progress(round_index + 1, trial_index, total, label)
        accepted = choose_improving_topology_evaluation(
            current,
            evaluations,
            minimum_objective_gain=resolved.minimum_objective_gain,
        )
        round_record = {
            "round": round_index + 1,
            "seedCount": len(seeds),
            "trialCount": len(round_trials),
            "accepted": accepted is not None,
            "bestObjectiveGain": round(
                max(
                    (value.objective - current.objective for value in evaluations),
                    default=0.0,
                ),
                6,
            ),
        }
        if accepted is None:
            round_records.append(round_record)
            break
        round_record["acceptedState"] = accepted.record()
        round_record["acceptedDelta"] = _evaluation_delta(accepted, current)
        round_records.append(round_record)
        current = accepted
        accepted_rounds += 1

    baseline_local = _selection_record(
        evidence,
        factors,
        selected,
        workspace.unary,
        solver_settings,
        sweeps=0,
        changed_last_sweep=0,
        initialization="topology-refinement-input",
    )
    selected_local = _selection_record(
        evidence,
        factors,
        current.configuration_index_by_cell,
        workspace.unary,
        solver_settings,
        sweeps=accepted_rounds,
        changed_last_sweep=0,
        initialization="topology-aware-sheet-neighborhoods",
    )
    selection_data, patch_manifest, graph_manifest = _write_selected_state(
        output, workspace, current, identity_sha256
    )
    baseline_local_metrics = _state_metrics(
        evidence, baseline_local, selected
    )
    selected_local_metrics = _state_metrics(
        evidence, selected_local, selected
    )
    changed_configurations = sum(
        first != second
        for first, second in zip(
            selected, current.configuration_index_by_cell
        )
    )
    summary = {
        "schema": "pareidolia.cubical-sheet-topology-refinement-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "method": {
            "AcusEvidenceMutable": False,
            "configurationMove": "complete physical stack hyperedge",
            "neighborhood": (
                "changed stacks plus every touched current sheet component"
            ),
            "frozenExterior": "all retained joins outside reopened sheets",
            "acceptanceObjective": (
                "Acus/stack unary plus globally retained correspondence benefit"
            ),
            "hardConstraints": [
                "one join per patch trace",
                "order-preserving face correspondences",
                "one patch per sheet component per cell",
                "crossing-feature consistency",
                "orientable polygon parity",
            ],
            "componentSizeReward": False,
        },
        "normalization": {
            "accepted": normalization_accepted,
            "delta": _evaluation_delta(normalized, initial),
        },
        "baseline": {
            "configuration": baseline_local_metrics,
            "globalTopology": initial.record(),
        },
        "selected": {
            **selected_local_metrics,
            "globalTopology": current.record(),
        },
        "delta": {
            **_evaluation_delta(current, initial),
            "changedConfigurations": changed_configurations,
            "coveredEvidenceFraction": round(
                float(selected_local_metrics["coveredEvidenceFraction"])
                - float(baseline_local_metrics["coveredEvidenceFraction"]),
                6,
            ),
            "locallyMatchedFaceTraces": (
                selected_local.matched_trace_count
                - baseline_local.matched_trace_count
            ),
        },
        "localToGlobalTopologyTax": {
            "locallyMatchedFaceTraces": selected_local.matched_trace_count,
            "globallyRetainedJoins": current.retained_join_count,
            "rejectedLocalMatches": (
                selected_local.matched_trace_count - current.retained_join_count
            ),
            "survivalFraction": round(
                current.retained_join_count
                / max(selected_local.matched_trace_count, 1),
                6,
            ),
        },
        "search": {
            "roundsAttempted": len(round_records),
            "roundsAccepted": accepted_rounds,
            "trials": len(trial_records),
            "matchCacheEntries": len(workspace.match_cache),
            "rounds": round_records,
            "trialRecords": trial_records,
        },
        "artifacts": {
            "configurationSelection": selection_data,
            "selectedPatches": {
                "manifest": "selected-patches-v1.json",
                "manifestSha256": sha256_file(output / "selected-patches-v1.json"),
                "data": patch_manifest["data"],
            },
            "surfaceGraph": {
                "manifest": "surface-graph-v1.json",
                "manifestSha256": sha256_file(output / "surface-graph-v1.json"),
                "data": graph_manifest["data"],
            },
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(summary_path, summary)
    selection_identity = {
        **identity,
        "source": "topology-aware whole-sheet configuration refinement",
        "settings": solver_settings.record(),
    }
    selection_identity["identitySha256"] = canonical_json_hash(selection_identity)
    atomic_json(
        selection_manifest_path,
        {
            "schema": SHEET_TOPOLOGY_REFINEMENT_SCHEMA,
            "version": SHEET_TOPOLOGY_REFINEMENT_VERSION,
            "state": "complete",
            "identity": selection_identity,
            "summary": summary_path.name,
            "data": selection_data,
        },
    )
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_TOPOLOGY_REFINEMENT_SCHEMA,
            "version": SHEET_TOPOLOGY_REFINEMENT_VERSION,
            "state": "complete",
            "identity": identity,
            "summary": summary_path.name,
            "data": selection_data,
            "elapsedSeconds": summary["elapsedSeconds"],
        },
    )
    return summary
