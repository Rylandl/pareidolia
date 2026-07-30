from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import (
    SurfaceBlock,
    SurfaceJoinSelection,
    select_surface_joins,
    surface_block_from_retained_joins,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch
from .matching import (
    TraceMatch,
    TraceMatchSettings,
    face_patch_ranks,
    match_face_traces,
)
from .surface_graph import (
    component_statistics,
    join_key,
    read_surface_graph,
    write_surface_graph,
)
from .topology import GridFace, Int3, cell_face


SHEET_JOIN_CATALOG_SCHEMA = "pareidolia.cubical-sheet-join-catalog"
SHEET_JOIN_CATALOG_VERSION = 1
SHEET_JOIN_CATALOG_STEM = "sheet-join-catalog-v1"
SHEET_RESTITCH_SCHEMA = "pareidolia.cubical-sheet-restitch"
SHEET_RESTITCH_VERSION = 1

JoinKey = tuple[int, int, int, Int3]
TraceKey = tuple[int, GridFace]


@dataclass(frozen=True, slots=True)
class SheetMatchingPolicy:
    strict_settings: TraceMatchSettings
    quarter_turn_enabled: bool
    maximum_quarter_turn_normal_degrees: float
    maximum_quarter_turn_fiber_degrees: float

    @classmethod
    def from_cluster_root(cls, cluster_root: str | Path) -> SheetMatchingPolicy:
        root = Path(cluster_root).resolve()
        manifest = json.loads((root / "cluster-reselection-v1.json").read_text())
        policy = manifest["identity"]["seamMatchingPolicy"]
        parallel = policy["parallelMatching"]
        quarter = policy["quarterTurnAdmission"]
        return cls(
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
        )

    def record(self) -> dict[str, Any]:
        return {
            "strictSettings": asdict(self.strict_settings),
            "quarterTurnEnabled": self.quarter_turn_enabled,
            "maximumQuarterTurnNormalDegrees": (
                self.maximum_quarter_turn_normal_degrees
            ),
            "maximumQuarterTurnFiberDegrees": (
                self.maximum_quarter_turn_fiber_degrees
            ),
        }


@dataclass(frozen=True, slots=True)
class SheetStitchingSettings:
    minimum_join_benefit: float = 0.0
    quarter_turn_penalty: float = 0.75
    restart_count: int = 12
    priority_jitter_fraction: float = 0.35
    exchange_round_count: int = 2
    exchange_trials_per_round: int = 24

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_join_benefit):
            raise ValueError("minimum join benefit must be finite")
        if not math.isfinite(self.quarter_turn_penalty) or self.quarter_turn_penalty < 0:
            raise ValueError("quarter-turn penalty must be finite and nonnegative")
        if self.restart_count < 2:
            raise ValueError("sheet stitching requires at least two global proposals")
        if (
            not math.isfinite(self.priority_jitter_fraction)
            or not 0.0 <= self.priority_jitter_fraction <= 1.0
        ):
            raise ValueError("priority jitter fraction must lie in [0, 1]")
        if self.exchange_round_count < 0:
            raise ValueError("exchange round count must be nonnegative")
        if self.exchange_trials_per_round < 0:
            raise ValueError("exchange trial count must be nonnegative")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class SheetJoinCandidate:
    match: TraceMatch
    family: str
    first_rank: int
    second_rank: int
    benefit: float
    currently_retained: bool

    @property
    def key(self) -> JoinKey:
        return join_key(self.match)

    @property
    def trace_keys(self) -> tuple[TraceKey, TraceKey]:
        return (
            (self.match.first_patch_id, self.match.face),
            (self.match.second_patch_id, self.match.face),
        )


@dataclass(frozen=True, slots=True)
class SheetJoinCatalog:
    candidates: tuple[SheetJoinCandidate, ...]
    interior_face_count: int
    unstable_face_count: int

    def by_key(self) -> dict[JoinKey, SheetJoinCandidate]:
        return {value.key: value for value in self.candidates}

    def statistics(self) -> dict[str, Any]:
        degrees: Counter[TraceKey] = Counter(
            trace
            for candidate in self.candidates
            for trace in candidate.trace_keys
        )
        degree_values = np.asarray(tuple(degrees.values()), dtype=np.int64)
        return {
            "interiorFaces": self.interior_face_count,
            "unstableFaces": self.unstable_face_count,
            "candidates": len(self.candidates),
            "positiveBenefitCandidates": sum(
                value.benefit > 0.0 for value in self.candidates
            ),
            "strictCandidates": sum(
                value.family == "strict" for value in self.candidates
            ),
            "quarterTurnCandidates": sum(
                value.family == "quarter-turn" for value in self.candidates
            ),
            "currentlyRetainedCandidates": sum(
                value.currently_retained for value in self.candidates
            ),
            "traceResources": len(degrees),
            "candidateDegreeQuantiles": {
                name: round(float(value), 4)
                for name, value in zip(
                    ("minimum", "median", "p90", "maximum"),
                    np.percentile(degree_values, (0, 50, 90, 100))
                    if len(degree_values)
                    else (0.0, 0.0, 0.0, 0.0),
                )
            },
        }


@dataclass(frozen=True, slots=True)
class SheetRestitchResult:
    block: SurfaceBlock
    summary: dict[str, Any]
    proposal_records: tuple[dict[str, Any], ...]


def match_sheet_join_candidate(
    first: ClippedPatch,
    second: ClippedPatch,
    face: GridFace,
    policy: SheetMatchingPolicy,
    *,
    grid: Any,
) -> tuple[TraceMatch, str] | None:
    first_trace = first.trace_on(face)
    second_trace = second.trace_on(face)
    if first_trace is None or second_trace is None:
        return None
    strict = match_face_traces(
        first_trace,
        first.estimate,
        second_trace,
        second.estimate,
        policy.strict_settings,
        grid=grid,
    )
    if strict.accepted:
        return strict, "strict"
    if not policy.quarter_turn_enabled:
        return None
    quarter = match_face_traces(
        first_trace,
        first.estimate,
        second_trace,
        second.estimate,
        TraceMatchSettings(orthogonal_fiber_equivalence=True),
        grid=grid,
    )
    if not (
        quarter.accepted
        and quarter.fiber_quarter_turn is True
        and math.degrees(quarter.normal_angle_radians)
        <= policy.maximum_quarter_turn_normal_degrees
        and quarter.fiber_angle_radians is not None
        and math.degrees(quarter.fiber_angle_radians)
        <= policy.maximum_quarter_turn_fiber_degrees
    ):
        return None
    return quarter, "quarter-turn"


def enumerate_sheet_join_catalog(
    block: SurfaceBlock,
    policy: SheetMatchingPolicy,
    *,
    settings: SheetStitchingSettings | None = None,
) -> SheetJoinCatalog:
    """Enumerate every pair-gated correspondence between selected patches.

    This deliberately retains alternatives discarded by the per-face dynamic
    program. Face order is recorded as a constraint rather than used to select
    one alignment prematurely.
    """

    resolved = settings or SheetStitchingSettings()
    by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in block.patches:
        by_cell[patch.cell_xyz].append(patch)
    for values in by_cell.values():
        values.sort(
            key=lambda value: (
                value.estimate.height_from_cell_center,
                value.patch_id,
            )
        )
    retained = {join_key(value) for value in block.joins}
    candidates: dict[JoinKey, SheetJoinCandidate] = {}
    interior_faces = 0
    unstable_faces = 0
    for lower in sorted(by_cell, key=lambda value: (value[2], value[1], value[0])):
        for axis in range(3):
            neighbor_values = list(lower)
            neighbor_values[axis] += 1
            upper = tuple(neighbor_values)
            if not block.bounds.contains_cell(upper) or upper not in by_cell:
                continue
            face = cell_face(lower, axis, 1)
            first_values = tuple(by_cell[lower])
            second_values = tuple(by_cell[upper])
            if not any(value.trace_on(face) is not None for value in first_values):
                continue
            if not any(value.trace_on(face) is not None for value in second_values):
                continue
            interior_faces += 1
            try:
                first_ranks, second_ranks, _ = face_patch_ranks(
                    first_values,
                    second_values,
                    face,
                )
            except ValueError:
                unstable_faces += 1
                continue
            for first in first_values:
                if first.patch_id not in first_ranks:
                    continue
                for second in second_values:
                    if second.patch_id not in second_ranks:
                        continue
                    matched = match_sheet_join_candidate(
                        first,
                        second,
                        face,
                        policy,
                        grid=block.grid,
                    )
                    if matched is None:
                        continue
                    match, family = matched
                    benefit = (
                        2.0
                        * policy.strict_settings.unmatched_negative_log_likelihood
                        - match.negative_log_likelihood
                        - (
                            resolved.quarter_turn_penalty
                            if family == "quarter-turn"
                            else 0.0
                        )
                    )
                    key = join_key(match)
                    candidates[key] = SheetJoinCandidate(
                        match,
                        family,
                        first_ranks[first.patch_id],
                        second_ranks[second.patch_id],
                        float(benefit),
                        key in retained,
                    )
    missing = retained - set(candidates)
    if missing:
        raise ValueError(
            "current retained graph contains joins absent from the complete "
            f"policy catalog: {sorted(missing)[:2]}"
        )
    return SheetJoinCatalog(
        tuple(sorted(candidates.values(), key=lambda value: value.key)),
        interior_faces,
        unstable_faces,
    )


def _stable_jitter(key: JoinKey, restart: int) -> float:
    digest = hashlib.blake2b(
        repr((restart, key)).encode("utf-8"), digest_size=8
    ).digest()
    integer = int.from_bytes(digest, "little")
    return 2.0 * (integer / float((1 << 64) - 1)) - 1.0


def _face_optimal_candidate_keys(
    candidates: tuple[SheetJoinCandidate, ...],
) -> frozenset[JoinKey]:
    """Solve every face's weighted order-preserving trace matching exactly."""

    by_face: dict[GridFace, list[SheetJoinCandidate]] = defaultdict(list)
    for value in candidates:
        by_face[value.match.face].append(value)
    selected: set[JoinKey] = set()
    for values in by_face.values():
        first_count = 1 + max(value.first_rank for value in values)
        second_count = 1 + max(value.second_rank for value in values)
        candidate_at = {
            (value.first_rank, value.second_rank): value for value in values
        }
        # Each state carries (benefit, retained count, lexicographic keys).
        scores = np.zeros((first_count + 1, second_count + 1), dtype=np.float64)
        counts = np.zeros((first_count + 1, second_count + 1), dtype=np.int32)
        paths: list[list[tuple[JoinKey, ...]]] = [
            [tuple() for _ in range(second_count + 1)]
            for _ in range(first_count + 1)
        ]

        def better(
            first: tuple[float, int, tuple[JoinKey, ...]],
            second: tuple[float, int, tuple[JoinKey, ...]],
        ) -> tuple[float, int, tuple[JoinKey, ...]]:
            if first[0] > second[0] + 1.0e-12:
                return first
            if second[0] > first[0] + 1.0e-12:
                return second
            if first[1] != second[1]:
                return first if first[1] > second[1] else second
            return first if first[2] <= second[2] else second

        for first_index in range(1, first_count + 1):
            for second_index in range(1, second_count + 1):
                options = (
                    (
                        float(scores[first_index - 1, second_index]),
                        int(counts[first_index - 1, second_index]),
                        paths[first_index - 1][second_index],
                    ),
                    (
                        float(scores[first_index, second_index - 1]),
                        int(counts[first_index, second_index - 1]),
                        paths[first_index][second_index - 1],
                    ),
                )
                chosen = better(options[0], options[1])
                candidate = candidate_at.get((first_index - 1, second_index - 1))
                if candidate is not None:
                    diagonal = (
                        float(scores[first_index - 1, second_index - 1])
                        + candidate.benefit,
                        int(counts[first_index - 1, second_index - 1]) + 1,
                        (*paths[first_index - 1][second_index - 1], candidate.key),
                    )
                    chosen = better(chosen, diagonal)
                scores[first_index, second_index] = chosen[0]
                counts[first_index, second_index] = chosen[1]
                paths[first_index][second_index] = chosen[2]
        selected.update(paths[first_count][second_count])
    return frozenset(selected)


def _block_record(
    block: SurfaceBlock,
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    unmatched_cost: float,
) -> dict[str, Any]:
    benefit = sum(candidate_by_key[join_key(value)].benefit for value in block.joins)
    interior_endpoints = 2 * len(block.joins) + len(block.unresolved_interior_traces)
    objective_cost = unmatched_cost * interior_endpoints - benefit
    components = component_statistics(block, maximum_records=8)
    return {
        "patches": len(block.patches),
        "retainedJoins": len(block.joins),
        "components": len(block.components),
        "largestComponentPatchCount": max(
            (len(value.patch_ids) for value in block.components), default=0
        ),
        "unresolvedInteriorTraceEndpoints": len(
            block.unresolved_interior_traces
        ),
        "retainedInteriorTraceFraction": round(
            2 * len(block.joins) / max(interior_endpoints, 1), 6
        ),
        "totalJoinBenefit": round(float(benefit), 6),
        "objectiveCostWithoutConstant": round(float(objective_cost), 6),
        "deferredJoinsByReason": dict(
            sorted(Counter(value.reason for value in block.deferred_joins).items())
        ),
        "componentSizeDistribution": {
            "largestOccupiedCellCount": components["largestOccupiedCellCount"],
            "medianOccupiedCellCount": components["medianOccupiedCellCount"],
            "componentsAtLeastCells": components["componentsAtLeastCells"],
        },
    }


def _component_partition(
    patches: tuple[ClippedPatch, ...],
    joins: tuple[TraceMatch, ...],
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    parent = {value.patch_id: value.patch_id for value in patches}
    size = {value.patch_id: 1 for value in patches}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            following = parent[value]
            parent[value] = root
            value = following
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
    unordered: dict[int, list[int]] = defaultdict(list)
    for patch in patches:
        unordered[find(patch.patch_id)].append(patch.patch_id)
    members = {
        min(values): tuple(sorted(values))
        for values in unordered.values()
    }
    component_by_patch = {
        patch_id: component_id
        for component_id, values in members.items()
        for patch_id in values
    }
    return component_by_patch, members


def _selection_record(
    patches: tuple[ClippedPatch, ...],
    selection: SurfaceJoinSelection,
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    unmatched_cost: float,
    interior_endpoint_count: int,
) -> dict[str, Any]:
    benefit = sum(
        candidate_by_key[join_key(value)].benefit for value in selection.joins
    )
    _, members = _component_partition(patches, selection.joins)
    sizes = np.asarray([len(value) for value in members.values()], dtype=np.int64)
    unresolved = interior_endpoint_count - 2 * len(selection.joins)
    if unresolved < 0:
        raise RuntimeError("sheet selection retained more traces than physically exist")
    thresholds = (8, 16, 32, 64, 128, 256, 512)
    return {
        "patches": len(patches),
        "retainedJoins": len(selection.joins),
        "components": len(members),
        "largestComponentPatchCount": int(np.max(sizes)) if len(sizes) else 0,
        "unresolvedInteriorTraceEndpoints": unresolved,
        "retainedInteriorTraceFraction": round(
            2 * len(selection.joins) / max(interior_endpoint_count, 1), 6
        ),
        "totalJoinBenefit": round(float(benefit), 6),
        "objectiveCostWithoutConstant": round(
            float(unmatched_cost * interior_endpoint_count - benefit), 6
        ),
        "deferredJoinsByReason": dict(
            sorted(
                Counter(value.reason for value in selection.deferred_joins).items()
            )
        ),
        "componentSizeDistribution": {
            "largestOccupiedCellCount": int(np.max(sizes)) if len(sizes) else 0,
            "medianOccupiedCellCount": (
                round(float(np.median(sizes)), 4) if len(sizes) else 0.0
            ),
            "componentsAtLeastCells": {
                str(value): int(np.sum(sizes >= value)) for value in thresholds
            },
        },
    }


def _selection_quality(
    selection: SurfaceJoinSelection,
    patches: tuple[ClippedPatch, ...],
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
) -> tuple[float, int, int, int]:
    benefit = sum(
        candidate_by_key[join_key(value)].benefit for value in selection.joins
    )
    _, members = _component_partition(patches, selection.joins)
    largest = max((len(value) for value in members.values()), default=0)
    return benefit, len(selection.joins), -len(members), largest


def _exchange_sheet_neighborhoods(
    block: SurfaceBlock,
    eligible: tuple[SheetJoinCandidate, ...],
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    initial: SurfaceJoinSelection,
    settings: SheetStitchingSettings,
) -> tuple[SurfaceJoinSelection, tuple[dict[str, Any], ...], dict[str, Any]]:
    """Improve a graph by reopening complete current sheet components.

    A trial never patches one dangling endpoint in isolation. It removes every
    join inside the one or two current sheet components touched by a focal
    alternative, then reconstructs that entire induced candidate graph while
    the rest of the block remains fixed. This permits wrong transitive joins to
    disappear and displaced traces to rematch as a chain.
    """

    current = initial
    records: list[dict[str, Any]] = []
    accepted_count = 0
    attempted_count = 0
    skipped_duplicate_neighborhoods = 0
    for round_index in range(settings.exchange_round_count):
        selected_keys = {join_key(value) for value in current.joins}
        component_by_patch, members = _component_partition(
            block.patches, current.joins
        )
        selected_by_trace: dict[TraceKey, SheetJoinCandidate] = {}
        for value in current.joins:
            candidate = candidate_by_key[join_key(value)]
            for trace in candidate.trace_keys:
                selected_by_trace[trace] = candidate
        opportunities: list[
            tuple[float, int, int, float, int, JoinKey, SheetJoinCandidate]
        ] = []
        for candidate in eligible:
            if candidate.key in selected_keys:
                continue
            displaced = {
                selected_by_trace[trace].key: selected_by_trace[trace]
                for trace in candidate.trace_keys
                if trace in selected_by_trace
            }
            displaced_benefit = sum(
                value.benefit for value in displaced.values()
            )
            first_component = component_by_patch[candidate.match.first_patch_id]
            second_component = component_by_patch[candidate.match.second_patch_id]
            bridge = int(first_component != second_component)
            active_size = len(members[first_component]) + (
                len(members[second_component]) if bridge else 0
            )
            opportunities.append(
                (
                    candidate.benefit - displaced_benefit,
                    len(displaced),
                    bridge,
                    candidate.benefit,
                    active_size,
                    candidate.key,
                    candidate,
                )
            )
        opportunities.sort(
            key=lambda value: (
                -value[0],
                -value[2],
                -value[3],
                -value[4],
                value[5],
            )
        )
        open_opportunities = [value for value in opportunities if value[1] == 0]
        occupied_opportunities = [value for value in opportunities if value[1] > 0]
        scheduled: list[
            tuple[float, int, int, float, int, JoinKey, SheetJoinCandidate]
        ] = []
        open_index = 0
        occupied_index = 0
        while (
            len(scheduled) < settings.exchange_trials_per_round
            and (
                open_index < len(open_opportunities)
                or occupied_index < len(occupied_opportunities)
            )
        ):
            if open_index < len(open_opportunities):
                scheduled.append(open_opportunities[open_index])
                open_index += 1
            if (
                len(scheduled) < settings.exchange_trials_per_round
                and occupied_index < len(occupied_opportunities)
            ):
                scheduled.append(occupied_opportunities[occupied_index])
                occupied_index += 1
        accepted_this_round = 0
        attempted_this_round = 0
        seen_neighborhoods: Counter[frozenset[int]] = Counter()
        round_gain = 0.0
        for pressure, displaced_count, _, _, _, _, focal in scheduled:
            if attempted_this_round >= settings.exchange_trials_per_round:
                break
            live_selected_keys = {join_key(value) for value in current.joins}
            if focal.key in live_selected_keys:
                continue
            component_by_patch, members = _component_partition(
                block.patches, current.joins
            )
            active_components = frozenset(
                (
                    component_by_patch[focal.match.first_patch_id],
                    component_by_patch[focal.match.second_patch_id],
                )
            )
            # Different focal edges in the same component pair are useful, but
            # cap repeated full reconstructions of one unchanged neighborhood.
            if seen_neighborhoods[active_components] >= 3:
                skipped_duplicate_neighborhoods += 1
                continue
            seen_neighborhoods[active_components] += 1
            active_patch_ids = {
                patch_id
                for component_id in active_components
                for patch_id in members[component_id]
            }
            outside = tuple(
                value
                for value in current.joins
                if value.first_patch_id not in active_patch_ids
                and value.second_patch_id not in active_patch_ids
            )
            if any(
                (value.first_patch_id in active_patch_ids)
                != (value.second_patch_id in active_patch_ids)
                for value in current.joins
            ):
                raise RuntimeError("a current component crosses its own exchange cut")
            internal = tuple(
                value
                for value in eligible
                if value.match.first_patch_id in active_patch_ids
                and value.match.second_patch_id in active_patch_ids
            )
            universe = {
                join_key(value): value for value in outside
            }
            universe.update({value.key: value.match for value in internal})
            fixed = frozenset(join_key(value) for value in outside)
            priorities = {
                key: candidate_by_key[key].benefit for key in universe
            }
            maximum_priority = max(priorities.values(), default=0.0)
            priorities[focal.key] = maximum_priority + abs(maximum_priority) + 1.0
            proposal = select_surface_joins(
                block.patches,
                tuple(universe.values()),
                fixed_join_keys=fixed,
                candidate_priorities=priorities,
            )
            attempted_this_round += 1
            attempted_count += 1
            before_quality = _selection_quality(
                current, block.patches, candidate_by_key
            )
            after_quality = _selection_quality(
                proposal, block.patches, candidate_by_key
            )
            focal_retained = focal.key in {
                join_key(value) for value in proposal.joins
            }
            before_keys = {join_key(value) for value in current.joins}
            after_keys = {join_key(value) for value in proposal.joins}
            accepted = after_quality > before_quality
            record = {
                "round": round_index,
                "trial": attempted_this_round - 1,
                "focalJoin": {
                    "firstPatchId": focal.match.first_patch_id,
                    "secondPatchId": focal.match.second_patch_id,
                    "faceAxis": focal.match.face.axis,
                    "faceAnchorXYZ": list(focal.match.face.anchor_xyz),
                    "family": focal.family,
                    "benefit": round(focal.benefit, 6),
                    "occupiedReplacementPressure": round(pressure, 6),
                },
                "activeComponents": len(active_components),
                "activePatches": len(active_patch_ids),
                "componentBridge": len(active_components) == 2,
                "occupiedAlternative": displaced_count > 0,
                "focalRetained": focal_retained,
                "removedJoins": len(before_keys - after_keys),
                "addedJoins": len(after_keys - before_keys),
                "benefitDelta": round(after_quality[0] - before_quality[0], 6),
                "accepted": accepted,
            }
            records.append(record)
            if accepted:
                current = proposal
                accepted_this_round += 1
                accepted_count += 1
                round_gain += after_quality[0] - before_quality[0]
        records.append(
            {
                "round": round_index,
                "summary": True,
                "attempted": attempted_this_round,
                "accepted": accepted_this_round,
                "benefitGain": round(round_gain, 6),
            }
        )
        if accepted_this_round == 0:
            break
    return current, tuple(records), {
        "attempted": attempted_count,
        "accepted": accepted_count,
        "skippedDuplicateNeighborhoods": skipped_duplicate_neighborhoods,
        "roundsCompleted": sum(value.get("summary") is True for value in records),
    }


def _gap_census(
    block: SurfaceBlock,
    catalog: SheetJoinCatalog,
) -> dict[str, int]:
    candidates_by_trace: dict[TraceKey, list[SheetJoinCandidate]] = defaultdict(list)
    for candidate in catalog.candidates:
        for trace in candidate.trace_keys:
            candidates_by_trace[trace].append(candidate)
    joined_traces = {
        (patch_id, value.face)
        for value in block.joins
        for patch_id in (value.first_patch_id, value.second_patch_id)
    }
    component_by_patch = dict(block.component_by_patch)
    patch_by_id = {value.patch_id: value for value in block.patches}
    patches_by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in block.patches:
        patches_by_cell[patch.cell_xyz].append(patch)
    result: Counter[str] = Counter()
    for boundary in block.unresolved_interior_traces:
        trace_key = (boundary.patch_id, boundary.trace.face)
        compatible = candidates_by_trace.get(trace_key, ())
        source_component = component_by_patch[boundary.patch_id]
        occupied = False
        same_component = False
        for candidate in compatible:
            other = (
                candidate.match.second_patch_id
                if candidate.match.first_patch_id == boundary.patch_id
                else candidate.match.first_patch_id
            )
            occupied = occupied or (other, boundary.trace.face) in joined_traces
            same_component = same_component or (
                component_by_patch[other] == source_component
            )
        if occupied:
            result["compatible-occupied"] += 1
        elif same_component:
            result["compatible-open-same-component"] += 1
        elif compatible:
            result["compatible-open-bridge"] += 1
        else:
            source = patch_by_id[boundary.patch_id]
            lower, upper = boundary.trace.face.adjacent_cells()
            target = upper if source.cell_xyz == lower else lower
            target_crosses = any(
                value.trace_on(boundary.trace.face) is not None
                for value in patches_by_cell.get(target, ())
            )
            result[
                "selected-incompatible" if target_crosses else "selected-misses-face"
            ] += 1
    return dict(sorted(result.items()))


def restitch_sheet_graph(
    block: SurfaceBlock,
    catalog: SheetJoinCatalog,
    policy: SheetMatchingPolicy,
    *,
    settings: SheetStitchingSettings | None = None,
) -> SheetRestitchResult:
    """Rebuild and then non-monotonically exchange whole sheet neighborhoods."""

    resolved = settings or SheetStitchingSettings()
    by_key = catalog.by_key()
    eligible = tuple(
        value
        for value in catalog.candidates
        if value.benefit > resolved.minimum_join_benefit
    )
    if not eligible:
        raise ValueError("sheet join catalog contains no positive-benefit candidates")
    degree: Counter[TraceKey] = Counter(
        trace for value in eligible for trace in value.trace_keys
    )
    face_optimal = _face_optimal_candidate_keys(eligible)
    priority_span = max((value.benefit for value in eligible), default=1.0)
    interior_endpoint_count = (
        2 * len(block.joins) + len(block.unresolved_interior_traces)
    )
    proposals: list[tuple[SurfaceJoinSelection, dict[str, Any]]] = []
    declared = SurfaceJoinSelection(tuple(block.joins), tuple())
    proposals.append(
        (
            declared,
            {
                "proposalIndex": -1,
                "proposal": "declared-topology-safe-input",
                **_selection_record(
                    block.patches,
                    declared,
                    by_key,
                    policy.strict_settings.unmatched_negative_log_likelihood,
                    interior_endpoint_count,
                ),
            },
        )
    )
    for restart in range(resolved.restart_count):
        priorities: dict[JoinKey, float] = {}
        for candidate in eligible:
            if restart == 0:
                priority = candidate.benefit
                proposal = "join-benefit"
            elif restart == 1:
                priority = candidate.benefit + (
                    2.0 * priority_span if candidate.key in face_optimal else 0.0
                )
                proposal = "exact-face-alignment"
            elif restart == 2:
                first, second = candidate.trace_keys
                priority = candidate.benefit / math.sqrt(
                    max(degree[first] * degree[second], 1)
                )
                proposal = "opportunity-normalized"
            else:
                priority = candidate.benefit * (
                    1.0
                    + resolved.priority_jitter_fraction
                    * _stable_jitter(candidate.key, restart)
                )
                proposal = "deterministic-perturbation"
            priorities[candidate.key] = float(priority)
        selection = select_surface_joins(
            block.patches,
            (value.match for value in eligible),
            candidate_priorities=priorities,
        )
        record = {
            "proposalIndex": restart,
            "proposal": proposal,
            **_selection_record(
                block.patches,
                selection,
                by_key,
                policy.strict_settings.unmatched_negative_log_likelihood,
                interior_endpoint_count,
            ),
        }
        proposals.append((selection, record))
    baseline = _block_record(
        block,
        by_key,
        policy.strict_settings.unmatched_negative_log_likelihood,
    )
    best_selection, best_record = min(
        proposals,
        key=lambda value: (
            -float(value[1]["totalJoinBenefit"]),
            -int(value[1]["retainedJoins"]),
            int(value[1]["components"]),
            int(value[1]["proposalIndex"]),
        ),
    )
    exchanged, exchange_records, exchange_statistics = _exchange_sheet_neighborhoods(
        block,
        eligible,
        by_key,
        best_selection,
        resolved,
    )
    exchanged_record = {
        "proposalIndex": int(best_record["proposalIndex"]),
        "proposal": f"{best_record['proposal']} + sheet-neighborhood-exchange",
        **_selection_record(
            block.patches,
            exchanged,
            by_key,
            policy.strict_settings.unmatched_negative_log_likelihood,
            interior_endpoint_count,
        ),
    }
    best_block = surface_block_from_retained_joins(
        block.grid,
        block.bounds,
        block.patches,
        exchanged.joins,
    )
    summary = {
        "baseline": {
            **baseline,
            "gapCensus": _gap_census(block, catalog),
        },
        "best": {
            **exchanged_record,
            "gapCensus": _gap_census(best_block, catalog),
        },
        "initialGlobalProposal": best_record,
        "sheetNeighborhoodExchange": {
            **exchange_statistics,
            "trials": list(exchange_records),
        },
        "delta": {
            key: round(float(exchanged_record[key]) - float(baseline[key]), 6)
            for key in (
                "retainedJoins",
                "components",
                "largestComponentPatchCount",
                "unresolvedInteriorTraceEndpoints",
                "retainedInteriorTraceFraction",
                "totalJoinBenefit",
                "objectiveCostWithoutConstant",
            )
        },
    }
    return SheetRestitchResult(
        best_block,
        summary,
        tuple(value for _, value in proposals),
    )


def _write_catalog(
    output: Path,
    catalog: SheetJoinCatalog,
    *,
    identity_sha256: str,
) -> dict[str, Any]:
    data_path = output / f"{SHEET_JOIN_CATALOG_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray(
                [value.match.first_patch_id for value in catalog.candidates],
                dtype=np.uint64,
            ),
            secondPatchId=np.asarray(
                [value.match.second_patch_id for value in catalog.candidates],
                dtype=np.uint64,
            ),
            faceAxis=np.asarray(
                [value.match.face.axis for value in catalog.candidates],
                dtype=np.int8,
            ),
            faceAnchorXYZ=np.asarray(
                [value.match.face.anchor_xyz for value in catalog.candidates],
                dtype=np.int32,
            ).reshape(len(catalog.candidates), 3),
            family=np.asarray(
                [value.family == "quarter-turn" for value in catalog.candidates],
                dtype=np.uint8,
            ),
            firstRank=np.asarray(
                [value.first_rank for value in catalog.candidates], dtype=np.int16
            ),
            secondRank=np.asarray(
                [value.second_rank for value in catalog.candidates], dtype=np.int16
            ),
            benefit=np.asarray(
                [value.benefit for value in catalog.candidates], dtype=np.float32
            ),
            currentlyRetained=np.asarray(
                [value.currently_retained for value in catalog.candidates],
                dtype=np.uint8,
            ),
            negativeLogLikelihood=np.asarray(
                [
                    value.match.negative_log_likelihood
                    for value in catalog.candidates
                ],
                dtype=np.float32,
            ),
            score=np.asarray(
                [value.match.score for value in catalog.candidates],
                dtype=np.float32,
            ),
        )
    temporary.replace(data_path)
    manifest = {
        "schema": SHEET_JOIN_CATALOG_SCHEMA,
        "version": SHEET_JOIN_CATALOG_VERSION,
        "state": "complete",
        "identitySha256": identity_sha256,
        "statistics": catalog.statistics(),
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(output / f"{SHEET_JOIN_CATALOG_STEM}.json", manifest)
    return manifest


def _write_join_selection(
    output: Path,
    block: SurfaceBlock,
    by_key: Mapping[JoinKey, SheetJoinCandidate],
) -> dict[str, Any]:
    path = output / "sheet-restitch-joins-v1.npz"
    joins = tuple(block.joins)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray(
                [value.first_patch_id for value in joins], dtype=np.uint64
            ),
            secondPatchId=np.asarray(
                [value.second_patch_id for value in joins], dtype=np.uint64
            ),
            faceAxis=np.asarray([value.face.axis for value in joins], dtype=np.int8),
            faceAnchorXYZ=np.asarray(
                [value.face.anchor_xyz for value in joins], dtype=np.int32
            ).reshape(len(joins), 3),
            catalogBenefit=np.asarray(
                [by_key[join_key(value)].benefit for value in joins],
                dtype=np.float32,
            ),
        )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def run_block_sheet_restitching(
    cluster_root: str | Path,
    materialized_root: str | Path,
    output_root: str | Path,
    *,
    settings: SheetStitchingSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a complete join catalog and globally re-stitch one block graph."""

    started = time.monotonic()
    resolved = settings or SheetStitchingSettings()
    cluster = Path(cluster_root).resolve()
    materialized = Path(materialized_root).resolve()
    output = Path(output_root).resolve()
    if output == materialized:
        raise ValueError("sheet restitch output must differ from its input graph")
    policy = SheetMatchingPolicy.from_cluster_root(cluster)
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_RESTITCH_SCHEMA,
        "version": SHEET_RESTITCH_VERSION,
        "clusterRoot": str(cluster),
        "clusterManifestSha256": sha256_file(
            cluster / "cluster-reselection-v1.json"
        ),
        "materializedRoot": str(materialized),
        "selectedPatchManifestSha256": sha256_file(
            materialized / "selected-patches-v1.json"
        ),
        "selectedPatchDataSha256": sha256_file(
            materialized / "selected-patches-v1.npz"
        ),
        "surfaceGraphManifestSha256": sha256_file(
            materialized / "surface-graph-v1.json"
        ),
        "surfaceGraphDataSha256": sha256_file(
            materialized / "surface-graph-v1.npz"
        ),
        "policy": policy.record(),
        "settings": resolved.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_stitching.py",
                "block.py",
                "matching.py",
                "surface_graph.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "sheet-restitch-v1.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet restitch output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SHEET_RESTITCH_SCHEMA,
        "version": SHEET_RESTITCH_VERSION,
        "state": "cataloging",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)
    block = read_surface_graph(materialized, verify=True)
    loaded = time.monotonic()
    catalog = enumerate_sheet_join_catalog(
        block,
        policy,
        settings=resolved,
    )
    catalog_manifest = _write_catalog(
        output,
        catalog,
        identity_sha256=identity_sha256,
    )
    cataloged = time.monotonic()
    manifest["state"] = "solving"
    atomic_json(manifest_path, manifest)
    result = restitch_sheet_graph(
        block,
        catalog,
        policy,
        settings=resolved,
    )
    solved = time.monotonic()
    selection = _write_join_selection(output, result.block, catalog.by_key())
    for name in ("selected-patches-v1.json", "selected-patches-v1.npz"):
        source_path = materialized / name
        target_path = output / name
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(target_path)
    graph_manifest = write_surface_graph(
        output,
        result.block,
        semantics=(
            "fixed Acus patch geometry with complete face correspondence "
            "catalog, exact face alignment, and topology-safe whole-sheet "
            "neighborhood exchanges"
        ),
        provenance={
            "inputRoot": str(materialized),
            "inputSurfaceGraphSha256": identity["surfaceGraphDataSha256"],
            "sheetRestitchIdentitySha256": identity_sha256,
            "candidateCatalogSha256": catalog_manifest["data"]["sha256"],
        },
    )
    summary = {
        "schema": "pareidolia.cubical-sheet-restitch-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(materialized),
        "settings": resolved.record(),
        "method": {
            "geometry": "fixed selected patch loops",
            "candidateUniverse": (
                "all pair-gated shared-face correspondences, including "
                "alternatives outside the locally optimal face alignment"
            ),
            "optimization": (
                "exact per-face alignment, whole-block deterministic proposals, "
                "and non-monotone whole-sheet neighborhood exchanges"
            ),
            "hardConstraints": [
                "one join per patch trace",
                "order-preserving face correspondences",
                "one patch per sheet component per cell",
                "crossing-feature consistency",
                "orientable polygon parity",
            ],
            "mutatesAcusGeometry": False,
            "materializesGraph": True,
        },
        "catalog": catalog_manifest["statistics"],
        "restitch": result.summary,
        "proposals": list(result.proposal_records),
        "artifacts": {
            "candidateManifest": f"{SHEET_JOIN_CATALOG_STEM}.json",
            "candidateData": catalog_manifest["data"],
            "selectedJoins": selection,
            "selectedPatches": {
                "manifest": "selected-patches-v1.json",
                "manifestSha256": sha256_file(
                    output / "selected-patches-v1.json"
                ),
                "data": "selected-patches-v1.npz",
                "dataSha256": sha256_file(output / "selected-patches-v1.npz"),
            },
            "surfaceGraph": {
                "manifest": "surface-graph-v1.json",
                "manifestSha256": sha256_file(output / "surface-graph-v1.json"),
                "data": graph_manifest["data"],
            },
        },
        "timingSeconds": {
            "loading": round(loaded - started, 6),
            "cataloging": round(cataloged - loaded, 6),
            "solving": round(solved - cataloged, 6),
            "total": round(time.monotonic() - started, 6),
        },
    }
    atomic_json(summary_path, summary)
    manifest.update(
        {
            "state": "complete",
            "summary": summary_path.name,
            "elapsedSeconds": summary["timingSeconds"]["total"],
        }
    )
    atomic_json(manifest_path, manifest)
    return summary
