from __future__ import annotations

import hashlib
import json
import math
import shutil
import time
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, replace
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
from .sheet_curvature import SheetCurvatureAnalysis, analyze_sheet_curvature
from .sheet_lamination import (
    LayerExclusion,
    analyze_sheet_lamination,
    enumerate_layer_exclusions,
)
from .sheet_stack import OrderedMatchEvidence, ordered_stack_posterior
from .sheet_signed_graph import SignedEdge, SignedPartition, signed_graph_partition
from .sheet_transport import patch_stack_ranks
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
SHEETLET_GRAPH_SCHEMA = "pareidolia.cubical-typed-sheetlet-graph"
SHEETLET_GRAPH_VERSION = 1
SHEETLET_GRAPH_STEM = "sheetlet-graph-v1"

JoinKey = tuple[int, int, int, Int3]
TraceKey = tuple[int, GridFace]
COLLISION_CUT_ORDERS = frozenset(("forward", "reverse", "both"))


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
    unmatched_trace_penalty: float = 0.0
    restart_count: int = 12
    priority_jitter_fraction: float = 0.35
    exchange_round_count: int = 2
    exchange_trials_per_round: int = 24
    collision_cut_enabled: bool = True
    collision_cut_limit: int = 0
    collision_cut_order: str = "forward"
    curvature_refinement_enabled: bool = False
    curvature_neighborhood_radius: int = 3
    curvature_minimum_branch_support: int = 3
    curvature_robust_standard_deviations: float = 3.0
    curvature_minimum_calibration_joins: int = 32
    curvature_round_count: int = 3
    curvature_cut_penalty_weight: float = 1.0
    layer_partition_enabled: bool = False
    stack_transport_enabled: bool = False
    signed_partition_enabled: bool = True
    layer_repulsion_minimum_overlap_fraction: float = 0.05
    layer_repulsion_minimum_normal_separation_cells: float = 0.15
    layer_repulsion_scale: float = 0.0
    layer_exclusion_proximity_radius_cells: int = 2
    layer_exclusion_minimum_overlap_fraction: float = 0.25
    layer_exclusion_minimum_normal_separation_cells: float = 0.35
    layer_exclusion_maximum_normal_angle_degrees: float = 0.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.minimum_join_benefit):
            raise ValueError("minimum join benefit must be finite")
        if not math.isfinite(self.quarter_turn_penalty) or self.quarter_turn_penalty < 0:
            raise ValueError("quarter-turn penalty must be finite and nonnegative")
        if (
            not math.isfinite(self.unmatched_trace_penalty)
            or self.unmatched_trace_penalty < 0.0
        ):
            raise ValueError("unmatched trace penalty must be finite and nonnegative")
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
        if self.collision_cut_limit < 0:
            raise ValueError("collision-cut limit must be nonnegative")
        if self.collision_cut_order not in COLLISION_CUT_ORDERS:
            raise ValueError(
                "collision-cut order must be one of "
                f"{sorted(COLLISION_CUT_ORDERS)}"
            )
        if not isinstance(self.curvature_refinement_enabled, bool):
            raise ValueError("curvature-refinement enabled state must be boolean")
        if self.curvature_neighborhood_radius < 1:
            raise ValueError("curvature neighborhood radius must be positive")
        if self.curvature_minimum_branch_support < 1:
            raise ValueError("minimum curvature branch support must be positive")
        if (
            not math.isfinite(self.curvature_robust_standard_deviations)
            or self.curvature_robust_standard_deviations <= 0.0
        ):
            raise ValueError("curvature deviations must be finite and positive")
        if self.curvature_minimum_calibration_joins < 1:
            raise ValueError("minimum curvature calibration joins must be positive")
        if self.curvature_round_count < 1:
            raise ValueError("curvature refinement rounds must be positive")
        if (
            not math.isfinite(self.curvature_cut_penalty_weight)
            or self.curvature_cut_penalty_weight < 0.0
        ):
            raise ValueError("curvature cut penalty must be finite and nonnegative")
        if not isinstance(self.layer_partition_enabled, bool):
            raise ValueError("layer-partition enabled state must be boolean")
        if not isinstance(self.stack_transport_enabled, bool):
            raise ValueError("stack-transport enabled state must be boolean")
        if not isinstance(self.signed_partition_enabled, bool):
            raise ValueError("signed-partition enabled state must be boolean")
        if not (
            math.isfinite(self.layer_repulsion_minimum_overlap_fraction)
            and 0.0 <= self.layer_repulsion_minimum_overlap_fraction <= 1.0
        ):
            raise ValueError("layer-repulsion overlap must lie in [0, 1]")
        if (
            not math.isfinite(
                self.layer_repulsion_minimum_normal_separation_cells
            )
            or self.layer_repulsion_minimum_normal_separation_cells < 0.0
        ):
            raise ValueError("layer-repulsion separation must be nonnegative")
        if not math.isfinite(self.layer_repulsion_scale) or self.layer_repulsion_scale < 0.0:
            raise ValueError("layer-repulsion scale must be nonnegative")
        if self.layer_exclusion_proximity_radius_cells < 1:
            raise ValueError("layer-exclusion proximity radius must be positive")
        if not (
            math.isfinite(self.layer_exclusion_minimum_overlap_fraction)
            and 0.0
            <= self.layer_exclusion_minimum_overlap_fraction
            <= 1.0
        ):
            raise ValueError("layer-exclusion overlap must lie in [0, 1]")
        if (
            not math.isfinite(
                self.layer_exclusion_minimum_normal_separation_cells
            )
            or self.layer_exclusion_minimum_normal_separation_cells < 0.0
        ):
            raise ValueError(
                "layer-exclusion normal separation must be nonnegative"
            )
        if (
            not math.isfinite(
                self.layer_exclusion_maximum_normal_angle_degrees
            )
            or not 0.0
            <= self.layer_exclusion_maximum_normal_angle_degrees
            <= 90.0
        ):
            raise ValueError("layer-exclusion normal angle must lie in [0, 90]")

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
    face_probability: float = math.nan
    face_regret: float = math.nan
    partition_benefit: float | None = None

    @property
    def key(self) -> JoinKey:
        return join_key(self.match)

    @property
    def trace_keys(self) -> tuple[TraceKey, TraceKey]:
        return (
            (self.match.first_patch_id, self.match.face),
            (self.match.second_patch_id, self.match.face),
        )

    @property
    def objective_benefit(self) -> float:
        return (
            self.partition_benefit
            if self.partition_benefit is not None
            else self.benefit
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
            "positivePartitionBenefitCandidates": sum(
                value.objective_benefit > 0.0 for value in self.candidates
            ),
            "contextualizedFaceCandidates": sum(
                value.partition_benefit is not None for value in self.candidates
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
    catalog: SheetJoinCatalog
    summary: dict[str, Any]
    proposal_records: tuple[dict[str, Any], ...]
    layer_exclusions: tuple[LayerExclusion, ...]
    layer_repulsions: tuple[LayerExclusion, ...]
    layer_repulsion_penalties: Mapping[tuple[int, int], float]


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


def contextualize_sheet_join_catalog(
    catalog: SheetJoinCatalog,
) -> tuple[SheetJoinCatalog, dict[str, Any]]:
    """Replace independent edge scores with exact whole-face marginals.

    The raw pair likelihood remains available as ``benefit``.  The partition
    benefit is the edge log odds after summing over every order-preserving
    partial alignment of the complete stack on that face.  An isolated edge
    therefore keeps its original score, while a locally plausible shear edge
    that excludes a much stronger collective alignment becomes unattractive.
    """

    contextualized = sum(
        value.partition_benefit is not None for value in catalog.candidates
    )
    if contextualized:
        if contextualized != len(catalog.candidates):
            raise ValueError("sheet join catalog is only partly contextualized")
        probability = np.asarray(
            [value.face_probability for value in catalog.candidates],
            dtype=np.float64,
        )
        return catalog, {
            "enabled": True,
            "source": "precontextualized catalog",
            "faces": len({value.match.face for value in catalog.candidates}),
            "candidates": len(catalog.candidates),
            "positiveLogOddsCandidates": sum(
                value.objective_benefit > 0.0 for value in catalog.candidates
            ),
            "probabilityQuantiles": {
                name: round(float(value), 6)
                for name, value in zip(
                    ("minimum", "p10", "median", "p90", "maximum"),
                    np.percentile(probability, (0, 10, 50, 90, 100))
                    if len(probability)
                    else (0.0,) * 5,
                )
            },
        }
    by_face: dict[GridFace, list[SheetJoinCandidate]] = defaultdict(list)
    for value in catalog.candidates:
        by_face[value.match.face].append(value)
    marginal_by_key: dict[JoinKey, Any] = {}
    face_log_partitions: list[float] = []
    for values in by_face.values():
        posterior = ordered_stack_posterior(
            OrderedMatchEvidence(
                value.key,
                value.first_rank,
                value.second_rank,
                value.benefit,
            )
            for value in values
        )
        marginal_by_key.update(posterior.by_key())
        face_log_partitions.append(posterior.log_partition)
    candidates = tuple(
        replace(
            value,
            face_probability=marginal_by_key[value.key].probability,
            face_regret=marginal_by_key[value.key].maximum_score_regret,
            partition_benefit=marginal_by_key[value.key].log_odds,
        )
        for value in catalog.candidates
    )
    probability = np.asarray(
        [value.face_probability for value in candidates], dtype=np.float64
    )
    regret = np.asarray([value.face_regret for value in candidates], dtype=np.float64)
    contextual_catalog = SheetJoinCatalog(
        candidates,
        catalog.interior_face_count,
        catalog.unstable_face_count,
    )
    return contextual_catalog, {
        "enabled": True,
        "source": "exact ordered-face partition function",
        "faces": len(by_face),
        "candidates": len(candidates),
        "positiveLogOddsCandidates": sum(
            value.objective_benefit > 0.0 for value in candidates
        ),
        "negativeLogOddsDespitePositivePairBenefit": sum(
            value.benefit > 0.0 and value.objective_benefit <= 0.0
            for value in candidates
        ),
        "probabilityQuantiles": {
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "p10", "median", "p90", "maximum"),
                np.percentile(probability, (0, 10, 50, 90, 100))
                if len(probability)
                else (0.0,) * 5,
            )
        },
        "maximumScoreRegretQuantiles": {
            name: round(float(value), 6)
            for name, value in zip(
                ("minimum", "median", "p90", "maximum"),
                np.percentile(regret, (0, 50, 90, 100))
                if len(regret)
                else (0.0,) * 4,
            )
        },
        "meanFaceLogPartition": round(
            float(np.mean(face_log_partitions)) if face_log_partitions else 0.0,
            6,
        ),
    }


def _stable_jitter(key: JoinKey, restart: int) -> float:
    digest = hashlib.blake2b(
        repr((restart, key)).encode("utf-8"), digest_size=8
    ).digest()
    integer = int.from_bytes(digest, "little")
    return 2.0 * (integer / float((1 << 64) - 1)) - 1.0


def _face_optimal_candidate_keys(
    candidates: tuple[SheetJoinCandidate, ...],
    retained_join_reward: float = 0.0,
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
                        + candidate.objective_benefit
                        + retained_join_reward,
                        int(counts[first_index - 1, second_index - 1]) + 1,
                        (*paths[first_index - 1][second_index - 1], candidate.key),
                    )
                    chosen = better(chosen, diagonal)
                scores[first_index, second_index] = chosen[0]
                counts[first_index, second_index] = chosen[1]
                paths[first_index][second_index] = chosen[2]
        selected.update(paths[first_count][second_count])
    return frozenset(selected)


def _minimum_undirected_join_cut(
    member_patch_ids: tuple[int, ...],
    candidates: tuple[SheetJoinCandidate, ...],
    capacities: Mapping[JoinKey, float],
    source_patch_id: int,
    target_patch_id: int,
) -> tuple[frozenset[JoinKey], float]:
    """Return one deterministic minimum-capacity cut in a sheet component."""

    patch_ids = tuple(sorted(member_patch_ids))
    index_by_patch = {patch_id: index for index, patch_id in enumerate(patch_ids)}
    source = index_by_patch[source_patch_id]
    target = index_by_patch[target_patch_id]
    # Residual edge records are mutable [target, reverse-index, capacity].
    adjacency: list[list[list[float | int]]] = [[] for _ in patch_ids]

    def add_directed(first: int, second: int, capacity: float) -> None:
        forward: list[float | int] = [second, len(adjacency[second]), capacity]
        reverse: list[float | int] = [first, len(adjacency[first]), 0.0]
        adjacency[first].append(forward)
        adjacency[second].append(reverse)

    for candidate in candidates:
        first = index_by_patch[candidate.match.first_patch_id]
        second = index_by_patch[candidate.match.second_patch_id]
        capacity = max(float(capacities[candidate.key]), 1.0e-9)
        # Two directed arcs give an undirected edge its capacity in either cut
        # direction while retaining ordinary residual reverse arcs.
        add_directed(first, second, capacity)
        add_directed(second, first, capacity)

    epsilon = 1.0e-12
    while True:
        level = [-1] * len(patch_ids)
        level[source] = 0
        queue: deque[int] = deque((source,))
        while queue:
            first = queue.popleft()
            for edge in adjacency[first]:
                second = int(edge[0])
                if float(edge[2]) > epsilon and level[second] < 0:
                    level[second] = level[first] + 1
                    queue.append(second)
        if level[target] < 0:
            break
        cursor = [0] * len(patch_ids)

        def send(first: int, available: float) -> float:
            if first == target:
                return available
            while cursor[first] < len(adjacency[first]):
                edge = adjacency[first][cursor[first]]
                second = int(edge[0])
                capacity = float(edge[2])
                if capacity > epsilon and level[second] == level[first] + 1:
                    pushed = send(second, min(available, capacity))
                    if pushed > epsilon:
                        edge[2] = capacity - pushed
                        reverse = adjacency[second][int(edge[1])]
                        reverse[2] = float(reverse[2]) + pushed
                        return pushed
                cursor[first] += 1
            return 0.0

        while send(source, math.inf) > epsilon:
            pass

    reachable = {source}
    queue = deque((source,))
    while queue:
        first = queue.popleft()
        for edge in adjacency[first]:
            second = int(edge[0])
            if float(edge[2]) > epsilon and second not in reachable:
                reachable.add(second)
                queue.append(second)
    cut = frozenset(
        candidate.key
        for candidate in candidates
        if (
            index_by_patch[candidate.match.first_patch_id] in reachable
        ) != (
            index_by_patch[candidate.match.second_patch_id] in reachable
        )
    )
    return cut, sum(float(capacities[key]) for key in cut)


def _collision_cut_proposal(
    patches: tuple[ClippedPatch, ...],
    eligible: tuple[SheetJoinCandidate, ...],
    face_optimal_keys: frozenset[JoinKey],
    priorities: Mapping[JoinKey, float],
    *,
    maximum_cuts: int,
    reverse: bool = False,
    layer_exclusions: tuple[LayerExclusion, ...] = tuple(),
    stack_rank_by_patch: Mapping[int, int] | None = None,
) -> tuple[SurfaceJoinSelection, dict[str, Any]]:
    """Segment a dense attractive graph against lifted repulsive constraints.

    Greedy construction commits to early edges before it knows which dense
    sheet they create. This proposal reverses that order: begin with every
    independently optimal face edge, then separate both duplicate cell layers
    and tangent-overlap exclusions by minimum-cost sheet cuts. The ordinary
    exact validator remains authoritative for all local topology constraints.
    """

    candidate_by_key = {value.key: value for value in eligible}
    patch_by_id = {value.patch_id: value for value in patches}
    remaining = set(face_optimal_keys)
    removed: set[JoinKey] = set()
    cut_capacity = 0.0
    cut_records: list[dict[str, Any]] = []
    cut_index = 0
    while maximum_cuts == 0 or cut_index < maximum_cuts:
        joins = tuple(candidate_by_key[key].match for key in sorted(remaining))
        component_by_patch, members = _component_partition(patches, joins)
        collision: tuple[str, int, Int3 | None, int, int, float] | None = None
        for component_id, member_ids in sorted(
            members.items(), reverse=reverse
        ):
            by_cell: dict[Int3, list[int]] = defaultdict(list)
            for patch_id in member_ids:
                by_cell[patch_by_id[patch_id].cell_xyz].append(patch_id)
            duplicate = next(
                (
                    (cell, tuple(sorted(values)))
                    for cell, values in sorted(by_cell.items(), reverse=reverse)
                    if len(values) > 1
                ),
                None,
            )
            if duplicate is not None:
                cell, values = duplicate
                if reverse:
                    values = tuple(reversed(values))
                collision = (
                    "same-cell",
                    component_id,
                    cell,
                    values[0],
                    values[1],
                    math.inf,
                )
                break
        if collision is None:
            exclusion_values = (
                tuple(reversed(layer_exclusions))
                if reverse
                else layer_exclusions
            )
            for exclusion in exclusion_values:
                source, target = exclusion.pair
                component_id = component_by_patch[source]
                if component_by_patch[target] == component_id:
                    collision = (
                        "layer-exclusion",
                        component_id,
                        None,
                        source,
                        target,
                        exclusion.severity,
                    )
                    break
        if collision is None:
            break
        kind, component_id, cell, source, target, severity = collision
        member_ids = members[component_id]
        member_set = set(member_ids)
        internal = tuple(
            candidate_by_key[key]
            for key in sorted(remaining)
            if candidate_by_key[key].match.first_patch_id in member_set
            and candidate_by_key[key].match.second_patch_id in member_set
        )
        cut, capacity = _minimum_undirected_join_cut(
            member_ids,
            internal,
            priorities,
            source,
            target,
        )
        if not cut:
            raise RuntimeError("sheet graph conflict has no separating cut")
        remaining.difference_update(cut)
        removed.update(cut)
        cut_capacity += capacity
        cut_records.append(
            {
                "cut": cut_index + 1,
                "kind": kind,
                "componentPatches": len(member_ids),
                "cellXYZ": list(cell) if cell is not None else None,
                "separatedPatchIds": [source, target],
                "exclusionSeverity": (
                    round(float(severity), 6)
                    if math.isfinite(severity)
                    else None
                ),
                "removedJoins": len(cut),
                "capacity": round(capacity, 6),
            }
        )
        cut_index += 1

    repaired_joins = tuple(
        candidate_by_key[key].match for key in sorted(remaining)
    )
    safe_seed = select_surface_joins(
        patches,
        repaired_joins,
        candidate_priorities={key: priorities[key] for key in remaining},
        incompatible_patch_pairs=frozenset(
            value.pair for value in layer_exclusions
        ),
        stack_rank_by_patch=stack_rank_by_patch,
    )
    safe_keys = frozenset(join_key(value) for value in safe_seed.joins)
    completed = select_surface_joins(
        patches,
        tuple(value.match for value in eligible),
        fixed_join_keys=safe_keys,
        candidate_priorities=priorities,
        incompatible_patch_pairs=frozenset(
            value.pair for value in layer_exclusions
        ),
        stack_rank_by_patch=stack_rank_by_patch,
    )
    _, members = _component_partition(
        patches,
        tuple(candidate_by_key[key].match for key in sorted(remaining)),
    )
    collision_groups = 0
    for member_ids in members.values():
        counts = Counter(patch_by_id[value].cell_xyz for value in member_ids)
        collision_groups += sum(value > 1 for value in counts.values())
    component_by_patch = {
        patch_id: component_id
        for component_id, member_ids in members.items()
        for patch_id in member_ids
    }
    remaining_layer_conflicts = sum(
        component_by_patch[value.first_patch_id]
        == component_by_patch[value.second_patch_id]
        for value in layer_exclusions
    )
    return completed, {
        "cutOrder": "reverse" if reverse else "forward",
        "startingFaceOptimalJoins": len(face_optimal_keys),
        "cutIterations": len(cut_records),
        "cutLimitReached": (
            collision_groups > 0 or remaining_layer_conflicts > 0
        ),
        "remainingCollisionGroupsBeforeValidation": collision_groups,
        "remainingLayerExclusionsBeforeValidation": (
            remaining_layer_conflicts
        ),
        "removedJoinCount": len(removed),
        "removedJoinCapacity": round(cut_capacity, 6),
        "cutsByKind": dict(
            sorted(Counter(value["kind"] for value in cut_records).items())
        ),
        "topologySafeSeedJoins": len(safe_seed.joins),
        "completedJoins": len(completed.joins),
        "cuts": cut_records[:32],
    }


def _block_record(
    block: SurfaceBlock,
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    unmatched_cost: float,
    unmatched_trace_penalty: float = 0.0,
    layer_repulsion_penalties: Mapping[tuple[int, int], float] | None = None,
) -> dict[str, Any]:
    benefit = sum(candidate_by_key[join_key(value)].benefit for value in block.joins)
    partition_benefit = sum(
        candidate_by_key[join_key(value)].objective_benefit
        for value in block.joins
    )
    interior_endpoints = 2 * len(block.joins) + len(block.unresolved_interior_traces)
    component_by_patch = dict(block.component_by_patch)
    layer_repulsion = sum(
        penalty
        for (first, second), penalty in (
            layer_repulsion_penalties or {}
        ).items()
        if component_by_patch[first] == component_by_patch[second]
    )
    topology_objective = (
        benefit
        - layer_repulsion
        - unmatched_trace_penalty * len(block.unresolved_interior_traces)
    )
    objective_cost = unmatched_cost * interior_endpoints - benefit + layer_repulsion
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
        "totalPartitionBenefit": round(float(partition_benefit), 6),
        "totalLayerRepulsion": round(float(layer_repulsion), 6),
        "topologyObjective": round(float(topology_objective), 6),
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
    unmatched_trace_penalty: float = 0.0,
    layer_repulsion_penalties: Mapping[tuple[int, int], float] | None = None,
) -> dict[str, Any]:
    benefit = sum(
        candidate_by_key[join_key(value)].benefit for value in selection.joins
    )
    partition_benefit = sum(
        candidate_by_key[join_key(value)].objective_benefit
        for value in selection.joins
    )
    component_by_patch, members = _component_partition(patches, selection.joins)
    layer_repulsion = sum(
        penalty
        for (first, second), penalty in (
            layer_repulsion_penalties or {}
        ).items()
        if component_by_patch[first] == component_by_patch[second]
    )
    sizes = np.asarray([len(value) for value in members.values()], dtype=np.int64)
    unresolved = interior_endpoint_count - 2 * len(selection.joins)
    if unresolved < 0:
        raise RuntimeError("sheet selection retained more traces than physically exist")
    topology_objective = (
        benefit - layer_repulsion - unmatched_trace_penalty * unresolved
    )
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
        "totalPartitionBenefit": round(float(partition_benefit), 6),
        "totalLayerRepulsion": round(float(layer_repulsion), 6),
        "topologyObjective": round(float(topology_objective), 6),
        "objectiveCostWithoutConstant": round(
            float(
                unmatched_cost * interior_endpoint_count
                - benefit
                + layer_repulsion
            ),
            6,
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
    unmatched_trace_penalty: float = 0.0,
    layer_repulsion_penalties: Mapping[tuple[int, int], float] | None = None,
) -> tuple[float, float, int, int, int]:
    benefit = sum(
        candidate_by_key[join_key(value)].benefit for value in selection.joins
    )
    component_by_patch, members = _component_partition(patches, selection.joins)
    layer_repulsion = sum(
        penalty
        for (first, second), penalty in (
            layer_repulsion_penalties or {}
        ).items()
        if component_by_patch[first] == component_by_patch[second]
    )
    largest = max((len(value) for value in members.values()), default=0)
    adjusted = (
        benefit
        - layer_repulsion
        + 2.0 * unmatched_trace_penalty * len(selection.joins)
    )
    return adjusted, benefit, len(selection.joins), -len(members), largest


def _exchange_sheet_neighborhoods(
    block: SurfaceBlock,
    eligible: tuple[SheetJoinCandidate, ...],
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    initial: SurfaceJoinSelection,
    settings: SheetStitchingSettings,
    incompatible_patch_pairs: frozenset[tuple[int, int]] = frozenset(),
    stack_rank_by_patch: Mapping[int, int] | None = None,
    layer_repulsion_penalties: Mapping[tuple[int, int], float] | None = None,
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
    retained_join_reward = 2.0 * settings.unmatched_trace_penalty
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
                value.objective_benefit + retained_join_reward
                for value in displaced.values()
            )
            first_component = component_by_patch[candidate.match.first_patch_id]
            second_component = component_by_patch[candidate.match.second_patch_id]
            bridge = int(first_component != second_component)
            active_size = len(members[first_component]) + (
                len(members[second_component]) if bridge else 0
            )
            opportunities.append(
                (
                    candidate.objective_benefit
                    + retained_join_reward
                    - displaced_benefit,
                    len(displaced),
                    bridge,
                    candidate.objective_benefit,
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
                key: candidate_by_key[key].objective_benefit
                + retained_join_reward
                for key in universe
            }
            maximum_priority = max(priorities.values(), default=0.0)
            priorities[focal.key] = maximum_priority + abs(maximum_priority) + 1.0
            proposal = select_surface_joins(
                block.patches,
                tuple(universe.values()),
                fixed_join_keys=fixed,
                candidate_priorities=priorities,
                incompatible_patch_pairs=incompatible_patch_pairs,
                stack_rank_by_patch=stack_rank_by_patch,
            )
            attempted_this_round += 1
            attempted_count += 1
            before_quality = _selection_quality(
                current,
                block.patches,
                candidate_by_key,
                settings.unmatched_trace_penalty,
                layer_repulsion_penalties,
            )
            after_quality = _selection_quality(
                proposal,
                block.patches,
                candidate_by_key,
                settings.unmatched_trace_penalty,
                layer_repulsion_penalties,
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
                    "partitionBenefit": round(
                        focal.objective_benefit, 6
                    ),
                    "occupiedReplacementPressure": round(pressure, 6),
                },
                "activeComponents": len(active_components),
                "activePatches": len(active_patch_ids),
                "componentBridge": len(active_components) == 2,
                "occupiedAlternative": displaced_count > 0,
                "focalRetained": focal_retained,
                "removedJoins": len(before_keys - after_keys),
                "addedJoins": len(after_keys - before_keys),
                "benefitDelta": round(after_quality[1] - before_quality[1], 6),
                "topologyObjectiveDelta": round(
                    after_quality[0] - before_quality[0], 6
                ),
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
                "topologyObjectiveGain": round(round_gain, 6),
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


def _analyze_selection_curvature(
    patches: tuple[ClippedPatch, ...],
    selection: SurfaceJoinSelection,
    settings: SheetStitchingSettings,
    *,
    calibration: SheetCurvatureAnalysis | None = None,
) -> SheetCurvatureAnalysis:
    return analyze_sheet_curvature(
        patches,
        selection.joins,
        neighborhood_radius=settings.curvature_neighborhood_radius,
        minimum_branch_support=settings.curvature_minimum_branch_support,
        robust_standard_deviations=(
            settings.curvature_robust_standard_deviations
        ),
        minimum_calibration_joins=settings.curvature_minimum_calibration_joins,
        calibration=calibration,
    )


def _curvature_quality(
    selection: SurfaceJoinSelection,
    analysis: SheetCurvatureAnalysis,
) -> tuple[int, float, int]:
    flagged = tuple(value for value in analysis.join_curvature if value.flagged)
    excess = sum(max(value.pressure - 1.0, 0.0) ** 2 for value in flagged)
    # Geometry is authoritative.  Join cardinality only breaks a tie between
    # solutions with indistinguishable abrupt-curvature evidence.
    return len(flagged), round(float(excess), 12), -len(selection.joins)


def _curvature_refine_selection(
    block: SurfaceBlock,
    eligible: tuple[SheetJoinCandidate, ...],
    candidate_by_key: Mapping[JoinKey, SheetJoinCandidate],
    initial: SurfaceJoinSelection,
    settings: SheetStitchingSettings,
    incompatible_patch_pairs: frozenset[tuple[int, int]] = frozenset(),
    stack_rank_by_patch: Mapping[int, int] | None = None,
) -> tuple[
    SurfaceJoinSelection,
    SheetCurvatureAnalysis,
    SheetCurvatureAnalysis,
    dict[str, Any],
]:
    """Cut abrupt hinges, then exactly rematch only the opened topology.

    Every normal comparison is axial.  Calibration is frozen from the input
    graph so repeated rounds cannot tighten their own noise model by peeling
    off the current tail.  A focal bad join is separated with a minimum cut
    whose capacities preserve high-benefit, low-curvature joins.  The ordinary
    exact surface validator remains the sole authority for every refill.
    """

    calibration = _analyze_selection_curvature(
        block.patches,
        initial,
        settings,
    )
    if not calibration.calibration_sufficient:
        return initial, calibration, calibration, {
            "enabled": True,
            "applied": False,
            "reason": "insufficient robust calibration support",
            "forbiddenJoinCount": 0,
            "rounds": [],
        }
    retained_join_reward = 2.0 * settings.unmatched_trace_penalty
    eligible_by_key = {value.key: value for value in eligible}
    current = initial
    current_analysis = calibration
    forbidden: set[JoinKey] = set()
    round_records: list[dict[str, Any]] = []
    for round_index in range(settings.curvature_round_count):
        flagged = tuple(
            sorted(
                (value for value in current_analysis.join_curvature if value.flagged),
                key=lambda value: (-value.pressure, value.key),
            )
        )
        if not flagged:
            break
        analysis_by_key = current_analysis.by_key()
        remaining = {join_key(value): value for value in current.joins}
        round_forbidden: set[JoinKey] = set()
        cut_records: list[dict[str, Any]] = []
        focal_cut_count = 0
        for focal in flagged:
            if focal.key not in remaining:
                continue
            remaining_joins = tuple(remaining[key] for key in sorted(remaining))
            component_by_patch, members = _component_partition(
                block.patches, remaining_joins
            )
            source = focal.key[0]
            target = focal.key[1]
            source_component = component_by_patch[source]
            if component_by_patch[target] != source_component:
                continue
            member_ids = members[source_component]
            member_set = set(member_ids)
            internal = tuple(
                candidate_by_key[key]
                for key in sorted(remaining)
                if remaining[key].first_patch_id in member_set
                and remaining[key].second_patch_id in member_set
            )
            capacities: dict[JoinKey, float] = {}
            for candidate in internal:
                pressure = analysis_by_key[candidate.key].pressure
                curvature_excess = max(pressure - 1.0, 0.0)
                preservation = max(
                    candidate.objective_benefit + retained_join_reward,
                    1.0e-6,
                )
                capacities[candidate.key] = preservation / (
                    1.0
                    + settings.curvature_cut_penalty_weight
                    * curvature_excess**2
                )
            cut, capacity = _minimum_undirected_join_cut(
                member_ids,
                internal,
                capacities,
                source,
                target,
            )
            if not cut:
                raise RuntimeError("abrupt sheet hinge has no separating cut")
            remaining = {
                key: value for key, value in remaining.items() if key not in cut
            }
            round_forbidden.update(cut)
            focal_cut_count += 1
            if len(cut_records) < 16:
                cut_records.append(
                    {
                        "focalJoin": focal.record(),
                        "componentPatches": len(member_ids),
                        "removedJoins": len(cut),
                        "capacity": round(float(capacity), 6),
                    }
                )
        if not round_forbidden:
            break
        candidate_forbidden = forbidden | round_forbidden
        cut_selection = SurfaceJoinSelection(
            tuple(remaining[key] for key in sorted(remaining)),
            tuple(),
        )
        # Removing joins cannot create a topology violation, but replaying the
        # graph here makes that invariant executable rather than assumed.
        surface_block_from_retained_joins(
            block.grid,
            block.bounds,
            block.patches,
            cut_selection.joins,
        )
        safe_keys = frozenset(remaining)
        universe = dict(remaining)
        universe.update(
            {
                key: value.match
                for key, value in eligible_by_key.items()
                if key not in candidate_forbidden
                and math.degrees(value.match.normal_angle_radians)
                <= calibration.direct_scale.upper_limit_degrees + 1.0e-9
            }
        )
        direct_limit = calibration.direct_scale.upper_limit_degrees
        priorities: dict[JoinKey, float] = {}
        for key in universe:
            candidate = candidate_by_key[key]
            direct_degrees = math.degrees(candidate.match.normal_angle_radians)
            direct_pressure = direct_degrees / max(direct_limit, 1.0e-6)
            curvature_excess = max(direct_pressure - 1.0, 0.0)
            priorities[key] = (
                max(candidate.objective_benefit + retained_join_reward, 1.0e-6)
                / (
                    1.0
                    + settings.curvature_cut_penalty_weight
                    * curvature_excess**2
                )
            )
        refilled = select_surface_joins(
            block.patches,
            tuple(universe.values()),
            fixed_join_keys=safe_keys,
            candidate_priorities=priorities,
            incompatible_patch_pairs=incompatible_patch_pairs,
            stack_rank_by_patch=stack_rank_by_patch,
        )
        cut_analysis = _analyze_selection_curvature(
            block.patches,
            cut_selection,
            settings,
            calibration=calibration,
        )
        refilled_analysis = _analyze_selection_curvature(
            block.patches,
            refilled,
            settings,
            calibration=calibration,
        )
        refilled_quality = _curvature_quality(refilled, refilled_analysis)
        before_quality = _curvature_quality(current, current_analysis)
        # Intermediate rounds retain every topology-safe smooth alternative so
        # newly exposed multiscale inconsistencies can be cut on the following
        # pass.  The final round closes with the clean cut state rather than
        # knowingly leaving a fresh abrupt hinge behind.
        if (
            round_index + 1 < settings.curvature_round_count
            and refilled_quality < before_quality
        ):
            proposal = refilled
            proposal_analysis = refilled_analysis
            proposal_kind = "minimum-cut + exact open-trace refill"
        else:
            proposal = cut_selection
            proposal_analysis = cut_analysis
            proposal_kind = "minimum-cut"
        after_quality = _curvature_quality(proposal, proposal_analysis)
        accepted = after_quality < before_quality
        round_records.append(
            {
                "round": round_index,
                "inputFlaggedJoins": before_quality[0],
                "inputCurvatureExcess": before_quality[1],
                "focalCuts": focal_cut_count,
                "forbiddenJoinsAdded": len(round_forbidden),
                "cutOnlyJoins": len(cut_selection.joins),
                "refilledJoins": len(refilled.joins),
                "proposal": proposal_kind,
                "outputFlaggedJoins": after_quality[0],
                "outputCurvatureExcess": after_quality[1],
                "accepted": accepted,
                "cuts": cut_records,
            }
        )
        if not accepted:
            break
        current = proposal
        current_analysis = proposal_analysis
        forbidden = candidate_forbidden
    return current, calibration, current_analysis, {
        "enabled": True,
        "applied": current is not initial,
        "forbiddenJoinCount": len(forbidden),
        "rounds": round_records,
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


def _compile_layer_partition_constraints(
    block: SurfaceBlock,
    settings: SheetStitchingSettings,
) -> tuple[tuple[LayerExclusion, ...], float, dict[str, Any]]:
    if not settings.layer_partition_enabled:
        return tuple(), 0.0, {"enabled": False}
    configured_limit = settings.layer_exclusion_maximum_normal_angle_degrees
    calibration_record: dict[str, Any]
    if configured_limit > 0.0:
        normal_limit = configured_limit
        calibration_record = {
            "source": "configured",
            "maximumNormalAngleDegrees": round(normal_limit, 6),
        }
    else:
        curvature = _analyze_selection_curvature(
            block.patches,
            SurfaceJoinSelection(tuple(block.joins), tuple()),
            settings,
        )
        if curvature.calibration_sufficient:
            normal_limit = curvature.direct_scale.upper_limit_degrees
            source = "robust retained-join calibration"
        else:
            normal_limit = 30.0
            source = "insufficient-calibration fallback"
        calibration_record = {
            "source": source,
            "maximumNormalAngleDegrees": round(normal_limit, 6),
            "calibrationSufficient": curvature.calibration_sufficient,
            "directBendScale": curvature.direct_scale.record(),
        }
    exclusions = enumerate_layer_exclusions(
        block.patches,
        cell_size_xyz=block.grid.cell_size_xyz,
        maximum_parallel_normal_angle_degrees=normal_limit,
        proximity_radius_cells=(
            settings.layer_exclusion_proximity_radius_cells
        ),
        minimum_overlap_fraction=(
            settings.layer_exclusion_minimum_overlap_fraction
        ),
        minimum_normal_separation_cells=(
            settings.layer_exclusion_minimum_normal_separation_cells
        ),
    )
    return exclusions, normal_limit, {
        "enabled": True,
        "normalCalibration": calibration_record,
        "candidateExclusions": len(exclusions),
        "sameCellExclusions": sum(
            value.cell_delta_xyz == (0, 0, 0) for value in exclusions
        ),
        "crossCellExclusions": sum(
            value.cell_delta_xyz != (0, 0, 0) for value in exclusions
        ),
    }


def _compile_signed_layer_repulsions(
    block: SurfaceBlock,
    settings: SheetStitchingSettings,
    maximum_parallel_normal_angle_degrees: float,
    automatic_scale: float,
) -> tuple[
    tuple[LayerExclusion, ...],
    dict[tuple[int, int], float],
    dict[str, Any],
]:
    if not (
        settings.layer_partition_enabled
        and settings.signed_partition_enabled
    ):
        return tuple(), {}, {"enabled": False}
    repulsions = enumerate_layer_exclusions(
        block.patches,
        cell_size_xyz=block.grid.cell_size_xyz,
        maximum_parallel_normal_angle_degrees=(
            maximum_parallel_normal_angle_degrees
        ),
        proximity_radius_cells=(
            settings.layer_exclusion_proximity_radius_cells
        ),
        minimum_overlap_fraction=(
            settings.layer_repulsion_minimum_overlap_fraction
        ),
        minimum_normal_separation_cells=(
            settings.layer_repulsion_minimum_normal_separation_cells
        ),
    )
    scale = settings.layer_repulsion_scale or automatic_scale
    penalties = {
        value.pair: scale * value.severity for value in repulsions
    }
    return repulsions, penalties, {
        "enabled": True,
        "semantics": (
            "soft lifted evidence that two nodes occupy distinct local layers"
        ),
        "repulsions": len(repulsions),
        "scale": round(scale, 6),
        "scaleSource": (
            "configured"
            if settings.layer_repulsion_scale > 0.0
            else "one unmatched-trace negative log likelihood"
        ),
        "minimumOverlapFraction": (
            settings.layer_repulsion_minimum_overlap_fraction
        ),
        "minimumNormalSeparationCells": (
            settings.layer_repulsion_minimum_normal_separation_cells
        ),
        "totalPenalty": round(sum(penalties.values()), 6),
    }


def _signed_partition_proposal(
    block: SurfaceBlock,
    candidates: tuple[SheetJoinCandidate, ...],
    layer_repulsion_penalties: Mapping[tuple[int, int], float],
    incompatible_patch_pairs: frozenset[tuple[int, int]],
) -> tuple[SurfaceJoinSelection, SignedPartition, dict[str, Any]]:
    by_cell: dict[Int3, list[int]] = defaultdict(list)
    for patch in block.patches:
        by_cell[patch.cell_xyz].append(patch.patch_id)
    same_cell_pairs = tuple(
        (first, second)
        for values in by_cell.values()
        for index, first in enumerate(values)
        for second in values[index + 1 :]
    )
    hard_pairs = tuple(same_cell_pairs) + tuple(incompatible_patch_pairs)
    partition = signed_graph_partition(
        (value.patch_id for value in block.patches),
        (
            SignedEdge(
                value.match.first_patch_id,
                value.match.second_patch_id,
                value.benefit,
            )
            for value in candidates
        ),
        (
            SignedEdge(first, second, penalty)
            for (first, second), penalty in layer_repulsion_penalties.items()
        ),
        hard_separate_pairs=hard_pairs,
    )
    allowed = tuple(
        value
        for value in candidates
        if partition.component_by_node[value.match.first_patch_id]
        == partition.component_by_node[value.match.second_patch_id]
    )
    selection = select_surface_joins(
        block.patches,
        (value.match for value in allowed),
        candidate_priorities={value.key: value.benefit for value in allowed},
        incompatible_patch_pairs=incompatible_patch_pairs,
    )
    return selection, partition, {
        **partition.record(),
        "sameCellHardSeparations": len(same_cell_pairs),
        "liftedLayerHardSeparations": len(incompatible_patch_pairs),
        "allowedContinuationCandidates": len(allowed),
        "topologySafeJoins": len(selection.joins),
        "topologyDeferredByReason": dict(
            sorted(
                Counter(
                    value.reason for value in selection.deferred_joins
                ).items()
            )
        ),
    }


def restitch_sheet_graph(
    block: SurfaceBlock,
    catalog: SheetJoinCatalog,
    policy: SheetMatchingPolicy,
    *,
    settings: SheetStitchingSettings | None = None,
) -> SheetRestitchResult:
    """Rebuild and then non-monotonically exchange whole sheet neighborhoods."""

    resolved = settings or SheetStitchingSettings()
    face_partition_statistics: dict[str, Any] = {"enabled": False}
    if resolved.layer_partition_enabled:
        catalog, face_partition_statistics = contextualize_sheet_join_catalog(
            catalog
        )
    by_key = catalog.by_key()
    raw_eligible = tuple(
        value
        for value in catalog.candidates
        if value.benefit > resolved.minimum_join_benefit
    )
    eligible = tuple(
        value
        for value in catalog.candidates
        if value.objective_benefit > resolved.minimum_join_benefit
    )
    if not eligible:
        raise ValueError("sheet join catalog contains no positive-benefit candidates")
    (
        layer_exclusions,
        layer_normal_limit,
        layer_partition_statistics,
    ) = _compile_layer_partition_constraints(block, resolved)
    incompatible_patch_pairs = frozenset(
        value.pair for value in layer_exclusions
    )
    (
        layer_repulsions,
        layer_repulsion_penalties,
        signed_partition_statistics,
    ) = _compile_signed_layer_repulsions(
        block,
        resolved,
        layer_normal_limit,
        policy.strict_settings.unmatched_negative_log_likelihood,
    )
    stack_rank_by_patch: Mapping[int, int] | None = None
    if resolved.layer_partition_enabled and resolved.stack_transport_enabled:
        _, stack_rank_by_patch = patch_stack_ranks(block.patches)
    degree: Counter[TraceKey] = Counter(
        trace for value in eligible for trace in value.trace_keys
    )
    retained_join_reward = 2.0 * resolved.unmatched_trace_penalty
    face_optimal = _face_optimal_candidate_keys(eligible, retained_join_reward)
    priority_span = max(
        (value.objective_benefit + retained_join_reward for value in eligible),
        default=1.0,
    )
    interior_endpoint_count = (
        2 * len(block.joins) + len(block.unresolved_interior_traces)
    )
    proposals: list[tuple[SurfaceJoinSelection, dict[str, Any]]] = []
    if signed_partition_statistics["enabled"]:
        (
            signed_selection,
            _,
            signed_proposal_statistics,
        ) = _signed_partition_proposal(
            block,
            raw_eligible,
            layer_repulsion_penalties,
            incompatible_patch_pairs,
        )
        proposals.append(
            (
                signed_selection,
                {
                    "proposalIndex": -3,
                    "proposal": "signed lifted component partition",
                    "proposalDiagnostics": signed_proposal_statistics,
                    **_selection_record(
                        block.patches,
                        signed_selection,
                        by_key,
                        policy.strict_settings.unmatched_negative_log_likelihood,
                        interior_endpoint_count,
                        resolved.unmatched_trace_penalty,
                        layer_repulsion_penalties,
                    ),
                },
            )
        )
    if incompatible_patch_pairs or stack_rank_by_patch is not None:
        declared = select_surface_joins(
            block.patches,
            block.joins,
            candidate_priorities={
                join_key(value): by_key[join_key(value)].objective_benefit
                + retained_join_reward
                for value in block.joins
            },
            incompatible_patch_pairs=incompatible_patch_pairs,
            stack_rank_by_patch=stack_rank_by_patch,
        )
        declared_name = (
            "declared-input + lifted layer and stack-transport constraints"
        )
    else:
        declared = SurfaceJoinSelection(tuple(block.joins), tuple())
        declared_name = "declared-topology-safe-input"
    proposals.append(
        (
            declared,
            {
                "proposalIndex": -1,
                "proposal": declared_name,
                **_selection_record(
                    block.patches,
                    declared,
                    by_key,
                    policy.strict_settings.unmatched_negative_log_likelihood,
                    interior_endpoint_count,
                    resolved.unmatched_trace_penalty,
                    layer_repulsion_penalties,
                ),
            },
        )
    )
    if resolved.collision_cut_enabled:
        objective_priorities = {
            value.key: value.objective_benefit + retained_join_reward
            for value in eligible
        }
        if resolved.collision_cut_order == "forward":
            cut_orders = (False,)
        elif resolved.collision_cut_order == "reverse":
            cut_orders = (True,)
        else:
            cut_orders = (False, True)
        for order_index, reverse in enumerate(cut_orders):
            cut_selection, cut_diagnostics = _collision_cut_proposal(
                block.patches,
                eligible,
                face_optimal,
                objective_priorities,
                maximum_cuts=resolved.collision_cut_limit,
                reverse=reverse,
                layer_exclusions=layer_exclusions,
                stack_rank_by_patch=stack_rank_by_patch,
            )
            order_name = "reverse" if reverse else "forward"
            proposals.append(
                (
                    cut_selection,
                    {
                        "proposalIndex": -2 - order_index,
                        "proposal": (
                            "dense-face-alignment + minimum collision cuts "
                            f"({order_name})"
                        ),
                        "proposalDiagnostics": cut_diagnostics,
                        **_selection_record(
                            block.patches,
                            cut_selection,
                            by_key,
                            policy.strict_settings.unmatched_negative_log_likelihood,
                            interior_endpoint_count,
                            resolved.unmatched_trace_penalty,
                            layer_repulsion_penalties,
                        ),
                    },
                )
            )
    for restart in range(resolved.restart_count):
        priorities: dict[JoinKey, float] = {}
        for candidate in eligible:
            objective_benefit = (
                candidate.objective_benefit + retained_join_reward
            )
            if restart == 0:
                priority = objective_benefit
                proposal = "join-benefit"
            elif restart == 1:
                priority = objective_benefit + (
                    2.0 * priority_span if candidate.key in face_optimal else 0.0
                )
                proposal = "exact-face-alignment"
            elif restart == 2:
                first, second = candidate.trace_keys
                priority = objective_benefit / math.sqrt(
                    max(degree[first] * degree[second], 1)
                )
                proposal = "opportunity-normalized"
            elif restart == 3:
                first, second = candidate.trace_keys
                conflict_pressure = max(degree[first] + degree[second] - 1, 1)
                priority = objective_benefit / conflict_pressure
                proposal = "endpoint-scarcity-normalized"
            elif restart == 4:
                first, second = candidate.trace_keys
                conflict_pressure = max(degree[first] * degree[second], 1)
                # This proposal deliberately makes scarce trace resources the
                # primary order and likelihood only the deterministic tie-break.
                # It exposes cardinality solutions that strongest-first greedy
                # selection cannot reach; exact objective scoring still decides
                # whether the proposal survives.
                priority = (
                    1.0 / conflict_pressure
                    + 1.0e-3 * objective_benefit / max(priority_span, 1.0e-12)
                )
                proposal = "scarce-traces-first"
            else:
                priority = objective_benefit * (
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
            incompatible_patch_pairs=incompatible_patch_pairs,
            stack_rank_by_patch=stack_rank_by_patch,
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
                resolved.unmatched_trace_penalty,
                layer_repulsion_penalties,
            ),
        }
        proposals.append((selection, record))
    baseline = _block_record(
        block,
        by_key,
        policy.strict_settings.unmatched_negative_log_likelihood,
        resolved.unmatched_trace_penalty,
        layer_repulsion_penalties,
    )
    best_selection, best_record = min(
        proposals,
        key=lambda value: (
            -float(value[1]["topologyObjective"]),
            -int(value[1]["retainedJoins"]),
            int(value[1]["components"]),
            int(value[1]["proposalIndex"]),
        ),
    )
    refinement_candidates = (
        raw_eligible
        if best_record["proposal"] == "signed lifted component partition"
        else eligible
    )
    refinement_incompatible_pairs = (
        incompatible_patch_pairs
    )
    exchanged, exchange_records, exchange_statistics = _exchange_sheet_neighborhoods(
        block,
        refinement_candidates,
        by_key,
        best_selection,
        resolved,
        refinement_incompatible_pairs,
        stack_rank_by_patch,
        layer_repulsion_penalties,
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
            resolved.unmatched_trace_penalty,
            layer_repulsion_penalties,
        ),
    }
    if resolved.curvature_refinement_enabled:
        (
            refined,
            curvature_before,
            curvature_after,
            curvature_statistics,
        ) = _curvature_refine_selection(
            block,
            refinement_candidates,
            by_key,
            exchanged,
            resolved,
            refinement_incompatible_pairs,
            stack_rank_by_patch,
        )
        final_selection = refined
        final_record = {
            "proposalIndex": int(best_record["proposalIndex"]),
            "proposal": (
                f"{best_record['proposal']} + sheet-neighborhood-exchange "
                "+ curvature-aware topology refinement"
            ),
            **_selection_record(
                block.patches,
                refined,
                by_key,
                policy.strict_settings.unmatched_negative_log_likelihood,
                interior_endpoint_count,
                resolved.unmatched_trace_penalty,
                layer_repulsion_penalties,
            ),
        }
        curvature_summary = {
            **curvature_statistics,
            "before": curvature_before.record(),
            "after": curvature_after.record(),
        }
    else:
        final_selection = exchanged
        final_record = exchanged_record
        curvature_summary = {
            "enabled": False,
            "applied": False,
        }
    best_block = surface_block_from_retained_joins(
        block.grid,
        block.bounds,
        block.patches,
        final_selection.joins,
    )
    if resolved.layer_partition_enabled:
        before_lamination = analyze_sheet_lamination(
            block.patches,
            block.joins,
            cell_size_xyz=block.grid.cell_size_xyz,
            maximum_parallel_normal_angle_degrees=layer_normal_limit,
            proximity_radius_cells=(
                resolved.layer_exclusion_proximity_radius_cells
            ),
            minimum_overlap_fraction=(
                resolved.layer_exclusion_minimum_overlap_fraction
            ),
            minimum_normal_separation_cells=(
                resolved.layer_exclusion_minimum_normal_separation_cells
            ),
            exclusions=layer_exclusions,
        )
        after_lamination = analyze_sheet_lamination(
            block.patches,
            final_selection.joins,
            cell_size_xyz=block.grid.cell_size_xyz,
            maximum_parallel_normal_angle_degrees=layer_normal_limit,
            proximity_radius_cells=(
                resolved.layer_exclusion_proximity_radius_cells
            ),
            minimum_overlap_fraction=(
                resolved.layer_exclusion_minimum_overlap_fraction
            ),
            minimum_normal_separation_cells=(
                resolved.layer_exclusion_minimum_normal_separation_cells
            ),
            exclusions=layer_exclusions,
        )
        layer_partition_summary = {
            **layer_partition_statistics,
            "before": before_lamination.record(),
            "after": after_lamination.record(),
        }
    else:
        layer_partition_summary = layer_partition_statistics
    stack_transport_summary = {
        "enabled": stack_rank_by_patch is not None,
        "rankSemantics": (
            "height-ordered selected Acus sheetlets within each cell"
            if stack_rank_by_patch is not None
            else None
        ),
        "inputJoinsRejectedForNonzeroHolonomy": sum(
            value.reason == "stack-transport-cycle"
            for value in declared.deferred_joins
        ),
        "acceptedGraphNonzeroHolonomy": (
            0 if stack_rank_by_patch is not None else None
        ),
        "constraint": (
            "all sheet components share one integer cell gauge"
            if stack_rank_by_patch is not None
            else None
        ),
    }
    summary = {
        "baseline": {
            **baseline,
            "gapCensus": _gap_census(block, catalog),
        },
        "best": {
            **final_record,
            "gapCensus": _gap_census(best_block, catalog),
        },
        "initialGlobalProposal": best_record,
        "preCurvatureRefinement": exchanged_record,
        "sheetNeighborhoodExchange": {
            **exchange_statistics,
            "trials": list(exchange_records),
        },
        "sheetCurvatureRefinement": curvature_summary,
        "orderedFacePartition": face_partition_statistics,
        "stackTransport": stack_transport_summary,
        "signedLayerRepulsion": signed_partition_statistics,
        "sheetletLayerPartition": layer_partition_summary,
        "delta": {
            key: round(float(final_record[key]) - float(baseline[key]), 6)
            for key in (
                "retainedJoins",
                "components",
                "largestComponentPatchCount",
                "unresolvedInteriorTraceEndpoints",
                "retainedInteriorTraceFraction",
                "totalJoinBenefit",
                "totalPartitionBenefit",
                "totalLayerRepulsion",
                "topologyObjective",
                "objectiveCostWithoutConstant",
            )
        },
    }
    return SheetRestitchResult(
        best_block,
        catalog,
        summary,
        tuple(value for _, value in proposals),
        layer_exclusions,
        layer_repulsions,
        layer_repulsion_penalties,
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
            faceProbability=np.asarray(
                [value.face_probability for value in catalog.candidates],
                dtype=np.float32,
            ),
            faceRegret=np.asarray(
                [value.face_regret for value in catalog.candidates],
                dtype=np.float32,
            ),
            partitionBenefit=np.asarray(
                [
                    (
                        value.partition_benefit
                        if value.partition_benefit is not None
                        else math.nan
                    )
                    for value in catalog.candidates
                ],
                dtype=np.float32,
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
            catalogPartitionBenefit=np.asarray(
                [by_key[join_key(value)].objective_benefit for value in joins],
                dtype=np.float32,
            ),
            faceProbability=np.asarray(
                [by_key[join_key(value)].face_probability for value in joins],
                dtype=np.float32,
            ),
        )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def _write_typed_sheetlet_graph(
    output: Path,
    block: SurfaceBlock,
    catalog: SheetJoinCatalog,
    exclusions: tuple[LayerExclusion, ...],
    repulsions: tuple[LayerExclusion, ...],
    repulsion_penalties: Mapping[tuple[int, int], float],
    *,
    identity_sha256: str,
) -> dict[str, Any]:
    """Persist fixed active nodes plus attractive and repulsive typed edges."""

    path = output / f"{SHEETLET_GRAPH_STEM}.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    patches = tuple(sorted(block.patches, key=lambda value: value.patch_id))
    candidates = catalog.candidates
    fiber_values = np.full((len(patches), 3), np.nan, dtype=np.float32)
    for index, patch in enumerate(patches):
        if patch.estimate.fiber_xyz is not None:
            fiber_values[index] = np.asarray(
                patch.estimate.fiber_xyz, dtype=np.float32
            )
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            nodePatchId=np.asarray(
                [value.patch_id for value in patches], dtype=np.uint64
            ),
            nodeCellXYZ=np.asarray(
                [value.cell_xyz for value in patches], dtype=np.int32
            ).reshape(len(patches), 3),
            nodeNormalXYZ=np.asarray(
                [value.estimate.normal_xyz for value in patches],
                dtype=np.float32,
            ).reshape(len(patches), 3),
            nodeFiberXYZ=fiber_values,
            nodeHeight=np.asarray(
                [value.estimate.height_from_cell_center for value in patches],
                dtype=np.float32,
            ),
            nodeConfidence=np.asarray(
                [value.estimate.confidence for value in patches],
                dtype=np.float32,
            ),
            continuationFirstPatchId=np.asarray(
                [value.match.first_patch_id for value in candidates],
                dtype=np.uint64,
            ),
            continuationSecondPatchId=np.asarray(
                [value.match.second_patch_id for value in candidates],
                dtype=np.uint64,
            ),
            continuationFaceAxis=np.asarray(
                [value.match.face.axis for value in candidates], dtype=np.int8
            ),
            continuationFaceAnchorXYZ=np.asarray(
                [value.match.face.anchor_xyz for value in candidates],
                dtype=np.int32,
            ).reshape(len(candidates), 3),
            continuationFamily=np.asarray(
                [value.family == "quarter-turn" for value in candidates],
                dtype=np.uint8,
            ),
            continuationBenefit=np.asarray(
                [value.benefit for value in candidates], dtype=np.float32
            ),
            continuationPartitionBenefit=np.asarray(
                [value.objective_benefit for value in candidates],
                dtype=np.float32,
            ),
            continuationFaceProbability=np.asarray(
                [value.face_probability for value in candidates],
                dtype=np.float32,
            ),
            continuationFaceRegret=np.asarray(
                [value.face_regret for value in candidates],
                dtype=np.float32,
            ),
            continuationNegativeLogLikelihood=np.asarray(
                [
                    value.match.negative_log_likelihood
                    for value in candidates
                ],
                dtype=np.float32,
            ),
            continuationNormalAngleDegrees=np.asarray(
                [
                    math.degrees(value.match.normal_angle_radians)
                    for value in candidates
                ],
                dtype=np.float32,
            ),
            continuationFiberAngleDegrees=np.asarray(
                [
                    (
                        math.degrees(value.match.fiber_angle_radians)
                        if value.match.fiber_angle_radians is not None
                        else math.nan
                    )
                    for value in candidates
                ],
                dtype=np.float32,
            ),
            exclusionFirstPatchId=np.asarray(
                [value.first_patch_id for value in exclusions], dtype=np.uint64
            ),
            exclusionSecondPatchId=np.asarray(
                [value.second_patch_id for value in exclusions], dtype=np.uint64
            ),
            exclusionReferenceNormalXYZ=np.asarray(
                [value.reference_normal_xyz for value in exclusions],
                dtype=np.float32,
            ).reshape(len(exclusions), 3),
            exclusionCellDeltaXYZ=np.asarray(
                [value.cell_delta_xyz for value in exclusions], dtype=np.int16
            ).reshape(len(exclusions), 3),
            exclusionOverlapFraction=np.asarray(
                [value.overlap_fraction for value in exclusions],
                dtype=np.float32,
            ),
            exclusionNormalSeparationCells=np.asarray(
                [value.normal_separation_cells for value in exclusions],
                dtype=np.float32,
            ),
            exclusionNormalAngleDegrees=np.asarray(
                [value.normal_angle_degrees for value in exclusions],
                dtype=np.float32,
            ),
            exclusionFiberAngleDegrees=np.asarray(
                [
                    (
                        value.fiber_angle_degrees
                        if value.fiber_angle_degrees is not None
                        else math.nan
                    )
                    for value in exclusions
                ],
                dtype=np.float32,
            ),
            exclusionSeverity=np.asarray(
                [value.severity for value in exclusions], dtype=np.float32
            ),
            repulsionFirstPatchId=np.asarray(
                [value.first_patch_id for value in repulsions], dtype=np.uint64
            ),
            repulsionSecondPatchId=np.asarray(
                [value.second_patch_id for value in repulsions], dtype=np.uint64
            ),
            repulsionCellDeltaXYZ=np.asarray(
                [value.cell_delta_xyz for value in repulsions], dtype=np.int16
            ).reshape(len(repulsions), 3),
            repulsionOverlapFraction=np.asarray(
                [value.overlap_fraction for value in repulsions],
                dtype=np.float32,
            ),
            repulsionNormalSeparationCells=np.asarray(
                [value.normal_separation_cells for value in repulsions],
                dtype=np.float32,
            ),
            repulsionWeight=np.asarray(
                [repulsion_penalties[value.pair] for value in repulsions],
                dtype=np.float32,
            ),
        )
    temporary.replace(path)
    manifest = {
        "schema": SHEETLET_GRAPH_SCHEMA,
        "version": SHEETLET_GRAPH_VERSION,
        "state": "complete",
        "identitySha256": identity_sha256,
        "nodeSemantics": "fixed active Acus sheetlet candidate",
        "edgeTypes": {
            "continuation": (
                "attractive shared-face same-surface hypothesis with "
                "raw pair likelihood and exact ordered-stack marginal"
            ),
            "exclusion": (
                "repulsive lifted pair that cannot share one local surface "
                "chart"
            ),
            "repulsion": (
                "soft lifted distinct-layer evidence charged once when both "
                "nodes enter one component"
            ),
        },
        "counts": {
            "nodes": len(patches),
            "continuationEdges": len(candidates),
            "exclusionEdges": len(exclusions),
            "repulsionEdges": len(repulsions),
        },
        "data": {
            "path": path.name,
            "bytes": path.stat().st_size,
            "sha256": sha256_file(path),
        },
    }
    atomic_json(output / f"{SHEETLET_GRAPH_STEM}.json", manifest)
    return manifest


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
                "sheet_curvature.py",
                "sheet_lamination.py",
                "sheet_stack.py",
                "sheet_signed_graph.py",
                "sheet_transport.py",
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
    catalog_manifest = _write_catalog(
        output,
        result.catalog,
        identity_sha256=identity_sha256,
    )
    typed_graph_manifest = _write_typed_sheetlet_graph(
        output,
        block,
        result.catalog,
        result.layer_exclusions,
        result.layer_repulsions,
        result.layer_repulsion_penalties,
        identity_sha256=identity_sha256,
    )
    selection = _write_join_selection(
        output, result.block, result.catalog.by_key()
    )
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
            "catalog, dense collision-cut segmentation, exact topology "
            "validation, whole-sheet neighborhood exchanges, and optional "
            "axial curvature-aware hinge refinement"
            ", and optional lifted layer-exclusion partitioning"
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
                "exact per-face alignment, minimum-cost segmentation of dense "
                "same-cell collision components, whole-block deterministic "
                "proposals, non-monotone whole-sheet neighborhood exchanges, "
                "optional robust curvature cuts with exact open-trace refill, "
                "exact ordered-face marginalization, and optional lifted "
                "attractive/repulsive graph partitioning"
            ),
            "hardConstraints": [
                "one join per patch trace",
                "order-preserving face correspondences",
                "path-independent integer transport of the full cell stack",
                "one patch per sheet component per cell",
                "crossing-feature consistency",
                "orientable polygon parity",
                "optional transitive layer-exclusion pairs",
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
            "typedSheetletGraph": {
                "manifest": f"{SHEETLET_GRAPH_STEM}.json",
                "data": typed_graph_manifest["data"],
                "counts": typed_graph_manifest["counts"],
            },
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
