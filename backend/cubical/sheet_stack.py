from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from typing import Hashable, Iterable


CandidateKey = Hashable


@dataclass(frozen=True, slots=True)
class OrderedMatchEvidence:
    """One possible correspondence in an ordered bipartite stack."""

    key: CandidateKey
    first_rank: int
    second_rank: int
    log_weight: float

    def __post_init__(self) -> None:
        if self.first_rank < 0 or self.second_rank < 0:
            raise ValueError("ordered stack ranks must be nonnegative")
        if not math.isfinite(self.log_weight):
            raise ValueError("ordered stack weights must be finite")


@dataclass(frozen=True, slots=True)
class OrderedMatchMarginal:
    """Exact evidence for one edge across all noncrossing matchings."""

    key: CandidateKey
    probability: float
    log_odds: float
    maximum_score_regret: float


@dataclass(frozen=True, slots=True)
class OrderedStackPosterior:
    """Partition function and edge marginals for one shared-face stack."""

    first_count: int
    second_count: int
    log_partition: float
    maximum_score: float
    marginals: tuple[OrderedMatchMarginal, ...]

    def by_key(self) -> dict[CandidateKey, OrderedMatchMarginal]:
        return {value.key: value for value in self.marginals}


def _logsumexp(values: Iterable[float]) -> float:
    entries = tuple(values)
    if not entries:
        return -math.inf
    maximum = max(entries)
    if maximum == -math.inf:
        return maximum
    return maximum + math.log(sum(math.exp(value - maximum) for value in entries))


def _log_difference(first: float, second: float) -> float:
    """Return log(exp(first) - exp(second)) for first >= second."""

    if second == -math.inf:
        return first
    if second > first + 1.0e-10:
        raise ValueError("logarithmic difference would be negative")
    delta = second - first
    if delta >= 0.0:
        # In exact arithmetic an edge never occupies every matching because
        # the all-unmatched state is always present.  Guard only roundoff here.
        delta = -1.0e-15
    return first + math.log(-math.expm1(delta))


def ordered_stack_posterior(
    evidence: Iterable[OrderedMatchEvidence],
) -> OrderedStackPosterior:
    """Evaluate every order-preserving partial matching exactly.

    An unmatched row or column has zero log weight.  Consequently a candidate
    log weight is a log likelihood ratio against leaving both of its ports
    unmatched.  The recurrence anchors every matching on the decision for its
    first remaining row, so matchings with skipped columns are counted once.
    """

    values = tuple(evidence)
    if not values:
        return OrderedStackPosterior(0, 0, 0.0, 0.0, tuple())
    keys = tuple(value.key for value in values)
    if len(set(keys)) != len(keys):
        raise ValueError("ordered stack candidates require unique keys")
    rank_pairs = tuple(
        (value.first_rank, value.second_rank) for value in values
    )
    if len(set(rank_pairs)) != len(rank_pairs):
        raise ValueError("an ordered stack has multiple candidates at one rank pair")
    first_count = 1 + max(value.first_rank for value in values)
    second_count = 1 + max(value.second_rank for value in values)
    by_row: dict[int, tuple[OrderedMatchEvidence, ...]] = {}
    for first_rank in range(first_count):
        by_row[first_rank] = tuple(
            sorted(
                (
                    value
                    for value in values
                    if value.first_rank == first_rank
                ),
                key=lambda value: (value.second_rank, repr(value.key)),
            )
        )

    @lru_cache(maxsize=None)
    def log_partition(
        first_start: int,
        first_stop: int,
        second_start: int,
        second_stop: int,
    ) -> float:
        @lru_cache(maxsize=None)
        def solve(first_rank: int, second_floor: int) -> float:
            if first_rank >= first_stop:
                return 0.0
            options = [solve(first_rank + 1, second_floor)]
            options.extend(
                value.log_weight
                + solve(first_rank + 1, value.second_rank + 1)
                for value in by_row[first_rank]
                if second_floor <= value.second_rank < second_stop
            )
            return _logsumexp(options)

        return solve(first_start, second_start)

    @lru_cache(maxsize=None)
    def maximum_score(
        first_start: int,
        first_stop: int,
        second_start: int,
        second_stop: int,
    ) -> float:
        @lru_cache(maxsize=None)
        def solve(first_rank: int, second_floor: int) -> float:
            if first_rank >= first_stop:
                return 0.0
            options = [solve(first_rank + 1, second_floor)]
            options.extend(
                value.log_weight
                + solve(first_rank + 1, value.second_rank + 1)
                for value in by_row[first_rank]
                if second_floor <= value.second_rank < second_stop
            )
            return max(options)

        return solve(first_start, second_start)

    total_log_partition = log_partition(
        0, first_count, 0, second_count
    )
    total_maximum = maximum_score(0, first_count, 0, second_count)
    marginals: list[OrderedMatchMarginal] = []
    for value in values:
        forced_log_partition = (
            log_partition(
                0,
                value.first_rank,
                0,
                value.second_rank,
            )
            + value.log_weight
            + log_partition(
                value.first_rank + 1,
                first_count,
                value.second_rank + 1,
                second_count,
            )
        )
        probability = math.exp(forced_log_partition - total_log_partition)
        probability = min(max(probability, 0.0), 1.0)
        without_log_partition = _log_difference(
            total_log_partition, forced_log_partition
        )
        log_odds = forced_log_partition - without_log_partition
        forced_maximum = (
            maximum_score(
                0,
                value.first_rank,
                0,
                value.second_rank,
            )
            + value.log_weight
            + maximum_score(
                value.first_rank + 1,
                first_count,
                value.second_rank + 1,
                second_count,
            )
        )
        marginals.append(
            OrderedMatchMarginal(
                value.key,
                probability,
                log_odds,
                max(total_maximum - forced_maximum, 0.0),
            )
        )
    marginals.sort(key=lambda value: repr(value.key))
    return OrderedStackPosterior(
        first_count,
        second_count,
        total_log_partition,
        total_maximum,
        tuple(marginals),
    )
