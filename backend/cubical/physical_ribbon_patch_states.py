from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_collective import (
    _conflict_neighbors,
    _objective,
    _shared_endpoint_conflicts,
    _triangle_region_counts,
    optimize_collective_patch,
)
from .physical_ribbon_configuration import _adjacency, _component_labels
from .physical_ribbon_continuity import (
    write_continuity_overview,
    write_largest_component_montage,
)
from .physical_ribbon_patch_holes import (
    PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
    PhysicalRibbonPatchHoleSettings,
    build_physical_ribbon_surface_complex,
    extract_surface_boundary_loops,
)


PHYSICAL_RIBBON_PATCH_STATE_SCHEMA = "pareidolia.physical-ribbon-patch-state"
PHYSICAL_RIBBON_PATCH_STATE_VERSION = 1
PHYSICAL_RIBBON_PATCH_STATE_STEM = "physical-ribbon-patch-state-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonPatchStateSettings:
    """General gates for haloed, collective surface reconfiguration.

    A state is not a ribbon or a cell.  It is every CT-supported closed-hole
    frontier belonging to one already reconstructed surface component.  The
    selected surface outside the alternating interface conflicts is an
    immutable halo.  The complete local matching is optimized at once and is
    committed only when exact reconstruction makes that component denser.
    """

    minimum_added_ribbon_count: int = 2
    minimum_objective_gain: float = 0.0
    minimum_context_profile_correlation: float = 0.50
    minimum_competing_layer_margin: float = 0.10
    minimum_patch_coverage: float = 0.20
    minimum_retained_boundary_fraction: float = 0.05
    minimum_boundary_anchor_count: int = 1
    minimum_added_two_core_fraction: float = 0.50
    minimum_added_tangent_ratio: float = 0.01
    minimum_surface_realization_fraction: float = 0.80
    minimum_triangle_area_retention: float = 0.98
    minimum_density_only_retained_boundary_fraction: float = 0.60
    minimum_density_only_boundary_anchor_count: int = 3
    candidate_coverage_weight: float = 0.08
    continuity_weight_multiplier: float = 1.0
    surface_alignment_weight_multiplier: float = 1.0
    maximum_local_optimization_sweeps: int = 8
    maximum_component_state_variants: int = 16
    maximum_forced_exclusion_trials: int = 24
    maximum_individual_hole_scopes: int = 8
    use_exact_binary_optimizer: bool = True
    maximum_exact_binary_nodes: int = 512
    maximum_exact_binary_states: int = 2
    exact_binary_time_limit_seconds: float = 30.0
    maximum_exact_state_rounds: int = 64
    minimum_variant_hamming_fraction: float = 0.02
    screen_ct_supported_counterexamples: bool = True
    minimum_counterexample_ct_supported_fraction: float = 0.90
    maximum_preview_components: int = 64

    def __post_init__(self) -> None:
        positive_integer = (
            self.minimum_added_ribbon_count,
            self.minimum_boundary_anchor_count,
            self.minimum_density_only_boundary_anchor_count,
            self.maximum_local_optimization_sweeps,
            self.maximum_component_state_variants,
            self.maximum_exact_binary_nodes,
            self.maximum_exact_binary_states,
            self.maximum_exact_state_rounds,
            self.maximum_preview_components,
        )
        if any(value < 1 for value in positive_integer):
            raise ValueError("patch-state integer settings must be positive")
        if self.maximum_forced_exclusion_trials < 0:
            raise ValueError("forced-exclusion trial count must be nonnegative")
        if self.maximum_individual_hole_scopes < 0:
            raise ValueError("individual-hole scope count must be nonnegative")
        if not math.isfinite(self.minimum_objective_gain):
            raise ValueError("patch-state objective gate must be finite")
        if (
            not math.isfinite(self.exact_binary_time_limit_seconds)
            or self.exact_binary_time_limit_seconds <= 0.0
        ):
            raise ValueError("exact binary time limit must be finite and positive")
        for value in (
            self.candidate_coverage_weight,
            self.continuity_weight_multiplier,
            self.surface_alignment_weight_multiplier,
        ):
            if not math.isfinite(value) or value < 0.0:
                raise ValueError("patch-state objective weights must be nonnegative")
        for value in (
            self.minimum_context_profile_correlation,
            self.minimum_patch_coverage,
            self.minimum_retained_boundary_fraction,
            self.minimum_added_two_core_fraction,
            self.minimum_added_tangent_ratio,
            self.minimum_surface_realization_fraction,
            self.minimum_triangle_area_retention,
            self.minimum_density_only_retained_boundary_fraction,
            self.minimum_variant_hamming_fraction,
            self.minimum_counterexample_ct_supported_fraction,
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError("patch-state fractions must lie in [0, 1]")
        if not math.isfinite(self.minimum_competing_layer_margin):
            raise ValueError("competing-layer margin must be finite")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_holes_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("holes root must identify exactly one complete artifact")
    return matches[0]


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def _resolve_depth_field_manifest(
    root: str | Path,
) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema") == "pareidolia.physical-ribbon-depth-field"
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("depth-field root must identify exactly one complete artifact")
    return matches[0]


def _validate_depth_field(
    holes_path: Path,
    holes_manifest: Mapping[str, Any],
    holes: Mapping[str, np.ndarray],
    depth_path: Path,
    depth_manifest: Mapping[str, Any],
    depth: Mapping[str, np.ndarray],
) -> None:
    reference = depth_manifest.get("identity", {}).get("holes", {})
    if (
        reference.get("manifestPath") != str(holes_path)
        or reference.get("manifestSha256") != sha256_file(holes_path)
        or reference.get("dataSha256") != holes_manifest["data"]["sha256"]
    ):
        raise ValueError("depth field was not measured on the requested holes")
    if not np.array_equal(depth["holePatchOffset"], holes["patchOffset"]):
        raise ValueError("depth field and hole patch rasters differ")
    if not np.array_equal(depth["holeLoopIndex"], holes["scoredLoopIndex"]):
        raise ValueError("depth field and scored hole order differ")
    if not np.array_equal(depth["candidateDepthOffset"], holes["patchCandidateOffset"]):
        raise ValueError("depth field and ribbon candidate order differ")


def _coverage_objective(
    selected: np.ndarray,
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    coverage: np.ndarray,
    pixel_weight: np.ndarray,
) -> float:
    value = _objective(selected, node_score, edge_first, edge_second, edge_weight)
    if coverage.size and np.any(selected):
        covered = np.any(coverage[selected], axis=0)
        value += float(np.sum(pixel_weight[covered]))
    return value


def _highs_version(*, required: bool) -> str | None:
    try:
        import highspy
    except ImportError as error:
        if required:
            raise RuntimeError(
                "exact patch-state optimization requires highspy; install "
                "requirements-optimization.txt"
            ) from error
        return None
    return str(highspy.Highs().version())


def optimize_collective_patch_coverage_binary(
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    conflict_first: np.ndarray,
    conflict_second: np.ndarray,
    coverage: np.ndarray,
    pixel_weight: np.ndarray,
    *,
    maximum_states: int,
    time_limit_seconds: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Globally solve complete patch assignments as a binary program.

    Node ownership, exact profile crossings, pairwise continuation, and
    saturated whole-field CT coverage are represented explicitly.  The second
    objective is coverage-lexicographic: no unary or continuation preference
    can trade away one supported raster pixel.  This is a bounded whole-hole
    check, not a cell or frontier growth rule.
    """

    import highspy

    node_score = np.asarray(node_score, dtype=np.float64)
    edge_first = np.asarray(edge_first, dtype=np.int32)
    edge_second = np.asarray(edge_second, dtype=np.int32)
    edge_weight = np.asarray(edge_weight, dtype=np.float64)
    conflict_first = np.asarray(conflict_first, dtype=np.int32)
    conflict_second = np.asarray(conflict_second, dtype=np.int32)
    coverage = np.asarray(coverage, dtype=bool)
    pixel_weight = np.asarray(pixel_weight, dtype=np.float64)
    node_count = len(node_score)
    if maximum_states < 1:
        raise ValueError("binary patch-state count must be positive")
    if coverage.shape != (node_count, len(pixel_weight)):
        raise ValueError("binary patch coverage matrix has the wrong shape")
    if len(edge_first) != len(edge_second) or len(edge_first) != len(edge_weight):
        raise ValueError("binary patch continuation arrays differ in length")
    if len(conflict_first) != len(conflict_second):
        raise ValueError("binary patch conflict arrays differ in length")

    regularizer_bound = float(
        np.sum(np.abs(node_score)) + np.sum(np.abs(edge_weight))
    )
    positive_pixel = pixel_weight[pixel_weight > 0.0]
    coverage_epsilon = (
        float(np.min(positive_pixel)) / max(4.0 * regularizer_bound, 1.0)
        if len(positive_pixel)
        else 0.0
    )
    profiles = (
        ("binary-canonical", 1.0, 1.0, 1.0),
        (
            "binary-coverage-lexicographic",
            coverage_epsilon,
            coverage_epsilon,
            1.0,
        ),
    )
    per_profile = max(int(math.ceil(maximum_states / len(profiles))), 1)
    discovered: dict[bytes, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    for profile_index, (
        profile_name,
        node_scale,
        edge_scale,
        coverage_scale,
    ) in enumerate(profiles):
        solver = highspy.Highs()
        solver.silent()
        solver.setOptionValue("time_limit", float(time_limit_seconds))
        solver.setOptionValue("mip_rel_gap", 0.0)
        node_variable = solver.addBinaries(node_count)
        edge_variable = solver.addBinaries(len(edge_first))
        pixel_variable = solver.addBinaries(len(pixel_weight))
        for first, second in zip(conflict_first, conflict_second):
            solver.addConstr(
                node_variable[int(first)] + node_variable[int(second)] <= 1.0
            )
        for row, (first, second) in enumerate(zip(edge_first, edge_second)):
            pair = edge_variable[row]
            left = node_variable[int(first)]
            right = node_variable[int(second)]
            solver.addConstr(pair <= left)
            solver.addConstr(pair <= right)
            solver.addConstr(pair >= left + right - 1.0)
        for pixel in range(len(pixel_weight)):
            covering = np.flatnonzero(coverage[:, pixel])
            if len(covering):
                solver.addConstr(
                    pixel_variable[pixel]
                    <= solver.qsum(
                        node_variable[int(node)] for node in covering
                    )
                )
            else:
                solver.addConstr(pixel_variable[pixel] <= 0.0)
        objective = solver.qsum(
            float(node_scale * value) * node_variable[row]
            for row, value in enumerate(node_score)
        )
        objective += solver.qsum(
            float(edge_scale * value) * edge_variable[row]
            for row, value in enumerate(edge_weight)
        )
        objective += solver.qsum(
            float(coverage_scale * value) * pixel_variable[row]
            for row, value in enumerate(pixel_weight)
        )
        for solution_rank in range(per_profile):
            solver.maximize(objective)
            status = solver.getModelStatus()
            solution = solver.getSolution()
            status_name = solver.modelStatusToString(status)
            if not solution.value_valid:
                run_records.append(
                    {
                        "profile": profile_name,
                        "rank": solution_rank,
                        "status": status_name,
                        "solutionAvailable": False,
                    }
                )
                break
            selected = np.asarray(solver.val(node_variable)) > 0.5
            key = selected.tobytes()
            canonical = _coverage_objective(
                selected,
                node_score,
                edge_first,
                edge_second,
                edge_weight,
                coverage,
                pixel_weight,
            )
            covered_fraction = (
                float(np.mean(np.any(coverage[selected], axis=0)))
                if coverage.shape[1]
                else 0.0
            )
            record = {
                "selected": selected,
                "canonicalObjective": canonical,
                "profile": profile_name,
                "profileIndex": profile_index,
                "profileObjective": float(solver.getObjectiveValue()),
                "provenOptimal": status == highspy.HighsModelStatus.kOptimal,
                "coveredPixelFraction": covered_fraction,
            }
            prior = discovered.get(key)
            if prior is None or canonical > float(prior["canonicalObjective"]):
                discovered[key] = record
            information = solver.getInfo()
            run_records.append(
                {
                    "profile": profile_name,
                    "rank": solution_rank,
                    "status": status_name,
                    "solutionAvailable": True,
                    "provenOptimal": bool(record["provenOptimal"]),
                    "canonicalObjective": round(canonical, 6),
                    "coveredPixelFraction": round(covered_fraction, 6),
                    "mipGap": round(float(information.mip_gap), 9),
                    "mipNodeCount": int(information.mip_node_count),
                }
            )
            selected_node = np.flatnonzero(selected)
            unselected_node = np.flatnonzero(~selected)
            solver.addConstr(
                solver.qsum(
                    node_variable[int(node)] for node in selected_node
                )
                - solver.qsum(
                    node_variable[int(node)] for node in unselected_node
                )
                <= float(len(selected_node) - 1)
            )
    ordered = sorted(
        discovered.values(),
        key=lambda record: (
            float(record["canonicalObjective"]),
            float(record["coveredPixelFraction"]),
            bool(record["provenOptimal"]),
        ),
        reverse=True,
    )[:maximum_states]
    return ordered, {
        "solver": "HiGHS mixed-integer programming",
        "solverVersion": str(highspy.Highs().version()),
        "nodeCount": node_count,
        "continuationProductVariableCount": len(edge_first),
        "coverageVariableCount": len(pixel_weight),
        "conflictConstraintCount": len(conflict_first),
        "profileCount": len(profiles),
        "retainedStateCount": len(ordered),
        "provenOptimalStateCount": int(
            sum(bool(record["provenOptimal"]) for record in ordered)
        ),
        "runs": run_records,
    }


def optimize_collective_patch_coverage(
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    conflict_first: np.ndarray,
    conflict_second: np.ndarray,
    coverage: np.ndarray,
    pixel_weight: np.ndarray,
    *,
    maximum_sweeps: int,
    initial_selection: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one complete patch with a saturated dense-coverage term.

    Candidate density is a property of the entire missing surface, not a unary
    reward on one ribbon.  Coverage therefore pays once per CT-supported patch
    pixel even when many competing ribbons land there.  Deterministic global
    starts construct complete matchings before exact add/remove/swap ascent,
    allowing a coherent surface to cross the first-node barrier.
    """

    states, statistics = optimize_collective_patch_coverage_ensemble(
        node_score,
        edge_first,
        edge_second,
        edge_weight,
        conflict_first,
        conflict_second,
        coverage,
        pixel_weight,
        maximum_sweeps=maximum_sweeps,
        initial_selection=initial_selection,
        maximum_states=1,
        maximum_forced_exclusion_trials=0,
        minimum_hamming_fraction=0.0,
    )
    return states[0], statistics


def _optimize_coverage_from_start(
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    conflicts: tuple[np.ndarray, ...],
    coverage: np.ndarray,
    pixel_weight: np.ndarray,
    selected_value: np.ndarray,
    *,
    maximum_sweeps: int,
    forbidden: np.ndarray,
) -> tuple[np.ndarray, float, int]:
    selected = np.asarray(selected_value, dtype=bool).copy()
    selected[forbidden] = False
    current = _coverage_objective(
        selected,
        node_score,
        edge_first,
        edge_second,
        edge_weight,
        coverage,
        pixel_weight,
    )
    sweeps = 0
    for sweep in range(maximum_sweeps):
        changed = False
        # Exact one-variable removals preserve pair and saturated coverage
        # accounting; simultaneous marginal pruning would double-count.
        while np.any(selected):
            active = np.flatnonzero(selected)
            best_remove_gain = 1.0e-9
            best_remove = -1
            for node_value in active:
                trial = selected.copy()
                trial[int(node_value)] = False
                value = _coverage_objective(
                    trial,
                    node_score,
                    edge_first,
                    edge_second,
                    edge_weight,
                    coverage,
                    pixel_weight,
                )
                gain = value - current
                if gain > best_remove_gain:
                    best_remove_gain = gain
                    best_remove = int(node_value)
            if best_remove < 0:
                break
            selected[best_remove] = False
            current += best_remove_gain
            changed = True
        best_gain = 1.0e-9
        best_trial: np.ndarray | None = None
        for node_value in np.flatnonzero(~selected & ~forbidden):
            node = int(node_value)
            trial = selected.copy()
            if len(conflicts[node]):
                trial[conflicts[node]] = False
            trial[node] = True
            value = _coverage_objective(
                trial,
                node_score,
                edge_first,
                edge_second,
                edge_weight,
                coverage,
                pixel_weight,
            )
            gain = value - current
            if gain > best_gain:
                best_gain = gain
                best_trial = trial
        if best_trial is not None:
            selected = best_trial
            current += best_gain
            changed = True
        sweeps = sweep + 1
        if not changed:
            break
    return (
        selected,
        _coverage_objective(
            selected,
            node_score,
            edge_first,
            edge_second,
            edge_weight,
            coverage,
            pixel_weight,
        ),
        sweeps,
    )


def _normalized_hamming(first: np.ndarray, second: np.ndarray) -> float:
    relevant = np.asarray(first, dtype=bool) | np.asarray(second, dtype=bool)
    denominator = max(int(np.count_nonzero(relevant)), 1)
    return (
        float(np.count_nonzero(np.asarray(first) != np.asarray(second)))
        / denominator
    )


def optimize_collective_patch_coverage_ensemble(
    node_score: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
    conflict_first: np.ndarray,
    conflict_second: np.ndarray,
    coverage: np.ndarray,
    pixel_weight: np.ndarray,
    *,
    maximum_sweeps: int,
    initial_selection: np.ndarray,
    maximum_states: int,
    maximum_forced_exclusion_trials: int,
    minimum_hamming_fraction: float,
    use_exact_binary_optimizer: bool = False,
    maximum_exact_binary_nodes: int = 512,
    maximum_exact_binary_states: int = 4,
    exact_binary_time_limit_seconds: float = 60.0,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    """Return diverse complete local optima under a fixed objective ensemble.

    A residual patch is a discrete correspondence problem: several globally
    coherent matchings may have nearly identical proxy score but different
    exact topology.  We therefore optimize a declared ensemble of physical
    priorities and retain distinct whole-patch states for exact reconstruction.
    Forced exclusions branch only on choices made by the canonical-best
    ensemble state; they do not grow or judge one cell in isolation.
    """

    node_score = np.asarray(node_score, dtype=np.float64)
    edge_first = np.asarray(edge_first, dtype=np.int32)
    edge_second = np.asarray(edge_second, dtype=np.int32)
    edge_weight = np.asarray(edge_weight, dtype=np.float64)
    coverage = np.asarray(coverage, dtype=bool)
    pixel_weight = np.asarray(pixel_weight, dtype=np.float64)
    initial = np.asarray(initial_selection, dtype=bool)
    node_count = len(node_score)
    if maximum_states < 1:
        raise ValueError("maximum_states must be positive")
    if maximum_forced_exclusion_trials < 0:
        raise ValueError("maximum_forced_exclusion_trials must be nonnegative")
    if maximum_exact_binary_nodes < 1 or maximum_exact_binary_states < 1:
        raise ValueError("exact binary optimizer bounds must be positive")
    if not 0.0 <= minimum_hamming_fraction <= 1.0:
        raise ValueError("minimum_hamming_fraction must lie in [0, 1]")
    if coverage.shape != (node_count, len(pixel_weight)):
        raise ValueError("patch coverage matrix has the wrong shape")
    conflicts = _conflict_neighbors(
        node_count,
        np.asarray(conflict_first, dtype=np.int32),
        np.asarray(conflict_second, dtype=np.int32),
    )
    incident = np.zeros(node_count, dtype=np.float64)
    np.add.at(incident, edge_first, edge_weight)
    np.add.at(incident, edge_second, edge_weight)
    coverage_potential = coverage.astype(np.float64) @ pixel_weight
    # (name, unary multiplier, continuity multiplier, dense-coverage multiplier)
    profiles = (
        ("balanced", 1.0, 1.0, 1.0),
        ("coverage", 0.75, 0.75, 1.6),
        ("continuity", 0.85, 1.6, 0.65),
        ("unary", 1.55, 0.60, 0.80),
        ("coverage-continuity", 0.65, 1.30, 1.35),
        ("conservative", 1.25, 1.25, 0.45),
    )
    discovered: dict[bytes, dict[str, Any]] = {}
    run_records: list[dict[str, Any]] = []
    zero_forbidden = np.zeros(node_count, dtype=bool)
    for profile_index, (
        name,
        unary_scale,
        edge_scale,
        coverage_scale,
    ) in enumerate(profiles):
        profile_node = node_score * unary_scale
        profile_edge = edge_weight * edge_scale
        profile_pixel = pixel_weight * coverage_scale
        rankings = (
            profile_node
            + 0.5 * incident * edge_scale
            + coverage_potential * coverage_scale,
            profile_node
            + incident * edge_scale
            + 0.5 * coverage_potential * coverage_scale,
            coverage_potential * coverage_scale
            + 0.25 * (profile_node + incident * edge_scale),
            profile_node
            + 0.25 * incident * edge_scale
            + 2.0 * coverage_potential * coverage_scale,
        )
        starts = [initial.copy()]
        for rank_score in rankings:
            selected = np.zeros(node_count, dtype=bool)
            order = np.lexsort((np.arange(node_count), -rank_score))
            for node_value in order:
                node = int(node_value)
                if rank_score[node] <= 0.0:
                    continue
                if len(conflicts[node]) and np.any(selected[conflicts[node]]):
                    continue
                selected[node] = True
            for node_value in np.flatnonzero(initial & ~selected):
                node = int(node_value)
                if len(conflicts[node]) and np.any(selected[conflicts[node]]):
                    continue
                selected[node] = True
            starts.append(selected)
        start_seen: set[bytes] = set()
        for start_index, start in enumerate(starts):
            start_key = start.tobytes()
            if start_key in start_seen:
                continue
            start_seen.add(start_key)
            initial_value = _coverage_objective(
                start,
                profile_node,
                edge_first,
                edge_second,
                profile_edge,
                coverage,
                profile_pixel,
            )
            selected, profile_value, sweeps = _optimize_coverage_from_start(
                profile_node,
                edge_first,
                edge_second,
                profile_edge,
                conflicts,
                coverage,
                profile_pixel,
                start,
                maximum_sweeps=maximum_sweeps,
                forbidden=zero_forbidden,
            )
            canonical_value = _coverage_objective(
                selected,
                node_score,
                edge_first,
                edge_second,
                edge_weight,
                coverage,
                pixel_weight,
            )
            key = selected.tobytes()
            record = discovered.get(key)
            if record is None or canonical_value > float(record["canonicalObjective"]):
                discovered[key] = {
                    "selected": selected.copy(),
                    "canonicalObjective": canonical_value,
                    "profile": name,
                    "profileIndex": profile_index,
                    "profileObjective": profile_value,
                }
            run_records.append(
                {
                    "profile": name,
                    "start": start_index,
                    "initialObjective": round(initial_value, 6),
                    "finalProfileObjective": round(profile_value, 6),
                    "canonicalObjective": round(canonical_value, 6),
                    "sweeps": sweeps,
                    "coveredPixelFraction": round(
                        float(np.mean(np.any(coverage[selected], axis=0)))
                        if coverage.shape[1]
                        else 0.0,
                        6,
                    ),
                }
            )

    binary_records: list[dict[str, Any]] = []
    binary_statistics: dict[str, Any] = {
        "requested": bool(use_exact_binary_optimizer),
        "eligible": bool(node_count <= maximum_exact_binary_nodes),
        "ran": False,
    }
    if use_exact_binary_optimizer and node_count <= maximum_exact_binary_nodes:
        binary_records, binary_details = (
            optimize_collective_patch_coverage_binary(
                node_score,
                edge_first,
                edge_second,
                edge_weight,
                conflict_first,
                conflict_second,
                coverage,
                pixel_weight,
                maximum_states=maximum_exact_binary_states,
                time_limit_seconds=exact_binary_time_limit_seconds,
            )
        )
        binary_statistics = {
            "requested": True,
            "eligible": True,
            "ran": True,
            **binary_details,
        }
        for record in binary_records:
            record["exactBinaryState"] = True
            selected = np.asarray(record["selected"], dtype=bool)
            key = selected.tobytes()
            prior = discovered.get(key)
            if prior is None or float(record["canonicalObjective"]) > float(
                prior["canonicalObjective"]
            ):
                discovered[key] = record
            else:
                prior["exactBinaryState"] = True
                prior["provenOptimal"] = bool(record["provenOptimal"])

    ordered = sorted(
        discovered.values(),
        key=lambda record: (
            float(record["canonicalObjective"]),
            int(np.count_nonzero(record["selected"])),
            -int(record["profileIndex"]),
        ),
        reverse=True,
    )
    if ordered and maximum_forced_exclusion_trials:
        branch_seed = np.asarray(ordered[0]["selected"], dtype=bool)
        branch = np.flatnonzero(branch_seed & ~initial)
        leverage = coverage_potential[branch] + incident[branch]
        branch_order = branch[np.lexsort((branch, -leverage))][
            :maximum_forced_exclusion_trials
        ]
        for node_value in branch_order:
            forbidden = np.zeros(node_count, dtype=bool)
            forbidden[int(node_value)] = True
            start = branch_seed.copy()
            start[int(node_value)] = False
            # Re-admit incumbents displaced only by the excluded choice, then
            # re-optimize the entire matching rather than filling its cell.
            for incumbent_value in np.flatnonzero(initial & ~start):
                incumbent = int(incumbent_value)
                if len(conflicts[incumbent]) and np.any(
                    start[conflicts[incumbent]]
                ):
                    continue
                start[incumbent] = True
            selected, value, sweeps = _optimize_coverage_from_start(
                node_score,
                edge_first,
                edge_second,
                edge_weight,
                conflicts,
                coverage,
                pixel_weight,
                start,
                maximum_sweeps=maximum_sweeps,
                forbidden=forbidden,
            )
            key = selected.tobytes()
            if key not in discovered:
                discovered[key] = {
                    "selected": selected.copy(),
                    "canonicalObjective": value,
                    "profile": "forced-exclusion",
                    "profileIndex": len(profiles),
                    "profileObjective": value,
                    "excludedLocalNode": int(node_value),
                }
            run_records.append(
                {
                    "profile": "forced-exclusion",
                    "excludedLocalNode": int(node_value),
                    "canonicalObjective": round(value, 6),
                    "sweeps": sweeps,
                    "coveredPixelFraction": round(
                        float(np.mean(np.any(coverage[selected], axis=0)))
                        if coverage.shape[1]
                        else 0.0,
                        6,
                    ),
                }
            )
        ordered = sorted(
            discovered.values(),
            key=lambda record: (
                float(record["canonicalObjective"]),
                int(np.count_nonzero(record["selected"])),
                -int(record["profileIndex"]),
            ),
            reverse=True,
        )

    mandatory_binary_keys = list(
        dict.fromkeys(
            np.asarray(record["selected"], dtype=bool).tobytes()
            for record in binary_records
        )
    )[:maximum_states]
    mandatory_binary = [
        discovered[key] for key in mandatory_binary_keys if key in discovered
    ]
    regular_budget = max(maximum_states - len(mandatory_binary), 0)
    retained: list[dict[str, Any]] = []
    for record in ordered:
        if len(retained) >= regular_budget:
            break
        selected = np.asarray(record["selected"], dtype=bool)
        if selected.tobytes() in mandatory_binary_keys:
            continue
        if retained and min(
            _normalized_hamming(selected, prior["selected"])
            for prior in retained
        ) < minimum_hamming_fraction:
            continue
        retained.append(record)
    retained.extend(mandatory_binary)
    retained.sort(
        key=lambda record: (
            float(record["canonicalObjective"]),
            int(np.count_nonzero(record["selected"])),
            -int(record["profileIndex"]),
        ),
        reverse=True,
    )
    if not retained:
        retained = [
            {
                "selected": initial.copy(),
                "canonicalObjective": _coverage_objective(
                    initial,
                    node_score,
                    edge_first,
                    edge_second,
                    edge_weight,
                    coverage,
                    pixel_weight,
                ),
                "profile": "incumbent",
                "profileIndex": -1,
                "profileObjective": 0.0,
            }
        ]
    state_records = []
    for rank, record in enumerate(retained):
        selected = np.asarray(record["selected"], dtype=bool)
        state_records.append(
            {
                "rank": rank,
                "profile": record["profile"],
                "canonicalObjective": round(float(record["canonicalObjective"]), 6),
                "selectedCount": int(np.count_nonzero(selected)),
                "coveredPixelFraction": round(
                    float(np.mean(np.any(coverage[selected], axis=0)))
                    if coverage.shape[1]
                    else 0.0,
                    6,
                ),
                "minimumPriorHammingFraction": round(
                    min(
                        (
                            _normalized_hamming(selected, prior["selected"])
                            for prior in retained[:rank]
                        ),
                        default=1.0,
                    ),
                    6,
                ),
                **(
                    {"excludedLocalNode": int(record["excludedLocalNode"])}
                    if "excludedLocalNode" in record
                    else {}
                ),
                **(
                    {
                        "exactBinaryState": True,
                        "provenOptimal": bool(record["provenOptimal"]),
                    }
                    if record.get("exactBinaryState")
                    else {}
                ),
            }
        )
    best = np.asarray(retained[0]["selected"], dtype=bool)
    return [np.asarray(record["selected"], dtype=bool) for record in retained], {
        "algorithm": (
            "diverse whole-patch matching ensemble with saturated dense CT "
            "coverage"
        ),
        "singleCellGrowth": False,
        "objectiveProfileCount": len(profiles),
        "optimizationRunCount": len(run_records),
        "discoveredStateCount": len(discovered),
        "retainedStateCount": len(retained),
        "pixelCount": int(coverage.shape[1]),
        "coveredPixelFraction": round(
            float(np.mean(np.any(coverage[best], axis=0)))
            if coverage.shape[1]
            else 0.0,
            6,
        ),
        "objective": round(float(retained[0]["canonicalObjective"]), 6),
        "states": state_records,
        "runs": run_records,
        "exactBinaryOptimization": binary_statistics,
    }


def _loop_vertices(holes: Mapping[str, np.ndarray], loop_index: int) -> np.ndarray:
    offset = np.asarray(holes["loopOffset"], dtype=np.int64)
    vertex = np.asarray(holes["loopVertexFrontierIndex"], dtype=np.int32)
    return vertex[int(offset[loop_index]) : int(offset[loop_index + 1])]


def _surface_view(holes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = (
        "selected",
        "component",
        "componentSize",
        "signedNormalXYZ",
        "tangentUxyz",
        "tangentVxyz",
        "chartUV",
        "integrationResidualVoxels",
        "edgeFirstFrontierIndex",
        "edgeSecondFrontierIndex",
        "edgeSelected",
        "triangleFrontierIndex",
        "triangleAreaVoxelsSquared",
        "triangleNormalResidualDegrees",
        "midpointXYZ",
        "thicknessVoxels",
    )
    return {name: np.asarray(holes[name]) for name in names}


def _loops_view(holes: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = (
        "loopOffset",
        "loopVertexFrontierIndex",
        "loopTriangleRegion",
        "loopTopologyComponent",
        "loopKind",
        "loopAreaChartVoxelsSquared",
        "loopPerimeterChartVoxels",
        "loopDiameterChartVoxels",
        "loopMedianThicknessVoxels",
        "loopMeanBoundaryEdgeVoxels",
        "loopMacroEligible",
    )
    return {name: np.asarray(holes[name]) for name in names}


def _component_loop_counts(
    loops: Mapping[str, np.ndarray],
) -> dict[int, dict[str, int]]:
    component = np.asarray(loops["loopTopologyComponent"], dtype=np.int32)
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    macro = np.asarray(loops["loopMacroEligible"], dtype=np.uint8) > 0
    result: dict[int, dict[str, int]] = {}
    for component_id in np.unique(component):
        member = component == component_id
        result[int(component_id)] = {
            "interiorHoleCount": int(np.count_nonzero(member & (kind == 1))),
            "macroHoleCount": int(np.count_nonzero(member & macro)),
        }
    return result


def _component_surface_metrics(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
) -> dict[int, dict[str, float | int]]:
    component = np.asarray(surface["component"], dtype=np.int32)
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    area = np.asarray(surface["triangleAreaVoxelsSquared"], dtype=np.float32)
    region_count = _triangle_region_counts(surface)
    loop_count = _component_loop_counts(loops)
    result: dict[int, dict[str, float | int]] = {}
    if len(triangle):
        triangle_component = component[triangle[:, 0]]
        for component_id in np.unique(triangle_component):
            member = triangle_component == component_id
            result[int(component_id)] = {
                "triangleCount": int(np.count_nonzero(member)),
                "triangleAreaVoxelsSquared": float(np.sum(area[member])),
                "triangleRegionCount": int(region_count.get(int(component_id), 0)),
                **loop_count.get(
                    int(component_id),
                    {"interiorHoleCount": 0, "macroHoleCount": 0},
                ),
            }
    for component_id, counts in loop_count.items():
        result.setdefault(
            component_id,
            {
                "triangleCount": 0,
                "triangleAreaVoxelsSquared": 0.0,
                "triangleRegionCount": int(region_count.get(component_id, 0)),
                **counts,
            },
        )
    return result


def _selection_conflicts(
    selected: np.ndarray,
    source: np.ndarray,
    target: np.ndarray,
    crossing_first: np.ndarray,
    crossing_second: np.ndarray,
) -> tuple[int, int]:
    node = np.flatnonzero(selected)
    endpoint = np.concatenate((source[node], target[node]))
    return (
        int(len(endpoint) - len(np.unique(endpoint))),
        int(np.count_nonzero(selected[crossing_first] & selected[crossing_second])),
    )


def _lineage_audit(
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    final_selected: np.ndarray,
    final_component: np.ndarray,
) -> tuple[dict[int, int], dict[str, int]]:
    mapping: dict[int, int] = {}
    split_count = 0
    deleted_count = 0
    for component_id in np.unique(baseline_component[baseline_selected]):
        inherited = (
            baseline_selected
            & final_selected
            & (baseline_component == component_id)
        )
        values = np.unique(final_component[inherited])
        values = values[values >= 0]
        if not len(values):
            deleted_count += 1
            continue
        if len(values) != 1:
            split_count += 1
            continue
        mapping[int(component_id)] = int(values[0])
    fusion_count = 0
    for component_id in np.unique(final_component[final_selected]):
        member = final_selected & (final_component == component_id)
        inherited = np.unique(
            baseline_component[member & baseline_selected]
        )
        inherited = inherited[inherited >= 0]
        fusion_count += int(len(inherited) > 1)
    return mapping, {
        "deletedPriorComponentCount": int(deleted_count),
        "splitPriorComponentCount": int(split_count),
        "crossPriorComponentFusionCount": int(fusion_count),
    }


def _local_edges(
    options: np.ndarray,
    offset: np.ndarray,
    neighbor: np.ndarray,
    weight: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, int]]:
    local = {int(value): index for index, value in enumerate(options)}
    first_values: list[int] = []
    second_values: list[int] = []
    weights: list[float] = []
    for first_local, first_global in enumerate(options):
        begin, end = int(offset[first_global]), int(offset[first_global + 1])
        for second_global, edge_weight in zip(
            neighbor[begin:end], weight[begin:end]
        ):
            second_local = local.get(int(second_global))
            if second_local is None or first_local >= second_local:
                continue
            first_values.append(first_local)
            second_values.append(second_local)
            weights.append(float(edge_weight))
    return (
        np.asarray(first_values, dtype=np.int32),
        np.asarray(second_values, dtype=np.int32),
        np.asarray(weights, dtype=np.float32),
        local,
    )


def _group_candidates(
    hole_rows: np.ndarray,
    holes: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, dict[int, dict[int, int]]]:
    offset = np.asarray(holes["patchCandidateOffset"], dtype=np.int64)
    candidate = np.asarray(holes["patchCandidateFrontierIndex"], dtype=np.int32)
    alignment = np.asarray(holes["patchCandidateSurfaceAlignment"], dtype=np.float32)
    nearest = np.asarray(holes["patchCandidateNearestPixel"], dtype=np.int32)
    score: dict[int, float] = {}
    row_nearest: dict[int, dict[int, int]] = {}
    for row_value in hole_rows:
        row = int(row_value)
        start, stop = int(offset[row]), int(offset[row + 1])
        mapping: dict[int, int] = {}
        for node, value, pixel in zip(
            candidate[start:stop], alignment[start:stop], nearest[start:stop]
        ):
            node_value = int(node)
            score[node_value] = max(score.get(node_value, 0.0), float(value))
            mapping[node_value] = int(pixel)
        row_nearest[row] = mapping
    nodes = np.asarray(sorted(score), dtype=np.int32)
    values = np.asarray([score[int(node)] for node in nodes], dtype=np.float32)
    return nodes, values, row_nearest


def _component_candidate_coverage(
    hole_rows: np.ndarray,
    options: np.ndarray,
    local: Mapping[int, int],
    row_nearest: Mapping[int, Mapping[int, int]],
    holes: Mapping[str, np.ndarray],
    depth_field: Mapping[str, np.ndarray],
    *,
    pixel_weight: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    patch_offset = np.asarray(holes["patchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(holes["patchXYZ"], dtype=np.float32)
    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    candidate_offset = np.asarray(holes["patchCandidateOffset"], dtype=np.int64)
    candidate_node = np.asarray(
        holes["patchCandidateFrontierIndex"], dtype=np.int32
    )
    depth_compatible = np.asarray(
        depth_field["candidateDepthCompatible"], dtype=np.uint8
    ) > 0
    ct_supported = np.asarray(depth_field["pixelCtSupported"], dtype=np.uint8) > 0
    total_pixels = int(
        sum(
            int(patch_offset[int(row) + 1] - patch_offset[int(row)])
            for row in hole_rows
        )
    )
    coverage = np.zeros((len(options), total_pixels), dtype=bool)
    weights = np.zeros(total_pixels, dtype=np.float64)
    pixel_cursor = 0
    compatible_candidate_count = 0
    for row_value in hole_rows:
        row = int(row_value)
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        local_patch = patch_xyz[start:stop]
        count = len(local_patch)
        local_supported = ct_supported[start:stop]
        weights[pixel_cursor : pixel_cursor + count] = (
            pixel_weight * local_supported.astype(np.float64)
        )
        candidate_start, candidate_stop = (
            int(candidate_offset[row]),
            int(candidate_offset[row + 1]),
        )
        compatible_by_node: dict[int, bool] = {}
        for node, compatible in zip(
            candidate_node[candidate_start:candidate_stop],
            depth_compatible[candidate_start:candidate_stop],
        ):
            node_value = int(node)
            compatible_by_node[node_value] = compatible_by_node.get(
                node_value, False
            ) or bool(compatible)
        loop = int(scored_loop[row])
        radius = float(holes["loopMeanBoundaryEdgeVoxels"][loop])
        for node, nearest in row_nearest[row].items():
            option = local.get(int(node))
            if option is None or not compatible_by_node.get(int(node), False):
                continue
            compatible_candidate_count += 1
            distance = np.linalg.norm(
                local_patch - local_patch[int(nearest)][None, :], axis=1
            )
            coverage[
                option, pixel_cursor : pixel_cursor + count
            ] |= (distance <= radius) & local_supported
        pixel_cursor += count
    coverable = np.any(coverage, axis=0) if total_pixels else np.empty(0, dtype=bool)
    return coverage, weights, {
        "patchPixelCount": int(total_pixels),
        "ctSupportedPixelCount": int(np.count_nonzero(weights)),
        "ctSupportedPixelFraction": round(
            float(np.count_nonzero(weights) / total_pixels)
            if total_pixels
            else 0.0,
            6,
        ),
        "depthCompatibleCandidateCount": int(compatible_candidate_count),
        "coverablePixelFraction": round(
            float(np.mean(coverable)) if len(coverable) else 0.0, 6
        ),
    }


def _proposal_hole_metrics(
    hole_rows: np.ndarray,
    row_nearest: Mapping[int, Mapping[int, int]],
    added: set[int],
    trial_selected: np.ndarray,
    holes: Mapping[str, np.ndarray],
    topology_offset: np.ndarray,
    topology_neighbor: np.ndarray,
) -> list[dict[str, Any]]:
    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    patch_offset = np.asarray(holes["patchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(holes["patchXYZ"], dtype=np.float32)
    mean_edge = np.asarray(holes["loopMeanBoundaryEdgeVoxels"], dtype=np.float32)
    selected_model = np.asarray(holes["selectedModel"], dtype=np.int32)
    correlation = np.asarray(
        holes["zeroShiftContextProfileCorrelation"], dtype=np.float32
    )
    margin = np.asarray(holes["zeroShiftCompetingMargin"], dtype=np.float32)
    records: list[dict[str, Any]] = []
    for row_value in hole_rows:
        row = int(row_value)
        loop = int(scored_loop[row])
        candidate_added = sorted(added.intersection(row_nearest[row]))
        if not candidate_added:
            continue
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        patch = patch_xyz[start:stop]
        projected = patch[
            np.asarray(
                [row_nearest[row][node] for node in candidate_added],
                dtype=np.int32,
            )
        ]
        coverage_radius = float(mean_edge[loop])
        covered = np.any(
            np.linalg.norm(patch[:, None, :] - projected[None, :, :], axis=2)
            <= coverage_radius,
            axis=1,
        )
        boundary = _loop_vertices(holes, loop)
        boundary_set = set(int(value) for value in boundary)
        anchors: set[int] = set()
        for node in candidate_added:
            begin, end = int(topology_offset[node]), int(topology_offset[node + 1])
            for neighbor in topology_neighbor[begin:end]:
                value = int(neighbor)
                if value in boundary_set and trial_selected[value]:
                    anchors.add(value)
        model = int(selected_model[row])
        records.append(
            {
                "holeRow": row,
                "loopIndex": loop,
                "addedRibbonCount": len(candidate_added),
                "coverage": float(np.mean(covered)) if len(covered) else 0.0,
                "retainedBoundaryFraction": float(np.mean(trial_selected[boundary])),
                "boundaryAnchorCount": len(anchors),
                "contextProfileCorrelation": float(correlation[row, model]),
                "competingLayerMargin": float(margin[row, model]),
            }
        )
    return records


def _added_geometry(
    added: np.ndarray,
    selected: np.ndarray,
    midpoint: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_weight: np.ndarray,
) -> tuple[float, float]:
    if not len(added):
        return 0.0, 0.0
    added_mask = np.zeros(len(selected), dtype=bool)
    added_mask[added] = True
    active = selected[edge_first] & selected[edge_second]
    incident = active & (added_mask[edge_first] | added_mask[edge_second])
    degree = np.zeros(len(selected), dtype=np.int32)
    np.add.at(degree, edge_first[incident], 1)
    np.add.at(degree, edge_second[incident], 1)
    two_core = float(np.mean(degree[added] >= 2))
    if np.count_nonzero(incident) < 2:
        return two_core, 0.0
    direction = midpoint[edge_second[incident]] - midpoint[edge_first[incident]]
    direction /= np.maximum(np.linalg.norm(direction, axis=1, keepdims=True), 1.0e-6)
    covariance = np.einsum(
        "i,ij,ik->jk", edge_weight[incident], direction, direction
    )
    eigenvalue = np.linalg.eigvalsh(covariance)
    tangent_ratio = float(eigenvalue[1] / max(float(eigenvalue[2]), 1.0e-6))
    return two_core, tangent_ratio


def _patch_state_groups(
    rows_by_component: Mapping[int, list[int]],
    holes: Mapping[str, np.ndarray],
    *,
    maximum_individual_hole_scopes: int,
) -> list[tuple[int, np.ndarray, str, int]]:
    """Declare complete component and individual-loop matching scopes.

    Component scope permits coupled re-pairing where nearby holes interact.
    Individual-loop scope freezes other hole candidates into the halo, which
    prevents a valid closure from inheriting topology debt created around a
    different loop on the same sheet.
    """

    patch_offset = np.asarray(holes["patchOffset"], dtype=np.int64)
    groups: list[tuple[int, np.ndarray, str, int]] = []
    for target_component, row_values in sorted(rows_by_component.items()):
        rows = np.asarray(row_values, dtype=np.int32)
        groups.append((int(target_component), rows, "component", -1))
        if len(rows) <= 1 or maximum_individual_hole_scopes == 0:
            continue
        ranked = sorted(
            (int(value) for value in rows),
            key=lambda row: (
                -int(patch_offset[row + 1] - patch_offset[row]),
                row,
            ),
        )[:maximum_individual_hole_scopes]
        groups.extend(
            (
                int(target_component),
                np.asarray((row,), dtype=np.int32),
                "hole",
                row,
            )
            for row in ranked
        )
    return groups


def _patch_state_proxy_screen(
    objective_gain: float,
    ct_supported_fraction: float,
    *,
    depth_field_present: bool,
    settings: PhysicalRibbonPatchStateSettings,
) -> tuple[bool, bool, bool]:
    """Route a state to exact screening without treating proxy sign as truth."""

    objective_qualified = objective_gain >= settings.minimum_objective_gain
    counterexample_qualified = bool(
        settings.screen_ct_supported_counterexamples
        and depth_field_present
        and ct_supported_fraction
        >= settings.minimum_counterexample_ct_supported_fraction
    )
    return (
        objective_qualified or counterexample_qualified,
        objective_qualified,
        counterexample_qualified and not objective_qualified,
    )


def solve_component_patch_states(
    holes: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    continuity_weight: float,
    surface_alignment_weight: float,
    settings: PhysicalRibbonPatchStateSettings,
    depth_field: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    component = np.asarray(configuration["component"], dtype=np.int32)
    unary = np.asarray(configuration["nodeUnaryScore"], dtype=np.float32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    midpoint = np.asarray(ribbon["midpointXYZ"], dtype=np.float32)[frontier]
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_score = np.asarray(topology["edgeScore"], dtype=np.float32)
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    topology_offset, topology_neighbor, topology_weight = _adjacency(
        len(frontier), edge_first, edge_second, edge_score
    )
    crossing_offset, crossing_neighbor, _ = _adjacency(
        len(frontier), crossing_first, crossing_second
    )
    interface_owner = np.full(
        len(np.asarray(ribbon["interfaceCandidateDegree"])), -1, dtype=np.int32
    )
    selected_node = np.flatnonzero(selected)
    interface_owner[source[selected_node]] = selected_node
    interface_owner[target[selected_node]] = selected_node

    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    loop_component = np.asarray(holes["loopTopologyComponent"], dtype=np.int32)
    rows_by_component: dict[int, list[int]] = defaultdict(list)
    for row, loop in enumerate(scored_loop):
        rows_by_component[int(loop_component[int(loop)])].append(row)

    patch_groups = _patch_state_groups(
        rows_by_component,
        holes,
        maximum_individual_hole_scopes=settings.maximum_individual_hole_scopes,
    )
    proposals: list[dict[str, Any]] = []
    proposal_signature_by_component: dict[
        int, set[tuple[bytes, bytes]]
    ] = defaultdict(set)
    rejected_external_candidate: set[int] = set()
    binary_scope_records: list[dict[str, Any]] = []
    for target_component, hole_rows, scope_kind, scope_hole_row in patch_groups:
        raw_candidate, raw_alignment, row_nearest = _group_candidates(
            hole_rows, holes
        )
        alignment_by_node = {
            int(node): float(value)
            for node, value in zip(raw_candidate, raw_alignment)
        }
        candidates: list[int] = []
        incumbents: set[int] = set()
        for node_value in raw_candidate:
            node = int(node_value)
            if selected[node]:
                continue
            blockers: set[int] = set()
            for owner in (interface_owner[source[node]], interface_owner[target[node]]):
                if owner >= 0:
                    blockers.add(int(owner))
            begin, end = int(crossing_offset[node]), int(crossing_offset[node + 1])
            blockers.update(
                int(value)
                for value in crossing_neighbor[begin:end]
                if selected[int(value)]
            )
            if any(component[value] != target_component for value in blockers):
                rejected_external_candidate.add(node)
                continue
            candidates.append(node)
            incumbents.update(blockers)
        candidate = np.asarray(sorted(set(candidates)), dtype=np.int32)
        incumbent = np.asarray(sorted(incumbents), dtype=np.int32)
        options = np.concatenate((incumbent, candidate))
        if not len(candidate) or not len(options):
            continue
        local_first, local_second, local_edge_score, local = _local_edges(
            options,
            topology_offset,
            topology_neighbor,
            topology_weight,
        )
        fixed_selected = selected.copy()
        fixed_selected[incumbent] = False
        fixed_support = np.zeros(len(options), dtype=np.float64)
        for option_index, node_value in enumerate(options):
            begin, end = (
                int(topology_offset[node_value]),
                int(topology_offset[node_value + 1]),
            )
            neighbor = topology_neighbor[begin:end]
            weight = topology_weight[begin:end]
            fixed_support[option_index] = float(
                np.sum(weight[fixed_selected[neighbor]])
            )
        node_score = unary[options].astype(np.float64)
        node_score += (
            continuity_weight
            * settings.continuity_weight_multiplier
            * fixed_support
        )
        for option_index, node_value in enumerate(options):
            node_score[option_index] += (
                surface_alignment_weight
                * settings.surface_alignment_weight_multiplier
                * alignment_by_node.get(int(node_value), 0.0)
            )
        local_edge_weight = (
            continuity_weight
            * settings.continuity_weight_multiplier
            * local_edge_score
        )
        shared_first, shared_second = _shared_endpoint_conflicts(
            source[options], target[options]
        )
        cross_local_first: list[int] = []
        cross_local_second: list[int] = []
        for first_local, first_global in enumerate(options):
            begin, end = (
                int(crossing_offset[first_global]),
                int(crossing_offset[first_global + 1]),
            )
            for second_global in crossing_neighbor[begin:end]:
                second_local = local.get(int(second_global))
                if second_local is not None and first_local < second_local:
                    cross_local_first.append(first_local)
                    cross_local_second.append(second_local)
        conflict_first = np.concatenate(
            (shared_first, np.asarray(cross_local_first, dtype=np.int32))
        )
        conflict_second = np.concatenate(
            (shared_second, np.asarray(cross_local_second, dtype=np.int32))
        )
        baseline_local = selected[options]
        baseline_objective = _objective(
            baseline_local,
            node_score,
            local_first,
            local_second,
            local_edge_weight,
        )
        coverage_stats: dict[str, Any] = {
            "patchPixelCount": 0,
            "ctSupportedPixelCount": 0,
            "ctSupportedPixelFraction": 0.0,
            "depthCompatibleCandidateCount": 0,
            "coverablePixelFraction": 0.0,
        }
        coverage_solver: dict[str, Any] | None = None
        coverage: np.ndarray | None = None
        pixel_weight: np.ndarray | None = None
        if depth_field is not None and settings.candidate_coverage_weight > 0.0:
            coverage, pixel_weight, coverage_stats = _component_candidate_coverage(
                hole_rows,
                options,
                local,
                row_nearest,
                holes,
                depth_field,
                pixel_weight=settings.candidate_coverage_weight,
            )
            optimized_states, coverage_solver = (
                optimize_collective_patch_coverage_ensemble(
                    node_score,
                    local_first,
                    local_second,
                    local_edge_weight,
                    conflict_first,
                    conflict_second,
                    coverage,
                    pixel_weight,
                    maximum_sweeps=settings.maximum_local_optimization_sweeps,
                    initial_selection=baseline_local,
                    maximum_states=settings.maximum_component_state_variants,
                    maximum_forced_exclusion_trials=(
                        settings.maximum_forced_exclusion_trials
                    ),
                    minimum_hamming_fraction=(
                        settings.minimum_variant_hamming_fraction
                    ),
                    use_exact_binary_optimizer=(
                        settings.use_exact_binary_optimizer
                    ),
                    maximum_exact_binary_nodes=(
                        settings.maximum_exact_binary_nodes
                    ),
                    maximum_exact_binary_states=(
                        settings.maximum_exact_binary_states
                    ),
                    exact_binary_time_limit_seconds=(
                        settings.exact_binary_time_limit_seconds
                    ),
                )
            )
            binary_scope_records.append(
                {
                    "targetComponent": int(target_component),
                    "scopeKind": scope_kind,
                    "scopeHoleRow": int(scope_hole_row),
                    **coverage_solver["exactBinaryOptimization"],
                }
            )
            baseline_objective = _coverage_objective(
                baseline_local,
                node_score,
                local_first,
                local_second,
                local_edge_weight,
                coverage,
                pixel_weight,
            )
        else:
            optimized, _ = optimize_collective_patch(
                node_score,
                local_first,
                local_second,
                local_edge_weight,
                conflict_first,
                conflict_second,
                maximum_sweeps=settings.maximum_local_optimization_sweeps,
                initial_selection=baseline_local,
            )
            optimized_states = [optimized]

        state_records = (
            coverage_solver.get("states", ()) if coverage_solver is not None else ()
        )
        for variant_rank, optimized in enumerate(optimized_states):
            # The optimizer may discover that an unrelated weak incumbent has
            # a negative marginal.  This is an alternating surface repair, not
            # a pruning pass: remove incumbents only when a chosen alternative
            # physically excludes them.
            chosen_added_local = optimized & ~baseline_local
            required_removed = np.zeros(len(options), dtype=bool)
            for first, second in zip(conflict_first, conflict_second):
                if chosen_added_local[first] and baseline_local[second]:
                    required_removed[second] = True
                if chosen_added_local[second] and baseline_local[first]:
                    required_removed[first] = True
            proposed = baseline_local.copy()
            proposed[required_removed] = False
            proposed[chosen_added_local] = True
            added = options[proposed & ~baseline_local]
            removed = options[baseline_local & ~proposed]
            signature = (added.tobytes(), removed.tobytes())
            if signature in proposal_signature_by_component[target_component]:
                continue
            proposal_signature_by_component[target_component].add(signature)
            if coverage is None or pixel_weight is None:
                proposal_objective = _objective(
                    proposed,
                    node_score,
                    local_first,
                    local_second,
                    local_edge_weight,
                )
            else:
                proposal_objective = _coverage_objective(
                    proposed,
                    node_score,
                    local_first,
                    local_second,
                    local_edge_weight,
                    coverage,
                    pixel_weight,
                )
            trial_selected = selected.copy()
            trial_selected[removed] = False
            trial_selected[added] = True
            two_core, tangent_ratio = _added_geometry(
                added,
                trial_selected,
                midpoint,
                edge_first,
                edge_second,
                edge_score,
            )
            hole_metrics = _proposal_hole_metrics(
                hole_rows,
                row_nearest,
                set(int(value) for value in added),
                trial_selected,
                holes,
                topology_offset,
                topology_neighbor,
            )
            objective_gain = float(proposal_objective - baseline_objective)
            (
                proxy_screened,
                objective_qualified,
                counterexample_qualified,
            ) = _patch_state_proxy_screen(
                objective_gain,
                float(coverage_stats["ctSupportedPixelFraction"]),
                depth_field_present=depth_field is not None,
                settings=settings,
            )
            physical_qualified = (
                len(added) >= settings.minimum_added_ribbon_count
                and two_core >= settings.minimum_added_two_core_fraction
                and tangent_ratio >= settings.minimum_added_tangent_ratio
                and bool(hole_metrics)
                and all(
                    record["contextProfileCorrelation"]
                    >= settings.minimum_context_profile_correlation
                    and record["competingLayerMargin"]
                    >= settings.minimum_competing_layer_margin
                    and record["coverage"] >= settings.minimum_patch_coverage
                    and record["retainedBoundaryFraction"]
                    >= settings.minimum_retained_boundary_fraction
                    and record["boundaryAnchorCount"]
                    >= settings.minimum_boundary_anchor_count
                    for record in hole_metrics
                )
            )
            qualified = physical_qualified and proxy_screened
            state_record = (
                state_records[variant_rank]
                if variant_rank < len(state_records)
                else {"profile": "single-objective"}
            )
            proposals.append(
                {
                    "targetComponent": int(target_component),
                    "scopeKind": scope_kind,
                    "scopeHoleRow": int(scope_hole_row),
                    "variantRank": int(variant_rank),
                    "variantProfile": str(state_record.get("profile", "unknown")),
                    "holeRows": hole_rows,
                    "options": options,
                    "candidateCount": int(len(candidate)),
                    "added": added.astype(np.int32),
                    "removed": removed.astype(np.int32),
                    "objectiveGain": objective_gain,
                    "twoCoreFraction": two_core,
                    "tangentRatio": tangent_ratio,
                    "holeMetrics": hole_metrics,
                    "coverageStats": coverage_stats,
                    "selectedCoverageFraction": float(
                        state_record.get("coveredPixelFraction", 0.0)
                    ),
                    "coverageSolver": coverage_solver,
                    "objectiveQualified": bool(objective_qualified),
                    "ctCounterexampleQualified": bool(
                        physical_qualified
                        and counterexample_qualified
                    ),
                    "qualified": bool(qualified),
                }
            )

    # Compose only the highest-ranked qualified variant per component here.
    # The runner reconstructs every rank in separate exact screening rounds.
    proposal_order: list[int] = []
    for target_component in sorted(rows_by_component):
        choices = [
            row
            for row, proposal in enumerate(proposals)
            if int(proposal["targetComponent"]) == target_component
            and bool(proposal["qualified"])
        ]
        if choices:
            proposal_order.append(
                min(choices, key=lambda row: int(proposals[row]["variantRank"]))
            )
    composed = selected.copy()
    applied = np.zeros(len(proposals), dtype=bool)
    for row in proposal_order:
        proposal = proposals[row]
        if not proposal["qualified"]:
            continue
        trial = composed.copy()
        trial[proposal["removed"]] = False
        trial[proposal["added"]] = True
        if _selection_conflicts(
            trial, source, target, crossing_first, crossing_second
        ) != (0, 0):
            continue
        composed = trial
        applied[row] = True
    for row, proposal in enumerate(proposals):
        proposal["composed"] = bool(applied[row])
    final_component, final_size = _component_labels(
        composed, edge_first, edge_second
    )
    arrays = {
        "selected": composed.astype(np.uint8),
        "component": final_component.astype(np.int32),
        "componentSize": final_size.astype(np.int32),
    }
    statistics = {
        "scoredHoleCount": int(len(scored_loop)),
        "surfaceComponentPatchCount": int(len(rows_by_component)),
        "patchStateScopeCount": int(len(patch_groups)),
        "individualHoleScopeCount": int(
            sum(scope_kind == "hole" for _, _, scope_kind, _ in patch_groups)
        ),
        "componentStateVariantCount": int(len(proposals)),
        "candidateRibbonCount": int(
            sum(
                max(
                    (
                        int(proposal["candidateCount"])
                        for proposal in proposals
                        if int(proposal["targetComponent"]) == target_component
                    ),
                    default=0,
                )
                for target_component in rows_by_component
            )
        ),
        "externallyBlockedCandidateCount": int(len(rejected_external_candidate)),
        "qualifiedPatchCount": int(
            sum(proposal["qualified"] for proposal in proposals)
        ),
        "ctCounterexampleQualifiedPatchCount": int(
            sum(
                proposal["ctCounterexampleQualified"]
                for proposal in proposals
            )
        ),
        "composedPatchCount": int(np.count_nonzero(applied)),
        "proposedAddedRibbonCount": int(
            sum(len(proposals[row]["added"]) for row in np.flatnonzero(applied))
        ),
        "proposedRemovedRibbonCount": int(
            sum(len(proposals[row]["removed"]) for row in np.flatnonzero(applied))
        ),
        "singleCellGrowth": False,
        "decisionUnit": (
            "all CT-supported closed-hole frontiers on one surface component"
        ),
        "fixedHalo": (
            "every selected ribbon not in an alternating interface or "
            "crossing conflict"
        ),
        "denseCoverageObjective": bool(
            depth_field is not None and settings.candidate_coverage_weight > 0.0
        ),
        "exactBinaryScopes": binary_scope_records,
    }
    return arrays, proposals, statistics


def _proposal_arrays(
    proposals: list[Mapping[str, Any]], accepted: np.ndarray
) -> dict[str, np.ndarray]:
    hole_offset = [0]
    hole_values: list[int] = []
    option_offset = [0]
    option_values: list[int] = []
    added_offset = [0]
    added_values: list[int] = []
    removed_offset = [0]
    removed_values: list[int] = []
    metric_offset = [0]
    metric_hole_row: list[int] = []
    metric_coverage: list[float] = []
    metric_retained: list[float] = []
    metric_anchor: list[int] = []
    metric_correlation: list[float] = []
    metric_margin: list[float] = []
    for proposal in proposals:
        hole_values.extend(int(value) for value in proposal["holeRows"])
        hole_offset.append(len(hole_values))
        option_values.extend(int(value) for value in proposal["options"])
        option_offset.append(len(option_values))
        added_values.extend(int(value) for value in proposal["added"])
        added_offset.append(len(added_values))
        removed_values.extend(int(value) for value in proposal["removed"])
        removed_offset.append(len(removed_values))
        for record in proposal["holeMetrics"]:
            metric_hole_row.append(int(record["holeRow"]))
            metric_coverage.append(float(record["coverage"]))
            metric_retained.append(float(record["retainedBoundaryFraction"]))
            metric_anchor.append(int(record["boundaryAnchorCount"]))
            metric_correlation.append(float(record["contextProfileCorrelation"]))
            metric_margin.append(float(record["competingLayerMargin"]))
        metric_offset.append(len(metric_hole_row))
    return {
        "patchTargetPriorComponent": np.asarray(
            [proposal["targetComponent"] for proposal in proposals], dtype=np.int32
        ),
        "patchScopeKind": np.asarray(
            [proposal.get("scopeKind", "component") for proposal in proposals],
            dtype="U16",
        ),
        "patchScopeHoleRow": np.asarray(
            [proposal.get("scopeHoleRow", -1) for proposal in proposals],
            dtype=np.int32,
        ),
        "patchVariantRank": np.asarray(
            [proposal.get("variantRank", 0) for proposal in proposals], dtype=np.int16
        ),
        "patchVariantProfile": np.asarray(
            [proposal.get("variantProfile", "unknown") for proposal in proposals],
            dtype="U32",
        ),
        "patchHoleRowOffset": np.asarray(hole_offset, dtype=np.int64),
        "patchHoleRow": np.asarray(hole_values, dtype=np.int32),
        "patchOptionOffset": np.asarray(option_offset, dtype=np.int64),
        "patchOptionFrontierIndex": np.asarray(option_values, dtype=np.int32),
        "patchCandidateCount": np.asarray(
            [proposal["candidateCount"] for proposal in proposals], dtype=np.int32
        ),
        "patchCtSupportedPixelCount": np.asarray(
            [
                proposal["coverageStats"]["ctSupportedPixelCount"]
                for proposal in proposals
            ],
            dtype=np.int32,
        ),
        "patchDepthCompatibleCandidateCount": np.asarray(
            [
                proposal["coverageStats"]["depthCompatibleCandidateCount"]
                for proposal in proposals
            ],
            dtype=np.int32,
        ),
        "patchCoverablePixelFraction": np.asarray(
            [
                proposal["coverageStats"]["coverablePixelFraction"]
                for proposal in proposals
            ],
            dtype=np.float32,
        ),
        "patchSelectedCoverageFraction": np.asarray(
            [
                proposal.get("selectedCoverageFraction", 0.0)
                for proposal in proposals
            ],
            dtype=np.float32,
        ),
        "patchAddedOffset": np.asarray(added_offset, dtype=np.int64),
        "patchAddedFrontierIndex": np.asarray(added_values, dtype=np.int32),
        "patchRemovedOffset": np.asarray(removed_offset, dtype=np.int64),
        "patchRemovedFrontierIndex": np.asarray(removed_values, dtype=np.int32),
        "patchObjectiveGain": np.asarray(
            [proposal["objectiveGain"] for proposal in proposals], dtype=np.float32
        ),
        "patchObjectiveQualified": np.asarray(
            [proposal.get("objectiveQualified", True) for proposal in proposals],
            dtype=np.uint8,
        ),
        "patchCtCounterexampleQualified": np.asarray(
            [
                proposal.get("ctCounterexampleQualified", False)
                for proposal in proposals
            ],
            dtype=np.uint8,
        ),
        "patchAddedTwoCoreFraction": np.asarray(
            [proposal["twoCoreFraction"] for proposal in proposals], dtype=np.float32
        ),
        "patchAddedTangentRatio": np.asarray(
            [proposal["tangentRatio"] for proposal in proposals], dtype=np.float32
        ),
        "patchQualified": np.asarray(
            [proposal["qualified"] for proposal in proposals], dtype=np.uint8
        ),
        "patchAccepted": np.asarray(accepted, dtype=np.uint8),
        "patchMetricOffset": np.asarray(metric_offset, dtype=np.int64),
        "patchMetricHoleRow": np.asarray(metric_hole_row, dtype=np.int32),
        "patchMetricCoverage": np.asarray(metric_coverage, dtype=np.float32),
        "patchMetricRetainedBoundaryFraction": np.asarray(
            metric_retained, dtype=np.float32
        ),
        "patchMetricBoundaryAnchorCount": np.asarray(metric_anchor, dtype=np.int32),
        "patchMetricContextProfileCorrelation": np.asarray(
            metric_correlation, dtype=np.float32
        ),
        "patchMetricCompetingLayerMargin": np.asarray(metric_margin, dtype=np.float32),
    }


def _exact_patch_audit(
    proposal: Mapping[str, Any],
    *,
    proposal_row: int,
    baseline_metrics: Mapping[int, Mapping[str, float | int]],
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    final_selected: np.ndarray,
    final_component: np.ndarray,
    mapping: Mapping[int, int],
    final_metrics: Mapping[int, Mapping[str, float | int]],
    triangle_node: np.ndarray,
    settings: PhysicalRibbonPatchStateSettings,
    round_index: int,
    lineage_preserved: bool | None = None,
) -> tuple[bool, dict[str, Any]]:
    target_component = int(proposal["targetComponent"])
    final_component_id = int(mapping.get(target_component, -1))
    before = dict(baseline_metrics.get(target_component, {}))
    after = dict(final_metrics.get(final_component_id, {}))
    added = np.asarray(proposal["added"], dtype=np.int32)
    realization = float(np.mean(triangle_node[added])) if len(added) else 0.0
    area_retention = float(after.get("triangleAreaVoxelsSquared", 0.0)) / max(
        float(before.get("triangleAreaVoxelsSquared", 0.0)), 1.0e-6
    )
    lineage_ok = final_component_id >= 0
    if lineage_preserved is not None:
        lineage_ok = bool(lineage_preserved)
    elif lineage_ok:
        member = final_selected & (final_component == final_component_id)
        inherited = np.unique(baseline_component[member & baseline_selected])
        inherited = inherited[inherited >= 0]
        lineage_ok = (
            len(inherited) == 1 and int(inherited[0]) == target_component
        )
    non_regression = (
        int(after.get("triangleRegionCount", 0))
        <= int(before.get("triangleRegionCount", 0))
        and int(after.get("interiorHoleCount", 0))
        <= int(before.get("interiorHoleCount", 0))
        and int(after.get("macroHoleCount", 0))
        <= int(before.get("macroHoleCount", 0))
        and area_retention >= settings.minimum_triangle_area_retention
    )
    topology_improved = (
        int(after.get("triangleRegionCount", 0))
        < int(before.get("triangleRegionCount", 0))
        or int(after.get("interiorHoleCount", 0))
        < int(before.get("interiorHoleCount", 0))
        or int(after.get("macroHoleCount", 0))
        < int(before.get("macroHoleCount", 0))
    )
    minimum_retained = min(
        (
            float(record["retainedBoundaryFraction"])
            for record in proposal["holeMetrics"]
        ),
        default=0.0,
    )
    minimum_anchor = min(
        (int(record["boundaryAnchorCount"]) for record in proposal["holeMetrics"]),
        default=0,
    )
    density_improved = (
        int(after.get("triangleCount", 0)) > int(before.get("triangleCount", 0))
        and minimum_retained
        >= settings.minimum_density_only_retained_boundary_fraction
        and minimum_anchor >= settings.minimum_density_only_boundary_anchor_count
    )
    improved = topology_improved or density_improved
    accepted = bool(
        lineage_ok
        and non_regression
        and improved
        and realization >= settings.minimum_surface_realization_fraction
    )
    return accepted, {
        "patchRow": int(proposal_row),
        "variantRank": int(proposal.get("variantRank", 0)),
        "variantProfile": str(proposal.get("variantProfile", "unknown")),
        "screenRound": int(round_index),
        "priorComponent": target_component,
        "finalComponent": final_component_id,
        "accepted": accepted,
        "lineagePreserved": bool(lineage_ok),
        "topologyNonRegression": bool(non_regression),
        "topologyImproved": bool(topology_improved),
        "densityImproved": bool(density_improved),
        "addedSurfaceRealizationFraction": realization,
        "triangleAreaRetention": area_retention,
        "before": before,
        "after": after,
    }


def _exact_variant_key(
    proposal: Mapping[str, Any], audit: Mapping[str, Any]
) -> tuple[float, ...]:
    before = audit["before"]
    after = audit["after"]
    return (
        float(
            int(before.get("macroHoleCount", 0))
            - int(after.get("macroHoleCount", 0))
        ),
        float(
            int(before.get("interiorHoleCount", 0))
            - int(after.get("interiorHoleCount", 0))
        ),
        float(
            int(before.get("triangleRegionCount", 0))
            - int(after.get("triangleRegionCount", 0))
        ),
        float(int(after.get("triangleCount", 0)) - int(before.get("triangleCount", 0))),
        float(audit.get("triangleAreaRetention", 0.0)),
        float(proposal.get("objectiveGain", 0.0)),
        -float(proposal.get("variantRank", 0)),
    )


def _compose_patch_rows(
    baseline_selected: np.ndarray,
    proposals: list[Mapping[str, Any]],
    rows: list[int],
    source: np.ndarray,
    target: np.ndarray,
    crossing_first: np.ndarray,
    crossing_second: np.ndarray,
) -> tuple[np.ndarray, list[int], list[int]]:
    selected = np.asarray(baseline_selected, dtype=bool).copy()
    applied: list[int] = []
    conflicted: list[int] = []
    for row in rows:
        proposal = proposals[row]
        trial = selected.copy()
        trial[np.asarray(proposal["removed"], dtype=np.int32)] = False
        trial[np.asarray(proposal["added"], dtype=np.int32)] = True
        if _selection_conflicts(
            trial, source, target, crossing_first, crossing_second
        ) != (0, 0):
            conflicted.append(int(row))
            continue
        selected = trial
        applied.append(int(row))
    return selected, applied, conflicted


@dataclass(frozen=True, slots=True)
class _ComponentExactGraph:
    target_component: int
    local_to_global: np.ndarray
    baseline_selected: np.ndarray
    edge_first: np.ndarray
    edge_second: np.ndarray
    edge_score: np.ndarray
    external_selected_neighbor: np.ndarray


def _prepare_component_exact_graph(
    target_component: int,
    proposals: list[Mapping[str, Any]],
    proposal_rows: list[int],
    baseline_selected: np.ndarray,
    baseline_component: np.ndarray,
    topology: Mapping[str, np.ndarray],
) -> _ComponentExactGraph:
    member = np.flatnonzero(
        np.asarray(baseline_selected, dtype=bool)
        & (np.asarray(baseline_component, dtype=np.int32) == target_component)
    )
    additions = [
        np.asarray(proposals[row]["added"], dtype=np.int32)
        for row in proposal_rows
        if len(proposals[row]["added"])
    ]
    local_to_global = np.unique(
        np.concatenate((member, *additions)) if additions else member
    ).astype(np.int32)
    global_to_local = np.full(len(baseline_selected), -1, dtype=np.int32)
    global_to_local[local_to_global] = np.arange(
        len(local_to_global), dtype=np.int32
    )
    first_global = np.asarray(
        topology["edgeFirstFrontierIndex"], dtype=np.int32
    )
    second_global = np.asarray(
        topology["edgeSecondFrontierIndex"], dtype=np.int32
    )
    first_local = global_to_local[first_global]
    second_local = global_to_local[second_global]
    induced = (first_local >= 0) & (second_local >= 0)
    external_selected_neighbor = np.zeros(len(local_to_global), dtype=bool)
    first_inside = (first_local >= 0) & (second_local < 0)
    if np.any(first_inside):
        external = np.asarray(baseline_selected, dtype=bool)[
            second_global[first_inside]
        ]
        external_selected_neighbor[first_local[first_inside][external]] = True
    second_inside = (second_local >= 0) & (first_local < 0)
    if np.any(second_inside):
        external = np.asarray(baseline_selected, dtype=bool)[
            first_global[second_inside]
        ]
        external_selected_neighbor[second_local[second_inside][external]] = True
    return _ComponentExactGraph(
        target_component=int(target_component),
        local_to_global=local_to_global,
        baseline_selected=np.asarray(baseline_selected, dtype=bool)[
            local_to_global
        ],
        edge_first=first_local[induced].astype(np.int32),
        edge_second=second_local[induced].astype(np.int32),
        edge_score=np.asarray(topology["edgeScore"], dtype=np.float32)[induced],
        external_selected_neighbor=external_selected_neighbor,
    )


def _reconstruct_component_graph_state(
    graph: _ComponentExactGraph,
    proposal: Mapping[str, Any],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonPatchHoleSettings,
) -> tuple[
    bool,
    dict[str, np.ndarray] | None,
    dict[str, np.ndarray] | None,
    np.ndarray,
]:
    local_selected = graph.baseline_selected.copy()
    global_to_local = {
        int(global_node): local_node
        for local_node, global_node in enumerate(graph.local_to_global)
    }
    for global_node in np.asarray(proposal["removed"], dtype=np.int32):
        local_selected[global_to_local[int(global_node)]] = False
    for global_node in np.asarray(proposal["added"], dtype=np.int32):
        local_selected[global_to_local[int(global_node)]] = True
    if np.any(local_selected & graph.external_selected_neighbor):
        return False, None, None, np.empty(0, dtype=np.int32)
    local_component, _ = _component_labels(
        local_selected, graph.edge_first, graph.edge_second
    )
    inherited = graph.baseline_selected & local_selected
    inherited_component = np.unique(local_component[inherited])
    inherited_component = inherited_component[inherited_component >= 0]
    if len(inherited_component) != 1:
        return False, None, None, np.empty(0, dtype=np.int32)
    target_local_component = int(inherited_component[0])
    added_local = np.asarray(
        [global_to_local[int(value)] for value in proposal["added"]],
        dtype=np.int32,
    )
    if len(added_local) and np.any(
        local_component[added_local] != target_local_component
    ):
        return False, None, None, np.empty(0, dtype=np.int32)
    component_node = np.flatnonzero(
        local_selected & (local_component == target_local_component)
    ).astype(np.int32)
    component_global = graph.local_to_global[component_node]
    component_remap = np.full(len(graph.local_to_global), -1, dtype=np.int32)
    component_remap[component_node] = np.arange(
        len(component_node), dtype=np.int32
    )
    first = component_remap[graph.edge_first]
    second = component_remap[graph.edge_second]
    active = (first >= 0) & (second >= 0)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    local_topology = {
        "frontierRibbonCandidate": frontier[component_global],
        "edgeFirstFrontierIndex": first[active],
        "edgeSecondFrontierIndex": second[active],
        "edgeScore": graph.edge_score[active],
    }
    local_configuration = {
        "selected": np.ones(len(component_global), dtype=np.uint8),
        "component": np.zeros(len(component_global), dtype=np.int32),
    }
    surface, _ = build_physical_ribbon_surface_complex(
        ribbon, local_topology, local_configuration, settings=settings
    )
    loops, _ = extract_surface_boundary_loops(surface, settings=settings)
    return True, surface, loops, component_global


def run_physical_ribbon_patch_states(
    holes_root: str | Path,
    output_root: str | Path,
    *,
    depth_field_root: str | Path | None = None,
    settings: PhysicalRibbonPatchStateSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonPatchStateSettings()
    holes_path, holes_manifest = _resolve_holes_manifest(holes_root)
    holes_data_path = holes_path.parent / str(holes_manifest["data"]["path"])
    holes = _load_npz(holes_data_path, holes_manifest["data"]["sha256"])
    depth_path: Path | None = None
    depth_manifest: dict[str, Any] | None = None
    depth_field: dict[str, np.ndarray] | None = None
    if depth_field_root is not None:
        depth_path, depth_manifest = _resolve_depth_field_manifest(depth_field_root)
        depth_data_path = depth_path.parent / str(depth_manifest["data"]["path"])
        depth_field = _load_npz(
            depth_data_path, depth_manifest["data"]["sha256"]
        )
        _validate_depth_field(
            holes_path,
            holes_manifest,
            holes,
            depth_path,
            depth_manifest,
            depth_field,
        )
    configuration_reference = holes_manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        configuration,
        topology_path,
        topology_manifest,
        topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path) != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("hole analysis configuration provenance changed")
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    surface_alignment_weight = float(
        holes_manifest.get("identity", {})
        .get("settings", {})
        .get("surface_alignment_weight", 0.35)
    )
    identity = {
        "schema": PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_STATE_VERSION,
        "holes": _reference(holes_path, holes_manifest),
        "configuration": _reference(configuration_path, configuration_manifest),
        "topologyContinuity": _reference(topology_path, topology_manifest),
        "ribbonBank": _reference(ribbon_path, ribbon_manifest),
        "continuityWeight": continuity_weight,
        "surfaceAlignmentWeight": surface_alignment_weight,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    if depth_field is not None and resolved.use_exact_binary_optimizer:
        identity["exactBinarySolver"] = {
            "name": "HiGHS",
            "version": _highs_version(required=True),
        }
    if depth_path is not None and depth_manifest is not None:
        identity["depthField"] = _reference(depth_path, depth_manifest)
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_PATCH_STATE_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_PATCH_STATE_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    baseline_surface = _surface_view(holes)
    baseline_loops = _loops_view(holes)
    baseline_metrics = _component_surface_metrics(baseline_surface, baseline_loops)
    arrays, proposals, solve_stats = solve_component_patch_states(
        holes,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        surface_alignment_weight=surface_alignment_weight,
        settings=resolved,
        depth_field=depth_field,
    )
    solved = time.monotonic()
    surface_settings = PhysicalRibbonPatchHoleSettings()
    baseline_selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    baseline_component = np.asarray(configuration["component"], dtype=np.int32)
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )

    def reconstruct(
        selected: np.ndarray,
    ) -> tuple[
        dict[str, np.ndarray],
        dict[str, np.ndarray],
        dict[str, np.ndarray],
    ]:
        component_value, size_value = _component_labels(
            selected, edge_first, edge_second
        )
        selection = {
            "selected": selected.astype(np.uint8),
            "component": component_value.astype(np.int32),
            "componentSize": size_value.astype(np.int32),
        }
        surface, _ = build_physical_ribbon_surface_complex(
            ribbon, topology, selection, settings=surface_settings
        )
        loops, _ = extract_surface_boundary_loops(
            surface, settings=surface_settings
        )
        return selection, surface, loops

    qualified_by_component: dict[int, list[int]] = defaultdict(list)
    for row, proposal in enumerate(proposals):
        if proposal["qualified"]:
            qualified_by_component[int(proposal["targetComponent"])].append(row)
    for rows in qualified_by_component.values():
        rows.sort(key=lambda row: int(proposals[row].get("variantRank", 0)))
    exact_graph_by_component = {
        target_component: _prepare_component_exact_graph(
            target_component,
            proposals,
            rows,
            baseline_selected,
            baseline_component,
            topology,
        )
        for target_component, rows in sorted(qualified_by_component.items())
    }
    exact_round_count = min(
        resolved.maximum_exact_state_rounds,
        max((len(rows) for rows in qualified_by_component.values()), default=0),
    )
    exact_records: list[dict[str, Any]] = []
    exact_record_by_row: dict[int, dict[str, Any]] = {}
    round_records: list[dict[str, Any]] = []
    for round_index in range(exact_round_count):
        requested_rows = [
            rows[round_index]
            for _, rows in sorted(qualified_by_component.items())
            if round_index < len(rows)
        ]
        applied_rows: list[int] = []
        conflicted_rows: list[int] = []
        round_accepted: list[int] = []
        for row in requested_rows:
            round_selected, applied, conflicted = _compose_patch_rows(
                baseline_selected,
                proposals,
                [row],
                source,
                target,
                crossing_first,
                crossing_second,
            )
            if conflicted or not applied:
                conflicted_rows.append(row)
                continue
            applied_rows.append(row)
            target_component = int(proposals[row]["targetComponent"])
            (
                lineage_ok,
                local_surface,
                local_loops,
                local_to_global,
            ) = _reconstruct_component_graph_state(
                exact_graph_by_component[target_component],
                proposals[row],
                ribbon,
                topology,
                settings=surface_settings,
            )
            final_component_id = target_component if lineage_ok else -1
            mapping = (
                {target_component: final_component_id} if lineage_ok else {}
            )
            round_metrics: dict[int, dict[str, float | int]] = {}
            triangle_node = np.zeros(len(round_selected), dtype=bool)
            if lineage_ok and local_surface is not None and local_loops is not None:
                local_metrics = _component_surface_metrics(
                    local_surface, local_loops
                )
                round_metrics[final_component_id] = dict(
                    local_metrics.get(0, {})
                )
                triangle = np.asarray(
                    local_surface["triangleFrontierIndex"], dtype=np.int32
                )
                if len(triangle):
                    triangle_node[
                        local_to_global[np.unique(triangle)]
                    ] = True
            is_valid, record = _exact_patch_audit(
                proposals[row],
                proposal_row=row,
                baseline_metrics=baseline_metrics,
                baseline_selected=baseline_selected,
                baseline_component=baseline_component,
                final_selected=round_selected,
                final_component=baseline_component,
                mapping=mapping,
                final_metrics=round_metrics,
                triangle_node=triangle_node,
                settings=resolved,
                round_index=round_index,
                lineage_preserved=lineage_ok,
            )
            record["reconstructionScope"] = "affected strict component"
            exact_records.append(record)
            exact_record_by_row[row] = record
            if is_valid:
                round_accepted.append(row)
        round_records.append(
            {
                "round": round_index,
                "requestedPatchRows": requested_rows,
                "appliedPatchRows": applied_rows,
                "conflictedPatchRows": conflicted_rows,
                "exactAcceptedPatchRows": round_accepted,
            }
        )
    surfaced = time.monotonic()

    chosen_rows: list[int] = []
    for target_component, rows in sorted(qualified_by_component.items()):
        valid = [
            row
            for row in rows
            if row in exact_record_by_row and exact_record_by_row[row]["accepted"]
        ]
        if valid:
            chosen_rows.append(
                max(
                    valid,
                    key=lambda row: _exact_variant_key(
                        proposals[row], exact_record_by_row[row]
                    ),
                )
            )
    chosen_rows.sort(
        key=lambda row: _exact_variant_key(
            proposals[row], exact_record_by_row[row]
        ),
        reverse=True,
    )

    accepted = np.zeros(len(proposals), dtype=bool)
    final_audits: list[dict[str, Any]] = []
    final_conflicted_rows: list[int] = []
    while True:
        final_selected, applied_rows, conflicted_rows = _compose_patch_rows(
            baseline_selected,
            proposals,
            chosen_rows,
            source,
            target,
            crossing_first,
            crossing_second,
        )
        final_conflicted_rows.extend(conflicted_rows)
        arrays, final_surface, final_loops = reconstruct(final_selected)
        final_component = np.asarray(arrays["component"], dtype=np.int32)
        mapping, _ = _lineage_audit(
            baseline_selected,
            baseline_component,
            final_selected,
            final_component,
        )
        final_metrics = _component_surface_metrics(final_surface, final_loops)
        triangle = np.asarray(final_surface["triangleFrontierIndex"], dtype=np.int32)
        triangle_node = np.zeros(len(final_selected), dtype=bool)
        if len(triangle):
            triangle_node[np.unique(triangle)] = True
        final_audits = []
        failed_rows: list[int] = []
        for row in applied_rows:
            is_valid, record = _exact_patch_audit(
                proposals[row],
                proposal_row=row,
                baseline_metrics=baseline_metrics,
                baseline_selected=baseline_selected,
                baseline_component=baseline_component,
                final_selected=final_selected,
                final_component=final_component,
                mapping=mapping,
                final_metrics=final_metrics,
                triangle_node=triangle_node,
                settings=resolved,
                round_index=-1,
            )
            final_audits.append(record)
            if not is_valid:
                failed_rows.append(row)
        if not failed_rows:
            accepted[applied_rows] = True
            break
        chosen_rows = [row for row in applied_rows if row not in failed_rows]
    finished_surface = time.monotonic()

    conflict_count = _selection_conflicts(
        final_selected, source, target, crossing_first, crossing_second
    )
    _, final_lineage = _lineage_audit(
        baseline_selected,
        baseline_component,
        final_selected,
        final_component,
    )
    if conflict_count != (0, 0) or any(final_lineage.values()):
        raise RuntimeError("exact patch-state replay violated a hard invariant")
    final_loop_stats = {
        "triangleRegionCount": int(
            sum(_triangle_region_counts(final_surface).values())
        ),
        "interiorHoleLoopCount": int(
            np.count_nonzero(np.asarray(final_loops["loopKind"]) == 1)
        ),
        "macroEligibleHoleCount": int(
            np.count_nonzero(np.asarray(final_loops["loopMacroEligible"]) > 0)
        ),
    }
    baseline_loop_stats = holes_manifest["loops"]
    output_arrays = {
        **_proposal_arrays(proposals, accepted),
        "selected": arrays["selected"],
        "component": arrays["component"],
        "componentSize": final_surface["componentSize"],
        "signedNormalXYZ": final_surface["signedNormalXYZ"],
        "tangentUxyz": final_surface["tangentUxyz"],
        "tangentVxyz": final_surface["tangentVxyz"],
        "chartUV": final_surface["chartUV"],
        "integrationResidualVoxels": final_surface["integrationResidualVoxels"],
        "triangleFrontierIndex": final_surface["triangleFrontierIndex"],
        "triangleAreaVoxelsSquared": final_surface["triangleAreaVoxelsSquared"],
        "triangleNormalResidualDegrees": final_surface[
            "triangleNormalResidualDegrees"
        ],
        "midpointXYZ": final_surface["midpointXYZ"],
        "thicknessVoxels": final_surface["thicknessVoxels"],
    }
    _write_npz(data_path, output_arrays)
    view = {**topology, **arrays}
    world = configuration_manifest["geometry"]["ownedWorldBounds"]
    overview = write_continuity_overview(
        ribbon,
        view,
        np.asarray(world["startXYZ"], dtype=np.float32),
        np.asarray(world["stopXYZExclusive"], dtype=np.float32),
        output / "patch-state-ribbon-components.png",
        maximum_components=resolved.maximum_preview_components,
    )
    montage = write_largest_component_montage(
        ribbon,
        view,
        output / "largest-patch-state-ribbon-components.png",
    )
    finished = time.monotonic()
    before_triangle = len(np.asarray(baseline_surface["triangleFrontierIndex"]))
    after_triangle = len(np.asarray(final_surface["triangleFrontierIndex"]))
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
        "version": PHYSICAL_RIBBON_PATCH_STATE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": configuration_manifest["geometry"],
        "patchStates": {
            **solve_stats,
            "acceptedPatchCount": int(np.count_nonzero(accepted)),
            "acceptedAddedRibbonCount": int(
                sum(len(proposals[row]["added"]) for row in np.flatnonzero(accepted))
            ),
            "acceptedRemovedRibbonCount": int(
                sum(len(proposals[row]["removed"]) for row in np.flatnonzero(accepted))
            ),
            "selectedRibbonCountBefore": int(np.count_nonzero(baseline_selected)),
            "selectedRibbonCountAfter": int(np.count_nonzero(final_selected)),
            "interfaceConflictCount": conflict_count[0],
            "crossingConflictCount": conflict_count[1],
            **final_lineage,
            "exactScreenRoundCount": int(exact_round_count),
            "exactScreenedVariantCount": int(len(exact_records)),
            "exactValidVariantCount": int(
                sum(bool(record["accepted"]) for record in exact_records)
            ),
            "exactScreenRounds": round_records,
            "exactPatchAudits": exact_records,
            "finalExactPatchAudits": final_audits,
            "finalConflictedPatchRows": sorted(set(final_conflicted_rows)),
        },
        "exactTopology": {
            "strictTriangleCountBefore": int(before_triangle),
            "strictTriangleCountAfter": int(after_triangle),
            "strictTriangleCountDelta": int(after_triangle - before_triangle),
            "triangleRegionCountBefore": int(
                baseline_loop_stats["triangleRegionCount"]
            ),
            "triangleRegionCountAfter": final_loop_stats["triangleRegionCount"],
            "triangleRegionCountDelta": int(
                final_loop_stats["triangleRegionCount"]
                - baseline_loop_stats["triangleRegionCount"]
            ),
            "interiorHoleLoopCountBefore": int(
                baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "interiorHoleLoopCountAfter": final_loop_stats[
                "interiorHoleLoopCount"
            ],
            "interiorHoleLoopCountDelta": int(
                final_loop_stats["interiorHoleLoopCount"]
                - baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "macroEligibleHoleCountBefore": int(
                baseline_loop_stats["macroEligibleHoleCount"]
            ),
            "macroEligibleHoleCountAfter": final_loop_stats[
                "macroEligibleHoleCount"
            ],
            "macroEligibleHoleCountDelta": int(
                final_loop_stats["macroEligibleHoleCount"]
                - baseline_loop_stats["macroEligibleHoleCount"]
            ),
        },
        "timingSeconds": {
            "collectivePatchOptimization": round(solved - started, 6),
            "componentLocalExactScreening": round(surfaced - solved, 6),
            "finalExactSurface": round(finished_surface - surfaced, 6),
            "writingAndPreviews": round(finished - finished_surface, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(output_arrays),
        },
        "artifacts": {
            "componentOverview": overview.name,
            "largestComponentMontage": montage.name,
        },
        "method": {
            "decisionUnit": (
                "all CT-supported hole frontiers on one reconstructed "
                "surface component"
            ),
            "optimization": (
                "diverse collective alternating-interface matchings with "
                "saturated whole-patch dense CT coverage plus bounded binary "
                "whole-scope optimization inside a fixed selected halo"
                if depth_field is not None
                else (
                    "collective alternating interface matching inside a fixed "
                    "selected halo"
                )
            ),
            "physicalEvidence": (
                "whole-patch native-CT air-material-air profile and "
                "displaced-layer competition"
            ),
            "exactAcceptance": (
                "multiple complete states are reconstructed on their exact "
                "strict component graph; a final whole-volume rebuild proves "
                "surface realization, lineage, triangle density, connected "
                "regions, and closed-hole results"
            ),
            "singleCellGrowth": False,
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
