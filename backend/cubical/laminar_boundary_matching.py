from __future__ import annotations

import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import _percentile_record
from .material_surface_graph import (
    MATERIAL_SURFACE_GRAPH_SCHEMA,
    MaterialSurfaceGraphSettings,
    write_material_surface_cross_sections,
)
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
    PHYSICAL_MID_SURFACE_VERSION,
    _components,
    _write_npz,
)


LAMINAR_BOUNDARY_MATCHING_SCHEMA = "pareidolia.laminar-boundary-matching"
LAMINAR_BOUNDARY_MATCHING_VERSION = 1


@dataclass(frozen=True, slots=True)
class LaminarBoundaryMatchingSettings:
    """Global selection controls for local two-boundary correspondences.

    Every candidate pairs two observed material/air boundary surfels.  The
    optimizer may use each physical surfel at most once and receives a reward
    only when *both* boundaries of neighboring candidates continue together.
    This makes page identity a property of a locally coherent ribbon rather
    than an inherited component label.
    """

    maximum_continuity_normal_degrees: float = 25.0
    maximum_continuity_height_sampling_steps: float = 1.0
    maximum_continuity_distance_sampling_steps: float = 3.0
    maximum_local_thickness_change_sampling_steps: float = 1.5
    enable_one_face_geometric_closure: bool = True
    maximum_missing_face_gap_sampling_steps: float = 2.5
    maximum_missing_face_signed_normal_degrees: float = 25.0
    one_face_closure_affinity_scale: float = 0.5
    missing_face_gap_scale_sampling_steps: float = 2.0
    missing_face_normal_scale_degrees: float = 20.0
    candidate_base_reward: float = 1.0
    local_evidence_reward: float = 0.35
    pair_cost_reward: float = 0.35
    pair_cost_scale: float = 2.0
    thickness_agreement_reward: float = 0.2
    thickness_residual_scale_sampling_steps: float = 1.0
    continuity_reward: float = 1.5
    continuity_normal_scale_degrees: float = 20.0
    continuity_height_scale_sampling_steps: float = 0.75
    mip_relative_gap: float = 0.001
    maximum_solver_seconds: float = 120.0
    maximum_component_solver_seconds: float = 15.0
    solver_random_seed: int = 0

    def __post_init__(self) -> None:
        if not 0.0 < self.maximum_continuity_normal_degrees < 89.0:
            raise ValueError("continuity normal gate must lie in (0, 89) degrees")
        positive = (
            self.maximum_continuity_height_sampling_steps,
            self.maximum_continuity_distance_sampling_steps,
            self.maximum_local_thickness_change_sampling_steps,
            self.maximum_missing_face_gap_sampling_steps,
            self.missing_face_gap_scale_sampling_steps,
            self.missing_face_normal_scale_degrees,
            self.candidate_base_reward,
            self.pair_cost_scale,
            self.thickness_residual_scale_sampling_steps,
            self.continuity_normal_scale_degrees,
            self.continuity_height_scale_sampling_steps,
            self.maximum_solver_seconds,
            self.maximum_component_solver_seconds,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("boundary matching scales must be finite and positive")
        nonnegative = (
            self.local_evidence_reward,
            self.pair_cost_reward,
            self.thickness_agreement_reward,
            self.continuity_reward,
            self.one_face_closure_affinity_scale,
            self.mip_relative_gap,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in nonnegative):
            raise ValueError("boundary matching rewards must be finite and nonnegative")
        if self.mip_relative_gap >= 1.0:
            raise ValueError("MIP relative gap must be smaller than one")
        if not 0.0 < self.maximum_missing_face_signed_normal_degrees < 89.0:
            raise ValueError(
                "missing-face signed-normal gate must lie in (0, 89) degrees"
            )
        if self.solver_random_seed < 0:
            raise ValueError("solver random seed must be nonnegative")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_mid_surface(root: str | Path) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{PHYSICAL_MID_SURFACE_STEM}.json"


def _load(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(path.read_text())
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"artifact data changed after its manifest: {path}")
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return manifest, arrays


def _candidate_rewards(
    source: Mapping[str, np.ndarray],
    dense_node: np.ndarray,
    *,
    sampling_stride_voxels: float,
    settings: LaminarBoundaryMatchingSettings,
) -> np.ndarray:
    pair_cost = np.asarray(source["pairCost"], dtype=np.float64)[dense_node]
    lower_evidence = np.asarray(
        source["lowerLocalEvidenceScore"], dtype=np.float64
    )[dense_node]
    upper_evidence = np.asarray(
        source["upperLocalEvidenceScore"], dtype=np.float64
    )[dense_node]
    thickness_residual = np.abs(
        np.asarray(source["thicknessResidualVoxels"], dtype=np.float64)[dense_node]
    ) / float(sampling_stride_voxels)
    pair_quality = np.exp(
        -np.maximum(pair_cost, 0.0) / settings.pair_cost_scale
    )
    thickness_quality = np.exp(
        -thickness_residual / settings.thickness_residual_scale_sampling_steps
    )
    evidence = np.clip(0.5 * (lower_evidence + upper_evidence), 0.0, 1.0)
    reward = (
        settings.candidate_base_reward
        + settings.local_evidence_reward * evidence
        + settings.pair_cost_reward * pair_quality
        + settings.thickness_agreement_reward * thickness_quality
    )
    if np.any(~np.isfinite(reward)) or np.any(reward <= 0.0):
        raise ValueError("local boundary candidate rewards must be finite and positive")
    return reward


def compile_laminar_boundary_problem(
    source: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    *,
    sampling_stride_voxels: float,
    settings: LaminarBoundaryMatchingSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compile hard mate conflicts and exact two-face continuation rewards."""

    node_kind = np.asarray(source["nodeKind"], dtype=np.uint8)
    dense_node = np.flatnonzero(node_kind == 1).astype(np.int32)
    if not len(dense_node):
        raise ValueError("boundary matching requires dense paired boundary nodes")
    lower = np.asarray(source["lowerSurfaceNode"], dtype=np.int64)[dense_node]
    upper = np.asarray(source["upperSurfaceNode"], dtype=np.int64)[dense_node]
    surface_node_count = len(surface["positionXYZ"])
    if (
        np.any(lower < 0)
        or np.any(upper < 0)
        or np.any(lower >= surface_node_count)
        or np.any(upper >= surface_node_count)
    ):
        raise ValueError("dense boundary candidates leave the material surface table")
    if np.any(lower == upper):
        raise ValueError("a physical thickness candidate cannot mate a surfel to itself")

    first = np.minimum(lower, upper)
    second = np.maximum(lower, upper)
    raw_reward = _candidate_rewards(
        source,
        dense_node,
        sampling_stride_voxels=sampling_stride_voxels,
        settings=settings,
    )
    order = np.lexsort((dense_node, -raw_reward, second, first))
    ordered_first = first[order]
    ordered_second = second[order]
    unique = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        unique[1:] = (ordered_first[1:] != ordered_first[:-1]) | (
            ordered_second[1:] != ordered_second[:-1]
        )
    retained = order[unique]
    source_node = dense_node[retained]
    boundary = np.stack((first[retained], second[retained]), axis=1).astype(
        np.int32
    )
    candidate_reward = raw_reward[retained].astype(np.float32)

    incident: dict[int, list[tuple[int, int]]] = defaultdict(list)
    for candidate, (left, right) in enumerate(boundary):
        incident[int(left)].append((candidate, 0))
        incident[int(right)].append((candidate, 1))

    # For a pair of midpoint candidates, the four bits encode which endpoint
    # pairs are linked by the original dense surface graph.  A legitimate
    # continuation requires either (0-0 and 1-1) or (0-1 and 1-0): both
    # physical faces must continue, not merely one convenient edge.
    endpoint_mask: dict[tuple[int, int], int] = defaultdict(int)
    endpoint_score: dict[tuple[int, int], np.ndarray] = defaultdict(
        lambda: np.zeros(4, dtype=np.float64)
    )
    surface_first = np.asarray(surface["edgeFirstNode"], dtype=np.int64)
    surface_second = np.asarray(surface["edgeSecondNode"], dtype=np.int64)
    surface_score = np.asarray(surface["edgeScore"], dtype=np.float64)
    for left, right, score in zip(surface_first, surface_second, surface_score):
        left_candidate = incident.get(int(left))
        right_candidate = incident.get(int(right))
        if not left_candidate or not right_candidate:
            continue
        for first_candidate, first_endpoint in left_candidate:
            for second_candidate, second_endpoint in right_candidate:
                if first_candidate == second_candidate:
                    continue
                if first_candidate < second_candidate:
                    pair = (first_candidate, second_candidate)
                    bit = first_endpoint * 2 + second_endpoint
                else:
                    pair = (second_candidate, first_candidate)
                    bit = second_endpoint * 2 + first_endpoint
                endpoint_mask[pair] |= 1 << bit
                endpoint_score[pair][bit] = max(
                    endpoint_score[pair][bit], float(score)
                )

    midpoint = np.asarray(source["midpointXYZ"], dtype=np.float64)[source_node]
    normal = np.asarray(source["normalXYZ"], dtype=np.float64)[source_node]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-12)
    thickness = np.asarray(source["thicknessVoxels"], dtype=np.float64)[source_node]
    surface_position = np.asarray(surface["positionXYZ"], dtype=np.float64)
    surface_normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float64)
    surface_normal /= np.maximum(
        np.linalg.norm(surface_normal, axis=1, keepdims=True), 1.0e-12
    )
    continuity_first: list[int] = []
    continuity_second: list[int] = []
    continuity_affinity: list[float] = []
    continuity_kind: list[int] = []
    continuity_normal: list[float] = []
    continuity_height: list[float] = []
    continuity_distance: list[float] = []
    continuity_thickness_change: list[float] = []
    continuity_missing_face_gap: list[float] = []
    continuity_missing_face_normal: list[float] = []
    stride = float(sampling_stride_voxels)
    for pair, mask in endpoint_mask.items():
        parallel = bool(mask & (1 << 0)) and bool(mask & (1 << 3))
        crossed = bool(mask & (1 << 1)) and bool(mask & (1 << 2))
        left, right = pair
        # Conflicting candidates can never be jointly selected.  Dropping
        # their pairwise reward shrinks the MIP without changing its optimum.
        if len(set(map(int, boundary[left])) & set(map(int, boundary[right]))):
            continue
        cosine = float(np.dot(normal[left], normal[right]))
        angle = float(
            np.degrees(np.arccos(np.clip(abs(cosine), 0.0, 1.0)))
        )
        aligned_right = normal[right] * (1.0 if cosine >= 0.0 else -1.0)
        average_normal = normal[left] + aligned_right
        average_normal /= max(float(np.linalg.norm(average_normal)), 1.0e-12)
        delta = midpoint[right] - midpoint[left]
        height = abs(float(np.dot(delta, average_normal))) / stride
        distance = float(np.linalg.norm(delta)) / stride
        thickness_change = abs(float(thickness[right] - thickness[left])) / stride
        if (
            angle > settings.maximum_continuity_normal_degrees
            or height > settings.maximum_continuity_height_sampling_steps
            or distance > settings.maximum_continuity_distance_sampling_steps
            or thickness_change
            > settings.maximum_local_thickness_change_sampling_steps
        ):
            continue
        score = endpoint_score[pair]
        missing_face_gap = 0.0
        missing_face_normal = 0.0
        if parallel or crossed:
            alternative_score: list[float] = []
            if parallel:
                alternative_score.append(math.sqrt(score[0] * score[3]))
            if crossed:
                alternative_score.append(math.sqrt(score[1] * score[2]))
            base_affinity = max(alternative_score)
            kind = 0
        else:
            if not settings.enable_one_face_geometric_closure:
                continue
            closure_alternative: list[tuple[float, float, float]] = []
            for bit in range(4):
                if not mask & (1 << bit):
                    continue
                left_endpoint, right_endpoint = divmod(bit, 2)
                missing_left = 1 - left_endpoint
                missing_right = 1 - right_endpoint
                missing_left_node = int(boundary[left, missing_left])
                missing_right_node = int(boundary[right, missing_right])
                gap = float(
                    np.linalg.norm(
                        surface_position[missing_right_node]
                        - surface_position[missing_left_node]
                    )
                    / stride
                )
                signed_cosine = float(
                    np.dot(
                        surface_normal[missing_left_node],
                        surface_normal[missing_right_node],
                    )
                )
                signed_angle = float(
                    np.degrees(
                        np.arccos(np.clip(signed_cosine, -1.0, 1.0))
                    )
                )
                if (
                    gap > settings.maximum_missing_face_gap_sampling_steps
                    or signed_angle
                    > settings.maximum_missing_face_signed_normal_degrees
                ):
                    continue
                closure_score = float(score[bit]) * math.exp(
                    -0.5
                    * (gap / settings.missing_face_gap_scale_sampling_steps) ** 2
                    -0.5
                    * (signed_angle / settings.missing_face_normal_scale_degrees)
                    ** 2
                )
                closure_alternative.append((closure_score, gap, signed_angle))
            if not closure_alternative:
                continue
            base_affinity, missing_face_gap, missing_face_normal = max(
                closure_alternative, key=lambda value: value[0]
            )
            base_affinity *= settings.one_face_closure_affinity_scale
            kind = 1
        affinity = base_affinity * math.exp(
            -0.5 * (angle / settings.continuity_normal_scale_degrees) ** 2
            -0.5
            * (height / settings.continuity_height_scale_sampling_steps) ** 2
        )
        continuity_first.append(left)
        continuity_second.append(right)
        continuity_affinity.append(affinity)
        continuity_kind.append(kind)
        continuity_normal.append(angle)
        continuity_height.append(height)
        continuity_distance.append(distance)
        continuity_thickness_change.append(thickness_change)
        continuity_missing_face_gap.append(missing_face_gap)
        continuity_missing_face_normal.append(missing_face_normal)

    arrays = {
        "sourceMidSurfaceNode": source_node.astype(np.int32),
        "boundaryFirstSurfaceNode": boundary[:, 0].astype(np.int32),
        "boundarySecondSurfaceNode": boundary[:, 1].astype(np.int32),
        "candidateReward": candidate_reward,
        "continuityFirstCandidate": np.asarray(
            continuity_first, dtype=np.int32
        ),
        "continuitySecondCandidate": np.asarray(
            continuity_second, dtype=np.int32
        ),
        "continuityAffinity": np.asarray(
            continuity_affinity, dtype=np.float32
        ),
        "continuityKind": np.asarray(continuity_kind, dtype=np.uint8),
        "continuityNormalDegrees": np.asarray(
            continuity_normal, dtype=np.float32
        ),
        "continuityHeightSamplingSteps": np.asarray(
            continuity_height, dtype=np.float32
        ),
        "continuityDistanceSamplingSteps": np.asarray(
            continuity_distance, dtype=np.float32
        ),
        "continuityThicknessChangeSamplingSteps": np.asarray(
            continuity_thickness_change, dtype=np.float32
        ),
        "continuityMissingFaceGapSamplingSteps": np.asarray(
            continuity_missing_face_gap, dtype=np.float32
        ),
        "continuityMissingFaceSignedNormalDegrees": np.asarray(
            continuity_missing_face_normal, dtype=np.float32
        ),
    }
    incident_multiplicity = np.asarray(
        [len(value) for value in incident.values()], dtype=np.int32
    )
    summary = {
        "inputDenseCandidateCount": int(len(dense_node)),
        "duplicateBoundaryPairCount": int(len(dense_node) - len(source_node)),
        "boundaryCandidateCount": int(len(source_node)),
        "usedBoundarySurfelCount": int(len(incident)),
        "ambiguousBoundarySurfelCount": int(
            np.count_nonzero(incident_multiplicity > 1)
        ),
        "maximumCandidateMatesPerBoundarySurfel": int(
            incident_multiplicity.max(initial=0)
        ),
        "continuityCount": int(len(continuity_first)),
        "exactTwoFaceContinuityCount": int(
            np.count_nonzero(np.asarray(continuity_kind) == 0)
        ),
        "oneFaceAnchoredClosureCount": int(
            np.count_nonzero(np.asarray(continuity_kind) == 1)
        ),
        "candidateMateMultiplicity": _percentile_record(incident_multiplicity),
        "candidateReward": _percentile_record(candidate_reward),
        "continuityAffinity": _percentile_record(
            np.asarray(continuity_affinity)
        ),
        "continuityNormalDegrees": _percentile_record(
            np.asarray(continuity_normal)
        ),
        "continuityHeightSamplingSteps": _percentile_record(
            np.asarray(continuity_height)
        ),
        "continuityThicknessChangeSamplingSteps": _percentile_record(
            np.asarray(continuity_thickness_change)
        ),
        "closureMissingFaceGapSamplingSteps": _percentile_record(
            np.asarray(continuity_missing_face_gap)[
                np.asarray(continuity_kind) == 1
            ]
        ),
        "closureMissingFaceSignedNormalDegrees": _percentile_record(
            np.asarray(continuity_missing_face_normal)[
                np.asarray(continuity_kind) == 1
            ]
        ),
    }
    return arrays, summary


def solve_laminar_boundary_problem(
    problem: Mapping[str, np.ndarray],
    *,
    settings: LaminarBoundaryMatchingSettings,
    output_flag: bool = False,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Globally select a collision-free, two-face-coherent correspondence set."""

    try:
        import highspy
    except ImportError as exc:  # pragma: no cover - exercised by lean installs
        raise RuntimeError(
            "laminar boundary matching requires requirements-optimization.txt"
        ) from exc

    boundary = np.stack(
        (
            np.asarray(problem["boundaryFirstSurfaceNode"], dtype=np.int32),
            np.asarray(problem["boundarySecondSurfaceNode"], dtype=np.int32),
        ),
        axis=1,
    )
    reward = np.asarray(problem["candidateReward"], dtype=np.float64)
    edge_first = np.asarray(problem["continuityFirstCandidate"], dtype=np.int32)
    edge_second = np.asarray(problem["continuitySecondCandidate"], dtype=np.int32)
    affinity = np.asarray(problem["continuityAffinity"], dtype=np.float64)
    candidate_count = len(boundary)
    if reward.shape != (candidate_count,):
        raise ValueError("candidate rewards do not match the boundary pair table")
    if any(len(value) != len(edge_first) for value in (edge_second, affinity)):
        raise ValueError("continuity arrays are not aligned")

    incident: dict[int, list[int]] = defaultdict(list)
    for candidate, (left, right) in enumerate(boundary):
        incident[int(left)].append(candidate)
        incident[int(right)].append(candidate)

    parent = np.arange(candidate_count, dtype=np.int32)
    tree_size = np.ones(candidate_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if tree_size[first_root] < tree_size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        tree_size[first_root] += tree_size[second_root]

    for first, second in zip(edge_first, edge_second):
        union(int(first), int(second))
    conflict_constraint_count = 0
    for members in incident.values():
        if len(members) <= 1:
            continue
        conflict_constraint_count += 1
        for following in members[1:]:
            union(members[0], following)

    members_by_root: dict[int, list[int]] = defaultdict(list)
    for candidate in range(candidate_count):
        members_by_root[find(candidate)].append(candidate)
    root_by_candidate = np.asarray(
        [find(candidate) for candidate in range(candidate_count)], dtype=np.int32
    )
    edges_by_root: dict[int, list[int]] = defaultdict(list)
    for edge, first in enumerate(edge_first):
        edges_by_root[int(root_by_candidate[int(first)])].append(edge)
    conflicts_by_root: dict[int, list[list[int]]] = defaultdict(list)
    for members in incident.values():
        if len(members) > 1:
            conflicts_by_root[int(root_by_candidate[members[0]])].append(members)

    # Components are independent both in hard mate conflicts and in pairwise
    # continuity rewards.  Solving them separately is mathematically exact and
    # avoids asking branch-and-bound to rediscover a block-diagonal structure.
    selected = np.zeros(candidate_count, dtype=bool)
    component_status: list[str] = []
    objective = 0.0
    dual_bound = 0.0
    mip_node_count = 0
    solved_component_count = 0
    direct_component_count = 0
    solve_started = time.monotonic()
    local_index = np.full(candidate_count, -1, dtype=np.int32)
    ordered_roots = sorted(
        members_by_root,
        key=lambda value: (-len(members_by_root[value]), value),
    )
    for component_rank, root in enumerate(ordered_roots):
        members = np.asarray(members_by_root[root], dtype=np.int32)
        component_edges = np.asarray(edges_by_root.get(root, ()), dtype=np.int32)
        conflict_groups = conflicts_by_root.get(root, ())
        if not conflict_groups:
            selected[members] = True
            component_objective = float(np.sum(reward[members]))
            if len(component_edges):
                component_objective += settings.continuity_reward * float(
                    np.sum(affinity[component_edges])
                )
            objective += component_objective
            dual_bound += component_objective
            direct_component_count += 1
            continue

        remaining = settings.maximum_solver_seconds - (
            time.monotonic() - solve_started
        )
        if remaining <= 0.0:
            # A deterministic collision-safe fallback is preferable to losing
            # the entire completed prefix when a user-supplied time cap is too
            # small.  The manifest records this as non-optimal.
            degree_reward = np.zeros(len(members), dtype=np.float64)
            local_index[members] = np.arange(len(members), dtype=np.int32)
            for edge in component_edges:
                left = int(local_index[edge_first[edge]])
                right = int(local_index[edge_second[edge]])
                value = settings.continuity_reward * float(affinity[edge])
                degree_reward[left] += value
                degree_reward[right] += value
            priority = reward[members] + degree_reward
            used: set[int] = set()
            for local in np.lexsort((members, -priority)):
                candidate = int(members[local])
                endpoints = tuple(map(int, boundary[candidate]))
                if endpoints[0] in used or endpoints[1] in used:
                    continue
                selected[candidate] = True
                used.update(endpoints)
            local_index[members] = -1
            component_status.append("time-cap greedy fallback")
            continue

        local_index[members] = np.arange(len(members), dtype=np.int32)
        model = highspy.Highs()
        model.setOptionValue("output_flag", bool(output_flag))
        model.setOptionValue(
            "time_limit",
            float(min(settings.maximum_component_solver_seconds, remaining)),
        )
        model.setOptionValue("mip_rel_gap", float(settings.mip_relative_gap))
        model.setOptionValue(
            "random_seed", int(settings.solver_random_seed + component_rank)
        )
        candidate_variable = model.addVariables(
            len(members),
            lb=0,
            ub=1,
            obj=reward[members].tolist(),
            type=highspy.HighsVarType.kInteger,
        )
        continuity_variable = model.addVariables(
            len(component_edges),
            lb=0,
            ub=1,
            obj=(settings.continuity_reward * affinity[component_edges]).tolist(),
            type=highspy.HighsVarType.kInteger,
        )
        for group in conflict_groups:
            model.addConstr(
                sum(
                    (
                        candidate_variable[int(local_index[candidate])]
                        for candidate in group
                    ),
                    start=0,
                )
                <= 1
            )
        for local_edge, edge in enumerate(component_edges):
            left = int(local_index[edge_first[edge]])
            right = int(local_index[edge_second[edge]])
            model.addConstr(continuity_variable[local_edge] <= candidate_variable[left])
            model.addConstr(continuity_variable[local_edge] <= candidate_variable[right])
        model.maximize()
        solution = model.getSolution()
        component_selected = np.asarray(
            solution.col_value[: len(members)], dtype=np.float64
        ) > 0.5
        selected[members[component_selected]] = True
        model_info = model.getInfo()
        status = model.modelStatusToString(model.getModelStatus())
        component_status.append(status)
        objective += float(model_info.objective_function_value)
        component_dual = float(model_info.mip_dual_bound)
        dual_bound += (
            component_dual
            if math.isfinite(component_dual)
            else float(model_info.objective_function_value)
        )
        mip_node_count += int(model_info.mip_node_count)
        solved_component_count += 1
        local_index[members] = -1

    if not np.any(selected):
        raise RuntimeError(
            "boundary matching returned no feasible candidates"
        )
    used = np.bincount(
        boundary[selected].ravel(),
        minlength=int(boundary.max(initial=-1)) + 1,
    )
    if used.max(initial=0) > 1:
        raise RuntimeError("global boundary solve violated the one-mate constraint")
    selected_continuity = selected[edge_first] & selected[edge_second]
    continuity_kind = np.asarray(problem["continuityKind"], dtype=np.uint8)
    fallback_count = sum(value == "time-cap greedy fallback" for value in component_status)
    nonoptimal = [
        value
        for value in component_status
        if value not in ("Optimal", "time-cap greedy fallback")
    ]
    if fallback_count:
        status = "Feasible with time-cap greedy fallback"
        aggregate_gap = None
    elif nonoptimal:
        status = "Feasible; some interaction components reached a solver limit"
        aggregate_gap = max(
            0.0, (dual_bound - objective) / max(abs(objective), 1.0e-12)
        )
    else:
        status = "Optimal"
        aggregate_gap = max(
            0.0, (dual_bound - objective) / max(abs(objective), 1.0e-12)
        )
    arrays = {
        "selectedCandidate": selected,
        "selectedContinuity": selected_continuity,
    }
    summary = {
        "solver": "HiGHS mixed-integer programming",
        "status": status,
        "objective": objective,
        "mipGap": aggregate_gap,
        "mipNodeCount": int(mip_node_count),
        "interactionComponentCount": int(len(members_by_root)),
        "directInteractionComponentCount": int(direct_component_count),
        "solvedInteractionComponentCount": int(solved_component_count),
        "fallbackInteractionComponentCount": int(fallback_count),
        "nonoptimalInteractionComponentCount": int(len(nonoptimal)),
        "conflictConstraintCount": int(conflict_constraint_count),
        "selectedCandidateCount": int(np.count_nonzero(selected)),
        "selectedCandidateFraction": float(np.mean(selected)),
        "selectedContinuityCount": int(np.count_nonzero(selected_continuity)),
        "selectedExactTwoFaceContinuityCount": int(
            np.count_nonzero(selected_continuity & (continuity_kind == 0))
        ),
        "selectedOneFaceAnchoredClosureCount": int(
            np.count_nonzero(selected_continuity & (continuity_kind == 1))
        ),
        "usedBoundarySurfelCount": int(np.count_nonzero(used)),
        "maximumSelectedMatesPerBoundarySurfel": int(used.max(initial=0)),
    }
    return arrays, summary


def _selected_catalog_arrays(
    source: Mapping[str, np.ndarray],
    problem: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    source_node = np.asarray(problem["sourceMidSurfaceNode"], dtype=np.int32)
    selected = np.asarray(selection["selectedCandidate"], dtype=bool)
    retained_source = source_node[selected]
    candidate_to_output = np.full(len(source_node), -1, dtype=np.int32)
    candidate_to_output[selected] = np.arange(
        np.count_nonzero(selected), dtype=np.int32
    )
    continuity_selected = np.asarray(selection["selectedContinuity"], dtype=bool)
    edge_first = candidate_to_output[
        np.asarray(problem["continuityFirstCandidate"], dtype=np.int32)[
            continuity_selected
        ]
    ]
    edge_second = candidate_to_output[
        np.asarray(problem["continuitySecondCandidate"], dtype=np.int32)[
            continuity_selected
        ]
    ]
    edge_score = np.asarray(problem["continuityAffinity"], dtype=np.float32)[
        continuity_selected
    ]
    edge_kind = np.asarray(problem["continuityKind"], dtype=np.uint8)[
        continuity_selected
    ]
    component, component_size = _components(
        len(retained_source), edge_first, edge_second
    )

    source_node_count = len(source["midpointXYZ"])
    source_edge_count = len(source["edgeFirstNode"])
    source_edge_fields = {
        name
        for name, value in source.items()
        if np.asarray(value).ndim >= 1 and len(value) == source_edge_count
    }
    arrays: dict[str, np.ndarray] = {}
    for name, value in source.items():
        item = np.asarray(value)
        if name in source_edge_fields:
            continue
        if item.ndim >= 1 and len(item) == source_node_count:
            arrays[name] = item[retained_source]
    arrays["physicalSheetLabel"] = component.astype(np.int32)
    arrays["componentId"] = component.astype(np.int32)
    arrays["edgeFirstNode"] = edge_first.astype(np.int32)
    arrays["edgeSecondNode"] = edge_second.astype(np.int32)
    arrays["edgeBoundarySupportMask"] = np.full(
        len(edge_first), 3, dtype=np.uint8
    )
    arrays["edgeKind"] = (4 + edge_kind).astype(np.uint8)
    arrays["edgeScore"] = edge_score
    return arrays, component_size


def run_laminar_boundary_matching(
    mid_surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: LaminarBoundaryMatchingSettings | None = None,
    force: bool = False,
    solver_output: bool = False,
) -> dict[str, Any]:
    """Persist the globally selected, exact two-face midpoint graph."""

    started = time.monotonic()
    resolved = settings or LaminarBoundaryMatchingSettings()
    source_path = _resolve_mid_surface(mid_surface_root)
    source_manifest, source = _load(source_path)
    if (
        source_manifest.get("schema") != PHYSICAL_MID_SURFACE_SCHEMA
        or source_manifest.get("state") != "complete"
    ):
        raise ValueError("boundary matching requires a complete midpoint catalog")
    surface_path = Path(
        source_manifest["identity"]["materialSurface"]["manifestPath"]
    ).resolve()
    surface_manifest, surface = _load(surface_path)
    if surface_manifest.get("schema") != MATERIAL_SURFACE_GRAPH_SCHEMA:
        raise ValueError("midpoint catalog references an invalid surface graph")
    if (
        source_manifest["identity"]["materialSurface"]["manifestSha256"]
        != sha256_file(surface_path)
    ):
        raise ValueError("midpoint catalog and material surface graph differ")
    stride = float(source_manifest["geometry"]["samplingStrideVoxels"])
    identity: dict[str, Any] = {
        "schema": LAMINAR_BOUNDARY_MATCHING_SCHEMA,
        "version": LAMINAR_BOUNDARY_MATCHING_VERSION,
        "candidateMidSurface": {
            "manifestPath": str(source_path),
            "manifestSha256": sha256_file(source_path),
            "dataSha256": source_manifest["data"]["sha256"],
        },
        "materialSurface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.json"
    data_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.npz"
    preview_path = output / "laminar-boundary-matching-cross-sections.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    problem, problem_summary = compile_laminar_boundary_problem(
        source,
        surface,
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    selection, solver_summary = solve_laminar_boundary_problem(
        problem, settings=resolved, output_flag=solver_output
    )
    arrays, component_size = _selected_catalog_arrays(source, problem, selection)
    _write_npz(data_path, arrays)

    volume = VolumeSource.open(
        source_manifest["source"]["path"],
        source_manifest["source"].get("metadataPath"),
    )
    owned_record = source_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    write_material_surface_cross_sections(
        volume,
        owned,
        arrays["midpointXYZ"],
        arrays["componentId"],
        component_size,
        preview_path,
        display_high_raw=float(source_manifest["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=int(stride),
        settings=MaterialSurfaceGraphSettings(
            minimum_component_samples_for_preview=8,
            maximum_preview_components=128,
        ),
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_VERSION,
        "constructionSchema": LAMINAR_BOUNDARY_MATCHING_SCHEMA,
        "constructionVersion": LAMINAR_BOUNDARY_MATCHING_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_manifest["source"],
        "geometry": source_manifest["geometry"],
        "calibration": source_manifest["calibration"],
        "counts": {
            **{
                key: value
                for key, value in problem_summary.items()
                if not isinstance(value, dict)
            },
            **{
                key: value
                for key, value in solver_summary.items()
                if key
                not in {
                    "solver",
                    "status",
                    "objective",
                    "mipGap",
                    "mipNodeCount",
                }
            },
            "midSurfaceNodeCount": int(len(arrays["midpointXYZ"])),
            "midSurfaceEdgeCount": int(len(arrays["edgeFirstNode"])),
            "midSurfaceComponentCount": int(len(component_size)),
            "componentsAtLeast128Nodes": int(
                np.count_nonzero(component_size >= 128)
            ),
            "largestComponentSizes": component_size[:32].astype(int).tolist(),
        },
        "distributions": {
            key: value
            for key, value in problem_summary.items()
            if isinstance(value, dict)
        },
        "solver": solver_summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "decisionVariable": (
                "one locally reconstructed correspondence between two opposing "
                "observed material/air boundary surfels"
            ),
            "hardConstraint": (
                "every physical boundary surfel has at most one selected mate"
            ),
            "continuityReward": (
                "two candidates receive affinity only when both of their "
                "physical boundaries continue; exact two-face graph edges have "
                "full weight, while a one-face edge may receive lower-weight "
                "closure only when the missing face independently agrees in "
                "position, signed normal, thickness, and midpoint geometry"
            ),
            "identityPolicy": (
                "selected two-face continuity components define page fragments; "
                "historical ribbon and grown-sheet labels are discarded"
            ),
            "acusRole": "none; fiber evidence is reserved for later ply analysis",
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
