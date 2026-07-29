from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rectify import _trilinear, grayscale_png
from .slab_carrier_assembly import (
    _carrier_boundary,
    _transported_fiber_angle,
)
from .slab_carrier_growth import _flake_arrays
from .slab_sheetlet_carriers import (
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _sample_stack,
    _texture_profile,
)


CARRIER_BRIDGE_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _bridge_endpoint_scores(
    first: dict[str, np.ndarray], second: dict[str, np.ndarray]
) -> dict[str, np.ndarray]:
    first_count, second_count = len(first["point"]), len(second["point"])
    first_index = np.repeat(np.arange(first_count, dtype=np.int64), second_count)
    second_index = np.tile(np.arange(second_count, dtype=np.int64), first_count)
    delta = second["point"][second_index] - first["point"][first_index]
    distance = np.linalg.norm(delta, axis=1)
    unit = delta / np.maximum(distance[:, None], 1.0e-8)
    first_normal = first["normal"][first_index]
    second_normal = second["normal"][second_index].copy()
    second_normal[np.sum(first_normal * second_normal, axis=1) < 0.0] *= -1.0
    normal_dot = np.clip(np.sum(first_normal * second_normal, axis=1), 0.0, 1.0)
    normal_bend = np.degrees(np.arccos(normal_dot))
    plane_residual = 0.5 * (
        np.abs(np.sum(delta * first_normal, axis=1))
        + np.abs(np.sum(delta * second_normal, axis=1))
    )
    fiber_angle = _transported_fiber_angle(
        first_normal,
        second_normal,
        first["fiber"][first_index],
        second["fiber"][second_index],
    )
    first_facing = np.sum(first["outward"][first_index] * unit, axis=1)
    second_facing = np.sum(second["outward"][second_index] * -unit, axis=1)
    facing = np.minimum(first_facing, second_facing)
    score = np.exp(
        -0.5
        * (
            ((distance - 40.0) / 96.0) ** 2
            + (plane_residual / 4.0) ** 2
            + (fiber_angle / 12.0) ** 2
            + (normal_bend / 22.0) ** 2
        )
    ) * np.sqrt(np.clip(first_facing, 0.0, 1.0) * np.clip(second_facing, 0.0, 1.0))
    valid = (
        (distance >= 40.0)
        & (distance <= 128.0)
        & (plane_residual <= 10.0)
        & (fiber_angle <= 35.0)
        & (normal_bend <= 50.0)
        & (facing >= 0.1)
        & (score >= 0.08)
    )
    return {
        "firstIndex": first_index,
        "secondIndex": second_index,
        "valid": valid,
        "score": score.astype(np.float32),
        "distance": distance.astype(np.float32),
        "planeResidual": plane_residual.astype(np.float32),
        "fiberAngle": fiber_angle.astype(np.float32),
        "normalBend": normal_bend.astype(np.float32),
        "facing": facing.astype(np.float32),
    }


def _interpolated_bridge(
    first_point: np.ndarray,
    second_point: np.ndarray,
    first_normal: np.ndarray,
    second_normal: np.ndarray,
    first_fiber: np.ndarray,
    second_fiber: np.ndarray,
    spacing: float = 8.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    distance = float(np.linalg.norm(second_point - first_point))
    sample_count = max(3, int(math.ceil(distance / spacing)) + 1)
    t = np.linspace(0.0, 1.0, sample_count, dtype=np.float32)[1:-1]
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
    if float(np.dot(first_fiber, second_fiber)) < 0.0:
        second_fiber = -second_fiber
    point = first_point[None, :] * (1.0 - t[:, None]) + second_point[None, :] * t[:, None]
    normal = first_normal[None, :] * (1.0 - t[:, None]) + second_normal[None, :] * t[:, None]
    normal /= np.maximum(np.linalg.norm(normal, axis=1, keepdims=True), 1.0e-8)
    fiber = first_fiber[None, :] * (1.0 - t[:, None]) + second_fiber[None, :] * t[:, None]
    fiber -= np.sum(fiber * normal, axis=1, keepdims=True) * normal
    fiber /= np.maximum(np.linalg.norm(fiber, axis=1, keepdims=True), 1.0e-8)
    return t, point.astype(np.float32), normal.astype(np.float32), fiber.astype(np.float32)


def _ct_bridge_evidence(
    source: np.ndarray,
    points: np.ndarray,
    normals: np.ndarray,
) -> dict[str, float]:
    offsets = np.asarray([-10.0, -6.0, -3.0, 0.0, 3.0, 6.0, 10.0], dtype=np.float32)
    query = points[:, None, :] + normals[:, None, :] * offsets[None, :, None]
    profile = _trilinear(source, query.reshape(-1, 3)).reshape(len(points), len(offsets))
    center = np.max(profile[:, 2:5], axis=1)
    flank = np.max(profile[:, [0, 1, 5, 6]], axis=1)
    material_fraction = float(np.mean(center >= 24.0))
    ridge_fraction = float(np.mean(center >= flank + 4.0))
    nonzero_fraction = float(np.mean(np.max(profile, axis=1) > 0.0))
    score = material_fraction * (0.75 + 0.25 * ridge_fraction)
    return {
        "score": round(score, 4),
        "materialFraction": round(material_fraction, 4),
        "ridgeFraction": round(ridge_fraction, 4),
        "nonzeroFraction": round(nonzero_fraction, 4),
        "medianCenterIntensity": round(float(np.median(center)), 3),
        "medianFlankIntensity": round(float(np.median(flank)), 3),
    }


def _flake_bridge_evidence(
    points: np.ndarray,
    normals: np.ndarray,
    fibers: np.ndarray,
    t: np.ndarray,
    arrays: dict[str, np.ndarray],
    available: np.ndarray,
) -> dict[str, Any]:
    low = np.min(points, axis=0) - 28.0
    high = np.max(points, axis=0) + 28.0
    center = arrays["center"]
    candidate = available[
        np.all((center[available] >= low[None, :]) & (center[available] <= high[None, :]), axis=1)
    ]
    if not len(candidate):
        return {"score": 0.0, "coverageFraction": 0.0, "supportFlakeIndices": []}
    distance2 = np.sum(
        (center[candidate, None, :] - points[None, :, :]) ** 2, axis=2
    )
    nearest_sample = np.argmin(distance2, axis=1)
    nearest_distance = np.sqrt(distance2[np.arange(len(candidate)), nearest_sample])
    keep = nearest_distance <= 28.0
    candidate = candidate[keep]
    nearest_sample = nearest_sample[keep]
    nearest_distance = nearest_distance[keep]
    if not len(candidate):
        return {"score": 0.0, "coverageFraction": 0.0, "supportFlakeIndices": []}
    delta = center[candidate] - points[nearest_sample]
    plane_residual = np.abs(np.sum(delta * normals[nearest_sample], axis=1))
    normal_angle = np.degrees(
        np.arccos(
            np.clip(
                np.abs(np.sum(arrays["normal"][candidate] * normals[nearest_sample], axis=1)),
                0.0,
                1.0,
            )
        )
    )
    fiber_angle = _transported_fiber_angle(
        normals[nearest_sample],
        arrays["normal"][candidate],
        fibers[nearest_sample],
        arrays["fiber"][candidate],
    )
    support_score = np.exp(
        -0.5
        * (
            (plane_residual / 5.0) ** 2
            + (normal_angle / 15.0) ** 2
            + (fiber_angle / 15.0) ** 2
            + (nearest_distance / 28.0) ** 2
        )
    ) * (0.75 + 0.25 * np.clip(arrays["quality"][candidate] / 0.35, 0.0, 1.0))
    valid = support_score >= 0.35
    if not np.any(valid):
        return {"score": 0.0, "coverageFraction": 0.0, "supportFlakeIndices": []}
    candidate = candidate[valid]
    nearest_sample = nearest_sample[valid]
    support_score = support_score[valid]
    bin_count = max(3, int(math.ceil(len(t) / 2.0)))
    bins = np.clip((t[nearest_sample] * bin_count).astype(np.int32), 0, bin_count - 1)
    coverage = len(np.unique(bins)) / bin_count
    order = np.argsort(support_score)[::-1]
    selected = candidate[order[: min(24, len(order))]]
    score = coverage * (0.75 + 0.25 * float(np.median(support_score)))
    return {
        "score": round(score, 4),
        "coverageFraction": round(float(coverage), 4),
        "supportFlakeCount": int(len(candidate)),
        "medianSupportScore": round(float(np.median(support_score)), 4),
        "supportFlakeIndices": selected.astype(int).tolist(),
    }


def _load_fixed_states(
    root: Path, arrays: dict[str, np.ndarray]
) -> tuple[list[dict[str, Any]], set[int]]:
    iteration = json.loads((root / "sheetlet-carrier-iteration-v1.json").read_text())
    with np.load(root / iteration["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"])
        member_offset = np.asarray(payload["memberOffset"])
    states = []
    reserved: set[int] = set()
    for index, summary in enumerate(iteration["states"]):
        low, high = int(member_offset[index]), int(member_offset[index + 1])
        members = set(int(value) for value in member_index[low:high])
        reserved.update(members)
        cells = {
            tuple(int(value) for value in arrays["cell"][member]) for member in members
        }
        states.append(
            {
                "members": members,
                "occupiedCells": cells,
                "sourceRanks": summary["sourceRanks"],
                "initialStateRanks": [int(summary["rank"])],
            }
        )
    return states, reserved


def _merge_bridges(
    states: list[dict[str, Any]],
    bridges: list[dict[str, Any]],
    threshold: float,
    arrays: dict[str, np.ndarray],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int, int]:
    parent = np.arange(len(states), dtype=np.int32)
    group_cells = [set(state["occupiedCells"]) for state in states]
    retained = []
    conflict_count = 0

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    for bridge in sorted(bridges, key=lambda value: value["bridgeScore"], reverse=True):
        if float(bridge["bridgeScore"]) < threshold:
            continue
        first, second = find(int(bridge["source"])), find(int(bridge["target"]))
        if first == second:
            retained.append(bridge)
            continue
        if not group_cells[first].isdisjoint(group_cells[second]):
            conflict_count += 1
            continue
        parent[second] = first
        group_cells[first].update(group_cells[second])
        group_cells[second] = set()
        retained.append(bridge)
    groups: dict[int, list[int]] = {}
    for index in range(len(states)):
        groups.setdefault(find(index), []).append(index)
    retained_by_root: dict[int, list[dict[str, Any]]] = {}
    for bridge in retained:
        retained_by_root.setdefault(find(int(bridge["source"])), []).append(bridge)
    claimed_support: set[int] = set().union(*(state["members"] for state in states))
    support_added = 0
    merged = []
    for root_index, indices in groups.items():
        members = set().union(*(states[index]["members"] for index in indices))
        cells = set().union(*(states[index]["occupiedCells"] for index in indices))
        for bridge in sorted(
            retained_by_root.get(root_index, []),
            key=lambda value: value["bridgeScore"],
            reverse=True,
        ):
            for flake_index in bridge["flakeEvidence"].get("supportFlakeIndices", []):
                flake_index = int(flake_index)
                cell = tuple(int(value) for value in arrays["cell"][flake_index])
                if flake_index in claimed_support or cell in cells:
                    continue
                members.add(flake_index)
                cells.add(cell)
                claimed_support.add(flake_index)
                support_added += 1
        merged.append(
            {
                "members": members,
                "occupiedCells": cells,
                "sourceRanks": sorted(
                    {
                        value
                        for index in indices
                        for value in states[index]["sourceRanks"]
                    }
                ),
                "initialStateRanks": sorted(
                    {
                        value
                        for index in indices
                        for value in states[index]["initialStateRanks"]
                    }
                ),
            }
        )
    merged.sort(key=lambda value: len(value["members"]), reverse=True)
    return merged, retained, conflict_count, support_added


def build_long_range_bridges(
    output_root: str | Path,
    selected_threshold: float = 0.36,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / f"sheetlet-carrier-bridges-v{CARRIER_BRIDGE_VERSION}.json"
    artifact_path = root / f"sheetlet-carrier-bridges-v{CARRIER_BRIDGE_VERSION}.npz"
    settings = {
        "minimumGapVoxels": 40.0,
        "maximumGapVoxels": 128.0,
        "bridgeSampleSpacingVoxels": 8.0,
        "selectedThreshold": selected_threshold,
        "thresholdSweep": [0.28, 0.32, 0.36, 0.4, 0.45],
    }
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("settings") == settings:
            return cached
    source_path, _, _, flakes = _load_carrier_catalog(root)
    source = np.load(source_path, mmap_mode="r")
    arrays = _flake_arrays(flakes)
    states, reserved = _load_fixed_states(root, arrays)
    available = np.asarray(
        [index for index in range(len(flakes)) if index not in reserved], dtype=np.int64
    )
    boundaries = []
    started = time.monotonic()
    for state in states:
        carrier = _mls_carrier(
            [flakes[index] for index in sorted(state["members"])],
            pixel_step=4.0,
            bandwidth=48.0,
            support_radius=48.0,
            maximum_pixels=192,
        )
        boundaries.append(_carrier_boundary(carrier))
    bridges = []
    overlapping_cell_pairs = 0
    endpoint_candidate_count = 0
    for first_index in range(len(states)):
        for second_index in range(first_index + 1, len(states)):
            if not states[first_index]["occupiedCells"].isdisjoint(
                states[second_index]["occupiedCells"]
            ):
                overlapping_cell_pairs += 1
                continue
            endpoint = _bridge_endpoint_scores(
                boundaries[first_index], boundaries[second_index]
            )
            valid = np.flatnonzero(endpoint["valid"])
            if not len(valid):
                continue
            endpoint_candidate_count += len(valid)
            top = valid[np.argsort(endpoint["score"][valid])[::-1][:12]]
            best_bridge = None
            for position in top:
                a = int(endpoint["firstIndex"][position])
                b = int(endpoint["secondIndex"][position])
                first_boundary, second_boundary = boundaries[first_index], boundaries[second_index]
                t, point, normal, fiber = _interpolated_bridge(
                    first_boundary["point"][a],
                    second_boundary["point"][b],
                    first_boundary["normal"][a],
                    second_boundary["normal"][b],
                    first_boundary["fiber"][a],
                    second_boundary["fiber"][b],
                )
                ct_evidence = _ct_bridge_evidence(source, point, normal)
                flake_evidence = _flake_bridge_evidence(
                    point, normal, fiber, t, arrays, available
                )
                evidence_score = (
                    0.75 * float(ct_evidence["score"])
                    + 0.25 * float(flake_evidence["score"])
                )
                bridge_score = float(endpoint["score"][position]) * (
                    0.5 + 0.5 * evidence_score
                )
                bridge = {
                    "source": first_index,
                    "target": second_index,
                    "endpointScore": round(float(endpoint["score"][position]), 4),
                    "bridgeScore": round(bridge_score, 4),
                    "distanceVoxels": round(float(endpoint["distance"][position]), 3),
                    "planeResidualVoxels": round(
                        float(endpoint["planeResidual"][position]), 3
                    ),
                    "fiberAngleDeg": round(float(endpoint["fiberAngle"][position]), 3),
                    "normalBendDeg": round(float(endpoint["normalBend"][position]), 3),
                    "facingCosine": round(float(endpoint["facing"][position]), 4),
                    "ctEvidence": ct_evidence,
                    "flakeEvidence": flake_evidence,
                    "firstPointXYZ": np.round(first_boundary["point"][a], 3).tolist(),
                    "secondPointXYZ": np.round(second_boundary["point"][b], 3).tolist(),
                }
                if best_bridge is None or bridge_score > best_bridge["bridgeScore"]:
                    best_bridge = bridge
            if best_bridge is not None:
                bridges.append(best_bridge)
    bridges.sort(key=lambda value: value["bridgeScore"], reverse=True)
    sweeps = []
    selected_states = None
    selected_retained = None
    selected_support_added = 0
    for threshold in settings["thresholdSweep"]:
        merged, retained, conflicts, support_added = _merge_bridges(
            states, bridges, float(threshold), arrays
        )
        sweeps.append(
            {
                "threshold": threshold,
                "acceptedBridgeCount": len(
                    [value for value in bridges if value["bridgeScore"] >= threshold]
                ),
                "retainedBridgeCount": len(retained),
                "cellConflictRejectedCount": conflicts,
                "stateCount": len(merged),
                "largestStateFlakeCount": max(len(state["members"]) for state in merged),
                "bridgeSupportFlakeCount": support_added,
            }
        )
        if abs(float(threshold) - selected_threshold) < 1.0e-8:
            selected_states = merged
            selected_retained = retained
            selected_support_added = support_added
    if selected_states is None or selected_retained is None:
        raise ValueError("selected_threshold must be in thresholdSweep")
    member_values = []
    member_offsets = [0]
    state_summaries = []
    for rank, state in enumerate(selected_states, start=1):
        members = np.asarray(sorted(state["members"]), dtype=np.uint32)
        member_values.append(members)
        member_offsets.append(member_offsets[-1] + len(members))
        state_summaries.append(
            {
                "rank": rank,
                "initialStateRanks": state["initialStateRanks"],
                "sourceRanks": state["sourceRanks"],
                "flakeCount": len(members),
                "uniqueCellCount": len(state["occupiedCells"]),
            }
        )
    all_members = np.concatenate(member_values)
    _atomic_npz(
        artifact_path,
        memberIndex=all_members,
        memberOffset=np.asarray(member_offsets, dtype=np.uint32),
    )
    result = {
        "identity": {"version": CARRIER_BRIDGE_VERSION, "source": str(source_path)},
        "settings": settings,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "inputStateCount": len(states),
            "outputStateCount": len(selected_states),
            "overlappingCellPairCount": overlapping_cell_pairs,
            "endpointCandidateCount": endpoint_candidate_count,
            "candidateBridgeCount": len(bridges),
            "retainedBridgeCount": len(selected_retained),
            "bridgeSupportFlakeCount": selected_support_added,
            "repeatedFlakeAssignmentCount": int(len(all_members) - len(np.unique(all_members))),
            "sameSheetCellCollisionCount": int(
                sum(len(state["members"]) - len(state["occupiedCells"]) for state in selected_states)
            ),
        },
        "sweep": sweeps,
        "bridges": bridges,
        "retainedBridges": selected_retained,
        "states": state_summaries,
        "artifact": str(artifact_path.relative_to(root)),
    }
    _atomic_json(summary_path, result)
    return result


def build_bridge_previews(
    output_root: str | Path,
    top_count: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    bridges = build_long_range_bridges(root)
    _, _, _, flakes = _load_carrier_catalog(root)
    with np.load(root / bridges["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"])
        member_offset = np.asarray(payload["memberOffset"])
    source_path = Path(bridges["identity"]["source"])
    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    selected = bridges["states"][:top_count]
    artifact_root = root / f"sheetlet-bridges-v{CARRIER_BRIDGE_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"summary-top{len(selected)}.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    outputs = []
    started = time.monotonic()
    for output_rank, state in enumerate(selected, start=1):
        state_index = int(state["rank"]) - 1
        low, high = int(member_offset[state_index]), int(member_offset[state_index + 1])
        indices = member_index[low:high]
        carrier = _mls_carrier([flakes[int(index)] for index in indices])
        stack, sampling = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        yield_stats = _carrier_yield(carrier["stats"], texture)
        candidate_root = artifact_root / f"rank-{output_rank:02d}"
        candidate_root.mkdir(parents=True, exist_ok=True)
        _atomic_npz(
            candidate_root / "carrier.npz",
            uValues=carrier["uValues"],
            vValues=carrier["vValues"],
            surfaceXYZ=carrier["surfaceXYZ"],
            normalXYZ=carrier["normalXYZ"],
            fiberXYZ=carrier["fiberXYZ"],
            supportMask=carrier["supportMask"],
            memberIndex=indices,
        )
        _atomic_npz(
            candidate_root / "depth-stack.npz",
            depthOffsets=depth_offsets,
            intensity=stack,
        )
        mask = carrier["supportMask"]
        center_index = int(np.argmin(np.abs(depth_offsets)))
        best_index = int(
            np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"]))
        )
        images = {
            "center.png": _contrast(stack[center_index], mask),
            "best-texture.png": _contrast(stack[best_index], mask),
            "depth-montage.png": np.concatenate(
                [_contrast(stack[index], mask) for index in (4, 8, 12, 16, 20)], axis=1
            ),
        }
        for filename, image in images.items():
            (candidate_root / filename).write_bytes(grayscale_png(image))
        output = {
            "state": state,
            "carrier": carrier["stats"],
            "texture": {key: texture[key] for key in texture if key != "planes"},
            "yield": yield_stats,
            "sampling": sampling,
            "artifacts": {
                "bestTextureImage": str(
                    (candidate_root / "best-texture.png").relative_to(root)
                ),
                "depthMontage": str(
                    (candidate_root / "depth-montage.png").relative_to(root)
                ),
            },
        }
        _atomic_json(candidate_root / "summary.json", output)
        outputs.append(output)
    result = {
        "identity": {"version": CARRIER_BRIDGE_VERSION},
        "settings": {"topCount": len(selected)},
        "stats": {
            **bridges["stats"],
            "previewElapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        },
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
