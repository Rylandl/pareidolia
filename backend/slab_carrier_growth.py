from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rectify import grayscale_png
from .slab_carrier_assembly import _transported_fiber_angle
from .slab_sheetlet_carriers import (
    _carrier_frame,
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _sample_stack,
    _texture_profile,
)


CARRIER_GROWTH_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _flake_arrays(flakes: list[dict[str, Any]]) -> dict[str, np.ndarray]:
    return {
        "center": np.asarray([flake["center"] for flake in flakes], dtype=np.float32),
        "normal": np.asarray([flake["normal"] for flake in flakes], dtype=np.float32),
        "fiber": np.asarray([flake["fiber"] for flake in flakes], dtype=np.float32),
        "quality": np.asarray([flake["quality"] for flake in flakes], dtype=np.float32),
        "cell": np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32),
    }


def _score_growth_candidates(
    member_indices: np.ndarray,
    candidate_indices: np.ndarray,
    arrays: dict[str, np.ndarray],
    flakes: list[dict[str, Any]],
    bandwidth: float = 48.0,
) -> dict[str, np.ndarray]:
    member_indices = np.asarray(member_indices, dtype=np.int64)
    candidate_indices = np.asarray(candidate_indices, dtype=np.int64)
    if not len(candidate_indices):
        empty = np.empty(0, dtype=np.float32)
        return {
            "score": empty,
            "heightResidual": empty,
            "normalAngle": empty,
            "fiberAngle": empty,
            "nearestPlanarDistance": empty,
        }
    member_flakes = [flakes[int(index)] for index in member_indices]
    frame = _carrier_frame(member_flakes)
    centers = arrays["center"][member_indices]
    normals = arrays["normal"][member_indices].copy()
    fibers = arrays["fiber"][member_indices].copy()
    quality = np.maximum(arrays["quality"][member_indices], 0.02)
    normals[(normals @ frame["normal"]) < 0.0] *= -1.0
    fibers[(fibers @ frame["uAxis"]) < 0.0] *= -1.0
    relative = centers - frame["origin"]
    node_u = relative @ frame["uAxis"]
    node_v = relative @ frame["vAxis"]
    node_h = relative @ frame["normal"]
    denominator = np.maximum(normals @ frame["normal"], 0.2)
    gradient_u = -(normals @ frame["uAxis"]) / denominator
    gradient_v = -(normals @ frame["vAxis"]) / denominator

    candidate_relative = arrays["center"][candidate_indices] - frame["origin"]
    query_u = candidate_relative @ frame["uAxis"]
    query_v = candidate_relative @ frame["vAxis"]
    query_h = candidate_relative @ frame["normal"]
    predicted_height = np.empty(len(candidate_indices), dtype=np.float32)
    predicted_normal = np.empty((len(candidate_indices), 3), dtype=np.float32)
    predicted_fiber = np.empty((len(candidate_indices), 3), dtype=np.float32)
    nearest = np.empty(len(candidate_indices), dtype=np.float32)
    for start in range(0, len(candidate_indices), 4096):
        stop = min(start + 4096, len(candidate_indices))
        delta_u = query_u[start:stop, None] - node_u[None, :]
        delta_v = query_v[start:stop, None] - node_v[None, :]
        distance2 = delta_u**2 + delta_v**2
        weights = np.exp(-0.5 * distance2 / (bandwidth**2)) * quality[None, :]
        weights[distance2 > (2.75 * bandwidth) ** 2] = 0.0
        sums = np.maximum(np.sum(weights, axis=1), 1.0e-8)
        local_height = (
            node_h[None, :]
            + delta_u * gradient_u[None, :]
            + delta_v * gradient_v[None, :]
        )
        predicted_height[start:stop] = np.sum(weights * local_height, axis=1) / sums
        blended_normal = weights @ normals
        blended_normal /= np.maximum(
            np.linalg.norm(blended_normal, axis=1, keepdims=True), 1.0e-8
        )
        predicted_normal[start:stop] = blended_normal
        blended_fiber = weights @ fibers
        blended_fiber -= (
            np.sum(blended_fiber * blended_normal, axis=1, keepdims=True)
            * blended_normal
        )
        blended_fiber /= np.maximum(
            np.linalg.norm(blended_fiber, axis=1, keepdims=True), 1.0e-8
        )
        predicted_fiber[start:stop] = blended_fiber
        nearest[start:stop] = np.sqrt(np.min(distance2, axis=1))

    candidate_normal = arrays["normal"][candidate_indices]
    normal_dot = np.clip(
        np.abs(np.sum(predicted_normal * candidate_normal, axis=1)), 0.0, 1.0
    )
    normal_angle = np.degrees(np.arccos(normal_dot))
    fiber_angle = _transported_fiber_angle(
        predicted_normal,
        candidate_normal,
        predicted_fiber,
        arrays["fiber"][candidate_indices],
    )
    height_residual = np.abs(query_h - predicted_height)
    reach_excess = np.maximum(nearest - 52.0, 0.0)
    candidate_quality = np.clip(arrays["quality"][candidate_indices] / 0.35, 0.0, 1.0)
    agreement = np.exp(
        -0.5
        * (
            (height_residual / 4.0) ** 2
            + (normal_angle / 12.0) ** 2
            + (fiber_angle / 12.0) ** 2
            + (reach_excess / 24.0) ** 2
        )
    )
    score = agreement * (0.72 + 0.28 * np.sqrt(candidate_quality))
    score[
        (nearest > 100.0)
        | (height_residual > 14.0)
        | (normal_angle > 40.0)
        | (fiber_angle > 40.0)
    ] = 0.0
    return {
        "score": score.astype(np.float32),
        "heightResidual": height_residual.astype(np.float32),
        "normalAngle": normal_angle.astype(np.float32),
        "fiberAngle": fiber_angle.astype(np.float32),
        "nearestPlanarDistance": nearest.astype(np.float32),
    }


def grow_carrier_hypotheses(
    output_root: str | Path,
    seed_count: int = 12,
    maximum_rounds: int = 16,
    score_threshold: float = 0.62,
    minimum_margin: float = 0.04,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / f"sheetlet-carrier-growth-v{CARRIER_GROWTH_VERSION}.json"
    artifact_path = root / f"sheetlet-carrier-growth-v{CARRIER_GROWTH_VERSION}.npz"
    settings = {
        "seedCount": seed_count,
        "maximumRounds": maximum_rounds,
        "scoreThreshold": score_threshold,
        "minimumBestVsSecondMargin": minimum_margin,
        "neighborhood": "26 adjacent Acus cells",
        "assignment": "best unclaimed flake per seed-cell, then global score order",
    }
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("settings") == settings:
            return cached

    assembly = json.loads(
        (root / "sheetlet-carrier-assembly-v1.json").read_text()
    )
    seeds = assembly["selected"]["topComponents"][:seed_count]
    _, _, flake_component, flakes = _load_carrier_catalog(root)
    arrays = _flake_arrays(flakes)
    cell_shape = np.max(arrays["cell"], axis=0) + 1
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(arrays["cell"]):
        by_cell.setdefault(tuple(int(value) for value in cell), []).append(index)
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]

    states = []
    reserved: set[int] = set()
    for seed_rank, seed in enumerate(seeds, start=1):
        component_ids = np.asarray(seed["sheetletComponentIds"], dtype=np.uint32)
        members = set(int(value) for value in np.flatnonzero(np.isin(flake_component, component_ids)))
        reserved.update(members)
        occupied_cells = {
            tuple(int(value) for value in arrays["cell"][index]) for index in members
        }
        states.append(
            {
                "seedRank": seed_rank,
                "assemblyComponentId": int(seed["componentId"]),
                "sourceRanks": seed["sourceRanks"],
                "initialMembers": set(members),
                "members": members,
                "occupiedCells": occupied_cells,
                "rounds": [],
            }
        )

    started = time.monotonic()
    round_summaries = []
    for round_index in range(1, maximum_rounds + 1):
        proposals = []
        for state_index, state in enumerate(states):
            neighboring_cells: set[tuple[int, int, int]] = set()
            for cell in state["occupiedCells"]:
                for offset in offsets:
                    neighbor = tuple(cell[axis] + offset[axis] for axis in range(3))
                    if any(
                        neighbor[axis] < 0 or neighbor[axis] >= int(cell_shape[axis])
                        for axis in range(3)
                    ):
                        continue
                    if neighbor not in state["occupiedCells"] and neighbor in by_cell:
                        neighboring_cells.add(neighbor)
            candidate_values = sorted(
                {
                    index
                    for cell in neighboring_cells
                    for index in by_cell[cell]
                    if index not in reserved
                }
            )
            if not candidate_values:
                state["rounds"].append({"candidateCount": 0, "acceptedCount": 0})
                continue
            candidate_indices = np.asarray(candidate_values, dtype=np.int64)
            member_indices = np.asarray(sorted(state["members"]), dtype=np.int64)
            scored = _score_growth_candidates(
                member_indices, candidate_indices, arrays, flakes
            )
            candidate_cells = arrays["cell"][candidate_indices]
            candidate_cell_code = (
                candidate_cells[:, 0].astype(np.int64)
                + int(cell_shape[0])
                * (
                    candidate_cells[:, 1].astype(np.int64)
                    + int(cell_shape[1]) * candidate_cells[:, 2].astype(np.int64)
                )
            )
            order = np.argsort(scored["score"])[::-1]
            seen_cells: set[int] = set()
            best_by_cell = []
            for position in order:
                code = int(candidate_cell_code[position])
                if code in seen_cells:
                    continue
                seen_cells.add(code)
                same_cell = np.flatnonzero(candidate_cell_code == code)
                cell_scores = np.sort(scored["score"][same_cell])[::-1]
                best_score = float(scored["score"][position])
                second_score = float(cell_scores[1]) if len(cell_scores) > 1 else 0.0
                if best_score < score_threshold or best_score - second_score < minimum_margin:
                    continue
                best_by_cell.append(position)
                proposals.append(
                    {
                        "state": state_index,
                        "flake": int(candidate_indices[position]),
                        "cell": tuple(int(value) for value in candidate_cells[position]),
                        "score": best_score,
                        "margin": best_score - second_score,
                        "heightResidual": float(scored["heightResidual"][position]),
                        "normalAngle": float(scored["normalAngle"][position]),
                        "fiberAngle": float(scored["fiberAngle"][position]),
                        "nearestPlanarDistance": float(
                            scored["nearestPlanarDistance"][position]
                        ),
                    }
                )
            state["rounds"].append(
                {
                    "candidateCount": len(candidate_indices),
                    "eligibleCellCount": len(best_by_cell),
                    "acceptedCount": 0,
                }
            )

        accepted = []
        for proposal in sorted(proposals, key=lambda value: value["score"], reverse=True):
            state = states[proposal["state"]]
            if proposal["flake"] in reserved or proposal["cell"] in state["occupiedCells"]:
                continue
            state["members"].add(proposal["flake"])
            state["occupiedCells"].add(proposal["cell"])
            state["rounds"][-1]["acceptedCount"] += 1
            reserved.add(proposal["flake"])
            accepted.append(proposal)
        round_summaries.append(
            {
                "round": round_index,
                "proposalCount": len(proposals),
                "acceptedCount": len(accepted),
                "medianAcceptedScore": round(
                    float(np.median([value["score"] for value in accepted])), 4
                )
                if accepted
                else None,
                "medianAcceptedHeightResidualVoxels": round(
                    float(np.median([value["heightResidual"] for value in accepted])), 3
                )
                if accepted
                else None,
                "medianAcceptedNormalAngleDeg": round(
                    float(np.median([value["normalAngle"] for value in accepted])), 3
                )
                if accepted
                else None,
                "medianAcceptedFiberAngleDeg": round(
                    float(np.median([value["fiberAngle"] for value in accepted])), 3
                )
                if accepted
                else None,
            }
        )
        if not accepted:
            break

    member_values = []
    member_offsets = [0]
    seed_summaries = []
    for state in states:
        members = np.asarray(sorted(state["members"]), dtype=np.uint32)
        member_values.append(members)
        member_offsets.append(member_offsets[-1] + len(members))
        added = len(state["members"] - state["initialMembers"])
        seed_summaries.append(
            {
                "seedRank": state["seedRank"],
                "assemblyComponentId": state["assemblyComponentId"],
                "sourceRanks": state["sourceRanks"],
                "initialFlakeCount": len(state["initialMembers"]),
                "grownFlakeCount": len(state["members"]),
                "addedFlakeCount": added,
                "growthFraction": round(added / max(len(state["initialMembers"]), 1), 4),
                "rounds": state["rounds"],
            }
        )
    all_member_indices = np.concatenate(member_values)
    repeated_seed_cell_count = sum(
        len(state["members"]) - len(state["occupiedCells"]) for state in states
    )
    repeated_flake_count = len(all_member_indices) - len(np.unique(all_member_indices))
    _atomic_npz(
        artifact_path,
        memberIndex=all_member_indices,
        memberOffset=np.asarray(member_offsets, dtype=np.uint32),
    )
    result = {
        "identity": {
            "version": CARRIER_GROWTH_VERSION,
            "assemblyVersion": assembly["identity"]["version"],
        },
        "settings": settings,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "seedCount": len(states),
            "completedRoundCount": len(round_summaries),
            "initialFlakeCount": sum(
                len(state["initialMembers"]) for state in states
            ),
            "grownFlakeCount": sum(len(state["members"]) for state in states),
            "addedFlakeCount": sum(
                len(state["members"] - state["initialMembers"]) for state in states
            ),
            "repeatedFlakeAssignmentCount": int(repeated_flake_count),
            "sameSeedCellCollisionCount": int(repeated_seed_cell_count),
        },
        "rounds": round_summaries,
        "seeds": seed_summaries,
        "artifact": str(artifact_path.relative_to(root)),
    }
    _atomic_json(summary_path, result)
    return result


def build_growth_previews(
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    growth = grow_carrier_hypotheses(root)
    _, _, _, flakes = _load_carrier_catalog(root)
    with np.load(root / growth["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"])
        member_offset = np.asarray(payload["memberOffset"])
    source_path = Path(
        json.loads((root / "analysis.json").read_text())["identity"]["source"]
    )
    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    artifact_root = root / f"sheetlet-growth-v{CARRIER_GROWTH_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / "summary.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    assembly_preview = json.loads(
        (root / "sheetlet-assemblies-v1/summary-top12.json").read_text()
    )
    seed_by_rank = {int(value["rank"]): value for value in assembly_preview["candidates"]}
    outputs = []
    started = time.monotonic()
    for seed_index, seed in enumerate(growth["seeds"]):
        low, high = int(member_offset[seed_index]), int(member_offset[seed_index + 1])
        indices = member_index[low:high]
        carrier = _mls_carrier([flakes[int(index)] for index in indices])
        stack, sampling_stats = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        yield_stats = _carrier_yield(carrier["stats"], texture)
        candidate_root = artifact_root / f"seed-{seed_index + 1:02d}"
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
        best_index = int(
            np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"]))
        )
        center_index = int(np.argmin(np.abs(depth_offsets)))
        selected_depths = [4, 8, 12, 16, 20]
        images = {
            "center.png": _contrast(stack[center_index], mask),
            "best-texture.png": _contrast(stack[best_index], mask),
            "depth-montage.png": np.concatenate(
                [_contrast(stack[index], mask) for index in selected_depths], axis=1
            ),
        }
        for filename, image in images.items():
            (candidate_root / filename).write_bytes(grayscale_png(image))
        baseline = seed_by_rank[seed_index + 1]
        output = {
            "seed": seed,
            "carrier": carrier["stats"],
            "texture": {
                key: texture[key]
                for key in (
                    "bestDepthOffsetVoxels",
                    "bestTextureScore",
                    "centerTextureScore",
                    "medianTextureScoreAcrossDepth",
                    "depthPeakSharpness",
                    "bestPlane",
                    "centerPlane",
                )
            },
            "yield": yield_stats,
            "growthCost": {
                "seedMedianHeightResidualVoxels": baseline["carrier"][
                    "medianNodeHeightResidualVoxels"
                ],
                "grownMedianHeightResidualVoxels": carrier["stats"][
                    "medianNodeHeightResidualVoxels"
                ],
                "seedMedianNormalResidualDeg": baseline["carrier"][
                    "medianNodeNormalResidualDeg"
                ],
                "grownMedianNormalResidualDeg": carrier["stats"][
                    "medianNodeNormalResidualDeg"
                ],
            },
            "sampling": sampling_stats,
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
        "identity": {
            "version": CARRIER_GROWTH_VERSION,
            "growthVersion": growth["identity"]["version"],
        },
        "settings": growth["settings"],
        "stats": {
            **growth["stats"],
            "previewElapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        },
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
