from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rectify import grayscale_png
from .slab_carrier_assembly import _carrier_boundary, _score_point_pairs
from .slab_carrier_growth import _flake_arrays, _score_growth_candidates
from .slab_sheetlet_carriers import (
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _sample_stack,
    _texture_profile,
)


CARRIER_ITERATION_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _assembly_seed_groups(root: Path) -> list[dict[str, Any]]:
    with np.load(root / "sheetlet-carrier-assembly-v1-components.npz") as payload:
        assembly_component = np.asarray(payload["component"], dtype=np.int32)
    with np.load(root / "sheetlet-carrier-boundaries-v1.npz") as payload:
        sheetlet_component = np.asarray(payload["componentId"], dtype=np.uint32)
        member_count = np.asarray(payload["memberCount"], dtype=np.uint32)
        source_rank = np.asarray(payload["sourceRank"], dtype=np.uint32)
    component_ids, counts = np.unique(assembly_component, return_counts=True)
    groups = []
    for component_id, carrier_count in zip(component_ids, counts):
        if int(carrier_count) < 2:
            continue
        member = assembly_component == component_id
        groups.append(
            {
                "assemblyComponentIds": [int(component_id)],
                "sheetletComponentIds": sheetlet_component[member].astype(int).tolist(),
                "sourceRanks": source_rank[member].astype(int).tolist(),
                "carrierCount": int(carrier_count),
                "initialFlakeCount": int(np.sum(member_count[member])),
            }
        )
    groups.sort(
        key=lambda value: (value["carrierCount"], value["initialFlakeCount"]),
        reverse=True,
    )
    return groups


def _occupied_cells(
    members: set[int], cell_array: np.ndarray
) -> set[tuple[int, int, int]]:
    return {
        tuple(int(value) for value in cell_array[index]) for index in members
    }


def _grow_states(
    states: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    flakes: list[dict[str, Any]],
    by_cell: dict[tuple[int, int, int], list[int]],
    maximum_rounds: int,
    score_threshold: float,
    minimum_margin: float,
) -> dict[str, Any]:
    cell_shape = np.max(arrays["cell"], axis=0) + 1
    offsets = [
        (dx, dy, dz)
        for dx in (-1, 0, 1)
        for dy in (-1, 0, 1)
        for dz in (-1, 0, 1)
        if (dx, dy, dz) != (0, 0, 0)
    ]
    reserved = set().union(*(state["members"] for state in states))
    rounds = []
    total_added = 0
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
            candidates = sorted(
                {
                    index
                    for cell in neighboring_cells
                    for index in by_cell[cell]
                    if index not in reserved
                }
            )
            if not candidates:
                continue
            candidate_indices = np.asarray(candidates, dtype=np.int64)
            scored = _score_growth_candidates(
                np.asarray(sorted(state["members"]), dtype=np.int64),
                candidate_indices,
                arrays,
                flakes,
            )
            candidate_cells = arrays["cell"][candidate_indices]
            cell_code = (
                candidate_cells[:, 0].astype(np.int64)
                + int(cell_shape[0])
                * (
                    candidate_cells[:, 1].astype(np.int64)
                    + int(cell_shape[1]) * candidate_cells[:, 2].astype(np.int64)
                )
            )
            for code in np.unique(cell_code):
                positions = np.flatnonzero(cell_code == code)
                order = positions[np.argsort(scored["score"][positions])[::-1]]
                best = int(order[0])
                best_score = float(scored["score"][best])
                second_score = float(scored["score"][order[1]]) if len(order) > 1 else 0.0
                if best_score < score_threshold or best_score - second_score < minimum_margin:
                    continue
                proposals.append(
                    {
                        "state": state_index,
                        "flake": int(candidate_indices[best]),
                        "cell": tuple(int(value) for value in candidate_cells[best]),
                        "score": best_score,
                        "heightResidual": float(scored["heightResidual"][best]),
                        "normalAngle": float(scored["normalAngle"][best]),
                        "fiberAngle": float(scored["fiberAngle"][best]),
                    }
                )
        accepted = []
        for proposal in sorted(proposals, key=lambda value: value["score"], reverse=True):
            state = states[proposal["state"]]
            if proposal["flake"] in reserved or proposal["cell"] in state["occupiedCells"]:
                continue
            state["members"].add(proposal["flake"])
            state["occupiedCells"].add(proposal["cell"])
            reserved.add(proposal["flake"])
            accepted.append(proposal)
        rounds.append(
            {
                "round": round_index,
                "proposalCount": len(proposals),
                "acceptedCount": len(accepted),
                "medianAcceptedScore": round(
                    float(np.median([value["score"] for value in accepted])), 4
                )
                if accepted
                else None,
                "medianHeightResidualVoxels": round(
                    float(np.median([value["heightResidual"] for value in accepted])), 3
                )
                if accepted
                else None,
                "medianNormalAngleDeg": round(
                    float(np.median([value["normalAngle"] for value in accepted])), 3
                )
                if accepted
                else None,
                "medianFiberAngleDeg": round(
                    float(np.median([value["fiberAngle"] for value in accepted])), 3
                )
                if accepted
                else None,
            }
        )
        total_added += len(accepted)
        if not accepted:
            break
    return {"addedFlakeCount": total_added, "rounds": rounds}


def _state_boundary_edges(
    states: list[dict[str, Any]], flakes: list[dict[str, Any]]
) -> tuple[list[dict[str, Any]], int]:
    boundaries = []
    for state in states:
        carrier = _mls_carrier(
            [flakes[index] for index in sorted(state["members"])],
            pixel_step=4.0,
            bandwidth=48.0,
            support_radius=48.0,
            maximum_pixels=192,
        )
        boundaries.append(_carrier_boundary(carrier))
    edges = []
    cell_conflicts = 0
    for first in range(len(states)):
        for second in range(first + 1, len(states)):
            if not states[first]["occupiedCells"].isdisjoint(
                states[second]["occupiedCells"]
            ):
                cell_conflicts += 1
                continue
            first_boundary, second_boundary = boundaries[first], boundaries[second]
            if not len(first_boundary["point"]) or not len(second_boundary["point"]):
                continue
            arrays = {
                key: np.concatenate([first_boundary[key], second_boundary[key]])
                for key in ("point", "normal", "fiber", "outward")
            }
            first_index = np.repeat(
                np.arange(len(first_boundary["point"]), dtype=np.int64),
                len(second_boundary["point"]),
            )
            second_index = np.tile(
                np.arange(len(second_boundary["point"]), dtype=np.int64)
                + len(first_boundary["point"]),
                len(first_boundary["point"]),
            )
            scored = _score_point_pairs(first_index, second_index, arrays)
            valid = np.flatnonzero(scored["valid"])
            if not len(valid):
                continue
            source_support = len(np.unique(first_index[valid]))
            target_support = len(np.unique(second_index[valid]))
            support = min(source_support, target_support)
            if support < 2:
                continue
            scores = np.sort(scored["score"][valid])[::-1]
            aggregate = float(np.median(scores[: min(5, len(scores))]))
            aggregate *= 0.85 + 0.15 * min(support / 6.0, 1.0)
            best = int(valid[np.argmax(scored["score"][valid])])
            edges.append(
                {
                    "source": first,
                    "target": second,
                    "score": aggregate,
                    "support": support,
                    "distanceVoxels": float(scored["distance"][best]),
                    "planeResidualVoxels": float(scored["planeResidual"][best]),
                    "fiberAngleDeg": float(scored["fiberAngle"][best]),
                    "normalBendDeg": float(scored["normalBend"][best]),
                }
            )
    edges.sort(key=lambda value: value["score"], reverse=True)
    return edges, cell_conflicts


def _merge_states(
    states: list[dict[str, Any]],
    edges: list[dict[str, Any]],
    threshold: float,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], int]:
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

    for edge in edges:
        if float(edge["score"]) < threshold:
            continue
        first, second = find(int(edge["source"])), find(int(edge["target"]))
        if first == second:
            retained.append(edge)
            continue
        if not group_cells[first].isdisjoint(group_cells[second]):
            conflict_count += 1
            continue
        parent[second] = first
        group_cells[first].update(group_cells[second])
        group_cells[second] = set()
        retained.append(edge)
    groups: dict[int, list[int]] = {}
    for index in range(len(states)):
        groups.setdefault(find(index), []).append(index)
    merged_states = []
    for indices in groups.values():
        members = set().union(*(states[index]["members"] for index in indices))
        merged_states.append(
            {
                "members": members,
                "occupiedCells": set().union(
                    *(states[index]["occupiedCells"] for index in indices)
                ),
                "assemblyComponentIds": sorted(
                    {
                        value
                        for index in indices
                        for value in states[index]["assemblyComponentIds"]
                    }
                ),
                "sourceRanks": sorted(
                    {
                        value
                        for index in indices
                        for value in states[index]["sourceRanks"]
                    }
                ),
                "initialFlakeCount": sum(
                    int(states[index]["initialFlakeCount"]) for index in indices
                ),
            }
        )
    merged_states.sort(key=lambda value: len(value["members"]), reverse=True)
    merge_count = len(states) - len(merged_states)
    return merged_states, retained, conflict_count


def iterate_carrier_hypotheses(
    output_root: str | Path,
    maximum_cycles: int = 6,
    maximum_growth_rounds: int = 16,
    growth_score_threshold: float = 0.62,
    growth_minimum_margin: float = 0.04,
    boundary_score_threshold: float = 0.45,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    summary_path = root / f"sheetlet-carrier-iteration-v{CARRIER_ITERATION_VERSION}.json"
    artifact_path = root / f"sheetlet-carrier-iteration-v{CARRIER_ITERATION_VERSION}.npz"
    settings = {
        "maximumCycles": maximum_cycles,
        "maximumGrowthRoundsPerCycle": maximum_growth_rounds,
        "growthScoreThreshold": growth_score_threshold,
        "growthMinimumMargin": growth_minimum_margin,
        "boundaryScoreThreshold": boundary_score_threshold,
    }
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("settings") == settings:
            return cached
    groups = _assembly_seed_groups(root)
    _, _, flake_component, flakes = _load_carrier_catalog(root)
    arrays = _flake_arrays(flakes)
    by_cell: dict[tuple[int, int, int], list[int]] = {}
    for index, cell in enumerate(arrays["cell"]):
        by_cell.setdefault(tuple(int(value) for value in cell), []).append(index)
    states = []
    for group in groups:
        component_ids = np.asarray(group["sheetletComponentIds"], dtype=np.uint32)
        members = set(
            int(value) for value in np.flatnonzero(np.isin(flake_component, component_ids))
        )
        states.append(
            {
                "members": members,
                "occupiedCells": _occupied_cells(members, arrays["cell"]),
                "assemblyComponentIds": group["assemblyComponentIds"],
                "sourceRanks": group["sourceRanks"],
                "initialFlakeCount": group["initialFlakeCount"],
            }
        )
    states.sort(key=lambda value: len(value["members"]), reverse=True)
    initial_state_count = len(states)
    initial_flake_count = sum(len(state["members"]) for state in states)
    cycles = []
    started = time.monotonic()
    all_retained_edges = []
    for cycle_index in range(1, maximum_cycles + 1):
        before_count = len(states)
        growth = _grow_states(
            states,
            arrays,
            flakes,
            by_cell,
            maximum_growth_rounds,
            growth_score_threshold,
            growth_minimum_margin,
        )
        edges, pair_cell_conflicts = _state_boundary_edges(states, flakes)
        states, retained, transitive_conflicts = _merge_states(
            states, edges, boundary_score_threshold
        )
        merge_count = before_count - len(states)
        all_retained_edges.extend(
            [{"cycle": cycle_index, **edge} for edge in retained]
        )
        cycles.append(
            {
                "cycle": cycle_index,
                "stateCountBefore": before_count,
                "growth": growth,
                "boundaryCandidateCount": len(edges),
                "retainedBoundaryCount": len(retained),
                "pairCellConflictCount": pair_cell_conflicts,
                "transitiveCellConflictCount": transitive_conflicts,
                "mergeCount": merge_count,
                "stateCountAfter": len(states),
            }
        )
        if growth["addedFlakeCount"] == 0 and merge_count == 0:
            break
    member_values = []
    offsets = [0]
    final_states = []
    for final_rank, state in enumerate(states, start=1):
        members = np.asarray(sorted(state["members"]), dtype=np.uint32)
        member_values.append(members)
        offsets.append(offsets[-1] + len(members))
        final_states.append(
            {
                "rank": final_rank,
                "assemblyComponentIds": state["assemblyComponentIds"],
                "sourceRanks": state["sourceRanks"],
                "initialFlakeCount": int(state["initialFlakeCount"]),
                "finalFlakeCount": len(members),
                "addedFlakeCount": len(members) - int(state["initialFlakeCount"]),
                "growthFraction": round(
                    (len(members) - int(state["initialFlakeCount"]))
                    / max(int(state["initialFlakeCount"]), 1),
                    4,
                ),
                "uniqueCellCount": len(state["occupiedCells"]),
            }
        )
    all_members = np.concatenate(member_values)
    _atomic_npz(
        artifact_path,
        memberIndex=all_members,
        memberOffset=np.asarray(offsets, dtype=np.uint32),
    )
    result = {
        "identity": {"version": CARRIER_ITERATION_VERSION},
        "settings": settings,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "initialStateCount": initial_state_count,
            "finalStateCount": len(states),
            "completedCycleCount": len(cycles),
            "initialFlakeCount": initial_flake_count,
            "finalFlakeCount": int(len(all_members)),
            "addedFlakeCount": int(len(all_members) - initial_flake_count),
            "repeatedFlakeAssignmentCount": int(len(all_members) - len(np.unique(all_members))),
            "sameSheetCellCollisionCount": int(
                sum(len(state["members"]) - len(state["occupiedCells"]) for state in states)
            ),
        },
        "cycles": cycles,
        "retainedBoundaryEdges": all_retained_edges,
        "states": final_states,
        "artifact": str(artifact_path.relative_to(root)),
    }
    _atomic_json(summary_path, result)
    return result


def build_iteration_previews(
    output_root: str | Path,
    top_count: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    iteration = iterate_carrier_hypotheses(root)
    _, _, _, flakes = _load_carrier_catalog(root)
    with np.load(root / iteration["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"])
        member_offset = np.asarray(payload["memberOffset"])
    source_path = Path(
        json.loads((root / "analysis.json").read_text())["identity"]["source"]
    )
    source = np.load(source_path, mmap_mode="r")
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    selected_states = iteration["states"][:top_count]
    artifact_root = root / f"sheetlet-iteration-v{CARRIER_ITERATION_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"summary-top{len(selected_states)}.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    outputs = []
    started = time.monotonic()
    for output_rank, state in enumerate(selected_states, start=1):
        state_index = int(state["rank"]) - 1
        low, high = int(member_offset[state_index]), int(member_offset[state_index + 1])
        indices = member_index[low:high]
        carrier = _mls_carrier([flakes[int(index)] for index in indices])
        stack, sampling = _sample_stack(source, carrier, depth_offsets)
        texture = _texture_profile(stack, carrier["supportMask"], depth_offsets)
        yield_stats = _carrier_yield(carrier["stats"], texture)
        output_root_path = artifact_root / f"rank-{output_rank:02d}"
        output_root_path.mkdir(parents=True, exist_ok=True)
        _atomic_npz(
            output_root_path / "carrier.npz",
            uValues=carrier["uValues"],
            vValues=carrier["vValues"],
            surfaceXYZ=carrier["surfaceXYZ"],
            normalXYZ=carrier["normalXYZ"],
            fiberXYZ=carrier["fiberXYZ"],
            supportMask=carrier["supportMask"],
            memberIndex=indices,
        )
        _atomic_npz(
            output_root_path / "depth-stack.npz",
            depthOffsets=depth_offsets,
            intensity=stack,
        )
        mask = carrier["supportMask"]
        center_index = int(np.argmin(np.abs(depth_offsets)))
        best_index = int(
            np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"]))
        )
        montage_indices = [4, 8, 12, 16, 20]
        images = {
            "center.png": _contrast(stack[center_index], mask),
            "best-texture.png": _contrast(stack[best_index], mask),
            "depth-montage.png": np.concatenate(
                [_contrast(stack[index], mask) for index in montage_indices], axis=1
            ),
        }
        for filename, image in images.items():
            (output_root_path / filename).write_bytes(grayscale_png(image))
        output = {
            "state": state,
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
            "sampling": sampling,
            "artifacts": {
                "bestTextureImage": str(
                    (output_root_path / "best-texture.png").relative_to(root)
                ),
                "depthMontage": str(
                    (output_root_path / "depth-montage.png").relative_to(root)
                ),
            },
        }
        _atomic_json(output_root_path / "summary.json", output)
        outputs.append(output)
    result = {
        "identity": {"version": CARRIER_ITERATION_VERSION},
        "settings": {"topCount": len(selected_states)},
        "stats": {
            **iteration["stats"],
            "previewElapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        },
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
