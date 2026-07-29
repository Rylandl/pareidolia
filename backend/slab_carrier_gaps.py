from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .rectify import grayscale_png
from .slab_carrier_bridges import CARRIER_BRIDGE_VERSION
from .slab_carrier_growth import _flake_arrays, _score_growth_candidates
from .slab_sheetlet_carriers import (
    _carrier_yield,
    _contrast,
    _load_carrier_catalog,
    _mls_carrier,
    _plane_texture,
    _sample_stack,
    _texture_profile,
)


CARRIER_GAP_VERSION = 2


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _internal_gap_components(
    support_mask: np.ndarray,
    pixel_step: float,
    minimum_area_square_voxels: float = 256.0,
) -> list[dict[str, Any]]:
    """Return unsupported 4-connected regions fully enclosed by support."""
    support = np.asarray(support_mask, dtype=bool)
    if support.ndim != 2:
        raise ValueError("carrier support mask must be two-dimensional")
    missing = ~support
    seen = np.zeros(support.shape, dtype=bool)
    minimum_pixels = max(
        1, int(np.ceil(float(minimum_area_square_voxels) / max(pixel_step**2, 1.0e-8)))
    )
    components: list[dict[str, Any]] = []
    height, width = support.shape
    for seed_y, seed_x in np.argwhere(missing):
        seed_y, seed_x = int(seed_y), int(seed_x)
        if seen[seed_y, seed_x]:
            continue
        stack = [(seed_y, seed_x)]
        seen[seed_y, seed_x] = True
        pixels: list[tuple[int, int]] = []
        touches_border = False
        support_contact = 0
        while stack:
            y, x = stack.pop()
            pixels.append((y, x))
            touches_border = touches_border or y == 0 or x == 0 or y == height - 1 or x == width - 1
            for neighbor_y, neighbor_x in (
                (y - 1, x),
                (y + 1, x),
                (y, x - 1),
                (y, x + 1),
            ):
                if not (0 <= neighbor_y < height and 0 <= neighbor_x < width):
                    continue
                if support[neighbor_y, neighbor_x]:
                    support_contact += 1
                elif not seen[neighbor_y, neighbor_x]:
                    seen[neighbor_y, neighbor_x] = True
                    stack.append((neighbor_y, neighbor_x))
        if touches_border or len(pixels) < minimum_pixels:
            continue
        coordinates = np.asarray(pixels, dtype=np.int32)
        low = np.min(coordinates, axis=0)
        high = np.max(coordinates, axis=0) + 1
        mask = np.zeros(support.shape, dtype=bool)
        mask[coordinates[:, 0], coordinates[:, 1]] = True
        components.append(
            {
                "mask": mask,
                "pixelCount": int(len(pixels)),
                "areaSquareVoxels": round(float(len(pixels) * pixel_step**2), 2),
                "bboxYX": [int(low[0]), int(high[0]), int(low[1]), int(high[1])],
                "centroidYX": np.round(np.mean(coordinates, axis=0), 2).tolist(),
                "spanVoxelsUV": [
                    round(float((high[1] - low[1]) * pixel_step), 2),
                    round(float((high[0] - low[0]) * pixel_step), 2),
                ],
                "supportContactEdges": int(support_contact),
            }
        )
    components.sort(key=lambda value: float(value["areaSquareVoxels"]), reverse=True)
    for index, component in enumerate(components, start=1):
        component["gapId"] = index
    return components


def _project_flakes_to_carrier(
    carrier: dict[str, Any], centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    frame = carrier["frame"]
    relative = centers - frame["origin"]
    u = relative @ frame["uAxis"]
    v = relative @ frame["vAxis"]
    u_step = float(carrier["uValues"][1] - carrier["uValues"][0])
    v_step = float(carrier["vValues"][1] - carrier["vValues"][0])
    x = np.rint((u - float(carrier["uValues"][0])) / u_step).astype(np.int32)
    y = np.rint((v - float(carrier["vValues"][0])) / v_step).astype(np.int32)
    inside = (
        (x >= 0)
        & (x < carrier["supportMask"].shape[1])
        & (y >= 0)
        & (y < carrier["supportMask"].shape[0])
    )
    return y, x, inside


def _cell_codes(cells: np.ndarray, cell_shape: np.ndarray) -> np.ndarray:
    return (
        cells[:, 0].astype(np.int64)
        + int(cell_shape[0])
        * (
            cells[:, 1].astype(np.int64)
            + int(cell_shape[1]) * cells[:, 2].astype(np.int64)
        )
    )


def _axial_angle_difference(first: float, second: float) -> float:
    difference = abs(float(first) - float(second)) % 180.0
    return min(difference, 180.0 - difference)


def _gap_ct_evidence(
    carrier: dict[str, Any],
    stack: np.ndarray,
    depth_offsets: np.ndarray,
    gap: dict[str, Any],
    air_threshold: float,
) -> dict[str, Any]:
    region = np.asarray(gap["mask"], dtype=bool) & carrier["supportMask"]
    supported_pixels = int(np.count_nonzero(region))
    if supported_pixels < 24:
        return {
            "expandedSupportPixelCount": supported_pixels,
            "expandedSupportFraction": round(
                supported_pixels / max(int(gap["pixelCount"]), 1), 4
            ),
            "materialFraction": 0.0,
            "bestDepthOffsetVoxels": None,
            "bestTextureScore": 0.0,
            "centerTextureScore": 0.0,
            "predictedFiberAngleDeg": None,
            "observedFiberAngleDeg": None,
            "fiberAngleResidualDeg": None,
            "depthAlignedTextureScore": 0.0,
        }
    planes = [_plane_texture(image, region, block_size=8) for image in stack]
    scores = np.asarray([float(plane["textureScore"]) for plane in planes])
    best_index = int(np.argmax(scores))
    center_index = int(np.argmin(np.abs(depth_offsets)))
    fibers = carrier["fiberXYZ"][region]
    frame = carrier["frame"]
    fiber_u = fibers @ frame["uAxis"]
    fiber_v = fibers @ frame["vAxis"]
    angles = np.arctan2(fiber_v, fiber_u)
    axial = np.sum(np.exp(2.0j * angles))
    predicted = float(np.degrees(0.5 * np.angle(axial)) % 180.0)
    observed_value = planes[best_index]["dominantFiberAngleDeg"]
    observed = float(observed_value) if observed_value is not None else None
    residual = _axial_angle_difference(predicted, observed) if observed is not None else None
    best_depth = float(depth_offsets[best_index])
    alignment = (
        float(np.exp(-0.5 * (float(residual) / 15.0) ** 2))
        if residual is not None
        else 0.0
    )
    depth_alignment = float(np.exp(-0.5 * (best_depth / 4.0) ** 2))
    center_values = stack[center_index][region]
    return {
        "expandedSupportPixelCount": supported_pixels,
        "expandedSupportFraction": round(
            supported_pixels / max(int(gap["pixelCount"]), 1), 4
        ),
        "materialFraction": round(
            float(np.mean(center_values > float(air_threshold))), 4
        ),
        "bestDepthOffsetVoxels": best_depth,
        "bestTextureScore": round(float(scores[best_index]), 4),
        "centerTextureScore": round(float(scores[center_index]), 4),
        "predictedFiberAngleDeg": round(predicted, 3),
        "observedFiberAngleDeg": round(observed, 3) if observed is not None else None,
        "fiberAngleResidualDeg": round(float(residual), 3) if residual is not None else None,
        "depthAlignedTextureScore": round(
            float(scores[best_index]) * alignment * depth_alignment, 4
        ),
    }


def _state_gap_proposals(
    member_indices: np.ndarray,
    owner: np.ndarray,
    arrays: dict[str, np.ndarray],
    flakes: list[dict[str, Any]],
    cell_shape: np.ndarray,
    minimum_gap_area_square_voxels: float,
    score_threshold: float,
    minimum_margin: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    members = np.asarray(member_indices, dtype=np.int64)
    carrier = _mls_carrier([flakes[int(index)] for index in members])
    gaps = _internal_gap_components(
        carrier["supportMask"],
        float(carrier["stats"]["pixelStepVoxels"]),
        minimum_gap_area_square_voxels,
    )
    if not gaps:
        return [], {
            "gapCount": 0,
            "gapAreaSquareVoxels": 0.0,
            "candidateFlakeCount": 0,
            "compatibleFlakeCount": 0,
            "claimedCompatibleFlakeCount": 0,
            "eligibleCellCount": 0,
            "gaps": [],
        }
    gap_label = np.zeros(carrier["supportMask"].shape, dtype=np.int32)
    for gap in gaps:
        gap_label[gap["mask"]] = int(gap["gapId"])
    y, x, inside = _project_flakes_to_carrier(carrier, arrays["center"])
    candidates = np.flatnonzero(inside)
    labels = gap_label[y[candidates], x[candidates]]
    keep = (
        (labels > 0)
        & ~np.isin(candidates, members)
        & (arrays["normalFamily"][candidates] == 0)
    )
    candidates = candidates[keep]
    labels = labels[keep]
    if not len(candidates):
        compact_gaps = [{key: value for key, value in gap.items() if key != "mask"} for gap in gaps]
        return [], {
            "gapCount": len(gaps),
            "gapAreaSquareVoxels": round(sum(float(gap["areaSquareVoxels"]) for gap in gaps), 2),
            "candidateFlakeCount": 0,
            "compatibleFlakeCount": 0,
            "claimedCompatibleFlakeCount": 0,
            "eligibleCellCount": 0,
            "gaps": compact_gaps,
        }
    scored = _score_growth_candidates(members, candidates, arrays, flakes)
    codes = _cell_codes(arrays["cell"][candidates], cell_shape)
    proposals: list[dict[str, Any]] = []
    compatible = scored["score"] >= score_threshold
    for code in np.unique(codes):
        positions = np.flatnonzero(codes == code)
        order = positions[np.argsort(scored["score"][positions])[::-1]]
        best = int(order[0])
        second_score = float(scored["score"][order[1]]) if len(order) > 1 else 0.0
        best_score = float(scored["score"][best])
        flake_index = int(candidates[best])
        if (
            best_score < score_threshold
            or best_score - second_score < minimum_margin
            or int(owner[flake_index]) >= 0
        ):
            continue
        proposals.append(
            {
                "flake": flake_index,
                "cell": tuple(int(value) for value in arrays["cell"][flake_index]),
                "gapId": int(labels[best]),
                "score": best_score,
                "margin": best_score - second_score,
                "heightResidual": float(scored["heightResidual"][best]),
                "normalAngle": float(scored["normalAngle"][best]),
                "fiberAngle": float(scored["fiberAngle"][best]),
                "nearestPlanarDistance": float(scored["nearestPlanarDistance"][best]),
            }
        )
    compact_gaps = [{key: value for key, value in gap.items() if key != "mask"} for gap in gaps]
    return proposals, {
        "gapCount": len(gaps),
        "gapAreaSquareVoxels": round(sum(float(gap["areaSquareVoxels"]) for gap in gaps), 2),
        "candidateFlakeCount": int(len(candidates)),
        "compatibleFlakeCount": int(np.count_nonzero(compatible)),
        "claimedCompatibleFlakeCount": int(
            np.count_nonzero(compatible & (owner[candidates] >= 0))
        ),
        "eligibleCellCount": len(proposals),
        "gaps": compact_gaps,
    }


def fill_carrier_gaps(
    output_root: str | Path,
    top_count: int = 24,
    maximum_rounds: int = 4,
    minimum_gap_area_square_voxels: float = 256.0,
    score_threshold: float = 0.62,
    minimum_margin: float = 0.04,
    force: bool = False,
) -> dict[str, Any]:
    """Grow only into enclosed carrier holes using existing Acus evidence."""
    root = Path(output_root)
    summary_path = root / f"sheetlet-carrier-gaps-v{CARRIER_GAP_VERSION}.json"
    artifact_path = root / f"sheetlet-carrier-gaps-v{CARRIER_GAP_VERSION}.npz"
    settings = {
        "topCount": int(top_count),
        "maximumRounds": int(maximum_rounds),
        "minimumGapAreaSquareVoxels": float(minimum_gap_area_square_voxels),
        "scoreThreshold": float(score_threshold),
        "minimumBestVsSecondMargin": float(minimum_margin),
        "scope": "fully enclosed unsupported carrier regions only",
        "normalFamilyConstraint": "legacy primary family only",
    }
    bridge_path = root / f"sheetlet-carrier-bridges-v{CARRIER_BRIDGE_VERSION}.json"
    bridges = json.loads(bridge_path.read_text())
    identity = {
        "version": CARRIER_GAP_VERSION,
        "source": bridges["identity"]["source"],
        "bridgeIdentity": bridges["identity"],
        "input": str(bridge_path),
    }
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity and cached.get("settings") == settings:
            return cached
    with np.load(root / bridges["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"], dtype=np.uint32)
        member_offset = np.asarray(payload["memberOffset"], dtype=np.uint64)
    _, _, _, flakes = _load_carrier_catalog(root)
    arrays = _flake_arrays(flakes)
    cell_shape = np.max(arrays["cell"], axis=0) + 1
    states = []
    owner = np.full(len(flakes), -1, dtype=np.int32)
    for state_index, state in enumerate(bridges["states"]):
        low, high = int(member_offset[state_index]), int(member_offset[state_index + 1])
        members = set(int(value) for value in member_index[low:high])
        occupied = {tuple(int(value) for value in arrays["cell"][index]) for index in members}
        for index in members:
            if owner[index] >= 0:
                raise ValueError("bridge catalog contains repeated flake assignments")
            owner[index] = state_index
        states.append(
            {
                "input": state,
                "members": members,
                "initialMembers": set(members),
                "occupiedCells": occupied,
                "rounds": [],
            }
        )

    selected_count = min(max(1, int(top_count)), len(states))
    started = time.monotonic()
    rounds = []
    for round_index in range(1, max(1, int(maximum_rounds)) + 1):
        proposals = []
        for state_index, state in enumerate(states[:selected_count]):
            state_proposals, diagnostics = _state_gap_proposals(
                np.asarray(sorted(state["members"]), dtype=np.int64),
                owner,
                arrays,
                flakes,
                cell_shape,
                minimum_gap_area_square_voxels,
                score_threshold,
                minimum_margin,
            )
            for proposal in state_proposals:
                proposal["state"] = state_index
                proposals.append(proposal)
            state["rounds"].append(
                {
                    "round": round_index,
                    **diagnostics,
                    "acceptedCount": 0,
                }
            )
        accepted = []
        for proposal in sorted(proposals, key=lambda value: float(value["score"]), reverse=True):
            state_index = int(proposal["state"])
            state = states[state_index]
            flake_index = int(proposal["flake"])
            cell = proposal["cell"]
            if owner[flake_index] >= 0 or cell in state["occupiedCells"]:
                continue
            state["members"].add(flake_index)
            state["occupiedCells"].add(cell)
            state["rounds"][-1]["acceptedCount"] += 1
            owner[flake_index] = state_index
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
        if not accepted:
            break

    values = []
    offsets = [0]
    state_summaries = []
    for state_index, state in enumerate(states):
        members = np.asarray(sorted(state["members"]), dtype=np.uint32)
        values.append(members)
        offsets.append(offsets[-1] + len(members))
        added = len(state["members"] - state["initialMembers"])
        summary = {
            **state["input"],
            "inputBridgeRank": int(state["input"]["rank"]),
            "initialFlakeCount": len(state["initialMembers"]),
            "flakeCount": len(state["members"]),
            "uniqueCellCount": len(state["occupiedCells"]),
            "gapAddedFlakeCount": added,
        }
        if state_index < selected_count:
            final_carrier = _mls_carrier([flakes[int(index)] for index in members])
            final_gaps = _internal_gap_components(
                final_carrier["supportMask"],
                float(final_carrier["stats"]["pixelStepVoxels"]),
                minimum_gap_area_square_voxels,
            )
            summary["gapFill"] = {
                "rounds": state["rounds"],
                "finalGapCount": len(final_gaps),
                "finalGapAreaSquareVoxels": round(
                    sum(float(gap["areaSquareVoxels"]) for gap in final_gaps), 2
                ),
            }
        state_summaries.append(summary)

    repeated = int(len(np.concatenate(values)) - len(np.unique(np.concatenate(values))))
    cell_collisions = sum(
        len(state["members"]) - len(state["occupiedCells"]) for state in states
    )
    _atomic_npz(
        artifact_path,
        memberIndex=np.concatenate(values),
        memberOffset=np.asarray(offsets, dtype=np.uint64),
    )
    result = {
        "identity": identity,
        "settings": settings,
        "artifact": artifact_path.name,
        "states": state_summaries,
        "rounds": rounds,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "inputStateCount": len(states),
            "auditedStateCount": selected_count,
            "inputFlakeCount": int(len(member_index)),
            "outputFlakeCount": int(sum(len(state["members"]) for state in states)),
            "addedFlakeCount": int(sum(len(state["members"] - state["initialMembers"]) for state in states)),
            "repeatedFlakeAssignmentCount": repeated,
            "sameSheetCellCollisionCount": int(cell_collisions),
        },
    }
    _atomic_json(summary_path, result)
    return result


def build_gap_fill_previews(
    output_root: str | Path,
    top_count: int = 12,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    gaps = json.loads((root / f"sheetlet-carrier-gaps-v{CARRIER_GAP_VERSION}.json").read_text())
    with np.load(root / gaps["artifact"]) as payload:
        member_index = np.asarray(payload["memberIndex"], dtype=np.uint32)
        member_offset = np.asarray(payload["memberOffset"], dtype=np.uint64)
    bridges = json.loads(
        (
            root / f"sheetlet-carrier-bridges-v{CARRIER_BRIDGE_VERSION}.json"
        ).read_text()
    )
    with np.load(root / bridges["artifact"]) as payload:
        initial_index = np.asarray(payload["memberIndex"], dtype=np.uint32)
        initial_offset = np.asarray(payload["memberOffset"], dtype=np.uint64)
    source_path, _, _, flakes = _load_carrier_catalog(root)
    source = np.load(source_path, mmap_mode="r")
    analysis = json.loads((root / "analysis.json").read_text())
    air_threshold = float(analysis["normalization"]["airThreshold"])
    depth_offsets = np.arange(-12.0, 12.01, 1.0, dtype=np.float32)
    selected_count = min(max(1, int(top_count)), int(gaps["settings"]["topCount"]))
    artifact_root = root / f"sheetlet-gaps-v{CARRIER_GAP_VERSION}"
    artifact_root.mkdir(parents=True, exist_ok=True)
    summary_path = artifact_root / f"summary-top{selected_count}.json"
    if summary_path.is_file() and not force:
        return json.loads(summary_path.read_text())
    outputs = []
    started = time.monotonic()
    for output_rank in range(1, selected_count + 1):
        state_index = output_rank - 1
        low, high = int(member_offset[state_index]), int(member_offset[state_index + 1])
        members = member_index[low:high]
        initial_low, initial_high = int(initial_offset[state_index]), int(initial_offset[state_index + 1])
        initial_members = initial_index[initial_low:initial_high]
        initial_carrier = _mls_carrier([flakes[int(index)] for index in initial_members])
        expanded_initial_carrier = _mls_carrier(
            [flakes[int(index)] for index in initial_members], support_radius=112.0
        )
        carrier = _mls_carrier([flakes[int(index)] for index in members])
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
            memberIndex=members,
        )
        _atomic_npz(
            candidate_root / "depth-stack.npz",
            depthOffsets=depth_offsets,
            intensity=stack,
        )
        mask = carrier["supportMask"]
        center_index = int(np.argmin(np.abs(depth_offsets)))
        best_index = int(np.argmin(np.abs(depth_offsets - texture["bestDepthOffsetVoxels"])))
        gap_map = np.zeros(mask.shape, dtype=np.uint8)
        initial_mask = initial_carrier["supportMask"]
        gap_map[initial_mask] = 48
        initial_gaps = _internal_gap_components(
            initial_mask,
            float(initial_carrier["stats"]["pixelStepVoxels"]),
            float(gaps["settings"]["minimumGapAreaSquareVoxels"]),
        )
        gap_depth_offsets = np.arange(-6.0, 6.01, 1.0, dtype=np.float32)
        expanded_stack, _ = _sample_stack(
            source, expanded_initial_carrier, gap_depth_offsets
        )
        gap_evidence = [
            {
                **{key: value for key, value in gap.items() if key != "mask"},
                "ctEvidence": _gap_ct_evidence(
                    expanded_initial_carrier,
                    expanded_stack,
                    gap_depth_offsets,
                    gap,
                    air_threshold,
                ),
            }
            for gap in initial_gaps
        ]
        initial_gap_mask = np.zeros(mask.shape, dtype=bool)
        for gap in initial_gaps:
            initial_gap_mask |= gap["mask"]
        gap_map[initial_gap_mask] = 160
        newly_supported = initial_gap_mask & mask
        gap_map[newly_supported] = 255
        images = {
            "center.png": _contrast(stack[center_index], mask),
            "best-texture.png": _contrast(stack[best_index], mask),
            "gap-map.png": gap_map,
        }
        for filename, image in images.items():
            (candidate_root / filename).write_bytes(grayscale_png(image))
        output = {
            "state": gaps["states"][state_index],
            "carrier": carrier["stats"],
            "texture": {key: texture[key] for key in texture if key != "planes"},
            "yield": yield_stats,
            "sampling": sampling,
            "gapPixels": {
                "initial": int(np.count_nonzero(initial_gap_mask)),
                "newlySupported": int(np.count_nonzero(newly_supported)),
                "remaining": int(np.count_nonzero(initial_gap_mask & ~mask)),
            },
            "gapEvidence": gap_evidence,
            "artifacts": {
                "bestTextureImage": str((candidate_root / "best-texture.png").relative_to(root)),
                "gapMap": str((candidate_root / "gap-map.png").relative_to(root)),
            },
        }
        _atomic_json(candidate_root / "summary.json", output)
        outputs.append(output)
    result = {
        "identity": {"version": CARRIER_GAP_VERSION},
        "settings": {"topCount": selected_count},
        "stats": {
            **gaps["stats"],
            "previewElapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        },
        "candidates": outputs,
    }
    _atomic_json(summary_path, result)
    return result
