from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .acus import _plane_basis
from .slab_flakes import (
    FLAKE_CACHE_VERSION,
    _match_flakes_at_step,
    _track_sizes,
    slab_flake_plane,
)


FLAKE_AUDIT_VERSION = 1


def _atomic_compact_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _median(values: list[float], digits: int = 4) -> float | None:
    if not values:
        return None
    return round(float(np.median(np.asarray(values, dtype=np.float64))), digits)


def _cell_pair_count(flakes: list[dict[str, Any]], cell_step: int) -> int:
    cells = {
        (int(flake["cellIndex"][0]), int(flake["cellIndex"][1]))
        for flake in flakes
    }
    return sum(
        1
        for cell_x, cell_y in cells
        for offset_x, offset_y in ((cell_step, 0), (0, cell_step))
        if (cell_x + offset_x, cell_y + offset_y) in cells
    )


def _summary(
    flakes: list[dict[str, Any]],
    links: list[dict[str, Any]],
    cell_pair_count: int,
    minimum_score: float,
) -> dict[str, Any]:
    accepted = [link for link in links if float(link["score"]) >= minimum_score]
    track_sizes = [
        size
        for size in _track_sizes(len(flakes), links, minimum_score)
        if size >= 2
    ]
    linked_flakes = {
        int(endpoint)
        for link in accepted
        for endpoint in (link["source"], link["target"])
    }
    return {
        "cellPairCount": cell_pair_count,
        "mutualLinkCount": len(links),
        "acceptedLinkCount": len(accepted),
        "acceptedLinksPerCellPair": round(
            len(accepted) / max(cell_pair_count, 1), 4
        ),
        "linkedFlakeFraction": round(len(linked_flakes) / max(len(flakes), 1), 4),
        "linkedTrackCount": len(track_sizes),
        "largestTrackSize": max(track_sizes, default=0),
        "medianTrackSize": _median([float(value) for value in track_sizes], 2),
        "medianScore": _median([float(link["score"]) for link in accepted]),
        "medianRawCompatibility": _median(
            [float(link["rawCompatibility"]) for link in accepted]
        ),
        "medianPositionResidualVoxels": _median(
            [float(link["positionResidualVoxels"]) for link in accepted], 3
        ),
        "medianNormalAngleDeg": _median(
            [float(link["normalAngleDeg"]) for link in accepted], 3
        ),
        "medianFiberAngleDeg": _median(
            [float(link["fiberAngleDeg"]) for link in accepted], 3
        ),
        "medianSharedNeedleFraction": _median(
            [float(link["sharedNeedleFraction"]) for link in accepted]
        ),
    }


def _aggregate_null(summaries: list[dict[str, Any]]) -> dict[str, Any]:
    if not summaries:
        return {"repetitions": 0}
    numeric_keys = [
        key
        for key, value in summaries[0].items()
        if isinstance(value, (int, float)) and value is not None
    ]
    result: dict[str, Any] = {"repetitions": len(summaries)}
    for key in numeric_keys:
        values = [float(summary[key]) for summary in summaries if summary.get(key) is not None]
        result[key] = _median(values, 4)
        if values:
            result[f"{key}P10"] = round(float(np.percentile(values, 10.0)), 4)
            result[f"{key}P90"] = round(float(np.percentile(values, 90.0)), 4)
    return result


def _copy_with_fiber_shuffle(
    flakes: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    axial_angles = []
    for flake in flakes:
        normal = np.asarray(flake["normal"], dtype=np.float32)
        u_axis, v_axis = _plane_basis(normal)
        fiber = np.asarray(flake["fiber"], dtype=np.float32)
        axial_angles.append(math.atan2(float(np.dot(fiber, v_axis)), float(np.dot(fiber, u_axis))) % math.pi)
    shuffled_angles = np.asarray(axial_angles, dtype=np.float64)[rng.permutation(len(flakes))]
    output = []
    for flake, angle in zip(flakes, shuffled_angles):
        copied = dict(flake)
        normal = np.asarray(flake["normal"], dtype=np.float32)
        u_axis, v_axis = _plane_basis(normal)
        fiber = math.cos(float(angle)) * u_axis + math.sin(float(angle)) * v_axis
        fiber /= max(float(np.linalg.norm(fiber)), 1.0e-8)
        dominant = int(np.argmax(np.abs(fiber)))
        if fiber[dominant] < 0.0:
            fiber = -fiber
        copied["fiber"] = fiber.tolist()
        output.append(copied)
    return output


def _copy_with_depth_shuffle(
    flakes: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    shuffled_depths = np.asarray(
        [float(flake["depthOffset"]) for flake in flakes], dtype=np.float32
    )[rng.permutation(len(flakes))]
    output = []
    for flake, shuffled_depth in zip(flakes, shuffled_depths):
        copied = dict(flake)
        normal = np.asarray(flake["normal"], dtype=np.float32)
        center = np.asarray(flake["center"], dtype=np.float32)
        center += normal * (float(shuffled_depth) - float(flake["depthOffset"]))
        copied["center"] = center.tolist()
        copied["depthOffset"] = float(shuffled_depth)
        output.append(copied)
    return output


def _copy_with_spatial_shuffle(
    flakes: list[dict[str, Any]], rng: np.random.Generator
) -> list[dict[str, Any]]:
    by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    cell_centers: dict[tuple[int, int], np.ndarray] = {}
    cell_z_indices: dict[tuple[int, int], int] = {}
    for flake in flakes:
        key = (int(flake["cellIndex"][0]), int(flake["cellIndex"][1]))
        by_cell.setdefault(key, []).append(flake)
        cell_centers[key] = np.asarray(flake["cellCenter"], dtype=np.float32)
        cell_z_indices[key] = int(flake["cellIndex"][2])
    source_keys = sorted(by_cell)
    target_keys = [source_keys[index] for index in rng.permutation(len(source_keys))]
    output = []
    for source_key, target_key in zip(source_keys, target_keys):
        source_center = cell_centers[source_key]
        target_center = cell_centers[target_key]
        delta = target_center - source_center
        for flake in by_cell[source_key]:
            copied = dict(flake)
            copied["cellIndex"] = [
                target_key[0],
                target_key[1],
                cell_z_indices[target_key],
            ]
            copied["cellCenter"] = target_center.tolist()
            copied["center"] = (
                np.asarray(flake["center"], dtype=np.float32) + delta
            ).tolist()
            output.append(copied)
    for index, flake in enumerate(output):
        flake["id"] = index
    return output


def _attach_memberships(
    root: Path, flakes: list[dict[str, Any]], z_index: int, maximum_flakes: int
) -> None:
    membership_path = root / (
        f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k{maximum_flakes}-members.npz"
    )
    with np.load(membership_path) as membership:
        offsets = np.asarray(membership["offsets"], dtype=np.int64)
        ids = np.asarray(membership["ids"], dtype=np.uint32)
    if len(offsets) != len(flakes) + 1:
        raise ValueError("flake membership cache does not match the flake plane")
    for index, flake in enumerate(flakes):
        flake["_needleIds"] = set(
            int(value) for value in ids[offsets[index] : offsets[index + 1]]
        )


def _null_sweeps(
    flakes: list[dict[str, Any]],
    steps: list[int],
    minimum_score: float,
    repetitions: int,
    seed: int,
) -> dict[str, dict[int, list[dict[str, Any]]]]:
    factories: dict[str, Callable[[list[dict[str, Any]], np.random.Generator], list[dict[str, Any]]]] = {
        "fiber": _copy_with_fiber_shuffle,
        "depth": _copy_with_depth_shuffle,
        "spatial": _copy_with_spatial_shuffle,
    }
    output = {
        name: {step: [] for step in steps}
        for name in factories
    }
    for repetition in range(repetitions):
        for null_index, (name, factory) in enumerate(factories.items()):
            rng = np.random.default_rng(seed + repetition * 101 + null_index * 10007)
            shuffled = factory(flakes, rng)
            for step in steps:
                cell_pairs = _cell_pair_count(shuffled, step)
                links = _match_flakes_at_step(shuffled, step)
                output[name][step].append(
                    _summary(shuffled, links, cell_pairs, minimum_score)
                )
    return output


def slab_flake_audit(
    output_root: str | Path,
    z_index: int,
    repetitions: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    repetitions = int(np.clip(repetitions, 2, 8))
    maximum_flakes = 3
    flake_result = slab_flake_plane(root, z_index, maximum_flakes)
    minimum_score = float(flake_result["settings"]["defaultTrackScore"])
    grid_stride = int(flake_result["settings"]["gridStride"])
    cube_size = int(flake_result["settings"]["cubeSize"])
    steps = [1, 2, 3]
    identity = {
        "version": FLAKE_AUDIT_VERSION,
        "flakeIdentity": flake_result["identity"],
        "repetitions": repetitions,
        "minimumScore": minimum_score,
        "steps": steps,
    }
    cache_path = root / f"flake-audit-v{FLAKE_AUDIT_VERSION}-z{z_index}-r{repetitions}.json"
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    flakes = [dict(flake) for flake in flake_result["flakes"]]
    _attach_memberships(root, flakes, z_index, maximum_flakes)
    observed: dict[int, tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    for step in steps:
        cell_pairs = _cell_pair_count(flakes, step)
        links = _match_flakes_at_step(flakes, step)
        observed[step] = (
            links,
            _summary(flakes, links, cell_pairs, minimum_score),
        )
    nulls = _null_sweeps(
        flakes,
        steps,
        minimum_score,
        repetitions,
        seed=358 + z_index * 1009,
    )
    base_density = max(
        float(observed[1][1]["acceptedLinksPerCellPair"]), 1.0e-8
    )
    sweeps = []
    for step in steps:
        spacing = step * grid_stride
        observed_links, observed_summary = observed[step]
        fiber_null = _aggregate_null(nulls["fiber"][step])
        depth_null = _aggregate_null(nulls["depth"][step])
        spatial_null = _aggregate_null(nulls["spatial"][step])
        density = float(observed_summary["acceptedLinksPerCellPair"])
        spatial_density = float(spatial_null.get("acceptedLinksPerCellPair") or 0.0)
        fiber_density = float(fiber_null.get("acceptedLinksPerCellPair") or 0.0)
        sweeps.append(
            {
                "cellStep": step,
                "spacingVoxels": spacing,
                "overlapFraction": round(max(0.0, 1.0 - spacing / cube_size), 4),
                "gapVoxels": max(0, spacing - cube_size),
                "independentWindows": spacing >= cube_size,
                "linkSurvivalVs32": round(density / base_density, 4),
                "spatialNullDensityRatio": round(
                    density / max(spatial_density, 1.0e-8), 4
                ),
                "fiberNullDensityRatio": round(
                    density / max(fiber_density, 1.0e-8), 4
                ),
                "observed": observed_summary,
                "nulls": {
                    "fiber": fiber_null,
                    "depth": depth_null,
                    "spatial": spatial_null,
                },
                "links": observed_links,
            }
        )
    for flake in flakes:
        flake.pop("_needleIds", None)
    result = {
        "identity": identity,
        "view": flake_result["view"],
        "settings": {
            "gridStride": grid_stride,
            "cubeSize": cube_size,
            "minimumLinkScore": minimum_score,
            "repetitions": repetitions,
            "nulls": ["fiber-rematched", "depth-rematched", "spatial-rematched"],
        },
        "sweeps": sweeps,
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "flakeCount": len(flakes),
            "constraint": (
                "all nulls rerun mutual matching; 64- and 96-voxel links use "
                "non-overlapping Acus windows"
            ),
        },
    }
    _atomic_compact_json(cache_path, result)
    return result
