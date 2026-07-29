from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any

import numpy as np

from .acus import _plane_basis
from .slab_flakes import (
    FLAKE_CACHE_VERSION,
    _axial_angle_degrees,
    _catalog_records_for_cell,
    _fit_cell_flakes,
    slab_flake_plane,
)


FLAKE_HOLDOUT_VERSION = 1


def _atomic_compact_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, separators=(",", ":")) + "\n")
    temporary.replace(path)


def _median(values: list[float], digits: int = 4) -> float | None:
    if not values:
        return None
    return round(float(np.median(np.asarray(values, dtype=np.float64))), digits)


def _stable_fold(record_ids: np.ndarray, seed: int) -> np.ndarray:
    """Split catalog records without using their score, position, or orientation."""
    values = np.asarray(record_ids, dtype=np.uint64) + np.uint64(seed)
    values = (values ^ (values >> np.uint64(30))) * np.uint64(0xBF58476D1CE4E5B9)
    values = (values ^ (values >> np.uint64(27))) * np.uint64(0x94D049BB133111EB)
    values ^= values >> np.uint64(31)
    return (values & np.uint64(1)).astype(bool)


def _replication_metrics(first: dict[str, Any], second: dict[str, Any]) -> dict[str, float]:
    first_normal = np.asarray(first["normal"], dtype=np.float32)
    second_normal = np.asarray(second["normal"], dtype=np.float32)
    if float(np.dot(first_normal, second_normal)) < 0.0:
        second_normal = -second_normal
    normal_angle = _axial_angle_degrees(first_normal, second_normal)
    fiber_angle = _axial_angle_degrees(
        np.asarray(first["fiber"], dtype=np.float32),
        np.asarray(second["fiber"], dtype=np.float32),
    )
    delta = np.asarray(second["center"], dtype=np.float32) - np.asarray(
        first["center"], dtype=np.float32
    )
    position_residual = 0.5 * (
        abs(float(np.dot(first_normal, delta)))
        + abs(float(np.dot(second_normal, delta)))
    )
    depth_delta = abs(float(first["depthOffset"]) - float(second["depthOffset"]))
    quality = math.sqrt(float(first["quality"]) * float(second["quality"]))
    score = quality * math.exp(
        -0.5
        * (
            (position_residual / 4.0) ** 2
            + (normal_angle / 8.0) ** 2
            + (fiber_angle / 12.0) ** 2
        )
    )
    return {
        "score": score,
        "positionResidual": position_residual,
        "depthDelta": depth_delta,
        "normalAngle": normal_angle,
        "fiberAngle": fiber_angle,
    }


def _mutual_cell_matches(
    first: list[dict[str, Any]], second: list[dict[str, Any]]
) -> list[dict[str, Any]]:
    if not first or not second:
        return []
    candidates: list[tuple[int, int, dict[str, float]]] = []
    for first_index, first_flake in enumerate(first):
        for second_index, second_flake in enumerate(second):
            metrics = _replication_metrics(first_flake, second_flake)
            if (
                metrics["positionResidual"] <= 12.0
                and metrics["normalAngle"] <= 16.0
                and metrics["fiberAngle"] <= 36.0
                and metrics["score"] >= 0.02
            ):
                candidates.append((first_index, second_index, metrics))
    best_first: dict[int, tuple[int, int, dict[str, float]]] = {}
    best_second: dict[int, tuple[int, int, dict[str, float]]] = {}
    for candidate in sorted(candidates, key=lambda value: value[2]["score"], reverse=True):
        best_first.setdefault(candidate[0], candidate)
        best_second.setdefault(candidate[1], candidate)
    return [
        {
            "first": first_index,
            "second": second_index,
            **metrics,
        }
        for first_index, second_index, metrics in best_first.values()
        if best_second.get(second_index, (-1, -1, {}))[0] == first_index
    ]


def _representative(
    first: dict[str, Any], second: dict[str, Any], match: dict[str, Any]
) -> dict[str, Any]:
    normal = np.asarray(first["normal"], dtype=np.float32)
    normal /= max(float(np.linalg.norm(normal)), 1.0e-8)
    fiber_first = np.asarray(first["fiber"], dtype=np.float32)
    fiber_second = np.asarray(second["fiber"], dtype=np.float32)
    if float(np.dot(fiber_first, fiber_second)) < 0.0:
        fiber_second = -fiber_second
    fiber = fiber_first + fiber_second
    fiber -= normal * float(np.dot(fiber, normal))
    fiber /= max(float(np.linalg.norm(fiber)), 1.0e-8)
    return {
        "normal": normal.tolist(),
        "fiber": fiber.tolist(),
        "center": (
            0.5
            * (
                np.asarray(first["center"], dtype=np.float32)
                + np.asarray(second["center"], dtype=np.float32)
            )
        ).tolist(),
        "depthOffset": 0.5
        * (float(first["depthOffset"]) + float(second["depthOffset"])),
        "quality": math.sqrt(float(first["quality"]) * float(second["quality"])),
        "foldScore": float(match["score"]),
        "foldDepthDelta": float(match["depthDelta"]),
        "foldPositionResidual": float(match["positionResidual"]),
        "foldFiberDelta": float(match["fiberAngle"]),
        "supportA": float(first["effectiveSupport"]),
        "supportB": float(second["effectiveSupport"]),
        "needleCountA": int(first["needleCount"]),
        "needleCountB": int(second["needleCount"]),
    }


def _shuffle_fold_modes(
    by_cell: dict[tuple[int, int], list[dict[str, Any]]], rng: np.random.Generator
) -> dict[tuple[int, int], list[dict[str, Any]]]:
    indexed = [(cell, flake) for cell, flakes in by_cell.items() for flake in flakes]
    if not indexed:
        return {}
    donor_order = rng.permutation(len(indexed))
    output: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for target_index, (cell, target) in enumerate(indexed):
        donor = indexed[int(donor_order[target_index])][1]
        copied = dict(target)
        normal = np.asarray(target["normal"], dtype=np.float32)
        u_axis, v_axis = _plane_basis(normal)
        donor_normal = np.asarray(donor["normal"], dtype=np.float32)
        donor_u, donor_v = _plane_basis(donor_normal)
        donor_fiber = np.asarray(donor["fiber"], dtype=np.float32)
        donor_angle = math.atan2(
            float(np.dot(donor_fiber, donor_v)), float(np.dot(donor_fiber, donor_u))
        ) % math.pi
        fiber = math.cos(donor_angle) * u_axis + math.sin(donor_angle) * v_axis
        fiber /= max(float(np.linalg.norm(fiber)), 1.0e-8)
        donor_depth = float(donor["depthOffset"])
        center = np.asarray(target["center"], dtype=np.float32)
        center += normal * (donor_depth - float(target["depthOffset"]))
        copied["fiber"] = fiber.tolist()
        copied["center"] = center.tolist()
        copied["depthOffset"] = donor_depth
        copied["quality"] = donor["quality"]
        copied["effectiveSupport"] = donor["effectiveSupport"]
        copied["needleCount"] = donor["needleCount"]
        output.setdefault(cell, []).append(copied)
    return output


def _match_fold_maps(
    first_by_cell: dict[tuple[int, int], list[dict[str, Any]]],
    second_by_cell: dict[tuple[int, int], list[dict[str, Any]]],
) -> list[tuple[tuple[int, int], dict[str, Any], dict[str, Any], dict[str, Any]]]:
    matches = []
    for cell, first in first_by_cell.items():
        second = second_by_cell.get(cell, [])
        for match in _mutual_cell_matches(first, second):
            matches.append(
                (
                    cell,
                    first[int(match["first"])],
                    second[int(match["second"])],
                    match,
                )
            )
    return matches


def slab_flake_holdout(
    output_root: str | Path,
    z_index: int,
    repetitions: int = 4,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    repetitions = int(np.clip(repetitions, 2, 8))
    maximum_flakes = 3
    full_result = slab_flake_plane(root, z_index, maximum_flakes)
    identity = {
        "version": FLAKE_HOLDOUT_VERSION,
        "flakeCacheVersion": FLAKE_CACHE_VERSION,
        "flakeIdentity": full_result["identity"],
        "repetitions": repetitions,
        "splitSeed": 0xAC05 + z_index * 4099,
    }
    cache_path = root / f"flake-holdout-v{FLAKE_HOLDOUT_VERSION}-z{z_index}-r{repetitions}.json"
    if cache_path.is_file() and not force:
        cached = json.loads(cache_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    analysis = json.loads((root / "analysis.json").read_text())
    grid = json.loads((root / "grid.json").read_text())
    settings = analysis["identity"]["settings"]
    cells = np.load(root / "cells.npy", mmap_mode="r")
    catalog = np.load(root / "needles.npy", mmap_mode="r")
    counts = np.load(root / "needle-counts.npy", mmap_mode="r")
    bin_shape_zyx = tuple(int(value) for value in analysis["binShapeZYX"])
    split_seed = int(identity["splitSeed"])
    first_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    second_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    eligible_cell_count = 0
    for cell_y, center_y in enumerate(grid["y"]):
        for cell_x, center_x in enumerate(grid["x"]):
            cell = cells[z_index, cell_y, cell_x]
            if not bool(cell["valid"]):
                continue
            cell_center = np.asarray(
                [center_x, center_y, grid["z"][z_index]], dtype=np.float32
            )
            records, record_ids = _catalog_records_for_cell(
                catalog, counts, cell_center, settings, bin_shape_zyx
            )
            fold = _stable_fold(record_ids, split_seed)
            if int(np.count_nonzero(fold)) < 4 or int(np.count_nonzero(~fold)) < 4:
                continue
            eligible_cell_count += 1
            fit_arguments = (
                cell_center,
                np.asarray(cell["normal"], dtype=np.float32),
                float(cell["normalConfidence"]),
                (cell_x, cell_y, z_index),
                int(settings["cubeSize"]),
                maximum_flakes,
                4.0,
                12.0,
                4,
            )
            first = _fit_cell_flakes(records[fold], record_ids[fold], *fit_arguments)
            second = _fit_cell_flakes(records[~fold], record_ids[~fold], *fit_arguments)
            key = (cell_x, cell_y)
            if first:
                first_by_cell[key] = first
            if second:
                second_by_cell[key] = second

    observed = _match_fold_maps(first_by_cell, second_by_cell)
    representatives_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    validated_pairs = []
    for cell, first, second, match in observed:
        representative = _representative(first, second, match)
        representative["validated"] = bool(
            float(match["score"]) >= 0.08
            and min(
                float(first["effectiveSupport"]), float(second["effectiveSupport"])
            )
            >= 3.0
        )
        representatives_by_cell.setdefault(cell, []).append(representative)
        if representative["validated"]:
            validated_pairs.append(representative)

    null_pair_counts: list[float] = []
    null_validated_counts: list[float] = []
    null_fiber_angles: list[float] = []
    rng = np.random.default_rng(split_seed + 0x51E37)
    for _ in range(repetitions):
        shuffled = _shuffle_fold_modes(second_by_cell, rng)
        null_matches = _match_fold_maps(first_by_cell, shuffled)
        null_pair_counts.append(float(len(null_matches)))
        validated = [
            match
            for _, first, second, match in null_matches
            if float(match["score"]) >= 0.08
            and min(float(first["effectiveSupport"]), float(second["effectiveSupport"]))
            >= 3.0
        ]
        null_validated_counts.append(float(len(validated)))
        null_fiber_angles.extend(float(match["fiberAngle"]) for match in validated)

    full_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for flake in full_result["flakes"]:
        full_by_cell.setdefault(
            (int(flake["cellIndex"][0]), int(flake["cellIndex"][1])), []
        ).append(flake)
    validation_by_flake: list[dict[str, Any]] = []
    validated_full_count = 0
    for cell, full_flakes in full_by_cell.items():
        representatives = representatives_by_cell.get(cell, [])
        matched_by_full = {
            int(match["first"]): (int(match["second"]), match)
            for match in _mutual_cell_matches(full_flakes, representatives)
        }
        for full_index, flake in enumerate(full_flakes):
            mapped = matched_by_full.get(full_index)
            if mapped is None:
                validation_by_flake.append(
                    {"flakeId": int(flake["id"]), "validated": False, "validationScore": 0.0}
                )
                continue
            representative_index, mapping = mapped
            representative = representatives[representative_index]
            validated = bool(representative["validated"] and float(mapping["score"]) >= 0.06)
            validated_full_count += int(validated)
            validation_by_flake.append(
                {
                    "flakeId": int(flake["id"]),
                    "validated": validated,
                    "validationScore": round(float(representative["foldScore"]), 4),
                    "foldDepthDeltaVoxels": round(float(representative["foldDepthDelta"]), 3),
                    "foldPositionResidualVoxels": round(
                        float(representative["foldPositionResidual"]), 3
                    ),
                    "foldFiberDeltaDeg": round(float(representative["foldFiberDelta"]), 3),
                    "supportA": round(float(representative["supportA"]), 3),
                    "supportB": round(float(representative["supportB"]), 3),
                    "needleCountA": int(representative["needleCountA"]),
                    "needleCountB": int(representative["needleCountB"]),
                }
            )
    validation_by_flake.sort(key=lambda value: int(value["flakeId"]))
    observed_validated_count = len(validated_pairs)
    null_validated_median = float(np.median(null_validated_counts)) if null_validated_counts else 0.0
    elapsed_ms = (time.monotonic() - started) * 1000.0
    result = {
        "identity": identity,
        "view": full_result["view"],
        "settings": {
            "split": "stable 64-bit hash of global needle catalog id",
            "folds": 2,
            "minimumNeedlesPerFoldMode": 4,
            "minimumValidationScore": 0.08,
            "repetitions": repetitions,
            "null": "fully rematched fold-B depth/fiber mode permutation",
        },
        "validationByFlake": validation_by_flake,
        "stats": {
            "elapsedMs": round(elapsed_ms, 2),
            "cacheHit": False,
            "eligibleCellCount": eligible_cell_count,
            "foldACellCount": len(first_by_cell),
            "foldBCellCount": len(second_by_cell),
            "foldAFlakeCount": sum(len(value) for value in first_by_cell.values()),
            "foldBFlakeCount": sum(len(value) for value in second_by_cell.values()),
            "replicatedPairCount": len(observed),
            "validatedPairCount": observed_validated_count,
            "validatedCellCount": sum(
                any(bool(value["validated"]) for value in values)
                for values in representatives_by_cell.values()
            ),
            "validatedFullFlakeCount": validated_full_count,
            "validatedFullFlakeFraction": round(
                validated_full_count / max(len(full_result["flakes"]), 1), 4
            ),
            "medianFoldDepthDeltaVoxels": _median(
                [float(value["foldDepthDelta"]) for value in validated_pairs], 3
            ),
            "medianFoldPositionResidualVoxels": _median(
                [float(value["foldPositionResidual"]) for value in validated_pairs], 3
            ),
            "medianFoldFiberDeltaDeg": _median(
                [float(value["foldFiberDelta"]) for value in validated_pairs], 3
            ),
            "nullValidatedPairCount": round(null_validated_median, 2),
            "validatedPairNullRatio": round(
                observed_validated_count / max(null_validated_median, 1.0e-8), 4
            ),
            "nullMedianFiberDeltaDeg": _median(null_fiber_angles, 3),
            "constraint": (
                "each fold is fit independently from disjoint raw needle records; "
                "the null is permuted before mutual rematching"
            ),
        },
    }
    _atomic_compact_json(cache_path, result)
    return result
