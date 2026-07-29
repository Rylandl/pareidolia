from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .acus import _robust_common_normal


NORMAL_FAMILY_VERSION = 3
NORMAL_FAMILY_DTYPE = np.dtype(
    [
        ("secondaryFitted", "u1"),
        ("standaloneCandidate", "u1"),
        ("included", "u1"),
        ("primaryNormal", "<f4", (3,)),
        ("secondaryNormal", "<f4", (3,)),
        ("primaryConfidence", "<f4"),
        ("secondaryConfidence", "<f4"),
        ("secondaryStandaloneConfidence", "<f4"),
        ("primaryCoverage", "<f4"),
        ("secondaryCoverage", "<f4"),
        ("ambiguousFraction", "<f4"),
        ("overlapFraction", "<f4"),
        ("unassignedFraction", "<f4"),
        ("primaryInlierLimitDeg", "<f4"),
        ("secondaryInlierLimitDeg", "<f4"),
        ("secondaryMedianResidualDeg", "<f4"),
        ("normalAngleDeg", "<f4"),
        ("primaryNeedleCount", "<u2"),
        ("secondaryNeedleCount", "<u2"),
        ("ambiguousNeedleCount", "<u2"),
        ("alignedNeighborCount", "u1"),
        ("componentId", "<i4"),
        ("componentSize", "<u4"),
    ]
)


DEFAULT_SETTINGS: dict[str, float | int] = {
    "minimumNeedles": 6,
    "minimumSecondaryCoverage": 0.15,
    "minimumSecondaryConfidence": 0.20,
    "minimumNormalAngleDeg": 20.0,
    "assignmentMarginDeg": 3.0,
    "maximumFamilyResidualDeg": 22.0,
    "maximumNeighborAngleDeg": 12.0,
    "minimumSpatialComponentSize": 3,
}


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _catalog_records_for_cell(
    catalog: np.ndarray,
    counts: np.ndarray,
    center: np.ndarray,
    settings: dict[str, Any],
    bin_shape_zyx: tuple[int, int, int],
) -> tuple[np.ndarray, np.ndarray]:
    cube_size = int(settings["cubeSize"])
    half_cube = cube_size * 0.5
    halo = int(settings["halo"])
    bin_size = int(settings["binSize"])
    maximum_per_bin = int(settings["maxNeedlesPerBin"])
    maximum_needles = int(settings["maxNeedles"])
    radius = max(3, int(math.ceil(float(settings["scale"]) * 2.5)))
    low_xyz = center - half_cube - radius
    high_xyz = center + half_cube + radius
    bin_low_xyz = np.floor((low_xyz - halo) / bin_size).astype(int)
    bin_high_xyz = np.floor((high_xyz - halo) / bin_size).astype(int)
    bin_low_xyz = np.maximum(bin_low_xyz, 0)
    bin_high_xyz = np.minimum(
        bin_high_xyz,
        np.asarray(
            [bin_shape_zyx[2] - 1, bin_shape_zyx[1] - 1, bin_shape_zyx[0] - 1]
        ),
    )
    ids: list[int] = []
    for bin_z in range(bin_low_xyz[2], bin_high_xyz[2] + 1):
        for bin_y in range(bin_low_xyz[1], bin_high_xyz[1] + 1):
            base = (bin_z * bin_shape_zyx[1] + bin_y) * bin_shape_zyx[2]
            ids.extend(
                base + bin_x
                for bin_x in range(bin_low_xyz[0], bin_high_xyz[0] + 1)
            )
    if not ids:
        return catalog[:0, :0].reshape(-1), np.empty(0, dtype=np.int64)
    bin_ids = np.asarray(ids, dtype=np.int64)
    slots = np.arange(maximum_per_bin, dtype=np.int64)
    slot_mask = slots[None, :] < counts[bin_ids, None]
    records = catalog[bin_ids][slot_mask]
    record_ids = (bin_ids[:, None] * maximum_per_bin + slots[None, :])[slot_mask]
    if len(records):
        inside = np.all(np.abs(records["center"] - center) <= half_cube, axis=1)
        records = records[inside]
        record_ids = record_ids[inside]
    if len(records) > maximum_needles:
        chosen = np.argpartition(records["score"], -maximum_needles)[-maximum_needles:]
        records = records[chosen]
        record_ids = record_ids[chosen]
    return records, record_ids


def _orient_axial(vector: np.ndarray) -> np.ndarray:
    result = np.asarray(vector, dtype=np.float32).copy()
    result /= max(float(np.linalg.norm(result)), 1.0e-8)
    dominant = int(np.argmax(np.abs(result)))
    if result[dominant] < 0.0:
        result = -result
    return result


def _axial_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
    )


def _plane_residual_degrees(
    directions: np.ndarray, normal: np.ndarray
) -> np.ndarray:
    return np.degrees(
        np.arcsin(np.clip(np.abs(directions @ normal), 0.0, 1.0))
    )


def _adaptive_inlier_limit(
    residual_degrees: np.ndarray, maximum_residual: float
) -> float:
    if not len(residual_degrees):
        return 8.0
    return max(
        8.0,
        min(float(maximum_residual), float(np.median(residual_degrees)) * 2.8),
    )


def _fit_normal_family(
    directions: np.ndarray,
    weights: np.ndarray,
    maximum_residual: float,
) -> tuple[np.ndarray, float, float, float] | None:
    if len(directions) < 6:
        return None
    normal, eigenvalues, _ = _robust_common_normal(directions, weights)
    normal = _orient_axial(normal)
    residual = _plane_residual_degrees(directions, normal)
    inlier_limit = _adaptive_inlier_limit(residual, maximum_residual)
    inliers = residual <= inlier_limit
    confidence = float(
        np.clip(
            (
                (float(eigenvalues[1]) - float(eigenvalues[0]))
                / max(float(eigenvalues[2]), 1.0e-7)
            )
            * float(np.mean(inliers)),
            0.0,
            1.0,
        )
    )
    return normal, confidence, inlier_limit, float(np.median(residual))


def _normal_family_assignment(
    directions: np.ndarray,
    primary_normal: np.ndarray,
    secondary_normal: np.ndarray,
    primary_limit: float,
    secondary_limit: float,
    assignment_margin: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Describe a two-plane assignment without changing primary ownership.

    The primary family remains the complete legacy input.  ``secondary`` is
    therefore the strict, non-overlapping supply admitted to the additive
    family fit.  ``ambiguous`` records only eligible needles whose residuals
    differ by less than the declared margin; ``overlap`` records the wider set
    compatible with both planes.  These diagnostics are intentionally not
    used as spatial-inclusion evidence.
    """
    primary_residual = _plane_residual_degrees(directions, primary_normal)
    secondary_residual = _plane_residual_degrees(directions, secondary_normal)
    primary_eligible = primary_residual <= primary_limit
    secondary_eligible = secondary_residual <= secondary_limit
    overlap = primary_eligible & secondary_eligible
    ambiguous = overlap & (
        np.abs(primary_residual - secondary_residual) < assignment_margin
    )
    secondary = (
        secondary_eligible & (primary_residual >= primary_limit + assignment_margin)
    )
    unassigned = ~(primary_eligible | secondary)
    return primary_eligible, secondary, ambiguous, overlap, unassigned


def _infer_cell_normal_families(
    records: np.ndarray,
    primary_seed: np.ndarray,
    primary_seed_confidence: float,
    settings: dict[str, float | int] | None = None,
) -> dict[str, Any]:
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    directions = np.asarray(records["direction"], dtype=np.float32)
    weights = np.asarray(records["score"], dtype=np.float32)
    primary_seed = _orient_axial(primary_seed)
    total_weight = max(float(np.sum(weights)), 1.0e-8)
    maximum_residual = float(resolved["maximumFamilyResidualDeg"])
    minimum_needles = int(resolved["minimumNeedles"])
    primary_residual = _plane_residual_degrees(directions, primary_seed)
    primary_limit = _adaptive_inlier_limit(primary_residual, maximum_residual)
    primary_initial = primary_residual <= primary_limit
    margin = float(resolved["assignmentMarginDeg"])
    secondary_supply = primary_residual >= primary_limit + margin
    base = {
        "secondaryFitted": False,
        "standaloneCandidate": False,
        "primaryNormal": primary_seed,
        "secondaryNormal": np.zeros(3, dtype=np.float32),
        "primaryConfidence": float(primary_seed_confidence),
        "secondaryConfidence": 0.0,
        "secondaryStandaloneConfidence": 0.0,
        "primaryCoverage": float(np.sum(weights[primary_initial]) / total_weight),
        "secondaryCoverage": 0.0,
        "ambiguousFraction": 0.0,
        "overlapFraction": 0.0,
        "unassignedFraction": float(
            np.sum(weights[~primary_initial]) / total_weight
        ),
        "primaryInlierLimitDeg": primary_limit,
        "secondaryInlierLimitDeg": 0.0,
        "secondaryMedianResidualDeg": 90.0,
        "normalAngleDeg": 0.0,
        "primaryNeedleCount": int(np.count_nonzero(primary_initial)),
        "secondaryNeedleCount": 0,
        "ambiguousNeedleCount": 0,
    }
    if (
        len(records) < minimum_needles
        or int(np.count_nonzero(secondary_supply)) < minimum_needles
    ):
        return base

    secondary_fit = _fit_normal_family(
        directions[secondary_supply], weights[secondary_supply], maximum_residual
    )
    if secondary_fit is None:
        return base
    (
        secondary_normal,
        secondary_standalone_confidence,
        secondary_limit,
        _,
    ) = secondary_fit
    for _ in range(2):
        secondary_residual = _plane_residual_degrees(
            directions, secondary_normal
        )
        secondary = secondary_supply & (secondary_residual <= secondary_limit)
        if int(np.count_nonzero(secondary)) < minimum_needles:
            return base
        secondary_fit = _fit_normal_family(
            directions[secondary],
            weights[secondary],
            maximum_residual,
        )
        if secondary_fit is None:
            return base
        (
            secondary_normal,
            secondary_confidence,
            secondary_limit,
            secondary_median,
        ) = secondary_fit

    (
        primary_initial,
        secondary,
        ambiguous,
        overlap,
        unassigned,
    ) = _normal_family_assignment(
        directions,
        primary_seed,
        secondary_normal,
        primary_limit,
        secondary_limit,
        margin,
    )
    if int(np.count_nonzero(secondary)) < minimum_needles:
        return base
    primary_coverage = float(np.sum(weights[primary_initial]) / total_weight)
    secondary_coverage = float(np.sum(weights[secondary]) / total_weight)
    ambiguous_fraction = float(np.sum(weights[ambiguous]) / total_weight)
    overlap_fraction = float(np.sum(weights[overlap]) / total_weight)
    unassigned_fraction = float(np.sum(weights[unassigned]) / total_weight)
    normal_angle = _axial_angle_degrees(primary_seed, secondary_normal)
    standalone = bool(
        secondary_coverage >= float(resolved["minimumSecondaryCoverage"])
        and secondary_standalone_confidence
        >= float(resolved["minimumSecondaryConfidence"])
        and normal_angle >= float(resolved["minimumNormalAngleDeg"])
    )
    return {
        "secondaryFitted": True,
        "standaloneCandidate": standalone,
        "primaryNormal": primary_seed,
        "secondaryNormal": secondary_normal,
        "primaryConfidence": float(primary_seed_confidence),
        "secondaryConfidence": secondary_confidence,
        "secondaryStandaloneConfidence": secondary_standalone_confidence,
        "primaryCoverage": primary_coverage,
        "secondaryCoverage": secondary_coverage,
        "ambiguousFraction": ambiguous_fraction,
        "overlapFraction": overlap_fraction,
        "unassignedFraction": unassigned_fraction,
        "primaryInlierLimitDeg": primary_limit,
        "secondaryInlierLimitDeg": secondary_limit,
        "secondaryMedianResidualDeg": secondary_median,
        "normalAngleDeg": normal_angle,
        "primaryNeedleCount": int(np.count_nonzero(primary_initial)),
        "secondaryNeedleCount": int(np.count_nonzero(secondary)),
        "ambiguousNeedleCount": int(np.count_nonzero(ambiguous)),
    }


def _union_components(
    values: np.ndarray,
    maximum_neighbor_angle: float,
) -> tuple[int, int]:
    standalone = np.asarray(values["standaloneCandidate"], dtype=bool)
    flat_standalone = standalone.reshape(-1)
    parent = np.full(values.size, -1, dtype=np.int32)
    candidate_ids = np.flatnonzero(flat_standalone).astype(np.int32)
    parent[candidate_ids] = candidate_ids
    tree_size = np.ones(values.size, dtype=np.uint32)
    flat_degree = values["alignedNeighborCount"].reshape(-1)
    index_grid = np.arange(values.size, dtype=np.int32).reshape(values.shape)
    aligned_edge_count = 0

    def find(index: int) -> int:
        root = index
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[index]) != index:
            next_index = int(parent[index])
            parent[index] = root
            index = next_index
        return root

    def union(first: int, second: int) -> None:
        first_root, second_root = find(first), find(second)
        if first_root == second_root:
            return
        if int(tree_size[first_root]) < int(tree_size[second_root]):
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        tree_size[first_root] += tree_size[second_root]

    normals = np.asarray(values["secondaryNormal"], dtype=np.float32)
    for axis in range(3):
        left = [slice(None)] * 3
        right = [slice(None)] * 3
        left[axis] = slice(0, -1)
        right[axis] = slice(1, None)
        left_key, right_key = tuple(left), tuple(right)
        eligible = standalone[left_key] & standalone[right_key]
        if not np.any(eligible):
            continue
        dot = np.abs(np.sum(normals[left_key] * normals[right_key], axis=-1))
        angle = np.degrees(np.arccos(np.clip(dot, 0.0, 1.0)))
        aligned = eligible & (angle <= maximum_neighbor_angle)
        first_ids = index_grid[left_key][aligned]
        second_ids = index_grid[right_key][aligned]
        aligned_edge_count += len(first_ids)
        flat_degree[first_ids] = np.minimum(flat_degree[first_ids] + 1, 255)
        flat_degree[second_ids] = np.minimum(flat_degree[second_ids] + 1, 255)
        for first, second in zip(first_ids, second_ids):
            union(int(first), int(second))

    roots = np.fromiter((find(int(index)) for index in candidate_ids), dtype=np.int32)
    unique_roots, inverse, counts = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    flat_component_id = values["componentId"].reshape(-1)
    flat_component_size = values["componentSize"].reshape(-1)
    flat_component_id[candidate_ids] = inverse.astype(np.int32)
    flat_component_size[candidate_ids] = counts[inverse].astype(np.uint32)
    return len(unique_roots), aligned_edge_count


def _normal_family_partitions(
    records: np.ndarray,
    record_ids: np.ndarray,
    cell: np.void,
    family: np.void,
    settings: dict[str, float | int] | None = None,
) -> list[dict[str, Any]]:
    if not bool(family["included"]):
        return [
            {
                "normalFamily": 0,
                "records": records,
                "recordIds": record_ids,
                "normal": np.asarray(cell["normal"], dtype=np.float32),
                "normalConfidence": float(cell["normalConfidence"]),
                "familyCoverage": 1.0,
                "ambiguousFraction": 0.0,
                "familyComponentId": None,
                "familyComponentSize": 0,
            }
        ]
    resolved = {**DEFAULT_SETTINGS, **(settings or {})}
    directions = np.asarray(records["direction"], dtype=np.float32)
    _, secondary, _, _, _ = _normal_family_assignment(
        directions,
        np.asarray(cell["normal"], dtype=np.float32),
        np.asarray(family["secondaryNormal"], dtype=np.float32),
        float(family["primaryInlierLimitDeg"]),
        float(family["secondaryInlierLimitDeg"]),
        float(resolved["assignmentMarginDeg"]),
    )
    common = {
        "ambiguousFraction": float(family["ambiguousFraction"]),
        "familyComponentId": int(family["componentId"]),
        "familyComponentSize": int(family["componentSize"]),
    }
    return [
        {
            **common,
            "normalFamily": 0,
            "records": records,
            "recordIds": record_ids,
            "normal": np.asarray(cell["normal"], dtype=np.float32),
            "normalConfidence": float(cell["normalConfidence"]),
            "familyCoverage": 1.0,
        },
        {
            **common,
            "normalFamily": 1,
            "records": records[secondary],
            "recordIds": record_ids[secondary],
            "normal": np.asarray(family["secondaryNormal"], dtype=np.float32),
            "normalConfidence": float(family["secondaryConfidence"]),
            "familyCoverage": float(family["secondaryCoverage"]),
        },
    ]


def build_normal_families(
    output_root: str | Path,
    force: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
) -> dict[str, Any]:
    root = Path(output_root)
    analysis = json.loads((root / "analysis.json").read_text())
    if analysis.get("state") != "complete":
        raise ValueError("the cross-scroll slab analysis is not complete")
    grid = json.loads((root / "grid.json").read_text())
    identity = {
        "version": NORMAL_FAMILY_VERSION,
        "analysisIdentity": analysis["identity"],
        "settings": DEFAULT_SETTINGS,
    }
    summary_path = root / f"normal-families-v{NORMAL_FAMILY_VERSION}.json"
    array_path = root / f"normal-families-v{NORMAL_FAMILY_VERSION}.npy"
    if summary_path.is_file() and array_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    settings = analysis["identity"]["settings"]
    bin_shape = tuple(int(value) for value in analysis["binShapeZYX"])
    cells = np.load(root / "cells.npy", mmap_mode="r")
    catalog = np.load(root / "needles.npy", mmap_mode="r")
    counts = np.load(root / "needle-counts.npy", mmap_mode="r")
    values = np.zeros(cells.shape, dtype=NORMAL_FAMILY_DTYPE)
    values["componentId"] = -1
    valid_indices = np.argwhere(cells["valid"] > 0)
    total = len(valid_indices)
    for completed, (cell_z, cell_y, cell_x) in enumerate(valid_indices, start=1):
        center = np.asarray(
            [grid["x"][cell_x], grid["y"][cell_y], grid["z"][cell_z]],
            dtype=np.float32,
        )
        records, _ = _catalog_records_for_cell(
            catalog, counts, center, settings, bin_shape
        )
        cell = cells[cell_z, cell_y, cell_x]
        inferred = _infer_cell_normal_families(
            records,
            np.asarray(cell["normal"], dtype=np.float32),
            float(cell["normalConfidence"]),
        )
        target = values[cell_z, cell_y, cell_x]
        for field in NORMAL_FAMILY_DTYPE.names or ():
            if field in inferred:
                target[field] = inferred[field]
        if progress is not None and (completed % 10000 == 0 or completed == total):
            progress(completed, total, (time.monotonic() - started) * 1000.0)

    component_count, aligned_edge_count = _union_components(
        values, float(DEFAULT_SETTINGS["maximumNeighborAngleDeg"])
    )
    included = (
        (values["standaloneCandidate"] > 0)
        & (
            values["componentSize"]
            >= int(DEFAULT_SETTINGS["minimumSpatialComponentSize"])
        )
    )
    values["included"] = included.astype(np.uint8)
    substantial_components = np.unique(values["componentId"][included])
    component_catalog = []
    for component_id in substantial_components:
        member = included & (values["componentId"] == component_id)
        indices = np.argwhere(member)
        component_catalog.append(
            {
                "componentId": int(component_id),
                "cellCount": int(len(indices)),
                "planeCount": int(len(np.unique(indices[:, 0]))),
                "minimumCellXYZ": [
                    int(np.min(indices[:, 2])),
                    int(np.min(indices[:, 1])),
                    int(np.min(indices[:, 0])),
                ],
                "maximumCellXYZ": [
                    int(np.max(indices[:, 2])),
                    int(np.max(indices[:, 1])),
                    int(np.max(indices[:, 0])),
                ],
                "medianCoverage": round(
                    float(np.median(values["secondaryCoverage"][member])), 4
                ),
                "medianConfidence": round(
                    float(np.median(values["secondaryConfidence"][member])), 4
                ),
                "medianStandaloneConfidence": round(
                    float(
                        np.median(
                            values["secondaryStandaloneConfidence"][member]
                        )
                    ),
                    4,
                ),
                "medianNormalAngleDeg": round(
                    float(np.median(values["normalAngleDeg"][member])), 3
                ),
            }
        )
    component_catalog.sort(key=lambda value: int(value["cellCount"]), reverse=True)
    temporary_path = array_path.with_suffix(array_path.suffix + ".tmp")
    with temporary_path.open("wb") as handle:
        np.save(handle, values)
    temporary_path.replace(array_path)
    fitted = values["secondaryFitted"] > 0
    standalone = values["standaloneCandidate"] > 0
    stats = {
        "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
        "cacheHit": False,
        "validCellCount": total,
        "secondaryFittedCellCount": int(np.count_nonzero(fitted)),
        "standaloneCandidateCellCount": int(np.count_nonzero(standalone)),
        "includedSecondaryCellCount": int(np.count_nonzero(included)),
        "includedSecondaryCellFraction": round(
            float(np.count_nonzero(included)) / max(total, 1), 5
        ),
        "candidateComponentCount": component_count,
        "includedComponentCount": len(substantial_components),
        "alignedCandidateEdgeCount": aligned_edge_count,
        "largestIncludedComponentSize": max(
            (int(value["cellCount"]) for value in component_catalog), default=0
        ),
        "medianIncludedCoverage": round(
            float(np.median(values["secondaryCoverage"][included])), 4
        )
        if np.any(included)
        else None,
        "medianIncludedConfidence": round(
            float(np.median(values["secondaryConfidence"][included])), 4
        )
        if np.any(included)
        else None,
        "medianIncludedAmbiguousFraction": round(
            float(np.median(values["ambiguousFraction"][included])), 4
        )
        if np.any(included)
        else None,
        "medianIncludedOverlapFraction": round(
            float(np.median(values["overlapFraction"][included])), 4
        )
        if np.any(included)
        else None,
        "includedCellsByPlane": [
            int(np.count_nonzero(included[z_index]))
            for z_index in range(included.shape[0])
        ],
        "constraint": (
            "secondary candidates are recorded from standalone per-cell evidence; "
            "neighbor agreement only controls pipeline inclusion"
        ),
    }
    result = {
        "identity": identity,
        "settings": DEFAULT_SETTINGS,
        "stats": stats,
        "components": component_catalog[:100],
        "artifacts": {"families": array_path.name},
    }
    _atomic_json(summary_path, result)
    return result


def load_normal_families(
    output_root: str | Path,
    force: bool = False,
    progress: Callable[[int, int, float], None] | None = None,
) -> tuple[dict[str, Any], np.ndarray]:
    result = build_normal_families(output_root, force=force, progress=progress)
    root = Path(output_root)
    values = np.load(
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.npy", mmap_mode="r"
    )
    return result, values
