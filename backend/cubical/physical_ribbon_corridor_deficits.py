from __future__ import annotations

import json
import math
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_one_sided import (
    PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA,
    _load_frontier,
    _load_stage_inputs,
)
from .physical_ribbon_corridor_variants import (
    _corridor_settings_from_manifest,
)
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
)
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_CORRIDOR_DEFICITS_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-deficits"
)
PHYSICAL_RIBBON_CORRIDOR_DEFICITS_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_DEFICITS_STEM = (
    "physical-ribbon-corridor-deficits-v1"
)


def _finite_distribution(values: list[float]) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": len(finite),
        "minimum": round(float(np.min(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def _nearest_distance_and_index(
    point: np.ndarray, target: np.ndarray, *, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    point = np.asarray(point, dtype=np.float32)
    target = np.asarray(target, dtype=np.float32)
    distance = np.full(len(point), np.inf, dtype=np.float32)
    index = np.full(len(point), -1, dtype=np.int32)
    if not len(point) or not len(target):
        return distance, index
    for begin in range(0, len(point), batch_size):
        end = min(begin + batch_size, len(point))
        delta = point[begin:end, None, :] - target[None, :, :]
        squared = np.sum(delta * delta, axis=2)
        nearest = np.argmin(squared, axis=1)
        index[begin:end] = nearest
        distance[begin:end] = np.sqrt(
            squared[np.arange(end - begin), nearest]
        )
    return distance, index


def _minimum_set_distance(first: np.ndarray, second: np.ndarray) -> float:
    if not len(first) or not len(second):
        return math.inf
    best = math.inf
    for begin in range(0, len(first), 256):
        delta = first[begin : begin + 256, None, :] - second[None, :, :]
        best = min(best, float(np.min(np.linalg.norm(delta, axis=2))))
    return best


def _largest_four_connected_fraction(mask: np.ndarray) -> float:
    mask = np.asarray(mask, dtype=bool)
    if mask.ndim != 2 or not np.any(mask):
        return 0.0
    seen = np.zeros_like(mask)
    largest = 0
    rows, columns = mask.shape
    for row, column in np.argwhere(mask):
        row = int(row)
        column = int(column)
        if seen[row, column]:
            continue
        seen[row, column] = True
        stack = [(row, column)]
        size = 0
        while stack:
            current_row, current_column = stack.pop()
            size += 1
            for next_row, next_column in (
                (current_row - 1, current_column),
                (current_row + 1, current_column),
                (current_row, current_column - 1),
                (current_row, current_column + 1),
            ):
                if (
                    0 <= next_row < rows
                    and 0 <= next_column < columns
                    and mask[next_row, next_column]
                    and not seen[next_row, next_column]
                ):
                    seen[next_row, next_column] = True
                    stack.append((next_row, next_column))
        largest = max(largest, size)
    return largest / mask.size


def _dominant_region(
    arc_xyz: np.ndarray,
    triangle_center: np.ndarray,
    triangle_region: np.ndarray,
) -> int:
    _, nearest = _nearest_distance_and_index(arc_xyz, triangle_center)
    valid = nearest >= 0
    if not np.any(valid):
        return -1
    count = Counter(int(value) for value in triangle_region[nearest[valid]])
    return min(count, key=lambda value: (-count[value], value))


def _best_failed_variant(
    row: int, replay: Mapping[str, np.ndarray]
) -> int:
    offset = np.asarray(replay["corridorVariantOffset"], dtype=np.int64)
    if row + 1 >= len(offset):
        return -1
    begin, end = int(offset[row]), int(offset[row + 1])
    split = np.asarray(replay["corridorVariantComponentSplit"]) > 0
    conflict = np.asarray(replay["corridorVariantHardConflict"]) > 0
    exact = np.asarray(replay["corridorVariantExactConnected"]) > 0
    region_before = np.asarray(
        replay["corridorVariantTriangleRegionCountBefore"], dtype=np.int32
    )
    region_after = np.asarray(
        replay["corridorVariantTriangleRegionCountAfter"], dtype=np.int32
    )
    area_before = np.asarray(
        replay["corridorVariantTriangleAreaBefore"], dtype=np.float32
    )
    area_after = np.asarray(
        replay["corridorVariantTriangleAreaAfter"], dtype=np.float32
    )
    objective = np.asarray(
        replay["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    coverage = np.asarray(
        replay["corridorVariantPatchCoverage"], dtype=np.float32
    )
    shared = np.asarray(
        replay["corridorVariantSharedArcRegionFraction"], dtype=np.float32
    )
    eligible = [
        index
        for index in range(begin, end)
        if not split[index] and not conflict[index] and not exact[index]
    ]
    if not eligible:
        return -1
    return max(
        eligible,
        key=lambda index: (
            float(shared[index]),
            int(region_before[index] - region_after[index]),
            float(area_after[index] - area_before[index]),
            float(objective[index]),
            float(coverage[index]),
            -index,
        ),
    )


def _reconstruct_failed_variant(
    row: int,
    variant_index: int,
    corridor: Mapping[str, np.ndarray],
    replay: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    *,
    surface_settings: Any,
) -> tuple[dict[str, np.ndarray], np.ndarray, int]:
    scored_corridor = np.asarray(corridor["scoredCorridorIndex"], dtype=np.int32)
    component_id = int(
        np.asarray(corridor["corridorTopologyComponent"], dtype=np.int32)[
            int(scored_corridor[row])
        ]
    )
    added_offset = np.asarray(
        replay["corridorVariantAddedOffset"], dtype=np.int64
    )
    added_value = np.asarray(
        replay["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        replay["corridorVariantRemovedOffset"], dtype=np.int64
    )
    removed_value = np.asarray(
        replay["corridorVariantRemovedFrontierIndex"], dtype=np.int32
    )
    added = added_value[
        int(added_offset[variant_index]) : int(added_offset[variant_index + 1])
    ]
    removed = removed_value[
        int(removed_offset[variant_index]) : int(removed_offset[variant_index + 1])
    ]
    baseline_selected = np.asarray(configuration["selected"]) > 0
    original_component = np.asarray(configuration["component"], dtype=np.int32)
    selected = baseline_selected.copy()
    selected[removed] = False
    selected[added] = True
    local_selected = selected & (original_component == component_id)
    local_selected[added] = selected[added]
    local_component, _ = _component_labels(
        local_selected,
        np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32),
        np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32),
    )
    local_configuration = dict(configuration)
    local_configuration["selected"] = local_selected.astype(np.uint8)
    local_configuration["component"] = local_component
    surface, _ = build_physical_ribbon_surface_complex(
        ribbon, topology, local_configuration, settings=surface_settings
    )
    return surface, local_selected, component_id


def _failed_surface_metrics(
    row: int,
    corridor: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    local_selected: np.ndarray,
    topology: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    patch_offset = np.asarray(corridor["corridorPatchOffset"], dtype=np.int64)
    begin, end = int(patch_offset[row]), int(patch_offset[row + 1])
    patch = np.asarray(corridor["corridorPatchXYZ"], dtype=np.float32)[begin:end]
    patch_thickness = np.asarray(
        corridor["corridorPatchThicknessVoxels"], dtype=np.float32
    )[begin:end]
    scale = np.maximum(patch_thickness, 1.0e-3)
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    midpoint = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    triangle_center = (
        np.mean(midpoint[triangles], axis=1)
        if len(triangles)
        else np.empty((0, 3), dtype=np.float32)
    )
    triangle_distance, _ = _nearest_distance_and_index(patch, triangle_center)
    normalized_triangle_distance = triangle_distance / scale
    selected_node = np.flatnonzero(local_selected)
    node_distance, nearest_node_local = _nearest_distance_and_index(
        patch, midpoint[selected_node]
    )
    normalized_node_distance = node_distance / scale
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    selected_edge = local_selected[first] & local_selected[second]
    degree = np.bincount(
        np.concatenate((first[selected_edge], second[selected_edge])),
        minlength=len(local_selected),
    )
    nearest_degree = np.zeros(len(patch), dtype=np.int32)
    valid_node = nearest_node_local >= 0
    nearest_degree[valid_node] = degree[
        selected_node[nearest_node_local[valid_node]]
    ]
    rows = int(np.asarray(corridor["corridorPatchRows"])[row])
    columns = int(np.asarray(corridor["corridorPatchColumns"])[row])
    unsupported = normalized_triangle_distance > 0.5
    if rows * columns == len(unsupported):
        unsupported_grid = unsupported.reshape(rows, columns)
        row_margin = max(1, rows // 5)
        column_margin = max(1, columns // 5)
        central = unsupported_grid[
            row_margin : rows - row_margin,
            column_margin : columns - column_margin,
        ]
        central_unsupported = float(np.mean(central)) if central.size else 0.0
        largest_unsupported = _largest_four_connected_fraction(
            unsupported_grid
        )
    else:
        central_unsupported = math.nan
        largest_unsupported = math.nan

    scored_corridor = np.asarray(corridor["scoredCorridorIndex"], dtype=np.int32)
    corridor_index = int(scored_corridor[row])
    pair_offset = np.asarray(corridor["corridorPairOffset"], dtype=np.int64)
    pair_begin, pair_end = (
        int(pair_offset[corridor_index]),
        int(pair_offset[corridor_index + 1]),
    )
    first_edge = np.asarray(corridor["corridorFirstBoundaryEdge"], dtype=np.int32)[
        pair_begin:pair_end
    ]
    second_edge = np.asarray(
        corridor["corridorSecondBoundaryEdge"], dtype=np.int32
    )[pair_begin:pair_end]
    boundary_xyz = np.asarray(
        corridor["boundaryEdgeMidpointXYZ"], dtype=np.float32
    )
    triangle_region = _triangle_region_labels(triangles)
    first_region = _dominant_region(
        boundary_xyz[first_edge], triangle_center, triangle_region
    )
    second_region = _dominant_region(
        boundary_xyz[second_edge], triangle_center, triangle_region
    )
    if first_region >= 0 and second_region >= 0:
        if first_region == second_region:
            region_separation = 0.0
        else:
            region_separation = _minimum_set_distance(
                triangle_center[triangle_region == first_region],
                triangle_center[triangle_region == second_region],
            )
    else:
        region_separation = math.inf
    median_thickness = float(np.median(scale)) if len(scale) else 1.0
    connection = _evaluate_corridor_connections(
        surface, corridor, corridor
    )
    return {
        "patchPixelCount": len(patch),
        "triangleCount": len(triangles),
        "triangleRegionCount": int(len(np.unique(triangle_region))),
        "triangleCoverageAtQuarterThickness": round(
            float(np.mean(normalized_triangle_distance <= 0.25)), 6
        ),
        "triangleCoverageAtHalfThickness": round(
            float(np.mean(normalized_triangle_distance <= 0.5)), 6
        ),
        "triangleCoverageAtOneThickness": round(
            float(np.mean(normalized_triangle_distance <= 1.0)), 6
        ),
        "triangleDistanceThicknesses": _finite_distribution(
            [float(value) for value in normalized_triangle_distance]
        ),
        "nodeCoverageAtHalfThickness": round(
            float(np.mean(normalized_node_distance <= 0.5)), 6
        ),
        "nodeDistanceThicknesses": _finite_distribution(
            [float(value) for value in normalized_node_distance]
        ),
        "nearestSelectedStrictDegree": _finite_distribution(
            [float(value) for value in nearest_degree]
        ),
        "centralUnsupportedFraction": (
            round(central_unsupported, 6)
            if math.isfinite(central_unsupported)
            else None
        ),
        "largestUnsupportedPatchFraction": (
            round(largest_unsupported, 6)
            if math.isfinite(largest_unsupported)
            else None
        ),
        "firstDominantTriangleRegion": first_region,
        "secondDominantTriangleRegion": second_region,
        "dominantRegionSeparationThicknesses": (
            round(region_separation / median_thickness, 6)
            if math.isfinite(region_separation)
            else None
        ),
        "boundaryArcSharedRegionFraction": round(
            float(connection["boundaryArcSharedRegionFraction"][row]), 6
        ),
        "patchTriangleCoverageLegacy": round(
            float(connection["patchTriangleCoverage"][row]), 6
        ),
    }


def run_physical_ribbon_corridor_deficits(
    replay_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    replay_path, replay_manifest, replay = _load_replay_artifact(replay_root)
    if replay_manifest.get("schema") != PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA:
        raise ValueError("corridor deficit analysis requires an exact one-sided replay")
    frontier_path, frontier_manifest, frontier = _load_frontier(
        replay_manifest["identity"]["frontier"]["manifestPath"]
    )
    (
        corridor_path,
        corridor_manifest,
        corridor,
        _,
        _,
        _,
        _,
        configuration_manifest,
        base_configuration,
        ribbon,
    ) = _load_stage_inputs(frontier_manifest)
    if (
        sha256_file(frontier_path)
        != replay_manifest["identity"]["frontier"]["manifestSha256"]
        or frontier_manifest["data"]["sha256"]
        != replay_manifest["identity"]["frontier"]["dataSha256"]
    ):
        raise ValueError("exact replay frontier has changed")
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_DEFICITS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_DEFICITS_VERSION,
        "replay": {
            "manifestPath": str(replay_path),
            "manifestSha256": sha256_file(replay_path),
            "dataSha256": replay_manifest["data"]["sha256"],
        },
        "frontier": {
            "manifestPath": str(frontier_path),
            "manifestSha256": sha256_file(frontier_path),
            "dataSha256": frontier_manifest["data"]["sha256"],
        },
        "corridors": {
            "manifestPath": str(corridor_path),
            "manifestSha256": sha256_file(corridor_path),
            "dataSha256": corridor_manifest["data"]["sha256"],
        },
        "implementationSha256": sha256_file(Path(__file__)),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_DEFICITS_STEM}.json"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
        ):
            return cached

    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    started = time.monotonic()
    if progress is not None:
        progress("reconstructing the exact replay baseline")
    baseline_surface, _ = build_physical_ribbon_surface_complex(
        ribbon,
        frontier,
        frontier,
        settings=corridor_settings.surface_settings(),
    )
    remapped = _remap_corridor_surface(
        corridor,
        baseline_surface,
        base_configuration,
        frontier,
        np.asarray(frontier["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    evidence = np.asarray(replay["corridorEvidenceEligible"]) > 0
    chosen_exact = np.asarray(replay["corridorChosenExactVariant"], dtype=np.int32)
    failed_rows = np.flatnonzero(evidence & (chosen_exact < 0))
    variant_offset = np.asarray(replay["corridorVariantOffset"], dtype=np.int64)
    records: list[dict[str, Any]] = []
    reconstructed = 0
    for completed, row_value in enumerate(failed_rows, start=1):
        row = int(row_value)
        variant_count = int(variant_offset[row + 1] - variant_offset[row])
        best_variant = _best_failed_variant(row, replay)
        record: dict[str, Any] = {
            "corridorRow": row,
            "enumeratedVariantCount": variant_count,
            "bestFailedVariantIndex": best_variant,
        }
        if best_variant < 0:
            record["status"] = (
                "no-complete-one-sided-matching"
                if variant_count == 0
                else "all-complete-matchings-split-or-conflict"
            )
        else:
            local_surface, local_selected, component_id = (
                _reconstruct_failed_variant(
                    row,
                    best_variant,
                    remapped,
                    replay,
                    ribbon,
                    frontier,
                    frontier,
                    surface_settings=corridor_settings.surface_settings(),
                )
            )
            record.update(
                {
                    "status": "reconstructed-but-disconnected",
                    "topologyComponent": component_id,
                    "variantRank": int(
                        np.asarray(replay["corridorVariantRank"])[best_variant]
                    ),
                    "variantPatchCoverage": round(
                        float(
                            np.asarray(replay["corridorVariantPatchCoverage"])[
                                best_variant
                            ]
                        ),
                        6,
                    ),
                    "variantLocalObjectiveDelta": round(
                        float(
                            np.asarray(
                                replay["corridorVariantLocalObjectiveDelta"]
                            )[best_variant]
                        ),
                        6,
                    ),
                    **_failed_surface_metrics(
                        row,
                        remapped,
                        local_surface,
                        local_selected,
                        frontier,
                    ),
                }
            )
            reconstructed += 1
        selected_model = int(
            np.asarray(remapped["corridorSelectedModel"], dtype=np.int32)[row]
        )
        record["nativeCt"] = {
            "profileCorrelation": round(
                float(
                    np.asarray(
                        remapped["corridorZeroShiftContextProfileCorrelation"]
                    )[row, selected_model]
                ),
                6,
            ),
            "competingLayerMargin": round(
                float(
                    np.asarray(remapped["corridorZeroShiftCompetingMargin"])[
                        row, selected_model
                    ]
                ),
                6,
            ),
            "boundaryTraceCorrelation": round(
                float(
                    np.asarray(remapped["corridorBoundaryTraceCorrelation"])[
                        row, selected_model
                    ]
                ),
                6,
            ),
            "minimumCurvatureRadiusThicknesses": round(
                float(
                    np.asarray(
                        remapped[
                            "corridorModelMinimumCurvatureRadiusThicknesses"
                        ]
                    )[row, selected_model]
                ),
                6,
            ),
        }
        records.append(record)
        if progress is not None and (
            completed == len(failed_rows) or completed % 5 == 0
        ):
            progress(
                f"corridor deficits {completed}/{len(failed_rows)} · "
                f"reconstructed {reconstructed}"
            )
    finished = time.monotonic()
    reconstructed_records = [
        record
        for record in records
        if record["status"] == "reconstructed-but-disconnected"
    ]
    status_count = Counter(record["status"] for record in records)
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_DEFICITS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_DEFICITS_VERSION,
        "state": "complete",
        "identity": identity,
        "statistics": {
            "failedCtCorridorCount": len(failed_rows),
            "statusCounts": dict(sorted(status_count.items())),
            "triangleCoverageAtHalfThickness": _finite_distribution(
                [
                    float(record["triangleCoverageAtHalfThickness"])
                    for record in reconstructed_records
                ]
            ),
            "centralUnsupportedFraction": _finite_distribution(
                [
                    float(record["centralUnsupportedFraction"])
                    for record in reconstructed_records
                    if record["centralUnsupportedFraction"] is not None
                ]
            ),
            "largestUnsupportedPatchFraction": _finite_distribution(
                [
                    float(record["largestUnsupportedPatchFraction"])
                    for record in reconstructed_records
                    if record["largestUnsupportedPatchFraction"] is not None
                ]
            ),
            "dominantRegionSeparationThicknesses": _finite_distribution(
                [
                    float(record["dominantRegionSeparationThicknesses"])
                    for record in reconstructed_records
                    if record["dominantRegionSeparationThicknesses"] is not None
                ]
            ),
            "records": records,
            "identityLabelsUsed": False,
        },
        "timingSeconds": {
            "total": round(finished - started, 6),
        },
        "method": {
            "decisionUnit": "best reconstructable failed state per CT-supported strip",
            "measurements": (
                "dense patch-to-triangle and patch-to-node support, largest "
                "interior unsupported region, and dominant triangle-island separation"
            ),
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
