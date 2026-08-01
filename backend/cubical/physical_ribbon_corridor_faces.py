from __future__ import annotations

import heapq
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .needle_surface import _delaunay_triangles, _triangle_geometry
from .physical_ribbon_bridging import _write_npz
from .physical_ribbon_corridor_deficits import (
    _best_failed_variant,
    _dominant_region,
    _reconstruct_failed_variant,
)
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


PHYSICAL_RIBBON_CORRIDOR_FACES_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-faces"
)
PHYSICAL_RIBBON_CORRIDOR_FACES_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_FACES_STEM = "physical-ribbon-corridor-faces-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorFaceSettings:
    """Physical gates for faces that repair a CT-supported surface corridor.

    Strict ribbon edges continue to define sheet identity.  These settings
    only authorize missing faces from the already integrated chart Delaunay
    tessellation, so a supplemental face cannot fuse graph components.
    """

    maximum_center_distance_thicknesses: float = 0.25
    maximum_center_height_thicknesses: float = 0.15
    maximum_center_tangent_raster_steps: float = 2.25
    maximum_ct_normal_residual_degrees: float = 55.0
    maximum_triangle_edge_thicknesses: float = 1.25
    maximum_path_face_count: int = 6
    minimum_arc_region_fraction: float = 0.50
    maximum_arc_triangle_distance_edges: float = 2.0

    def __post_init__(self) -> None:
        positive = (
            self.maximum_center_distance_thicknesses,
            self.maximum_center_height_thicknesses,
            self.maximum_center_tangent_raster_steps,
            self.maximum_ct_normal_residual_degrees,
            self.maximum_triangle_edge_thicknesses,
            self.minimum_arc_region_fraction,
            self.maximum_arc_triangle_distance_edges,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("corridor-face physical gates must be finite and positive")
        if self.maximum_ct_normal_residual_degrees >= 90.0:
            raise ValueError("corridor-face CT normal residual must be below 90 degrees")
        if self.maximum_path_face_count < 1:
            raise ValueError("corridor-face paths must allow at least one face")
        if self.minimum_arc_region_fraction > 1.0:
            raise ValueError("corridor-face arc fraction cannot exceed one")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _finite_distribution(values: Sequence[float]) -> dict[str, float | int]:
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


def _retained_chart_nodes(
    surface: Mapping[str, np.ndarray],
    *,
    minimum_component_ribbon_count: int,
    minimum_chart_separation_voxels: float,
    component_ids: set[int] | None = None,
) -> list[np.ndarray]:
    component = np.asarray(surface["component"], dtype=np.int32)
    component_size = np.asarray(surface["componentSize"], dtype=np.int32)
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    result: list[np.ndarray] = []
    components = np.unique(
        component[component_size >= minimum_component_ribbon_count]
    )
    for component_id in components:
        if component_id < 0 or (
            component_ids is not None and int(component_id) not in component_ids
        ):
            continue
        nodes = np.flatnonzero(
            (component == component_id)
            & np.all(np.isfinite(chart_uv), axis=1)
        )
        order = np.lexsort((nodes, chart_uv[nodes, 1], chart_uv[nodes, 0]))
        retained: list[int] = []
        for node_value in nodes[order]:
            node = int(node_value)
            if retained and np.any(
                np.linalg.norm(
                    chart_uv[np.asarray(retained, dtype=np.int32)]
                    - chart_uv[node],
                    axis=1,
                )
                < minimum_chart_separation_voxels
            ):
                continue
            retained.append(node)
        if len(retained) >= 3:
            result.append(np.asarray(retained, dtype=np.int32))
    return result


def _raw_missing_delaunay_faces(
    surface: Mapping[str, np.ndarray],
    *,
    minimum_component_ribbon_count: int,
    minimum_chart_separation_voxels: float,
    component_ids: set[int] | None = None,
) -> np.ndarray:
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    existing = {
        tuple(sorted(int(value) for value in triangle))
        for triangle in np.asarray(
            surface["triangleFrontierIndex"], dtype=np.int32
        )
    }
    missing: dict[tuple[int, int, int], tuple[int, int, int]] = {}
    for nodes in _retained_chart_nodes(
        surface,
        minimum_component_ribbon_count=minimum_component_ribbon_count,
        minimum_chart_separation_voxels=minimum_chart_separation_voxels,
        component_ids=component_ids,
    ):
        for local_triangle in _delaunay_triangles(chart_uv[nodes]):
            triangle = tuple(int(value) for value in nodes[local_triangle])
            key = tuple(sorted(triangle))
            if key not in existing:
                missing[key] = triangle
    return np.asarray(
        [missing[key] for key in sorted(missing)], dtype=np.int32
    ).reshape((-1, 3))


def _corridor_boundary_regions(
    row: int,
    surface: Mapping[str, np.ndarray],
    corridor: Mapping[str, np.ndarray],
) -> tuple[int, int, np.ndarray]:
    triangles = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    triangle_region = _triangle_region_labels(triangles)
    center = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    triangle_center = np.mean(center[triangles], axis=1)
    scored_corridor = np.asarray(
        corridor["scoredCorridorIndex"], dtype=np.int32
    )
    corridor_index = int(scored_corridor[row])
    pair_offset = np.asarray(corridor["corridorPairOffset"], dtype=np.int64)
    begin, end = (
        int(pair_offset[corridor_index]),
        int(pair_offset[corridor_index + 1]),
    )
    first_edge = np.asarray(
        corridor["corridorFirstBoundaryEdge"], dtype=np.int32
    )[begin:end]
    second_edge = np.asarray(
        corridor["corridorSecondBoundaryEdge"], dtype=np.int32
    )[begin:end]
    boundary_xyz = np.asarray(
        corridor["boundaryEdgeMidpointXYZ"], dtype=np.float32
    )
    return (
        _dominant_region(boundary_xyz[first_edge], triangle_center, triangle_region),
        _dominant_region(boundary_xyz[second_edge], triangle_center, triangle_region),
        triangle_region,
    )


def _face_metrics(
    triangle: np.ndarray,
    surface: Mapping[str, np.ndarray],
    patch_xyz: np.ndarray,
    patch_normal_xyz: np.ndarray,
    patch_thickness_voxels: np.ndarray,
    *,
    raster_step_voxels: float,
) -> tuple[np.ndarray, dict[str, float]]:
    center = np.asarray(surface["midpointXYZ"], dtype=np.float64)
    signed_normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float64)
    oriented, area, node_normal_residual, maximum_edge = _triangle_geometry(
        tuple(int(value) for value in triangle), center, signed_normal
    )
    oriented_array = np.asarray(oriented, dtype=np.int32)
    points = center[oriented_array]
    centroid = np.mean(points, axis=0)
    distance = np.linalg.norm(patch_xyz - centroid, axis=1)
    nearest = int(np.argmin(distance))
    thickness = max(float(patch_thickness_voxels[nearest]), 1.0e-6)
    patch_normal = np.asarray(patch_normal_xyz[nearest], dtype=np.float64)
    patch_normal /= max(float(np.linalg.norm(patch_normal)), 1.0e-12)
    delta = centroid - patch_xyz[nearest]
    signed_height = float(np.dot(delta, patch_normal))
    tangent = math.sqrt(
        max(float(np.dot(delta, delta)) - signed_height**2, 0.0)
    )
    cross = np.cross(points[1] - points[0], points[2] - points[0])
    cross /= max(float(np.linalg.norm(cross)), 1.0e-12)
    ct_normal_residual = math.degrees(
        math.acos(
            np.clip(abs(float(np.dot(cross, patch_normal))), -1.0, 1.0)
        )
    )
    return oriented_array, {
        "areaVoxelsSquared": float(area),
        "nodeNormalResidualDegrees": float(node_normal_residual),
        "maximumEdgeVoxels": float(maximum_edge),
        "centerDistanceThicknesses": float(distance[nearest] / thickness),
        "centerHeightThicknesses": float(abs(signed_height) / thickness),
        "centerTangentRasterSteps": float(
            tangent / max(raster_step_voxels, 1.0e-6)
        ),
        "ctNormalResidualDegrees": float(ct_normal_residual),
        "maximumEdgeThicknesses": float(maximum_edge / thickness),
    }


def _face_is_physical(
    metrics: Mapping[str, float],
    *,
    settings: PhysicalRibbonCorridorFaceSettings,
) -> bool:
    return (
        metrics["areaVoxelsSquared"] > 0.0
        and metrics["centerDistanceThicknesses"]
        <= settings.maximum_center_distance_thicknesses
        and metrics["centerHeightThicknesses"]
        <= settings.maximum_center_height_thicknesses
        and metrics["centerTangentRasterSteps"]
        <= settings.maximum_center_tangent_raster_steps
        and metrics["ctNormalResidualDegrees"]
        <= settings.maximum_ct_normal_residual_degrees
        and metrics["maximumEdgeThicknesses"]
        <= settings.maximum_triangle_edge_thicknesses
    )


def _face_path_cost(
    metrics: Mapping[str, float],
    settings: PhysicalRibbonCorridorFaceSettings,
) -> float:
    return 1.0 + (
        0.35
        * metrics["centerDistanceThicknesses"]
        / settings.maximum_center_distance_thicknesses
        + 0.35
        * metrics["centerHeightThicknesses"]
        / settings.maximum_center_height_thicknesses
        + 0.15
        * metrics["centerTangentRasterSteps"]
        / settings.maximum_center_tangent_raster_steps
        + 0.35
        * metrics["ctNormalResidualDegrees"]
        / settings.maximum_ct_normal_residual_degrees
        + 0.20
        * metrics["maximumEdgeThicknesses"]
        / settings.maximum_triangle_edge_thicknesses
    )


def _minimum_face_path(
    existing_triangles: np.ndarray,
    existing_triangle_region: np.ndarray,
    candidate_triangles: np.ndarray,
    candidate_cost: np.ndarray,
    *,
    first_region: int,
    second_region: int,
    candidate_eligible: np.ndarray | None = None,
) -> tuple[np.ndarray, float]:
    """Find a minimum-cost dual path without changing topology components."""

    if first_region < 0 or second_region < 0 or first_region == second_region:
        return np.empty(0, dtype=np.int32), math.inf
    if candidate_eligible is None:
        candidate_eligible = np.ones(len(candidate_triangles), dtype=bool)
    candidate_eligible = np.asarray(candidate_eligible, dtype=bool)
    edge_entity: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for triangle_index, triangle in enumerate(existing_triangles):
        region = int(existing_triangle_region[triangle_index])
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge_entity[(min(int(first), second), max(int(first), second))].append(
                (0, region)
            )
    for candidate_index, triangle in enumerate(candidate_triangles):
        if not candidate_eligible[candidate_index]:
            continue
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge_entity[(min(int(first), second), max(int(first), second))].append(
                (1, candidate_index)
            )
    adjacency: dict[tuple[int, int], set[tuple[int, int]]] = defaultdict(set)
    for entities in edge_entity.values():
        unique = list(dict.fromkeys(entities))
        for first in unique:
            adjacency[first].update(second for second in unique if second != first)
    start = (0, int(first_region))
    goal = (0, int(second_region))
    best: dict[tuple[int, int], float] = {start: 0.0}
    parent: dict[tuple[int, int], tuple[int, int]] = {}
    pending: list[tuple[float, int, int]] = [(0.0, *start)]
    while pending:
        cost, kind, value = heapq.heappop(pending)
        current = (kind, value)
        if cost != best.get(current):
            continue
        if current == goal:
            path: list[int] = []
            while current != start:
                if current[0] == 1:
                    path.append(current[1])
                current = parent[current]
            path.reverse()
            return np.asarray(path, dtype=np.int32), cost
        for following in sorted(adjacency.get(current, ())):
            step = (
                float(candidate_cost[following[1]])
                if following[0] == 1
                else 0.0
            )
            following_cost = cost + step
            if following_cost + 1.0e-12 >= best.get(following, math.inf):
                continue
            best[following] = following_cost
            parent[following] = current
            heapq.heappush(
                pending, (following_cost, following[0], following[1])
            )
    return np.empty(0, dtype=np.int32), math.inf


def _attached_candidate_closure(
    existing_triangles: np.ndarray,
    candidate_triangles: np.ndarray,
    candidate_eligible: np.ndarray,
) -> np.ndarray:
    """Grow all eligible Delaunay faces attached to the current surface."""

    candidate_eligible = np.asarray(candidate_eligible, dtype=bool)
    edge_candidate: dict[tuple[int, int], list[int]] = defaultdict(list)
    for candidate_index, triangle in enumerate(candidate_triangles):
        if not candidate_eligible[candidate_index]:
            continue
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge_candidate[(min(int(first), second), max(int(first), second))].append(
                candidate_index
            )
    pending: list[int] = []
    queued = np.zeros(len(candidate_triangles), dtype=bool)
    for triangle in existing_triangles:
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            for candidate_index in edge_candidate.get(edge, ()):
                if not queued[candidate_index]:
                    queued[candidate_index] = True
                    pending.append(candidate_index)
    retained = np.zeros(len(candidate_triangles), dtype=bool)
    while pending:
        candidate_index = pending.pop()
        if retained[candidate_index]:
            continue
        retained[candidate_index] = True
        triangle = candidate_triangles[candidate_index]
        for edge_index, first in enumerate(triangle):
            second = int(triangle[(edge_index + 1) % 3])
            edge = (min(int(first), second), max(int(first), second))
            for neighbor in edge_candidate.get(edge, ()):
                if not queued[neighbor]:
                    queued[neighbor] = True
                    pending.append(neighbor)
    return retained


def _screen_corridor_face_path(
    row: int,
    surface: Mapping[str, np.ndarray],
    corridor: Mapping[str, np.ndarray],
    *,
    surface_settings: Any,
    settings: PhysicalRibbonCorridorFaceSettings,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    existing = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    first_region, second_region, existing_region = _corridor_boundary_regions(
        row, surface, corridor
    )
    component = np.asarray(surface["component"], dtype=np.int32)
    region_triangles = existing[
        (existing_region == first_region) | (existing_region == second_region)
    ]
    target_components = {
        int(value)
        for value in np.unique(component[region_triangles])
        if value >= 0
    }
    missing = _raw_missing_delaunay_faces(
        surface,
        minimum_component_ribbon_count=(
            surface_settings.minimum_component_ribbon_count
        ),
        minimum_chart_separation_voxels=(
            surface_settings.minimum_chart_separation_voxels
        ),
        component_ids=target_components,
    )
    patch_offset = np.asarray(corridor["corridorPatchOffset"], dtype=np.int64)
    begin, end = int(patch_offset[row]), int(patch_offset[row + 1])
    patch_xyz = np.asarray(corridor["corridorPatchXYZ"], dtype=np.float32)[
        begin:end
    ]
    patch_normal = np.asarray(
        corridor["corridorPatchNormalXYZ"], dtype=np.float32
    )[begin:end]
    patch_thickness = np.asarray(
        corridor["corridorPatchThicknessVoxels"], dtype=np.float32
    )[begin:end]
    raster_step = float(
        np.asarray(corridor["corridorRasterStepVoxels"], dtype=np.float32)[row]
    )
    oriented: list[np.ndarray] = []
    metrics: list[dict[str, float]] = []
    for triangle in missing:
        value, record = _face_metrics(
            triangle,
            surface,
            patch_xyz,
            patch_normal,
            patch_thickness,
            raster_step_voxels=raster_step,
        )
        oriented.append(value)
        metrics.append(record)
    candidate_triangle = (
        np.asarray(oriented, dtype=np.int32).reshape((-1, 3))
        if oriented
        else np.empty((0, 3), dtype=np.int32)
    )
    physical = np.asarray(
        [
            _face_is_physical(value, settings=settings)
            and value["areaVoxelsSquared"]
            >= surface_settings.minimum_triangle_area_voxels_squared
            for value in metrics
        ],
        dtype=bool,
    )
    cost = np.asarray(
        [_face_path_cost(value, settings) for value in metrics],
        dtype=np.float64,
    )
    raw_path, _ = _minimum_face_path(
        existing,
        existing_region,
        candidate_triangle,
        np.ones(len(candidate_triangle), dtype=np.float64),
        first_region=first_region,
        second_region=second_region,
    )
    path, path_cost = _minimum_face_path(
        existing,
        existing_region,
        candidate_triangle,
        cost,
        first_region=first_region,
        second_region=second_region,
        candidate_eligible=physical,
    )
    physical_closure = _attached_candidate_closure(
        existing, candidate_triangle, physical
    )
    path_triangle = candidate_triangle[path]
    augmented = dict(surface)
    if len(path_triangle):
        augmented["triangleFrontierIndex"] = np.vstack(
            (existing, path_triangle)
        ).astype(np.int32)
    connection = _evaluate_corridor_connections(
        augmented,
        corridor,
        corridor,
        minimum_arc_region_fraction=settings.minimum_arc_region_fraction,
        maximum_arc_triangle_distance_edges=(
            settings.maximum_arc_triangle_distance_edges
        ),
    )
    exact_connected = bool(connection["boundaryArcsConnected"][row])
    eligible = bool(
        len(path)
        and len(path) <= settings.maximum_path_face_count
        and exact_connected
    )
    path_metrics = [metrics[int(index)] for index in path]
    record: dict[str, Any] = {
        "corridorRow": row,
        "firstTriangleRegion": first_region,
        "secondTriangleRegion": second_region,
        "rawMissingDelaunayFaceCount": len(candidate_triangle),
        "physicalCandidateFaceCount": int(np.count_nonzero(physical)),
        "attachedPhysicalClosureFaceCount": int(
            np.count_nonzero(physical_closure)
        ),
        "rawMinimumPathFaceCount": len(raw_path),
        "physicalPathFaceCount": len(path),
        "physicalPathCost": (
            round(float(path_cost), 6) if math.isfinite(path_cost) else None
        ),
        "exactConnected": exact_connected,
        "sharedArcRegionFraction": round(
            float(connection["boundaryArcSharedRegionFraction"][row]), 6
        ),
        "eligible": eligible,
        "path": {
            key: _finite_distribution([value[key] for value in path_metrics])
            for key in (
                "centerDistanceThicknesses",
                "centerHeightThicknesses",
                "centerTangentRasterSteps",
                "ctNormalResidualDegrees",
                "maximumEdgeThicknesses",
                "maximumEdgeVoxels",
                "nodeNormalResidualDegrees",
                "areaVoxelsSquared",
            )
        },
    }
    arrays = {
        "candidateTriangleFrontierIndex": candidate_triangle,
        "candidatePhysicalEligible": physical.astype(np.uint8),
        "candidatePhysicalClosure": physical_closure.astype(np.uint8),
        "candidatePathSelected": np.isin(
            np.arange(len(candidate_triangle)), path
        ).astype(np.uint8),
        "candidatePathCost": cost.astype(np.float32),
    }
    for key in (
        "centerDistanceThicknesses",
        "centerHeightThicknesses",
        "centerTangentRasterSteps",
        "ctNormalResidualDegrees",
        "maximumEdgeThicknesses",
        "maximumEdgeVoxels",
        "nodeNormalResidualDegrees",
        "areaVoxelsSquared",
    ):
        arrays[f"candidate{key[0].upper()}{key[1:]}"] = np.asarray(
            [value[key] for value in metrics], dtype=np.float32
        )
    return record, arrays


def run_physical_ribbon_corridor_faces(
    replay_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorFaceSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorFaceSettings()
    replay_path, replay_manifest, replay = _load_replay_artifact(replay_root)
    if replay_manifest.get("schema") != PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA:
        raise ValueError("corridor faces require an exact one-sided replay")
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
        configuration_path,
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
        raise ValueError("corridor-face replay frontier has changed")
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FACES_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FACES_VERSION,
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
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "deficitImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_deficits.py")
        ),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FACES_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FACES_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    surface_settings = corridor_settings.surface_settings()
    started = time.monotonic()
    if progress is not None:
        progress("reconstructing the exact replay baseline")
    baseline_surface, _ = build_physical_ribbon_surface_complex(
        ribbon,
        frontier,
        frontier,
        settings=surface_settings,
    )
    remapped = _remap_corridor_surface(
        corridor,
        baseline_surface,
        base_configuration,
        frontier,
        np.asarray(frontier["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    evidence = np.asarray(replay["corridorEvidenceEligible"]) > 0
    chosen_exact = np.asarray(
        replay["corridorChosenExactVariant"], dtype=np.int32
    )
    failed_rows = np.flatnonzero(evidence & (chosen_exact < 0))
    records: list[dict[str, Any]] = []
    candidate_offset = [0]
    candidate_arrays: dict[str, list[np.ndarray]] = defaultdict(list)
    variant_index_values: list[int] = []
    for completed, row_value in enumerate(failed_rows, start=1):
        row = int(row_value)
        variant_index = _best_failed_variant(row, replay)
        record: dict[str, Any] = {
            "corridorRow": row,
            "bestFailedVariantIndex": variant_index,
        }
        if variant_index < 0:
            record["status"] = "no-reconstructable-state"
        else:
            local_surface, _, component_id = _reconstruct_failed_variant(
                row,
                variant_index,
                remapped,
                replay,
                ribbon,
                frontier,
                frontier,
                surface_settings=surface_settings,
            )
            screened, arrays = _screen_corridor_face_path(
                row,
                local_surface,
                remapped,
                surface_settings=surface_settings,
                settings=resolved,
            )
            record.update(screened)
            record.update(
                {
                    "status": (
                        "physical-face-path"
                        if screened["eligible"]
                        else "no-physical-face-path"
                    ),
                    "topologyComponent": component_id,
                    "variantRank": int(
                        np.asarray(replay["corridorVariantRank"])[variant_index]
                    ),
                }
            )
            for key, value in arrays.items():
                candidate_arrays[key].append(value)
            candidate_offset.append(
                candidate_offset[-1]
                + len(arrays["candidateTriangleFrontierIndex"])
            )
            variant_index_values.append(variant_index)
        records.append(record)
        if progress is not None and (
            completed == len(failed_rows) or completed % 4 == 0
        ):
            progress(
                f"corridor faces {completed}/{len(failed_rows)} · "
                f"eligible {sum(value.get('eligible', False) for value in records)}"
            )

    arrays: dict[str, np.ndarray] = {
        "screenedCorridorRow": np.asarray(
            [
                int(value["corridorRow"])
                for value in records
                if value.get("bestFailedVariantIndex", -1) >= 0
            ],
            dtype=np.int32,
        ),
        "screenedVariantIndex": np.asarray(variant_index_values, dtype=np.int32),
        "candidateOffset": np.asarray(candidate_offset, dtype=np.int64),
    }
    for key, chunks in candidate_arrays.items():
        tail_shape = chunks[0].shape[1:] if chunks else ()
        arrays[key] = (
            np.concatenate(chunks, axis=0)
            if chunks
            else np.empty((0, *tail_shape), dtype=np.float32)
        )
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    eligible_records = [value for value in records if value.get("eligible")]
    status_count = Counter(value["status"] for value in records)
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FACES_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FACES_VERSION,
        "state": "complete",
        "identity": identity,
        "statistics": {
            "failedCtCorridorCount": len(failed_rows),
            "statusCounts": dict(sorted(status_count.items())),
            "eligiblePhysicalFacePathCount": len(eligible_records),
            "eligibleCorridorRows": [
                int(value["corridorRow"]) for value in eligible_records
            ],
            "pathFaceCount": _finite_distribution(
                [float(value["physicalPathFaceCount"]) for value in eligible_records]
            ),
            "pathCost": _finite_distribution(
                [float(value["physicalPathCost"]) for value in eligible_records]
            ),
            "records": records,
            "identityLabelsUsed": False,
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "method": {
            "sheetIdentity": "unchanged strict ribbon graph components",
            "meshRepair": (
                "minimum path through missing faces of the existing chart "
                "Delaunay tessellation, gated against native CT in units of "
                "local papyrus thickness"
            ),
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
