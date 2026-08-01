from __future__ import annotations

import hashlib
import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    _context_vertices,
    _point_in_polygon,
    _rasterize_polygon,
    _selected_surface_adjacency,
    extract_surface_boundary_loops,
    score_surface_patch_holes,
    write_patch_hole_montage,
)
from .physical_ribbon_patch_states import _surface_view
from .physical_ribbon_surface_holes import _resolve_surface_manifest


PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA = "pareidolia.physical-ribbon-open-bays"
PHYSICAL_RIBBON_OPEN_BAYS_VERSION = 1
PHYSICAL_RIBBON_OPEN_BAYS_STEM = "physical-ribbon-open-bays-v1"

CACHEABLE_COMPLETION_REJECTION_REASONS = frozenset(
    (
        "insufficient collective depth-field CT support",
        "completion contains a physically degenerate triangle",
        "completion contains an overlong triangle edge",
        "completion triangle contradicts the local CT normal field",
        "constructed mesh normals lose whole-patch CT support",
        "constructed surface profile does not match its boundary context",
        "a displaced competing layer explains the constructed surface",
    )
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonOpenBaySettings(PhysicalRibbonPatchHoleSettings):
    """Find compact missing surface bays along one outer sheet frontier.

    A bay is a complete boundary-arc state, never a one-cell growth move.  Its
    existing arc is replaced by one new mouth edge only after the full added
    area has coherent native-CT support.  Geometry enumeration deliberately
    rewards area gain together with boundary shortening, which favors filling
    compact deficits over extending thin tendrils.
    """

    minimum_arc_edge_count: int = 3
    maximum_arc_edge_count: int = 32
    minimum_mouth_width_boundary_edges: float = 0.75
    maximum_mouth_width_thicknesses: float = 4.0
    minimum_perimeter_reduction_boundary_edges: float = 0.50
    minimum_bay_area_chart_voxels_squared: float = 1.0
    minimum_bay_compactness: float = 0.025
    minimum_exterior_raster_fraction: float = 0.90
    owned_boundary_exclusion_voxels: float = 2.0
    geometry_raster_step_voxels: float = 0.75
    maximum_geometry_raster_pixels: int = 2048
    maximum_candidates_per_outer_loop: int = 4
    maximum_arc_overlap_fraction: float = 0.60
    maximum_scored_holes: int = 128
    maximum_preview_holes: int = 24
    patch_pixel_step_voxels: float = 0.75
    maximum_patch_pixels: int = 8192

    def __post_init__(self) -> None:
        PhysicalRibbonPatchHoleSettings.__post_init__(self)
        if self.minimum_arc_edge_count < 2:
            raise ValueError("open bays require a multi-edge existing frontier")
        if self.maximum_arc_edge_count < self.minimum_arc_edge_count:
            raise ValueError("open-bay arc edge limits must be ordered")
        positive = (
            self.minimum_mouth_width_boundary_edges,
            self.maximum_mouth_width_thicknesses,
            self.minimum_perimeter_reduction_boundary_edges,
            self.minimum_bay_area_chart_voxels_squared,
            self.minimum_bay_compactness,
            self.minimum_exterior_raster_fraction,
            self.owned_boundary_exclusion_voxels,
            self.geometry_raster_step_voxels,
            self.maximum_arc_overlap_fraction,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("open-bay physical scales must be finite and positive")
        if not 0.0 < self.minimum_exterior_raster_fraction <= 1.0:
            raise ValueError("open-bay exterior fraction must lie in (0, 1]")
        if not 0.0 < self.maximum_arc_overlap_fraction <= 1.0:
            raise ValueError("open-bay overlap fraction must lie in (0, 1]")
        if self.maximum_geometry_raster_pixels < 64:
            raise ValueError("open-bay geometry raster cap is too small")
        if self.maximum_candidates_per_outer_loop < 1:
            raise ValueError("open-bay per-loop candidate cap must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _cross_2d(first: np.ndarray, second: np.ndarray, third: np.ndarray) -> float:
    left = second - first
    right = third - first
    return float(left[0] * right[1] - left[1] * right[0])


def _proper_segments_intersect(
    first_start: np.ndarray,
    first_stop: np.ndarray,
    second_start: np.ndarray,
    second_stop: np.ndarray,
) -> bool:
    first_side = (
        _cross_2d(first_start, first_stop, second_start),
        _cross_2d(first_start, first_stop, second_stop),
    )
    second_side = (
        _cross_2d(second_start, second_stop, first_start),
        _cross_2d(second_start, second_stop, first_stop),
    )
    return (
        first_side[0] * first_side[1] < -1.0e-10
        and second_side[0] * second_side[1] < -1.0e-10
    )


def _simple_polygon(points: np.ndarray) -> bool:
    polygon = np.asarray(points, dtype=np.float64)
    if len(polygon) < 3:
        return False
    for first in range(len(polygon)):
        first_next = (first + 1) % len(polygon)
        for second in range(first + 1, len(polygon)):
            second_next = (second + 1) % len(polygon)
            if len({first, first_next, second, second_next}) < 4:
                continue
            if _proper_segments_intersect(
                polygon[first],
                polygon[first_next],
                polygon[second],
                polygon[second_next],
            ):
                return False
    return True


def _polygon_area(points: np.ndarray) -> float:
    polygon = np.asarray(points, dtype=np.float64)
    return 0.5 * abs(
        float(
            np.sum(
                polygon[:, 0] * np.roll(polygon[:, 1], -1)
                - polygon[:, 1] * np.roll(polygon[:, 0], -1)
            )
        )
    )


def _arc_edges(nodes: np.ndarray) -> frozenset[tuple[int, int]]:
    return frozenset(
        (min(int(first), int(second)), max(int(first), int(second)))
        for first, second in zip(nodes, nodes[1:])
    )


def _hash_array(digest: Any, name: str, values: np.ndarray) -> None:
    array = np.ascontiguousarray(values)
    digest.update(name.encode("utf-8"))
    digest.update(str(array.dtype).encode("ascii"))
    digest.update(np.asarray(array.shape, dtype=np.int64).tobytes())
    digest.update(array.tobytes())


def _bay_evidence_fingerprint(
    surface: Mapping[str, np.ndarray],
    boundary: np.ndarray,
    context: np.ndarray,
    component_id: int,
) -> str:
    """Hash every deterministic input to bay fitting and native-CT scoring."""

    nodes = tuple(int(value) for value in np.asarray(boundary, dtype=np.int32))
    reverse = tuple(reversed(nodes))
    canonical_boundary = np.asarray(min(nodes, reverse), dtype=np.int32)
    context_nodes = np.asarray(sorted(int(value) for value in context), dtype=np.int32)
    digest = hashlib.sha256()
    digest.update(np.asarray((component_id,), dtype=np.int32).tobytes())
    _hash_array(digest, "boundary", canonical_boundary)
    _hash_array(digest, "context", context_nodes)
    for name in (
        "chartUV",
        "midpointXYZ",
        "signedNormalXYZ",
        "thicknessVoxels",
    ):
        _hash_array(digest, name, np.asarray(surface[name])[context_nodes])
    return digest.hexdigest()


def _owned_boundary_distance(
    points_xyz: np.ndarray,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    if owned_bounds is None or not len(points_xyz):
        return float("inf")
    low, high = owned_bounds
    distance = np.concatenate(
        (points_xyz - low[None, :], (high - 1.0)[None, :] - points_xyz),
        axis=1,
    )
    return float(np.min(distance))


def _loop_open_bay_candidates(
    nodes: np.ndarray,
    chart_uv: np.ndarray,
    midpoint_xyz: np.ndarray,
    thickness: np.ndarray,
    *,
    outer_loop_index: int,
    component_id: int,
    triangle_region: int,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
    settings: PhysicalRibbonOpenBaySettings,
    rejection_count: Counter[str],
) -> list[dict[str, Any]]:
    """Enumerate exterior concavities on one simple outer chart polygon."""

    boundary = np.asarray(nodes, dtype=np.int32)
    if len(boundary) < settings.minimum_arc_edge_count + 2:
        rejection_count["outer loop too short"] += 1
        return []
    if len(np.unique(boundary)) != len(boundary):
        rejection_count["outer loop repeats a pinch vertex"] += 1
        return []
    outer = np.asarray(chart_uv[boundary], dtype=np.float64)
    if not _simple_polygon(outer):
        rejection_count["outer loop is not simple in its intrinsic chart"] += 1
        return []
    full_edges = tuple(
        (int(boundary[index]), int(boundary[(index + 1) % len(boundary)]))
        for index in range(len(boundary))
    )
    candidates: list[dict[str, Any]] = []
    maximum_arc = min(settings.maximum_arc_edge_count, len(boundary) - 2)
    for start in range(len(boundary)):
        for edge_count in range(settings.minimum_arc_edge_count, maximum_arc + 1):
            position = (start + np.arange(edge_count + 1)) % len(boundary)
            arc = boundary[position]
            if len(np.unique(arc)) != len(arc):
                rejection_count["candidate arc repeats a vertex"] += 1
                continue
            polygon = np.asarray(chart_uv[arc], dtype=np.float64)
            if not _simple_polygon(polygon):
                rejection_count["candidate bay is not simple"] += 1
                continue
            edge_length = np.linalg.norm(polygon[1:] - polygon[:-1], axis=1)
            arc_length = float(np.sum(edge_length))
            mean_edge = arc_length / max(edge_count, 1)
            mouth_length = float(np.linalg.norm(polygon[-1] - polygon[0]))
            median_thickness = float(np.median(thickness[arc]))
            if mouth_length < settings.minimum_mouth_width_boundary_edges * mean_edge:
                rejection_count["mouth is below local boundary resolution"] += 1
                continue
            if mouth_length > settings.maximum_mouth_width_thicknesses * median_thickness:
                rejection_count["mouth exceeds the local physical scale"] += 1
                continue
            reduction = arc_length - mouth_length
            if reduction < settings.minimum_perimeter_reduction_boundary_edges * mean_edge:
                rejection_count["arc does not materially shorten the frontier"] += 1
                continue
            area = _polygon_area(polygon)
            if area < settings.minimum_bay_area_chart_voxels_squared:
                rejection_count["candidate bay area is below resolution"] += 1
                continue
            perimeter = arc_length + mouth_length
            compactness = 4.0 * math.pi * area / max(perimeter * perimeter, 1.0e-12)
            if compactness < settings.minimum_bay_compactness:
                rejection_count["candidate bay is a thin tendril"] += 1
                continue

            first_node, second_node = int(arc[0]), int(arc[-1])
            mouth_start, mouth_stop = polygon[0], polygon[-1]
            intersects = False
            for edge_start_node, edge_stop_node in full_edges:
                if first_node in (edge_start_node, edge_stop_node) or second_node in (
                    edge_start_node,
                    edge_stop_node,
                ):
                    continue
                if _proper_segments_intersect(
                    mouth_start,
                    mouth_stop,
                    chart_uv[edge_start_node],
                    chart_uv[edge_stop_node],
                ):
                    intersects = True
                    break
            if intersects:
                rejection_count["mouth crosses the existing outer frontier"] += 1
                continue

            # A noncrossing chord through the occupied side cuts off existing
            # surface rather than filling a bay.  This exact mouth-side test is
            # cheap enough for every hypothesis; dense exterior rastering is
            # deferred until after per-loop dominance suppression.
            if _point_in_polygon(0.5 * (mouth_start + mouth_stop), outer):
                rejection_count["candidate lies on the occupied side of the frontier"] += 1
                continue
            owned_distance = _owned_boundary_distance(
                midpoint_xyz[arc], owned_bounds
            )
            if owned_distance <= settings.owned_boundary_exclusion_voxels:
                rejection_count["candidate arc reaches an owned-volume face"] += 1
                continue
            objective = (
                area
                * reduction
                * compactness
                / max(mouth_length, mean_edge, 1.0e-6)
            )
            candidates.append(
                {
                    "outerLoopIndex": outer_loop_index,
                    "component": component_id,
                    "triangleRegion": triangle_region,
                    "arcStartPosition": start,
                    "arcEdgeCount": edge_count,
                    "arcNodes": arc.astype(np.int32),
                    "arcEdges": _arc_edges(arc),
                    "mouthFirst": first_node,
                    "mouthSecond": second_node,
                    "mouthLengthVoxels": mouth_length,
                    "arcLengthVoxels": arc_length,
                    "perimeterReductionVoxels": reduction,
                    "areaChartVoxelsSquared": area,
                    "compactness": compactness,
                    "exteriorRasterFraction": float("nan"),
                    "ownedBoundaryDistanceVoxels": owned_distance,
                    "medianThicknessVoxels": median_thickness,
                    "objective": objective,
                }
            )
    return candidates


def enumerate_surface_open_bays(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    *,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
    settings: PhysicalRibbonOpenBaySettings,
    excluded_evidence_fingerprints: frozenset[str] = frozenset(),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    midpoint_xyz = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    vertex = np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32)
    kind = np.asarray(loops["loopKind"], dtype=np.uint8)
    component = np.asarray(loops["loopTopologyComponent"], dtype=np.int32)
    region = np.asarray(loops["loopTriangleRegion"], dtype=np.int32)
    surface_component = np.asarray(surface["component"], dtype=np.int32)
    adjacency = _selected_surface_adjacency(surface)
    rejection_count: Counter[str] = Counter()
    by_loop: dict[int, list[dict[str, Any]]] = defaultdict(list)
    outer_loop = np.flatnonzero(kind == 0)
    outer_touch_count = 0
    for loop_index in outer_loop:
        nodes = vertex[int(offset[loop_index]) : int(offset[loop_index + 1])]
        if (
            _owned_boundary_distance(midpoint_xyz[nodes], owned_bounds)
            <= settings.owned_boundary_exclusion_voxels
        ):
            outer_touch_count += 1
        by_loop[int(loop_index)].extend(
            _loop_open_bay_candidates(
                nodes,
                chart_uv,
                midpoint_xyz,
                thickness,
                outer_loop_index=int(loop_index),
                component_id=int(component[loop_index]),
                triangle_region=int(region[loop_index]),
                owned_bounds=owned_bounds,
                settings=settings,
                rejection_count=rejection_count,
            )
        )

    geometry_ranked: list[dict[str, Any]] = []
    suppressed_overlap_count = 0
    for loop_index in sorted(by_loop):
        ranked = sorted(
            by_loop[loop_index],
            key=lambda record: (
                -float(record["objective"]),
                -float(record["areaChartVoxelsSquared"]),
                int(record["arcStartPosition"]),
                int(record["arcEdgeCount"]),
            ),
        )
        local: list[dict[str, Any]] = []
        for record in ranked:
            current_edges = record["arcEdges"]
            if any(
                len(current_edges & previous["arcEdges"])
                / max(min(len(current_edges), len(previous["arcEdges"])), 1)
                >= settings.maximum_arc_overlap_fraction
                for previous in local
            ):
                suppressed_overlap_count += 1
                continue
            local.append(record)
            if len(local) >= settings.maximum_candidates_per_outer_loop:
                break
        geometry_ranked.extend(local)
    geometry_ranked.sort(
        key=lambda record: (
            -float(record["objective"]),
            -float(record["areaChartVoxelsSquared"]),
            int(record["outerLoopIndex"]),
        )
    )
    retained: list[dict[str, Any]] = []
    cached_rejection_count = 0
    for record in geometry_ranked:
        polygon = chart_uv[record["arcNodes"]]
        context = _context_vertices(
            np.asarray(record["arcNodes"], dtype=np.int32),
            adjacency,
            surface_component,
            graph_hops=settings.context_graph_hops,
        )
        evidence_fingerprint = _bay_evidence_fingerprint(
            surface,
            np.asarray(record["arcNodes"], dtype=np.int32),
            context,
            int(record["component"]),
        )
        if evidence_fingerprint in excluded_evidence_fingerprints:
            cached_rejection_count += 1
            continue
        outer_loop_index = int(record["outerLoopIndex"])
        outer_nodes = vertex[
            int(offset[outer_loop_index]) : int(offset[outer_loop_index + 1])
        ]
        outer = chart_uv[outer_nodes]
        raster, _ = _rasterize_polygon(
            polygon,
            requested_step=settings.geometry_raster_step_voxels,
            maximum_pixels=settings.maximum_geometry_raster_pixels,
        )
        exterior_fraction = float(
            np.mean([not _point_in_polygon(point, outer) for point in raster])
        )
        if exterior_fraction < settings.minimum_exterior_raster_fraction:
            rejection_count[
                "candidate raster overlaps the occupied side of the frontier"
            ] += 1
            continue
        record["exteriorRasterFraction"] = exterior_fraction
        record["objective"] = float(record["objective"]) * exterior_fraction
        record["evidenceFingerprint"] = evidence_fingerprint
        retained.append(record)
        if len(retained) >= settings.maximum_scored_holes:
            break

    loop_offset = [0]
    loop_vertex: list[int] = []
    for record in retained:
        loop_vertex.extend(int(value) for value in record["arcNodes"])
        loop_offset.append(len(loop_vertex))
    count = len(retained)
    arrays = {
        "loopOffset": np.asarray(loop_offset, dtype=np.int64),
        "loopVertexFrontierIndex": np.asarray(loop_vertex, dtype=np.int32),
        "loopTriangleRegion": np.asarray(
            [record["triangleRegion"] for record in retained], dtype=np.int32
        ),
        "loopTopologyComponent": np.asarray(
            [record["component"] for record in retained], dtype=np.int32
        ),
        "loopKind": np.ones(count, dtype=np.uint8),
        "loopAreaChartVoxelsSquared": np.asarray(
            [record["areaChartVoxelsSquared"] for record in retained],
            dtype=np.float32,
        ),
        "loopPerimeterChartVoxels": np.asarray(
            [
                record["arcLengthVoxels"] + record["mouthLengthVoxels"]
                for record in retained
            ],
            dtype=np.float32,
        ),
        "loopDiameterChartVoxels": np.asarray(
            [
                np.max(
                    np.linalg.norm(
                        chart_uv[record["arcNodes"]][:, None, :]
                        - chart_uv[record["arcNodes"]][None, :, :],
                        axis=2,
                    )
                )
                for record in retained
            ],
            dtype=np.float32,
        ),
        "loopMedianThicknessVoxels": np.asarray(
            [record["medianThicknessVoxels"] for record in retained],
            dtype=np.float32,
        ),
        "loopMeanBoundaryEdgeVoxels": np.asarray(
            [
                (record["arcLengthVoxels"] + record["mouthLengthVoxels"])
                / max(len(record["arcNodes"]), 1)
                for record in retained
            ],
            dtype=np.float32,
        ),
        "loopMacroEligible": np.ones(count, dtype=np.uint8),
        "baySourceOuterLoopIndex": np.asarray(
            [record["outerLoopIndex"] for record in retained], dtype=np.int32
        ),
        "bayArcStartPosition": np.asarray(
            [record["arcStartPosition"] for record in retained], dtype=np.int32
        ),
        "bayArcEdgeCount": np.asarray(
            [record["arcEdgeCount"] for record in retained], dtype=np.int16
        ),
        "bayMouthFirstFrontierIndex": np.asarray(
            [record["mouthFirst"] for record in retained], dtype=np.int32
        ),
        "bayMouthSecondFrontierIndex": np.asarray(
            [record["mouthSecond"] for record in retained], dtype=np.int32
        ),
        "bayMouthLengthVoxels": np.asarray(
            [record["mouthLengthVoxels"] for record in retained], dtype=np.float32
        ),
        "bayArcLengthVoxels": np.asarray(
            [record["arcLengthVoxels"] for record in retained], dtype=np.float32
        ),
        "bayPerimeterReductionVoxels": np.asarray(
            [record["perimeterReductionVoxels"] for record in retained],
            dtype=np.float32,
        ),
        "bayCompactness": np.asarray(
            [record["compactness"] for record in retained], dtype=np.float32
        ),
        "bayExteriorRasterFraction": np.asarray(
            [record["exteriorRasterFraction"] for record in retained],
            dtype=np.float32,
        ),
        "bayOwnedBoundaryDistanceVoxels": np.asarray(
            [record["ownedBoundaryDistanceVoxels"] for record in retained],
            dtype=np.float32,
        ),
        "bayGeometryObjective": np.asarray(
            [record["objective"] for record in retained], dtype=np.float32
        ),
        "bayEvidenceSha256": np.asarray(
            [record["evidenceFingerprint"] for record in retained], dtype="S64"
        ),
    }
    return arrays, {
        "outerBoundaryLoopCount": int(len(outer_loop)),
        "outerLoopsTouchingOwnedBoundaryCount": outer_touch_count,
        "rawGeometryCandidateCount": int(sum(len(value) for value in by_loop.values())),
        "outerLoopsWithGeometryCandidateCount": int(
            sum(bool(value) for value in by_loop.values())
        ),
        "overlapSuppressedCandidateCount": suppressed_overlap_count,
        "retainedGeometryCandidateCount": count,
        "cachedEvidenceRejectionCount": cached_rejection_count,
        "excludedEvidenceFingerprintCount": len(excluded_evidence_fingerprints),
        "geometryRejectionCount": dict(sorted(rejection_count.items())),
        "decisionUnit": "one complete existing boundary arc plus one proposed mouth",
        "singleCellGrowth": False,
        "identityLabelsUsed": False,
    }


def build_physical_ribbon_open_bays(
    surface: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
    settings: PhysicalRibbonOpenBaySettings,
    excluded_evidence_fingerprints: frozenset[str] = frozenset(),
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    surface_view = _surface_view(surface)
    source_loops, loop_statistics = extract_surface_boundary_loops(
        surface_view, settings=settings
    )
    bay_loops, bay_statistics = enumerate_surface_open_bays(
        surface_view,
        source_loops,
        owned_bounds=owned_bounds,
        settings=settings,
        excluded_evidence_fingerprints=excluded_evidence_fingerprints,
    )
    selected = np.arange(len(bay_loops["loopKind"]), dtype=np.int32)
    scored, scoring_statistics = score_surface_patch_holes(
        surface_view,
        bay_loops,
        source,
        settings=settings,
        loop_indices=selected,
    )
    count = len(selected)
    arrays = {
        **surface_view,
        **bay_loops,
        **scored,
        "patchCandidateOffset": np.zeros(count + 1, dtype=np.int64),
        "patchCandidateFrontierIndex": np.empty(0, dtype=np.int32),
        "patchCandidateNearestPixel": np.empty(0, dtype=np.int32),
        "patchCandidateSurfaceAlignment": np.empty(0, dtype=np.float32),
        "candidateBankUsed": np.zeros(1, dtype=np.uint8),
    }
    return arrays, {
        "sourceLoops": loop_statistics,
        "bayGeometry": bay_statistics,
        "scoring": scoring_statistics,
        "candidateBankUsed": False,
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def _resolve_prior_manifest(
    root: str | Path,
    *,
    schema: str,
    label: str,
) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema") == schema and manifest.get("state") == "complete":
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(f"{label} root must identify one complete artifact")
    return matches[0]


def _prior_completion_holes(
    completion_manifest: Mapping[str, Any],
    *,
    settings: PhysicalRibbonOpenBaySettings,
    source_identity: Mapping[str, Any],
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    reference = completion_manifest.get("identity", {}).get("holes", {})
    path_value = reference.get("manifestPath")
    if not path_value:
        raise ValueError("prior completion does not identify its bay artifact")
    path = Path(str(path_value))
    if not path.is_file() or sha256_file(path) != reference.get("manifestSha256"):
        raise ValueError("prior completion bay manifest changed")
    manifest = json.loads(path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("data", {}).get("sha256")
        != reference.get("dataSha256")
    ):
        raise ValueError("prior completion was not built from open bays")
    if canonical_json_hash(manifest.get("identity", {}).get("settings", {})) != (
        canonical_json_hash(settings.record())
    ):
        raise ValueError("prior open-bay settings differ from the current run")
    if canonical_json_hash(manifest.get("identity", {}).get("source", {})) != (
        canonical_json_hash(source_identity)
    ):
        raise ValueError("prior open-bay evidence belongs to another volume")
    data = _load_npz(
        path.parent / str(manifest["data"]["path"]),
        str(reference["dataSha256"]),
    )
    return manifest, data


def _completion_record_fingerprint(
    holes: Mapping[str, np.ndarray],
    record: Mapping[str, Any],
) -> str:
    row = int(record["holeRow"])
    loop = int(record["loopIndex"])
    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    if row < 0 or row >= len(scored_loop) or int(scored_loop[row]) != loop:
        raise ValueError("prior completion row and bay loop differ")
    loop_offset = np.asarray(holes["loopOffset"], dtype=np.int64)
    loop_vertex = np.asarray(holes["loopVertexFrontierIndex"], dtype=np.int32)
    boundary = loop_vertex[
        int(loop_offset[loop]) : int(loop_offset[loop + 1])
    ]
    context_offset = np.asarray(holes["contextOffset"], dtype=np.int64)
    context_vertex = np.asarray(
        holes["contextVertexFrontierIndex"], dtype=np.int32
    )
    context = context_vertex[
        int(context_offset[row]) : int(context_offset[row + 1])
    ]
    return _bay_evidence_fingerprint(
        holes,
        boundary,
        context,
        int(np.asarray(holes["loopTopologyComponent"])[loop]),
    )


def _cached_rejection_evidence(
    prior_completion_roots: Sequence[str | Path],
    prior_texture_audit_roots: Sequence[str | Path],
    *,
    settings: PhysicalRibbonOpenBaySettings,
    source_identity: Mapping[str, Any],
    current_surface_reference: Mapping[str, Any],
) -> tuple[frozenset[str], list[dict[str, Any]], dict[str, int]]:
    fingerprints: set[str] = set()
    references: list[dict[str, Any]] = []
    intrinsic_count = 0
    same_surface_count = 0
    texture_count = 0
    completion_cache: dict[
        Path, tuple[dict[str, Any], dict[str, np.ndarray]]
    ] = {}

    def completion_inputs(
        path: Path,
        manifest: dict[str, Any],
    ) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
        cached = completion_cache.get(path)
        if cached is None:
            cached = _prior_completion_holes(
                manifest,
                settings=settings,
                source_identity=source_identity,
            )
            completion_cache[path] = cached
        return cached

    for root in prior_completion_roots:
        path, manifest = _resolve_prior_manifest(
            root,
            schema="pareidolia.physical-ribbon-dense-completion",
            label="prior completion",
        )
        holes_manifest, holes = completion_inputs(path, manifest)
        same_surface = (
            holes_manifest.get("identity", {}).get("surface")
            == current_surface_reference
        )
        for record in manifest.get("completions", ()):
            reasons = set(record.get("rejectionReasons", ()))
            if bool(record.get("accepted")):
                continue
            intrinsic = bool(reasons & CACHEABLE_COMPLETION_REJECTION_REASONS)
            if not intrinsic and not same_surface:
                continue
            fingerprints.add(_completion_record_fingerprint(holes, record))
            if intrinsic:
                intrinsic_count += 1
            else:
                same_surface_count += 1
        references.append(
            {
                "kind": "intrinsicCompletionRejections",
                "manifestPath": str(path),
                "manifestSha256": sha256_file(path),
            }
        )

    for root in prior_texture_audit_roots:
        audit_path, audit_manifest = _resolve_prior_manifest(
            root,
            schema="pareidolia.physical-ribbon-flattened-audit",
            label="prior texture audit",
        )
        surface_reference = audit_manifest.get("identity", {}).get("surface", {})
        path_value = surface_reference.get("manifestPath")
        if not path_value:
            raise ValueError("prior texture audit does not identify its completion")
        completion_path = Path(str(path_value))
        if (
            not completion_path.is_file()
            or sha256_file(completion_path)
            != surface_reference.get("manifestSha256")
        ):
            raise ValueError("prior texture-audit completion changed")
        completion_manifest = json.loads(completion_path.read_text())
        if (
            completion_manifest.get("schema")
            != "pareidolia.physical-ribbon-dense-completion"
            or completion_manifest.get("data", {}).get("sha256")
            != surface_reference.get("dataSha256")
        ):
            raise ValueError("prior texture audit source is not a dense completion")
        _, holes = completion_inputs(completion_path, completion_manifest)
        excluded_rows = {
            int(value)
            for key in (
                "boundaryTextureIncompatibleCompletionHoleRows",
                "boundaryTextureUnmeasuredCompletionHoleRows",
            )
            for value in audit_manifest.get("audit", {}).get(key, ())
        }
        record_by_row = {
            int(record["holeRow"]): record
            for record in completion_manifest.get("completions", ())
            if bool(record.get("accepted"))
        }
        if not excluded_rows.issubset(record_by_row):
            raise ValueError("texture-excluded row is not an accepted completion")
        for row in excluded_rows:
            fingerprints.add(
                _completion_record_fingerprint(holes, record_by_row[row])
            )
            texture_count += 1
        references.append(
            {
                "kind": "incompatibleOrUnmeasuredTexture",
                "manifestPath": str(audit_path),
                "manifestSha256": sha256_file(audit_path),
            }
        )

    return frozenset(fingerprints), references, {
        "priorIntrinsicRejectionCount": intrinsic_count,
        "priorSameSurfaceRejectionCount": same_surface_count,
        "priorTextureRejectionCount": texture_count,
        "uniqueCachedEvidenceFingerprintCount": len(fingerprints),
    }


def run_physical_ribbon_open_bays(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonOpenBaySettings | None = None,
    prior_completion_roots: Sequence[str | Path] = (),
    prior_texture_audit_roots: Sequence[str | Path] = (),
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonOpenBaySettings()
    surface_path, surface_manifest = _resolve_surface_manifest(surface_root)
    surface = _load_npz(
        surface_path.parent / str(surface_manifest["data"]["path"]),
        surface_manifest["data"]["sha256"],
    )
    current_surface_reference = {
        "manifestPath": str(surface_path),
        "manifestSha256": sha256_file(surface_path),
        "dataSha256": surface_manifest["data"]["sha256"],
    }
    source_record = surface_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    (
        excluded_evidence_fingerprints,
        rejection_evidence_references,
        rejection_cache_statistics,
    ) = _cached_rejection_evidence(
        prior_completion_roots,
        prior_texture_audit_roots,
        settings=resolved,
        source_identity=source.source_identity,
        current_surface_reference=current_surface_reference,
    )
    geometry = surface_manifest.get("geometry", {})
    world_bounds = geometry.get("ownedWorldBounds")
    owned_bounds = (
        (
            np.asarray(world_bounds["startXYZ"], dtype=np.float64),
            np.asarray(world_bounds["stopXYZExclusive"], dtype=np.float64),
        )
        if world_bounds is not None
        else None
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA,
        "version": PHYSICAL_RIBBON_OPEN_BAYS_VERSION,
        "surface": current_surface_reference,
        "topologyContinuity": surface_manifest["identity"][
            "topologyContinuity"
        ],
        "source": source.source_identity,
        "settings": resolved.record(),
        "priorRejectionEvidence": rejection_evidence_references,
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_OPEN_BAYS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_OPEN_BAYS_STEM}.npz"
    preview_path = output / "physical-ribbon-open-bays.png"
    if (
        not force
        and manifest_path.is_file()
        and data_path.is_file()
        and preview_path.is_file()
    ):
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    arrays, statistics = build_physical_ribbon_open_bays(
        surface,
        source,
        owned_bounds=owned_bounds,
        settings=resolved,
        excluded_evidence_fingerprints=excluded_evidence_fingerprints,
    )
    statistics["rejectionCache"] = rejection_cache_statistics
    analyzed = time.monotonic()
    _write_npz(data_path, arrays)
    write_patch_hole_montage(
        arrays,
        arrays,
        arrays,
        preview_path,
        maximum_holes=resolved.maximum_preview_holes,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA,
        "version": PHYSICAL_RIBBON_OPEN_BAYS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": geometry,
        "analysis": statistics,
        "timingSeconds": {
            "enumerationAndNativeCt": round(analyzed - started, 6),
            "writingAndPreview": round(finished - analyzed, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"openBayMontage": preview_path.name},
        "method": {
            "decisionUnit": (
                "one complete multi-edge outer-boundary arc and one proposed "
                "replacement mouth, scored as one surface state"
            ),
            "densityObjective": (
                "rank gained exterior chart area together with exact frontier "
                "shortening; thin outward tendrils are rejected geometrically"
            ),
            "rawCtEvidence": (
                "whole-bay air-material-air profiles against the inherited "
                "same-sheet arc context and displaced competing layers"
            ),
            "blockExitHandling": (
                "arcs reaching an owned-volume face are classified as exits "
                "and excluded from autonomous completion"
            ),
            "candidateRole": "no ribbon-bank candidate is required or consulted",
            "rejectionCache": (
                "intrinsic exact failures are reused only when the complete "
                "arc plus its fitted context has an identical evidence hash; "
                "collision and topology failures are reused only while the "
                "entire source surface is byte-identical; "
                "proposal-local texture-incompatible or unmeasured states "
                "are likewise excluded, while topology and collision-only "
                "failures are reconsidered after any surface mutation"
            ),
            "selectionMutated": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
