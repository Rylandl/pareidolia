from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_patch_corridors import (
    PhysicalRibbonPatchCorridorSettings,
    extract_surface_patch_corridors,
    score_surface_patch_corridors,
    write_patch_corridor_montage,
)
from .physical_ribbon_patch_holes import extract_surface_boundary_loops
from .physical_ribbon_surface_holes import _resolve_surface_manifest
from .surface_topology import split_nonmanifold_surface_vertices


PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA = (
    "pareidolia.physical-ribbon-surface-corridors"
)
PHYSICAL_RIBBON_SURFACE_CORRIDORS_VERSION = 1
PHYSICAL_RIBBON_SURFACE_CORRIDORS_STEM = (
    "physical-ribbon-surface-corridors-v1"
)


def _tuple_fields(values: dict[str, Any]) -> dict[str, Any]:
    for name in (
        "hermite_tensions",
        "profile_depth_fractions",
        "competing_shift_thicknesses",
    ):
        if name in values:
            values[name] = tuple(values[name])
    return values


@dataclass(frozen=True, slots=True)
class PhysicalRibbonSurfaceCorridorSettings:
    """Find complete paired-frontier states on any materialized surface."""

    owned_boundary_exclusion_voxels: float = 2.0
    require_distinct_triangle_regions: bool = True
    corridors: PhysicalRibbonPatchCorridorSettings = field(
        default_factory=lambda: PhysicalRibbonPatchCorridorSettings(
            maximum_scored_corridors=256,
            maximum_preview_corridors=32,
        )
    )

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.owned_boundary_exclusion_voxels)
            or self.owned_boundary_exclusion_voxels <= 0.0
        ):
            raise ValueError(
                "surface-corridor owned-boundary exclusion must be positive"
            )

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any]
    ) -> "PhysicalRibbonSurfaceCorridorSettings":
        allowed = {
            "owned_boundary_exclusion_voxels",
            "require_distinct_triangle_regions",
            "corridors",
        }
        unexpected = set(record) - allowed
        if unexpected:
            raise ValueError(
                "unknown surface-corridor settings: "
                + ", ".join(sorted(unexpected))
            )
        corridor_values = record.get("corridors", {})
        if not isinstance(corridor_values, Mapping):
            raise ValueError("surface-corridor settings require a corridors object")
        values = _tuple_fields(dict(corridor_values))
        values.setdefault("maximum_scored_corridors", 256)
        values.setdefault("maximum_preview_corridors", 32)
        return cls(
            owned_boundary_exclusion_voxels=float(
                record.get("owned_boundary_exclusion_voxels", 2.0)
            ),
            require_distinct_triangle_regions=bool(
                record.get("require_distinct_triangle_regions", True)
            ),
            corridors=PhysicalRibbonPatchCorridorSettings(**values),
        )

    def record(self) -> dict[str, Any]:
        return {
            "owned_boundary_exclusion_voxels": (
                self.owned_boundary_exclusion_voxels
            ),
            "require_distinct_triangle_regions": (
                self.require_distinct_triangle_regions
            ),
            "corridors": self.corridors.record(),
        }


def _finite_distribution(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "minimum": round(float(np.min(finite)), 6),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90.0)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def _owned_boundary_distance(
    points_xyz: np.ndarray,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> float:
    points = np.asarray(points_xyz, dtype=np.float64)
    if owned_bounds is None or not len(points):
        return math.inf
    low, high = owned_bounds
    distance = np.concatenate(
        (points - low[None, :], (high - 1.0)[None, :] - points), axis=1
    )
    return float(np.min(distance))


def _loop_vertex_sets(
    loops: Mapping[str, np.ndarray], loop_rows: np.ndarray
) -> dict[int, set[int]]:
    offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    vertex = np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32)
    return {
        int(row): set(
            int(value)
            for value in vertex[int(offset[row]) : int(offset[row + 1])]
        )
        for row in np.unique(loop_rows)
    }


def _cyclic_edge_span(
    start: int,
    stop: int,
    count: int,
    *,
    direction: int,
) -> list[int]:
    """Return an inclusive, proper cyclic edge span in one direction."""

    if direction not in (-1, 1):
        raise ValueError("corridor arc direction must be signed")
    if count < 3 or not 0 <= start < count or not 0 <= stop < count:
        raise ValueError("corridor arc positions are outside their loop")
    result = [start]
    while result[-1] != stop:
        if len(result) >= count:
            raise ValueError("corridor arc consumes a complete boundary loop")
        result.append((result[-1] + direction) % count)
    return result


def _arc_node_walk(
    loop_nodes: np.ndarray,
    anchor_positions: np.ndarray,
    *,
    direction: int,
) -> tuple[np.ndarray, int]:
    """Expand sparse monotone edge anchors to one exact boundary-node walk."""

    nodes = np.asarray(loop_nodes, dtype=np.int32)
    anchors = np.asarray(anchor_positions, dtype=np.int32)
    if len(anchors) < 2:
        raise ValueError("corridor completion requires multiple anchors per arc")
    positions = _cyclic_edge_span(
        int(anchors[0]),
        int(anchors[-1]),
        len(nodes),
        direction=direction,
    )
    position_index = {value: index for index, value in enumerate(positions)}
    try:
        anchor_progress = [position_index[int(value)] for value in anchors]
    except KeyError as error:
        raise ValueError("corridor anchor lies outside its completed arc") from error
    if any(
        second <= first
        for first, second in zip(anchor_progress, anchor_progress[1:])
    ):
        raise ValueError("corridor anchors are not monotone along their arc")
    if direction > 0:
        walk = [int(nodes[positions[0]])]
        walk.extend(int(nodes[(value + 1) % len(nodes)]) for value in positions)
    else:
        walk = [int(nodes[(positions[0] + 1) % len(nodes)])]
        walk.extend(int(nodes[value]) for value in positions)
    return np.asarray(walk, dtype=np.int32), len(positions)


def _surface_edge_incidence(surface: Mapping[str, np.ndarray]) -> Counter[tuple[int, int]]:
    incidence: Counter[tuple[int, int]] = Counter()
    for triangle in np.asarray(surface["triangleFrontierIndex"], dtype=np.int32):
        incidence.update(
            tuple(
                sorted(
                    (
                        int(triangle[index]),
                        int(triangle[(index + 1) % 3]),
                    )
                )
            )
            for index in range(3)
        )
    return incidence


def reconstruct_surface_corridor_boundary(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    corridor_index: int,
    *,
    incidence: Mapping[tuple[int, int], int] | None = None,
) -> dict[str, np.ndarray | int | float]:
    """Recover the exact two-arc strip boundary represented by sparse anchors.

    The first and second arc walks follow correspondence order.  The patch
    boundary follows the first arc and the reverse of the second, leaving one
    new mouth edge at each longitudinal end.  No edge, node, or raster pixel
    is admitted independently.
    """

    pair_offset = np.asarray(corridors["corridorPairOffset"], dtype=np.int64)
    if not 0 <= corridor_index < len(pair_offset) - 1:
        raise ValueError("corridor index is outside the catalog")
    start, stop = int(pair_offset[corridor_index]), int(pair_offset[corridor_index + 1])
    first_anchor = np.asarray(
        corridors["corridorFirstBoundaryEdge"], dtype=np.int32
    )[start:stop]
    second_anchor = np.asarray(
        corridors["corridorSecondBoundaryEdge"], dtype=np.int32
    )[start:stop]
    if len(first_anchor) != len(second_anchor) or len(first_anchor) < 2:
        raise ValueError("corridor does not have paired multi-edge anchors")

    edge_loop = np.asarray(corridors["boundaryEdgeLoopIndex"], dtype=np.int32)
    edge_position = np.asarray(
        corridors["boundaryEdgeLoopPosition"], dtype=np.int32
    )
    first_loop_values = np.unique(edge_loop[first_anchor])
    second_loop_values = np.unique(edge_loop[second_anchor])
    if len(first_loop_values) != 1 or len(second_loop_values) != 1:
        raise ValueError("corridor anchors cross loop identities")
    first_loop, second_loop = int(first_loop_values[0]), int(second_loop_values[0])
    if first_loop == second_loop:
        raise ValueError("corridor completion requires two distinct outer loops")
    if (
        first_loop != int(corridors["corridorFirstLoopIndex"][corridor_index])
        or second_loop != int(corridors["corridorSecondLoopIndex"][corridor_index])
    ):
        raise ValueError("corridor loop provenance differs from its edge anchors")

    loop_offset = np.asarray(loops["loopOffset"], dtype=np.int64)
    loop_vertex = np.asarray(loops["loopVertexFrontierIndex"], dtype=np.int32)

    def loop_nodes(loop_index: int) -> np.ndarray:
        return loop_vertex[
            int(loop_offset[loop_index]) : int(loop_offset[loop_index + 1])
        ]

    first_arc, first_edge_count = _arc_node_walk(
        loop_nodes(first_loop), edge_position[first_anchor], direction=1
    )
    pairing_direction = int(
        corridors["corridorPairingDirection"][corridor_index]
    )
    second_arc, second_edge_count = _arc_node_walk(
        loop_nodes(second_loop),
        edge_position[second_anchor],
        direction=pairing_direction,
    )
    boundary = np.concatenate((first_arc, second_arc[::-1])).astype(np.int32)
    if len(boundary) < 6 or np.any(boundary == np.roll(boundary, -1)):
        raise ValueError("corridor boundary contains a degenerate step")
    mouths = np.asarray(
        (
            (int(first_arc[-1]), int(second_arc[-1])),
            (int(second_arc[0]), int(first_arc[0])),
        ),
        dtype=np.int32,
    )
    if np.any(mouths[:, 0] == mouths[:, 1]):
        raise ValueError("corridor mouth collapses to a shared endpoint")

    def walk_edges(walk: np.ndarray) -> list[tuple[int, int]]:
        return [
            tuple(sorted((int(first), int(second))))
            for first, second in zip(walk[:-1], walk[1:])
        ]

    attachment = walk_edges(first_arc) + walk_edges(second_arc)
    mouth_edges = [tuple(sorted((int(first), int(second)))) for first, second in mouths]
    complete_boundary = [
        tuple(
            sorted(
                (
                    int(boundary[index]),
                    int(boundary[(index + 1) % len(boundary)]),
                )
            )
        )
        for index in range(len(boundary))
    ]
    if Counter(complete_boundary) != Counter((*attachment, *mouth_edges)):
        raise ValueError("corridor boundary does not preserve both exact arcs")
    if any(count != 1 for count in Counter(complete_boundary).values()):
        raise ValueError("corridor boundary traverses a mesh edge more than once")

    edge_incidence = incidence or _surface_edge_incidence(surface)
    if any(edge_incidence.get(edge, 0) != 1 for edge in attachment):
        raise ValueError("corridor attachment is not an exact open frontier")
    if any(edge_incidence.get(edge, 0) != 0 for edge in mouth_edges):
        raise ValueError("corridor mouth already exists on the surface")

    chart_uv = np.asarray(surface["chartUV"], dtype=np.float64)[boundary]
    signed_area = 0.5 * float(
        np.sum(
            chart_uv[:, 0] * np.roll(chart_uv[:, 1], -1)
            - chart_uv[:, 1] * np.roll(chart_uv[:, 0], -1)
        )
    )
    if not math.isfinite(signed_area) or abs(signed_area) <= 1.0e-6:
        raise ValueError("corridor boundary has no intrinsic area")
    return {
        "boundaryVertexFrontierIndex": boundary,
        "mouthFrontierIndex": mouths,
        "firstArcEdgeCount": first_edge_count,
        "secondArcEdgeCount": second_edge_count,
        "attachmentEdgeCount": len(attachment),
        "signedAreaChartVoxelsSquared": signed_area,
    }


def build_surface_corridor_completion_domains(
    surface: Mapping[str, np.ndarray],
    loops: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    evidence: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Compile evidence-eligible corridors into exact two-frontier domains."""

    scored_corridor = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    evidence_eligible = np.asarray(
        evidence["corridorEvidenceEligible"], dtype=np.uint8
    ) > 0
    incidence = _surface_edge_incidence(surface)
    domain_scored_row: list[int] = []
    domain_corridor: list[int] = []
    boundary_offset = [0]
    boundary_vertex: list[int] = []
    mouth_values: list[np.ndarray] = []
    first_edge_count: list[int] = []
    second_edge_count: list[int] = []
    attachment_edge_count: list[int] = []
    signed_area: list[float] = []
    exact_eligible = np.zeros(len(scored_corridor), dtype=np.uint8)
    failure_count: Counter[str] = Counter()
    for scored_row in np.flatnonzero(evidence_eligible):
        corridor_index = int(scored_corridor[scored_row])
        try:
            state = reconstruct_surface_corridor_boundary(
                surface,
                loops,
                corridors,
                corridor_index,
                incidence=incidence,
            )
        except ValueError as error:
            failure_count[str(error)] += 1
            continue
        exact_eligible[scored_row] = 1
        boundary = np.asarray(
            state["boundaryVertexFrontierIndex"], dtype=np.int32
        )
        domain_scored_row.append(int(scored_row))
        domain_corridor.append(corridor_index)
        boundary_vertex.extend(int(value) for value in boundary)
        boundary_offset.append(len(boundary_vertex))
        mouth_values.append(
            np.asarray(state["mouthFrontierIndex"], dtype=np.int32)
        )
        first_edge_count.append(int(state["firstArcEdgeCount"]))
        second_edge_count.append(int(state["secondArcEdgeCount"]))
        attachment_edge_count.append(int(state["attachmentEdgeCount"]))
        signed_area.append(float(state["signedAreaChartVoxelsSquared"]))
    arrays = {
        "corridorCompletionDomainEligible": exact_eligible,
        "corridorCompletionScoredRow": np.asarray(
            domain_scored_row, dtype=np.int32
        ),
        "corridorCompletionCorridorIndex": np.asarray(
            domain_corridor, dtype=np.int32
        ),
        "corridorCompletionBoundaryOffset": np.asarray(
            boundary_offset, dtype=np.int64
        ),
        "corridorCompletionBoundaryVertexFrontierIndex": np.asarray(
            boundary_vertex, dtype=np.int32
        ),
        "corridorCompletionMouthFrontierIndex": (
            np.asarray(mouth_values, dtype=np.int32).reshape((-1, 2, 2))
            if mouth_values
            else np.empty((0, 2, 2), dtype=np.int32)
        ),
        "corridorCompletionFirstArcEdgeCount": np.asarray(
            first_edge_count, dtype=np.int32
        ),
        "corridorCompletionSecondArcEdgeCount": np.asarray(
            second_edge_count, dtype=np.int32
        ),
        "corridorCompletionAttachmentEdgeCount": np.asarray(
            attachment_edge_count, dtype=np.int32
        ),
        "corridorCompletionSignedAreaChartVoxelsSquared": np.asarray(
            signed_area, dtype=np.float32
        ),
    }
    return arrays, {
        "evidenceEligibleCorridorCount": int(np.count_nonzero(evidence_eligible)),
        "exactCompletionDomainCount": len(domain_scored_row),
        "invalidCompletionDomainCount": int(
            np.count_nonzero(evidence_eligible) - len(domain_scored_row)
        ),
        "invalidCompletionDomainReasonCount": dict(sorted(failure_count.items())),
        "decisionUnit": "two complete inherited arcs and both end mouths",
        "selectionMutated": False,
    }


def _gather_ragged_rows(
    values: np.ndarray,
    offset: np.ndarray,
    rows: Sequence[int],
) -> tuple[np.ndarray, np.ndarray]:
    pieces = [
        np.asarray(values)[int(offset[row]) : int(offset[row + 1])]
        for row in rows
    ]
    lengths = [len(value) for value in pieces]
    result_offset = np.concatenate(
        ([0], np.cumsum(lengths, dtype=np.int64))
    ).astype(np.int64)
    if pieces:
        return np.concatenate(pieces), result_offset
    return np.asarray(values)[:0].copy(), result_offset


def surface_corridor_completion_view(
    arrays: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Expose exact corridor domains through the generic dense-field contract."""

    rows = np.asarray(arrays["corridorCompletionScoredRow"], dtype=np.int32)
    corridor = np.asarray(
        arrays["corridorCompletionCorridorIndex"], dtype=np.int32
    )
    old_patch_offset = np.asarray(arrays["corridorPatchOffset"], dtype=np.int64)
    patch_xyz, patch_offset = _gather_ragged_rows(
        np.asarray(arrays["corridorPatchXYZ"], dtype=np.float32),
        old_patch_offset,
        rows,
    )
    patch_normal, normal_offset = _gather_ragged_rows(
        np.asarray(arrays["corridorPatchNormalXYZ"], dtype=np.float32),
        old_patch_offset,
        rows,
    )
    patch_uv, uv_offset = _gather_ragged_rows(
        np.asarray(arrays["corridorPatchUV"], dtype=np.float32),
        old_patch_offset,
        rows,
    )
    patch_thickness, thickness_offset = _gather_ragged_rows(
        np.asarray(arrays["corridorPatchThicknessVoxels"], dtype=np.float32),
        old_patch_offset,
        rows,
    )
    if not (
        np.array_equal(patch_offset, normal_offset)
        and np.array_equal(patch_offset, uv_offset)
        and np.array_equal(patch_offset, thickness_offset)
    ):
        raise ValueError("corridor completion patch fields differ")

    boundary_offset = np.asarray(
        arrays["corridorCompletionBoundaryOffset"], dtype=np.int64
    )
    boundary_vertex = np.asarray(
        arrays["corridorCompletionBoundaryVertexFrontierIndex"],
        dtype=np.int32,
    )
    midpoint = np.asarray(arrays["midpointXYZ"], dtype=np.float32)
    chart_uv = np.asarray(arrays["chartUV"], dtype=np.float32)
    mean_edge: list[float] = []
    area: list[float] = []
    median_thickness: list[float] = []
    boundary_parameter_uv: list[np.ndarray] = []
    first_arc_count = np.asarray(
        arrays["corridorCompletionFirstArcEdgeCount"], dtype=np.int32
    )
    second_arc_count = np.asarray(
        arrays["corridorCompletionSecondArcEdgeCount"], dtype=np.int32
    )
    patch_rows = np.asarray(arrays["corridorPatchRows"], dtype=np.int32)
    patch_columns = np.asarray(arrays["corridorPatchColumns"], dtype=np.int32)
    raster_step = np.asarray(
        arrays["corridorRasterStepVoxels"], dtype=np.float32
    )
    for domain_row in range(len(rows)):
        start, stop = int(boundary_offset[domain_row]), int(boundary_offset[domain_row + 1])
        boundary = boundary_vertex[start:stop]
        edge_length = np.linalg.norm(
            midpoint[np.roll(boundary, -1)] - midpoint[boundary], axis=1
        )
        mean_edge.append(float(np.mean(edge_length)))
        uv = chart_uv[boundary]
        area.append(
            abs(
                0.5
                * float(
                    np.sum(
                        uv[:, 0] * np.roll(uv[:, 1], -1)
                        - uv[:, 1] * np.roll(uv[:, 0], -1)
                    )
                )
            )
        )
        patch_start, patch_stop = int(patch_offset[domain_row]), int(patch_offset[domain_row + 1])
        median_thickness.append(
            float(np.median(patch_thickness[patch_start:patch_stop]))
        )
        first_vertex_count = int(first_arc_count[domain_row]) + 1
        second_vertex_count = int(second_arc_count[domain_row]) + 1
        if first_vertex_count + second_vertex_count != len(boundary):
            raise ValueError("corridor arc counts differ from its exact boundary")
        source_row = int(rows[domain_row])
        domain_length = (
            int(patch_rows[source_row]) - 1
        ) * float(raster_step[source_row])
        domain_width = (
            int(patch_columns[source_row]) - 1
        ) * float(raster_step[source_row])

        def normalized_distance(points: np.ndarray) -> np.ndarray:
            distance = np.concatenate(
                (
                    np.zeros(1, dtype=np.float64),
                    np.cumsum(np.linalg.norm(np.diff(points, axis=0), axis=1)),
                )
            )
            return distance / max(float(distance[-1]), 1.0e-12)

        first_fraction = normalized_distance(
            midpoint[boundary[:first_vertex_count]]
        )
        second_fraction = normalized_distance(
            midpoint[boundary[first_vertex_count:]]
        )
        boundary_parameter_uv.append(
            np.vstack(
                (
                    np.column_stack(
                        (
                            np.zeros(first_vertex_count),
                            domain_length * first_fraction,
                        )
                    ),
                    np.column_stack(
                        (
                            np.full(second_vertex_count, domain_width),
                            domain_length * (1.0 - second_fraction),
                        )
                    ),
                )
            ).astype(np.float32)
        )

    selected_model = np.asarray(arrays["corridorSelectedModel"], dtype=np.int32)[rows]
    rank = np.asarray(arrays["corridorModelRankScore"], dtype=np.float32)
    selected_rank = rank[rows, selected_model]
    raster_coordinates: list[np.ndarray] = []
    for scored_row in rows:
        row_count = int(patch_rows[scored_row])
        column_count = int(patch_columns[scored_row])
        row_coordinate, column_coordinate = np.indices(
            (row_count, column_count), dtype=np.int32
        )
        raster_coordinates.append(
            np.column_stack(
                (column_coordinate.reshape(-1), row_coordinate.reshape(-1))
            )
        )
    result = {key: np.asarray(value) for key, value in arrays.items()}
    result.update(
        {
            "scoredLoopIndex": np.arange(len(rows), dtype=np.int32),
            "loopOffset": boundary_offset,
            "loopVertexFrontierIndex": boundary_vertex,
            "loopTopologyComponent": np.asarray(
                arrays["corridorTopologyComponent"], dtype=np.int32
            )[corridor],
            "loopMedianThicknessVoxels": np.asarray(
                median_thickness, dtype=np.float32
            ),
            "loopMeanBoundaryEdgeVoxels": np.asarray(
                mean_edge, dtype=np.float32
            ),
            "loopAreaChartVoxelsSquared": np.asarray(area, dtype=np.float32),
            "loopMacroEligible": np.zeros(len(rows), dtype=np.uint8),
            "loopKind": np.zeros(len(rows), dtype=np.uint8),
            "patchOffset": patch_offset,
            "patchXYZ": patch_xyz.astype(np.float32, copy=False),
            "patchNormalXYZ": patch_normal.astype(np.float32, copy=False),
            "patchUV": patch_uv.astype(np.float32, copy=False),
            "patchRasterCoordinateUV": (
                np.concatenate(raster_coordinates).astype(np.int32)
                if raster_coordinates
                else np.empty((0, 2), dtype=np.int32)
            ),
            "rasterStepVoxels": np.asarray(
                arrays["corridorRasterStepVoxels"], dtype=np.float32
            )[rows],
            "contextMedianProfile": np.asarray(
                arrays["corridorContextMedianProfile"], dtype=np.float32
            )[rows],
            "contextPhysicalScore": np.asarray(
                arrays["corridorContextPhysicalScore"], dtype=np.float32
            )[rows],
            "localIntensityScale": np.asarray(
                arrays["corridorLocalIntensityScale"], dtype=np.float32
            )[rows],
            "patchCandidateOffset": np.zeros(len(rows) + 1, dtype=np.int64),
            "patchCandidateFrontierIndex": np.empty(0, dtype=np.int32),
            "patchCandidateNearestPixel": np.empty(0, dtype=np.int32),
            "patchCandidateSurfaceAlignment": np.empty(0, dtype=np.float32),
            "candidateBankUsed": np.zeros(1, dtype=np.uint8),
            "denseCorridorMode": np.ones(1, dtype=np.uint8),
            "denseCorridorSourceScoredRow": rows,
            "denseCorridorSourceCorridorIndex": corridor,
            "denseCorridorMouthFrontierIndex": np.asarray(
                arrays["corridorCompletionMouthFrontierIndex"], dtype=np.int32
            ),
            "denseCorridorAttachmentEdgeCount": np.asarray(
                arrays["corridorCompletionAttachmentEdgeCount"], dtype=np.int32
            ),
            "denseCorridorFirstArcEdgeCount": first_arc_count,
            "denseCorridorSecondArcEdgeCount": second_arc_count,
            "denseCorridorBoundaryParameterUV": (
                np.concatenate(boundary_parameter_uv).astype(np.float32)
                if boundary_parameter_uv
                else np.empty((0, 2), dtype=np.float32)
            ),
            "denseCorridorRankScore": selected_rank.astype(np.float32),
        }
    )
    return result


def classify_surface_corridor_evidence(
    loops: Mapping[str, np.ndarray],
    corridors: Mapping[str, np.ndarray],
    scored: Mapping[str, np.ndarray],
    *,
    settings: PhysicalRibbonSurfaceCorridorSettings,
    owned_bounds: tuple[np.ndarray, np.ndarray] | None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Classify whole strips without consulting ribbon-bank candidates."""

    corridor_rows = np.asarray(scored["scoredCorridorIndex"], dtype=np.int32)
    selected_model = np.asarray(
        scored["corridorSelectedModel"], dtype=np.int32
    )
    scored_rows = np.arange(len(corridor_rows), dtype=np.int32)
    profile = np.asarray(
        scored["corridorZeroShiftContextProfileCorrelation"], dtype=np.float32
    )[scored_rows, selected_model]
    trace = np.asarray(
        scored["corridorBoundaryTraceCorrelation"], dtype=np.float32
    )[scored_rows, selected_model]
    margin = np.asarray(
        scored["corridorZeroShiftCompetingMargin"], dtype=np.float32
    )[scored_rows, selected_model]
    physical = np.asarray(
        scored["corridorModelShiftPhysicalScore"], dtype=np.float32
    )
    shifts = np.asarray(
        scored["corridorCompetingShiftThicknesses"], dtype=np.float32
    )
    zero_shift = int(np.flatnonzero(shifts == 0.0)[0])
    selected_physical = physical[scored_rows, selected_model, zero_shift]
    radius = np.asarray(
        scored["corridorModelMinimumCurvatureRadiusThicknesses"],
        dtype=np.float32,
    )[scored_rows, selected_model]
    anisotropy = np.asarray(
        scored["corridorTextureAnisotropy"], dtype=np.float32
    )[scored_rows, selected_model]
    ct_eligible = (
        (profile >= settings.corridors.minimum_context_profile_correlation)
        & (trace >= settings.corridors.minimum_boundary_trace_correlation)
        & (margin >= settings.corridors.minimum_zero_shift_margin)
    )

    first_loop = np.asarray(
        corridors["corridorFirstLoopIndex"], dtype=np.int32
    )[corridor_rows]
    second_loop = np.asarray(
        corridors["corridorSecondLoopIndex"], dtype=np.int32
    )[corridor_rows]
    region = np.asarray(loops["loopTriangleRegion"], dtype=np.int32)
    distinct_region = region[first_loop] != region[second_loop]
    loop_sets = _loop_vertex_sets(
        loops, np.concatenate((first_loop, second_loop))
    )
    shared_vertex_count = np.asarray(
        [
            len(loop_sets[int(first)] & loop_sets[int(second)])
            for first, second in zip(first_loop, second_loop)
        ],
        dtype=np.int32,
    )

    patch_offset = np.asarray(scored["corridorPatchOffset"], dtype=np.int64)
    patch_xyz = np.asarray(scored["corridorPatchXYZ"], dtype=np.float32)
    boundary_distance = np.asarray(
        [
            _owned_boundary_distance(
                patch_xyz[int(patch_offset[row]) : int(patch_offset[row + 1])],
                owned_bounds,
            )
            for row in range(len(corridor_rows))
        ],
        dtype=np.float32,
    )
    touches_owned_boundary = (
        boundary_distance < settings.owned_boundary_exclusion_voxels
    )
    eligible = ct_eligible & ~touches_owned_boundary
    if settings.require_distinct_triangle_regions:
        eligible &= distinct_region

    arrays = {
        "corridorCtEvidenceEligible": ct_eligible.astype(np.uint8),
        "corridorConnectsDistinctTriangleRegions": distinct_region.astype(
            np.uint8
        ),
        "corridorSharedBoundaryVertexCount": shared_vertex_count,
        "corridorOwnedBoundaryDistanceVoxels": boundary_distance,
        "corridorTouchesOwnedBoundary": touches_owned_boundary.astype(np.uint8),
        "corridorEvidenceEligible": eligible.astype(np.uint8),
    }
    return arrays, {
        "scoredCorridorCount": int(len(corridor_rows)),
        "ctEvidenceEligibleCorridorCount": int(np.count_nonzero(ct_eligible)),
        "distinctTriangleRegionCorridorCount": int(
            np.count_nonzero(distinct_region)
        ),
        "sharedBoundaryVertexCorridorCount": int(
            np.count_nonzero(shared_vertex_count)
        ),
        "ownedBoundaryCorridorCount": int(
            np.count_nonzero(touches_owned_boundary)
        ),
        "evidenceEligibleCorridorCount": int(np.count_nonzero(eligible)),
        "eligibleSharedBoundaryVertexCorridorCount": int(
            np.count_nonzero(eligible & (shared_vertex_count > 0))
        ),
        "eligibleProfileCorrelation": _finite_distribution(profile[eligible]),
        "eligibleBoundaryTraceCorrelation": _finite_distribution(trace[eligible]),
        "eligibleCompetingLayerMargin": _finite_distribution(margin[eligible]),
        "eligiblePhysicalScore": _finite_distribution(selected_physical[eligible]),
        "eligibleCurvatureRadiusThicknesses": _finite_distribution(
            radius[eligible]
        ),
        "eligibleTextureAnisotropy": _finite_distribution(anisotropy[eligible]),
        "decisionUnit": "two aligned multi-edge outer-frontier arcs",
        "candidateBankUsed": False,
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_surface_corridors(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonSurfaceCorridorSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonSurfaceCorridorSettings()
    surface_path, surface_manifest = _resolve_surface_manifest(surface_root)
    surface = _load_npz(
        surface_path.parent / str(surface_manifest["data"]["path"]),
        surface_manifest["data"]["sha256"],
    )
    surface, topology_normalization = split_nonmanifold_surface_vertices(
        surface
    )
    source_record = surface_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
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
        "schema": PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_CORRIDORS_VERSION,
        "surface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": surface_manifest["identity"][
            "topologyContinuity"
        ],
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": {
            "surfaceCorridors": sha256_file(Path(__file__)),
            "corridorGeometryAndScoring": sha256_file(
                Path(extract_surface_patch_corridors.__code__.co_filename)
            ),
            "surfaceTopologyNormalization": sha256_file(
                Path(split_nonmanifold_surface_vertices.__code__.co_filename)
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_SURFACE_CORRIDORS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_SURFACE_CORRIDORS_STEM}.npz"
    preview_path = output / "physical-ribbon-surface-corridors.png"
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
    loops, loop_statistics = extract_surface_boundary_loops(
        surface, settings=resolved.corridors.surface_settings()
    )
    looped = time.monotonic()
    corridors, corridor_statistics = extract_surface_patch_corridors(
        surface, loops, settings=resolved.corridors
    )
    extracted = time.monotonic()
    scored, scoring_statistics = score_surface_patch_corridors(
        surface, corridors, source, settings=resolved.corridors
    )
    scored_at = time.monotonic()
    evidence, evidence_statistics = classify_surface_corridor_evidence(
        loops,
        corridors,
        scored,
        settings=resolved,
        owned_bounds=owned_bounds,
    )
    classified = time.monotonic()
    completion_domains, completion_domain_statistics = (
        build_surface_corridor_completion_domains(
            surface,
            loops,
            corridors,
            scored,
            evidence,
        )
    )
    compiled = time.monotonic()
    arrays = {
        **surface,
        **loops,
        **corridors,
        **scored,
        **evidence,
        **completion_domains,
    }
    _write_npz(data_path, arrays)
    write_patch_corridor_montage(
        corridors,
        scored,
        preview_path,
        maximum_corridors=resolved.corridors.maximum_preview_corridors,
        reconfiguration=evidence,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_SURFACE_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_CORRIDORS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": geometry,
        "loops": loop_statistics,
        "corridors": corridor_statistics,
        "scoring": scoring_statistics,
        "evidence": evidence_statistics,
        "completionDomains": completion_domain_statistics,
        "surfaceTopologyNormalization": topology_normalization,
        "timingSeconds": {
            "boundaryLoops": round(looped - started, 6),
            "corridorGeometry": round(extracted - looped, 6),
            "nativeCtAndFlattenedTexture": round(scored_at - extracted, 6),
            "evidenceClassification": round(classified - scored_at, 6),
            "exactCompletionDomains": round(compiled - classified, 6),
            "writingAndPreview": round(finished - compiled, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"corridorMontage": preview_path.name},
        "method": {
            "decisionUnit": (
                "one complete strip bounded by two aligned multi-edge outer "
                "frontier arcs; no cell, edge, or raster pixel is selected alone"
            ),
            "surfaceModel": (
                "ruled and cubic-Hermite strips constrained by both endpoint "
                "positions, normals, and longitudinal boundary traces"
            ),
            "rawCtEvidence": (
                "whole-strip air-material-air normal profiles, displaced-layer "
                "competition, and flattened boundary texture continuity"
            ),
            "blockExitHandling": (
                "any scored strip approaching an owned block face is classified "
                "but excluded from autonomous eligibility"
            ),
            "selectionMutated": False,
            "surfaceTopology": (
                "vertex-only incident triangle fans are split before any "
                "boundary or chart reasoning; the materialized triangle "
                "1-skeleton replaces the old candidate graph"
            ),
            "candidateBankUsed": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
