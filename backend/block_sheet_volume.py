from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_SHEET_ROOT = (
    PROJECT_ROOT
    / "work/multiseam-2x2-b00c03c"
    / "material-surface-graph-v1"
)
DEFAULT_VOLUME_PATH = Path(
    "/mnt/t5/acus-cross-scroll/pherc0358-z7168-d512-yfull-xfull.npy"
)


def configured_sheet_root() -> Path:
    return Path(os.environ.get("PAREIDOLIA_BLOCK_SHEET_ROOT", DEFAULT_SHEET_ROOT))


def configured_volume_path() -> Path:
    return Path(os.environ.get("PAREIDOLIA_BLOCK_VOLUME", DEFAULT_VOLUME_PATH))


def _required(values: Any, name: str) -> np.ndarray:
    if name not in values:
        raise ValueError(f"block sheet artifact is missing {name}")
    return np.asarray(values[name])


def _round_list(values: np.ndarray, digits: int = 4) -> list[float]:
    return [round(float(value), digits) for value in values]


def _load_legacy_block_sheet_payload(root_value: str) -> dict[str, Any]:
    root = Path(root_value)
    patch_manifest_path = root / "selected-patches-v1.json"
    patch_data_path = root / "selected-patches-v1.npz"
    graph_data_path = root / "surface-graph-v1.npz"
    typed_graph_data_path = root / "sheetlet-graph-v1.npz"
    summary_path = root / "summary.json"
    if not patch_manifest_path.is_file() or not patch_data_path.is_file():
        raise FileNotFoundError(f"block sheet geometry is unavailable at {root}")
    if not graph_data_path.is_file():
        raise FileNotFoundError(f"block sheet graph is unavailable at {graph_data_path}")

    manifest = json.loads(patch_manifest_path.read_text())
    with np.load(patch_data_path, allow_pickle=False) as stored:
        patch_id = _required(stored, "patchId").astype(np.uint64, copy=False)
        cell_xyz = _required(stored, "cellXYZ").astype(np.int32, copy=False)
        confidence = _required(stored, "confidence").astype(np.float64, copy=False)
        normal_xyz = _required(stored, "normalXYZ").astype(np.float64, copy=False)
        vertex_offset = _required(stored, "vertexOffset").astype(np.int64, copy=False)
        vertex_axis = _required(stored, "vertexEdgeAxis").astype(np.int64, copy=False)
        vertex_anchor = _required(stored, "vertexEdgeAnchor").astype(np.float64, copy=False)
        vertex_t = _required(stored, "vertexT").astype(np.float64, copy=False)

    with np.load(graph_data_path, allow_pickle=False) as stored:
        graph_patch_id = _required(stored, "patchId").astype(np.uint64, copy=False)
        graph_component_id = _required(stored, "componentId").astype(np.uint64, copy=False)
        retained_first_patch_id = _required(stored, "firstPatchId").astype(
            np.uint64, copy=False
        )
        retained_second_patch_id = _required(stored, "secondPatchId").astype(
            np.uint64, copy=False
        )
        retained_join_count = len(retained_first_patch_id)

    if len(patch_id) != len(cell_xyz) or len(vertex_offset) != len(patch_id) + 1:
        raise ValueError("block sheet patch arrays have inconsistent lengths")
    if len(graph_patch_id) != len(graph_component_id):
        raise ValueError("block sheet component arrays have inconsistent lengths")

    component_by_patch = {
        int(current_patch): int(current_component)
        for current_patch, current_component in zip(graph_patch_id, graph_component_id)
    }
    angle_by_pair: dict[tuple[int, int], tuple[float, float, bool]] = {}
    if typed_graph_data_path.is_file():
        with np.load(typed_graph_data_path, allow_pickle=False) as stored:
            first = _required(stored, "continuationFirstPatchId")
            second = _required(stored, "continuationSecondPatchId")
            normal = _required(stored, "continuationNormalAngleDegrees")
            fiber = _required(stored, "continuationFiberAngleDegrees")
            family = _required(stored, "continuationFamily")
            angle_by_pair = {
                (min(int(a), int(b)), max(int(a), int(b))): (
                    float(normal_angle),
                    float(fiber_angle),
                    bool(int(family_value)),
                )
                for a, b, normal_angle, fiber_angle, family_value in zip(
                    first, second, normal, fiber, family
                )
            }
    try:
        patch_component = np.asarray(
            [component_by_patch[int(value)] for value in patch_id], dtype=np.uint64
        )
    except KeyError as exc:
        raise ValueError(f"surface graph omits selected patch {exc.args[0]}") from exc

    component_values, component_counts = np.unique(patch_component, return_counts=True)
    ranked_components = sorted(
        (
            (int(component), int(count))
            for component, count in zip(component_values, component_counts)
        ),
        key=lambda value: (-value[1], value[0]),
    )
    rank_by_component = {
        component: rank for rank, (component, _count) in enumerate(ranked_components, 1)
    }

    grid = manifest.get("grid", {})
    shape_cells = np.asarray(grid.get("shapeCellsXYZ", ()), dtype=np.int64)
    cell_size = np.asarray(grid.get("cellSizeXYZ", ()), dtype=np.float64)
    origin = np.asarray(grid.get("originXYZ", ()), dtype=np.float64)
    if shape_cells.shape != (3,) or cell_size.shape != (3,) or origin.shape != (3,):
        raise ValueError("block sheet grid must define three-dimensional shape, size, and origin")
    extent = shape_cells.astype(np.float64) * cell_size

    component_accumulators: dict[int, dict[str, Any]] = {
        component: {
            "rank": rank_by_component[component],
            "stableId": str(component),
            "patchCount": count,
            "confidenceTotal": 0.0,
            "boundsMinimum": np.full(3, np.inf, dtype=np.float64),
            "boundsMaximum": np.full(3, -np.inf, dtype=np.float64),
            "normalJoinAngles": [],
            "strictFiberJoinAngles": [],
            "quarterTurnResiduals": [],
        }
        for component, count in ranked_components
    }
    patches: list[dict[str, Any]] = []
    for index, current_patch_id in enumerate(patch_id):
        low = int(vertex_offset[index])
        high = int(vertex_offset[index + 1])
        vertices_grid = vertex_anchor[low:high].copy()
        for vertex_index, axis in enumerate(vertex_axis[low:high]):
            vertices_grid[vertex_index, int(axis)] += vertex_t[low + vertex_index]
        vertices_local = vertices_grid * cell_size
        component = int(patch_component[index])
        accumulator = component_accumulators[component]
        accumulator["confidenceTotal"] += float(confidence[index])
        accumulator["boundsMinimum"] = np.minimum(
            accumulator["boundsMinimum"], np.min(vertices_local, axis=0)
        )
        accumulator["boundsMaximum"] = np.maximum(
            accumulator["boundsMaximum"], np.max(vertices_local, axis=0)
        )
        patches.append(
            {
                "id": str(int(current_patch_id)),
                "component": int(accumulator["rank"]),
                "componentSize": int(accumulator["patchCount"]),
                "cell": [int(value) for value in cell_xyz[index]],
                "confidence": round(float(confidence[index]), 6),
                "normal": _round_list(normal_xyz[index], 6),
                "vertices": [_round_list(vertex) for vertex in vertices_local],
            }
        )

    for first_patch_id, second_patch_id in zip(
        retained_first_patch_id, retained_second_patch_id
    ):
        first_value = int(first_patch_id)
        second_value = int(second_patch_id)
        component = component_by_patch[first_value]
        if component_by_patch[second_value] != component:
            raise ValueError("retained sheet join crosses component identities")
        angles = angle_by_pair.get(
            (min(first_value, second_value), max(first_value, second_value))
        )
        if angles is None:
            continue
        normal_angle, fiber_angle, quarter_turn = angles
        accumulator = component_accumulators[component]
        if np.isfinite(normal_angle):
            accumulator["normalJoinAngles"].append(normal_angle)
        if np.isfinite(fiber_angle):
            accumulator[
                "quarterTurnResiduals"
                if quarter_turn
                else "strictFiberJoinAngles"
            ].append(fiber_angle)

    def angle_statistics(values: list[float]) -> dict[str, float | int]:
        samples = np.asarray(values, dtype=np.float64)
        return {
            "count": len(samples),
            "p90Degrees": round(float(np.percentile(samples, 90)), 4)
            if len(samples)
            else 0.0,
            "maximumDegrees": round(float(np.max(samples)), 4)
            if len(samples)
            else 0.0,
        }

    components: list[dict[str, Any]] = []
    for component, _count in ranked_components:
        accumulator = component_accumulators[component]
        patch_count = int(accumulator["patchCount"])
        components.append(
            {
                "rank": int(accumulator["rank"]),
                "stableId": accumulator["stableId"],
                "patchCount": patch_count,
                "meanConfidence": round(
                    float(accumulator["confidenceTotal"]) / max(patch_count, 1), 6
                ),
                "boundsMinimumXYZ": _round_list(accumulator["boundsMinimum"]),
                "boundsMaximumXYZ": _round_list(accumulator["boundsMaximum"]),
                "joinAngles": {
                    "normal": angle_statistics(
                        accumulator["normalJoinAngles"]
                    ),
                    "strictFiber": angle_statistics(
                        accumulator["strictFiberJoinAngles"]
                    ),
                    "quarterTurnResidual": angle_statistics(
                        accumulator["quarterTurnResiduals"]
                    ),
                },
            }
        )

    summary: dict[str, Any] = {}
    if summary_path.is_file():
        summary = json.loads(summary_path.read_text())
    best = summary.get("restitch", {}).get("best", {})
    curvature = summary.get("restitch", {}).get("sheetCurvatureRefinement", {})
    layer_partition = summary.get("restitch", {}).get(
        "sheetletLayerPartition", {}
    )
    signed_partition = summary.get("restitch", {}).get(
        "signedLayerRepulsion", {}
    )
    tangent_sidedness = summary.get("restitch", {}).get(
        "tangentSidedness", {}
    )
    curvature_by_component = {
        str(value["componentId"]): value
        for value in curvature.get("after", {}).get("components", ())
    }
    for component in components:
        record = curvature_by_component.get(component["stableId"])
        if record is None:
            continue
        component["curvature"] = {
            "flaggedJoins": int(record.get("flaggedJoins", 0)),
            "maximumPressure": float(record.get("maximumPressure", 0.0)),
            "directBendP90Degrees": float(
                record.get("directBendDegrees", {}).get("p90") or 0.0
            ),
            "branchContrastP90Degrees": float(
                record.get("branchContrastDegrees", {}).get("p90") or 0.0
            ),
            "normalConeP90DegreesDiagnosticOnly": float(
                record.get("globalNormalConeDegreesDiagnosticOnly", {}).get("p90")
                or 0.0
            ),
        }
    return {
        "schema": "pareidolia.block-sheet-volume",
        "version": 1,
        "variant": root.name,
        "grid": {
            "shapeCellsXYZ": [int(value) for value in shape_cells],
            "cellSizeXYZ": _round_list(cell_size),
            "originXYZ": _round_list(origin),
            "extentXYZ": _round_list(extent),
            "coordinateUnit": str(grid.get("coordinateUnit", "source-voxel")),
        },
        "stats": {
            "patchCount": len(patches),
            "componentCount": len(components),
            "retainedJoinCount": int(retained_join_count),
            "largestComponentPatchCount": int(
                best.get(
                    "largestComponentPatchCount",
                    components[0]["patchCount"] if components else 0,
                )
            ),
            "unresolvedInteriorTraceEndpoints": int(
                best.get("unresolvedInteriorTraceEndpoints", 0)
            ),
            "retainedInteriorTraceFraction": float(
                best.get("retainedInteriorTraceFraction", 0.0)
            ),
            "curvatureFlaggedJoinsBefore": int(
                curvature.get("before", {}).get("flaggedJoins", 0)
            ),
            "curvatureFlaggedJoinsAfter": int(
                curvature.get("after", {}).get("flaggedJoins", 0)
            ),
            "layerConflictsBefore": int(
                layer_partition.get("before", {}).get("conflictCount", 0)
            ),
            "layerConflictsAfter": int(
                layer_partition.get("after", {}).get("conflictCount", 0)
            ),
            "modeledLayerRepulsionPairs": int(
                signed_partition.get("repulsions", 0)
            ),
            "internalLayerRepulsionCost": float(
                best.get("totalLayerRepulsion", 0.0)
            ),
            "foldbackExclusionPairs": int(
                tangent_sidedness.get("foldbackExclusions", 0)
            ),
            "acceptedFoldbackPairs": int(
                tangent_sidedness.get("acceptedGraphFoldbackPairs", 0)
            ),
            "maximumNormalJoinAngleDegrees": max(
                (
                    component["joinAngles"]["normal"]["maximumDegrees"]
                    for component in components
                ),
                default=0.0,
            ),
            "maximumStrictFiberJoinAngleDegrees": max(
                (
                    component["joinAngles"]["strictFiber"]["maximumDegrees"]
                    for component in components
                ),
                default=0.0,
            ),
            "maximumQuarterTurnResidualDegrees": max(
                (
                    component["joinAngles"]["quarterTurnResidual"][
                        "maximumDegrees"
                    ]
                    for component in components
                ),
                default=0.0,
            ),
        },
        "components": components,
        "patches": patches,
    }


def _dense_completion_paths(root: Path) -> tuple[Path, Path]:
    stem = "physical-ribbon-dense-completion-v1"
    return root / f"{stem}.json", root / f"{stem}.npz"


def _material_surface_graph_paths(root: Path) -> tuple[Path, Path]:
    stem = "material-interface-surface-graph-v1"
    return root / f"{stem}.json", root / f"{stem}.npz"


def _percentile(values: np.ndarray, percentile: float) -> float:
    finite = np.asarray(values, dtype=np.float64)
    finite = finite[np.isfinite(finite)]
    return float(np.percentile(finite, percentile)) if len(finite) else 0.0


def _load_material_surface_graph_payload(root: Path) -> dict[str, Any]:
    manifest_path, data_path = _material_surface_graph_paths(root)
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"material surface graph is unavailable at {root}")
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pareidolia.material-interface-surface-graph":
        raise ValueError(f"unsupported material surface graph in {manifest_path}")
    if manifest.get("state") != "complete":
        raise ValueError(f"material surface graph is incomplete: {manifest_path}")
    with np.load(data_path, allow_pickle=False) as stored:
        position_world = _required(stored, "positionXYZ").astype(
            np.float64, copy=False
        )
        component = _required(stored, "componentId").astype(
            np.int64, copy=False
        )
        evidence = _required(stored, "localEvidenceScore").astype(
            np.float64, copy=False
        )
        macro_confidence = _required(
            stored, "macroOrientationConfidence"
        ).astype(np.float64, copy=False)
        raw_macro_error = _required(stored, "rawToMacroNormalDegrees").astype(
            np.float64, copy=False
        )
        pre_component = _required(stored, "preCollisionComponentId").astype(
            np.int64, copy=False
        )
        edge_first = _required(stored, "edgeFirstNode").astype(
            np.int64, copy=False
        )
    if position_world.ndim != 2 or position_world.shape[1] != 3:
        raise ValueError("material surface positions must have shape (node, 3)")
    if any(
        len(values) != len(position_world)
        for values in (
            component,
            evidence,
            macro_confidence,
            raw_macro_error,
            pre_component,
        )
    ):
        raise ValueError("material surface node arrays have inconsistent lengths")
    geometry = manifest.get("geometry", {})
    owned_bounds = geometry.get("ownedWorldBounds", {})
    origin = np.asarray(owned_bounds.get("startXYZ", ()), dtype=np.float64)
    stop = np.asarray(
        owned_bounds.get("stopXYZExclusive", ()), dtype=np.float64
    )
    if origin.shape != (3,) or stop.shape != (3,) or np.any(stop <= origin):
        raise ValueError("material surface graph has invalid owned world bounds")
    extent = stop - origin
    position_local = position_world - origin[None, :]
    if len(position_local) and (
        not np.all(np.isfinite(position_local))
        or np.any(position_local < -1.0e-3)
        or np.any(position_local > extent + 1.0e-3)
    ):
        raise ValueError("material surface nodes lie outside the owned volume")

    component_count = int(np.max(component)) + 1 if len(component) else 0
    component_size = np.bincount(component, minlength=component_count)
    split_pre_component = {
        int(value)
        for value in np.unique(pre_component)
        if len(np.unique(component[pre_component == value])) > 1
    }
    maximum_display_components = 256
    displayed_component_count = min(component_count, maximum_display_components)
    displayed_node = component < displayed_component_count
    displayed_index = np.flatnonzero(displayed_node)
    components: list[dict[str, Any]] = []
    for component_id in range(displayed_component_count):
        member = component == component_id
        point = position_local[member]
        prior = np.unique(pre_component[member])
        components.append(
            {
                "rank": component_id + 1,
                "stableId": str(component_id),
                "triangleCount": 0,
                "nodeCount": int(np.count_nonzero(member)),
                "surfaceAreaVoxelsSquared": 0.0,
                "boundsMinimumXYZ": _round_list(np.min(point, axis=0)),
                "boundsMaximumXYZ": _round_list(np.max(point, axis=0)),
                "normalResidualDegrees": {
                    "median": round(_percentile(raw_macro_error[member], 50), 4),
                    "p90": round(_percentile(raw_macro_error[member], 90), 4),
                    "maximum": round(
                        _percentile(raw_macro_error[member], 100), 4
                    ),
                },
                "experimentalTriangleCount": 0,
                "experimentalCompletionRows": [],
                "meanEvidence": round(float(np.mean(evidence[member])), 6),
                "meanMacroConfidence": round(
                    float(np.mean(macro_confidence[member])), 6
                ),
                "splitByStratumGuard": bool(
                    len(prior) == 1 and int(prior[0]) in split_pre_component
                ),
            }
        )
    interface_nodes = [
        [
            round(float(position_local[index, 0]), 3),
            round(float(position_local[index, 1]), 3),
            round(float(position_local[index, 2]), 3),
            int(component[index]) + 1,
        ]
        for index in displayed_index
    ]
    counts = manifest.get("counts", {})
    source = manifest.get("source", {})
    try:
        manifest_label = str(manifest_path.relative_to(PROJECT_ROOT))
    except ValueError:
        manifest_label = str(manifest_path)
    return {
        "schema": "pareidolia.block-interface-volume",
        "version": 1,
        "representation": "material-interface-graph",
        "variant": root.name,
        "artifact": {
            "manifestPath": manifest_label,
            "state": str(manifest.get("state", "unknown")),
            "method": "macro-tangent signed material-interface graph",
        },
        "source": {
            "path": str(source.get("path", "")),
            "metadataPath": str(source.get("metadataPath", "")),
            "name": Path(str(source.get("path", "source volume"))).name,
            "voxelSizeMicrons": float(source.get("voxelSizeMicrons", 0.0)),
        },
        "grid": {
            "shapeCellsXYZ": [int(round(value)) for value in extent],
            "cellSizeXYZ": [1.0, 1.0, 1.0],
            "originXYZ": _round_list(origin),
            "extentXYZ": _round_list(extent),
            "coordinateUnit": str(geometry.get("coordinateUnit", "source-voxel")),
        },
        "stats": {
            "triangleCount": 0,
            "nodeCount": int(len(position_world)),
            "displayedNodeCount": int(len(displayed_index)),
            "componentCount": component_count,
            "displayedComponentCount": displayed_component_count,
            "largestComponentTriangleCount": 0,
            "largestComponentNodeCount": int(component_size[0]) if len(component_size) else 0,
            "baselineTriangleCount": 0,
            "experimentalTriangleCount": 0,
            "acceptedCompletionCount": 0,
            "attemptedCompletionCount": 0,
            "regionCountBefore": int(counts.get("preCollisionComponentCount", 0)),
            "regionCountAfter": int(counts.get("componentCount", component_count)),
            "medianNormalResidualDegrees": round(
                _percentile(raw_macro_error, 50), 4
            ),
            "p90NormalResidualDegrees": round(
                _percentile(raw_macro_error, 90), 4
            ),
            "retainedEdgeCount": int(len(edge_first)),
            "columnConflictRejectedEdgeCount": int(
                counts.get("columnConflictRejectedEdgeCount", 0)
            ),
            "eligibleNodeFraction": float(counts.get("eligibleNodeFraction", 0.0)),
        },
        "components": components,
        "vertices": [],
        "triangles": [],
        "interfaceNodes": interface_nodes,
    }


def _load_dense_surface_payload(root: Path) -> dict[str, Any]:
    manifest_path, data_path = _dense_completion_paths(root)
    if not manifest_path.is_file() or not data_path.is_file():
        raise FileNotFoundError(f"cubical surface geometry is unavailable at {root}")

    manifest = json.loads(manifest_path.read_text())
    if manifest.get("schema") != "pareidolia.physical-ribbon-dense-completion":
        raise ValueError(f"unsupported cubical surface schema in {manifest_path}")
    if manifest.get("state") != "complete":
        raise ValueError(f"cubical surface artifact is not complete: {manifest_path}")

    with np.load(data_path, allow_pickle=False) as stored:
        midpoint_world = _required(stored, "midpointXYZ").astype(
            np.float64, copy=False
        )
        node_component = _required(stored, "component").astype(
            np.int64, copy=False
        )
        triangle_node = _required(stored, "triangleFrontierIndex").astype(
            np.int64, copy=False
        )
        triangle_area = _required(stored, "triangleAreaVoxelsSquared").astype(
            np.float64, copy=False
        )
        triangle_normal_residual = _required(
            stored, "triangleNormalResidualDegrees"
        ).astype(np.float64, copy=False)
        base_triangle_count_values = _required(stored, "baseTriangleCount").astype(
            np.int64, copy=False
        )
        proposal_accepted = _required(stored, "proposalAccepted").astype(
            np.uint8, copy=False
        )
        proposal_hole_row = _required(stored, "proposalHoleRow").astype(
            np.int64, copy=False
        )
        completion_triangle_offset = _required(
            stored, "completionTriangleOffset"
        ).astype(np.int64, copy=False)

    if midpoint_world.ndim != 2 or midpoint_world.shape[1] != 3:
        raise ValueError("cubical surface midpointXYZ must have shape (node, 3)")
    if triangle_node.ndim != 2 or triangle_node.shape[1] != 3:
        raise ValueError("cubical surface triangles must have shape (triangle, 3)")
    if len(node_component) != len(midpoint_world):
        raise ValueError("cubical surface node arrays have inconsistent lengths")
    if len(triangle_area) != len(triangle_node) or len(triangle_normal_residual) != len(
        triangle_node
    ):
        raise ValueError("cubical surface triangle arrays have inconsistent lengths")
    if len(base_triangle_count_values) != 1:
        raise ValueError("cubical surface baseTriangleCount must contain one value")
    if len(completion_triangle_offset) != len(proposal_accepted) + 1:
        raise ValueError("cubical surface completion offsets have inconsistent lengths")
    if len(proposal_hole_row) != len(proposal_accepted):
        raise ValueError("cubical surface proposal arrays have inconsistent lengths")
    if len(triangle_node) and (
        int(np.min(triangle_node)) < 0
        or int(np.max(triangle_node)) >= len(midpoint_world)
    ):
        raise ValueError("cubical surface triangle references an invalid node")

    geometry = manifest.get("geometry", {})
    owned_bounds = geometry.get("ownedWorldBounds", {})
    origin = np.asarray(owned_bounds.get("startXYZ", ()), dtype=np.float64)
    stop = np.asarray(owned_bounds.get("stopXYZExclusive", ()), dtype=np.float64)
    if origin.shape != (3,) or stop.shape != (3,) or np.any(stop <= origin):
        raise ValueError("cubical surface manifest has invalid owned world bounds")
    extent = stop - origin

    triangle_component = node_component[triangle_node]
    if len(triangle_component) and not np.all(
        triangle_component == triangle_component[:, :1]
    ):
        raise ValueError("cubical surface triangle crosses component identities")
    triangle_component = triangle_component[:, 0]

    base_triangle_count = int(base_triangle_count_values[0])
    if not 0 <= base_triangle_count <= len(triangle_node):
        raise ValueError("cubical surface base triangle count is out of range")
    added_triangle_count = len(triangle_node) - base_triangle_count
    added_hole_row = np.full(len(triangle_node), -1, dtype=np.int64)
    for index, (low, high) in enumerate(
        zip(completion_triangle_offset[:-1], completion_triangle_offset[1:])
    ):
        low_value = int(low)
        high_value = int(high)
        if low_value < 0 or high_value < low_value or high_value > added_triangle_count:
            raise ValueError("cubical surface completion triangle offset is out of range")
        if int(proposal_accepted[index]) and high_value > low_value:
            added_hole_row[
                base_triangle_count + low_value : base_triangle_count + high_value
            ] = int(proposal_hole_row[index])
    if added_triangle_count and np.any(added_hole_row[base_triangle_count:] < 0):
        raise ValueError("cubical surface has unattributed completion triangles")

    component_values, component_triangle_counts = np.unique(
        triangle_component, return_counts=True
    )
    ranked_components = sorted(
        (
            (int(component), int(count))
            for component, count in zip(component_values, component_triangle_counts)
        ),
        key=lambda value: (-value[1], value[0]),
    )
    rank_by_component = {
        component: rank for rank, (component, _count) in enumerate(ranked_components, 1)
    }
    triangle_count_by_component = dict(ranked_components)

    used_node = (
        np.unique(triangle_node.reshape(-1))
        if len(triangle_node)
        else np.empty(0, dtype=np.int64)
    )
    compact_by_node = np.full(len(midpoint_world), -1, dtype=np.int64)
    compact_by_node[used_node] = np.arange(len(used_node), dtype=np.int64)
    compact_triangle_node = compact_by_node[triangle_node]
    compact_vertices = midpoint_world[used_node] - origin
    if len(compact_vertices) and (
        not np.all(np.isfinite(compact_vertices))
        or np.any(compact_vertices < -1e-3)
        or np.any(compact_vertices > extent + 1e-3)
    ):
        raise ValueError("cubical surface vertices lie outside the owned volume")

    components: list[dict[str, Any]] = []
    for component, triangle_count in ranked_components:
        triangle_mask = triangle_component == component
        triangle_indices = np.flatnonzero(triangle_mask)
        component_nodes = np.unique(triangle_node[triangle_indices].reshape(-1))
        local_vertices = midpoint_world[component_nodes] - origin
        residual = triangle_normal_residual[triangle_mask]
        component_added_rows = sorted(
            int(value)
            for value in np.unique(added_hole_row[triangle_mask])
            if int(value) >= 0
        )
        components.append(
            {
                "rank": int(rank_by_component[component]),
                "stableId": str(component),
                "triangleCount": int(triangle_count),
                "nodeCount": int(len(component_nodes)),
                "surfaceAreaVoxelsSquared": round(
                    float(np.sum(triangle_area[triangle_mask])), 3
                ),
                "boundsMinimumXYZ": _round_list(np.min(local_vertices, axis=0)),
                "boundsMaximumXYZ": _round_list(np.max(local_vertices, axis=0)),
                "normalResidualDegrees": {
                    "median": round(_percentile(residual, 50), 4),
                    "p90": round(_percentile(residual, 90), 4),
                    "maximum": round(_percentile(residual, 100), 4),
                },
                "experimentalTriangleCount": int(
                    np.count_nonzero(added_hole_row[triangle_mask] >= 0)
                ),
                "experimentalCompletionRows": component_added_rows,
            }
        )

    triangles = [
        {
            "id": int(index),
            "component": int(rank_by_component[int(component)]),
            "componentSize": int(triangle_count_by_component[int(component)]),
            "vertices": [int(value) for value in compact_triangle_node[index]],
            "areaVoxelsSquared": round(float(triangle_area[index]), 5),
            "normalResidualDegrees": round(
                float(triangle_normal_residual[index]), 4
            )
            if np.isfinite(triangle_normal_residual[index])
            else None,
            "experimental": bool(index >= base_triangle_count),
            "completionRow": int(added_hole_row[index])
            if added_hole_row[index] >= 0
            else None,
        }
        for index, component in enumerate(triangle_component)
    ]

    analysis = manifest.get("analysis", {})
    source = manifest.get("source", {})
    try:
        manifest_label = str(manifest_path.relative_to(PROJECT_ROOT))
    except ValueError:
        manifest_label = str(manifest_path)
    return {
        "schema": "pareidolia.block-surface-volume",
        "version": 2,
        "variant": root.name,
        "artifact": {
            "manifestPath": manifest_label,
            "state": str(manifest.get("state", "unknown")),
            "method": "dense cubical surface completion",
        },
        "source": {
            "path": str(source.get("path", "")),
            "metadataPath": str(source.get("metadataPath", "")),
            "name": Path(str(source.get("path", "source volume"))).name,
            "voxelSizeMicrons": float(source.get("voxelSizeMicrons", 0.0)),
        },
        "grid": {
            "shapeCellsXYZ": [int(round(value)) for value in extent],
            "cellSizeXYZ": [1.0, 1.0, 1.0],
            "originXYZ": _round_list(origin),
            "extentXYZ": _round_list(extent),
            "coordinateUnit": str(geometry.get("coordinateUnit", "source-voxel")),
        },
        "stats": {
            "triangleCount": int(len(triangle_node)),
            "nodeCount": int(len(used_node)),
            "componentCount": int(len(components)),
            "largestComponentTriangleCount": int(
                components[0]["triangleCount"] if components else 0
            ),
            "baselineTriangleCount": int(base_triangle_count),
            "experimentalTriangleCount": int(added_triangle_count),
            "acceptedCompletionCount": int(
                analysis.get(
                    "acceptedHoleCount", np.count_nonzero(proposal_accepted)
                )
            ),
            "attemptedCompletionCount": int(
                analysis.get("attemptedHoleCount", len(proposal_accepted))
            ),
            "regionCountBefore": int(analysis.get("triangleRegionCountBefore", 0)),
            "regionCountAfter": int(analysis.get("triangleRegionCountAfter", 0)),
            "medianNormalResidualDegrees": round(
                _percentile(triangle_normal_residual, 50), 4
            ),
            "p90NormalResidualDegrees": round(
                _percentile(triangle_normal_residual, 90), 4
            ),
        },
        "components": components,
        "vertices": [_round_list(vertex) for vertex in compact_vertices],
        "triangles": triangles,
    }


@lru_cache(maxsize=4)
def _load_block_sheet_payload(root_value: str) -> dict[str, Any]:
    root = Path(root_value)
    graph_manifest, graph_data = _material_surface_graph_paths(root)
    if graph_manifest.is_file() or graph_data.is_file():
        return _load_material_surface_graph_payload(root)
    dense_manifest, dense_data = _dense_completion_paths(root)
    if dense_manifest.is_file() or dense_data.is_file():
        return _load_dense_surface_payload(root)
    return _load_legacy_block_sheet_payload(root_value)


def load_block_sheet_payload(root: str | Path | None = None) -> dict[str, Any]:
    selected_root = Path(root) if root is not None else configured_sheet_root()
    return _load_block_sheet_payload(str(selected_root.resolve()))


@lru_cache(maxsize=8)
def _load_block_volume(
    sheet_root_value: str,
    volume_path_value: str,
    stride: int,
) -> tuple[bytes, dict[str, Any]]:
    payload = _load_block_sheet_payload(sheet_root_value)
    grid = payload["grid"]
    block_origin = np.asarray(grid["originXYZ"], dtype=np.int64)
    extent = np.asarray(grid["extentXYZ"], dtype=np.int64)
    if np.any(extent <= 0):
        raise ValueError("block extent must be positive")

    volume_path = Path(volume_path_value)
    if not volume_path.is_file():
        raise FileNotFoundError(f"source block volume is unavailable at {volume_path}")
    sidecar_path = volume_path.with_suffix(".json")
    sidecar = json.loads(sidecar_path.read_text()) if sidecar_path.is_file() else {}
    source_origin = np.asarray(sidecar.get("originXYZ", (0, 0, 0)), dtype=np.int64)
    if source_origin.shape != (3,):
        raise ValueError("source volume origin must contain three coordinates")

    source = np.load(volume_path, mmap_mode="r")
    if source.ndim != 3:
        raise ValueError("source block volume must be a three-dimensional ZYX array")
    low_xyz = block_origin - source_origin
    high_xyz = low_xyz + extent
    shape_xyz = np.asarray((source.shape[2], source.shape[1], source.shape[0]))
    if np.any(low_xyz < 0) or np.any(high_xyz > shape_xyz):
        raise ValueError(
            "sheet block lies outside the configured source volume: "
            f"local bounds {low_xyz.tolist()}–{high_xyz.tolist()}, "
            f"source shape {shape_xyz.tolist()}"
        )

    x0, y0, z0 = (int(value) for value in low_xyz)
    x1, y1, z1 = (int(value) for value in high_xyz)
    sampled = np.asarray(source[z0:z1:stride, y0:y1:stride, x0:x1:stride])
    if sampled.dtype != np.uint8:
        values = sampled.astype(np.float32)
        sample = values[::4, ::4, ::4]
        low, high = np.percentile(sample, (1.0, 99.5))
        if high <= low:
            high = low + 1.0
        sampled_u8 = np.clip((values - low) * (255.0 / (high - low)), 0, 255).astype(
            np.uint8
        )
    else:
        sampled_u8 = np.ascontiguousarray(sampled)

    distribution_sample = sampled_u8[::4, ::4, ::4]
    percentiles = np.percentile(distribution_sample, (1.0, 50.0, 90.0, 99.0))
    shape_sampled_xyz = [
        int(sampled_u8.shape[2]),
        int(sampled_u8.shape[1]),
        int(sampled_u8.shape[0]),
    ]
    metadata = {
        "shapeXYZ": shape_sampled_xyz,
        "stride": int(stride),
        "originXYZ": [int(value) for value in block_origin],
        "extentXYZ": [int(value) for value in extent],
        "percentiles": _round_list(percentiles, 2),
        "source": str(sidecar.get("name", volume_path.name)),
    }
    return sampled_u8.tobytes(order="C"), metadata


def load_block_volume(
    *,
    sheet_root: str | Path | None = None,
    volume_path: str | Path | None = None,
    stride: int = 2,
) -> tuple[bytes, dict[str, Any]]:
    stride = int(stride)
    if stride not in (1, 2, 3, 4):
        raise ValueError("block volume stride must be one of 1, 2, 3, or 4")
    selected_sheet_root = (
        Path(sheet_root) if sheet_root is not None else configured_sheet_root()
    )
    if volume_path is not None:
        selected_volume_path = Path(volume_path)
    elif "PAREIDOLIA_BLOCK_VOLUME" in os.environ:
        selected_volume_path = configured_volume_path()
    else:
        payload = load_block_sheet_payload(selected_sheet_root)
        source_path = payload.get("source", {}).get("path")
        selected_volume_path = Path(source_path) if source_path else configured_volume_path()
    return _load_block_volume(
        str(selected_sheet_root.resolve()),
        str(selected_volume_path.resolve()),
        stride,
    )
