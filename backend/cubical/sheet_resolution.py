from __future__ import annotations

import json
import math
import time
from collections import defaultdict, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .block import SurfaceBlock
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch
from .matching import (
    TraceMatchSettings,
    align_face_patches,
    trace_tangent_side_offsets,
)
from .surface_graph import read_surface_graph
from .topology import GridFace, Int3, cell_face


SHEET_RESOLUTION_AUDIT_SCHEMA = "pareidolia.cubical-sheet-resolution-audit"
SHEET_RESOLUTION_AUDIT_VERSION = 1
SHEET_RESOLUTION_AUDIT_STEM = "sheet-resolution-audit-v1"


@dataclass(frozen=True, slots=True)
class SheetResolutionAuditSettings:
    """Policy-independent signals that one planar sheetlet is too coarse.

    A zero normal limit derives the locally linear regime from the retained
    graph.  The audit deliberately relaxes only normal/fiber agreement: edge
    ownership, endpoint agreement, and order-preserving face alignment remain
    active.  Its matches are evidence for local re-analysis, never joins that
    may be inserted into the surface graph.
    """

    normal_limit_degrees: float = 0.0
    robust_standard_deviations: float = 3.0
    minimum_normal_limit_degrees: float = 15.0
    minimum_coherent_layers: int = 2
    maximum_refinement_factor: int = 4
    voxel_size_microns: float | None = None

    def __post_init__(self) -> None:
        for name, value in (
            ("normal limit", self.normal_limit_degrees),
            ("minimum normal limit", self.minimum_normal_limit_degrees),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 90.0:
                raise ValueError(f"{name} must lie in [0, 90]")
        if (
            not math.isfinite(self.robust_standard_deviations)
            or self.robust_standard_deviations <= 0.0
        ):
            raise ValueError("robust standard deviations must be positive")
        if self.minimum_coherent_layers < 1:
            raise ValueError("minimum coherent layers must be positive")
        if (
            self.maximum_refinement_factor < 1
            or self.maximum_refinement_factor
            & (self.maximum_refinement_factor - 1)
        ):
            raise ValueError("maximum refinement factor must be a power of two")
        if self.voxel_size_microns is not None and (
            not math.isfinite(self.voxel_size_microns)
            or self.voxel_size_microns <= 0.0
        ):
            raise ValueError("voxel size must be finite and positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _angle_summary(values: Iterable[float]) -> dict[str, float | int | None]:
    array = np.asarray(tuple(values), dtype=np.float64)
    if not len(array):
        return {
            "count": 0,
            "medianDegrees": None,
            "p90Degrees": None,
            "maximumDegrees": None,
        }
    return {
        "count": len(array),
        "medianDegrees": round(float(np.median(array)), 6),
        "p90Degrees": round(float(np.percentile(array, 90)), 6),
        "maximumDegrees": round(float(np.max(array)), 6),
    }


def _resolve_normal_limit(
    block: SurfaceBlock,
    settings: SheetResolutionAuditSettings,
) -> tuple[float, dict[str, Any]]:
    retained = np.asarray(
        [math.degrees(value.normal_angle_radians) for value in block.joins],
        dtype=np.float64,
    )
    if settings.normal_limit_degrees > 0.0:
        return settings.normal_limit_degrees, {
            "source": "configured absolute axial-normal limit",
            "retainedJoinCount": len(retained),
            "appliedLimitDegrees": settings.normal_limit_degrees,
        }
    if not len(retained):
        return settings.minimum_normal_limit_degrees, {
            "source": "no retained joins; declared physical floor",
            "retainedJoinCount": 0,
            "appliedLimitDegrees": settings.minimum_normal_limit_degrees,
        }
    median = float(np.median(retained))
    mad = float(np.median(np.abs(retained - median)))
    robust_std = 1.4826 * mad
    limit = min(
        max(
            median + settings.robust_standard_deviations * robust_std,
            settings.minimum_normal_limit_degrees,
        ),
        90.0,
    )
    return limit, {
        "source": "retained-join median plus scaled MAD with physical floor",
        "retainedJoinCount": len(retained),
        "medianDegrees": round(median, 6),
        "madDegrees": round(mad, 6),
        "robustStandardDeviationDegrees": round(robust_std, 6),
        "robustStandardDeviations": settings.robust_standard_deviations,
        "minimumLimitDegrees": settings.minimum_normal_limit_degrees,
        "appliedLimitDegrees": round(limit, 6),
    }


def suggested_refinement_factor(
    normal_angle_degrees: float,
    normal_limit_degrees: float,
    maximum_factor: int,
) -> int:
    """Return a bounded power-of-two stride refinement for one bend."""

    if normal_angle_degrees <= normal_limit_degrees:
        return 1
    required = int(math.ceil(normal_angle_degrees / normal_limit_degrees))
    factor = 1
    while factor < required and factor < maximum_factor:
        factor *= 2
    return min(factor, maximum_factor)


def _relaxed_correspondence_settings() -> TraceMatchSettings:
    # A very large skip cost makes the dynamic program maximize geometrically
    # admissible cardinality.  Endpoint and corner gates retain their defaults;
    # only normal/fiber agreement and its contribution to the joint gate are
    # relaxed.  These correspondences are diagnostic and cannot become joins.
    return TraceMatchSettings(
        maximum_normal_z=1.0e6,
        maximum_fiber_z=1.0e6,
        maximum_absolute_normal_angle_radians=0.5 * math.pi,
        maximum_absolute_fiber_residual_radians=0.5 * math.pi,
        maximum_reduced_chi_square=1.0e12,
        unmatched_negative_log_likelihood=1.0e6,
        orthogonal_fiber_equivalence=True,
    )


def _world_bounds_record(
    block: SurfaceBlock,
    start: Int3,
    stop: Int3,
    voxel_size_microns: float | None,
) -> dict[str, Any]:
    low = block.grid.vertex_world(start)
    high = block.grid.vertex_world(stop)
    result: dict[str, Any] = {
        "startXYZ": [round(float(value), 6) for value in low],
        "stopXYZExclusive": [round(float(value), 6) for value in high],
        "coordinateUnit": block.grid.coordinate_unit,
    }
    if voxel_size_microns is not None and block.grid.coordinate_unit == "source-voxel":
        result["extentMicronsXYZ"] = [
            round(float(value) * voxel_size_microns, 6)
            for value in high - low
        ]
    return result


def _connected_regions(
    refinement_factor_xyz: np.ndarray,
    pressure_xyz: np.ndarray,
    maximum_angle_xyz: np.ndarray,
    coherent_faces: tuple[dict[str, Any], ...],
    block: SurfaceBlock,
    settings: SheetResolutionAuditSettings,
) -> list[dict[str, Any]]:
    active = {
        tuple(int(value) for value in row)
        for row in np.argwhere(refinement_factor_xyz >= 2)
    }
    regions: list[dict[str, Any]] = []
    while active:
        seed = min(active, key=lambda value: (value[2], value[1], value[0]))
        active.remove(seed)
        queue: deque[Int3] = deque((seed,))
        members: list[Int3] = []
        while queue:
            cell = queue.popleft()
            members.append(cell)
            for axis in range(3):
                for delta in (-1, 1):
                    neighbor = list(cell)
                    neighbor[axis] += delta
                    key = tuple(neighbor)
                    if key in active:
                        active.remove(key)
                        queue.append(key)  # type: ignore[arg-type]
        member_set = set(members)
        faces = [
            value
            for value in coherent_faces
            if any(tuple(cell) in member_set for cell in value["adjacentCellsXYZ"])
        ]
        member_array = np.asarray(members, dtype=np.int32)
        start = tuple(int(value) for value in np.min(member_array, axis=0))
        stop = tuple(int(value) + 1 for value in np.max(member_array, axis=0))
        factors = [int(refinement_factor_xyz[cell]) for cell in members]
        maximum_factor = max(factors)
        stride = [
            float(value) / maximum_factor for value in block.grid.cell_size_xyz
        ]
        region: dict[str, Any] = {
            "cellCount": len(members),
            "startCellXYZ": list(start),
            "stopCellXYZExclusive": list(stop),
            "extentCellsXYZ": [stop[axis] - start[axis] for axis in range(3)],
            "coherentFaceCount": len(faces),
            "highBendCorrespondenceCount": sum(
                int(value["highBendCorrespondenceCount"]) for value in faces
            ),
            "foldbackCorrespondenceCount": sum(
                int(value["foldbackCorrespondenceCount"]) for value in faces
            ),
            "maximumNormalAngleDegrees": round(
                max(float(maximum_angle_xyz[cell]) for cell in members), 6
            ),
            "totalResolutionPressure": round(
                sum(float(pressure_xyz[cell]) for cell in members), 6
            ),
            "recommendedRefinementFactor": maximum_factor,
            "recommendedCellStride": [round(value, 6) for value in stride],
            "worldBounds": _world_bounds_record(
                block, start, stop, settings.voxel_size_microns
            ),
        }
        if (
            settings.voxel_size_microns is not None
            and block.grid.coordinate_unit == "source-voxel"
        ):
            region["recommendedCellSizeMicronsXYZ"] = [
                round(value * settings.voxel_size_microns, 6) for value in stride
            ]
        regions.append(region)
    regions.sort(
        key=lambda value: (
            -int(value["recommendedRefinementFactor"]),
            -float(value["maximumNormalAngleDegrees"]),
            -int(value["highBendCorrespondenceCount"]),
            value["startCellXYZ"][2],
            value["startCellXYZ"][1],
            value["startCellXYZ"][0],
        )
    )
    for index, value in enumerate(regions, start=1):
        value["rank"] = index
    return regions


def analyze_sheet_resolution(
    block: SurfaceBlock,
    *,
    settings: SheetResolutionAuditSettings | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Locate coherent bends that exceed one-planar-sheetlet resolution."""

    resolved = settings or SheetResolutionAuditSettings()
    normal_limit, calibration = _resolve_normal_limit(block, resolved)
    relaxed_settings = _relaxed_correspondence_settings()
    patches_by_cell: dict[Int3, list[ClippedPatch]] = defaultdict(list)
    for patch in block.patches:
        patches_by_cell[patch.cell_xyz].append(patch)
    for values in patches_by_cell.values():
        values.sort(key=lambda value: value.patch_id)

    face_records: list[dict[str, Any]] = []
    high_angles: list[float] = []
    analyzed_angles: list[float] = []
    unstable_faces = 0
    for lower in sorted(patches_by_cell, key=lambda value: (value[2], value[1], value[0])):
        for axis in range(3):
            upper_values = list(lower)
            upper_values[axis] += 1
            upper = tuple(upper_values)
            if not block.bounds.contains_cell(upper):
                continue
            face = cell_face(lower, axis, 1)
            first = tuple(patches_by_cell[lower])
            second = tuple(patches_by_cell.get(upper, ()))
            if not first or not second:
                continue
            if not any(value.trace_on(face) is not None for value in first):
                continue
            if not any(value.trace_on(face) is not None for value in second):
                continue
            try:
                alignment = align_face_patches(
                    first,
                    second,
                    face,
                    relaxed_settings,
                    grid=block.grid,
                )
            except ValueError:
                unstable_faces += 1
                continue
            match_records: list[tuple[float, bool]] = []
            first_by_id = {value.patch_id: value for value in first}
            second_by_id = {value.patch_id: value for value in second}
            for match in alignment.matches:
                angle = math.degrees(match.normal_angle_radians)
                analyzed_angles.append(angle)
                offsets = trace_tangent_side_offsets(
                    first_by_id[match.first_patch_id],
                    second_by_id[match.second_patch_id],
                    match,
                )
                foldback = offsets is None or offsets[0] * offsets[1] >= 0.0
                match_records.append((angle, foldback))
            beyond = [value for value in match_records if value[0] > normal_limit]
            if not beyond:
                continue
            angles = [value[0] for value in beyond]
            high_angles.extend(angles)
            coherent = len(beyond) >= resolved.minimum_coherent_layers
            maximum_angle = max(angles)
            factor = suggested_refinement_factor(
                maximum_angle,
                normal_limit,
                resolved.maximum_refinement_factor,
            )
            median_angle = float(np.median(np.asarray(angles, dtype=np.float64)))
            radius_cells = 1.0 / max(math.radians(median_angle), 1.0e-12)
            radius_world = block.grid.cell_size_xyz[axis] * radius_cells
            record: dict[str, Any] = {
                "faceAxis": axis,
                "faceAnchorXYZ": list(face.anchor_xyz),
                "adjacentCellsXYZ": [list(lower), list(upper)],
                "alignedCorrespondenceCount": len(match_records),
                "highBendCorrespondenceCount": len(beyond),
                "foldbackCorrespondenceCount": sum(value[1] for value in beyond),
                "coherentAcrossLayers": coherent,
                "medianNormalAngleDegrees": round(median_angle, 6),
                "maximumNormalAngleDegrees": round(maximum_angle, 6),
                "estimatedRadiusCells": round(radius_cells, 6),
                "estimatedRadiusWorldUnits": round(radius_world, 6),
                "recommendedRefinementFactor": factor if coherent else 1,
            }
            if (
                resolved.voxel_size_microns is not None
                and block.grid.coordinate_unit == "source-voxel"
            ):
                record["estimatedRadiusMicrons"] = round(
                    radius_world * resolved.voxel_size_microns, 6
                )
            face_records.append(record)

    face_records.sort(
        key=lambda value: (
            -int(value["coherentAcrossLayers"]),
            -int(value["recommendedRefinementFactor"]),
            -int(value["highBendCorrespondenceCount"]),
            -float(value["maximumNormalAngleDegrees"]),
            value["faceAxis"],
            value["faceAnchorXYZ"],
        )
    )
    coherent_faces = tuple(
        value for value in face_records if value["coherentAcrossLayers"]
    )
    shape = block.grid.shape_cells_xyz
    pressure = np.zeros(shape, dtype=np.float32)
    maximum_angle = np.zeros(shape, dtype=np.float32)
    refinement_factor = np.ones(shape, dtype=np.uint8)
    coherent_face_count = np.zeros(shape, dtype=np.uint16)
    foldback_count = np.zeros(shape, dtype=np.uint16)
    for face in coherent_faces:
        angles_pressure = (
            float(face["medianNormalAngleDegrees"]) / normal_limit - 1.0
        ) * int(face["highBendCorrespondenceCount"])
        for cell_values in face["adjacentCellsXYZ"]:
            cell = tuple(int(value) for value in cell_values)
            pressure[cell] += max(angles_pressure, 0.0)
            maximum_angle[cell] = max(
                maximum_angle[cell], float(face["maximumNormalAngleDegrees"])
            )
            refinement_factor[cell] = max(
                refinement_factor[cell], int(face["recommendedRefinementFactor"])
            )
            coherent_face_count[cell] += 1
            foldback_count[cell] += int(face["foldbackCorrespondenceCount"])

    regions = _connected_regions(
        refinement_factor,
        pressure,
        maximum_angle,
        coherent_faces,
        block,
        resolved,
    )
    arrays = {
        "cellResolutionPressureXYZ": pressure,
        "cellMaximumNormalAngleDegreesXYZ": maximum_angle,
        "cellRecommendedRefinementFactorXYZ": refinement_factor,
        "cellCoherentFaceCountXYZ": coherent_face_count,
        "cellFoldbackCorrespondenceCountXYZ": foldback_count,
        "faceAxis": np.asarray(
            [value["faceAxis"] for value in face_records], dtype=np.int8
        ),
        "faceAnchorXYZ": np.asarray(
            [value["faceAnchorXYZ"] for value in face_records], dtype=np.int32
        ).reshape(len(face_records), 3),
        "faceHighBendCorrespondenceCount": np.asarray(
            [value["highBendCorrespondenceCount"] for value in face_records],
            dtype=np.uint16,
        ),
        "faceFoldbackCorrespondenceCount": np.asarray(
            [value["foldbackCorrespondenceCount"] for value in face_records],
            dtype=np.uint16,
        ),
        "faceCoherentAcrossLayers": np.asarray(
            [value["coherentAcrossLayers"] for value in face_records], dtype=np.uint8
        ),
        "faceMedianNormalAngleDegrees": np.asarray(
            [value["medianNormalAngleDegrees"] for value in face_records],
            dtype=np.float32,
        ),
        "faceMaximumNormalAngleDegrees": np.asarray(
            [value["maximumNormalAngleDegrees"] for value in face_records],
            dtype=np.float32,
        ),
        "faceRecommendedRefinementFactor": np.asarray(
            [value["recommendedRefinementFactor"] for value in face_records],
            dtype=np.uint8,
        ),
    }
    summary = {
        "method": {
            "sheetletModel": "one planar Acus layer patch per selected layer and cell",
            "directions": "axial/unsigned normal angles",
            "correspondence": (
                "order-preserving face alignment with endpoint and corner gates "
                "retained while normal/fiber gates are relaxed"
            ),
            "interpretation": (
                "coherent high rotation across several ordered layers requests "
                "finer raw-Acus analysis; diagnostic pairs are never graph joins"
            ),
            "radiusProxy": "one face-normal cell step divided by median bend radians",
            "refinementRule": (
                "smallest bounded power of two that brings maximum per-step bend "
                "inside the calibrated locally linear normal limit"
            ),
        },
        "calibration": calibration,
        "grid": {
            "shapeCellsXYZ": list(block.grid.shape_cells_xyz),
            "cellSizeXYZ": list(block.grid.cell_size_xyz),
            "originXYZ": list(block.grid.origin_xyz),
            "coordinateUnit": block.grid.coordinate_unit,
            "voxelSizeMicrons": resolved.voxel_size_microns,
        },
        "statistics": {
            "patches": len(block.patches),
            "retainedJoins": len(block.joins),
            "analyzedCorrespondences": len(analyzed_angles),
            "unstableFaces": unstable_faces,
            "facesWithHighBendCorrespondences": len(face_records),
            "coherentHighBendFaces": len(coherent_faces),
            "coherentHighBendCorrespondences": sum(
                int(value["highBendCorrespondenceCount"])
                for value in coherent_faces
            ),
            "coherentFoldbackCorrespondences": sum(
                int(value["foldbackCorrespondenceCount"])
                for value in coherent_faces
            ),
            "cellsRecommendedForRefinement": int(
                np.count_nonzero(refinement_factor >= 2)
            ),
            "cellsRecommendedAtLeast4x": int(
                np.count_nonzero(refinement_factor >= 4)
            ),
            "connectedRefinementRegions": len(regions),
            "allAlignedNormalAngles": _angle_summary(analyzed_angles),
            "highBendNormalAngles": _angle_summary(high_angles),
        },
        "topFaces": face_records[:64],
        "refinementRegions": regions[:64],
    }
    return summary, arrays


def run_sheet_resolution_audit(
    graph_root: str | Path,
    output_root: str | Path,
    *,
    settings: SheetResolutionAuditSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write a reloadable adaptive-resolution target artifact."""

    started = time.monotonic()
    resolved = settings or SheetResolutionAuditSettings()
    source = Path(graph_root).resolve()
    output = Path(output_root).resolve()
    if output == source:
        raise ValueError("sheet-resolution output must differ from its graph root")
    module_root = Path(__file__).resolve().parent
    identity: dict[str, Any] = {
        "schema": SHEET_RESOLUTION_AUDIT_SCHEMA,
        "version": SHEET_RESOLUTION_AUDIT_VERSION,
        "graphRoot": str(source),
        "surfaceGraphManifestSha256": sha256_file(
            source / "surface-graph-v1.json"
        ),
        "surfaceGraphDataSha256": sha256_file(source / "surface-graph-v1.npz"),
        "selectedPatchManifestSha256": sha256_file(
            source / "selected-patches-v1.json"
        ),
        "selectedPatchDataSha256": sha256_file(
            source / "selected-patches-v1.npz"
        ),
        "settings": resolved.record(),
        "implementationSha256": {
            "sheet_resolution.py": sha256_file(Path(__file__)),
            "matching.py": sha256_file(module_root / "matching.py"),
            "surface_graph.py": sha256_file(module_root / "surface_graph.py"),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output_path = output / f"{SHEET_RESOLUTION_AUDIT_STEM}.json"
    if output_path.is_file() and not force:
        prior = json.loads(output_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("sheet-resolution output belongs to another identity")
        if prior.get("state") == "complete":
            return prior
    block = read_surface_graph(source, verify=True)
    summary, arrays = analyze_sheet_resolution(block, settings=resolved)
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / f"{SHEET_RESOLUTION_AUDIT_STEM}.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(data_path)
    payload = {
        "schema": SHEET_RESOLUTION_AUDIT_SCHEMA,
        "version": SHEET_RESOLUTION_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        **summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
        "elapsedSeconds": round(time.monotonic() - started, 6),
    }
    atomic_json(output_path, payload)
    return payload
