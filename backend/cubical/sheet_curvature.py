from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .geometry import ClippedPatch, axial_angle_radians, canonical_axis
from .matching import TraceMatch
from .surface_graph import join_key


JoinKey = tuple[int, int, int, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class RobustAngularScale:
    sample_count: int
    median_degrees: float
    mad_degrees: float
    robust_standard_deviation_degrees: float
    upper_limit_degrees: float

    def record(self) -> dict[str, Any]:
        return {
            "sampleCount": self.sample_count,
            "medianDegrees": round(self.median_degrees, 6),
            "madDegrees": round(self.mad_degrees, 6),
            "robustStandardDeviationDegrees": round(
                self.robust_standard_deviation_degrees, 6
            ),
            "upperLimitDegrees": round(self.upper_limit_degrees, 6),
        }


@dataclass(frozen=True, slots=True)
class JoinCurvature:
    key: JoinKey
    direct_bend_degrees: float
    branch_bend_degrees: float | None
    branch_contrast_degrees: float | None
    first_support: int
    second_support: int
    first_dispersion_p90_degrees: float | None
    second_dispersion_p90_degrees: float | None
    pressure: float
    flagged: bool

    def record(self) -> dict[str, Any]:
        return {
            "firstPatchId": self.key[0],
            "secondPatchId": self.key[1],
            "faceAxis": self.key[2],
            "faceAnchorXYZ": list(self.key[3]),
            "directBendDegrees": round(self.direct_bend_degrees, 6),
            "branchBendDegrees": (
                round(self.branch_bend_degrees, 6)
                if self.branch_bend_degrees is not None
                else None
            ),
            "branchContrastDegrees": (
                round(self.branch_contrast_degrees, 6)
                if self.branch_contrast_degrees is not None
                else None
            ),
            "firstSupport": self.first_support,
            "secondSupport": self.second_support,
            "firstDispersionP90Degrees": (
                round(self.first_dispersion_p90_degrees, 6)
                if self.first_dispersion_p90_degrees is not None
                else None
            ),
            "secondDispersionP90Degrees": (
                round(self.second_dispersion_p90_degrees, 6)
                if self.second_dispersion_p90_degrees is not None
                else None
            ),
            "pressure": round(self.pressure, 6),
            "flagged": self.flagged,
        }


@dataclass(frozen=True, slots=True)
class SheetCurvatureAnalysis:
    direct_scale: RobustAngularScale
    branch_contrast_scale: RobustAngularScale
    join_curvature: tuple[JoinCurvature, ...]
    component_records: tuple[dict[str, Any], ...]
    calibration_sufficient: bool

    def by_key(self) -> dict[JoinKey, JoinCurvature]:
        return {value.key: value for value in self.join_curvature}

    def record(self, *, maximum_components: int = 32) -> dict[str, Any]:
        flagged = tuple(value for value in self.join_curvature if value.flagged)
        return {
            "method": {
                "directions": "axial/unsigned normal tensors",
                "localSignal": "unsigned normal bend across each retained join",
                "multiscaleSignal": (
                    "difference between axial mean normals in graph-local "
                    "half-neighborhoods, less their within-side dispersion"
                ),
                "globalNormalConeIsDiagnosticOnly": True,
            },
            "calibrationSufficient": self.calibration_sufficient,
            "directBendScale": self.direct_scale.record(),
            "branchContrastScale": self.branch_contrast_scale.record(),
            "retainedJoins": len(self.join_curvature),
            "flaggedJoins": len(flagged),
            "maximumPressure": round(
                max((value.pressure for value in self.join_curvature), default=0.0),
                6,
            ),
            "components": list(self.component_records[:maximum_components]),
        }


def _robust_angular_scale(
    values: Iterable[float],
    *,
    standard_deviations: float,
) -> RobustAngularScale:
    samples = np.asarray(tuple(float(value) for value in values), dtype=np.float64)
    if not len(samples):
        return RobustAngularScale(0, 0.0, 0.0, 0.0, math.inf)
    median = float(np.median(samples))
    mad = float(np.median(np.abs(samples - median)))
    robust_std = 1.4826 * mad
    return RobustAngularScale(
        len(samples),
        median,
        mad,
        robust_std,
        median + standard_deviations * robust_std,
    )


def _axial_mean_normal(
    patch_by_id: Mapping[int, ClippedPatch],
    distances: Mapping[int, int],
) -> np.ndarray:
    accumulator = np.zeros((3, 3), dtype=np.float64)
    for patch_id, distance in distances.items():
        patch = patch_by_id[patch_id]
        normal = canonical_axis(patch.estimate.normal_xyz)
        weight = max(float(patch.estimate.confidence), 1.0e-6) / (1.0 + distance)
        accumulator += weight * np.outer(normal, normal)
    eigenvalues, eigenvectors = np.linalg.eigh(accumulator)
    if float(eigenvalues[-1]) <= 1.0e-12:
        raise ValueError("sheet neighborhood does not define an axial mean normal")
    return canonical_axis(eigenvectors[:, -1])


def _normal_dispersion_p90(
    patch_by_id: Mapping[int, ClippedPatch],
    patch_ids: Iterable[int],
    mean_normal: np.ndarray,
) -> float:
    angles = np.asarray(
        [
            math.degrees(
                axial_angle_radians(
                    patch_by_id[patch_id].estimate.normal_xyz,
                    mean_normal,
                )
            )
            for patch_id in patch_ids
        ],
        dtype=np.float64,
    )
    return float(np.percentile(angles, 90)) if len(angles) else 0.0


def _component_partition(
    patch_ids: Iterable[int],
    joins: tuple[TraceMatch, ...],
) -> tuple[dict[int, int], dict[int, tuple[int, ...]]]:
    parent = {patch_id: patch_id for patch_id in patch_ids}
    size = {patch_id: 1 for patch_id in patch_ids}

    def find(value: int) -> int:
        root = value
        while parent[root] != root:
            root = parent[root]
        while parent[value] != value:
            following = parent[value]
            parent[value] = root
            value = following
        return root

    def union(first: int, second: int) -> None:
        first_root = find(first)
        second_root = find(second)
        if first_root == second_root:
            return
        if size[first_root] < size[second_root]:
            first_root, second_root = second_root, first_root
        parent[second_root] = first_root
        size[first_root] += size.pop(second_root)

    for value in joins:
        union(value.first_patch_id, value.second_patch_id)
    unordered: dict[int, list[int]] = defaultdict(list)
    for patch_id in parent:
        unordered[find(patch_id)].append(patch_id)
    members = {
        min(values): tuple(sorted(values)) for values in unordered.values()
    }
    by_patch = {
        patch_id: component_id
        for component_id, values in members.items()
        for patch_id in values
    }
    return by_patch, members


def _half_neighborhood(
    patch_by_id: Mapping[int, ClippedPatch],
    adjacency: Mapping[int, tuple[tuple[int, JoinKey], ...]],
    *,
    root: int,
    blocked_key: JoinKey,
    face_axis: int,
    face_coordinate: int,
    lower_side: bool,
    radius: int,
) -> dict[int, int]:
    result = {root: 0}
    queue: deque[int] = deque((root,))
    while queue:
        first = queue.popleft()
        distance = result[first]
        if distance >= radius:
            continue
        for second, key in adjacency.get(first, ()):
            if key == blocked_key or second in result:
                continue
            coordinate = patch_by_id[second].cell_xyz[face_axis]
            if lower_side:
                allowed = coordinate < face_coordinate
            else:
                allowed = coordinate >= face_coordinate
            if not allowed:
                continue
            result[second] = distance + 1
            queue.append(second)
    return result


def _raw_join_geometry(
    patches: tuple[ClippedPatch, ...],
    joins: tuple[TraceMatch, ...],
    *,
    neighborhood_radius: int,
    minimum_branch_support: int,
) -> tuple[dict[JoinKey, dict[str, Any]], dict[int, int], dict[int, tuple[int, ...]]]:
    patch_by_id = {value.patch_id: value for value in patches}
    adjacency_values: dict[int, list[tuple[int, JoinKey]]] = defaultdict(list)
    for value in joins:
        key = join_key(value)
        adjacency_values[value.first_patch_id].append((value.second_patch_id, key))
        adjacency_values[value.second_patch_id].append((value.first_patch_id, key))
    adjacency = {
        patch_id: tuple(sorted(values, key=lambda value: (value[0], value[1])))
        for patch_id, values in adjacency_values.items()
    }
    component_by_patch, members = _component_partition(patch_by_id, joins)
    raw: dict[JoinKey, dict[str, Any]] = {}
    for value in joins:
        key = join_key(value)
        lower_cell, upper_cell = value.face.adjacent_cells()
        first_cell = patch_by_id[value.first_patch_id].cell_xyz
        second_cell = patch_by_id[value.second_patch_id].cell_xyz
        if {first_cell, second_cell} != {lower_cell, upper_cell}:
            raise ValueError("retained join endpoints do not occupy adjacent face cells")
        if first_cell == lower_cell:
            lower_patch_id = value.first_patch_id
            upper_patch_id = value.second_patch_id
        else:
            lower_patch_id = value.second_patch_id
            upper_patch_id = value.first_patch_id
        lower = _half_neighborhood(
            patch_by_id,
            adjacency,
            root=lower_patch_id,
            blocked_key=key,
            face_axis=value.face.axis,
            face_coordinate=value.face.anchor_xyz[value.face.axis],
            lower_side=True,
            radius=neighborhood_radius,
        )
        upper = _half_neighborhood(
            patch_by_id,
            adjacency,
            root=upper_patch_id,
            blocked_key=key,
            face_axis=value.face.axis,
            face_coordinate=value.face.anchor_xyz[value.face.axis],
            lower_side=False,
            radius=neighborhood_radius,
        )
        branch_bend: float | None = None
        branch_contrast: float | None = None
        lower_dispersion: float | None = None
        upper_dispersion: float | None = None
        if (
            len(lower) >= minimum_branch_support
            and len(upper) >= minimum_branch_support
        ):
            lower_mean = _axial_mean_normal(patch_by_id, lower)
            upper_mean = _axial_mean_normal(patch_by_id, upper)
            branch_bend = math.degrees(
                axial_angle_radians(lower_mean, upper_mean)
            )
            lower_dispersion = _normal_dispersion_p90(
                patch_by_id, lower, lower_mean
            )
            upper_dispersion = _normal_dispersion_p90(
                patch_by_id, upper, upper_mean
            )
            branch_contrast = max(
                0.0,
                branch_bend - max(lower_dispersion, upper_dispersion),
            )
        raw[key] = {
            "direct": math.degrees(value.normal_angle_radians),
            "branch": branch_bend,
            "contrast": branch_contrast,
            "firstSupport": len(lower),
            "secondSupport": len(upper),
            "firstDispersion": lower_dispersion,
            "secondDispersion": upper_dispersion,
        }
    return raw, component_by_patch, members


def _safe_ratio(value: float, limit: float) -> float:
    if math.isinf(limit):
        return 0.0
    if value <= 1.0e-9:
        return 0.0
    return value / max(limit, 1.0e-6)


def analyze_sheet_curvature(
    patches: Iterable[ClippedPatch],
    joins: Iterable[TraceMatch],
    *,
    neighborhood_radius: int = 3,
    minimum_branch_support: int = 3,
    robust_standard_deviations: float = 3.0,
    minimum_calibration_joins: int = 32,
    calibration: SheetCurvatureAnalysis | None = None,
) -> SheetCurvatureAnalysis:
    """Measure abrupt sheet bends without assigning a sign to any normal.

    The direct signal catches a one-cell hinge.  The multiscale signal compares
    coherent graph-local neighborhoods on opposite sides of the shared face;
    subtracting their internal dispersion keeps smooth macro-curvature from
    looking like a discontinuity.  Whole-component normal spread is reported
    for inspection but is never a rejection rule.
    """

    patch_values = tuple(patches)
    join_values = tuple(joins)
    if neighborhood_radius < 1:
        raise ValueError("curvature neighborhood radius must be positive")
    if minimum_branch_support < 1:
        raise ValueError("minimum branch support must be positive")
    if robust_standard_deviations <= 0.0 or not math.isfinite(
        robust_standard_deviations
    ):
        raise ValueError("robust curvature deviations must be finite and positive")
    if minimum_calibration_joins < 1:
        raise ValueError("minimum curvature calibration joins must be positive")
    raw, component_by_patch, members = _raw_join_geometry(
        patch_values,
        join_values,
        neighborhood_radius=neighborhood_radius,
        minimum_branch_support=minimum_branch_support,
    )
    if calibration is None:
        direct_scale = _robust_angular_scale(
            (value["direct"] for value in raw.values()),
            standard_deviations=robust_standard_deviations,
        )
        contrast_scale = _robust_angular_scale(
            (
                float(value["contrast"])
                for value in raw.values()
                # Exact zero is the common, intended outcome when the observed
                # between-side bend is fully explained by smooth within-side
                # curvature.  Calibrating the positive residual population avoids
                # turning that point mass into a zero-width noise model.
                if value["contrast"] is not None
                and float(value["contrast"]) > 1.0e-9
            ),
            standard_deviations=robust_standard_deviations,
        )
        calibration_sufficient = (
            direct_scale.sample_count >= minimum_calibration_joins
            and contrast_scale.sample_count >= minimum_calibration_joins
        )
    else:
        direct_scale = calibration.direct_scale
        contrast_scale = calibration.branch_contrast_scale
        calibration_sufficient = calibration.calibration_sufficient
    curvature: list[JoinCurvature] = []
    for key, value in sorted(raw.items()):
        direct_pressure = _safe_ratio(
            float(value["direct"]), direct_scale.upper_limit_degrees
        )
        contrast = value["contrast"]
        branch_pressure = (
            _safe_ratio(float(contrast), contrast_scale.upper_limit_degrees)
            if contrast is not None
            else 0.0
        )
        pressure = max(direct_pressure, branch_pressure)
        flagged = calibration_sufficient and pressure > 1.0 + 1.0e-9
        curvature.append(
            JoinCurvature(
                key,
                float(value["direct"]),
                value["branch"],
                contrast,
                int(value["firstSupport"]),
                int(value["secondSupport"]),
                value["firstDispersion"],
                value["secondDispersion"],
                pressure,
                flagged,
            )
        )
    curvature_by_key = {value.key: value for value in curvature}
    patch_by_id = {value.patch_id: value for value in patch_values}
    joins_by_component: dict[int, list[TraceMatch]] = defaultdict(list)
    for value in join_values:
        joins_by_component[component_by_patch[value.first_patch_id]].append(value)
    component_records: list[dict[str, Any]] = []
    for component_id, member_ids in members.items():
        component_joins = joins_by_component.get(component_id, ())
        normals = {
            patch_id: 0 for patch_id in member_ids
        }
        mean_normal = _axial_mean_normal(patch_by_id, normals)
        cone = np.asarray(
            [
                math.degrees(
                    axial_angle_radians(
                        patch_by_id[patch_id].estimate.normal_xyz,
                        mean_normal,
                    )
                )
                for patch_id in member_ids
            ],
            dtype=np.float64,
        )
        direct = np.asarray(
            [curvature_by_key[join_key(value)].direct_bend_degrees for value in component_joins],
            dtype=np.float64,
        )
        contrast = np.asarray(
            [
                curvature_by_key[join_key(value)].branch_contrast_degrees
                for value in component_joins
                if curvature_by_key[join_key(value)].branch_contrast_degrees is not None
            ],
            dtype=np.float64,
        )
        flagged_count = sum(
            curvature_by_key[join_key(value)].flagged for value in component_joins
        )

        def percentile(values: np.ndarray, value: float) -> float | None:
            return round(float(np.percentile(values, value)), 6) if len(values) else None

        cells = np.asarray(
            [patch_by_id[patch_id].cell_xyz for patch_id in member_ids],
            dtype=np.int64,
        )
        extent = np.max(cells, axis=0) - np.min(cells, axis=0) + 1
        component_records.append(
            {
                "componentId": component_id,
                "patchCount": len(member_ids),
                "joinCount": len(component_joins),
                "extentCellsXYZ": [int(value) for value in extent],
                "flaggedJoins": flagged_count,
                "maximumPressure": round(
                    max(
                        (
                            curvature_by_key[join_key(value)].pressure
                            for value in component_joins
                        ),
                        default=0.0,
                    ),
                    6,
                ),
                "directBendDegrees": {
                    "median": percentile(direct, 50),
                    "p90": percentile(direct, 90),
                    "maximum": percentile(direct, 100),
                },
                "branchContrastDegrees": {
                    "median": percentile(contrast, 50),
                    "p90": percentile(contrast, 90),
                    "maximum": percentile(contrast, 100),
                },
                "globalNormalConeDegreesDiagnosticOnly": {
                    "median": percentile(cone, 50),
                    "p90": percentile(cone, 90),
                    "maximum": percentile(cone, 100),
                },
            }
        )
    component_records.sort(
        key=lambda value: (
            -int(value["patchCount"]),
            int(value["componentId"]),
        )
    )
    return SheetCurvatureAnalysis(
        direct_scale,
        contrast_scale,
        tuple(curvature),
        tuple(component_records),
        calibration_sufficient,
    )
