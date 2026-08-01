from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import _draw_text
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_patch_holes import (
    PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
    _loop_vertices,
    _sample_normal_profiles,
)
from .physical_ribbon_surface_holes import PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA
from .physical_ribbon_open_bays import PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA


PHYSICAL_RIBBON_DEPTH_FIELD_SCHEMA = "pareidolia.physical-ribbon-depth-field"
PHYSICAL_RIBBON_DEPTH_FIELD_VERSION = 1
PHYSICAL_RIBBON_DEPTH_FIELD_STEM = "physical-ribbon-depth-field-v1"


def _resolve_holes_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("schema")
            in {
                PHYSICAL_RIBBON_PATCH_HOLES_SCHEMA,
                PHYSICAL_RIBBON_SURFACE_HOLES_SCHEMA,
                PHYSICAL_RIBBON_OPEN_BAYS_SCHEMA,
            }
            and manifest.get("state") == "complete"
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(
            "holes root must identify exactly one complete patch, surface-hole, "
            "or open-bay artifact"
        )
    return matches[0]


@dataclass(frozen=True, slots=True)
class PhysicalRibbonDepthFieldSettings:
    """Dense, label-free CT evidence over complete missing surface patches.

    A pixel is never an expansion decision.  Every hole is one ordered-label
    field whose scan evidence, boundary attachment, and spatial regularity are
    optimized together.  Pairwise costs are truncated so a coherent
    delamination step can survive instead of being smoothed into the wrong
    layer.
    """

    minimum_shift_thicknesses: float = -1.5
    maximum_shift_thicknesses: float = 1.5
    shift_step_thicknesses: float = 0.125
    profile_depth_fractions: tuple[float, ...] = (
        -0.85,
        -0.65,
        -0.35,
        0.0,
        0.35,
        0.65,
        0.85,
    )
    profile_correlation_weight: float = 0.35
    spatial_smoothness_weight: float = 0.18
    smoothness_truncation_thicknesses: float = 0.75
    boundary_anchor_weight: float = 0.18
    boundary_anchor_decay_raster_steps: float = 2.0
    minimum_context_contrast_fraction: float = 0.50
    minimum_profile_correlation: float = 0.35
    competing_label_separation_thicknesses: float = 0.50
    candidate_shift_tolerance_thicknesses: float = 0.30
    minimum_coherent_support_fraction: float = 0.70
    minimum_candidate_coverage_fraction: float = 0.75
    maximum_solver_sweeps: int = 12
    maximum_preview_holes: int = 24

    def __post_init__(self) -> None:
        finite = (
            self.minimum_shift_thicknesses,
            self.maximum_shift_thicknesses,
            self.shift_step_thicknesses,
            self.profile_correlation_weight,
            self.spatial_smoothness_weight,
            self.smoothness_truncation_thicknesses,
            self.boundary_anchor_weight,
            self.boundary_anchor_decay_raster_steps,
            self.minimum_context_contrast_fraction,
            self.minimum_profile_correlation,
            self.competing_label_separation_thicknesses,
            self.candidate_shift_tolerance_thicknesses,
            self.minimum_coherent_support_fraction,
            self.minimum_candidate_coverage_fraction,
        )
        if any(not math.isfinite(value) for value in finite):
            raise ValueError("depth-field settings must be finite")
        if not self.minimum_shift_thicknesses < self.maximum_shift_thicknesses:
            raise ValueError("depth-field shift bounds must be ordered")
        if self.shift_step_thicknesses <= 0.0:
            raise ValueError("depth-field shift step must be positive")
        if self.spatial_smoothness_weight < 0.0 or self.boundary_anchor_weight < 0.0:
            raise ValueError("depth-field regularization weights cannot be negative")
        if (
            self.smoothness_truncation_thicknesses <= 0.0
            or self.boundary_anchor_decay_raster_steps <= 0.0
            or self.competing_label_separation_thicknesses <= 0.0
            or self.candidate_shift_tolerance_thicknesses <= 0.0
        ):
            raise ValueError("depth-field physical scales must be positive")
        for value in (
            self.minimum_context_contrast_fraction,
            self.minimum_coherent_support_fraction,
            self.minimum_candidate_coverage_fraction,
        ):
            if not 0.0 <= value <= 1.0:
                raise ValueError("depth-field fractions must lie in [0, 1]")
        if not -1.0 <= self.minimum_profile_correlation <= 1.0:
            raise ValueError("profile correlation threshold must lie in [-1, 1]")
        if self.maximum_solver_sweeps < 1 or self.maximum_preview_holes < 1:
            raise ValueError("depth-field iteration counts must be positive")
        depths = tuple(float(value) for value in self.profile_depth_fractions)
        if (
            len(depths) < 5
            or tuple(sorted(depths)) != depths
            or 0.0 not in depths
        ):
            raise ValueError("profile depths must be sorted and include zero")

    def shifts(self) -> np.ndarray:
        count = int(
            round(
                (self.maximum_shift_thicknesses - self.minimum_shift_thicknesses)
                / self.shift_step_thicknesses
            )
        )
        values = self.minimum_shift_thicknesses + self.shift_step_thicknesses * np.arange(
            count + 1, dtype=np.float32
        )
        if abs(float(values[-1]) - self.maximum_shift_thicknesses) > 1.0e-5:
            raise ValueError("shift interval must be divisible by its step")
        return values

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _grid_coordinates(patch_uv: np.ndarray, step: float) -> np.ndarray:
    uv = np.asarray(patch_uv, dtype=np.float64)
    if not len(uv):
        return np.empty((0, 2), dtype=np.int32)
    coordinates = np.rint((uv - np.min(uv, axis=0)) / max(step, 1.0e-6)).astype(
        np.int32
    )
    if len(np.unique(coordinates, axis=0)) != len(coordinates):
        raise ValueError("patch raster does not map to unique grid coordinates")
    return coordinates


def _grid_edges(coordinates: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    lookup = {tuple(int(v) for v in value): row for row, value in enumerate(coordinates)}
    first: list[int] = []
    second: list[int] = []
    for row, (u_value, v_value) in enumerate(coordinates):
        for offset in ((1, 0), (0, 1)):
            neighbor = lookup.get((int(u_value + offset[0]), int(v_value + offset[1])))
            if neighbor is not None:
                first.append(row)
                second.append(neighbor)
    return np.asarray(first, dtype=np.int32), np.asarray(second, dtype=np.int32)


def _grid_lines(coordinates: np.ndarray, axis: int) -> tuple[np.ndarray, ...]:
    """Return maximal contiguous raster lines parallel to ``axis``."""

    other = 1 - axis
    groups: dict[int, list[tuple[int, int]]] = {}
    for row, value in enumerate(coordinates):
        groups.setdefault(int(value[other]), []).append((int(value[axis]), row))
    lines: list[np.ndarray] = []
    for fixed in sorted(groups):
        ordered = sorted(groups[fixed])
        start = 0
        for stop in range(1, len(ordered) + 1):
            separated = stop == len(ordered) or ordered[stop][0] != ordered[stop - 1][0] + 1
            if separated:
                lines.append(
                    np.asarray([row for _, row in ordered[start:stop]], dtype=np.int32)
                )
                start = stop
    return tuple(lines)


def _adjacency(
    node_count: int, first: np.ndarray, second: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    degree = np.bincount(
        np.concatenate((first, second)), minlength=node_count
    ).astype(np.int64)
    offset = np.concatenate(([0], np.cumsum(degree))).astype(np.int64)
    cursor = offset[:-1].copy()
    neighbor = np.empty(int(offset[-1]), dtype=np.int32)
    for left, right in zip(first, second):
        neighbor[cursor[left]] = right
        cursor[left] += 1
        neighbor[cursor[right]] = left
        cursor[right] += 1
    return offset, neighbor


def _field_objective(
    unary: np.ndarray,
    labels: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
    pair_score: np.ndarray,
) -> float:
    row = np.arange(len(labels))
    return float(
        np.sum(unary[row, labels])
        + np.sum(pair_score[labels[first], labels[second]])
    )


def _optimize_line(
    line: np.ndarray,
    labels: np.ndarray,
    unary: np.ndarray,
    pair_score: np.ndarray,
    offset: np.ndarray,
    neighbor: np.ndarray,
    line_membership: np.ndarray,
) -> np.ndarray:
    label_count = unary.shape[1]
    emission = unary[line].astype(np.float64).copy()
    for position, node in enumerate(line):
        for adjacent in neighbor[int(offset[node]) : int(offset[node + 1])]:
            if line_membership[adjacent]:
                continue
            emission[position] += pair_score[:, labels[adjacent]]
    score = emission[0].copy()
    back = np.empty((len(line), label_count), dtype=np.int16)
    back[0] = -1
    for position in range(1, len(line)):
        transition = score[:, None] + pair_score
        parent = np.argmax(transition, axis=0)
        score = transition[parent, np.arange(label_count)] + emission[position]
        back[position] = parent.astype(np.int16)
    result = np.empty(len(line), dtype=np.int16)
    result[-1] = int(np.argmax(score))
    for position in range(len(line) - 1, 0, -1):
        result[position - 1] = back[position, result[position]]
    return result


def solve_collective_depth_labels(
    unary_score: np.ndarray,
    coordinates: np.ndarray,
    shifts: np.ndarray,
    *,
    smoothness_weight: float,
    truncation_thicknesses: float,
    maximum_sweeps: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Optimize one complete patch using alternating exact raster-line solves.

    Each row and column is updated as one Viterbi chain while its orthogonal
    neighbors remain fixed.  This crosses isolated unary barriers that a
    single-pixel frontier or ICM update cannot cross.  Multiple global starts
    make the result independent of one greedy growth history.
    """

    unary = np.asarray(unary_score, dtype=np.float64)
    coordinates = np.asarray(coordinates, dtype=np.int32)
    shifts = np.asarray(shifts, dtype=np.float64)
    if unary.ndim != 2 or len(unary) != len(coordinates):
        raise ValueError("depth-field unary and coordinates differ")
    if unary.shape[1] != len(shifts) or not len(shifts):
        raise ValueError("depth-field unary and labels differ")
    first, second = _grid_edges(coordinates)
    offset, neighbor = _adjacency(len(coordinates), first, second)
    pair_score = -smoothness_weight * np.minimum(
        np.abs(shifts[:, None] - shifts[None, :]), truncation_thicknesses
    )
    independent = np.argmax(unary, axis=1).astype(np.int16)
    zero_label = int(np.argmin(np.abs(shifts)))
    constant_score = np.sum(unary, axis=0)
    start_labels = [independent, np.full(len(unary), zero_label, dtype=np.int16)]
    for label in np.argsort(constant_score, kind="stable")[-3:][::-1]:
        start_labels.append(np.full(len(unary), int(label), dtype=np.int16))
    unique_starts: list[np.ndarray] = []
    seen: set[bytes] = set()
    for value in start_labels:
        key = value.tobytes()
        if key not in seen:
            seen.add(key)
            unique_starts.append(value)
    lines = (_grid_lines(coordinates, 0), _grid_lines(coordinates, 1))
    best_labels = independent.copy()
    best_objective = _field_objective(unary, best_labels, first, second, pair_score)
    run_records: list[dict[str, Any]] = []
    for start_index, start in enumerate(unique_starts):
        labels = start.copy()
        initial = _field_objective(unary, labels, first, second, pair_score)
        sweeps = 0
        for sweep in range(maximum_sweeps):
            before = _field_objective(unary, labels, first, second, pair_score)
            for axis_lines in lines:
                for line in axis_lines:
                    membership = np.zeros(len(labels), dtype=bool)
                    membership[line] = True
                    labels[line] = _optimize_line(
                        line,
                        labels,
                        unary,
                        pair_score,
                        offset,
                        neighbor,
                        membership,
                    )
            sweeps = sweep + 1
            after = _field_objective(unary, labels, first, second, pair_score)
            if after + 1.0e-8 < before:
                raise RuntimeError("collective depth-field objective decreased")
            if after <= before + 1.0e-7:
                break
        objective = _field_objective(unary, labels, first, second, pair_score)
        run_records.append(
            {
                "start": start_index,
                "initialObjective": round(initial, 6),
                "finalObjective": round(objective, 6),
                "sweeps": sweeps,
            }
        )
        if objective > best_objective + 1.0e-8:
            best_objective = objective
            best_labels = labels.copy()
    return best_labels.astype(np.int16), {
        "algorithm": "multi-start alternating exact row/column Viterbi fields",
        "singlePixelGrowth": False,
        "nodeCount": int(len(unary)),
        "edgeCount": int(len(first)),
        "startCount": len(unique_starts),
        "independentObjective": round(
            _field_objective(unary, independent, first, second, pair_score), 6
        ),
        "collectiveObjective": round(best_objective, 6),
        "runs": run_records,
    }


def _point_segment_distance(
    points: np.ndarray, polygon: np.ndarray
) -> np.ndarray:
    distance = np.full(len(points), np.inf, dtype=np.float64)
    for first, second in zip(polygon, np.roll(polygon, -1, axis=0)):
        edge = second - first
        denominator = max(float(np.dot(edge, edge)), 1.0e-12)
        parameter = np.clip(np.einsum("ij,j->i", points - first, edge) / denominator, 0.0, 1.0)
        projected = first[None, :] + parameter[:, None] * edge[None, :]
        distance = np.minimum(distance, np.linalg.norm(points - projected, axis=1))
    return distance.astype(np.float32)


def _profile_fields(
    profiles: np.ndarray,
    context_profile: np.ndarray,
    depth_fractions: np.ndarray,
    intensity_scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(profiles, dtype=np.float32)
    depth = np.asarray(depth_fractions, dtype=np.float32)
    inside = np.abs(depth) <= 0.36
    outside = np.abs(depth) >= 0.64
    physical = (
        np.nanmean(values[..., inside], axis=-1)
        - np.nanmean(values[..., outside], axis=-1)
    ) / max(float(intensity_scale), 1.0)
    centered = values - np.nanmean(values, axis=-1, keepdims=True)
    context = np.asarray(context_profile, dtype=np.float32)
    context = context - np.nanmean(context)
    numerator = np.nansum(centered * context[None, None, :], axis=-1)
    denominator = np.sqrt(np.nansum(centered * centered, axis=-1)) * max(
        float(np.linalg.norm(context)), 1.0e-6
    )
    correlation = numerator / np.maximum(denominator, 1.0e-6)
    correlation[~np.all(np.isfinite(values), axis=-1)] = np.nan
    return physical.astype(np.float32), correlation.astype(np.float32)


def _coherent_supported_fraction(
    supported: np.ndarray, first: np.ndarray, second: np.ndarray
) -> float:
    """Measure support continuity relative to the raster's own components.

    A pinched polygon can rasterize into several disconnected islands.  That
    is a property of the fitted patch domain, not missing CT support.  Within
    each such island we retain its largest supported region, then normalize
    their sum by the complete patch area.
    """

    supported = np.asarray(supported, dtype=bool)
    if not len(supported):
        return 0.0
    offset, neighbor = _adjacency(len(supported), first, second)
    unseen_grid = set(range(len(supported)))
    retained = 0
    while unseen_grid:
        stack = [unseen_grid.pop()]
        grid_component: list[int] = []
        while stack:
            node = stack.pop()
            grid_component.append(node)
            for adjacent in neighbor[int(offset[node]) : int(offset[node + 1])]:
                value = int(adjacent)
                if value in unseen_grid:
                    unseen_grid.remove(value)
                    stack.append(value)
        unseen_support = {
            value for value in grid_component if supported[value]
        }
        largest = 0
        while unseen_support:
            stack = [unseen_support.pop()]
            size = 0
            while stack:
                node = stack.pop()
                size += 1
                for adjacent in neighbor[
                    int(offset[node]) : int(offset[node + 1])
                ]:
                    value = int(adjacent)
                    if value in unseen_support:
                        unseen_support.remove(value)
                        stack.append(value)
            largest = max(largest, size)
        retained += largest
    return float(retained) / len(supported)


def _coverage_fields(
    holes: Mapping[str, np.ndarray],
    row: int,
    patch_xyz: np.ndarray,
    patch_normal: np.ndarray,
    chosen_shift: np.ndarray,
    median_thickness: float,
    radius_voxels: float,
    shift_tolerance: float,
) -> dict[str, np.ndarray]:
    candidate_offset = np.asarray(holes["patchCandidateOffset"], dtype=np.int64)
    start, stop = int(candidate_offset[row]), int(candidate_offset[row + 1])
    candidate = np.asarray(holes["patchCandidateFrontierIndex"], dtype=np.int32)[start:stop]
    nearest = np.asarray(holes["patchCandidateNearestPixel"], dtype=np.int32)[start:stop]
    alignment = np.asarray(holes["patchCandidateSurfaceAlignment"], dtype=np.float32)[start:stop]
    direct = np.zeros(len(patch_xyz), dtype=bool)
    compatible_direct = np.zeros(len(patch_xyz), dtype=bool)
    maximum_alignment = np.zeros(len(patch_xyz), dtype=np.float32)
    if len(candidate):
        candidate_xyz = np.asarray(holes["midpointXYZ"], dtype=np.float32)[candidate]
        signed_shift = np.einsum(
            "ij,ij->i", candidate_xyz - patch_xyz[nearest], patch_normal[nearest]
        ) / max(median_thickness, 1.0e-6)
        compatible = np.abs(signed_shift - chosen_shift[nearest]) <= shift_tolerance
        direct[nearest] = True
        compatible_direct[nearest[compatible]] = True
        np.maximum.at(maximum_alignment, nearest, alignment)
    else:
        signed_shift = np.empty(0, dtype=np.float32)
        compatible = np.empty(0, dtype=bool)

    def nearest_distance(member: np.ndarray) -> np.ndarray:
        pixels = np.unique(nearest[member]) if len(nearest) else np.empty(0, dtype=np.int32)
        if not len(pixels):
            return np.full(len(patch_xyz), np.inf, dtype=np.float32)
        return np.min(
            np.linalg.norm(patch_xyz[:, None, :] - patch_xyz[pixels][None, :, :], axis=2),
            axis=1,
        ).astype(np.float32)

    distance = nearest_distance(np.ones(len(nearest), dtype=bool))
    compatible_distance = nearest_distance(compatible)
    return {
        "candidateDirect": direct.astype(np.uint8),
        "candidateCompatibleDirect": compatible_direct.astype(np.uint8),
        "candidateNearestDistanceVoxels": distance,
        "candidateCompatibleNearestDistanceVoxels": compatible_distance,
        "candidateNearby": (distance <= radius_voxels).astype(np.uint8),
        "candidateCompatibleNearby": (
            compatible_distance <= radius_voxels
        ).astype(np.uint8),
        "candidateMaximumAlignment": maximum_alignment,
        "candidateSignedShiftThicknesses": signed_shift.astype(np.float32),
        "candidateDepthCompatible": compatible.astype(np.uint8),
    }


def _failure_class(
    coherent_support: float,
    candidate_coverage: float,
    *,
    minimum_coherent_support: float,
    minimum_candidate_coverage: float,
) -> str:
    if coherent_support < minimum_coherent_support:
        return "ct-or-model-ambiguous"
    if candidate_coverage < minimum_candidate_coverage:
        return "ribbon-hypothesis-limited"
    return "assignment-or-topology-limited"


def build_physical_ribbon_depth_fields(
    holes: Mapping[str, np.ndarray],
    source: VolumeSource,
    *,
    settings: PhysicalRibbonDepthFieldSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    shifts = settings.shifts()
    depth = np.asarray(settings.profile_depth_fractions, dtype=np.float32)
    volume = source.memmap()
    scored_loop = np.asarray(holes["scoredLoopIndex"], dtype=np.int32)
    patch_offset = np.asarray(holes["patchOffset"], dtype=np.int64)
    patch_uv_all = np.asarray(holes["patchUV"], dtype=np.float32)
    patch_xyz_all = np.asarray(holes["patchXYZ"], dtype=np.float32)
    patch_normal_all = np.asarray(holes["patchNormalXYZ"], dtype=np.float32)
    chart_uv = np.asarray(holes["chartUV"], dtype=np.float32)
    context_profile = np.asarray(holes["contextMedianProfile"], dtype=np.float32)
    intensity_scale = np.asarray(holes["localIntensityScale"], dtype=np.float32)
    context_score = np.asarray(holes["contextPhysicalScore"], dtype=np.float32)
    candidate_bank_used = bool(
        np.asarray(
            holes.get("candidateBankUsed", np.ones(1, dtype=np.uint8)),
            dtype=np.uint8,
        )[0]
    )

    coordinate_values: list[np.ndarray] = []
    edge_first_values: list[np.ndarray] = []
    edge_second_values: list[np.ndarray] = []
    profile_values: list[np.ndarray] = []
    physical_values: list[np.ndarray] = []
    correlation_values: list[np.ndarray] = []
    unary_values: list[np.ndarray] = []
    independent_values: list[np.ndarray] = []
    collective_values: list[np.ndarray] = []
    chosen_physical_values: list[np.ndarray] = []
    chosen_correlation_values: list[np.ndarray] = []
    chosen_margin_values: list[np.ndarray] = []
    chosen_intensity_values: list[np.ndarray] = []
    support_values: list[np.ndarray] = []
    boundary_distance_values: list[np.ndarray] = []
    candidate_direct_values: list[np.ndarray] = []
    compatible_direct_values: list[np.ndarray] = []
    candidate_distance_values: list[np.ndarray] = []
    compatible_distance_values: list[np.ndarray] = []
    candidate_nearby_values: list[np.ndarray] = []
    compatible_nearby_values: list[np.ndarray] = []
    candidate_alignment_values: list[np.ndarray] = []
    candidate_shift_values: list[np.ndarray] = []
    candidate_compatible_values: list[np.ndarray] = []
    candidate_depth_offset = [0]
    records: list[dict[str, Any]] = []
    global_pixel_offset = 0
    for row, loop_value in enumerate(scored_loop):
        loop = int(loop_value)
        start, stop = int(patch_offset[row]), int(patch_offset[row + 1])
        patch_uv = patch_uv_all[start:stop]
        patch_xyz = patch_xyz_all[start:stop]
        patch_normal = patch_normal_all[start:stop]
        step = float(holes["rasterStepVoxels"][row])
        thickness = float(holes["loopMedianThicknessVoxels"][loop])
        coordinates = _grid_coordinates(patch_uv, step)
        edge_first, edge_second = _grid_edges(coordinates)
        boundary = _loop_vertices(holes, loop)
        boundary_distance = _point_segment_distance(patch_uv, chart_uv[boundary])
        profiles = _sample_normal_profiles(
            source,
            volume,
            patch_xyz,
            patch_normal,
            np.full(len(patch_xyz), thickness, dtype=np.float32),
            depth,
            shifts,
        ).transpose(1, 0, 2)
        physical, correlation = _profile_fields(
            profiles,
            context_profile[row],
            depth,
            float(intensity_scale[row]),
        )
        data_unary = physical + settings.profile_correlation_weight * correlation
        data_unary = np.nan_to_num(data_unary, nan=-4.0, posinf=4.0, neginf=-4.0)
        anchor_strength = settings.boundary_anchor_weight * np.exp(
            -boundary_distance
            / max(step * settings.boundary_anchor_decay_raster_steps, 1.0e-6)
        )
        anchor_penalty = np.minimum(
            np.abs(shifts)[None, :], settings.smoothness_truncation_thicknesses
        )
        unary = data_unary - anchor_strength[:, None] * anchor_penalty
        independent = np.argmax(unary, axis=1).astype(np.int16)
        collective, solver = solve_collective_depth_labels(
            unary,
            coordinates,
            shifts,
            smoothness_weight=settings.spatial_smoothness_weight,
            truncation_thicknesses=settings.smoothness_truncation_thicknesses,
            maximum_sweeps=settings.maximum_solver_sweeps,
        )
        pixel = np.arange(len(patch_xyz))
        chosen_physical = physical[pixel, collective]
        chosen_correlation = correlation[pixel, collective]
        chosen_unary = data_unary[pixel, collective]
        far_margin = np.full(len(patch_xyz), np.nan, dtype=np.float32)
        for pixel_row, label in enumerate(collective):
            competitor = np.abs(shifts - shifts[label]) >= settings.competing_label_separation_thicknesses
            if np.any(competitor):
                far_margin[pixel_row] = float(
                    chosen_unary[pixel_row] - np.max(data_unary[pixel_row, competitor])
                )
        center_depth = int(np.argmin(np.abs(depth)))
        chosen_intensity = profiles[pixel, collective, center_depth]
        supported = (
            (chosen_physical >= settings.minimum_context_contrast_fraction * max(float(context_score[row]), 0.10))
            & (chosen_correlation >= settings.minimum_profile_correlation)
            & np.isfinite(chosen_intensity)
        )
        coverage = _coverage_fields(
            holes,
            row,
            patch_xyz,
            patch_normal,
            shifts[collective],
            thickness,
            float(holes["loopMeanBoundaryEdgeVoxels"][loop]),
            settings.candidate_shift_tolerance_thicknesses,
        )
        coherent_fraction = _coherent_supported_fraction(
            supported, edge_first, edge_second
        )
        support_fraction = float(np.mean(supported)) if len(supported) else 0.0
        compatible_coverage = float(np.mean(coverage["candidateCompatibleNearby"]))
        direct_coverage = float(np.mean(coverage["candidateCompatibleDirect"]))
        failure = (
            _failure_class(
                coherent_fraction,
                compatible_coverage,
                minimum_coherent_support=(
                    settings.minimum_coherent_support_fraction
                ),
                minimum_candidate_coverage=(
                    settings.minimum_candidate_coverage_fraction
                ),
            )
            if candidate_bank_used
            else (
                "surface-completion-ready"
                if coherent_fraction
                >= settings.minimum_coherent_support_fraction
                else "ct-or-model-ambiguous"
            )
        )
        chosen_shift = shifts[collective]
        edge_jump = (
            np.abs(chosen_shift[edge_first] - chosen_shift[edge_second])
            if len(edge_first)
            else np.empty(0, dtype=np.float32)
        )
        records.append(
            {
                "holeRow": row,
                "loopIndex": loop,
                "component": int(holes["loopTopologyComponent"][loop]),
                "patchPixelCount": int(len(patch_xyz)),
                "candidateCount": int(
                    holes["patchCandidateOffset"][row + 1]
                    - holes["patchCandidateOffset"][row]
                ),
                "ctSupportedFraction": round(support_fraction, 6),
                "largestCtSupportedRegionFractionOfPatch": round(
                    coherent_fraction, 6
                ),
                "candidateDirectCoverageFraction": round(direct_coverage, 6),
                "candidateNearbyCoverageFraction": round(
                    float(np.mean(coverage["candidateNearby"])), 6
                ),
                "depthCompatibleCandidateNearbyCoverageFraction": round(
                    compatible_coverage, 6
                ),
                "ctSupportedButCandidateUncoveredFraction": round(
                    float(
                        np.mean(
                            supported
                            & ~(coverage["candidateCompatibleNearby"] > 0)
                        )
                    ),
                    6,
                ),
                "medianCollectiveShiftThicknesses": round(
                    float(np.median(chosen_shift)), 6
                ),
                "collectiveShiftIqrThicknesses": round(
                    float(np.percentile(chosen_shift, 75) - np.percentile(chosen_shift, 25)),
                    6,
                ),
                "maximumNeighborShiftJumpThicknesses": round(
                    float(np.max(edge_jump)) if len(edge_jump) else 0.0, 6
                ),
                "medianPhysicalScore": round(float(np.median(chosen_physical)), 6),
                "medianProfileCorrelation": round(float(np.median(chosen_correlation)), 6),
                "medianFarLayerMargin": round(float(np.nanmedian(far_margin)), 6),
                "failureClass": failure,
                "solver": solver,
            }
        )
        coordinate_values.append(coordinates)
        edge_first_values.append(edge_first + global_pixel_offset)
        edge_second_values.append(edge_second + global_pixel_offset)
        profile_values.append(profiles.astype(np.float32))
        physical_values.append(physical)
        correlation_values.append(correlation)
        unary_values.append(unary.astype(np.float32))
        independent_values.append(independent)
        collective_values.append(collective)
        chosen_physical_values.append(chosen_physical.astype(np.float32))
        chosen_correlation_values.append(chosen_correlation.astype(np.float32))
        chosen_margin_values.append(far_margin)
        chosen_intensity_values.append(chosen_intensity.astype(np.float32))
        support_values.append(supported.astype(np.uint8))
        boundary_distance_values.append(boundary_distance)
        candidate_direct_values.append(coverage["candidateDirect"])
        compatible_direct_values.append(coverage["candidateCompatibleDirect"])
        candidate_distance_values.append(coverage["candidateNearestDistanceVoxels"])
        compatible_distance_values.append(
            coverage["candidateCompatibleNearestDistanceVoxels"]
        )
        candidate_nearby_values.append(coverage["candidateNearby"])
        compatible_nearby_values.append(coverage["candidateCompatibleNearby"])
        candidate_alignment_values.append(coverage["candidateMaximumAlignment"])
        candidate_shift_values.append(coverage["candidateSignedShiftThicknesses"])
        candidate_compatible_values.append(coverage["candidateDepthCompatible"])
        candidate_depth_offset.append(
            candidate_depth_offset[-1]
            + len(coverage["candidateSignedShiftThicknesses"])
        )
        global_pixel_offset += len(patch_xyz)

    def joined(values: Sequence[np.ndarray], shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        return np.concatenate(values).astype(dtype) if values else np.empty(shape, dtype=dtype)

    pixel_count = len(patch_uv_all)
    label_count = len(shifts)
    profile_count = len(depth)
    arrays = {
        "holePatchOffset": patch_offset.astype(np.int64),
        "holeLoopIndex": scored_loop.astype(np.int32),
        "patchUV": patch_uv_all,
        "patchXYZ": patch_xyz_all,
        "patchNormalXYZ": patch_normal_all,
        "patchGridCoordinateUV": joined(coordinate_values, (0, 2), np.int32),
        "patchEdgeFirstPixel": joined(edge_first_values, (0,), np.int32),
        "patchEdgeSecondPixel": joined(edge_second_values, (0,), np.int32),
        "shiftThicknesses": shifts,
        "profileDepthFractions": depth,
        "pixelNormalProfile": joined(
            profile_values, (0, label_count, profile_count), np.float32
        ).reshape((pixel_count, label_count, profile_count)),
        "pixelPhysicalScore": joined(
            physical_values, (0, label_count), np.float32
        ).reshape((pixel_count, label_count)),
        "pixelProfileCorrelation": joined(
            correlation_values, (0, label_count), np.float32
        ).reshape((pixel_count, label_count)),
        "pixelUnaryScore": joined(
            unary_values, (0, label_count), np.float32
        ).reshape((pixel_count, label_count)),
        "pixelIndependentLabel": joined(independent_values, (0,), np.int16),
        "pixelCollectiveLabel": joined(collective_values, (0,), np.int16),
        "pixelCollectivePhysicalScore": joined(
            chosen_physical_values, (0,), np.float32
        ),
        "pixelCollectiveProfileCorrelation": joined(
            chosen_correlation_values, (0,), np.float32
        ),
        "pixelCollectiveFarLayerMargin": joined(
            chosen_margin_values, (0,), np.float32
        ),
        "pixelCollectiveCenterIntensity": joined(
            chosen_intensity_values, (0,), np.float32
        ),
        "pixelCtSupported": joined(support_values, (0,), np.uint8),
        "pixelBoundaryDistanceVoxels": joined(
            boundary_distance_values, (0,), np.float32
        ),
        "pixelCandidateDirect": joined(candidate_direct_values, (0,), np.uint8),
        "pixelCandidateDepthCompatibleDirect": joined(
            compatible_direct_values, (0,), np.uint8
        ),
        "pixelCandidateNearestDistanceVoxels": joined(
            candidate_distance_values, (0,), np.float32
        ),
        "pixelCandidateDepthCompatibleNearestDistanceVoxels": joined(
            compatible_distance_values, (0,), np.float32
        ),
        "pixelCandidateNearby": joined(candidate_nearby_values, (0,), np.uint8),
        "pixelCandidateDepthCompatibleNearby": joined(
            compatible_nearby_values, (0,), np.uint8
        ),
        "pixelCandidateMaximumAlignment": joined(
            candidate_alignment_values, (0,), np.float32
        ),
        "candidateDepthOffset": np.asarray(candidate_depth_offset, dtype=np.int64),
        "candidateSignedShiftThicknesses": joined(
            candidate_shift_values, (0,), np.float32
        ),
        "candidateDepthCompatible": joined(
            candidate_compatible_values, (0,), np.uint8
        ),
    }
    class_count: dict[str, int] = {}
    for record in records:
        name = str(record["failureClass"])
        class_count[name] = class_count.get(name, 0) + 1
    return arrays, records, {
        "holeCount": len(records),
        "patchPixelCount": int(pixel_count),
        "profileSampleCount": int(pixel_count * label_count * profile_count),
        "ctSupportedPixelCount": int(np.count_nonzero(arrays["pixelCtSupported"])),
        "failureClassCount": class_count,
        "candidateBankUsed": candidate_bank_used,
        "singlePixelGrowth": False,
        "selectionMutated": False,
        "identityLabelsUsed": False,
    }


def _color_map(
    values: np.ndarray,
    coordinates: np.ndarray,
    color: np.ndarray,
    *,
    background: tuple[int, int, int] = (12, 18, 24),
) -> np.ndarray:
    shape = np.max(coordinates, axis=0) + 1
    image = np.full((int(shape[1]), int(shape[0]), 3), background, dtype=np.uint8)
    image[coordinates[:, 1], coordinates[:, 0]] = color
    return image


def _fit_map(image: np.ndarray, width: int, height: int) -> np.ndarray:
    scale = min(width / max(image.shape[1], 1), height / max(image.shape[0], 1))
    target_width = max(int(round(image.shape[1] * scale)), 1)
    target_height = max(int(round(image.shape[0] * scale)), 1)
    row = np.minimum(
        (np.arange(target_height) * image.shape[0] / target_height).astype(int),
        image.shape[0] - 1,
    )
    column = np.minimum(
        (np.arange(target_width) * image.shape[1] / target_width).astype(int),
        image.shape[1] - 1,
    )
    return image[row[:, None], column[None, :]]


def write_depth_field_montage(
    arrays: Mapping[str, np.ndarray],
    records: Sequence[Mapping[str, Any]],
    path: str | Path,
    *,
    maximum_holes: int,
) -> Path:
    count = min(len(records), maximum_holes)
    panel_height, width = 246, 1120
    canvas = np.full((max(count, 1) * panel_height, width, 3), (7, 11, 16), dtype=np.uint8)
    if not count:
        _draw_text(canvas, 18, 18, "NO DEPTH FIELDS", (224, 231, 239), scale=2)
    offset = np.asarray(arrays["holePatchOffset"], dtype=np.int64)
    coordinates = np.asarray(arrays["patchGridCoordinateUV"], dtype=np.int32)
    shifts = np.asarray(arrays["shiftThicknesses"], dtype=np.float32)
    labels = np.asarray(arrays["pixelCollectiveLabel"], dtype=np.int16)
    intensity = np.asarray(arrays["pixelCollectiveCenterIntensity"], dtype=np.float32)
    physical = np.asarray(arrays["pixelCollectivePhysicalScore"], dtype=np.float32)
    support = np.asarray(arrays["pixelCtSupported"], dtype=np.uint8) > 0
    candidate = np.asarray(
        arrays["pixelCandidateDepthCompatibleNearby"], dtype=np.uint8
    ) > 0
    headings = ("ACTUAL CT", "DEPTH SHIFT", "PHYSICAL SCORE", "CT / BANK")
    for row in range(count):
        start, stop = int(offset[row]), int(offset[row + 1])
        coord = coordinates[start:stop]
        local_intensity = intensity[start:stop]
        finite = local_intensity[np.isfinite(local_intensity)]
        low, high = (
            (float(np.percentile(finite, 5)), float(np.percentile(finite, 95)))
            if len(finite)
            else (0.0, 1.0)
        )
        gray = np.clip((local_intensity - low) / max(high - low, 1.0), 0.0, 1.0)
        actual_color = np.repeat(np.rint(255.0 * gray)[:, None], 3, axis=1).astype(np.uint8)
        local_shift = shifts[labels[start:stop]]
        scale = max(float(np.max(np.abs(shifts))), 1.0e-6)
        shift_color = np.column_stack(
            (
                np.rint(44 + 211 * np.clip(local_shift / scale, 0.0, 1.0)),
                np.rint(226 - 150 * np.abs(local_shift) / scale),
                np.rint(44 + 211 * np.clip(-local_shift / scale, 0.0, 1.0)),
            )
        ).astype(np.uint8)
        local_physical = physical[start:stop]
        score_low = float(np.percentile(local_physical, 5))
        score_high = float(np.percentile(local_physical, 95))
        normalized = np.clip(
            (local_physical - score_low) / max(score_high - score_low, 1.0e-6),
            0.0,
            1.0,
        )
        score_color = np.column_stack(
            (
                np.rint(30 + 225 * normalized),
                np.rint(50 + 190 * normalized),
                np.rint(100 - 70 * normalized),
            )
        ).astype(np.uint8)
        local_support = support[start:stop]
        local_candidate = candidate[start:stop]
        coverage_color = np.full((len(coord), 3), (30, 40, 49), dtype=np.uint8)
        coverage_color[local_candidate & ~local_support] = (205, 72, 191)
        coverage_color[local_support & ~local_candidate] = (255, 190, 54)
        coverage_color[local_support & local_candidate] = (64, 220, 143)
        maps = (
            _color_map(local_intensity, coord, actual_color),
            _color_map(local_shift, coord, shift_color),
            _color_map(local_physical, coord, score_color),
            _color_map(local_support, coord, coverage_color),
        )
        y_base = row * panel_height
        record = records[row]
        _draw_text(
            canvas,
            10,
            y_base + 8,
            (
                f"H {record['holeRow']} C {record['component']} P {record['patchPixelCount']} "
                f"CT {record['ctSupportedFraction']:.2f} BANK {record['depthCompatibleCandidateNearbyCoverageFraction']:.2f}"
            ),
            (224, 231, 239),
            scale=2,
        )
        _draw_text(
            canvas,
            10,
            y_base + 34,
            str(record["failureClass"]).upper(),
            (255, 190, 54),
        )
        for column, (heading, image) in enumerate(zip(headings, maps)):
            x_base = 10 + column * 276
            _draw_text(canvas, x_base, y_base + 55, heading, (165, 179, 194))
            fitted = _fit_map(image, 252, 166)
            x_value = x_base + (252 - fitted.shape[1]) // 2
            y_value = y_base + 76 + (166 - fitted.shape[0]) // 2
            canvas[
                y_value : y_value + fitted.shape[0],
                x_value : x_value + fitted.shape[1],
            ] = fitted
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_bytes(rgb_png(canvas))
    return output


def run_physical_ribbon_depth_fields(
    holes_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonDepthFieldSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonDepthFieldSettings()
    holes_path, holes_manifest = _resolve_holes_manifest(holes_root)
    holes_data_path = holes_path.parent / str(holes_manifest["data"]["path"])
    holes = _load_npz(holes_data_path, holes_manifest["data"]["sha256"])
    source_record = holes_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DEPTH_FIELD_SCHEMA,
        "version": PHYSICAL_RIBBON_DEPTH_FIELD_VERSION,
        "holes": {
            "manifestPath": str(holes_path),
            "manifestSha256": sha256_file(holes_path),
            "dataSha256": holes_manifest["data"]["sha256"],
        },
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_DEPTH_FIELD_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_DEPTH_FIELD_STEM}.npz"
    montage_path = output / "physical-ribbon-depth-fields.png"
    if not force and manifest_path.is_file() and data_path.is_file() and montage_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256") == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    arrays, records, statistics = build_physical_ribbon_depth_fields(
        holes, source, settings=resolved
    )
    analyzed = time.monotonic()
    _write_npz(data_path, arrays)
    write_depth_field_montage(
        arrays,
        records,
        montage_path,
        maximum_holes=resolved.maximum_preview_holes,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_DEPTH_FIELD_SCHEMA,
        "version": PHYSICAL_RIBBON_DEPTH_FIELD_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": holes_manifest.get("geometry", {}),
        "analysis": statistics,
        "holes": records,
        "timingSeconds": {
            "denseCtField": round(analyzed - started, 6),
            "writingAndPreview": round(finished - analyzed, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"depthFieldMontage": montage_path.name},
        "method": {
            "decisionUnit": "one complete missing-surface raster, never one pixel or cell",
            "rawCtEvidence": "dense air-material-air profiles at every raster pixel and normal-depth label",
            "optimization": "multi-start alternating exact row/column Viterbi fields with truncated physical smoothness",
            "boundaryCondition": "soft attachment to the fitted surrounding surface, strongest at the complete frontier",
            "defectHandling": "truncated depth-jump cost retains coherent delamination steps; no common layer depth is forced",
            "candidateRole": (
                "ribbon-bank hypotheses are audited after the CT field and "
                "never define its data term"
                if statistics["candidateBankUsed"]
                else "candidate-free surface completion; no ribbon bank was consulted"
            ),
            "selectionMutated": False,
            "singlePixelGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
