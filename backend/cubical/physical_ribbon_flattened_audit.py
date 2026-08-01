from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .flatten import (
    ComponentMesh,
    SurfaceChart,
    _draw_text,
    rasterize_chart,
    sample_depth_stack,
)
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_continuity import _draw_line
from .physical_ribbon_patch_holes import PhysicalRibbonPatchHoleSettings
from .physical_ribbon_patch_states import (
    PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
    _prepare_component_exact_graph,
    _reconstruct_component_graph_state,
)


PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA = (
    "pareidolia.physical-ribbon-flattened-audit"
)
PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION = 1
PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM = "physical-ribbon-flattened-audit-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonFlattenedAuditSettings:
    maximum_components: int = 8
    pixel_step_voxels: float = 0.5
    maximum_raster_pixels: int = 768
    depth_fractions: tuple[float, ...] = (-0.35, 0.0, 0.35)
    structure_window_radius_pixels: int = 4
    minimum_orientation_coherence: float = 0.15
    minimum_boundary_edge_measurements: int = 6
    maximum_median_excess_floor_degrees: float = 5.0
    maximum_control_spread_fraction: float = 0.25

    def __post_init__(self) -> None:
        if self.maximum_components < 1 or self.maximum_raster_pixels < 32:
            raise ValueError("flattened audit dimensions must be positive")
        if not math.isfinite(self.pixel_step_voxels) or self.pixel_step_voxels <= 0:
            raise ValueError("flattened audit pixel step must be positive")
        if not self.depth_fractions or any(
            not math.isfinite(value) or not -1.0 <= value <= 1.0
            for value in self.depth_fractions
        ):
            raise ValueError("depth fractions must be finite and lie in [-1, 1]")
        if self.structure_window_radius_pixels < 1:
            raise ValueError("structure window radius must be positive")
        if self.minimum_boundary_edge_measurements < 3:
            raise ValueError("texture compatibility requires several boundary edges")
        if not 0.0 <= self.minimum_orientation_coherence <= 1.0:
            raise ValueError("orientation coherence gate must lie in [0, 1]")
        if (
            not math.isfinite(self.maximum_median_excess_floor_degrees)
            or self.maximum_median_excess_floor_degrees < 0.0
        ):
            raise ValueError("texture median-excess floor must be finite and nonnegative")
        if (
            not math.isfinite(self.maximum_control_spread_fraction)
            or not 0.0 <= self.maximum_control_spread_fraction <= 1.0
        ):
            raise ValueError("texture control-spread fraction must lie in [0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _box_sum(values: np.ndarray, radius: int) -> np.ndarray:
    width = 2 * radius + 1
    padded = np.pad(
        np.asarray(values, dtype=np.float32),
        ((radius, radius), (radius, radius)),
        mode="constant",
    )
    vertical_integral = np.pad(padded, ((1, 0), (0, 0))).cumsum(axis=0)
    vertical = vertical_integral[width:] - vertical_integral[:-width]
    horizontal_integral = np.pad(vertical, ((0, 0), (1, 0))).cumsum(axis=1)
    return horizontal_integral[:, width:] - horizontal_integral[:, :-width]


def _percentiles(values: np.ndarray) -> dict[str, float | int]:
    finite = np.asarray(values)[np.isfinite(values)]
    if not len(finite):
        return {"count": 0}
    return {
        "count": int(len(finite)),
        "median": round(float(np.median(finite)), 6),
        "p90": round(float(np.percentile(finite, 90)), 6),
        "maximum": round(float(np.max(finite)), 6),
    }


def boundary_texture_compatibility(
    added_statistics: Mapping[str, float | int],
    baseline_statistics: Mapping[str, float | int],
    *,
    minimum_measurements: int,
    median_excess_floor_degrees: float,
    control_spread_fraction: float,
) -> dict[str, float | bool | None]:
    measured = (
        int(added_statistics.get("count", 0)) >= minimum_measurements
        and int(baseline_statistics.get("count", 0)) >= minimum_measurements
        and "median" in added_statistics
        and "median" in baseline_statistics
        and "p90" in baseline_statistics
    )
    if not measured:
        return {
            "measured": False,
            "compatible": None,
        }
    baseline_median = float(baseline_statistics["median"])
    baseline_spread = max(
        float(baseline_statistics["p90"]) - baseline_median,
        0.0,
    )
    allowance = max(
        median_excess_floor_degrees,
        control_spread_fraction * baseline_spread,
    )
    excess = float(added_statistics["median"]) - baseline_median
    return {
        "measured": True,
        "compatible": bool(excess <= allowance),
        "medianExcessDegrees": round(excess, 6),
        "medianExcessAllowanceDegrees": round(allowance, 6),
    }


def flattened_texture_structure(
    image: np.ndarray,
    mask: np.ndarray,
    *,
    window_radius: int,
    minimum_coherence: float,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Measure local axial texture continuity on one flattened CT plane."""

    values = np.asarray(image, dtype=np.float32)
    valid_mask = np.asarray(mask, dtype=bool)
    gradient_y = np.zeros_like(values)
    gradient_x = np.zeros_like(values)
    gradient_y[1:-1] = 0.5 * (values[2:] - values[:-2])
    gradient_x[:, 1:-1] = 0.5 * (values[:, 2:] - values[:, :-2])
    weight = valid_mask.astype(np.float32)
    count = _box_sum(weight, window_radius)
    xx = _box_sum(gradient_x * gradient_x * weight, window_radius)
    yy = _box_sum(gradient_y * gradient_y * weight, window_radius)
    xy = _box_sum(gradient_x * gradient_y * weight, window_radius)
    trace = xx + yy
    anisotropy = np.sqrt(np.maximum((xx - yy) ** 2 + 4.0 * xy**2, 0.0))
    coherence = anisotropy / np.maximum(trace, 1.0e-6)
    angle = 0.5 * np.arctan2(2.0 * xy, xx - yy)
    full_window = float((2 * window_radius + 1) ** 2)
    oriented = (
        valid_mask
        & (count >= 0.60 * full_window)
        & (trace > 1.0e-4)
        & np.isfinite(coherence)
    )
    reliable = oriented & (coherence >= minimum_coherence)
    disagreement: list[np.ndarray] = []
    for first, second in (
        ((slice(None), slice(None, -1)), (slice(None), slice(1, None))),
        ((slice(None, -1), slice(None)), (slice(1, None), slice(None))),
    ):
        pair = reliable[first] & reliable[second]
        if not np.any(pair):
            continue
        difference = 0.5 * np.degrees(
            np.arccos(
                np.clip(
                    np.cos(2.0 * (angle[first][pair] - angle[second][pair])),
                    -1.0,
                    1.0,
                )
            )
        )
        disagreement.append(difference)
    adjacent_disagreement = (
        np.concatenate(disagreement)
        if disagreement
        else np.empty(0, dtype=np.float32)
    )
    masked_values = values[valid_mask]
    contrast = (
        float(np.percentile(masked_values, 90) - np.percentile(masked_values, 10))
        if len(masked_values)
        else 0.0
    )
    reliable_fraction = float(np.count_nonzero(reliable)) / max(
        float(np.count_nonzero(valid_mask)), 1.0
    )
    median_coherence = (
        float(np.median(coherence[oriented])) if np.any(oriented) else 0.0
    )
    statistics = {
        "supportedPixelCount": int(np.count_nonzero(valid_mask)),
        "orientedPixelCount": int(np.count_nonzero(oriented)),
        "reliableOrientationPixelCount": int(np.count_nonzero(reliable)),
        "reliableOrientationFraction": round(reliable_fraction, 6),
        "medianCoherence": round(median_coherence, 6),
        "coherence": _percentiles(coherence[oriented]),
        "adjacentAxialDisagreementDegrees": _percentiles(adjacent_disagreement),
        "intensityP90MinusP10": round(contrast, 6),
        "structureScore": round(
            median_coherence * math.sqrt(max(reliable_fraction, 0.0)), 6
        ),
    }
    return {
        "angleRadians": angle.astype(np.float32),
        "coherence": coherence.astype(np.float32),
        "reliable": reliable.astype(np.uint8),
    }, statistics


def _resolve_surface_manifest(root: str | Path) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    if value.is_file():
        candidates = (value,)
    else:
        candidates = tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    required = {
        "selected",
        "component",
        "chartUV",
        "triangleFrontierIndex",
        "signedNormalXYZ",
        "midpointXYZ",
        "thicknessVoxels",
    }
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if (
            manifest.get("state") == "complete"
            and required.issubset(manifest.get("data", {}).get("fields", ()))
            and manifest.get("method", {}).get("identityLabelsUsed") is False
        ):
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError("surface root must identify exactly one flattened surface artifact")
    return matches[0]


def _load_topology(
    surface_manifest: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    reference = surface_manifest.get("identity", {}).get("topologyContinuity")
    if reference is None:
        reference = surface_manifest.get("identity", {}).get("frontier")
    if reference is None:
        raise ValueError("surface does not identify its continuation topology")
    path = Path(reference["manifestPath"])
    if sha256_file(path) != reference["manifestSha256"]:
        raise ValueError("surface topology manifest changed")
    manifest = json.loads(path.read_text())
    if manifest["data"]["sha256"] != reference["dataSha256"]:
        raise ValueError("surface topology data identity differs")
    data_path = path.parent / str(manifest["data"]["path"])
    return path, manifest, _load_npz(data_path, reference["dataSha256"])


def _added_nodes(
    surface_manifest: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
) -> np.ndarray:
    selected = np.asarray(surface["selected"], dtype=np.uint8) > 0
    proposal_offset = surface.get("proposalOffset")
    proposal_node = surface.get("proposalFrontierIndex")
    proposal_accepted = surface.get("proposalAccepted")
    added = np.zeros(len(selected), dtype=bool)
    if (
        proposal_offset is not None
        and proposal_node is not None
        and proposal_accepted is not None
    ):
        offset = np.asarray(proposal_offset, dtype=np.int64)
        node = np.asarray(proposal_node, dtype=np.int32)
        for row in np.flatnonzero(np.asarray(proposal_accepted) > 0):
            added[node[offset[row] : offset[row + 1]]] = True
        return added & selected
    configuration_reference = surface_manifest.get("identity", {}).get(
        "configuration"
    )
    if configuration_reference is None:
        return added
    path = Path(configuration_reference["manifestPath"])
    if sha256_file(path) != configuration_reference["manifestSha256"]:
        raise ValueError("baseline configuration manifest changed")
    manifest = json.loads(path.read_text())
    data_path = path.parent / str(manifest["data"]["path"])
    baseline = _load_npz(data_path, configuration_reference["dataSha256"])
    baseline_selected = np.asarray(baseline["selected"], dtype=np.uint8) > 0
    if len(baseline_selected) != len(selected):
        raise ValueError("surface and baseline frontiers differ")
    return selected & ~baseline_selected


def _normalize_rows(values: np.ndarray) -> np.ndarray:
    result = np.asarray(values, dtype=np.float32)
    return result / np.maximum(np.linalg.norm(result, axis=1, keepdims=True), 1.0e-6)


def _node_pixel(
    chart_uv: np.ndarray,
    node: int,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
) -> tuple[int, int]:
    value = (chart_uv[node] - chart_low) / pixel_step + padding
    return int(round(float(value[1]))), int(round(float(value[0])))


def _edge_orientation_disagreement(
    angle: np.ndarray,
    coherence: np.ndarray,
    chart_uv: np.ndarray,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    edge_mask: np.ndarray,
    chart_low: np.ndarray,
    pixel_step: float,
    padding: float,
    minimum_coherence: float,
    sample_radius_pixels: int = 2,
) -> dict[str, float | int]:
    def sample(y_value: int, x_value: int) -> float | None:
        y_start = max(y_value - sample_radius_pixels, 0)
        y_stop = min(y_value + sample_radius_pixels + 1, angle.shape[0])
        x_start = max(x_value - sample_radius_pixels, 0)
        x_stop = min(x_value + sample_radius_pixels + 1, angle.shape[1])
        local_coherence = coherence[y_start:y_stop, x_start:x_stop]
        local_angle = angle[y_start:y_stop, x_start:x_stop]
        member = local_coherence >= minimum_coherence
        if np.count_nonzero(member) < 3:
            return None
        weight = np.square(local_coherence[member].astype(np.float64))
        doubled = 2.0 * local_angle[member].astype(np.float64)
        vector_x = float(np.sum(weight * np.cos(doubled)))
        vector_y = float(np.sum(weight * np.sin(doubled)))
        if math.hypot(vector_x, vector_y) < 0.15 * float(np.sum(weight)):
            return None
        return 0.5 * math.atan2(vector_y, vector_x)

    values: list[float] = []
    rows, columns = angle.shape
    for first, second in zip(edge_first[edge_mask], edge_second[edge_mask]):
        first_y, first_x = _node_pixel(
            chart_uv, int(first), chart_low, pixel_step, padding
        )
        second_y, second_x = _node_pixel(
            chart_uv, int(second), chart_low, pixel_step, padding
        )
        if not (
            0 <= first_y < rows
            and 0 <= first_x < columns
            and 0 <= second_y < rows
            and 0 <= second_x < columns
        ):
            continue
        first_angle = sample(first_y, first_x)
        second_angle = sample(second_y, second_x)
        if first_angle is None or second_angle is None:
            continue
        delta = 0.5 * math.degrees(
            math.acos(
                float(
                    np.clip(
                        math.cos(
                            2.0
                            * (
                                first_angle - second_angle
                            )
                        ),
                        -1.0,
                        1.0,
                    )
                )
            )
        )
        values.append(delta)
    return _percentiles(np.asarray(values, dtype=np.float32))


def _variant_topology_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    before = record.get("before", {})
    after = record.get("after", {})
    return (
        float(
            int(before.get("macroHoleCount", 0))
            - int(after.get("macroHoleCount", 0))
        ),
        float(
            int(before.get("interiorHoleCount", 0))
            - int(after.get("interiorHoleCount", 0))
        ),
        float(
            int(before.get("triangleRegionCount", 0))
            - int(after.get("triangleRegionCount", 0))
        ),
        float(
            int(after.get("triangleCount", 0))
            - int(before.get("triangleCount", 0))
        ),
        float(record.get("triangleAreaRetention", 0.0)),
        -float(record.get("variantRank", 0)),
    )


def _rank_exact_variant_rows(
    exact_records: list[Mapping[str, Any]], maximum_count: int
) -> list[Mapping[str, Any]]:
    """Share a bounded texture budget across affected sheet components."""

    by_component: dict[int, list[Mapping[str, Any]]] = {}
    seen_rows: set[int] = set()
    for record in exact_records:
        row = int(record["patchRow"])
        if row in seen_rows:
            continue
        seen_rows.add(row)
        by_component.setdefault(int(record["priorComponent"]), []).append(record)
    for records in by_component.values():
        records.sort(key=_variant_topology_key, reverse=True)
    ranked: list[Mapping[str, Any]] = []
    depth = 0
    while len(ranked) < maximum_count:
        added = False
        for component_id in sorted(by_component):
            records = by_component[component_id]
            if depth >= len(records):
                continue
            ranked.append(records[depth])
            added = True
            if len(ranked) >= maximum_count:
                break
        if not added:
            break
        depth += 1
    return ranked


def _proposal_records_from_arrays(
    surface: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    added_offset = np.asarray(surface["patchAddedOffset"], dtype=np.int64)
    added_node = np.asarray(
        surface["patchAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(surface["patchRemovedOffset"], dtype=np.int64)
    removed_node = np.asarray(
        surface["patchRemovedFrontierIndex"], dtype=np.int32
    )
    return [
        {
            "added": added_node[
                int(added_offset[row]) : int(added_offset[row + 1])
            ],
            "removed": removed_node[
                int(removed_offset[row]) : int(removed_offset[row + 1])
            ],
        }
        for row in range(len(added_offset) - 1)
    ]


def _write_variant_surface(
    output: Path,
    patch_state_path: Path,
    surface_manifest: Mapping[str, Any],
    local_surface: Mapping[str, np.ndarray],
    component_global: np.ndarray,
    proposal: Mapping[str, Any],
    *,
    patch_row: int,
    prior_component: int,
) -> Path:
    output.mkdir(parents=True, exist_ok=True)
    data_path = output / "physical-ribbon-patch-variant-surface-v1.npz"
    manifest_path = output / "physical-ribbon-patch-variant-surface-v1.json"
    added = set(int(value) for value in proposal["added"])
    local_added = np.asarray(
        [index for index, value in enumerate(component_global) if int(value) in added],
        dtype=np.int32,
    )
    arrays = {
        name: np.asarray(value)
        for name, value in local_surface.items()
    }
    arrays["proposalOffset"] = np.asarray((0, len(local_added)), dtype=np.int64)
    arrays["proposalFrontierIndex"] = local_added
    arrays["proposalAccepted"] = np.ones(1, dtype=np.uint8)
    arrays["sourceFrontierIndex"] = np.asarray(component_global, dtype=np.int32)
    _write_npz(data_path, arrays)
    identity = {
        "patchState": {
            "manifestPath": str(patch_state_path),
            "manifestSha256": sha256_file(patch_state_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": surface_manifest["identity"][
            "topologyContinuity"
        ],
        "patchRow": int(patch_row),
        "priorComponent": int(prior_component),
    }
    payload = {
        "schema": "pareidolia.physical-ribbon-patch-variant-surface",
        "version": 1,
        "state": "complete",
        "identity": identity,
        "source": surface_manifest["source"],
        "geometry": surface_manifest["geometry"],
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "decisionUnit": "one exact-valid complete patch matching",
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return manifest_path


def _run_patch_variant_audit(
    surface_path: Path,
    surface_manifest: Mapping[str, Any],
    surface: Mapping[str, np.ndarray],
    output: Path,
    montage_path: Path,
    identity: Mapping[str, Any],
    *,
    settings: PhysicalRibbonFlattenedAuditSettings,
    force: bool,
    started: float,
) -> dict[str, Any]:
    exact_records = [
        record
        for record in surface_manifest.get("patchStates", {}).get(
            "exactPatchAudits", ()
        )
        if bool(record.get("accepted"))
    ]
    selected_records = _rank_exact_variant_rows(
        exact_records, settings.maximum_components
    )
    configuration_reference = surface_manifest["identity"]["configuration"]
    (
        _,
        _,
        configuration,
        _,
        _,
        topology,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    proposals = _proposal_records_from_arrays(surface)
    baseline_selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    baseline_component = np.asarray(configuration["component"], dtype=np.int32)
    rows_by_component: dict[int, list[int]] = {}
    for record in selected_records:
        rows_by_component.setdefault(int(record["priorComponent"]), []).append(
            int(record["patchRow"])
        )
    graph_by_component = {
        component_id: _prepare_component_exact_graph(
            component_id,
            proposals,
            rows,
            baseline_selected,
            baseline_component,
            topology,
        )
        for component_id, rows in sorted(rows_by_component.items())
    }
    variant_records: list[dict[str, Any]] = []
    artifact_records: list[dict[str, Any]] = []
    variant_root = output / "variants"
    for record in selected_records:
        row = int(record["patchRow"])
        component_id = int(record["priorComponent"])
        ok, local_surface, _, component_global = _reconstruct_component_graph_state(
            graph_by_component[component_id],
            proposals[row],
            ribbon,
            topology,
            settings=PhysicalRibbonPatchHoleSettings(),
        )
        if not ok or local_surface is None:
            raise RuntimeError("exact-valid patch variant no longer reconstructs")
        local_root = variant_root / f"patch-{row:06d}"
        local_manifest_path = _write_variant_surface(
            local_root / "surface",
            surface_path,
            surface_manifest,
            local_surface,
            component_global,
            proposals[row],
            patch_row=row,
            prior_component=component_id,
        )
        audit_root = local_root / "audit"
        local_audit = run_physical_ribbon_flattened_audit(
            local_manifest_path,
            audit_root,
            settings=settings,
            force=force,
        )
        components = local_audit.get("audit", {}).get("components", ())
        if len(components) != 1:
            raise RuntimeError("patch variant audit did not yield one component")
        variant = {
            **components[0],
            "componentId": component_id,
            "patchRow": row,
            "variantRank": int(record.get("variantRank", 0)),
            "variantProfile": str(record.get("variantProfile", "unknown")),
            "objectiveGain": round(
                float(np.asarray(surface["patchObjectiveGain"])[row]), 6
            ),
            "exactAudit": dict(record),
        }
        if "patchScopeKind" in surface:
            variant["scopeKind"] = str(surface["patchScopeKind"][row])
            variant["scopeHoleRow"] = int(surface["patchScopeHoleRow"][row])
        variant_records.append(variant)
        artifact_records.append(
            {
                "patchRow": row,
                "surfaceManifest": str(local_manifest_path),
                "auditManifest": str(
                    audit_root
                    / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
                ),
                "montage": str(audit_root / "physical-ribbon-flattened-audit.png"),
            }
        )

    canvas = np.full(
        (max(80 + 34 * len(variant_records), 120), 1100, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    _draw_text(
        canvas,
        16,
        14,
        "EXACT PATCH VARIANTS / FLATTENED NATIVE CT",
        (224, 231, 239),
        scale=2,
    )
    for index, variant in enumerate(variant_records):
        compatible = variant["boundaryTextureCompatible"]
        color = (
            (102, 227, 159)
            if compatible is True
            else (255, 105, 120)
            if compatible is False
            else (174, 184, 199)
        )
        depth_count = int(variant["boundaryTextureCompatibleDepthCount"])
        _draw_text(
            canvas,
            18,
            64 + 34 * index,
            (
                f"P {variant['patchRow']:>3}  C {variant['componentId']:>4}  "
                f"V {variant['variantRank']:>2}  {variant['variantProfile']:<20} "
                f"CT {depth_count}/{variant['boundaryTextureMeasuredDepthCount']}"
            ),
            color,
        )
    montage_path.write_bytes(rgb_png(canvas))
    finished = time.monotonic()
    compatible_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is True
    ]
    incompatible_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is False
    ]
    unmeasured_rows = [
        int(record["patchRow"])
        for record in variant_records
        if record["boundaryTextureCompatible"] is None
    ]
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "state": "complete",
        "identity": dict(identity),
        "source": surface_manifest["source"],
        "audit": {
            "mode": "all exact-valid complete patch variants",
            "exactGeometryVariantCount": len(exact_records),
            "flattenedVariantCount": len(variant_records),
            "omittedVariantCount": len(exact_records) - len(variant_records),
            "boundaryTextureCompatibleVariantCount": len(compatible_rows),
            "boundaryTextureIncompatibleVariantCount": len(incompatible_rows),
            "boundaryTextureUnmeasuredVariantCount": len(unmeasured_rows),
            "boundaryTextureCompatiblePatchRows": compatible_rows,
            "boundaryTextureIncompatiblePatchRows": incompatible_rows,
            "boundaryTextureUnmeasuredPatchRows": unmeasured_rows,
            "variants": variant_records,
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "artifacts": {
            "montage": montage_path.name,
            "variantAudits": artifact_records,
        },
        "method": {
            "measurement": (
                "every retained exact-valid complete matching is rebuilt, "
                "flattened, and sampled from native CT at fixed depths"
            ),
            "compatibility": (
                "variant boundaries are compared with same-surface control "
                "edges before one compatible state is chosen per component"
            ),
            "acceptanceMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(
        output / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json", payload
    )
    return payload


def run_physical_ribbon_flattened_audit(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonFlattenedAuditSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonFlattenedAuditSettings()
    surface_path, surface_manifest = _resolve_surface_manifest(surface_root)
    surface_data_path = surface_path.parent / str(surface_manifest["data"]["path"])
    surface = _load_npz(surface_data_path, surface_manifest["data"]["sha256"])
    topology_path, topology_manifest, topology = _load_topology(surface_manifest)
    source_record = surface_manifest["source"]
    source = VolumeSource.open(source_record["path"], source_record.get("metadataPath"))
    identity = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "surface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "topologyContinuity": {
            "manifestPath": str(topology_path),
            "manifestSha256": sha256_file(topology_path),
            "dataSha256": topology_manifest["data"]["sha256"],
        },
        "source": source.source_identity,
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
    montage_path = output / "physical-ribbon-flattened-audit.png"
    if not force and manifest_path.is_file() and montage_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if cached.get("identity", {}).get("identitySha256") == identity[
            "identitySha256"
        ]:
            return cached

    started = time.monotonic()
    if (
        surface_manifest.get("schema") == PHYSICAL_RIBBON_PATCH_STATE_SCHEMA
        and any(
            bool(record.get("accepted"))
            for record in surface_manifest.get("patchStates", {}).get(
                "exactPatchAudits", ()
            )
        )
    ):
        return _run_patch_variant_audit(
            surface_path,
            surface_manifest,
            surface,
            output,
            montage_path,
            identity,
            settings=resolved,
            force=force,
            started=started,
        )

    selected = np.asarray(surface["selected"], dtype=np.uint8) > 0
    component = np.asarray(surface["component"], dtype=np.int32)
    added = _added_nodes(surface_manifest, surface)
    triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    chart_uv = np.asarray(surface["chartUV"], dtype=np.float32)
    midpoint = np.asarray(surface["midpointXYZ"], dtype=np.float32)
    normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float32)
    thickness = np.asarray(surface["thicknessVoxels"], dtype=np.float32)
    changed_component, changed_count = np.unique(
        component[added & (component >= 0)], return_counts=True
    )
    if not len(changed_component):
        labels, counts = np.unique(component[selected], return_counts=True)
        order = np.argsort(-counts)
        ranked = labels[order[: resolved.maximum_components]]
    else:
        size = np.bincount(component[selected])
        order = sorted(
            range(len(changed_component)),
            key=lambda index: (
                -int(changed_count[index]),
                -int(size[changed_component[index]]),
                int(changed_component[index]),
            ),
        )
        ranked = changed_component[order[: resolved.maximum_components]]

    columns = 2
    tile_width, tile_height = 650, 560
    canvas_rows = max(int(math.ceil(len(ranked) / columns)), 1)
    canvas = np.full(
        (canvas_rows * tile_height, columns * tile_width, 3),
        (7, 11, 16),
        dtype=np.uint8,
    )
    records: list[dict[str, Any]] = []
    displayed = 0
    for component_id_value in ranked:
        component_id = int(component_id_value)
        component_triangle = triangle[
            np.all(component[triangle] == component_id, axis=1)
        ]
        if not len(component_triangle):
            continue
        vertex = np.unique(component_triangle)
        if not np.all(np.isfinite(chart_uv[vertex])):
            continue
        local_triangle = np.searchsorted(vertex, component_triangle).astype(np.int32)
        triangle_normal = _normalize_rows(
            np.mean(normal[component_triangle], axis=1)
        )
        mesh = ComponentMesh(
            component_id=component_id,
            patch_ids=(component_id,),
            vertex_xyz=midpoint[vertex].astype(np.float64),
            polygons=(),
            polygon_patch_ids=np.empty(0, dtype=np.uint64),
            triangles=local_triangle,
            triangle_patch_ids=np.full(
                len(local_triangle), component_id + 1, dtype=np.uint64
            ),
            triangle_normal_xyz=triangle_normal.astype(np.float64),
            statistics={},
        )
        chart = SurfaceChart(
            uv=chart_uv[vertex].astype(np.float64),
            anchor_vertices=(),
            statistics={},
        )
        raster = rasterize_chart(
            mesh,
            chart,
            pixel_step_voxels=resolved.pixel_step_voxels,
            maximum_pixels=resolved.maximum_raster_pixels,
            padding_pixels=5,
        )
        median_thickness = float(np.median(thickness[vertex]))
        depth_offsets = tuple(
            float(value) * median_thickness for value in resolved.depth_fractions
        )
        stack, sampling_stats = sample_depth_stack(source, raster, depth_offsets)
        depth_records: list[dict[str, Any]] = []
        structures: list[dict[str, np.ndarray]] = []
        chart_low = np.min(chart.uv, axis=0)
        component_added = added & (component == component_id)
        mesh_edge = np.sort(
            np.concatenate(
                (
                    component_triangle[:, (0, 1)],
                    component_triangle[:, (1, 2)],
                    component_triangle[:, (2, 0)],
                ),
                axis=0,
            ),
            axis=1,
        )
        mesh_edge = np.unique(mesh_edge, axis=0)
        mesh_first = mesh_edge[:, 0]
        mesh_second = mesh_edge[:, 1]
        boundary_edge = (
            added[mesh_first] ^ added[mesh_second]
        )
        interior_added_edge = (
            added[mesh_first] & added[mesh_second]
        )
        baseline_edge = ~added[mesh_first] & ~added[mesh_second]
        for depth_index, (depth_fraction, depth_offset) in enumerate(
            zip(resolved.depth_fractions, depth_offsets)
        ):
            structure, structure_stats = flattened_texture_structure(
                stack[depth_index],
                raster.mask,
                window_radius=resolved.structure_window_radius_pixels,
                minimum_coherence=resolved.minimum_orientation_coherence,
            )
            structure_stats["depthFraction"] = round(float(depth_fraction), 6)
            structure_stats["depthOffsetVoxels"] = round(float(depth_offset), 6)
            structure_stats["addedBoundaryAxialDisagreementDegrees"] = (
                _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    boundary_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
            )
            structure_stats["addedInteriorAxialDisagreementDegrees"] = (
                _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    interior_added_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
            )
            structure_stats["baselineMeshAxialDisagreementDegrees"] = (
                _edge_orientation_disagreement(
                    structure["angleRadians"],
                    structure["coherence"],
                    chart_uv,
                    mesh_first,
                    mesh_second,
                    baseline_edge,
                    chart_low,
                    raster.pixel_step_voxels,
                    5.0,
                    resolved.minimum_orientation_coherence,
                )
            )
            structures.append(structure)
            depth_records.append(structure_stats)
        compatible_depth_count = 0
        measured_depth_count = 0
        for record in depth_records:
            added_statistics = record[
                "addedBoundaryAxialDisagreementDegrees"
            ]
            baseline_statistics = record[
                "baselineMeshAxialDisagreementDegrees"
            ]
            compatibility = boundary_texture_compatibility(
                added_statistics,
                baseline_statistics,
                minimum_measurements=resolved.minimum_boundary_edge_measurements,
                median_excess_floor_degrees=(
                    resolved.maximum_median_excess_floor_degrees
                ),
                control_spread_fraction=resolved.maximum_control_spread_fraction,
            )
            if not compatibility["measured"]:
                record["boundaryTextureCompatibilityMeasured"] = False
                record["boundaryTextureCompatible"] = None
                continue
            measured_depth_count += 1
            compatible = bool(compatibility["compatible"])
            compatible_depth_count += int(compatible)
            record["boundaryTextureCompatibilityMeasured"] = True
            record["boundaryTextureMedianExcessDegrees"] = compatibility[
                "medianExcessDegrees"
            ]
            record["boundaryTextureMedianExcessAllowanceDegrees"] = (
                compatibility["medianExcessAllowanceDegrees"]
            )
            record["boundaryTextureCompatible"] = bool(compatible)
        texture_compatible = (
            bool(compatible_depth_count)
            if measured_depth_count
            else None
        )
        chosen_depth = max(
            range(len(depth_records)),
            key=lambda index: (
                float(depth_records[index]["structureScore"]),
                float(depth_records[index]["intensityP90MinusP10"]),
                -abs(float(resolved.depth_fractions[index])),
            ),
        )
        plane = stack[chosen_depth].astype(np.float32)
        values = plane[raster.mask]
        low, high = (
            np.percentile(values, (1.0, 99.0)) if len(values) else (0.0, 1.0)
        )
        normalized = np.clip(
            (plane - float(low)) / max(float(high - low), 1.0), 0.0, 1.0
        )
        grayscale = np.rint(12.0 + 243.0 * normalized).astype(np.uint8)
        grayscale[~raster.mask] = 0
        image = np.repeat(grayscale[:, :, None], 3, axis=2)
        image[raster.overlap_mask] = (255, 58, 58)
        for node in np.flatnonzero(component_added):
            y_value, x_value = _node_pixel(
                chart_uv,
                int(node),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            if not (
                1 <= y_value < image.shape[0] - 1
                and 1 <= x_value < image.shape[1] - 1
            ):
                continue
            image[y_value - 1 : y_value + 2, x_value] = (255, 83, 201)
            image[y_value, x_value - 1 : x_value + 2] = (255, 83, 201)
        for first, second in zip(mesh_first[boundary_edge], mesh_second[boundary_edge]):
            first_y, first_x = _node_pixel(
                chart_uv,
                int(first),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            second_y, second_x = _node_pixel(
                chart_uv,
                int(second),
                chart_low,
                raster.pixel_step_voxels,
                5.0,
            )
            _draw_line(
                image,
                np.asarray((first_x, first_y), dtype=np.float32),
                np.asarray((second_x, second_y), dtype=np.float32),
                (255, 184, 72),
            )
        scale = min(
            620.0 / max(image.shape[1], 1),
            490.0 / max(image.shape[0], 1),
        )
        target_width = max(int(round(image.shape[1] * scale)), 1)
        target_height = max(int(round(image.shape[0] * scale)), 1)
        row_index = np.minimum(
            (np.arange(target_height) * image.shape[0] / target_height).astype(int),
            image.shape[0] - 1,
        )
        column_index = np.minimum(
            (np.arange(target_width) * image.shape[1] / target_width).astype(int),
            image.shape[1] - 1,
        )
        fitted = image[row_index[:, None], column_index[None, :]]
        tile_x = (displayed % columns) * tile_width
        tile_y = (displayed // columns) * tile_height
        image_x = tile_x + (tile_width - target_width) // 2
        image_y = tile_y + 54 + (490 - target_height) // 2
        canvas[
            image_y : image_y + target_height,
            image_x : image_x + target_width,
        ] = fitted
        canvas[tile_y : tile_y + 3, tile_x : tile_x + tile_width] = (
            255,
            83,
            201,
        )
        chosen_record = depth_records[chosen_depth]
        _draw_text(
            canvas,
            tile_x + 10,
            tile_y + 12,
            (
                f"C {component_id} N {len(vertex)} +{np.count_nonzero(component_added)} "
                f"D {resolved.depth_fractions[chosen_depth]:+.2f} "
                f"Q {chosen_record['medianCoherence']:.2f}"
            ),
            (224, 231, 239),
        )
        records.append(
            {
                "componentId": component_id,
                "ribbonCount": int(np.count_nonzero(component == component_id)),
                "surfaceVertexCount": int(len(vertex)),
                "triangleCount": int(len(component_triangle)),
                "addedRibbonCount": int(np.count_nonzero(component_added)),
                "addedBoundaryEdgeCount": int(np.count_nonzero(boundary_edge)),
                "addedInteriorEdgeCount": int(np.count_nonzero(interior_added_edge)),
                "chosenDepthIndex": int(chosen_depth),
                "boundaryTextureMeasuredDepthCount": measured_depth_count,
                "boundaryTextureCompatibleDepthCount": compatible_depth_count,
                "boundaryTextureCompatible": texture_compatible,
                "raster": raster.statistics,
                "sampling": sampling_stats,
                "depths": depth_records,
            }
        )
        displayed += 1
    if not displayed:
        _draw_text(canvas, 20, 20, "NO ELIGIBLE CHANGED SURFACE", (224, 231, 239), scale=2)
    montage_path.write_bytes(rgb_png(canvas))
    finished = time.monotonic()
    compatible_component = [
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is True
    ]
    incompatible_component = [
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is False
    ]
    unmeasured_component = [
        int(record["componentId"])
        for record in records
        if record["boundaryTextureCompatible"] is None
    ]
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
        "version": PHYSICAL_RIBBON_FLATTENED_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "audit": {
            "addedRibbonCount": int(np.count_nonzero(added)),
            "changedComponentCount": int(len(changed_component)),
            "flattenedComponentCount": int(displayed),
            "boundaryTextureCompatibleComponentCount": len(
                compatible_component
            ),
            "boundaryTextureIncompatibleComponentCount": len(
                incompatible_component
            ),
            "boundaryTextureUnmeasuredComponentCount": len(
                unmeasured_component
            ),
            "boundaryTextureCompatibleComponents": compatible_component,
            "boundaryTextureIncompatibleComponents": incompatible_component,
            "boundaryTextureUnmeasuredComponents": unmeasured_component,
            "components": records,
            "depthChoice": (
                "display choice maximizes a fixed-depth local structure score; "
                "all fixed depth metrics remain reported"
            ),
            "magenta": "new collectively admitted ribbon centers",
            "yellow": "strict continuation edges from new ribbons to the prior sheet",
            "red": "nonadjacent chart overlap",
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "artifacts": {"montage": montage_path.name},
        "method": {
            "measurement": (
                "native CT sampled on exact intrinsic charts at fixed physical "
                "depth fractions with local axial structure-tensor continuity"
            ),
            "compatibility": (
                "a changed boundary is compatible when at least one fixed "
                "depth has a median axial disagreement no farther above its "
                "same-surface control than the larger of a fixed noise floor "
                "and a declared fraction of the control median-to-p90 spread"
            ),
            "acceptanceMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
