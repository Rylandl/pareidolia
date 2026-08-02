from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import rgb_png
from .isolated_slab import _percentile_record
from .isolated_slab import ISOLATED_SLAB_SCHEMA, ISOLATED_SLAB_STEM
from .material_interface import MATERIAL_INTERFACE_SCHEMA, MATERIAL_INTERFACE_STEM
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA, PAIRED_SURFACE_BANK_STEM
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PAIRED_SURFACE_GROWTH_STEM,
)
from .paired_boundary_surface import (
    PAIRED_BOUNDARY_SURFACE_SCHEMA,
    PAIRED_BOUNDARY_SURFACE_STEM,
)
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
)


TIFXYZ_SURFACE_AUDIT_SCHEMA = "pareidolia.tifxyz-surface-control-audit"
TIFXYZ_SURFACE_AUDIT_VERSION = 1
TIFXYZ_SURFACE_AUDIT_STEM = "tifxyz-surface-control-audit-v1"


@dataclass(frozen=True, slots=True)
class TifxyzSurfaceAuditSettings:
    """Geometry gates for an independent known-surface control.

    The TIFXYZ surface is one observed face of a known-unrolled papyrus sheet,
    while the reconstruction represents its physical midpoint.  A candidate
    therefore matches truth when either one of its two observed boundaries is
    close to the known surface.  No fitted reconstruction labels are used.
    """

    component_representation: str = "paired-profiles"
    chart_window_radius_pixels: int = 96
    truth_support_margin_voxels: float = 12.0
    maximum_tangent_residual_voxels: float = 6.0
    maximum_boundary_height_residual_voxels: float = 3.0
    maximum_normal_residual_degrees: float = 25.0
    minimum_component_nodes: int = 8
    maximum_reported_components: int = 32
    nearest_batch_size: int = 64
    projection_size: int = 384

    def __post_init__(self) -> None:
        if self.component_representation not in (
            "paired-profiles",
            "boundary-tracks",
            "certified-patches",
            "certified-boundary-faces",
        ):
            raise ValueError(
                "truth-audit component representation must be paired-profiles "
                "boundary-tracks, certified-patches, or certified-boundary-faces"
            )
        integers = (
            self.chart_window_radius_pixels,
            self.minimum_component_nodes,
            self.maximum_reported_components,
            self.nearest_batch_size,
            self.projection_size,
        )
        if any(value < 1 for value in integers):
            raise ValueError("truth-audit counts and image size must be positive")
        positive = (
            self.truth_support_margin_voxels,
            self.maximum_tangent_residual_voxels,
            self.maximum_boundary_height_residual_voxels,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("truth-audit distances must be finite and positive")
        if not 0.0 < self.maximum_normal_residual_degrees < 90.0:
            raise ValueError("truth-audit normal gate must lie in (0, 90)")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve_mid_surface(root: str | Path) -> Path:
    value = Path(root).resolve()
    if value.is_file():
        return value
    candidates = (
        value / f"{PHYSICAL_MID_SURFACE_STEM}.json",
        value / f"{PAIRED_SURFACE_BANK_STEM}.json",
        value / f"{PAIRED_SURFACE_GROWTH_STEM}.json",
        value / f"{ISOLATED_SLAB_STEM}.json",
        value / f"{MATERIAL_INTERFACE_STEM}.json",
        value / f"{PAIRED_BOUNDARY_SURFACE_STEM}.json",
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise ValueError(f"no auditable physical geometry artifact exists at {value}")


def _normalized(values: np.ndarray) -> np.ndarray:
    item = np.asarray(values, dtype=np.float64)
    return item / np.maximum(np.linalg.norm(item, axis=-1, keepdims=True), 1.0e-12)


def _load_mid_surface(
    root: str | Path,
    *,
    component_representation: str = "paired-profiles",
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = _resolve_mid_surface(root)
    manifest = json.loads(manifest_path.read_text())
    if manifest.get("state") != "complete":
        raise ValueError("TIFXYZ audit requires a complete geometry artifact")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("physical mid-surface data differs from its manifest")
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    schema = manifest.get("schema")
    if schema == PHYSICAL_MID_SURFACE_SCHEMA:
        standardized = arrays
    elif schema == PAIRED_SURFACE_BANK_SCHEMA:
        standardized = {
            "midpointXYZ": arrays["midpointXYZ"],
            "boundaryLowerXYZ": arrays["boundaryLowerXYZ"],
            "boundaryUpperXYZ": arrays["boundaryUpperXYZ"],
            "normalXYZ": arrays["normalXYZ"],
            "componentId": arrays["seedComponentId"],
        }
    elif schema == PAIRED_SURFACE_GROWTH_SCHEMA:
        bank_path = Path(manifest["identity"]["candidateBank"]["manifestPath"])
        bank_manifest = json.loads(bank_path.read_text())
        bank_data_path = bank_path.parent / str(bank_manifest["data"]["path"])
        if sha256_file(bank_data_path) != bank_manifest["data"]["sha256"]:
            raise ValueError("paired growth references modified candidate geometry")
        with np.load(bank_data_path, allow_pickle=False) as stored:
            bank = {name: np.asarray(stored[name]) for name in stored.files}
        selected = np.asarray(arrays["selected"], dtype=bool)
        standardized = {
            "midpointXYZ": bank["midpointXYZ"][selected],
            "boundaryLowerXYZ": bank["boundaryLowerXYZ"][selected],
            "boundaryUpperXYZ": bank["boundaryUpperXYZ"][selected],
            "normalXYZ": bank["normalXYZ"][selected],
            "componentId": arrays["selectedLabel"][selected],
        }
        manifest = {**manifest, "source": bank_manifest["source"]}
    elif schema == ISOLATED_SLAB_SCHEMA:
        standardized = {
            "midpointXYZ": arrays["midpointXYZ"],
            "boundaryLowerXYZ": arrays["boundaryFirstXYZ"],
            "boundaryUpperXYZ": arrays["boundarySecondXYZ"],
            "normalXYZ": arrays["normalXYZ"],
            "componentId": arrays["componentId"],
        }
    elif schema == MATERIAL_INTERFACE_SCHEMA:
        standardized = {
            "midpointXYZ": arrays["positionXYZ"],
            "boundaryLowerXYZ": arrays["positionXYZ"],
            "boundaryUpperXYZ": arrays["positionXYZ"],
            "normalXYZ": arrays["signedNormalXYZ"],
            "componentId": np.arange(len(arrays["positionXYZ"]), dtype=np.int32),
        }
    elif schema == PAIRED_BOUNDARY_SURFACE_SCHEMA:
        direct_record = manifest["identity"]["directSurface"]
        direct_path = Path(str(direct_record["manifestPath"])).resolve()
        if sha256_file(direct_path) != direct_record["manifestSha256"]:
            raise ValueError("paired-boundary audit source manifest has changed")
        direct_manifest = json.loads(direct_path.read_text())
        direct_data_path = direct_path.parent / str(direct_manifest["data"]["path"])
        if (
            direct_manifest.get("schema") != PHYSICAL_MID_SURFACE_SCHEMA
            or direct_manifest.get("state") != "complete"
            or direct_manifest["data"]["sha256"] != direct_record["dataSha256"]
            or sha256_file(direct_data_path) != direct_record["dataSha256"]
        ):
            raise ValueError("paired-boundary audit source geometry has changed")
        with np.load(direct_data_path, allow_pickle=False) as stored:
            direct = {name: np.asarray(stored[name]) for name in stored.files}
        if component_representation == "certified-boundary-faces":
            component = np.asarray(
                arrays["endpointCertifiedFaceComponentId"], dtype=np.int32
            )
            retained = component >= 0
            endpoint = np.asarray(arrays["endpointXYZ"])[retained]
            standardized = {
                "midpointXYZ": endpoint,
                "boundaryLowerXYZ": endpoint,
                "boundaryUpperXYZ": endpoint,
                "normalXYZ": np.asarray(arrays["endpointInwardNormalXYZ"])[
                    retained
                ],
                "componentId": component[retained],
            }
        else:
            component = np.asarray(
                arrays["profileAssemblyComponentId"], dtype=np.int32
            )
            if len(component) != len(direct["midpointXYZ"]):
                raise ValueError("paired-boundary patches are not profile aligned")
            standardized = {
                "midpointXYZ": direct["midpointXYZ"],
                "boundaryLowerXYZ": direct["boundaryLowerXYZ"],
                "boundaryUpperXYZ": direct["boundaryUpperXYZ"],
                "normalXYZ": direct["normalXYZ"],
                "componentId": component,
            }
        manifest = {
            **manifest,
            "source": direct_manifest["source"],
            "geometry": direct_manifest["geometry"],
        }
    else:
        raise ValueError(f"unsupported TIFXYZ audit geometry schema: {schema}")
    return manifest_path, manifest, standardized


def load_tifxyz_surface_patch(
    root: str | Path,
    *,
    center_pixel_yx: tuple[int, int],
    radius_pixels: int,
    source_level: int,
    level_zero_origin_xyz: np.ndarray,
    local_low_xyz: np.ndarray,
    local_high_xyz: np.ndarray,
    margin_voxels: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Load one bounded TIFXYZ chart window and estimate its local normals."""

    try:
        import tifffile  # type: ignore[import-not-found]
    except ImportError as error:  # pragma: no cover - environment diagnostic
        raise RuntimeError(
            "TIFXYZ truth auditing requires requirements-truth.txt"
        ) from error

    tifxyz = Path(root).resolve()
    paths = [tifxyz / f"{axis}.tif" for axis in "xyz"]
    if not all(path.is_file() for path in paths):
        raise ValueError(f"TIFXYZ directory is incomplete: {tifxyz}")
    mapped = [tifffile.memmap(path, mode="r") for path in paths]
    if any(item.ndim != 2 for item in mapped) or any(
        item.shape != mapped[0].shape for item in mapped[1:]
    ):
        raise ValueError("TIFXYZ coordinate rasters must be aligned 2-D arrays")
    center_y, center_x = (int(value) for value in center_pixel_yx)
    low_y = max(1, center_y - radius_pixels)
    high_y = min(mapped[0].shape[0] - 1, center_y + radius_pixels + 1)
    low_x = max(1, center_x - radius_pixels)
    high_x = min(mapped[0].shape[1] - 1, center_x + radius_pixels + 1)
    if high_y - low_y < 3 or high_x - low_x < 3:
        raise ValueError("known-surface chart window is too small")
    raw = np.stack(
        [np.asarray(item[low_y:high_y, low_x:high_x], dtype=np.float64) for item in mapped],
        axis=-1,
    )
    finite = np.all(np.isfinite(raw), axis=2) & (np.linalg.norm(raw, axis=2) > 1.0)
    horizontal = np.zeros_like(raw)
    vertical = np.zeros_like(raw)
    horizontal[:, 1:-1] = raw[:, 2:] - raw[:, :-2]
    vertical[1:-1] = raw[2:] - raw[:-2]
    neighborhood = np.zeros_like(finite)
    neighborhood[1:-1, 1:-1] = (
        finite[1:-1, 1:-1]
        & finite[1:-1, :-2]
        & finite[1:-1, 2:]
        & finite[:-2, 1:-1]
        & finite[2:, 1:-1]
    )
    normal = np.cross(horizontal, vertical)
    normal_length = np.linalg.norm(normal, axis=2)
    valid = neighborhood & (normal_length > 1.0e-6)
    normal[valid] /= normal_length[valid, None]
    scale = float(2**int(source_level))
    local = (raw - np.asarray(level_zero_origin_xyz, dtype=np.float64)) / scale
    lower = np.asarray(local_low_xyz, dtype=np.float64) - float(margin_voxels)
    upper = np.asarray(local_high_xyz, dtype=np.float64) + float(margin_voxels)
    valid &= np.all((local >= lower) & (local < upper), axis=2)
    point = local[valid]
    point_normal = normal[valid]
    if len(point) < 8:
        raise ValueError("known TIFXYZ surface has insufficient samples in the block")

    horizontal_spacing = np.linalg.norm(horizontal[valid], axis=1) / scale
    vertical_spacing = np.linalg.norm(vertical[valid], axis=1) / scale
    summary = {
        "tifShapeYX": [int(value) for value in mapped[0].shape],
        "loadedWindowYX": [low_y, high_y, low_x, high_x],
        "validSurfaceSampleCount": int(len(point)),
        "horizontalTwoPixelSpacingVoxels": _percentile_record(horizontal_spacing),
        "verticalTwoPixelSpacingVoxels": _percentile_record(vertical_spacing),
    }
    return point.astype(np.float32), point_normal.astype(np.float32), summary


def _node_to_truth_metrics(
    midpoint_xyz: np.ndarray,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    truth_xyz: np.ndarray,
    truth_normal_xyz: np.ndarray,
    *,
    batch_size: int,
) -> dict[str, np.ndarray]:
    """Correspond each reconstructed node to the nearest truth chart sample."""

    midpoint = np.asarray(midpoint_xyz, dtype=np.float64)
    lower = np.asarray(lower_xyz, dtype=np.float64)
    upper = np.asarray(upper_xyz, dtype=np.float64)
    normal = _normalized(normal_xyz)
    truth = np.asarray(truth_xyz, dtype=np.float64)
    truth_normal = _normalized(truth_normal_xyz)
    output = {
        "truthIndex": np.empty(len(midpoint), dtype=np.int32),
        "midpointTangentResidualVoxels": np.empty(len(midpoint), dtype=np.float32),
        "midpointSignedHeightVoxels": np.empty(len(midpoint), dtype=np.float32),
        "lowerSignedHeightVoxels": np.empty(len(midpoint), dtype=np.float32),
        "upperSignedHeightVoxels": np.empty(len(midpoint), dtype=np.float32),
        "boundaryHeightResidualVoxels": np.empty(len(midpoint), dtype=np.float32),
        "closestBoundarySide": np.empty(len(midpoint), dtype=np.uint8),
        "normalResidualDegrees": np.empty(len(midpoint), dtype=np.float32),
    }
    for start in range(0, len(midpoint), batch_size):
        stop = min(start + batch_size, len(midpoint))
        displacement = midpoint[start:stop, None, :] - truth[None, :, :]
        distance_squared = np.einsum("bti,bti->bt", displacement, displacement)
        nearest = np.argmin(distance_squared, axis=1)
        row = np.arange(stop - start)
        nearest_truth = truth[nearest]
        nearest_normal = truth_normal[nearest]
        midpoint_delta = midpoint[start:stop] - nearest_truth
        midpoint_height = np.einsum("bi,bi->b", midpoint_delta, nearest_normal)
        tangent_squared = np.maximum(
            np.einsum("bi,bi->b", midpoint_delta, midpoint_delta)
            - midpoint_height * midpoint_height,
            0.0,
        )
        lower_height = np.einsum(
            "bi,bi->b", lower[start:stop] - nearest_truth, nearest_normal
        )
        upper_height = np.einsum(
            "bi,bi->b", upper[start:stop] - nearest_truth, nearest_normal
        )
        upper_closer = np.abs(upper_height) < np.abs(lower_height)
        output["truthIndex"][start:stop] = nearest
        output["midpointTangentResidualVoxels"][start:stop] = np.sqrt(
            tangent_squared
        )
        output["midpointSignedHeightVoxels"][start:stop] = midpoint_height
        output["lowerSignedHeightVoxels"][start:stop] = lower_height
        output["upperSignedHeightVoxels"][start:stop] = upper_height
        output["boundaryHeightResidualVoxels"][start:stop] = np.where(
            upper_closer, np.abs(upper_height), np.abs(lower_height)
        )
        output["closestBoundarySide"][start:stop] = upper_closer.astype(np.uint8)
        cosine = np.abs(np.einsum("bi,bi->b", normal[start:stop], nearest_normal))
        output["normalResidualDegrees"][start:stop] = np.degrees(
            np.arccos(np.clip(cosine, 0.0, 1.0))
        )
    return output


def _truth_to_node_metrics(
    truth_xyz: np.ndarray,
    truth_normal_xyz: np.ndarray,
    midpoint_xyz: np.ndarray,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    *,
    batch_size: int,
    maximum_tangent_residual_voxels: float,
    maximum_boundary_height_residual_voxels: float,
    maximum_normal_residual_degrees: float,
) -> dict[str, np.ndarray]:
    """Find the reconstructed boundary best supported at every truth sample."""

    truth = np.asarray(truth_xyz, dtype=np.float64)
    truth_normal = _normalized(truth_normal_xyz)
    midpoint = np.asarray(midpoint_xyz, dtype=np.float64)
    lower = np.asarray(lower_xyz, dtype=np.float64)
    upper = np.asarray(upper_xyz, dtype=np.float64)
    normal = _normalized(normal_xyz)
    output = {
        "nodeIndex": np.empty(len(truth), dtype=np.int32),
        "tangentResidualVoxels": np.empty(len(truth), dtype=np.float32),
        "boundaryHeightResidualVoxels": np.empty(len(truth), dtype=np.float32),
        "normalResidualDegrees": np.empty(len(truth), dtype=np.float32),
    }
    for start in range(0, len(truth), batch_size):
        stop = min(start + batch_size, len(truth))
        truth_batch = truth[start:stop]
        truth_normal_batch = truth_normal[start:stop]
        midpoint_delta = midpoint[None, :, :] - truth_batch[:, None, :]
        midpoint_height = np.einsum(
            "bni,bi->bn", midpoint_delta, truth_normal_batch
        )
        tangent_squared = np.maximum(
            np.einsum("bni,bni->bn", midpoint_delta, midpoint_delta)
            - midpoint_height * midpoint_height,
            0.0,
        )
        lower_height = np.einsum(
            "bni,bi->bn", lower[None, :, :] - truth_batch[:, None, :], truth_normal_batch
        )
        upper_height = np.einsum(
            "bni,bi->bn", upper[None, :, :] - truth_batch[:, None, :], truth_normal_batch
        )
        boundary_height = np.minimum(np.abs(lower_height), np.abs(upper_height))
        cosine = np.abs(truth_normal_batch @ normal.T)
        normal_degrees = np.degrees(np.arccos(np.clip(cosine, 0.0, 1.0)))
        tangent = np.sqrt(tangent_squared)
        eligible = (
            (tangent <= maximum_tangent_residual_voxels)
            & (boundary_height <= maximum_boundary_height_residual_voxels)
            & (normal_degrees <= maximum_normal_residual_degrees)
        )
        normalized_metric = (
            (tangent / maximum_tangent_residual_voxels) ** 2
            + (boundary_height / maximum_boundary_height_residual_voxels) ** 2
            + (normal_degrees / maximum_normal_residual_degrees) ** 2
        )
        eligible_metric = np.where(eligible, normalized_metric, np.inf)
        has_eligible = np.any(eligible, axis=1)
        nearest = np.argmin(eligible_metric, axis=1)
        fallback = np.argmin(normalized_metric, axis=1)
        nearest[~has_eligible] = fallback[~has_eligible]
        row = np.arange(stop - start)
        output["nodeIndex"][start:stop] = nearest
        output["tangentResidualVoxels"][start:stop] = np.sqrt(
            tangent_squared[row, nearest]
        )
        output["boundaryHeightResidualVoxels"][start:stop] = boundary_height[
            row, nearest
        ]
        output["normalResidualDegrees"][start:stop] = normal_degrees[row, nearest]
    return output


def audit_mid_surfaces_against_truth(
    midpoint_xyz: np.ndarray,
    lower_xyz: np.ndarray,
    upper_xyz: np.ndarray,
    normal_xyz: np.ndarray,
    component_id: np.ndarray,
    truth_xyz: np.ndarray,
    truth_normal_xyz: np.ndarray,
    *,
    settings: TifxyzSurfaceAuditSettings | None = None,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Measure orientation, face localization, coverage, and fragmentation."""

    resolved = settings or TifxyzSurfaceAuditSettings()
    midpoint = np.asarray(midpoint_xyz, dtype=np.float64)
    lower = np.asarray(lower_xyz, dtype=np.float64)
    upper = np.asarray(upper_xyz, dtype=np.float64)
    normal = _normalized(normal_xyz)
    component = np.asarray(component_id, dtype=np.int32)
    truth = np.asarray(truth_xyz, dtype=np.float64)
    truth_normal = _normalized(truth_normal_xyz)
    if any(len(item) != len(midpoint) for item in (lower, upper, normal, component)):
        raise ValueError("mid-surface arrays are not aligned")
    if len(truth) != len(truth_normal) or not len(truth):
        raise ValueError("known-surface points and normals are not aligned")

    node_metrics = _node_to_truth_metrics(
        midpoint,
        lower,
        upper,
        normal,
        truth,
        truth_normal,
        batch_size=resolved.nearest_batch_size,
    )
    node_match = (
        (
            node_metrics["midpointTangentResidualVoxels"]
            <= resolved.maximum_tangent_residual_voxels
        )
        & (
            node_metrics["boundaryHeightResidualVoxels"]
            <= resolved.maximum_boundary_height_residual_voxels
        )
        & (
            node_metrics["normalResidualDegrees"]
            <= resolved.maximum_normal_residual_degrees
        )
    )
    truth_metrics = _truth_to_node_metrics(
        truth,
        truth_normal,
        midpoint,
        lower,
        upper,
        normal,
        batch_size=resolved.nearest_batch_size,
        maximum_tangent_residual_voxels=resolved.maximum_tangent_residual_voxels,
        maximum_boundary_height_residual_voxels=(
            resolved.maximum_boundary_height_residual_voxels
        ),
        maximum_normal_residual_degrees=resolved.maximum_normal_residual_degrees,
    )
    truth_match = (
        (truth_metrics["tangentResidualVoxels"] <= resolved.maximum_tangent_residual_voxels)
        & (
            truth_metrics["boundaryHeightResidualVoxels"]
            <= resolved.maximum_boundary_height_residual_voxels
        )
        & (
            truth_metrics["normalResidualDegrees"]
            <= resolved.maximum_normal_residual_degrees
        )
    )
    truth_component = component[truth_metrics["nodeIndex"]]
    labeled_truth_match = truth_match & (truth_component >= 0)
    covered_component, covered_count = np.unique(
        truth_component[labeled_truth_match], return_counts=True
    )
    component_reports: list[dict[str, Any]] = []
    for value, count in zip(*np.unique(component, return_counts=True)):
        if value < 0 or count < resolved.minimum_component_nodes:
            continue
        member = component == value
        matched = member & node_match
        covered = truth_match & (truth_component == value)
        overlap = member & (
            node_metrics["midpointTangentResidualVoxels"]
            <= resolved.maximum_tangent_residual_voxels
        )
        side = node_metrics["closestBoundarySide"][matched]
        lower_fraction = float(np.mean(side == 0)) if len(side) else 0.0
        upper_fraction = float(np.mean(side == 1)) if len(side) else 0.0
        score = (
            float(np.count_nonzero(covered)) / max(len(truth), 1)
            * math.sqrt(float(np.count_nonzero(matched)) / max(int(count), 1))
        )
        component_reports.append(
            {
                "componentId": int(value),
                "nodeCount": int(count),
                "nodeCountOverKnownChart": int(np.count_nonzero(overlap)),
                "matchedNodeCount": int(np.count_nonzero(matched)),
                "matchedNodeFraction": round(
                    float(np.count_nonzero(matched)) / max(int(count), 1), 6
                ),
                "coveredTruthSampleCount": int(np.count_nonzero(covered)),
                "truthCoverageFraction": round(
                    float(np.count_nonzero(covered)) / max(len(truth), 1), 6
                ),
                "controlScore": round(score, 8),
                "closestBoundarySideConsistency": round(
                    max(lower_fraction, upper_fraction), 6
                ),
                "boundaryHeightResidualVoxels": _percentile_record(
                    node_metrics["boundaryHeightResidualVoxels"][matched]
                ),
                "normalResidualDegrees": _percentile_record(
                    node_metrics["normalResidualDegrees"][matched]
                ),
                "midpointSignedHeightVoxels": _percentile_record(
                    node_metrics["midpointSignedHeightVoxels"][matched]
                ),
            }
        )
    component_reports.sort(
        key=lambda item: (
            -float(item["controlScore"]),
            -int(item["coveredTruthSampleCount"]),
            int(item["componentId"]),
        )
    )
    summary = {
        "counts": {
            "midSurfaceNodeCount": int(len(midpoint)),
            "knownSurfaceSampleCount": int(len(truth)),
            "matchedMidSurfaceNodeCount": int(np.count_nonzero(node_match)),
            "coveredKnownSurfaceSampleCount": int(np.count_nonzero(truth_match)),
            "coveringComponentCount": int(len(covered_component)),
            "coveredByUnlabelledCandidateCount": int(
                np.count_nonzero(truth_match & (truth_component < 0))
            ),
        },
        "coverage": {
            "knownSurfaceSampleFraction": round(float(np.mean(truth_match)), 6),
            "largestSingleComponentTruthFraction": round(
                float(covered_count.max(initial=0)) / max(len(truth), 1), 6
            ),
            "coveredTruthComponentSizes": [
                {
                    "componentId": int(value),
                    "truthSampleCount": int(count),
                }
                for value, count in sorted(
                    zip(covered_component, covered_count),
                    key=lambda item: (-int(item[1]), int(item[0])),
                )
            ],
        },
        "matchedGeometry": {
            "boundaryHeightResidualVoxels": _percentile_record(
                truth_metrics["boundaryHeightResidualVoxels"][truth_match]
            ),
            "tangentResidualVoxels": _percentile_record(
                truth_metrics["tangentResidualVoxels"][truth_match]
            ),
            "normalResidualDegrees": _percentile_record(
                truth_metrics["normalResidualDegrees"][truth_match]
            ),
        },
        "components": component_reports[: resolved.maximum_reported_components],
    }
    arrays = {
        **node_metrics,
        "nodeMatchesTruth": node_match,
        "truthNearestNode": truth_metrics["nodeIndex"],
        "truthTangentResidualVoxels": truth_metrics["tangentResidualVoxels"],
        "truthBoundaryHeightResidualVoxels": truth_metrics[
            "boundaryHeightResidualVoxels"
        ],
        "truthNormalResidualDegrees": truth_metrics["normalResidualDegrees"],
        "truthMatched": truth_match,
        "truthMatchedComponentId": truth_component,
    }
    return summary, arrays


def _write_projection(
    path: Path,
    truth_xyz: np.ndarray,
    midpoint_xyz: np.ndarray,
    component_id: np.ndarray,
    best_component_ids: list[int],
    low_xyz: np.ndarray,
    high_xyz: np.ndarray,
    *,
    size: int,
) -> None:
    image = np.full((size, size * 3, 3), (8, 12, 17), dtype=np.uint8)
    projections = ((0, 1), (0, 2), (1, 2))
    palette = ((255, 177, 66), (255, 91, 150), (180, 121, 255), (92, 211, 255))

    def draw(points: np.ndarray, panel: int, axes: tuple[int, int], color: tuple[int, int, int], radius: int) -> None:
        if not len(points):
            return
        span = np.maximum(high_xyz - low_xyz, 1.0)
        x = np.rint((points[:, axes[0]] - low_xyz[axes[0]]) / span[axes[0]] * (size - 1)).astype(np.int32)
        y = np.rint((1.0 - (points[:, axes[1]] - low_xyz[axes[1]]) / span[axes[1]]) * (size - 1)).astype(np.int32)
        for dx in range(-radius, radius + 1):
            for dy in range(-radius, radius + 1):
                valid = (x + dx >= 0) & (x + dx < size) & (y + dy >= 0) & (y + dy < size)
                image[y[valid] + dy, panel * size + x[valid] + dx] = color

    for panel, axes in enumerate(projections):
        draw(midpoint_xyz, panel, axes, (35, 49, 61), 0)
        draw(truth_xyz, panel, axes, (45, 235, 213), 1)
        for rank, value in enumerate(best_component_ids):
            draw(
                midpoint_xyz[component_id == value],
                panel,
                axes,
                palette[rank % len(palette)],
                1,
            )
    path.write_bytes(rgb_png(image))


def run_tifxyz_surface_audit(
    tifxyz_root: str | Path,
    mid_surface_root: str | Path,
    source_metadata_path: str | Path,
    output_root: str | Path,
    *,
    center_pixel_yx: tuple[int, int] | None = None,
    settings: TifxyzSurfaceAuditSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Audit raw-CT reconstruction against one independently unrolled surface."""

    started = time.monotonic()
    resolved = settings or TifxyzSurfaceAuditSettings()
    manifest_path, manifest, arrays = _load_mid_surface(
        mid_surface_root,
        component_representation=resolved.component_representation,
    )
    metadata_path = Path(source_metadata_path).resolve()
    metadata = json.loads(metadata_path.read_text())
    known = metadata.get("knownSurface", {})
    resolved_center = center_pixel_yx or tuple(known.get("centerPixelYX", ()))
    if len(resolved_center) != 2:
        raise ValueError("known TIFXYZ center pixel must be supplied")
    source_level = int(metadata.get("sourceLevel", 0))
    level_zero_origin = np.asarray(metadata["originXYZ"], dtype=np.float64)
    source_origin = np.asarray(manifest["source"]["sourceOriginXYZ"], dtype=np.float64)
    if not np.array_equal(source_origin, level_zero_origin):
        raise ValueError("reconstruction and truth metadata origins differ")
    owned = manifest["geometry"]["ownedVoxelBounds"]
    low = np.asarray(owned["startXYZ"], dtype=np.float64)
    high = np.asarray(owned["stopXYZExclusive"], dtype=np.float64)
    tifxyz = Path(tifxyz_root).resolve()
    truth, truth_normal, truth_summary = load_tifxyz_surface_patch(
        tifxyz,
        center_pixel_yx=(int(resolved_center[0]), int(resolved_center[1])),
        radius_pixels=resolved.chart_window_radius_pixels,
        source_level=source_level,
        level_zero_origin_xyz=level_zero_origin,
        local_low_xyz=low,
        local_high_xyz=high,
        margin_voxels=resolved.truth_support_margin_voxels,
    )
    truth_owned = np.all((truth >= low) & (truth < high), axis=1)
    if np.count_nonzero(truth_owned) < 8:
        raise ValueError("known surface has insufficient samples in owned bounds")
    midpoint = np.asarray(arrays["midpointXYZ"], dtype=np.float64) - source_origin
    lower = np.asarray(arrays["boundaryLowerXYZ"], dtype=np.float64) - source_origin
    upper = np.asarray(arrays["boundaryUpperXYZ"], dtype=np.float64) - source_origin
    normal = np.asarray(arrays["normalXYZ"], dtype=np.float64)
    component = np.asarray(arrays["componentId"], dtype=np.int32)
    if resolved.component_representation == "boundary-tracks":
        required = ("lowerFaceComponentId", "upperFaceComponentId")
        if any(name not in arrays for name in required):
            raise ValueError(
                "boundary-track truth audit requires per-profile face components"
            )
        endpoint = np.empty((2 * len(midpoint), 3), dtype=np.float64)
        endpoint[0::2] = lower
        endpoint[1::2] = upper
        endpoint_component = np.empty(2 * len(midpoint), dtype=np.int32)
        endpoint_component[0::2] = np.asarray(
            arrays["lowerFaceComponentId"], dtype=np.int32
        )
        endpoint_component[1::2] = np.asarray(
            arrays["upperFaceComponentId"], dtype=np.int32
        )
        if not np.any(endpoint_component >= 0):
            raise ValueError("geometry artifact contains no reconstructed boundary tracks")
        midpoint = endpoint
        lower = endpoint
        upper = endpoint
        normal = np.repeat(normal, 2, axis=0)
        component = endpoint_component
    audit, audit_arrays = audit_mid_surfaces_against_truth(
        midpoint,
        lower,
        upper,
        normal,
        component,
        truth[truth_owned],
        truth_normal[truth_owned],
        settings=resolved,
    )
    identity: dict[str, Any] = {
        "schema": TIFXYZ_SURFACE_AUDIT_SCHEMA,
        "version": TIFXYZ_SURFACE_AUDIT_VERSION,
        "midSurface": {
            "manifestPath": str(manifest_path),
            "manifestSha256": sha256_file(manifest_path),
            "dataSha256": manifest["data"]["sha256"],
        },
        "sourceMetadata": {
            "path": str(metadata_path),
            "sha256": sha256_file(metadata_path),
        },
        "tifxyz": {
            "path": str(tifxyz),
            "coordinateSha256": {
                axis: sha256_file(tifxyz / f"{axis}.tif") for axis in "xyz"
            },
        },
        "centerPixelYX": [int(value) for value in resolved_center],
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_output = output / f"{TIFXYZ_SURFACE_AUDIT_STEM}.json"
    data_output = output / f"{TIFXYZ_SURFACE_AUDIT_STEM}.npz"
    projection_output = output / "known-surface-control-projections.png"
    if not force and manifest_output.is_file() and data_output.is_file():
        cached = json.loads(manifest_output.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_output)
        ):
            return cached

    temporary = data_output.with_suffix(data_output.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            truthXYZ=truth[truth_owned],
            truthNormalXYZ=truth_normal[truth_owned],
            **audit_arrays,
        )
    temporary.replace(data_output)
    best_ids = [int(item["componentId"]) for item in audit["components"][:4]]
    _write_projection(
        projection_output,
        truth[truth_owned],
        midpoint,
        component,
        best_ids,
        low,
        high,
        size=resolved.projection_size,
    )
    payload: dict[str, Any] = {
        "schema": TIFXYZ_SURFACE_AUDIT_SCHEMA,
        "version": TIFXYZ_SURFACE_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "truthSurface": {
            **truth_summary,
            "ownedSurfaceSampleCount": int(np.count_nonzero(truth_owned)),
            "knownPointLocalXYZ": known.get("localXYZ"),
            "knownNormalXYZ": known.get("normalXYZ"),
        },
        **audit,
        "data": {
            "path": data_output.name,
            "bytes": data_output.stat().st_size,
            "sha256": sha256_file(data_output),
            "fields": ["truthXYZ", "truthNormalXYZ", *audit_arrays],
        },
        "artifacts": {"projection": projection_output.name},
        "method": {
            "truth": "one independently unrolled PHerc1667 TIFXYZ surface face",
            "representation": resolved.component_representation,
            "match": (
                "one reconstructed physical boundary track must agree with truth "
                "in tangent position, normal height, and unsigned normal"
                if resolved.component_representation
                in ("boundary-tracks", "certified-boundary-faces")
                else (
                    "one locally two-face-certified papyrus patch must place either "
                    "physical boundary on truth with tangent, height, and unsigned-"
                    "normal agreement"
                    if resolved.component_representation == "certified-patches"
                    else "either reconstructed physical boundary must agree with "
                    "truth in tangent position, normal height, and unsigned normal"
                )
            ),
            "labelsUsed": False,
            "fittedOffsetUsed": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_output, payload)
    return payload
