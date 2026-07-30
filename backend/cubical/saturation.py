from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

import numpy as np

from .contracts import (
    RawAcusSettings,
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    resolve_pipeline_manifest,
    sha256_file,
)
from .export import rgb_png
from .raw_acus import NeedleTable, read_calibration, read_needle_artifact
from .tables import PatchTable, read_patch_shard
from .topology import Int3


SATURATION_SCHEMA = "pareidolia.cubical-sheet-saturation"
SATURATION_VERSION = 2


@dataclass(frozen=True, slots=True)
class SheetSaturationSettings:
    """Calibrated structural-evidence-to-sheet assignment audit settings."""

    distance_radii_voxels: tuple[float, ...] = (
        1.0,
        2.0,
        2.5,
        3.0,
        4.0,
        6.0,
        8.0,
        12.0,
        16.0,
    )
    joint_residual_limits: tuple[float, ...] = (1.0, 1.5, 2.0, 2.5, 3.0, 4.0)
    assignment_share_thresholds: tuple[float, ...] = (0.5, 0.67, 0.8, 0.9)
    primary_joint_residual_limit: float = 2.5
    primary_confident_share: float = 0.8
    cell_overview_scale: int = 24

    def __post_init__(self) -> None:
        radii = tuple(float(value) for value in self.distance_radii_voxels)
        joint = tuple(float(value) for value in self.joint_residual_limits)
        shares = tuple(float(value) for value in self.assignment_share_thresholds)
        if (
            not radii
            or any(not math.isfinite(value) or value <= 0.0 for value in radii)
            or any(second <= first for first, second in zip(radii, radii[1:]))
        ):
            raise ValueError("saturation radii must be increasing finite positives")
        if (
            not joint
            or any(not math.isfinite(value) or value <= 0.0 for value in joint)
            or any(second <= first for first, second in zip(joint, joint[1:]))
        ):
            raise ValueError("joint residual limits must be increasing finite positives")
        if (
            not shares
            or any(not 0.0 < value <= 1.0 for value in shares)
            or any(second <= first for first, second in zip(shares, shares[1:]))
        ):
            raise ValueError("assignment shares must be increasing values in (0, 1]")
        if (
            not math.isfinite(self.primary_joint_residual_limit)
            or self.primary_joint_residual_limit <= 0.0
        ):
            raise ValueError("primary joint residual limit must be finite and positive")
        if not 0.0 < self.primary_confident_share <= 1.0:
            raise ValueError("primary confident share must lie in (0, 1]")
        if self.cell_overview_scale <= 0:
            raise ValueError("cell overview scale must be positive")
        object.__setattr__(self, "distance_radii_voxels", radii)
        object.__setattr__(self, "joint_residual_limits", joint)
        object.__setattr__(self, "assignment_share_thresholds", shares)

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["distance_radii_voxels"] = list(self.distance_radii_voxels)
        values["joint_residual_limits"] = list(self.joint_residual_limits)
        values["assignment_share_thresholds"] = list(
            self.assignment_share_thresholds
        )
        return values


@dataclass(frozen=True, slots=True)
class CellEvidenceAssignment:
    nearest_distance_voxels: np.ndarray
    best_fiber_residual_degrees: np.ndarray
    best_joint_residual: np.ndarray
    best_orthogonal_joint_residual: np.ndarray
    best_assignment_share: np.ndarray
    best_patch_index: np.ndarray


def classify_cell_structural_evidence(
    centers_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    *,
    cell_center_xyz: np.ndarray,
    patch_normals_xyz: np.ndarray,
    patch_heights: np.ndarray,
    patch_fibers_xyz: np.ndarray,
    patch_fiber_std_degrees: np.ndarray,
    patch_confidence: np.ndarray,
    depth_sigma_voxels: float,
    orientation_kernel_degrees: float,
) -> CellEvidenceAssignment:
    """Assign unsigned finite-fiber samples to selected local layer hypotheses.

    The residual is two dimensional: normal distance from the fitted layer and
    axial angular distance from its fitted fiber direction.  Patch confidence
    affects competition between plausible layers but cannot make a geometrically
    remote sample count as explained.
    """

    centers = np.asarray(centers_xyz, dtype=np.float64)
    directions = np.asarray(directions_xyz, dtype=np.float64)
    normals = np.asarray(patch_normals_xyz, dtype=np.float64)
    heights = np.asarray(patch_heights, dtype=np.float64)
    fibers = np.asarray(patch_fibers_xyz, dtype=np.float64)
    fiber_std = np.asarray(patch_fiber_std_degrees, dtype=np.float64)
    confidence = np.asarray(patch_confidence, dtype=np.float64)
    sample_count = len(centers)
    if centers.shape != (sample_count, 3) or directions.shape != (sample_count, 3):
        raise ValueError("structural evidence centers and directions must be N x 3")
    patch_count = len(normals)
    expected = {
        "patch_normals_xyz": normals.shape == (patch_count, 3),
        "patch_heights": heights.shape == (patch_count,),
        "patch_fibers_xyz": fibers.shape == (patch_count, 3),
        "patch_fiber_std_degrees": fiber_std.shape == (patch_count,),
        "patch_confidence": confidence.shape == (patch_count,),
    }
    invalid = [name for name, valid in expected.items() if not valid]
    if invalid:
        raise ValueError(f"invalid patch assignment arrays: {invalid}")
    if depth_sigma_voxels <= 0.0 or orientation_kernel_degrees <= 0.0:
        raise ValueError("assignment kernels must be positive")
    empty_float = np.full(sample_count, np.inf, dtype=np.float64)
    if not patch_count:
        return CellEvidenceAssignment(
            empty_float.copy(),
            empty_float.copy(),
            empty_float.copy(),
            empty_float.copy(),
            np.zeros(sample_count, dtype=np.float64),
            np.full(sample_count, -1, dtype=np.int64),
        )
    valid_patch = (
        np.all(np.isfinite(normals), axis=1)
        & np.all(np.isfinite(fibers), axis=1)
        & np.isfinite(heights)
        & np.isfinite(fiber_std)
        & np.isfinite(confidence)
        & (confidence > 0.0)
    )
    if not np.any(valid_patch):
        return CellEvidenceAssignment(
            empty_float.copy(),
            empty_float.copy(),
            empty_float.copy(),
            empty_float.copy(),
            np.zeros(sample_count, dtype=np.float64),
            np.full(sample_count, -1, dtype=np.int64),
        )
    retained = np.flatnonzero(valid_patch)
    normals = normals[retained]
    heights = heights[retained]
    fibers = fibers[retained]
    fiber_std = fiber_std[retained]
    confidence = confidence[retained]
    offsets = centers - np.asarray(cell_center_xyz, dtype=np.float64)[None, :]
    distance = np.abs(offsets @ normals.T - heights[None, :])
    axial_dot = np.clip(np.abs(directions @ fibers.T), 0.0, 1.0)
    angular = np.degrees(np.arccos(axial_dot))
    orthogonal_fibers = np.cross(normals, fibers)
    orthogonal_fibers /= np.maximum(
        np.linalg.norm(orthogonal_fibers, axis=1, keepdims=True), 1.0e-12
    )
    orthogonal_dot = np.clip(
        np.abs(directions @ orthogonal_fibers.T), 0.0, 1.0
    )
    orthogonal_angular = np.degrees(np.arccos(orthogonal_dot))
    angular_sigma = np.hypot(orientation_kernel_degrees, fiber_std)
    joint_squared = (distance / depth_sigma_voxels) ** 2 + (
        angular / angular_sigma[None, :]
    ) ** 2
    orthogonal_joint_squared = (distance / depth_sigma_voxels) ** 2 + (
        orthogonal_angular / angular_sigma[None, :]
    ) ** 2
    local_best = np.argmin(joint_squared, axis=1)
    row = np.arange(sample_count)
    best_joint = np.sqrt(joint_squared[row, local_best])
    log_likelihood = np.log(np.maximum(confidence, 1.0e-12))[None, :] - 0.5 * joint_squared
    maximum = np.max(log_likelihood, axis=1, keepdims=True)
    relative = np.exp(np.maximum(log_likelihood - maximum, -745.0))
    best_share = np.max(relative, axis=1) / np.maximum(np.sum(relative, axis=1), 1.0e-300)
    return CellEvidenceAssignment(
        np.min(distance, axis=1),
        angular[row, local_best],
        best_joint,
        np.sqrt(np.min(orthogonal_joint_squared, axis=1)),
        best_share,
        retained[local_best],
    )


def _boundary_class(cell: Int3, shape: Int3) -> int:
    touched = sum(cell[axis] in (0, shape[axis] - 1) for axis in range(3))
    return min(touched, 3)


def _fraction(numerator: float | int, denominator: float | int) -> float:
    return float(numerator) / max(float(denominator), 1.0e-300)


def _histogram_quantiles(
    counts: np.ndarray, edges: np.ndarray
) -> dict[str, float] | None:
    total = float(np.sum(counts, dtype=np.float64))
    if total <= 0.0:
        return None
    cumulative = np.cumsum(counts, dtype=np.float64)
    result: dict[str, float] = {}
    for name, quantile in (
        ("minimum", 0.0),
        ("p10", 0.1),
        ("p25", 0.25),
        ("median", 0.5),
        ("p75", 0.75),
        ("p90", 0.9),
        ("p95", 0.95),
        ("p99", 0.99),
        ("maximum", 1.0),
    ):
        target = max(np.finfo(np.float64).eps, quantile * total)
        if quantile >= 1.0:
            target = np.nextafter(total, -math.inf)
        index = min(
            int(np.searchsorted(cumulative, target, side="left")), len(counts) - 1
        )
        result[name] = round(float(0.5 * (edges[index] + edges[index + 1])), 6)
    return result


def _uint8_quantiles(histogram: np.ndarray) -> dict[str, int]:
    cumulative = np.cumsum(histogram, dtype=np.uint64)
    total = int(cumulative[-1])
    return {
        name: int(np.searchsorted(cumulative, max(1, math.ceil(quantile * total))))
        for name, quantile in (
            ("minimum", 0.0),
            ("p01", 0.01),
            ("p10", 0.1),
            ("median", 0.5),
            ("p90", 0.9),
            ("p99", 0.99),
            ("maximum", 1.0),
        )
    }


def _color_fraction(values: np.ndarray) -> np.ndarray:
    selected = np.clip(np.nan_to_num(values, nan=0.0), 0.0, 1.0)
    red = np.clip(2.0 * (1.0 - selected), 0.0, 1.0)
    green = np.clip(2.0 * selected, 0.0, 1.0)
    blue = 0.15 + 0.18 * selected
    return np.rint(255.0 * np.stack((red, green, blue), axis=-1)).astype(np.uint8)


def _color_layers(values: np.ndarray) -> np.ndarray:
    selected = np.clip(
        values / max(float(np.max(values, initial=1.0)), 1.0), 0.0, 1.0
    )
    return np.rint(
        255.0
        * np.stack(
            (
                0.15 + 0.75 * selected,
                0.12 + 0.75 * np.sqrt(selected),
                0.3 + 0.6 * (1.0 - selected),
            ),
            axis=-1,
        )
    ).astype(np.uint8)


def _write_overview(
    path: Path,
    evidence_mass: np.ndarray,
    confident_mass: np.ndarray,
    ambiguous_mass: np.ndarray,
    unexplained_mass: np.ndarray,
    layer_count: np.ndarray,
    scale: int,
) -> None:
    total_xy = np.sum(evidence_mass, axis=2, dtype=np.float64)

    def ratio(values: np.ndarray) -> np.ndarray:
        return np.divide(
            np.sum(values, axis=2, dtype=np.float64),
            total_xy,
            out=np.zeros_like(total_xy),
            where=total_xy > 0.0,
        )

    confident_xy = ratio(confident_mass)
    ambiguous_xy = ratio(ambiguous_mass)
    unexplained_xy = ratio(unexplained_mass)
    density_xy = total_xy / max(float(np.max(total_xy, initial=1.0)), 1.0)
    density_rgb = np.rint(
        255.0 * np.repeat(density_xy[:, :, None], 3, axis=2)
    ).astype(np.uint8)
    panels = (
        _color_fraction(confident_xy),
        _color_fraction(1.0 - ambiguous_xy),
        _color_fraction(1.0 - unexplained_xy),
        density_rgb,
        _color_layers(np.mean(layer_count, axis=2, dtype=np.float64)),
    )
    panels = tuple(np.transpose(value, (1, 0, 2)) for value in panels)
    separator = np.full(
        (panels[0].shape[0], 1, 3), (255, 255, 255), dtype=np.uint8
    )
    image = np.concatenate(
        tuple(value for panel in panels for value in (panel, separator))[:-1], axis=1
    )
    image = np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(rgb_png(image))
    temporary.replace(path)


def canonical_needle_artifact_identity(
    pipeline_root: Path,
) -> tuple[str, list[Path]]:
    manifests = sorted(pipeline_root.glob("extraction-tiles/*/needles-v1.json"))
    if not manifests:
        raise ValueError("raw Acus pipeline contains no canonical extraction tiles")
    records: list[dict[str, str]] = []
    for path in manifests:
        payload = json.loads(path.read_text())
        records.append(
            {
                "tileId": str(payload["region"]["id"]),
                "manifestSha256": sha256_file(path),
                "dataSha256": str(payload["data"]["sha256"]),
            }
        )
    return canonical_json_hash(records), manifests


def load_owned_canonical_needles(
    pipeline_root: Path,
    manifests: list[Path],
    *,
    pipeline_identity: str,
    window_start_xyz: np.ndarray,
    window_stop_xyz: np.ndarray,
) -> NeedleTable:
    parts: list[NeedleTable] = []
    for manifest_path in manifests:
        table = read_needle_artifact(
            manifest_path.with_suffix(""),
            identity_sha256=pipeline_identity,
            verify=True,
        )
        owned = np.all(
            (table.center_xyz >= window_start_xyz[None, :])
            & (table.center_xyz < window_stop_xyz[None, :]),
            axis=1,
        )
        if np.any(owned):
            parts.append(
                NeedleTable(
                    table.center_xyz[owned],
                    table.direction_xyz[owned],
                    table.score[owned],
                    table.axial_coverage[owned],
                    table.support_score[owned],
                )
            )
    if not parts:
        return NeedleTable.empty()
    result = NeedleTable(
        np.concatenate([value.center_xyz for value in parts]),
        np.concatenate([value.direction_xyz for value in parts]),
        np.concatenate([value.score for value in parts]),
        np.concatenate([value.axial_coverage for value in parts]),
        np.concatenate([value.support_score for value in parts]),
    )
    result.validate()
    return result


def _identity(
    root: Path,
    pipeline_root: Path,
    source: VolumeSource,
    settings: SheetSaturationSettings,
    needle_artifact_identity: str,
) -> dict[str, Any]:
    implementation_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": SATURATION_SCHEMA,
        "version": SATURATION_VERSION,
        "inputRoot": str(root),
        "pipelineRoot": str(pipeline_root),
        "inputPatchManifestSha256": sha256_file(root / "selected-patches-v1.json"),
        "inputPatchDataSha256": sha256_file(root / "selected-patches-v1.npz"),
        "canonicalNeedleArtifactSetSha256": needle_artifact_identity,
        "sourceIdentitySha256": source.source_identity["identitySha256"],
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(implementation_root / name)
            for name in (
                "saturation.py",
                "tables.py",
                "contracts.py",
                "raw_acus.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _class_record(
    count: int,
    mass: float,
    *,
    total_count: int,
    total_mass: float,
) -> dict[str, Any]:
    return {
        "evidencePointCount": int(count),
        "evidencePointFraction": round(_fraction(count, total_count), 7),
        "evidenceMass": round(float(mass), 7),
        "evidenceMassFraction": round(_fraction(mass, total_mass), 7),
    }


def run_sheet_saturation_audit(
    input_root: str | Path,
    output_root: str | Path,
    *,
    settings: SheetSaturationSettings | None = None,
    force: bool = False,
    progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Measure selected sheets against independently extracted fiber evidence."""

    started = time.monotonic()
    resolved = settings or SheetSaturationSettings()
    root = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output == root:
        raise ValueError("saturation output must differ from its selected-patch root")
    pipeline_root, pipeline = resolve_pipeline_manifest(root)
    pipeline_identity = str(pipeline["identity"]["identitySha256"])
    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(source_values["path"], source_values.get("metadataPath"))
    if source.source_identity["identitySha256"] != source_values["identitySha256"]:
        raise ValueError("native CT source identity changed since reconstruction")
    raw_settings = RawAcusSettings(**pipeline["identity"]["settings"])
    calibration = read_calibration(
        pipeline_root / "calibration-v1.json", identity_sha256=pipeline_identity
    )
    patches = read_patch_shard(root / "selected-patches-v1", verify=True)
    if patches.grid.coordinate_unit != "source-voxel":
        raise ValueError("saturation requires source-voxel patch coordinates")
    stride = int(raw_settings.cell_stride_voxels)
    if not np.allclose(patches.grid.cell_size_xyz, stride, atol=1.0e-6):
        raise ValueError("saturation requires one raw-Acus stride per cell")
    window_values = pipeline["identity"]["window"]
    window_start = np.asarray(window_values["originVoxelXYZ"], dtype=np.float64)
    shape = tuple(int(value) for value in patches.grid.shape_cells_xyz)
    window_stop = window_start + np.asarray(shape, dtype=np.float64) * stride
    needle_identity, needle_manifests = canonical_needle_artifact_identity(
        pipeline_root
    )
    identity = _identity(root, pipeline_root, source, resolved, needle_identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "saturation.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("saturation output belongs to another identity")
        if (
            not force
            and prior.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SATURATION_SCHEMA,
        "version": SATURATION_VERSION,
        "state": "loading-structural-evidence",
        "identity": identity,
        "inputRoot": str(root),
        "pipelineRoot": str(pipeline_root),
    }
    atomic_json(manifest_path, manifest)

    needles = load_owned_canonical_needles(
        pipeline_root,
        needle_manifests,
        pipeline_identity=pipeline_identity,
        window_start_xyz=window_start,
        window_stop_xyz=window_stop,
    )
    if not needles.count:
        raise ValueError("reconstruction window contains no canonical Acus evidence")
    cell_xyz = np.floor(
        (np.asarray(needles.center_xyz, dtype=np.float64) - window_start[None, :])
        / stride
    ).astype(np.int32)
    if np.any((cell_xyz < 0) | (cell_xyz >= np.asarray(shape)[None, :])):
        raise ValueError("owned canonical needle mapped outside the reconstruction grid")
    evidence_weight = (
        np.asarray(needles.score, dtype=np.float64)
        * np.sqrt(
            np.maximum(
                np.asarray(needles.axial_coverage, dtype=np.float64)
                * np.asarray(needles.support_score, dtype=np.float64),
                0.0,
            )
        )
    )
    rows_by_cell: dict[Int3, list[int]] = {}
    for row, values in enumerate(patches.cell_xyz):
        rows_by_cell.setdefault(tuple(int(value) for value in values), []).append(row)
    evidence_by_cell: dict[Int3, np.ndarray] = {
        cell: np.flatnonzero(np.all(cell_xyz == cell, axis=1))
        for cell in {
            tuple(int(value) for value in row) for row in cell_xyz
        }
    }

    evidence_count_by_cell = np.zeros(shape, dtype=np.uint32)
    evidence_mass_by_cell = np.zeros(shape, dtype=np.float64)
    layer_count = np.zeros(shape, dtype=np.uint8)
    class_count_by_cell = np.zeros((*shape, 3), dtype=np.uint32)
    class_mass_by_cell = np.zeros((*shape, 3), dtype=np.float64)
    distance_count_by_cell = np.zeros(
        (*shape, len(resolved.distance_radii_voxels)), dtype=np.uint32
    )
    distance_mass_by_cell = np.zeros_like(distance_count_by_cell, dtype=np.float64)
    joint_count_by_cell = np.zeros(
        (*shape, len(resolved.joint_residual_limits)), dtype=np.uint32
    )
    joint_mass_by_cell = np.zeros_like(joint_count_by_cell, dtype=np.float64)
    joint_confident_count_by_cell = np.zeros(
        (
            *shape,
            len(resolved.joint_residual_limits),
            len(resolved.assignment_share_thresholds),
        ),
        dtype=np.uint32,
    )
    joint_confident_mass_by_cell = np.zeros_like(
        joint_confident_count_by_cell, dtype=np.float64
    )
    distance_edges = np.linspace(0.0, stride * math.sqrt(3.0), 257)
    angular_edges = np.linspace(0.0, 90.0, 181)
    joint_edges = np.linspace(0.0, 20.0, 321)
    share_edges = np.linspace(0.0, 1.0, 101)
    histograms = {
        "distanceCount": np.zeros(len(distance_edges) - 1, dtype=np.uint64),
        "distanceMass": np.zeros(len(distance_edges) - 1, dtype=np.float64),
        "angularCount": np.zeros(len(angular_edges) - 1, dtype=np.uint64),
        "angularMass": np.zeros(len(angular_edges) - 1, dtype=np.float64),
        "jointCount": np.zeros(len(joint_edges) - 1, dtype=np.uint64),
        "jointMass": np.zeros(len(joint_edges) - 1, dtype=np.float64),
        "shareCount": np.zeros(len(share_edges) - 1, dtype=np.uint64),
        "shareMass": np.zeros(len(share_edges) - 1, dtype=np.float64),
    }
    ordered_cells = [
        (x, y, z)
        for z in range(shape[2])
        for y in range(shape[1])
        for x in range(shape[0])
    ]
    category_cell_count = np.zeros(4, dtype=np.uint32)
    category_evidence_count = np.zeros(4, dtype=np.uint64)
    category_evidence_mass = np.zeros(4, dtype=np.float64)
    category_class_count = np.zeros((4, 3), dtype=np.uint64)
    category_class_mass = np.zeros((4, 3), dtype=np.float64)
    evidence_class = np.full(needles.count, 2, dtype=np.uint8)
    evidence_nearest_distance = np.full(needles.count, np.inf, dtype=np.float32)
    evidence_best_fiber_residual = np.full(needles.count, np.inf, dtype=np.float32)
    evidence_best_joint_residual = np.full(needles.count, np.inf, dtype=np.float32)
    evidence_best_orthogonal_joint_residual = np.full(
        needles.count, np.inf, dtype=np.float32
    )
    evidence_best_assignment_share = np.zeros(needles.count, dtype=np.float32)
    manifest["state"] = "classifying"
    atomic_json(manifest_path, manifest)
    primary_joint = float(resolved.primary_joint_residual_limit)
    primary_share = float(resolved.primary_confident_share)
    for number, cell in enumerate(ordered_cells, start=1):
        rows = np.asarray(rows_by_cell.get(cell, ()), dtype=np.int64)
        sample = evidence_by_cell.get(cell, np.empty(0, dtype=np.int64))
        layer_count[cell] = len(rows)
        evidence_count_by_cell[cell] = len(sample)
        weights = evidence_weight[sample]
        evidence_mass_by_cell[cell] = float(np.sum(weights))
        category = _boundary_class(cell, shape)
        category_cell_count[category] += 1
        category_evidence_count[category] += len(sample)
        category_evidence_mass[category] += float(np.sum(weights))
        if len(sample):
            center = window_start + (np.asarray(cell, dtype=np.float64) + 0.5) * stride
            assignment = classify_cell_structural_evidence(
                needles.center_xyz[sample],
                needles.direction_xyz[sample],
                cell_center_xyz=center,
                patch_normals_xyz=patches.normal_xyz[rows],
                patch_heights=patches.height[rows],
                patch_fibers_xyz=patches.fiber_xyz[rows],
                patch_fiber_std_degrees=np.degrees(
                    patches.fiber_angular_std_radians[rows].astype(np.float64)
                ),
                patch_confidence=patches.confidence[rows],
                depth_sigma_voxels=raw_settings.depth_kernel_voxels,
                orientation_kernel_degrees=raw_settings.orientation_kernel_degrees,
            )
            supported = assignment.best_joint_residual <= primary_joint
            confident = supported & (
                assignment.best_assignment_share >= primary_share
            )
            ambiguous = supported & ~confident
            unexplained = ~supported
            masks = (confident, ambiguous, unexplained)
            evidence_class[sample[confident]] = 0
            evidence_class[sample[ambiguous]] = 1
            evidence_nearest_distance[sample] = assignment.nearest_distance_voxels
            evidence_best_fiber_residual[sample] = (
                assignment.best_fiber_residual_degrees
            )
            evidence_best_joint_residual[sample] = assignment.best_joint_residual
            evidence_best_orthogonal_joint_residual[sample] = (
                assignment.best_orthogonal_joint_residual
            )
            evidence_best_assignment_share[sample] = assignment.best_assignment_share
            for class_index, mask in enumerate(masks):
                count = int(np.count_nonzero(mask))
                mass = float(np.sum(weights[mask]))
                class_count_by_cell[cell + (class_index,)] = count
                class_mass_by_cell[cell + (class_index,)] = mass
                category_class_count[category, class_index] += count
                category_class_mass[category, class_index] += mass
            for radius_index, radius in enumerate(resolved.distance_radii_voxels):
                mask = assignment.nearest_distance_voxels <= radius
                distance_count_by_cell[cell + (radius_index,)] = int(
                    np.count_nonzero(mask)
                )
                distance_mass_by_cell[cell + (radius_index,)] = float(
                    np.sum(weights[mask])
                )
            for joint_index, limit in enumerate(resolved.joint_residual_limits):
                mask = assignment.best_joint_residual <= limit
                joint_count_by_cell[cell + (joint_index,)] = int(
                    np.count_nonzero(mask)
                )
                joint_mass_by_cell[cell + (joint_index,)] = float(
                    np.sum(weights[mask])
                )
                for share_index, share in enumerate(
                    resolved.assignment_share_thresholds
                ):
                    selected = mask & (assignment.best_assignment_share >= share)
                    joint_confident_count_by_cell[
                        cell + (joint_index, share_index)
                    ] = int(np.count_nonzero(selected))
                    joint_confident_mass_by_cell[
                        cell + (joint_index, share_index)
                    ] = float(np.sum(weights[selected]))
            for prefix, values, edges in (
                ("distance", assignment.nearest_distance_voxels, distance_edges),
                ("angular", assignment.best_fiber_residual_degrees, angular_edges),
                ("joint", assignment.best_joint_residual, joint_edges),
                ("share", assignment.best_assignment_share, share_edges),
            ):
                finite = np.isfinite(values)
                histograms[f"{prefix}Count"] += np.histogram(
                    values[finite], bins=edges
                )[0].astype(np.uint64)
                histograms[f"{prefix}Mass"] += np.histogram(
                    values[finite], bins=edges, weights=weights[finite]
                )[0]
        if progress is not None:
            progress(number, len(ordered_cells))

    total_count = needles.count
    total_mass = float(np.sum(evidence_weight))
    total_class_count = np.sum(class_count_by_cell, axis=(0, 1, 2), dtype=np.uint64)
    total_class_mass = np.sum(class_mass_by_cell, axis=(0, 1, 2), dtype=np.float64)
    total_distance_count = np.sum(
        distance_count_by_cell, axis=(0, 1, 2), dtype=np.uint64
    )
    total_distance_mass = np.sum(
        distance_mass_by_cell, axis=(0, 1, 2), dtype=np.float64
    )
    total_joint_count = np.sum(
        joint_count_by_cell, axis=(0, 1, 2), dtype=np.uint64
    )
    total_joint_mass = np.sum(joint_mass_by_cell, axis=(0, 1, 2), dtype=np.float64)
    total_joint_confident_count = np.sum(
        joint_confident_count_by_cell, axis=(0, 1, 2), dtype=np.uint64
    )
    total_joint_confident_mass = np.sum(
        joint_confident_mass_by_cell, axis=(0, 1, 2), dtype=np.float64
    )

    source_start = window_start.astype(np.int64)
    source_stop = window_stop.astype(np.int64)
    raw_block = source.memmap()[
        int(source_start[2]) : int(source_stop[2]),
        int(source_start[1]) : int(source_stop[1]),
        int(source_start[0]) : int(source_stop[0]),
    ]
    intensity_histogram = np.bincount(
        np.asarray(raw_block).reshape(-1), minlength=256
    ).astype(np.uint64)
    volume_voxels = int(np.sum(intensity_histogram, dtype=np.uint64))
    gate = float(calibration.air_threshold_raw)
    above_gate = int(
        np.sum(intensity_histogram[int(math.floor(gate)) + 1 :], dtype=np.uint64)
    )
    gate_fraction = _fraction(above_gate, volume_voxels)
    attenuation_gate_separates_occupancy = gate_fraction < 0.98

    table_path = output / "cell-saturation-v2.npz"
    temporary = table_path.with_suffix(table_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            distanceRadiiVoxels=np.asarray(
                resolved.distance_radii_voxels, dtype=np.float32
            ),
            jointResidualLimits=np.asarray(
                resolved.joint_residual_limits, dtype=np.float32
            ),
            assignmentShareThresholds=np.asarray(
                resolved.assignment_share_thresholds, dtype=np.float32
            ),
            evidencePointCount=evidence_count_by_cell,
            evidenceMass=evidence_mass_by_cell,
            selectedLayerCount=layer_count,
            classPointCount=class_count_by_cell,
            classMass=class_mass_by_cell,
            evidenceCellXYZ=cell_xyz,
            evidenceClass=evidence_class,
            evidenceWeight=evidence_weight.astype(np.float32),
            evidenceNearestPlaneDistanceVoxels=evidence_nearest_distance,
            evidenceBestFiberResidualDegrees=evidence_best_fiber_residual,
            evidenceBestJointResidual=evidence_best_joint_residual,
            evidenceBestOrthogonalJointResidual=(
                evidence_best_orthogonal_joint_residual
            ),
            evidenceBestAssignmentShare=evidence_best_assignment_share,
            distanceCoveredPointCount=distance_count_by_cell,
            distanceCoveredMass=distance_mass_by_cell,
            jointSupportedPointCount=joint_count_by_cell,
            jointSupportedMass=joint_mass_by_cell,
            jointConfidentPointCount=joint_confident_count_by_cell,
            jointConfidentMass=joint_confident_mass_by_cell,
            rawIntensityHistogram=intensity_histogram,
            nearestDistanceHistogramEdges=distance_edges.astype(np.float32),
            bestFiberResidualHistogramEdges=angular_edges.astype(np.float32),
            bestJointResidualHistogramEdges=joint_edges.astype(np.float32),
            bestAssignmentShareHistogramEdges=share_edges.astype(np.float32),
            **histograms,
        )
    temporary.replace(table_path)
    overview_path = output / "overview.png"
    _write_overview(
        overview_path,
        evidence_mass_by_cell,
        class_mass_by_cell[..., 0],
        class_mass_by_cell[..., 1],
        class_mass_by_cell[..., 2],
        layer_count,
        resolved.cell_overview_scale,
    )

    distance_curve = [
        {
            "radiusVoxels": float(radius),
            "radiusMicrons": round(float(radius * source.voxel_size_microns), 5),
            "evidencePointFraction": round(
                _fraction(total_distance_count[index], total_count), 7
            ),
            "evidenceMassFraction": round(
                _fraction(total_distance_mass[index], total_mass), 7
            ),
        }
        for index, radius in enumerate(resolved.distance_radii_voxels)
    ]
    joint_curve = [
        {
            "jointResidualLimit": float(limit),
            "supportedEvidencePointFraction": round(
                _fraction(total_joint_count[index], total_count), 7
            ),
            "supportedEvidenceMassFraction": round(
                _fraction(total_joint_mass[index], total_mass), 7
            ),
            "assignmentShareAtLeast": {
                str(share): {
                    "evidencePointFraction": round(
                        _fraction(
                            total_joint_confident_count[index, share_index],
                            total_count,
                        ),
                        7,
                    ),
                    "evidenceMassFraction": round(
                        _fraction(
                            total_joint_confident_mass[index, share_index],
                            total_mass,
                        ),
                        7,
                    ),
                }
                for share_index, share in enumerate(
                    resolved.assignment_share_thresholds
                )
            },
        }
        for index, limit in enumerate(resolved.joint_residual_limits)
    ]
    class_names = ("confidentlyAssigned", "ambiguous", "unexplained")
    classes = {
        name: _class_record(
            int(total_class_count[index]),
            float(total_class_mass[index]),
            total_count=total_count,
            total_mass=total_mass,
        )
        for index, name in enumerate(class_names)
    }
    unexplained_mask = evidence_class == 2
    maximum_depth_residual = (
        primary_joint * raw_settings.depth_kernel_voxels
    )
    depth_gap_mask = unexplained_mask & (
        evidence_nearest_distance > maximum_depth_residual
    )
    near_plane_fiber_gap_mask = unexplained_mask & ~depth_gap_mask
    orthogonal_ply_rescue_mask = unexplained_mask & (
        evidence_best_orthogonal_joint_residual <= primary_joint
    )

    def failure_record(mask: np.ndarray) -> dict[str, Any]:
        count = int(np.count_nonzero(mask))
        mass = float(np.sum(evidence_weight[mask]))
        record = _class_record(
            count,
            mass,
            total_count=total_count,
            total_mass=total_mass,
        )
        record["fractionOfUnexplainedMass"] = round(
            _fraction(mass, total_class_mass[2]), 7
        )
        return record

    ordered_quality = np.argsort(needles.score, kind="stable")
    quality_deciles = []
    for decile, indices in enumerate(np.array_split(ordered_quality, 10), start=1):
        decile_mass = float(np.sum(evidence_weight[indices]))
        orthogonal_mass = float(
            np.sum(evidence_weight[indices[orthogonal_ply_rescue_mask[indices]]])
        )
        directly_supported_mass = float(
            np.sum(evidence_weight[indices[evidence_class[indices] != 2]])
        )
        quality_deciles.append(
            {
                "decile": decile,
                "minimumNeedleScore": round(float(np.min(needles.score[indices])), 7),
                "maximumNeedleScore": round(float(np.max(needles.score[indices])), 7),
                "evidencePointCount": len(indices),
                "evidenceMass": round(decile_mass, 7),
                "classEvidenceMassFraction": {
                    class_name: round(
                        _fraction(
                            np.sum(
                                evidence_weight[
                                    indices[evidence_class[indices] == class_index]
                                ]
                            ),
                            decile_mass,
                        ),
                        7,
                    )
                    for class_index, class_name in enumerate(class_names)
                },
                "orthogonalCompatibleUnexplainedEvidenceMassFraction": round(
                    _fraction(orthogonal_mass, decile_mass), 7
                ),
                "directlySupportedOrOrthogonalCompatibleEvidenceMassFraction": round(
                    _fraction(
                        directly_supported_mass + orthogonal_mass,
                        decile_mass,
                    ),
                    7,
                ),
            }
        )
    categories: dict[str, Any] = {}
    for category, name in enumerate(
        ("interior", "outer-face", "outer-edge", "outer-corner")
    ):
        categories[name] = {
            "cellCount": int(category_cell_count[category]),
            "evidencePointCount": int(category_evidence_count[category]),
            "evidenceMass": round(float(category_evidence_mass[category]), 7),
            "classes": {
                class_name: _class_record(
                    int(category_class_count[category, class_index]),
                    float(category_class_mass[category, class_index]),
                    total_count=int(category_evidence_count[category]),
                    total_mass=float(category_evidence_mass[category]),
                )
                for class_index, class_name in enumerate(class_names)
            },
        }
    worst_cells = sorted(
        (
            {
                "cellXYZ": list(cell),
                "selectedLayerCount": int(layer_count[cell]),
                "evidencePointCount": int(evidence_count_by_cell[cell]),
                "evidenceMass": round(float(evidence_mass_by_cell[cell]), 7),
                "confidentEvidenceMassFraction": round(
                    _fraction(
                        class_mass_by_cell[cell + (0,)], evidence_mass_by_cell[cell]
                    ),
                    7,
                ),
                "ambiguousEvidenceMassFraction": round(
                    _fraction(
                        class_mass_by_cell[cell + (1,)], evidence_mass_by_cell[cell]
                    ),
                    7,
                ),
                "unexplainedEvidenceMassFraction": round(
                    _fraction(
                        class_mass_by_cell[cell + (2,)], evidence_mass_by_cell[cell]
                    ),
                    7,
                ),
                "unexplainedEvidenceMass": round(
                    float(class_mass_by_cell[cell + (2,)]), 7
                ),
            }
            for cell in ordered_cells
            if evidence_mass_by_cell[cell] > 0.0
        ),
        key=lambda value: (
            -value["unexplainedEvidenceMass"],
            -value["unexplainedEvidenceMassFraction"],
            value["cellXYZ"][2],
            value["cellXYZ"][1],
            value["cellXYZ"][0],
        ),
    )[:64]
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-sheet-saturation-summary",
        "version": SATURATION_VERSION,
        "identitySha256": identity_sha256,
        "inputRoot": str(root),
        "grid": json.loads((root / "selected-patches-v1.json").read_text())["grid"],
        "attenuationDiagnostic": {
            "volumeVoxels": volume_voxels,
            "rawIntensityQuantiles": _uint8_quantiles(intensity_histogram),
            "calibrationAdmissionGateRaw": gate,
            "fractionAboveCalibrationAdmissionGate": round(gate_fraction, 7),
            "gateSeparatesVoxelOccupancy": attenuation_gate_separates_occupancy,
            "interpretation": (
                "The raw Acus gate is suitable only for rejecting empty calibration cubes; "
                "this block's broad attenuation field requires structural evidence for "
                "sheet-utilization measurement."
            ),
        },
        "structuralEvidence": {
            "population": "canonical, source-owned, finite-length Acus needles",
            "polarity": "unsigned/axial",
            "pointCount": total_count,
            "mass": round(total_mass, 7),
            "massRule": "score times square root of axial coverage times support score",
            "cellsWithEvidence": int(np.count_nonzero(evidence_count_by_cell)),
            "candidateThreshold": raw_settings.candidate_threshold,
            "needleLengthVoxels": raw_settings.needle_length_voxels,
            "needleLengthMicrons": round(
                raw_settings.needle_length_voxels * source.voxel_size_microns, 5
            ),
        },
        "selectedGeometry": {
            "patches": patches.patch_count,
            "cells": int(np.prod(np.asarray(shape, dtype=np.int64))),
            "cellsWithoutSelectedPlane": int(np.count_nonzero(layer_count == 0)),
            "meanPlanesPerCell": round(float(np.mean(layer_count)), 7),
            "maximumPlanesPerCell": int(np.max(layer_count, initial=0)),
        },
        "assignmentModel": {
            "directions": "axial/unsigned",
            "depthSigmaVoxels": raw_settings.depth_kernel_voxels,
            "depthSigmaMicrons": round(
                raw_settings.depth_kernel_voxels * source.voxel_size_microns, 5
            ),
            "fiberSigma": (
                "quadrature sum of Acus orientation kernel and each fitted layer's "
                "fiber angular standard deviation"
            ),
            "jointResidual": "Euclidean norm of standardized depth and axial-fiber residuals",
            "bestAssignmentShare": (
                "largest confidence-weighted Gaussian likelihood divided by all selected "
                "layer likelihoods in the owning cell"
            ),
            "primaryJointResidualLimit": primary_joint,
            "primaryJointCoverageInterpretation": (
                "2.5 is approximately a 95% two-dimensional Gaussian residual region"
                if abs(primary_joint - 2.5) < 1.0e-9
                else "configured standardized residual region"
            ),
            "primaryConfidentShare": primary_share,
        },
        "primaryClasses": classes,
        "unexplainedDecomposition": {
            "maximumDepthResidualVoxels": maximum_depth_residual,
            "maximumDepthResidualMicrons": round(
                maximum_depth_residual * source.voxel_size_microns, 5
            ),
            "noPlaneWithinDepthGate": failure_record(depth_gap_mask),
            "nearPlaneButNoCompatibleFiberMode": failure_record(
                near_plane_fiber_gap_mask
            ),
        },
        "orthogonalPlyDiagnostic": {
            "interpretation": (
                "Diagnostic only: unexplained evidence whose unsigned fiber is "
                "compatible with the in-plane axis orthogonal to a selected layer "
                "at the same joint residual limit. It is not assigned or promoted."
            ),
            "compatibleUnexplainedEvidence": failure_record(
                orthogonal_ply_rescue_mask
            ),
            "directlySupportedOrOrthogonalCompatibleEvidenceMassFraction": round(
                _fraction(
                    total_class_mass[0]
                    + total_class_mass[1]
                    + np.sum(evidence_weight[orthogonal_ply_rescue_mask]),
                    total_mass,
                ),
                7,
            ),
        },
        "assignmentByNeedleScoreDecile": quality_deciles,
        "coverageByPlaneDistance": distance_curve,
        "coverageByJointResidual": joint_curve,
        "residualDistributions": {
            "nearestPlaneDistanceVoxels": {
                "pointWeighted": _histogram_quantiles(
                    histograms["distanceCount"], distance_edges
                ),
                "evidenceMassWeighted": _histogram_quantiles(
                    histograms["distanceMass"], distance_edges
                ),
            },
            "bestFiberResidualDegrees": {
                "pointWeighted": _histogram_quantiles(
                    histograms["angularCount"], angular_edges
                ),
                "evidenceMassWeighted": _histogram_quantiles(
                    histograms["angularMass"], angular_edges
                ),
            },
            "bestJointResidual": {
                "pointWeighted": _histogram_quantiles(
                    histograms["jointCount"], joint_edges
                ),
                "evidenceMassWeighted": _histogram_quantiles(
                    histograms["jointMass"], joint_edges
                ),
            },
            "bestAssignmentShare": {
                "pointWeighted": _histogram_quantiles(
                    histograms["shareCount"], share_edges
                ),
                "evidenceMassWeighted": _histogram_quantiles(
                    histograms["shareMass"], share_edges
                ),
            },
        },
        "spatialCategories": categories,
        "worstUnexplainedCells": worst_cells,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "cellTable": table_path.name,
            "overview": overview_path.name,
            "overviewPanels": [
                "confident mass fraction",
                "one minus ambiguous mass fraction",
                "one minus unexplained mass fraction",
                "structural evidence mass",
                "mean selected layers",
            ],
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
