from __future__ import annotations

import json
import math
import shutil
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any, Callable, Hashable, Iterable, Mapping

import numpy as np

from .block import (
    BlockBounds,
    SurfaceBlock,
    assemble_surface_hierarchy,
    rebuild_surface_block,
)
from .contracts import (
    RawAcusSettings,
    atomic_json,
    canonical_json_hash,
    resolve_pipeline_manifest,
    sha256_file,
)
from .continuity import apply_join_continuity_refinement
from .geometry import axial_angle_radians
from .mode_bank import MODE_BANK_SCHEMA, load_mode_bank
from .stratigraphy import LayerModeTable
from .surface_graph import read_surface_graph, write_surface_graph
from .tables import PatchTable, read_patch_shard
from .topology import GridSpec

if TYPE_CHECKING:
    from .sheet_evidence import BlockSheetEvidence


STRATIGRAPHIC_CONTINUITY_SCHEMA = (
    "pareidolia.cubical-stratigraphic-continuity"
)
STRATIGRAPHIC_CONTINUITY_VERSION = 1
FINGERPRINT_SCHEMA = "pareidolia.cubical-stratigraphic-fingerprints"
FINGERPRINT_VERSION = 1
CANDIDATE_CONTINUITY_STEM = "candidate-stratigraphic-continuity-v1"

JoinKey = tuple[int, int, int, tuple[int, int, int]]


@dataclass(frozen=True, slots=True)
class StratigraphicContinuitySettings:
    """General graph-context and robust-tail settings for mode-bank continuity."""

    neighborhood_radius_hops: int = 3
    minimum_side_patches: int = 3
    minimum_context_modes: int = 2
    minimum_coverage_fraction: float = 0.5
    minimum_common_depth_span_voxels: float = 8.0
    maximum_anchor_height_residual_voxels: float = 0.5
    maximum_anchor_normal_residual_degrees: float = 0.5
    maximum_anchor_fiber_residual_degrees: float = 1.0
    outlier_standard_deviations: float = 4.0
    minimum_log_scale: float = 0.12
    minimum_calibration_joins: int = 32

    def __post_init__(self) -> None:
        integer_positive = (
            self.neighborhood_radius_hops,
            self.minimum_side_patches,
            self.minimum_context_modes,
            self.minimum_calibration_joins,
        )
        if any(int(value) <= 0 for value in integer_positive):
            raise ValueError("stratigraphic integer settings must be positive")
        finite_positive = (
            self.minimum_common_depth_span_voxels,
            self.maximum_anchor_height_residual_voxels,
            self.maximum_anchor_normal_residual_degrees,
            self.maximum_anchor_fiber_residual_degrees,
            self.outlier_standard_deviations,
            self.minimum_log_scale,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("stratigraphic floating settings must be finite and positive")
        if not 0.0 < self.minimum_coverage_fraction <= 1.0:
            raise ValueError("minimum coverage fraction must lie in (0, 1]")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class PatchFingerprintTable:
    """One mode-bank distribution anchored on every selected cubical patch."""

    patch_id: np.ndarray
    anchor_valid: np.ndarray
    anchor_shard_index: np.ndarray
    anchor_mode_index: np.ndarray
    anchor_height_residual_voxels: np.ndarray
    anchor_normal_residual_degrees: np.ndarray
    anchor_fiber_residual_degrees: np.ndarray
    context_mode_count: np.ndarray
    support_low_voxels: np.ndarray
    support_high_voxels: np.ndarray
    normal_xyz: np.ndarray
    depth_offsets_voxels: np.ndarray
    density: np.ndarray
    orientation_moment: np.ndarray

    @property
    def patch_count(self) -> int:
        return int(len(self.patch_id))

    @property
    def depth_count(self) -> int:
        return int(len(self.depth_offsets_voxels))

    def validate(self) -> None:
        patches = self.patch_count
        depths = self.depth_count
        expected = {
            "patch_id": (patches,),
            "anchor_valid": (patches,),
            "anchor_shard_index": (patches,),
            "anchor_mode_index": (patches,),
            "anchor_height_residual_voxels": (patches,),
            "anchor_normal_residual_degrees": (patches,),
            "anchor_fiber_residual_degrees": (patches,),
            "context_mode_count": (patches,),
            "support_low_voxels": (patches,),
            "support_high_voxels": (patches,),
            "normal_xyz": (patches, 3),
            "depth_offsets_voxels": (depths,),
            "density": (patches, depths),
            "orientation_moment": (patches, depths),
        }
        for name, shape in expected.items():
            value = getattr(self, name)
            if value.shape != shape:
                raise ValueError(f"{name} has shape {value.shape}, expected {shape}")
            if np.issubdtype(value.dtype, np.floating) and np.any(~np.isfinite(value)):
                raise ValueError(f"{name} contains non-finite values")
        if len(np.unique(self.patch_id)) != patches:
            raise ValueError("fingerprint patch IDs must be unique")
        if depths < 3 or np.any(np.diff(self.depth_offsets_voxels) <= 0.0):
            raise ValueError("fingerprint depth offsets must be strictly increasing")
        if np.any(self.density < 0.0):
            raise ValueError("fingerprint density must be nonnegative")
        if np.any(np.abs(self.orientation_moment) > self.density + 2.0e-4):
            raise ValueError("orientation moment must remain bounded by density")
        if np.any(self.support_low_voxels >= self.support_high_voxels):
            raise ValueError("fingerprint support intervals must be nonempty")
        if patches and np.any(
            np.abs(np.linalg.norm(self.normal_xyz, axis=1) - 1.0) > 2.0e-4
        ):
            raise ValueError("fingerprint normals must be unit axes")

    def arrays(self) -> dict[str, np.ndarray]:
        self.validate()
        return {
            "patchId": self.patch_id.astype(np.uint64, copy=False),
            "anchorValid": self.anchor_valid.astype(bool, copy=False),
            "anchorShardIndex": self.anchor_shard_index.astype(np.int16, copy=False),
            "anchorModeIndex": self.anchor_mode_index.astype(np.int32, copy=False),
            "anchorHeightResidualVoxels": self.anchor_height_residual_voxels.astype(
                np.float32, copy=False
            ),
            "anchorNormalResidualDegrees": self.anchor_normal_residual_degrees.astype(
                np.float32, copy=False
            ),
            "anchorFiberResidualDegrees": self.anchor_fiber_residual_degrees.astype(
                np.float32, copy=False
            ),
            "contextModeCount": self.context_mode_count.astype(np.uint16, copy=False),
            "supportLowVoxels": self.support_low_voxels.astype(np.float32, copy=False),
            "supportHighVoxels": self.support_high_voxels.astype(np.float32, copy=False),
            "normalXYZ": self.normal_xyz.astype(np.float32, copy=False),
            "depthOffsetsVoxels": self.depth_offsets_voxels.astype(
                np.float32, copy=False
            ),
            "density": self.density.astype(np.float32, copy=False),
            "orientationMoment": self.orientation_moment.astype(
                np.float32, copy=False
            ),
        }


def _mode_cell_lookup(
    mode_tables: Mapping[str, LayerModeTable],
) -> dict[tuple[int, int, int], tuple[int, LayerModeTable, int]]:
    result: dict[tuple[int, int, int], tuple[int, LayerModeTable, int]] = {}
    for shard_index, shard_id in enumerate(sorted(mode_tables)):
        table = mode_tables[shard_id]
        table.validate()
        for cell_index, values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in values)
            if cell in result:
                raise ValueError(f"mode-bank cell {cell} is owned by multiple shards")
            result[cell] = (shard_index, table, cell_index)
    return result


def _mode_weight(
    table: LayerModeTable, mode_index: int, raw_settings: RawAcusSettings
) -> float:
    support = float(
        np.clip(
            table.effective_support[mode_index]
            / max(raw_settings.minimum_needles_per_cell, 1),
            0.0,
            1.0,
        )
    )
    terms = np.clip(
        np.asarray(
            (
                table.evidence_score[mode_index],
                table.material_probability[mode_index],
                table.confidence[mode_index],
                support,
            ),
            dtype=np.float64,
        ),
        0.0,
        1.0,
    )
    return float(np.prod(terms) ** 0.25)


def _sheet_evidence_mode_tables(
    evidence: BlockSheetEvidence,
    target_grid: GridSpec,
) -> dict[str, LayerModeTable]:
    """Adapt one combined evidence bank into the established fingerprint input.

    Sheet evidence preserves the complete Acus mode measurements, but owned
    graph crops rebase cell indices while retaining world geometry and stable
    mode IDs.  This adapter resolves that integer grid translation once and
    exposes one ordinary mode table, allowing the same fingerprint algorithm
    to operate on single-bank and composed multi-block evidence.
    """

    evidence.validate()
    if (
        evidence.grid.coordinate_unit != target_grid.coordinate_unit
        or not np.allclose(
            evidence.grid.cell_size_xyz,
            target_grid.cell_size_xyz,
            atol=1.0e-10,
            rtol=0.0,
        )
    ):
        raise ValueError("sheet evidence and selected graph use incompatible grids")
    cell_size = np.asarray(target_grid.cell_size_xyz, dtype=np.float64)
    offset_float = (
        np.asarray(target_grid.origin_xyz, dtype=np.float64)
        - np.asarray(evidence.grid.origin_xyz, dtype=np.float64)
    ) / cell_size
    offset = np.rint(offset_float).astype(np.int64)
    if not np.allclose(offset_float, offset, atol=1.0e-7, rtol=0.0):
        raise ValueError("sheet evidence and selected graph origins are not cell-aligned")

    arrays = evidence.arrays
    source_cells = np.asarray(arrays["modeCellXYZ"], dtype=np.int64)
    target_cells = source_cells - offset[None, :]
    inside = np.all(target_cells >= 0, axis=1) & np.all(
        target_cells
        < np.asarray(target_grid.shape_cells_xyz, dtype=np.int64)[None, :],
        axis=1,
    )
    valid = inside & (np.asarray(arrays["modeGeometryStatus"]) == 0)
    indices_by_cell: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for mode_index in np.flatnonzero(valid):
        cell = tuple(int(value) for value in target_cells[mode_index])
        indices_by_cell[cell].append(int(mode_index))
    cells = tuple(sorted(indices_by_cell))
    indices = np.asarray(
        [index for cell in cells for index in indices_by_cell[cell]],
        dtype=np.int64,
    )
    offsets = np.zeros(len(cells) + 1, dtype=np.uint64)
    for cell_index, cell in enumerate(cells):
        offsets[cell_index + 1] = offsets[cell_index] + len(indices_by_cell[cell])
    count = len(indices)
    table = LayerModeTable(
        cell_xyz=np.asarray(cells, dtype=np.int32).reshape(len(cells), 3),
        mode_offset=offsets,
        normal_hypothesis=np.asarray(
            arrays["modeNormalHypothesis"][indices], dtype=np.int8
        ),
        normal_xyz=np.asarray(arrays["modeNormalXYZ"][indices], dtype=np.float32),
        height=np.asarray(arrays["modeHeight"][indices], dtype=np.float32),
        covariance=np.asarray(
            arrays["modeCovariance"][indices], dtype=np.float32
        ),
        fiber_xyz=np.asarray(arrays["modeFiberXYZ"][indices], dtype=np.float32),
        fiber_angular_std_radians=np.asarray(
            arrays["modeFiberAngularStdRadians"][indices], dtype=np.float32
        ),
        confidence=np.asarray(
            arrays["modeConfidence"][indices], dtype=np.float32
        ),
        source_depth_voxels=np.zeros(count, dtype=np.float32),
        source_orientation_degrees=np.zeros(count, dtype=np.float32),
        evidence_score=np.asarray(
            arrays["modeEvidenceScore"][indices], dtype=np.float32
        ),
        material_probability=np.asarray(
            arrays["modeMaterialProbability"][indices], dtype=np.float32
        ),
        effective_support=np.asarray(
            arrays["modeEffectiveSupport"][indices], dtype=np.float32
        ),
    )
    table.validate()
    return {"combined-sheet-evidence": table}


def _raw_settings_from_sheet_evidence(
    evidence_root: Path,
) -> tuple[RawAcusSettings, tuple[dict[str, Any], ...]]:
    """Resolve and verify the common immutable Acus settings behind a bank.

    Subblock evidence points at its parent contract; the parent records every
    source saturation bank.  All composed inputs must use identical Acus
    settings because their depth kernels are compared in one physical frame.
    """

    current = evidence_root.resolve()
    visited: set[Path] = set()
    inputs: list[Mapping[str, Any]] | None = None
    while inputs is None:
        if current in visited:
            raise ValueError("sheet-evidence source chain contains a cycle")
        visited.add(current)
        manifest = json.loads((current / "sheet-evidence-v1.json").read_text())
        identity = manifest.get("identity", {})
        declared_inputs = identity.get("inputs")
        if declared_inputs is not None:
            inputs = list(declared_inputs)
            break
        parent = identity.get("sourceRoot")
        if parent is None:
            raise ValueError("sheet evidence does not identify its Acus sources")
        current = Path(parent).resolve()

    records: list[dict[str, Any]] = []
    settings_records: list[dict[str, Any]] = []
    for value in inputs:
        candidate_root = Path(str(value["candidateRoot"])).resolve()
        variant = json.loads((candidate_root / "variant.json").read_text())
        variant_identity = variant.get("identity", {})
        pipeline_root_value = (
            variant_identity.get("pipelineRoot")
            or variant.get("pipelineRoot")
            or variant_identity.get("inputRoot")
            or variant.get("inputRoot")
        )
        if pipeline_root_value is None:
            raise ValueError("sheet-evidence candidate lacks a pipeline root")
        pipeline_root, pipeline = resolve_pipeline_manifest(
            Path(str(pipeline_root_value)).resolve()
        )
        settings_record = dict(pipeline["identity"]["settings"])
        settings_records.append(settings_record)
        records.append(
            {
                "candidateRoot": str(candidate_root),
                "pipelineRoot": str(pipeline_root),
                "pipelineIdentitySha256": pipeline["identity"]["identitySha256"],
                "settingsSha256": canonical_json_hash(settings_record),
            }
        )
    if not settings_records:
        raise ValueError("sheet evidence contains no source Acus settings")
    reference = canonical_json_hash(settings_records[0])
    if any(canonical_json_hash(value) != reference for value in settings_records[1:]):
        raise ValueError("composed sheet-evidence inputs use different Acus settings")
    return RawAcusSettings(**settings_records[0]), tuple(records)


def _candidate_restitch_source(
    candidate_root: Path,
    selected_root: Path,
) -> dict[str, Any]:
    """Validate and identify the complete correspondence universe to score."""

    manifest_path = candidate_root / "sheet-restitch-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "pareidolia.cubical-sheet-restitch"
        or manifest.get("state") != "complete"
    ):
        raise ValueError("candidate source is not a complete sheet restitch")
    selected_hashes = {
        "manifest": sha256_file(selected_root / "selected-patches-v1.json"),
        "data": sha256_file(selected_root / "selected-patches-v1.npz"),
    }
    source_hashes = {
        "manifest": sha256_file(candidate_root / "selected-patches-v1.json"),
        "data": sha256_file(candidate_root / "selected-patches-v1.npz"),
    }
    if selected_hashes != source_hashes:
        raise ValueError(
            "candidate restitch and stratigraphic graph have different patches"
        )
    candidate_manifest = candidate_root / "sheet-join-catalog-v1.json"
    candidate_data = candidate_root / "sheet-join-catalog-v1.npz"
    return {
        "kind": "sheet-restitch-candidate-universe",
        "root": str(candidate_root),
        "manifestSha256": sha256_file(manifest_path),
        "candidateManifestSha256": sha256_file(candidate_manifest),
        "candidateDataSha256": sha256_file(candidate_data),
        "selectedPatchManifestSha256": source_hashes["manifest"],
        "selectedPatchDataSha256": source_hashes["data"],
    }


def _candidate_catalog_for_block(
    candidate_root: Path,
    block: SurfaceBlock,
) -> Any:
    """Reconstruct an auditable candidate catalog under its frozen policy."""

    from .matching import TraceMatchSettings
    from .sheet_stitching import (
        SheetMatchingPolicy,
        SheetStitchingSettings,
        enumerate_sheet_join_catalog,
    )

    manifest = json.loads((candidate_root / "sheet-restitch-v1.json").read_text())
    identity = manifest["identity"]
    policy_record = identity["policy"]
    policy = SheetMatchingPolicy(
        TraceMatchSettings(**policy_record["strictSettings"]),
        bool(policy_record["quarterTurnEnabled"]),
        float(policy_record["maximumQuarterTurnNormalDegrees"]),
        float(policy_record["maximumQuarterTurnFiberDegrees"]),
    )
    stitching_settings = SheetStitchingSettings(**identity["settings"])
    return enumerate_sheet_join_catalog(
        block,
        policy,
        settings=stitching_settings,
        allow_retained_policy_rejection=False,
    )


def build_patch_fingerprints(
    patches: PatchTable,
    mode_tables: Mapping[str, LayerModeTable],
    raw_settings: RawAcusSettings,
    settings: StratigraphicContinuitySettings | None = None,
    *,
    progress: Callable[[int, int], None] | None = None,
) -> tuple[PatchFingerprintTable, dict[str, Any]]:
    """Anchor the full local Acus mode distribution on each selected patch."""

    resolved = settings or StratigraphicContinuitySettings()
    patches.validate()
    modes_by_cell = _mode_cell_lookup(mode_tables)
    selected_cells = {
        tuple(int(value) for value in cell) for cell in patches.cell_xyz
    }
    missing = selected_cells - set(modes_by_cell)
    if missing:
        raise ValueError(f"mode bank is missing {len(missing)} selected cells")

    depth_step = float(raw_settings.depth_bin_voxels)
    maximum_relative_depth = float(np.linalg.norm(patches.grid.cell_size_xyz))
    extent_steps = int(math.ceil(maximum_relative_depth / depth_step)) + 1
    depths = (
        np.arange(-extent_steps, extent_steps + 1, dtype=np.float64) * depth_step
    )
    count = patches.patch_count
    anchor_valid = np.zeros(count, dtype=bool)
    anchor_shard_index = np.full(count, -1, dtype=np.int16)
    anchor_mode_index = np.full(count, -1, dtype=np.int32)
    anchor_height_residual = np.full(count, 1.0e6, dtype=np.float32)
    anchor_normal_residual = np.full(count, 180.0, dtype=np.float32)
    anchor_fiber_residual = np.full(count, 90.0, dtype=np.float32)
    context_mode_count = np.zeros(count, dtype=np.uint16)
    support_low = np.empty(count, dtype=np.float32)
    support_high = np.empty(count, dtype=np.float32)
    density = np.zeros((count, len(depths)), dtype=np.float32)
    orientation_moment = np.zeros_like(density)
    cell_sizes = np.asarray(patches.grid.cell_size_xyz, dtype=np.float64)

    for patch_index in range(count):
        cell = tuple(int(value) for value in patches.cell_xyz[patch_index])
        shard_index, table, cell_index = modes_by_cell[cell]
        normal = np.asarray(patches.normal_xyz[patch_index], dtype=np.float64)
        height = float(patches.height[patch_index])
        fiber = np.asarray(patches.fiber_xyz[patch_index], dtype=np.float64)
        family = int(patches.normal_family[patch_index])
        radius = 0.5 * float(np.sum(cell_sizes * np.abs(normal)))
        support_low[patch_index] = -radius - height
        support_high[patch_index] = radius - height
        family_modes = [
            mode_index
            for mode_index in table.mode_indices_for_cell(cell_index)
            if int(table.normal_hypothesis[mode_index]) == family
        ]
        if not family_modes or not np.all(np.isfinite(fiber)):
            if progress is not None:
                progress(patch_index + 1, count)
            continue

        candidates: list[tuple[float, float, float, int]] = []
        for mode_index in family_modes:
            mode_normal = np.asarray(table.normal_xyz[mode_index], dtype=np.float64)
            mode_height = float(table.height[mode_index])
            normal_dot = float(np.dot(normal, mode_normal))
            if normal_dot < 0.0:
                mode_height = -mode_height
            height_residual = abs(mode_height - height)
            normal_residual = math.degrees(
                math.acos(float(np.clip(abs(normal_dot), 0.0, 1.0)))
            )
            fiber_residual = math.degrees(
                axial_angle_radians(fiber, table.fiber_xyz[mode_index])
            )
            candidates.append(
                (
                    height_residual,
                    normal_residual,
                    fiber_residual,
                    mode_index,
                )
            )
        best = min(candidates)
        anchor_height_residual[patch_index] = best[0]
        anchor_normal_residual[patch_index] = best[1]
        anchor_fiber_residual[patch_index] = best[2]
        valid = bool(
            best[0] <= resolved.maximum_anchor_height_residual_voxels
            and best[1] <= resolved.maximum_anchor_normal_residual_degrees
            and best[2] <= resolved.maximum_anchor_fiber_residual_degrees
        )
        if not valid:
            if progress is not None:
                progress(patch_index + 1, count)
            continue
        anchor = best[3]
        anchor_valid[patch_index] = True
        anchor_shard_index[patch_index] = shard_index
        anchor_mode_index[patch_index] = anchor

        retained_modes = 0
        for mode_index in family_modes:
            if mode_index == anchor:
                continue
            mode_normal = np.asarray(table.normal_xyz[mode_index], dtype=np.float64)
            mode_height = float(table.height[mode_index])
            denominator = float(np.dot(mode_normal, normal))
            if denominator < 0.0:
                denominator = -denominator
                mode_height = -mode_height
            if denominator <= 0.2:
                continue
            weight = _mode_weight(table, mode_index, raw_settings)
            if weight <= 1.0e-8:
                continue
            relative_depth = mode_height / denominator - height
            kernel = weight * np.exp(
                -0.5
                * (
                    (depths - relative_depth)
                    / raw_settings.depth_kernel_voxels
                )
                ** 2
            )
            fiber_cosine = abs(
                float(np.dot(fiber, table.fiber_xyz[mode_index]))
            )
            axial_orientation = 2.0 * min(fiber_cosine, 1.0) ** 2 - 1.0
            density[patch_index] += kernel.astype(np.float32)
            orientation_moment[patch_index] += (
                kernel * axial_orientation
            ).astype(np.float32)
            retained_modes += 1
        context_mode_count[patch_index] = retained_modes
        if progress is not None:
            progress(patch_index + 1, count)

    result = PatchFingerprintTable(
        patches.patch_id.copy(),
        anchor_valid,
        anchor_shard_index,
        anchor_mode_index,
        anchor_height_residual,
        anchor_normal_residual,
        anchor_fiber_residual,
        context_mode_count,
        support_low,
        support_high,
        patches.normal_xyz.copy(),
        depths.astype(np.float32),
        density,
        orientation_moment,
    )
    result.validate()
    valid = result.anchor_valid
    statistics = {
        "patches": count,
        "anchoredPatches": int(np.count_nonzero(valid)),
        "unanchoredPatches": int(np.count_nonzero(~valid)),
        "contextModes": int(np.sum(result.context_mode_count, dtype=np.int64)),
        "medianContextModesPerAnchoredPatch": (
            round(float(np.median(result.context_mode_count[valid])), 6)
            if np.any(valid)
            else None
        ),
        "maximumAnchorHeightResidualVoxels": (
            round(float(np.max(result.anchor_height_residual_voxels[valid])), 7)
            if np.any(valid)
            else None
        ),
        "maximumAnchorNormalResidualDegrees": (
            round(float(np.max(result.anchor_normal_residual_degrees[valid])), 7)
            if np.any(valid)
            else None
        ),
        "maximumAnchorFiberResidualDegrees": (
            round(float(np.max(result.anchor_fiber_residual_degrees[valid])), 7)
            if np.any(valid)
            else None
        ),
        "depthSamples": result.depth_count,
        "depthStepVoxels": depth_step,
    }
    return result, statistics


def _support_mask(
    table: PatchFingerprintTable, patch_index: int, *, reverse: bool = False
) -> np.ndarray:
    low = float(table.support_low_voxels[patch_index])
    high = float(table.support_high_voxels[patch_index])
    if reverse:
        low, high = -high, -low
    return (table.depth_offsets_voxels >= low) & (
        table.depth_offsets_voxels <= high
    )


def _score_distributions(
    depth_offsets: np.ndarray,
    first_density: np.ndarray,
    first_moment: np.ndarray,
    first_support: np.ndarray,
    second_density: np.ndarray,
    second_moment: np.ndarray,
    second_support: np.ndarray,
) -> dict[str, float] | None:
    common = first_support & second_support
    if np.count_nonzero(common) < 2:
        return None
    selected_depths = depth_offsets[common]
    common_span = float(selected_depths[-1] - selected_depths[0])
    first = np.asarray(first_density[common], dtype=np.float64)
    second = np.asarray(second_density[common], dtype=np.float64)
    first_mass = float(np.sum(first))
    second_mass = float(np.sum(second))
    if first_mass <= 1.0e-10 or second_mass <= 1.0e-10:
        return None
    first_probability = first / first_mass
    second_probability = second / second_mass
    overlap = np.sqrt(first_probability * second_probability)
    density_similarity = float(np.sum(overlap))
    first_orientation = np.divide(
        first_moment[common],
        first,
        out=np.zeros_like(first),
        where=first > 1.0e-10,
    )
    second_orientation = np.divide(
        second_moment[common],
        second,
        out=np.zeros_like(second),
        where=second > 1.0e-10,
    )
    orientation_mismatch = float(
        np.sum(overlap * np.abs(first_orientation - second_orientation) * 0.5)
    )
    density_mismatch = max(0.0, 1.0 - density_similarity)
    mismatch = float(np.clip(density_mismatch + orientation_mismatch, 0.0, 1.0))
    return {
        "mismatch": mismatch,
        "densityMismatch": density_mismatch,
        "orientationMismatch": orientation_mismatch,
        "similarity": 1.0 - mismatch,
        "commonDepthSpanVoxels": common_span,
        "firstMass": first_mass,
        "secondMass": second_mass,
    }


def score_patch_fingerprints(
    table: PatchFingerprintTable,
    first_patch_index: int,
    second_patch_index: int,
    settings: StratigraphicContinuitySettings | None = None,
    *,
    fiber_quarter_turn: bool = False,
) -> dict[str, Any]:
    """Compare anchored distributions after transporting both axial gauges.

    The orientation moment is expressed in each patch's local fiber frame.
    Rotating that frame by 90 degrees negates ``cos(2 theta)``; a quarter-turn
    continuation therefore negates the second moment before comparison.  This
    is a frame-gauge operation, not an orientation sign choice.
    """

    resolved = settings or StratigraphicContinuitySettings()
    if not (
        bool(table.anchor_valid[first_patch_index])
        and bool(table.anchor_valid[second_patch_index])
    ):
        return {"status": "unanchored-patch"}
    if min(
        int(table.context_mode_count[first_patch_index]),
        int(table.context_mode_count[second_patch_index]),
    ) < resolved.minimum_context_modes:
        return {"status": "insufficient-context-modes"}
    reverse = bool(
        float(
            np.dot(
                table.normal_xyz[first_patch_index],
                table.normal_xyz[second_patch_index],
            )
        )
        < 0.0
    )
    second_density = table.density[second_patch_index]
    second_moment = table.orientation_moment[second_patch_index]
    if reverse:
        second_density = second_density[::-1]
        second_moment = second_moment[::-1]
    if fiber_quarter_turn:
        second_moment = -second_moment
    score = _score_distributions(
        table.depth_offsets_voxels,
        table.density[first_patch_index],
        table.orientation_moment[first_patch_index],
        _support_mask(table, first_patch_index),
        second_density,
        second_moment,
        _support_mask(table, second_patch_index, reverse=reverse),
    )
    if score is None:
        return {"status": "insufficient-common-density", "normalGaugeReversed": reverse}
    if score["commonDepthSpanVoxels"] < resolved.minimum_common_depth_span_voxels:
        return {
            "status": "insufficient-common-depth",
            "normalGaugeReversed": reverse,
            **score,
        }
    return {"status": "scored", "normalGaugeReversed": reverse, **score}


def _side_patch_gauges(
    start_patch_id: int,
    excluded_join_key: Hashable | None,
    face_axis: int,
    face_coordinate: int,
    lower_side: bool,
    radius: int,
    adjacency: Mapping[int, list[tuple[int, Hashable, bool]]],
    cell_by_patch: Mapping[int, tuple[int, int, int]],
    *,
    start_fiber_quarter_turn: bool = False,
) -> dict[int, bool]:
    gauge = {start_patch_id: start_fiber_quarter_turn}
    frontier = {start_patch_id}
    for _ in range(radius):
        following: set[int] = set()
        for patch_id in frontier:
            for neighbor_id, join_key, quarter_turn in adjacency[patch_id]:
                if join_key == excluded_join_key:
                    continue
                coordinate = cell_by_patch[neighbor_id][face_axis]
                inside = (
                    coordinate < face_coordinate
                    if lower_side
                    else coordinate >= face_coordinate
                )
                if inside:
                    neighbor_gauge = gauge[patch_id] ^ quarter_turn
                    if neighbor_id in gauge:
                        if gauge[neighbor_id] != neighbor_gauge:
                            raise ValueError(
                                "sheet neighborhood has contradictory fiber-frame parity"
                            )
                        continue
                    gauge[neighbor_id] = neighbor_gauge
                    following.add(neighbor_id)
        frontier = following
    return gauge


def _aggregate_fingerprints(
    table: PatchFingerprintTable,
    patch_indices_and_fiber_gauges: list[tuple[int, bool]],
    reference_normal: np.ndarray,
    settings: StratigraphicContinuitySettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int] | None:
    densities: list[np.ndarray] = []
    moments: list[np.ndarray] = []
    supports: list[np.ndarray] = []
    for patch_index, fiber_quarter_turn in patch_indices_and_fiber_gauges:
        if not bool(table.anchor_valid[patch_index]) or int(
            table.context_mode_count[patch_index]
        ) < settings.minimum_context_modes:
            continue
        reverse = bool(
            float(np.dot(table.normal_xyz[patch_index], reference_normal)) < 0.0
        )
        density = table.density[patch_index]
        moment = table.orientation_moment[patch_index]
        support = _support_mask(table, patch_index, reverse=reverse)
        if reverse:
            density = density[::-1]
            moment = moment[::-1]
        if fiber_quarter_turn:
            moment = -moment
        mass = float(np.sum(density[support]))
        if mass <= 1.0e-10:
            continue
        densities.append(np.asarray(density, dtype=np.float64) / mass)
        moments.append(np.asarray(moment, dtype=np.float64) / mass)
        supports.append(support)
    if not densities:
        return None
    return (
        np.mean(densities, axis=0),
        np.mean(moments, axis=0),
        np.mean(supports, axis=0),
        len(densities),
    )


def _score_join_neighborhood(
    match: Any,
    excluded_join_key: JoinKey | None,
    block: SurfaceBlock,
    fingerprint_table: PatchFingerprintTable,
    fingerprint_index_by_patch: Mapping[int, int],
    adjacency: Mapping[int, list[tuple[int, JoinKey, bool]]],
    cell_by_patch: Mapping[int, tuple[int, int, int]],
    settings: StratigraphicContinuitySettings,
) -> dict[str, Any]:
    first_cell = cell_by_patch[match.first_patch_id]
    second_cell = cell_by_patch[match.second_patch_id]
    face_coordinate = match.face.anchor_xyz[match.face.axis]
    first_lower = first_cell[match.face.axis] < face_coordinate
    second_lower = second_cell[match.face.axis] < face_coordinate
    if first_lower == second_lower:
        raise ValueError("joined patches do not lie on opposite sides of their face")
    first_gauges = _side_patch_gauges(
        match.first_patch_id,
        excluded_join_key,
        match.face.axis,
        face_coordinate,
        first_lower,
        settings.neighborhood_radius_hops,
        adjacency,
        cell_by_patch,
    )
    second_gauges = _side_patch_gauges(
        match.second_patch_id,
        excluded_join_key,
        match.face.axis,
        face_coordinate,
        second_lower,
        settings.neighborhood_radius_hops,
        adjacency,
        cell_by_patch,
        start_fiber_quarter_turn=bool(match.fiber_quarter_turn),
    )
    reference_normal = fingerprint_table.normal_xyz[
        fingerprint_index_by_patch[match.first_patch_id]
    ]
    first = _aggregate_fingerprints(
        fingerprint_table,
        [
            (fingerprint_index_by_patch[value], first_gauges[value])
            for value in sorted(first_gauges)
        ],
        reference_normal,
        settings,
    )
    second = _aggregate_fingerprints(
        fingerprint_table,
        [
            (fingerprint_index_by_patch[value], second_gauges[value])
            for value in sorted(second_gauges)
        ],
        reference_normal,
        settings,
    )
    if first is None or second is None:
        return {"status": "insufficient-neighborhood-context"}
    first_density, first_moment, first_coverage, first_count = first
    second_density, second_moment, second_coverage, second_count = second
    if min(first_count, second_count) < settings.minimum_side_patches:
        return {
            "status": "insufficient-side-patches",
            "firstSidePatchCount": first_count,
            "secondSidePatchCount": second_count,
        }
    score = _score_distributions(
        fingerprint_table.depth_offsets_voxels,
        first_density,
        first_moment,
        first_coverage >= settings.minimum_coverage_fraction,
        second_density,
        second_moment,
        second_coverage >= settings.minimum_coverage_fraction,
    )
    if score is None:
        return {
            "status": "insufficient-neighborhood-density",
            "firstSidePatchCount": first_count,
            "secondSidePatchCount": second_count,
        }
    if score["commonDepthSpanVoxels"] < settings.minimum_common_depth_span_voxels:
        return {
            "status": "insufficient-neighborhood-depth",
            "firstSidePatchCount": first_count,
            "secondSidePatchCount": second_count,
            **score,
        }
    return {
        "status": "scored",
        "firstSidePatchCount": first_count,
        "secondSidePatchCount": second_count,
        **score,
    }


def _join_key(match: Any) -> JoinKey:
    return (
        int(match.first_patch_id),
        int(match.second_patch_id),
        int(match.face.axis),
        tuple(int(value) for value in match.face.anchor_xyz),
    )


def score_stratigraphic_continuity(
    block: SurfaceBlock,
    patches: PatchTable,
    fingerprints: PatchFingerprintTable,
    settings: StratigraphicContinuitySettings | None = None,
    *,
    matches: Iterable[Any] | None = None,
    progress: Callable[[int, int], None] | None = None,
) -> list[dict[str, Any]]:
    """Score joins locally and from independent half-space neighborhoods.

    ``block`` is the frozen context graph.  By default its retained joins are
    scored exactly as before.  Supplying ``matches`` evaluates a complete
    candidate universe against that same context: an already-retained edge is
    removed while its two sides are aggregated, whereas an alternative edge
    compares the existing half-space neighborhoods without pretending that
    the candidate has already been selected.
    """

    resolved = settings or StratigraphicContinuitySettings()
    fingerprint_index = {
        int(patch_id): index for index, patch_id in enumerate(fingerprints.patch_id)
    }
    if set(fingerprint_index) != {value.patch_id for value in block.patches}:
        raise ValueError("fingerprint and assembled patch sets differ")
    patch_index = {
        int(patch_id): index for index, patch_id in enumerate(patches.patch_id)
    }
    cell_by_patch = {
        patch_id: tuple(int(value) for value in patches.cell_xyz[index])
        for patch_id, index in patch_index.items()
    }
    retained_keys = {_join_key(value) for value in block.joins}
    adjacency: dict[int, list[tuple[int, JoinKey, bool]]] = defaultdict(list)
    for retained_match in block.joins:
        key = _join_key(retained_match)
        quarter_turn = bool(retained_match.fiber_quarter_turn)
        adjacency[retained_match.first_patch_id].append(
            (retained_match.second_patch_id, key, quarter_turn)
        )
        adjacency[retained_match.second_patch_id].append(
            (retained_match.first_patch_id, key, quarter_turn)
        )

    records: list[dict[str, Any]] = []
    scored_matches = tuple(block.joins if matches is None else matches)
    total = len(scored_matches)
    for match_index, match in enumerate(scored_matches):
        key = _join_key(match)
        local = score_patch_fingerprints(
            fingerprints,
            fingerprint_index[match.first_patch_id],
            fingerprint_index[match.second_patch_id],
            resolved,
            fiber_quarter_turn=bool(match.fiber_quarter_turn),
        )
        neighborhood = _score_join_neighborhood(
            match,
            key if key in retained_keys else None,
            block,
            fingerprints,
            fingerprint_index,
            adjacency,
            cell_by_patch,
            resolved,
        )
        records.append(
            {
                "key": key,
                "currentlyRetained": key in retained_keys,
                "geometryScore": float(match.score),
                "local": local,
                "neighborhood": neighborhood,
            }
        )
        if progress is not None:
            progress(match_index + 1, total)
    records.sort(key=lambda value: value["key"])
    return records


def _robust_calibration(values: np.ndarray, minimum_scale: float) -> tuple[float, float]:
    center = float(np.median(values))
    raw_scale = 1.4826 * float(np.median(np.abs(values - center)))
    return center, max(raw_scale, minimum_scale)


def _mismatch_transform(value: float) -> float:
    return -math.log(max(1.0 - float(value), 1.0e-6))


def _calibrate_records(
    records: list[dict[str, Any]],
    settings: StratigraphicContinuitySettings,
    *,
    calibration_keys: frozenset[JoinKey] | None = None,
) -> dict[str, Any]:
    calibration: dict[str, Any] = {}
    for axis in range(3):
        scored = [
            value
            for value in records
            if value["key"][2] == axis
            and value["local"]["status"] == "scored"
            and value["neighborhood"]["status"] == "scored"
        ]
        calibrators = [
            value
            for value in scored
            if calibration_keys is None or value["key"] in calibration_keys
        ]
        if len(calibrators) < settings.minimum_calibration_joins:
            calibration[str(axis)] = {
                "state": "insufficient-sample",
                "count": len(calibrators),
                "minimumCount": settings.minimum_calibration_joins,
            }
            continue
        local_values = np.asarray(
            [
                _mismatch_transform(value["local"]["mismatch"])
                for value in calibrators
            ],
            dtype=np.float64,
        )
        neighborhood_values = np.asarray(
            [
                _mismatch_transform(value["neighborhood"]["mismatch"])
                for value in calibrators
            ],
            dtype=np.float64,
        )
        local_center, local_scale = _robust_calibration(
            local_values, settings.minimum_log_scale
        )
        neighborhood_center, neighborhood_scale = _robust_calibration(
            neighborhood_values, settings.minimum_log_scale
        )
        calibration[str(axis)] = {
            "state": "calibrated",
            "count": len(calibrators),
            "scoredCount": len(scored),
            "transform": "negative log similarity",
            "local": {
                "median": round(local_center, 7),
                "effectiveRobustScale": round(local_scale, 7),
                "threshold": round(
                    local_center
                    + settings.outlier_standard_deviations * local_scale,
                    7,
                ),
            },
            "neighborhood": {
                "median": round(neighborhood_center, 7),
                "effectiveRobustScale": round(neighborhood_scale, 7),
                "threshold": round(
                    neighborhood_center
                    + settings.outlier_standard_deviations * neighborhood_scale,
                    7,
                ),
            },
        }
        for value in scored:
            local_z = (
                _mismatch_transform(value["local"]["mismatch"]) - local_center
            ) / local_scale
            neighborhood_z = (
                _mismatch_transform(value["neighborhood"]["mismatch"])
                - neighborhood_center
            ) / neighborhood_scale
            value["localRobustOutlierZ"] = local_z
            value["neighborhoodRobustOutlierZ"] = neighborhood_z
            value["rejected"] = bool(
                local_z > settings.outlier_standard_deviations
                and neighborhood_z > settings.outlier_standard_deviations
            )
            value["rejectionReason"] = (
                "local-and-multicell-stratigraphic-outlier"
                if value["rejected"]
                else None
            )
    for value in records:
        if "rejected" not in value:
            value["localRobustOutlierZ"] = None
            value["neighborhoodRobustOutlierZ"] = None
            value["rejected"] = False
            value["rejectionReason"] = None
    return calibration


def _write_fingerprint_artifact(
    prefix: Path,
    table: PatchFingerprintTable,
    *,
    identity_sha256: str,
    shard_ids: list[str],
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    data_path = prefix.with_suffix(".npz")
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **table.arrays())
    temporary.replace(data_path)
    payload = {
        "schema": FINGERPRINT_SCHEMA,
        "version": FINGERPRINT_VERSION,
        "identitySha256": identity_sha256,
        "shardIds": shard_ids,
        "statistics": dict(statistics),
        "model": {
            "anchor": "selected patch matched to its exact same-family full-bank mode",
            "depth": "signed relative distance along the selected patch normal gauge",
            "orientation": (
                "axial second-harmonic fiber agreement relative to the anchor; "
                "quarter-turn joins transport its sign as a Z2 frame gauge"
            ),
            "weight": (
                "geometric mean of mode evidence, material probability, confidence, "
                "and effective-support saturation"
            ),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(prefix.with_suffix(".json"), payload)
    return payload


def read_patch_fingerprints(
    prefix: str | Path,
    *,
    identity_sha256: str | None = None,
    verify: bool = True,
) -> PatchFingerprintTable:
    """Load one complete fixed-width selected-patch fingerprint artifact."""

    path = Path(prefix)
    manifest_path = path.with_suffix(".json")
    data_path = path.with_suffix(".npz")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != FINGERPRINT_SCHEMA
        or int(manifest.get("version", -1)) != FINGERPRINT_VERSION
    ):
        raise ValueError("fingerprint artifact schema/version mismatch")
    if (
        identity_sha256 is not None
        and manifest.get("identitySha256") != identity_sha256
    ):
        raise ValueError("fingerprint artifact identity mismatch")
    if verify and manifest.get("data", {}).get("sha256") != sha256_file(data_path):
        raise ValueError("fingerprint artifact data hash mismatch")
    with np.load(data_path) as values:
        table = PatchFingerprintTable(
            patch_id=values["patchId"].copy(),
            anchor_valid=values["anchorValid"].copy(),
            anchor_shard_index=values["anchorShardIndex"].copy(),
            anchor_mode_index=values["anchorModeIndex"].copy(),
            anchor_height_residual_voxels=values[
                "anchorHeightResidualVoxels"
            ].copy(),
            anchor_normal_residual_degrees=values[
                "anchorNormalResidualDegrees"
            ].copy(),
            anchor_fiber_residual_degrees=values[
                "anchorFiberResidualDegrees"
            ].copy(),
            context_mode_count=values["contextModeCount"].copy(),
            support_low_voxels=values["supportLowVoxels"].copy(),
            support_high_voxels=values["supportHighVoxels"].copy(),
            normal_xyz=values["normalXYZ"].copy(),
            depth_offsets_voxels=values["depthOffsetsVoxels"].copy(),
            density=values["density"].copy(),
            orientation_moment=values["orientationMoment"].copy(),
        )
    table.validate()
    return table


def _optional_float(value: float | None) -> float:
    return float(value) if value is not None else math.nan


def _write_join_table(path: Path, records: list[dict[str, Any]]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    keys = [value["key"] for value in records]
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray([value[0] for value in keys], dtype=np.uint64),
            secondPatchId=np.asarray([value[1] for value in keys], dtype=np.uint64),
            faceAxis=np.asarray([value[2] for value in keys], dtype=np.int8),
            faceAnchorXYZ=np.asarray([value[3] for value in keys], dtype=np.int32),
            geometryScore=np.asarray(
                [value["geometryScore"] for value in records], dtype=np.float32
            ),
            localScored=np.asarray(
                [value["local"]["status"] == "scored" for value in records],
                dtype=bool,
            ),
            localMismatch=np.asarray(
                [value["local"].get("mismatch", np.nan) for value in records],
                dtype=np.float32,
            ),
            localDensityMismatch=np.asarray(
                [
                    value["local"].get("densityMismatch", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            localOrientationMismatch=np.asarray(
                [
                    value["local"].get("orientationMismatch", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            localCommonDepthSpanVoxels=np.asarray(
                [
                    value["local"].get("commonDepthSpanVoxels", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            normalGaugeReversed=np.asarray(
                [value["local"].get("normalGaugeReversed", False) for value in records],
                dtype=bool,
            ),
            neighborhoodScored=np.asarray(
                [
                    value["neighborhood"]["status"] == "scored"
                    for value in records
                ],
                dtype=bool,
            ),
            neighborhoodMismatch=np.asarray(
                [
                    value["neighborhood"].get("mismatch", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            neighborhoodDensityMismatch=np.asarray(
                [
                    value["neighborhood"].get("densityMismatch", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            neighborhoodOrientationMismatch=np.asarray(
                [
                    value["neighborhood"].get("orientationMismatch", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            neighborhoodCommonDepthSpanVoxels=np.asarray(
                [
                    value["neighborhood"].get("commonDepthSpanVoxels", np.nan)
                    for value in records
                ],
                dtype=np.float32,
            ),
            firstSidePatchCount=np.asarray(
                [
                    value["neighborhood"].get("firstSidePatchCount", 0)
                    for value in records
                ],
                dtype=np.uint16,
            ),
            secondSidePatchCount=np.asarray(
                [
                    value["neighborhood"].get("secondSidePatchCount", 0)
                    for value in records
                ],
                dtype=np.uint16,
            ),
            localRobustOutlierZ=np.asarray(
                [
                    _optional_float(value["localRobustOutlierZ"])
                    for value in records
                ],
                dtype=np.float32,
            ),
            neighborhoodRobustOutlierZ=np.asarray(
                [
                    _optional_float(value["neighborhoodRobustOutlierZ"])
                    for value in records
                ],
                dtype=np.float32,
            ),
            currentlyRetained=np.asarray(
                [value.get("currentlyRetained", True) for value in records],
                dtype=bool,
            ),
            retained=np.asarray(
                [not value["rejected"] for value in records], dtype=bool
            ),
        )
    temporary.replace(path)


def _resolve_pipeline(root: Path) -> tuple[Path, dict[str, Any]]:
    return resolve_pipeline_manifest(root)


def _identity(
    root: Path,
    mode_source: Mapping[str, Any],
    raw_settings: RawAcusSettings,
    settings: StratigraphicContinuitySettings,
    join_refinement_root: Path | None,
    candidate_source: Mapping[str, Any] | None,
) -> dict[str, Any]:
    implementation_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": STRATIGRAPHIC_CONTINUITY_SCHEMA,
        "version": STRATIGRAPHIC_CONTINUITY_VERSION,
        "inputRoot": str(root),
        "inputPatchManifestSha256": sha256_file(root / "selected-patches-v1.json"),
        "inputPatchDataSha256": sha256_file(root / "selected-patches-v1.npz"),
        "inputSurfaceGraph": (
            {
                "manifestSha256": sha256_file(root / "surface-graph-v1.json"),
                "dataSha256": sha256_file(root / "surface-graph-v1.npz"),
            }
            if (root / "surface-graph-v1.json").is_file()
            else None
        ),
        "modeSource": dict(mode_source),
        "rawAcusSettings": asdict(raw_settings),
        "settings": settings.record(),
        "joinRefinementRoot": (
            str(join_refinement_root) if join_refinement_root is not None else None
        ),
        "candidateSource": (
            dict(candidate_source) if candidate_source is not None else None
        ),
        "implementationSha256": {
            name: sha256_file(implementation_root / name)
            for name in (
                "stratigraphic_continuity.py",
                "mode_bank.py",
                "sheet_evidence.py",
                "stratigraphy.py",
                "block.py",
                "continuity.py",
                "contracts.py",
                "geometry.py",
                "surface_graph.py",
                "tables.py",
                "matching.py",
                "sheet_stitching.py",
            )
        },
    }
    if join_refinement_root is not None:
        payload["joinRefinementSha256"] = {
            "manifest": sha256_file(join_refinement_root / "refinement.json"),
            "table": sha256_file(join_refinement_root / "join-continuity-v1.npz"),
        }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _component_summary(block: SurfaceBlock, limit: int = 20) -> dict[str, Any]:
    ordered = sorted(
        block.components, key=lambda value: (-len(value.patch_ids), value.component_id)
    )
    return {
        "componentCount": len(ordered),
        "top": [
            {
                "rank": rank,
                "componentId": value.component_id,
                "patchCount": len(value.patch_ids),
            }
            for rank, value in enumerate(ordered[:limit], start=1)
        ],
    }


def _percentiles(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    array = np.asarray(values, dtype=np.float64)
    return {
        name: round(float(np.percentile(array, quantile)), 7)
        for name, quantile in (
            ("minimum", 0),
            ("p10", 10),
            ("median", 50),
            ("p90", 90),
            ("p95", 95),
            ("p99", 99),
            ("maximum", 100),
        )
    }


def run_stratigraphic_continuity_refinement(
    input_root: str | Path,
    mode_bank_root: str | Path | None,
    output_root: str | Path,
    *,
    sheet_evidence_root: str | Path | None = None,
    candidate_restitch_root: str | Path | None = None,
    settings: StratigraphicContinuitySettings | None = None,
    join_refinement_root: str | Path | None = None,
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3),
    force: bool = False,
    fingerprint_progress: Callable[[int, int], None] | None = None,
    join_progress: Callable[[int, int], None] | None = None,
) -> dict[str, Any]:
    """Refine joins only when local and multi-cell Acus signatures disagree.

    Exactly one complete mode source is required. ``mode_bank_root`` preserves
    the original single-pipeline path; ``sheet_evidence_root`` consumes the
    composed stable-ID bank used by block-level inference.  An optional frozen
    restitch source expands the same test from retained joins to every
    geometrically admissible correspondence without changing its calibration
    population.
    """

    started = time.monotonic()
    resolved = settings or StratigraphicContinuitySettings()
    root = Path(input_root).resolve()
    bank_root = (
        Path(mode_bank_root).resolve() if mode_bank_root is not None else None
    )
    evidence_root = (
        Path(sheet_evidence_root).resolve()
        if sheet_evidence_root is not None
        else None
    )
    if (bank_root is None) == (evidence_root is None):
        raise ValueError("choose exactly one mode bank or sheet-evidence source")
    output = Path(output_root).resolve()
    prior_root = (
        Path(join_refinement_root).resolve()
        if join_refinement_root is not None
        else None
    )
    candidate_root = (
        Path(candidate_restitch_root).resolve()
        if candidate_restitch_root is not None
        else None
    )
    mode_root = bank_root if bank_root is not None else evidence_root
    assert mode_root is not None
    input_roots = {root, mode_root}
    if prior_root is not None:
        input_roots.add(prior_root)
    if candidate_root is not None:
        input_roots.add(candidate_root)
    if output in input_roots:
        raise ValueError("stratigraphic output must differ from every input root")
    candidate_source = (
        _candidate_restitch_source(candidate_root, root)
        if candidate_root is not None
        else None
    )
    evidence: BlockSheetEvidence | None = None
    source_pipeline_records: tuple[dict[str, Any], ...] = tuple()
    if bank_root is not None:
        _, pipeline = _resolve_pipeline(root)
        bank_manifest = json.loads((bank_root / "mode-bank.json").read_text())
        if (
            bank_manifest.get("schema") != MODE_BANK_SCHEMA
            or bank_manifest.get("state") != "complete"
        ):
            raise ValueError("stratigraphic continuity requires a complete mode bank")
        pipeline_identity = str(pipeline["identity"]["identitySha256"])
        if (
            bank_manifest["identity"]["inputPipelineIdentitySha256"]
            != pipeline_identity
        ):
            raise ValueError("mode bank and selected reconstruction have different inputs")
        raw_settings = RawAcusSettings(**pipeline["identity"]["settings"])
        mode_source = {
            "kind": "mode-bank",
            "root": str(bank_root),
            "manifestSha256": sha256_file(bank_root / "mode-bank.json"),
        }
    else:
        assert evidence_root is not None
        from .sheet_evidence import read_block_sheet_evidence

        evidence = read_block_sheet_evidence(evidence_root, verify=True)
        raw_settings, source_pipeline_records = _raw_settings_from_sheet_evidence(
            evidence_root
        )
        evidence_manifest = evidence_root / "sheet-evidence-v1.json"
        evidence_data = evidence_root / str(evidence.manifest["data"]["path"])
        mode_source = {
            "kind": "block-sheet-evidence",
            "root": str(evidence_root),
            "manifestSha256": sha256_file(evidence_manifest),
            "dataSha256": sha256_file(evidence_data),
            "modePatchManifestSha256": sha256_file(
                evidence_root / "mode-patches-v1.json"
            ),
            "modePatchDataSha256": sha256_file(
                evidence_root / "mode-patches-v1.npz"
            ),
            "sourcePipelines": list(source_pipeline_records),
        }
    identity = _identity(
        root,
        mode_source,
        raw_settings,
        resolved,
        prior_root,
        candidate_source,
    )
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "stratigraphic-refinement.json"
    if manifest_path.is_file():
        previous = json.loads(manifest_path.read_text())
        if previous.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("stratigraphic output belongs to another identity")
        if (
            not force
            and previous.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": STRATIGRAPHIC_CONTINUITY_SCHEMA,
        "version": STRATIGRAPHIC_CONTINUITY_VERSION,
        "state": "assembling",
        "identity": identity,
        "inputRoot": str(root),
        "modeBankRoot": str(bank_root) if bank_root is not None else None,
        "sheetEvidenceRoot": (
            str(evidence_root) if evidence_root is not None else None
        ),
        "joinRefinementRoot": str(prior_root) if prior_root is not None else None,
        "candidateRestitchRoot": (
            str(candidate_root) if candidate_root is not None else None
        ),
    }
    atomic_json(manifest_path, manifest)

    patches = read_patch_shard(root / "selected-patches-v1", verify=True)
    if (root / "surface-graph-v1.json").is_file():
        baseline = read_surface_graph(root, table=patches, verify=True)
    else:
        baseline = assemble_surface_hierarchy(
            patches.grid,
            BlockBounds((0, 0, 0), patches.grid.shape_cells_xyz),
            patches.to_patches(),
            maximum_leaf_shape_cells_xyz=leaf_shape_cells_xyz,
        )
    if prior_root is not None:
        baseline = apply_join_continuity_refinement(baseline, prior_root)
    manifest["state"] = "fingerprinting"
    atomic_json(manifest_path, manifest)
    if bank_root is not None:
        _, mode_tables = load_mode_bank(bank_root, verify=True)
        shard_ids = sorted(mode_tables)
    else:
        assert evidence is not None
        valid_mode_ids = {
            int(value)
            for value, status in zip(
                evidence.arrays["modeId"],
                evidence.arrays["modeGeometryStatus"],
            )
            if int(status) == 0
        }
        missing_ids = {int(value) for value in patches.patch_id} - valid_mode_ids
        if missing_ids:
            raise ValueError(
                "selected graph contains patches absent from sheet evidence: "
                f"{sorted(missing_ids)[:2]}"
            )
        mode_tables = _sheet_evidence_mode_tables(evidence, patches.grid)
        shard_ids = sorted(mode_tables)
    fingerprints, fingerprint_statistics = build_patch_fingerprints(
        patches,
        mode_tables,
        raw_settings,
        resolved,
        progress=fingerprint_progress,
    )
    fingerprint_manifest = _write_fingerprint_artifact(
        output / "patch-stratigraphic-fingerprints-v1",
        fingerprints,
        identity_sha256=identity_sha256,
        shard_ids=shard_ids,
        statistics=fingerprint_statistics,
    )

    manifest["state"] = "scoring"
    atomic_json(manifest_path, manifest)
    baseline_keys = frozenset(_join_key(value) for value in baseline.joins)
    candidate_records: list[dict[str, Any]] | None = None
    candidate_table_path: Path | None = None
    candidate_statistics: dict[str, Any] | None = None
    if candidate_root is not None:
        catalog = _candidate_catalog_for_block(candidate_root, baseline)
        candidate_records = score_stratigraphic_continuity(
            baseline,
            patches,
            fingerprints,
            resolved,
            matches=(value.match for value in catalog.candidates),
            progress=join_progress,
        )
        calibration = _calibrate_records(
            candidate_records,
            resolved,
            calibration_keys=baseline_keys,
        )
        candidate_by_key = {
            value["key"]: value for value in candidate_records
        }
        missing_baseline = baseline_keys - set(candidate_by_key)
        if missing_baseline:
            raise ValueError(
                "candidate universe omits retained stratigraphic joins: "
                f"{sorted(missing_baseline)[:2]}"
            )
        records = [candidate_by_key[key] for key in sorted(baseline_keys)]
        candidate_table_path = output / f"{CANDIDATE_CONTINUITY_STEM}.npz"
        _write_join_table(candidate_table_path, candidate_records)
        candidate_statistics = {
            **catalog.statistics(),
            "localScoredCandidates": sum(
                value["local"]["status"] == "scored"
                for value in candidate_records
            ),
            "neighborhoodScoredCandidates": sum(
                value["neighborhood"]["status"] == "scored"
                for value in candidate_records
            ),
            "jointlyCalibratedCandidates": sum(
                value["localRobustOutlierZ"] is not None
                for value in candidate_records
            ),
            "rejectedCandidates": sum(
                value["rejected"] for value in candidate_records
            ),
            "rejectedRetainedJoins": sum(
                value["rejected"] and value["currentlyRetained"]
                for value in candidate_records
            ),
            "rejectedAlternativeCandidates": sum(
                value["rejected"] and not value["currentlyRetained"]
                for value in candidate_records
            ),
        }
    else:
        records = score_stratigraphic_continuity(
            baseline,
            patches,
            fingerprints,
            resolved,
            progress=join_progress,
        )
        calibration = _calibrate_records(records, resolved)
    record_by_key = {value["key"]: value for value in records}
    retained = [
        match
        for match in baseline.joins
        if not record_by_key[_join_key(match)]["rejected"]
    ]
    refined = rebuild_surface_block(baseline, retained)
    table_path = output / "join-stratigraphic-continuity-v1.npz"
    _write_join_table(table_path, records)
    table_sha256 = sha256_file(table_path)

    # A refinement is itself a complete, reusable graph checkpoint.  Keeping
    # the filtered graph beside its scoring table avoids an implicit
    # "input graph + modifier" state and lets another fixed-point round consume
    # the exact output directly.  Copy through a temporary path so interruption
    # cannot leave a half-written patch artifact behind.
    for name in ("selected-patches-v1.json", "selected-patches-v1.npz"):
        source_path = root / name
        target_path = output / name
        temporary_path = target_path.with_suffix(target_path.suffix + ".tmp")
        shutil.copyfile(source_path, temporary_path)
        temporary_path.replace(target_path)
    graph_manifest = write_surface_graph(
        output,
        refined,
        semantics=(
            "fixed selected Acus patch geometry after joint local and "
            "multicell complete-mode stratigraphic outlier rejection"
        ),
        provenance={
            "inputRoot": str(root),
            "stratigraphicRefinementIdentitySha256": identity_sha256,
            "stratigraphicJoinTableSha256": table_sha256,
        },
    )

    baseline_component = dict(baseline.component_by_patch)
    refined_component = dict(refined.component_by_patch)
    rejected_by_component = Counter(
        baseline_component[value["key"][0]]
        for value in records
        if value["rejected"]
    )
    baseline_to_refined: dict[int, set[int]] = defaultdict(set)
    for patch_id, component_id in baseline.component_by_patch:
        baseline_to_refined[component_id].add(refined_component[patch_id])
    split_components = {
        str(component_id): {
            "rejectedJoins": rejected_by_component[component_id],
            "refinedComponentCount": len(parts),
            "refinedComponentIds": sorted(parts),
        }
        for component_id, parts in sorted(baseline_to_refined.items())
        if len(parts) > 1
    }
    component_sizes = {
        value.component_id: len(value.patch_ids) for value in baseline.components
    }
    ranked = sorted(
        (
            value
            for value in records
            if value["localRobustOutlierZ"] is not None
            and value["neighborhoodRobustOutlierZ"] is not None
        ),
        key=lambda value: (
            min(
                value["localRobustOutlierZ"],
                value["neighborhoodRobustOutlierZ"],
            ),
            value["neighborhood"]["mismatch"],
        ),
        reverse=True,
    )
    candidates = [
        {
            "firstPatchId": value["key"][0],
            "secondPatchId": value["key"][1],
            "faceAxis": value["key"][2],
            "faceAnchorXYZ": list(value["key"][3]),
            "baselineComponentId": baseline_component[value["key"][0]],
            "baselineComponentSize": component_sizes[
                baseline_component[value["key"][0]]
            ],
            "localMismatch": round(value["local"]["mismatch"], 7),
            "neighborhoodMismatch": round(
                value["neighborhood"]["mismatch"], 7
            ),
            "localRobustOutlierZ": round(value["localRobustOutlierZ"], 5),
            "neighborhoodRobustOutlierZ": round(
                value["neighborhoodRobustOutlierZ"], 5
            ),
            "firstSidePatchCount": value["neighborhood"][
                "firstSidePatchCount"
            ],
            "secondSidePatchCount": value["neighborhood"][
                "secondSidePatchCount"
            ],
            "rejected": value["rejected"],
        }
        for value in ranked[:64]
    ]
    local_scored = [
        value["local"]["mismatch"]
        for value in records
        if value["local"]["status"] == "scored"
    ]
    neighborhood_scored = [
        value["neighborhood"]["mismatch"]
        for value in records
        if value["neighborhood"]["status"] == "scored"
    ]
    artifacts: dict[str, Any] = {
        "fingerprints": fingerprint_manifest["data"]["path"],
        "fingerprintManifest": "patch-stratigraphic-fingerprints-v1.json",
        "fingerprintSha256": fingerprint_manifest["data"]["sha256"],
        "joinTable": table_path.name,
        "joinTableSha256": table_sha256,
        "selectedPatches": {
            "manifest": "selected-patches-v1.json",
            "manifestSha256": sha256_file(output / "selected-patches-v1.json"),
            "data": "selected-patches-v1.npz",
            "dataSha256": sha256_file(output / "selected-patches-v1.npz"),
        },
        "surfaceGraph": {
            "manifest": "surface-graph-v1.json",
            "manifestSha256": sha256_file(output / "surface-graph-v1.json"),
            "data": graph_manifest["data"],
        },
    }
    if candidate_table_path is not None:
        artifacts.update(
            {
                "candidateJoinTable": candidate_table_path.name,
                "candidateJoinTableSha256": sha256_file(candidate_table_path),
            }
        )
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-stratigraphic-continuity-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(root),
        "modeBankRoot": str(bank_root) if bank_root is not None else None,
        "sheetEvidenceRoot": (
            str(evidence_root) if evidence_root is not None else None
        ),
        "modeSource": mode_source,
        "candidateSource": candidate_source,
        "joinRefinementRoot": str(prior_root) if prior_root is not None else None,
        "method": (
            "anchor every selected patch in the complete same-family Acus mode bank; "
            "compare relative depth/fiber distributions locally and after averaging "
            "graph-connected half-space neighborhoods; cut only a joint robust "
            "outer-tail disagreement at both scales"
        ),
        "directions": (
            "normal and fiber axes remain unsigned; normal sign transports depth "
            "order, while quarter-turn joins transport the cos(2 theta) orientation "
            "moment as a Z2 fiber-frame sign"
        ),
        "fingerprints": fingerprint_statistics,
        "counts": {
            "baselineJoins": len(baseline.joins),
            "localScoredJoins": len(local_scored),
            "neighborhoodScoredJoins": len(neighborhood_scored),
            "jointlyCalibratedJoins": sum(
                value["localRobustOutlierZ"] is not None for value in records
            ),
            "rejectedJoins": sum(value["rejected"] for value in records),
            "retainedJoins": len(refined.joins),
            "splitBaselineComponents": len(split_components),
        },
        "scoreDistributions": {
            "localMismatch": _percentiles(local_scored),
            "neighborhoodMismatch": _percentiles(neighborhood_scored),
        },
        "calibration": calibration,
        "candidateUniverse": candidate_statistics,
        "baseline": _component_summary(baseline),
        "refined": _component_summary(refined),
        "splitComponents": split_components,
        "rejectedByBaselineComponent": {
            str(key): value for key, value in sorted(rejected_by_component.items())
        },
        "rankedCandidates": candidates,
        "artifacts": artifacts,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["table"] = table_path.name
    manifest["tableSha256"] = summary["artifacts"]["joinTableSha256"]
    manifest["surfaceGraph"] = summary["artifacts"]["surfaceGraph"]
    if candidate_table_path is not None:
        manifest["candidateTable"] = candidate_table_path.name
        manifest["candidateTableSha256"] = summary["artifacts"][
            "candidateJoinTableSha256"
        ]
    atomic_json(manifest_path, manifest)
    return summary


def apply_stratigraphic_continuity_refinement(
    block: SurfaceBlock, refinement_root: str | Path
) -> SurfaceBlock:
    """Apply one complete stratigraphic table to its exact baseline join set."""

    root = Path(refinement_root)
    manifest = json.loads((root / "stratigraphic-refinement.json").read_text())
    if (
        manifest.get("schema") != STRATIGRAPHIC_CONTINUITY_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("stratigraphic refinement is not complete")
    table_path = root / manifest["table"]
    if sha256_file(table_path) != manifest["tableSha256"]:
        raise ValueError("stratigraphic continuity table hash mismatch")
    with np.load(table_path, allow_pickle=False) as values:
        keys = [
            (
                int(first),
                int(second),
                int(axis),
                tuple(int(value) for value in anchor),
            )
            for first, second, axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            )
        ]
        retained_flags = np.asarray(values["retained"], dtype=bool)
    current = {_join_key(value): value for value in block.joins}
    if set(keys) != set(current):
        raise ValueError("stratigraphic table and baseline join sets differ")
    retained = [current[key] for key, keep in zip(keys, retained_flags) if keep]
    return rebuild_surface_block(block, retained)


def read_candidate_stratigraphic_admissibility(
    refinement_root: str | Path,
) -> tuple[frozenset[JoinKey], frozenset[JoinKey], dict[str, Any]]:
    """Read the complete, robustly calibrated candidate admissibility gate."""

    root = Path(refinement_root)
    manifest = json.loads((root / "stratigraphic-refinement.json").read_text())
    if (
        manifest.get("schema") != STRATIGRAPHIC_CONTINUITY_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("candidateTable") is None
    ):
        raise ValueError(
            "stratigraphic refinement has no complete candidate universe"
        )
    table_path = root / str(manifest["candidateTable"])
    if sha256_file(table_path) != manifest["candidateTableSha256"]:
        raise ValueError("candidate stratigraphic table hash mismatch")
    with np.load(table_path, allow_pickle=False) as values:
        keys = tuple(
            (
                int(first),
                int(second),
                int(axis),
                tuple(int(value) for value in anchor),
            )
            for first, second, axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            )
        )
        admissible_flags = np.asarray(values["retained"], dtype=bool)
        current_flags = np.asarray(values["currentlyRetained"], dtype=bool)
        local_z = np.asarray(values["localRobustOutlierZ"], dtype=np.float64)
        neighborhood_z = np.asarray(
            values["neighborhoodRobustOutlierZ"], dtype=np.float64
        )
    if len(keys) != len(set(keys)):
        raise ValueError("candidate stratigraphic table contains duplicate joins")
    universe = frozenset(keys)
    admissible = frozenset(
        key for key, keep in zip(keys, admissible_flags) if bool(keep)
    )
    return universe, admissible, {
        "sourceRoot": str(root.resolve()),
        "identitySha256": manifest["identity"]["identitySha256"],
        "candidates": len(keys),
        "jointlyCalibratedCandidates": int(
            np.count_nonzero(np.isfinite(local_z) & np.isfinite(neighborhood_z))
        ),
        "rejectedCandidates": len(universe - admissible),
        "retainedInputCandidates": int(np.count_nonzero(current_flags)),
        "rejectedRetainedInputCandidates": int(
            np.count_nonzero(current_flags & ~admissible_flags)
        ),
        "tableSha256": manifest["candidateTableSha256"],
        "criterion": (
            "reject only joint local and multicell complete-mode robust outliers"
        ),
    }
