from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .block import BlockBounds, SurfaceBlock, assemble_surface_hierarchy
from .contracts import (
    RawAcusSettings,
    VolumeSource,
    atomic_json,
    canonical_json_hash,
    resolve_pipeline_manifest,
    sha256_file,
)
from .evidence import read_evidence_artifact
from .export import write_block_obj, write_block_projection_png
from .mode_bank import MODE_BANK_SCHEMA, load_mode_bank
from .pipeline import patch_table_from_selection, write_selection_artifact
from .saturation import (
    canonical_needle_artifact_identity,
    load_owned_canonical_needles,
)
from .selection import optimize_configurations
from .stratigraphic_continuity import build_patch_fingerprints
from .stratigraphy import (
    ConfigurationTable,
    LayerModeTable,
    _layer_reward,
    _transition_reward,
)
from .tables import read_patch_shard, write_patch_shard
from .topology import Int3


SATURATION_RESELECTION_SCHEMA = "pareidolia.cubical-saturation-reselection"
SATURATION_RESELECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class SaturationReselectionSettings:
    """Operational bounds for full-bank physical stratigraphy reselection."""

    joint_residual_limit: float = 2.5
    maximum_configurations_per_cell: int = 10
    maximum_configurations_per_coverage: int = 2
    pairwise_scale: float = 0.35
    interior_unmatched_trace_penalty: float = 0.0
    maximum_sweeps: int = 12
    leaf_shape_cells_xyz: Int3 = (4, 4, 3)

    def __post_init__(self) -> None:
        finite_positive = (self.joint_residual_limit, self.pairwise_scale)
        if any(not math.isfinite(value) or value <= 0.0 for value in finite_positive):
            raise ValueError("reselection residual and pairwise scales must be positive")
        if (
            not math.isfinite(self.interior_unmatched_trace_penalty)
            or self.interior_unmatched_trace_penalty < 0.0
        ):
            raise ValueError("unmatched trace penalty must be finite and nonnegative")
        if self.maximum_configurations_per_cell < 2:
            raise ValueError("reselection requires at least two configurations per cell")
        if self.maximum_configurations_per_coverage <= 0:
            raise ValueError("coverage diversity cap must be positive")
        if self.maximum_sweeps <= 0:
            raise ValueError("maximum sweeps must be positive")
        leaf = tuple(int(value) for value in self.leaf_shape_cells_xyz)
        if len(leaf) != 3 or any(value <= 0 for value in leaf):
            raise ValueError("leaf shape must be a positive XYZ triple")
        object.__setattr__(self, "leaf_shape_cells_xyz", leaf)

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["leaf_shape_cells_xyz"] = list(self.leaf_shape_cells_xyz)
        return values


@dataclass(frozen=True, slots=True)
class RetainedSaturationConfiguration:
    cell_xyz: Int3
    shard_id: str
    normal_hypothesis: int
    mode_indices: tuple[int, ...]
    evidence_log_score: float
    physical_log_score: float
    total_log_score: float
    covered_evidence_mass: float
    total_evidence_mass: float
    coverage_mask: int
    is_current: bool

    @property
    def covered_evidence_fraction(self) -> float:
        return self.covered_evidence_mass / max(self.total_evidence_mass, 1.0e-12)


def configuration_evidence_log_score(
    likelihood_sum: np.ndarray,
    confidence_sum: float,
    evidence_weight: np.ndarray,
    *,
    background_likelihood: float,
) -> float:
    """Evidence log score for a confidence-weighted layer mixture plus background.

    The background is fixed at the Gaussian likelihood on the configured joint
    residual contour.  Normalizing the layer mixture by its confidence mass
    prevents adding modes from increasing the score without explaining new
    evidence.
    """

    likelihood = np.asarray(likelihood_sum, dtype=np.float64)
    weights = np.asarray(evidence_weight, dtype=np.float64)
    if likelihood.shape != weights.shape:
        raise ValueError("configuration likelihood and evidence weights must align")
    if confidence_sum <= 0.0:
        return 0.0
    if not 0.0 < background_likelihood <= 1.0:
        raise ValueError("background likelihood must lie in (0, 1]")
    mixture = likelihood / confidence_sum
    return float(np.sum(weights * np.log1p(mixture / background_likelihood)))


def _mask_mass(mask: int, weights: np.ndarray, cache: dict[int, float]) -> float:
    if mask in cache:
        return cache[mask]
    value = 0.0
    remaining = mask
    while remaining:
        bit = remaining & -remaining
        value += float(weights[bit.bit_length() - 1])
        remaining ^= bit
    cache[mask] = value
    return value


def enumerate_cell_saturation_configurations(
    cell_xyz: Int3,
    shard_id: str,
    table: LayerModeTable,
    cell_index: int,
    centers_xyz: np.ndarray,
    directions_xyz: np.ndarray,
    evidence_weight: np.ndarray,
    *,
    cell_center_xyz: np.ndarray,
    normal_confidence: np.ndarray,
    current_mode_indices: tuple[int, ...],
    source: VolumeSource,
    raw_settings: RawAcusSettings,
    settings: SaturationReselectionSettings | None = None,
) -> tuple[tuple[RetainedSaturationConfiguration, ...], dict[str, Any]]:
    """Enumerate diverse physical full-bank stacks for one independently owned cell."""

    resolved = settings or SaturationReselectionSettings()
    centers = np.asarray(centers_xyz, dtype=np.float64)
    directions = np.asarray(directions_xyz, dtype=np.float64)
    weights = np.asarray(evidence_weight, dtype=np.float64)
    if centers.shape != (len(weights), 3) or directions.shape != (len(weights), 3):
        raise ValueError("cell evidence arrays must be aligned N x 3 samples")
    available = tuple(int(value) for value in table.mode_indices_for_cell(cell_index))
    available_set = set(available)
    current_set = {int(value) for value in current_mode_indices}
    if not current_set.issubset(available_set):
        raise ValueError("current selected stack is not anchored in this mode-bank cell")
    current_families = {
        int(table.normal_hypothesis[value]) for value in current_set
    }
    if len(current_families) > 1:
        raise ValueError("current cell stack spans multiple normal families")
    total_mass = float(np.sum(weights))
    empty_score = -0.35 * math.log1p(total_mass)
    empty = RetainedSaturationConfiguration(
        cell_xyz,
        shard_id,
        -1,
        (),
        0.0,
        empty_score,
        empty_score,
        0.0,
        total_mass,
        0,
        not current_set,
    )
    if not len(weights):
        if current_set:
            family = next(iter(current_families))
            ordered_current = tuple(
                sorted(current_set, key=lambda value: float(table.height[value]))
            )
            physical = sum(_layer_reward(table.mode(value)) for value in ordered_current)
            for first, second in zip(ordered_current, ordered_current[1:]):
                transition = _transition_reward(
                    table.mode(first), table.mode(second), source, raw_settings
                )
                if transition is None:
                    raise ValueError("anchored current stack is not physically ordered")
                physical += transition
            physical += math.log(max(float(normal_confidence[family]), 0.03))
            current = RetainedSaturationConfiguration(
                cell_xyz,
                shard_id,
                family,
                ordered_current,
                0.0,
                physical,
                physical,
                0.0,
                0.0,
                0,
                True,
            )
            return (current, empty), {
                "enumeratedPaths": 0,
                "retainedConfigurations": 2,
                "anyModeSupportedEvidenceMass": 0.0,
                "bestNormalFamilyAnyModeSupportedEvidenceMass": 0.0,
                "oracleCoveredEvidenceMass": 0.0,
                "retainedOracleCoveredEvidenceMass": 0.0,
                "localUnaryCoveredEvidenceMass": 0.0,
                "currentCoveredEvidenceMass": 0.0,
                "totalEvidenceMass": 0.0,
            }
        return (empty,), {
            "enumeratedPaths": 0,
            "retainedConfigurations": 1,
            "anyModeSupportedEvidenceMass": 0.0,
            "bestNormalFamilyAnyModeSupportedEvidenceMass": 0.0,
            "oracleCoveredEvidenceMass": 0.0,
            "retainedOracleCoveredEvidenceMass": 0.0,
            "localUnaryCoveredEvidenceMass": 0.0,
            "currentCoveredEvidenceMass": 0.0,
            "totalEvidenceMass": 0.0,
        }

    background = math.exp(-0.5 * resolved.joint_residual_limit**2)
    by_coverage: dict[int, list[RetainedSaturationConfiguration]] = defaultdict(list)
    current_candidate: RetainedSaturationConfiguration | None = None
    oracle_candidate: RetainedSaturationConfiguration | None = None
    enumerated_paths = 0
    mass_cache: dict[int, float] = {0: 0.0}
    any_mode_mask = 0
    best_family_any_mode_mass = 0.0

    def retain(candidate: RetainedSaturationConfiguration) -> None:
        nonlocal current_candidate, oracle_candidate
        if oracle_candidate is None or (
            candidate.covered_evidence_mass,
            candidate.total_log_score,
            -len(candidate.mode_indices),
            candidate.mode_indices,
        ) > (
            oracle_candidate.covered_evidence_mass,
            oracle_candidate.total_log_score,
            -len(oracle_candidate.mode_indices),
            oracle_candidate.mode_indices,
        ):
            oracle_candidate = candidate
        if candidate.is_current:
            current_candidate = candidate
        values = by_coverage[candidate.coverage_mask]
        values.append(candidate)
        values.sort(
            key=lambda value: (
                -value.total_log_score,
                len(value.mode_indices),
                value.normal_hypothesis,
                value.mode_indices,
            )
        )
        del values[resolved.maximum_configurations_per_coverage :]

    available_index = np.asarray(available, dtype=np.int64)
    for family_value in sorted(
        set(int(value) for value in table.normal_hypothesis[available_index])
    ):
        family_modes = np.asarray(
            [
                value
                for value in available
                if int(table.normal_hypothesis[value]) == family_value
            ],
            dtype=np.int64,
        )
        family_modes = family_modes[np.argsort(table.height[family_modes])]
        offsets = centers - np.asarray(cell_center_xyz, dtype=np.float64)[None, :]
        distance = np.abs(
            offsets @ table.normal_xyz[family_modes].astype(np.float64).T
            - table.height[family_modes][None, :]
        )
        axial_dot = np.clip(
            np.abs(
                directions
                @ table.fiber_xyz[family_modes].astype(np.float64).T
            ),
            0.0,
            1.0,
        )
        angular = np.degrees(np.arccos(axial_dot))
        angular_sigma = np.hypot(
            raw_settings.orientation_kernel_degrees,
            np.degrees(
                table.fiber_angular_std_radians[family_modes].astype(np.float64)
            ),
        )
        joint_squared = (
            distance / raw_settings.depth_kernel_voxels
        ) ** 2 + (angular / angular_sigma[None, :]) ** 2
        confidence = np.maximum(
            table.confidence[family_modes].astype(np.float64), 1.0e-12
        )
        mode_likelihood = confidence[None, :] * np.exp(-0.5 * joint_squared)
        mode_masks: list[int] = []
        for mode_column in range(len(family_modes)):
            mask = 0
            for sample_index in np.flatnonzero(
                joint_squared[:, mode_column]
                <= resolved.joint_residual_limit**2
            ):
                mask |= 1 << int(sample_index)
            mode_masks.append(mask)
            any_mode_mask |= mask
        family_any_mode_mask = 0
        for mask in mode_masks:
            family_any_mode_mask |= mask
        best_family_any_mode_mass = max(
            best_family_any_mode_mass,
            _mask_mass(family_any_mode_mask, weights, mass_cache),
        )
        modes = [table.mode(int(value)) for value in family_modes]
        layer_rewards = [_layer_reward(value) for value in modes]
        transition: list[list[float | None]] = [
            [None] * len(family_modes) for _ in range(len(family_modes))
        ]
        for first in range(len(family_modes)):
            for second in range(first + 1, len(family_modes)):
                transition[first][second] = _transition_reward(
                    modes[first], modes[second], source, raw_settings
                )
        current_signature = tuple(
            int(value) for value in family_modes if int(value) in current_set
        )
        normal_term = math.log(max(float(normal_confidence[family_value]), 0.03))

        def visit(
            last: int,
            chosen: tuple[int, ...],
            likelihood_sum: np.ndarray,
            confidence_sum: float,
            coverage_mask: int,
            physical_score: float,
        ) -> None:
            nonlocal enumerated_paths
            for column in range(last + 1, len(family_modes)):
                transition_reward = 0.0
                if last >= 0:
                    resolved_transition = transition[last][column]
                    if resolved_transition is None:
                        continue
                    transition_reward = resolved_transition
                mode_index = int(family_modes[column])
                next_chosen = chosen + (mode_index,)
                next_likelihood = likelihood_sum + mode_likelihood[:, column]
                next_confidence = confidence_sum + float(confidence[column])
                next_mask = coverage_mask | mode_masks[column]
                next_physical = (
                    physical_score + layer_rewards[column] + transition_reward
                )
                evidence_score = configuration_evidence_log_score(
                    next_likelihood,
                    next_confidence,
                    weights,
                    background_likelihood=background,
                )
                physical_with_normal = next_physical + normal_term
                candidate = RetainedSaturationConfiguration(
                    cell_xyz,
                    shard_id,
                    family_value,
                    next_chosen,
                    evidence_score,
                    physical_with_normal,
                    evidence_score + physical_with_normal,
                    _mask_mass(next_mask, weights, mass_cache),
                    total_mass,
                    next_mask,
                    bool(
                        current_set
                        and family_value in current_families
                        and next_chosen == current_signature
                    ),
                )
                enumerated_paths += 1
                retain(candidate)
                visit(
                    column,
                    next_chosen,
                    next_likelihood,
                    next_confidence,
                    next_mask,
                    next_physical,
                )

        visit(
            -1,
            (),
            np.zeros(len(weights), dtype=np.float64),
            0.0,
            0,
            0.0,
        )

    if current_set and current_candidate is None:
        raise ValueError("physical enumeration did not recover the anchored current stack")
    if oracle_candidate is None:
        raise RuntimeError("nonempty evidence and mode bank produced no physical stack")
    candidates = [value for values in by_coverage.values() for value in values]
    candidates.sort(
        key=lambda value: (
            -value.total_log_score,
            -value.covered_evidence_mass,
            len(value.mode_indices),
            value.normal_hypothesis,
            value.mode_indices,
        )
    )
    retained = candidates[: resolved.maximum_configurations_per_cell]
    retained_identities = {
        (value.normal_hypothesis, value.mode_indices) for value in retained
    }
    oracle_identity = (
        oracle_candidate.normal_hypothesis,
        oracle_candidate.mode_indices,
    )
    if oracle_identity not in retained_identities:
        retained[-1] = oracle_candidate
    required = [empty]
    if current_candidate is not None:
        required.append(current_candidate)
    identities = {
        (value.normal_hypothesis, value.mode_indices) for value in retained
    }
    for value in required:
        key = (value.normal_hypothesis, value.mode_indices)
        if key not in identities:
            retained.append(value)
            identities.add(key)
    retained.sort(
        key=lambda value: (
            -value.total_log_score,
            not value.is_current,
            len(value.mode_indices),
            value.normal_hypothesis,
            value.mode_indices,
        )
    )
    current_mass = (
        current_candidate.covered_evidence_mass
        if current_candidate is not None
        else 0.0
    )
    retained_oracle = max(
        retained,
        key=lambda value: (
            value.covered_evidence_mass,
            value.total_log_score,
            -len(value.mode_indices),
            value.mode_indices,
        ),
    )
    local_unary = max(
        retained,
        key=lambda value: (value.total_log_score, -len(value.mode_indices)),
    )
    return tuple(retained), {
        "enumeratedPaths": enumerated_paths,
        "uniqueCoveragePatterns": len(by_coverage),
        "retainedConfigurations": len(retained),
        "anyModeSupportedEvidenceMass": _mask_mass(
            any_mode_mask, weights, mass_cache
        ),
        "bestNormalFamilyAnyModeSupportedEvidenceMass": (
            best_family_any_mode_mass
        ),
        "oracleCoveredEvidenceMass": oracle_candidate.covered_evidence_mass,
        "retainedOracleCoveredEvidenceMass": retained_oracle.covered_evidence_mass,
        "localUnaryCoveredEvidenceMass": local_unary.covered_evidence_mass,
        "currentCoveredEvidenceMass": current_mass,
        "totalEvidenceMass": total_mass,
    }


def _configuration_table(
    cells: list[tuple[Int3, tuple[RetainedSaturationConfiguration, ...]]],
    mode_tables: Mapping[str, LayerModeTable],
) -> tuple[ConfigurationTable, dict[str, np.ndarray], list[RetainedSaturationConfiguration]]:
    configurations = [value for _, values in cells for value in values]
    configuration_offset = np.zeros(len(cells) + 1, dtype=np.uint64)
    for index, (_, values) in enumerate(cells):
        configuration_offset[index + 1] = configuration_offset[index] + len(values)
    log_weights = np.empty(len(configurations), dtype=np.float32)
    cursor = 0
    for _, values in cells:
        scores = np.asarray([value.total_log_score for value in values], dtype=np.float64)
        maximum = float(np.max(scores))
        normalizer = maximum + math.log(float(np.sum(np.exp(scores - maximum))))
        log_weights[cursor : cursor + len(values)] = (scores - normalizer).astype(
            np.float32
        )
        cursor += len(values)
    layer_offset = np.zeros(len(configurations) + 1, dtype=np.uint64)
    for index, value in enumerate(configurations):
        layer_offset[index + 1] = layer_offset[index] + len(value.mode_indices)
    layer_sources = [
        (value.shard_id, mode_index)
        for value in configurations
        for mode_index in value.mode_indices
    ]

    def layer_array(name: str, shape: tuple[int, ...], dtype: Any) -> np.ndarray:
        if not layer_sources:
            return np.empty((0, *shape), dtype=dtype)
        values = [getattr(mode_tables[shard_id], name)[mode] for shard_id, mode in layer_sources]
        return np.asarray(values, dtype=dtype).reshape(len(layer_sources), *shape)

    table = ConfigurationTable(
        np.asarray([cell for cell, _ in cells], dtype=np.int32),
        configuration_offset,
        np.concatenate(
            [np.arange(len(values), dtype=np.uint16) for _, values in cells]
        ),
        log_weights,
        np.asarray(
            [value.normal_hypothesis for value in configurations], dtype=np.int8
        ),
        layer_offset,
        layer_array("normal_xyz", (3,), np.float32),
        layer_array("height", (), np.float32),
        layer_array("covariance", (6,), np.float32),
        layer_array("fiber_xyz", (3,), np.float32),
        layer_array("fiber_angular_std_radians", (), np.float32),
        layer_array("confidence", (), np.float32),
        layer_array("evidence_score", (), np.float32),
        layer_array("material_probability", (), np.float32),
        layer_array("effective_support", (), np.float32),
    )
    table.validate()
    shard_names = sorted(mode_tables)
    shard_index = {value: index for index, value in enumerate(shard_names)}
    metadata = {
        "sourceShardIndex": np.asarray(
            [shard_index[value.shard_id] for value in configurations], dtype=np.int16
        ),
        "sourceModeOffset": layer_offset.copy(),
        "sourceModeIndex": np.asarray(
            [mode for value in configurations for mode in value.mode_indices],
            dtype=np.int32,
        ),
        "evidenceLogScore": np.asarray(
            [value.evidence_log_score for value in configurations], dtype=np.float32
        ),
        "physicalLogScore": np.asarray(
            [value.physical_log_score for value in configurations], dtype=np.float32
        ),
        "totalLogScore": np.asarray(
            [value.total_log_score for value in configurations], dtype=np.float32
        ),
        "coveredEvidenceMass": np.asarray(
            [value.covered_evidence_mass for value in configurations], dtype=np.float32
        ),
        "totalEvidenceMass": np.asarray(
            [value.total_evidence_mass for value in configurations], dtype=np.float32
        ),
        "isCurrent": np.asarray(
            [value.is_current for value in configurations], dtype=np.uint8
        ),
        "shardNames": np.asarray(shard_names, dtype=f"U{max(map(len, shard_names))}"),
    }
    return table, metadata, configurations


def _write_configuration_candidates(
    output: Path,
    table: ConfigurationTable,
    metadata: Mapping[str, np.ndarray],
    *,
    identity_sha256: str,
    statistics: Mapping[str, Any],
) -> dict[str, Any]:
    data_path = output / "saturation-configurations-v1.npz"
    temporary = data_path.with_suffix(data_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **table.arrays(), **metadata)
    temporary.replace(data_path)
    payload = {
        "schema": "pareidolia.cubical-saturation-configurations",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": dict(statistics),
        "scoreModel": {
            "evidence": (
                "confidence-normalized Gaussian layer mixture against a background "
                "fixed at the configured joint-residual contour"
            ),
            "physicalPrior": (
                "raw Acus fitted-mode evidence, support, confidence, spacing, "
                "non-crossing, and soft orthogonal-ply affinity"
            ),
            "directions": "axial/unsigned",
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
        },
    }
    atomic_json(output / "saturation-configurations-v1.json", payload)
    return payload


def _block_statistics(block: SurfaceBlock) -> dict[str, Any]:
    sizes = sorted((len(value.patch_ids) for value in block.components), reverse=True)
    return {
        "patches": len(block.patches),
        "candidateJoins": len(block.candidate_joins),
        "retainedJoins": len(block.joins),
        "deferredJoins": len(block.deferred_joins),
        "deferredByReason": dict(
            sorted(Counter(value.reason for value in block.deferred_joins).items())
        ),
        "components": len(block.components),
        "largestComponentPatchCount": max(sizes, default=0),
        "topComponentPatchCounts": sizes[:20],
        "exteriorTraces": len(block.exterior_traces),
        "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
    }


def _identity(
    input_root: Path,
    mode_bank_root: Path,
    pipeline_root: Path,
    settings: SaturationReselectionSettings,
) -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": SATURATION_RESELECTION_SCHEMA,
        "version": SATURATION_RESELECTION_VERSION,
        "inputRoot": str(input_root),
        "inputPatchManifestSha256": sha256_file(
            input_root / "selected-patches-v1.json"
        ),
        "inputPatchDataSha256": sha256_file(input_root / "selected-patches-v1.npz"),
        "modeBankRoot": str(mode_bank_root),
        "modeBankManifestSha256": sha256_file(mode_bank_root / "mode-bank.json"),
        "pipelineRoot": str(pipeline_root),
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(root / name)
            for name in (
                "saturation_reselection.py",
                "saturation.py",
                "stratigraphy.py",
                "selection.py",
                "matching.py",
                "block.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def run_saturation_reselection(
    input_root: str | Path,
    mode_bank_root: str | Path,
    output_root: str | Path,
    *,
    settings: SaturationReselectionSettings | None = None,
    force: bool = False,
    progress: Callable[[int, int, Int3, int], None] | None = None,
) -> dict[str, Any]:
    """Reselect every cell from full-bank physical stacks using owned Acus evidence."""

    started = time.monotonic()
    resolved = settings or SaturationReselectionSettings()
    input_path = Path(input_root).resolve()
    bank_path = Path(mode_bank_root).resolve()
    output = Path(output_root).resolve()
    if output in (input_path, bank_path):
        raise ValueError("saturation reselection output must differ from every input")
    pipeline_root, pipeline = resolve_pipeline_manifest(input_path)
    pipeline_identity = str(pipeline["identity"]["identitySha256"])
    bank_manifest = json.loads((bank_path / "mode-bank.json").read_text())
    if (
        bank_manifest.get("schema") != MODE_BANK_SCHEMA
        or bank_manifest.get("state") != "complete"
        or bank_manifest["identity"]["inputPipelineIdentitySha256"]
        != pipeline_identity
    ):
        raise ValueError("mode bank and selected reconstruction have different inputs")
    identity = _identity(input_path, bank_path, pipeline_root, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("saturation reselection output belongs to another identity")
        if (
            not force
            and prior.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SATURATION_RESELECTION_SCHEMA,
        "version": SATURATION_RESELECTION_VERSION,
        "state": "loading",
        "identity": identity,
        "inputRoot": str(input_path),
        "pipelineRoot": str(pipeline_root),
        "modeBankRoot": str(bank_path),
    }
    atomic_json(manifest_path, manifest)

    source_values = pipeline["identity"]["source"]
    source = VolumeSource.open(
        source_values["path"], source_values.get("metadataPath")
    )
    raw_settings = RawAcusSettings(**pipeline["identity"]["settings"])
    selected = read_patch_shard(input_path / "selected-patches-v1", verify=True)
    _, mode_tables = load_mode_bank(bank_path, verify=True)
    fingerprints, fingerprint_stats = build_patch_fingerprints(
        selected, mode_tables, raw_settings
    )
    if not np.all(fingerprints.anchor_valid):
        raise ValueError("saturation reselection requires every selected layer to be bank-anchored")
    evidence_tables = {
        shard_id: read_evidence_artifact(
            pipeline_root / "shards" / shard_id / "evidence-v1",
            identity_sha256=pipeline_identity,
            verify=True,
        )
        for shard_id in mode_tables
    }
    needle_identity, needle_manifests = canonical_needle_artifact_identity(
        pipeline_root
    )
    del needle_identity
    window_values = pipeline["identity"]["window"]
    window_start = np.asarray(window_values["originVoxelXYZ"], dtype=np.float64)
    shape = tuple(int(value) for value in selected.grid.shape_cells_xyz)
    stride = raw_settings.cell_stride_voxels
    window_stop = window_start + np.asarray(shape, dtype=np.float64) * stride
    needles = load_owned_canonical_needles(
        pipeline_root,
        needle_manifests,
        pipeline_identity=pipeline_identity,
        window_start_xyz=window_start,
        window_stop_xyz=window_stop,
    )
    needle_cell = np.floor(
        (needles.center_xyz.astype(np.float64) - window_start[None, :]) / stride
    ).astype(np.int32)
    needle_weight = needles.score.astype(np.float64) * np.sqrt(
        np.maximum(
            needles.axial_coverage.astype(np.float64)
            * needles.support_score.astype(np.float64),
            0.0,
        )
    )
    evidence_by_cell = {
        cell: np.flatnonzero(np.all(needle_cell == cell, axis=1))
        for cell in {
            tuple(int(value) for value in row) for row in needle_cell
        }
    }
    patch_rows_by_cell: dict[Int3, list[int]] = defaultdict(list)
    for row, values in enumerate(selected.cell_xyz):
        patch_rows_by_cell[tuple(int(value) for value in values)].append(row)
    cell_lookup: dict[Int3, tuple[str, LayerModeTable, int]] = {}
    for shard_id in sorted(mode_tables):
        table = mode_tables[shard_id]
        for cell_index, values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in values)
            if cell in cell_lookup:
                raise ValueError(f"mode-bank cell {cell} has multiple owners")
            cell_lookup[cell] = shard_id, table, cell_index

    manifest["state"] = "enumerating-physical-configurations"
    atomic_json(manifest_path, manifest)
    enumeration_started = time.monotonic()
    ordered_cells = [
        (x, y, z)
        for z in range(shape[2])
        for y in range(shape[1])
        for x in range(shape[0])
    ]
    cell_configurations: list[
        tuple[Int3, tuple[RetainedSaturationConfiguration, ...]]
    ] = []
    enumeration_stats: list[dict[str, Any]] = []
    for number, cell in enumerate(ordered_cells, start=1):
        shard_id, table, cell_index = cell_lookup[cell]
        sample = evidence_by_cell.get(cell, np.empty(0, dtype=np.int64))
        patch_rows = np.asarray(patch_rows_by_cell.get(cell, ()), dtype=np.int64)
        current_modes = tuple(
            sorted(
                {
                    int(fingerprints.anchor_mode_index[row])
                    for row in patch_rows
                }
            )
        )
        center = window_start + (np.asarray(cell, dtype=np.float64) + 0.5) * stride
        configurations, statistics = enumerate_cell_saturation_configurations(
            cell,
            shard_id,
            table,
            cell_index,
            needles.center_xyz[sample],
            needles.direction_xyz[sample],
            needle_weight[sample],
            cell_center_xyz=center,
            normal_confidence=evidence_tables[shard_id].normal_confidence[cell_index],
            current_mode_indices=current_modes,
            source=source,
            raw_settings=raw_settings,
            settings=resolved,
        )
        cell_configurations.append((cell, configurations))
        enumeration_stats.append(statistics)
        if progress is not None:
            progress(number, len(ordered_cells), cell, statistics["enumeratedPaths"])

    configuration_table, metadata, flattened = _configuration_table(
        cell_configurations, mode_tables
    )
    enumeration_summary = {
        "cells": len(ordered_cells),
        "enumeratedPhysicalPaths": sum(
            int(value["enumeratedPaths"]) for value in enumeration_stats
        ),
        "retainedConfigurations": configuration_table.configuration_count,
        "retainedLayers": configuration_table.layer_count,
        "meanRetainedConfigurationsPerCell": round(
            configuration_table.configuration_count / len(ordered_cells), 7
        ),
        "currentSupportedEvidenceMass": round(
            sum(float(value["currentCoveredEvidenceMass"]) for value in enumeration_stats),
            7,
        ),
        "localUnarySupportedEvidenceMass": round(
            sum(
                float(value["localUnaryCoveredEvidenceMass"])
                for value in enumeration_stats
            ),
            7,
        ),
        "retainedOracleSupportedEvidenceMass": round(
            sum(
                float(value["retainedOracleCoveredEvidenceMass"])
                for value in enumeration_stats
            ),
            7,
        ),
        "localOracleSupportedEvidenceMass": round(
            sum(float(value["oracleCoveredEvidenceMass"]) for value in enumeration_stats),
            7,
        ),
        "anyModeSupportedEvidenceMass": round(
            sum(
                float(value["anyModeSupportedEvidenceMass"])
                for value in enumeration_stats
            ),
            7,
        ),
        "bestNormalFamilyAnyModeSupportedEvidenceMass": round(
            sum(
                float(value["bestNormalFamilyAnyModeSupportedEvidenceMass"])
                for value in enumeration_stats
            ),
            7,
        ),
        "totalEvidenceMass": round(
            sum(float(value["totalEvidenceMass"]) for value in enumeration_stats), 7
        ),
    }
    total_enumerated_mass = max(
        float(enumeration_summary["totalEvidenceMass"]), 1.0e-12
    )
    enumeration_summary["supportFractions"] = {
        name: round(float(enumeration_summary[key]) / total_enumerated_mass, 7)
        for name, key in (
            ("current", "currentSupportedEvidenceMass"),
            ("localUnary", "localUnarySupportedEvidenceMass"),
            ("retainedPhysicalOracle", "retainedOracleSupportedEvidenceMass"),
            ("completePhysicalOracle", "localOracleSupportedEvidenceMass"),
            (
                "bestNormalFamilyAnyMode",
                "bestNormalFamilyAnyModeSupportedEvidenceMass",
            ),
            ("anyMode", "anyModeSupportedEvidenceMass"),
        )
    }
    _write_configuration_candidates(
        output,
        configuration_table,
        metadata,
        identity_sha256=identity_sha256,
        statistics=enumeration_summary,
    )
    enumeration_finished = time.monotonic()

    manifest["state"] = "global-selection"
    atomic_json(manifest_path, manifest)
    selection_started = time.monotonic()
    selection = optimize_configurations(
        selected.grid,
        (configuration_table,),
        pairwise_scale=resolved.pairwise_scale,
        interior_unmatched_trace_penalty=resolved.interior_unmatched_trace_penalty,
        maximum_sweeps=resolved.maximum_sweeps,
    )
    selection_manifest = write_selection_artifact(
        output, selection, identity_sha256
    )
    selected_configuration_indices = np.asarray(
        [value.source_configuration_index for value in selection.selected_options],
        dtype=np.int64,
    )
    chosen_records = [flattened[index] for index in selected_configuration_indices]
    selected_supported_mass = float(
        sum(value.covered_evidence_mass for value in chosen_records)
    )
    total_evidence_mass = float(enumeration_summary["totalEvidenceMass"])
    changed_cells = sum(not value.is_current for value in chosen_records)
    selection_finished = time.monotonic()

    manifest["state"] = "assembling"
    atomic_json(manifest_path, manifest)
    assembly_started = time.monotonic()
    baseline = assemble_surface_hierarchy(
        selected.grid,
        BlockBounds((0, 0, 0), selected.grid.shape_cells_xyz),
        selected.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    reselected_table = patch_table_from_selection(
        selected.grid, [configuration_table], selection
    )
    reselected = assemble_surface_hierarchy(
        selected.grid,
        BlockBounds((0, 0, 0), selected.grid.shape_cells_xyz),
        reselected_table.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        reselected_table,
        settings={
            "source": "full-bank saturation-aware physical reselection",
            **resolved.record(),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "inputRoot": str(input_path),
            "modeBankRoot": str(bank_path),
            "directions": "axial/unsigned",
        },
        compressed=True,
    )
    obj_path = write_block_obj(reselected, output / "surface.obj")
    projection_path = write_block_projection_png(
        reselected, output / "projections.png", maximum_components=128
    )
    largest_path = write_block_projection_png(
        reselected, output / "largest-component.png", maximum_components=1
    )
    top_twelve_path = write_block_projection_png(
        reselected, output / "top-12-components.png", maximum_components=12
    )
    before = _block_statistics(baseline)
    after = _block_statistics(reselected)
    finished = time.monotonic()
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-saturation-reselection-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(input_path),
        "modeBankRoot": str(bank_path),
        "grid": patch_manifest["grid"],
        "method": {
            "candidatePopulation": "complete independently fitted Acus mode bank",
            "localConstraint": (
                "one unsigned normal family, physical minimum spacing, and no crossing "
                "inside the complete cell"
            ),
            "evidenceScore": (
                "confidence-normalized Gaussian mode mixture over independently owned "
                "finite-length Acus evidence"
            ),
            "globalSelection": (
                "ICM over normalized local evidence-plus-physical scores and shared-face "
                "trace likelihoods"
            ),
            "topology": "collision-safe and orientability-safe cubical assembly",
        },
        "fingerprintAnchoring": fingerprint_stats,
        "enumeration": enumeration_summary,
        "selection": selection_manifest["statistics"],
        "selectedPrediction": {
            "changedCells": changed_cells,
            "unchangedCells": len(chosen_records) - changed_cells,
            "supportedEvidenceMass": round(selected_supported_mass, 7),
            "supportedEvidenceMassFraction": round(
                selected_supported_mass / max(total_evidence_mass, 1.0e-12), 7
            ),
            "currentSupportedEvidenceMassFraction": round(
                float(enumeration_summary["currentSupportedEvidenceMass"])
                / max(total_evidence_mass, 1.0e-12),
                7,
            ),
            "localOracleSupportedEvidenceMassFraction": round(
                float(enumeration_summary["localOracleSupportedEvidenceMass"])
                / max(total_evidence_mass, 1.0e-12),
                7,
            ),
        },
        "baseline": before,
        "reselected": after,
        "delta": {
            key: int(after[key]) - int(before[key])
            for key in (
                "patches",
                "candidateJoins",
                "retainedJoins",
                "deferredJoins",
                "components",
                "largestComponentPatchCount",
                "exteriorTraces",
                "unresolvedInteriorTraces",
            )
        },
        "timingSeconds": {
            "loading": round(enumeration_started - started, 6),
            "enumerationAndCandidateWrite": round(
                enumeration_finished - enumeration_started, 6
            ),
            "globalSelection": round(selection_finished - selection_started, 6),
            "assemblyAndExports": round(finished - assembly_started, 6),
            "total": round(finished - started, 6),
        },
        "artifacts": {
            "candidateConfigurations": "saturation-configurations-v1.npz",
            "selection": "selection-v1.npz",
            "selectedPatches": "selected-patches-v1.npz",
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponent": largest_path.name,
            "topTwelveComponents": top_twelve_path.name,
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
