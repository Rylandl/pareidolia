from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .block import BlockBounds, assemble_surface_hierarchy
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import write_block_obj, write_block_projection_png
from .pipeline import patch_table_from_selection, write_selection_artifact
from .saturation_reselection import _block_statistics
from .selection import optimize_configurations
from .stratigraphy import ConfigurationTable
from .tables import read_patch_shard, write_patch_shard


SATURATION_CANDIDATE_SELECTION_SCHEMA = (
    "pareidolia.cubical-saturation-candidate-selection"
)
SATURATION_CANDIDATE_SELECTION_VERSION = 1


@dataclass(frozen=True, slots=True)
class SaturationCandidateSelectionSettings:
    """Global selection settings over one immutable physical candidate bank."""

    coverage_reward_scale: float = 0.0
    unary_scale: float = 1.0
    pairwise_scale: float = 0.2
    interior_unmatched_trace_penalty: float = 0.0
    maximum_sweeps: int = 12
    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3)
    write_visuals: bool = True

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.coverage_reward_scale)
            or self.coverage_reward_scale < 0.0
        ):
            raise ValueError("coverage reward scale must be finite and nonnegative")
        if self.unary_scale <= 0.0 or self.pairwise_scale <= 0.0:
            raise ValueError("selection energy scales must be positive")
        if (
            not math.isfinite(self.interior_unmatched_trace_penalty)
            or self.interior_unmatched_trace_penalty < 0.0
        ):
            raise ValueError("unmatched trace penalty must be finite and nonnegative")
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


def _configuration_table_from_values(
    values: Mapping[str, np.ndarray],
) -> ConfigurationTable:
    table = ConfigurationTable(
        np.asarray(values["cellXYZ"], dtype=np.int32),
        np.asarray(values["configurationOffset"], dtype=np.uint64),
        np.asarray(values["configurationId"], dtype=np.uint16),
        np.asarray(values["configurationLogWeight"], dtype=np.float32),
        np.asarray(values["normalHypothesis"], dtype=np.int8),
        np.asarray(values["layerOffset"], dtype=np.uint64),
        np.asarray(values["layerNormalXYZ"], dtype=np.float32),
        np.asarray(values["layerHeight"], dtype=np.float32),
        np.asarray(values["layerCovariance"], dtype=np.float32),
        np.asarray(values["layerFiberXYZ"], dtype=np.float32),
        np.asarray(values["layerFiberAngularStdRadians"], dtype=np.float32),
        np.asarray(values["layerConfidence"], dtype=np.float32),
        np.asarray(values["layerEvidenceScore"], dtype=np.float32),
        np.asarray(values["layerMaterialProbability"], dtype=np.float32),
        np.asarray(values["layerEffectiveSupport"], dtype=np.float32),
    )
    table.validate()
    return table


def load_saturation_candidates(
    root: str | Path,
    *,
    verify: bool = True,
) -> tuple[ConfigurationTable, dict[str, np.ndarray], dict[str, Any]]:
    source = Path(root).resolve()
    manifest = json.loads(
        (source / "saturation-configurations-v1.json").read_text()
    )
    if (
        manifest.get("schema")
        != "pareidolia.cubical-saturation-configurations"
        or int(manifest.get("version", -1)) != 1
    ):
        raise ValueError("candidate root is not a saturation configuration bank")
    data_path = source / str(manifest["data"]["path"])
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("saturation candidate content hash mismatch")
    with np.load(data_path) as values:
        table = _configuration_table_from_values(values)
        metadata = {
            name: np.asarray(values[name])
            for name in (
                "evidenceLogScore",
                "physicalLogScore",
                "totalLogScore",
                "coveredEvidenceMass",
                "totalEvidenceMass",
                "isCurrent",
            )
        }
    if any(len(value) != table.configuration_count for value in metadata.values()):
        raise ValueError("candidate score metadata does not align with configurations")
    return table, metadata, manifest


def reweight_saturation_candidates(
    table: ConfigurationTable,
    metadata: Mapping[str, np.ndarray],
    *,
    coverage_reward_scale: float,
) -> ConfigurationTable:
    """Add an explicit utilization reward and normalize within each cell."""

    scores = np.asarray(metadata["totalLogScore"], dtype=np.float64).copy()
    scores += coverage_reward_scale * np.asarray(
        metadata["coveredEvidenceMass"], dtype=np.float64
    )
    log_weight = np.empty(table.configuration_count, dtype=np.float32)
    for start, stop in zip(
        table.configuration_offset[:-1], table.configuration_offset[1:]
    ):
        low = int(start)
        high = int(stop)
        maximum = float(np.max(scores[low:high]))
        normalizer = maximum + math.log(
            float(np.sum(np.exp(scores[low:high] - maximum)))
        )
        log_weight[low:high] = (scores[low:high] - normalizer).astype(np.float32)
    reweighted = ConfigurationTable(
        table.cell_xyz,
        table.configuration_offset,
        table.configuration_id,
        log_weight,
        table.normal_hypothesis,
        table.layer_offset,
        table.layer_normal_xyz,
        table.layer_height,
        table.layer_covariance,
        table.layer_fiber_xyz,
        table.layer_fiber_angular_std_radians,
        table.layer_confidence,
        table.layer_evidence_score,
        table.layer_material_probability,
        table.layer_effective_support,
    )
    reweighted.validate()
    return reweighted


def _identity(
    candidate_root: Path,
    candidate_manifest: Mapping[str, Any],
    input_root: Path,
    settings: SaturationCandidateSelectionSettings,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": SATURATION_CANDIDATE_SELECTION_SCHEMA,
        "version": SATURATION_CANDIDATE_SELECTION_VERSION,
        "candidateRoot": str(candidate_root),
        "candidateIdentitySha256": candidate_manifest["identitySha256"],
        "candidateDataSha256": candidate_manifest["data"]["sha256"],
        "inputRoot": str(input_root),
        "inputPatchManifestSha256": sha256_file(
            input_root / "selected-patches-v1.json"
        ),
        "inputPatchDataSha256": sha256_file(input_root / "selected-patches-v1.npz"),
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "saturation_selection.py",
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


def run_saturation_candidate_selection(
    candidate_root: str | Path,
    output_root: str | Path,
    *,
    settings: SaturationCandidateSelectionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Reselect and assemble an immutable saturation candidate bank."""

    started = time.monotonic()
    resolved = settings or SaturationCandidateSelectionSettings()
    candidates = Path(candidate_root).resolve()
    output = Path(output_root).resolve()
    if output == candidates:
        raise ValueError("candidate selection output must differ from its input")
    candidate_variant = json.loads((candidates / "variant.json").read_text())
    if candidate_variant.get("state") != "complete":
        raise ValueError("candidate selection requires a completed candidate root")
    input_root = Path(candidate_variant["inputRoot"]).resolve()
    table, metadata, candidate_manifest = load_saturation_candidates(candidates)
    identity = _identity(candidates, candidate_manifest, input_root, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "variant.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("candidate selection output belongs to another identity")
        if (
            not force
            and prior.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SATURATION_CANDIDATE_SELECTION_SCHEMA,
        "version": SATURATION_CANDIDATE_SELECTION_VERSION,
        "state": "selecting",
        "identity": identity,
        "candidateRoot": str(candidates),
        "inputRoot": str(input_root),
    }
    atomic_json(manifest_path, manifest)

    baseline_table = read_patch_shard(input_root / "selected-patches-v1", verify=True)
    weighted_table = reweight_saturation_candidates(
        table,
        metadata,
        coverage_reward_scale=resolved.coverage_reward_scale,
    )
    selection_started = time.monotonic()
    selection = optimize_configurations(
        baseline_table.grid,
        (weighted_table,),
        unary_scale=resolved.unary_scale,
        pairwise_scale=resolved.pairwise_scale,
        interior_unmatched_trace_penalty=resolved.interior_unmatched_trace_penalty,
        maximum_sweeps=resolved.maximum_sweeps,
    )
    selection_finished = time.monotonic()
    selection_manifest = write_selection_artifact(output, selection, identity_sha256)
    selected_indices = np.asarray(
        [value.source_configuration_index for value in selection.selected_options],
        dtype=np.int64,
    )
    covered_mass = np.asarray(metadata["coveredEvidenceMass"], dtype=np.float64)
    total_mass_by_configuration = np.asarray(
        metadata["totalEvidenceMass"], dtype=np.float64
    )
    total_mass = float(np.sum(total_mass_by_configuration[table.configuration_offset[:-1]]))
    selected_supported_mass = float(np.sum(covered_mass[selected_indices]))
    changed_cells = int(
        np.count_nonzero(np.asarray(metadata["isCurrent"])[selected_indices] == 0)
    )

    selected_table = patch_table_from_selection(
        baseline_table.grid, [weighted_table], selection
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        selected_table,
        settings={
            "source": "immutable full-bank saturation candidate selection",
            **resolved.record(),
        },
        provenance={
            "variantIdentitySha256": identity_sha256,
            "candidateRoot": str(candidates),
            "directions": "axial/unsigned",
        },
        compressed=True,
    )

    manifest["state"] = "assembling"
    atomic_json(manifest_path, manifest)
    assembly_started = time.monotonic()
    candidate_summary_path = candidates / "summary.json"
    baseline: dict[str, Any] | None = None
    if candidate_summary_path.is_file():
        candidate_summary = json.loads(candidate_summary_path.read_text())
        candidate_leaf = tuple(
            candidate_variant.get("identity", {})
            .get("settings", {})
            .get("leaf_shape_cells_xyz", ())
        )
        if (
            candidate_leaf == resolved.leaf_shape_cells_xyz
            and isinstance(candidate_summary.get("baseline"), dict)
        ):
            baseline = dict(candidate_summary["baseline"])
    if baseline is None:
        baseline_block = assemble_surface_hierarchy(
            baseline_table.grid,
            BlockBounds((0, 0, 0), baseline_table.grid.shape_cells_xyz),
            baseline_table.to_patches(),
            maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
        )
        baseline = _block_statistics(baseline_block)
    block = assemble_surface_hierarchy(
        baseline_table.grid,
        BlockBounds((0, 0, 0), baseline_table.grid.shape_cells_xyz),
        selected_table.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    selected_statistics = _block_statistics(block)
    assembly_finished = time.monotonic()

    artifacts: dict[str, Any] = {
        "candidateConfigurations": str(
            candidates / "saturation-configurations-v1.npz"
        ),
        "selection": "selection-v1.npz",
        "selectedPatches": "selected-patches-v1.npz",
    }
    if resolved.write_visuals:
        obj_path = write_block_obj(block, output / "surface.obj")
        projection_path = write_block_projection_png(
            block, output / "projections.png", maximum_components=128
        )
        largest_path = write_block_projection_png(
            block, output / "largest-component.png", maximum_components=1
        )
        top_twelve_path = write_block_projection_png(
            block, output / "top-12-components.png", maximum_components=12
        )
        artifacts.update(
            {
                "mesh": obj_path.name,
                "projections": projection_path.name,
                "largestComponent": largest_path.name,
                "topTwelveComponents": top_twelve_path.name,
            }
        )
    finished = time.monotonic()
    enumeration = candidate_manifest["statistics"]
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-saturation-candidate-selection-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "candidateRoot": str(candidates),
        "inputRoot": str(input_root),
        "settings": resolved.record(),
        "grid": patch_manifest["grid"],
        "enumeration": enumeration,
        "selection": selection_manifest["statistics"],
        "selectedPrediction": {
            "changedCells": changed_cells,
            "unchangedCells": table.cell_count - changed_cells,
            "supportedEvidenceMass": round(selected_supported_mass, 7),
            "supportedEvidenceMassFraction": round(
                selected_supported_mass / max(total_mass, 1.0e-12), 7
            ),
            "coverageRewardScale": resolved.coverage_reward_scale,
        },
        "baseline": baseline,
        "selected": selected_statistics,
        "delta": {
            key: int(selected_statistics[key]) - int(baseline[key])
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
            "loading": round(selection_started - started, 6),
            "selection": round(selection_finished - selection_started, 6),
            "assembly": round(assembly_finished - assembly_started, 6),
            "exports": round(finished - assembly_finished, 6),
            "total": round(finished - started, 6),
        },
        "artifacts": artifacts,
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
