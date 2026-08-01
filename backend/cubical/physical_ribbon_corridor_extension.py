from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_corridor_variants import (
    PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM,
    PHYSICAL_RIBBON_CORRIDOR_VARIANTS_VERSION,
    PhysicalRibbonCorridorVariantSettings,
    _cached_checkpoint,
    _checkpoint,
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
    enumerate_corridor_reconfiguration_variants,
    screen_exact_corridor_variants,
)


PHYSICAL_RIBBON_CORRIDOR_EXTENSION_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-extension"
)
PHYSICAL_RIBBON_CORRIDOR_EXTENSION_VERSION = 1

_EXACT_VARIANT_FIELDS = (
    "corridorVariantExactConnected",
    "corridorVariantComponentSplit",
    "corridorVariantHardConflict",
    "corridorVariantSurfaceEligible",
    "corridorVariantTriangleRegionCountBefore",
    "corridorVariantTriangleRegionCountAfter",
    "corridorVariantTriangleCountBefore",
    "corridorVariantTriangleCountAfter",
    "corridorVariantTriangleAreaBefore",
    "corridorVariantTriangleAreaAfter",
    "corridorVariantSharedArcRegionFraction",
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorExtensionSettings:
    maximum_variants_per_corridor: int = 16
    screen_only_previously_unresolved_corridors: bool = True

    def __post_init__(self) -> None:
        if not 2 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("extended corridor variant count must lie in [2, 16]")
        if not self.screen_only_previously_unresolved_corridors:
            raise ValueError(
                "the extension stage must preserve and target prior exact evidence"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_variant_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("extension requires a complete label-free variant artifact")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _variant_signature(
    variants: Mapping[str, np.ndarray], variant_index: int
) -> tuple[int, tuple[int, ...], tuple[int, ...]]:
    added_offset = np.asarray(
        variants["corridorVariantAddedOffset"], dtype=np.int64
    )
    added_value = np.asarray(
        variants["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        variants["corridorVariantRemovedOffset"], dtype=np.int64
    )
    removed_value = np.asarray(
        variants["corridorVariantRemovedFrontierIndex"], dtype=np.int32
    )
    return (
        int(np.asarray(variants["corridorVariantRow"])[variant_index]),
        tuple(
            int(value)
            for value in added_value[
                int(added_offset[variant_index]) : int(
                    added_offset[variant_index + 1]
                )
            ]
        ),
        tuple(
            int(value)
            for value in removed_value[
                int(removed_offset[variant_index]) : int(
                    removed_offset[variant_index + 1]
                )
            ]
        ),
    )


def _delta_variant_arrays(
    full: Mapping[str, np.ndarray],
    variant_indices: np.ndarray,
    *,
    corridor_count: int,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    row = np.asarray(full["corridorVariantRow"], dtype=np.int32)[variant_indices]
    order = np.lexsort(
        (
            np.asarray(full["corridorVariantRank"], dtype=np.int16)[variant_indices],
            row,
        )
    )
    source_index = variant_indices[order]
    row = row[order]
    counts = np.bincount(row, minlength=corridor_count)
    corridor_offset = np.concatenate(
        (np.zeros(1, dtype=np.int64), np.cumsum(counts, dtype=np.int64))
    )
    added_offset = np.asarray(full["corridorVariantAddedOffset"], dtype=np.int64)
    added_value = np.asarray(
        full["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        full["corridorVariantRemovedOffset"], dtype=np.int64
    )
    removed_value = np.asarray(
        full["corridorVariantRemovedFrontierIndex"], dtype=np.int32
    )
    delta_added_offset = [0]
    delta_added: list[int] = []
    delta_removed_offset = [0]
    delta_removed: list[int] = []
    for variant_index in source_index:
        delta_added.extend(
            int(value)
            for value in added_value[
                int(added_offset[variant_index]) : int(
                    added_offset[variant_index + 1]
                )
            ]
        )
        delta_added_offset.append(len(delta_added))
        delta_removed.extend(
            int(value)
            for value in removed_value[
                int(removed_offset[variant_index]) : int(
                    removed_offset[variant_index + 1]
                )
            ]
        )
        delta_removed_offset.append(len(delta_removed))
    per_variant_fields = (
        "corridorVariantRow",
        "corridorVariantRank",
        "corridorVariantLocalObjective",
        "corridorVariantLocalObjectiveDelta",
        "corridorVariantPatchCoverage",
        "corridorVariantFirstArcAnchorCount",
        "corridorVariantSecondArcAnchorCount",
        "corridorVariantRetainedBoundaryFraction",
    )
    arrays = {
        name: np.asarray(full[name])[source_index] for name in per_variant_fields
    }
    arrays.update(
        {
            "corridorVariantOffset": corridor_offset,
            "corridorVariantAddedOffset": np.asarray(
                delta_added_offset, dtype=np.int64
            ),
            "corridorVariantAddedFrontierIndex": np.asarray(
                delta_added, dtype=np.int32
            ),
            "corridorVariantRemovedOffset": np.asarray(
                delta_removed_offset, dtype=np.int64
            ),
            "corridorVariantRemovedFrontierIndex": np.asarray(
                delta_removed, dtype=np.int32
            ),
        }
    )
    return arrays, source_index


def _merge_exact_evidence(
    full: Mapping[str, np.ndarray],
    prior: Mapping[str, np.ndarray],
    delta_exact: Mapping[str, np.ndarray],
    prior_to_full: np.ndarray,
    delta_to_full: np.ndarray,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    variant_count = len(np.asarray(full["corridorVariantRow"]))
    merged: dict[str, np.ndarray] = {}
    for name in _EXACT_VARIANT_FIELDS:
        prior_value = np.asarray(prior[name])
        delta_value = np.asarray(delta_exact[name])
        result = np.zeros(variant_count, dtype=prior_value.dtype)
        result[prior_to_full] = prior_value
        result[delta_to_full] = delta_value
        merged[name] = result
    row = np.asarray(full["corridorVariantRow"], dtype=np.int32)
    objective_delta = np.asarray(
        full["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    coverage = np.asarray(
        full["corridorVariantPatchCoverage"], dtype=np.float32
    )
    region_before = merged["corridorVariantTriangleRegionCountBefore"]
    region_after = merged["corridorVariantTriangleRegionCountAfter"]
    area_before = merged["corridorVariantTriangleAreaBefore"]
    area_after = merged["corridorVariantTriangleAreaAfter"]
    eligible = merged["corridorVariantSurfaceEligible"] > 0
    corridor_count = len(np.asarray(full["corridorVariantOffset"])) - 1
    chosen = np.full(corridor_count, -1, dtype=np.int32)
    for corridor_row in np.unique(row[eligible]):
        best_key = (0.0, 0.0, 0.0, 0.0)
        best_variant = -1
        for variant_index in np.flatnonzero(eligible & (row == corridor_row)):
            key = (
                float(region_before[variant_index] - region_after[variant_index]),
                float(area_after[variant_index] - area_before[variant_index]),
                float(objective_delta[variant_index]),
                float(coverage[variant_index]),
            )
            if key > best_key:
                best_key = key
                best_variant = int(variant_index)
        chosen[int(corridor_row)] = best_variant
    merged["corridorChosenExactVariant"] = chosen
    prior_chosen = np.asarray(prior["corridorChosenExactVariant"], dtype=np.int32)
    prior_resolved = set(int(value) for value in np.flatnonzero(prior_chosen >= 0))
    resolved = set(int(value) for value in np.flatnonzero(chosen >= 0))
    newly_resolved = sorted(resolved - prior_resolved)
    return merged, {
        "variantCount": variant_count,
        "exactConnectedVariantCount": int(
            np.count_nonzero(merged["corridorVariantExactConnected"])
        ),
        "surfaceEligibleVariantCount": int(np.count_nonzero(eligible)),
        "corridorWithExactVariantCount": len(resolved),
        "componentSplitVariantCount": int(
            np.count_nonzero(merged["corridorVariantComponentSplit"])
        ),
        "hardConflictVariantCount": int(
            np.count_nonzero(merged["corridorVariantHardConflict"])
        ),
        "newlyResolvedCorridorCount": len(newly_resolved),
        "newlyResolvedCorridorRows": newly_resolved,
        "selectionPriority": (
            "triangle-region reduction, supported area gain, local factor "
            "objective, patch coverage"
        ),
        "identityLabelsUsed": False,
    }


def run_physical_ribbon_corridor_extension(
    prior_variant_root: str | Path,
    configuration_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorExtensionSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorExtensionSettings()
    prior_path, prior_manifest, prior = _load_variant_artifact(
        prior_variant_root
    )
    prior_maximum = int(
        prior_manifest["identity"]["settings"]["maximum_variants_per_corridor"]
    )
    if resolved.maximum_variants_per_corridor <= prior_maximum:
        raise ValueError("extension maximum must exceed the prior variant depth")
    corridor_reference = prior_manifest["identity"]["corridors"]
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_reference["manifestPath"]
    )
    if (
        sha256_file(corridor_path) != corridor_reference["manifestSha256"]
        or corridor_manifest["data"]["sha256"]
        != corridor_reference["dataSha256"]
    ):
        raise ValueError("prior variant corridor input has changed")
    (
        configuration_path,
        configuration_manifest,
        configuration,
        _,
        _,
        topology,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_root)
    expected_configuration = prior_manifest["identity"]["configuration"]
    if (
        sha256_file(configuration_path)
        != expected_configuration["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != expected_configuration["dataSha256"]
    ):
        raise ValueError("prior variants and configuration do not match")
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    prior_variant_values = dict(prior_manifest["identity"]["settings"])
    prior_variant_values["maximum_variants_per_corridor"] = (
        resolved.maximum_variants_per_corridor
    )
    variant_settings = PhysicalRibbonCorridorVariantSettings(
        **prior_variant_values
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_EXTENSION_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_EXTENSION_VERSION,
        "priorVariants": {
            "manifestPath": str(prior_path),
            "manifestSha256": sha256_file(prior_path),
            "dataSha256": prior_manifest["data"]["sha256"],
        },
        "corridors": corridor_reference,
        "configuration": expected_configuration,
        "settings": variant_settings.record(),
        "extensionSettings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "variantImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_variants.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_VARIANTS_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256")
            == sha256_file(data_path)
        ):
            return cached
    started = time.monotonic()
    if progress is not None:
        progress(
            f"enumerating {resolved.maximum_variants_per_corridor} complete "
            "states per CT corridor"
        )
    continuity_weight = float(
        configuration_manifest.get("identity", {})
        .get("settings", {})
        .get("continuity_weight", 0.45)
    )
    full_variants, enumeration_stats = enumerate_corridor_reconfiguration_variants(
        corridor,
        corridor,
        corridor,
        ribbon,
        topology,
        configuration,
        continuity_weight=continuity_weight,
        corridor_settings=corridor_settings,
        settings=variant_settings,
    )
    enumerated_at = time.monotonic()
    full_signature = {
        _variant_signature(full_variants, index): index
        for index in range(len(full_variants["corridorVariantRow"]))
    }
    prior_signature = [
        _variant_signature(prior, index)
        for index in range(len(prior["corridorVariantRow"]))
    ]
    if len(full_signature) != len(full_variants["corridorVariantRow"]):
        raise RuntimeError("extended enumeration produced duplicate variants")
    missing = [value for value in prior_signature if value not in full_signature]
    if missing:
        raise RuntimeError("extended enumeration does not preserve the prior prefix")
    prior_to_full = np.asarray(
        [full_signature[value] for value in prior_signature], dtype=np.int32
    )
    prior_signature_set = set(prior_signature)
    prior_chosen = np.asarray(prior["corridorChosenExactVariant"], dtype=np.int32)
    enumerated_rows = set(
        int(value) for value in np.unique(prior["corridorVariantRow"])
    )
    unresolved_rows = (
        set(int(value) for value in np.flatnonzero(prior_chosen < 0))
        & enumerated_rows
    )
    new_variant_index = np.asarray(
        [
            index
            for index in range(len(full_variants["corridorVariantRow"]))
            if _variant_signature(full_variants, index) not in prior_signature_set
            and int(full_variants["corridorVariantRow"][index]) in unresolved_rows
        ],
        dtype=np.int32,
    )
    delta_variants, delta_to_full = _delta_variant_arrays(
        full_variants,
        new_variant_index,
        corridor_count=len(full_variants["corridorVariantOffset"]) - 1,
    )
    extension_identity = {
        **identity,
        "stage": "targeted-exact-extension",
        "priorPrefixCount": len(prior_to_full),
        "deltaVariantCount": len(delta_to_full),
        "deltaSignatureSha256": canonical_json_hash(
            [
                _variant_signature(full_variants, int(value))
                for value in delta_to_full
            ]
        ),
    }
    extension_identity["identitySha256"] = canonical_json_hash(
        extension_identity
    )
    checkpoint_stem = "corridor-variant-extension-exact-v1"
    cached_exact = None if force else _cached_checkpoint(
        output, checkpoint_stem, extension_identity["identitySha256"]
    )
    if cached_exact is None:
        if progress is not None:
            progress(
                f"exact-screening {len(delta_to_full)} new variants across "
                f"{len(set(int(value) for value in delta_variants['corridorVariantRow']))} "
                "unresolved corridors"
            )
        delta_exact, delta_stats = screen_exact_corridor_variants(
            corridor,
            corridor,
            corridor,
            delta_variants,
            ribbon,
            topology,
            configuration,
            corridor_settings=corridor_settings,
            settings=variant_settings,
            progress=progress,
        )
        checkpoint_path, checkpoint_manifest = _checkpoint(
            output,
            checkpoint_stem,
            extension_identity,
            delta_exact,
            delta_stats,
        )
    else:
        delta_exact, checkpoint_manifest = cached_exact
        delta_stats = checkpoint_manifest["statistics"]
        checkpoint_path = output / f"{checkpoint_stem}.json"
    screened_at = time.monotonic()
    merged_exact, exact_stats = _merge_exact_evidence(
        full_variants,
        prior,
        delta_exact,
        prior_to_full,
        delta_to_full,
    )
    arrays = {**full_variants, **merged_exact}
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_VARIANTS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": corridor_manifest.get("geometry", {}),
        "enumeration": enumeration_stats,
        "exactScreen": exact_stats,
        "extension": {
            "priorMaximumVariantsPerCorridor": prior_maximum,
            "extendedMaximumVariantsPerCorridor": (
                resolved.maximum_variants_per_corridor
            ),
            "priorVariantCount": len(prior_to_full),
            "preservedPriorVariantCount": len(prior_to_full),
            "targetUnresolvedCorridorCount": len(unresolved_rows),
            "newTargetedVariantCount": len(delta_to_full),
            "newExactScreen": delta_stats,
            "checkpoint": checkpoint_path.name,
        },
        "timingSeconds": {
            "enumerationAndPrefixAudit": round(enumerated_at - started, 6),
            "targetedExactScreen": round(screened_at - enumerated_at, 6),
            "mergeAndWrite": round(finished - screened_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "decisionUnit": (
                "new complete joint matchings only for previously unresolved "
                "CT-supported corridors"
            ),
            "priorEvidence": "signature-preserved and copied without reconstruction",
            "mutation": "immutable extension artifact; all inputs remain unchanged",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
