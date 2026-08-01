from __future__ import annotations

import hashlib
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping

import numpy as np

from .contracts import VolumeSource, atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_extension import _delta_variant_arrays
from .physical_ribbon_corridor_frontier import (
    PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM,
    _load_prior_replay,
)
from .physical_ribbon_corridor_sets import (
    PhysicalRibbonCorridorSetSettings,
    optimize_exact_corridor_variant_sets,
)
from .physical_ribbon_corridor_variants import (
    PhysicalRibbonCorridorVariantSettings,
    _cached_checkpoint,
    _checkpoint,
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
    compile_exact_variant_reconfiguration,
    enumerate_corridor_reconfiguration_variants,
    screen_exact_corridor_variants,
)
from .physical_ribbon_patch_corridors import (
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    replay_patch_corridor_reconfigurations,
    solve_patch_corridor_reconfigurations,
    write_patch_corridor_montage,
    write_replayed_corridor_fragment_montage,
)


PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA = (
    "pareidolia.physical-ribbon-one-sided-corridors"
)
PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_VERSION = 1
PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_STEM = (
    "physical-ribbon-one-sided-corridors-v1"
)


def _array_mapping_sha256(arrays: Mapping[str, np.ndarray]) -> str:
    """Hash exact-stage array inputs independently of report serialization.

    The exact corridor screen is the expensive part of this stage.  Its cache
    must be invalidated when any numeric input changes, but not when only the
    surrounding manifest or preview code changes.
    """

    digest = hashlib.sha256()
    for name in sorted(arrays):
        value = np.asarray(arrays[name])
        if value.dtype.hasobject:
            raise TypeError(f"cannot fingerprint object array {name!r}")
        contiguous = np.ascontiguousarray(value)
        name_bytes = name.encode("utf-8")
        dtype_bytes = contiguous.dtype.str.encode("ascii")
        digest.update(len(name_bytes).to_bytes(4, "little"))
        digest.update(name_bytes)
        digest.update(len(dtype_bytes).to_bytes(2, "little"))
        digest.update(dtype_bytes)
        digest.update(len(contiguous.shape).to_bytes(2, "little"))
        digest.update(np.asarray(contiguous.shape, dtype="<i8").tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


@dataclass(frozen=True, slots=True)
class PhysicalRibbonOneSidedCorridorSettings:
    maximum_variants_per_corridor: int = 8
    minimum_one_sided_additions_per_variant: int = 1
    maximum_preview_components: int = 8
    component_assignment_beam_width: int = 256
    maximum_retained_states_per_component: int = 16
    global_assignment_beam_width: int = 4096

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("one-sided corridor variant count must lie in [1, 16]")
        if self.minimum_one_sided_additions_per_variant < 1:
            raise ValueError("one-sided states must add at least one one-sided ribbon")
        if self.maximum_preview_components < 1:
            raise ValueError("preview component count must be positive")
        if self.component_assignment_beam_width < 1:
            raise ValueError("component assignment beam width must be positive")
        if self.maximum_retained_states_per_component < 2:
            raise ValueError("at least two component states must be retained")
        if self.global_assignment_beam_width < 1:
            raise ValueError("global assignment beam width must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_frontier(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "one-sided solve requires a complete label-free corridor frontier"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return (
        manifest_path,
        manifest,
        _load_npz(data_path, manifest["data"]["sha256"]),
    )


def _variant_one_sided_addition_count(
    variants: Mapping[str, np.ndarray],
    frontier_bank_index: np.ndarray,
    one_sided_bank_mask: np.ndarray,
) -> np.ndarray:
    offset = np.asarray(variants["corridorVariantAddedOffset"], dtype=np.int64)
    value = np.asarray(
        variants["corridorVariantAddedFrontierIndex"], dtype=np.int32
    )
    result = np.zeros(len(offset) - 1, dtype=np.int16)
    for variant_index in range(len(result)):
        added = value[int(offset[variant_index]) : int(offset[variant_index + 1])]
        if len(added):
            result[variant_index] = int(
                np.count_nonzero(
                    one_sided_bank_mask[frontier_bank_index[added]]
                )
            )
    return result


def _load_stage_inputs(
    frontier_manifest: Mapping[str, Any],
) -> tuple[
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
    dict[str, np.ndarray],
]:
    corridor_reference = frontier_manifest["identity"]["corridors"]
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_reference["manifestPath"]
    )
    if (
        sha256_file(corridor_path) != corridor_reference["manifestSha256"]
        or corridor_manifest["data"]["sha256"]
        != corridor_reference["dataSha256"]
    ):
        raise ValueError("corridor frontier source corridors changed")
    prior_reference = frontier_manifest["identity"]["priorReplay"]
    prior_path, prior_manifest, prior = _load_prior_replay(
        prior_reference["manifestPath"]
    )
    if (
        sha256_file(prior_path) != prior_reference["manifestSha256"]
        or prior_manifest["data"]["sha256"] != prior_reference["dataSha256"]
    ):
        raise ValueError("corridor frontier prior replay changed")
    configuration_reference = frontier_manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        base_configuration,
        _,
        _,
        _,
        _,
        _,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path)
        != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("corridor frontier base configuration changed")
    return (
        corridor_path,
        corridor_manifest,
        corridor,
        prior_path,
        prior_manifest,
        prior,
        configuration_path,
        configuration_manifest,
        base_configuration,
        ribbon,
    )


def run_physical_ribbon_one_sided_corridors(
    frontier_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonOneSidedCorridorSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonOneSidedCorridorSettings()
    frontier_path, frontier_manifest, frontier = _load_frontier(frontier_root)
    (
        corridor_path,
        corridor_manifest,
        corridor,
        prior_path,
        prior_manifest,
        prior,
        configuration_path,
        configuration_manifest,
        base_configuration,
        ribbon,
    ) = _load_stage_inputs(frontier_manifest)
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_VERSION,
        "frontier": {
            "manifestPath": str(frontier_path),
            "manifestSha256": sha256_file(frontier_path),
            "dataSha256": frontier_manifest["data"]["sha256"],
        },
        "corridors": {
            "manifestPath": str(corridor_path),
            "manifestSha256": sha256_file(corridor_path),
            "dataSha256": corridor_manifest["data"]["sha256"],
        },
        "priorReplay": {
            "manifestPath": str(prior_path),
            "manifestSha256": sha256_file(prior_path),
            "dataSha256": prior_manifest["data"]["sha256"],
        },
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "variantImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_variants.py")
        ),
        "setImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_sets.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_STEM}.npz"
    montage_path = output / "physical-ribbon-one-sided-corridors.png"
    fragment_path = output / "physical-ribbon-one-sided-fragments.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    variant_settings = PhysicalRibbonCorridorVariantSettings(
        maximum_variants_per_corridor=resolved.maximum_variants_per_corridor,
        minimum_variant_patch_coverage=0.45,
        minimum_anchor_count_per_arc=1,
        minimum_surface_area_retention=0.98,
        maximum_preview_components=resolved.maximum_preview_components,
    )
    set_settings = PhysicalRibbonCorridorSetSettings(
        component_assignment_beam_width=(
            resolved.component_assignment_beam_width
        ),
        maximum_retained_states_per_component=(
            resolved.maximum_retained_states_per_component
        ),
        global_assignment_beam_width=resolved.global_assignment_beam_width,
        minimum_surface_area_retention=0.98,
        maximum_preview_components=resolved.maximum_preview_components,
    )
    started = time.monotonic()
    if progress is not None:
        progress("reconstructing the cumulative surface in the targeted frontier")
    surface, surface_stats = build_physical_ribbon_surface_complex(
        ribbon,
        frontier,
        frontier,
        settings=corridor_settings.surface_settings(),
    )
    original_to_target = np.asarray(
        frontier["originalFrontierToTargetFrontier"], dtype=np.int32
    )
    remapped = _remap_corridor_surface(
        corridor,
        surface,
        base_configuration,
        frontier,
        original_to_target,
    )
    surfaced_at = time.monotonic()
    if progress is not None:
        progress("solving complete residual matchings with one-sided support")
    continuity_weight = float(
        configuration_manifest["identity"]["settings"]["continuity_weight"]
    )
    reconfiguration, reconfiguration_stats = solve_patch_corridor_reconfigurations(
        remapped,
        remapped,
        ribbon,
        frontier,
        frontier,
        continuity_weight=continuity_weight,
        settings=corridor_settings,
    )
    reconfigured_at = time.monotonic()
    full_variants, enumeration_stats = enumerate_corridor_reconfiguration_variants(
        remapped,
        remapped,
        reconfiguration,
        ribbon,
        frontier,
        frontier,
        continuity_weight=continuity_weight,
        corridor_settings=corridor_settings,
        settings=variant_settings,
    )
    frontier_bank = np.asarray(
        frontier["frontierRibbonCandidate"], dtype=np.int32
    )
    bank_count = len(np.asarray(ribbon["sourceInterface"]))
    one_sided_bank_mask = np.zeros(bank_count, dtype=bool)
    one_sided_bank_mask[
        np.asarray(frontier["oneSidedCandidateBankIndex"], dtype=np.int32)
    ] = True
    one_sided_count = _variant_one_sided_addition_count(
        full_variants, frontier_bank, one_sided_bank_mask
    )
    target_rows = set(
        int(value) for value in np.asarray(frontier["targetCorridorRow"])
    )
    full_row = np.asarray(full_variants["corridorVariantRow"], dtype=np.int32)
    target_index = np.asarray(
        [
            index
            for index, row in enumerate(full_row)
            if int(row) in target_rows
            and int(one_sided_count[index])
            >= resolved.minimum_one_sided_additions_per_variant
        ],
        dtype=np.int32,
    )
    target_variants, target_to_full = _delta_variant_arrays(
        full_variants,
        target_index,
        corridor_count=len(full_variants["corridorVariantOffset"]) - 1,
    )
    target_one_sided_count = one_sided_count[target_to_full]
    enumerated_at = time.monotonic()
    checkpoint_identity = {
        "schema": PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA,
        "stage": "targeted-exact-screen",
        "version": 2,
        "frontierDataSha256": frontier_manifest["data"]["sha256"],
        "ribbonBankDataSha256": frontier_manifest["identity"]["ribbonBank"][
            "dataSha256"
        ],
        "remappedArraysSha256": _array_mapping_sha256(remapped),
        "targetVariantArraysSha256": _array_mapping_sha256(target_variants),
        "corridorSettings": corridor_settings.record(),
        "variantSettings": variant_settings.record(),
        "exactImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_variants.py")
        ),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_corridors.py")
        ),
        "configurationImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_configuration.py")
        ),
        "variantRows": [
            int(value) for value in target_variants["corridorVariantRow"]
        ],
        "variantRanks": [
            int(value) for value in target_variants["corridorVariantRank"]
        ],
        "variantOneSidedCounts": [
            int(value) for value in target_one_sided_count
        ],
    }
    checkpoint_identity["identitySha256"] = canonical_json_hash(
        checkpoint_identity
    )
    cached_exact = None if force else _cached_checkpoint(
        output,
        "one-sided-corridor-exact-v1",
        checkpoint_identity["identitySha256"],
    )
    if cached_exact is None:
        if progress is not None:
            progress(
                f"exact-screening {len(target_to_full)} one-sided states across "
                f"{len(set(int(value) for value in target_variants['corridorVariantRow']))} corridors"
            )
        exact, exact_stats = screen_exact_corridor_variants(
            remapped,
            remapped,
            remapped,
            target_variants,
            ribbon,
            frontier,
            frontier,
            corridor_settings=corridor_settings,
            settings=variant_settings,
            progress=progress,
        )
        checkpoint_path, _ = _checkpoint(
            output,
            "one-sided-corridor-exact-v1",
            checkpoint_identity,
            exact,
            exact_stats,
        )
    else:
        exact, checkpoint_manifest = cached_exact
        exact_stats = checkpoint_manifest["statistics"]
        checkpoint_path = output / "one-sided-corridor-exact-v1.json"
    screened_at = time.monotonic()
    exact_variants = {**target_variants, **exact}
    if progress is not None:
        progress("optimizing exact one-sided corridor states jointly")
    selected_sets, set_stats = optimize_exact_corridor_variant_sets(
        remapped,
        remapped,
        remapped,
        exact_variants,
        ribbon,
        frontier,
        frontier,
        corridor_settings=corridor_settings,
        settings=set_settings,
        progress=progress,
    )
    solved_at = time.monotonic()
    exact_override = dict(exact_variants)
    exact_override["corridorChosenExactVariant"] = selected_sets[
        "corridorChosenGlobalVariant"
    ]
    compiled = compile_exact_variant_reconfiguration(
        reconfiguration, target_variants, exact_override
    )
    if progress is not None:
        progress("replaying the one-sided assignment on the cumulative surface")
    replay, replay_stats = replay_patch_corridor_reconfigurations(
        remapped,
        remapped,
        remapped,
        compiled,
        ribbon,
        frontier,
        frontier,
        settings=corridor_settings,
    )
    replayed_at = time.monotonic()
    arrays = {
        **reconfiguration,
        **target_variants,
        **exact,
        **selected_sets,
        **replay,
        "targetVariantFullEnumerationIndex": target_to_full,
        "targetVariantOneSidedAdditionCount": target_one_sided_count,
    }
    _write_npz(data_path, arrays)
    source_record = configuration_manifest["source"]
    source = VolumeSource.open(
        source_record["path"], source_record.get("metadataPath")
    )
    write_patch_corridor_montage(
        remapped,
        remapped,
        montage_path,
        maximum_corridors=corridor_settings.maximum_preview_corridors,
        reconfiguration=compiled,
        replay=replay,
    )
    _, fragment_stats = write_replayed_corridor_fragment_montage(
        remapped,
        remapped,
        remapped,
        replay,
        source,
        fragment_path,
        maximum_components=resolved.maximum_preview_components,
    )
    finished = time.monotonic()
    successful_rows = np.flatnonzero(
        np.asarray(replay["corridorReplayProposalSuccessful"]) > 0
    )
    # The targeted frontier has already resolved whether its corridor catalog
    # is the same catalog as the prior replay.  A cumulative hole replay (or a
    # refreshed corridor census) intentionally has no row-aligned success
    # mask, so carrying the frontier's audited count avoids coupling unrelated
    # catalogs at serialization time.
    prior_success_count = int(
        frontier_manifest["targets"]["priorSuccessfulCorridorCount"]
    )
    baseline_triangle = np.asarray(surface["triangleFrontierIndex"], dtype=np.int32)
    final_triangle = np.asarray(
        replay["corridorReplayTriangleFrontierIndex"], dtype=np.int32
    )
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_SCHEMA,
        "version": PHYSICAL_RIBBON_ONE_SIDED_CORRIDORS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": source_record,
        "geometry": frontier_manifest["geometry"],
        "surface": surface_stats,
        "reconfiguration": reconfiguration_stats,
        "enumeration": enumeration_stats,
        "target": {
            "priorSuccessfulCorridorCount": prior_success_count,
            "residualCorridorCount": len(target_rows),
            "oneSidedVariantCount": len(target_to_full),
            "corridorWithOneSidedVariantCount": len(
                set(int(value) for value in target_variants["corridorVariantRow"])
            ),
            "exactResolvedCorridorCount": int(
                np.count_nonzero(exact["corridorChosenExactVariant"] >= 0)
            ),
            "globallySelectedNewCorridorCount": len(
                selected_sets["globalChosenVariantIndex"]
            ),
            "successfulNewCorridorCount": len(successful_rows),
            "successfulNewCorridorRows": [
                int(value) for value in successful_rows
            ],
            "cumulativeSuccessfulCorridorCount": (
                prior_success_count + len(successful_rows)
            ),
        },
        "exactScreen": {**exact_stats, "checkpoint": checkpoint_path.name},
        "componentSetOptimization": set_stats,
        "counterfactualReplay": replay_stats,
        "surfaceAudit": {
            "edgeConnectedTriangleRegionCountBefore": int(
                len(np.unique(_triangle_region_labels(baseline_triangle)))
            ),
            "edgeConnectedTriangleRegionCountAfter": int(
                len(np.unique(_triangle_region_labels(final_triangle)))
            ),
            "retainedTriangleCountBefore": len(baseline_triangle),
            "retainedTriangleCountAfter": len(final_triangle),
        },
        "flattenedReplayFragments": fragment_stats,
        "timingSeconds": {
            "surfaceAndRemap": round(surfaced_at - started, 6),
            "corridorReconfiguration": round(
                reconfigured_at - surfaced_at, 6
            ),
            "variantEnumeration": round(enumerated_at - reconfigured_at, 6),
            "exactScreen": round(screened_at - enumerated_at, 6),
            "componentAndGlobalOptimization": round(
                solved_at - screened_at, 6
            ),
            "replay": round(replayed_at - solved_at, 6),
            "writingAndPreviews": round(finished - replayed_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "corridorMontage": montage_path.name,
            "flattenedReplayFragments": fragment_path.name,
        },
        "method": {
            "decisionUnit": (
                "complete CT-supported residual-corridor matching containing "
                "at least one explicitly targeted one-sided ribbon"
            ),
            "acceptance": (
                "exact full-sheet connection, joint component/block assignment, "
                "and density-preserving cumulative replay"
            ),
            "singleCellGrowth": False,
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
