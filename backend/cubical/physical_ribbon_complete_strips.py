from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_corridor_dormant import _remap_corridor_surface
from .physical_ribbon_corridor_extension import _delta_variant_arrays
from .physical_ribbon_corridor_face_replay import (
    PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_SCHEMA,
    PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_STEM,
    _hard_conflict_counts,
)
from .physical_ribbon_corridor_faces import (
    PhysicalRibbonCorridorFaceSettings,
    _finite_distribution,
    _screen_corridor_face_path,
)
from .physical_ribbon_corridor_one_sided import (
    _load_frontier,
    _load_stage_inputs,
)
from .physical_ribbon_corridor_sets import _variant_values
from .physical_ribbon_corridor_variants import (
    PhysicalRibbonCorridorVariantSettings,
    _corridor_settings_from_manifest,
    enumerate_corridor_reconfiguration_variants,
)
from .physical_ribbon_patch_corridors import (
    _evaluate_corridor_connections,
    _triangle_region_labels,
    build_physical_ribbon_surface_complex,
    solve_patch_corridor_reconfigurations,
)
from .physical_ribbon_replay_configuration import _load_replay_artifact


PHYSICAL_RIBBON_COMPLETE_STRIPS_SCHEMA = (
    "pareidolia.physical-ribbon-complete-strips"
)
PHYSICAL_RIBBON_COMPLETE_STRIPS_VERSION = 1
PHYSICAL_RIBBON_COMPLETE_STRIPS_STEM = "physical-ribbon-complete-strips-v1"


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCompleteStripSettings:
    """Whole-strip assignment and CT-face screening settings.

    Candidate provenance is deliberately absent from this contract. A complete
    both-arc matching may use bidirectional or one-sided ribbons; its physical
    support, topology, and dense CT strip decide whether it survives.
    """

    maximum_variants_per_corridor: int = 16
    minimum_variant_patch_coverage: float = 0.45
    minimum_anchor_count_per_arc: int = 1
    minimum_strict_surface_area_retention: float = 0.98
    minimum_preclosure_surface_area_retention: float = 0.95

    def __post_init__(self) -> None:
        if not 1 <= self.maximum_variants_per_corridor <= 16:
            raise ValueError("complete-strip variant count must lie in [1, 16]")
        if not 0.0 < self.minimum_variant_patch_coverage <= 1.0:
            raise ValueError("complete-strip patch coverage must lie in (0, 1]")
        if self.minimum_anchor_count_per_arc < 1:
            raise ValueError("complete strips require an anchor on both arcs")
        if not 0.0 < self.minimum_strict_surface_area_retention <= 1.0:
            raise ValueError("strict surface-area retention must lie in (0, 1]")
        if not 0.0 < self.minimum_preclosure_surface_area_retention <= (
            self.minimum_strict_surface_area_retention
        ):
            raise ValueError(
                "preclosure area retention must be positive and no larger "
                "than final area retention"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_face_replay_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value
        if value.is_file()
        else value / f"{PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != PHYSICAL_RIBBON_CORRIDOR_FACE_REPLAY_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "complete strips require a complete label-free corridor-face replay"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return manifest_path, manifest, _load_npz(
        data_path, manifest["data"]["sha256"]
    )


def _strict_surface(
    replay: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    strict_count = int(
        np.asarray(replay["baseStrictTriangleCount"]).reshape(-1)[0]
    )
    result = {key: np.asarray(value) for key, value in replay.items()}
    for key in (
        "triangleFrontierIndex",
        "triangleAreaVoxelsSquared",
        "triangleNormalResidualDegrees",
    ):
        result[key] = result[key][:strict_count]
    return result


def _residual_corridor_rows(
    replay_manifest: Mapping[str, Any],
    prior_exact: Mapping[str, np.ndarray],
) -> np.ndarray:
    evidence = np.asarray(prior_exact["corridorEvidenceEligible"]) > 0
    prior_success = (
        np.asarray(prior_exact["corridorReplayProposalSuccessful"]) > 0
    )
    face_success = np.zeros(len(evidence), dtype=bool)
    chosen = np.asarray(
        replay_manifest.get("optimization", {}).get("chosenCorridorRows", ()),
        dtype=np.int32,
    )
    if len(chosen):
        face_success[chosen] = True
    return np.flatnonzero(evidence & ~prior_success & ~face_success).astype(
        np.int32
    )


def _split_audit(
    local_selected: np.ndarray,
    local_component: np.ndarray,
    original_component: np.ndarray,
    component_id: int,
) -> dict[str, Any]:
    retained = local_selected & (original_component == component_id)
    labels, counts = np.unique(
        local_component[retained][local_component[retained] >= 0],
        return_counts=True,
    )
    total = int(np.sum(counts))
    order = np.argsort(-counts, kind="stable") if len(counts) else np.empty(0)
    ordered = counts[order] if len(counts) else counts
    return {
        "split": len(labels) != 1,
        "descendantCount": len(labels),
        "descendantRibbonCounts": [int(value) for value in ordered],
        "largestDescendantFraction": (
            float(ordered[0] / total) if total and len(ordered) else 0.0
        ),
        "detachedRibbonCount": (
            int(total - ordered[0]) if total and len(ordered) else total
        ),
    }


def _selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    if not record.get("eligible"):
        return (-math.inf,)
    strict = bool(record.get("strictConnected"))
    path_cost = float(record.get("physicalPathCost") or 0.0)
    path_faces = float(record.get("physicalPathFaceCount") or 0.0)
    return (
        1.0 if strict else 0.0,
        -path_cost,
        -path_faces,
        float(record["triangleRegionCountBefore"])
        - float(record["triangleRegionCountAfter"]),
        float(record.get("augmentedAreaRetention", record["strictAreaRetention"])),
        float(record["strictAreaRetention"]),
        float(record["localObjectiveDelta"]),
        float(record["patchCoverage"]),
        -float(record["variantRank"]),
    )


def _area_retention_decision(
    strict_retention: float,
    augmented_retention: float,
    settings: PhysicalRibbonCompleteStripSettings,
) -> str:
    if strict_retention >= settings.minimum_strict_surface_area_retention:
        return "strict-area-sufficient"
    if (
        strict_retention >= settings.minimum_preclosure_surface_area_retention
        and augmented_retention
        >= settings.minimum_strict_surface_area_retention
    ):
        return "ct-closure-replaces-area"
    return "insufficient-area"


def _screen_complete_strip_variants(
    rows: Sequence[int],
    variants: Mapping[str, np.ndarray],
    corridor: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    configuration: Mapping[str, np.ndarray],
    baseline_surface: Mapping[str, np.ndarray],
    *,
    surface_settings: Any,
    face_settings: PhysicalRibbonCorridorFaceSettings,
    settings: PhysicalRibbonCompleteStripSettings,
    progress: Callable[[str], None] | None = None,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    variant_count = len(np.asarray(variants["corridorVariantRow"]))
    hard_conflict = np.zeros(variant_count, dtype=np.uint8)
    component_split = np.zeros(variant_count, dtype=np.uint8)
    descendant_count = np.zeros(variant_count, dtype=np.int16)
    detached_ribbon_count = np.zeros(variant_count, dtype=np.int32)
    largest_descendant_fraction = np.zeros(variant_count, dtype=np.float32)
    strict_connected = np.zeros(variant_count, dtype=np.uint8)
    strict_area_retention = np.zeros(variant_count, dtype=np.float32)
    augmented_area_retention = np.zeros(variant_count, dtype=np.float32)
    closure_area = np.zeros(variant_count, dtype=np.float32)
    region_before = np.zeros(variant_count, dtype=np.int32)
    region_after = np.zeros(variant_count, dtype=np.int32)
    triangle_before = np.zeros(variant_count, dtype=np.int32)
    triangle_after = np.zeros(variant_count, dtype=np.int32)
    physical_eligible = np.zeros(variant_count, dtype=np.uint8)
    physical_path_count = np.zeros(variant_count, dtype=np.int16)
    physical_path_cost = np.full(variant_count, np.inf, dtype=np.float32)
    physical_candidate_count = np.zeros(variant_count, dtype=np.int32)
    physical_closure_count = np.zeros(variant_count, dtype=np.int32)
    shared_arc_fraction = np.zeros(variant_count, dtype=np.float32)

    selected_baseline = np.asarray(configuration["selected"]) > 0
    original_component = np.asarray(configuration["component"], dtype=np.int32)
    first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    baseline_triangle = np.asarray(
        baseline_surface["triangleFrontierIndex"], dtype=np.int32
    )
    baseline_area = np.asarray(
        baseline_surface["triangleAreaVoxelsSquared"], dtype=np.float32
    )
    baseline_region = _triangle_region_labels(baseline_triangle)
    scored_corridor = np.asarray(corridor["scoredCorridorIndex"], dtype=np.int32)
    corridor_component = np.asarray(
        corridor["corridorTopologyComponent"], dtype=np.int32
    )
    variant_offset = np.asarray(variants["corridorVariantOffset"], dtype=np.int64)
    variant_rank = np.asarray(variants["corridorVariantRank"], dtype=np.int16)
    objective_delta = np.asarray(
        variants["corridorVariantLocalObjectiveDelta"], dtype=np.float32
    )
    coverage = np.asarray(
        variants["corridorVariantPatchCoverage"], dtype=np.float32
    )
    records: list[dict[str, Any]] = []
    row_records: list[dict[str, Any]] = []

    for completed, row_value in enumerate(rows, start=1):
        row = int(row_value)
        component_id = int(corridor_component[int(scored_corridor[row])])
        base_index = np.flatnonzero(
            np.all(original_component[baseline_triangle] == component_id, axis=1)
        )
        base_area = float(np.sum(baseline_area[base_index]))
        base_regions = (
            len(np.unique(baseline_region[base_index])) if len(base_index) else 0
        )
        current_row_records: list[dict[str, Any]] = []
        for variant_index in range(
            int(variant_offset[row]), int(variant_offset[row + 1])
        ):
            added = _variant_values(variants, variant_index, "Added")
            removed = _variant_values(variants, variant_index, "Removed")
            selected = selected_baseline.copy()
            selected[removed] = False
            selected[added] = True
            interface_conflict, crossing_conflict = _hard_conflict_counts(
                selected, topology, ribbon
            )
            hard = bool(interface_conflict or crossing_conflict)
            hard_conflict[variant_index] = int(hard)
            local_selected = selected & (original_component == component_id)
            local_selected[added] = selected[added]
            local_component, _ = _component_labels(local_selected, first, second)
            split = _split_audit(
                local_selected,
                local_component,
                original_component,
                component_id,
            )
            component_split[variant_index] = int(split["split"])
            descendant_count[variant_index] = int(split["descendantCount"])
            detached_ribbon_count[variant_index] = int(
                split["detachedRibbonCount"]
            )
            largest_descendant_fraction[variant_index] = float(
                split["largestDescendantFraction"]
            )
            record: dict[str, Any] = {
                "corridorRow": row,
                "variantIndex": variant_index,
                "variantRank": int(variant_rank[variant_index]),
                "addedRibbonCount": len(added),
                "removedRibbonCount": len(removed),
                "localObjectiveDelta": round(
                    float(objective_delta[variant_index]), 6
                ),
                "patchCoverage": round(float(coverage[variant_index]), 6),
                "interfaceConflictCount": interface_conflict,
                "crossingConflictCount": crossing_conflict,
                "componentSplit": split,
                "strictConnected": False,
                "strictAreaRetention": 0.0,
                "ctClosureAreaVoxelsSquared": 0.0,
                "augmentedAreaRetention": 0.0,
                "areaRetentionDecision": "unreconstructed",
                "triangleRegionCountBefore": base_regions,
                "triangleRegionCountAfter": 0,
                "triangleCountBefore": len(base_index),
                "triangleCountAfter": 0,
                "physicalPathFaceCount": 0,
                "physicalPathCost": None,
                "physicalCandidateFaceCount": 0,
                "attachedPhysicalClosureFaceCount": 0,
                "sharedArcRegionFraction": 0.0,
                "eligible": False,
            }
            if hard or split["split"]:
                current_row_records.append(record)
                records.append(record)
                continue
            local_configuration = dict(configuration)
            local_configuration["selected"] = local_selected.astype(np.uint8)
            local_configuration["component"] = local_component
            local_surface, _ = build_physical_ribbon_surface_complex(
                ribbon,
                topology,
                local_configuration,
                settings=surface_settings,
            )
            local_triangle = np.asarray(
                local_surface["triangleFrontierIndex"], dtype=np.int32
            )
            local_regions = (
                len(np.unique(_triangle_region_labels(local_triangle)))
                if len(local_triangle)
                else 0
            )
            local_area = float(
                np.sum(local_surface["triangleAreaVoxelsSquared"])
            )
            retention = local_area / max(base_area, 1.0e-6)
            strict_connection = _evaluate_corridor_connections(
                local_surface,
                corridor,
                corridor,
                minimum_arc_region_fraction=(
                    face_settings.minimum_arc_region_fraction
                ),
                maximum_arc_triangle_distance_edges=(
                    face_settings.maximum_arc_triangle_distance_edges
                ),
            )
            is_strict_connected = bool(
                strict_connection["boundaryArcsConnected"][row]
            )
            strict_connected[variant_index] = int(is_strict_connected)
            strict_area_retention[variant_index] = retention
            region_before[variant_index] = base_regions
            region_after[variant_index] = local_regions
            triangle_before[variant_index] = len(base_index)
            triangle_after[variant_index] = len(local_triangle)
            if is_strict_connected:
                screened = {
                    "physicalPathFaceCount": 0,
                    "physicalPathCost": 0.0,
                    "physicalCandidateFaceCount": 0,
                    "attachedPhysicalClosureFaceCount": 0,
                    "sharedArcRegionFraction": float(
                        strict_connection["boundaryArcSharedRegionFraction"][row]
                    ),
                    "eligible": True,
                }
                screened_arrays: dict[str, np.ndarray] = {}
            else:
                screened, screened_arrays = _screen_corridor_face_path(
                    row,
                    local_surface,
                    corridor,
                    surface_settings=surface_settings,
                    settings=face_settings,
                )
            candidate_closure_area = 0.0
            if screened_arrays:
                closure_mask = np.asarray(
                    screened_arrays["candidatePhysicalClosure"], dtype=np.uint8
                ) > 0
                candidate_closure_area = float(
                    np.sum(
                        np.asarray(
                            screened_arrays["candidateAreaVoxelsSquared"],
                            dtype=np.float32,
                        )[closure_mask]
                    )
                )
            augmented_retention = (
                local_area + candidate_closure_area
            ) / max(base_area, 1.0e-6)
            area_decision = _area_retention_decision(
                retention, augmented_retention, settings
            )
            eligible = bool(
                screened["eligible"]
                and area_decision != "insufficient-area"
                and local_regions <= base_regions
            )
            augmented_area_retention[variant_index] = augmented_retention
            closure_area[variant_index] = candidate_closure_area
            physical_eligible[variant_index] = int(eligible)
            physical_path_count[variant_index] = int(
                screened["physicalPathFaceCount"]
            )
            if screened["physicalPathCost"] is not None:
                physical_path_cost[variant_index] = float(
                    screened["physicalPathCost"]
                )
            physical_candidate_count[variant_index] = int(
                screened["physicalCandidateFaceCount"]
            )
            physical_closure_count[variant_index] = int(
                screened["attachedPhysicalClosureFaceCount"]
            )
            shared_arc_fraction[variant_index] = float(
                screened["sharedArcRegionFraction"]
            )
            record.update(
                {
                    "strictConnected": is_strict_connected,
                    "strictAreaRetention": round(retention, 6),
                    "ctClosureAreaVoxelsSquared": round(
                        candidate_closure_area, 6
                    ),
                    "augmentedAreaRetention": round(
                        augmented_retention, 6
                    ),
                    "areaRetentionDecision": area_decision,
                    "triangleRegionCountAfter": local_regions,
                    "triangleCountAfter": len(local_triangle),
                    "physicalPathFaceCount": int(
                        screened["physicalPathFaceCount"]
                    ),
                    "physicalPathCost": (
                        round(float(screened["physicalPathCost"]), 6)
                        if screened["physicalPathCost"] is not None
                        else None
                    ),
                    "physicalCandidateFaceCount": int(
                        screened["physicalCandidateFaceCount"]
                    ),
                    "attachedPhysicalClosureFaceCount": int(
                        screened["attachedPhysicalClosureFaceCount"]
                    ),
                    "sharedArcRegionFraction": round(
                        float(screened["sharedArcRegionFraction"]), 6
                    ),
                    "eligible": eligible,
                }
            )
            current_row_records.append(record)
            records.append(record)
        eligible_records = [
            value for value in current_row_records if value["eligible"]
        ]
        chosen = (
            max(eligible_records, key=_selection_key)
            if eligible_records
            else None
        )
        if chosen is not None:
            status = "physical-complete-strip"
        elif current_row_records and all(
            value["componentSplit"]["split"] for value in current_row_records
        ):
            status = "all-complete-strips-split-component"
        elif current_row_records and all(
            value["interfaceConflictCount"] or value["crossingConflictCount"]
            for value in current_row_records
        ):
            status = "all-complete-strips-conflict"
        elif current_row_records:
            status = "no-physical-complete-strip"
        else:
            status = "no-complete-strip"
        row_records.append(
            {
                "corridorRow": row,
                "status": status,
                "variantCount": len(current_row_records),
                "componentPreservingVariantCount": sum(
                    not value["componentSplit"]["split"]
                    and not value["interfaceConflictCount"]
                    and not value["crossingConflictCount"]
                    for value in current_row_records
                ),
                "eligibleVariantCount": len(eligible_records),
                "chosenVariantIndex": (
                    int(chosen["variantIndex"]) if chosen is not None else -1
                ),
                "chosen": chosen,
            }
        )
        if progress is not None:
            progress(
                f"complete strips {completed}/{len(rows)} · row {row} · "
                f"{status} · eligible {len(eligible_records)}"
            )

    arrays = {
        "corridorVariantHardConflict": hard_conflict,
        "corridorVariantComponentSplit": component_split,
        "corridorVariantDescendantCount": descendant_count,
        "corridorVariantDetachedRibbonCount": detached_ribbon_count,
        "corridorVariantLargestDescendantFraction": (
            largest_descendant_fraction
        ),
        "corridorVariantStrictConnected": strict_connected,
        "corridorVariantStrictAreaRetention": strict_area_retention,
        "corridorVariantCtClosureAreaVoxelsSquared": closure_area,
        "corridorVariantAugmentedAreaRetention": augmented_area_retention,
        "corridorVariantTriangleRegionCountBefore": region_before,
        "corridorVariantTriangleRegionCountAfter": region_after,
        "corridorVariantTriangleCountBefore": triangle_before,
        "corridorVariantTriangleCountAfter": triangle_after,
        "corridorVariantPhysicalEligible": physical_eligible,
        "corridorVariantPhysicalPathFaceCount": physical_path_count,
        "corridorVariantPhysicalPathCost": physical_path_cost,
        "corridorVariantPhysicalCandidateFaceCount": physical_candidate_count,
        "corridorVariantPhysicalClosureFaceCount": physical_closure_count,
        "corridorVariantSharedArcRegionFraction": shared_arc_fraction,
        "corridorChosenCompleteStripVariant": np.asarray(
            [value["chosenVariantIndex"] for value in row_records],
            dtype=np.int32,
        ),
    }
    status_count = Counter(value["status"] for value in row_records)
    stats = {
        "targetCorridorCount": len(rows),
        "variantCount": variant_count,
        "hardConflictVariantCount": int(np.count_nonzero(hard_conflict)),
        "componentSplitVariantCount": int(np.count_nonzero(component_split)),
        "componentPreservingVariantCount": int(
            np.count_nonzero(~component_split.astype(bool) & ~hard_conflict.astype(bool))
        ),
        "strictConnectedVariantCount": int(np.count_nonzero(strict_connected)),
        "physicalEligibleVariantCount": int(np.count_nonzero(physical_eligible)),
        "ctClosureAreaReplacementVariantCount": sum(
            value["eligible"]
            and value["areaRetentionDecision"] == "ct-closure-replaces-area"
            for value in records
        ),
        "corridorWithPhysicalCompletionCount": sum(
            value["eligibleVariantCount"] > 0 for value in row_records
        ),
        "statusCounts": dict(sorted(status_count.items())),
        "eligiblePathFaceCount": _finite_distribution(
            physical_path_count[physical_eligible > 0]
        ),
        "eligiblePathCost": _finite_distribution(
            physical_path_cost[physical_eligible > 0]
        ),
        "rows": row_records,
        "variants": records,
        "identityLabelsUsed": False,
    }
    return arrays, records, stats


def run_physical_ribbon_complete_strips(
    replay_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCompleteStripSettings | None = None,
    force: bool = False,
    progress: Callable[[str], None] | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCompleteStripSettings()
    replay_path, replay_manifest, replay = _load_face_replay_artifact(replay_root)
    prior_reference = replay_manifest["identity"]["replay"]
    prior_path, prior_manifest, prior_exact = _load_replay_artifact(
        prior_reference["manifestPath"]
    )
    if (
        sha256_file(prior_path) != prior_reference["manifestSha256"]
        or prior_manifest["data"]["sha256"] != prior_reference["dataSha256"]
    ):
        raise ValueError("complete-strip prior exact replay has changed")
    frontier_path, frontier_manifest, frontier = _load_frontier(
        replay_manifest["identity"]["frontier"]["manifestPath"]
    )
    (
        corridor_path,
        corridor_manifest,
        corridor,
        _,
        _,
        _,
        configuration_path,
        configuration_manifest,
        base_configuration,
        ribbon,
    ) = _load_stage_inputs(frontier_manifest)
    face_settings = PhysicalRibbonCorridorFaceSettings(
        **replay_manifest["identity"]["faceSettings"]
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_COMPLETE_STRIPS_SCHEMA,
        "version": PHYSICAL_RIBBON_COMPLETE_STRIPS_VERSION,
        "replay": {
            "manifestPath": str(replay_path),
            "manifestSha256": sha256_file(replay_path),
            "dataSha256": replay_manifest["data"]["sha256"],
        },
        "priorExactReplay": prior_reference,
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
        "configuration": {
            "manifestPath": str(configuration_path),
            "manifestSha256": sha256_file(configuration_path),
            "dataSha256": configuration_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "faceSettings": face_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "variantImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_variants.py")
        ),
        "faceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_faces.py")
        ),
        "surfaceImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_patch_holes.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_COMPLETE_STRIPS_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_COMPLETE_STRIPS_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    baseline_surface = _strict_surface(replay)
    configuration = dict(frontier)
    configuration["selected"] = np.asarray(replay["selected"], dtype=np.uint8)
    configuration["component"] = np.asarray(replay["component"], dtype=np.int32)
    remapped = _remap_corridor_surface(
        corridor,
        baseline_surface,
        base_configuration,
        configuration,
        np.asarray(frontier["originalFrontierToTargetFrontier"], dtype=np.int32),
    )
    target_rows = _residual_corridor_rows(replay_manifest, prior_exact)
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    variant_settings = PhysicalRibbonCorridorVariantSettings(
        maximum_variants_per_corridor=resolved.maximum_variants_per_corridor,
        minimum_variant_patch_coverage=(
            resolved.minimum_variant_patch_coverage
        ),
        minimum_anchor_count_per_arc=resolved.minimum_anchor_count_per_arc,
        minimum_surface_area_retention=(
            resolved.minimum_strict_surface_area_retention
        ),
        maximum_preview_components=1,
    )
    continuity_weight = float(
        configuration_manifest["identity"]["settings"]["continuity_weight"]
    )
    if progress is not None:
        progress(
            f"solving whole-strip factor graphs for {len(target_rows)} residual corridors"
        )
    reconfiguration, reconfiguration_stats = solve_patch_corridor_reconfigurations(
        remapped,
        remapped,
        ribbon,
        frontier,
        configuration,
        continuity_weight=continuity_weight,
        settings=corridor_settings,
    )
    reconfigured_at = time.monotonic()
    if progress is not None:
        progress("enumerating complete both-arc strip matchings")
    full_variants, enumeration_stats = enumerate_corridor_reconfiguration_variants(
        remapped,
        remapped,
        reconfiguration,
        ribbon,
        frontier,
        configuration,
        continuity_weight=continuity_weight,
        corridor_settings=corridor_settings,
        settings=variant_settings,
    )
    target_index = np.flatnonzero(
        np.isin(full_variants["corridorVariantRow"], target_rows)
    ).astype(np.int32)
    target_variants, target_to_full = _delta_variant_arrays(
        full_variants,
        target_index,
        corridor_count=len(full_variants["corridorVariantOffset"]) - 1,
    )
    enumerated_at = time.monotonic()
    if progress is not None:
        progress(
            f"screening {len(target_to_full)} complete strips against native CT"
        )
    screen, _, screen_stats = _screen_complete_strip_variants(
        target_rows,
        target_variants,
        remapped,
        ribbon,
        frontier,
        configuration,
        baseline_surface,
        surface_settings=corridor_settings.surface_settings(),
        face_settings=face_settings,
        settings=resolved,
        progress=progress,
    )
    screened_at = time.monotonic()
    arrays = {
        **target_variants,
        **screen,
        "targetCorridorRow": target_rows,
        "targetVariantFullEnumerationIndex": target_to_full,
    }
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_COMPLETE_STRIPS_SCHEMA,
        "version": PHYSICAL_RIBBON_COMPLETE_STRIPS_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "target": {
            "residualCorridorCount": len(target_rows),
            "residualCorridorRows": [int(value) for value in target_rows],
            "completeStripVariantCount": len(target_to_full),
        },
        "reconfiguration": reconfiguration_stats,
        "enumeration": enumeration_stats,
        "screen": screen_stats,
        "timingSeconds": {
            "factorGraphs": round(reconfigured_at - started, 6),
            "enumeration": round(enumerated_at - reconfigured_at, 6),
            "physicalScreen": round(screened_at - enumerated_at, 6),
            "writing": round(finished - screened_at, 6),
            "total": round(finished - started, 6),
        },
        "method": {
            "decisionUnit": (
                "complete both-arc native-CT strip matching, independent of "
                "whether its ribbons entered through a one-sided frontier"
            ),
            "topology": (
                "interface/crossing conflicts and inherited-component splits "
                "are measured before any surface completion"
            ),
            "surface": (
                "strict connectivity or an attached, native-CT-gated face path"
            ),
            "selectionMutated": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
