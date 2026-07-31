from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clear_ribbon import CLEAR_RIBBON_SCHEMA, CLEAR_RIBBON_STEM
from .clear_ribbon_selection import (
    CLEAR_RIBBON_SELECTION_SCHEMA,
    CLEAR_RIBBON_SELECTION_STEM,
    SELECTION_CLASS_NEW_CLEAR_CORE,
)
from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import _percentile_record
from .one_sided_growth import (
    ONE_SIDED_GROWTH_SCHEMA,
    OneSidedGrowthSettings,
    _retained_component_membership,
    build_one_sided_continuity_graph,
    grow_one_sided_interfaces,
    write_one_sided_growth_cross_sections,
    write_one_sided_growth_projection,
)
from .one_sided_interface import ONE_SIDED_INTERFACE_SCHEMA


CLEAR_RIBBON_FEEDBACK_SCHEMA = "pareidolia.clear-ribbon-interface-feedback"
CLEAR_RIBBON_FEEDBACK_VERSION = 1
CLEAR_RIBBON_FEEDBACK_STEM = "clear-ribbon-interface-feedback-v1"

SEED_SOURCE_NONE = 0
SEED_SOURCE_BASELINE = 1
SEED_SOURCE_NEW_CLEAR_CORE = 2


def _load_npz(path: Path, expected_sha256: str) -> dict[str, np.ndarray]:
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"artifact data hash differs from manifest: {path}")
    with np.load(path) as values:
        return {name: np.asarray(values[name]) for name in values.files}


def _load_inputs(
    root: str | Path,
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
    Path,
    dict[str, Any],
    dict[str, np.ndarray],
]:
    value = Path(root).resolve()
    selection_path = (
        value
        if value.is_file()
        else value / f"{CLEAR_RIBBON_SELECTION_STEM}.json"
    )
    selection_manifest = json.loads(selection_path.read_text())
    if (
        selection_manifest.get("schema") != CLEAR_RIBBON_SELECTION_SCHEMA
        or selection_manifest.get("state") != "complete"
        or int(selection_manifest.get("version", 0)) < 2
    ):
        raise ValueError(
            "clear-ribbon feedback requires a complete v2 or newer selection"
        )
    selection_data_path = selection_path.parent / str(
        selection_manifest["data"]["path"]
    )
    selection = _load_npz(
        selection_data_path, selection_manifest["data"]["sha256"]
    )

    bank_path = Path(
        selection_manifest["identity"]["ribbonBank"]["manifestPath"]
    )
    if (
        sha256_file(bank_path)
        != selection_manifest["identity"]["ribbonBank"]["manifestSha256"]
    ):
        raise ValueError("clear-ribbon bank changed after ribbon selection")
    bank_manifest = json.loads(bank_path.read_text())
    if bank_manifest.get("schema") != CLEAR_RIBBON_SCHEMA:
        raise ValueError("clear-ribbon selection references the wrong bank schema")
    bank_data_path = bank_path.parent / str(bank_manifest["data"]["path"])
    bank = _load_npz(bank_data_path, bank_manifest["data"]["sha256"])

    growth_path = Path(
        bank_manifest["identity"]["oneSidedGrowth"]["manifestPath"]
    )
    if (
        sha256_file(growth_path)
        != bank_manifest["identity"]["oneSidedGrowth"]["manifestSha256"]
    ):
        raise ValueError("one-sided growth changed after clear-ribbon banking")
    growth_manifest = json.loads(growth_path.read_text())
    if growth_manifest.get("schema") != ONE_SIDED_GROWTH_SCHEMA:
        raise ValueError("clear-ribbon bank references the wrong growth schema")
    growth_data_path = growth_path.parent / str(growth_manifest["data"]["path"])
    growth = _load_npz(growth_data_path, growth_manifest["data"]["sha256"])

    interface_path = Path(
        growth_manifest["identity"]["interfaceBank"]["manifestPath"]
    )
    if (
        sha256_file(interface_path)
        != growth_manifest["identity"]["interfaceBank"]["manifestSha256"]
    ):
        raise ValueError("interface bank changed after one-sided growth")
    interface_manifest = json.loads(interface_path.read_text())
    if interface_manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA:
        raise ValueError("one-sided growth references the wrong interface schema")
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    interface = _load_npz(
        interface_data_path, interface_manifest["data"]["sha256"]
    )
    return (
        selection_path,
        selection_manifest,
        selection,
        bank_path,
        bank_manifest,
        bank,
        growth_path,
        growth_manifest,
        growth,
        interface_path,
        interface_manifest,
        interface,
    )


def build_clear_ribbon_feedback_seeds(
    baseline_growth: Mapping[str, np.ndarray],
    ribbons: Mapping[str, np.ndarray],
    ribbon_selection: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Add only genuinely new two-face cores to existing assembly anchors."""

    selection_class = np.asarray(ribbon_selection["selectionClass"])
    selected = np.asarray(ribbon_selection["selected"]) > 0
    new_ribbon = selected & (
        selection_class == SELECTION_CLASS_NEW_CLEAR_CORE
    )
    if np.any(
        (np.asarray(ribbons["lowerComponentSeedLabelCount"])[new_ribbon] != 0)
        | (np.asarray(ribbons["upperComponentSeedLabelCount"])[new_ribbon] != 0)
    ):
        raise ValueError(
            "new clear core touches an already seeded signed component"
        )

    effective_seed = np.asarray(
        baseline_growth["seedAssemblyLabel"], dtype=np.int32
    ).copy()
    seed_source = np.where(
        effective_seed >= 0, SEED_SOURCE_BASELINE, SEED_SOURCE_NONE
    ).astype(np.uint8)
    ribbon_seed_label = np.full(len(effective_seed), -1, dtype=np.int32)
    ribbon_seed_side = np.full(len(effective_seed), 255, dtype=np.uint8)
    ribbon_seed_conflict = np.zeros(len(effective_seed), dtype=np.uint8)

    new_label = np.asarray(
        ribbon_selection["selectedAssemblyLabel"], dtype=np.int32
    )[new_ribbon]
    lower = np.asarray(ribbons["lowerInterface"], dtype=np.int32)[new_ribbon]
    upper = np.asarray(ribbons["upperInterface"], dtype=np.int32)[new_ribbon]
    endpoint_interface = np.concatenate((lower, upper))
    endpoint_label = np.concatenate((new_label, new_label))
    endpoint_side = np.concatenate(
        (
            np.zeros(len(lower), dtype=np.uint8),
            np.ones(len(upper), dtype=np.uint8),
        )
    )
    order = np.lexsort((endpoint_label, endpoint_interface))
    ordered_interface = endpoint_interface[order]
    start = 0
    accepted = 0
    conflicting = 0
    while start < len(order):
        stop = start + 1
        interface = int(ordered_interface[start])
        while (
            stop < len(order)
            and int(ordered_interface[stop]) == interface
        ):
            stop += 1
        endpoint = order[start:stop]
        labels = np.unique(endpoint_label[endpoint])
        conflict = len(labels) != 1 or (
            effective_seed[interface] >= 0
            and int(effective_seed[interface]) != int(labels[0])
        )
        if conflict:
            ribbon_seed_conflict[interface] = 1
            conflicting += 1
        else:
            label = int(labels[0])
            effective_seed[interface] = label
            seed_source[interface] = SEED_SOURCE_NEW_CLEAR_CORE
            ribbon_seed_label[interface] = label
            sides = np.unique(endpoint_side[endpoint])
            if len(sides) == 1:
                ribbon_seed_side[interface] = int(sides[0])
            accepted += 1
        start = stop

    accepted_mask = seed_source == SEED_SOURCE_NEW_CLEAR_CORE
    return {
        "effectiveSeedLabel": effective_seed,
        "seedSource": seed_source,
        "ribbonSeedLabel": ribbon_seed_label,
        "ribbonSeedBoundarySide": ribbon_seed_side,
        "ribbonSeedConflict": ribbon_seed_conflict,
    }, {
        "newClearCoreRibbonCount": int(np.count_nonzero(new_ribbon)),
        "newClearCoreLabelCount": int(len(np.unique(new_label))),
        "ribbonEndpointCount": int(len(endpoint_interface)),
        "uniqueRibbonEndpointInterfaceCount": int(
            len(np.unique(endpoint_interface))
        ),
        "acceptedRibbonSeedInterfaceCount": accepted,
        "conflictingRibbonSeedInterfaceCount": conflicting,
        "acceptedRibbonSeedWithKnownSideCount": int(
            np.count_nonzero(accepted_mask & (ribbon_seed_side < 2))
        ),
        "acceptedRibbonSeedWithAmbiguousSideCount": int(
            np.count_nonzero(accepted_mask & (ribbon_seed_side == 255))
        ),
    }


def audit_baseline_preservation(
    baseline_growth: Mapping[str, np.ndarray],
    refined: Mapping[str, np.ndarray],
) -> dict[str, int]:
    baseline_selected = np.asarray(baseline_growth["selected"]) > 0
    baseline_label = np.asarray(
        baseline_growth["selectedLabel"], dtype=np.int32
    )
    refined_selected = np.asarray(refined["selected"]) > 0
    refined_label = np.asarray(refined["selectedLabel"], dtype=np.int32)
    lost = baseline_selected & ~refined_selected
    changed = (
        baseline_selected
        & refined_selected
        & (baseline_label != refined_label)
    )
    if np.any(lost) or np.any(changed):
        raise RuntimeError(
            "clear-ribbon feedback changed a baseline interface assignment"
        )
    return {
        "baselineSelectedInterfaceCount": int(np.count_nonzero(baseline_selected)),
        "preservedBaselineInterfaceCount": int(
            np.count_nonzero(baseline_selected & refined_selected)
        ),
        "lostBaselineInterfaceCount": int(np.count_nonzero(lost)),
        "changedBaselineLabelCount": int(np.count_nonzero(changed)),
        "newlySelectedInterfaceCount": int(
            np.count_nonzero(refined_selected & ~baseline_selected)
        ),
    }


def _new_label_records(
    ribbons: Mapping[str, np.ndarray],
    ribbon_selection: Mapping[str, np.ndarray],
    feedback_seed: Mapping[str, np.ndarray],
    refined: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    new_ribbon = (
        np.asarray(ribbon_selection["selectionClass"])
        == SELECTION_CLASS_NEW_CLEAR_CORE
    ) & (np.asarray(ribbon_selection["selected"]) > 0)
    ribbon_label = np.asarray(
        ribbon_selection["selectedAssemblyLabel"], dtype=np.int32
    )
    selected_label = np.asarray(refined["selectedLabel"], dtype=np.int32)
    interface_component = np.asarray(
        refined["interfaceContinuityComponent"], dtype=np.int32
    )
    seed_source = np.asarray(feedback_seed["seedSource"])
    records = []
    for label in np.unique(ribbon_label[new_ribbon]):
        selected = selected_label == label
        component = np.unique(interface_component[selected])
        records.append(
            {
                "label": int(label),
                "ribbonCount": int(
                    np.count_nonzero(new_ribbon & (ribbon_label == label))
                ),
                "endpointSeedInterfaceCount": int(
                    np.count_nonzero(
                        (seed_source == SEED_SOURCE_NEW_CLEAR_CORE)
                        & (selected_label == label)
                    )
                ),
                "selectedInterfaceCount": int(np.count_nonzero(selected)),
                "grownInterfaceCount": int(
                    np.count_nonzero(
                        selected
                        & (seed_source != SEED_SOURCE_NEW_CLEAR_CORE)
                    )
                ),
                "continuityComponentCount": int(
                    np.count_nonzero(component >= 0)
                ),
                "selectedPathBottleneck": _percentile_record(
                    np.asarray(refined["pathBottleneck"])[selected]
                ),
            }
        )
    records.sort(key=lambda value: (-value["selectedInterfaceCount"], value["label"]))
    return records


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_clear_ribbon_feedback(
    selection_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    (
        selection_path,
        selection_manifest,
        ribbon_selection,
        bank_path,
        bank_manifest,
        ribbons,
        growth_path,
        growth_manifest,
        baseline_growth,
        interface_path,
        interface_manifest,
        interfaces,
    ) = _load_inputs(selection_root)
    settings = OneSidedGrowthSettings(
        **growth_manifest["identity"]["settings"]
    )
    identity: dict[str, Any] = {
        "schema": CLEAR_RIBBON_FEEDBACK_SCHEMA,
        "version": CLEAR_RIBBON_FEEDBACK_VERSION,
        "ribbonSelection": {
            "manifestPath": str(selection_path),
            "manifestSha256": sha256_file(selection_path),
            "dataSha256": selection_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "baselineOneSidedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "interfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "growthSettings": settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{CLEAR_RIBBON_FEEDBACK_STEM}.json"
    data_path = output / f"{CLEAR_RIBBON_FEEDBACK_STEM}.npz"
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
    feedback_seed, seed_stats = build_clear_ribbon_feedback_seeds(
        baseline_growth, ribbons, ribbon_selection
    )
    seeded = time.monotonic()
    augmented_interfaces = {
        **interfaces,
        "seedSurfaceLabel": feedback_seed["effectiveSeedLabel"],
    }
    processing_shape = tuple(
        int(value)
        for value in growth_manifest["geometry"]["processingShapeSamplingXYZ"]
    )
    stride = int(bank_manifest["samplingStrideVoxels"])
    graph, graph_stats = build_one_sided_continuity_graph(
        augmented_interfaces,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=settings,
    )
    membership = _retained_component_membership(
        augmented_interfaces, graph, settings=settings
    )
    graphed = time.monotonic()
    refined, growth_stats = grow_one_sided_interfaces(
        augmented_interfaces,
        graph,
        settings=settings,
        membership=membership,
        seed_label_override=feedback_seed["effectiveSeedLabel"],
    )
    preservation = audit_baseline_preservation(baseline_growth, refined)
    new_labels = _new_label_records(
        ribbons, ribbon_selection, feedback_seed, refined
    )
    grown = time.monotonic()
    arrays = {**refined, **graph, **feedback_seed}
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        interface_manifest["source"]["path"],
        interface_manifest["source"]["metadataPath"],
    )
    owned_record = interface_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    world_record = interface_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    new_label_values = np.asarray(
        [value["label"] for value in new_labels], dtype=np.int32
    )
    new_selected = np.isin(refined["selectedLabel"], new_label_values)
    preview = {
        "selectedLabel": np.where(
            new_selected, refined["selectedLabel"], -1
        ).astype(np.int32),
        "lockedSeed": (
            new_selected
            & (feedback_seed["seedSource"] == SEED_SOURCE_NEW_CLEAR_CORE)
        ).astype(np.uint8),
        "selected": new_selected.astype(np.uint8),
    }
    projection = write_one_sided_growth_projection(
        interfaces,
        preview,
        world_start,
        world_stop,
        settings,
        output / "new-clear-core-interface-growth.png",
    )
    cross_sections = write_one_sided_growth_cross_sections(
        source,
        owned,
        interfaces,
        preview,
        output / "new-clear-core-interface-cross-sections.png",
        display_high_raw=float(interface_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=stride,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": CLEAR_RIBBON_FEEDBACK_SCHEMA,
        "version": CLEAR_RIBBON_FEEDBACK_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interface_manifest["source"],
        "geometry": interface_manifest["geometry"],
        "calibration": interface_manifest["calibration"],
        "seeding": seed_stats,
        "graph": graph_stats,
        "growth": growth_stats,
        "baselinePreservation": preservation,
        "newClearCoreLabels": new_labels,
        "timingSeconds": {
            "feedbackSeeding": round(seeded - started, 6),
            "graphAndComponents": round(graphed - seeded, 6),
            "growthAndAudit": round(grown - graphed, 6),
            "writingAndArtifacts": round(finished - grown, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "newCoreGrowth": projection.name,
            "newCoreCrossSections": cross_sections.name,
        },
        "method": {
            "seedSource": "two exact faces of v2 genuinely unseeded ribbon cores",
            "graphPolicy": "rebuild signed face-adjacent topology with new seeds",
            "growthPolicy": "preserve baseline; defer every multi-label component",
            "changesExistingAssignments": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
