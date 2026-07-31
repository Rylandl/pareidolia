from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clear_ribbon_feedback import (
    CLEAR_RIBBON_FEEDBACK_SCHEMA,
    audit_baseline_preservation,
)
from .clear_ribbon_paired_feedback import (
    BOUNDARY_MATCH_UNOWNED,
    CLEAR_RIBBON_PAIRED_FEEDBACK_SCHEMA,
    CLEAR_RIBBON_PAIRED_FEEDBACK_STEM,
    SELECTION_CLASS_NEW_CLEAR_CORE,
    SELECTION_CLASS_PAIRED_GROWTH,
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


CLEAR_CORE_INTERFACE_REFINEMENT_SCHEMA = (
    "pareidolia.clear-core-paired-interface-refinement"
)
CLEAR_CORE_INTERFACE_REFINEMENT_VERSION = 1
CLEAR_CORE_INTERFACE_REFINEMENT_STEM = "clear-core-interface-refinement-v1"

SEED_SOURCE_NONE = 0
SEED_SOURCE_FROZEN_INTERFACE = 1
SEED_SOURCE_PAIRED_ENDPOINT = 2


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
]:
    value = Path(root).resolve()
    paired_path = (
        value
        if value.is_file()
        else value / f"{CLEAR_RIBBON_PAIRED_FEEDBACK_STEM}.json"
    )
    paired_manifest = json.loads(paired_path.read_text())
    if (
        paired_manifest.get("schema")
        != CLEAR_RIBBON_PAIRED_FEEDBACK_SCHEMA
        or paired_manifest.get("state") != "complete"
        or int(paired_manifest.get("version", 0)) < 2
    ):
        raise ValueError(
            "interface refinement requires complete v2 paired feedback"
        )
    if (
        paired_manifest["boundaryOwnership"][
            "selectedForeignLabelMatchedEndpointCount"
        ]
        != 0
    ):
        raise ValueError("paired feedback retained a foreign-owned boundary")
    paired_data_path = paired_path.parent / str(
        paired_manifest["data"]["path"]
    )
    paired = _load_npz(
        paired_data_path, paired_manifest["data"]["sha256"]
    )

    baseline_path = Path(
        paired_manifest["identity"]["interfaceFeedback"]["manifestPath"]
    )
    if (
        sha256_file(baseline_path)
        != paired_manifest["identity"]["interfaceFeedback"]["manifestSha256"]
    ):
        raise ValueError("interface feedback changed after paired feedback")
    baseline_manifest = json.loads(baseline_path.read_text())
    if (
        baseline_manifest.get("schema") != CLEAR_RIBBON_FEEDBACK_SCHEMA
        or baseline_manifest.get("state") != "complete"
    ):
        raise ValueError("paired feedback references invalid interface feedback")
    baseline_data_path = baseline_path.parent / str(
        baseline_manifest["data"]["path"]
    )
    baseline = _load_npz(
        baseline_data_path, baseline_manifest["data"]["sha256"]
    )

    interface_path = Path(
        paired_manifest["identity"]["interfaceBank"]["manifestPath"]
    )
    if (
        sha256_file(interface_path)
        != paired_manifest["identity"]["interfaceBank"]["manifestSha256"]
    ):
        raise ValueError("interface bank changed after paired feedback")
    interface_manifest = json.loads(interface_path.read_text())
    if (
        interface_manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA
        or interface_manifest.get("state") != "complete"
    ):
        raise ValueError("paired feedback references invalid interface bank")
    interface_data_path = interface_path.parent / str(
        interface_manifest["data"]["path"]
    )
    interfaces = _load_npz(
        interface_data_path, interface_manifest["data"]["sha256"]
    )

    growth_path = Path(
        baseline_manifest["identity"]["baselineOneSidedGrowth"][
            "manifestPath"
        ]
    )
    if (
        sha256_file(growth_path)
        != baseline_manifest["identity"]["baselineOneSidedGrowth"][
            "manifestSha256"
        ]
    ):
        raise ValueError("one-sided growth changed after interface feedback")
    growth_manifest = json.loads(growth_path.read_text())
    if growth_manifest.get("schema") != ONE_SIDED_GROWTH_SCHEMA:
        raise ValueError("interface feedback references wrong growth schema")
    return (
        paired_path,
        paired_manifest,
        paired,
        baseline_path,
        baseline_manifest,
        baseline,
        interface_path,
        interface_manifest,
        interfaces,
        growth_path,
        growth_manifest,
    )


def build_paired_endpoint_interface_seeds(
    paired: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Freeze current ownership and add only safely unowned paired endpoints."""

    baseline_selected = np.asarray(baseline["selected"]) > 0
    baseline_label = np.asarray(baseline["selectedLabel"], dtype=np.int32)
    interface_conflict = np.asarray(interfaces["seedConflict"]) > 0
    if np.any(baseline_selected & interface_conflict):
        raise ValueError("current interface selection contains a seed conflict")
    effective_seed = np.where(
        baseline_selected, baseline_label, -1
    ).astype(np.int32)
    seed_source = np.where(
        baseline_selected,
        SEED_SOURCE_FROZEN_INTERFACE,
        SEED_SOURCE_NONE,
    ).astype(np.uint8)
    endpoint_seed_label = np.full(len(effective_seed), -1, dtype=np.int32)
    endpoint_seed_side = np.full(len(effective_seed), 255, dtype=np.uint8)
    endpoint_seed_conflict = np.zeros(len(effective_seed), dtype=np.uint8)

    selection_class = np.asarray(paired["selectionClass"])
    new_selected = (
        (selection_class == SELECTION_CLASS_NEW_CLEAR_CORE)
        | (selection_class == SELECTION_CLASS_PAIRED_GROWTH)
    ) & (np.asarray(paired["selected"]) > 0)
    paired_label = np.asarray(paired["selectedLabel"], dtype=np.int32)
    endpoint_interface_parts = []
    endpoint_label_parts = []
    endpoint_side_parts = []
    for interface_name, class_name, side in (
        ("lowerMatchedInterface", "lowerBoundaryOwnershipClass", 0),
        ("upperMatchedInterface", "upperBoundaryOwnershipClass", 1),
    ):
        selected = new_selected & (
            np.asarray(paired[class_name]) == BOUNDARY_MATCH_UNOWNED
        )
        endpoint_interface_parts.append(
            np.asarray(paired[interface_name], dtype=np.int32)[selected]
        )
        endpoint_label_parts.append(paired_label[selected])
        endpoint_side_parts.append(
            np.full(np.count_nonzero(selected), side, dtype=np.uint8)
        )
    endpoint_interface = np.concatenate(endpoint_interface_parts)
    endpoint_label = np.concatenate(endpoint_label_parts)
    endpoint_side = np.concatenate(endpoint_side_parts)
    if np.any(endpoint_interface < 0):
        raise ValueError("unowned endpoint class has no matched interface")
    order = np.lexsort((endpoint_label, endpoint_interface))
    ordered_interface = endpoint_interface[order]
    accepted = 0
    conflicting = 0
    ambiguous_side = 0
    start = 0
    while start < len(order):
        stop = start + 1
        interface = int(ordered_interface[start])
        while (
            stop < len(order)
            and int(ordered_interface[stop]) == interface
        ):
            stop += 1
        observation = order[start:stop]
        labels = np.unique(endpoint_label[observation])
        conflict = (
            len(labels) != 1
            or effective_seed[interface] >= 0
            or interface_conflict[interface]
        )
        if conflict:
            endpoint_seed_conflict[interface] = 1
            conflicting += 1
        else:
            label = int(labels[0])
            sides = np.unique(endpoint_side[observation])
            side = int(sides[0]) if len(sides) == 1 else 255
            ambiguous_side += int(side == 255)
            effective_seed[interface] = label
            seed_source[interface] = SEED_SOURCE_PAIRED_ENDPOINT
            endpoint_seed_label[interface] = label
            endpoint_seed_side[interface] = side
            accepted += 1
        start = stop
    return {
        "effectiveSeedLabel": effective_seed,
        "refinementSeedSource": seed_source,
        "pairedEndpointSeedLabel": endpoint_seed_label,
        "pairedEndpointSeedBoundarySide": endpoint_seed_side,
        "pairedEndpointSeedConflict": endpoint_seed_conflict,
    }, {
        "frozenInterfaceSeedCount": int(np.count_nonzero(baseline_selected)),
        "pairedEndpointObservationCount": int(len(endpoint_interface)),
        "uniquePairedEndpointInterfaceCount": int(
            len(np.unique(endpoint_interface))
        ),
        "acceptedPairedEndpointSeedCount": accepted,
        "conflictingPairedEndpointSeedCount": conflicting,
        "ambiguousBoundarySideSeedCount": ambiguous_side,
        "newSeedLabelCount": int(
            len(np.unique(endpoint_label)) if len(endpoint_label) else 0
        ),
    }


def _new_label_records(
    paired: Mapping[str, np.ndarray],
    baseline: Mapping[str, np.ndarray],
    seeds: Mapping[str, np.ndarray],
    refined: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    selection_class = np.asarray(paired["selectionClass"])
    labels = np.unique(
        np.asarray(paired["selectedLabel"])[
            (selection_class == SELECTION_CLASS_NEW_CLEAR_CORE)
            | (selection_class == SELECTION_CLASS_PAIRED_GROWTH)
        ]
    )
    baseline_selected = np.asarray(baseline["selected"]) > 0
    selected_label = np.asarray(refined["selectedLabel"], dtype=np.int32)
    source = np.asarray(seeds["refinementSeedSource"])
    records = []
    for label in labels:
        selected = selected_label == label
        added = selected & ~baseline_selected
        records.append(
            {
                "label": int(label),
                "pairedEndpointSeedCount": int(
                    np.count_nonzero(
                        (source == SEED_SOURCE_PAIRED_ENDPOINT)
                        & (selected_label == label)
                    )
                ),
                "selectedInterfaceCount": int(np.count_nonzero(selected)),
                "newlySelectedInterfaceCount": int(np.count_nonzero(added)),
                "newlySelectedPathBottleneck": _percentile_record(
                    np.asarray(refined["pathBottleneck"])[added]
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


def run_clear_core_interface_refinement(
    paired_feedback_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    (
        paired_path,
        paired_manifest,
        paired,
        baseline_path,
        baseline_manifest,
        baseline,
        interface_path,
        interface_manifest,
        interfaces,
        growth_path,
        growth_manifest,
    ) = _load_inputs(paired_feedback_root)
    settings = OneSidedGrowthSettings(
        **growth_manifest["identity"]["settings"]
    )
    identity: dict[str, Any] = {
        "schema": CLEAR_CORE_INTERFACE_REFINEMENT_SCHEMA,
        "version": CLEAR_CORE_INTERFACE_REFINEMENT_VERSION,
        "pairedFeedback": {
            "manifestPath": str(paired_path),
            "manifestSha256": sha256_file(paired_path),
            "dataSha256": paired_manifest["data"]["sha256"],
        },
        "baselineInterfaceFeedback": {
            "manifestPath": str(baseline_path),
            "manifestSha256": sha256_file(baseline_path),
            "dataSha256": baseline_manifest["data"]["sha256"],
        },
        "interfaceBank": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "oneSidedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "growthSettings": settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{CLEAR_CORE_INTERFACE_REFINEMENT_STEM}.json"
    data_path = output / f"{CLEAR_CORE_INTERFACE_REFINEMENT_STEM}.npz"
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
    seeds, seed_stats = build_paired_endpoint_interface_seeds(
        paired, baseline, interfaces
    )
    seeded = time.monotonic()
    augmented = {
        **interfaces,
        "seedSurfaceLabel": seeds["effectiveSeedLabel"],
        "seedConflict": np.maximum(
            np.asarray(interfaces["seedConflict"], dtype=np.uint8),
            seeds["pairedEndpointSeedConflict"],
        ),
    }
    processing_shape = tuple(
        int(value)
        for value in interface_manifest["geometry"][
            "processingShapeSamplingXYZ"
        ]
    )
    ribbon_path = Path(
        paired_manifest["identity"]["ribbonBank"]["manifestPath"]
    )
    if (
        sha256_file(ribbon_path)
        != paired_manifest["identity"]["ribbonBank"]["manifestSha256"]
    ):
        raise ValueError("ribbon bank changed after paired feedback")
    ribbon_manifest = json.loads(ribbon_path.read_text())
    stride = int(ribbon_manifest["samplingStrideVoxels"])
    graph, graph_stats = build_one_sided_continuity_graph(
        augmented,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=settings,
    )
    membership = _retained_component_membership(
        augmented, graph, settings=settings
    )
    graphed = time.monotonic()
    refined, growth_stats = grow_one_sided_interfaces(
        augmented,
        graph,
        settings=settings,
        membership=membership,
        seed_label_override=seeds["effectiveSeedLabel"],
    )
    preservation = audit_baseline_preservation(baseline, refined)
    records = _new_label_records(paired, baseline, seeds, refined)
    grown = time.monotonic()
    arrays = {**refined, **graph, **seeds}
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
    world_stop = np.asarray(
        world_record["stopXYZExclusive"], dtype=np.float64
    )
    new_labels = np.asarray(
        [value["label"] for value in records], dtype=np.int32
    )
    new_selected = np.isin(refined["selectedLabel"], new_labels)
    preview = {
        "selectedLabel": np.where(
            new_selected, refined["selectedLabel"], -1
        ).astype(np.int32),
        "lockedSeed": (
            seeds["refinementSeedSource"] == SEED_SOURCE_PAIRED_ENDPOINT
        ).astype(np.uint8),
        "selected": new_selected.astype(np.uint8),
    }
    projection = write_one_sided_growth_projection(
        interfaces,
        preview,
        world_start,
        world_stop,
        settings,
        output / "paired-refined-interface-growth.png",
    )
    cross_sections = write_one_sided_growth_cross_sections(
        source,
        owned,
        interfaces,
        preview,
        output / "paired-refined-interface-cross-sections.png",
        display_high_raw=float(
            interface_manifest["calibration"]["displayHighRaw"]
        ),
        sampling_stride=stride,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": CLEAR_CORE_INTERFACE_REFINEMENT_SCHEMA,
        "version": CLEAR_CORE_INTERFACE_REFINEMENT_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interface_manifest["source"],
        "geometry": interface_manifest["geometry"],
        "calibration": interface_manifest["calibration"],
        "seeding": seed_stats,
        "graph": graph_stats,
        "growth": growth_stats,
        "baselinePreservation": preservation,
        "newClearCoreLabels": records,
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
        "seedSource": {
            "0": "not a refinement seed",
            "1": "immutable prior interface assignment",
            "2": "safe unowned endpoint of selected paired profile",
        },
        "method": {
            "baselinePolicy": "freeze every prior interface assignment",
            "newSeedPolicy": (
                "only conflict-free paired endpoints matched to unowned "
                "signed interfaces"
            ),
            "growthPolicy": "defer every multi-label interface component",
            "changesExistingAssignments": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
