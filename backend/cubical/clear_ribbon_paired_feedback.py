from __future__ import annotations

import heapq
import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clear_ribbon import CLEAR_RIBBON_SCHEMA
from .clear_ribbon_feedback import (
    CLEAR_RIBBON_FEEDBACK_SCHEMA,
    CLEAR_RIBBON_FEEDBACK_STEM,
)
from .clear_ribbon_selection import (
    CLEAR_RIBBON_SELECTION_SCHEMA,
    SELECTION_CLASS_NEW_CLEAR_CORE as RIBBON_CLASS_NEW_CLEAR_CORE,
)
from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import _percentile_record
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PairedSurfaceGrowthSettings,
    _adjacency,
    write_growth_cross_sections,
    write_growth_projection,
)


CLEAR_RIBBON_PAIRED_FEEDBACK_SCHEMA = (
    "pareidolia.clear-ribbon-paired-profile-feedback"
)
CLEAR_RIBBON_PAIRED_FEEDBACK_VERSION = 1
CLEAR_RIBBON_PAIRED_FEEDBACK_STEM = "clear-ribbon-paired-feedback-v1"

SELECTION_CLASS_UNSELECTED = 0
SELECTION_CLASS_BASELINE = 1
SELECTION_CLASS_NEW_CLEAR_CORE = 2
SELECTION_CLASS_PAIRED_GROWTH = 3


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
    feedback_path = (
        value
        if value.is_file()
        else value / f"{CLEAR_RIBBON_FEEDBACK_STEM}.json"
    )
    feedback_manifest = json.loads(feedback_path.read_text())
    if (
        feedback_manifest.get("schema") != CLEAR_RIBBON_FEEDBACK_SCHEMA
        or feedback_manifest.get("state") != "complete"
    ):
        raise ValueError("paired feedback requires complete interface feedback")
    preservation = feedback_manifest["baselinePreservation"]
    if (
        preservation["lostBaselineInterfaceCount"] != 0
        or preservation["changedBaselineLabelCount"] != 0
        or feedback_manifest["seeding"]["conflictingRibbonSeedInterfaceCount"]
        != 0
    ):
        raise ValueError("interface feedback did not preserve its baseline")

    selection_path = Path(
        feedback_manifest["identity"]["ribbonSelection"]["manifestPath"]
    )
    if (
        sha256_file(selection_path)
        != feedback_manifest["identity"]["ribbonSelection"]["manifestSha256"]
    ):
        raise ValueError("ribbon selection changed after interface feedback")
    selection_manifest = json.loads(selection_path.read_text())
    if selection_manifest.get("schema") != CLEAR_RIBBON_SELECTION_SCHEMA:
        raise ValueError("interface feedback references the wrong selection schema")
    selection_data_path = selection_path.parent / str(
        selection_manifest["data"]["path"]
    )
    selection = _load_npz(
        selection_data_path, selection_manifest["data"]["sha256"]
    )

    ribbon_path = Path(
        feedback_manifest["identity"]["ribbonBank"]["manifestPath"]
    )
    if (
        sha256_file(ribbon_path)
        != feedback_manifest["identity"]["ribbonBank"]["manifestSha256"]
    ):
        raise ValueError("ribbon bank changed after interface feedback")
    ribbon_manifest = json.loads(ribbon_path.read_text())
    if ribbon_manifest.get("schema") != CLEAR_RIBBON_SCHEMA:
        raise ValueError("interface feedback references the wrong ribbon schema")
    ribbon_data_path = ribbon_path.parent / str(ribbon_manifest["data"]["path"])
    ribbons = _load_npz(ribbon_data_path, ribbon_manifest["data"]["sha256"])

    growth_path = Path(
        ribbon_manifest["identity"]["pairedGrowth"]["manifestPath"]
    )
    if (
        sha256_file(growth_path)
        != ribbon_manifest["identity"]["pairedGrowth"]["manifestSha256"]
    ):
        raise ValueError("paired growth changed after ribbon banking")
    growth_manifest = json.loads(growth_path.read_text())
    if growth_manifest.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA:
        raise ValueError("ribbon bank references the wrong paired-growth schema")
    growth_data_path = growth_path.parent / str(growth_manifest["data"]["path"])
    growth = _load_npz(growth_data_path, growth_manifest["data"]["sha256"])

    bank_path = Path(ribbon_manifest["identity"]["pairedBank"]["manifestPath"])
    if (
        sha256_file(bank_path)
        != ribbon_manifest["identity"]["pairedBank"]["manifestSha256"]
    ):
        raise ValueError("paired bank changed after ribbon banking")
    bank_manifest = json.loads(bank_path.read_text())
    if bank_manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA:
        raise ValueError("ribbon bank references the wrong paired-bank schema")
    bank_data_path = bank_path.parent / str(bank_manifest["data"]["path"])
    bank = _load_npz(bank_data_path, bank_manifest["data"]["sha256"])
    return (
        feedback_path,
        feedback_manifest,
        selection_path,
        selection_manifest,
        selection,
        ribbon_path,
        ribbon_manifest,
        ribbons,
        growth_path,
        growth_manifest,
        growth,
        bank_path,
        bank_manifest,
        bank,
    )


def build_paired_feedback_seeds(
    bank: Mapping[str, np.ndarray],
    baseline_growth: Mapping[str, np.ndarray],
    ribbons: Mapping[str, np.ndarray],
    ribbon_selection: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    new_ribbon = (
        np.asarray(ribbon_selection["selected"]) > 0
    ) & (
        np.asarray(ribbon_selection["selectionClass"])
        == RIBBON_CLASS_NEW_CLEAR_CORE
    )
    candidate = np.asarray(
        ribbons["pairedCandidateIndex"], dtype=np.int32
    )[new_ribbon]
    label = np.asarray(
        ribbon_selection["selectedAssemblyLabel"], dtype=np.int32
    )[new_ribbon]
    if len(np.unique(candidate)) != len(candidate):
        raise ValueError("new clear cores repeat a paired candidate")
    if np.any(np.asarray(baseline_growth["selected"])[candidate] > 0):
        raise ValueError("new clear core is already selected by baseline growth")
    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    if np.any(evidence[candidate] < settings.minimum_candidate_evidence):
        raise ValueError("new clear core falls below paired-graph eligibility")

    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    shape_zyx = processing_shape_sampling_xyz[::-1]
    flat = np.ravel_multi_index(key[:, ::-1].T, shape_zyx)
    baseline_selected = np.asarray(baseline_growth["selected"]) > 0
    occupied_flat = set(int(value) for value in flat[baseline_selected])
    if any(int(value) in occupied_flat for value in flat[candidate]):
        raise ValueError("new clear core collides with a baseline spatial key")
    seed_label = np.full(len(evidence), -1, dtype=np.int32)
    seed_label[candidate] = label
    return {
        "newSeedCandidate": candidate,
        "newSeedLabel": seed_label,
    }, {
        "newClearCoreLabelCount": int(len(np.unique(label))),
        "newClearCoreRibbonCount": int(len(candidate)),
        "newSeedCandidateCount": int(len(candidate)),
        "newSeedSpatialKeyCount": int(len(np.unique(flat[candidate]))),
        "newSeedEvidence": _percentile_record(evidence[candidate]),
    }


def label_free_paired_components(
    bank: Mapping[str, np.ndarray],
    baseline_growth: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    feedback_seed: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    node_count = len(bank["localEvidenceScore"])
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    flat = np.ravel_multi_index(
        key[:, ::-1].T, processing_shape_sampling_xyz[::-1]
    )
    baseline_selected = np.asarray(baseline_growth["selected"]) > 0
    occupied_key = np.zeros(np.prod(processing_shape_sampling_xyz), dtype=bool)
    occupied_key[flat[baseline_selected]] = True
    seed_label = np.asarray(feedback_seed["newSeedLabel"], dtype=np.int32)
    eligible = (
        (np.asarray(bank["localEvidenceScore"]) >= settings.minimum_candidate_evidence)
        & ~occupied_key[flat]
    ) | (seed_label >= 0)

    edge_first = np.asarray(graph["edgeFirstCandidate"], dtype=np.int32)
    edge_second = np.asarray(graph["edgeSecondCandidate"], dtype=np.int32)
    edge_affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    retained_edge = (
        eligible[edge_first]
        & eligible[edge_second]
        & (edge_affinity >= settings.minimum_growth_bottleneck)
    )
    parent = np.arange(node_count, dtype=np.int32)

    def find(value: int) -> int:
        current = value
        while int(parent[current]) != current:
            parent[current] = parent[int(parent[current])]
            current = int(parent[current])
        return current

    for first, second in zip(
        edge_first[retained_edge], edge_second[retained_edge]
    ):
        first_root = find(int(first))
        second_root = find(int(second))
        if first_root != second_root:
            if first_root < second_root:
                parent[second_root] = first_root
            else:
                parent[first_root] = second_root

    eligible_index = np.flatnonzero(eligible).astype(np.int32)
    root = np.asarray([find(int(value)) for value in eligible_index], dtype=np.int32)
    _unique_root, inverse, component_size = np.unique(
        root, return_inverse=True, return_counts=True
    )
    component = np.full(node_count, -1, dtype=np.int32)
    component[eligible_index] = inverse.astype(np.int32)
    component_count = len(component_size)
    labels_by_component: dict[int, set[int]] = {}
    component_seed_count = np.zeros(component_count, dtype=np.int32)
    for node in np.flatnonzero(seed_label >= 0):
        value = int(component[node])
        labels_by_component.setdefault(value, set()).add(int(seed_label[node]))
        component_seed_count[value] += 1
    component_label_count = np.zeros(component_count, dtype=np.uint16)
    component_sole_label = np.full(component_count, -1, dtype=np.int32)
    for value, labels in labels_by_component.items():
        component_label_count[value] = len(labels)
        if len(labels) == 1:
            component_sole_label[value] = next(iter(labels))

    key_label_sets: dict[int, set[int]] = {}
    single_label_candidate = np.zeros(node_count, dtype=bool)
    single_label_candidate[eligible_index] = (
        component_label_count[inverse] == 1
    )
    for node in np.flatnonzero(single_label_candidate):
        value = int(component[node])
        key_label_sets.setdefault(int(flat[node]), set()).add(
            int(component_sole_label[value])
        )
    cross_label_key = np.zeros(node_count, dtype=np.uint8)
    for node in np.flatnonzero(eligible):
        labels = key_label_sets.get(int(flat[node]), set())
        if len(labels) > 1:
            cross_label_key[node] = 1

    return {
        "eligibleFreeCandidate": eligible.astype(np.uint8),
        "freeComponent": component,
        "componentCandidateCount": component_size.astype(np.int32),
        "componentNewSeedCount": component_seed_count,
        "componentNewLabelCount": component_label_count,
        "componentSoleNewLabel": component_sole_label,
        "crossLabelSpatialKeyConflict": cross_label_key,
    }, {
        "eligibleFreeCandidateCount": int(np.count_nonzero(eligible)),
        "retainedFreeEdgeCount": int(np.count_nonzero(retained_edge)),
        "freeComponentCount": component_count,
        "unseededComponentCount": int(np.count_nonzero(component_label_count == 0)),
        "singleLabelComponentCount": int(
            np.count_nonzero(component_label_count == 1)
        ),
        "contestedComponentCount": int(np.count_nonzero(component_label_count > 1)),
        "contestedCandidateCount": int(
            np.sum(component_size[component_label_count > 1])
        ),
        "crossLabelSpatialKeyConflictCandidateCount": int(
            np.count_nonzero(cross_label_key)
        ),
        "componentSize": _percentile_record(component_size),
        "seededComponentSize": _percentile_record(
            component_size[component_label_count > 0]
        ),
    }


def grow_clear_cores_in_paired_graph(
    bank: Mapping[str, np.ndarray],
    baseline_growth: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    feedback_seed: Mapping[str, np.ndarray],
    membership: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: PairedSurfaceGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    node_count = len(bank["localEvidenceScore"])
    edge_first = np.asarray(graph["edgeFirstCandidate"], dtype=np.int32)
    edge_second = np.asarray(graph["edgeSecondCandidate"], dtype=np.int32)
    edge_affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    offset, neighbor, adjacency_edge, adjacency_affinity = _adjacency(
        node_count, edge_first, edge_second, edge_affinity
    )
    component = np.asarray(membership["freeComponent"], dtype=np.int32)
    component_label_count = np.asarray(membership["componentNewLabelCount"])
    component_sole_label = np.asarray(
        membership["componentSoleNewLabel"], dtype=np.int32
    )
    cross_label_key = np.asarray(
        membership["crossLabelSpatialKeyConflict"]
    ) > 0
    seed_label = np.asarray(feedback_seed["newSeedLabel"], dtype=np.int32)
    new_seed = seed_label >= 0
    baseline_selected = np.asarray(baseline_growth["selected"]) > 0
    label = np.asarray(
        baseline_growth["selectedLabel"], dtype=np.int32
    ).copy()
    path_score = np.asarray(
        baseline_growth["pathBottleneck"], dtype=np.float32
    ).copy()
    parent = np.asarray(
        baseline_growth["parentCandidate"], dtype=np.int32
    ).copy()
    parent_edge = np.asarray(
        baseline_growth["parentContinuityEdge"], dtype=np.int32
    ).copy()
    selection_class = np.zeros(node_count, dtype=np.uint8)
    selection_class[baseline_selected] = SELECTION_CLASS_BASELINE
    collision_rejected = np.zeros(node_count, dtype=np.uint8)
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    occupied = np.full(
        processing_shape_sampling_xyz[::-1], -1, dtype=np.int32
    )
    for node in np.flatnonzero(baseline_selected):
        x, y, z = key[node]
        if occupied[z, y, x] >= 0:
            raise RuntimeError("baseline paired selection violates key exclusivity")
        occupied[z, y, x] = node

    for node in np.flatnonzero(new_seed):
        x, y, z = key[node]
        if occupied[z, y, x] >= 0:
            raise RuntimeError("new clear-core seed collides with selected key")
        occupied[z, y, x] = node
        label[node] = seed_label[node]
        path_score[node] = 1.0
        parent[node] = -1
        parent_edge[node] = -1
        selection_class[node] = SELECTION_CLASS_NEW_CLEAR_CORE

    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    queue: list[tuple[float, int, int, int, int]] = []
    proposal_count = 0
    for node in np.flatnonzero(new_seed):
        value = int(component[node])
        if value < 0 or int(component_label_count[value]) != 1:
            continue
        for adjacency_index in range(int(offset[node]), int(offset[node + 1])):
            target = int(neighbor[adjacency_index])
            if component[target] != value or cross_label_key[target]:
                continue
            score = min(
                float(adjacency_affinity[adjacency_index]),
                float(evidence[target]),
            )
            if score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue,
                (
                    -score,
                    target,
                    int(seed_label[node]),
                    int(node),
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1

    while queue:
        negative_score, node, proposed_label, proposed_parent, proposed_edge = (
            heapq.heappop(queue)
        )
        if label[node] >= 0 or cross_label_key[node]:
            continue
        value = int(component[node])
        if (
            value < 0
            or int(component_label_count[value]) != 1
            or int(component_sole_label[value]) != proposed_label
        ):
            continue
        x, y, z = key[node]
        if occupied[z, y, x] >= 0:
            collision_rejected[node] = 1
            continue
        score = -negative_score
        occupied[z, y, x] = node
        label[node] = proposed_label
        path_score[node] = score
        parent[node] = proposed_parent
        parent_edge[node] = proposed_edge
        selection_class[node] = SELECTION_CLASS_PAIRED_GROWTH
        for adjacency_index in range(int(offset[node]), int(offset[node + 1])):
            target = int(neighbor[adjacency_index])
            if (
                label[target] >= 0
                or component[target] != value
                or cross_label_key[target]
            ):
                continue
            candidate_score = min(
                score,
                float(adjacency_affinity[adjacency_index]),
                float(evidence[target]),
            )
            if candidate_score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue,
                (
                    -candidate_score,
                    target,
                    proposed_label,
                    node,
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1

    selected = label >= 0
    same_label_edge = (
        selected[edge_first]
        & selected[edge_second]
        & (label[edge_first] == label[edge_second])
        & (edge_affinity >= settings.minimum_growth_bottleneck)
    )
    same_label_neighbor_count = np.bincount(
        np.concatenate(
            (edge_first[same_label_edge], edge_second[same_label_edge])
        ),
        minlength=node_count,
    ).astype(np.uint16)
    grown = selection_class == SELECTION_CLASS_PAIRED_GROWTH
    baseline_lost = baseline_selected & ~selected
    baseline_changed = baseline_selected & (
        label != np.asarray(baseline_growth["selectedLabel"])
    )
    if np.any(baseline_lost) or np.any(baseline_changed):
        raise RuntimeError("paired feedback changed a baseline assignment")
    return {
        "selectedLabel": label,
        "pathBottleneck": path_score,
        "parentCandidate": parent,
        "parentContinuityEdge": parent_edge,
        "selectionClass": selection_class,
        "selected": selected.astype(np.uint8),
        "selectedSameLabelNeighborCount": same_label_neighbor_count,
        "collisionRejected": collision_rejected,
    }, {
        "baselineSelectedCandidateCount": int(np.count_nonzero(baseline_selected)),
        "preservedBaselineCandidateCount": int(
            np.count_nonzero(baseline_selected & selected)
        ),
        "newClearCoreSeedCount": int(np.count_nonzero(new_seed)),
        "grownPairedCandidateCount": int(np.count_nonzero(grown)),
        "selectedCandidateCount": int(np.count_nonzero(selected)),
        "proposalCount": proposal_count,
        "collisionRejectedCandidateCount": int(
            np.count_nonzero(collision_rejected)
        ),
        "grownPathBottleneck": _percentile_record(path_score[grown]),
        "grownLocalEvidence": _percentile_record(evidence[grown]),
        "grownSameLabelNeighborCount": _percentile_record(
            same_label_neighbor_count[grown]
        ),
        "grownWithAtLeastTwoSameLabelNeighborsFraction": (
            round(float(np.mean(same_label_neighbor_count[grown] >= 2)), 6)
            if np.any(grown)
            else None
        ),
    }


def _new_label_records(
    selection: Mapping[str, np.ndarray],
    feedback_seed: Mapping[str, np.ndarray],
) -> list[dict[str, Any]]:
    label = np.asarray(selection["selectedLabel"], dtype=np.int32)
    selection_class = np.asarray(selection["selectionClass"])
    seed_label = np.asarray(feedback_seed["newSeedLabel"], dtype=np.int32)
    records = []
    for value in np.unique(seed_label[seed_label >= 0]):
        selected = label == value
        seed = seed_label == value
        grown = selected & (
            selection_class == SELECTION_CLASS_PAIRED_GROWTH
        )
        records.append(
            {
                "label": int(value),
                "seedCandidateCount": int(np.count_nonzero(seed)),
                "selectedCandidateCount": int(np.count_nonzero(selected)),
                "grownCandidateCount": int(np.count_nonzero(grown)),
                "growthFactor": round(
                    np.count_nonzero(selected) / max(np.count_nonzero(seed), 1),
                    6,
                ),
                "pathBottleneck": _percentile_record(
                    np.asarray(selection["pathBottleneck"])[grown]
                ),
            }
        )
    records.sort(key=lambda value: (-value["selectedCandidateCount"], value["label"]))
    return records


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_clear_ribbon_paired_feedback(
    feedback_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    (
        feedback_path,
        feedback_manifest,
        selection_path,
        selection_manifest,
        ribbon_selection,
        ribbon_path,
        ribbon_manifest,
        ribbons,
        growth_path,
        growth_manifest,
        baseline_growth,
        bank_path,
        bank_manifest,
        bank,
    ) = _load_inputs(feedback_root)
    graph = {
        name: values
        for name, values in baseline_growth.items()
        if name.startswith("edge")
    }
    settings = PairedSurfaceGrowthSettings(
        **growth_manifest["identity"]["settings"]
    )
    processing_shape = tuple(
        int(value)
        for value in bank_manifest["geometry"]["processingShapeSamplingXYZ"]
    )
    identity: dict[str, Any] = {
        "schema": CLEAR_RIBBON_PAIRED_FEEDBACK_SCHEMA,
        "version": CLEAR_RIBBON_PAIRED_FEEDBACK_VERSION,
        "interfaceFeedback": {
            "manifestPath": str(feedback_path),
            "manifestSha256": sha256_file(feedback_path),
            "dataSha256": feedback_manifest["data"]["sha256"],
        },
        "ribbonSelection": {
            "manifestPath": str(selection_path),
            "manifestSha256": sha256_file(selection_path),
            "dataSha256": selection_manifest["data"]["sha256"],
        },
        "ribbonBank": {
            "manifestPath": str(ribbon_path),
            "manifestSha256": sha256_file(ribbon_path),
            "dataSha256": ribbon_manifest["data"]["sha256"],
        },
        "baselinePairedGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "pairedBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "growthSettings": settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{CLEAR_RIBBON_PAIRED_FEEDBACK_STEM}.json"
    data_path = output / f"{CLEAR_RIBBON_PAIRED_FEEDBACK_STEM}.npz"
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
    feedback_seed, seed_stats = build_paired_feedback_seeds(
        bank,
        baseline_growth,
        ribbons,
        ribbon_selection,
        processing_shape_sampling_xyz=processing_shape,
        settings=settings,
    )
    seeded = time.monotonic()
    membership, component_stats = label_free_paired_components(
        bank,
        baseline_growth,
        graph,
        feedback_seed,
        processing_shape_sampling_xyz=processing_shape,
        settings=settings,
    )
    componentized = time.monotonic()
    grown, growth_stats = grow_clear_cores_in_paired_graph(
        bank,
        baseline_growth,
        graph,
        feedback_seed,
        membership,
        processing_shape_sampling_xyz=processing_shape,
        settings=settings,
    )
    new_labels = _new_label_records(grown, feedback_seed)
    solved = time.monotonic()
    arrays = {**grown, **membership, **feedback_seed}
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"]["metadataPath"]
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    world_record = bank_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    new_label_values = np.asarray(
        [value["label"] for value in new_labels], dtype=np.int32
    )
    new_selected = np.isin(grown["selectedLabel"], new_label_values)
    preview = {
        "selectedLabel": np.where(
            new_selected, grown["selectedLabel"], -1
        ).astype(np.int32),
        "lockedSeed": (
            grown["selectionClass"] == SELECTION_CLASS_NEW_CLEAR_CORE
        ).astype(np.uint8),
        "selected": new_selected.astype(np.uint8),
    }
    projection = write_growth_projection(
        bank,
        preview,
        world_start,
        world_stop,
        settings,
        output / "new-clear-core-paired-growth.png",
    )
    cross_sections = write_growth_cross_sections(
        source,
        owned,
        bank,
        preview,
        output / "new-clear-core-paired-cross-sections.png",
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=int(ribbon_manifest["samplingStrideVoxels"]),
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": CLEAR_RIBBON_PAIRED_FEEDBACK_SCHEMA,
        "version": CLEAR_RIBBON_PAIRED_FEEDBACK_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": bank_manifest["geometry"],
        "calibration": bank_manifest["calibration"],
        "seeding": seed_stats,
        "components": component_stats,
        "growth": growth_stats,
        "newClearCoreLabels": new_labels,
        "timingSeconds": {
            "feedbackSeeding": round(seeded - started, 6),
            "freeComponentLabeling": round(componentized - seeded, 6),
            "growthAndAudit": round(solved - componentized, 6),
            "writingAndArtifacts": round(finished - solved, 6),
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
        "selectionClass": {
            "0": "unselected",
            "1": "immutable baseline paired selection",
            "2": "new clear-ribbon seed",
            "3": "paired-profile growth from one unambiguous new identity",
        },
        "method": {
            "baselinePolicy": "immutable selected candidates and occupied keys",
            "ambiguityPolicy": "defer multi-label free components and key conflicts",
            "growthPolicy": "maximum-bottleneck over persisted paired graph",
            "changesExistingAssignments": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
