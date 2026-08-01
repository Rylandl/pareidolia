from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import (
    PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
    PHYSICAL_RIBBON_CONFIGURATION_STEM,
    PHYSICAL_RIBBON_CONFIGURATION_VERSION,
    _load_continuity_artifact,
)
from .physical_ribbon_continuity import (
    PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
    PHYSICAL_RIBBON_CONTINUITY_VERSION,
    PhysicalRibbonContinuitySettings,
)


PHYSICAL_RIBBON_REPLAY_CONFIGURATION_SCHEMA = (
    "pareidolia.physical-ribbon-replay-configuration"
)
PHYSICAL_RIBBON_REPLAY_CONFIGURATION_VERSION = 1

_REPLAY_MANIFEST_STEMS = (
    "physical-ribbon-one-sided-corridors-v1",
    "physical-ribbon-dormant-corridors-v1",
    "physical-ribbon-patch-corridors-v1",
    "physical-ribbon-complete-strip-replay-v1",
    "physical-ribbon-lineage-strip-replay-v1",
    "physical-ribbon-cumulative-corridor-replay-v1",
    "physical-ribbon-cumulative-hole-replay-v1",
    "physical-ribbon-collective-v1",
    "physical-ribbon-patch-state-v1",
    "physical-ribbon-texture-gate-v1",
)
_TOPOLOGY_FIELDS = (
    "frontierRibbonCandidate",
    "frontierMidpointKeyXYZ",
    "continuitySupportDegree",
    "continuitySupportScore",
    "tangentRankRatio",
    "selectionObjective",
    "selected",
    "component",
    "edgeFirstFrontierIndex",
    "edgeSecondFrontierIndex",
    "edgeScore",
    "edgeSelected",
    "edgeNormalDegrees",
    "edgeMidpointHeightResidualVoxels",
    "edgeBoundaryHeightResidualVoxels",
    "edgeThicknessChangeVoxels",
    "edgeBoundaryShiftDifferenceVoxels",
)
_CROSSING_FIELDS = (
    "crossingFirstFrontierIndex",
    "crossingSecondFrontierIndex",
    "crossingDistanceVoxels",
    "crossingFirstParameter",
    "crossingSecondParameter",
)


def _load_replay_artifact(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    if value.is_file():
        manifest_path = value
    else:
        matches = [
            value / f"{stem}.json"
            for stem in _REPLAY_MANIFEST_STEMS
            if (value / f"{stem}.json").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(
                "replay root must contain exactly one supported replay manifest"
            )
        manifest_path = matches[0]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("replay configuration requires a complete label-free replay")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    arrays = _load_npz(data_path, manifest["data"]["sha256"])
    required = {"corridorReplaySelected", "corridorReplayComponent"}
    if not required.issubset(arrays) and {"selected", "component"}.issubset(
        arrays
    ):
        # Exact strip replays expose the cumulative state under the ordinary
        # surface field names.  Normalize that newer contract here so the
        # materializer remains the single entry point for every exact replay.
        arrays = dict(arrays)
        arrays["corridorReplaySelected"] = np.asarray(
            arrays["selected"], dtype=np.uint8
        )
        arrays["corridorReplayComponent"] = np.asarray(
            arrays["component"], dtype=np.int32
        )
    if not required.issubset(arrays):
        raise ValueError("artifact does not contain a cumulative corridor replay")
    return manifest_path, manifest, arrays


def _topology_reference(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    identity = manifest["identity"]
    for name in ("frontier", "expandedContinuity", "topologyContinuity"):
        reference = identity.get(name)
        if reference is not None:
            return reference
    configuration_reference = identity.get("configuration")
    if configuration_reference is None:
        raise ValueError("replay artifact does not identify its topology")
    configuration_path = Path(configuration_reference["manifestPath"])
    if sha256_file(configuration_path) != configuration_reference["manifestSha256"]:
        raise ValueError("replay configuration reference has changed")
    configuration = json.loads(configuration_path.read_text())
    return configuration["identity"].get(
        "topologyContinuity", configuration["identity"]["continuity"]
    )


def _configuration_reference(manifest: Mapping[str, Any]) -> Mapping[str, Any]:
    reference = manifest["identity"].get("configuration")
    if reference is None:
        raise ValueError("replay artifact does not identify its source configuration")
    return reference


def _load_replay_topology(
    reference: Mapping[str, Any],
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    manifest_path = Path(reference["manifestPath"])
    if sha256_file(manifest_path) != reference["manifestSha256"]:
        raise ValueError("replay topology manifest has changed")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError("replay topology must be complete and label-free")
    if manifest["data"]["sha256"] != reference["dataSha256"]:
        raise ValueError("replay topology data identity differs")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    return manifest_path, manifest, _load_npz(data_path, reference["dataSha256"])


def _continuity_settings_from_provenance(
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    values = dict(manifest.get("identity", {}).get("settings", {}))
    try:
        PhysicalRibbonContinuitySettings(**values)
    except TypeError:
        identity = manifest.get("identity", {})
        for name in ("bidirectionalContinuity", "sourceTopology"):
            reference = identity.get(name)
            if reference is None:
                continue
            manifest_path = Path(reference["manifestPath"])
            if sha256_file(manifest_path) != reference["manifestSha256"]:
                raise ValueError("continuity settings provenance has changed")
            source = json.loads(manifest_path.read_text())
            if source["data"]["sha256"] != reference["dataSha256"]:
                raise ValueError("continuity settings data provenance differs")
            return _continuity_settings_from_provenance(source)
        raise ValueError("topology does not preserve physical continuity settings")
    return values


def _selection_component_sizes(
    selected: np.ndarray, component: np.ndarray
) -> np.ndarray:
    labels = component[selected]
    if np.any(labels < 0):
        raise ValueError("selected replay ribbons must have component labels")
    if np.any(component[~selected] >= 0):
        raise ValueError("unselected replay ribbons cannot own component labels")
    if not len(labels):
        return np.empty(0, dtype=np.int32)
    unique = np.unique(labels)
    if not np.array_equal(unique, np.arange(len(unique), dtype=unique.dtype)):
        raise ValueError("replay component labels must be contiguous")
    return np.bincount(labels, minlength=len(unique)).astype(np.int32)


def _materialize_replay_arrays(
    replay: Mapping[str, np.ndarray],
    topology: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    support_topology: Mapping[str, np.ndarray] | None = None,
    constraint_configuration: Mapping[str, np.ndarray] | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    missing = [name for name in _TOPOLOGY_FIELDS if name not in topology]
    if missing:
        raise ValueError(f"replay topology is missing fields: {missing}")
    selected = np.asarray(replay["corridorReplaySelected"], dtype=np.uint8) > 0
    component = np.asarray(replay["corridorReplayComponent"], dtype=np.int32)
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    if len(selected) != len(frontier) or len(component) != len(frontier):
        raise ValueError("replay selection and topology frontier differ")
    component_size = _selection_component_sizes(selected, component)
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    edge_selected = selected[edge_first] & selected[edge_second]
    support = topology if support_topology is None else support_topology
    support_frontier = np.asarray(
        support["frontierRibbonCandidate"], dtype=np.int32
    )
    bank_selected = np.zeros(
        len(np.asarray(ribbon["sourceInterface"])), dtype=bool
    )
    bank_selected[frontier[selected]] = True
    support_selected = bank_selected[support_frontier]
    support_first = np.asarray(
        support["edgeFirstFrontierIndex"], dtype=np.int32
    )
    support_second = np.asarray(
        support["edgeSecondFrontierIndex"], dtype=np.int32
    )
    support_edge_selected = (
        support_selected[support_first] & support_selected[support_second]
    )
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    selected_interfaces = np.concatenate((source[selected], target[selected]))
    interface_conflict_count = int(
        len(selected_interfaces) - len(np.unique(selected_interfaces))
    )
    # A materialized continuity artifact deliberately contains only the
    # immutable continuation graph.  Its paired configuration owns the hard
    # crossing constraints and conditioned unary scores.  The first replay in
    # a chain may still reference an older topology that carries those fields,
    # so support both contracts without weakening either audit.
    constraint_source = (
        topology
        if constraint_configuration is None
        else constraint_configuration
    )
    crossing_arrays: dict[str, np.ndarray] = {}
    for name in _CROSSING_FIELDS:
        source_arrays = topology if name in topology else constraint_source
        if name not in source_arrays:
            raise ValueError(f"replay topology is missing {name}")
        crossing_arrays[name] = np.asarray(source_arrays[name])
    crossing_first = np.asarray(
        crossing_arrays["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        crossing_arrays["crossingSecondFrontierIndex"], dtype=np.int32
    )
    crossing_conflict_count = int(
        np.count_nonzero(selected[crossing_first] & selected[crossing_second])
    )
    if interface_conflict_count or crossing_conflict_count:
        raise ValueError(
            "cumulative replay violates interface or crossing hard constraints"
        )
    materialized_topology = {
        name: np.asarray(topology[name]).copy() for name in _TOPOLOGY_FIELDS
    }
    materialized_topology["selected"] = selected.astype(np.uint8)
    materialized_topology["component"] = component.copy()
    materialized_topology["edgeSelected"] = edge_selected.astype(np.uint8)
    unary_source = (
        topology if "nodeUnaryScore" in topology else constraint_source
    )
    if "nodeUnaryScore" not in unary_source:
        raise ValueError("replay topology is missing conditioned unary scores")
    configuration = {
        **crossing_arrays,
        "nodeUnaryScore": np.asarray(unary_source["nodeUnaryScore"]).copy(),
        "initialSelected": selected.astype(np.uint8),
        "selected": selected.astype(np.uint8),
        "component": component.copy(),
        "supportEdgeSelected": support_edge_selected.astype(np.uint8),
        "edgeSelected": edge_selected.astype(np.uint8),
    }
    selected_edge_count = int(np.count_nonzero(edge_selected))
    statistics = {
        "frontierCandidateCount": len(frontier),
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "selectedInterfaceCount": int(len(selected_interfaces)),
        "selectedSupportEdgeCount": int(
            np.count_nonzero(support_edge_selected)
        ),
        "selectedContinuityEdgeCount": selected_edge_count,
        "selectedCrossingConflictCount": crossing_conflict_count,
        "selectedInterfaceConflictCount": interface_conflict_count,
        "componentCount": len(component_size),
        "componentWithAtLeast8RibbonsCount": int(
            np.count_nonzero(component_size >= 8)
        ),
        "componentWithAtLeast32RibbonsCount": int(
            np.count_nonzero(component_size >= 32)
        ),
        "largestComponentRibbonCounts": [
            int(value) for value in np.sort(component_size)[::-1][:32]
        ],
        "identityLabelsUsed": False,
    }
    return materialized_topology, configuration, statistics


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def run_physical_ribbon_replay_configuration(
    replay_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    replay_path, replay_manifest, replay = _load_replay_artifact(replay_root)
    topology_path, topology_manifest, topology = _load_replay_topology(
        _topology_reference(replay_manifest)
    )
    configuration_reference = _configuration_reference(replay_manifest)
    (
        configuration_path,
        configuration_manifest,
        source_configuration,
        _,
        _,
        _,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path)
        != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("replay source configuration has changed")
    topology_ribbon = topology_manifest["identity"].get("ribbonBank")
    if (
        topology_ribbon is not None
        and topology_ribbon["dataSha256"] != ribbon_manifest["data"]["sha256"]
    ):
        raise ValueError("replay topology and configuration use different banks")
    support_reference = configuration_manifest["identity"]["continuity"]
    support_path, support_manifest, support_topology = (
        _load_continuity_artifact(support_reference["manifestPath"])
    )
    if (
        sha256_file(support_path) != support_reference["manifestSha256"]
        or support_manifest["data"]["sha256"]
        != support_reference["dataSha256"]
    ):
        raise ValueError("source configuration support continuity has changed")

    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    continuity_path = output / "physical-ribbon-continuity-v1.json"
    continuity_data_path = output / "physical-ribbon-continuity-v1.npz"
    final_path = output / f"{PHYSICAL_RIBBON_CONFIGURATION_STEM}.json"
    final_data_path = output / f"{PHYSICAL_RIBBON_CONFIGURATION_STEM}.npz"
    source_replay = _reference(replay_path, replay_manifest)
    source_topology = _reference(topology_path, topology_manifest)
    implementation_sha = sha256_file(Path(__file__))
    continuity_identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
        "version": PHYSICAL_RIBBON_CONTINUITY_VERSION,
        "ribbonBank": _reference(ribbon_path, ribbon_manifest),
        "sourceReplay": source_replay,
        "sourceTopology": source_topology,
        "supportContinuity": _reference(support_path, support_manifest),
        "settings": _continuity_settings_from_provenance(topology_manifest),
        "implementationSha256": implementation_sha,
    }
    continuity_identity["identitySha256"] = canonical_json_hash(
        continuity_identity
    )
    configuration_identity_base: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CONFIGURATION_VERSION,
        "sourceReplay": source_replay,
        "sourceConfiguration": _reference(
            configuration_path, configuration_manifest
        ),
        "ribbonBank": _reference(ribbon_path, ribbon_manifest),
        "interfaceBank": configuration_manifest["identity"]["interfaceBank"],
        "settings": configuration_manifest["identity"]["settings"],
        "implementationSha256": implementation_sha,
    }
    cached_identity = dict(configuration_identity_base)
    cached_identity["continuityIdentitySha256"] = continuity_identity[
        "identitySha256"
    ]
    cached_identity_sha = canonical_json_hash(cached_identity)
    if not force and final_path.is_file() and final_data_path.is_file():
        cached = json.loads(final_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("materializationIdentitySha256")
            == cached_identity_sha
            and cached.get("data", {}).get("sha256")
            == sha256_file(final_data_path)
            and continuity_path.is_file()
            and continuity_data_path.is_file()
        ):
            return cached

    started = time.monotonic()
    materialized_topology, configuration, statistics = (
        _materialize_replay_arrays(
            replay,
            topology,
            ribbon,
            support_topology=support_topology,
            constraint_configuration=source_configuration,
        )
    )
    materialized_at = time.monotonic()
    _write_npz(continuity_data_path, materialized_topology)
    continuity_payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONTINUITY_SCHEMA,
        "version": PHYSICAL_RIBBON_CONTINUITY_VERSION,
        "state": "complete",
        "identity": continuity_identity,
        "source": replay_manifest.get(
            "source", configuration_manifest["source"]
        ),
        "geometry": topology_manifest.get(
            "geometry", configuration_manifest.get("geometry", {})
        ),
        "continuity": {
            "frontierCandidateCount": statistics["frontierCandidateCount"],
            "compatibleContinuationEdgeCount": len(
                materialized_topology["edgeFirstFrontierIndex"]
            ),
            "selectedRibbonCount": statistics["selectedRibbonCount"],
            "selectedContinuationEdgeCount": statistics[
                "selectedContinuityEdgeCount"
            ],
            "componentCount": statistics["componentCount"],
            "componentWithAtLeast8RibbonsCount": statistics[
                "componentWithAtLeast8RibbonsCount"
            ],
            "componentWithAtLeast32RibbonsCount": statistics[
                "componentWithAtLeast32RibbonsCount"
            ],
            "largestComponentRibbonCounts": statistics[
                "largestComponentRibbonCounts"
            ],
            "identityLabelsUsed": False,
        },
        "data": {
            "path": continuity_data_path.name,
            "bytes": continuity_data_path.stat().st_size,
            "sha256": sha256_file(continuity_data_path),
            "fields": list(materialized_topology),
        },
        "method": {
            "topology": "immutable strict continuation graph from the replay frontier",
            "selection": "cumulative exact corridor replay materialized as baseline state",
            "identityLabelsUsed": False,
        },
    }
    atomic_json(continuity_path, continuity_payload)
    topology_reference = _reference(continuity_path, continuity_payload)
    configuration_identity = dict(configuration_identity_base)
    configuration_identity["continuity"] = _reference(
        support_path, support_manifest
    )
    configuration_identity["topologyContinuity"] = topology_reference
    configuration_identity["materializationIdentitySha256"] = (
        cached_identity_sha
    )
    configuration_identity["identitySha256"] = canonical_json_hash(
        configuration_identity
    )
    _write_npz(final_data_path, configuration)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CONFIGURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_CONFIGURATION_VERSION,
        "state": "complete",
        "identity": configuration_identity,
        "source": continuity_payload["source"],
        "geometry": continuity_payload["geometry"],
        "crossings": {
            "crossingPairCount": len(
                configuration["crossingFirstFrontierIndex"]
            ),
            "selectedCrossingConflictCount": statistics[
                "selectedCrossingConflictCount"
            ],
        },
        "configuration": statistics,
        "timingSeconds": {
            "materializationAndAudit": round(materialized_at - started, 6),
            "writing": round(finished - materialized_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": final_data_path.name,
            "bytes": final_data_path.stat().st_size,
            "sha256": sha256_file(final_data_path),
            "fields": list(configuration),
        },
        "artifacts": {
            "materializedContinuity": continuity_path.name,
        },
        "method": {
            "objective": "preserve the exact cumulative replay without reoptimization",
            "hardConstraints": "one ribbon per interface and no enforced profile crossing",
            "alternatives": "all unselected frontier candidates remain available",
            "topology": "the replay frontier strict-continuation graph defines component identity",
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
        "materialization": {
            "schema": PHYSICAL_RIBBON_REPLAY_CONFIGURATION_SCHEMA,
            "version": PHYSICAL_RIBBON_REPLAY_CONFIGURATION_VERSION,
            "sourceReplaySchema": replay_manifest["schema"],
        },
    }
    atomic_json(final_path, payload)
    return payload
