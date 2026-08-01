from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import (
    PhysicalRibbonConfigurationSettings,
    _load_continuity_artifact,
)
from .physical_ribbon_continuity import (
    PhysicalRibbonContinuitySettings,
    build_paired_boundary_continuity,
)
from .physical_ribbon_corridor_dormant import (
    PHYSICAL_RIBBON_DORMANT_CORRIDORS_STEM,
    _condition_configuration_on_expanded_frontier,
    _load_interfaces,
    _union_crossing_continuity,
)
from .physical_ribbon_corridor_variants import (
    _corridor_settings_from_manifest,
    _load_corridor_artifact,
)
from .physical_ribbon_patch_corridors import _corridor_geometric_candidates


PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA = (
    "pareidolia.physical-ribbon-corridor-frontier"
)
PHYSICAL_RIBBON_CORRIDOR_FRONTIER_VERSION = 1
PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM = (
    "physical-ribbon-corridor-frontier-v1"
)


@dataclass(frozen=True, slots=True)
class PhysicalRibbonCorridorFrontierSettings:
    require_unidirectional_candidates: bool = True
    target_only_unresolved_ct_corridors: bool = True

    def __post_init__(self) -> None:
        if not self.require_unidirectional_candidates:
            raise ValueError(
                "the targeted frontier must remain a one-sided-candidate test"
            )
        if not self.target_only_unresolved_ct_corridors:
            raise ValueError(
                "the targeted frontier must preserve prior exact decisions"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_prior_replay(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    if value.is_file():
        manifest_path = value
    else:
        stems = (
            PHYSICAL_RIBBON_DORMANT_CORRIDORS_STEM,
            "physical-ribbon-one-sided-corridors-v1",
            "physical-ribbon-patch-corridors-v1",
        )
        matches = [
            value / f"{stem}.json"
            for stem in stems
            if (value / f"{stem}.json").is_file()
        ]
        if len(matches) != 1:
            raise ValueError(
                "prior replay root must contain exactly one supported manifest"
            )
        manifest_path = matches[0]
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("state") != "complete"
        or manifest.get("method", {}).get("identityLabelsUsed") is not False
    ):
        raise ValueError(
            "corridor frontier requires a complete label-free cumulative replay"
        )
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    arrays = _load_npz(data_path, manifest["data"]["sha256"])
    if not {
        "corridorReplaySelected",
        "corridorReplayComponent",
        "corridorReplayProposalSuccessful",
    }.issubset(arrays):
        raise ValueError("prior artifact does not contain a cumulative replay")
    return (
        manifest_path,
        manifest,
        arrays,
    )


def _prior_topology_reference(
    manifest: Mapping[str, Any],
) -> Mapping[str, Any]:
    identity = manifest["identity"]
    for name in ("frontier", "expandedContinuity", "topologyContinuity"):
        reference = identity.get(name)
        if reference is not None:
            return reference
    raise ValueError("prior replay does not identify its topology")


def _map_frontier_by_bank(
    source_frontier: np.ndarray,
    target_frontier: np.ndarray,
    *,
    ribbon_bank_count: int,
) -> np.ndarray:
    bank_to_target = np.full(ribbon_bank_count, -1, dtype=np.int32)
    bank_to_target[np.asarray(target_frontier, dtype=np.int32)] = np.arange(
        len(target_frontier), dtype=np.int32
    )
    result = bank_to_target[np.asarray(source_frontier, dtype=np.int32)]
    if np.any(result < 0):
        raise ValueError("targeted frontier does not contain the source frontier")
    return result


def _sampling_geometry(
    manifest: Mapping[str, Any],
) -> tuple[np.ndarray, int]:
    geometry = manifest["geometry"]
    source_origin = np.asarray(
        manifest["source"]["sourceOriginXYZ"], dtype=np.float32
    )
    processing_start = np.asarray(
        geometry["processingVoxelBounds"]["startXYZ"], dtype=np.float32
    )
    processing_stop = np.asarray(
        geometry["processingVoxelBounds"]["stopXYZExclusive"],
        dtype=np.float32,
    )
    processing_shape = np.asarray(
        geometry["processingShapeSamplingXYZ"], dtype=np.int32
    )
    stride_xyz = (processing_stop - processing_start) / processing_shape
    if not np.allclose(stride_xyz, stride_xyz[0]):
        raise ValueError("corridor frontier requires isotropic sampling")
    return source_origin + processing_start, int(round(float(stride_xyz[0])))


def _collect_unidirectional_corridor_candidates(
    target_rows: np.ndarray,
    corridor: Mapping[str, np.ndarray],
    ribbon: Mapping[str, np.ndarray],
    selected_bank: np.ndarray,
    *,
    corridor_settings: Any,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    bank_count = len(np.asarray(ribbon["sourceInterface"]))
    full_frontier = np.arange(bank_count, dtype=np.int32)
    selected = np.zeros(bank_count, dtype=np.uint8)
    selected[selected_bank] = 1
    topology = {"frontierRibbonCandidate": full_frontier}
    configuration = {"selected": selected}
    offsets = [0]
    bank_values: list[int] = []
    nearest_values: list[int] = []
    height_values: list[float] = []
    tangent_values: list[float] = []
    normal_values: list[float] = []
    thickness_values: list[float] = []
    alignment_values: list[float] = []
    records: list[dict[str, Any]] = []
    bidirectional = np.asarray(ribbon["bidirectional"]) > 0
    for row_value in target_rows:
        row = int(row_value)
        geometric = _corridor_geometric_candidates(
            row,
            corridor,
            ribbon,
            topology,
            configuration,
            settings=corridor_settings,
        )
        bank = np.asarray(geometric["frontierIndex"], dtype=np.int32)
        retained = ~bidirectional[bank]
        bank = bank[retained]
        bank_values.extend(int(value) for value in bank)
        nearest_values.extend(
            int(value)
            for value in np.asarray(geometric["nearestPatchPixel"])[retained]
        )
        height_values.extend(
            float(value)
            for value in np.asarray(geometric["heightResidualVoxels"])[retained]
        )
        tangent_values.extend(
            float(value)
            for value in np.asarray(geometric["tangentResidualVoxels"])[retained]
        )
        normal_values.extend(
            float(value)
            for value in np.asarray(geometric["normalResidualDegrees"])[retained]
        )
        thickness_values.extend(
            float(value)
            for value in np.asarray(geometric["thicknessRatio"])[retained]
        )
        alignment = np.asarray(geometric["surfaceAlignment"])[retained]
        alignment_values.extend(float(value) for value in alignment)
        offsets.append(len(bank_values))
        records.append(
            {
                "corridorRow": row,
                "unidirectionalCandidateCount": len(bank),
                "medianSurfaceAlignment": (
                    round(float(np.median(alignment)), 6)
                    if len(alignment)
                    else None
                ),
                "maximumSurfaceAlignment": (
                    round(float(np.max(alignment)), 6)
                    if len(alignment)
                    else None
                ),
            }
        )
    values = np.asarray(bank_values, dtype=np.int32)
    return {
        "targetCorridorRow": np.asarray(target_rows, dtype=np.int32),
        "targetCorridorCandidateOffset": np.asarray(offsets, dtype=np.int64),
        "targetCorridorCandidateBankIndex": values,
        "targetCorridorCandidateNearestPatchPixel": np.asarray(
            nearest_values, dtype=np.int32
        ),
        "targetCorridorCandidateHeightResidualVoxels": np.asarray(
            height_values, dtype=np.float32
        ),
        "targetCorridorCandidateTangentResidualVoxels": np.asarray(
            tangent_values, dtype=np.float32
        ),
        "targetCorridorCandidateNormalResidualDegrees": np.asarray(
            normal_values, dtype=np.float32
        ),
        "targetCorridorCandidateThicknessRatio": np.asarray(
            thickness_values, dtype=np.float32
        ),
        "targetCorridorCandidateSurfaceAlignment": np.asarray(
            alignment_values, dtype=np.float32
        ),
        "oneSidedCandidateBankIndex": np.unique(values),
    }, {
        "targetCorridorCount": len(target_rows),
        "candidateOccurrenceCount": len(values),
        "uniqueUnidirectionalCandidateCount": len(np.unique(values)),
        "corridors": records,
        "decision": (
            "all one-sided bank candidates passing the existing complete-strip "
            "geometry gate for a still-unresolved native-CT corridor"
        ),
        "identityLabelsUsed": False,
    }


def _artifact_reference(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    return {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
        "dataSha256": manifest["data"]["sha256"],
    }


def run_physical_ribbon_corridor_frontier(
    corridor_root: str | Path,
    prior_replay_root: str | Path,
    configuration_root: str | Path,
    bidirectional_continuity_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonCorridorFrontierSettings | None = None,
    force: bool = False,
    progress: Any | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonCorridorFrontierSettings()
    corridor_path, corridor_manifest, corridor = _load_corridor_artifact(
        corridor_root
    )
    prior_path, prior_manifest, prior = _load_prior_replay(prior_replay_root)
    (
        configuration_path,
        configuration_manifest,
        base_configuration,
        base_topology_path,
        base_topology_manifest,
        base_topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_root)
    continuity_path, continuity_manifest, continuity = (
        _load_continuity_artifact(bidirectional_continuity_root)
    )
    prior_configuration_reference = prior_manifest["identity"].get(
        "configuration"
    )
    if prior_configuration_reference is None:
        raise ValueError("prior replay does not identify its configuration")
    prior_topology_reference = _prior_topology_reference(prior_manifest)
    if (
        prior_configuration_reference["dataSha256"]
        != configuration_manifest["data"]["sha256"]
        or prior_topology_reference["dataSha256"]
        != continuity_manifest["data"]["sha256"]
        or continuity_manifest["identity"]["ribbonBank"]["dataSha256"]
        != ribbon_manifest["data"]["sha256"]
    ):
        raise ValueError("corridor frontier inputs do not share one ribbon state")
    prior_corridor_reference = prior_manifest["identity"].get("corridors")
    same_corridor_artifact = prior_path == corridor_path
    if not same_corridor_artifact and (
        prior_corridor_reference is None
        or prior_corridor_reference["dataSha256"]
        != corridor_manifest["data"]["sha256"]
    ):
        raise ValueError("prior replay decisions belong to different corridors")
    support_reference = configuration_manifest["identity"]["continuity"]
    support_path, support_manifest, support_topology = (
        _load_continuity_artifact(support_reference["manifestPath"])
    )
    if (
        sha256_file(support_path) != support_reference["manifestSha256"]
        or support_manifest["data"]["sha256"]
        != support_reference["dataSha256"]
    ):
        raise ValueError("configuration support continuity has changed")
    interface_path, interface_manifest, interfaces = _load_interfaces(
        ribbon_manifest
    )
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FRONTIER_VERSION,
        "corridors": _artifact_reference(corridor_path, corridor_manifest),
        "priorReplay": _artifact_reference(prior_path, prior_manifest),
        "configuration": _artifact_reference(
            configuration_path, configuration_manifest
        ),
        "baseTopology": _artifact_reference(
            base_topology_path, base_topology_manifest
        ),
        "bidirectionalContinuity": _artifact_reference(
            continuity_path, continuity_manifest
        ),
        "supportContinuity": _artifact_reference(
            support_path, support_manifest
        ),
        "ribbonBank": _artifact_reference(ribbon_path, ribbon_manifest),
        "interfaceBank": _artifact_reference(interface_path, interface_manifest),
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
        "continuityImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_continuity.py")
        ),
        "conditioningImplementationSha256": sha256_file(
            Path(__file__).with_name("physical_ribbon_corridor_dormant.py")
        ),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_CORRIDOR_FRONTIER_STEM}.npz"
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
    prior_selected = np.asarray(prior["corridorReplaySelected"]) > 0
    continuity_frontier = np.asarray(
        continuity["frontierRibbonCandidate"], dtype=np.int32
    )
    if len(prior_selected) != len(continuity_frontier):
        raise ValueError("prior replay and bidirectional frontier differ")
    selected_bank = continuity_frontier[prior_selected]
    successful = np.asarray(
        prior["corridorReplayProposalSuccessful"], dtype=np.uint8
    ) > 0
    evidence = np.asarray(corridor["corridorEvidenceEligible"]) > 0
    target_rows = np.flatnonzero(evidence & ~successful).astype(np.int32)
    corridor_settings = _corridor_settings_from_manifest(corridor_manifest)
    if progress is not None:
        progress(
            f"collecting complete-strip one-sided candidates for "
            f"{len(target_rows)} unresolved corridors"
        )
    candidate_arrays, candidate_stats = (
        _collect_unidirectional_corridor_candidates(
            target_rows,
            corridor,
            ribbon,
            selected_bank,
            corridor_settings=corridor_settings,
        )
    )
    collected_at = time.monotonic()
    one_sided = np.asarray(
        candidate_arrays["oneSidedCandidateBankIndex"], dtype=np.int32
    )
    targeted_frontier = np.unique(
        np.concatenate((continuity_frontier, one_sided))
    ).astype(np.int32)
    processing_origin, stride = _sampling_geometry(continuity_manifest)
    continuity_settings = PhysicalRibbonContinuitySettings(
        **continuity_manifest["identity"]["settings"]
    )
    if progress is not None:
        progress(
            f"building strict continuity for {len(targeted_frontier)} targeted "
            "ribbons"
        )
    topology, topology_stats = build_paired_boundary_continuity(
        ribbon,
        interfaces,
        processing_world_start_xyz=processing_origin,
        sampling_stride_voxels=stride,
        settings=continuity_settings,
        frontier_bank_index=targeted_frontier,
    )
    topologized_at = time.monotonic()
    crossing_topology, crossing_stats = _union_crossing_continuity(
        topology,
        support_topology,
        ribbon_bank_count=len(np.asarray(ribbon["sourceInterface"])),
    )
    prior_configuration = {
        "selected": np.asarray(prior["corridorReplaySelected"], dtype=np.uint8),
        "component": np.asarray(prior["corridorReplayComponent"], dtype=np.int32),
    }
    if progress is not None:
        progress("conditioning the targeted frontier on the cumulative replay")
    conditioned, prior_to_target, conditioning_stats = (
        _condition_configuration_on_expanded_frontier(
            ribbon,
            interfaces,
            continuity,
            topology,
            prior_configuration,
            crossing_topology=crossing_topology,
            continuity_manifest=continuity_manifest,
            configuration_settings=PhysicalRibbonConfigurationSettings(
                **configuration_manifest["identity"]["settings"]
            ),
        )
    )
    original_to_target = _map_frontier_by_bank(
        np.asarray(base_topology["frontierRibbonCandidate"], dtype=np.int32),
        np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32),
        ribbon_bank_count=len(np.asarray(ribbon["sourceInterface"])),
    )
    conditioning_stats["crossingContinuity"] = crossing_stats
    conditioned_at = time.monotonic()
    arrays = {
        **topology,
        **conditioned,
        **candidate_arrays,
        "priorFrontierToTargetFrontier": prior_to_target,
        "originalFrontierToTargetFrontier": original_to_target,
    }
    _write_npz(data_path, arrays)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_CORRIDOR_FRONTIER_SCHEMA,
        "version": PHYSICAL_RIBBON_CORRIDOR_FRONTIER_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": continuity_manifest["geometry"],
        "targets": {
            "ctEvidenceCorridorCount": int(np.count_nonzero(evidence)),
            "priorSuccessfulCorridorCount": int(np.count_nonzero(successful)),
            "unresolvedCorridorCount": len(target_rows),
            "unresolvedCorridorRows": [int(value) for value in target_rows],
        },
        "candidates": candidate_stats,
        "topology": topology_stats,
        "conditioning": conditioning_stats,
        "timingSeconds": {
            "candidateCollection": round(collected_at - started, 6),
            "targetedContinuity": round(topologized_at - collected_at, 6),
            "crossingsAndConditioning": round(
                conditioned_at - topologized_at, 6
            ),
            "writing": round(finished - conditioned_at, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "method": {
            "admissionUnit": "complete unresolved native-CT corridor strip",
            "candidatePolicy": (
                "all geometrically compatible one-sided ribbon hypotheses in "
                "those strips; no global or single-cell growth"
            ),
            "baseSelection": "the complete prior cumulative exact replay",
            "selectionMutated": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
