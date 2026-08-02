from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .isolated_slab import _percentile_record
from .macro_orientation import MACRO_ORIENTATION_SCHEMA, MACRO_ORIENTATION_STEM
from .material_interface import MATERIAL_INTERFACE_SCHEMA
from .material_surface_graph import (
    MATERIAL_SURFACE_GRAPH_SCHEMA,
    MATERIAL_SURFACE_GRAPH_STEM,
    MaterialSurfaceGraphSettings,
    write_material_surface_cross_sections,
)
from .one_sided_interface import (
    OneSidedInterfaceSettings,
    match_signed_interface_endpoints,
)
from .paired_surface_bank import PAIRED_SURFACE_BANK_SCHEMA, PAIRED_SURFACE_BANK_STEM
from .paired_surface_growth import (
    PAIRED_SURFACE_GROWTH_SCHEMA,
    PAIRED_SURFACE_GROWTH_STEM,
)
from .physical_mid_surface import (
    PHYSICAL_MID_SURFACE_SCHEMA,
    PHYSICAL_MID_SURFACE_STEM,
    PHYSICAL_MID_SURFACE_VERSION,
    PhysicalMidSurfaceSettings,
    _components,
    _midpoint_edges,
    _normalized,
    _profile_attachment_edges,
    _write_npz,
    local_profile_thickness_prior,
    pair_physical_boundary_faces,
    propagate_profile_thickness_prior,
)


LAMINAR_RIBBON_SCHEMA = "pareidolia.laminar-ribbon-mid-surface-catalog"
LAMINAR_RIBBON_VERSION = 1


@dataclass(frozen=True, slots=True)
class LaminarRibbonSettings:
    """Evidence gates for turning two dense boundary faces into one ribbon.

    The construction is deliberately independent of historical sheet labels.
    A ribbon exists only when repeated, physically bounded air-papyrus-air
    profiles support the same unordered pair of signed boundary components and
    agree with the locally dominant laminar field.
    """

    minimum_profile_evidence: float = 0.35
    minimum_macro_confidence: float = 0.35
    maximum_profile_to_macro_normal_degrees: float = 25.0
    maximum_median_profile_to_macro_normal_degrees: float = 20.0
    # This is a hypothesis bank, not the identity solver.  Keep short local
    # face-pair observations here; the downstream boundary-matching MIP owns
    # the one-mate constraint and decides which hypotheses are globally
    # compatible.  Requiring a large pre-assembled patch at this stage makes
    # real sheets disappear precisely where the CT evidence is interrupted.
    minimum_support_profiles: int = 4
    minimum_conservative_support_profiles: int = 2
    minimum_support_extent_sampling_steps: float = 3.0
    maximum_selected_ribbons: int = 768
    maximum_endpoint_position_residual_sampling_steps: float = 0.75
    maximum_endpoint_normal_degrees: float = 15.0
    endpoint_normal_cost_scale_degrees: float = 15.0

    def __post_init__(self) -> None:
        for value, name in (
            (self.minimum_profile_evidence, "profile evidence"),
            (self.minimum_macro_confidence, "macro confidence"),
        ):
            if not math.isfinite(value) or not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must lie in [0, 1]")
        for value in (
            self.maximum_profile_to_macro_normal_degrees,
            self.maximum_median_profile_to_macro_normal_degrees,
            self.maximum_endpoint_normal_degrees,
        ):
            if not math.isfinite(value) or not 0.0 < value < 90.0:
                raise ValueError("laminar ribbon angle gates must lie in (0, 90)")
        for value in (
            self.minimum_support_extent_sampling_steps,
            self.maximum_endpoint_position_residual_sampling_steps,
            self.endpoint_normal_cost_scale_degrees,
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError("laminar ribbon distances must be positive")
        if (
            self.minimum_support_profiles < 1
            or self.minimum_conservative_support_profiles < 1
            or self.maximum_selected_ribbons < 1
        ):
            raise ValueError("laminar ribbon counts must be positive")
        if self.minimum_conservative_support_profiles > self.minimum_support_profiles:
            raise ValueError(
                "conservative ribbon support cannot exceed total support"
            )

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _load(path: Path) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    manifest = json.loads(path.read_text())
    data_path = path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError(f"artifact data changed after its manifest: {path}")
    with np.load(data_path, allow_pickle=False) as stored:
        arrays = {name: np.asarray(stored[name]) for name in stored.files}
    return manifest, arrays


def _macro_profile_assignment(
    spatial_key_xyz: np.ndarray,
    profile_normal_xyz: np.ndarray,
    macro_manifest: Mapping[str, Any],
    macro: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """Map profile keys to the generic, cross-layer laminar tensor.

    Physical per-bin modes are intentionally not used here. They are useful
    for preserving alternatives, but a repeated transverse false family can
    also become a mode. The generic tensor is the independent macro guard.
    """

    key = np.asarray(spatial_key_xyz, dtype=np.int64)
    normal = _normalized(np.asarray(profile_normal_xyz, dtype=np.float64))
    support = int(
        macro_manifest["identity"]["resolvedScale"]["supportSamplingSteps"]
    )
    profile_bin_key = key // support
    field_bin_key = np.asarray(macro["binKeyXYZ"], dtype=np.int64)
    low = np.minimum(profile_bin_key.min(axis=0), field_bin_key.min(axis=0))
    high = np.maximum(profile_bin_key.max(axis=0), field_bin_key.max(axis=0))
    shape = high - low + 1

    def flat(values: np.ndarray) -> np.ndarray:
        return np.ravel_multi_index(
            tuple((values - low[None, :]).T), tuple(int(v) for v in shape)
        )

    field_flat = flat(field_bin_key)
    order = np.argsort(field_flat, kind="stable")
    sorted_flat = field_flat[order]
    query = flat(profile_bin_key)
    insertion = np.searchsorted(sorted_flat, query)
    inside = insertion < len(sorted_flat)
    mapped = np.full(len(query), -1, dtype=np.int32)
    mapped[inside] = order[insertion[inside]]
    exact = np.zeros(len(query), dtype=bool)
    exact[inside] = sorted_flat[insertion[inside]] == query[inside]
    mapped[~exact] = -1

    field_normal = np.zeros_like(normal)
    confidence = np.zeros(len(normal), dtype=np.float32)
    trusted = np.zeros(len(normal), dtype=bool)
    available = mapped >= 0
    field_normal[available] = np.asarray(macro["normalXYZ"])[mapped[available]]
    confidence[available] = np.asarray(macro["orientationConfidence"])[
        mapped[available]
    ]
    trusted[available] = np.asarray(macro["trusted"])[mapped[available]] > 0
    cosine = np.zeros(len(normal), dtype=np.float64)
    cosine[available] = np.abs(
        np.einsum("ij,ij->i", normal[available], field_normal[available])
    )
    residual = np.full(len(normal), 90.0, dtype=np.float32)
    residual[available] = np.degrees(
        np.arccos(np.clip(cosine[available], 0.0, 1.0))
    ).astype(np.float32)
    return {
        "macroBinIndex": mapped,
        "macroNormalXYZ": field_normal.astype(np.float32),
        "macroConfidence": confidence,
        "macroTrusted": trusted,
        "macroNormalResidualDegrees": residual,
    }


def _deduplicate_pair_keys(
    candidate: np.ndarray,
    pair_code: np.ndarray,
    spatial_key: np.ndarray,
    conservative: np.ndarray,
    evidence: np.ndarray,
) -> np.ndarray:
    """Retain one best profile for each face-pair and lattice key."""

    index = np.asarray(candidate, dtype=np.int32)
    pair = np.asarray(pair_code, dtype=np.uint64)
    key = np.asarray(spatial_key, dtype=np.int64)
    order = np.lexsort(
        (
            index,
            -np.asarray(evidence, dtype=np.float64),
            -np.asarray(conservative, dtype=np.int8),
            key[:, 2],
            key[:, 1],
            key[:, 0],
            pair,
        )
    )
    ordered_pair = pair[order]
    ordered_key = key[order]
    unique = np.ones(len(order), dtype=bool)
    if len(order) > 1:
        unique[1:] = (ordered_pair[1:] != ordered_pair[:-1]) | np.any(
            ordered_key[1:] != ordered_key[:-1], axis=1
        )
    return index[order[unique]]


def discover_laminar_ribbons(
    bank: Mapping[str, np.ndarray],
    surface: Mapping[str, np.ndarray],
    interfaces: Mapping[str, np.ndarray],
    macro_manifest: Mapping[str, Any],
    macro: Mapping[str, np.ndarray],
    *,
    processing_start_xyz: np.ndarray,
    source_origin_xyz: np.ndarray,
    processing_shape_sampling_xyz: tuple[int, int, int],
    sampling_stride_voxels: int,
    settings: LaminarRibbonSettings,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], dict[str, Any]]:
    """Find repeated physical profiles supporting the same two dense faces."""

    midpoint = np.asarray(bank["midpointXYZ"], dtype=np.float32)
    profile_normal = np.asarray(bank["normalXYZ"], dtype=np.float32)
    lower = np.asarray(bank["boundaryLowerXYZ"], dtype=np.float32)
    upper = np.asarray(bank["boundaryUpperXYZ"], dtype=np.float32)
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    conservative = np.asarray(bank["isolatedConservative"], dtype=bool)
    candidate_count = len(midpoint)
    if any(
        len(value) != candidate_count
        for value in (profile_normal, lower, upper, key, evidence, conservative)
    ):
        raise ValueError("paired profile arrays are not aligned")

    endpoint_match = match_signed_interface_endpoints(
        np.concatenate((lower, upper), axis=0),
        np.concatenate((profile_normal, -profile_normal), axis=0),
        np.asarray(interfaces["positionXYZ"]),
        np.asarray(interfaces["signedNormalXYZ"]),
        np.asarray(interfaces["processingKeyXYZ"]),
        processing_start_xyz=np.asarray(processing_start_xyz, dtype=np.float64),
        source_origin_xyz=np.asarray(source_origin_xyz, dtype=np.float64),
        processing_shape_sampling_xyz=processing_shape_sampling_xyz,
        stride=int(sampling_stride_voxels),
        settings=OneSidedInterfaceSettings(
            maximum_seed_position_residual_sampling_steps=(
                settings.maximum_endpoint_position_residual_sampling_steps
            ),
            maximum_seed_normal_degrees=settings.maximum_endpoint_normal_degrees,
            seed_match_normal_scale_degrees=(
                settings.endpoint_normal_cost_scale_degrees
            ),
        ),
    )
    endpoint_interface = np.asarray(endpoint_match["interfaceIndex"], dtype=np.int32)
    interface_to_surface = np.full(
        len(interfaces["positionXYZ"]), -1, dtype=np.int32
    )
    source_interface = np.asarray(surface["interfaceIndex"], dtype=np.int64)
    interface_to_surface[source_interface] = np.arange(
        len(source_interface), dtype=np.int32
    )
    endpoint_surface = np.full(len(endpoint_interface), -1, dtype=np.int32)
    matched = endpoint_interface >= 0
    endpoint_surface[matched] = interface_to_surface[endpoint_interface[matched]]
    lower_surface = endpoint_surface[:candidate_count]
    upper_surface = endpoint_surface[candidate_count:]
    lower_component = np.full(candidate_count, -1, dtype=np.int32)
    upper_component = np.full(candidate_count, -1, dtype=np.int32)
    component = np.asarray(surface["componentId"], dtype=np.int32)
    lower_available = lower_surface >= 0
    upper_available = upper_surface >= 0
    lower_component[lower_available] = component[lower_surface[lower_available]]
    upper_component[upper_available] = component[upper_surface[upper_available]]

    macro_assignment = _macro_profile_assignment(
        key, profile_normal, macro_manifest, macro
    )
    macro_residual = macro_assignment["macroNormalResidualDegrees"]
    eligible = (
        lower_available
        & upper_available
        & (lower_component != upper_component)
        & (evidence >= settings.minimum_profile_evidence)
        & macro_assignment["macroTrusted"]
        & (
            macro_assignment["macroConfidence"]
            >= settings.minimum_macro_confidence
        )
        & (
            macro_residual
            <= settings.maximum_profile_to_macro_normal_degrees
        )
    )
    eligible_candidate = np.flatnonzero(eligible).astype(np.int32)
    first_component = np.minimum(
        lower_component[eligible_candidate], upper_component[eligible_candidate]
    ).astype(np.int64)
    second_component = np.maximum(
        lower_component[eligible_candidate], upper_component[eligible_candidate]
    ).astype(np.int64)
    pair_code = (
        (first_component.astype(np.uint64) << np.uint64(32))
        | second_component.astype(np.uint32).astype(np.uint64)
    )
    support_candidate = _deduplicate_pair_keys(
        eligible_candidate,
        pair_code,
        key[eligible_candidate],
        conservative[eligible_candidate],
        evidence[eligible_candidate],
    )
    support_first = np.minimum(
        lower_component[support_candidate], upper_component[support_candidate]
    ).astype(np.int64)
    support_second = np.maximum(
        lower_component[support_candidate], upper_component[support_candidate]
    ).astype(np.int64)
    support_pair = (
        (support_first.astype(np.uint64) << np.uint64(32))
        | support_second.astype(np.uint32).astype(np.uint64)
    )
    order = np.argsort(support_pair, kind="stable")
    ordered_pair = support_pair[order]
    unique_pair, start, count = np.unique(
        ordered_pair, return_index=True, return_counts=True
    )
    surface_component_size = np.bincount(component)
    records: list[dict[str, Any]] = []
    support_by_pair: dict[int, np.ndarray] = {}
    stride = float(sampling_stride_voxels)
    for code, low, size in zip(unique_pair, start, count):
        member = support_candidate[order[low : low + size]]
        first = int(code >> np.uint64(32))
        second = int(code & np.uint64(0xFFFFFFFF))
        orientation = lower_component[member] == first
        orientation_fraction = float(np.mean(orientation))
        orientation_consistency = max(
            orientation_fraction, 1.0 - orientation_fraction
        )
        extent = float(np.linalg.norm(np.ptp(midpoint[member], axis=0)) / stride)
        conservative_count = int(np.count_nonzero(conservative[member]))
        residual = np.asarray(macro_residual[member], dtype=np.float64)
        selected = bool(
            len(member) >= settings.minimum_support_profiles
            and conservative_count
            >= settings.minimum_conservative_support_profiles
            and extent >= settings.minimum_support_extent_sampling_steps
            and float(np.median(residual))
            <= settings.maximum_median_profile_to_macro_normal_degrees
        )
        records.append(
            {
                "pairCode": int(code),
                "faceComponentFirst": first,
                "faceComponentSecond": second,
                "firstFaceNodeCount": int(surface_component_size[first]),
                "secondFaceNodeCount": int(surface_component_size[second]),
                "supportProfileCount": int(len(member)),
                "conservativeSupportProfileCount": conservative_count,
                "seedSupportProfileCount": int(
                    np.count_nonzero(np.asarray(bank["seedComponentId"])[member] >= 0)
                ),
                "supportExtentSamplingSteps": extent,
                "canonicalOrientationConsistency": orientation_consistency,
                "medianMacroNormalResidualDegrees": float(np.median(residual)),
                "p90MacroNormalResidualDegrees": float(np.percentile(residual, 90)),
                "medianLocalEvidence": float(np.median(evidence[member])),
                "medianThicknessVoxels": float(
                    np.median(np.asarray(bank["thicknessVoxels"])[member])
                ),
                "selected": selected,
            }
        )
        support_by_pair[int(code)] = member
    selected_records = [value for value in records if value["selected"]]
    selected_records.sort(
        key=lambda value: (
            -int(value["supportProfileCount"]),
            -int(value["conservativeSupportProfileCount"]),
            -float(value["supportExtentSamplingSteps"]),
            float(value["medianMacroNormalResidualDegrees"]),
            int(value["pairCode"]),
        )
    )
    selected_records = selected_records[: settings.maximum_selected_ribbons]
    selected_pair = {int(value["pairCode"]): rank for rank, value in enumerate(selected_records)}
    for record in records:
        record["ribbonLabel"] = selected_pair.get(int(record["pairCode"]), -1)
    records.sort(
        key=lambda value: (
            int(value["ribbonLabel"]) < 0,
            int(value["ribbonLabel"]) if int(value["ribbonLabel"]) >= 0 else 0,
            -int(value["supportProfileCount"]),
            int(value["pairCode"]),
        )
    )
    selected_support: list[np.ndarray] = []
    selected_label: list[np.ndarray] = []
    for record in selected_records:
        member = support_by_pair[int(record["pairCode"])]
        selected_support.append(member)
        selected_label.append(
            np.full(len(member), int(record["ribbonLabel"]), dtype=np.int32)
        )
    support_index = (
        np.concatenate(selected_support)
        if selected_support
        else np.empty(0, dtype=np.int32)
    )
    ribbon_label = (
        np.concatenate(selected_label)
        if selected_label
        else np.empty(0, dtype=np.int32)
    )
    endpoint_cost = np.asarray(endpoint_match["matchCost"], dtype=np.float32)
    discovery = {
        "supportProfileCandidateIndex": support_index.astype(np.int32),
        "supportRibbonLabel": ribbon_label,
        "lowerSurfaceNode": lower_surface[support_index].astype(np.int32),
        "upperSurfaceNode": upper_surface[support_index].astype(np.int32),
        "lowerFaceComponent": lower_component[support_index].astype(np.int32),
        "upperFaceComponent": upper_component[support_index].astype(np.int32),
        "lowerEndpointMatchCost": endpoint_cost[:candidate_count][support_index],
        "upperEndpointMatchCost": endpoint_cost[candidate_count:][support_index],
        "macroNormalResidualDegrees": macro_residual[support_index].astype(np.float32),
    }
    summary = {
        "profileCandidateCount": candidate_count,
        "lowerEndpointMatchedToSurfaceCount": int(np.count_nonzero(lower_available)),
        "upperEndpointMatchedToSurfaceCount": int(np.count_nonzero(upper_available)),
        "bothEndpointsMatchedToSurfaceCount": int(
            np.count_nonzero(lower_available & upper_available)
        ),
        "macroEligibleProfileCountBeforePairKeyDeduplication": int(
            len(eligible_candidate)
        ),
        "macroEligibleProfileCount": int(len(support_candidate)),
        "supportedFacePairCount": int(len(records)),
        "selectedRibbonCount": int(len(selected_records)),
        "selectedSupportProfileCount": int(len(support_index)),
        "selectedSupportMacroResidualDegrees": _percentile_record(
            macro_residual[support_index]
        ),
        "selectedSupportLocalEvidence": _percentile_record(evidence[support_index]),
    }
    return discovery, records, summary


def _group_members(values: np.ndarray) -> dict[int, np.ndarray]:
    labels = np.asarray(values, dtype=np.int32)
    order = np.argsort(labels, kind="stable")
    unique, start, count = np.unique(labels[order], return_index=True, return_counts=True)
    return {
        int(value): order[low : low + size]
        for value, low, size in zip(unique, start, count)
    }


def _expanded_ribbon_faces(
    surface: Mapping[str, np.ndarray],
    selected_records: list[dict[str, Any]],
) -> dict[str, np.ndarray]:
    component = np.asarray(surface["componentId"], dtype=np.int32)
    members = _group_members(component)
    edge_first = np.asarray(surface["edgeFirstNode"], dtype=np.int32)
    edge_second = np.asarray(surface["edgeSecondNode"], dtype=np.int32)
    same_component = component[edge_first] == component[edge_second]
    edge_ids = np.flatnonzero(same_component)
    edge_members = _group_members(component[edge_first[edge_ids]])
    source_parts: list[np.ndarray] = []
    label_parts: list[np.ndarray] = []
    side_parts: list[np.ndarray] = []
    expanded_first: list[np.ndarray] = []
    expanded_second: list[np.ndarray] = []
    original_to_expanded = np.full(len(component), -1, dtype=np.int32)
    offset = 0
    for record in selected_records:
        label = int(record["ribbonLabel"])
        for side, component_id in enumerate(
            (
                int(record["faceComponentFirst"]),
                int(record["faceComponentSecond"]),
            )
        ):
            node = members[component_id].astype(np.int32)
            expanded = offset + np.arange(len(node), dtype=np.int32)
            original_to_expanded[node] = expanded
            source_parts.append(node)
            label_parts.append(np.full(len(node), label, dtype=np.int32))
            side_parts.append(np.full(len(node), side, dtype=np.uint8))
            component_edge_local = edge_members.get(component_id)
            if component_edge_local is not None:
                original_edge = edge_ids[component_edge_local]
                expanded_first.append(original_to_expanded[edge_first[original_edge]])
                expanded_second.append(original_to_expanded[edge_second[original_edge]])
            original_to_expanded[node] = -1
            offset += len(node)
    return {
        "sourceSurfaceNode": np.concatenate(source_parts).astype(np.int32),
        "ribbonLabel": np.concatenate(label_parts).astype(np.int32),
        "boundarySide": np.concatenate(side_parts).astype(np.uint8),
        "edgeFirstNode": (
            np.concatenate(expanded_first).astype(np.int32)
            if expanded_first
            else np.empty(0, dtype=np.int32)
        ),
        "edgeSecondNode": (
            np.concatenate(expanded_second).astype(np.int32)
            if expanded_second
            else np.empty(0, dtype=np.int32)
        ),
    }


def run_laminar_ribbon_catalog(
    paired_bank_root: str | Path,
    paired_growth_root: str | Path,
    material_surface_root: str | Path,
    macro_orientation_root: str | Path,
    output_root: str | Path,
    *,
    settings: LaminarRibbonSettings | None = None,
    mid_surface_settings: PhysicalMidSurfaceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build physical mid-surfaces from macro-aligned paired boundary components."""

    started = time.monotonic()
    resolved = settings or LaminarRibbonSettings()
    mid_settings = mid_surface_settings or PhysicalMidSurfaceSettings(
        adopt_contextual_profiles=False,
        maximum_geodesic_profile_distance_sampling_steps=8.0,
        maximum_one_sided_proxy_profile_distance_sampling_steps=8.0,
    )
    bank_path = _resolve(paired_bank_root, PAIRED_SURFACE_BANK_STEM)
    growth_path = _resolve(paired_growth_root, PAIRED_SURFACE_GROWTH_STEM)
    surface_path = _resolve(material_surface_root, MATERIAL_SURFACE_GRAPH_STEM)
    macro_path = _resolve(macro_orientation_root, MACRO_ORIENTATION_STEM)
    bank_manifest, bank = _load(bank_path)
    growth_manifest, growth = _load(growth_path)
    surface_manifest, surface = _load(surface_path)
    macro_manifest, macro = _load(macro_path)
    if bank_manifest.get("schema") != PAIRED_SURFACE_BANK_SCHEMA:
        raise ValueError("laminar ribbons require a paired profile bank")
    if growth_manifest.get("schema") != PAIRED_SURFACE_GROWTH_SCHEMA:
        raise ValueError("laminar ribbons require paired continuity edges")
    if surface_manifest.get("schema") != MATERIAL_SURFACE_GRAPH_SCHEMA:
        raise ValueError("laminar ribbons require the unlabelled material surface graph")
    if macro_manifest.get("schema") != MACRO_ORIENTATION_SCHEMA:
        raise ValueError("laminar ribbons require a macro orientation field")
    if (
        growth_manifest["identity"]["candidateBank"]["manifestSha256"]
        != sha256_file(bank_path)
    ):
        raise ValueError("paired continuity graph belongs to another profile bank")
    if (
        surface_manifest["identity"]["macroOrientation"]["manifestSha256"]
        != sha256_file(macro_path)
    ):
        raise ValueError("material surface graph belongs to another macro field")
    interface_path = Path(
        surface_manifest["identity"]["interfaces"]["manifestPath"]
    ).resolve()
    interface_manifest, interfaces = _load(interface_path)
    if interface_manifest.get("schema") != MATERIAL_INTERFACE_SCHEMA:
        raise ValueError("material surface graph references an invalid interface field")
    if (
        bank_manifest.get("source") != interface_manifest.get("source")
        or bank_manifest.get("geometry") != interface_manifest.get("geometry")
    ):
        raise ValueError("paired profiles and dense interfaces do not share geometry")
    stride = int(
        interface_manifest["identity"]["settings"]["sampling_stride_voxels"]
    )
    identity: dict[str, Any] = {
        "schema": LAMINAR_RIBBON_SCHEMA,
        "version": LAMINAR_RIBBON_VERSION,
        "pairedBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "pairedContinuity": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth_manifest["data"]["sha256"],
        },
        "materialSurface": {
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface_manifest["data"]["sha256"],
        },
        "macroOrientation": {
            "manifestPath": str(macro_path),
            "manifestSha256": sha256_file(macro_path),
            "dataSha256": macro_manifest["data"]["sha256"],
        },
        "interfaces": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interface_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "midSurfaceSettings": mid_settings.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.json"
    data_path = output / f"{PHYSICAL_MID_SURFACE_STEM}.npz"
    preview_path = output / "laminar-ribbon-cross-sections.png"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    discovery, ribbon_records, discovery_summary = discover_laminar_ribbons(
        bank,
        surface,
        interfaces,
        macro_manifest,
        macro,
        processing_start_xyz=np.asarray(
            interface_manifest["geometry"]["processingVoxelBounds"]["startXYZ"]
        ),
        source_origin_xyz=np.asarray(
            interface_manifest["source"]["sourceOriginXYZ"]
        ),
        processing_shape_sampling_xyz=tuple(
            int(value)
            for value in interface_manifest["geometry"][
                "processingShapeSamplingXYZ"
            ]
        ),
        sampling_stride_voxels=stride,
        settings=resolved,
    )
    selected_records = [
        value for value in ribbon_records if int(value["ribbonLabel"]) >= 0
    ]
    selected_records.sort(key=lambda value: int(value["ribbonLabel"]))
    if not selected_records:
        raise RuntimeError("no face pair passed the laminar ribbon evidence gates")

    expanded = _expanded_ribbon_faces(surface, selected_records)
    source_surface_node = expanded["sourceSurfaceNode"]
    position = np.asarray(surface["positionXYZ"], dtype=np.float64)[
        source_surface_node
    ]
    signed_normal = np.asarray(surface["signedNormalXYZ"], dtype=np.float64)[
        source_surface_node
    ]
    local_evidence = np.asarray(surface["localEvidenceScore"], dtype=np.float32)[
        source_surface_node
    ]
    face_component = np.asarray(surface["componentId"], dtype=np.int32)[
        source_surface_node
    ]
    face_label = expanded["ribbonLabel"]
    face_side = expanded["boundarySide"]

    profile_candidate = discovery["supportProfileCandidateIndex"]
    profile_label = discovery["supportRibbonLabel"]
    profile_lower_component = discovery["lowerFaceComponent"]
    record_first = np.asarray(
        [value["faceComponentFirst"] for value in selected_records], dtype=np.int32
    )
    keep_orientation = profile_lower_component == record_first[profile_label]
    raw_lower = np.asarray(bank["boundaryLowerXYZ"], dtype=np.float64)[
        profile_candidate
    ]
    raw_upper = np.asarray(bank["boundaryUpperXYZ"], dtype=np.float64)[
        profile_candidate
    ]
    raw_normal = np.asarray(bank["normalXYZ"], dtype=np.float64)[profile_candidate]
    profile_lower = np.where(keep_orientation[:, None], raw_lower, raw_upper)
    profile_upper = np.where(keep_orientation[:, None], raw_upper, raw_lower)
    profile_normal = raw_normal * np.where(keep_orientation, 1.0, -1.0)[:, None]
    profile_midpoint = np.asarray(bank["midpointXYZ"], dtype=np.float64)[
        profile_candidate
    ]
    profile_thickness = np.asarray(bank["thicknessVoxels"], dtype=np.float64)[
        profile_candidate
    ]
    thickness_by_label = {
        label: profile_thickness[profile_label == label]
        for label in range(len(selected_records))
    }
    direct_prior = local_profile_thickness_prior(
        position,
        signed_normal,
        face_label,
        face_side,
        profile_lower,
        profile_upper,
        profile_normal,
        profile_thickness,
        profile_label,
        profile_candidate,
        sampling_stride_voxels=stride,
        settings=mid_settings,
    )
    prior = propagate_profile_thickness_prior(
        position,
        signed_normal,
        face_label,
        face_side,
        expanded["edgeFirstNode"],
        expanded["edgeSecondNode"],
        direct_prior,
        sampling_stride_voxels=stride,
        settings=mid_settings,
    )
    pairing = pair_physical_boundary_faces(
        position,
        signed_normal,
        face_label,
        face_side,
        thickness_by_label,
        sampling_stride_voxels=stride,
        settings=mid_settings,
        local_thickness_prior=prior["expectedThicknessVoxels"],
    )
    lower_node = np.asarray(pairing["lowerNode"], dtype=np.int32)
    upper_node = np.asarray(pairing["upperNode"], dtype=np.int32)
    dense_midpoint = 0.5 * (position[lower_node] + position[upper_node])
    dense_normal = _normalized(
        signed_normal[lower_node] - signed_normal[upper_node]
    )
    dense_label = face_label[lower_node]
    if not np.array_equal(dense_label, face_label[upper_node]):
        raise RuntimeError("dense ribbon correspondence crossed a face-pair identity")

    midpoint_by_face_node = np.full(len(position), -1, dtype=np.int32)
    pair_index = np.arange(len(lower_node), dtype=np.int32)
    midpoint_by_face_node[lower_node] = pair_index
    if len(upper_node):
        pair_cost = np.asarray(pairing["pairCost"], dtype=np.float64)
        order = np.lexsort((pair_index, pair_cost, upper_node))
        ordered_upper = upper_node[order]
        first_for_upper = np.concatenate(
            (np.ones(1, dtype=bool), ordered_upper[1:] != ordered_upper[:-1])
        )
        midpoint_by_face_node[ordered_upper[first_for_upper]] = pair_index[
            order[first_for_upper]
        ]
    (
        surface_edge_first,
        surface_edge_second,
        surface_edge_support,
        surface_edge_score,
        surface_edge_summary,
    ) = _midpoint_edges(
        midpoint_by_face_node,
        expanded["edgeFirstNode"],
        expanded["edgeSecondNode"],
        face_side,
        dense_midpoint,
        dense_normal,
        dense_label,
        sampling_stride_voxels=stride,
        settings=mid_settings,
    )

    candidate_to_profile = np.full(len(bank["midpointXYZ"]), -1, dtype=np.int32)
    candidate_to_profile[profile_candidate] = np.arange(
        len(profile_candidate), dtype=np.int32
    )
    candidate_ribbon = np.full(len(bank["midpointXYZ"]), -1, dtype=np.int32)
    candidate_ribbon[profile_candidate] = profile_label
    growth_first = np.asarray(growth["edgeFirstCandidate"], dtype=np.int32)
    growth_second = np.asarray(growth["edgeSecondCandidate"], dtype=np.int32)
    profile_edge_valid = (
        (candidate_ribbon[growth_first] >= 0)
        & (candidate_ribbon[growth_first] == candidate_ribbon[growth_second])
    )
    profile_edge_first = candidate_to_profile[growth_first[profile_edge_valid]]
    profile_edge_second = candidate_to_profile[growth_second[profile_edge_valid]]
    profile_edge_score = np.asarray(growth["edgeAffinity"], dtype=np.float32)[
        profile_edge_valid
    ]
    lower_profile_candidate = prior["profileCandidateIndex"][lower_node]
    upper_profile_candidate = prior["profileCandidateIndex"][upper_node]
    (
        attachment_profile,
        attachment_dense,
        attachment_score,
    ) = _profile_attachment_edges(
        dense_midpoint,
        dense_normal,
        dense_label,
        lower_profile_candidate,
        upper_profile_candidate,
        candidate_to_profile,
        profile_midpoint,
        profile_normal,
        profile_label,
        sampling_stride_voxels=stride,
        settings=mid_settings,
    )
    profile_count = len(profile_midpoint)
    midpoint = np.concatenate((profile_midpoint, dense_midpoint), axis=0)
    normal = np.concatenate((profile_normal, dense_normal), axis=0)
    label = np.concatenate((profile_label, dense_label)).astype(np.int32)
    edge_first = np.concatenate(
        (
            profile_edge_first.astype(np.int32),
            profile_count + surface_edge_first,
            attachment_profile,
        )
    )
    edge_second = np.concatenate(
        (
            profile_edge_second.astype(np.int32),
            profile_count + surface_edge_second,
            profile_count + attachment_dense,
        )
    )
    edge_score = np.concatenate(
        (profile_edge_score, surface_edge_score, attachment_score)
    ).astype(np.float32)
    edge_support = np.concatenate(
        (
            np.zeros(len(profile_edge_first), dtype=np.uint8),
            surface_edge_support,
            np.full(len(attachment_profile), 3, dtype=np.uint8),
        )
    )
    edge_kind = np.concatenate(
        (
            np.zeros(len(profile_edge_first), dtype=np.uint8),
            np.ones(len(surface_edge_first), dtype=np.uint8),
            np.full(len(attachment_profile), 2, dtype=np.uint8),
        )
    )
    component, component_size = _components(
        len(midpoint), edge_first, edge_second
    )
    if np.any(label[edge_first] != label[edge_second]):
        raise RuntimeError("laminar ribbon graph crossed a ribbon identity")

    profile_float = np.full(profile_count, np.nan, dtype=np.float32)
    profile_int = np.full(profile_count, -1, dtype=np.int32)
    dense_expected = 0.5 * (
        prior["expectedThicknessVoxels"][lower_node]
        + prior["expectedThicknessVoxels"][upper_node]
    )
    profile_evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)[
        profile_candidate
    ]
    arrays = {
        "midpointXYZ": midpoint.astype(np.float32),
        "normalXYZ": normal.astype(np.float32),
        "boundaryLowerXYZ": np.concatenate(
            (profile_lower, position[lower_node]), axis=0
        ).astype(np.float32),
        "boundaryUpperXYZ": np.concatenate(
            (profile_upper, position[upper_node]), axis=0
        ).astype(np.float32),
        "nodeKind": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.uint8),
                np.ones(len(dense_midpoint), dtype=np.uint8),
            )
        ),
        "profileAdopted": np.zeros(len(midpoint), dtype=bool),
        "profileEndpointSupportCount": np.concatenate(
            (
                np.full(profile_count, 2, dtype=np.uint8),
                np.zeros(len(dense_midpoint), dtype=np.uint8),
            )
        ),
        "profileCanonicalOrientation": np.concatenate(
            (
                np.where(keep_orientation, 1, -1).astype(np.int8),
                np.zeros(len(dense_midpoint), dtype=np.int8),
            )
        ),
        "profileCandidateIndex": np.concatenate(
            (profile_candidate, np.full(len(dense_midpoint), -1, dtype=np.int32))
        ),
        "lowerSurfaceNode": np.concatenate(
            (profile_int, source_surface_node[lower_node])
        ).astype(np.int32),
        "upperSurfaceNode": np.concatenate(
            (profile_int, source_surface_node[upper_node])
        ).astype(np.int32),
        "lowerFaceComponentId": np.concatenate(
            (record_first[profile_label], face_component[lower_node])
        ).astype(np.int32),
        "upperFaceComponentId": np.concatenate(
            (
                np.asarray(
                    [value["faceComponentSecond"] for value in selected_records],
                    dtype=np.int32,
                )[profile_label],
                face_component[upper_node],
            )
        ).astype(np.int32),
        "physicalSheetLabel": label,
        "componentId": component.astype(np.int32),
        "thicknessVoxels": np.concatenate(
            (profile_thickness.astype(np.float32), pairing["thicknessVoxels"])
        ),
        "pairCost": np.concatenate((profile_float, pairing["pairCost"])),
        "thicknessResidualVoxels": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                (pairing["thicknessVoxels"] - dense_expected).astype(np.float32),
            )
        ),
        "tangentResidualSamplingSteps": np.concatenate(
            (profile_float, pairing["tangentResidualSamplingSteps"])
        ),
        "lowerDirectionDegrees": np.concatenate(
            (profile_float, pairing["lowerDirectionDegrees"])
        ),
        "upperDirectionDegrees": np.concatenate(
            (profile_float, pairing["upperDirectionDegrees"])
        ),
        "opposingNormalDegrees": np.concatenate(
            (profile_float, pairing["opposingNormalDegrees"])
        ),
        "lowerLocalEvidenceScore": np.concatenate(
            (profile_evidence, local_evidence[lower_node])
        ),
        "upperLocalEvidenceScore": np.concatenate(
            (profile_evidence, local_evidence[upper_node])
        ),
        "lowerExpectedThicknessVoxels": np.concatenate(
            (profile_thickness, prior["expectedThicknessVoxels"][lower_node])
        ).astype(np.float32),
        "upperExpectedThicknessVoxels": np.concatenate(
            (profile_thickness, prior["expectedThicknessVoxels"][upper_node])
        ).astype(np.float32),
        "lowerProfileDistanceSamplingSteps": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                prior["profileDistanceSamplingSteps"][lower_node],
            )
        ),
        "upperProfileDistanceSamplingSteps": np.concatenate(
            (
                np.zeros(profile_count, dtype=np.float32),
                prior["profileDistanceSamplingSteps"][upper_node],
            )
        ),
        "lowerProfileCandidateIndex": np.concatenate(
            (profile_candidate, lower_profile_candidate)
        ).astype(np.int32),
        "upperProfileCandidateIndex": np.concatenate(
            (profile_candidate, upper_profile_candidate)
        ).astype(np.int32),
        "oneSidedSourceSurfaceNode": np.full(
            len(midpoint), -1, dtype=np.int32
        ),
        "oneSidedPhysicalBoundarySide": np.full(
            len(midpoint), 255, dtype=np.uint8
        ),
        "edgeFirstNode": edge_first.astype(np.int32),
        "edgeSecondNode": edge_second.astype(np.int32),
        "edgeBoundarySupportMask": edge_support,
        "edgeKind": edge_kind,
        "edgeScore": edge_score,
    }
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"].get("metadataPath")
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    write_material_surface_cross_sections(
        source,
        owned,
        midpoint,
        component,
        component_size,
        preview_path,
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=MaterialSurfaceGraphSettings(
            minimum_component_samples_for_preview=8,
            maximum_preview_components=128,
        ),
    )
    geometry = dict(bank_manifest["geometry"])
    geometry["samplingStrideVoxels"] = stride
    payload: dict[str, Any] = {
        "schema": PHYSICAL_MID_SURFACE_SCHEMA,
        "version": PHYSICAL_MID_SURFACE_VERSION,
        "constructionSchema": LAMINAR_RIBBON_SCHEMA,
        "constructionVersion": LAMINAR_RIBBON_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": geometry,
        "calibration": bank_manifest["calibration"],
        "counts": {
            **discovery_summary,
            "expandedBoundaryFaceNodeCount": int(len(position)),
            "faceNodeWithDirectThicknessPriorCount": int(
                np.count_nonzero(direct_prior["profileCandidateIndex"] >= 0)
            ),
            "faceNodeWithPropagatedThicknessPriorCount": int(
                np.count_nonzero(prior["profileCandidateIndex"] >= 0)
            ),
            "denseBoundaryPairCount": int(len(dense_midpoint)),
            "midSurfaceNodeCount": int(len(midpoint)),
            "midSurfaceEdgeCount": int(len(edge_first)),
            "midSurfaceComponentCount": int(len(component_size)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_size >= 128)),
            "largestComponentSizes": component_size[:32].astype(int).tolist(),
        },
        "distributions": {
            "ribbonSupportProfileCount": _percentile_record(
                np.asarray(
                    [value["supportProfileCount"] for value in selected_records]
                )
            ),
            "ribbonSupportExtentSamplingSteps": _percentile_record(
                np.asarray(
                    [
                        value["supportExtentSamplingSteps"]
                        for value in selected_records
                    ]
                )
            ),
            "densePairThicknessVoxels": _percentile_record(
                pairing["thicknessVoxels"]
            ),
            "densePairTangentResidualSamplingSteps": _percentile_record(
                pairing["tangentResidualSamplingSteps"]
            ),
        },
        "ribbons": ribbon_records,
        "pairingCensus": pairing["census"],
        "surfaceEdgeProjection": surface_edge_summary,
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "primitive": (
                "unordered pair of dense signed boundary components supported "
                "by repeated air-papyrus-air profiles"
            ),
            "macroGuard": (
                "each support profile must agree with the independent generic "
                "cross-layer laminar orientation tensor"
            ),
            "identityPolicy": (
                "historical grown sheet labels are ignored; each supported "
                "face pair remains an independent physical ribbon hypothesis"
            ),
            "denseReconstruction": (
                "locally thickness-conditioned reciprocal correspondence between "
                "the two observed boundary components"
            ),
            "oneSidedInference": False,
            "acusRole": "none; fiber evidence is reserved for later ply analysis",
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
