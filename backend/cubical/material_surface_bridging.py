from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import VolumeSource, VoxelBounds, atomic_json, canonical_json_hash, sha256_file
from .macro_orientation import (
    MACRO_ORIENTATION_SCHEMA,
    MACRO_ORIENTATION_STEM,
    sample_orientation_field,
)
from .material_interface import MATERIAL_INTERFACE_SCHEMA, MATERIAL_INTERFACE_STEM
from .material_surface_graph import (
    MaterialSurfaceGraphSettings,
    _load_npz,
    _percentiles,
    _tangent_column_records,
    write_material_surface_cross_sections,
)
from .material_surface_growth import (
    MATERIAL_SURFACE_GROWTH_SCHEMA,
    MATERIAL_SURFACE_GROWTH_STEM,
    _build_neighbor_interface_matrix,
    _neighbor_offsets,
    _support_geometry,
)


MATERIAL_SURFACE_BRIDGING_SCHEMA = "pareidolia.material-interface-boundary-bridging"
MATERIAL_SURFACE_BRIDGING_VERSION = 1
MATERIAL_SURFACE_BRIDGING_STEM = "material-interface-boundary-bridging-v1"


@dataclass(frozen=True, slots=True)
class MaterialSurfaceBridgingSettings:
    minimum_local_evidence: float = 0.1
    neighbor_radius_sampling_steps: float = math.sqrt(3.0)
    minimum_anchor_samples_per_component: int = 2
    maximum_support_angular_gap_degrees: float = 180.0
    minimum_tangent_rank_ratio: float = 0.08
    maximum_candidate_plane_residual_sampling_steps: float = 1.15
    maximum_support_height_standard_deviation_sampling_steps: float = 0.8
    maximum_signed_normal_disagreement_degrees: float = 70.0
    minimum_component_opposition_degrees: float = 120.0
    candidate_cluster_radius_sampling_steps: float = math.sqrt(3.0)
    minimum_bundle_candidate_count: int = 3
    minimum_bundle_anchor_count_per_component: int = 3
    minimum_bundle_tangent_column_count: int = 2
    minimum_bundle_span_sampling_steps: float = 2.0

    def __post_init__(self) -> None:
        if not 0.0 <= self.minimum_local_evidence <= 1.0:
            raise ValueError("bridge evidence threshold must lie in [0, 1]")
        if self.neighbor_radius_sampling_steps < 1.0:
            raise ValueError("bridge neighbor radius must be at least one sample")
        if self.minimum_anchor_samples_per_component < 2:
            raise ValueError("a bridge candidate requires multiple anchors on each side")
        if not 0.0 < self.maximum_support_angular_gap_degrees <= 180.0:
            raise ValueError("bridge support must enclose the candidate")
        if not 0.0 <= self.minimum_tangent_rank_ratio <= 1.0:
            raise ValueError("bridge tangent rank must lie in [0, 1]")
        positive = (
            self.maximum_candidate_plane_residual_sampling_steps,
            self.maximum_support_height_standard_deviation_sampling_steps,
            self.candidate_cluster_radius_sampling_steps,
            self.minimum_bundle_span_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("bridge geometry thresholds must be finite and positive")
        if not 0.0 < self.maximum_signed_normal_disagreement_degrees < 90.0:
            raise ValueError("bridge signed-normal cap must lie in (0, 90)")
        if not 90.0 <= self.minimum_component_opposition_degrees <= 180.0:
            raise ValueError("bridge anchors must oppose across at least 90 degrees")
        integer_positive = (
            self.minimum_bundle_candidate_count,
            self.minimum_bundle_anchor_count_per_component,
            self.minimum_bundle_tangent_column_count,
        )
        if any(value < 1 for value in integer_positive):
            raise ValueError("bridge bundle counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _component_opposition_degrees(
    candidate_position: np.ndarray,
    support_position: np.ndarray,
    support_component: np.ndarray,
    first_component: int,
    second_component: int,
    plane_normal: np.ndarray,
    *,
    sampling_stride_voxels: int,
) -> float:
    delta = (
        np.asarray(support_position, dtype=np.float64)
        - np.asarray(candidate_position, dtype=np.float64)[None, :]
    ) / float(sampling_stride_voxels)
    delta -= np.einsum("ij,j->i", delta, plane_normal)[:, None] * plane_normal[None, :]
    first_direction = np.mean(delta[support_component == first_component], axis=0)
    second_direction = np.mean(delta[support_component == second_component], axis=0)
    first_length = float(np.linalg.norm(first_direction))
    second_length = float(np.linalg.norm(second_direction))
    if min(first_length, second_length) < 0.2:
        return 0.0
    cosine = float(
        np.dot(first_direction, second_direction) / (first_length * second_length)
    )
    return math.degrees(math.acos(float(np.clip(cosine, -1.0, 1.0))))


def _candidate_score(
    evidence: float,
    geometry: Mapping[str, Any],
    opposition_degrees: float,
    settings: MaterialSurfaceBridgingSettings,
) -> float:
    gap = float(geometry["maximumSupportAngularGapDegrees"])
    rank = float(geometry["tangentRankRatio"])
    residual = float(geometry["candidatePlaneResidualSamplingSteps"])
    signed = float(geometry["signedNormalDisagreementDegrees"])
    enclosure = max(0.0, 1.0 - gap / 360.0)
    plane_quality = math.exp(
        -0.5
        * (
            residual
            / max(settings.maximum_candidate_plane_residual_sampling_steps, 1.0e-9)
        )
        ** 2
    )
    signed_quality = max(0.0, math.cos(math.radians(signed)))
    opposition_quality = max(0.0, (opposition_degrees - 90.0) / 90.0)
    return float(
        max(evidence, 0.05)
        * enclosure
        * math.sqrt(max(rank, 0.0))
        * plane_quality
        * signed_quality
        * opposition_quality
    )


def _cluster_pair_candidates(
    records: list[dict[str, Any]],
    key: np.ndarray,
    *,
    radius: float,
) -> list[list[dict[str, Any]]]:
    by_key = {
        tuple(int(value) for value in key[int(record["interfaceIndex"])]): index
        for index, record in enumerate(records)
    }
    offsets = _neighbor_offsets(radius)
    unseen = set(range(len(records)))
    clusters: list[list[dict[str, Any]]] = []
    while unseen:
        start = unseen.pop()
        stack = [start]
        cluster = [records[start]]
        while stack:
            current = stack.pop()
            current_key = key[int(records[current]["interfaceIndex"])]
            for offset in offsets:
                adjacent = by_key.get(
                    tuple(int(value) for value in current_key + offset)
                )
                if adjacent is None or adjacent not in unseen:
                    continue
                unseen.remove(adjacent)
                stack.append(adjacent)
                cluster.append(records[adjacent])
        clusters.append(cluster)
    return clusters


def _find(parent: np.ndarray, value: int) -> int:
    root = int(value)
    while int(parent[root]) != root:
        root = int(parent[root])
    while int(parent[value]) != value:
        following = int(parent[value])
        parent[value] = root
        value = following
    return root


def _merge_physical_face_identity(
    first_label: int,
    first_side: int,
    second_label: int,
    second_side: int,
) -> tuple[bool, int, int]:
    """Combine optional immutable signed-face identities without crossing faces."""

    first = int(first_label)
    second = int(second_label)
    first_boundary = int(first_side)
    second_boundary = int(second_side)
    if (first >= 0) != (first_boundary <= 1) or (second >= 0) != (
        second_boundary <= 1
    ):
        raise ValueError("physical sheet and boundary-side identities disagree")
    if first >= 0 and second >= 0 and (
        first != second or first_boundary != second_boundary
    ):
        return False, -1, 255
    if first >= 0:
        return True, first, first_boundary
    if second >= 0:
        return True, second, second_boundary
    return True, -1, 255


def _compatible_union(
    first_state: dict[int, tuple[float, float]],
    second_state: dict[int, tuple[float, float]],
    candidate: np.ndarray,
    tangent_column: np.ndarray,
    normal_depth: np.ndarray,
    *,
    maximum_depth_range: float,
) -> tuple[bool, dict[int, tuple[float, float]]]:
    if len(first_state) >= len(second_state):
        combined = dict(first_state)
        following = second_state
    else:
        combined = dict(second_state)
        following = first_state
    for column_id, interval in following.items():
        existing = combined.get(column_id)
        if existing is None:
            combined[column_id] = interval
            continue
        low = min(existing[0], interval[0])
        high = max(existing[1], interval[1])
        if high - low > maximum_depth_range:
            return False, {}
        combined[column_id] = (low, high)
    for interface_index in candidate:
        column_id = int(tangent_column[interface_index])
        depth = float(normal_depth[interface_index])
        existing = combined.get(column_id)
        low = depth if existing is None else min(existing[0], depth)
        high = depth if existing is None else max(existing[1], depth)
        if high - low > maximum_depth_range:
            return False, {}
        combined[column_id] = (low, high)
    return True, combined


def run_material_surface_bridging(
    interface_root: str | Path,
    macro_root: str | Path,
    growth_root: str | Path,
    output_path: str | Path,
    *,
    settings: MaterialSurfaceBridgingSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve(interface_root, MATERIAL_INTERFACE_STEM)
    macro_path = _resolve(macro_root, MACRO_ORIENTATION_STEM)
    growth_path = _resolve(growth_root, MATERIAL_SURFACE_GROWTH_STEM)
    interfaces = json.loads(interface_path.read_text())
    macro = json.loads(macro_path.read_text())
    growth = json.loads(growth_path.read_text())
    if interfaces.get("schema") != MATERIAL_INTERFACE_SCHEMA or interfaces.get("state") != "complete":
        raise ValueError("boundary bridging requires complete material interfaces")
    if macro.get("schema") != MACRO_ORIENTATION_SCHEMA or macro.get("state") != "complete":
        raise ValueError("boundary bridging requires a complete macro orientation field")
    if growth.get("schema") != MATERIAL_SURFACE_GROWTH_SCHEMA or growth.get("state") != "complete":
        raise ValueError("boundary bridging requires complete interior growth")
    if growth["identity"]["interfaces"]["manifestSha256"] != sha256_file(interface_path):
        raise ValueError("interior growth was not derived from the supplied interfaces")
    if growth["identity"]["macroOrientation"]["manifestSha256"] != sha256_file(macro_path):
        raise ValueError("interior growth was not derived from the supplied macro field")

    resolved = settings or MaterialSurfaceBridgingSettings()
    seed_graph_identity = growth["identity"]["seedGraph"]
    seed_graph_path = Path(seed_graph_identity["manifestPath"]).resolve()
    if sha256_file(seed_graph_path) != seed_graph_identity["manifestSha256"]:
        raise ValueError("interior-growth seed graph changed after growth")
    seed_graph = json.loads(seed_graph_path.read_text())
    graph_settings = MaterialSurfaceGraphSettings(
        **seed_graph["identity"]["settings"]
    )
    identity: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_BRIDGING_SCHEMA,
        "version": MATERIAL_SURFACE_BRIDGING_VERSION,
        "interfaces": {
            "manifestPath": str(interface_path),
            "manifestSha256": sha256_file(interface_path),
            "dataSha256": interfaces["data"]["sha256"],
        },
        "macroOrientation": {
            "manifestPath": str(macro_path),
            "manifestSha256": sha256_file(macro_path),
            "dataSha256": macro["data"]["sha256"],
        },
        "interiorGrowth": {
            "manifestPath": str(growth_path),
            "manifestSha256": sha256_file(growth_path),
            "dataSha256": growth["data"]["sha256"],
        },
        "seedGraph": dict(seed_graph_identity),
        "settings": resolved.record(),
        "inheritedStratumGuard": {
            "tangentColumnWidthSamplingSteps": graph_settings.tangent_column_width_sampling_steps,
            "maximumColumnDepthRangeSamplingSteps": graph_settings.maximum_column_depth_range_sampling_steps,
        },
        "implementationSha256": {
            "material_surface_bridging.py": sha256_file(Path(__file__)),
            "material_surface_growth.py": sha256_file(
                Path(__file__).with_name("material_surface_growth.py")
            ),
            "material_surface_graph.py": sha256_file(
                Path(__file__).with_name("material_surface_graph.py")
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_SURFACE_BRIDGING_STEM}.json"
    data_path = output / f"{MATERIAL_SURFACE_BRIDGING_STEM}.npz"
    preview_path = output / "material-surface-bridging-cross-sections.png"
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
    interface_arrays = _load_npz(interface_path, interfaces)
    macro_arrays = _load_npz(macro_path, macro)
    growth_arrays = _load_npz(growth_path, growth)
    position = np.asarray(interface_arrays["positionXYZ"], dtype=np.float64)
    signed_normal = np.asarray(interface_arrays["signedNormalXYZ"], dtype=np.float64)
    key = np.asarray(interface_arrays["processingKeyXYZ"], dtype=np.int64)
    evidence = np.asarray(interface_arrays["localEvidenceScore"], dtype=np.float64)
    orientation = sample_orientation_field(macro_arrays)
    sample_bin = orientation["sampleBinIndex"]
    sample_group = orientation["sampleGroupIndex"]
    macro_normal = orientation["sampleNormalXYZ"]
    macro_confidence = orientation["sampleOrientationConfidence"]
    macro_trusted = orientation["sampleTrusted"]
    orientation_source = orientation["sampleOrientationSource"]
    orientation_center = orientation["sampleGroupCenterXYZ"]
    raw_macro_angle = np.degrees(
        np.arccos(
            np.clip(np.abs(np.einsum("ij,ij->i", signed_normal, macro_normal)), 0.0, 1.0)
        )
    )
    growth_interface = np.asarray(growth_arrays["interfaceIndex"], dtype=np.int64)
    growth_component = np.asarray(growth_arrays["componentId"], dtype=np.int32)
    growth_sheet_normal = np.asarray(growth_arrays["sheetNormalXYZ"], dtype=np.float64)
    growth_round = np.asarray(growth_arrays["growthRound"], dtype=np.int16)
    component_count = int(np.max(growth_component)) + 1 if len(growth_component) else 0
    component_by_interface = np.full(len(position), -1, dtype=np.int32)
    component_by_interface[growth_interface] = growth_component
    sheet_normal = macro_normal.copy()
    sheet_normal[growth_interface] = growth_sheet_normal

    processing_shape = np.asarray(
        interfaces["geometry"]["processingShapeSamplingXYZ"], dtype=np.int64
    )
    grid = np.full(tuple(processing_shape[::-1]), -1, dtype=np.int32)
    if np.any(grid[key[:, 2], key[:, 1], key[:, 0]] >= 0):
        raise ValueError("material interfaces contain duplicate processing keys")
    grid[key[:, 2], key[:, 1], key[:, 0]] = np.arange(
        len(position), dtype=np.int32
    )
    stride = int(interfaces["identity"]["settings"]["sampling_stride_voxels"])
    column_record, normal_depth = _tangent_column_records(
        position,
        sample_group,
        orientation_center,
        macro_normal,
        stride=stride,
        width_sampling_steps=graph_settings.tangent_column_width_sampling_steps,
    )
    _, tangent_column = np.unique(column_record, axis=0, return_inverse=True)
    tangent_column = tangent_column.astype(np.int32)

    inactive = np.flatnonzero(
        (component_by_interface < 0)
        & (evidence >= resolved.minimum_local_evidence)
        & macro_trusted
        & (macro_confidence >= graph_settings.minimum_macro_confidence)
        & (
            raw_macro_angle
            <= graph_settings.maximum_raw_to_macro_normal_degrees
        )
    )
    offsets = _neighbor_offsets(resolved.neighbor_radius_sampling_steps)
    neighbor_interface = _build_neighbor_interface_matrix(
        inactive, key, grid, processing_shape, offsets
    )
    neighbor_component = np.full_like(neighbor_interface, -1)
    exists = neighbor_interface >= 0
    neighbor_component[exists] = component_by_interface[neighbor_interface[exists]]
    candidate_accounting = {
        "inactiveEvidenceCandidateCount": int(len(inactive)),
        "candidatesWithAnyActiveNeighbor": int(
            np.count_nonzero(np.any(neighbor_component >= 0, axis=1))
        ),
        "rejectedNotExactlyTwoSupportedComponents": 0,
        "rejectedOpenSupport": 0,
        "rejectedTangentRank": 0,
        "rejectedPlaneResidual": 0,
        "rejectedSupportHeightSpread": 0,
        "rejectedSignedNormal": 0,
        "rejectedComponentOpposition": 0,
        "qualifiedCandidateCount": 0,
    }
    pair_candidates: dict[tuple[int, int], list[dict[str, Any]]] = {}
    for row, interface_index in enumerate(inactive):
        active = neighbor_component[row] >= 0
        if not np.any(active):
            continue
        components, counts = np.unique(
            neighbor_component[row, active], return_counts=True
        )
        supported = components[
            counts >= resolved.minimum_anchor_samples_per_component
        ]
        if len(supported) != 2:
            candidate_accounting["rejectedNotExactlyTwoSupportedComponents"] += 1
            continue
        first_component, second_component = sorted(int(value) for value in supported)
        support_mask = (neighbor_component[row] == first_component) | (
            neighbor_component[row] == second_component
        )
        support = neighbor_interface[row, support_mask].astype(np.int64)
        support_component = neighbor_component[row, support_mask]
        geometry = _support_geometry(
            position[interface_index],
            signed_normal[interface_index],
            position[support],
            sheet_normal[support],
            signed_normal[support],
            sampling_stride_voxels=stride,
        )
        if (
            float(geometry["maximumSupportAngularGapDegrees"])
            > resolved.maximum_support_angular_gap_degrees + 1.0e-6
        ):
            candidate_accounting["rejectedOpenSupport"] += 1
            continue
        if float(geometry["tangentRankRatio"]) < resolved.minimum_tangent_rank_ratio:
            candidate_accounting["rejectedTangentRank"] += 1
            continue
        if (
            float(geometry["candidatePlaneResidualSamplingSteps"])
            > resolved.maximum_candidate_plane_residual_sampling_steps
        ):
            candidate_accounting["rejectedPlaneResidual"] += 1
            continue
        if (
            float(geometry["supportHeightStandardDeviationSamplingSteps"])
            > resolved.maximum_support_height_standard_deviation_sampling_steps
        ):
            candidate_accounting["rejectedSupportHeightSpread"] += 1
            continue
        if (
            float(geometry["signedNormalDisagreementDegrees"])
            > resolved.maximum_signed_normal_disagreement_degrees
        ):
            candidate_accounting["rejectedSignedNormal"] += 1
            continue
        opposition = _component_opposition_degrees(
            position[interface_index],
            position[support],
            support_component,
            first_component,
            second_component,
            np.asarray(geometry["sheetNormalXYZ"]),
            sampling_stride_voxels=stride,
        )
        if opposition < resolved.minimum_component_opposition_degrees:
            candidate_accounting["rejectedComponentOpposition"] += 1
            continue
        score = _candidate_score(
            float(evidence[interface_index]), geometry, opposition, resolved
        )
        pair_candidates.setdefault((first_component, second_component), []).append(
            {
                "interfaceIndex": int(interface_index),
                "support": support,
                "supportComponent": support_component.astype(np.int32),
                "geometry": geometry,
                "oppositionDegrees": opposition,
                "score": score,
            }
        )
        candidate_accounting["qualifiedCandidateCount"] += 1

    bundle_accounting = {
        "componentPairCount": int(len(pair_candidates)),
        "spatialClusterCount": 0,
        "rejectedSmallCandidateRun": 0,
        "rejectedInsufficientAnchors": 0,
        "rejectedInsufficientTangentColumns": 0,
        "rejectedShortSpan": 0,
        "qualifiedBundleCount": 0,
    }
    bundles: list[dict[str, Any]] = []
    for pair, records in pair_candidates.items():
        for cluster in _cluster_pair_candidates(
            records, key, radius=resolved.candidate_cluster_radius_sampling_steps
        ):
            bundle_accounting["spatialClusterCount"] += 1
            if len(cluster) < resolved.minimum_bundle_candidate_count:
                bundle_accounting["rejectedSmallCandidateRun"] += 1
                continue
            first_anchor: set[int] = set()
            second_anchor: set[int] = set()
            candidate = np.asarray(
                [int(record["interfaceIndex"]) for record in cluster],
                dtype=np.int64,
            )
            for record in cluster:
                for interface_index, component_id in zip(
                    record["support"], record["supportComponent"]
                ):
                    (first_anchor if int(component_id) == pair[0] else second_anchor).add(
                        int(interface_index)
                    )
            if (
                len(first_anchor) < resolved.minimum_bundle_anchor_count_per_component
                or len(second_anchor)
                < resolved.minimum_bundle_anchor_count_per_component
            ):
                bundle_accounting["rejectedInsufficientAnchors"] += 1
                continue
            column_count = len(np.unique(tangent_column[candidate]))
            if column_count < resolved.minimum_bundle_tangent_column_count:
                bundle_accounting["rejectedInsufficientTangentColumns"] += 1
                continue
            span = float(
                np.linalg.norm(np.ptp(position[candidate], axis=0)) / stride
            )
            if span < resolved.minimum_bundle_span_sampling_steps:
                bundle_accounting["rejectedShortSpan"] += 1
                continue
            mean_score = float(
                np.mean([float(record["score"]) for record in cluster])
            )
            score = (
                mean_score
                * math.log1p(len(cluster))
                * min(len(first_anchor), len(second_anchor), 8)
                / 8.0
                * min(span / 4.0, 1.0)
            )
            bundles.append(
                {
                    "componentFirst": pair[0],
                    "componentSecond": pair[1],
                    "records": cluster,
                    "candidate": candidate,
                    "firstAnchor": np.asarray(sorted(first_anchor), dtype=np.int64),
                    "secondAnchor": np.asarray(sorted(second_anchor), dtype=np.int64),
                    "candidateCount": len(cluster),
                    "tangentColumnCount": column_count,
                    "spanSamplingSteps": span,
                    "meanCandidateScore": mean_score,
                    "score": score,
                }
            )
            bundle_accounting["qualifiedBundleCount"] += 1
    bundles.sort(
        key=lambda value: (
            -float(value["score"]),
            int(value["componentFirst"]),
            int(value["componentSecond"]),
        )
    )

    parent = np.arange(component_count, dtype=np.int32)
    growth_physical_label = (
        np.asarray(growth_arrays["physicalSheetLabel"], dtype=np.int32)
        if "physicalSheetLabel" in growth_arrays
        else np.full(len(growth_interface), -1, dtype=np.int32)
    )
    growth_physical_side = (
        np.asarray(growth_arrays["physicalBoundarySide"], dtype=np.uint8)
        if "physicalBoundarySide" in growth_arrays
        else np.full(len(growth_interface), 255, dtype=np.uint8)
    )
    growth_physical_anchor = (
        np.asarray(growth_arrays["physicalSeedAnchor"], dtype=np.uint8)
        if "physicalSeedAnchor" in growth_arrays
        else np.zeros(len(growth_interface), dtype=np.uint8)
    )
    component_physical_label = np.full(component_count, -1, dtype=np.int32)
    component_physical_side = np.full(component_count, 255, dtype=np.uint8)
    component_physical_anchor_count = np.zeros(component_count, dtype=np.int32)
    if np.any((growth_physical_label >= 0) != (growth_physical_side <= 1)):
        raise ValueError(
            "interior-growth physical sheet and boundary-side identities disagree"
        )
    for component_id in range(component_count):
        member = growth_component == component_id
        identities = np.unique(
            2 * growth_physical_label[member & (growth_physical_label >= 0)]
            + growth_physical_side[member & (growth_physical_label >= 0)].astype(
                np.int32
            )
        )
        if len(identities) > 1:
            raise ValueError(
                "interior-growth component contains multiple physical boundary-face identities"
            )
        if len(identities) == 1:
            identity_value = int(identities[0])
            component_physical_label[component_id] = identity_value // 2
            component_physical_side[component_id] = identity_value % 2
        component_physical_anchor_count[component_id] = int(
            np.count_nonzero(growth_physical_anchor[member] == 1)
        )
    component_state: list[dict[int, tuple[float, float]] | None] = [
        {} for _ in range(component_count)
    ]
    for interface_index, component_id in zip(growth_interface, growth_component):
        state = component_state[int(component_id)]
        if state is None:
            raise RuntimeError("component column state is unavailable")
        column_id = int(tangent_column[interface_index])
        depth = float(normal_depth[interface_index])
        existing_interval = state.get(column_id)
        if existing_interval is None:
            state[column_id] = (depth, depth)
        else:
            low = min(existing_interval[0], depth)
            high = max(existing_interval[1], depth)
            if high - low > graph_settings.maximum_column_depth_range_sampling_steps:
                raise ValueError("interior-growth component violates its stratum guard")
            state[column_id] = (low, high)

    candidate_assigned = np.zeros(len(position), dtype=bool)
    accepted_candidate: list[int] = []
    accepted_candidate_component: list[int] = []
    accepted_candidate_normal: list[np.ndarray] = []
    accepted_candidate_bundle: list[int] = []
    accepted_candidate_score: list[float] = []
    accepted_candidate_opposition: list[float] = []
    accepted_candidate_angular_gap: list[float] = []
    accepted_candidate_plane_residual: list[float] = []
    bridge_edge_first_interface: list[int] = []
    bridge_edge_second_interface: list[int] = []
    bridge_edge_score: list[float] = []
    accepted_bundle_records: list[dict[str, Any]] = []
    merge_accounting = {
        "consideredBundleCount": int(len(bundles)),
        "rejectedReusedCandidate": 0,
        "rejectedStratumCollision": 0,
        "rejectedPhysicalSeedConflict": 0,
        "acceptedMergeBundleCount": 0,
        "acceptedInternalBundleCount": 0,
        "acceptedCandidateCount": 0,
    }
    for bundle_id, bundle in enumerate(bundles):
        candidate = np.asarray(bundle["candidate"], dtype=np.int64)
        if np.any(candidate_assigned[candidate]):
            merge_accounting["rejectedReusedCandidate"] += 1
            continue
        first_root = _find(parent, int(bundle["componentFirst"]))
        second_root = _find(parent, int(bundle["componentSecond"]))
        first_state = component_state[first_root]
        second_state = component_state[second_root]
        if first_state is None or second_state is None:
            raise RuntimeError("bridge component state is unavailable")
        first_label = int(component_physical_label[first_root])
        second_label = int(component_physical_label[second_root])
        first_side = int(component_physical_side[first_root])
        second_side = int(component_physical_side[second_root])
        (
            physical_identity_compatible,
            merged_physical_label,
            merged_physical_side,
        ) = (
            _merge_physical_face_identity(
                first_label, first_side, second_label, second_side
            )
        )
        if first_root != second_root and not physical_identity_compatible:
            merge_accounting["rejectedPhysicalSeedConflict"] += 1
            continue
        if first_root == second_root:
            compatible, combined_state = _compatible_union(
                first_state,
                {},
                candidate,
                tangent_column,
                normal_depth,
                maximum_depth_range=graph_settings.maximum_column_depth_range_sampling_steps,
            )
        else:
            compatible, combined_state = _compatible_union(
                first_state,
                second_state,
                candidate,
                tangent_column,
                normal_depth,
                maximum_depth_range=graph_settings.maximum_column_depth_range_sampling_steps,
            )
        if not compatible:
            merge_accounting["rejectedStratumCollision"] += 1
            continue
        merged_components = first_root != second_root
        if merged_components:
            if len(first_state) >= len(second_state):
                retained_root, removed_root = first_root, second_root
            else:
                retained_root, removed_root = second_root, first_root
            parent[removed_root] = retained_root
            component_state[retained_root] = combined_state
            component_state[removed_root] = None
            component_physical_label[retained_root] = merged_physical_label
            component_physical_side[retained_root] = merged_physical_side
            component_physical_label[removed_root] = -1
            component_physical_side[removed_root] = 255
            component_physical_anchor_count[retained_root] = (
                component_physical_anchor_count[first_root]
                + component_physical_anchor_count[second_root]
            )
            component_physical_anchor_count[removed_root] = 0
            assigned_root = retained_root
            merge_accounting["acceptedMergeBundleCount"] += 1
        else:
            component_state[first_root] = combined_state
            assigned_root = first_root
            merge_accounting["acceptedInternalBundleCount"] += 1
        accepted_bundle_index = len(accepted_bundle_records)
        accepted_bundle_records.append(
            {**bundle, "mergedComponents": merged_components}
        )
        candidate_assigned[candidate] = True
        for record in bundle["records"]:
            interface_index = int(record["interfaceIndex"])
            accepted_candidate.append(interface_index)
            accepted_candidate_component.append(assigned_root)
            accepted_candidate_normal.append(
                np.asarray(record["geometry"]["sheetNormalXYZ"], dtype=np.float64)
            )
            accepted_candidate_bundle.append(accepted_bundle_index)
            accepted_candidate_score.append(float(record["score"]))
            accepted_candidate_opposition.append(
                float(record["oppositionDegrees"])
            )
            accepted_candidate_angular_gap.append(
                float(record["geometry"]["maximumSupportAngularGapDegrees"])
            )
            accepted_candidate_plane_residual.append(
                float(
                    record["geometry"][
                        "candidatePlaneResidualSamplingSteps"
                    ]
                )
            )
            for anchor in record["support"]:
                bridge_edge_first_interface.append(interface_index)
                bridge_edge_second_interface.append(int(anchor))
                bridge_edge_score.append(float(record["score"]))
    merge_accounting["acceptedCandidateCount"] = len(accepted_candidate)

    root_by_component = np.asarray(
        [_find(parent, value) for value in range(component_count)], dtype=np.int32
    )
    bridge_candidate = np.asarray(accepted_candidate, dtype=np.int64)
    bridge_root = np.asarray(
        [_find(parent, value) for value in accepted_candidate_component],
        dtype=np.int32,
    )
    active_interface = np.concatenate((growth_interface, bridge_candidate))
    active_root = np.concatenate((root_by_component[growth_component], bridge_root))
    root_value, root_count = np.unique(active_root, return_counts=True)
    root_physical_label = component_physical_label[root_value]
    root_physical_side = component_physical_side[root_value]
    root_physical_anchor_count = component_physical_anchor_count[root_value]
    priority = (
        root_physical_anchor_count
        >= graph_settings.minimum_physical_anchor_samples_for_priority
    )
    order = np.lexsort((root_value, -root_count, ~priority))
    rank_by_root = np.full(component_count, -1, dtype=np.int32)
    rank_by_root[root_value[order]] = np.arange(len(root_value), dtype=np.int32)
    component = rank_by_root[active_root]
    component_size = root_count[order]
    component_physical_label_final = root_physical_label[order]
    component_physical_side_final = root_physical_side[order]
    component_physical_anchor_count_final = root_physical_anchor_count[order]
    active_physical_label = component_physical_label_final[component]
    active_physical_side = component_physical_side_final[component]
    active_physical_anchor = np.concatenate(
        (
            growth_physical_anchor,
            np.zeros(len(bridge_candidate), dtype=np.uint8),
        )
    )
    node_by_interface = np.full(len(position), -1, dtype=np.int32)
    node_by_interface[active_interface] = np.arange(len(active_interface), dtype=np.int32)

    growth_edge_first_node = np.asarray(growth_arrays["edgeFirstNode"], dtype=np.int64)
    growth_edge_second_node = np.asarray(growth_arrays["edgeSecondNode"], dtype=np.int64)
    growth_edge_first_interface = growth_interface[growth_edge_first_node]
    growth_edge_second_interface = growth_interface[growth_edge_second_node]
    edge_first_interface = np.concatenate(
        (
            growth_edge_first_interface,
            np.asarray(bridge_edge_first_interface, dtype=np.int64),
        )
    )
    edge_second_interface = np.concatenate(
        (
            growth_edge_second_interface,
            np.asarray(bridge_edge_second_interface, dtype=np.int64),
        )
    )
    edge_first = node_by_interface[edge_first_interface]
    edge_second = node_by_interface[edge_second_interface]
    growth_edge_score = np.asarray(growth_arrays["edgeScore"], dtype=np.float32)
    edge_score = np.concatenate(
        (
            growth_edge_score,
            np.asarray(bridge_edge_score, dtype=np.float32),
        )
    )
    edge_kind = np.concatenate(
        (
            np.asarray(growth_arrays["edgeKind"], dtype=np.uint8),
            np.full(len(bridge_edge_score), 2, dtype=np.uint8),
        )
    )

    active_sheet_normal = np.concatenate(
        (
            growth_sheet_normal,
            np.asarray(accepted_candidate_normal, dtype=np.float64).reshape(-1, 3),
        ),
        axis=0,
    )
    active_growth_round = np.concatenate(
        (growth_round, np.full(len(bridge_candidate), -1, dtype=np.int16))
    )
    source_component = np.concatenate(
        (growth_component, np.full(len(bridge_candidate), -1, dtype=np.int32))
    )
    prior_bridge_bundle_id = (
        np.asarray(growth_arrays["bridgeBundleId"], dtype=np.int32)
        if "bridgeBundleId" in growth_arrays
        else np.full(len(growth_interface), -1, dtype=np.int32)
    )
    prior_bundle = prior_bridge_bundle_id[prior_bridge_bundle_id >= 0]
    bridge_bundle_offset = int(np.max(prior_bundle)) + 1 if len(prior_bundle) else 0
    bridge_bundle_id = np.concatenate(
        (
            prior_bridge_bundle_id,
            bridge_bundle_offset
            + np.asarray(accepted_candidate_bundle, dtype=np.int32),
        )
    )
    prior_bridge_score = (
        np.asarray(growth_arrays["bridgeScore"], dtype=np.float32)
        if "bridgeScore" in growth_arrays
        else np.full(len(growth_interface), np.nan, dtype=np.float32)
    )
    bridge_score = np.concatenate(
        (
            prior_bridge_score,
            np.asarray(accepted_candidate_score, dtype=np.float32),
        )
    )
    prior_bridge_opposition = (
        np.asarray(growth_arrays["bridgeOppositionDegrees"], dtype=np.float32)
        if "bridgeOppositionDegrees" in growth_arrays
        else np.full(len(growth_interface), np.nan, dtype=np.float32)
    )
    bridge_opposition = np.concatenate(
        (
            prior_bridge_opposition,
            np.asarray(accepted_candidate_opposition, dtype=np.float32),
        )
    )
    prior_bridge_angular_gap = (
        np.asarray(
            growth_arrays["bridgeSupportAngularGapDegrees"], dtype=np.float32
        )
        if "bridgeSupportAngularGapDegrees" in growth_arrays
        else np.full(len(growth_interface), np.nan, dtype=np.float32)
    )
    bridge_angular_gap = np.concatenate(
        (
            prior_bridge_angular_gap,
            np.asarray(accepted_candidate_angular_gap, dtype=np.float32),
        )
    )
    prior_bridge_plane_residual = (
        np.asarray(
            growth_arrays["bridgePlaneResidualSamplingSteps"], dtype=np.float32
        )
        if "bridgePlaneResidualSamplingSteps" in growth_arrays
        else np.full(len(growth_interface), np.nan, dtype=np.float32)
    )
    bridge_plane_residual = np.concatenate(
        (
            prior_bridge_plane_residual,
            np.asarray(accepted_candidate_plane_residual, dtype=np.float32),
        )
    )
    accepted_bundle_component_first = np.asarray(
        [int(value["componentFirst"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_component_second = np.asarray(
        [int(value["componentSecond"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_candidate_count = np.asarray(
        [int(value["candidateCount"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_first_anchor_count = np.asarray(
        [len(value["firstAnchor"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_second_anchor_count = np.asarray(
        [len(value["secondAnchor"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_tangent_column_count = np.asarray(
        [int(value["tangentColumnCount"]) for value in accepted_bundle_records],
        dtype=np.int32,
    )
    accepted_bundle_span = np.asarray(
        [float(value["spanSamplingSteps"]) for value in accepted_bundle_records],
        dtype=np.float32,
    )
    accepted_bundle_mean_candidate_score = np.asarray(
        [float(value["meanCandidateScore"]) for value in accepted_bundle_records],
        dtype=np.float32,
    )
    accepted_bundle_score = np.asarray(
        [float(value["score"]) for value in accepted_bundle_records],
        dtype=np.float32,
    )
    accepted_bundle_merged = np.asarray(
        [bool(value["mergedComponents"]) for value in accepted_bundle_records],
        dtype=np.uint8,
    )
    preserved_growth_metrics: dict[str, np.ndarray] = {}
    for name in (
        "growthSupportCount",
        "growthComponentDominance",
        "growthSupportAngularGapDegrees",
        "growthTangentRankRatio",
        "growthPlaneResidualSamplingSteps",
        "growthSupportHeightStandardDeviationSamplingSteps",
        "growthSignedNormalDisagreementDegrees",
        "growthScore",
    ):
        if name not in growth_arrays:
            continue
        prior = np.asarray(growth_arrays[name])
        fill = (
            np.zeros(len(bridge_candidate), dtype=prior.dtype)
            if np.issubdtype(prior.dtype, np.integer)
            else np.full(len(bridge_candidate), np.nan, dtype=prior.dtype)
        )
        preserved_growth_metrics[name] = np.concatenate((prior, fill))
    arrays = {
        "interfaceIndex": active_interface.astype(np.int32),
        "positionXYZ": position[active_interface].astype(np.float32),
        "signedNormalXYZ": signed_normal[active_interface].astype(np.float32),
        "sheetNormalXYZ": active_sheet_normal.astype(np.float32),
        "macroNormalXYZ": macro_normal[active_interface].astype(np.float32),
        "macroOrientationConfidence": macro_confidence[active_interface].astype(np.float32),
        "orientationSource": orientation_source[active_interface].astype(np.uint8),
        "physicalSeedAnchor": active_physical_anchor,
        "physicalSheetLabel": active_physical_label.astype(np.int32),
        "physicalBoundarySide": active_physical_side.astype(np.uint8),
        "componentPhysicalSheetLabel": component_physical_label_final.astype(
            np.int32
        ),
        "componentPhysicalBoundarySide": component_physical_side_final.astype(
            np.uint8
        ),
        "componentPhysicalAnchorCount": component_physical_anchor_count_final.astype(
            np.int32
        ),
        "rawToMacroNormalDegrees": raw_macro_angle[active_interface].astype(np.float32),
        "localEvidenceScore": evidence[active_interface].astype(np.float32),
        "sourceComponentId": source_component,
        "componentId": component.astype(np.int32),
        "growthRound": active_growth_round,
        "bridgeBundleId": bridge_bundle_id,
        "bridgeScore": bridge_score,
        "bridgeOppositionDegrees": bridge_opposition,
        "bridgeSupportAngularGapDegrees": bridge_angular_gap,
        "bridgePlaneResidualSamplingSteps": bridge_plane_residual,
        "tangentColumnId": tangent_column[active_interface],
        "normalDepthSamplingSteps": normal_depth[active_interface],
        **preserved_growth_metrics,
        "edgeFirstNode": edge_first.astype(np.int32),
        "edgeSecondNode": edge_second.astype(np.int32),
        "edgeScore": edge_score,
        "edgeKind": edge_kind,
        "acceptedBundleComponentFirst": accepted_bundle_component_first,
        "acceptedBundleComponentSecond": accepted_bundle_component_second,
        "acceptedBundleCandidateCount": accepted_bundle_candidate_count,
        "acceptedBundleFirstAnchorCount": accepted_bundle_first_anchor_count,
        "acceptedBundleSecondAnchorCount": accepted_bundle_second_anchor_count,
        "acceptedBundleTangentColumnCount": accepted_bundle_tangent_column_count,
        "acceptedBundleSpanSamplingSteps": accepted_bundle_span,
        "acceptedBundleMeanCandidateScore": accepted_bundle_mean_candidate_score,
        "acceptedBundleScore": accepted_bundle_score,
        "acceptedBundleMergedComponents": accepted_bundle_merged,
    }
    _write_npz(data_path, arrays)
    source = VolumeSource.open(
        interfaces["source"]["path"], interfaces["source"].get("metadataPath")
    )
    owned_record = interfaces["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(int(value) for value in owned_record["startXYZ"]),
        tuple(int(value) for value in owned_record["stopXYZExclusive"]),
    )
    write_material_surface_cross_sections(
        source,
        owned,
        position[active_interface],
        component,
        component_size,
        preview_path,
        display_high_raw=float(interfaces["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=graph_settings,
    )
    substantial = component_size >= graph_settings.minimum_component_samples_for_preview
    payload: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_BRIDGING_SCHEMA,
        "version": MATERIAL_SURFACE_BRIDGING_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interfaces["source"],
        "geometry": interfaces["geometry"],
        "counts": {
            "interfaceSampleCount": int(len(position)),
            "interiorGrowthNodeCount": int(len(growth_interface)),
            "bridgeCandidateNodeCount": int(
                np.count_nonzero(bridge_bundle_id >= 0)
            ),
            "newBridgeCandidateNodeCount": int(len(bridge_candidate)),
            "activeNodeCount": int(len(active_interface)),
            "activeNodeFraction": round(len(active_interface) / max(len(position), 1), 6),
            "componentCountBefore": int(component_count),
            "componentCount": int(len(component_size)),
            "physicallyAnchoredComponentCount": int(
                np.count_nonzero(component_physical_label_final >= 0)
            ),
            "nodesInPhysicallyAnchoredComponents": int(
                np.sum(
                    component_size[component_physical_label_final >= 0]
                )
            ),
            "physicalAnchorNodeCount": int(
                np.count_nonzero(active_physical_anchor == 1)
            ),
            "initialComponentCount": int(
                growth.get("counts", {}).get(
                    "initialComponentCount", component_count
                )
            ),
            "newComponentMergeCount": int(
                component_count - len(component_size)
            ),
            "componentMergeCount": int(
                growth.get("counts", {}).get("componentMergeCount", 0)
                + component_count
                - len(component_size)
            ),
            "retainedPriorEdgeCount": int(len(growth_edge_score)),
            "bridgeEdgeCount": int(len(bridge_edge_score)),
            "retainedEdgeCount": int(len(edge_score)),
            "componentsAtLeast8Nodes": int(np.count_nonzero(component_size >= 8)),
            "componentsAtLeast32Nodes": int(np.count_nonzero(component_size >= 32)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_size >= 128)),
            "nodesInPreviewSizedComponents": int(np.sum(component_size[substantial])),
            "largestComponentSizes": [int(value) for value in component_size[:32]],
        },
        "candidateAccounting": candidate_accounting,
        "bundleAccounting": bundle_accounting,
        "mergeAccounting": merge_accounting,
        "distributions": {
            "componentSize": _percentiles(component_size),
            "acceptedBundleScore": _percentiles(accepted_bundle_score),
            "acceptedBridgeCandidateEvidence": _percentiles(evidence[bridge_candidate]),
            "acceptedBridgeCandidateScore": _percentiles(
                np.asarray(accepted_candidate_score, dtype=np.float64)
            ),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "bridgeUnit": (
                "spatially connected run of unused signed interfaces with multiple "
                "anchors in exactly two components and opposing tangent support"
            ),
            "minimumRepeatedEvidence": resolved.minimum_bundle_candidate_count,
            "mergeOrder": "descending bundle score",
            "transitiveLayerGuard": (
                "every union plus bridge candidates must preserve the inherited "
                "tangent-column normal-depth interval"
            ),
            "physicalIdentityGuard": (
                "an unanchored component may attach to one physical boundary face, "
                "but different sheets or opposite sides of one sheet never merge"
            ),
            "singleEdgeMerges": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
