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
    write_material_surface_cross_sections,
)
from .material_surface_growth import (
    _build_neighbor_interface_matrix,
    _neighbor_offsets,
    _resolve_seed_surface,
    _root_graph_manifest,
    _support_geometry,
)


MATERIAL_SURFACE_FRONTIER_SCHEMA = "pareidolia.material-interface-frontier-bundles"
MATERIAL_SURFACE_FRONTIER_VERSION = 1
MATERIAL_SURFACE_FRONTIER_STEM = "material-interface-frontier-bundles-v1"


@dataclass(frozen=True, slots=True)
class MaterialSurfaceFrontierSettings:
    minimum_local_evidence: float = 0.5
    neighbor_radius_sampling_steps: float = math.sqrt(3.0)
    minimum_anchor_samples: int = 2
    minimum_open_support_gap_degrees: float = 180.0
    maximum_open_support_gap_degrees: float = 300.0
    maximum_candidate_plane_residual_sampling_steps: float = 1.15
    maximum_support_height_standard_deviation_sampling_steps: float = 0.8
    maximum_signed_normal_disagreement_degrees: float = 70.0
    minimum_expansion_displacement_sampling_steps: float = 0.5
    minimum_sheet_thickness_microns: float = 80.0
    maximum_sheet_thickness_microns: float = 400.0
    maximum_opposing_face_tangent_distance_sampling_steps: float = 1.75
    maximum_opposing_face_normal_degrees: float = 50.0
    minimum_opposing_face_evidence: float = 0.5
    candidate_cluster_radius_sampling_steps: float = math.sqrt(3.0)
    minimum_bundle_candidate_count: int = 4
    minimum_bundle_anchor_count: int = 4
    minimum_bundle_opposing_face_count: int = 3
    minimum_bundle_tangent_column_count: int = 3
    minimum_bundle_span_sampling_steps: float = 3.0
    minimum_expansion_direction_coherence: float = 0.55
    maximum_bundle_normal_p90_degrees: float = 25.0

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_local_evidence,
            self.minimum_opposing_face_evidence,
            self.minimum_expansion_direction_coherence,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("frontier evidence and coherence must lie in [0, 1]")
        if self.neighbor_radius_sampling_steps < 1.0:
            raise ValueError("frontier neighbor radius must be at least one sample")
        if self.minimum_anchor_samples < 2:
            raise ValueError("frontier candidates require multiple anchors")
        if not (
            0.0
            < self.minimum_open_support_gap_degrees
            <= self.maximum_open_support_gap_degrees
            < 360.0
        ):
            raise ValueError("frontier open-support angular interval is invalid")
        if not 0.0 < self.maximum_signed_normal_disagreement_degrees < 90.0:
            raise ValueError("frontier signed-normal cap must lie in (0, 90)")
        if not 0.0 < self.maximum_opposing_face_normal_degrees < 90.0:
            raise ValueError("opposing-face normal cap must lie in (0, 90)")
        positive = (
            self.maximum_candidate_plane_residual_sampling_steps,
            self.maximum_support_height_standard_deviation_sampling_steps,
            self.minimum_expansion_displacement_sampling_steps,
            self.minimum_sheet_thickness_microns,
            self.maximum_sheet_thickness_microns,
            self.maximum_opposing_face_tangent_distance_sampling_steps,
            self.candidate_cluster_radius_sampling_steps,
            self.minimum_bundle_span_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("frontier physical thresholds must be finite and positive")
        if self.maximum_sheet_thickness_microns <= self.minimum_sheet_thickness_microns:
            raise ValueError("frontier sheet-thickness interval must increase")
        integer_positive = (
            self.minimum_bundle_candidate_count,
            self.minimum_bundle_anchor_count,
            self.minimum_bundle_opposing_face_count,
            self.minimum_bundle_tangent_column_count,
        )
        if any(value < 1 for value in integer_positive):
            raise ValueError("frontier bundle counts must be positive")
        if not 0.0 < self.maximum_bundle_normal_p90_degrees < 90.0:
            raise ValueError("frontier bundle normal cap must lie in (0, 90)")

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


def _cluster_candidates(
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


def _opposing_faces(
    candidate: np.ndarray,
    plane_normal: np.ndarray,
    key: np.ndarray,
    grid: np.ndarray,
    processing_shape_xyz: np.ndarray,
    position: np.ndarray,
    signed_normal: np.ndarray,
    evidence: np.ndarray,
    *,
    sampling_stride_voxels: int,
    voxel_size_microns: float,
    settings: MaterialSurfaceFrontierSettings,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    count = len(candidate)
    best_index = np.full(count, -1, dtype=np.int32)
    best_score = np.full(count, -np.inf, dtype=np.float64)
    best_thickness = np.full(count, np.nan, dtype=np.float64)
    best_tangent = np.full(count, np.nan, dtype=np.float64)
    sampling_microns = sampling_stride_voxels * voxel_size_microns
    minimum_depth = settings.minimum_sheet_thickness_microns / sampling_microns
    maximum_depth = settings.maximum_sheet_thickness_microns / sampling_microns
    depth_values = range(
        max(1, int(math.floor(minimum_depth)) - 1),
        int(math.ceil(maximum_depth)) + 2,
    )
    reach = int(
        math.ceil(settings.maximum_opposing_face_tangent_distance_sampling_steps)
    )
    offsets = np.asarray(
        [
            (dx, dy, dz)
            for dz in range(-reach, reach + 1)
            for dy in range(-reach, reach + 1)
            for dx in range(-reach, reach + 1)
            if math.sqrt(dx * dx + dy * dy + dz * dz) <= reach + 1.0e-9
        ],
        dtype=np.int64,
    )
    candidate_key = key[candidate]
    candidate_position = position[candidate]
    normal_cosine = math.cos(
        math.radians(settings.maximum_opposing_face_normal_degrees)
    )
    for depth_value in depth_values:
        center = np.rint(
            candidate_key + plane_normal * float(depth_value)
        ).astype(np.int64)
        for offset in offsets:
            query = center + offset[None, :]
            valid_query = np.all(
                (query >= 0) & (query < processing_shape_xyz[None, :]), axis=1
            )
            if not np.any(valid_query):
                continue
            row = np.flatnonzero(valid_query)
            selected = query[valid_query]
            interface_index = grid[
                selected[:, 2], selected[:, 1], selected[:, 0]
            ]
            exists = (interface_index >= 0) & (
                interface_index != candidate[row]
            )
            if not np.any(exists):
                continue
            row = row[exists]
            interface_index = interface_index[exists]
            displacement = (
                position[interface_index] - candidate_position[row]
            ) / float(sampling_stride_voxels)
            depth = np.einsum(
                "ij,ij->i", displacement, plane_normal[row]
            )
            tangent = displacement - depth[:, None] * plane_normal[row]
            tangent_distance = np.linalg.norm(tangent, axis=1)
            opposing_cosine = np.einsum(
                "ij,ij->i", signed_normal[interface_index], -plane_normal[row]
            )
            good = (
                (depth >= minimum_depth)
                & (depth <= maximum_depth)
                & (
                    tangent_distance
                    <= settings.maximum_opposing_face_tangent_distance_sampling_steps
                )
                & (opposing_cosine >= normal_cosine)
                & (
                    evidence[interface_index]
                    >= settings.minimum_opposing_face_evidence
                )
            )
            if not np.any(good):
                continue
            row = row[good]
            interface_index = interface_index[good]
            depth = depth[good]
            tangent_distance = tangent_distance[good]
            opposing_cosine = opposing_cosine[good]
            score = (
                evidence[interface_index]
                * opposing_cosine
                * np.exp(
                    -0.5
                    * (
                        tangent_distance
                        / max(
                            settings.maximum_opposing_face_tangent_distance_sampling_steps,
                            1.0e-9,
                        )
                    )
                    ** 2
                )
            )
            improved = score > best_score[row]
            if not np.any(improved):
                continue
            improved_row = row[improved]
            best_score[improved_row] = score[improved]
            best_index[improved_row] = interface_index[improved]
            best_thickness[improved_row] = depth[improved] * sampling_microns
            best_tangent[improved_row] = tangent_distance[improved]
    return best_index, best_score, best_thickness, best_tangent


def run_material_surface_frontier_census(
    interface_root: str | Path,
    macro_root: str | Path,
    surface_root: str | Path,
    output_path: str | Path,
    *,
    settings: MaterialSurfaceFrontierSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve(interface_root, MATERIAL_INTERFACE_STEM)
    macro_path = _resolve(macro_root, MACRO_ORIENTATION_STEM)
    surface_path = _resolve_seed_surface(surface_root)
    interfaces = json.loads(interface_path.read_text())
    macro = json.loads(macro_path.read_text())
    surface = json.loads(surface_path.read_text())
    if interfaces.get("schema") != MATERIAL_INTERFACE_SCHEMA or interfaces.get("state") != "complete":
        raise ValueError("frontier census requires complete material interfaces")
    if macro.get("schema") != MACRO_ORIENTATION_SCHEMA or macro.get("state") != "complete":
        raise ValueError("frontier census requires a complete macro orientation field")
    if surface.get("state") != "complete":
        raise ValueError("frontier census requires a complete signed surface")
    if surface["identity"]["interfaces"]["manifestSha256"] != sha256_file(interface_path):
        raise ValueError("frontier surface was not derived from the supplied interfaces")
    if surface["identity"]["macroOrientation"]["manifestSha256"] != sha256_file(macro_path):
        raise ValueError("frontier surface was not derived from the supplied macro field")
    graph_path, graph = _root_graph_manifest(surface_path, surface)
    graph_settings = MaterialSurfaceGraphSettings(**graph["identity"]["settings"])
    resolved = settings or MaterialSurfaceFrontierSettings()
    identity: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_FRONTIER_SCHEMA,
        "version": MATERIAL_SURFACE_FRONTIER_VERSION,
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
        "surface": {
            "schema": str(surface["schema"]),
            "manifestPath": str(surface_path),
            "manifestSha256": sha256_file(surface_path),
            "dataSha256": surface["data"]["sha256"],
        },
        "seedGraph": {
            "manifestPath": str(graph_path),
            "manifestSha256": sha256_file(graph_path),
            "dataSha256": graph["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": {
            "material_surface_frontier.py": sha256_file(Path(__file__)),
            "material_surface_growth.py": sha256_file(
                Path(__file__).with_name("material_surface_growth.py")
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_SURFACE_FRONTIER_STEM}.json"
    data_path = output / f"{MATERIAL_SURFACE_FRONTIER_STEM}.npz"
    preview_path = output / "material-surface-frontier-bundles.png"
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
    surface_arrays = _load_npz(surface_path, surface)
    position = np.asarray(interface_arrays["positionXYZ"], dtype=np.float64)
    signed_normal = np.asarray(interface_arrays["signedNormalXYZ"], dtype=np.float64)
    key = np.asarray(interface_arrays["processingKeyXYZ"], dtype=np.int64)
    evidence = np.asarray(interface_arrays["localEvidenceScore"], dtype=np.float64)
    orientation = sample_orientation_field(macro_arrays)
    macro_normal = orientation["sampleNormalXYZ"]
    macro_confidence = orientation["sampleOrientationConfidence"]
    macro_trusted = orientation["sampleTrusted"]
    surface_interface = np.asarray(surface_arrays["interfaceIndex"], dtype=np.int64)
    surface_component = np.asarray(surface_arrays["componentId"], dtype=np.int32)
    surface_sheet_normal = np.asarray(
        surface_arrays["sheetNormalXYZ"], dtype=np.float64
    )
    component_count = int(np.max(surface_component)) + 1 if len(surface_component) else 0
    component_by_interface = np.full(len(position), -1, dtype=np.int32)
    component_by_interface[surface_interface] = surface_component
    sheet_normal = macro_normal.copy()
    sheet_normal[surface_interface] = surface_sheet_normal
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
    inactive = np.flatnonzero(
        (component_by_interface < 0)
        & (evidence >= resolved.minimum_local_evidence)
        & macro_trusted
        & (macro_confidence >= graph_settings.minimum_macro_confidence)
    )
    neighbor_interface = _build_neighbor_interface_matrix(
        inactive,
        key,
        grid,
        processing_shape,
        _neighbor_offsets(resolved.neighbor_radius_sampling_steps),
    )
    neighbor_component = np.full_like(neighbor_interface, -1)
    exists = neighbor_interface >= 0
    neighbor_component[exists] = component_by_interface[neighbor_interface[exists]]
    accounting = {
        "inactiveHighEvidenceCandidateCount": int(len(inactive)),
        "candidatesWithAnyActiveNeighbor": int(
            np.count_nonzero(np.any(neighbor_component >= 0, axis=1))
        ),
        "rejectedInsufficientAnchors": 0,
        "rejectedAmbiguousComponent": 0,
        "rejectedNotOpenFrontier": 0,
        "rejectedPlaneResidual": 0,
        "rejectedSupportHeightSpread": 0,
        "rejectedSignedNormal": 0,
        "rejectedExpansionDisplacement": 0,
        "locallyQualifiedCandidateCount": 0,
        "rejectedMissingOpposingFace": 0,
        "pairedCandidateCount": 0,
    }
    local_records: list[dict[str, Any]] = []
    for row, interface_index in enumerate(inactive):
        active = neighbor_component[row] >= 0
        if np.count_nonzero(active) < resolved.minimum_anchor_samples:
            accounting["rejectedInsufficientAnchors"] += 1
            continue
        components = np.unique(neighbor_component[row, active])
        if len(components) != 1:
            accounting["rejectedAmbiguousComponent"] += 1
            continue
        component_id = int(components[0])
        support = neighbor_interface[
            row, neighbor_component[row] == component_id
        ].astype(np.int64)
        geometry = _support_geometry(
            position[interface_index],
            signed_normal[interface_index],
            position[support],
            sheet_normal[support],
            signed_normal[support],
            sampling_stride_voxels=stride,
        )
        angular_gap = float(geometry["maximumSupportAngularGapDegrees"])
        if not (
            angular_gap > resolved.minimum_open_support_gap_degrees + 1.0e-6
            and angular_gap <= resolved.maximum_open_support_gap_degrees
        ):
            accounting["rejectedNotOpenFrontier"] += 1
            continue
        if (
            float(geometry["candidatePlaneResidualSamplingSteps"])
            > resolved.maximum_candidate_plane_residual_sampling_steps
        ):
            accounting["rejectedPlaneResidual"] += 1
            continue
        if (
            float(geometry["supportHeightStandardDeviationSamplingSteps"])
            > resolved.maximum_support_height_standard_deviation_sampling_steps
        ):
            accounting["rejectedSupportHeightSpread"] += 1
            continue
        if (
            float(geometry["signedNormalDisagreementDegrees"])
            > resolved.maximum_signed_normal_disagreement_degrees
        ):
            accounting["rejectedSignedNormal"] += 1
            continue
        plane_normal = np.asarray(geometry["sheetNormalXYZ"], dtype=np.float64)
        displacement = (
            position[interface_index] - np.mean(position[support], axis=0)
        ) / float(stride)
        expansion = displacement - float(np.dot(displacement, plane_normal)) * plane_normal
        expansion_length = float(np.linalg.norm(expansion))
        if expansion_length < resolved.minimum_expansion_displacement_sampling_steps:
            accounting["rejectedExpansionDisplacement"] += 1
            continue
        accounting["locallyQualifiedCandidateCount"] += 1
        local_records.append(
            {
                "interfaceIndex": int(interface_index),
                "component": component_id,
                "support": support,
                "geometry": geometry,
                "expansionDirectionXYZ": expansion / expansion_length,
                "expansionDisplacementSamplingSteps": expansion_length,
            }
        )

    local_candidate = np.asarray(
        [int(record["interfaceIndex"]) for record in local_records], dtype=np.int64
    )
    local_normal = np.asarray(
        [record["geometry"]["sheetNormalXYZ"] for record in local_records],
        dtype=np.float64,
    ).reshape(-1, 3)
    opposing_index, opposing_score, thickness_microns, opposing_tangent = (
        _opposing_faces(
            local_candidate,
            local_normal,
            key,
            grid,
            processing_shape,
            position,
            signed_normal,
            evidence,
            sampling_stride_voxels=stride,
            voxel_size_microns=float(interfaces["source"]["voxelSizeMicrons"]),
            settings=resolved,
        )
        if len(local_candidate)
        else (
            np.empty(0, dtype=np.int32),
            np.empty(0),
            np.empty(0),
            np.empty(0),
        )
    )
    paired_records: list[dict[str, Any]] = []
    for index, record in enumerate(local_records):
        if opposing_index[index] < 0:
            accounting["rejectedMissingOpposingFace"] += 1
            continue
        paired_records.append(
            {
                **record,
                "opposingInterfaceIndex": int(opposing_index[index]),
                "opposingScore": float(opposing_score[index]),
                "thicknessMicrons": float(thickness_microns[index]),
                "opposingTangentDistanceSamplingSteps": float(
                    opposing_tangent[index]
                ),
            }
        )
    accounting["pairedCandidateCount"] = len(paired_records)

    records_by_component: dict[int, list[dict[str, Any]]] = {}
    for record in paired_records:
        records_by_component.setdefault(int(record["component"]), []).append(record)
    bundle_accounting = {
        "componentWithPairedCandidatesCount": int(len(records_by_component)),
        "spatialClusterCount": 0,
        "rejectedSmallCandidateRun": 0,
        "rejectedInsufficientAnchors": 0,
        "rejectedInsufficientOpposingFaces": 0,
        "rejectedInsufficientTangentColumns": 0,
        "rejectedShortSpan": 0,
        "rejectedDirectionIncoherence": 0,
        "rejectedNormalIncoherence": 0,
        "qualifiedBundleCount": 0,
    }
    bundles: list[dict[str, Any]] = []
    width = graph_settings.tangent_column_width_sampling_steps
    for component_id, records in records_by_component.items():
        for cluster in _cluster_candidates(
            records,
            key,
            radius=resolved.candidate_cluster_radius_sampling_steps,
        ):
            bundle_accounting["spatialClusterCount"] += 1
            if len(cluster) < resolved.minimum_bundle_candidate_count:
                bundle_accounting["rejectedSmallCandidateRun"] += 1
                continue
            candidate = np.asarray(
                [int(record["interfaceIndex"]) for record in cluster],
                dtype=np.int64,
            )
            anchors = np.unique(
                np.concatenate([record["support"] for record in cluster])
            )
            if len(anchors) < resolved.minimum_bundle_anchor_count:
                bundle_accounting["rejectedInsufficientAnchors"] += 1
                continue
            opposing = np.unique(
                [int(record["opposingInterfaceIndex"]) for record in cluster]
            )
            if len(opposing) < resolved.minimum_bundle_opposing_face_count:
                bundle_accounting["rejectedInsufficientOpposingFaces"] += 1
                continue
            normal = np.asarray(
                [record["geometry"]["sheetNormalXYZ"] for record in cluster],
                dtype=np.float64,
            )
            reference = normal[0]
            normal[np.einsum("ij,j->i", normal, reference) < 0.0] *= -1.0
            mean_normal = np.sum(normal, axis=0)
            mean_normal /= max(float(np.linalg.norm(mean_normal)), 1.0e-9)
            delta = position[candidate] / float(stride)
            reference_axis = int(np.argmin(np.abs(mean_normal)))
            axis = np.zeros(3)
            axis[reference_axis] = 1.0
            first = np.cross(mean_normal, axis)
            first /= max(float(np.linalg.norm(first)), 1.0e-9)
            second = np.cross(mean_normal, first)
            coordinate = np.column_stack((delta @ first, delta @ second))
            column_record = np.rint(coordinate / width).astype(np.int64)
            column_count = len(np.unique(column_record, axis=0))
            if column_count < resolved.minimum_bundle_tangent_column_count:
                bundle_accounting["rejectedInsufficientTangentColumns"] += 1
                continue
            span = float(np.linalg.norm(np.ptp(position[candidate], axis=0)) / stride)
            if span < resolved.minimum_bundle_span_sampling_steps:
                bundle_accounting["rejectedShortSpan"] += 1
                continue
            direction = np.asarray(
                [record["expansionDirectionXYZ"] for record in cluster],
                dtype=np.float64,
            )
            direction_coherence = float(
                np.linalg.norm(np.mean(direction, axis=0))
            )
            if direction_coherence < resolved.minimum_expansion_direction_coherence:
                bundle_accounting["rejectedDirectionIncoherence"] += 1
                continue
            normal_angle = np.degrees(
                np.arccos(np.clip(np.abs(normal @ mean_normal), 0.0, 1.0))
            )
            normal_p90 = float(np.percentile(normal_angle, 90))
            if normal_p90 > resolved.maximum_bundle_normal_p90_degrees:
                bundle_accounting["rejectedNormalIncoherence"] += 1
                continue
            score = float(
                np.mean([record["opposingScore"] for record in cluster])
                * direction_coherence
                * math.log1p(len(cluster))
                * min(span / 6.0, 1.0)
            )
            bundles.append(
                {
                    "component": component_id,
                    "records": cluster,
                    "candidate": candidate,
                    "candidateCount": len(cluster),
                    "anchorCount": len(anchors),
                    "opposingFaceCount": len(opposing),
                    "tangentColumnCount": column_count,
                    "spanSamplingSteps": span,
                    "directionCoherence": direction_coherence,
                    "normalP90Degrees": normal_p90,
                    "score": score,
                }
            )
            bundle_accounting["qualifiedBundleCount"] += 1
    bundles.sort(
        key=lambda value: (-float(value["score"]), int(value["component"]))
    )

    candidate_records: list[dict[str, Any]] = []
    for bundle_id, bundle in enumerate(bundles):
        for record in bundle["records"]:
            candidate_records.append({**record, "bundleId": bundle_id})
    candidate_interface = np.asarray(
        [int(record["interfaceIndex"]) for record in candidate_records],
        dtype=np.int32,
    )
    candidate_bundle = np.asarray(
        [int(record["bundleId"]) for record in candidate_records], dtype=np.int32
    )
    bundle_size = np.asarray(
        [int(bundle["candidateCount"]) for bundle in bundles], dtype=np.int32
    )
    arrays = {
        "candidateInterfaceIndex": candidate_interface,
        "candidateComponentId": np.asarray(
            [int(record["component"]) for record in candidate_records],
            dtype=np.int32,
        ),
        "candidateBundleId": candidate_bundle,
        "candidatePositionXYZ": position[candidate_interface].astype(np.float32),
        "candidateSheetNormalXYZ": np.asarray(
            [record["geometry"]["sheetNormalXYZ"] for record in candidate_records],
            dtype=np.float32,
        ).reshape(-1, 3),
        "candidateExpansionDirectionXYZ": np.asarray(
            [record["expansionDirectionXYZ"] for record in candidate_records],
            dtype=np.float32,
        ).reshape(-1, 3),
        "candidateSupportCount": np.asarray(
            [len(record["support"]) for record in candidate_records], dtype=np.int16
        ),
        "candidateOpposingInterfaceIndex": np.asarray(
            [int(record["opposingInterfaceIndex"]) for record in candidate_records],
            dtype=np.int32,
        ),
        "candidateThicknessMicrons": np.asarray(
            [float(record["thicknessMicrons"]) for record in candidate_records],
            dtype=np.float32,
        ),
        "candidateOpposingScore": np.asarray(
            [float(record["opposingScore"]) for record in candidate_records],
            dtype=np.float32,
        ),
        "bundleComponentId": np.asarray(
            [int(bundle["component"]) for bundle in bundles], dtype=np.int32
        ),
        "bundleCandidateCount": bundle_size,
        "bundleAnchorCount": np.asarray(
            [int(bundle["anchorCount"]) for bundle in bundles], dtype=np.int32
        ),
        "bundleOpposingFaceCount": np.asarray(
            [int(bundle["opposingFaceCount"]) for bundle in bundles], dtype=np.int32
        ),
        "bundleTangentColumnCount": np.asarray(
            [int(bundle["tangentColumnCount"]) for bundle in bundles], dtype=np.int32
        ),
        "bundleSpanSamplingSteps": np.asarray(
            [float(bundle["spanSamplingSteps"]) for bundle in bundles],
            dtype=np.float32,
        ),
        "bundleDirectionCoherence": np.asarray(
            [float(bundle["directionCoherence"]) for bundle in bundles],
            dtype=np.float32,
        ),
        "bundleNormalP90Degrees": np.asarray(
            [float(bundle["normalP90Degrees"]) for bundle in bundles],
            dtype=np.float32,
        ),
        "bundleScore": np.asarray(
            [float(bundle["score"]) for bundle in bundles], dtype=np.float32
        ),
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
        position[candidate_interface],
        candidate_bundle,
        bundle_size,
        preview_path,
        display_high_raw=float(interfaces["calibration"]["displayHighRaw"]),
        sampling_stride_voxels=stride,
        settings=graph_settings,
    )
    payload: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_FRONTIER_SCHEMA,
        "version": MATERIAL_SURFACE_FRONTIER_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interfaces["source"],
        "geometry": interfaces["geometry"],
        "counts": {
            "interfaceSampleCount": int(len(position)),
            "surfaceNodeCount": int(len(surface_interface)),
            "surfaceComponentCount": component_count,
            "locallyQualifiedCandidateCount": len(local_records),
            "pairedCandidateCount": len(paired_records),
            "qualifiedBundleCandidateCount": int(len(candidate_interface)),
            "qualifiedBundleCount": int(len(bundles)),
            "largestBundleSizes": [int(value) for value in bundle_size[:32]],
        },
        "candidateAccounting": accounting,
        "bundleAccounting": bundle_accounting,
        "distributions": {
            "bundleSize": _percentiles(bundle_size),
            "bundleScore": _percentiles(arrays["bundleScore"]),
            "candidateThicknessMicrons": _percentiles(
                arrays["candidateThicknessMicrons"]
            ),
            "candidateOpposingScore": _percentiles(
                arrays["candidateOpposingScore"]
            ),
            "bundleDirectionCoherence": _percentiles(
                arrays["bundleDirectionCoherence"]
            ),
            "bundleNormalP90Degrees": _percentiles(
                arrays["bundleNormalP90Degrees"]
            ),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"bundleCrossSections": preview_path.name},
        "method": {
            "candidate": (
                "one high-evidence unused face with multiple anchors in exactly "
                "one component and open tangent support"
            ),
            "physicalPair": (
                "an opposite signed face 80-400 microns inward with bounded "
                "tangent displacement"
            ),
            "bundle": (
                "a spatially connected, extended candidate run with repeated "
                "anchors, distinct opposite faces, coherent normals, and a "
                "shared outward tangent direction"
            ),
            "mutatesSurface": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
