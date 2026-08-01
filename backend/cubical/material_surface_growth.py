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
    MATERIAL_SURFACE_GRAPH_SCHEMA,
    MATERIAL_SURFACE_GRAPH_STEM,
    MaterialSurfaceGraphSettings,
    _load_npz,
    _percentiles,
    _plane_basis,
    _tangent_column_records,
    write_material_surface_cross_sections,
)


MATERIAL_SURFACE_GROWTH_SCHEMA = "pareidolia.material-interface-interior-growth"
MATERIAL_SURFACE_GROWTH_VERSION = 1
MATERIAL_SURFACE_GROWTH_STEM = "material-interface-interior-growth-v1"


@dataclass(frozen=True, slots=True)
class MaterialSurfaceGrowthSettings:
    """Conservative recovery of holes inside an existing interface sheet.

    A candidate is admitted only when one existing component surrounds it in
    the local tangent plane.  This is deliberately not an outward-frontier or
    component-merging stage.
    """

    minimum_local_evidence: float = 0.1
    neighbor_radius_sampling_steps: float = math.sqrt(3.0)
    minimum_dominant_support_samples: int = 4
    minimum_component_dominance_fraction: float = 0.75
    maximum_support_angular_gap_degrees: float = 180.0
    minimum_tangent_rank_ratio: float = 0.08
    maximum_candidate_plane_residual_sampling_steps: float = 1.15
    maximum_support_height_standard_deviation_sampling_steps: float = 0.8
    maximum_signed_normal_disagreement_degrees: float = 70.0
    maximum_growth_rounds: int = 8
    minimum_round_additions: int = 1

    def __post_init__(self) -> None:
        fractions = (
            self.minimum_local_evidence,
            self.minimum_component_dominance_fraction,
            self.minimum_tangent_rank_ratio,
        )
        if any(not 0.0 <= value <= 1.0 for value in fractions):
            raise ValueError("growth confidence and rank thresholds must lie in [0, 1]")
        if self.neighbor_radius_sampling_steps < 1.0:
            raise ValueError("growth neighbor radius must be at least one sample")
        if self.minimum_dominant_support_samples < 3:
            raise ValueError("interior growth requires at least three support samples")
        if not 0.0 < self.maximum_support_angular_gap_degrees <= 180.0:
            raise ValueError(
                "interior angular gap must lie in (0, 180] so support encloses the candidate"
            )
        positive = (
            self.maximum_candidate_plane_residual_sampling_steps,
            self.maximum_support_height_standard_deviation_sampling_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("growth geometry thresholds must be finite and positive")
        if not 0.0 < self.maximum_signed_normal_disagreement_degrees < 90.0:
            raise ValueError("signed-normal disagreement must lie in (0, 90)")
        if self.maximum_growth_rounds < 1 or self.minimum_round_additions < 1:
            raise ValueError("growth round limits must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _resolve_seed_surface(root: str | Path) -> Path:
    value = Path(root).resolve()
    if value.is_file():
        candidate = value
    else:
        fixed_point = value / "material-surface-fixed-point-v1.json"
        if fixed_point.is_file():
            candidate = fixed_point
        else:
            candidate = Path()
    if candidate.is_file():
        manifest = json.loads(candidate.read_text())
        if manifest.get("schema") == "pareidolia.material-interface-fixed-point":
            final = manifest.get("finalSurface", {})
            final_path = Path(str(final.get("manifestPath", ""))).resolve()
            if not final_path.is_file() or sha256_file(final_path) != final.get(
                "manifestSha256"
            ):
                raise ValueError("fixed-point final signed surface is unavailable or changed")
            return final_path
        return candidate
    stems = (
        "material-interface-frontier-bundles-v1",
        "material-interface-boundary-bridging-v1",
        MATERIAL_SURFACE_GROWTH_STEM,
        MATERIAL_SURFACE_GRAPH_STEM,
    )
    for stem in stems:
        candidate = value / f"{stem}.json"
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(f"signed material surface artifact is unavailable at {value}")


def _root_graph_manifest(
    surface_path: Path, surface: Mapping[str, Any]
) -> tuple[Path, dict[str, Any]]:
    if surface.get("schema") == MATERIAL_SURFACE_GRAPH_SCHEMA:
        return surface_path, dict(surface)
    graph_identity = surface.get("identity", {}).get("seedGraph")
    if not isinstance(graph_identity, Mapping):
        raise ValueError("derived signed surface omits its root seed graph")
    graph_path = Path(str(graph_identity["manifestPath"])).resolve()
    if sha256_file(graph_path) != graph_identity["manifestSha256"]:
        raise ValueError("root seed graph changed after derived surface creation")
    graph = json.loads(graph_path.read_text())
    if (
        graph.get("schema") != MATERIAL_SURFACE_GRAPH_SCHEMA
        or graph.get("state") != "complete"
    ):
        raise ValueError("root seed graph is not complete")
    return graph_path, graph


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def _neighbor_offsets(radius: float) -> np.ndarray:
    reach = int(math.ceil(radius))
    values = [
        (dx, dy, dz)
        for dz in range(-reach, reach + 1)
        for dy in range(-reach, reach + 1)
        for dx in range(-reach, reach + 1)
        if (dx, dy, dz) != (0, 0, 0)
        and math.sqrt(dx * dx + dy * dy + dz * dz) <= radius + 1.0e-9
    ]
    return np.asarray(values, dtype=np.int64)


def _normalize(value: np.ndarray) -> np.ndarray:
    length = float(np.linalg.norm(value))
    return value / max(length, 1.0e-9)


def _support_geometry(
    candidate_position: np.ndarray,
    candidate_signed_normal: np.ndarray,
    support_position: np.ndarray,
    support_sheet_normal: np.ndarray,
    support_signed_normal: np.ndarray,
    *,
    sampling_stride_voxels: int,
) -> dict[str, Any]:
    """Fit one local signed tangent plane and measure enclosure around a candidate."""

    axial = np.asarray(support_sheet_normal, dtype=np.float64).copy()
    reference = axial[0]
    axial[np.einsum("ij,j->i", axial, reference) < 0.0] *= -1.0
    plane_normal = _normalize(np.sum(axial, axis=0))
    signed_mean = np.sum(np.asarray(support_signed_normal, dtype=np.float64), axis=0)
    if float(np.dot(plane_normal, signed_mean)) < 0.0:
        plane_normal *= -1.0

    delta = (
        np.asarray(support_position, dtype=np.float64)
        - np.asarray(candidate_position, dtype=np.float64)[None, :]
    ) / float(sampling_stride_voxels)
    height = np.einsum("ij,j->i", delta, plane_normal)
    tangent = delta - height[:, None] * plane_normal[None, :]
    first, second = _plane_basis(plane_normal[None, :])
    coordinate = np.column_stack(
        (
            np.einsum("ij,j->i", tangent, first[0]),
            np.einsum("ij,j->i", tangent, second[0]),
        )
    )
    second_moment = coordinate.T @ coordinate / max(len(coordinate), 1)
    eigenvalue = np.linalg.eigvalsh(second_moment)
    rank_ratio = float(eigenvalue[0] / max(eigenvalue[1], 1.0e-9))
    angle = np.sort(np.mod(np.arctan2(coordinate[:, 1], coordinate[:, 0]), 2.0 * math.pi))
    gap = np.diff(np.concatenate((angle, angle[:1] + 2.0 * math.pi)))
    maximum_gap = math.degrees(float(np.max(gap)))
    signed_cosine = float(
        np.dot(_normalize(np.asarray(candidate_signed_normal, dtype=np.float64)), plane_normal)
    )
    signed_angle = math.degrees(math.acos(float(np.clip(signed_cosine, -1.0, 1.0))))
    return {
        "sheetNormalXYZ": plane_normal,
        "maximumSupportAngularGapDegrees": maximum_gap,
        "tangentRankRatio": rank_ratio,
        "candidatePlaneResidualSamplingSteps": abs(float(np.mean(height))),
        "supportHeightStandardDeviationSamplingSteps": float(np.std(height)),
        "signedNormalDisagreementDegrees": signed_angle,
    }


def _proposal_score(
    *,
    evidence: float,
    support_count: int,
    dominance: float,
    angular_gap_degrees: float,
    rank_ratio: float,
    plane_residual: float,
    signed_normal_degrees: float,
    settings: MaterialSurfaceGrowthSettings,
) -> float:
    enclosure = max(0.0, 1.0 - angular_gap_degrees / 360.0)
    geometry = math.exp(
        -0.5
        * (
            plane_residual
            / max(settings.maximum_candidate_plane_residual_sampling_steps, 1.0e-9)
        )
        ** 2
    )
    signed = max(0.0, math.cos(math.radians(signed_normal_degrees)))
    return float(
        max(evidence, 0.05)
        * dominance
        * min(support_count / 8.0, 1.0)
        * enclosure
        * math.sqrt(max(rank_ratio, 0.0))
        * geometry
        * signed
    )


def _build_neighbor_interface_matrix(
    candidate_index: np.ndarray,
    key: np.ndarray,
    grid: np.ndarray,
    processing_shape_xyz: np.ndarray,
    offsets: np.ndarray,
) -> np.ndarray:
    candidate_key = key[candidate_index]
    neighbor = np.full((len(candidate_index), len(offsets)), -1, dtype=np.int32)
    for offset_index, offset in enumerate(offsets):
        query = candidate_key + offset[None, :]
        valid = np.all(
            (query >= 0) & (query < processing_shape_xyz[None, :]), axis=1
        )
        if not np.any(valid):
            continue
        selected = query[valid]
        neighbor[valid, offset_index] = grid[
            selected[:, 2], selected[:, 1], selected[:, 0]
        ]
    return neighbor


def run_material_surface_growth(
    interface_root: str | Path,
    macro_root: str | Path,
    graph_root: str | Path,
    output_path: str | Path,
    *,
    settings: MaterialSurfaceGrowthSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve(interface_root, MATERIAL_INTERFACE_STEM)
    macro_path = _resolve(macro_root, MACRO_ORIENTATION_STEM)
    seed_surface_path = _resolve_seed_surface(graph_root)
    interfaces = json.loads(interface_path.read_text())
    macro = json.loads(macro_path.read_text())
    seed_surface = json.loads(seed_surface_path.read_text())
    if interfaces.get("schema") != MATERIAL_INTERFACE_SCHEMA or interfaces.get("state") != "complete":
        raise ValueError("interior growth requires complete material interfaces")
    if macro.get("schema") != MACRO_ORIENTATION_SCHEMA or macro.get("state") != "complete":
        raise ValueError("interior growth requires a complete macro orientation field")
    supported_surface_schemas = {
        MATERIAL_SURFACE_GRAPH_SCHEMA,
        MATERIAL_SURFACE_GROWTH_SCHEMA,
        "pareidolia.material-interface-boundary-bridging",
    }
    if (
        seed_surface.get("schema") not in supported_surface_schemas
        or seed_surface.get("state") != "complete"
    ):
        raise ValueError("interior growth requires a complete signed surface artifact")
    if macro["identity"]["interfaces"]["manifestSha256"] != sha256_file(interface_path):
        raise ValueError("macro field was not derived from the supplied interfaces")
    seed_identity = seed_surface["identity"]
    if seed_identity["interfaces"]["manifestSha256"] != sha256_file(interface_path):
        raise ValueError("seed surface was not derived from the supplied interfaces")
    if seed_identity["macroOrientation"]["manifestSha256"] != sha256_file(macro_path):
        raise ValueError("seed surface was not derived from the supplied macro field")

    resolved = settings or MaterialSurfaceGrowthSettings()
    graph_path, graph = _root_graph_manifest(seed_surface_path, seed_surface)
    graph_settings = MaterialSurfaceGraphSettings(**graph["identity"]["settings"])
    identity: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_GROWTH_SCHEMA,
        "version": MATERIAL_SURFACE_GROWTH_VERSION,
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
        "seedGraph": {
            "manifestPath": str(graph_path),
            "manifestSha256": sha256_file(graph_path),
            "dataSha256": graph["data"]["sha256"],
        },
        "seedSurface": {
            "schema": str(seed_surface["schema"]),
            "manifestPath": str(seed_surface_path),
            "manifestSha256": sha256_file(seed_surface_path),
            "dataSha256": seed_surface["data"]["sha256"],
        },
        "settings": resolved.record(),
        "inheritedStratumGuard": {
            "tangentColumnWidthSamplingSteps": graph_settings.tangent_column_width_sampling_steps,
            "maximumColumnDepthRangeSamplingSteps": graph_settings.maximum_column_depth_range_sampling_steps,
        },
        "implementationSha256": {
            "material_surface_growth.py": sha256_file(Path(__file__)),
            "material_surface_graph.py": sha256_file(
                Path(__file__).with_name("material_surface_graph.py")
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_SURFACE_GROWTH_STEM}.json"
    data_path = output / f"{MATERIAL_SURFACE_GROWTH_STEM}.npz"
    preview_path = output / "material-surface-growth-cross-sections.png"
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
    seed_arrays = _load_npz(seed_surface_path, seed_surface)
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
    if any(len(value) != len(position) for value in (signed_normal, key, evidence, sample_bin)):
        raise ValueError("interface and macro sample arrays are not aligned")
    raw_macro_angle = np.degrees(
        np.arccos(
            np.clip(np.abs(np.einsum("ij,ij->i", signed_normal, macro_normal)), 0.0, 1.0)
        )
    )

    seed_interface = np.asarray(seed_arrays["interfaceIndex"], dtype=np.int64)
    seed_component = np.asarray(seed_arrays["componentId"], dtype=np.int32)
    seed_pre_component = (
        np.asarray(seed_arrays["preCollisionComponentId"], dtype=np.int32)
        if "preCollisionComponentId" in seed_arrays
        else seed_component.copy()
    )
    if len(seed_interface) != len(seed_component):
        raise ValueError("seed graph node arrays have inconsistent lengths")
    if len(np.unique(seed_interface)) != len(seed_interface):
        raise ValueError("seed graph repeats interface samples")
    component_count = int(np.max(seed_component)) + 1 if len(seed_component) else 0
    seed_physical_label = (
        np.asarray(seed_arrays["physicalSheetLabel"], dtype=np.int32)
        if "physicalSheetLabel" in seed_arrays
        else np.full(len(seed_interface), -1, dtype=np.int32)
    )
    seed_physical_side = (
        np.asarray(seed_arrays["physicalBoundarySide"], dtype=np.uint8)
        if "physicalBoundarySide" in seed_arrays
        else np.full(len(seed_interface), 255, dtype=np.uint8)
    )
    seed_physical_anchor = (
        np.asarray(seed_arrays["physicalSeedAnchor"], dtype=np.uint8)
        if "physicalSeedAnchor" in seed_arrays
        else np.zeros(len(seed_interface), dtype=np.uint8)
    )
    if np.any((seed_physical_label >= 0) != (seed_physical_side <= 1)):
        raise ValueError(
            "seed surface physical sheet and boundary-side identities disagree"
        )
    component_physical_label = np.full(component_count, -1, dtype=np.int32)
    component_physical_side = np.full(component_count, 255, dtype=np.uint8)
    for component_id in range(component_count):
        member = seed_component == component_id
        identities = np.unique(
            2 * seed_physical_label[member & (seed_physical_label >= 0)]
            + seed_physical_side[member & (seed_physical_label >= 0)].astype(
                np.int32
            )
        )
        if len(identities) > 1:
            raise ValueError(
                "seed surface component contains multiple physical boundary-face identities"
            )
        if len(identities) == 1:
            identity_value = int(identities[0])
            component_physical_label[component_id] = identity_value // 2
            component_physical_side[component_id] = identity_value % 2
    component_physical_anchor_count = np.bincount(
        seed_component,
        weights=(seed_physical_anchor == 1).astype(np.int32),
        minlength=component_count,
    ).astype(np.int32)
    component_by_interface = np.full(len(position), -1, dtype=np.int32)
    component_by_interface[seed_interface] = seed_component
    growth_round = np.full(len(position), -1, dtype=np.int16)
    if "growthRound" in seed_arrays:
        growth_round[seed_interface] = np.asarray(
            seed_arrays["growthRound"], dtype=np.int16
        )
    else:
        growth_round[seed_interface] = 0
    prior_growth_round = growth_round[growth_round > 0]
    growth_round_offset = (
        int(np.max(prior_growth_round)) if len(prior_growth_round) else 0
    )
    sheet_normal = macro_normal.copy()
    if "sheetNormalXYZ" in seed_arrays:
        sheet_normal[seed_interface] = np.asarray(
            seed_arrays["sheetNormalXYZ"], dtype=np.float64
        )

    processing_shape = np.asarray(
        interfaces["geometry"]["processingShapeSamplingXYZ"], dtype=np.int64
    )
    grid = np.full(tuple(processing_shape[::-1]), -1, dtype=np.int32)
    existing = grid[key[:, 2], key[:, 1], key[:, 0]]
    if np.any(existing >= 0):
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

    column_state: list[dict[int, tuple[float, float]]] = [
        {} for _ in range(component_count)
    ]
    for interface_index, component_id in zip(seed_interface, seed_component):
        column_id = int(tangent_column[interface_index])
        depth = float(normal_depth[interface_index])
        current = column_state[int(component_id)].get(column_id)
        if current is None:
            column_state[int(component_id)][column_id] = (depth, depth)
        else:
            low = min(current[0], depth)
            high = max(current[1], depth)
            if high - low > graph_settings.maximum_column_depth_range_sampling_steps:
                raise ValueError("seed surface violates its tangent-column stratum guard")
            column_state[int(component_id)][column_id] = (low, high)

    metric_arrays = {
        "growthSupportCount": np.zeros(len(position), dtype=np.int16),
        "growthComponentDominance": np.full(len(position), np.nan, dtype=np.float32),
        "growthSupportAngularGapDegrees": np.full(len(position), np.nan, dtype=np.float32),
        "growthTangentRankRatio": np.full(len(position), np.nan, dtype=np.float32),
        "growthPlaneResidualSamplingSteps": np.full(len(position), np.nan, dtype=np.float32),
        "growthSupportHeightStandardDeviationSamplingSteps": np.full(
            len(position), np.nan, dtype=np.float32
        ),
        "growthSignedNormalDisagreementDegrees": np.full(
            len(position), np.nan, dtype=np.float32
        ),
        "growthScore": np.full(len(position), np.nan, dtype=np.float32),
    }
    for name, values in metric_arrays.items():
        if name in seed_arrays:
            values[seed_interface] = np.asarray(seed_arrays[name], dtype=values.dtype)
    bridge_metadata_arrays: dict[str, np.ndarray] = {
        "bridgeBundleId": np.full(len(position), -1, dtype=np.int32),
        "bridgeScore": np.full(len(position), np.nan, dtype=np.float32),
        "bridgeOppositionDegrees": np.full(
            len(position), np.nan, dtype=np.float32
        ),
        "bridgeSupportAngularGapDegrees": np.full(
            len(position), np.nan, dtype=np.float32
        ),
        "bridgePlaneResidualSamplingSteps": np.full(
            len(position), np.nan, dtype=np.float32
        ),
    }
    for name, values in bridge_metadata_arrays.items():
        if name in seed_arrays:
            values[seed_interface] = np.asarray(seed_arrays[name], dtype=values.dtype)
    offsets = _neighbor_offsets(resolved.neighbor_radius_sampling_steps)
    growth_edge_interface_first: list[int] = []
    growth_edge_interface_second: list[int] = []
    growth_edge_score: list[float] = []
    accepted_interface_order: list[int] = []
    round_summaries: list[dict[str, Any]] = []
    for round_index in range(1, resolved.maximum_growth_rounds + 1):
        candidate_index = np.flatnonzero(
            (component_by_interface < 0)
            & (evidence >= resolved.minimum_local_evidence)
            & macro_trusted
            & (macro_confidence >= graph_settings.minimum_macro_confidence)
            & (
                raw_macro_angle
                <= graph_settings.maximum_raw_to_macro_normal_degrees
            )
        )
        if not len(candidate_index):
            break
        neighbor_interface = _build_neighbor_interface_matrix(
            candidate_index, key, grid, processing_shape, offsets
        )
        neighbor_component = np.full_like(neighbor_interface, -1)
        exists = neighbor_interface >= 0
        neighbor_component[exists] = component_by_interface[
            neighbor_interface[exists]
        ]
        active_neighbor = neighbor_component >= 0
        active_count = np.sum(active_neighbor, axis=1)
        evaluated_row = np.flatnonzero(
            active_count >= resolved.minimum_dominant_support_samples
        )
        accounting = {
            "round": round_index,
            "inactiveEvidenceCandidateCount": int(len(candidate_index)),
            "candidatesWithAnyActiveNeighbor": int(np.count_nonzero(active_count)),
            "candidatesWithMinimumActiveNeighbors": int(len(evaluated_row)),
            "rejectedLowDominantSupport": 0,
            "rejectedLowComponentDominance": 0,
            "rejectedOpenSupport": 0,
            "rejectedTangentRank": 0,
            "rejectedPlaneResidual": 0,
            "rejectedSupportHeightSpread": 0,
            "rejectedSignedNormal": 0,
            "rejectedStratumCollision": 0,
            "accepted": 0,
        }
        proposals: list[dict[str, Any]] = []
        evaluated_metrics: dict[str, list[float]] = {
            "supportCount": [],
            "componentDominance": [],
            "supportAngularGapDegrees": [],
            "tangentRankRatio": [],
            "planeResidualSamplingSteps": [],
            "supportHeightStandardDeviationSamplingSteps": [],
            "signedNormalDisagreementDegrees": [],
            "score": [],
        }
        current_component_sizes = np.bincount(
            component_by_interface[component_by_interface >= 0],
            minlength=component_count,
        )
        for row in evaluated_row:
            active = active_neighbor[row]
            components, counts = np.unique(
                neighbor_component[row, active], return_counts=True
            )
            maximum = int(np.max(counts))
            tied = components[counts == maximum]
            if len(tied) > 1:
                dominant_component = int(
                    tied[np.argmax(current_component_sizes[tied])]
                )
            else:
                dominant_component = int(tied[0])
            support_mask = (
                neighbor_component[row] == dominant_component
            )
            support = neighbor_interface[row, support_mask].astype(np.int64)
            if len(support) < resolved.minimum_dominant_support_samples:
                accounting["rejectedLowDominantSupport"] += 1
                continue
            dominance = len(support) / max(int(active_count[row]), 1)
            if dominance < resolved.minimum_component_dominance_fraction:
                accounting["rejectedLowComponentDominance"] += 1
                continue
            interface_index = int(candidate_index[row])
            geometry = _support_geometry(
                position[interface_index],
                signed_normal[interface_index],
                position[support],
                sheet_normal[support],
                signed_normal[support],
                sampling_stride_voxels=stride,
            )
            angular_gap = float(geometry["maximumSupportAngularGapDegrees"])
            rank_ratio = float(geometry["tangentRankRatio"])
            plane_residual = float(
                geometry["candidatePlaneResidualSamplingSteps"]
            )
            height_std = float(
                geometry["supportHeightStandardDeviationSamplingSteps"]
            )
            signed_angle = float(geometry["signedNormalDisagreementDegrees"])
            score = _proposal_score(
                evidence=float(evidence[interface_index]),
                support_count=len(support),
                dominance=dominance,
                angular_gap_degrees=angular_gap,
                rank_ratio=rank_ratio,
                plane_residual=plane_residual,
                signed_normal_degrees=signed_angle,
                settings=resolved,
            )
            for name, value in (
                ("supportCount", len(support)),
                ("componentDominance", dominance),
                ("supportAngularGapDegrees", angular_gap),
                ("tangentRankRatio", rank_ratio),
                ("planeResidualSamplingSteps", plane_residual),
                ("supportHeightStandardDeviationSamplingSteps", height_std),
                ("signedNormalDisagreementDegrees", signed_angle),
                ("score", score),
            ):
                evaluated_metrics[name].append(float(value))
            if angular_gap > resolved.maximum_support_angular_gap_degrees + 1.0e-6:
                accounting["rejectedOpenSupport"] += 1
                continue
            if rank_ratio < resolved.minimum_tangent_rank_ratio:
                accounting["rejectedTangentRank"] += 1
                continue
            if plane_residual > resolved.maximum_candidate_plane_residual_sampling_steps:
                accounting["rejectedPlaneResidual"] += 1
                continue
            if height_std > resolved.maximum_support_height_standard_deviation_sampling_steps:
                accounting["rejectedSupportHeightSpread"] += 1
                continue
            if signed_angle > resolved.maximum_signed_normal_disagreement_degrees:
                accounting["rejectedSignedNormal"] += 1
                continue
            proposals.append(
                {
                    "interfaceIndex": interface_index,
                    "component": dominant_component,
                    "support": support,
                    "supportCount": len(support),
                    "dominance": dominance,
                    "geometry": geometry,
                    "score": score,
                }
            )

        proposals.sort(
            key=lambda value: (-float(value["score"]), int(value["interfaceIndex"]))
        )
        accepted: list[dict[str, Any]] = []
        for proposal in proposals:
            interface_index = int(proposal["interfaceIndex"])
            component_id = int(proposal["component"])
            column_id = int(tangent_column[interface_index])
            depth = float(normal_depth[interface_index])
            current = column_state[component_id].get(column_id)
            if current is not None and (
                max(current[1], depth) - min(current[0], depth)
                > graph_settings.maximum_column_depth_range_sampling_steps
            ):
                accounting["rejectedStratumCollision"] += 1
                continue
            column_state[component_id][column_id] = (
                depth if current is None else min(current[0], depth),
                depth if current is None else max(current[1], depth),
            )
            component_by_interface[interface_index] = component_id
            growth_round[interface_index] = growth_round_offset + round_index
            geometry = proposal["geometry"]
            sheet_normal[interface_index] = geometry["sheetNormalXYZ"]
            metric_arrays["growthSupportCount"][interface_index] = int(
                proposal["supportCount"]
            )
            metric_arrays["growthComponentDominance"][interface_index] = float(
                proposal["dominance"]
            )
            metric_arrays["growthSupportAngularGapDegrees"][interface_index] = float(
                geometry["maximumSupportAngularGapDegrees"]
            )
            metric_arrays["growthTangentRankRatio"][interface_index] = float(
                geometry["tangentRankRatio"]
            )
            metric_arrays["growthPlaneResidualSamplingSteps"][interface_index] = float(
                geometry["candidatePlaneResidualSamplingSteps"]
            )
            metric_arrays[
                "growthSupportHeightStandardDeviationSamplingSteps"
            ][interface_index] = float(
                geometry["supportHeightStandardDeviationSamplingSteps"]
            )
            metric_arrays[
                "growthSignedNormalDisagreementDegrees"
            ][interface_index] = float(geometry["signedNormalDisagreementDegrees"])
            metric_arrays["growthScore"][interface_index] = float(proposal["score"])
            accepted_interface_order.append(interface_index)
            for support_interface in proposal["support"]:
                growth_edge_interface_first.append(interface_index)
                growth_edge_interface_second.append(int(support_interface))
                growth_edge_score.append(float(proposal["score"]))
            accepted.append(proposal)
        accounting["accepted"] = len(accepted)
        round_summaries.append(
            {
                **accounting,
                "evaluatedDistributions": {
                    name: _percentiles(np.asarray(values, dtype=np.float64))
                    for name, values in evaluated_metrics.items()
                },
            }
        )
        if len(accepted) < resolved.minimum_round_additions:
            break

    active_interface = np.concatenate(
        (
            seed_interface.astype(np.int64),
            np.asarray(accepted_interface_order, dtype=np.int64),
        )
    )
    base_component = component_by_interface[active_interface]
    base_size = np.bincount(base_component, minlength=component_count)
    priority = (
        component_physical_anchor_count
        >= graph_settings.minimum_physical_anchor_samples_for_priority
    )
    component_order = np.lexsort(
        (np.arange(component_count), -base_size, ~priority)
    )
    rank_by_component = np.empty(component_count, dtype=np.int32)
    rank_by_component[component_order] = np.arange(component_count, dtype=np.int32)
    component = rank_by_component[base_component]
    component_size = base_size[component_order]
    component_physical_label = component_physical_label[component_order]
    component_physical_side = component_physical_side[component_order]
    component_physical_anchor_count = component_physical_anchor_count[
        component_order
    ]
    active_physical_label = component_physical_label[component]
    active_physical_side = component_physical_side[component]
    physical_anchor_by_interface = np.zeros(len(position), dtype=np.uint8)
    physical_anchor_by_interface[seed_interface] = seed_physical_anchor
    node_by_interface = np.full(len(position), -1, dtype=np.int32)
    node_by_interface[active_interface] = np.arange(len(active_interface), dtype=np.int32)

    seed_first_node = np.asarray(seed_arrays["edgeFirstNode"], dtype=np.int64)
    seed_second_node = np.asarray(seed_arrays["edgeSecondNode"], dtype=np.int64)
    seed_edge_interface_first = seed_interface[seed_first_node]
    seed_edge_interface_second = seed_interface[seed_second_node]
    growth_first = np.asarray(growth_edge_interface_first, dtype=np.int64)
    growth_second = np.asarray(growth_edge_interface_second, dtype=np.int64)
    edge_interface_first = np.concatenate((seed_edge_interface_first, growth_first))
    edge_interface_second = np.concatenate((seed_edge_interface_second, growth_second))
    edge_first = node_by_interface[edge_interface_first]
    edge_second = node_by_interface[edge_interface_second]
    seed_score = np.asarray(seed_arrays["edgeScore"], dtype=np.float32)
    edge_score = np.concatenate(
        (seed_score, np.asarray(growth_edge_score, dtype=np.float32))
    )
    seed_edge_kind = (
        np.asarray(seed_arrays["edgeKind"], dtype=np.uint8)
        if "edgeKind" in seed_arrays
        else np.zeros(len(seed_score), dtype=np.uint8)
    )
    edge_kind = np.concatenate(
        (seed_edge_kind, np.ones(len(growth_edge_score), dtype=np.uint8))
    )
    pre_component_by_interface = np.full(len(position), -1, dtype=np.int32)
    pre_component_by_interface[seed_interface] = seed_pre_component
    grown = growth_round[active_interface] > 0
    pre_component_by_interface[active_interface[grown]] = base_component[grown]
    arrays = {
        "interfaceIndex": active_interface.astype(np.int32),
        "positionXYZ": position[active_interface].astype(np.float32),
        "signedNormalXYZ": signed_normal[active_interface].astype(np.float32),
        "sheetNormalXYZ": sheet_normal[active_interface].astype(np.float32),
        "macroNormalXYZ": macro_normal[active_interface].astype(np.float32),
        "macroOrientationConfidence": macro_confidence[active_interface].astype(np.float32),
        "orientationSource": orientation_source[active_interface].astype(np.uint8),
        "physicalSeedAnchor": physical_anchor_by_interface[active_interface],
        "physicalSheetLabel": active_physical_label.astype(np.int32),
        "physicalBoundarySide": active_physical_side.astype(np.uint8),
        "componentPhysicalSheetLabel": component_physical_label,
        "componentPhysicalBoundarySide": component_physical_side,
        "componentPhysicalAnchorCount": component_physical_anchor_count,
        "rawToMacroNormalDegrees": raw_macro_angle[active_interface].astype(np.float32),
        "localEvidenceScore": evidence[active_interface].astype(np.float32),
        "seedComponentId": base_component.astype(np.int32),
        "preCollisionComponentId": pre_component_by_interface[active_interface],
        "componentId": component.astype(np.int32),
        "growthRound": growth_round[active_interface],
        **{
            name: values[active_interface]
            for name, values in bridge_metadata_arrays.items()
        },
        "tangentColumnId": tangent_column[active_interface],
        "normalDepthSamplingSteps": normal_depth[active_interface],
        **{
            name: values[active_interface]
            for name, values in metric_arrays.items()
        },
        "edgeFirstNode": edge_first.astype(np.int32),
        "edgeSecondNode": edge_second.astype(np.int32),
        "edgeScore": edge_score,
        "edgeKind": edge_kind,
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
    added_count = len(accepted_interface_order)
    substantial = component_size >= graph_settings.minimum_component_samples_for_preview
    added_interface = np.asarray(accepted_interface_order, dtype=np.int64)
    seed_counts = seed_surface.get("counts", {})
    initial_component_count = int(
        seed_counts.get(
            "initialComponentCount",
            seed_counts.get("componentCountBefore", component_count),
        )
    )
    prior_component_merge_count = int(
        seed_counts.get("componentMergeCount", 0)
    )
    cumulative_bridge_candidate_count = int(
        np.count_nonzero(
            bridge_metadata_arrays["bridgeBundleId"][active_interface] >= 0
        )
    )
    payload: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_GROWTH_SCHEMA,
        "version": MATERIAL_SURFACE_GROWTH_VERSION,
        "state": "complete",
        "identity": identity,
        "source": interfaces["source"],
        "geometry": interfaces["geometry"],
        "counts": {
            "interfaceSampleCount": int(len(position)),
            "seedNodeCount": int(len(seed_interface)),
            "grownNodeCount": int(added_count),
            "totalInteriorGrownNodeCount": int(
                np.count_nonzero(growth_round[active_interface] > 0)
            ),
            "activeNodeCount": int(len(active_interface)),
            "seedNodeFraction": round(len(seed_interface) / max(len(position), 1), 6),
            "activeNodeFraction": round(len(active_interface) / max(len(position), 1), 6),
            "remainingInterfaceSampleCount": int(len(position) - len(active_interface)),
            "componentCount": int(component_count),
            "physicallyAnchoredComponentCount": int(
                np.count_nonzero(component_physical_label >= 0)
            ),
            "nodesInPhysicallyAnchoredComponents": int(
                np.sum(component_size[component_physical_label >= 0])
            ),
            "physicalAnchorNodeCount": int(
                np.count_nonzero(
                    physical_anchor_by_interface[active_interface] == 1
                )
            ),
            "initialComponentCount": initial_component_count,
            "priorComponentMergeCount": prior_component_merge_count,
            "componentMergeCount": prior_component_merge_count,
            "bridgeCandidateNodeCount": cumulative_bridge_candidate_count,
            "retainedSeedEdgeCount": int(len(seed_score)),
            "growthEdgeCount": int(len(growth_edge_score)),
            "retainedEdgeCount": int(len(edge_score)),
            "completedGrowthRounds": int(
                max((record["round"] for record in round_summaries), default=0)
            ),
            "componentsAtLeast8Nodes": int(np.count_nonzero(component_size >= 8)),
            "componentsAtLeast32Nodes": int(np.count_nonzero(component_size >= 32)),
            "componentsAtLeast128Nodes": int(np.count_nonzero(component_size >= 128)),
            "nodesInPreviewSizedComponents": int(np.sum(component_size[substantial])),
            "largestComponentSizes": [int(value) for value in component_size[:32]],
        },
        "growthRounds": round_summaries,
        "distributions": {
            "componentSize": _percentiles(component_size),
            "addedLocalEvidenceScore": _percentiles(evidence[added_interface]),
            "addedSupportCount": _percentiles(
                metric_arrays["growthSupportCount"][added_interface]
            ),
            "addedSupportAngularGapDegrees": _percentiles(
                metric_arrays["growthSupportAngularGapDegrees"][added_interface]
            ),
            "addedTangentRankRatio": _percentiles(
                metric_arrays["growthTangentRankRatio"][added_interface]
            ),
            "addedPlaneResidualSamplingSteps": _percentiles(
                metric_arrays["growthPlaneResidualSamplingSteps"][added_interface]
            ),
            "addedSignedNormalDisagreementDegrees": _percentiles(
                metric_arrays["growthSignedNormalDisagreementDegrees"][added_interface]
            ),
            "addedScore": _percentiles(metric_arrays["growthScore"][added_interface]),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {"componentCrossSections": preview_path.name},
        "method": {
            "candidatePool": "unused immutable signed material-interface samples",
            "seedSurfaceSchema": str(seed_surface["schema"]),
            "assignment": "one dominant existing component; components are never merged",
            "interiorTest": (
                "multi-neighbor tangent support with no open angular half-plane, "
                "two-dimensional support rank, and local plane fit"
            ),
            "signedFaceTest": "candidate air-to-material normal agrees with the supported face",
            "transitiveLayerGuard": (
                "inherits the seed graph tangent-column normal-depth interval"
            ),
            "outwardFrontierGrowth": False,
            "componentMerging": False,
        },
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
