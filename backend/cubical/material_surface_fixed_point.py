from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .macro_orientation import MACRO_ORIENTATION_STEM
from .material_interface import MATERIAL_INTERFACE_STEM
from .material_surface_bridging import (
    MATERIAL_SURFACE_BRIDGING_STEM,
    MaterialSurfaceBridgingSettings,
    run_material_surface_bridging,
)
from .material_surface_growth import (
    MATERIAL_SURFACE_GROWTH_STEM,
    MaterialSurfaceGrowthSettings,
    _resolve_seed_surface,
    run_material_surface_growth,
)


MATERIAL_SURFACE_FIXED_POINT_SCHEMA = "pareidolia.material-interface-fixed-point"
MATERIAL_SURFACE_FIXED_POINT_VERSION = 1
MATERIAL_SURFACE_FIXED_POINT_STEM = "material-surface-fixed-point-v1"


@dataclass(frozen=True, slots=True)
class MaterialSurfaceFixedPointSettings:
    maximum_cycles: int = 8

    def __post_init__(self) -> None:
        if self.maximum_cycles < 1:
            raise ValueError("fixed-point reconstruction requires at least one cycle")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _resolve(root: str | Path, stem: str) -> Path:
    value = Path(root).resolve()
    return value if value.is_file() else value / f"{stem}.json"


def _audit_final_surface(
    manifest_path: Path, manifest: dict[str, Any]
) -> dict[str, Any]:
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("fixed-point final surface data changed after creation")
    with np.load(data_path, allow_pickle=False) as stored:
        interface_index = np.asarray(stored["interfaceIndex"], dtype=np.int64)
        component = np.asarray(stored["componentId"], dtype=np.int64)
        tangent_column = np.asarray(stored["tangentColumnId"], dtype=np.int64)
        normal_depth = np.asarray(
            stored["normalDepthSamplingSteps"], dtype=np.float64
        )
        edge_first = np.asarray(stored["edgeFirstNode"], dtype=np.int64)
        edge_second = np.asarray(stored["edgeSecondNode"], dtype=np.int64)
        growth_round = np.asarray(stored["growthRound"], dtype=np.int64)
        bridge_bundle = np.asarray(stored["bridgeBundleId"], dtype=np.int64)
        physical_label = (
            np.asarray(stored["physicalSheetLabel"], dtype=np.int64)
            if "physicalSheetLabel" in stored
            else np.full(len(component), -1, dtype=np.int64)
        )
        physical_side = (
            np.asarray(stored["physicalBoundarySide"], dtype=np.uint8)
            if "physicalBoundarySide" in stored
            else np.full(len(component), 255, dtype=np.uint8)
        )
    node_count = len(component)
    aligned = all(
        len(value) == node_count
        for value in (
            interface_index,
            tangent_column,
            normal_depth,
            growth_round,
            bridge_bundle,
            physical_label,
            physical_side,
        )
    )
    if not aligned:
        raise ValueError("fixed-point final node arrays have inconsistent lengths")
    if len(np.unique(interface_index)) != node_count:
        raise ValueError("fixed-point final surface repeats an interface sample")
    if len(edge_first) != len(edge_second) or (
        len(edge_first)
        and (
            int(np.min(edge_first)) < 0
            or int(np.min(edge_second)) < 0
            or int(np.max(edge_first)) >= node_count
            or int(np.max(edge_second)) >= node_count
        )
    ):
        raise ValueError("fixed-point final edge arrays are invalid")
    cross_component_edges = int(
        np.count_nonzero(component[edge_first] != component[edge_second])
    )
    order = np.lexsort((tangent_column, component))
    ordered_component = component[order]
    ordered_column = tangent_column[order]
    ordered_depth = normal_depth[order]
    start = (
        np.concatenate(
            (
                np.zeros(1, dtype=np.int64),
                1
                + np.flatnonzero(
                    (ordered_component[1:] != ordered_component[:-1])
                    | (ordered_column[1:] != ordered_column[:-1])
                ),
            )
        )
        if node_count
        else np.empty(0, dtype=np.int64)
    )
    low = np.minimum.reduceat(ordered_depth, start) if len(start) else np.empty(0)
    high = np.maximum.reduceat(ordered_depth, start) if len(start) else np.empty(0)
    depth_range = high - low
    maximum_depth_range = float(
        manifest["identity"]["inheritedStratumGuard"][
            "maximumColumnDepthRangeSamplingSteps"
        ]
    )
    depth_violations = int(
        np.count_nonzero(depth_range > maximum_depth_range + 1.0e-6)
    )
    component_count = len(np.unique(component))
    invalid_physical_identity_nodes = int(
        np.count_nonzero((physical_label >= 0) != (physical_side <= 1))
    )
    physical_identity_violations = 0
    for component_id in np.unique(component):
        member = (component == component_id) & (physical_label >= 0)
        identities = np.unique(
            2 * physical_label[member] + physical_side[member].astype(np.int64)
        )
        physical_identity_violations += int(len(identities) > 1)
    expected_component_count = int(manifest["counts"]["componentCount"])
    if (
        cross_component_edges
        or depth_violations
        or invalid_physical_identity_nodes
        or physical_identity_violations
        or component_count != expected_component_count
    ):
        raise RuntimeError("fixed-point final surface failed structural invariants")
    return {
        "nodeCount": node_count,
        "uniqueInterfaceSampleCount": int(len(np.unique(interface_index))),
        "edgeCount": int(len(edge_first)),
        "crossComponentEdgeCount": cross_component_edges,
        "componentCount": int(component_count),
        "componentCountMatchesManifest": component_count == expected_component_count,
        "tangentColumnIntervalCount": int(len(depth_range)),
        "maximumColumnDepthRangeSamplingSteps": round(
            float(np.max(depth_range)) if len(depth_range) else 0.0, 6
        ),
        "allowedColumnDepthRangeSamplingSteps": maximum_depth_range,
        "columnDepthRangeViolationCount": depth_violations,
        "physicalIdentityViolationComponentCount": physical_identity_violations,
        "invalidPhysicalIdentityNodeCount": invalid_physical_identity_nodes,
        "physicallyAnchoredNodeCount": int(
            np.count_nonzero(physical_label >= 0)
        ),
        "interiorGrowthNodeCount": int(np.count_nonzero(growth_round > 0)),
        "bridgeCandidateNodeCount": int(np.count_nonzero(bridge_bundle >= 0)),
        "bridgeBundleCount": int(
            len(np.unique(bridge_bundle[bridge_bundle >= 0]))
        ),
        "passed": True,
    }


def run_material_surface_fixed_point(
    interface_root: str | Path,
    macro_root: str | Path,
    seed_surface_root: str | Path,
    output_path: str | Path,
    *,
    settings: MaterialSurfaceFixedPointSettings | None = None,
    growth_settings: MaterialSurfaceGrowthSettings | None = None,
    bridging_settings: MaterialSurfaceBridgingSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    interface_path = _resolve(interface_root, MATERIAL_INTERFACE_STEM)
    macro_path = _resolve(macro_root, MACRO_ORIENTATION_STEM)
    seed_surface_path = _resolve_seed_surface(seed_surface_root)
    interfaces = json.loads(interface_path.read_text())
    macro = json.loads(macro_path.read_text())
    seed_surface = json.loads(seed_surface_path.read_text())
    resolved = settings or MaterialSurfaceFixedPointSettings()
    resolved_growth = growth_settings or MaterialSurfaceGrowthSettings()
    resolved_bridging = bridging_settings or MaterialSurfaceBridgingSettings()
    identity: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_FIXED_POINT_SCHEMA,
        "version": MATERIAL_SURFACE_FIXED_POINT_VERSION,
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
        "seedSurface": {
            "schema": str(seed_surface["schema"]),
            "manifestPath": str(seed_surface_path),
            "manifestSha256": sha256_file(seed_surface_path),
            "dataSha256": seed_surface["data"]["sha256"],
        },
        "settings": resolved.record(),
        "growthSettings": resolved_growth.record(),
        "bridgingSettings": resolved_bridging.record(),
        "implementationSha256": {
            "material_surface_fixed_point.py": sha256_file(Path(__file__)),
            "material_surface_growth.py": sha256_file(
                Path(__file__).with_name("material_surface_growth.py")
            ),
            "material_surface_bridging.py": sha256_file(
                Path(__file__).with_name("material_surface_bridging.py")
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{MATERIAL_SURFACE_FIXED_POINT_STEM}.json"
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        final = cached.get("finalSurface", {})
        final_path = Path(str(final.get("manifestPath", "")))
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and final_path.is_file()
            and final.get("manifestSha256") == sha256_file(final_path)
        ):
            return cached

    started = time.monotonic()
    current_surface = seed_surface_path
    cycles: list[dict[str, Any]] = []
    converged = False
    for cycle in range(1, resolved.maximum_cycles + 1):
        growth_root = output / f"cycle-{cycle:02d}-growth"
        growth = run_material_surface_growth(
            interface_path,
            macro_path,
            current_surface,
            growth_root,
            settings=resolved_growth,
            force=force,
        )
        growth_path = growth_root / f"{MATERIAL_SURFACE_GROWTH_STEM}.json"
        bridging_root = output / f"cycle-{cycle:02d}-bridging"
        bridging = run_material_surface_bridging(
            interface_path,
            macro_path,
            growth_path,
            bridging_root,
            settings=resolved_bridging,
            force=force,
        )
        bridging_path = bridging_root / f"{MATERIAL_SURFACE_BRIDGING_STEM}.json"
        grown = int(growth["counts"]["grownNodeCount"])
        bridge_candidates = int(
            bridging["counts"]["newBridgeCandidateNodeCount"]
        )
        merges = int(bridging["counts"]["newComponentMergeCount"])
        cycles.append(
            {
                "cycle": cycle,
                "seedSurfaceManifestPath": str(current_surface),
                "growthManifestPath": str(growth_path),
                "bridgingManifestPath": str(bridging_path),
                "grownNodeCount": grown,
                "bridgeCandidateNodeCount": bridge_candidates,
                "componentMergeCount": merges,
                "activeNodeCount": int(bridging["counts"]["activeNodeCount"]),
                "activeNodeFraction": float(
                    bridging["counts"]["activeNodeFraction"]
                ),
                "componentCount": int(bridging["counts"]["componentCount"]),
                "largestComponentSizes": list(
                    bridging["counts"]["largestComponentSizes"]
                ),
                "stratumCollisionRejectedBundleCount": int(
                    bridging["mergeAccounting"]["rejectedStratumCollision"]
                ),
                "physicalIdentityRejectedBundleCount": int(
                    bridging["mergeAccounting"].get(
                        "rejectedPhysicalSeedConflict", 0
                    )
                ),
            }
        )
        current_surface = bridging_path
        if grown == 0 and bridge_candidates == 0 and merges == 0:
            converged = True
            break

    final_surface = json.loads(current_surface.read_text())
    final_invariants = _audit_final_surface(current_surface, final_surface)
    payload: dict[str, Any] = {
        "schema": MATERIAL_SURFACE_FIXED_POINT_SCHEMA,
        "version": MATERIAL_SURFACE_FIXED_POINT_VERSION,
        "state": "complete",
        "identity": identity,
        "convergence": {
            "converged": converged,
            "completedCycles": len(cycles),
            "maximumCycles": resolved.maximum_cycles,
            "criterion": (
                "zero enclosed-hole additions, zero bridge-face additions, "
                "and zero component merges in one complete cycle"
            ),
        },
        "cycles": cycles,
        "finalSurface": {
            "schema": str(final_surface["schema"]),
            "manifestPath": str(current_surface),
            "manifestSha256": sha256_file(current_surface),
            "dataPath": str(
                current_surface.parent / str(final_surface["data"]["path"])
            ),
            "dataSha256": final_surface["data"]["sha256"],
        },
        "finalCounts": final_surface["counts"],
        "finalInvariants": final_invariants,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
    }
    atomic_json(manifest_path, payload)
    return payload
