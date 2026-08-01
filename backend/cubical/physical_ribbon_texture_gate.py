from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_bridging import _load_inputs, _load_npz, _write_npz
from .physical_ribbon_configuration import _component_labels
from .physical_ribbon_flattened_audit import (
    PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
)
from .physical_ribbon_patch_holes import (
    PhysicalRibbonPatchHoleSettings,
    build_physical_ribbon_surface_complex,
    extract_surface_boundary_loops,
)
from .physical_ribbon_patch_states import (
    PHYSICAL_RIBBON_PATCH_STATE_SCHEMA,
    _lineage_audit,
    _loops_view,
    _resolve_holes_manifest,
    _selection_conflicts,
    _surface_view,
)


PHYSICAL_RIBBON_TEXTURE_GATE_SCHEMA = "pareidolia.physical-ribbon-texture-gate"
PHYSICAL_RIBBON_TEXTURE_GATE_VERSION = 1
PHYSICAL_RIBBON_TEXTURE_GATE_STEM = "physical-ribbon-texture-gate-v1"


_SURFACE_FIELDS = (
    "componentSize",
    "signedNormalXYZ",
    "tangentUxyz",
    "tangentVxyz",
    "chartUV",
    "integrationResidualVoxels",
    "triangleFrontierIndex",
    "triangleAreaVoxelsSquared",
    "triangleNormalResidualDegrees",
    "midpointXYZ",
    "thicknessVoxels",
)


def _resolve_manifest(
    root: str | Path, schema: str
) -> tuple[Path, dict[str, Any]]:
    value = Path(root).resolve()
    candidates = (value,) if value.is_file() else tuple(sorted(value.glob("*.json")))
    matches: list[tuple[Path, dict[str, Any]]] = []
    for path in candidates:
        try:
            manifest = json.loads(path.read_text())
        except (OSError, json.JSONDecodeError):
            continue
        if manifest.get("schema") == schema and manifest.get("state") == "complete":
            matches.append((path, manifest))
    if len(matches) != 1:
        raise ValueError(f"root must identify exactly one complete {schema} artifact")
    return matches[0]


def _reference(path: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    result: dict[str, Any] = {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
    }
    if "data" in manifest:
        result["dataSha256"] = manifest["data"]["sha256"]
    return result


def texture_patch_decisions(
    patch_state: Mapping[str, np.ndarray],
    audit_payload: Mapping[str, Any],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Measure alternatives and choose the best compatible state per component."""

    variant_records = audit_payload.get("audit", {}).get("variants")
    if variant_records is not None:
        patch_count = len(np.asarray(patch_state["patchAccepted"]))
        target_component = np.asarray(
            patch_state["patchTargetPriorComponent"], dtype=np.int32
        )
        geometry_accepted = np.zeros(patch_count, dtype=bool)
        measured = np.zeros(patch_count, dtype=bool)
        compatible = np.zeros(patch_count, dtype=bool)
        accepted = np.zeros(patch_count, dtype=bool)
        records_by_component: dict[int, list[Mapping[str, Any]]] = {}
        seen_rows: set[int] = set()
        for record in variant_records:
            row = int(record["patchRow"])
            if row < 0 or row >= patch_count or row in seen_rows:
                raise ValueError("texture audit contains an invalid patch row")
            seen_rows.add(row)
            exact_audit = record.get("exactAudit", {})
            if not bool(exact_audit.get("accepted")):
                raise ValueError("texture audit variant is not exact-geometry valid")
            component_id = int(record["componentId"])
            if int(target_component[row]) != component_id:
                raise ValueError("texture audit variant targets a different component")
            geometry_accepted[row] = True
            value = record.get("boundaryTextureCompatible")
            if value is None:
                continue
            measured[row] = True
            compatible[row] = bool(value)
            if bool(value):
                records_by_component.setdefault(component_id, []).append(record)

        def exact_key(record: Mapping[str, Any]) -> tuple[float, ...]:
            exact = record["exactAudit"]
            before = exact["before"]
            after = exact["after"]
            return (
                float(
                    int(before.get("macroHoleCount", 0))
                    - int(after.get("macroHoleCount", 0))
                ),
                float(
                    int(before.get("interiorHoleCount", 0))
                    - int(after.get("interiorHoleCount", 0))
                ),
                float(
                    int(before.get("triangleRegionCount", 0))
                    - int(after.get("triangleRegionCount", 0))
                ),
                float(
                    int(after.get("triangleCount", 0))
                    - int(before.get("triangleCount", 0))
                ),
                float(exact.get("triangleAreaRetention", 0.0)),
                float(record.get("objectiveGain", 0.0)),
                -float(record.get("variantRank", 0)),
                -float(record["patchRow"]),
            )

        for records in records_by_component.values():
            winner = max(records, key=exact_key)
            accepted[int(winner["patchRow"])] = True
        return geometry_accepted, measured, compatible, accepted

    geometry_accepted = np.asarray(patch_state["patchAccepted"], dtype=np.uint8) > 0
    added_offset = np.asarray(patch_state["patchAddedOffset"], dtype=np.int64)
    added_node = np.asarray(
        patch_state["patchAddedFrontierIndex"], dtype=np.int32
    )
    component = np.asarray(patch_state["component"], dtype=np.int32)
    records = {
        int(record["componentId"]): record
        for record in audit_payload.get("audit", {}).get("components", ())
    }
    measured = np.zeros(len(geometry_accepted), dtype=bool)
    compatible = np.zeros(len(geometry_accepted), dtype=bool)
    final_component = np.full(len(geometry_accepted), -1, dtype=np.int32)
    for row in np.flatnonzero(geometry_accepted):
        added = added_node[int(added_offset[row]) : int(added_offset[row + 1])]
        values = np.unique(component[added])
        values = values[values >= 0]
        if len(values) != 1:
            raise ValueError("accepted patch does not map to one final component")
        component_id = int(values[0])
        final_component[row] = component_id
        record = records.get(component_id)
        if record is None:
            raise ValueError(
                "texture audit omitted an accepted patch component; rerun it "
                "with a larger maximum_components setting"
            )
        value = record.get("boundaryTextureCompatible")
        if value is None:
            raise ValueError(
                "texture audit lacks enough boundary measurements for an "
                "accepted patch component"
            )
        measured[row] = True
        compatible[row] = bool(value)
    return geometry_accepted, measured, compatible, geometry_accepted & compatible


def _surface_from_arrays(arrays: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    names = ("selected", "component", *_SURFACE_FIELDS)
    return {name: np.asarray(arrays[name]) for name in names}


def run_physical_ribbon_texture_gate(
    patch_state_root: str | Path,
    texture_audit_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    patch_path, patch_manifest = _resolve_manifest(
        patch_state_root, PHYSICAL_RIBBON_PATCH_STATE_SCHEMA
    )
    patch_data_path = patch_path.parent / str(patch_manifest["data"]["path"])
    patch_arrays = _load_npz(
        patch_data_path, patch_manifest["data"]["sha256"]
    )
    audit_path, audit_manifest = _resolve_manifest(
        texture_audit_root, PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA
    )
    audit_surface = audit_manifest["identity"]["surface"]
    if (
        audit_surface["manifestPath"] != str(patch_path)
        or audit_surface["manifestSha256"] != sha256_file(patch_path)
        or audit_surface["dataSha256"] != patch_manifest["data"]["sha256"]
    ):
        raise ValueError("texture audit does not measure the requested patch state")

    holes_reference = patch_manifest["identity"]["holes"]
    holes_path, holes_manifest = _resolve_holes_manifest(
        holes_reference["manifestPath"]
    )
    if (
        sha256_file(holes_path) != holes_reference["manifestSha256"]
        or holes_manifest["data"]["sha256"] != holes_reference["dataSha256"]
    ):
        raise ValueError("patch-state hole provenance changed")
    holes_data_path = holes_path.parent / str(holes_manifest["data"]["path"])
    holes = _load_npz(holes_data_path, holes_manifest["data"]["sha256"])

    configuration_reference = patch_manifest["identity"]["configuration"]
    (
        configuration_path,
        configuration_manifest,
        configuration,
        topology_path,
        topology_manifest,
        topology,
        ribbon_path,
        ribbon_manifest,
        ribbon,
    ) = _load_inputs(configuration_reference["manifestPath"])
    if (
        sha256_file(configuration_path) != configuration_reference["manifestSha256"]
        or configuration_manifest["data"]["sha256"]
        != configuration_reference["dataSha256"]
    ):
        raise ValueError("patch-state baseline configuration changed")
    identity = {
        "schema": PHYSICAL_RIBBON_TEXTURE_GATE_SCHEMA,
        "version": PHYSICAL_RIBBON_TEXTURE_GATE_VERSION,
        "patchState": _reference(patch_path, patch_manifest),
        "textureAudit": _reference(audit_path, audit_manifest),
        "configuration": _reference(configuration_path, configuration_manifest),
        "topologyContinuity": _reference(topology_path, topology_manifest),
        "ribbonBank": _reference(ribbon_path, ribbon_manifest),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_TEXTURE_GATE_STEM}.json"
    data_path = output / f"{PHYSICAL_RIBBON_TEXTURE_GATE_STEM}.npz"
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
    (
        geometry_accepted,
        texture_measured,
        texture_compatible,
        final_patch_accepted,
    ) = (
        texture_patch_decisions(patch_arrays, audit_manifest)
    )
    selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    added_offset = np.asarray(patch_arrays["patchAddedOffset"], dtype=np.int64)
    added_node = np.asarray(
        patch_arrays["patchAddedFrontierIndex"], dtype=np.int32
    )
    removed_offset = np.asarray(
        patch_arrays["patchRemovedOffset"], dtype=np.int64
    )
    removed_node = np.asarray(
        patch_arrays["patchRemovedFrontierIndex"], dtype=np.int32
    )
    for row in np.flatnonzero(final_patch_accepted):
        selected[
            removed_node[
                int(removed_offset[row]) : int(removed_offset[row + 1])
            ]
        ] = False
        selected[
            added_node[int(added_offset[row]) : int(added_offset[row + 1])]
        ] = True
    edge_first = np.asarray(topology["edgeFirstFrontierIndex"], dtype=np.int32)
    edge_second = np.asarray(topology["edgeSecondFrontierIndex"], dtype=np.int32)
    component, component_size = _component_labels(selected, edge_first, edge_second)
    selection = {
        "selected": selected.astype(np.uint8),
        "component": component.astype(np.int32),
        "componentSize": component_size.astype(np.int32),
    }
    patch_selected = np.asarray(patch_arrays["selected"], dtype=np.uint8) > 0
    baseline_selected = np.asarray(configuration["selected"], dtype=np.uint8) > 0
    if np.array_equal(selected, patch_selected):
        surface = _surface_from_arrays(patch_arrays)
    elif np.array_equal(selected, baseline_selected):
        surface = _surface_view(holes)
    else:
        surface, _ = build_physical_ribbon_surface_complex(
            ribbon,
            topology,
            selection,
            settings=PhysicalRibbonPatchHoleSettings(),
        )
    loops, loop_stats = extract_surface_boundary_loops(
        surface, settings=PhysicalRibbonPatchHoleSettings()
    )
    frontier = np.asarray(topology["frontierRibbonCandidate"], dtype=np.int32)
    source = np.asarray(ribbon["sourceInterface"], dtype=np.int32)[frontier]
    target = np.asarray(ribbon["targetInterface"], dtype=np.int32)[frontier]
    crossing_first = np.asarray(
        configuration["crossingFirstFrontierIndex"], dtype=np.int32
    )
    crossing_second = np.asarray(
        configuration["crossingSecondFrontierIndex"], dtype=np.int32
    )
    conflicts = _selection_conflicts(
        selected, source, target, crossing_first, crossing_second
    )
    _, lineage = _lineage_audit(
        baseline_selected,
        np.asarray(configuration["component"], dtype=np.int32),
        selected,
        component,
    )
    if conflicts != (0, 0) or any(lineage.values()):
        raise RuntimeError("texture-gated patch state violated a hard invariant")

    output_arrays = dict(patch_arrays)
    output_arrays["patchGeometryAccepted"] = geometry_accepted.astype(np.uint8)
    output_arrays["patchTextureMeasured"] = texture_measured.astype(np.uint8)
    output_arrays["patchTextureCompatible"] = texture_compatible.astype(np.uint8)
    output_arrays["patchAccepted"] = final_patch_accepted.astype(np.uint8)
    output_arrays["selected"] = selection["selected"]
    output_arrays["component"] = selection["component"]
    for name in _SURFACE_FIELDS:
        output_arrays[name] = np.asarray(surface[name])
    _write_npz(data_path, output_arrays)
    finished = time.monotonic()
    baseline_triangle_count = len(np.asarray(holes["triangleFrontierIndex"]))
    final_triangle_count = len(np.asarray(surface["triangleFrontierIndex"]))
    baseline_loop_stats = holes_manifest["loops"]
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_TEXTURE_GATE_SCHEMA,
        "version": PHYSICAL_RIBBON_TEXTURE_GATE_VERSION,
        "state": "complete",
        "identity": identity,
        "source": configuration_manifest["source"],
        "geometry": configuration_manifest["geometry"],
        "textureGate": {
            "geometryAcceptedPatchCount": int(np.count_nonzero(geometry_accepted)),
            "textureMeasuredPatchCount": int(np.count_nonzero(texture_measured)),
            "textureCompatiblePatchCount": int(
                np.count_nonzero(geometry_accepted & texture_compatible)
            ),
            "textureSelectedPatchCount": int(
                np.count_nonzero(final_patch_accepted)
            ),
            "textureRejectedPatchCount": int(
                np.count_nonzero(
                    geometry_accepted & texture_measured & ~texture_compatible
                )
            ),
            "textureUnmeasuredPatchCount": int(
                np.count_nonzero(geometry_accepted & ~texture_measured)
            ),
            "textureCompatibleAlternativeCount": int(
                np.count_nonzero(
                    geometry_accepted
                    & texture_compatible
                    & ~final_patch_accepted
                )
            ),
            "acceptedAddedRibbonCount": int(
                sum(
                    int(added_offset[row + 1] - added_offset[row])
                    for row in np.flatnonzero(final_patch_accepted)
                )
            ),
            "acceptedRemovedRibbonCount": int(
                sum(
                    int(removed_offset[row + 1] - removed_offset[row])
                    for row in np.flatnonzero(final_patch_accepted)
                )
            ),
            "selectedRibbonCountBefore": int(np.count_nonzero(baseline_selected)),
            "selectedRibbonCountAfter": int(np.count_nonzero(selected)),
            "interfaceConflictCount": conflicts[0],
            "crossingConflictCount": conflicts[1],
            **lineage,
        },
        "exactTopology": {
            "strictTriangleCountBefore": baseline_triangle_count,
            "strictTriangleCountAfter": final_triangle_count,
            "strictTriangleCountDelta": final_triangle_count
            - baseline_triangle_count,
            "triangleRegionCountBefore": int(
                baseline_loop_stats["triangleRegionCount"]
            ),
            "triangleRegionCountAfter": int(loop_stats["triangleRegionCount"]),
            "interiorHoleLoopCountBefore": int(
                baseline_loop_stats["interiorHoleLoopCount"]
            ),
            "interiorHoleLoopCountAfter": int(
                loop_stats["interiorHoleLoopCount"]
            ),
            "macroEligibleHoleCountBefore": int(
                baseline_loop_stats["macroEligibleHoleCount"]
            ),
            "macroEligibleHoleCountAfter": int(
                loop_stats["macroEligibleHoleCount"]
            ),
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(output_arrays),
        },
        "method": {
            "decision": (
                "retain each exact geometry patch only when its flattened "
                "native-CT boundary is compatible with same-surface controls"
            ),
            "selectionMutated": True,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
