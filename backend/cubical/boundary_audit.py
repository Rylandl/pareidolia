from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .tables import read_patch_shard


BOUNDARY_SPLIT_AUDIT_SCHEMA = "pareidolia.cubical-boundary-split-audit"
BOUNDARY_SPLIT_AUDIT_VERSION = 1
BOUNDARY_RESELECTION_AUDIT_SCHEMA = (
    "pareidolia.cubical-boundary-reselection-split-audit"
)
BOUNDARY_RESELECTION_AUDIT_VERSION = 1
INDEPENDENT_BOUNDARY_AUDIT_SCHEMA = (
    "pareidolia.cubical-independent-boundary-audit"
)
INDEPENDENT_BOUNDARY_AUDIT_VERSION = 1


def _pair(first: int, second: int) -> tuple[int, int]:
    return (first, second) if first <= second else (second, first)


def _ratio(numerator: int, denominator: int) -> float:
    return round(numerator / max(denominator, 1), 7)


def run_boundary_split_audit(
    full_packet_root: str | Path,
    merge_root: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Compare split/recompose seam decisions with one full-block packet graph."""

    full = Path(full_packet_root).resolve()
    merge = Path(merge_root).resolve()
    output = Path(output_path).resolve()
    packet_manifest = json.loads((full / "packets.json").read_text())
    merge_manifest = json.loads((merge / "boundary-merge-v1.json").read_text())
    if (
        packet_manifest.get("schema")
        != "pareidolia.cubical-dual-axis-sheet-packets"
        or packet_manifest.get("state") != "complete"
    ):
        raise ValueError("full packet reference is incomplete")
    if (
        merge_manifest.get("schema")
        != "pareidolia.cubical-boundary-band-merge"
        or merge_manifest.get("state") != "complete"
    ):
        raise ValueError("boundary merge reference is incomplete")
    adjacency = merge_manifest["adjacency"]
    axis = int(adjacency["axis"])
    lower_input = int(adjacency["lowerInput"])
    input_records = merge_manifest["identity"]["inputs"]
    lower_boundary = json.loads(
        (
            Path(input_records[lower_input]["root"])
            / "boundary-band-v1.json"
        ).read_text()
    )
    lower_shape = lower_boundary["grid"]["shapeCellsXYZ"]
    seam_coordinate = (
        int(adjacency["offsetsCellsXYZ"][lower_input][axis])
        + int(lower_shape[axis])
    )
    identity: dict[str, Any] = {
        "schema": BOUNDARY_SPLIT_AUDIT_SCHEMA,
        "version": BOUNDARY_SPLIT_AUDIT_VERSION,
        "fullPacketRoot": str(full),
        "fullPacketGraphSha256": sha256_file(full / "packet-graph-v1.npz"),
        "mergeRoot": str(merge),
        "mergeArtifactSha256": sha256_file(merge / "boundary-merge-v1.npz"),
        "axis": axis,
        "seamCoordinate": seam_coordinate,
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    if output.is_file():
        prior = json.loads(output.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("split audit output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior

    with np.load(full / "packet-graph-v1.npz") as values:
        full_all_pairs = {
            _pair(int(first), int(second))
            for first, second in zip(
                values["firstPatchId"], values["secondPatchId"]
            )
        }
        mask = (values["faceAxis"] == axis) & (
            values["faceAnchorXYZ"][:, axis] == seam_coordinate
        )
        full_pairs = {
            _pair(int(first), int(second))
            for first, second in zip(
                values["firstPatchId"][mask],
                values["secondPatchId"][mask],
            )
        }
        full_quarter_pairs = {
            _pair(int(first), int(second))
            for first, second, quarter in zip(
                values["firstPatchId"][mask],
                values["secondPatchId"][mask],
                values["fiberQuarterTurn"][mask],
            )
            if int(quarter) == 1
        }
        full_component_count = len(set(int(value) for value in values["componentId"]))
    with np.load(merge / "boundary-merge-v1.npz") as values:
        merge_pairs = {
            _pair(int(first), int(second))
            for first, second in zip(
                values["matchFirstPatchId"], values["matchSecondPatchId"]
            )
        }
        merge_quarter_pairs = {
            _pair(int(first), int(second))
            for first, second, quarter in zip(
                values["matchFirstPatchId"],
                values["matchSecondPatchId"],
                values["matchFiberQuarterTurn"],
            )
            if int(quarter) == 1
        }
        selected_pairs = {
            _pair(int(first), int(second))
            for first, second, selected in zip(
                values["matchFirstPatchId"],
                values["matchSecondPatchId"],
                values["matchSelectedBridge"],
            )
            if int(selected) == 1
        }
    overlap = full_pairs & merge_pairs
    child_component_counts: list[int] = []
    child_internal_pairs: set[tuple[int, int]] = set()
    boundary_manifests: list[dict[str, Any]] = []
    for record in input_records:
        boundary = json.loads(
            (Path(record["root"]) / "boundary-band-v1.json").read_text()
        )
        boundary_manifests.append(boundary)
        child_component_counts.append(
            int(boundary["statistics"]["totalGraphComponents"])
        )
        packet_root = boundary.get("packetRoot")
        if packet_root is None:
            raise ValueError("split audit requires packet graphs for both children")
        with np.load(Path(packet_root) / "packet-graph-v1.npz") as values:
            child_internal_pairs.update(
                _pair(int(first), int(second))
                for first, second in zip(
                    values["firstPatchId"], values["secondPatchId"]
                )
            )
    recomposed_component_count = (
        sum(child_component_counts)
        - int(merge_manifest["statistics"]["retainedComponentBridges"])
    )
    full_internal_pairs = full_all_pairs - full_pairs
    child_only_internal = child_internal_pairs - full_internal_pairs
    full_only_internal = full_internal_pairs - child_internal_pairs
    selected_root = Path(packet_manifest["identity"]["inputRoot"])
    selected = read_patch_shard(selected_root / "selected-patches-v1")
    cell_by_patch = {
        int(patch_id): tuple(int(value) for value in cell)
        for patch_id, cell in zip(selected.patch_id, selected.cell_xyz)
    }

    def seam_distance(pair: tuple[int, int]) -> int:
        coordinates = [cell_by_patch[value][axis] for value in pair]
        if max(coordinates) < seam_coordinate:
            return seam_coordinate - 1 - max(coordinates)
        if min(coordinates) >= seam_coordinate:
            return min(coordinates) - seam_coordinate
        raise ValueError("an internal child join unexpectedly crosses the seam")

    def histogram(pairs: set[tuple[int, int]]) -> dict[str, int]:
        values: dict[int, int] = {}
        for pair in pairs:
            distance = seam_distance(pair)
            values[distance] = values.get(distance, 0) + 1
        return {str(key): values[key] for key in sorted(values)}

    band_depth = min(
        int(value["settings"]["depth_cells"])
        for value in boundary_manifests
    )
    all_internal_differences = child_only_internal | full_only_internal
    result: dict[str, Any] = {
        "schema": BOUNDARY_SPLIT_AUDIT_SCHEMA,
        "version": BOUNDARY_SPLIT_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "scope": (
            "deterministic geometry-contract audit only; child geometry was "
            "partitioned from the full selected artifact rather than inferred "
            "independently from raw CT"
        ),
        "seam": {"axis": axis, "coordinate": seam_coordinate},
        "joinAgreement": {
            "fullRetainedSeamJoins": len(full_pairs),
            "recomposedAlignedMatches": len(merge_pairs),
            "exactOverlap": len(overlap),
            "recomposedAgreementFraction": _ratio(len(overlap), len(merge_pairs)),
            "fullReferenceRecoveryFraction": _ratio(len(overlap), len(full_pairs)),
            "fullOnly": len(full_pairs - merge_pairs),
            "recomposedOnly": len(merge_pairs - full_pairs),
            "fullQuarterTurnJoins": len(full_quarter_pairs),
            "recomposedQuarterTurnMatches": len(merge_quarter_pairs),
            "retainedForestBridges": len(selected_pairs),
            "retainedForestBridgesInFullReference": len(selected_pairs & full_pairs),
        },
        "componentAgreement": {
            "childComponents": child_component_counts,
            "recomposedComponents": recomposed_component_count,
            "fullReferenceComponents": full_component_count,
            "excessRecomposedComponents": (
                recomposed_component_count - full_component_count
            ),
        },
        "interiorImmutability": {
            "serializedBoundaryDepthCells": band_depth,
            "childOnlyInternalJoins": len(child_only_internal),
            "fullOnlyInternalJoins": len(full_only_internal),
            "childOnlyByDistanceFromSeamCells": histogram(
                child_only_internal
            ),
            "fullOnlyByDistanceFromSeamCells": histogram(full_only_internal),
            "maximumDifferenceDistanceFromSeamCells": max(
                (seam_distance(value) for value in all_internal_differences),
                default=None,
            ),
            "differencesOutsideSerializedBoundaryBand": sum(
                seam_distance(value) >= band_depth
                for value in all_internal_differences
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def run_boundary_reselection_split_audit(
    full_packet_root: str | Path,
    reselection_root: str | Path,
    output_path: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Verify a deterministic narrow-band solve against the unsplit graph."""

    full = Path(full_packet_root).resolve()
    reselection = Path(reselection_root).resolve()
    output = Path(output_path).resolve()
    packet_manifest = json.loads((full / "packets.json").read_text())
    manifest_path = reselection / "boundary-reselection-v1.json"
    reselection_manifest = json.loads(manifest_path.read_text())
    if (
        packet_manifest.get("schema")
        != "pareidolia.cubical-dual-axis-sheet-packets"
        or packet_manifest.get("state") != "complete"
    ):
        raise ValueError("full packet reference is incomplete")
    if (
        reselection_manifest.get("schema")
        != "pareidolia.cubical-boundary-band-reselection"
        or reselection_manifest.get("state") != "complete"
    ):
        raise ValueError("boundary reselection reference is incomplete")
    artifact_path = reselection / "boundary-reselection-v1.npz"
    identity: dict[str, Any] = {
        "schema": BOUNDARY_RESELECTION_AUDIT_SCHEMA,
        "version": BOUNDARY_RESELECTION_AUDIT_VERSION,
        "fullPacketRoot": str(full),
        "fullPacketGraphSha256": sha256_file(full / "packet-graph-v1.npz"),
        "reselectionRoot": str(reselection),
        "reselectionManifestSha256": sha256_file(manifest_path),
        "reselectionArtifactSha256": sha256_file(artifact_path),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    if output.is_file():
        prior = json.loads(output.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("reselection audit output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior

    with np.load(full / "packet-graph-v1.npz") as values:
        full_joins = {
            (
                int(first),
                int(second),
                int(axis),
                tuple(int(value) for value in anchor),
            )
            for first, second, axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            )
        }
        full_quarter = {
            (
                int(first),
                int(second),
                int(axis),
                tuple(int(value) for value in anchor),
            )
            for first, second, axis, anchor, quarter in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
                values["fiberQuarterTurn"],
            )
            if int(quarter) == 1
        }
        full_component_count = len(set(int(value) for value in values["componentId"]))

    adjacency = reselection_manifest["adjacency"]
    axis = int(adjacency["axis"])
    lower_input = int(adjacency["lowerInput"])
    upper_input = int(adjacency["upperInput"])
    depth = int(reselection_manifest["slab"]["depthCellsPerInput"])
    boundary_roots = [
        Path(value["root"])
        for value in reselection_manifest["identity"]["inputs"]
    ]
    deep_joins: set[tuple[int, int, int, tuple[int, int, int]]] = set()
    for side, boundary_root in enumerate(boundary_roots):
        boundary = json.loads((boundary_root / "boundary-band-v1.json").read_text())
        selected_root = Path(boundary["selectedRoot"])
        packet_root = Path(boundary["packetRoot"])
        selected = read_patch_shard(selected_root / "selected-patches-v1")
        cell_by_patch = {
            int(patch_id): tuple(int(value) for value in cell)
            for patch_id, cell in zip(selected.patch_id, selected.cell_xyz)
        }
        facing_side = 1 if side == lower_input else 0

        def mutable(patch_id: int) -> bool:
            coordinate = cell_by_patch[patch_id][axis]
            return (
                coordinate < depth
                if facing_side == 0
                else coordinate >= selected.grid.shape_cells_xyz[axis] - depth
            )

        offset = tuple(int(value) for value in adjacency["offsetsCellsXYZ"][side])
        with np.load(packet_root / "packet-graph-v1.npz") as values:
            for first, second, face_axis, anchor in zip(
                values["firstPatchId"],
                values["secondPatchId"],
                values["faceAxis"],
                values["faceAnchorXYZ"],
            ):
                first_id = int(first)
                second_id = int(second)
                if mutable(first_id) or mutable(second_id):
                    continue
                deep_joins.add(
                    (
                        first_id,
                        second_id,
                        int(face_axis),
                        tuple(
                            int(anchor[index]) + offset[index]
                            for index in range(3)
                        ),
                    )
                )

    with np.load(artifact_path) as values:
        source_patch = {
            int(patch_id): int(source_id)
            for patch_id, source_id in zip(
                values["patchId"], values["patchSourcePatchId"]
            )
        }
        band_joins: set[tuple[int, int, int, tuple[int, int, int]]] = set()
        unmappable = 0
        for first, second, face_axis, anchor in zip(
            values["joinFirstPatchId"],
            values["joinSecondPatchId"],
            values["joinFaceAxis"],
            values["joinFaceAnchorXYZ"],
        ):
            first_source = source_patch[int(first)]
            second_source = source_patch[int(second)]
            if first_source < 0 or second_source < 0:
                unmappable += 1
                continue
            band_joins.add(
                (
                    first_source,
                    second_source,
                    int(face_axis),
                    tuple(int(value) for value in anchor),
                )
            )
        changed = int(np.sum(values["selectedConfigurationChanged"]))
    recomposed = deep_joins | band_joins
    overlap = recomposed & full_joins
    result: dict[str, Any] = {
        "schema": BOUNDARY_RESELECTION_AUDIT_SCHEMA,
        "version": BOUNDARY_RESELECTION_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "scope": (
            "deterministic geometry-contract audit; both child candidate banks "
            "were partitioned from one full selected reconstruction"
        ),
        "configurationAgreement": {
            "changedMutableCells": changed,
            "unchangedMutableCells": int(
                reselection_manifest["statistics"]["unchangedConfigurations"]
            ),
        },
        "joinAgreement": {
            "frozenDeepJoins": len(deep_joins),
            "jointBandJoins": len(band_joins),
            "unmappableChangedConfigurationJoins": unmappable,
            "recomposedJoins": len(recomposed),
            "fullReferenceJoins": len(full_joins),
            "exactOverlap": len(overlap),
            "fullReferenceRecoveryFraction": _ratio(len(overlap), len(full_joins)),
            "recomposedAgreementFraction": _ratio(len(overlap), len(recomposed)),
            "fullOnly": len(full_joins - recomposed),
            "recomposedOnly": len(recomposed - full_joins),
            "fullQuarterTurnJoins": len(full_quarter),
            "exactJoinGraphRecovered": recomposed == full_joins,
        },
        "componentAgreement": {
            "recomposedComponents": int(
                reselection_manifest["statistics"]["recomposedComponents"]
            ),
            "fullReferenceComponents": full_component_count,
            "delta": int(
                reselection_manifest["statistics"]["recomposedComponents"]
            )
            - full_component_count,
        },
        "interiorImmutability": {
            "depthCells": depth,
            "privateInteriorReadByReselection": False,
            "frozenJoinsReusedByAuditOnly": len(deep_joins),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result


def _axial_angle_degrees(first: np.ndarray, second: np.ndarray) -> float:
    return math.degrees(
        math.acos(float(np.clip(abs(np.dot(first, second)), 0.0, 1.0)))
    )


def _patch_groups(
    table: Any,
    *,
    offset: tuple[int, int, int] = (0, 0, 0),
    allowed_patch_ids: set[int] | None = None,
) -> dict[tuple[int, int, int], list[int]]:
    groups: dict[tuple[int, int, int], list[int]] = defaultdict(list)
    for row, (patch_id, cell) in enumerate(zip(table.patch_id, table.cell_xyz)):
        if allowed_patch_ids is not None and int(patch_id) not in allowed_patch_ids:
            continue
        resolved = tuple(int(cell[axis]) + offset[axis] for axis in range(3))
        groups[resolved].append(row)
    for rows in groups.values():
        rows.sort(
            key=lambda row: (
                float(table.height[row]),
                int(table.local_order[row]),
                int(table.patch_id[row]),
            )
        )
    return groups


def _configuration_agrees(
    first: Any,
    first_rows: list[int],
    second: Any,
    second_rows: list[int],
    *,
    height_tolerance_voxels: float,
    normal_tolerance_degrees: float,
    fiber_tolerance_degrees: float,
) -> tuple[bool, bool]:
    layer_count_agrees = len(first_rows) == len(second_rows)
    if not layer_count_agrees:
        return False, False
    for first_row, second_row in zip(first_rows, second_rows):
        if (
            abs(float(first.height[first_row]) - float(second.height[second_row]))
            > height_tolerance_voxels
            or _axial_angle_degrees(
                first.normal_xyz[first_row], second.normal_xyz[second_row]
            )
            > normal_tolerance_degrees
        ):
            return False, True
        first_fiber = first.fiber_xyz[first_row]
        second_fiber = second.fiber_xyz[second_row]
        if np.all(np.isfinite(first_fiber)) and np.all(np.isfinite(second_fiber)):
            if (
                _axial_angle_degrees(first_fiber, second_fiber)
                > fiber_tolerance_degrees
            ):
                return False, True
        elif np.any(np.isfinite(first_fiber)) or np.any(np.isfinite(second_fiber)):
            return False, True
    return True, True


def run_independent_boundary_audit(
    full_packet_root: str | Path,
    selected_merge_root: str | Path,
    reselection_root: str | Path,
    output_path: str | Path,
    *,
    height_tolerance_voxels: float = 1.0e-3,
    normal_tolerance_degrees: float = 0.01,
    fiber_tolerance_degrees: float = 0.01,
    force: bool = False,
) -> dict[str, Any]:
    """Compare independent-block seam refinement with one full-context reference."""

    tolerances = (
        height_tolerance_voxels,
        normal_tolerance_degrees,
        fiber_tolerance_degrees,
    )
    if any(not math.isfinite(value) or value < 0.0 for value in tolerances):
        raise ValueError("configuration agreement tolerances must be nonnegative")

    full = Path(full_packet_root).resolve()
    selected_merge = Path(selected_merge_root).resolve()
    reselection = Path(reselection_root).resolve()
    output = Path(output_path).resolve()
    packet_manifest = json.loads((full / "packets.json").read_text())
    merge_manifest_path = selected_merge / "boundary-merge-v1.json"
    merge_manifest = json.loads(merge_manifest_path.read_text())
    reselection_manifest_path = reselection / "boundary-reselection-v1.json"
    reselection_manifest = json.loads(reselection_manifest_path.read_text())
    if (
        packet_manifest.get("schema")
        != "pareidolia.cubical-dual-axis-sheet-packets"
        or packet_manifest.get("state") != "complete"
    ):
        raise ValueError("full packet reference is incomplete")
    if (
        merge_manifest.get("schema")
        != "pareidolia.cubical-boundary-band-merge"
        or merge_manifest.get("state") != "complete"
    ):
        raise ValueError("selected-only boundary merge is incomplete")
    if (
        reselection_manifest.get("schema")
        != "pareidolia.cubical-boundary-band-reselection"
        or reselection_manifest.get("state") != "complete"
    ):
        raise ValueError("joint boundary reselection is incomplete")
    if merge_manifest["identity"]["inputs"] != reselection_manifest["identity"][
        "inputs"
    ]:
        raise ValueError("selected-only and joint comparisons use different inputs")
    if merge_manifest["adjacency"] != reselection_manifest["adjacency"]:
        raise ValueError("selected-only and joint comparisons use different adjacency")
    identity: dict[str, Any] = {
        "schema": INDEPENDENT_BOUNDARY_AUDIT_SCHEMA,
        "version": INDEPENDENT_BOUNDARY_AUDIT_VERSION,
        "fullPacketRoot": str(full),
        "fullPacketGraphSha256": sha256_file(full / "packet-graph-v1.npz"),
        "selectedMergeRoot": str(selected_merge),
        "selectedMergeManifestSha256": sha256_file(merge_manifest_path),
        "reselectionRoot": str(reselection),
        "reselectionManifestSha256": sha256_file(reselection_manifest_path),
        "tolerances": {
            "heightVoxels": height_tolerance_voxels,
            "normalDegrees": normal_tolerance_degrees,
            "fiberDegrees": fiber_tolerance_degrees,
        },
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    if output.is_file():
        prior = json.loads(output.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("independent boundary audit output has another identity")
        if not force and prior.get("state") == "complete":
            return prior

    full_selected_root = Path(packet_manifest["identity"]["inputRoot"])
    full_selected = read_patch_shard(full_selected_root / "selected-patches-v1")
    with np.load(full / "packet-graph-v1.npz") as values:
        full_component_count = len(set(int(value) for value in values["componentId"]))
    adjacency = reselection_manifest["adjacency"]
    offsets = tuple(
        tuple(int(value) for value in offset)
        for offset in adjacency["offsetsCellsXYZ"]
    )
    boundary_roots = tuple(
        Path(value["root"])
        for value in reselection_manifest["identity"]["inputs"]
    )
    boundary_manifests = tuple(
        json.loads((root / "boundary-band-v1.json").read_text())
        for root in boundary_roots
    )
    baseline_tables = tuple(
        read_patch_shard(Path(value["selectedRoot"]) / "selected-patches-v1")
        for value in boundary_manifests
    )
    baseline_groups_by_side = tuple(
        _patch_groups(table, offset=offsets[side])
        for side, table in enumerate(baseline_tables)
    )
    baseline_owner: dict[tuple[int, int, int], int] = {}
    for side, table in enumerate(baseline_tables):
        offset = offsets[side]
        for z in range(table.grid.shape_cells_xyz[2]):
            for y in range(table.grid.shape_cells_xyz[1]):
                for x in range(table.grid.shape_cells_xyz[0]):
                    cell = (
                        x + offset[0],
                        y + offset[1],
                        z + offset[2],
                    )
                    if cell in baseline_owner:
                        raise ValueError("independent boundary inputs overlap")
                    baseline_owner[cell] = side
    full_groups = _patch_groups(full_selected)
    joint_table = read_patch_shard(reselection / "selected-band-patches-v1")
    with np.load(reselection / "boundary-reselection-v1.npz") as values:
        mutable_patch_ids = {
            int(patch_id)
            for patch_id, anchor in zip(values["patchId"], values["patchIsAnchor"])
            if int(anchor) == 0
        }
        mutable_cells = {
            tuple(int(value) for value in cell)
            for cell in values["selectedCellCombinedXYZ"]
        }
        changed_cells = {
            tuple(int(value) for value in cell)
            for cell, changed in zip(
                values["selectedCellCombinedXYZ"],
                values["selectedConfigurationChanged"],
            )
            if int(changed) == 1
        }
    joint_groups = _patch_groups(
        joint_table, allowed_patch_ids=mutable_patch_ids
    )

    baseline_exact: dict[tuple[int, int, int], bool] = {}
    joint_exact: dict[tuple[int, int, int], bool] = {}
    baseline_count: dict[tuple[int, int, int], bool] = {}
    joint_count: dict[tuple[int, int, int], bool] = {}
    for cell in mutable_cells:
        if cell not in baseline_owner:
            raise ValueError("joint mutable cell is outside both boundary inputs")
        side = baseline_owner[cell]
        baseline_rows = baseline_groups_by_side[side].get(cell, [])
        baseline_exact[cell], baseline_count[cell] = _configuration_agrees(
            baseline_tables[side],
            baseline_rows,
            full_selected,
            full_groups.get(cell, []),
            height_tolerance_voxels=height_tolerance_voxels,
            normal_tolerance_degrees=normal_tolerance_degrees,
            fiber_tolerance_degrees=fiber_tolerance_degrees,
        )
        joint_exact[cell], joint_count[cell] = _configuration_agrees(
            joint_table,
            joint_groups.get(cell, []),
            full_selected,
            full_groups.get(cell, []),
            height_tolerance_voxels=height_tolerance_voxels,
            normal_tolerance_degrees=normal_tolerance_degrees,
            fiber_tolerance_degrees=fiber_tolerance_degrees,
        )

    child_components = [
        int(value["statistics"]["totalGraphComponents"])
        for value in boundary_manifests
    ]
    selected_only_components = sum(child_components) - int(
        merge_manifest["statistics"]["retainedComponentBridges"]
    )

    def by_axis_coordinate(values: Mapping[tuple[int, int, int], bool]) -> dict[str, Any]:
        coordinates = sorted({cell[int(adjacency["axis"])] for cell in mutable_cells})
        return {
            str(coordinate): {
                "cells": sum(
                    cell[int(adjacency["axis"])] == coordinate
                    for cell in mutable_cells
                ),
                "agreements": sum(
                    agreement
                    for cell, agreement in values.items()
                    if cell[int(adjacency["axis"])] == coordinate
                ),
            }
            for coordinate in coordinates
        }

    result: dict[str, Any] = {
        "schema": INDEPENDENT_BOUNDARY_AUDIT_SCHEMA,
        "version": INDEPENDENT_BOUNDARY_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "scope": (
            "The adjacent blocks were inferred independently from native CT with "
            "one shared source-level calibration. The full-context reconstruction "
            "is a consistency reference, not ground truth."
        ),
        "configurationAgreement": {
            "mutableCells": len(mutable_cells),
            "jointChangedCells": len(changed_cells),
            "baselineExactFullContext": sum(baseline_exact.values()),
            "jointExactFullContext": sum(joint_exact.values()),
            "baselineLayerCountAgreement": sum(baseline_count.values()),
            "jointLayerCountAgreement": sum(joint_count.values()),
            "changedCellsBaselineExact": sum(
                baseline_exact[value] for value in changed_cells
            ),
            "changedCellsJointExact": sum(
                joint_exact[value] for value in changed_cells
            ),
            "changedTowardExactFullContext": sum(
                not baseline_exact[value] and joint_exact[value]
                for value in changed_cells
            ),
            "changedAwayFromExactFullContext": sum(
                baseline_exact[value] and not joint_exact[value]
                for value in changed_cells
            ),
            "baselineBySeamAxisCoordinate": by_axis_coordinate(baseline_exact),
            "jointBySeamAxisCoordinate": by_axis_coordinate(joint_exact),
        },
        "componentConsistency": {
            "independentChildComponents": child_components,
            "selectedOnlyRecomposedComponents": selected_only_components,
            "jointBandRecomposedComponents": int(
                reselection_manifest["statistics"]["recomposedComponents"]
            ),
            "fullContextComponents": full_component_count,
            "selectedOnlyDeltaFromFullContext": (
                selected_only_components - full_component_count
            ),
            "jointBandDeltaFromFullContext": int(
                reselection_manifest["statistics"]["recomposedComponents"]
            )
            - full_component_count,
            "componentGapClosedByJointBand": selected_only_components
            - int(reselection_manifest["statistics"]["recomposedComponents"]),
        },
        "seamEvidence": {
            "selectedOnlyAlignedMatches": int(
                merge_manifest["statistics"]["alignedMatches"]
            ),
            "selectedOnlyRetainedBridges": int(
                merge_manifest["statistics"]["retainedComponentBridges"]
            ),
            "jointStrictCandidates": int(
                reselection_manifest["statistics"]["strictCandidateJoins"]
            ),
            "jointQuarterTurnCandidates": int(
                reselection_manifest["statistics"]["quarterTurnCandidateJoins"]
            ),
            "jointRetainedBandJoins": int(
                reselection_manifest["statistics"]["retainedBandJoins"]
            ),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result
