from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .tables import read_patch_shard


BOUNDARY_SPLIT_AUDIT_SCHEMA = "pareidolia.cubical-boundary-split-audit"
BOUNDARY_SPLIT_AUDIT_VERSION = 1


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
