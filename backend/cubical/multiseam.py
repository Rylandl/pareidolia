from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Hashable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file


MULTISEAM_AUDIT_SCHEMA = "pareidolia.cubical-multiseam-audit"
MULTISEAM_AUDIT_VERSION = 1

Int3 = tuple[int, int, int]
CellKey = tuple[str, Int3]
PatchKey = tuple[Hashable, ...]
JoinKey = tuple[PatchKey, PatchKey, int, Int3]


@dataclass(frozen=True, slots=True)
class _BoundaryRecord:
    key: str
    root: Path
    manifest: Mapping[str, Any]
    origin_xyz: tuple[float, float, float]
    cell_size_xyz: tuple[float, float, float]
    shape_cells_xyz: Int3
    coordinate_unit: str


@dataclass(slots=True)
class _SolutionRecord:
    index: int
    root: Path
    manifest: Mapping[str, Any]
    arrays: dict[str, np.ndarray]
    boundary_keys: tuple[str, str]
    decisions: dict[CellKey, tuple[int, bool]]
    patch_keys: set[PatchKey]
    join_keys: set[JoinKey]
    component_by_patch: dict[PatchKey, int]


def _int3(values: Any) -> Int3:
    result = tuple(int(value) for value in values)
    if len(result) != 3:
        raise ValueError("expected an XYZ triple")
    return result  # type: ignore[return-value]


def _float3(values: Any) -> tuple[float, float, float]:
    result = tuple(float(value) for value in values)
    if len(result) != 3 or any(not math.isfinite(value) for value in result):
        raise ValueError("expected a finite XYZ triple")
    return result  # type: ignore[return-value]


def _directory_bytes(path: Path) -> int:
    return sum(value.stat().st_size for value in path.rglob("*") if value.is_file())


def _load_solution_manifest(root: Path) -> tuple[dict[str, Any], Path]:
    manifest_path = root / "boundary-reselection-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema")
        != "pareidolia.cubical-boundary-band-reselection"
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError(f"incomplete boundary reselection: {root}")
    artifact = manifest["artifacts"]["reselection"]
    artifact_path = root / str(artifact["path"])
    if sha256_file(artifact_path) != artifact["sha256"]:
        raise ValueError(f"boundary reselection hash mismatch: {root}")
    return manifest, artifact_path


def _load_boundary(
    input_record: Mapping[str, Any],
) -> _BoundaryRecord:
    root = Path(str(input_record["root"])).resolve()
    manifest_path = root / "boundary-band-v1.json"
    if sha256_file(manifest_path) != input_record["manifestSha256"]:
        raise ValueError(f"boundary manifest hash mismatch: {root}")
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "pareidolia.cubical-boundary-band"
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError(f"incomplete boundary input: {root}")
    key = str(manifest["identity"]["identitySha256"])
    if key != input_record["boundaryIdentitySha256"]:
        raise ValueError(f"boundary identity mismatch: {root}")
    grid = manifest["grid"]
    return _BoundaryRecord(
        key,
        root,
        manifest,
        _float3(grid["originXYZ"]),
        _float3(grid["cellSizeXYZ"]),
        _int3(grid["shapeCellsXYZ"]),
        str(grid["coordinateUnit"]),
    )


def _lattice_offset(
    origin_xyz: tuple[float, float, float],
    global_origin_xyz: tuple[float, float, float],
    cell_size_xyz: tuple[float, float, float],
) -> Int3:
    values: list[int] = []
    for axis in range(3):
        coordinate = (
            origin_xyz[axis] - global_origin_xyz[axis]
        ) / cell_size_xyz[axis]
        rounded = round(coordinate)
        if not math.isclose(coordinate, rounded, abs_tol=1.0e-6):
            raise ValueError("block origins do not share one integer cell lattice")
        values.append(int(rounded))
    return tuple(values)  # type: ignore[return-value]


def _patch_key(
    boundary_key: str,
    source_patch_id: int,
    source_configuration_index: int,
    layer_index: int,
) -> PatchKey:
    if source_patch_id >= 0:
        return ("selected", boundary_key, source_patch_id)
    if source_configuration_index < 0 or layer_index < 0:
        raise ValueError("non-anchor patch lacks physical candidate provenance")
    return (
        "candidate",
        boundary_key,
        source_configuration_index,
        layer_index,
    )


def _sorted_patch_pair(first: PatchKey, second: PatchKey) -> tuple[PatchKey, PatchKey]:
    return (first, second) if repr(first) <= repr(second) else (second, first)


def _load_records(
    roots: tuple[Path, ...],
) -> tuple[
    tuple[_SolutionRecord, ...],
    dict[str, _BoundaryRecord],
    tuple[float, float, float],
    tuple[float, float, float],
]:
    manifests_and_artifacts = tuple(_load_solution_manifest(root) for root in roots)
    boundaries: dict[str, _BoundaryRecord] = {}
    for manifest, _ in manifests_and_artifacts:
        inputs = manifest["identity"]["inputs"]
        if len(inputs) != 2:
            raise ValueError("each boundary reselection must have two inputs")
        for input_record in inputs:
            boundary = _load_boundary(input_record)
            prior = boundaries.get(boundary.key)
            if prior is not None and prior.root != boundary.root:
                raise ValueError("one boundary identity resolves to multiple roots")
            boundaries[boundary.key] = boundary
    if len(boundaries) < 3:
        raise ValueError("multi-seam audit requires at least three unique blocks")
    units = {value.coordinate_unit for value in boundaries.values()}
    sizes = {value.cell_size_xyz for value in boundaries.values()}
    if len(units) != 1 or len(sizes) != 1:
        raise ValueError("multi-seam blocks use incompatible grids")
    cell_size_xyz = next(iter(sizes))
    global_origin_xyz = tuple(
        min(value.origin_xyz[axis] for value in boundaries.values())
        for axis in range(3)
    )
    boundary_offsets = {
        key: _lattice_offset(
            value.origin_xyz,
            global_origin_xyz,
            cell_size_xyz,
        )
        for key, value in boundaries.items()
    }

    owned_cells: dict[Int3, str] = {}
    for key, boundary in boundaries.items():
        offset = boundary_offsets[key]
        for z in range(boundary.shape_cells_xyz[2]):
            for y in range(boundary.shape_cells_xyz[1]):
                for x in range(boundary.shape_cells_xyz[0]):
                    global_cell = (
                        x + offset[0],
                        y + offset[1],
                        z + offset[2],
                    )
                    if global_cell in owned_cells:
                        raise ValueError("multi-seam input blocks overlap in cell ownership")
                    owned_cells[global_cell] = key

    solutions: list[_SolutionRecord] = []
    for index, (root, (manifest, artifact_path)) in enumerate(
        zip(roots, manifests_and_artifacts)
    ):
        with np.load(artifact_path) as values:
            arrays = {name: np.asarray(values[name]) for name in values.files}
        input_records = manifest["identity"]["inputs"]
        boundary_keys = tuple(
            str(value["boundaryIdentitySha256"]) for value in input_records
        )
        if len(set(boundary_keys)) != 2:
            raise ValueError("a seam cannot join a boundary artifact to itself")
        decisions: dict[CellKey, tuple[int, bool]] = {}
        for side, cell, configuration, changed in zip(
            arrays["selectedCellInput"],
            arrays["selectedCellLocalXYZ"],
            arrays["selectedSourceConfigurationIndex"],
            arrays["selectedConfigurationChanged"],
        ):
            side_index = int(side)
            if side_index not in (0, 1):
                raise ValueError("selected seam cell references an invalid input")
            key = (boundary_keys[side_index], _int3(cell))
            if key in decisions:
                raise ValueError("one seam selects a cell more than once")
            decisions[key] = (int(configuration), bool(changed))

        patch_by_temporary_id: dict[int, PatchKey] = {}
        patch_keys: set[PatchKey] = set()
        for patch_id, side, source_patch, source_configuration, layer in zip(
            arrays["patchId"],
            arrays["patchInput"],
            arrays["patchSourcePatchId"],
            arrays["patchSourceConfigurationIndex"],
            arrays["patchLayerIndex"],
        ):
            side_index = int(side)
            if side_index not in (0, 1):
                raise ValueError("selected seam patch references an invalid input")
            key = _patch_key(
                boundary_keys[side_index],
                int(source_patch),
                int(source_configuration),
                int(layer),
            )
            if key in patch_keys:
                raise ValueError("one seam realizes a physical patch more than once")
            patch_by_temporary_id[int(patch_id)] = key
            patch_keys.add(key)

        solution_grid = manifest["grid"]
        solution_origin = _float3(solution_grid["originXYZ"])
        solution_offset = _lattice_offset(
            solution_origin,
            global_origin_xyz,
            cell_size_xyz,
        )
        join_keys: set[JoinKey] = set()
        for first, second, axis, anchor in zip(
            arrays["joinFirstPatchId"],
            arrays["joinSecondPatchId"],
            arrays["joinFaceAxis"],
            arrays["joinFaceAnchorXYZ"],
        ):
            pair = _sorted_patch_pair(
                patch_by_temporary_id[int(first)],
                patch_by_temporary_id[int(second)],
            )
            local_anchor = _int3(anchor)
            global_anchor = tuple(
                local_anchor[value] + solution_offset[value]
                for value in range(3)
            )
            join_keys.add((pair[0], pair[1], int(axis), global_anchor))
        if len(join_keys) != len(arrays["joinFirstPatchId"]):
            raise ValueError("canonical multi-seam join keys are not unique")
        component_by_patch = {
            patch_by_temporary_id[int(patch_id)]: int(component_id)
            for patch_id, component_id in zip(
                arrays["componentPatchId"],
                arrays["componentId"],
            )
        }
        if set(component_by_patch) != patch_keys:
            raise ValueError("multi-seam component map does not cover every patch")
        solutions.append(
            _SolutionRecord(
                index,
                root,
                manifest,
                arrays,
                boundary_keys,  # type: ignore[arg-type]
                decisions,
                patch_keys,
                join_keys,
                component_by_patch,
            )
        )
    return tuple(solutions), boundaries, global_origin_xyz, cell_size_xyz


def _configuration_consistency(
    solutions: tuple[_SolutionRecord, ...],
    boundaries: Mapping[str, _BoundaryRecord],
) -> dict[str, Any]:
    observations: dict[CellKey, list[tuple[int, int, bool]]] = defaultdict(list)
    for solution in solutions:
        for key, (configuration, changed) in solution.decisions.items():
            observations[key].append((solution.index, configuration, changed))
    overlaps = {
        key: tuple(values)
        for key, values in observations.items()
        if len(values) > 1
    }
    disagreement = {
        key: values
        for key, values in overlaps.items()
        if len({value[1] for value in values}) > 1
    }
    by_block: dict[str, dict[str, int]] = {}
    for boundary_key in sorted(boundaries):
        block_values = {
            key: values for key, values in overlaps.items() if key[0] == boundary_key
        }
        block_disagreements = sum(
            len({value[1] for value in values}) > 1
            for values in block_values.values()
        )
        by_block[boundary_key] = {
            "overlapCells": len(block_values),
            "agreementCells": len(block_values) - block_disagreements,
            "disagreementCells": block_disagreements,
        }
    records = []
    for (boundary_key, cell), values in sorted(disagreement.items()):
        records.append(
            {
                "boundaryIdentitySha256": boundary_key,
                "boundaryRoot": str(boundaries[boundary_key].root),
                "cellLocalXYZ": list(cell),
                "selections": [
                    {
                        "solutionIndex": solution,
                        "sourceConfigurationIndex": configuration,
                        "changedFromBlockSelection": changed,
                    }
                    for solution, configuration, changed in values
                ],
            }
        )
    return {
        "uniqueMutableCells": len(observations),
        "selectionObservations": sum(len(value) for value in observations.values()),
        "overlapCells": len(overlaps),
        "agreementCells": len(overlaps) - len(disagreement),
        "disagreementCells": len(disagreement),
        "agreementFraction": round(
            (len(overlaps) - len(disagreement)) / max(len(overlaps), 1),
            7,
        ),
        "overlapObservationsChanged": sum(
            changed for values in overlaps.values() for _, _, changed in values
        ),
        "byBlock": by_block,
        "disagreements": records,
    }


def _topology_consistency(
    solutions: tuple[_SolutionRecord, ...],
) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for first_index, first in enumerate(solutions):
        for second in solutions[first_index + 1 :]:
            shared_cells = set(first.decisions) & set(second.decisions)
            if not shared_cells:
                continue
            common_patches = first.patch_keys & second.patch_keys
            first_eligible = {
                value
                for value in first.join_keys
                if value[0] in common_patches and value[1] in common_patches
            }
            second_eligible = {
                value
                for value in second.join_keys
                if value[0] in common_patches and value[1] in common_patches
            }
            overlap = first_eligible & second_eligible
            contingency = Counter(
                (
                    first.component_by_patch[patch],
                    second.component_by_patch[patch],
                )
                for patch in common_patches
            )
            first_component_size = Counter(
                first.component_by_patch[patch] for patch in common_patches
            )
            second_component_size = Counter(
                second.component_by_patch[patch] for patch in common_patches
            )

            def pair_count(values: Mapping[Hashable, int]) -> int:
                return sum(value * (value - 1) // 2 for value in values.values())

            same_first = pair_count(first_component_size)
            same_second = pair_count(second_component_size)
            same_both = pair_count(contingency)
            connectivity_disagreements = (
                same_first + same_second - 2 * same_both
            )
            comparisons.append(
                {
                    "firstSolutionIndex": first.index,
                    "secondSolutionIndex": second.index,
                    "sharedMutableCells": len(shared_cells),
                    "commonPhysicalPatches": len(common_patches),
                    "firstEligibleJoins": len(first_eligible),
                    "secondEligibleJoins": len(second_eligible),
                    "exactJoins": len(overlap),
                    "firstOnlyJoins": len(first_eligible - second_eligible),
                    "secondOnlyJoins": len(second_eligible - first_eligible),
                    "joinJaccard": round(
                        len(overlap)
                        / max(len(first_eligible | second_eligible), 1),
                        7,
                    ),
                    "firstInducedComponents": len(first_component_size),
                    "secondInducedComponents": len(second_component_size),
                    "coComponentPairsFirst": same_first,
                    "coComponentPairsSecond": same_second,
                    "coComponentPairsBoth": same_both,
                    "coComponentPairDisagreements": connectivity_disagreements,
                    "componentPartitionsAgree": connectivity_disagreements == 0,
                }
            )
    return {
        "crossingSeamPairs": len(comparisons),
        "comparisons": comparisons,
        "allCommonJoinsAgree": all(
            value["firstOnlyJoins"] == 0 and value["secondOnlyJoins"] == 0
            for value in comparisons
        ),
        "allCommonComponentPartitionsAgree": all(
            value["componentPartitionsAgree"] for value in comparisons
        ),
    }


def _linear_runs(cells: np.ndarray, shape: Int3) -> int:
    if len(cells) == 0:
        return 0
    linear = (
        cells[:, 0].astype(np.int64)
        + shape[0]
        * (
            cells[:, 1].astype(np.int64)
            + shape[1] * cells[:, 2].astype(np.int64)
        )
    )
    ordered = np.unique(linear)
    return 1 + int(np.count_nonzero(np.diff(ordered) > 1))


def _boundary_storage(boundary: _BoundaryRecord) -> dict[str, Any]:
    topology_record = boundary.manifest["artifacts"]["frozenTopology"]
    topology_path = boundary.root / str(topology_record["path"])
    if sha256_file(topology_path) != topology_record["sha256"]:
        raise ValueError(f"frozen topology hash mismatch: {boundary.root}")
    with np.load(topology_path) as values:
        raw_bytes = sum(np.asarray(values[name]).nbytes for name in values.files)
        cells = np.asarray(values["componentCellXYZ"], dtype=np.int32)
        offsets = np.asarray(values["componentCellOffset"], dtype=np.uint64)
        runs = sum(
            _linear_runs(cells[int(low) : int(high)], boundary.shape_cells_xyz)
            for low, high in zip(offsets[:-1], offsets[1:])
        )
        owner_count = len(values["ownerPatchId"])
        observation_count = len(values["observationPatchId"])
    volume = math.prod(boundary.shape_cells_xyz)
    index_bytes = 4 if volume <= np.iinfo(np.uint32).max else 8
    cell_count = len(cells)
    return {
        "boundaryIdentitySha256": boundary.key,
        "root": str(boundary.root),
        "directoryBytes": _directory_bytes(boundary.root),
        "frozenTopologyCompressedBytes": topology_path.stat().st_size,
        "frozenTopologyRawArrayBytes": raw_bytes,
        "componentCellRecords": cell_count,
        "componentCellXYZBytes": int(cells.nbytes),
        "linearIndexBytesEstimate": cell_count * index_bytes,
        "linearRunCount": runs,
        "linearRunBytesEstimate": runs * 2 * index_bytes,
        "crossingObservations": observation_count,
        "crossingOwners": owner_count,
    }


def _storage_summary(
    solutions: tuple[_SolutionRecord, ...],
    boundaries: Mapping[str, _BoundaryRecord],
) -> dict[str, Any]:
    records = [_boundary_storage(boundaries[key]) for key in sorted(boundaries)]
    usage = Counter(
        boundary_key
        for solution in solutions
        for boundary_key in solution.boundary_keys
    )
    unique_bytes = sum(int(value["directoryBytes"]) for value in records)
    pairwise_bytes = sum(
        int(value["directoryBytes"])
        * usage[str(value["boundaryIdentitySha256"])]
        for value in records
    )
    return {
        "uniqueBoundaryArtifacts": len(records),
        "uniqueBoundaryBytes": unique_bytes,
        "pairwiseInputBytesWithDuplication": pairwise_bytes,
        "pairwiseDuplicationFactor": round(
            pairwise_bytes / max(unique_bytes, 1), 7
        ),
        "frozenTopologyCompressedBytes": sum(
            int(value["frozenTopologyCompressedBytes"]) for value in records
        ),
        "frozenTopologyRawArrayBytes": sum(
            int(value["frozenTopologyRawArrayBytes"]) for value in records
        ),
        "componentCellRecords": sum(
            int(value["componentCellRecords"]) for value in records
        ),
        "componentCellXYZBytes": sum(
            int(value["componentCellXYZBytes"]) for value in records
        ),
        "linearIndexBytesEstimate": sum(
            int(value["linearIndexBytesEstimate"]) for value in records
        ),
        "linearRunBytesEstimate": sum(
            int(value["linearRunBytesEstimate"]) for value in records
        ),
        "boundaries": records,
    }


def _load_cluster_solution(
    root: Path,
    boundaries: Mapping[str, _BoundaryRecord],
    global_origin_xyz: tuple[float, float, float],
    cell_size_xyz: tuple[float, float, float],
) -> _SolutionRecord:
    manifest_path = root / "cluster-reselection-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema")
        != "pareidolia.cubical-boundary-cluster-reselection"
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError(f"incomplete cluster reselection: {root}")
    artifact_record = manifest["artifacts"]["reselection"]
    artifact_path = root / str(artifact_record["path"])
    if sha256_file(artifact_path) != artifact_record["sha256"]:
        raise ValueError(f"cluster reselection hash mismatch: {root}")
    old_key_by_selected_root = {
        str(Path(str(value.manifest["identity"]["selectedRoot"])).resolve()): key
        for key, value in boundaries.items()
    }
    cluster_keys: list[str] = []
    for input_record in manifest["identity"]["inputs"]:
        boundary_root = Path(str(input_record["root"])).resolve()
        boundary_manifest_path = boundary_root / "boundary-band-v1.json"
        if sha256_file(boundary_manifest_path) != input_record["manifestSha256"]:
            raise ValueError("cluster input boundary manifest hash mismatch")
        boundary_manifest = json.loads(boundary_manifest_path.read_text())
        selected_root = str(
            Path(str(boundary_manifest["identity"]["selectedRoot"])).resolve()
        )
        if selected_root not in old_key_by_selected_root:
            raise ValueError(
                "cluster input does not correspond to a pairwise-audit block"
            )
        old_key = old_key_by_selected_root[selected_root]
        old_manifest = boundaries[old_key].manifest
        for name in ("selectedPatchDataSha256", "candidateDataSha256"):
            if (
                boundary_manifest["identity"].get(name)
                != old_manifest["identity"].get(name)
            ):
                raise ValueError("cluster and pairwise block evidence differs")
        cluster_keys.append(old_key)
    if len(set(cluster_keys)) != len(boundaries) or set(cluster_keys) != set(
        boundaries
    ):
        raise ValueError("cluster does not contain every pairwise-audit block once")

    with np.load(artifact_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    decisions: dict[CellKey, tuple[int, bool]] = {}
    for block, cell, configuration, changed in zip(
        arrays["selectedCellInput"],
        arrays["selectedCellLocalXYZ"],
        arrays["selectedSourceConfigurationIndex"],
        arrays["selectedConfigurationChanged"],
    ):
        block_index = int(block)
        if not 0 <= block_index < len(cluster_keys):
            raise ValueError("cluster decision references an invalid block")
        key = (cluster_keys[block_index], _int3(cell))
        if key in decisions:
            raise ValueError("cluster selects one physical cell more than once")
        decisions[key] = (int(configuration), bool(changed))

    patch_by_id: dict[int, PatchKey] = {}
    patch_keys: set[PatchKey] = set()
    for patch_id, block, source_patch, source_configuration, layer in zip(
        arrays["patchId"],
        arrays["patchInput"],
        arrays["patchSourcePatchId"],
        arrays["patchSourceConfigurationIndex"],
        arrays["patchLayerIndex"],
    ):
        block_index = int(block)
        if not 0 <= block_index < len(cluster_keys):
            raise ValueError("cluster patch references an invalid block")
        key = _patch_key(
            cluster_keys[block_index],
            int(source_patch),
            int(source_configuration),
            int(layer),
        )
        if key in patch_keys:
            raise ValueError("cluster realizes one physical patch more than once")
        patch_by_id[int(patch_id)] = key
        patch_keys.add(key)
    cluster_grid = manifest["grid"]
    cluster_offset = _lattice_offset(
        _float3(cluster_grid["originXYZ"]),
        global_origin_xyz,
        cell_size_xyz,
    )
    join_keys: set[JoinKey] = set()
    for first, second, axis, anchor in zip(
        arrays["joinFirstPatchId"],
        arrays["joinSecondPatchId"],
        arrays["joinFaceAxis"],
        arrays["joinFaceAnchorXYZ"],
    ):
        pair = _sorted_patch_pair(
            patch_by_id[int(first)], patch_by_id[int(second)]
        )
        local_anchor = _int3(anchor)
        global_anchor = tuple(
            local_anchor[value] + cluster_offset[value] for value in range(3)
        )
        join_keys.add((pair[0], pair[1], int(axis), global_anchor))
    component_by_patch = {
        patch_by_id[int(patch_id)]: int(component_id)
        for patch_id, component_id in zip(
            arrays["componentPatchId"], arrays["componentId"]
        )
    }
    if set(component_by_patch) != patch_keys:
        raise ValueError("cluster component map does not cover every patch")
    return _SolutionRecord(
        -1,
        root,
        manifest,
        arrays,
        tuple(cluster_keys),  # type: ignore[arg-type]
        decisions,
        patch_keys,
        join_keys,
        component_by_patch,
    )


def _cluster_configuration_comparison(
    solutions: tuple[_SolutionRecord, ...],
    cluster: _SolutionRecord,
    boundaries: Mapping[str, _BoundaryRecord],
) -> dict[str, Any]:
    observations: dict[CellKey, list[tuple[int, int, bool]]] = defaultdict(list)
    for solution in solutions:
        for key, (configuration, changed) in solution.decisions.items():
            observations[key].append((solution.index, configuration, changed))
    missing = set(observations) - set(cluster.decisions)
    extra = set(cluster.decisions) - set(observations)
    if missing or extra:
        raise ValueError(
            "cluster and pairwise networks cover different mutable-cell unions"
        )
    overlap = {key: value for key, value in observations.items() if len(value) > 1}
    disagreements = {
        key: value
        for key, value in overlap.items()
        if len({item[1] for item in value}) > 1
    }
    disagreement_records: list[dict[str, Any]] = []
    resolved_to_pairwise = 0
    resolved_to_new = 0
    for (boundary_key, cell), values in sorted(disagreements.items()):
        cluster_configuration, cluster_changed = cluster.decisions[
            (boundary_key, cell)
        ]
        pairwise_values = {value[1] for value in values}
        if cluster_configuration in pairwise_values:
            resolved_to_pairwise += 1
            classification = "selected-one-pairwise-alternative"
        else:
            resolved_to_new += 1
            classification = "selected-new-joint-alternative"
        disagreement_records.append(
            {
                "boundaryIdentitySha256": boundary_key,
                "boundaryRoot": str(boundaries[boundary_key].root),
                "cellLocalXYZ": list(cell),
                "pairwiseSelections": [
                    {
                        "solutionIndex": solution,
                        "sourceConfigurationIndex": configuration,
                        "changedFromBlockSelection": changed,
                    }
                    for solution, configuration, changed in values
                ],
                "clusterSelection": {
                    "sourceConfigurationIndex": cluster_configuration,
                    "changedFromBlockSelection": cluster_changed,
                },
                "classification": classification,
            }
        )
    agreed_overlap = set(overlap) - set(disagreements)
    cluster_differs_on_agreed = sum(
        cluster.decisions[key][0] != values[0][1]
        for key, values in overlap.items()
        if key in agreed_overlap
    )
    per_solution = []
    for solution in solutions:
        mismatches = sum(
            cluster.decisions[key][0] != value[0]
            for key, value in solution.decisions.items()
        )
        per_solution.append(
            {
                "solutionIndex": solution.index,
                "root": str(solution.root),
                "sharedMutableCells": len(solution.decisions),
                "matchingSelections": len(solution.decisions) - mismatches,
                "differentSelections": mismatches,
            }
        )
    return {
        "mutableCells": len(cluster.decisions),
        "pairwiseOverlapCells": len(overlap),
        "pairwiseAgreementCells": len(overlap) - len(disagreements),
        "pairwiseDisagreementCells": len(disagreements),
        "disagreementsResolvedToPairwiseAlternative": resolved_to_pairwise,
        "disagreementsResolvedToNewJointAlternative": resolved_to_new,
        "clusterDiffersOnPairwiseAgreementCells": cluster_differs_on_agreed,
        "clusterChangedConfigurations": sum(
            changed for _, changed in cluster.decisions.values()
        ),
        "perPairwiseSolution": per_solution,
        "disagreementResolutions": disagreement_records,
    }


def _component_pair_counts(
    first: _SolutionRecord,
    second: _SolutionRecord,
    common_patches: set[PatchKey],
) -> tuple[int, int, int, int]:
    contingency = Counter(
        (
            first.component_by_patch[patch],
            second.component_by_patch[patch],
        )
        for patch in common_patches
    )
    first_sizes = Counter(
        first.component_by_patch[patch] for patch in common_patches
    )
    second_sizes = Counter(
        second.component_by_patch[patch] for patch in common_patches
    )

    def pair_count(values: Mapping[Hashable, int]) -> int:
        return sum(value * (value - 1) // 2 for value in values.values())

    same_first = pair_count(first_sizes)
    same_second = pair_count(second_sizes)
    same_both = pair_count(contingency)
    return (
        same_first,
        same_second,
        same_both,
        same_first + same_second - 2 * same_both,
    )


def _cluster_topology_comparison(
    solutions: tuple[_SolutionRecord, ...],
    cluster: _SolutionRecord,
) -> dict[str, Any]:
    records: list[dict[str, Any]] = []
    for solution in solutions:
        common_patches = solution.patch_keys & cluster.patch_keys
        pairwise_joins = {
            value
            for value in solution.join_keys
            if value[0] in common_patches and value[1] in common_patches
        }
        cluster_joins = {
            value
            for value in cluster.join_keys
            if value[0] in common_patches and value[1] in common_patches
        }
        overlap = pairwise_joins & cluster_joins
        first_pairs, cluster_pairs, common_pairs, disagreements = (
            _component_pair_counts(solution, cluster, common_patches)
        )
        records.append(
            {
                "solutionIndex": solution.index,
                "root": str(solution.root),
                "commonPhysicalPatches": len(common_patches),
                "pairwiseEligibleJoins": len(pairwise_joins),
                "clusterEligibleJoins": len(cluster_joins),
                "exactJoins": len(overlap),
                "pairwiseOnlyJoins": len(pairwise_joins - cluster_joins),
                "clusterOnlyJoins": len(cluster_joins - pairwise_joins),
                "joinJaccard": round(
                    len(overlap) / max(len(pairwise_joins | cluster_joins), 1),
                    7,
                ),
                "coComponentPairsPairwise": first_pairs,
                "coComponentPairsCluster": cluster_pairs,
                "coComponentPairsBoth": common_pairs,
                "coComponentPairDisagreements": disagreements,
                "componentPartitionsAgree": disagreements == 0,
            }
        )
    return {
        "clusterPhysicalPatches": len(cluster.patch_keys),
        "clusterRetainedJoins": len(cluster.join_keys),
        "comparisons": records,
    }


def run_multiseam_audit(
    reselection_roots: tuple[str | Path, ...],
    output_path: str | Path,
    *,
    cluster_root: str | Path | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Audit overlapping decisions from a network of pairwise band solves."""

    roots = tuple(Path(value).resolve() for value in reselection_roots)
    if len(roots) < 2 or len(set(roots)) != len(roots):
        raise ValueError("multi-seam audit requires at least two distinct solutions")
    output = Path(output_path).resolve()
    cluster = None if cluster_root is None else Path(cluster_root).resolve()
    identity: dict[str, Any] = {
        "schema": MULTISEAM_AUDIT_SCHEMA,
        "version": MULTISEAM_AUDIT_VERSION,
        "inputs": [
            {
                "root": str(root),
                "manifestSha256": sha256_file(
                    root / "boundary-reselection-v1.json"
                ),
                "artifactSha256": sha256_file(
                    root / "boundary-reselection-v1.npz"
                ),
            }
            for root in roots
        ],
        "implementationSha256": sha256_file(Path(__file__)),
    }
    if cluster is not None:
        identity["cluster"] = {
            "root": str(cluster),
            "manifestSha256": sha256_file(
                cluster / "cluster-reselection-v1.json"
            ),
            "artifactSha256": sha256_file(
                cluster / "cluster-reselection-v1.npz"
            ),
        }
    identity["identitySha256"] = canonical_json_hash(identity)
    if output.is_file():
        prior = json.loads(output.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("multi-seam audit output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior

    solutions, boundaries, global_origin, cell_size = _load_records(roots)
    offsets = {
        key: _lattice_offset(value.origin_xyz, global_origin, cell_size)
        for key, value in boundaries.items()
    }
    global_shape = tuple(
        max(
            offsets[key][axis] + boundary.shape_cells_xyz[axis]
            for key, boundary in boundaries.items()
        )
        for axis in range(3)
    )
    cluster_solution = (
        None
        if cluster is None
        else _load_cluster_solution(
            cluster,
            boundaries,
            global_origin,
            cell_size,
        )
    )
    result: dict[str, Any] = {
        "schema": MULTISEAM_AUDIT_SCHEMA,
        "version": MULTISEAM_AUDIT_VERSION,
        "state": "complete",
        "identity": identity,
        "layout": {
            "blocks": len(boundaries),
            "pairwiseSolutions": len(solutions),
            "globalOriginXYZ": list(global_origin),
            "cellSizeXYZ": list(cell_size),
            "shapeCellsXYZ": list(global_shape),
            "boundaries": [
                {
                    "boundaryIdentitySha256": key,
                    "root": str(boundaries[key].root),
                    "offsetCellsXYZ": list(offsets[key]),
                    "shapeCellsXYZ": list(boundaries[key].shape_cells_xyz),
                    "seamUses": sum(
                        key in solution.boundary_keys for solution in solutions
                    ),
                }
                for key in sorted(boundaries)
            ],
            "solutions": [
                {
                    "index": value.index,
                    "root": str(value.root),
                    "axis": int(value.manifest["adjacency"]["axis"]),
                    "boundaryIdentitySha256": list(value.boundary_keys),
                    "mutableCells": len(value.decisions),
                    "retainedBandJoins": len(value.join_keys),
                }
                for value in solutions
            ],
        },
        "configurationConsistency": _configuration_consistency(
            solutions, boundaries
        ),
        "topologyConsistency": _topology_consistency(solutions),
        "storage": _storage_summary(solutions, boundaries),
        "interpretation": (
            "Overlapping seam solves are locally composable only when they "
            "select one physical configuration per shared cell and agree on "
            "retained joins among their common physical patches. Disagreement "
            "identifies the bounded corner domain that needs a joint solve."
        ),
    }
    if cluster_solution is not None:
        result["clusterComparison"] = {
            "root": str(cluster_solution.root),
            "configuration": _cluster_configuration_comparison(
                solutions, cluster_solution, boundaries
            ),
            "topology": _cluster_topology_comparison(
                solutions, cluster_solution
            ),
        }
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(output, result)
    return result
