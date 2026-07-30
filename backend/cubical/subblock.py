from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import clip_plane_to_cell
from .saturation_selection import load_saturation_candidates
from .stratigraphy import ConfigurationTable
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridSpec, Int3


SUBBLOCK_SCHEMA = "pareidolia.cubical-selected-patch-subblock"
SUBBLOCK_VERSION = 1


def _validate_bounds(start: Int3, stop: Int3, shape: Int3) -> None:
    if any(
        start[axis] < 0
        or stop[axis] <= start[axis]
        or stop[axis] > shape[axis]
        for axis in range(3)
    ):
        raise ValueError("subblock bounds must be a positive subset of the grid")


def _candidate_root(source: Path) -> Path | None:
    if (source / "saturation-configurations-v1.json").is_file():
        return source
    summary_path = source / "summary.json"
    if summary_path.is_file():
        value = json.loads(summary_path.read_text()).get("candidateRoot")
        if value:
            return Path(value).resolve()
    return None


def _identity(
    source: Path,
    start: Int3,
    stop: Int3,
    candidate_root: Path | None,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": SUBBLOCK_SCHEMA,
        "version": SUBBLOCK_VERSION,
        "sourceRoot": str(source),
        "sourcePatchManifestSha256": sha256_file(
            source / "selected-patches-v1.json"
        ),
        "sourcePatchDataSha256": sha256_file(
            source / "selected-patches-v1.npz"
        ),
        "startCellXYZ": list(start),
        "stopCellXYZExclusive": list(stop),
        "candidateRoot": None if candidate_root is None else str(candidate_root),
        "candidateDataSha256": (
            None
            if candidate_root is None
            else sha256_file(candidate_root / "saturation-configurations-v1.npz")
        ),
        "selectionDataSha256": (
            sha256_file(source / "selection-v1.npz")
            if candidate_root is not None
            else None
        ),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "subblock.py",
                "saturation_selection.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def _configuration_subblock(
    source: Path,
    candidate_root: Path,
    output: Path,
    start: Int3,
    stop: Int3,
    identity_sha256: str,
) -> dict[str, Any]:
    selection_path = source / "selection-v1.npz"
    selection_manifest_path = source / "selection-v1.json"
    if not selection_path.is_file() or not selection_manifest_path.is_file():
        raise ValueError("candidate-bearing subblock requires selected configurations")
    selection_manifest = json.loads(selection_manifest_path.read_text())
    if (
        selection_manifest.get("schema")
        != "pareidolia.raw-acus-configuration-selection"
        or sha256_file(selection_path) != selection_manifest["data"]["sha256"]
    ):
        raise ValueError("source selected-configuration artifact is invalid")
    table, metadata, source_manifest = load_saturation_candidates(candidate_root)
    variant_path = source / "variant.json"
    if variant_path.is_file():
        variant = json.loads(variant_path.read_text())
        expected_candidate_sha256 = variant.get("identity", {}).get(
            "candidateDataSha256"
        )
        if (
            expected_candidate_sha256 is not None
            and expected_candidate_sha256 != source_manifest["data"]["sha256"]
        ):
            raise ValueError(
                "candidate bank differs from the selected configuration source"
            )
    cell_indices = [
        index
        for index, cell in enumerate(table.cell_xyz)
        if all(
            start[axis] <= int(cell[axis]) < stop[axis]
            for axis in range(3)
        )
    ]
    configuration_indices: list[int] = []
    configuration_offset = np.zeros(len(cell_indices) + 1, dtype=np.uint64)
    for output_index, source_index in enumerate(cell_indices):
        configuration_indices.extend(table.configurations_for_cell(source_index))
        configuration_offset[output_index + 1] = len(configuration_indices)
    configuration_index = np.asarray(configuration_indices, dtype=np.int64)
    layer_indices: list[int] = []
    layer_offset = np.zeros(len(configuration_indices) + 1, dtype=np.uint64)
    for output_index, source_index in enumerate(configuration_indices):
        low = int(table.layer_offset[source_index])
        high = int(table.layer_offset[source_index + 1])
        layer_indices.extend(range(low, high))
        layer_offset[output_index + 1] = len(layer_indices)
    layer_index = np.asarray(layer_indices, dtype=np.int64)
    rebased_cells = table.cell_xyz[np.asarray(cell_indices, dtype=np.int64)].copy()
    rebased_cells -= np.asarray(start, dtype=np.int32)
    subset = ConfigurationTable(
        rebased_cells,
        configuration_offset,
        table.configuration_id[configuration_index],
        table.configuration_log_weight[configuration_index],
        table.normal_hypothesis[configuration_index],
        layer_offset,
        table.layer_normal_xyz[layer_index],
        table.layer_height[layer_index],
        table.layer_covariance[layer_index],
        table.layer_fiber_xyz[layer_index],
        table.layer_fiber_angular_std_radians[layer_index],
        table.layer_confidence[layer_index],
        table.layer_evidence_score[layer_index],
        table.layer_material_probability[layer_index],
        table.layer_effective_support[layer_index],
    )
    subset.validate()
    subset_metadata = {
        name: np.asarray(values)[configuration_index]
        for name, values in metadata.items()
    }
    candidate_path = output / "saturation-configurations-v1.npz"
    temporary = candidate_path.with_suffix(candidate_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **subset.arrays(), **subset_metadata)
    temporary.replace(candidate_path)
    candidate_sha256 = sha256_file(candidate_path)
    candidate_identity = canonical_json_hash(
        {
            "sourceIdentitySha256": source_manifest["identitySha256"],
            "sourceDataSha256": source_manifest["data"]["sha256"],
            "subblockIdentitySha256": identity_sha256,
        }
    )
    candidate_manifest = {
        "schema": "pareidolia.cubical-saturation-configurations",
        "version": 1,
        "identitySha256": candidate_identity,
        "sourceRoot": str(candidate_root),
        "statistics": {
            "cells": subset.cell_count,
            "retainedConfigurations": subset.configuration_count,
            "retainedLayers": subset.layer_count,
        },
        "data": {
            "path": candidate_path.name,
            "bytes": candidate_path.stat().st_size,
            "sha256": candidate_sha256,
        },
    }
    atomic_json(output / "saturation-configurations-v1.json", candidate_manifest)

    with np.load(selection_path) as values:
        selection = {name: np.asarray(values[name]) for name in values.files}
    selected_by_cell = {
        tuple(int(value) for value in cell): row
        for row, cell in enumerate(selection["cellXYZ"])
    }
    candidate_cells = {
        tuple(int(value) for value in cell) for cell in table.cell_xyz
    }
    if (
        len(selected_by_cell) != table.cell_count
        or set(selected_by_cell) != candidate_cells
    ):
        raise ValueError("selected configuration cells do not span the candidate bank")
    selected_source = np.asarray(
        selection["sourceConfigurationIndex"], dtype=np.uint64
    )
    if np.any(selected_source >= table.configuration_count):
        raise ValueError("selected configuration index lies outside candidate bank")
    configuration_cell_index = np.repeat(
        np.arange(table.cell_count, dtype=np.int64),
        np.diff(table.configuration_offset.astype(np.int64)),
    )
    source_cells = table.cell_xyz[
        configuration_cell_index[selected_source.astype(np.int64)]
    ]
    if not np.array_equal(source_cells, selection["cellXYZ"]):
        raise ValueError("selected configuration index belongs to another cell")
    old_to_new = {
        int(source_index): new_index
        for new_index, source_index in enumerate(configuration_indices)
    }
    selected_rows = np.asarray(
        [
            selected_by_cell[
                tuple(int(value) for value in table.cell_xyz[source_index])
            ]
            for source_index in cell_indices
        ],
        dtype=np.int64,
    )
    old_selected = selected_source[selected_rows]
    try:
        new_selected = np.asarray(
            [old_to_new[int(value)] for value in old_selected], dtype=np.uint32
        )
    except KeyError as error:
        raise ValueError("selected configuration is absent from its subblock bank") from error
    selection_values = {
        "cellXYZ": rebased_cells,
        "optionId": np.arange(len(rebased_cells), dtype=np.uint64),
        "sourceTableIndex": np.zeros(len(rebased_cells), dtype=np.uint32),
        "sourceConfigurationIndex": new_selected,
        "localConfigurationId": np.asarray(
            selection["localConfigurationId"]
        )[selected_rows],
        "configurationLogWeight": np.asarray(
            selection["configurationLogWeight"]
        )[selected_rows],
        "selectedLayerCount": np.asarray(selection["selectedLayerCount"])[
            selected_rows
        ],
    }
    subblock_selection_path = output / "selection-v1.npz"
    temporary = subblock_selection_path.with_suffix(
        subblock_selection_path.suffix + ".tmp"
    )
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **selection_values)
    temporary.replace(subblock_selection_path)
    selection_sha256 = sha256_file(subblock_selection_path)
    atomic_json(
        output / "selection-v1.json",
        {
            "schema": "pareidolia.raw-acus-configuration-selection",
            "version": 1,
            "identitySha256": identity_sha256,
            "statistics": {
                "cellCount": len(rebased_cells),
                "nonemptyCellCount": int(
                    np.count_nonzero(selection_values["selectedLayerCount"])
                ),
                "selectedLayerCount": int(
                    np.sum(selection_values["selectedLayerCount"])
                ),
            },
            "data": {
                "path": subblock_selection_path.name,
                "bytes": subblock_selection_path.stat().st_size,
                "sha256": selection_sha256,
            },
        },
    )
    atomic_json(
        output / "variant.json",
        {
            "schema": SUBBLOCK_SCHEMA,
            "version": SUBBLOCK_VERSION,
            "state": "complete",
            "identity": {
                "identitySha256": identity_sha256,
                "candidateDataSha256": candidate_sha256,
            },
        },
    )
    return {
        "candidateRoot": str(output),
        "cells": subset.cell_count,
        "configurations": subset.configuration_count,
        "layers": subset.layer_count,
        "candidateDataSha256": candidate_sha256,
        "selectionDataSha256": selection_sha256,
    }


def extract_selected_patch_subblock(
    source_root: str | Path,
    output_root: str | Path,
    *,
    start_cell_xyz: Int3,
    stop_cell_xyz_exclusive: Int3,
    force: bool = False,
) -> dict[str, Any]:
    """Rebase one selected-patch subset for deterministic composition audits.

    This is a geometry-contract utility. It does not replace running Acus and
    physical selection independently on a new raw-CT block.
    """

    source = Path(source_root).resolve()
    output = Path(output_root).resolve()
    if source == output:
        raise ValueError("subblock output must differ from its source")
    start = tuple(int(value) for value in start_cell_xyz)
    stop = tuple(int(value) for value in stop_cell_xyz_exclusive)
    if len(start) != 3 or len(stop) != 3:
        raise ValueError("subblock bounds require XYZ triples")
    table = read_patch_shard(source / "selected-patches-v1", verify=True)
    _validate_bounds(start, stop, table.grid.shape_cells_xyz)
    candidates = _candidate_root(source)
    if candidates is not None and not (source / "selection-v1.npz").is_file():
        candidates = None
    identity = _identity(source, start, stop, candidates)
    manifest_path = output / "subblock.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("subblock output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": SUBBLOCK_SCHEMA,
        "version": SUBBLOCK_VERSION,
        "state": "building",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    source_patches = table.to_patches()
    source_by_id = {value.patch_id: value for value in source_patches}
    rows = np.asarray(
        [
            index
            for index, cell in enumerate(table.cell_xyz)
            if all(
                start[axis] <= int(cell[axis]) < stop[axis]
                for axis in range(3)
            )
        ],
        dtype=np.int64,
    )
    shape = tuple(stop[axis] - start[axis] for axis in range(3))
    origin = tuple(
        float(value) for value in table.grid.vertex_world(start)
    )
    grid = GridSpec(
        shape,
        table.grid.cell_size_xyz,
        origin,
        table.grid.coordinate_unit,
    )
    patches = []
    for row in rows:
        patch_id = int(table.patch_id[row])
        source_patch = source_by_id[patch_id]
        cell = tuple(
            source_patch.cell_xyz[axis] - start[axis]
            for axis in range(3)
        )
        patch = clip_plane_to_cell(
            grid,
            cell,
            source_patch.estimate,
            patch_id=patch_id,
        )
        if patch is None:
            raise ValueError("rebased subblock plane no longer intersects its cell")
        patches.append(patch)
    subset = PatchTable.from_patches(
        grid,
        tuple(patches),
        configuration_id={
            int(table.patch_id[row]): int(table.configuration_id[row])
            for row in rows
        },
        configuration_log_weight={
            int(table.patch_id[row]): float(table.configuration_log_weight[row])
            for row in rows
        },
        local_order={
            int(table.patch_id[row]): int(table.local_order[row]) for row in rows
        },
        normal_family={
            int(table.patch_id[row]): int(table.normal_family[row]) for row in rows
        },
    )
    patch_manifest = write_patch_shard(
        output / "selected-patches-v1",
        subset,
        settings={
            "start_cell_xyz": list(start),
            "stop_cell_xyz_exclusive": list(stop),
        },
        provenance={
            "sourceRoot": str(source),
            "sourceIdentitySha256": identity["identitySha256"],
            "purpose": "deterministic geometry-contract composition audit",
        },
        compressed=True,
    )
    configuration_artifact = (
        None
        if candidates is None
        else _configuration_subblock(
            source,
            candidates,
            output,
            start,
            stop,
            str(identity["identitySha256"]),
        )
    )
    manifest.update(
        {
            "state": "complete",
            "grid": patch_manifest["grid"],
            "patches": patch_manifest["counts"]["patches"],
            "configurations": configuration_artifact,
            "artifact": {
                "manifest": "selected-patches-v1.json",
                "data": "selected-patches-v1.npz",
                "sha256": patch_manifest["data"]["sha256"],
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
