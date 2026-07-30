from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import clip_plane_to_cell
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


def _identity(source: Path, start: Int3, stop: Int3) -> dict[str, Any]:
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
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "subblock.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


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
    identity = _identity(source, start, stop)
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
    manifest.update(
        {
            "state": "complete",
            "grid": patch_manifest["grid"],
            "patches": patch_manifest["counts"]["patches"],
            "artifact": {
                "manifest": "selected-patches-v1.json",
                "data": "selected-patches-v1.npz",
                "sha256": patch_manifest["data"]["sha256"],
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
