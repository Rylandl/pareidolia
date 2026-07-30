from __future__ import annotations

import hashlib
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import (
    ClippedPatch,
    DegeneratePlaneIntersection,
    PlaneEstimate,
    clip_plane_to_cell,
)
from .saturation_selection import load_saturation_candidates
from .tables import PatchTable, read_patch_shard, write_patch_shard
from .topology import GridSpec, Int3


SHEET_EVIDENCE_SCHEMA = "pareidolia.cubical-block-sheet-evidence"
SHEET_EVIDENCE_VERSION = 1
SHEET_EVIDENCE_STEM = "sheet-evidence-v1"


@dataclass(frozen=True, slots=True)
class SheetEvidenceInput:
    candidate_root: Path
    offset_cells_xyz: Int3 = (0, 0, 0)

    def __post_init__(self) -> None:
        root = Path(self.candidate_root).resolve()
        offset = tuple(int(value) for value in self.offset_cells_xyz)
        if len(offset) != 3 or any(value < 0 for value in offset):
            raise ValueError("sheet-evidence offsets must be nonnegative XYZ triples")
        object.__setattr__(self, "candidate_root", root)
        object.__setattr__(self, "offset_cells_xyz", offset)

    def record(self) -> dict[str, Any]:
        return {
            "candidateRoot": str(self.candidate_root),
            "offsetCellsXYZ": list(self.offset_cells_xyz),
        }


@dataclass(frozen=True, slots=True)
class SheetEvidenceSettings:
    clipping_tolerance_scale: float = 1.0e-8

    def __post_init__(self) -> None:
        if (
            not math.isfinite(self.clipping_tolerance_scale)
            or self.clipping_tolerance_scale <= 0.0
        ):
            raise ValueError("clipping tolerance scale must be finite and positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True, slots=True)
class BlockSheetEvidence:
    grid: GridSpec
    arrays: Mapping[str, np.ndarray]
    mode_patches: PatchTable
    manifest: Mapping[str, Any]

    @property
    def cell_count(self) -> int:
        return int(len(self.arrays["cellXYZ"]))

    @property
    def mode_count(self) -> int:
        return int(len(self.arrays["modeId"]))

    @property
    def configuration_count(self) -> int:
        return int(len(self.arrays["configurationId"]))

    def validate(self) -> None:
        arrays = self.arrays
        required = (
            "cellXYZ",
            "modeId",
            "modeCellXYZ",
            "modeNormalHypothesis",
            "modeGeometryStatus",
            "configurationOffset",
            "configurationId",
            "configurationNormalHypothesis",
            "configurationIsCurrent",
            "configurationGeometryValid",
            "configurationModeOffset",
            "configurationModeId",
        )
        if any(name not in arrays for name in required):
            raise ValueError("block sheet evidence lacks required arrays")
        cells = np.asarray(arrays["cellXYZ"], dtype=np.int64)
        mode_ids = np.asarray(arrays["modeId"], dtype=np.uint64)
        mode_cells = np.asarray(arrays["modeCellXYZ"], dtype=np.int64)
        status = np.asarray(arrays["modeGeometryStatus"], dtype=np.uint8)
        configuration_offset = np.asarray(
            arrays["configurationOffset"], dtype=np.uint64
        )
        configuration_ids = np.asarray(arrays["configurationId"], dtype=np.uint64)
        membership_offset = np.asarray(
            arrays["configurationModeOffset"], dtype=np.uint64
        )
        membership = np.asarray(arrays["configurationModeId"], dtype=np.uint64)
        if cells.shape != (self.cell_count, 3) or len(np.unique(cells, axis=0)) != len(cells):
            raise ValueError("sheet-evidence cells must be unique XYZ rows")
        if mode_cells.shape != (self.mode_count, 3) or len(np.unique(mode_ids)) != len(mode_ids):
            raise ValueError("sheet-evidence modes must have unique IDs and XYZ cells")
        if status.shape != (self.mode_count,) or np.any(status > 2):
            raise ValueError("sheet-evidence mode geometry statuses are invalid")
        if (
            configuration_offset.shape != (self.cell_count + 1,)
            or int(configuration_offset[0]) != 0
            or int(configuration_offset[-1]) != self.configuration_count
            or np.any(np.diff(configuration_offset) < 1)
        ):
            raise ValueError("sheet-evidence configuration offsets are invalid")
        if len(np.unique(configuration_ids)) != self.configuration_count:
            raise ValueError("sheet-evidence configuration IDs are not unique")
        if (
            membership_offset.shape != (self.configuration_count + 1,)
            or int(membership_offset[0]) != 0
            or int(membership_offset[-1]) != len(membership)
            or np.any(np.diff(membership_offset) < 0)
        ):
            raise ValueError("sheet-evidence mode-membership offsets are invalid")
        known_modes = {int(value) for value in mode_ids}
        if any(int(value) not in known_modes for value in membership):
            raise ValueError("sheet-evidence configurations reference absent modes")
        cell_set = {tuple(int(value) for value in row) for row in cells}
        if any(
            tuple(int(value) for value in row) not in cell_set for row in mode_cells
        ):
            raise ValueError("sheet-evidence modes reference absent owned cells")
        if any(
            not all(0 <= int(row[axis]) < self.grid.shape_cells_xyz[axis] for axis in range(3))
            for row in cells
        ):
            raise ValueError("sheet-evidence cells fall outside the declared grid")
        mode_index = {int(value): index for index, value in enumerate(mode_ids)}
        configuration_cell_index = np.repeat(
            np.arange(self.cell_count, dtype=np.int64),
            np.diff(configuration_offset).astype(np.int64),
        )
        config_family = np.asarray(
            arrays["configurationNormalHypothesis"], dtype=np.int16
        )
        mode_family = np.asarray(arrays["modeNormalHypothesis"], dtype=np.int16)
        declared_valid = np.asarray(
            arrays["configurationGeometryValid"], dtype=np.uint8
        )
        for configuration_index, cell_index in enumerate(configuration_cell_index):
            low = int(membership_offset[configuration_index])
            high = int(membership_offset[configuration_index + 1])
            indices = [mode_index[int(value)] for value in membership[low:high]]
            if any(
                tuple(int(value) for value in mode_cells[index])
                != tuple(int(value) for value in cells[cell_index])
                for index in indices
            ):
                raise ValueError("one configuration spans multiple cubical cells")
            if indices and any(
                int(mode_family[index]) != int(config_family[configuration_index])
                for index in indices
            ):
                raise ValueError("configuration modes span multiple normal families")
            valid = all(int(status[index]) == 0 for index in indices)
            if bool(declared_valid[configuration_index]) != valid:
                raise ValueError("configuration geometry-valid flag is inconsistent")
        current = np.asarray(arrays["configurationIsCurrent"], dtype=np.uint8)
        if current.shape != (self.configuration_count,) or any(
            int(np.sum(current[int(low):int(high)])) != 1
            for low, high in zip(configuration_offset[:-1], configuration_offset[1:])
        ):
            raise ValueError("each sheet-evidence cell must identify one current stack")
        valid_patch_ids = {
            int(mode_ids[index]) for index in np.flatnonzero(status == 0)
        }
        if {int(value) for value in self.mode_patches.patch_id} != valid_patch_ids:
            raise ValueError("mode patch shard does not match valid evidence modes")


@dataclass(frozen=True, slots=True)
class _InputBank:
    spec: SheetEvidenceInput
    table: Any
    arrays: dict[str, np.ndarray]
    metadata: dict[str, np.ndarray]
    candidate_manifest: dict[str, Any]
    variant: dict[str, Any]
    grid: GridSpec
    mode_bank_identity_sha256: str


@dataclass(frozen=True, slots=True)
class _ModeRecord:
    mode_id: int
    cell_xyz: Int3
    input_index: int
    source_shard_index: int
    source_mode_index: int
    normal_hypothesis: int
    normal_xyz: tuple[float, float, float]
    height: float
    covariance: tuple[float, float, float, float, float, float]
    fiber_xyz: tuple[float, float, float]
    fiber_angular_std_radians: float
    confidence: float
    evidence_score: float
    material_probability: float
    effective_support: float

    def estimate(self) -> PlaneEstimate:
        packed = np.asarray(self.covariance, dtype=np.float64)
        covariance = np.zeros((3, 3), dtype=np.float64)
        covariance[(0, 0, 0, 1, 1, 2), (0, 1, 2, 1, 2, 2)] = packed
        covariance[(1, 2, 2), (0, 0, 1)] = packed[[1, 2, 4]]
        return PlaneEstimate(
            self.normal_xyz,
            self.height,
            tuple(tuple(float(value) for value in row) for row in covariance),
            self.fiber_xyz,
            self.fiber_angular_std_radians,
            self.confidence,
        )


def _stable_uint64(*values: object) -> int:
    digest = hashlib.blake2b(
        json.dumps(values, separators=(",", ":"), sort_keys=True).encode("utf-8"),
        digest_size=8,
    ).digest()
    return int.from_bytes(digest, "little")


def _grid_from_patch_manifest(path: Path) -> GridSpec:
    manifest = json.loads(path.read_text())
    grid = manifest["grid"]
    return GridSpec(
        tuple(int(value) for value in grid["shapeCellsXYZ"]),
        tuple(float(value) for value in grid["cellSizeXYZ"]),
        tuple(float(value) for value in grid["originXYZ"]),
        str(grid["coordinateUnit"]),
    )


def _load_input(spec: SheetEvidenceInput) -> _InputBank:
    table, metadata, candidate_manifest = load_saturation_candidates(
        spec.candidate_root,
        verify=True,
    )
    variant = json.loads((spec.candidate_root / "variant.json").read_text())
    if variant.get("state") != "complete":
        raise ValueError("sheet evidence requires completed saturation candidates")
    input_root = Path(variant["inputRoot"]).resolve()
    grid = _grid_from_patch_manifest(input_root / "selected-patches-v1.json")
    mode_bank_root = Path(variant["modeBankRoot"]).resolve()
    mode_bank = json.loads((mode_bank_root / "mode-bank.json").read_text())
    mode_identity = str(mode_bank["identity"]["identitySha256"])
    data_path = spec.candidate_root / str(candidate_manifest["data"]["path"])
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    required = (
        "sourceShardIndex",
        "sourceModeOffset",
        "sourceModeIndex",
        "shardNames",
    )
    if any(name not in arrays for name in required):
        raise ValueError("saturation candidates lack immutable source-mode references")
    if not np.array_equal(arrays["sourceModeOffset"], table.layer_offset):
        raise ValueError("source-mode membership does not align with configuration layers")
    if len(arrays["sourceShardIndex"]) != table.configuration_count:
        raise ValueError("source-shard membership does not align with configurations")
    return _InputBank(
        spec,
        table,
        arrays,
        metadata,
        candidate_manifest,
        variant,
        grid,
        mode_identity,
    )


def _merged_grid(inputs: tuple[_InputBank, ...]) -> GridSpec:
    if not inputs:
        raise ValueError("sheet evidence requires at least one candidate input")
    reference = inputs[0]
    size = np.asarray(reference.grid.cell_size_xyz, dtype=np.float64)
    base_origin = np.asarray(reference.grid.origin_xyz, dtype=np.float64) - size * np.asarray(
        reference.spec.offset_cells_xyz,
        dtype=np.float64,
    )
    stop = np.zeros(3, dtype=np.int64)
    for value in inputs:
        if value.grid.coordinate_unit != reference.grid.coordinate_unit or not np.allclose(
            value.grid.cell_size_xyz,
            reference.grid.cell_size_xyz,
            atol=1.0e-10,
            rtol=0.0,
        ):
            raise ValueError("sheet-evidence inputs use incompatible grids")
        inferred = np.asarray(value.grid.origin_xyz, dtype=np.float64) - size * np.asarray(
            value.spec.offset_cells_xyz,
            dtype=np.float64,
        )
        if not np.allclose(inferred, base_origin, atol=1.0e-5, rtol=0.0):
            raise ValueError("sheet-evidence offsets disagree with world-space origins")
        stop = np.maximum(
            stop,
            np.asarray(value.spec.offset_cells_xyz, dtype=np.int64)
            + np.asarray(value.grid.shape_cells_xyz, dtype=np.int64),
        )
    return GridSpec(
        tuple(int(value) for value in stop),
        tuple(float(value) for value in size),
        tuple(float(value) for value in base_origin),
        reference.grid.coordinate_unit,
    )


def _mode_matches(first: _ModeRecord, second: _ModeRecord) -> bool:
    if (
        first.cell_xyz != second.cell_xyz
        or first.normal_hypothesis != second.normal_hypothesis
    ):
        return False
    first_values = np.asarray(
        (
            *first.normal_xyz,
            first.height,
            *first.covariance,
            *first.fiber_xyz,
            first.fiber_angular_std_radians,
            first.confidence,
            first.evidence_score,
            first.material_probability,
            first.effective_support,
        ),
        dtype=np.float64,
    )
    second_values = np.asarray(
        (
            *second.normal_xyz,
            second.height,
            *second.covariance,
            *second.fiber_xyz,
            second.fiber_angular_std_radians,
            second.confidence,
            second.evidence_score,
            second.material_probability,
            second.effective_support,
        ),
        dtype=np.float64,
    )
    return bool(np.allclose(first_values, second_values, atol=2.0e-6, rtol=1.0e-6))


def _write_evidence_data(
    output: Path,
    *,
    cells: tuple[Int3, ...],
    modes: tuple[_ModeRecord, ...],
    mode_geometry_status: np.ndarray,
    configuration_offset: np.ndarray,
    configuration_records: tuple[dict[str, Any], ...],
    configuration_mode_offset: np.ndarray,
    configuration_mode_id: np.ndarray,
) -> dict[str, Any]:
    path = output / f"{SHEET_EVIDENCE_STEM}.npz"
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            cellXYZ=np.asarray(cells, dtype=np.int32).reshape(len(cells), 3),
            modeId=np.asarray([value.mode_id for value in modes], dtype=np.uint64),
            modeCellXYZ=np.asarray(
                [value.cell_xyz for value in modes], dtype=np.int32
            ).reshape(len(modes), 3),
            modeInputIndex=np.asarray(
                [value.input_index for value in modes], dtype=np.uint16
            ),
            modeSourceShardIndex=np.asarray(
                [value.source_shard_index for value in modes], dtype=np.int16
            ),
            modeSourceIndex=np.asarray(
                [value.source_mode_index for value in modes], dtype=np.int32
            ),
            modeNormalHypothesis=np.asarray(
                [value.normal_hypothesis for value in modes], dtype=np.int8
            ),
            modeNormalXYZ=np.asarray(
                [value.normal_xyz for value in modes], dtype=np.float32
            ).reshape(len(modes), 3),
            modeHeight=np.asarray([value.height for value in modes], dtype=np.float32),
            modeCovariance=np.asarray(
                [value.covariance for value in modes], dtype=np.float32
            ).reshape(len(modes), 6),
            modeFiberXYZ=np.asarray(
                [value.fiber_xyz for value in modes], dtype=np.float32
            ).reshape(len(modes), 3),
            modeFiberAngularStdRadians=np.asarray(
                [value.fiber_angular_std_radians for value in modes],
                dtype=np.float32,
            ),
            modeConfidence=np.asarray(
                [value.confidence for value in modes], dtype=np.float32
            ),
            modeEvidenceScore=np.asarray(
                [value.evidence_score for value in modes], dtype=np.float32
            ),
            modeMaterialProbability=np.asarray(
                [value.material_probability for value in modes], dtype=np.float32
            ),
            modeEffectiveSupport=np.asarray(
                [value.effective_support for value in modes], dtype=np.float32
            ),
            modeGeometryStatus=np.asarray(mode_geometry_status, dtype=np.uint8),
            configurationOffset=np.asarray(configuration_offset, dtype=np.uint64),
            configurationId=np.asarray(
                [value["configurationId"] for value in configuration_records],
                dtype=np.uint64,
            ),
            configurationInputIndex=np.asarray(
                [value["inputIndex"] for value in configuration_records],
                dtype=np.uint16,
            ),
            configurationSourceIndex=np.asarray(
                [value["sourceIndex"] for value in configuration_records],
                dtype=np.uint32,
            ),
            configurationLocalId=np.asarray(
                [value["localId"] for value in configuration_records],
                dtype=np.uint16,
            ),
            configurationLogWeight=np.asarray(
                [value["logWeight"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationNormalHypothesis=np.asarray(
                [value["normalHypothesis"] for value in configuration_records],
                dtype=np.int8,
            ),
            configurationEvidenceLogScore=np.asarray(
                [value["evidenceLogScore"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationPhysicalLogScore=np.asarray(
                [value["physicalLogScore"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationTotalLogScore=np.asarray(
                [value["totalLogScore"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationCoveredEvidenceMass=np.asarray(
                [value["coveredEvidenceMass"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationTotalEvidenceMass=np.asarray(
                [value["totalEvidenceMass"] for value in configuration_records],
                dtype=np.float32,
            ),
            configurationIsCurrent=np.asarray(
                [value["isCurrent"] for value in configuration_records],
                dtype=np.uint8,
            ),
            configurationGeometryValid=np.asarray(
                [value["geometryValid"] for value in configuration_records],
                dtype=np.uint8,
            ),
            configurationModeOffset=np.asarray(
                configuration_mode_offset, dtype=np.uint64
            ),
            configurationModeId=np.asarray(
                configuration_mode_id, dtype=np.uint64
            ),
        )
    temporary.replace(path)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": sha256_file(path),
    }


def read_block_sheet_evidence(
    root: str | Path,
    *,
    verify: bool = True,
) -> BlockSheetEvidence:
    source = Path(root).resolve()
    manifest = json.loads((source / f"{SHEET_EVIDENCE_STEM}.json").read_text())
    if (
        manifest.get("schema") != SHEET_EVIDENCE_SCHEMA
        or int(manifest.get("version", -1)) != SHEET_EVIDENCE_VERSION
        or manifest.get("state") != "complete"
    ):
        raise ValueError("unsupported or incomplete block sheet evidence")
    data_path = source / str(manifest["data"]["path"])
    if verify and sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("block sheet-evidence content hash mismatch")
    grid_record = manifest["identity"]["grid"]
    grid = GridSpec(
        tuple(int(value) for value in grid_record["shapeCellsXYZ"]),
        tuple(float(value) for value in grid_record["cellSizeXYZ"]),
        tuple(float(value) for value in grid_record["originXYZ"]),
        str(grid_record["coordinateUnit"]),
    )
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    mode_patches = read_patch_shard(source / "mode-patches-v1", verify=verify)
    if mode_patches.grid != grid:
        raise ValueError("sheet-evidence mode patches use another grid")
    evidence = BlockSheetEvidence(grid, arrays, mode_patches, manifest)
    evidence.validate()
    return evidence


def compile_block_sheet_evidence(
    inputs: Iterable[SheetEvidenceInput],
    output_root: str | Path,
    *,
    settings: SheetEvidenceSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Compile immutable Acus modes and physical stacks into one block contract."""

    started = time.monotonic()
    specs = tuple(inputs)
    if not specs:
        raise ValueError("sheet evidence requires at least one input")
    resolved = settings or SheetEvidenceSettings()
    output = Path(output_root).resolve()
    banks = tuple(_load_input(value) for value in specs)
    grid = _merged_grid(banks)
    module_root = Path(__file__).resolve().parent
    input_identity = []
    for bank in banks:
        data_path = bank.spec.candidate_root / str(
            bank.candidate_manifest["data"]["path"]
        )
        input_identity.append(
            {
                **bank.spec.record(),
                "candidateIdentitySha256": bank.candidate_manifest[
                    "identitySha256"
                ],
                "candidateDataSha256": sha256_file(data_path),
                "modeBankIdentitySha256": bank.mode_bank_identity_sha256,
            }
        )
    identity: dict[str, Any] = {
        "schema": SHEET_EVIDENCE_SCHEMA,
        "version": SHEET_EVIDENCE_VERSION,
        "inputs": input_identity,
        "grid": {
            "shapeCellsXYZ": list(grid.shape_cells_xyz),
            "cellSizeXYZ": list(grid.cell_size_xyz),
            "originXYZ": list(grid.origin_xyz),
            "coordinateUnit": grid.coordinate_unit,
        },
        "settings": resolved.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_evidence.py",
                "saturation_selection.py",
                "stratigraphy.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / f"{SHEET_EVIDENCE_STEM}.json"
    summary_path = output / "summary.json"
    if manifest_path.is_file() and not force:
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("sheet-evidence output belongs to another identity")
        if prior.get("state") == "complete" and summary_path.is_file():
            return json.loads(summary_path.read_text())
    output.mkdir(parents=True, exist_ok=True)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_EVIDENCE_SCHEMA,
            "version": SHEET_EVIDENCE_VERSION,
            "state": "compiling",
            "identity": identity,
        },
    )

    occupied_cells: dict[Int3, int] = {}
    modes_by_source: dict[tuple[str, str, int], _ModeRecord] = {}
    mode_id_owner: dict[int, tuple[str, str, int]] = {}
    cell_configurations: dict[Int3, list[dict[str, Any]]] = {}
    config_membership_keys: dict[int, tuple[tuple[str, str, int], ...]] = {}
    for input_index, bank in enumerate(banks):
        table = bank.table
        arrays = bank.arrays
        shard_names = tuple(str(value) for value in arrays["shardNames"])
        source_shard = np.asarray(arrays["sourceShardIndex"], dtype=np.int64)
        source_mode_offset = np.asarray(arrays["sourceModeOffset"], dtype=np.uint64)
        source_mode_index = np.asarray(arrays["sourceModeIndex"], dtype=np.int64)
        config_cell_index = np.repeat(
            np.arange(table.cell_count, dtype=np.int64),
            np.diff(table.configuration_offset).astype(np.int64),
        )
        for local_cell_index, local_cell_values in enumerate(table.cell_xyz):
            global_cell = tuple(
                int(local_cell_values[axis]) + bank.spec.offset_cells_xyz[axis]
                for axis in range(3)
            )
            if global_cell in occupied_cells:
                raise ValueError(
                    "sheet-evidence inputs overlap cell ownership at "
                    f"{global_cell}"
                )
            occupied_cells[global_cell] = input_index
            cell_configurations[global_cell] = []
        for configuration_index in range(table.configuration_count):
            local_cell_index = int(config_cell_index[configuration_index])
            local_cell = table.cell_xyz[local_cell_index]
            global_cell = tuple(
                int(local_cell[axis]) + bank.spec.offset_cells_xyz[axis]
                for axis in range(3)
            )
            shard_index = int(source_shard[configuration_index])
            if not 0 <= shard_index < len(shard_names):
                raise ValueError("configuration references an absent source shard")
            layer_start = int(table.layer_offset[configuration_index])
            layer_stop = int(table.layer_offset[configuration_index + 1])
            source_start = int(source_mode_offset[configuration_index])
            source_stop = int(source_mode_offset[configuration_index + 1])
            if layer_stop - layer_start != source_stop - source_start:
                raise ValueError("configuration layer/source-mode counts disagree")
            membership: list[tuple[str, str, int]] = []
            for layer_index, source_index in zip(
                range(layer_start, layer_stop),
                source_mode_index[source_start:source_stop],
            ):
                source_key = (
                    bank.mode_bank_identity_sha256,
                    shard_names[shard_index],
                    int(source_index),
                )
                mode_id = _stable_uint64("mode", *source_key)
                if mode_id in mode_id_owner and mode_id_owner[mode_id] != source_key:
                    raise RuntimeError("stable sheet-mode ID collision")
                mode_id_owner[mode_id] = source_key
                record = _ModeRecord(
                    mode_id,
                    global_cell,
                    input_index,
                    shard_index,
                    int(source_index),
                    int(table.normal_hypothesis[configuration_index]),
                    tuple(float(value) for value in table.layer_normal_xyz[layer_index]),
                    float(table.layer_height[layer_index]),
                    tuple(float(value) for value in table.layer_covariance[layer_index]),
                    tuple(float(value) for value in table.layer_fiber_xyz[layer_index]),
                    float(table.layer_fiber_angular_std_radians[layer_index]),
                    float(table.layer_confidence[layer_index]),
                    float(table.layer_evidence_score[layer_index]),
                    float(table.layer_material_probability[layer_index]),
                    float(table.layer_effective_support[layer_index]),
                )
                prior = modes_by_source.get(source_key)
                if prior is not None and not _mode_matches(prior, record):
                    raise ValueError(
                        "one immutable source mode has inconsistent configuration geometry"
                    )
                modes_by_source.setdefault(source_key, record)
                membership.append(source_key)
            configuration_id = _stable_uint64(
                "configuration",
                bank.candidate_manifest["identitySha256"],
                configuration_index,
            )
            record = {
                "configurationId": configuration_id,
                "inputIndex": input_index,
                "sourceIndex": configuration_index,
                "localId": int(table.configuration_id[configuration_index]),
                "logWeight": float(table.configuration_log_weight[configuration_index]),
                "normalHypothesis": int(table.normal_hypothesis[configuration_index]),
                "evidenceLogScore": float(
                    bank.metadata["evidenceLogScore"][configuration_index]
                ),
                "physicalLogScore": float(
                    bank.metadata["physicalLogScore"][configuration_index]
                ),
                "totalLogScore": float(
                    bank.metadata["totalLogScore"][configuration_index]
                ),
                "coveredEvidenceMass": float(
                    bank.metadata["coveredEvidenceMass"][configuration_index]
                ),
                "totalEvidenceMass": float(
                    bank.metadata["totalEvidenceMass"][configuration_index]
                ),
                "isCurrent": bool(bank.metadata["isCurrent"][configuration_index]),
                "geometryValid": True,
            }
            cell_configurations[global_cell].append(record)
            config_membership_keys[configuration_id] = tuple(membership)

    cells = tuple(sorted(cell_configurations, key=lambda value: (value[2], value[1], value[0])))
    modes = tuple(
        sorted(
            modes_by_source.values(),
            key=lambda value: (
                value.cell_xyz[2],
                value.cell_xyz[1],
                value.cell_xyz[0],
                value.height,
                value.mode_id,
            ),
        )
    )
    mode_by_source = {
        key: modes_by_source[key].mode_id for key in modes_by_source
    }
    patches: list[ClippedPatch] = []
    mode_geometry_status = np.zeros(len(modes), dtype=np.uint8)
    tolerance = max(grid.cell_size_xyz) * resolved.clipping_tolerance_scale
    for mode_index, mode in enumerate(modes):
        try:
            patch = clip_plane_to_cell(
                grid,
                mode.cell_xyz,
                mode.estimate(),
                patch_id=mode.mode_id,
                tolerance=tolerance,
            )
        except DegeneratePlaneIntersection:
            mode_geometry_status[mode_index] = 2
            continue
        if patch is None:
            mode_geometry_status[mode_index] = 1
            continue
        patches.append(patch)
    invalid_mode_ids = {
        mode.mode_id
        for mode, status in zip(modes, mode_geometry_status)
        if int(status) != 0
    }

    configuration_offset = np.zeros(len(cells) + 1, dtype=np.uint64)
    configuration_records: list[dict[str, Any]] = []
    configuration_mode_offset = [0]
    configuration_mode_ids: list[int] = []
    for cell_index, cell in enumerate(cells):
        values = sorted(
            cell_configurations[cell],
            key=lambda value: (
                value["sourceIndex"],
                value["configurationId"],
            ),
        )
        for value in values:
            membership = tuple(
                mode_by_source[key]
                for key in config_membership_keys[value["configurationId"]]
            )
            value["geometryValid"] = not bool(invalid_mode_ids & set(membership))
            configuration_records.append(value)
            configuration_mode_ids.extend(membership)
            configuration_mode_offset.append(len(configuration_mode_ids))
        configuration_offset[cell_index + 1] = len(configuration_records)
    configuration_mode_offset_array = np.asarray(
        configuration_mode_offset, dtype=np.uint64
    )
    configuration_mode_id_array = np.asarray(configuration_mode_ids, dtype=np.uint64)

    patch_table = PatchTable.from_patches(
        grid,
        tuple(patches),
        normal_family={value.mode_id: value.normal_hypothesis for value in modes},
    )
    patch_manifest = write_patch_shard(
        output / "mode-patches-v1",
        patch_table,
        settings={
            "semantics": "one clipped patch per unique immutable Acus layer mode"
        },
        provenance={
            "sheetEvidenceIdentitySha256": identity_sha256,
            "candidateRoots": [str(value.spec.candidate_root) for value in banks],
        },
        compressed=True,
    )
    data_record = _write_evidence_data(
        output,
        cells=cells,
        modes=modes,
        mode_geometry_status=mode_geometry_status,
        configuration_offset=configuration_offset,
        configuration_records=tuple(configuration_records),
        configuration_mode_offset=configuration_mode_offset_array,
        configuration_mode_id=configuration_mode_id_array,
    )
    elapsed = time.monotonic() - started
    statistics = {
        "ownedCells": len(cells),
        "uniqueAcusModes": len(modes),
        "validModePatches": len(patches),
        "modesMissingCell": int(np.sum(mode_geometry_status == 1)),
        "degenerateModePatches": int(np.sum(mode_geometry_status == 2)),
        "physicalConfigurations": len(configuration_records),
        "configurationModeMemberships": len(configuration_mode_ids),
        "geometryValidConfigurations": sum(
            bool(value["geometryValid"]) for value in configuration_records
        ),
        "currentConfigurations": sum(
            bool(value["isCurrent"]) for value in configuration_records
        ),
        "meanUniqueModesPerCell": round(len(modes) / max(len(cells), 1), 6),
        "meanConfigurationsPerCell": round(
            len(configuration_records) / max(len(cells), 1), 6
        ),
    }
    summary = {
        "schema": "pareidolia.cubical-block-sheet-evidence-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "statistics": statistics,
        "artifacts": {
            "evidence": data_record,
            "modePatches": {
                "manifest": "mode-patches-v1.json",
                "manifestSha256": sha256_file(output / "mode-patches-v1.json"),
                "data": patch_manifest["data"],
            },
        },
        "contract": {
            "AcusEvidenceMutableDuringSheetSolve": False,
            "modeNodes": "unique source-referenced fitted Acus planes",
            "configurationHyperedges": (
                "physically valid within-cell stacks referencing mode IDs"
            ),
            "nextStage": (
                "enumerate shared-face mode correspondences and jointly select "
                "configuration hyperedges plus topology-safe sheet edges"
            ),
        },
        "elapsedSeconds": round(elapsed, 6),
    }
    atomic_json(summary_path, summary)
    atomic_json(
        manifest_path,
        {
            "schema": SHEET_EVIDENCE_SCHEMA,
            "version": SHEET_EVIDENCE_VERSION,
            "state": "complete",
            "identity": identity,
            "statistics": statistics,
            "summary": summary_path.name,
            "data": data_record,
            "modePatches": {
                "manifest": "mode-patches-v1.json",
                "data": "mode-patches-v1.npz",
            },
            "elapsedSeconds": round(elapsed, 6),
        },
    )
    return summary
