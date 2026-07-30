from __future__ import annotations

import json
import math
import time
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .block import (
    BlockBounds,
    SurfaceBlock,
    assemble_surface_hierarchy,
    extend_surface_block_joins,
)
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .export import write_block_obj, write_block_projection_png
from .matching import TraceMatchSettings
from .tables import read_patch_shard


DUAL_AXIS_PACKET_SCHEMA = "pareidolia.cubical-dual-axis-sheet-packets"
DUAL_AXIS_PACKET_VERSION = 1


@dataclass(frozen=True, slots=True)
class DualAxisPacketSettings:
    """Connectivity settings for the sheet-level, not single-ply, graph."""

    leaf_shape_cells_xyz: tuple[int, int, int] = (4, 4, 3)
    maximum_preview_components: int = 128
    maximum_normal_angle_degrees: float = 15.0
    maximum_fiber_frame_residual_degrees: float = 15.0

    def __post_init__(self) -> None:
        leaf = tuple(int(value) for value in self.leaf_shape_cells_xyz)
        if len(leaf) != 3 or any(value <= 0 for value in leaf):
            raise ValueError("packet leaf shape must be a positive XYZ triple")
        if self.maximum_preview_components <= 0:
            raise ValueError("packet preview component count must be positive")
        angles = (
            self.maximum_normal_angle_degrees,
            self.maximum_fiber_frame_residual_degrees,
        )
        if any(
            not math.isfinite(value) or not 0.0 < value <= 90.0
            for value in angles
        ):
            raise ValueError("packet absolute angular gates must lie in (0, 90]")
        object.__setattr__(self, "leaf_shape_cells_xyz", leaf)

    def record(self) -> dict[str, Any]:
        values = asdict(self)
        values["leaf_shape_cells_xyz"] = list(self.leaf_shape_cells_xyz)
        return values


def _block_statistics(block: SurfaceBlock) -> dict[str, Any]:
    sizes = sorted((len(value.patch_ids) for value in block.components), reverse=True)
    return {
        "patches": len(block.patches),
        "candidateJoins": len(block.candidate_joins),
        "retainedJoins": len(block.joins),
        "deferredJoins": len(block.deferred_joins),
        "deferredByReason": dict(
            sorted(Counter(value.reason for value in block.deferred_joins).items())
        ),
        "components": len(block.components),
        "largestComponentPatchCount": max(sizes, default=0),
        "topComponentPatchCounts": sizes[:20],
        "exteriorTraces": len(block.exterior_traces),
        "unresolvedInteriorTraces": len(block.unresolved_interior_traces),
    }


def _quarter_turn_statistics(block: SurfaceBlock) -> dict[str, Any]:
    candidate_count = sum(
        value.fiber_quarter_turn is True for value in block.candidate_joins
    )
    retained = tuple(
        value for value in block.joins if value.fiber_quarter_turn is True
    )

    def quantiles(values: list[float]) -> dict[str, float | None]:
        if not values:
            return {"median": None, "p90": None, "maximum": None}
        result = np.percentile(np.asarray(values, dtype=np.float64), (50, 90, 100))
        return {
            name: round(float(value), 7)
            for name, value in zip(("median", "p90", "maximum"), result)
        }

    return {
        "candidateJoins": candidate_count,
        "retainedJoins": len(retained),
        "retainedFraction": round(
            len(retained) / max(candidate_count, 1), 7
        ),
        "normalResidualDegrees": quantiles(
            [float(np.degrees(value.normal_angle_radians)) for value in retained]
        ),
        "endpointResidualZ": quantiles(
            [
                max((endpoint.z for endpoint in value.endpoint_agreements), default=0.0)
                for value in retained
            ]
        ),
        "fiberFrameResidualDegrees": quantiles(
            [float(np.degrees(value.fiber_angle_radians or 0.0)) for value in retained]
        ),
        "reducedChiSquare": quantiles(
            [value.reduced_chi_square for value in retained]
        ),
    }


def _write_graph(path: Path, block: SurfaceBlock) -> None:
    joins = tuple(block.joins)
    component_by_patch = dict(block.component_by_patch)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            firstPatchId=np.asarray(
                [value.first_patch_id for value in joins], dtype=np.uint64
            ),
            secondPatchId=np.asarray(
                [value.second_patch_id for value in joins], dtype=np.uint64
            ),
            faceAxis=np.asarray([value.face.axis for value in joins], dtype=np.int8),
            faceAnchorXYZ=np.asarray(
                [value.face.anchor_xyz for value in joins], dtype=np.int32
            ).reshape(len(joins), 3),
            fiberQuarterTurn=np.asarray(
                [value.fiber_quarter_turn is True for value in joins], dtype=np.uint8
            ),
            fiberFrameResidualDegrees=np.asarray(
                [
                    np.degrees(value.fiber_angle_radians)
                    if value.fiber_angle_radians is not None
                    else np.nan
                    for value in joins
                ],
                dtype=np.float32,
            ),
            patchId=np.asarray(sorted(component_by_patch), dtype=np.uint64),
            componentId=np.asarray(
                [component_by_patch[value] for value in sorted(component_by_patch)],
                dtype=np.uint64,
            ),
        )
    temporary.replace(path)


def _identity(
    input_root: Path, settings: DualAxisPacketSettings
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": DUAL_AXIS_PACKET_SCHEMA,
        "version": DUAL_AXIS_PACKET_VERSION,
        "inputRoot": str(input_root),
        "inputPatchManifestSha256": sha256_file(
            input_root / "selected-patches-v1.json"
        ),
        "inputPatchDataSha256": sha256_file(input_root / "selected-patches-v1.npz"),
        "settings": settings.record(),
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "sheet_packets.py",
                "matching.py",
                "block.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def run_dual_axis_packet_connectivity(
    input_root: str | Path,
    output_root: str | Path,
    *,
    settings: DualAxisPacketSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Build a separate sheet-packet graph with 90-degree fiber equivalence.

    The input single-ply geometry and graph are immutable. This graph changes
    only the fiber comparison at shared faces; endpoint, normal, ordered-trace,
    collision, crossing-topology, and orientability constraints are unchanged.
    """

    started = time.monotonic()
    resolved = settings or DualAxisPacketSettings()
    source = Path(input_root).resolve()
    output = Path(output_root).resolve()
    if output == source:
        raise ValueError("packet output must differ from its selected-patch input")
    identity = _identity(source, resolved)
    identity_sha256 = str(identity["identitySha256"])
    manifest_path = output / "packets.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity_sha256:
            raise ValueError("packet output belongs to another identity")
        if (
            not force
            and prior.get("state") == "complete"
            and (output / "summary.json").is_file()
        ):
            return json.loads((output / "summary.json").read_text())
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": DUAL_AXIS_PACKET_SCHEMA,
        "version": DUAL_AXIS_PACKET_VERSION,
        "state": "assembling",
        "identity": identity,
        "inputRoot": str(source),
    }
    atomic_json(manifest_path, manifest)
    patches = read_patch_shard(source / "selected-patches-v1", verify=True)
    strict_block = assemble_surface_hierarchy(
        patches.grid,
        BlockBounds((0, 0, 0), patches.grid.shape_cells_xyz),
        patches.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
    )
    proposal_block = assemble_surface_hierarchy(
        patches.grid,
        BlockBounds((0, 0, 0), patches.grid.shape_cells_xyz),
        patches.to_patches(),
        maximum_leaf_shape_cells_xyz=resolved.leaf_shape_cells_xyz,
        settings=TraceMatchSettings(
            orthogonal_fiber_equivalence=True,
        ),
    )
    packet_candidates = tuple(
        value
        for value in proposal_block.candidate_joins
        if value.fiber_quarter_turn is True
        and math.degrees(value.normal_angle_radians)
        <= resolved.maximum_normal_angle_degrees
        and value.fiber_angle_radians is not None
        and math.degrees(value.fiber_angle_radians)
        <= resolved.maximum_fiber_frame_residual_degrees
    )
    block = extend_surface_block_joins(strict_block, packet_candidates)
    graph_path = output / "packet-graph-v1.npz"
    _write_graph(graph_path, block)
    obj_path = write_block_obj(block, output / "surface.obj")
    projection_path = write_block_projection_png(
        block,
        output / "projections.png",
        maximum_components=resolved.maximum_preview_components,
    )
    largest_path = write_block_projection_png(
        block, output / "largest-component.png", maximum_components=1
    )
    top_twelve_path = write_block_projection_png(
        block, output / "top-12-components.png", maximum_components=12
    )
    packet_statistics = _block_statistics(block)
    strict_statistics = _block_statistics(strict_block)
    delta = {
        key: int(packet_statistics[key]) - int(strict_statistics[key])
        for key in (
            "patches",
            "candidateJoins",
            "retainedJoins",
            "deferredJoins",
            "components",
            "largestComponentPatchCount",
            "exteriorTraces",
            "unresolvedInteriorTraces",
        )
    }
    summary: dict[str, Any] = {
        "schema": "pareidolia.cubical-dual-axis-sheet-packet-summary",
        "version": 1,
        "identitySha256": identity_sha256,
        "inputRoot": str(source),
        "settings": resolved.record(),
        "method": {
            "semantics": "sheet packet, not single fiber ply",
            "normalAndFiberPolarity": "axial/unsigned",
            "fiberFrame": (
                "parallel and orthogonal transported in-plane fiber axes are "
                "equivalent only in this separate packet graph"
            ),
            "strictGraphIsImmutable": True,
            "quarterTurnAdmission": (
                "absolute normal and fiber-frame residual gates followed by full "
                "incremental topology selection"
            ),
            "preservedHardConstraints": (
                "shared-face endpoint/normal gates, ordered trace alignment, "
                "same-cell collision, crossing topology, and orientability"
            ),
        },
        "strictReference": {
            "source": "fresh assembly of the exact selected patch artifact",
            "statistics": strict_statistics,
        },
        "quarterTurnAdmission": {
            "discoveredCandidates": sum(
                value.fiber_quarter_turn is True
                for value in proposal_block.candidate_joins
            ),
            "absoluteGateCandidates": len(packet_candidates),
        },
        "packets": packet_statistics,
        "quarterTurnJoinEvidence": _quarter_turn_statistics(block),
        "deltaFromStrictReference": delta,
        "timingSeconds": {"total": round(time.monotonic() - started, 6)},
        "artifacts": {
            "graph": graph_path.name,
            "mesh": obj_path.name,
            "projections": projection_path.name,
            "largestComponent": largest_path.name,
            "topTwelveComponents": top_twelve_path.name,
        },
    }
    atomic_json(output / "summary.json", summary)
    manifest["state"] = "complete"
    manifest["summary"] = "summary.json"
    manifest["elapsedSeconds"] = summary["timingSeconds"]["total"]
    atomic_json(manifest_path, manifest)
    return summary
