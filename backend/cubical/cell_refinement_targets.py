from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .cell_refinement import load_cluster_cell_context
from .contracts import atomic_json, canonical_json_hash, sha256_file
from .topology import GridFace, Int3


CELL_REFINEMENT_TARGET_SCHEMA = "pareidolia.cubical-cell-refinement-targets"
CELL_REFINEMENT_TARGET_VERSION = 1
CELL_REFINEMENT_TARGET_STEM = "cell-refinement-targets-v1"


@dataclass(frozen=True, slots=True)
class CellRefinementTargetSettings:
    neighborhood_radius_cells: int = 1
    maximum_targets: int = 128
    minimum_recoverable_evidence_mass: float = 1.0e-6
    minimum_incident_open_trace_endpoints: int = 1

    def __post_init__(self) -> None:
        if self.neighborhood_radius_cells < 0:
            raise ValueError("target neighborhood radius must be nonnegative")
        if self.maximum_targets <= 0:
            raise ValueError("maximum refinement targets must be positive")
        if (
            not math.isfinite(self.minimum_recoverable_evidence_mass)
            or self.minimum_recoverable_evidence_mass < 0.0
        ):
            raise ValueError(
                "minimum recoverable evidence mass must be finite and nonnegative"
            )
        if self.minimum_incident_open_trace_endpoints < 0:
            raise ValueError("minimum incident open traces must be nonnegative")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _percentile_ranks(values: np.ndarray) -> np.ndarray:
    """Return deterministic midpoint empirical percentiles, preserving ties."""

    if values.ndim != 1:
        raise ValueError("percentile ranks require a one-dimensional array")
    if not len(values):
        return np.empty(0, dtype=np.float64)
    unique, inverse, counts = np.unique(
        values.astype(np.float64, copy=False),
        return_inverse=True,
        return_counts=True,
    )
    del unique
    prior = np.cumsum(counts) - counts
    percentiles = (prior + 0.5 * counts) / len(values)
    return percentiles[inverse]


def _target_cell_for_face(source_cell: Int3, face: GridFace) -> Int3:
    lower, upper = face.adjacent_cells()
    if source_cell == lower:
        return upper
    if source_cell == upper:
        return lower
    raise ValueError("open trace source cell does not touch its face")


def _spatially_separated(
    records: list[dict[str, Any]],
    *,
    radius_cells: int,
    maximum_targets: int,
) -> list[dict[str, Any]]:
    # Two closed neighborhoods of radius r are disjoint when their center
    # Chebyshev distance is greater than 2r.
    minimum_distance = 2 * radius_cells + 1
    selected: list[dict[str, Any]] = []
    centers: list[Int3] = []
    for value in records:
        cell = tuple(int(item) for item in value["cellXYZ"])
        if any(
            max(abs(cell[axis] - prior[axis]) for axis in range(3))
            < minimum_distance
            for prior in centers
        ):
            continue
        selected.append(value)
        centers.append(cell)
        if len(selected) >= maximum_targets:
            break
    return selected


def rank_cell_refinement_targets(
    cluster_root: str | Path,
    materialized_root: str | Path,
    *,
    settings: CellRefinementTargetSettings | None = None,
) -> dict[str, Any]:
    """Rank evidence-recoverable cells that also touch unresolved topology."""

    resolved = settings or CellRefinementTargetSettings()
    context = load_cluster_cell_context(cluster_root, materialized_root)
    source_open: Counter[Int3] = Counter()
    incident_open: Counter[Int3] = Counter()
    faces_by_cell: dict[Int3, set[GridFace]] = defaultdict(set)
    components_by_cell: dict[Int3, set[int]] = defaultdict(set)
    patch_by_id = {value.patch_id: value for value in context.block.patches}
    for boundary in context.block.unresolved_interior_traces:
        source_cell = patch_by_id[boundary.patch_id].cell_xyz
        target_cell = _target_cell_for_face(source_cell, boundary.trace.face)
        source_open[source_cell] += 1
        incident_open[source_cell] += 1
        incident_open[target_cell] += 1
        faces_by_cell[source_cell].add(boundary.trace.face)
        faces_by_cell[target_cell].add(boundary.trace.face)
        components_by_cell[source_cell].add(boundary.component_id)
        components_by_cell[target_cell].add(boundary.component_id)

    records: list[dict[str, Any]] = []
    for cell in sorted(
        context.owner_by_cell,
        key=lambda value: (value[2], value[1], value[0]),
    ):
        indices = tuple(context.configuration_indices(cell))
        current_index = int(context.selected_by_cell[cell])
        current_covered, total = context.evidence(cell, current_index)
        oracle_index = max(
            indices,
            key=lambda value: (
                context.evidence(cell, value)[0],
                context.option(cell, value).log_weight,
                -value,
            ),
        )
        oracle_covered, oracle_total = context.evidence(cell, oracle_index)
        if not math.isclose(total, oracle_total, rel_tol=0.0, abs_tol=1.0e-6):
            raise ValueError("candidate evidence totals disagree within a cell")
        recoverable = max(oracle_covered - current_covered, 0.0)
        records.append(
            {
                "cellXYZ": list(cell),
                "inputIndex": context.owner_by_cell[cell][0],
                "localCellXYZ": list(context.owner_by_cell[cell][1]),
                "currentSourceConfigurationIndex": current_index,
                "oracleSourceConfigurationIndex": int(oracle_index),
                "currentLayerCount": len(
                    context.option(cell, current_index).patches
                ),
                "oracleLayerCount": len(context.option(cell, oracle_index).patches),
                "coveredEvidenceMass": round(current_covered, 6),
                "oracleCoveredEvidenceMass": round(oracle_covered, 6),
                "recoverableEvidenceMass": round(recoverable, 6),
                "totalEvidenceMass": round(total, 6),
                "evidenceUtilization": round(
                    current_covered / max(total, 1.0e-12), 6
                ),
                "oracleEvidenceUtilization": round(
                    oracle_covered / max(total, 1.0e-12), 6
                ),
                "recoverableUtilization": round(
                    recoverable / max(total, 1.0e-12), 6
                ),
                "sourceOpenTraceEndpointCount": int(source_open[cell]),
                "incidentOpenTraceEndpointCount": int(incident_open[cell]),
                "incidentOpenFaceCount": len(faces_by_cell[cell]),
                "incidentComponentCount": len(components_by_cell[cell]),
            }
        )

    evidence_values = np.asarray(
        [value["recoverableEvidenceMass"] for value in records],
        dtype=np.float64,
    )
    topology_values = np.asarray(
        [value["incidentOpenTraceEndpointCount"] for value in records],
        dtype=np.float64,
    )
    evidence_percentiles = _percentile_ranks(evidence_values)
    topology_percentiles = _percentile_ranks(topology_values)
    for index, value in enumerate(records):
        evidence_percentile = float(evidence_percentiles[index])
        topology_percentile = float(topology_percentiles[index])
        value["recoverableEvidencePercentile"] = round(evidence_percentile, 6)
        value["openTopologyPercentile"] = round(topology_percentile, 6)
        value["jointPriority"] = round(
            math.sqrt(evidence_percentile * topology_percentile), 6
        )

    eligible = [
        value
        for value in records
        if float(value["recoverableEvidenceMass"])
        >= resolved.minimum_recoverable_evidence_mass
        and int(value["incidentOpenTraceEndpointCount"])
        >= resolved.minimum_incident_open_trace_endpoints
    ]
    eligible.sort(
        key=lambda value: (
            -float(value["jointPriority"]),
            -float(value["recoverableEvidenceMass"]),
            -int(value["incidentOpenTraceEndpointCount"]),
            value["cellXYZ"][2],
            value["cellXYZ"][1],
            value["cellXYZ"][0],
        )
    )
    separated = _spatially_separated(
        eligible,
        radius_cells=resolved.neighborhood_radius_cells,
        maximum_targets=resolved.maximum_targets,
    )
    evidence_ranking = sorted(
        eligible,
        key=lambda value: (
            -float(value["recoverableEvidenceMass"]),
            -int(value["incidentOpenTraceEndpointCount"]),
            value["cellXYZ"][2],
            value["cellXYZ"][1],
            value["cellXYZ"][0],
        ),
    )[: resolved.maximum_targets]
    return {
        "settings": resolved.record(),
        "statistics": {
            "cells": len(records),
            "eligibleCells": len(eligible),
            "spatiallySeparatedTargets": len(separated),
            "totalRecoverableEvidenceMass": round(float(np.sum(evidence_values)), 6),
            "cellsWithRecoverableEvidence": int(np.count_nonzero(evidence_values > 0.0)),
            "cellsTouchingOpenTopology": int(np.count_nonzero(topology_values > 0.0)),
            "incidentOpenTraceEndpointCounts": {
                str(value): int(np.count_nonzero(topology_values >= value))
                for value in (1, 8, 16, 24, 32)
            },
        },
        "prioritySemantics": {
            "recoverableEvidencePercentile": (
                "midpoint empirical percentile of Acus mass recoverable by the "
                "retained candidate oracle"
            ),
            "openTopologyPercentile": (
                "midpoint empirical percentile of unresolved trace endpoints "
                "incident to the cell"
            ),
            "jointPriority": (
                "geometric mean of the independent evidence and topology percentiles"
            ),
            "spatialSeparation": (
                "selected target neighborhoods do not overlap at the requested radius"
            ),
        },
        "spatiallySeparatedRanking": separated,
        "jointRanking": eligible[: resolved.maximum_targets],
        "evidenceRanking": evidence_ranking,
    }


def run_cell_refinement_target_ranking(
    cluster_root: str | Path,
    materialized_root: str | Path,
    output_root: str | Path,
    *,
    settings: CellRefinementTargetSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    """Write a hashed target census for one reloadable refinement baseline."""

    resolved = settings or CellRefinementTargetSettings()
    cluster = Path(cluster_root).resolve()
    materialized = Path(materialized_root).resolve()
    output = Path(output_root).resolve()
    identity: dict[str, Any] = {
        "schema": CELL_REFINEMENT_TARGET_SCHEMA,
        "version": CELL_REFINEMENT_TARGET_VERSION,
        "clusterRoot": str(cluster),
        "clusterManifestSha256": sha256_file(
            cluster / "cluster-reselection-v1.json"
        ),
        "clusterDataSha256": sha256_file(cluster / "cluster-reselection-v1.npz"),
        "materializedRoot": str(materialized),
        "surfaceGraphManifestSha256": sha256_file(
            materialized / "surface-graph-v1.json"
        ),
        "surfaceGraphDataSha256": sha256_file(
            materialized / "surface-graph-v1.npz"
        ),
        "settings": resolved.record(),
        "implementationSha256": {
            "cell_refinement_targets.py": sha256_file(Path(__file__)),
            "cell_refinement.py": sha256_file(
                Path(__file__).resolve().parent / "cell_refinement.py"
            ),
        },
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output_path = output / f"{CELL_REFINEMENT_TARGET_STEM}.json"
    if output_path.is_file() and not force:
        prior = json.loads(output_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("cell-refinement target output belongs to another identity")
        return prior
    output.mkdir(parents=True, exist_ok=True)
    ranking = rank_cell_refinement_targets(
        cluster,
        materialized,
        settings=resolved,
    )
    payload = {
        "schema": CELL_REFINEMENT_TARGET_SCHEMA,
        "version": CELL_REFINEMENT_TARGET_VERSION,
        "identity": identity,
        **ranking,
    }
    atomic_json(output_path, payload)
    return payload
