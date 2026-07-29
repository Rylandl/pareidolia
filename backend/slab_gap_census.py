from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .slab_carrier_gaps import (
    CARRIER_GAP_VERSION,
    _gap_ct_evidence,
    _internal_gap_components,
)
from .slab_sheetlet_carriers import _load_carrier_catalog, _mls_carrier, _sample_stack


GAP_CENSUS_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _cropped_gap_context(
    carrier: dict[str, Any], gap: dict[str, Any], block_size: int = 8
) -> tuple[dict[str, Any], dict[str, Any]]:
    y0, y1, x0, x1 = (int(value) for value in gap["bboxYX"])
    block_size = max(1, int(block_size))
    y0 = max(0, (y0 // block_size) * block_size)
    x0 = max(0, (x0 // block_size) * block_size)
    y1 = min(
        carrier["supportMask"].shape[0],
        ((y1 + block_size - 1) // block_size) * block_size,
    )
    x1 = min(
        carrier["supportMask"].shape[1],
        ((x1 + block_size - 1) // block_size) * block_size,
    )
    region = (slice(y0, y1), slice(x0, x1))
    cropped = {
        "surfaceXYZ": carrier["surfaceXYZ"][region],
        "normalXYZ": carrier["normalXYZ"][region],
        "fiberXYZ": carrier["fiberXYZ"][region],
        "supportMask": carrier["supportMask"][region],
        "frame": carrier["frame"],
    }
    cropped_gap = {
        **{key: value for key, value in gap.items() if key != "mask"},
        "mask": gap["mask"][region],
    }
    return cropped, cropped_gap


def _ct_gate(
    evidence: dict[str, Any],
    minimum_score: float,
    minimum_material_fraction: float,
    maximum_depth_offset: float,
    maximum_fiber_residual: float,
) -> dict[str, Any]:
    score = float(evidence.get("depthAlignedTextureScore") or 0.0)
    material = float(evidence.get("materialFraction") or 0.0)
    depth_value = evidence.get("bestDepthOffsetVoxels")
    fiber_value = evidence.get("fiberAngleResidualDeg")
    depth = abs(float(depth_value)) if depth_value is not None else float("inf")
    fiber = float(fiber_value) if fiber_value is not None else float("inf")
    slack = {
        "depthAlignedTextureScore": round(score - minimum_score, 4),
        "materialFraction": round(material - minimum_material_fraction, 4),
        "absoluteDepthOffsetVoxels": (
            round(maximum_depth_offset - depth, 3) if np.isfinite(depth) else None
        ),
        "fiberAngleResidualDeg": (
            round(maximum_fiber_residual - fiber, 3) if np.isfinite(fiber) else None
        ),
    }
    normalized_slack = {
        "depthAlignedTextureScore": round(
            score / max(minimum_score, 1.0e-8) - 1.0, 4
        ),
        "materialFraction": round(
            material / max(minimum_material_fraction, 1.0e-8) - 1.0, 4
        ),
        "absoluteDepthOffsetVoxels": (
            round(1.0 - depth / max(maximum_depth_offset, 1.0e-8), 4)
            if np.isfinite(depth)
            else None
        ),
        "fiberAngleResidualDeg": (
            round(1.0 - fiber / max(maximum_fiber_residual, 1.0e-8), 4)
            if np.isfinite(fiber)
            else None
        ),
    }
    failures = []
    if score < minimum_score:
        failures.append("texture-score")
    if material < minimum_material_fraction:
        failures.append("material")
    if depth > maximum_depth_offset:
        failures.append("depth")
    if fiber > maximum_fiber_residual:
        failures.append("fiber")
    finite_normalized = [
        float(value) for value in normalized_slack.values() if value is not None
    ]
    return {
        "queuedForDenseAcus": not failures,
        "rejectionReasons": failures,
        "thresholdSlack": slack,
        "normalizedThresholdSlack": normalized_slack,
        "minimumNormalizedThresholdSlack": round(min(finite_normalized), 4)
        if finite_normalized
        else None,
    }


def census_carrier_gaps(
    output_root: str | Path,
    minimum_gap_area_square_voxels: float = 256.0,
    minimum_ct_score: float = 0.5,
    minimum_material_fraction: float = 0.35,
    maximum_depth_offset: float = 4.0,
    maximum_fiber_residual: float = 12.0,
    force: bool = False,
) -> dict[str, Any]:
    """Audit CT evidence only inside every enclosed final-carrier gap."""
    root = Path(output_root)
    summary_path = root / f"sheetlet-gap-census-v{GAP_CENSUS_VERSION}.json"
    gap_path = root / f"sheetlet-carrier-gaps-v{CARRIER_GAP_VERSION}.json"
    gap_summary = json.loads(gap_path.read_text())
    artifact_path = root / gap_summary["artifact"]
    input_identity = {
        "summary": _content_identity(gap_path),
        "artifact": _content_identity(artifact_path),
    }
    settings = {
        "minimumGapAreaSquareVoxels": float(minimum_gap_area_square_voxels),
        "depthOffsetsVoxels": [float(value) for value in np.arange(-6.0, 6.01, 1.0)],
        "minimumDepthAlignedTextureScore": float(minimum_ct_score),
        "minimumMaterialFraction": float(minimum_material_fraction),
        "maximumAbsoluteDepthOffsetVoxels": float(maximum_depth_offset),
        "maximumFiberResidualDeg": float(maximum_fiber_residual),
        "scope": "fully enclosed gaps in every final carrier",
    }
    if summary_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if (
            cached.get("settings") == settings
            and cached.get("identity", {}).get("inputArtifacts") == input_identity
        ):
            return cached

    started = time.monotonic()
    analysis = json.loads((root / "analysis.json").read_text())
    air_threshold = float(analysis["normalization"]["airThreshold"])
    source_path, _, _, flakes = _load_carrier_catalog(root)
    source = np.load(source_path, mmap_mode="r")
    with np.load(artifact_path) as payload:
        member_index = np.asarray(payload["memberIndex"], dtype=np.uint32)
        member_offset = np.asarray(payload["memberOffset"], dtype=np.uint64)
    depth_offsets = np.arange(-6.0, 6.01, 1.0, dtype=np.float32)
    states = []
    gap_count = 0
    queued_gap_count = 0
    for state_index, source_state in enumerate(gap_summary["states"]):
        rank = state_index + 1
        low = int(member_offset[state_index])
        high = int(member_offset[state_index + 1])
        members = member_index[low:high]
        member_flakes = [flakes[int(index)] for index in members]
        carrier = _mls_carrier(member_flakes)
        gaps = _internal_gap_components(
            carrier["supportMask"],
            float(carrier["stats"]["pixelStepVoxels"]),
            minimum_gap_area_square_voxels,
        )
        gap_outputs = []
        if gaps:
            expanded = _mls_carrier(member_flakes, support_radius=112.0)
            for gap in gaps:
                cropped_carrier, cropped_gap = _cropped_gap_context(expanded, gap)
                stack, sampling = _sample_stack(
                    source, cropped_carrier, depth_offsets
                )
                evidence = _gap_ct_evidence(
                    cropped_carrier,
                    stack,
                    depth_offsets,
                    cropped_gap,
                    air_threshold,
                )
                gate = _ct_gate(
                    evidence,
                    minimum_ct_score,
                    minimum_material_fraction,
                    maximum_depth_offset,
                    maximum_fiber_residual,
                )
                gap_outputs.append(
                    {
                        **{key: value for key, value in gap.items() if key != "mask"},
                        "ctEvidence": evidence,
                        "gate": gate,
                        "sampling": sampling,
                    }
                )
                gap_count += 1
                queued_gap_count += int(gate["queuedForDenseAcus"])
        states.append(
            {
                "rank": rank,
                "sourceState": source_state,
                "flakeCount": len(members),
                "carrier": carrier["stats"],
                "gapCount": len(gaps),
                "gapAreaSquareVoxels": round(
                    sum(float(value["areaSquareVoxels"]) for value in gaps), 2
                ),
                "queuedGapCount": sum(
                    int(value["gate"]["queuedForDenseAcus"])
                    for value in gap_outputs
                ),
                "gaps": gap_outputs,
            }
        )
        if rank % 10 == 0 or rank == len(gap_summary["states"]):
            elapsed = max(time.monotonic() - started, 1.0e-6)
            print(
                f"gap census {rank}/{len(gap_summary['states'])} · "
                f"gaps {gap_count} · queued {queued_gap_count} · "
                f"{rank / elapsed:.2f} carriers/s",
                flush=True,
            )

    queued_ranks = [
        int(value["rank"]) for value in states if int(value["queuedGapCount"]) > 0
    ]
    result = {
        "identity": {
            "version": GAP_CENSUS_VERSION,
            "analysis": analysis["identity"],
            "inputArtifacts": input_identity,
        },
        "settings": settings,
        "states": states,
        "queue": {
            "ranks": queued_ranks,
            "gapCount": queued_gap_count,
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "carrierCount": len(states),
            "carrierWithGapCount": sum(int(value["gapCount"] > 0) for value in states),
            "gapCount": gap_count,
            "queuedRankCount": len(queued_ranks),
            "queuedGapCount": queued_gap_count,
            "gapAreaSquareVoxels": round(
                sum(float(value["gapAreaSquareVoxels"]) for value in states), 2
            ),
        },
    }
    _atomic_json(summary_path, result)
    return result
