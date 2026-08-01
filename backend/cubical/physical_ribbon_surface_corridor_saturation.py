from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_dense_completion import (
    PHYSICAL_RIBBON_DENSE_COMPLETION_STEM,
    PhysicalRibbonDenseCompletionSettings,
    _texture_compatible_hole_rows,
    run_physical_ribbon_dense_completion,
)
from .physical_ribbon_depth_fields import (
    PHYSICAL_RIBBON_DEPTH_FIELD_STEM,
    PhysicalRibbonDepthFieldSettings,
    run_physical_ribbon_depth_fields,
)
from .physical_ribbon_flattened_audit import (
    PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM,
    PhysicalRibbonFlattenedAuditSettings,
    run_physical_ribbon_flattened_audit,
)
from .physical_ribbon_surface_corridors import (
    PHYSICAL_RIBBON_SURFACE_CORRIDORS_STEM,
    PhysicalRibbonSurfaceCorridorSettings,
    run_physical_ribbon_surface_corridors,
)
from .physical_ribbon_surface_holes import _resolve_surface_manifest
from .surface_topology import triangle_edge_region_labels


PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_SCHEMA = (
    "pareidolia.physical-ribbon-surface-corridor-saturation"
)
PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_VERSION = 1
PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_STEM = (
    "physical-ribbon-surface-corridor-saturation-v1"
)

ProgressCallback = Callable[[int, str, Mapping[str, Any]], None]


def _settings_section(
    record: Mapping[str, Any], name: str
) -> dict[str, Any]:
    value = record.get(name, {})
    if not isinstance(value, Mapping):
        raise ValueError(f"{name} settings must contain one object")
    return dict(value)


def _tuple_fields(
    values: dict[str, Any], names: Sequence[str]
) -> dict[str, Any]:
    for name in names:
        if name in values:
            values[name] = tuple(values[name])
    return values


@dataclass(frozen=True, slots=True)
class PhysicalRibbonSurfaceCorridorSaturationSettings:
    """Reconstruct complete paired-frontier strips to a fixed point."""

    maximum_rounds: int = 16
    surface_corridors: PhysicalRibbonSurfaceCorridorSettings = field(
        default_factory=PhysicalRibbonSurfaceCorridorSettings
    )
    depth_fields: PhysicalRibbonDepthFieldSettings = field(
        default_factory=PhysicalRibbonDepthFieldSettings
    )
    dense_completion: PhysicalRibbonDenseCompletionSettings = field(
        default_factory=lambda: PhysicalRibbonDenseCompletionSettings(
            maximum_completed_holes=256
        )
    )
    flattened_audit: PhysicalRibbonFlattenedAuditSettings = field(
        default_factory=lambda: PhysicalRibbonFlattenedAuditSettings(
            maximum_components=256
        )
    )

    def __post_init__(self) -> None:
        if self.maximum_rounds < 1:
            raise ValueError(
                "surface-corridor saturation requires at least one round"
            )
        scored_cap = (
            self.surface_corridors.corridors.maximum_scored_corridors
        )
        if self.dense_completion.maximum_completed_holes < scored_cap:
            raise ValueError(
                "dense completion must evaluate every scored surface "
                "corridor before a fixed point can be declared"
            )
        if (
            self.flattened_audit.maximum_components
            < self.dense_completion.maximum_completed_holes
        ):
            raise ValueError(
                "flattened audit must be able to measure every independently "
                "accepted surface corridor"
            )

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any]
    ) -> "PhysicalRibbonSurfaceCorridorSaturationSettings":
        allowed = {
            "maximum_rounds",
            "surface_corridors",
            "depth_fields",
            "dense_completion",
            "flattened_audit",
        }
        unexpected = set(record) - allowed
        if unexpected:
            raise ValueError(
                "unknown surface-corridor saturation settings: "
                + ", ".join(sorted(unexpected))
            )
        corridor_values = _settings_section(record, "surface_corridors")
        depth_values = _tuple_fields(
            _settings_section(record, "depth_fields"),
            ("profile_depth_fractions",),
        )
        dense_values = _tuple_fields(
            _settings_section(record, "dense_completion"),
            (
                "profile_depth_fractions",
                "competing_shift_thicknesses",
                "interior_boundary_separation_hypotheses_voxels",
                "attachment_collar_outward_tangent_ratio_hypotheses",
            ),
        )
        audit_values = _tuple_fields(
            _settings_section(record, "flattened_audit"),
            (
                "depth_fractions",
                "native_seam_inward_range_voxels",
                "native_seam_edge_parameters",
                "native_seam_scale_hypotheses",
            ),
        )
        dense_values.setdefault("maximum_completed_holes", 256)
        audit_values.setdefault("maximum_components", 256)
        return cls(
            maximum_rounds=int(record.get("maximum_rounds", 16)),
            surface_corridors=(
                PhysicalRibbonSurfaceCorridorSettings.from_record(
                    corridor_values
                )
            ),
            depth_fields=PhysicalRibbonDepthFieldSettings(**depth_values),
            dense_completion=PhysicalRibbonDenseCompletionSettings(
                **dense_values
            ),
            flattened_audit=PhysicalRibbonFlattenedAuditSettings(
                **audit_values
            ),
        )

    def record(self) -> dict[str, Any]:
        return {
            "maximum_rounds": self.maximum_rounds,
            "surface_corridors": self.surface_corridors.record(),
            "depth_fields": self.depth_fields.record(),
            "dense_completion": self.dense_completion.record(),
            "flattened_audit": self.flattened_audit.record(),
        }


def _manifest_reference(
    path: Path, manifest: Mapping[str, Any]
) -> dict[str, Any]:
    reference: dict[str, Any] = {
        "manifestPath": str(path),
        "manifestSha256": sha256_file(path),
    }
    data_sha = manifest.get("data", {}).get("sha256")
    if data_sha is not None:
        reference["dataSha256"] = data_sha
    return reference


def _surface_reference(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, Any]]:
    path, manifest = _resolve_surface_manifest(root)
    return path, manifest, _manifest_reference(path, manifest)


def _completion_accepted_rows(
    completion: Mapping[str, Any],
) -> frozenset[int]:
    rows = [
        int(record["holeRow"])
        for record in completion.get("completions", ())
        if bool(record.get("accepted"))
    ]
    if len(rows) != len(set(rows)):
        raise ValueError(
            "dense completion accepted one corridor row more than once"
        )
    declared = int(
        completion.get("analysis", {}).get("acceptedHoleCount", -1)
    )
    if declared != len(rows):
        raise ValueError(
            "dense-completion accepted count differs from its records"
        )
    return frozenset(rows)


def _surface_corridor_enumeration_exhausted(
    corridor_manifest: Mapping[str, Any],
) -> bool:
    total = int(
        corridor_manifest.get("corridors", {}).get(
            "multiAnchorCorridorCount", 0
        )
    )
    scored = int(
        corridor_manifest.get("evidence", {}).get(
            "scoredCorridorCount", 0
        )
    )
    return scored >= total


def _notify(
    progress: ProgressCallback | None,
    round_index: int,
    stage: str,
    values: Mapping[str, Any],
) -> None:
    if progress is not None:
        progress(round_index, stage, values)


def run_physical_ribbon_surface_corridor_saturation(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonSurfaceCorridorSaturationSettings | None = None,
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonSurfaceCorridorSaturationSettings()
    _, initial_manifest, initial_reference = _surface_reference(surface_root)
    implementation = {
        "orchestration": sha256_file(Path(__file__)),
        "surfaceCorridors": sha256_file(
            Path(run_physical_ribbon_surface_corridors.__code__.co_filename)
        ),
        "depthFields": sha256_file(
            Path(run_physical_ribbon_depth_fields.__code__.co_filename)
        ),
        "denseCompletion": sha256_file(
            Path(run_physical_ribbon_dense_completion.__code__.co_filename)
        ),
        "flattenedAudit": sha256_file(
            Path(run_physical_ribbon_flattened_audit.__code__.co_filename)
        ),
        "surfaceTopology": sha256_file(
            Path(triangle_edge_region_labels.__code__.co_filename)
        ),
    }
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_VERSION,
        "surface": initial_reference,
        "settings": resolved.record(),
        "implementationSha256": implementation,
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = (
        output
        / f"{PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_STEM}.json"
    )
    if not force and manifest_path.is_file():
        cached = json.loads(manifest_path.read_text())
        final_reference = cached.get("analysis", {}).get("finalSurface", {})
        final_path_value = final_reference.get("manifestPath")
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and final_path_value
            and Path(str(final_path_value)).is_file()
            and sha256_file(Path(str(final_path_value)))
            == final_reference.get("manifestSha256")
        ):
            return cached

    started = time.monotonic()
    current_surface_root = Path(surface_root).resolve()
    current_surface_reference = initial_reference
    rounds: list[dict[str, Any]] = []
    cumulative = {
        "acceptedCorridorCount": 0,
        "attachmentCollarAcceptedCorridorCount": 0,
        "multiRegionSectorAcceptedCorridorCount": 0,
        "topologyNormalizationDuplicatedVertexCount": 0,
        "addedDenseNodeCount": 0,
        "addedTriangleCount": 0,
        "outerLoopReduction": 0,
        "triangleRegionReduction": 0,
        "boundaryEdgeDelta": 0,
    }
    stop_reason = "maximum-rounds-reached"
    saturated = False

    for round_index in range(1, resolved.maximum_rounds + 1):
        round_root = output / f"round-{round_index:03d}"
        corridor_root = round_root / "surface-corridors"
        _notify(
            progress,
            round_index,
            "enumerating-surface-corridors",
            {"surface": current_surface_reference},
        )
        corridor_manifest = run_physical_ribbon_surface_corridors(
            current_surface_root,
            corridor_root,
            settings=resolved.surface_corridors,
            force=force,
        )
        corridor_path = (
            corridor_root / f"{PHYSICAL_RIBBON_SURFACE_CORRIDORS_STEM}.json"
        )
        corridor_count = int(
            corridor_manifest.get("corridors", {}).get(
                "multiAnchorCorridorCount", 0
            )
        )
        scored_count = int(
            corridor_manifest.get("evidence", {}).get(
                "scoredCorridorCount", 0
            )
        )
        domain_count = int(
            corridor_manifest.get("completionDomains", {}).get(
                "exactCompletionDomainCount", 0
            )
        )
        normalization_count = int(
            corridor_manifest.get("surfaceTopologyNormalization", {}).get(
                "duplicatedVertexCount", 0
            )
        )
        enumeration_exhausted = _surface_corridor_enumeration_exhausted(
            corridor_manifest
        )
        round_record: dict[str, Any] = {
            "round": round_index,
            "inputSurface": current_surface_reference,
            "surfaceCorridors": _manifest_reference(
                corridor_path, corridor_manifest
            ),
            "multiAnchorCorridorCount": corridor_count,
            "scoredCorridorCount": scored_count,
            "exactCompletionDomainCount": domain_count,
            "enumerationExhausted": enumeration_exhausted,
            "topologyNormalizationDuplicatedVertexCount": (
                normalization_count
            ),
        }
        _notify(
            progress,
            round_index,
            "surface-corridors-enumerated",
            {
                "corridorCount": corridor_count,
                "scoredCount": scored_count,
                "completionDomainCount": domain_count,
            },
        )
        if domain_count == 0 and normalization_count == 0:
            round_record["appliedCorridorCount"] = 0
            if enumeration_exhausted:
                round_record["outcome"] = "corridor-evidence-saturated"
                stop_reason = "corridor-evidence-saturated"
                saturated = True
            else:
                round_record["outcome"] = "scoring-cap-stalled"
                stop_reason = "scoring-cap-stalled"
            rounds.append(round_record)
            break

        depth_root = round_root / "depth-field"
        _notify(progress, round_index, "solving-strip-depth-fields", {})
        depth_manifest = run_physical_ribbon_depth_fields(
            corridor_root,
            depth_root,
            settings=resolved.depth_fields,
            force=force,
        )
        depth_path = depth_root / f"{PHYSICAL_RIBBON_DEPTH_FIELD_STEM}.json"
        round_record["depthField"] = _manifest_reference(
            depth_path, depth_manifest
        )

        completion_root = round_root / "completion-ungated"
        _notify(progress, round_index, "reconstructing-complete-strips", {})
        completion_manifest = run_physical_ribbon_dense_completion(
            corridor_root,
            depth_root,
            completion_root,
            settings=resolved.dense_completion,
            force=force,
        )
        completion_path = (
            completion_root / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.json"
        )
        accepted_rows = _completion_accepted_rows(completion_manifest)
        round_record["ungatedCompletion"] = _manifest_reference(
            completion_path, completion_manifest
        )
        round_record["ungatedAcceptedCorridorCount"] = len(accepted_rows)
        completion_analysis = completion_manifest.get("analysis", {})
        round_record["ungatedAttachmentCollarEligibleCorridorCount"] = int(
            completion_analysis.get("attachmentCollarEligibleHoleCount", 0)
        )
        round_record["ungatedAttachmentCollarHypothesisCount"] = int(
            completion_analysis.get("attachmentCollarHypothesisCount", 0)
        )
        round_record["ungatedAttachmentCollarAcceptedCorridorCount"] = int(
            completion_analysis.get("attachmentCollarAcceptedHoleCount", 0)
        )
        round_record["ungatedMultiRegionSectorEligibleCorridorCount"] = int(
            completion_analysis.get("multiRegionSectorEligibleHoleCount", 0)
        )
        round_record["ungatedMultiRegionSectorHypothesisCount"] = int(
            completion_analysis.get("multiRegionSectorHypothesisCount", 0)
        )
        round_record["ungatedMultiRegionSectorAcceptedCorridorCount"] = int(
            completion_analysis.get("multiRegionSectorAcceptedHoleCount", 0)
        )
        round_record["textureGatePasses"] = []
        _notify(
            progress,
            round_index,
            "complete-strips-reconstructed",
            {"acceptedCorridorCount": len(accepted_rows)},
        )

        authoritative_root = completion_root
        authoritative_manifest = completion_manifest
        authoritative_rows = accepted_rows
        if accepted_rows:
            audit_root = round_root / "flat-audit-ungated"
            _notify(progress, round_index, "auditing-flattened-texture", {})
            audit_manifest = run_physical_ribbon_flattened_audit(
                completion_root,
                audit_root,
                settings=resolved.flattened_audit,
                force=force,
            )
            audit_path = (
                audit_root / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
            )
            compatible_rows = _texture_compatible_hole_rows(
                completion_manifest, audit_manifest
            )
            round_record["ungatedTextureAudit"] = _manifest_reference(
                audit_path, audit_manifest
            )
            round_record["ungatedTextureCompatibleCorridorCount"] = len(
                compatible_rows
            )
            round_record[
                "ungatedTextureRejectedOrUnmeasuredCorridorCount"
            ] = len(accepted_rows) - len(compatible_rows)

            prior_audit_root = audit_root
            maximum_gate_passes = len(accepted_rows) + 1
            for gate_pass in range(1, maximum_gate_passes + 1):
                gated_root = round_root / f"texture-gate-{gate_pass:03d}"
                _notify(
                    progress,
                    round_index,
                    "replaying-texture-compatible-strips",
                    {
                        "gatePass": gate_pass,
                        "eligibleCorridorCount": len(compatible_rows),
                    },
                )
                gated_manifest = run_physical_ribbon_dense_completion(
                    corridor_root,
                    depth_root,
                    gated_root,
                    settings=resolved.dense_completion,
                    texture_audit_root=prior_audit_root,
                    force=force,
                )
                gated_path = (
                    gated_root
                    / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.json"
                )
                gated_rows = _completion_accepted_rows(gated_manifest)
                gate_record: dict[str, Any] = {
                    "pass": gate_pass,
                    "eligibleCorridorCount": len(compatible_rows),
                    "completion": _manifest_reference(
                        gated_path, gated_manifest
                    ),
                    "acceptedCorridorCount": len(gated_rows),
                }
                round_record["textureGatePasses"].append(gate_record)
                authoritative_root = gated_root
                authoritative_manifest = gated_manifest
                authoritative_rows = gated_rows
                if not gated_rows:
                    break

                gated_audit_root = (
                    round_root
                    / f"texture-gate-{gate_pass:03d}-flat-audit"
                )
                gated_audit_manifest = run_physical_ribbon_flattened_audit(
                    gated_root,
                    gated_audit_root,
                    settings=resolved.flattened_audit,
                    force=force,
                )
                gated_audit_path = (
                    gated_audit_root
                    / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
                )
                gated_compatible_rows = _texture_compatible_hole_rows(
                    gated_manifest, gated_audit_manifest
                )
                gate_record["textureAudit"] = _manifest_reference(
                    gated_audit_path, gated_audit_manifest
                )
                gate_record["textureCompatibleCorridorCount"] = len(
                    gated_compatible_rows
                )
                gate_record[
                    "textureRejectedOrUnmeasuredCorridorCount"
                ] = len(gated_rows) - len(gated_compatible_rows)
                if gated_compatible_rows == gated_rows:
                    break
                if not gated_compatible_rows < gated_rows:
                    raise ValueError(
                        "surface-corridor texture gate did not strictly "
                        "reduce an incompatible accepted set"
                    )
                compatible_rows = gated_compatible_rows
                prior_audit_root = gated_audit_root
            else:
                raise ValueError(
                    "surface-corridor texture gate did not converge"
                )

        applied_count = len(authoritative_rows)
        round_record["appliedCorridorCount"] = applied_count
        if not applied_count and normalization_count == 0:
            if enumeration_exhausted:
                round_record["outcome"] = (
                    "exact-and-texture-evidence-saturated"
                )
                stop_reason = "exact-and-texture-evidence-saturated"
                saturated = True
            else:
                round_record["outcome"] = "scoring-cap-stalled"
                stop_reason = "scoring-cap-stalled"
            rounds.append(round_record)
            break

        final_analysis = authoritative_manifest.get("analysis", {})
        current_surface_root = authoritative_root
        _, _, current_surface_reference = _surface_reference(
            current_surface_root
        )
        round_record["authoritativeCompletion"] = current_surface_reference
        round_record["outcome"] = (
            "advanced-texture-compatible-surface"
            if applied_count
            else "advanced-topology-normalization-only"
        )
        round_record["addedDenseNodeCount"] = int(
            final_analysis.get("addedDenseNodeCount", 0)
        )
        round_record["addedTriangleCount"] = int(
            final_analysis.get("addedTriangleCount", 0)
        )
        round_record["attachmentCollarAcceptedCorridorCount"] = int(
            final_analysis.get("attachmentCollarAcceptedHoleCount", 0)
        )
        round_record["multiRegionSectorAcceptedCorridorCount"] = int(
            final_analysis.get("multiRegionSectorAcceptedHoleCount", 0)
        )
        outer_reduction = int(
            final_analysis.get("outerLoopCountBefore", 0)
            - final_analysis.get("outerLoopCountAfter", 0)
        )
        region_reduction = int(
            final_analysis.get("triangleRegionCountBefore", 0)
            - final_analysis.get("triangleRegionCountAfter", 0)
        )
        boundary_delta = int(
            final_analysis.get("boundaryEdgeCountAfter", 0)
            - final_analysis.get("boundaryEdgeCountBefore", 0)
        )
        round_record["outerLoopReduction"] = outer_reduction
        round_record["triangleRegionReduction"] = region_reduction
        round_record["boundaryEdgeDelta"] = boundary_delta
        cumulative["acceptedCorridorCount"] += applied_count
        cumulative["attachmentCollarAcceptedCorridorCount"] += round_record[
            "attachmentCollarAcceptedCorridorCount"
        ]
        cumulative["multiRegionSectorAcceptedCorridorCount"] += round_record[
            "multiRegionSectorAcceptedCorridorCount"
        ]
        cumulative[
            "topologyNormalizationDuplicatedVertexCount"
        ] += normalization_count
        cumulative["addedDenseNodeCount"] += round_record[
            "addedDenseNodeCount"
        ]
        cumulative["addedTriangleCount"] += round_record[
            "addedTriangleCount"
        ]
        cumulative["outerLoopReduction"] += outer_reduction
        cumulative["triangleRegionReduction"] += region_reduction
        cumulative["boundaryEdgeDelta"] += boundary_delta
        rounds.append(round_record)
        _notify(
            progress,
            round_index,
            "surface-advanced",
            {
                "appliedCorridorCount": applied_count,
                "outerLoopReduction": outer_reduction,
                "triangleRegionReduction": region_reduction,
            },
        )

    final_path, _, final_reference = _surface_reference(current_surface_root)
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_SURFACE_CORRIDOR_SATURATION_VERSION,
        "state": "complete",
        "identity": identity,
        "source": initial_manifest.get("source", {}),
        "analysis": {
            "saturated": saturated,
            "stopReason": stop_reason,
            "completedRoundCount": len(rounds),
            "initialSurface": initial_reference,
            "finalSurface": final_reference,
            "cumulative": cumulative,
            "rounds": rounds,
        },
        "timingSeconds": {"total": round(finished - started, 6)},
        "artifacts": {
            "finalSurfaceRoot": str(final_path.parent),
            "finalSurfaceManifest": str(final_path),
        },
        "method": {
            "decisionUnit": (
                "two complete mutually facing multi-edge frontiers and the "
                "entire dense strip between them; an eligible transverse "
                "island is solved only as one complete sector joining three "
                "regions; no cell, node, edge, or raster pixel is admitted alone"
            ),
            "fixedPoint": (
                "every paired-frontier corridor on a stationary surface must "
                "fit below the declared scoring cap and receive exact "
                "reconstruction before saturation can be claimed"
            ),
            "textureGate": (
                "every admitted strip is independently flattened against "
                "native CT and replayed through an explicit compatible-row "
                "gate until all materialized additions have exhaustive fiber "
                "verdicts"
            ),
            "topology": (
                "vertex-only fan pinches are normalized before enumeration; "
                "ordinary strips must prove exact 2-to-1 outer-loop and "
                "triangle-region merges, while transverse-island sectors must "
                "prove exact 3-to-1 merges, always without intersections, "
                "chart overlap, or a non-manifold edge"
            ),
            "attachmentCollar": (
                "a strip blocked only by exhaustive seam-local intersections "
                "with its own declared attachment regions may retry with a collision-"
                "footprint-sized outward half-space collar; no crossing is "
                "ignored and all native-CT, topology, chart, and flattened-"
                "fiber gates are rerun"
            ),
            "multiRegionSector": (
                "a third region participates only when it is one complete "
                "disk island with exact physical-clone endpoints on both "
                "fronts; both boundary paths and both strip sectors are "
                "enumerated and audited as whole-surface alternatives"
            ),
            "sourceSurfaceMutated": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
