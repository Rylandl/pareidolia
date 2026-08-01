from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .physical_ribbon_dense_completion import (
    PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA,
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
    PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
    PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM,
    PhysicalRibbonFlattenedAuditSettings,
    run_physical_ribbon_flattened_audit,
)
from .physical_ribbon_open_bays import (
    PHYSICAL_RIBBON_OPEN_BAYS_STEM,
    PhysicalRibbonOpenBaySettings,
    _resolve_prior_manifest,
    run_physical_ribbon_open_bays,
)
from .physical_ribbon_surface_holes import _resolve_surface_manifest


PHYSICAL_RIBBON_OPEN_BAY_SATURATION_SCHEMA = (
    "pareidolia.physical-ribbon-open-bay-saturation"
)
PHYSICAL_RIBBON_OPEN_BAY_SATURATION_VERSION = 1
PHYSICAL_RIBBON_OPEN_BAY_SATURATION_STEM = (
    "physical-ribbon-open-bay-saturation-v1"
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
class PhysicalRibbonOpenBaySaturationSettings:
    """Run complete frontier-bay reconstruction to an evidence fixed point."""

    maximum_rounds: int = 16
    open_bays: PhysicalRibbonOpenBaySettings = field(
        default_factory=PhysicalRibbonOpenBaySettings
    )
    depth_fields: PhysicalRibbonDepthFieldSettings = field(
        default_factory=PhysicalRibbonDepthFieldSettings
    )
    dense_completion: PhysicalRibbonDenseCompletionSettings = field(
        default_factory=lambda: PhysicalRibbonDenseCompletionSettings(
            maximum_completed_holes=128
        )
    )
    flattened_audit: PhysicalRibbonFlattenedAuditSettings = field(
        default_factory=lambda: PhysicalRibbonFlattenedAuditSettings(
            maximum_components=128
        )
    )

    def __post_init__(self) -> None:
        if self.maximum_rounds < 1:
            raise ValueError("open-bay saturation requires at least one round")
        if (
            self.dense_completion.maximum_completed_holes
            < self.open_bays.maximum_scored_holes
        ):
            raise ValueError(
                "dense completion must evaluate every scored open bay before "
                "a fixed point can be declared"
            )
        if (
            self.flattened_audit.maximum_components
            < self.dense_completion.maximum_completed_holes
        ):
            raise ValueError(
                "flattened audit must be able to measure every independently "
                "accepted completion"
            )

    @classmethod
    def from_record(
        cls, record: Mapping[str, Any]
    ) -> "PhysicalRibbonOpenBaySaturationSettings":
        allowed = {
            "maximum_rounds",
            "open_bays",
            "depth_fields",
            "dense_completion",
            "flattened_audit",
        }
        unexpected = set(record) - allowed
        if unexpected:
            raise ValueError(
                "unknown open-bay saturation settings: "
                + ", ".join(sorted(unexpected))
            )
        open_values = _tuple_fields(
            _settings_section(record, "open_bays"),
            ("profile_depth_fractions", "competing_shift_thicknesses"),
        )
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
        dense_values.setdefault("maximum_completed_holes", 128)
        audit_values.setdefault("maximum_components", 128)
        return cls(
            maximum_rounds=int(record.get("maximum_rounds", 16)),
            open_bays=PhysicalRibbonOpenBaySettings(**open_values),
            depth_fields=PhysicalRibbonDepthFieldSettings(**depth_values),
            dense_completion=PhysicalRibbonDenseCompletionSettings(**dense_values),
            flattened_audit=PhysicalRibbonFlattenedAuditSettings(**audit_values),
        )

    def record(self) -> dict[str, Any]:
        return {
            "maximum_rounds": self.maximum_rounds,
            "open_bays": self.open_bays.record(),
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


def _unique_roots(values: Sequence[str | Path]) -> list[Path]:
    result: list[Path] = []
    seen: set[Path] = set()
    for value in values:
        path = Path(value).resolve()
        if path in seen:
            continue
        seen.add(path)
        result.append(path)
    return result


def _append_unique(values: list[Path], value: Path) -> None:
    resolved = value.resolve()
    if resolved not in values:
        values.append(resolved)


def _prior_evidence_references(
    completion_roots: Sequence[Path],
    texture_audit_roots: Sequence[Path],
    *,
    settings: PhysicalRibbonOpenBaySaturationSettings,
) -> list[dict[str, Any]]:
    references: list[dict[str, Any]] = []
    for root in completion_roots:
        path, manifest = _resolve_prior_manifest(
            root,
            schema=PHYSICAL_RIBBON_DENSE_COMPLETION_SCHEMA,
            label="prior completion",
        )
        if canonical_json_hash(
            manifest.get("identity", {}).get("settings", {})
        ) != canonical_json_hash(settings.dense_completion.record()):
            raise ValueError(
                "prior completion settings differ from the saturation solve"
            )
        references.append(
            {
                "kind": "denseCompletion",
                **_manifest_reference(path, manifest),
            }
        )
    for root in texture_audit_roots:
        path, manifest = _resolve_prior_manifest(
            root,
            schema=PHYSICAL_RIBBON_FLATTENED_AUDIT_SCHEMA,
            label="prior texture audit",
        )
        if canonical_json_hash(
            manifest.get("identity", {}).get("settings", {})
        ) != canonical_json_hash(settings.flattened_audit.record()):
            raise ValueError(
                "prior flattened-audit settings differ from the saturation solve"
            )
        references.append(
            {
                "kind": "flattenedTextureAudit",
                **_manifest_reference(path, manifest),
            }
        )
    return references


def _completion_accepted_rows(
    completion: Mapping[str, Any],
) -> frozenset[int]:
    rows = [
        int(record["holeRow"])
        for record in completion.get("completions", ())
        if bool(record.get("accepted"))
    ]
    if len(rows) != len(set(rows)):
        raise ValueError("dense completion accepted one hole row more than once")
    declared = int(completion.get("analysis", {}).get("acceptedHoleCount", -1))
    if declared != len(rows):
        raise ValueError("dense-completion accepted count differs from its records")
    return frozenset(rows)


def _stationary_round_decision(
    *,
    retained_candidate_count: int,
    maximum_scored_holes: int,
    failure_kind: str,
) -> tuple[bool, str]:
    """Decide whether a no-mutation round exhausted or merely filled its cap."""

    if retained_candidate_count >= maximum_scored_holes:
        return True, "scoring-cap-exhausted-retry-same-surface"
    return False, f"{failure_kind}-evidence-saturated"


def _notify(
    progress: ProgressCallback | None,
    round_index: int,
    stage: str,
    values: Mapping[str, Any],
) -> None:
    if progress is not None:
        progress(round_index, stage, values)


def run_physical_ribbon_open_bay_saturation(
    surface_root: str | Path,
    output_root: str | Path,
    *,
    settings: PhysicalRibbonOpenBaySaturationSettings | None = None,
    prior_completion_roots: Sequence[str | Path] = (),
    prior_texture_audit_roots: Sequence[str | Path] = (),
    force: bool = False,
    progress: ProgressCallback | None = None,
) -> dict[str, Any]:
    resolved = settings or PhysicalRibbonOpenBaySaturationSettings()
    _, initial_manifest, initial_reference = _surface_reference(
        surface_root
    )
    completion_evidence = _unique_roots(prior_completion_roots)
    texture_evidence = _unique_roots(prior_texture_audit_roots)
    prior_references = _prior_evidence_references(
        completion_evidence,
        texture_evidence,
        settings=resolved,
    )
    implementation = {
        "orchestration": sha256_file(Path(__file__)),
        "openBays": sha256_file(
            Path(run_physical_ribbon_open_bays.__code__.co_filename)
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
    }
    identity: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_OPEN_BAY_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_OPEN_BAY_SATURATION_VERSION,
        "surface": initial_reference,
        "priorRejectionEvidence": prior_references,
        "settings": resolved.record(),
        "implementationSha256": implementation,
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{PHYSICAL_RIBBON_OPEN_BAY_SATURATION_STEM}.json"
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
        "acceptedBayCount": 0,
        "addedDenseNodeCount": 0,
        "addedTriangleCount": 0,
        "boundaryEdgeReduction": 0,
    }
    stop_reason = "maximum-rounds-reached"
    saturated = False

    for round_index in range(1, resolved.maximum_rounds + 1):
        round_root = output / f"round-{round_index:03d}"
        open_root = round_root / "open-bays"
        _notify(
            progress,
            round_index,
            "enumerating-open-bays",
            {"surface": current_surface_reference},
        )
        open_manifest = run_physical_ribbon_open_bays(
            current_surface_root,
            open_root,
            settings=resolved.open_bays,
            prior_completion_roots=tuple(completion_evidence),
            prior_texture_audit_roots=tuple(texture_evidence),
            force=force,
        )
        open_path = open_root / f"{PHYSICAL_RIBBON_OPEN_BAYS_STEM}.json"
        open_analysis = open_manifest.get("analysis", {})
        geometry = open_analysis.get("bayGeometry", {})
        scoring = open_analysis.get("scoring", {})
        retained_count = int(geometry.get("retainedGeometryCandidateCount", 0))
        scored_count = int(scoring.get("scoredHoleCount", 0))
        round_record: dict[str, Any] = {
            "round": round_index,
            "inputSurface": current_surface_reference,
            "openBays": _manifest_reference(open_path, open_manifest),
            "rawGeometryCandidateCount": int(
                geometry.get("rawGeometryCandidateCount", 0)
            ),
            "cachedEvidenceRejectionCount": int(
                geometry.get("cachedEvidenceRejectionCount", 0)
            ),
            "retainedGeometryCandidateCount": retained_count,
            "scoredHoleCount": scored_count,
        }
        _notify(
            progress,
            round_index,
            "open-bays-enumerated",
            {
                "retainedCandidateCount": retained_count,
                "scoredHoleCount": scored_count,
            },
        )
        if scored_count == 0:
            round_record["outcome"] = "no-uncached-complete-bays"
            rounds.append(round_record)
            stop_reason = "candidate-evidence-saturated"
            saturated = True
            break

        depth_root = round_root / "depth-field"
        _notify(progress, round_index, "solving-depth-fields", {})
        depth_manifest = run_physical_ribbon_depth_fields(
            open_root,
            depth_root,
            settings=resolved.depth_fields,
            force=force,
        )
        depth_path = depth_root / f"{PHYSICAL_RIBBON_DEPTH_FIELD_STEM}.json"
        round_record["depthField"] = _manifest_reference(
            depth_path, depth_manifest
        )

        completion_root = round_root / "completion"
        _notify(progress, round_index, "reconstructing-complete-bays", {})
        completion_manifest = run_physical_ribbon_dense_completion(
            open_root,
            depth_root,
            completion_root,
            settings=resolved.dense_completion,
            force=force,
        )
        completion_path = (
            completion_root / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.json"
        )
        _append_unique(completion_evidence, completion_root)
        accepted_rows = _completion_accepted_rows(completion_manifest)
        round_record["ungatedCompletion"] = _manifest_reference(
            completion_path, completion_manifest
        )
        round_record["ungatedAcceptedHoleCount"] = len(accepted_rows)
        round_record["textureGatePasses"] = []
        _notify(
            progress,
            round_index,
            "complete-bays-reconstructed",
            {"acceptedHoleCount": len(accepted_rows)},
        )

        if not accepted_rows:
            retry, reason = _stationary_round_decision(
                retained_candidate_count=retained_count,
                maximum_scored_holes=resolved.open_bays.maximum_scored_holes,
                failure_kind="exact",
            )
            round_record["appliedHoleCount"] = 0
            round_record["outcome"] = reason
            rounds.append(round_record)
            if retry:
                continue
            stop_reason = reason
            saturated = True
            break

        audit_root = round_root / "flat-audit"
        _notify(progress, round_index, "auditing-flattened-texture", {})
        audit_manifest = run_physical_ribbon_flattened_audit(
            completion_root,
            audit_root,
            settings=resolved.flattened_audit,
            force=force,
        )
        audit_path = audit_root / f"{PHYSICAL_RIBBON_FLATTENED_AUDIT_STEM}.json"
        _append_unique(texture_evidence, audit_root)
        compatible_rows = _texture_compatible_hole_rows(
            completion_manifest, audit_manifest
        )
        round_record["ungatedTextureAudit"] = _manifest_reference(
            audit_path, audit_manifest
        )
        round_record["ungatedTextureCompatibleHoleCount"] = len(
            compatible_rows
        )
        round_record["ungatedTextureRejectedOrUnmeasuredHoleCount"] = (
            len(accepted_rows) - len(compatible_rows)
        )

        authoritative_root = completion_root
        authoritative_manifest = completion_manifest
        authoritative_rows = accepted_rows
        prior_audit_root = audit_root
        gate_pass = 0
        while authoritative_rows and len(compatible_rows) < len(
            authoritative_rows
        ):
            if not compatible_rows:
                authoritative_rows = frozenset()
                break
            gate_pass += 1
            gated_root = round_root / f"texture-gate-{gate_pass:03d}"
            _notify(
                progress,
                round_index,
                "replaying-texture-compatible-bays",
                {
                    "gatePass": gate_pass,
                    "eligibleHoleCount": len(compatible_rows),
                },
            )
            gated_manifest = run_physical_ribbon_dense_completion(
                open_root,
                depth_root,
                gated_root,
                settings=resolved.dense_completion,
                texture_audit_root=prior_audit_root,
                force=force,
            )
            gated_path = (
                gated_root / f"{PHYSICAL_RIBBON_DENSE_COMPLETION_STEM}.json"
            )
            _append_unique(completion_evidence, gated_root)
            gated_rows = _completion_accepted_rows(gated_manifest)
            gate_record: dict[str, Any] = {
                "pass": gate_pass,
                "eligibleHoleCount": len(compatible_rows),
                "completion": _manifest_reference(gated_path, gated_manifest),
                "acceptedHoleCount": len(gated_rows),
            }
            round_record["textureGatePasses"].append(gate_record)
            authoritative_root = gated_root
            authoritative_manifest = gated_manifest
            authoritative_rows = gated_rows
            if not gated_rows:
                compatible_rows = frozenset()
                break
            gated_audit_root = (
                round_root / f"texture-gate-{gate_pass:03d}-flat-audit"
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
            _append_unique(texture_evidence, gated_audit_root)
            compatible_rows = _texture_compatible_hole_rows(
                gated_manifest, gated_audit_manifest
            )
            gate_record["textureAudit"] = _manifest_reference(
                gated_audit_path, gated_audit_manifest
            )
            gate_record["textureCompatibleHoleCount"] = len(compatible_rows)
            gate_record["textureRejectedOrUnmeasuredHoleCount"] = (
                len(gated_rows) - len(compatible_rows)
            )
            prior_audit_root = gated_audit_root

        applied_count = len(authoritative_rows)
        round_record["appliedHoleCount"] = applied_count
        if not applied_count:
            retry, reason = _stationary_round_decision(
                retained_candidate_count=retained_count,
                maximum_scored_holes=resolved.open_bays.maximum_scored_holes,
                failure_kind="texture",
            )
            round_record["outcome"] = reason
            rounds.append(round_record)
            if retry:
                continue
            stop_reason = reason
            saturated = True
            break

        final_analysis = authoritative_manifest.get("analysis", {})
        current_surface_root = authoritative_root
        _, _, current_surface_reference = _surface_reference(
            current_surface_root
        )
        round_record["authoritativeCompletion"] = current_surface_reference
        round_record["outcome"] = "advanced-texture-compatible-surface"
        round_record["addedDenseNodeCount"] = int(
            final_analysis.get("addedDenseNodeCount", 0)
        )
        round_record["addedTriangleCount"] = int(
            final_analysis.get("addedTriangleCount", 0)
        )
        round_record["boundaryEdgeReduction"] = int(
            final_analysis.get("boundaryEdgeReduction", 0)
        )
        round_record["triangleRegionDelta"] = int(
            final_analysis.get("triangleRegionCountAfter", 0)
            - final_analysis.get("triangleRegionCountBefore", 0)
        )
        round_record["interiorHoleDelta"] = int(
            final_analysis.get("interiorHoleCountAfter", 0)
            - final_analysis.get("interiorHoleCountBefore", 0)
        )
        cumulative["acceptedBayCount"] += applied_count
        cumulative["addedDenseNodeCount"] += round_record[
            "addedDenseNodeCount"
        ]
        cumulative["addedTriangleCount"] += round_record[
            "addedTriangleCount"
        ]
        cumulative["boundaryEdgeReduction"] += round_record[
            "boundaryEdgeReduction"
        ]
        rounds.append(round_record)
        _notify(
            progress,
            round_index,
            "surface-advanced",
            {
                "appliedHoleCount": applied_count,
                "boundaryEdgeReduction": round_record[
                    "boundaryEdgeReduction"
                ],
            },
        )

    final_path, _, final_reference = _surface_reference(
        current_surface_root
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": PHYSICAL_RIBBON_OPEN_BAY_SATURATION_SCHEMA,
        "version": PHYSICAL_RIBBON_OPEN_BAY_SATURATION_VERSION,
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
                "one complete multi-edge frontier bay reconstructed as one "
                "dense constrained surface; no cell or pixel is admitted alone"
            ),
            "fixedPoint": (
                "unchanged evidence-hashed failures are removed before each "
                "ranking cap, and a stationary surface is retried until the "
                "uncached candidate set itself is exhausted"
            ),
            "textureGate": (
                "every exact completion is flattened against native CT and "
                "replayed until every retained proposal has an exhaustive, "
                "proposal-local compatible fiber verdict"
            ),
            "topology": (
                "each accepted stage reruns exact collision, manifold, loop, "
                "and component audits before becoming the next source surface"
            ),
            "sourceSurfaceMutated": False,
            "singleCellGrowth": False,
            "identityLabelsUsed": False,
        },
    }
    atomic_json(manifest_path, payload)
    return payload
