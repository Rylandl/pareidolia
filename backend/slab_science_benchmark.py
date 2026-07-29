from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_normal_families import NORMAL_FAMILY_VERSION
from .slab_normal_family_evaluation import NORMAL_FAMILY_EVALUATION_VERSION
from .slab_sheetlet_carriers import CARRIER_SCREEN_VERSION
from .slab_sheetlet_explore import SHEETLET_EXPLORE_VERSION


SCIENCE_BENCHMARK_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _content_identity(path: Path) -> dict[str, Any]:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(4 * 1024 * 1024):
            digest.update(block)
    return {
        "path": path.name,
        "bytes": path.stat().st_size,
        "sha256": digest.hexdigest(),
    }


def _active_artifact_paths(root: Path, plane_count: int) -> list[Path]:
    paths = [
        root / "analysis.json",
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.json",
        root / f"normal-families-v{NORMAL_FAMILY_VERSION}.npy",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-candidates.json",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz",
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-edges.npz",
        root / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json",
        root
        / f"normal-family-evaluation-v{NORMAL_FAMILY_EVALUATION_VERSION}.json",
    ]
    for z_index in range(plane_count):
        paths.extend(
            [
                root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json",
                root
                / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3-members.npz",
            ]
        )
    missing = [path.name for path in paths if not path.is_file()]
    if missing:
        raise ValueError(
            "the active benchmark artifact set is incomplete: "
            + ", ".join(missing)
        )
    return paths


def collect_science_metrics(output_root: str | Path) -> dict[str, Any]:
    root = Path(output_root)
    analysis = json.loads((root / "analysis.json").read_text())
    grid = json.loads((root / "grid.json").read_text())
    family = json.loads(
        (root / f"normal-families-v{NORMAL_FAMILY_VERSION}.json").read_text()
    )
    graph = json.loads(
        (root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json").read_text()
    )
    evaluation = json.loads(
        (
            root
            / f"normal-family-evaluation-v{NORMAL_FAMILY_EVALUATION_VERSION}.json"
        ).read_text()
    )
    selected = graph["stats"]["selected"]
    family_stats = family["stats"]
    evaluation_stats = evaluation["stats"]
    secondary_screen = evaluation.get("screenComparison", {}).get(
        "secondarySeeds", {}
    )
    return {
        "sourceIdentity": analysis["identity"],
        "analysis": {
            "gridShapeZYX": [len(grid["z"]), len(grid["y"]), len(grid["x"])],
            "validCellCount": int(analysis["validCellCount"]),
            "needleCount": int(analysis["needleCount"]),
        },
        "normalFamilies": {
            key: family_stats[key]
            for key in (
                "secondaryFittedCellCount",
                "standaloneCandidateCellCount",
                "includedSecondaryCellCount",
                "includedSecondaryCellFraction",
                "candidateComponentCount",
                "includedComponentCount",
                "alignedCandidateEdgeCount",
                "largestIncludedComponentSize",
                "medianIncludedCoverage",
                "medianIncludedConfidence",
                "medianIncludedAmbiguousFraction",
                "medianIncludedOverlapFraction",
                "includedCellsByPlane",
            )
        },
        "flakeAudit": {
            key: evaluation_stats[key]
            for key in (
                "primaryFlakeCount",
                "secondaryFlakeCount",
                "primaryNumericMismatchCount",
                "primaryMembershipMismatchCount",
                "crossFamilySharedNeedleCount",
            )
        },
        "sheetletGraph": {
            key: selected[key]
            for key in (
                "linkedNodeCount",
                "retainedLinkCount",
                "componentCount",
                "largestComponentSize",
                "longSpanComponentCount",
                "allAxialPlaneComponentCount",
                "allSixPlaneComponentCount",
                "cellCollisionCount",
                "medianEdgeResidualVoxels",
                "medianFiberAngleDeg",
                "medianNormalBendDeg",
                "secondaryLinkedNodeCount",
                "secondaryLinkedNodeFraction",
            )
        },
        "secondarySeeds": {
            "candidateCount": evaluation_stats["secondaryCandidateCount"],
            "nodeCount": evaluation_stats["secondaryNodeCountInCandidates"],
            "cellCount": evaluation_stats["secondaryCandidateCellCount"],
            "carrierHeightResidualVoxels": evaluation_stats[
                "secondaryCarrierHeightResidualVoxels"
            ],
            "carrierNormalResidualDeg": evaluation_stats[
                "secondaryCarrierNormalResidualDeg"
            ],
            "grossSupportedAreaSquareVoxels": secondary_screen.get(
                "grossSupportedAreaSquareVoxels"
            ),
            "supportedAreaSquareVoxels": secondary_screen.get(
                "supportedAreaSquareVoxels"
            ),
            "fitFactor": secondary_screen.get("fitFactor"),
            "bestTextureScore": secondary_screen.get("bestTextureScore"),
        },
        "declaredCriteria": evaluation["declaredCriteria"],
    }


def _guard_results(metrics: dict[str, Any], baseline: dict[str, Any]) -> dict[str, bool]:
    current_graph = metrics["sheetletGraph"]
    baseline_graph = baseline["metrics"]["sheetletGraph"]
    flake = metrics["flakeAudit"]
    secondary = metrics["secondarySeeds"]
    return {
        "constructionCriteriaPass": bool(
            metrics["declaredCriteria"]["allConstructionCriteriaPass"]
        ),
        "primaryValuesPreserved": int(flake["primaryNumericMismatchCount"]) == 0,
        "primaryMembershipsPreserved": int(
            flake["primaryMembershipMismatchCount"]
        )
        == 0,
        "needleOwnershipDisjoint": int(flake["crossFamilySharedNeedleCount"]) == 0,
        "cellCollisionsRemainZero": int(current_graph["cellCollisionCount"]) == 0,
        "longSpanRetentionAtLeast98Percent": int(
            current_graph["longSpanComponentCount"]
        )
        >= 0.98 * int(baseline_graph["longSpanComponentCount"]),
        "allPlaneRetentionAtLeast98Percent": int(
            current_graph["allAxialPlaneComponentCount"]
        )
        >= 0.98 * int(baseline_graph["allAxialPlaneComponentCount"]),
        "secondaryMedianHeightAtMost3Voxels": float(
            secondary["carrierHeightResidualVoxels"]["median"]
        )
        <= 3.0,
        "secondaryMedianNormalAtMost6Deg": float(
            secondary["carrierNormalResidualDeg"]["median"]
        )
        <= 6.0,
    }


def _metric_differences(
    baseline: Any, current: Any, prefix: str = ""
) -> list[dict[str, Any]]:
    if isinstance(baseline, dict) and isinstance(current, dict):
        output = []
        for key in sorted(set(baseline) | set(current)):
            path = f"{prefix}.{key}" if prefix else key
            if key not in baseline or key not in current:
                output.append(
                    {
                        "metric": path,
                        "baseline": baseline.get(key),
                        "current": current.get(key),
                        "delta": None,
                    }
                )
            else:
                output.extend(_metric_differences(baseline[key], current[key], path))
        return output
    if isinstance(baseline, list) and isinstance(current, list):
        if baseline == current:
            return []
        return [
            {
                "metric": prefix,
                "baseline": baseline,
                "current": current,
                "delta": None,
            }
        ]
    if baseline == current:
        return []
    numeric = (
        isinstance(baseline, (int, float))
        and not isinstance(baseline, bool)
        and isinstance(current, (int, float))
        and not isinstance(current, bool)
    )
    return [
        {
            "metric": prefix,
            "baseline": baseline,
            "current": current,
            "delta": float(current) - float(baseline) if numeric else None,
        }
    ]


def freeze_science_benchmark(
    output_root: str | Path,
    destination: str | Path,
) -> dict[str, Any]:
    root = Path(output_root)
    metrics = collect_science_metrics(root)
    plane_count = int(metrics["analysis"]["gridShapeZYX"][0])
    artifacts = [
        _content_identity(path)
        for path in _active_artifact_paths(root, plane_count)
    ]
    payload = {
        "identity": {
            "version": SCIENCE_BENCHMARK_VERSION,
            "name": "cross-scroll-z512-multinormal",
            "sourceIdentity": metrics["sourceIdentity"],
        },
        "scope": (
            "frozen multi-normal evidence, flake, independent sheetlet graph, "
            "and secondary-carrier construction metrics; excludes legacy "
            "single-normal carrier descendants"
        ),
        "metrics": metrics,
        "artifacts": artifacts,
        "stats": {
            "artifactCount": len(artifacts),
            "artifactBytes": sum(int(value["bytes"]) for value in artifacts),
        },
    }
    _atomic_json(Path(destination), payload)
    return payload


def compare_science_benchmark(
    output_root: str | Path,
    baseline_path: str | Path,
    verify_artifacts: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    baseline = json.loads(Path(baseline_path).read_text())
    current = collect_science_metrics(root)
    differences = _metric_differences(baseline["metrics"], current)
    artifact_audit = None
    if verify_artifacts:
        expected = {value["path"]: value for value in baseline["artifacts"]}
        current_artifacts = {
            value["path"]: value
            for value in (
                _content_identity(path)
                for path in _active_artifact_paths(
                    root, int(current["analysis"]["gridShapeZYX"][0])
                )
            )
        }
        artifact_audit = {
            "unchangedCount": sum(
                current_artifacts.get(key) == value
                for key, value in expected.items()
            ),
            "changed": [
                key
                for key, value in expected.items()
                if current_artifacts.get(key) != value
            ],
        }
    guards = _guard_results(current, baseline)
    return {
        "identity": baseline["identity"],
        "stats": {
            "metricDifferenceCount": len(differences),
            "guardCount": len(guards),
            "passedGuardCount": sum(guards.values()),
            "allGuardsPass": all(guards.values()),
        },
        "guards": guards,
        "metricDifferences": differences,
        "artifactAudit": artifact_audit,
        "currentMetrics": current,
    }
