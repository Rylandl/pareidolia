from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION, slab_flake_plane
from .slab_normal_families import NORMAL_FAMILY_VERSION
from .slab_sheetlet_carriers import (
    CARRIER_SCREEN_VERSION,
    _load_carrier_catalog,
    _mls_carrier,
)
from .slab_sheetlet_explore import SHEETLET_EXPLORE_VERSION


NORMAL_FAMILY_EVALUATION_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _quantiles(values: list[float], digits: int = 4) -> dict[str, float | None]:
    if not values:
        return {key: None for key in ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")}
    array = np.asarray(values, dtype=np.float64)
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    percentiles = (0, 10, 25, 50, 75, 90, 100)
    return {
        name: round(float(value), digits)
        for name, value in zip(names, np.percentile(array, percentiles))
    }


def _legacy_flake_equal(
    baseline: dict[str, Any], current: dict[str, Any]
) -> bool:
    return all(
        current.get(key) == value
        for key, value in baseline.items()
        if key != "id"
    )


def _selected_graph_stats(payload: dict[str, Any]) -> dict[str, Any]:
    return payload["stats"]["selected"]


def evaluate_normal_families(
    output_root: str | Path,
    baseline_flake_version: int = 3,
    baseline_explore_version: int = 1,
    baseline_screen_version: int = 1,
) -> dict[str, Any]:
    """Evaluate the additive family model against the preserved legacy graph.

    This is an exploratory construction audit, not evidence that either normal
    family is a physical papyrus sheet.  The declared gates intentionally test
    preservation, collision safety, graph participation, and carrier fit using
    metrics that do not include the neighbor filter used for family inclusion.
    """
    root = Path(output_root)
    started = time.monotonic()
    grid = json.loads((root / "grid.json").read_text())
    plane_count = len(grid["z"])
    baseline_component_path = (
        root / f"sheetlets-explore-v{baseline_explore_version}-components.npz"
    )
    current_component_path = (
        root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}-components.npz"
    )
    with np.load(baseline_component_path) as payload:
        baseline_source_z = np.asarray(payload["sourceZIndex"], dtype=np.int32)
        baseline_source_flake = np.asarray(payload["sourceFlakeId"], dtype=np.int32)
        baseline_component = np.asarray(payload["component"], dtype=np.int32)
    with np.load(current_component_path) as payload:
        current_source_z = np.asarray(payload["sourceZIndex"], dtype=np.int32)
        current_source_flake = np.asarray(payload["sourceFlakeId"], dtype=np.int32)
        current_component = np.asarray(payload["component"], dtype=np.int32)

    baseline_component_by_plane: list[np.ndarray] = []
    for z_index in range(plane_count):
        member = baseline_source_z == z_index
        maximum_id = int(np.max(baseline_source_flake[member], initial=-1))
        lookup = np.full(maximum_id + 1, -1, dtype=np.int32)
        lookup[baseline_source_flake[member]] = baseline_component[member]
        baseline_component_by_plane.append(lookup)

    numeric_mismatch_count = 0
    membership_mismatch_count = 0
    primary_flake_count = 0
    secondary_flake_count = 0
    cross_family_shared_needle_count = 0
    current_to_baseline_by_plane: list[np.ndarray] = []
    baseline_cells_by_plane: list[np.ndarray] = []
    plane_audits = []
    for z_index in range(plane_count):
        baseline_path = root / (
            f"flakes-v{baseline_flake_version}-z{z_index}-k3.json"
        )
        baseline_membership_path = root / (
            f"flakes-v{baseline_flake_version}-z{z_index}-k3-members.npz"
        )
        current = slab_flake_plane(root, z_index, 3)
        current_path = root / f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3.json"
        current_membership_path = root / (
            f"flakes-v{FLAKE_CACHE_VERSION}-z{z_index}-k3-members.npz"
        )
        baseline = json.loads(baseline_path.read_text())
        # Read the cached JSON as well as the function result so this audit
        # covers the persisted representation consumed by later processes.
        current = json.loads(current_path.read_text())
        old_flakes = baseline["flakes"]
        new_flakes = current["flakes"]
        primary = [
            flake for flake in new_flakes if int(flake.get("normalFamily", 0)) == 0
        ]
        secondary = [
            flake for flake in new_flakes if int(flake.get("normalFamily", 0)) == 1
        ]
        primary_flake_count += len(primary)
        secondary_flake_count += len(secondary)
        numeric_mismatch_count += abs(len(old_flakes) - len(primary))
        pair_count = min(len(old_flakes), len(primary))
        numeric_mismatch_count += sum(
            not _legacy_flake_equal(old_flakes[index], primary[index])
            for index in range(pair_count)
        )

        new_to_old = np.full(len(new_flakes), -1, dtype=np.int32)
        for old_index, flake in enumerate(primary[:pair_count]):
            new_to_old[int(flake["id"])] = old_index
        current_to_baseline_by_plane.append(new_to_old)
        baseline_cells_by_plane.append(
            np.asarray([flake["cellIndex"] for flake in old_flakes], dtype=np.int32)
        )

        with np.load(baseline_membership_path) as payload:
            old_offsets = np.asarray(payload["offsets"], dtype=np.int64)
            old_ids = np.asarray(payload["ids"], dtype=np.int64)
        with np.load(current_membership_path) as payload:
            new_offsets = np.asarray(payload["offsets"], dtype=np.int64)
            new_ids = np.asarray(payload["ids"], dtype=np.int64)
        for old_index, flake in enumerate(primary[:pair_count]):
            new_index = int(flake["id"])
            old_values = old_ids[old_offsets[old_index] : old_offsets[old_index + 1]]
            new_values = new_ids[new_offsets[new_index] : new_offsets[new_index + 1]]
            membership_mismatch_count += int(not np.array_equal(old_values, new_values))

        secondary_cells = {
            tuple(int(value) for value in flake["cellIndex"])
            for flake in secondary
        }
        family_membership: dict[tuple[int, int, int], list[set[int]]] = {
            cell: [set(), set()] for cell in secondary_cells
        }
        for flake in new_flakes:
            cell = tuple(int(value) for value in flake["cellIndex"])
            if cell not in family_membership:
                continue
            family_index = int(flake.get("normalFamily", 0))
            low, high = new_offsets[int(flake["id"]) : int(flake["id"]) + 2]
            family_membership[cell][family_index].update(
                int(value) for value in new_ids[int(low) : int(high)]
            )
        plane_shared = sum(
            len(primary_ids.intersection(secondary_ids))
            for primary_ids, secondary_ids in family_membership.values()
        )
        cross_family_shared_needle_count += plane_shared
        plane_audits.append(
            {
                "zIndex": z_index,
                "baselineFlakeCount": len(old_flakes),
                "primaryFlakeCount": len(primary),
                "secondaryFlakeCount": len(secondary),
                "crossFamilySharedNeedleCount": plane_shared,
            }
        )

    baseline_for_current = np.full(len(current_component), -1, dtype=np.int32)
    for z_index in range(plane_count):
        member = current_source_z == z_index
        current_ids = current_source_flake[member]
        old_ids = current_to_baseline_by_plane[z_index][current_ids]
        valid = old_ids >= 0
        node_indices = np.flatnonzero(member)
        lookup = baseline_component_by_plane[z_index]
        baseline_for_current[node_indices[valid]] = lookup[old_ids[valid]]

    _, candidate_payload, loaded_component, flakes = _load_carrier_catalog(root)
    if not np.array_equal(loaded_component.astype(np.int32), current_component):
        raise ValueError("the active carrier catalog does not match its component artifact")
    family = np.asarray(
        [int(flake.get("normalFamily", 0)) for flake in flakes], dtype=np.uint8
    )
    cell = np.asarray([flake["cellIndex"] for flake in flakes], dtype=np.int32)
    quality = np.asarray([flake["quality"] for flake in flakes], dtype=np.float32)
    augmented_candidates = [
        candidate
        for candidate in candidate_payload["candidates"]
        if int(candidate.get("secondaryFamilyNodeCount", 0)) > 0
    ]
    augmented_ids = {int(value["componentId"]) for value in augmented_candidates}
    members_by_component: dict[int, list[int]] = {value: [] for value in augmented_ids}
    for node_index in np.flatnonzero(np.isin(current_component, list(augmented_ids))):
        members_by_component[int(current_component[node_index])].append(int(node_index))

    selected_baseline_components = {
        int(value)
        for component_id in augmented_ids
        for value in baseline_for_current[members_by_component[component_id]]
        if int(value) >= 0
    }
    baseline_cell_sets = {value: set() for value in selected_baseline_components}
    for node_index, component_id in enumerate(baseline_component):
        key = int(component_id)
        if key not in baseline_cell_sets:
            continue
        z_index = int(baseline_source_z[node_index])
        flake_id = int(baseline_source_flake[node_index])
        baseline_cell_sets[key].add(
            tuple(int(value) for value in baseline_cells_by_plane[z_index][flake_id])
        )

    secondary_height: list[float] = []
    secondary_normal: list[float] = []
    matched_primary_height: list[float] = []
    matched_primary_normal: list[float] = []
    matched_quality_delta: list[float] = []
    component_audits = []
    secondary_new_cell_count = 0
    secondary_node_count = 0
    bridge_component_count = 0
    for candidate in augmented_candidates:
        component_id = int(candidate["componentId"])
        member_indices = np.asarray(
            members_by_component[component_id], dtype=np.int64
        )
        member_flakes = [flakes[int(index)] for index in member_indices]
        carrier = _mls_carrier(
            member_flakes,
            pixel_step=4.0,
            bandwidth=48.0,
            support_radius=48.0,
            maximum_pixels=192,
        )
        member_family = family[member_indices]
        member_quality = quality[member_indices]
        height = np.asarray(carrier["nodeHeightResidualVoxels"])
        normal = np.asarray(carrier["nodeNormalResidualDeg"])
        secondary_positions = np.flatnonzero(member_family == 1)
        primary_positions = np.flatnonzero(member_family == 0)
        secondary_height.extend(float(height[index]) for index in secondary_positions)
        secondary_normal.extend(float(normal[index]) for index in secondary_positions)
        if len(primary_positions):
            for secondary_position in secondary_positions:
                match = primary_positions[
                    int(
                        np.argmin(
                            np.abs(
                                member_quality[primary_positions]
                                - member_quality[secondary_position]
                            )
                        )
                    )
                ]
                matched_primary_height.append(float(height[match]))
                matched_primary_normal.append(float(normal[match]))
                matched_quality_delta.append(
                    abs(float(member_quality[match] - member_quality[secondary_position]))
                )
        source_components = sorted(
            {
                int(value)
                for value in baseline_for_current[member_indices]
                if int(value) >= 0
            }
        )
        bridge_component_count += int(len(source_components) >= 2)
        baseline_cells = set().union(
            *(baseline_cell_sets[value] for value in source_components)
        ) if source_components else set()
        secondary_cells = {
            tuple(int(value) for value in cell[member_indices[index]])
            for index in secondary_positions
        }
        new_cells = len(secondary_cells - baseline_cells)
        secondary_new_cell_count += new_cells
        secondary_node_count += len(secondary_positions)
        component_audits.append(
            {
                "componentId": component_id,
                "memberCount": len(member_indices),
                "secondaryNodeCount": len(secondary_positions),
                "sourceBaselineComponentCount": len(source_components),
                "newSecondaryCellCount": new_cells,
                "medianSecondaryHeightResidualVoxels": round(
                    float(np.median(height[secondary_positions])), 3
                ),
                "medianSecondaryNormalResidualDeg": round(
                    float(np.median(normal[secondary_positions])), 3
                ),
            }
        )

    baseline_graph = json.loads(
        (root / f"sheetlets-explore-v{baseline_explore_version}.json").read_text()
    )
    current_graph = json.loads(
        (root / f"sheetlets-explore-v{SHEETLET_EXPLORE_VERSION}.json").read_text()
    )
    old_selected = _selected_graph_stats(baseline_graph)
    new_selected = _selected_graph_stats(current_graph)
    graph_comparison = {
        key: {
            "baseline": old_selected.get(key),
            "current": new_selected.get(key),
            "delta": (
                float(new_selected[key]) - float(old_selected[key])
                if old_selected.get(key) is not None and new_selected.get(key) is not None
                else None
            ),
        }
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
        )
    }

    screen_comparison: dict[str, Any] | None = None
    baseline_screen_path = root / (
        f"sheetlet-carrier-screen-v{baseline_screen_version}.json"
    )
    current_screen_path = root / f"sheetlet-carrier-screen-v{CARRIER_SCREEN_VERSION}.json"
    if baseline_screen_path.is_file() and current_screen_path.is_file():
        baseline_screen = json.loads(baseline_screen_path.read_text())
        current_screen = json.loads(current_screen_path.read_text())
        old_by_component = {
            int(value["componentId"]): value for value in baseline_screen["candidates"]
        }
        new_by_component = {
            int(value["componentId"]): value for value in current_screen["candidates"]
        }
        secondary_seed_screen = [
            value
            for value in current_screen["candidates"]
            if int(
                value.get("carrier", {})
                .get("normalFamilies", {})
                .get("1", {})
                .get("flakeCount", 0)
            )
            == int(value["memberCount"])
        ]
        deltas: dict[str, list[float]] = {
            "heightResidualVoxels": [],
            "normalResidualDeg": [],
            "supportedAreaSquareVoxels": [],
            "bestTextureScore": [],
            "constructionYieldScore": [],
        }
        for audit in component_audits:
            component_id = int(audit["componentId"])
            members = np.asarray(members_by_component[component_id], dtype=np.int64)
            sources = sorted(
                {
                    int(value)
                    for value in baseline_for_current[members]
                    if int(value) >= 0
                }
            )
            if len(sources) != 1 or sources[0] not in old_by_component:
                continue
            if component_id not in new_by_component:
                continue
            old = old_by_component[sources[0]]
            new = new_by_component[component_id]
            deltas["heightResidualVoxels"].append(
                float(new["carrier"]["medianNodeHeightResidualVoxels"])
                - float(old["carrier"]["medianNodeHeightResidualVoxels"])
            )
            deltas["normalResidualDeg"].append(
                float(new["carrier"]["medianNodeNormalResidualDeg"])
                - float(old["carrier"]["medianNodeNormalResidualDeg"])
            )
            deltas["supportedAreaSquareVoxels"].append(
                float(new["yield"]["supportedAreaSquareVoxels"])
                - float(old["yield"]["supportedAreaSquareVoxels"])
            )
            deltas["bestTextureScore"].append(
                float(new["texture"]["bestTextureScore"])
                - float(old["texture"]["bestTextureScore"])
            )
            deltas["constructionYieldScore"].append(
                float(new["yield"]["constructionYieldScore"])
                - float(old["yield"]["constructionYieldScore"])
            )
        screen_comparison = {
            "singleBaselineSourceComparisonCount": len(
                deltas["heightResidualVoxels"]
            ),
            "deltas": {key: _quantiles(values) for key, values in deltas.items()},
            "heightResidualDegradeOver025Count": sum(
                value > 0.25 for value in deltas["heightResidualVoxels"]
            ),
            "normalResidualDegradeOver1DegCount": sum(
                value > 1.0 for value in deltas["normalResidualDeg"]
            ),
            "secondarySeeds": {
                "count": len(secondary_seed_screen),
                "grossSupportedAreaSquareVoxels": round(
                    sum(
                        float(value["yield"]["supportedAreaSquareVoxels"])
                        for value in secondary_seed_screen
                    ),
                    2,
                ),
                "supportedAreaSquareVoxels": _quantiles(
                    [
                        float(value["yield"]["supportedAreaSquareVoxels"])
                        for value in secondary_seed_screen
                    ],
                    2,
                ),
                "fitFactor": _quantiles(
                    [
                        float(value["yield"]["fitFactor"])
                        for value in secondary_seed_screen
                    ]
                ),
                "bestTextureScore": _quantiles(
                    [
                        float(value["texture"]["bestTextureScore"])
                        for value in secondary_seed_screen
                    ]
                ),
            },
        }

    secondary_height_stats = _quantiles(secondary_height, 3)
    secondary_normal_stats = _quantiles(secondary_normal, 3)
    matched_height_stats = _quantiles(matched_primary_height, 3)
    matched_normal_stats = _quantiles(matched_primary_normal, 3)
    criteria = {
        "primaryFlakesExactlyPreserved": numeric_mismatch_count == 0
        and membership_mismatch_count == 0,
        "crossFamilyNeedleOwnershipIsDisjoint": cross_family_shared_needle_count == 0,
        "sheetletGraphHasNoWithinComponentCellCollisions": int(
            new_selected["cellCollisionCount"]
        )
        == 0,
        "secondaryNodesParticipateInGraph": float(
            new_selected.get("secondaryLinkedNodeFraction", 0.0)
        )
        >= 0.25,
        "medianSecondaryCarrierHeightResidualAtMost3Voxels": float(
            secondary_height_stats["median"]
            if secondary_height_stats["median"] is not None
            else np.inf
        )
        <= 3.0,
        "medianSecondaryCarrierNormalResidualAtMost6Deg": float(
            secondary_normal_stats["median"]
            if secondary_normal_stats["median"] is not None
            else np.inf
        )
        <= 6.0,
        "retainsAtLeast98PercentOfLongSpanComponents": int(
            new_selected["longSpanComponentCount"]
        )
        >= 0.98 * int(old_selected["longSpanComponentCount"]),
        "retainsAtLeast98PercentOfAllPlaneComponents": int(
            new_selected["allAxialPlaneComponentCount"]
        )
        >= 0.98 * int(old_selected["allAxialPlaneComponentCount"]),
    }
    component_audits.sort(
        key=lambda value: (
            -int(value["sourceBaselineComponentCount"]),
            -int(value["secondaryNodeCount"]),
        )
    )
    result = {
        "identity": {
            "version": NORMAL_FAMILY_EVALUATION_VERSION,
            "normalFamilyVersion": NORMAL_FAMILY_VERSION,
            "flakeVersion": FLAKE_CACHE_VERSION,
            "sheetletExploreVersion": SHEETLET_EXPLORE_VERSION,
            "carrierScreenVersion": CARRIER_SCREEN_VERSION,
            "baselineFlakeVersion": baseline_flake_version,
            "baselineExploreVersion": baseline_explore_version,
            "baselineScreenVersion": baseline_screen_version,
        },
        "declaredCriteria": {
            "note": (
                "Neighbor agreement controls secondary-family inclusion and is not "
                "reused as an evaluation criterion. Holdout replication is reported "
                "separately and is not inferred by this construction audit."
            ),
            "criteria": criteria,
            "allConstructionCriteriaPass": all(criteria.values()),
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "primaryFlakeCount": primary_flake_count,
            "secondaryFlakeCount": secondary_flake_count,
            "primaryNumericMismatchCount": numeric_mismatch_count,
            "primaryMembershipMismatchCount": membership_mismatch_count,
            "crossFamilySharedNeedleCount": cross_family_shared_needle_count,
            "secondaryCandidateCount": len(augmented_candidates),
            "crossBaselineBridgeCandidateCount": bridge_component_count,
            "secondaryNodeCountInCandidates": secondary_node_count,
            "secondaryCandidateCellCount": secondary_new_cell_count,
            "secondaryCarrierHeightResidualVoxels": secondary_height_stats,
            "qualityMatchedPrimaryHeightResidualVoxels": matched_height_stats,
            "secondaryCarrierNormalResidualDeg": secondary_normal_stats,
            "qualityMatchedPrimaryNormalResidualDeg": matched_normal_stats,
            "qualityMatchAbsoluteDelta": _quantiles(matched_quality_delta, 4),
        },
        "graphComparison": graph_comparison,
        "screenComparison": screen_comparison,
        "planeAudits": plane_audits,
        "componentAudits": component_audits,
        "constraint": (
            "one local surface per Acus cell inside a sheetlet component; distinct "
            "normal families remain separate components"
        ),
    }
    output_path = root / (
        f"normal-family-evaluation-v{NORMAL_FAMILY_EVALUATION_VERSION}.json"
    )
    _atomic_json(output_path, result)
    return result
