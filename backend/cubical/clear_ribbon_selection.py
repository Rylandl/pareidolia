from __future__ import annotations

import colorsys
import heapq
import json
import math
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .clear_ribbon import CLEAR_RIBBON_SCHEMA, CLEAR_RIBBON_STEM, _draw_line
from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .export import rgb_png
from .isolated_slab import _percentile_record


CLEAR_RIBBON_SELECTION_SCHEMA = "pareidolia.clear-ribbon-collision-safe-selection"
CLEAR_RIBBON_SELECTION_VERSION = 1
CLEAR_RIBBON_SELECTION_STEM = "clear-ribbon-selection-v1"

SELECTION_CLASS_UNSELECTED = 0
SELECTION_CLASS_UPSTREAM_ANCHOR = 1
SELECTION_CLASS_ANCHORED_GROWTH = 2
SELECTION_CLASS_NEW_CLEAR_CORE = 3


@dataclass(frozen=True, slots=True)
class ClearRibbonSelectionSettings:
    minimum_growth_bottleneck: float = 0.35
    minimum_new_component_ribbons: int = 8
    maximum_preview_labels: int = 128

    def __post_init__(self) -> None:
        if not 0.0 < self.minimum_growth_bottleneck <= 1.0:
            raise ValueError("clear-ribbon growth bottleneck must lie in (0, 1]")
        if (
            self.minimum_new_component_ribbons < 1
            or self.maximum_preview_labels < 1
        ):
            raise ValueError("clear-ribbon selection counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_bank(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value if value.is_file() else value / f"{CLEAR_RIBBON_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != CLEAR_RIBBON_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("clear-ribbon selection requires a complete ribbon bank")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("clear-ribbon data hash differs from its manifest")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return manifest_path, manifest, arrays


def _adjacency(
    ribbon_count: int,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    affinity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.concatenate((edge_first, edge_second)).astype(np.int32)
    target = np.concatenate((edge_second, edge_first)).astype(np.int32)
    edge = np.tile(np.arange(len(edge_first), dtype=np.int32), 2)
    weight = np.concatenate((affinity, affinity)).astype(np.float32)
    order = np.argsort(source, kind="stable")
    count = np.bincount(source, minlength=ribbon_count)
    offset = np.concatenate(((0,), np.cumsum(count, dtype=np.int64)))
    return offset, target[order], edge[order], weight[order]


def select_clear_ribbons(
    bank: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    settings: ClearRibbonSelectionSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Select collision-free anchored growth and substantial new clear cores."""

    ribbon_count = len(bank["pairedCandidateIndex"])
    component = np.asarray(bank["ribbonComponent"], dtype=np.int32)
    component_size = np.asarray(bank["componentRibbonCount"], dtype=np.int32)
    component_assembly_count = np.asarray(bank["componentAssemblyCount"])
    component_sole_assembly = np.asarray(
        bank["componentSoleAssemblyLabel"], dtype=np.int32
    )
    key = np.asarray(bank["spatialKeyXYZ"], dtype=np.int32)
    local_evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    upstream_selected = np.asarray(bank["selectedPairedSurface"]) > 0
    upstream_label = np.asarray(bank["selectedAssemblyLabel"], dtype=np.int32)
    edge_first = np.asarray(bank["edgeFirstRibbon"], dtype=np.int32)
    edge_second = np.asarray(bank["edgeSecondRibbon"], dtype=np.int32)
    edge_affinity = np.asarray(bank["edgeAffinity"], dtype=np.float32)
    offset, neighbor, adjacency_edge, adjacency_affinity = _adjacency(
        ribbon_count, edge_first, edge_second, edge_affinity
    )

    label = np.full(ribbon_count, -1, dtype=np.int32)
    path_score = np.zeros(ribbon_count, dtype=np.float32)
    parent = np.full(ribbon_count, -1, dtype=np.int32)
    parent_edge = np.full(ribbon_count, -1, dtype=np.int32)
    selection_class = np.zeros(ribbon_count, dtype=np.uint8)
    occupied = np.full(
        processing_shape_sampling_xyz[::-1], -1, dtype=np.int32
    )
    anchor_key_collision = 0
    for ribbon in np.flatnonzero(upstream_selected):
        z, y, x = key[ribbon, 2], key[ribbon, 1], key[ribbon, 0]
        if occupied[z, y, x] >= 0:
            anchor_key_collision += 1
            continue
        if upstream_label[ribbon] < 0:
            raise ValueError("selected clear ribbon has no upstream assembly")
        occupied[z, y, x] = ribbon
        label[ribbon] = upstream_label[ribbon]
        path_score[ribbon] = 1.0
        selection_class[ribbon] = SELECTION_CLASS_UPSTREAM_ANCHOR
    if anchor_key_collision:
        raise ValueError("upstream selected ribbons violate spatial-key exclusivity")

    collision_rejected = np.zeros(ribbon_count, dtype=bool)
    proposal_count = 0
    queue: list[tuple[float, int, int, int, int]] = []
    for ribbon in np.flatnonzero(upstream_selected):
        value = int(component[ribbon])
        if component_assembly_count[value] != 1:
            continue
        for adjacency_index in range(
            int(offset[ribbon]), int(offset[ribbon + 1])
        ):
            target = int(neighbor[adjacency_index])
            score = min(
                float(adjacency_affinity[adjacency_index]),
                float(local_evidence[target]),
            )
            if score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue,
                (
                    -score,
                    target,
                    int(label[ribbon]),
                    int(ribbon),
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1
    while queue:
        negative_score, ribbon, proposed_label, proposed_parent, proposed_edge = (
            heapq.heappop(queue)
        )
        if label[ribbon] >= 0 or collision_rejected[ribbon]:
            continue
        value = int(component[ribbon])
        if (
            component_assembly_count[value] != 1
            or int(component_sole_assembly[value]) != proposed_label
        ):
            continue
        z, y, x = key[ribbon, 2], key[ribbon, 1], key[ribbon, 0]
        if occupied[z, y, x] >= 0:
            collision_rejected[ribbon] = True
            continue
        score = -negative_score
        label[ribbon] = proposed_label
        path_score[ribbon] = score
        parent[ribbon] = proposed_parent
        parent_edge[ribbon] = proposed_edge
        selection_class[ribbon] = SELECTION_CLASS_ANCHORED_GROWTH
        occupied[z, y, x] = ribbon
        for adjacency_index in range(
            int(offset[ribbon]), int(offset[ribbon + 1])
        ):
            target = int(neighbor[adjacency_index])
            if label[target] >= 0 or component[target] != value:
                continue
            candidate_score = min(
                score,
                float(adjacency_affinity[adjacency_index]),
                float(local_evidence[target]),
            )
            if candidate_score < settings.minimum_growth_bottleneck:
                continue
            heapq.heappush(
                queue,
                (
                    -candidate_score,
                    target,
                    proposed_label,
                    ribbon,
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1

    maximum_existing_label = int(np.max(upstream_label, initial=-1))
    unseeded_component = np.flatnonzero(
        (component_assembly_count == 0)
        & (component_size >= settings.minimum_new_component_ribbons)
    )
    component_median_evidence = np.zeros(len(component_size), dtype=np.float32)
    for value in unseeded_component:
        component_median_evidence[value] = float(
            np.median(local_evidence[component == value])
        )
    new_order = sorted(
        (int(value) for value in unseeded_component),
        key=lambda value: (
            -int(component_size[value]),
            -float(component_median_evidence[value]),
            value,
        ),
    )
    new_label_by_component = np.full(len(component_size), -1, dtype=np.int32)
    rejected_new_component_count = 0
    for value in new_order:
        candidates = np.flatnonzero(component == value)
        free = occupied[
            key[candidates, 2], key[candidates, 1], key[candidates, 0]
        ] < 0
        candidates = candidates[free]
        if not len(candidates):
            rejected_new_component_count += 1
            continue
        seed_order = np.lexsort((candidates, -local_evidence[candidates]))
        seed = int(candidates[seed_order[0]])
        proposed_label = maximum_existing_label + 1 + value
        local_queue: list[tuple[float, int, int, int]] = [
            (-float(local_evidence[seed]), seed, -1, -1)
        ]
        added: list[int] = []
        locally_blocked: set[int] = set()
        while local_queue:
            negative_score, ribbon, proposed_parent, proposed_edge = heapq.heappop(
                local_queue
            )
            if label[ribbon] >= 0 or ribbon in locally_blocked:
                continue
            z, y, x = key[ribbon, 2], key[ribbon, 1], key[ribbon, 0]
            if occupied[z, y, x] >= 0:
                collision_rejected[ribbon] = True
                locally_blocked.add(ribbon)
                continue
            score = -negative_score
            label[ribbon] = proposed_label
            path_score[ribbon] = score
            parent[ribbon] = proposed_parent
            parent_edge[ribbon] = proposed_edge
            selection_class[ribbon] = SELECTION_CLASS_NEW_CLEAR_CORE
            occupied[z, y, x] = ribbon
            added.append(ribbon)
            for adjacency_index in range(
                int(offset[ribbon]), int(offset[ribbon + 1])
            ):
                target = int(neighbor[adjacency_index])
                if label[target] >= 0 or component[target] != value:
                    continue
                candidate_score = min(
                    score,
                    float(adjacency_affinity[adjacency_index]),
                    float(local_evidence[target]),
                )
                if candidate_score < settings.minimum_growth_bottleneck:
                    continue
                heapq.heappush(
                    local_queue,
                    (
                        -candidate_score,
                        target,
                        ribbon,
                        int(adjacency_edge[adjacency_index]),
                    ),
                )
                proposal_count += 1
        if len(added) < settings.minimum_new_component_ribbons:
            rejected_new_component_count += 1
            for ribbon in added:
                z, y, x = key[ribbon, 2], key[ribbon, 1], key[ribbon, 0]
                if int(occupied[z, y, x]) == ribbon:
                    occupied[z, y, x] = -1
                label[ribbon] = -1
                path_score[ribbon] = 0.0
                parent[ribbon] = -1
                parent_edge[ribbon] = -1
                selection_class[ribbon] = SELECTION_CLASS_UNSELECTED
            continue
        new_label_by_component[value] = proposed_label

    selected = label >= 0
    same_label_edge = (
        selected[edge_first]
        & selected[edge_second]
        & (label[edge_first] == label[edge_second])
        & (edge_affinity >= settings.minimum_growth_bottleneck)
    )
    same_label_neighbor_count = np.bincount(
        np.concatenate(
            (edge_first[same_label_edge], edge_second[same_label_edge])
        ),
        minlength=ribbon_count,
    ).astype(np.uint16)
    component_selected_count = np.bincount(
        component[selected], minlength=len(component_size)
    ).astype(np.int32)
    new_selected = selection_class == SELECTION_CLASS_NEW_CLEAR_CORE
    new_label, new_size = np.unique(label[new_selected], return_counts=True)
    anchored_growth = selection_class == SELECTION_CLASS_ANCHORED_GROWTH
    arrays = {
        "selectedAssemblyLabel": label,
        "pathBottleneck": path_score,
        "parentRibbon": parent,
        "parentContinuityEdge": parent_edge,
        "selectionClass": selection_class,
        "selected": selected.astype(np.uint8),
        "selectedSameLabelNeighborCount": same_label_neighbor_count,
        "collisionRejected": collision_rejected.astype(np.uint8),
        "componentSelectedRibbonCount": component_selected_count,
        "newAssemblyLabelByComponent": new_label_by_component,
    }
    return arrays, {
        "ribbonCount": ribbon_count,
        "upstreamAnchorCount": int(np.count_nonzero(upstream_selected)),
        "anchoredGrowthCount": int(np.count_nonzero(anchored_growth)),
        "newClearCoreRibbonCount": int(np.count_nonzero(new_selected)),
        "selectedRibbonCount": int(np.count_nonzero(selected)),
        "newClearCoreCount": int(len(new_label)),
        "rejectedNewComponentCount": rejected_new_component_count,
        "collisionRejectedRibbonCount": int(np.count_nonzero(collision_rejected)),
        "proposalCount": proposal_count,
        "anchoredGrowthPathBottleneck": _percentile_record(
            path_score[anchored_growth]
        ),
        "newClearCorePathBottleneck": _percentile_record(
            path_score[new_selected]
        ),
        "newClearCoreSize": _percentile_record(new_size),
        "selectedSameLabelNeighborCount": _percentile_record(
            same_label_neighbor_count[selected]
        ),
    }


def _label_colors(labels: np.ndarray, maximum: int) -> dict[int, tuple[int, int, int]]:
    value, count = np.unique(labels[labels >= 0], return_counts=True)
    order = np.lexsort((value, -count))[:maximum]
    return {
        int(value[index]): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.07 + 0.61803398875 * rank) % 1.0, 0.68, 0.98
            )
        )
        for rank, index in enumerate(order)
    }


def write_clear_ribbon_selection_projection(
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    path: str | Path,
    *,
    maximum_labels: int,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    point = np.asarray(bank["midpointXYZ"])
    label = np.asarray(selection["selectedAssemblyLabel"], dtype=np.int32)
    colors = _label_colors(label, maximum_labels)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        for value, color in colors.items():
            points = point[label == value]
            normalized = (
                points[:, list(axes)] - world_start_xyz[None, list(axes)]
            ) / width[None, list(axes)]
            x = np.rint(
                offset + margin + normalized[:, 0] * (panel_size - 2 * margin)
            ).astype(np.int32)
            y = np.rint(
                panel_size - margin - normalized[:, 1] * (panel_size - 2 * margin)
            ).astype(np.int32)
            valid = (
                (x >= offset)
                & (x < offset + panel_size)
                & (y >= 0)
                & (y < panel_size)
            )
            canvas[y[valid], x[valid]] = color
        border = (64, 72, 84)
        canvas[margin, offset + margin : offset + panel_size - margin] = border
        canvas[
            panel_size - margin,
            offset + margin : offset + panel_size - margin,
        ] = border
        canvas[margin : panel_size - margin, offset + margin] = border
        canvas[
            margin : panel_size - margin, offset + panel_size - margin
        ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def write_clear_ribbon_selection_cross_sections(
    source: VolumeSource,
    owned: VoxelBounds,
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    path: str | Path,
    *,
    display_high_raw: float,
    sampling_stride: int,
) -> Path:
    output = Path(path)
    volume = source.memmap()
    midpoint = np.asarray(bank["midpointXYZ"])
    lower = np.asarray(bank["boundaryLowerXYZ"])
    upper = np.asarray(bank["boundaryUpperXYZ"])
    selected = np.asarray(selection["selected"]) > 0
    selection_class = np.asarray(selection["selectionClass"])
    world_start = np.asarray(owned.start_xyz) + np.asarray(source.origin_xyz)
    z_values = np.linspace(owned.start_xyz[2], owned.stop_xyz_exclusive[2] - 1, 3)
    y_values = np.linspace(owned.start_xyz[1], owned.stop_xyz_exclusive[1] - 1, 3)
    views = [("z", int(round(value))) for value in z_values] + [
        ("y", int(round(value))) for value in y_values
    ]
    panel_width = owned.shape_xyz[0]
    panel_height = max(owned.shape_xyz[1], owned.shape_xyz[2])
    canvas = np.full(
        (2 * panel_height, 3 * panel_width, 3), (7, 10, 14), dtype=np.uint8
    )
    tolerance = max(1.0, sampling_stride)
    colors = {
        SELECTION_CLASS_UPSTREAM_ANCHOR: (43, 226, 190),
        SELECTION_CLASS_ANCHORED_GROWTH: (90, 178, 255),
        SELECTION_CLASS_NEW_CLEAR_CORE: (255, 174, 62),
    }
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[2]
            near = np.flatnonzero(
                selected & (np.abs(midpoint[:, 2] - coordinate) <= tolerance)
            )
            axes = (0, 1)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[1]
            near = np.flatnonzero(
                selected & (np.abs(midpoint[:, 1] - coordinate) <= tolerance)
            )
            axes = (0, 2)
        gray = np.clip(
            np.asarray(raw, dtype=np.float32) / max(display_high_raw, 1.0) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        for ribbon in near:
            first = lower[ribbon, list(axes)] - world_start[list(axes)]
            second = upper[ribbon, list(axes)] - world_start[list(axes)]
            _draw_line(
                panel,
                first,
                second,
                colors[int(selection_class[ribbon])],
            )
        row, column = divmod(view_index, 3)
        y0 = row * panel_height + (panel_height - panel.shape[0]) // 2
        x0 = column * panel_width
        canvas[y0 : y0 + panel.shape[0], x0 : x0 + panel.shape[1]] = panel
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def _write_npz(path: Path, arrays: Mapping[str, np.ndarray]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def run_clear_ribbon_selection(
    bank_root: str | Path,
    output_root: str | Path,
    *,
    settings: ClearRibbonSelectionSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or ClearRibbonSelectionSettings()
    bank_path, bank_manifest, bank = _load_bank(bank_root)
    identity: dict[str, Any] = {
        "schema": CLEAR_RIBBON_SELECTION_SCHEMA,
        "version": CLEAR_RIBBON_SELECTION_VERSION,
        "ribbonBank": {
            "manifestPath": str(bank_path),
            "manifestSha256": sha256_file(bank_path),
            "dataSha256": bank_manifest["data"]["sha256"],
        },
        "settings": resolved.record(),
        "implementationSha256": sha256_file(Path(__file__)),
    }
    identity["identitySha256"] = canonical_json_hash(identity)
    output = Path(output_root).resolve()
    output.mkdir(parents=True, exist_ok=True)
    manifest_path = output / f"{CLEAR_RIBBON_SELECTION_STEM}.json"
    data_path = output / f"{CLEAR_RIBBON_SELECTION_STEM}.npz"
    if not force and manifest_path.is_file() and data_path.is_file():
        cached = json.loads(manifest_path.read_text())
        if (
            cached.get("state") == "complete"
            and cached.get("identity", {}).get("identitySha256")
            == identity["identitySha256"]
            and cached.get("data", {}).get("sha256") == sha256_file(data_path)
        ):
            return cached

    started = time.monotonic()
    processing_shape = tuple(
        int(value)
        for value in bank_manifest["geometry"]["processingShapeSamplingXYZ"]
    )
    selection, stats = select_clear_ribbons(
        bank,
        processing_shape_sampling_xyz=processing_shape,
        settings=resolved,
    )
    selected = time.monotonic()
    _write_npz(data_path, selection)
    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"]["metadataPath"]
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]),
        tuple(owned_record["stopXYZExclusive"]),
    )
    world_record = bank_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    projection = write_clear_ribbon_selection_projection(
        bank,
        selection,
        world_start,
        world_stop,
        output / "selected-clear-ribbons.png",
        maximum_labels=resolved.maximum_preview_labels,
    )
    cross_sections = write_clear_ribbon_selection_cross_sections(
        source,
        owned,
        bank,
        selection,
        output / "selected-clear-ribbon-cross-sections.png",
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=int(bank_manifest["samplingStrideVoxels"]),
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": CLEAR_RIBBON_SELECTION_SCHEMA,
        "version": CLEAR_RIBBON_SELECTION_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": bank_manifest["geometry"],
        "calibration": bank_manifest["calibration"],
        "selection": stats,
        "timingSeconds": {
            "selection": round(selected - started, 6),
            "writingAndArtifacts": round(finished - selected, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(selection),
        },
        "artifacts": {
            "selection": projection.name,
            "crossSections": cross_sections.name,
        },
        "selectionClass": {
            "0": "unselected or explicitly deferred",
            "1": "upstream selected paired-surface anchor",
            "2": "collision-safe growth in a single-assembly component",
            "3": "new unseeded clear core surviving minimum size",
        },
        "method": {
            "propagation": "maximum-bottleneck ribbon forest",
            "hardConstraint": "at most one ribbon per source-lattice key",
            "contestedComponentPolicy": "preserve upstream anchors; defer interior",
            "newIdentityPolicy": (
                "unseeded strict component must retain the configured ribbon "
                "minimum after global key exclusivity"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
