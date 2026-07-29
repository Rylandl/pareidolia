from __future__ import annotations

import hashlib
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .slab_flakes import FLAKE_CACHE_VERSION
from .slab_monotone_layers import (
    MONOTONE_LAYER_VERSION,
    _parity_consistent_links,
    window_artifact_suffix,
)
from .slab_sheetlet_explore import _components_without_cell_collisions
from .slab_window_scheduler import WINDOW_SCHEDULER_VERSION


GLOBAL_MONOTONE_GRAPH_VERSION = 1


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2) + "\n")
    temporary.replace(path)


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
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


def _quantiles(values: np.ndarray, digits: int = 4) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    names = ("minimum", "p10", "p25", "median", "p75", "p90", "maximum")
    if not len(values):
        return {name: None for name in names}
    return {
        name: round(float(value), digits)
        for name, value in zip(
            names, np.percentile(values, (0, 10, 25, 50, 75, 90, 100))
        )
    }


def _pair_key(values: np.ndarray) -> np.ndarray:
    contiguous = np.ascontiguousarray(values, dtype=np.uint64)
    return contiguous.view(np.dtype([("first", "<u8"), ("second", "<u8")])).reshape(
        -1
    )


def _canonical_edge_scores(
    arrays: dict[str, np.ndarray], node_identity: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    first = node_identity[arrays["source"].astype(np.int64)]
    second = node_identity[arrays["target"].astype(np.int64)]
    values = np.stack((np.minimum(first, second), np.maximum(first, second)), axis=1)
    if not len(values):
        return np.empty((0, 2), dtype=np.uint64), np.empty(0, dtype=np.float32)
    pair, first_index, inverse = np.unique(
        values, axis=0, return_index=True, return_inverse=True
    )
    score = arrays["score"].astype(np.float32)
    selected_score = score[first_index]
    if np.any(np.abs(selected_score[inverse] - score) > 1.0e-6):
        raise ValueError("one canonical raw edge has conflicting local scores")
    return pair, selected_score


def _node_cells(
    root: Path, node_identity: np.ndarray
) -> tuple[np.ndarray, list[dict[str, Any]]]:
    source_z = (node_identity >> np.uint64(32)).astype(np.int32)
    source_id = (node_identity & np.uint64(0xFFFFFFFF)).astype(np.int64)
    cell = np.empty((len(node_identity), 3), dtype=np.uint16)
    identities = []
    for z_index in np.unique(source_z):
        path = root / f"flakes-v{FLAKE_CACHE_VERSION}-z{int(z_index)}-k3.json"
        payload = json.loads(path.read_text())
        identities.append(payload["identity"])
        selected = np.flatnonzero(source_z == z_index)
        flakes = payload["flakes"]
        maximum_id = len(flakes) - 1
        if np.any(source_id[selected] > maximum_id):
            raise ValueError("consensus node references a missing source flake")
        for node_index in selected:
            flake = flakes[int(source_id[node_index])]
            if int(flake["id"]) != int(source_id[node_index]):
                raise ValueError("flake cache IDs are not dense and index aligned")
            if int(flake.get("normalFamily", 0)) != 0:
                raise ValueError("global monotone graph contains a non-primary flake")
            cell[node_index] = np.asarray(flake["cellIndex"], dtype=np.uint16)
    return cell, identities


def _component_plane_counts(
    component: np.ndarray, cell: np.ndarray, component_count: int
) -> np.ndarray:
    pairs = np.stack((component.astype(np.int64), cell[:, 2].astype(np.int64)), axis=1)
    unique = np.unique(pairs, axis=0)
    return np.bincount(unique[:, 0], minlength=component_count).astype(np.uint8)


def build_global_monotone_graph(
    output_root: str | Path,
    force: bool = False,
) -> dict[str, Any]:
    root = Path(output_root)
    schedule_stem = f"tiled-window-schedule-v{WINDOW_SCHEDULER_VERSION}"
    schedule_summary_path = root / f"{schedule_stem}.json"
    schedule_artifact_path = root / f"{schedule_stem}.npz"
    schedule = json.loads(schedule_summary_path.read_text())
    monotone_paths = [
        root
        / (
            f"monotone-layer-window-v{MONOTONE_LAYER_VERSION}"
            f"{window_artifact_suffix(value['originCellXYZ'])}.npz"
        )
        for value in schedule["windows"]
    ]
    identity = {
        "version": GLOBAL_MONOTONE_GRAPH_VERSION,
        "scheduleIdentity": schedule["identity"],
        "scheduleArtifact": _content_identity(schedule_artifact_path),
        "monotoneArtifacts": [_content_identity(path) for path in monotone_paths],
    }
    stem = f"global-monotone-graph-v{GLOBAL_MONOTONE_GRAPH_VERSION}"
    summary_path = root / f"{stem}.json"
    artifact_path = root / f"{stem}.npz"
    if summary_path.is_file() and artifact_path.is_file() and not force:
        cached = json.loads(summary_path.read_text())
        if cached.get("identity") == identity:
            cached["stats"]["cacheHit"] = True
            cached["stats"]["elapsedMs"] = 0.0
            return cached

    started = time.monotonic()
    with np.load(schedule_artifact_path) as payload:
        tiled = {key: np.asarray(payload[key]) for key in payload.files}
    node_identity = tiled["nodeIdentity"].astype(np.uint64)
    cell, _ = _node_cells(root, node_identity)
    raw_pair = tiled["rawEdgeIdentity"].astype(np.uint64)
    raw_parity = tiled["rawEdgeRelativeParity"].astype(np.int8)
    raw_unanimous = tiled["rawEdgeUnanimous"].astype(bool)
    parity_unanimous = tiled["rawEdgeParityUnanimous"].astype(bool)
    selected_consensus = raw_unanimous & parity_unanimous & (raw_parity != 0)
    consensus_pair = raw_pair[selected_consensus]
    consensus_parity = raw_parity[selected_consensus]
    consensus_key = _pair_key(consensus_pair)

    minimum_score = np.full(len(consensus_pair), np.inf, dtype=np.float32)
    maximum_score = np.full(len(consensus_pair), -np.inf, dtype=np.float32)
    score_observation_count = np.zeros(len(consensus_pair), dtype=np.uint8)
    for path in monotone_paths:
        with np.load(path) as payload:
            monotone = {key: np.asarray(payload[key]) for key in payload.files}
        local_identity = (
            monotone["sourceZIndex"].astype(np.uint64) << np.uint64(32)
        ) | monotone["sourceFlakeId"].astype(np.uint64)
        local_pair, local_score = _canonical_edge_scores(monotone, local_identity)
        position = np.searchsorted(consensus_key, _pair_key(local_pair))
        inside = position < len(consensus_pair)
        valid = np.zeros(len(local_pair), dtype=bool)
        valid[inside] = np.all(
            consensus_pair[position[inside]] == local_pair[inside], axis=1
        )
        position = position[valid]
        local_score = local_score[valid]
        np.minimum.at(minimum_score, position, local_score)
        np.maximum.at(maximum_score, position, local_score)
        np.add.at(score_observation_count, position, np.uint8(1))
    if np.any(~np.isfinite(minimum_score)):
        raise ValueError("a consensus raw edge has no score observation")

    source = np.searchsorted(node_identity, consensus_pair[:, 0]).astype(np.uint32)
    target = np.searchsorted(node_identity, consensus_pair[:, 1]).astype(np.uint32)
    if np.any(node_identity[source] != consensus_pair[:, 0]) or np.any(
        node_identity[target] != consensus_pair[:, 1]
    ):
        raise ValueError("a consensus edge endpoint is absent from the node catalog")
    grid = json.loads((root / "grid.json").read_text())
    grid_shape_xyz = np.asarray(
        [len(grid["x"]), len(grid["y"]), len(grid["z"])], dtype=np.int64
    )
    cell_code = (
        cell[:, 0].astype(np.int64)
        + grid_shape_xyz[0]
        * (
            cell[:, 1].astype(np.int64)
            + grid_shape_xyz[1] * cell[:, 2].astype(np.int64)
        )
    )
    _, _, _, collision_retained = _components_without_cell_collisions(
        len(node_identity), cell_code, source, target, minimum_score
    )
    retained, branch_parity, parity_frustrated, parity_stats = (
        _parity_consistent_links(
            len(node_identity),
            source,
            target,
            consensus_parity,
            minimum_score,
            collision_retained,
        )
    )
    component, component_sizes, degree, retained_again = (
        _components_without_cell_collisions(
            len(node_identity),
            cell_code,
            source[retained],
            target[retained],
            minimum_score[retained],
        )
    )
    if not np.all(retained_again):
        raise RuntimeError("global parity filtering violated collision invariants")
    component_count = len(component_sizes)
    plane_count = _component_plane_counts(component, cell, component_count)
    linked_component = component_sizes >= 2
    linked_node = component_sizes[component] >= 2
    pair_code = np.stack((component.astype(np.int64), cell_code), axis=1)
    cell_collision_count = len(pair_code) - len(np.unique(pair_code, axis=0))
    top_components = []
    for component_index in np.argsort(component_sizes)[::-1]:
        size = int(component_sizes[component_index])
        if size < 2 or len(top_components) >= 20:
            break
        member = component == component_index
        member_cell = cell[member].astype(np.int32)
        top_components.append(
            {
                "componentId": int(component_index),
                "flakeCount": size,
                "cellCount": len(np.unique(member_cell, axis=0)),
                "axialPlaneCount": int(plane_count[component_index]),
                "originCellXYZ": np.min(member_cell, axis=0).astype(int).tolist(),
                "stopCellXYZExclusive": (
                    np.max(member_cell, axis=0) + 1
                ).astype(int).tolist(),
            }
        )

    _atomic_npz(
        artifact_path,
        nodeIdentity=node_identity,
        nodeCellIndex=cell,
        component=component.astype(np.uint32),
        componentSize=component_sizes[component].astype(np.uint32),
        componentPlaneCount=plane_count[component],
        degree=degree,
        branchParity=branch_parity,
        edgeIdentity=consensus_pair,
        edgeSourceNodeIndex=source,
        edgeTargetNodeIndex=target,
        edgeMinimumScore=minimum_score,
        edgeMaximumScore=maximum_score,
        edgeScoreObservationCount=score_observation_count,
        edgeRelativeParity=consensus_parity,
        collisionRetained=collision_retained,
        parityFrustrated=parity_frustrated,
        retained=retained,
    )
    result = {
        "identity": identity,
        "contract": {
            "scope": (
                "one sparse whole-volume graph over unanimous tiled primary-family "
                "matches; no dense surface, page, winding, side, or sheet identity"
            ),
            "directions": (
                "normals remain unsigned; edge parity is a relative gauge coordinate "
                "and parity-frustrated cycle edges are explicitly excluded"
            ),
            "collision": (
                "greedy descending-score connectivity retains at most one flake per "
                "Acus cell in every component"
            ),
            "inputRule": (
                "only raw edges and relative parities accepted by every observing "
                "window enter the global solve"
            ),
        },
        "stats": {
            "elapsedMs": round((time.monotonic() - started) * 1000.0, 2),
            "cacheHit": False,
            "nodeCount": len(node_identity),
            "rawTiledEdgeCount": len(raw_pair),
            "unanimousInputEdgeCount": len(consensus_pair),
            "nonunanimousInputEdgeCount": int(np.count_nonzero(~selected_consensus)),
            "edgeScoreObservationCount": _quantiles(score_observation_count),
            "edgeScoreRange": _quantiles(maximum_score - minimum_score, 7),
            "collisionRetainedEdgeCount": int(
                np.count_nonzero(collision_retained)
            ),
            "collisionRejectedEdgeCount": int(
                len(collision_retained) - np.count_nonzero(collision_retained)
            ),
            "parityFrustratedEdgeCount": int(
                np.count_nonzero(parity_frustrated)
            ),
            "retainedEdgeCount": int(np.count_nonzero(retained)),
            "componentCount": component_count,
            "linkedComponentCount": int(np.count_nonzero(linked_component)),
            "linkedNodeCount": int(np.count_nonzero(linked_node)),
            "largestComponentSize": int(np.max(component_sizes, initial=0)),
            "multiPlaneComponentCount": int(
                np.count_nonzero(linked_component & (plane_count >= 2))
            ),
            "allAxialPlaneComponentCount": int(
                np.count_nonzero(linked_component & (plane_count == grid_shape_xyz[2]))
            ),
            "longSpanComponentCount": int(
                np.count_nonzero(linked_component & (plane_count >= 11))
            ),
            "cellCollisionCount": cell_collision_count,
            "parity": parity_stats,
            "topComponents": top_components,
        },
        "artifact": _content_identity(artifact_path),
    }
    _atomic_json(summary_path, result)
    return result
