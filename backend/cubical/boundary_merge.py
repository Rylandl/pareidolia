from __future__ import annotations

import json
import math
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .contracts import atomic_json, canonical_json_hash, sha256_file
from .geometry import ClippedPatch, clip_plane_to_cell
from .matching import TraceMatch, TraceMatchSettings, align_face_patches
from .tables import PatchTable, read_patch_shard
from .topology import GridEdge, GridFace, GridSpec, Int3


BOUNDARY_MERGE_SCHEMA = "pareidolia.cubical-boundary-band-merge"
BOUNDARY_MERGE_VERSION = 1


Feature = GridEdge | Int3
ComponentNode = tuple[int, int]
CrossingNode = tuple[int, int]


@dataclass(slots=True)
class _BoundaryInput:
    side: int
    root: Path
    manifest: dict[str, Any]
    patches: PatchTable
    interface: dict[str, np.ndarray]
    patch_component: dict[int, int]
    component_cells: dict[int, set[Int3]]
    crossing_features: dict[int, Feature]


@dataclass(frozen=True, slots=True)
class _Adjacency:
    axis: int
    lower_side: int
    upper_side: int
    combined_grid: GridSpec
    offsets: tuple[Int3, Int3]


@dataclass(frozen=True, slots=True)
class _PatchReference:
    side: int
    source_patch_id: int
    source_component_id: int


class _ComponentUnion:
    def __init__(
        self,
        occupied_cells: dict[ComponentNode, set[Int3]],
    ) -> None:
        self.parent = {value: value for value in occupied_cells}
        self.cells = {
            value: set(cells) for value, cells in occupied_cells.items()
        }

    def find(self, value: ComponentNode) -> ComponentNode:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def collides(self, first: ComponentNode, second: ComponentNode) -> bool:
        first_root = self.find(first)
        second_root = self.find(second)
        return bool(self.cells[first_root] & self.cells[second_root])

    def union(self, first: ComponentNode, second: ComponentNode) -> None:
        first_root = self.find(first)
        second_root = self.find(second)
        if first_root == second_root:
            return
        if repr(first_root) > repr(second_root):
            first_root, second_root = second_root, first_root
        self.parent[second_root] = first_root
        self.cells[first_root].update(self.cells.pop(second_root))


def _combine_features(first: Feature, second: Feature) -> Feature | None:
    if isinstance(first, GridEdge) and isinstance(second, GridEdge):
        if first == second:
            return first
        shared = set(first.endpoint_vertices()) & set(second.endpoint_vertices())
        return next(iter(shared)) if len(shared) == 1 else None
    if isinstance(first, GridEdge):
        return second if second in first.endpoint_vertices() else None
    if isinstance(second, GridEdge):
        return first if first in second.endpoint_vertices() else None
    return first if first == second else None


class _CrossingUnion:
    def __init__(self, features: dict[CrossingNode, Feature]) -> None:
        self.parent = {value: value for value in features}
        self.feature = dict(features)

    def find(self, value: CrossingNode) -> CrossingNode:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def feasible(
        self, pairs: Iterable[tuple[CrossingNode, CrossingNode]]
    ) -> bool:
        roots = {
            self.find(value)
            for pair in pairs
            for value in pair
        }
        local_parent = {value: value for value in roots}

        def local_find(value: CrossingNode) -> CrossingNode:
            while local_parent[value] != value:
                local_parent[value] = local_parent[local_parent[value]]
                value = local_parent[value]
            return value

        for first, second in pairs:
            first_root = local_find(self.find(first))
            second_root = local_find(self.find(second))
            if first_root != second_root:
                local_parent[second_root] = first_root
        groups: dict[CrossingNode, list[Feature]] = defaultdict(list)
        for root in roots:
            groups[local_find(root)].append(self.feature[root])
        for features in groups.values():
            combined = features[0]
            for feature in features[1:]:
                result = _combine_features(combined, feature)
                if result is None:
                    return False
                combined = result
        return True

    def union_pairs(
        self, pairs: Iterable[tuple[CrossingNode, CrossingNode]]
    ) -> None:
        pairs = tuple(pairs)
        if not self.feasible(pairs):
            raise ValueError("crossing union is infeasible")
        for first, second in pairs:
            first_root = self.find(first)
            second_root = self.find(second)
            if first_root == second_root:
                continue
            combined = _combine_features(
                self.feature[first_root], self.feature[second_root]
            )
            if combined is None:
                raise RuntimeError("validated crossing union became infeasible")
            if repr(first_root) > repr(second_root):
                first_root, second_root = second_root, first_root
            self.parent[second_root] = first_root
            self.feature[first_root] = combined
            del self.feature[second_root]


def _load_boundary(root: str | Path, side: int) -> _BoundaryInput:
    resolved = Path(root).resolve()
    manifest_path = resolved / "boundary-band-v1.json"
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != "pareidolia.cubical-boundary-band"
        or int(manifest.get("version", -1)) != 1
        or manifest.get("state") != "complete"
    ):
        raise ValueError("boundary input is not a completed version-1 artifact")
    interface_record = manifest["artifacts"]["interface"]
    interface_path = resolved / interface_record["path"]
    if sha256_file(interface_path) != interface_record["sha256"]:
        raise ValueError("boundary interface content hash mismatch")
    patches = read_patch_shard(resolved / "boundary-patches-v1", verify=True)
    with np.load(interface_path) as values:
        interface = {name: np.asarray(values[name]) for name in values.files}
    patch_ids = np.asarray(interface["boundaryPatchId"], dtype=np.uint64)
    if not np.array_equal(patch_ids, patches.patch_id):
        raise ValueError("boundary patch and component tables disagree")
    patch_component = {
        int(patch_id): int(component_id)
        for patch_id, component_id in zip(
            patch_ids, interface["boundaryPatchComponentId"]
        )
    }
    component_ids = np.asarray(interface["componentId"], dtype=np.uint64)
    offsets = np.asarray(interface["componentCellOffset"], dtype=np.uint64)
    cells = np.asarray(interface["componentCellXYZ"], dtype=np.int32)
    if (
        len(offsets) != len(component_ids) + 1
        or int(offsets[0]) != 0
        or int(offsets[-1]) != len(cells)
        or np.any(np.diff(offsets.astype(np.int64)) < 0)
    ):
        raise ValueError("boundary component occupancy offsets are invalid")
    component_cells = {
        int(component_id): {
            tuple(int(value) for value in cell)
            for cell in cells[int(offsets[index]) : int(offsets[index + 1])]
        }
        for index, component_id in enumerate(component_ids)
    }
    crossing_features: dict[int, Feature] = {}
    for group_id, kind, edge_axis, anchor in zip(
        interface["crossingGroupId"],
        interface["crossingGroupFeatureKind"],
        interface["crossingGroupFeatureEdgeAxis"],
        interface["crossingGroupFeatureAnchorXYZ"],
    ):
        anchor_xyz = tuple(int(value) for value in anchor)
        if int(kind) == 0:
            crossing_features[int(group_id)] = GridEdge(int(edge_axis), anchor_xyz)
        elif int(kind) == 1:
            crossing_features[int(group_id)] = anchor_xyz
        else:
            raise ValueError("boundary crossing feature kind is unsupported")
    trace_groups = set(
        int(value)
        for name in (
            "traceFirstCrossingGroupId",
            "traceSecondCrossingGroupId",
        )
        for value in interface[name]
    )
    if not trace_groups <= set(crossing_features):
        raise ValueError("boundary trace references an absent crossing group")
    return _BoundaryInput(
        side,
        resolved,
        manifest,
        patches,
        interface,
        patch_component,
        component_cells,
        crossing_features,
    )


def _adjacency(first: _BoundaryInput, second: _BoundaryInput) -> _Adjacency:
    first_grid = first.patches.grid
    second_grid = second.patches.grid
    if first_grid.coordinate_unit != second_grid.coordinate_unit:
        raise ValueError("boundary grids use different coordinate units")
    first_size = np.asarray(first_grid.cell_size_xyz, dtype=np.float64)
    second_size = np.asarray(second_grid.cell_size_xyz, dtype=np.float64)
    if not np.allclose(first_size, second_size, rtol=0.0, atol=1.0e-9):
        raise ValueError("boundary grids use different cell sizes")
    origins = (
        np.asarray(first_grid.origin_xyz, dtype=np.float64),
        np.asarray(second_grid.origin_xyz, dtype=np.float64),
    )
    stops = (
        origins[0]
        + first_size * np.asarray(first_grid.shape_cells_xyz, dtype=np.float64),
        origins[1]
        + first_size * np.asarray(second_grid.shape_cells_xyz, dtype=np.float64),
    )
    tolerance = max(float(np.max(first_size)) * 1.0e-7, 1.0e-8)
    candidates: list[tuple[int, int, int]] = []
    for axis in range(3):
        other_axes = [value for value in range(3) if value != axis]
        tangential_match = np.allclose(
            origins[0][other_axes],
            origins[1][other_axes],
            rtol=0.0,
            atol=tolerance,
        ) and np.allclose(
            stops[0][other_axes],
            stops[1][other_axes],
            rtol=0.0,
            atol=tolerance,
        )
        if not tangential_match:
            continue
        if abs(float(stops[0][axis] - origins[1][axis])) <= tolerance:
            candidates.append((axis, 0, 1))
        if abs(float(stops[1][axis] - origins[0][axis])) <= tolerance:
            candidates.append((axis, 1, 0))
    if len(candidates) != 1:
        raise ValueError("boundary blocks must tile one complete shared world face")
    axis, lower_side, upper_side = candidates[0]
    grids = (first_grid, second_grid)
    lower_grid = grids[lower_side]
    upper_grid = grids[upper_side]
    combined_shape = list(lower_grid.shape_cells_xyz)
    combined_shape[axis] += upper_grid.shape_cells_xyz[axis]
    combined_grid = GridSpec(
        tuple(combined_shape),
        lower_grid.cell_size_xyz,
        lower_grid.origin_xyz,
        lower_grid.coordinate_unit,
    )
    offsets: list[Int3] = []
    combined_origin = np.asarray(combined_grid.origin_xyz, dtype=np.float64)
    for grid in grids:
        raw = (
            np.asarray(grid.origin_xyz, dtype=np.float64) - combined_origin
        ) / first_size
        rounded = np.rint(raw).astype(np.int64)
        if not np.allclose(raw, rounded, rtol=0.0, atol=1.0e-7):
            raise ValueError("boundary grid origin is not cell aligned")
        offsets.append(tuple(int(value) for value in rounded))
    return _Adjacency(
        axis,
        lower_side,
        upper_side,
        combined_grid,
        (offsets[0], offsets[1]),
    )


def _offset_edge(edge: GridEdge, offset: Int3) -> GridEdge:
    return GridEdge(
        edge.axis,
        tuple(edge.anchor_xyz[axis] + offset[axis] for axis in range(3)),
    )


def _offset_feature(feature: Feature, offset: Int3) -> Feature:
    if isinstance(feature, GridEdge):
        return _offset_edge(feature, offset)
    return tuple(feature[axis] + offset[axis] for axis in range(3))


def _touching_trace_rows(
    boundary: _BoundaryInput,
    *,
    axis: int,
    side: int,
) -> np.ndarray:
    return np.flatnonzero(
        (boundary.interface["traceFaceAxis"] == axis)
        & (boundary.interface["traceFaceSide"] == side)
    )


def _rebase_touching_patches(
    boundary: _BoundaryInput,
    adjacency: _Adjacency,
    touching_side: int,
    next_patch_id: int,
) -> tuple[
    tuple[ClippedPatch, ...],
    dict[int, _PatchReference],
    dict[tuple[int, GridEdge], CrossingNode],
    dict[CrossingNode, Feature],
    int,
]:
    rows = _touching_trace_rows(
        boundary,
        axis=adjacency.axis,
        side=touching_side,
    )
    source_patch_ids = sorted(
        {int(boundary.interface["tracePatchId"][row]) for row in rows}
    )
    source_by_id = {
        patch.patch_id: patch for patch in boundary.patches.to_patches()
    }
    offset = adjacency.offsets[boundary.side]
    rebased: list[ClippedPatch] = []
    references: dict[int, _PatchReference] = {}
    temporary_by_source: dict[int, int] = {}
    for source_patch_id in source_patch_ids:
        source = source_by_id[source_patch_id]
        cell = tuple(
            source.cell_xyz[axis] + offset[axis] for axis in range(3)
        )
        temporary_id = next_patch_id
        next_patch_id += 1
        patch = clip_plane_to_cell(
            adjacency.combined_grid,
            cell,
            source.estimate,
            patch_id=temporary_id,
        )
        if patch is None:
            raise ValueError("rebased boundary plane no longer intersects its cell")
        rebased.append(patch)
        temporary_by_source[source_patch_id] = temporary_id
        references[temporary_id] = _PatchReference(
            boundary.side,
            source_patch_id,
            boundary.patch_component[source_patch_id],
        )
    endpoint_groups: dict[tuple[int, GridEdge], CrossingNode] = {}
    referenced_groups: set[int] = set()
    for row in rows:
        source_patch_id = int(boundary.interface["tracePatchId"][row])
        temporary_id = temporary_by_source[source_patch_id]
        for prefix in ("First", "Second"):
            source_edge = GridEdge(
                int(boundary.interface[f"trace{prefix}EdgeAxis"][row]),
                tuple(
                    int(value)
                    for value in boundary.interface[
                        f"trace{prefix}EdgeAnchorXYZ"
                    ][row]
                ),
            )
            group_id = int(
                boundary.interface[f"trace{prefix}CrossingGroupId"][row]
            )
            endpoint_groups[(temporary_id, _offset_edge(source_edge, offset))] = (
                boundary.side,
                group_id,
            )
            referenced_groups.add(group_id)
    features = {
        (boundary.side, group_id): _offset_feature(
            boundary.crossing_features[group_id], offset
        )
        for group_id in referenced_groups
    }
    return (
        tuple(rebased),
        references,
        endpoint_groups,
        features,
        next_patch_id,
    )


def _ordered_packet_alignment(
    first_patches: Iterable[ClippedPatch],
    second_patches: Iterable[ClippedPatch],
    face: GridFace,
    policy: dict[str, Any],
    grid: GridSpec,
) -> tuple[
    tuple[TraceMatch, ...],
    tuple[int, ...],
    tuple[int, ...],
    dict[int, int],
    dict[int, int],
]:
    first_values = tuple(first_patches)
    second_values = tuple(second_patches)
    parallel_policy = policy["parallelMatching"]
    strict = align_face_patches(
        first_values,
        second_values,
        face,
        TraceMatchSettings(
            orthogonal_fiber_equivalence=bool(
                parallel_policy["orthogonalFiberEquivalence"]
            ),
            maximum_absolute_normal_angle_radians=math.radians(
                float(parallel_policy["maximumNormalAngleDegrees"])
            ),
            maximum_absolute_fiber_residual_radians=math.radians(
                float(parallel_policy["maximumFiberResidualDegrees"])
            ),
        ),
        grid=grid,
    )
    quarter_policy = policy["quarterTurnAdmission"]
    proposal = (
        align_face_patches(
            first_values,
            second_values,
            face,
            TraceMatchSettings(orthogonal_fiber_equivalence=True),
            grid=grid,
        )
        if bool(quarter_policy["enabled"])
        else None
    )
    order_axis = strict.order_axis_xyz or (
        None if proposal is None else proposal.order_axis_xyz
    )
    if order_axis is None:
        return ((), (), (), {}, {})
    axis = np.asarray(order_axis, dtype=np.float64)

    def ranks(values: tuple[ClippedPatch, ...]) -> dict[int, int]:
        ordered = sorted(
            (
                (
                    float(np.dot(trace.midpoint_xyz, axis)),
                    patch.patch_id,
                )
                for patch in values
                if (trace := patch.trace_on(face)) is not None
            )
        )
        return {patch_id: index for index, (_, patch_id) in enumerate(ordered)}

    first_rank = ranks(first_values)
    second_rank = ranks(second_values)
    chosen = list(strict.matches)
    if proposal is not None:
        strict_keys = {
            (value.first_patch_id, value.second_patch_id) for value in chosen
        }
        normal_cap = float(quarter_policy["maximumNormalAngleDegrees"])
        fiber_cap = float(quarter_policy["maximumFiberFrameResidualDegrees"])
        chosen.extend(
            candidate
            for candidate in proposal.matches
            if candidate.fiber_quarter_turn is True
            and (candidate.first_patch_id, candidate.second_patch_id)
            not in strict_keys
            and math.degrees(candidate.normal_angle_radians) <= normal_cap
            and candidate.fiber_angle_radians is not None
            and math.degrees(candidate.fiber_angle_radians) <= fiber_cap
        )
    chosen.sort(
        key=lambda value: (
            first_rank[value.first_patch_id],
            second_rank[value.second_patch_id],
        )
    )
    matched_first = {value.first_patch_id for value in chosen}
    matched_second = {value.second_patch_id for value in chosen}
    return (
        tuple(chosen),
        tuple(sorted(set(first_rank) - matched_first)),
        tuple(sorted(set(second_rank) - matched_second)),
        first_rank,
        second_rank,
    )


def _match_crossing_pairs(
    match: TraceMatch,
    endpoint_groups: dict[tuple[int, GridEdge], CrossingNode],
) -> tuple[tuple[CrossingNode, CrossingNode], ...]:
    try:
        return tuple(
            (
                endpoint_groups[(match.first_patch_id, value.first_edge)],
                endpoint_groups[(match.second_patch_id, value.second_edge)],
            )
            for value in match.endpoint_agreements
        )
    except KeyError as error:
        raise ValueError("seam match endpoint lacks a crossing certificate") from error


def _quantiles(values: list[float]) -> dict[str, float | None]:
    if not values:
        return {"median": None, "p90": None, "maximum": None}
    result = np.percentile(np.asarray(values, dtype=np.float64), (50, 90, 100))
    return {
        name: round(float(value), 7)
        for name, value in zip(("median", "p90", "maximum"), result)
    }


def _identity(
    first: _BoundaryInput,
    second: _BoundaryInput,
    adjacency: _Adjacency,
) -> dict[str, Any]:
    module_root = Path(__file__).resolve().parent
    payload: dict[str, Any] = {
        "schema": BOUNDARY_MERGE_SCHEMA,
        "version": BOUNDARY_MERGE_VERSION,
        "inputs": [
            {
                "root": str(value.root),
                "manifestSha256": sha256_file(
                    value.root / "boundary-band-v1.json"
                ),
                "boundaryIdentitySha256": value.manifest["identity"][
                    "identitySha256"
                ],
            }
            for value in (first, second)
        ],
        "adjacency": {
            "axis": adjacency.axis,
            "lowerInput": adjacency.lower_side,
            "upperInput": adjacency.upper_side,
            "offsetsCellsXYZ": [list(value) for value in adjacency.offsets],
        },
        "seamMatchingPolicy": first.manifest["seamMatchingPolicy"],
        "implementationSha256": {
            name: sha256_file(module_root / name)
            for name in (
                "boundary_merge.py",
                "boundary_band.py",
                "matching.py",
                "geometry.py",
                "tables.py",
                "contracts.py",
            )
        },
    }
    payload["identitySha256"] = canonical_json_hash(payload)
    return payload


def run_boundary_band_merge(
    first_root: str | Path,
    second_root: str | Path,
    output_root: str | Path,
    *,
    force: bool = False,
) -> dict[str, Any]:
    """Match adjacent boundary bands while treating both interiors as immutable.

    The retained component bridges form a collision-safe, crossing-safe forest.
    Redundant face matches remain serialized as evidence but are not needed to
    establish connectivity, so no orientation cycle can be introduced here.
    """

    started = time.monotonic()
    first = _load_boundary(first_root, 0)
    second = _load_boundary(second_root, 1)
    if first.root == second.root:
        raise ValueError("boundary merge requires two distinct inputs")
    if first.manifest["graphSemantics"] != second.manifest["graphSemantics"]:
        raise ValueError("boundary inputs use different graph semantics")
    if first.manifest["seamMatchingPolicy"] != second.manifest["seamMatchingPolicy"]:
        raise ValueError("boundary inputs use different seam matching policies")
    adjacency = _adjacency(first, second)
    output = Path(output_root).resolve()
    if output in (first.root, second.root):
        raise ValueError("boundary merge output must differ from both inputs")
    identity = _identity(first, second, adjacency)
    manifest_path = output / "boundary-merge-v1.json"
    if manifest_path.is_file():
        prior = json.loads(manifest_path.read_text())
        if prior.get("identity", {}).get("identitySha256") != identity[
            "identitySha256"
        ]:
            raise ValueError("boundary merge output belongs to another identity")
        if not force and prior.get("state") == "complete":
            return prior
    output.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, Any] = {
        "schema": BOUNDARY_MERGE_SCHEMA,
        "version": BOUNDARY_MERGE_VERSION,
        "state": "building",
        "identity": identity,
    }
    atomic_json(manifest_path, manifest)

    boundaries = (first, second)
    lower = boundaries[adjacency.lower_side]
    upper = boundaries[adjacency.upper_side]
    lower_values = _rebase_touching_patches(
        lower,
        adjacency,
        1,
        1,
    )
    upper_values = _rebase_touching_patches(
        upper,
        adjacency,
        0,
        lower_values[4],
    )
    lower_patches, lower_refs, lower_endpoints, lower_features, _ = lower_values
    upper_patches, upper_refs, upper_endpoints, upper_features, _ = upper_values
    references = {**lower_refs, **upper_refs}
    endpoint_groups = {**lower_endpoints, **upper_endpoints}
    crossing_union = _CrossingUnion({**lower_features, **upper_features})

    lower_by_face: dict[GridFace, list[ClippedPatch]] = defaultdict(list)
    upper_by_face: dict[GridFace, list[ClippedPatch]] = defaultdict(list)
    seam_coordinate = (
        adjacency.offsets[lower.side][adjacency.axis]
        + lower.patches.grid.shape_cells_xyz[adjacency.axis]
    )
    for patch in lower_patches:
        for trace in patch.traces:
            if (
                trace.face.axis == adjacency.axis
                and trace.face.anchor_xyz[adjacency.axis] == seam_coordinate
            ):
                lower_by_face[trace.face].append(patch)
    for patch in upper_patches:
        for trace in patch.traces:
            if (
                trace.face.axis == adjacency.axis
                and trace.face.anchor_xyz[adjacency.axis] == seam_coordinate
            ):
                upper_by_face[trace.face].append(patch)
    matches: list[TraceMatch] = []
    unmatched: list[tuple[int, int, Int3]] = []
    match_ranks: dict[tuple[int, int, int, Int3], tuple[int, int]] = {}
    for face in sorted(set(lower_by_face) | set(upper_by_face)):
        (
            face_matches,
            unmatched_first,
            unmatched_second,
            first_rank,
            second_rank,
        ) = _ordered_packet_alignment(
            lower_by_face.get(face, ()),
            upper_by_face.get(face, ()),
            face,
            first.manifest["seamMatchingPolicy"],
            adjacency.combined_grid,
        )
        matches.extend(face_matches)
        for value in face_matches:
            match_ranks[
                (
                    value.first_patch_id,
                    value.second_patch_id,
                    value.face.axis,
                    value.face.anchor_xyz,
                )
            ] = (
                first_rank[value.first_patch_id],
                second_rank[value.second_patch_id],
            )
        unmatched.extend(
            (value, adjacency.axis, face.anchor_xyz)
            for value in unmatched_first
        )
        unmatched.extend(
            (value, adjacency.axis, face.anchor_xyz)
            for value in unmatched_second
        )

    matches_by_components: dict[
        tuple[ComponentNode, ComponentNode], list[TraceMatch]
    ] = defaultdict(list)
    for match in matches:
        first_ref = references[match.first_patch_id]
        second_ref = references[match.second_patch_id]
        first_node = (first_ref.side, first_ref.source_component_id)
        second_node = (second_ref.side, second_ref.source_component_id)
        matches_by_components[(first_node, second_node)].append(match)

    occupied_cells: dict[ComponentNode, set[Int3]] = {}
    for boundary in boundaries:
        offset = adjacency.offsets[boundary.side]
        for component_id, cells in boundary.component_cells.items():
            occupied_cells[(boundary.side, component_id)] = {
                tuple(cell[axis] + offset[axis] for axis in range(3))
                for cell in cells
            }
    component_union = _ComponentUnion(occupied_cells)
    pair_order = sorted(
        matches_by_components,
        key=lambda pair: (
            0
            if any(
                value.fiber_quarter_turn is not True
                for value in matches_by_components[pair]
            )
            else 1,
            -len(matches_by_components[pair]),
            float(
                np.mean(
                    [
                        value.negative_log_likelihood
                        for value in matches_by_components[pair]
                    ]
                )
            ),
            pair,
        ),
    )
    selected_matches: set[tuple[int, int, int, Int3]] = set()
    used_seam_patches: set[int] = set()
    selected_face_ranks: dict[GridFace, list[tuple[int, int]]] = defaultdict(list)
    pair_disposition: dict[
        tuple[ComponentNode, ComponentNode], tuple[str, TraceMatch | None]
    ] = {}
    for first_node, second_node in pair_order:
        pair = (first_node, second_node)
        if component_union.find(first_node) == component_union.find(second_node):
            pair_disposition[pair] = ("redundant-component-cycle", None)
            continue
        if component_union.collides(first_node, second_node):
            pair_disposition[pair] = ("component-cell-collision", None)
            continue
        candidates = sorted(
            matches_by_components[pair],
            key=lambda value: (
                value.fiber_quarter_turn is True,
                -value.score,
                value.negative_log_likelihood,
                value.face.anchor_xyz,
                value.first_patch_id,
                value.second_patch_id,
            ),
        )
        trace_feasible = []
        for value in candidates:
            if (
                value.first_patch_id in used_seam_patches
                or value.second_patch_id in used_seam_patches
            ):
                continue
            key = (
                value.first_patch_id,
                value.second_patch_id,
                value.face.axis,
                value.face.anchor_xyz,
            )
            first_rank, second_rank = match_ranks[key]
            if any(
                (first_rank - retained_first)
                * (second_rank - retained_second)
                < 0
                for retained_first, retained_second in selected_face_ranks[
                    value.face
                ]
            ):
                continue
            trace_feasible.append(value)
        if not trace_feasible:
            pair_disposition[pair] = ("seam-trace-conflict", None)
            continue
        representative = next(
            (
                value
                for value in trace_feasible
                if crossing_union.feasible(
                    _match_crossing_pairs(value, endpoint_groups)
                )
            ),
            None,
        )
        if representative is None:
            pair_disposition[pair] = ("crossing-topology-cycle", None)
            continue
        component_union.union(first_node, second_node)
        crossing_union.union_pairs(
            _match_crossing_pairs(representative, endpoint_groups)
        )
        pair_disposition[pair] = ("retained-forest-bridge", representative)
        selected_key = (
            representative.first_patch_id,
            representative.second_patch_id,
            representative.face.axis,
            representative.face.anchor_xyz,
        )
        selected_matches.add(selected_key)
        used_seam_patches.update(
            (representative.first_patch_id, representative.second_patch_id)
        )
        selected_face_ranks[representative.face].append(
            match_ranks[selected_key]
        )

    artifact_path = output / "boundary-merge-v1.npz"
    temporary = artifact_path.with_suffix(artifact_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        match_first_refs = [references[value.first_patch_id] for value in matches]
        match_second_refs = [references[value.second_patch_id] for value in matches]
        pair_records = list(pair_order)
        np.savez_compressed(
            handle,
            matchFirstInput=np.asarray(
                [value.side for value in match_first_refs], dtype=np.int8
            ),
            matchFirstPatchId=np.asarray(
                [value.source_patch_id for value in match_first_refs],
                dtype=np.uint64,
            ),
            matchFirstComponentId=np.asarray(
                [value.source_component_id for value in match_first_refs],
                dtype=np.uint64,
            ),
            matchSecondInput=np.asarray(
                [value.side for value in match_second_refs], dtype=np.int8
            ),
            matchSecondPatchId=np.asarray(
                [value.source_patch_id for value in match_second_refs],
                dtype=np.uint64,
            ),
            matchSecondComponentId=np.asarray(
                [value.source_component_id for value in match_second_refs],
                dtype=np.uint64,
            ),
            matchFaceAxis=np.asarray(
                [value.face.axis for value in matches], dtype=np.int8
            ),
            matchFaceAnchorXYZ=np.asarray(
                [value.face.anchor_xyz for value in matches], dtype=np.int32
            ).reshape(len(matches), 3),
            matchScore=np.asarray(
                [value.score for value in matches], dtype=np.float32
            ),
            matchNegativeLogLikelihood=np.asarray(
                [value.negative_log_likelihood for value in matches],
                dtype=np.float32,
            ),
            matchMaximumEndpointZ=np.asarray(
                [
                    max(
                        (endpoint.z for endpoint in value.endpoint_agreements),
                        default=np.nan,
                    )
                    for value in matches
                ],
                dtype=np.float32,
            ),
            matchNormalResidualDegrees=np.asarray(
                [math.degrees(value.normal_angle_radians) for value in matches],
                dtype=np.float32,
            ),
            matchFiberFrameResidualDegrees=np.asarray(
                [
                    math.degrees(value.fiber_angle_radians)
                    if value.fiber_angle_radians is not None
                    else np.nan
                    for value in matches
                ],
                dtype=np.float32,
            ),
            matchFiberQuarterTurn=np.asarray(
                [
                    -1
                    if value.fiber_quarter_turn is None
                    else int(value.fiber_quarter_turn)
                    for value in matches
                ],
                dtype=np.int8,
            ),
            matchSelectedBridge=np.asarray(
                [
                    (
                        value.first_patch_id,
                        value.second_patch_id,
                        value.face.axis,
                        value.face.anchor_xyz,
                    )
                    in selected_matches
                    for value in matches
                ],
                dtype=np.uint8,
            ),
            pairFirstInput=np.asarray(
                [value[0][0] for value in pair_records], dtype=np.int8
            ),
            pairFirstComponentId=np.asarray(
                [value[0][1] for value in pair_records], dtype=np.uint64
            ),
            pairSecondInput=np.asarray(
                [value[1][0] for value in pair_records], dtype=np.int8
            ),
            pairSecondComponentId=np.asarray(
                [value[1][1] for value in pair_records], dtype=np.uint64
            ),
            pairSupportMatchCount=np.asarray(
                [len(matches_by_components[value]) for value in pair_records],
                dtype=np.uint32,
            ),
            pairDisposition=np.asarray(
                [pair_disposition[value][0] for value in pair_records],
                dtype="U32",
            ),
            unmatchedInput=np.asarray(
                [references[value[0]].side for value in unmatched], dtype=np.int8
            ),
            unmatchedPatchId=np.asarray(
                [references[value[0]].source_patch_id for value in unmatched],
                dtype=np.uint64,
            ),
            unmatchedFaceAxis=np.asarray(
                [value[1] for value in unmatched], dtype=np.int8
            ),
            unmatchedFaceAnchorXYZ=np.asarray(
                [value[2] for value in unmatched], dtype=np.int32
            ).reshape(len(unmatched), 3),
        )
    temporary.replace(artifact_path)

    deferred = Counter(value[0] for value in pair_disposition.values())
    accepted_matches = list(matches)
    combined_grid = adjacency.combined_grid
    manifest.update(
        {
            "state": "complete",
            "graphSemantics": first.manifest["graphSemantics"],
            "method": {
                "interiorsImmutable": True,
                "matching": (
                    "order-preserving selected-trace alignment on each shared "
                    "unit face using the serialized seam policy"
                ),
                "selection": (
                    "one exact representative per component-pair, ranked by "
                    "multi-face support, retained as a cell-collision-safe and "
                    "crossing-feature-safe component forest"
                ),
                "orientationGuarantee": (
                    "the retained component graph is acyclic, so every bridge "
                    "admits a consistent relative orientation gauge"
                ),
                "redundantMatches": (
                    "serialized as corroborating evidence, not topology edges"
                ),
            },
            "grid": {
                "shapeCellsXYZ": list(combined_grid.shape_cells_xyz),
                "cellSizeXYZ": list(combined_grid.cell_size_xyz),
                "originXYZ": list(combined_grid.origin_xyz),
                "coordinateUnit": combined_grid.coordinate_unit,
            },
            "adjacency": {
                "axis": adjacency.axis,
                "lowerInput": adjacency.lower_side,
                "upperInput": adjacency.upper_side,
                "offsetsCellsXYZ": [list(value) for value in adjacency.offsets],
            },
            "statistics": {
                "seamUnitFaces": int(
                    np.prod(
                        [
                            combined_grid.shape_cells_xyz[axis]
                            for axis in range(3)
                            if axis != adjacency.axis
                        ]
                    )
                ),
                "lowerTouchingPatches": len(lower_patches),
                "upperTouchingPatches": len(upper_patches),
                "alignedMatches": len(matches),
                "unmatchedTraces": len(unmatched),
                "componentPairHypotheses": len(pair_order),
                "retainedComponentBridges": deferred[
                    "retained-forest-bridge"
                ],
                "componentPairDisposition": dict(sorted(deferred.items())),
                "quarterTurnMatches": sum(
                    value.fiber_quarter_turn is True for value in matches
                ),
                "matchResiduals": {
                    "endpointZ": _quantiles(
                        [
                            max(
                                endpoint.z
                                for endpoint in value.endpoint_agreements
                            )
                            for value in accepted_matches
                        ]
                    ),
                    "normalDegrees": _quantiles(
                        [
                            math.degrees(value.normal_angle_radians)
                            for value in accepted_matches
                        ]
                    ),
                    "fiberFrameDegrees": _quantiles(
                        [
                            math.degrees(value.fiber_angle_radians)
                            for value in accepted_matches
                            if value.fiber_angle_radians is not None
                        ]
                    ),
                },
            },
            "timingSeconds": {
                "total": round(time.monotonic() - started, 6)
            },
            "artifact": {
                "path": artifact_path.name,
                "bytes": artifact_path.stat().st_size,
                "sha256": sha256_file(artifact_path),
            },
        }
    )
    atomic_json(manifest_path, manifest)
    return manifest
