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

from .contracts import (
    VolumeSource,
    VoxelBounds,
    atomic_json,
    canonical_json_hash,
    sha256_file,
)
from .export import rgb_png
from .isolated_slab import _percentile_record
from .one_sided_interface import (
    ONE_SIDED_INTERFACE_SCHEMA,
    ONE_SIDED_INTERFACE_STEM,
)


ONE_SIDED_GROWTH_SCHEMA = "pareidolia.one-sided-interface-contextual-growth"
ONE_SIDED_GROWTH_VERSION = 1
ONE_SIDED_GROWTH_STEM = "one-sided-interface-growth-v1"


@dataclass(frozen=True, slots=True)
class OneSidedGrowthSettings:
    minimum_candidate_evidence: float = 0.35
    minimum_growth_bottleneck: float = 0.35
    link_radius_sampling_steps: float = 1.0
    maximum_position_distance_sampling_steps: float = 3.0
    maximum_signed_normal_degrees: float = 30.0
    maximum_height_sampling_steps: float = 1.15
    affinity_normal_sigma_degrees: float = 15.0
    affinity_height_sigma_steps: float = 0.75
    defer_contested_components: bool = True
    minimum_association_side_component_count: int = 2
    minimum_association_balanced_seed_support: int = 5
    minimum_association_seed_side_purity: float = 0.8
    minimum_association_orientation_purity: float = 0.8
    maximum_preview_labels: int = 128

    def __post_init__(self) -> None:
        for value, name in (
            (self.minimum_candidate_evidence, "minimum candidate evidence"),
            (self.minimum_growth_bottleneck, "minimum growth bottleneck"),
            (
                self.minimum_association_seed_side_purity,
                "minimum association seed-side purity",
            ),
            (
                self.minimum_association_orientation_purity,
                "minimum association orientation purity",
            ),
        ):
            if not 0.0 < value <= 1.0:
                raise ValueError(f"{name} must lie in (0, 1]")
        positive = (
            self.link_radius_sampling_steps,
            self.maximum_position_distance_sampling_steps,
            self.maximum_signed_normal_degrees,
            self.maximum_height_sampling_steps,
            self.affinity_normal_sigma_degrees,
            self.affinity_height_sigma_steps,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("one-sided growth scales must be finite and positive")
        if not 0.0 < self.maximum_signed_normal_degrees < 90.0:
            raise ValueError("signed interface normal cap must lie in (0, 90)")
        if self.minimum_candidate_evidence > self.minimum_growth_bottleneck:
            raise ValueError(
                "candidate evidence gate cannot exceed the growth bottleneck"
            )
        if self.maximum_preview_labels < 1:
            raise ValueError("one-sided preview label count must be positive")
        if (
            self.minimum_association_side_component_count < 1
            or self.minimum_association_balanced_seed_support < 1
        ):
            raise ValueError("one-sided association support counts must be positive")

    def record(self) -> dict[str, Any]:
        return asdict(self)


def _load_interface_bank(
    root: str | Path,
) -> tuple[Path, dict[str, Any], dict[str, np.ndarray]]:
    value = Path(root).resolve()
    manifest_path = (
        value if value.is_file() else value / f"{ONE_SIDED_INTERFACE_STEM}.json"
    )
    manifest = json.loads(manifest_path.read_text())
    if (
        manifest.get("schema") != ONE_SIDED_INTERFACE_SCHEMA
        or manifest.get("state") != "complete"
    ):
        raise ValueError("one-sided growth requires a complete interface bank")
    data_path = manifest_path.parent / str(manifest["data"]["path"])
    if sha256_file(data_path) != manifest["data"]["sha256"]:
        raise ValueError("one-sided interface data hash differs from its manifest")
    with np.load(data_path) as values:
        arrays = {name: np.asarray(values[name]) for name in values.files}
    return manifest_path, manifest, arrays


def _continuity_metrics(
    first: np.ndarray,
    second: np.ndarray,
    bank: Mapping[str, np.ndarray],
    *,
    stride: int,
) -> dict[str, np.ndarray]:
    first_normal = np.asarray(bank["signedNormalXYZ"])[first].astype(np.float64)
    second_normal = np.asarray(bank["signedNormalXYZ"])[second].astype(np.float64)
    dot = np.einsum("ij,ij->i", first_normal, second_normal)
    average_normal = first_normal + second_normal
    average_normal /= np.maximum(
        np.linalg.norm(average_normal, axis=1, keepdims=True), 1.0e-9
    )
    delta = (
        np.asarray(bank["positionXYZ"])[second]
        - np.asarray(bank["positionXYZ"])[first]
    ).astype(np.float64)
    return {
        "positionDistanceSamplingSteps": (
            np.linalg.norm(delta, axis=1) / stride
        ).astype(np.float32),
        "signedNormalDegrees": np.degrees(
            np.arccos(np.clip(dot, -1.0, 1.0))
        ).astype(np.float32),
        "heightSamplingSteps": (
            np.abs(np.einsum("ij,ij->i", delta, average_normal)) / stride
        ).astype(np.float32),
    }


def build_one_sided_continuity_graph(
    bank: Mapping[str, np.ndarray],
    *,
    processing_shape_sampling_xyz: tuple[int, int, int],
    stride: int,
    settings: OneSidedGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Build signed local-continuation edges over one-sided interfaces."""

    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    seed = np.asarray(bank["seedSurfaceLabel"], dtype=np.int32) >= 0
    conflict = np.asarray(bank["seedConflict"]) > 0
    eligible = ((evidence >= settings.minimum_candidate_evidence) | seed) & ~conflict
    eligible_index = np.flatnonzero(eligible).astype(np.int32)
    key = np.asarray(bank["processingKeyXYZ"])[eligible_index].astype(np.int32)
    grid = np.full(processing_shape_sampling_xyz[::-1], -1, dtype=np.int32)
    grid[key[:, 2], key[:, 1], key[:, 0]] = eligible_index
    reach = int(math.ceil(settings.link_radius_sampling_steps))
    edge_first_parts: list[np.ndarray] = []
    edge_second_parts: list[np.ndarray] = []
    affinity_parts: list[np.ndarray] = []
    metric_parts: dict[str, list[np.ndarray]] = {}
    considered = 0
    gate_rejection = {"distance": 0, "normal": 0, "height": 0}
    for dz in range(-reach, reach + 1):
        for dy in range(-reach, reach + 1):
            for dx in range(-reach, reach + 1):
                if (dz, dy, dx) <= (0, 0, 0):
                    continue
                if (
                    dx * dx + dy * dy + dz * dz
                    > settings.link_radius_sampling_steps**2 + 1.0e-9
                ):
                    continue
                target_key = key + np.asarray((dx, dy, dz), dtype=np.int32)
                valid = np.all(
                    (target_key >= 0)
                    & (
                        target_key
                        < np.asarray(
                            processing_shape_sampling_xyz, dtype=np.int32
                        )[None, :]
                    ),
                    axis=1,
                )
                first = eligible_index[valid]
                target_key = target_key[valid]
                second = grid[
                    target_key[:, 2], target_key[:, 1], target_key[:, 0]
                ]
                exists = second >= 0
                first = first[exists]
                second = second[exists]
                if not len(first):
                    continue
                considered += len(first)
                metrics = _continuity_metrics(
                    first, second, bank, stride=stride
                )
                accepted = (
                    metrics["positionDistanceSamplingSteps"]
                    <= settings.maximum_position_distance_sampling_steps
                )
                gate_rejection["distance"] += int(np.count_nonzero(~accepted))
                prior = accepted.copy()
                accepted &= (
                    metrics["signedNormalDegrees"]
                    <= settings.maximum_signed_normal_degrees
                )
                gate_rejection["normal"] += int(
                    np.count_nonzero(prior & ~accepted)
                )
                prior = accepted.copy()
                accepted &= (
                    metrics["heightSamplingSteps"]
                    <= settings.maximum_height_sampling_steps
                )
                gate_rejection["height"] += int(
                    np.count_nonzero(prior & ~accepted)
                )
                if not np.any(accepted):
                    continue
                selected_metrics = {
                    name: values[accepted] for name, values in metrics.items()
                }
                exponent = (
                    (
                        selected_metrics["signedNormalDegrees"]
                        / settings.affinity_normal_sigma_degrees
                    )
                    ** 2
                    + (
                        selected_metrics["heightSamplingSteps"]
                        / settings.affinity_height_sigma_steps
                    )
                    ** 2
                )
                edge_first_parts.append(first[accepted].astype(np.int32))
                edge_second_parts.append(second[accepted].astype(np.int32))
                affinity_parts.append(np.exp(-0.5 * exponent).astype(np.float32))
                for name, values in selected_metrics.items():
                    metric_parts.setdefault(name, []).append(
                        values.astype(np.float32)
                    )
    edge_first = (
        np.concatenate(edge_first_parts)
        if edge_first_parts
        else np.empty(0, dtype=np.int32)
    )
    edge_second = (
        np.concatenate(edge_second_parts)
        if edge_second_parts
        else np.empty(0, dtype=np.int32)
    )
    affinity = (
        np.concatenate(affinity_parts)
        if affinity_parts
        else np.empty(0, dtype=np.float32)
    )
    graph = {
        "edgeFirstInterface": edge_first,
        "edgeSecondInterface": edge_second,
        "edgeAffinity": affinity,
        **{
            f"edge{name[0].upper()}{name[1:]}": np.concatenate(parts)
            for name, parts in metric_parts.items()
        },
    }
    return graph, {
        "interfaceCount": int(len(evidence)),
        "eligibleInterfaceCount": int(len(eligible_index)),
        "excludedSeedConflictCount": int(np.count_nonzero(conflict)),
        "consideredPairCount": considered,
        "continuityEdgeCount": int(len(edge_first)),
        "gateRejectionCount": gate_rejection,
        "edgeAffinity": _percentile_record(affinity),
        "edgeSignedNormalDegrees": _percentile_record(
            graph.get("edgeSignedNormalDegrees", np.empty(0))
        ),
        "edgeHeightSamplingSteps": _percentile_record(
            graph.get("edgeHeightSamplingSteps", np.empty(0))
        ),
    }


def _adjacency(
    node_count: int,
    edge_first: np.ndarray,
    edge_second: np.ndarray,
    affinity: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    source = np.concatenate((edge_first, edge_second)).astype(np.int32)
    target = np.concatenate((edge_second, edge_first)).astype(np.int32)
    edge = np.concatenate(
        (
            np.arange(len(edge_first), dtype=np.int32),
            np.arange(len(edge_first), dtype=np.int32),
        )
    )
    weight = np.concatenate((affinity, affinity)).astype(np.float32)
    order = np.argsort(source, kind="stable")
    count = np.bincount(source, minlength=node_count)
    offset = np.concatenate(((0,), np.cumsum(count, dtype=np.int64)))
    return offset, target[order], edge[order], weight[order]


def _retained_component_membership(
    bank: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    settings: OneSidedGrowthSettings,
) -> dict[str, np.ndarray]:
    """Label continuity components without collapsing their seed ambiguity.

    The one-sided boundary field is dense, so its default graph follows only
    processing-lattice sites that share a face.  Longer links appropriate for
    sparse paired profiles can join nearby, distinct material interfaces.  A
    component that touches multiple immutable surface identities is recorded
    as contested and may be deferred by the growth stage.
    """

    node_count = len(bank["positionXYZ"])
    evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    seed_label = np.asarray(bank["seedSurfaceLabel"], dtype=np.int32)
    conflict = np.asarray(bank["seedConflict"]) > 0
    eligible = (
        (evidence >= settings.minimum_candidate_evidence) | (seed_label >= 0)
    ) & ~conflict
    parent = np.arange(node_count, dtype=np.int32)
    size = np.ones(node_count, dtype=np.int32)

    def find(value: int) -> int:
        root = value
        while int(parent[root]) != root:
            root = int(parent[root])
        while int(parent[value]) != value:
            following = int(parent[value])
            parent[value] = root
            value = following
        return root

    first = np.asarray(graph["edgeFirstInterface"], dtype=np.int32)
    second = np.asarray(graph["edgeSecondInterface"], dtype=np.int32)
    affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    retained = affinity >= settings.minimum_growth_bottleneck
    for left, right in zip(first[retained], second[retained]):
        left_root = find(int(left))
        right_root = find(int(right))
        if left_root == right_root:
            continue
        if size[left_root] < size[right_root]:
            left_root, right_root = right_root, left_root
        parent[right_root] = left_root
        size[left_root] += size[right_root]

    eligible_index = np.flatnonzero(eligible).astype(np.int32)
    roots = np.asarray(
        [find(int(value)) for value in eligible_index], dtype=np.int32
    )
    _unique_root, inverse, component_size = np.unique(
        roots, return_inverse=True, return_counts=True
    )
    interface_component = np.full(node_count, -1, dtype=np.int32)
    interface_component[eligible_index] = inverse.astype(np.int32)
    component_count = len(component_size)
    component_seed_label_count = np.zeros(component_count, dtype=np.uint16)
    component_seed_interface_count = np.zeros(component_count, dtype=np.int32)
    component_sole_seed_label = np.full(component_count, -1, dtype=np.int32)
    labels_by_component: dict[int, set[int]] = {}
    for node in np.flatnonzero((seed_label >= 0) & eligible):
        component = int(interface_component[node])
        labels_by_component.setdefault(component, set()).add(
            int(seed_label[node])
        )
        component_seed_interface_count[component] += 1
    for component, labels in labels_by_component.items():
        component_seed_label_count[component] = len(labels)
        if len(labels) == 1:
            component_sole_seed_label[component] = next(iter(labels))
    return {
        "eligibleInterface": eligible.astype(np.uint8),
        "interfaceContinuityComponent": interface_component,
        "componentInterfaceCount": component_size.astype(np.int32),
        "componentSeedInterfaceCount": component_seed_interface_count,
        "componentSeedLabelCount": component_seed_label_count,
        "componentSoleSeedLabel": component_sole_seed_label,
    }


def _relabel_component_membership(
    membership: Mapping[str, np.ndarray],
    seed_label: np.ndarray,
) -> dict[str, np.ndarray]:
    """Recount component identities after a conservative seed association."""

    interface_component = np.asarray(
        membership["interfaceContinuityComponent"], dtype=np.int32
    )
    eligible = np.asarray(membership["eligibleInterface"]) > 0
    component_count = len(membership["componentInterfaceCount"])
    component_seed_label_count = np.zeros(component_count, dtype=np.uint16)
    component_seed_interface_count = np.zeros(component_count, dtype=np.int32)
    component_sole_seed_label = np.full(component_count, -1, dtype=np.int32)
    labels_by_component: dict[int, set[int]] = {}
    for node in np.flatnonzero((seed_label >= 0) & eligible):
        component = int(interface_component[node])
        labels_by_component.setdefault(component, set()).add(
            int(seed_label[node])
        )
        component_seed_interface_count[component] += 1
    for component, labels in labels_by_component.items():
        component_seed_label_count[component] = len(labels)
        if len(labels) == 1:
            component_sole_seed_label[component] = next(iter(labels))
    return {
        **membership,
        "componentSeedInterfaceCount": component_seed_interface_count,
        "componentSeedLabelCount": component_seed_label_count,
        "componentSoleSeedLabel": component_sole_seed_label,
    }


def associate_surface_labels_from_boundary_components(
    bank: Mapping[str, np.ndarray],
    membership: Mapping[str, np.ndarray],
    *,
    settings: OneSidedGrowthSettings,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Associate sheet fragments only when both signed faces agree.

    A continuity component is one connected air-to-material boundary patch.
    Two paired-surface identities become an association candidate when they
    occur on such a patch.  The association is accepted only when independent
    components support both corresponding physical faces with a consistent
    canonical-side parity.  This prevents a single exposed face or a queue tie
    from merging sheets.
    """

    seed_label = np.asarray(bank["seedSurfaceLabel"], dtype=np.int32)
    seed_side = np.asarray(bank["seedBoundarySide"], dtype=np.uint8)
    interface_component = np.asarray(
        membership["interfaceContinuityComponent"], dtype=np.int32
    )
    component_size = np.asarray(
        membership["componentInterfaceCount"], dtype=np.int32
    )
    initial_seed_label_count = np.asarray(
        membership["componentSeedLabelCount"]
    )
    incidence: dict[int, dict[int, list[int]]] = {}
    for node in np.flatnonzero(seed_label >= 0):
        component = int(interface_component[node])
        if component < 0 or initial_seed_label_count[component] < 2:
            continue
        label = int(seed_label[node])
        counts = incidence.setdefault(component, {}).setdefault(label, [0, 0])
        counts[int(seed_side[node])] += 1

    support: dict[tuple[int, int], dict[str, list[int]]] = {}
    for component, labels in incidence.items():
        pure: dict[int, tuple[int, int]] = {}
        for label, side_count in labels.items():
            total = sum(side_count)
            dominant_side = int(side_count[1] > side_count[0])
            purity = max(side_count) / max(total, 1)
            if purity >= settings.minimum_association_seed_side_purity:
                pure[label] = dominant_side, total
        ordered = sorted(pure)
        for first_index, first in enumerate(ordered):
            first_side, first_count = pure[first]
            for second in ordered[first_index + 1 :]:
                second_side, second_count = pure[second]
                orientation = 2 * first_side + second_side
                record = support.setdefault(
                    (first, second),
                    {
                        "componentCount": [0, 0, 0, 0],
                        "balancedSeedSupport": [0, 0, 0, 0],
                        "interfaceSupport": [0, 0, 0, 0],
                    },
                )
                record["componentCount"][orientation] += 1
                record["balancedSeedSupport"][orientation] += min(
                    first_count, second_count
                )
                record["interfaceSupport"][orientation] += int(
                    component_size[component]
                )

    records: list[dict[str, Any]] = []
    for (first, second), value in support.items():
        component_count = value["componentCount"]
        balanced = value["balancedSeedSupport"]
        same_support = min(balanced[0], balanced[3])
        flipped_support = min(balanced[1], balanced[2])
        same_components = min(component_count[0], component_count[3])
        flipped_components = min(component_count[1], component_count[2])
        flipped = (flipped_support, flipped_components) > (
            same_support,
            same_components,
        )
        if flipped:
            side_indices = (1, 2)
            bilateral_support = flipped_support
            bilateral_components = flipped_components
        else:
            side_indices = (0, 3)
            bilateral_support = same_support
            bilateral_components = same_components
        dominant_support = sum(balanced[index] for index in side_indices)
        orientation_purity = dominant_support / max(sum(balanced), 1)
        threshold_accepted = bool(
            bilateral_support
            >= settings.minimum_association_balanced_seed_support
            and bilateral_components
            >= settings.minimum_association_side_component_count
            and orientation_purity
            >= settings.minimum_association_orientation_purity
        )
        records.append(
            {
                "firstSurfaceLabel": first,
                "secondSurfaceLabel": second,
                "componentCountBySidePair": component_count,
                "balancedSeedSupportBySidePair": balanced,
                "interfaceSupportBySidePair": value["interfaceSupport"],
                "orientationFlipped": flipped,
                "bilateralComponentCount": bilateral_components,
                "bilateralBalancedSeedSupport": bilateral_support,
                "orientationPurity": orientation_purity,
                "thresholdAccepted": threshold_accepted,
                "accepted": False,
                "orientationConflict": False,
            }
        )
    records.sort(
        key=lambda value: (
            not value["thresholdAccepted"],
            -value["bilateralBalancedSeedSupport"],
            -value["bilateralComponentCount"],
            value["firstSurfaceLabel"],
            value["secondSurfaceLabel"],
        )
    )

    surface_label = np.unique(seed_label[seed_label >= 0]).astype(np.int32)
    maximum_label = int(np.max(surface_label, initial=-1))
    parent = np.arange(maximum_label + 1, dtype=np.int32)
    parity = np.zeros(maximum_label + 1, dtype=np.uint8)

    def find(value: int) -> tuple[int, int]:
        if int(parent[value]) == value:
            return value, 0
        root, parent_parity = find(int(parent[value]))
        parity[value] ^= parent_parity
        parent[value] = root
        return root, int(parity[value])

    for record in records:
        if not record["thresholdAccepted"]:
            continue
        first = int(record["firstSurfaceLabel"])
        second = int(record["secondSurfaceLabel"])
        relation = int(record["orientationFlipped"])
        first_root, first_parity = find(first)
        second_root, second_parity = find(second)
        if first_root == second_root:
            if (first_parity ^ second_parity) != relation:
                record["orientationConflict"] = True
                continue
            record["accepted"] = True
            continue
        relative = first_parity ^ second_parity ^ relation
        if first_root < second_root:
            parent[second_root] = first_root
            parity[second_root] = relative
        else:
            parent[first_root] = second_root
            parity[first_root] = relative
        record["accepted"] = True

    surface_assembly = np.empty_like(surface_label)
    surface_orientation = np.empty(len(surface_label), dtype=np.uint8)
    for index, label in enumerate(surface_label):
        root, orientation = find(int(label))
        surface_assembly[index] = root
        surface_orientation[index] = orientation
    assembly_by_label = {
        int(label): int(assembly)
        for label, assembly in zip(surface_label, surface_assembly)
    }
    orientation_by_label = {
        int(label): int(orientation)
        for label, orientation in zip(surface_label, surface_orientation)
    }
    seed_assembly = np.full(len(seed_label), -1, dtype=np.int32)
    seed_orientation = np.zeros(len(seed_label), dtype=np.uint8)
    for node in np.flatnonzero(seed_label >= 0):
        label = int(seed_label[node])
        seed_assembly[node] = assembly_by_label[label]
        seed_orientation[node] = orientation_by_label[label]
    assembly_value, assembly_member_count = np.unique(
        surface_assembly, return_counts=True
    )
    arrays = {
        "seedAssemblyLabel": seed_assembly,
        "seedAssemblyOrientationParity": seed_orientation,
        "surfaceLabel": surface_label,
        "surfaceAssemblyLabel": surface_assembly,
        "surfaceAssemblyOrientationParity": surface_orientation,
        "associationFirstSurfaceLabel": np.asarray(
            [value["firstSurfaceLabel"] for value in records], dtype=np.int32
        ),
        "associationSecondSurfaceLabel": np.asarray(
            [value["secondSurfaceLabel"] for value in records], dtype=np.int32
        ),
        "associationComponentCountBySidePair": np.asarray(
            [value["componentCountBySidePair"] for value in records],
            dtype=np.int32,
        ),
        "associationBalancedSeedSupportBySidePair": np.asarray(
            [value["balancedSeedSupportBySidePair"] for value in records],
            dtype=np.int32,
        ),
        "associationInterfaceSupportBySidePair": np.asarray(
            [value["interfaceSupportBySidePair"] for value in records],
            dtype=np.int32,
        ),
        "associationOrientationFlipped": np.asarray(
            [value["orientationFlipped"] for value in records], dtype=np.uint8
        ),
        "associationAccepted": np.asarray(
            [value["accepted"] for value in records], dtype=np.uint8
        ),
    }
    accepted = [value for value in records if value["accepted"]]
    return arrays, {
        "inputSurfaceLabelCount": int(len(surface_label)),
        "candidatePairCount": int(len(records)),
        "thresholdAcceptedPairCount": int(
            sum(value["thresholdAccepted"] for value in records)
        ),
        "acceptedPairCount": int(len(accepted)),
        "orientationConflictPairCount": int(
            sum(value["orientationConflict"] for value in records)
        ),
        "surfaceAssemblyCount": int(len(assembly_value)),
        "multiLabelAssemblyCount": int(
            np.count_nonzero(assembly_member_count > 1)
        ),
        "largestAssemblyLabelCount": int(
            np.max(assembly_member_count, initial=0)
        ),
        "strongestPairs": records[:128],
    }


def _graph_component_census(
    bank: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    settings: OneSidedGrowthSettings,
    membership: Mapping[str, np.ndarray] | None = None,
    seed_label_override: np.ndarray | None = None,
) -> dict[str, Any]:
    resolved = membership or _retained_component_membership(
        bank, graph, settings=settings
    )
    component_size = np.asarray(resolved["componentInterfaceCount"])
    seed_label_count = np.asarray(resolved["componentSeedLabelCount"])
    interface_component = np.asarray(
        resolved["interfaceContinuityComponent"], dtype=np.int32
    )
    seed_label = np.asarray(
        seed_label_override
        if seed_label_override is not None
        else bank["seedSurfaceLabel"],
        dtype=np.int32,
    )
    labels_by_component: dict[int, set[int]] = {}
    for node in np.flatnonzero((seed_label >= 0) & (interface_component >= 0)):
        labels_by_component.setdefault(int(interface_component[node]), set()).add(
            int(seed_label[node])
        )
    largest_records: list[dict[str, Any]] = []
    order = np.argsort(-component_size)
    for component in order:
        count = int(component_size[component])
        labels = sorted(labels_by_component.get(int(component), set()))
        if len(largest_records) < 128:
            largest_records.append(
                {
                    "component": int(component),
                    "interfaceCount": count,
                    "seedLabelCount": int(len(labels)),
                    "seedLabels": labels[:32],
                }
            )
    return {
        "componentCount": int(len(component_size)),
        "seededSingleLabelComponentCount": int(
            np.count_nonzero(seed_label_count == 1)
        ),
        "unseededComponentCount": int(np.count_nonzero(seed_label_count == 0)),
        "contestedMultiLabelComponentCount": int(
            np.count_nonzero(seed_label_count > 1)
        ),
        "componentSize": _percentile_record(component_size),
        "componentCountAtLeast8": int(np.count_nonzero(component_size >= 8)),
        "componentCountAtLeast32": int(np.count_nonzero(component_size >= 32)),
        "componentCountAtLeast128": int(np.count_nonzero(component_size >= 128)),
        "largestComponents": largest_records,
    }


def grow_one_sided_interfaces(
    bank: Mapping[str, np.ndarray],
    graph: Mapping[str, np.ndarray],
    *,
    settings: OneSidedGrowthSettings,
    membership: Mapping[str, np.ndarray] | None = None,
    seed_label_override: np.ndarray | None = None,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Grow identities without resolving contested boundary components."""

    node_count = len(bank["positionXYZ"])
    resolved_membership = membership or _retained_component_membership(
        bank, graph, settings=settings
    )
    interface_component = np.asarray(
        resolved_membership["interfaceContinuityComponent"], dtype=np.int32
    )
    component_seed_label_count = np.asarray(
        resolved_membership["componentSeedLabelCount"]
    )
    component_sole_seed_label = np.asarray(
        resolved_membership["componentSoleSeedLabel"], dtype=np.int32
    )
    seed_label = np.asarray(
        seed_label_override
        if seed_label_override is not None
        else bank["seedSurfaceLabel"],
        dtype=np.int32,
    )
    locked_seed = seed_label >= 0
    local_evidence = np.asarray(bank["localEvidenceScore"], dtype=np.float32)
    edge_first = np.asarray(graph["edgeFirstInterface"], dtype=np.int32)
    edge_second = np.asarray(graph["edgeSecondInterface"], dtype=np.int32)
    edge_affinity = np.asarray(graph["edgeAffinity"], dtype=np.float32)
    offset, neighbor, adjacency_edge, adjacency_affinity = _adjacency(
        node_count, edge_first, edge_second, edge_affinity
    )
    label = np.full(node_count, -1, dtype=np.int32)
    path_score = np.zeros(node_count, dtype=np.float32)
    parent = np.full(node_count, -1, dtype=np.int32)
    parent_edge = np.full(node_count, -1, dtype=np.int32)
    queue: list[tuple[float, int, int, int, int]] = [
        (-1.0, int(node), int(seed_label[node]), -1, -1)
        for node in np.flatnonzero(locked_seed)
    ]
    heapq.heapify(queue)
    proposal_count = len(queue)
    protected_seed_rejection = 0
    while queue:
        negative_score, node, proposed_label, proposed_parent, proposed_edge = (
            heapq.heappop(queue)
        )
        score = -negative_score
        if label[node] >= 0:
            continue
        if score < settings.minimum_growth_bottleneck:
            break
        if locked_seed[node] and int(seed_label[node]) != proposed_label:
            protected_seed_rejection += 1
            continue
        label[node] = proposed_label
        path_score[node] = score
        parent[node] = proposed_parent
        parent_edge[node] = proposed_edge
        component = int(interface_component[node])
        if (
            settings.defer_contested_components
            and (
                component < 0
                or int(component_seed_label_count[component]) != 1
                or int(component_sole_seed_label[component]) != proposed_label
            )
        ):
            continue
        for adjacency_index in range(int(offset[node]), int(offset[node + 1])):
            target = int(neighbor[adjacency_index])
            if label[target] >= 0:
                continue
            if locked_seed[target] and int(seed_label[target]) != proposed_label:
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
                    node,
                    int(adjacency_edge[adjacency_index]),
                ),
            )
            proposal_count += 1
    selected = label >= 0
    grown = selected & ~locked_seed
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
        minlength=node_count,
    ).astype(np.uint16)
    selected_label, selected_count = np.unique(
        label[selected], return_counts=True
    )
    seed_values, seed_count = np.unique(seed_label[locked_seed], return_counts=True)
    seed_size = {
        int(value): int(count) for value, count in zip(seed_values, seed_count)
    }
    records = [
        {
            "label": int(value),
            "seedCount": seed_size.get(int(value), 0),
            "selectedCount": int(count),
            "grownCount": int(count) - seed_size.get(int(value), 0),
        }
        for value, count in zip(selected_label, selected_count)
    ]
    records.sort(key=lambda value: (-value["selectedCount"], value["label"]))
    eligible = np.asarray(resolved_membership["eligibleInterface"]) > 0
    valid_component = interface_component >= 0
    contested = np.zeros(node_count, dtype=bool)
    unseeded = np.zeros(node_count, dtype=bool)
    contested[valid_component] = (
        component_seed_label_count[interface_component[valid_component]] > 1
    )
    unseeded[valid_component] = (
        component_seed_label_count[interface_component[valid_component]] == 0
    )
    return {
        "selectedLabel": label,
        "pathBottleneck": path_score,
        "parentInterface": parent,
        "parentContinuityEdge": parent_edge,
        "lockedSeed": locked_seed.astype(np.uint8),
        "selected": selected.astype(np.uint8),
        "selectedSameLabelNeighborCount": same_label_neighbor_count,
        **resolved_membership,
    }, {
        "lockedSeedCount": int(np.count_nonzero(locked_seed)),
        "selectedInterfaceCount": int(np.count_nonzero(selected)),
        "grownInterfaceCount": int(np.count_nonzero(grown)),
        "selectedLabelCount": int(len(selected_label)),
        "proposalCount": proposal_count,
        "protectedSeedRejectionCount": protected_seed_rejection,
        "deferredContestedInterfaceCount": int(
            np.count_nonzero(eligible & contested & ~locked_seed)
        ),
        "unseededInterfaceCount": int(np.count_nonzero(eligible & unseeded)),
        "pathBottleneck": _percentile_record(path_score[grown]),
        "grownLocalEvidence": _percentile_record(local_evidence[grown]),
        "grownSameLabelNeighborCount": _percentile_record(
            same_label_neighbor_count[grown]
        ),
        "grownWithAtLeastTwoSameLabelNeighborsFraction": (
            round(float(np.mean(same_label_neighbor_count[grown] >= 2)), 6)
            if np.any(grown)
            else None
        ),
        "largestLabels": records[:128],
    }


def _label_colors(labels: np.ndarray, maximum: int) -> dict[int, tuple[int, int, int]]:
    value, count = np.unique(labels[labels >= 0], return_counts=True)
    order = np.lexsort((value, -count))[:maximum]
    return {
        int(value[index]): tuple(
            int(round(255.0 * channel))
            for channel in colorsys.hsv_to_rgb(
                (0.09 + 0.61803398875 * rank) % 1.0, 0.68, 0.98
            )
        )
        for rank, index in enumerate(order)
    }


def write_one_sided_growth_projection(
    bank: Mapping[str, np.ndarray],
    selection: Mapping[str, np.ndarray],
    world_start_xyz: np.ndarray,
    world_stop_xyz: np.ndarray,
    settings: OneSidedGrowthSettings,
    path: str | Path,
    *,
    delta_only: bool = False,
    panel_size: int = 640,
) -> Path:
    output = Path(path)
    label = np.asarray(selection["selectedLabel"], dtype=np.int32)
    locked = np.asarray(selection["lockedSeed"]) > 0
    selected = label >= 0
    point = np.asarray(bank["positionXYZ"])
    colors = _label_colors(label, settings.maximum_preview_labels)
    canvas = np.full((panel_size, 3 * panel_size, 3), (7, 10, 14), dtype=np.uint8)
    margin = max(12, panel_size // 30)
    width = np.maximum(world_stop_xyz - world_start_xyz, 1.0)
    for panel, axes in enumerate(((0, 1), (0, 2), (1, 2))):
        offset = panel * panel_size
        for value in sorted(colors, reverse=True):
            mask = selected & (label == value)
            if delta_only:
                mask &= ~locked
            points = point[mask]
            if not len(points):
                continue
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
            canvas[y[valid], x[valid]] = colors[value]
        border = (64, 72, 84)
        canvas[margin, offset + margin : offset + panel_size - margin] = border
        canvas[
            panel_size - margin,
            offset + margin : offset + panel_size - margin,
        ] = border
        canvas[margin : panel_size - margin, offset + margin] = border
        canvas[
            margin : panel_size - margin,
            offset + panel_size - margin,
        ] = border
    temporary = output.with_suffix(output.suffix + ".tmp")
    temporary.write_bytes(rgb_png(canvas))
    temporary.replace(output)
    return output


def write_one_sided_growth_cross_sections(
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
    selected = np.asarray(selection["selected"]) > 0
    locked = np.asarray(selection["lockedSeed"]) > 0
    point = np.asarray(bank["positionXYZ"])
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
    for view_index, (axis, source_index) in enumerate(views):
        if axis == "z":
            raw = volume[
                source_index,
                owned.start_xyz[1] : owned.stop_xyz_exclusive[1],
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[2]
            near = selected & (np.abs(point[:, 2] - coordinate) <= tolerance)
            x = np.rint(point[near, 0] - world_start[0]).astype(np.int32)
            y = np.rint(point[near, 1] - world_start[1]).astype(np.int32)
        else:
            raw = volume[
                owned.start_xyz[2] : owned.stop_xyz_exclusive[2],
                source_index,
                owned.start_xyz[0] : owned.stop_xyz_exclusive[0],
            ]
            coordinate = source_index + source.origin_xyz[1]
            near = selected & (np.abs(point[:, 1] - coordinate) <= tolerance)
            x = np.rint(point[near, 0] - world_start[0]).astype(np.int32)
            y = np.rint(point[near, 2] - world_start[2]).astype(np.int32)
        gray = np.clip(
            np.asarray(raw, dtype=np.float32) / max(display_high_raw, 1.0) * 255.0,
            0,
            255,
        ).astype(np.uint8)
        panel = np.repeat(gray[:, :, None], 3, axis=2)
        locked_near = locked[near]
        valid = (
            (x >= 1)
            & (x < panel.shape[1] - 1)
            & (y >= 1)
            & (y < panel.shape[0] - 1)
        )
        x, y, locked_near = x[valid], y[valid], locked_near[valid]
        color = np.where(
            locked_near[:, None],
            np.asarray((38, 238, 202), dtype=np.uint8),
            np.asarray((255, 164, 62), dtype=np.uint8),
        )
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                panel[y + dy, x + dx] = color
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


def run_one_sided_growth(
    bank_root: str | Path,
    output_root: str | Path,
    *,
    settings: OneSidedGrowthSettings | None = None,
    force: bool = False,
) -> dict[str, Any]:
    resolved = settings or OneSidedGrowthSettings()
    bank_path, bank_manifest, bank = _load_interface_bank(bank_root)
    paired_bank_path = Path(
        bank_manifest["identity"]["pairedBank"]["manifestPath"]
    )
    paired_bank_manifest = json.loads(paired_bank_path.read_text())
    slab_path = Path(
        paired_bank_manifest["identity"]["isolatedSlabs"]["manifestPath"]
    )
    slab_manifest = json.loads(slab_path.read_text())
    stride = int(slab_manifest["identity"]["settings"]["sampling_stride_voxels"])
    processing_shape = tuple(
        int(value)
        for value in bank_manifest["geometry"]["processingShapeSamplingXYZ"]
    )
    identity: dict[str, Any] = {
        "schema": ONE_SIDED_GROWTH_SCHEMA,
        "version": ONE_SIDED_GROWTH_VERSION,
        "interfaceBank": {
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
    manifest_path = output / f"{ONE_SIDED_GROWTH_STEM}.json"
    data_path = output / f"{ONE_SIDED_GROWTH_STEM}.npz"
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
    stage = time.monotonic()
    graph, graph_stats = build_one_sided_continuity_graph(
        bank,
        processing_shape_sampling_xyz=processing_shape,
        stride=stride,
        settings=resolved,
    )
    graphed = time.monotonic()
    initial_membership = _retained_component_membership(
        bank, graph, settings=resolved
    )
    association, association_stats = (
        associate_surface_labels_from_boundary_components(
            bank, initial_membership, settings=resolved
        )
    )
    effective_seed_label = np.asarray(
        association["seedAssemblyLabel"], dtype=np.int32
    )
    membership = _relabel_component_membership(
        initial_membership, effective_seed_label
    )
    component_stats = _graph_component_census(
        bank,
        graph,
        settings=resolved,
        membership=membership,
        seed_label_override=effective_seed_label,
    )
    censused = time.monotonic()
    selection, growth_stats = grow_one_sided_interfaces(
        bank,
        graph,
        settings=resolved,
        membership=membership,
        seed_label_override=effective_seed_label,
    )
    grown = time.monotonic()
    arrays = {**selection, **association, **graph}
    _write_npz(data_path, arrays)

    source = VolumeSource.open(
        bank_manifest["source"]["path"], bank_manifest["source"]["metadataPath"]
    )
    owned_record = bank_manifest["geometry"]["ownedVoxelBounds"]
    owned = VoxelBounds(
        tuple(owned_record["startXYZ"]), tuple(owned_record["stopXYZExclusive"])
    )
    world_record = bank_manifest["geometry"]["ownedWorldBounds"]
    world_start = np.asarray(world_record["startXYZ"], dtype=np.float64)
    world_stop = np.asarray(world_record["stopXYZExclusive"], dtype=np.float64)
    projection = write_one_sided_growth_projection(
        bank,
        selection,
        world_start,
        world_stop,
        resolved,
        output / "grown-one-sided-interfaces.png",
    )
    delta_projection = write_one_sided_growth_projection(
        bank,
        selection,
        world_start,
        world_stop,
        resolved,
        output / "one-sided-growth-only.png",
        delta_only=True,
    )
    cross_sections = write_one_sided_growth_cross_sections(
        source,
        owned,
        bank,
        selection,
        output / "one-sided-growth-cross-sections.png",
        display_high_raw=float(bank_manifest["calibration"]["displayHighRaw"]),
        sampling_stride=stride,
    )
    finished = time.monotonic()
    payload: dict[str, Any] = {
        "schema": ONE_SIDED_GROWTH_SCHEMA,
        "version": ONE_SIDED_GROWTH_VERSION,
        "state": "complete",
        "identity": identity,
        "source": bank_manifest["source"],
        "geometry": bank_manifest["geometry"],
        "graph": graph_stats,
        "association": association_stats,
        "components": component_stats,
        "growth": growth_stats,
        "timingSeconds": {
            "graph": round(graphed - stage, 6),
            "componentCensus": round(censused - graphed, 6),
            "growth": round(grown - censused, 6),
            "writingAndArtifacts": round(finished - grown, 6),
            "total": round(finished - started, 6),
        },
        "data": {
            "path": data_path.name,
            "bytes": data_path.stat().st_size,
            "sha256": sha256_file(data_path),
            "fields": list(arrays),
        },
        "artifacts": {
            "grownInterfaces": projection.name,
            "growthOnly": delta_projection.name,
            "crossSections": cross_sections.name,
        },
        "method": {
            "geometryUnit": "signed air-to-material CT interface",
            "topology": (
                "shared-face processing-lattice continuity by default; "
                "longer sparse-field jumps are opt-in"
            ),
            "propagation": (
                "seed-protected maximum-bottleneck forest restricted to "
                "single-label continuity components"
            ),
            "seedAssociation": (
                "two paired-surface identities may merge only when multiple "
                "boundary components support both corresponding signed faces"
            ),
            "changesInterfaceGeometry": False,
            "infersMissingOppositeFace": False,
            "hardBarrier": (
                "interfaces with conflicting paired-surface seeds and all "
                "unseeded interiors of multi-label continuity components"
            ),
        },
    }
    atomic_json(manifest_path, payload)
    return payload
