from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .geometry import ClippedPatch, DegeneratePlaneIntersection, clip_plane_to_cell
from .matching import TraceMatchSettings, align_face_patches
from .stratigraphy import ConfigurationTable
from .topology import GridSpec, Int3, cell_face


@dataclass(frozen=True, slots=True)
class ConfigurationOption:
    option_id: int
    cell_xyz: Int3
    source_table_index: int
    source_configuration_index: int
    local_configuration_id: int
    log_weight: float
    patches: tuple[ClippedPatch, ...]
    degenerate_layer_count: int


@dataclass(frozen=True, slots=True)
class ConfigurationSelection:
    selected_options: tuple[ConfigurationOption, ...]
    patches: tuple[ClippedPatch, ...]
    sweeps: int
    changed_last_sweep: int
    unary_energy: float
    pairwise_energy: float
    pairwise_reward: float
    continuation_energy: float
    total_energy: float
    pairwise_evaluation_count: int
    interior_unmatched_trace_count: int
    degenerate_layer_count: int


def configuration_options(
    grid: GridSpec,
    tables: Iterable[ConfigurationTable],
) -> tuple[dict[Int3, tuple[ConfigurationOption, ...]], int]:
    by_cell: dict[Int3, list[ConfigurationOption]] = {}
    patch_id = 1
    option_id = 0
    degenerate_total = 0
    for table_index, table in enumerate(tables):
        table.validate()
        for cell_index, cell_values in enumerate(table.cell_xyz):
            cell = tuple(int(value) for value in cell_values)
            if not grid.contains_cell(cell):
                raise ValueError(f"configuration cell {cell} lies outside the grid")
            if cell in by_cell:
                raise ValueError(f"configuration cell {cell} is owned by multiple shards")
            options: list[ConfigurationOption] = []
            for configuration_index in table.configurations_for_cell(cell_index):
                patches: list[ClippedPatch] = []
                degenerate = 0
                for estimate in table.estimates_for_configuration(configuration_index):
                    try:
                        patch = clip_plane_to_cell(
                            grid, cell, estimate, patch_id=patch_id
                        )
                    except DegeneratePlaneIntersection:
                        degenerate += 1
                        patch = None
                    if patch is not None:
                        patches.append(patch)
                        patch_id += 1
                patches.sort(
                    key=lambda value: (
                        value.estimate.height_from_cell_center,
                        value.patch_id,
                    )
                )
                degenerate_total += degenerate
                options.append(
                    ConfigurationOption(
                        option_id,
                        cell,
                        table_index,
                        configuration_index,
                        int(table.configuration_id[configuration_index]),
                        float(table.configuration_log_weight[configuration_index]),
                        tuple(patches),
                        degenerate,
                    )
                )
                option_id += 1
            by_cell[cell] = options
    return {
        key: tuple(sorted(value, key=lambda option: option.option_id))
        for key, value in by_cell.items()
    }, degenerate_total


def _neighbor_pairs(grid: GridSpec) -> tuple[tuple[Int3, Int3, int], ...]:
    result: list[tuple[Int3, Int3, int]] = []
    for iz in range(grid.shape_cells_xyz[2]):
        for iy in range(grid.shape_cells_xyz[1]):
            for ix in range(grid.shape_cells_xyz[0]):
                first = (ix, iy, iz)
                for axis in range(3):
                    second = list(first)
                    second[axis] += 1
                    resolved = tuple(second)
                    if grid.contains_cell(resolved):
                        result.append((first, resolved, axis))
    return tuple(result)


def optimize_configurations(
    grid: GridSpec,
    tables: Iterable[ConfigurationTable],
    *,
    matching_settings: TraceMatchSettings | None = None,
    unary_scale: float = 1.0,
    pairwise_scale: float = 0.35,
    interior_unmatched_trace_penalty: float = 0.0,
    maximum_sweeps: int = 12,
) -> ConfigurationSelection:
    """Select one local stratigraphy per cell with face-relative likelihoods.

    The pairwise reward is the face alignment likelihood relative to leaving
    all traces unmatched. ``interior_unmatched_trace_penalty`` optionally adds
    a soft birth/death cost when both neighboring configurations contain
    layers but some shared-face traces remain unmatched. Empty neighbors stay
    neutral, preserving a representation for air and real page boundaries.
    """

    if unary_scale <= 0.0 or pairwise_scale <= 0.0 or maximum_sweeps <= 0:
        raise ValueError("selection scales and maximum sweeps must be positive")
    if (
        not math.isfinite(interior_unmatched_trace_penalty)
        or interior_unmatched_trace_penalty < 0.0
    ):
        raise ValueError("interior unmatched-trace penalty must be finite and nonnegative")
    resolved_matching = matching_settings or TraceMatchSettings()
    options_by_cell, degenerate_total = configuration_options(grid, tables)
    expected_cells = int(np.prod(grid.shape_cells_xyz))
    if len(options_by_cell) != expected_cells:
        missing = [
            (ix, iy, iz)
            for iz in range(grid.shape_cells_xyz[2])
            for iy in range(grid.shape_cells_xyz[1])
            for ix in range(grid.shape_cells_xyz[0])
            if (ix, iy, iz) not in options_by_cell
        ]
        raise ValueError(
            f"configuration shards cover {len(options_by_cell)}/{expected_cells} cells; "
            f"first missing cells: {missing[:4]}"
        )
    pairs = _neighbor_pairs(grid)
    neighbors: dict[Int3, list[tuple[Int3, int, bool]]] = {
        cell: [] for cell in options_by_cell
    }
    for first, second, axis in pairs:
        neighbors[first].append((second, axis, True))
        neighbors[second].append((first, axis, False))

    pair_cache: dict[tuple[int, int, int, bool], float] = {}
    pair_details: dict[
        tuple[int, int, int, bool], tuple[float, float, int]
    ] = {}

    def pair_energy(
        first: ConfigurationOption,
        second: ConfigurationOption,
        axis: int,
        first_is_lower: bool,
    ) -> float:
        key = (first.option_id, second.option_id, axis, first_is_lower)
        if key in pair_cache:
            return pair_cache[key]
        lower = first if first_is_lower else second
        upper = second if first_is_lower else first
        face = cell_face(lower.cell_xyz, axis, 1)
        lower_traces = sum(patch.trace_on(face) is not None for patch in lower.patches)
        upper_traces = sum(patch.trace_on(face) is not None for patch in upper.patches)
        baseline = resolved_matching.unmatched_negative_log_likelihood * (
            lower_traces + upper_traces
        )
        if not lower_traces and not upper_traces:
            relative = 0.0
            unmatched = 0
        else:
            try:
                alignment = align_face_patches(
                    lower.patches,
                    upper.patches,
                    face,
                    resolved_matching,
                    grid=grid,
                )
                relative = alignment.negative_log_likelihood - baseline
                unmatched = len(alignment.unmatched_first_patch_ids) + len(
                    alignment.unmatched_second_patch_ids
                )
            except ValueError:
                relative = 0.0
                unmatched = lower_traces + upper_traces
        reward = pairwise_scale * min(float(relative), 0.0)
        continuation = (
            interior_unmatched_trace_penalty * unmatched
            if lower.patches and upper.patches
            else 0.0
        )
        value = reward + continuation
        pair_cache[key] = value
        pair_details[key] = (reward, continuation, unmatched)
        reverse_key = (second.option_id, first.option_id, axis, not first_is_lower)
        pair_cache[reverse_key] = value
        pair_details[reverse_key] = (reward, continuation, unmatched)
        return value

    selected: dict[Int3, ConfigurationOption] = {
        cell: min(
            options,
            key=lambda value: (
                unary_scale * -value.log_weight
                + resolved_matching.unmatched_negative_log_likelihood
                * value.degenerate_layer_count,
                value.option_id,
            ),
        )
        for cell, options in options_by_cell.items()
    }
    ordered_cells = sorted(options_by_cell, key=lambda cell: (cell[2], cell[1], cell[0]))
    changed = 0
    completed_sweeps = 0
    for sweep in range(maximum_sweeps):
        completed_sweeps = sweep + 1
        changed = 0
        traversal = ordered_cells if sweep % 2 == 0 else list(reversed(ordered_cells))
        for cell in traversal:
            best_option = selected[cell]
            best_energy = math.inf
            for option in options_by_cell[cell]:
                energy = unary_scale * -option.log_weight
                energy += (
                    resolved_matching.unmatched_negative_log_likelihood
                    * option.degenerate_layer_count
                )
                for neighbor, axis, is_lower in neighbors[cell]:
                    energy += pair_energy(option, selected[neighbor], axis, is_lower)
                if (energy, option.option_id) < (best_energy, best_option.option_id):
                    best_energy = energy
                    best_option = option
            if best_option.option_id != selected[cell].option_id:
                selected[cell] = best_option
                changed += 1
        if not changed:
            break

    selected_options = tuple(selected[cell] for cell in ordered_cells)
    selected_patches = tuple(
        patch for option in selected_options for patch in option.patches
    )
    unary_energy = sum(
        unary_scale * -option.log_weight
        + resolved_matching.unmatched_negative_log_likelihood
        * option.degenerate_layer_count
        for option in selected_options
    )
    pairwise_energy = 0.0
    pairwise_reward = 0.0
    continuation_energy = 0.0
    interior_unmatched_trace_count = 0
    for first, second, axis in pairs:
        first_option = selected[first]
        second_option = selected[second]
        pairwise_energy += pair_energy(first_option, second_option, axis, True)
        detail = pair_details[(first_option.option_id, second_option.option_id, axis, True)]
        pairwise_reward += detail[0]
        continuation_energy += detail[1]
        if first_option.patches and second_option.patches:
            interior_unmatched_trace_count += detail[2]
    return ConfigurationSelection(
        selected_options,
        selected_patches,
        completed_sweeps,
        changed,
        float(unary_energy),
        float(pairwise_energy),
        float(pairwise_reward),
        float(continuation_energy),
        float(unary_energy + pairwise_energy),
        len(pair_cache) // 2,
        interior_unmatched_trace_count,
        degenerate_total,
    )
