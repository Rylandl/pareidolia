from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .geometry import (
    ClippedPatch,
    DegeneratePlaneIntersection,
    PlaneEstimate,
    clip_plane_to_cell,
    plane_basis,
)
from .topology import GridSpec


@dataclass(frozen=True, slots=True)
class SyntheticStackSettings:
    sheet_count: int = 4
    base_height_cells: float = 0.63
    sheet_spacing_cells: float | None = None
    curvature_amplitude_cells: float = 0.12
    angular_standard_deviation_radians: float = math.radians(1.5)
    height_standard_deviation_cells: float = 0.025
    observation_noise_scale: float = 0.25
    missing_patch_fraction: float = 0.0
    random_seed: int = 7

    def __post_init__(self) -> None:
        if self.sheet_count <= 0:
            raise ValueError("synthetic stack needs at least one sheet")
        if self.sheet_spacing_cells is not None and self.sheet_spacing_cells <= 0.0:
            raise ValueError("sheet spacing must be positive")
        if self.curvature_amplitude_cells < 0.0:
            raise ValueError("curvature amplitude must be nonnegative")
        if self.angular_standard_deviation_radians < 0.0:
            raise ValueError("angular standard deviation must be nonnegative")
        if self.height_standard_deviation_cells < 0.0:
            raise ValueError("height standard deviation must be nonnegative")
        if self.observation_noise_scale < 0.0:
            raise ValueError("observation noise scale must be nonnegative")
        if not 0.0 <= self.missing_patch_fraction < 1.0:
            raise ValueError("missing patch fraction must lie in [0, 1)")


@dataclass(frozen=True, slots=True)
class SyntheticScene:
    grid: GridSpec
    settings: SyntheticStackSettings
    patches: tuple[ClippedPatch, ...]
    truth_sheet_by_patch: tuple[tuple[int, int], ...]
    degenerate_candidate_count: int

    @property
    def truth_map(self) -> dict[int, int]:
        return dict(self.truth_sheet_by_patch)


def _surface(
    grid: GridSpec,
    settings: SyntheticStackSettings,
    sheet_index: int,
    x_grid: float,
    y_grid: float,
) -> tuple[float, float, float]:
    shape_x, shape_y, shape_z = grid.shape_cells_xyz
    spacing = settings.sheet_spacing_cells
    if spacing is None:
        spacing = (
            (shape_z - 0.71 - settings.base_height_cells)
            / max(settings.sheet_count - 1, 1)
            if settings.sheet_count > 1
            else 1.0
        )
    phase = 0.37 * sheet_index
    x_angle = 2.0 * math.pi * (x_grid / max(shape_x, 1) + phase)
    y_angle = 2.0 * math.pi * (y_grid / max(shape_y, 1) - 0.5 * phase)
    amplitude = settings.curvature_amplitude_cells
    height = (
        settings.base_height_cells
        + sheet_index * spacing
        + amplitude * (0.65 * math.sin(x_angle) + 0.35 * math.cos(y_angle))
    )
    derivative_x = (
        amplitude * 0.65 * math.cos(x_angle) * 2.0 * math.pi / max(shape_x, 1)
    )
    derivative_y = (
        -amplitude * 0.35 * math.sin(y_angle) * 2.0 * math.pi / max(shape_y, 1)
    )
    return height, derivative_x, derivative_y


def generate_synthetic_stack(
    grid: GridSpec, settings: SyntheticStackSettings | None = None
) -> SyntheticScene:
    """Generate tangent-plane observations of smooth analytic sheet graphs."""

    resolved = settings or SyntheticStackSettings()
    rng = np.random.default_rng(resolved.random_seed)
    patches: list[ClippedPatch] = []
    truth: list[tuple[int, int]] = []
    degenerate = 0
    patch_id = 0
    cell_z_size = grid.cell_size_xyz[2]
    for sheet_index in range(resolved.sheet_count):
        for y_index in range(grid.shape_cells_xyz[1]):
            for x_index in range(grid.shape_cells_xyz[0]):
                x_grid = x_index + 0.5
                y_grid = y_index + 0.5
                height_grid, derivative_x_grid, derivative_y_grid = _surface(
                    grid, resolved, sheet_index, x_grid, y_grid
                )
                point = grid.vertex_world((x_grid, y_grid, height_grid))
                derivative_x = (
                    derivative_x_grid
                    * grid.cell_size_xyz[2]
                    / grid.cell_size_xyz[0]
                )
                derivative_y = (
                    derivative_y_grid
                    * grid.cell_size_xyz[2]
                    / grid.cell_size_xyz[1]
                )
                true_normal = np.asarray(
                    (-derivative_x, -derivative_y, 1.0), dtype=np.float64
                )
                true_normal /= float(np.linalg.norm(true_normal))
                u_axis, v_axis = plane_basis(true_normal)
                angular_noise = (
                    rng.normal(size=2)
                    * resolved.angular_standard_deviation_radians
                    * resolved.observation_noise_scale
                )
                observed_normal = (
                    true_normal
                    + angular_noise[0] * u_axis
                    + angular_noise[1] * v_axis
                )
                observed_normal /= float(np.linalg.norm(observed_normal))
                if sheet_index % 2:
                    fiber = np.asarray((0.0, 1.0, derivative_y), dtype=np.float64)
                else:
                    fiber = np.asarray((1.0, 0.0, derivative_x), dtype=np.float64)
                for z_index in range(grid.shape_cells_xyz[2]):
                    cell = (x_index, y_index, z_index)
                    cell_center = grid.cell_center_world(cell)
                    height = float(np.dot(observed_normal, point - cell_center))
                    height += float(
                        rng.normal()
                        * resolved.height_standard_deviation_cells
                        * cell_z_size
                        * resolved.observation_noise_scale
                    )
                    estimate = PlaneEstimate.isotropic(
                        observed_normal,
                        height,
                        resolved.angular_standard_deviation_radians,
                        resolved.height_standard_deviation_cells * cell_z_size,
                        fiber_xyz=fiber,
                        fiber_angular_std_radians=math.radians(2.0),
                        confidence=0.95,
                    )
                    try:
                        patch = clip_plane_to_cell(
                            grid, cell, estimate, patch_id=patch_id
                        )
                    except DegeneratePlaneIntersection:
                        degenerate += 1
                        continue
                    if patch is None:
                        continue
                    if rng.random() < resolved.missing_patch_fraction:
                        continue
                    patches.append(patch)
                    truth.append((patch_id, sheet_index))
                    patch_id += 1
    return SyntheticScene(
        grid=grid,
        settings=resolved,
        patches=tuple(patches),
        truth_sheet_by_patch=tuple(truth),
        degenerate_candidate_count=degenerate,
    )
