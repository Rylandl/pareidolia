from __future__ import annotations

import base64
import io
import json
import math
import struct
import time
import zlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np


def _gaussian_kernel(sigma: float) -> np.ndarray:
    sigma = max(0.55, float(sigma))
    radius = max(1, int(math.ceil(2.5 * sigma)))
    x = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-(x * x) / (2.0 * sigma * sigma))
    return kernel / kernel.sum()


def _convolve_axis(array: np.ndarray, kernel: np.ndarray, axis: int) -> np.ndarray:
    radius = len(kernel) // 2
    pads = [(0, 0)] * array.ndim
    pads[axis] = (radius, radius)
    padded = np.pad(array, pads, mode="edge")
    out = np.zeros_like(array, dtype=np.float32)
    for offset, weight in enumerate(kernel):
        src = [slice(None)] * array.ndim
        src[axis] = slice(offset, offset + array.shape[axis])
        out += float(weight) * padded[tuple(src)]
    return out


def gaussian_blur_3d(array: np.ndarray, sigma: float) -> np.ndarray:
    kernel = _gaussian_kernel(sigma)
    out = array.astype(np.float32, copy=False)
    for axis in range(3):
        out = _convolve_axis(out, kernel, axis)
    return out


def _trilinear(array: np.ndarray, xyz: np.ndarray, *, outside: float = 0.0) -> np.ndarray:
    """Sample a ZYX scalar volume at (..., 3) XYZ coordinates."""
    points = np.asarray(xyz, dtype=np.float32)
    flat = points.reshape(-1, 3)
    x, y, z = flat[:, 0], flat[:, 1], flat[:, 2]
    valid = (
        (x >= 0.0)
        & (x <= array.shape[2] - 1)
        & (y >= 0.0)
        & (y <= array.shape[1] - 1)
        & (z >= 0.0)
        & (z <= array.shape[0] - 1)
    )
    x0 = np.clip(np.floor(x).astype(np.int64), 0, array.shape[2] - 1)
    y0 = np.clip(np.floor(y).astype(np.int64), 0, array.shape[1] - 1)
    z0 = np.clip(np.floor(z).astype(np.int64), 0, array.shape[0] - 1)
    x1 = np.minimum(x0 + 1, array.shape[2] - 1)
    y1 = np.minimum(y0 + 1, array.shape[1] - 1)
    z1 = np.minimum(z0 + 1, array.shape[0] - 1)
    fx, fy, fz = x - x0, y - y0, z - z0

    c000 = array[z0, y0, x0]
    c001 = array[z0, y0, x1]
    c010 = array[z0, y1, x0]
    c011 = array[z0, y1, x1]
    c100 = array[z1, y0, x0]
    c101 = array[z1, y0, x1]
    c110 = array[z1, y1, x0]
    c111 = array[z1, y1, x1]
    c00 = c000 * (1 - fx) + c001 * fx
    c01 = c010 * (1 - fx) + c011 * fx
    c10 = c100 * (1 - fx) + c101 * fx
    c11 = c110 * (1 - fx) + c111 * fx
    c0 = c00 * (1 - fy) + c01 * fy
    c1 = c10 * (1 - fy) + c11 * fy
    values = c0 * (1 - fz) + c1 * fz
    values = np.where(valid, values, outside)
    return values.reshape(points.shape[:-1])


def _png_chunk(kind: bytes, payload: bytes) -> bytes:
    return (
        struct.pack(">I", len(payload))
        + kind
        + payload
        + struct.pack(">I", zlib.crc32(kind + payload) & 0xFFFFFFFF)
    )


def grayscale_png(image: np.ndarray) -> bytes:
    pixels = np.asarray(image, dtype=np.uint8)
    if pixels.ndim != 2:
        raise ValueError("grayscale_png expects a 2D array")
    height, width = pixels.shape
    rows = b"".join(b"\x00" + pixels[row].tobytes() for row in range(height))
    header = struct.pack(">IIBBBBB", width, height, 8, 0, 0, 0, 0)
    return (
        b"\x89PNG\r\n\x1a\n"
        + _png_chunk(b"IHDR", header)
        + _png_chunk(b"IDAT", zlib.compress(rows, 6))
        + _png_chunk(b"IEND", b"")
    )


def _data_png(image: np.ndarray) -> str:
    return "data:image/png;base64," + base64.b64encode(grayscale_png(image)).decode("ascii")


def synthetic_scroll(shape: tuple[int, int, int] = (144, 192, 192)) -> np.ndarray:
    """Create a deterministic curved lamination volume with alternating fiber texture."""
    z_size, y_size, x_size = shape
    z, y, x = np.indices(shape, dtype=np.float32)
    cx = (x_size - 1) * 0.5
    cy = (y_size - 1) * 0.5
    cz = (z_size - 1) * 0.5
    dz = z - cz
    dx = x - cx
    dy = y - cy

    # The slow centerline drift makes the local cylinders genuinely warped in 3D.
    warped_x = dx - 0.018 * dz - 0.0007 * dz * dz
    warped_y = dy + 0.0009 * dz * dz - 0.012 * dz
    radius = np.sqrt(warped_x * warped_x + warped_y * warped_y)
    theta = np.arctan2(warped_y, warped_x)
    phase = np.mod(radius + 0.7 * np.sin(theta * 2.0 + dz * 0.018), 8.0)
    inside_scroll = (radius >= 18.0) & (radius <= min(x_size, y_size) * 0.44)
    inside_scroll &= np.abs(dz) < z_size * 0.44
    material = inside_scroll & (phase < 5.4)

    edge = np.exp(-((phase - 0.35) / 0.65) ** 2) + np.exp(-((phase - 5.0) / 0.7) ** 2)
    ply = phase < 2.7
    arc = theta * np.maximum(radius, 1.0)
    fibers_a = 0.5 + 0.5 * np.cos(arc * 0.75 + dz * 0.045 + np.sin(dz * 0.08))
    fibers_b = 0.5 + 0.5 * np.cos(dz * 0.72 + arc * 0.035 + np.sin(arc * 0.06))
    fibers = np.where(ply, fibers_a, fibers_b)
    broad = 8.0 * np.sin(theta * 5.0 + dz * 0.027) + 5.0 * np.cos(radius * 0.12)
    values = np.where(material, 72.0 + 78.0 * edge + 32.0 * fibers + broad, 8.0)
    values += 3.0 * np.sin(x * 0.17 + y * 0.11 + z * 0.07)
    return np.clip(values, 0, 255).astype(np.uint8)


@dataclass
class VolumeData:
    array: np.ndarray
    name: str
    voxel_size: float = 1.0
    voxel_unit: str = "voxel"
    origin_xyz: tuple[int, int, int] = (0, 0, 0)
    source_kind: str = "synthetic"
    source_url: str | None = None
    source_level: int = 0
    source_shape_zyx: tuple[int, int, int] | None = None
    suggested_seed_override: dict[str, int] | None = None

    @classmethod
    def load(cls, path: str | Path | None = None) -> "VolumeData":
        if path is None:
            return cls(synthetic_scroll(), "Synthetic rolled papyrus")
        source = Path(path)
        if source.suffix.lower() != ".npy":
            raise ValueError("the pilot currently accepts a 3D .npy volume")
        array = np.load(source, mmap_mode="r")
        if array.ndim != 3:
            raise ValueError(f"expected a 3D ZYX array, received shape {array.shape}")
        sidecar_path = source.with_suffix(".json")
        sidecar: dict[str, Any] = {}
        if sidecar_path.is_file():
            sidecar = json.loads(sidecar_path.read_text())
        origin = tuple(int(value) for value in sidecar.get("originXYZ", (0, 0, 0)))
        source_shape = sidecar.get("sourceShapeZYX")
        return cls(
            array,
            str(sidecar.get("name", source.name)),
            voxel_size=float(sidecar.get("voxelSizeMicrons", 1.0)),
            voxel_unit="µm" if "voxelSizeMicrons" in sidecar else "voxel",
            origin_xyz=origin,
            source_kind=str(sidecar.get("sourceKind", "npy")),
            source_url=sidecar.get("sourceUrl"),
            source_level=int(sidecar.get("sourceLevel", 0)),
            source_shape_zyx=(tuple(int(value) for value in source_shape) if source_shape else None),
            suggested_seed_override=sidecar.get("suggestedSeed"),
        )

    def __post_init__(self) -> None:
        if self.array.ndim != 3:
            raise ValueError("volume must have ZYX dimensions")
        sample = np.asarray(self.array[:: max(1, self.array.shape[0] // 32), ::4, ::4], dtype=np.float32)
        self.low = float(np.percentile(sample, 1.0))
        self.high = float(np.percentile(sample, 99.5))
        if self.high <= self.low:
            self.high = self.low + 1.0
        self.air_threshold = self.low + 0.12 * (self.high - self.low)
        self.suggested_seed = self._find_seed()
        if self.suggested_seed_override:
            candidate = {
                axis: int(self.suggested_seed_override[axis]) for axis in ("x", "y", "z")
            }
            if all(0 <= candidate[axis] < self.shape_xyz[index] for index, axis in enumerate(("x", "y", "z"))):
                self.suggested_seed = candidate

    @property
    def shape_xyz(self) -> tuple[int, int, int]:
        z, y, x = self.array.shape
        return x, y, z

    def _find_seed(self) -> dict[str, int]:
        z = self.array.shape[0] // 2
        y = self.array.shape[1] // 2
        line = np.asarray(self.array[z, y], dtype=np.float32)
        center = self.array.shape[2] // 2
        candidates = np.where(line > self.air_threshold)[0]
        candidates = candidates[candidates > center + max(4, self.array.shape[2] // 8)]
        x = int(candidates[len(candidates) // 2]) if len(candidates) else center
        return {"x": x, "y": y, "z": z}

    def normalize(self, image: np.ndarray) -> np.ndarray:
        scaled = (np.asarray(image, dtype=np.float32) - self.low) / (self.high - self.low)
        return np.clip(scaled * 255.0, 0, 255).astype(np.uint8)

    def slice(self, axis: str, index: int) -> np.ndarray:
        axis = axis.lower()
        if axis == "z":
            if not 0 <= index < self.array.shape[0]:
                raise IndexError("z slice is outside the volume")
            image = self.array[index, :, :]
        elif axis == "y":
            if not 0 <= index < self.array.shape[1]:
                raise IndexError("y slice is outside the volume")
            image = self.array[:, index, :]
        elif axis == "x":
            if not 0 <= index < self.array.shape[2]:
                raise IndexError("x slice is outside the volume")
            image = self.array[:, :, index]
        else:
            raise ValueError("axis must be x, y, or z")
        return self.normalize(np.asarray(image))

    def _extract_cube(
        self, seed_xyz: tuple[int, int, int], size: int, maximum_size: int
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        size = int(size)
        if not 16 <= size <= maximum_size:
            raise ValueError(f"cube size must be between 16 and {maximum_size} voxels")
        sx, sy, sz = (int(value) for value in seed_xyz)
        if not (0 <= sx < self.array.shape[2] and 0 <= sy < self.array.shape[1] and 0 <= sz < self.array.shape[0]):
            raise ValueError("cube center is outside the volume")

        half = size // 2
        origin_xyz = (sx - half, sy - half, sz - half)
        x0, y0, z0 = origin_xyz
        x1, y1, z1 = x0 + size, y0 + size, z0 + size
        source_x0, source_x1 = max(0, x0), min(self.array.shape[2], x1)
        source_y0, source_y1 = max(0, y0), min(self.array.shape[1], y1)
        source_z0, source_z1 = max(0, z0), min(self.array.shape[0], z1)
        cube = np.zeros((size, size, size), dtype=np.uint8)
        cube[
            source_z0 - z0 : source_z1 - z0,
            source_y0 - y0 : source_y1 - y0,
            source_x0 - x0 : source_x1 - x0,
        ] = self.normalize(
            np.asarray(
                self.array[source_z0:source_z1, source_y0:source_y1, source_x0:source_x1]
            )
        )
        return cube, origin_xyz

    def cube(self, seed_xyz: tuple[int, int, int], size: int) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Return a normalized N³ ZYX display cube, zero-padded at loaded-data bounds."""
        return self._extract_cube(seed_xyz, size, 128)

    def context_cube(
        self, seed_xyz: tuple[int, int, int], size: int
    ) -> tuple[np.ndarray, tuple[int, int, int]]:
        """Return a strict real-data context cube for analysis, never synthetic padding."""
        size = int(size)
        sx, sy, sz = (int(value) for value in seed_xyz)
        half = size // 2
        origin = (sx - half, sy - half, sz - half)
        x0, y0, z0 = origin
        x1, y1, z1 = x0 + size, y0 + size, z0 + size
        if not (
            0 <= x0 < x1 <= self.array.shape[2]
            and 0 <= y0 < y1 <= self.array.shape[1]
            and 0 <= z0 < z1 <= self.array.shape[0]
        ):
            raise ValueError(
                "the padded Acus context extends outside the loaded volume; "
                "choose a more interior seed, smaller cube, or larger source cuboid"
            )
        return self._extract_cube(seed_xyz, size, 256)

    def metadata(self) -> dict[str, Any]:
        x, y, z = self.shape_xyz
        ox, oy, oz = self.origin_xyz
        metadata: dict[str, Any] = {
            "name": self.name,
            "shape": {"x": x, "y": y, "z": z},
            "dtype": str(self.array.dtype),
            "voxelSize": self.voxel_size,
            "voxelUnit": self.voxel_unit,
            "origin": {"x": ox, "y": oy, "z": oz},
            "suggestedSeed": self.suggested_seed,
            "globalSuggestedSeed": {
                "x": ox + self.suggested_seed["x"],
                "y": oy + self.suggested_seed["y"],
                "z": oz + self.suggested_seed["z"],
            },
            "phaseNeutral": True,
            "sourceKind": self.source_kind,
            "sourceLevel": self.source_level,
        }
        if self.source_shape_zyx:
            sz, sy, sx = self.source_shape_zyx
            metadata["sourceShape"] = {"x": sx, "y": sy, "z": sz}
        if self.source_url:
            metadata["sourceUrl"] = self.source_url
        return metadata


class LocalTensorField:
    def __init__(self, volume: VolumeData, seed: np.ndarray, radius: int, extent: float):
        margin = int(math.ceil(extent + max(5, radius * 3)))
        x, y, z = seed
        z0 = max(0, int(math.floor(z)) - margin)
        y0 = max(0, int(math.floor(y)) - margin)
        x0 = max(0, int(math.floor(x)) - margin)
        z1 = min(volume.array.shape[0], int(math.ceil(z)) + margin + 1)
        y1 = min(volume.array.shape[1], int(math.ceil(y)) + margin + 1)
        x1 = min(volume.array.shape[2], int(math.ceil(x)) + margin + 1)
        self.origin = np.array([x0, y0, z0], dtype=np.float32)
        crop = np.asarray(volume.array[z0:z1, y0:y1, x0:x1], dtype=np.float32)
        smoothed = gaussian_blur_3d(crop, max(0.8, radius * 0.38))
        gz, gy, gx = np.gradient(smoothed)
        integration_sigma = max(1.0, radius * 0.62)
        tensors = [
            gx * gx,
            gx * gy,
            gx * gz,
            gy * gy,
            gy * gz,
            gz * gz,
        ]
        self.components = [gaussian_blur_3d(value, integration_sigma) for value in tensors]

    def sample(self, global_xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        local = np.asarray(global_xyz, dtype=np.float32) - self.origin
        values = [_trilinear(component, local) for component in self.components]
        shape = local.shape[:-1]
        matrix = np.empty(shape + (3, 3), dtype=np.float32)
        matrix[..., 0, 0] = values[0]
        matrix[..., 0, 1] = matrix[..., 1, 0] = values[1]
        matrix[..., 0, 2] = matrix[..., 2, 0] = values[2]
        matrix[..., 1, 1] = values[3]
        matrix[..., 1, 2] = matrix[..., 2, 1] = values[4]
        matrix[..., 2, 2] = values[5]
        eigenvalues, eigenvectors = np.linalg.eigh(matrix)
        normals = eigenvectors[..., :, 2]
        total = np.maximum(eigenvalues.sum(axis=-1), 1.0e-6)
        confidence = np.clip((eigenvalues[..., 2] - eigenvalues[..., 1]) / total, 0.0, 1.0)
        return normals.astype(np.float32), confidence.astype(np.float32)


def _basis_from_normal(normal: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = np.asarray(normal, dtype=np.float32)
    n /= max(float(np.linalg.norm(n)), 1.0e-8)
    axis = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    v = axis - n * float(np.dot(axis, n))
    if np.linalg.norm(v) < 0.15:
        axis = np.array([0.0, 1.0, 0.0], dtype=np.float32)
        v = axis - n * float(np.dot(axis, n))
    v /= max(float(np.linalg.norm(v)), 1.0e-8)
    u = np.cross(v, n)
    u /= max(float(np.linalg.norm(u)), 1.0e-8)
    v = np.cross(n, u)
    v /= max(float(np.linalg.norm(v)), 1.0e-8)
    return u, v, n


def _mesh_normals(positions: np.ndarray, reference: np.ndarray) -> np.ndarray:
    dv = np.empty_like(positions)
    du = np.empty_like(positions)
    dv[1:-1] = positions[2:] - positions[:-2]
    dv[0] = positions[1] - positions[0]
    dv[-1] = positions[-1] - positions[-2]
    du[:, 1:-1] = positions[:, 2:] - positions[:, :-2]
    du[:, 0] = positions[:, 1] - positions[:, 0]
    du[:, -1] = positions[:, -1] - positions[:, -2]
    normals = np.cross(du, dv)
    lengths = np.linalg.norm(normals, axis=-1, keepdims=True)
    normals /= np.maximum(lengths, 1.0e-8)
    signs = np.where(np.sum(normals * reference, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
    return normals * signs


def _fit_carrier(
    field: LocalTensorField,
    seed: np.ndarray,
    patch_size: float,
    grid_size: int,
    iterations: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    n0, seed_confidence = field.sample(seed[None, None, :])
    u_axis, v_axis, n_axis = _basis_from_normal(n0[0, 0])
    if float(seed_confidence[0, 0]) < 0.012:
        raise ValueError("bulk lamination orientation is not coherent enough at this seed")

    coords = np.linspace(-patch_size * 0.5, patch_size * 0.5, grid_size, dtype=np.float32)
    vv, uu = np.meshgrid(coords, coords, indexing="ij")
    base = seed + uu[..., None] * u_axis + vv[..., None] * v_axis
    heights = np.zeros((grid_size, grid_size), dtype=np.float32)
    step = float(coords[1] - coords[0])
    center = grid_size // 2
    confidence = np.zeros_like(heights)

    for _ in range(iterations):
        positions = base + heights[..., None] * n_axis
        directors, confidence = field.sample(positions)
        signs = np.where(np.sum(directors * n_axis, axis=-1, keepdims=True) < 0.0, -1.0, 1.0)
        directors *= signs
        denominator = np.maximum(np.sum(directors * n_axis, axis=-1), 0.18)
        grad_u = -np.sum(directors * u_axis, axis=-1) / denominator
        grad_v = -np.sum(directors * v_axis, axis=-1) / denominator

        total = np.zeros_like(heights)
        count = np.zeros_like(heights)
        total[:, 1:] += heights[:, :-1] + step * 0.5 * (grad_u[:, :-1] + grad_u[:, 1:])
        count[:, 1:] += 1.0
        total[:, :-1] += heights[:, 1:] - step * 0.5 * (grad_u[:, :-1] + grad_u[:, 1:])
        count[:, :-1] += 1.0
        total[1:] += heights[:-1] + step * 0.5 * (grad_v[:-1] + grad_v[1:])
        count[1:] += 1.0
        total[:-1] += heights[1:] - step * 0.5 * (grad_v[:-1] + grad_v[1:])
        count[:-1] += 1.0
        proposal = total / np.maximum(count, 1.0)
        proposal -= proposal[center, center]
        proposal = np.clip(proposal, -patch_size * 0.42, patch_size * 0.42)
        heights = 0.52 * heights + 0.48 * proposal
        heights[center, center] = 0.0

    positions = base + heights[..., None] * n_axis
    mesh_normals = _mesh_normals(positions, n_axis)
    directors, confidence = field.sample(positions)
    alignment = np.abs(np.sum(mesh_normals * directors, axis=-1))
    return positions, mesh_normals, confidence * alignment, (u_axis, v_axis, n_axis)


def _point_cross_sections(volume: VolumeData, seed: np.ndarray, radius: int, stride: int = 3) -> list[list[float]]:
    sx, sy, sz = [int(round(value)) for value in seed]
    x0, x1 = max(0, sx - radius), min(volume.array.shape[2] - 1, sx + radius)
    y0, y1 = max(0, sy - radius), min(volume.array.shape[1] - 1, sy + radius)
    z0, z1 = max(0, sz - radius), min(volume.array.shape[0] - 1, sz + radius)
    points: list[list[float]] = []

    def add(x: int, y: int, z: int, plane: int) -> None:
        value = float(volume.array[z, y, x])
        normalized = float(np.clip((value - volume.low) / (volume.high - volume.low), 0.0, 1.0))
        if normalized > 0.055:
            points.append([float(x), float(y), float(z), normalized, float(plane)])

    for y in range(y0, y1 + 1, stride):
        for x in range(x0, x1 + 1, stride):
            add(x, y, sz, 0)
    for z in range(z0, z1 + 1, stride):
        for x in range(x0, x1 + 1, stride):
            add(x, sy, z, 1)
    for z in range(z0, z1 + 1, stride):
        for y in range(y0, y1 + 1, stride):
            add(sx, y, z, 2)
    return points


def fit_local_chart(volume: VolumeData, request: dict[str, Any]) -> dict[str, Any]:
    started = time.perf_counter()
    seed_obj = request.get("seed") or {}
    seed = np.array(
        [float(seed_obj.get("x", 0)), float(seed_obj.get("y", 0)), float(seed_obj.get("z", 0))],
        dtype=np.float32,
    )
    shape = np.array(volume.shape_xyz, dtype=np.float32)
    if np.any(seed < 1.0) or np.any(seed > shape - 2.0):
        raise ValueError("seed must be inside the volume with a one-voxel margin")
    seed_value = float(_trilinear(volume.array, seed[None, :])[0])
    if seed_value <= volume.air_threshold:
        raise ValueError("the selected seed appears to be air; choose a papyrus voxel")

    patch_size = float(np.clip(request.get("patchSize", 72.0), 16.0, 192.0))
    grid_size = int(np.clip(request.get("gridSize", 33), 11, 65))
    if grid_size % 2 == 0:
        grid_size += 1
    depth = float(np.clip(request.get("depth", 32.0), 4.0, 96.0))
    depth_samples = int(np.clip(request.get("depthSamples", 33), 5, 65))
    if depth_samples % 2 == 0:
        depth_samples += 1
    field_radius = int(np.clip(request.get("fieldRadius", 5), 2, 12))
    iterations = int(np.clip(request.get("iterations", 36), 4, 100))

    extent = patch_size * 0.75 + depth * 0.55
    field = LocalTensorField(volume, seed, field_radius, extent)
    positions, normals, confidence, basis = _fit_carrier(
        field, seed, patch_size, grid_size, iterations
    )
    offsets = np.linspace(-depth * 0.5, depth * 0.5, depth_samples, dtype=np.float32)
    sample_xyz = positions[None, ...] + offsets[:, None, None, None] * normals[None, ...]
    rectified = _trilinear(volume.array, sample_xyz)
    center = depth_samples // 2
    row = grid_size // 2
    col = grid_size // 2
    center_image = volume.normalize(rectified[center])
    cross_u = volume.normalize(rectified[:, row, :])
    cross_v = volume.normalize(rectified[:, :, col])

    directors, _ = field.sample(positions)
    dot = np.clip(np.abs(np.sum(normals * directors, axis=-1)), 0.0, 1.0)
    angle = np.degrees(np.arccos(dot))
    mesh_positions = positions.reshape(-1, 3)
    mesh_confidence = confidence.reshape(-1)
    u_axis, v_axis, n_axis = basis
    elapsed = (time.perf_counter() - started) * 1000.0
    global_seed = seed + np.asarray(volume.origin_xyz, dtype=np.float32)
    return {
        "seed": {"x": float(seed[0]), "y": float(seed[1]), "z": float(seed[2])},
        "globalSeed": {
            "x": float(global_seed[0]),
            "y": float(global_seed[1]),
            "z": float(global_seed[2]),
        },
        "mesh": {
            "rows": grid_size,
            "cols": grid_size,
            "positions": np.round(mesh_positions, 4).tolist(),
            "confidence": np.round(mesh_confidence, 4).tolist(),
        },
        "pointCloud": _point_cross_sections(volume, seed, int(min(48.0, patch_size * 0.52))),
        "rectified": {
            "center": _data_png(center_image),
            "crossU": _data_png(cross_u),
            "crossV": _data_png(cross_v),
        },
        "stats": {
            "elapsedMs": round(elapsed, 1),
            "meanConfidence": round(float(np.mean(confidence)), 4),
            "p95NormalErrorDeg": round(float(np.percentile(angle, 95.0)), 2),
            "maxCarrierOffset": round(float(np.max(np.abs(np.sum((positions - seed) * n_axis, axis=-1)))), 2),
            "seedValue": round(seed_value, 2),
            "gridSize": grid_size,
            "depthSamples": depth_samples,
            "basis": {
                "u": np.round(u_axis, 5).tolist(),
                "v": np.round(v_axis, 5).tolist(),
                "n": np.round(n_axis, 5).tolist(),
            },
            "claim": "phase-neutral carrier chart; no layer identity assigned",
        },
    }
