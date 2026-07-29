"use client";

import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type KeyboardEvent as ReactKeyboardEvent,
  type PointerEvent as ReactPointerEvent,
} from "react";
import { VolumeScene } from "./VolumeScene";

type Axis = "x" | "y" | "z";
type Seed = { x: number; y: number; z: number };
type Dimensions = { x: number; y: number; z: number };
type ConnectionState = "connecting" | "connected" | "error";
type SliceView = { zoom: number; centerX: number; centerY: number };
type VolumeInfo = {
  name: string;
  origin: Seed;
  voxelSize: number;
  voxelUnit: string;
  sourceKind: string;
};

type Mesh = {
  rows: number;
  cols: number;
  positions: number[][];
  confidence: number[];
};

type RectifiedImages = {
  center: string;
  crossU: string;
  crossV: string;
};

type FitResult = {
  seed: Seed;
  globalSeed?: Seed;
  mesh: Mesh;
  pointCloud: number[][];
  rectified: RectifiedImages;
  stats: Record<string, unknown>;
};

type FitParameters = {
  patchSize: number;
  gridSize: number;
  depth: number;
  depthSamples: number;
  fieldRadius: number;
  iterations: number;
};

type SliceDefinition = {
  axis: Axis;
  title: string;
  code: string;
  horizontal: Axis;
  vertical: Axis;
};

const DEFAULT_DIMENSIONS: Dimensions = { x: 256, y: 256, z: 192 };
const DEFAULT_SEED: Seed = { x: 128, y: 128, z: 96 };
const DEFAULT_VOLUME_INFO: VolumeInfo = {
  name: "Volume",
  origin: { x: 0, y: 0, z: 0 },
  voxelSize: 1,
  voxelUnit: "voxel",
  sourceKind: "unknown",
};
const DEFAULT_PARAMETERS: FitParameters = {
  patchSize: 72,
  gridSize: 33,
  depth: 32,
  depthSamples: 33,
  fieldRadius: 5,
  iterations: 36,
};

const SLICE_DEFINITIONS: SliceDefinition[] = [
  { axis: "z", title: "Axial", code: "XY · Z", horizontal: "x", vertical: "y" },
  { axis: "y", title: "Coronal", code: "XZ · Y", horizontal: "x", vertical: "z" },
  { axis: "x", title: "Sagittal", code: "YZ · X", horizontal: "y", vertical: "z" },
];

const PARAMETER_DEFINITIONS: Array<{
  key: keyof FitParameters;
  label: string;
  detail: string;
  min: number;
  max: number;
  step: number;
}> = [
  {
    key: "patchSize",
    label: "Patch size",
    detail: "Physical width of the local chart",
    min: 16,
    max: 192,
    step: 8,
  },
  {
    key: "gridSize",
    label: "Grid size",
    detail: "Mesh samples along each edge",
    min: 11,
    max: 65,
    step: 2,
  },
  {
    key: "depth",
    label: "Sampling depth",
    detail: "Total slab thickness in voxels",
    min: 4,
    max: 96,
    step: 2,
  },
  {
    key: "depthSamples",
    label: "Depth samples",
    detail: "Planes sampled through the chart",
    min: 5,
    max: 65,
    step: 2,
  },
  {
    key: "fieldRadius",
    label: "Field radius",
    detail: "Neighborhood for bulk orientation",
    min: 2,
    max: 12,
    step: 1,
  },
  {
    key: "iterations",
    label: "Fit iterations",
    detail: "Surface relaxation budget",
    min: 4,
    max: 100,
    step: 4,
  },
];

function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
}

function coordinateLimit(dimensions: Dimensions, axis: Axis) {
  return Math.max(0, Math.floor(dimensions[axis]) - 1);
}

function clampSeed(seed: Seed, dimensions: Dimensions): Seed {
  return {
    x: clamp(Math.round(seed.x), 0, coordinateLimit(dimensions, "x")),
    y: clamp(Math.round(seed.y), 0, coordinateLimit(dimensions, "y")),
    z: clamp(Math.round(seed.z), 0, coordinateLimit(dimensions, "z")),
  };
}

function readDimensions(payload: unknown): Dimensions {
  if (!payload || typeof payload !== "object") return DEFAULT_DIMENSIONS;
  const body = payload as Record<string, unknown>;
  const candidate = body.shape ?? body.dimensions ?? body.size;

  if (Array.isArray(candidate) && candidate.length >= 3) {
    const [x, y, z] = candidate.map(Number);
    if ([x, y, z].every((value) => Number.isFinite(value) && value > 0)) {
      return { x, y, z };
    }
  }

  if (candidate && typeof candidate === "object") {
    const dimensions = candidate as Record<string, unknown>;
    const x = Number(dimensions.x ?? dimensions.width);
    const y = Number(dimensions.y ?? dimensions.height);
    const z = Number(dimensions.z ?? dimensions.depth);
    if ([x, y, z].every((value) => Number.isFinite(value) && value > 0)) {
      return { x, y, z };
    }
  }

  return DEFAULT_DIMENSIONS;
}

function readSuggestedSeed(payload: unknown, dimensions: Dimensions): Seed | null {
  if (!payload || typeof payload !== "object") return null;
  const body = payload as Record<string, unknown>;
  const candidate = body.suggestedSeed ?? body.suggested_seed ?? body.seed;
  if (!candidate || typeof candidate !== "object") return null;
  const seed = candidate as Record<string, unknown>;
  const x = Number(seed.x);
  const y = Number(seed.y);
  const z = Number(seed.z);
  if (![x, y, z].every(Number.isFinite)) return null;
  return clampSeed({ x, y, z }, dimensions);
}

function readVolumeInfo(payload: unknown): VolumeInfo {
  if (!payload || typeof payload !== "object") return DEFAULT_VOLUME_INFO;
  const body = payload as Record<string, unknown>;
  const originBody =
    body.origin && typeof body.origin === "object"
      ? (body.origin as Record<string, unknown>)
      : {};
  const origin = {
    x: Number(originBody.x) || 0,
    y: Number(originBody.y) || 0,
    z: Number(originBody.z) || 0,
  };
  const voxelSize = Number(body.voxelSize);
  return {
    name: typeof body.name === "string" ? body.name : DEFAULT_VOLUME_INFO.name,
    origin,
    voxelSize: Number.isFinite(voxelSize) && voxelSize > 0 ? voxelSize : 1,
    voxelUnit: typeof body.voxelUnit === "string" ? body.voxelUnit : "voxel",
    sourceKind: typeof body.sourceKind === "string" ? body.sourceKind : "unknown",
  };
}

function imageSource(value: string, apiBase: string) {
  if (!value) return "";
  if (/^(data:|blob:|https?:\/\/)/i.test(value)) return value;
  if (value.startsWith("/")) return `${apiBase}${value}`;
  return `data:image/png;base64,${value}`;
}

function formatStat(value: unknown) {
  if (typeof value === "number") {
    if (!Number.isFinite(value)) return "—";
    if (Math.abs(value) >= 1000 || (Math.abs(value) > 0 && Math.abs(value) < 0.01)) {
      return value.toExponential(2);
    }
    return Number.isInteger(value) ? String(value) : value.toFixed(3);
  }
  if (typeof value === "boolean") return value ? "yes" : "no";
  if (typeof value === "string") return value;
  return JSON.stringify(value);
}

function statusMessage(error: unknown) {
  return error instanceof Error ? error.message : "The backend returned an unknown error.";
}

function sliceUrl(apiBase: string, axis: Axis, index: number) {
  const query = new URLSearchParams({ axis, index: String(index) });
  return `${apiBase}/api/slice?${query.toString()}`;
}

function SliceCanvas({
  definition,
  dimensions,
  seed,
  apiBase,
  onSeedChange,
}: {
  definition: SliceDefinition;
  dimensions: Dimensions;
  seed: Seed;
  apiBase: string;
  onSeedChange: (nextSeed: Seed) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const imageRef = useRef<HTMLImageElement | null>(null);
  const [imageState, setImageState] = useState<"loading" | "ready" | "error">(
    "loading",
  );
  const [renderTick, setRenderTick] = useState(0);
  const [view, setView] = useState<SliceView>({ zoom: 1, centerX: 0.5, centerY: 0.5 });
  const index = seed[definition.axis];
  const horizontalSize = dimensions[definition.horizontal];
  const verticalSize = dimensions[definition.vertical];
  const url = apiBase ? sliceUrl(apiBase, definition.axis, index) : "";

  useEffect(() => {
    if (!url) return;
    let active = true;
    const image = new Image();
    setImageState("loading");
    image.onload = () => {
      if (!active) return;
      imageRef.current = image;
      setImageState("ready");
      setRenderTick((value) => value + 1);
    };
    image.onerror = () => {
      if (!active) return;
      imageRef.current = null;
      setImageState("error");
      setRenderTick((value) => value + 1);
    };
    image.src = url;
    return () => {
      active = false;
      image.onload = null;
      image.onerror = null;
    };
  }, [url]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);
      context.fillStyle = "#142225";
      context.fillRect(0, 0, bounds.width, bounds.height);

      const image = imageRef.current;
      if (image) {
        context.imageSmoothingEnabled = true;
        const visibleSpan = 1 / view.zoom;
        const sourceLeft = view.centerX - visibleSpan * 0.5;
        const sourceTop = view.centerY - visibleSpan * 0.5;
        context.drawImage(
          image,
          sourceLeft * image.naturalWidth,
          sourceTop * image.naturalHeight,
          visibleSpan * image.naturalWidth,
          visibleSpan * image.naturalHeight,
          0,
          0,
          bounds.width,
          bounds.height,
        );
      } else {
        const gradient = context.createLinearGradient(0, 0, bounds.width, bounds.height);
        gradient.addColorStop(0, "#172b2e");
        gradient.addColorStop(1, "#0d181a");
        context.fillStyle = gradient;
        context.fillRect(0, 0, bounds.width, bounds.height);
        context.fillStyle = "rgba(230, 235, 230, 0.56)";
        context.font = "12px ui-monospace, monospace";
        context.textAlign = "center";
        context.fillText(
          imageState === "error" ? "slice unavailable" : "loading slice…",
          bounds.width / 2,
          bounds.height / 2,
        );
      }

      const horizontal = seed[definition.horizontal] / Math.max(1, horizontalSize - 1);
      const vertical = seed[definition.vertical] / Math.max(1, verticalSize - 1);
      const visibleSpan = 1 / view.zoom;
      const crossX =
        ((horizontal - (view.centerX - visibleSpan * 0.5)) / visibleSpan) * bounds.width;
      const crossY =
        ((vertical - (view.centerY - visibleSpan * 0.5)) / visibleSpan) * bounds.height;

      context.save();
      context.strokeStyle = "rgba(241, 151, 76, 0.96)";
      context.lineWidth = 1;
      context.setLineDash([5, 4]);
      context.beginPath();
      context.moveTo(crossX, 0);
      context.lineTo(crossX, bounds.height);
      context.moveTo(0, crossY);
      context.lineTo(bounds.width, crossY);
      context.stroke();
      context.setLineDash([]);
      context.fillStyle = "#f1974c";
      context.strokeStyle = "rgba(15, 29, 31, 0.9)";
      context.lineWidth = 2;
      context.beginPath();
      context.arc(crossX, crossY, 4.5, 0, Math.PI * 2);
      context.fill();
      context.stroke();
      context.restore();
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [
    definition.horizontal,
    definition.vertical,
    horizontalSize,
    imageState,
    renderTick,
    seed,
    verticalSize,
    view,
  ]);

  const selectAt = useCallback(
    (horizontalFraction: number, verticalFraction: number) => {
      const next = { ...seed };
      next[definition.horizontal] = clamp(
        Math.round(horizontalFraction * Math.max(0, horizontalSize - 1)),
        0,
        Math.max(0, horizontalSize - 1),
      );
      next[definition.vertical] = clamp(
        Math.round(verticalFraction * Math.max(0, verticalSize - 1)),
        0,
        Math.max(0, verticalSize - 1),
      );
      onSeedChange(next);
    },
    [definition.horizontal, definition.vertical, horizontalSize, onSeedChange, seed, verticalSize],
  );

  const handlePointer = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const bounds = event.currentTarget.getBoundingClientRect();
    const screenX = clamp((event.clientX - bounds.left) / Math.max(1, bounds.width), 0, 1);
    const screenY = clamp((event.clientY - bounds.top) / Math.max(1, bounds.height), 0, 1);
    const visibleSpan = 1 / view.zoom;
    selectAt(
      clamp(view.centerX + (screenX - 0.5) * visibleSpan, 0, 1),
      clamp(view.centerY + (screenY - 0.5) * visibleSpan, 0, 1),
    );
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      setView((value) => {
        if (event.ctrlKey || event.metaKey) {
          const pointerX = clamp((event.clientX - bounds.left) / bounds.width, 0, 1);
          const pointerY = clamp((event.clientY - bounds.top) / bounds.height, 0, 1);
          const oldSpan = 1 / value.zoom;
          const worldX = value.centerX + (pointerX - 0.5) * oldSpan;
          const worldY = value.centerY + (pointerY - 0.5) * oldSpan;
          const zoom = clamp(value.zoom * Math.exp(-event.deltaY * 0.018), 1, 32);
          const newSpan = 1 / zoom;
          const halfSpan = newSpan * 0.5;
          return {
            zoom,
            centerX: clamp(worldX - (pointerX - 0.5) * newSpan, halfSpan, 1 - halfSpan),
            centerY: clamp(worldY - (pointerY - 0.5) * newSpan, halfSpan, 1 - halfSpan),
          };
        }

        const span = 1 / value.zoom;
        const halfSpan = span * 0.5;
        return {
          ...value,
          centerX: clamp(
            value.centerX + (event.deltaX / Math.max(1, bounds.width)) * span,
            halfSpan,
            1 - halfSpan,
          ),
          centerY: clamp(
            value.centerY + (event.deltaY / Math.max(1, bounds.height)) * span,
            halfSpan,
            1 - halfSpan,
          ),
        };
      });
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

  const handleKeyDown = (event: ReactKeyboardEvent<HTMLCanvasElement>) => {
    const next = { ...seed };
    if (event.key === "ArrowLeft") next[definition.horizontal] -= 1;
    else if (event.key === "ArrowRight") next[definition.horizontal] += 1;
    else if (event.key === "ArrowUp") next[definition.vertical] -= 1;
    else if (event.key === "ArrowDown") next[definition.vertical] += 1;
    else return;
    event.preventDefault();
    onSeedChange(clampSeed(next, dimensions));
  };

  return (
    <section className="viewport-card" aria-labelledby={`${definition.axis}-view-title`}>
      <div className="viewport-header">
        <div className="viewport-title">
          <h3 id={`${definition.axis}-view-title`}>{definition.title}</h3>
          <span className="plane-code">{definition.code}</span>
        </div>
        <div className="slice-control">
          <label htmlFor={`${definition.axis}-slice`}>
            {definition.title} slice, {definition.axis.toUpperCase()} coordinate
          </label>
          <input
            id={`${definition.axis}-slice`}
            type="range"
            min={0}
            max={coordinateLimit(dimensions, definition.axis)}
            value={index}
            onChange={(event) =>
              onSeedChange(
                clampSeed(
                  { ...seed, [definition.axis]: Number(event.target.value) },
                  dimensions,
                ),
              )
            }
          />
          <output className="coordinate-value" htmlFor={`${definition.axis}-slice`}>
            {definition.axis.toUpperCase()} {index}
          </output>
        </div>
      </div>
      <div className="slice-stage">
        <canvas
          ref={canvasRef}
          tabIndex={0}
          role="img"
          aria-label={`${definition.title} scan slice. Click to set ${definition.horizontal.toUpperCase()} and ${definition.vertical.toUpperCase()}; use arrow keys for fine adjustment.`}
          onPointerDown={handlePointer}
          onKeyDown={handleKeyDown}
        />
        <p className="canvas-note">
          {definition.horizontal.toUpperCase()} horizontal · {definition.vertical.toUpperCase()} vertical
          {" · pinch zoom · two-finger pan"}
        </p>
        <div className="canvas-view-controls">
          <span>{view.zoom.toFixed(1)}×</span>
          <button
            className="quiet-button"
            type="button"
            onClick={() => setView({ zoom: 1, centerX: 0.5, centerY: 0.5 })}
            disabled={view.zoom === 1 && view.centerX === 0.5 && view.centerY === 0.5}
          >
            Reset
          </button>
        </div>
      </div>
    </section>
  );
}

type ProjectedPoint = { x: number; y: number; depth: number; scale: number };

function SurfaceScene({
  dimensions,
  seed,
  result,
  patchSize,
}: {
  dimensions: Dimensions;
  seed: Seed;
  result: FitResult | null;
  patchSize: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const [orbit, setOrbit] = useState({ yaw: -0.68, pitch: 0.48, zoom: 1 });

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;

    const draw = () => {
      const bounds = canvas.getBoundingClientRect();
      if (!bounds.width || !bounds.height) return;
      const ratio = Math.min(window.devicePixelRatio || 1, 2);
      canvas.width = Math.round(bounds.width * ratio);
      canvas.height = Math.round(bounds.height * ratio);
      const context = canvas.getContext("2d");
      if (!context) return;
      context.setTransform(ratio, 0, 0, ratio, 0, 0);
      context.clearRect(0, 0, bounds.width, bounds.height);

      const background = context.createRadialGradient(
        bounds.width * 0.55,
        bounds.height * 0.42,
        0,
        bounds.width * 0.5,
        bounds.height * 0.5,
        Math.max(bounds.width, bounds.height) * 0.72,
      );
      background.addColorStop(0, "#203638");
      background.addColorStop(1, "#0c1719");
      context.fillStyle = background;
      context.fillRect(0, 0, bounds.width, bounds.height);

      const extent = Math.max(dimensions.x, dimensions.y, dimensions.z, 1);
      const span = Math.min(bounds.width, bounds.height) * 0.72 * orbit.zoom;
      const cosYaw = Math.cos(orbit.yaw);
      const sinYaw = Math.sin(orbit.yaw);
      const cosPitch = Math.cos(orbit.pitch);
      const sinPitch = Math.sin(orbit.pitch);

      const project = (point: number[]): ProjectedPoint => {
        const x = (point[0] - seed.x) / extent;
        const y = (point[1] - seed.y) / extent;
        const z = (point[2] - seed.z) / extent;
        const yawX = cosYaw * x + sinYaw * z;
        const yawZ = -sinYaw * x + cosYaw * z;
        const pitchY = cosPitch * y - sinPitch * yawZ;
        const pitchZ = sinPitch * y + cosPitch * yawZ;
        const perspective = 1 / Math.max(0.62, 1.55 - pitchZ);
        return {
          x: bounds.width / 2 + yawX * span * perspective,
          y: bounds.height / 2 - pitchY * span * perspective,
          depth: pitchZ,
          scale: perspective,
        };
      };

      const radius = Math.min(patchSize * 0.62, extent * 0.38);
      const planeDefinitions = [
        {
          points: [
            [seed.x - radius, seed.y - radius, seed.z],
            [seed.x + radius, seed.y - radius, seed.z],
            [seed.x + radius, seed.y + radius, seed.z],
            [seed.x - radius, seed.y + radius, seed.z],
          ],
          fill: "rgba(216, 123, 49, 0.08)",
          stroke: "rgba(235, 148, 78, 0.40)",
        },
        {
          points: [
            [seed.x - radius, seed.y, seed.z - radius],
            [seed.x + radius, seed.y, seed.z - radius],
            [seed.x + radius, seed.y, seed.z + radius],
            [seed.x - radius, seed.y, seed.z + radius],
          ],
          fill: "rgba(39, 151, 155, 0.07)",
          stroke: "rgba(69, 177, 181, 0.33)",
        },
        {
          points: [
            [seed.x, seed.y - radius, seed.z - radius],
            [seed.x, seed.y + radius, seed.z - radius],
            [seed.x, seed.y + radius, seed.z + radius],
            [seed.x, seed.y - radius, seed.z + radius],
          ],
          fill: "rgba(133, 174, 165, 0.05)",
          stroke: "rgba(133, 174, 165, 0.27)",
        },
      ];

      const projectedPlanes = planeDefinitions
        .map((plane) => {
          const points = plane.points.map(project);
          return {
            ...plane,
            points,
            depth: points.reduce((total, point) => total + point.depth, 0) / points.length,
          };
        })
        .sort((a, b) => a.depth - b.depth);

      for (const plane of projectedPlanes) {
        context.beginPath();
        plane.points.forEach((point, index) => {
          if (index === 0) context.moveTo(point.x, point.y);
          else context.lineTo(point.x, point.y);
        });
        context.closePath();
        context.fillStyle = plane.fill;
        context.fill();
        context.strokeStyle = plane.stroke;
        context.lineWidth = 1;
        context.stroke();
      }

      const meshPositions = (result?.mesh.positions ?? []).filter(
        (point) => Array.isArray(point) && point.length >= 3 && point.every(Number.isFinite),
      );
      if (result && meshPositions.length >= result.mesh.rows * result.mesh.cols) {
        const cells: Array<{
          points: ProjectedPoint[];
          depth: number;
          confidence: number;
        }> = [];
        for (let row = 0; row < result.mesh.rows - 1; row += 1) {
          for (let col = 0; col < result.mesh.cols - 1; col += 1) {
            const indices = [
              row * result.mesh.cols + col,
              row * result.mesh.cols + col + 1,
              (row + 1) * result.mesh.cols + col + 1,
              (row + 1) * result.mesh.cols + col,
            ];
            const points = indices.map((index) => project(meshPositions[index]));
            const confidence = indices.reduce(
              (sum, index) => sum + (Number(result.mesh.confidence[index]) || 0),
              0,
            ) / indices.length;
            cells.push({
              points,
              depth: points.reduce((sum, point) => sum + point.depth, 0) / points.length,
              confidence,
            });
          }
        }
        cells.sort((a, b) => a.depth - b.depth);
        for (const cell of cells) {
          context.beginPath();
          cell.points.forEach((point, index) => {
            if (index === 0) context.moveTo(point.x, point.y);
            else context.lineTo(point.x, point.y);
          });
          context.closePath();
          const confidence = clamp(cell.confidence, 0, 1);
          context.fillStyle = `rgba(58, 166, 159, ${0.15 + confidence * 0.32})`;
          context.fill();
          context.strokeStyle = `rgba(186, 225, 214, ${0.22 + confidence * 0.38})`;
          context.lineWidth = 0.65;
          context.stroke();
        }
      }

      const cloud = (result?.pointCloud ?? [])
        .filter((point) => Array.isArray(point) && point.length >= 3 && point.every(Number.isFinite))
        .filter((_, index, source) => index % Math.max(1, Math.ceil(source.length / 1800)) === 0)
        .map((point) => ({ source: point, projected: project(point) }))
        .sort((a, b) => a.projected.depth - b.projected.depth);

      for (const point of cloud) {
        const intensity = Number(point.source[3]);
        const alpha = Number.isFinite(intensity)
          ? 0.18 + clamp(intensity > 1 ? intensity / 255 : intensity, 0, 1) * 0.54
          : 0.42;
        context.fillStyle = `rgba(222, 220, 197, ${alpha})`;
        const size = clamp(1.25 * point.projected.scale, 0.7, 2.2);
        context.fillRect(point.projected.x - size / 2, point.projected.y - size / 2, size, size);
      }

      const projectedSeed = project([seed.x, seed.y, seed.z]);
      context.beginPath();
      context.arc(projectedSeed.x, projectedSeed.y, 7, 0, Math.PI * 2);
      context.fillStyle = "rgba(241, 151, 76, 0.18)";
      context.fill();
      context.beginPath();
      context.arc(projectedSeed.x, projectedSeed.y, 3.2, 0, Math.PI * 2);
      context.fillStyle = "#f1974c";
      context.fill();

      context.fillStyle = "rgba(228, 236, 230, 0.66)";
      context.font = "10px ui-monospace, monospace";
      context.textAlign = "left";
      context.fillText(result ? "FIT SURFACE + LOCAL SAMPLE" : "CROSS-SECTIONS AT SEED", 12, 19);
    };

    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [dimensions, orbit, patchSize, result, seed]);

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = { x: event.clientX, y: event.clientY };
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    if (!dragRef.current) return;
    const deltaX = event.clientX - dragRef.current.x;
    const deltaY = event.clientY - dragRef.current.y;
    dragRef.current = { x: event.clientX, y: event.clientY };
    setOrbit((value) => ({
      ...value,
      yaw: value.yaw + deltaX * 0.009,
      pitch: clamp(value.pitch + deltaY * 0.009, -1.35, 1.35),
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const sensitivity = event.ctrlKey || event.metaKey ? 0.018 : 0.002;
      setOrbit((value) => ({
        ...value,
        zoom: clamp(value.zoom * Math.exp(-event.deltaY * sensitivity), 0.35, 16),
      }));
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

  const resetOrbit = () => setOrbit({ yaw: -0.68, pitch: 0.48, zoom: 1 });

  return (
    <section className="viewport-card" aria-labelledby="surface-view-title">
      <div className="viewport-header">
        <div className="viewport-title">
          <h3 id="surface-view-title">Local geometry</h3>
          <span className="plane-code">3D</span>
        </div>
        <div className="scene-toolbar">
          <span className="coordinate-value">
            drag to orbit · pinch/wheel zoom · {orbit.zoom.toFixed(1)}×
          </span>
          <button className="quiet-button" type="button" onClick={resetOrbit}>
            Reset view
          </button>
        </div>
      </div>
      <div className="scene-stage">
        <canvas
          ref={canvasRef}
          tabIndex={0}
          role="img"
          aria-label="Orbitable 3D preview of the seed cross-sections and fitted local surface"
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
      </div>
    </section>
  );
}

function FitResults({ result, apiBase }: { result: FitResult | null; apiBase: string }) {
  if (!result) {
    return (
      <p className="empty-result">
        No surface has been fit. Choose a non-air voxel in any slice, then run the local chart fit.
      </p>
    );
  }

  const stats = Object.entries(result.stats ?? {}).slice(0, 6);
  const images = [
    ["Center", result.rectified?.center],
    ["Across U", result.rectified?.crossU],
    ["Across V", result.rectified?.crossV],
  ] as const;

  return (
    <>
      {stats.length > 0 && (
        <div className="stat-grid" aria-label="Fit statistics">
          {stats.map(([label, value]) => (
            <div className="stat-cell" key={label}>
              <span>{label.replaceAll("_", " ")}</span>
              <strong title={formatStat(value)}>{formatStat(value)}</strong>
            </div>
          ))}
        </div>
      )}
      <div className="rectified-strip" aria-label="Rectified patch previews">
        {images.map(([label, value]) => (
          <div className="rectified-item" key={label}>
            <span className="rectified-label">{label}</span>
            {value ? (
              // The API may return an absolute URL, an API-relative path, or PNG base64.
              // eslint-disable-next-line @next/next/no-img-element
              <img src={imageSource(value, apiBase)} alt={`${label} rectified patch`} />
            ) : (
              <div aria-label={`${label} preview unavailable`} />
            )}
          </div>
        ))}
      </div>
    </>
  );
}

export function RectifierWorkbench() {
  const [apiBase, setApiBase] = useState("");
  const [connection, setConnection] = useState<ConnectionState>("connecting");
  const [connectionDetail, setConnectionDetail] = useState("Locating volume service…");
  const [dimensions, setDimensions] = useState(DEFAULT_DIMENSIONS);
  const [volumeInfo, setVolumeInfo] = useState(DEFAULT_VOLUME_INFO);
  const [seed, setSeed] = useState(DEFAULT_SEED);
  const [parameters, setParameters] = useState(DEFAULT_PARAMETERS);
  const [fitResult, setFitResult] = useState<FitResult | null>(null);
  const [fitState, setFitState] = useState<"idle" | "running" | "success" | "error">(
    "idle",
  );
  const [fitMessage, setFitMessage] = useState(
    "The fitter follows bulk curvature without assigning a papyrus layer.",
  );
  const fitAbortRef = useRef<AbortController | null>(null);

  useEffect(() => {
    setApiBase(window.location.origin);
  }, []);

  useEffect(() => {
    if (!apiBase) return;
    const controller = new AbortController();
    setConnection("connecting");
    setConnectionDetail("Reading volume geometry…");

    Promise.all([
      fetch(`${apiBase}/health`, { signal: controller.signal }).then((response) => {
        if (!response.ok) throw new Error(`Health check returned ${response.status}`);
        return response;
      }),
      fetch(`${apiBase}/api/volume`, { signal: controller.signal }).then(async (response) => {
        if (!response.ok) throw new Error(`Volume metadata returned ${response.status}`);
        return response.json() as Promise<unknown>;
      }),
    ])
      .then(([, volume]) => {
        const nextDimensions = readDimensions(volume);
        const nextVolumeInfo = readVolumeInfo(volume);
        const suggestedSeed = readSuggestedSeed(volume, nextDimensions);
        setDimensions(nextDimensions);
        setVolumeInfo(nextVolumeInfo);
        setSeed(
          suggestedSeed ?? {
            x: Math.floor(nextDimensions.x / 2),
            y: Math.floor(nextDimensions.y / 2),
            z: Math.floor(nextDimensions.z / 2),
          },
        );
        setConnection("connected");
        setConnectionDetail(
          `${nextVolumeInfo.name} · ${nextDimensions.x} × ${nextDimensions.y} × ${nextDimensions.z}`,
        );
      })
      .catch((error: unknown) => {
        if (controller.signal.aborted) return;
        setConnection("error");
        setConnectionDetail(statusMessage(error));
      });

    return () => controller.abort();
  }, [apiBase]);

  useEffect(() => () => fitAbortRef.current?.abort(), []);

  const applySeed = useCallback(
    (nextSeed: Seed) => {
      setSeed(clampSeed(nextSeed, dimensions));
      if (fitResult) {
        setFitResult(null);
        setFitState("idle");
        setFitMessage("Seed changed. Run the fit again for this location.");
      }
    },
    [dimensions, fitResult],
  );

  const runFit = async () => {
    if (!apiBase) return;
    fitAbortRef.current?.abort();
    const controller = new AbortController();
    fitAbortRef.current = controller;
    setFitState("running");
    setFitMessage("Estimating the bulk normal field and relaxing the local chart…");

    try {
      const response = await fetch(`${apiBase}/api/fit`, {
        method: "POST",
        headers: { "content-type": "application/json" },
        body: JSON.stringify({ seed, ...parameters }),
        signal: controller.signal,
      });
      if (!response.ok) {
        let detail = "";
        try {
          const payload = (await response.json()) as Record<string, unknown>;
          detail = String(payload.detail ?? payload.error ?? "");
        } catch {
          detail = await response.text();
        }
        throw new Error(detail || `Surface fit returned ${response.status}`);
      }
      const result = (await response.json()) as FitResult;
      if (!result.mesh || !Array.isArray(result.mesh.positions)) {
        throw new Error("Surface fit response did not include a mesh.");
      }
      setFitResult(result);
      setFitState("success");
      setFitMessage(
        `Fit complete: ${result.mesh.rows} × ${result.mesh.cols} chart at (${seed.x}, ${seed.y}, ${seed.z}).`,
      );
    } catch (error: unknown) {
      if (controller.signal.aborted) return;
      setFitResult(null);
      setFitState("error");
      setFitMessage(statusMessage(error));
    } finally {
      if (fitAbortRef.current === controller) fitAbortRef.current = null;
    }
  };

  const statsLabel = useMemo(
    () =>
      `${volumeInfo.name} · ${dimensions.x} by ${dimensions.y} by ${dimensions.z} voxels · ` +
      `${formatStat(volumeInfo.voxelSize)} ${volumeInfo.voxelUnit} sampling`,
    [dimensions, volumeInfo],
  );
  const globalSeed = useMemo(
    () => ({
      x: seed.x + volumeInfo.origin.x,
      y: seed.y + volumeInfo.origin.y,
      z: seed.z + volumeInfo.origin.z,
    }),
    [seed, volumeInfo.origin],
  );

  return (
    <div className="lab-shell">
      <header className="lab-header">
        <div className="brand-lockup">
          <span className="brand-mark" aria-hidden="true" />
          <div>
            <p className="eyebrow">Local fiber-volume study</p>
            <h1>Rectifier Lab</h1>
          </div>
        </div>
        <div className="connection-block" role="status" aria-live="polite">
          <div className="connection-copy">
            <span className="connection-label">{connectionDetail}</span>
            <span className="connection-url" title={apiBase || "Resolving backend"}>
              {apiBase || "Resolving backend…"}
            </span>
          </div>
          <span className="status-dot" data-state={connection} aria-hidden="true" />
        </div>
      </header>

      <main className="lab-main">
        <div className="intro-row">
          <div>
            <h2>Compare local fiber geometry across the loaded scan.</h2>
            <p>
              Choose a voxel in the linked scan views, inspect its local N × N × N Acus fit, then
              compare that result with the normal and orientation-pattern field across the loaded
              volume. Neither workflow assigns a papyrus layer.
            </p>
          </div>
          <div
            className="seed-readout"
            aria-label={`Selected loaded-volume voxel ${seed.x}, ${seed.y}, ${seed.z}; source scan voxel ${globalSeed.x}, ${globalSeed.y}, ${globalSeed.z}`}
          >
            <span>Selected loaded-volume voxel</span>
            <strong>
              X {seed.x} · Y {seed.y} · Z {seed.z}
            </strong>
            <small>
              Scan X {globalSeed.x} · Y {globalSeed.y} · Z {globalSeed.z}
            </small>
          </div>
        </div>

        <div className="workspace-grid volume-inspection">
          <div className="viewport-grid" aria-label="Linked scan, local Acus, and loaded-volume analysis views">
            {SLICE_DEFINITIONS.map((definition) => (
              <SliceCanvas
                key={definition.axis}
                definition={definition}
                dimensions={dimensions}
                seed={seed}
                apiBase={apiBase}
                onSeedChange={applySeed}
              />
            ))}
            <VolumeScene
              apiBase={apiBase}
              dimensions={dimensions}
              seed={seed}
              onSelectSeed={applySeed}
            />
          </div>

          <aside className="sidebar" aria-label="Fit controls and results">
            <section className="control-panel" aria-labelledby="fit-controls-title">
              <div className="panel-heading">
                <h3 id="fit-controls-title">Local chart fit</h3>
                <p>{statsLabel}. Parameters are sent directly to the local fitting service.</p>
              </div>
              <div className="parameter-list">
                {PARAMETER_DEFINITIONS.map((parameter) => (
                  <div className="parameter-row" key={parameter.key}>
                    <div className="parameter-copy">
                      <label htmlFor={`parameter-${parameter.key}`}>{parameter.label}</label>
                      <small>{parameter.detail}</small>
                    </div>
                    <input
                      id={`parameter-${parameter.key}`}
                      type="number"
                      min={parameter.min}
                      max={parameter.max}
                      step={parameter.step}
                      value={parameters[parameter.key]}
                      onChange={(event) => {
                        const nextValue = clamp(
                          Number(event.target.value),
                          parameter.min,
                          parameter.max,
                        );
                        setParameters((value) => ({
                          ...value,
                          [parameter.key]: Number.isFinite(nextValue)
                            ? nextValue
                            : value[parameter.key],
                        }));
                      }}
                    />
                  </div>
                ))}
              </div>
              <div className="fit-actions">
                <button
                  className="fit-button"
                  type="button"
                  disabled={fitState === "running" || connection !== "connected"}
                  onClick={runFit}
                >
                  {fitState === "running" ? "Fitting surface…" : "Fit surface at seed"}
                </button>
                <p className="fit-status" data-kind={fitState} role="status" aria-live="polite">
                  {fitMessage}
                </p>
              </div>
            </section>

            <section className="results-panel" aria-labelledby="fit-results-title">
              <div className="panel-heading">
                <h3 id="fit-results-title">Rectified sample</h3>
                <p>Center plane and orthogonal depth cross-sections from the fitted chart.</p>
              </div>
              <div className="results-body">
                <FitResults result={fitResult} apiBase={apiBase} />
              </div>
            </section>

            <p className="method-note">
              <strong>Interpretation</strong>
              <span>
                This surface is a local coordinate frame through the seed. It is not a segmented
                papyrus layer and makes no winding assignment.
              </span>
            </p>
          </aside>
        </div>
      </main>
    </div>
  );
}
