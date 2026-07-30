"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

type Point3 = [number, number, number];
type Orbit = { yaw: number; pitch: number; zoom: number };

type BlockPatch = {
  id: string;
  component: number;
  componentSize: number;
  cell: [number, number, number];
  confidence: number;
  normal: Point3;
  vertices: Point3[];
};

type BlockComponent = {
  rank: number;
  stableId: string;
  patchCount: number;
  meanConfidence: number;
  boundsMinimumXYZ: Point3;
  boundsMaximumXYZ: Point3;
};

type BlockSheetResult = {
  schema: string;
  version: number;
  variant: string;
  grid: {
    shapeCellsXYZ: Point3;
    cellSizeXYZ: Point3;
    originXYZ: Point3;
    extentXYZ: Point3;
    coordinateUnit: string;
  };
  stats: {
    patchCount: number;
    componentCount: number;
    retainedJoinCount: number;
    largestComponentPatchCount: number;
    unresolvedInteriorTraceEndpoints: number;
    retainedInteriorTraceFraction: number;
  };
  components: BlockComponent[];
  patches: BlockPatch[];
};

type VolumePayload = {
  bytes: Uint8Array;
  shapeXYZ: Point3;
  stride: number;
  percentiles: [number, number, number, number];
};

type ClipAxis = "none" | "x" | "y" | "z";

type VolumeRenderer = {
  gl: WebGL2RenderingContext;
  program: WebGLProgram;
  texture: WebGLTexture;
  vao: WebGLVertexArrayObject;
  uniforms: {
    aspect: WebGLUniformLocation | null;
    yaw: WebGLUniformLocation | null;
    pitch: WebGLUniformLocation | null;
    zoom: WebGLUniformLocation | null;
    threshold: WebGLUniformLocation | null;
    density: WebGLUniformLocation | null;
    steps: WebGLUniformLocation | null;
    volume: WebGLUniformLocation | null;
    extent: WebGLUniformLocation | null;
    clipAxis: WebGLUniformLocation | null;
    clipFraction: WebGLUniformLocation | null;
  };
};

type ProjectedPatch = {
  patch: BlockPatch;
  points: Array<{ x: number; y: number }>;
  depth: number;
};

const DEFAULT_ORBIT: Orbit = { yaw: -0.72, pitch: 0.48, zoom: 1.35 };
const COMPONENT_SIZE_OPTIONS = [1, 2, 4, 8, 16, 32, 64, 128];

const VERTEX_SHADER = `#version 300 es
out vec2 vUv;
void main() {
  vec2 position = vec2(
    (gl_VertexID == 1) ? 3.0 : -1.0,
    (gl_VertexID == 2) ? 3.0 : -1.0
  );
  vUv = position * 0.5 + 0.5;
  gl_Position = vec4(position, 0.0, 1.0);
}`;

const FRAGMENT_SHADER = `#version 300 es
precision highp float;
precision highp sampler3D;

in vec2 vUv;
out vec4 outColor;

uniform sampler3D uVolume;
uniform float uAspect;
uniform float uYaw;
uniform float uPitch;
uniform float uZoom;
uniform float uThreshold;
uniform float uDensity;
uniform int uSteps;
uniform vec3 uExtent;
uniform int uClipAxis;
uniform float uClipFraction;

mat3 rotateX(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat3(1.0, 0.0, 0.0, 0.0, c, s, 0.0, -s, c);
}

mat3 rotateY(float angle) {
  float c = cos(angle);
  float s = sin(angle);
  return mat3(c, 0.0, -s, 0.0, 1.0, 0.0, s, 0.0, c);
}

vec2 intersectBox(vec3 origin, vec3 direction, vec3 halfExtent) {
  vec3 safeDirection = sign(direction) * max(abs(direction), vec3(0.00001));
  vec3 inverseDirection = 1.0 / safeDirection;
  vec3 t0 = (-halfExtent - origin) * inverseDirection;
  vec3 t1 = (halfExtent - origin) * inverseDirection;
  vec3 nearValues = min(t0, t1);
  vec3 farValues = max(t0, t1);
  return vec2(
    max(max(nearValues.x, nearValues.y), nearValues.z),
    min(min(farValues.x, farValues.y), farValues.z)
  );
}

bool clipped(vec3 coordinate) {
  if (uClipAxis == 0) return coordinate.x > uClipFraction;
  if (uClipAxis == 1) return coordinate.y > uClipFraction;
  if (uClipAxis == 2) return coordinate.z > uClipFraction;
  return false;
}

void main() {
  vec2 screen = vUv * 2.0 - 1.0;
  screen.x *= uAspect;
  screen /= max(uZoom, 0.05);

  vec3 rayOrigin = vec3(0.0, 0.0, 1.65);
  vec3 rayDirection = normalize(vec3(screen, -1.55));
  mat3 cameraRotation = rotateY(uYaw) * rotateX(uPitch);
  rayOrigin = cameraRotation * rayOrigin;
  rayDirection = cameraRotation * rayDirection;

  vec3 background = mix(vec3(0.025, 0.055, 0.06), vec3(0.065, 0.115, 0.12), vUv.y);
  vec3 halfExtent = uExtent * 0.5;
  vec2 bounds = intersectBox(rayOrigin, rayDirection, halfExtent);
  float start = max(bounds.x, 0.0);
  if (bounds.y <= start || uDensity <= 0.0) {
    outColor = vec4(background, 1.0);
    return;
  }

  float stepLength = (bounds.y - start) / float(max(uSteps, 1));
  vec4 accumulated = vec4(0.0);
  for (int index = 0; index < 320; index += 1) {
    if (index >= uSteps || accumulated.a > 0.985) break;
    float distanceAlongRay = start + (float(index) + 0.5) * stepLength;
    vec3 position = rayOrigin + rayDirection * distanceAlongRay;
    vec3 coordinate = position / uExtent + vec3(0.5);
    if (clipped(coordinate)) continue;
    float value = texture(uVolume, coordinate).r;
    float signal = smoothstep(uThreshold, min(1.0, uThreshold + 0.24), value);
    float alpha = 1.0 - exp(-signal * uDensity * 0.026);
    float warmth = smoothstep(uThreshold, 0.82, value);
    vec3 sampleColor = mix(vec3(0.19, 0.43, 0.43), vec3(0.91, 0.74, 0.48), warmth);
    sampleColor *= 0.7 + 0.3 * (1.0 - float(index) / float(max(uSteps, 1)));
    accumulated.rgb += (1.0 - accumulated.a) * sampleColor * alpha;
    accumulated.a += (1.0 - accumulated.a) * alpha;
  }

  outColor = vec4(accumulated.rgb + (1.0 - accumulated.a) * background, 1.0);
}`;

function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
}

function parsePointHeader(value: string | null, label: string): Point3 {
  const values = value?.split(",").map(Number) ?? [];
  if (values.length !== 3 || values.some((entry) => !Number.isFinite(entry))) {
    throw new Error(`Block volume omitted its ${label}.`);
  }
  return values as Point3;
}

function compileShader(gl: WebGL2RenderingContext, type: number, source: string) {
  const shader = gl.createShader(type);
  if (!shader) throw new Error("Could not allocate a WebGL shader.");
  gl.shaderSource(shader, source);
  gl.compileShader(shader);
  if (!gl.getShaderParameter(shader, gl.COMPILE_STATUS)) {
    const message = gl.getShaderInfoLog(shader) || "Unknown shader compilation error.";
    gl.deleteShader(shader);
    throw new Error(message);
  }
  return shader;
}

function createVolumeRenderer(canvas: HTMLCanvasElement): VolumeRenderer {
  const gl = canvas.getContext("webgl2", {
    alpha: false,
    antialias: true,
    depth: false,
    premultipliedAlpha: false,
  });
  if (!gl) throw new Error("This browser does not provide WebGL 2 volume rendering.");
  const vertex = compileShader(gl, gl.VERTEX_SHADER, VERTEX_SHADER);
  const fragment = compileShader(gl, gl.FRAGMENT_SHADER, FRAGMENT_SHADER);
  const program = gl.createProgram();
  if (!program) throw new Error("Could not allocate the block volume program.");
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || "Unknown volume program link error.";
    gl.deleteProgram(program);
    throw new Error(message);
  }
  const texture = gl.createTexture();
  const vao = gl.createVertexArray();
  if (!texture || !vao) throw new Error("Could not allocate the block volume texture.");
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_3D, texture);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(
    gl.TEXTURE_3D,
    0,
    gl.R8,
    1,
    1,
    1,
    0,
    gl.RED,
    gl.UNSIGNED_BYTE,
    new Uint8Array([0]),
  );
  return {
    gl,
    program,
    texture,
    vao,
    uniforms: {
      aspect: gl.getUniformLocation(program, "uAspect"),
      yaw: gl.getUniformLocation(program, "uYaw"),
      pitch: gl.getUniformLocation(program, "uPitch"),
      zoom: gl.getUniformLocation(program, "uZoom"),
      threshold: gl.getUniformLocation(program, "uThreshold"),
      density: gl.getUniformLocation(program, "uDensity"),
      steps: gl.getUniformLocation(program, "uSteps"),
      volume: gl.getUniformLocation(program, "uVolume"),
      extent: gl.getUniformLocation(program, "uExtent"),
      clipAxis: gl.getUniformLocation(program, "uClipAxis"),
      clipFraction: gl.getUniformLocation(program, "uClipFraction"),
    },
  };
}

function resizeCanvas(canvas: HTMLCanvasElement) {
  const bounds = canvas.getBoundingClientRect();
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }
  return { width, height, cssWidth: bounds.width, cssHeight: bounds.height, ratio };
}

function projectPoint(
  point: Point3,
  extent: Point3,
  orbit: Orbit,
  width: number,
  height: number,
) {
  const maximumExtent = Math.max(...extent);
  const x = (point[0] - extent[0] * 0.5) / maximumExtent;
  const y = (point[1] - extent[1] * 0.5) / maximumExtent;
  const z = (point[2] - extent[2] * 0.5) / maximumExtent;
  const cosYaw = Math.cos(orbit.yaw);
  const sinYaw = Math.sin(orbit.yaw);
  const cosPitch = Math.cos(orbit.pitch);
  const sinPitch = Math.sin(orbit.pitch);
  const viewX = cosYaw * x - sinYaw * z;
  const yawZ = sinYaw * x + cosYaw * z;
  const viewY = cosPitch * y + sinPitch * yawZ;
  const viewZ = -sinPitch * y + cosPitch * yawZ;
  const distance = 1.65 - viewZ;
  if (distance <= 0.01) return null;
  const aspect = width / height;
  const ndcX = ((viewX / distance) * 1.55 * orbit.zoom) / aspect;
  const ndcY = (viewY / distance) * 1.55 * orbit.zoom;
  return {
    x: (ndcX + 1) * width * 0.5,
    y: (1 - ndcY) * height * 0.5,
    depth: distance,
  };
}

function clipIndex(axis: ClipAxis) {
  return axis === "x" ? 0 : axis === "y" ? 1 : axis === "z" ? 2 : -1;
}

function clipPolygon(vertices: Point3[], axis: number, cutoff: number): Point3[] {
  if (axis < 0) return vertices;
  const result: Point3[] = [];
  for (let index = 0; index < vertices.length; index += 1) {
    const current = vertices[index];
    const previous = vertices[(index + vertices.length - 1) % vertices.length];
    const currentInside = current[axis] <= cutoff + 1e-6;
    const previousInside = previous[axis] <= cutoff + 1e-6;
    if (currentInside !== previousInside) {
      const denominator = current[axis] - previous[axis];
      const t = Math.abs(denominator) < 1e-9 ? 0 : (cutoff - previous[axis]) / denominator;
      result.push([
        previous[0] + (current[0] - previous[0]) * t,
        previous[1] + (current[1] - previous[1]) * t,
        previous[2] + (current[2] - previous[2]) * t,
      ]);
    }
    if (currentInside) result.push(current);
  }
  return result;
}

function pointInPolygon(point: { x: number; y: number }, polygon: Array<{ x: number; y: number }>) {
  let inside = false;
  for (let index = 0, previous = polygon.length - 1; index < polygon.length; previous = index++) {
    const first = polygon[index];
    const second = polygon[previous];
    const intersects =
      first.y > point.y !== second.y > point.y &&
      point.x < ((second.x - first.x) * (point.y - first.y)) / (second.y - first.y) + first.x;
    if (intersects) inside = !inside;
  }
  return inside;
}

function componentHue(rank: number) {
  return (rank * 137.508 + 168) % 360;
}

function drawLine(
  context: CanvasRenderingContext2D,
  first: { x: number; y: number } | null,
  second: { x: number; y: number } | null,
) {
  if (!first || !second) return;
  context.moveTo(first.x, first.y);
  context.lineTo(second.x, second.y);
}

export function BlockVolumeExplorer() {
  const volumeCanvasRef = useRef<HTMLCanvasElement>(null);
  const overlayCanvasRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<VolumeRenderer | null>(null);
  const hitPatchesRef = useRef<ProjectedPatch[]>([]);
  const dragRef = useRef<{
    x: number;
    y: number;
    startX: number;
    startY: number;
    moved: boolean;
  } | null>(null);
  const [result, setResult] = useState<BlockSheetResult | null>(null);
  const [volume, setVolume] = useState<VolumePayload | null>(null);
  const [rendererReady, setRendererReady] = useState(false);
  const [state, setState] = useState<"loading" | "ready" | "error">("loading");
  const [message, setMessage] = useState("Loading the solved block and source voxels…");
  const [retry, setRetry] = useState(0);
  const [orbit, setOrbit] = useState(DEFAULT_ORBIT);
  const [threshold, setThreshold] = useState(0.38);
  const [density, setDensity] = useState(3.4);
  const [sheetOpacity, setSheetOpacity] = useState(0.34);
  const [minimumComponentSize, setMinimumComponentSize] = useState(1);
  const [clipAxis, setClipAxis] = useState<ClipAxis>("none");
  const [clipFraction, setClipFraction] = useState(1);
  const [showVolume, setShowVolume] = useState(true);
  const [showSheets, setShowSheets] = useState(true);
  const [showEdges, setShowEdges] = useState(true);
  const [selectedComponent, setSelectedComponent] = useState<number | null>(null);
  const [isolateSelected, setIsolateSelected] = useState(false);

  const selected = useMemo(
    () => result?.components.find((component) => component.rank === selectedComponent) ?? null,
    [result, selectedComponent],
  );

  const visiblePatchCount = useMemo(() => {
    if (!result || !showSheets) return 0;
    return result.patches.reduce(
      (count, patch) =>
        count +
        (patch.componentSize >= minimumComponentSize &&
        (!isolateSelected || selectedComponent === patch.component)
          ? 1
          : 0),
      0,
    );
  }, [isolateSelected, minimumComponentSize, result, selectedComponent, showSheets]);

  const drawVolume = useCallback(() => {
    const renderer = rendererRef.current;
    const canvas = volumeCanvasRef.current;
    if (!renderer || !canvas || !result) return;
    const { width, height } = resizeCanvas(canvas);
    const maximumExtent = Math.max(...result.grid.extentXYZ);
    const normalizedExtent = result.grid.extentXYZ.map((value) => value / maximumExtent) as Point3;
    const axis = clipIndex(clipAxis);
    const { gl, uniforms } = renderer;
    gl.viewport(0, 0, width, height);
    gl.useProgram(renderer.program);
    gl.bindVertexArray(renderer.vao);
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, renderer.texture);
    gl.uniform1i(uniforms.volume, 0);
    gl.uniform1f(uniforms.aspect, width / height);
    gl.uniform1f(uniforms.yaw, orbit.yaw);
    gl.uniform1f(uniforms.pitch, orbit.pitch);
    gl.uniform1f(uniforms.zoom, orbit.zoom);
    gl.uniform1f(uniforms.threshold, threshold);
    gl.uniform1f(uniforms.density, showVolume ? density : 0);
    gl.uniform1i(uniforms.steps, volume ? Math.min(320, Math.max(96, volume.shapeXYZ[2])) : 160);
    gl.uniform3f(uniforms.extent, ...normalizedExtent);
    gl.uniform1i(uniforms.clipAxis, axis);
    gl.uniform1f(uniforms.clipFraction, clipFraction);
    gl.drawArrays(gl.TRIANGLES, 0, 3);
  }, [clipAxis, clipFraction, density, orbit, result, showVolume, threshold, volume]);

  const drawOverlay = useCallback(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas || !result) return;
    const { cssWidth: width, cssHeight: height, ratio } = resizeCanvas(canvas);
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const extent = result.grid.extentXYZ;
    const axis = clipIndex(clipAxis);
    const cutoff = axis >= 0 ? extent[axis] * clipFraction : Infinity;
    const projected: ProjectedPatch[] = [];
    if (showSheets) {
      for (const patch of result.patches) {
        if (patch.componentSize < minimumComponentSize) continue;
        if (isolateSelected && patch.component !== selectedComponent) continue;
        const clipped = clipPolygon(patch.vertices, axis, cutoff);
        if (clipped.length < 3) continue;
        const points = clipped.map((vertex) => projectPoint(vertex, extent, orbit, width, height));
        if (points.some((point) => !point)) continue;
        projected.push({
          patch,
          points: points.map((point) => ({ x: point!.x, y: point!.y })),
          depth: points.reduce((sum, point) => sum + point!.depth, 0) / points.length,
        });
      }
      projected.sort((first, second) => second.depth - first.depth);
      for (const entry of projected) {
        const isSelected = entry.patch.component === selectedComponent;
        const hue = componentHue(entry.patch.component);
        const confidenceScale = 0.58 + 0.42 * entry.patch.confidence;
        const alpha = clamp(sheetOpacity * confidenceScale * (selectedComponent && !isSelected ? 0.38 : 1), 0, 0.92);
        context.beginPath();
        context.moveTo(entry.points[0].x, entry.points[0].y);
        for (let index = 1; index < entry.points.length; index += 1) {
          context.lineTo(entry.points[index].x, entry.points[index].y);
        }
        context.closePath();
        context.fillStyle = `hsla(${hue}, 66%, ${isSelected ? 68 : 58}%, ${isSelected ? Math.max(alpha, 0.62) : alpha})`;
        context.fill();
        if (showEdges || isSelected) {
          context.lineWidth = isSelected ? 1.65 : 0.55;
          context.strokeStyle = isSelected
            ? "rgba(255, 232, 173, 0.96)"
            : `hsla(${hue}, 72%, 78%, ${Math.min(alpha + 0.16, 0.72)})`;
          context.stroke();
        }
      }
    }
    hitPatchesRef.current = projected;

    const corners: Point3[] = [
      [0, 0, 0],
      [extent[0], 0, 0],
      [0, extent[1], 0],
      [extent[0], extent[1], 0],
      [0, 0, extent[2]],
      [extent[0], 0, extent[2]],
      [0, extent[1], extent[2]],
      [extent[0], extent[1], extent[2]],
    ];
    const boxPoints = corners.map((corner) => projectPoint(corner, extent, orbit, width, height));
    const boxEdges = [
      [0, 1], [0, 2], [1, 3], [2, 3],
      [4, 5], [4, 6], [5, 7], [6, 7],
      [0, 4], [1, 5], [2, 6], [3, 7],
    ];
    context.beginPath();
    for (const [first, second] of boxEdges) drawLine(context, boxPoints[first], boxPoints[second]);
    context.lineWidth = 1;
    context.strokeStyle = "rgba(221, 238, 230, 0.44)";
    context.stroke();

    const origin = boxPoints[0];
    if (origin) {
      const axes: Array<[number, string, string]> = [
        [1, "X", "rgba(245, 179, 112, 0.88)"],
        [2, "Y", "rgba(123, 213, 204, 0.9)"],
        [4, "Z", "rgba(190, 175, 239, 0.9)"],
      ];
      context.font = "600 11px ui-monospace, SFMono-Regular, Menlo, monospace";
      for (const [cornerIndex, label, color] of axes) {
        const endpoint = boxPoints[cornerIndex];
        if (!endpoint) continue;
        context.fillStyle = color;
        context.fillText(label, endpoint.x + 5, endpoint.y - 5);
      }
    }
  }, [
    clipAxis,
    clipFraction,
    isolateSelected,
    minimumComponentSize,
    orbit,
    result,
    selectedComponent,
    sheetOpacity,
    showEdges,
    showSheets,
  ]);

  useEffect(() => {
    const canvas = volumeCanvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = createVolumeRenderer(canvas);
      setRendererReady(true);
    } catch (error) {
      queueMicrotask(() => {
        setState("error");
        setMessage(error instanceof Error ? error.message : "Volume renderer failed.");
      });
      return;
    }
    return () => {
      const renderer = rendererRef.current;
      if (renderer) {
        renderer.gl.deleteTexture(renderer.texture);
        renderer.gl.deleteVertexArray(renderer.vao);
        renderer.gl.deleteProgram(renderer.program);
      }
      rendererRef.current = null;
      setRendererReady(false);
    };
  }, []);

  useEffect(() => {
    const controller = new AbortController();
    setState("loading");
    setMessage("Loading 4,784 solved patches and the source block volume…");
    Promise.all([
      fetch("/api/block/sheets", { signal: controller.signal }).then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error || `Block sheets returned ${response.status}.`);
        }
        return response.json() as Promise<BlockSheetResult>;
      }),
      fetch("/api/block/volume?stride=2", { signal: controller.signal }).then(async (response) => {
        if (!response.ok) {
          const payload = (await response.json().catch(() => null)) as { error?: string } | null;
          throw new Error(payload?.error || `Block volume returned ${response.status}.`);
        }
        const shapeXYZ = parsePointHeader(response.headers.get("X-Volume-Shape-XYZ"), "shape");
        const stride = Number(response.headers.get("X-Volume-Stride") ?? "2");
        const percentiles = (response.headers.get("X-Volume-Percentiles") ?? "0,0,0,0")
          .split(",")
          .map(Number) as [number, number, number, number];
        const bytes = new Uint8Array(await response.arrayBuffer());
        const expected = shapeXYZ[0] * shapeXYZ[1] * shapeXYZ[2];
        if (bytes.length !== expected) {
          throw new Error(`Block volume returned ${bytes.length} bytes; expected ${expected}.`);
        }
        return { bytes, shapeXYZ, stride, percentiles } satisfies VolumePayload;
      }),
    ])
      .then(([sheetResult, volumeResult]) => {
        if (controller.signal.aborted) return;
        setResult(sheetResult);
        setVolume(volumeResult);
        setState("ready");
        setMessage("");
      })
      .catch((error) => {
        if (controller.signal.aborted) return;
        setState("error");
        setMessage(error instanceof Error ? error.message : "The solved block is unavailable.");
      });
    return () => controller.abort();
  }, [retry]);

  useEffect(() => {
    const renderer = rendererRef.current;
    if (!rendererReady || !renderer || !volume) return;
    const { gl } = renderer;
    gl.activeTexture(gl.TEXTURE0);
    gl.bindTexture(gl.TEXTURE_3D, renderer.texture);
    gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
    gl.texImage3D(
      gl.TEXTURE_3D,
      0,
      gl.R8,
      volume.shapeXYZ[0],
      volume.shapeXYZ[1],
      volume.shapeXYZ[2],
      0,
      gl.RED,
      gl.UNSIGNED_BYTE,
      volume.bytes,
    );
    drawVolume();
  }, [drawVolume, rendererReady, volume]);

  useEffect(() => {
    drawVolume();
    drawOverlay();
  }, [drawOverlay, drawVolume]);

  useEffect(() => {
    const volumeCanvas = volumeCanvasRef.current;
    const overlayCanvas = overlayCanvasRef.current;
    if (!volumeCanvas || !overlayCanvas) return;
    const observer = new ResizeObserver(() => {
      drawVolume();
      drawOverlay();
    });
    observer.observe(volumeCanvas);
    observer.observe(overlayCanvas);
    return () => observer.disconnect();
  }, [drawOverlay, drawVolume]);

  useEffect(() => {
    const canvas = overlayCanvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const sensitivity = event.ctrlKey || event.metaKey ? 0.018 : 0.002;
      setOrbit((value) => ({
        ...value,
        zoom: clamp(value.zoom * Math.exp(-event.deltaY * sensitivity), 0.42, 9),
      }));
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

  const handlePointerDown = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    event.currentTarget.setPointerCapture(event.pointerId);
    dragRef.current = {
      x: event.clientX,
      y: event.clientY,
      startX: event.clientX,
      startY: event.clientY,
      moved: false,
    };
  };

  const handlePointerMove = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    if (!drag) return;
    const deltaX = event.clientX - drag.x;
    const deltaY = event.clientY - drag.y;
    drag.x = event.clientX;
    drag.y = event.clientY;
    if (Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4) {
      drag.moved = true;
    }
    setOrbit((value) => ({
      ...value,
      yaw: value.yaw + deltaX * 0.009,
      pitch: clamp(value.pitch + deltaY * 0.009, -1.52, 1.52),
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    const drag = dragRef.current;
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
    if (!drag || drag.moved) return;
    const bounds = event.currentTarget.getBoundingClientRect();
    const point = { x: event.clientX - bounds.left, y: event.clientY - bounds.top };
    const hit = [...hitPatchesRef.current]
      .reverse()
      .find((entry) => pointInPolygon(point, entry.points));
    setSelectedComponent((value) => (hit?.patch.component === value ? null : hit?.patch.component ?? null));
    if (!hit) setIsolateSelected(false);
  };

  const clearSelection = () => {
    setSelectedComponent(null);
    setIsolateSelected(false);
  };

  return (
    <main className="block-volume-page">
      <header className="block-volume-header">
        <nav className="block-volume-nav" aria-label="Experiment pages">
          <Link href="/cross-scroll">← Cross-scroll slices</Link>
          <Link href="/">Local workbench</Link>
        </nav>
        <div>
          <p className="eyebrow">Acus · owned core reconstruction</p>
          <h1>Block volume + solved sheets</h1>
        </div>
        <p className="block-volume-summary">
          {result
            ? `${result.grid.extentXYZ.join(" × ")} vox · ${result.stats.patchCount.toLocaleString()} patches · ${result.stats.componentCount.toLocaleString()} sheets · ${result.stats.retainedJoinCount.toLocaleString()} joins`
            : message}
        </p>
      </header>

      <section className="block-volume-toolbar" aria-label="Block volume display controls">
        <label className="block-volume-range-control">
          <span>Volume threshold</span>
          <strong>{Math.round(threshold * 255)}</strong>
          <input
            type="range"
            min={0.16}
            max={0.78}
            step={0.01}
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value))}
          />
        </label>
        <label className="block-volume-range-control">
          <span>Volume density</span>
          <strong>{density.toFixed(1)}×</strong>
          <input
            type="range"
            min={0.4}
            max={8}
            step={0.1}
            value={density}
            onChange={(event) => setDensity(Number(event.target.value))}
          />
        </label>
        <label className="block-volume-range-control">
          <span>Sheet opacity</span>
          <strong>{Math.round(sheetOpacity * 100)}%</strong>
          <input
            type="range"
            min={0.06}
            max={0.82}
            step={0.02}
            value={sheetOpacity}
            onChange={(event) => setSheetOpacity(Number(event.target.value))}
          />
        </label>
        <label className="block-volume-select-control">
          <span>Minimum sheet size</span>
          <select
            value={minimumComponentSize}
            onChange={(event) => setMinimumComponentSize(Number(event.target.value))}
          >
            {COMPONENT_SIZE_OPTIONS.map((value) => (
              <option value={value} key={value}>
                {value === 1 ? "All components" : `${value}+ cells`}
              </option>
            ))}
          </select>
        </label>
        <label className="block-volume-select-control">
          <span>Cutaway axis</span>
          <select
            value={clipAxis}
            onChange={(event) => {
              setClipAxis(event.target.value as ClipAxis);
              setClipFraction(1);
            }}
          >
            <option value="none">No cutaway</option>
            <option value="x">X plane</option>
            <option value="y">Y plane</option>
            <option value="z">Z plane</option>
          </select>
        </label>
        <label className="block-volume-range-control block-volume-cut-control">
          <span>Visible depth</span>
          <strong>{clipAxis === "none" ? "full" : `${Math.round(clipFraction * 100)}%`}</strong>
          <input
            type="range"
            min={0.04}
            max={1}
            step={0.01}
            value={clipFraction}
            disabled={clipAxis === "none"}
            onChange={(event) => setClipFraction(Number(event.target.value))}
          />
        </label>
        <div className="block-volume-toggle-group" aria-label="Visible layers">
          <button type="button" aria-pressed={showVolume} onClick={() => setShowVolume((value) => !value)}>
            Volume
          </button>
          <button type="button" aria-pressed={showSheets} onClick={() => setShowSheets((value) => !value)}>
            Sheets
          </button>
          <button type="button" aria-pressed={showEdges} onClick={() => setShowEdges((value) => !value)}>
            Edges
          </button>
          <button type="button" onClick={() => setOrbit(DEFAULT_ORBIT)}>
            Reset view
          </button>
        </div>
      </section>

      <section className="block-volume-viewer" aria-live="polite">
        <div className="block-volume-stage">
          <canvas ref={volumeCanvasRef} className="block-volume-canvas" aria-hidden="true" />
          <canvas
            ref={overlayCanvasRef}
            className="block-sheet-overlay"
            tabIndex={0}
            role="img"
            aria-label="Orbitable source volume containing every retained sheet patch. Drag to orbit, pinch or scroll to zoom, and click a sheet to inspect its component."
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={() => {
              dragRef.current = null;
            }}
            onKeyDown={(event) => {
              if (event.key === "Escape") clearSelection();
            }}
          />

          <div className="block-volume-legend" aria-hidden="true">
            <span><i data-kind="volume" />CT material</span>
            <span><i data-kind="sheet" />sheet identity</span>
            <span>drag orbit · pinch zoom · click sheet</span>
          </div>

          <div className="block-volume-count">
            {state === "ready"
              ? `${visiblePatchCount.toLocaleString()} / ${result?.stats.patchCount.toLocaleString()} patches visible`
              : message}
          </div>

          {state === "error" ? (
            <div className="block-volume-error" role="alert">
              <strong>Block viewer could not load</strong>
              <span>{message}</span>
              <button type="button" onClick={() => setRetry((value) => value + 1)}>
                Retry
              </button>
            </div>
          ) : state === "loading" ? (
            <div className="block-volume-loading" role="status">
              <span />
              <p>{message}</p>
            </div>
          ) : null}

          {selected ? (
            <div className="block-volume-selection">
              <div>
                <span>Selected sheet</span>
                <strong>#{selected.rank} · {selected.patchCount} cells</strong>
                <small>
                  mean confidence {selected.meanConfidence.toFixed(2)} · span {selected.boundsMaximumXYZ
                    .map((value, index) => Math.round(value - selected.boundsMinimumXYZ[index]))
                    .join(" × ")} vox
                </small>
              </div>
              <button
                type="button"
                aria-pressed={isolateSelected}
                onClick={() => setIsolateSelected((value) => !value)}
              >
                {isolateSelected ? "Show all" : "Isolate"}
              </button>
              <button type="button" onClick={clearSelection}>Clear</button>
            </div>
          ) : null}
        </div>
      </section>
    </main>
  );
}
