"use client";

import Link from "next/link";
import {
  useCallback,
  useEffect,
  useId,
  useMemo,
  useRef,
  useState,
  type PointerEvent as ReactPointerEvent,
} from "react";

export type Seed = { x: number; y: number; z: number };
export type Orbit = { yaw: number; pitch: number; zoom: number };
type Point3 = [number, number, number];
type AcusNeedle = {
  center: Point3;
  start: Point3;
  end: Point3;
  score: number;
  robustWeight: number;
  normalCoordinate: number;
  familyAngleDeg: number;
  planeResidualDeg: number;
  inlier: boolean;
};
type OrientationProfile = {
  normalCoordinateRange: [number, number];
  depthCenters: number[];
  orientationCentersDeg: number[];
  density: number[][];
  slices: Array<{
    normalCoordinate: number;
    support: number;
    concentration: number;
    twoModeCoverage: number;
    dominantAngles: Array<{ angleDeg: number; relativeStrength: number }>;
  }>;
  stats: {
    meanOrientationConcentration: number;
    meanTwoModeCoverage: number;
    coveredDepthFraction: number;
    normalBandwidthVoxels: number;
    orientationBandwidthDeg: number;
    interpretation: string;
  };
};
type AcusResult = {
  seed: Seed;
  cube: { size: number };
  needles: AcusNeedle[];
  normal: Point3;
  normalLine: { start: Point3; end: Point3 };
  orientationProfile: OrientationProfile;
  settings: {
    needleLength: number;
    requestedPadding: number;
    effectivePadding: number;
    minimumPadding: number;
    contextSize: number;
    crossSectionRadius: number;
    paddingSufficient: boolean;
  };
  stats: {
    elapsedMs: number;
    computeBackend: "gpu" | "cpu";
    computeDevice: string | null;
    lineFieldMs: number;
    lineFieldBatchSize: number;
    lineFieldBatchLaunches: number;
    needleCount: number;
    inlierCount: number;
    normalConfidence: number;
    medianPlaneResidualDeg: number;
    boundaryNeedleCount: number;
    boundaryTangentialFraction: number | null;
    medianAxialCoverage: number;
    constraint: string;
  };
};
type AcusFieldCell = {
  row: number;
  column: number;
  offsetU: number;
  offsetV: number;
  seed: Seed;
  anchorLocalCenter: Point3;
  isAnchor: boolean;
  valid: boolean;
  error?: string;
  normal?: Point3;
  normalLine?: { start: Point3; end: Point3 };
  normalAngleDeg?: number;
  profileCorrelation?: number;
  bestDepthLagVoxels?: number;
  normalConfidence?: number;
  twoModeCoverage?: number;
  needleCount?: number;
};
type AcusFieldResult = {
  seed: Seed;
  cube: { size: number };
  anchor: AcusResult;
  grid: {
    size: number;
    spacingVoxels: number;
    basisU: Point3;
    basisV: Point3;
    normal: Point3;
  };
  cells: AcusFieldCell[];
  stats: {
    elapsedMs: number;
    computeBackend: "gpu" | "cpu";
    computeDevice: string | null;
    lineFieldMs: number;
    lineFieldBatchSize: number;
    lineFieldBatchLaunches: number;
    validCellCount: number;
    medianNormalAngleDeg: number | null;
    p90NormalAngleDeg: number | null;
    medianProfileCorrelation: number | null;
    medianAbsoluteDepthLagVoxels: number | null;
    warning: string;
  };
};
type AcusAuditSweep = {
  spacingVoxels: number;
  validNeighborCount: number;
  medianOverlapFraction: number | null;
  medianNormalAngleDeg: number | null;
  p90NormalAngleDeg: number | null;
  medianNormalBootstrapP90Deg: number | null;
  medianNormalToUncertaintyRatio: number | null;
  medianProfileCorrelation: number | null;
  medianProfileNull: number | null;
  medianProfileExcess: number | null;
  significantProfileFraction: number | null;
  medianAbsoluteDepthLagVoxels: number | null;
  elapsedMs: number;
};
type AcusAuditResult = {
  seed: Seed;
  cube: { size: number };
  anchor: AcusResult;
  spacings: number[];
  sweeps: AcusAuditSweep[];
  stats: {
    elapsedMs: number;
    computeBackend: "gpu" | "cpu";
    computeDevice: string | null;
    lineFieldMs: number;
    lineFieldBatchLaunches: number;
    bootstrapRepetitions: number;
    nullRepetitions: number;
    nullHypothesis: string;
    warning: string;
  };
};
type PaddingAuditSweep = {
  requestedPadding: number;
  effectivePadding: number;
  minimumPadding: number;
  paddingSufficient: boolean;
  contextSize: number;
  needleCount: number;
  boundaryNeedleCount: number;
  boundaryTangentialFraction: number | null;
  medianAxialCoverage: number;
  normalAngleToReferenceDeg: number;
  profileCorrelationToReference: number;
  bestDepthLagToReferenceVoxels: number;
  normalConfidence: number;
  elapsedMs: number;
};
type PaddingAuditResult = {
  seed: Seed;
  cube: { size: number };
  referencePadding: number;
  needleLength: number;
  sweeps: PaddingAuditSweep[];
  failures: Array<{ requestedPadding: number; error: string }>;
  stats: {
    elapsedMs: number;
    criterion: string;
    interpretation: string;
  };
};
type RegionCell = {
  index: [number, number, number];
  center: Point3;
  valid: boolean;
  normal?: Point3;
  needleCount: number;
  inlierFraction?: number;
  normalConfidence?: number;
  coplanarity?: number;
  medianPlaneResidualDeg?: number;
  medianAxialCoverage?: number;
  twoModeCoverage?: number;
  coveredDepthFraction?: number;
  neighborCount?: number;
  neighborNormalMedianDeg?: number | null;
  neighborPatternMedian?: number | null;
};
export type RegionResult = {
  shape: Seed;
  globalOrigin: Seed;
  settings: {
    cubeSize: number;
    scale: number;
    candidateSpacing: number;
    maxNeedles: number;
    needleLength: number;
    requestedPadding: number;
    effectivePadding: number;
    minimumPadding: number;
    gridStride: number;
    tileCore: number;
    catalogBinSize: number;
    maxNeedlesPerBin: number;
  };
  grid: { x: number[]; y: number[]; z: number[]; availableZ?: number[]; layout: string };
  view?: {
    mode: "slice" | "volume";
    zIndex: number | null;
    z: number | null;
    sourceGridShapeZYX: [number, number, number];
    displayGridShapeZYX: [number, number, number];
  };
  cells: RegionCell[];
  stats: {
    elapsedMs: number;
    cacheHit: boolean;
    cacheKey: string;
    computeBackend: "gpu" | "cpu";
    computeDevice: string | null;
    lineFieldMs: number;
    lineFieldBatchLaunches: number;
    strengthScale: number;
    tileCount: number;
    candidateBlockCount: number;
    selectedCandidateCount: number;
    rawNeedleCount: number;
    needleCount: number;
    cellCount: number;
    validCellCount: number;
    medianNormalConfidence: number | null;
    medianNeighborNormalDeg: number | null;
    medianNeighborPattern: number | null;
    macroRadialFitCellCount?: number;
    macroRadialCenterXY?: [number, number] | null;
    medianMacroRadialResidualDeg?: number | null;
    p90MacroRadialResidualDeg?: number | null;
    macroRadialNullMedianDeg?: number | null;
    macroRadialExcessDeg?: number | null;
    macroCenterDriftXY?: [number, number] | null;
    macroCenterDriftVoxels?: number | null;
    medianAbsoluteNormalZ?: number | null;
    constraint: string;
  };
};
export type FlakeResult = {
  view: { mode: "slice"; zIndex: number; z: number };
  settings: {
    maximumFlakesPerCell: number;
    depthBandwidthVoxels: number;
    angleBandwidthDeg: number;
    minimumNeedles: number;
    defaultTrackScore: number;
    gridStride: number;
    cubeSize: number;
  };
  flakes: Array<{
    id: number;
    cellIndex: [number, number, number];
    cellCenter: Point3;
    center: Point3;
    normal: Point3;
    fiber: Point3;
    crossFiber: Point3;
    depthOffset: number;
    planeOffset: number;
    radiusFiber: number;
    radiusCrossFiber: number;
    thickness: number;
    needleCount: number;
    effectiveSupport: number;
    supportFraction: number;
    fiberConcentration: number;
    medianFiberResidualDeg: number;
    quality: number;
    fiberAngleXYDeg: number | null;
    validated?: boolean;
    validationScore?: number;
    foldDepthDeltaVoxels?: number;
    foldPositionResidualVoxels?: number;
    foldFiberDeltaDeg?: number;
    supportA?: number;
    supportB?: number;
    sheetletId?: number;
    sheetletSize?: number;
    sheetletZSpanVoxels?: number;
    sheetletDegree?: number;
  }>;
  links: Array<{
    source: number;
    target: number;
    score: number;
    rawCompatibility?: number;
    positionResidualVoxels: number;
    normalAngleDeg: number;
    fiberAngleDeg: number;
    sharedNeedleFraction?: number;
    axis?: "x" | "y" | "z";
    endpointValidation?: number;
  }>;
  sheetletLinks?: FlakeResult["links"];
  stats: {
    elapsedMs: number;
    cacheHit: boolean;
    validCellCount: number;
    fittedCellCount: number;
    flakeCount: number;
    candidateLinkCount: number;
    acceptedLinkCount: number;
    linkedTrackCount: number;
    largestTrackSize: number;
    medianTrackSize: number | null;
    medianFlakesPerFittedCell: number;
    medianQuality: number | null;
    medianPositionResidualVoxels: number | null;
    medianNormalAngleDeg: number | null;
    medianFiberAngleDeg: number | null;
    fiberShuffledMedianDeg: number | null;
    medianSharedNeedleFraction: number | null;
    constraint: string;
  };
};
export type FlakeAuditSummary = {
  cellPairCount: number;
  mutualLinkCount: number;
  acceptedLinkCount: number;
  acceptedLinksPerCellPair: number;
  linkedFlakeFraction: number;
  linkedTrackCount: number;
  largestTrackSize: number;
  medianTrackSize: number | null;
  medianScore: number | null;
  medianRawCompatibility: number | null;
  medianPositionResidualVoxels: number | null;
  medianNormalAngleDeg: number | null;
  medianFiberAngleDeg: number | null;
  medianSharedNeedleFraction: number | null;
  repetitions?: number;
};
export type FlakeAuditSweep = {
  cellStep: number;
  spacingVoxels: number;
  overlapFraction: number;
  gapVoxels: number;
  independentWindows: boolean;
  linkSurvivalVs32: number;
  spatialNullDensityRatio: number;
  fiberNullDensityRatio: number;
  observed: FlakeAuditSummary;
  nulls: {
    fiber: FlakeAuditSummary;
    depth: FlakeAuditSummary;
    spatial: FlakeAuditSummary;
  };
  links: FlakeResult["links"];
};
export type FlakeAuditResult = {
  view: { mode: "slice"; zIndex: number; z: number };
  settings: {
    gridStride: number;
    cubeSize: number;
    minimumLinkScore: number;
    repetitions: number;
    nulls: string[];
  };
  sweeps: FlakeAuditSweep[];
  stats: {
    elapsedMs: number;
    cacheHit: boolean;
    flakeCount: number;
    constraint: string;
  };
};
export type FlakeHoldoutResult = {
  view: { mode: "slice"; zIndex: number; z: number };
  settings: {
    split: string;
    folds: number;
    minimumNeedlesPerFoldMode: number;
    minimumValidationScore: number;
    repetitions: number;
    null: string;
  };
  validationByFlake: Array<{
    flakeId: number;
    validated: boolean;
    validationScore: number;
    foldDepthDeltaVoxels?: number;
    foldPositionResidualVoxels?: number;
    foldFiberDeltaDeg?: number;
    supportA?: number;
    supportB?: number;
    needleCountA?: number;
    needleCountB?: number;
  }>;
  stats: {
    elapsedMs: number;
    cacheHit: boolean;
    eligibleCellCount: number;
    replicatedPairCount: number;
    validatedPairCount: number;
    validatedCellCount: number;
    validatedFullFlakeCount: number;
    validatedFullFlakeFraction: number;
    medianFoldDepthDeltaVoxels: number | null;
    medianFoldPositionResidualVoxels: number | null;
    medianFoldFiberDeltaDeg: number | null;
    nullValidatedPairCount: number;
    validatedPairNullRatio: number;
    nullMedianFiberDeltaDeg: number | null;
    constraint: string;
  };
};
export type SheetletResult = {
  view: { mode: "slice"; zIndex: number; z: number };
  settings: {
    spacingVoxels: number;
    cellStep: number;
    minimumLinkScore: number;
    heldoutValidatedNodesOnly: boolean;
    null: string;
    repetitions: number;
  };
  nodes: Array<{
    id: number;
    zIndex: number;
    flakeId: number;
    sheetletId: number;
    sheetletSize: number;
    sheetletZSpanVoxels: number;
    degree: number;
    validationScore: number;
  }>;
  links: FlakeResult["links"];
  stats: {
    elapsedMs: number;
    cacheHit: boolean;
    nodeCount: number;
    acceptedLinkCount: number;
    acceptedXLinkCount: number;
    acceptedYLinkCount: number;
    acceptedZLinkCount: number;
    sheetletCount: number;
    linkedNodeCount: number;
    largestSheetletSize: number;
    medianSheetletSize: number | null;
    medianPositionResidualVoxels: number | null;
    medianNormalAngleDeg: number | null;
    medianFiberAngleDeg: number | null;
    medianZFiberAngleDeg: number | null;
    null: {
      acceptedLinkCount: number | null;
      acceptedZLinkCount: number | null;
      largestSheetletSize: number | null;
    };
    linkNullRatio: number;
    zLinkNullRatio: number;
    constraint: string;
  };
};
type RegionMetric = "normal" | "pattern" | "confidence" | "depth";
type RegionVectorMode = "normal" | "slice-tangent" | "flakes" | "tracks" | "sheetlets";
type FlakeColorMode = "depth" | "fiber" | "quality" | "validation" | "track" | "sheetlet";
type SlabStatus = {
  configured: boolean;
  state: string;
  source?: {
    shapeZYX?: [number, number, number];
    originZYX?: [number, number, number];
  };
  fetch?: {
    state?: string;
    completedCount?: number;
    totalCount?: number;
    remainingCount?: number;
    downloadMiBPerSecond?: number;
    updatedAt?: string;
  };
  analysis?: {
    state?: string;
    completedTileCount?: number;
    tileCount?: number;
    calibrationCompletedCount?: number;
    calibrationTileCount?: number;
    needleCount?: number;
    completedCellCount?: number;
    cellCount?: number;
    validCellCount?: number;
    tilesPerSecondThisRun?: number;
    cellsPerSecondThisRun?: number;
    updatedAt?: string;
    completedAt?: string;
  };
  settings?: {
    gridStride?: number;
    cubeSize?: number;
  };
  shapeZYX?: [number, number, number];
};
type RenderState = {
  orbit: Orbit;
  threshold: number;
  density: number;
  steps: number;
};

type Renderer = {
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
  };
  hasVolume: boolean;
};

const DEFAULT_ORBIT: Orbit = { yaw: -0.62, pitch: 0.48, zoom: 1 };

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

vec2 intersectBox(vec3 origin, vec3 direction) {
  vec3 safeDirection = sign(direction) * max(abs(direction), vec3(0.00001));
  vec3 inverseDirection = 1.0 / safeDirection;
  vec3 t0 = (vec3(-0.5) - origin) * inverseDirection;
  vec3 t1 = (vec3(0.5) - origin) * inverseDirection;
  vec3 nearValues = min(t0, t1);
  vec3 farValues = max(t0, t1);
  return vec2(
    max(max(nearValues.x, nearValues.y), nearValues.z),
    min(min(farValues.x, farValues.y), farValues.z)
  );
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

  vec3 background = mix(vec3(0.035, 0.075, 0.08), vec3(0.08, 0.14, 0.15), vUv.y);
  vec2 bounds = intersectBox(rayOrigin, rayDirection);
  float start = max(bounds.x, 0.0);
  if (bounds.y <= start) {
    outColor = vec4(background, 1.0);
    return;
  }

  float stepLength = (bounds.y - start) / float(max(uSteps, 1));
  vec4 accumulated = vec4(0.0);
  for (int index = 0; index < 256; index += 1) {
    if (index >= uSteps || accumulated.a > 0.985) break;
    float distanceAlongRay = start + (float(index) + 0.5) * stepLength;
    vec3 position = rayOrigin + rayDirection * distanceAlongRay + vec3(0.5);
    float value = texture(uVolume, position).r;
    float signal = smoothstep(uThreshold, min(1.0, uThreshold + 0.28), value);
    float alpha = 1.0 - exp(-signal * uDensity * 0.035);
    float warmth = smoothstep(uThreshold, 1.0, value);
    vec3 sampleColor = mix(vec3(0.17, 0.48, 0.49), vec3(1.0, 0.79, 0.47), warmth);
    sampleColor *= 0.72 + 0.28 * (1.0 - float(index) / float(max(uSteps, 1)));
    accumulated.rgb += (1.0 - accumulated.a) * sampleColor * alpha;
    accumulated.a += (1.0 - accumulated.a) * alpha;
  }

  outColor = vec4(accumulated.rgb + (1.0 - accumulated.a) * background, 1.0);
}`;

function clamp(value: number, low: number, high: number) {
  return Math.min(Math.max(value, low), high);
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

function createRenderer(canvas: HTMLCanvasElement): Renderer {
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
  if (!program) throw new Error("Could not allocate the volume-rendering program.");
  gl.attachShader(program, vertex);
  gl.attachShader(program, fragment);
  gl.linkProgram(program);
  gl.deleteShader(vertex);
  gl.deleteShader(fragment);
  if (!gl.getProgramParameter(program, gl.LINK_STATUS)) {
    const message = gl.getProgramInfoLog(program) || "Unknown volume-renderer link error.";
    gl.deleteProgram(program);
    throw new Error(message);
  }

  const texture = gl.createTexture();
  const vao = gl.createVertexArray();
  if (!texture || !vao) throw new Error("Could not allocate the volume texture.");
  gl.bindVertexArray(vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_3D, texture);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_MAG_FILTER, gl.LINEAR);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
  gl.texParameteri(gl.TEXTURE_3D, gl.TEXTURE_WRAP_R, gl.CLAMP_TO_EDGE);
  gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
  gl.texImage3D(gl.TEXTURE_3D, 0, gl.R8, 1, 1, 1, 0, gl.RED, gl.UNSIGNED_BYTE, new Uint8Array([0]));

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
    },
    hasVolume: false,
  };
}

function drawRenderer(renderer: Renderer, canvas: HTMLCanvasElement, state: RenderState) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width * ratio));
  const height = Math.max(1, Math.round(bounds.height * ratio));
  if (canvas.width !== width || canvas.height !== height) {
    canvas.width = width;
    canvas.height = height;
  }

  const { gl, uniforms } = renderer;
  gl.viewport(0, 0, width, height);
  gl.useProgram(renderer.program);
  gl.bindVertexArray(renderer.vao);
  gl.activeTexture(gl.TEXTURE0);
  gl.bindTexture(gl.TEXTURE_3D, renderer.texture);
  gl.uniform1i(uniforms.volume, 0);
  gl.uniform1f(uniforms.aspect, width / height);
  gl.uniform1f(uniforms.yaw, state.orbit.yaw);
  gl.uniform1f(uniforms.pitch, state.orbit.pitch);
  gl.uniform1f(uniforms.zoom, state.orbit.zoom);
  gl.uniform1f(uniforms.threshold, state.threshold);
  gl.uniform1f(uniforms.density, state.density);
  gl.uniform1i(uniforms.steps, state.steps);
  gl.drawArrays(gl.TRIANGLES, 0, 3);
}

function projectPoint(
  point: Point3,
  cubeSize: number,
  orbit: Orbit,
  width: number,
  height: number,
) {
  const x = point[0] / cubeSize - 0.5;
  const y = point[1] / cubeSize - 0.5;
  const z = point[2] / cubeSize - 0.5;
  const cosYaw = Math.cos(orbit.yaw);
  const sinYaw = Math.sin(orbit.yaw);
  const cosPitch = Math.cos(orbit.pitch);
  const sinPitch = Math.sin(orbit.pitch);

  // Invert the exact Ry(yaw) · Rx(pitch) camera rotation used in the shader.
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

function drawAcusOverlay(
  canvas: HTMLCanvasElement,
  result: AcusResult | null,
  field: AcusFieldResult | null,
  cubeSize: number,
  orbit: Orbit,
) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  const pixelWidth = Math.round(width * ratio);
  const pixelHeight = Math.round(height * ratio);
  if (canvas.width !== pixelWidth || canvas.height !== pixelHeight) {
    canvas.width = pixelWidth;
    canvas.height = pixelHeight;
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  if (!result) return;

  const projected = result.needles
    .map((needle) => ({
      needle,
      start: projectPoint(needle.start, cubeSize, orbit, width, height),
      end: projectPoint(needle.end, cubeSize, orbit, width, height),
    }))
    .filter(
      (entry): entry is typeof entry & {
        start: NonNullable<typeof entry.start>;
        end: NonNullable<typeof entry.end>;
      } => Boolean(entry.start && entry.end),
    )
    .sort((a, b) => b.start.depth + b.end.depth - a.start.depth - a.end.depth);

  context.lineCap = "round";
  for (const { needle, start, end } of projected) {
    const alpha = needle.inlier ? 0.42 + needle.score * 0.48 : 0.22;
    context.strokeStyle = needle.inlier
      ? `hsla(${28 + needle.familyAngleDeg * 1.05} 88% 67% / ${alpha})`
      : "rgba(232, 111, 91, 0.3)";
    context.lineWidth = needle.inlier ? 1.25 + needle.score * 1.2 : 1;
    context.beginPath();
    context.moveTo(start.x, start.y);
    context.lineTo(end.x, end.y);
    context.stroke();
  }

  if (field) {
    for (const cell of field.cells) {
      if (!cell.valid || cell.isAnchor || !cell.normalLine) continue;
      const start = projectPoint(cell.normalLine.start, cubeSize, orbit, width, height);
      const end = projectPoint(cell.normalLine.end, cubeSize, orbit, width, height);
      const center = projectPoint(cell.anchorLocalCenter, cubeSize, orbit, width, height);
      if (!start || !end || !center) continue;
      const angle = cell.normalAngleDeg ?? 30;
      const hue = 166 - Math.min(angle / 15, 1) * 150;
      context.strokeStyle = `hsla(${hue} 84% 67% / 0.92)`;
      context.lineWidth = 1.7;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
      context.fillStyle = `hsla(${hue} 90% 72% / 0.95)`;
      context.beginPath();
      context.arc(center.x, center.y, 2.1, 0, Math.PI * 2);
      context.fill();
    }
  }

  const normalStart = projectPoint(result.normalLine.start, cubeSize, orbit, width, height);
  const normalEnd = projectPoint(result.normalLine.end, cubeSize, orbit, width, height);
  if (normalStart && normalEnd) {
    context.strokeStyle = "rgba(4, 18, 20, 0.72)";
    context.lineWidth = 5;
    context.beginPath();
    context.moveTo(normalStart.x, normalStart.y);
    context.lineTo(normalEnd.x, normalEnd.y);
    context.stroke();
    context.strokeStyle = "rgba(137, 246, 235, 0.96)";
    context.lineWidth = 2.2;
    context.beginPath();
    context.moveTo(normalStart.x, normalStart.y);
    context.lineTo(normalEnd.x, normalEnd.y);
    context.stroke();
    context.fillStyle = "rgba(184, 255, 247, 0.98)";
    context.font = "700 11px ui-monospace, SFMono-Regular, Menlo, monospace";
    context.fillText("n", normalEnd.x + 5, normalEnd.y - 5);
  }
}

function drawOrientationProfile(
  canvas: HTMLCanvasElement,
  profile: OrientationProfile,
  needles: AcusNeedle[],
) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  context.fillStyle = "#0a1719";
  context.fillRect(0, 0, width, height);

  const margin = { left: 40, right: 12, top: 43, bottom: 37 };
  const plotWidth = Math.max(1, width - margin.left - margin.right);
  const plotHeight = Math.max(1, height - margin.top - margin.bottom);
  const [minimumDepth, maximumDepth] = profile.normalCoordinateRange;
  const depthSpan = Math.max(maximumDepth - minimumDepth, 1);
  const mapX = (value: number) => margin.left + ((value - minimumDepth) / depthSpan) * plotWidth;
  const mapY = (value: number) => margin.top + (1 - value / 180) * plotHeight;

  context.save();
  context.beginPath();
  context.rect(margin.left, margin.top, plotWidth, plotHeight);
  context.clip();
  const depthStep =
    profile.depthCenters.length > 1
      ? Math.abs(profile.depthCenters[1] - profile.depthCenters[0])
      : depthSpan;
  const angleStep =
    profile.orientationCentersDeg.length > 1
      ? Math.abs(profile.orientationCentersDeg[1] - profile.orientationCentersDeg[0])
      : 5;
  for (let depthIndex = 0; depthIndex < profile.depthCenters.length; depthIndex += 1) {
    const depth = profile.depthCenters[depthIndex];
    for (
      let angleIndex = 0;
      angleIndex < profile.orientationCentersDeg.length;
      angleIndex += 1
    ) {
      const value = profile.density[depthIndex]?.[angleIndex] ?? 0;
      if (value < 0.015) continue;
      const angle = profile.orientationCentersDeg[angleIndex];
      const x0 = mapX(depth - depthStep * 0.55);
      const x1 = mapX(depth + depthStep * 0.55);
      const y0 = mapY(angle + angleStep * 0.6);
      const y1 = mapY(angle - angleStep * 0.6);
      const alpha = 0.04 + Math.pow(value, 0.7) * 0.73;
      context.fillStyle = `hsla(${28 + angle * 1.05} 88% 62% / ${alpha})`;
      context.fillRect(x0, y0, Math.max(1, x1 - x0), Math.max(1, y1 - y0));
    }
  }

  context.setLineDash([3, 4]);
  context.strokeStyle = "rgba(255, 193, 116, 0.55)";
  context.lineWidth = 1;
  context.beginPath();
  context.moveTo(mapX(0), margin.top);
  context.lineTo(mapX(0), margin.top + plotHeight);
  context.stroke();
  context.setLineDash([]);

  for (const needle of needles) {
    const x = mapX(needle.normalCoordinate);
    const y = mapY(needle.familyAngleDeg);
    context.fillStyle = needle.inlier
      ? `hsla(${28 + needle.familyAngleDeg * 1.05} 96% 78% / ${0.4 + needle.score * 0.55})`
      : "rgba(235, 110, 91, 0.34)";
    context.beginPath();
    context.arc(x, y, needle.inlier ? 1.25 + needle.robustWeight * 0.65 : 1, 0, Math.PI * 2);
    context.fill();
  }

  for (const slice of profile.slices) {
    for (const peak of slice.dominantAngles) {
      context.fillStyle = `rgba(225, 255, 250, ${0.34 + peak.relativeStrength * 0.55})`;
      context.beginPath();
      context.arc(
        mapX(slice.normalCoordinate),
        mapY(peak.angleDeg),
        1.1 + peak.relativeStrength * 1.1,
        0,
        Math.PI * 2,
      );
      context.fill();
    }
  }
  context.restore();

  context.strokeStyle = "rgba(222, 235, 230, 0.2)";
  context.fillStyle = "rgba(229, 239, 234, 0.64)";
  context.lineWidth = 1;
  context.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (const angle of [0, 45, 90, 135, 180]) {
    const y = mapY(angle);
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
    context.fillText(String(angle), margin.left - 6, y);
  }
  context.textAlign = "center";
  context.textBaseline = "top";
  for (const depth of [minimumDepth, 0, maximumDepth]) {
    context.fillText(Math.round(depth).toString(), mapX(depth), margin.top + plotHeight + 7);
  }
  context.fillStyle = "rgba(235, 244, 239, 0.78)";
  context.font = "10px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillText("signed position along n (voxels)", margin.left + plotWidth * 0.5, height - 13);
  context.save();
  context.translate(11, margin.top + plotHeight * 0.5);
  context.rotate(-Math.PI * 0.5);
  context.fillText("orientation θ (deg)", 0, 0);
  context.restore();
  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  context.fillStyle = "rgba(235, 244, 239, 0.82)";
  context.font = "700 10px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillText("ORIENTATION ALONG n", margin.left, 17);
  context.textAlign = "right";
  context.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = "rgba(224, 248, 242, 0.72)";
  context.fillText(
    `${Math.round(profile.stats.meanTwoModeCoverage * 100)}% two-mode · ${Math.round(
      profile.stats.coveredDepthFraction * 100,
    )}% depth`,
    width - margin.right,
    17,
  );
  context.fillStyle = "rgba(225, 238, 233, 0.52)";
  context.fillText("unsigned 0° ≡ 180°", width - margin.right, 30);
}

function OrientationProfileView({ result }: { result: AcusResult }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawOrientationProfile(canvas, result.orientationProfile, result.needles);
  }, [result]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  const profile = result.orientationProfile;
  return (
    <div className="orientation-profile">
      <canvas
        ref={canvasRef}
        role="img"
        aria-label={`Unsigned needle orientation by signed position along the recovered normal. ${Math.round(
          profile.stats.meanTwoModeCoverage * 100,
        )} percent of supported orientation density lies near the two strongest local modes.`}
      />
    </div>
  );
}

function fieldCellColor(cell: AcusFieldCell, metric: "normal" | "profile" | "lag") {
  if (!cell.valid) return "rgba(145, 151, 149, 0.14)";
  if (metric === "normal") {
    const value = Math.min(cell.normalAngleDeg ?? 30, 20) / 20;
    return `hsla(${158 - value * 145} 66% 45% / ${0.24 + value * 0.28})`;
  }
  if (metric === "profile") {
    const value = clamp(cell.profileCorrelation ?? 0, 0, 1);
    return `hsla(${15 + value * 140} 66% 45% / ${0.18 + value * 0.38})`;
  }
  const lag = cell.bestDepthLagVoxels ?? 0;
  const magnitude = Math.min(Math.abs(lag) / 12, 1);
  const hue = lag < 0 ? 190 : 32;
  return `hsla(${hue} 72% 50% / ${0.14 + magnitude * 0.5})`;
}

function fieldCellValue(cell: AcusFieldCell, metric: "normal" | "profile" | "lag") {
  if (!cell.valid) return "—";
  if (metric === "normal") return `${(cell.normalAngleDeg ?? 0).toFixed(1)}°`;
  if (metric === "profile") return `${Math.round((cell.profileCorrelation ?? 0) * 100)}%`;
  const value = cell.bestDepthLagVoxels ?? 0;
  return `${value > 0 ? "+" : ""}${value.toFixed(1)}`;
}

function FieldMetricMap({
  result,
  metric,
  label,
}: {
  result: AcusFieldResult;
  metric: "normal" | "profile" | "lag";
  label: string;
}) {
  return (
    <div className="field-map">
      <span className="field-map-label">{label}</span>
      <div className="field-cell-grid">
        {result.cells.map((cell) => (
          <div
            className="field-cell"
            data-anchor={cell.isAnchor ? "true" : "false"}
            key={`${metric}-${cell.row}-${cell.column}`}
            style={{ background: fieldCellColor(cell, metric) }}
            aria-label={`${label}, U offset ${cell.offsetU}, V offset ${cell.offsetV}: ${
              cell.valid ? fieldCellValue(cell, metric) : cell.error || "fit unavailable"
            }`}
          >
            <strong>{fieldCellValue(cell, metric)}</strong>
            {cell.isAnchor ? <small>seed</small> : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function AcusFieldPanel({ result }: { result: AcusFieldResult }) {
  const normal = result.stats.medianNormalAngleDeg;
  const correlation = result.stats.medianProfileCorrelation;
  const lag = result.stats.medianAbsoluteDepthLagVoxels;
  return (
    <section className="acus-field-panel" aria-label="Acus tangent-plane neighborhood consistency">
      <div className="field-panel-heading">
        <div>
          <strong>3 × 3 tangent field</strong>
          <span>
            {result.grid.spacingVoxels}-voxel spacing · {result.stats.computeBackend.toUpperCase()}{" "}
            {result.stats.lineFieldBatchSize}-context batch
          </span>
        </div>
        <p>
          median Δn {normal === null ? "—" : `${normal.toFixed(1)}°`} · profile{" "}
          {correlation === null ? "—" : `${Math.round(correlation * 100)}%`} · |lag|{" "}
          {lag === null ? "—" : `${lag.toFixed(1)} vox`} · {(result.stats.elapsedMs / 1000).toFixed(1)} s
        </p>
      </div>
      <div className="field-map-grid">
        <FieldMetricMap result={result} metric="normal" label="Normal deviation" />
        <FieldMetricMap result={result} metric="profile" label="Profile match" />
        <FieldMetricMap result={result} metric="lag" label="Best depth lag" />
      </div>
      <p className="field-warning">Overlap warning: sweep center spacing before interpreting coherence.</p>
    </section>
  );
}

function drawAuditPlot(canvas: HTMLCanvasElement, result: AcusAuditResult) {
  const bounds = canvas.getBoundingClientRect();
  if (!bounds.width || !bounds.height) return;
  const ratio = Math.min(window.devicePixelRatio || 1, 2);
  const width = Math.max(1, Math.round(bounds.width));
  const height = Math.max(1, Math.round(bounds.height));
  if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
    canvas.width = Math.round(width * ratio);
    canvas.height = Math.round(height * ratio);
  }
  const context = canvas.getContext("2d");
  if (!context) return;
  context.setTransform(ratio, 0, 0, ratio, 0, 0);
  context.clearRect(0, 0, width, height);
  const margin = { left: 36, right: 12, top: 30, bottom: 29 };
  const plotWidth = Math.max(width - margin.left - margin.right, 1);
  const plotHeight = Math.max(height - margin.top - margin.bottom, 1);
  const minimumSpacing = Math.min(...result.spacings);
  const maximumSpacing = Math.max(...result.spacings);
  const spacingRange = Math.max(maximumSpacing - minimumSpacing, 1);
  const mapX = (value: number) =>
    margin.left + ((value - minimumSpacing) / spacingRange) * plotWidth;
  const mapY = (value: number) => margin.top + (1 - clamp(value, 0, 1)) * plotHeight;

  context.strokeStyle = "rgba(67, 82, 83, 0.2)";
  context.fillStyle = "rgba(67, 82, 83, 0.72)";
  context.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.textAlign = "right";
  context.textBaseline = "middle";
  for (const value of [0, 0.25, 0.5, 0.75, 1]) {
    const y = mapY(value);
    context.beginPath();
    context.moveTo(margin.left, y);
    context.lineTo(margin.left + plotWidth, y);
    context.stroke();
    context.fillText(`${Math.round(value * 100)}`, margin.left - 5, y);
  }
  context.textAlign = "center";
  context.textBaseline = "top";
  for (const sweep of result.sweeps) {
    context.fillText(String(sweep.spacingVoxels), mapX(sweep.spacingVoxels), margin.top + plotHeight + 6);
  }

  const series = [
    { key: "medianProfileCorrelation", color: "#176c70", dash: [] as number[] },
    { key: "medianProfileNull", color: "#8b7770", dash: [4, 4] },
    { key: "medianOverlapFraction", color: "#d87b31", dash: [2, 3] },
  ] as const;
  for (const line of series) {
    context.strokeStyle = line.color;
    context.fillStyle = line.color;
    context.lineWidth = 2;
    context.setLineDash([...line.dash]);
    context.beginPath();
    let started = false;
    for (const sweep of result.sweeps) {
      const value = sweep[line.key];
      if (value === null) continue;
      const x = mapX(sweep.spacingVoxels);
      const y = mapY(value);
      if (!started) {
        context.moveTo(x, y);
        started = true;
      } else {
        context.lineTo(x, y);
      }
    }
    context.stroke();
    context.setLineDash([]);
    for (const sweep of result.sweeps) {
      const value = sweep[line.key];
      if (value === null) continue;
      context.beginPath();
      context.arc(mapX(sweep.spacingVoxels), mapY(value), 2.7, 0, Math.PI * 2);
      context.fill();
    }
  }

  context.textAlign = "left";
  context.textBaseline = "alphabetic";
  context.font = "700 10px ui-monospace, SFMono-Regular, Menlo, monospace";
  context.fillStyle = "#26393b";
  context.fillText("COHERENCE VS CENTER SPACING", margin.left, 16);
  context.font = "9px ui-monospace, SFMono-Regular, Menlo, monospace";
  const legend = [
    ["#176c70", "profile"],
    ["#8b7770", "shuffled null"],
    ["#d87b31", "cube overlap"],
  ] as const;
  let legendX = margin.left;
  for (const [color, label] of legend) {
    context.fillStyle = color;
    context.fillRect(legendX, 21, 10, 2);
    context.fillStyle = "rgba(67, 82, 83, 0.78)";
    context.fillText(label, legendX + 14, 25);
    legendX += context.measureText(label).width + 31;
  }
}

function AuditPlot({ result }: { result: AcusAuditResult }) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    drawAuditPlot(canvas, result);
  }, [result]);
  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    draw();
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);
  return (
    <canvas
      ref={canvasRef}
      role="img"
      aria-label="Profile agreement, shuffled-depth null agreement, and cube overlap across the audited center spacings"
    />
  );
}

function auditPercent(value: number | null) {
  return value === null ? "—" : `${Math.round(value * 100)}%`;
}

function AcusAuditPanel({ result }: { result: AcusAuditResult }) {
  const furthest = result.sweeps[result.sweeps.length - 1];
  return (
    <section className="acus-audit-panel" aria-label="Acus spacing and uncertainty audit">
      <div className="audit-panel-heading">
        <div>
          <strong>Spacing and uncertainty audit</strong>
          <span>
            N={result.cube.size} · {result.stats.bootstrapRepetitions} block bootstraps ·{" "}
            {result.stats.nullRepetitions} depth shuffles ·{" "}
            {result.stats.computeBackend.toUpperCase()}
          </span>
        </div>
        <p>
          lowest overlap {auditPercent(furthest.medianOverlapFraction)} · excess profile{" "}
          {furthest.medianProfileExcess === null
            ? "—"
            : `${furthest.medianProfileExcess >= 0 ? "+" : ""}${Math.round(
                furthest.medianProfileExcess * 100,
              )} points`} · {(result.stats.elapsedMs / 1000).toFixed(1)} s
        </p>
      </div>
      <div className="audit-body">
        <div className="audit-plot">
          <AuditPlot result={result} />
        </div>
        <div className="audit-table-wrap">
          <table className="audit-table">
            <thead>
              <tr>
                <th>Step</th>
                <th>Overlap</th>
                <th>Δn / boot p90</th>
                <th>Profile / null</th>
                <th>Excess</th>
                <th>p≤.05</th>
              </tr>
            </thead>
            <tbody>
              {result.sweeps.map((sweep) => (
                <tr key={sweep.spacingVoxels}>
                  <th>{sweep.spacingVoxels}</th>
                  <td>{auditPercent(sweep.medianOverlapFraction)}</td>
                  <td>
                    {sweep.medianNormalAngleDeg?.toFixed(1) ?? "—"}° /{" "}
                    {sweep.medianNormalBootstrapP90Deg?.toFixed(1) ?? "—"}°
                  </td>
                  <td>
                    {auditPercent(sweep.medianProfileCorrelation)} /{" "}
                    {auditPercent(sweep.medianProfileNull)}
                  </td>
                  <td>
                    {sweep.medianProfileExcess === null
                      ? "—"
                      : `${sweep.medianProfileExcess >= 0 ? "+" : ""}${Math.round(
                          sweep.medianProfileExcess * 100,
                        )}`}
                  </td>
                  <td>{auditPercent(sweep.significantProfileFraction)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
      <p className="audit-warning">
        Null: transported neighbor depth rows are shuffled. Spacing changes overlap and sampled
        location together; this is evidence triage, not a sheet claim.
      </p>
    </section>
  );
}

function PaddingAuditPanel({ result }: { result: PaddingAuditResult }) {
  const sufficient = result.sweeps.filter((sweep) => sweep.paddingSufficient);
  const firstSufficient = sufficient[0];
  const reference = result.sweeps.find(
    (sweep) => sweep.requestedPadding === result.referencePadding,
  );
  return (
    <section className="padding-audit-panel" aria-label="Needle context halo stability audit">
      <div className="padding-panel-heading">
        <div>
          <strong>Halo and face-bias audit</strong>
          <span>
            needle L={result.needleLength.toFixed(0)} · reference halo {result.referencePadding}
          </span>
        </div>
        <p>
          first sufficient halo {firstSufficient?.requestedPadding ?? "—"} · face tangent{" "}
          {firstSufficient?.boundaryTangentialFraction === null ||
          firstSufficient?.boundaryTangentialFraction === undefined
            ? "—"
            : `${Math.round(firstSufficient.boundaryTangentialFraction * 100)}%`} →{" "}
          {reference?.boundaryTangentialFraction === null ||
          reference?.boundaryTangentialFraction === undefined
            ? "—"
            : `${Math.round(reference.boundaryTangentialFraction * 100)}%`} ·{" "}
          {(result.stats.elapsedMs / 1000).toFixed(1)} s
        </p>
      </div>
      <div className="padding-table-wrap">
        <table className="audit-table padding-table">
          <thead>
            <tr>
              <th>Halo</th>
              <th>Context</th>
              <th>Sufficient</th>
              <th>Boundary tangent</th>
              <th>Axial coverage</th>
              <th>Δn to ref</th>
              <th>Profile to ref</th>
            </tr>
          </thead>
          <tbody>
            {result.sweeps.map((sweep) => (
              <tr key={sweep.requestedPadding}>
                <th>{sweep.requestedPadding}</th>
                <td>{sweep.contextSize}³</td>
                <td data-kind={sweep.paddingSufficient ? "pass" : "warn"}>
                  {sweep.paddingSufficient ? "yes" : `no (<${sweep.minimumPadding})`}
                </td>
                <td>
                  {sweep.boundaryTangentialFraction === null
                    ? "—"
                    : `${Math.round(sweep.boundaryTangentialFraction * 100)}%`}
                </td>
                <td>{Math.round(sweep.medianAxialCoverage * 100)}%</td>
                <td>{sweep.normalAngleToReferenceDeg.toFixed(2)}°</td>
                <td>{Math.round(sweep.profileCorrelationToReference * 100)}%</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {result.failures.length ? (
        <p className="padding-warning">Some halos exceeded the loaded volume and were not fit.</p>
      ) : (
        <p className="padding-warning">
          Boundary tangent means a boundary-near needle lies within 15° of its nearest cube face.
        </p>
      )}
    </section>
  );
}

const REGION_DEFAULT_ORBIT: Orbit = { yaw: -0.68, pitch: 0.48, zoom: 1.15 };

function projectRegionPoint(
  point: Point3,
  shape: Seed,
  orbit: Orbit,
  width: number,
  height: number,
) {
  const span = Math.max(shape.x, shape.y, shape.z);
  const x = (point[0] - shape.x * 0.5) / span;
  const y = (point[1] - shape.y * 0.5) / span;
  const z = (point[2] - shape.z * 0.5) / span;
  const cosYaw = Math.cos(orbit.yaw);
  const sinYaw = Math.sin(orbit.yaw);
  const cosPitch = Math.cos(orbit.pitch);
  const sinPitch = Math.sin(orbit.pitch);
  const viewX = cosYaw * x - sinYaw * z;
  const yawZ = sinYaw * x + cosYaw * z;
  const viewY = cosPitch * y + sinPitch * yawZ;
  const viewZ = -sinPitch * y + cosPitch * yawZ;
  const distance = 1.55 - viewZ;
  if (distance <= 0.01) return null;
  const aspect = width / height;
  const ndcX = ((viewX / distance) * 1.7 * orbit.zoom) / aspect;
  const ndcY = (viewY / distance) * 1.7 * orbit.zoom;
  return {
    x: (ndcX + 1) * width * 0.5,
    y: (1 - ndcY) * height * 0.5,
    depth: distance,
  };
}

function regionMetricValue(cell: RegionCell, metric: RegionMetric) {
  if (metric === "normal") {
    return 1 - clamp((cell.neighborNormalMedianDeg ?? 12) / 12, 0, 1);
  }
  if (metric === "pattern") return clamp(cell.neighborPatternMedian ?? 0, 0, 1);
  if (metric === "depth") return clamp(cell.coveredDepthFraction ?? 0, 0, 1);
  return clamp(cell.normalConfidence ?? 0, 0, 1);
}

function buildFlakeTracks(result: FlakeResult | null, minimumScore: number) {
  if (!result) {
    return {
      activeLinks: [] as FlakeResult["links"],
      byFlake: new Map<number, { id: number; size: number }>(),
      linkedTrackCount: 0,
      largestTrackSize: 0,
    };
  }
  const parent = result.flakes.map((_, index) => index);
  const find = (initial: number) => {
    let index = initial;
    while (parent[index] !== index) {
      parent[index] = parent[parent[index]];
      index = parent[index];
    }
    return index;
  };
  const activeLinks = result.links.filter((link) => link.score >= minimumScore);
  for (const link of activeLinks) {
    const sourceRoot = find(link.source);
    const targetRoot = find(link.target);
    if (sourceRoot !== targetRoot) parent[targetRoot] = sourceRoot;
  }
  const members = new Map<number, number[]>();
  for (let index = 0; index < result.flakes.length; index += 1) {
    const root = find(index);
    const values = members.get(root) ?? [];
    values.push(index);
    members.set(root, values);
  }
  const byFlake = new Map<number, { id: number; size: number }>();
  let linkedTrackCount = 0;
  let largestTrackSize = 0;
  for (const [root, values] of members) {
    if (values.length >= 2) linkedTrackCount += 1;
    largestTrackSize = Math.max(largestTrackSize, values.length);
    for (const index of values) byFlake.set(index, { id: root, size: values.length });
  }
  return { activeLinks, byFlake, linkedTrackCount, largestTrackSize };
}

function flakeHue(
  flake: FlakeResult["flakes"][number],
  colorMode: FlakeColorMode,
  track: { id: number; size: number } | undefined,
  cubeSize: number,
) {
  if (colorMode === "depth") {
    const phase = clamp(flake.depthOffset / Math.max(cubeSize * 0.5, 1) * 0.5 + 0.5, 0, 1);
    return 24 + phase * 210;
  }
  if (colorMode === "fiber") {
    return flake.fiberAngleXYDeg === null ? 195 : 18 + (flake.fiberAngleXYDeg / 180) * 300;
  }
  if (colorMode === "track") {
    return track && track.size >= 2 ? (track.id * 137.508) % 360 : 195;
  }
  if (colorMode === "sheetlet") {
    return flake.sheetletSize && flake.sheetletSize >= 2 && flake.sheetletId !== undefined
      ? (flake.sheetletId * 137.508) % 360
      : 195;
  }
  if (colorMode === "validation") {
    return 18 + clamp((flake.validationScore ?? 0) / 0.28, 0, 1) * 158;
  }
  return 18 + flake.quality * 158;
}

export function RegionOverview({
  result,
  title = "Volume-scale evidence field",
  defaultVectorMode = "normal",
  defaultOrbit = REGION_DEFAULT_ORBIT,
  granularity = 1,
  glyphScale = 1,
  flakeResult = null,
  flakeLoading = false,
  auditSweep = null,
  selectedSeed,
  onSelectSeed,
}: {
  result: RegionResult;
  title?: string;
  defaultVectorMode?: RegionVectorMode;
  defaultOrbit?: Orbit;
  granularity?: number;
  glyphScale?: number;
  flakeResult?: FlakeResult | null;
  flakeLoading?: boolean;
  auditSweep?: FlakeAuditSweep | null;
  selectedSeed?: Seed;
  onSelectSeed?: (seed: Seed) => void;
}) {
  const headingId = useId();
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const projectedCellsRef = useRef<Array<{ cell: RegionCell; x: number; y: number }>>([]);
  const projectedFlakesRef = useRef<
    Array<{ flake: FlakeResult["flakes"][number]; x: number; y: number }>
  >([]);
  const dragRef = useRef<{
    x: number;
    y: number;
    startX: number;
    startY: number;
    moved: boolean;
  } | null>(null);
  const [orbit, setOrbit] = useState(defaultOrbit);
  const [metric, setMetric] = useState<RegionMetric>("normal");
  const [vectorMode, setVectorMode] = useState<RegionVectorMode>(defaultVectorMode);
  const [flakeColorMode, setFlakeColorMode] = useState<FlakeColorMode>(
    defaultVectorMode === "sheetlets" ? "sheetlet" : "depth",
  );
  const [trackThreshold, setTrackThreshold] = useState(
    flakeResult?.settings.defaultTrackScore ?? 0.12,
  );
  const [selectedFlakeId, setSelectedFlakeId] = useState<number | null>(null);
  const flakeTracks = useMemo(
    () => buildFlakeTracks(flakeResult, trackThreshold),
    [flakeResult, trackThreshold],
  );
  const sheetletTracks = useMemo(
    () =>
      buildFlakeTracks(
        flakeResult?.sheetletLinks
          ? { ...flakeResult, links: flakeResult.sheetletLinks }
          : null,
        flakeResult?.settings.defaultTrackScore ?? 0.12,
      ),
    [flakeResult],
  );
  const selectedFlake =
    selectedFlakeId === null ? null : flakeResult?.flakes[selectedFlakeId] ?? null;

  const draw = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(bounds.width));
    const height = Math.max(1, Math.round(bounds.height));
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
      canvas.width = Math.round(width * ratio);
      canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    context.clearRect(0, 0, width, height);
    const gradient = context.createRadialGradient(
      width * 0.5,
      height * 0.42,
      10,
      width * 0.5,
      height * 0.5,
      Math.max(width, height) * 0.7,
    );
    gradient.addColorStop(0, "#153034");
    gradient.addColorStop(1, "#071315");
    context.fillStyle = gradient;
    context.fillRect(0, 0, width, height);

    const corners: Point3[] = [
      [0, 0, 0],
      [result.shape.x, 0, 0],
      [0, result.shape.y, 0],
      [result.shape.x, result.shape.y, 0],
      [0, 0, result.shape.z],
      [result.shape.x, 0, result.shape.z],
      [0, result.shape.y, result.shape.z],
      [result.shape.x, result.shape.y, result.shape.z],
    ];
    const cornerPoints = corners.map((point) =>
      projectRegionPoint(point, result.shape, orbit, width, height),
    );
    const edges = [
      [0, 1], [0, 2], [0, 4], [1, 3], [1, 5], [2, 3],
      [2, 6], [3, 7], [4, 5], [4, 6], [5, 7], [6, 7],
    ];
    context.strokeStyle = "rgba(190, 223, 216, 0.22)";
    context.lineWidth = 1;
    for (const [first, second] of edges) {
      const start = cornerPoints[first];
      const end = cornerPoints[second];
      if (!start || !end) continue;
      context.beginPath();
      context.moveTo(start.x, start.y);
      context.lineTo(end.x, end.y);
      context.stroke();
    }

    context.lineCap = "round";
    if ((vectorMode === "flakes" || vectorMode === "tracks" || vectorMode === "sheetlets") && flakeResult) {
      projectedCellsRef.current = [];
      const displayedFlakes = flakeResult.flakes.filter((flake) => {
        if (vectorMode === "tracks") {
          return (flakeTracks.byFlake.get(flake.id)?.size ?? 0) >= 2;
        }
        if (vectorMode === "sheetlets") {
          return (flake.sheetletSize ?? 0) >= 2;
        }
        return (
          flake.cellIndex[0] % granularity === 0 &&
          flake.cellIndex[1] % granularity === 0
        );
      });
      const displayedIds = new Set(displayedFlakes.map((flake) => flake.id));
      if (vectorMode === "tracks" || vectorMode === "sheetlets") {
        const visibleTracks = vectorMode === "sheetlets" ? sheetletTracks : flakeTracks;
        for (const link of visibleTracks.activeLinks) {
          if (!displayedIds.has(link.source) || !displayedIds.has(link.target)) continue;
          const source = projectRegionPoint(
            flakeResult.flakes[link.source].center,
            result.shape,
            orbit,
            width,
            height,
          );
          const target = projectRegionPoint(
            flakeResult.flakes[link.target].center,
            result.shape,
            orbit,
            width,
            height,
          );
          if (!source || !target) continue;
          const strength = clamp(
            (link.score - trackThreshold) / Math.max(0.34 - trackThreshold, 0.02),
            0,
            1,
          );
          context.strokeStyle = `rgba(255, 220, 151, ${0.12 + strength * 0.48})`;
          context.lineWidth = 0.65 + strength * 1.25;
          context.beginPath();
          context.moveTo(source.x, source.y);
          context.lineTo(target.x, target.y);
          context.stroke();
        }
      }
      const flakeScale = 0.6 + glyphScale * 1.8;
      const projectedFlakes = displayedFlakes
        .map((flake) => {
          const radiusFiber = flake.radiusFiber * flakeScale;
          const radiusCross = flake.radiusCrossFiber * flakeScale;
          const corners: Point3[] = [
            [
              flake.center[0] - flake.fiber[0] * radiusFiber - flake.crossFiber[0] * radiusCross,
              flake.center[1] - flake.fiber[1] * radiusFiber - flake.crossFiber[1] * radiusCross,
              flake.center[2] - flake.fiber[2] * radiusFiber - flake.crossFiber[2] * radiusCross,
            ],
            [
              flake.center[0] + flake.fiber[0] * radiusFiber - flake.crossFiber[0] * radiusCross,
              flake.center[1] + flake.fiber[1] * radiusFiber - flake.crossFiber[1] * radiusCross,
              flake.center[2] + flake.fiber[2] * radiusFiber - flake.crossFiber[2] * radiusCross,
            ],
            [
              flake.center[0] + flake.fiber[0] * radiusFiber + flake.crossFiber[0] * radiusCross,
              flake.center[1] + flake.fiber[1] * radiusFiber + flake.crossFiber[1] * radiusCross,
              flake.center[2] + flake.fiber[2] * radiusFiber + flake.crossFiber[2] * radiusCross,
            ],
            [
              flake.center[0] - flake.fiber[0] * radiusFiber + flake.crossFiber[0] * radiusCross,
              flake.center[1] - flake.fiber[1] * radiusFiber + flake.crossFiber[1] * radiusCross,
              flake.center[2] - flake.fiber[2] * radiusFiber + flake.crossFiber[2] * radiusCross,
            ],
          ];
          const center = projectRegionPoint(flake.center, result.shape, orbit, width, height);
          const projectedCorners = corners.map((corner) =>
            projectRegionPoint(corner, result.shape, orbit, width, height),
          );
          const fiberStart = projectRegionPoint(
            [
              flake.center[0] - flake.fiber[0] * radiusFiber,
              flake.center[1] - flake.fiber[1] * radiusFiber,
              flake.center[2] - flake.fiber[2] * radiusFiber,
            ],
            result.shape,
            orbit,
            width,
            height,
          );
          const fiberEnd = projectRegionPoint(
            [
              flake.center[0] + flake.fiber[0] * radiusFiber,
              flake.center[1] + flake.fiber[1] * radiusFiber,
              flake.center[2] + flake.fiber[2] * radiusFiber,
            ],
            result.shape,
            orbit,
            width,
            height,
          );
          if (!center || !fiberStart || !fiberEnd || projectedCorners.some((point) => !point)) {
            return null;
          }
          return {
            flake,
            center,
            corners: projectedCorners as Array<NonNullable<(typeof projectedCorners)[number]>>,
            fiberStart,
            fiberEnd,
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
        .sort((left, right) => right.center.depth - left.center.depth);
      projectedFlakesRef.current = projectedFlakes.map(({ flake, center }) => ({
        flake,
        x: center.x,
        y: center.y,
      }));
      for (const { flake, center, corners: flakeCorners, fiberStart, fiberEnd } of projectedFlakes) {
        const track =
          vectorMode === "sheetlets"
            ? sheetletTracks.byFlake.get(flake.id)
            : flakeTracks.byFlake.get(flake.id);
        const hue = flakeHue(
          flake,
          flakeColorMode,
          track,
          flakeResult.settings.cubeSize,
        );
        const isolated = vectorMode === "tracks" && (!track || track.size < 2);
        context.fillStyle = `hsla(${hue} 74% 57% / ${isolated ? 0.035 : 0.08 + flake.quality * 0.24})`;
        context.strokeStyle = `hsla(${hue} 88% 70% / ${isolated ? 0.12 : 0.32 + flake.quality * 0.58})`;
        context.lineWidth = selectedFlakeId === flake.id ? 1.8 : 0.55 + flake.quality * 1.05;
        context.beginPath();
        context.moveTo(flakeCorners[0].x, flakeCorners[0].y);
        for (let index = 1; index < flakeCorners.length; index += 1) {
          context.lineTo(flakeCorners[index].x, flakeCorners[index].y);
        }
        context.closePath();
        context.fill();
        context.stroke();
        context.strokeStyle = `hsla(${hue} 96% 78% / ${isolated ? 0.18 : 0.5 + flake.quality * 0.42})`;
        context.lineWidth = selectedFlakeId === flake.id ? 2.1 : 0.7 + flake.quality * 1.15;
        context.beginPath();
        context.moveTo(fiberStart.x, fiberStart.y);
        context.lineTo(fiberEnd.x, fiberEnd.y);
        context.stroke();
        if (selectedFlakeId === flake.id) {
          context.strokeStyle = "rgba(255, 239, 194, 0.98)";
          context.lineWidth = 1.4;
          context.beginPath();
          context.arc(center.x, center.y, 5, 0, Math.PI * 2);
          context.stroke();
        }
      }
    } else {
      projectedFlakesRef.current = [];
      const glyphLength = Math.max(
        4,
        result.settings.gridStride * 0.38,
        Math.max(result.shape.x, result.shape.y, result.shape.z) * 0.006,
      ) * glyphScale;
      const glyphs = result.cells
        .filter(
          (cell) =>
            cell.index[0] % granularity === 0 && cell.index[1] % granularity === 0,
        )
        .filter((cell): cell is RegionCell & { normal: Point3 } => cell.valid && Boolean(cell.normal))
        .map((cell) => {
          const normal = cell.normal;
          const xyMagnitude = Math.hypot(normal[0], normal[1]);
          if (vectorMode === "slice-tangent" && xyMagnitude < 0.25) return null;
          const vector: Point3 =
            vectorMode === "slice-tangent"
              ? [-normal[1] / xyMagnitude, normal[0] / xyMagnitude, 0]
              : normal;
          const start: Point3 = [
            cell.center[0] - vector[0] * glyphLength,
            cell.center[1] - vector[1] * glyphLength,
            cell.center[2] - vector[2] * glyphLength,
          ];
          const end: Point3 = [
            cell.center[0] + vector[0] * glyphLength,
            cell.center[1] + vector[1] * glyphLength,
            cell.center[2] + vector[2] * glyphLength,
          ];
          return {
            cell,
            center: projectRegionPoint(cell.center, result.shape, orbit, width, height),
            start: projectRegionPoint(start, result.shape, orbit, width, height),
            end: projectRegionPoint(end, result.shape, orbit, width, height),
          };
        })
        .filter((entry): entry is NonNullable<typeof entry> => entry !== null)
        .filter(
          (entry): entry is typeof entry & {
            center: NonNullable<typeof entry.center>;
            start: NonNullable<typeof entry.start>;
            end: NonNullable<typeof entry.end>;
          } => Boolean(entry.center && entry.start && entry.end),
        )
        .sort((left, right) => right.center.depth - left.center.depth);

      projectedCellsRef.current = glyphs.map(({ cell, center }) => ({
        cell,
        x: center.x,
        y: center.y,
      }));
      for (const { cell, center, start, end } of glyphs) {
        const value = regionMetricValue(cell, metric);
        const hue = 18 + value * 158;
        context.strokeStyle = `hsla(${hue} 78% 66% / ${0.28 + value * 0.62})`;
        context.lineWidth = 0.8 + value * 1.7;
        context.beginPath();
        context.moveTo(start.x, start.y);
        context.lineTo(end.x, end.y);
        context.stroke();
        context.fillStyle = `hsla(${hue} 88% 72% / ${0.42 + value * 0.52})`;
        context.beginPath();
        context.arc(center.x, center.y, 1.2 + value * 1.3, 0, Math.PI * 2);
        context.fill();
      }
    }

    const selected = selectedSeed
      ? projectRegionPoint(
          [selectedSeed.x, selectedSeed.y, selectedSeed.z],
          result.shape,
          orbit,
          width,
          height,
        )
      : null;
    if (selected) {
      context.strokeStyle = "rgba(255, 203, 125, 0.98)";
      context.lineWidth = 1.8;
      context.beginPath();
      context.arc(selected.x, selected.y, 7, 0, Math.PI * 2);
      context.stroke();
      context.fillStyle = "rgba(255, 225, 172, 0.98)";
      context.beginPath();
      context.arc(selected.x, selected.y, 2.2, 0, Math.PI * 2);
      context.fill();
    }
  }, [
    flakeColorMode,
    flakeResult,
    flakeTracks,
    sheetletTracks,
    glyphScale,
    granularity,
    metric,
    orbit,
    result,
    selectedFlakeId,
    selectedSeed,
    trackThreshold,
    vectorMode,
  ]);

  useEffect(() => {
    draw();
    const canvas = canvasRef.current;
    if (!canvas) return;
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => observer.disconnect();
  }, [draw]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const sensitivity = event.ctrlKey || event.metaKey ? 0.018 : 0.002;
      setOrbit((value) => ({
        ...value,
        zoom: clamp(value.zoom * Math.exp(-event.deltaY * sensitivity), 0.45, 8),
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
    drag.moved ||= Math.hypot(event.clientX - drag.startX, event.clientY - drag.startY) > 4;
    setOrbit((value) => ({
      ...value,
      yaw: value.yaw + deltaX * 0.009,
      pitch: clamp(value.pitch + deltaY * 0.009, -1.5, 1.5),
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
    const x = event.clientX - bounds.left;
    const y = event.clientY - bounds.top;
    if (vectorMode === "flakes" || vectorMode === "tracks" || vectorMode === "sheetlets") {
      const nearestFlake = projectedFlakesRef.current.reduce<{
        flake: FlakeResult["flakes"][number];
        distance: number;
      } | null>((best, projected) => {
        const distance = Math.hypot(projected.x - x, projected.y - y);
        return !best || distance < best.distance
          ? { flake: projected.flake, distance }
          : best;
      }, null);
      setSelectedFlakeId(
        nearestFlake && nearestFlake.distance <= 15 ? nearestFlake.flake.id : null,
      );
      return;
    }
    const nearest = projectedCellsRef.current.reduce<{
      cell: RegionCell;
      distance: number;
    } | null>((best, projected) => {
      const distance = Math.hypot(projected.x - x, projected.y - y);
      return !best || distance < best.distance ? { cell: projected.cell, distance } : best;
    }, null);
    if (nearest && nearest.distance <= 15 && onSelectSeed) {
      onSelectSeed({
        x: nearest.cell.center[0],
        y: nearest.cell.center[1],
        z: nearest.cell.center[2],
      });
    }
  };

  const metricLabels: Array<[RegionMetric, string]> = [
    ["normal", "Normal stability"],
    ["pattern", "Pattern"],
    ["confidence", "Confidence"],
    ["depth", "Depth support"],
  ];
  return (
    <section className="region-overview" aria-labelledby={headingId}>
      <div className="region-heading">
        <div>
          <strong id={headingId}>{title}</strong>
          <span>
            {Math.ceil(result.grid.x.length / granularity)} × {Math.ceil(result.grid.y.length / granularity)} × {result.grid.z.length} displayed · N=
            {result.settings.cubeSize} · display step {result.settings.gridStride * granularity} · halo {result.settings.effectivePadding}
          </span>
        </div>
        <p>
          {result.stats.validCellCount}/{result.stats.cellCount} valid · {result.stats.needleCount.toLocaleString()} needles · median Δn {result.stats.medianNeighborNormalDeg?.toFixed(1) ?? "—"}° · pattern {result.stats.medianNeighborPattern === null ? "—" : `${Math.round(result.stats.medianNeighborPattern * 100)}%`} · {result.stats.cacheHit ? "cached" : `${(result.stats.elapsedMs / 1000).toFixed(1)} s`} · {result.stats.computeBackend.toUpperCase()}
          {result.stats.medianMacroRadialResidualDeg !== undefined &&
          result.stats.medianMacroRadialResidualDeg !== null
            ? ` · radial ${result.stats.medianMacroRadialResidualDeg.toFixed(1)}° vs ${result.stats.macroRadialNullMedianDeg?.toFixed(1) ?? "—"}° shuffled`
            : ""}
          {flakeResult
            ? auditSweep
              ? ` · ${flakeResult.stats.flakeCount.toLocaleString()} flakes · ${auditSweep.spacingVoxels}-vox links · fiber Δ ${auditSweep.observed.medianFiberAngleDeg?.toFixed(1) ?? "—"}° vs ${auditSweep.nulls.fiber.medianFiberAngleDeg?.toFixed(1) ?? "—"}° rematched null · ${Math.round((auditSweep.observed.medianSharedNeedleFraction ?? 0) * 100)}% shared evidence`
              : ` · ${flakeResult.stats.flakeCount.toLocaleString()} flakes · fiber Δ ${flakeResult.stats.medianFiberAngleDeg?.toFixed(1) ?? "—"}° vs ${flakeResult.stats.fiberShuffledMedianDeg?.toFixed(1) ?? "—"}° shuffled · ${Math.round((flakeResult.stats.medianSharedNeedleFraction ?? 0) * 100)}% shared evidence`
            : flakeLoading
              ? " · fitting flakes…"
              : ""}
        </p>
      </div>
      <div className="region-controls">
        <div className="region-control-group" aria-label="Displayed field primitive">
          <span>Display</span>
          <button
            className="region-metric-button"
            data-active={vectorMode === "normal"}
            type="button"
            onClick={() => setVectorMode("normal")}
          >
            Normal n
          </button>
          <button
            className="region-metric-button"
            data-active={vectorMode === "slice-tangent"}
            type="button"
            onClick={() => setVectorMode("slice-tangent")}
          >
            XY page tangent t
          </button>
          <button
            className="region-metric-button"
            data-active={vectorMode === "flakes"}
            type="button"
            onClick={() => setVectorMode("flakes")}
            disabled={!flakeResult}
          >
            Fiber flakes
          </button>
          <button
            className="region-metric-button"
            data-active={vectorMode === "tracks"}
            type="button"
            onClick={() => setVectorMode("tracks")}
            disabled={!flakeResult}
          >
            Linked tracks
          </button>
          <button
            className="region-metric-button"
            data-active={vectorMode === "sheetlets"}
            type="button"
            onClick={() => {
              setVectorMode("sheetlets");
              setFlakeColorMode("sheetlet");
            }}
            disabled={!flakeResult?.sheetletLinks}
          >
            3D sheetlets
          </button>
        </div>
        <div className="region-control-group" aria-label="Evidence color metric">
          <span>Color</span>
          {vectorMode === "flakes" || vectorMode === "tracks" || vectorMode === "sheetlets"
            ? ([
                ["depth", "Depth phase"],
                ["fiber", "Fiber direction"],
                ["quality", "Fit quality"],
                ["validation", "Held-out fit"],
                ["track", "Track identity"],
                ["sheetlet", "Sheetlet identity"],
              ] as Array<[FlakeColorMode, string]>).map(([value, label]) => (
                <button
                  className="region-metric-button"
                  data-active={flakeColorMode === value}
                  key={value}
                  type="button"
                  onClick={() => setFlakeColorMode(value)}
                >
                  {label}
                </button>
              ))
            : metricLabels.map(([value, label]) => (
                <button
                  className="region-metric-button"
                  data-active={metric === value}
                  key={value}
                  type="button"
                  onClick={() => setMetric(value)}
                >
                  {label}
                </button>
              ))}
        </div>
        {vectorMode === "tracks" ? (
          <label className="region-track-threshold">
            <span>Link evidence</span>
            <strong>{trackThreshold.toFixed(2)}</strong>
            <input
              type="range"
              min={0.05}
              max={0.3}
              step={0.01}
              value={trackThreshold}
              onChange={(event) => setTrackThreshold(Number(event.target.value))}
            />
          </label>
        ) : null}
        <button className="region-metric-button" type="button" onClick={() => setOrbit(defaultOrbit)}>
          Reset view
        </button>
      </div>
      <div className="region-canvas-wrap">
        <canvas
          ref={canvasRef}
          role="img"
          aria-label={`Orbitable loaded-volume Acus ${
            vectorMode === "normal"
              ? "normal"
              : vectorMode === "slice-tangent"
                ? "XY page-tangent"
                : vectorMode === "flakes"
                  ? "fiber flake"
                  : vectorMode === "tracks"
                    ? "linked flake track"
                    : "3D sheetlet"
          } evidence field`}
          onPointerDown={handlePointerDown}
          onPointerMove={handlePointerMove}
          onPointerUp={handlePointerUp}
          onPointerCancel={handlePointerUp}
        />
      </div>
      <p className="region-legend">
        {vectorMode === "normal"
          ? "Glyphs show the unsigned recovered normal n. "
          : vectorMode === "slice-tangent"
            ? "Glyphs show t = rotate90(projectXY(n)); weak XY projections are omitted. "
            : vectorMode === "flakes"
              ? "Finite patches are joint depth–orientation modes; the bright axis is the unsigned fitted fiber direction. Click a patch for its fit. "
              : vectorMode === "tracks"
                ? `${flakeTracks.activeLinks.length.toLocaleString()} mutual ${auditSweep?.spacingVoxels ?? 32}-voxel links form ${flakeTracks.linkedTrackCount.toLocaleString()} multi-flake tracks at this threshold; unlinked flakes are hidden. `
                : `${sheetletTracks.activeLinks.length.toLocaleString()} in-plane edges belong to 3D components built only from held-out-replicated flakes at non-overlapping 64-voxel X/Y/Z spacing; isolated flakes are hidden. `}
        Drag to orbit and pinch or wheel to zoom{onSelectSeed ? ", then click a glyph to inspect that seed locally" : ""}. Flakes remain hypotheses; no physical sheets are assigned.
      </p>
      {selectedFlake ? (
        <p className="region-flake-detail">
          Flake {selectedFlake.id.toLocaleString()} · cell {selectedFlake.cellIndex[0]},{" "}
          {selectedFlake.cellIndex[1]} · depth {selectedFlake.depthOffset.toFixed(1)} vox · {selectedFlake.needleCount} needles · fiber residual {selectedFlake.medianFiberResidualDeg.toFixed(1)}° · thickness {selectedFlake.thickness.toFixed(1)} vox · quality {Math.round(selectedFlake.quality * 100)}%
          {selectedFlake.validationScore !== undefined
            ? ` · held-out ${selectedFlake.validated ? "replicated" : "not replicated"} at ${selectedFlake.validationScore.toFixed(3)}${selectedFlake.foldFiberDeltaDeg !== undefined ? ` · fold Δ ${selectedFlake.foldFiberDeltaDeg.toFixed(1)}° / ${selectedFlake.foldDepthDeltaVoxels?.toFixed(1) ?? "—"} vox` : ""}`
            : ""}
          {selectedFlake.sheetletSize && selectedFlake.sheetletSize >= 2
            ? ` · sheetlet ${selectedFlake.sheetletId} · ${selectedFlake.sheetletSize} flakes · Z span ${selectedFlake.sheetletZSpanVoxels?.toFixed(0) ?? "—"} vox`
            : ""}
        </p>
      ) : null}
    </section>
  );
}

function SlabJobPanel({ status }: { status: SlabStatus }) {
  const fetchComplete = status.fetch?.state === "complete";
  const analysisState = status.analysis?.state;
  let label = "Preparing source slab";
  let completed = status.fetch?.completedCount ?? 0;
  let total = status.fetch?.totalCount ?? 1;
  let detail = `${completed.toLocaleString()} / ${total.toLocaleString()} source chunks`;
  if (fetchComplete && !analysisState) {
    label = "Source slab ready";
    completed = total = 1;
    detail = "GPU analysis will start after source validation";
  } else if (analysisState === "calibrating") {
    label = "Calibrating global ridge strength";
    completed = status.analysis?.calibrationCompletedCount ?? 0;
    total = status.analysis?.calibrationTileCount ?? 96;
    detail = `${completed.toLocaleString()} / ${total.toLocaleString()} material-bearing samples`;
  } else if (analysisState?.includes("extract")) {
    label = "Extracting finite needles";
    completed = status.analysis?.completedTileCount ?? 0;
    total = status.analysis?.tileCount ?? 1;
    detail = `${completed.toLocaleString()} / ${total.toLocaleString()} haloed GPU tiles`;
  } else if (analysisState?.includes("summar")) {
    label = "Summarizing local Acus fields";
    completed = status.analysis?.completedCellCount ?? 0;
    total = status.analysis?.cellCount ?? 1;
    detail = `${completed.toLocaleString()} / ${total.toLocaleString()} N³ centers`;
  } else if (analysisState === "complete") {
    label = "Cross-scroll field complete";
    completed = total = 1;
    detail = `${(status.analysis?.validCellCount ?? 0).toLocaleString()} valid cells · ${(status.analysis?.needleCount ?? 0).toLocaleString()} needles`;
  }
  const progress = clamp(completed / Math.max(total, 1), 0, 1);
  const shape = status.source?.shapeZYX ?? status.shapeZYX;
  return (
    <section className="slab-job-panel" aria-label="Cross-scroll slab analysis progress">
      <div className="slab-job-heading">
        <div>
          <span>Native cross-scroll experiment</span>
          <strong>{label}</strong>
        </div>
        <p>
          {shape ? `${shape[2].toLocaleString()} × ${shape[1].toLocaleString()} × ${shape[0].toLocaleString()} voxels` : "Full transverse slab"}
          {status.settings?.gridStride ? ` · step ${status.settings.gridStride}` : ""}
        </p>
      </div>
      <div className="slab-progress-track" aria-label={`${Math.round(progress * 100)} percent complete`}>
        <span style={{ width: `${progress * 100}%` }} />
      </div>
      <div className="slab-job-detail">
        <span>{detail}</span>
        <span>
          {analysisState?.includes("extract") && status.analysis?.tilesPerSecondThisRun
            ? `${status.analysis.tilesPerSecondThisRun.toFixed(2)} tile/s`
            : analysisState?.includes("summar") && status.analysis?.cellsPerSecondThisRun
              ? `${status.analysis.cellsPerSecondThisRun.toFixed(0)} cell/s`
              : status.fetch?.downloadMiBPerSecond
                ? `${status.fetch.downloadMiBPerSecond.toFixed(1)} MiB/s`
                : "resumable"}
        </span>
      </div>
    </section>
  );
}

export function VolumeScene({
  apiBase,
  dimensions,
  seed,
  onSelectSeed,
}: {
  apiBase: string;
  dimensions: Seed;
  seed: Seed;
  onSelectSeed: (seed: Seed) => void;
}) {
  const canvasRef = useRef<HTMLCanvasElement>(null);
  const overlayRef = useRef<HTMLCanvasElement>(null);
  const rendererRef = useRef<Renderer | null>(null);
  const dragRef = useRef<{ x: number; y: number } | null>(null);
  const analysisAbortRef = useRef<AbortController | null>(null);
  const fieldAbortRef = useRef<AbortController | null>(null);
  const auditAbortRef = useRef<AbortController | null>(null);
  const paddingAuditAbortRef = useRef<AbortController | null>(null);
  const regionAbortRef = useRef<AbortController | null>(null);
  const [cubeSize, setCubeSize] = useState(64);
  const [needleLength, setNeedleLength] = useState(16);
  const [contextPadding, setContextPadding] = useState(16);
  const [orbit, setOrbit] = useState(DEFAULT_ORBIT);
  const [threshold, setThreshold] = useState(0.28);
  const [density, setDensity] = useState(4.2);
  const [status, setStatus] = useState("Waiting for local voxels…");
  const [volumeReady, setVolumeReady] = useState(false);
  const [acusResult, setAcusResult] = useState<AcusResult | null>(null);
  const [analysisState, setAnalysisState] = useState<"idle" | "running" | "error">("idle");
  const [analysisMessage, setAnalysisMessage] = useState("");
  const [analysisKey, setAnalysisKey] = useState("");
  const [fieldSpacing, setFieldSpacing] = useState(8);
  const [fieldResult, setFieldResult] = useState<AcusFieldResult | null>(null);
  const [fieldState, setFieldState] = useState<"idle" | "running" | "error">("idle");
  const [fieldMessage, setFieldMessage] = useState("");
  const [fieldKey, setFieldKey] = useState("");
  const [auditResult, setAuditResult] = useState<AcusAuditResult | null>(null);
  const [auditState, setAuditState] = useState<"idle" | "running" | "error">("idle");
  const [auditMessage, setAuditMessage] = useState("");
  const [auditKey, setAuditKey] = useState("");
  const [paddingAuditResult, setPaddingAuditResult] = useState<PaddingAuditResult | null>(null);
  const [paddingAuditState, setPaddingAuditState] = useState<"idle" | "running" | "error">(
    "idle",
  );
  const [paddingAuditMessage, setPaddingAuditMessage] = useState("");
  const [paddingAuditKey, setPaddingAuditKey] = useState("");
  const [regionStride, setRegionStride] = useState(16);
  const [regionResult, setRegionResult] = useState<RegionResult | null>(null);
  const [regionState, setRegionState] = useState<"idle" | "running" | "error">("idle");
  const [regionMessage, setRegionMessage] = useState("");
  const [regionKey, setRegionKey] = useState("");
  const [slabStatusData, setSlabStatusData] = useState<SlabStatus | null>(null);
  const [slabMessage, setSlabMessage] = useState("");
  const currentKey = `${seed.x}:${seed.y}:${seed.z}:${cubeSize}:${needleLength}:${contextPadding}`;
  const currentFieldKey = `${currentKey}:${fieldSpacing}`;
  const currentRegionKey = `${cubeSize}:${needleLength}:${contextPadding}:${regionStride}`;
  const visibleAcusResult =
    acusResult &&
    acusResult.seed.x === seed.x &&
    acusResult.seed.y === seed.y &&
    acusResult.seed.z === seed.z &&
    acusResult.cube.size === cubeSize &&
    acusResult.settings.needleLength === needleLength &&
    acusResult.settings.requestedPadding === contextPadding
      ? acusResult
      : null;
  const visibleAnalysisState = analysisKey === currentKey ? analysisState : "idle";
  const visibleFieldResult =
    fieldResult &&
    fieldResult.seed.x === seed.x &&
    fieldResult.seed.y === seed.y &&
    fieldResult.seed.z === seed.z &&
    fieldResult.cube.size === cubeSize &&
    fieldResult.grid.spacingVoxels === fieldSpacing &&
    fieldResult.anchor.settings.needleLength === needleLength &&
    fieldResult.anchor.settings.requestedPadding === contextPadding
      ? fieldResult
      : null;
  const visibleFieldState = fieldKey === currentFieldKey ? fieldState : "idle";
  const visibleAuditResult =
    auditResult &&
    auditResult.seed.x === seed.x &&
    auditResult.seed.y === seed.y &&
    auditResult.seed.z === seed.z &&
    auditResult.cube.size === cubeSize &&
    auditResult.anchor.settings.needleLength === needleLength &&
    auditResult.anchor.settings.requestedPadding === contextPadding
      ? auditResult
      : null;
  const visibleAuditState = auditKey === currentKey ? auditState : "idle";
  const visiblePaddingAuditResult =
    paddingAuditResult &&
    paddingAuditResult.seed.x === seed.x &&
    paddingAuditResult.seed.y === seed.y &&
    paddingAuditResult.seed.z === seed.z &&
    paddingAuditResult.cube.size === cubeSize &&
    paddingAuditResult.needleLength === needleLength
      ? paddingAuditResult
      : null;
  const visiblePaddingAuditState =
    paddingAuditKey === currentKey ? paddingAuditState : "idle";
  const visibleRegionResult =
    regionResult &&
    regionResult.settings.cubeSize === cubeSize &&
    regionResult.settings.needleLength === needleLength &&
    regionResult.settings.requestedPadding === contextPadding &&
    regionResult.settings.gridStride === regionStride
      ? regionResult
      : null;
  const visibleRegionState = regionKey === currentRegionKey ? regionState : "idle";
  const renderStateRef = useRef<RenderState>({
    orbit,
    threshold,
    density,
    steps: 96,
  });
  const draw = useCallback(() => {
    const renderer = rendererRef.current;
    const canvas = canvasRef.current;
    if (!renderer || !canvas) return;
    drawRenderer(renderer, canvas, renderStateRef.current);
  }, []);

  const drawOverlay = useCallback(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    drawAcusOverlay(overlay, visibleAcusResult, visibleFieldResult, cubeSize, orbit);
  }, [cubeSize, orbit, visibleAcusResult, visibleFieldResult]);

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    try {
      rendererRef.current = createRenderer(canvas);
      draw();
    } catch (error) {
      queueMicrotask(() =>
        setStatus(error instanceof Error ? error.message : "Volume renderer initialization failed."),
      );
      return;
    }
    const observer = new ResizeObserver(draw);
    observer.observe(canvas);
    return () => {
      observer.disconnect();
      const renderer = rendererRef.current;
      if (renderer) {
        renderer.gl.deleteTexture(renderer.texture);
        renderer.gl.deleteVertexArray(renderer.vao);
        renderer.gl.deleteProgram(renderer.program);
      }
      rendererRef.current = null;
    };
  }, [draw]);

  useEffect(() => {
    if (!apiBase) return;
    const controller = new AbortController();
    const timer = window.setTimeout(async () => {
      setVolumeReady(false);
      setStatus(`Loading ${cubeSize}³ voxels…`);
      const query = new URLSearchParams({
        x: String(seed.x),
        y: String(seed.y),
        z: String(seed.z),
        size: String(cubeSize),
      });
      try {
        const response = await fetch(`${apiBase}/api/cube?${query}`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Local volume returned ${response.status}.`);
        const data = new Uint8Array(await response.arrayBuffer());
        const expected = cubeSize ** 3;
        if (data.length !== expected) {
          throw new Error(`Local volume returned ${data.length} bytes; expected ${expected}.`);
        }
        const renderer = rendererRef.current;
        if (!renderer) throw new Error("Volume renderer is unavailable.");
        const { gl } = renderer;
        gl.activeTexture(gl.TEXTURE0);
        gl.bindTexture(gl.TEXTURE_3D, renderer.texture);
        gl.pixelStorei(gl.UNPACK_ALIGNMENT, 1);
        gl.texImage3D(
          gl.TEXTURE_3D,
          0,
          gl.R8,
          cubeSize,
          cubeSize,
          cubeSize,
          0,
          gl.RED,
          gl.UNSIGNED_BYTE,
          data,
        );
        renderer.hasVolume = true;
        setVolumeReady(true);
        setStatus(`${cubeSize} × ${cubeSize} × ${cubeSize} voxels centered at seed`);
        draw();
      } catch (error) {
        if (controller.signal.aborted) return;
        setStatus(error instanceof Error ? error.message : "Local volume request failed.");
      }
    }, 160);
    return () => {
      window.clearTimeout(timer);
      controller.abort();
    };
  }, [apiBase, cubeSize, draw, seed.x, seed.y, seed.z]);

  useEffect(() => {
    if (!apiBase) return;
    const controller = new AbortController();
    let timer: number | undefined;
    const refresh = async () => {
      try {
        const response = await fetch(`${apiBase}/api/slab/status`, {
          signal: controller.signal,
        });
        if (!response.ok) throw new Error(`Slab status returned ${response.status}.`);
        const payload = (await response.json()) as SlabStatus;
        if (!controller.signal.aborted) {
          setSlabStatusData(payload);
          setSlabMessage("");
        }
      } catch (error) {
        if (!controller.signal.aborted) {
          setSlabMessage(error instanceof Error ? error.message : "Slab status is unavailable.");
        }
      } finally {
        if (!controller.signal.aborted) timer = window.setTimeout(refresh, 5000);
      }
    };
    void refresh();
    return () => {
      controller.abort();
      if (timer !== undefined) window.clearTimeout(timer);
    };
  }, [apiBase]);

  useEffect(() => {
    renderStateRef.current = {
      orbit,
      threshold,
      density,
      steps: Math.min(256, Math.max(48, Math.round(cubeSize * 1.5))),
    };
    draw();
  }, [cubeSize, density, draw, orbit, threshold]);

  useEffect(() => {
    const overlay = overlayRef.current;
    if (!overlay) return;
    drawOverlay();
    const observer = new ResizeObserver(drawOverlay);
    observer.observe(overlay);
    return () => observer.disconnect();
  }, [drawOverlay]);

  useEffect(() => {
    analysisAbortRef.current?.abort();
    analysisAbortRef.current = null;
  }, [currentKey]);

  useEffect(() => {
    paddingAuditAbortRef.current?.abort();
    paddingAuditAbortRef.current = null;
  }, [currentKey]);

  useEffect(() => {
    fieldAbortRef.current?.abort();
    fieldAbortRef.current = null;
  }, [currentFieldKey]);

  useEffect(() => {
    auditAbortRef.current?.abort();
    auditAbortRef.current = null;
  }, [currentKey]);

  useEffect(() => {
    regionAbortRef.current?.abort();
    regionAbortRef.current = null;
  }, [currentRegionKey]);

  useEffect(
    () => () => {
      analysisAbortRef.current?.abort();
      fieldAbortRef.current?.abort();
      auditAbortRef.current?.abort();
      paddingAuditAbortRef.current?.abort();
      regionAbortRef.current?.abort();
    },
    [],
  );

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const handleWheel = (event: WheelEvent) => {
      event.preventDefault();
      const sensitivity = event.ctrlKey || event.metaKey ? 0.018 : 0.002;
      setOrbit((value) => ({
        ...value,
        zoom: clamp(value.zoom * Math.exp(-event.deltaY * sensitivity), 0.45, 8),
      }));
    };
    canvas.addEventListener("wheel", handleWheel, { passive: false });
    return () => canvas.removeEventListener("wheel", handleWheel);
  }, []);

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
      pitch: clamp(value.pitch + deltaY * 0.009, -1.5, 1.5),
    }));
  };

  const handlePointerUp = (event: ReactPointerEvent<HTMLCanvasElement>) => {
    dragRef.current = null;
    if (event.currentTarget.hasPointerCapture(event.pointerId)) {
      event.currentTarget.releasePointerCapture(event.pointerId);
    }
  };

  const fitNeedles = async () => {
    analysisAbortRef.current?.abort();
    const controller = new AbortController();
    analysisAbortRef.current = controller;
    setAnalysisKey(currentKey);
    setAnalysisState("running");
    setAnalysisMessage("Extracting line-like ridges and solving the shared normal…");
    try {
      const response = await fetch(`${apiBase}/api/needles`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed,
          cubeSize,
          scale: 1.25,
          spacing: 4,
          maxNeedles: 160,
          needleLength,
          contextPadding,
        }),
        signal: controller.signal,
      });
      const payload = (await response.json()) as AcusResult & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Acus returned ${response.status}.`);
      if (controller.signal.aborted) return;
      setAcusResult(payload);
      setAnalysisState("idle");
      setAnalysisMessage("");
    } catch (error) {
      if (controller.signal.aborted) return;
      setAnalysisState("error");
      setAnalysisMessage(error instanceof Error ? error.message : "Acus analysis failed.");
    } finally {
      if (analysisAbortRef.current === controller) analysisAbortRef.current = null;
    }
  };

  const fitField = async () => {
    fieldAbortRef.current?.abort();
    const controller = new AbortController();
    fieldAbortRef.current = controller;
    setFieldKey(currentFieldKey);
    setFieldState("running");
    setFieldMessage(`Fitting nine independent cubes at ${fieldSpacing}-voxel spacing…`);
    try {
      const response = await fetch(`${apiBase}/api/field`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed,
          cubeSize,
          scale: 1.25,
          spacing: 4,
          maxNeedles: 160,
          gridSize: 3,
          fieldSpacing,
          needleLength,
          contextPadding,
        }),
        signal: controller.signal,
      });
      const payload = (await response.json()) as AcusFieldResult & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Acus field returned ${response.status}.`);
      if (controller.signal.aborted) return;
      setAcusResult(payload.anchor);
      setFieldResult(payload);
      setFieldState("idle");
      setFieldMessage("");
    } catch (error) {
      if (controller.signal.aborted) return;
      setFieldState("error");
      setFieldMessage(error instanceof Error ? error.message : "Acus field analysis failed.");
    } finally {
      if (fieldAbortRef.current === controller) fieldAbortRef.current = null;
    }
  };

  const runAudit = async () => {
    auditAbortRef.current?.abort();
    const controller = new AbortController();
    auditAbortRef.current = controller;
    setAuditKey(currentKey);
    setAuditState("running");
    setAuditMessage("Auditing five spacings with block bootstraps and shuffled-depth nulls…");
    try {
      const response = await fetch(`${apiBase}/api/audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed,
          cubeSize,
          scale: 1.25,
          spacing: 4,
          maxNeedles: 160,
          fieldSpacings: [4, 8, 16, 24, 32],
          bootstrapRepetitions: 48,
          nullRepetitions: 32,
          needleLength,
          contextPadding,
        }),
        signal: controller.signal,
      });
      const payload = (await response.json()) as AcusAuditResult & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Acus audit returned ${response.status}.`);
      if (controller.signal.aborted) return;
      setAcusResult(payload.anchor);
      setAuditResult(payload);
      setAuditState("idle");
      setAuditMessage("");
    } catch (error) {
      if (controller.signal.aborted) return;
      setAuditState("error");
      setAuditMessage(error instanceof Error ? error.message : "Acus spacing audit failed.");
    } finally {
      if (auditAbortRef.current === controller) auditAbortRef.current = null;
    }
  };

  const runPaddingAudit = async () => {
    paddingAuditAbortRef.current?.abort();
    const controller = new AbortController();
    const halfLengthHalo = clamp(Math.ceil((needleLength * 0.5) / 4) * 4, 0, 48);
    const largerHalo = clamp(Math.ceil((needleLength * 1.5) / 4) * 4, 0, 48);
    const paddingValues = Array.from(
      new Set([0, halfLengthHalo, needleLength, largerHalo]),
    ).sort((left, right) => left - right);
    paddingAuditAbortRef.current = controller;
    setPaddingAuditKey(currentKey);
    setPaddingAuditState("running");
    setPaddingAuditMessage(`Testing ${paddingValues.join(", ")} voxel context halos…`);
    try {
      const response = await fetch(`${apiBase}/api/padding-audit`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          seed,
          cubeSize,
          scale: 1.25,
          spacing: 4,
          maxNeedles: 160,
          needleLength,
          paddingValues,
        }),
        signal: controller.signal,
      });
      const payload = (await response.json()) as PaddingAuditResult & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Halo audit returned ${response.status}.`);
      if (controller.signal.aborted) return;
      setPaddingAuditResult(payload);
      setPaddingAuditState("idle");
      setPaddingAuditMessage("");
    } catch (error) {
      if (controller.signal.aborted) return;
      setPaddingAuditState("error");
      setPaddingAuditMessage(error instanceof Error ? error.message : "Acus halo audit failed.");
    } finally {
      if (paddingAuditAbortRef.current === controller) paddingAuditAbortRef.current = null;
    }
  };

  const runRegionBake = async () => {
    regionAbortRef.current?.abort();
    const controller = new AbortController();
    regionAbortRef.current = controller;
    setRegionKey(currentRegionKey);
    setRegionState("running");
    setRegionMessage("Analyzing finite needles and local summaries across the loaded volume…");
    try {
      const response = await fetch(`${apiBase}/api/region`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          cubeSize,
          scale: 1.25,
          spacing: 4,
          maxNeedles: 160,
          needleLength,
          contextPadding,
          gridStride: regionStride,
          tileCore: 56,
          catalogBinSize: 32,
          maxNeedlesPerBin: 32,
        }),
        signal: controller.signal,
      });
      const payload = (await response.json()) as RegionResult & { error?: string };
      if (!response.ok) throw new Error(payload.error || `Volume analysis returned ${response.status}.`);
      if (controller.signal.aborted) return;
      setRegionResult(payload);
      setRegionState("idle");
      setRegionMessage("");
    } catch (error) {
      if (controller.signal.aborted) return;
      setRegionState("error");
      setRegionMessage(error instanceof Error ? error.message : "Loaded-volume analysis failed.");
    } finally {
      if (regionAbortRef.current === controller) regionAbortRef.current = null;
    }
  };

  const acusStatus = visibleAcusResult
    ? `${visibleAcusResult.stats.needleCount} L${visibleAcusResult.settings.needleLength.toFixed(0)} needles · halo ${visibleAcusResult.settings.effectivePadding} · ${Math.round(
        visibleAcusResult.stats.medianAxialCoverage * 100,
      )}% axial support · ${visibleAcusResult.stats.inlierCount} inliers · ${Math.round(
        visibleAcusResult.stats.normalConfidence * 100,
      )}% normal confidence · ${visibleAcusResult.stats.medianPlaneResidualDeg.toFixed(1)}° median off-plane · ${visibleAcusResult.stats.elapsedMs.toFixed(0)} ms`
    : analysisKey === currentKey
      ? analysisMessage
      : "";
  const fieldStatus =
    visibleFieldState === "running" || visibleFieldState === "error"
      ? fieldKey === currentFieldKey
        ? fieldMessage
        : acusStatus
      : acusStatus;
  const resultStatus =
    visibleAuditState === "running" || visibleAuditState === "error"
      ? auditKey === currentKey
        ? auditMessage
        : fieldStatus
      : fieldStatus;
  const localStatus =
    visiblePaddingAuditState === "running" || visiblePaddingAuditState === "error"
      ? paddingAuditKey === currentKey
        ? paddingAuditMessage
        : resultStatus
      : resultStatus;
  const finalStatus = localStatus;

  return (
    <>
    <section className="viewport-card local-volume-card" aria-labelledby="volume-view-title">
      <div className="viewport-header volume-header">
        <div className="viewport-title">
          <h3 id="volume-view-title">Local Acus inspector</h3>
          <span className="plane-code">N³</span>
        </div>
        <div className="volume-toolbar">
          <label title="Minimum normalized intensity contributing opacity">
            <span>Threshold</span>
            <input
              type="range"
              min={0.05}
              max={0.75}
              step={0.01}
              value={threshold}
              onChange={(event) => setThreshold(Number(event.target.value))}
            />
          </label>
          <label title="Opacity accumulated through the cube">
            <span>Density</span>
            <input
              type="range"
              min={0.5}
              max={12}
              step={0.1}
              value={density}
              onChange={(event) => setDensity(Number(event.target.value))}
            />
          </label>
          <button className="quiet-button" type="button" onClick={() => setOrbit(DEFAULT_ORBIT)}>
            Reset view
          </button>
        </div>
      </div>
      <div className="local-analysis-controls">
        <div className="local-control-group">
          <strong>Needle fit</strong>
          <label>
            <span>Cube N</span>
            <input
              className="cube-size-input"
              type="number"
              min={16}
              max={128}
              step={8}
              value={cubeSize}
              onChange={(event) =>
                setCubeSize(clamp(Math.round(Number(event.target.value) / 8) * 8, 16, 128))
              }
            />
          </label>
          <label title="Finite support length used to validate every ridge needle">
            <span>Needle L</span>
            <input
              className="cube-size-input needle-length-input"
              type="number"
              min={6}
              max={48}
              step={2}
              value={needleLength}
              onChange={(event) => {
                const nextLength = clamp(Math.round(Number(event.target.value) / 2) * 2, 6, 48);
                setNeedleLength(nextLength);
                setContextPadding(nextLength);
              }}
            />
          </label>
          <label title="Real scan-data halo loaded around the inner analysis cube">
            <span>Halo</span>
            <input
              className="cube-size-input context-padding-input"
              type="number"
              min={0}
              max={48}
              step={2}
              value={contextPadding}
              onChange={(event) =>
                setContextPadding(clamp(Math.round(Number(event.target.value) / 2) * 2, 0, 48))
              }
            />
          </label>
          <button
            className="quiet-button acus-button"
            type="button"
            onClick={fitNeedles}
            disabled={
              !volumeReady ||
              visibleAnalysisState === "running" ||
              visibleFieldState === "running" ||
              visibleAuditState === "running" ||
              visiblePaddingAuditState === "running" ||
              visibleRegionState === "running"
            }
            title="Fit unsigned needle-like ridge primitives and their shared orthogonal direction"
          >
            {visibleAnalysisState === "running" ? "Fitting…" : "Fit Acus"}
          </button>
        </div>
        <div className="local-control-group">
          <strong>Neighborhood checks</strong>
          <label title="Center-to-center spacing of the 3 by 3 tangent-plane field">
            <span>Step</span>
            <input
              className="cube-size-input field-spacing-input"
              type="number"
              min={4}
              max={32}
              step={4}
              value={fieldSpacing}
              onChange={(event) =>
                setFieldSpacing(clamp(Math.round(Number(event.target.value) / 4) * 4, 4, 32))
              }
            />
          </label>
          <button
            className="quiet-button field-button"
            type="button"
            onClick={fitField}
            disabled={
              !visibleAcusResult ||
              visibleFieldState === "running" ||
              visibleAnalysisState === "running" ||
              visibleAuditState === "running" ||
              visiblePaddingAuditState === "running" ||
              visibleRegionState === "running"
            }
            title="Fit a 3 by 3 tangent-plane neighborhood and compare normals and depth profiles"
          >
            {visibleFieldState === "running" ? "Mapping…" : "Map 3×3"}
          </button>
          <button
            className="quiet-button audit-button"
            type="button"
            onClick={runAudit}
            disabled={
              !visibleAcusResult ||
              visibleAnalysisState === "running" ||
              visibleFieldState === "running" ||
              visibleAuditState === "running" ||
              visiblePaddingAuditState === "running" ||
              visibleRegionState === "running"
            }
            title="Sweep 4 to 32 voxel field spacing with block-bootstrap and shuffled-depth controls"
          >
            {visibleAuditState === "running" ? "Auditing…" : "Audit 4–32"}
          </button>
          <button
            className="quiet-button padding-audit-button"
            type="button"
            onClick={runPaddingAudit}
            disabled={
              !visibleAcusResult ||
              visibleAnalysisState === "running" ||
              visibleFieldState === "running" ||
              visibleAuditState === "running" ||
              visiblePaddingAuditState === "running" ||
              visibleRegionState === "running"
            }
            title="Compare face bias and fit stability across 0 to 24 voxels of real context"
          >
            {visiblePaddingAuditState === "running" ? "Testing halo…" : "Audit halo"}
          </button>
        </div>
      </div>
      <div className={`acus-view-grid${visibleAcusResult ? " has-profile" : ""}`}>
        <div className="scene-stage volume-stage">
          <canvas
            ref={canvasRef}
            tabIndex={0}
            role="img"
            aria-label={`Orbitable volumetric rendering of a ${cubeSize} by ${cubeSize} by ${cubeSize} voxel cube centered at the selected seed`}
            onPointerDown={handlePointerDown}
            onPointerMove={handlePointerMove}
            onPointerUp={handlePointerUp}
            onPointerCancel={handlePointerUp}
          />
          <canvas ref={overlayRef} className="volume-overlay" aria-hidden="true" />
          {visibleAcusResult ? (
            <div className="acus-legend" aria-hidden="true">
              <span className="acus-needle-key">ridge needle / orientation</span>
              <span className="acus-normal-key">shared normal n</span>
              {visibleFieldResult ? (
                <span className="acus-field-key">neighbor normals</span>
              ) : null}
              <span className="acus-compute-key">
                {visibleAcusResult.stats.computeBackend.toUpperCase()} dense{" "}
                {visibleAcusResult.stats.lineFieldMs.toFixed(0)} ms
              </span>
            </div>
          ) : null}
          <span className="volume-seed-marker" aria-hidden="true" />
          <p
            className="volume-status"
            data-kind={
              visibleAnalysisState === "error" ||
              visibleFieldState === "error" ||
              visibleAuditState === "error" ||
              visiblePaddingAuditState === "error"
                ? "error"
                : visibleAcusResult
                  ? "result"
                  : "volume"
            }
            role="status"
            aria-live="polite"
          >
            {finalStatus || status} · drag orbit · pinch/wheel zoom · {orbit.zoom.toFixed(1)}×
          </p>
        </div>
        {visibleAcusResult ? <OrientationProfileView result={visibleAcusResult} /> : null}
        {visibleFieldResult ? <AcusFieldPanel result={visibleFieldResult} /> : null}
        {visibleAuditResult ? <AcusAuditPanel result={visibleAuditResult} /> : null}
        {visiblePaddingAuditResult ? (
          <PaddingAuditPanel result={visiblePaddingAuditResult} />
        ) : null}
      </div>
    </section>
    <section className="viewport-card region-analysis-card" aria-labelledby="region-analysis-title">
      <div className="viewport-header region-analysis-header">
        <div className="region-analysis-copy">
          <div className="viewport-title">
            <h3 id="region-analysis-title">Volume-scale analysis</h3>
            <span className="plane-code">XYZ FIELD</span>
          </div>
          <p>
            Track the full cross-scroll slab experiment, or run the same reusable needle catalog
            and local-normal summary on the currently loaded 256³ crop.
          </p>
        </div>
        <div className="region-analysis-actions">
          <label title="Center-to-center spacing of local summaries across the loaded volume">
            <span>Grid step</span>
            <input
              className="cube-size-input region-spacing-input"
              type="number"
              min={8}
              max={64}
              step={8}
              value={regionStride}
              onChange={(event) =>
                setRegionStride(clamp(Math.round(Number(event.target.value) / 8) * 8, 8, 64))
              }
            />
          </label>
          <button
            className="quiet-button region-bake-button"
            type="button"
            onClick={runRegionBake}
            disabled={
              !volumeReady ||
              visibleAnalysisState === "running" ||
              visibleFieldState === "running" ||
              visibleAuditState === "running" ||
              visiblePaddingAuditState === "running" ||
              visibleRegionState === "running"
            }
            title="Analyze a reusable finite-needle catalog and local evidence summaries across the loaded volume"
          >
            {visibleRegionState === "running" ? "Analyzing crop…" : "Analyze current crop"}
          </button>
        </div>
      </div>
      {slabStatusData?.fetch ? <SlabJobPanel status={slabStatusData} /> : null}
      {slabMessage ? <p className="slab-job-message">{slabMessage}</p> : null}
      {slabStatusData?.analysis?.state === "complete" ? (
        <Link className="cross-scroll-launch" href="/cross-scroll">
          <span>Cross-scroll field</span>
          <strong>Open full-window slice explorer →</strong>
        </Link>
      ) : null}
      <p
        className="region-analysis-status"
        data-kind={visibleRegionState === "error" ? "error" : visibleRegionResult ? "result" : "ready"}
        role="status"
        aria-live="polite"
      >
        {visibleRegionState === "running" || visibleRegionState === "error"
          ? regionKey === currentRegionKey
            ? regionMessage
            : "Analysis settings changed."
          : visibleRegionResult
            ? `${visibleRegionResult.stats.validCellCount.toLocaleString()} valid cells · ${visibleRegionResult.stats.needleCount.toLocaleString()} finite needles · ${visibleRegionResult.stats.cacheHit ? "loaded from cache" : `${(visibleRegionResult.stats.elapsedMs / 1000).toFixed(1)} s`}`
            : `${dimensions.x} × ${dimensions.y} × ${dimensions.z} loaded voxels · ready for a ${regionStride}-voxel evidence grid`}
      </p>
      {visibleRegionResult ? (
        <RegionOverview
          result={visibleRegionResult}
          title="Current-crop evidence field"
          selectedSeed={seed}
          onSelectSeed={onSelectSeed}
        />
      ) : (
        <div className="region-analysis-empty">
          <strong>One global pass, many local summaries</strong>
          <p>
            The expensive ridge field is computed once across the loaded volume. N={cubeSize}
            local normals are then summarized every {regionStride} voxels, with a real-data halo
            of {contextPadding} voxels and no inferred sheet connections.
          </p>
        </div>
      )}
    </section>
    </>
  );
}
