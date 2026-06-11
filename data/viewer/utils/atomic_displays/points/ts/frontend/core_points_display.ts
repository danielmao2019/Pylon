import * as THREE from "three";
import {
  createTrackballCameraControls,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import {
  createThreeDisplayContainer,
  createThreePerspectiveCamera,
  createThreeScene,
  createThreeWebGLRenderer,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
import type { LeafVNode, VNode } from "web/reconcile/reconcile";
import type { PointDisplayResponse } from "./types/display_response";

// Smallest world-space point size the display falls back to when no explicit
// size is requested; also the floor the bounding-sphere-relative auto-size is
// clamped against. Exported so consumers can express a requested point size as
// a multiple of it.
export const DEFAULT_POINT_SIZE_FLOOR = 0.005;

// Fraction of the geometry bounding-sphere radius used as the heuristic default
// point size when no explicit pointSize is supplied. Lib-owned default, overridable.
export const DEFAULT_POINT_SIZE_RATIO = 0.002;

// Uniform fallback color used when the geometry has no per-point colors AND the
// caller does not supply pointColor. Lib-owned default, overridable.
export const DEFAULT_POINT_COLOR = "#cccccc";

interface PlyProperty {
  type: string;
  name: string;
}

interface PlyHeader {
  format: string;
  vertexCount: number;
  properties: PlyProperty[];
}

interface PlyPropertyIndices {
  x: number;
  y: number;
  z: number;
  red: number;
  green: number;
  blue: number;
}

interface PlyPropertyOffset {
  offset: number;
  type: string;
}

interface PlyPropertyOffsets {
  stride: number;
  x: PlyPropertyOffset;
  y: PlyPropertyOffset;
  z: PlyPropertyOffset;
  red?: PlyPropertyOffset;
  green?: PlyPropertyOffset;
  blue?: PlyPropertyOffset;
}

export function renderPointsDisplay({
  displayResponse,
  initialCameraState = null,
  pointSize,
  pointColor,
}: {
  displayResponse: PointDisplayResponse;
  initialCameraState?: CameraState | null;
  pointSize?: number;
  pointColor?: string;
}): VNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `points:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createPointsScene({
        displayResponse,
        initialCameraState,
        pointSize,
        pointColor,
      });
      const controls = createTrackballCameraControls({ container, camera, renderer, initialCameraState });
      _registerSceneResize({ container, camera, renderer, controls });
      _publishCameraState({ container, controls });
      controls.addEventListener("change", () => {
        _publishCameraState({ container, controls });
      });
      renderPointsScene({ scene, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

function createPointsScene({
  displayResponse,
  initialCameraState,
  pointSize,
  pointColor,
}: {
  displayResponse: PointDisplayResponse;
  initialCameraState: CameraState | null;
  pointSize?: number;
  pointColor?: string;
}): {
  container: HTMLDivElement;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
} {
  const container = createThreeDisplayContainer({ pointerEventsSuppressed: false });
  const scene = createThreeScene();
  const camera = createThreePerspectiveCamera({ initialCameraState });
  const renderer = createThreeWebGLRenderer({ container });
  loadPointGeometry({ displayResponse })
    .then((geometry) => scene.add(createThreePoints({ geometry, pointSize, pointColor })))
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      container.replaceChildren(_renderPointsStatus(`Failed to load point cloud: ${message}`));
    });
  return { container, scene, camera, renderer };
}

export async function loadPointGeometry({
  displayResponse,
}: {
  displayResponse: PointDisplayResponse;
}): Promise<THREE.BufferGeometry> {
  if (displayResponse.url === null) {
    throw new Error("point display response url is null");
  }
  const response = await fetch(displayResponse.url);
  if (!response.ok) {
    throw new Error(`unable to load point cloud: HTTP ${response.status}`);
  }
  const buffer = await response.arrayBuffer();
  return parsePlyBuffer({ buffer });
}

export function createThreePoints({
  geometry,
  pointSize,
  pointColor,
}: {
  geometry: THREE.BufferGeometry;
  pointSize?: number;
  pointColor?: string;
}): THREE.Points {
  // An explicit requested size pins the world-space point size directly;
  // otherwise fall back to the bounding-sphere-relative heuristic (floored).
  geometry.computeBoundingSphere();
  const boundingRadius = geometry.boundingSphere?.radius ?? 0;
  const effectiveSize =
    pointSize ?? Math.max(DEFAULT_POINT_SIZE_FLOOR, boundingRadius * DEFAULT_POINT_SIZE_RATIO);
  // pointColor (when supplied) replaces per-point colors with a uniform color;
  // otherwise use the geometry's per-point colors, falling back to the lib
  // default uniform color when the geometry carries none.
  let useVertexColors: boolean;
  let effectiveColor: string | undefined;
  if (pointColor !== undefined) {
    useVertexColors = false;
    effectiveColor = pointColor;
  } else if (geometry.hasAttribute("color")) {
    useVertexColors = true;
    effectiveColor = undefined;
  } else {
    useVertexColors = false;
    effectiveColor = DEFAULT_POINT_COLOR;
  }
  const material = new THREE.PointsMaterial({
    vertexColors: useVertexColors,
    size: effectiveSize,
    ...(effectiveColor !== undefined ? { color: effectiveColor } : {}),
  });
  return new THREE.Points(geometry, material);
}

function _registerSceneResize({
  container,
  camera,
  renderer,
  controls,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ThreeTrackballCameraControls;
}): void {
  const resize = () => {
    const width = Math.max(1, container.clientWidth || 1);
    const height = Math.max(1, container.clientHeight || 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
    controls.handleResize();
  };
  resize();
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(resize).observe(container);
  }
  window.addEventListener("resize", resize);
}

function parsePlyBuffer({
  buffer,
}: {
  buffer: ArrayBuffer;
}): THREE.BufferGeometry {
  const headerBytes = buffer.slice(0, Math.min(buffer.byteLength, 1048576));
  const headerText = new TextDecoder("utf-8").decode(headerBytes);
  const endIndex = headerText.indexOf("end_header");
  if (endIndex < 0) {
    throw new Error("PLY header is missing end_header");
  }
  const headerPrefix = headerText.slice(0, endIndex);
  const rawHeader = headerText.slice(0, endIndex + "end_header".length);
  const encoder = new TextEncoder();
  const bytes = new Uint8Array(buffer);
  let dataOffset = encoder.encode(rawHeader).length;
  while (
    dataOffset < bytes.length &&
    (bytes[dataOffset] === 10 || bytes[dataOffset] === 13)
  ) {
    dataOffset += 1;
  }

  const header = readPlyHeader({ headerText: headerPrefix });
  if (header.format === "ascii") {
    return parseAsciiPlyGeometry({ buffer, dataOffset, header });
  }
  if (header.format === "binary_little_endian") {
    return parseBinaryLittleEndianPlyGeometry({
      buffer,
      dataOffset,
      header,
    });
  }
  throw new Error(`unsupported PLY format ${header.format}`);
}

function readPlyHeader({ headerText }: { headerText: string }): PlyHeader {
  const lines = headerText.split(/\r?\n/);
  let format = "";
  let vertexCount = 0;
  let inVertex = false;
  const properties: PlyProperty[] = [];
  for (const line of lines) {
    const parts = line.trim().split(/\s+/);
    if (parts.length === 0 || parts[0] === "") {
      continue;
    }
    if (parts[0] === "format") {
      format = parts[1];
      continue;
    }
    if (parts[0] === "element") {
      inVertex = parts[1] === "vertex";
      if (inVertex) {
        vertexCount = Number(parts[2]);
      }
      continue;
    }
    if (parts[0] === "property" && inVertex) {
      if (parts[1] === "list") {
        throw new Error("vertex list properties are not supported");
      }
      properties.push({ type: parts[1], name: parts[2] });
    }
  }
  if (!format) {
    throw new Error("PLY format is missing");
  }
  if (!Number.isFinite(vertexCount) || vertexCount < 1) {
    throw new Error(`PLY vertex count is invalid: ${vertexCount}`);
  }
  return { format, vertexCount, properties };
}

function parseAsciiPlyGeometry({
  buffer,
  dataOffset,
  header,
}: {
  buffer: ArrayBuffer;
  dataOffset: number;
  header: PlyHeader;
}): THREE.BufferGeometry {
  const dataText = new TextDecoder("utf-8").decode(buffer.slice(dataOffset));
  const lines = dataText.trim().split(/\r?\n/);
  const indices = plyPropertyIndices({ properties: header.properties });
  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  for (let index = 0; index < header.vertexCount; index += 1) {
    const parts = lines[index]?.trim().split(/\s+/);
    if (parts === undefined || parts.length < header.properties.length) {
      throw new Error(`ASCII PLY row is missing vertex data: ${index}`);
    }
    writeGeometryVertex({
      positions,
      colors,
      index,
      x: Number(parts[indices.x]),
      y: Number(parts[indices.y]),
      z: Number(parts[indices.z]),
      red: readAsciiColorComponent({ parts, index: indices.red }),
      green: readAsciiColorComponent({ parts, index: indices.green }),
      blue: readAsciiColorComponent({ parts, index: indices.blue }),
    });
  }
  return createPointBufferGeometry({ positions, colors });
}

function parseBinaryLittleEndianPlyGeometry({
  buffer,
  dataOffset,
  header,
}: {
  buffer: ArrayBuffer;
  dataOffset: number;
  header: PlyHeader;
}): THREE.BufferGeometry {
  const view = new DataView(buffer);
  const offsets = plyPropertyOffsets({ properties: header.properties });
  const positions = new Float32Array(header.vertexCount * 3);
  const colors = new Float32Array(header.vertexCount * 3);
  for (let index = 0; index < header.vertexCount; index += 1) {
    const base = dataOffset + index * offsets.stride;
    writeGeometryVertex({
      positions,
      colors,
      index,
      x: readBinaryScalar({
        view,
        offset: base + offsets.x.offset,
        type: offsets.x.type,
      }),
      y: readBinaryScalar({
        view,
        offset: base + offsets.y.offset,
        type: offsets.y.type,
      }),
      z: readBinaryScalar({
        view,
        offset: base + offsets.z.offset,
        type: offsets.z.type,
      }),
      red: readBinaryColorComponent({ view, base, offset: offsets.red }),
      green: readBinaryColorComponent({ view, base, offset: offsets.green }),
      blue: readBinaryColorComponent({ view, base, offset: offsets.blue }),
    });
  }
  return createPointBufferGeometry({ positions, colors });
}

function plyPropertyIndices({
  properties,
}: {
  properties: PlyProperty[];
}): PlyPropertyIndices {
  const names = properties.map((property) => property.name);
  const x = names.indexOf("x");
  const y = names.indexOf("y");
  const z = names.indexOf("z");
  if (x < 0 || y < 0 || z < 0) {
    throw new Error("PLY vertex coordinates are missing");
  }
  return {
    x,
    y,
    z,
    red: names.indexOf("red"),
    green: names.indexOf("green"),
    blue: names.indexOf("blue"),
  };
}

function plyPropertyOffsets({
  properties,
}: {
  properties: PlyProperty[];
}): PlyPropertyOffsets {
  const offsets: Record<string, PlyPropertyOffset> = {};
  let offset = 0;
  for (const property of properties) {
    offsets[property.name] = { offset, type: property.type };
    offset += plyScalarTypeSize({ type: property.type });
  }
  if (
    offsets.x === undefined ||
    offsets.y === undefined ||
    offsets.z === undefined
  ) {
    throw new Error("PLY vertex coordinates are missing");
  }
  return {
    stride: offset,
    x: offsets.x,
    y: offsets.y,
    z: offsets.z,
    red: offsets.red,
    green: offsets.green,
    blue: offsets.blue,
  };
}

function writeGeometryVertex({
  positions,
  colors,
  index,
  x,
  y,
  z,
  red,
  green,
  blue,
}: {
  positions: Float32Array;
  colors: Float32Array;
  index: number;
  x: number;
  y: number;
  z: number;
  red: number;
  green: number;
  blue: number;
}): void {
  const positionOffset = index * 3;
  positions[positionOffset] = x;
  positions[positionOffset + 1] = y;
  positions[positionOffset + 2] = z;
  colors[positionOffset] = normalizeColorComponent({ value: red });
  colors[positionOffset + 1] = normalizeColorComponent({ value: green });
  colors[positionOffset + 2] = normalizeColorComponent({ value: blue });
}

function createPointBufferGeometry({
  positions,
  colors,
}: {
  positions: Float32Array;
  colors: Float32Array;
}): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.setAttribute("color", new THREE.BufferAttribute(colors, 3));
  geometry.computeBoundingSphere();
  geometry.computeBoundingBox();
  return geometry;
}

function readAsciiColorComponent({
  parts,
  index,
}: {
  parts: string[];
  index: number;
}): number {
  if (index < 0) {
    return 180;
  }
  return Number(parts[index]);
}

function readBinaryColorComponent({
  view,
  base,
  offset,
}: {
  view: DataView;
  base: number;
  offset: PlyPropertyOffset | undefined;
}): number {
  if (offset === undefined) {
    return 180;
  }
  return readBinaryScalar({
    view,
    offset: base + offset.offset,
    type: offset.type,
  });
}

function normalizeColorComponent({ value }: { value: number }): number {
  if (!Number.isFinite(value)) {
    return 0.7;
  }
  if (value <= 1) {
    return Math.min(Math.max(value, 0), 1);
  }
  return Math.min(Math.max(value / 255, 0), 1);
}

function plyScalarTypeSize({ type }: { type: string }): number {
  const scalarTypeSizes: Record<string, number> = {
    char: 1,
    double: 8,
    float: 4,
    float32: 4,
    float64: 8,
    int: 4,
    int16: 2,
    int32: 4,
    int8: 1,
    short: 2,
    uchar: 1,
    uint: 4,
    uint16: 2,
    uint32: 4,
    uint8: 1,
    ushort: 2,
  };
  const size = scalarTypeSizes[type];
  if (size === undefined) {
    throw new Error(`unsupported PLY scalar type ${type}`);
  }
  return size;
}

function readBinaryScalar({
  view,
  offset,
  type,
}: {
  view: DataView;
  offset: number;
  type: string;
}): number {
  if (type === "char" || type === "int8") {
    return view.getInt8(offset);
  }
  if (type === "uchar" || type === "uint8") {
    return view.getUint8(offset);
  }
  if (type === "short" || type === "int16") {
    return view.getInt16(offset, true);
  }
  if (type === "ushort" || type === "uint16") {
    return view.getUint16(offset, true);
  }
  if (type === "int" || type === "int32") {
    return view.getInt32(offset, true);
  }
  if (type === "uint" || type === "uint32") {
    return view.getUint32(offset, true);
  }
  if (type === "float" || type === "float32") {
    return view.getFloat32(offset, true);
  }
  if (type === "double" || type === "float64") {
    return view.getFloat64(offset, true);
  }
  throw new Error(`unsupported PLY scalar type ${type}`);
}

function renderPointsScene({
  scene,
  camera,
  renderer,
  controls,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ThreeTrackballCameraControls;
}): void {
  startThreeSceneRenderLoop({ scene, camera, renderer, controls });
}

// Render an inline status card for a missing-resource or load-failure points display.
function _renderPointsStatus(message: string): HTMLElement {
  const status = document.createElement("div");
  status.className = "points-display-scene__status";
  status.style.display = "flex";
  status.style.alignItems = "center";
  status.style.justifyContent = "center";
  status.style.width = "100%";
  status.style.height = "100%";
  status.style.padding = "1rem";
  status.style.color = "#888";
  status.style.fontStyle = "italic";
  status.textContent = message;
  return status;
}

// Publish the current trackball camera state onto the container so peers sync.
function _publishCameraState({
  container,
  controls,
}: {
  container: HTMLDivElement;
  controls: ThreeTrackballCameraControls;
}): void {
  const cameraState = controls.getCameraState();
  if (cameraState === null) {
    return;
  }
  container.dataset.cameraState = JSON.stringify(cameraState);
  container.dispatchEvent(
    new CustomEvent<CameraState>("camera-pose-change", {
      bubbles: true,
      detail: cameraState,
    }),
  );
}

