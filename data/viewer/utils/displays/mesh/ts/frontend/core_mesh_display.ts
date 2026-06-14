import * as THREE from "three";
import type { LeafVNode, VNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { MeshDisplayResponse } from "./types/display_response";
import {
  createTrackballCameraControls,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
import {
  createThreeDisplayContainer,
  createThreePerspectiveCamera,
  createThreeScene,
  createThreeWebGLRenderer,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";

// Uniform fallback color used when geometry has no texture AND has no vertex
// colors AND the caller does not supply meshColor; lib-owned default, overridable.
const DEFAULT_MESH_COLOR = "#cccccc";

// Opaque default applied when the caller does not supply meshOpacity; the
// material's `transparent` flag flips true automatically when opacity is less
// than 1; lib-owned default, overridable.
const DEFAULT_MESH_OPACITY = 1.0;

// Fallback side mode for visibility under arbitrary camera framings when the
// caller does not supply meshSide; lib-owned default, overridable.
const DEFAULT_MESH_SIDE: THREE.Side = THREE.DoubleSide;

const NEUTRAL_GRAY = 0.75;

const _objCache = new Map<string, Promise<ParsedObj>>();
const _textureCache = new Map<string, Promise<THREE.Texture>>();

// One distinct UV/positions corner of a parsed Wavefront OBJ, expanded so
// each face corner is its own render vertex; this admits per-corner UV
// indexing where the `vt` count differs from the `v` (geometry) count.
interface ParsedObj {
  positions: Float32Array;
  cornerVertexIndices: Uint32Array;
  geometryVertexCount: number;
  vertexColors: Float32Array | null;
  uvs: Float32Array | null;
  mtllibName: string | null;
}

// The wire payload of a sparse heatmap resource: a reference to the shared
// column geometry plus the part's non-zero (indices, values) delta.
interface SparseHeatmapResource {
  geometryUrl: string;
  indices: Int32Array;
  values: Float32Array;
}

// A pre-loaded mesh payload: per-corner-expanded geometry buffers plus the
// resolved texture representation (a UV texture map, per-vertex colors, or
// neither). createThreeMesh consumes this synchronously.
//
// vertexColorComponents is the itemSize of the vertexColors buffer: 3 for a
// dense opaque RGB payload, 4 for a sparse-heatmap overlay RGBA payload whose
// alpha-0 vertices must render transparent and reveal the base layer beneath.
interface MeshPayload {
  positions: Float32Array;
  uvs: Float32Array | null;
  vertexColors: Float32Array | null;
  vertexColorComponents: number;
  texture: THREE.Texture | null;
}

// Render a self-contained mesh display element initialized at initialCameraState.
export function renderMeshDisplay({
  displayResponse,
  initialCameraState = null,
  meshColor,
  meshOpacity,
  meshSide,
}: {
  displayResponse: MeshDisplayResponse;
  initialCameraState?: CameraState | null;
  meshColor?: string;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): VNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `mesh:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createMeshScene({
        displayResponse,
        initialCameraState,
        meshColor,
        meshOpacity,
        meshSide,
      });
      const controls = createTrackballCameraControls({ container, camera, renderer, initialCameraState });
      _registerSceneResize({ container, camera, renderer, controls });
      _publishCameraState({ container, controls });
      controls.addEventListener("change", () => {
        _publishCameraState({ container, controls });
      });
      renderMeshScene({ scene, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

// Compose container, scene, camera, renderer; the mesh payload is loaded
// asynchronously and the THREE.Mesh joins the scene on resolve.
export function createMeshScene({
  displayResponse,
  initialCameraState,
  meshColor,
  meshOpacity,
  meshSide,
}: {
  displayResponse: MeshDisplayResponse;
  initialCameraState: CameraState | null;
  meshColor?: string;
  meshOpacity?: number;
  meshSide?: THREE.Side;
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
  loadMeshPayload({ displayResponse })
    .then((payload) =>
      scene.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })),
    )
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      container.replaceChildren(_renderMeshStatus(`Failed to load mesh: ${message}`));
    });
  return { container, scene, camera, renderer };
}

// Async-load the mesh payload from displayResponse.url; resolve a sparse-heatmap
// delta against its referenced geometry, otherwise read the dense resource as-is.
export async function loadMeshPayload({
  displayResponse,
}: {
  displayResponse: MeshDisplayResponse;
}): Promise<MeshPayload> {
  if (displayResponse.url === null) {
    throw new Error("mesh display response url is null");
  }
  if (displayResponse.display_kind === "sparse_heatmap_mesh") {
    const sparse = await _fetchSparseHeatmapResource(displayResponse.url);
    const parsed = await _fetchObj(sparse.geometryUrl);
    return _resolveSparseHeatmapPayload({ parsed, sparse });
  }
  const parsed = await _fetchObj(displayResponse.url);
  const texture = await _resolveTexture({ parsed, primaryUrl: displayResponse.url });
  return {
    positions: parsed.positions,
    uvs: parsed.uvs,
    vertexColors: parsed.vertexColors,
    vertexColorComponents: 3,
    texture,
  };
}

// Sync-build THREE.BufferGeometry + THREE.MeshBasicMaterial + THREE.Mesh from a
// pre-loaded payload.
export function createThreeMesh({
  payload,
  meshColor,
  meshOpacity,
  meshSide,
}: {
  payload: MeshPayload;
  displayResponse: MeshDisplayResponse;
  meshColor?: string;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): THREE.Mesh {
  const geometry = _buildBaseGeometry(payload);
  const effectiveOpacity = meshOpacity ?? DEFAULT_MESH_OPACITY;
  const effectiveSide = meshSide ?? DEFAULT_MESH_SIDE;
  let useTexture: boolean;
  let useVertexColors: boolean;
  let effectiveColor: string | undefined;
  if (meshColor !== undefined) {
    useTexture = false;
    useVertexColors = false;
    effectiveColor = meshColor;
  } else if (payload.texture !== null) {
    useTexture = true;
    useVertexColors = false;
    effectiveColor = undefined;
  } else if (payload.vertexColors !== null) {
    useTexture = false;
    useVertexColors = true;
    effectiveColor = undefined;
  } else {
    useTexture = false;
    useVertexColors = false;
    effectiveColor = DEFAULT_MESH_COLOR;
  }
  const material = new THREE.MeshBasicMaterial({
    vertexColors: useVertexColors,
    side: effectiveSide,
    opacity: effectiveOpacity,
    transparent: effectiveOpacity < 1 || (useVertexColors && payload.vertexColorComponents === 4),
    ...(useTexture ? { map: payload.texture } : {}),
    ...(effectiveColor !== undefined ? { color: effectiveColor } : {}),
  });
  return new THREE.Mesh(geometry, material);
}

// Mount the render loop for the composed mesh scene.
export function renderMeshScene({
  scene,
  camera,
  renderer,
  controls,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ReturnType<typeof createTrackballCameraControls>;
}): void {
  startThreeSceneRenderLoop({ scene, camera, renderer, controls });
}

// Resolve a sparse heatmap (indices, values) delta against its referenced
// geometry into a per-corner RGBA overlay payload: corners whose geometry
// vertex is in `indices` carry that vertex's scalar→rgb heatmap color at alpha
// 1; every other corner carries alpha 0, so outside the delta the overlay is
// fully transparent and the textured base layer beneath shows through.
function _resolveSparseHeatmapPayload({
  parsed,
  sparse,
}: {
  parsed: ParsedObj;
  sparse: SparseHeatmapResource;
}): MeshPayload {
  const cornerCount = parsed.cornerVertexIndices.length;
  const colors = new Float32Array(cornerCount * 4);
  const rgb = _mapScalarsToRgb(sparse.values);
  const rgbByGeometryVertex = new Map<number, [number, number, number]>();
  for (let i = 0; i < sparse.indices.length; i++) {
    rgbByGeometryVertex.set(sparse.indices[i], [
      rgb[i * 3] / 255,
      rgb[i * 3 + 1] / 255,
      rgb[i * 3 + 2] / 255,
    ]);
  }
  for (let corner = 0; corner < cornerCount; corner++) {
    const vertexIndex = parsed.cornerVertexIndices[corner];
    const color = rgbByGeometryVertex.get(vertexIndex);
    if (color === undefined) {
      continue;
    }
    colors[corner * 4] = color[0];
    colors[corner * 4 + 1] = color[1];
    colors[corner * 4 + 2] = color[2];
    colors[corner * 4 + 3] = 1.0;
  }
  return {
    positions: parsed.positions,
    uvs: null,
    vertexColors: colors,
    vertexColorComponents: 4,
    texture: null,
  };
}

// Fetch and parse a Wavefront OBJ url once; subsequent callers share the promise.
function _fetchObj(url: string): Promise<ParsedObj> {
  const cached = _objCache.get(url);
  if (cached !== undefined) {
    return cached;
  }
  const promise = fetch(url).then(async (response) => {
    if (!response.ok) {
      throw new Error(`GET ${url} failed: ${response.status}`);
    }
    return _parseObj(await response.text());
  });
  _objCache.set(url, promise);
  return promise;
}

// Parse a Wavefront OBJ supporting `v x y z [r g b]`, `vt u v`, `mtllib`, and
// polygon-fan `f` lines whose corners are `v`, `v/vt`, `v//vn`, or `v/vt/vn`.
function _parseObj(text: string): ParsedObj {
  const vPositions: number[] = [];
  const vColors: number[] = [];
  const vtCoords: number[] = [];
  let sawVertexColors = false;
  let mtllibName: string | null = null;
  const cornerVertexTokens: number[] = [];
  const cornerUvTokens: number[] = [];
  let sawAnyUv = false;

  for (const raw of text.split("\n")) {
    if (raw.length === 0) {
      continue;
    }
    const c0 = raw.charCodeAt(0);
    if (c0 === 118 /* 'v' */ && raw.charCodeAt(1) === 32 /* ' ' */) {
      const parts = raw.trim().split(/\s+/);
      vPositions.push(parseFloat(parts[1]), parseFloat(parts[2]), parseFloat(parts[3]));
      if (parts.length >= 7) {
        sawVertexColors = true;
        vColors.push(parseFloat(parts[4]), parseFloat(parts[5]), parseFloat(parts[6]));
      } else {
        vColors.push(NEUTRAL_GRAY, NEUTRAL_GRAY, NEUTRAL_GRAY);
      }
    } else if (c0 === 118 /* 'v' */ && raw.charCodeAt(1) === 116 /* 't' */) {
      const parts = raw.trim().split(/\s+/);
      vtCoords.push(parseFloat(parts[1]), parseFloat(parts[2]));
    } else if (c0 === 102 /* 'f' */ && raw.charCodeAt(1) === 32 /* ' ' */) {
      const parts = raw.trim().split(/\s+/);
      const corners: Array<{ v: number; vt: number }> = [];
      for (let p = 1; p < parts.length; p++) {
        corners.push(_parseFaceCorner(parts[p]));
      }
      for (let j = 1; j < corners.length - 1; j++) {
        const fan: Array<{ v: number; vt: number }> = [corners[0], corners[j], corners[j + 1]];
        for (const corner of fan) {
          cornerVertexTokens.push(corner.v);
          cornerUvTokens.push(corner.vt);
          if (corner.vt >= 0) {
            sawAnyUv = true;
          }
        }
      }
    } else if (raw.startsWith("mtllib")) {
      const parts = raw.trim().split(/\s+/);
      if (parts.length >= 2) {
        mtllibName = parts.slice(1).join(" ");
      }
    }
  }

  const geometryVertexCount = vPositions.length / 3;
  const cornerCount = cornerVertexTokens.length;
  const positions = new Float32Array(cornerCount * 3);
  const cornerVertexIndices = new Uint32Array(cornerCount);
  const useUvs = sawAnyUv && vtCoords.length > 0;
  const vertexColors = sawVertexColors ? new Float32Array(cornerCount * 3) : null;
  const uvs = useUvs ? new Float32Array(cornerCount * 2) : null;

  for (let corner = 0; corner < cornerCount; corner++) {
    const vIndex = cornerVertexTokens[corner];
    if (vIndex < 0 || vIndex >= geometryVertexCount) {
      throw new Error(`OBJ face references out-of-range vertex index: ${vIndex}`);
    }
    cornerVertexIndices[corner] = vIndex;
    positions[corner * 3] = vPositions[vIndex * 3];
    positions[corner * 3 + 1] = vPositions[vIndex * 3 + 1];
    positions[corner * 3 + 2] = vPositions[vIndex * 3 + 2];
    if (vertexColors !== null) {
      vertexColors[corner * 3] = vColors[vIndex * 3];
      vertexColors[corner * 3 + 1] = vColors[vIndex * 3 + 1];
      vertexColors[corner * 3 + 2] = vColors[vIndex * 3 + 2];
    }
    if (uvs !== null) {
      const vtIndex = cornerUvTokens[corner];
      if (vtIndex < 0 || vtIndex * 2 + 1 >= vtCoords.length) {
        throw new Error(`OBJ face corner is missing a valid UV index: ${vtIndex}`);
      }
      uvs[corner * 2] = vtCoords[vtIndex * 2];
      uvs[corner * 2 + 1] = vtCoords[vtIndex * 2 + 1];
    }
  }
  return { positions, cornerVertexIndices, geometryVertexCount, vertexColors, uvs, mtllibName };
}

// Parse one OBJ face token (`v`, `v/vt`, `v//vn`, or `v/vt/vn`) into 0-based indices.
function _parseFaceCorner(token: string): { v: number; vt: number } {
  const fields = token.split("/");
  const v = parseInt(fields[0], 10) - 1;
  const vt = fields.length >= 2 && fields[1].length > 0 ? parseInt(fields[1], 10) - 1 : -1;
  return { v, vt };
}

// Resolve the OBJ's texture: load the MTL `map_Kd` image when the OBJ declares
// UVs and an `mtllib`; a UV-textured mesh with no `map_Kd` is a hard error.
async function _resolveTexture({
  parsed,
  primaryUrl,
}: {
  parsed: ParsedObj;
  primaryUrl: string;
}): Promise<THREE.Texture | null> {
  if (parsed.uvs === null || parsed.mtllibName === null) {
    return null;
  }
  const textureName = await _fetchMtlTextureName(_siblingUrl(primaryUrl, parsed.mtllibName));
  if (textureName === null) {
    throw new Error(`mesh OBJ declares UVs but its MTL has no map_Kd: ${primaryUrl}`);
  }
  return _fetchTexture(_siblingUrl(primaryUrl, textureName));
}

// Fetch a `.mtl` sibling and parse its `map_Kd` texture-image filename.
async function _fetchMtlTextureName(mtlUrl: string): Promise<string | null> {
  const response = await fetch(mtlUrl);
  if (!response.ok) {
    throw new Error(`GET ${mtlUrl} failed: ${response.status}`);
  }
  const text = await response.text();
  for (const raw of text.split("\n")) {
    const line = raw.trim();
    if (line.startsWith("map_Kd")) {
      const parts = line.split(/\s+/);
      if (parts.length >= 2) {
        return parts.slice(1).join(" ");
      }
    }
  }
  return null;
}

// Load a texture image url once into a THREE.Texture; subsequent callers share the promise.
function _fetchTexture(textureUrl: string): Promise<THREE.Texture> {
  const cached = _textureCache.get(textureUrl);
  if (cached !== undefined) {
    return cached;
  }
  const loader = new THREE.TextureLoader();
  const promise = new Promise<THREE.Texture>((resolve, reject) => {
    loader.load(
      textureUrl,
      (texture: THREE.Texture) => {
        texture.colorSpace = THREE.SRGBColorSpace;
        texture.flipY = true;
        texture.needsUpdate = true;
        resolve(texture);
      },
      undefined,
      () => reject(new Error(`unable to load texture image: ${textureUrl}`)),
    );
  });
  _textureCache.set(textureUrl, promise);
  return promise;
}

// Resolve a sibling resource url (mtl / texture image) relative to the primary url.
function _siblingUrl(primaryUrl: string, siblingName: string): string {
  const slash = primaryUrl.lastIndexOf("/");
  if (slash < 0) {
    return siblingName;
  }
  return primaryUrl.slice(0, slash + 1) + siblingName;
}

// Fetch and decode a sparse heatmap wire resource: geometry reference + delta.
async function _fetchSparseHeatmapResource(url: string): Promise<SparseHeatmapResource> {
  const response = await fetch(url);
  if (!response.ok) {
    throw new Error(`GET ${url} failed: ${response.status}`);
  }
  const raw = (await response.json()) as {
    geometry_url?: unknown;
    indices?: unknown;
    values?: unknown;
  };
  if (typeof raw.geometry_url !== "string" || raw.geometry_url.length === 0) {
    throw new Error(`sparse heatmap resource is missing geometry_url: ${url}`);
  }
  if (!Array.isArray(raw.indices) || !Array.isArray(raw.values)) {
    throw new Error(`sparse heatmap resource is missing indices/values arrays: ${url}`);
  }
  return {
    geometryUrl: raw.geometry_url,
    indices: Int32Array.from(raw.indices as number[]),
    values: Float32Array.from(raw.values as number[]),
  };
}

const _HEATMAP_PALETTE_STOPS: ReadonlyArray<number> = [0.0, 0.25, 0.5, 0.75, 1.0];
const _HEATMAP_PALETTE_COLORS: ReadonlyArray<readonly [number, number, number]> = [
  [0, 0, 255],
  [0, 255, 255],
  [0, 255, 0],
  [255, 255, 0],
  [255, 0, 0],
];

// Map non-negative scalars to RGB through a fixed continuous heatmap palette.
function _mapScalarsToRgb(values: Float32Array): Uint8Array {
  let maxValue = 0.0;
  for (let i = 0; i < values.length; i++) {
    if (values[i] > maxValue) {
      maxValue = values[i];
    }
  }
  const denom = Math.max(maxValue, 1e-12);
  const rgb = new Uint8Array(values.length * 3);
  for (let i = 0; i < values.length; i++) {
    const normalized = Math.min(1.0, Math.max(0.0, values[i] / denom));
    let segment = 0;
    while (
      segment < _HEATMAP_PALETTE_STOPS.length - 2
      && normalized >= _HEATMAP_PALETTE_STOPS[segment + 1]
    ) {
      segment += 1;
    }
    const left = _HEATMAP_PALETTE_STOPS[segment];
    const right = _HEATMAP_PALETTE_STOPS[segment + 1];
    const extent = Math.max(right - left, 1e-12);
    const fraction = Math.min(1.0, Math.max(0.0, (normalized - left) / extent));
    const c0 = _HEATMAP_PALETTE_COLORS[segment];
    const c1 = _HEATMAP_PALETTE_COLORS[segment + 1];
    rgb[i * 3] = Math.round(c0[0] + (c1[0] - c0[0]) * fraction);
    rgb[i * 3 + 1] = Math.round(c0[1] + (c1[1] - c0[1]) * fraction);
    rgb[i * 3 + 2] = Math.round(c0[2] + (c1[2] - c0[2]) * fraction);
  }
  return rgb;
}

// Build the per-corner-expanded base geometry: positions, optional UVs and colors.
function _buildBaseGeometry(payload: MeshPayload): THREE.BufferGeometry {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute("position", new THREE.BufferAttribute(payload.positions, 3));
  if (payload.uvs !== null) {
    geometry.setAttribute("uv", new THREE.BufferAttribute(payload.uvs, 2));
  }
  if (payload.vertexColors !== null) {
    geometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(payload.vertexColors), payload.vertexColorComponents),
    );
  }
  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();
  return geometry;
}

// Keep the renderer and camera aspect synced to the container size.
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
  const resize = (): void => {
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

// Render an inline status card for a missing-resource or load-failure mesh display.
function _renderMeshStatus(message: string): HTMLElement {
  const status = document.createElement("div");
  status.className = "mesh-display-scene__status";
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
