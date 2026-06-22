import * as THREE from "three";
import type { LeafVNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { MeshDisplayResponse } from "./types/display_response";
import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
import {
  createSpatialDisplayScene,
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

// The logical (indexed) result of parsing a Wavefront OBJ: geometry vertices
// and faces, plus optional per-vertex colors and a per-face UV indexing layer
// (vertsUvs + facesUvs) that admits a `vt` count differing from the `v` count.
interface ParsedObj {
  verts: Float32Array;
  faces: Uint32Array;
  vertexColor: Float32Array | null;
  vertsUvs: Float32Array | null;
  facesUvs: Uint32Array | null;
  mtllibName: string | null;
}

// The wire payload of a sparse heatmap resource: a reference to the shared
// column geometry plus the part's non-zero (indices, values) delta.
interface SparseHeatmapResource {
  geometryUrl: string;
  indices: Int32Array;
  values: Float32Array;
}

// Render mirror of the data structure's MeshTextureVertexColor: per-vertex
// colors aligned 1:1 with verts. C is 3 for a dense opaque RGB payload, or 4
// for a sparse-heatmap overlay RGBA payload whose alpha-0 vertices must render
// transparent and reveal the base layer beneath.
interface MeshTextureVertexColor {
  kind: "vertex_color";
  vertexColor: Float32Array;
}

// Render mirror of the data structure's MeshTextureUVTextureMap: a
// per-face-indexed UV texture map.
interface MeshTextureUVTextureMap {
  kind: "uv_texture_map";
  uvTextureMap: THREE.Texture;
  vertsUvs: Float32Array;
  facesUvs: Uint32Array;
}

// The render-side mirror of the Mesh data structure: indexed geometry (verts,
// faces) plus an optional polymorphic MeshTexture. createThreeMesh consumes
// this synchronously, expanding it into the non-indexed corner render domain.
interface MeshPayload {
  verts: Float32Array;
  faces: Uint32Array;
  texture: MeshTextureVertexColor | MeshTextureUVTextureMap | null;
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
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `mesh:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createSpatialDisplayScene({
        initialCameraState,
      });
      const object = createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide });
      scene.add(object);
      const controls = createTrackballCameraControls({ container, camera, renderer, initialCameraState });
      renderMeshScene({ scene, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

// Part-B: returns a THREE.Group for the mesh, populated with the THREE.Mesh once
// the async payload load resolves.
export function createMeshObject({
  displayResponse,
  meshColor,
  meshOpacity,
  meshSide,
}: {
  displayResponse: MeshDisplayResponse;
  meshColor?: string;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): THREE.Object3D {
  const group = new THREE.Group();
  loadMeshPayload({ displayResponse })
    .then((payload) =>
      group.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })),
    )
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`unable to load mesh: ${message}`);
    });
  return group;
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
  const texture = await _resolveMeshTexture({ parsed, primaryUrl: displayResponse.url });
  return {
    verts: parsed.verts,
    faces: parsed.faces,
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
  const cornerCount = payload.faces.length;

  // geometry = non-indexed BufferGeometry whose position attribute gathers
  // payload.verts by payload.faces (corner domain); userData.cornerVertexIndices
  // = payload.faces is this geometry's corner→vertex map.
  const geometry = new THREE.BufferGeometry();
  const positions = new Float32Array(cornerCount * 3);
  for (let corner = 0; corner < cornerCount; corner++) {
    const vertexIndex = payload.faces[corner];
    positions[corner * 3] = payload.verts[vertexIndex * 3];
    positions[corner * 3 + 1] = payload.verts[vertexIndex * 3 + 1];
    positions[corner * 3 + 2] = payload.verts[vertexIndex * 3 + 2];
  }
  geometry.setAttribute("position", new THREE.BufferAttribute(positions, 3));
  geometry.userData.cornerVertexIndices = payload.faces;

  const effectiveOpacity = meshOpacity ?? DEFAULT_MESH_OPACITY;
  const effectiveSide = meshSide ?? DEFAULT_MESH_SIDE;
  let useTexture: boolean;
  let useVertexColors: boolean;
  let rgbaVertexColors: boolean;
  let textureMap: THREE.Texture | undefined;
  let effectiveColor: string | undefined;
  if (meshColor !== undefined) {
    useTexture = false;
    useVertexColors = false;
    rgbaVertexColors = false;
    textureMap = undefined;
    effectiveColor = meshColor;
  } else if (payload.texture !== null && payload.texture.kind === "uv_texture_map") {
    // Add a uv attribute gathering payload.texture.vertsUvs by payload.texture.facesUvs.
    const uvs = new Float32Array(cornerCount * 2);
    for (let corner = 0; corner < cornerCount; corner++) {
      const uvIndex = payload.texture.facesUvs[corner];
      uvs[corner * 2] = payload.texture.vertsUvs[uvIndex * 2];
      uvs[corner * 2 + 1] = payload.texture.vertsUvs[uvIndex * 2 + 1];
    }
    geometry.setAttribute("uv", new THREE.BufferAttribute(uvs, 2));
    useTexture = true;
    useVertexColors = false;
    rgbaVertexColors = false;
    textureMap = payload.texture.uvTextureMap;
    effectiveColor = undefined;
  } else if (payload.texture !== null && payload.texture.kind === "vertex_color") {
    // Add a color attribute gathering payload.texture.vertexColor by payload.faces.
    const vertexCount = payload.verts.length / 3;
    const components = payload.texture.vertexColor.length / Math.max(1, vertexCount);
    const colors = new Float32Array(cornerCount * components);
    for (let corner = 0; corner < cornerCount; corner++) {
      const vertexIndex = payload.faces[corner];
      for (let component = 0; component < components; component++) {
        colors[corner * components + component] =
          payload.texture.vertexColor[vertexIndex * components + component];
      }
    }
    geometry.setAttribute("color", new THREE.BufferAttribute(colors, components));
    useTexture = false;
    useVertexColors = true;
    rgbaVertexColors = components === 4;
    textureMap = undefined;
    effectiveColor = undefined;
  } else {
    useTexture = false;
    useVertexColors = false;
    rgbaVertexColors = false;
    textureMap = undefined;
    effectiveColor = DEFAULT_MESH_COLOR;
  }

  geometry.computeVertexNormals();
  geometry.computeBoundingBox();
  geometry.computeBoundingSphere();

  const material = new THREE.MeshBasicMaterial({
    vertexColors: useVertexColors,
    side: effectiveSide,
    opacity: effectiveOpacity,
    transparent: effectiveOpacity < 1 || (useVertexColors && rgbaVertexColors),
    ...(useTexture ? { map: textureMap } : {}),
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
// geometry into a per-vertex RGBA MeshTextureVertexColor overlay: vertices in
// `indices` carry that vertex's scalar→rgb heatmap color at alpha 1; every
// other vertex carries alpha 0, so outside the delta the overlay is fully
// transparent and the textured base layer beneath shows through.
function _resolveSparseHeatmapPayload({
  parsed,
  sparse,
}: {
  parsed: ParsedObj;
  sparse: SparseHeatmapResource;
}): MeshPayload {
  const vertexCount = parsed.verts.length / 3;
  const vertexColor = new Float32Array(vertexCount * 4);
  const rgb = _mapScalarsToRgb(sparse.values);
  for (let i = 0; i < sparse.indices.length; i++) {
    const vertexIndex = sparse.indices[i];
    if (vertexIndex < 0 || vertexIndex >= vertexCount) {
      throw new Error(
        `sparse heatmap references out-of-range vertex index: ${vertexIndex} (vertexCount=${vertexCount})`,
      );
    }
    vertexColor[vertexIndex * 4] = rgb[i * 3] / 255;
    vertexColor[vertexIndex * 4 + 1] = rgb[i * 3 + 1] / 255;
    vertexColor[vertexIndex * 4 + 2] = rgb[i * 3 + 2] / 255;
    vertexColor[vertexIndex * 4 + 3] = 1.0;
  }
  return {
    verts: parsed.verts,
    faces: parsed.faces,
    texture: { kind: "vertex_color", vertexColor },
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
// polygon-fan `f` lines whose corners are `v`, `v/vt`, `v//vn`, or `v/vt/vn`,
// into logical (indexed) geometry: verts + triangulated faces, plus optional
// per-vertex colors and a per-face UV indexing layer (vertsUvs + facesUvs).
function _parseObj(text: string): ParsedObj {
  const vPositions: number[] = [];
  const vColors: number[] = [];
  const vtCoords: number[] = [];
  let sawVertexColors = false;
  let mtllibName: string | null = null;
  const faceVertexTokens: number[] = [];
  const faceUvTokens: number[] = [];
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
          faceVertexTokens.push(corner.v);
          faceUvTokens.push(corner.vt);
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
  const cornerCount = faceVertexTokens.length;
  const useUvs = sawAnyUv && vtCoords.length > 0;
  const verts = new Float32Array(vPositions);
  const faces = new Uint32Array(cornerCount);
  const vertexColor = sawVertexColors ? new Float32Array(vColors) : null;
  const facesUvs = useUvs ? new Uint32Array(cornerCount) : null;
  const vertsUvs = useUvs ? new Float32Array(vtCoords) : null;

  for (let corner = 0; corner < cornerCount; corner++) {
    const vIndex = faceVertexTokens[corner];
    if (vIndex < 0 || vIndex >= geometryVertexCount) {
      throw new Error(`OBJ face references out-of-range vertex index: ${vIndex}`);
    }
    faces[corner] = vIndex;
    if (facesUvs !== null) {
      const vtIndex = faceUvTokens[corner];
      if (vtIndex < 0 || vtIndex * 2 + 1 >= vtCoords.length) {
        throw new Error(`OBJ face corner is missing a valid UV index: ${vtIndex}`);
      }
      facesUvs[corner] = vtIndex;
    }
  }
  return { verts, faces, vertexColor, vertsUvs, facesUvs, mtllibName };
}

// Parse one OBJ face token (`v`, `v/vt`, `v//vn`, or `v/vt/vn`) into 0-based indices.
function _parseFaceCorner(token: string): { v: number; vt: number } {
  const fields = token.split("/");
  const v = parseInt(fields[0], 10) - 1;
  const vt = fields.length >= 2 && fields[1].length > 0 ? parseInt(fields[1], 10) - 1 : -1;
  return { v, vt };
}

// Resolve the parsed OBJ into a MeshTexture: a MeshTextureUVTextureMap when the
// OBJ declares UVs and an `mtllib` (loading the MTL `map_Kd` image; a UV-textured
// mesh with no `map_Kd` is a hard error), else a MeshTextureVertexColor when the
// OBJ carries per-vertex colors, else null.
async function _resolveMeshTexture({
  parsed,
  primaryUrl,
}: {
  parsed: ParsedObj;
  primaryUrl: string;
}): Promise<MeshTextureVertexColor | MeshTextureUVTextureMap | null> {
  if (parsed.vertsUvs !== null && parsed.facesUvs !== null && parsed.mtllibName !== null) {
    const textureName = await _fetchMtlTextureName(_siblingUrl(primaryUrl, parsed.mtllibName));
    if (textureName === null) {
      throw new Error(`mesh OBJ declares UVs but its MTL has no map_Kd: ${primaryUrl}`);
    }
    const uvTextureMap = await _fetchTexture(_siblingUrl(primaryUrl, textureName));
    return {
      kind: "uv_texture_map",
      uvTextureMap,
      vertsUvs: parsed.vertsUvs,
      facesUvs: parsed.facesUvs,
    };
  }
  if (parsed.vertexColor !== null) {
    return { kind: "vertex_color", vertexColor: parsed.vertexColor };
  }
  return null;
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
