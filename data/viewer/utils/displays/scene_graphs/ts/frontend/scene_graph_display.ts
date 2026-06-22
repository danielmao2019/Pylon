import * as THREE from "three";
import {
  createTrackballCameraControls,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import {
  createSpatialDisplayScene,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
import type { LeafVNode } from "web/reconcile/reconcile";
import type { SceneGraphDisplayResponse } from "./types/display_response";

// Heuristic default world-space size for node markers when the caller does not
// supply nodeSize. Lib-owned default, overridable.
export const DEFAULT_NODE_SIZE = 0.02;

// Neutral gray fallback color for edge lines, used when the payload does not
// carry an edge color AND the caller does not supply edgeColor. Lib-owned
// default, overridable.
export const DEFAULT_EDGE_COLOR = "#888888";

// Line-width fallback for edges when the caller does not supply edgeWidth.
// Lib-owned default, overridable.
export const DEFAULT_EDGE_WIDTH = 1.0;

// Font-size (px) fallback for overlay labels when the caller does not supply
// labelFontSize. Lib-owned default, overridable.
export const DEFAULT_LABEL_FONT_SIZE = 12;

// Text-color fallback for overlay labels when the caller does not supply
// labelColor. Lib-owned default, overridable.
export const DEFAULT_LABEL_COLOR = "#000000";

// Minimal payload shape consumed by the scene-graph display: node positions and
// optional per-node colors baked into a point cloud, edge endpoint positions
// and optional per-edge colors baked into a line set, and the per-node label
// entries projected onto the HTML overlay each frame.
interface SceneGraphLabelEntry {
  text: string;
  position: { x: number; y: number; z: number };
}

interface SceneGraphPayload {
  nodePositions: number[];
  nodeColors?: number[];
  edgePositions: number[];
  edgeColors?: number[];
  labels: SceneGraphLabelEntry[];
}

export function renderSceneGraphDisplay({
  displayResponse,
  initialCameraState = null,
  nodeSize,
  edgeColor,
  edgeWidth,
  labelFontSize,
  labelColor,
}: {
  displayResponse: SceneGraphDisplayResponse;
  initialCameraState?: CameraState | null;
  nodeSize?: number;
  edgeColor?: string;
  edgeWidth?: number;
  labelFontSize?: number;
  labelColor?: string;
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `scene_graph:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createSpatialDisplayScene({
        initialCameraState,
      });
      const { object, labels, labelOverlay } = createSceneGraphObject({
        container,
        displayResponse,
        nodeSize,
        edgeColor,
        edgeWidth,
        labelFontSize,
        labelColor,
      });
      scene.add(object);
      const controls = createTrackballCameraControls({ container, camera, renderer, initialCameraState });
      renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor });
      return container;
    },
  };
  return leaf;
}

// Part-B: builds the HTML label overlay and returns a THREE.Group + mutable labels
// array, both populated from the THREE.Points + label data once the async payload
// load resolves.
function createSceneGraphObject({
  container,
  displayResponse,
  nodeSize,
  edgeColor,
  edgeWidth,
  labelFontSize,
  labelColor,
}: {
  container: HTMLDivElement;
  displayResponse: SceneGraphDisplayResponse;
  nodeSize?: number;
  edgeColor?: string;
  edgeWidth?: number;
  labelFontSize?: number;
  labelColor?: string;
}): { object: THREE.Object3D; labels: object[]; labelOverlay: HTMLDivElement } {
  const labelOverlay = createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor });
  // Initially empty; mutated on async resolve so renderSceneGraphScene's per-frame
  // projection sees the populated list.
  const group = new THREE.Group();
  const labels: object[] = [];
  loadSceneGraphPayload({ displayResponse })
    .then((payload) => {
      const built = createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth });
      group.add(built.points);
      labels.push(...built.labels);
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      throw new Error(`unable to load scene graph: ${message}`);
    });
  return { object: group, labels, labelOverlay };
}

async function loadSceneGraphPayload({
  displayResponse,
}: {
  displayResponse: SceneGraphDisplayResponse;
}): Promise<SceneGraphPayload> {
  if (displayResponse.url === null) {
    throw new Error("scene graph display response url is null");
  }
  const response = await fetch(displayResponse.url);
  if (!response.ok) {
    throw new Error(`unable to load scene graph: HTTP ${response.status}`);
  }
  return (await response.json()) as SceneGraphPayload;
}

function createThreeSceneGraphPoints({
  payload,
  nodeSize,
  edgeColor,
  edgeWidth,
}: {
  payload: SceneGraphPayload;
  nodeSize?: number;
  edgeColor?: string;
  edgeWidth?: number;
}): { points: THREE.Points; labels: object[] } {
  // An explicit nodeSize pins the world-space marker size directly; otherwise
  // fall back to the lib default.
  const effectiveNodeSize = nodeSize ?? DEFAULT_NODE_SIZE;
  // An explicit edgeWidth pins the edge line width directly; otherwise fall
  // back to the lib default. The line set is colored below.
  const effectiveEdgeWidth = edgeWidth ?? DEFAULT_EDGE_WIDTH;
  // edgeColor (when supplied) replaces per-edge colors with a uniform color;
  // otherwise use the payload's per-edge colors, falling back to the lib
  // default uniform color when the payload carries none.
  let useEdgeVertexColors: boolean;
  let effectiveEdgeColor: string | undefined;
  if (edgeColor !== undefined) {
    useEdgeVertexColors = false;
    effectiveEdgeColor = edgeColor;
  } else if (payload.edgeColors !== undefined && payload.edgeColors.length > 0) {
    useEdgeVertexColors = true;
    effectiveEdgeColor = undefined;
  } else {
    useEdgeVertexColors = false;
    effectiveEdgeColor = DEFAULT_EDGE_COLOR;
  }

  const nodeGeometry = new THREE.BufferGeometry();
  nodeGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(payload.nodePositions), 3),
  );
  const useNodeVertexColors = payload.nodeColors !== undefined && payload.nodeColors.length > 0;
  if (useNodeVertexColors) {
    nodeGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(payload.nodeColors as number[]), 3),
    );
  }
  const nodeMaterial = new THREE.PointsMaterial({
    vertexColors: useNodeVertexColors,
    size: effectiveNodeSize,
  });
  const points = new THREE.Points(nodeGeometry, nodeMaterial);

  const edgeGeometry = new THREE.BufferGeometry();
  edgeGeometry.setAttribute(
    "position",
    new THREE.BufferAttribute(new Float32Array(payload.edgePositions), 3),
  );
  if (useEdgeVertexColors) {
    edgeGeometry.setAttribute(
      "color",
      new THREE.BufferAttribute(new Float32Array(payload.edgeColors as number[]), 3),
    );
  }
  const edgeMaterial = new THREE.LineBasicMaterial({
    vertexColors: useEdgeVertexColors,
    linewidth: effectiveEdgeWidth,
    ...(effectiveEdgeColor !== undefined ? { color: effectiveEdgeColor } : {}),
  });
  points.add(new THREE.LineSegments(edgeGeometry, edgeMaterial));

  // Each label entry pairs its overlay HTML node with the world position the
  // per-frame projection drives it from.
  const labels: object[] = payload.labels.map((entry) => {
    const node = document.createElement("div");
    node.style.position = "absolute";
    node.style.whiteSpace = "nowrap";
    node.textContent = entry.text;
    return {
      node,
      position: new THREE.Vector3(entry.position.x, entry.position.y, entry.position.z),
    };
  });
  return { points, labels };
}

function createThreeSceneGraphLabelOverlay({
  container,
  labelFontSize,
  labelColor,
}: {
  container: HTMLDivElement;
  labelFontSize?: number;
  labelColor?: string;
}): HTMLDivElement {
  const effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE;
  const effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR;
  const overlay = document.createElement("div");
  overlay.style.position = "absolute";
  overlay.style.inset = "0";
  overlay.style.width = "100%";
  overlay.style.height = "100%";
  overlay.style.overflow = "hidden";
  overlay.style.pointerEvents = "none";
  overlay.style.fontSize = `${effectiveLabelFontSize}px`;
  overlay.style.color = effectiveLabelColor;
  container.append(overlay);
  return overlay;
}

function renderSceneGraphScene({
  scene,
  camera,
  renderer,
  controls,
  labels,
  labelOverlay,
  labelFontSize,
  labelColor,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ThreeTrackballCameraControls;
  labels: object[];
  labelOverlay: HTMLDivElement;
  labelFontSize?: number;
  labelColor?: string;
}): void {
  startThreeSceneRenderLoop({
    scene,
    camera,
    renderer,
    controls,
    onAfterRender: () => _projectLabelsOntoOverlay({ camera, labels, labelOverlay, labelFontSize, labelColor }),
  });
}

function _projectLabelsOntoOverlay({
  camera,
  labels,
  labelOverlay,
  labelFontSize,
  labelColor,
}: {
  camera: THREE.PerspectiveCamera;
  labels: object[];
  labelOverlay: HTMLDivElement;
  labelFontSize?: number;
  labelColor?: string;
}): void {
  const effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE;
  const effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR;
  const width = Math.max(1, labelOverlay.clientWidth || 1);
  const height = Math.max(1, labelOverlay.clientHeight || 1);
  for (const label of labels) {
    const { node, position } = label as {
      node: HTMLDivElement;
      position: THREE.Vector3;
    };
    if (node.parentElement !== labelOverlay) {
      labelOverlay.append(node);
    }
    const projected = position.clone().project(camera);
    // Cull labels behind the camera or outside the normalized device cube.
    const offscreen =
      projected.z > 1 ||
      projected.x < -1 ||
      projected.x > 1 ||
      projected.y < -1 ||
      projected.y > 1;
    if (offscreen) {
      node.style.display = "none";
      continue;
    }
    const left = ((projected.x + 1) / 2) * width;
    const top = ((1 - projected.y) / 2) * height;
    node.style.display = "block";
    node.style.left = `${left}px`;
    node.style.top = `${top}px`;
    node.style.fontSize = `${effectiveLabelFontSize}px`;
    node.style.color = effectiveLabelColor;
  }
}
