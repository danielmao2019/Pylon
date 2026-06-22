import * as THREE from "three";
import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
import {
  createSpatialDisplayScene,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
import {
  registerSpatialLayerRenderer,
  type SpatialLayerRenderer,
} from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
import type { LeafVNode } from "web/reconcile/reconcile";
import type { Aabb3dDisplayResponse } from "./types/display_response";

// Wireframe color and screen-space linewidth for the 3D axis-aligned boxes.
// WebGL ignores LineBasicMaterial.linewidth, so the boxes are drawn as
// LineSegments2 with a LineMaterial whose linewidth is in screen pixels.
const AABB_3D_BOX_COLOR = 0x4da6ff;
const AABB_3D_BOX_LINEWIDTH = 2;

// Score-label sprite sizing: the label world height is a fraction of the boxes'
// bounding-sphere radius so labels track scene scale under zoom; the 4:1 ratio
// matches the 256x64 label canvas aspect.
const AABB_3D_LABEL_HEIGHT_RATIO = 0.04;
const AABB_3D_LABEL_ASPECT = 4;

// Renders a self-contained 3D-box display initialized at initialCameraState.
//
// Args:
//   displayResponse: the 3D box overlay response carrying the inline boxes (each
//     [min_x, min_y, min_z, max_x, max_y, max_z], world coordinates) and the
//     optional per-box scores.
//   initialCameraState: initial framing for the camera (camera-to-world extrinsics
//     + intrinsics); null uses the camera's default framing.
//
// Returns:
//   A LeafVNode whose render() mounts the spatial box overlay.
export function renderAabb3dDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: Aabb3dDisplayResponse;
  initialCameraState?: CameraState | null;
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `aabb_3d:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createSpatialDisplayScene({
        initialCameraState,
      });
      const object = createAabb3dObject({ displayResponse });
      scene.add(object);
      const controls = createTrackballCameraControls({
        container,
        camera,
        renderer,
        initialCameraState,
      });
      renderAabb3dScene({ scene, camera, renderer, controls });
      return container;
    },
  };
  return leaf;
}

// Part-B: builds the inline 3D axis-aligned boxes and optional per-box score
// labels into a THREE.Group and returns it for the layered container to add. Each
// box is a 6-float [min_x, min_y, min_z, max_x, max_y, max_z] drawn as a 12-edge
// LineSegments2 wireframe, with its score (when present) rendered as a label sprite
// at the box top, sized to the boxes' shared bounding radius.
//
// Args:
//   displayResponse: the 3D box overlay response carrying the boxes and optional
//     per-box scores.
//
// Returns:
//   The THREE.Group holding every box wireframe and score label.
export function createAabb3dObject({
  displayResponse,
}: {
  displayResponse: Aabb3dDisplayResponse;
}): THREE.Object3D {
  const boxes = displayResponse.aabbs;
  const scores = displayResponse.scores;
  const boundingRadius = _boxesBoundingRadius({ boxes });
  const group = new THREE.Group();
  for (let boxIndex = 0; boxIndex < boxes.length; boxIndex += 1) {
    const box = boxes[boxIndex];
    const [minX, minY, minZ, maxX, maxY, maxZ] = box;
    const boxGroup = new THREE.Group();
    boxGroup.add(_createBoxLines({ box }));
    if (scores !== null) {
      const sprite = _createScoreLabelSprite({ score: scores[boxIndex] });
      sprite.position.set((minX + maxX) / 2, maxY, (minZ + maxZ) / 2);
      const height = boundingRadius * AABB_3D_LABEL_HEIGHT_RATIO;
      sprite.scale.set(height * AABB_3D_LABEL_ASPECT, height, 1);
      sprite.renderOrder = 1001;
      boxGroup.add(sprite);
    }
    group.add(boxGroup);
  }
  return group;
}

// Drives the 3D-box display render loop with the supplied trackball controls.
//
// Args:
//   scene: the scene the box group was added into.
//   camera: the display camera.
//   renderer: the display renderer.
//   controls: the camera's trackball controls, updated each frame.
//
// Returns:
//   void.
function renderAabb3dScene({
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

// Build one box's wireframe as a width-capable LineSegments2: its 12 edges are
// packed as start/end vertex pairs into a LineSegmentsGeometry and drawn with a
// LineMaterial, since WebGL ignores LineBasicMaterial.linewidth. depthTest is off
// so the box overlays the underlying point cloud, and the material resolution is
// seeded to the current viewport (LineSegments2's onBeforeRender keeps it synced
// each frame thereafter).
//
// Args:
//   box: the box's 6-float [min_x, min_y, min_z, max_x, max_y, max_z] world AABB,
//     drawn as 12 edges.
//
// Returns:
//   The LineSegments2 holding this box's 12 AABB edges.
function _createBoxLines({ box }: { box: number[] }): LineSegments2 {
  const [minX, minY, minZ, maxX, maxY, maxZ] = box;
  // The eight AABB corners, indexed 0..7 as the bit pattern (x, y, z) low->high.
  const corners: [number, number, number][] = [
    [minX, minY, minZ],
    [maxX, minY, minZ],
    [minX, maxY, minZ],
    [maxX, maxY, minZ],
    [minX, minY, maxZ],
    [maxX, minY, maxZ],
    [minX, maxY, maxZ],
    [maxX, maxY, maxZ],
  ];
  // The 12 edges as corner-index pairs: 4 along x, 4 along y, 4 along z.
  const edges: [number, number][] = [
    [0, 1],
    [2, 3],
    [4, 5],
    [6, 7],
    [0, 2],
    [1, 3],
    [4, 6],
    [5, 7],
    [0, 4],
    [1, 5],
    [2, 6],
    [3, 7],
  ];
  const positions: number[] = [];
  for (const [a, b] of edges) {
    positions.push(...corners[a], ...corners[b]);
  }
  const geometry = new LineSegmentsGeometry();
  geometry.setPositions(positions);
  const material = new LineMaterial({
    color: AABB_3D_BOX_COLOR,
    linewidth: AABB_3D_BOX_LINEWIDTH,
    depthTest: false,
    transparent: true,
    resolution: new THREE.Vector2(
      typeof window === "undefined" ? 1 : window.innerWidth,
      typeof window === "undefined" ? 1 : window.innerHeight,
    ),
  });
  const boxLines = new LineSegments2(geometry, material);
  boxLines.renderOrder = 1000;
  return boxLines;
}

// Create a score-label sprite rendering one box's score onto a canvas texture.
//
// Args:
//   score: the per-box score the label renders.
//
// Returns:
//   A THREE.Sprite carrying the rendered score texture.
function _createScoreLabelSprite({ score }: { score: number }): THREE.Sprite {
  const canvas = document.createElement("canvas");
  canvas.width = 256;
  canvas.height = 64;
  const context = canvas.getContext("2d");
  if (context === null) {
    throw new Error("aabb 3d score label canvas 2d context is unavailable");
  }
  context.fillStyle = "rgba(77,166,255,0.85)";
  context.fillRect(0, 0, canvas.width, canvas.height);
  context.fillStyle = "#ffffff";
  context.font = "bold 36px monospace";
  context.textBaseline = "middle";
  context.fillText(score.toFixed(2), 8, canvas.height / 2);
  const texture = new THREE.CanvasTexture(canvas);
  const spriteMaterial = new THREE.SpriteMaterial({
    map: texture,
    depthTest: false,
    transparent: true,
  });
  return new THREE.Sprite(spriteMaterial);
}

// Compute the shared bounding-sphere radius of every box's corners so score
// labels are sized to the overlay's extent and track it under zoom. Falls back to
// 1 when there are no boxes.
//
// Args:
//   boxes: the inline 3D boxes, each a 6-float [min_x, min_y, min_z, max_x, max_y,
//     max_z] world AABB.
//
// Returns:
//   The bounding-sphere radius of all boxes' corners, or 1 when empty.
function _boxesBoundingRadius({ boxes }: { boxes: number[][] }): number {
  if (boxes.length === 0) {
    return 1;
  }
  const boundingBox = new THREE.Box3();
  for (const box of boxes) {
    const [minX, minY, minZ, maxX, maxY, maxZ] = box;
    boundingBox.expandByPoint(new THREE.Vector3(minX, minY, minZ));
    boundingBox.expandByPoint(new THREE.Vector3(maxX, maxY, maxZ));
  }
  const sphere = new THREE.Sphere();
  boundingBox.getBoundingSphere(sphere);
  return sphere.radius > 0 ? sphere.radius : 1;
}

// Module-load self-registration of the spatial aabb-3d layer renderer. The
// registry erases the layer's display response to the base DisplayResponse, so
// the aabb-3d part-B is registered through the registry's renderer type at the
// erasure boundary.
registerSpatialLayerRenderer({
  displayKind: "aabb_3d",
  layerRenderer: createAabb3dObject as SpatialLayerRenderer,
});
