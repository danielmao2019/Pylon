import * as THREE from "three";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import {
  DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";

// Any spatial display container augmented with an additive base-camera pick seam:
// a consumer can raycast a pointer position (in client/CSS pixels) against the
// container's scenes via the camera without owning the camera, renderer, or scenes
// itself. The base HTMLDivElement contract is unchanged — `pickAt` is an additive
// property other container consumers ignore.
export type PickableThreeContainer = HTMLDivElement & {
  pickAt: (clientX: number, clientY: number) => THREE.Object3D | null;
};

// Shared part-A "create scene" step for every spatial display (standalone
// renderers and the layered container alike): composes the one
// container/scene/camera/renderer and nothing else; callers create and add their
// own object(s) separately.
//
// Args:
//   initialCameraState: the single source of initial framing (camera-to-world
//     extrinsics + intrinsics) overlaid onto the camera; null uses the camera's
//     default framing.
//   pointerEventsSuppressed: when true the container suppresses pointer events so
//     an underlying base spatial display remains the interaction source; defaults
//     to false.
//
// Returns:
//   The composed container, empty scene, perspective camera, and WebGL renderer.
export function createSpatialDisplayScene({
  initialCameraState,
  pointerEventsSuppressed = false,
}: {
  initialCameraState: CameraState | null;
  pointerEventsSuppressed?: boolean;
}): {
  container: HTMLDivElement;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
} {
  const container = createThreeDisplayContainer({ pointerEventsSuppressed });
  const camera = createThreePerspectiveCamera({ initialCameraState });
  const renderer = createThreeWebGLRenderer({ container });
  const scene = createThreeScene();
  return { container, scene, camera, renderer };
}

export function createThreeDisplayContainer({
  pointerEventsSuppressed,
}: {
  pointerEventsSuppressed: boolean;
}): HTMLDivElement {
  const container = document.createElement("div");
  container.style.position = "absolute";
  container.style.inset = "0";
  container.style.width = "100%";
  container.style.height = "100%";
  container.style.overflow = "hidden";
  if (pointerEventsSuppressed) {
    container.style.pointerEvents = "none";
  }
  return container;
}

export function createThreeScene(): THREE.Scene {
  const scene = new THREE.Scene();
  return scene;
}

export function createThreePerspectiveCamera({
  initialCameraState,
}: {
  initialCameraState: CameraState | null;
}): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(
    DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV,
    1,
    0.01,
    1000,
  );
  camera.position.set(0, 0, 1);
  camera.up.set(0, 1, 0);
  camera.lookAt(new THREE.Vector3(0, 0, 0));
  camera.updateProjectionMatrix();
  if (initialCameraState !== null) {
    applyCameraStateToThreeCamera({ camera, cameraState: initialCameraState });
  }
  return camera;
}

export function createThreeWebGLRenderer({
  container,
}: {
  container: HTMLDivElement;
}): THREE.WebGLRenderer {
  const renderer = new THREE.WebGLRenderer({
    alpha: true,
    antialias: true,
    preserveDrawingBuffer: true,
  });
  renderer.setClearColor(0x000000, 0);
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  container.append(renderer.domElement);
  return renderer;
}

// Installs a base-camera pick seam (`pickAt`) onto any spatial display container so a
// consumer can hit-test the given scenes via the camera without owning the camera,
// renderer, or scenes itself. The base HTMLDivElement contract is untouched — `pickAt`
// is an additive property; containers and consumers that never call it are unaffected.
//
// Args:
//   container: the spatial display container the pick seam is attached onto; its
//     bounding rect converts client/CSS pixels to NDC.
//   camera: the base camera the raycaster casts from, so picks match what the user
//     sees.
//   scenes: the scenes whose objects are raycast against, iterated in order; the first
//     scene with a hit wins.
//
// Returns:
//   void.
export function attachThreeScenePickSeam({
  container,
  camera,
  scenes,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
  scenes: readonly THREE.Scene[];
}): void {
  const raycaster = new THREE.Raycaster();
  const pickAt = (clientX: number, clientY: number): THREE.Object3D | null => {
    const rect = container.getBoundingClientRect();
    if (rect.width <= 0 || rect.height <= 0) {
      return null;
    }
    const ndc = new THREE.Vector2(
      ((clientX - rect.left) / rect.width) * 2 - 1,
      -((clientY - rect.top) / rect.height) * 2 + 1,
    );
    raycaster.setFromCamera(ndc, camera);
    for (const scene of scenes) {
      const intersections = raycaster.intersectObjects(scene.children, true);
      if (intersections.length > 0) {
        return intersections[0].object;
      }
    }
    return null;
  };
  (container as PickableThreeContainer).pickAt = pickAt;
}

export function startThreeSceneRenderLoop({
  scene,
  camera,
  renderer,
  controls,
  onAfterRender,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  controls: ThreeTrackballCameraControls | null;
  onAfterRender?: () => void;
}): void {
  let wasConnected = false;
  const draw = (): void => {
    const connected = renderer.domElement.isConnected;
    if (connected) {
      wasConnected = true;
    }
    if (wasConnected && !connected) {
      renderer.dispose();
      renderer.forceContextLoss();
      return;
    }
    if (controls !== null) {
      controls.update();
    }
    renderer.render(scene, camera);
    if (onAfterRender !== undefined) {
      onAfterRender();
    }
    window.requestAnimationFrame(draw);
  };
  window.requestAnimationFrame(draw);
}

export function applyCameraStateToThreeCamera({
  camera,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  cameraState: CameraState;
}): void {
  const position = cameraState.extrinsics.position;
  const quaternion = cameraState.extrinsics.quaternion;
  const up = cameraState.extrinsics.up;
  const fov = cameraState.intrinsics.fov;
  const aspect = cameraState.intrinsics.aspect;
  const near = cameraState.intrinsics.near;
  const far = cameraState.intrinsics.far;
  if (
    !_isVectorRecord(position) ||
    !_isVectorRecord(up) ||
    typeof fov !== "number" ||
    typeof aspect !== "number" ||
    typeof near !== "number" ||
    typeof far !== "number"
  ) {
    return;
  }
  camera.position.set(position.x, position.y, position.z);
  camera.up.set(up.x, up.y, up.z);
  if (_isQuaternionRecord(quaternion)) {
    camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
  }
  camera.fov = fov;
  camera.aspect = aspect;
  camera.near = near;
  camera.far = far;
  camera.updateProjectionMatrix();
}

function _isVectorRecord(value: unknown): value is { x: number; y: number; z: number } {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { x?: unknown }).x === "number" &&
    typeof (value as { y?: unknown }).y === "number" &&
    typeof (value as { z?: unknown }).z === "number"
  );
}

function _isQuaternionRecord(
  value: unknown,
): value is { x: number; y: number; z: number; w: number } {
  return _isVectorRecord(value) && typeof (value as { w?: unknown }).w === "number";
}
