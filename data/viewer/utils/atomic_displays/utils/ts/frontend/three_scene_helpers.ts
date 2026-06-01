import * as THREE from "three";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import {
  DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";

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
    _applyInitialCameraState({ camera, initialCameraState });
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
  const draw = (): void => {
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

function _applyInitialCameraState({
  camera,
  initialCameraState,
}: {
  camera: THREE.PerspectiveCamera;
  initialCameraState: CameraState;
}): void {
  const position = initialCameraState.extrinsics.position;
  const quaternion = initialCameraState.extrinsics.quaternion;
  const up = initialCameraState.extrinsics.up;
  const fov = initialCameraState.intrinsics.fov;
  const aspect = initialCameraState.intrinsics.aspect;
  const near = initialCameraState.intrinsics.near;
  const far = initialCameraState.intrinsics.far;
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
