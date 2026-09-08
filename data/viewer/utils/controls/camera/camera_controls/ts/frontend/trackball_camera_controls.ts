import * as THREE from "three";
import { TrackballControls as ThreeTrackballControlsImpl } from "three/examples/jsm/controls/TrackballControls.js";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";

export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45;

// Squared length below which a cross product no longer defines a direction: the
// view direction runs parallel to the roll-lock axis and their cross product
// collapses, so the camera right axis is carried instead of re-derived.
const ROLL_LOCK_DEGENERACY_EPSILON_SQUARED = 1e-12;
// Radians of camera rotation per pixel of roll-locked left-drag.
const ROLL_LOCKED_ROTATE_SPEED = 0.005;

type CameraStateListener = (cameraState: CameraState) => void;

export interface TrackballCameraControls {
  getCameraState: () => CameraState | null;
  applyCameraState: (cameraState: CameraState | null) => void;
  subscribeCameraStateChange: (
    listener: CameraStateListener,
  ) => () => void;
}

export interface ThreeTrackballCameraControls extends TrackballCameraControls {
  target: THREE.Vector3;
  noRotate: boolean;
  noZoom: boolean;
  noPan: boolean;
  minDistance: number;
  maxDistance: number;
  rollLockAxis: THREE.Vector3 | null;
  addEventListener: (type: "change", listener: () => void) => void;
  handleResize: () => void;
  update: () => void;
}

// Builds, validates, and returns the trackball controls, seeding them from
// initialCameraState and observing the container's data-camera-state attribute
// for external sync.
//
// Args:
//   container: the display container the controls observe data-camera-state on.
//   camera: the perspective camera the controls drive.
//   renderer: the WebGL renderer whose canvas receives the pointer events.
//   initialCameraState: initial framing (camera-to-world extrinsics + intrinsics);
//     null uses the camera's default framing.
//   lockRoll: the world-space axis to lock camera roll about, in the scene's own
//     world frame; null leaves the controls exactly as three constructed them.
//     This module owns no axis of its own, so the axis is always the caller's.
//
// Returns:
//   The validated trackball controls.
export function createTrackballCameraControls({
  container,
  camera,
  renderer,
  initialCameraState = null,
  lockRoll = null,
}: {
  container: HTMLElement;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  initialCameraState?: CameraState | null;
  lockRoll?: THREE.Vector3 | null;
}): ThreeTrackballCameraControls {
  const controls = createRendererTrackballCameraControls({
    camera,
    renderer,
    lockRoll,
  });
  assertTrackballCameraControls({ controls, renderer, lockRoll });
  if (initialCameraState !== null) {
    controls.applyCameraState(initialCameraState);
  }
  const observer = new MutationObserver(() => {
    const serializedCameraState = container.dataset.cameraState;
    if (serializedCameraState === undefined) {
      return;
    }
    controls.applyCameraState(JSON.parse(serializedCameraState) as CameraState);
  });
  observer.observe(container, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });
  return controls;
}

// Constructs the renderer-specific trackball controls wiring left-drag rotate,
// right-drag pan, wheel zoom, and context-menu suppression.
//
// Args:
//   camera: the perspective camera the controls drive.
//   renderer: the WebGL renderer whose canvas receives the pointer events.
//   lockRoll: the world-space axis to lock camera roll about; null leaves the
//     controls exactly as three constructed them, so a caller naming no axis
//     renders what it rendered before this argument existed.
//
// Returns:
//   The renderer-specific trackball controls.
function createRendererTrackballCameraControls({
  camera,
  renderer,
  lockRoll,
}: {
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
  lockRoll: THREE.Vector3 | null;
}): ThreeTrackballCameraControls {
  const threeControls = new ThreeTrackballControlsImpl(camera, renderer.domElement);
  const listeners = new Set<CameraStateListener>();
  const rollLockAxis = lockRoll === null ? null : lockRoll.clone().normalize();
  threeControls.rotateSpeed = 3;
  threeControls.zoomSpeed = 1.5;
  threeControls.panSpeed = 0.8;
  threeControls.staticMoving = true;
  renderer.domElement.addEventListener("contextmenu", (event: MouseEvent) => {
    event.preventDefault();
  });

  if (rollLockAxis !== null) {
    // Three's own rotation is the free trackball that carries camera.up along
    // with the drag; the roll-locked left-drag below replaces it, leaving three's
    // right-drag pan and wheel zoom untouched.
    threeControls.noRotate = true;

    const cameraRightAxis = new THREE.Vector3().crossVectors(
      threeControls.target.clone().sub(camera.position),
      rollLockAxis,
    );
    if (cameraRightAxis.lengthSq() < ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
      // The camera already looks along the lock axis, so every axis perpendicular
      // to the lock axis is an equally valid camera right axis to start from.
      cameraRightAxis.set(1, 0, 0).cross(rollLockAxis);
      if (cameraRightAxis.lengthSq() < ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
        cameraRightAxis.set(0, 1, 0).cross(rollLockAxis);
      }
    }
    cameraRightAxis.normalize();

    let leftDragActive = false;
    let lastClientX = 0;
    let lastClientY = 0;
    renderer.domElement.addEventListener("pointerdown", (event: PointerEvent) => {
      if (event.button !== 0) {
        return;
      }
      leftDragActive = true;
      lastClientX = event.clientX;
      lastClientY = event.clientY;
    });
    window.addEventListener("pointerup", () => {
      leftDragActive = false;
    });
    window.addEventListener("pointermove", (event: PointerEvent) => {
      if (!leftDragActive) {
        return;
      }
      const deltaX = event.clientX - lastClientX;
      const deltaY = event.clientY - lastClientY;
      lastClientX = event.clientX;
      lastClientY = event.clientY;
      const offset = camera.position.clone().sub(threeControls.target);
      const yaw = new THREE.Quaternion().setFromAxisAngle(
        rollLockAxis,
        -deltaX * ROLL_LOCKED_ROTATE_SPEED,
      );
      offset.applyQuaternion(yaw);
      cameraRightAxis.applyQuaternion(yaw);
      const yawedRightAxis = new THREE.Vector3().crossVectors(
        offset.clone().negate(),
        rollLockAxis,
      );
      if (yawedRightAxis.lengthSq() >= ROLL_LOCK_DEGENERACY_EPSILON_SQUARED) {
        yawedRightAxis.normalize();
        // Past a pole the re-derived axis points the opposite way, which would
        // reverse the next pitch and bounce the camera off the pole; taking the
        // carried axis's orientation pitches straight through and out the far side.
        if (yawedRightAxis.dot(cameraRightAxis) < 0) {
          yawedRightAxis.negate();
        }
        cameraRightAxis.copy(yawedRightAxis);
      }
      const pitch = new THREE.Quaternion().setFromAxisAngle(
        cameraRightAxis,
        -deltaY * ROLL_LOCKED_ROTATE_SPEED,
      );
      offset.applyQuaternion(pitch);
      camera.position.copy(threeControls.target).add(offset);
      camera.up
        .crossVectors(cameraRightAxis, offset.clone().negate().normalize())
        .normalize();
      camera.lookAt(threeControls.target);
      threeControls.dispatchEvent({ type: "change" });
    });
  }

  threeControls.addEventListener("change", () => {
    const cameraState = buildThreeTrackballCameraState({
      camera,
      controls: threeControls,
    });
    for (const listener of listeners) {
      listener(cameraState);
    }
  });
  return Object.assign(threeControls, {
    rollLockAxis,
    getCameraState: () =>
      buildThreeTrackballCameraState({
        camera,
        controls: threeControls,
      }),
    applyCameraState: (cameraState: CameraState | null): void => {
      applyThreeTrackballCameraState({
        camera,
        controls: threeControls,
        cameraState,
      });
    },
    subscribeCameraStateChange: (listener: CameraStateListener) => {
      if (typeof listener !== "function") {
        throw new Error("camera state listener must be a function");
      }
      listeners.add(listener);
      return () => {
        listeners.delete(listener);
      };
    },
  });
}

// Validates the constructed controls satisfy every trackball contract by running
// the mouse-mapping, no-orbit, no-pose-clamp, and roll-lock assertions.
//
// Args:
//   controls: the constructed trackball controls.
//   renderer: the WebGL renderer whose canvas the controls listen on.
//   lockRoll: the world-space axis the controls were asked to lock roll about;
//     null asserts the free trackball.
//
// Returns:
//   void.
function assertTrackballCameraControls({
  controls,
  renderer,
  lockRoll,
}: {
  controls: ThreeTrackballCameraControls;
  renderer: THREE.WebGLRenderer;
  lockRoll: THREE.Vector3 | null;
}): void {
  assertTrackballMouseMapping({ controls, renderer });
  assertNoOrbitCameraControls({ controls });
  assertNoCameraPoseClamps({ controls, lockRoll });
  assertRollLock({ controls, lockRoll });
}

// Asserts the controls map left-drag to rotate, right-drag to pan, and wheel to
// zoom, and that the canvas suppresses its context menu.
//
// Args:
//   controls: the constructed trackball controls.
//   renderer: the WebGL renderer whose canvas the controls listen on.
//
// Returns:
//   void.
function assertTrackballMouseMapping({
  controls,
  renderer,
}: {
  controls: ThreeTrackballCameraControls;
  renderer: THREE.WebGLRenderer;
}): void {
  // Three's trackball fixes left-drag to rotation, right-drag to pan, and the
  // wheel to zoom, so each mapping is live exactly when its disable flag is off;
  // a roll-locked construction hands the left-drag to its own rotation instead.
  const leftDragRotates = !controls.noRotate || controls.rollLockAxis !== null;
  if (!leftDragRotates || controls.noPan || controls.noZoom) {
    throw new Error(
      `invalid trackball camera controls: noRotate=${controls.noRotate} noPan=${controls.noPan} noZoom=${controls.noZoom}`,
    );
  }
  // Context-menu suppression lives in a listener, so the only way to read it back
  // is to put a cancelable contextmenu event through the canvas; every listener on
  // it does nothing but preventDefault, so the probe leaves no state behind.
  const contextMenuProbe = new MouseEvent("contextmenu", {
    bubbles: false,
    cancelable: true,
  });
  renderer.domElement.dispatchEvent(contextMenuProbe);
  if (!contextMenuProbe.defaultPrevented) {
    throw new Error("context menu blocks trackball panning");
  }
}

// Asserts the controls do not use forbidden orbit-style target-locked camera
// semantics.
//
// Args:
//   controls: the constructed trackball controls.
//
// Returns:
//   void.
function assertNoOrbitCameraControls({
  controls,
}: {
  controls: ThreeTrackballCameraControls;
}): void {
  if (!(controls instanceof ThreeTrackballControlsImpl)) {
    throw new Error("orbit-style camera controls are forbidden");
  }
}

// Asserts the controls impose no camera-pose restriction on polar angle, azimuth
// angle, target lock, distance, pan, translation, or rotation.
//
// Args:
//   controls: the constructed trackball controls.
//   lockRoll: the world-space axis the controls were asked to lock roll about;
//     null forbids every rotation restriction.
//
// Returns:
//   void.
function assertNoCameraPoseClamps({
  controls,
  lockRoll,
}: {
  controls: ThreeTrackballCameraControls;
  lockRoll: THREE.Vector3 | null;
}): void {
  // Three's trackball has no polar, azimuth, target-lock, or translation clamp to
  // read: its whole pose-restriction surface is the pan flag and the distance
  // bounds, which stay at the unbounded defaults.
  if (
    controls.noPan ||
    controls.minDistance !== 0 ||
    controls.maxDistance !== Infinity
  ) {
    throw new Error(
      `restricted camera pose controls: noPan=${controls.noPan} minDistance=${controls.minDistance} maxDistance=${controls.maxDistance}`,
    );
  }
  if (lockRoll === null) {
    if (controls.noRotate) {
      throw new Error("restricted camera pose controls: noRotate=true");
    }
    return;
  }
  if (controls.noRotate && controls.rollLockAxis === null) {
    throw new Error(
      "roll lock must cost only the roll axis: three's rotation is off and no roll-locked rotation replaced it",
    );
  }
}

// Asserts roll is held about lockRoll when one is supplied and left free when
// none is, this module owning no axis of its own.
//
// Args:
//   controls: the constructed trackball controls.
//   lockRoll: the world-space axis the controls were asked to lock roll about;
//     null asserts the camera right axis is constrained against no axis at all.
//
// Returns:
//   void.
function assertRollLock({
  controls,
  lockRoll,
}: {
  controls: ThreeTrackballCameraControls;
  lockRoll: THREE.Vector3 | null;
}): void {
  if (lockRoll !== null) {
    if (
      controls.rollLockAxis === null ||
      !controls.rollLockAxis.equals(lockRoll.clone().normalize())
    ) {
      throw new Error(
        "roll-locked camera controls must keep the camera right axis perpendicular to the supplied axis",
      );
    }
    return;
  }
  if (controls.rollLockAxis !== null) {
    throw new Error(
      "free trackball camera controls must leave camera roll unconstrained",
    );
  }
}

function buildThreeTrackballCameraState({
  camera,
  controls,
}: {
  camera: THREE.PerspectiveCamera;
  controls: ThreeTrackballControlsImpl;
}): CameraState {
  return {
    intrinsics: {
      aspect: camera.aspect,
      far: camera.far,
      fov: camera.fov,
      near: camera.near,
      projection: "perspective-three",
    },
    extrinsics: {
      position: vectorToRecord(camera.position),
      quaternion: quaternionToRecord(camera.quaternion),
      target: vectorToRecord(controls.target),
      up: vectorToRecord(camera.up),
    },
    intr_convention: "three_trackball",
    extr_convention: "three_trackball",
    name: null,
    id: null,
  };
}

function applyThreeTrackballCameraState({
  camera,
  controls,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  controls: ThreeTrackballControlsImpl;
  cameraState: CameraState | null;
}): void {
  if (cameraState === null || cameraState.extr_convention !== "three_trackball") {
    return;
  }
  const position = cameraState.extrinsics.position;
  const quaternion = cameraState.extrinsics.quaternion;
  const target = cameraState.extrinsics.target;
  const up = cameraState.extrinsics.up;
  const aspect = cameraState.intrinsics.aspect;
  const far = cameraState.intrinsics.far;
  const fov = cameraState.intrinsics.fov;
  const near = cameraState.intrinsics.near;
  if (
    !isVectorRecord(position) ||
    !isQuaternionRecord(quaternion) ||
    !isVectorRecord(target) ||
    !isVectorRecord(up) ||
    typeof aspect !== "number" ||
    typeof far !== "number" ||
    typeof fov !== "number" ||
    typeof near !== "number"
  ) {
    return;
  }
  camera.position.set(position.x, position.y, position.z);
  camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
  camera.up.set(up.x, up.y, up.z);
  controls.target.set(target.x, target.y, target.z);
  camera.aspect = aspect;
  camera.far = far;
  camera.fov = fov;
  camera.near = near;
  camera.updateProjectionMatrix();
  controls.update();
}

function vectorToRecord(vector: THREE.Vector3): Record<string, number> {
  return {
    x: vector.x,
    y: vector.y,
    z: vector.z,
  };
}

function quaternionToRecord(
  quaternion: THREE.Quaternion,
): Record<string, number> {
  return {
    x: quaternion.x,
    y: quaternion.y,
    z: quaternion.z,
    w: quaternion.w,
  };
}

function isVectorRecord(value: unknown): value is {
  x: number;
  y: number;
  z: number;
} {
  return (
    typeof value === "object" &&
    value !== null &&
    typeof (value as { x?: unknown }).x === "number" &&
    typeof (value as { y?: unknown }).y === "number" &&
    typeof (value as { z?: unknown }).z === "number"
  );
}

function isQuaternionRecord(value: unknown): value is {
  x: number;
  y: number;
  z: number;
  w: number;
} {
  return (
    isVectorRecord(value) &&
    typeof (value as { w?: unknown }).w === "number"
  );
}
