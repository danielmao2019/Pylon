import * as THREE from "three";
import { TrackballControls as ThreeTrackballControlsImpl } from "three/examples/jsm/controls/TrackballControls.js";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";

type CameraStateListener = (cameraState: CameraState) => void;

interface RendererTrackballCameraControls {
  targetElement: HTMLElement;
  getCameraState: () => CameraState | null;
  applyCameraState: (cameraState: CameraState | null) => void;
  subscribeCameraStateChange: (
    listener: CameraStateListener,
  ) => () => void;
}

export interface TrackballCameraControls {
  getCameraState: () => CameraState | null;
  applyCameraState: (cameraState: CameraState | null) => void;
  subscribeCameraStateChange: (
    listener: CameraStateListener,
  ) => () => void;
}

export interface ThreeTrackballCameraControls extends TrackballCameraControls {
  target: THREE.Vector3;
  addEventListener: (type: "change", listener: () => void) => void;
  handleResize: () => void;
  update: () => void;
}

export function createTrackballCameraControls(args: {
  targetElement: HTMLElement;
  initialCameraState?: CameraState | null;
}): TrackballCameraControls;

export function createTrackballCameraControls(args: {
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
}): ThreeTrackballCameraControls;

export function createTrackballCameraControls(args:
  | {
      targetElement: HTMLElement;
      initialCameraState?: CameraState | null;
    }
  | {
      camera: THREE.PerspectiveCamera;
      renderer: THREE.WebGLRenderer;
    },
): TrackballCameraControls | ThreeTrackballCameraControls {
  if ("camera" in args) {
    return createThreeTrackballCameraControls(args);
  }
  const { targetElement, initialCameraState = null } = args;
  const controls = createRendererTrackballCameraControls({
    targetElement,
    initialCameraState,
  });
  assertTrackballCameraControls(controls);
  return controls;
}

function createThreeTrackballCameraControls(args: {
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
}): ThreeTrackballCameraControls {
  const { camera, renderer } = args;
  const threeControls = new ThreeTrackballControlsImpl(camera, renderer.domElement);
  const listeners = new Set<CameraStateListener>();
  threeControls.rotateSpeed = 3;
  threeControls.zoomSpeed = 1.5;
  threeControls.panSpeed = 0.8;
  threeControls.staticMoving = true;
  renderer.domElement.addEventListener("contextmenu", (event: MouseEvent) => {
    event.preventDefault();
  });
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
    convention: "three_trackball",
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
  if (cameraState === null || cameraState.convention !== "three_trackball") {
    return;
  }
  const position = cameraState.extrinsics.position;
  const quaternion = cameraState.extrinsics.quaternion;
  const target = cameraState.extrinsics.target;
  const up = cameraState.extrinsics.up;
  if (
    !isVectorRecord(position) ||
    !isQuaternionRecord(quaternion) ||
    !isVectorRecord(target) ||
    !isVectorRecord(up)
  ) {
    return;
  }
  camera.position.set(position.x, position.y, position.z);
  camera.quaternion.set(quaternion.x, quaternion.y, quaternion.z, quaternion.w);
  camera.up.set(up.x, up.y, up.z);
  controls.target.set(target.x, target.y, target.z);
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

function createRendererTrackballCameraControls(args: {
  targetElement: HTMLElement;
  initialCameraState: CameraState | null;
}): RendererTrackballCameraControls {
  const { targetElement, initialCameraState } = args;
  let currentCameraState = initialCameraState;
  let internallyWrittenCameraStateToken: string | null | undefined = undefined;
  const listeners: CameraStateListener[] = [];

  const setInternallyWrittenCameraStateToken = (
    token: string | null,
  ): void => {
    internallyWrittenCameraStateToken = token;
  };

  const applyCameraState = (cameraState: CameraState | null): void => {
    currentCameraState = cameraState;
    writeInternalCameraStateToTargetElement({
      targetElement,
      cameraState,
      setInternallyWrittenCameraStateToken,
    });
    postCameraStateToEmbeddedRenderer({
      targetElement,
      cameraState,
    });
  };

  const emitCameraStateChange = (cameraState: CameraState): void => {
    currentCameraState = cameraState;
    writeInternalCameraStateToTargetElement({
      targetElement,
      cameraState,
      setInternallyWrittenCameraStateToken,
    });
    for (const listener of listeners) {
      listener(cameraState);
    }
    targetElement.dispatchEvent(
      new CustomEvent<CameraState>("camera-pose-change", {
        bubbles: true,
        detail: cameraState,
      }),
    );
  };

  const mutationObserver = new MutationObserver(() => {
    const targetElementCameraStateToken =
      readCameraStateTokenFromTargetElement(targetElement);
    if (
      internallyWrittenCameraStateToken !== undefined &&
      targetElementCameraStateToken === internallyWrittenCameraStateToken
    ) {
      internallyWrittenCameraStateToken = undefined;
      return;
    }
    internallyWrittenCameraStateToken = undefined;
    applyExternalCameraState(readCameraStateFromTargetElement(targetElement));
  });
  mutationObserver.observe(targetElement, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });

  window.addEventListener("message", (event: MessageEvent<unknown>) => {
    if (!isEmbeddedRendererMessageSource({ targetElement, source: event.source })) {
      return;
    }
    if (event.origin !== window.location.origin) {
      return;
    }
    const message = event.data;
    if (!isTrackballCameraStateChangeMessage(message)) {
      return;
    }
    emitCameraStateChange(message.cameraState);
  });
  if (targetElement instanceof HTMLIFrameElement) {
    targetElement.addEventListener("load", () => {
      postCameraStateToEmbeddedRenderer({
        targetElement,
        cameraState: currentCameraState,
      });
    });
  }

  targetElement.dataset.cameraControlMode = "trackball";
  targetElement.dataset.trackballMouseMapping =
    "left-drag-rotate/right-drag-pan/wheel-zoom";
  targetElement.dataset.contextMenuBehavior = "suppressed-for-trackball-pan";
  if (currentCameraState !== null) {
    applyCameraState(currentCameraState);
  }

  function applyExternalCameraState(cameraState: CameraState | null): void {
    currentCameraState = cameraState;
    postCameraStateToEmbeddedRenderer({
      targetElement,
      cameraState,
    });
  }

  return {
    targetElement,
    getCameraState: () => currentCameraState,
    applyCameraState,
    subscribeCameraStateChange: (listener: CameraStateListener) => {
      if (typeof listener !== "function") {
        throw new Error("camera state listener must be a function");
      }
      listeners.push(listener);
      return () => {
        const index = listeners.indexOf(listener);
        if (index >= 0) {
          listeners.splice(index, 1);
        }
      };
    },
  };
}

function assertTrackballCameraControls(
  controls: RendererTrackballCameraControls,
): void {
  assertTrackballMouseMapping(controls);
  assertNoOrbitCameraControls(controls);
  assertNoCameraPoseClamps(controls);
}

function assertTrackballMouseMapping(
  controls: RendererTrackballCameraControls,
): void {
  const mapping = controls.targetElement.dataset.trackballMouseMapping;
  if (mapping !== "left-drag-rotate/right-drag-pan/wheel-zoom") {
    throw new Error("invalid trackball camera controls");
  }
  if (
    controls.targetElement.dataset.contextMenuBehavior !==
    "suppressed-for-trackball-pan"
  ) {
    throw new Error("context menu blocks trackball panning");
  }
}

function assertNoOrbitCameraControls(
  controls: RendererTrackballCameraControls,
): void {
  const mode = controls.targetElement.dataset.cameraControlMode;
  const family = controls.targetElement.dataset.cameraControlFamily;
  if (mode === "orbit" || family === "orbit") {
    throw new Error("orbit-style camera controls are forbidden");
  }
}

function assertNoCameraPoseClamps(
  controls: RendererTrackballCameraControls,
): void {
  const forbiddenRestrictionKeys = [
    "cameraPolarAngleLimit",
    "cameraAzimuthAngleLimit",
    "cameraTargetLock",
    "cameraDistanceBounds",
    "cameraPanLimit",
    "cameraTranslationLimit",
    "cameraRotationLimit",
  ];
  const restrictedKey = forbiddenRestrictionKeys.find(
    (key) => controls.targetElement.dataset[key] !== undefined,
  );
  if (restrictedKey !== undefined) {
    throw new Error(`restricted camera pose controls: ${restrictedKey}`);
  }
}

function readCameraStateFromTargetElement(
  targetElement: HTMLElement,
): CameraState | null {
  const serializedCameraState = targetElement.dataset.cameraState;
  if (serializedCameraState === undefined) {
    return null;
  }
  const parsedCameraState: unknown = JSON.parse(serializedCameraState);
  if (!isCameraState(parsedCameraState)) {
    throw new Error("target camera state does not match CameraState");
  }
  return parsedCameraState;
}

function readCameraStateTokenFromTargetElement(
  targetElement: HTMLElement,
): string | null {
  return targetElement.dataset.cameraState ?? null;
}

function serializeCameraState(cameraState: CameraState | null): string | null {
  if (cameraState === null) {
    return null;
  }
  return JSON.stringify(cameraState);
}

function writeInternalCameraStateToTargetElement(args: {
  targetElement: HTMLElement;
  cameraState: CameraState | null;
  setInternallyWrittenCameraStateToken: (token: string | null) => void;
}): void {
  const {
    targetElement,
    cameraState,
    setInternallyWrittenCameraStateToken,
  } = args;
  const serializedCameraState = serializeCameraState(cameraState);
  if (
    writeCameraStateToTargetElement({
      targetElement,
      cameraState,
      serializedCameraState,
    })
  ) {
    setInternallyWrittenCameraStateToken(serializedCameraState);
  }
}

function writeCameraStateToTargetElement(args: {
  targetElement: HTMLElement;
  cameraState: CameraState | null;
  serializedCameraState: string | null;
}): boolean {
  const { targetElement, cameraState, serializedCameraState } = args;
  if (
    readCameraStateTokenFromTargetElement(targetElement) ===
    serializedCameraState
  ) {
    return false;
  }
  if (cameraState === null) {
    delete targetElement.dataset.cameraState;
    return true;
  }
  if (serializedCameraState === null) {
    throw new Error("serialized camera state is unexpectedly null");
  }
  targetElement.dataset.cameraState = serializedCameraState;
  return true;
}

function postCameraStateToEmbeddedRenderer(args: {
  targetElement: HTMLElement;
  cameraState: CameraState | null;
}): void {
  const { targetElement, cameraState } = args;
  if (!(targetElement instanceof HTMLIFrameElement)) {
    return;
  }
  const targetWindow = targetElement.contentWindow;
  if (targetWindow === null) {
    return;
  }
  targetWindow.postMessage(
    {
      cameraState,
      type: "trackball-camera-state",
    },
    window.location.origin,
  );
}

function isEmbeddedRendererMessageSource(args: {
  targetElement: HTMLElement;
  source: MessageEventSource | null;
}): boolean {
  const { targetElement, source } = args;
  return (
    targetElement instanceof HTMLIFrameElement &&
    source !== null &&
    source === targetElement.contentWindow
  );
}

function isTrackballCameraStateChangeMessage(
  value: unknown,
): value is { type: "trackball-camera-state-change"; cameraState: CameraState } {
  return (
    isRecord(value) &&
    value.type === "trackball-camera-state-change" &&
    isCameraState(value.cameraState)
  );
}

function isCameraState(value: unknown): value is CameraState {
  return (
    isRecord(value) &&
    isRecord(value.intrinsics) &&
    isRecord(value.extrinsics) &&
    typeof value.convention === "string" &&
    (value.name === null || typeof value.name === "string") &&
    (value.id === null || typeof value.id === "string")
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return value !== null && typeof value === "object";
}
