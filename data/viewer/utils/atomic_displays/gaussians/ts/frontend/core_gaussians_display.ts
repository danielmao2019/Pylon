import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import * as THREE from "three";
import {
  createTrackballCameraControls,
  type ThreeTrackballCameraControls,
} from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { GaussianDisplayResponse } from "./types/display_response";

interface GaussiansScene {
  renderer: THREE.WebGLRenderer;
  viewer: GaussianSplats3D.Viewer;
}

export function renderGaussiansDisplay({
  displayResponse,
}: {
  displayResponse: GaussianDisplayResponse;
}): HTMLElement {
  if (displayResponse.url === null) {
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder-surface";
    placeholder.textContent =
      "Placeholder for a benchmark result that is not materialized yet.";
    return placeholder;
  }

  const container = createGaussiansContainer({
    resourceUrl: displayResponse.url,
  });
  const status = createGaussiansStatus();
  container.append(status);
  void loadAndRenderGaussiansDisplay({
    container,
    status,
    displayResponse,
  });
  return container;
}

async function loadAndRenderGaussiansDisplay({
  container,
  status,
  displayResponse,
}: {
  container: HTMLDivElement;
  status: HTMLDivElement;
  displayResponse: GaussianDisplayResponse;
}): Promise<void> {
  if (displayResponse.url === null) {
    status.textContent = `${displayResponse.title}: missing artifact`;
    return;
  }
  try {
    const scene = createGaussiansScene({ container });
    await scene.viewer.addSplatScene(displayResponse.url, {
      showLoadingUI: false,
      progressiveLoad: false,
    });
    scene.viewer.start();
    const controls = createTrackballCameraControls({
      camera: scene.viewer.camera as THREE.PerspectiveCamera,
      renderer: scene.renderer,
    });
    registerGaussiansSceneResize({
      container,
      scene,
      controls,
    });
    registerGaussiansSceneCameraSync({
      container,
      scene,
      controls,
      displayResponse,
    });
    renderGaussiansScene({ scene, controls });
    status.textContent = `${displayResponse.title}: Loaded`;
    publishGaussiansCameraState({
      container,
      scene,
      controls,
      displayResponse,
    });
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    status.textContent = `${displayResponse.title}: Error: ${message}`;
    console.error("Failed to load Gaussian splat artifact:", error);
  }
}

function createGaussiansContainer({
  resourceUrl,
}: {
  resourceUrl: string;
}): HTMLDivElement {
  const container = document.createElement("div");
  container.className = "artifact-frame gaussian-display-scene";
  container.style.position = "relative";
  container.style.overflow = "hidden";
  container.style.minHeight = "320px";
  container.style.background = "#020617";
  container.dataset.gaussianDisplayRenderer = "three";
  container.dataset.resourceUrl = resourceUrl;
  return container;
}

function createGaussiansStatus(): HTMLDivElement {
  const status = document.createElement("div");
  status.className = "gaussian-display-scene__status";
  status.textContent = "Loading Gaussian splats";
  status.style.position = "absolute";
  status.style.left = "10px";
  status.style.top = "10px";
  status.style.zIndex = "2";
  status.style.borderRadius = "6px";
  status.style.padding = "5px 7px";
  status.style.background = "rgba(2, 6, 23, 0.72)";
  status.style.color = "#f8fafc";
  status.style.fontSize = "12px";
  status.style.fontWeight = "700";
  return status;
}

function createGaussiansScene({
  container,
}: {
  container: HTMLDivElement;
}): GaussiansScene {
  const renderer = new THREE.WebGLRenderer({
    antialias: false,
    precision: "highp",
    preserveDrawingBuffer: true,
  });
  renderer.setPixelRatio(window.devicePixelRatio || 1);
  renderer.autoClear = true;
  renderer.setClearColor(new THREE.Color(0x000000), 0.0);
  renderer.domElement.style.display = "block";
  renderer.domElement.style.width = "100%";
  renderer.domElement.style.height = "100%";
  container.append(renderer.domElement);
  const initialWidth = container.clientWidth || 1;
  const initialHeight = container.clientHeight || 1;
  renderer.setSize(initialWidth, initialHeight);

  const viewer = new GaussianSplats3D.Viewer({
    rootElement: container,
    renderer,
    selfDrivenMode: true,
    useBuiltInControls: false,
    sphericalHarmonicsDegree: 3,
    gpuAcceleratedSort: false,
    sharedMemoryForWorkers: false,
    dynamicScene: false,
  });

  return { renderer, viewer };
}

function renderGaussiansScene({
  scene: _scene,
  controls,
}: {
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
}): void {
  const tick = (): void => {
    controls.update();
    requestAnimationFrame(tick);
  };
  requestAnimationFrame(tick);
}

function registerGaussiansSceneResize({
  container,
  scene,
  controls,
}: {
  container: HTMLDivElement;
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
}): void {
  const resize = (): void => {
    const width = container.clientWidth || 1;
    const height = container.clientHeight || 1;
    scene.renderer.setSize(width, height);
    const camera = scene.viewer.camera as THREE.PerspectiveCamera;
    if (camera !== undefined && camera.isPerspectiveCamera === true) {
      camera.aspect = width / height;
      camera.updateProjectionMatrix();
    }
    controls.handleResize();
  };
  window.addEventListener("resize", resize);
  const resizeObserver = new ResizeObserver(resize);
  resizeObserver.observe(container);
  resize();
}

function registerGaussiansSceneCameraSync({
  container,
  scene,
  controls,
  displayResponse,
}: {
  container: HTMLDivElement;
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
  displayResponse: GaussianDisplayResponse;
}): void {
  const applyContainerCameraState = (): void => {
    const cameraState = readCameraStateFromContainer(container);
    if (cameraState === null) {
      return;
    }
    applyCameraStateToGaussianScene({
      scene,
      controls,
      cameraState,
    });
  };
  const observer = new MutationObserver(applyContainerCameraState);
  observer.observe(container, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });
  controls.addEventListener("change", () => {
    publishGaussiansCameraState({
      container,
      scene,
      controls,
      displayResponse,
    });
  });
  applyContainerCameraState();
}

function publishGaussiansCameraState({
  container,
  scene,
  controls,
  displayResponse,
}: {
  container: HTMLDivElement;
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
  displayResponse: GaussianDisplayResponse;
}): void {
  const cameraState = buildGaussiansCameraState({
    scene,
    controls,
    displayResponse,
  });
  container.dataset.cameraState = JSON.stringify(cameraState);
  container.dispatchEvent(
    new CustomEvent<CameraState>("camera-pose-change", {
      bubbles: true,
      detail: cameraState,
    }),
  );
}

function buildGaussiansCameraState({
  scene,
  controls,
  displayResponse,
}: {
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
  displayResponse: GaussianDisplayResponse;
}): CameraState {
  const camera = scene.viewer.camera as THREE.PerspectiveCamera;
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
      target: vectorToRecord(controls.target),
      up: vectorToRecord(camera.up),
    },
    convention: "trackball_gaussian_splatting",
    name: displayResponse.title,
    id: displayResponse.url,
  };
}

function readCameraStateFromContainer(
  container: HTMLElement,
): CameraState | null {
  const serializedCameraState = container.dataset.cameraState;
  if (serializedCameraState === undefined) {
    return null;
  }
  const parsedCameraState: unknown = JSON.parse(serializedCameraState);
  if (!isCameraState(parsedCameraState)) {
    return null;
  }
  return parsedCameraState;
}

function applyCameraStateToGaussianScene({
  scene,
  controls,
  cameraState,
}: {
  scene: GaussiansScene;
  controls: ThreeTrackballCameraControls;
  cameraState: CameraState;
}): void {
  const camera = scene.viewer.camera as THREE.PerspectiveCamera;
  if (camera === undefined || camera.isPerspectiveCamera !== true) {
    return;
  }
  const position = readFiniteVector(cameraState.extrinsics.position);
  const target = readFiniteVector(cameraState.extrinsics.target);
  const up = readFiniteVector(cameraState.extrinsics.up);
  if (position === null || target === null || up === null) {
    return;
  }
  camera.position.set(position.x, position.y, position.z);
  camera.up.set(up.x, up.y, up.z);
  camera.lookAt(target.x, target.y, target.z);
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

function readFiniteVector(value: unknown): {
  x: number;
  y: number;
  z: number;
} | null {
  if (
    value === null ||
    typeof value !== "object" ||
    typeof (value as { x?: unknown }).x !== "number" ||
    typeof (value as { y?: unknown }).y !== "number" ||
    typeof (value as { z?: unknown }).z !== "number"
  ) {
    return null;
  }
  const vector = value as { x: number; y: number; z: number };
  if (
    !Number.isFinite(vector.x) ||
    !Number.isFinite(vector.y) ||
    !Number.isFinite(vector.z)
  ) {
    return null;
  }
  return { x: vector.x, y: vector.y, z: vector.z };
}

function isCameraState(value: unknown): value is CameraState {
  if (value === null || typeof value !== "object") {
    return false;
  }
  const record = value as Record<string, unknown>;
  return (
    record.intrinsics !== null &&
    typeof record.intrinsics === "object" &&
    record.extrinsics !== null &&
    typeof record.extrinsics === "object" &&
    typeof record.convention === "string"
  );
}
