import * as THREE from "three";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { CameraDisplayResponse } from "./types/display_response";

interface FiniteVector {
  x: number;
  y: number;
  z: number;
}

interface FiniteQuaternion extends FiniteVector {
  w: number;
}

type CameraVisualizationVectorPayload = [number, number, number];

interface CameraVisualizationLinePayload {
  start: CameraVisualizationVectorPayload;
  end: CameraVisualizationVectorPayload;
  color: CameraVisualizationVectorPayload;
}

interface CameraVisualizationPayload {
  center: CameraVisualizationVectorPayload;
  center_color: CameraVisualizationVectorPayload;
  axes: CameraVisualizationLinePayload[];
  frustum_lines: CameraVisualizationLinePayload[];
}

export function renderCameraDisplay({
  displayResponse,
}: {
  displayResponse: CameraDisplayResponse;
}): HTMLElement {
  if (Object.keys(displayResponse.meta_info).length !== 0) {
    throw new Error("camera display meta_info must be an empty object");
  }
  if (displayResponse.url === null) {
    throw new Error("camera display response url is null");
  }

  const container = createCameraDisplayContainer({
    resourceUrl: displayResponse.url,
  });
  void loadAndRenderCameraDisplay({
    container,
    displayResponse,
  });
  return container;
}

function createCameraDisplayContainer({
  resourceUrl,
}: {
  resourceUrl: string;
}): HTMLDivElement {
  const container = document.createElement("div");
  container.className = "artifact-frame camera-display-scene";
  container.style.position = "relative";
  container.style.overflow = "hidden";
  container.style.background = "transparent";
  container.style.pointerEvents = "none";
  container.dataset.resourceUrl = resourceUrl;
  return container;
}

async function loadAndRenderCameraDisplay({
  container,
  displayResponse,
}: {
  container: HTMLDivElement;
  displayResponse: CameraDisplayResponse;
}): Promise<void> {
  const resourceUrl = displayResponse.url;
  if (resourceUrl === null) {
    throw new Error("camera display response url is null");
  }
  const response = await fetch(resourceUrl);
  if (!response.ok) {
    throw new Error(`unable to load camera visualization: HTTP ${response.status}`);
  }
  const cameraVisualizations = validateCameraVisualizationPayloads({
    value: await response.json(),
  });
  const scene = new THREE.Scene();
  const cameraOverlay = createThreeCameraOverlayLines({
    cameraVisualizations,
  });
  scene.add(cameraOverlay);
  const camera = createThreePerspectiveCamera();
  const renderer = createThreeWebGLRenderer({ container });
  registerCameraDisplayVisibility({
    container,
    cameraOverlay,
  });
  registerCameraDisplayResize({
    container,
    camera,
    renderer,
  });
  registerCameraDisplayCameraSync({
    container,
    camera,
  });
  startCameraDisplayRenderLoop({
    scene,
    camera,
    renderer,
  });
}

function createThreePerspectiveCamera(): THREE.PerspectiveCamera {
  const camera = new THREE.PerspectiveCamera(60, 1, 0.01, 1000);
  camera.position.set(0, 0, 1);
  camera.up.set(0, 1, 0);
  camera.lookAt(new THREE.Vector3(0, 0, 0));
  camera.updateProjectionMatrix();
  return camera;
}

function createThreeWebGLRenderer({
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
  renderer.domElement.style.pointerEvents = "none";
  container.append(renderer.domElement);
  return renderer;
}

function registerCameraDisplayResize({
  container,
  camera,
  renderer,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
}): void {
  const resize = () => {
    const width = Math.max(1, container.clientWidth || 1);
    const height = Math.max(1, container.clientHeight || 1);
    camera.aspect = width / height;
    camera.updateProjectionMatrix();
    renderer.setSize(width, height, false);
  };
  resize();
  if (typeof ResizeObserver !== "undefined") {
    new ResizeObserver(resize).observe(container);
  }
  window.addEventListener("resize", resize);
}

function registerCameraDisplayVisibility({
  container,
  cameraOverlay,
}: {
  container: HTMLDivElement;
  cameraOverlay: THREE.Group;
}): void {
  const updateVisibility = () => {
    const showCameras = container.dataset.showCameras !== "false";
    cameraOverlay.visible = showCameras;
    container.dataset.cameraOverlayVisible = showCameras ? "true" : "false";
    container.dataset.cameraOverlayCount = String(
      cameraOverlay.userData.cameraCount ?? 0,
    );
    container.dataset.cameraOverlayLineCount = String(
      cameraOverlay.userData.lineCount ?? 0,
    );
    container.dataset.cameraOverlayFirstAxisLength = String(
      cameraOverlay.userData.firstAxisLength ?? "",
    );
    container.dataset.cameraOverlayFirstFrustumLength = String(
      cameraOverlay.userData.firstFrustumLength ?? "",
    );
  };
  updateVisibility();
  new MutationObserver(updateVisibility).observe(container, {
    attributeFilter: ["data-show-cameras"],
    attributes: true,
  });
}

function registerCameraDisplayCameraSync({
  container,
  camera,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
}): void {
  const applyContainerCameraState = () => {
    const cameraState = readCameraStateFromContainer({ container });
    if (cameraState === null) {
      return;
    }
    if (applyCameraState({ camera, cameraState })) {
      container.dataset.cameraAppliedState = JSON.stringify(cameraState);
      return;
    }
    delete container.dataset.cameraAppliedState;
  };
  const observer = new MutationObserver(applyContainerCameraState);
  observer.observe(container, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });
  applyContainerCameraState();
}

function startCameraDisplayRenderLoop({
  scene,
  camera,
  renderer,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
}): void {
  const draw = () => {
    renderer.render(scene, camera);
    window.requestAnimationFrame(draw);
  };
  window.requestAnimationFrame(draw);
}

function createThreeCameraOverlayLines({
  cameraVisualizations,
}: {
  cameraVisualizations: CameraVisualizationPayload[];
}): THREE.Group {
  const overlay = new THREE.Group();
  overlay.userData.cameraCount = cameraVisualizations.length;
  overlay.userData.lineCount = 0;
  overlay.renderOrder = 999;
  for (const cameraVisualization of cameraVisualizations) {
    overlay.add(createThreeCameraCenter({ cameraVisualization }));
    for (const line of [
      ...cameraVisualization.axes,
      ...cameraVisualization.frustum_lines,
    ]) {
      overlay.add(createThreeCameraOverlayLine({ line }));
      overlay.userData.lineCount += 1;
    }
  }
  if (cameraVisualizations.length > 0) {
    overlay.userData.firstAxisLength = cameraVisualizationLineLength({
      line: cameraVisualizations[0].axes[0],
    });
    overlay.userData.firstFrustumLength = cameraVisualizationLineLength({
      line: cameraVisualizations[0].frustum_lines[0],
    });
  }
  return overlay;
}

function createThreeCameraCenter({
  cameraVisualization,
}: {
  cameraVisualization: CameraVisualizationPayload;
}): THREE.Points {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute(cameraVisualization.center, 3),
  );
  const material = new THREE.PointsMaterial({
    color: new THREE.Color(...cameraVisualization.center_color),
    depthTest: false,
    depthWrite: false,
    size: 0.03,
  });
  const center = new THREE.Points(geometry, material);
  center.renderOrder = 999;
  return center;
}

function createThreeCameraOverlayLine({
  line,
}: {
  line: CameraVisualizationLinePayload;
}): THREE.Line {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([...line.start, ...line.end], 3),
  );
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(...line.color),
    depthTest: false,
    depthWrite: false,
  });
  const overlayLine = new THREE.Line(geometry, material);
  overlayLine.renderOrder = 999;
  return overlayLine;
}

function validateCameraVisualizationPayloads({
  value,
}: {
  value: unknown;
}): CameraVisualizationPayload[] {
  if (!Array.isArray(value)) {
    throw new Error("camera visualization payload must be an array");
  }
  const cameraVisualizations = value.map((cameraVisualization, cameraIndex) =>
    validateCameraVisualizationPayload({
      value: cameraVisualization,
      cameraIndex,
    }),
  );
  assertCameraVisualizationPayloadShape({ cameraVisualizations });
  return cameraVisualizations;
}

function validateCameraVisualizationPayload({
  value,
  cameraIndex,
}: {
  value: unknown;
  cameraIndex: number;
}): CameraVisualizationPayload {
  if (!isRecord(value)) {
    throw new Error(`camera visualization entry must be an object: ${cameraIndex}`);
  }
  return {
    center: validateCameraVisualizationVector({
      value: value.center,
      label: `camera ${cameraIndex} center`,
    }),
    center_color: validateCameraVisualizationVector({
      value: value.center_color,
      label: `camera ${cameraIndex} center_color`,
    }),
    axes: validateCameraVisualizationLines({
      value: value.axes,
      label: `camera ${cameraIndex} axes`,
    }),
    frustum_lines: validateCameraVisualizationLines({
      value: value.frustum_lines,
      label: `camera ${cameraIndex} frustum_lines`,
    }),
  };
}

function validateCameraVisualizationLines({
  value,
  label,
}: {
  value: unknown;
  label: string;
}): CameraVisualizationLinePayload[] {
  if (!Array.isArray(value)) {
    throw new Error(`${label} must be an array`);
  }
  return value.map((lineValue, lineIndex) =>
    validateCameraVisualizationLine({
      value: lineValue,
      label: `${label} line ${lineIndex}`,
    }),
  );
}

function validateCameraVisualizationLine({
  value,
  label,
}: {
  value: unknown;
  label: string;
}): CameraVisualizationLinePayload {
  if (!isRecord(value)) {
    throw new Error(`${label} must be an object`);
  }
  return {
    start: validateCameraVisualizationVector({
      value: value.start,
      label: `${label} start`,
    }),
    end: validateCameraVisualizationVector({
      value: value.end,
      label: `${label} end`,
    }),
    color: validateCameraVisualizationVector({
      value: value.color,
      label: `${label} color`,
    }),
  };
}

function validateCameraVisualizationVector({
  value,
  label,
}: {
  value: unknown;
  label: string;
}): CameraVisualizationVectorPayload {
  if (
    !Array.isArray(value) ||
    value.length !== 3 ||
    value.some((entry) => typeof entry !== "number" || !Number.isFinite(entry))
  ) {
    throw new Error(`${label} must be a finite 3-vector`);
  }
  return [value[0], value[1], value[2]];
}

function assertCameraVisualizationPayloadShape({
  cameraVisualizations,
}: {
  cameraVisualizations: CameraVisualizationPayload[];
}): void {
  for (let index = 0; index < cameraVisualizations.length; index += 1) {
    const cameraVisualization = cameraVisualizations[index];
    if (cameraVisualization.axes.length !== 3) {
      throw new Error(`camera ${index} must contain three axes`);
    }
    if (cameraVisualization.frustum_lines.length !== 8) {
      throw new Error(`camera ${index} must contain eight frustum lines`);
    }
  }
}

function cameraVisualizationLineLength({
  line,
}: {
  line: CameraVisualizationLinePayload;
}): number {
  const dx = line.end[0] - line.start[0];
  const dy = line.end[1] - line.start[1];
  const dz = line.end[2] - line.start[2];
  return Math.sqrt(dx * dx + dy * dy + dz * dz);
}

function readCameraStateFromContainer({
  container,
}: {
  container: HTMLElement;
}): CameraState | null {
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

function applyCameraState({
  camera,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  cameraState: CameraState;
}): boolean {
  if (applyThreePointCameraState({ camera, cameraState })) {
    return true;
  }
  return applyGaussianTrackballCameraState({ camera, cameraState });
}

function applyThreePointCameraState({
  camera,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  cameraState: CameraState;
}): boolean {
  if (cameraState.convention !== "three_pointcloud") {
    return false;
  }
  const position = readFiniteVector(cameraState.extrinsics.position);
  const quaternion = readFiniteQuaternion(cameraState.extrinsics.quaternion);
  const up = readFiniteVector(cameraState.extrinsics.up);
  if (position === null || quaternion === null || up === null) {
    return false;
  }
  camera.position.copy(vectorFromRecord({ vector: position }));
  camera.quaternion.copy(quaternionFromRecord({ quaternion }));
  camera.up.copy(vectorFromRecord({ vector: up }));
  applyPerspectiveIntrinsics({ camera, cameraState });
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return true;
}

function applyGaussianTrackballCameraState({
  camera,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  cameraState: CameraState;
}): boolean {
  if (cameraState.convention !== "trackball_gaussian_splatting") {
    return false;
  }
  const position = readFiniteVector(cameraState.extrinsics.position);
  const target = readFiniteVector(cameraState.extrinsics.target);
  const up = readFiniteVector(cameraState.extrinsics.up);
  if (position === null || target === null || up === null) {
    return false;
  }
  camera.position.copy(vectorFromRecord({ vector: position }));
  camera.up.copy(vectorFromRecord({ vector: up }));
  camera.lookAt(vectorFromRecord({ vector: target }));
  applyPerspectiveIntrinsics({ camera, cameraState });
  camera.updateProjectionMatrix();
  camera.updateMatrixWorld(true);
  return true;
}

function applyPerspectiveIntrinsics({
  camera,
  cameraState,
}: {
  camera: THREE.PerspectiveCamera;
  cameraState: CameraState;
}): void {
  const fov = cameraState.intrinsics.fov;
  const aspect = cameraState.intrinsics.aspect;
  const near = cameraState.intrinsics.near;
  const far = cameraState.intrinsics.far;
  if (typeof fov === "number" && Number.isFinite(fov)) {
    camera.fov = fov;
  }
  if (typeof aspect === "number" && Number.isFinite(aspect)) {
    camera.aspect = aspect;
  }
  if (typeof near === "number" && Number.isFinite(near)) {
    camera.near = near;
  }
  if (typeof far === "number" && Number.isFinite(far)) {
    camera.far = far;
  }
}

function readFiniteVector(value: unknown): FiniteVector | null {
  if (!isRecord(value)) {
    return null;
  }
  if (
    typeof value.x !== "number" ||
    typeof value.y !== "number" ||
    typeof value.z !== "number"
  ) {
    return null;
  }
  if (
    !Number.isFinite(value.x) ||
    !Number.isFinite(value.y) ||
    !Number.isFinite(value.z)
  ) {
    return null;
  }
  return {
    x: value.x,
    y: value.y,
    z: value.z,
  };
}

function readFiniteQuaternion(value: unknown): FiniteQuaternion | null {
  const vector = readFiniteVector(value);
  if (vector === null || !isRecord(value)) {
    return null;
  }
  if (typeof value.w !== "number" || !Number.isFinite(value.w)) {
    return null;
  }
  return {
    ...vector,
    w: value.w,
  };
}

function vectorFromRecord({
  vector,
}: {
  vector: FiniteVector;
}): THREE.Vector3 {
  return new THREE.Vector3(vector.x, vector.y, vector.z);
}

function quaternionFromRecord({
  quaternion,
}: {
  quaternion: FiniteQuaternion;
}): THREE.Quaternion {
  return new THREE.Quaternion(
    quaternion.x,
    quaternion.y,
    quaternion.z,
    quaternion.w,
  );
}

function isCameraState(value: unknown): value is CameraState {
  return (
    isRecord(value) &&
    isRecord(value.intrinsics) &&
    isRecord(value.extrinsics) &&
    typeof value.convention === "string"
  );
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
