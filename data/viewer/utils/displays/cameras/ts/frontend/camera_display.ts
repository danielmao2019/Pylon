import * as THREE from "three";
import type { LeafVNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import {
  applyCameraStateToThreeCamera,
  createThreeDisplayContainer,
  createThreePerspectiveCamera,
  createThreeScene,
  createThreeWebGLRenderer,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
import type { CameraDisplayResponse } from "./types/display_response";

// Transparent frustum overlay opacity applied when the caller does not supply
// frustumOpacity. A dynamic render property (the per-frame hover dimming
// multiplies it), not a baked glyph style — glyph size + color are baked by
// camera_vis. Lib-owned default, overridable.
export const DEFAULT_FRUSTUM_OPACITY = 0.5;

// Fraction of a frame's base opacity applied to a NON-contributing frame when a
// per-frame opacity predicate dims it. A clearly-visible fade so contributing
// frames stand out against the dimmed rest. Lib-owned default.
export const NON_CONTRIBUTING_FRAME_OPACITY_RATIO = 0.12;

// Control surface a caller uses to drive per-frame opacity on a rendered cameras
// display: `setContributingFrames` re-fades every per-camera group so the frames
// whose 0-based `cameraIndex` satisfies `isContributing` keep their base opacity
// and the rest are dimmed to `NON_CONTRIBUTING_FRAME_OPACITY_RATIO` of base. The
// predicate is latched, so calls made before the async payload mounts still apply
// once the per-camera groups are added to the scene.
export interface CameraFrameOpacityControl {
  container: HTMLElement;
  setContributingFrames: (isContributing: (cameraIndex: number) => boolean) => void;
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
  center_size: number;
  axes: CameraVisualizationLinePayload[];
  frustum_lines: CameraVisualizationLinePayload[];
}

type CamerasPayload = CameraVisualizationPayload[];

export function renderCameraDisplay({
  displayResponse,
  initialCameraState = null,
  frustumOpacity,
  onFrameOpacityControl,
}: {
  displayResponse: CameraDisplayResponse;
  initialCameraState?: CameraState | null;
  frustumOpacity?: number;
  onFrameOpacityControl?: (control: CameraFrameOpacityControl) => void;
}): LeafVNode {
  if (Object.keys(displayResponse.meta_info).length !== 0) {
    throw new Error("camera display meta_info must be an empty object");
  }
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `cameras:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const { container, scene, camera, renderer } = createCamerasScene({
        displayResponse,
        initialCameraState,
        frustumOpacity,
        onFrameOpacityControl,
      });
      renderCamerasScene({ scene, camera, renderer });
      return container;
    },
  };
  return leaf;
}

function createCamerasScene({
  displayResponse,
  initialCameraState,
  frustumOpacity,
  onFrameOpacityControl,
}: {
  displayResponse: CameraDisplayResponse;
  initialCameraState: CameraState | null;
  frustumOpacity?: number;
  onFrameOpacityControl?: (control: CameraFrameOpacityControl) => void;
}): {
  container: HTMLDivElement;
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
} {
  const container = createThreeDisplayContainer({ pointerEventsSuppressed: true });
  const scene = createThreeScene();
  const camera = createThreePerspectiveCamera({ initialCameraState });
  const renderer = createThreeWebGLRenderer({ container });
  followSyncedCameraPose({ container, camera });

  // Per-frame opacity control. The cameras payload loads asynchronously, so the
  // latest predicate is latched and re-applied once `createThreeCameras` has been
  // added to the scene. Until then `overlay` is null and the predicate is held.
  let overlay: THREE.Object3D | null = null;
  let latestPredicate: ((cameraIndex: number) => boolean) | null = null;
  const setContributingFrames = (
    isContributing: (cameraIndex: number) => boolean,
  ): void => {
    latestPredicate = isContributing;
    if (overlay !== null) {
      _applyFrameOpacity({ overlay, isContributing });
    }
  };
  if (onFrameOpacityControl !== undefined) {
    onFrameOpacityControl({ container, setContributingFrames });
  }

  loadCamerasPayload({ displayResponse })
    .then((payload) => {
      overlay = createThreeCameras({ payload, frustumOpacity });
      scene.add(overlay);
      if (latestPredicate !== null) {
        _applyFrameOpacity({ overlay, isContributing: latestPredicate });
      }
    })
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      container.replaceChildren(
        _renderCamerasStatus(`Failed to load camera visualization: ${message}`),
      );
    });
  return { container, scene, camera, renderer };
}

// Apply a per-frame opacity predicate to a built cameras overlay: walk every
// per-camera group (tagged with `userData.cameraIndex`), and for every child
// material set the opacity to the material's captured base when the group's
// `cameraIndex` is contributing, or to `NON_CONTRIBUTING_FRAME_OPACITY_RATIO` of
// that base otherwise. The base opacity is captured on first touch under
// `material.userData.baseOpacity`, and `material.transparent` is forced true so
// PointsMaterial centers can fade alongside the line materials.
function _applyFrameOpacity({
  overlay,
  isContributing,
}: {
  overlay: THREE.Object3D;
  isContributing: (cameraIndex: number) => boolean;
}): void {
  for (const child of overlay.children) {
    const cameraIndex = child.userData.cameraIndex;
    if (typeof cameraIndex !== "number") {
      continue;
    }
    const contributing = isContributing(cameraIndex);
    child.traverse((object) => {
      const material = (object as THREE.Mesh).material;
      if (material === undefined || material === null || Array.isArray(material)) {
        return;
      }
      const typedMaterial = material as THREE.Material & { opacity: number };
      if (typeof typedMaterial.userData.baseOpacity !== "number") {
        typedMaterial.userData.baseOpacity = typedMaterial.opacity;
      }
      const baseOpacity = typedMaterial.userData.baseOpacity as number;
      typedMaterial.transparent = true;
      typedMaterial.opacity = contributing
        ? baseOpacity
        : baseOpacity * NON_CONTRIBUTING_FRAME_OPACITY_RATIO;
      typedMaterial.needsUpdate = true;
    });
  }
}

// Make the frustum camera follow the camera-sync pose mirrored onto the
// container as `data-camera-state`. Applies the current value (if present) once,
// then watches the attribute for later sync updates. Since the cameras display
// has no trackball controls, the synced state is applied straight to the camera.
// A null/absent dataset value is a no-op, so a standalone (non-synced) caller
// never sees `data-camera-state` and behaves exactly as before.
function followSyncedCameraPose({
  container,
  camera,
}: {
  container: HTMLDivElement;
  camera: THREE.PerspectiveCamera;
}): void {
  const applyDatasetCameraState = (): void => {
    const raw = container.dataset.cameraState;
    if (raw === undefined) {
      return;
    }
    try {
      applyCameraStateToThreeCamera({
        camera,
        cameraState: JSON.parse(raw) as CameraState,
      });
    } catch {
      // ignore unparseable dataset values
    }
  };
  applyDatasetCameraState();
  const observer = new MutationObserver(() => {
    applyDatasetCameraState();
  });
  observer.observe(container, {
    attributeFilter: ["data-camera-state"],
    attributes: true,
  });
}

async function loadCamerasPayload({
  displayResponse,
}: {
  displayResponse: CameraDisplayResponse;
}): Promise<CamerasPayload> {
  if (displayResponse.url === null) {
    throw new Error("camera display response url is null");
  }
  const response = await fetch(displayResponse.url);
  if (!response.ok) {
    throw new Error(`unable to load camera visualization: HTTP ${response.status}`);
  }
  return validateCameraVisualizationPayloads({ value: await response.json() });
}

function createThreeCameras({
  payload,
  frustumOpacity,
}: {
  payload: CamerasPayload;
  frustumOpacity?: number;
}): THREE.Object3D {
  const effectiveFrustumOpacity = frustumOpacity ?? DEFAULT_FRUSTUM_OPACITY;
  const overlay = new THREE.Group();
  overlay.userData.cameraCount = payload.length;
  overlay.userData.lineCount = 0;
  overlay.renderOrder = 999;
  // Each camera's center Points plus its axes and frustum lines are wrapped in a
  // per-camera group tagged with the camera's 0-based position in the payload, so
  // a per-frame opacity setter can fade whole frames by `cameraIndex` without
  // knowing what the camera index means.
  for (let cameraIndex = 0; cameraIndex < payload.length; cameraIndex += 1) {
    const cameraVisualization = payload[cameraIndex];
    const cameraGroup = new THREE.Group();
    cameraGroup.userData.cameraIndex = cameraIndex;
    cameraGroup.add(
      createThreeCameraCenter({
        cameraVisualization,
      }),
    );
    for (const line of cameraVisualization.axes) {
      cameraGroup.add(createThreeCameraOverlayLine({ line, frustumOpacity: effectiveFrustumOpacity }));
      overlay.userData.lineCount += 1;
    }
    for (const line of cameraVisualization.frustum_lines) {
      cameraGroup.add(createThreeCameraOverlayLine({ line, frustumOpacity: effectiveFrustumOpacity }));
      overlay.userData.lineCount += 1;
    }
    overlay.add(cameraGroup);
  }
  if (payload.length > 0) {
    overlay.userData.firstAxisLength = cameraVisualizationLineLength({
      line: payload[0].axes[0],
    });
    overlay.userData.firstFrustumLength = cameraVisualizationLineLength({
      line: payload[0].frustum_lines[0],
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
  // center_size is a world-unit size baked by camera_vis (from point_size), so
  // sizeAttenuation must be true for `size` to be interpreted in world space.
  const material = new THREE.PointsMaterial({
    color: new THREE.Color(...cameraVisualization.center_color),
    depthTest: false,
    depthWrite: false,
    size: cameraVisualization.center_size,
    sizeAttenuation: true,
  });
  const center = new THREE.Points(geometry, material);
  center.renderOrder = 999;
  return center;
}

function createThreeCameraOverlayLine({
  line,
  frustumOpacity,
}: {
  line: CameraVisualizationLinePayload;
  frustumOpacity: number;
}): THREE.Line {
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([...line.start, ...line.end], 3),
  );
  // The backend bakes a per-line color into every axes/frustum line.
  const material = new THREE.LineBasicMaterial({
    color: new THREE.Color(...line.color),
    depthTest: false,
    depthWrite: false,
    opacity: frustumOpacity,
    transparent: true,
  });
  const overlayLine = new THREE.Line(geometry, material);
  overlayLine.renderOrder = 999;
  return overlayLine;
}

function validateCameraVisualizationPayloads({
  value,
}: {
  value: unknown;
}): CamerasPayload {
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
    center_size: validateCameraVisualizationScalar({
      value: value.center_size,
      label: `camera ${cameraIndex} center_size`,
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

function validateCameraVisualizationScalar({
  value,
  label,
}: {
  value: unknown;
  label: string;
}): number {
  if (typeof value !== "number" || !Number.isFinite(value)) {
    throw new Error(`${label} must be a finite number`);
  }
  return value;
}

function assertCameraVisualizationPayloadShape({
  cameraVisualizations,
}: {
  cameraVisualizations: CamerasPayload;
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

function renderCamerasScene({
  scene,
  camera,
  renderer,
}: {
  scene: THREE.Scene;
  camera: THREE.PerspectiveCamera;
  renderer: THREE.WebGLRenderer;
}): void {
  startThreeSceneRenderLoop({ scene, camera, renderer, controls: null });
}

// Render an inline status card for a load-failure cameras display.
function _renderCamerasStatus(message: string): HTMLElement {
  const status = document.createElement("div");
  status.className = "camera-display-scene__status";
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

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}
