import * as THREE from "three";
import type { LeafVNode, VNode } from "web/reconcile/reconcile";
import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import {
  createThreeDisplayContainer,
  createThreePerspectiveCamera,
  createThreeScene,
  createThreeWebGLRenderer,
  startThreeSceneRenderLoop,
} from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
import type { CameraDisplayResponse } from "./types/display_response";

// Last-resort default frustum line color, applied when the per-camera payload
// entry does not carry a color AND the caller does not supply frustumColor.
// Per-entry payload colors still take precedence over frustumColor. Lib-owned
// default, overridable.
export const DEFAULT_FRUSTUM_COLOR = "#888888";

// Transparent frustum overlay opacity applied when the caller does not supply
// frustumOpacity. Lib-owned default, overridable.
export const DEFAULT_FRUSTUM_OPACITY = 0.5;

// Marker size for the camera center point used when the caller does not supply
// centerMarkerSize. Lib-owned default, overridable.
export const DEFAULT_CENTER_MARKER_SIZE = 0.01;

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

type CamerasPayload = CameraVisualizationPayload[];

export function renderCameraDisplay({
  displayResponse,
  initialCameraState = null,
  frustumColor,
  frustumOpacity,
  centerMarkerSize,
}: {
  displayResponse: CameraDisplayResponse;
  initialCameraState?: CameraState | null;
  frustumColor?: string;
  frustumOpacity?: number;
  centerMarkerSize?: number;
}): VNode {
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
        frustumColor,
        frustumOpacity,
        centerMarkerSize,
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
  frustumColor,
  frustumOpacity,
  centerMarkerSize,
}: {
  displayResponse: CameraDisplayResponse;
  initialCameraState: CameraState | null;
  frustumColor?: string;
  frustumOpacity?: number;
  centerMarkerSize?: number;
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
  loadCamerasPayload({ displayResponse })
    .then((payload) =>
      scene.add(
        createThreeCameras({ payload, frustumColor, frustumOpacity, centerMarkerSize }),
      ),
    )
    .catch((error) => {
      const message = error instanceof Error ? error.message : String(error);
      container.replaceChildren(
        _renderCamerasStatus(`Failed to load camera visualization: ${message}`),
      );
    });
  return { container, scene, camera, renderer };
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
  frustumColor,
  frustumOpacity,
  centerMarkerSize,
}: {
  payload: CamerasPayload;
  frustumColor?: string;
  frustumOpacity?: number;
  centerMarkerSize?: number;
}): THREE.Object3D {
  const effectiveCenterMarkerSize = centerMarkerSize ?? DEFAULT_CENTER_MARKER_SIZE;
  const effectiveFrustumOpacity = frustumOpacity ?? DEFAULT_FRUSTUM_OPACITY;
  const overlay = new THREE.Group();
  overlay.userData.cameraCount = payload.length;
  overlay.userData.lineCount = 0;
  overlay.renderOrder = 999;
  for (const cameraVisualization of payload) {
    overlay.add(
      createThreeCameraCenter({
        cameraVisualization,
        centerMarkerSize: effectiveCenterMarkerSize,
      }),
    );
    for (const line of cameraVisualization.axes) {
      overlay.add(createThreeCameraOverlayLine({ line, frustumColor, frustumOpacity: effectiveFrustumOpacity }));
      overlay.userData.lineCount += 1;
    }
    for (const line of cameraVisualization.frustum_lines) {
      overlay.add(createThreeCameraOverlayLine({ line, frustumColor, frustumOpacity: effectiveFrustumOpacity }));
      overlay.userData.lineCount += 1;
    }
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
  centerMarkerSize,
}: {
  cameraVisualization: CameraVisualizationPayload;
  centerMarkerSize: number;
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
    size: centerMarkerSize,
  });
  const center = new THREE.Points(geometry, material);
  center.renderOrder = 999;
  return center;
}

function createThreeCameraOverlayLine({
  line,
  frustumColor,
  frustumOpacity,
}: {
  line: CameraVisualizationLinePayload;
  frustumColor?: string;
  frustumOpacity: number;
}): THREE.Line {
  // Per-entry payload colors take precedence over frustumColor, which in turn
  // takes precedence over the lib-owned DEFAULT_FRUSTUM_COLOR.
  let effectiveFrustumColor: THREE.Color;
  if (line.color !== undefined) {
    effectiveFrustumColor = new THREE.Color(...line.color);
  } else if (frustumColor !== undefined) {
    effectiveFrustumColor = new THREE.Color(frustumColor);
  } else {
    effectiveFrustumColor = new THREE.Color(DEFAULT_FRUSTUM_COLOR);
  }
  const geometry = new THREE.BufferGeometry();
  geometry.setAttribute(
    "position",
    new THREE.Float32BufferAttribute([...line.start, ...line.end], 3),
  );
  const material = new THREE.LineBasicMaterial({
    color: effectiveFrustumColor,
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
