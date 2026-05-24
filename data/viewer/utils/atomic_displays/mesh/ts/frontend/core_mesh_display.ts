import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
import type { MeshDisplayResponse } from "./types/display_response";

export function renderMeshDisplay({
  displayResponse,
}: {
  displayResponse: MeshDisplayResponse;
}): HTMLElement {
  if (displayResponse.url === null) {
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder-surface";
    placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
    return placeholder;
  }

  const scene = createMeshScene({ displayResponse });
  createTrackballCameraControls({ targetElement: scene });
  return renderMeshScene({ scene });
}

function createMeshScene({
  displayResponse,
}: {
  displayResponse: MeshDisplayResponse;
}): HTMLIFrameElement {
  if (displayResponse.url === null) {
    throw new Error("mesh display response url is null");
  }
  const iframe = document.createElement("iframe");
  iframe.className = "artifact-frame";
  iframe.src = displayResponse.url;
  iframe.title = displayResponse.title;
  return iframe;
}

function renderMeshScene({ scene }: { scene: HTMLIFrameElement }): HTMLElement {
  return scene;
}
