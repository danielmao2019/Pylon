import * as GaussianSplats3D from "@mkkellogg/gaussian-splats-3d";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import { createThreeDisplayContainer } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
import type { LeafVNode, VNode } from "web/reconcile/reconcile";
import type { GaussianDisplayResponse } from "./types/display_response";

// Delegates rendering to the external Gaussian-splat package; the package owns
// URL loading, scene assembly, camera controls, and the render loop. The
// wrapper only supplies the host container and the artifact url.
export function renderGaussiansDisplay({
  displayResponse,
}: {
  displayResponse: GaussianDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `gaussians:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const container = createThreeDisplayContainer({ pointerEventsSuppressed: false });
      if (displayResponse.url === null) {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder-surface";
        placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
        container.append(placeholder);
        return container;
      }
      const viewer = new GaussianSplats3D.Viewer({
        rootElement: container,
        selfDrivenMode: true,
        useBuiltInControls: true,
        sphericalHarmonicsDegree: 3,
        gpuAcceleratedSort: false,
        sharedMemoryForWorkers: false,
        dynamicScene: false,
      });
      void viewer
        .addSplatScene(displayResponse.url, {
          showLoadingUI: false,
          progressiveLoad: false,
        })
        .then(() => viewer.start())
        .catch((error) => {
          const message = error instanceof Error ? error.message : String(error);
          console.error("Failed to load Gaussian splat artifact:", error);
          const status = document.createElement("div");
          status.className = "placeholder-surface";
          status.textContent = `Failed to load Gaussian splats: ${message}`;
          container.append(status);
        });
      return container;
    },
  };
  return leaf;
}
