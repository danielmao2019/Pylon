import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import { renderColorPCDisplay } from "data/viewer/utils/atomic_displays/points/ts/frontend/apis";
import type { ColorPCDisplayResponse } from "data/viewer/utils/atomic_displays/points/ts/frontend/types/display_response";
import type { LeafVNode, VNode } from "web/reconcile/reconcile";
import type { SceneGraphDisplayResponse } from "./types/display_response";

export function renderSceneGraphDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: SceneGraphDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  if (displayResponse.url === null) {
    const leaf: LeafVNode = {
      kind: "leaf",
      key: `scene_graph:${displayResponse.slot_id}`,
      props: {},
      render: () => {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder-surface";
        placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
        return placeholder;
      },
    };
    return leaf;
  }

  return renderColorPCDisplay({
    displayResponse: {
      ...displayResponse,
      display_kind: "color_pc",
    } as ColorPCDisplayResponse,
    initialCameraState,
  });
}
