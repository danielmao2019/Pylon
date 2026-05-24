import { renderColorPCDisplay } from "data/viewer/utils/atomic_displays/points/ts/frontend/apis";
import type { ColorPCDisplayResponse } from "data/viewer/utils/atomic_displays/points/ts/frontend/types/display_response";
import type { SceneGraphDisplayResponse } from "./types/display_response";

export function renderSceneGraphDisplay({
  displayResponse,
}: {
  displayResponse: SceneGraphDisplayResponse;
}): HTMLElement {
  return renderSceneGraphDisplayPanel({ displayResponse });
}

function renderSceneGraphDisplayPanel({
  displayResponse,
}: {
  displayResponse: SceneGraphDisplayResponse;
}): HTMLElement {
  if (displayResponse.url === null) {
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder-surface";
    placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
    return placeholder;
  }

  return renderColorPCDisplay({
    displayResponse: {
      ...displayResponse,
      display_kind: "color_pc",
    } as ColorPCDisplayResponse,
  });
}
