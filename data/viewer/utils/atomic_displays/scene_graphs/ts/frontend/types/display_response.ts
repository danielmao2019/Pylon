import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";

export interface SceneGraphDisplayResponse extends DisplayResponse {
  display_kind: "scene_graph";
  original_overlay_url?: string | null;
}
