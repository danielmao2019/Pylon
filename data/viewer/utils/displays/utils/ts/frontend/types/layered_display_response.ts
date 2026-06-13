import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

export interface LayeredDisplayResponse extends DisplayResponse {
  display_kind: "layered";
  base_display_response: DisplayResponse;
  aux_display_responses: DisplayResponse[];
  layer_class: "raster" | "spatial";
}
