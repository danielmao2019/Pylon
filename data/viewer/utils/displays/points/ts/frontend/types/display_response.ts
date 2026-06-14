import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

export type PointDisplayResponse = DisplayResponse;

export interface ColorPCDisplayResponse extends PointDisplayResponse {
  display_kind: "color_pc";
}

export interface SegmentationPCDisplayResponse extends PointDisplayResponse {
  display_kind: "segmentation_pc";
  original_overlay_url?: string | null;
}
