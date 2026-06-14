import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

export interface GaussianDisplayResponse extends DisplayResponse {}

export interface ColorGSDisplayResponse extends GaussianDisplayResponse {
  display_kind: "color_gs";
}

export interface SegmentationGSDisplayResponse extends GaussianDisplayResponse {
  display_kind: "segmentation_gs";
  original_overlay_url?: string | null;
}
