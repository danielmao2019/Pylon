import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";

export interface PixelDisplayResponse extends DisplayResponse {}

export interface ColorImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "color_image";
}

export interface DepthImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "depth_image";
}

export interface EdgeImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "edge_image";
}

export interface NormalImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "normal_image";
}

export interface SegmentationImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "segmentation_image";
  original_overlay_url?: string | null;
}

export interface InstanceSurrogateImageDisplayResponse extends PixelDisplayResponse {
  display_kind: "instance_surrogate_image";
  original_overlay_url?: string | null;
}
