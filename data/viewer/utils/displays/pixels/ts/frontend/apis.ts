import type { VNode } from "web/reconcile/reconcile";
import { renderPixelsDisplay } from "./core_pixels_display";
import {
  registerRasterLayerRenderer,
  type RasterLayerRenderer,
} from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
import type {
  ColorImageDisplayResponse,
  DepthImageDisplayResponse,
  EdgeImageDisplayResponse,
  InstanceSurrogateImageDisplayResponse,
  NormalImageDisplayResponse,
  SegmentationImageDisplayResponse,
} from "./types/display_response";

// color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
const DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear";
// depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
const DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest";
// edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
const DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest";
// normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
const DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest";
// segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
const DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest";
// instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
const DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest";

export function renderColorImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_COLOR_IMAGE_INTERPOLATION,
}: {
  displayResponse: ColorImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

export function renderDepthImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_DEPTH_IMAGE_INTERPOLATION,
}: {
  displayResponse: DepthImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

export function renderEdgeImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_EDGE_IMAGE_INTERPOLATION,
}: {
  displayResponse: EdgeImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

export function renderNormalImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_NORMAL_IMAGE_INTERPOLATION,
}: {
  displayResponse: NormalImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

export function renderSegmentationImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION,
}: {
  displayResponse: SegmentationImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

export function renderInstanceSurrogateImageDisplay({
  displayResponse,
  imageInterpolation = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION,
}: {
  displayResponse: InstanceSurrogateImageDisplayResponse;
  imageInterpolation?: string;
}): VNode {
  return renderPixelsDisplay({ displayResponse, imageInterpolation });
}

// Module-load self-registration of the raster color-image layer renderer. The
// registry erases the layer's display response to the base DisplayResponse, so
// the color-image part-B is registered through the registry's renderer type at
// the erasure boundary.
registerRasterLayerRenderer({
  displayKind: "color_image",
  layerRenderer: renderColorImageDisplay as RasterLayerRenderer,
});
