import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { LeafVNode } from "web/reconcile/reconcile";
import { renderPointsDisplay, createPointsObject } from "./core_points_display";
import { registerSpatialLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
import type {
  ColorPCDisplayResponse,
  SegmentationPCDisplayResponse,
} from "./types/display_response";

export function renderColorPCDisplay({
  displayResponse,
  initialCameraState = null,
  pointSize,
  pointColor,
}: {
  displayResponse: ColorPCDisplayResponse;
  initialCameraState?: CameraState | null;
  pointSize?: number;
  pointColor?: string;
}): LeafVNode {
  return renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor });
}

export function renderSegmentationPCDisplay({
  displayResponse,
  initialCameraState = null,
  pointSize,
}: {
  displayResponse: SegmentationPCDisplayResponse;
  initialCameraState?: CameraState | null;
  pointSize?: number;
}): LeafVNode {
  return renderPointsDisplay({ displayResponse, initialCameraState, pointSize });
}

// Module-load self-registration of the spatial color-pc layer renderer.
registerSpatialLayerRenderer({ displayKind: "color_pc", layerRenderer: createPointsObject });
