import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { VNode } from "web/reconcile/reconcile";
import { renderPointsDisplay } from "./core_points_display";
import type {
  ColorPCDisplayResponse,
  SegmentationPCDisplayResponse,
} from "./types/display_response";

export function renderColorPCDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: ColorPCDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderPointsDisplay({ displayResponse, initialCameraState });
}

export function renderSegmentationPCDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: SegmentationPCDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderPointsDisplay({ displayResponse, initialCameraState });
}
