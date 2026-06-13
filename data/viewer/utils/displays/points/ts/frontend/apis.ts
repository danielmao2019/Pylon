import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { VNode } from "web/reconcile/reconcile";
import { renderPointsDisplay } from "./core_points_display";
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
}): VNode {
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
}): VNode {
  return renderPointsDisplay({ displayResponse, initialCameraState, pointSize });
}
