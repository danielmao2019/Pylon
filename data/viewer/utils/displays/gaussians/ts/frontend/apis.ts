import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { LeafVNode } from "web/reconcile/reconcile";
import { renderGaussiansDisplay } from "./core_gaussians_display";
import type {
  ColorGSDisplayResponse,
  SegmentationGSDisplayResponse,
} from "./types/display_response";

export function renderColorGSDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: ColorGSDisplayResponse;
  initialCameraState?: CameraState | null;
}): LeafVNode {
  return renderGaussiansDisplay({ displayResponse, initialCameraState });
}

export function renderSegmentationGSDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: SegmentationGSDisplayResponse;
  initialCameraState?: CameraState | null;
}): LeafVNode {
  return renderGaussiansDisplay({ displayResponse, initialCameraState });
}
