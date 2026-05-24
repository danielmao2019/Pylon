import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
import type { VNode } from "web/reconcile/reconcile";
import { renderMeshDisplay } from "./core_mesh_display";
import type {
  ColorMeshDisplayResponse,
  HeatmapMeshDisplayResponse,
  SegmentationMeshDisplayResponse,
  SparseHeatmapMeshDisplayResponse,
} from "./types/display_response";

export function renderColorMeshDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: ColorMeshDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderMeshDisplay({ displayResponse, initialCameraState });
}

export function renderSegmentationMeshDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: SegmentationMeshDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderMeshDisplay({ displayResponse, initialCameraState });
}

export function renderHeatmapMeshDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: HeatmapMeshDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderMeshDisplay({ displayResponse, initialCameraState });
}

export function renderSparseHeatmapMeshDisplay({
  displayResponse,
  initialCameraState = null,
}: {
  displayResponse: SparseHeatmapMeshDisplayResponse;
  initialCameraState?: CameraState | null;
}): VNode {
  return renderMeshDisplay({ displayResponse, initialCameraState });
}
