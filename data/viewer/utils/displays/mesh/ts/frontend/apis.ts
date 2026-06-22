import * as THREE from "three";
import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
import type { LeafVNode } from "web/reconcile/reconcile";
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
  meshColor,
  meshOpacity,
  meshSide,
}: {
  displayResponse: ColorMeshDisplayResponse;
  initialCameraState?: CameraState | null;
  meshColor?: string;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): LeafVNode {
  return renderMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide });
}

// Renders backend-colorized mesh display and legend derived from meta_info;
// per-element colors are already baked in by the backend's class-id -> rgb
// mapping, so no meshColor override is exposed here.
export function renderSegmentationMeshDisplay({
  displayResponse,
  initialCameraState = null,
  meshOpacity,
  meshSide,
}: {
  displayResponse: SegmentationMeshDisplayResponse;
  initialCameraState?: CameraState | null;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): LeafVNode {
  return renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide });
}

// Renders backend-colorized mesh display and continuous-palette legend derived
// from meta_info (scalar min/max); per-element colors are already baked in by
// the backend's scalar -> rgb mapping, so no meshColor override is exposed here.
export function renderHeatmapMeshDisplay({
  displayResponse,
  initialCameraState = null,
  meshOpacity,
  meshSide,
}: {
  displayResponse: HeatmapMeshDisplayResponse;
  initialCameraState?: CameraState | null;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): LeafVNode {
  return renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide });
}

// Renders the sparse heatmap mesh display and continuous-palette legend from
// meta_info (scalar min/max); per-element colors are already baked in by the
// backend's scalar -> rgb mapping, so no meshColor override is exposed here.
export function renderSparseHeatmapMeshDisplay({
  displayResponse,
  initialCameraState = null,
  meshOpacity,
  meshSide,
}: {
  displayResponse: SparseHeatmapMeshDisplayResponse;
  initialCameraState?: CameraState | null;
  meshOpacity?: number;
  meshSide?: THREE.Side;
}): LeafVNode {
  return renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide });
}
