import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";

export interface MeshDisplayResponse extends DisplayResponse {}

export interface ColorMeshDisplayResponse extends MeshDisplayResponse {
  display_kind: "color_mesh";
}

export interface SegmentationMeshDisplayResponse extends MeshDisplayResponse {
  display_kind: "segmentation_mesh";
  original_overlay_url?: string | null;
}

export interface HeatmapMeshDisplayResponse extends MeshDisplayResponse {
  display_kind: "heatmap_mesh";
}

export interface SparseHeatmapMeshDisplayResponse extends MeshDisplayResponse {
  display_kind: "sparse_heatmap_mesh";
}
