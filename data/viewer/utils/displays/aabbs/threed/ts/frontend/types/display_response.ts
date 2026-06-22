import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

// Spatial overlay response: inline axis-aligned 3D boxes (each a 6-float box
// [min_x, min_y, min_z, max_x, max_y, max_z], world coordinates) with optional
// per-box scores, composed as an aux layer over a point cloud.
export interface Aabb3dDisplayResponse extends DisplayResponse {
  display_kind: "aabb_3d";
  aabbs: number[][];
  scores: number[] | null;
}
