import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";

// Raster overlay response: inline axis-aligned 2D boxes (each a 4-float box
// [min_x, min_y, max_x, max_y], image-pixel coordinates) with optional per-box
// scores, composed as an aux layer over an image.
export interface Aabb2dDisplayResponse extends DisplayResponse {
  display_kind: "aabb_2d";
  aabbs: number[][];
  scores: number[] | null;
}
