import { renderMeshDisplay } from "./core_mesh_display";
import type {
  ColorMeshDisplayResponse,
  SegmentationMeshDisplayResponse,
} from "./types/display_response";

export function renderColorMeshDisplay({
  displayResponse,
}: {
  displayResponse: ColorMeshDisplayResponse;
}): HTMLElement {
  return renderMeshDisplay({ displayResponse });
}

export function renderSegmentationMeshDisplay({
  displayResponse,
}: {
  displayResponse: SegmentationMeshDisplayResponse;
}): HTMLElement {
  return renderMeshDisplay({ displayResponse });
}
