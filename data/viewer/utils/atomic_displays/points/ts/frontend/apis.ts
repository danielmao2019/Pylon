import { renderPointsDisplay } from "./core_points_display";
import type {
  ColorPCDisplayResponse,
  SegmentationPCDisplayResponse,
} from "./types/display_response";

export function renderColorPCDisplay({
  displayResponse,
}: {
  displayResponse: ColorPCDisplayResponse;
}): HTMLElement {
  return renderPointsDisplay({ displayResponse });
}

export function renderSegmentationPCDisplay({
  displayResponse,
}: {
  displayResponse: SegmentationPCDisplayResponse;
}): HTMLElement {
  return renderPointsDisplay({ displayResponse });
}
