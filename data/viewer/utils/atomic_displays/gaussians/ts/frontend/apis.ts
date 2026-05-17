import { renderGaussiansDisplay } from "./core_gaussians_display";
import type {
  ColorGSDisplayResponse,
  SegmentationGSDisplayResponse,
} from "./types/display_response";

export function renderColorGSDisplay({
  displayResponse,
}: {
  displayResponse: ColorGSDisplayResponse;
}): HTMLElement {
  return renderGaussiansDisplay({ displayResponse });
}

export function renderSegmentationGSDisplay({
  displayResponse,
}: {
  displayResponse: SegmentationGSDisplayResponse;
}): HTMLElement {
  return renderGaussiansDisplay({ displayResponse });
}
