import { renderPixelsDisplay } from "./core_pixels_display";
import type {
  ColorImageDisplayResponse,
  DepthImageDisplayResponse,
  EdgeImageDisplayResponse,
  InstanceSurrogateImageDisplayResponse,
  NormalImageDisplayResponse,
  SegmentationImageDisplayResponse,
} from "./types/display_response";

export function renderColorImageDisplay({
  displayResponse,
}: {
  displayResponse: ColorImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}

export function renderDepthImageDisplay({
  displayResponse,
}: {
  displayResponse: DepthImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}

export function renderEdgeImageDisplay({
  displayResponse,
}: {
  displayResponse: EdgeImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}

export function renderNormalImageDisplay({
  displayResponse,
}: {
  displayResponse: NormalImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}

export function renderSegmentationImageDisplay({
  displayResponse,
}: {
  displayResponse: SegmentationImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}

export function renderInstanceSurrogateImageDisplay({
  displayResponse,
}: {
  displayResponse: InstanceSurrogateImageDisplayResponse;
}): HTMLElement {
  return renderPixelsDisplay({ displayResponse });
}
