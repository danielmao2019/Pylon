import type { PixelDisplayResponse } from "./types/display_response";

export function renderPixelsDisplay({
  displayResponse,
}: {
  displayResponse: PixelDisplayResponse;
}): HTMLElement {
  if (displayResponse.url === null) {
    const placeholder = document.createElement("div");
    placeholder.className = "placeholder-surface";
    placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
    return placeholder;
  }

  const image = document.createElement("img");
  image.className = "artifact-image";
  image.src = displayResponse.url;
  image.alt = displayResponse.title;
  return image;
}
