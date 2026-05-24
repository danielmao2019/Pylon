import type { PlaceholderDisplayResponse } from "./types/display_response";

export function renderPlaceholderDisplay(
  { displayResponse }: { displayResponse: PlaceholderDisplayResponse },
): HTMLElement {
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder-surface";
  placeholder.textContent = displayResponse.message;
  return placeholder;
}
