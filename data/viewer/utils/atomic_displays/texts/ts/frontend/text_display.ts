import type { TextDisplayResponse } from "./types/display_response";

export function renderTextDisplay({
  displayResponse,
}: {
  displayResponse: TextDisplayResponse;
}): HTMLElement {
  const pre = document.createElement("pre");
  pre.className = "text-display";
  pre.textContent = displayResponse.text;
  return pre;
}
