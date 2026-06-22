import type { LeafVNode } from "web/reconcile/reconcile";
import type { PlaceholderDisplayResponse } from "./types/display_response";

// Render the missing-result placeholder UI from PlaceholderDisplayResponse.message.
export function renderPlaceholderDisplay({
  displayResponse,
}: {
  displayResponse: PlaceholderDisplayResponse;
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `placeholder:${displayResponse.slot_id}`,
    props: {},
    render: () => _renderPlaceholderElement({ displayResponse }),
  };
  return leaf;
}

// Build the placeholder card element.
function _renderPlaceholderElement({
  displayResponse,
}: {
  displayResponse: PlaceholderDisplayResponse;
}): HTMLElement {
  const placeholder = document.createElement("div");
  placeholder.className = "placeholder-surface";
  placeholder.style.display = "flex";
  placeholder.style.alignItems = "center";
  placeholder.style.justifyContent = "center";
  placeholder.style.width = "100%";
  placeholder.style.height = "100%";
  placeholder.style.padding = "1rem";
  placeholder.style.color = "#888";
  placeholder.style.fontStyle = "italic";
  placeholder.textContent = displayResponse.message;
  return placeholder;
}
