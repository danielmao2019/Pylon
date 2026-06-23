import type { LeafVNode } from "web/reconcile/reconcile";
import type { TextDisplayResponse } from "./types/display_response";

export function renderTextDisplay({
  displayResponse,
}: {
  displayResponse: TextDisplayResponse;
}): LeafVNode {
  return {
    kind: "leaf",
    key: displayResponse.url ?? `text:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      const pre = document.createElement("pre");
      pre.className = "text-display";
      pre.textContent = displayResponse.text;
      return pre;
    },
  };
}
