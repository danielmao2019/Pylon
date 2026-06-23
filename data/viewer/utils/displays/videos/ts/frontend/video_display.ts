import type { LeafVNode } from "web/reconcile/reconcile";
import type { VideoDisplayResponse } from "./types/display_response";

export function renderVideoDisplay({
  displayResponse,
}: {
  displayResponse: VideoDisplayResponse;
}): LeafVNode {
  return {
    kind: "leaf",
    key: displayResponse.url ?? `video:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      if (displayResponse.url === null) {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder-surface";
        placeholder.textContent = "Placeholder for a benchmark result that is not materialized yet.";
        return placeholder;
      }

      const video = document.createElement("video");
      video.className = "artifact-video";
      video.src = displayResponse.url;
      video.controls = true;
      video.muted = true;
      return video;
    },
  };
}
