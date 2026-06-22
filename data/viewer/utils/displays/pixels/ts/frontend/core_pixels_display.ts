import type { LeafVNode } from "web/reconcile/reconcile";
import type { PixelDisplayResponse } from "./types/display_response";

// Renders a self-contained pixel-image display element from the resolved
// interpolation choice; modality-agnostic. The interpolation choice maps onto
// the browser's `image-rendering` policy so caller intent (e.g. preserve crisp
// class-id boundaries vs. smooth natural-image content) is honored on upscale:
// "nearest" pins crisp nearest-neighbor sampling, any other choice smooths.
export function renderPixelsDisplay({
  displayResponse,
  imageInterpolation,
}: {
  displayResponse: PixelDisplayResponse;
  imageInterpolation: string;
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `pixels:${displayResponse.slot_id}`,
    props: {},
    render: () => {
      if (displayResponse.url === null) {
        const placeholder = document.createElement("div");
        placeholder.className = "placeholder-surface";
        placeholder.textContent =
          "Placeholder for a benchmark result that is not materialized yet.";
        return placeholder;
      }
      const image = document.createElement("img");
      image.className = "artifact-image";
      image.src = displayResponse.url;
      image.alt = displayResponse.title;
      image.style.imageRendering = imageInterpolation === "nearest" ? "pixelated" : "auto";
      return image;
    },
  };
  return leaf;
}
