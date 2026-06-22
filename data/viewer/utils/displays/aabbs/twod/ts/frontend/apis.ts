import type { LeafVNode } from "web/reconcile/reconcile";
import {
  registerRasterLayerRenderer,
  type RasterLayerRenderer,
} from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
import type { Aabb2dDisplayResponse } from "./types/display_response";

const SVG_NS = "http://www.w3.org/2000/svg";

// Box stroke color and width, and score-label style for the 2D box overlay.
const AABB_2D_BOX_STROKE = "#4da6ff";
const AABB_2D_BOX_STROKE_WIDTH = "2";
const AABB_2D_LABEL_FONT_SIZE = "14";

// Renders the inline 2D axis-aligned boxes and their optional per-box score
// labels as a raster overlay layer over an image frame: a LeafVNode whose
// absolutely-positioned full-bleed SVG draws each box as a <rect> with its score
// <text>, so a consumer composes it over an image as a raster aux layer. The SVG
// is full-bleed with preserveAspectRatio="none" and its <rect>s carry base-image
// pixel coordinates; the layered container owns the viewBox, setting it to the
// shared raster frustum on the base image's load.
//
// Args:
//   displayResponse: the 2D box overlay response carrying the inline boxes (each
//     [min_x, min_y, max_x, max_y], image-pixel coordinates) and the optional
//     per-box scores.
//
// Returns:
//   A LeafVNode whose render() mounts the raster box overlay.
export function renderAabb2dDisplay({
  displayResponse,
}: {
  displayResponse: Aabb2dDisplayResponse;
}): LeafVNode {
  const leaf: LeafVNode = {
    kind: "leaf",
    key: displayResponse.url ?? `aabb_2d:${displayResponse.slot_id}`,
    props: {},
    render: () => _buildBoxesOverlay({ displayResponse }),
  };
  return leaf;
}

// Build the box overlay element: an absolutely-positioned full-bleed div hosting
// a full-bleed SVG (preserveAspectRatio="none"), drawing each box as a stroke
// <rect> and its score (when present) as a <text> just above the box, in
// base-image pixel coordinates. The layered container owns the SVG viewBox.
//
// Args:
//   displayResponse: the 2D box overlay response carrying the boxes and optional
//     per-box scores.
//
// Returns:
//   The overlay div hosting the box/score SVG.
function _buildBoxesOverlay({
  displayResponse,
}: {
  displayResponse: Aabb2dDisplayResponse;
}): HTMLElement {
  const boxes = displayResponse.aabbs;
  const scores = displayResponse.scores;
  const overlay = document.createElement("div");
  overlay.style.position = "absolute";
  overlay.style.inset = "0";
  overlay.style.pointerEvents = "none";
  const svg = document.createElementNS(SVG_NS, "svg");
  svg.setAttribute("preserveAspectRatio", "none");
  svg.style.position = "absolute";
  svg.style.inset = "0";
  svg.style.width = "100%";
  svg.style.height = "100%";
  svg.style.pointerEvents = "none";
  overlay.append(svg);
  for (let boxIndex = 0; boxIndex < boxes.length; boxIndex += 1) {
    const [x1, y1, x2, y2] = boxes[boxIndex];
    const rect = document.createElementNS(SVG_NS, "rect");
    rect.setAttribute("x", String(x1));
    rect.setAttribute("y", String(y1));
    rect.setAttribute("width", String(x2 - x1));
    rect.setAttribute("height", String(y2 - y1));
    rect.setAttribute("fill", "none");
    rect.setAttribute("stroke", AABB_2D_BOX_STROKE);
    rect.setAttribute("stroke-width", AABB_2D_BOX_STROKE_WIDTH);
    svg.append(rect);
    if (scores !== null) {
      const label = document.createElementNS(SVG_NS, "text");
      label.setAttribute("x", String(x1));
      label.setAttribute("y", String(Math.max(y1 - 3, 10)));
      label.setAttribute("fill", AABB_2D_BOX_STROKE);
      label.setAttribute("font-size", AABB_2D_LABEL_FONT_SIZE);
      label.setAttribute("font-family", "monospace");
      label.textContent = scores[boxIndex].toFixed(2);
      svg.append(label);
    }
  }
  return overlay;
}

// Module-load self-registration of the raster aabb-2d layer renderer. The
// registry erases the layer's display response to the base DisplayResponse, so
// the aabb-2d part-B is registered through the registry's renderer type at the
// erasure boundary.
registerRasterLayerRenderer({
  displayKind: "aabb_2d",
  layerRenderer: renderAabb2dDisplay as RasterLayerRenderer,
});
