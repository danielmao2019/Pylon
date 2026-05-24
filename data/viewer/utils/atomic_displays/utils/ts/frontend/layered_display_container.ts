import type { ElementVNode, VNode } from "web/reconcile/reconcile";

// Stack the provided child VNodes in given order into one layered-container ElementVNode.
//
// The first layer is the base; later layers are stacked on top in array order.
// Every layer is absolutely positioned to fill the container, so the stack
// composites visually while every layer keeps its own VNode identity.
export function renderLayeredDisplayContainer({
  layers,
  slotId,
}: {
  layers: readonly VNode[];
  slotId: string;
}): VNode {
  if (layers.length === 0) {
    throw new Error("layered display container requires at least one layer");
  }
  if (slotId.length === 0) {
    throw new Error("layered display container requires a non-empty slotId");
  }
  const children: VNode[] = layers.map((layer, index) =>
    _wrapLayer({ layer, slotId, index }),
  );
  const container: ElementVNode = {
    kind: "element",
    tag: "div",
    key: slotId,
    props: {
      className: "layered-display-container",
      style: { position: "relative", width: "100%", height: "100%" },
    },
    children,
  };
  return container;
}

// Wrap one layer VNode in an absolutely-positioned cell so the layers composite.
function _wrapLayer({
  layer,
  slotId,
  index,
}: {
  layer: VNode;
  slotId: string;
  index: number;
}): ElementVNode {
  return {
    kind: "element",
    tag: "div",
    key: `${slotId}/layer/${index}`,
    props: {
      className: "layered-display-container__layer",
      style: { position: "absolute", inset: "0", width: "100%", height: "100%" },
    },
    children: [layer],
  };
}
