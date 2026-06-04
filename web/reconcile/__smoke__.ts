// Type-level smoke test for reconcileInto exercising mount, patch-in-place, prop diff, and key-replacement.

import { createElementVNode, reconcileInto, type VNode } from "./reconcile";

// Build a small VNode tree and re-run reconcileInto to verify identity preservation behaviors.
function _smokeTrace(): void {
  const root: HTMLElement = document.createElement("div");

  const handler1: (event: Event) => void = (_event: Event): void => {};
  const treeA: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      {
        kind: "element",
        tag: "h1",
        key: "title",
        props: { text: "Hello" },
        children: [],
      },
      {
        kind: "element",
        tag: "button",
        key: "btn",
        props: { onclick: handler1, text: "Click" },
        children: [],
      },
    ],
  };
  reconcileInto({ root, virtualTree: treeA });

  const sectionAfterA: HTMLElement = root.firstChild as HTMLElement;

  const treeB: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      {
        kind: "element",
        tag: "h1",
        key: "title",
        props: { text: "Hello, world" },
        children: [],
      },
      {
        kind: "element",
        tag: "button",
        key: "btn",
        props: { onclick: handler1, text: "Click" },
        children: [],
      },
    ],
  };
  reconcileInto({ root, virtualTree: treeB });

  const sectionAfterB: HTMLElement = root.firstChild as HTMLElement;
  if (sectionAfterA !== sectionAfterB) {
    throw new Error("identity not preserved across reconcileInto calls with same key");
  }

  const treeC: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      {
        kind: "element",
        tag: "h1",
        key: "title",
        props: { text: "Hello, world" },
        children: [],
      },
      {
        kind: "element",
        tag: "button",
        key: "btn-v2",
        props: { text: "Click v2" },
        children: [],
      },
    ],
  };
  reconcileInto({ root, virtualTree: treeC });

  // createElementVNode: build the tree with the constructor instead of literals — type-checks its signature and that it yields a VNode reconcileInto accepts.
  const built: VNode = createElementVNode("article", { key: "built", className: "c" }, [
    createElementVNode("h2", {}, ["Heading"]),
    createElementVNode("p", { id: "p" }, ["paragraph"]),
  ]);
  reconcileInto({ root, virtualTree: built });
}

_smokeTrace;
