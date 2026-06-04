// Runtime smoke harness: builds a jsdom DOM and asserts the four reconcileInto behaviors.

import { JSDOM } from "jsdom";
import { createElementVNode, reconcileInto, type VNode } from "./reconcile";

// Set up jsdom globals and exercise mount / patch / replace flows with assertions.
function _runSmoke(): void {
  const dom: JSDOM = new JSDOM("<!doctype html><html><body></body></html>");
  (globalThis as unknown as { document: Document }).document = dom.window.document;
  (globalThis as unknown as { HTMLElement: typeof HTMLElement }).HTMLElement = dom.window.HTMLElement;

  const root: HTMLElement = dom.window.document.createElement("div");
  dom.window.document.body.appendChild(root);

  const handler1: (event: Event) => void = (_event: Event): void => {};
  const treeA: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      { kind: "element", tag: "h1", key: "title", props: { text: "Hello" }, children: [] },
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

  const sectionA: HTMLElement = root.firstElementChild as HTMLElement;
  if (sectionA === null || sectionA.tagName.toLowerCase() !== "section") {
    throw new Error("a) initial mount did not produce <section> child");
  }
  const h1A: HTMLElement = sectionA.children[0] as HTMLElement;
  const buttonA: HTMLElement = sectionA.children[1] as HTMLElement;
  if (h1A.textContent !== "Hello") {
    throw new Error("a) initial mount did not set h1 textContent");
  }
  if (sectionA.className !== "card" || sectionA.id !== "main") {
    throw new Error("a) initial mount did not patch className/id");
  }

  const treeB: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      { kind: "element", tag: "h1", key: "title", props: { text: "Hello, world" }, children: [] },
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

  const sectionB: HTMLElement = root.firstElementChild as HTMLElement;
  if (sectionB !== sectionA) {
    throw new Error("b) section identity not preserved across reconcileInto");
  }
  const h1B: HTMLElement = sectionB.children[0] as HTMLElement;
  if (h1B !== h1A) {
    throw new Error("c) h1 identity not preserved; node was recreated instead of patched");
  }
  if (h1B.textContent !== "Hello, world") {
    throw new Error("c) h1 textContent not updated to new text prop");
  }

  const treeC: VNode = {
    kind: "element",
    tag: "section",
    key: "root",
    props: { className: "card", id: "main" },
    children: [
      { kind: "element", tag: "h1", key: "title", props: { text: "Hello, world" }, children: [] },
      { kind: "element", tag: "button", key: "btn-v2", props: { text: "Click v2" }, children: [] },
    ],
  };
  reconcileInto({ root, virtualTree: treeC });

  const sectionC: HTMLElement = root.firstElementChild as HTMLElement;
  const buttonC: HTMLElement = sectionC.children[1] as HTMLElement;
  if (buttonC === buttonA) {
    throw new Error("d) rekeyed button retained identity; expected unmount+remount");
  }
  if (buttonC.textContent !== "Click v2") {
    throw new Error("d) replacement button did not receive new text prop");
  }
  if (buttonA.parentNode !== null) {
    throw new Error("d) old keyed button was not detached after replacement");
  }

  let leafRenderCount: number = 0;
  const leafFactory: () => HTMLElement = (): HTMLElement => {
    leafRenderCount += 1;
    const el: HTMLElement = dom.window.document.createElement("canvas");
    el.setAttribute("data-leaf", "yes");
    return el;
  };
  const leafTree1: VNode = {
    kind: "element",
    tag: "div",
    key: "host",
    props: {},
    children: [{ kind: "leaf", key: "viewer", props: { className: "viewer" }, render: leafFactory }],
  };
  const leafRoot: HTMLElement = dom.window.document.createElement("div");
  reconcileInto({ root: leafRoot, virtualTree: leafTree1 });
  const leafTree2: VNode = {
    kind: "element",
    tag: "div",
    key: "host",
    props: {},
    children: [
      { kind: "leaf", key: "viewer", props: { className: "viewer active" }, render: leafFactory },
    ],
  };
  reconcileInto({ root: leafRoot, virtualTree: leafTree2 });
  if (leafRenderCount !== 1) {
    throw new Error(`e) leaf render should run once per key; got ${leafRenderCount}`);
  }
  const leafEl: HTMLElement = (leafRoot.firstElementChild as HTMLElement).firstElementChild as HTMLElement;
  if (leafEl.className !== "viewer active") {
    throw new Error("e) leaf props were not patched on rerender");
  }

  // f) createElementVNode: key lifting, prop bag, string-child normalization, and that the built tree reconciles.
  const builtHeading = createElementVNode("h2", {}, ["Heading"]);
  const builtTree = createElementVNode("article", { key: "built", className: "built" }, [
    builtHeading,
    createElementVNode("p", { id: "p" }, ["paragraph"]),
  ]);
  if (builtTree.tag !== "article") {
    throw new Error("f) createElementVNode did not build the expected element");
  }
  if (builtTree.key !== "built" || "key" in builtTree.props) {
    throw new Error("f) createElementVNode did not lift key out of the prop bag");
  }
  if (builtTree.props.className !== "built") {
    throw new Error("f) createElementVNode dropped a non-key prop");
  }
  if (builtHeading.children.length !== 1 || builtHeading.children[0].kind !== "leaf") {
    throw new Error("f) createElementVNode did not normalize a string child to a text leaf");
  }
  const builtRoot: HTMLElement = dom.window.document.createElement("div");
  reconcileInto({ root: builtRoot, virtualTree: builtTree });
  const builtArticle: HTMLElement = builtRoot.firstElementChild as HTMLElement;
  if (builtArticle === null || builtArticle.tagName.toLowerCase() !== "article" || builtArticle.className !== "built") {
    throw new Error("f) createElementVNode tree did not reconcile to the expected element");
  }
  if (builtArticle.textContent === null || !builtArticle.textContent.includes("Heading")) {
    throw new Error("f) createElementVNode string child did not render its text");
  }

  console.log("smoke OK");
}

_runSmoke();
