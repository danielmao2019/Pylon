// Identity-preserving DOM patch driver consumed by any TS SPA's route render step.

export type VNode = ElementVNode | LeafVNode;

export interface ElementVNode {
  kind: "element";
  tag: string;
  key: string | null;
  props: Record<string, unknown>;
  children: VNode[];
}

// Wraps an imperative HTMLElement factory under a stable key so the reconciler reuses the produced node across renders.
export interface LeafVNode {
  kind: "leaf";
  key: string;
  props: Record<string, unknown>;
  render: () => HTMLElement;
}

interface ReconcilerState {
  previousTree: VNode | null;
  domByVNode: WeakMap<object, HTMLElement>;
}

const _rootStates: WeakMap<HTMLElement, ReconcilerState> = new WeakMap();

// Bring root's subtree into agreement with virtualTree, preserving DOM-node identity wherever VNode identity is unchanged.
export function reconcileInto({
  root,
  virtualTree,
}: {
  root: HTMLElement;
  virtualTree: VNode;
}): void {
  const state: ReconcilerState = _getOrCreateState({ root });
  const previousTree: VNode | null = state.previousTree;
  _reconcileAtRoot({
    parent: root,
    previousVNode: previousTree,
    currentVNode: virtualTree,
    state,
  });
  state.previousTree = virtualTree;
}

// Return the per-root reconciler state, creating it lazily on first use.
function _getOrCreateState({ root }: { root: HTMLElement }): ReconcilerState {
  const existing: ReconcilerState | undefined = _rootStates.get(root);
  if (existing !== undefined) {
    return existing;
  }
  const created: ReconcilerState = {
    previousTree: null,
    domByVNode: new WeakMap<object, HTMLElement>(),
  };
  _rootStates.set(root, created);
  return created;
}

// Reconcile the single VNode that occupies the root container, mounting/patching/replacing as needed.
function _reconcileAtRoot({
  parent,
  previousVNode,
  currentVNode,
  state,
}: {
  parent: HTMLElement;
  previousVNode: VNode | null;
  currentVNode: VNode;
  state: ReconcilerState;
}): void {
  if (previousVNode !== null && _sameIdentity({ a: previousVNode, b: currentVNode })) {
    const existingDom: HTMLElement | undefined = state.domByVNode.get(previousVNode);
    if (existingDom === undefined) {
      _replaceOnlyChild({ parent, previousVNode, currentVNode, state });
      return;
    }
    state.domByVNode.set(currentVNode, existingDom);
    _patchInPlace({ dom: existingDom, previousVNode, currentVNode, state });
    return;
  }
  if (previousVNode === null) {
    const mounted: HTMLElement = _mount({ vnode: currentVNode, state });
    parent.appendChild(mounted);
    return;
  }
  _replaceOnlyChild({ parent, previousVNode, currentVNode, state });
}

// Replace the previous DOM child under parent with a freshly mounted node for currentVNode.
function _replaceOnlyChild({
  parent,
  previousVNode,
  currentVNode,
  state,
}: {
  parent: HTMLElement;
  previousVNode: VNode;
  currentVNode: VNode;
  state: ReconcilerState;
}): void {
  const previousDom: HTMLElement | undefined = state.domByVNode.get(previousVNode);
  const mounted: HTMLElement = _mount({ vnode: currentVNode, state });
  if (previousDom !== undefined && previousDom.parentNode === parent) {
    parent.replaceChild(mounted, previousDom);
  } else {
    parent.appendChild(mounted);
  }
}

// Decide whether two VNodes refer to the same logical entity (same kind, key, and tag-for-element).
function _sameIdentity({ a, b }: { a: VNode; b: VNode }): boolean {
  if (a.kind !== b.kind) {
    return false;
  }
  if (a.key !== b.key) {
    return false;
  }
  if (a.kind === "element" && b.kind === "element" && a.tag !== b.tag) {
    return false;
  }
  return true;
}

// Patch an existing DOM node in place against a new VNode of matching identity.
function _patchInPlace({
  dom,
  previousVNode,
  currentVNode,
  state,
}: {
  dom: HTMLElement;
  previousVNode: VNode;
  currentVNode: VNode;
  state: ReconcilerState;
}): void {
  if (currentVNode.kind === "leaf") {
    _patchProps({
      dom,
      previousProps: previousVNode.props,
      currentProps: currentVNode.props,
    });
    return;
  }
  const previousElement: ElementVNode = previousVNode as ElementVNode;
  _reconcileChildren({
    parent: dom,
    previousChildren: previousElement.children,
    currentChildren: currentVNode.children,
    state,
  });
  _patchProps({
    dom,
    previousProps: previousElement.props,
    currentProps: currentVNode.props,
  });
}

// Create the DOM subtree for a freshly mounted VNode and record its identity mapping.
function _mount({
  vnode,
  state,
}: {
  vnode: VNode;
  state: ReconcilerState;
}): HTMLElement {
  if (vnode.kind === "leaf") {
    const element: HTMLElement = vnode.render();
    _patchProps({ dom: element, previousProps: {}, currentProps: vnode.props });
    state.domByVNode.set(vnode, element);
    return element;
  }
  const element: HTMLElement = document.createElement(vnode.tag);
  state.domByVNode.set(vnode, element);
  for (const child of vnode.children) {
    const childDom: HTMLElement = _mount({ vnode: child, state });
    element.appendChild(childDom);
  }
  _patchProps({ dom: element, previousProps: {}, currentProps: vnode.props });
  return element;
}

// Detach a previously mounted DOM node from its parent so it can be replaced.
function _unmount({
  vnode,
  state,
}: {
  vnode: VNode;
  state: ReconcilerState;
}): void {
  const dom: HTMLElement | undefined = state.domByVNode.get(vnode);
  if (dom === undefined) {
    return;
  }
  if (dom.parentNode !== null) {
    dom.parentNode.removeChild(dom);
  }
}

// Reconcile a list of children under parent by pairing previous and current VNodes positionally with identity matching.
function _reconcileChildren({
  parent,
  previousChildren,
  currentChildren,
  state,
}: {
  parent: HTMLElement;
  previousChildren: VNode[];
  currentChildren: VNode[];
  state: ReconcilerState;
}): void {
  const previousByKey: Map<string, { vnode: VNode; index: number }> = new Map();
  for (let i = 0; i < previousChildren.length; i++) {
    const previous: VNode = previousChildren[i];
    const compositeKey: string = _compositeKey({ vnode: previous, index: i });
    previousByKey.set(compositeKey, { vnode: previous, index: i });
  }

  const usedPreviousIndices: Set<number> = new Set();
  const resultDoms: HTMLElement[] = [];
  for (let i = 0; i < currentChildren.length; i++) {
    const current: VNode = currentChildren[i];
    const compositeKey: string = _compositeKey({ vnode: current, index: i });
    const matched: { vnode: VNode; index: number } | undefined = previousByKey.get(compositeKey);
    if (matched !== undefined && _sameIdentity({ a: matched.vnode, b: current })) {
      const existingDom: HTMLElement | undefined = state.domByVNode.get(matched.vnode);
      if (existingDom !== undefined) {
        state.domByVNode.set(current, existingDom);
        _patchInPlace({
          dom: existingDom,
          previousVNode: matched.vnode,
          currentVNode: current,
          state,
        });
        usedPreviousIndices.add(matched.index);
        resultDoms.push(existingDom);
        continue;
      }
    }
    const mounted: HTMLElement = _mount({ vnode: current, state });
    resultDoms.push(mounted);
  }

  for (let i = 0; i < previousChildren.length; i++) {
    if (!usedPreviousIndices.has(i)) {
      _unmount({ vnode: previousChildren[i], state });
    }
  }

  _alignChildren({ parent, desiredDoms: resultDoms });
}

// Produce a stable per-position identifier so positional children without explicit keys still reconcile.
function _compositeKey({ vnode, index }: { vnode: VNode; index: number }): string {
  if (vnode.kind === "leaf") {
    return `leaf:${vnode.key}`;
  }
  const keyPart: string = vnode.key === null ? `@${index}` : vnode.key;
  return `element:${vnode.tag}:${keyPart}`;
}

// Reorder/append the owned VNode-child DOM nodes under parent so they appear in the desired sequence, leaving foreign nodes (e.g. text nodes from a text prop) alone.
function _alignChildren({
  parent,
  desiredDoms,
}: {
  parent: HTMLElement;
  desiredDoms: HTMLElement[];
}): void {
  for (let i = desiredDoms.length - 1; i >= 0; i--) {
    const desired: HTMLElement = desiredDoms[i];
    const nextSibling: HTMLElement | null = i + 1 < desiredDoms.length ? desiredDoms[i + 1] : null;
    if (desired.parentNode === parent && desired.nextSibling === nextSibling) {
      continue;
    }
    parent.insertBefore(desired, nextSibling);
  }
}

// Apply the diff between previousProps and currentProps to dom, handling well-known prop names plus generic attributes.
function _patchProps({
  dom,
  previousProps,
  currentProps,
}: {
  dom: HTMLElement;
  previousProps: Record<string, unknown>;
  currentProps: Record<string, unknown>;
}): void {
  for (const name of Object.keys(previousProps)) {
    if (!(name in currentProps)) {
      _removeProp({ dom, name, previousValue: previousProps[name] });
    }
  }
  for (const name of Object.keys(currentProps)) {
    const previousValue: unknown = previousProps[name];
    const currentValue: unknown = currentProps[name];
    if (previousValue === currentValue) {
      continue;
    }
    _setProp({ dom, name, previousValue, currentValue });
  }
}

// Apply a single prop update to dom by name, dispatching across the well-known prop names.
function _setProp({
  dom,
  name,
  previousValue,
  currentValue,
}: {
  dom: HTMLElement;
  name: string;
  previousValue: unknown;
  currentValue: unknown;
}): void {
  if (name === "className") {
    dom.className = currentValue === null || currentValue === undefined ? "" : String(currentValue);
    return;
  }
  if (name === "id") {
    dom.id = currentValue === null || currentValue === undefined ? "" : String(currentValue);
    return;
  }
  if (name === "text") {
    dom.textContent = currentValue === null || currentValue === undefined ? "" : String(currentValue);
    return;
  }
  if (name === "value") {
    (dom as HTMLInputElement).value =
      currentValue === null || currentValue === undefined ? "" : String(currentValue);
    return;
  }
  if (name === "checked") {
    (dom as HTMLInputElement).checked = Boolean(currentValue);
    return;
  }
  if (name === "hidden") {
    dom.hidden = Boolean(currentValue);
    return;
  }
  if (name === "style") {
    _patchStyle({
      dom,
      previousStyle: (previousValue as Record<string, unknown> | undefined) ?? {},
      currentStyle: (currentValue as Record<string, unknown> | undefined) ?? {},
    });
    return;
  }
  if (_isEventName({ name })) {
    const eventName: string = name.slice(2).toLowerCase();
    if (typeof previousValue === "function") {
      dom.removeEventListener(eventName, previousValue as EventListener);
    }
    if (typeof currentValue === "function") {
      dom.addEventListener(eventName, currentValue as EventListener);
    }
    return;
  }
  if (currentValue === null || currentValue === undefined || currentValue === false) {
    dom.removeAttribute(name);
    return;
  }
  dom.setAttribute(name, String(currentValue));
}

// Clear a previously set prop from dom when it no longer appears in currentProps.
function _removeProp({
  dom,
  name,
  previousValue,
}: {
  dom: HTMLElement;
  name: string;
  previousValue: unknown;
}): void {
  if (name === "className") {
    dom.className = "";
    return;
  }
  if (name === "id") {
    dom.id = "";
    return;
  }
  if (name === "text") {
    dom.textContent = "";
    return;
  }
  if (name === "value") {
    (dom as HTMLInputElement).value = "";
    return;
  }
  if (name === "checked") {
    (dom as HTMLInputElement).checked = false;
    return;
  }
  if (name === "hidden") {
    dom.hidden = false;
    return;
  }
  if (name === "style") {
    _patchStyle({
      dom,
      previousStyle: (previousValue as Record<string, unknown> | undefined) ?? {},
      currentStyle: {},
    });
    return;
  }
  if (_isEventName({ name })) {
    const eventName: string = name.slice(2).toLowerCase();
    if (typeof previousValue === "function") {
      dom.removeEventListener(eventName, previousValue as EventListener);
    }
    return;
  }
  dom.removeAttribute(name);
}

// Decide whether a prop name encodes a DOM event listener (e.g. onclick, onchange, oninput).
function _isEventName({ name }: { name: string }): boolean {
  if (name.length < 3) {
    return false;
  }
  if (!name.startsWith("on")) {
    return false;
  }
  const third: string = name.charAt(2);
  return third === third.toLowerCase() && third !== third.toUpperCase()
    ? true
    : third >= "a" && third <= "z";
}

// Apply the diff between previousStyle and currentStyle onto dom.style.
function _patchStyle({
  dom,
  previousStyle,
  currentStyle,
}: {
  dom: HTMLElement;
  previousStyle: Record<string, unknown>;
  currentStyle: Record<string, unknown>;
}): void {
  for (const property of Object.keys(previousStyle)) {
    if (!(property in currentStyle)) {
      dom.style.removeProperty(property);
    }
  }
  for (const property of Object.keys(currentStyle)) {
    const previousValue: unknown = previousStyle[property];
    const currentValue: unknown = currentStyle[property];
    if (previousValue === currentValue) {
      continue;
    }
    if (currentValue === null || currentValue === undefined) {
      dom.style.removeProperty(property);
      continue;
    }
    (dom.style as unknown as Record<string, string>)[property] = String(currentValue);
  }
}
