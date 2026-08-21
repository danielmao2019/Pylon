# Web Code Structure

## 1. Code structure trees

`./web/reconcile/reconcile.ts`

```text
reconcile.ts
├── # VNode constructor plus identity-preserving DOM patch driver, consumed by any TS SPA's route render step.
├── type VNode = ElementVNode | LeafVNode
├── interface ElementVNode
│   ├── # Declares a plain DOM element by its tag, stable key, prop bag and child VNodes, the shape the reconciler patches into place.
│   ├── kind: "element"
│   ├── tag: string
│   ├── key: string | null
│   ├── props: Record<string, unknown>
│   └── children: VNode[]
├── interface LeafVNode
│   ├── # Wraps an imperative HTMLElement factory under a stable key so the reconciler reuses the produced node across renders.
│   ├── kind: "leaf"
│   ├── key: string
│   ├── props: Record<string, unknown>
│   └── render: () => HTMLElement
├── function createElementVNode(tag: string, props: Record<string, unknown>, children: Array<VNode | string>): ElementVNode
│   ├── # Constructs an ElementVNode, normalizing the authoring shape into web's strict VNode union so call-sites express a tree rather than literals.
│   ├── impls lift `key` from props (defaulting to null)
│   ├── impls keep the remainder as the prop bag
│   ├── impls normalizes children: a bare string becomes a text leaf VNode, an existing VNode passes through
│   └── return ElementVNode { kind: "element", tag, key, props, children: normalized }
├── function reconcileInto({ root, virtualTree }: { root: HTMLElement; virtualTree: VNode }): void
│   ├── # Brings root's subtree into agreement with virtualTree, preserving DOM-node identity wherever VNode identity is unchanged.
│   ├── calls _getOrCreateState({ root })  # state
│   ├── impls previousTree = state.previousTree
│   ├── calls _reconcileAtRoot({ parent: root, previousVNode: previousTree, currentVNode: virtualTree, state })
│   └── impls state.previousTree = virtualTree
├── function _getOrCreateState({ root }: { root: HTMLElement }): ReconcilerState
│   ├── # Returns the root's reconciler state, creating and registering it on first use.
│   ├── if _rootStates already holds an entry for root
│   │   └── return that entry
│   ├── impls created = { previousTree: null, domByVNode: new WeakMap() }
│   ├── impls _rootStates.set(root, created)
│   └── return created
├── function _reconcileAtRoot({ parent, previousVNode, currentVNode, state }: { parent: HTMLElement; previousVNode: VNode | null; currentVNode: VNode; state: ReconcilerState }): void
│   ├── # Reconciles the root's single child: patch in place when identity holds, else mount or replace.
│   ├── if previousVNode is not null
│   │   ├── calls _sameIdentity({ a: previousVNode, b: currentVNode })
│   │   ├── if identity holds and state.domByVNode has no dom for previousVNode
│   │   │   ├── calls _replaceOnlyChild({ parent, previousVNode, currentVNode, state })
│   │   │   └── return
│   │   └── if identity holds
│   │       ├── impls state.domByVNode.set(currentVNode, that existing dom)
│   │       ├── calls _patchInPlace({ dom: existingDom, previousVNode, currentVNode, state })
│   │       └── return
│   ├── if previousVNode is null
│   │   ├── calls _mount({ vnode: currentVNode, state })
│   │   ├── impls parent.appendChild(that mounted element)
│   │   └── return
│   └── calls _replaceOnlyChild({ parent, previousVNode, currentVNode, state })
├── function _replaceOnlyChild({ parent, previousVNode, currentVNode, state }: { parent: HTMLElement; previousVNode: VNode; currentVNode: VNode; state: ReconcilerState }): void
│   ├── # Mounts the current vnode over the previous one's dom, appending when that dom is not parent's child.
│   ├── calls _mount({ vnode: currentVNode, state })
│   ├── if the previous dom exists and its parentNode is parent
│   │   └── impls parent.replaceChild(mounted, previousDom)
│   └── else
│       └── impls parent.appendChild(mounted)
├── function _patchInPlace({ dom, previousVNode, currentVNode, state }: { dom: HTMLElement; previousVNode: VNode; currentVNode: VNode; state: ReconcilerState }): void
│   ├── # Patches props onto an existing dom, recursing into children for element vnodes.
│   ├── if currentVNode is a leaf
│   │   ├── calls _patchProps({ dom, previousProps: previousVNode.props, currentProps: currentVNode.props })
│   │   └── return
│   ├── calls _reconcileChildren({ parent: dom, previousChildren: previousElement.children, currentChildren: currentVNode.children, state })
│   └── calls _patchProps({ dom, previousProps: previousElement.props, currentProps: currentVNode.props })
├── function _reconcileChildren({ parent, previousChildren, currentChildren, state }: { parent: HTMLElement; previousChildren: VNode[]; currentChildren: VNode[]; state: ReconcilerState }): void
│   ├── # Keyed child reconciliation: reuse matched doms, mount the rest, unmount the unused, then order them.
│   ├── for each previous child and its index
│   │   ├── calls _compositeKey({ vnode: previous, index: i })
│   │   └── impls previousByKey.set(that key, { vnode: previous, index: i })
│   ├── for each current child and its index
│   │   ├── calls _compositeKey({ vnode: current, index: i })
│   │   ├── if previousByKey holds that key
│   │   │   ├── calls _sameIdentity({ a: matched.vnode, b: current })
│   │   │   └── if identity holds and matched has a dom
│   │   │       ├── impls state.domByVNode.set(current, existingDom)
│   │   │       ├── impls the matched child re-enters _patchInPlace, recursing down the tree
│   │   │       ├── impls usedPreviousIndices.add(matched.index)
│   │   │       ├── impls resultDoms.push(existingDom)
│   │   │       └── impls continue  # the matched child keeps its dom; nothing below runs for it
│   │   ├── calls _mount({ vnode: current, state })
│   │   └── impls resultDoms.push(that mounted element)
│   ├── for each previous index not in usedPreviousIndices
│   │   └── calls _unmount({ vnode: previousChildren[i], state })
│   └── calls _alignChildren({ parent, desiredDoms: resultDoms })
├── function _compositeKey({ vnode, index }: { vnode: VNode; index: number }): string
│   ├── # Gives every child a stable identifier so unkeyed positional children still reconcile.
│   ├── if vnode is a leaf
│   │   └── return `leaf:${vnode.key}`
│   ├── impls keyPart = vnode.key, or `@${index}` when the key is null
│   └── return `element:${vnode.tag}:${keyPart}`
├── function _sameIdentity({ a, b }: { a: VNode; b: VNode }): boolean
│   ├── # Two vnodes share identity when kind, key and (for elements) tag all agree.
│   ├── if a.kind differs from b.kind
│   │   └── return false
│   ├── if a.key differs from b.key
│   │   └── return false
│   ├── if both are elements whose tag differs
│   │   └── return false
│   └── return true
├── function _mount({ vnode, state }: { vnode: VNode; state: ReconcilerState }): HTMLElement
│   ├── # Builds the dom for a vnode, registering it in state and recursing into element children.
│   ├── if vnode is a leaf
│   │   ├── impls element = vnode.render()
│   │   ├── calls _patchProps({ dom: element, previousProps: {}, currentProps: vnode.props })
│   │   ├── impls state.domByVNode.set(vnode, element)
│   │   └── return element
│   ├── impls element = document.createElement(vnode.tag)
│   ├── impls state.domByVNode.set(vnode, element)
│   ├── for each child of vnode.children
│   │   ├── calls _mount({ vnode: child, state })
│   │   └── impls element.appendChild(that child dom)
│   ├── calls _patchProps({ dom: element, previousProps: {}, currentProps: vnode.props })
│   └── return element
├── function _unmount({ vnode, state }: { vnode: VNode; state: ReconcilerState }): void
│   ├── # Detaches a vnode's dom from its parent, if it has one.
│   ├── if state.domByVNode has no dom for vnode
│   │   └── return
│   └── if that dom has a parentNode
│       └── impls dom.parentNode.removeChild(dom)
├── function _alignChildren({ parent, desiredDoms }: { parent: HTMLElement; desiredDoms: HTMLElement[] }): void
│   ├── # Orders parent's children to match desiredDoms, moving only those already out of place.
│   └── for each desired dom from last to first
│       ├── impls nextSibling = the following desired dom, or null at the end
│       ├── if desired is already parent's child and already precedes nextSibling
│       │   └── impls continue  # already in place; no move needed
│       └── impls parent.insertBefore(desired, nextSibling)
├── function _patchProps({ dom, previousProps, currentProps }: { dom: HTMLElement; previousProps: Record<string, unknown>; currentProps: Record<string, unknown> }): void
│   ├── # Removes the props that disappeared, then sets the ones whose value changed.
│   ├── for each name of previousProps absent from currentProps
│   │   └── calls _removeProp({ dom, name, previousValue: previousProps[name] })
│   └── for each name of currentProps
│       ├── if the current value equals the previous one
│       │   └── impls continue
│       └── calls _setProp({ dom, name, previousValue, currentValue })
├── function _setProp({ dom, name, previousValue, currentValue }: { dom: HTMLElement; name: string; previousValue: unknown; currentValue: unknown }): void
│   ├── # Applies one prop, dispatching on the special names before falling back to an attribute.
│   ├── if name is className, id, text or value
│   │   ├── impls the matching dom property takes String(currentValue), or "" when null or undefined
│   │   └── return
│   ├── if name is checked or hidden
│   │   ├── impls the matching dom property takes Boolean(currentValue)
│   │   └── return
│   ├── if name is style
│   │   ├── calls _patchStyle({ dom, previousStyle: previousValue ?? {}, currentStyle: currentValue ?? {} })
│   │   └── return
│   ├── calls _isEventName({ name })
│   ├── if it is an event name
│   │   ├── impls eventName = name.slice(2).toLowerCase()
│   │   ├── impls a function previousValue is removed as a listener
│   │   ├── impls a function currentValue is added as a listener
│   │   └── return
│   ├── if currentValue is null, undefined or false
│   │   ├── impls dom.removeAttribute(name)
│   │   └── return
│   └── impls dom.setAttribute(name, String(currentValue))
├── function _removeProp({ dom, name, previousValue }: { dom: HTMLElement; name: string; previousValue: unknown }): void
│   ├── # Clears one prop, mirroring the special-name dispatch _setProp uses.
│   ├── if name is className, id, text or value
│   │   ├── impls the matching dom property is cleared to ""
│   │   └── return
│   ├── if name is checked
│   │   ├── impls (dom as HTMLInputElement).checked = false
│   │   └── return
│   ├── if name is hidden
│   │   ├── impls dom.hidden = false
│   │   └── return
│   ├── if name is style
│   │   ├── calls _patchStyle({ dom, previousStyle: previousValue ?? {}, currentStyle: {} })
│   │   └── return
│   ├── calls _isEventName({ name })
│   ├── if it is an event name
│   │   ├── impls a function previousValue is removed as a listener
│   │   └── return
│   └── impls dom.removeAttribute(name)
├── function _isEventName({ name }: { name: string }): boolean
│   ├── # A prop is an event when it is "on" followed by a lowercase letter.
│   ├── if name is shorter than 3 characters or does not start with "on"
│   │   └── return false
│   └── return whether the third character is lowercase, or else falls in "a" through "z"
└── function _patchStyle({ dom, previousStyle, currentStyle }: { dom: HTMLElement; previousStyle: Record<string, unknown>; currentStyle: Record<string, unknown> }): void
    ├── # Removes the style properties that disappeared, then writes the ones whose value changed.
    ├── for each property of previousStyle absent from currentStyle
    │   └── impls dom.style.removeProperty(property)
    └── for each property of currentStyle
        ├── if the current value equals the previous one
        │   └── impls continue  # nothing to write
        ├── if the current value is null or undefined
        │   ├── impls dom.style.removeProperty(property)
        │   └── impls continue
        └── impls dom.style[property] = String(currentValue)
```
