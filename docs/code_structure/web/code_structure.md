# Web Code Structure

## 1. Code structure trees

`./web/reconcile/reconcile.ts`

```text
reconcile.ts
├── # VNode constructor plus identity-preserving DOM patch driver, consumed by any TS SPA's route render step.
├── type VNode = ElementVNode | LeafVNode
├── interface ElementVNode
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
│   ├── impls lifts `key` from props (defaulting to null) and keeps the remainder as the prop bag
│   ├── impls normalizes children: a bare string becomes a text leaf VNode, an existing VNode passes through
│   └── return ElementVNode { kind: "element", tag, key, props, children: normalized }
└── function reconcileInto({ root, virtualTree }: { root: HTMLElement; virtualTree: VNode }): void
    ├── # Bring root's subtree into agreement with virtualTree, preserving DOM-node identity wherever VNode identity is unchanged.
    ├── impls reads previously reconciled VNode tree associated with root
    ├── for each VNode position in virtualTree paired against the previous tree
    │   ├── if the previous and current VNode identities match
    │   │   ├── impls patches differing props on the existing DOM node
    │   │   └── impls descends into children
    │   ├── else if no previous VNode exists at this position
    │   │   └── impls mounts a new DOM node from the current VNode
    │   └── else
    │       ├── impls unmounts the previous DOM node
    │       └── impls mounts a replacement DOM node from the current VNode
    ├── impls records virtualTree as the previous tree associated with root
    └── return
```
