# All-Benchmarks Refactor Tree

We expand this tree one hierarchy level at a time. At every checkpoint the user reviews and approves, then we descend. Once the tree is fully agreed, the actual code refactor is performed in one pass.

## 2. Function structure trees

`./web/reconcile/reconcile.ts`

```text
reconcile.ts
├── # Identity-preserving DOM patch driver consumed by any TS SPA's route render step.
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
