# All-Benchmarks Refactor Tree

We expand this tree one hierarchy level at a time. At every checkpoint the user reviews and approves, then we descend. Once the tree is fully agreed, the actual code refactor is performed in one pass.

## 2. Folder structure trees

`./web/`

```text
web/
└── reconcile/
    └── reconcile.ts          # VNode types + reconcileInto: identity-preserving DOM patch driver
```
