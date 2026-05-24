# All-Benchmarks Refactor Tree

We expand this tree one hierarchy level at a time. At every checkpoint the user reviews and approves, then we descend. Once the tree is fully agreed, the actual code refactor is performed in one pass.

## 0. Refactor scope

`./web/` is a new top-level shared-frontend-infrastructure namespace, sibling to `./data/`. It hosts TS-frontend primitives that are not viewer-specific and not benchmark-specific — primitives any TypeScript SPA in this repo can depend on.

No existing source surface is migrated into `./web/`. The namespace is created empty in this refactor; the first inhabitant is `./web/reconcile/`, motivated by the route-render lifecycle rule in the ts-apps skill (`~/repos/AGENTS.md/agent_skills/coding/ts-apps/SKILL.md`, route rules item 6): render is reconcile-and-patch, not wholesale rebuild. The benchmark viewer's route render step consumes `reconcileInto` from this namespace; future TS SPAs in this repo do the same.

Out of scope for this tree:

- Any existing `./data/`, `./project/`, or task-local frontend code. Those are covered by the sibling refactor trees.
- A reconciler implementation. The skeleton declares the public surface (`VNode`, `reconcileInto`); the implementation strategy (custom diff loop, ref-keyed cache, etc.) is decided at code time.
