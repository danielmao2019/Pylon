# `tasks/20260903_implement_batched_pc_render/` tests skeleton

## Tests implementation structure

```text
equivalence proof expectations  # agent-conducted procedures, each checked by running prove_equivalence.py against a clean checkout of main and reading the report it writes
├── A run from a clean worktree exits zero, and its report names the main commit it rendered against and this branch's commit.
├── Every single-camera render at point size one, handed over alone or as a batch of one, equals main's exactly on cpu and on cuda, for every renderer and mask option.
├── On cpu, every batched point preparation and every batched depth render equals its cameras rendered one by one exactly, at every point size and mask option.
├── On cuda, every batched point preparation agrees with each camera's own up to floating-point rounding, flipping a point's validity only where it sits that close to a cull boundary, and rasterizing or dilating a batch equals doing so to each of its slices exactly.
├── The report records how many cuda depth pixels differ end to end between a batch and its cameras one by one, so the rounding the preparation carries into a render is on record rather than hidden.
├── Above one pixel, the report's point-size summary shows main's kernel matching this branch's at odd sizes only and main's depth and normal_2d entries ignoring the point size, which is where the differences from main come from.
├── The report's record counts equal the enumerated grid of devices, scenes, cameras, renderers, point sizes and mask options, so no comparison was skipped.
├── The scenes reach pixels that many points land on, a cloud of so few points that CUDA picks its small-matrix kernels, points behind some cameras, and points outside some images.
└── A render perturbed in one element, and a preparation perturbed well past rounding, are each reported unequal, so neither comparison is vacuous.
```
