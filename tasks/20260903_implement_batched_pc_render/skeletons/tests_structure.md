# `tasks/20260903_implement_batched_pc_render/` tests skeleton

## Tests implementation structure

```text
equivalence proof expectations  # agent-conducted procedures, each checked by running prove_equivalence.py against a clean checkout of main and reading the report it writes
├── A run from a clean worktree exits zero, and its report names the main commit it rendered against and this branch's commit.
├── Every single-camera render at point size one, handed over alone or as a batch of one, equals main's exactly on cpu and on cuda, for every renderer and mask option.
├── Every batched point preparation and every batched depth render equals its cameras rendered one by one exactly, at every point size and mask option, on cpu and on cuda.
├── Above one pixel, the report's point-size summary shows main's kernel matching this branch's at odd sizes only and main's depth and normal_2d entries ignoring the point size, which is where the differences from main come from.
├── The report's record counts equal the enumerated grid of devices, scenes, cameras, renderers, point sizes and mask options, so no comparison was skipped.
├── The scenes reach pixels that many points land on, a cloud of so few points that CUDA picks its small-matrix kernels, points behind some cameras, and points outside some images.
└── A render perturbed in one element is reported unequal, so the comparison is not vacuous.
```
