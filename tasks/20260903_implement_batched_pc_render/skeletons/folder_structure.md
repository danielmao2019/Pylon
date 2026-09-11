# `tasks/20260903_implement_batched_pc_render/` folder skeleton

## Code folder structure

```text
tasks/20260903_implement_batched_pc_render/
├── task.md  # the task spec
├── skeletons/  # this task's folder, code and tests skeleton
├── scene_rendering.py  # rebuilds a stored scene's cloud and cameras and renders one camera through a named entry, on either checkout
├── render_on_main.py  # renders every scene with main's code, run as a child process inside a checkout of main
├── prove_equivalence.py  # renders the same scenes on this branch and proves both equivalences, against main and across the batch
├── logs/  # the captured terminal output of each run
└── outputs/  # the scenes, main's renders and the equivalence report, behind a symlink to external storage
```

## Tests folder structure

```text
tasks/20260903_implement_batched_pc_render/  # the proof's expectations are agent-conducted procedures, drawn in tests_structure.md
```
