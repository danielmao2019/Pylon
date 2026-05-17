# All-Benchmarks Refactor Tree

We expand this tree one hierarchy level at a time. At every checkpoint the user reviews and approves, then we descend. Once the tree is fully agreed, the actual code refactor is performed in one pass.

## 0. Refactor scope

The refactor affects existing source surfaces that should not remain as the active implementation in their current form. These paths are either migrated into `project/benchmarks/*`, replaced by the new shared `data.viewer` structure, updated in place because they consume a shared surface being replaced, or removed after their behavior is represented by the new structure. New destination files are defined in the folder and code structure trees below, not in this existing-file scope list. In this section, `folder/*` means the existing source subtree under that folder, excluding generated caches, logs, reports, and outputs.

Shared `data.viewer` display and camera scope:

- `data/viewer/utils/atomic_displays/__init__.py`
- `data/viewer/utils/atomic_displays/*_display.py`
- `data/viewer/utils/atomic_displays/*_display*.js`
- `data/viewer/utils/camera_utils.py`
- `data/viewer/utils/gaussian_splatting.py`
- `data/viewer/utils/segmentation.py`
- `data/viewer/utils/camera_sync/*.py`
- `data/viewer/utils/camera_sync/*.js`

The `camera_utils.py` entry is included because it is the existing camera-state update/reset utility that overlaps the new generic `CameraState`, `CameraSyncState`, and trackball camera-control ownership. Preserve the `data.viewer` camera utility behavior itself; update external dataset-viewer callsites only to reflect the refactor and do not add old-path shims.

The `gaussian_splatting.py` entry is included because the new Gaussian atomic display tree absorbs or replaces its existing Gaussian-splat display metadata responsibility.

The `segmentation.py` entry is included because existing atomic display modules depend on its display color and segmentation rendering helpers. Preserve the display helper behavior itself; update external non-display dataset callsites only to reflect the refactor and do not add old-path shims.

Shared `data.viewer` display and camera test scope:

- `tests/data/viewer/utils/atomic_displays/*`
- `tests/data/viewer/utils/camera_sync/*`

These tests are affected because they validate the existing atomic-display and camera-sync surfaces being replaced. Other `tests/data/viewer/*` tests are not blanket-in-scope by path alone.

Task-local shared-viewer consumer scope:

- `project/tasks/20260414_try_tuning_pi_long_params/viewer/backend/paths.py`
- `project/tasks/20260414_try_tuning_pi_long_params/viewer/callbacks/register.py`
- `project/tasks/20260418_try_combining_a_and_b/viewer/backend/paths.py`
- `project/tasks/20260418_try_combining_a_and_b/viewer/backend/payloads.py`
- `project/tasks/20260418_try_combining_a_and_b/viewer/callbacks/register.py`
- `project/tasks/20260418_try_combining_a_and_b/viewer/layout/components.py`
- `project/tasks/20260419_cubical_plane_fitting/viewer/layout/components.py`
- `project/tasks/20260419_cubical_plane_fitting/viewer/routes/register.py`

Callsites outside `data.viewer` and `project` may be updated only when they directly import or call a `data.viewer` API whose path or contract changed in this refactor. Those edits must be limited to reflecting the `data.viewer` refactor and the intentional changes listed below; they must not make the caller's broader behavior a refactor target, and they must not preserve old call contracts through compatibility shims, alias modules, re-exports, or dual old/new API paths.

Explicitly out of scope:

- `project/tasks/20260414_try_tuning_pi_long_params/viewer/*` except the exact files listed in task-local shared-viewer consumer scope
- `project/tasks/20260418_try_combining_a_and_b/viewer/*` except the exact files listed in task-local shared-viewer consumer scope
- `project/tasks/20260419_cubical_plane_fitting/viewer/*` except the exact files listed in task-local shared-viewer consumer scope
- other task-local viewers outside `project/tasks/shared_viewers/*` and `project/tasks/benchmarks/*/viewer/*`, unless explicitly folded into the benchmark viewer later
- `data/viewer/dataset/*`
- `data/viewer/three_d_scene/*`
- `data/viewer/utils/dataset_utils.py`
- `data/viewer/utils/debug.py`
- `data/viewer/utils/display_utils.py`
- `data/viewer/utils/settings_config.py`
- `data/viewer/utils/debounce.py`
- `data/viewer/utils/structure_validation.py`
- `tests/data/viewer/utils/test_structure_validation.py`
- `data/datasets/*`
- `data/dataloaders/*`

Generated and runtime artifacts are ignored rather than tracked as refactor scope entries.

## 0.1. Intentional changes during refactor

This section records behavior and scope changes that are intentionally not equivalent to `main`. Anything not listed here should be treated as a preservation requirement: the refactor may move, rename, or restructure code, but it should not silently drop, add, or change product behavior.

Intentional viewer and display-surface changes:

- The original-overlay toggle is an intentional refactor upgrade. The overlay chain is expected to exist across backend display-response construction, `LayeredDisplayResponse` child responses, frontend content-view state, benchmark output views, and the layered-display container.
- Camera sync is a core part of the refactor, not optional. Generic camera state, camera-sync mechanics, and trackball camera controls belong under `data.viewer`; the benchmark viewer owns the project-specific per-leaf-scene camera-state bundle, resume policy, and registration of mounted benchmark display panels.

Intentional `data.viewer` changes:

- `data.viewer` atomic displays are intentionally reorganized into modality-owned subtrees. Existing display capabilities from `main` are preserved as capabilities: camera, point cloud, mesh, color image, depth image, edge image, normal image, segmentation image, and instance-surrogate image. Their old flat module locations are not preserved through shims.
- Additional first-class display/data support is intentionally added for video, text, table, scene graph, Gaussian/3DGS display responses (`color_gs` and `segmentation_gs`), placeholder display responses, TypeScript backend/frontend atomic-display response/rendering contracts, and layered-display response/container support. Nothing from the existing display capability set is intentionally removed.
- Generic `data.viewer` can support more display kinds than the benchmark viewer exposes. The benchmark viewer is a closed project-supported subset unless the taxonomy and display-slot contract explicitly add another display kind.
- `DisplayResponse.meta_info` is intentional renderer metadata: it may carry loading hints, parser/format hints, display statistics, class/color metadata, and other renderer-owned display facts.
- `meta_info` must not carry primary display payloads, rendered legend objects, subtitles, other presentation objects, or artifact availability state such as `available` or `missing`; missing selected artifact-backed display positions are represented structurally by `PlaceholderDisplaySlot` and `PlaceholderDisplayResponse`, while intentionally empty outputs remain `EmptyDisplayResponse`.
- For `CameraDisplayResponse`, `meta_info` is an empty object; the camera layer data is the main-branch camera-vis JSON payload fetched from `url`, not metadata.
- Point-cloud display capability is preserved as a Three.js WebGL display under `data.viewer` ownership: the TypeScript point renderer owns `THREE.Scene`, `THREE.WebGLRenderer`, `THREE.PerspectiveCamera`, `THREE.BufferGeometry`, `THREE.PointsMaterial`, and `THREE.Points` built from the point resource URL and `meta_info`.
- Scene-graph display is intentionally unified with the rest of the spatial atomic displays. The output is one `LayeredDisplayResponse` whose `base_display_response` is the scene-graph and whose `original_overlay_display_response` is the spatial input the scene-graph was constructed from — any spatial atomic display (point cloud, mesh, Gaussian splats, or other future spatial display kinds), toggled by the existing original-overlay control. This collapses main's 20260414 two-panel side-by-side layout (`scene_overlay` panel + `graph_only` panel) into the same single-display shape used by points, mesh, and gaussians outputs. The original_overlay layer's specific spatial encoding is the method's concern and not encoded in the scene-graph display itself. Scene-graph functionality from main is preserved as capabilities: sphere-sampled nodes, line-sampled edges, and HTML labels projected per frame from 3D positions.

Intentional task/project-boundary changes:

- The active refactor ownership scope is under `data.viewer` and `project`. Changes outside those roots are allowed only for callsites that directly consume changed `data.viewer` APIs, and only to reflect the refactor and intentional behavior changes in this section; unrelated caller behavior remains outside scope.
- Cubical plane fitting is part of this refactor only to the extent that it is represented in the benchmark taxonomy, benchmark viewer, method-wrapper integration, or accepted camera-sync/viewer path. Broader cubical-plane-fitting algorithm or task-local behavior changes are not intentional unless added to this section later.
- Generated caches, logs, reports, and method outputs are not refactor source-code files. When a concrete task directory is explicitly in play, outputs must follow the task-execution artifact layout; otherwise generated runtime artifacts should not be committed as part of the refactor.

Explicitly not intentional unless later added here:

- Compatibility shims, legacy re-exports, alias modules, and fallback adapters are never intentional refactor artifacts.

