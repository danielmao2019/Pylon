# Data Viewer Folder Structure

## Code folder structure

`./data/viewer/utils/`

```text
utils/
├── displays/                        # the display modalities (renamed from atomic_displays)
│   ├── utils/
│   │   ├── class_colors.py          # shared class-id to RGB palette utility
│   │   ├── heatmap_colors.py        # shared non-negative-scalar to RGB palette utility (continuous; analogous to class_colors)
│   │   └── ts/
│   │       ├── backend/
│   │       │   └── schemas/
│   │       │       ├── display_response.py # base atomic DisplayResponse schema
│   │       │       └── layered_display_response.py # composite LayeredDisplayResponse schema: one base + a generic list of auxiliary layers
│   │       └── frontend/
│   │           ├── layered_display_container.ts # composes one LayeredDisplayResponse into a shared spatial scene or stacked raster container, dispatching each layer to its registry-resolved part-B renderer
│   │           ├── layer_renderer_registry.ts # display_kind -> per-layer part-B registry (spatial THREE-object builder / raster node builder) the renderers register into and the container looks up
│   │           ├── register_layer_renderers.ts # eager-glob-imports every display modality's frontend apis (Vite import.meta.glob) so each self-registers; imported once by the container
│   │           ├── three_scene_helpers.ts # shared three.js scene/perspective-camera/WebGL-renderer/display-container factories + the createSpatialDisplayScene part-A composer + render-loop starter
│   │           └── types/
│   │               ├── display_response.ts # base atomic DisplayResponse interface
│   │               └── layered_display_response.ts # composite LayeredDisplayResponse interface: one base + a generic list of auxiliary layers
│   ├── points/                      # point-set display modality
│   │   ├── dash/
│   │   │   ├── core_points_display.py # Dash points display object core
│   │   │   └── apis.py              # Dash point-display APIs
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend point-display response schemas: url + meta_info
│   │       │   ├── core_points_display.py # TS DisplayResponse core for points
│   │       │   └── apis.py          # TS backend point-display APIs
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS point-display response interfaces: url + meta_info
│   │           ├── core_points_display.ts # TS points UI core
│   │           └── apis.ts          # TS frontend point-display APIs
│   ├── pixels/                      # pixel-grid display modality
│   │   ├── dash/
│   │   │   ├── core_pixels_display.py # Dash pixels display object core
│   │   │   └── apis.py              # Dash pixel-display APIs
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend pixel-display response schemas: url + meta_info
│   │       │   ├── core_pixels_display.py # TS DisplayResponse core for pixels
│   │       │   └── apis.py          # TS backend pixel-display APIs
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS pixel-display response interfaces: url + meta_info
│   │           ├── core_pixels_display.ts # TS pixels UI core
│   │           └── apis.ts          # TS frontend pixel-display APIs
│   ├── placeholders/                # missing-result placeholder display modality
│   │   ├── dash/
│   │   │   └── placeholder_display.py
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend placeholder-display response schema: message
│   │       │   └── placeholder_display.py
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS placeholder-display response interface: message
│   │           └── placeholder_display.ts
│   ├── videos/                      # video display modality
│   │   ├── dash/
│   │   │   └── video_display.py
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend video-display response schema: url + empty meta_info
│   │       │   └── video_display.py
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS video-display response interface: url + empty meta_info
│   │           └── video_display.ts
│   ├── texts/                       # text display modality
│   │   ├── dash/
│   │   │   └── text_display.py
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend text-display response schema: url + text + empty meta_info
│   │       │   └── text_display.py
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS text-display response interface: url + text + empty meta_info
│   │           └── text_display.ts
│   ├── tables/                      # tabular display modality
│   │   ├── dash/
│   │   │   └── table_display.py
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend table-display response schema: url + empty meta_info
│   │       │   └── table_display.py
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS table-display response interface: url + empty meta_info
│   │           └── table_display.ts
│   ├── scene_graphs/                # scene-graph display modality
│   │   ├── dash/
│   │   │   └── scene_graph_display.py
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend scene-graph-display response schema: url + empty meta_info
│   │       │   └── scene_graph_display.py
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS scene-graph-display response interface: url + empty meta_info
│   │           └── scene_graph_display.ts
│   ├── mesh/                        # triangle-mesh display modality
│   │   ├── dash/
│   │   │   ├── core_mesh_display.py # Dash mesh display object core
│   │   │   └── apis.py              # Dash mesh-display APIs
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend mesh-display response schemas: url + meta_info
│   │       │   ├── core_mesh_display.py # TS DisplayResponse core for meshes
│   │       │   └── apis.py          # TS backend mesh-display APIs
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS mesh-display response interfaces: url + meta_info
│   │           ├── core_mesh_display.ts # TS mesh UI core
│   │           └── apis.ts          # TS frontend mesh-display APIs
│   ├── gaussians/                   # Gaussian-splat display modality
│   │   ├── dash/
│   │   │   ├── core_gaussians_display.py # Dash Gaussian display object core
│   │   │   └── apis.py              # Dash Gaussian-display APIs
│   │   └── ts/
│   │       ├── backend/
│   │       │   ├── schemas/
│   │       │   │   └── display_response.py # TS backend Gaussian-display response schemas: url + meta_info
│   │       │   ├── core_gaussians_display.py # TS DisplayResponse core for Gaussians
│   │       │   └── apis.py          # TS backend Gaussian-display APIs
│   │       └── frontend/
│   │           ├── types/
│   │           │   └── display_response.ts # TS Gaussian-display response interfaces: url + meta_info
│   │           ├── core_gaussians_display.ts # TS Gaussian UI core
│   │           └── apis.ts          # TS frontend Gaussian-display APIs
│   ├── aabbs/                       # axis-aligned-box overlay display modality: 3D boxes over point clouds, 2D boxes over images
│   │   ├── threed/
│   │   │   └── ts/
│   │   │       ├── backend/
│   │   │       │   ├── schemas/
│   │   │       │   │   └── display_response.py # Aabb3dDisplayResponse: inline 3D boxes + optional per-box scores
│   │   │       │   └── apis.py      # create_aabb_3d_display_response
│   │   │       └── frontend/
│   │   │           ├── types/
│   │   │           │   └── display_response.ts # Aabb3dDisplayResponse interface
│   │   │           └── apis.ts      # renderAabb3dDisplay (standalone) + createAabb3dObject (part-B) for the spatial 3D boxes + score labels; self-registers aabb_3d
│   │   └── twod/
│   │       └── ts/
│   │           ├── backend/
│   │           │   ├── schemas/
│   │           │   │   └── display_response.py # Aabb2dDisplayResponse: inline 2D boxes + optional per-box scores
│   │           │   └── apis.py      # create_aabb_2d_display_response
│   │           └── frontend/
│   │               ├── types/
│   │               │   └── display_response.ts # Aabb2dDisplayResponse interface
│   │               └── apis.ts      # renderAabb2dDisplay: raster overlay of 2D boxes + score labels
│   └── cameras/                     # camera-vis display modality
│       ├── dash/
│       │   └── camera_display.py
│       └── ts/
│           ├── backend/
│           │   ├── schemas/
│           │   │   └── display_response.py # TS backend camera-display response schema: camera-vis JSON payload URL + empty meta_info
│           │   ├── core_camera_display.py # TS DisplayResponse core for cameras
│           │   └── apis.py # TS backend camera-display APIs
│           └── frontend/
│               ├── types/
│               │   └── display_response.ts # TS camera-display response interface: camera-vis JSON payload URL + empty meta_info
│               └── camera_display.ts
├── controls/                        # viewer controls: camera state/controls/sync, and selectors
│   ├── camera/                      # camera state, trackball controls, and cross-display sync
│   │   ├── camera_state/            # generic serialized viewer camera state shared by spatial viewers
│   │   │   ├── dash/
│   │   │   │   └── camera_state.py           # Dash/Python CameraState contract
│   │   │   └── ts/
│   │   │       ├── backend/
│   │   │       │   ├── schemas/
│   │   │       │   │   └── camera_state.py  # TS backend CameraState schema
│   │   │       │   └── camera_state.py      # Camera -> TS backend CameraState conversion
│   │   │       └── frontend/
│   │   │           └── types.ts             # CameraState interface
│   │   ├── camera_controls/         # generic trackball 3D viewer camera controls
│   │   │   ├── dash/
│   │   │   │   └── trackball_camera_controls.py # trackball controls; left-drag rotate, right-drag pan, wheel zoom
│   │   │   └── ts/
│   │   │       └── frontend/
│   │   │           └── trackball_camera_controls.ts # trackball controls; left-drag rotate, right-drag pan, wheel zoom
│   │   └── camera_sync/             # synchronized viewer-camera state shared across spatial displays
│   │       ├── dash/
│   │       │   └── camera_sync.py           # Dash camera-sync store and callback helpers
│   │       └── ts/
│   │           └── frontend/
│   │               ├── types.ts             # CameraSyncState interface
│   │               └── camera_sync.ts       # generic CameraSyncState store with camera-sync-specific additional APIs
│   └── selectors/                   # generic hierarchical-cascade selector shared by viewers: a SelectorResponse option tree rendered as a dropdown cascade with parent-change re-mount and root-leaf path completion, so an app supplies only its option tree plus a path-change handler
│       ├── dash/
│       │   └── selector_cascade.py      # Dash cascade selector: the dropdown stack from a SelectorResponse, re-rendered per parent change, each level change completed to a full root-leaf path
│       └── ts/
│           ├── backend/
│           │   └── schemas/
│           │       └── selector_response.py # SelectorResponse + SelectionNode schema: one axis's (value, label, children) option tree, plus a tree-builder from an app's (value, label, children) tuples
│           └── frontend/
│               ├── types/
│               │   └── selector_response.ts # SelectorResponse + SelectionNode interfaces mirroring the backend schema
│               ├── selection_path.ts        # generic root-leaf selection-path helper: complete a level change to a full root-leaf path (chosen value + first-child descent to a leaf)
│               └── selector_cascade.ts      # reusable cascade renderer: (SelectorResponse, current path, onPathChange) -> the dropdown-stack VNode; each <select> keyed by its option-set identity so a coarser-level change re-mounts it; on change the chosen value is completed to a root-leaf path (via selection_path) before onPathChange
└── note: unspecified existing data/viewer/utils entries stay untouched; specified entries live only in this tree
```

## Tests folder structure

`./tests/data/viewer/`

```text
tests/data/viewer/
├── backend/            # backend display, state, initialization, transform, and edge-case tests
├── dataset/            # dataset-app integration tests
├── fixtures/           # shared mock-dataset fixtures
├── utils/              # viewer-utils tests + per-display-modality test packages mirroring the displays code modules above
└── test_debounce.py    # debounce helper test
```

