# All-Benchmarks Refactor Tree

We expand this tree one hierarchy level at a time. At every checkpoint the user reviews and approves, then we descend. Once the tree is fully agreed, the actual code refactor is performed in one pass.

## 2. Folder structure trees

`./data/viewer/utils/`

```text
utils/
├── atomic_displays/
│   ├── utils/
│   │   ├── class_colors.py          # shared class-id to RGB palette utility
│   │   └── ts/
│   │       ├── backend/
│   │       │   └── schemas/
│   │       │       └── display_response.py # base atomic DisplayResponse schema
│   │       └── frontend/
│   │           ├── layered_display_container.ts # wraps base, original-overlay, and camera layer elements with caller-owned visibility state
│   │           └── types/
│   │               └── display_response.ts # base atomic DisplayResponse interface
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
│   └── cameras/                     # camera-vis display modality
│       ├── dash/
│       │   └── camera_display.py
│       └── ts/
│           ├── backend/
│           │   ├── schemas/
│           │   │   └── display_response.py # TS backend camera-display response schema: camera-vis JSON payload URL + empty meta_info
│           │   └── camera_display.py # camera artifact -> camera-vis JSON payload URL
│           └── frontend/
│               ├── types/
│               │   └── display_response.ts # TS camera-display response interface: camera-vis JSON payload URL + empty meta_info
│               └── camera_display.ts
├── camera_state/                    # generic serialized viewer camera state shared by spatial viewers
│   ├── dash/
│   │   └── camera_state.py           # Dash/Python CameraState contract
│   └── ts/
│       ├── backend/
│       │   ├── schemas/
│       │   │   └── camera_state.py  # TS backend CameraState schema
│       │   └── camera_state.py      # Camera -> TS backend CameraState conversion
│       └── frontend/
│           └── types.ts             # CameraState interface
├── camera_controls/                 # generic trackball 3D viewer camera controls
│   ├── dash/
│   │   └── trackball_camera_controls.py # trackball controls; left-drag rotate, right-drag pan, wheel zoom
│   └── ts/
│       └── frontend/
│           └── trackball_camera_controls.ts # trackball controls; left-drag rotate, right-drag pan, wheel zoom
├── camera_sync/                     # synchronized viewer-camera state shared across spatial displays
│   ├── dash/
│   │   └── camera_sync.py           # Dash camera-sync store and callback helpers
│   └── ts/
│       └── frontend/
│           ├── types.ts             # CameraSyncState interface
│           └── camera_sync.ts       # generic CameraSyncState store with camera-sync-specific additional APIs
└── note: unspecified existing data/viewer/utils entries stay untouched; specified entries live only in this tree
```

