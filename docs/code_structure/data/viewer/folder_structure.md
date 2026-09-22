# Data Viewer Folder Structure

## Code folder structure

```text
data/viewer/utils/
├── displays/  # the display modalities (renamed from atomic_displays)
│   ├── utils/
│   ├── points/  # point-set display modality
│   ├── pixels/  # pixel-grid display modality
│   ├── placeholders/  # missing-result placeholder display modality
│   ├── videos/  # video display modality
│   ├── texts/  # text display modality
│   ├── tables/  # tabular display modality
│   ├── scene_graphs/  # scene-graph display modality
│   ├── mesh/  # triangle-mesh display modality
│   ├── gaussians/  # Gaussian-splat display modality
│   ├── aabbs/  # axis-aligned-box overlay display modality: 3D boxes over point clouds, 2D boxes over images
│   └── cameras/  # camera-vis display modality
├── controls/  # viewer controls: camera state/controls/sync, and selectors
│   ├── camera/  # camera state, trackball controls, and cross-display sync
│   └── selectors/  # generic hierarchical-cascade selector shared by viewers: a SelectorResponse option tree rendered as a dropdown cascade with parent-change re-mount and root-leaf path completion, so an app supplies only its option tree plus a path-change handler
└── note: unspecified existing data/viewer/utils entries stay untouched; specified entries live only in this tree
```

## Tests folder structure

```text
tests/data/viewer/
├── backend/          # backend display, state, initialization, transform, and edge-case tests
├── dataset/          # dataset-app integration tests
├── fixtures/         # shared mock-dataset fixtures
├── utils/            # viewer-utils tests + per-display-modality test packages mirroring the displays code modules above
│   └── displays/
│       ├── conftest.py
│       ├── depth_display/
│       ├── edge_display/
│       ├── image_display/
│       ├── instance_surrogate_display/
│       ├── layered_display/
│       ├── mesh_display/
│       ├── normal_display/
│       ├── point_cloud_display/
│       ├── segmentation_display/
│       ├── test_module_imports.py
│       └── test_dash_display_camera_controls.py  # Dash display-level roll-lock tests: the free trackball each Plotly display factory renders without lock_roll, and the roll-locked controls it renders when handed one
└── test_debounce.py  # debounce helper test
```

