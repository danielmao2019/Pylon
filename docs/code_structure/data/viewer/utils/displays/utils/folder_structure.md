# Data Viewer Display Utils Folder Structure

## Code folder structure

```text
data/viewer/utils/displays/utils/
├── class_colors.py    # shared class-id to RGB palette utility
├── heatmap_colors.py  # shared non-negative-scalar to RGB palette utility (continuous; analogous to class_colors)
└── ts/
    ├── backend/
    │   └── schemas/
    │       ├── display_response.py          # base atomic DisplayResponse schema
    │       └── layered_display_response.py  # composite LayeredDisplayResponse schema: one base + a generic list of auxiliary layers
    └── frontend/
        ├── layered_display_container.ts  # composes one LayeredDisplayResponse into a shared spatial scene or stacked raster container, routing on the backend-stamped layer_class and dispatching each layer to its registry-resolved part-B renderer
        ├── layer_renderer_registry.ts    # display_kind -> per-layer part-B registry (spatial THREE-object builder / raster node builder) the renderers register into and the container looks up
        ├── register_layer_renderers.ts   # eager-glob-imports every display modality's frontend apis (Vite import.meta.glob) so each self-registers; imported once by the container
        ├── three_scene_helpers.ts        # shared three.js scene/perspective-camera/WebGL-renderer/display-container factories + the createSpatialDisplayScene part-A composer + render-loop starter
        └── types/
            ├── display_response.ts          # base atomic DisplayResponse interface
            └── layered_display_response.ts  # composite LayeredDisplayResponse interface: one base + a generic list of auxiliary layers
```
