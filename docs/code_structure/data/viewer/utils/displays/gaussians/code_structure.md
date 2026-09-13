# Data Viewer Gaussians Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/gaussians/dash/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.displays.gaussians.dash.core_gaussians_display import create_dash_gaussians_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_gs_display
│   ├── # Builds a Dash color Gaussian-splat display from an already-colorized Gaussian path.
│   └── calls create_dash_gaussians_display
├── def create_segmentation_gs_display
│   ├── # Builds a Dash segmentation Gaussian-splat display by recoloring each Gaussian from its class id.
│   ├── impls reads segmentation Gaussian class ids from segmentation_gs_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_gs_class_ids))
│   ├── calls _map_segmentation_gs_to_rgb(segmentation_gs_path=segmentation_gs_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_gaussians_display
└── def _map_segmentation_gs_to_rgb
    ├── # Recolors the segmentation Gaussian's per-Gaussian class ids to RGB via the class-to-RGB mapping.
    ├── impls assert isinstance(segmentation_gs_path, str)
    ├── impls assert isinstance(class_id_to_rgb, dict)
    └── raise NotImplementedError("Dash segmentation-to-color Gaussian mapping is declared by the skeleton but not exercised by any caller in this branch.")
```

`data/viewer/utils/displays/gaussians/dash/core_gaussians_display.py`

```text
core_gaussians_display.py
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── def create_dash_gaussians_display
│   ├── # Renders a Dash Gaussian-splat display element with trackball camera controls.
│   ├── calls create_dash_gaussians_scene
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_gaussians_component
│   └── return
├── def create_dash_gaussians_scene
│   ├── # Builds the Dash Gaussian-splat display scene from Gaussian data and display metadata.
│   ├── impls Dash Gaussian-splat display scene from Gaussian data and display metadata  # impls-node-one-step:skip
│   └── return
└── def create_dash_gaussians_component
    ├── # Assembles the Dash component that hosts the Gaussian-splat scene and its trackball camera controls.
    ├── impls assert isinstance(title, str)
    └── raise NotImplementedError("Dash Gaussian component assembly is declared by the skeleton but not exercised by any caller in this branch.")
```

### Backend schemas

`data/viewer/utils/displays/gaussians/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class GaussianDisplayResponse(DisplayResponse)
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorGSDisplayResponse(GaussianDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_gs"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class SegmentationGSDisplayResponse(GaussianDisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_gs"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/gaussians/ts/backend/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.displays.gaussians.ts.backend.core_gaussians_display import create_gaussians_display_response_core
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_gs_display_response
│   ├── # intentional thin wrapper: passes color Gaussian field directly to core response
│   ├── calls create_gaussians_display_response_core
│   └── return
├── def create_segmentation_gs_display_response
│   ├── # Creates a segmentation Gaussian response from a class-labeled Gaussian resource.
│   ├── impls reads segmentation Gaussian class ids from segmentation_gs_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_gs_class_ids))
│   ├── calls _map_segmentation_gs_to_rgb(segmentation_gs_path=segmentation_gs_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_gs_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_gaussians_display_response_core
│   └── return
├── def _map_segmentation_gs_to_rgb
│   └── # Writes a backend-colorized Gaussian resource by applying the class-to-RGB mapping to the segmentation Gaussian's class ids.
└── def _build_segmentation_gs_meta_info
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`data/viewer/utils/displays/gaussians/ts/backend/core_gaussians_display.py`

```text
core_gaussians_display.py
└── def create_gaussians_display_response_core
    ├── # Creates a Gaussian display response from the loadable Gaussian resource path and caller-provided display metadata.
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/gaussians/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface GaussianDisplayResponse extends DisplayResponse
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorGSDisplayResponse extends GaussianDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_gs"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface SegmentationGSDisplayResponse extends GaussianDisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_gs"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/gaussians/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { ColorGSDisplayResponse, SegmentationGSDisplayResponse } from "./types/display_response";
├── import { renderGaussiansDisplay } from "./core_gaussians_display";
├── function renderColorGSDisplay({ displayResponse, initialCameraState }: { displayResponse: ColorGSDisplayResponse; initialCameraState?: CameraState | null }): LeafVNode
│   ├── # Renders a color Gaussian-splat display from an already-colorized Gaussian resource.
│   ├── calls renderGaussiansDisplay({ displayResponse, initialCameraState })
│   └── return
└── function renderSegmentationGSDisplay({ displayResponse, initialCameraState }: { displayResponse: SegmentationGSDisplayResponse; initialCameraState?: CameraState | null }): LeafVNode
    ├── # renders backend-colorized segmentation display and legend derived from meta_info
    ├── calls renderGaussiansDisplay({ displayResponse, initialCameraState })
    └── return
```

`data/viewer/utils/displays/gaussians/ts/frontend/core_gaussians_display.ts`

```text
core_gaussians_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { GaussianDisplayResponse } from "./types/display_response";
├── import { createThreeDisplayContainer } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
└── function renderGaussiansDisplay({ displayResponse, initialCameraState }: { displayResponse: GaussianDisplayResponse; initialCameraState?: CameraState | null }): LeafVNode
    ├── # Delegates rendering to the external Gaussian-splat package; the package owns URL loading, scene assembly, camera controls, and the render loop.
    ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })                    → container
    ├── impls invoke the external Gaussian-splat package's mount API with { container, url: displayResponse.url, initialCameraState, meta_info: displayResponse.meta_info }
    └── return LeafVNode keyed by displayResponse.url
```
