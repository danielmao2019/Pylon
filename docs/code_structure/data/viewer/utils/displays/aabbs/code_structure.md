# Data Viewer AABBs Display Code Structure

## 1. Code structure trees

### Backend schemas

`data/viewer/utils/displays/aabbs/threed/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class Aabb3dDisplayResponse(DisplayResponse)
    ├── # Spatial overlay response: inline axis-aligned 3D boxes (each a 6-float box) with optional per-box scores, composed as an aux layer over a point cloud.
    ├── display_kind = "aabb_3d"  # common field
    ├── aabbs: List[List[float]]
    └── scores: Optional[List[float]]
```

### Backend

`data/viewer/utils/displays/aabbs/threed/ts/backend/apis.py`

```text
apis.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.aabbs.threed.ts.backend.schemas.display_response import Aabb3dDisplayResponse
└── def create_aabb_3d_display_response(slot_id: str, title: str, aabbs: List[List[float]], scores: Optional[List[float]] = None) -> Aabb3dDisplayResponse
    ├── # Creates a 3D axis-aligned-box overlay response from inline boxes and optional per-box scores.
    ├── calls Aabb3dDisplayResponse
    └── return
```

### Frontend

`data/viewer/utils/displays/aabbs/threed/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface Aabb3dDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "aabb_3d"  # common field
    ├── aabbs
    └── scores
```

`data/viewer/utils/displays/aabbs/threed/ts/frontend/apis.ts`

```text
apis.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { Aabb3dDisplayResponse } from "./types/display_response";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { registerSpatialLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── function renderAabb3dDisplay({ displayResponse, initialCameraState }: { displayResponse: Aabb3dDisplayResponse; initialCameraState?: CameraState | null }): LeafVNode
│   ├── # Renders a self-contained 3D-box display initialized at initialCameraState.
│   ├── calls createSpatialDisplayScene({ initialCameraState })   → { container, scene, camera, renderer }
│   ├── calls createAabb3dObject({ displayResponse })             → object
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })   → controls
│   ├── calls renderAabb3dScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createAabb3dObject({ displayResponse }: { displayResponse: Aabb3dDisplayResponse }): THREE.Object3D
│   ├── # Part-B: builds the inline 3D axis-aligned boxes and optional per-box score labels into a THREE.Group and returns it for the layered container to add.
│   ├── impls group = new THREE.Group()
│   ├── impls build the box-edges meshes from displayResponse.aabbs
│   ├── impls build the score labels from displayResponse.scores
│   ├── impls add the box-edges meshes to group
│   ├── impls add the score labels to group
│   └── return group
├── function renderAabb3dScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Drives the 3D-box display render loop with the supplied trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
└── impls registerSpatialLayerRenderer({ displayKind: "aabb_3d", layerRenderer: createAabb3dObject })  # module-load self-registration of the spatial aabb-3d layer renderer
```

### Backend schemas

`data/viewer/utils/displays/aabbs/twod/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class Aabb2dDisplayResponse(DisplayResponse)
    ├── # Raster overlay response: inline axis-aligned 2D boxes (each a 4-float box) with optional per-box scores, composed as an aux layer over an image.
    ├── display_kind = "aabb_2d"  # common field
    ├── aabbs: List[List[float]]
    └── scores: Optional[List[float]]
```

### Backend

`data/viewer/utils/displays/aabbs/twod/ts/backend/apis.py`

```text
apis.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.aabbs.twod.ts.backend.schemas.display_response import Aabb2dDisplayResponse
└── def create_aabb_2d_display_response(slot_id: str, title: str, aabbs: List[List[float]], scores: Optional[List[float]] = None) -> Aabb2dDisplayResponse
    ├── # Creates a 2D axis-aligned-box overlay response from inline boxes and optional per-box scores.
    ├── calls Aabb2dDisplayResponse
    └── return
```

### Frontend

`data/viewer/utils/displays/aabbs/twod/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface Aabb2dDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "aabb_2d"  # common field
    ├── aabbs
    └── scores
```

`data/viewer/utils/displays/aabbs/twod/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { Aabb2dDisplayResponse } from "./types/display_response";
├── import { registerRasterLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── function renderAabb2dDisplay({ displayResponse }: { displayResponse: Aabb2dDisplayResponse }): LeafVNode
│   ├── # Renders the inline 2D axis-aligned boxes and their optional per-box score labels as a full-bleed raster SVG overlay; the layered container sets its viewBox to the shared frustum on the base image's load.
│   ├── impls build the full-bleed SVG box overlay (preserveAspectRatio="none") from displayResponse.aabbs
│   ├── impls build the score labels from displayResponse.scores
│   └── return
└── impls registerRasterLayerRenderer({ displayKind: "aabb_2d", layerRenderer: renderAabb2dDisplay })  # module-load self-registration of the raster aabb-2d layer renderer
```
