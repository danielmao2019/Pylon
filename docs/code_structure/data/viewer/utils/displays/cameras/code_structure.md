# Data Viewer Cameras Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/cameras/dash/camera_display.py`

```text
camera_display.py
└── def create_camera_display
    └── # Builds the Dash camera-trajectory display from a loaded camera artifact.
```

### Backend schemas

`data/viewer/utils/displays/cameras/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class CameraDisplayResponse(DisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "camera"  # common field
    ├── url        # common field; camera-vis JSON payload URL
    └── meta_info  # common field; empty object for camera display
```

### Backend

`data/viewer/utils/displays/cameras/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, List, Optional, Tuple
├── from data.structures.three_d.camera.camera_vis import cameras_vis
├── from data.structures.three_d.camera.cameras import Cameras
├── from data.viewer.utils.displays.cameras.ts.backend.core_camera_display import create_camera_display_response_core
├── from data.viewer.utils.displays.cameras.ts.backend.schemas.display_response import CameraDisplayResponse
├── def create_camera_display_response(slot_id: str, title: str, cameras: Optional[Cameras], frustum_size: Optional[float] = None, frustum_color: Optional[Tuple[int, int, int]] = None, point_size: Optional[float] = None, point_color: Optional[Tuple[int, int, int]] = None) -> CameraDisplayResponse
│   ├── # Creates a camera display response from a caller-supplied Cameras; the caller may override the baked glyph styles, otherwise each None resolves to the cameras_vis module-global default.
│   ├── calls _map_camera_params_to_vis
│   ├── calls create_camera_display_response_core
│   └── return
├── def _map_camera_params_to_vis(cameras, frustum_size: Optional[float], frustum_color: Optional[Tuple[int, int, int]], point_size: Optional[float], point_color: Optional[Tuple[int, int, int]]) -> List[Dict[str, Any]]
│   ├── # Maps a Cameras collection to the JSON-able camera-vis payload (the camera sibling of _map_segmentation_pc_to_rgb), applying the caller's baked styles or their cameras_vis defaults.
│   ├── calls cameras_vis(cameras=cameras, frustum_size=frustum_size, frustum_color=frustum_color, point_size=point_size, point_color=point_color)  # cameras_vis resolves each None to its module-global style default
│   ├── for each camera-vis entry
│   │   └── calls _serialize_camera_vis_entry
│   └── return
├── def _serialize_camera_vis_entry(camera_vis_entry) -> Dict[str, Any]
│   ├── # Converts one camera-vis entry into the JSON shape consumed by the camera renderer.
│   ├── impls serializes center, center_color, and center_size  # impls-node-one-step:skip
│   ├── for each line in axes
│   │   └── calls _serialize_camera_vis_line
│   ├── for each line in frustum_lines
│   │   └── calls _serialize_camera_vis_line
│   └── return
└── def _serialize_camera_vis_line(camera_vis_line) -> Dict[str, Any]
    ├── # Converts one camera-vis line segment into plain start, end, and color lists.
    ├── impls serializes start, end, and color  # impls-node-one-step:skip
    └── return
```

`data/viewer/utils/displays/cameras/ts/backend/core_camera_display.py`

```text
core_camera_display.py
└── def create_camera_display_response_core(slot_id: str, title: str, camera_vis_payload: List[Dict[str, Any]], meta_info: Optional[Dict[str, Any]] = None) -> CameraDisplayResponse
    ├── # Creates a camera display response from the already-mapped camera-vis payload, exposing it through a frontend-loadable URL.
    ├── impls serializes camera_vis_payload to a json string
    ├── impls builds the camera-vis data URL by base64-encoding that json string
    ├── impls copies caller-provided meta_info into response metadata (empty object for camera display)
    └── return
```

### Frontend

`data/viewer/utils/displays/cameras/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface CameraDisplayResponse extends DisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "camera"  # common field
    ├── url        # common field; camera-vis JSON payload URL
    └── meta_info  # common field; empty object for camera display
```

`data/viewer/utils/displays/cameras/ts/frontend/camera_display.ts`

```text
camera_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { CameraDisplayResponse } from "./types/display_response";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_FRUSTUM_OPACITY = 0.5  # number — overlay render opacity applied when the caller does not supply frustumOpacity; a dynamic render property (the per-frame hover dimming multiplies it), not a baked glyph style — glyph size + color are baked by camera_vis
├── function renderCameraDisplay({ displayResponse, initialCameraState, frustumOpacity }: { displayResponse: CameraDisplayResponse; initialCameraState?: CameraState | null; frustumOpacity?: number }): LeafVNode
│   ├── # Builds a non-interactive transparent layer from the camera-vis JSON payload (glyph sizes + colors baked by camera_vis), initialized at initialCameraState.
│   ├── throw if CameraDisplayResponse.meta_info is not an empty object
│   ├── calls createSpatialDisplayScene({ initialCameraState, pointerEventsSuppressed: true })
│   ├── calls createCameraObject({ displayResponse, frustumOpacity })   → object
│   ├── impls scene.add(object)
│   ├── calls renderCamerasScene({ scene, camera, renderer })
│   └── return LeafVNode keyed by displayResponse.url
├── function createCameraObject({ displayResponse, frustumOpacity }: { displayResponse: CameraDisplayResponse; frustumOpacity?: number }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the camera frustums, populated once the async camera-vis payload load resolves.
│   ├── impls group = new THREE.Group()
│   ├── impls loadCamerasPayload({ displayResponse }).then(payload => group.add(createThreeCameras({ payload, frustumOpacity })))
│   └── return group
├── async function loadCamerasPayload({ displayResponse }: { displayResponse: CameraDisplayResponse }): Promise<CamerasPayload>
│   ├── # Async-fetches the camera-vis JSON payload from displayResponse.url and hands the decoded body to the payload validator.
│   ├── if displayResponse.url === null
│   │   └── throw new Error("camera display response url is null")
│   ├── impls response = await fetch(displayResponse.url)
│   ├── if !response.ok
│   │   └── throw new Error(`unable to load camera visualization: HTTP ${response.status}`)
│   ├── impls payload = validateCameraVisualizationPayloads({ value: await response.json() })
│   └── return payload
├── function createThreeCameras({ payload, frustumOpacity }: { payload: CamerasPayload; frustumOpacity?: number }): THREE.Object3D
│   ├── # Sync-builds the transparent Three.js centers + line segments from a pre-validated camera-vis payload, reading every baked glyph size + color from the payload.
│   ├── impls effectiveFrustumOpacity = frustumOpacity ?? DEFAULT_FRUSTUM_OPACITY
│   ├── for each entry in payload
│   │   ├── impls renders the center point at entry.center_size colored by entry.center_color
│   │   └── impls renders the axes + frustum lines each at its baked per-line color
│   └── return
└── function renderCamerasScene({ scene, camera, renderer }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }): void
    ├── # Drives the render loop; the cameras-overlay has no trackball controls — its camera is externally synced through the camera-sync registry observing the display element's data-camera-state attribute.
    ├── impls exposes the display element under displayResponse.slot_id so the caller can register it as a camera-sync target
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls: null })
    └── return
```
