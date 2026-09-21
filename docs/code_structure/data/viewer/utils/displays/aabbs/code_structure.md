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
├── import { LineSegments2 } from "three/examples/jsm/lines/LineSegments2.js";
├── import { LineSegmentsGeometry } from "three/examples/jsm/lines/LineSegmentsGeometry.js";
├── import { LineMaterial } from "three/examples/jsm/lines/LineMaterial.js";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── import { registerSpatialLayerRenderer, type SpatialLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { Aabb3dDisplayResponse } from "./types/display_response";
├── const AABB_3D_BOX_COLOR = 0x4da6ff       # wireframe color of the 3D boxes
├── const AABB_3D_BOX_LINEWIDTH = 2          # screen-pixel edge width, which only LineMaterial honours since WebGL ignores LineBasicMaterial.linewidth
├── const AABB_3D_LABEL_HEIGHT_RATIO = 0.04  # label world height as a fraction of the boxes' bounding-sphere radius, so labels track scene scale under zoom
├── const AABB_3D_LABEL_ASPECT = 4           # label width-to-height ratio, matching the 256x64 label canvas
├── export function renderAabb3dDisplay({ displayResponse, initialCameraState = null, }: { displayResponse: Aabb3dDisplayResponse; initialCameraState?: CameraState | null; }): LeafVNode
│   ├── # Renders a self-contained 3D-box display initialized at initialCameraState.
│   ├── () => [local]
│   │   ├── # The leaf's render: mounts the spatial box overlay and returns its container.
│   │   ├── calls createSpatialDisplayScene({ initialCameraState })  # -> { container, scene, camera, renderer }
│   │   ├── calls createAabb3dObject({ displayResponse })            # -> object
│   │   ├── impls scene.add(object)
│   │   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  # -> controls
│   │   ├── calls renderAabb3dScene({ scene, camera, renderer, controls })
│   │   └── return container
│   ├── impls leaf = the LeafVNode keyed by displayResponse.url or `aabb_3d:${displayResponse.slot_id}`, with empty props and that render  # impls-node-one-step:skip — one constructor's fields
│   └── return leaf
├── export function createAabb3dObject({ displayResponse, }: { displayResponse: Aabb3dDisplayResponse; }): THREE.Object3D
│   ├── # Builds the inline 3D axis-aligned boxes and their optional per-box score labels into one THREE.Group.
│   ├── impls boxes = displayResponse.aabbs
│   ├── impls scores = displayResponse.scores
│   ├── calls _boxesBoundingRadius({ boxes })  # -> boundingRadius
│   ├── impls group = new THREE.Group()
│   ├── for boxIndex over every index of boxes
│   │   ├── impls box = boxes[boxIndex]
│   │   ├── impls (minX, minY, minZ, maxX, maxY, maxZ) = box
│   │   ├── impls boxGroup = new THREE.Group()
│   │   ├── calls _createBoxLines({ box })
│   │   ├── impls boxGroup.add(that box's line segments)
│   │   ├── if scores is not null
│   │   │   ├── calls _createScoreLabelSprite({ score: scores[boxIndex] })  # -> sprite
│   │   │   ├── impls sprite.position.set((minX + maxX) / 2, maxY, (minZ + maxZ) / 2)
│   │   │   ├── impls height = boundingRadius * AABB_3D_LABEL_HEIGHT_RATIO
│   │   │   ├── impls sprite.scale.set(height * AABB_3D_LABEL_ASPECT, height, 1)
│   │   │   ├── impls sprite.renderOrder = 1001
│   │   │   └── impls boxGroup.add(sprite)
│   │   └── impls group.add(boxGroup)
│   └── return group
├── function renderAabb3dScene({ scene, camera, renderer, controls, }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
│   ├── # Drives the 3D-box display render loop with the supplied trackball controls.
│   └── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
├── function _createBoxLines({ box }: { box: number[] }): LineSegments2
│   ├── # Builds one box's 12 AABB edges as a width-capable LineSegments2 that overlays the scene.
│   ├── impls (minX, minY, minZ, maxX, maxY, maxZ) = box
│   ├── impls corners = the eight AABB corners, indexed 0..7 by the (x, y, z) bit pattern low→high
│   ├── impls edges = the 12 corner-index pairs, 4 along x, 4 along y, 4 along z
│   ├── impls positions = []
│   ├── for each (a, b) in edges
│   │   └── impls positions.push(...corners[a], ...corners[b])
│   ├── impls geometry = new LineSegmentsGeometry()
│   ├── impls geometry.setPositions(positions)
│   ├── if window is undefined
│   │   └── impls 1
│   ├── else
│   │   └── impls window.innerWidth
│   ├── if window is undefined
│   │   └── impls 1
│   ├── else
│   │   └── impls window.innerHeight
│   ├── impls material = new LineMaterial(color AABB_3D_BOX_COLOR, linewidth AABB_3D_BOX_LINEWIDTH, depthTest off, transparent, resolution that viewport size)
│   ├── impls boxLines = new LineSegments2(geometry, material)
│   ├── impls boxLines.renderOrder = 1000
│   └── return boxLines
├── function _createScoreLabelSprite({ score }: { score: number }): THREE.Sprite
│   ├── # Renders one box's score onto a canvas texture carried by a sprite.
│   ├── impls canvas = document.createElement("canvas")
│   ├── impls canvas.width = 256
│   ├── impls canvas.height = 64
│   ├── impls context = canvas.getContext("2d")
│   ├── if context is null
│   │   └── throw aabb 3d score label canvas 2d context is unavailable
│   ├── impls context.fillStyle = "rgba(77,166,255,0.85)"
│   ├── impls context.fillRect(0, 0, canvas.width, canvas.height)
│   ├── impls context.fillStyle = "#ffffff"
│   ├── impls context.font = "bold 36px monospace"
│   ├── impls context.textBaseline = "middle"
│   ├── impls context.fillText(score.toFixed(2), 8, canvas.height / 2)
│   ├── impls texture = new THREE.CanvasTexture(canvas)
│   ├── impls spriteMaterial = new THREE.SpriteMaterial(map texture, depthTest off, transparent)
│   ├── impls sprite = the THREE.Sprite carrying spriteMaterial
│   └── return sprite
├── function _boxesBoundingRadius({ boxes }: { boxes: number[][] }): number
│   ├── # Computes the bounding-sphere radius of every box's corners, so score labels are sized to the overlay's extent.
│   ├── if boxes is empty
│   │   └── return 1
│   ├── impls boundingBox = new THREE.Box3()
│   ├── for each box in boxes
│   │   ├── impls (minX, minY, minZ, maxX, maxY, maxZ) = box
│   │   ├── impls boundingBox.expandByPoint(new THREE.Vector3(minX, minY, minZ))
│   │   └── impls boundingBox.expandByPoint(new THREE.Vector3(maxX, maxY, maxZ))
│   ├── impls sphere = new THREE.Sphere()
│   ├── impls boundingBox.getBoundingSphere(sphere)
│   ├── if sphere.radius > 0
│   │   └── impls sphere.radius
│   ├── else
│   │   └── impls 1
│   └── return  # that radius
└── calls registerSpatialLayerRenderer({ displayKind: "aabb_3d", layerRenderer: createAabb3dObject as SpatialLayerRenderer })  # module-load self-registration of the spatial aabb-3d layer renderer
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
