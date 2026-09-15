# Data Viewer Points Display Code Structure

## 1. Code structure trees

`data/viewer/utils/displays/points/dash/apis.py`

```text
apis.py
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.displays.points.dash.core_points_display import create_dash_points_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display
│   ├── # Builds a Dash color point-cloud display from an already-colorized point-cloud path.
│   └── calls create_dash_points_display
├── def create_segmentation_pc_display
│   ├── # Builds a Dash segmentation point-cloud display by recoloring each point from its class id.
│   ├── calls load_point_cloud
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=segmentation_pc_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_points_display
└── def _map_segmentation_pc_to_rgb
    └── # Recolors the segmentation point cloud's per-point class labels to RGB via the class-to-RGB mapping for Dash display.
```

`data/viewer/utils/displays/points/dash/core_points_display.py`

```text
core_points_display.py
├── from typing import Optional, Tuple
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── DEFAULT_POINT_SIZE_FLOOR = 0.005  # absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when point_size is not supplied
├── DEFAULT_POINT_SIZE_RATIO = 0.002  # fraction of point-cloud bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── DEFAULT_POINT_COLOR = "#cccccc"   # uniform fallback color used when the point cloud has no per-point colors AND the caller does not supply point_color; lib-owned default, overridable
├── def create_dash_points_display(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None, lock_roll: Optional[Tuple[float, float, float]] = None) -> dcc.Graph
│   ├── # Renders a Dash point-cloud display element; point_size and point_color overrides are opt-in. point_color when supplied replaces per-point colors with a uniform color so the consumer can override the rendered look without rebuilding the data.
│   ├── calls create_dash_points_scene(point_cloud=point_cloud, point_size=point_size, point_color=point_color)
│   ├── calls create_dash_trackball_camera_controls(lock_roll=lock_roll)
│   ├── calls create_dash_points_component(scene=scene, controls=controls)
│   └── return
├── def create_dash_points_scene(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None) -> go.Scatter3d
│   ├── # Sync-builds the Plotly Scatter3d trace from the point cloud.
│   ├── impls bounding_radius = point_cloud bounding-sphere radius
│   ├── impls effective_size = point_size if point_size is not None else max(DEFAULT_POINT_SIZE_FLOOR, bounding_radius * DEFAULT_POINT_SIZE_RATIO)
│   ├── if point_color is not None
│   │   └── impls effective_color = point_color
│   ├── elif point_cloud has per-point rgb
│   │   └── impls effective_color = point_cloud.per_point_rgb
│   ├── else
│   │   └── impls effective_color = DEFAULT_POINT_COLOR
│   ├── impls trace = go.Scatter3d(x=..., y=..., z=..., mode="markers", marker=dict(size=effective_size, color=effective_color))
│   └── return trace
└── def create_dash_points_component(scene, controls)  # controls: the Plotly gl3d controls create_dash_trackball_camera_controls built
    ├── # Assembles the Dash component that hosts the point-cloud scene under its trackball camera controls.
    ├── impls assert isinstance(scene, go.Scatter3d)
    ├── impls display = dcc.Graph(figure=go.Figure(data=[scene], layout={"scene": controls["scene"]}))
    ├── if controls["graph_id"] is not None
    │   └── impls display.id = controls["graph_id"]  # the pattern-matching id the roll-lock callback holds this graph by
    └── return display  # the point-cloud display element
```

### Backend schemas

`data/viewer/utils/displays/points/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PointDisplayResponse(DisplayResponse)
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── class ColorPCDisplayResponse(PointDisplayResponse)
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── class SegmentationPCDisplayResponse(PointDisplayResponse)
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

### Backend

`data/viewer/utils/displays/points/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Optional, Tuple
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.structures.three_d.point_cloud.io.save_point_cloud import save_point_cloud
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.displays.points.ts.backend.core_points_display import create_points_display_response_core
├── from data.viewer.utils.displays.points.ts.backend.schemas.display_response import SegmentationPCDisplayResponse
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display_response
│   ├── # Creates a color point-cloud response from an already colorized point resource.
│   ├── impls point-display meta_info is empty metadata
│   ├── calls create_points_display_response_core
│   └── return
├── def create_segmentation_pc_display_response(segmentation_pc_path: str, slot_id: str, title: str, class_id_to_rgb: Optional[Dict[int, Tuple[int, int, int]]] = None) -> SegmentationPCDisplayResponse
│   ├── # Creates a segmentation point-cloud response from a class-labeled point resource; the caller may override the class-id → rgb mapping, otherwise the lib computes the default mapping via map_class_ids_to_rgb.
│   ├── calls load_point_cloud
│   ├── impls effective_class_id_to_rgb = class_id_to_rgb if class_id_to_rgb is not None else map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb
│   ├── calls _build_segmentation_pc_meta_info
│   ├── calls create_points_display_response_core
│   └── return
├── def _map_segmentation_pc_to_rgb(segmentation_pc_path: str, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> str
│   ├── # Writes a backend-colorized point-cloud resource using the class-to-RGB mapping.
│   ├── impls assert isinstance(segmentation_pc_path, str)
│   ├── impls assert isinstance(class_id_to_rgb, dict)
│   ├── calls load_point_cloud(filepath=segmentation_pc_path, device="cpu")
│   ├── impls label = the loaded cloud's own class ids, cast to torch.int64
│   ├── impls rgb = a float32 zeros tensor of shape (segmentation_pc.num_points, 3) on the cloud's device
│   ├── for each class_id, color in class_id_to_rgb.items()
│   │   └── impls rgb[label == int(class_id)] = color  # a label with no mapping entry keeps rgb 0
│   ├── impls colorized_data = the cloud's fields other than xyz / rgb / colors, carrying rgb under the "rgb" key
│   ├── impls colorized_pc = PointCloud(xyz=segmentation_pc.xyz, data=colorized_data)
│   ├── impls output_path = the deterministic colorized display path derived from segmentation_pc_path
│   ├── calls save_point_cloud(pc=colorized_pc, output_filepath=str(output_path))
│   ├── impls colorized_pc_path = str(output_path)
│   └── return colorized_pc_path  # the colorized point-cloud path the response serves
└── def _build_segmentation_pc_meta_info(class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> Dict[str, Any]
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`data/viewer/utils/displays/points/ts/backend/core_points_display.py`

```text
core_points_display.py
└── def create_points_display_response_core
    ├── # Creates a point display response from the loadable point resource path and caller-provided display metadata.
    ├── impls builds frontend resource url from point_cloud_path
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

### Frontend

`data/viewer/utils/displays/points/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── interface PointDisplayResponse extends DisplayResponse
│   ├── slot_id       # common field
│   ├── title         # common field
│   ├── display_kind  # common field
│   ├── url           # common field
│   └── meta_info     # common field
├── interface ColorPCDisplayResponse extends PointDisplayResponse
│   ├── slot_id  # common field
│   ├── title    # common field
│   ├── display_kind = "color_pc"  # common field
│   ├── url        # common field
│   └── meta_info  # common field
└── interface SegmentationPCDisplayResponse extends PointDisplayResponse
    ├── slot_id  # common field
    ├── title    # common field
    ├── display_kind = "segmentation_pc"  # common field
    ├── url        # common field
    └── meta_info  # common field
```

`data/viewer/utils/displays/points/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { ColorPCDisplayResponse, SegmentationPCDisplayResponse } from "./types/display_response";
├── import { renderPointsDisplay, createPointsObject } from "./core_points_display";
├── import { registerSpatialLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── function renderColorPCDisplay({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: ColorPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string }): LeafVNode
│   ├── # Renders a color point-cloud display with opt-in pointSize and pointColor overrides.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor })
│   └── return
├── function renderSegmentationPCDisplay({ displayResponse, initialCameraState, pointSize }: { displayResponse: SegmentationPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number }): LeafVNode
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info; per-point colors are already baked in by the backend's class-id → rgb mapping, so no color override is exposed here.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize })
│   └── return
└── impls registerSpatialLayerRenderer({ displayKind: "color_pc", layerRenderer: createPointsObject })  # module-load self-registration of the spatial color-pc layer renderer
```

`data/viewer/utils/displays/points/ts/frontend/core_points_display.ts`

```text
core_points_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { PointDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_POINT_SIZE_FLOOR = 0.005  # number — absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when pointSize is not supplied
├── const DEFAULT_POINT_SIZE_RATIO = 0.002  # number — fraction of geometry bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── const DEFAULT_POINT_COLOR = "#cccccc"   # hex color — uniform fallback used when geometry has no per-point colors AND the caller does not supply pointColor; lib-owned default, overridable
├── function renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor, lockRoll = null }: { displayResponse: PointDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string; lockRoll?: THREE.Vector3 | null }): LeafVNode
│   ├── # Renders a self-contained point-cloud display element initialized at initialCameraState.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createPointsObject({ displayResponse, pointSize, pointColor })   → object
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState, lockRoll })
│   ├── calls renderPointsScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createPointsObject({ displayResponse, pointSize, pointColor }: { displayResponse: PointDisplayResponse; pointSize?: number; pointColor?: string }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the point cloud, populated with the THREE.Points once the async geometry load resolves.
│   ├── impls group = new THREE.Group()
│   ├── impls loadPointGeometry({ displayResponse }).then(geometry => group.add(createThreePoints({ geometry, pointSize, pointColor })))
│   └── return group
├── async function loadPointGeometry({ displayResponse }: { displayResponse: PointDisplayResponse }): Promise<THREE.BufferGeometry>
│   ├── # Async-loads the point-cloud resource from displayResponse.url and returns a BufferGeometry with `position` and (when colors are present) `color` attributes.
│   ├── impls assert displayResponse.url !== null
│   ├── impls response = await fetch(displayResponse.url); buffer = await response.arrayBuffer()
│   ├── calls parsePlyBuffer({ buffer })                                                          → geometry
│   └── return geometry
├── function parsePlyBuffer({ buffer }: { buffer: ArrayBuffer }): THREE.BufferGeometry
│   ├── # Parses a PLY buffer (ASCII or binary little-endian) into a BufferGeometry with `position` and `color` attributes; internal PLY scalar/property parsing is private to this function.
│   ├── impls headerText = the buffer's leading 1048576 bytes decoded as utf-8
│   ├── impls endIndex = headerText.indexOf("end_header")
│   ├── if endIndex < 0
│   │   └── throw new Error("PLY header is missing end_header")
│   ├── impls dataOffset = the encoded byte length of headerText through the end of "end_header"
│   ├── impls advance dataOffset past each following newline byte, 10 or 13
│   ├── impls header = the vertex element's declared format, count, and scalar properties, read from the text before "end_header"  # impls-node-one-step:skip
│   ├── if header.format === "ascii"
│   │   ├── impls geometry = the post-header lines split on whitespace, read into the vertex position/color attributes
│   │   ├── impls a color channel the header's properties leave out reads 180
│   │   └── return geometry
│   ├── if header.format === "binary_little_endian"
│   │   ├── impls geometry = the post-header bytes read little-endian at each property's own offset/type, into the vertex position/color attributes
│   │   ├── impls a color channel the header's properties leave out reads 180
│   │   └── return geometry
│   └── throw new Error(`unsupported PLY format ${header.format}`)
├── function createThreePoints({ geometry, pointSize, pointColor }: { geometry: THREE.BufferGeometry; pointSize?: number; pointColor?: string }): THREE.Points
│   ├── # Sync-builds THREE.PointsMaterial + THREE.Points from the loaded geometry.
│   ├── impls geometry.computeBoundingSphere(); boundingRadius = geometry.boundingSphere.radius
│   ├── impls effectiveSize = pointSize ?? Math.max(DEFAULT_POINT_SIZE_FLOOR, boundingRadius * DEFAULT_POINT_SIZE_RATIO)
│   ├── if pointColor !== undefined
│   │   └── impls useVertexColors = false; effectiveColor = pointColor
│   ├── else if geometry.hasAttribute("color")
│   │   └── impls useVertexColors = true; effectiveColor = undefined
│   ├── else
│   │   └── impls useVertexColors = false; effectiveColor = DEFAULT_POINT_COLOR
│   ├── impls material = new THREE.PointsMaterial({ vertexColors: useVertexColors, size: effectiveSize, ...(effectiveColor !== undefined ? { color: effectiveColor } : {}) })  # constructor literal is exactly these keys; no other constructor key; no post-construction mutation of material
│   └── return new THREE.Points(geometry, material)  # no post-construction mutation of points
└── function renderPointsScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── # Drives the point-cloud render loop with the supplied trackball controls.
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```
