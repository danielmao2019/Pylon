# Data Viewer Code Structure

## 1. Inheritance / type trees

`./data/viewer/utils/atomic_displays/utils/ts/backend/schemas/display_response.py`

Backend modality-specific display response schema files.

```text
class DisplayResponse
├── class PointDisplayResponse
│   ├── class ColorPCDisplayResponse
│   └── class SegmentationPCDisplayResponse
├── class PixelDisplayResponse
│   ├── class ColorImageDisplayResponse
│   ├── class DepthImageDisplayResponse
│   ├── class EdgeImageDisplayResponse
│   ├── class NormalImageDisplayResponse
│   ├── class SegmentationImageDisplayResponse
│   └── class InstanceSurrogateImageDisplayResponse
├── class VideoDisplayResponse
├── class TextDisplayResponse
├── class TableDisplayResponse
├── class SceneGraphDisplayResponse
├── class MeshDisplayResponse
│   ├── class ColorMeshDisplayResponse
│   ├── class SegmentationMeshDisplayResponse
│   ├── class HeatmapMeshDisplayResponse
│   └── class SparseHeatmapMeshDisplayResponse
├── class GaussianDisplayResponse
│   ├── class ColorGSDisplayResponse
│   └── class SegmentationGSDisplayResponse
├── class CameraDisplayResponse
├── class PlaceholderDisplayResponse
└── class LayeredDisplayResponse
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response.ts`

Frontend modality-specific display response type files.

```text
interface DisplayResponse
├── interface PointDisplayResponse
│   ├── interface ColorPCDisplayResponse
│   └── interface SegmentationPCDisplayResponse
├── interface PixelDisplayResponse
│   ├── interface ColorImageDisplayResponse
│   ├── interface DepthImageDisplayResponse
│   ├── interface EdgeImageDisplayResponse
│   ├── interface NormalImageDisplayResponse
│   ├── interface SegmentationImageDisplayResponse
│   └── interface InstanceSurrogateImageDisplayResponse
├── interface VideoDisplayResponse
├── interface TextDisplayResponse
├── interface TableDisplayResponse
├── interface SceneGraphDisplayResponse
├── interface MeshDisplayResponse
│   ├── interface ColorMeshDisplayResponse
│   ├── interface SegmentationMeshDisplayResponse
│   ├── interface HeatmapMeshDisplayResponse
│   └── interface SparseHeatmapMeshDisplayResponse
├── interface GaussianDisplayResponse
│   ├── interface ColorGSDisplayResponse
│   └── interface SegmentationGSDisplayResponse
├── interface CameraDisplayResponse
├── interface PlaceholderDisplayResponse
└── interface LayeredDisplayResponse
```

## 2. Code structure trees

Files below are grouped by folder structure; within a runtime folder, API/caller files appear before core/helper files when call order matters.

The base atomic `DisplayResponse` is owned by `./data/viewer/utils/atomic_displays/utils/ts/`; each modality-specific response inherits from that base under the matching `./data/viewer` modality.
`display_kind` selects the atomic renderer, `url` and typed response fields identify loadable resources, and `meta_info` carries renderer-owned loading hints plus display statistics/details such as class/color metadata.
`meta_info` must not encode primary display payloads, rendered legends, presentation objects, or artifact availability state such as `available` or `missing`.
Backend `data.viewer` camera-display code loads the selected camera artifact, interprets camera conventions, and prepares the camera-vis JSON payload exposed through `CameraDisplayResponse.url`.
`CameraDisplayResponse.meta_info` is empty because the camera-vis JSON payload is the camera display payload.
That payload is the main-branch camera visualization contract: a camera trajectory list whose entries preserve the `camera_vis()` semantics of `center`, `center_color`, `axes`, and `frustum_lines`, with every line carrying `start`, `end`, and `color`.
`camera_vis()` owns construction of one camera's visual primitive: `frustum_scale` is the single visualization scale for the camera frustum glyph, and camera intrinsics shape the frustum.
Missing intrinsics normalized to an identity matrix naturally produce the default frustum at the same `frustum_scale`; no separate intrinsics-provenance field is needed for camera display geometry.
`cameras_vis()` owns applying that primitive across a `Cameras` collection.
Backend `data.viewer` camera-display code owns serializing the generic camera-vis payload and exposing it at `CameraDisplayResponse.url`; frontend `data.viewer` owns rendering those centers and line segments.
Concrete artifact-backed `DisplayResponse` variants represent materialized displays, not unavailable displays with a status flag.

`CameraState`, `CameraSyncState`, and trackball camera controls are generic `data.viewer` contracts: `CameraState` serializes a viewer camera, `CameraSyncState` is the per-source entry storing one source's id, its registered target ids, and its current camera state (multiple sources coexist as independent entries keyed by source_id), and every 3D point-cloud, Gaussian, or mesh display must create trackball camera controls through `create_dash_trackball_camera_controls` in Dash or `createTrackballCameraControls` in TS.
TypeScript point-cloud displays use a Three.js WebGL point scene with `THREE.PerspectiveCamera` and `THREE.Points`; the point renderer builds the geometry from the selected point resource URL and renderer metadata.
Trackball controls must map left-button drag to camera rotation, right-button drag to camera panning, and mouse-wheel scroll to camera zoom; the viewer canvas must suppress the default browser context menu so right-button drag remains available for panning.
The implementation is not split by control-library family.
The camera-control helper owns renderer-specific control construction and must return controls that expose trackball camera-pose updates.
Orbit-style target-locked controls are forbidden, and no display may impose camera-pose restrictions through polar angle, azimuth angle, target lock, distance bounds, pan limits, translation limits, or rotation limits.
It uses the repo's serialized `Camera` contract, including extrinsics, intrinsics, convention, name, and id.

`./data/viewer/utils/atomic_displays/utils/class_colors.py`

```text
class_colors.py
├── from typing import Dict, Tuple
├── import torch
└── def map_class_ids_to_rgb(class_ids: torch.Tensor) -> Dict[int, Tuple[int, int, int]]
```

`./data/viewer/utils/atomic_displays/utils/heatmap_colors.py`

```text
heatmap_colors.py
├── import torch
└── def map_scalars_to_rgb(scalars: torch.Tensor) -> torch.Tensor
    ├── # Maps non-negative scalars to RGB via a fixed continuous heatmap palette.
    ├── assert scalars is non-negative
    └── return torch.Tensor of shape (*scalars.shape, 3)
```

`./data/viewer/utils/atomic_displays/utils/ts/backend/schemas/display_response.py`

```text
display_response.py
└── class DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind                                 # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response.ts`

```text
display_response.ts
└── interface DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind                                 # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/utils/ts/backend/schemas/layered_display_response.py`

```text
layered_display_response.py
├── from typing import List
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class LayeredDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "layered"                     # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    ├── base_display_response: DisplayResponse                # the single base layer
    └── aux_display_responses: List[DisplayResponse]          # ordered auxiliary layers stacked on top of the base; consumer-agnostic — each consumer assigns its own per-layer semantics (e.g. spatial input, camera frustums, per-part heatmaps) and owns its own visibility state
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/types/layered_display_response.ts`

```text
layered_display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface LayeredDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind: "layered"                      # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    ├── base_display_response: DisplayResponse
    └── aux_display_responses: DisplayResponse[]
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/layered_display_container.ts`

```text
layered_display_container.ts
├── import type { VNode } from "web/reconcile/reconcile";
└── function renderLayeredDisplayContainer({ layers, slotId }: { layers: readonly VNode[]; slotId: string }): VNode
    ├── # Stacks the provided child VNodes in given order into one layered-container ElementVNode.
    ├── assert layers is non-empty
    ├── assert slotId is non-empty
    └── return ElementVNode keyed by slotId with layers as identity-keyed children
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers.ts`

```text
three_scene_helpers.ts
├── import * as THREE from "three";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import { createTrackballCameraControls, DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── function createThreeDisplayContainer({ pointerEventsSuppressed }: { pointerEventsSuppressed: boolean }): HTMLDivElement
│   ├── # Shared display container for every TS atomic spatial display.
│   ├── impls absolutely-positioned full-bleed HTMLDivElement that owns the Three.js canvas
│   ├── if pointerEventsSuppressed
│   │   └── impls sets style.pointerEvents = "none" so the underlying base spatial display remains the interaction source
│   └── return
├── function createThreeScene({ object }: { object: THREE.Object3D }): THREE.Scene
│   ├── # Shared scene factory used by every TS atomic spatial display.
│   ├── impls creates THREE.Scene; scene.background stays unset so the renderer's clear color is what gets visibly drawn
│   ├── impls adds object to THREE.Scene
│   └── return
├── function createThreePerspectiveCamera({ initialCameraState }: { initialCameraState: CameraState | null }): THREE.PerspectiveCamera
│   ├── # Shared PerspectiveCamera factory used by every TS atomic spatial display; the consumer-supplied initialCameraState is the single source of initial framing (no lib-side fit-to-object — the lib does not know what the consumer considers a sensible default framing, and per-display fits across modalities mounted in one layered container produce inconsistent poses).
│   ├── impls THREE.PerspectiveCamera(fov=DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV, ...) at default aspect/near/far/position
│   ├── if initialCameraState is not null
│   │   └── impls overlays initialCameraState (every field — both intrinsics and extrinsics) onto the camera so first paint matches the source display
│   └── return
├── function createThreeWebGLRenderer({ container }: { container: HTMLDivElement }): THREE.WebGLRenderer
│   ├── # Shared WebGL renderer factory for every TS atomic spatial display.
│   ├── impls THREE.WebGLRenderer constructed with `alpha: true` and cleared transparent via `setClearColor(0x000000, 0)` so the canvas is transparent by default; consumers that want an opaque backdrop apply a CSS `background-color` to the marker
│   ├── impls canvas mounted inside the provided container
│   └── return
└── function startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> | null; onAfterRender?: () => void }): void
    ├── # Shared requestAnimationFrame loop; controls is null for passive overlays whose camera is externally synced, and onAfterRender lets a caller append a per-frame step (e.g. scene_graph's label projection).
    ├── if controls is not null
    │   └── impls calls controls.update() each frame
    ├── impls calls THREE.WebGLRenderer.render(scene, camera) each frame
    ├── if onAfterRender is provided
    │   └── impls invokes onAfterRender after each render
    └── return
```

`./data/viewer/utils/atomic_displays/points/dash/apis.py`

```text
apis.py
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.atomic_displays.points.dash.core_points_display import create_dash_points_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display
│   └── calls create_dash_points_display
├── def create_segmentation_pc_display
│   ├── calls load_point_cloud
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb(segmentation_pc_path=segmentation_pc_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_points_display
└── def _map_segmentation_pc_to_rgb
```

`./data/viewer/utils/atomic_displays/points/dash/core_points_display.py`

```text
core_points_display.py
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── def create_dash_points_display
│   ├── calls create_dash_points_scene
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_points_component
│   └── return
├── def create_dash_points_scene
│   ├── impls Dash point-display scene from point-cloud data and display metadata
│   └── return
└── def create_dash_points_component
    └── return
```

`./data/viewer/utils/atomic_displays/points/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PointDisplayResponse(DisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class ColorPCDisplayResponse(PointDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_pc"                    # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── class SegmentationPCDisplayResponse(PointDisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "segmentation_pc"             # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/points/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.atomic_displays.points.ts.backend.core_points_display import create_points_display_response
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display_response
│   ├── # Creates a color point-cloud response from an already colorized point resource.
│   ├── impls point-display meta_info is empty metadata
│   ├── calls create_points_display_response
│   └── return
├── def create_segmentation_pc_display_response
│   ├── # Creates a segmentation point-cloud response from a class-labeled point resource.
│   ├── calls load_point_cloud
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
│   ├── calls _map_segmentation_pc_to_rgb
│   ├── calls _build_segmentation_pc_meta_info
│   ├── calls create_points_display_response
│   └── return
├── def _map_segmentation_pc_to_rgb(segmentation_pc_path: str, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> str
│   ├── # Writes a backend-colorized point-cloud resource using the class-to-RGB mapping.
│   └── return
└── def _build_segmentation_pc_meta_info(class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> Dict[str, Any]
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`./data/viewer/utils/atomic_displays/points/ts/backend/core_points_display.py`

```text
core_points_display.py
└── def create_points_display_response
    ├── # Creates a point display response from the loadable point resource path and caller-provided display metadata.
    ├── impls builds frontend resource url from point_cloud_path
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/atomic_displays/points/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
├── interface PointDisplayResponse extends DisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface ColorPCDisplayResponse extends PointDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_pc"                    # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── interface SegmentationPCDisplayResponse extends PointDisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "segmentation_pc"             # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/points/ts/frontend/apis.ts`

```text
apis.ts
├── import type { VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { ColorPCDisplayResponse, SegmentationPCDisplayResponse } from "./types/display_response";
├── import { renderPointsDisplay } from "./core_points_display";
├── function renderColorPCDisplay({ displayResponse, initialCameraState }: { displayResponse: ColorPCDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState })
│   └── return
└── function renderSegmentationPCDisplay({ displayResponse, initialCameraState }: { displayResponse: SegmentationPCDisplayResponse; initialCameraState?: CameraState | null }): VNode
    ├── # renders backend-colorized segmentation display and legend derived from meta_info
    ├── calls renderPointsDisplay({ displayResponse, initialCameraState })
    └── return
```

`./data/viewer/utils/atomic_displays/points/ts/frontend/core_points_display.ts`

```text
core_points_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { PointDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createThreeDisplayContainer, createThreePerspectiveCamera, createThreeScene, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
├── function renderPointsDisplay({ displayResponse, initialCameraState }: { displayResponse: PointDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # Renders a self-contained point-cloud display element initialized at initialCameraState.
│   ├── calls createPointsScene({ displayResponse, initialCameraState })
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderPointsScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createPointsScene({ displayResponse, initialCameraState }: { displayResponse: PointDisplayResponse; initialCameraState: CameraState | null }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes the point-cloud scene's container, scene, camera, and renderer from displayResponse.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })
│   ├── calls createThreePoints
│   ├── calls createThreeScene({ object: points })
│   ├── calls createThreePerspectiveCamera({ initialCameraState })
│   ├── calls createThreeWebGLRenderer({ container })
│   └── return
├── function createThreePoints({ displayResponse }: { displayResponse: PointDisplayResponse }): THREE.Points
│   ├── impls loads point-cloud resource from displayResponse.url
│   ├── impls parses point positions and colors using displayResponse.meta_info parser/format hints
│   ├── impls creates THREE.BufferGeometry with position and optional color BufferAttribute
│   ├── impls creates THREE.PointsMaterial with vertexColors enabled when geometry has color attribute
│   ├── impls creates THREE.Points from geometry and material
│   └── return
└── function renderPointsScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/dash/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.atomic_displays.pixels.dash.core_pixels_display import create_dash_pixels_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_image_display
│   └── calls create_dash_pixels_display
├── def create_depth_image_display
│   ├── calls _map_depth_image_to_rgb
│   └── calls create_dash_pixels_display
├── def create_edge_image_display
│   ├── calls _map_edge_image_to_rgb
│   └── calls create_dash_pixels_display
├── def create_normal_image_display
│   ├── calls _map_normal_image_to_rgb
│   └── calls create_dash_pixels_display
├── def create_segmentation_image_display
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display
├── def create_instance_surrogate_image_display
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display
├── def _map_depth_image_to_rgb
├── def _map_edge_image_to_rgb
├── def _map_normal_image_to_rgb
├── def _map_segmentation_image_to_rgb
└── def _map_instance_surrogate_image_to_rgb
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/dash/core_pixels_display.py`

```text
core_pixels_display.py
└── def create_dash_pixels_display
```

`./data/viewer/utils/atomic_displays/pixels/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class PixelDisplayResponse(DisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class ColorImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_image"                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class DepthImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "depth_image"                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class EdgeImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "edge_image"                  # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class NormalImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "normal_image"                # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class SegmentationImageDisplayResponse(PixelDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "segmentation_image"          # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── class InstanceSurrogateImageDisplayResponse(PixelDisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "instance_surrogate_image"    # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/pixels/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.viewer.utils.atomic_displays.pixels.ts.backend.core_pixels_display import create_pixels_display_response
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_image_display_response
│   ├── # intentional thin wrapper: passes color image directly to core response
│   ├── calls create_pixels_display_response
│   └── return
├── def create_depth_image_display_response
│   ├── # maps depth image to color image before core response
│   ├── calls _map_depth_image_to_rgb
│   ├── calls create_pixels_display_response
│   └── return
├── def create_edge_image_display_response
│   ├── # maps edge image to color image before core response
│   ├── calls _map_edge_image_to_rgb
│   ├── calls create_pixels_display_response
│   └── return
├── def create_normal_image_display_response
│   ├── # maps normal image to color image before core response
│   ├── calls _map_normal_image_to_rgb
│   ├── calls create_pixels_display_response
│   └── return
├── def create_segmentation_image_display_response
│   ├── # Creates a segmentation image response from a class-labeled image resource.
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response
│   └── return
├── def create_instance_surrogate_image_display_response
│   ├── # maps instance-surrogate image to color image before core response
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_instance_surrogate_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response
│   └── return
├── def _map_depth_image_to_rgb
├── def _map_edge_image_to_rgb
├── def _map_normal_image_to_rgb
├── def _map_segmentation_image_to_rgb
├── def _build_segmentation_image_meta_info
│   ├── # Builds factual class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _map_instance_surrogate_image_to_rgb
├── def _build_instance_surrogate_image_meta_info
│   ├── # Builds factual class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
```

`./data/viewer/utils/atomic_displays/pixels/ts/backend/core_pixels_display.py`

```text
core_pixels_display.py
└── def create_pixels_display_response
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
├── interface PixelDisplayResponse extends DisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface ColorImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_image"                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface DepthImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "depth_image"                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface EdgeImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "edge_image"                  # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface NormalImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "normal_image"                # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface SegmentationImageDisplayResponse extends PixelDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "segmentation_image"          # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── interface InstanceSurrogateImageDisplayResponse extends PixelDisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "instance_surrogate_image"    # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/pixels/ts/frontend/apis.ts`

```text
apis.ts
├── import type { VNode } from "web/reconcile/reconcile";
├── import type { ColorImageDisplayResponse, DepthImageDisplayResponse, EdgeImageDisplayResponse, InstanceSurrogateImageDisplayResponse, NormalImageDisplayResponse, SegmentationImageDisplayResponse } from "./types/display_response";
├── import { renderPixelsDisplay } from "./core_pixels_display";
├── function renderColorImageDisplay({ displayResponse }: { displayResponse: ColorImageDisplayResponse }): VNode
│   ├── calls renderPixelsDisplay({ displayResponse })
│   └── return
├── function renderDepthImageDisplay({ displayResponse }: { displayResponse: DepthImageDisplayResponse }): VNode
│   ├── calls renderPixelsDisplay({ displayResponse })
│   └── return
├── function renderEdgeImageDisplay({ displayResponse }: { displayResponse: EdgeImageDisplayResponse }): VNode
│   ├── calls renderPixelsDisplay({ displayResponse })
│   └── return
├── function renderNormalImageDisplay({ displayResponse }: { displayResponse: NormalImageDisplayResponse }): VNode
│   ├── calls renderPixelsDisplay({ displayResponse })
│   └── return
├── function renderSegmentationImageDisplay({ displayResponse }: { displayResponse: SegmentationImageDisplayResponse }): VNode
│   ├── # renders backend-colorized segmentation display and legend derived from meta_info
│   ├── calls renderPixelsDisplay({ displayResponse })
│   └── return
└── function renderInstanceSurrogateImageDisplay({ displayResponse }: { displayResponse: InstanceSurrogateImageDisplayResponse }): VNode
    ├── # renders backend-colorized image display and legend derived from meta_info
    ├── calls renderPixelsDisplay({ displayResponse })
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/ts/frontend/core_pixels_display.ts`

```text
core_pixels_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { PixelDisplayResponse } from "./types/display_response";
└── function renderPixelsDisplay({ displayResponse }: { displayResponse: PixelDisplayResponse }): VNode
    ├── impls complete pixel-display UI from DisplayResponse url and meta_info
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/placeholders/dash/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display
```

`./data/viewer/utils/atomic_displays/placeholders/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class PlaceholderDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "placeholder"                 # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── message                                      # additional field
```

`./data/viewer/utils/atomic_displays/placeholders/ts/backend/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display_response
    ├── impls builds missing-result placeholder response from message
    └── return
```

`./data/viewer/utils/atomic_displays/placeholders/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface PlaceholderDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "placeholder"                 # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── message                                      # additional field
```

`./data/viewer/utils/atomic_displays/placeholders/ts/frontend/placeholder_display.ts`

```text
placeholder_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { PlaceholderDisplayResponse } from "./types/display_response";
└── function renderPlaceholderDisplay({ displayResponse }: { displayResponse: PlaceholderDisplayResponse }): VNode
    ├── impls complete missing-result placeholder UI from PlaceholderDisplayResponse.message
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/videos/dash/video_display.py`

```text
video_display.py
└── def create_video_display
```

`./data/viewer/utils/atomic_displays/videos/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class VideoDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "video"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/videos/ts/backend/video_display.py`

```text
video_display.py
└── def create_video_display_response
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty video metadata
    └── return
```

`./data/viewer/utils/atomic_displays/videos/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface VideoDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "video"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/videos/ts/frontend/video_display.ts`

```text
video_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { VideoDisplayResponse } from "./types/display_response";
└── function renderVideoDisplay({ displayResponse }: { displayResponse: VideoDisplayResponse }): VNode
    ├── impls complete video-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/texts/dash/text_display.py`

```text
text_display.py
└── def create_text_display
```

`./data/viewer/utils/atomic_displays/texts/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TextDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "text"                        # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── text                                         # additional field
```

`./data/viewer/utils/atomic_displays/texts/ts/backend/text_display.py`

```text
text_display.py
└── def create_text_display_response
    ├── impls stores text in TextDisplayResponse.text
    ├── impls sets meta_info to empty text metadata
    └── return
```

`./data/viewer/utils/atomic_displays/texts/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface TextDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "text"                        # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── text                                         # additional field
```

`./data/viewer/utils/atomic_displays/texts/ts/frontend/text_display.ts`

```text
text_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { TextDisplayResponse } from "./types/display_response";
└── function renderTextDisplay({ displayResponse }: { displayResponse: TextDisplayResponse }): VNode
    ├── impls complete text-display UI from TextDisplayResponse.text
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/tables/dash/table_display.py`

```text
table_display.py
└── def create_table_display
```

`./data/viewer/utils/atomic_displays/tables/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TableDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "table"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/tables/ts/backend/table_display.py`

```text
table_display.py
└── def create_table_display_response
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty table metadata
    └── return
```

`./data/viewer/utils/atomic_displays/tables/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface TableDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "table"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/tables/ts/frontend/table_display.ts`

```text
table_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { TableDisplayResponse } from "./types/display_response";
└── function renderTableDisplay({ displayResponse }: { displayResponse: TableDisplayResponse }): VNode
    ├── impls complete table-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/scene_graphs/dash/scene_graph_display.py`

```text
scene_graph_display.py
└── def create_scene_graph_display
```

`./data/viewer/utils/atomic_displays/scene_graphs/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class SceneGraphDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "scene_graph"                 # common field
    ├── url                                          # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/scene_graphs/ts/backend/scene_graph_display.py`

```text
scene_graph_display.py
├── from data.viewer.utils.atomic_displays.scene_graphs.ts.backend.schemas.display_response import SceneGraphDisplayResponse
├── def create_scene_graph_display_response
│   ├── # Builds the scene-graph base-layer response from a method-agnostic graph payload.
│   ├── # inputs: graph_nodes, graph_edges, object_nodes, scene_scale_reference_points, slot_id, title
│   ├── calls bake_scene_graph_payload
│   ├── impls builds frontend resource url pointing at the baked scene-graph payload
│   ├── impls sets meta_info to empty scene-graph metadata
│   └── return
├── def bake_scene_graph_payload
│   ├── # Bakes the full method-agnostic scene-graph asset served at SceneGraphDisplayResponse.url.
│   ├── calls estimate_scene_scale
│   ├── calls bake_scene_graph_geometry
│   ├── calls bake_scene_graph_labels
│   └── return
├── def bake_scene_graph_geometry
│   ├── # Bakes sphere-sampled nodes + line-sampled edges into the scene-graph geometry asset.
│   ├── calls sample_node_spheres
│   ├── calls sample_edge_lines
│   └── return
├── def bake_scene_graph_labels
│   ├── # Bakes per-object-node labels (text, position, color, class identity, frequency) offset above each position by scene_scale.
│   └── return
├── def estimate_scene_scale
│   ├── # Returns the world-units diagonal of the union of object positions, camera trajectory, and graph_nodes positions.
│   └── return
├── def sample_node_spheres
│   ├── # Samples each graph node into a sphere-shaped point patch, with radius derived from node_type and scene_scale, colored by node.color.
│   └── return
└── def sample_edge_lines
    ├── # Samples each graph edge into a densely-sampled line from source.position to target.position, colored by edge color.
    └── return
```

`./data/viewer/utils/atomic_displays/scene_graphs/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface SceneGraphDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "scene_graph"                 # common field
    ├── url                                          # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/scene_graphs/ts/frontend/scene_graph_display.ts`

```text
scene_graph_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { SceneGraphDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createThreeDisplayContainer, createThreePerspectiveCamera, createThreeScene, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
├── function renderSceneGraphDisplay({ displayResponse, initialCameraState }: { displayResponse: SceneGraphDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # Renders a self-contained scene-graph display: baked node/edge geometry plus HTML label overlay projected per frame.
│   ├── calls createSceneGraphScene({ displayResponse, initialCameraState })
│   ├── calls createTrackballCameraControls({ camera, renderer, initialCameraState })
│   ├── calls renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay })
│   └── return LeafVNode keyed by displayResponse.url
├── function createSceneGraphScene({ displayResponse, initialCameraState }: { displayResponse: SceneGraphDisplayResponse; initialCameraState: CameraState | null }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; labels: object[]; labelOverlay: HTMLDivElement }
│   ├── # Composes the scene-graph scene's container, scene, camera, renderer, label data, and the HTML label-overlay container from displayResponse.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })
│   ├── calls createThreeSceneGraphPoints
│   ├── calls createThreeScene({ object: points })
│   ├── calls createThreePerspectiveCamera({ initialCameraState })
│   ├── calls createThreeWebGLRenderer({ container })
│   ├── calls createThreeSceneGraphLabelOverlay({ container })
│   └── return
├── function createThreeSceneGraphPoints({ displayResponse }: { displayResponse: SceneGraphDisplayResponse }): { points: THREE.Points; labels: object[] }
│   ├── impls fetches the scene-graph payload from displayResponse.url
│   ├── impls parses node/edge geometry into THREE.Points (positions + colors, vertexColors enabled)
│   ├── impls parses the payload's label list as the per-frame label data
│   └── return
├── function createThreeSceneGraphLabelOverlay({ container }: { container: HTMLDivElement }): HTMLDivElement
│   ├── impls absolutely-positioned HTML overlay container layered above the canvas, returned and mounted inside the display container
│   └── return
├── function renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; labels: object[]; labelOverlay: HTMLDivElement }): void
│   ├── # Drives the render + label-projection loop by wrapping the shared startThreeSceneRenderLoop with an onAfterRender step that projects labels each frame.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender: () => _projectLabelsOntoOverlay({ camera, labels, labelOverlay }) })
│   └── return
└── function _projectLabelsOntoOverlay({ camera, labels, labelOverlay }: { camera: THREE.PerspectiveCamera; labels: object[]; labelOverlay: HTMLDivElement }): void
    ├── # Per-frame step: projects each label's world position into overlay-pixel coordinates, updates the HTML node positions, and culls offscreen labels.
    ├── impls projects each label's world position to NDC via camera, then converts to overlay-pixel coordinates
    ├── impls updates each label's HTML node position (left/top), and culls labels behind the camera or outside the viewport
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/dash/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.atomic_displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.atomic_displays.utils.heatmap_colors import map_scalars_to_rgb
├── def create_color_mesh_display
│   └── calls create_dash_mesh_display
├── def create_segmentation_mesh_display
│   ├── impls reads segmentation mesh class ids from segmentation_mesh_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(segmentation_mesh_path=segmentation_mesh_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_mesh_display
├── def create_heatmap_mesh_display
│   ├── impls reads heatmap mesh scalar values from heatmap_mesh_path (per-vertex 1-D or per-texel 2-D, non-negative)
│   ├── calls map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
│   ├── calls _map_heatmap_mesh_to_rgb(heatmap_mesh_path=heatmap_mesh_path, scalar_rgb=scalar_rgb)
│   └── calls create_dash_mesh_display
├── def _map_segmentation_mesh_to_rgb
│   ├── # Applies class_id_to_rgb to the segmentation mesh's class-id storage.
│   ├── if class-id storage is per-vertex
│   │   └── impls assigns class_id_to_rgb[c] as the per-vertex RGB for class id c
│   ├── elif class-id storage is per-texel
│   │   └── impls assigns class_id_to_rgb[c] as the per-texel RGB on the UV texture map
│   └── return colored mesh
└── def _map_heatmap_mesh_to_rgb
    ├── # Writes scalar_rgb onto the heatmap mesh's scalar storage.
    ├── if scalar storage is per-vertex
    │   └── impls assigns scalar_rgb as the per-vertex RGB
    ├── elif scalar storage is per-texel
    │   └── impls assigns scalar_rgb as the per-texel RGB on the UV texture map
    └── return colored mesh
```

`./data/viewer/utils/atomic_displays/mesh/dash/core_mesh_display.py`

```text
core_mesh_display.py
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── def create_dash_mesh_display
│   ├── calls create_dash_mesh_scene
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_mesh_component
│   └── return
├── def create_dash_mesh_scene
│   ├── if mesh texture representation is vertex color
│   │   ├── calls _create_vertex_color_mesh_display
│   │   └── return
│   ├── elif mesh texture representation is UV texture map
│   │   ├── calls _create_uv_texture_map_mesh_display
│   │   └── return
│   └── else
│       └── raise unsupported mesh texture representation
├── def _create_vertex_color_mesh_display
├── def _create_uv_texture_map_mesh_display
└── def create_dash_mesh_component
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class MeshDisplayResponse(DisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class ColorMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_mesh"                  # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class SegmentationMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "segmentation_mesh"           # common field
│   ├── url                                          # common field — the class-colorized mesh resource
│   └── meta_info                                    # common field
├── class HeatmapMeshDisplayResponse(MeshDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "heatmap_mesh"                # common field
│   ├── url                                          # common field — the heatmap-colorized mesh resource
│   └── meta_info                                    # common field
└── class SparseHeatmapMeshDisplayResponse(MeshDisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "sparse_heatmap_mesh"         # common field
    ├── url                                          # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/mesh/ts/backend/apis.py`

```text
apis.py
├── from pathlib import Path
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.viewer.utils.atomic_displays.mesh.ts.backend.core_mesh_display import create_mesh_display_response
├── from data.viewer.utils.atomic_displays.mesh.ts.backend.schemas.display_response import ColorMeshDisplayResponse, HeatmapMeshDisplayResponse, SegmentationMeshDisplayResponse, SparseHeatmapMeshDisplayResponse
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.atomic_displays.utils.heatmap_colors import map_scalars_to_rgb
├── def create_color_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> ColorMeshDisplayResponse
│   ├── # Intentional thin wrapper: writes the color mesh resource at output_path and returns ColorMeshDisplayResponse with the caller-provided url.
│   ├── calls create_mesh_display_response
│   └── return
├── def create_segmentation_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> SegmentationMeshDisplayResponse
│   ├── # Creates a segmentation mesh response from a class-labeled mesh resource read from input_path; processed mesh is written to output_path.
│   ├── impls reads segmentation mesh class ids from input_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(input_path=input_path, output_path=output_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_mesh_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_mesh_display_response
│   └── return
├── def create_heatmap_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> HeatmapMeshDisplayResponse
│   ├── # Creates a heatmap mesh response from a non-negative-scalar-labeled mesh resource read from input_path; processed mesh is written to output_path.
│   ├── impls reads heatmap mesh scalar values from input_path (per-vertex 1-D or per-texel 2-D, non-negative)
│   ├── calls map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
│   ├── calls _map_heatmap_mesh_to_rgb(input_path=input_path, output_path=output_path, scalar_rgb=scalar_rgb)
│   ├── calls _build_heatmap_mesh_meta_info(scalars=heatmap_mesh_scalars)
│   ├── calls create_mesh_display_response
│   └── return
├── def create_sparse_heatmap_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> SparseHeatmapMeshDisplayResponse
│   ├── # Creates a sparse heatmap mesh response; writes the sparse (indices, values) delta resource to output_path.
│   ├── impls reads the (indices, values) delta and the geometry reference from input_path
│   ├── calls _write_sparse_heatmap_resource(input_path=input_path, output_path=output_path)
│   ├── calls _build_sparse_heatmap_mesh_meta_info(indices=indices, values=values)
│   └── return SparseHeatmapMeshDisplayResponse with slot_id, title, url, meta_info from caller-provided args
├── def _map_segmentation_mesh_to_rgb(input_path: Path, output_path: Path, class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> None
│   ├── # Reads segmentation mesh from input_path, applies class_id_to_rgb, writes the resulting color mesh to output_path.
│   ├── if class-id storage is per-vertex
│   │   └── impls assigns class_id_to_rgb[c] as the per-vertex RGB for class id c
│   ├── elif class-id storage is per-texel
│   │   └── impls assigns class_id_to_rgb[c] as the per-texel RGB on the UV texture map
│   └── return
├── def _map_heatmap_mesh_to_rgb(input_path: Path, output_path: Path, scalar_rgb: torch.Tensor) -> None
│   ├── # Reads heatmap mesh from input_path, writes scalar_rgb onto its scalar storage, and saves the resulting color mesh to output_path.
│   ├── if scalar storage is per-vertex
│   │   └── impls assigns scalar_rgb as the per-vertex RGB
│   ├── elif scalar storage is per-texel
│   │   └── impls assigns scalar_rgb as the per-texel RGB on the UV texture map
│   └── return
├── def _write_sparse_heatmap_resource(input_path: Path, output_path: Path) -> None
│   ├── # Writes the (indices, values) delta + geometry reference from input_path to output_path as the wire resource.
│   └── return
├── def _build_segmentation_mesh_meta_info
│   ├── # Builds class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _build_heatmap_mesh_meta_info
│   ├── # Builds scalar-range metadata from the input scalars.
│   ├── impls stores scalar min/max
│   └── return
└── def _build_sparse_heatmap_mesh_meta_info
    ├── # Builds scalar-range + non-zero-count metadata from the input sparse arrays.
    ├── impls stores values min/max and number of non-zero entries
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/ts/backend/core_mesh_display.py`

```text
core_mesh_display.py
├── from pathlib import Path
├── from typing import Any, Dict
├── from data.viewer.utils.atomic_displays.mesh.ts.backend.schemas.display_response import MeshDisplayResponse
├── def create_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> MeshDisplayResponse
│   ├── if mesh texture representation is vertex color
│   │   └── calls _create_vertex_color_mesh_display
│   ├── elif mesh texture representation is UV texture map
│   │   └── calls _create_uv_texture_map_mesh_display
│   ├── else
│   │   └── raise unsupported mesh texture representation
│   ├── impls writes the processed mesh resource bytes to output_path
│   └── return MeshDisplayResponse with slot_id, title, url, meta_info from caller-provided args
├── def _create_vertex_color_mesh_display
└── def _create_uv_texture_map_mesh_display
```

`./data/viewer/utils/atomic_displays/mesh/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
├── interface MeshDisplayResponse extends DisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface ColorMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_mesh"                  # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface SegmentationMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "segmentation_mesh"           # common field
│   ├── url                                          # common field — the class-colorized mesh resource
│   └── meta_info                                    # common field
├── interface HeatmapMeshDisplayResponse extends MeshDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "heatmap_mesh"                # common field
│   ├── url                                          # common field — the heatmap-colorized mesh resource
│   └── meta_info                                    # common field
└── interface SparseHeatmapMeshDisplayResponse extends MeshDisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "sparse_heatmap_mesh"         # common field
    ├── url                                          # common field — the sparse heatmap wire resource: a shared-geometry reference plus the sparse (indices, values) delta
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/mesh/ts/frontend/core_mesh_display.ts`

```text
core_mesh_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { MeshDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createThreeDisplayContainer, createThreePerspectiveCamera, createThreeScene, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
├── function renderMeshDisplay({ displayResponse, initialCameraState }: { displayResponse: MeshDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # Renders a self-contained mesh display element initialized at initialCameraState.
│   ├── calls createMeshScene({ displayResponse, initialCameraState })
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderMeshScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createMeshScene({ displayResponse, initialCameraState }: { displayResponse: MeshDisplayResponse; initialCameraState: CameraState | null }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes the mesh-display scene's container, scene, camera, and renderer from displayResponse.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })
│   ├── calls createThreeMesh
│   ├── calls createThreeScene({ object: mesh })
│   ├── calls createThreePerspectiveCamera({ initialCameraState })
│   ├── calls createThreeWebGLRenderer({ container })
│   └── return
├── function createThreeMesh({ displayResponse }: { displayResponse: MeshDisplayResponse }): THREE.Mesh
│   ├── if the url resource is a sparse heatmap resource
│   │   └── impls resolves the (indices, values) delta against the referenced geometry
│   ├── else
│   │   └── impls reads the dense mesh resource from displayResponse.url
│   ├── impls creates THREE.BufferGeometry with position and optional uv / vertex-color BufferAttribute from resolved geometry and meta_info
│   ├── impls creates THREE.MeshBasicMaterial (textured / vertex-colored / fallback uniform color) per displayResponse.meta_info
│   ├── impls creates THREE.Mesh from geometry and material
│   └── return
└── function renderMeshScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/ts/frontend/apis.ts`

```text
apis.ts
├── import type { VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { ColorMeshDisplayResponse, SegmentationMeshDisplayResponse, HeatmapMeshDisplayResponse, SparseHeatmapMeshDisplayResponse } from "./types/display_response";
├── import { renderMeshDisplay } from "./core_mesh_display";
├── function renderColorMeshDisplay({ displayResponse, initialCameraState }: { displayResponse: ColorMeshDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState })
│   └── return
├── function renderSegmentationMeshDisplay({ displayResponse, initialCameraState }: { displayResponse: SegmentationMeshDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # renders backend-colorized mesh display and legend derived from meta_info
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState })
│   └── return
├── function renderHeatmapMeshDisplay({ displayResponse, initialCameraState }: { displayResponse: HeatmapMeshDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # renders backend-colorized mesh display and continuous-palette legend derived from meta_info (scalar min/max)
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState })
│   └── return
└── function renderSparseHeatmapMeshDisplay({ displayResponse, initialCameraState }: { displayResponse: SparseHeatmapMeshDisplayResponse; initialCameraState?: CameraState | null }): VNode
    ├── # renders the sparse heatmap mesh display and continuous-palette legend from meta_info (scalar min/max)
    ├── calls renderMeshDisplay({ displayResponse, initialCameraState })
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/dash/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.atomic_displays.gaussians.dash.core_gaussians_display import create_dash_gaussians_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_gs_display
│   └── calls create_dash_gaussians_display
├── def create_segmentation_gs_display
│   ├── impls reads segmentation Gaussian class ids from segmentation_gs_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_gs_class_ids))
│   ├── calls _map_segmentation_gs_to_rgb(segmentation_gs_path=segmentation_gs_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_gaussians_display
└── def _map_segmentation_gs_to_rgb
```

`./data/viewer/utils/atomic_displays/gaussians/dash/core_gaussians_display.py`

```text
core_gaussians_display.py
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── def create_dash_gaussians_display
│   ├── calls create_dash_gaussians_scene
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_gaussians_component
│   └── return
├── def create_dash_gaussians_scene
│   ├── impls Dash Gaussian-splat display scene from Gaussian data and display metadata
│   └── return
└── def create_dash_gaussians_component
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── class GaussianDisplayResponse(DisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── class ColorGSDisplayResponse(GaussianDisplayResponse)
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_gs"                    # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── class SegmentationGSDisplayResponse(GaussianDisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "segmentation_gs"             # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/gaussians/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.viewer.utils.atomic_displays.gaussians.ts.backend.core_gaussians_display import create_gaussians_display_response
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_gs_display_response
│   ├── # intentional thin wrapper: passes color Gaussian field directly to core response
│   ├── calls create_gaussians_display_response
│   └── return
├── def create_segmentation_gs_display_response
│   ├── # Creates a segmentation Gaussian response from a class-labeled Gaussian resource.
│   ├── impls reads segmentation Gaussian class ids from segmentation_gs_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_gs_class_ids))
│   ├── calls _map_segmentation_gs_to_rgb(segmentation_gs_path=segmentation_gs_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_gs_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_gaussians_display_response
│   └── return
├── def _map_segmentation_gs_to_rgb
└── def _build_segmentation_gs_meta_info
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/ts/backend/core_gaussians_display.py`

```text
core_gaussians_display.py
└── def create_gaussians_display_response
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
├── interface GaussianDisplayResponse extends DisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind                                 # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
├── interface ColorGSDisplayResponse extends GaussianDisplayResponse
│   ├── slot_id                                      # common field
│   ├── title                                        # common field
│   ├── display_kind = "color_gs"                    # common field
│   ├── url                                          # common field
│   └── meta_info                                    # common field
└── interface SegmentationGSDisplayResponse extends GaussianDisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "segmentation_gs"             # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/atomic_displays/gaussians/ts/frontend/apis.ts`

```text
apis.ts
├── import type { VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { ColorGSDisplayResponse, SegmentationGSDisplayResponse } from "./types/display_response";
├── import { renderGaussiansDisplay } from "./core_gaussians_display";
├── function renderColorGSDisplay({ displayResponse, initialCameraState }: { displayResponse: ColorGSDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── calls renderGaussiansDisplay({ displayResponse, initialCameraState })
│   └── return
└── function renderSegmentationGSDisplay({ displayResponse, initialCameraState }: { displayResponse: SegmentationGSDisplayResponse; initialCameraState?: CameraState | null }): VNode
    ├── # renders backend-colorized segmentation display and legend derived from meta_info
    ├── calls renderGaussiansDisplay({ displayResponse, initialCameraState })
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/ts/frontend/core_gaussians_display.ts`

```text
core_gaussians_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { GaussianDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── function renderGaussiansDisplay({ displayResponse, initialCameraState }: { displayResponse: GaussianDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # Renders a self-contained Gaussian display element initialized at initialCameraState.
│   ├── calls createGaussiansScene
│   ├── calls createTrackballCameraControls({ camera, renderer, initialCameraState })
│   ├── calls renderGaussiansScene
│   └── return LeafVNode keyed by displayResponse.url
├── function createGaussiansScene
│   ├── impls Gaussian-splat display scene from DisplayResponse url and meta_info
│   └── return
└── function renderGaussiansScene
    └── return
```

`./data/viewer/utils/atomic_displays/cameras/dash/camera_display.py`

```text
camera_display.py
└── def create_camera_display
```

`./data/viewer/utils/atomic_displays/cameras/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class CameraDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "camera"                      # common field
    ├── url                                          # common field; camera-vis JSON payload URL
    └── meta_info                                    # common field; empty object for camera display
```

`./data/structures/three_d/camera/camera_vis.py`

```text
camera_vis.py
├── from typing import Any, Dict, List, Optional
├── import torch
├── from data.structures.three_d.camera.camera import Camera
├── from data.structures.three_d.camera.cameras import Cameras
├── def camera_vis(camera: Camera, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> Dict[str, Any]
│   ├── # Builds one camera visualization primitive from a Camera whose intrinsics may be absent.
│   ├── impls computes center, center_color, and axes from camera center, right, forward, and up
│   ├── impls computes frustum lines from camera intrinsics and frustum_scale
│   └── return
└── def cameras_vis(cameras: Cameras, frustum_scale: float, frustum_color: Optional[torch.Tensor] = None) -> List[Dict[str, Any]]
    ├── # Builds a camera-trajectory visualization payload from a Cameras collection.
    ├── for each camera
    │   └── calls camera_vis
    └── return
```

`./data/viewer/utils/atomic_displays/cameras/ts/backend/camera_display.py`

```text
camera_display.py
├── from data.structures.three_d.camera.camera_vis import cameras_vis
├── def create_camera_display_response
│   ├── # Creates a camera display response whose URL points at the camera-vis JSON payload.
│   ├── impls loads camera artifact from camera_artifact_path into a Cameras collection
│   ├── calls _build_camera_vis_payload
│   ├── impls exposes the camera-vis JSON payload through a frontend-loadable URL without writing a benchmark camera-visualization artifact to disk
│   ├── impls constructs CameraDisplayResponse with a distinct camera layer slot_id, title, url, and meta_info={}
│   └── return
├── def _build_camera_vis_payload
│   ├── # Converts generic camera visualization primitives into the JSON payload exposed by CameraDisplayResponse.url.
│   ├── calls cameras_vis
│   ├── for each camera-vis entry
│   │   └── calls _serialize_camera_vis_entry
│   └── return
├── def _serialize_camera_vis_entry
│   ├── # Converts one camera-vis entry into the JSON shape consumed by the camera renderer.
│   ├── impls serializes center and center_color
│   ├── for each line in axes
│   │   └── calls _serialize_camera_vis_line
│   ├── for each line in frustum_lines
│   │   └── calls _serialize_camera_vis_line
│   └── return
└── def _serialize_camera_vis_line
    ├── # Converts one camera-vis line segment into plain start, end, and color lists.
    ├── impls serializes start, end, and color
    └── return
```

`./data/viewer/utils/atomic_displays/cameras/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/atomic_displays/utils/ts/frontend/types/display_response";
└── interface CameraDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "camera"                      # common field
    ├── url                                          # common field; camera-vis JSON payload URL
    └── meta_info                                    # common field; empty object for camera display
```

`./data/viewer/utils/atomic_displays/cameras/ts/frontend/camera_display.ts`

```text
camera_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { CameraDisplayResponse } from "./types/display_response";
├── import { createThreeDisplayContainer, createThreePerspectiveCamera, createThreeScene, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
├── function renderCameraDisplay({ displayResponse, initialCameraState }: { displayResponse: CameraDisplayResponse; initialCameraState?: CameraState | null }): VNode
│   ├── # Builds a non-interactive transparent layer from the main-branch camera-vis JSON payload, initialized at initialCameraState.
│   ├── throw if CameraDisplayResponse.meta_info is not an empty object
│   ├── calls createCamerasScene({ displayResponse, initialCameraState })
│   ├── calls renderCamerasScene({ scene, camera, renderer })
│   └── return LeafVNode keyed by displayResponse.url
├── function createCamerasScene({ displayResponse, initialCameraState }: { displayResponse: CameraDisplayResponse; initialCameraState: CameraState | null }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes the cameras-overlay scene's container, scene, camera, and renderer from the camera-vis payload.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: true })
│   ├── calls createThreeCameras
│   ├── calls createThreeScene({ object: cameras })
│   ├── calls createThreePerspectiveCamera({ initialCameraState })
│   ├── calls createThreeWebGLRenderer({ container })
│   └── return
├── function createThreeCameras({ displayResponse }: { displayResponse: CameraDisplayResponse }): THREE.Object3D
│   ├── impls fetches camera-vis JSON payload from displayResponse.url
│   ├── impls validates each payload entry has center, center_color, axes, and frustum_lines
│   ├── impls validates every axes/frustum line has start, end, and color
│   ├── impls builds transparent Three.js centers and line segments from the fetched payload
│   └── return
└── function renderCamerasScene({ scene, camera, renderer }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }): void
    ├── # Drives the render loop; the cameras-overlay has no trackball controls — its camera is externally synced through the camera-sync registry observing the display element's data-camera-state attribute.
    ├── impls exposes the display element under displayResponse.slot_id so the caller can register it as a camera-sync target
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls: null })
    └── return
```

`./data/viewer/utils/camera_state/dash/camera_state.py`

```text
camera_state.py
└── class CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/camera_state/ts/backend/schemas/camera_state.py`

```text
camera_state.py
└── class CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/camera_state/ts/backend/camera_state.py`

```text
camera_state.py
├── from data.structures.three_d.camera import Camera
├── from data.viewer.utils.camera_state.ts.backend.schemas.camera_state import CameraState
└── def create_camera_state_from_camera
    ├── # preserves Camera intrinsics, extrinsics, convention, name, and id
    ├── impls converts Camera to TS backend CameraState schema
    └── return
```

`./data/viewer/utils/camera_state/ts/frontend/types.ts`

```text
types.ts
└── interface CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/camera_controls/dash/trackball_camera_controls.py`

```text
trackball_camera_controls.py
├── def create_dash_trackball_camera_controls
│   ├── calls create_dash_renderer_trackball_camera_controls
│   ├── calls assert_dash_trackball_camera_controls
│   └── return
├── def create_dash_renderer_trackball_camera_controls
│   ├── impls Dash renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu
│   └── return
├── def assert_dash_trackball_camera_controls
│   ├── calls assert_dash_trackball_mouse_mapping
│   ├── calls assert_dash_no_orbit_camera_controls
│   ├── calls assert_dash_no_camera_pose_clamps
│   └── return
├── def assert_dash_trackball_mouse_mapping
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── raise invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── raise context menu blocks trackball panning
│   └── return
├── def assert_dash_no_orbit_camera_controls
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── raise orbit-style camera controls are forbidden
│   └── return
└── def assert_dash_no_camera_pose_clamps
    ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
    │   └── raise restricted camera pose controls
    └── return
```

`./data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls.ts`

```text
trackball_camera_controls.ts
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── export const DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV: number = 45
│   └── # Shared vertical-FOV (degrees) every TS spatial display must construct its THREE.PerspectiveCamera with — 45° is the standard 50mm-equivalent lens FOV, trading perspective realism against off-center foreshortening for the orbit-around-near-scene-content use case this lib targets.
├── interface TrackballCameraControls
│   ├── getCameraState
│   │   └── # serializes the entire camera state (every CameraState field — both intrinsics and extrinsics) into a CameraState
│   ├── applyCameraState
│   │   └── # applies the entire CameraState (every field — both intrinsics and extrinsics) to the underlying camera and controls
│   └── subscribeCameraStateChange
├── function createTrackballCameraControls
│   ├── calls createRendererTrackballCameraControls
│   ├── calls assertTrackballCameraControls
│   ├── if initialCameraState is not null
│   │   └── calls controls.applyCameraState(initialCameraState)
│   ├── impls MutationObserver on container's `data-camera-state` attribute → controls.applyCameraState(parsed state)
│   └── return
├── function createRendererTrackballCameraControls
│   ├── impls renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu
│   └── return
├── function assertTrackballCameraControls
│   ├── calls assertTrackballMouseMapping
│   ├── calls assertNoOrbitCameraControls
│   ├── calls assertNoCameraPoseClamps
│   └── return
├── function assertTrackballMouseMapping
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── throw invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── throw context menu blocks trackball panning
│   └── return
├── function assertNoOrbitCameraControls
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── throw orbit-style camera controls are forbidden
│   └── return
└── function assertNoCameraPoseClamps
    ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
    │   └── throw restricted camera pose controls
    └── return
```

`./data/viewer/utils/camera_sync/dash/camera_sync.py`

```text
camera_sync.py
├── from data.viewer.utils.camera_state.dash.camera_state import CameraState
├── def create_camera_sync_store
│   ├── impls creates Dash store holding a mapping from source id to its CameraSyncState entry (source id, target ids, current camera state)
│   └── return
├── def register_camera_sync_callbacks
│   ├── calls _sync_camera_to_current_targets
│   └── return
├── def _sync_camera_to_current_targets
│   ├── calls _set_camera_state_from_source_camera
│   ├── for each current target id from Dash callback inputs or layout pattern ids registered under the firing source
│   │   ├── if target id is source id
│   │   │   └── continue
│   │   └── calls apply_camera_state_to_target
│   └── return
├── def _set_camera_state_from_source_camera
│   └── return
└── def apply_camera_state_to_target
    ├── impls applies the source's CameraSyncState.camera_state to a Dash spatial-display target registered under that source
    └── return
```

`./data/viewer/utils/camera_sync/ts/frontend/types.ts`

```text
types.ts
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
└── interface CameraSyncState
    ├── source_id    # the source this entry belongs to; one CameraSyncState exists per source
    ├── target_ids   # targets registered under this source
    └── camera_state # this source's current camera state
```

`./data/viewer/utils/camera_sync/ts/frontend/camera_sync.ts`

```text
camera_sync.ts
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { CameraSyncState } from "./types";
└── class CameraSyncRegistry
    ├── # Per-source camera-sync registry: each source_id owns an independent CameraSyncState and target element pool, so apply operations stay confined to their source's own pool.
    ├── _state_by_source_id    # Record<source_id, CameraSyncState> — per-source CameraSyncState entries
    ├── _targets_by_source_id  # Record<source_id, Map<target_id, HTMLElement>> — per-source target element registry
    ├── _listeners             # Array<(camera_sync_state: CameraSyncState) => void>
    ├── loadCameraSyncState
    │   ├── # Common API: seeds one source's CameraSyncState entry from a caller-provided camera state.
    │   ├── impls sets this._state_by_source_id[source_id] to a fresh entry with the caller-provided CameraState and empty target_ids
    │   ├── impls sets this._targets_by_source_id[source_id] to a fresh empty Map
    │   └── return
    ├── getCameraSyncState
    │   ├── # Common API: reads the current committed CameraSyncState for the given source.
    │   └── return this._state_by_source_id[source_id]
    ├── subscribeCameraSyncState
    │   ├── # Additional API: registers listeners that fire on every apply with the updated source's CameraSyncState.
    │   ├── impls appends listener to this._listeners
    │   └── return a callback that removes listener from this._listeners
    ├── registerCameraSyncTarget
    │   ├── # Additional API: registers one display panel as a camera-sync target under a specific source; each source owns its own target pool.
    │   ├── impls idempotently sets this._targets_by_source_id[source_id].set(target_id, target_element)
    │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
    │   ├── calls this._apply_camera_state_to_element  # target_element, this._state_by_source_id[source_id].camera_state
    │   └── return
    ├── unregisterCameraSyncTarget
    │   ├── # Additional API: unregisters one display panel from a source's target set.
    │   ├── impls idempotently deletes this._targets_by_source_id[source_id].delete(target_id)
    │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
    │   └── return
    ├── applyCameraSyncStateToTargets
    │   ├── # Additional API: applies a caller-owned CameraState to every target registered under one source.
    │   ├── impls replaces this._state_by_source_id[source_id] with a new entry carrying current target_ids and the caller-provided CameraState
    │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
    │   │   └── calls this._apply_camera_state_to_element  # target_element, camera_state
    │   ├── calls this._emit_camera_sync_state             # this._state_by_source_id[source_id]
    │   └── return
    ├── applySourceCameraStateToTargets
    │   ├── # Additional API: ingests camera movement from a source display and propagates it to that source's other registered targets.
    │   ├── if source_id not in this._targets_by_source_id
    │   │   └── throw
    │   ├── impls replaces this._state_by_source_id[source_id] with a new entry carrying current target_ids and the source display CameraState
    │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
    │   │   ├── if target_id == source_id
    │   │   │   └── continue
    │   │   └── calls this._apply_camera_state_to_element  # target_element, camera_state
    │   ├── calls this._emit_camera_sync_state             # this._state_by_source_id[source_id]
    │   └── return
    ├── _apply_camera_state_to_element
    │   ├── # Writes a CameraState onto an element's `data-camera-state` attribute; mesh / point-cloud display containers observe this attribute and re-apply to their trackball controls.
    │   └── impls sets target_element.dataset.cameraState to the serialized CameraState (or deletes the attribute when CameraState is null)
    └── _emit_camera_sync_state
        ├── # Notifies every subscriber with the just-updated source's CameraSyncState.
        └── for each listener in this._listeners
            └── impls listener(camera_sync_state)
```

