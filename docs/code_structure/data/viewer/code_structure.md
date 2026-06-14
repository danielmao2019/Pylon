# Data Viewer Code Structure

## 1. Inheritance / type trees

`./data/viewer/utils/atomic_displays/utils/ts/backend/schemas/display_response.py`

Backend modality-specific display response schema files.

```text
class DisplayResponse(BaseModel)
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
    └── # Maps each distinct class id to a deterministic RGB color from a fixed class-color palette.
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
├── from pydantic import BaseModel
└── class DisplayResponse(BaseModel)
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
├── from typing import List, Literal
├── from data.viewer.utils.atomic_displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── RASTER_DISPLAY_KINDS     # frozenset[str]: color_image, depth_image, edge_image, normal_image, segmentation_image, instance_surrogate_image, video — the single source of the raster/spatial taxonomy
├── SPATIAL_DISPLAY_KINDS    # frozenset[str]: color_pc, segmentation_pc, color_gs, segmentation_gs, scene_graph, camera
└── class LayeredDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "layered"                     # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    ├── base_display_response: DisplayResponse                # the single base layer
    ├── aux_display_responses: List[DisplayResponse]          # ordered auxiliary layers stacked on top of the base; each consumer assigns its own per-layer semantics and owns its own visibility state
    ├── layer_class: Literal["raster", "spatial"]            # the single composable class shared by all non-placeholder layers; assigned in model_post_init and serialized so the frontend reads it instead of re-deriving the taxonomy
    ├── def model_post_init [override]
    │   ├── # Pydantic post-construction hook: rejects a layered response whose non-placeholder layers do not all resolve to a single composable class, and records that class as layer_class.
    │   ├── for each layer in base_display_response and aux_display_responses
    │   │   └── calls _display_class_of
    │   ├── if the resolved non-placeholder classes are not all identical
    │   │   └── raise ValueError
    │   ├── impls self.layer_class = the single resolved non-placeholder class
    │   └── return
    └── def _display_class_of
        ├── # Maps a layer's display_kind to "raster", "spatial", or "placeholder", raising for non-layerable text-based kinds.
        ├── if display_kind == "placeholder"
        │   └── return  # passive stand-in, compatible with any class
        ├── elif display_kind in RASTER_DISPLAY_KINDS
        │   └── return  # "raster"
        ├── elif display_kind in SPATIAL_DISPLAY_KINDS
        │   └── return  # "spatial"
        └── else
            └── raise ValueError  # text, table, and other non-layerable kinds
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
    ├── aux_display_responses: DisplayResponse[]
    └── layer_class: "raster" | "spatial"            # backend-stamped (layered_display_response.layer_class); the frontend reads it instead of re-deriving the raster/spatial taxonomy
```

`./data/viewer/utils/atomic_displays/utils/ts/frontend/layered_display_container.ts`

```text
layered_display_container.ts
├── import * as THREE from "three";
├── import type { ElementVNode, VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import { createThreeDisplayContainer, createThreeScene, createThreePerspectiveCamera, createThreeWebGLRenderer, startThreeSceneRenderLoop } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
├── import { createTrackballCameraControls } from "data/viewer/utils/camera_controls/ts/frontend/trackball_camera_controls";
├── export type SpatialLayerContributor = (scene: THREE.Scene) => void   # a spatial layer builds its THREE object(s) into its OWN scene; one scene per layer lets the render loop composite layers without Z-fighting coincident geometry
├── export type LayerSpec = { id: string; visible: boolean; displayClass: "spatial"; contributeToScene: SpatialLayerContributor } | { id: string; visible: boolean; displayClass: "raster"; node: VNode }   # every layer (base + ALL aux) is passed regardless of visibility — `id` identifies it, `visible` is its current toggle state; the consumer rebuilds this list each render with stable `id`s and updated `visible` flags
├── function renderLayeredDisplayContainer({ layers, slotId, initialCameraState }: { layers: readonly LayerSpec[]; slotId: string; initialCameraState: CameraState | null }): VNode
│   ├── # Composes a layered display into ONE WebGL context per cell; routes on the backend-stamped class (layers[0].displayClass, homogeneous per layered_display_response.model_post_init).
│   ├── if layers[0].displayClass == "spatial"
│   │   └── return renderSpatialLayeredContainer({ layers, slotId, initialCameraState })
│   └── if layers[0].displayClass == "raster"
│       └── return renderRasterLayeredContainer({ layers, slotId })
├── function renderSpatialLayeredContainer({ layers, slotId, initialCameraState }: { layers: readonly LayerSpec[]; slotId: string; initialCameraState: CameraState | null }): VNode
│   ├── # Renders the spatial layers into ONE shared context as a LeafVNode keyed by the STABLE slotId (never re-mounts on toggle) — the base layer owns the camera/controls and aux follow it; the visible set rides as a data-visible-layers prop the render loop reads each frame.
│   ├── calls createLayeredSpatialScene({ layers, initialCameraState })                          → { container, baseScene, auxScenes, camera, renderer }
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })  → controls   # the one camera, owned by the base layer; aux carry no controls
│   ├── calls _registerSceneResize({ container, camera, renderer, controls })
│   ├── calls _publishCameraState({ container, controls })                                        # publish the initial base-camera pose
│   ├── impls controls.addEventListener("change", () => _publishCameraState({ container, controls }))   # re-publish on change so the consumer can observe and persist this cell's base-camera pose
│   ├── calls renderLayeredSpatialScene({ container, baseScene, auxScenes, camera, renderer, controls })
│   └── return LeafVNode keyed by slotId with props { "data-visible-layers": layers.filter(l => l.visible).map(l => l.id).join(",") }   # the reconciler mirrors the visible set onto this attribute every render; the render loop reads it back each frame
├── function createLayeredSpatialScene({ layers, initialCameraState }: { layers: readonly LayerSpec[]; initialCameraState: CameraState | null }): { container: HTMLDivElement; baseScene: THREE.Scene; auxScenes: { id: string; scene: THREE.Scene }[]; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes the one shared container/camera/renderer + one THREE.Scene per layer (layers[0] = base, the rest = aux), each populated by its contributor.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })   → container
│   ├── calls createThreePerspectiveCamera({ initialCameraState })              → camera
│   ├── calls createThreeWebGLRenderer({ container })                           → renderer
│   ├── impls baseScene = createThreeScene(); layers[0].contributeToScene(baseScene)
│   ├── impls auxScenes = layers.slice(1).map(l => { scene = createThreeScene(); l.contributeToScene(scene); return { id: l.id, scene } })
│   └── return { container, baseScene, auxScenes, camera, renderer }
├── function renderLayeredSpatialScene({ container, baseScene, auxScenes, camera, renderer, controls }: { container: HTMLElement; baseScene: THREE.Scene; auxScenes: readonly { id: string; scene: THREE.Scene }[]; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Reuses startThreeSceneRenderLoop for the base pass; in onAfterRender re-reads container.dataset.visibleLayers each frame and draws only the listed aux scenes over the base (clearDepth between), so toggling the attribute toggles a layer with no observer.
│   ├── calls startThreeSceneRenderLoop({ scene: baseScene, camera, renderer, controls, onAfterRender: () => { ids = new Set((container.dataset.visibleLayers ?? "").split(",").filter(s => s.length > 0)); renderer.autoClear = false; auxScenes.forEach(({ id, scene }) => { if (ids.has(id)) { renderer.clearDepth(); renderer.render(scene, camera) } }); renderer.autoClear = true } })
│   └── return
├── function renderRasterLayeredContainer({ layers, slotId }: { layers: readonly LayerSpec[]; slotId: string }): VNode
│   ├── # Stacks the VISIBLE raster (2D image/video) layer nodes as absolutely-positioned full-bleed elements; each node is keyed by its layer id so the reconciler adds/removes only the toggled raster layers — DOM stacking has no shared-context cost, so raster needs no one-context observer.
│   ├── impls children = layers.filter(l => l.visible).map(l => ElementVNode div keyed `${slotId}/layer/${l.id}`, style { position: absolute, inset: 0 }, children [l.node])
│   └── return ElementVNode keyed by slotId, props { className: "layered-display-container", style { position: relative, full-bleed } }, children
├── function _registerSceneResize({ container, camera, renderer, controls }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Keeps the renderer size + camera aspect synced to the one shared container via a ResizeObserver — the layered container's own copy of the per-display resize helper.
│   ├── impls resize = () => { camera.aspect = container width/height; camera.updateProjectionMatrix(); renderer.setSize(container width, height, false); controls.handleResize() }
│   ├── impls resize()
│   ├── impls new ResizeObserver(resize).observe(container)
│   └── impls window.addEventListener("resize", resize)
└── function _publishCameraState({ container, controls }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls> }): void
    ├── # Publishes the controls' base-camera state onto the container — dataset.cameraState plus a bubbling camera-pose-change event — so the consumer can observe this cell's base-camera pose (e.g. persist it across mode cells); the layered container's own copy of the per-display publish helper.
    ├── impls cameraState = controls.getCameraState()
    ├── if cameraState is null
    │   └── return
    ├── impls container.dataset.cameraState = JSON.stringify(cameraState)
    └── impls container.dispatchEvent(new CustomEvent("camera-pose-change", { bubbles: true, detail: cameraState }))
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
├── function createThreeScene(): THREE.Scene
│   ├── # Shared empty-scene factory used by every TS atomic spatial display; callers scene.add their own object(s).
│   ├── impls creates THREE.Scene; scene.background stays unset so the renderer's clear color is what gets visibly drawn
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
    ├── # Shared requestAnimationFrame loop driving one base scene each frame; self-stops and releases its WebGL context once its canvas leaves the DOM, so an unmounted cell (mode/scene switch, or a layer-set re-mount) never leaks an active context. onAfterRender lets a caller append a per-frame step (e.g. the layered container's aux passes, or scene_graph's label projection).
    ├── impls wasConnected = false   # the canvas is not appended until after render() returns, so only a later disconnect counts as an unmount
    ├── def draw
    │   ├── # The requestAnimationFrame callback: stops and frees the context once the canvas leaves the DOM, otherwise renders one frame and reschedules itself.
    │   ├── impls connected = renderer.domElement.isConnected; if connected then wasConnected = true
    │   ├── if wasConnected and not connected                                       # canvas detached → the cell was unmounted
    │   │   ├── impls renderer.dispose(); renderer.forceContextLoss()
    │   │   └── return                                                              # stop the loop without rescheduling
    │   ├── if controls is not null
    │   │   └── impls controls.update()
    │   ├── impls renderer.render(scene, camera)
    │   ├── if onAfterRender is provided
    │   │   └── impls onAfterRender()
    │   └── impls window.requestAnimationFrame(draw)
    └── impls window.requestAnimationFrame(draw)
```

`./data/viewer/utils/atomic_displays/points/dash/apis.py`

```text
apis.py
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.atomic_displays.points.dash.core_points_display import create_dash_points_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
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

`./data/viewer/utils/atomic_displays/points/dash/core_points_display.py`

```text
core_points_display.py
├── from typing import Optional
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── DEFAULT_POINT_SIZE_FLOOR = 0.005                            # absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when point_size is not supplied
├── DEFAULT_POINT_SIZE_RATIO = 0.002                            # fraction of point-cloud bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── DEFAULT_POINT_COLOR = "#cccccc"                             # uniform fallback color used when the point cloud has no per-point colors AND the caller does not supply point_color; lib-owned default, overridable
├── def create_dash_points_display(point_cloud: PointCloud, point_size: Optional[float] = None, point_color: Optional[str] = None) -> dcc.Graph
│   ├── # Renders a Dash point-cloud display element; point_size and point_color overrides are opt-in. point_color when supplied replaces per-point colors with a uniform color so the consumer can override the rendered look without rebuilding the data.
│   ├── calls create_dash_points_scene(point_cloud=point_cloud, point_size=point_size, point_color=point_color)
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_points_component
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
└── def create_dash_points_component
    ├── # Assembles the Dash component that hosts the point-cloud scene and its trackball camera controls.
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
├── from typing import Any, Dict, Optional, Tuple
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
├── from data.viewer.utils.atomic_displays.points.ts.backend.core_points_display import create_points_display_response
├── from data.viewer.utils.atomic_displays.points.ts.backend.schemas.display_response import SegmentationPCDisplayResponse
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_pc_display_response
│   ├── # Creates a color point-cloud response from an already colorized point resource.
│   ├── impls point-display meta_info is empty metadata
│   ├── calls create_points_display_response
│   └── return
├── def create_segmentation_pc_display_response(segmentation_pc_path: str, slot_id: str, title: str, class_id_to_rgb: Optional[Dict[int, Tuple[int, int, int]]] = None) -> SegmentationPCDisplayResponse
│   ├── # Creates a segmentation point-cloud response from a class-labeled point resource; the caller may override the class-id → rgb mapping, otherwise the lib computes the default mapping via map_class_ids_to_rgb.
│   ├── calls load_point_cloud
│   ├── impls effective_class_id_to_rgb = class_id_to_rgb if class_id_to_rgb is not None else map_class_ids_to_rgb(class_ids=torch.unique(segmentation_pc.label))
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
├── function renderColorPCDisplay({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: ColorPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string }): VNode
│   ├── # Renders a color point-cloud display with opt-in pointSize and pointColor overrides.
│   ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor })
│   └── return
└── function renderSegmentationPCDisplay({ displayResponse, initialCameraState, pointSize }: { displayResponse: SegmentationPCDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number }): VNode
    ├── # Renders the backend-colorized segmentation display and legend derived from meta_info; per-point colors are already baked in by the backend's class-id → rgb mapping, so no color override is exposed here.
    ├── calls renderPointsDisplay({ displayResponse, initialCameraState, pointSize })
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
├── const DEFAULT_POINT_SIZE_FLOOR = 0.005   # number — absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when pointSize is not supplied
├── const DEFAULT_POINT_SIZE_RATIO = 0.002   # number — fraction of geometry bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── const DEFAULT_POINT_COLOR = "#cccccc"    # hex color — uniform fallback used when geometry has no per-point colors AND the caller does not supply pointColor; lib-owned default, overridable
├── function renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: PointDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string }): VNode
│   ├── # Renders a self-contained point-cloud display element initialized at initialCameraState.
│   ├── calls createPointsScene({ displayResponse, initialCameraState, pointSize, pointColor })
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderPointsScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createPointsScene({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: PointDisplayResponse; initialCameraState: CameraState | null; pointSize?: number; pointColor?: string }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes container, scene, camera, renderer; the THREE.Points is loaded asynchronously and added to the scene when ready.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })                    → container
│   ├── calls createThreePerspectiveCamera({ initialCameraState })                              → camera
│   ├── calls createThreeWebGLRenderer({ container })                                           → renderer
│   ├── calls createThreeScene()                                                 → scene                  # initially empty; THREE.Points joins on async resolve
│   ├── impls loadPointGeometry({ displayResponse }).then(geometry => scene.add(createThreePoints({ geometry, pointSize, pointColor })))
│   └── return { container, scene, camera, renderer }
├── async function loadPointGeometry({ displayResponse }: { displayResponse: PointDisplayResponse }): Promise<THREE.BufferGeometry>
│   ├── # Async-loads the point-cloud resource from displayResponse.url and returns a BufferGeometry with `position` and (when colors are present) `color` attributes.
│   ├── impls assert displayResponse.url !== null
│   ├── impls response = await fetch(displayResponse.url); buffer = await response.arrayBuffer()
│   ├── calls parsePlyBuffer({ buffer })                                                          → geometry
│   └── return geometry
├── function parsePlyBuffer({ buffer }: { buffer: ArrayBuffer }): THREE.BufferGeometry
│   └── # Parses a PLY buffer (ASCII or binary little-endian) into a BufferGeometry with `position` and optional `color` attributes; internal PLY scalar/property parsing is private to this function.
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
│   ├── impls material = new THREE.PointsMaterial({ vertexColors: useVertexColors, size: effectiveSize, ...(effectiveColor !== undefined ? { color: effectiveColor } : {}) })   # constructor literal is exactly these keys; no other constructor key; no post-construction mutation of material
│   └── return new THREE.Points(geometry, material)                                                # no post-construction mutation of points
└── function renderPointsScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── # Drives the point-cloud render loop with the supplied trackball controls.
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/dash/apis.py`

```text
apis.py
├── import torch
├── from dash import dcc
├── from data.viewer.utils.atomic_displays.pixels.dash.core_pixels_display import create_dash_pixels_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"                # color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
├── DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"               # depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
├── DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"                # edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
├── DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"              # normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
├── DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"        # segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
├── DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"  # instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
├── def create_color_image_display(color_image_path: str, image_interpolation: str = DEFAULT_COLOR_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash color-image display from an image path, defaulting to linear interpolation.
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_depth_image_display(depth_image_path: str, image_interpolation: str = DEFAULT_DEPTH_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash depth-image display from a depth-map path, colorizing it through the heatmap palette.
│   ├── calls _map_depth_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_edge_image_display(edge_image_path: str, image_interpolation: str = DEFAULT_EDGE_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash edge-image display from an edge-map path, colorizing it to RGB.
│   ├── calls _map_edge_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_normal_image_display(normal_image_path: str, image_interpolation: str = DEFAULT_NORMAL_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Builds a Dash normal-image display from a normal-map path, colorizing the normal vectors to RGB.
│   ├── calls _map_normal_image_to_rgb
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_segmentation_image_display(segmentation_image_path: str, image_interpolation: str = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Renders the backend-colorized segmentation image display.
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def create_instance_surrogate_image_display(image_path: str, image_interpolation: str = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION) -> dcc.Graph
│   ├── # Renders the backend-colorized instance-surrogate image display.
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_pixels_display(image_interpolation=image_interpolation)
├── def _map_depth_image_to_rgb
│   └── # Maps the depth image to RGB through the continuous heatmap palette for Dash display.
├── def _map_edge_image_to_rgb
│   └── # Maps the edge image to RGB for Dash display.
├── def _map_normal_image_to_rgb
│   └── # Maps the normal vectors to RGB for Dash display.
├── def _map_segmentation_image_to_rgb
│   └── # Maps the segmentation image's per-pixel class ids to RGB via the class-to-RGB mapping for Dash display.
└── def _map_instance_surrogate_image_to_rgb
    ├── # Maps the instance-surrogate class-id image to RGB via the class-to-RGB mapping for Dash display.
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/dash/core_pixels_display.py`

```text
core_pixels_display.py
├── from typing import Any
├── from dash import dcc
└── def create_dash_pixels_display(image: Any, image_interpolation: str) -> dcc.Graph
    ├── # Renders a Dash pixel-image display element from the resolved interpolation choice; modality-agnostic.
    └── return
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
│   └── # Writes a backend-colorized image resource by mapping the depth image through the continuous heatmap palette.
├── def _map_edge_image_to_rgb
│   └── # Writes a backend-colorized image resource by mapping the edge image to RGB.
├── def _map_normal_image_to_rgb
│   └── # Writes a backend-colorized image resource by mapping the normal vectors to RGB.
├── def _map_segmentation_image_to_rgb
│   └── # Writes a backend-colorized image resource by applying the class-to-RGB mapping to the segmentation image.
├── def _build_segmentation_image_meta_info
│   ├── # Builds factual class/color metadata from the class-to-RGB mapping.
│   ├── impls stores `class_id_to_rgb`
│   └── return
├── def _map_instance_surrogate_image_to_rgb
│   └── # Writes a backend-colorized image resource by applying the class-to-RGB mapping to the instance-surrogate class-id image.
└── def _build_instance_surrogate_image_meta_info
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/ts/backend/core_pixels_display.py`

```text
core_pixels_display.py
└── def create_pixels_display_response
    ├── # Creates a pixel-image display response from the loadable image resource path and caller-provided display metadata.
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
├── const DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"                # color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
├── const DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"               # depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
├── const DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"                # edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
├── const DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"              # normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
├── const DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"        # segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
├── const DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"  # instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
├── function renderColorImageDisplay({ displayResponse, imageInterpolation = DEFAULT_COLOR_IMAGE_INTERPOLATION }: { displayResponse: ColorImageDisplayResponse; imageInterpolation?: string }): VNode
│   ├── # Renders a color-image display, defaulting to linear interpolation for natural-image content.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderDepthImageDisplay({ displayResponse, imageInterpolation = DEFAULT_DEPTH_IMAGE_INTERPOLATION }: { displayResponse: DepthImageDisplayResponse; imageInterpolation?: string }): VNode
│   ├── # Renders a depth-image display, defaulting to nearest interpolation to preserve exact metric depths.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderEdgeImageDisplay({ displayResponse, imageInterpolation = DEFAULT_EDGE_IMAGE_INTERPOLATION }: { displayResponse: EdgeImageDisplayResponse; imageInterpolation?: string }): VNode
│   ├── # Renders an edge-image display, defaulting to nearest interpolation to preserve edge crispness.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderNormalImageDisplay({ displayResponse, imageInterpolation = DEFAULT_NORMAL_IMAGE_INTERPOLATION }: { displayResponse: NormalImageDisplayResponse; imageInterpolation?: string }): VNode
│   ├── # Renders a normal-image display, defaulting to nearest interpolation to preserve unit-length normals.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderSegmentationImageDisplay({ displayResponse, imageInterpolation = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION }: { displayResponse: SegmentationImageDisplayResponse; imageInterpolation?: string }): VNode
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
└── function renderInstanceSurrogateImageDisplay({ displayResponse, imageInterpolation = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION }: { displayResponse: InstanceSurrogateImageDisplayResponse; imageInterpolation?: string }): VNode
    ├── # Renders the backend-colorized image display and legend derived from meta_info.
    ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
    └── return
```

`./data/viewer/utils/atomic_displays/pixels/ts/frontend/core_pixels_display.ts`

```text
core_pixels_display.ts
├── import type { LeafVNode, VNode } from "web/reconcile/reconcile";
├── import type { PixelDisplayResponse } from "./types/display_response";
└── function renderPixelsDisplay({ displayResponse, imageInterpolation }: { displayResponse: PixelDisplayResponse; imageInterpolation: string }): VNode
    ├── # Renders a self-contained pixel-image display element from the resolved interpolation choice; modality-agnostic.
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/placeholders/dash/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display
    └── # Builds the Dash missing-result placeholder display from a message.
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
    ├── # Creates a placeholder display response standing in for a missing result, carrying the message inline.
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
    ├── # Renders the missing-result placeholder UI from the response's message.
    ├── impls complete missing-result placeholder UI from PlaceholderDisplayResponse.message
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/videos/dash/video_display.py`

```text
video_display.py
└── def create_video_display
    └── # Builds the Dash video display from a video path.
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
    ├── # Creates a video display response from a loadable video resource.
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
    ├── # Renders the complete video-display UI from the video resource URL.
    ├── impls complete video-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/texts/dash/text_display.py`

```text
text_display.py
└── def create_text_display
    └── # Builds the Dash text display from a text string.
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
    ├── # Creates a text display response carrying the text payload inline.
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
    ├── # Renders the complete text-display UI from the response's text field.
    ├── impls complete text-display UI from TextDisplayResponse.text
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/tables/dash/table_display.py`

```text
table_display.py
└── def create_table_display
    └── # Builds the Dash table display from tabular data.
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
    ├── # Creates a table display response from a loadable table resource.
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
    ├── # Renders the complete table-display UI from the table resource URL.
    ├── impls complete table-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/scene_graphs/dash/scene_graph_display.py`

```text
scene_graph_display.py
└── def create_scene_graph_display
    └── # Builds the Dash scene-graph display from a method-agnostic graph payload.
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
├── import torch
├── from data.viewer.utils.atomic_displays.scene_graphs.ts.backend.schemas.display_response import SceneGraphDisplayResponse
├── def create_scene_graph_display_response(graph_nodes: torch.Tensor, graph_edges: torch.Tensor, object_nodes: torch.Tensor, scene_scale_reference_points: torch.Tensor, slot_id: str, title: str) -> SceneGraphDisplayResponse
│   ├── # Builds the scene-graph base-layer response from a method-agnostic graph payload.
│   ├── calls bake_scene_graph_payload(graph_nodes=graph_nodes, graph_edges=graph_edges, object_nodes=object_nodes, scene_scale_reference_points=scene_scale_reference_points)
│   ├── impls builds frontend resource url pointing at the baked scene-graph payload
│   ├── impls sets meta_info to empty scene-graph metadata
│   └── return SceneGraphDisplayResponse(slot_id=slot_id, title=title, url=url, meta_info=meta_info)
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
├── const DEFAULT_NODE_SIZE = 0.02            # number — heuristic default size for node markers when the caller does not supply nodeSize; lib-owned default, overridable
├── const DEFAULT_EDGE_COLOR = "#888888"      # hex color — neutral gray fallback for edge lines when the payload does not carry an edge color AND the caller does not supply edgeColor; lib-owned default, overridable
├── const DEFAULT_EDGE_WIDTH = 1.0            # number — line width fallback for edges when the caller does not supply edgeWidth; lib-owned default, overridable
├── const DEFAULT_LABEL_FONT_SIZE = 12        # px — font size fallback for overlay labels when the caller does not supply labelFontSize; lib-owned default, overridable
├── const DEFAULT_LABEL_COLOR = "#000000"     # hex color — text color fallback for overlay labels when the caller does not supply labelColor; lib-owned default, overridable
├── function renderSceneGraphDisplay({ displayResponse, initialCameraState, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor }: { displayResponse: SceneGraphDisplayResponse; initialCameraState?: CameraState | null; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string }): VNode
│   ├── # Renders a self-contained scene-graph display: baked node/edge geometry plus HTML label overlay projected per frame.
│   ├── calls createSceneGraphScene({ displayResponse, initialCameraState, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor })
│   ├── calls createTrackballCameraControls({ camera, renderer, initialCameraState })
│   ├── calls renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor })
│   └── return LeafVNode keyed by displayResponse.url
├── function createSceneGraphScene({ displayResponse, initialCameraState, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor }: { displayResponse: SceneGraphDisplayResponse; initialCameraState: CameraState | null; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; labels: object[]; labelOverlay: HTMLDivElement }
│   ├── # Composes container, scene, camera, renderer, label-overlay, and a mutable labels array; payload is loaded asynchronously and Points + labels join on resolve.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })                    → container
│   ├── calls createThreeScene()                                                 → scene                  # initially empty; THREE.Points joins on async resolve
│   ├── calls createThreePerspectiveCamera({ initialCameraState })                              → camera
│   ├── calls createThreeWebGLRenderer({ container })                                           → renderer
│   ├── calls createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor })       → labelOverlay
│   ├── impls labels: object[] = []                                                              # initially empty; mutated on async resolve so renderSceneGraphScene's per-frame projection sees the populated list
│   ├── impls loadSceneGraphPayload({ displayResponse }).then(payload => { const built = createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth }); scene.add(built.points); labels.push(...built.labels); })
│   └── return { container, scene, camera, renderer, labels, labelOverlay }
├── async function loadSceneGraphPayload({ displayResponse }: { displayResponse: SceneGraphDisplayResponse }): Promise<SceneGraphPayload>
│   └── # Async-loads the scene-graph payload from displayResponse.url and returns the parsed payload (node/edge positions + colors + label entries).
├── function createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth }: { payload: SceneGraphPayload; nodeSize?: number; edgeColor?: string; edgeWidth?: number }): { points: THREE.Points; labels: object[] }
│   ├── # Sync-builds THREE.Points + per-frame label data from a pre-loaded payload.
│   ├── impls effectiveNodeSize = nodeSize ?? DEFAULT_NODE_SIZE
│   ├── impls effectiveEdgeWidth = edgeWidth ?? DEFAULT_EDGE_WIDTH
│   ├── if edgeColor !== undefined
│   │   └── impls useEdgeVertexColors = false; effectiveEdgeColor = edgeColor
│   ├── else if payload has per-edge colors
│   │   └── impls useEdgeVertexColors = true; effectiveEdgeColor = undefined
│   ├── else
│   │   └── impls useEdgeVertexColors = false; effectiveEdgeColor = DEFAULT_EDGE_COLOR
│   └── return
├── function createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor }: { container: HTMLDivElement; labelFontSize?: number; labelColor?: string }): HTMLDivElement
│   ├── # Builds the absolutely-positioned HTML overlay container layered above the canvas; labelFontSize / labelColor apply as the overlay's default font-size and color (per-label inline styles still take precedence).
│   ├── impls effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE
│   ├── impls effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR
│   ├── impls absolutely-positioned HTML overlay container layered above the canvas with default font-size = effectiveLabelFontSize px and color = effectiveLabelColor, returned and mounted inside the display container
│   └── return
├── function renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; labels: object[]; labelOverlay: HTMLDivElement; labelFontSize?: number; labelColor?: string }): void
│   ├── # Drives the render + label-projection loop by wrapping the shared startThreeSceneRenderLoop with an onAfterRender step that projects labels each frame.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender: () => _projectLabelsOntoOverlay({ camera, labels, labelOverlay, labelFontSize, labelColor }) })
│   └── return
└── function _projectLabelsOntoOverlay({ camera, labels, labelOverlay, labelFontSize, labelColor }: { camera: THREE.PerspectiveCamera; labels: object[]; labelOverlay: HTMLDivElement; labelFontSize?: number; labelColor?: string }): void
    ├── # Per-frame step: projects each label's world position into overlay-pixel coordinates, updates the HTML node positions and per-label font-size/color, and culls offscreen labels.
    ├── impls effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE
    ├── impls effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR
    ├── impls projects each label's world position to NDC via camera, then converts to overlay-pixel coordinates
    ├── impls updates each label's HTML node position (left/top), font-size = effectiveLabelFontSize px, color = effectiveLabelColor, and culls labels behind the camera or outside the viewport
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/dash/apis.py`

```text
apis.py
├── from typing import Optional
├── import torch
├── from dash import dcc
├── from data.viewer.utils.atomic_displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.atomic_displays.utils.heatmap_colors import map_scalars_to_rgb
├── def create_color_mesh_display(color_mesh_path: str, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # Builds a Dash color mesh display from a mesh path, with opt-in mesh_color, mesh_opacity, and mesh_side overrides.
│   └── calls create_dash_mesh_display(mesh_color=mesh_color, mesh_opacity=mesh_opacity, mesh_side=mesh_side)
├── def create_segmentation_mesh_display(segmentation_mesh_path: str, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # renders backend-colorized segmentation mesh display; per-element colors are already baked in by the backend's class-id → rgb mapping, so no mesh_color override is exposed here.
│   ├── impls reads segmentation mesh class ids from segmentation_mesh_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(segmentation_mesh_path=segmentation_mesh_path, class_id_to_rgb=class_id_to_rgb)
│   └── calls create_dash_mesh_display(mesh_opacity=mesh_opacity, mesh_side=mesh_side)
├── def create_heatmap_mesh_display(heatmap_mesh_path: str, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # renders backend-colorized heatmap mesh display; per-element colors are already baked in by the backend's scalar → rgb mapping, so no mesh_color override is exposed here.
│   ├── impls reads heatmap mesh scalar values from heatmap_mesh_path (per-vertex 1-D or per-texel 2-D, non-negative)
│   ├── calls map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
│   ├── calls _map_heatmap_mesh_to_rgb(heatmap_mesh_path=heatmap_mesh_path, scalar_rgb=scalar_rgb)
│   └── calls create_dash_mesh_display(mesh_opacity=mesh_opacity, mesh_side=mesh_side)
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
├── from typing import Any, Optional
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── DEFAULT_MESH_COLOR = "#cccccc"                             # uniform fallback color used when geometry has no texture AND has no per-vertex colors AND the caller does not supply mesh_color; lib-owned default, overridable
├── DEFAULT_MESH_OPACITY = 1.0                                 # opaque default applied when the caller does not supply mesh_opacity; lib-owned default, overridable
├── DEFAULT_MESH_SIDE = "double"                               # fallback side mode for visibility under arbitrary camera framings when the caller does not supply mesh_side; lib-owned default, overridable
├── def create_dash_mesh_display(mesh: Any, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> dcc.Graph
│   ├── # Renders a Dash mesh display element with trackball camera controls; mesh_color, mesh_opacity, and mesh_side overrides are opt-in.
│   ├── calls create_dash_mesh_scene(mesh=mesh, mesh_color=mesh_color, mesh_opacity=mesh_opacity, mesh_side=mesh_side)
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_mesh_component
│   └── return
├── def create_dash_mesh_scene(mesh: Any, mesh_color: Optional[str] = None, mesh_opacity: Optional[float] = None, mesh_side: Optional[str] = None) -> go.Mesh3d
│   ├── # Sync-builds the Plotly Mesh3d trace from the mesh.
│   ├── impls effective_opacity = mesh_opacity if mesh_opacity is not None else DEFAULT_MESH_OPACITY
│   ├── impls effective_side = mesh_side if mesh_side is not None else DEFAULT_MESH_SIDE
│   ├── if mesh texture representation is vertex color
│   │   ├── calls _create_dash_vertex_color_mesh_scene(mesh=mesh, mesh_color=mesh_color, effective_opacity=effective_opacity, effective_side=effective_side)
│   │   └── return
│   ├── elif mesh texture representation is UV texture map
│   │   ├── calls _create_dash_uv_texture_map_mesh_scene(mesh=mesh, mesh_color=mesh_color, effective_opacity=effective_opacity, effective_side=effective_side)
│   │   └── return
│   └── else
│       └── raise unsupported mesh texture representation
├── def _create_dash_vertex_color_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Builds the Plotly Mesh3d trace for a per-vertex-colored mesh, resolving the effective color.
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── elif mesh has per-vertex rgb
│   │   └── impls effective_color = mesh.per_vertex_rgb
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
├── def _create_dash_uv_texture_map_mesh_scene(mesh: Any, mesh_color: Optional[str], effective_opacity: float, effective_side: str) -> go.Mesh3d
│   ├── # Builds the Plotly Mesh3d trace for a UV-texture-mapped mesh, resolving the effective color.
│   ├── if mesh_color is not None
│   │   └── impls effective_color = mesh_color
│   ├── elif mesh has uv_texture_map
│   │   └── impls effective_color = sample(mesh.uv_texture_map, mesh.uv)
│   ├── else
│   │   └── impls effective_color = DEFAULT_MESH_COLOR
│   └── return
└── def create_dash_mesh_component
    ├── # Assembles the Dash component that hosts the Mesh3d scene and its trackball camera controls.
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
│   ├── # Writes the processed mesh resource to output_path and returns the mesh display response, dispatching on the mesh texture representation.
│   ├── if mesh texture representation is vertex color
│   │   └── calls _create_vertex_color_mesh_display_response
│   ├── elif mesh texture representation is UV texture map
│   │   └── calls _create_uv_texture_map_mesh_display_response
│   ├── else
│   │   └── raise unsupported mesh texture representation
│   ├── impls writes the processed mesh resource bytes to output_path
│   └── return MeshDisplayResponse with slot_id, title, url, meta_info from caller-provided args
├── def _create_vertex_color_mesh_display_response
│   └── # Builds the mesh display response for a per-vertex-colored mesh.
└── def _create_uv_texture_map_mesh_display_response
    └── # Builds the mesh display response for a UV-texture-mapped mesh.
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
├── const DEFAULT_MESH_COLOR = "#cccccc"          # hex color — uniform fallback used when geometry has no texture AND has no vertex colors AND the caller does not supply meshColor; lib-owned default, overridable
├── const DEFAULT_MESH_OPACITY = 1.0              # number — opaque default applied when the caller does not supply meshOpacity; material's `transparent` flag flips true automatically when opacity is less than 1; lib-owned default, overridable
├── const DEFAULT_MESH_SIDE = THREE.DoubleSide    # THREE.Side — fallback side mode for visibility under arbitrary camera framings when the caller does not supply meshSide; lib-owned default, overridable
├── function renderMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide }: { displayResponse: MeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): VNode
│   ├── # Renders a self-contained mesh display element initialized at initialCameraState.
│   ├── calls createMeshScene({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide })
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderMeshScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createMeshScene({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide }: { displayResponse: MeshDisplayResponse; initialCameraState: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes container, scene, camera, renderer; mesh payload is loaded asynchronously and THREE.Mesh joins the scene on resolve.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })                    → container
│   ├── calls createThreePerspectiveCamera({ initialCameraState })                              → camera
│   ├── calls createThreeWebGLRenderer({ container })                                           → renderer
│   ├── calls createThreeScene()                                                 → scene                  # initially empty; THREE.Mesh joins on async resolve
│   ├── impls loadMeshPayload({ displayResponse }).then(payload => scene.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })))
│   └── return { container, scene, camera, renderer }
├── async function loadMeshPayload({ displayResponse }: { displayResponse: MeshDisplayResponse }): Promise<MeshPayload>
│   ├── # Async-loads the mesh payload from displayResponse.url; resolves a sparse-heatmap delta against its referenced geometry, otherwise reads the dense resource as-is.
│   ├── if the url resource is a sparse heatmap resource
│   │   └── impls resolves the (indices, values) delta against the referenced geometry into a per-vertex RGBA color payload — vertices in `indices` carry their scalar→rgb heatmap color at alpha 1; every other vertex carries alpha 0 — so a sparse heatmap renders as an overlay that reveals the base layer beneath outside the delta, not a full opaque mesh  → payload
│   ├── else
│   │   └── impls reads the dense mesh resource from displayResponse.url               → payload
│   └── return payload
├── function createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide }: { payload: MeshPayload; displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): THREE.Mesh
│   ├── # Sync-builds THREE.BufferGeometry + THREE.MeshBasicMaterial + THREE.Mesh from a pre-loaded payload.
│   ├── impls effectiveOpacity = meshOpacity ?? DEFAULT_MESH_OPACITY
│   ├── impls effectiveSide = meshSide ?? DEFAULT_MESH_SIDE
│   ├── if meshColor !== undefined
│   │   └── impls useTexture = false; useVertexColors = false; effectiveColor = meshColor
│   ├── else if payload has uv texture map
│   │   └── impls useTexture = true; useVertexColors = false; effectiveColor = undefined
│   ├── else if payload has vertex colors
│   │   └── impls useTexture = false; useVertexColors = true; effectiveColor = undefined
│   ├── else
│   │   └── impls useTexture = false; useVertexColors = false; effectiveColor = DEFAULT_MESH_COLOR
│   ├── impls material = new THREE.MeshBasicMaterial({ vertexColors: useVertexColors, side: effectiveSide, opacity: effectiveOpacity, transparent: effectiveOpacity < 1 || (useVertexColors && payload colors carry per-vertex alpha), ...(useTexture ? { map: payload.texture } : {}), ...(effectiveColor !== undefined ? { color: effectiveColor } : {}) })   # constructor literal is exactly these keys; vertexColors honors a 4-component (RGBA) color attribute so an overlay payload's alpha-0 vertices render fully transparent and reveal the layer beneath; no other constructor key; no post-construction mutation of material
│   └── return new THREE.Mesh(geometry, material)                                                # no post-construction mutation of mesh
└── function renderMeshScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── # Drives the mesh render loop with the supplied trackball controls.
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`./data/viewer/utils/atomic_displays/mesh/ts/frontend/apis.ts`

```text
apis.ts
├── import * as THREE from "three";
├── import type { VNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/camera_state/ts/frontend/types";
├── import type { ColorMeshDisplayResponse, SegmentationMeshDisplayResponse, HeatmapMeshDisplayResponse, SparseHeatmapMeshDisplayResponse } from "./types/display_response";
├── import { renderMeshDisplay } from "./core_mesh_display";
├── function renderColorMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide }: { displayResponse: ColorMeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): VNode
│   ├── # Renders a color mesh display with opt-in meshColor, meshOpacity, and meshSide overrides.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide })
│   └── return
├── function renderSegmentationMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: SegmentationMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): VNode
│   ├── # renders backend-colorized mesh display and legend derived from meta_info; per-element colors are already baked in by the backend's class-id → rgb mapping, so no meshColor override is exposed here.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
├── function renderHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: HeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): VNode
│   ├── # renders backend-colorized mesh display and continuous-palette legend derived from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping, so no meshColor override is exposed here.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
└── function renderSparseHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: SparseHeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): VNode
    ├── # renders the sparse heatmap mesh display and continuous-palette legend from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping, so no meshColor override is exposed here.
    ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/dash/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.atomic_displays.gaussians.dash.core_gaussians_display import create_dash_gaussians_display
├── from data.viewer.utils.atomic_displays.utils.class_colors import map_class_ids_to_rgb
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
    └── # Recolors the segmentation Gaussian's per-Gaussian class ids to RGB via the class-to-RGB mapping.
```

`./data/viewer/utils/atomic_displays/gaussians/dash/core_gaussians_display.py`

```text
core_gaussians_display.py
├── from data.viewer.utils.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
├── def create_dash_gaussians_display
│   ├── # Renders a Dash Gaussian-splat display element with trackball camera controls.
│   ├── calls create_dash_gaussians_scene
│   ├── calls create_dash_trackball_camera_controls
│   ├── calls create_dash_gaussians_component
│   └── return
├── def create_dash_gaussians_scene
│   ├── # Builds the Dash Gaussian-splat display scene from Gaussian data and display metadata.
│   ├── impls Dash Gaussian-splat display scene from Gaussian data and display metadata
│   └── return
└── def create_dash_gaussians_component
    ├── # Assembles the Dash component that hosts the Gaussian-splat scene and its trackball camera controls.
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
│   └── # Writes a backend-colorized Gaussian resource by applying the class-to-RGB mapping to the segmentation Gaussian's class ids.
└── def _build_segmentation_gs_meta_info
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`./data/viewer/utils/atomic_displays/gaussians/ts/backend/core_gaussians_display.py`

```text
core_gaussians_display.py
└── def create_gaussians_display_response
    ├── # Creates a Gaussian display response from the loadable Gaussian resource path and caller-provided display metadata.
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
│   ├── # Renders a color Gaussian-splat display from an already-colorized Gaussian resource.
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
├── import { createThreeDisplayContainer } from "data/viewer/utils/atomic_displays/utils/ts/frontend/three_scene_helpers";
└── function renderGaussiansDisplay({ displayResponse, initialCameraState }: { displayResponse: GaussianDisplayResponse; initialCameraState?: CameraState | null }): VNode
    ├── # Delegates rendering to the external Gaussian-splat package; the package owns URL loading, scene assembly, camera controls, and the render loop.
    ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: false })                    → container
    ├── impls invoke the external Gaussian-splat package's mount API with { container, url: displayResponse.url, initialCameraState, meta_info: displayResponse.meta_info }   # the external package handles fetch + parse + scene + camera + controls + render loop internally; the wrapper does not duplicate any of those concerns
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/atomic_displays/cameras/dash/camera_display.py`

```text
camera_display.py
└── def create_camera_display
    └── # Builds the Dash camera-trajectory display from a loaded camera artifact.
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

`./data/viewer/utils/atomic_displays/cameras/ts/backend/camera_display.py`

```text
camera_display.py
├── from data.structures.three_d.camera.camera_vis import cameras_vis
├── from data.structures.three_d.camera.cameras import Cameras
├── def create_camera_display_response
│   ├── # Creates a camera display response whose URL points at the camera-vis JSON payload.
│   ├── calls _load_opencv_c2w_cameras
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
├── def _serialize_camera_vis_line
│   ├── # Converts one camera-vis line segment into plain start, end, and color lists.
│   ├── impls serializes start, end, and color
│   └── return
├── def _load_opencv_c2w_cameras
│   ├── # Loads the camera display's on-disk camera artifact (one 4x4 OpenCV camera-to-world matrix per row, 16 floats) into a CPU Cameras collection in the opencv convention.
│   ├── for each non-empty pose row
│   │   ├── impls parses 16 floats into a 4x4 camera-to-world matrix
│   │   └── calls _stabilize_rotation
│   ├── calls Cameras
│   └── return
└── def _stabilize_rotation
    ├── # Projects one near-rotation matrix onto SO(3) via SVD so a parsed camera-to-world rotation is a proper rotation with determinant +1.
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
├── const DEFAULT_FRUSTUM_COLOR = "#888888"        # hex color — last-resort default frustum line color used when the per-camera payload entry does not carry a color AND the caller does not supply frustumColor; per-entry payload colors still take precedence over frustumColor; lib-owned default, overridable
├── const DEFAULT_FRUSTUM_OPACITY = 0.5            # number — transparent frustum overlay default applied when the caller does not supply frustumOpacity; lib-owned default, overridable
├── const DEFAULT_CENTER_MARKER_SIZE = 0.01        # number — marker size for the camera center point used when the caller does not supply centerMarkerSize; lib-owned default, overridable
├── function renderCameraDisplay({ displayResponse, initialCameraState, frustumColor, frustumOpacity, centerMarkerSize }: { displayResponse: CameraDisplayResponse; initialCameraState?: CameraState | null; frustumColor?: string; frustumOpacity?: number; centerMarkerSize?: number }): VNode
│   ├── # Builds a non-interactive transparent layer from the main-branch camera-vis JSON payload, initialized at initialCameraState.
│   ├── throw if CameraDisplayResponse.meta_info is not an empty object
│   ├── calls createCamerasScene({ displayResponse, initialCameraState, frustumColor, frustumOpacity, centerMarkerSize })
│   ├── calls renderCamerasScene({ scene, camera, renderer })
│   └── return LeafVNode keyed by displayResponse.url
├── function createCamerasScene({ displayResponse, initialCameraState, frustumColor, frustumOpacity, centerMarkerSize }: { displayResponse: CameraDisplayResponse; initialCameraState: CameraState | null; frustumColor?: string; frustumOpacity?: number; centerMarkerSize?: number }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Composes container, scene, camera, renderer; camera-vis payload is loaded asynchronously and the cameras Object3D joins the scene on resolve.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed: true })                     → container
│   ├── calls createThreeScene()                                                 → scene                  # initially empty; cameras Object3D joins on async resolve
│   ├── calls createThreePerspectiveCamera({ initialCameraState })                              → camera
│   ├── calls createThreeWebGLRenderer({ container })                                           → renderer
│   ├── impls loadCamerasPayload({ displayResponse }).then(payload => scene.add(createThreeCameras({ payload, frustumColor, frustumOpacity, centerMarkerSize })))
│   └── return { container, scene, camera, renderer }
├── async function loadCamerasPayload({ displayResponse }: { displayResponse: CameraDisplayResponse }): Promise<CamerasPayload>
│   └── # Async-loads the camera-vis JSON payload from displayResponse.url and validates each entry has center / center_color / axes / frustum_lines and that every axes/frustum line carries start / end / color; returns the validated payload.
├── function createThreeCameras({ payload, frustumColor, frustumOpacity, centerMarkerSize }: { payload: CamerasPayload; frustumColor?: string; frustumOpacity?: number; centerMarkerSize?: number }): THREE.Object3D
│   ├── # Sync-builds the transparent Three.js centers + line segments from a pre-validated camera-vis payload.
│   ├── impls effectiveCenterMarkerSize = centerMarkerSize ?? DEFAULT_CENTER_MARKER_SIZE
│   ├── impls effectiveFrustumOpacity = frustumOpacity ?? DEFAULT_FRUSTUM_OPACITY
│   ├── for each entry in payload
│   │   ├── if entry.frustum_lines carries per-line color
│   │   │   └── impls effectiveFrustumColor = entry frustum_lines color
│   │   ├── elif frustumColor !== undefined
│   │   │   └── impls effectiveFrustumColor = frustumColor
│   │   └── else
│   │       └── impls effectiveFrustumColor = DEFAULT_FRUSTUM_COLOR
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
│   ├── # Builds and validates the Dash trackball controls that every 3D Dash spatial display must use.
│   ├── calls create_dash_renderer_trackball_camera_controls
│   ├── calls assert_dash_trackball_camera_controls
│   └── return
├── def create_dash_renderer_trackball_camera_controls
│   ├── # Constructs the Dash renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls Dash renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu
│   └── return
├── def assert_dash_trackball_camera_controls
│   ├── # Validates the constructed Dash controls satisfy every trackball contract by running the mouse-mapping, no-orbit, and no-pose-clamp assertions.
│   ├── calls assert_dash_trackball_mouse_mapping
│   ├── calls assert_dash_no_orbit_camera_controls
│   ├── calls assert_dash_no_camera_pose_clamps
│   └── return
├── def assert_dash_trackball_mouse_mapping
│   ├── # Asserts the Dash controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── raise invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── raise context menu blocks trackball panning
│   └── return
├── def assert_dash_no_orbit_camera_controls
│   ├── # Asserts the Dash controls do not use forbidden orbit-style target-locked camera semantics.
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── raise orbit-style camera controls are forbidden
│   └── return
└── def assert_dash_no_camera_pose_clamps
    ├── # Asserts the Dash controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
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
│   ├── # Builds, validates, and returns the trackball controls, seeding them from initialCameraState and observing the container's data-camera-state attribute for external sync.
│   ├── calls createRendererTrackballCameraControls
│   ├── calls assertTrackballCameraControls
│   ├── if initialCameraState is not null
│   │   └── calls controls.applyCameraState(initialCameraState)
│   ├── impls MutationObserver on container's `data-camera-state` attribute → controls.applyCameraState(parsed state)
│   └── return
├── function createRendererTrackballCameraControls
│   ├── # Constructs the renderer-specific trackball controls wiring left-drag rotate, right-drag pan, wheel zoom, and context-menu suppression.
│   ├── impls renderer-specific trackball camera controls with left-button rotation, right-button panning, mouse-wheel zoom, and suppressed canvas context menu
│   └── return
├── function assertTrackballCameraControls
│   ├── # Validates the constructed controls satisfy every trackball contract by running the mouse-mapping, no-orbit, and no-pose-clamp assertions.
│   ├── calls assertTrackballMouseMapping
│   ├── calls assertNoOrbitCameraControls
│   ├── calls assertNoCameraPoseClamps
│   └── return
├── function assertTrackballMouseMapping
│   ├── # Asserts the controls map left-drag to rotate, right-drag to pan, and wheel to zoom, and that the canvas suppresses its context menu.
│   ├── if controls do not map left-button drag to rotation, right-button drag to panning, and mouse-wheel scroll to zoom
│   │   └── throw invalid trackball camera controls
│   ├── if viewer canvas does not suppress the default browser context menu
│   │   └── throw context menu blocks trackball panning
│   └── return
├── function assertNoOrbitCameraControls
│   ├── # Asserts the controls do not use forbidden orbit-style target-locked camera semantics.
│   ├── if controls use orbit-style target-locked camera semantics
│   │   └── throw orbit-style camera controls are forbidden
│   └── return
└── function assertNoCameraPoseClamps
    ├── # Asserts the controls impose no camera-pose restriction on polar angle, azimuth angle, target lock, distance, pan, translation, or rotation.
    ├── if controls restrict polar angle, azimuth angle, target lock, distance bounds, pan, translation, or rotation
    │   └── throw restricted camera pose controls
    └── return
```

`./data/viewer/utils/camera_sync/dash/camera_sync.py`

```text
camera_sync.py
├── def create_camera_sync_store
│   ├── # Creates the Dash store that holds the per-source camera-sync registry keyed by source id.
│   ├── impls creates Dash store holding a mapping from source id to its CameraSyncState entry (source id, target ids, current camera state)
│   └── return
├── def register_camera_sync_callbacks
│   ├── # Registers the Dash callbacks that observe each source display's camera and fan its state out to its targets.
│   ├── calls _sync_camera_to_current_targets
│   └── return
├── def _sync_camera_to_current_targets
│   ├── # Dash callback body that commits the firing source's camera and pushes it to every other target registered under that source.
│   ├── calls _set_camera_state_from_source_camera
│   ├── for each current target id from Dash callback inputs or layout pattern ids registered under the firing source
│   │   ├── if target id is source id
│   │   │   └── continue
│   │   └── calls apply_camera_state_to_target
│   └── return
├── def _set_camera_state_from_source_camera
│   ├── # Commits the firing source display's current camera state into that source's CameraSyncState entry in the store.
│   └── return
└── def apply_camera_state_to_target
    ├── # Applies one source's current camera state to a single registered Dash spatial-display target.
    ├── impls applies the source's CameraSyncState.camera_state to a Dash spatial-display target registered under that source
    └── return
```

`./data/viewer/utils/camera_sync/ts/frontend/types.ts`

```text
types.ts
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
├── class CameraSyncRegistry
│   ├── # Per-source camera-sync registry: each source_id owns an independent CameraSyncState and target element pool, so apply operations stay confined to their source's own pool.
│   ├── _state_by_source_id    # Record<source_id, CameraSyncState> — per-source CameraSyncState entries
│   ├── _targets_by_source_id  # Record<source_id, Map<target_id, HTMLElement>> — per-source target element registry
│   ├── _listeners             # Array<(camera_sync_state: CameraSyncState) => void>
│   ├── loadCameraSyncState
│   │   ├── # Common API: seeds one source's CameraSyncState entry from a caller-provided camera state.
│   │   ├── impls sets this._state_by_source_id[source_id] to a fresh entry with the caller-provided CameraState and empty target_ids
│   │   ├── impls sets this._targets_by_source_id[source_id] to a fresh empty Map
│   │   └── return
│   ├── getCameraSyncState
│   │   ├── # Common API: reads the current committed CameraSyncState for the given source.
│   │   └── return this._state_by_source_id[source_id]
│   ├── subscribeCameraSyncState
│   │   ├── # Additional API: registers listeners that fire on every apply with the updated source's CameraSyncState.
│   │   ├── impls appends listener to this._listeners
│   │   └── return a callback that removes listener from this._listeners
│   ├── registerCameraSyncTarget
│   │   ├── # Additional API: registers one display panel as a camera-sync target under a specific source; each source owns its own target pool.
│   │   ├── impls idempotently sets this._targets_by_source_id[source_id].set(target_id, target_element)
│   │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
│   │   ├── calls this._apply_camera_state_to_element  # target_element, this._state_by_source_id[source_id].camera_state
│   │   └── return
│   ├── unregisterCameraSyncTarget
│   │   ├── # Additional API: unregisters one display panel from a source's target set.
│   │   ├── impls idempotently deletes this._targets_by_source_id[source_id].delete(target_id)
│   │   ├── impls updates this._state_by_source_id[source_id].target_ids from this._targets_by_source_id[source_id].keys()
│   │   └── return
│   ├── applyCameraSyncStateToTargets
│   │   ├── # Additional API: applies a caller-owned CameraState to every target registered under one source.
│   │   ├── impls replaces this._state_by_source_id[source_id] with a new entry carrying current target_ids and the caller-provided CameraState
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   └── calls this._apply_camera_state_to_element  # target_element, camera_state
│   │   ├── calls this._emit_camera_sync_state             # this._state_by_source_id[source_id]
│   │   └── return
│   ├── applySourceCameraStateToTargets
│   │   ├── # Additional API: ingests camera movement from a source display and propagates it to that source's other registered targets.
│   │   ├── if source_id not in this._targets_by_source_id
│   │   │   └── throw
│   │   ├── impls replaces this._state_by_source_id[source_id] with a new entry carrying current target_ids and the source display CameraState
│   │   ├── for each (target_id, target_element) in this._targets_by_source_id[source_id]
│   │   │   ├── if target_id == source_id
│   │   │   │   └── continue
│   │   │   └── calls this._apply_camera_state_to_element  # target_element, camera_state
│   │   ├── calls this._emit_camera_sync_state             # this._state_by_source_id[source_id]
│   │   └── return
│   ├── _apply_camera_state_to_element
│   │   ├── # Writes a CameraState onto an element's `data-camera-state` attribute; mesh / point-cloud display containers observe this attribute and re-apply to their trackball controls.
│   │   └── impls sets target_element.dataset.cameraState to the serialized CameraState (or deletes the attribute when CameraState is null)
│   └── _emit_camera_sync_state
│       ├── # Notifies every subscriber with the just-updated source's CameraSyncState.
│       └── for each listener in this._listeners
│           └── impls listener(camera_sync_state)
└── const cameraSyncRegistry = new CameraSyncRegistry()    # the single document-global registry instance shared by every spatial display in the document; consumers import this instance and call its methods
```

