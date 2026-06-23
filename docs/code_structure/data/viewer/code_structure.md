# Data Viewer Code Structure

## 1. Inheritance / type trees

`./data/viewer/utils/displays/utils/ts/backend/schemas/display_response.py`

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
├── class Aabb3dDisplayResponse
├── class Aabb2dDisplayResponse
├── class PlaceholderDisplayResponse
└── class LayeredDisplayResponse
```

`./data/viewer/utils/displays/utils/ts/frontend/types/display_response.ts`

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
├── interface Aabb3dDisplayResponse
├── interface Aabb2dDisplayResponse
├── interface PlaceholderDisplayResponse
└── interface LayeredDisplayResponse
```

## 2. Code structure trees

Files below are grouped by folder structure; within a runtime folder, API/caller files appear before core/helper files when call order matters.

The base atomic `DisplayResponse` is owned by `./data/viewer/utils/displays/utils/ts/`; each modality-specific response inherits from that base under the matching `./data/viewer` modality.
`display_kind` selects the atomic renderer, `url` and typed response fields identify loadable resources, and `meta_info` carries renderer-owned loading hints plus display statistics/details such as class/color metadata.
`meta_info` must not encode primary display payloads, rendered legends, presentation objects, or artifact availability state such as `available` or `missing`.
Backend `data.viewer` camera-display code loads the selected camera artifact, interprets camera conventions, and prepares the camera-vis JSON payload exposed through `CameraDisplayResponse.url`.
`CameraDisplayResponse.meta_info` is empty because the camera-vis JSON payload is the camera display payload.
That payload is the main-branch camera visualization contract: a camera trajectory list whose entries preserve the `camera_vis()` semantics of `center`, `center_color`, `center_size`, `axes`, and `frustum_lines`, with every line carrying `start`, `end`, and `color`.
`camera_vis()` owns construction of one camera's visual primitive: `frustum_size` is the world-unit size of the camera frustum glyph's frustum + axis lines, and camera intrinsics shape the frustum.
Missing intrinsics normalized to an identity matrix naturally produce the default frustum at the same `frustum_size`; no separate intrinsics-provenance field is needed for camera display geometry.
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

`./data/viewer/utils/displays/utils/class_colors.py`

```text
class_colors.py
├── from typing import Dict, Tuple
├── import torch
└── def map_class_ids_to_rgb(class_ids: torch.Tensor) -> Dict[int, Tuple[int, int, int]]
    └── # Maps each distinct class id to a deterministic RGB color from a fixed class-color palette.
```

`./data/viewer/utils/displays/utils/heatmap_colors.py`

```text
heatmap_colors.py
├── import torch
└── def map_scalars_to_rgb(scalars: torch.Tensor) -> torch.Tensor
    ├── # Maps non-negative scalars to RGB via a fixed continuous heatmap palette.
    ├── assert scalars is non-negative
    └── return torch.Tensor of shape (*scalars.shape, 3)
```

`./data/viewer/utils/displays/utils/ts/backend/schemas/display_response.py`

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

`./data/viewer/utils/displays/utils/ts/frontend/types/display_response.ts`

```text
display_response.ts
└── interface DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind                                 # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/utils/ts/backend/schemas/layered_display_response.py`

```text
layered_display_response.py
├── from typing import List, Literal
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
├── RASTER_DISPLAY_KINDS     # frozenset[str]: color_image, depth_image, edge_image, normal_image, segmentation_image, instance_surrogate_image, video, aabb_2d — the single source of the raster/spatial taxonomy
├── SPATIAL_DISPLAY_KINDS    # frozenset[str]: color_pc, segmentation_pc, color_gs, segmentation_gs, scene_graph, camera, aabb_3d
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

`./data/viewer/utils/displays/utils/ts/frontend/types/layered_display_response.ts`

```text
layered_display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
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

`./data/viewer/utils/displays/utils/ts/frontend/layered_display_container.ts`

```text
layered_display_container.ts
├── import * as THREE from "three";
├── import { reconcileInto } from "web/reconcile/reconcile";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { LayeredDisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/layered_display_response";
├── import { getSpatialLayerRenderer, getRasterLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── import "data/viewer/utils/displays/utils/ts/frontend/register_layer_renderers";   # side-effect: eager-glob-loads every modality so its self-registration populates the registry before any render
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop, attachThreeScenePickSeam } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── function renderLayeredDisplay({ layeredDisplayResponse, initialCameraState }: { layeredDisplayResponse: LayeredDisplayResponse; initialCameraState: CameraState | null }): LeafVNode
│   ├── # Composes one layered display response into a shared spatial WebGL scene or a stacked raster DOM container per cell, routing on the backend-stamped layer_class.
│   ├── if layeredDisplayResponse.layer_class == "spatial"
│   │   └── return renderLayeredSpatialDisplay({ layeredDisplayResponse, initialCameraState })
│   └── if layeredDisplayResponse.layer_class == "raster"
│       └── return renderLayeredRasterDisplay({ layeredDisplayResponse })
├── function renderLayeredSpatialDisplay({ layeredDisplayResponse, initialCameraState }: { layeredDisplayResponse: LayeredDisplayResponse; initialCameraState: CameraState | null }): LeafVNode
│   ├── # Renders the base + aux spatial layers into one shared scene/camera as a slot_id-keyed LeafVNode, the shared camera owning the framing and the additive pick seam.
│   ├── calls createSpatialDisplayScene({ initialCameraState })                                     → { container, scene, camera, renderer }
│   ├── calls createLayerObjects({ layeredDisplayResponse })                                        → layerObjects
│   ├── impls layerObjects.forEach(object => scene.add(object))
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })    → controls   # the one shared camera owns the controls
│   ├── calls _alignSpatialFrustum({ container, camera, renderer, controls })
│   ├── calls _syncCameraState({ container, controls })                                             # publish this cell's shared-camera pose now and on every change for cross-cell sync
│   ├── calls attachThreeScenePickSeam({ container, camera, scenes: [scene] })                      # augment the container with the pickAt seam over the one shared scene
│   ├── calls renderLayeredSpatialScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by layeredDisplayResponse.slot_id
├── function createLayerObjects({ layeredDisplayResponse }: { layeredDisplayResponse: LayeredDisplayResponse }): THREE.Object3D[]
│   ├── # Builds the THREE object for every layer by dispatching each layer's display response to its registry-resolved spatial renderer.
│   ├── impls layerObjects = []
│   ├── for each layer in [base_display_response, ...aux_display_responses]
│   │   ├── calls getSpatialLayerRenderer({ displayKind: layer.display_kind })   → layerRenderer
│   │   └── impls layerObjects.push(layerRenderer({ displayResponse: layer }))
│   └── return layerObjects
├── function renderLayeredSpatialScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Drives the shared layered-scene render loop with the base-camera trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
├── function renderLayeredRasterDisplay({ layeredDisplayResponse }: { layeredDisplayResponse: LayeredDisplayResponse }): LeafVNode
│   ├── # Stacks the base + aux raster layers full-bleed in ONE shared coordinate frame as a slot_id-keyed LeafVNode whose render() materializes each layer and gives every aux overlay the base image's natural pixel extent on its load.
│   ├── impls container = div { className: "layered-display-container", style { position: relative, full-bleed } }
│   ├── for each layer in [base_display_response, ...aux_display_responses]
│   │   ├── calls getRasterLayerRenderer({ displayKind: layer.display_kind })   → layerRenderer
│   │   ├── impls cell = div { style { position: absolute, inset: 0, full-bleed } }; container.append(cell)
│   │   └── calls reconcileInto({ root: cell, virtualTree: layerRenderer({ displayResponse: layer }) })   # mount the layer's LeafVNode into its cell
│   ├── impls on the base raster layer's image load (or immediately if already complete), sets each aux overlay's SVG viewBox to _alignRasterFrustum({ baseImage }) (the base image's natural extent)
│   └── return LeafVNode keyed by layeredDisplayResponse.slot_id whose render() returns container
├── function _alignSpatialFrustum({ container, camera, renderer, controls }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Aligns the spatial cell's shared frustum to the cell: sets the renderer size and camera aspect from the container and re-applies on resize via a ResizeObserver.
│   ├── impls resize = () => { camera.aspect = container width/height; camera.updateProjectionMatrix(); renderer.setSize(container width, height, false); controls.handleResize() }
│   ├── impls resize()
│   ├── impls new ResizeObserver(resize).observe(container)
│   └── impls window.addEventListener("resize", resize)
├── function _syncCameraState({ container, controls }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Publishes this cell's shared-camera pose now and re-publishes on every controls change, so other cells can observe and sync to it.
│   ├── calls _publishCameraState({ container, controls })                                          # initial pose
│   ├── impls controls.addEventListener("change", () => _publishCameraState({ container, controls }))   # re-publish on change
│   └── return
├── function _publishCameraState({ container, controls }: { container: HTMLDivElement; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Publishes the controls' shared-camera state onto the container (dataset.cameraState plus a bubbling camera-pose-change event) so the consumer can persist this cell's camera pose — the layered container's copy of the per-display publish helper.
│   ├── impls cameraState = controls.getCameraState()
│   ├── if cameraState is null
│   │   └── return
│   ├── impls container.dataset.cameraState = JSON.stringify(cameraState)
│   └── impls container.dispatchEvent(new CustomEvent("camera-pose-change", { bubbles: true, detail: cameraState }))
└── function _alignRasterFrustum({ baseImage }: { baseImage: HTMLImageElement }): { width: number; height: number }
    ├── # Resolves the raster cell's shared frustum from the base image's intrinsic natural pixel extent { width: baseImage.naturalWidth, height: baseImage.naturalHeight } — the one coordinate grid every aux overlay maps onto.
    └── return
```

`./data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry.ts`

```text
layer_renderer_registry.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
├── export type SpatialLayerRenderer = ({ displayResponse }: { displayResponse: DisplayResponse }) => THREE.Object3D   # one spatial display response's part-B: build and return the THREE object the layered container adds to its shared scene
├── export type RasterLayerRenderer = ({ displayResponse }: { displayResponse: DisplayResponse }) => LeafVNode   # one raster display response's part-B: build and return the full-bleed node the layered container stacks; the container aligns the aux overlays to the shared raster frustum on the base image's load
├── const _spatialLayerRenderers = new Map<string, SpatialLayerRenderer>()   # display_kind -> spatial part-B; the module's single owner of the spatial registry, mutated only through the functions below
├── const _rasterLayerRenderers = new Map<string, RasterLayerRenderer>()     # display_kind -> raster part-B; the module's single owner of the raster registry, mutated only through the functions below
├── function registerSpatialLayerRenderer({ displayKind, layerRenderer }: { displayKind: string; layerRenderer: SpatialLayerRenderer }): void
│   ├── # Register a spatial display kind's part-B so the layered container can build that kind's THREE object by display_kind lookup.
│   ├── impls _spatialLayerRenderers.set(displayKind, layerRenderer)
│   └── return
├── function registerRasterLayerRenderer({ displayKind, layerRenderer }: { displayKind: string; layerRenderer: RasterLayerRenderer }): void
│   ├── # Register a raster display kind's part-B so the layered container can build that kind's node by display_kind lookup.
│   ├── impls _rasterLayerRenderers.set(displayKind, layerRenderer)
│   └── return
├── function getSpatialLayerRenderer({ displayKind }: { displayKind: string }): SpatialLayerRenderer
│   ├── # Resolve the spatial part-B registered for a display kind, throwing when none is registered.
│   ├── impls layerRenderer = _spatialLayerRenderers.get(displayKind)
│   ├── if layerRenderer === undefined
│   │   └── throw new Error
│   └── return layerRenderer
└── function getRasterLayerRenderer({ displayKind }: { displayKind: string }): RasterLayerRenderer
    ├── # Resolve the raster part-B registered for a display kind, throwing when none is registered.
    ├── impls layerRenderer = _rasterLayerRenderers.get(displayKind)
    ├── if layerRenderer === undefined
    │   └── throw new Error
    └── return layerRenderer
```

`./data/viewer/utils/displays/utils/ts/frontend/register_layer_renderers.ts`

```text
register_layer_renderers.ts
├── # Eager-imports every display modality's frontend apis module (Vite import.meta.glob) so each modality's module-load self-registration runs; new modalities are auto-discovered with no edit here.
└── impls import.meta.glob("data/viewer/utils/displays/**/ts/frontend/apis.ts", { eager: true })
```

`./data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers.ts`

```text
three_scene_helpers.ts
├── import * as THREE from "three";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import { createTrackballCameraControls, DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── export type PickableThreeContainer = HTMLDivElement & { pickAt: (clientX: number, clientY: number) => THREE.Object3D | null }   # any spatial display container augmented with an additive base-camera pick seam: a consumer raycasts a pointer position against the container's scenes via the camera without owning the camera/renderer/scenes; the base HTMLDivElement contract is unchanged
├── function createSpatialDisplayScene({ initialCameraState, pointerEventsSuppressed = false }: { initialCameraState: CameraState | null; pointerEventsSuppressed?: boolean }): { container: HTMLDivElement; scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer }
│   ├── # Shared part-A "create scene" step for every spatial display (standalone renderers and the layered container alike): composes the one container/scene/camera/renderer and nothing else; callers create and add their own object(s) separately.
│   ├── calls createThreeDisplayContainer({ pointerEventsSuppressed })   → container
│   ├── calls createThreePerspectiveCamera({ initialCameraState })              → camera
│   ├── calls createThreeWebGLRenderer({ container })                           → renderer
│   ├── calls createThreeScene()                                                → scene
│   └── return { container, scene, camera, renderer }
├── function createThreeDisplayContainer({ pointerEventsSuppressed }: { pointerEventsSuppressed: boolean }): HTMLDivElement
│   ├── # Shared display container for every TS atomic spatial display.
│   ├── impls absolutely-positioned full-bleed HTMLDivElement that owns the Three.js canvas
│   ├── if pointerEventsSuppressed
│   │   └── impls sets style.pointerEvents = "none" so the underlying base spatial display remains the interaction source
│   └── return
├── function createThreePerspectiveCamera({ initialCameraState }: { initialCameraState: CameraState | null }): THREE.PerspectiveCamera
│   ├── # Shared PerspectiveCamera factory for every TS atomic spatial display; the consumer-supplied initialCameraState is the single source of initial framing, with no lib-side fit-to-object.
│   ├── impls THREE.PerspectiveCamera(fov=DEFAULT_TRACKBALL_PERSPECTIVE_CAMERA_FOV, ...) at default aspect/near/far/position
│   ├── if initialCameraState is not null
│   │   └── impls overlays initialCameraState (every field — both intrinsics and extrinsics) onto the camera so first paint matches the source display
│   └── return
├── function createThreeWebGLRenderer({ container }: { container: HTMLDivElement }): THREE.WebGLRenderer
│   ├── # Shared WebGL renderer factory for every TS atomic spatial display.
│   ├── impls THREE.WebGLRenderer constructed with `alpha: true` and cleared transparent via `setClearColor(0x000000, 0)` so the canvas is transparent by default; consumers that want an opaque backdrop apply a CSS `background-color` to the marker
│   ├── impls canvas mounted inside the provided container
│   └── return
├── function createThreeScene(): THREE.Scene
│   ├── # Shared empty-scene factory used by every TS atomic spatial display; callers scene.add their own object(s).
│   ├── impls creates THREE.Scene; scene.background stays unset so the renderer's clear color is what gets visibly drawn
│   └── return
├── function attachThreeScenePickSeam({ container, camera, scenes }: { container: HTMLDivElement; camera: THREE.PerspectiveCamera; scenes: readonly THREE.Scene[] }): void
│   ├── # Installs a base-camera pickAt seam onto any spatial display container so a consumer can hit-test the given scenes via the camera without owning the camera, renderer, or scenes.
│   ├── impls raycaster = new THREE.Raycaster()
│   ├── impls pickAt = (clientX, clientY) => NDC from the container rect (null if empty), camera raycast over each scene, return first hit object else null
│   ├── impls (container as PickableThreeContainer).pickAt = pickAt   # additive seam; base HTMLDivElement contract unchanged
│   └── return
└── function startThreeSceneRenderLoop({ scene, camera, renderer, controls, onAfterRender }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> | null; onAfterRender?: () => void }): void
    ├── # Shared requestAnimationFrame loop driving one base scene each frame; self-stops and frees its WebGL context once the canvas leaves the DOM, with an optional onAfterRender hook for per-frame caller steps.
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

`./data/viewer/utils/displays/points/dash/apis.py`

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

`./data/viewer/utils/displays/points/dash/core_points_display.py`

```text
core_points_display.py
├── from typing import Optional
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.structures.three_d.point_cloud.point_cloud import PointCloud
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
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

`./data/viewer/utils/displays/points/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
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

`./data/viewer/utils/displays/points/ts/backend/apis.py`

```text
apis.py
├── from typing import Any, Dict, Optional, Tuple
├── import torch
├── from data.structures.three_d.point_cloud.io.load_point_cloud import load_point_cloud
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
│   └── return
└── def _build_segmentation_pc_meta_info(class_id_to_rgb: Dict[int, Tuple[int, int, int]]) -> Dict[str, Any]
    ├── # Builds factual class/color metadata from the class-to-RGB mapping.
    ├── impls stores `class_id_to_rgb`
    └── return
```

`./data/viewer/utils/displays/points/ts/backend/core_points_display.py`

```text
core_points_display.py
└── def create_points_display_response_core
    ├── # Creates a point display response from the loadable point resource path and caller-provided display metadata.
    ├── impls builds frontend resource url from point_cloud_path
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/displays/points/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
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

`./data/viewer/utils/displays/points/ts/frontend/apis.ts`

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
└── impls registerSpatialLayerRenderer({ displayKind: "color_pc", layerRenderer: createPointsObject })   # module-load self-registration of the spatial color-pc layer renderer
```

`./data/viewer/utils/displays/points/ts/frontend/core_points_display.ts`

```text
core_points_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { PointDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_POINT_SIZE_FLOOR = 0.005   # number — absolute floor for visibility at typical canonical-world camera framings; used by the bounding-sphere heuristic when pointSize is not supplied
├── const DEFAULT_POINT_SIZE_RATIO = 0.002   # number — fraction of geometry bounding-sphere radius used as the heuristic default size; lib-owned default, documented + overridable
├── const DEFAULT_POINT_COLOR = "#cccccc"    # hex color — uniform fallback used when geometry has no per-point colors AND the caller does not supply pointColor; lib-owned default, overridable
├── function renderPointsDisplay({ displayResponse, initialCameraState, pointSize, pointColor }: { displayResponse: PointDisplayResponse; initialCameraState?: CameraState | null; pointSize?: number; pointColor?: string }): LeafVNode
│   ├── # Renders a self-contained point-cloud display element initialized at initialCameraState.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createPointsObject({ displayResponse, pointSize, pointColor })   → object
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderPointsScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createPointsObject({ displayResponse, pointSize, pointColor }: { displayResponse: PointDisplayResponse; pointSize?: number; pointColor?: string }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the point cloud, populated with the THREE.Points once the async geometry load resolves.
│   ├── impls group = new THREE.Group(); loadPointGeometry({ displayResponse }).then(geometry => group.add(createThreePoints({ geometry, pointSize, pointColor })))
│   └── return group
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

`./data/viewer/utils/displays/pixels/dash/apis.py`

```text
apis.py
├── import torch
├── from dash import dcc
├── from data.viewer.utils.displays.pixels.dash.core_pixels_display import create_dash_pixels_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
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

`./data/viewer/utils/displays/pixels/dash/core_pixels_display.py`

```text
core_pixels_display.py
├── from typing import Any
├── from dash import dcc
└── def create_dash_pixels_display(image: Any, image_interpolation: str) -> dcc.Graph
    ├── # Renders a Dash pixel-image display element from the resolved interpolation choice; modality-agnostic.
    └── return
```

`./data/viewer/utils/displays/pixels/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
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

`./data/viewer/utils/displays/pixels/ts/backend/apis.py`

```text
apis.py
├── import torch
├── from data.viewer.utils.displays.pixels.ts.backend.core_pixels_display import create_pixels_display_response_core
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── def create_color_image_display_response
│   ├── # intentional thin wrapper: passes color image directly to core response
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_depth_image_display_response
│   ├── # maps depth image to color image before core response
│   ├── calls _map_depth_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_edge_image_display_response
│   ├── # maps edge image to color image before core response
│   ├── calls _map_edge_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_normal_image_display_response
│   ├── # maps normal image to color image before core response
│   ├── calls _map_normal_image_to_rgb
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_segmentation_image_display_response
│   ├── # Creates a segmentation image response from a class-labeled image resource.
│   ├── impls reads segmentation image tensor from segmentation_image_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_image))
│   ├── calls _map_segmentation_image_to_rgb(segmentation_image_path=segmentation_image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response_core
│   └── return
├── def create_instance_surrogate_image_display_response
│   ├── # maps instance-surrogate image to color image before core response
│   ├── impls builds integer instance-surrogate class-id image from offset-magnitude quantile bins
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(instance_surrogate_class_id_image))
│   ├── calls _map_instance_surrogate_image_to_rgb(image_path=image_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_instance_surrogate_image_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_pixels_display_response_core
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

`./data/viewer/utils/displays/pixels/ts/backend/core_pixels_display.py`

```text
core_pixels_display.py
└── def create_pixels_display_response_core
    ├── # Creates a pixel-image display response from the loadable image resource path and caller-provided display metadata.
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/displays/pixels/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
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

`./data/viewer/utils/displays/pixels/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { ColorImageDisplayResponse, DepthImageDisplayResponse, EdgeImageDisplayResponse, InstanceSurrogateImageDisplayResponse, NormalImageDisplayResponse, SegmentationImageDisplayResponse } from "./types/display_response";
├── import { renderPixelsDisplay } from "./core_pixels_display";
├── import { registerRasterLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── const DEFAULT_COLOR_IMAGE_INTERPOLATION = "linear"                # color images: linear interpolation smooths between RGB samples, appropriate for natural-image content
├── const DEFAULT_DEPTH_IMAGE_INTERPOLATION = "nearest"               # depth images: nearest preserves exact metric depth samples; linear would invent midpoint depths that don't exist in the data
├── const DEFAULT_EDGE_IMAGE_INTERPOLATION = "nearest"                # edge images: nearest preserves edge crispness; linear would smooth edges and defeat their purpose
├── const DEFAULT_NORMAL_IMAGE_INTERPOLATION = "nearest"              # normal images: nearest preserves unit-length normal vectors; linear interpolation between normals produces non-unit results
├── const DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION = "nearest"        # segmentation images: nearest preserves class-id integrity; linear would invent fractional class ids
├── const DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION = "nearest"  # instance-surrogate images: nearest preserves class-id integrity (same reason as segmentation)
├── function renderColorImageDisplay({ displayResponse, imageInterpolation = DEFAULT_COLOR_IMAGE_INTERPOLATION }: { displayResponse: ColorImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a color-image display, defaulting to linear interpolation for natural-image content.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderDepthImageDisplay({ displayResponse, imageInterpolation = DEFAULT_DEPTH_IMAGE_INTERPOLATION }: { displayResponse: DepthImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a depth-image display, defaulting to nearest interpolation to preserve exact metric depths.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderEdgeImageDisplay({ displayResponse, imageInterpolation = DEFAULT_EDGE_IMAGE_INTERPOLATION }: { displayResponse: EdgeImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders an edge-image display, defaulting to nearest interpolation to preserve edge crispness.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderNormalImageDisplay({ displayResponse, imageInterpolation = DEFAULT_NORMAL_IMAGE_INTERPOLATION }: { displayResponse: NormalImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders a normal-image display, defaulting to nearest interpolation to preserve unit-length normals.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderSegmentationImageDisplay({ displayResponse, imageInterpolation = DEFAULT_SEGMENTATION_IMAGE_INTERPOLATION }: { displayResponse: SegmentationImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders the backend-colorized segmentation display and legend derived from meta_info.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
├── function renderInstanceSurrogateImageDisplay({ displayResponse, imageInterpolation = DEFAULT_INSTANCE_SURROGATE_IMAGE_INTERPOLATION }: { displayResponse: InstanceSurrogateImageDisplayResponse; imageInterpolation?: string }): LeafVNode
│   ├── # Renders the backend-colorized image display and legend derived from meta_info.
│   ├── calls renderPixelsDisplay({ displayResponse, imageInterpolation })
│   └── return
└── impls registerRasterLayerRenderer({ displayKind: "color_image", layerRenderer: renderColorImageDisplay })   # module-load self-registration of the raster color-image layer renderer
```

`./data/viewer/utils/displays/pixels/ts/frontend/core_pixels_display.ts`

```text
core_pixels_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PixelDisplayResponse } from "./types/display_response";
└── function renderPixelsDisplay({ displayResponse, imageInterpolation }: { displayResponse: PixelDisplayResponse; imageInterpolation: string }): LeafVNode
    ├── # Renders a self-contained pixel-image display element from the resolved interpolation choice; modality-agnostic.
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/displays/placeholders/dash/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display
    └── # Builds the Dash missing-result placeholder display from a message.
```

`./data/viewer/utils/displays/placeholders/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class PlaceholderDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "placeholder"                 # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── message                                      # additional field
```

`./data/viewer/utils/displays/placeholders/ts/backend/placeholder_display.py`

```text
placeholder_display.py
└── def create_placeholder_display_response
    ├── # Creates a placeholder display response standing in for a missing result, carrying the message inline.
    ├── impls builds missing-result placeholder response from message
    └── return
```

`./data/viewer/utils/displays/placeholders/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface PlaceholderDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "placeholder"                 # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── message                                      # additional field
```

`./data/viewer/utils/displays/placeholders/ts/frontend/placeholder_display.ts`

```text
placeholder_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { PlaceholderDisplayResponse } from "./types/display_response";
└── function renderPlaceholderDisplay({ displayResponse }: { displayResponse: PlaceholderDisplayResponse }): LeafVNode
    ├── # Renders the missing-result placeholder UI from the response's message.
    ├── impls complete missing-result placeholder UI from PlaceholderDisplayResponse.message
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/displays/videos/dash/video_display.py`

```text
video_display.py
└── def create_video_display
    └── # Builds the Dash video display from a video path.
```

`./data/viewer/utils/displays/videos/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class VideoDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "video"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/videos/ts/backend/video_display.py`

```text
video_display.py
└── def create_video_display_response
    ├── # Creates a video display response from a loadable video resource.
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty video metadata
    └── return
```

`./data/viewer/utils/displays/videos/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface VideoDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "video"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/videos/ts/frontend/video_display.ts`

```text
video_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { VideoDisplayResponse } from "./types/display_response";
└── function renderVideoDisplay({ displayResponse }: { displayResponse: VideoDisplayResponse }): LeafVNode
    ├── # Renders the complete video-display UI from the video resource URL.
    ├── impls complete video-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/displays/texts/dash/text_display.py`

```text
text_display.py
└── def create_text_display
    └── # Builds the Dash text display from a text string.
```

`./data/viewer/utils/displays/texts/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TextDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "text"                        # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── text                                         # additional field
```

`./data/viewer/utils/displays/texts/ts/backend/text_display.py`

```text
text_display.py
└── def create_text_display_response
    ├── # Creates a text display response carrying the text payload inline.
    ├── impls stores text in TextDisplayResponse.text
    ├── impls sets meta_info to empty text metadata
    └── return
```

`./data/viewer/utils/displays/texts/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface TextDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "text"                        # common field
    ├── url                                          # common field
    ├── meta_info                                    # common field
    └── text                                         # additional field
```

`./data/viewer/utils/displays/texts/ts/frontend/text_display.ts`

```text
text_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { TextDisplayResponse } from "./types/display_response";
└── function renderTextDisplay({ displayResponse }: { displayResponse: TextDisplayResponse }): LeafVNode
    ├── # Renders the complete text-display UI from the response's text field.
    ├── impls complete text-display UI from TextDisplayResponse.text
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/displays/tables/dash/table_display.py`

```text
table_display.py
└── def create_table_display
    └── # Builds the Dash table display from tabular data.
```

`./data/viewer/utils/displays/tables/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class TableDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "table"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/tables/ts/backend/table_display.py`

```text
table_display.py
└── def create_table_display_response
    ├── # Creates a table display response from a loadable table resource.
    ├── impls builds frontend resource url
    ├── impls sets meta_info to empty table metadata
    └── return
```

`./data/viewer/utils/displays/tables/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface TableDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "table"                       # common field
    ├── url                                          # common field
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/tables/ts/frontend/table_display.ts`

```text
table_display.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { TableDisplayResponse } from "./types/display_response";
└── function renderTableDisplay({ displayResponse }: { displayResponse: TableDisplayResponse }): LeafVNode
    ├── # Renders the complete table-display UI from the table resource URL.
    ├── impls complete table-display UI from DisplayResponse url
    └── return LeafVNode keyed by displayResponse.url
```

`./data/viewer/utils/displays/scene_graphs/dash/scene_graph_display.py`

```text
scene_graph_display.py
└── def create_scene_graph_display
    └── # Builds the Dash scene-graph display from a method-agnostic graph payload.
```

`./data/viewer/utils/displays/scene_graphs/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class SceneGraphDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "scene_graph"                 # common field
    ├── url                                          # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/scene_graphs/ts/backend/scene_graph_display.py`

```text
scene_graph_display.py
├── import torch
├── from data.viewer.utils.displays.scene_graphs.ts.backend.schemas.display_response import SceneGraphDisplayResponse
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

`./data/viewer/utils/displays/scene_graphs/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface SceneGraphDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "scene_graph"                 # common field
    ├── url                                          # common field; serves the scene-graph payload (no leaked encoding)
    └── meta_info                                    # common field
```

`./data/viewer/utils/displays/scene_graphs/ts/frontend/scene_graph_display.ts`

```text
scene_graph_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { SceneGraphDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_NODE_SIZE = 0.02            # number — heuristic default size for node markers when the caller does not supply nodeSize; lib-owned default, overridable
├── const DEFAULT_EDGE_COLOR = "#888888"      # hex color — neutral gray fallback for edge lines when the payload does not carry an edge color AND the caller does not supply edgeColor; lib-owned default, overridable
├── const DEFAULT_EDGE_WIDTH = 1.0            # number — line width fallback for edges when the caller does not supply edgeWidth; lib-owned default, overridable
├── const DEFAULT_LABEL_FONT_SIZE = 12        # px — font size fallback for overlay labels when the caller does not supply labelFontSize; lib-owned default, overridable
├── const DEFAULT_LABEL_COLOR = "#000000"     # hex color — text color fallback for overlay labels when the caller does not supply labelColor; lib-owned default, overridable
├── function renderSceneGraphDisplay({ displayResponse, initialCameraState, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor }: { displayResponse: SceneGraphDisplayResponse; initialCameraState?: CameraState | null; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string }): LeafVNode
│   ├── # Renders a self-contained scene-graph display: baked node/edge geometry plus HTML label overlay projected per frame.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createSceneGraphObject({ container, displayResponse, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor })   → { object, labels, labelOverlay }
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderSceneGraphScene({ scene, camera, renderer, controls, labels, labelOverlay, labelFontSize, labelColor })
│   └── return LeafVNode keyed by displayResponse.url
├── function createSceneGraphObject({ container, displayResponse, nodeSize, edgeColor, edgeWidth, labelFontSize, labelColor }: { container: HTMLDivElement; displayResponse: SceneGraphDisplayResponse; nodeSize?: number; edgeColor?: string; edgeWidth?: number; labelFontSize?: number; labelColor?: string }): { object: THREE.Object3D; labels: object[]; labelOverlay: HTMLDivElement }
│   ├── # Part-B: builds the HTML label overlay and returns a THREE.Group + mutable labels array, both populated from the THREE.Points + label data once the async payload load resolves.
│   ├── calls createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor })   → labelOverlay
│   ├── impls group = new THREE.Group(); labels: object[] = []
│   ├── impls loadSceneGraphPayload({ displayResponse }).then(payload => { const built = createThreeSceneGraphPoints({ payload, nodeSize, edgeColor, edgeWidth }); group.add(built.points); labels.push(...built.labels); })
│   └── return { object: group, labels, labelOverlay }
├── function createThreeSceneGraphLabelOverlay({ container, labelFontSize, labelColor }: { container: HTMLDivElement; labelFontSize?: number; labelColor?: string }): HTMLDivElement
│   ├── # Builds the absolutely-positioned HTML overlay container layered above the canvas; labelFontSize / labelColor apply as the overlay's default font-size and color (per-label inline styles still take precedence).
│   ├── impls effectiveLabelFontSize = labelFontSize ?? DEFAULT_LABEL_FONT_SIZE
│   ├── impls effectiveLabelColor = labelColor ?? DEFAULT_LABEL_COLOR
│   ├── impls absolutely-positioned HTML overlay container layered above the canvas with default font-size = effectiveLabelFontSize px and color = effectiveLabelColor, returned and mounted inside the display container
│   └── return
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

`./data/viewer/utils/displays/mesh/dash/apis.py`

```text
apis.py
├── from typing import Optional
├── import torch
├── from dash import dcc
├── from data.viewer.utils.displays.mesh.dash.core_mesh_display import create_dash_mesh_display
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb
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

`./data/viewer/utils/displays/mesh/dash/core_mesh_display.py`

```text
core_mesh_display.py
├── from typing import Any, Optional
├── import plotly.graph_objects as go
├── from dash import dcc
├── from data.viewer.utils.controls.camera.camera_controls.dash.trackball_camera_controls import create_dash_trackball_camera_controls
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

`./data/viewer/utils/displays/mesh/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
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

`./data/viewer/utils/displays/mesh/ts/backend/apis.py`

```text
apis.py
├── from pathlib import Path
├── from typing import Any, Dict, Tuple
├── import torch
├── from data.viewer.utils.displays.mesh.ts.backend.core_mesh_display import create_mesh_display_response_core
├── from data.viewer.utils.displays.mesh.ts.backend.schemas.display_response import ColorMeshDisplayResponse, HeatmapMeshDisplayResponse, SegmentationMeshDisplayResponse, SparseHeatmapMeshDisplayResponse
├── from data.viewer.utils.displays.utils.class_colors import map_class_ids_to_rgb
├── from data.viewer.utils.displays.utils.heatmap_colors import map_scalars_to_rgb
├── def create_color_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> ColorMeshDisplayResponse
│   ├── # Intentional thin wrapper: writes the color mesh resource at output_path and returns ColorMeshDisplayResponse with the caller-provided url.
│   ├── calls create_mesh_display_response_core
│   └── return
├── def create_segmentation_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> SegmentationMeshDisplayResponse
│   ├── # Creates a segmentation mesh response from a class-labeled mesh resource read from input_path; processed mesh is written to output_path.
│   ├── impls reads segmentation mesh class ids from input_path
│   ├── calls map_class_ids_to_rgb(class_ids=torch.unique(segmentation_mesh_class_ids))
│   ├── calls _map_segmentation_mesh_to_rgb(input_path=input_path, output_path=output_path, class_id_to_rgb=class_id_to_rgb)
│   ├── calls _build_segmentation_mesh_meta_info(class_id_to_rgb=class_id_to_rgb)
│   ├── calls create_mesh_display_response_core
│   └── return
├── def create_heatmap_mesh_display_response(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> HeatmapMeshDisplayResponse
│   ├── # Creates a heatmap mesh response from a non-negative-scalar-labeled mesh resource read from input_path; processed mesh is written to output_path.
│   ├── impls reads heatmap mesh scalar values from input_path (per-vertex 1-D or per-texel 2-D, non-negative)
│   ├── calls map_scalars_to_rgb(scalars=heatmap_mesh_scalars)
│   ├── calls _map_heatmap_mesh_to_rgb(input_path=input_path, output_path=output_path, scalar_rgb=scalar_rgb)
│   ├── calls _build_heatmap_mesh_meta_info(scalars=heatmap_mesh_scalars)
│   ├── calls create_mesh_display_response_core
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

`./data/viewer/utils/displays/mesh/ts/backend/core_mesh_display.py`

```text
core_mesh_display.py
├── from pathlib import Path
├── from typing import Any, Dict
├── from data.viewer.utils.displays.mesh.ts.backend.schemas.display_response import MeshDisplayResponse
├── def create_mesh_display_response_core(input_path: Path, output_path: Path, url: str, slot_id: str, title: str, meta_info: Dict[str, Any]) -> MeshDisplayResponse
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

`./data/viewer/utils/displays/mesh/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
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

`./data/viewer/utils/displays/mesh/ts/frontend/core_mesh_display.ts`

```text
core_mesh_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { MeshDisplayResponse } from "./types/display_response";
├── import { createTrackballCameraControls } from "data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_MESH_COLOR = "#cccccc"          # hex color — uniform fallback used when geometry has no texture AND has no vertex colors AND the caller does not supply meshColor; lib-owned default, overridable
├── const DEFAULT_MESH_OPACITY = 1.0              # number — opaque default applied when the caller does not supply meshOpacity; material's `transparent` flag flips true automatically when opacity is less than 1; lib-owned default, overridable
├── const DEFAULT_MESH_SIDE = THREE.DoubleSide    # THREE.Side — fallback side mode for visibility under arbitrary camera framings when the caller does not supply meshSide; lib-owned default, overridable
├── interface MeshPayload
│   ├── # The render-side mirror of the Mesh data structure: geometry (verts, faces) plus an optional MeshTexture.
│   ├── verts: Float32Array                                                # [V, 3] flattened — mirrors Mesh.verts
│   ├── faces: Uint32Array                                                 # [F, 3] flattened — mirrors Mesh.faces
│   └── texture: MeshTextureVertexColor | MeshTextureUVTextureMap | null   # mirrors Mesh.texture (Optional[MeshTexture])
├── interface MeshTextureVertexColor
│   ├── # Render mirror of the data structure's MeshTextureVertexColor: per-vertex colors aligned 1:1 with verts.
│   ├── kind: "vertex_color"
│   └── vertexColor: Float32Array                                          # [V, C] per-vertex colors, C in {3, 4}
├── interface MeshTextureUVTextureMap
│   ├── # Render mirror of the data structure's MeshTextureUVTextureMap: a per-face-indexed UV texture map.
│   ├── kind: "uv_texture_map"
│   ├── uvTextureMap: THREE.Texture                                        # the texture image
│   ├── vertsUvs: Float32Array                                             # [VT, 2] UV coordinates
│   └── facesUvs: Uint32Array                                              # [F, 3] flattened — per-face UV-vertex indices
├── function renderMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide }: { displayResponse: MeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
│   ├── # Renders a self-contained mesh display element initialized at initialCameraState.
│   ├── calls createSpatialDisplayScene({ initialCameraState })
│   ├── calls createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide })   → object
│   ├── impls scene.add(object)
│   ├── calls createTrackballCameraControls({ container, camera, renderer, initialCameraState })
│   ├── calls renderMeshScene({ scene, camera, renderer, controls })
│   └── return LeafVNode keyed by displayResponse.url
├── function createMeshObject({ displayResponse, meshColor, meshOpacity, meshSide }: { displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): THREE.Object3D
│   ├── # Part-B: returns a THREE.Group for the mesh, populated with the THREE.Mesh once the async payload load resolves.
│   ├── impls group = new THREE.Group(); loadMeshPayload({ displayResponse }).then(payload => group.add(createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide })))
│   └── return group
├── async function loadMeshPayload({ displayResponse }: { displayResponse: MeshDisplayResponse }): Promise<MeshPayload>
│   ├── # Async-loads the mesh payload from displayResponse.url; resolves a sparse-heatmap delta against its referenced geometry, otherwise reads the dense resource as-is.
│   ├── if the url resource is a sparse heatmap resource
│   │   └── impls resolves the (indices, values) delta into a MeshPayload whose texture is a MeshTextureVertexColor — `indices` vertices at alpha 1 with their scalar→rgb color, every other vertex at alpha 0 (a base-revealing overlay)
│   ├── else
│   │   └── impls reads the dense mesh resource from displayResponse.url into a MeshPayload — verts + faces, plus its parsed MeshTexture (a MeshTextureUVTextureMap when the OBJ carries a material/UVs, else a MeshTextureVertexColor, else null)
│   └── return payload
├── function createThreeMesh({ payload, displayResponse, meshColor, meshOpacity, meshSide }: { payload: MeshPayload; displayResponse: MeshDisplayResponse; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): THREE.Mesh
│   ├── # Sync-builds THREE.BufferGeometry + THREE.MeshBasicMaterial + THREE.Mesh from a pre-loaded payload.
│   ├── impls geometry = non-indexed THREE.BufferGeometry whose position attribute gathers payload.verts by payload.faces (each of the F faces contributes its 3 corner positions), so render corner c maps to logical vertex payload.faces[c]
│   ├── impls set geometry.userData.cornerVertexIndices = payload.faces   # payload.faces flattened IS this non-indexed geometry's corner→vertex map, so a downstream consumer can gather a per-logical-vertex field into the corner render domain
│   ├── impls effectiveOpacity = meshOpacity ?? DEFAULT_MESH_OPACITY
│   ├── impls effectiveSide = meshSide ?? DEFAULT_MESH_SIDE
│   ├── if meshColor !== undefined
│   │   └── impls useTexture = false; useVertexColors = false; effectiveColor = meshColor
│   ├── else if payload.texture is a MeshTextureUVTextureMap
│   │   └── impls add a uv attribute to geometry gathering payload.texture.vertsUvs by payload.texture.facesUvs; useTexture = true; useVertexColors = false; effectiveColor = undefined
│   ├── else if payload.texture is a MeshTextureVertexColor
│   │   └── impls add a color attribute to geometry gathering payload.texture.vertexColor by payload.faces; useTexture = false; useVertexColors = true; effectiveColor = undefined
│   ├── else
│   │   └── impls useTexture = false; useVertexColors = false; effectiveColor = DEFAULT_MESH_COLOR
│   ├── impls material = MeshBasicMaterial { vertexColors: useVertexColors, side: effectiveSide, opacity: effectiveOpacity, transparent when opacity<1 or RGBA vertex colors, map: payload.texture.uvTextureMap when useTexture, color: effectiveColor when set }   # RGBA alpha-0 corners render transparent
│   └── return new THREE.Mesh(geometry, material)                                                # no post-construction mutation of mesh
└── function renderMeshScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls>; }): void
    ├── # Drives the mesh render loop with the supplied trackball controls.
    ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
    └── return
```

`./data/viewer/utils/displays/mesh/ts/frontend/apis.ts`

```text
apis.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { ColorMeshDisplayResponse, SegmentationMeshDisplayResponse, HeatmapMeshDisplayResponse, SparseHeatmapMeshDisplayResponse } from "./types/display_response";
├── import { renderMeshDisplay } from "./core_mesh_display";
├── function renderColorMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide }: { displayResponse: ColorMeshDisplayResponse; initialCameraState?: CameraState | null; meshColor?: string; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
│   ├── # Renders a color mesh display with opt-in meshColor, meshOpacity, and meshSide overrides.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshColor, meshOpacity, meshSide })
│   └── return
├── function renderSegmentationMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: SegmentationMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
│   ├── # renders backend-colorized mesh display and legend derived from meta_info; per-element colors are already baked in by the backend's class-id → rgb mapping, so no meshColor override is exposed here.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
├── function renderHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: HeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
│   ├── # renders backend-colorized mesh display and continuous-palette legend derived from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping, so no meshColor override is exposed here.
│   ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
│   └── return
└── function renderSparseHeatmapMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide }: { displayResponse: SparseHeatmapMeshDisplayResponse; initialCameraState?: CameraState | null; meshOpacity?: number; meshSide?: THREE.Side }): LeafVNode
    ├── # renders the sparse heatmap mesh display and continuous-palette legend from meta_info (scalar min/max); per-element colors are already baked in by the backend's scalar → rgb mapping, so no meshColor override is exposed here.
    ├── calls renderMeshDisplay({ displayResponse, initialCameraState, meshOpacity, meshSide })
    └── return
```

`./data/viewer/utils/displays/gaussians/dash/apis.py`

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
    └── # Recolors the segmentation Gaussian's per-Gaussian class ids to RGB via the class-to-RGB mapping.
```

`./data/viewer/utils/displays/gaussians/dash/core_gaussians_display.py`

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
│   ├── impls Dash Gaussian-splat display scene from Gaussian data and display metadata
│   └── return
└── def create_dash_gaussians_component
    ├── # Assembles the Dash component that hosts the Gaussian-splat scene and its trackball camera controls.
    └── return
```

`./data/viewer/utils/displays/gaussians/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
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

`./data/viewer/utils/displays/gaussians/ts/backend/apis.py`

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

`./data/viewer/utils/displays/gaussians/ts/backend/core_gaussians_display.py`

```text
core_gaussians_display.py
└── def create_gaussians_display_response_core
    ├── # Creates a Gaussian display response from the loadable Gaussian resource path and caller-provided display metadata.
    ├── impls builds frontend resource url
    ├── impls copies caller-provided meta_info into response metadata
    └── return
```

`./data/viewer/utils/displays/gaussians/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
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

`./data/viewer/utils/displays/gaussians/ts/frontend/apis.ts`

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

`./data/viewer/utils/displays/gaussians/ts/frontend/core_gaussians_display.ts`

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

`./data/viewer/utils/displays/cameras/dash/camera_display.py`

```text
camera_display.py
└── def create_camera_display
    └── # Builds the Dash camera-trajectory display from a loaded camera artifact.
```

`./data/viewer/utils/displays/cameras/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class CameraDisplayResponse(DisplayResponse)
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "camera"                      # common field
    ├── url                                          # common field; camera-vis JSON payload URL
    └── meta_info                                    # common field; empty object for camera display
```

`./data/viewer/utils/displays/cameras/ts/backend/apis.py`

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
│   ├── calls cameras_vis                          # forwards frustum_size/frustum_color/point_size/point_color untouched; cameras_vis resolves each None to its module-global style default
│   ├── for each camera-vis entry
│   │   └── calls _serialize_camera_vis_entry
│   └── return
├── def _serialize_camera_vis_entry(camera_vis_entry) -> Dict[str, Any]
│   ├── # Converts one camera-vis entry into the JSON shape consumed by the camera renderer.
│   ├── impls serializes center, center_color, and center_size
│   ├── for each line in axes
│   │   └── calls _serialize_camera_vis_line
│   ├── for each line in frustum_lines
│   │   └── calls _serialize_camera_vis_line
│   └── return
└── def _serialize_camera_vis_line(camera_vis_line) -> Dict[str, Any]
    ├── # Converts one camera-vis line segment into plain start, end, and color lists.
    ├── impls serializes start, end, and color
    └── return
```

`./data/viewer/utils/displays/cameras/ts/backend/core_camera_display.py`

```text
core_camera_display.py
└── def create_camera_display_response_core(slot_id: str, title: str, camera_vis_payload: List[Dict[str, Any]], meta_info: Optional[Dict[str, Any]] = None) -> CameraDisplayResponse
    ├── # Creates a camera display response from the already-mapped camera-vis payload, exposing it through a frontend-loadable URL.
    ├── impls builds the camera-vis data URL from camera_vis_payload (json then base64)
    ├── impls copies caller-provided meta_info into response metadata (empty object for camera display)
    └── return
```

`./data/viewer/utils/displays/cameras/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface CameraDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "camera"                      # common field
    ├── url                                          # common field; camera-vis JSON payload URL
    └── meta_info                                    # common field; empty object for camera display
```

`./data/viewer/utils/displays/cameras/ts/frontend/camera_display.ts`

```text
camera_display.ts
├── import * as THREE from "three";
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
├── import type { CameraDisplayResponse } from "./types/display_response";
├── import { createSpatialDisplayScene, startThreeSceneRenderLoop } from "data/viewer/utils/displays/utils/ts/frontend/three_scene_helpers";
├── const DEFAULT_FRUSTUM_OPACITY = 0.5            # number — overlay render opacity applied when the caller does not supply frustumOpacity; a dynamic render property (the per-frame hover dimming multiplies it), not a baked glyph style — glyph size + color are baked by camera_vis
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
│   ├── impls group = new THREE.Group(); loadCamerasPayload({ displayResponse }).then(payload => group.add(createThreeCameras({ payload, frustumOpacity })))
│   └── return group
├── async function loadCamerasPayload({ displayResponse }: { displayResponse: CameraDisplayResponse }): Promise<CamerasPayload>
│   └── # Async-loads the camera-vis JSON payload from displayResponse.url and validates each entry has center / center_color / center_size / axes / frustum_lines and that every axes/frustum line carries start / end / color; returns the validated payload.
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

`./data/viewer/utils/controls/camera/camera_state/dash/camera_state.py`

```text
camera_state.py
└── class CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/controls/camera/camera_state/ts/backend/schemas/camera_state.py`

```text
camera_state.py
└── class CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/controls/camera/camera_state/ts/backend/camera_state.py`

```text
camera_state.py
├── from data.structures.three_d.camera import Camera
├── from data.viewer.utils.controls.camera.camera_state.ts.backend.schemas.camera_state import CameraState
└── def create_camera_state_from_camera
    ├── # preserves Camera intrinsics, extrinsics, convention, name, and id
    ├── impls converts Camera to TS backend CameraState schema
    └── return
```

`./data/viewer/utils/controls/camera/camera_state/ts/frontend/types.ts`

```text
types.ts
└── interface CameraState
    ├── intrinsics
    ├── extrinsics
    ├── convention
    ├── name
    └── id
```

`./data/viewer/utils/controls/camera/camera_controls/dash/trackball_camera_controls.py`

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

`./data/viewer/utils/controls/camera/camera_controls/ts/frontend/trackball_camera_controls.ts`

```text
trackball_camera_controls.ts
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
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

`./data/viewer/utils/controls/camera/camera_sync/dash/camera_sync.py`

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

`./data/viewer/utils/controls/camera/camera_sync/ts/frontend/types.ts`

```text
types.ts
└── interface CameraSyncState
    ├── source_id    # the source this entry belongs to; one CameraSyncState exists per source
    ├── target_ids   # targets registered under this source
    └── camera_state # this source's current camera state
```

`./data/viewer/utils/controls/camera/camera_sync/ts/frontend/camera_sync.ts`

```text
camera_sync.ts
├── import type { CameraState } from "data/viewer/utils/controls/camera/camera_state/ts/frontend/types";
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

`./data/viewer/utils/controls/selectors/ts/backend/schemas/selector_response.py`

```text
selector_response.py
├── from typing import List
├── from pydantic import BaseModel
├── def build_selector_response
│   ├── # Build a SelectorResponse from an app's nested (value, label, children) option tuple — the app owns the tree shape, the lib owns the schema.
│   ├── calls _to_selection_node       # convert the imaginary-root tuple
│   └── return            # SelectorResponse(root=converted imaginary root)
├── def _to_selection_node
│   ├── # Recursion helper: convert one (value, label, children) tuple into a SelectionNode, recursing into each child tuple.
│   ├── for each child tuple
│   │   └── calls _to_selection_node
│   ├── calls SelectionNode
│   └── return            # a SelectionNode holding its converted children
├── class SelectorResponse(BaseModel)
│   ├── # One selector axis: the imaginary root of its option tree, descended recursively along the selection path to render the cascade.
│   └── root: SelectionNode
└── class SelectionNode(BaseModel)
    ├── # One option node of a selector axis: its value, display label, and child nodes (empty at a leaf), so parentage is the nesting itself.
    ├── value: str
    ├── label: str
    └── children: List[SelectionNode]
```

`./data/viewer/utils/controls/selectors/ts/frontend/types/selector_response.ts`

```text
selector_response.ts
├── interface SelectorResponse
│   ├── # One selector axis: the imaginary root of its option tree — mirrors the backend SelectorResponse schema.
│   └── root: SelectionNode
└── interface SelectionNode
    ├── # One option node of a selector axis: value, label, and child nodes (empty at a leaf) — mirrors the backend SelectionNode schema.
    ├── value: string
    ├── label: string
    └── children: SelectionNode[]
```

`./data/viewer/utils/controls/selectors/ts/frontend/selection_path.ts`

```text
selection_path.ts
├── import type { SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
└── function completeRootLeafPath({ root, path, level, value }: { root: SelectionNode; path: string[]; level: number; value: string }): string[]
    ├── # Complete a selector level change into a full root-leaf path: the prefix up to the chosen level, the chosen value, then each deeper level's first child descended to a leaf — so a non-leaf choice resets every finer level to its first option.
    ├── for each deeper level until the descended node has no children
    │   └── impls append the descended node's first child's value, then descend into it
    └── return            # the completed root-leaf path
```

`./data/viewer/utils/controls/selectors/ts/frontend/selector_cascade.ts`

```text
selector_cascade.ts
├── import type { ElementVNode, LeafVNode } from "web/reconcile/reconcile";
├── import type { SelectorResponse, SelectionNode } from "data/viewer/utils/controls/selectors/ts/frontend/types/selector_response";
├── import { completeRootLeafPath } from "data/viewer/utils/controls/selectors/ts/frontend/selection_path";
├── function renderSelectorCascade({ axisKey, response, path, onPathChange }: { axisKey: string; response: SelectorResponse; path: string[]; onPathChange: (next: string[]) => void }): ElementVNode
│   ├── # Render one selector axis as a cascade of native <select> dropdowns: descend the response's imaginary root along the current path, one dropdown per level to a leaf; the app supplies only the option tree, the current path, and an onPathChange handler.
│   ├── calls _renderSelectorLevel        # collect the per-level <select> leaves from the imaginary root down
│   └── return            # a container ElementVNode wrapping the collected <select> leaves
└── function _renderSelectorLevel({ node, level, axisKey, path, onPathChange }: { node: SelectionNode; level: number; axisKey: string; path: string[]; onPathChange: (next: string[]) => void }): LeafVNode[]
    ├── # Recursion helper: collect the <select> leaves from this level down; the base case (a node with no children) contributes none.
    ├── if node has no children
    │   └── return            # [] — base case: a leaf level adds no dropdown
    ├── impls the <select> is a reconciler leaf keyed `${axisKey}-select-${level}-${path[level-1] ?? "root"}` (its option-set identity) so a coarser-level change re-mounts it with this parent's children
    ├── impls build a native <select> over node's children
    ├── function _onLevelChange [local]
    │   ├── # The <select> change handler: report the completed root-leaf path to onPathChange.
    │   ├── calls completeRootLeafPath
    │   └── calls onPathChange
    ├── calls _onLevelChange           # bound as the <select>'s change listener
    ├── calls _renderSelectorLevel     # recurse into the path-selected child to collect the deeper levels' leaves
    └── return            # [this level's <select> leaf, ...the deeper levels' leaves]
```

`./data/viewer/utils/controls/selectors/dash/selector_cascade.py`

```text
selector_cascade.py
├── from typing import List
├── from data.viewer.utils.controls.selectors.ts.backend.schemas.selector_response import SelectorResponse, SelectionNode
├── def render_selector_cascade(response: SelectorResponse, path: List[str])
│   ├── # Render one selector axis as a Dash cascade of dropdowns from a SelectorResponse and the current path: one dropdown per level, descending the imaginary root along the path to a leaf, re-rendered per parent change.
│   ├── calls _render_selector_level
│   └── return            # the dropdown-stack Dash component
├── def _render_selector_level(node: SelectionNode, level: int, path: List[str])
│   ├── # Recursion helper: a Dash dropdown over this node's children, then recurse into the child the path selects, stopping at a leaf.
│   ├── if this node has children
│   │   └── calls _render_selector_level
│   └── return
└── def complete_root_leaf_path(node: SelectionNode, path: List[str])
    ├── # Complete a Dash level change into a full root-leaf path: the chosen value, then each deeper level's first child descended to a leaf.
    ├── for each deeper level until the descended node has no children
    │   └── impls append the descended node's first child's value, then descend into it
    └── return            # the completed root-leaf path
```

`./data/viewer/utils/displays/aabbs/threed/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class Aabb3dDisplayResponse(DisplayResponse)
    ├── # Spatial overlay response: inline axis-aligned 3D boxes (each a 6-float box) with optional per-box scores, composed as an aux layer over a point cloud.
    ├── display_kind = "aabb_3d"                     # common field
    ├── aabbs: List[List[float]]
    └── scores: Optional[List[float]]
```

`./data/viewer/utils/displays/aabbs/threed/ts/backend/apis.py`

```text
apis.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.aabbs.threed.ts.backend.schemas.display_response import Aabb3dDisplayResponse
└── def create_aabb_3d_display_response(slot_id: str, title: str, aabbs: List[List[float]], scores: Optional[List[float]] = None) -> Aabb3dDisplayResponse
    ├── # Creates a 3D axis-aligned-box overlay response from inline boxes and optional per-box scores.
    ├── calls Aabb3dDisplayResponse
    └── return
```

`./data/viewer/utils/displays/aabbs/threed/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface Aabb3dDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "aabb_3d"                     # common field
    ├── aabbs
    └── scores
```

`./data/viewer/utils/displays/aabbs/threed/ts/frontend/apis.ts`

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
│   ├── impls group = new THREE.Group(); build the box-edges meshes and score labels from displayResponse.aabbs and displayResponse.scores; add each to group
│   └── return group
├── function renderAabb3dScene({ scene, camera, renderer, controls }: { scene: THREE.Scene; camera: THREE.PerspectiveCamera; renderer: THREE.WebGLRenderer; controls: ReturnType<typeof createTrackballCameraControls> }): void
│   ├── # Drives the 3D-box display render loop with the supplied trackball controls.
│   ├── calls startThreeSceneRenderLoop({ scene, camera, renderer, controls })
│   └── return
└── impls registerSpatialLayerRenderer({ displayKind: "aabb_3d", layerRenderer: createAabb3dObject })   # module-load self-registration of the spatial aabb-3d layer renderer
```

`./data/viewer/utils/displays/aabbs/twod/ts/backend/schemas/display_response.py`

```text
display_response.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.utils.ts.backend.schemas.display_response import DisplayResponse
└── class Aabb2dDisplayResponse(DisplayResponse)
    ├── # Raster overlay response: inline axis-aligned 2D boxes (each a 4-float box) with optional per-box scores, composed as an aux layer over an image.
    ├── display_kind = "aabb_2d"                     # common field
    ├── aabbs: List[List[float]]
    └── scores: Optional[List[float]]
```

`./data/viewer/utils/displays/aabbs/twod/ts/backend/apis.py`

```text
apis.py
├── from typing import List, Optional
├── from data.viewer.utils.displays.aabbs.twod.ts.backend.schemas.display_response import Aabb2dDisplayResponse
└── def create_aabb_2d_display_response(slot_id: str, title: str, aabbs: List[List[float]], scores: Optional[List[float]] = None) -> Aabb2dDisplayResponse
    ├── # Creates a 2D axis-aligned-box overlay response from inline boxes and optional per-box scores.
    ├── calls Aabb2dDisplayResponse
    └── return
```

`./data/viewer/utils/displays/aabbs/twod/ts/frontend/types/display_response.ts`

```text
display_response.ts
├── import type { DisplayResponse } from "data/viewer/utils/displays/utils/ts/frontend/types/display_response";
└── interface Aabb2dDisplayResponse extends DisplayResponse
    ├── slot_id                                      # common field
    ├── title                                        # common field
    ├── display_kind = "aabb_2d"                     # common field
    ├── aabbs
    └── scores
```

`./data/viewer/utils/displays/aabbs/twod/ts/frontend/apis.ts`

```text
apis.ts
├── import type { LeafVNode } from "web/reconcile/reconcile";
├── import type { Aabb2dDisplayResponse } from "./types/display_response";
├── import { registerRasterLayerRenderer } from "data/viewer/utils/displays/utils/ts/frontend/layer_renderer_registry";
├── function renderAabb2dDisplay({ displayResponse }: { displayResponse: Aabb2dDisplayResponse }): LeafVNode
│   ├── # Renders the inline 2D axis-aligned boxes and their optional per-box score labels as a full-bleed raster SVG overlay; the layered container sets its viewBox to the shared frustum on the base image's load.
│   ├── impls build the full-bleed SVG box overlay (preserveAspectRatio="none") and score labels from displayResponse.aabbs and displayResponse.scores
│   └── return
└── impls registerRasterLayerRenderer({ displayKind: "aabb_2d", layerRenderer: renderAabb2dDisplay })   # module-load self-registration of the raster aabb-2d layer renderer
```

